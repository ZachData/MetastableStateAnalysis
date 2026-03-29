"""
analysis.py — Post-training feature analysis.

Every analysis function has the same signature:

    def metric_fn(
        crosscoder: Crosscoder,
        prompt_store: PromptActivationStore,
        artifacts: dict,
        config: dict,
    ) -> dict

Adding a new experiment = writing a new function and registering it.
No touching the training loop, no touching the data pipeline.

Registry
--------
feature_lifetimes         — decoder norm profiles, bimodality test
v_subspace_alignment      — project decoder directions onto V's eigenbasis
cluster_identity           — match features to HDBSCAN clusters
violation_layer_features   — features active specifically at violation layers
multilayer_fraction        — fraction of features with genuine cross-layer span
positional_control         — check if "long-lived" features are just positional
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Callable, Optional
from scipy import stats

from .crosscoder import Crosscoder
from .data import PromptActivationStore


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    """Decorator to register an analysis function."""
    def decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return decorator


def run_all_analyses(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: Optional[dict] = None,
    only: Optional[list[str]] = None,
) -> dict:
    """
    Run all registered analyses (or a subset if `only` is specified).

    Returns dict keyed by analysis name.
    """
    config = config or {}
    results = {}
    targets = only if only else list(_REGISTRY.keys())

    for name in targets:
        if name not in _REGISTRY:
            print(f"  Warning: analysis '{name}' not in registry, skipping")
            continue
        print(f"  Running analysis: {name}")
        try:
            results[name] = _REGISTRY[name](
                crosscoder, prompt_store, artifacts, config
            )
        except Exception as e:
            print(f"    Failed: {e}")
            results[name] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_crosscoder_on_prompt(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    prompt_key: str,
) -> dict:
    """Run crosscoder forward on a prompt, return z and x_hat."""
    x = prompt_store.get_stacked_tensor(prompt_key)
    device = next(crosscoder.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        out = crosscoder(x)
    return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in out.items()}


# ---------------------------------------------------------------------------
# Analysis: Feature lifetimes (Prediction 1)
# ---------------------------------------------------------------------------

@register("feature_lifetimes")
def feature_lifetimes(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Compute feature lifetime profiles from decoder norms.

    A feature's "lifetime" is the number of contiguous layers where its
    decoder norm exceeds a threshold (default: 10% of its max norm).

    Prediction: bimodal distribution — short-lived (1-5 layers) vs
    long-lived (20+ layers), corresponding to repulsive vs attractive
    subspace alignment.
    """
    threshold_frac = config.get("lifetime_threshold_frac", 0.1)
    norms = crosscoder.decoder_norms().numpy()  # (n_features, n_layers)
    n_features, n_layers = norms.shape

    lifetimes = np.zeros(n_features, dtype=np.int32)
    peak_layers = np.zeros(n_features, dtype=np.int32)
    max_norms = norms.max(axis=1)  # (n_features,)

    for f in range(n_features):
        if max_norms[f] < 1e-10:
            continue
        threshold = max_norms[f] * threshold_frac
        active = norms[f] > threshold

        # Longest contiguous run of True
        best_run = 0
        current_run = 0
        for val in active:
            if val:
                current_run += 1
                best_run = max(best_run, current_run)
            else:
                current_run = 0
        lifetimes[f] = best_run
        peak_layers[f] = int(np.argmax(norms[f]))

    # Bimodality test (Hartigan's dip test approximation via histogram)
    alive_lifetimes = lifetimes[max_norms > 1e-10]

    # Simple bimodality heuristic: check if the distribution has two modes
    # using a kernel density estimate
    bimodal_score = float("nan")
    if len(alive_lifetimes) > 20:
        hist, edges = np.histogram(alive_lifetimes, bins=min(n_layers, 30))
        # Count local maxima in smoothed histogram
        smoothed = np.convolve(hist, np.ones(3) / 3, mode="same")
        local_max = 0
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                local_max += 1
        bimodal_score = local_max  # 2 = bimodal, 1 = unimodal

    return {
        "lifetimes": lifetimes.tolist(),
        "peak_layers": peak_layers.tolist(),
        "decoder_norms": norms.tolist(),
        "max_norms": max_norms.tolist(),
        "mean_lifetime": float(alive_lifetimes.mean()) if len(alive_lifetimes) > 0 else 0,
        "median_lifetime": float(np.median(alive_lifetimes)) if len(alive_lifetimes) > 0 else 0,
        "bimodal_score": bimodal_score,
        "n_short_lived": int((alive_lifetimes <= 3).sum()),
        "n_long_lived": int((alive_lifetimes >= n_layers // 2).sum()),
        "n_alive": int(len(alive_lifetimes)),
    }


# ---------------------------------------------------------------------------
# Analysis: Multi-layer fraction (Surprise 1 check)
# ---------------------------------------------------------------------------

@register("multilayer_fraction")
def multilayer_fraction(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    What fraction of features have decoder norm above threshold at 3+ layers?

    If most features are single-layer, the crosscoder just recovered
    per-layer SAE features — cross-layer superposition isn't a factor.
    """
    min_layers = config.get("multilayer_min_layers", 3)
    threshold_frac = config.get("multilayer_threshold_frac", 0.1)

    norms = crosscoder.decoder_norms().numpy()  # (n_features, n_layers)
    max_norms = norms.max(axis=1)

    alive = max_norms > 1e-10
    n_alive = int(alive.sum())

    multilayer_count = 0
    layers_active_per_feature = []
    for f in range(norms.shape[0]):
        if not alive[f]:
            continue
        threshold = max_norms[f] * threshold_frac
        n_active_layers = int((norms[f] > threshold).sum())
        layers_active_per_feature.append(n_active_layers)
        if n_active_layers >= min_layers:
            multilayer_count += 1

    return {
        "multilayer_fraction": multilayer_count / max(n_alive, 1),
        "multilayer_count": multilayer_count,
        "n_alive": n_alive,
        "layers_active_distribution": (
            np.histogram(layers_active_per_feature,
                         bins=range(1, norms.shape[1] + 2))[0].tolist()
            if layers_active_per_feature else []
        ),
    }


# ---------------------------------------------------------------------------
# Analysis: V subspace alignment (Prediction 2)
# ---------------------------------------------------------------------------

@register("v_subspace_alignment")
def v_subspace_alignment(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Project each feature's decoder direction onto V's attractive and
    repulsive subspaces.

    Requires artifacts:
      "v_projectors" : dict with "sym_attract" and "sym_repulse"
                       each (d_model, d_model) ndarray.
                       For ALBERT (shared weights): one projector set.
                       For GPT-2 (per-layer): list of projector sets.
      "is_per_layer" : bool

    Prediction: long-lived features → attractive subspace,
                short-lived features → repulsive subspace.
    """
    projectors = artifacts.get("v_projectors")
    if projectors is None:
        return {"error": "v_projectors not in artifacts"}

    is_per_layer = artifacts.get("is_per_layer", False)
    directions = crosscoder.decoder_directions().numpy()  # (L, F, d)
    norms = crosscoder.decoder_norms().numpy()             # (F, L)
    L, F, d = directions.shape

    # Per-feature, per-layer projection fractions
    attract_frac = np.zeros((F, L))
    repulse_frac = np.zeros((F, L))

    for layer_idx in range(L):
        if is_per_layer and isinstance(projectors, list):
            # Map crosscoder layer index to model layer index.
            # The crosscoder samples a subset of layers; the projectors
            # cover all model layers.  layer_indices tells us the mapping.
            layer_indices = artifacts.get("layer_indices", list(range(len(projectors))))
            model_layer = layer_indices[layer_idx] if layer_idx < len(layer_indices) else layer_idx
            proj_idx = min(model_layer, len(projectors) - 1)
            P_att = projectors[proj_idx]["sym_attract"]
            P_rep = projectors[proj_idx]["sym_repulse"]
        else:
            P_att = projectors["sym_attract"]
            P_rep = projectors["sym_repulse"]

        dirs = directions[layer_idx]  # (F, d)

        # Project each feature's direction
        proj_att = dirs @ P_att  # (F, d)
        proj_rep = dirs @ P_rep  # (F, d)

        att_energy = np.sum(proj_att ** 2, axis=1)  # (F,)
        rep_energy = np.sum(proj_rep ** 2, axis=1)  # (F,)

        attract_frac[:, layer_idx] = att_energy
        repulse_frac[:, layer_idx] = rep_energy

    # Aggregate: weighted by decoder norm (so layers where the feature
    # is actually active matter more)
    weights = norms / (norms.sum(axis=1, keepdims=True) + 1e-10)
    per_feature_attract = (attract_frac * weights).sum(axis=1)
    per_feature_repulse = (repulse_frac * weights).sum(axis=1)

    # Classify: dominant subspace for each feature
    dominant = np.where(
        per_feature_attract > per_feature_repulse, "attractive", "repulsive"
    )
    # Mixed: neither dominates (both < 0.6)
    mixed = (per_feature_attract < 0.6) & (per_feature_repulse < 0.6)
    dominant[mixed] = "mixed"

    return {
        "per_feature_attract": per_feature_attract.tolist(),
        "per_feature_repulse": per_feature_repulse.tolist(),
        "per_feature_dominant": dominant.tolist(),
        "attract_frac_per_layer": attract_frac.tolist(),
        "repulse_frac_per_layer": repulse_frac.tolist(),
        "n_attractive": int((dominant == "attractive").sum()),
        "n_repulsive": int((dominant == "repulsive").sum()),
        "n_mixed": int((dominant == "mixed").sum()),
    }


# ---------------------------------------------------------------------------
# Analysis: Cluster identity features (Prediction 3)
# ---------------------------------------------------------------------------

@register("cluster_identity")
def cluster_identity(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Match crosscoder features to HDBSCAN clusters.

    For each prompt, run the crosscoder to get feature activations per token.
    For each feature, test whether it activates primarily on the tokens
    of one HDBSCAN cluster (from Phase 1).

    Requires artifacts:
      "hdbscan_labels" : dict[prompt_key -> dict[layer_idx -> (n_tokens,) int array]]

    A feature is a "cluster identity feature" if:
      - It fires on > 70% of tokens in one cluster
      - It fires on < 10% of tokens outside that cluster
    """
    hdbscan_labels = artifacts.get("hdbscan_labels")
    if hdbscan_labels is None:
        return {"error": "hdbscan_labels not in artifacts"}

    selectivity_high = config.get("cluster_selectivity_high", 0.7)
    selectivity_low = config.get("cluster_selectivity_low", 0.1)

    results_per_prompt = {}

    for prompt_key in prompt_store.keys():
        if prompt_key not in hdbscan_labels:
            continue

        out = _run_crosscoder_on_prompt(crosscoder, prompt_store, prompt_key)
        z = out["z"].numpy()  # (n_tokens, n_features)
        active = z > 0

        prompt_labels = hdbscan_labels[prompt_key]
        identity_features = []

        for layer_key, labels in prompt_labels.items():
            labels = np.array(labels)
            cluster_ids = sorted(set(labels) - {-1})

            for feat_idx in range(z.shape[1]):
                if not active[:, feat_idx].any():
                    continue

                feat_active = active[:, feat_idx]

                for cid in cluster_ids:
                    in_cluster = labels == cid
                    n_cluster = in_cluster.sum()
                    n_outside = (~in_cluster & (labels != -1)).sum()

                    if n_cluster == 0:
                        continue

                    recall = feat_active[in_cluster].sum() / n_cluster
                    if n_outside > 0:
                        false_positive = feat_active[~in_cluster & (labels != -1)].sum() / n_outside
                    else:
                        false_positive = 0.0

                    if recall >= selectivity_high and false_positive <= selectivity_low:
                        identity_features.append({
                            "feature": int(feat_idx),
                            "cluster_id": int(cid),
                            "layer_key": str(layer_key),
                            "recall": float(recall),
                            "false_positive_rate": float(false_positive),
                            "cluster_size": int(n_cluster),
                        })

        results_per_prompt[prompt_key] = {
            "n_identity_features": len(identity_features),
            "identity_features": identity_features,
        }

    return results_per_prompt


# ---------------------------------------------------------------------------
# Analysis: Violation-layer features (Prediction 4)
# ---------------------------------------------------------------------------

@register("violation_layer_features")
def violation_layer_features(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Find features that activate specifically at violation layers.

    Requires artifacts:
      "violation_layers" : dict[prompt_key -> list[int]]  (from Phase 1)
      "layer_indices"    : list[int]  (which model layers the crosscoder covers)

    For each prompt, compare feature activations at violation layers vs
    non-violation layers.  Features with z-score > 2 at violation layers
    are flagged.
    """
    violation_layers = artifacts.get("violation_layers")
    layer_indices = artifacts.get("layer_indices")
    if violation_layers is None or layer_indices is None:
        return {"error": "violation_layers or layer_indices not in artifacts"}

    z_threshold = config.get("violation_z_threshold", 2.0)
    results_per_prompt = {}

    for prompt_key in prompt_store.keys():
        if prompt_key not in violation_layers:
            continue

        out = _run_crosscoder_on_prompt(crosscoder, prompt_store, prompt_key)
        z = out["z"].numpy()  # (n_tokens, n_features)

        # Map violation layers to crosscoder layer indices
        v_layers = set(violation_layers[prompt_key])
        sampled_violation = [
            i for i, l in enumerate(layer_indices) if l in v_layers
        ]
        sampled_non_violation = [
            i for i, l in enumerate(layer_indices) if l not in v_layers
        ]

        if not sampled_violation or not sampled_non_violation:
            results_per_prompt[prompt_key] = {"n_violation_features": 0}
            continue

        # Feature activation is summed across tokens (we want layer-level signal).
        # But the crosscoder gives one z per token, not per layer.
        # Instead, look at which features are active on tokens and check
        # if the decoder norms at violation layers are elevated.
        # This is a different test: which features have high decoder norm
        # specifically at violation layers?

        norms = crosscoder.decoder_norms().numpy()  # (F, L)
        violation_norms = norms[:, sampled_violation].mean(axis=1)
        non_violation_norms = norms[:, sampled_non_violation].mean(axis=1)

        pop_std = non_violation_norms.std()
        if pop_std < 1e-10:
            results_per_prompt[prompt_key] = {"n_violation_features": 0}
            continue

        z_scores = (violation_norms - non_violation_norms) / (pop_std + 1e-10)

        violation_features = []
        for f in range(len(z_scores)):
            if z_scores[f] > z_threshold:
                violation_features.append({
                    "feature": int(f),
                    "z_score": float(z_scores[f]),
                    "violation_norm": float(violation_norms[f]),
                    "non_violation_norm": float(non_violation_norms[f]),
                })

        violation_features.sort(key=lambda x: x["z_score"], reverse=True)
        results_per_prompt[prompt_key] = {
            "n_violation_features": len(violation_features),
            "violation_features": violation_features[:50],
        }

    return results_per_prompt


# ---------------------------------------------------------------------------
# Analysis: Positional control (Surprise 3 check)
# ---------------------------------------------------------------------------

@register("positional_control")
def positional_control(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Check whether long-lived features correlate with token position.

    For each feature, compute the Spearman correlation between feature
    activation and token position across all prompts.  High correlation
    means the feature is positional, not dynamical.
    """
    norms = crosscoder.decoder_norms().numpy()  # (F, L)
    max_norms = norms.max(axis=1)

    # Identify long-lived features (active at 50%+ of layers)
    threshold_frac = config.get("lifetime_threshold_frac", 0.1)
    long_lived = []
    for f in range(norms.shape[0]):
        if max_norms[f] < 1e-10:
            continue
        threshold = max_norms[f] * threshold_frac
        n_active = int((norms[f] > threshold).sum())
        if n_active >= norms.shape[1] // 2:
            long_lived.append(f)

    if not long_lived:
        return {"n_long_lived": 0, "n_positional": 0}

    # For each long-lived feature, check position correlation
    positional_features = []
    for feat_idx in long_lived:
        all_acts = []
        all_positions = []

        for prompt_key in prompt_store.keys():
            out = _run_crosscoder_on_prompt(crosscoder, prompt_store, prompt_key)
            z = out["z"][:, feat_idx].numpy()  # (n_tokens,)
            positions = np.arange(len(z))
            all_acts.extend(z.tolist())
            all_positions.extend(positions.tolist())

        if len(all_acts) < 10:
            continue

        rho, pval = stats.spearmanr(all_acts, all_positions)
        if abs(rho) > 0.5 and pval < 0.05:
            positional_features.append({
                "feature": feat_idx,
                "rho": float(rho),
                "pval": float(pval),
            })

    return {
        "n_long_lived": len(long_lived),
        "n_positional": len(positional_features),
        "positional_fraction": len(positional_features) / max(len(long_lived), 1),
        "positional_features": positional_features,
    }


# ---------------------------------------------------------------------------
# Analysis: Lifetime vs V-alignment correlation (Predictions 1+2 combined)
# ---------------------------------------------------------------------------

@register("lifetime_vs_alignment")
def lifetime_vs_alignment(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Test the central prediction: long-lived features align with V's
    attractive subspace, short-lived with repulsive.

    Requires both feature_lifetimes and v_subspace_alignment to have
    been computed (or computes them inline).

    Returns Spearman correlation between lifetime and attractive_fraction.
    """
    projectors = artifacts.get("v_projectors")
    if projectors is None:
        return {"error": "v_projectors not in artifacts"}

    # Compute lifetimes
    lt_result = feature_lifetimes(crosscoder, prompt_store, artifacts, config)
    lifetimes = np.array(lt_result["lifetimes"])

    # Compute alignment
    va_result = v_subspace_alignment(crosscoder, prompt_store, artifacts, config)
    attract = np.array(va_result["per_feature_attract"])

    # Filter to alive features
    alive = np.array(lt_result["max_norms"]) > 1e-10
    if alive.sum() < 10:
        return {"error": "fewer than 10 alive features"}

    lt_alive = lifetimes[alive]
    att_alive = attract[alive]

    rho, pval = stats.spearmanr(lt_alive, att_alive)

    return {
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "n_features": int(alive.sum()),
        "prediction_confirmed": rho > 0.2 and pval < 0.05,
        "interpretation": (
            "Positive correlation: long-lived features align with "
            "attractive subspace, as predicted."
            if rho > 0.2 and pval < 0.05
            else "No significant correlation between lifetime and "
                 "attractive alignment."
        ),
    }


# ---------------------------------------------------------------------------
# Save / load results
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str | Path):
    """Save analysis results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(results), f, indent=2)
    print(f"  Analysis results saved to {path}")
