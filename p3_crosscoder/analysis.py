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




# ---------------------------------------------------------------------------
# Summary registry
# ---------------------------------------------------------------------------

_SUMMARY_REGISTRY: dict[str, Callable[[dict], str]] = {}


def register_summary(name: str):
    """Decorator to register a summarizer for an analysis result dict."""
    def decorator(fn):
        _SUMMARY_REGISTRY[name] = fn
        return fn
    return decorator


def _default_summarize(name: str, result: dict) -> str:
    """
    Fallback summarizer used when no @register_summary exists for `name`.
    Emits top-level scalar values and array shapes.
    """
    if "error" in result:
        return f"ERROR: {result['error']}"
    lines = []
    for k, v in result.items():
        if isinstance(v, (int, float, str)) and not isinstance(v, bool):
            lines.append(f"{k}: {v}")
        elif isinstance(v, bool):
            lines.append(f"{k}: {v}")
        elif isinstance(v, (list, np.ndarray)):
            arr = np.asarray(v) if not isinstance(v, np.ndarray) else v
            lines.append(f"{k}: array{list(arr.shape)}")
        elif isinstance(v, dict):
            lines.append(f"{k}: dict({len(v)} keys)")
    return "\n".join(lines) if lines else "(no scalar fields)"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_ARRAY_OFFLOAD_THRESHOLD = 500  # elements; arrays larger than this go to .npz


def _save_json(data, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            data, f, indent=2,
            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
        )


def _extract_arrays(result: dict, threshold: int = _ARRAY_OFFLOAD_THRESHOLD) -> dict:
    """
    Pull out numpy arrays larger than `threshold` elements for .npz offload.
    Does not recurse into nested dicts.
    """
    return {
        k: v for k, v in result.items()
        if isinstance(v, np.ndarray) and v.size > threshold
    }


def _json_safe(result: dict, arrays_offloaded: set) -> dict:
    """Replace offloaded arrays with shape-note strings."""
    out = {}
    for k, v in result.items():
        if k in arrays_offloaded:
            shape = list(v.shape) if hasattr(v, "shape") else "?"
            out[k] = f"<offloaded to .npz shape={shape}>"
        else:
            out[k] = v
    return out
def run_all_analyses(
    crosscoder: "Crosscoder",
    prompt_store: "PromptActivationStore",
    artifacts: dict,
    config: Optional[dict] = None,
    only: Optional[list] = None,
    out_dir: Optional[Path] = None,
) -> "tuple[list, dict] | dict":
    """
    Run all registered analyses (or a subset if `only` is specified).

    Parameters
    ----------
    out_dir : if provided, each analysis result is written to
              out_dir/analyses/<n>.json as it completes.  Heavy numpy arrays
              (> _ARRAY_OFFLOAD_THRESHOLD elements) are split into a sibling
              out_dir/analyses/<n>.npz.  An index is written at the end.
              Returns (summary_blocks, index) when out_dir is given.

              If out_dir is None (legacy mode) returns the plain dict keyed
              by analysis name.

    Returns
    -------
    (summary_blocks, index)  when out_dir is provided
        summary_blocks : list of (name, text_block) pairs
        index          : dict[name -> {file, has_error, has_npz}]

    dict[name -> result]  when out_dir is None (backward compat)
    """
    config  = config or {}
    targets = only if only else list(_REGISTRY.keys())

    if out_dir is None:
        # ---- Legacy path ----
        results = {}
        for name in targets:
            if name not in _REGISTRY:
                print(f"  Warning: analysis '{name}' not in registry, skipping")
                continue
            print(f"  Running analysis: {name}")
            try:
                results[name] = _REGISTRY[name](crosscoder, prompt_store, artifacts, config)
            except Exception as e:
                print(f"    Failed: {e}")
                results[name] = {"error": str(e)}
        return results

    # ---- Streaming path ----
    analyses_dir = Path(out_dir) / "analyses"
    analyses_dir.mkdir(parents=True, exist_ok=True)

    summary_blocks: list = []
    index: dict = {}

    for name in targets:
        if name not in _REGISTRY:
            print(f"  Warning: analysis '{name}' not in registry, skipping")
            continue
        print(f"  Running analysis: {name}")
        try:
            result = _REGISTRY[name](crosscoder, prompt_store, artifacts, config)
        except Exception as e:
            print(f"    Failed: {e}")
            result = {"error": str(e)}

        arrays        = _extract_arrays(result)
        offloaded_keys: set = set()
        if arrays:
            np.savez_compressed(analyses_dir / f"{name}.npz", **arrays)
            offloaded_keys = set(arrays.keys())

        out_file = analyses_dir / f"{name}.json"
        _save_json(_json_safe(result, offloaded_keys), out_file)

        summarizer = _SUMMARY_REGISTRY.get(name)
        if summarizer is not None:
            try:
                block = summarizer(result)
            except Exception as e:
                block = f"(summarizer failed: {e})\n" + _default_summarize(name, result)
        else:
            block = _default_summarize(name, result)

        summary_blocks.append((name, block))
        index[name] = {
            "file":      str(out_file.relative_to(out_dir)),
            "has_error": "error" in result,
            "has_npz":   bool(arrays),
        }
        print(f"    -> {out_file.name}" + (" + .npz" if arrays else ""))

    _save_json(index, analyses_dir / "index.json")
    return summary_blocks, index



def save_results(results: dict, path: Path):
    """
    Backward-compatible single-file serializer.

    Still used by callers that haven't migrated to the streaming path.
    Writes one combined JSON, replacing numpy arrays with .tolist().
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            results, f, indent=2,
            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
        )
    print(f"  Results saved to {path}")


def summarize_steering(steer_summary: dict) -> str:
    """
    Convert a steering summary dict (from summarise_steering) into a
    text block for summary.txt.
    """
    if not steer_summary or "error" in steer_summary:
        return f"ERROR: {steer_summary.get('error', 'no steering results')}"
    lines = []
    text = steer_summary.get("text_summary", "")
    if text:
        lines.extend(text.splitlines()[:30])
        extra = len(text.splitlines()) - 30
        if extra > 0:
            lines.append(f"... ({extra} more lines in steering/steering_summary.json)")
    else:
        lines = [
            f"n_experiments: {steer_summary.get('n_experiments', '?')}",
            f"n_significant: {steer_summary.get('n_significant', '?')}",
        ]
    return "\n".join(lines)



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
# Helper: per-feature, per-layer activity scores from data
# ---------------------------------------------------------------------------

def _compute_feature_layer_scores(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
) -> np.ndarray:
    """
    Compute per-feature, per-layer activity scores from actual prompt data.

    Because normalize_decoder() keeps all W_dec column norms at 1.0,
    decoder_norms() returns a constant (F, L) matrix and cannot distinguish
    which layers a feature actually matters at.

    Instead, for each feature f and layer l we compute the mean squared
    projection of the true residual x[l] onto W_dec[l, f, :], averaged
    over tokens where the feature fires.  Features with genuinely different
    decoder directions across layers will show different per-layer scores;
    features whose decoder direction at a given layer is orthogonal to the
    actual residuals at that layer will score near zero there.

    Parameters
    ----------
    crosscoder   : trained Crosscoder (on any device)
    prompt_store : PromptActivationStore with eval prompts

    Returns
    -------
    scores : (n_features, n_layers) float64 ndarray
    """
    device = next(crosscoder.parameters()).device
    W_dec = crosscoder.W_dec.detach().cpu().numpy()  # (L, F, d)
    L, F, _ = W_dec.shape

    scores = np.zeros((F, L), dtype=np.float64)
    counts = np.zeros(F, dtype=np.float64)

    for prompt_key in prompt_store.keys():
        x = prompt_store.get_stacked_tensor(prompt_key)   # (T, L, d)
        with torch.no_grad():
            out = crosscoder(x.to(device))
        z = out["z"].cpu().numpy()      # (T, F)
        x_np = x.numpy()               # (T, L, d)

        active = z > 0                 # (T, F)

        for l in range(L):
            x_l = x_np[:, l, :]       # (T, d)
            # Projection of every token onto every decoder direction at l
            proj = x_l @ W_dec[l].T   # (T, F)
            # Mean squared projection, zeroed for inactive tokens
            scores[:, l] += (proj ** 2 * active).sum(axis=0)

        counts += active.sum(axis=0)

    # Average over active tokens; features that never fire → scores stay 0
    safe_counts = np.maximum(counts, 1.0)[:, np.newaxis]
    scores /= safe_counts

    return scores


# ---------------------------------------------------------------------------
# Projector coverage diagnostic
# ---------------------------------------------------------------------------

def _projector_coverage(projectors, is_per_layer: bool, d: int) -> dict:
    """
    Compute the effective coverage of the V-subspace projectors.

    For a projector P = U @ U.T built from k eigenvectors in R^d:
      - rank(P) = k  (== trace(P) for a proper projection matrix)
      - A random unit vector projects exactly k/d of its squared norm
        into P on average, with std ~ sqrt(k) / d.

    When k/d is small the classifier cannot distinguish true subspace
    alignment from noise: attract_dominance concentrates tightly around
    k_att / (k_att + k_rep) for every feature regardless of true alignment.

    The minimum useful coverage is ~20% of d.  Below that the test is
    uninformative and should not be reported as a finding.

    Parameters
    ----------
    projectors   : single dict or list of dicts with "sym_attract" / "sym_repulse"
    is_per_layer : whether projectors is a list (one per model layer)
    d            : ambient dimension (d_model)

    Returns
    -------
    dict with:
      k_attract       : rank of the attractive projector
      k_repulse       : rank of the repulsive projector
      coverage_attract: k_attract / d
      coverage_repulse: k_repulse / d
      min_coverage    : min of the two
      underpowered    : True if min_coverage < min_coverage_threshold (0.20)
      note            : human-readable explanation when underpowered
    """
    # For per-layer models use the first layer's projectors as representative.
    p = projectors[0] if is_per_layer else projectors

    # rank(P) = trace(P) for an orthogonal projector (eigenvalues 0 or 1).
    k_att = int(round(float(np.trace(p["sym_attract"]))))
    k_rep = int(round(float(np.trace(p["sym_repulse"]))))

    cov_att = k_att / d
    cov_rep = k_rep / d
    min_cov = min(cov_att, cov_rep)

    # Expected attract_dominance for a random unit vector:
    #   E[attract_dominance] = k_att / (k_att + k_rep)
    # When k_att == k_rep this is exactly 0.5; std ~ 1 / sqrt(k_att + k_rep).
    expected_dominance = k_att / (k_att + k_rep) if (k_att + k_rep) > 0 else 0.5
    classifier_std = 1.0 / np.sqrt(k_att + k_rep) if (k_att + k_rep) > 0 else float("inf")

    threshold = 0.20
    underpowered = min_cov < threshold

    note = (
        f"Projectors are low-rank (k_att={k_att}, k_rep={k_rep}, d={d}). "
        f"Coverage: attract={cov_att:.1%}, repulse={cov_rep:.1%} of ambient space. "
        f"A random unit vector has expected attract_dominance={expected_dominance:.3f} "
        f"± {classifier_std:.3f} — the classifier cannot separate true alignment "
        f"from noise at this rank. Minimum useful coverage is {threshold:.0%} of d "
        f"(requires k ≥ {int(threshold * d)}). "
        f"Results from v_subspace_alignment and lifetime_vs_alignment are uninformative "
        f"and should not be treated as negative findings."
    ) if underpowered else (
        f"Projectors have sufficient coverage "
        f"(k_att={k_att}, k_rep={k_rep}, d={d}, "
        f"min_coverage={min_cov:.1%} ≥ {threshold:.0%})."
    )

    return {
        "k_attract": k_att,
        "k_repulse": k_rep,
        "d_model": d,
        "coverage_attract": float(cov_att),
        "coverage_repulse": float(cov_rep),
        "min_coverage": float(min_cov),
        "expected_dominance_random": float(expected_dominance),
        "classifier_std_random": float(classifier_std),
        "underpowered": underpowered,
        "min_coverage_threshold": threshold,
        "note": note,
    }



# ---------------------------------------------------------------------------
# Helpers: plateau clustering
# ---------------------------------------------------------------------------

def _closest_sampled_layer(model_layer: int, layer_indices: list) -> int:
    """
    Return the crosscoder layer index (position in layer_indices) whose
    model layer number is closest to model_layer.

    Parameters
    ----------
    model_layer   : target model layer number (from plateau_layers artifact)
    layer_indices : list of model layer numbers the crosscoder was trained on

    Returns
    -------
    int — index into layer_indices (i.e. the crosscoder layer axis position)
    """
    diffs = [abs(ml - model_layer) for ml in layer_indices]
    return int(np.argmin(diffs))


def _spectral_bipartition(acts: np.ndarray) -> np.ndarray:
    """
    Compute the dominant spectral bipartition of a token activation matrix.

    Builds a cosine-similarity graph on the unit-sphere-projected activations,
    constructs the symmetric normalized Laplacian, and thresholds the sign of
    the Fiedler vector (2nd-smallest eigenvector) to produce binary labels.

    Parameters
    ----------
    acts : (n_tokens, d_model) — raw residual stream activations at one layer

    Returns
    -------
    labels : (n_tokens,) int array — 0 or 1 per token
    """
    # Project onto unit sphere (matches Phase 1's layernorm_to_sphere convention)
    norms = np.linalg.norm(acts, axis=1, keepdims=True).clip(min=1e-8)
    acts_n = acts / norms                              # (T, d)

    # Cosine similarity matrix, clipped to [0, 1] to remove repulsive edges.
    # Negative similarities correspond to opposing directions and should not
    # count as affinity — keeping them creates a bipartition by sign rather
    # than by density, which is exactly what we do NOT want here.
    S = (acts_n @ acts_n.T).clip(min=0.0)             # (T, T)

    # Symmetric normalized Laplacian: L_sym = I - D^{-1/2} S D^{-1/2}
    deg = S.sum(axis=1)                                # (T,)
    deg_inv_sqrt = np.where(deg > 1e-10, 1.0 / np.sqrt(deg), 0.0)
    S_norm = deg_inv_sqrt[:, None] * S * deg_inv_sqrt[None, :]

    # Fiedler vector: 2nd-smallest eigenvector of the symmetric normalized
    # Laplacian L_sym = I - S_norm.
    # scipy.linalg.eigh returns eigenvalues in ascending order, so
    # subset_by_index=[0, 1] fetches the trivial constant vector (eigenvalue 0)
    # and the Fiedler vector (eigenvalue = algebraic connectivity).
    # Thresholding the sign of the Fiedler vector gives the bipartition.
    from scipy.linalg import eigh
    n = S_norm.shape[0]
    L_sym = np.eye(n) - S_norm
    eigvals, eigvecs = eigh(L_sym, subset_by_index=[0, min(1, n - 1)])
    # eigvecs[:, 1] is the Fiedler vector; degenerate case (n=2) uses col 0
    fiedler = eigvecs[:, 1] if eigvecs.shape[1] > 1 else eigvecs[:, 0]

    labels = (fiedler >= 0.0).astype(int)
    # Normalize so the larger group is always label 0
    if labels.sum() < len(labels) / 2:
        labels = 1 - labels
    return labels


def _hdbscan_within_partitions(
    acts: np.ndarray,
    partition_labels: np.ndarray,
    min_cluster_size: int = 3,
) -> np.ndarray:
    """
    Run HDBSCAN independently within each spectral partition and merge
    into a single label array with globally unique cluster IDs.

    Parameters
    ----------
    acts             : (n_tokens, d_model) activations
    partition_labels : (n_tokens,) 0/1 labels from _spectral_bipartition
    min_cluster_size : passed to hdbscan.HDBSCAN

    Returns
    -------
    labels : (n_tokens,) int array
      Noise tokens → -1
      Partition 0 clusters → 0, 1, 2, ...
      Partition 1 clusters → k0, k0+1, ... where k0 = n_clusters in partition 0

    Notes
    -----
    Uses cosine metric.  Falls back to a single-cluster label if a partition
    is too small for HDBSCAN or if hdbscan is not installed.
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        # Graceful fallback: treat each partition as one cluster
        merged = np.where(partition_labels == 0, 0, 1).astype(int)
        return merged

    norms = np.linalg.norm(acts, axis=1, keepdims=True).clip(min=1e-8)
    acts_n = acts / norms

    merged = np.full(len(acts), -1, dtype=int)
    cluster_offset = 0

    for part_id in (0, 1):
        mask = partition_labels == part_id
        idx = np.where(mask)[0]
        if len(idx) < min_cluster_size:
            continue

        part_acts = acts_n[idx]                         # (n_part, d)
        mcs = max(min_cluster_size, max(3, len(idx) // 10))

        try:
            clusterer = hdbscan_lib.HDBSCAN(
                min_cluster_size=mcs,
                metric="euclidean",   # on unit-normed vecs, euclidean ≈ cosine
                core_dist_n_jobs=1,
            )
            part_labels = clusterer.fit_predict(part_acts)  # -1 = noise
        except Exception:
            # Any HDBSCAN failure → treat partition as one cluster
            part_labels = np.zeros(len(idx), dtype=int)

        # Remap cluster ids to global offset, keep -1 as noise
        unique_clusters = sorted(set(part_labels) - {-1})
        remap = {c: (i + cluster_offset) for i, c in enumerate(unique_clusters)}
        for local_i, global_i in enumerate(idx):
            lbl = part_labels[local_i]
            merged[global_i] = remap[lbl] if lbl != -1 else -1

        cluster_offset += len(unique_clusters)

    return merged


def _compute_plateau_clusters(
    prompt_store: "PromptActivationStore",
    layer_indices: list,
    plateau_layers: dict,
    min_cluster_size: int = 3,
) -> dict:
    """
    Compute two-resolution cluster labels at mid-plateau layers for every
    eval prompt.

    Called by both plateau_clustering (registered analysis) and
    cluster_identity (which calls it inline when plateau_layers is available).

    Parameters
    ----------
    prompt_store    : PromptActivationStore
    layer_indices   : list of model layer numbers the crosscoder covers
    plateau_layers  : {prompt_key: [mid_layer_num, ...]}
    min_cluster_size: passed through to HDBSCAN

    Returns
    -------
    {
      prompt_key: {
        mid_layer_num (int): {
          "spectral"          : [label, ...]  — 0/1, n_tokens entries
          "hdbscan"           : [label, ...]  — finer, -1 = noise
          "n_hdbscan_clusters": int
          "sampled_layer_idx" : int           — crosscoder layer axis position used
          "actual_model_layer": int           — the sampled model layer used
          "distance_to_target": int           — |actual - requested|
        }
      }
    }
    """
    result = {}

    for prompt_key in prompt_store.keys():
        if prompt_key not in plateau_layers:
            continue

        x = prompt_store.get_stacked_tensor(prompt_key).numpy()  # (T, L, d)
        T, L, d = x.shape
        prompt_result = {}

        for mid_layer in plateau_layers[prompt_key]:
            cc_idx = _closest_sampled_layer(mid_layer, layer_indices)
            actual_model_layer = layer_indices[cc_idx]
            acts = x[:, cc_idx, :]  # (T, d)

            if T < 4:
                # Degenerate: too few tokens to cluster
                prompt_result[mid_layer] = {
                    "spectral": [0] * T,
                    "hdbscan": list(range(T)),
                    "n_hdbscan_clusters": T,
                    "sampled_layer_idx": cc_idx,
                    "actual_model_layer": actual_model_layer,
                    "distance_to_target": abs(actual_model_layer - mid_layer),
                    "note": "degenerate: fewer than 4 tokens",
                }
                continue

            spectral_labels = _spectral_bipartition(acts)
            hdbscan_labels  = _hdbscan_within_partitions(
                acts, spectral_labels, min_cluster_size=min_cluster_size
            )
            n_hdbscan = len(set(hdbscan_labels) - {-1})

            prompt_result[mid_layer] = {
                "spectral":           spectral_labels.tolist(),
                "hdbscan":            hdbscan_labels.tolist(),
                "n_hdbscan_clusters": n_hdbscan,
                "sampled_layer_idx":  cc_idx,
                "actual_model_layer": actual_model_layer,
                "distance_to_target": abs(actual_model_layer - mid_layer),
            }

        if prompt_result:
            result[prompt_key] = prompt_result

    return result


# ---------------------------------------------------------------------------
# Analysis: Feature lifetimes (Prediction 1)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Summarizers — one per registered analysis
# ---------------------------------------------------------------------------

@register_summary("feature_lifetimes")
def _summarize_feature_lifetimes(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    bc      = r.get("bimodality_coefficient", float("nan"))
    test    = r.get("bimodality_test", "?")
    valley  = r.get("valley_threshold")
    lines = [
        f"n_features: {r.get('n_features', '?')}",
        f"bimodality_coefficient: {bc:.4f}" if isinstance(bc, float) else f"bimodality_coefficient: {bc}",
        f"bimodality_test: {test}",
        f"valley_threshold: {valley}",
        f"n_short_lived: {r.get('n_short_lived', '?')}",
        f"n_long_lived: {r.get('n_long_lived', '?')}",
        f"n_dead: {r.get('n_dead', '?')}",
    ]
    verdict = r.get("verdict") or test
    if verdict:
        lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


@register_summary("v_subspace_alignment")
def _summarize_v_subspace_alignment(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    rho = r.get("spearman_rho", float("nan"))
    p   = r.get("spearman_p",   float("nan"))
    lines = [
        f"spearman_rho: {rho:.4f}" if isinstance(rho, float) else f"spearman_rho: {rho}",
        f"spearman_p: {p:.4f}"     if isinstance(p, float)   else f"spearman_p: {p}",
        f"n_repulsive_dominant: {r.get('n_repulsive_dominant', '?')}",
        f"n_attractive_dominant: {r.get('n_attractive_dominant', '?')}",
    ]
    interp = r.get("interpretation", "")
    if interp:
        lines.append(f"interpretation: {interp}")
    return "\n".join(lines)


@register_summary("cluster_identity")
def _summarize_cluster_identity(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    n_prompts = sum(1 for k, v in r.items() if isinstance(v, dict) and k != "overall")
    lines = [f"n_prompts_run: {n_prompts}"]
    overall = r.get("overall", {})
    if overall:
        lines.append(f"clustering_source: {overall.get('clustering_source', '?')} ")
    for pk, pd in r.items():
        if not isinstance(pd, dict) or pk == "overall":
            continue
        lines.append(f"  {pk}: {len(pd)} plateau layer(s) evaluated")
    return "\n".join(lines)


@register_summary("violation_layer_features")
def _summarize_violation_layer_features(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    frac = r.get("fraction_violation_specific")
    lines = [
        f"n_violation_layers: {r.get('n_violation_layers', '?')}",
        f"n_features_at_violations: {r.get('n_features_at_violations', '?')}",
    ]
    if frac is not None:
        lines.append(f"fraction_violation_specific: {frac:.4f}")
    verdict = r.get("verdict", "")
    if verdict:
        lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


@register_summary("multilayer_fraction")
def _summarize_multilayer_fraction(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    frac = r.get("multilayer_fraction", float("nan"))
    lines = [
        f"multilayer_fraction: {frac:.4f}" if isinstance(frac, float) else f"multilayer_fraction: {frac}",
        f"n_multilayer: {r.get('multilayer_count', r.get('n_multilayer', '?'))}",
        f"n_alive: {r.get('n_alive', r.get('n_features_total', '?'))}",
        f"min_layers_threshold: {r.get('min_layers_threshold', '?')}",
    ]
    verdict = r.get("verdict", "")
    if verdict:
        lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


@register_summary("positional_control")
def _summarize_positional_control(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    frac_pos = r.get("frac_positional_among_long_lived")
    lines = [f"n_positional: {r.get('n_positional', '?')}"]
    if frac_pos is not None:
        lines.append(f"frac_positional_among_long_lived: {frac_pos:.4f}")
    verdict = r.get("verdict", "")
    if verdict:
        lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


@register_summary("feature_cluster_correlation")
def _summarize_feature_cluster_correlation(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    overall = r.get("overall", {})
    lines = [
        f"n_prompts_run: {overall.get('n_prompts_run', '?')}",
        f"clustering_source: {overall.get('clustering_source', '?')}",
        f"top_k: {overall.get('top_k', '?')}",
    ]
    for pk, layer_dict in r.items():
        if pk == "overall" or not isinstance(layer_dict, dict):
            continue
        for layer_key, info in layer_dict.items():
            if not isinstance(info, dict):
                continue
            for res_key in ("spectral", "hdbscan"):
                res = info.get(res_key, {})
                top = res.get("top_features", [])
                if top:
                    top_f = top[0].get("f_spectral", top[0].get("f_stat", "?"))
                    n_above = res.get("n_features_above_min", "?")
                    if isinstance(top_f, float):
                        lines.append(
                            f"  {pk}/{layer_key}/{res_key}: top_f={top_f:.2f}, n_above_min={n_above}"
                        )
                    else:
                        lines.append(f"  {pk}/{layer_key}/{res_key}: top_f={top_f}")
    return "\n".join(lines)


@register_summary("inspect_top_features")
def _summarize_inspect_top_features(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    text_report = r.get("text_report", "")
    if text_report:
        lines = text_report.splitlines()
        trimmed = lines[:40]
        if len(lines) > 40:
            trimmed.append(f"... ({len(lines) - 40} more lines in analyses/inspect_top_features.txt)")
        return "\n".join(trimmed)
    return f"n_features_inspected: {r.get('n_features_inspected', '?')}"


@register_summary("ffn_repulsive_feature_alignment")
def _summarize_ffn_repulsive(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    mean_cs = r.get("mean_cos_sim_to_ffn_delta") or r.get("mean_cosine_repulsive")
    lines = [
        f"n_violations_checked: {r.get('n_violations_checked', '?')}",
        f"n_features_aligned: {r.get('n_features_aligned', '?')}",
    ]
    if mean_cs is not None:
        lines.append(f"mean_cosine: {mean_cs:.4f}")
    verdict = r.get("verdict", "")
    if verdict:
        lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


@register_summary("cross_term_feature_weighting")
def _summarize_cross_term(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    mean_w = r.get("mean_cross_term_weight")
    lines = [f"n_layers_checked: {r.get('n_layers_checked', '?')}"]
    if mean_w is not None:
        lines.append(f"mean_cross_term_weight: {mean_w:.4f}")
    verdict = r.get("verdict", "")
    if verdict:
        lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


@register_summary("induction_feature_tagging")
def _summarize_induction(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    frac = r.get("induction_fraction")
    lines = [
        f"n_induction_tagged: {r.get('n_induction_tagged', '?')}",
        f"n_features_checked: {r.get('n_features_checked', '?')}",
    ]
    if frac is not None:
        lines.append(f"induction_fraction: {frac:.4f}")
    verdict = r.get("verdict", "")
    if verdict:
        lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


@register_summary("decoder_violation_projection")
def _summarize_decoder_violation(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    per_viol = r.get("per_violation", [])
    mean_exp = r.get("mean_explained")
    lines = [f"n_violations: {len(per_viol)}"]
    if mean_exp is not None:
        lines.append(f"mean_explained_variance: {mean_exp:.4f}")
    verdict = r.get("verdict", "")
    if verdict:
        lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


@register_summary("lifetime_centroid_decomposition")
def _summarize_lifetime_centroid(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    mean_rep = r.get("mean_repulsive_cos")
    mean_att = r.get("mean_attractive_cos")
    lines = [f"n_long_lived_inspected: {r.get('n_long_lived_inspected', '?')}"]
    if mean_rep is not None:
        lines.append(f"mean_repulsive_cos: {mean_rep:.4f}")
    if mean_att is not None:
        lines.append(f"mean_attractive_cos: {mean_att:.4f}")
    verdict = r.get("verdict", "")
    if verdict:
        lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


@register_summary("coactivation_at_merges")
def _summarize_coactivation_at_merges(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    mean_coact    = r.get("mean_coactivation_at_merges")
    mean_baseline = r.get("mean_coactivation_baseline")
    lines = [f"n_merge_layers_checked: {r.get('n_merge_layers_checked', '?')}"]
    if mean_coact is not None:
        lines.append(f"mean_coactivation_at_merges: {mean_coact:.4f}")
    if mean_baseline is not None:
        lines.append(f"mean_coactivation_baseline: {mean_baseline:.4f}")
    verdict = r.get("verdict", "")
    if verdict:
        lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


@register_summary("cluster_identity_diff")
def _summarize_cluster_identity_diff(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    mean_overlap = r.get("mean_feature_overlap")
    lines = [f"n_layers_compared: {r.get('n_layers_compared', '?')}"]
    if mean_overlap is not None:
        lines.append(f"mean_feature_overlap: {mean_overlap:.4f}")
    verdict = r.get("verdict", "")
    if verdict:
        lines.append(f"verdict: {verdict}")
    return "\n".join(lines)

@register("feature_lifetimes")
def feature_lifetimes(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Compute feature lifetime profiles and classify features into short-lived
    vs long-lived populations.

    A feature's "lifetime" is the length of its longest contiguous run of
    active layers, where "active" means the data-driven layer score exceeds
    threshold_frac * max_score.

    Bimodality is tested with the bimodality coefficient:

        BC = (gamma_1^2 + 1) / gamma_2

    where gamma_1 is skewness and gamma_2 is kurtosis (not excess).
    BC > 5/9 ≈ 0.555 reliably indicates bimodality (SAS Institute, 1990).
    This replaces the previous local-maxima count on a smoothed histogram,
    which returned 2 for any skewed unimodal distribution with a shoulder.

    When bimodal, the valley threshold is the lifetime value at the minimum
    KDE density between the two highest-density peaks.  Features are
    classified as "short_lived" (lifetime < valley) or "long_lived"
    (lifetime >= valley).  This classification is the anchor for downstream
    analyses: cluster_identity and violation_layer_features can stratify
    their results by lifetime class.

    Dead features (max_score < 1e-10) are classified as "dead".

    Returns
    -------
    dict with all previous fields plus:
      bimodality_coefficient : float — BC value
      bimodality_test        : "bimodal" | "unimodal" | "insufficient_data"
      valley_threshold       : int or None — lifetime at density minimum
      n_short_lived          : count below valley_threshold (or <= 3 if unimodal)
      n_long_lived           : count at/above valley_threshold (or >= L/2 if unimodal)
      lifetime_class         : (n_features,) list — "short_lived" | "long_lived" | "dead"
      short_lived_indices    : feature indices classified as short_lived
      long_lived_indices     : feature indices classified as long_lived
    """
    from scipy.stats import gaussian_kde

    threshold_frac = config.get("lifetime_threshold_frac", 0.1)

    # Use data-driven scores rather than decoder_norms() — normalize_decoder()
    # keeps all W_dec column norms at exactly 1.0, making norms uninformative.
    scores = _compute_feature_layer_scores(crosscoder, prompt_store)
    # scores: (n_features, n_layers)
    n_features, n_layers = scores.shape

    lifetimes = np.zeros(n_features, dtype=np.int32)
    peak_layers = np.zeros(n_features, dtype=np.int32)
    max_scores = scores.max(axis=1)  # (n_features,)

    for f in range(n_features):
        if max_scores[f] < 1e-10:
            continue
        threshold = max_scores[f] * threshold_frac
        active = scores[f] > threshold

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
        peak_layers[f] = int(np.argmax(scores[f]))

    alive_mask = max_scores > 1e-10
    alive_lifetimes = lifetimes[alive_mask].astype(float)
    alive_indices = np.where(alive_mask)[0]

    # ------------------------------------------------------------------
    # Bimodality coefficient
    # BC = (skewness^2 + 1) / kurtosis
    # where kurtosis is the non-excess (Pearson) kurtosis = excess + 3.
    # BC > 5/9 ≈ 0.555 indicates bimodality.
    # Reference: SAS Institute (1990), also Pfister et al. (2013).
    # ------------------------------------------------------------------
    bimodality_coefficient = float("nan")
    bimodality_test = "insufficient_data"
    valley_threshold = None

    n_alive = len(alive_lifetimes)
    if n_alive >= 20:
        from scipy.stats import skew, kurtosis as scipy_kurtosis

        g1 = float(skew(alive_lifetimes))
        # scipy kurtosis defaults to excess (Fisher); add 3 for Pearson
        g2 = float(scipy_kurtosis(alive_lifetimes, fisher=True)) + 3.0

        if g2 > 1e-10:
            # BC is only meaningful when Pearson kurtosis is positive.
            # Negative g2 (platykurtic distributions) produces a negative BC
            # which is not interpretable as evidence of unimodality.
            bimodality_coefficient = (g1 ** 2 + 1.0) / g2
        else:
            bimodality_coefficient = float("nan")

        BC_THRESHOLD = 5.0 / 9.0  # ≈ 0.5556
        if not np.isnan(bimodality_coefficient):
            if bimodality_coefficient > BC_THRESHOLD:
                bimodality_test = "bimodal"
            else:
                bimodality_test = "unimodal"

    # ------------------------------------------------------------------
    # Valley threshold: minimum KDE density between the two largest peaks.
    # Only computed when bimodal.  Uses a fine grid over [min, max] lifetime.
    # If only one peak is found (unusual for a bimodal distribution), falls
    # back to the median as the split point.
    # ------------------------------------------------------------------
    if bimodality_test == "bimodal" and n_alive >= 20:
        try:
            lt_min, lt_max = alive_lifetimes.min(), alive_lifetimes.max()
            if lt_max > lt_min:
                kde = gaussian_kde(alive_lifetimes)
                # 500-point grid for sub-layer precision
                grid = np.linspace(lt_min, lt_max, 500)
                density = kde(grid)

                # Find all local maxima
                from scipy.signal import argrelextrema
                peak_idx = argrelextrema(density, np.greater, order=5)[0]

                if len(peak_idx) >= 2:
                    # Two largest peaks
                    top2 = peak_idx[np.argsort(density[peak_idx])[-2:]]
                    lo, hi = sorted(top2)
                    # Minimum density in the valley between them
                    valley_idx = lo + np.argmin(density[lo:hi + 1])
                    valley_threshold = int(round(float(grid[valley_idx])))
                else:
                    # KDE found only one peak despite BC > threshold.
                    # Fall back to median as split point.
                    valley_threshold = int(np.median(alive_lifetimes))
        except Exception:
            valley_threshold = int(np.median(alive_lifetimes))

    # ------------------------------------------------------------------
    # Feature classification
    # ------------------------------------------------------------------
    # Determine the split point for counting short vs long.
    # Priority: (1) KDE valley, (2) median for unimodal/fallback,
    # (3) legacy fixed thresholds for n_short_lived / n_long_lived counts
    # are retained in output for backward compatibility.
    if valley_threshold is not None:
        split = valley_threshold
    else:
        split = int(np.median(alive_lifetimes)) if n_alive > 0 else n_layers // 2

    lifetime_class = np.full(n_features, "dead", dtype=object)
    for idx in alive_indices:
        if lifetimes[idx] < split:
            lifetime_class[idx] = "short_lived"
        else:
            lifetime_class[idx] = "long_lived"

    short_lived_indices = [int(i) for i in alive_indices if lifetime_class[i] == "short_lived"]
    long_lived_indices  = [int(i) for i in alive_indices if lifetime_class[i] == "long_lived"]

    # Legacy fixed-threshold counts kept for backward compatibility with
    # the summary printer and any existing result parsers.
    n_short_legacy = int((alive_lifetimes <= 3).sum())
    n_long_legacy  = int((alive_lifetimes >= n_layers // 2).sum())

    return {
        # --- core per-feature arrays ---
        "lifetimes": lifetimes.tolist(),
        "peak_layers": peak_layers.tolist(),
        "feature_layer_scores": scores.tolist(),
        "max_scores": max_scores.tolist(),
        "lifetime_class": lifetime_class.tolist(),
        "short_lived_indices": short_lived_indices,
        "long_lived_indices": long_lived_indices,

        # --- distribution summary ---
        "n_alive": int(n_alive),
        "mean_lifetime": float(alive_lifetimes.mean()) if n_alive > 0 else 0.0,
        "median_lifetime": float(np.median(alive_lifetimes)) if n_alive > 0 else 0.0,

        # --- bimodality ---
        "bimodality_coefficient": float(bimodality_coefficient) if not np.isnan(bimodality_coefficient) else None,
        "bimodality_test": bimodality_test,
        "bc_threshold": 5.0 / 9.0,
        "valley_threshold": valley_threshold,
        "n_short_lived": len(short_lived_indices),
        "n_long_lived": len(long_lived_indices),

        # --- legacy fixed-threshold counts (backward compat) ---
        "n_short_lived_legacy": n_short_legacy,
        "n_long_lived_legacy": n_long_legacy,

        # --- deprecated field kept for backward compat ---
        "bimodal_score": 2 if bimodality_test == "bimodal" else 1,
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

    scores = _compute_feature_layer_scores(crosscoder, prompt_store)
    max_scores = scores.max(axis=1)

    alive = max_scores > 1e-10
    n_alive = int(alive.sum())

    multilayer_count = 0
    layers_active_per_feature = []
    for f in range(scores.shape[0]):
        if not alive[f]:
            continue
        threshold = max_scores[f] * threshold_frac
        n_active_layers = int((scores[f] > threshold).sum())
        layers_active_per_feature.append(n_active_layers)
        if n_active_layers >= min_layers:
            multilayer_count += 1

    return {
        "multilayer_fraction": multilayer_count / max(n_alive, 1),
        "multilayer_count": multilayer_count,
        "n_alive": n_alive,
        "layers_active_distribution": (
            np.histogram(layers_active_per_feature,
                         bins=range(1, scores.shape[1] + 2))[0].tolist()
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
    L, F, d = directions.shape

    # Gate on projector coverage before running the analysis.
    # Low-rank projectors (k << d) make the classifier blind: a random unit
    # vector projects k/d of its energy into each subspace regardless of true
    # alignment, so attract_dominance concentrates near 0.5 for all features.
    # Running the analysis in this regime produces uninformative output that
    # looks like a negative finding.  Return the diagnostic and stop.
    coverage = _projector_coverage(projectors, is_per_layer, d)
    if coverage["underpowered"]:
        print(f"    [v_subspace_alignment] UNDERPOWERED — skipping. "
              f"k_att={coverage['k_attract']}, k_rep={coverage['k_repulse']}, "
              f"d={d}, min_coverage={coverage['min_coverage']:.1%} "
              f"(need >=20%)")
        return {"underpowered": True, "coverage": coverage}

    # Weight per-layer contributions by where each feature actually activates.
    # decoder_norms() is all-ones after normalize_decoder(), so it gives uniform
    # weights — which for per-layer GPT-2 averages early repulsive layers with
    # late attractive layers and cancels real signal.
    scores = _compute_feature_layer_scores(crosscoder, prompt_store)  # (F, L)
    # Use scores as weights; fall back to uniform if all zeros (no prompt data)
    score_sum = scores.sum(axis=1, keepdims=True)  # (F, 1)
    weights = np.where(score_sum > 1e-10, scores / (score_sum + 1e-10),
                       np.ones_like(scores) / L)    # (F, L)

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

    # Aggregate: weighted by data-driven activation scores so that layers where
    # the feature actually fires contribute more than inactive layers.
    per_feature_attract = (attract_frac * weights).sum(axis=1)
    per_feature_repulse = (repulse_frac * weights).sum(axis=1)

    # Normalize to fraction of captured variance (attract + repulse may not sum
    # to 1.0 if there are complex eigenvalue pairs excluded from the projectors)
    total = per_feature_attract + per_feature_repulse + 1e-10
    attract_dominance = per_feature_attract / total  # (F,) in [0, 1]

    # Classify by relative dominance, not absolute threshold.
    # The absolute threshold (e.g. 0.6) was designed for low-rank projectors.
    # When rep_frac ≈ 0.5 (e.g. ALBERT-xlarge = 0.569), a random unit vector
    # projects ~0.569 into the repulsive subspace — always below 0.6 — so
    # everything would be "mixed" regardless of true alignment.
    #
    # margin=0.1 means a feature must be 60/40 toward one subspace to be
    # classified as that type. Adjust via config["v_alignment_margin"].
    margin = config.get("v_alignment_margin", 0.1)
    dominant = np.full(F, "mixed", dtype=object)
    dominant[attract_dominance > 0.5 + margin] = "attractive"
    dominant[attract_dominance < 0.5 - margin] = "repulsive"

    return {
        "per_feature_attract": per_feature_attract.tolist(),
        "per_feature_repulse": per_feature_repulse.tolist(),
        "attract_dominance": attract_dominance.tolist(),
        "per_feature_dominant": dominant.tolist(),
        "attract_frac_per_layer": attract_frac.tolist(),
        "repulse_frac_per_layer": repulse_frac.tolist(),
        "n_attractive": int((dominant == "attractive").sum()),
        "n_repulsive": int((dominant == "repulsive").sum()),
        "n_mixed": int((dominant == "mixed").sum()),
        "mean_attract_dominance": float(attract_dominance.mean()),
        "std_attract_dominance": float(attract_dominance.std()),
    }


# ---------------------------------------------------------------------------
# Analysis: Plateau clustering (Step 5)
# ---------------------------------------------------------------------------

@register("plateau_clustering")
def plateau_clustering(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Compute two-resolution cluster labels at mid-plateau layers.

    For each eval prompt, at each mid-plateau layer identified by Phase 1:

    1. Spectral k=2 bipartition — dominant semantic split, well-powered even
       on short prompts (50-300 tokens).  Uses the Fiedler vector of the
       cosine-similarity graph.

    2. Within-partition HDBSCAN — finer subcluster structure within each
       half of the bipartition.  Cluster IDs are globally unique (partition 0
       clusters come first, then partition 1).  Noise tokens get label -1.

    Requires artifacts:
      "plateau_layers"  : {prompt_key: [mid_layer_num, ...]}  (from Phase 1)
      "layer_indices"   : [int, ...]  (crosscoder-to-model layer mapping)

    Returns the full _compute_plateau_clusters output dict, plus a summary
    of cluster sizes per prompt × layer.

    The result is also available to cluster_identity, which calls
    _compute_plateau_clusters directly when plateau_layers is present
    rather than requiring this analysis to run first.
    """
    plateau_layers = artifacts.get("plateau_layers")
    layer_indices  = artifacts.get("layer_indices")

    if plateau_layers is None:
        return {"error": "plateau_layers not in artifacts — run Phase 1 for this model first"}
    if layer_indices is None:
        return {"error": "layer_indices not in artifacts"}

    min_cluster_size = config.get("plateau_min_cluster_size", 3)
    clusters = _compute_plateau_clusters(
        prompt_store, layer_indices, plateau_layers, min_cluster_size
    )

    # Build a human-readable summary alongside the full label data
    summary = {}
    for pk, layer_dict in clusters.items():
        summary[pk] = {}
        for mid_layer, info in layer_dict.items():
            spec = np.array(info["spectral"])
            hdb  = np.array(info["hdbscan"])
            spec_sizes = {int(c): int((spec == c).sum()) for c in (0, 1)}
            hdb_sizes  = {int(c): int((hdb == c).sum())
                          for c in sorted(set(hdb.tolist())) if c != -1}
            summary[pk][mid_layer] = {
                "actual_model_layer":  info["actual_model_layer"],
                "distance_to_target":  info["distance_to_target"],
                "n_tokens":            len(spec),
                "spectral_sizes":      spec_sizes,
                "n_hdbscan_clusters":  info["n_hdbscan_clusters"],
                "hdbscan_sizes":       hdb_sizes,
                "n_noise":             int((hdb == -1).sum()),
            }

    return {
        "clusters":  clusters,
        "summary":   summary,
        "n_prompts": len(clusters),
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
    Match crosscoder features to cluster labels at mid-plateau layers.

    Label source priority:
      1. Fresh spectral k=2 bipartition at mid-plateau layers, computed here
         from prompt_store activations.  Used when "plateau_layers" and
         "layer_indices" are in artifacts (i.e. Phase 1 was run for this
         model).  The dominant bipartition is well-powered on short prompts
         and directly interpretable as the main semantic split.

      2. Phase 1 HDBSCAN labels ("hdbscan_labels" artifact) — used as
         fallback when plateau_layers is absent.

      3. Neither available → returns an error.

    The selectivity test is unchanged in both cases:
      recall    >= selectivity_high (default 0.7)
      false positive rate <= selectivity_low  (default 0.1)

    When using plateau clusters, "layer_key" in each identity_feature entry
    is the mid-plateau model layer number (int), and "clustering_source" is
    "plateau_spectral" or "plateau_hdbscan" depending on config
    ["cluster_identity_source"] (default "spectral").

    When using Phase 1 HDBSCAN labels, "layer_key" is the original key from
    the artifact and "clustering_source" is "phase1_hdbscan".
    """
    plateau_layers = artifacts.get("plateau_layers")
    layer_indices  = artifacts.get("layer_indices")
    hdbscan_labels = artifacts.get("hdbscan_labels")

    use_plateau = (plateau_layers is not None and layer_indices is not None)
    use_phase1  = (hdbscan_labels is not None)

    if not use_plateau and not use_phase1:
        return {
            "error": (
                "Neither plateau_layers nor hdbscan_labels in artifacts. "
                "Run Phase 1 for this model to enable cluster_identity."
            )
        }

    selectivity_high = config.get("cluster_selectivity_high", 0.7)
    selectivity_low  = config.get("cluster_selectivity_low",  0.1)
    # "spectral" uses the k=2 bipartition; "hdbscan" uses finer within-partition
    # subclusters.  Spectral is strongly preferred: more robust at small n_tokens.
    cluster_source = config.get("cluster_identity_source", "spectral")
    min_cluster_size = config.get("plateau_min_cluster_size", 3)

    # --- Build the label dict used for matching ---
    # Structure: {prompt_key: {layer_key: np.ndarray(n_tokens,)}}
    label_dict: dict = {}
    clustering_source_name: str

    if use_plateau:
        clustering_source_name = f"plateau_{cluster_source}"
        plateau_clusters = _compute_plateau_clusters(
            prompt_store, layer_indices, plateau_layers, min_cluster_size
        )
        for pk, layer_map in plateau_clusters.items():
            label_dict[pk] = {}
            for mid_layer, info in layer_map.items():
                label_dict[pk][mid_layer] = np.array(info[cluster_source])
    else:
        clustering_source_name = "phase1_hdbscan"
        for pk, layer_map in hdbscan_labels.items():
            label_dict[pk] = {lk: np.array(v) for lk, v in layer_map.items()}

    # --- Selectivity test ---
    results_per_prompt = {}

    for prompt_key in prompt_store.keys():
        if prompt_key not in label_dict:
            continue

        out = _run_crosscoder_on_prompt(crosscoder, prompt_store, prompt_key)
        z      = out["z"].numpy()   # (n_tokens, n_features)
        active = z > 0

        identity_features = []

        for layer_key, labels in label_dict[prompt_key].items():
            cluster_ids = sorted(set(labels.tolist()) - {-1})

            for feat_idx in range(z.shape[1]):
                if not active[:, feat_idx].any():
                    continue

                feat_active = active[:, feat_idx]

                for cid in cluster_ids:
                    in_cluster  = labels == cid
                    n_cluster   = int(in_cluster.sum())
                    n_outside   = int((~in_cluster & (labels != -1)).sum())

                    if n_cluster == 0:
                        continue

                    recall = float(feat_active[in_cluster].sum() / n_cluster)
                    false_positive = (
                        float(feat_active[~in_cluster & (labels != -1)].sum() / n_outside)
                        if n_outside > 0 else 0.0
                    )

                    if recall >= selectivity_high and false_positive <= selectivity_low:
                        identity_features.append({
                            "feature":            int(feat_idx),
                            "cluster_id":         int(cid),
                            "layer_key":          int(layer_key) if isinstance(layer_key, (int, np.integer)) else str(layer_key),
                            "recall":             recall,
                            "false_positive_rate": false_positive,
                            "cluster_size":       n_cluster,
                            "clustering_source":  clustering_source_name,
                        })

        results_per_prompt[prompt_key] = {
            "n_identity_features": len(identity_features),
            "identity_features":   identity_features,
            "clustering_source":   clustering_source_name,
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

        # Feature activity at violation vs non-violation layers.
        # Use data-driven layer scores rather than decoder_norms() —
        # normalize_decoder() keeps all W_dec column norms at exactly 1.0,
        # so decoder_norms() is a constant (F, L) matrix of ones and produces
        # zero z-scores for every feature.
        scores = _compute_feature_layer_scores(crosscoder, prompt_store)  # (F, L)
        violation_scores     = scores[:, sampled_violation].mean(axis=1)
        non_violation_scores = scores[:, sampled_non_violation].mean(axis=1)

        pop_std = non_violation_scores.std()
        if pop_std < 1e-10:
            results_per_prompt[prompt_key] = {"n_violation_features": 0}
            continue

        z_scores = (violation_scores - non_violation_scores) / (pop_std + 1e-10)

        violation_features = []
        for f in range(len(z_scores)):
            if z_scores[f] > z_threshold:
                violation_features.append({
                    "feature": int(f),
                    "z_score": float(z_scores[f]),
                    "violation_score":     float(violation_scores[f]),
                    "non_violation_score": float(non_violation_scores[f]),
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
    scores = _compute_feature_layer_scores(crosscoder, prompt_store)
    max_scores = scores.max(axis=1)

    # Identify long-lived features (active at 50%+ of layers)
    threshold_frac = config.get("lifetime_threshold_frac", 0.1)
    long_lived = []
    for f in range(scores.shape[0]):
        if max_scores[f] < 1e-10:
            continue
        threshold = max_scores[f] * threshold_frac
        n_active = int((scores[f] > threshold).sum())
        if n_active >= scores.shape[1] // 2:
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

        if np.std(all_acts) < 1e-10:
            continue  # feature never varies — not informative for position correlation

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
# P2 → P3: FFN-repulsive feature alignment
# ---------------------------------------------------------------------------

@register("ffn_repulsive_feature_alignment")
def ffn_repulsive_feature_alignment(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Test whether violation-layer features have decoder directions aligned
    with the FFN update direction at those layers.

    Closes the loop: V shapes geometry → FFN executes → feature captures.

    Requires artifacts:
      "ffn_subspace"            : dict from Phase 2 ffn_subspace analysis
      "layer_indices"           : list[int]
      "phase2_dir"              : Path to Phase 2 run (with ffn_deltas_raw.npz)

    Also requires violation_layer_features to have been run first (uses
    its output from the registry if available, otherwise runs it).
    """
    ffn_sub = artifacts.get("ffn_subspace")
    layer_indices = artifacts.get("layer_indices")
    phase2_dir = artifacts.get("phase2_dir")

    if not ffn_sub or not layer_indices or not phase2_dir:
        return {"error": "Requires ffn_subspace, layer_indices, and phase2_dir artifacts"}

    phase2_dir = Path(phase2_dir)
    ffn_path = phase2_dir / "ffn_deltas_raw.npz"
    if not ffn_path.exists():
        return {"error": f"FFN deltas not found at {ffn_path}"}

    W_dec = crosscoder.W_dec.detach().cpu().float().numpy()  # (L_cc, F, d)
    ffn_deltas = np.load(ffn_path)["arr_0"]  # (n_model_layers, n_tokens, d)

    per_violation = ffn_sub.get("per_violation", [])
    if not per_violation:
        return {"error": "No per-violation results in ffn_subspace"}

    # Get violation-layer feature indices — run the analysis if needed
    vlf = artifacts.get("_violation_layer_features_result")
    if vlf is None:
        if "violation_layer_features" in _REGISTRY:
            vlf = _REGISTRY["violation_layer_features"](
                crosscoder, prompt_store, artifacts, config
            )
        else:
            return {"error": "violation_layer_features not available"}

    results = []
    for viol in per_violation:
        v_layer = viol.get("layer")
        ffn_role = viol.get("ffn_role", viol.get("classification", ""))

        if v_layer not in layer_indices or v_layer >= ffn_deltas.shape[0]:
            continue
        cc_idx = layer_indices.index(v_layer)

        # Mean FFN delta direction at this layer
        ffn_dir = ffn_deltas[v_layer].mean(axis=0)
        ffn_norm = np.linalg.norm(ffn_dir)
        if ffn_norm < 1e-10:
            continue
        ffn_dir = ffn_dir / ffn_norm

        # Collect violation feature indices across prompts
        vl_features = set()
        for pk, pdata in vlf.items():
            if not isinstance(pdata, dict):
                continue
            for vf in pdata.get("violation_features", []):
                vl = vf.get("layer") or vf.get("violation_layer")
                if vl == v_layer:
                    vl_features.update(
                        vf.get("feature_indices", vf.get("features", []))
                    )
        vl_features = sorted(vl_features)
        if not vl_features:
            continue

        # Cosine of each feature's decoder direction with FFN direction
        dec_at_layer = W_dec[cc_idx]
        cosines = []
        for fidx in vl_features:
            if fidx >= dec_at_layer.shape[0]:
                continue
            d_vec = dec_at_layer[fidx]
            d_norm = np.linalg.norm(d_vec)
            if d_norm < 1e-10:
                continue
            cosines.append({
                "feature": int(fidx),
                "cosine_with_ffn": float(np.dot(d_vec / d_norm, ffn_dir)),
            })

        if cosines:
            cos_vals = [c["cosine_with_ffn"] for c in cosines]
            results.append({
                "violation_layer": v_layer,
                "ffn_role": ffn_role,
                "n_features_tested": len(cosines),
                "mean_abs_cosine": float(np.mean(np.abs(cos_vals))),
                "mean_cosine": float(np.mean(cos_vals)),
                "fraction_aligned_030": float(np.mean([abs(c) > 0.3 for c in cos_vals])),
                "per_feature": cosines,
            })

    return {
        "n_violations_tested": len(results),
        "per_violation": results,
        "mean_alignment": float(np.mean([r["mean_abs_cosine"] for r in results]))
        if results else None,
    }


# ---------------------------------------------------------------------------
# P2 → P3: Cross-term feature weighting
# ---------------------------------------------------------------------------

@register("cross_term_feature_weighting")
def cross_term_feature_weighting(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Test whether high-F features from feature_cluster_correlation are the
    ones whose activations differ most on cross-term-dominant token pairs.

    Requires artifacts:
      "cross_term_results"         : dict from Phase 2 cross_term_analysis
      "feature_cluster_correlation": dict (or will be computed)
    """
    ct = artifacts.get("cross_term_results")
    if not ct:
        return {"error": "Requires cross_term_results artifact from Phase 2"}

    per_violation = ct.get("per_violation", [])
    ct_pairs = []
    for v in per_violation:
        if v.get("cross_dominant"):
            for p in v.get("top_pairs", []):
                ct_pairs.append((p["i"], p["j"]))
    ct_pairs = list(set(ct_pairs))
    if not ct_pairs:
        return {"error": "No cross-term dominant pairs"}

    # Get or compute feature_cluster_correlation
    fcc = artifacts.get("_fcc_result")
    if fcc is None and "feature_cluster_correlation" in _REGISTRY:
        fcc = _REGISTRY["feature_cluster_correlation"](
            crosscoder, prompt_store, artifacts, config
        )

    results_per_prompt = {}
    for prompt_key in prompt_store.keys():
        out = _run_crosscoder_on_prompt(crosscoder, prompt_store, prompt_key)
        z = out["z"].numpy()
        n_tokens = z.shape[0]

        valid_pairs = [(i, j) for i, j in ct_pairs if i < n_tokens and j < n_tokens]
        if not valid_pairs:
            continue

        ct_token_set = set()
        for i, j in valid_pairs:
            ct_token_set.add(i)
            ct_token_set.add(j)
        ct_tokens = sorted(ct_token_set)
        non_ct = [t for t in range(n_tokens) if t not in ct_token_set]
        if not non_ct:
            continue

        activation_diff = z[ct_tokens].mean(axis=0) - z[non_ct].mean(axis=0)

        # F-stats from FCC
        fcc_prompt = (fcc or {}).get(prompt_key, {})
        f_stats = fcc_prompt.get("f_stats")
        if f_stats is None:
            continue
        f_stats = np.array(f_stats)

        n_f = min(len(f_stats), len(activation_diff))
        if n_f < 5:
            continue

        rho, p_val = stats.spearmanr(f_stats[:n_f], activation_diff[:n_f])
        results_per_prompt[prompt_key] = {
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "n_features": n_f,
            "n_ct_pairs": len(valid_pairs),
        }

    return {
        "per_prompt": results_per_prompt,
        "mean_rho": float(np.mean([r["spearman_rho"] for r in results_per_prompt.values()]))
        if results_per_prompt else None,
    }


# ---------------------------------------------------------------------------
# P1 → P3: Induction feature tagging
# ---------------------------------------------------------------------------

@register("induction_feature_tagging")
def induction_feature_tagging(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Tag features as "induction-correlated" vs "semantic-correlated" using
    Phase 1's per-pair HDBSCAN agreement flags.

    Mutual-NN pairs in the same HDBSCAN cluster = "semantic".
    Mutual-NN pairs in different clusters = "artifact" (induction candidates).
    Features firing preferentially on artifact-pair tokens are tagged induction.

    Requires artifacts:
      "pair_agreement" : dict with per-layer {semantic: [...], artifact: [...]}
    """
    pair_data = artifacts.get("pair_agreement")
    if not pair_data:
        return {"error": "Requires pair_agreement artifact from Phase 1"}

    # Aggregate across layers: collect all semantic and artifact token indices
    sem_tokens = set()
    art_tokens = set()
    for layer_key, layer_pa in pair_data.items():
        if not isinstance(layer_pa, dict):
            continue
        for pair in layer_pa.get("semantic", []):
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                sem_tokens.add(pair[0])
                sem_tokens.add(pair[1])
        for pair in layer_pa.get("artifact", []):
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                art_tokens.add(pair[0])
                art_tokens.add(pair[1])

    sem_only = sem_tokens - art_tokens
    art_only = art_tokens - sem_tokens

    if len(sem_only) < 2 or len(art_only) < 2:
        return {"error": f"Insufficient exclusive tokens: {len(sem_only)} sem, {len(art_only)} art"}

    feature_ratios = {}  # feature_idx -> [artifact_ratio_per_prompt]
    for prompt_key in prompt_store.keys():
        out = _run_crosscoder_on_prompt(crosscoder, prompt_store, prompt_key)
        z = out["z"].numpy()
        n_tokens = z.shape[0]

        sem_idx = sorted(t for t in sem_only if t < n_tokens)
        art_idx = sorted(t for t in art_only if t < n_tokens)
        if len(sem_idx) < 2 or len(art_idx) < 2:
            continue

        sem_mean = z[sem_idx].mean(axis=0)
        art_mean = z[art_idx].mean(axis=0)

        for f in range(z.shape[1]):
            total = sem_mean[f] + art_mean[f]
            if total < 1e-10:
                continue
            ratio = float(art_mean[f] / total)
            feature_ratios.setdefault(f, []).append(ratio)

    tagged = {}
    for f, ratios in feature_ratios.items():
        mean_r = float(np.mean(ratios))
        tag = "induction" if mean_r > 0.6 else ("semantic" if mean_r < 0.4 else "ambiguous")
        tagged[f] = {"tag": tag, "mean_artifact_ratio": mean_r}

    n_ind = sum(1 for v in tagged.values() if v["tag"] == "induction")
    n_sem = sum(1 for v in tagged.values() if v["tag"] == "semantic")
    n_amb = sum(1 for v in tagged.values() if v["tag"] == "ambiguous")

    return {
        "n_tagged": len(tagged),
        "n_induction": n_ind,
        "n_semantic": n_sem,
        "n_ambiguous": n_amb,
        "per_feature": tagged,
    }


# ---------------------------------------------------------------------------
# P3 → P2: Decoder directions as violation probes
# ---------------------------------------------------------------------------

@register("decoder_violation_projection")
def decoder_violation_projection(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Project Phase 2 displacement vectors at violation layers onto the
    decoder directions of violation-layer features. If a small number of
    feature directions explain most displacement, violations are
    interpretable through the crosscoder.

    Requires artifacts:
      "phase2_dir"    : Path with trajectory.npz
      "layer_indices" : list[int]
    """
    phase2_dir = artifacts.get("phase2_dir")
    layer_indices = artifacts.get("layer_indices")
    if not phase2_dir or not layer_indices:
        return {"error": "Requires phase2_dir and layer_indices"}

    phase2_dir = Path(phase2_dir)
    traj_path = phase2_dir / "trajectory.npz"
    if not traj_path.exists():
        # Try activations from Phase 1
        p1_dir = artifacts.get("phase1_dir")
        if p1_dir:
            traj_path = Path(p1_dir) / "activations.npz"
        if not traj_path.exists():
            return {"error": "No trajectory data found"}

    traj_data = np.load(traj_path)
    # Handle different key conventions
    traj_key = "activations" if "activations" in traj_data else "arr_0"
    traj = traj_data[traj_key]  # (n_layers, n_tokens, d)

    W_dec = crosscoder.W_dec.detach().cpu().float().numpy()

    # Get violation-layer features
    vlf = artifacts.get("_violation_layer_features_result")
    if vlf is None and "violation_layer_features" in _REGISTRY:
        vlf = _REGISTRY["violation_layer_features"](
            crosscoder, prompt_store, artifacts, config
        )
    if not vlf or "error" in vlf:
        return {"error": "violation_layer_features not available"}

    results = []
    for pk, pdata in vlf.items():
        if not isinstance(pdata, dict):
            continue
        for vf_info in pdata.get("violation_features", []):
            v_layer = vf_info.get("layer") or vf_info.get("violation_layer")
            if v_layer is None or v_layer not in layer_indices:
                continue
            if v_layer < 1 or v_layer >= traj.shape[0]:
                continue
            cc_idx = layer_indices.index(v_layer)

            feat_idx = vf_info.get("feature_indices", vf_info.get("features", []))
            if not feat_idx:
                continue

            displacement = traj[v_layer] - traj[v_layer - 1]  # (T, d)
            disp_mean = displacement.mean(axis=0)
            disp_norm_sq = np.dot(disp_mean, disp_mean)
            if disp_norm_sq < 1e-20:
                continue

            dec_dirs = W_dec[cc_idx, feat_idx, :]  # (K, d)
            projections = dec_dirs @ disp_mean
            recon = projections[:, None] * dec_dirs
            explained = float(np.dot(recon.sum(axis=0), disp_mean) / disp_norm_sq)
            explained = min(max(explained, 0.0), 1.0)

            results.append({
                "prompt_key": pk,
                "violation_layer": v_layer,
                "n_features": len(feat_idx),
                "explained_variance": explained,
                "top_projections": sorted(
                    [{"feature": int(f), "projection": float(p)}
                     for f, p in zip(feat_idx, projections)],
                    key=lambda x: abs(x["projection"]),
                    reverse=True
                )[:5],
            })

    return {
        "per_violation": results,
        "mean_explained": float(np.mean([r["explained_variance"] for r in results]))
        if results else None,
    }


# ---------------------------------------------------------------------------
# Phase 4 completion: co-activation at merge layers
# ---------------------------------------------------------------------------

@register("coactivation_at_merges")
def coactivation_at_merges(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Feature co-activation matrix at merge layers. Identifies coordinated
    reorganization events — sets of features changing simultaneously.

    Requires artifacts:
      "merge_layers"  : list[int] — layers where cluster count drops
      "layer_indices"  : list[int]
    """
    merge_layers_raw = artifacts.get("merge_layers")
    layer_indices = artifacts.get("layer_indices", [])
    if not merge_layers_raw:
        return {"error": "Requires merge_layers artifact from Phase 1"}

    # Normalize: merge_layers can be a list or dict
    if isinstance(merge_layers_raw, dict):
        all_merges = set()
        for v in merge_layers_raw.values():
            if isinstance(v, list):
                all_merges.update(v)
        merge_layers = sorted(all_merges)
    else:
        merge_layers = list(merge_layers_raw)

    top_k = config.get("coactivation_top_k", 50)
    results = {}

    for prompt_key in prompt_store.keys():
        out = _run_crosscoder_on_prompt(crosscoder, prompt_store, prompt_key)
        z = out["z"].numpy()
        active = (z > 0).astype(float)

        prompt_results = []
        for m_layer in merge_layers:
            if m_layer not in layer_indices:
                continue

            feat_scores = active.sum(axis=0)
            top_feats = np.argsort(feat_scores)[-top_k:]

            sub = active[:, top_feats]
            coact = (sub.T @ sub) / max(sub.shape[0], 1)
            np.fill_diagonal(coact, 0)

            n = coact.shape[0]
            coord_score = float(coact.sum() / max(n * (n - 1), 1))

            # Connected components via thresholding
            high = coact > 0.3
            groups = _connected_components_bfs(high)
            multi_groups = [g for g in groups if len(g) > 1]

            prompt_results.append({
                "merge_layer": m_layer,
                "coordination_score": coord_score,
                "n_coordinated_groups": len(multi_groups),
                "largest_group_size": max((len(g) for g in groups), default=0),
                "top_features": top_feats.tolist(),
            })

        if prompt_results:
            results[prompt_key] = prompt_results

    return results


def _connected_components_bfs(adj: np.ndarray) -> list:
    """BFS connected components on a boolean adjacency matrix."""
    n = adj.shape[0]
    visited = set()
    components = []
    for start in range(n):
        if start in visited:
            continue
        comp = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            for nb in range(n):
                if adj[node, nb] and nb not in visited:
                    queue.append(nb)
        components.append(comp)
    return components


# ---------------------------------------------------------------------------
# Phase 5 completion: cluster identity diff across merge
# ---------------------------------------------------------------------------

@register("cluster_identity_diff")
def cluster_identity_diff(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    At each merge event, compare cluster-selective feature sets at
    pre-merge and post-merge layers. Which features die, which activate?

    Requires artifacts:
      "merge_layers"    : list[int] or dict
      "hdbscan_labels"  : dict[str(layer) -> list[int]]
      "layer_indices"   : list[int]
    """
    merge_layers_raw = artifacts.get("merge_layers")
    hdb_labels = artifacts.get("hdbscan_labels")
    layer_indices = artifacts.get("layer_indices", [])

    if not merge_layers_raw or not hdb_labels:
        return {"error": "Requires merge_layers and hdbscan_labels from Phase 1"}

    if isinstance(merge_layers_raw, dict):
        all_merges = set()
        for v in merge_layers_raw.values():
            if isinstance(v, list):
                all_merges.update(v)
        merge_layers = sorted(all_merges)
    else:
        merge_layers = list(merge_layers_raw)

    f_threshold = config.get("cluster_selective_f_threshold", 3.0)
    results = {}

    for prompt_key in prompt_store.keys():
        out = _run_crosscoder_on_prompt(crosscoder, prompt_store, prompt_key)
        z = out["z"].numpy()

        prompt_results = []
        for m_layer in merge_layers:
            # Find nearest sampled pre/post layers
            pre_candidates = [l for l in layer_indices if l < m_layer]
            post_candidates = [l for l in layer_indices if l >= m_layer]
            if not pre_candidates or not post_candidates:
                continue

            pre_layer = max(pre_candidates)
            post_layer = min(post_candidates)

            pre_labels = hdb_labels.get(str(pre_layer))
            post_labels = hdb_labels.get(str(post_layer))
            if pre_labels is None or post_labels is None:
                continue

            pre_labels = np.array(pre_labels)
            post_labels = np.array(post_labels)

            pre_feats = _cluster_selective_set(z, pre_labels, f_threshold)
            post_feats = _cluster_selective_set(z, post_labels, f_threshold)

            deaths = pre_feats - post_feats
            births = post_feats - pre_feats
            persistent = pre_feats & post_feats

            prompt_results.append({
                "merge_layer": m_layer,
                "pre_layer": pre_layer,
                "post_layer": post_layer,
                "n_pre_features": len(pre_feats),
                "n_post_features": len(post_feats),
                "n_deaths": len(deaths),
                "n_births": len(births),
                "n_persistent": len(persistent),
                "death_indices": sorted(deaths)[:20],
                "birth_indices": sorted(births)[:20],
            })

        if prompt_results:
            results[prompt_key] = prompt_results

    return results


# ---------------------------------------------------------------------------
# P3 → P2: Lifetime class into centroid decomposition
# ---------------------------------------------------------------------------

@register("lifetime_centroid_decomposition")
def lifetime_centroid_decomposition(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Decompose Phase 1 centroid movement into short-lived vs long-lived
    feature directions. Tests: do centroids move in long-lived directions
    during plateaus and short-lived directions at merge events?

    Requires artifacts:
      "phase1_dir"    : Path with centroid_trajectories.npz
      "layer_indices" : list[int]
    Also needs feature_lifetimes to have run (uses lifetime_class).
    """
    phase1_dir = artifacts.get("phase1_dir")
    layer_indices = artifacts.get("layer_indices", [])
    if not phase1_dir:
        return {"error": "Requires phase1_dir artifact"}

    centroid_path = Path(phase1_dir) / "centroid_trajectories.npz"
    if not centroid_path.exists():
        return {"error": f"Centroid trajectories not found at {centroid_path}"}

    # Get lifetime classification
    lt_result = artifacts.get("_lifetime_result")
    if lt_result is None and "feature_lifetimes" in _REGISTRY:
        lt_result = _REGISTRY["feature_lifetimes"](
            crosscoder, prompt_store, artifacts, config
        )
    if not lt_result or "error" in lt_result:
        return {"error": "feature_lifetimes not available"}

    lt_class = lt_result.get("lifetime_class", [])
    short_idx = [i for i, c in enumerate(lt_class) if c == "short_lived"]
    long_idx = [i for i, c in enumerate(lt_class) if c == "long_lived"]
    if not short_idx or not long_idx:
        return {"error": "Need both short and long lived features"}

    W_dec = crosscoder.W_dec.detach().cpu().float().numpy()
    centroids = np.load(centroid_path, allow_pickle=True)

    results = []
    for cc_i in range(len(layer_indices) - 1):
        model_layer = layer_indices[cc_i]

        k = min(20, len(short_idx), len(long_idx))
        if k == 0:
            continue

        short_dirs = W_dec[cc_i, short_idx[:k*3], :]
        long_dirs = W_dec[cc_i, long_idx[:k*3], :]

        U_s = np.linalg.svd(short_dirs.T, full_matrices=False)[0][:, :k]
        U_l = np.linalg.svd(long_dirs.T, full_matrices=False)[0][:, :k]

        # Load centroids — format: trajectory_N keys, each (lifespan, d)
        # We need centroid positions at model_layer
        # Try to reconstruct from the trajectory arrays
        centroid_vecs_curr = []
        centroid_vecs_next = []
        for key in centroids.files:
            traj_arr = centroids[key]  # (lifespan, d)
            # We don't have layer mapping for centroid trajectories,
            # so use cc_i and cc_i+1 as indices if they fit
            if cc_i < traj_arr.shape[0] and cc_i + 1 < traj_arr.shape[0]:
                centroid_vecs_curr.append(traj_arr[cc_i])
                centroid_vecs_next.append(traj_arr[cc_i + 1])

        if not centroid_vecs_curr:
            continue

        # Mean centroid displacement
        c_curr = np.stack(centroid_vecs_curr).mean(axis=0)
        c_next = np.stack(centroid_vecs_next).mean(axis=0)
        delta = c_next - c_curr
        delta_norm_sq = np.dot(delta, delta)
        if delta_norm_sq < 1e-20:
            continue

        proj_short = float(np.sum((U_s.T @ delta) ** 2) / delta_norm_sq)
        proj_long = float(np.sum((U_l.T @ delta) ** 2) / delta_norm_sq)

        results.append({
            "model_layer": model_layer,
            "next_layer": layer_indices[cc_i + 1],
            "frac_in_short_lived": proj_short,
            "frac_in_long_lived": proj_long,
            "frac_orthogonal": max(0.0, 1.0 - proj_short - proj_long),
        })

    return {"per_layer": results}


def _cluster_selective_set(
    z: np.ndarray,
    labels: np.ndarray,
    f_threshold: float,
) -> set:
    """Feature indices with F-stat > threshold for cluster discrimination."""
    unique = set(labels) - {-1}
    if len(unique) < 2:
        return set()

    n_tok = min(z.shape[0], len(labels))
    z = z[:n_tok]
    labels = labels[:n_tok]

    selective = set()
    for f in range(z.shape[1]):
        groups = [z[labels == c, f] for c in unique if (labels == c).sum() > 0]
        if len(groups) < 2:
            continue
        try:
            f_stat, _ = stats.f_oneway(*groups)
            if np.isfinite(f_stat) and f_stat > f_threshold:
                selective.add(f)
        except Exception:
            continue
    return selective


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

    # Gate on projector coverage.  v_subspace_alignment returns
    # {"underpowered": True, "coverage": ...} when projectors are low-rank.
    # Propagate that here so the summary printer can display the diagnostic
    # rather than a spurious ρ value.
    va_result = v_subspace_alignment(crosscoder, prompt_store, artifacts, config)
    if va_result.get("underpowered"):
        return {
            "underpowered": True,
            "coverage": va_result["coverage"],
            "note": (
                "lifetime_vs_alignment not run: v_subspace_alignment is "
                "underpowered at this projector rank. "
                "rho=-0.028 (or any value) from a previous run is uninformative, "
                "not a negative finding. Rerun with full-rank projectors "
                f"(k >= {va_result['coverage']['min_coverage_threshold'] * va_result['coverage']['d_model']:.0f})."
            ),
        }

    # Compute lifetimes
    lt_result = feature_lifetimes(crosscoder, prompt_store, artifacts, config)
    lifetimes = np.array(lt_result["lifetimes"])

    # Alignment already computed above.
    attract = np.array(va_result["attract_dominance"])

    # Filter to alive features
    alive = np.array(lt_result["max_scores"]) > 1e-10
    if alive.sum() < 10:
        return {"error": "fewer than 10 alive features"}

    lt_alive = lifetimes[alive]
    att_alive = attract[alive]

    if np.std(lt_alive) < 1e-10:
        return {
            "error": "lifetime array is constant — all features span the same number of layers. "
                     "Check that the model trained long enough or increase prompt diversity.",
            "lifetime_variance": float(np.var(lt_alive)),
        }
    if np.std(att_alive) < 1e-10:
        return {
            "error": "alignment array is constant — V projectors may not be loaded correctly.",
            "alignment_variance": float(np.var(att_alive)),
        }

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
# Analysis: Feature-cluster F-statistic correlation (Step 6)
# ---------------------------------------------------------------------------

@register("feature_cluster_correlation")
def feature_cluster_correlation(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Rank all features by discriminative power over cluster membership at
    mid-plateau layers, using the one-way ANOVA F-statistic.

    This is a bottom-up approach: rather than starting from cluster centroids
    and asking which features are active, we ask which features have activation
    distributions that differ across clusters.  This surfaces weak-but-consistent
    signals that the top-down approach misses.

    The test is applied at two clustering resolutions (when plateau_layers is
    available):

      spectral : k=2 bipartition — one F-statistic per feature, testing
                 whether mean activation differs between the two halves.  With
                 only two groups this reduces to a two-sample t-test (F = t²),
                 but framing it as F keeps the interface consistent with the
                 HDBSCAN case.

      hdbscan  : fine-grained subclusters within each partition — one
                 F-statistic per feature across all non-noise cluster groups.

    For each (prompt, mid_layer, clustering_resolution) combination, features
    are ranked by F-statistic descending.  The top_k highest-F features are
    returned with their statistics.

    Requires artifacts:
      "plateau_layers"  : {prompt_key: [mid_layer_num, ...]}
      "layer_indices"   : [int, ...]

    Falls back to Phase 1 hdbscan_labels if plateau_layers is absent, with the
    same F-statistic test across HDBSCAN cluster groups.

    Parameters (via config)
    -----------------------
    fcc_top_k          : int   — features to return per combo  (default 20)
    fcc_min_f          : float — minimum F to include in output (default 1.0)
    fcc_min_group_size : int   — skip clusters with fewer tokens (default 2)

    Returns
    -------
    {
      prompt_key: {
        mid_layer (or layer_key): {
          "spectral": {
            "top_features": [
              {
                "feature"         : int,
                "f_stat"          : float,
                "eta_squared"     : float,   # effect size F/(F + df_within)
                "mean_by_cluster" : [float, float],
                "lifetime_class"  : str,     # from feature_lifetimes if available
              }, ...
            ],
            "n_features_tested"   : int,
            "n_features_above_min": int,
          },
          "hdbscan": { ... same structure ... },
        }
      },
      "overall": {
        "n_prompts_run"   : int,
        "clustering_source": str,
      }
    }
    """
    from scipy.stats import f_oneway

    plateau_layers = artifacts.get("plateau_layers")
    layer_indices  = artifacts.get("layer_indices")
    hdbscan_labels = artifacts.get("hdbscan_labels")

    use_plateau = plateau_layers is not None and layer_indices is not None
    use_phase1  = hdbscan_labels is not None

    if not use_plateau and not use_phase1:
        return {
            "error": (
                "Neither plateau_layers nor hdbscan_labels in artifacts. "
                "Run Phase 1 for this model to enable feature_cluster_correlation."
            )
        }

    top_k          = config.get("fcc_top_k",          20)
    min_f          = config.get("fcc_min_f",           1.0)
    min_group_size = config.get("fcc_min_group_size",  2)

    # ------------------------------------------------------------------
    # Lifetime classes from feature_lifetimes (best-effort: may not be in
    # results yet since analyses run independently).  Compute inline.
    # ------------------------------------------------------------------
    try:
        lt_result = feature_lifetimes(crosscoder, prompt_store, artifacts, config)
        lifetime_class_arr = lt_result.get("lifetime_class", [])
    except Exception:
        lifetime_class_arr = []

    def _lifetime_class(feat_idx: int) -> str:
        if feat_idx < len(lifetime_class_arr):
            return str(lifetime_class_arr[feat_idx])
        return "unknown"

    # ------------------------------------------------------------------
    # Build label source
    # ------------------------------------------------------------------
    def _f_and_eta(groups: list) -> tuple:
        """
        One-way ANOVA F-statistic and eta-squared effect size.

        Parameters
        ----------
        groups : list of 1-D arrays, one per cluster group

        Returns
        -------
        (f_stat, eta_squared) — (nan, nan) if test cannot be run
        """
        groups = [g for g in groups if len(g) >= min_group_size]
        if len(groups) < 2:
            return float("nan"), float("nan")
        # All-zero across all groups → not informative
        all_vals = np.concatenate(groups)
        if all_vals.max() < 1e-10:
            return float("nan"), float("nan")

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                f_stat, p_val = f_oneway(*groups)
        except Exception:
            return float("nan"), float("nan")

        if np.isnan(f_stat):
            return float("nan"), float("nan")

        # eta-squared = SS_between / SS_total
        # For one-way ANOVA: eta² = F * df_between / (F * df_between + df_within)
        k = len(groups)
        n = len(all_vals)
        df_between = k - 1
        df_within  = n - k
        if df_within <= 0:
            return float(f_stat), float("nan")
        eta_sq = (f_stat * df_between) / (f_stat * df_between + df_within)

        return float(f_stat), float(eta_sq)

    def _rank_features(z: np.ndarray, labels: np.ndarray) -> list:
        """
        Rank all features by F-statistic over cluster groups.

        Parameters
        ----------
        z      : (n_tokens, n_features) — crosscoder sparse activations
        labels : (n_tokens,) int        — cluster labels, -1 = noise/excluded

        Returns
        -------
        list of dicts, sorted by f_stat descending, filtered by min_f
        """
        cluster_ids = sorted(set(labels.tolist()) - {-1})
        if len(cluster_ids) < 2:
            return []

        n_features = z.shape[1]
        records = []

        for feat_idx in range(n_features):
            feat_acts = z[:, feat_idx]          # (T,)

            # Group activations by cluster, excluding noise
            groups = [
                feat_acts[labels == cid]
                for cid in cluster_ids
            ]

            f_stat, eta_sq = _f_and_eta(groups)
            if np.isnan(f_stat) or f_stat < min_f:
                continue

            mean_by_cluster = [
                float(feat_acts[labels == cid].mean())
                for cid in cluster_ids
            ]

            records.append({
                "feature":          int(feat_idx),
                "f_stat":           f_stat,
                "eta_squared":      eta_sq,
                "mean_by_cluster":  mean_by_cluster,
                "cluster_ids":      [int(c) for c in cluster_ids],
                "lifetime_class":   _lifetime_class(feat_idx),
            })

        records.sort(key=lambda r: r["f_stat"], reverse=True)

        return records

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    results_per_prompt: dict = {}
    clustering_source_name: str

    if use_plateau:
        clustering_source_name = "plateau"
        plateau_clusters = _compute_plateau_clusters(
            prompt_store, layer_indices, plateau_layers,
            min_cluster_size=config.get("plateau_min_cluster_size", 3),
        )

        for prompt_key, layer_map in plateau_clusters.items():
            prompt_result = {}

            out = _run_crosscoder_on_prompt(crosscoder, prompt_store, prompt_key)
            z = out["z"].numpy()                        # (T, F)

            for mid_layer, cluster_info in layer_map.items():
                spec_labels = np.array(cluster_info["spectral"])
                hdb_labels  = np.array(cluster_info["hdbscan"])
                actual      = cluster_info["actual_model_layer"]
                dist        = cluster_info["distance_to_target"]

                spec_records = _rank_features(z, spec_labels)
                hdb_records  = _rank_features(z, hdb_labels)

                def _summary(records):
                    top = records[:top_k]
                    return {
                        "top_features":        top,
                        "n_features_tested":   z.shape[1],  # crosscoder dictionary size
                        "n_features_above_min": len(records),
                        "actual_model_layer":  actual,
                        "distance_to_target":  dist,
                    }

                prompt_result[int(mid_layer)] = {
                    "spectral": _summary(spec_records),
                    "hdbscan":  _summary(hdb_records),
                }

            results_per_prompt[prompt_key] = prompt_result

    else:
        clustering_source_name = "phase1_hdbscan"

        for prompt_key in prompt_store.keys():
            if prompt_key not in hdbscan_labels:
                continue

            out = _run_crosscoder_on_prompt(crosscoder, prompt_store, prompt_key)
            z = out["z"].numpy()
            prompt_result = {}

            for layer_key, labels in hdbscan_labels[prompt_key].items():
                labels_arr = np.array(labels)
                records    = _rank_features(z, labels_arr)
                prompt_result[layer_key] = {
                    "hdbscan": {
                        "top_features":         records[:top_k],
                        "n_features_above_min": len(records),
                    }
                }

            results_per_prompt[prompt_key] = prompt_result

    return {
        **results_per_prompt,
        "overall": {
            "n_prompts_run":    len(results_per_prompt),
            "clustering_source": clustering_source_name,
            "top_k":            top_k,
            "min_f":            min_f,
        },
    }


# ---------------------------------------------------------------------------
# Analysis: Top-feature token inspection (Step 7)
# ---------------------------------------------------------------------------

@register("inspect_top_features")
def inspect_top_features(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    For each top feature from feature_cluster_correlation, show which tokens
    it fires on and at what activation level, alongside cluster membership.

    This is the semantic content check: a feature that discriminates cluster A
    from cluster B should fire on tokens whose meaning matches the dominant
    semantic split in the passage — the same split Phase 1 identified through
    nearest-neighbor cycle analysis.

    For each (prompt, mid_layer, resolution) combination, the top
    inspect_n_features features are processed.  For each feature, two views
    are produced:

      by_activation : tokens sorted by activation value descending
                      (shows what the feature responds to most strongly)

      by_position   : all firing tokens in reading order with their cluster
                      label (shows whether the feature fires coherently within
                      a passage or scatters across it)

    The "firing" threshold is activation > 0 (the crosscoder already applies
    sparsity via BatchTopK, so any nonzero value is a genuine activation).

    A compact text rendering is also produced for each feature — a one-line
    summary readable directly in the run log or a text file — so inspection
    does not require parsing JSON.

    Parameters (via config)
    -----------------------
    inspect_n_features   : int  — how many top features to inspect per combo
                                  (default 10, taken from fcc top_k)
    inspect_top_tokens   : int  — tokens shown in by_activation view (default 20)
    inspect_min_act      : float — minimum activation to include a token (default 0.0,
                                   i.e. all firing tokens)
    inspect_resolution   : str  — "spectral" | "hdbscan" | "both" (default "spectral")

    Requires artifacts:
      "plateau_layers"  — for mid-plateau layer selection
      "layer_indices"   — crosscoder layer mapping

    Falls back to Phase 1 hdbscan_labels if plateau_layers is absent.
    Runs feature_cluster_correlation inline to get the top-feature ranking.

    Returns
    -------
    {
      prompt_key: {
        mid_layer: {
          resolution: {          # "spectral" and/or "hdbscan"
            feature_idx: {
              "feature":       int,
              "f_stat":        float,
              "eta_squared":   float,
              "lifetime_class": str,
              "by_activation": [
                {"token": str, "position": int, "activation": float, "cluster": int},
                ...  (top inspect_top_tokens, sorted by activation desc)
              ],
              "by_position": [
                {"token": str, "position": int, "activation": float, "cluster": int},
                ...  (all firing tokens, sorted by position)
              ],
              "cluster_token_sets": {
                cluster_id: [token_str, ...]  # unique tokens firing in that cluster
              },
              "text_summary":  str,  # one-line human-readable rendering
            }
          }
        }
      },
      "text_report": str,  # full multi-line report, printable to stdout or file
    }
    """
    plateau_layers = artifacts.get("plateau_layers")
    layer_indices  = artifacts.get("layer_indices")
    hdbscan_labels = artifacts.get("hdbscan_labels")

    use_plateau = plateau_layers is not None and layer_indices is not None
    use_phase1  = hdbscan_labels is not None

    if not use_plateau and not use_phase1:
        return {
            "error": (
                "Neither plateau_layers nor hdbscan_labels in artifacts. "
                "Run Phase 1 for this model to enable inspect_top_features."
            )
        }

    n_features_inspect = config.get("inspect_n_features",  10)
    n_top_tokens       = config.get("inspect_top_tokens",  20)
    min_act            = config.get("inspect_min_act",     0.0)
    resolution_cfg     = config.get("inspect_resolution",  "spectral")

    # ------------------------------------------------------------------
    # Run FCC inline to get the ranked feature list.
    # Use a config copy with top_k = n_features_inspect to avoid computing
    # F-stats for thousands of features we won't inspect.
    # ------------------------------------------------------------------
    fcc_config = dict(config)
    fcc_config["fcc_top_k"] = n_features_inspect
    fcc_result = feature_cluster_correlation(
        crosscoder, prompt_store, artifacts, fcc_config
    )

    if "error" in fcc_result:
        return {"error": f"feature_cluster_correlation failed: {fcc_result['error']}"}

    # ------------------------------------------------------------------
    # Build cluster label source (same logic as cluster_identity / fcc)
    # ------------------------------------------------------------------
    if use_plateau:
        plateau_clusters = _compute_plateau_clusters(
            prompt_store, layer_indices, plateau_layers,
            min_cluster_size=config.get("plateau_min_cluster_size", 3),
        )
    else:
        plateau_clusters = None

    # ------------------------------------------------------------------
    # Helper: render one feature as a compact text line
    # ------------------------------------------------------------------
    def _text_summary(feature_idx: int, f_stat: float, eta_sq: float,
                      lt_class: str, by_act: list, n_clusters: int) -> str:
        """
        One-line rendering:
          f{idx} F={f:.1f} η²={e:.3f} ({cls})  [C0: tok1 tok2 ...]  [C1: tok3 tok4 ...]
        """
        cluster_buckets: dict = {}
        for entry in by_act:
            c = entry["cluster"]
            if c == -1:
                continue
            cluster_buckets.setdefault(c, [])
            if len(cluster_buckets[c]) < 6:
                cluster_buckets[c].append(entry["token"])

        cluster_parts = "  ".join(
            f"[C{c}: {' '.join(toks)}]"
            for c, toks in sorted(cluster_buckets.items())
        )
        eta_str = f"{eta_sq:.3f}" if not (eta_sq != eta_sq) else "n/a"
        return (
            f"f{feature_idx:5d}  F={f_stat:8.2f}  η²={eta_str}  ({lt_class:12s})  "
            f"{cluster_parts}"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    results_per_prompt: dict = {}
    text_lines: list = []

    def _process_resolution(prompt_key, layer_key, resolution, labels_arr, z, tokens):
        """
        Build per-feature inspection dicts for one (prompt, layer, resolution).
        Returns (feature_dict, text_block).
        """
        # Get top features for this combo from FCC result
        layer_result = fcc_result.get(prompt_key, {}).get(
            int(layer_key) if isinstance(layer_key, (int, np.integer)) else layer_key, {}
        )
        res_result = layer_result.get(resolution, {})
        top_features_list = res_result.get("top_features", [])

        feature_results = {}
        text_block_lines = []
        hdr = f"  {prompt_key}  layer={layer_key}  [{resolution}]"
        text_block_lines.append(hdr)
        text_block_lines.append("  " + "-" * (len(hdr) - 2))

        for feat_rec in top_features_list[:n_features_inspect]:
            feat_idx   = feat_rec["feature"]
            f_stat     = feat_rec["f_stat"]
            eta_sq     = feat_rec.get("eta_squared", float("nan"))
            lt_class   = feat_rec.get("lifetime_class", "unknown")

            act_values = z[:, feat_idx]  # (n_tokens,)

            # Build token-level records for all firing tokens.
            # Warn if stored token list length diverges from the activation
            # tensor length — this indicates a tokenizer/truncation mismatch
            # between extraction and analysis time.
            n_acts   = len(act_values)
            n_labels = len(labels_arr)
            n_toks   = len(tokens)
            if not (n_acts == n_labels == n_toks):
                min_len = min(n_acts, n_labels, n_toks)
                print(
                    f"    [inspect_top_features] Length mismatch for {prompt_key}: "
                    f"activations={n_acts}, labels={n_labels}, tokens={n_toks}. "
                    f"Truncating to {min_len}."
                )
                act_values  = act_values[:min_len]
                labels_arr  = labels_arr[:min_len]
                tokens_view = tokens[:min_len]
            else:
                tokens_view = tokens

            firing_records = []
            for pos, (tok, act, lbl) in enumerate(
                zip(tokens_view, act_values.tolist(), labels_arr.tolist())
            ):
                if act > min_act:
                    firing_records.append({
                        "token":      tok,
                        "position":   pos,
                        "activation": round(float(act), 6),
                        "cluster":    int(lbl),
                    })

            # Two views
            by_activation = sorted(
                firing_records, key=lambda r: r["activation"], reverse=True
            )[:n_top_tokens]

            by_position = sorted(firing_records, key=lambda r: r["position"])

            # Per-cluster unique token sets (ordered by mean activation desc)
            cluster_ids = sorted(set(labels_arr.tolist()) - {-1})
            cluster_token_sets = {}
            for cid in cluster_ids:
                cid_records = [r for r in firing_records if r["cluster"] == cid]
                # Deduplicate while preserving order of first occurrence by position
                seen = set()
                unique_toks = []
                for r in sorted(cid_records, key=lambda x: x["activation"], reverse=True):
                    t = r["token"]
                    if t not in seen:
                        seen.add(t)
                        unique_toks.append(t)
                cluster_token_sets[cid] = unique_toks

            n_clusters = len(cluster_ids)
            summary_line = _text_summary(
                feat_idx, f_stat, eta_sq, lt_class, by_activation, n_clusters
            )
            text_block_lines.append("    " + summary_line)

            feature_results[feat_idx] = {
                "feature":            feat_idx,
                "f_stat":             f_stat,
                "eta_squared":        eta_sq,
                "lifetime_class":     lt_class,
                "n_firing_tokens":    len(firing_records),
                "by_activation":      by_activation,
                "by_position":        by_position,
                "cluster_token_sets": {str(k): v for k, v in cluster_token_sets.items()},
                "text_summary":       summary_line,
            }

        return feature_results, "\n".join(text_block_lines)

    for prompt_key in prompt_store.keys():
        prompt_tokens = prompt_store.prompts[prompt_key]["tokens"]

        out = _run_crosscoder_on_prompt(crosscoder, prompt_store, prompt_key)
        z = out["z"].numpy()  # (n_tokens, n_features)

        prompt_result: dict = {}

        if use_plateau and plateau_clusters is not None:
            if prompt_key not in plateau_clusters:
                continue
            for mid_layer, cluster_info in plateau_clusters[prompt_key].items():

                layer_result: dict = {}
                resolutions = (
                    ["spectral", "hdbscan"]
                    if resolution_cfg == "both"
                    else [resolution_cfg]
                )

                for res in resolutions:
                    labels_arr = np.array(cluster_info[res])
                    feat_dict, text_block = _process_resolution(
                        prompt_key, mid_layer, res, labels_arr, z, prompt_tokens
                    )
                    layer_result[res] = feat_dict
                    text_lines.append(text_block)

                prompt_result[int(mid_layer)] = layer_result

        else:
            # Phase 1 hdbscan_labels fallback
            if prompt_key not in hdbscan_labels:
                continue
            for layer_key, labels in hdbscan_labels[prompt_key].items():
                labels_arr = np.array(labels)
                feat_dict, text_block = _process_resolution(
                    prompt_key, layer_key, "hdbscan", labels_arr, z, prompt_tokens
                )
                prompt_result[layer_key] = {"hdbscan": feat_dict}
                text_lines.append(text_block)

        if prompt_result:
            results_per_prompt[prompt_key] = prompt_result

    text_report = "\n\n".join(text_lines)

    return {
        **results_per_prompt,
        "text_report": text_report,
    }


# ---------------------------------------------------------------------------
# Analysis: Scale correspondence (Step 8)
# ---------------------------------------------------------------------------

@register("scale_correspondence")
def scale_correspondence(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    artifacts: dict,
    config: dict,
) -> dict:
    """
    Test whether features that discriminate the spectral k=2 bipartition also
    discriminate HDBSCAN subclusters at finer resolution, or whether different
    features operate at each scale.

    Two F-stat vectors are computed per feature:

      F_spectral : one-way ANOVA across the two spectral partitions (k=2).
                   Tests global-split discrimination.

      F_within   : for each spectral partition independently, one-way ANOVA
                   across its HDBSCAN subclusters.  The per-partition F-stats
                   are averaged (weighted by partition size) to produce one
                   scalar per feature.  This is the correct operationalisation
                   of "subcluster selectivity": a feature that fires broadly
                   on all of partition 0 will have low F_within (no subcluster
                   structure within that partition), whereas a feature that
                   fires on one specific subcluster of partition 0 will have
                   high F_within.

    Using F_within instead of global HDBSCAN F prevents a confound: because
    HDBSCAN subclusters are nested inside spectral partitions, a feature with
    high F_spectral will mechanically also have high global F_hdbscan (it
    discriminates cluster 0 from cluster 3 even if both are in partition 1).
    F_within removes this confound.

    Spearman rank correlation between F_spectral and F_within answers:
      high rho (> 0.5)  : same features operate at both scales — feature
                          selectivity is hierarchical (global split = local
                          subclusters, just at different granularity).
      near-zero rho     : scale-specific features — the global split and
                          local subclusters are governed by distinct feature
                          sets.
      negative rho      : features that discriminate the global split are
                          anti-discriminative locally, firing broadly within
                          one partition rather than on specific subclusters.

    Population classification uses top/bottom quartile thresholds on the
    valid-feature F distributions:
      shared_top          : top-quartile at both scales
      spectral_exclusive  : top-quartile F_spectral, bottom-quartile F_within
      subcluster_exclusive: top-quartile F_within,   bottom-quartile F_spectral

    Each population carries a sorted feature list (with both F-stats) and a
    lifetime class breakdown, so the results are directly interpretable without
    rerunning other analyses.

    Requires artifacts:
      "plateau_layers"  : {prompt_key: [mid_layer_num, ...]}
      "layer_indices"   : [int, ...]

    Returns
    -------
    {
      prompt_key: {
        mid_layer (int): {
          "spearman_rho":            float,
          "spearman_pval":           float,
          "n_features_both":         int,
          "f_spectral":              [float, ...],  # (n_features,) nan for untested
          "f_within":                [float, ...],  # (n_features,) nan for untested
          "spectral_exclusive":      [{feature, f_spectral, f_within, lifetime_class}, ...],
          "subcluster_exclusive":    [{feature, f_spectral, f_within, lifetime_class}, ...],
          "shared_top":              [{feature, f_spectral, f_within, lifetime_class}, ...],
          "lifetime_breakdown": {
            "spectral_exclusive":    {"short_lived": int, "long_lived": int, "other": int},
            "subcluster_exclusive":  {"short_lived": int, "long_lived": int, "other": int},
            "shared_top":            {"short_lived": int, "long_lived": int, "other": int},
          },
          "actual_model_layer":      int,
          "n_hdbscan_clusters":      int,
          "partition_sizes":         [int, int],
        }
      }
    }
    """
    from scipy.stats import spearmanr, f_oneway
    import warnings

    plateau_layers   = artifacts.get("plateau_layers")
    layer_indices    = artifacts.get("layer_indices")

    if plateau_layers is None or layer_indices is None:
        return {"error": "plateau_layers or layer_indices not in artifacts"}

    min_group_size   = config.get("fcc_min_group_size",      2)
    min_cluster_size = config.get("plateau_min_cluster_size", 3)
    top_n_per_pop    = config.get("sc_top_n_per_population", 20)

    # ------------------------------------------------------------------
    # Lifetime class for breakdown annotation
    # ------------------------------------------------------------------
    try:
        lt_result          = feature_lifetimes(crosscoder, prompt_store, artifacts, config)
        lifetime_class_arr = lt_result.get("lifetime_class", [])
    except Exception:
        lifetime_class_arr = []

    def _lc(idx: int) -> str:
        return str(lifetime_class_arr[idx]) if idx < len(lifetime_class_arr) else "unknown"

    def _breakdown(feature_records: list) -> dict:
        return {
            "short_lived": sum(1 for r in feature_records if r["lifetime_class"] == "short_lived"),
            "long_lived":  sum(1 for r in feature_records if r["lifetime_class"] == "long_lived"),
            "other":       sum(1 for r in feature_records
                               if r["lifetime_class"] not in ("short_lived", "long_lived")),
        }

    # ------------------------------------------------------------------
    # F-stat helpers
    # ------------------------------------------------------------------
    def _f_one_vs_rest(groups: list) -> float:
        """One-way ANOVA F across groups; nan if not computable."""
        groups = [g for g in groups if len(g) >= min_group_size]
        if len(groups) < 2:
            return float("nan")
        all_vals = np.concatenate(groups)
        if all_vals.max() < 1e-10:
            return float("nan")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                f_stat, _ = f_oneway(*groups)
        except Exception:
            return float("nan")
        return float("nan") if np.isnan(f_stat) else float(f_stat)

    def _spectral_f_all(z: np.ndarray, spec_labels: np.ndarray) -> np.ndarray:
        """
        (n_features,) F-stat for spectral k=2 split.

        For k=2 this is equivalent to a one-way ANOVA with two groups,
        which equals the squared Welch t-statistic.  Computed feature-by-feature
        to reuse _f_one_vs_rest.
        """
        cids = sorted(set(spec_labels.tolist()) - {-1})
        if len(cids) < 2:
            return np.full(z.shape[1], float("nan"))
        f_arr = np.full(z.shape[1], float("nan"))
        for fi in range(z.shape[1]):
            acts = z[:, fi]
            if acts.max() < 1e-10:
                continue
            groups = [acts[spec_labels == c] for c in cids]
            f_arr[fi] = _f_one_vs_rest(groups)
        return f_arr

    def _within_partition_f_all(
        z: np.ndarray,
        spec_labels: np.ndarray,
        hdb_labels: np.ndarray,
    ) -> np.ndarray:
        """
        (n_features,) size-weighted mean within-partition F-stat.

        For each spectral partition p in {0, 1}:
          - Identify the HDBSCAN subclusters that fall within partition p
            (the majority-vote partition of each HDBSCAN cluster).
          - Compute one-way ANOVA F across those subclusters for tokens
            in partition p only.

        Returns the partition-size-weighted average F across the two partitions.
        NaN if a partition has fewer than 2 subclusters.
        """
        spec_parts = sorted(set(spec_labels.tolist()) - {-1})
        if len(spec_parts) < 2:
            return np.full(z.shape[1], float("nan"))

        hdb_ids = sorted(set(hdb_labels.tolist()) - {-1})
        if len(hdb_ids) < 2:
            return np.full(z.shape[1], float("nan"))

        # Assign each HDBSCAN cluster to a spectral partition by majority vote
        hdb_to_part: dict = {}
        for hid in hdb_ids:
            hdb_mask = hdb_labels == hid
            part_votes = [int((hdb_mask & (spec_labels == p)).sum())
                          for p in spec_parts]
            hdb_to_part[hid] = spec_parts[int(np.argmax(part_votes))]

        # Per-partition subcluster lists
        part_subclusters: dict = {p: [] for p in spec_parts}
        for hid, part in hdb_to_part.items():
            part_subclusters[part].append(hid)

        # Only compute for partitions with >= 2 subclusters
        active_parts = [p for p, sc in part_subclusters.items() if len(sc) >= 2]
        if not active_parts:
            return np.full(z.shape[1], float("nan"))

        part_sizes = {p: int((spec_labels == p).sum()) for p in active_parts}
        total_size = sum(part_sizes.values())

        # Initialize to nan: features that receive no valid ANOVA contribution
        # (groups too small, all-zero activations, etc.) must remain nan so
        # np.isfinite() correctly excludes them from the both_valid mask.
        # Initializing to 0 would make them finite and land in the bottom
        # quartile of f_within, artificially inflating spectral_exclusive counts.
        f_arr = np.full(z.shape[1], float("nan"))
        weight_sum = 0.0
        # Track which features received at least one valid weighted contribution
        f_valid_weight = np.zeros(z.shape[1])

        for part in active_parts:
            part_mask  = spec_labels == part
            sub_ids    = part_subclusters[part]
            weight     = part_sizes[part] / total_size
            weight_sum += weight

            for fi in range(z.shape[1]):
                acts = z[:, fi]
                if acts.max() < 1e-10:
                    continue
                groups = [
                    acts[hdb_labels == sid]
                    for sid in sub_ids
                    if (hdb_labels == sid).sum() >= min_group_size
                ]
                f = _f_one_vs_rest(groups)
                if not np.isnan(f):
                    if np.isnan(f_arr[fi]):
                        f_arr[fi] = 0.0
                    f_arr[fi] += weight * f
                    f_valid_weight[fi] += weight

        if weight_sum < 1e-10:
            return np.full(z.shape[1], float("nan"))

        # Normalize each feature by its own accumulated valid weight so that
        # features with contributions from only one partition are still correctly
        # scaled (rather than being halved by a global weight_sum of 1.0).
        valid_mask = f_valid_weight > 1e-10
        f_arr[valid_mask] /= f_valid_weight[valid_mask]
        # Features with no valid contribution remain nan (set during init)

        return f_arr

    # ------------------------------------------------------------------
    # Build sorted feature record list for a population mask
    # ------------------------------------------------------------------
    def _sorted_records(mask: np.ndarray, f_spec: np.ndarray,
                        f_within: np.ndarray, top_n: int) -> list:
        indices = np.where(mask)[0]
        # Sort by F_spectral descending as primary key
        indices = indices[np.argsort(f_spec[indices])[::-1]]
        records = []
        for fi in indices[:top_n]:
            records.append({
                "feature":        int(fi),
                "f_spectral":     float(f_spec[fi]),
                "f_within":       float(f_within[fi]),
                "lifetime_class": _lc(int(fi)),
            })
        return records

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    plateau_clusters = _compute_plateau_clusters(
        prompt_store, layer_indices, plateau_layers, min_cluster_size
    )

    results_out: dict = {}

    for prompt_key in prompt_store.keys():
        if prompt_key not in plateau_clusters:
            continue

        out = _run_crosscoder_on_prompt(crosscoder, prompt_store, prompt_key)
        z   = out["z"].numpy()   # (T, F)

        prompt_result: dict = {}

        for mid_layer, cluster_info in plateau_clusters[prompt_key].items():
            spec_labels = np.array(cluster_info["spectral"])
            hdb_labels  = np.array(cluster_info["hdbscan"])
            n_hdb       = cluster_info["n_hdbscan_clusters"]

            # Partition sizes
            partition_sizes = [int((spec_labels == p).sum()) for p in (0, 1)]

            if n_hdb < 2:
                prompt_result[int(mid_layer)] = {
                    "error": f"only {n_hdb} HDBSCAN cluster(s) — need ≥ 2 for within-partition F",
                    "n_hdbscan_clusters": n_hdb,
                    "actual_model_layer": cluster_info["actual_model_layer"],
                }
                continue

            f_spec   = _spectral_f_all(z, spec_labels)        # (F,)
            f_within = _within_partition_f_all(z, spec_labels, hdb_labels)  # (F,)

            # Features with finite F at both scales
            both_valid = np.isfinite(f_spec) & np.isfinite(f_within)
            n_both     = int(both_valid.sum())

            if n_both < 10:
                prompt_result[int(mid_layer)] = {
                    "error": f"only {n_both} features with valid F at both scales",
                    "n_features_both": n_both,
                    "n_hdbscan_clusters": n_hdb,
                    "actual_model_layer": cluster_info["actual_model_layer"],
                }
                continue

            f_spec_v   = f_spec[both_valid]
            f_within_v = f_within[both_valid]

            rho, pval = spearmanr(f_spec_v, f_within_v)

            # Quartile thresholds on valid features
            q75_spec   = float(np.percentile(f_spec_v,   75))
            q25_spec   = float(np.percentile(f_spec_v,   25))
            q75_within = float(np.percentile(f_within_v, 75))
            q25_within = float(np.percentile(f_within_v, 25))

            # Population masks (over ALL features, not just valid ones)
            spec_ex_mask = (
                np.isfinite(f_spec) & np.isfinite(f_within) &
                (f_spec   >= q75_spec)   & (f_within <= q25_within)
            )
            sub_ex_mask  = (
                np.isfinite(f_spec) & np.isfinite(f_within) &
                (f_within >= q75_within) & (f_spec   <= q25_spec)
            )
            shared_mask  = (
                np.isfinite(f_spec) & np.isfinite(f_within) &
                (f_spec   >= q75_spec)   & (f_within >= q75_within)
            )

            spectral_exclusive   = _sorted_records(spec_ex_mask, f_spec, f_within, top_n_per_pop)
            subcluster_exclusive = _sorted_records(sub_ex_mask,  f_spec, f_within, top_n_per_pop)
            shared_top           = _sorted_records(shared_mask,  f_spec, f_within, top_n_per_pop)

            prompt_result[int(mid_layer)] = {
                "spearman_rho":      float(rho),
                "spearman_pval":     float(pval),
                "n_features_both":   n_both,
                "f_spectral":        [
                    float(v) if np.isfinite(v) else None
                    for v in f_spec.tolist()
                ],
                "f_within":          [
                    float(v) if np.isfinite(v) else None
                    for v in f_within.tolist()
                ],
                "spectral_exclusive":    spectral_exclusive,
                "subcluster_exclusive":  subcluster_exclusive,
                "shared_top":            shared_top,
                "lifetime_breakdown": {
                    "spectral_exclusive":   _breakdown(spectral_exclusive),
                    "subcluster_exclusive": _breakdown(subcluster_exclusive),
                    "shared_top":           _breakdown(shared_top),
                },
                "actual_model_layer":  cluster_info["actual_model_layer"],
                "n_hdbscan_clusters":  n_hdb,
                "partition_sizes":     partition_sizes,
            }

        if prompt_result:
            results_out[prompt_key] = prompt_result

    return results_out



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
        if isinstance(obj, np.bool_):
            return bool(obj)
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
