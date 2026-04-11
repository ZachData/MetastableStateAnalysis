"""
steering.py — Causal steering test for cluster-selective crosscoder features.

Steps 9–11 from the analysis plan.

At a mid-plateau layer, inject a scaled decoder direction into the residual
stream for all tokens in a target cluster, then continue the forward pass.
Measure whether the merge event shifts earlier (destabilise), later (stabilise),
or not at all (null).

Entry points
------------
    run_steering(...)                -> list[SteeringResult]   (parts 9–10)
    analyse_lifetime_correspondence(results) -> dict           (part 11)
    analyse_pair_tracking(results)   -> dict                   (phase-3 addition)
    summarise_steering(results)      -> dict
    save_steering_results(results, summary, path)
"""

from __future__ import annotations

import dataclasses
import json
import warnings
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .crosscoder import Crosscoder
from .data import PromptActivationStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SteeringResult:
    """Outcome of one steering experiment."""

    prompt_key:            str
    mid_layer:             int    # requested mid-plateau layer (model-layer space)
    actual_layer:          int    # actual sampled layer used
    feature_idx:           int
    f_stat:                float  # F-stat from FCC (how cluster-selective this feature is)
    lifetime_class:        str    # "short_lived" | "long_lived" | "dead" | "unknown"
    alpha:                 float  # perturbation scale applied
    cluster_targeted:      int    # spectral partition that received the perturbation

    merge_layer_baseline:  int    # model-layer of merge event in unperturbed run (-1 = none)
    merge_layer_perturbed: int    # model-layer of merge event in perturbed run   (-1 = none)
    merge_delta:           int    # merge_perturbed - merge_baseline (0 if either is -1)

    ip_mean_baseline:      list   # ip_mean at each snapshot layer after steering point
    ip_mean_perturbed:     list
    layer_nums_after:      list   # model-layer numbers corresponding to trajectory lists

    fiedler_baseline:      list   # Fiedler eigenvalue trajectory (secondary signal)
    fiedler_perturbed:     list

    # Step-10 outcome
    outcome: str  # "stabilise" | "destabilise" | "null" | "no_baseline_merge"

    # Phase-3 addition: pair-level causal tracking per snapshot layer
    pair_tracking: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _ip_mean(acts: np.ndarray) -> float:
    """Mean pairwise cosine similarity of a (T, d) activation matrix."""
    norms = np.linalg.norm(acts, axis=1, keepdims=True).clip(min=1e-8)
    acts_n = acts / norms
    S = acts_n @ acts_n.T
    T = acts.shape[0]
    if T < 2:
        return float(np.diag(S).mean())
    return float((S.sum() - np.trace(S)) / (T * (T - 1)))


def _fiedler(acts: np.ndarray) -> float:
    """
    Algebraic connectivity (Fiedler eigenvalue) of the cosine-similarity graph.

    Low → near-disconnected (cluster structure, plateau state).
    High → well-mixed (collapsed, post-merge state).
    """
    from scipy.linalg import eigh
    norms = np.linalg.norm(acts, axis=1, keepdims=True).clip(min=1e-8)
    acts_n = acts / norms
    S = (acts_n @ acts_n.T).clip(min=0.0)
    deg = S.sum(axis=1)
    di = np.where(deg > 1e-10, 1.0 / np.sqrt(deg), 0.0)
    S_norm = di[:, None] * S * di[None, :]
    n = S_norm.shape[0]
    L = np.eye(n) - S_norm
    vals, _ = eigh(L, subset_by_index=[0, min(1, n - 1)])
    return float(vals[1]) if len(vals) > 1 else float(vals[0])


def _alpha_scale(decoder_dir: np.ndarray, acts: np.ndarray) -> float:
    """
    α = RMS of (acts @ decoder_dir), clamped to [1e-3, 1.0].

    Calibrates the perturbation to the natural scale of variation in the
    decoder direction at the steering layer, so the intervention is neither
    invisible nor geometry-destroying.
    """
    proj = acts @ decoder_dir
    rms = float(np.sqrt((proj ** 2).mean()))
    return float(np.clip(rms, 1e-3, 1.0))


def _detect_merge(
    ip_traj:   list,
    layer_nums: list,
    plat_mean: float,
    plat_std:  float,
    sigma:     float,
) -> int:
    """First layer where ip_mean > plat_mean + sigma * plat_std. -1 if none."""
    threshold = plat_mean + sigma * plat_std
    for l, v in zip(layer_nums, ip_traj):
        if v > threshold:
            return l
    return -1


# ---------------------------------------------------------------------------
# Pair-level causal tracking helpers  (Phase-3 addition)
# ---------------------------------------------------------------------------

def _nearest_neighbors(acts: np.ndarray) -> np.ndarray:
    """(T,) array of nearest-neighbor indices by cosine similarity."""
    norms = np.linalg.norm(acts, axis=1, keepdims=True)
    normed = acts / np.maximum(norms, 1e-10)
    sim = normed @ normed.T
    np.fill_diagonal(sim, -2.0)
    return sim.argmax(axis=1)


def _mutual_nn_pairs(nn: np.ndarray) -> set:
    """Set of (min(i,j), max(i,j)) where i and j are mutual NNs."""
    pairs = set()
    for i, j in enumerate(nn):
        if nn[j] == i:
            pairs.add((min(i, j), max(i, j)))
    return pairs


# ---------------------------------------------------------------------------
# ALBERT forward pass with optional perturbation
# ---------------------------------------------------------------------------

def _run_albert(
    model,
    inputs:         dict,
    max_iter:       int,
    snapshot_steps: set,
    device:         str,
    steer_step:     Optional[int]          = None,
    perturbation:   Optional[torch.Tensor] = None,  # (T, d)
) -> dict:
    """
    ALBERT iterative forward pass, snapshotting hidden states at snapshot_steps.

    If steer_step is not None, adds perturbation to hidden state *before*
    the albert_layer call at that step.  This is equivalent to patching the
    residual stream between steps steer_step-1 and steer_step.

    Returns {step_int: np.ndarray(T, d) float32}
    """
    snaps: dict = {}

    with torch.no_grad(), torch.autocast(
        device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")
    ):
        emb = model.embeddings(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs.get("token_type_ids"),
        )
        hidden = model.encoder.embedding_hidden_mapping_in(emb)
        attn_mask = model.get_extended_attention_mask(
            inputs["attention_mask"], inputs["input_ids"].shape
        )
        albert_layer = model.encoder.albert_layer_groups[0].albert_layers[0]

        if 0 in snapshot_steps:
            snaps[0] = hidden[0].float().cpu().numpy()

        for step in range(1, max_iter + 1):
            if steer_step is not None and step == steer_step and perturbation is not None:
                hidden = hidden.clone()
                hidden[0] = hidden[0] + perturbation.to(
                    dtype=hidden.dtype, device=hidden.device
                )

            out    = albert_layer(hidden, attention_mask=attn_mask)
            hidden = out[0]

            if step in snapshot_steps:
                snaps[step] = hidden[0].float().cpu().numpy()

    return snaps


# ---------------------------------------------------------------------------
# GPT-2 forward pass with optional perturbation
# ---------------------------------------------------------------------------

def _run_gpt2(
    model,
    inputs:       dict,
    layer_indices: list,
    steer_layer:  Optional[int]           = None,  # HF layer index (0-based)
    perturbation: Optional[torch.Tensor]  = None,  # (T, d)
) -> dict:
    """
    GPT-2 forward pass with optional one-shot activation patch.

    Requires output_hidden_states=True (added here).
    Returns {model_layer_num: np.ndarray(T, d)} for all layers in layer_indices.
    """
    fired = [False]
    hook_handle = [None]

    if steer_layer is not None and perturbation is not None:
        def _hook(module, inp, output):
            if fired[0]:
                return output
            fired[0] = True
            h = output[0]
            patched = h + perturbation.to(dtype=h.dtype, device=h.device)
            hook_handle[0].remove()
            return (patched,) + output[1:]

        hook_handle[0] = model.transformer.h[steer_layer].register_forward_hook(_hook)

    inputs_with_hs = {**inputs, "output_hidden_states": True}
    try:
        with torch.no_grad():
            out = model(**inputs_with_hs)
    finally:
        # Always remove the hook — even if forward raises — so it doesn't
        # persist and corrupt subsequent forward passes on the same model.
        if hook_handle[0] is not None and not fired[0]:
            hook_handle[0].remove()

    # hidden_states[i] is (1, T, d) for layer i (0 = embedding)
    return {
        layer_indices[i]: out.hidden_states[i][0].float().cpu().numpy()
        for i in range(len(layer_indices))
    }


# ---------------------------------------------------------------------------
# Core: steer one feature
# ---------------------------------------------------------------------------

def _steer_one_feature(
    model,
    tokenizer,
    crosscoder:         Crosscoder,
    prompt_text:        str,
    prompt_key:         str,
    steer_layer_model:  int,
    steer_layer_cc_idx: int,
    feature_idx:        int,
    f_stat:             float,
    cluster_labels:     np.ndarray,
    target_cluster:     int,
    layer_indices:      list,
    is_albert:          bool,
    device:             str,
    config:             dict,
) -> SteeringResult:
    """Run baseline and perturbed forward passes; classify the outcome."""

    lt_arr = config.get("_lifetime_class_arr", [])
    lifetime_class = str(lt_arr[feature_idx]) if feature_idx < len(lt_arr) else "unknown"

    # Decoder direction: (d,) unit vector at the steering crosscoder layer
    W_dec = crosscoder.W_dec.detach().cpu().float().numpy()  # (L, F, d)
    dec_dir = W_dec[steer_layer_cc_idx, feature_idx, :]
    dec_dir = dec_dir / (np.linalg.norm(dec_dir) + 1e-8)

    # Tokenize
    inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    max_iter = max(layer_indices)

    # Snapshot every sampled layer at or after the steering point
    snapshot_layers = sorted(l for l in layer_indices if l >= steer_layer_model)
    snapshot_set    = set(snapshot_layers)

    # ---- Baseline ----
    if is_albert:
        base_snaps = _run_albert(
            model, inputs, max_iter, snapshot_set, device
        )
    else:
        base_snaps = _run_gpt2(model, inputs, layer_indices)

    # Activations at the steering layer for alpha calibration
    steer_acts = base_snaps.get(steer_layer_model)
    if steer_acts is None:
        closest = min(base_snaps, key=lambda x: abs(x - steer_layer_model))
        steer_acts = base_snaps[closest]

    alpha = _alpha_scale(dec_dir, steer_acts) * config.get("steering_alpha_multiplier", 1.0)

    # Build perturbation: only on tokens belonging to target_cluster
    T = steer_acts.shape[0]
    n_tokens_in_labels = len(cluster_labels)
    if T != n_tokens_in_labels:
        # Token count mismatch between stored activations and re-tokenised text.
        # Use the minimum length and warn — this can happen with whitespace.
        T_use = min(T, n_tokens_in_labels)
        cluster_labels_use = cluster_labels[:T_use]
    else:
        T_use = T
        cluster_labels_use = cluster_labels

    pert_np = np.zeros((T, dec_dir.shape[0]), dtype=np.float32)
    mask    = np.zeros(T, dtype=bool)
    mask[:T_use] = (cluster_labels_use == target_cluster)
    pert_np[mask] = alpha * dec_dir
    pert_t = torch.from_numpy(pert_np)

    # ---- Perturbed ----
    if is_albert:
        pert_snaps = _run_albert(
            model, inputs, max_iter, snapshot_set, device,
            steer_step=steer_layer_model, perturbation=pert_t,
        )
    else:
        # HuggingFace GPT-2 layer index = model layer number (0-based)
        pert_snaps = _run_gpt2(
            model, inputs, layer_indices,
            steer_layer=steer_layer_model, perturbation=pert_t,
        )

    # ---- Trajectories ----
    ip_base = [_ip_mean(base_snaps[l]) for l in snapshot_layers if l in base_snaps]
    ip_pert = [_ip_mean(pert_snaps[l]) for l in snapshot_layers if l in pert_snaps]
    fi_base = [_fiedler(base_snaps[l]) for l in snapshot_layers if l in base_snaps]
    fi_pert = [_fiedler(pert_snaps[l]) for l in snapshot_layers if l in pert_snaps]

    # Plateau reference: first quarter of trajectory gives baseline geometry
    n_ref   = max(2, len(ip_base) // 4)
    p_mean  = float(np.mean(ip_base[:n_ref]))
    p_std   = max(float(np.std(ip_base[:n_ref])), 1e-4)
    sigma   = config.get("steering_merge_threshold_sigma", 2.0)

    merge_base = _detect_merge(ip_base, snapshot_layers, p_mean, p_std, sigma)
    merge_pert = _detect_merge(ip_pert, snapshot_layers, p_mean, p_std, sigma)

    # ---- Pair-level causal tracking (Phase-3 addition) ----
    pair_tracking = {}
    for snap_layer in snapshot_layers:
        b_acts = base_snaps.get(snap_layer)
        p_acts = pert_snaps.get(snap_layer)
        if b_acts is None or p_acts is None:
            continue
        b_nn = _nearest_neighbors(b_acts)
        p_nn = _nearest_neighbors(p_acts)
        b_mutual = _mutual_nn_pairs(b_nn)
        p_mutual = _mutual_nn_pairs(p_nn)

        formed = p_mutual - b_mutual
        broken = b_mutual - p_mutual
        stable = b_mutual & p_mutual

        pair_tracking[snap_layer] = {
            "n_baseline_pairs": len(b_mutual),
            "n_perturbed_pairs": len(p_mutual),
            "n_formed":         len(formed),
            "n_broken":         len(broken),
            "n_stable":         len(stable),
            "pair_jaccard":     len(stable) / max(len(b_mutual | p_mutual), 1),
        }

    # ---- Outcome classification ----
    if merge_base == -1:
        outcome     = "no_baseline_merge"
        merge_delta = 0
    elif merge_pert == -1:
        outcome     = "stabilise"
        # Lower bound: merge shifted from merge_base to beyond last snapshot.
        merge_delta = snapshot_layers[-1] - merge_base if snapshot_layers else 0
    elif merge_pert > merge_base:
        outcome     = "stabilise"
        merge_delta = merge_pert - merge_base
    elif merge_pert < merge_base:
        outcome     = "destabilise"
        merge_delta = merge_pert - merge_base   # negative
    else:
        outcome     = "null"
        merge_delta = 0

    return SteeringResult(
        prompt_key            = prompt_key,
        mid_layer             = steer_layer_model,
        actual_layer          = steer_layer_model,
        feature_idx           = feature_idx,
        f_stat                = f_stat,
        lifetime_class        = lifetime_class,
        alpha                 = alpha,
        cluster_targeted      = int(target_cluster),
        merge_layer_baseline  = merge_base,
        merge_layer_perturbed = merge_pert,
        merge_delta           = merge_delta,
        ip_mean_baseline      = ip_base,
        ip_mean_perturbed     = ip_pert,
        layer_nums_after      = list(snapshot_layers),
        fiedler_baseline      = fi_base,
        fiedler_perturbed     = fi_pert,
        outcome               = outcome,
        pair_tracking         = pair_tracking,
    )


# ---------------------------------------------------------------------------
# Public entry point (parts 9–10)
# ---------------------------------------------------------------------------

def run_steering(
    model,
    tokenizer,
    crosscoder:         Crosscoder,
    prompt_store:       PromptActivationStore,
    prompts:            dict,           # {prompt_key: raw_text} from core.config.PROMPTS
    fcc_results:        dict,
    plateau_layers:     dict,
    layer_indices:      list,
    is_albert:          bool,
    device:             str,
    config:             dict,
    lifetime_class_arr: list = None,
) -> list:
    """
    Part 9: for each top FCC feature at each mid-plateau layer, run the
    steering experiment (baseline + perturbed forward pass).

    Part 10: classify each experiment as stabilise / destabilise / null /
    no_baseline_merge based on whether the merge event layer shifts.

    Parameters
    ----------
    model              : loaded HuggingFace model
    tokenizer          : matching tokenizer
    crosscoder         : trained Crosscoder
    prompt_store       : PromptActivationStore (for cluster labels)
    prompts            : {prompt_key: raw_text} — original prompt strings
    fcc_results        : output of feature_cluster_correlation
    plateau_layers     : {prompt_key: [mid_layer_num, ...]}
    layer_indices      : model layer numbers covered by the crosscoder
    is_albert          : True for ALBERT, False for GPT-2
    device             : "cuda" or "cpu"
    config             : see keys below
    lifetime_class_arr : (n_features,) list from feature_lifetimes["lifetime_class"]

    Config keys
    -----------
    steering_n_features           : top features to steer per (prompt, layer) [default 5]
    steering_alpha_multiplier     : scale α up/down [default 1.0]
    steering_merge_threshold_sigma: σ above plateau = merge [default 2.0]
    plateau_min_cluster_size      : passed to _compute_plateau_clusters [default 3]

    Returns
    -------
    list[SteeringResult]
    """
    from .analysis import _compute_plateau_clusters

    n_steer = config.get("steering_n_features", 5)
    steer_cfg = dict(config)
    steer_cfg["_lifetime_class_arr"] = lifetime_class_arr or []

    plateau_clusters = _compute_plateau_clusters(
        prompt_store, layer_indices, plateau_layers,
        min_cluster_size=config.get("plateau_min_cluster_size", 3),
    )

    results: list[SteeringResult] = []

    for prompt_key in prompt_store.keys():
        if prompt_key not in plateau_layers:
            continue
        if prompt_key not in fcc_results:
            continue
        if prompt_key not in plateau_clusters:
            continue

        # Raw text for re-tokenization. Fall back to joining stored tokens
        # only if the raw text is unavailable (should not normally happen).
        prompt_text = prompts.get(prompt_key, "")

        for mid_layer in plateau_layers[prompt_key]:
            layer_fcc = fcc_results[prompt_key].get(str(mid_layer), {})
            if not layer_fcc:
                continue

            # Top-N features by F-stat
            features_sorted = sorted(
                layer_fcc.items(), key=lambda kv: kv[1].get("f_stat", 0), reverse=True
            )[:n_steer]

            cc_layer_idx = layer_indices.index(mid_layer) if mid_layer in layer_indices else None
            if cc_layer_idx is None:
                continue

            spec_labels, target_cluster = plateau_clusters[prompt_key].get(mid_layer, (None, None))
            if spec_labels is None:
                continue

            for feat_key, feat_info in features_sorted:
                feature_idx = int(feat_key)
                f_stat      = float(feat_info.get("f_stat", 0.0))
                try:
                    r = _steer_one_feature(
                        model=model,
                        tokenizer=tokenizer,
                        crosscoder=crosscoder,
                        prompt_text=prompt_text,
                        prompt_key=prompt_key,
                        steer_layer_model=mid_layer,
                        steer_layer_cc_idx=cc_layer_idx,
                        feature_idx=feature_idx,
                        f_stat=f_stat,
                        cluster_labels=spec_labels,
                        target_cluster=target_cluster,
                        layer_indices=layer_indices,
                        is_albert=is_albert,
                        device=device,
                        config=steer_cfg,
                    )
                    print(f"-> {r.outcome}  "
                          f"merge: {r.merge_layer_baseline} -> {r.merge_layer_perturbed}  "
                          f"delta={r.merge_delta:+d}  alpha={r.alpha:.4f}")
                    results.append(r)
                except Exception as e:
                    import traceback
                    print(f"-> ERROR: {e}")
                    traceback.print_exc()

    return results


# ---------------------------------------------------------------------------
# Part 11: lifetime correspondence analysis
# ---------------------------------------------------------------------------

def analyse_lifetime_correspondence(results: list) -> dict:
    """
    Part 11: test whether lifetime class predicts steering outcome direction.

    Prediction:
      long_lived  -> stabilise   (feature encodes cluster identity/stability)
      short_lived -> destabilise (feature encodes pre-merge instability)

    The prediction is tested with Fisher's exact test on a 2x2 contingency
    table:

                     stabilise   destabilise+null
      long_lived      a               b
      short_lived     c               d

    "no_baseline_merge" experiments are excluded from the table because they
    provide no evidence about direction.
    """
    from scipy.stats import fisher_exact
    from collections import Counter, defaultdict

    by_lt: dict = defaultdict(Counter)
    deltas_long  = []
    deltas_short = []
    excluded = 0

    for r in results:
        if r.outcome == "no_baseline_merge":
            excluded += 1
            continue
        by_lt[r.lifetime_class][r.outcome] += 1
        if r.merge_layer_baseline != -1 and r.merge_layer_perturbed != -1:
            if r.lifetime_class == "long_lived":
                deltas_long.append(r.merge_delta)
            elif r.lifetime_class == "short_lived":
                deltas_short.append(r.merge_delta)

    a = by_lt["long_lived"]["stabilise"]
    b = sum(v for k, v in by_lt["long_lived"].items() if k != "stabilise")
    c = by_lt["short_lived"]["stabilise"]
    d = sum(v for k, v in by_lt["short_lived"].items() if k != "stabilise")
    n_in_table = a + b + c + d

    mean_delta_long  = float(np.mean(deltas_long))  if deltas_long  else float("nan")
    mean_delta_short = float(np.mean(deltas_short)) if deltas_short else float("nan")

    if n_in_table < 4:
        return {
            "error": (
                "Insufficient data for Fisher's exact test "
                f"(n={n_in_table} after excluding {excluded} no_baseline_merge). "
                "Run more steering experiments or ensure both lifetime "
                "classes are represented."
            ),
            "contingency_table": {"long_stabilise": a, "long_other": b,
                                   "short_stabilise": c, "short_other": d},
            "n_excluded_no_baseline_merge": excluded,
            "n_in_table": n_in_table,
            "outcome_by_lifetime": {lt: dict(c2) for lt, c2 in by_lt.items()},
            "mean_delta_long":  mean_delta_long,
            "mean_delta_short": mean_delta_short,
        }

    # Fisher's exact test (one-tailed: OR > 1 = long_lived more likely to stabilise)
    oddsratio, pval_two = fisher_exact([[a, b], [c, d]], alternative="greater")

    confirmed = bool(oddsratio > 1.0 and pval_two < 0.05)

    if confirmed:
        interp = (
            f"Prediction CONFIRMED (OR={oddsratio:.2f}, p={pval_two:.4f}): "
            f"long-lived features are more likely to stabilise the cluster "
            f"(mean delta {mean_delta_long:+.1f} layers) than short-lived features "
            f"(mean delta {mean_delta_short:+.1f} layers). "
            f"Lifetime structure and causal dynamical role are aligned."
        )
    elif oddsratio < 1.0 and pval_two < 0.05:
        interp = (
            f"Prediction REVERSED (OR={oddsratio:.2f}, p={pval_two:.4f}): "
            f"long-lived features are paradoxically more likely to destabilise. "
            f"The lifetime axis may be capturing something other than cluster stability."
        )
    else:
        interp = (
            f"Prediction NULL (OR={oddsratio:.2f}, p={pval_two:.4f}): "
            f"no significant association between lifetime class and steering outcome. "
            f"Possible causes: too few experiments, lifetime axis captures "
            f"syntax/position rather than dynamical role, or the cluster is not "
            f"causally upstream of the merge event."
        )

    return {
        "contingency_table": {
            "long_stabilise":  a,
            "long_other":      b,
            "short_stabilise": c,
            "short_other":     d,
        },
        "odds_ratio":               float(oddsratio),
        "fisher_pval":              float(pval_two),
        "prediction_confirmed":     confirmed,
        "mean_delta_long":          mean_delta_long,
        "mean_delta_short":         mean_delta_short,
        "outcome_by_lifetime":      {lt: dict(c2) for lt, c2 in by_lt.items()},
        "n_excluded_no_baseline_merge": excluded,
        "n_in_table":               n_in_table,
        "interpretation":           interp,
    }


# ---------------------------------------------------------------------------
# Phase-3 addition: pair tracking summary across all experiments
# ---------------------------------------------------------------------------

def analyse_pair_tracking(results: list) -> dict:
    """
    Summarise pair-level causal outcomes across all steering experiments.

    For each outcome class (stabilise/destabilise/null), report the
    mean pair Jaccard similarity and mean pairs broken/formed.
    """
    by_outcome = {}
    for r in results:
        pt = r.pair_tracking if hasattr(r, "pair_tracking") else {}
        if not pt:
            continue
        outcome = r.outcome
        for layer, stats in pt.items():
            by_outcome.setdefault(outcome, []).append(stats)

    summary = {}
    for outcome, stat_list in by_outcome.items():
        jaccards = [s["pair_jaccard"] for s in stat_list]
        broken   = [s["n_broken"]     for s in stat_list]
        formed   = [s["n_formed"]     for s in stat_list]
        summary[outcome] = {
            "n_observations":    len(stat_list),
            "mean_pair_jaccard": float(np.mean(jaccards)),
            "mean_pairs_broken": float(np.mean(broken)),
            "mean_pairs_formed": float(np.mean(formed)),
        }
    return summary


# ---------------------------------------------------------------------------
# Summary (parts 9–10 aggregate)
# ---------------------------------------------------------------------------

def summarise_steering(results: list) -> dict:
    """
    Aggregate steps 9–10 outcomes, then run step-11 lifetime correspondence.

    Returns a dict suitable for JSON serialisation and terminal printing.
    """
    from collections import Counter, defaultdict

    outcomes = Counter(r.outcome for r in results)
    by_lt: dict = defaultdict(Counter)
    deltas = []

    for r in results:
        by_lt[r.lifetime_class][r.outcome] += 1
        if r.merge_layer_baseline != -1 and r.merge_layer_perturbed != -1:
            deltas.append(r.merge_delta)

    mean_delta = float(np.mean(deltas)) if deltas else float("nan")

    # Part 11
    lt_analysis = analyse_lifetime_correspondence(results)

    lines = [
        f"Steering results  ({len(results)} experiments)",
        f"  stabilise:         {outcomes.get('stabilise', 0)}",
        f"  destabilise:       {outcomes.get('destabilise', 0)}",
        f"  null:              {outcomes.get('null', 0)}",
        f"  no_baseline_merge: {outcomes.get('no_baseline_merge', 0)}",
        f"  mean merge delta:  {mean_delta:+.1f} layers",
        "",
        "Lifetime correspondence (part 11):",
    ]
    if "error" in lt_analysis:
        lines.append(f"  {lt_analysis['error']}")
    else:
        ct = lt_analysis["contingency_table"]
        lines += [
            f"  long_lived:  stabilise={ct['long_stabilise']}  "
            f"other={ct['long_other']}  "
            f"mean_delta={lt_analysis['mean_delta_long']:+.1f}",
            f"  short_lived: stabilise={ct['short_stabilise']}  "
            f"other={ct['short_other']}  "
            f"mean_delta={lt_analysis['mean_delta_short']:+.1f}",
            f"  OR={lt_analysis['odds_ratio']:.2f}  p={lt_analysis['fisher_pval']:.4f}",
            f"  {lt_analysis['interpretation']}",
        ]
    for lt, counts in sorted(by_lt.items()):
        lines.append("  " + lt + ": " + "  ".join(
            f"{k}={v}" for k, v in sorted(counts.items())
        ))

    return {
        "n_total":                len(results),
        "outcomes":               dict(outcomes),
        "by_lifetime":            {lt: dict(c) for lt, c in by_lt.items()},
        "merge_deltas":           deltas,
        "mean_merge_delta":       mean_delta,
        "lifetime_correspondence": lt_analysis,
        "text_summary":           "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_steering_results(results: list, summary: dict, path: str | Path):
    """Serialise results + summary to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "results": [dataclasses.asdict(r) for r in results],
            },
            f,
            indent=2,
        )
    print(f"  Steering results saved to {path}")
