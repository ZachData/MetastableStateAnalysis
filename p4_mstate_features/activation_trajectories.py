"""
activation_trajectories.py — Track 1: per-token feature activation across layers.

Core question: do crosscoder features track metastable cluster structure
through their activation patterns, even though their decoder directions
don't align with V?

This module computes:
  1. Per-token activation trajectories: tensor (n_features, n_tokens, n_layers)
  2. Feature plateau detection via rolling variance
  3. Feature–cluster mutual information (extends Phase 3's F-statistic)
  4. Feature plateau–cluster plateau alignment test

All functions follow the Phase 3 analysis signature where possible:
    (crosscoder, prompt_store, artifacts, config) -> dict

but some are pure-computation helpers called by chorus.py and analysis.py.
"""

import numpy as np
import torch
from typing import Optional
from scipy import stats
from dataclasses import dataclass

from p3_crosscoder.crosscoder import Crosscoder
from p3_crosscoder.data import PromptActivationStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ActivationTrajectory:
    """Per-token feature activations across layers for one prompt."""
    prompt_key: str
    z_per_layer: np.ndarray       # (n_tokens, n_features, n_layers)
    active_per_layer: np.ndarray  # (n_tokens, n_features, n_layers) bool
    layer_indices: list           # which model layers these correspond to


# ---------------------------------------------------------------------------
# 1. Extract per-token, per-layer activation trajectories
# ---------------------------------------------------------------------------

def extract_activation_trajectories(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    layer_indices: list,
) -> dict[str, ActivationTrajectory]:
    """
    For each prompt, run the crosscoder and reshape activations into
    per-layer feature activation values.

    The crosscoder takes (T, L*d) and produces z of shape (T, F).
    But z is a *single* activation per feature per token — it doesn't
    vary by layer. What varies by layer is whether the feature is
    *relevant* at that layer, measured by the projection of the
    per-layer residual onto the feature's per-layer decoder direction.

    So the "activation trajectory" for feature f at token t across
    layers is: z[t,f] * ||W_dec[l,f,:]|| * cos(angle between
    x[t,l,:] and W_dec[l,f,:]). But since decoder norms are 1
    (normalized), this simplifies to z[t,f] * (x[t,l,:] @ W_dec[l,f,:]).

    We compute two views:
      - z_per_layer[t,f,l] = z[t,f] * proj[t,f,l]  (weighted by relevance)
      - active_per_layer[t,f,l] = z[t,f] > 0        (binary, same across layers)

    The second is what matters for cluster correspondence — a feature
    either fires on a token or it doesn't. The first matters for
    plateau detection — a feature can fire but be irrelevant at some
    layers.
    """
    device = next(crosscoder.parameters()).device
    W_dec = crosscoder.W_dec.detach().cpu().numpy()  # (L, F, d)
    L, F, d = W_dec.shape

    results = {}
    for pk in prompt_store.keys():
        x = prompt_store.get_stacked_tensor(pk)  # (T, L, d)
        T = x.shape[0]

        with torch.no_grad():
            out = crosscoder(x.to(device))
        z = out["z"].cpu().numpy()  # (T, F)
        x_np = x.numpy()           # (T, L, d)

        # z_per_layer: weight each feature's activation by its
        # projection onto the actual residual at that layer
        z_per_layer = np.zeros((T, F, L), dtype=np.float32)
        for l in range(L):
            # proj[t,f] = x[t,l,:] @ W_dec[l,f,:].T
            proj = x_np[:, l, :] @ W_dec[l].T  # (T, F)
            z_per_layer[:, :, l] = z * proj

        active = (z > 0)  # (T, F)
        active_per_layer = np.broadcast_to(
            active[:, :, np.newaxis], (T, F, L)
        ).copy()

        results[pk] = ActivationTrajectory(
            prompt_key=pk,
            z_per_layer=z_per_layer,
            active_per_layer=active_per_layer,
            layer_indices=layer_indices,
        )

    return results


# ---------------------------------------------------------------------------
# 2. Feature plateau detection
# ---------------------------------------------------------------------------

def detect_feature_plateaus(
    traj: ActivationTrajectory,
    var_threshold: float = 0.01,
    min_plateau_len: int = 3,
) -> dict:
    """
    For each feature, find windows of consecutive layers where the
    activation trajectory has low variance across tokens.

    A feature plateau means: this feature's relevance is stable over
    a range of layers. If it also fires on tokens belonging to one
    HDBSCAN cluster, it's a metastable cluster identity feature.

    Parameters
    ----------
    traj : ActivationTrajectory
    var_threshold : max rolling variance to count as plateau
    min_plateau_len : minimum consecutive layers to count

    Returns
    -------
    dict with:
      per_feature: list of {feature_idx, plateaus: [{start, end, length}]}
      summary: {n_features_with_plateaus, mean_plateau_length, ...}
    """
    z = traj.z_per_layer  # (T, F, L)
    T, F, L = z.shape

    # Rolling variance across layers for each feature
    # Use a window of min_plateau_len
    per_feature = []
    total_plateaus = 0
    total_length = 0

    for f in range(F):
        # Check if feature is ever active
        if not traj.active_per_layer[:, f, :].any():
            continue

        # Compute variance of z_per_layer values across layers,
        # using a rolling window
        feat_vals = z[:, f, :]  # (T, L)
        # Mean activation per layer for this feature
        mean_per_layer = feat_vals.mean(axis=0)  # (L,)

        # Find stable windows: consecutive layers where the
        # mean activation doesn't change much
        plateaus = _find_stable_windows(
            mean_per_layer, var_threshold, min_plateau_len
        )

        if plateaus:
            total_plateaus += len(plateaus)
            total_length += sum(p["length"] for p in plateaus)
            per_feature.append({
                "feature_idx": int(f),
                "plateaus": plateaus,
                "n_plateaus": len(plateaus),
                "max_plateau_length": max(p["length"] for p in plateaus),
            })

    n_with = len(per_feature)
    return {
        "per_feature": per_feature,
        "summary": {
            "n_features_with_plateaus": n_with,
            "n_features_total": F,
            "fraction_with_plateaus": n_with / max(F, 1),
            "total_plateaus": total_plateaus,
            "mean_plateau_length": total_length / max(total_plateaus, 1),
        },
    }


def _find_stable_windows(
    signal: np.ndarray,
    var_threshold: float,
    min_len: int,
) -> list:
    """Find maximal runs where rolling variance is below threshold."""
    L = len(signal)
    if L < min_len:
        return []

    # Rolling variance with window = min_len
    plateaus = []
    i = 0
    while i <= L - min_len:
        window = signal[i:i + min_len]
        if np.var(window) < var_threshold:
            # Extend as far as possible
            j = i + min_len
            while j < L:
                extended = signal[i:j + 1]
                if np.var(extended) < var_threshold:
                    j += 1
                else:
                    break
            plateaus.append({
                "start": int(i),
                "end": int(j - 1),
                "length": int(j - i),
            })
            i = j
        else:
            i += 1

    return plateaus


# ---------------------------------------------------------------------------
# 3. Feature–cluster mutual information
# ---------------------------------------------------------------------------

def feature_cluster_mi(
    traj: ActivationTrajectory,
    hdbscan_labels: dict,
    layer_indices: list,
) -> dict:
    """
    At each layer with HDBSCAN labels, compute mutual information
    between each feature's binary activation pattern and cluster
    membership.

    This extends Phase 3's feature_cluster_correlation (F-statistic)
    with an information-theoretic measure. MI is more sensitive to
    nonlinear relationships — a feature could fire on a subset of
    one cluster that the F-statistic misses.

    Parameters
    ----------
    traj : ActivationTrajectory
    hdbscan_labels : dict[str(layer) -> list[int]] from Phase 1
    layer_indices : list of model layer indices matching crosscoder layers

    Returns
    -------
    dict with per_layer results containing MI scores per feature
    """
    z = traj.z_per_layer  # (T, F, L)
    active = traj.active_per_layer  # (T, F, L)
    T, F, L = z.shape

    results = {}
    for layer_key, labels in hdbscan_labels.items():
        labels_arr = np.array(labels)
        if len(labels_arr) != T:
            continue

        # Find which crosscoder layer index this corresponds to
        try:
            model_layer = int(layer_key.replace("layer_", ""))
        except (ValueError, AttributeError):
            continue

        if model_layer not in layer_indices:
            continue
        cc_layer_idx = layer_indices.index(model_layer)

        # Exclude noise points (label -1)
        valid = labels_arr >= 0
        if valid.sum() < 10:
            continue

        labels_valid = labels_arr[valid]
        n_clusters = len(set(labels_valid))
        if n_clusters < 2:
            continue

        feature_mis = []
        for f in range(F):
            feat_active = active[valid, f, cc_layer_idx].astype(int)
            mi = _mutual_information(feat_active, labels_valid)
            # Normalize by min(H(feature), H(cluster))
            h_feat = _entropy(feat_active)
            h_clust = _entropy(labels_valid)
            nmi = mi / max(min(h_feat, h_clust), 1e-10)

            feature_mis.append({
                "feature_idx": int(f),
                "mi": float(mi),
                "nmi": float(nmi),
                "activation_rate": float(feat_active.mean()),
            })

        # Sort by NMI descending
        feature_mis.sort(key=lambda x: x["nmi"], reverse=True)

        results[layer_key] = {
            "n_clusters": n_clusters,
            "n_tokens_valid": int(valid.sum()),
            "top_features": feature_mis[:50],
            "mean_nmi": float(np.mean([x["nmi"] for x in feature_mis])),
            "max_nmi": float(feature_mis[0]["nmi"]) if feature_mis else 0.0,
        }

    return results


def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute MI between discrete variables x and y."""
    # Joint distribution
    xy_pairs = list(zip(x.tolist(), y.tolist()))
    n = len(xy_pairs)
    from collections import Counter
    joint = Counter(xy_pairs)
    x_counts = Counter(x.tolist())
    y_counts = Counter(y.tolist())

    mi = 0.0
    for (xi, yi), count in joint.items():
        p_xy = count / n
        p_x = x_counts[xi] / n
        p_y = y_counts[yi] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log(p_xy / (p_x * p_y))
    return mi


def _entropy(x: np.ndarray) -> float:
    """Shannon entropy of a discrete variable."""
    from collections import Counter
    n = len(x)
    counts = Counter(x.tolist())
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * np.log(p)
    return h


# ---------------------------------------------------------------------------
# 4. Feature plateau–cluster plateau alignment
# ---------------------------------------------------------------------------

def plateau_alignment(
    feature_plateaus: dict,
    cluster_plateaus: list,
    layer_indices: list,
) -> dict:
    """
    Test whether feature activation plateaus overlap with cluster count
    plateaus from Phase 1. This is the original Phase 4 falsification
    criterion.

    Parameters
    ----------
    feature_plateaus : output of detect_feature_plateaus
    cluster_plateaus : list of {start, end, mid, length} from Phase 1
                       (in model layer space)
    layer_indices : crosscoder layer indices (model layer space)

    Returns
    -------
    dict with overlap statistics and the falsification test result
    """
    if not cluster_plateaus or not feature_plateaus.get("per_feature"):
        return {
            "error": "Missing feature or cluster plateaus",
            "falsification": "untestable",
        }

    # Convert cluster plateaus to crosscoder layer space
    cc_cluster_plateaus = []
    for cp in cluster_plateaus:
        start_cc = _model_to_cc_layer(cp["start"], layer_indices)
        end_cc = _model_to_cc_layer(cp["end"], layer_indices)
        if start_cc is not None and end_cc is not None:
            cc_cluster_plateaus.append((start_cc, end_cc))

    if not cc_cluster_plateaus:
        return {"error": "No cluster plateaus map to crosscoder layers"}

    # For each feature plateau, compute overlap with any cluster plateau
    overlaps = []
    n_aligned = 0
    n_total = 0

    for feat_info in feature_plateaus["per_feature"]:
        for fp in feat_info["plateaus"]:
            n_total += 1
            fp_range = set(range(fp["start"], fp["end"] + 1))

            best_overlap = 0.0
            for cp_start, cp_end in cc_cluster_plateaus:
                cp_range = set(range(cp_start, cp_end + 1))
                intersection = len(fp_range & cp_range)
                union = len(fp_range | cp_range)
                iou = intersection / max(union, 1)
                best_overlap = max(best_overlap, iou)

            overlaps.append(best_overlap)
            if best_overlap > 0.3:
                n_aligned += 1

    alignment_rate = n_aligned / max(n_total, 1)

    # Null distribution: random plateau placement
    # Under null, expected overlap depends on coverage fraction
    total_cc_layers = len(layer_indices)
    cluster_coverage = sum(
        e - s + 1 for s, e in cc_cluster_plateaus
    ) / max(total_cc_layers, 1)

    return {
        "n_feature_plateaus": n_total,
        "n_aligned": n_aligned,
        "alignment_rate": float(alignment_rate),
        "mean_overlap_iou": float(np.mean(overlaps)) if overlaps else 0.0,
        "cluster_layer_coverage": float(cluster_coverage),
        "falsification": (
            "pass" if alignment_rate > cluster_coverage + 0.1
            else "fail" if alignment_rate < cluster_coverage - 0.05
            else "inconclusive"
        ),
    }


def _model_to_cc_layer(
    model_layer: int, layer_indices: list
) -> Optional[int]:
    """Map a model layer to the nearest crosscoder layer index."""
    if model_layer in layer_indices:
        return layer_indices.index(model_layer)
    # Find nearest
    dists = [abs(model_layer - li) for li in layer_indices]
    min_dist = min(dists)
    if min_dist <= 2:  # within 2 layers
        return dists.index(min_dist)
    return None


# ---------------------------------------------------------------------------
# 5. Coordinated reorganization at merge layers
# ---------------------------------------------------------------------------

def merge_feature_dynamics(
    traj: ActivationTrajectory,
    merge_layers: list,
    layer_indices: list,
    cluster_identity_features: Optional[dict] = None,
) -> dict:
    """
    At each merge layer, identify features whose activation pattern
    changes sharply. Then test: do the features that die correspond
    to the pre-merge cluster identity features?

    This extends Phase 3's coactivation_at_merges with the causal
    question: are the dying features the ones that *defined* the
    pre-merge clusters?

    Parameters
    ----------
    traj : ActivationTrajectory
    merge_layers : model layers where cluster count drops
    layer_indices : crosscoder layer indices
    cluster_identity_features : optional dict from feature_cluster_mi,
        keyed by layer_key, containing top features per cluster

    Returns
    -------
    dict with per-merge-layer results
    """
    z = traj.z_per_layer  # (T, F, L)
    T, F, L = z.shape

    results = []
    for m_layer in merge_layers:
        cc_idx = _model_to_cc_layer(m_layer, layer_indices)
        if cc_idx is None or cc_idx < 1 or cc_idx >= L:
            continue

        # Change in feature relevance across the merge
        pre = z[:, :, cc_idx - 1]   # (T, F)
        post = z[:, :, cc_idx]      # (T, F)

        # Per-feature: mean absolute change across tokens
        delta = np.abs(post - pre).mean(axis=0)  # (F,)

        # Features that die: active pre, inactive post
        pre_active = (np.abs(pre) > 1e-6).mean(axis=0)   # (F,)
        post_active = (np.abs(post) > 1e-6).mean(axis=0)  # (F,)
        dying = (pre_active > 0.1) & (post_active < 0.05)
        born = (pre_active < 0.05) & (post_active > 0.1)

        dying_idx = np.where(dying)[0].tolist()
        born_idx = np.where(born)[0].tolist()

        # Top changed features
        top_changed = np.argsort(delta)[-20:].tolist()

        entry = {
            "merge_layer": int(m_layer),
            "cc_layer_idx": int(cc_idx),
            "n_dying": len(dying_idx),
            "n_born": len(born_idx),
            "dying_features": dying_idx[:50],
            "born_features": born_idx[:50],
            "top_changed_features": top_changed,
            "mean_delta": float(delta.mean()),
            "max_delta": float(delta.max()),
        }

        # If we have cluster identity features, check overlap
        if cluster_identity_features:
            pre_layer_key = f"layer_{layer_indices[cc_idx - 1]}"
            if pre_layer_key in cluster_identity_features:
                ci_top = cluster_identity_features[pre_layer_key]
                ci_feats = set(
                    f["feature_idx"] for f in ci_top.get("top_features", [])[:20]
                )
                overlap = len(ci_feats & set(dying_idx))
                entry["cluster_identity_overlap"] = {
                    "n_ci_features": len(ci_feats),
                    "n_dying_that_were_ci": overlap,
                    "fraction": overlap / max(len(ci_feats), 1),
                }

        results.append(entry)

    return {
        "per_merge": results,
        "summary": {
            "n_merges_analyzed": len(results),
            "mean_dying": float(np.mean([r["n_dying"] for r in results]))
            if results else 0.0,
            "mean_born": float(np.mean([r["n_born"] for r in results]))
            if results else 0.0,
        },
    }
