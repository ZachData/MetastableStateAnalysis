"""
chorus.py — Track 1: co-activation cliques and cluster correspondence.

The chorus hypothesis: individual crosscoder features may be random w.r.t.
clusters, but *sets* of co-active features may not be. Cluster identity
is encoded in the joint activation pattern, not in any single feature.

This module:
  1. Computes co-activation matrices at each layer
  2. Extracts feature cliques (connected components at threshold)
  3. Tests whether tokens activating the same clique share a cluster
  4. Quantifies chorus–cluster correspondence via adjusted Rand index
"""

import numpy as np
from collections import Counter
from typing import Optional

from phase3.crosscoder import Crosscoder
from phase3.data import PromptActivationStore

from .activation_trajectories import (
    ActivationTrajectory,
    extract_activation_trajectories,
    _model_to_cc_layer,
)


# ---------------------------------------------------------------------------
# 1. Co-activation matrix
# ---------------------------------------------------------------------------

def compute_coactivation(
    traj: ActivationTrajectory,
    cc_layer_idx: int,
    active_threshold: float = 0.0,
) -> np.ndarray:
    """
    Co-activation matrix for features at a given crosscoder layer.

    Entry (i,j) = fraction of tokens where both feature i and feature j
    have |z_per_layer| > active_threshold, among tokens where at least
    one of them is active.

    Parameters
    ----------
    traj : ActivationTrajectory
    cc_layer_idx : index into the crosscoder's layer dimension
    active_threshold : minimum |activation| to count as active

    Returns
    -------
    coact : (n_features, n_features) symmetric matrix, values in [0,1]
    """
    z = traj.z_per_layer[:, :, cc_layer_idx]  # (T, F)
    active = (np.abs(z) > active_threshold).astype(np.float32)  # (T, F)
    T, F = active.shape

    # Co-activation: (F, F) = active.T @ active / T
    coact = (active.T @ active) / max(T, 1)
    np.fill_diagonal(coact, 0.0)

    return coact


# ---------------------------------------------------------------------------
# 2. Extract cliques
# ---------------------------------------------------------------------------

def extract_cliques(
    coact: np.ndarray,
    threshold: float = 0.3,
    min_clique_size: int = 2,
) -> list[list[int]]:
    """
    Connected components of the co-activation graph thresholded at
    `threshold`. Each component is a "chorus" — a set of features
    that tend to fire together.

    Parameters
    ----------
    coact : (F, F) co-activation matrix
    threshold : minimum co-activation to draw an edge
    min_clique_size : discard singletons

    Returns
    -------
    List of cliques, each a list of feature indices, sorted by size desc.
    """
    adj = coact > threshold
    F = adj.shape[0]
    visited = set()
    components = []

    for start in range(F):
        if start in visited:
            continue
        # BFS
        comp = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            for nb in range(F):
                if adj[node, nb] and nb not in visited:
                    queue.append(nb)
        if len(comp) >= min_clique_size:
            components.append(sorted(comp))

    components.sort(key=len, reverse=True)
    return components


# ---------------------------------------------------------------------------
# 3. Clique–cluster correspondence
# ---------------------------------------------------------------------------

def clique_cluster_correspondence(
    traj: ActivationTrajectory,
    cliques: list[list[int]],
    labels: np.ndarray,
    cc_layer_idx: int,
    active_threshold: float = 0.0,
) -> dict:
    """
    For each clique, identify which tokens activate *all* features in
    the clique. Then check whether those tokens predominantly belong
    to one HDBSCAN cluster.

    Parameters
    ----------
    traj : ActivationTrajectory
    cliques : list of feature index lists (from extract_cliques)
    labels : (T,) HDBSCAN cluster labels for this layer
    cc_layer_idx : crosscoder layer index
    active_threshold : min activation to count

    Returns
    -------
    dict with per-clique purity scores and overall statistics
    """
    z = traj.z_per_layer[:, :, cc_layer_idx]  # (T, F)
    active = np.abs(z) > active_threshold
    T = z.shape[0]

    valid = labels >= 0  # exclude noise
    if valid.sum() < 10:
        return {"error": "Too few valid tokens"}

    per_clique = []
    for ci, clique in enumerate(cliques):
        # Tokens where ALL features in the clique are active
        all_active = active[:, clique].all(axis=1)  # (T,)
        # Intersect with valid labels
        mask = all_active & valid
        n_tokens = int(mask.sum())

        if n_tokens < 3:
            per_clique.append({
                "clique_idx": ci,
                "clique_size": len(clique),
                "n_tokens": n_tokens,
                "purity": 0.0,
                "dominant_cluster": -1,
            })
            continue

        clique_labels = labels[mask]
        counts = Counter(clique_labels.tolist())
        dominant = counts.most_common(1)[0]
        purity = dominant[1] / n_tokens

        per_clique.append({
            "clique_idx": ci,
            "clique_size": len(clique),
            "features": clique,
            "n_tokens": n_tokens,
            "purity": float(purity),
            "dominant_cluster": int(dominant[0]),
            "cluster_distribution": {
                int(k): int(v) for k, v in counts.items()
            },
        })

    purities = [c["purity"] for c in per_clique if c["n_tokens"] >= 3]

    return {
        "per_clique": per_clique,
        "summary": {
            "n_cliques": len(cliques),
            "n_cliques_with_tokens": sum(
                1 for c in per_clique if c["n_tokens"] >= 3
            ),
            "mean_purity": float(np.mean(purities)) if purities else 0.0,
            "median_purity": float(np.median(purities)) if purities else 0.0,
            "n_high_purity": sum(1 for p in purities if p > 0.8),
        },
    }


# ---------------------------------------------------------------------------
# 4. Token-to-clique assignment → adjusted Rand index vs clusters
# ---------------------------------------------------------------------------

def chorus_cluster_ari(
    traj: ActivationTrajectory,
    cliques: list[list[int]],
    labels: np.ndarray,
    cc_layer_idx: int,
    active_threshold: float = 0.0,
) -> dict:
    """
    Assign each token to its best-matching clique (the clique with
    the most active features for that token), then compute the
    adjusted Rand index between clique assignments and HDBSCAN labels.

    ARI > 0 means clique structure agrees with cluster structure
    beyond chance. This is the strongest test of the chorus hypothesis.

    Parameters
    ----------
    traj : ActivationTrajectory
    cliques : list of feature index lists
    labels : (T,) HDBSCAN labels
    cc_layer_idx : crosscoder layer index

    Returns
    -------
    dict with ARI and supporting statistics
    """
    z = traj.z_per_layer[:, :, cc_layer_idx]  # (T, F)
    active = np.abs(z) > active_threshold
    T = z.shape[0]

    valid = labels >= 0
    if valid.sum() < 10 or not cliques:
        return {"error": "Insufficient data", "ari": 0.0}

    # Assign each token to the clique with most active features
    # -1 = no clique matches well
    assignments = np.full(T, -1, dtype=int)
    for t in range(T):
        if not valid[t]:
            continue
        best_score = 0
        best_ci = -1
        for ci, clique in enumerate(cliques):
            score = active[t, clique].sum()
            if score > best_score:
                best_score = score
                best_ci = ci
        if best_score >= 2:  # require at least 2 features active
            assignments[t] = best_ci

    # Filter to tokens with valid labels AND clique assignments
    both_valid = valid & (assignments >= 0)
    if both_valid.sum() < 10:
        return {"error": "Too few tokens with both labels", "ari": 0.0}

    ari = _adjusted_rand_index(
        assignments[both_valid], labels[both_valid]
    )

    return {
        "ari": float(ari),
        "n_tokens_matched": int(both_valid.sum()),
        "n_cliques_used": len(set(assignments[both_valid].tolist())),
        "n_clusters": len(set(labels[both_valid].tolist())),
    }


def _adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Compute ARI between two integer label arrays."""
    from collections import Counter
    n = len(a)
    if n < 2:
        return 0.0

    # Contingency table
    pairs = list(zip(a.tolist(), b.tolist()))
    contingency = Counter(pairs)

    # Row and column sums
    a_counts = Counter(a.tolist())
    b_counts = Counter(b.tolist())

    def comb2(x):
        return x * (x - 1) / 2

    sum_comb_nij = sum(comb2(v) for v in contingency.values())
    sum_comb_ai = sum(comb2(v) for v in a_counts.values())
    sum_comb_bj = sum(comb2(v) for v in b_counts.values())
    comb_n = comb2(n)

    if comb_n == 0:
        return 0.0

    expected = sum_comb_ai * sum_comb_bj / comb_n
    max_index = 0.5 * (sum_comb_ai + sum_comb_bj)
    denom = max_index - expected

    if abs(denom) < 1e-12:
        return 0.0 if abs(sum_comb_nij - expected) < 1e-12 else 1.0

    return (sum_comb_nij - expected) / denom


# ---------------------------------------------------------------------------
# 5. Full chorus analysis for one prompt at one layer
# ---------------------------------------------------------------------------

def analyze_chorus_at_layer(
    traj: ActivationTrajectory,
    labels: np.ndarray,
    cc_layer_idx: int,
    coact_threshold: float = 0.3,
    min_clique_size: int = 2,
    active_threshold: float = 0.0,
) -> dict:
    """
    Run the full chorus pipeline at a single layer:
    co-activation → cliques → purity → ARI.

    Returns combined results dict.
    """
    coact = compute_coactivation(traj, cc_layer_idx, active_threshold)
    cliques = extract_cliques(coact, coact_threshold, min_clique_size)

    if not cliques:
        return {
            "n_cliques": 0,
            "purity": {"summary": {"mean_purity": 0.0}},
            "ari": {"ari": 0.0},
            "coact_density": 0.0,
        }

    purity = clique_cluster_correspondence(
        traj, cliques, labels, cc_layer_idx, active_threshold
    )
    ari = chorus_cluster_ari(
        traj, cliques, labels, cc_layer_idx, active_threshold
    )

    # Co-activation graph density
    F = coact.shape[0]
    n_edges = (coact > coact_threshold).sum() / 2
    max_edges = F * (F - 1) / 2
    density = n_edges / max(max_edges, 1)

    return {
        "n_cliques": len(cliques),
        "largest_clique": len(cliques[0]) if cliques else 0,
        "clique_sizes": [len(c) for c in cliques],
        "purity": purity,
        "ari": ari,
        "coact_density": float(density),
    }


# ---------------------------------------------------------------------------
# 6. Sweep co-activation thresholds
# ---------------------------------------------------------------------------

def sweep_thresholds(
    traj: ActivationTrajectory,
    labels: np.ndarray,
    cc_layer_idx: int,
    thresholds: Optional[list] = None,
) -> list[dict]:
    """
    Run chorus analysis at multiple co-activation thresholds.
    Useful for finding the threshold that maximizes ARI — if
    no threshold gives ARI > 0.1, the chorus hypothesis fails
    at this layer.
    """
    if thresholds is None:
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    results = []
    for th in thresholds:
        r = analyze_chorus_at_layer(traj, labels, cc_layer_idx, th)
        r["threshold"] = th
        results.append(r)

    return results
