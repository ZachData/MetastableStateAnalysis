"""
p5b_manifold/isometry_test.py — Sub-experiment B.

Compute pairwise geodesic distances on Mh and My and test whether they
are correlated (approximate isometry), following Wurgaft §2.3.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from .manifold_fit import eval_manifold, eval_behavior_manifold


# ---------------------------------------------------------------------------
# Distance primitives
# ---------------------------------------------------------------------------

def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Hellinger distance between two probability distributions.

    dH(p, q) = (1/√2) ‖√p − √q‖₂

    Returns float in [0, 1/√2].
    """
    sp = np.sqrt(np.clip(p, 0.0, None))
    sq = np.sqrt(np.clip(q, 0.0, None))
    return float(np.linalg.norm(sp - sq) / np.sqrt(2))


def geodesic_distance_manifold(
    mh:    dict,
    u_i:   float,
    u_j:   float,
    n_pts: int = 150,
) -> float:
    """
    Arc-length distance along Mh between intrinsic coordinates u_i and u_j.

    Discretizes the path into n_pts waypoints, evaluates the spline, and
    accumulates Euclidean distances in PCA space.

    Parameters
    ----------
    mh    : output of fit_activation_manifold
    u_i   : start intrinsic coordinate
    u_j   : end intrinsic coordinate (may wrap for periodic)
    n_pts : number of discretization points

    Returns
    -------
    arc_length : float
    """
    periodic = mh.get("periodic", False)

    if periodic and u_j < u_i:
        # Take the shorter arc accounting for periodicity
        arc_fwd  = _arc_length_segment(mh, u_i, 1.0, n_pts // 2)
        arc_fwd += _arc_length_segment(mh, 0.0, u_j, n_pts // 2)
        arc_bwd  = _arc_length_segment(mh, u_j, u_i, n_pts)
        return min(arc_fwd, arc_bwd)

    return _arc_length_segment(mh, u_i, u_j, n_pts)


def _arc_length_segment(mh: dict, u_start: float, u_end: float, n: int) -> float:
    t   = np.linspace(u_start, u_end, n)
    pts = eval_manifold(mh, t)                  # (n, k)
    diffs  = np.diff(pts, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def geodesic_distance_behavior(
    my:    dict,
    u_i:   float,
    u_j:   float,
    n_pts: int = 150,
) -> float:
    """
    Arc-length distance along My between u_i and u_j, measured in Hellinger units.
    """
    periodic = my.get("periodic", False)

    if periodic and u_j < u_i:
        arc_fwd  = _arc_length_behavior_segment(my, u_i, 1.0, n_pts // 2)
        arc_fwd += _arc_length_behavior_segment(my, 0.0, u_j, n_pts // 2)
        arc_bwd  = _arc_length_behavior_segment(my, u_j, u_i, n_pts)
        return min(arc_fwd, arc_bwd)

    return _arc_length_behavior_segment(my, u_i, u_j, n_pts)


def _arc_length_behavior_segment(
    my: dict, u_start: float, u_end: float, n: int
) -> float:
    t   = np.linspace(u_start, u_end, n)
    pts = eval_behavior_manifold(my, t)            # (n, vocab)
    total = 0.0
    for idx in range(n - 1):
        total += hellinger_distance(pts[idx], pts[idx + 1])
    return total


# ---------------------------------------------------------------------------
# Pairwise distance matrices
# ---------------------------------------------------------------------------

def pairwise_distances(
    mh:        dict,
    my:        dict,
    u_coords:  np.ndarray,
    raw_centroids: np.ndarray,
    n_pts:     int = 150,
) -> dict:
    """
    Compute all pairwise distances between control points.

    Returns
    -------
    dict with:
      d_manifold   : (n_pairs,) — geodesic distances on Mh
      d_behavior   : (n_pairs,) — geodesic distances on My
      d_linear     : (n_pairs,) — Euclidean distances between raw centroids
      n_pairs      : int
      pair_indices : (n_pairs, 2) — (i, j) index pairs
    """
    n = len(u_coords)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pairs = len(pairs)

    d_m  = np.zeros(n_pairs)
    d_b  = np.zeros(n_pairs)
    d_l  = np.zeros(n_pairs)

    for k, (i, j) in enumerate(pairs):
        d_m[k] = geodesic_distance_manifold(mh, u_coords[i], u_coords[j], n_pts)
        d_b[k] = geodesic_distance_behavior(my, u_coords[i], u_coords[j], n_pts)
        d_l[k] = float(np.linalg.norm(raw_centroids[i] - raw_centroids[j]))

    return {
        "d_manifold":   d_m,
        "d_behavior":   d_b,
        "d_linear":     d_l,
        "n_pairs":      n_pairs,
        "pair_indices": np.array(pairs),
    }


# ---------------------------------------------------------------------------
# Isometry score
# ---------------------------------------------------------------------------

def isometry_score(
    d_manifold: np.ndarray,
    d_behavior: np.ndarray,
    d_linear:   np.ndarray,
) -> dict:
    """
    Compute Pearson correlations between distance vectors.

    Returns
    -------
    dict with r_manifold, r_linear, p_manifold, p_linear, n_pairs,
              p5b_b1_pass, p5b_b2_pass, p5b_b3_pass
    """
    r_m, p_m = pearsonr(d_manifold, d_behavior)
    r_l, p_l = pearsonr(d_linear,   d_behavior)
    n = len(d_manifold)

    return {
        "r_manifold":    float(r_m),
        "r_linear":      float(r_l),
        "p_manifold":    float(p_m),
        "p_linear":      float(p_l),
        "n_pairs":       n,
        "p5b_b1_pass":   bool(r_m > r_l),
        "p5b_b2_pass":   bool(r_m > 0.7),
        "p5b_b3_pass":   bool(r_m - r_l > 0.1),
    }


# ---------------------------------------------------------------------------
# MDS visualization helper
# ---------------------------------------------------------------------------

def mds_embed(dist_matrix: np.ndarray, n_dims: int = 2) -> np.ndarray:
    """
    Classical MDS embedding of a square distance matrix.

    Parameters
    ----------
    dist_matrix : (n, n) symmetric distance matrix
    n_dims      : target dimensionality

    Returns
    -------
    coords : (n, n_dims)
    """
    n = dist_matrix.shape[0]
    D2 = dist_matrix ** 2
    H  = np.eye(n) - np.ones((n, n)) / n
    B  = -0.5 * H @ D2 @ H
    vals, vecs = np.linalg.eigh(B)
    # Sort descending
    idx  = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    # Take top n_dims (clip negative eigenvalues)
    vals_pos = np.maximum(vals[:n_dims], 0.0)
    return vecs[:, :n_dims] * np.sqrt(vals_pos)[np.newaxis, :]


def pairwise_to_matrix(d_flat: np.ndarray, pair_indices: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct symmetric (n, n) matrix from flat upper-triangle vector."""
    M = np.zeros((n, n))
    for k, (i, j) in enumerate(pair_indices):
        M[i, j] = M[j, i] = d_flat[k]
    return M


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_isometry_test(
    mh:             dict,
    my:             dict,
    u_coords:       np.ndarray,
    raw_centroids:  np.ndarray,
    n_pts:          int = 150,
) -> dict:
    """
    Run the full isometry test and return results suitable for isometry.json.
    """
    pairs  = pairwise_distances(mh, my, u_coords, raw_centroids, n_pts)
    scores = isometry_score(pairs["d_manifold"], pairs["d_behavior"], pairs["d_linear"])
    n      = len(u_coords)

    # MDS embeddings
    M_manifold = pairwise_to_matrix(pairs["d_manifold"], pairs["pair_indices"], n)
    M_behavior = pairwise_to_matrix(pairs["d_behavior"], pairs["pair_indices"], n)
    M_linear   = pairwise_to_matrix(pairs["d_linear"],   pairs["pair_indices"], n)

    mds_m = mds_embed(M_manifold)
    mds_b = mds_embed(M_behavior)
    mds_l = mds_embed(M_linear)

    return {
        **scores,
        "d_manifold":  pairs["d_manifold"].tolist(),
        "d_behavior":  pairs["d_behavior"].tolist(),
        "d_linear":    pairs["d_linear"].tolist(),
        "pair_indices": pairs["pair_indices"].tolist(),
        "_mds": {
            "manifold": mds_m,
            "behavior": mds_b,
            "linear":   mds_l,
        },
    }
