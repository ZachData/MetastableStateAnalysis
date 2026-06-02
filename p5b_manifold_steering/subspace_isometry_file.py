"""
p5b_manifold/subspace_isometry.py — Sub-experiment D.

Test whether the S (real/symmetric) subspace of V gives better isometry
with My than the full activation space or the A (antisymmetric) subspace.

This is the direct connection to Phase 6's prediction: metastable cluster
structure lives in S. If true, Wurgaft's Mh should have higher isometry
with My when measured in S-projected coordinates than in A-projected ones.

Imports from existing project scripts:
  p6_subspace.subspace_build : build_global_projectors (optional; projectors
  are passed in as numpy arrays, so this module has no hard dependency on p6)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def project_centroids(centroids: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Project centroids onto the column space of U.

    Parameters
    ----------
    centroids : (n, d)
    U         : (d, k) orthonormal basis

    Returns
    -------
    projected : (n, k) — coordinates in the basis U
    """
    return centroids @ U


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """Upper-triangle pairwise Euclidean distances as flat (n_pairs,) vector."""
    n = len(X)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(float(np.linalg.norm(X[i] - X[j])))
    return np.array(out, dtype=np.float64)


def subspace_isometry_score(
    centroids:  np.ndarray,
    U_S:        np.ndarray,
    U_A:        np.ndarray,
    d_behavior: np.ndarray,
) -> dict:
    """
    Compare isometry with My for S-projected, A-projected, and full centroids.

    Parameters
    ----------
    centroids   : (n, d) — raw cluster centroids (not PCA-reduced)
    U_S         : (d, k_s) — real/symmetric subspace basis
                  (U_pos ∪ U_neg from Phase 2 projectors)
    U_A         : (d, k_a) — imaginary/antisymmetric subspace basis
    d_behavior  : (n_pairs,) — geodesic distances on My (from isometry_test)

    Returns
    -------
    dict with:
      r_S, r_A, r_full, r_linear : Pearson correlations with d_behavior
      p_S, p_A, p_full           : p-values
      n_pairs                    : int
      p5b_d1_pass : r_S > r_full ≥ r_A
      p5b_d2_pass : |r_A − r_linear| < 0.05
    """
    c_S = project_centroids(centroids, U_S)   # (n, k_s)
    c_A = project_centroids(centroids, U_A)   # (n, k_a)

    d_S    = _pairwise_euclidean(c_S)
    d_A    = _pairwise_euclidean(c_A)
    d_full = _pairwise_euclidean(centroids)

    n_pairs = len(d_behavior)
    assert len(d_S) == n_pairs, (
        f"d_S length {len(d_S)} != d_behavior length {n_pairs}"
    )

    r_S, p_S = pearsonr(d_S,    d_behavior)
    r_A, p_A = pearsonr(d_A,    d_behavior)
    r_f, p_f = pearsonr(d_full, d_behavior)
    # r_linear uses raw Euclidean in the full ambient space — same as r_full
    r_l = r_f
    p_l = p_f

    return {
        "r_S":         float(r_S),
        "r_A":         float(r_A),
        "r_full":      float(r_f),
        "r_linear":    float(r_l),
        "p_S":         float(p_S),
        "p_A":         float(p_A),
        "p_full":      float(p_f),
        "n_pairs":     n_pairs,
        "p5b_d1_pass": bool(r_S > r_f and r_f >= r_A),
        "p5b_d2_pass": bool(abs(r_A - r_l) < 0.05),
    }
