"""
p5b_manifold_steering/subspace_isometry.py — Sub-experiment D.

Test whether the S (real/symmetric) subspace of V gives better isometry
with My than the full activation space or the A (antisymmetric) subspace.

This is the direct connection to Phase 6's prediction: metastable cluster
structure lives in S. If true, Wurgaft's Mh should have higher isometry
with My when measured in S-projected coordinates than in A-projected ones.

Imports from existing project scripts:
  p6_subspace.subspace_build : build_global_projectors (optional; projectors
  are passed in as numpy arrays, so this module has no hard dependency on p6)

--------------------------------------------------------------------------
FIX 2026-07-21 — r_linear was an alias, not a control.

The previous implementation computed `r_l = r_f`, i.e. r_linear was set
equal to r_full by assignment. That made P5b-D2 (|r_A − r_linear| < 0.05)
read as |r_A − r_full| < 0.05, which is in direct tension with P5b-D1
(r_S > r_full >= r_A): D1 wants r_full above r_A, D2 wants them within
0.05 of each other. Neither criterion had an independent reference.

This is the same degeneracy Sub-exp B had, and it is resolved the same
way: the control is the AMBIENT residual stream — Euclidean distance
between UN-normalized centroids — not another view of the same normalized
ones. `centroids_raw` is now an explicit parameter. When it is omitted the
old aliasing behaviour is preserved so existing callers do not change
meaning silently, but `r_linear_is_alias: True` is set in the output so a
reader can tell which regime produced the number.
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
    X = np.asarray(X, dtype=np.float64)
    iu = np.triu_indices(X.shape[0], k=1)
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)[iu]


def _corr(x, y):
    """
    Pearson r, or None when undefined.

    A degenerate subspace collapses every distance to the same value; scipy
    then returns nan with a ConstantInputWarning rather than raising. nan
    is not valid JSON and `bool(nan > t)` is silently False, so an
    undefined correlation is reported as None and every criterion using it
    fails explicitly.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None, None
    r, p = pearsonr(x, y)
    if not np.isfinite(r):
        return None, None
    return float(r), float(p)


def subspace_isometry_score(
    centroids:     np.ndarray,
    U_S:           np.ndarray,
    U_A:           np.ndarray,
    d_behavior:    np.ndarray,
    centroids_raw: np.ndarray | None = None,
) -> dict:
    """
    Compare isometry with My for S-projected, A-projected, and full centroids.

    Parameters
    ----------
    centroids     : (n, d) — cluster centroids in the SPHERE frame, i.e.
                    L2-normalized, restricted to the same point set and
                    order as `d_behavior`. (The previous docstring said
                    "raw cluster centroids"; that was never what the caller
                    passed — load_plateau_centroids has always returned
                    L2-normalized rows. Corrected rather than changed.)
    U_S           : (d, k_s) — real/symmetric subspace basis
                    (U_pos ∪ U_neg from Phase 2 projectors)
    U_A           : (d, k_a) — imaginary/antisymmetric subspace basis
    d_behavior    : (n_pairs,) — behaviour distances, Hellinger, over the
                    same points in the same order
    centroids_raw : (n, d) optional — UN-normalized ambient centroids, the
                    genuine linear control. Omit only if unavailable.

    DIMENSION CAVEAT: r_S and r_A are only comparable when k_s and k_a are
    comparable. A larger subspace captures more of any geometry regardless
    of which subspace it is, so an unequal-dimension comparison confounds
    subspace identity with subspace size. `dim_S`/`dim_A` are reported so
    this is visible in the artifact; equalize them upstream where possible.

    Returns
    -------
    dict with r_S, r_A, r_full, r_linear, p-values, dims, n_pairs,
    r_linear_is_alias, p5b_d1_pass, p5b_d2_pass.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    d_behavior = np.asarray(d_behavior, dtype=np.float64)

    n_pairs = int(len(d_behavior))
    expected = centroids.shape[0] * (centroids.shape[0] - 1) // 2
    if expected != n_pairs:
        raise ValueError(
            f"subspace_isometry_score: {centroids.shape[0]} centroids imply "
            f"{expected} pairs but d_behavior has {n_pairs}. These must be "
            f"the same point set in the same order."
        )

    d_S    = _pairwise_euclidean(project_centroids(centroids, U_S))
    d_A    = _pairwise_euclidean(project_centroids(centroids, U_A))
    d_full = _pairwise_euclidean(centroids)

    r_S, p_S = _corr(d_S,    d_behavior)
    r_A, p_A = _corr(d_A,    d_behavior)
    r_f, p_f = _corr(d_full, d_behavior)

    is_alias = centroids_raw is None
    if is_alias:
        r_l, p_l = r_f, p_f
    else:
        d_lin = _pairwise_euclidean(np.asarray(centroids_raw, dtype=np.float64))
        r_l, p_l = _corr(d_lin, d_behavior)

    d1 = bool(
        r_S is not None and r_f is not None and r_A is not None
        and r_S > r_f and r_f >= r_A
    )
    d2 = bool(
        r_A is not None and r_l is not None and abs(r_A - r_l) < 0.05
    )

    return {
        "r_S":               r_S,
        "r_A":               r_A,
        "r_full":            r_f,
        "r_linear":          r_l,
        "p_S":               p_S,
        "p_A":               p_A,
        "p_full":            p_f,
        "p_linear":          p_l,
        "dim_S":             int(np.asarray(U_S).shape[1]),
        "dim_A":             int(np.asarray(U_A).shape[1]),
        "n_pairs":           n_pairs,
        "r_linear_is_alias": bool(is_alias),
        "p5b_d1_pass":       d1,
        "p5b_d2_pass":       d2,
    }
