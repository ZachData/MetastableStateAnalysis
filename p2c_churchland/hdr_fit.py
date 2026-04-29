"""
hdr_fit.py — C1 follow-up: HDR if jPCA results are borderline.

Implements the Lara, Cunningham & Churchland (2018) Hierarchical Decomposition
of Rotations (HDR). HDR avoids the cross-condition mean removal step that jPCA
requires, making it more robust when:
  - The number of conditions (prompts) is small
  - The mean trajectory itself carries rotational structure
  - jPCA gives a borderline R² ratio (0.3–0.5, flagged by p2cj1_marginal=True)

Run jPCA first. Run HDR only when jpca_result["p2cj1_marginal"] is True or
when the mean-removal step is suspected to be distorting the result.

HDR algorithm (simplified implementation)
------------------------------------------
1. For each condition c separately, fit ẋ_c = M_c x_c with M_c skew-symmetric.
   Each M_c is the per-condition skew-symmetric fit (no cross-condition mean removed).
2. Find the consensus rotation plane P* that maximizes:
       Σ_c ||P* M_c P*||²_F
   subject to P* being a rank-2 orthogonal projector.
   This is solved via a power-iteration / SVD approach on the aggregated
   antisymmetric operator.
3. Project data onto P* and compute the explained variance ratio (analog of R²).

Predictions re-tested (if jPCA was borderline):
  P2c-J1 : HDR explained variance ratio > 0.5
  P2c-J2 : HDR consensus plane within 30° of U_A (via jpca_alignment.principal_angles)

Functions
---------
fit_single_condition_skew  : per-condition skew-symmetric fit
consensus_rotation_plane   : find P* via aggregated SVD
hdr_variance_ratio         : explained variance ratio for P*
fit_hdr                    : full pipeline
hdr_to_json                : JSON-serializable summary
"""

from __future__ import annotations

import numpy as np

from p2c_churchland.jpca_fit import (
    pca_reduce,
    build_regression_mats,
    fit_unconstrained,
    fit_skew_symmetric,
    r2_score,
    extract_rotation_planes,
)
from p2c_churchland.jpca_alignment import principal_angles


# ---------------------------------------------------------------------------
# Per-condition skew-symmetric fit
# ---------------------------------------------------------------------------

def fit_single_condition_skew(
    x_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit ẋ = M_c x with M_c skew-symmetric for a single condition's trajectory.

    Parameters
    ----------
    x_c : (T, p) — trajectory for one condition in PC space

    Returns
    -------
    M_c    : (p, p) skew-symmetric fit for this condition
    X_reg  : (p, T-1) state matrix
    dX_reg : (p, T-1) velocity matrix
    """
    T, p = x_c.shape
    # Build single-condition regression matrices directly
    states = [x_c[t]     for t in range(T - 1)]
    vels   = [x_c[t+1] - x_c[t] for t in range(T - 1)]
    X_reg  = np.column_stack(states)   # (p, T-1)
    dX_reg = np.column_stack(vels)     # (p, T-1)

    M_unc = fit_unconstrained(X_reg, dX_reg)
    M_c   = fit_skew_symmetric(M_unc)
    return M_c, X_reg, dX_reg


# ---------------------------------------------------------------------------
# Consensus rotation plane
# ---------------------------------------------------------------------------

def consensus_rotation_plane(
    M_list: list[np.ndarray],
) -> np.ndarray:
    """
    Find the rank-2 consensus rotation plane P* that maximizes
        Σ_c ||P M_c P||²_F
    over all rank-2 orthogonal projectors P.

    Equivalent to finding the 2D subspace that captures the most
    antisymmetric dynamics summed across conditions.

    Solution: form the aggregated matrix
        G = Σ_c M_c^T M_c
    and take the top-2 eigenvectors of G as the consensus plane.

    Parameters
    ----------
    M_list : list of (p, p) skew-symmetric matrices, one per condition

    Returns
    -------
    plane : (p, 2) orthonormal basis for the consensus rotation plane
    """
    if not M_list:
        raise ValueError("M_list is empty.")

    p = M_list[0].shape[0]
    G = np.zeros((p, p), dtype=np.float64)

    for M_c in M_list:
        G += M_c.T @ M_c     # M^T M = -M M for skew-symmetric; positive semi-definite

    # Top-2 eigenvectors of G (symmetric PSD matrix)
    eigvals, eigvecs = np.linalg.eigh(G)   # ascending order
    # Take the two with largest eigenvalues
    idx = np.argsort(eigvals)[::-1][:2]
    plane = eigvecs[:, idx]                # (p, 2)

    # Orthonormalize
    plane, _ = np.linalg.qr(plane)
    return plane[:, :2]


# ---------------------------------------------------------------------------
# Explained variance ratio for HDR
# ---------------------------------------------------------------------------

def hdr_variance_ratio(
    plane_pc: np.ndarray,
    M_list: list[np.ndarray],
    X_list: list[np.ndarray],
    dX_list: list[np.ndarray],
) -> dict:
    """
    Compute how much of each condition's rotational dynamics is captured
    by the consensus plane P*.

    For each condition c:
      M_c_proj = P* M_c P*     (projection of M_c onto the consensus plane)
      r2_proj  = R² of M_c_proj on condition c's data
      r2_full  = R² of M_c (full per-condition fit) on condition c's data

    Ratio = mean(r2_proj) / mean(r2_full)

    Parameters
    ----------
    plane_pc : (p, 2) consensus rotation plane basis
    M_list   : list of (p, p) per-condition skew-symmetric fits
    X_list   : list of (p, N_c) per-condition state matrices
    dX_list  : list of (p, N_c) per-condition velocity matrices

    Returns
    -------
    dict with per-condition r2s and the aggregate ratio
    """
    P = plane_pc @ plane_pc.T    # (p, p) rank-2 projector

    r2_proj_list, r2_full_list = [], []
    for M_c, X_c, dX_c in zip(M_list, X_list, dX_list):
        M_proj = P @ M_c @ P
        r2_proj_list.append(r2_score(M_proj, X_c, dX_c))
        r2_full_list.append(r2_score(M_c,    X_c, dX_c))

    mean_r2_proj = float(np.mean(r2_proj_list))
    mean_r2_full = float(np.mean(r2_full_list))
    #prevent unconstrained R2 from small data
    if mean_r2_full <= 0.0:
        ratio = 0.0
    else:
        ratio = float(np.clip(mean_r2_proj / mean_r2_full, 0.0, 1.0))

    return {
        "per_condition_r2_proj": r2_proj_list,
        "per_condition_r2_full": r2_full_list,
        "mean_r2_proj":          mean_r2_proj,
        "mean_r2_full":          mean_r2_full,
        "variance_ratio":        ratio,
    }


# ---------------------------------------------------------------------------
# Full HDR pipeline
# ---------------------------------------------------------------------------

def fit_hdr(
    activations: np.ndarray,
    n_pc: int = 6,
    ua_planes: list[np.ndarray] | None = None,
    angle_threshold_deg: float = 30.0,
) -> dict:
    """
    Full HDR pipeline.

    Parameters
    ----------
    activations         : (n_cond, n_layers, d) — same format as fit_jpca.
                          No mean removal is applied here.
    n_pc                : number of PCs
    ua_planes           : optional list of (d, 2) U_A planes from p2b for J2 check
    angle_threshold_deg : threshold for P2c-J2 verdict

    Returns
    -------
    dict with:
      M_list              : list of per-condition skew-symmetric fits (in PC space)
      plane_pc            : (p, 2) consensus rotation plane in PC space
      plane_full          : (d, 2) consensus plane lifted to full d-space
      variance_ratio      : float (analog of jPCA R² ratio for HDR)
      p2cj1_hdr_holds     : bool — variance_ratio > 0.5
      p2cj2_hdr_holds     : bool — if ua_planes given, angle to UA < threshold
      ua_min_angle        : float — min principal angle to U_A (if ua_planes given)
      V_pc                : (d, p) PC basis
      var_explained       : (p,) variance fractions
      variance_breakdown  : full dict from hdr_variance_ratio
    """
    n_cond, n_layers, d = activations.shape
    assert n_layers >= 2, "Need at least 2 layers."

    # PCA (no mean removal)
    X_pc, V_pc, var_exp = pca_reduce(activations, n_components=n_pc)

    # Per-condition fits
    M_list, X_list, dX_list = [], [], []
    for c in range(n_cond):
        M_c, X_c, dX_c = fit_single_condition_skew(X_pc[c])
        M_list.append(M_c)
        X_list.append(X_c)
        dX_list.append(dX_c)

    # Consensus plane
    plane_pc   = consensus_rotation_plane(M_list)        # (p, 2)
    plane_full = V_pc @ plane_pc                          # (d, 2)

    # Re-orthonormalize in full space
    q1 = plane_full[:, 0] / max(np.linalg.norm(plane_full[:, 0]), 1e-12)
    q2 = plane_full[:, 1] - np.dot(plane_full[:, 1], q1) * q1
    q2 /= max(np.linalg.norm(q2), 1e-12)
    plane_full = np.column_stack([q1, q2])

    # Variance ratio
    vbkdn = hdr_variance_ratio(plane_pc, M_list, X_list, dX_list)

    # U_A alignment
    ua_min_angle = float("nan")
    p2cj2_holds  = False
    if ua_planes:
        angles = [principal_angles(plane_full, ua)[0] for ua in ua_planes]
        ua_min_angle = float(np.min(angles))
        p2cj2_holds  = ua_min_angle < angle_threshold_deg

    return {
        "M_list":           M_list,
        "plane_pc":         plane_pc,
        "plane_full":       plane_full,
        "variance_ratio":   vbkdn["variance_ratio"],
        "p2cj1_hdr_holds":  vbkdn["variance_ratio"] > 0.5,
        "p2cj2_hdr_holds":  p2cj2_holds,
        "ua_min_angle":     ua_min_angle,
        "V_pc":             V_pc,
        "var_explained":    var_exp,
        "variance_breakdown": vbkdn,
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def hdr_to_json(result: dict) -> dict:
    vbkdn = result["variance_breakdown"]
    return {
        "variance_ratio":        float(result["variance_ratio"]),
        "p2cj1_hdr_holds":       bool(result["p2cj1_hdr_holds"]),
        "p2cj2_hdr_holds":       bool(result["p2cj2_hdr_holds"]),
        "ua_min_angle":          float(result["ua_min_angle"]),
        "var_explained":         [float(v) for v in result["var_explained"]],
        "plane_pc":              result["plane_pc"].tolist(),
        "mean_r2_proj":          float(vbkdn["mean_r2_proj"]),
        "mean_r2_full":          float(vbkdn["mean_r2_full"]),
        "per_condition_r2_proj": [float(x) for x in vbkdn["per_condition_r2_proj"]],
        "per_condition_r2_full": [float(x) for x in vbkdn["per_condition_r2_full"]],
    }
