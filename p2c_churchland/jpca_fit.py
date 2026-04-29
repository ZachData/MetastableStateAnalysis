"""
jpca_fit.py — C1: jPCA on layer-to-layer Δx, plane extraction, frequency, R² ratio.

Implements the Churchland et al. 2012 jPCA fit ported to transformer layer
trajectories. The "time" axis is layer depth; the "conditions" are prompts.

Pipeline
--------
1. Stack per-prompt, per-layer mean activations: X (n_cond, n_layers, d)
2. Project to top-p PC subspace (default p=6) across all conditions × layers.
3. Remove cross-condition mean at each layer (required by jPCA; this is what
   HDR avoids — see hdr_fit.py for the alternative).
4. Compute Δx̃ = x̃^{L+1} - x̃^{L} in PC space.
5. Fit ẋ = M x unconstrained → M_unc; constrained to skew-symmetric → M_skew.
6. Compute R²_unc, R²_skew, R² ratio = R²_skew / R²_unc.
7. Eigendecompose M_skew → pure imaginary eigenvalues ±iω → rotation planes.
8. Lift rotation planes back to original d-dimensional space via PC basis.

Predictions tested (via jpca_alignment.py for J2):
  P2c-J1 : R² ratio > 0.5 in at least one model
  P2c-J2 : top jPCA planes within 30° of U_A (tested in jpca_alignment.py)

Functions
---------
pca_reduce             : project (n_cond, n_layers, d) → PC subspace
remove_condition_mean  : subtract cross-condition mean at each layer
build_regression_mats  : assemble X and ΔX matrices for regression
fit_unconstrained      : least-squares M such that ΔX ≈ M X
fit_skew_symmetric     : project M_unc to nearest skew-symmetric matrix
r2_score               : R² of a linear prediction ΔX̂ = M X
extract_rotation_planes: eigendecompose M_skew → planes in PC and full space
fit_jpca               : full pipeline, returns result dict
jpca_to_json           : JSON-serializable summary
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Step 1 – PCA reduction
# ---------------------------------------------------------------------------

def pca_reduce(
    X: np.ndarray,
    n_components: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project activations to the top-n_components PCs.

    Parameters
    ----------
    X            : (n_cond, n_layers, d) array — per-condition, per-layer
                   mean activations (or any (n_cond, T, d) trajectory array).
    n_components : number of PC dimensions to retain.

    Returns
    -------
    X_pc  : (n_cond, T, p) — projected activations, p = n_components
    V_pc  : (d, p) — PC basis (columns are top eigenvectors)
    var_explained : (p,) — fraction of variance explained by each PC
    """
    n_cond, T, d = X.shape
    X_flat = X.reshape(-1, d)               # (n_cond * T, d)
    X_c    = X_flat - X_flat.mean(axis=0)   # center

    # PCA via SVD on centered data
    p = min(n_components, d, n_cond * T)
    _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    V_pc = Vt[:p].T                         # (d, p)

    var_total    = float(np.sum(s ** 2))
    var_explained = (s[:p] ** 2) / max(var_total, 1e-30)

    X_pc = (X_flat @ V_pc).reshape(n_cond, T, p)
    return X_pc, V_pc, var_explained


# ---------------------------------------------------------------------------
# Step 2 – Remove cross-condition mean
# ---------------------------------------------------------------------------

def remove_condition_mean(X_pc: np.ndarray) -> np.ndarray:
    """
    Subtract the cross-condition mean trajectory from each condition.

    Parameters
    ----------
    X_pc : (n_cond, T, p) — trajectories in PC space

    Returns
    -------
    X_demeaned : (n_cond, T, p) — mean-removed trajectories
    """
    mean_traj = X_pc.mean(axis=0, keepdims=True)   # (1, T, p)
    return X_pc - mean_traj


# ---------------------------------------------------------------------------
# Step 3 – Build regression matrices
# ---------------------------------------------------------------------------

def build_regression_mats(
    X_pc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten demeaned trajectories into regression matrices.

    For each condition c and each layer transition t → t+1:
      state    = x̃_c^{(t)}
      velocity = x̃_c^{(t+1)} - x̃_c^{(t)}

    Parameters
    ----------
    X_pc : (n_cond, T, p) — mean-removed trajectories in PC space

    Returns
    -------
    X_reg  : (p, n_cond * (T-1)) — state matrix (column = one observation)
    dX_reg : (p, n_cond * (T-1)) — velocity matrix
    """
    n_cond, T, p = X_pc.shape
    states, vels = [], []
    for c in range(n_cond):
        for t in range(T - 1):
            states.append(X_pc[c, t])
            vels.append(X_pc[c, t + 1] - X_pc[c, t])

    X_reg  = np.column_stack(states)  # (p, N)
    dX_reg = np.column_stack(vels)    # (p, N)
    return X_reg, dX_reg


# ---------------------------------------------------------------------------
# Step 4 – Unconstrained fit
# ---------------------------------------------------------------------------

def fit_unconstrained(
    X_reg: np.ndarray,
    dX_reg: np.ndarray,
) -> np.ndarray:
    """
    Least-squares M_unc minimizing ||dX - M X||²_F.

    Closed form: M_unc = dX @ X^T @ (X @ X^T)^{-1}
    Using pseudoinverse for numerical stability.

    Parameters
    ----------
    X_reg  : (p, N) state matrix
    dX_reg : (p, N) velocity matrix

    Returns
    -------
    M_unc : (p, p) unconstrained best-fit matrix
    """
    # M_unc = dX @ pinv(X)
    M_unc = dX_reg @ np.linalg.pinv(X_reg)
    return M_unc


# ---------------------------------------------------------------------------
# Step 5 – Skew-symmetric projection
# ---------------------------------------------------------------------------

def fit_skew_symmetric(M_unc: np.ndarray) -> np.ndarray:
    """
    Project M_unc onto the space of skew-symmetric matrices.

    The orthogonal projection under Frobenius norm is:
        M_skew = (M_unc - M_unc^T) / 2

    This is the unique closest skew-symmetric matrix to M_unc.

    Parameters
    ----------
    M_unc : (p, p) unconstrained fit

    Returns
    -------
    M_skew : (p, p) skew-symmetric matrix
    """
    return 0.5 * (M_unc - M_unc.T)


# ---------------------------------------------------------------------------
# Step 6 – R² scoring
# ---------------------------------------------------------------------------

def r2_score(
    M: np.ndarray,
    X_reg: np.ndarray,
    dX_reg: np.ndarray,
) -> float:
    """
    R² of the prediction dX̂ = M X against actual dX.

    R² = 1 - ||dX - M X||²_F / ||dX||²_F

    A value of 1 is a perfect fit; 0 means no better than predicting zero.
    Negative values are possible (worse than zero prediction).

    Parameters
    ----------
    M      : (p, p) candidate matrix
    X_reg  : (p, N) state matrix
    dX_reg : (p, N) velocity matrix

    Returns
    -------
    r2 : float
    """
    residual = dX_reg - M @ X_reg
    ss_res  = float(np.sum(residual ** 2))
    ss_tot  = float(np.sum(dX_reg ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-30)


# ---------------------------------------------------------------------------
# Step 7 – Extract rotation planes
# ---------------------------------------------------------------------------

def extract_rotation_planes(
    M_skew: np.ndarray,
    V_pc: np.ndarray,
    top_k: int = 3,
) -> list[dict]:
    """
    Eigendecompose M_skew to extract rotation planes and frequencies.

    M_skew is real skew-symmetric → eigenvalues are pure imaginary: ±iω_j.
    Each conjugate pair defines a rotation plane spanned by Re(v_j), Im(v_j)
    in PC space; lift back to full d-space via V_pc.

    Parameters
    ----------
    M_skew : (p, p) skew-symmetric matrix in PC space
    V_pc   : (d, p) PC basis (columns are top eigenvectors)
    top_k  : number of rotation planes to return (sorted by |ω| descending)

    Returns
    -------
    planes : list of dicts, each with:
      omega       : float  — rotation frequency |ω| (rad per layer)
      plane_pc    : (p, 2) — plane basis in PC space (Re, Im of eigenvec)
      plane_full  : (d, 2) — plane basis lifted to full d-space
      projector_full : (d, d) — rank-2 orthogonal projector onto this plane
    """
    # Eigendecompose (use scipy for complex eigenvalues of real matrix)
    try:
        from scipy.linalg import eig
        eigvals, eigvecs = eig(M_skew)
    except ImportError:
        eigvals, eigvecs = np.linalg.eig(M_skew)

    # Pair up conjugate eigenvalues; take one from each pair (positive Im part)
    paired = []
    used = set()
    for i, lam in enumerate(eigvals):
        if i in used:
            continue
        omega = float(np.imag(lam))
        if omega > 1e-10:   # positive imaginary → take this one
            paired.append((abs(omega), i))
            # find conjugate
            for j in range(i + 1, len(eigvals)):
                if j not in used and abs(eigvals[j] - np.conj(lam)) < 1e-8:
                    used.add(j)
                    break
            used.add(i)

    # Sort by frequency descending
    paired.sort(key=lambda x: x[0], reverse=True)

    planes = []
    for omega, idx in paired[:top_k]:
        v = eigvecs[:, idx]             # (p,) complex eigenvector
        u1 = np.real(v)
        u2 = np.imag(v)

        # Orthonormalize within the plane
        u1 = u1 / max(np.linalg.norm(u1), 1e-12)
        u2 = u2 - np.dot(u2, u1) * u1
        u2 = u2 / max(np.linalg.norm(u2), 1e-12)

        plane_pc   = np.column_stack([u1, u2])               # (p, 2)
        plane_full = V_pc @ plane_pc                          # (d, 2)

        # Re-orthonormalize full-space plane
        q1 = plane_full[:, 0] / max(np.linalg.norm(plane_full[:, 0]), 1e-12)
        q2 = plane_full[:, 1] - np.dot(plane_full[:, 1], q1) * q1
        q2 /= max(np.linalg.norm(q2), 1e-12)
        plane_full = np.column_stack([q1, q2])

        proj = plane_full @ plane_full.T                      # (d, d) rank-2

        planes.append({
            "omega":           omega,
            "plane_pc":        plane_pc,
            "plane_full":      plane_full,
            "projector_full":  proj,
        })

    return planes


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def fit_jpca(
    activations: np.ndarray,
    n_pc: int = 6,
    top_k_planes: int = 3,
    remove_mean: bool = True,
) -> dict:
    """
    Full jPCA pipeline on transformer layer trajectories.

    Parameters
    ----------
    activations   : (n_cond, n_layers, d) — per-condition, per-layer
                    mean activations. "Conditions" are prompts.
                    For per-token analysis, stack tokens along the condition axis.
    n_pc          : number of PCs to reduce to (default 6, as in Churchland 2012)
    top_k_planes  : number of rotation planes to extract
    remove_mean   : if True, subtract cross-condition mean (standard jPCA).
                    Set False only for debugging; use hdr_fit.py instead for
                    mean-free analysis.

    Returns
    -------
    dict with:
      M_skew         : (p, p) skew-symmetric fit in PC space
      M_unc          : (p, p) unconstrained fit in PC space
      r2_skew        : float — R² of skew-symmetric fit
      r2_unc         : float — R² of unconstrained fit
      r2_ratio       : float — r2_skew / r2_unc  (P2c-J1: should exceed 0.5)
      planes         : list of plane dicts (from extract_rotation_planes)
      V_pc           : (d, p) PC basis
      var_explained  : (p,) variance fractions
      n_pc           : int
      n_cond         : int
      n_layers       : int
      p2cj1_holds    : bool — r2_ratio > 0.5
      p2cj1_marginal : bool — 0.3 < r2_ratio <= 0.5 (borderline; run HDR)
    """
    n_cond, n_layers, d = activations.shape
    assert n_layers >= 2, "Need at least 2 layers for Δx."

    # PCA
    X_pc, V_pc, var_exp = pca_reduce(activations, n_components=n_pc)

    # Remove cross-condition mean
    if remove_mean:
        X_pc = remove_condition_mean(X_pc)

    # Build regression matrices
    X_reg, dX_reg = build_regression_mats(X_pc)

    # Fit unconstrained + skew-symmetric
    M_unc  = fit_unconstrained(X_reg, dX_reg)
    M_skew = fit_skew_symmetric(M_unc)

    # R² scores
    r2_unc  = r2_score(M_unc,  X_reg, dX_reg)
    r2_skew = r2_score(M_skew, X_reg, dX_reg)
    r2_ratio = r2_skew / max(r2_unc, 1e-30)

    # Rotation planes
    planes = extract_rotation_planes(M_skew, V_pc, top_k=top_k_planes)

    return {
        "M_skew":        M_skew,
        "M_unc":         M_unc,
        "r2_skew":       r2_skew,
        "r2_unc":        r2_unc,
        "r2_ratio":      r2_ratio,
        "planes":        planes,
        "V_pc":          V_pc,
        "var_explained": var_exp,
        "n_pc":          n_pc,
        "n_cond":        n_cond,
        "n_layers":      n_layers,
        "p2cj1_holds":    r2_ratio > 0.5,
        "p2cj1_marginal": 0.3 < r2_ratio <= 0.5,
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def jpca_to_json(result: dict) -> dict:
    """
    JSON-serializable summary. Drops large arrays (M matrices, projectors,
    full-space plane bases). Keeps scalar stats and PC-space plane info.
    """
    planes_summary = [
        {
            "omega":        float(pl["omega"]),
            "plane_pc":     pl["plane_pc"].tolist(),
        }
        for pl in result["planes"]
    ]

    return {
        "r2_skew":       float(result["r2_skew"]),
        "r2_unc":        float(result["r2_unc"]),
        "r2_ratio":      float(result["r2_ratio"]),
        "n_pc":          int(result["n_pc"]),
        "n_cond":        int(result["n_cond"]),
        "n_layers":      int(result["n_layers"]),
        "var_explained": [float(v) for v in result["var_explained"]],
        "planes":        planes_summary,
        "p2cj1_holds":    bool(result["p2cj1_holds"]),
        "p2cj1_marginal": bool(result["p2cj1_marginal"]),
    }
