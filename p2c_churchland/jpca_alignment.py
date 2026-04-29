"""
jpca_alignment.py — C1: principal angles between jPCA planes and U_A.

Takes jPCA rotation planes (from jpca_fit.py) and the operator-side U_A
planes (from p2b_imaginary/rotational_schur.py) and computes the principal
angles between the two sets of subspaces.

Three diagnostic outcomes (from the README):
  Coincide (small angles) → V's rotational structure is exercised by data.
  Orthogonal             → Rotation is real but emerges from softmax/FFN, not V.
  Layer-localized        → Run per-window jPCA to find where.

Prediction tested:
  P2c-J2 : mean principal angle < 30° between top jPCA planes and U_A planes.
            Failure: angles uniform on [0°, 90°].

Functions
---------
principal_angles           : canonical angles between two subspaces
mean_principal_angle       : scalar summary across all jPCA-vs-UA plane pairs
jpca_ua_alignment          : full comparison, all planes × all UA planes
alignment_verdict          : P2c-J2 verdict + interpretation string
print_alignment            : terminal report
alignment_to_json          : JSON-serializable summary
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Principal angles between two subspaces
# ---------------------------------------------------------------------------

def principal_angles(
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """
    Compute the principal (canonical) angles between two subspaces.

    Parameters
    ----------
    A : (d, r_A) — orthonormal basis for subspace A
    B : (d, r_B) — orthonormal basis for subspace B

    Returns
    -------
    angles : (min(r_A, r_B),) array of angles in degrees, sorted ascending.

    Notes
    -----
    Principal angles θ_i are defined via the SVD of A^T B:
        A^T B = U Σ V^T,  Σ_{ii} = cos(θ_i)
    Values of Σ are clamped to [-1, 1] before arccos to handle
    numerical noise from near-parallel subspaces.
    """
    # Ensure orthonormality
    A = _orth(A)
    B = _orth(B)

    C = A.T @ B                                        # (r_A, r_B)
    sv = np.linalg.svd(C, compute_uv=False)            # singular values = cos(θ)
    sv = np.clip(sv, -1.0, 1.0)
    angles_rad = np.arccos(sv)
    return np.degrees(angles_rad)                       # ascending order from SVD


def _orth(M: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Return an orthonormal basis for the column space of M."""
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(s > tol * max(s[0], 1e-30)))
    return U[:, :r]


# ---------------------------------------------------------------------------
# Aggregate across plane pairs
# ---------------------------------------------------------------------------

def mean_principal_angle(
    jpca_planes: list[dict],
    ua_planes: list[np.ndarray],
) -> float:
    """
    Mean minimum principal angle across all jPCA planes vs all U_A planes.

    For each jPCA plane, find the minimum principal angle over all U_A planes
    (i.e., best-matching U_A plane). Average those minima.

    Parameters
    ----------
    jpca_planes : list of dicts with key "plane_full" — (d, 2) basis
    ua_planes   : list of (d, 2) U_A plane bases from p2b Schur decomposition

    Returns
    -------
    mean_min_angle : float — degrees; < 30° → P2c-J2 holds
    """
    if not jpca_planes or not ua_planes:
        return float("nan")

    min_angles = []
    for jp in jpca_planes:
        A = jp["plane_full"]
        mins = [principal_angles(A, ua)[0] for ua in ua_planes]
        min_angles.append(float(np.min(mins)))

    return float(np.mean(min_angles))


# ---------------------------------------------------------------------------
# Full alignment analysis
# ---------------------------------------------------------------------------

def jpca_ua_alignment(
    jpca_result: dict,
    ua_planes: list[np.ndarray],
    angle_threshold_deg: float = 30.0,
) -> dict:
    """
    Full principal-angle comparison between jPCA planes and U_A planes.

    Parameters
    ----------
    jpca_result         : output of jpca_fit.fit_jpca
    ua_planes           : list of (d, 2) U_A plane bases from p2b.
                          Obtain via:
                            planes_dict = build_rotation_plane_projectors(blocks, top_k)
                            ua_planes = planes_dict["top_k_planes"]
    angle_threshold_deg : threshold for P2c-J2 (default 30°)

    Returns
    -------
    dict with:
      pairwise_angles    : (n_jpca, n_ua) array of minimum principal angle
                           per pair [degrees]
      per_jpca_min_angle : (n_jpca,) best-matching U_A angle per jPCA plane
      mean_min_angle     : float — scalar summary
      p2cj2_holds        : bool — mean_min_angle < angle_threshold_deg
      jpca_omegas        : (n_jpca,) rotation frequencies
      angle_distribution : "aligned" | "orthogonal" | "mixed"
      interpretation     : str — one-line diagnostic
    """
    jp = jpca_result["planes"]
    n_jp = len(jp)
    n_ua = len(ua_planes)

    # Pairwise: for each (jPCA plane i, UA plane j), compute first principal angle
    pairwise = np.full((n_jp, n_ua), fill_value=float("nan"))
    for i, jplane in enumerate(jp):
        A = jplane["plane_full"]
        for j, ua in enumerate(ua_planes):
            B = ua if ua.ndim == 2 else ua.reshape(-1, 2)
            angles = principal_angles(A, B)
            pairwise[i, j] = float(angles[0]) if len(angles) > 0 else 90.0

    per_jpca_min = (
        pairwise.min(axis=1) if n_ua > 0 and n_jp > 0
        else np.full(n_jp, fill_value=float("nan"))
    )
    mean_min = float(np.nanmean(per_jpca_min)) if n_jp > 0 else float("nan")

    # Classify angle distribution
    if np.isnan(mean_min):
        dist = "unknown"
    elif mean_min < angle_threshold_deg:
        dist = "aligned"
    elif mean_min > 60.0:
        dist = "orthogonal"
    else:
        dist = "mixed"

    interp_map = {
        "aligned":    "jPCA planes coincide with U_A: V's rotational capacity is exercised.",
        "orthogonal": "jPCA planes orthogonal to U_A: rotation is real but driven by softmax/FFN, not V.",
        "mixed":      "Partial overlap: localized rotational computation; run per-window jPCA.",
        "unknown":    "Insufficient data for interpretation.",
    }

    omegas = np.array([float(p["omega"]) for p in jp])

    return {
        "pairwise_angles":    pairwise,
        "per_jpca_min_angle": per_jpca_min,
        "mean_min_angle":     mean_min,
        "p2cj2_holds":        mean_min < angle_threshold_deg,
        "jpca_omegas":        omegas,
        "angle_distribution": dist,
        "interpretation":     interp_map[dist],
        "angle_threshold_deg": angle_threshold_deg,
    }


# ---------------------------------------------------------------------------
# Verdict + terminal output
# ---------------------------------------------------------------------------

def alignment_verdict(result: dict) -> str:
    """One-line verdict string for P2c-J2."""
    v = "HOLDS" if result["p2cj2_holds"] else "FAILS"
    ang = result["mean_min_angle"]
    thr = result["angle_threshold_deg"]
    return (
        f"P2c-J2 {v}: mean min principal angle = {ang:.1f}° "
        f"(threshold {thr:.0f}°) — {result['interpretation']}"
    )


def print_alignment(result: dict) -> None:
    sep = "-" * 60
    print(sep)
    print("C1 — jPCA / U_A Alignment")
    print(sep)
    print(f"  Mean min principal angle: {result['mean_min_angle']:.2f}°")
    print(f"  Distribution type: {result['angle_distribution']}")
    print()
    print(f"  Per-jPCA-plane min angles (degrees):")
    for i, (ang, om) in enumerate(
        zip(result["per_jpca_min_angle"], result["jpca_omegas"])
    ):
        print(f"    Plane {i+1}  ω={om:.4f} rad/layer   min_angle={ang:.2f}°")
    print()
    print(f"  {alignment_verdict(result)}")
    print(sep)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def alignment_to_json(result: dict) -> dict:
    return {
        "mean_min_angle":     float(result["mean_min_angle"]),
        "per_jpca_min_angle": [float(x) for x in result["per_jpca_min_angle"]],
        "pairwise_angles":    result["pairwise_angles"].tolist(),
        "jpca_omegas":        [float(x) for x in result["jpca_omegas"]],
        "p2cj2_holds":        bool(result["p2cj2_holds"]),
        "angle_distribution": result["angle_distribution"],
        "interpretation":     result["interpretation"],
        "angle_threshold_deg": float(result["angle_threshold_deg"]),
    }
