"""
write_subspace.py — Track C: Write subspace alignment and channel orthogonality.

Each attention head writes into the residual stream via W_O.  The dominant
write directions are the left singular vectors of W_O.

Two tests:

C.1 — Write direction alignment
  For each head, compute what fraction of its top write directions land in
  the imaginary (A) vs real (S) subspace.  Cross-correlate with CC score
  from head_classify (prediction P6-C1).

C.2 — Write subspace orthogonality
  Partition heads into real-channel (align_rot < 0.4) and imaginary-channel
  (align_rot > 0.6).  Compute principal angles between the union of write
  subspaces for each partition.  Small principal angles = channels overlap
  (weak hypothesis).  Large principal angles = orthogonal (strong hypothesis).

Falsifiable prediction tested
------------------------------
P6-C1 : align_rot(h) and CC(h) are negatively correlated across heads.
         Spearman ρ(align_rot, -CC) > 0 (directional); strong result > 0.4.

Functions
---------
head_write_alignment    : per-head align_rot and align_real from W_O
principal_angles        : angles between two subspaces (via SVD)
channel_orthogonality   : C.2 — partition heads and compute principal angles
run_write_subspace      : full pipeline → SubResult
"""

import numpy as np
from scipy.stats import spearmanr

from p6_subspace.p6_io import SubResult, _fmt, _bullet, _verdict_line, SEP_THICK, SEP_THIN


# ---------------------------------------------------------------------------
# Per-head write alignment
# ---------------------------------------------------------------------------

def head_write_alignment(
    WO:    np.ndarray,
    P_A:   np.ndarray,
    P_S:   np.ndarray,
    top_r: int = 16,
) -> dict:
    """
    Fraction of top write directions in each channel.

    Parameters
    ----------
    WO    : (d_model, d_head) — output projection
    P_A   : (d_model, d_model)
    P_S   : (d_model, d_model)
    top_r : number of dominant write directions

    Returns
    -------
    dict with align_rot, align_real, sing_vals
    """
    U, s, _ = np.linalg.svd(WO, full_matrices=False)
    r       = min(top_r, U.shape[1])
    U_top   = U[:, :r]   # (d_model, r)

    rot_scores  = np.array([float(U_top[:, k] @ P_A @ U_top[:, k]) for k in range(r)])
    real_scores = np.array([float(U_top[:, k] @ P_S @ U_top[:, k]) for k in range(r)])

    return {
        "align_rot":  float(rot_scores.mean()),
        "align_real": float(real_scores.mean()),
        "sing_vals":  s[:r].tolist(),
    }


# ---------------------------------------------------------------------------
# Principal angles
# ---------------------------------------------------------------------------

def principal_angles(
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """
    Compute principal angles between subspaces spanned by columns of A and B.

    Uses the SVD-based formula: cos(θ_k) = σ_k(A^T B)

    Parameters
    ----------
    A : (d, p) — orthonormal basis for subspace 1
    B : (d, q) — orthonormal basis for subspace 2

    Returns
    -------
    angles : (min(p,q),) in radians, sorted ascending
    """
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.array([np.pi / 2])   # orthogonal by default

    # Orthonormalise just in case
    A_orth, _ = np.linalg.qr(A)
    B_orth, _ = np.linalg.qr(B)

    M = A_orth.T @ B_orth
    sigma = np.linalg.svd(M, compute_uv=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    return np.sort(np.arccos(sigma))


# ---------------------------------------------------------------------------
# Channel orthogonality (C.2)
# ---------------------------------------------------------------------------

def channel_orthogonality(
    head_records:  list[dict],
    wo_matrices:   list[np.ndarray],
    align_rot_lo:  float = 0.4,
    align_rot_hi:  float = 0.6,
) -> dict:
    """
    Partition heads and compute principal angles between channel write subspaces.

    Heads with align_rot < align_rot_lo → real-channel
    Heads with align_rot > align_rot_hi → imaginary-channel
    Heads in between are excluded from this test.

    Returns
    -------
    dict with:
      n_real_heads     : int
      n_imag_heads     : int
      principal_angles : (k,) in degrees, sorted ascending
      min_angle_deg    : float — smallest principal angle (0 = full overlap)
      mean_angle_deg   : float — mean principal angle
      max_angle_deg    : float — largest (90 = orthogonal)
      is_orthogonal    : bool  — mean angle > 60 degrees
    """
    real_heads = [
        r["head_idx"] for r in head_records if r["align_rot"] < align_rot_lo
    ]
    imag_heads = [
        r["head_idx"] for r in head_records if r["align_rot"] > align_rot_hi
    ]

    if not real_heads or not imag_heads:
        return {
            "n_real_heads": len(real_heads),
            "n_imag_heads": len(imag_heads),
            "principal_angles": None,
            "min_angle_deg":    None,
            "mean_angle_deg":   None,
            "max_angle_deg":    None,
            "is_orthogonal":    None,
        }

    # Collect and orthonormalise write directions per partition
    def _collect_write_vecs(head_indices):
        vecs = []
        for h in head_indices:
            WO = wo_matrices[h]
            U, _, _ = np.linalg.svd(WO, full_matrices=False)
            vecs.append(U)   # (d, d_head) or similar
        return np.column_stack(vecs) if vecs else np.zeros((wo_matrices[0].shape[0], 0))

    V_real = _collect_write_vecs(real_heads)
    V_imag = _collect_write_vecs(imag_heads)

    # Orthonormalise each collected basis
    Q_real, _ = np.linalg.qr(V_real)
    Q_imag, _ = np.linalg.qr(V_imag)

    angles_rad = principal_angles(Q_real, Q_imag)
    angles_deg = np.degrees(angles_rad)

    return {
        "n_real_heads":    len(real_heads),
        "n_imag_heads":    len(imag_heads),
        "principal_angles": angles_deg.tolist(),
        "min_angle_deg":    float(angles_deg.min()),
        "mean_angle_deg":   float(angles_deg.mean()),
        "max_angle_deg":    float(angles_deg.max()),
        "is_orthogonal":    bool(float(angles_deg.mean()) > 60.0),
    }


# ---------------------------------------------------------------------------
# Full pipeline → SubResult
# ---------------------------------------------------------------------------

def run_write_subspace(ctx: dict) -> SubResult:
    """
    Track C sub-experiment: write direction alignment and channel orthogonality.

    Required ctx keys
    -----------------
    wo_matrices         : list of (d_model, d_head) W_O per head
    projectors          : output of subspace_build
    layer_idx           : int (default 0)

    Optional ctx keys
    -----------------
    head_classify_result: for P6-C1 (CC scores); if absent, skips correlation
    top_r               : int (default 16) — singular vectors to use
    layer_name          : str
    """
    wo_matrices  = ctx["wo_matrices"]
    projectors   = ctx["projectors"]
    layer_idx    = ctx.get("layer_idx", 0)
    layer_name   = ctx.get("layer_name", "shared")
    top_r        = ctx.get("top_r", 16)

    pe   = projectors["per_layer"][layer_idx]
    P_A  = pe["P_A"]
    P_S  = pe["P_S"]

    n_heads = len(wo_matrices)

    # C.1 — Per-head write alignment
    alignments = [
        head_write_alignment(wo_matrices[h], P_A, P_S, top_r)
        for h in range(n_heads)
    ]
    head_records = [
        {"head_idx": h, **alignments[h]}
        for h in range(n_heads)
    ]

    # P6-C1 — Cross-head correlation with CC
    hc_result = ctx.get("head_classify_result")
    p6_c1_rho = None
    p6_c1_satisfied = None

    if hc_result and "head_records" in hc_result:
        cc_map = {
            r["head_idx"]: r["cc"]
            for r in hc_result["head_records"]
        }
        shared_heads = [h for h in range(n_heads) if h in cc_map]
        if len(shared_heads) >= 4:
            f_rot_vals = [alignments[h]["align_rot"] for h in shared_heads]
            cc_vals    = [cc_map[h]                   for h in shared_heads]
            rho, _     = spearmanr(f_rot_vals, [-c for c in cc_vals])
            p6_c1_rho  = float(rho) if np.isfinite(rho) else None
            p6_c1_satisfied = (p6_c1_rho is not None and p6_c1_rho > 0.4)

    # C.2 — Channel orthogonality
    ortho = channel_orthogonality(head_records, wo_matrices)

    mean_align_rot  = float(np.mean([a["align_rot"]  for a in alignments]))
    mean_align_real = float(np.mean([a["align_real"] for a in alignments]))

    payload = {
        "layer_name":       layer_name,
        "n_heads":          n_heads,
        "mean_align_rot":   mean_align_rot,
        "mean_align_real":  mean_align_real,
        "p6_c1_rho":        p6_c1_rho,
        "p6_c1_satisfied":  p6_c1_satisfied,
        "channel_ortho":    ortho,
        "head_records":     head_records,
    }

    # --- Summary lines ---
    lines = [
        SEP_THICK,
        "WRITE SUBSPACE ALIGNMENT  [Track C]",
        SEP_THICK,
        f"Layer:         {layer_name}",
        f"Heads:         {n_heads}",
        f"Singular vecs: top {top_r}",
        "",
        "C.1 — Per-head write direction alignment with S and A channels:",
        _bullet("mean align_rot  across all heads", mean_align_rot),
        _bullet("mean align_real across all heads", mean_align_real),
        "",
    ]

    if p6_c1_rho is not None:
        lines += [
            "P6-C1: Spearman ρ(align_rot, -CC) > 0.4?",
            _bullet("Spearman ρ(align_rot, -CC)", p6_c1_rho),
            _verdict_line(
                "P6-C1",
                p6_c1_satisfied,
                f"ρ={_fmt(p6_c1_rho)} (threshold 0.4)",
            ),
            "",
        ]
    else:
        lines += ["P6-C1: skipped — head_classify_result not available", ""]

    lines += [
        "C.2 — Channel write subspace orthogonality:",
        _bullet("real-channel heads (align_rot < 0.4)", ortho["n_real_heads"]),
        _bullet("imag-channel heads (align_rot > 0.6)", ortho["n_imag_heads"]),
    ]

    if ortho["mean_angle_deg"] is not None:
        lines += [
            _bullet("min principal angle (deg)", ortho["min_angle_deg"]),
            _bullet("mean principal angle (deg)", ortho["mean_angle_deg"]),
            _bullet("max principal angle (deg)", ortho["max_angle_deg"]),
            f"  Interpretation: mean angle > 60° → channels occupy orthogonal subspaces.",
            _verdict_line(
                "C.2 channel separation",
                ortho["is_orthogonal"],
                f"mean angle={_fmt(ortho['mean_angle_deg'])}° (threshold 60°)",
            ),
        ]
    else:
        lines.append("  C.2: insufficient heads in one or both partitions to compute angles.")

    lines += [
        "",
        "Per-head write alignment:",
        f"  {'head':>4}  {'align_rot':>10}  {'align_real':>10}  {'channel':>12}",
    ]
    for r in head_records:
        channel = (
            "imaginary" if r["align_rot"] > 0.6
            else "real" if r["align_rot"] < 0.4
            else "mixed"
        )
        lines.append(
            f"  {r['head_idx']:>4d}  "
            f"{_fmt(r['align_rot']):>10}  "
            f"{_fmt(r['align_real']):>10}  "
            f"{channel:>12}"
        )

    vc = {
        "ws_mean_align_rot":   mean_align_rot,
        "ws_mean_align_real":  mean_align_real,
        "ws_p6_c1_rho":        p6_c1_rho,
        "ws_p6_c1_satisfied":  p6_c1_satisfied,
        "ws_mean_principal_angle": ortho.get("mean_angle_deg"),
        "ws_channel_orthogonal":   ortho.get("is_orthogonal"),
    }

    return SubResult(
        name="write_subspace",
        applicable=True,
        payload=payload,
        summary_lines=lines,
        verdict_contribution=vc,
    )
