"""
p2d_operator_activation/table1_predictions.py — sub-experiment D3: the
paper's Table 1 as a falsifiable claim about our activations.

WHERE THIS COMES FROM

Section 9.2 drops the tangent projector, giving PURE self-attention (9.1),
analyzed in [GLPR24] under the rescaling z_i = e^{-tV} x_i. Table 1:

    V = I_d                    Q^T K > 0                vertices of a convex polytope
    lam_1(V) > 0 simple        <Q phi_1, K phi_1> > 0   UNION OF 3 PARALLEL HYPERPLANES
    V paranormal               Q^T K > 0                polytope x subspaces
    V = -I_d                   Q^T K = I_d              single cluster at the origin

Phase 2 already computes the LEFT column — `V_repulsive_local`,
`mixed_or_unattributed` and the rest are classifications of V's spectrum.
Table 1 turns each classification into a GEOMETRIC PREDICTION ABOUT THE
ACTIVATIONS, and none of them has been tested.

Row 2 is the sharpest available prediction anywhere in the paper: a real,
simple, positive top eigenvalue of the OV matrix predicts activations
concentrating on three parallel hyperplanes normal to phi_1 — i.e. the
scalar projection <phi_1, x_i> should be TRIMODAL. It costs a projection
and a histogram.

TWO HYPOTHESES THAT MUST BE CHECKED, NOT ONE

P-T1 as registered says "heads classified lam_1(V) > 0 simple show trimodal
<phi_1, x_i>". Table 1's row 2 has a SECOND condition that the registered
wording omits: <Q phi_1, K phi_1> > 0, i.e. phi_1^T M phi_1 > 0. A head
with a positive simple top eigenvalue but a negative QK form is not in
row 2 at all, and testing it would falsify a prediction the paper never
made — the same error the "Thm 6.1 unsupported" verdict row made. Both
conditions are checked here and heads are only counted as row-2 candidates
when both hold.

THE RESCALING CAVEAT, WHICH IS ALSO THE FALSIFIER

Table 1 describes the limit geometry of z_i = e^{-tV} x_i, NOT of x_i. The
rescaling is time-dependent and t is not observable from a fixed-depth
network, so a direct test on raw activations is testing a related but
weaker claim. That is exactly what P-T1's falsifier anticipates ("unimodal
=> Table 1 does not transfer past the rescaling"), so the honest procedure
is to test the raw projection, report unimodality as evidence about the
TRANSFER rather than about the theorem, and separately report the
rescaled projection at a few candidate t so the reader can see whether the
structure appears at any of them.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------

def classify_ov_row(W_OV: np.ndarray, M: np.ndarray,
                    simple_tol: float = 0.05,
                    real_tol: float = 1e-6) -> dict:
    """
    Which row of Table 1 does this head fall under?

    simple_tol : the top eigenvalue counts as SIMPLE when the next one is
                 at least this far away in relative terms. Numerical
                 simplicity is not a yes/no property of a non-normal
                 matrix — eigenvalues of a d = 1024 OV circuit come in
                 near-degenerate clusters — so a tolerance is unavoidable
                 and is reported alongside the verdict rather than buried.
    real_tol   : |Im(lam_1)| / |lam_1| below this counts as real. Row 2
                 requires a real positive top eigenvalue; a complex pair
                 is a rotation and belongs to the paranormal row at best.

    Returns the row label plus every quantity the decision used, so a
    reclassification under different tolerances needs no recomputation.
    """
    V = np.asarray(W_OV, dtype=np.float64)
    ev, evec = np.linalg.eig(V)
    order = np.argsort(-ev.real)
    ev, evec = ev[order], evec[:, order]

    lam1 = ev[0]
    mag1 = abs(lam1)
    is_real = (abs(lam1.imag) / mag1 < real_tol) if mag1 > 1e-15 else False
    gap = (abs(lam1 - ev[1]) / mag1) if len(ev) > 1 and mag1 > 1e-15 else np.inf
    is_simple = gap > simple_tol
    is_positive = lam1.real > 0

    phi1 = np.real(evec[:, 0]) if is_real else evec[:, 0].real
    nrm = np.linalg.norm(phi1)
    phi1 = phi1 / nrm if nrm > 1e-15 else phi1

    M = np.asarray(M, dtype=np.float64)
    qk_form = float(phi1 @ M @ phi1)      # <Q phi_1, K phi_1>

    # Row 1 / row 4 need V ~ +-I, which we test by how close V is to a
    # scalar multiple of the identity in the Frobenius sense.
    d = V.shape[0]
    scale = float(np.trace(V) / d)
    iso_resid = float(np.linalg.norm(V - scale * np.eye(d), "fro")
                      / max(np.linalg.norm(V, "fro"), 1e-15))

    if iso_resid < 0.10 and scale > 0:
        row = "row1_V_identity"
    elif iso_resid < 0.10 and scale < 0:
        row = "row4_V_negative_identity"
    elif is_real and is_simple and is_positive and qk_form > 0:
        row = "row2_three_hyperplanes"
    elif is_real and is_simple and is_positive:
        row = "row2_eigen_only_qk_fails"
    else:
        row = "unclassified"

    return {
        "row": row,
        "lam1_real": float(lam1.real), "lam1_imag": float(lam1.imag),
        "lam1_is_real": bool(is_real), "lam1_is_simple": bool(is_simple),
        "lam1_is_positive": bool(is_positive),
        "eigen_gap_rel": float(gap) if np.isfinite(gap) else float("inf"),
        "qk_form_phi1": qk_form,
        "qk_condition_met": bool(qk_form > 0),
        "phi1": phi1,
        "isotropy_residual": iso_resid,
        "V_scale": scale,
        "simple_tol": simple_tol, "real_tol": real_tol,
        # Row 2 needs BOTH conditions. Recorded explicitly because the
        # registered wording of P-T1 omits the second one.
        "row2_candidate": bool(row == "row2_three_hyperplanes"),
    }


# ---------------------------------------------------------------------------
# The trimodality test
# ---------------------------------------------------------------------------

def projection_modality(X: np.ndarray, phi: np.ndarray, grid: int = 512,
                        prominence_frac: float = 0.10,
                        bw: float = 1.0, center: bool = True) -> dict:
    """
    Modality of the scalar projection <phi, x_i>, by kernel density.

    center : subtract the mean projection first. The prediction is THREE
             PARALLEL hyperplanes, which is a statement about spacing, not
             about position — an uncentred projection of a strongly
             anisotropic cloud is dominated by the common mode and reads
             unimodal regardless of any structure on top of it.

    WHY A KDE AND NOT A HISTOGRAM. Counting local maxima of a histogram
    counts bin noise as structure, and the error is large enough to invert
    the answer: at 60 bins on 500 points, a plain Gaussian cloud scored
    NINE modes and a genuinely trimodal one scored four. Peak counting only
    means something on a smooth density. Silverman's rule sets the
    bandwidth from the sample, and `bw` scales it.

    THE BANDWIDTH IS THE WHOLE TEST, WHICH IS WHY IT IS REPORTED AND
    SCANNABLE. Any distribution can be made unimodal by over-smoothing and
    multimodal by under-smoothing; a modality claim at a single unstated
    bandwidth is not a measurement. `modality_stability` scans it. Treat a
    mode count that survives a factor-of-two bandwidth change as real and
    one that does not as an artifact.

    Returns the mode count, locations, and — the actual test — the SPACING
    REGULARITY. Three parallel hyperplanes are equally spaced, so three
    modes at arbitrary positions are much weaker evidence than three at
    positions x-s, x, x+s. `spacing_ratio` is the ratio of the two gaps for
    exactly three modes; near 1 is the prediction.
    """
    X = np.asarray(X, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64).ravel()
    p = X @ phi
    if center:
        p = p - p.mean()
    sd = float(p.std())
    n = p.size
    if sd < 1e-12 or n < 8:
        return {"n_modes": 0, "degenerate": True}

    # Silverman, scaled by bw.
    iqr = float(np.subtract(*np.percentile(p, [75, 25])))
    sigma = min(sd, iqr / 1.349) if iqr > 0 else sd
    h = bw * 0.9 * sigma * n ** (-0.2)
    if h <= 0:
        return {"n_modes": 0, "degenerate": True}

    lo, hi = p.min() - 3 * h, p.max() + 3 * h
    centres = np.linspace(lo, hi, grid)
    # Direct evaluation; n <= a few thousand tokens, grid 512.
    z = (centres[:, None] - p[None, :]) / h
    dens = np.exp(-0.5 * z ** 2).sum(axis=1) / (n * h * np.sqrt(2 * np.pi))

    thresh = prominence_frac * dens.max()
    peaks = [i for i in range(1, grid - 1)
             if dens[i] > dens[i - 1] and dens[i] >= dens[i + 1] and dens[i] > thresh]
    # Endpoints, for mass piled against the edge of the support.
    if dens[0] > dens[1] and dens[0] > thresh:
        peaks.insert(0, 0)
    if dens[-1] > dens[-2] and dens[-1] > thresh:
        peaks.append(grid - 1)

    locs = [float(centres[i]) for i in peaks]
    out = {
        "n_modes": len(peaks),
        "mode_locations": locs,
        "trimodal": len(peaks) == 3,
        "unimodal": len(peaks) <= 1,
        "projection_std": sd,
        "projection_mean": float((X @ phi).mean()),
        "bandwidth": float(h),
        "bw_scale": float(bw),
        "degenerate": False,
    }
    if len(locs) == 3:
        g1, g2 = locs[1] - locs[0], locs[2] - locs[1]
        out["spacing_ratio"] = float(min(g1, g2) / max(g1, g2)) if max(g1, g2) > 1e-12 else float("nan")
        out["equally_spaced"] = bool(out["spacing_ratio"] > 0.7)
    return out


def rescaled_modality(X: np.ndarray, W_OV: np.ndarray, phi: np.ndarray,
                      t_values=(0.0, 0.25, 0.5, 1.0, 2.0), **kw) -> list:
    """
    The same test on z_i = exp(-t V) x_i, for a few candidate t.

    Table 1's geometry is the limit geometry of the RESCALED variable, and
    t is not observable from a fixed-depth network. Rather than pick one,
    scan and report: if trimodality appears at some t and not at t = 0,
    that is evidence the structure is real and the rescaling is what hides
    it — a meaningfully different conclusion from "Table 1 does not
    transfer", and one the raw test alone cannot reach.

    exp(-tV) is computed by eigendecomposition of V rather than by a series,
    since V is non-normal and a truncated series diverges for the t and
    spectral radii involved here.
    """
    V = np.asarray(W_OV, dtype=np.float64)
    ev, evec = np.linalg.eig(V)
    try:
        evec_inv = np.linalg.inv(evec)
    except np.linalg.LinAlgError:
        return [{"t": None, "error": "eigenvector matrix singular; V is "
                                     "defective and exp(-tV) needs a Schur "
                                     "form. Use the raw test only."}]
    out = []
    for t in t_values:
        E = evec @ np.diag(np.exp(-t * ev)) @ evec_inv
        Z = np.real(np.asarray(X, dtype=np.float64) @ E.T)
        r = projection_modality(Z, phi, **kw)
        cond = float(np.linalg.cond(evec))
        out.append({"t": float(t), "cond_evec": cond,
                    # A large eigenvector condition number means the
                    # rescaling is numerically meaningless, and the
                    # modality below is noise. Reported per t rather than
                    # once, since the amplification grows with t.
                    "reliable": bool(cond < 1e8), **r})
    return out


# ---------------------------------------------------------------------------
# P-T1
# ---------------------------------------------------------------------------

def adjudicate_p_t1(results: list) -> dict:
    """
    P-T1: heads classified lam_1(V) > 0 simple show trimodal <phi_1, x_i>.
    Falsifier: unimodal => Table 1 does not transfer past the rescaling.

    results : per-head dicts each carrying `row2_candidate` and a
              `modality` sub-dict from projection_modality.

    Reports the trimodal rate among row-2 candidates AND among
    non-candidates. The second is the control and it is not optional: if
    non-candidates are trimodal at the same rate, trimodality is a property
    of the activations rather than of the classification, and a
    candidates-only number would read as confirmation.
    """
    cand = [r for r in results if r.get("row2_candidate")]
    other = [r for r in results if not r.get("row2_candidate")]

    def _rate(rs, key):
        vals = [bool(r.get("modality", {}).get(key)) for r in rs
                if r.get("modality") and not r["modality"].get("degenerate")]
        return (float(np.mean(vals)) if vals else float("nan")), len(vals)

    tri_c, n_c = _rate(cand, "trimodal")
    tri_o, n_o = _rate(other, "trimodal")
    uni_c, _ = _rate(cand, "unimodal")

    spaced = [r["modality"].get("equally_spaced") for r in cand
              if r.get("modality", {}).get("trimodal")]
    spaced_rate = float(np.mean([bool(s) for s in spaced])) if spaced else float("nan")

    if n_c == 0:
        verdict = ("NO CANDIDATES — no head satisfies BOTH row-2 conditions "
                   "(lam_1 real, simple, positive AND phi_1^T M phi_1 > 0). "
                   "P-T1 is untestable on this model, which is itself a "
                   "result about how far Table 1's regimes are from a "
                   "trained transformer.")
    elif np.isfinite(tri_c) and tri_c > 0.5 and (not np.isfinite(tri_o) or tri_c > 2 * tri_o):
        verdict = ("CONFIRMED — row-2 heads are trimodal at a rate the "
                   "controls do not match. Table 1 transfers.")
    elif np.isfinite(uni_c) and uni_c > 0.7:
        verdict = ("FALSIFIED — row-2 heads are predominantly unimodal. "
                   "Table 1's geometry does not transfer past the "
                   "z = exp(-tV)x rescaling. Check rescaled_modality "
                   "before concluding the theorem fails rather than the "
                   "transfer.")
    elif np.isfinite(tri_o) and np.isfinite(tri_c) and abs(tri_c - tri_o) < 0.1:
        verdict = ("NO SIGNAL — candidates and controls are trimodal at the "
                   "same rate. Trimodality is a property of the activations, "
                   "not of the classification.")
    else:
        verdict = "MIXED — see the rates."

    return {"verdict": verdict,
            "trimodal_rate_candidates": tri_c, "n_candidates": n_c,
            "trimodal_rate_controls": tri_o, "n_controls": n_o,
            "unimodal_rate_candidates": uni_c,
            "equally_spaced_rate": spaced_rate}


def modality_stability(X: np.ndarray, phi: np.ndarray,
                       bw_values=(0.5, 0.75, 1.0, 1.5, 2.0), **kw) -> dict:
    """
    Mode count across a bandwidth scan.

    A mode count is only a measurement if it survives the smoothing choice.
    This reports the count at each bandwidth and whether the modal answer
    is stable across the middle of the range. `stable_n_modes` is the count
    that holds over at least three consecutive bandwidths; None means the
    modality of this projection is not determined by the data.
    """
    counts = {}
    for b in bw_values:
        r = projection_modality(X, phi, bw=b, **kw)
        counts[float(b)] = int(r.get("n_modes", 0))

    vals = list(counts.values())
    best, run, cur, prev = None, 0, 0, None
    for v in vals:
        cur = cur + 1 if v == prev else 1
        prev = v
        if cur > run:
            run, best = cur, v
    return {"counts_by_bw": counts, "stable_n_modes": best if run >= 3 else None,
            "longest_run": run, "n_bandwidths": len(vals)}
