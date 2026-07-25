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
"""
APPEND THIS BLOCK to the end of p5b_manifold_steering/isometry_test.py.

Nothing existing is modified. `hellinger_distance`,
`geodesic_distance_manifold`, `geodesic_distance_behavior`,
`pairwise_distances`, `isometry_score`, `pairwise_to_matrix` and
`mds_embed` all stay exactly as they are — `pairwise_distances` and
`isometry_score` are unit-tested in test_phase5b.py, and the geodesic path
is still the right machinery for steering once that lands.

Add to the module's import block if not already present:

    from scipy.stats import pearsonr, spearmanr

--------------------------------------------------------------------------
WHAT THIS REPLACES AND WHY

Sub-exp B used to route through the fitted splines: d_manifold as arc
length along Mh, d_behavior as arc length along My, both evaluated at
arc-length coordinates u. Three things were wrong with that.

1. ORDERING. u comes from arc_length_params, whose "ordered sequence"
   precondition nothing upstream establishes — load_plateau_centroids
   iterates trajectories in cluster BIRTH order. Wurgaft's control points
   carry an intrinsic sequence; ours do not. The spline was threading a
   zigzag.

2. CIRCULARITY. d_manifold(u_i, u_j) is arc length along Mh between two
   coordinates *defined* as normalized arc length along Mh, so it is
   ≈ |u_i − u_j| × total_length almost by construction. The test's entire
   discriminating power sat on the My side.

3. DEGENERATE CONTROL. load_plateau_centroids returns L2-NORMALIZED
   centroids, so the old d_linear = ‖c_i − c_j‖ *is* the sphere-frame
   chordal distance. Drop the spline and d_linear becomes identical to
   sphere-frame d_manifold, making r_manifold == r_linear and P5b-B1/B3
   unfalsifiable. The spline was masking this; the contrast used to be
   curve-vs-chord, which needs the ordering we don't have (1).

Sub-exp B does not need interpolation — it only ever evaluates at control
points, and isometry between two finite point sets is the correlation of
their two pairwise-distance vectors. So: direct distances, and a control
redefined as FRAME-VS-RAW.

  d_frame : distance between centroids in the frame the model reads
  d_raw   : Euclidean distance between UN-normalized centroids, i.e. the
            ambient residual stream

P5b-B1 now reads: *the frame the model reads in predicts behavior better
than the ambient stream does.* This is not Wurgaft's B1. His curve-vs-chord
version is deferred as P5b-B1′, pending seriation — see WORKING-5b.md §3c.

--------------------------------------------------------------------------
PRE-REGISTRATION (design-5b.md §4). Read before touching thresholds.

Primary, verdict-bearing : frame="sphere", behavior_metric="hellinger"
Control                  : frame="raw", renormalize=False
Secondary, never verdict : frame="ln"; behavior_metric="sym_kl"

All four readings are written every run, labeled, with NO selection. The
design doc's own warning is the reason: the premise "is an identity claim
that could otherwise be argued into a positive result from ambiguous
correlations." Reporting the best of four IS that failure mode.
"""

# ===========================================================================
# --------------------------- BEGIN APPEND BLOCK ---------------------------
# ===========================================================================

import numpy as np
from scipy.stats import pearsonr, spearmanr


# --- Pre-registered constants ----------------------------------------------

SPHERE_GAP_ESCALATION = 0.10
"""
If core.polar.sphere_gap's max_pearson_gap over the plateau layers exceeds
this, the LN-frame reading becomes the headline and sphere is demoted, with
the gap recorded as justification.

Follows core/polar.py's own written contract: "where gaps are ~0,
sphere-frame conclusions transfer to the raw stream; where they spike, that
layer's sphere-frame clustering claims need the LN-frame / functional
arbiter before being trusted."

STATUS: proposed, not signed off. Fix or explicitly accept before the first
real run. Changing it AFTER seeing results is exactly the post-hoc move the
design doc prohibits.
"""

P5B_B2_THRESHOLD = 0.70
"""r_frame must exceed this. Calibrated against Wurgaft's 0.89-0.999."""

P5B_B3_THRESHOLD = 0.10
"""r_frame - r_raw must exceed this."""

PRIMARY_ACTIVATION_FRAME = "sphere"
PRIMARY_BEHAVIOR_METRIC  = "hellinger"

ISOMETRY_SCHEMA = "p5b_isometry_v2"


# --- Distance assembly ------------------------------------------------------

def direct_pairwise_distances(
    centroids_by_frame: dict,
    distributions,
    behavior_metric: str = "hellinger",
) -> dict:
    """
    Pairwise distance vectors for every requested activation frame plus the
    behavior side, all over the SAME point set in the SAME order.

    Parameters
    ----------
    centroids_by_frame : {frame_name: (n, d) array}. Every array must have
                         the same n and the same row ordering — they are
                         different frames of one point set, not different
                         point sets. Build them with
                         p5b_distances.frame_centroids using one shared
                         traj_ids list.
    distributions      : (n, vocab) — one output distribution per point,
                         from stack_behavior_by_traj_ids over the same
                         traj_ids.
    behavior_metric    : "hellinger" | "sym_kl"

    Returns
    -------
    {"d_act": {frame: (n_pairs,)}, "d_behavior": (n_pairs,),
     "n_points": int, "n_pairs": int, "behavior_metric": str}

    Raises on any row-count disagreement. That check is the whole point of
    this function: the bug it exists to prevent was two populations being
    silently compared, and a length assertion here is cheap relative to a
    meaningless correlation nobody notices.
    """
    from p5b_manifold_steering.p5b_distances import (
        activation_distance_matrix, behavior_distance_matrix, upper_triangle,
    )

    if not centroids_by_frame:
        raise ValueError("direct_pairwise_distances: no frames supplied")

    P = np.asarray(distributions)
    n = P.shape[0]
    for name, C in centroids_by_frame.items():
        C = np.asarray(C)
        if C.shape[0] != n:
            raise ValueError(
                f"direct_pairwise_distances: frame {name!r} has {C.shape[0]} "
                f"rows but there are {n} distributions. These must be the "
                f"same point set in the same order — build both by "
                f"iterating one traj_ids list."
            )
    if n < 3:
        raise ValueError(
            f"direct_pairwise_distances: need >= 3 points for a meaningful "
            f"correlation, got {n}"
        )

    d_act = {}
    for name, C in centroids_by_frame.items():
        # "raw" keeps norm information, so chordal is a genuine ambient
        # Euclidean distance there; on unit-norm frames it is the chord of
        # the great-circle arc. Same metric either way, deliberately, so
        # frame is the only thing varying between readings.
        d_act[name] = upper_triangle(
            activation_distance_matrix(np.asarray(C), metric="chordal")
        )

    d_beh = upper_triangle(behavior_distance_matrix(P, metric=behavior_metric))

    return {
        "d_act":           d_act,
        "d_behavior":      d_beh,
        "n_points":        int(n),
        "n_pairs":         int(d_beh.shape[0]),
        "behavior_metric": behavior_metric,
    }


# --- Scoring ----------------------------------------------------------------

def _safe_corr(x, y) -> tuple:
    """
    (pearson_r, spearman_rho), with None for either that is undefined.

    A constant input makes the correlation undefined, and scipy emits
    ConstantInputWarning and returns nan rather than raising. That happens
    for real here — a degenerate frame collapses all distances — and it is
    a finding, not an error, so it is recorded as None rather than swallowed
    or escalated.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None, None
    r, _   = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    r   = float(r)   if np.isfinite(r)   else None
    rho = float(rho) if np.isfinite(rho) else None
    return r, rho


def frame_vs_raw_score(d_frame, d_raw, d_behavior) -> dict:
    """
    Score one (activation frame, behavior metric) reading against the raw
    control.

    Returns
    -------
    {"r_frame", "r_raw", "delta", "rho_frame", "rho_raw",
     "b1_pass", "b2_pass", "b3_pass", "n_pairs"}

    b1 : r_frame > r_raw            — read frame beats ambient stream
    b2 : r_frame > P5B_B2_THRESHOLD — the correspondence is actually strong
    b3 : delta   > P5B_B3_THRESHOLD — and the margin is not noise

    All three are False when r_frame is None (undefined correlation). A
    degenerate frame must never pass by default.
    """
    r_f, rho_f = _safe_corr(d_frame, d_behavior)
    r_r, rho_r = _safe_corr(d_raw,   d_behavior)

    delta = (r_f - r_r) if (r_f is not None and r_r is not None) else None

    return {
        "r_frame":   r_f,
        "r_raw":     r_r,
        "delta":     delta,
        "rho_frame": rho_f,
        "rho_raw":   rho_r,
        "b1_pass":   bool(r_f is not None and r_r is not None and r_f > r_r),
        "b2_pass":   bool(r_f is not None and r_f > P5B_B2_THRESHOLD),
        "b3_pass":   bool(delta is not None and delta > P5B_B3_THRESHOLD),
        "n_pairs":   int(np.asarray(d_behavior).shape[0]),
    }


# --- Orchestration ----------------------------------------------------------

def resolve_primary_frame(frame_diagnostics: dict | None) -> dict:
    """
    Apply the pre-registered escalation rule.

    Returns {"activation_frame", "triggered", "max_pearson_gap",
             "threshold", "reason"}.

    Escalates sphere -> ln only when max_pearson_gap exceeds
    SPHERE_GAP_ESCALATION. Missing diagnostics do NOT escalate: an absent
    measurement is not evidence that the frames disagree.
    """
    gap = None
    if frame_diagnostics:
        gap = frame_diagnostics.get("max_pearson_gap")

    triggered = gap is not None and np.isfinite(gap) and gap > SPHERE_GAP_ESCALATION

    if triggered:
        reason = (
            f"max_pearson_gap={gap:.4f} > {SPHERE_GAP_ESCALATION}; sphere-frame "
            f"geometry diverges from the raw stream, so the LN-frame reading "
            f"is the headline per the pre-registered escalation rule."
        )
        frame = "ln"
    elif gap is None:
        reason = (
            "no sphere_gap diagnostics available; defaulting to the "
            "pre-registered primary frame without escalation."
        )
        frame = PRIMARY_ACTIVATION_FRAME
    else:
        reason = (
            f"max_pearson_gap={gap:.4f} <= {SPHERE_GAP_ESCALATION}; sphere-frame "
            f"conclusions transfer, no escalation."
        )
        frame = PRIMARY_ACTIVATION_FRAME

    return {
        "activation_frame": frame,
        "triggered":        bool(triggered),
        "max_pearson_gap":  (float(gap) if gap is not None else None),
        "threshold":        SPHERE_GAP_ESCALATION,
        "reason":           reason,
    }


def run_isometry_direct(
    centroids_by_frame: dict,
    distributions,
    traj_ids,
    frame_diagnostics: dict | None = None,
    behavior_space: str = "hellinger",
    behavior_metrics=("hellinger", "sym_kl"),
    ln_available: bool = True,
    coverage: dict | None = None,
) -> dict:
    """
    Full Sub-exp B. Produces the isometry.json payload.

    Parameters
    ----------
    centroids_by_frame : {frame: (n, d)}. MUST contain "raw" (the control)
                         and the primary frame. "ln" optional.
    distributions      : (n, vocab), same order as every centroid array
    traj_ids           : the identity list these rows correspond to;
                         recorded so a later reader can trace any row back
                         to a Phase 1 trajectory
    frame_diagnostics  : sphere_gap_by_layer output; drives escalation
    behavior_space     : how per-trajectory distributions were AGGREGATED
                         ("hellinger"/"mixture"). Recorded, not used here —
                         it changes the numbers, so it belongs in the
                         artifact.
    behavior_metrics   : which behavior distances to compute
    ln_available       : False on architectures where core/ln_frame.py has
                         no extraction path (GPT-2 today). When False, any
                         "ln" reading is omitted and the omission is
                         recorded rather than silently reporting a sphere
                         number under an "ln" label.
    coverage           : compute_behavior_trajectories' coverage dict,
                         summarized into the artifact

    Returns
    -------
    dict, schema p5b_isometry_v2. Verdicts are read from the PRIMARY block
    only; every other reading is present and labeled but non-verdict-bearing.
    """
    if "raw" not in centroids_by_frame:
        raise ValueError(
            "run_isometry_direct: the 'raw' control frame is required. "
            "Build it with p5b_distances.frame_centroids(..., frame='raw', "
            "renormalize=False)."
        )

    esc   = resolve_primary_frame(frame_diagnostics)
    frame = esc["activation_frame"]

    if frame == "ln" and (not ln_available or "ln" not in centroids_by_frame):
        frame = PRIMARY_ACTIVATION_FRAME
        esc["reason"] += (
            " LN escalation requested but no LN frame is available on this "
            "architecture (see core/ln_frame.py: GPT-NeoX only); falling "
            "back to sphere AND RECORDING THE FALLBACK."
        )
        esc["ln_fallback"] = True
    else:
        esc["ln_fallback"] = False

    if frame not in centroids_by_frame:
        raise ValueError(
            f"run_isometry_direct: primary frame {frame!r} not in "
            f"centroids_by_frame (have {sorted(centroids_by_frame)})"
        )

    readings = []
    primary_block = None

    for metric in behavior_metrics:
        pw = direct_pairwise_distances(
            centroids_by_frame, distributions, behavior_metric=metric
        )
        for fname, d_f in pw["d_act"].items():
            if fname == "raw":
                continue
            if fname == "ln" and not ln_available:
                continue
            score = frame_vs_raw_score(d_f, pw["d_act"]["raw"], pw["d_behavior"])
            block = {
                "activation_frame": fname,
                "behavior_metric":  metric,
                "is_primary":       bool(fname == frame and metric == PRIMARY_BEHAVIOR_METRIC),
                **score,
            }
            readings.append(block)
            if block["is_primary"]:
                primary_block = block

    if primary_block is None:
        raise ValueError(
            f"run_isometry_direct: no primary reading produced for frame="
            f"{frame!r}, metric={PRIMARY_BEHAVIOR_METRIC!r}. Check that "
            f"behavior_metrics includes the pre-registered primary metric."
        )

    cov_summary = None
    if coverage:
        fracs = [c["frac"] for c in coverage.values()]
        cov_summary = {
            "n_trajectories": len(coverage),
            "mean_frac":      float(np.mean(fracs)) if fracs else 0.0,
            "min_frac":       float(np.min(fracs))  if fracs else 0.0,
            "n_full":         int(sum(1 for f in fracs if f >= 1.0 - 1e-9)),
        }

    return {
        "schema":   ISOMETRY_SCHEMA,
        "n_points": int(np.asarray(distributions).shape[0]),
        "n_pairs":  int(readings[0]["n_pairs"]),
        "traj_ids": [int(t) for t in traj_ids],
        "primary": {
            "activation_frame": frame,
            "behavior_metric":  PRIMARY_BEHAVIOR_METRIC,
            "control_frame":    "raw",
        },
        "behavior_aggregation_space": behavior_space,
        "escalation":  esc,
        "ln_available": bool(ln_available),
        "coverage":    cov_summary,
        "readings":    readings,
        "thresholds": {
            "P5b-B2": P5B_B2_THRESHOLD,
            "P5b-B3": P5B_B3_THRESHOLD,
            "sphere_gap_escalation": SPHERE_GAP_ESCALATION,
        },
        "verdict": {
            "P5b-B1": primary_block["b1_pass"],
            "P5b-B2": primary_block["b2_pass"],
            "P5b-B3": primary_block["b3_pass"],
            "r_frame": primary_block["r_frame"],
            "r_raw":   primary_block["r_raw"],
            "delta":   primary_block["delta"],
        },
        "notes": (
            "P5b-B1 here is FRAME-VS-RAW: does the frame the model reads in "
            "predict behavior better than the ambient residual stream. This "
            "is not Wurgaft's curve-vs-chord B1, which is deferred as "
            "P5b-B1' pending a seriation method for the control points "
            "(WORKING-5b.md 3a/3c). Secondary readings are reported for "
            "completeness and are NOT verdict-bearing."
        ),
    }


# ===========================================================================
# ---------------------------- END APPEND BLOCK ----------------------------
# ===========================================================================