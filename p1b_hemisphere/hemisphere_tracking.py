"""
hemisphere_tracking.py — Block 1 of Phase 1h.

Given the per-layer Fiedler bipartition from Block 0, match hemisphere A
at layer L to hemisphere A at layer L+1 across the global sign flip of
the Fiedler vector, then detect four kinds of hemisphere events.

What Block 0 hands us
---------------------
assignments[L, i] ∈ {0, 1}   sign-partition of the raw Fiedler vector.
                              The sign choice is arbitrary per layer —
                              without alignment, consecutive layers'
                              labels have no shared meaning.

What Block 1 produces
---------------------
aligned_assignments[L, i]    the same labels under a chosen global sign
                              per layer s.t. label 0 at layer L
                              matches label 0 at layer L-1 as closely
                              as possible (max Jaccard of the two
                              possible alignments).
match_overlap[L]             Jaccard overlap of the matched assignment
                              for transition L → L+1 (the larger of the
                              two, after sign flip).  1.0 means perfect
                              identity preservation; 0.5 means the
                              matched half has at most half its members
                              in common — a de facto swap.
axis_rotation[L]             arccos(|<v_L, v_{L+1}>|) in radians, the
                              angle between consecutive Fiedler vectors.
                              Same quantity as phase1.fiedler_tracking
                              but expressed as an angle rather than a
                              cosine, since angles sum more usefully
                              across layers.
crossing_count[L]            tokens that switch sides between L and L+1
                              in the aligned frame.
events                       list of {type, layer, detail} dicts with
                              types:
                                birth     collapsed → strong_bipartition
                                collapse  strong_bipartition → collapsed
                                swap      both strong_bipartition,
                                          match_overlap < 0.5
                                shear     crossing_count in top 5%,
                                          match_overlap ≥ 0.5

Cross-reference
---------------
Every event is tagged with whether the corresponding Phase 1 transition
has a merge event (from results["cluster_tracking"]["events"]) and
whether the destination layer is in energy_violations[beta=1.0].  The
caller supplies these as dicts and the tagging is additive metadata,
not filtering.

Functions
---------
align_hemisphere_labels    : global sign flip + Jaccard match per pair.
compute_axis_rotation      : per-transition angle between Fiedler vectors.
detect_events              : classify regime transitions and shear spikes.
crossref_phase1            : tag events with Phase 1 merge / violation overlap.
analyze_hemisphere_tracking: full pipeline.
hemisphere_tracking_to_json: flat per-transition + event list + summary.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Jaccard at k=2 with sign-flip correction
# ---------------------------------------------------------------------------

def _jaccard(a_mask: np.ndarray, b_mask: np.ndarray) -> float:
    """Jaccard of two boolean sets; 0 when the union is empty."""
    inter = int(np.logical_and(a_mask, b_mask).sum())
    union = int(np.logical_or(a_mask, b_mask).sum())
    if union == 0:
        return 0.0
    return inter / union


def _match_overlap(labels_prev: np.ndarray, labels_curr: np.ndarray) -> tuple[float, bool]:
    """
    For two 0/1 label vectors, return (best_overlap, needs_flip).

    Two candidate alignments:
      identity: (prev==0 ↔ curr==0) and (prev==1 ↔ curr==1)
      flipped:  (prev==0 ↔ curr==1) and (prev==1 ↔ curr==0)

    Score each as the mean Jaccard of its two matched pairs.  Return the
    higher-scoring alignment.  `needs_flip` is True when the flipped
    alignment wins (ties broken toward identity).
    """
    p0 = labels_prev == 0
    p1 = labels_prev == 1
    c0 = labels_curr == 0
    c1 = labels_curr == 1

    # Mean of the two within-pair Jaccards.
    id_score   = 0.5 * (_jaccard(p0, c0) + _jaccard(p1, c1))
    flip_score = 0.5 * (_jaccard(p0, c1) + _jaccard(p1, c0))

    if flip_score > id_score:
        return flip_score, True
    return id_score, False


def align_hemisphere_labels(
    assignments: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """
    Align hemisphere labels across layers under the global sign flip.

    Each layer's assignment is flipped or not so that consecutive valid
    layers maximize their pairwise Jaccard.  The first valid layer is
    the anchor (its labels are used as-is).

    An invalid layer is left with -1 labels AND resets the anchor: the
    next valid layer after an invalid stretch is aligned to the most
    recent valid layer before the gap.  This is conservative; the
    alternative (re-anchoring to the first valid-after-gap layer) would
    lose the opportunity to cross invalid stretches correctly.

    Parameters
    ----------
    assignments : (n_layers, n_tokens) in {-1, 0, 1}.
    valid       : (n_layers,) bool.

    Returns
    -------
    dict with:
      aligned_assignments : (n_layers, n_tokens) int8, -1 for invalid layers.
      flips_applied       : (n_layers,) bool — whether each layer's labels
                            were flipped from the input.
      match_overlap       : (n_layers - 1,) float, nan at invalid transitions.
    """
    n_layers, n_tokens = assignments.shape
    aligned = np.full_like(assignments, -1, dtype=np.int8)
    flips   = np.zeros(n_layers, dtype=bool)
    overlap = np.full(n_layers - 1, np.nan, dtype=np.float64)

    # Anchor = most recent valid layer's aligned labels.
    anchor_L      : int | None          = None
    anchor_labels : np.ndarray | None   = None

    for L in range(n_layers):
        if not valid[L]:
            continue
        raw = assignments[L].astype(np.int8)

        if anchor_L is None:
            aligned[L] = raw
            anchor_L      = L
            anchor_labels = aligned[L]
            continue

        score, needs_flip = _match_overlap(anchor_labels, raw)
        if needs_flip:
            aligned[L] = 1 - raw
            flips[L]   = True
        else:
            aligned[L] = raw

        # Only the immediately adjacent transition gets an overlap entry;
        # transitions across invalid gaps stay nan.
        if anchor_L == L - 1:
            overlap[L - 1] = score

        anchor_L      = L
        anchor_labels = aligned[L]

    return {
        "aligned_assignments": aligned,
        "flips_applied":       flips,
        "match_overlap":       overlap,
    }


# ---------------------------------------------------------------------------
# Axis rotation
# ---------------------------------------------------------------------------

def compute_axis_rotation(
    fiedler_vecs: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Angle between consecutive Fiedler vectors, in radians, up to global sign.

    Returns
    -------
    (n_layers - 1,) float, nan at invalid transitions.
    """
    n_layers = fiedler_vecs.shape[0]
    out = np.full(n_layers - 1, np.nan, dtype=np.float64)

    for L in range(n_layers - 1):
        if not (valid[L] and valid[L + 1]):
            continue
        v1 = fiedler_vecs[L]
        v2 = fiedler_vecs[L + 1]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            continue
        cos = float(abs(np.dot(v1, v2) / (n1 * n2)))
        cos = max(-1.0, min(1.0, cos))
        out[L] = float(np.arccos(cos))

    return out


# ---------------------------------------------------------------------------
# Crossing count in aligned frame
# ---------------------------------------------------------------------------

def aligned_crossing_count(
    aligned_assignments: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Per-transition token count of hemisphere switches in the aligned frame.

    Returns
    -------
    (n_layers - 1,) int, 0 at invalid transitions.
    """
    n_layers, _ = aligned_assignments.shape
    out = np.zeros(n_layers - 1, dtype=np.int32)
    for L in range(n_layers - 1):
        if not (valid[L] and valid[L + 1]):
            continue
        a = aligned_assignments[L]
        b = aligned_assignments[L + 1]
        out[L] = int((a != b).sum())
    return out


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

# Minimum match_overlap to consider identity preserved.  Below this, a
# strong↔strong transition is a "swap" (axis rotated onto a different
# partition of the token set).
IDENTITY_THRESHOLD = 0.5

# Shear = high crossing count without regime change or swap.  Expressed as
# a percentile of the valid crossing-count distribution within the run,
# rather than an absolute count, so it scales with prompt length.
SHEAR_PERCENTILE = 95.0


def detect_events(
    regime: np.ndarray,
    match_overlap: np.ndarray,
    crossing_count: np.ndarray,
    valid: np.ndarray,
    identity_threshold: float = IDENTITY_THRESHOLD,
    shear_percentile: float = SHEAR_PERCENTILE,
) -> list[dict]:
    """
    Emit birth / collapse / swap / shear events.

    Parameters
    ----------
    regime         : (n_layers,) str from Block 0.
    match_overlap  : (n_layers - 1,) float, nan at invalid transitions.
    crossing_count : (n_layers - 1,) int.
    valid          : (n_layers,) bool.

    Event schema:
      {"type": "birth"|"collapse"|"swap"|"shear",
       "layer": int,          # destination layer (L+1 for transitions)
       "from_layer": int,     # origin layer
       "detail": {...}}
    """
    events: list[dict] = []
    n_layers = len(regime)
    n_trans  = n_layers - 1

    # Shear threshold from the valid crossing distribution.
    valid_trans = np.array(
        [valid[L] and valid[L + 1] for L in range(n_trans)], dtype=bool
    )
    cc_valid = crossing_count[valid_trans]
    if cc_valid.size >= 5:
        shear_thresh = float(np.percentile(cc_valid, shear_percentile))
    else:
        shear_thresh = float("inf")   # don't emit shear on tiny runs

    for L in range(n_trans):
        if not valid_trans[L]:
            continue
        r_from = str(regime[L])
        r_to   = str(regime[L + 1])
        ov     = float(match_overlap[L]) if np.isfinite(match_overlap[L]) else None
        cc     = int(crossing_count[L])

        # Birth / collapse — regime transitions first.
        if r_from == "collapsed" and r_to == "strong_bipartition":
            events.append({
                "type": "birth",
                "layer": L + 1,
                "from_layer": L,
                "detail": {
                    "match_overlap": ov,
                    "crossing_count": cc,
                },
            })
            continue
        if r_from == "strong_bipartition" and r_to == "collapsed":
            events.append({
                "type": "collapse",
                "layer": L + 1,
                "from_layer": L,
                "detail": {
                    "match_overlap": ov,
                    "crossing_count": cc,
                },
            })
            continue

        # Swap — both strong, axis has rotated to a different partition.
        if (r_from == "strong_bipartition"
                and r_to == "strong_bipartition"
                and ov is not None
                and ov < identity_threshold):
            events.append({
                "type": "swap",
                "layer": L + 1,
                "from_layer": L,
                "detail": {
                    "match_overlap": ov,
                    "crossing_count": cc,
                },
            })
            continue

        # Shear — high crossings but identity preserved (or not yet
        # meaningfully defined, e.g., weak↔weak).
        if cc >= shear_thresh and cc > 0:
            events.append({
                "type": "shear",
                "layer": L + 1,
                "from_layer": L,
                "detail": {
                    "match_overlap": ov,
                    "crossing_count": cc,
                    "shear_threshold": shear_thresh,
                    "regime_from": r_from,
                    "regime_to":   r_to,
                },
            })

    return events


# ---------------------------------------------------------------------------
# Cross-reference with Phase 1
# ---------------------------------------------------------------------------

def crossref_phase1(
    events: list[dict],
    axis_rotation: np.ndarray,
    crossing_count: np.ndarray,
    merge_transition_indices: set[int] | None = None,
    violation_layers: set[int] | None = None,
) -> dict:
    """
    Tag events with overlap against Phase 1 signals and compute aggregates.

    Parameters
    ----------
    events                    : from detect_events.
    axis_rotation             : (n_trans,) float.
    crossing_count            : (n_trans,) int.
    merge_transition_indices  : set of transition indices L where the
                                Phase 1 cluster_tracking event[L] reports
                                at least one merge.  Transition L is the
                                L → L+1 layer pair.
    violation_layers          : set of destination layer indices that
                                appear in energy_violations[beta=1.0].

    Returns
    -------
    dict with:
      events            : same list with "phase1" field added in detail.
      agg               : aggregates over the whole trajectory.
    """
    merges = merge_transition_indices or set()
    viols  = violation_layers         or set()

    out_events: list[dict] = []
    for ev in events:
        trans = ev["from_layer"]
        tag = {
            "merge_at_transition":    bool(trans in merges),
            "violation_at_layer":     bool(ev["layer"] in viols),
        }
        new = dict(ev)
        new["detail"] = dict(ev["detail"])
        new["detail"]["phase1"] = tag
        out_events.append(new)

    # Aggregates: axis rotation at merge vs non-merge transitions; crossing
    # count at violation vs non-violation transitions.  Use the whole
    # axis_rotation / crossing_count arrays, not just the event indices.
    n_trans = len(axis_rotation)
    merge_rot   : list[float] = []
    nomerge_rot : list[float] = []
    viol_cc     : list[int]   = []
    noviol_cc   : list[int]   = []

    for L in range(n_trans):
        r = axis_rotation[L]
        if np.isfinite(r):
            (merge_rot if L in merges else nomerge_rot).append(float(r))
        (viol_cc if (L + 1) in viols else noviol_cc).append(int(crossing_count[L]))

    def _m(xs):
        return float(np.mean(xs)) if xs else None

    agg = {
        "n_merges_in_run":             len(merges),
        "n_violations_in_run":         len(viols),
        "mean_axis_rotation_at_merge":    _m(merge_rot),
        "mean_axis_rotation_off_merge":   _m(nomerge_rot),
        "mean_crossing_at_violation":     _m(viol_cc),
        "mean_crossing_off_violation":    _m(noviol_cc),
    }

    return {"events": out_events, "agg": agg}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_hemisphere_tracking(
    block0: dict,
    merge_transition_indices: set[int] | None = None,
    violation_layers: set[int] | None = None,
    identity_threshold: float = IDENTITY_THRESHOLD,
    shear_percentile: float = SHEAR_PERCENTILE,
) -> dict:
    """
    Run Block 1 on a Block 0 result.

    Parameters
    ----------
    block0 : output of bipartition_detect.analyze_bipartition.

    Returns
    -------
    dict with everything Block 1 produces, ready for the aggregator.
    """
    aligned = align_hemisphere_labels(block0["assignments"], block0["valid"])
    axis_rot = compute_axis_rotation(block0["fiedler_vecs"], block0["valid"])
    cc = aligned_crossing_count(aligned["aligned_assignments"], block0["valid"])

    raw_events = detect_events(
        block0["regime"],
        aligned["match_overlap"],
        cc,
        block0["valid"],
        identity_threshold=identity_threshold,
        shear_percentile=shear_percentile,
    )
    xref = crossref_phase1(
        raw_events,
        axis_rot,
        cc,
        merge_transition_indices=merge_transition_indices,
        violation_layers=violation_layers,
    )

    return {
        "aligned_assignments": aligned["aligned_assignments"],
        "flips_applied":       aligned["flips_applied"],
        "match_overlap":       aligned["match_overlap"],
        "axis_rotation":       axis_rot,
        "crossing_count":      cc,
        "events":              xref["events"],
        "crossref":            xref["agg"],
        "thresholds": {
            "identity": identity_threshold,
            "shear_percentile": shear_percentile,
        },
    }


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def hemisphere_tracking_to_json(result: dict) -> dict:
    """Flat representation for the aggregator's per-run JSON."""
    n_trans = len(result["axis_rotation"])

    per_transition = []
    for L in range(n_trans):
        per_transition.append({
            "transition":      L,
            "from_layer":      L,
            "to_layer":        L + 1,
            "match_overlap":   _f(result["match_overlap"][L]),
            "axis_rotation":   _f(result["axis_rotation"][L]),
            "crossing_count":  int(result["crossing_count"][L]),
        })

    # Event type counts.
    counts: dict[str, int] = {}
    for ev in result["events"]:
        counts[ev["type"]] = counts.get(ev["type"], 0) + 1

    # Aggregates: mean axis rotation, max overlap, crossing counts, event totals.
    ar = np.asarray(result["axis_rotation"], dtype=np.float64)
    mo = np.asarray(result["match_overlap"], dtype=np.float64)
    cc = np.asarray(result["crossing_count"], dtype=np.float64)

    summary = {
        "n_transitions":       n_trans,
        "n_events":            len(result["events"]),
        "event_counts":        counts,
        "mean_axis_rotation":  _mean(ar),
        "max_axis_rotation":   _max(ar),
        "mean_match_overlap":  _mean(mo),
        "min_match_overlap":   _min(mo),
        "mean_crossing":       _mean(cc),
        "max_crossing":        _max(cc),
        "identity_preserved_fraction":
            _frac_ge(mo, IDENTITY_THRESHOLD),
        "thresholds":          result["thresholds"],
        "crossref":            result["crossref"],
    }

    return {
        "per_transition": per_transition,
        "events":         result["events"],
        "summary":        summary,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _mean(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def _max(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else None


def _min(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.min()) if arr.size else None


def _frac_ge(arr, threshold):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float((arr >= threshold).mean())
