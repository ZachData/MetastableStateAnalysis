"""
hemisphere_tracking.py — Block 1 of Phase 1h.

Given the per-layer Fiedler bipartition from Block 0, match hemisphere A
at layer L to hemisphere A at layer L+1 across the global sign flip of
the Fiedler vector, then detect five kinds of hemisphere events.

What Block 0 hands us
---------------------
assignments[L, i] ∈ {0, 1}   sign-partition of the raw Fiedler vector.
                              The sign choice is arbitrary per layer.

What Block 1 produces
---------------------
aligned_assignments[L, i]    labels under a chosen global sign per layer
                              s.t. label 0 at layer L matches label 0 at
                              layer L-1 as closely as possible (max
                              Jaccard of the two possible alignments).
match_overlap[L]             Jaccard overlap for transition L → L+1.
axis_rotation[L]             arccos(|<v_L, v_{L+1}>|) in radians.
crossing_count[L]            tokens that switch sides between L and L+1.
cumulative_axis_rotation[L]  nan-safe running sum of axis_rotation.
persistence_length[L]        consecutive strong_bipartition layers since
                             last disruptive event.
events                       list of {type, layer, detail} dicts:
                               birth   collapsed → strong_bipartition
                               collapse strong_bipartition → collapsed
                               swap    both strong, match_overlap < 0.5
                               shear   high crossing count
                               drift   sustained axis rotation over window

Cross-reference
---------------
Events are tagged with:
  phase1.at_merge          : bool — transition coincides with a Phase 1
                             cluster merge event.
  phase1.at_violation_layer: bool — destination layer is an energy-
                             violation layer at beta=1.0.

Functions
---------
align_hemisphere_labels       : global sign flip + Jaccard match per pair.
compute_axis_rotation         : per-transition angle between Fiedler vectors.
compute_cumulative_rotation   : nan-safe running sum of axis_rotation.
compute_persistence_lengths   : per-layer stable-run length.
aligned_crossing_count        : per-transition hemisphere switch count.
detect_events                 : classify regime transitions, shear, drift.
crossref_phase1               : tag events with Phase 1 overlap.
analyze_hemisphere_tracking   : full pipeline.
hemisphere_tracking_to_json   : flat per-transition + event list + summary.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Jaccard at k=2 with sign-flip correction
# ---------------------------------------------------------------------------

def _jaccard(a_mask: np.ndarray, b_mask: np.ndarray) -> float:
    inter = int(np.logical_and(a_mask, b_mask).sum())
    union = int(np.logical_or(a_mask, b_mask).sum())
    if union == 0:
        return 0.0
    return inter / union


def _match_overlap(labels_prev: np.ndarray, labels_curr: np.ndarray) -> tuple[float, bool]:
    """
    Return (best_overlap, needs_flip) for two 0/1 label vectors.
    Ties broken toward identity (no flip).
    """
    p0, p1 = labels_prev == 0, labels_prev == 1
    c0, c1 = labels_curr == 0, labels_curr == 1
    id_score   = 0.5 * (_jaccard(p0, c0) + _jaccard(p1, c1))
    flip_score = 0.5 * (_jaccard(p0, c1) + _jaccard(p1, c0))
    if flip_score > id_score:
        return flip_score, True
    return id_score, False


# ---------------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------------

def align_hemisphere_labels(
    assignments: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """
    Align hemisphere labels across layers under the global sign flip.

    Parameters
    ----------
    assignments : (n_layers, n_tokens) in {-1, 0, 1}.
    valid       : (n_layers,) bool.

    Returns
    -------
    dict with:
      aligned_assignments : (n_layers, n_tokens) int8, -1 for invalid layers.
      flips_applied       : (n_layers,) bool.
      match_overlap       : (n_layers - 1,) float, nan at invalid transitions.
    """
    n_layers, n_tokens = assignments.shape
    aligned = np.full_like(assignments, -1, dtype=np.int8)
    flips   = np.zeros(n_layers, dtype=bool)
    overlap = np.full(n_layers - 1, np.nan, dtype=np.float64)

    anchor_L:      int | None        = None
    anchor_labels: np.ndarray | None = None

    for L in range(n_layers):
        if not valid[L]:
            continue
        raw = assignments[L].astype(np.int8)

        if anchor_L is None:
            aligned[L]    = raw
            anchor_L      = L
            anchor_labels = aligned[L]
            continue

        score, needs_flip = _match_overlap(anchor_labels, raw)
        if needs_flip:
            aligned[L] = 1 - raw
            flips[L]   = True
        else:
            aligned[L] = raw

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
    Angle between consecutive Fiedler vectors (radians), up to global sign.
    Returns (n_layers - 1,) float, nan at invalid transitions.
    """
    n_layers = fiedler_vecs.shape[0]
    out = np.full(n_layers - 1, np.nan, dtype=np.float64)

    for L in range(n_layers - 1):
        if not (valid[L] and valid[L + 1]):
            continue
        v1, v2 = fiedler_vecs[L], fiedler_vecs[L + 1]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            continue
        cos = float(abs(np.dot(v1, v2) / (n1 * n2)))
        cos = max(-1.0, min(1.0, cos))
        out[L] = float(np.arccos(cos))

    return out


# ---------------------------------------------------------------------------
# Cumulative axis rotation
# ---------------------------------------------------------------------------

def compute_cumulative_rotation(axis_rotation: np.ndarray) -> np.ndarray:
    """
    Nan-safe running sum of per-transition axis rotation.
    nan transitions contribute 0 (treated as no-rotation, not a reset).
    Returns (n_transitions,) float.
    """
    out     = np.zeros(len(axis_rotation), dtype=np.float64)
    running = 0.0
    for i, v in enumerate(axis_rotation):
        if np.isfinite(v):
            running += float(v)
        out[i] = running
    return out


# ---------------------------------------------------------------------------
# Persistence length
# ---------------------------------------------------------------------------

def compute_persistence_lengths(
    regime: np.ndarray,
    events: list[dict],
) -> np.ndarray:
    """
    Per-layer count of consecutive strong_bipartition layers since the
    most recent disruptive event (birth, collapse, swap, drift).
    Shear is not disruptive.  Non-strong layers get 0.
    Returns (n_layers,) int.
    """
    n = len(regime)
    out = np.zeros(n, dtype=np.int32)
    disruptive_types  = {"birth", "collapse", "swap", "drift"}
    disruptive_layers: set[int] = {
        ev["layer"] for ev in events if ev["type"] in disruptive_types
    }

    run = 0
    for L in range(n):
        if str(regime[L]) != "strong_bipartition":
            run   = 0
            out[L] = 0
            continue
        run = 1 if L in disruptive_layers else run + 1
        out[L] = run

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
    Returns (n_layers - 1,) int, 0 at invalid transitions.
    """
    n_layers, _ = aligned_assignments.shape
    out = np.zeros(n_layers - 1, dtype=np.int32)
    for L in range(n_layers - 1):
        if not (valid[L] and valid[L + 1]):
            continue
        a, b   = aligned_assignments[L], aligned_assignments[L + 1]
        out[L] = int(((a != b) & (a >= 0) & (b >= 0)).sum())
    return out


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

IDENTITY_THRESHOLD  = 0.5
SHEAR_PERCENTILE    = 95.0
SHEAR_ABSOLUTE_FLOOR = 3
DRIFT_WINDOW_LAYERS  = 4
DRIFT_WINDOW_RAD     = np.pi / 4.0


def detect_events(
    regime: np.ndarray,
    match_overlap: np.ndarray,
    crossing_count: np.ndarray,
    valid: np.ndarray,
    axis_rotation: np.ndarray | None = None,
    identity_threshold: float = IDENTITY_THRESHOLD,
    shear_percentile: float = SHEAR_PERCENTILE,
    shear_absolute_floor: int = SHEAR_ABSOLUTE_FLOOR,
    drift_window_layers: int = DRIFT_WINDOW_LAYERS,
    drift_window_rad: float = DRIFT_WINDOW_RAD,
) -> list[dict]:
    """
    Emit birth / collapse / swap / shear / drift events.

    Parameters
    ----------
    regime         : (n_layers,) str from Block 0.
    match_overlap  : (n_layers - 1,) float, nan at invalid transitions.
    crossing_count : (n_layers - 1,) int.
    valid          : (n_layers,) bool.
    axis_rotation  : (n_layers - 1,) float — required for drift detection.

    Event schema
    ------------
    {"type": str, "layer": int, "from_layer": int, "detail": dict}
    """
    events:  list[dict] = []
    n_layers = len(regime)
    n_trans  = n_layers - 1

    valid_trans = np.array(
        [valid[L] and valid[L + 1] for L in range(n_trans)], dtype=bool
    )
    cc_valid = crossing_count[valid_trans]
    shear_thresh_pct = (
        float(np.percentile(cc_valid, shear_percentile))
        if cc_valid.size >= 5 else float("inf")
    )

    disrupted_transitions: set[int] = set()

    for L in range(n_trans):
        r_from = str(regime[L])
        r_to   = str(regime[L + 1])
        ov = float(match_overlap[L]) if np.isfinite(match_overlap[L]) else None
        cc = int(crossing_count[L])

        # Birth/collapse cross the valid/invalid boundary — must precede the guard
        if r_from == "collapsed" and r_to == "strong_bipartition":
            events.append({"type": "birth", "layer": L + 1, "from_layer": L,
                            "detail": {"match_overlap": ov, "crossing_count": cc}})
            disrupted_transitions.add(L)
            continue

        if r_from == "strong_bipartition" and r_to == "collapsed":
            events.append({"type": "collapse", "layer": L + 1, "from_layer": L,
                            "detail": {"match_overlap": ov, "crossing_count": cc}})
            disrupted_transitions.add(L)
            continue

        if not valid_trans[L]:
            continue
        r_from = str(regime[L])
        r_to   = str(regime[L + 1])
        ov     = float(match_overlap[L]) if np.isfinite(match_overlap[L]) else None
        cc     = int(crossing_count[L])

        if r_from == "collapsed" and r_to == "strong_bipartition":
            events.append({
                "type": "birth", "layer": L + 1, "from_layer": L,
                "detail": {"match_overlap": ov, "crossing_count": cc},
            })
            disrupted_transitions.add(L)
            continue

        if r_from == "strong_bipartition" and r_to == "collapsed":
            events.append({
                "type": "collapse", "layer": L + 1, "from_layer": L,
                "detail": {"match_overlap": ov, "crossing_count": cc},
            })
            disrupted_transitions.add(L)
            continue

        if (r_from == "strong_bipartition"
                and r_to == "strong_bipartition"
                and ov is not None
                and ov < identity_threshold):
            events.append({
                "type": "swap", "layer": L + 1, "from_layer": L,
                "detail": {"match_overlap": ov, "crossing_count": cc},
            })
            disrupted_transitions.add(L)
            continue

        is_high_pct = cc >= shear_thresh_pct
        is_high_abs = (shear_absolute_floor > 0
                       and cc >= shear_absolute_floor
                       and cc_valid.size >= 5)
        if (is_high_pct or is_high_abs) and cc > 0:
            events.append({
                "type": "shear", "layer": L + 1, "from_layer": L,
                "detail": {
                    "match_overlap":        ov,
                    "crossing_count":       cc,
                    "shear_threshold_pct":  shear_thresh_pct,
                    "shear_absolute_floor": shear_absolute_floor,
                    "triggered_by":         "percentile" if is_high_pct else "absolute_floor",
                    "regime_from":          r_from,
                    "regime_to":            r_to,
                },
            })

    # Drift pass
    if axis_rotation is not None and drift_window_layers >= 2:
        W = drift_window_layers
        for L_end in range(W - 1, n_trans):
            L_start = L_end - W + 1
            if any(t in disrupted_transitions for t in range(L_start, L_end + 1)):
                continue
            if not all(valid_trans[L_start : L_end + 1]):
                continue
            window_rot = axis_rotation[L_start : L_end + 1]
            if not np.all(np.isfinite(window_rot)):
                continue
            cum_rot = float(window_rot.sum())
            if cum_rot >= drift_window_rad:
                events.append({
                    "type": "drift",
                    "layer": L_end + 1,
                    "from_layer": L_start,
                    "detail": {
                        "window_start":        L_start,
                        "window_end":          L_end + 1,
                        "cumulative_rotation": cum_rot,
                        "window_threshold":    drift_window_rad,
                        "n_transitions":       W,
                        "regime_at_end":       str(regime[L_end + 1]),
                    },
                })

    events.sort(key=lambda e: (e["layer"], e["from_layer"]))
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
    Tag events with Phase 1 overlap and compute aggregates.

    Each event's detail gains a "phase1" sub-dict with:
      at_merge          : bool — transition index is in merge_transition_indices.
      at_violation_layer: bool — destination layer is in violation_layers.
    """
    merges = merge_transition_indices or set()
    viols  = violation_layers         or set()

    out_events: list[dict] = []
    for ev in events:
        trans = ev["from_layer"]
        # FIX: keys were "merge_at_transition" / "violation_at_layer".
        # Renamed to "at_merge" / "at_violation_layer" for consistency with
        # downstream consumers and test assertions.
        tag = {
            "at_merge":           bool(trans in merges),
            "at_violation_layer": bool(ev["layer"] in viols),
        }
        new         = dict(ev)
        new["detail"] = dict(ev["detail"])
        new["detail"]["phase1"] = tag
        out_events.append(new)

    n_trans = len(axis_rotation)
    merge_rot, nomerge_rot = [], []
    viol_cc,   noviol_cc   = [], []

    for L in range(n_trans):
        r = axis_rotation[L]
        if np.isfinite(r):
            (merge_rot if L in merges else nomerge_rot).append(float(r))
        (viol_cc if (L + 1) in viols else noviol_cc).append(int(crossing_count[L]))

    def _m(xs):
        return float(np.mean(xs)) if xs else None

    agg = {
        "n_merges_in_run":               len(merges),
        "n_violations_in_run":           len(viols),
        "mean_axis_rotation_at_merge":   _m(merge_rot),
        "mean_axis_rotation_off_merge":  _m(nomerge_rot),
        "mean_crossing_at_violation":    _m(viol_cc),
        "mean_crossing_off_violation":   _m(noviol_cc),
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
    shear_absolute_floor: int = SHEAR_ABSOLUTE_FLOOR,
    drift_window_layers: int = DRIFT_WINDOW_LAYERS,
    drift_window_rad: float = DRIFT_WINDOW_RAD,
) -> dict:
    """Run Block 1 on a Block 0 result."""
    aligned  = align_hemisphere_labels(block0["assignments"], block0["valid"])
    axis_rot = compute_axis_rotation(block0["fiedler_vecs"], block0["valid"])
    cum_rot  = compute_cumulative_rotation(axis_rot)
    cc       = aligned_crossing_count(aligned["aligned_assignments"], block0["valid"])

    raw_events = detect_events(
        block0["regime"],
        aligned["match_overlap"],
        cc,
        block0["valid"],
        axis_rotation=axis_rot,
        identity_threshold=identity_threshold,
        shear_percentile=shear_percentile,
        shear_absolute_floor=shear_absolute_floor,
        drift_window_layers=drift_window_layers,
        drift_window_rad=drift_window_rad,
    )
    persistence = compute_persistence_lengths(block0["regime"], raw_events)

    xref = crossref_phase1(
        raw_events, axis_rot, cc,
        merge_transition_indices=merge_transition_indices,
        violation_layers=violation_layers,
    )

    return {
        "aligned_assignments":      aligned["aligned_assignments"],
        "flips_applied":            aligned["flips_applied"],
        "match_overlap":            aligned["match_overlap"],
        "axis_rotation":            axis_rot,
        "cumulative_axis_rotation": cum_rot,
        "crossing_count":           cc,
        "persistence_length":       persistence,
        "events":                   xref["events"],
        "crossref":                 xref["agg"],
        "thresholds": {
            "identity":             identity_threshold,
            "shear_percentile":     shear_percentile,
            "shear_absolute_floor": shear_absolute_floor,
            "drift_window_layers":  drift_window_layers,
            "drift_window_rad":     drift_window_rad,
        },
    }


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def hemisphere_tracking_to_json(result: dict) -> dict:
    """Flat representation for the aggregator's per-run JSON."""
    n_trans  = len(result["axis_rotation"])
    n_layers = n_trans + 1

    per_transition = [
        {
            "transition":               L,
            "from_layer":               L,
            "to_layer":                 L + 1,
            "match_overlap":            _f(result["match_overlap"][L]),
            "axis_rotation":            _f(result["axis_rotation"][L]),
            "cumulative_axis_rotation": _f(result["cumulative_axis_rotation"][L]),
            "crossing_count":           int(result["crossing_count"][L]),
        }
        for L in range(n_trans)
    ]

    per_layer_persistence = [int(result["persistence_length"][L]) for L in range(n_layers)]

    counts: dict[str, int] = {}
    for ev in result["events"]:
        counts[ev["type"]] = counts.get(ev["type"], 0) + 1

    ar  = np.asarray(result["axis_rotation"],            dtype=np.float64)
    mo  = np.asarray(result["match_overlap"],            dtype=np.float64)
    cc  = np.asarray(result["crossing_count"],           dtype=np.float64)
    cum = np.asarray(result["cumulative_axis_rotation"], dtype=np.float64)
    pl  = np.asarray(result["persistence_length"],       dtype=np.float64)
    pl_pos = pl[pl > 0]

    summary = {
        "n_transitions":               n_trans,
        "n_events":                    len(result["events"]),
        "event_counts":                counts,
        "mean_axis_rotation":          _mean(ar),
        "max_axis_rotation":           _max(ar),
        "total_axis_rotation":         _f(cum[-1]) if len(cum) else None,
        "mean_match_overlap":          _mean(mo),
        "min_match_overlap":           _min(mo),
        "mean_crossing":               _mean(cc),
        "max_crossing":                _max(cc),
        "identity_preserved_fraction": _frac_ge(mo, IDENTITY_THRESHOLD),
        "mean_persistence_length":     _mean(pl_pos),
        "max_persistence_length":      _max(pl_pos),
        "thresholds":                  result["thresholds"],
        "crossref":                    result["crossref"],
    }

    return {
        "per_transition":        per_transition,
        "per_layer_persistence": per_layer_persistence,
        "events":                result["events"],
        "summary":               summary,
    }


# ---------------------------------------------------------------------------
# Helpers
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