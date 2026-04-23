"""
hemisphere_membership.py — Block 2 of Phase 1h.

Per-token hemisphere trajectories and HDBSCAN nesting test.

What Block 1 hands us
---------------------
aligned_assignments[L, i] ∈ {0, 1, -1}  hemisphere label at each layer,
                                          sign-aligned across layers.
                                          -1 = invalid layer.

What Block 2 produces
---------------------
Per-token
  hemisphere_trajectory   : (n_layers,)  int — aligned hemisphere label
                            at each layer (-1 = invalid).
  stability_score         : fraction of valid consecutive transitions where
                            the token stays in the same hemisphere.
                            1.0 = never crosses.  0.0 = crosses every transition.
  border_index            : mean of |v_L(i)| / mean_L(|v_L|), averaged over
                            valid layers.  Near 0 = token hugs the boundary
                            throughout.  Large = token sits deep in one side.
  first_stable_layer      : earliest layer at which the token's hemisphere
                            matches its final-layer assignment and does not
                            change again.  None if the token never stabilizes.
  dominant_hemisphere     : 0 or 1 — whichever label the token holds for
                            the most valid layers.  -1 if tied or no valid.

HDBSCAN nesting test (optional, requires cluster labels and plateau layers)
  For each HDBSCAN cluster c at plateau layer L, compute:
    r_c = |{i in c : hemisphere_label(L, i) == 0}| / |c|
  Values near 0 or 1 mean the cluster is nested in one hemisphere.
  Values near 0.5 mean the cluster is split.

  Per (cluster, layer):
    r_c                : float in [0, 1].
    nesting_class      : "nested_A" (r_c < tolerance), "nested_B"
                         (r_c > 1 - tolerance), "mixed" (|r_c - 0.5| < 0.15),
                         "partial" (otherwise).
  Per plateau layer:
    fully_nested_fraction : fraction of clusters that are nested_A or nested_B.
    mixed_fraction        : fraction that are mixed.
    mean_r_c_distance_from_half : mean |r_c - 0.5|.  1.0 = perfect nesting,
                                  0.0 = every cluster is evenly split.

Functions
---------
compute_token_trajectories  : stability, border index, first stable layer.
compute_hdbscan_nesting     : r_c values and nesting classification per cluster.
analyze_hemisphere_membership: full Block 2 pipeline.
membership_to_json          : JSON-serializable output.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Per-token trajectory statistics
# ---------------------------------------------------------------------------

def compute_token_trajectories(
    aligned_assignments: np.ndarray,
    fiedler_vecs: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """
    Compute per-token hemisphere statistics from aligned Block 1 assignments.

    Parameters
    ----------
    aligned_assignments : (n_layers, n_tokens) int8, -1 for invalid layers.
    fiedler_vecs        : (n_layers, n_tokens) float — raw Fiedler values
                          before sign partition.  Used to compute border_index.
    valid               : (n_layers,) bool.

    Returns
    -------
    dict with:
      stability_score     (n_tokens,) float in [0, 1]
      border_index        (n_tokens,) float ≥ 0
      first_stable_layer  (n_tokens,) int or -1 (never stable)
      dominant_hemisphere (n_tokens,) int in {0, 1, -1}
    """
    n_layers, n_tokens = aligned_assignments.shape

    stability    = np.full(n_tokens, np.nan, dtype=np.float64)
    border_idx   = np.full(n_tokens, np.nan, dtype=np.float64)
    first_stable = np.full(n_tokens, -1,     dtype=np.int32)
    dominant_hemi = np.full(n_tokens, -1,    dtype=np.int8)

    valid_layers = np.where(valid)[0]
    if len(valid_layers) == 0:
        return {
            "stability_score":     stability,
            "border_index":        border_idx,
            "first_stable_layer":  first_stable,
            "dominant_hemisphere": dominant_hemi,
        }

    # --- stability score ---
    # Count valid consecutive transitions where the token keeps its label.
    stay   = np.zeros(n_tokens, dtype=np.int32)
    total  = np.zeros(n_tokens, dtype=np.int32)
    for idx in range(len(valid_layers) - 1):
        L0 = valid_layers[idx]
        L1 = valid_layers[idx + 1]
        if L1 != L0 + 1:
            # Gap in valid layers — skip this transition; no alignment info.
            continue
        a0 = aligned_assignments[L0]
        a1 = aligned_assignments[L1]
        for i in range(n_tokens):
            if a0[i] == -1 or a1[i] == -1:
                continue
            total[i] += 1
            if a0[i] == a1[i]:
                stay[i] += 1
    for i in range(n_tokens):
        if total[i] > 0:
            stability[i] = float(stay[i]) / float(total[i])

    # --- border index ---
    # mean |v_L(i)| / mean_L |v_L| for valid layers.
    # The per-layer normalization removes depth-level scale differences.
    abs_fiedler = np.abs(fiedler_vecs)   # (n_layers, n_tokens)
    layer_means = abs_fiedler[valid].mean(axis=1, keepdims=True)  # (n_valid, 1)
    layer_means = np.where(layer_means < 1e-12, 1.0, layer_means)
    normed = abs_fiedler[valid] / layer_means   # (n_valid, n_tokens)
    border_idx[:] = normed.mean(axis=0)

    # --- dominant hemisphere ---
    counts_0 = (aligned_assignments[valid] == 0).sum(axis=0)
    counts_1 = (aligned_assignments[valid] == 1).sum(axis=0)
    for i in range(n_tokens):
        if counts_0[i] > counts_1[i]:
            dominant_hemi[i] = 0
        elif counts_1[i] > counts_0[i]:
            dominant_hemi[i] = 1
        # else: tied → stays -1

    # --- first stable layer ---
    # For each token, find the earliest valid layer at which its hemisphere
    # assignment matches dominant_hemisphere and never changes again.
    # "Never changes again" = all subsequent valid consecutive transitions
    # from that layer onward keep the same label.
    # Build the suffix-stability mask: stable_from[L, i] = True iff token i
    # stays in its label from layer L through all subsequent valid layers.
    final_label = aligned_assignments[valid_layers[-1]]   # (n_tokens,)

    for i in range(n_tokens):
        dom = int(dominant_hemi[i])
        if dom == -1:
            continue
        # Walk from the last valid layer backward to find the earliest
        # layer where the token is in dom and remains there.
        found = None
        consistent = True
        for idx in reversed(range(len(valid_layers))):
            L = valid_layers[idx]
            label = int(aligned_assignments[L, i])
            if label == -1:
                consistent = False
                continue
            if label != dom:
                consistent = False
                continue
            # label == dom at this layer
            if consistent:
                found = L
            else:
                break
        if found is not None:
            first_stable[i] = found

    return {
        "stability_score":     stability,
        "border_index":        border_idx,
        "first_stable_layer":  first_stable,
        "dominant_hemisphere": dominant_hemi,
    }


# ---------------------------------------------------------------------------
# HDBSCAN nesting test
# ---------------------------------------------------------------------------

# A cluster is "nested" if its r_c is outside [tolerance, 1-tolerance].
NESTING_TOLERANCE = 0.10
MIXED_HALF_WIDTH  = 0.15   # r_c within [0.5 - ε, 0.5 + ε] → "mixed"


def compute_hdbscan_nesting(
    aligned_assignments: np.ndarray,
    hdbscan_labels: dict[int, np.ndarray],
    valid: np.ndarray,
    plateau_layers: list[int] | None = None,
    nesting_tolerance: float = NESTING_TOLERANCE,
    mixed_half_width: float  = MIXED_HALF_WIDTH,
) -> dict:
    """
    For each HDBSCAN cluster at each (plateau) layer, compute the fraction
    of its tokens that fall in hemisphere 0, and classify the cluster.

    Parameters
    ----------
    aligned_assignments : (n_layers, n_tokens) int8 from Block 1.
    hdbscan_labels      : mapping layer_index → (n_tokens,) int array of
                          HDBSCAN cluster labels (-1 = noise).  Only
                          plateau layers need to be present; missing
                          layers are silently skipped.
    valid               : (n_layers,) bool.
    plateau_layers      : list of layer indices to analyze.  If None,
                          all layers present in hdbscan_labels are used.
    nesting_tolerance   : r_c < tol or r_c > 1-tol → "nested".
    mixed_half_width    : |r_c - 0.5| < width → "mixed".

    Returns
    -------
    dict with:
      per_layer   : {layer_idx: {"clusters": [...], "summary": {...}}}
      overall     : aggregated across all analyzed layers.
    """
    if plateau_layers is None:
        plateau_layers = sorted(hdbscan_labels.keys())

    per_layer: dict[int, dict] = {}

    all_r_c: list[float] = []
    all_nested = 0
    all_mixed  = 0
    all_total  = 0

    for L in plateau_layers:
        if L >= aligned_assignments.shape[0]:
            continue
        if not valid[L]:
            continue
        if L not in hdbscan_labels:
            continue

        cluster_ids = hdbscan_labels[L]
        hemi        = aligned_assignments[L]   # (n_tokens,)

        unique_clusters = [c for c in np.unique(cluster_ids) if c >= 0]
        cluster_records: list[dict] = []
        layer_r_c: list[float] = []
        layer_nested = 0
        layer_mixed  = 0

        for c in unique_clusters:
            mask = cluster_ids == c
            members_hemi = hemi[mask]
            # Exclude invalid hemisphere labels.
            valid_members = members_hemi[members_hemi >= 0]
            if valid_members.size == 0:
                continue

            r_c = float((valid_members == 0).sum()) / float(valid_members.size)
            layer_r_c.append(r_c)

            if r_c < nesting_tolerance:
                nc = "nested_B"
            elif r_c > 1.0 - nesting_tolerance:
                nc = "nested_A"
            elif abs(r_c - 0.5) < mixed_half_width:
                nc = "mixed"
            else:
                nc = "partial"

            is_nested = nc in ("nested_A", "nested_B")
            is_mixed  = nc == "mixed"
            layer_nested += int(is_nested)
            layer_mixed  += int(is_mixed)

            cluster_records.append({
                "cluster_id":     int(c),
                "size":           int(valid_members.size),
                "r_c":            r_c,
                "nesting_class":  nc,
            })

        n_clusters = len(cluster_records)
        all_r_c.extend(layer_r_c)
        all_nested += layer_nested
        all_mixed  += layer_mixed
        all_total  += n_clusters

        per_layer[L] = {
            "clusters": cluster_records,
            "summary": {
                "n_clusters":                 n_clusters,
                "fully_nested_fraction":
                    float(layer_nested / n_clusters) if n_clusters else None,
                "mixed_fraction":
                    float(layer_mixed / n_clusters) if n_clusters else None,
                "mean_r_c_distance_from_half":
                    float(np.mean([abs(r - 0.5) for r in layer_r_c]))
                    if layer_r_c else None,
            },
        }

    overall = {
        "n_analyzed_layers": len(per_layer),
        "total_clusters":    all_total,
        "fully_nested_fraction":
            float(all_nested / all_total) if all_total else None,
        "mixed_fraction":
            float(all_mixed / all_total) if all_total else None,
        "mean_r_c_distance_from_half":
            float(np.mean([abs(r - 0.5) for r in all_r_c])) if all_r_c else None,
        "nesting_tolerance": nesting_tolerance,
        "mixed_half_width":  mixed_half_width,
    }

    return {"per_layer": per_layer, "overall": overall}


# ---------------------------------------------------------------------------
# Full Block 2 pipeline
# ---------------------------------------------------------------------------

def analyze_hemisphere_membership(
    block0: dict,
    block1: dict,
    hdbscan_labels: dict[int, np.ndarray] | None = None,
    plateau_layers: list[int] | None = None,
    token_strings: list[str] | None = None,
) -> dict:
    """
    Run Block 2 on Block 0 + Block 1 results.

    Parameters
    ----------
    block0         : output of bipartition_detect.analyze_bipartition.
    block1         : output of hemisphere_tracking.analyze_hemisphere_tracking.
    hdbscan_labels : optional.  If provided, the HDBSCAN nesting test runs.
    plateau_layers : layers to use for nesting test.  If None and
                     hdbscan_labels is provided, all keyed layers are used.
    token_strings  : optional list of token string representations for
                     the JSON output.

    Returns
    -------
    dict with:
      token_trajectories : output of compute_token_trajectories
      hdbscan_nesting    : output of compute_hdbscan_nesting, or None
    """
    traj = compute_token_trajectories(
        block1["aligned_assignments"],
        block0["fiedler_vecs"],
        block0["valid"],
    )

    nesting = None
    if hdbscan_labels is not None:
        nesting = compute_hdbscan_nesting(
            block1["aligned_assignments"],
            hdbscan_labels,
            block0["valid"],
            plateau_layers=plateau_layers,
        )

    return {
        "token_trajectories": traj,
        "hdbscan_nesting":    nesting,
        "n_tokens":           block0["n_tokens"],
        "n_layers":           block0["n_layers"],
        "token_strings":      token_strings,
    }


# ---------------------------------------------------------------------------
# JSON serializer
# ---------------------------------------------------------------------------

def membership_to_json(result: dict) -> dict:
    """
    Flat JSON-serializable form of Block 2 output.

    Per-token entries are sorted by stability_score ascending so the most
    volatile tokens appear first — they are usually the most interesting.
    """
    traj = result["token_trajectories"]
    n    = result["n_tokens"]
    n_L  = result["n_layers"]
    tok_strs = result.get("token_strings") or [None] * n

    per_token = []
    for i in range(n):
        per_token.append({
            "token_idx":          i,
            "token_str":          tok_strs[i],
            "stability_score":    _f(traj["stability_score"][i]),
            "border_index":       _f(traj["border_index"][i]),
            "first_stable_layer": int(traj["first_stable_layer"][i])
                                  if traj["first_stable_layer"][i] >= 0 else None,
            "dominant_hemisphere": int(traj["dominant_hemisphere"][i])
                                   if traj["dominant_hemisphere"][i] >= 0 else None,
            "hemisphere_trajectory": [
                int(traj["stability_score"].shape[0])   # placeholder; caller writes npz
            ],
        })

    # Sort by stability ascending.
    per_token.sort(key=lambda x: (x["stability_score"] is None, x["stability_score"] or 1.0))

    # Aggregate stats.
    ss = traj["stability_score"]
    bi = traj["border_index"]
    ss_valid = ss[np.isfinite(ss)]
    bi_valid = bi[np.isfinite(bi)]
    fsl      = traj["first_stable_layer"]
    fsl_valid = fsl[fsl >= 0]
    dom       = traj["dominant_hemisphere"]

    summary = {
        "n_tokens":                    n,
        "n_layers":                    n_L,
        "mean_stability_score":        _f(ss_valid.mean()) if ss_valid.size else None,
        "min_stability_score":         _f(ss_valid.min())  if ss_valid.size else None,
        "fraction_stability_below_0.5":
            float((ss_valid < 0.5).mean()) if ss_valid.size else None,
        "mean_border_index":           _f(bi_valid.mean()) if bi_valid.size else None,
        "mean_first_stable_layer":     _f(fsl_valid.mean()) if fsl_valid.size else None,
        "fraction_never_stable":       float((fsl == -1).mean()),
        "hemisphere_0_count":          int((dom == 0).sum()),
        "hemisphere_1_count":          int((dom == 1).sum()),
    }

    out: dict = {"per_token": per_token, "summary": summary}
    if result["hdbscan_nesting"] is not None:
        out["hdbscan_nesting"] = result["hdbscan_nesting"]
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x
