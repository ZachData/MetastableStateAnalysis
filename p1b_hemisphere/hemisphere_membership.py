"""
hemisphere_membership.py — Block 2 of Phase 1h.

Per-token hemisphere trajectories and HDBSCAN nesting test.

What Block 1 hands us
---------------------
aligned_assignments[L, i] ∈ {0, 1, -1}  hemisphere label per layer,
                                          sign-aligned across layers.
                                          -1 = invalid layer.

What Block 2 produces
---------------------
Per-token
  hemisphere_trajectory   : (n_layers,)  int  — aligned hemisphere label
                            at each layer (-1 = invalid).
  stability_score         : fraction of consecutive valid transitions where
                            the token stays in the same hemisphere.
  border_index            : mean of |v_L(i)| / mean_L(|v_L|), averaged
                            over valid layers.  Large = deep in one side.
  first_stable_layer      : earliest layer at which the token's hemisphere
                            matches its final-layer assignment and does not
                            change again.  -1 if never.
  dominant_hemisphere     : 0 or 1 — label held for the most valid layers.
                            -1 if tied or no valid layers.

HDBSCAN nesting test
  r_c = fraction of cluster c's tokens that are in hemisphere 0.
  Per (cluster, layer):
    r_c             : float in [0, 1].
    nesting_class   : "nested_B"  r_c < tolerance  (cluster sits in hemi 1)
                      "nested_A"  r_c > 1-tolerance (cluster sits in hemi 0)
                      "mixed"     |r_c − 0.5| < mixed_half_width
                      "partial"   otherwise

Functions
---------
compute_token_trajectories   : stability, border index, first stable layer.
compute_hdbscan_nesting      : r_c values and nesting classification per cluster.
analyze_hemisphere_membership: full Block 2 pipeline.
membership_to_json           : JSON-serializable output.
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
    fiedler_vecs        : (n_layers, n_tokens) float — raw Fiedler values.
    valid               : (n_layers,) bool.

    Returns
    -------
    dict with:
      stability_score     (n_tokens,) float in [0, 1]
      border_index        (n_tokens,) float ≥ 0
      first_stable_layer  (n_tokens,) int, -1 if never stable
      dominant_hemisphere (n_tokens,) int in {0, 1, -1}
    """
    n_layers, n_tokens = aligned_assignments.shape

    stability     = np.full(n_tokens, np.nan, dtype=np.float64)
    border_idx    = np.full(n_tokens, np.nan, dtype=np.float64)
    first_stable  = np.full(n_tokens, -1,     dtype=np.int32)
    dominant_hemi = np.full(n_tokens, -1,     dtype=np.int8)

    valid_layers = np.where(valid)[0]
    if len(valid_layers) == 0:
        return {
            "stability_score":     stability,
            "border_index":        border_idx,
            "first_stable_layer":  first_stable,
            "dominant_hemisphere": dominant_hemi,
        }

    # Stability score: fraction of consecutive valid transitions where token
    # stays in the same hemisphere.
    stay  = np.zeros(n_tokens, dtype=np.int32)
    total = np.zeros(n_tokens, dtype=np.int32)
    for idx in range(len(valid_layers) - 1):
        L0 = valid_layers[idx]
        L1 = valid_layers[idx + 1]
        if L1 != L0 + 1:
            continue  # gap — skip
        a0 = aligned_assignments[L0]
        a1 = aligned_assignments[L1]
        counted = (a0 >= 0) & (a1 >= 0)
        total += counted.astype(np.int32)
        stay  += ((a0 == a1) & counted).astype(np.int32)

    with np.errstate(invalid="ignore"):
        stability = np.where(total > 0, stay / total, np.nan).astype(np.float64)

    # Border index: mean |v(i)| / mean_layer(|v|) over valid layers.
    scale_sums  = np.zeros(n_tokens, dtype=np.float64)
    scale_count = np.zeros(n_tokens, dtype=np.int32)
    for L in valid_layers:
        fv        = fiedler_vecs[L]
        layer_std = float(np.mean(np.abs(fv)))
        if layer_std < 1e-12:
            continue
        scale_sums  += np.abs(fv) / layer_std
        scale_count += 1

    with np.errstate(invalid="ignore"):
        border_idx = np.where(
            scale_count > 0, scale_sums / scale_count, np.nan
        ).astype(np.float64)

    # Dominant hemisphere and first stable layer.
    count0 = np.zeros(n_tokens, dtype=np.int32)
    count1 = np.zeros(n_tokens, dtype=np.int32)
    for L in valid_layers:
        a = aligned_assignments[L]
        count0 += (a == 0).astype(np.int32)
        count1 += (a == 1).astype(np.int32)

    dominant_hemi = np.where(
        count0 > count1, np.int8(0),
        np.where(count1 > count0, np.int8(1), np.int8(-1))
    ).astype(np.int8)

    for i in range(n_tokens):
        dom = int(dominant_hemi[i])
        if dom < 0:
            continue
        for L in reversed(valid_layers.tolist()):
            if aligned_assignments[L, i] != dom:
                break
            first_stable[i] = L

    return {
        "stability_score":     stability,
        "border_index":        border_idx,
        "first_stable_layer":  first_stable,
        "dominant_hemisphere": dominant_hemi,
    }


# ---------------------------------------------------------------------------
# HDBSCAN nesting test
# ---------------------------------------------------------------------------

def compute_hdbscan_nesting(
    aligned_assignments: np.ndarray,
    hdbscan_labels: dict[int, np.ndarray],
    valid: np.ndarray,
    plateau_layers: list[int] | None = None,
    nesting_tolerance: float = 0.15,
    mixed_half_width:  float = 0.15,
) -> dict:
    """
    Compute r_c and nesting classification for each HDBSCAN cluster.

    r_c = fraction of cluster tokens in hemisphere 0.

    Nesting classification (FIX: docstring was previously inverted):
      "nested_B"  r_c < nesting_tolerance         cluster sits in hemisphere 1
      "nested_A"  r_c > 1 - nesting_tolerance      cluster sits in hemisphere 0
      "mixed"     |r_c - 0.5| < mixed_half_width   cluster splits across hemispheres
      "partial"   otherwise

    Parameters
    ----------
    aligned_assignments : (n_layers, n_tokens) int8.
    hdbscan_labels      : {layer_idx: (n_tokens,) int32} — HDBSCAN cluster
                          labels.  -1 = noise.
    valid               : (n_layers,) bool.
    plateau_layers      : if provided, only these layers are analyzed.
    """
    layers_to_use = plateau_layers if plateau_layers is not None \
                    else [L for L in hdbscan_labels if valid[L]]

    per_layer: dict[int, dict] = {}
    all_r_c:  list[float] = []
    all_nested = all_mixed = all_total = 0

    for L in layers_to_use:
        if not valid[L] or L not in hdbscan_labels:
            continue
        labels    = hdbscan_labels[L]
        hemi_row  = aligned_assignments[L]
        clusters  = [c for c in np.unique(labels) if c >= 0]

        cluster_records: list[dict] = []
        layer_r_c:    list[float] = []
        layer_nested = layer_mixed = 0

        for c in clusters:
            members_hemi = hemi_row[labels == c]
            valid_members = members_hemi[members_hemi >= 0]
            if valid_members.size == 0:
                continue

            r_c = float((valid_members == 0).sum()) / float(valid_members.size)
            layer_r_c.append(r_c)

            if r_c < nesting_tolerance:
                nc = "nested_B"      # almost all in hemisphere 1
            elif r_c > 1.0 - nesting_tolerance:
                nc = "nested_A"      # almost all in hemisphere 0
            elif abs(r_c - 0.5) < mixed_half_width:
                nc = "mixed"
            else:
                nc = "partial"

            layer_nested += int(nc in ("nested_A", "nested_B"))
            layer_mixed  += int(nc == "mixed")
            cluster_records.append({
                "cluster_id":    int(c),
                "size":          int(valid_members.size),
                "r_c":           r_c,
                "nesting_class": nc,
            })

        n_clusters = len(cluster_records)
        all_r_c.extend(layer_r_c)
        all_nested += layer_nested
        all_mixed  += layer_mixed
        all_total  += n_clusters

        per_layer[L] = {
            "clusters": cluster_records,
            "summary": {
                "n_clusters":
                    n_clusters,
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
        "n_analyzed_layers":          len(per_layer),
        "total_clusters":             all_total,
        "fully_nested_fraction":
            float(all_nested / all_total) if all_total else None,
        "mixed_fraction":
            float(all_mixed / all_total) if all_total else None,
        "mean_r_c_distance_from_half":
            float(np.mean([abs(r - 0.5) for r in all_r_c])) if all_r_c else None,
        "nesting_tolerance":          nesting_tolerance,
        "mixed_half_width":           mixed_half_width,
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
    """Run Block 2 on Block 0 + Block 1 results."""
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
        # Keep a reference to aligned_assignments for trajectory serialisation.
        "_aligned_assignments": block1["aligned_assignments"],
    }


# ---------------------------------------------------------------------------
# JSON serializer
# ---------------------------------------------------------------------------

def membership_to_json(result: dict) -> dict:
    """
    Flat JSON-serializable form of Block 2 output.

    Per-token entries are sorted by stability_score ascending so the most
    volatile tokens appear first.

    hemisphere_trajectory is the full sequence of per-layer aligned
    hemisphere labels (-1 for invalid layers), serialized as a list of ints.
    """
    traj     = result["token_trajectories"]
    n        = result["n_tokens"]
    n_L      = result["n_layers"]
    tok_strs = result.get("token_strings") or [None] * n

    # FIX: was `[int(traj["stability_score"].shape[0])]` — a single-element
    # placeholder that conveyed no trajectory information.  Now we serialize
    # the actual per-layer assignment sequence for each token.
    aligned = result.get("_aligned_assignments")   # (n_layers, n_tokens) int8

    per_token = []
    for i in range(n):
        if aligned is not None:
            trajectory = [int(aligned[L, i]) for L in range(n_L)]
        else:
            trajectory = None

        per_token.append({
            "token_idx":           i,
            "token_str":           tok_strs[i],
            "stability_score":     _f(traj["stability_score"][i]),
            "border_index":        _f(traj["border_index"][i]),
            "first_stable_layer":  int(traj["first_stable_layer"][i])
                                   if traj["first_stable_layer"][i] >= 0 else None,
            "dominant_hemisphere": int(traj["dominant_hemisphere"][i])
                                   if traj["dominant_hemisphere"][i] >= 0 else None,
            "hemisphere_trajectory": trajectory,
        })

    per_token.sort(
        key=lambda x: (x["stability_score"] is None, x["stability_score"] or 1.0)
    )

    ss       = traj["stability_score"]
    bi       = traj["border_index"]
    ss_valid = ss[np.isfinite(ss)]
    bi_valid = bi[np.isfinite(bi)]
    fsl      = traj["first_stable_layer"]
    fsl_valid = fsl[fsl >= 0]
    dom      = traj["dominant_hemisphere"]

    summary = {
        "n_tokens":                     n,
        "n_layers":                     n_L,
        "mean_stability_score":         _f(ss_valid.mean()) if ss_valid.size else None,
        "min_stability_score":          _f(ss_valid.min())  if ss_valid.size else None,
        "fraction_stability_below_0.5":
            float((ss_valid < 0.5).mean()) if ss_valid.size else None,
        "mean_border_index":            _f(bi_valid.mean()) if bi_valid.size else None,
        "mean_first_stable_layer":      _f(fsl_valid.mean()) if fsl_valid.size else None,
        "fraction_never_stable":        float((fsl == -1).mean()),
        "hemisphere_0_count":           int((dom == 0).sum()),
        "hemisphere_1_count":           int((dom == 1).sum()),
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