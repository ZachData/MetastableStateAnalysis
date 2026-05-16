"""
centroid_velocity.py — Track B: Centroid velocity decomposition + merge geometry.

At each layer transition L → L+1, decompose the cluster centroid displacement:

  Δx̄_C = Π_S Δx̄_C  +  Π_A Δx̄_C

and measure how much of the motion is in the real (S) vs imaginary (A) channel.

Also tracks inter-centroid distance in S and A subspaces separately as a
function of layer (D.2.5 merge geometry): the merge should be driven by
real-subspace convergence, not imaginary rotation.

Falsifiable predictions tested
-------------------------------
P6-R3 : At plateau layers, |Π_S Δx̄_C| is small (centroid settled);
         at merge layers, |Π_S Δx̄_C| spikes in the merge direction.
         The ratio r_S(L) = |Π_S Δx̄_C| / |Δx̄_C| is elevated at merge layers.
         Requires both relative gap (diff > 0.05) AND absolute level (> 0.20).

P6-D5 : Inter-centroid distance in S subspace decreases monotonically as
         merge approaches; inter-centroid distance in A does not.

Functions
---------
decompose_centroid_delta  : split one centroid step into S and A components
centroid_velocity_profile : all-layers profile for one cluster
intercentroid_distances   : d_S and d_A between two cluster centroids per layer
merge_geometry_test       : P6-D5 test for one merge event
run_centroid_velocity     : full pipeline → SubResult

Fixes:
  1 : Per-layer projectors (not layer-0 only)
  2 : d_S measured in full U_S (not U_pos-only)
  3 : P6-R3 requires both relative gap AND absolute level
  4 : Named warning when merge_events absent
"""

import numpy as np
from scipy.stats import spearmanr


# -----------
# Core decomposition
# -----------

def decompose_centroid_delta(
    delta:  np.ndarray,
    P_S:    np.ndarray,
    P_A:    np.ndarray,
) -> dict:
    """
    Decompose centroid displacement vector into S and A components.

    Parameters
    ----------
    delta : (d,) — centroid displacement Δx̄_C
    P_S   : (d, d) — real-channel projector
    P_A   : (d, d) — imaginary-channel projector

    Returns
    -------
    dict with:
      delta_S    : (d,) component in S
      delta_A    : (d,) component in A
      norm_S     : float — ||Π_S Δ||
      norm_A     : float — ||Π_A Δ||
      norm_total : float — ||Δ||
      r_S        : float — norm_S / norm_total (fraction of motion in real channel)
    """
    delta_S    = P_S @ delta
    delta_A    = P_A @ delta
    norm_S     = float(np.linalg.norm(delta_S))
    norm_A     = float(np.linalg.norm(delta_A))
    norm_total = float(np.linalg.norm(delta))
    r_S        = norm_S / max(norm_total, 1e-12)

    return {
        "delta_S":    delta_S,
        "delta_A":    delta_A,
        "norm_S":     norm_S,
        "norm_A":     norm_A,
        "norm_total": norm_total,
        "r_S":        r_S,
    }


# -----------
# Per-cluster velocity profile
# -----------

def centroid_velocity_profile(
    activations_per_layer: list[np.ndarray],
    labels_per_layer:      list[np.ndarray],
    layer_types:           list[str],
    layer_names:           list[str],
    cluster_id:            int,
    P_S:                   np.ndarray | None = None,
    P_A:                   np.ndarray | None = None,
    proj_per_layer:        list[dict] | None = None,
) -> list[dict]:
    """
    Compute centroid displacement decomposition for one cluster across all layers.

    Parameters
    ----------
    activations_per_layer : list of (n, d) — one per layer/iteration
    labels_per_layer      : list of (n,) HDBSCAN labels — one per layer
    layer_types           : list of str — "plateau" | "merge" | "other"
    layer_names           : list of str
    cluster_id            : which cluster to track
    P_S, P_A              : default projectors (fallback if proj_per_layer not provided)
    proj_per_layer        : optional list of dicts with per-layer projectors (FIX 1)

    Returns
    -------
    list of dicts, one per layer transition (len = n_layers - 1)
    """
    results = []
    n_layers = len(activations_per_layer)

    for L in range(n_layers - 1):
        X_curr  = activations_per_layer[L]
        X_next  = activations_per_layer[L + 1]
        lab_cur = labels_per_layer[L]
        lab_nxt = labels_per_layer[L + 1]

        # Centroid at layer L (tokens in cluster at L)
        mask_cur = lab_cur == cluster_id
        mask_nxt = lab_nxt == cluster_id

        if mask_cur.sum() < 2:
            continue

        centroid_cur = X_curr[mask_cur].mean(axis=0)

        if mask_nxt.sum() < 2:
            # Cluster dissolved (merged) — use all tokens from old members
            member_indices = np.where(mask_cur)[0]
            centroid_nxt   = X_next[member_indices].mean(axis=0)
        else:
            centroid_nxt = X_next[mask_nxt].mean(axis=0)

        delta = centroid_nxt - centroid_cur

        # FIX 1: Select per-layer projectors
        if proj_per_layer is not None:
            p_S = proj_per_layer[L]["P_S"]
            p_A = proj_per_layer[L]["P_A"]
        else:
            p_S, p_A = P_S, P_A

        decomp = decompose_centroid_delta(delta, p_S, p_A)

        results.append({
            "layer_from":   layer_names[L],
            "layer_to":     layer_names[L + 1],
            "layer_type_L": layer_types[L],
            "n_tokens_L":   int(mask_cur.sum()),
            "n_tokens_Lp1": int(mask_nxt.sum()),
            "norm_S":       decomp["norm_S"],
            "norm_A":       decomp["norm_A"],
            "norm_total":   decomp["norm_total"],
            "r_S":          decomp["r_S"],
            "cluster_id":   cluster_id,
        })

    return results


# -----------
# Inter-centroid distances in S and A subspaces (D.2.5)
# -----------

def intercentroid_distances(
    activations_per_layer: list[np.ndarray],
    labels_per_layer:      list[np.ndarray],
    layer_names:           list[str],
    c1:                    int,
    c2:                    int,
    U_pos:                 np.ndarray | None = None,
    U_A:                   np.ndarray | None = None,
    proj_per_layer:        list[dict] | None = None,
) -> list[dict]:
    """
    Track distance between centroids of clusters c1 and c2 in S and A subspaces.

    D.2.5 prediction: d_S decreases monotonically approaching merge; d_A does not.

    FIX 1: Use proj_per_layer if available.
    FIX 2: Use full U_S (not just U_pos) for d_S measurement.

    Returns
    -------
    list of dicts with layer_name, d_S, d_A, d_total for each layer where
    both clusters exist.
    """
    results = []

    for L, (X, labels, lname) in enumerate(
        zip(activations_per_layer, labels_per_layer, layer_names)
    ):
        m1 = labels == c1
        m2 = labels == c2
        if m1.sum() < 2 or m2.sum() < 2:
            continue

        mu1 = X[m1].mean(axis=0)
        mu2 = X[m2].mean(axis=0)
        diff = mu1 - mu2

        d_total = float(np.linalg.norm(diff))

        if proj_per_layer is not None:
            pe = proj_per_layer[L]
            # FIX 2: Use full U_S (not just U_pos)
            u_S = pe["U_S"]
            d_S = float(np.linalg.norm(u_S.T @ diff)) if u_S.shape[1] > 0 else 0.0
            u_A = pe["U_A"]
            d_A = float(np.linalg.norm(u_A.T @ diff)) if u_A.shape[1] > 0 else 0.0
        else:
            d_S = float(np.linalg.norm(U_pos.T @ diff)) if U_pos is not None else 0.0
            d_A = float(np.linalg.norm(U_A.T @ diff)) if U_A is not None else 0.0

        results.append({
            "layer_name": lname,
            "d_S": d_S,
            "d_A": d_A,
            "d_total": d_total,
            "n1": int(m1.sum()),
            "n2": int(m2.sum()),
        })

    return results


def merge_geometry_test(
    dist_sequence: list[dict],
    window:        int = 3,
) -> dict:
    """
    P6-D5: test whether d_S decreases monotonically near the merge while d_A does not.

    Uses the last `window` layers before the cluster pair disappears.

    Returns
    -------
    dict with:
      d_S_trend_rho  : Spearman correlation of d_S with layer index (negative = decreasing)
      d_A_trend_rho  : Spearman correlation of d_A with layer index
      p6_d5_satisfied: bool — d_S trend more negative than d_A trend
    """
    if len(dist_sequence) < 2:
        return {"d_S_trend_rho": None, "d_A_trend_rho": None, "p6_d5_satisfied": False}

    tail = dist_sequence[-window:]
    idx  = np.arange(len(tail))
    d_S  = np.array([r["d_S"]    for r in tail])
    d_A  = np.array([r["d_A"]    for r in tail])

    rho_S, _ = spearmanr(idx, d_S)
    rho_A, _ = spearmanr(idx, d_A)

    rho_S = float(rho_S) if np.isfinite(rho_S) else 0.0
    rho_A = float(rho_A) if np.isfinite(rho_A) else 0.0

    # d_S should be decreasing (negative rho) AND more so than d_A
    p6_d5 = (rho_S < 0) and (rho_S < rho_A)

    return {
        "d_S_trend_rho":   rho_S,
        "d_A_trend_rho":   rho_A,
        "p6_d5_satisfied": p6_d5,
    }


# -----------
# Full pipeline → SubResult
# -----------

def run_centroid_velocity(ctx: dict):
    """
    Track B sub-experiment: centroid velocity decomposition.

    FIX 1, 2, 3, 4: Per-layer projectors, full U_S distance, absolute threshold, warnings.

    Required ctx keys
    -----------------
    activations_per_layer : list of (n, d) per layer/iteration
    labels_per_layer      : list of (n,) HDBSCAN labels
    layer_type_labels     : list of str
    layer_names           : list of str
    projectors            : output of subspace_build

    Optional ctx keys
    -----------------
    tracked_cluster_ids   : list[int] — clusters to track (default: all unique)
    merge_events          : list of dicts with cluster merge info
    """
    acts         = ctx["activations_per_layer"]
    labels       = ctx["labels_per_layer"]
    layer_types  = ctx["layer_type_labels"]
    layer_names  = ctx["layer_names"]
    projectors   = ctx.get("projectors")
    merge_events = ctx.get("merge_events", [])

    # FIX 1: Build per-layer projector list
    proj_per_layer = None
    if projectors is not None:
        proj_entries = projectors["per_layer"]
        if len(proj_entries) == 1 and len(acts) > 1:
            proj_entries = proj_entries * len(acts)
        proj_per_layer = proj_entries

    # FIX 4: Explicit warning for missing merge_events
    merge_events_key_present = "merge_events" in ctx
    merge_events_empty = not merge_events

    all_labels = np.unique(np.concatenate([l[l >= 0] for l in labels if (l >= 0).any()]))
    tracked = ctx.get("tracked_cluster_ids", all_labels.tolist())

    # 1. Per-cluster velocity profile
    all_steps: list[dict] = []
    for cid in tracked:
        steps = centroid_velocity_profile(
            acts, labels, layer_types, layer_names, int(cid),
            proj_per_layer=proj_per_layer,
        )
        all_steps.extend(steps)

    if not all_steps:
        return {
            "name": "centroid_velocity",
            "applicable": False,
            "payload": {},
            "verdict_contribution": {},
        }

    # 2. Aggregate by layer type
    def _mean_r_S(steps, ltype):
        vals = [s["r_S"] for s in steps if s["layer_type_L"] == ltype]
        return float(np.mean(vals)) if vals else None

    mean_r_S_plateau = _mean_r_S(all_steps, "plateau")
    mean_r_S_merge = _mean_r_S(all_steps, "merge")
    mean_r_S_other = _mean_r_S(all_steps, "other")

    # FIX 3: Require both relative gap AND absolute level
    P6_R3_ABS_THRESHOLD = 0.20
    p6_r3 = None
    if mean_r_S_merge is not None and mean_r_S_plateau is not None:
        relative_elevated = mean_r_S_merge > mean_r_S_plateau + 0.05
        absolute_elevated = mean_r_S_merge > P6_R3_ABS_THRESHOLD
        p6_r3 = relative_elevated and absolute_elevated

    # 3. Merge geometry (D.2.5) for each merge event
    merge_geom_results = []
    for event in merge_events:
        prev_ids = event.get("prev_ids", [])
        if len(prev_ids) < 2:
            continue
        c1, c2 = int(prev_ids[0]), int(prev_ids[1])
        dist_seq = intercentroid_distances(
            acts, labels, layer_names, c1, c2, proj_per_layer=proj_per_layer
        )
        if not dist_seq:
            continue
        mg = merge_geometry_test(dist_seq)
        merge_geom_results.append({
            "c1": c1, "c2": c2,
            "n_layers_coexist": len(dist_seq),
            **mg,
        })

    n_p6d5_pass = sum(1 for r in merge_geom_results if r["p6_d5_satisfied"])
    p6_d5_satisfied = (
        n_p6d5_pass > len(merge_geom_results) // 2
        if merge_geom_results else None
    )

    payload = {
        "n_cluster_steps":   len(all_steps),
        "mean_r_S_plateau":  mean_r_S_plateau,
        "mean_r_S_merge":    mean_r_S_merge,
        "mean_r_S_other":    mean_r_S_other,
        "p6_r3":             p6_r3,
        "n_merge_events":    len(merge_geom_results),
        "n_p6d5_pass":       n_p6d5_pass,
        "merge_geometry":    merge_geom_results,
        "merge_events_warning_absent": not merge_events_key_present,
        "merge_events_warning_empty": merge_events_empty,
    }

    vc = {
        "vel_mean_r_S_plateau": mean_r_S_plateau,
        "vel_mean_r_S_merge":   mean_r_S_merge,
        "vel_p6_r3_satisfied":  p6_r3,
        "vel_n_p6d5_pass":      n_p6d5_pass,
        "vel_p6_d5_satisfied":  p6_d5_satisfied,
    }

    return {
        "name": "centroid_velocity",
        "applicable": True,
        "payload": payload,
        "verdict_contribution": vc,
    }