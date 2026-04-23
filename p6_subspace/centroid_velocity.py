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

P6-D5 : Inter-centroid distance in S subspace decreases monotonically as
         merge approaches; inter-centroid distance in A does not.

Functions
---------
decompose_centroid_delta  : split one centroid step into S and A components
centroid_velocity_profile : all-layers profile for one cluster
intercentroid_distances   : d_S and d_A between two cluster centroids per layer
merge_geometry            : P6-D5 test for one merge event
run_centroid_velocity     : full pipeline → SubResult
"""

import numpy as np
from scipy.stats import spearmanr

from p6_subspace.p6_io import SubResult, _fmt, _bullet, _verdict_line, SEP_THICK, SEP_THIN


# ---------------------------------------------------------------------------
# Core decomposition
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-cluster velocity profile
# ---------------------------------------------------------------------------

def centroid_velocity_profile(
    activations_per_layer: list[np.ndarray],
    labels_per_layer:      list[np.ndarray],
    layer_types:           list[str],
    layer_names:           list[str],
    P_S:                   np.ndarray,
    P_A:                   np.ndarray,
    cluster_id:            int,
) -> list[dict]:
    """
    Compute centroid displacement decomposition for one cluster across all layers.

    Parameters
    ----------
    activations_per_layer : list of (n, d) — one per layer
    labels_per_layer      : list of (n,) HDBSCAN labels — one per layer
    layer_types           : list of str — "plateau" | "merge" | "other"
    layer_names           : list of str
    P_S, P_A              : projectors
    cluster_id            : which cluster to track

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
        decomp = decompose_centroid_delta(delta, P_S, P_A)

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


# ---------------------------------------------------------------------------
# Inter-centroid distances in S and A subspaces (D.2.5)
# ---------------------------------------------------------------------------

def intercentroid_distances(
    activations_per_layer: list[np.ndarray],
    labels_per_layer:      list[np.ndarray],
    layer_names:           list[str],
    U_pos:                 np.ndarray,
    U_A:                   np.ndarray,
    c1:                    int,
    c2:                    int,
) -> list[dict]:
    """
    Track distance between centroids of clusters c1 and c2 in S and A subspaces.

    D.2.5 prediction: d_S decreases monotonically approaching merge; d_A does not.

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
        d_S     = float(np.linalg.norm(U_pos.T @ diff)) if U_pos.shape[1] > 0 else 0.0
        d_A     = float(np.linalg.norm(U_A.T @ diff))   if U_A.shape[1] > 0  else 0.0

        results.append({
            "layer_name": lname,
            "d_S":        d_S,
            "d_A":        d_A,
            "d_total":    d_total,
            "n1":         int(m1.sum()),
            "n2":         int(m2.sum()),
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


# ---------------------------------------------------------------------------
# Full pipeline → SubResult
# ---------------------------------------------------------------------------

def run_centroid_velocity(ctx: dict) -> SubResult:
    """
    Track B sub-experiment: centroid velocity decomposition.

    Required ctx keys
    -----------------
    activations_per_layer : list of (n, d) per layer/iteration
    labels_per_layer      : list of (n,) HDBSCAN labels
    layer_type_labels     : list of str
    layer_names           : list of str
    projectors            : output of subspace_build
    merge_events          : list of dicts, each with 'layer_from', 'prev_ids',
                            'curr_id' (from Phase 1 cluster_tracking output)

    Optional ctx keys
    -----------------
    tracked_cluster_ids   : list[int] — clusters to track (default: all unique)
    """
    acts         = ctx["activations_per_layer"]
    labels       = ctx["labels_per_layer"]
    layer_types  = ctx["layer_type_labels"]
    layer_names  = ctx["layer_names"]
    projectors   = ctx["projectors"]
    merge_events = ctx.get("merge_events", [])

    # Broadcast single projector entry for ALBERT
    proj_entries = projectors["per_layer"]
    # Use layer 0 projector as the reference (for ALBERT, only one exists)
    # For multi-layer models, we use per-layer projectors when indexing velocity steps
    # Here we use the first entry's P_S / P_A as the canonical projectors
    # (they represent the global model geometry)
    pe0  = proj_entries[0]
    P_S  = pe0["P_S"]
    P_A  = pe0["P_A"]
    U_pos = pe0["U_pos"]
    U_A   = pe0["U_A"]

    # Determine which clusters to track
    all_labels = np.unique(np.concatenate([l[l >= 0] for l in labels if (l >= 0).any()]))
    tracked = ctx.get("tracked_cluster_ids", all_labels.tolist())

    # 1. Per-cluster velocity profile
    all_steps: list[dict] = []
    for cid in tracked:
        steps = centroid_velocity_profile(
            acts, labels, layer_types, layer_names, P_S, P_A, int(cid)
        )
        all_steps.extend(steps)

    if not all_steps:
        return SubResult(
            name="centroid_velocity",
            applicable=False,
            payload={},
            summary_lines=["centroid_velocity: no valid cluster steps found"],
            verdict_contribution={},
        )

    # 2. Aggregate by layer type
    def _mean_r_S(steps, ltype):
        vals = [s["r_S"] for s in steps if s["layer_type_L"] == ltype]
        return float(np.mean(vals)) if vals else None

    mean_r_S_plateau = _mean_r_S(all_steps, "plateau")
    mean_r_S_merge   = _mean_r_S(all_steps, "merge")
    mean_r_S_other   = _mean_r_S(all_steps, "other")

    # P6-R3: r_S elevated at merge vs plateau
    p6_r3 = None
    if mean_r_S_merge is not None and mean_r_S_plateau is not None:
        p6_r3 = mean_r_S_merge > mean_r_S_plateau + 0.05

    # 3. Merge geometry (D.2.5) for each merge event
    merge_geom_results = []
    for event in merge_events:
        prev_ids = event.get("prev_ids", [])
        if len(prev_ids) < 2:
            continue
        c1, c2 = int(prev_ids[0]), int(prev_ids[1])
        dist_seq = intercentroid_distances(acts, labels, layer_names, U_pos, U_A, c1, c2)
        if not dist_seq:
            continue
        mg = merge_geometry_test(dist_seq)
        merge_geom_results.append({
            "c1": c1, "c2": c2,
            "n_layers_coexist": len(dist_seq),
            **mg,
        })

    n_p6d5_pass = sum(1 for r in merge_geom_results if r["p6_d5_satisfied"])

    payload = {
        "n_cluster_steps":   len(all_steps),
        "mean_r_S_plateau":  mean_r_S_plateau,
        "mean_r_S_merge":    mean_r_S_merge,
        "mean_r_S_other":    mean_r_S_other,
        "p6_r3":             p6_r3,
        "n_merge_events":    len(merge_geom_results),
        "n_p6d5_pass":       n_p6d5_pass,
        "merge_geometry":    merge_geom_results,
        # Omit per-step list from payload (too large); summary covers it
    }

    # --- Summary lines ---
    lines = [
        SEP_THICK,
        "CENTROID VELOCITY DECOMPOSITION  [Track B]",
        SEP_THICK,
        f"Cluster steps analysed:  {len(all_steps)}",
        f"Merge events analysed:   {len(merge_geom_results)}",
        "",
        "r_S = |Π_S Δx̄| / |Δx̄|  (fraction of centroid motion in real channel):",
        _bullet("mean r_S at plateau layers", mean_r_S_plateau),
        _bullet("mean r_S at merge layers",   mean_r_S_merge),
        _bullet("mean r_S at other layers",   mean_r_S_other),
        "",
        "Prediction P6-R3: r_S elevated at merge layers vs plateau layers.",
        _verdict_line(
            "P6-R3",
            p6_r3,
            f"r_S_merge={_fmt(mean_r_S_merge)} vs r_S_plateau={_fmt(mean_r_S_plateau)}"
            f" (threshold: diff > 0.05)",
        ),
        "",
        "D.2.5 — Inter-centroid distance in S vs A approaching merge events:",
        _bullet("merge events with P6-D5 pass", n_p6d5_pass),
        _bullet("total merge events tested",    len(merge_geom_results)),
    ]

    if merge_geom_results:
        lines += ["", "  Per-merge-event geometry:"]
        lines.append(f"  {'c1':>4} {'c2':>4}  {'rho_S':>8}  {'rho_A':>8}  {'P6-D5':>6}")
        for r in merge_geom_results:
            lines.append(
                f"  {r['c1']:>4d} {r['c2']:>4d}  "
                f"{_fmt(r['d_S_trend_rho']):>8}  "
                f"{_fmt(r['d_A_trend_rho']):>8}  "
                f"{'pass' if r['p6_d5_satisfied'] else 'fail':>6}"
            )

    p6_d5_satisfied = n_p6d5_pass > len(merge_geom_results) // 2 if merge_geom_results else None
    lines += [
        "",
        _verdict_line(
            "P6-D5",
            p6_d5_satisfied,
            f"{n_p6d5_pass}/{len(merge_geom_results)} merge events: "
            "d_S decreases more monotonically than d_A",
        ),
    ]

    vc = {
        "vel_mean_r_S_plateau": mean_r_S_plateau,
        "vel_mean_r_S_merge":   mean_r_S_merge,
        "vel_p6_r3_satisfied":  p6_r3,
        "vel_n_p6d5_pass":      n_p6d5_pass,
        "vel_p6_d5_satisfied":  p6_d5_satisfied,
    }

    return SubResult(
        name="centroid_velocity",
        applicable=True,
        payload=payload,
        summary_lines=lines,
        verdict_contribution=vc,
    )
