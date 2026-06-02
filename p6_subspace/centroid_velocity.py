"""
centroid_velocity.py — Track B: Centroid velocity decomposition + merge geometry.

At each layer transition L → L+1, decompose the cluster centroid displacement:

  Δx̄_C = Π_S Δx̄_C  +  Π_A Δx̄_C

and measure how much of the motion is in the real (S) vs imaginary (A) channel.

Also tracks inter-centroid distance in S and A subspaces separately as a
function of layer (P6-D5 merge geometry): the merge should be driven by
real-subspace convergence, not imaginary rotation.

Falsifiable predictions tested
-------------------------------
P6-R3 : At plateau layers, |Π_S Δx̄_C| is small (centroid settled);
         at merge layers, |Π_S Δx̄_C| spikes in the merge direction.
         The ratio r_S(L) = |Π_S Δx̄_C| / |Δx̄_C| is elevated at merge layers.

P6-D5 : Inter-centroid distance in S subspace decreases monotonically as
         merge approaches; inter-centroid distance in A does not.

Bug fixes applied in this version
-----------------------------------
1. Per-layer projectors:  centroid_velocity_profile and intercentroid_distances
   previously used a single layer-0 projector for all layers. This was correct
   only for ALBERT. Now both functions accept proj_per_layer (list of dicts, one
   per layer) and use the layer-appropriate P_S / P_A / U_S at each step.
   Fixed args P_S / P_A / U_pos are kept as fallback for backward compatibility.

2. U_S for distance metric:  intercentroid_distances previously measured d_S as
   ‖U_pos^T (μ_1−μ_2)‖, projecting only onto the *attractive* half of S.
   P6-R2 predicts cluster separation lives in U_neg (repulsive), so using U_pos
   would produce a near-zero, noisy signal. Distance is now measured in the full
   real subspace U_S = span(U_pos ∪ U_neg). U_pos and U_neg components are also
   reported separately for diagnostics.

3. P6-R3 verdict:  the previous check (mean_r_S_merge > mean_r_S_plateau + 0.05)
   allowed a relative pass even when both values were near zero (e.g., 0.07 vs
   0.02), which is not scientifically meaningful. Now requires *both* conditions:
     a) relative: mean_r_S_merge > mean_r_S_plateau + 0.05
     b) absolute: mean_r_S_merge > 0.20

4. Empty merge_events warning:  if ctx contains no merge_events (key absent or
   empty list), P6-D5 cannot be evaluated. Now emitted as a named warning in the
   summary rather than silently producing n_merge_events = 0.

Functions
---------
decompose_centroid_delta  : split one centroid step into S and A components
centroid_velocity_profile : all-layers profile for one cluster
intercentroid_distances   : d_S and d_A between two cluster centroids per layer
merge_geometry_test       : P6-D5 test for one merge event
run_centroid_velocity     : full pipeline → SubResult
"""

import numpy as np
from scipy.stats import spearmanr

from p6_subspace.p6_io import SubResult, _fmt, _bullet, _verdict_line, SEP_THICK, SEP_THIN


# ---------------------------------------------------------------------------
# Core decomposition
# ---------------------------------------------------------------------------

def decompose_centroid_delta(
    delta: np.ndarray,
    P_S:   np.ndarray,
    P_A:   np.ndarray,
    ) -> dict:
    """
    Decompose centroid displacement vector into S and A components.

    Returns
    -------
    dict with norm_S, norm_A, norm_total, r_S (fraction in S channel).
    """
    v_S   = P_S @ delta
    v_A   = P_A @ delta
    n_S   = float(np.linalg.norm(v_S))
    n_A   = float(np.linalg.norm(v_A))
    n_tot = float(np.linalg.norm(delta))

    r_S = n_S / max(n_tot, 1e-12)

    return {
        "delta_S":    v_S,
        "delta_A":    v_A,
        "norm_S":     n_S,
        "norm_A":     n_A,
        "norm_total": n_tot,
        "r_S":        r_S,
    }


# ---------------------------------------------------------------------------
# Per-cluster, per-layer velocity profile
# ---------------------------------------------------------------------------

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
    Compute centroid velocity decomposition for one cluster at each layer
    transition L → L+1.

    Parameters
    ----------
    P_S, P_A        : fixed projectors used when proj_per_layer is None
                      (backward-compatible fallback; incorrect for per-layer models)
    proj_per_layer  : list of projector dicts, one per layer, each with keys
                      "P_S" and "P_A".  When provided, the layer-L projector is
                      used for the L→L+1 transition.  Broadcast a single-entry
                      list for ALBERT before calling.

    Returns
    -------
    list of dicts, one per tracked layer transition.
    """
    results  = []
    n_layers = len(activations_per_layer)

    for L in range(n_layers - 1):
        X_curr  = activations_per_layer[L]
        X_next  = activations_per_layer[L + 1]
        lab_cur = labels_per_layer[L]
        lab_nxt = labels_per_layer[L + 1]

        mask_cur = lab_cur == cluster_id
        mask_nxt = lab_nxt == cluster_id

        if mask_cur.sum() < 2:
            continue

        centroid_cur = X_curr[mask_cur].mean(axis=0)

        if mask_nxt.sum() < 2:
            # Cluster dissolved — track same token positions into next layer
            centroid_nxt = X_next[np.where(mask_cur)[0]].mean(axis=0)
        else:
            centroid_nxt = X_next[mask_nxt].mean(axis=0)

        delta = centroid_nxt - centroid_cur

        # Select per-layer projectors or fall back to fixed
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


# ---------------------------------------------------------------------------
# Inter-centroid distances in S and A subspaces (P6-D5)
# ---------------------------------------------------------------------------

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

    P6-D5 prediction: d_S decreases monotonically approaching merge; d_A does not.

    Distance in S uses the full real basis U_S = span(U_pos ∪ U_neg) so that
    repulsive-channel separation (predicted by P6-R2) is captured. U_pos and
    U_neg components are also reported separately for diagnostics.

    Parameters
    ----------
    U_pos, U_A      : fixed bases used when proj_per_layer is None (fallback).
                      NOTE: in the fallback path, only U_pos is available; d_S
                      will reflect the attractive subspace only — label clearly.
    proj_per_layer  : list of projector dicts per layer, each with keys
                      "U_S" (full real basis), "U_pos", "U_neg", "U_A".

    Returns
    -------
    list of dicts with layer_name, d_S, d_S_pos, d_S_neg, d_A, d_total
    for each layer where both clusters exist.
    """
    results = []

    for L, (X, labels, lname) in enumerate(
        zip(activations_per_layer, labels_per_layer, layer_names)
    ):
        m1 = labels == c1
        m2 = labels == c2
        if m1.sum() < 2 or m2.sum() < 2:
            continue

        mu1  = X[m1].mean(axis=0)
        mu2  = X[m2].mean(axis=0)
        diff = mu1 - mu2

        d_total = float(np.linalg.norm(diff))

        if proj_per_layer is not None:
            pe    = proj_per_layer[L]
            u_S   = pe["U_S"]    # full real basis (U_pos ∪ U_neg)
            u_pos = pe["U_pos"]
            u_neg = pe.get("U_neg", np.zeros((u_S.shape[0], 0)))
            u_A   = pe["U_A"]
            # d_S measured in full real subspace (fixes U_pos-only bug)
            d_S     = float(np.linalg.norm(u_S.T   @ diff))
            d_S_pos = float(np.linalg.norm(u_pos.T @ diff))
            d_S_neg = float(np.linalg.norm(u_neg.T @ diff)) if u_neg.shape[1] > 0 else 0.0
            d_A     = float(np.linalg.norm(u_A.T   @ diff))
        else:
            # Fallback: U_pos only (legacy; cannot capture repulsive separation)
            d_S     = float(np.linalg.norm(U_pos.T @ diff)) if U_pos is not None else 0.0
            d_S_pos = d_S
            d_S_neg = float("nan")
            d_A     = float(np.linalg.norm(U_A.T   @ diff)) if U_A   is not None else 0.0

        results.append({
            "layer_name": lname,
            "d_S":        d_S,
            "d_S_pos":    d_S_pos,
            "d_S_neg":    d_S_neg,
            "d_A":        d_A,
            "d_total":    d_total,
            "n1":         int(m1.sum()),
            "n2":         int(m2.sum()),
        })

    return results


# ---------------------------------------------------------------------------
# Merge geometry test (P6-D5)
# ---------------------------------------------------------------------------

def merge_geometry_test(
    dist_sequence: list[dict],
    window:        int = 3,
) -> dict:
    """
    P6-D5: test whether d_S decreases monotonically near merge while d_A does not.

    Uses the last `window` layers before the cluster pair disappears.

    Returns
    -------
    dict with d_S_trend_rho, d_A_trend_rho, p6_d5_satisfied.
    """
    if len(dist_sequence) < 2:
        return {"d_S_trend_rho": None, "d_A_trend_rho": None, "p6_d5_satisfied": False}

    tail = dist_sequence[-window:]
    idx  = np.arange(len(tail))
    d_S  = np.array([r["d_S"] for r in tail])
    d_A  = np.array([r["d_A"] for r in tail])

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
    activations_per_layer : list of (n, d)
    labels_per_layer      : list of (n,) HDBSCAN cluster labels
    layer_type_labels     : list of str  ("plateau" | "merge" | "other")
    layer_names           : list of str
    projectors            : output of subspace_build (strongly recommended)

    Optional ctx keys
    -----------------
    merge_events         : list of merge event dicts from Phase 1
                           Each must have "prev_ids": [c1, c2].
                           Required for P6-D5; absence emits a named warning.
    tracked_cluster_ids  : list[int] (default: all unique non-noise labels)
    """
    acts        = ctx["activations_per_layer"]
    labels      = ctx["labels_per_layer"]
    layer_types = ctx["layer_type_labels"]
    layer_names = ctx["layer_names"]

    # -----------------------------------------------------------------------
    # Build per-layer projector list (Fix 1: per-layer, not layer-0 only)
    # -----------------------------------------------------------------------
    projectors     = ctx.get("projectors")
    proj_per_layer = None

    if projectors is not None:
        proj_entries = projectors["per_layer"]
        # Broadcast single entry for ALBERT; use per-layer otherwise
        if len(proj_entries) == 1 and len(acts) > 1:
            proj_entries = proj_entries * len(acts)
        proj_per_layer = proj_entries

    # -----------------------------------------------------------------------
    # Fallback fixed projectors (used only when projectors absent from ctx)
    # -----------------------------------------------------------------------
    P_S_fixed   = None
    P_A_fixed   = None
    U_pos_fixed = None
    U_A_fixed   = None

    if proj_per_layer is None and projectors is not None:
        pe0 = projectors["per_layer"][0]
        P_S_fixed   = pe0["P_S"]
        P_A_fixed   = pe0["P_A"]
        U_pos_fixed = pe0["U_pos"]
        U_A_fixed   = pe0["U_A"]

    # -----------------------------------------------------------------------
    # Merge events (P6-D5)
    # -----------------------------------------------------------------------
    merge_events_key_present = "merge_events" in ctx
    merge_events             = ctx.get("merge_events", [])
    merge_events_empty       = not merge_events

    # -----------------------------------------------------------------------
    # 1. Per-cluster velocity profiles
    # -----------------------------------------------------------------------
    all_labels = np.unique(
        np.concatenate([l[l >= 0] for l in labels if (l >= 0).any()])
    )
    tracked = ctx.get("tracked_cluster_ids", all_labels.tolist())

    all_steps: list[dict] = []
    for cid in tracked:
        steps = centroid_velocity_profile(
            acts, labels, layer_types, layer_names, int(cid),
            P_S=P_S_fixed,
            P_A=P_A_fixed,
            proj_per_layer=proj_per_layer,
        )
        all_steps.extend(steps)

    if not all_steps:
        return SubResult(
            name="centroid_velocity",
            applicable=False,
            payload={},
            summary_lines=["centroid_velocity: no tracked clusters with ≥2 tokens"],
            verdict_contribution={},
        )

    # -----------------------------------------------------------------------
    # 2. P6-R3: r_S elevated at merge vs plateau
    # -----------------------------------------------------------------------
    def _mean_r_S(steps, ltype):
        vals = [s["r_S"] for s in steps if s["layer_type_L"] == ltype]
        return float(np.mean(vals)) if vals else None

    mean_r_S_plateau = _mean_r_S(all_steps, "plateau")
    mean_r_S_merge   = _mean_r_S(all_steps, "merge")
    mean_r_S_other   = _mean_r_S(all_steps, "other")

    # Fix 3: require BOTH a relative gap and a meaningful absolute level.
    P6_R3_ABS_THRESHOLD = 0.20

    p6_r3 = None
    if mean_r_S_merge is not None and mean_r_S_plateau is not None:
        relative_elevated = mean_r_S_merge > mean_r_S_plateau + 0.05
        absolute_elevated = mean_r_S_merge > P6_R3_ABS_THRESHOLD
        p6_r3 = relative_elevated and absolute_elevated

    # -----------------------------------------------------------------------
    # 3. Merge geometry (P6-D5)
    # -----------------------------------------------------------------------
    merge_geom_results: list[dict] = []

    for event in merge_events:
        prev_ids = event.get("prev_ids", [])
        if len(prev_ids) < 2:
            continue
        c1, c2 = int(prev_ids[0]), int(prev_ids[1])

        dist_seq = intercentroid_distances(
            acts, labels, layer_names, c1, c2,
            U_pos=U_pos_fixed,
            U_A=U_A_fixed,
            proj_per_layer=proj_per_layer,
        )
        if not dist_seq:
            continue

        mg = merge_geometry_test(dist_seq)
        merge_geom_results.append({
            "c1": c1,
            "c2": c2,
            "n_layers_coexist": len(dist_seq),
            **mg,
        })

    n_p6d5_pass = sum(1 for r in merge_geom_results if r["p6_d5_satisfied"])

    # -----------------------------------------------------------------------
    # Payload
    # -----------------------------------------------------------------------
    payload = {
        "using_per_layer_projectors": proj_per_layer is not None,
        "n_cluster_steps":            len(all_steps),
        "mean_r_S_plateau":           mean_r_S_plateau,
        "mean_r_S_merge":             mean_r_S_merge,
        "mean_r_S_other":             mean_r_S_other,
        "p6_r3":                      p6_r3,
        "p6_r3_abs_threshold":        P6_R3_ABS_THRESHOLD,
        "n_merge_events":             len(merge_geom_results),
        "merge_events_key_present":   merge_events_key_present,
        "n_p6d5_pass":                n_p6d5_pass,
        "merge_geometry":             merge_geom_results,
    }

    # -----------------------------------------------------------------------
    # Summary lines
    # -----------------------------------------------------------------------
    proj_mode = (
        "per-layer (correct)"
        if proj_per_layer is not None
        else "WARN: projectors missing — no decomposition possible"
    )

    lines = [
        SEP_THICK,
        "CENTROID VELOCITY DECOMPOSITION  [Track B]",
        SEP_THICK,
        f"Projector mode:          {proj_mode}",
        f"Cluster steps analysed:  {len(all_steps)}",
    ]

    # Fix 4: explicit warning when merge_events cannot be evaluated
    if not merge_events_key_present:
        lines.append(
            "WARN: 'merge_events' key absent from ctx — P6-D5 cannot be evaluated."
            " Ensure Phase 1 exports merge_events with 'prev_ids' lists."
        )
    elif merge_events_empty:
        lines.append(
            "WARN: merge_events is empty — no merge events detected in Phase 1."
            " P6-D5 verdict will be None."
        )
    else:
        lines.append(f"Merge events analysed:   {len(merge_geom_results)}")

    lines += [
        "",
        "r_S = |Π_S Δx̄| / |Δx̄|  (fraction of centroid motion in real channel):",
        _bullet("mean r_S at plateau layers", mean_r_S_plateau),
        _bullet("mean r_S at merge layers",   mean_r_S_merge),
        _bullet("mean r_S at other layers",   mean_r_S_other),
        "",
        "Prediction P6-R3: r_S materially elevated at merge vs plateau.",
        f"  Threshold: merge_r_S > plateau_r_S + 0.05  AND  merge_r_S > {P6_R3_ABS_THRESHOLD}",
        _verdict_line(
            "P6-R3",
            p6_r3,
            f"r_S_merge={_fmt(mean_r_S_merge)}"
            f" vs r_S_plateau={_fmt(mean_r_S_plateau)}"
            f" (abs threshold {P6_R3_ABS_THRESHOLD})",
        ),
        "",
        "P6-D5 — Inter-centroid distance in S (full real basis) vs A approaching merge:",
        "  d_S now uses U_S = span(U_pos ∪ U_neg); repulsive-channel separation included.",
        _bullet("merge events with P6-D5 pass", n_p6d5_pass),
        _bullet("total merge events tested",    len(merge_geom_results)),
    ]

    if merge_geom_results:
        lines += ["", "  Per-merge-event geometry:"]
        lines.append(
            f"  {'c1':>4} {'c2':>4}  {'rho_S_d':>8}  {'rho_A_d':>8}  {'P6-D5':>6}"
        )
        for r in merge_geom_results:
            lines.append(
                f"  {r['c1']:>4d} {r['c2']:>4d}  "
                f"{_fmt(r['d_S_trend_rho']):>8}  "
                f"{_fmt(r['d_A_trend_rho']):>8}  "
                f"{'pass' if r['p6_d5_satisfied'] else 'fail':>6}"
            )

    p6_d5_satisfied = (
        n_p6d5_pass > len(merge_geom_results) // 2
        if merge_geom_results
        else None
    )
    lines.append(
        _verdict_line(
            "P6-D5",
            p6_d5_satisfied,
            f"{n_p6d5_pass}/{len(merge_geom_results)} merge events: "
            "d_S (full S) decreases more monotonically than d_A"
            if merge_geom_results
            else "no merge events — cannot evaluate",
        )
    )

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