"""
subexp_wrappers.py — Thin wrappers for every p2_eigenspectra sub-experiment.

Each wrapper:
  1. Calls the underlying analysis function with the right ctx fields.
  2. Calls the corresponding *_summary_lines() function to produce LLM-ready text.
  3. Returns a SubResult with payload, summary_lines, and verdict_contribution.

The *_summary_lines() functions are the source of truth for text generation;
the existing print_* functions in the analysis modules call them and print the
result, so CLI behaviour is unchanged.

Sub-experiments in dependency order (orchestrator iterates in this order):
  trajectory
  layer_v_events
  head_ov
  decomposed_violations
  ffn_subspace
  continuous_correlations
  ov_norm_confound
  zone_comparison
  attractive_zone_violations
"""

from __future__ import annotations

from pathlib import Path

from p2_eigenspectra.subresult import SubResult

# ---------------------------------------------------------------------------
# 1. trajectory
# ---------------------------------------------------------------------------

def _trajectory_subexp(ctx: dict) -> SubResult:
    """
    Wrap full_analysis (traj cross-reference: violations_beta*, rescaled).
    Depends on: ctx["traj"], ctx["ov_data"]
    """
    from p2_eigenspectra.analysis import full_analysis

    traj    = ctx["traj"]
    ov_data = ctx["ov_data"]

    result = full_analysis(traj, ov_data)

    lines = trajectory_summary_lines(result, ov_data)

    vc = {}
    for beta in [1.0, 5.0]:
        key = f"violations_beta{beta}"
        if key in result:
            s = result[key].get("summary", {})
            vc[f"beta{beta}_n_violations"]   = s.get("n_violations", 0)
            vc[f"beta{beta}_frac_overshoot"] = s.get("frac_overshoot", 0.0)
            vc[f"beta{beta}_frac_repulsive"] = s.get("frac_repulsive", 0.0)
            vc[f"beta{beta}_frac_self_neg"]  = s.get("frac_self_int_neg", 0.0)
    resc = result.get("rescaled", {})
    for beta in [1.0, 5.0]:
        bkey = f"beta_{beta}"
        if bkey in resc:
            vc[f"rescaled_improvement_beta{beta}"] = resc[bkey].get("improvement", 0)

    return SubResult(
        name="trajectory",
        applicable=True,
        payload=result,
        summary_lines=lines,
        verdict_contribution=vc,
    )


def trajectory_summary_lines(result: dict, ov_data: dict) -> list[str]:
    """Return LLM-ready lines summarising trajectory cross-reference."""
    from core.config import BETA_VALUES
    L = []
    is_per_layer = ov_data.get("is_per_layer", False)
    if is_per_layer:
        decomps = ov_data.get("decomps", [])
        mean_rep = sum(d["frac_repulsive"] for d in decomps) / max(len(decomps), 1)
        L.append(f"OV spectrum (per-layer mean): rep_frac={mean_rep:.3f}")
    else:
        d = ov_data.get("decomps", {})
        L.append(f"OV spectrum (shared): rep_frac={d.get('frac_repulsive', float('nan')):.3f}")

    for beta in BETA_VALUES:
        key = f"violations_beta{beta}"
        if key not in result:
            continue
        s = result[key].get("summary", {})
        n  = s.get("n_violations", 0)
        fo = s.get("frac_overshoot", 0.0)
        fr = s.get("frac_repulsive", 0.0)
        L.append(
            f"beta={beta}: {n} violations  "
            f"overshoot={fo:.2f}  repulsive={fr:.2f}"
        )

    resc = result.get("rescaled", {})
    for beta in [1.0, 5.0]:
        bkey = f"beta_{beta}"
        if bkey in resc:
            imp = resc[bkey].get("improvement", 0)
            n_v = result.get(f"violations_beta{beta}", {}).get("summary", {}).get("n_violations", 1)
            frac = imp / max(n_v, 1)
            L.append(f"rescaled frame (beta={beta}): {imp} violations eliminated ({frac:.0%})")

    return L


# ---------------------------------------------------------------------------
# 2. layer_v_events
# ---------------------------------------------------------------------------

def _layer_v_events_subexp(ctx: dict) -> SubResult:
    from p2_eigenspectra.layer_v_events import run_layer_v_analysis, layer_v_summary_lines

    result = run_layer_v_analysis(ctx["ov_data"], ctx["phase1_run_dir"], beta=1.0)

    if not result.get("applicable"):
        return SubResult(
            name="layer_v_events", applicable=False,
            payload=result,
            summary_lines=[f"layer_v_events: not applicable — {result.get('reason','')}"],
            verdict_contribution={},
        )

    lines = layer_v_summary_lines(result)
    zones = result.get("zones", {})
    ze    = result.get("zone_events", {})
    corr  = result.get("correlations", {})
    rv    = corr.get("repulsive_frac_vs_violation_indicator", {})

    vc = {
        "layer_v_crossover":              zones.get("crossover_layer"),
        "layer_v_n_repulsive":            zones.get("n_repulsive", 0),
        "layer_v_n_attractive":           zones.get("n_attractive", 0),
        "violations_in_repulsive_zone":   ze.get("repulsive", {}).get("n_violations", 0),
        "violations_in_attractive_zone":  ze.get("attractive", {}).get("n_violations", 0),
        "violation_rate_repulsive_zone":  ze.get("repulsive", {}).get("violation_rate", 0.0),
        "violation_rate_attractive_zone": ze.get("attractive", {}).get("violation_rate", 0.0),
        "violation_rate_transition_zone": ze.get("transition", {}).get("violation_rate", 0.0),
        "rho_repulsive_vs_violations":    rv.get("rho"),
    }

    return SubResult(
        name="layer_v_events", applicable=True,
        payload=result,
        summary_lines=lines,
        verdict_contribution=vc,
    )


# ---------------------------------------------------------------------------
# 3. head_ov
# ---------------------------------------------------------------------------

def _head_ov_subexp(ctx: dict) -> SubResult:
    from p2_eigenspectra.head_ov_analysis import run_head_analysis, head_ov_summary_lines

    result = run_head_analysis(ctx["ov_data"], ctx["phase1_run_dir"])

    if not result.get("applicable"):
        return SubResult(
            name="head_ov", applicable=False,
            payload=result,
            summary_lines=["head_ov: not applicable — ov_per_head not loaded"],
            verdict_contribution={},
        )

    lines = head_ov_summary_lines(result)
    xref  = result.get("xref", {})
    corr  = xref.get("correlation_mean", {})
    dt    = result.get("dynamic_test", {})

    vc = {
        "head_ov_fiedler_rho":  corr.get("rho"),
        "head_ov_fiedler_pval": corr.get("pval"),
    }
    if dt.get("applicable"):
        vc["head_ov_dynamic_z"] = dt.get("z_score_violation")

    return SubResult(
        name="head_ov", applicable=True,
        payload=result,
        summary_lines=lines,
        verdict_contribution=vc,
    )


# ---------------------------------------------------------------------------
# 4. decomposed_violations
# ---------------------------------------------------------------------------

def _decomposed_violations_subexp(ctx: dict) -> SubResult:
    from p2_eigenspectra.decompose import analyze_violations_decomposed

    traj       = ctx["trajectory_result"]  # payload from trajectory sub-experiment
    decomposed = ctx["decomposed"]
    events     = _extract_events(traj)

    if decomposed is None or events is None:
        return SubResult(
            name="decomposed_violations", applicable=False,
            payload={},
            summary_lines=["decomposed_violations: skipped — no decomposed deltas"],
            verdict_contribution={},
        )

    decomp_results = analyze_violations_decomposed(decomposed, events, beta=1.0)

    lines = decomposed_violations_summary_lines(decomp_results)

    n = len(decomp_results)
    n_ffn_drop  = sum(1 for d in decomp_results if d.get("ffn_sign")  == "drop")
    n_attn_drop = sum(1 for d in decomp_results if d.get("attn_sign") == "drop")
    import numpy as np
    mean_ffn_frac  = float(np.mean([min(d.get("ffn_frac",  0), 2.0) for d in decomp_results])) if n else 0.0
    mean_attn_frac = float(np.mean([min(d.get("attn_frac", 0), 2.0) for d in decomp_results])) if n else 0.0

    if   mean_ffn_frac  > 0.6: channel = "FFN"
    elif mean_attn_frac > 0.6: channel = "attention"
    else:                       channel = "mixed"

    vc = {
        "channel":                    channel,
        "decompose_n_violations":     n,
        "decompose_frac_ffn_drop":    n_ffn_drop  / n if n else 0,
        "decompose_frac_attn_drop":   n_attn_drop / n if n else 0,
        "decompose_mean_ffn_frac":    mean_ffn_frac,
        "decompose_mean_attn_frac":   mean_attn_frac,
    }

    return SubResult(
        name="decomposed_violations", applicable=True,
        payload=decomp_results,
        summary_lines=lines,
        verdict_contribution=vc,
    )


def decomposed_violations_summary_lines(decomp_results: list) -> list[str]:
    import numpy as np
    L = []
    n = len(decomp_results)
    if n == 0:
        L.append("decomposed_violations: 0 violations attributed")
        return L
    n_ffn_drop  = sum(1 for d in decomp_results if d.get("ffn_sign")  == "drop")
    n_attn_drop = sum(1 for d in decomp_results if d.get("attn_sign") == "drop")
    n_cross_drop= sum(1 for d in decomp_results if d.get("cross_sign") == "drop")
    mean_ffn  = float(np.mean([min(d.get("ffn_frac",  0), 2.0) for d in decomp_results]))
    mean_attn = float(np.mean([min(d.get("attn_frac", 0), 2.0) for d in decomp_results]))
    mean_cross= float(np.mean([d.get("delta_cross", 0) for d in decomp_results]))
    L.append(f"Decomposed attribution: {n} violations")
    L.append(f"  FFN drops:   {n_ffn_drop}/{n}  ({n_ffn_drop/n:.0%})  mean_frac={mean_ffn:.3f}")
    L.append(f"  Attn drops:  {n_attn_drop}/{n}  ({n_attn_drop/n:.0%})  mean_frac={mean_attn:.3f}")
    L.append(f"  Cross drops: {n_cross_drop}/{n}  ({n_cross_drop/n:.0%})  mean_delta={mean_cross:.3f}")
    if mean_ffn > 0.6:
        L.append("  Channel: FFN (mean_ffn_frac > 0.6)")
    elif mean_attn > 0.6:
        L.append("  Channel: attention (mean_attn_frac > 0.6)")
    else:
        L.append("  Channel: mixed")
    return L


# ---------------------------------------------------------------------------
# 5. ffn_subspace
# ---------------------------------------------------------------------------

def _ffn_subspace_subexp(ctx: dict) -> SubResult:
    from p2_eigenspectra.ffn_subspace import run_ffn_subspace_analysis, ffn_subspace_summary_lines

    stem_dir = ctx["stem_dir"]
    phase1_run_dir = ctx["phase1_run_dir"]
    # NEW: Check the directory where run_offline stored/found the deltas
    weights_decomposed_dir = Path(ctx.get("weights_dir", ".")) / ctx["stem"]

    # Try current output first
    result = run_ffn_subspace_analysis(stem_dir, ctx["ov_data"], phase1_run_dir=phase1_run_dir)
    
    # Try the weights source directory next
    if not result.get("applicable") and weights_decomposed_dir.exists():
        result = run_ffn_subspace_analysis(weights_decomposed_dir, ctx["ov_data"], phase1_run_dir=phase1_run_dir)
    
    # Fallback to Phase 1
    if not result.get("applicable"):
        result = run_ffn_subspace_analysis(phase1_run_dir, ctx["ov_data"])

    if not result.get("applicable"):
        return SubResult(
            name="ffn_subspace", applicable=False,
            payload=result,
            summary_lines=[f"ffn_subspace: not applicable — {result.get('reason', 'no ffn_deltas')}"],
            verdict_contribution={},
        )

    lines = ffn_subspace_summary_lines(result)
    s     = result.get("summary", {})
    zs    = result.get("zscores", {})
    z_rep = zs.get("ffn_repulse_frac", {}).get("z_score")

    vc = {
        "frac_ffn_amplifies_repulsive":  s.get("frac_amplifies_repulsive", 0.0),
        "frac_ffn_amplifies_attractive": s.get("frac_amplifies_attractive", 0.0),
        "frac_ffn_orthogonal":           s.get("frac_orthogonal", 0.0),
        "ffn_repulse_frac_at_violations": s.get("mean_ffn_repulse_frac"),
        "ffn_attract_frac_at_violations": s.get("mean_ffn_attract_frac"),
        "ffn_rep_zscore":                z_rep,
    }

    return SubResult(
        name="ffn_subspace", applicable=True,
        payload=result,
        summary_lines=lines,
        verdict_contribution=vc,
    )


# ---------------------------------------------------------------------------
# 6. continuous_correlations
# ---------------------------------------------------------------------------

def _continuous_correlations_subexp(ctx: dict) -> SubResult:
    from p2_eigenspectra.analysis_extended import continuous_energy_correlations
    from p2_eigenspectra.analysis_extended import continuous_correlations_summary_lines
    from p2_eigenspectra.trajectory import load_phase1_events

    events = load_phase1_events(ctx["phase1_run_dir"])
    result = continuous_energy_correlations(ctx["ov_data"], events, beta=1.0)

    if not result.get("applicable"):
        return SubResult(
            name="continuous_correlations", applicable=False,
            payload=result,
            summary_lines=[f"continuous_correlations: not applicable — {result.get('reason','')}"],
            verdict_contribution={},
        )

    lines = continuous_correlations_summary_lines(result)
    main  = result.get("repulsive_frac_vs_delta_E", {})
    vc = {
        "continuous_repfrac_vs_deltaE_rho":  main.get("rho"),
        "continuous_repfrac_vs_deltaE_pval": main.get("pval"),
    }
    return SubResult(
        name="continuous_correlations", applicable=True,
        payload=result,
        summary_lines=lines,
        verdict_contribution=vc,
    )


# ---------------------------------------------------------------------------
# 7. ov_norm_confound
# ---------------------------------------------------------------------------

def _ov_norm_confound_subexp(ctx: dict) -> SubResult:
    from p2_eigenspectra.analysis_extended import ov_norm_confound_check
    from p2_eigenspectra.analysis_extended import ov_norm_confound_summary_lines
    from p2_eigenspectra.trajectory import load_phase1_events

    events = load_phase1_events(ctx["phase1_run_dir"])
    result = ov_norm_confound_check(ctx["ov_data"], events, beta=1.0)

    if not result.get("applicable"):
        return SubResult(
            name="ov_norm_confound", applicable=False,
            payload=result,
            summary_lines=[f"ov_norm_confound: not applicable — {result.get('reason','')}"],
            verdict_contribution={},
        )

    lines   = ov_norm_confound_summary_lines(result)
    partial = result.get("partial_controlling_rep_frac", {}).get("ov_norm_vs_violations", {})
    vc = {
        "ov_norm_partial_rho":  partial.get("rho"),
        "ov_norm_partial_pval": partial.get("pval"),
        "ov_norm_is_confound":  (
            abs(partial.get("rho", 0) or 0) > 0.2 and (partial.get("pval", 1) or 1) < 0.05
        ),
    }
    return SubResult(
        name="ov_norm_confound", applicable=True,
        payload=result,
        summary_lines=lines,
        verdict_contribution=vc,
    )


# ---------------------------------------------------------------------------
# 8. zone_comparison
# ---------------------------------------------------------------------------

def _zone_comparison_subexp(ctx: dict) -> SubResult:
    from p2_eigenspectra.analysis_extended import compare_zone_methods
    from p2_eigenspectra.analysis_extended import zone_comparison_summary_lines
    from p2_eigenspectra.trajectory import load_phase1_events

    events = load_phase1_events(ctx["phase1_run_dir"])
    result = compare_zone_methods(ctx["ov_data"], events, beta=1.0)

    if not result.get("applicable"):
        return SubResult(
            name="zone_comparison", applicable=False,
            payload=result,
            summary_lines=["zone_comparison: not applicable"],
            verdict_contribution={},
        )

    lines = zone_comparison_summary_lines(result)
    return SubResult(
        name="zone_comparison", applicable=True,
        payload=result,
        summary_lines=lines,
        verdict_contribution={},
    )


# ---------------------------------------------------------------------------
# 9. attractive_zone_violations
# ---------------------------------------------------------------------------

def _attractive_zone_violations_subexp(ctx: dict) -> SubResult:
    from p2_eigenspectra.analysis_extended import attractive_zone_violation_analysis
    from p2_eigenspectra.analysis_extended import attractive_zone_violations_summary_lines

    ffn_sr   = ctx.get("ffn_subspace_result")
    lv_sr    = ctx.get("layer_v_events_result")

    if ffn_sr is None or lv_sr is None:
        return SubResult(
            name="attractive_zone_violations", applicable=False,
            payload={},
            summary_lines=["attractive_zone_violations: skipped — ffn_subspace or layer_v_events not available"],
            verdict_contribution={},
        )

    result = attractive_zone_violation_analysis(ffn_sr, lv_sr)

    if not result.get("applicable"):
        return SubResult(
            name="attractive_zone_violations", applicable=False,
            payload=result,
            summary_lines=[f"attractive_zone_violations: not applicable — {result.get('reason','')}"],
            verdict_contribution={},
        )

    lines    = attractive_zone_violations_summary_lines(result)
    att_zone = result.get("attractive", {})
    vc = {}
    if att_zone.get("n", 0) > 0:
        vc["att_zone_n_violations"]       = att_zone["n"]
        vc["att_zone_frac_amp_repulsive"]  = att_zone.get("frac_amplifies_repulsive")
        vc["att_zone_frac_amp_attractive"] = att_zone.get("frac_amplifies_attractive")
        vc["att_zone_frac_orthogonal"]     = att_zone.get("frac_orthogonal")

    return SubResult(
        name="attractive_zone_violations", applicable=True,
        payload=result,
        summary_lines=lines,
        verdict_contribution=vc,
    )


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _extract_events(traj_payload: dict) -> dict | None:
    """
    Pull the 'events' sub-dict from the trajectory payload.
    full_analysis stores events as traj["events"].
    """
    return traj_payload.get("events") if traj_payload else None
