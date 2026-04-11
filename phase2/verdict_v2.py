"""
verdict_v2.py — Revised falsification verdict logic.

The original reporting.py treats FFN_dominant and V_repulsive_supported
as mutually exclusive.  The data shows they co-occur: FFN drops energy
at ~100% of GPT-2 violations, yet the rescaled frame (which factors out V)
eliminates 100% of violations.  FFN is the channel; V is the cause.

This module provides build_verdict_v2 as a drop-in replacement for
reporting.build_verdict.

New verdict categories
----------------------
V_repulsive_local       : displacement test passes (>50% repulsive),
                          regardless of channel.  V is locally detectable.
                          (ALBERT-xlarge regime)

V_repulsive_via_FFN     : displacement test fails (<50%), but rescaled frame
                          eliminates >80% of violations AND FFN is the
                          proximal dropper at >50% of violations.
                          V is globally causal but operates through FFN.
                          (GPT-2-large/xl regime)

V_repulsive_via_attn    : displacement test fails, rescaled frame helps,
                          AND attention is the proximal dropper.
                          (would apply if ALBERT-xlarge had low displacement
                          but high rescaled improvement)

FFN_independent         : FFN drops at >50% of violations, rescaled frame
                          does NOT help (improvement < 20% of violations).
                          FFN is causing violations independently of V.

overshoot_dominant      : >50% of violations are overshoot-attributed

mixed_or_unattributed   : no single mechanism dominates

no_violations           : E_beta is monotone (shouldn't happen empirically)
"""

import numpy as np

from core.config import BETA_VALUES


def build_v_score(verdict: dict) -> float:
    """
    Compute a continuous V-score summarising the strength of evidence for
    V-repulsive dynamics as the cause of energy violations.

    Score = 0.40 × rescaled_frac
          + 0.25 × frac_repulsive_disp
          + 0.20 × frac_ffn_amplifies_repulsive
          - 0.15 × |ov_norm_partial_rho|

    Weights are theory-motivated:
      rescaled_frac              — strongest evidence (V factored out globally)
      frac_repulsive_disp        — local detection (displacement test)
      frac_ffn_amplifies_repulsive — confirmatory (FFN→V causal chain)
      |ov_norm_partial_rho|      — discount for spectral-norm confound

    Range: approximately [-0.15, 1.0].  Scores above ~0.5 consistently
    correspond to V_repulsive_via_FFN_confirmed or V_repulsive_local verdicts.

    Parameters
    ----------
    verdict : dict from build_verdict_v2 (all fields must be populated)

    Returns
    -------
    float — V-score ∈ [-0.15, 1.0] (approximately)
    """
    # --- Rescaled fraction: proportion of violations eliminated by rescaling ---
    n_viol = verdict.get("beta1.0_n_violations", 0)
    rescaled_imp = verdict.get("rescaled_improvement_beta1.0", 0)
    rescaled_frac = (rescaled_imp / max(n_viol, 1)) if n_viol > 0 else 0.0
    rescaled_frac = float(np.clip(rescaled_frac, 0.0, 1.0))

    # --- Displacement repulsive fraction (from trajectory analysis) ---
    # beta1.0_frac_repulsive is from classify_violations: fraction of violation
    # layers where the displacement test flags V-repulsive dominant.
    frac_rep_disp = float(np.clip(
        verdict.get("beta1.0_frac_repulsive", 0.0), 0.0, 1.0
    ))

    # --- FFN amplifies repulsive (from ffn_subspace) ---
    frac_ffn_amp = float(np.clip(
        verdict.get("frac_ffn_amplifies_repulsive", 0.0), 0.0, 1.0
    ))

    # --- OV norm confound discount ---
    ov_norm_partial = verdict.get("ov_norm_partial_rho", 0.0)
    if not np.isfinite(ov_norm_partial):
        ov_norm_partial = 0.0
    confound_penalty = float(np.clip(abs(ov_norm_partial), 0.0, 1.0))

    v_score = (
        0.40 * rescaled_frac
        + 0.25 * frac_rep_disp
        + 0.20 * frac_ffn_amp
        - 0.15 * confound_penalty
    )

    return float(v_score)


def build_verdict_v2(
    analysis_results: dict,
    ov_data: dict,
    model_name: str,
    prompt_key: str,
) -> dict:
    """
    Build a revised machine-readable verdict dict.

    Separates the causal question (is V responsible?) from the channel
    question (does the energy drop come through attention or FFN?).
    """
    verdict = {
        "model":  model_name,
        "prompt": prompt_key,
    }

    # --- OV spectrum ---
    if ov_data["is_per_layer"]:
        decomps = ov_data["decomps"]
        verdict["ov_frac_repulsive_mean"] = float(np.mean(
            [d["frac_repulsive"] for d in decomps]))
        verdict["ov_methods_agree_all"] = all(d["agree"] for d in decomps)
    else:
        d = ov_data["decomps"]
        verdict["ov_frac_repulsive"] = d["frac_repulsive"]
        verdict["ov_methods_agree"]  = d["agree"]

    # --- Per-beta metrics ---
    for beta in BETA_VALUES:
        key = f"violations_beta{beta}"
        if key in analysis_results:
            s = analysis_results[key]["summary"]
            verdict[f"beta{beta}_n_violations"]    = s["n_violations"]
            verdict[f"beta{beta}_frac_overshoot"]  = s["frac_overshoot"]
            verdict[f"beta{beta}_frac_repulsive"]  = s["frac_repulsive"]
            verdict[f"beta{beta}_frac_self_neg"]   = s["frac_self_int_neg"]

    # --- Rescaled improvement ---
    resc = analysis_results.get("rescaled", {})
    for beta in [1.0, 5.0]:
        bkey = f"beta_{beta}"
        if bkey in resc:
            verdict[f"rescaled_improvement_beta{beta}"] = resc[bkey]["improvement"]

    # --- Decompose (attn vs FFN) ---
    # channel is always written here so downstream verdict logic always finds it,
    # even when decomposed_violations is empty.
    verdict["channel"] = "unknown"

    decomp = analysis_results.get("decomposed_violations", [])
    if decomp:
        n = len(decomp)
        n_attn_drop  = sum(1 for d in decomp if d.get("attn_sign") == "drop")
        n_ffn_drop   = sum(1 for d in decomp if d.get("ffn_sign") == "drop")
        n_cross_drop = sum(1 for d in decomp if d.get("cross_sign") == "drop")
        n_attn_opposes = sum(1 for d in decomp if d.get("attn_opposes", False))
        n_ffn_opposes  = sum(1 for d in decomp if d.get("ffn_opposes",  False))
        # Signed drop-fracs from fix 3; cap at 2.0 to prevent huge means when
        # components nearly cancel.
        mean_attn_frac = float(np.mean([min(d["attn_frac"], 2.0) for d in decomp]))
        mean_ffn_frac  = float(np.mean([min(d["ffn_frac"],  2.0) for d in decomp]))
        mean_cross = float(np.mean([d.get("delta_cross", 0) for d in decomp]))

        verdict["decompose_n_violations"]    = n
        verdict["decompose_attn_drop"]       = n_attn_drop
        verdict["decompose_ffn_drop"]        = n_ffn_drop
        verdict["decompose_cross_drop"]      = n_cross_drop
        verdict["decompose_frac_attn_drop"]  = n_attn_drop / n if n else 0
        verdict["decompose_frac_ffn_drop"]   = n_ffn_drop  / n if n else 0
        verdict["decompose_frac_cross_drop"] = n_cross_drop / n if n else 0
        verdict["decompose_mean_attn_frac"]  = mean_attn_frac
        verdict["decompose_mean_ffn_frac"]   = mean_ffn_frac
        verdict["decompose_mean_cross_delta"] = mean_cross
        verdict["decompose_n_attn_opposes"]  = n_attn_opposes
        verdict["decompose_n_ffn_opposes"]   = n_ffn_opposes
        verdict["decompose_frac_attn_opposes"] = n_attn_opposes / n if n else 0
        verdict["decompose_frac_ffn_opposes"]  = n_ffn_opposes  / n if n else 0

        # Channel classification uses mean drop-fracs from fix 3.
        # FFN/attn thresholds are kept at 0.6; with signed fracs a value > 0.6
        # means that component explains > 60 % of the realised drop on average.
        if mean_ffn_frac > 0.6:
            verdict["channel"] = "FFN"
        elif mean_attn_frac > 0.6:
            verdict["channel"] = "attention"
        else:
            verdict["channel"] = "mixed"

    # --- Falsification verdict (revised) ---
    primary = analysis_results.get("violations_beta1.0", {}).get("summary", {})

    # --- Fix 7: decompose coverage mismatch warning ---
    # If the decompose path covered fewer violations than the primary summary,
    # record a warning.  Downstream callers that check decompose_frac_ffn_drop
    # or channel should treat the result with extra caution.
    n_primary = primary.get("n_violations", 0)
    n_decomposed = verdict.get("decompose_n_violations", 0)
    if n_primary > 0 and n_decomposed != n_primary:
        verdict["decompose_coverage_warning"] = True
        verdict["decompose_coverage_n_primary"]   = n_primary
        verdict["decompose_coverage_n_decomposed"] = n_decomposed
    else:
        verdict["decompose_coverage_warning"] = False
    n_violations = primary.get("n_violations", 0)
    frac_repulsive = primary.get("frac_repulsive", 0)
    frac_overshoot = primary.get("frac_overshoot", 0)

    rescaled_imp_1 = resc.get("beta_1.0", {}).get("improvement", 0)
    rescaled_frac = rescaled_imp_1 / max(n_violations, 1)

    ffn_frac_drop = verdict.get("decompose_frac_ffn_drop", 0)

    # Fix 4: Use decompose_n_violations (the count that actually went through the
    # energy_by_component path) rather than n_violations from the primary summary.
    # The two can differ when violations fall outside the t_idx bounds of the
    # saved delta arrays.  Requiring n >= 3 on the decomposed count prevents the
    # FFN_independent verdict from firing on runs where only 1-2 violations were
    # decomposed but several more exist that were never attributed.
    n_decomposed = verdict.get("decompose_n_violations", 0)

    if n_violations == 0:
        verdict["falsification"] = "no_violations"

    elif frac_overshoot > 0.5:
        verdict["falsification"] = "overshoot_dominant"

    elif frac_repulsive > 0.5:
        # Local displacement test passes — V is directly detectable
        verdict["falsification"] = "V_repulsive_local"

    elif rescaled_frac > 0.8 and ffn_frac_drop > 0.5:
        # Rescaled frame eliminates most violations, FFN is the channel
        verdict["falsification"] = "V_repulsive_via_FFN"

    elif rescaled_frac > 0.8 and ffn_frac_drop <= 0.5:
        # Rescaled frame works, attention is the channel
        verdict["falsification"] = "V_repulsive_via_attn"

    elif (ffn_frac_drop > 0.5 and rescaled_frac < 0.2
          and n_decomposed >= 3):        # Fix 4: guard on decomposed count
        # FFN drops but rescaling doesn't help — FFN is independent of V.
        # Require n_decomposed >= 3 (not n_violations) so we only fire when
        # enough violations actually went through the attribution path.
        verdict["falsification"] = "FFN_independent"

    else:
        verdict["falsification"] = "mixed_or_unattributed"

    # --- Layer-V zone events (per-layer models only) ---
    lv = analysis_results.get("layer_v_events", {})
    if lv.get("applicable"):
        zones = lv.get("zones", {})
        verdict["layer_v_crossover"] = zones.get("crossover_layer")
        verdict["layer_v_n_repulsive"] = zones.get("n_repulsive", 0)
        verdict["layer_v_n_attractive"] = zones.get("n_attractive", 0)

        ze = lv.get("zone_events", {})
        rep_zone = ze.get("repulsive", {})
        att_zone = ze.get("attractive", {})
        verdict["violations_in_repulsive_zone"]  = rep_zone.get("n_violations", 0)
        verdict["violations_in_attractive_zone"] = att_zone.get("n_violations", 0)
        # Fix 8: per-layer violation rates for normalised cross-model comparison.
        verdict["violation_rate_repulsive_zone"]  = rep_zone.get("violation_rate", 0.0)
        verdict["violation_rate_attractive_zone"] = att_zone.get("violation_rate", 0.0)
        verdict["violation_rate_transition_zone"] = (
            ze.get("transition", {}).get("violation_rate", 0.0)
        )

        corr = lv.get("correlations", {})
        rv = corr.get("repulsive_frac_vs_violation_indicator", {})
        verdict["rho_repulsive_vs_violations"] = rv.get("rho", float("nan"))

    # --- Per-head OV × Fiedler ---
    hov = analysis_results.get("head_ov", {})
    if hov.get("applicable"):
        xref = hov.get("xref", {})
        corr = xref.get("correlation_mean", {})
        verdict["head_ov_fiedler_rho"]  = corr.get("rho", float("nan"))
        verdict["head_ov_fiedler_pval"] = corr.get("pval", float("nan"))

    # --- FFN subspace projection (new) ---
    ffn_sub = analysis_results.get("ffn_subspace", {})
    if ffn_sub.get("applicable"):
        s = ffn_sub.get("summary", {})
        verdict["ffn_repulse_frac_at_violations"] = s.get("mean_ffn_repulse_frac", float("nan"))
        verdict["ffn_attract_frac_at_violations"] = s.get("mean_ffn_attract_frac", float("nan"))
        verdict["frac_ffn_amplifies_repulsive"]   = s.get("frac_amplifies_repulsive", 0)

        # Refine verdict using FFN subspace data.
        # Fix 5: require channel == "FFN" explicitly.
        # The previous guard was `channel != "attention"`, which fired when
        # channel was "unknown" (empty decomp path) — producing false _confirmed
        # verdicts on runs where the attribution path was never executed.
        # Requiring channel == "FFN" ensures the subspace projection result is
        # consistent with the decompose-based channel classification.
        channel = verdict.get("channel", "unknown")
        if (verdict["falsification"] == "V_repulsive_via_FFN"
                and s.get("frac_amplifies_repulsive", 0) > 0.5
                and channel == "FFN"):          # Fix 5: must be explicitly FFN
            verdict["falsification"] = "V_repulsive_via_FFN_confirmed"
            # FFN is pushing into V's repulsive subspace — strongest evidence

    # --- Extended analysis ---
    ext = analysis_results.get("extended", {})
    cc = ext.get("continuous_correlations", {})
    if cc.get("applicable"):
        for key in ["repulsive_frac_vs_delta_E"]:
            if key in cc:
                verdict[f"continuous_{key}_rho"] = cc[key]["rho"]
                verdict[f"continuous_{key}_pval"] = cc[key]["pval"]

    oc = ext.get("ov_norm_confound", {})
    if oc.get("applicable"):
        partial = oc["partial_controlling_rep_frac"]["ov_norm_vs_violations"]
        verdict["ov_norm_partial_rho"]  = partial["rho"]
        verdict["ov_norm_partial_pval"] = partial["pval"]
        verdict["ov_norm_is_confound"]  = abs(partial["rho"]) > 0.2 and partial["pval"] < 0.05

    # --- Attractive-zone violation analysis ---
    azv = ext.get("attractive_zone_violations", {})
    if azv.get("applicable"):
        att_zone = azv.get("attractive", {})
        if att_zone.get("n", 0) > 0:
            verdict["att_zone_n_violations"] = att_zone["n"]
            verdict["att_zone_frac_amp_repulsive"] = att_zone["frac_amplifies_repulsive"]
            verdict["att_zone_frac_amp_attractive"] = att_zone["frac_amplifies_attractive"]
            verdict["att_zone_frac_orthogonal"] = att_zone["frac_orthogonal"]

    # --- Fix 14: continuous V-score ---
    # Computed last so all contributing fields are already populated.
    verdict["v_score"] = build_v_score(verdict)

    return verdict
