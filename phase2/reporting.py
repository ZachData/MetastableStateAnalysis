"""
reporting.py — Phase 2 structured reporting.

Produces per-run terminal output and a machine-readable verdict dict.
"""

import json
import numpy as np
from pathlib import Path

from core.config import BETA_VALUES


def print_phase2_summary(
    analysis_results: dict,
    ov_data: dict,
    model_name: str,
    prompt_key: str,
) -> None:
    """Print concise Phase 2 terminal summary for one run."""

    print(f"\n{'='*70}")
    print(f"Phase 2: {model_name} | {prompt_key}")
    print(f"{'='*70}")

    # Weight decomposition
    if ov_data["is_per_layer"]:
        decomps = ov_data["decomps"]
        print(f"\n  OV eigenspectrum (per-layer, showing first/last):")
        for idx in [0, len(decomps) - 1]:
            d = decomps[idx]
            name = ov_data["layer_names"][idx]
            agree = "✓" if d["agree"] else "✗"
            print(f"    {name:12s}  attract={d['frac_attractive']:.2f}  "
                  f"repulse={d['frac_repulsive']:.2f}  "
                  f"sym_agree={agree}")
    else:
        d = ov_data["decomps"]
        agree = "✓" if d["agree"] else "✗"
        print(f"\n  OV eigenspectrum (shared):")
        print(f"    attract={d['frac_attractive']:.2f}  "
              f"repulse={d['frac_repulsive']:.2f}  "
              f"complex={d['frac_complex']:.2f}  sym_agree={agree}")

    # Violation classification
    for beta in [1.0]:  # Primary beta
        key = f"violations_beta{beta}"
        if key not in analysis_results:
            continue
        vc = analysis_results[key]
        s  = vc["summary"]
        print(f"\n  Energy violations (β={beta}): {s['n_violations']} total")
        if s["n_violations"] > 0:
            print(f"    Overshoot-explained:  {s['frac_overshoot']:.0%}")
            print(f"    V-repulsive-dominant: {s['frac_repulsive']:.0%}")
            print(f"    Self-int negative:    {s['frac_self_int_neg']:.0%}")

            # Per-violation detail
            for v in vc["per_violation"][:5]:  # Show first 5
                flags = []
                if v["overshoot"]:
                    flags.append("OVERSHOOT")
                if v["repulsive_dominant"]:
                    flags.append("V-REPULSIVE")
                if v["self_int_negative"]:
                    flags.append("NEG-SELF-INT")
                flag_str = " ".join(flags) if flags else "unattributed"
                print(f"      L{v['layer']:3d}  step={v['step_norm']:.3f}  "
                      f"rep_disp={v['repulse_disp_frac']:.3f}  "
                      f"att_disp={v['attract_disp_frac']:.3f}  "
                      f"→ {flag_str}")

    # Z-scores
    zkey = "zscores_beta1.0"
    if zkey in analysis_results:
        zs = analysis_results[zkey]
        print(f"\n  Violation-vs-population z-scores (β=1.0):")
        for metric in ["sym_repulse_activation", "step_norm",
                       "repulse_disp_frac", "self_int_mean"]:
            if metric in zs:
                z = zs[metric]
                print(f"    {metric:30s}  z={z['z_score']:+.2f}  "
                      f"(viol={z['v_mean']:.4f}  pop={z['pop_mean']:.4f})")

    # Plateau characterization
    plat = analysis_results.get("plateaus", {})
    if plat.get("n_plateaus", 0) > 0:
        print(f"\n  Plateau subspace profile ({plat['n_plateaus']} plateaus):")
        for p in plat["per_plateau"][:3]:
            print(f"    L{p['start']:3d}–{p['end']:3d}  "
                  f"attract={p['sym_attract_mean']:.3f}  "
                  f"repulse={p['sym_repulse_mean']:.3f}  "
                  f"self_int={p['self_int_mean']:.4f}  "
                  f"step={p['step_mean']:.3f}")

    # Rescaled comparison
    resc = analysis_results.get("rescaled", {})
    if resc:
        print(f"\n  Rescaled-frame comparison:")
        for beta in [1.0, 5.0]:
            bkey = f"beta_{beta}"
            if bkey in resc:
                r = resc[bkey]
                print(f"    β={beta}: violations {r['violations_original']} → "
                      f"{r['violations_rescaled']}  "
                      f"(Δ={r['improvement']:+d})")
        if "ip_mean_correlation" in resc:
            print(f"    ip_mean original↔rescaled correlation: "
                  f"{resc['ip_mean_correlation']:.3f}")

    # Merge prediction (GPT-2)
    mp = analysis_results.get("merge_prediction", {})
    if mp.get("applicable"):
        print(f"\n  Merge prediction (V repulsive frac vs merge indicator):")
        print(f"    Spearman ρ = {mp['spearman_rho']:.3f}  "
              f"p = {mp['spearman_pval']:.3f}")

    # Decomposed attn vs FFN attribution
    decomp = analysis_results.get("decomposed_violations", [])
    if decomp:
        print(f"\n  Attn vs FFN attribution at violation layers ({len(decomp)} violations):")
        n_attn_drop = sum(1 for d in decomp if d.get("attn_sign") == "drop")
        n_ffn_drop  = sum(1 for d in decomp if d.get("ffn_sign") == "drop")
        n_cross_drop = sum(1 for d in decomp if d.get("cross_sign") == "drop")
        mean_attn_frac = float(np.mean([d["attn_frac"] for d in decomp]))
        mean_ffn_frac  = float(np.mean([d["ffn_frac"] for d in decomp]))
        cross_deltas = [d.get("delta_cross", 0) for d in decomp]
        mean_cross = float(np.mean(cross_deltas)) if cross_deltas else 0
        print(f"    Attn causes drop:  {n_attn_drop}/{len(decomp)}  "
              f"mean weight: {mean_attn_frac:.2f}")
        print(f"    FFN  causes drop:  {n_ffn_drop}/{len(decomp)}  "
              f"mean weight: {mean_ffn_frac:.2f}")
        print(f"    Cross-term drops:  {n_cross_drop}/{len(decomp)}  "
              f"mean Δ_cross: {mean_cross:+.6f}")
        for d in decomp[:5]:
            cross = d.get("delta_cross", 0)
            cross_s = d.get("cross_sign", "?")
            print(f"      L{d['layer']:3d}  "
                  f"Δ_total={d['delta_total']:+.6f}  "
                  f"Δ_attn={d['delta_attn']:+.6f} ({d['attn_sign']})  "
                  f"Δ_ffn={d['delta_ffn']:+.6f} ({d['ffn_sign']})  "
                  f"Δ_cross={cross:+.6f} ({cross_s})")


def build_verdict(
    analysis_results: dict,
    ov_data: dict,
    model_name: str,
    prompt_key: str,
) -> dict:
    """
    Build a machine-readable verdict dict for cross-run aggregation.
    """
    verdict = {
        "model":  model_name,
        "prompt": prompt_key,
    }

    # OV spectrum
    if ov_data["is_per_layer"]:
        decomps = ov_data["decomps"]
        verdict["ov_frac_repulsive_mean"] = float(np.mean(
            [d["frac_repulsive"] for d in decomps]))
        verdict["ov_methods_agree_all"] = all(d["agree"] for d in decomps)
    else:
        d = ov_data["decomps"]
        verdict["ov_frac_repulsive"] = d["frac_repulsive"]
        verdict["ov_methods_agree"]  = d["agree"]

    # Per-beta attribution
    for beta in BETA_VALUES:
        key = f"violations_beta{beta}"
        if key in analysis_results:
            s = analysis_results[key]["summary"]
            verdict[f"beta{beta}_n_violations"]    = s["n_violations"]
            verdict[f"beta{beta}_frac_overshoot"]  = s["frac_overshoot"]
            verdict[f"beta{beta}_frac_repulsive"]  = s["frac_repulsive"]
            verdict[f"beta{beta}_frac_self_neg"]   = s["frac_self_int_neg"]

    # Rescaled improvement
    resc = analysis_results.get("rescaled", {})
    for beta in [1.0, 5.0]:
        bkey = f"beta_{beta}"
        if bkey in resc:
            verdict[f"rescaled_improvement_beta{beta}"] = resc[bkey]["improvement"]

    # Falsification status
    primary = analysis_results.get("violations_beta1.0", {}).get("summary", {})
    if primary.get("n_violations", 0) == 0:
        verdict["falsification"] = "no_violations"
    elif primary.get("frac_repulsive", 0) > 0.5:
        verdict["falsification"] = "V_repulsive_supported"
    elif primary.get("frac_overshoot", 0) > 0.5:
        verdict["falsification"] = "overshoot_dominant"
    else:
        verdict["falsification"] = "mixed_or_unattributed"

    # Decomposed attn vs FFN attribution
    decomp = analysis_results.get("decomposed_violations", [])
    if decomp:
        n_attn_drop = sum(1 for d in decomp if d.get("attn_sign") == "drop")
        n_ffn_drop  = sum(1 for d in decomp if d.get("ffn_sign") == "drop")
        verdict["decompose_n_violations"]  = len(decomp)
        verdict["decompose_attn_drop"]     = n_attn_drop
        verdict["decompose_ffn_drop"]      = n_ffn_drop
        verdict["decompose_frac_attn_drop"] = n_attn_drop / len(decomp) if decomp else 0
        verdict["decompose_frac_ffn_drop"]  = n_ffn_drop / len(decomp) if decomp else 0
        verdict["decompose_mean_attn_frac"] = float(np.mean([d["attn_frac"] for d in decomp]))
        verdict["decompose_mean_ffn_frac"]  = float(np.mean([d["ffn_frac"] for d in decomp]))

        # Refine falsification using decompose data
        if verdict["falsification"] == "mixed_or_unattributed":
            if n_ffn_drop > len(decomp) * 0.5:
                verdict["falsification"] = "FFN_dominant"
            elif n_attn_drop > len(decomp) * 0.5 and primary.get("frac_repulsive", 0) > 0.3:
                verdict["falsification"] = "V_repulsive_via_attn"

    # Layer-V zone events (per-layer models only)
    lv = analysis_results.get("layer_v_events", {})
    if lv.get("applicable"):
        zones = lv.get("zones", {})
        verdict["layer_v_crossover"] = zones.get("crossover_layer")
        verdict["layer_v_n_repulsive"] = zones.get("n_repulsive", 0)
        verdict["layer_v_n_attractive"] = zones.get("n_attractive", 0)

        ze = lv.get("zone_events", {})
        rep_zone = ze.get("repulsive", {})
        att_zone = ze.get("attractive", {})
        verdict["violations_in_repulsive_zone"] = rep_zone.get("n_violations", 0)
        verdict["violations_in_attractive_zone"] = att_zone.get("n_violations", 0)
        verdict["merges_in_repulsive_zone"]     = rep_zone.get("n_merges", 0)
        verdict["merges_in_attractive_zone"]    = att_zone.get("n_merges", 0)

        corr = lv.get("correlations", {})
        rv = corr.get("repulsive_frac_vs_violation_indicator", {})
        verdict["rho_repulsive_vs_violations"] = rv.get("rho", float("nan"))
        rm = corr.get("repulsive_frac_vs_merge_indicator", {})
        verdict["rho_repulsive_vs_merges"] = rm.get("rho", float("nan"))

    # Per-head OV × Fiedler cross-reference
    hov = analysis_results.get("head_ov", {})
    if hov.get("applicable"):
        xref = hov.get("xref", {})
        corr = xref.get("correlation_mean", {})
        verdict["head_ov_fiedler_rho"]  = corr.get("rho", float("nan"))
        verdict["head_ov_fiedler_pval"] = corr.get("pval", float("nan"))
        verdict["head_ov_prediction_match"] = xref.get("n_prediction_match", 0)
        verdict["head_ov_n_heads"] = xref.get("n_heads", 0)

    return verdict


def save_verdict(verdict: dict, save_dir: Path) -> None:
    """Save verdict to JSON."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "phase2_verdict.json"
    with open(path, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"  Verdict saved to {path}")
