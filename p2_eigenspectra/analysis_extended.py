"""
analysis_extended.py — Phase 2 analysis extensions.

Adds three capabilities missing from the original analysis.py:

1. Continuous ΔE correlations — replace binary violation indicators with
   actual energy change magnitude per layer.  More statistically powerful.

2. OV spectral norm confound check — partial correlation of OV norm vs
   violation indicator, controlling for repulsive fraction.  Tests whether
   spectral norm spikes independently predict violations.

3. Adaptive zone classification — replace fixed 0.55/0.45 thresholds with
   model-adaptive thresholds based on the actual distribution of per-layer
   repulsive fractions.

These plug into the existing pipeline via full_analysis_extended().

Functions
---------
continuous_energy_correlations : Spearman of V metrics vs ΔE per layer
ov_norm_confound_check         : partial correlation OV norm vs violations
classify_layers_adaptive       : model-adaptive zone thresholds
full_analysis_extended         : run all extensions, merge into existing results
"""

import numpy as np
from scipy.stats import spearmanr
from __future__ import annotations

from phase2.trajectory import load_phase1_events


# ---------------------------------------------------------------------------
# 1. Continuous energy correlations
# ---------------------------------------------------------------------------

def continuous_energy_correlations(
    ov_data: dict,
    phase1_events: dict,
    beta: float = 1.0,
) -> dict:
    """
    Correlate per-layer V metrics against continuous ΔE (energy change
    from layer L to L+1), not just binary violation indicators.

    This is more powerful than the binary test because it uses magnitude
    information.  A layer with a large energy drop and high repulsive
    fraction contributes more than one with a small drop.

    V metrics tested:
      - repulsive_frac per layer
      - coupling_product (repulsive_frac × qk_mean_norm) per layer

    Energy metrics:
      - delta_E : E(L+1) - E(L) per transition (negative = violation)
      - abs_delta_E : magnitude of change

    Returns
    -------
    dict of {metric_pair: {rho, pval, n}} correlations
    """
    if not ov_data["is_per_layer"]:
        return {"applicable": False, "reason": "shared weights"}

    decomps = ov_data["decomps"]
    qk_data = ov_data.get("qk_data", {})
    qk_norms = qk_data.get("qk_spectral_norms", [])

    energies = phase1_events["energies"].get(beta, [])
    n_layers = len(decomps)
    n_trans = len(energies) - 1

    if n_trans < 4:
        return {"applicable": False, "reason": "too few layers"}

    # Build per-layer V metrics
    rep_frac = np.array([d["frac_repulsive"] for d in decomps])

    qk_mean = np.zeros(n_layers)
    for i in range(min(n_layers, len(qk_norms))):
        qk_mean[i] = float(np.mean(qk_norms[i]))

    coupling = rep_frac * qk_mean

    # Build energy deltas (transition L→L+1 has index L)
    e_arr = np.array(energies[:n_trans + 1])
    delta_e = e_arr[1:] - e_arr[:-1]    # (n_trans,) — negative = violation

    n = min(n_trans, n_layers)
    delta_e = delta_e[:n]
    rep_trans = rep_frac[:n]
    coup_trans = coupling[:n]

    results = {"applicable": True}

    # Repulsive frac vs ΔE
    # Prediction: high repulsive frac → negative ΔE (energy drops)
    # So correlation should be negative
    mask = np.isfinite(rep_trans) & np.isfinite(delta_e)
    if mask.sum() >= 4:
        rho, pval = spearmanr(rep_trans[mask], delta_e[mask])
        results["repulsive_frac_vs_delta_E"] = {
            "rho": float(rho), "pval": float(pval), "n": int(mask.sum()),
            "interpretation": "negative ρ supports V-repulsive causing energy drops"
        }

    # Repulsive frac vs |ΔE|
    abs_delta = np.abs(delta_e)
    mask = np.isfinite(rep_trans) & np.isfinite(abs_delta)
    if mask.sum() >= 4:
        rho, pval = spearmanr(rep_trans[mask], abs_delta[mask])
        results["repulsive_frac_vs_abs_delta_E"] = {
            "rho": float(rho), "pval": float(pval), "n": int(mask.sum()),
        }

    # Coupling product vs ΔE
    mask = np.isfinite(coup_trans) & np.isfinite(delta_e) & (coup_trans > 0)
    if mask.sum() >= 4:
        rho, pval = spearmanr(coup_trans[mask], delta_e[mask])
        results["coupling_product_vs_delta_E"] = {
            "rho": float(rho), "pval": float(pval), "n": int(mask.sum()),
        }

    # Coupling product vs |ΔE|
    mask = np.isfinite(coup_trans) & np.isfinite(abs_delta) & (coup_trans > 0)
    if mask.sum() >= 4:
        rho, pval = spearmanr(coup_trans[mask], abs_delta[mask])
        results["coupling_product_vs_abs_delta_E"] = {
            "rho": float(rho), "pval": float(pval), "n": int(mask.sum()),
        }

    return results


# ---------------------------------------------------------------------------
# 2. OV spectral norm confound check
# ---------------------------------------------------------------------------

def ov_norm_confound_check(
    ov_data: dict,
    phase1_events: dict,
    beta: float = 1.0,
) -> dict:
    """
    Test whether OV spectral norm predicts violations independently of
    repulsive fraction.

    Method: compute rank-based partial correlation.
      1. Spearman of ov_norm vs violation_indicator (raw)
      2. Spearman of ov_norm vs violation_indicator, partialling out rep_frac
         (residualise both on rep_frac via rank regression)

    If the partial correlation is significant, OV norm is an independent
    predictor — meaning the spectral norm spikes (e.g., GPT-2-medium L10)
    are a confound.  If it drops to zero, norm only matters through its
    correlation with rep_frac.

    Returns
    -------
    dict with raw and partial correlations
    """
    if not ov_data["is_per_layer"]:
        return {"applicable": False, "reason": "shared weights"}

    decomps = ov_data["decomps"]
    # OV spectral norm isn't always in decomps — get from summary if available
    summary = ov_data.get("summary", {})
    layers_summary = summary.get("layers", {}) if summary else {}

    # Try to get OV norms from the ov_total matrices directly
    ov_totals = ov_data.get("ov_total", [])
    if isinstance(ov_totals, list) and len(ov_totals) > 0:
        from scipy.linalg import svdvals
        ov_norms = np.array([float(svdvals(ov)[0]) for ov in ov_totals])
    elif layers_summary:
        names = list(layers_summary.keys())
        norms = [layers_summary[n].get("ov_spectral_norm") for n in names]
        if all(v is not None for v in norms):
            ov_norms = np.array(norms)
        else:
            return {"applicable": False, "reason": "no OV norms available"}
    else:
        return {"applicable": False, "reason": "no OV data for norms"}

    rep_frac = np.array([d["frac_repulsive"] for d in decomps])

    violations = set(phase1_events["energy_violations"].get(beta, []))
    n = min(len(ov_norms), len(rep_frac))
    violation_ind = np.array([1.0 if i in violations else 0.0 for i in range(n)])

    ov_norms = ov_norms[:n]
    rep_frac = rep_frac[:n]

    if n < 6:
        return {"applicable": False, "reason": "too few layers"}

    # Raw correlation: OV norm vs violations
    rho_raw, pval_raw = spearmanr(ov_norms, violation_ind)

    # Partial correlation: residualise both on rep_frac via rank regression
    def _rank_residual(x, z):
        """Residuals of x regressed on z, using ranks."""
        from scipy.stats import rankdata
        rx = rankdata(x)
        rz = rankdata(z)
        # Simple linear regression of ranks
        slope = np.cov(rx, rz)[0, 1] / (np.var(rz) + 1e-12)
        return rx - slope * rz

    resid_norm = _rank_residual(ov_norms, rep_frac)
    resid_viol = _rank_residual(violation_ind, rep_frac)
    rho_partial, pval_partial = spearmanr(resid_norm, resid_viol)

    return {
        "applicable": True,
        "n_layers": n,
        "raw": {
            "ov_norm_vs_violations": {
                "rho": float(rho_raw), "pval": float(pval_raw),
            },
        },
        "partial_controlling_rep_frac": {
            "ov_norm_vs_violations": {
                "rho": float(rho_partial), "pval": float(pval_partial),
            },
        },
        "interpretation": (
            "If partial ρ ≈ 0, OV norm is not an independent predictor — "
            "it only correlates with violations through rep_frac. "
            "If partial ρ is significant, OV norm spikes are a confound."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Adaptive zone classification
# ---------------------------------------------------------------------------

def classify_layers_adaptive(
    ov_data: dict,
    method: str = "mad",
) -> dict:
    """
    Classify layers into repulsive/transition/attractive zones using
    model-adaptive thresholds instead of fixed 0.55/0.45.

    Methods:
      "mad"  : median ± 0.5 × MAD (median absolute deviation)
               Robust to outliers (layer 0, final layer).
      "std"  : mean ± 0.5 × std
      "fixed": original 0.55/0.45 thresholds (for comparison)

    Returns
    -------
    dict with zones, thresholds, and zone statistics
    """
    if not ov_data["is_per_layer"]:
        return {"applicable": False, "reason": "shared weights"}

    decomps = ov_data["decomps"]
    rep = np.array([d["frac_repulsive"] for d in decomps])
    n = len(rep)

    # Compute thresholds
    if method == "mad":
        median = np.median(rep)
        mad = np.median(np.abs(rep - median))
        thresh_high = median + 0.5 * mad   # above this = repulsive
        thresh_low  = median - 0.5 * mad   # below this = attractive
    elif method == "std":
        mean = np.mean(rep)
        std = np.std(rep)
        thresh_high = mean + 0.5 * std
        thresh_low  = mean - 0.5 * std
    elif method == "fixed":
        thresh_high = 0.55
        thresh_low  = 0.45
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure thresholds are sensible
    thresh_high = max(thresh_high, 0.50 + 1e-3)  # at least above 0.5
    thresh_low  = min(thresh_low, 0.50 - 1e-3)   # at least below 0.5

    # Classify
    zones = np.empty(n, dtype=object)
    for i in range(n):
        if rep[i] > thresh_high:
            zones[i] = "repulsive"
        elif rep[i] < thresh_low:
            zones[i] = "attractive"
        else:
            zones[i] = "transition"

    rep_layers = [i for i in range(n) if zones[i] == "repulsive"]
    att_layers = [i for i in range(n) if zones[i] == "attractive"]
    trans_layers = [i for i in range(n) if zones[i] == "transition"]

    # Crossover: first non-repulsive layer after the last repulsive
    crossover = None
    if rep_layers:
        for i in range(rep_layers[-1] + 1, n):
            if zones[i] != "repulsive":
                crossover = i
                break

    return {
        "applicable": True,
        "method": method,
        "thresh_high": float(thresh_high),
        "thresh_low": float(thresh_low),
        "zones": zones.tolist(),
        "n_repulsive": len(rep_layers),
        "n_transition": len(trans_layers),
        "n_attractive": len(att_layers),
        "crossover_layer": crossover,
        "repulsive_range": (rep_layers[0], rep_layers[-1]) if rep_layers else None,
        "attractive_range": (att_layers[0], att_layers[-1]) if att_layers else None,
    }


# ---------------------------------------------------------------------------
# Comparison: fixed vs adaptive zones
# ---------------------------------------------------------------------------

def compare_zone_methods(
    ov_data: dict,
    phase1_events: dict,
    beta: float = 1.0,
) -> dict:
    """
    Run zone classification with all methods and count violations per zone
    for each.  Shows how much the fixed threshold undercounts.

    Returns
    -------
    dict keyed by method, each with zone counts and violation localization
    """
    if not ov_data["is_per_layer"]:
        return {"applicable": False}

    violations = set(phase1_events["energy_violations"].get(beta, []))
    results = {}

    for method in ["fixed", "mad", "std"]:
        zones = classify_layers_adaptive(ov_data, method=method)
        if not zones.get("applicable"):
            continue

        zone_arr = zones["zones"]
        n = len(zone_arr)

        per_zone = {}
        for zname in ["repulsive", "transition", "attractive"]:
            zlayers = {i for i in range(n) if zone_arr[i] == zname}
            v_in = violations & zlayers
            per_zone[zname] = {
                "n_layers": len(zlayers),
                "n_violations": len(v_in),
                "violation_rate": len(v_in) / max(len(zlayers), 1),
            }

        results[method] = {
            "thresholds": (zones["thresh_low"], zones["thresh_high"]),
            "n_repulsive": zones["n_repulsive"],
            "n_attractive": zones["n_attractive"],
            "crossover": zones["crossover_layer"],
            "per_zone": per_zone,
        }

    return {"applicable": True, "methods": results}


# ---------------------------------------------------------------------------
# 4. Attractive-zone violation analysis
# ---------------------------------------------------------------------------

def attractive_zone_violation_analysis(
    ffn_subspace_result: dict,
    layer_v_result: dict,
) -> dict:
    """
    Filter FFN subspace per-violation results by zone to determine what
    drives violations in the attractive zone.

    For GPT-2-xl layers 39-44 (rep_frac < 0.4), are violations:
      - "orthogonal": paper's framework can't explain them
      - "amplifies_attractive": FFN overshoots in convergent direction
      - "amplifies_repulsive": FFN pushes into V-repulsive despite the
        layer being net-attractive

    Parameters
    ----------
    ffn_subspace_result : from ffn_subspace.run_ffn_subspace_analysis
    layer_v_result      : from layer_v_events.run_layer_v_analysis

    Returns
    -------
    dict with per-zone breakdown of FFN roles at violation layers
    """
    if not ffn_subspace_result.get("applicable"):
        return {"applicable": False, "reason": "no FFN subspace data"}
    if not layer_v_result.get("applicable"):
        return {"applicable": False, "reason": "no layer-V data"}

    zones = layer_v_result.get("zones", {}).get("zones", [])
    per_violation = ffn_subspace_result.get("per_violation", [])

    if not zones or not per_violation:
        return {"applicable": False, "reason": "empty data"}

    result = {"applicable": True}

    for zone_name in ["repulsive", "transition", "attractive"]:
        zone_layers = {i for i, z in enumerate(zones) if z == zone_name}
        # per_violation has "layer" field — violation at layer L means the
        # FFN at layer L-1 produced the update, so ffn_subspace uses t_idx = L-1
        # but the "layer" field in per_violation is the violation layer L
        zone_violations = [v for v in per_violation if v["layer"] in zone_layers]

        n = len(zone_violations)
        if n == 0:
            result[zone_name] = {"n": 0}
            continue

        n_amp_rep = sum(1 for v in zone_violations if v["role"] == "amplifies_repulsive")
        n_amp_att = sum(1 for v in zone_violations if v["role"] == "amplifies_attractive")
        n_orth    = sum(1 for v in zone_violations if v["role"] == "orthogonal")

        result[zone_name] = {
            "n": n,
            "frac_amplifies_repulsive": n_amp_rep / n,
            "frac_amplifies_attractive": n_amp_att / n,
            "frac_orthogonal": n_orth / n,
            "mean_ffn_repulse_frac": float(np.mean([v["ffn_repulse_frac"] for v in zone_violations])),
            "mean_ffn_attract_frac": float(np.mean([v["ffn_attract_frac"] for v in zone_violations])),
            "violations": [{"layer": v["layer"], "role": v["role"],
                            "rep": v["ffn_repulse_frac"], "att": v["ffn_attract_frac"]}
                           for v in zone_violations],
        }

    return result


# ---------------------------------------------------------------------------
# Full extended analysis
# ---------------------------------------------------------------------------

def full_analysis_extended(
    ov_data: dict,
    run_dir,
    beta: float = 1.0,
    ffn_subspace_result: dict = None,
    layer_v_result: dict = None,
) -> dict:
    """
    Run all Phase 2 analysis extensions for one model × prompt.

    Parameters
    ----------
    ffn_subspace_result : optional, from ffn_subspace.run_ffn_subspace_analysis
    layer_v_result      : optional, from layer_v_events.run_layer_v_analysis

    Returns
    -------
    dict with results keyed by test name
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    events = load_phase1_events(run_dir)

    results = {}

    # Continuous ΔE correlations
    results["continuous_correlations"] = continuous_energy_correlations(
        ov_data, events, beta
    )

    # OV norm confound
    results["ov_norm_confound"] = ov_norm_confound_check(
        ov_data, events, beta
    )

    # Adaptive zones comparison
    results["zone_comparison"] = compare_zone_methods(
        ov_data, events, beta
    )

    # Attractive-zone violation analysis (if data available)
    if ffn_subspace_result is not None and layer_v_result is not None:
        results["attractive_zone_violations"] = attractive_zone_violation_analysis(
            ffn_subspace_result, layer_v_result
        )

    return results


def print_extended_summary(results: dict, model_name: str, prompt_key: str):
    """Print concise extended analysis summary.  Delegates to *_summary_lines."""
    print(f"\n  Extended analysis ({model_name} | {prompt_key}):")
 
    for lines_fn, key in (
        (continuous_correlations_summary_lines, "continuous_correlations"),
        (ov_norm_confound_summary_lines,        "ov_norm_confound"),
        (zone_comparison_summary_lines,         "zone_comparison"),
        (attractive_zone_violations_summary_lines, "attractive_zone_violations"),
    ):
        sub = results.get(key, {})
        if sub:
            for line in lines_fn(sub):
                print(f"    {line}")


def continuous_correlations_summary_lines(result: dict) -> list[str]:
    """LLM-ready text for continuous_energy_correlations result."""
    if not result.get("applicable"):
        return [f"continuous_correlations: not applicable — {result.get('reason', '')}"]
 
    L = ["Continuous ΔE correlations (Spearman):"]
    for key in (
        "repulsive_frac_vs_delta_E",
        "coupling_product_vs_delta_E",
        "repulsive_frac_vs_abs_delta_E",
        "coupling_product_vs_abs_delta_E",
    ):
        if key in result:
            c = result[key]
            sig = "*" if c.get("pval", 1) < 0.05 else " "
            L.append(
                f"  {key:<42s}  ρ={c['rho']:+.3f}  p={c['pval']:.3f} {sig}"
                f"  n={c.get('n', '?')}"
            )
            if "interpretation" in c:
                L.append(f"    {c['interpretation']}")
 
    return L
 
 
def ov_norm_confound_summary_lines(result: dict) -> list[str]:
    """LLM-ready text for ov_norm_confound_check result."""
    if not result.get("applicable"):
        return [f"ov_norm_confound: not applicable — {result.get('reason', '')}"]
 
    raw     = result.get("raw", {}).get("ov_norm_vs_violations", {})
    partial = result.get("partial_controlling_rep_frac", {}).get("ov_norm_vs_violations", {})
 
    L = ["OV spectral norm confound check:"]
    L.append(f"  Raw correlation:     ρ={raw.get('rho', float('nan')):+.3f}  "
             f"p={raw.get('pval', float('nan')):.3f}")
    L.append(f"  Partial correlation: ρ={partial.get('rho', float('nan')):+.3f}  "
             f"p={partial.get('pval', float('nan')):.3f}  (controlling rep_frac)")
 
    p_rho  = partial.get("rho",  0) or 0
    p_pval = partial.get("pval", 1) or 1
    if abs(p_rho) < 0.1:
        L.append("  → OV norm is NOT an independent predictor of violations")
    elif p_pval < 0.05:
        L.append(f"  → OV norm IS an independent predictor (confound)  ρ_partial={p_rho:+.3f}")
    else:
        L.append(f"  → OV norm partial correlation not significant (p={p_pval:.3f})")
 
    interp = result.get("interpretation", "")
    if interp:
        L.append(f"  Note: {interp}")
 
    return L
 
 
def zone_comparison_summary_lines(result: dict) -> list[str]:
    """LLM-ready text for compare_zone_methods result."""
    if not result.get("applicable"):
        return ["zone_comparison: not applicable"]
 
    L = ["Zone classification comparison (fixed vs adaptive thresholds):"]
    for method, data in result.get("methods", {}).items():
        lo, hi = data.get("thresholds", (float("nan"), float("nan")))
        rep    = data.get("n_repulsive", "?")
        att    = data.get("n_attractive", "?")
        cross  = data.get("crossover", "?")
        pz     = data.get("per_zone", {})
        v_rep  = pz.get("repulsive", {}).get("n_violations", "?")
        v_att  = pz.get("attractive", {}).get("n_violations", "?")
        L.append(
            f"  {method:6s}  thresh=[{lo:.3f},{hi:.3f}]  "
            f"rep={rep:2}  att={att:2}  cross=L{cross}  "
            f"v_rep={v_rep}  v_att={v_att}"
        )
 
    return L
 
 
def attractive_zone_violations_summary_lines(result: dict) -> list[str]:
    """LLM-ready text for attractive_zone_violation_analysis result."""
    if not result.get("applicable"):
        return [f"attractive_zone_violations: not applicable — {result.get('reason', '')}"]
 
    L = ["FFN role by zone at violation layers:"]
    for zone_name in ("repulsive", "transition", "attractive"):
        z = result.get(zone_name, {})
        n = z.get("n", 0)
        if n == 0:
            L.append(f"  {zone_name:12s}: 0 violations")
            continue
        L.append(
            f"  {zone_name:12s} ({n:2d} violations):  "
            f"amp_rep={z.get('frac_amplifies_repulsive', 0):.0%}  "
            f"amp_att={z.get('frac_amplifies_attractive', 0):.0%}  "
            f"orth={z.get('frac_orthogonal', 0):.0%}  "
            f"mean_rep={z.get('mean_ffn_repulse_frac', float('nan')):.3f}"
        )
 
    return L

 
def build_verdict_v2_from_subresults(
    subresults: dict,   # dict[str, SubResult]
    ctx: dict,
) -> dict:
    """
    Build the machine-readable verdict from the SubResult registry output.
 
    Parameters
    ----------
    subresults : dict mapping spec.name -> SubResult (from run_one_prompt)
    ctx        : shared context dict; must contain ov_data, model_name,
                 prompt_key
 
    Returns
    -------
    verdict dict — same keys as build_verdict_v2, fully backwards compatible
    """
    import numpy as np
    from core.config import BETA_VALUES
 
    ov_data    = ctx["ov_data"]
    model_name = ctx["model_name"]
    prompt_key = ctx["prompt_key"]
 
    verdict = {"model": model_name, "prompt": prompt_key}
 
    # --- OV spectrum (from ov_data, not a sub-experiment) ---
    _add_ov_spectrum_fields(verdict, ov_data)
 
    # --- Merge all verdict_contribution dicts ---
    # Later specs win on key conflicts (dependency order is respected).
    for spec_name in [
        "trajectory",
        "layer_v_events",
        "head_ov",
        "decomposed_violations",
        "ffn_subspace",
        "continuous_correlations",
        "ov_norm_confound",
        "zone_comparison",
        "attractive_zone_violations",
    ]:
        sr = subresults.get(spec_name)
        if sr is not None:
            for k, v in sr.verdict_contribution.items():
                verdict[k] = v
 
    # Ensure channel always exists
    verdict.setdefault("channel", "unknown")
 
    # --- Falsification verdict ---
    verdict = _classify(verdict)
 
    # --- Coverage mismatch warning ---
    n_primary    = verdict.get("beta1.0_n_violations", 0) or 0
    n_decomposed = verdict.get("decompose_n_violations", 0) or 0
    if n_primary > 0 and n_decomposed != n_primary:
        verdict["decompose_coverage_warning"]     = True
        verdict["decompose_coverage_n_primary"]   = n_primary
        verdict["decompose_coverage_n_decomposed"] = n_decomposed
    else:
        verdict["decompose_coverage_warning"] = False
 
    # --- Continuous V-score (computed last so all fields are present) ---
    verdict["v_score"] = build_v_score(verdict)
 
    return verdict
 
 
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
 
def _add_ov_spectrum_fields(verdict: dict, ov_data: dict) -> None:
    """Populate verdict with OV spectrum scalars from ov_data."""
    import numpy as np
    if ov_data.get("is_per_layer"):
        decomps = ov_data.get("decomps", [])
        if decomps:
            verdict["ov_frac_repulsive_mean"] = float(
                np.mean([d["frac_repulsive"] for d in decomps])
            )
            verdict["ov_methods_agree_all"] = all(d.get("agree", False) for d in decomps)
    else:
        d = ov_data.get("decomps", {})
        if isinstance(d, dict):
            verdict["ov_frac_repulsive"] = d.get("frac_repulsive")
            verdict["ov_methods_agree"]  = d.get("agree")
 
 
def _classify(verdict: dict) -> dict:
    """
    Apply falsification logic and FFN-confirmed upgrade.
 
    This is the single source of truth for the categorical verdict.
    Fixes A and B are applied here.
    """
    import numpy as np
 
    n_violations   = verdict.get("beta1.0_n_violations", 0) or 0
    frac_repulsive = verdict.get("beta1.0_frac_repulsive", 0) or 0
    frac_overshoot = verdict.get("beta1.0_frac_overshoot", 0) or 0
    resc_imp       = verdict.get("rescaled_improvement_beta1.0", 0) or 0
    rescaled_frac  = resc_imp / max(n_violations, 1)
    ffn_frac_drop  = verdict.get("decompose_frac_ffn_drop", 0) or 0
    n_decomposed   = verdict.get("decompose_n_violations", 0) or 0
 
    if n_violations == 0:
        falsification = "no_violations"
    elif frac_overshoot > 0.5:
        falsification = "overshoot_dominant"
    elif frac_repulsive > 0.5:
        falsification = "V_repulsive_local"
    elif rescaled_frac > 0.8 and ffn_frac_drop > 0.5:
        falsification = "V_repulsive_via_FFN"
    elif rescaled_frac > 0.8 and ffn_frac_drop <= 0.5:
        falsification = "V_repulsive_via_attn"
    elif ffn_frac_drop > 0.5 and rescaled_frac < 0.2 and n_decomposed >= 3:
        # Fix B: guard on n_decomposed (not n_violations)
        falsification = "FFN_independent"
    else:
        falsification = "mixed_or_unattributed"
 
    verdict["falsification"] = falsification
 
    # --- FFN_confirmed upgrade ---
    # Fix A: require channel == "FFN" explicitly, not just != "attention"
    channel     = verdict.get("channel", "unknown")
    frac_amp    = verdict.get("frac_ffn_amplifies_repulsive", 0) or 0
    if (
        falsification == "V_repulsive_via_FFN"
        and frac_amp > 0.5
        and channel == "FFN"           # Fix A: explicit equality check
    ):
        verdict["falsification"] = "V_repulsive_via_FFN_confirmed"
 
    return verdict
 
 
# ---------------------------------------------------------------------------
# Shim — preserved for back-compat
# ---------------------------------------------------------------------------
 
def _build_verdict_v2_shim(analysis_results, ov_data, model_name, prompt_key):
    """
    Adapter: converts monolithic analysis dict into pseudo-subresults and
    calls build_verdict_v2_from_subresults.
 
    Used only by callers that have not yet been migrated to the registry path.
    """
    from phase2.subresult import SubResult
 
    # Build minimal SubResult-like wrappers from the old analysis dict
    def _sr(name, vc):
        return SubResult(name=name, applicable=True, payload={},
                         summary_lines=[], verdict_contribution=vc)
 
    from core.config import BETA_VALUES
    import numpy as np
 
    subresults = {}
 
    # trajectory
    traj_vc = {}
    for beta in BETA_VALUES:
        key = f"violations_beta{beta}"
        if key in analysis_results:
            s = analysis_results[key].get("summary", {})
            traj_vc[f"beta{beta}_n_violations"]   = s.get("n_violations", 0)
            traj_vc[f"beta{beta}_frac_overshoot"] = s.get("frac_overshoot", 0)
            traj_vc[f"beta{beta}_frac_repulsive"] = s.get("frac_repulsive", 0)
    resc = analysis_results.get("rescaled", {})
    for beta in [1.0, 5.0]:
        bkey = f"beta_{beta}"
        if bkey in resc:
            traj_vc[f"rescaled_improvement_beta{beta}"] = resc[bkey].get("improvement", 0)
    subresults["trajectory"] = _sr("trajectory", traj_vc)
 
    # decomposed_violations
    decomp = analysis_results.get("decomposed_violations", [])
    if decomp:
        n = len(decomp)
        n_ffn  = sum(1 for d in decomp if d.get("ffn_sign")  == "drop")
        n_attn = sum(1 for d in decomp if d.get("attn_sign") == "drop")
        mf = float(np.mean([min(d.get("ffn_frac",  0), 2.0) for d in decomp]))
        ma = float(np.mean([min(d.get("attn_frac", 0), 2.0) for d in decomp]))
        ch = "FFN" if mf > 0.6 else ("attention" if ma > 0.6 else "mixed")
        subresults["decomposed_violations"] = _sr("decomposed_violations", {
            "channel": ch,
            "decompose_n_violations":  n,
            "decompose_frac_ffn_drop": n_ffn  / n,
            "decompose_frac_attn_drop": n_attn / n,
            "decompose_mean_ffn_frac":  mf,
            "decompose_mean_attn_frac": ma,
        })
 
    # layer_v_events
    lv = analysis_results.get("layer_v_events", {})
    if lv.get("applicable"):
        zones = lv.get("zones", {})
        ze    = lv.get("zone_events", {})
        corr  = lv.get("correlations", {})
        rv    = corr.get("repulsive_frac_vs_violation_indicator", {})
        subresults["layer_v_events"] = _sr("layer_v_events", {
            "layer_v_crossover":              zones.get("crossover_layer"),
            "layer_v_n_repulsive":            zones.get("n_repulsive", 0),
            "layer_v_n_attractive":           zones.get("n_attractive", 0),
            "violations_in_repulsive_zone":   ze.get("repulsive", {}).get("n_violations", 0),
            "violations_in_attractive_zone":  ze.get("attractive", {}).get("n_violations", 0),
            "violation_rate_repulsive_zone":  ze.get("repulsive", {}).get("violation_rate", 0.0),
            "violation_rate_attractive_zone": ze.get("attractive", {}).get("violation_rate", 0.0),
            "violation_rate_transition_zone": ze.get("transition", {}).get("violation_rate", 0.0),
            "rho_repulsive_vs_violations":    rv.get("rho"),
        })
 
    # head_ov
    hov = analysis_results.get("head_ov", {})
    if hov.get("applicable"):
        xref = hov.get("xref", {})
        corr = xref.get("correlation_mean", {})
        subresults["head_ov"] = _sr("head_ov", {
            "head_ov_fiedler_rho":  corr.get("rho"),
            "head_ov_fiedler_pval": corr.get("pval"),
        })
 
    # ffn_subspace
    ffn = analysis_results.get("ffn_subspace", {})
    if ffn.get("applicable"):
        s  = ffn.get("summary", {})
        zs = ffn.get("zscores", {})
        subresults["ffn_subspace"] = _sr("ffn_subspace", {
            "frac_ffn_amplifies_repulsive":  s.get("frac_amplifies_repulsive", 0),
            "ffn_repulse_frac_at_violations": s.get("mean_ffn_repulse_frac"),
            "ffn_attract_frac_at_violations": s.get("mean_ffn_attract_frac"),
            "ffn_rep_zscore": zs.get("ffn_repulse_frac", {}).get("z_score"),
        })
 
    # extended
    ext = analysis_results.get("extended", {})
    cc  = ext.get("continuous_correlations", {})
    if cc.get("applicable"):
        key = cc.get("repulsive_frac_vs_delta_E", {})
        subresults["continuous_correlations"] = _sr("continuous_correlations", {
            "continuous_repfrac_vs_deltaE_rho":  key.get("rho"),
            "continuous_repfrac_vs_deltaE_pval": key.get("pval"),
        })
 
    oc = ext.get("ov_norm_confound", {})
    if oc.get("applicable"):
        partial = oc.get("partial_controlling_rep_frac", {}).get("ov_norm_vs_violations", {})
        subresults["ov_norm_confound"] = _sr("ov_norm_confound", {
            "ov_norm_partial_rho":  partial.get("rho"),
            "ov_norm_partial_pval": partial.get("pval"),
            "ov_norm_is_confound":  abs(partial.get("rho", 0) or 0) > 0.2
                                    and (partial.get("pval", 1) or 1) < 0.05,
        })
 
    azv = ext.get("attractive_zone_violations", {})
    if azv.get("applicable"):
        att = azv.get("attractive", {})
        if att.get("n", 0) > 0:
            subresults["attractive_zone_violations"] = _sr("attractive_zone_violations", {
                "att_zone_n_violations":       att["n"],
                "att_zone_frac_amp_repulsive":  att.get("frac_amplifies_repulsive"),
                "att_zone_frac_amp_attractive": att.get("frac_amplifies_attractive"),
                "att_zone_frac_orthogonal":     att.get("frac_orthogonal"),
            })
 
    ctx = {"ov_data": ov_data, "model_name": model_name, "prompt_key": prompt_key}
    return build_verdict_v2_from_subresults(subresults, ctx)