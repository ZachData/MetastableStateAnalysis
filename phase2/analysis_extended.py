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
    """Print concise summary of extended analysis."""
    print(f"\n  Extended analysis ({model_name} | {prompt_key}):")

    # Continuous correlations
    cc = results.get("continuous_correlations", {})
    if cc.get("applicable"):
        print(f"    Continuous ΔE correlations:")
        for key in ["repulsive_frac_vs_delta_E", "coupling_product_vs_delta_E"]:
            if key in cc:
                c = cc[key]
                sig = "*" if c["pval"] < 0.05 else " "
                print(f"      {key:40s}  ρ={c['rho']:+.3f}  p={c['pval']:.3f} {sig}")

    # OV norm confound
    oc = results.get("ov_norm_confound", {})
    if oc.get("applicable"):
        raw = oc["raw"]["ov_norm_vs_violations"]
        partial = oc["partial_controlling_rep_frac"]["ov_norm_vs_violations"]
        print(f"    OV norm confound:")
        print(f"      Raw:     ρ={raw['rho']:+.3f}  p={raw['pval']:.3f}")
        print(f"      Partial: ρ={partial['rho']:+.3f}  p={partial['pval']:.3f}")
        if abs(partial["rho"]) < 0.1:
            print(f"      → Norm is NOT an independent predictor")
        elif partial["pval"] < 0.05:
            print(f"      → Norm IS an independent predictor (confound)")

    # Zone comparison
    zc = results.get("zone_comparison", {})
    if zc.get("applicable"):
        print(f"    Zone classification comparison:")
        for method, data in zc.get("methods", {}).items():
            lo, hi = data["thresholds"]
            rep = data["n_repulsive"]
            att = data["n_attractive"]
            cross = data["crossover"]
            v_rep = data["per_zone"]["repulsive"]["n_violations"]
            v_att = data["per_zone"]["attractive"]["n_violations"]
            print(f"      {method:6s}  thresh=[{lo:.3f},{hi:.3f}]  "
                  f"rep={rep:2d}  att={att:2d}  cross=L{cross}  "
                  f"v_rep={v_rep}  v_att={v_att}")

    # Attractive-zone violation analysis
    azv = results.get("attractive_zone_violations", {})
    if azv.get("applicable"):
        print(f"    FFN role by zone at violation layers:")
        for zone_name in ["repulsive", "transition", "attractive"]:
            z = azv.get(zone_name, {})
            n = z.get("n", 0)
            if n == 0:
                continue
            print(f"      {zone_name:12s} ({n:2d} violations):  "
                  f"amp_rep={z['frac_amplifies_repulsive']:.0%}  "
                  f"amp_att={z['frac_amplifies_attractive']:.0%}  "
                  f"orth={z['frac_orthogonal']:.0%}  "
                  f"mean_rep={z['mean_ffn_repulse_frac']:.3f}")
