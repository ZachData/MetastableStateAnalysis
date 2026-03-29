"""
analysis.py — Cross-reference Phase 2 trajectory results with Phase 1 events.

Tests each competing explanation for energy violations and produces a
per-violation attribution.

Functions
---------
classify_violations      : per-violation-layer mechanism attribution
plateau_characterization : subspace dominance profile during plateaus
merge_prediction_test    : GPT-2 repulsive fraction vs merge location
rescaled_comparison      : metric improvement in rescaled frame
full_analysis            : run all tests, return structured verdict
"""

import numpy as np
from pathlib import Path

from core.config import BETA_VALUES
from phase1.reporting import detect_plateaus


# ---------------------------------------------------------------------------
# Violation classification
# ---------------------------------------------------------------------------

def classify_violations(
    traj_results: dict,
    beta: float = 1.0,
) -> dict:
    """
    For each energy violation layer, test which mechanism explains it.

    Tests applied per violation layer (all at transition L → L+1,
    so index into step/displacement arrays is L):

    1. Overshoot — step_mean[L] > overshoot_threshold
    2. Repulsive subspace — sym_repulse_disp_frac[L] > sym_attract_disp_frac[L]
       AND repulse frac is above the trajectory median
    3. Self-interaction — mean self_int at violation-involved tokens is negative

    Parameters
    ----------
    traj_results : dict from trajectory.analyze_trajectory_offline
    beta         : which beta's violations to classify

    Returns
    -------
    dict with:
      violations     : list of layer indices
      per_violation  : list of dicts, one per violation, each with:
        layer, overshoot (bool), repulsive_dominant (bool),
        self_int_negative (bool), step_norm (float),
        repulse_disp_frac (float), attract_disp_frac (float),
        mean_self_int_drop_tokens (float)
      summary : dict with aggregate fractions
    """
    events = traj_results["events"]
    steps  = traj_results["steps"]
    disp   = traj_results["disp"]
    si     = traj_results["self_int"]

    violations = events["energy_violations"].get(beta, [])
    drop_pairs = events["energy_drop_pairs"].get(beta, {})

    threshold     = steps["overshoot_threshold"]
    repulse_med   = float(np.median(disp["sym_repulse_disp_frac"]))

    per_violation = []
    for v_layer in violations:
        # Transition index: violation at layer v_layer means energy dropped
        # from v_layer-1 to v_layer.  The displacement index is v_layer-1.
        t_idx = v_layer - 1
        if t_idx < 0 or t_idx >= len(steps["step_mean"]):
            continue

        v = {"layer": v_layer}

        # --- Overshoot ---
        v["step_norm"] = float(steps["step_mean"][t_idx])
        v["overshoot"] = v["step_norm"] > threshold

        # --- Repulsive displacement ---
        r_frac = float(disp["sym_repulse_disp_frac"][t_idx])
        a_frac = float(disp["sym_attract_disp_frac"][t_idx])
        v["repulse_disp_frac"] = r_frac
        v["attract_disp_frac"] = a_frac
        v["repulsive_dominant"] = (r_frac > a_frac) and (r_frac > repulse_med)

        # --- Self-interaction of drop-pair tokens ---
        pairs = drop_pairs.get(v_layer, [])
        if pairs and v_layer < si["self_int"].shape[0]:
            involved_tokens = set()
            for p in pairs:
                involved_tokens.add(p[0])
                involved_tokens.add(p[1])
            involved = list(involved_tokens)
            si_vals = si["self_int"][v_layer, involved]
            v["mean_self_int_drop_tokens"] = float(si_vals.mean())
            v["self_int_negative"] = float(si_vals.mean()) < 0
        else:
            v["mean_self_int_drop_tokens"] = float("nan")
            v["self_int_negative"] = False

        per_violation.append(v)

    # Summary
    n = len(per_violation)
    summary = {
        "n_violations":       n,
        "frac_overshoot":     _frac(per_violation, "overshoot") if n else 0.0,
        "frac_repulsive":     _frac(per_violation, "repulsive_dominant") if n else 0.0,
        "frac_self_int_neg":  _frac(per_violation, "self_int_negative") if n else 0.0,
    }

    return {
        "violations":    violations,
        "per_violation": per_violation,
        "summary":       summary,
        "beta":          beta,
    }


def _frac(items, key):
    return float(np.mean([v[key] for v in items]))


# ---------------------------------------------------------------------------
# Subspace profile at violation vs non-violation layers
# ---------------------------------------------------------------------------

def violation_vs_population(traj_results: dict, beta: float = 1.0) -> dict:
    """
    Compare subspace metrics at violation layers vs all other layers.

    For each metric, compute z-score: (mean_violation - mean_pop) / std_pop.
    Positive z-score for repulsive metrics at violation layers supports
    the V-repulsion hypothesis.

    Returns
    -------
    dict with z-scores and p-value approximations for:
      repulse_activation, attract_activation, step_norm,
      self_int_mean, repulse_disp_frac
    """
    events   = traj_results["events"]
    subspace = traj_results["subspace"]
    steps    = traj_results["steps"]
    si       = traj_results["self_int"]
    disp     = traj_results["disp"]

    violations = set(events["energy_violations"].get(beta, []))
    n_layers   = len(subspace["sym_repulse_frac"])

    result = {}
    metrics = [
        ("sym_repulse_activation", subspace["sym_repulse_frac"], False),
        ("sym_attract_activation", subspace["sym_attract_frac"], False),
        ("self_int_mean",          si["self_int_mean"],          False),
        ("frac_negative_self_int", si["frac_negative"],          False),
    ]

    # Step and displacement are indexed by transition (L-1)
    # Map violation layer L to transition index L-1
    violation_transitions = {v - 1 for v in violations if v - 1 >= 0}
    n_trans = len(steps["step_mean"])

    trans_metrics = [
        ("step_norm",        steps["step_mean"],               True),
        ("repulse_disp_frac", disp["sym_repulse_disp_frac"],  True),
        ("attract_disp_frac", disp["sym_attract_disp_frac"],  True),
    ]

    for name, values, is_transition in metrics:
        v_set = violations if not is_transition else violation_transitions
        n     = n_layers if not is_transition else n_trans

        v_indices   = [i for i in range(n) if i in v_set]
        pop_indices = [i for i in range(n) if i not in v_set]

        if not v_indices or not pop_indices:
            result[name] = {"z_score": float("nan"), "v_mean": float("nan"),
                            "pop_mean": float("nan")}
            continue

        v_vals   = values[v_indices]
        pop_vals = values[pop_indices]
        pop_std  = float(np.std(pop_vals))

        result[name] = {
            "z_score":  float((np.mean(v_vals) - np.mean(pop_vals)) / (pop_std + 1e-12)),
            "v_mean":   float(np.mean(v_vals)),
            "pop_mean": float(np.mean(pop_vals)),
            "pop_std":  pop_std,
        }

    for name, values, _ in trans_metrics:
        v_indices   = [i for i in range(n_trans) if i in violation_transitions]
        pop_indices = [i for i in range(n_trans) if i not in violation_transitions]

        if not v_indices or not pop_indices:
            result[name] = {"z_score": float("nan"), "v_mean": float("nan"),
                            "pop_mean": float("nan")}
            continue

        v_vals   = values[v_indices]
        pop_vals = values[pop_indices]
        pop_std  = float(np.std(pop_vals))

        result[name] = {
            "z_score":  float((np.mean(v_vals) - np.mean(pop_vals)) / (pop_std + 1e-12)),
            "v_mean":   float(np.mean(v_vals)),
            "pop_mean": float(np.mean(pop_vals)),
            "pop_std":  pop_std,
        }

    return result


# ---------------------------------------------------------------------------
# Plateau characterization
# ---------------------------------------------------------------------------

def plateau_characterization(traj_results: dict) -> dict:
    """
    During Phase 1 plateau windows, characterize subspace dominance.

    Hypothesis: attractive subspace dominates during plateaus;
    repulsive subspace is quiet.

    Returns
    -------
    dict with:
      plateaus : list of (start, end) tuples
      per_plateau : list of dicts with mean attractive/repulsive fracs,
                    mean self-interaction, mean step norm
    """
    events   = traj_results["events"]
    subspace = traj_results["subspace"]
    si       = traj_results["self_int"]
    steps    = traj_results["steps"]

    # Detect plateaus in ip_mass_near_1 (same as Phase 1)
    mass_plateaus = detect_plateaus(events["ip_mass_near_1"], window=2, tol=0.05)
    # Also detect CKA plateaus
    cka_series = [v for v in events["cka_prev"] if not _isnan_val(v)]
    cka_plateaus = detect_plateaus(cka_series, window=2, tol=0.02) if cka_series else []

    per_plateau = []
    for start, end, _ in mass_plateaus:
        layers = list(range(start, end + 1))
        if not layers:
            continue

        p = {"start": start, "end": end}
        p["sym_attract_mean"]  = float(np.mean(subspace["sym_attract_frac"][layers]))
        p["sym_repulse_mean"]  = float(np.mean(subspace["sym_repulse_frac"][layers]))
        p["self_int_mean"]     = float(np.mean(si["self_int_mean"][layers]))
        p["frac_negative"]     = float(np.mean(si["frac_negative"][layers]))

        # Step norm: transitions within the plateau
        trans_layers = [l for l in layers if l > 0 and l - 1 < len(steps["step_mean"])]
        if trans_layers:
            p["step_mean"] = float(np.mean(steps["step_mean"][[l-1 for l in trans_layers]]))
        else:
            p["step_mean"] = float("nan")

        per_plateau.append(p)

    return {
        "mass_plateaus":  [(s, e) for s, e, _ in mass_plateaus],
        "per_plateau":    per_plateau,
        "n_plateaus":     len(per_plateau),
    }


def _isnan_val(v):
    return isinstance(v, float) and v != v


# ---------------------------------------------------------------------------
# Rescaled-frame comparison
# ---------------------------------------------------------------------------

def rescaled_comparison(traj_results: dict) -> dict:
    """
    Compare Phase 1 metrics in original vs rescaled coordinates.

    Key question: does the energy violation rate drop in the rescaled frame?

    Returns
    -------
    dict with per-beta violation counts (original vs rescaled),
    and ip_mean trajectory comparison.
    """
    events   = traj_results["events"]
    rescaled = traj_results["rescaled"]

    comparison = {}
    for beta in BETA_VALUES:
        n_orig     = len(events["energy_violations"].get(beta, []))
        n_rescaled = rescaled["n_violations"].get(beta, 0)
        comparison[f"beta_{beta}"] = {
            "violations_original": n_orig,
            "violations_rescaled": n_rescaled,
            "improvement":         n_orig - n_rescaled,
        }

    # IP mean correlation
    orig_ip = np.array(events["ip_mean"])
    resc_ip = rescaled["ip_mean"]
    n = min(len(orig_ip), len(resc_ip))
    if n > 2:
        corr = float(np.corrcoef(orig_ip[:n], resc_ip[:n])[0, 1])
    else:
        corr = float("nan")

    comparison["ip_mean_correlation"] = corr

    return comparison


# ---------------------------------------------------------------------------
# GPT-2 merge prediction (per-layer V)
# ---------------------------------------------------------------------------

def merge_prediction_test(
    ov_data: dict,
    phase1_events: dict,
) -> dict:
    """
    For per-layer models (GPT-2), test whether the repulsive eigenvalue
    fraction of V at each layer predicts merge events.

    Parameters
    ----------
    ov_data       : dict from weights.analyze_weights (must be per_layer)
    phase1_events : dict from trajectory.load_phase1_events

    Returns
    -------
    dict with:
      repulsive_per_layer : list of floats
      merge_layers        : list of ints (where spectral_k drops)
      correlation         : Spearman correlation (repulsive frac vs merge indicator)
    """
    if not ov_data["is_per_layer"]:
        return {"applicable": False, "reason": "shared weights (ALBERT)"}

    decomps  = ov_data["decomps"]
    rep_frac = [d["frac_repulsive"] for d in decomps]

    spectral_k = phase1_events["spectral_k"]
    merge_layers = []
    for i in range(1, len(spectral_k)):
        if spectral_k[i] < spectral_k[i-1]:
            merge_layers.append(i)

    # Align lengths (spectral_k may differ from number of weight layers)
    n = min(len(rep_frac), len(spectral_k))
    if n < 3:
        return {"applicable": False, "reason": "too few layers"}

    # Binary merge indicator
    merge_indicator = np.zeros(n)
    for m in merge_layers:
        if m < n:
            merge_indicator[m] = 1.0

    # Spearman correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(rep_frac[:n], merge_indicator)

    return {
        "applicable":         True,
        "repulsive_per_layer": rep_frac[:n],
        "merge_layers":       merge_layers,
        "spearman_rho":       float(rho),
        "spearman_pval":      float(pval),
    }


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def full_analysis(traj_results: dict, ov_data: dict) -> dict:
    """
    Run all Phase 2 cross-reference analyses.

    Returns
    -------
    dict with all analysis results keyed by test name.
    """
    results = {}

    # Per-beta violation classification
    for beta in BETA_VALUES:
        results[f"violations_beta{beta}"] = classify_violations(traj_results, beta)
        results[f"zscores_beta{beta}"]    = violation_vs_population(traj_results, beta)

    results["plateaus"] = plateau_characterization(traj_results)
    results["rescaled"] = rescaled_comparison(traj_results)

    # Merge prediction (GPT-2 only)
    if ov_data["is_per_layer"]:
        results["merge_prediction"] = merge_prediction_test(
            ov_data, traj_results["events"]
        )

    return results
