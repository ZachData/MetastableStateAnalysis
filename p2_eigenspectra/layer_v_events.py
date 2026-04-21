"""
layer_v_events.py — Per-layer V eigenspectrum vs Phase 1 event overlay.

For per-layer models (GPT-2, BERT), correlates the depth-dependent
repulsive fraction of V with Phase 1's merge events, energy violations,
cluster structure, and effective rank.

Tests three predictions:
  1. Energy violations concentrate in high-repulsive layers
  2. Merge events concentrate in low-repulsive (attractive) layers
  3. The repulsive-to-attractive crossover predicts the metastable window

Also computes the β × repulsive_frac product ("repulsive coupling
strength") per layer and tests whether this composite predicts violation
locations better than either factor alone.

Functions
---------
extract_perlayer_v_profile : repulsive frac, QK norm, coupling product per layer
correlate_with_phase1     : Spearman correlations against Phase 1 events
classify_layers           : partition into repulsive/transition/attractive zones
run_layer_v_analysis      : full pipeline for one model × prompt
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

from core.config import BETA_VALUES
from phase2.trajectory import load_phase1_events


# ---------------------------------------------------------------------------
# Extract per-layer V profile
# ---------------------------------------------------------------------------

def extract_perlayer_v_profile(ov_data: dict) -> dict:
    """
    Build per-layer arrays of V-related metrics.

    Parameters
    ----------
    ov_data : dict from weights.analyze_weights (must be per_layer)

    Returns
    -------
    dict with:
      repulsive_frac : (n_layers,) float
      qk_mean_norm   : (n_layers,) float — per-layer mean QK spectral norm
      ov_spectral_norm : (n_layers,) float
      coupling_product : (n_layers,) float — repulsive_frac × qk_mean_norm
      methods_agree    : (n_layers,) bool
      n_layers         : int
    """
    if not ov_data["is_per_layer"]:
        raise ValueError("extract_perlayer_v_profile requires per-layer model")

    decomps = ov_data["decomps"]
    qk_data = ov_data.get("qk_data", {})
    qk_norms = qk_data.get("qk_spectral_norms", [])

    n = len(decomps)
    rep_frac = np.array([d["frac_repulsive"] for d in decomps])

    qk_mean = np.zeros(n)
    for i in range(min(n, len(qk_norms))):
        qk_mean[i] = float(np.mean(qk_norms[i]))

    # Coupling product: how strongly the repulsive subspace is amplified
    coupling = rep_frac * qk_mean

    # Methods agreement
    agree = np.array([d.get("agree", True) for d in decomps])

    return {
        "repulsive_frac":   rep_frac,
        "qk_mean_norm":     qk_mean,
        "coupling_product":  coupling,
        "methods_agree":     agree,
        "n_layers":          n,
    }


# ---------------------------------------------------------------------------
# Correlate with Phase 1 events
# ---------------------------------------------------------------------------

def correlate_with_phase1(
    v_profile: dict,
    phase1_events: dict,
    beta: float = 1.0,
) -> dict:
    """
    Compute Spearman correlations between per-layer V metrics and
    Phase 1 event indicators.

    Phase 1 events tested:
      - violation_indicator : 1 at violation layers, 0 elsewhere
      - merge_indicator     : 1 where spectral_k drops, 0 elsewhere
      - energy_trajectory   : E_beta per layer
      - effective_rank      : per layer
      - ip_mean             : per layer

    V metrics tested:
      - repulsive_frac
      - coupling_product (repulsive_frac × QK norm)

    Returns
    -------
    dict of {metric_pair: {rho, pval, n}} dicts
    """
    n_v = v_profile["n_layers"]
    rep  = v_profile["repulsive_frac"]
    coup = v_profile["coupling_product"]

    # Build Phase 1 indicators
    violations = set(phase1_events["energy_violations"].get(beta, []))
    spectral_k = phase1_events["spectral_k"]
    energies   = phase1_events["energies"].get(beta, [])
    eff_rank   = phase1_events["effective_rank"]
    ip_mean    = phase1_events["ip_mean"]

    n = min(n_v, len(spectral_k), len(energies))
    if n < 4:
        return {"error": "too few layers for correlation"}

    # Indicators
    violation_ind = np.array([1.0 if i in violations else 0.0 for i in range(n)])
    merge_ind     = np.zeros(n)
    for i in range(1, n):
        if i < len(spectral_k) and spectral_k[i] < spectral_k[i-1]:
            merge_ind[i] = 1.0

    energy_arr  = np.array(energies[:n])
    rank_arr    = np.array(eff_rank[:n])
    ip_arr      = np.array(ip_mean[:n])

    results = {}

    v_metrics = {
        "repulsive_frac":  rep[:n],
        "coupling_product": coup[:n],
    }

    p1_metrics = {
        "violation_indicator": violation_ind,
        "merge_indicator":     merge_ind,
        "energy":              energy_arr,
        "effective_rank":      rank_arr,
        "ip_mean":             ip_arr,
        "spectral_k":          np.array(spectral_k[:n], dtype=float),
    }

    for v_name, v_arr in v_metrics.items():
        for p1_name, p1_arr in p1_metrics.items():
            # Filter nan
            mask = np.isfinite(v_arr) & np.isfinite(p1_arr)
            if mask.sum() < 4:
                results[f"{v_name}_vs_{p1_name}"] = {
                    "rho": float("nan"), "pval": float("nan"), "n": 0
                }
                continue
            rho, pval = spearmanr(v_arr[mask], p1_arr[mask])
            results[f"{v_name}_vs_{p1_name}"] = {
                "rho":  float(rho),
                "pval": float(pval),
                "n":    int(mask.sum()),
            }

    return results


# ---------------------------------------------------------------------------
# Layer zone classification
# ---------------------------------------------------------------------------

def classify_layers(v_profile: dict, threshold: float = 0.5) -> dict:
    """
    Partition layers into repulsive / transition / attractive zones.

    repulsive  : repulsive_frac > threshold + 0.05
    attractive : repulsive_frac < threshold - 0.05
    transition : everything else

    Returns
    -------
    dict with:
      zones        : (n_layers,) str array — "repulsive"/"transition"/"attractive"
      repulsive_range : (first, last) layer indices of the repulsive zone
      attractive_range: (first, last) layer indices of the attractive zone
      crossover_layer : layer where the zone first transitions from repulsive
    """
    rep = v_profile["repulsive_frac"]
    n   = len(rep)

    zones = np.empty(n, dtype=object)
    for i in range(n):
        if rep[i] > threshold + 0.05:
            zones[i] = "repulsive"
        elif rep[i] < threshold - 0.05:
            zones[i] = "attractive"
        else:
            zones[i] = "transition"

    # Find contiguous zones
    repulsive_layers = [i for i in range(n) if zones[i] == "repulsive"]
    attractive_layers = [i for i in range(n) if zones[i] == "attractive"]

    rep_range = (repulsive_layers[0], repulsive_layers[-1]) if repulsive_layers else None
    att_range = (attractive_layers[0], attractive_layers[-1]) if attractive_layers else None

    # Crossover: first layer after the repulsive zone that is not repulsive
    crossover = None
    if repulsive_layers:
        for i in range(repulsive_layers[-1] + 1, n):
            if zones[i] != "repulsive":
                crossover = i
                break

    return {
        "zones":            zones.tolist(),
        "repulsive_range":  rep_range,
        "attractive_range": att_range,
        "crossover_layer":  crossover,
        "n_repulsive":      len(repulsive_layers),
        "n_transition":     sum(1 for z in zones if z == "transition"),
        "n_attractive":     len(attractive_layers),
    }


# ---------------------------------------------------------------------------
# Test: do violations/merges localize by zone?
# ---------------------------------------------------------------------------

def zone_event_localization(
    zones: dict,
    phase1_events: dict,
    beta: float = 1.0,
) -> dict:
    """
    Count what fraction of violations and merges fall in each zone.

    If the repulsive mechanism is correct:
      - violations should be enriched in repulsive zones
      - merges should be enriched in attractive zones

    Returns
    -------
    dict with zone-wise event fractions
    """
    zone_arr    = zones["zones"]
    n           = len(zone_arr)
    violations  = set(phase1_events["energy_violations"].get(beta, []))
    spectral_k  = phase1_events["spectral_k"]

    merges = set()
    for i in range(1, min(n, len(spectral_k))):
        if spectral_k[i] < spectral_k[i-1]:
            merges.add(i)

    result = {}
    for zone_name in ["repulsive", "transition", "attractive"]:
        zone_layers = {i for i in range(n) if zone_arr[i] == zone_name}
        n_zone = len(zone_layers)
        if n_zone == 0:
            result[zone_name] = {
                "n_layers": 0,
                "n_violations": 0, "frac_violations": 0.0,
                "n_merges": 0, "frac_merges": 0.0,
            }
            continue

        v_in_zone = violations & zone_layers
        m_in_zone = merges & zone_layers

        result[zone_name] = {
            "n_layers":       n_zone,
            "n_violations":   len(v_in_zone),
            "frac_violations": len(v_in_zone) / max(len(violations), 1),
            "violation_rate":  len(v_in_zone) / n_zone,
            "n_merges":       len(m_in_zone),
            "frac_merges":    len(m_in_zone) / max(len(merges), 1),
            "merge_rate":     len(m_in_zone) / n_zone,
        }

    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_layer_v_analysis(
    ov_data: dict,
    run_dir: Path,
    beta: float = 1.0,
) -> dict:
    """
    Full per-layer V vs Phase 1 events analysis.

    Parameters
    ----------
    ov_data : from weights.analyze_weights (must be per_layer)
    run_dir : Phase 1 run directory

    Returns
    -------
    dict with v_profile, correlations, zones, zone_events
    """
    if not ov_data["is_per_layer"]:
        return {"applicable": False, "reason": "shared weights"}

    v_profile = extract_perlayer_v_profile(ov_data)
    events    = load_phase1_events(run_dir)
    corr      = correlate_with_phase1(v_profile, events, beta)
    zones     = classify_layers(v_profile)
    zone_ev   = zone_event_localization(zones, events, beta)

    return {
        "applicable":    True,
        "v_profile":     {k: v.tolist() if hasattr(v, 'tolist') else v
                          for k, v in v_profile.items()},
        "correlations":  corr,
        "zones":         zones,
        "zone_events":   zone_ev,
    }


def print_layer_v_summary(result: dict, model_name: str, prompt_key: str):
    """Print concise layer-V analysis summary.  Delegates to summary_lines."""
    if not result.get("applicable"):
        return
    print(f"\n  Layer-V analysis: {model_name} | {prompt_key}")
    for line in layer_v_summary_lines(result):
        print(f"    {line}")

 
def layer_v_summary_lines(result: dict) -> list[str]:
    """
    Return LLM-ready plain-text lines summarising layer-V analysis.
 
    Extracted from print_layer_v_summary so disk output and terminal output
    are always identical.
    """
    if not result.get("applicable"):
        return [f"layer_v_events: not applicable — {result.get('reason', '')}"]
 
    L = []
    zones = result["zones"]
    L.append(
        f"Layer-V zones: {zones['n_repulsive']} repulsive, "
        f"{zones['n_transition']} transition, "
        f"{zones['n_attractive']} attractive"
    )
    if zones.get("crossover_layer") is not None:
        L.append(f"  Crossover at layer {zones['crossover_layer']}")
 
    ze = result["zone_events"]
    for zone_name in ("repulsive", "transition", "attractive"):
        z = ze.get(zone_name, {})
        if z.get("n_layers", 0) == 0:
            continue
        L.append(
            f"  {zone_name:12s}: {z['n_layers']:2d} layers  "
            f"violations={z['n_violations']:2d} ({z['violation_rate']:.2f}/layer)  "
            f"merges={z.get('n_merges', 0):2d} ({z.get('merge_rate', 0.0):.2f}/layer)"
        )
 
    corr = result.get("correlations", {})
    L.append("Spearman correlations:")
    for key in (
        "repulsive_frac_vs_violation_indicator",
        "repulsive_frac_vs_merge_indicator",
        "coupling_product_vs_violation_indicator",
        "repulsive_frac_vs_effective_rank",
    ):
        if key in corr:
            c = corr[key]
            sig = "*" if c.get("pval", 1) < 0.05 else " "
            L.append(
                f"  {key:<45s}  ρ={c['rho']:+.3f}  p={c['pval']:.3f} {sig}"
            )
 
    return L