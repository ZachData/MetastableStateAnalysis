"""
head_ov_analysis.py — Per-head OV decomposition and Fiedler cross-reference.

Each attention head has its own OV circuit: OV_h = W_V^{(h)} @ W_O^{(h)}.
Some heads may be net-repulsive and others net-attractive.  Phase 1's
per-head Fiedler profiling classified heads by attention connectivity
(CLUSTER < 0.3, MIXED 0.3–0.7, MIXING > 0.7).

This module tests the prediction: repulsive-dominant heads should have
low Fiedler values (cluster-separating), attractive-dominant heads should
have high Fiedler values (mixing).

Functions
---------
analyze_per_head_ov       : eigendecompose each head's OV_h
load_phase1_fiedler       : extract per-head Fiedler from Phase 1 metrics
cross_reference_head_ov_fiedler : correlate OV sign with Fiedler
run_head_analysis         : full pipeline
"""

import json
import numpy as np
from pathlib import Path
from scipy.linalg import eigvals, svdvals
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Per-head OV analysis
# ---------------------------------------------------------------------------

def analyze_per_head_ov(ov_data: dict) -> dict:
    """
    Eigendecompose each head's OV_h matrix separately.

    For shared-weight models (ALBERT): one set of per-head results.
    For per-layer models (GPT-2): per-layer × per-head results.

    Returns
    -------
    dict with:
      per_head : list of dicts (one per head for ALBERT, or list of lists
                 for per-layer models), each containing:
        frac_repulsive, frac_attractive, spectral_norm, eig_real_mean
    """
    if ov_data["is_per_layer"]:
        all_layers = []
        for layer_heads in ov_data["ov_per_head"]:
            layer_results = [_analyze_single_head(OV_h) for OV_h in layer_heads]
            all_layers.append(layer_results)
        return {
            "is_per_layer": True,
            "per_layer_per_head": all_layers,
            "n_layers": len(all_layers),
            "n_heads": len(all_layers[0]) if all_layers else 0,
        }
    else:
        per_head = [_analyze_single_head(OV_h) for OV_h in ov_data["ov_per_head"]]
        return {
            "is_per_layer": False,
            "per_head": per_head,
            "n_heads": len(per_head),
        }


def _analyze_single_head(OV_h: np.ndarray) -> dict:
    """Eigendecompose one head's OV_h (d_model × d_model)."""
    eigs     = eigvals(OV_h)
    eig_real = np.real(eigs)
    sv       = svdvals(OV_h)

    frac_pos = float((eig_real > 0).mean())
    frac_neg = float((eig_real < 0).mean())

    return {
        "frac_repulsive":  frac_neg,
        "frac_attractive": frac_pos,
        "spectral_norm":   float(sv[0]) if len(sv) > 0 else 0.0,
        "eig_real_mean":   float(eig_real.mean()),
        "eig_real_std":    float(eig_real.std()),
        "n_positive":      int((eig_real > 0).sum()),
        "n_negative":      int((eig_real < 0).sum()),
    }


# ---------------------------------------------------------------------------
# Load Phase 1 per-head Fiedler values
# ---------------------------------------------------------------------------

def load_phase1_fiedler(run_dir: Path) -> dict:
    """
    Extract per-head Fiedler values from Phase 1 metrics.json.

    Returns
    -------
    dict with:
      fiedler_per_head_per_layer : list of lists (n_layers × n_heads)
      fiedler_mean_per_head      : (n_heads,) mean Fiedler across layers
      n_heads, n_layers
    """
    run_dir = Path(run_dir)
    with open(run_dir / "metrics.json") as f:
        results = json.load(f)

    layers = results.get("layers", [])
    fiedler_per_layer = []

    for layer in layers:
        sk = layer.get("sinkhorn", {})
        fph = sk.get("fiedler_per_head", [])
        if fph:
            fiedler_per_layer.append(fph)

    if not fiedler_per_layer:
        return {"n_heads": 0, "n_layers": 0}

    n_heads  = len(fiedler_per_layer[0])
    n_layers = len(fiedler_per_layer)

    # Mean Fiedler per head across layers
    arr = np.array(fiedler_per_layer)  # (n_layers, n_heads)
    mean_per_head = arr.mean(axis=0)   # (n_heads,)

    return {
        "fiedler_per_head_per_layer": fiedler_per_layer,
        "fiedler_mean_per_head":      mean_per_head.tolist(),
        "n_heads":  n_heads,
        "n_layers": n_layers,
    }


# ---------------------------------------------------------------------------
# Cross-reference
# ---------------------------------------------------------------------------

def cross_reference_head_ov_fiedler(
    head_ov: dict,
    fiedler_data: dict,
) -> dict:
    """
    Correlate per-head OV repulsive fraction with per-head mean Fiedler.

    Prediction: repulsive heads → low Fiedler (cluster-separating)
                attractive heads → high Fiedler (mixing)
    This would show as a negative Spearman correlation.

    For shared-weight models: one correlation across heads.
    For per-layer models: correlation at each layer and across
    the mean.

    Returns
    -------
    dict with:
      correlation_mean : {rho, pval, n} — across head means
      per_head_summary : list of dicts per head
      per_layer_correlations : list of {rho, pval} (per-layer models only)
    """
    fiedler_mean = fiedler_data.get("fiedler_mean_per_head", [])
    if not fiedler_mean:
        return {"error": "no Fiedler data"}

    n_heads = len(fiedler_mean)
    fiedler_arr = np.array(fiedler_mean)

    # Extract per-head repulsive fractions
    if not head_ov["is_per_layer"]:
        per_head = head_ov["per_head"]
        if len(per_head) != n_heads:
            return {"error": f"head count mismatch: OV has {len(per_head)}, "
                             f"Fiedler has {n_heads}"}
        rep_arr = np.array([h["frac_repulsive"] for h in per_head])
    else:
        # For per-layer: average across layers
        all_reps = []
        for layer_heads in head_ov["per_layer_per_head"]:
            if len(layer_heads) == n_heads:
                all_reps.append([h["frac_repulsive"] for h in layer_heads])
        if not all_reps:
            return {"error": "no matching per-layer head data"}
        rep_arr = np.mean(all_reps, axis=0)  # (n_heads,)

    # Correlation
    if len(rep_arr) < 3:
        return {"error": "too few heads"}

    rho, pval = spearmanr(rep_arr, fiedler_arr)

    # Per-head summary
    per_head_summary = []
    for h in range(n_heads):
        classification = ("CLUSTER" if fiedler_arr[h] < 0.3 else
                          "MIXING" if fiedler_arr[h] > 0.7 else "MIXED")
        sign = "repulsive" if rep_arr[h] > 0.5 else "attractive"
        per_head_summary.append({
            "head":            h,
            "frac_repulsive":  float(rep_arr[h]),
            "fiedler_mean":    float(fiedler_arr[h]),
            "fiedler_class":   classification,
            "ov_sign":         sign,
            "prediction_match": (sign == "repulsive" and classification == "CLUSTER") or
                                (sign == "attractive" and classification == "MIXING"),
        })

    # Per-layer correlations (per-layer models only)
    per_layer_corr = []
    if head_ov["is_per_layer"]:
        fiedler_layers = fiedler_data.get("fiedler_per_head_per_layer", [])
        for l_idx, layer_heads in enumerate(head_ov["per_layer_per_head"]):
            if l_idx >= len(fiedler_layers) or len(layer_heads) != n_heads:
                continue
            l_rep = np.array([h["frac_repulsive"] for h in layer_heads])
            l_fiedler = np.array(fiedler_layers[l_idx])
            if len(l_rep) >= 3:
                r, p = spearmanr(l_rep, l_fiedler)
                per_layer_corr.append({"layer": l_idx, "rho": float(r), "pval": float(p)})

    return {
        "correlation_mean": {"rho": float(rho), "pval": float(pval), "n": n_heads},
        "per_head_summary": per_head_summary,
        "per_layer_correlations": per_layer_corr,
        "n_prediction_match": sum(1 for h in per_head_summary if h["prediction_match"]),
        "n_heads": n_heads,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_head_analysis(
    ov_data: dict,
    run_dir: Path,
) -> dict:
    """
    Full per-head OV × Fiedler analysis for one model × prompt.

    Parameters
    ----------
    ov_data : from weights.analyze_weights
    run_dir : Phase 1 run directory

    Returns
    -------
    dict with head_ov, fiedler, cross_reference results
    """
    head_ov  = analyze_per_head_ov(ov_data)
    fiedler  = load_phase1_fiedler(run_dir)

    if fiedler["n_heads"] == 0:
        return {"applicable": False, "reason": "no Fiedler data in Phase 1 run"}

    xref = cross_reference_head_ov_fiedler(head_ov, fiedler)

    return {
        "applicable": True,
        "head_ov":    head_ov,
        "fiedler":    fiedler,
        "xref":       xref,
    }


def print_head_analysis_summary(result: dict, model_name: str, prompt_key: str):
    """Print concise per-head OV × Fiedler summary."""
    if not result.get("applicable"):
        return

    xref = result["xref"]
    if "error" in xref:
        print(f"\n  Head OV × Fiedler: {xref['error']}")
        return

    corr = xref["correlation_mean"]
    sig  = "*" if corr["pval"] < 0.05 else " "
    print(f"\n  Head OV × Fiedler ({model_name} | {prompt_key}):")
    print(f"    Correlation: ρ={corr['rho']:+.3f}  p={corr['pval']:.3f} {sig}  "
          f"(n={corr['n']} heads)")
    print(f"    Prediction matches: {xref['n_prediction_match']}/{xref['n_heads']}")

    # Show heads sorted by repulsive fraction
    heads = sorted(xref["per_head_summary"],
                   key=lambda h: h["frac_repulsive"], reverse=True)
    print(f"    Per head (sorted by repulsive frac):")
    for h in heads[:4]:
        match = "✓" if h["prediction_match"] else " "
        print(f"      H{h['head']:2d}: rep={h['frac_repulsive']:.3f}  "
              f"fiedler={h['fiedler_mean']:.3f} ({h['fiedler_class']:7s})  "
              f"ov={h['ov_sign']:10s}  {match}")
    if len(heads) > 8:
        print(f"      ...")
    for h in heads[-4:]:
        match = "✓" if h["prediction_match"] else " "
        print(f"      H{h['head']:2d}: rep={h['frac_repulsive']:.3f}  "
              f"fiedler={h['fiedler_mean']:.3f} ({h['fiedler_class']:7s})  "
              f"ov={h['ov_sign']:10s}  {match}")

    # Per-layer correlation summary if available
    plc = xref.get("per_layer_correlations", [])
    if plc:
        rhos = [c["rho"] for c in plc if np.isfinite(c["rho"])]
        sig_count = sum(1 for c in plc if c["pval"] < 0.05)
        if rhos:
            print(f"    Per-layer correlations: mean ρ={np.mean(rhos):+.3f}  "
                  f"significant at p<0.05: {sig_count}/{len(plc)} layers")
