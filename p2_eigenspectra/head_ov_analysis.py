"""
head_ov_analysis.py — Per-head OV decomposition and Fiedler cross-reference.

Each attention head has its own OV circuit: OV_h = W_V^{(h)} @ W_O^{(h)}.
Some heads may be net-repulsive and others net-attractive.  Phase 1's
per-head Fiedler profiling classified heads by attention connectivity
(CLUSTER < 0.3, MIXED 0.3-0.7, MIXING > 0.7).

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


def dynamic_head_activation_test(
        fiedler_data: dict,
        decomposed: dict,
        phase1_events: dict,
        beta: float = 1.0,
    ) -> dict:
    """
    Fix 12: Test whether low-Fiedler (clustering) heads receive more attention
    mass at violation layers than at non-violation layers.

    The static OV × Fiedler correlation (cross_reference_head_ov_fiedler) shows
    that repulsive-OV heads tend to have low Fiedler values globally.  This
    function tests the dynamic prediction: at violation layers specifically, is
    attention weight concentrated on the low-Fiedler heads?

    For each layer, the "weighted mean Fiedler" is:

        F̄(L) = Σ_h  A_h(L) · F_h

    where A_h(L) is the mean attention weight for head h at layer L (averaged
    over token pairs and tokens) and F_h is that head's mean Fiedler value.

    A high F̄(L) means attention is routing through mixing heads; a low F̄(L)
    means attention is routing through clustering heads.

    If violations are caused by repulsive-OV / low-Fiedler heads being activated,
    F̄ should be lower at violation layers than at non-violation layers.

    Parameters
    ----------
    fiedler_data    : from load_phase1_fiedler — contains per-layer per-head Fiedler
    decomposed      : from extract_decomposed_{albert,standard} — contains "attentions"
    phase1_events   : from load_phase1_events
    beta            : which beta's violations to compare

    Returns
    -------
    dict with:
      applicable (bool)
      weighted_fiedler_per_layer : (n_layers,) float — F̄(L)
      z_score_violation          : z-score of F̄ at violation vs non-violation layers
      v_mean, pop_mean, pop_std  : distribution statistics
      interpretation             : str
    """
    from scipy.stats import mannwhitneyu

    fiedler_layers = fiedler_data.get("fiedler_per_head_per_layer", [])
    attentions     = decomposed.get("attentions", [])
    violations     = set(phase1_events["energy_violations"].get(beta, []))

    if not fiedler_layers or not attentions:
        return {"applicable": False,
                "reason": "Missing Fiedler data or attention matrices"}

    n_layers_fiedler = len(fiedler_layers)
    n_layers_attn    = len(attentions)
    n_layers         = min(n_layers_fiedler, n_layers_attn)
    n_heads          = len(fiedler_layers[0]) if fiedler_layers else 0

    if n_heads == 0 or n_layers < 4:
        return {"applicable": False, "reason": "Too few layers or heads"}

    weighted_fiedler = np.full(n_layers, float("nan"))

    for L in range(n_layers):
        fiedler_h = np.array(fiedler_layers[L], dtype=np.float64)
        if len(fiedler_h) != n_heads:
            continue

        attn_mat = attentions[L]  # (n_heads, n_tokens, n_tokens) or similar
        if hasattr(attn_mat, "numpy"):
            attn_mat = attn_mat.numpy()
        attn_mat = np.array(attn_mat, dtype=np.float64)

        # Compute per-head mean attention weight A_h(L)
        # Shape can be (n_heads, seq, seq) or (seq, seq) — handle both
        if attn_mat.ndim == 3 and attn_mat.shape[0] == n_heads:
            # (n_heads, n_tokens, n_tokens) — mean over token pairs
            A_h = attn_mat.mean(axis=(1, 2))  # (n_heads,)
        elif attn_mat.ndim == 2:
            # Single attention matrix — uniform across heads
            A_h = np.ones(n_heads) / n_heads
        else:
            continue

        # Normalize A_h to sum to 1 (some models include diagonal)
        A_h = A_h / max(A_h.sum(), 1e-12)

        # Weighted mean Fiedler at this layer
        weighted_fiedler[L] = float(np.dot(A_h, fiedler_h))

    # Compare violation vs non-violation layers
    v_indices   = [L for L in range(n_layers) if L in violations and np.isfinite(weighted_fiedler[L])]
    pop_indices = [L for L in range(n_layers) if L not in violations and np.isfinite(weighted_fiedler[L])]

    if not v_indices or not pop_indices:
        return {
            "applicable":                 True,
            "weighted_fiedler_per_layer": weighted_fiedler.tolist(),
            "z_score_violation":          float("nan"),
            "v_mean":                     float("nan"),
            "pop_mean":                   float("nan"),
            "pop_std":                    float("nan"),
            "note":                       "Too few violation or non-violation layers",
        }

    v_vals   = weighted_fiedler[v_indices]
    pop_vals = weighted_fiedler[pop_indices]
    pop_std  = float(np.std(pop_vals))

    z_score = float((np.mean(v_vals) - np.mean(pop_vals)) / (pop_std + 1e-12))

    # Mann-Whitney U test (non-parametric, appropriate for small samples)
    try:
        stat, mw_pval = mannwhitneyu(v_vals, pop_vals, alternative="less")
        mw_pval = float(mw_pval)
    except Exception:
        mw_pval = float("nan")

    interpretation = (
        "Negative z-score: violation layers route through lower-Fiedler (clustering) "
        "heads, consistent with repulsive heads activating at violations."
        if z_score < 0 else
        "Positive z-score: violation layers route through higher-Fiedler (mixing) "
        "heads — inconsistent with V-repulsive prediction."
    )

    return {
        "applicable":                 True,
        "weighted_fiedler_per_layer": weighted_fiedler.tolist(),
        "v_indices":                  v_indices,
        "pop_indices":                pop_indices,
        "z_score_violation":          z_score,
        "v_mean":                     float(np.mean(v_vals)),
        "pop_mean":                   float(np.mean(pop_vals)),
        "pop_std":                    pop_std,
        "mw_pval":                    mw_pval,
        "n_violation_layers":         len(v_indices),
        "n_population_layers":        len(pop_indices),
        "interpretation":             interpretation,
    }


def head_ov_fiedler_crossref(
        head_fiedler_profile: dict,
        ov_per_head: list,
    ) -> dict:
    """
    Cross-reference Phase 1 head Fiedler classification with Phase 2
    per-head OV eigenspectra sign dominance.

    Prediction: CLUSTER heads (low Fiedler, routing tokens into separated
    groups) should have more attractive-dominant OV spectra (more positive
    eigenvalues), because attractive V pushes tokens together within
    clusters.  MIXING heads should have more repulsive spectra.

    Parameters
    ----------
    head_fiedler_profile : dict with "heads" list from Phase 1
                           Each entry: {"head", "mean_fiedler", "class", ...}
    ov_per_head          : list of dicts per head from Phase 2
                           Each entry: {"eigenvalues_real": [...], ...}
                           OR list of (d,) arrays of eigenvalues

    Returns
    -------
    dict with:
      spearman_rho            : correlation between Fiedler and attractive fraction
      p_value                 : significance
      prediction              : string summary
      per_class               : {CLUSTER/MIXED/MIXING: {n_heads, mean_attractive_frac}}
      per_head                : list of per-head records
    """
    from scipy import stats
    import numpy as np

    heads_info = head_fiedler_profile.get("heads", [])
    if not heads_info or not ov_per_head:
        return {"applicable": False, "error": "Missing inputs"}

    rows = []
    for h_info in heads_info:
        h = h_info["head"]
        if h >= len(ov_per_head):
            continue

        h_eig = ov_per_head[h]
        if isinstance(h_eig, dict):
            real_parts = np.array(h_eig.get("eigenvalues_real", []))
        elif hasattr(h_eig, "__len__"):
            real_parts = np.real(np.array(h_eig))
        else:
            continue

        if len(real_parts) == 0:
            continue

        n_pos = int(np.sum(real_parts > 0))
        n_neg = int(np.sum(real_parts < 0))
        attractive_frac = n_pos / max(n_pos + n_neg, 1)

        rows.append({
            "head": h,
            "fiedler_class": h_info["class"],
            "mean_fiedler": h_info["mean_fiedler"],
            "attractive_frac": attractive_frac,
            "n_positive_eig": n_pos,
            "n_negative_eig": n_neg,
        })

    if len(rows) < 3:
        return {"applicable": False, "error": "Insufficient heads"}

    fiedler_vals = [r["mean_fiedler"] for r in rows]
    attract_vals = [r["attractive_frac"] for r in rows]
    rho, p_val = stats.spearmanr(fiedler_vals, attract_vals)

    per_class = {}
    for cls in ["CLUSTER", "MIXED", "MIXING"]:
        cls_rows = [r for r in rows if r["fiedler_class"] == cls]
        if cls_rows:
            per_class[cls] = {
                "n_heads": len(cls_rows),
                "mean_attractive_frac": float(np.mean(
                    [r["attractive_frac"] for r in cls_rows]
                )),
            }

    return {
        "applicable": True,
        "spearman_rho": float(rho),
        "p_value": float(p_val),
        "n_heads": len(rows),
        "prediction": (
            "CONFIRMED: CLUSTER heads have more attractive OV spectra"
            if rho < -0.3 and p_val < 0.05
            else "NOT CONFIRMED: no significant Fiedler-OV relationship"
        ),
        "per_class": per_class,
        "per_head": rows,
    }


# ---------------------------------------------------------------------------
# Full pipeline (updated)
# ---------------------------------------------------------------------------

def run_head_analysis(
    ov_data: dict,
    run_dir: Path,
    decomposed: dict = None,
    phase1_events: dict = None,
    beta: float = 1.0,
) -> dict:
    """
    Full per-head OV × Fiedler analysis for one model × prompt.

    Parameters
    ----------
    ov_data         : from weights.analyze_weights
    run_dir         : Phase 1 run directory
    decomposed      : optional, from extract_decomposed_{albert,standard}
                      Required for dynamic_head_activation_test (fix 12).
    phase1_events   : optional, from load_phase1_events.
                      Required for dynamic_head_activation_test (fix 12).

    Returns
    -------
    dict with head_ov, fiedler, cross_reference, and dynamic_test results
    """
    head_ov = analyze_per_head_ov(ov_data)
    fiedler = load_phase1_fiedler(run_dir)

    if fiedler["n_heads"] == 0:
        return {"applicable": False, "reason": "no Fiedler data in Phase 1 run"}

    xref = cross_reference_head_ov_fiedler(head_ov, fiedler)

    result = {
        "applicable": True,
        "head_ov":    head_ov,
        "fiedler":    fiedler,
        "xref":       xref,
    }

    # Fix 12: dynamic head activation test (requires decomposed attentions)
    if decomposed is not None and phase1_events is not None:
        result["dynamic_test"] = dynamic_head_activation_test(
            fiedler_data   = fiedler,
            decomposed     = decomposed,
            phase1_events  = phase1_events,
            beta           = beta,
        )
    else:
        result["dynamic_test"] = {"applicable": False,
                                  "reason": "decomposed or phase1_events not provided"}
    
    # P1→P2 cross-reference
    fiedler_path = run_dir / "head_fiedler_profile.json"
    if fiedler_path.exists():
        with open(fiedler_path) as f:
            hfp = json.load(f)
        xref = head_ov_fiedler_crossref(hfp, ov_per_head_eigenvalues)
        result["fiedler_crossref"] = xref
    return result


def print_head_analysis_summary(result: dict, model_name: str, prompt_key: str):
    """Print concise head OV summary.  Delegates to head_ov_summary_lines."""
    if not result.get("applicable"):
        return
    print(f"\n  Head OV analysis ({model_name} | {prompt_key}):")
    for line in head_ov_summary_lines(result):
        print(f"    {line}")


def head_ov_summary_lines(result: dict) -> list[str]:
    """
    Return LLM-ready plain-text lines summarising head OV × Fiedler analysis.
 
    Extracted from print_head_analysis_summary so disk output and terminal
    output are always identical.
    """
    if not result.get("applicable"):
        return ["head_ov: not applicable"]
 
    xref = result.get("xref", {})
    if "error" in xref:
        return [f"head_ov: {xref['error']}"]
 
    L = []
    corr = xref.get("correlation_mean", {})
    sig  = "*" if corr.get("pval", 1) < 0.05 else " "
    L.append(
        f"Head OV × Fiedler:  "
        f"ρ={corr.get('rho', float('nan')):+.3f}  "
        f"p={corr.get('pval', float('nan')):.3f} {sig}  "
        f"(n={corr.get('n', '?')} heads)"
    )
    L.append(
        f"  Prediction matches: "
        f"{xref.get('n_prediction_match', '?')}/{xref.get('n_heads', '?')}"
    )
 
    # Top / bottom 4 heads by repulsive fraction
    heads = sorted(
        xref.get("per_head_summary", []),
        key=lambda h: h.get("frac_repulsive", 0), reverse=True,
    )
    if heads:
        L.append("  Per head (sorted by repulsive frac, top 4):")
        for h in heads[:4]:
            match = "✓" if h.get("prediction_match") else " "
            L.append(
                f"    H{h['head']:2d}: rep={h['frac_repulsive']:.3f}  "
                f"fiedler={h.get('fiedler_mean', float('nan')):.3f} "
                f"({h.get('fiedler_class', '?'):7s})  "
                f"ov={h.get('ov_sign', '?'):10s}  {match}"
            )
        if len(heads) > 8:
            L.append("    ...")
        for h in heads[-4:]:
            match = "✓" if h.get("prediction_match") else " "
            L.append(
                f"    H{h['head']:2d}: rep={h['frac_repulsive']:.3f}  "
                f"fiedler={h.get('fiedler_mean', float('nan')):.3f} "
                f"({h.get('fiedler_class', '?'):7s})  "
                f"ov={h.get('ov_sign', '?'):10s}  {match}"
            )
 
    # Per-layer correlations summary
    plc = xref.get("per_layer_correlations", [])
    if plc:
        rhos = [c["rho"] for c in plc if np.isfinite(c.get("rho", float("nan")))]
        sig_count = sum(1 for c in plc if c.get("pval", 1) < 0.05)
        if rhos:
            L.append(
                f"  Per-layer correlations: mean ρ={np.mean(rhos):+.3f}  "
                f"significant (p<0.05): {sig_count}/{len(plc)} layers"
            )
 
    # Dynamic activation test
    dt = result.get("dynamic_test", {})
    if dt.get("applicable"):
        z  = dt["z_score_violation"]
        vm = dt["v_mean"]
        pm = dt["pop_mean"]
        mp = dt.get("mw_pval", float("nan"))
        sig_dyn = "*" if np.isfinite(mp) and mp < 0.05 else " "
        L.append(
            f"  Dynamic activation test: z={z:+.3f}  "
            f"(viol={vm:.3f}  pop={pm:.3f}  MW p={mp:.3f}{sig_dyn})"
        )
        L.append(f"  {dt.get('interpretation', '')}")
 
    return L