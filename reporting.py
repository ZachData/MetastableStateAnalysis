"""
reporting.py — Text reports and terminal summaries.

Functions
---------
detect_plateaus          : find flat regions in a 1D signal
print_summary            : concise per-run terminal output
print_run_summary        : verbose terminal output from a saved run dir
generate_llm_report      : self-contained plain-text file for LLM analysis
generate_cross_run_report: comparative report across all model/prompt combos

Private helpers: _trend, _plateau_str, _merge_events, _method_agreement
"""

import numpy as np
from collections import Counter
from pathlib import Path

from config import BETA_VALUES, DISTANCE_THRESHOLDS, PROMPTS


# ---------------------------------------------------------------------------
# Signal-shape helpers
# ---------------------------------------------------------------------------

def _trend(values: list) -> str:
    """Describe trajectory shape in plain English."""
    if len(values) < 3:
        return "insufficient data"
    first_third = np.mean(values[:len(values) // 3])
    last_third  = np.mean(values[-len(values) // 3:])
    mid_third   = np.mean(values[len(values) // 3: 2 * len(values) // 3])
    delta       = last_third - first_third
    mid_dip     = mid_third < min(first_third, last_third) - 0.05 * abs(delta)
    mid_peak    = mid_third > max(first_third, last_third) + 0.05 * abs(delta)
    if abs(delta) < 0.02:
        return "flat"
    elif mid_dip:
        return f"rises then dips then rises (range {first_third:.3f}→{last_third:.3f})"
    elif mid_peak:
        return f"rises to peak then falls (range {first_third:.3f}→{last_third:.3f})"
    elif delta > 0:
        return f"monotone increase ({first_third:.3f}→{last_third:.3f})"
    else:
        return f"monotone decrease ({first_third:.3f}→{last_third:.3f})"


def _plateau_str(plateaus: list) -> str:
    if not plateaus:
        return "none detected"
    return "; ".join(
        f"layers {s}-{e} (width={e-s+1}, mean={v:.4f})"
        for s, e, v in plateaus
    )


def _merge_events(spectral_k: list) -> list:
    """
    Layers where spectral k drops (cluster merge events).

    Returns list of (layer, k_before, k_after).
    """
    return [
        (i, spectral_k[i - 1], spectral_k[i])
        for i in range(1, len(spectral_k))
        if spectral_k[i] < spectral_k[i - 1]
    ]


def _method_agreement(results: dict) -> list:
    """
    Layers where agglomerative, KMeans, spectral, and Sinkhorn cluster
    count estimates all agree within ±1.

    Returns list of (layer, [counts]).
    """
    mid_thresh       = float(DISTANCE_THRESHOLDS[len(DISTANCE_THRESHOLDS) // 2])
    agreement_layers = []
    for r in results["layers"]:
        agg_k  = r["clustering"]["agglomerative"].get(mid_thresh)
        km_k   = r["clustering"]["kmeans"]["best_k"]
        sp_k   = r["spectral"]["k_eigengap"]
        sk_k   = round(r.get("sinkhorn", {}).get("sinkhorn_cluster_count_mean", -99))
        counts = [k for k in [agg_k, km_k, sp_k, sk_k] if k and k > 0]
        if counts and (max(counts) - min(counts)) <= 1:
            agreement_layers.append((r["layer"], counts))
    return agreement_layers


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------

def detect_plateaus(values: list, window: int = 2, tol: float = 0.05) -> list:
    """
    Find contiguous windows where the signal is approximately flat.

    Parameters
    ----------
    values : 1D list/array
    window : minimum plateau width (number of steps)
    tol    : maximum relative span within the plateau

    Returns
    -------
    list of (start, end, mean_value) tuples
    """
    plateaus = []
    n = len(values)
    i = 0
    while i < n - window:
        segment = values[i:i + window + 1]
        span    = max(segment) - min(segment)
        ref     = abs(np.mean(segment)) + 1e-8
        if span / ref < tol:
            j = i + window
            while j < n - 1:
                extended = values[i:j + 2]
                if (max(extended) - min(extended)) / (abs(np.mean(extended)) + 1e-8) < tol:
                    j += 1
                else:
                    break
            plateaus.append((i, j, float(np.mean(values[i:j + 1]))))
            i = j + 1
        else:
            i += 1
    return plateaus


# ---------------------------------------------------------------------------
# Terminal summaries
# ---------------------------------------------------------------------------

def print_summary(results: dict):
    """Concise per-run terminal output with plateau detection."""
    model  = results["model"]
    prompt = results["prompt"]
    print(f"\n{'='*60}")
    print(f"Model: {model} | Prompt: {prompt}")
    print(f"Tokens: {results['n_tokens']} | d_model: {results['d_model']}")

    metrics = [
        ("Mass-near-1",   [r["ip_mass_near_1"]            for r in results["layers"]], 0.10),
        ("Effective rank",[r["effective_rank"]             for r in results["layers"]], 0.05),
        ("Spectral k",    [r["spectral"]["k_eigengap"]     for r in results["layers"]], 0.0),
    ]
    for name, vals, tol in metrics:
        plateaus = detect_plateaus(vals, window=2, tol=tol)
        print(f"\n  {name} plateaus (candidate metastable windows):")
        for start, end, val in plateaus:
            print(f"    Layers {start}–{end}  (mean={val:.3f})")

    has_sk = [r for r in results["layers"] if "sinkhorn" in r]
    if has_sk:
        fiedler = [r["sinkhorn"]["fiedler_mean"] for r in has_sk]
        print(f"\n  Sinkhorn Fiedler plateaus (low = cluster-separated):")
        for start, end, val in detect_plateaus(fiedler, window=2, tol=0.05):
            print(f"    Layers {start}–{end}  (mean={val:.4f})")


def print_run_summary(run_dir: Path):
    """
    Verbose text summary of a saved run — no plots, no model needed.

    Usage:
        python run.py --summary metastability_results/albert-base-v2_wiki_paragraph
    """
    from io_utils import load_run

    run_dir = Path(run_dir)
    results = load_run(run_dir)
    layers  = results["layers"]

    print(f"\n{'='*70}")
    print(f"MODEL  : {results['model']}")
    print(f"PROMPT : {results['prompt']}")
    print(f"TOKENS : {results['n_tokens']}   D_MODEL : {results['d_model']}")
    print(f"LAYERS : {results['n_layers']}")

    tokens = results.get("tokens", [])
    if tokens:
        print(f"\nTOKENS:")
        print("  " + "  ".join(f"{i}:{t}" for i, t in enumerate(tokens)))

    print(f"\n{'─'*70}")
    print(f"{'Layer':>6}  {'ip_mean':>8}  {'mass>0.9':>9}  "
          f"{'eff_rank':>9}  {'spec_k':>7}  {'fiedler':>8}  {'sk_k':>5}")
    print(f"{'─'*70}")

    for r in layers:
        sk = r.get("sinkhorn", {})
        print(
            f"{r['layer']:>6}  "
            f"{r['ip_mean']:>8.4f}  "
            f"{r['ip_mass_near_1']:>9.4f}  "
            f"{r['effective_rank']:>9.2f}  "
            f"{r['spectral']['k_eigengap']:>7}  "
            f"{sk.get('fiedler_mean', float('nan')):>8.4f}  "
            f"{sk.get('sinkhorn_cluster_count_mean', float('nan')):>5.1f}"
        )

    print(f"\n{'─'*70}")
    print("PLATEAU DETECTION:")
    mass1  = [r["ip_mass_near_1"]         for r in layers]
    erank  = [r["effective_rank"]          for r in layers]
    spec_k = [r["spectral"]["k_eigengap"]  for r in layers]

    for name, vals, tol in [
        ("Mass-near-1",   mass1,  0.10),
        ("Effective rank",erank,  0.05),
        ("Spectral k",    spec_k, 0.0),
    ]:
        plateaus = detect_plateaus(vals, window=2, tol=tol)
        print(f"\n  {name}:")
        if plateaus:
            for s, e, v in plateaus:
                print(f"    Layers {s:2d}–{e:2d}  mean={v:.4f}  width={e-s+1}")
        else:
            print("    No plateaus detected")

    has_sk = [r for r in layers if "sinkhorn" in r]
    if has_sk:
        fiedler = [r["sinkhorn"]["fiedler_mean"] for r in has_sk]
        print(f"\n  Sinkhorn Fiedler (low = cluster-separated):")
        for s, e, v in detect_plateaus(fiedler, window=2, tol=0.05):
            print(f"    Layers {s:2d}–{e:2d}  mean={v:.5f}  width={e-s+1}")


# ---------------------------------------------------------------------------
# LLM analysis report (single run)
# ---------------------------------------------------------------------------

def generate_llm_report(results: dict, save_dir: Path):
    """
    Write a self-contained plain-text analysis report intended to be pasted
    directly into an LLM context alongside the paper and codebase.

    Sections:
      1. Run metadata
      2. Theoretical predictions to check against
      3. Per-layer data table
      4. Trend descriptions
      5. Plateau locations
      6. Merge event detection
      7. Method agreement analysis
      8. Energy trajectory analysis
      9. Sinkhorn / attention analysis
      10. PCA variance trajectory
      11. Inner-product histogram summary
      12. Flagged anomalies + open questions
    """
    layers = results["layers"]
    tokens = results.get("tokens", [])
    model  = results["model"]
    prompt = results["prompt"]

    # --- Pre-compute derived series ---
    mass1     = [r["ip_mass_near_1"]            for r in layers]
    ip_mean   = [r["ip_mean"]                   for r in layers]
    ip_std    = [r["ip_std"]                    for r in layers]
    erank     = [r["effective_rank"]            for r in layers]
    spec_k    = [r["spectral"]["k_eigengap"]    for r in layers]
    eigengaps = [max(r["spectral"]["eigengaps"])for r in layers]

    has_sk    = [r for r in layers if "sinkhorn" in r]
    fiedler   = [r["sinkhorn"]["fiedler_mean"]                    for r in has_sk]
    sk_k      = [r["sinkhorn"]["sinkhorn_cluster_count_mean"]     for r in has_sk]
    sk_layers = [r["layer"]                                       for r in has_sk]
    attn_ent  = [r.get("attention_entropy_mean", float("nan"))    for r in has_sk]
    balance   = [r["sinkhorn"]["row_col_balance_mean"]            for r in has_sk]

    plateaus_mass = detect_plateaus(mass1,  window=2, tol=0.10)
    plateaus_rank = detect_plateaus(erank,  window=2, tol=0.05)
    plateaus_spk  = detect_plateaus(spec_k, window=2, tol=0.0)
    plateaus_fied = detect_plateaus(fiedler, window=2, tol=0.05) if fiedler else []
    merge_events  = _merge_events(spec_k)
    agreement     = _method_agreement(results)

    lines = []
    W = lines.append

    W("=" * 72)
    W("METASTABILITY PHASE 1 — LLM ANALYSIS REPORT")
    W("=" * 72)
    W("")
    W("CONTEXT FOR LLM")
    W("-" * 40)
    W("This report summarizes numerical results from Phase 1 of an empirical")
    W("study of metastable states in transformer residual streams, motivated")
    W("by Geshkovski et al. (2024) 'A Mathematical Perspective on Transformers'.")
    W("")
    W("Key theoretical predictions from the paper to check against:")
    W("  (a) Tokens cluster over layers — pairwise inner products drift toward 1")
    W("  (b) Clustering follows two timescales: fast initial grouping, slow merging")
    W("  (c) Metastable states appear as PLATEAUS in cluster count metrics")
    W("  (d) ALBERT (shared weights) should show cleaner dynamics than BERT/GPT2")
    W("  (e) High beta (sharp attention) → stronger metastability")
    W("  (f) Higher dimension d → faster convergence to single cluster (Theorem 6.1)")
    W("  (g) Interaction energy E_beta is monotone increasing along the trajectory")
    W("  (h) Sinkhorn Fiedler value should be LOW at metastable layers")
    W("  (i) Multiple clustering methods should AGREE at metastable windows")
    W("")

    W("RUN METADATA")
    W("-" * 40)
    W(f"  Model          : {model}")
    W(f"  Prompt key     : {prompt}")
    W(f"  Prompt text    : {results.get('prompt_text', PROMPTS.get(prompt, 'N/A'))}")
    W(f"  n_tokens       : {results['n_tokens']}")
    W(f"  d_model        : {results['d_model']}")
    W(f"  n_layers       : {results['n_layers']}")
    W(f"  Beta values    : {BETA_VALUES}")
    W(f"  Tokens         : {' | '.join(tokens)}")
    W("")

    W("PER-LAYER DATA TABLE")
    W("-" * 40)
    W("Columns: layer | ip_mean | ip_std | mass>0.9 | eff_rank | spec_k | "
      "max_eigengap | fiedler | sk_k | attn_entropy | energy_b1 | energy_b2 | energy_b5")
    W("")

    fiedler_by_layer = {r["layer"]: r["sinkhorn"]["fiedler_mean"]                 for r in has_sk}
    sk_k_by_layer    = {r["layer"]: r["sinkhorn"]["sinkhorn_cluster_count_mean"]  for r in has_sk}
    attn_by_layer    = {r["layer"]: r.get("attention_entropy_mean", float("nan")) for r in has_sk}

    W(f"{'L':>3}  {'ip_μ':>7}  {'ip_σ':>6}  {'m>0.9':>6}  "
      f"{'rank':>6}  {'sp_k':>5}  {'gap':>6}  "
      f"{'fied':>7}  {'sk_k':>5}  {'attn_H':>7}  "
      f"{'E_b1':>7}  {'E_b2':>7}  {'E_b5':>7}")
    W("-" * 95)

    for r in layers:
        li = r["layer"]
        eg = max(r["spectral"]["eigengaps"]) if r["spectral"]["eigengaps"] else float("nan")
        W(
            f"{li:>3}  "
            f"{r['ip_mean']:>7.4f}  "
            f"{r['ip_std']:>6.4f}  "
            f"{r['ip_mass_near_1']:>6.4f}  "
            f"{r['effective_rank']:>6.2f}  "
            f"{r['spectral']['k_eigengap']:>5}  "
            f"{eg:>6.4f}  "
            f"{fiedler_by_layer.get(li, float('nan')):>7.4f}  "
            f"{sk_k_by_layer.get(li, float('nan')):>5.1f}  "
            f"{attn_by_layer.get(li, float('nan')):>7.4f}  "
            f"{r['energies'].get(1.0, float('nan')):>7.4f}  "
            f"{r['energies'].get(2.0, float('nan')):>7.4f}  "
            f"{r['energies'].get(5.0, float('nan')):>7.4f}"
        )

    W("")
    W("TREND DESCRIPTIONS")
    W("-" * 40)
    W(f"  ip_mean        : {_trend(ip_mean)}")
    W(f"  ip_std         : {_trend(ip_std)}")
    W(f"  mass_near_1    : {_trend(mass1)}")
    W(f"  effective_rank : {_trend(erank)}")
    W(f"  spectral_k     : {_trend(spec_k)}")
    if fiedler:
        W(f"  fiedler        : {_trend(fiedler)}")
    if sk_k:
        W(f"  sinkhorn_k     : {_trend(sk_k)}")
    if attn_ent:
        W(f"  attn_entropy   : {_trend([x for x in attn_ent if not np.isnan(x)])}")
    for beta in BETA_VALUES:
        energies = [r["energies"].get(beta, float("nan")) for r in layers]
        W(f"  energy_b{beta:<4}  : {_trend(energies)}")
    W("")

    W("PLATEAU DETECTION  (candidate metastable windows)")
    W("-" * 40)
    W(f"  mass_near_1    : {_plateau_str(plateaus_mass)}")
    W(f"  effective_rank : {_plateau_str(plateaus_rank)}")
    W(f"  spectral_k     : {_plateau_str(plateaus_spk)}")
    W(f"  fiedler        : {_plateau_str(plateaus_fied)}")
    W("")

    all_plateau_layers = set()
    for group in [plateaus_mass, plateaus_rank, plateaus_spk, plateaus_fied]:
        for s, e, _ in group:
            all_plateau_layers.update(range(s, e + 1))
    if all_plateau_layers:
        W(f"  Layers appearing in ANY plateau: {sorted(all_plateau_layers)}")
        layer_count = Counter()
        for group in [plateaus_mass, plateaus_rank, plateaus_spk, plateaus_fied]:
            for s, e, _ in group:
                for l in range(s, e + 1):
                    layer_count[l] += 1
        multi = [l for l, c in sorted(layer_count.items()) if c >= 2]
        if multi:
            W(f"  Layers in 2+ plateau metrics (STRONGEST SIGNAL): {multi}")
    else:
        multi = []
    W("")

    W("MERGE EVENTS  (spectral k drops — cluster collapses)")
    W("-" * 40)
    if merge_events:
        for layer, k_before, k_after in merge_events:
            W(f"  Layer {layer:2d}: k {k_before} → {k_after}  "
              f"(ip_mean at this layer: {ip_mean[layer]:.4f}, "
              f"mass>0.9: {mass1[layer]:.4f})")
    else:
        W("  No merge events detected in spectral k")
    W("")

    W("METHOD AGREEMENT  (agglomerative / kmeans / spectral / sinkhorn)")
    W("-" * 40)
    W("Layers where all available methods agree within ±1 on cluster count:")
    if agreement:
        for layer, counts in agreement:
            W(f"  Layer {layer:2d}: counts={counts}")
    else:
        W("  No layers with full agreement across all methods")
    W("")

    W("ENERGY TRAJECTORY ANALYSIS")
    W("-" * 40)
    W("Theory predicts E_beta is monotone increasing along the trajectory.")
    W("Violations indicate deviation from idealized gradient flow.")
    for beta in BETA_VALUES:
        energies        = [r["energies"].get(beta, float("nan")) for r in layers]
        diffs           = np.diff(energies)
        n_violations    = int((diffs < -1e-6).sum())
        max_drop        = float(diffs.min()) if len(diffs) else float("nan")
        total_increase  = float(energies[-1] - energies[0])
        W(f"  beta={beta}: total_increase={total_increase:.6f}, "
          f"violations={n_violations}, max_single_drop={max_drop:.6f}")
    W("")

    W("SINKHORN / ATTENTION ANALYSIS")
    W("-" * 40)
    if has_sk:
        W("Fiedler value: second eigenvalue of doubly stochastic attention Laplacian.")
        W("Low Fiedler = attention routes tokens into near-disconnected clusters.")
        W(f"  Fiedler range      : {min(fiedler):.5f} – {max(fiedler):.5f}")
        W(f"  Lowest Fiedler at  : layer {sk_layers[int(np.argmin(fiedler))]}")
        W(f"  Highest Fiedler at : layer {sk_layers[int(np.argmax(fiedler))]}")
        W("")
        W("Row/col balance (std of column sums of raw attention):")
        W("0 = already doubly stochastic; high = far from idealized dynamics.")
        W(f"  Balance range : {min(balance):.5f} – {max(balance):.5f}")
        W(f"  Mean          : {np.mean(balance):.5f}")
        W("")
        W("Per-head Fiedler values at selected layers:")
        selected = [0, len(has_sk) // 4, len(has_sk) // 2, 3 * len(has_sk) // 4, -1]
        for idx in selected:
            r        = has_sk[idx]
            per_head = r["sinkhorn"]["fiedler_per_head"]
            W(f"  Layer {r['layer']:2d}: "
              f"min={min(per_head):.4f} max={max(per_head):.4f} "
              f"mean={np.mean(per_head):.4f}  "
              f"values={[round(v, 3) for v in per_head]}")
    else:
        W("  No Sinkhorn data available")
    W("")

    W("PCA VARIANCE TRAJECTORY")
    W("-" * 40)
    W("How much structure fits in 3 dimensions at each layer.")
    W("PC1 dominance increasing = tokens collapsing onto a line = strong clustering.")
    W(f"{'Layer':>6}  {'PC1':>7}  {'PC2':>7}  {'PC3':>7}  {'PC1+PC2+PC3':>12}")
    for r in layers:
        vr = r.get("pca_explained_variance", [])
        if len(vr) >= 3:
            W(f"{r['layer']:>6}  {vr[0]:>7.4f}  {vr[1]:>7.4f}  {vr[2]:>7.4f}  "
              f"{sum(vr[:3]):>12.4f}")
    W("")

    W("INNER PRODUCT HISTOGRAM SUMMARY")
    W("-" * 40)
    W("Distribution shape at each layer inferred from histogram.")
    W("'multimodal' = potential metastable state with distinct clusters.")
    bins        = np.linspace(-1, 1, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for r in layers:
        hist = np.array(r.get("ip_histogram", []))
        if len(hist) == 0:
            continue
        hist_norm = hist / (hist.sum() + 1e-10)
        peaks = [
            round(bin_centers[i], 2)
            for i in range(1, len(hist_norm) - 1)
            if hist_norm[i] > hist_norm[i - 1]
            and hist_norm[i] > hist_norm[i + 1]
            and hist_norm[i] > 0.02
        ]
        mass_neg  = float(hist_norm[bin_centers < -0.5].sum())
        mass_zero = float(hist_norm[np.abs(bin_centers) < 0.1].sum())
        mass_pos  = float(hist_norm[bin_centers > 0.9].sum())
        shape = (
            "multimodal" if len(peaks) > 2 else
            "bimodal"    if len(peaks) == 2 else
            "unimodal"   if len(peaks) == 1 else "flat"
        )
        W(f"  Layer {r['layer']:2d}: shape={shape:<12} "
          f"peaks~{peaks}  "
          f"mass_neg={mass_neg:.3f} mass_zero={mass_zero:.3f} mass_pos1={mass_pos:.3f}")
    W("")

    W("FLAGGED ANOMALIES AND OPEN QUESTIONS FOR LLM")
    W("-" * 40)
    e1 = [r["energies"].get(1.0, float("nan")) for r in layers]
    if np.any(np.diff(e1) < -1e-4):
        W("  [FLAG] Energy E_beta=1 is NON-MONOTONE. Theory predicts strict increase.")
        W("         This suggests V matrix has mixed eigenspectrum (some repulsive directions).")
        W("         Which layers show the drop? Does it correlate with merge events?")
    else:
        W("  [OK] Energy E_beta=1 is monotone increasing as theory predicts.")

    if len(multi) >= 3:
        W(f"  [SIGNAL] Strong multi-metric plateau at layers {multi}.")
        W("           Candidate metastable window. What do the tokens look like here?")

    if not merge_events:
        W("  [NOTE] No merge events detected. Possible explanations:")
        W("         (a) Too few layers for two-timescale dynamics")
        W("         (b) Prompt is too short — metastability needs more tokens")
        W("         (c) Spectral k threshold insensitive — check raw eigenvalue table")

    if fiedler and mass1:
        fiedler_low_layers = [sk_layers[i] for i, f in enumerate(fiedler) if f < np.median(fiedler)]
        mass_high_layers   = [i for i, m in enumerate(mass1) if m > np.median(mass1)]
        overlap = set(fiedler_low_layers) & set(mass_high_layers)
        if overlap:
            W(f"  [SIGNAL] Fiedler low AND mass>0.9 high at layers: {sorted(overlap)}")
            W("           Convergent evidence from attention routing AND activation geometry.")
        else:
            W("  [NOTE] Fiedler low layers and mass>0.9 high layers do NOT overlap.")
            W("         Attention routing and activation geometry telling different stories.")

    pc1_vals = [r["pca_explained_variance"][0]
                for r in layers if r.get("pca_explained_variance")]
    if pc1_vals and max(pc1_vals) > 0.8:
        peak_layer = int(np.argmax(pc1_vals))
        W(f"  [NOTE] PC1 explains {max(pc1_vals):.2%} of variance at layer {peak_layer}.")
        W("         Tokens nearly collinear — strong collapse signal.")

    W("")
    W("SUGGESTED ANALYSIS QUESTIONS FOR LLM")
    W("-" * 40)
    W("  1. Do plateau locations agree across metrics? Which agree and which diverge?")
    W("  2. Is energy monotone? If not, at which layers does it drop?")
    W("  3. Are merge events (spectral k drops) preceded by Fiedler value increases?")
    W("  4. Does the histogram shape match the plateau structure?")
    W("  5. Does row/col balance of raw attention correlate with Fiedler value?")
    W("  6. Which tokens end up in the same cluster at each metastable window?")
    W("  7. Does two-timescale structure appear (fast clustering, slow merging)?")
    W("")
    W("=" * 72)
    W("END OF REPORT")
    W("=" * 72)

    out_path = save_dir / "llm_report.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  LLM report: {out_path}")


# ---------------------------------------------------------------------------
# Cross-run comparative report
# ---------------------------------------------------------------------------

def generate_cross_run_report(all_results: list, save_dir: Path):
    """
    Write a comparative plain-text report across all model/prompt combinations.
    Intended for LLM analysis of trends across conditions.
    """
    lines = []
    W = lines.append

    W("=" * 72)
    W("CROSS-RUN COMPARATIVE REPORT — LLM ANALYSIS")
    W("=" * 72)
    W("")
    W("This report compares metastability signals across all model/prompt")
    W("combinations run in this session.")
    W("")
    W("KEY COMPARISON QUESTIONS:")
    W("  - Does ALBERT show stronger/cleaner metastability than BERT or GPT2?")
    W("  - Do longer prompts produce more metastable windows?")
    W("  - Do repeated tokens cluster immediately or evolve?")
    W("  - Is energy monotonicity model-dependent?")
    W("  - Do plateau locations agree across prompts for the same model?")
    W("")

    W("SUMMARY TABLE")
    W("-" * 40)
    W(f"{'Model':<25} {'Prompt':<22} {'Tokens':>6} {'Layers':>6} "
      f"{'MaxMass':>8} {'MinRank':>8} {'nPlateaus':>10} {'nMerges':>8} "
      f"{'EnergyOK':>9}")
    W("-" * 100)

    for r in all_results:
        layers_  = r["layers"]
        mass1_   = [l["ip_mass_near_1"]        for l in layers_]
        erank_   = [l["effective_rank"]         for l in layers_]
        spec_k_  = [l["spectral"]["k_eigengap"] for l in layers_]
        e1_      = [l["energies"].get(1.0, float("nan")) for l in layers_]
        plateaus = detect_plateaus(mass1_, window=2, tol=0.10)
        merges   = _merge_events(spec_k_)
        energy_ok = "YES" if np.all(np.diff(e1_) >= -1e-4) else "NO"
        W(f"{r['model']:<25} {r['prompt']:<22} {r['n_tokens']:>6} "
          f"{r['n_layers']:>6} {max(mass1_):>8.4f} {min(erank_):>8.2f} "
          f"{len(plateaus):>10} {len(merges):>8} {energy_ok:>9}")

    W("")
    W("PLATEAU LOCATIONS BY RUN")
    W("-" * 40)
    for r in all_results:
        mass1_  = [l["ip_mass_near_1"]        for l in r["layers"]]
        spec_k_ = [l["spectral"]["k_eigengap"] for l in r["layers"]]
        p_mass  = detect_plateaus(mass1_,  window=2, tol=0.10)
        p_spk   = detect_plateaus(spec_k_, window=2, tol=0.0)
        W(f"  {r['model']} | {r['prompt']}")
        W(f"    mass plateaus   : {_plateau_str(p_mass)}")
        W(f"    spectral k plat : {_plateau_str(p_spk)}")

    W("")
    W("MERGE EVENT LOCATIONS BY RUN")
    W("-" * 40)
    for r in all_results:
        spec_k_ = [l["spectral"]["k_eigengap"] for l in r["layers"]]
        merges  = _merge_events(spec_k_)
        W(f"  {r['model']} | {r['prompt']}")
        if merges:
            for layer, kb, ka in merges:
                W(f"    Layer {layer:2d}: k {kb}→{ka}")
        else:
            W("    No merge events")

    W("")
    W("ENERGY MONOTONICITY BY RUN")
    W("-" * 40)
    for r in all_results:
        for beta in BETA_VALUES:
            energies = [l["energies"].get(beta, float("nan")) for l in r["layers"]]
            diffs    = np.diff(energies)
            n_viol   = int((diffs < -1e-6).sum())
            W(f"  {r['model']:<25} | {r['prompt']:<22} | "
              f"beta={beta}: violations={n_viol}  "
              f"total_delta={energies[-1]-energies[0]:.5f}")

    W("")
    W("FLAGGED CROSS-RUN PATTERNS")
    W("-" * 40)

    albert_results = [r for r in all_results if "albert" in r["model"]]
    other_results  = [r for r in all_results if "albert" not in r["model"]]
    if albert_results and other_results:
        albert_max = max(
            max(l["ip_mass_near_1"] for l in r["layers"])
            for r in albert_results
        )
        other_max = max(
            max(l["ip_mass_near_1"] for l in r["layers"])
            for r in other_results
        )
        if albert_max > other_max:
            W("  [SIGNAL] ALBERT reaches higher mass>0.9 than other models.")
            W("           Consistent with theory: shared weights = cleaner dynamics.")
        else:
            W("  [NOTE] ALBERT does NOT show stronger clustering than other models.")

    by_model = {}
    for r in all_results:
        by_model.setdefault(r["model"], []).append(r)
    for model, runs in by_model.items():
        if len(runs) > 1:
            runs_by_length = sorted(runs, key=lambda x: x["n_tokens"])
            short_max = max(l["ip_mass_near_1"] for l in runs_by_length[0]["layers"])
            long_max  = max(l["ip_mass_near_1"] for l in runs_by_length[-1]["layers"])
            direction = "increases" if long_max > short_max else "decreases"
            W(f"  {model}: clustering {direction} with prompt length "
              f"({runs_by_length[0]['n_tokens']} tokens: {short_max:.4f} vs "
              f"{runs_by_length[-1]['n_tokens']} tokens: {long_max:.4f})")

    W("")
    W("=" * 72)
    W("END OF CROSS-RUN REPORT")
    W("=" * 72)

    out_path = save_dir / "llm_cross_run_report.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Cross-run LLM report: {out_path}")
