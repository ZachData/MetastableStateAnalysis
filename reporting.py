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


def _nn_cycles(nn_indices: list, tokens: list) -> list:
    """
    Find all cycles in the NN functional graph.

    Each node has exactly one outgoing edge (its nearest neighbour).  The
    minimal closed sets — sets S where ∀ i∈S: nn[i]∈S — are exactly the
    cycles.  Walking each node's chain until revisiting a node surfaces them.

    Parameters
    ----------
    nn_indices : (n_tokens,) list of int  —  nn_indices[i] = NN of token i
    tokens     : list of str             —  decoded token strings

    Returns
    -------
    list of (cluster, is_semantic) tuples where:
      cluster    : list of token strings (sorted by index)
      is_semantic: True if at least two members have distinct string values
                   (i.e. not a same-type duplicate pair)
    Clusters are returned in ascending order of their smallest member index.
    """
    n         = len(nn_indices)
    visited   = [False] * n
    in_cycle  = [False] * n
    cycles    = []

    for start in range(n):
        if visited[start]:
            continue
        path = []
        pos  = {}                     # node → position in current path
        curr = start
        while curr not in pos and not visited[curr]:
            pos[curr] = len(path)
            path.append(curr)
            curr = nn_indices[curr]

        if curr in pos and not in_cycle[curr]:
            # A new cycle starting at pos[curr] in path
            cycle = path[pos[curr]:]
            cycles.append(sorted(cycle))
            for node in cycle:
                in_cycle[node] = True

        for node in path:
            visited[node] = True

    # Sort cycles by smallest member index
    cycles.sort(key=lambda c: c[0])

    result = []
    for cycle in cycles:
        token_strs = [tokens[i] for i in cycle]
        # Semantic = at least two members differ in string value
        is_semantic = len(set(token_strs)) > 1
        result.append((token_strs, is_semantic))
    return result


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

    hdb_k_vals = [
        r["clustering"].get("hdbscan", {}).get("n_clusters", float("nan"))
        for r in results["layers"]
    ]
    has_hdbscan = any(not np.isnan(v) for v in hdb_k_vals)

    metrics = [
        ("Mass-near-1",   [r["ip_mass_near_1"]            for r in results["layers"]], 0.10),
        ("Effective rank",[r["effective_rank"]             for r in results["layers"]], 0.05),
        ("Spectral k",    [r["spectral"]["k_eigengap"]     for r in results["layers"]], 0.5),
    ]
    if has_hdbscan:
        metrics.append(("HDBSCAN k", hdb_k_vals, 0.5))
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

    nn_stab_pairs = [(r["layer"], r["nn_stability"])
                     for r in results["layers"]
                     if r.get("nn_stability") is not None
                     and r["effective_rank"] >= 2.0]
    if nn_stab_pairs:
        nn_series = [v for _, v in nn_stab_pairs]
        nn_plat   = detect_plateaus(nn_series, window=2, tol=0.02)
        n_suppressed = sum(1 for r in results["layers"]
                           if r.get("nn_stability") is not None
                           and r["effective_rank"] < 2.0)
        suffix = f" ({n_suppressed} degenerate layers excluded)" if n_suppressed else ""
        print(f"\n  NN Stability plateaus (metastable neighbourhood lock-in){suffix}:")
        if nn_plat:
            for s, e, val in nn_plat:
                layer_s = nn_stab_pairs[s][0]
                layer_e = nn_stab_pairs[e][0]
                print(f"    Layers {layer_s}–{layer_e}  (mean={val:.4f})")
        else:
            print(f"    No plateaus detected")


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
          f"{'eff_rank':>9}  {'spec_k':>7}  {'fiedler':>8}  {'sk_k':>5}  {'nn_stab':>8}")
    print(f"{'─'*80}")

    for r in layers:
        sk       = r.get("sinkhorn", {})
        nn_stab  = r.get("nn_stability")
        nn_str   = f"{nn_stab:>8.4f}" if nn_stab is not None else f"{'n/a':>8}"
        print(
            f"{r['layer']:>6}  "
            f"{r['ip_mean']:>8.4f}  "
            f"{r['ip_mass_near_1']:>9.4f}  "
            f"{r['effective_rank']:>9.2f}  "
            f"{r['spectral']['k_eigengap']:>7}  "
            f"{sk.get('fiedler_mean', float('nan')):>8.4f}  "
            f"{sk.get('sinkhorn_cluster_count_mean', float('nan')):>5.1f}  "
            f"{nn_str}"
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

    # NN stability plateaus — suppress degenerate layers (eff_rank < 2)
    nn_stab_vals    = [r.get("nn_stability") for r in layers]
    nn_stab_defined = [
        (i, v) for i, v in enumerate(nn_stab_vals)
        if v is not None and layers[i]["effective_rank"] >= 2.0
    ]
    if nn_stab_defined:
        nn_stab_series = [v for _, v in nn_stab_defined]
        nn_plat = detect_plateaus(nn_stab_series, window=2, tol=0.02)
        n_sup = sum(1 for r in layers
                    if r.get("nn_stability") is not None and r["effective_rank"] < 2.0)
        suffix = f" ({n_sup} degenerate layers excluded)" if n_sup else ""
        print(f"\n  NN Stability (high = tokens locked in stable neighbourhoods){suffix}:")
        if nn_plat:
            for s, e, v in nn_plat:
                layer_s = nn_stab_defined[s][0]
                layer_e = nn_stab_defined[e][0]
                print(f"    Layers {layer_s:2d}–{layer_e:2d}  mean={v:.4f}  width={e-s+1}")
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

    hdb_k       = [r["clustering"].get("hdbscan", {}).get("n_clusters", float("nan"))
                   for r in layers]
    has_hdbscan = any(not np.isnan(v) for v in hdb_k)

    plateaus_mass = detect_plateaus(mass1,  window=2, tol=0.10)
    plateaus_rank = detect_plateaus(erank,  window=2, tol=0.05)
    plateaus_spk  = detect_plateaus(spec_k, window=2, tol=0.5)
    plateaus_hdb  = detect_plateaus([v for v in hdb_k if not np.isnan(v)], window=2, tol=0.5) if has_hdbscan else []
    plateaus_fied = detect_plateaus(fiedler, window=2, tol=0.05) if fiedler else []

    # NN stability series: skip layer 0 (None) AND degenerate layers (eff_rank < 2).
    # When all tokens are a near-point-mass (rank ≈ 1–2), NN assignment is determined
    # by floating-point noise — the stability signal is meaningless there.
    NN_DEGENERATE_RANK = 2.0
    nn_stab_pairs   = [
        (r["layer"], r["nn_stability"])
        for r in layers
        if r.get("nn_stability") is not None
        and r["effective_rank"] >= NN_DEGENERATE_RANK
    ]
    nn_stab_series  = [v for _, v in nn_stab_pairs]
    plateaus_nn     = detect_plateaus(nn_stab_series, window=2, tol=0.02) if nn_stab_series else []
    # Plateau layer sets for TOKEN CLUSTER MEMBERSHIP (use NN stability primary signal)
    nn_plateau_layers = set()
    for s, e, _ in plateaus_nn:
        for idx in range(s, e + 1):
            nn_plateau_layers.add(nn_stab_pairs[idx][0])
    # Count of layers suppressed as degenerate (for report annotation)
    nn_degenerate_count = sum(
        1 for r in layers
        if r.get("nn_stability") is not None
        and r["effective_rank"] < NN_DEGENERATE_RANK
    )

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
    W("Columns: layer | ip_mean | ip_std | mass>0.9 | eff_rank | spec_k | spec_k2 | "
      "max_eigengap | fiedler | sk_k | attn_entropy | energy_b1 | energy_b2 | energy_b5")
    W("")

    fiedler_by_layer = {r["layer"]: r["sinkhorn"]["fiedler_mean"]                 for r in has_sk}
    sk_k_by_layer    = {r["layer"]: r["sinkhorn"]["sinkhorn_cluster_count_mean"]  for r in has_sk}
    attn_by_layer    = {r["layer"]: r.get("attention_entropy_mean", float("nan")) for r in has_sk}
    mid_thresh       = float(DISTANCE_THRESHOLDS[len(DISTANCE_THRESHOLDS) // 2])

    W("Columns: layer | ip_mean | ip_std | mass>0.9 | eff_rank | sp_k | k2 | agg_k | km_k | "
      "hdb_k | gap | fiedler | sk_k | attn_H | E_b1 | E_b2 | E_b5 | nn_stab")
    W(f"{'L':>3}  {'ip_μ':>7}  {'ip_σ':>6}  {'m>0.9':>6}  "
      f"{'rank':>7}  {'sp_k':>5}  {'k2':>4}  {'agg_k':>6}  {'km_k':>5}  {'hdb_k':>6}  {'gap':>6}  "
      f"{'fied':>7}  {'sk_k':>5}  {'attn_H':>7}  "
      f"{'E_b1':>7}  {'E_b2':>7}  {'E_b5':>7}  {'nn_stab':>8}")
    W("-" * 132)

    for r in layers:
        li      = r["layer"]
        eg      = max(r["spectral"]["eigengaps"]) if r["spectral"]["eigengaps"] else float("nan")
        hdb_n   = r["clustering"].get("hdbscan", {}).get("n_clusters", float("nan"))
        k2      = r["spectral"].get("k_second_gap", float("nan"))
        agg_k   = r["clustering"]["agglomerative"].get(mid_thresh, float("nan"))
        km_k    = r["clustering"]["kmeans"]["best_k"]
        nn_stab = r.get("nn_stability")
        nn_str  = f"{nn_stab:>8.4f}" if nn_stab is not None else f"{'n/a':>8}"
        W(
            f"{li:>3}  "
            f"{r['ip_mean']:>7.4f}  "
            f"{r['ip_std']:>6.4f}  "
            f"{r['ip_mass_near_1']:>6.4f}  "
            f"{r['effective_rank']:>7.2f}  "
            f"{r['spectral']['k_eigengap']:>5}  "
            f"{k2 if not (isinstance(k2, float) and np.isnan(k2)) else 'n/a':>4}  "
            f"{int(agg_k) if not (isinstance(agg_k, float) and np.isnan(agg_k)) else 'n/a':>6}  "
            f"{km_k:>5}  "
            f"{hdb_n if not (isinstance(hdb_n, float) and np.isnan(hdb_n)) else 'n/a':>6}  "
            f"{eg:>6.4f}  "
            f"{fiedler_by_layer.get(li, float('nan')):>7.4f}  "
            f"{sk_k_by_layer.get(li, float('nan')):>5.1f}  "
            f"{attn_by_layer.get(li, float('nan')):>7.4f}  "
            f"{r['energies'].get(1.0, float('nan')):>7.4f}  "
            f"{r['energies'].get(2.0, float('nan')):>7.4f}  "
            f"{r['energies'].get(5.0, float('nan')):>7.4f}  "
            f"{nn_str}"
        )

    W("")
    W("TREND DESCRIPTIONS  (fiedler and sinkhorn_k only — all others readable from table)")
    W("-" * 40)
    if fiedler:
        W(f"  fiedler        : {_trend(fiedler)}")
    if sk_k:
        W(f"  sinkhorn_k     : {_trend(sk_k)}")
    if attn_ent:
        W(f"  attn_entropy   : {_trend([x for x in attn_ent if not np.isnan(x)])}")
    W("")

    W("PLATEAU DETECTION  (candidate metastable windows)")
    W("-" * 40)
    W(f"  mass_near_1    : {_plateau_str(plateaus_mass)}")
    W(f"  effective_rank : {_plateau_str(plateaus_rank)}")
    W(f"  spectral_k     : {_plateau_str(plateaus_spk)}")
    if has_hdbscan:
        W(f"  hdbscan_k      : {_plateau_str(plateaus_hdb)}")
    W(f"  fiedler        : {_plateau_str(plateaus_fied)}")
    if nn_stab_pairs:
        # Remap plateau indices (into nn_stab_series) back to actual layer numbers
        nn_plat_str_parts = []
        for s, e, v in plateaus_nn:
            layer_s = nn_stab_pairs[s][0]
            layer_e = nn_stab_pairs[e][0]
            nn_plat_str_parts.append(f"layers {layer_s}-{layer_e} (width={e-s+1}, mean={v:.4f})")
        nn_plat_str = "; ".join(nn_plat_str_parts) if nn_plat_str_parts else "none detected"
        degenerate_note = (f"  [{nn_degenerate_count} degenerate layers suppressed "
                           f"(eff_rank < {NN_DEGENERATE_RANK})]") if nn_degenerate_count else ""
        W(f"  nn_stability   : {nn_plat_str}{degenerate_note}")
    W("")

    # Compute multi (layers in 2+ metrics) for use in flagged anomalies — not printed here
    layer_count = Counter()
    for group in [plateaus_mass, plateaus_rank, plateaus_spk, plateaus_hdb, plateaus_fied]:
        for s, e, _ in group:
            for l in range(s, e + 1):
                layer_count[l] += 1
    multi = [l for l, c in sorted(layer_count.items()) if c >= 2]

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
        energies       = [r["energies"].get(beta, float("nan")) for r in layers]
        diffs          = np.diff(energies)
        viol_layers    = [i + 1 for i, d in enumerate(diffs) if d < -1e-6]
        n_violations   = len(viol_layers)
        max_drop       = float(diffs.min()) if len(diffs) else float("nan")
        total_increase = float(energies[-1] - energies[0])
        viol_str       = str(viol_layers) if viol_layers else "none"
        W(f"  beta={beta}: total_increase={total_increase:.6f}, "
          f"violations={n_violations}, max_single_drop={max_drop:.6f}, "
          f"violation_layers={viol_str}")
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

    W("SPECTRAL EIGENVALUE TABLE")
    W("-" * 40)
    W("Raw Laplacian eigenvalues (λ₁ … λ_N) and eigengaps (Δλ) per layer.")
    W("A genuine eigengap at position k appears as a large Δλ[k] relative to its neighbours.")
    W("A smooth, monotone decay means k is an artifact of the threshold — no real structure.")
    W("")
    W("NOTE: λ₁=0 is always the trivial zero mode of the Laplacian.")
    W("  k_eigengap = dominant gap including Δλ₁  (k=1 throughout = tokens converging to 1 cluster)")
    W("  k_second_gap = dominant gap SKIPPING Δλ₁ (surfaces secondary structure hidden by zero mode)")
    W("")

    # Flag if k=1 throughout
    all_k1 = all(r["spectral"]["k_eigengap"] == 1 for r in layers)
    if all_k1:
        k2_vals = [r["spectral"].get("k_second_gap", 1) for r in layers]
        k2_non1 = [li for li, k2 in enumerate(k2_vals) if k2 > 1]
        W("  [NOTE] k_eigengap=1 for ALL layers. The trivial zero mode dominates throughout.")
        if k2_non1:
            W(f"         k_second_gap finds secondary structure at layers: {k2_non1}")
            W("         These layers have genuine non-trivial cluster geometry.")
        else:
            W("         k_second_gap also=1 throughout: spectrum decays smoothly, no secondary structure.")
        W("")
    n_evs = max(len(r["spectral"]["eigenvalues"]) for r in layers)
    ev_header = "  " + f"{'L':>3}  {'k':>3}  " + "  ".join(
        f"λ{i+1:>2}" for i in range(n_evs)
    ) + "   │  " + "  ".join(
        f"Δ{i+1:>2}" for i in range(n_evs - 1)
    )
    W(ev_header)
    W("  " + "-" * (len(ev_header) - 2))
    for r in layers:
        evs  = r["spectral"]["eigenvalues"]
        gaps = r["spectral"]["eigengaps"]
        k    = r["spectral"]["k_eigengap"]
        # Pad to n_evs so columns align regardless of per-layer eigenvalue count
        evs_padded  = list(evs)  + [float("nan")] * (n_evs - len(evs))
        gaps_padded = list(gaps) + [float("nan")] * (n_evs - 1 - len(gaps))
        ev_str  = "  ".join(f"{v:5.3f}" if not np.isnan(v) else "  ---" for v in evs_padded)
        gap_str = "  ".join(f"{v:5.3f}" if not np.isnan(v) else "  ---" for v in gaps_padded)
        W(f"  {r['layer']:>3}  {k:>3}  {ev_str}   │  {gap_str}")
    W("")
    W("Dominant gap per layer (the gap chosen to set k):")
    CONVERGED_THRESH = 1e-4   # all gaps below this = fully collapsed, ratios are float noise
    for r in layers:
        gaps = r["spectral"]["eigengaps"]
        k    = r["spectral"]["k_eigengap"]
        k2   = r["spectral"].get("k_second_gap", "n/a")
        sgr  = r["spectral"].get("second_gap_ratio", float("nan"))
        if gaps:
            dom_gap   = max(gaps)
            dom_idx   = int(np.argmax(gaps))
            runner_up = sorted(gaps)[-2] if len(gaps) > 1 else 0.0
            if dom_gap < CONVERGED_THRESH:
                # All gaps are floating-point noise — spectrum fully converged
                W(f"  Layer {r['layer']:2d}: k={k}  k2=n/a  "
                  f"dom_gap={dom_gap:.2e}@pos{dom_idx+1}  [CONVERGED — all gaps are noise]")
            else:
                ratio = dom_gap / (runner_up + 1e-10)
                W(f"  Layer {r['layer']:2d}: k={k}  k2={k2}  "
                  f"dom_gap={dom_gap:.4f}@pos{dom_idx+1}  ratio={ratio:.1f}x  "
                  f"second_gap_ratio={sgr:.1f}x")
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

    W("")

    # ------------------------------------------------------------------
    # TOKEN CLUSTER MEMBERSHIP
    # Printed only for layers that fall inside an NN-stability plateau.
    # At each such layer, finds stable token clusters: maximal sets where
    # every member's nearest neighbour points within the set.  In the NN
    # functional graph (each node has out-degree 1) these are precisely
    # the cycles.  Mutual pairs (i→j, j→i) are the simplest case.
    # ------------------------------------------------------------------
    W("TOKEN CLUSTER MEMBERSHIP")
    W("-" * 40)
    W("Stable clusters at NN-stability plateau layers.")
    W("A cluster is a maximal set S where ∀ i ∈ S: NN(i) ∈ S (cycles in the NN graph).")
    W("Mutual pairs and longer cycles both qualify.")
    W("SEMANTIC clusters: members have distinct token strings (structurally parallel tokens).")
    W("DUPLICATE clusters: all members are the same token string (positional copies).")
    W("")

    if not nn_plateau_layers:
        W("  No NN-stability plateaus detected — no stable cluster windows to report.")
    else:
        W(f"  Plateau layers: {sorted(nn_plateau_layers)}")
        W("")
        for r in layers:
            li = r["layer"]
            if li not in nn_plateau_layers:
                continue
            nn_idx   = r.get("nn_indices", [])
            if not nn_idx:
                continue
            all_clusters = _nn_cycles(nn_idx, tokens)
            stab_val     = r.get("nn_stability")
            stab_str     = f"{stab_val:.4f}" if stab_val is not None else "n/a"

            semantic  = [(c, s) for c, s in all_clusters if s]
            duplicate = [(c, s) for c, s in all_clusters if not s]

            W(f"  Layer {li:2d}  (nn_stability={stab_str}, "
              f"semantic={len(semantic)}, duplicate={len(duplicate)}, "
              f"total={len(all_clusters)})")

            if semantic:
                W(f"    --- SEMANTIC PAIRS / CYCLES ({len(semantic)}) ---")
                for ci, (cluster, _) in enumerate(semantic, 1):
                    W(f"    {ci:3d}. {' | '.join(cluster)}")
            else:
                W("    No semantic clusters found at this layer.")

            if duplicate:
                W(f"    --- DUPLICATE PAIRS ({len(duplicate)}) ---"
                  f"  (same token string, positional copies)")
                # Compact display: group identical tokens, show count
                dup_counter: Counter = Counter(c[0] for c, _ in duplicate)
                dup_str = "  ".join(
                    f"{tok!r}×{cnt}" if cnt > 1 else repr(tok)
                    for tok, cnt in dup_counter.most_common()
                )
                W(f"    {dup_str}")

            W("")

    W("FLAGGED ANOMALIES")
    W("-" * 40)

    # Energy monotonicity with specific layers
    e1 = [r["energies"].get(1.0, float("nan")) for r in layers]
    e1_viol = [i + 1 for i, d in enumerate(np.diff(e1)) if d < -1e-4]
    if e1_viol:
        W(f"  [FLAG] Energy E_beta=1 NON-MONOTONE at layers: {e1_viol}")
        W("         Suggests V matrix has repulsive directions not in gradient-flow model.")
        # Check overlap with merge events
        merge_layers = [layer for layer, _, _ in merge_events]
        overlap = set(e1_viol) & set(merge_layers)
        if overlap:
            W(f"         Energy drops COINCIDE with merge events at layers: {sorted(overlap)}")
        else:
            W("         Energy drops do NOT coincide with merge events.")
    else:
        W("  [OK] Energy E_beta=1 is monotone increasing as theory predicts.")

    # Cross-metric reset event detection.
    # A reset is a layer where ip_mean, effective_rank, and nn_stability all
    # reverse direction simultaneously — partial de-clustering.  Signature:
    #   ip_mean drops (tokens spread out), rank rises (more dimensions used),
    #   sinkhorn_k spikes (more clusters detected).
    # Thresholds are empirical: ip_mean drop > 0.05, rank rise > 5.
    reset_layers = []
    for i in range(1, len(layers)):
        prev, curr = layers[i - 1], layers[i]
        ip_drop   = prev["ip_mean"] - curr["ip_mean"]
        rank_rise = curr["effective_rank"] - prev["effective_rank"]
        sk_prev   = prev.get("sinkhorn", {}).get("sinkhorn_cluster_count_mean", None)
        sk_curr   = curr.get("sinkhorn", {}).get("sinkhorn_cluster_count_mean", None)
        sk_spike  = (sk_curr - sk_prev) if (sk_prev is not None and sk_curr is not None) else 0
        if ip_drop > 0.05 and rank_rise > 5:
            nn_drop = ""
            nn_prev = prev.get("nn_stability")
            nn_curr = curr.get("nn_stability")
            if nn_prev is not None and nn_curr is not None:
                nn_drop = f", nn_stab {nn_prev:.3f}→{nn_curr:.3f}"
            sk_note = f", sk_count +{sk_spike:.0f}" if sk_spike > 5 else ""
            reset_layers.append(
                f"  [RESET] Layer {curr['layer']}: ip_mean {prev['ip_mean']:.3f}→"
                f"{curr['ip_mean']:.3f} (Δ={-ip_drop:.3f}), "
                f"eff_rank {prev['effective_rank']:.1f}→{curr['effective_rank']:.1f}"
                f"{sk_note}{nn_drop}."
            )
    if reset_layers:
        W("")
        for line in reset_layers:
            W(line)
        W("         Partial de-clustering: tokens spread back out after prior convergence.")
        W("         Cross-check with energy violations at the same layers.")

    # Multi-metric plateau summary — only meaningful if the window is
    # narrower than half the run; a union spanning the whole run is vacuous.
    n_layers_total = results["n_layers"]
    if len(multi) >= 3 and len(multi) < 0.5 * n_layers_total:
        runs = []
        start = multi[0]
        for i in range(1, len(multi)):
            if multi[i] != multi[i-1] + 1:
                runs.append((start, multi[i-1]))
                start = multi[i]
        runs.append((start, multi[-1]))
        run_str = ", ".join(f"layers {s}–{e}" for s, e in runs)
        W(f"  [SIGNAL] Multi-metric plateau spans: {run_str}  ({len(multi)} layers total).")
        W("           Candidate metastable windows — cross-check with token PCA.")
    elif len(multi) >= 0.5 * n_layers_total:
        W(f"  [NOTE] Multi-metric plateau union spans {len(multi)}/{n_layers_total} layers "
          f"(≥50% of run) — individual metrics plateau at different windows; "
          f"no single coherent metastable window identified.")

    # Merge event context
    if not merge_events:
        W("  [NOTE] No merge events (spectral k drops) detected.")
        W("         Possible causes: too few layers, too few tokens, or spectral k stuck at 1.")
    else:
        for layer, k_before, k_after in merge_events:
            fied_at = fiedler_by_layer.get(layer, float("nan"))
            fied_prev = fiedler_by_layer.get(layer - 1, float("nan"))
            if not np.isnan(fied_at) and not np.isnan(fied_prev):
                fied_dir = "↑ ROSE" if fied_at > fied_prev else "↓ fell"
                W(f"  [MERGE] Layer {layer}: k {k_before}→{k_after}. "
                  f"Fiedler {fied_prev:.3f}→{fied_at:.3f} ({fied_dir} before merge).")
            else:
                W(f"  [MERGE] Layer {layer}: k {k_before}→{k_after}.")

    # Fiedler 0.5 crossing
    if fiedler:
        crossings = [sk_layers[i] for i in range(1, len(fiedler))
                     if (fiedler[i-1] < 0.5) != (fiedler[i] < 0.5)]
        if crossings:
            W(f"  [TRANSITION] Fiedler crosses 0.5 at layer(s): {crossings}")
            W("               Below 0.5 = cluster-separated attention; above = tokens mixing.")
        fiedler_mass_overlap = set(
            sk_layers[i] for i, f in enumerate(fiedler) if f < np.median(fiedler)
        ) & set(i for i, m in enumerate(mass1) if m > np.median(mass1))
        if fiedler_mass_overlap:
            W(f"  [SIGNAL] Fiedler low AND mass>0.9 high at: {sorted(fiedler_mass_overlap)}")
            W("           Convergent evidence from attention routing AND geometry.")
        else:
            W("  [NOTE] Fiedler-low layers and mass>0.9-high layers do not overlap.")
            W("         Attention routing and activation geometry are telling different stories.")

    # Effective rank collapse
    rank_low = [r["layer"] for r in layers if r["effective_rank"] < 10]
    if rank_low:
        W(f"  [COLLAPSE] Effective rank drops below 10 at layer {rank_low[0]} "
          f"(first of {len(rank_low)} layers).")
        W("             Tokens nearly collinear from this point — geometry is degenerate.")

    # PCA variance jump
    pca_totals = [sum(r["pca_explained_variance"][:3])
                  for r in layers if len(r.get("pca_explained_variance", [])) >= 3]
    if len(pca_totals) > 1:
        pca_steps = np.diff(pca_totals)
        max_step_idx = int(np.argmax(pca_steps))
        max_step = pca_steps[max_step_idx]
        if max_step > 0.05:
            W(f"  [PCA JUMP] Largest single-step increase in PC1+PC2+PC3: "
              f"+{max_step:.3f} at layer {max_step_idx}→{max_step_idx+1} "
              f"({pca_totals[max_step_idx]:.3f}→{pca_totals[max_step_idx+1]:.3f}).")
            W("             Rapid dimensional collapse — strong clustering event.")

    # PC1 dominance
    pc1_vals = [r["pca_explained_variance"][0]
                for r in layers if r.get("pca_explained_variance")]
    if pc1_vals and max(pc1_vals) > 0.8:
        peak_layer = int(np.argmax(pc1_vals))
        W(f"  [NOTE] PC1 explains {max(pc1_vals):.2%} of variance at layer {peak_layer}.")
        W("         Tokens nearly collinear — strong collapse signal.")

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
        p_spk   = detect_plateaus(spec_k_, window=2, tol=0.5)
        hdb_k_  = [l["clustering"].get("hdbscan", {}).get("n_clusters", float("nan"))
                   for l in r["layers"]]
        has_hdb = any(not np.isnan(v) for v in hdb_k_)
        p_hdb   = detect_plateaus([v for v in hdb_k_ if not np.isnan(v)], window=2, tol=0.5) if has_hdb else []
        W(f"  {r['model']} | {r['prompt']}")
        W(f"    mass plateaus   : {_plateau_str(p_mass)}")
        W(f"    spectral k plat : {_plateau_str(p_spk)}")
        if has_hdb:
            W(f"    hdbscan k plat  : {_plateau_str(p_hdb)}")

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
