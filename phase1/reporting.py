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

from scipy.stats import spearmanr

from core.config import BETA_VALUES, DISTANCE_THRESHOLDS, PROMPTS


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

    KMeans is excluded from the comparison when its best silhouette score
    is below 0.1 — at that point K_RANGE is bounded below at 2, so
    best_k=2 is a floor artefact rather than a genuine signal.

    Returns list of (layer, [counts]).
    """
    mid_thresh       = float(DISTANCE_THRESHOLDS[len(DISTANCE_THRESHOLDS) // 2])
    agreement_layers = []
    for r in results["layers"]:
        agg_k  = r["clustering"]["agglomerative"].get(mid_thresh)
        km_k   = r["clustering"]["kmeans"]["best_k"]
        km_sil = r["clustering"]["kmeans"]["best_silhouette"]
        sp_k   = r["spectral"]["k_eigengap"]
        sk_k   = round(r.get("sinkhorn", {}).get("sinkhorn_cluster_count_mean", -99))
        # Only include KMeans when silhouette is meaningful AND geometry is
        # non-degenerate. In the collapsed regime (effective_rank < 10) all
        # tokens are near-collinear, so any k≥2 partition scores a spurious
        # silhouette of ~0.1–0.3 purely from the geometry — not real structure.
        if km_sil >= 0.1 and r["effective_rank"] >= 10:
            counts = [k for k in [agg_k, km_k, sp_k, sk_k] if k and k > 0]
        else:
            counts = [k for k in [agg_k, sp_k, sk_k] if k and k > 0]
        if counts and (max(counts) - min(counts)) <= 1:
            agreement_layers.append((r["layer"], counts))
    return agreement_layers


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------

def _per_head_fiedler_profile(
    results: dict,
    active_rank_threshold: float = 10.0,
) -> list:
    """
    Compute per-head Fiedler statistics restricted to the *active phase*.

    The active phase is defined as layers where effective_rank >=
    active_rank_threshold.  Once tokens have collapsed to a near-point-mass
    (rank < threshold) every head trivially saturates to Fiedler ≈ 1.0 —
    there is only one cluster, so the doubly stochastic matrix is nearly
    uniform and the Laplacian has no gap.  Including those layers pulls every
    head's mean toward 1.0 regardless of early behaviour, making the
    CLUSTER/MIXED/MIXING classification meaningless.

    For each attention head h, collect its Fiedler value at every active-phase
    layer that has Sinkhorn data, then return a list of dicts with:
      head               : int   — head index
      mean               : float — mean Fiedler over active-phase layers
      std                : float — std  Fiedler over active-phase layers
      min_layer          : int   — layer index at which Fiedler is minimised
      classification     : str   — 'CLUSTER' (<0.3), 'MIXED' (0.3–0.7), 'MIXING' (>0.7)
      values             : list  — per-layer Fiedler values (active phase only)
      n_active_layers    : int   — number of layers included
      n_collapsed_layers : int   — number of Sinkhorn layers excluded as collapsed

    Returns an empty list if no Sinkhorn data is present.
    """
    all_sk  = [r for r in results["layers"] if "sinkhorn" in r]
    if not all_sk:
        return []

    # Restrict to active phase
    sk_layers = [r for r in all_sk if r["effective_rank"] >= active_rank_threshold]
    n_collapsed = len(all_sk) - len(sk_layers)

    # Fall back to all Sinkhorn layers if the threshold filters everything
    # (e.g. very shallow models where rank never reaches the threshold)
    if not sk_layers:
        sk_layers = all_sk
        n_collapsed = 0

    n_heads = len(sk_layers[0]["sinkhorn"].get("fiedler_per_head", []))
    if n_heads == 0:
        return []

    profiles = []
    for h in range(n_heads):
        vals      = [r["sinkhorn"]["fiedler_per_head"][h] for r in sk_layers]
        layer_ids = [r["layer"]                            for r in sk_layers]
        mean_f    = float(np.mean(vals))
        std_f     = float(np.std(vals))
        min_idx   = int(np.argmin(vals))
        min_layer = layer_ids[min_idx]

        if mean_f < 0.3:
            cls = "CLUSTER"
        elif mean_f > 0.7:
            cls = "MIXING"
        else:
            cls = "MIXED"

        profiles.append({
            "head":               h,
            "mean":               mean_f,
            "std":                std_f,
            "min_layer":          min_layer,
            "classification":     cls,
            "values":             vals,
            "n_active_layers":    len(sk_layers),
            "n_collapsed_layers": n_collapsed,
        })

    return profiles


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

    # CKA series (layer 0 is nan — skip)
    cka_vals_ps = [r.get("cka_prev", float("nan")) for r in results["layers"]]
    cka_defined = [(i, v) for i, v in enumerate(cka_vals_ps) if not np.isnan(v)]
    if cka_defined:
        cka_series_ps = [v for _, v in cka_defined]
        metrics.append(("CKA", cka_series_ps, 0.02))

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
          f"{'eff_rank':>9}  {'spec_k':>7}  {'fiedler':>8}  {'sk_k':>5}  {'nn_stab':>8}  {'cka':>7}")
    print(f"{'─'*90}")

    for r in layers:
        sk       = r.get("sinkhorn", {})
        nn_stab  = r.get("nn_stability")
        nn_str   = f"{nn_stab:>8.4f}" if nn_stab is not None else f"{'n/a':>8}"
        cka_val  = r.get("cka_prev", float("nan"))
        cka_str  = f"{cka_val:>7.4f}" if not np.isnan(cka_val) else f"{'n/a':>7}"
        print(
            f"{r['layer']:>6}  "
            f"{r['ip_mean']:>8.4f}  "
            f"{r['ip_mass_near_1']:>9.4f}  "
            f"{r['effective_rank']:>9.2f}  "
            f"{r['spectral']['k_eigengap']:>7}  "
            f"{sk.get('fiedler_mean', float('nan')):>8.4f}  "
            f"{sk.get('sinkhorn_cluster_count_mean', float('nan')):>5.1f}  "
            f"{nn_str}  "
            f"{cka_str}"
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

    # CKA plateaus (layer 0 is nan)
    cka_prs_defined = [
        (r["layer"], r.get("cka_prev", float("nan")))
        for r in layers if not np.isnan(r.get("cka_prev", float("nan")))
    ]
    if cka_prs_defined:
        cka_prs_series = [v for _, v in cka_prs_defined]
        cka_prs_plat   = detect_plateaus(cka_prs_series, window=2, tol=0.02)
        print(f"\n  CKA (consecutive-layer similarity — plateau = metastable):")
        if cka_prs_plat:
            for s, e, v in cka_prs_plat:
                layer_s = cka_prs_defined[s][0]
                layer_e = cka_prs_defined[e][0]
                print(f"    Layers {layer_s:2d}–{layer_e:2d}  mean={v:.4f}  width={e-s+1}")
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

    # CKA series: skip layer 0 (nan) and any other nan values
    cka_vals_raw  = [r.get("cka_prev", float("nan")) for r in layers]
    cka_pairs     = [(r["layer"], r.get("cka_prev", float("nan")))
                     for r in layers if not np.isnan(r.get("cka_prev", float("nan")))]
    cka_series    = [v for _, v in cka_pairs]
    plateaus_cka  = detect_plateaus(cka_series, window=2, tol=0.02) if cka_series else []

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

    # Pre-compute reset layer indices for cross-referencing in the PER-HEAD section.
    # A reset is a layer where ip_mean drops >0.05 AND effective_rank rises >5 —
    # the same thresholds used in the FLAGGED ANOMALIES section.
    reset_layer_indices = set()
    for i in range(1, len(layers)):
        prev, curr = layers[i - 1], layers[i]
        if (prev["ip_mean"] - curr["ip_mean"]) > 0.05 and (curr["effective_rank"] - prev["effective_rank"]) > 5:
            reset_layer_indices.add(curr["layer"])
    merge_layer_indices = {layer for layer, _, _ in merge_events}

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
    W("")

    fiedler_by_layer = {r["layer"]: r["sinkhorn"]["fiedler_mean"]                 for r in has_sk}
    sk_k_by_layer    = {r["layer"]: r["sinkhorn"]["sinkhorn_cluster_count_mean"]  for r in has_sk}
    attn_by_layer    = {r["layer"]: r.get("attention_entropy_mean", float("nan")) for r in has_sk}
    mid_thresh       = float(DISTANCE_THRESHOLDS[len(DISTANCE_THRESHOLDS) // 2])

    W("Columns: layer | ip_mean | ip_std | mass>0.9 | eff_rank | sp_k | k2 | agg_k | km_k | "
      "hdb_k | gap | fiedler | sk_k | attn_H | E_b1 | E_b2 | E_b5 | nn_stab | cka")
    W(f"{'L':>3}  {'ip_μ':>7}  {'ip_σ':>6}  {'m>0.9':>6}  "
      f"{'rank':>7}  {'sp_k':>5}  {'k2':>4}  {'agg_k':>6}  {'km_k':>5}  {'hdb_k':>6}  {'gap':>6}  "
      f"{'fied':>7}  {'sk_k':>5}  {'attn_H':>7}  "
      f"{'E_b1':>7}  {'E_b2':>7}  {'E_b5':>7}  {'nn_stab':>8}  {'cka':>7}")
    W("-" * 142)

    for r in layers:
        li      = r["layer"]
        eg      = max(r["spectral"]["eigengaps"]) if r["spectral"]["eigengaps"] else float("nan")
        hdb_n   = r["clustering"].get("hdbscan", {}).get("n_clusters", float("nan"))
        k2      = r["spectral"].get("k_second_gap", float("nan"))
        agg_k   = r["clustering"]["agglomerative"].get(mid_thresh, float("nan"))
        km_k    = r["clustering"]["kmeans"]["best_k"]
        nn_stab = r.get("nn_stability")
        nn_str  = f"{nn_stab:>8.4f}" if nn_stab is not None else f"{'n/a':>8}"
        cka_val = r.get("cka_prev", float("nan"))
        cka_str = f"{cka_val:>7.4f}" if not (isinstance(cka_val, float) and np.isnan(cka_val)) else f"{'n/a':>7}"
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
            f"{nn_str}  "
            f"{cka_str}"
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
    if cka_pairs:
        cka_plat_parts = []
        for s, e, v in plateaus_cka:
            layer_s = cka_pairs[s][0]
            layer_e = cka_pairs[e][0]
            cka_plat_parts.append(f"layers {layer_s}-{layer_e} (width={e-s+1}, mean={v:.4f})")
        cka_plat_str = "; ".join(cka_plat_parts) if cka_plat_parts else "none detected"
        W(f"  cka            : {cka_plat_str}")
        W("                   (CKA plateau = consecutive layers nearly identical = metastable)")
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
    # CKA plateaus are indexed by cka_pairs positions, remap to layer numbers
    for s, e, _ in plateaus_cka:
        for idx in range(s, e + 1):
            layer_count[cka_pairs[idx][0]] += 1
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

    # --- P1-1: HDBSCAN Cluster Tracking ---
    W("CLUSTER TRACKING  (P1-1 — HDBSCAN membership-based)")
    W("-" * 40)
    W("Tracks clusters across adjacent layers by Jaccard overlap of token")
    W("membership.  Replaces spectral-k counting with token-level accounting.")
    ct = results.get("cluster_tracking", {})
    ct_summary = ct.get("summary", {})
    if ct_summary.get("n_trajectories", 0) > 0:
        W(f"  Trajectories: {ct_summary['n_trajectories']}  "
          f"Mean lifespan: {ct_summary['mean_lifespan']:.1f} layers  "
          f"Max lifespan: {ct_summary['max_lifespan']} layers")
        W(f"  Total births: {ct_summary['total_births']}  "
          f"Deaths: {ct_summary['total_deaths']}  "
          f"Merges: {ct_summary['total_merges']}  "
          f"Peak clusters alive: {ct_summary['max_alive']}")
        # Per-transition detail for merge events
        merge_transitions = [
            ev for ev in ct.get("events", [])
            if ev.get("n_merges", 0) > 0
        ]
        if merge_transitions:
            W("  Merge transitions:")
            for ev in merge_transitions:
                lf = ev["layer_from"]
                lt = ev["layer_to"]
                W(f"    Layer {lf}→{lt}: "
                  f"{ev['n_merges']} merge(s), "
                  f"{ev['n_births']} birth(s), "
                  f"{ev['n_deaths']} death(s)")

                # Build token membership maps for both layers.
                # labels_from[i] = HDBSCAN cluster id for token i at layer_from.
                # Only built when HDBSCAN data is present in the layer record.
                def _tok_map(layer_idx: int) -> dict:
                    """Return {cluster_id: [token_string, ...]} for a layer."""
                    lr_data = layers[layer_idx] if layer_idx < len(layers) else {}
                    hdb     = lr_data.get("clustering", {}).get("hdbscan", {})
                    lbls    = hdb.get("labels")
                    if lbls is None:
                        return {}
                    result: dict = {}
                    for tok_idx, cid in enumerate(lbls):
                        if cid == -1:
                            continue
                        tok_str = tokens[tok_idx] if tok_idx < len(tokens) else f"tok{tok_idx}"
                        result.setdefault(cid, []).append(tok_str)
                    return result

                map_from = _tok_map(lf)
                map_to   = _tok_map(lt)

                for prev_ids, curr_id in ev.get("merges", []):
                    W(f"      Clusters {prev_ids} → {curr_id}")
                    # Show absorbing cluster membership after merge.
                    absorbing = map_to.get(curr_id, [])
                    if absorbing:
                        shown = absorbing[:15]
                        suffix = f" … (+{len(absorbing)-15} more)" if len(absorbing) > 15 else ""
                        W(f"        absorbing cluster {curr_id} now contains: "
                          f"{shown}{suffix}")
                    # For each absorbed cluster, show which tokens it contributed.
                    for pid in prev_ids:
                        donated = map_from.get(pid, [])
                        if donated:
                            shown = donated[:15]
                            suffix = f" … (+{len(donated)-15} more)" if len(donated) > 15 else ""
                            W(f"        ← cluster {pid} donated: {shown}{suffix}")
        # Long-lived trajectories (lifespan > 50% of total layers)
        long_lived = [
            t for t in ct.get("trajectories", [])
            if t["lifespan"] > results["n_layers"] * 0.5
        ]
        if long_lived:
            W(f"  Long-lived trajectories (>{results['n_layers']//2} layers): "
              f"{len(long_lived)}")
            for t in long_lived[:5]:
                W(f"    ID {t['id']}: layers {t['start_layer']}–{t['end_layer']} "
                  f"(lifespan {t['lifespan']})")
    else:
        W("  No HDBSCAN cluster tracking data (HDBSCAN not available or <2 layers)")
    W("")

    # --- P1-3: Multi-scale Nesting ---
    W("MULTI-SCALE NESTING  (P1-3 — spectral eigengap within HDBSCAN clusters)")
    W("-" * 40)
    W("Detects hierarchical organization: global bipartition nesting inside")
    W("local density structure.")
    nesting_layers_found = [
        lr["layer"] for lr in layers
        if lr.get("nesting", {}).get("has_nesting", False)
    ]
    if nesting_layers_found:
        W(f"  Nesting detected at {len(nesting_layers_found)} layers: {nesting_layers_found}")
        for li in nesting_layers_found[:5]:
            ns = layers[li]["nesting"]
            W(f"    Layer {li}: global_k={ns['global_spectral_k']}, "
              f"{ns['n_clusters_with_substructure']} sub-clusters with structure")
    else:
        W("  No hierarchical nesting detected at any layer")
    W("")

    # --- P1-4: Pair HDBSCAN Agreement ---
    W("PAIR HDBSCAN AGREEMENT  (P1-4 — induction head filtering)")
    W("-" * 40)
    W("Mutual-NN pairs tagged: semantic (same HDBSCAN cluster) vs artifact")
    W("(different clusters — likely attention/induction head driven).")
    # Aggregate and show at plateau layers specifically
    plateau_layers_set = set(results.get("plateau_layers", []))
    for lr in layers:
        pa = lr.get("pair_agreement", {})
        total_pairs = pa.get("n_semantic", 0) + pa.get("n_artifact", 0) + pa.get("n_noise", 0)
        if total_pairs > 0 and lr["layer"] in plateau_layers_set:
            W(f"  Layer {lr['layer']} (PLATEAU): "
              f"semantic={pa['n_semantic']} artifact={pa['n_artifact']} "
              f"noise={pa['n_noise']} "
              f"artifact_frac={pa['artifact_fraction']:.2f}")
            # Show up to 5 artifact pairs at this layer
            artifact_pairs = [p for p in pa.get("mutual_pairs", []) if p["tag"] == "artifact"]
            for p in artifact_pairs[:5]:
                W(f"    {p['tok_i']} (c{p['cluster_i']}) ↔ {p['tok_j']} (c{p['cluster_j']})")
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

    W("")

    W("VIOLATION DISTRIBUTION ANALYSIS")
    W("-" * 40)
    W("Three secondary analyses on the energy violation events.")
    W("  (a) Are violations concentrated inside identified plateau windows?")
    W("      Violations during stable periods are anomalous — the geometry should")
    W("      be static, so any disruption needs explanation.")
    W("  (b) Are violations larger at merge-event layers than elsewhere?")
    W("      If merges are energetically costly, mean |ΔE| should spike at merge layers.")
    W("  (c) Does violation magnitude predict the subsequent change in effective rank?")
    W("      A large energy drop followed by a rank rise = reset event.")
    W("")
    plateau_layer_set_viol = set(results.get("plateau_layers", []))
    for beta in [1.0, 2.0]:
        energies_b = [r["energies"].get(beta, float("nan")) for r in layers]
        diffs_b    = np.diff(energies_b)
        viol_idxs  = [i + 1 for i, d in enumerate(diffs_b) if d < -1e-6]
        if not viol_idxs:
            W(f"  beta={beta}: no violations — distribution analysis not applicable.")
            continue
        viol_magnitudes = [abs(diffs_b[i - 1]) for i in viol_idxs]

        # (a) Fraction of violations inside plateaus
        in_plateau  = [li for li in viol_idxs if li in plateau_layer_set_viol]
        frac_in_plat = len(in_plateau) / len(viol_idxs)
        W(f"  beta={beta}:  {len(viol_idxs)} violations total")
        W(f"    (a) {len(in_plateau)}/{len(viol_idxs)} ({frac_in_plat:.0%}) violations inside plateau windows"
          + ("  [ANOMALOUS — geometry should be static here]" if frac_in_plat > 0.3 else ""))

        # (b) Violations at merge layers vs non-merge layers
        merge_viol  = [abs(diffs_b[i - 1]) for i in viol_idxs if i in merge_layer_indices]
        other_viol  = [abs(diffs_b[i - 1]) for i in viol_idxs if i not in merge_layer_indices]
        if merge_viol and other_viol:
            W(f"    (b) mean |ΔE| at merge layers: {np.mean(merge_viol):.6f}  "
              f"vs non-merge: {np.mean(other_viol):.6f}"
              + ("  [MERGE IS COSTLIER]" if np.mean(merge_viol) > np.mean(other_viol) * 1.5 else ""))
        elif merge_viol:
            W(f"    (b) all violations are at merge layers (mean |ΔE|={np.mean(merge_viol):.6f})")
        else:
            W(f"    (b) no violations coincide with merge-event layers")

        # (c) Spearman ρ between |ΔE_L| and Δrank_{L+1}
        # Δrank = rank[L+1] - rank[L]; positive = rank rose (reset-like event)
        rho_pairs = []
        for i in viol_idxs:
            if i < len(layers) - 1:
                delta_e    = abs(diffs_b[i - 1])
                delta_rank = layers[i + 1]["effective_rank"] - layers[i]["effective_rank"]
                rho_pairs.append((delta_e, delta_rank))
        if len(rho_pairs) >= 4:
            rho_e, rho_p = spearmanr([p[0] for p in rho_pairs],
                                     [p[1] for p in rho_pairs])
            sig = "p<0.05" if rho_p < 0.05 else f"p={rho_p:.3f}"
            W(f"    (c) Spearman ρ(|ΔE|, Δrank_next) = {rho_e:.3f}  ({sig}, n={len(rho_pairs)})"
              + ("  [RESET SIGNAL: large drops precede rank rises]"
                 if rho_e > 0.3 and rho_p < 0.05 else ""))
        else:
            W(f"    (c) insufficient paired data for rank correlation (n={len(rho_pairs)} < 4)")
    W("")

    # ------------------------------------------------------------------
    # ENERGY DROP LOCALIZATION
    # Only printed when beta=1.0 violations exist.
    # ------------------------------------------------------------------
    SPECIAL_TOKENS = {"[CLS]", "[SEP]", "<s>", "</s>", "<pad>", "[PAD]",
                      "<|endoftext|>", "Ġ", "▁"}
    PUNCT_CHARS    = set(".,!?;:'\"-–—()[]{}…/\\")

    def _is_special(tok: str) -> bool:
        if tok in SPECIAL_TOKENS:
            return True
        if tok.startswith("##"):          # BERT word-piece continuation
            return False
        if len(tok) == 1 and tok in PUNCT_CHARS:
            return True
        return False

    def _flag_token(tok: str) -> str:
        """Return a short annotation string, or empty string."""
        if tok in ("[CLS]", "<s>"):
            return "[CLS]"
        if tok in ("[SEP]", "</s>"):
            return "[SEP]"
        if len(tok) == 1 and tok in PUNCT_CHARS:
            return "[PUNCT]"
        return ""

    # Collect beta=1.0 violation layers
    energies_b1   = [r["energies"].get(1.0, float("nan")) for r in layers]
    diffs_b1      = np.diff(energies_b1)
    viol_b1_layers = [i + 1 for i, d in enumerate(diffs_b1) if d < -1e-6]

    W("ENERGY DROP LOCALIZATION")
    W("-" * 40)
    if not viol_b1_layers:
        W("  No beta=1.0 energy violations — localization not applicable.")
    else:
        W("Per-pair contribution delta = [exp(β⟨xᵢ,xⱼ⟩_L+1) - exp(β⟨xᵢ,xⱼ⟩_L)] / (2β n²)")
        W("Top-5 most-negative pairs per violation layer (beta=1.0 shown; all betas stored).")
        W("Tokens flagged: [CLS], [SEP], [PUNCT] — structural/special-token repulsion.")
        W("Violation layers in the degenerate regime (eff_rank < 3) are suppressed as noise.")
        W("")
        for vl in viol_b1_layers:
            lr         = layers[vl]
            edp        = lr.get("energy_drop_pairs", {})
            # Support both new dict format {beta: [...]} and old flat list (beta=1.0)
            if isinstance(edp, dict):
                drop_pairs = edp.get(1.0, [])
            else:
                drop_pairs = edp
            e_before    = energies_b1[vl - 1]
            e_after     = energies_b1[vl]
            erank       = lr.get("effective_rank", float("nan"))

            # Identify if this layer is also a reset event
            is_reset = (
                vl > 0
                and (layers[vl - 1]["ip_mean"] - lr["ip_mean"]) > 0.05
                and (erank - layers[vl - 1]["effective_rank"]) > 5
            )
            reset_tag = "  ← RESET" if is_reset else ""

            W(f"  Layer {vl:2d}  (E_b1: {e_before:.6f} → {e_after:.6f}, "
              f"drop={e_after - e_before:.6f}, eff_rank={erank:.1f}){reset_tag}")

            if not drop_pairs:
                # empty means degenerate regime or layer 0 — state which
                if erank < 3.0:
                    W("    [SUPPRESSED — degenerate regime (eff_rank < 3): violation is floating-point noise]")
                else:
                    W("    [no pair data — layer 0 or data missing]")
                W("")
                continue

            # Show top-5
            top5 = drop_pairs[:5]
            flags_seen = []
            for rank, (i, j, delta) in enumerate(top5, 1):
                tok_i  = tokens[i] if i < len(tokens) else f"tok{i}"
                tok_j  = tokens[j] if j < len(tokens) else f"tok{j}"
                flag_i = _flag_token(tok_i)
                flag_j = _flag_token(tok_j)
                ann_i  = f" {flag_i}" if flag_i else ""
                ann_j  = f" {flag_j}" if flag_j else ""
                # Adaptive formatting: scientific notation when delta is very small
                if abs(delta) < 1e-4:
                    delta_str = f"{delta:+.3e}"
                else:
                    delta_str = f"{delta:+.6f}"
                W(f"    {rank}. ({i:3d},{j:3d})  δ={delta_str}  "
                  f"'{tok_i}'{ann_i} ↔ '{tok_j}'{ann_j}")
                for flag in [flag_i, flag_j]:
                    if flag and flag not in flags_seen:
                        flags_seen.append(flag)
            if flags_seen:
                W(f"    [FLAG] Special/structural tokens in top pairs: {flags_seen}")
                W("           Repulsion may be driven by positional/structural subspace,")
                W("           not by semantic content geometry.")
            if is_reset:
                W("    [RESET NOTE] This is a de-clustering layer (ip_mean drops sharply).")
                W("                 Top pairs reflect recently-formed semantic groups being broken apart —")
                W("                 contextually related tokens that were co-clustering prior to this layer.")
            W("")
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

    W("PER-HEAD FIEDLER PROFILE")
    W("-" * 40)
    W("Per-head mean/std Fiedler restricted to the active phase (effective_rank >= 10).")
    W("Post-collapse layers are excluded: once all tokens merge to one cluster, every head")
    W("trivially saturates to Fiedler ≈ 1.0 regardless of its structural role.")
    W("CLUSTER  = mean Fiedler < 0.3 (consistently routes tokens into separated clusters)")
    W("MIXED    = mean Fiedler 0.3–0.7 (variable behaviour across layers)")
    W("MIXING   = mean Fiedler > 0.7 (consistently allows tokens to mix freely)")
    W("")
    head_profiles = _per_head_fiedler_profile(results)
    if not head_profiles:
        W("  No per-head Fiedler data available.")
    else:
        n_active    = head_profiles[0]["n_active_layers"]
        n_collapsed = head_profiles[0]["n_collapsed_layers"]
        if n_collapsed > 0:
            W(f"  Active-phase layers used: {n_active}  "
              f"(collapsed layers excluded: {n_collapsed})")
        else:
            W(f"  Active-phase layers used: {n_active}  (no collapsed layers excluded)")
        W("")
        W(f"  {'Head':>5}  {'Mean':>7}  {'Std':>7}  {'Class':>8}  {'MinLayer':>9}")
        W("  " + "-" * 44)
        flagged_events = reset_layer_indices | merge_layer_indices
        for p in head_profiles:
            flag = " ← RESET/MERGE" if p["min_layer"] in flagged_events else ""
            W(f"  {p['head']:>5}  {p['mean']:>7.4f}  {p['std']:>7.4f}  "
              f"{p['classification']:>8}  {p['min_layer']:>9}{flag}")
        W("")
        cluster_heads = [p["head"] for p in head_profiles if p["classification"] == "CLUSTER"]
        mixing_heads  = [p["head"] for p in head_profiles if p["classification"] == "MIXING"]
        mixed_heads   = [p["head"] for p in head_profiles if p["classification"] == "MIXED"]
        W(f"  CLUSTER heads ({len(cluster_heads)}): {cluster_heads}")
        W(f"  MIXED   heads ({len(mixed_heads)}):   {mixed_heads}")
        W(f"  MIXING  heads ({len(mixing_heads)}):  {mixing_heads}")
        W("")
        # Always report the lowest-mean head — informative even without a CLUSTER bucket
        min_fied_head = min(head_profiles, key=lambda p: p["mean"])
        if cluster_heads:
            W(f"  Strongest clustering head: head {min_fied_head['head']} "
              f"(mean Fiedler={min_fied_head['mean']:.4f}, "
              f"reaches min at layer {min_fied_head['min_layer']})")
        else:
            W(f"  [NOTE] No heads classified as CLUSTER under active-phase thresholds.")
            W(f"         Lowest-mean head: head {min_fied_head['head']} "
              f"(mean={min_fied_head['mean']:.4f}, "
              f"reaches min at layer {min_fied_head['min_layer']}) — "
              f"closest to clustering behaviour.")
        # Flag heads whose min_layer coincides with a reset or merge event
        event_heads = [p["head"] for p in head_profiles if p["min_layer"] in flagged_events]
        if event_heads:
            W(f"  [NOTE] Head(s) with min_layer at a flagged reset/merge layer: {event_heads}")
            W("         These heads reach their lowest Fiedler during a transition event,")
            W("         not during a stable metastable plateau — treat with caution.")
    W("")

    W("")

    W("FIEDLER–CLUSTER CORRELATION")
    W("-" * 40)
    W("Spearman ρ between mean Fiedler value and HDBSCAN cluster count across layers.")
    W("Validates whether low Fiedler (disconnected attention graph) co-occurs with high")
    W("cluster count (geometric multi-cluster structure).  ρ < -0.4 = interpretable signal;")
    W("ρ ≈ 0 = Fiedler and geometry are telling different stories.")
    W("")
    # Build paired (fiedler, hdb_k) for layers that have both values.
    fied_hdb_pairs = [
        (r["sinkhorn"]["fiedler_mean"],
         r["clustering"].get("hdbscan", {}).get("n_clusters", float("nan")))
        for r in layers
        if "sinkhorn" in r
        and not np.isnan(r["clustering"].get("hdbscan", {}).get("n_clusters", float("nan")))
    ]
    if len(fied_hdb_pairs) >= 4:
        fied_arr = np.array([p[0] for p in fied_hdb_pairs])
        hdb_arr  = np.array([p[1] for p in fied_hdb_pairs])
        rho, pval = spearmanr(fied_arr, hdb_arr)
        sig = "significant" if pval < 0.05 else "not significant"
        direction = ""
        if not np.isnan(rho):
            if rho < -0.4:
                direction = "  [SIGNAL] Strong negative correlation — Fiedler tracks cluster geometry."
            elif rho < 0:
                direction = "  [WEAK] Negative but weak — partial alignment between metrics."
            else:
                direction = "  [NOTE] Non-negative correlation — Fiedler does NOT co-vary with cluster count."
        W(f"  Spearman ρ = {rho:.4f}  (p={pval:.4f}, {sig}, n={len(fied_hdb_pairs)} layers)")
        if direction:
            W(direction)
    else:
        W("  Insufficient paired data (need ≥4 layers with both Sinkhorn and HDBSCAN).")
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

    W("")

    W("FIEDLER BIPARTITION LABELING")
    W("-" * 40)
    W("The second Laplacian eigenvector (Fiedler vector) partitions tokens into two")
    W("hemispheres.  Printed at the first layer where spectral k=2, and at any layer")
    W("where the bipartition changes significantly from the previous layer.")
    W("Hypothesis: at early layers this separates CLS/SEP from content, or function")
    W("words from content words.  At metastable layers it should stabilise.")
    W("")
    # Find all layers where spectral k == 2 and bipartition data is available.
    k2_layers = [
        r for r in layers
        if r["spectral"]["k_eigengap"] == 2
        and r.get("fiedler_bipartition") is not None
    ]
    if not k2_layers:
        # Fall back to first layer with any bipartition data
        k2_layers = [r for r in layers if r.get("fiedler_bipartition") is not None][:1]

    def _bipartition_str(lr: dict) -> str:
        bp = lr.get("fiedler_bipartition")
        if bp is None:
            return "  (no bipartition data)"
        pos_tokens = [tokens[i] for i, s in enumerate(bp) if s >= 0 and i < len(tokens)]
        neg_tokens = [tokens[i] for i, s in enumerate(bp) if s <  0 and i < len(tokens)]
        pos_str = str(pos_tokens[:12]) + (" …" if len(pos_tokens) > 12 else "")
        neg_str = str(neg_tokens[:12]) + (" …" if len(neg_tokens) > 12 else "")
        return (f"  (+) side ({len(pos_tokens)} tokens): {pos_str}\n"
                f"  (−) side ({len(neg_tokens)} tokens): {neg_str}")

    prev_bp = None
    reported = 0
    for r in layers:
        bp = r.get("fiedler_bipartition")
        if bp is None:
            continue
        is_k2_layer = r["spectral"]["k_eigengap"] == 2
        # Detect bipartition flip: fraction of tokens that changed sign vs previous.
        if prev_bp is not None and len(bp) == len(prev_bp):
            n_flipped = sum(1 for a, b in zip(bp, prev_bp) if a != b)
            flip_frac = n_flipped / len(bp)
        else:
            flip_frac = 1.0  # first layer or size changed
        # Report at first k=2 layer, at any layer where >30% of tokens flip side,
        # and always at the very first layer, up to 5 total.
        should_report = (
            (reported == 0) or is_k2_layer or (flip_frac > 0.30)
        ) and reported < 5
        if should_report:
            flip_note = f"  [{flip_frac:.0%} tokens flipped side vs previous reported layer]" if reported > 0 else ""
            W(f"  Layer {r['layer']}  (spectral k={r['spectral']['k_eigengap']}){flip_note}")
            W(_bipartition_str(r))
            W("")
            reported += 1
        prev_bp = bp

    if reported == 0:
        W("  No bipartition data available (spectral eigengap did not run with return_fiedler_vec).")
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
            round(float(bin_centers[i]), 2)
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

    # CKA sharpest single-step drop — marks the end of a metastable plateau.
    # A large CKA decrease means representations changed substantially between
    # two consecutive layers — the clearest signal that a plateau has ended.
    # Note: CKA is suppressed (nan) when effective_rank < 3 to avoid
    # noise-dominated ratios in degenerate near-point-mass layers.
    cka_nan_count = sum(1 for v in cka_vals_raw if np.isnan(v))
    if cka_nan_count > 0:
        W(f"  [CKA NOTE] {cka_nan_count} layers suppressed (effective_rank < 3, degenerate regime).")
    if cka_pairs and len(cka_series) >= 2:
        cka_diffs    = np.diff(cka_series)
        drop_idx     = int(np.argmin(cka_diffs))
        drop_val     = float(cka_diffs[drop_idx])
        layer_before = cka_pairs[drop_idx][0]
        layer_after  = cka_pairs[drop_idx + 1][0]
        cka_before   = cka_series[drop_idx]
        cka_after    = cka_series[drop_idx + 1]
        rank_before  = layers[layer_before]["effective_rank"]
        rank_after   = layers[layer_after]["effective_rank"]
        if drop_val < -0.05:
            severity = "SHARP" if drop_val < -0.15 else "MILD"
            W(f"  [CKA DROP] {severity} CKA decrease: layer {layer_before}→{layer_after}  "
              f"Δ={drop_val:.4f}  ({cka_before:.4f}→{cka_after:.4f})")
            W(f"             eff_rank at boundary: {rank_before:.1f}→{rank_after:.1f}  "
              f"(high rank = meaningful regime; low = verify against other metrics)")
            W("             Marks the end of a metastable plateau — representations "
              "reorganise here.")
        else:
            W(f"  [CKA OK] No significant CKA drop detected (max decrease={drop_val:.4f}). "
              "Representations evolve smoothly.")
    W("")

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
        if max_step > 0.05 and layers[max_step_idx]["effective_rank"] >= 10:
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

def generate_cross_run_report(all_results: list, save_dir: Path, control_results: list = None):
    """
    Write a comparative plain-text report across all model/prompt combinations.
    Intended for LLM analysis of trends across conditions.

    Parameters
    ----------
    all_results     : results for metastability analysis (repeated_tokens excluded)
    control_results : repeated_tokens runs, reported separately as collapse controls
    """
    if control_results is None:
        control_results = []

    lines = []
    W = lines.append

    W("=" * 72)
    W("CROSS-RUN COMPARATIVE REPORT — LLM ANALYSIS")
    W("=" * 72)
    W("")
    W("This report compares metastability signals across all model/prompt")
    W("combinations run in this session.")
    W("")
    W("NOTE: repeated_tokens prompts are excluded from metastability analyses")
    W("(they test collapse of a degenerate initial distribution, not metastability).")
    if control_results:
        W(f"{len(control_results)} repeated_tokens runs reported separately as collapse controls.")
    W("")
    W("KEY COMPARISON QUESTIONS:")
    W("  - Does ALBERT show stronger/cleaner metastability than BERT or GPT2?")
    W("  - Do longer prompts produce more metastable windows?")
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
    W("PROMPT SENSITIVITY")
    W("-" * 40)
    W("For each model with ≥2 prompts, SD of plateau onset layer across prompts.")
    W("SD < 2 = plateau onset is a weight-level property (input-independent);")
    W("SD ≥ 2 = plateau onset is content-driven (changes with prompt).")
    W("")
    by_model_ps: dict = {}
    for r in all_results:
        by_model_ps.setdefault(r["model"], []).append(r)
    any_multi = False
    for model, runs in sorted(by_model_ps.items()):
        if len(runs) < 2:
            continue
        any_multi = True
        onsets = []
        for run in runs:
            mass_ = [l["ip_mass_near_1"] for l in run["layers"]]
            plats = detect_plateaus(mass_, window=2, tol=0.10)
            onset = plats[0][0] if plats else None
            onsets.append((run["prompt"], onset))
        valid = [(p, o) for p, o in onsets if o is not None]
        no_plateau = [(p, o) for p, o in onsets if o is None]
        W(f"  {model}:")
        for prompt, onset in valid:
            W(f"    {prompt:<26} plateau onset = layer {onset}")
        for prompt, _ in no_plateau:
            W(f"    {prompt:<26} no plateau detected")
        if len(valid) >= 2:
            onset_vals = [o for _, o in valid]
            sd = float(np.std(onset_vals))
            classification = "weight-level" if sd < 2.0 else "content-driven"
            W(f"    SD of onset across prompts: {sd:.2f}  → {classification}")
        W("")
    if not any_multi:
        W("  Only one run per model — prompt sensitivity requires ≥2 prompts per model.")
    W("")
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
    W("CROSS-PROMPT PER-HEAD FIEDLER CONSISTENCY")
    W("-" * 40)
    W("For each model, compare per-head Fiedler classifications across prompts.")
    W("Heads flagged as INCONSISTENT change role between CLUSTER and MIXING on different prompts.")
    W("Consistent CLUSTER heads are likely structural (syntax/position); inconsistent ones are content-sensitive.")
    W("")

    by_model_cross: dict = {}
    for r in all_results:
        by_model_cross.setdefault(r["model"], []).append(r)

    any_cross_data = False
    for model, runs in by_model_cross.items():
        if len(runs) < 2:
            continue
        any_cross_data = True

        # Build {head: [classification_per_prompt]}
        head_cls_by_prompt: dict = {}
        head_mean_by_prompt: dict = {}
        prompt_labels = []
        for run in runs:
            profiles = _per_head_fiedler_profile(run)
            if not profiles:
                continue
            prompt_labels.append(run["prompt"])
            for p in profiles:
                h = p["head"]
                head_cls_by_prompt.setdefault(h, []).append(p["classification"])
                head_mean_by_prompt.setdefault(h, []).append(p["mean"])

        if not head_cls_by_prompt:
            continue

        W(f"  Model: {model}  (prompts: {prompt_labels})")
        W(f"  {'Head':>5}  {'MeanFiedler':>12}  {'Classes':>30}  {'Status':>12}")
        W("  " + "-" * 65)

        inconsistent_heads = []
        for h in sorted(head_cls_by_prompt):
            classes   = head_cls_by_prompt[h]
            means     = head_mean_by_prompt[h]
            mean_all  = float(np.mean(means))
            cls_set   = set(classes)
            # Inconsistent = CLUSTER on at least one prompt AND MIXING on at least one
            if "CLUSTER" in cls_set and "MIXING" in cls_set:
                status = "INCONSISTENT"
                inconsistent_heads.append(h)
            elif len(cls_set) == 1:
                status = f"STABLE-{classes[0]}"
            else:
                status = "VARIABLE"
            cls_str = " / ".join(classes)
            W(f"  {h:>5}  {mean_all:>12.4f}  {cls_str:>30}  {status:>12}")

        if inconsistent_heads:
            W(f"  [FLAG] Content-sensitive heads (CLUSTER on some, MIXING on others): "
              f"{inconsistent_heads}")
            W("         These heads respond to prompt content rather than serving a fixed structural role.")
        else:
            W("  [OK] All heads show consistent classification across prompts for this model.")
        W("")

    if not any_cross_data:
        W("  Only one run per model — cross-prompt comparison requires ≥2 prompts per model.")
        W("")

    # --- CLUSTER TRACKING SUMMARY (P1-1) ---
    W("CLUSTER TRACKING SUMMARY (P1-1)")
    W("-" * 40)
    W("HDBSCAN cluster births, deaths, and merges tracked across layers.")
    W("Replaces spectral-k drop counting with token-level accounting.")
    W("")
    for r in all_results:
        ct = r.get("cluster_tracking", {})
        summary = ct.get("summary", {})
        if summary.get("n_trajectories", 0) > 0:
            W(f"  {r['model']} | {r['prompt']}")
            W(f"    Trajectories: {summary['n_trajectories']}  "
              f"Mean lifespan: {summary['mean_lifespan']:.1f}  "
              f"Max lifespan: {summary['max_lifespan']}")
            W(f"    Births: {summary['total_births']}  "
              f"Deaths: {summary['total_deaths']}  "
              f"Merges: {summary['total_merges']}  "
              f"Max alive: {summary['max_alive']}")
            # Report merge events at specific layer transitions
            merge_layers = [
                ev for ev in ct.get("events", [])
                if ev.get("n_merges", 0) > 0
            ]
            if merge_layers:
                labels = [
                    f"{ev['layer_from']}→{ev['layer_to']}"
                    for ev in merge_layers
                ]
                W(f"    Merge events at layer transitions: {labels}")
            W("")
    W("")

    # --- MULTI-SCALE NESTING SUMMARY (P1-3) ---
    W("MULTI-SCALE NESTING SUMMARY (P1-3)")
    W("-" * 40)
    W("Spectral eigengap within HDBSCAN clusters — detects hierarchical organization.")
    W("")
    for r in all_results:
        # Summarize nesting across layers
        nesting_layers = [
            lr["layer"] for lr in r["layers"]
            if lr.get("nesting", {}).get("has_nesting", False)
        ]
        if nesting_layers:
            W(f"  {r['model']} | {r['prompt']}")
            W(f"    Nesting detected at layers: {nesting_layers}")
            # Sample one layer for detail
            sample_lr = r["layers"][nesting_layers[0]]
            ns = sample_lr["nesting"]
            W(f"    Layer {nesting_layers[0]}: global_k={ns['global_spectral_k']}, "
              f"{ns['n_clusters_with_substructure']} sub-clusters with internal structure")
            W("")
    W("")

    # --- PAIR AGREEMENT SUMMARY (P1-4) ---
    W("PAIR HDBSCAN AGREEMENT SUMMARY (P1-4)")
    W("-" * 40)
    W("Mutual-NN pairs tagged as semantic (same cluster) vs artifact (different clusters).")
    W("")
    for r in all_results:
        # Aggregate across layers
        total_semantic = sum(lr.get("pair_agreement", {}).get("n_semantic", 0) for lr in r["layers"])
        total_artifact = sum(lr.get("pair_agreement", {}).get("n_artifact", 0) for lr in r["layers"])
        total_noise = sum(lr.get("pair_agreement", {}).get("n_noise", 0) for lr in r["layers"])
        total = total_semantic + total_artifact + total_noise
        if total > 0:
            W(f"  {r['model']} | {r['prompt']}")
            W(f"    Total mutual-NN pairs: {total}  "
              f"Semantic: {total_semantic} ({100*total_semantic/total:.0f}%)  "
              f"Artifact: {total_artifact} ({100*total_artifact/total:.0f}%)  "
              f"Noise: {total_noise} ({100*total_noise/total:.0f}%)")
            W("")
    W("")

    # --- COLLAPSE CONTROL RUNS (P1-2) ---
    if control_results:
        W("COLLAPSE CONTROL RUNS (repeated_tokens)")
        W("-" * 40)
        W("These runs test collapse speed of a degenerate initial distribution.")
        W("They are NOT metastability tests and are excluded from the above analyses.")
        W("")
        W("Two-timescale ratio: plateau_width / collapse_onset_layer.")
        W("Theory predicts this ratio >> 1 and growing with iteration depth.")
        W("A ratio near 1 means the metastable window is no wider than the fast collapse.")
        W("")
        # Pre-build {model_name: mean_plateau_width} from metastability runs.
        model_plateau_width: dict = {}
        for r in all_results:
            mass_ = [l["ip_mass_near_1"] for l in r["layers"]]
            plats = detect_plateaus(mass_, window=2, tol=0.10)
            widths = [e - s + 1 for s, e, _ in plats]
            if widths:
                model_plateau_width.setdefault(r["model"], []).extend(widths)
        mean_plateau_width: dict = {
            m: float(np.mean(ws)) for m, ws in model_plateau_width.items()
        }

        for r in control_results:
            layers_ = r["layers"]
            mass1_ = [l["ip_mass_near_1"] for l in layers_]
            erank_ = [l["effective_rank"] for l in layers_]
            e1_ = [l["energies"].get(1.0, float("nan")) for l in layers_]
            W(f"  {r['model']}")
            W(f"    Tokens: {r['n_tokens']}  Layers: {r['n_layers']}")
            W(f"    Mass>0.9 at layer 0: {mass1_[0]:.4f}  Final: {mass1_[-1]:.4f}")
            W(f"    Rank at layer 0: {erank_[0]:.2f}  Final: {erank_[-1]:.2f}")
            collapse_layers = [i for i, m in enumerate(mass1_) if m > 0.9]
            if collapse_layers:
                onset = collapse_layers[0]
                W(f"    Collapse onset (mass>0.9): layer {onset}")
                # Look up matching model's mean plateau width.
                mpw = mean_plateau_width.get(r["model"])
                if mpw is not None and onset > 0:
                    ratio = mpw / onset
                    interp = (
                        "metastable window >> fast collapse  [TWO-TIMESCALE CONFIRMED]"
                        if ratio > 2.0 else
                        "metastable window ≈ fast collapse  [WEAK SEPARATION]"
                        if ratio > 1.0 else
                        "metastable window NARROWER than collapse  [NO SEPARATION]"
                    )
                    W(f"    Mean plateau width (metastability runs): {mpw:.1f} layers")
                    W(f"    Ratio plateau_width / collapse_onset = {ratio:.2f}  → {interp}")
                elif mpw is None:
                    W(f"    No matching metastability run found for ratio computation.")
            else:
                W(f"    No collapse (mass never exceeds 0.9)")
            W("")
        W("")

    W("=" * 72)
    W("END OF CROSS-RUN REPORT")
    W("=" * 72)

    out_path = save_dir / "llm_cross_run_report.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Cross-run LLM report: {out_path}")
