"""
cross_term_analysis.py — Cross-term token-pair analysis for ALBERT-xlarge.

When the additive decomposition Δ_total = Δ_attn + Δ_ffn misses most of
the energy drop (i.e. |Δ_cross| > threshold), the cross-term

    Δ_cross = E(x + Δ_a + Δ_f) - E(x + Δ_a) - E(x + Δ_f) + E(x)

is the dominant mechanism.  This module identifies which token pairs drive
the cross-term and tests whether they overlap with Phase 1's drop_pairs,
tying the cross-term mechanism back to V's repulsive dynamics.

The per-pair cross-term interaction is:

    C_ij = Δa_i · Δf_j + Δf_i · Δa_j

where · is the dot product.  Positive C_ij means attention and FFN
updates on the two tokens are aligned (both pushing them together or
apart); negative means they are anti-aligned.  The energy change from
the cross-term between pair (i, j) is proportional to C_ij scaled by
exp(β⟨x_i, x_j⟩).

Functions
---------
pairwise_cross_term_matrix   : C_ij matrix for one violation layer
cross_term_dominant_pairs    : top-k pairs by |C_ij|
jaccard_with_drop_pairs      : overlap with Phase 1 energy drop pairs
run_cross_term_analysis      : full pipeline for one model × prompt
print_cross_term_summary     : terminal output
"""

import numpy as np
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Pair-wise cross-term matrix
# ---------------------------------------------------------------------------

def pairwise_cross_term_matrix(
    attn_delta: np.ndarray,
    ffn_delta: np.ndarray,
) -> np.ndarray:
    """
    Compute the symmetric cross-term interaction matrix C_ij.

    C_ij = Δa_i · Δf_j + Δf_i · Δa_j

    For i == j this reduces to 2 * Δa_i · Δf_i, the self cross-term.
    The off-diagonal entries represent the interaction between token i's
    attention update and token j's FFN update (and vice versa).

    Parameters
    ----------
    attn_delta : (n_tokens, d_model) float — attention residual update
    ffn_delta  : (n_tokens, d_model) float — FFN residual update

    Returns
    -------
    C : (n_tokens, n_tokens) float — symmetric cross-term matrix
    """
    # Δa_i · Δf_j = attn_delta @ ffn_delta.T  [entry (i,j)]
    af = attn_delta @ ffn_delta.T   # (n, n)
    C  = af + af.T                  # C_ij = Δa_i·Δf_j + Δf_i·Δa_j
    return C


def cross_term_dominant_pairs(
    C: np.ndarray,
    top_k: int = 10,
    off_diagonal_only: bool = True,
) -> list:
    """
    Return the top-k pairs (i, j) with largest |C_ij|, sorted descending.

    Parameters
    ----------
    C                : (n, n) cross-term matrix
    top_k            : number of pairs to return
    off_diagonal_only: if True, exclude (i, i) self-pairs

    Returns
    -------
    list of dicts with: i, j, C_ij, sign ("aligned" | "anti-aligned")
    """
    n = C.shape[0]
    abs_C = np.abs(C)

    if off_diagonal_only:
        np.fill_diagonal(abs_C, 0.0)

    # Get flat indices of top-k
    flat = abs_C.ravel()
    top_flat = np.argsort(flat)[::-1][:top_k * 2]  # over-sample for unique pairs

    seen = set()
    pairs = []
    for idx in top_flat:
        i, j = divmod(idx, n)
        if off_diagonal_only and i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        c_val = float(C[i, j])
        pairs.append({
            "i":     int(i),
            "j":     int(j),
            "C_ij":  c_val,
            "abs_C": float(abs_C[i, j]),
            "sign":  "aligned" if c_val > 0 else "anti-aligned",
        })
        if len(pairs) >= top_k:
            break

    return sorted(pairs, key=lambda p: p["abs_C"], reverse=True)


def jaccard_with_drop_pairs(
    cross_term_pairs: list,
    drop_pairs: list,
    top_k: Optional[int] = None,
) -> dict:
    """
    Measure overlap between cross-term dominant pairs and Phase 1 drop pairs.

    If the cross-term is driven by the same pairs as the energy drop (as
    recorded in Phase 1), the mechanism is linked to V's repulsive dynamics
    acting on semantically coherent token pairs.

    Parameters
    ----------
    cross_term_pairs : list of dicts from cross_term_dominant_pairs
    drop_pairs       : list of (i, j, delta) from phase1_events["energy_drop_pairs"]
    top_k            : if set, only use the top-k cross-term pairs

    Returns
    -------
    dict with:
      jaccard           : |intersection| / |union|
      n_cross_term      : number of cross-term pairs tested
      n_drop_pairs      : number of Phase 1 drop pairs
      n_shared          : pairs in both sets
      shared_pairs      : list of (i, j) tuples
    """
    if top_k is not None:
        cross_term_pairs = cross_term_pairs[:top_k]

    ct_set = {(min(p["i"], p["j"]), max(p["i"], p["j"]))
              for p in cross_term_pairs}

    # drop_pairs can be (i, j, delta) triples or (i, j) pairs
    dp_set = set()
    for p in drop_pairs:
        if len(p) >= 2:
            dp_set.add((min(p[0], p[1]), max(p[0], p[1])))

    intersection = ct_set & dp_set
    union        = ct_set | dp_set

    jaccard = len(intersection) / max(len(union), 1)

    return {
        "jaccard":       jaccard,
        "n_cross_term":  len(ct_set),
        "n_drop_pairs":  len(dp_set),
        "n_shared":      len(intersection),
        "shared_pairs":  sorted(intersection),
    }


# ---------------------------------------------------------------------------
# Per-violation cross-term analysis
# ---------------------------------------------------------------------------

def analyze_violation_cross_term(
    v_layer: int,
    attn_delta_raw: np.ndarray,
    ffn_delta_raw:  np.ndarray,
    delta_cross: float,
    delta_total: float,
    drop_pairs: list,
    top_k: int = 10,
    dominance_threshold: float = 0.5,
) -> dict:
    """
    Full cross-term analysis for one violation layer.

    Parameters
    ----------
    v_layer            : violation layer index
    attn_delta_raw     : (n_tokens, d_model) — raw (unnormed) attn delta at L-1
    ffn_delta_raw      : (n_tokens, d_model) — raw (unnormed) FFN delta at L-1
    delta_cross        : scalar cross-term energy from energy_by_component
    delta_total        : scalar total energy change
    drop_pairs         : list from phase1_events["energy_drop_pairs"][v_layer]
    top_k              : number of dominant pairs to extract
    dominance_threshold: |delta_cross| / |delta_total| must exceed this

    Returns
    -------
    dict with:
      layer, cross_dominant (bool), cross_frac,
      top_pairs, jaccard_result,
      total_C_energy (sum of |C_ij|),
      cross_mechanism ("V_linked" | "independent" | "ambiguous")
    """
    cross_frac = (abs(delta_cross) / max(abs(delta_total), 1e-12)
                  if np.isfinite(delta_cross) and np.isfinite(delta_total) else 0.0)
    cross_dominant = cross_frac > dominance_threshold

    C = pairwise_cross_term_matrix(attn_delta_raw, ffn_delta_raw)
    top_pairs = cross_term_dominant_pairs(C, top_k=top_k)
    jaccard_result = jaccard_with_drop_pairs(top_pairs, drop_pairs, top_k=top_k)

    total_C_energy = float(np.sum(np.abs(C)))

    # Classify the cross-term mechanism
    if not cross_dominant:
        mechanism = "not_dominant"
    elif jaccard_result["jaccard"] > 0.5:
        mechanism = "V_linked"       # same pairs → V's repulsive dynamics
    elif jaccard_result["jaccard"] > 0.2:
        mechanism = "partial_overlap"
    else:
        mechanism = "independent"    # different pairs → unrelated to V

    return {
        "layer":           v_layer,
        "cross_dominant":  cross_dominant,
        "cross_frac":      cross_frac,
        "delta_cross":     float(delta_cross) if np.isfinite(delta_cross) else 0.0,
        "delta_total":     float(delta_total),
        "top_pairs":       top_pairs,
        "jaccard_result":  jaccard_result,
        "total_C_energy":  total_C_energy,
        "cross_mechanism": mechanism,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_cross_term_analysis(
    ffn_dir: Path,
    phase1_run_dir: Path,
    decomposed_violations: list,
    beta: float = 1.0,
    top_k: int = 10,
    dominance_threshold: float = 0.5,
) -> dict:
    """
    Run cross-term token-pair analysis for all violation layers where the
    cross-term is dominant.

    Loads raw (unnormed) attn and FFN deltas.  Requires decomposed_violations
    from analyze_violations_decomposed — these provide delta_cross and
    delta_total for each violation, and are used to filter for dominant layers.

    Parameters
    ----------
    ffn_dir               : directory with attn_deltas_raw.npz and ffn_deltas_raw.npz
    phase1_run_dir        : Phase 1 run directory (for drop_pairs)
    decomposed_violations : list of dicts from decompose.analyze_violations_decomposed
    beta                  : which beta's drop_pairs to load
    top_k                 : dominant pairs to extract per violation
    dominance_threshold   : |Δ_cross| / |Δ_total| cutoff for "dominant"

    Returns
    -------
    dict with:
      applicable (bool), reason (str if not),
      per_violation: list of analyze_violation_cross_term results,
      summary: aggregate statistics
    """
    ffn_dir        = Path(ffn_dir)
    phase1_run_dir = Path(phase1_run_dir)

    # Load raw deltas (fix 2)
    attn_path = ffn_dir / "attn_deltas_raw.npz"
    ffn_path  = ffn_dir / "ffn_deltas_raw.npz"

    if not attn_path.exists() or not ffn_path.exists():
        return {"applicable": False,
                "reason": "Missing attn_deltas_raw.npz or ffn_deltas_raw.npz. "
                          "Re-run decompose.save_decomposed to generate raw files."}

    attn_deltas = np.load(attn_path)["attn_deltas"]   # (n_layers, n_tokens, d)
    ffn_deltas  = np.load(ffn_path)["ffn_deltas"]     # (n_layers, n_tokens, d)

    # Load Phase 1 events for drop_pairs
    import json
    from core.config import BETA_VALUES
    with open(phase1_run_dir / "metrics.json") as f:
        p1 = json.load(f)

    drop_pairs_by_layer = {}
    for layer in p1.get("layers", []):
        layer_idx = layer["layer"]
        raw = layer.get("energy_drop_pairs", {})
        if isinstance(raw, list):
            raw = {1.0: raw}
        else:
            raw = {float(k): v for k, v in raw.items()}
        pairs = raw.get(beta, [])
        if pairs:
            drop_pairs_by_layer[layer_idx] = pairs

    if not decomposed_violations:
        return {"applicable": False, "reason": "No decomposed_violations provided"}

    per_violation = []
    for dv in decomposed_violations:
        v_layer     = dv["layer"]
        delta_cross = dv.get("delta_cross", 0.0)
        delta_total = dv.get("delta_total", 0.0)

        t_idx = v_layer - 1
        if t_idx < 0 or t_idx >= attn_deltas.shape[0]:
            continue

        result = analyze_violation_cross_term(
            v_layer         = v_layer,
            attn_delta_raw  = attn_deltas[t_idx],
            ffn_delta_raw   = ffn_deltas[t_idx],
            delta_cross     = delta_cross,
            delta_total     = delta_total,
            drop_pairs      = drop_pairs_by_layer.get(v_layer, []),
            top_k           = top_k,
            dominance_threshold = dominance_threshold,
        )
        per_violation.append(result)

    if not per_violation:
        return {"applicable": True, "per_violation": [], "summary": {}}

    n = len(per_violation)
    n_dominant = sum(1 for v in per_violation if v["cross_dominant"])
    v_linked   = [v for v in per_violation if v["cross_mechanism"] == "V_linked"]
    indep      = [v for v in per_violation if v["cross_mechanism"] == "independent"]

    mean_jaccard_dominant = float(np.mean([
        v["jaccard_result"]["jaccard"]
        for v in per_violation if v["cross_dominant"]
    ])) if n_dominant > 0 else float("nan")

    summary = {
        "n_violations":           n,
        "n_cross_dominant":       n_dominant,
        "frac_cross_dominant":    n_dominant / n,
        "n_V_linked":             len(v_linked),
        "n_independent":          len(indep),
        "mean_jaccard_dominant":  mean_jaccard_dominant,
        "mean_cross_frac":        float(np.mean([v["cross_frac"] for v in per_violation])),
    }

    return {
        "applicable":    True,
        "per_violation": per_violation,
        "summary":       summary,
    }


def print_cross_term_summary(result: dict, model_name: str, prompt_key: str) -> None:
    """Print concise cross-term analysis summary."""
    if not result.get("applicable"):
        print(f"\n  Cross-term analysis not applicable: {result.get('reason')}")
        return

    s = result.get("summary", {})
    if not s:
        return

    print(f"\n  Cross-term analysis ({model_name} | {prompt_key}):")
    print(f"    {s['n_cross_dominant']}/{s['n_violations']} violations "
          f"cross-term dominant (>{s['mean_cross_frac']:.0%} of total ΔE)")
    if s["n_cross_dominant"] > 0:
        print(f"    V-linked (Jaccard>0.5): {s['n_V_linked']}  "
              f"Independent: {s['n_independent']}")
        print(f"    Mean Jaccard (dominant violations): "
              f"{s['mean_jaccard_dominant']:.3f}")

    for v in result["per_violation"][:5]:
        if not v["cross_dominant"]:
            continue
        jr = v["jaccard_result"]
        print(f"      L{v['layer']:3d}  "
              f"cross_frac={v['cross_frac']:.2f}  "
              f"Jaccard={jr['jaccard']:.3f}  "
              f"shared={jr['n_shared']}/{jr['n_drop_pairs']}  "
              f"→ {v['cross_mechanism']}")
        for p in v["top_pairs"][:3]:
            print(f"        ({p['i']},{p['j']})  "
                  f"C={p['C_ij']:+.4f}  {p['sign']}")
