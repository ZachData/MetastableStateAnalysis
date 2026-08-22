"""
p5b_manifold/merge_teleportation.py — Sub-experiment C.

At merge layers, do output distributions show "teleportation" — probability
mass jumping to non-adjacent tokens, passing through off-My regions of
behavior space? Mirrors Wurgaft's linear-steering pathology but applied to
the model's own dynamics at cluster merge events.

Imports from existing project scripts:
  - Phase 1 io_utils.load_run  → merge_events list
  - Phase 1 clustering.json    → plateau_layers list
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


# ---------------------------------------------------------------------------
# Per-event scoring
# ---------------------------------------------------------------------------

def teleportation_score(
    p_before: np.ndarray,
    p_at:     np.ndarray,
    p_after:  np.ndarray,
    top_k:    int = 5,
) -> dict:
    """
    Compute teleportation metrics at a single merge event.

    Parameters
    ----------
    p_before : (vocab,) — output distribution just before the merge
    p_at     : (vocab,) — output distribution at the merge layer
    p_after  : (vocab,) — output distribution just after the merge
    top_k    : number of top-k tokens to consider as "adjacent"

    Returns
    -------
    dict with:
      kl_divergence       — KL(p_at || p_before), measures jump magnitude
      non_adjacent_mass   — fraction of p_at mass on tokens NOT in top-k of p_before
      bhattacharyya_approx — -log(∑ √(p_at·p_before)), approx Bhat. distance
    """
    eps = 1e-10
    p_b = np.clip(p_before, eps, None)
    p_a = np.clip(p_at,     eps, None)

    kl  = float(np.sum(p_a * np.log(p_a / p_b)))

    # Non-adjacent mass: tokens not in top-k of p_before
    top_k_idx = set(np.argpartition(p_before, -top_k)[-top_k:].tolist())
    non_adj_mass = float(sum(p_at[i] for i in range(len(p_at)) if i not in top_k_idx))

    # Bhattacharyya coefficient
    bhat_coeff = float(np.sum(np.sqrt(np.clip(p_at, 0, None) *
                                       np.clip(p_before, 0, None))))
    bhat_dist  = float(-np.log(max(bhat_coeff, eps)))

    return {
        "kl_divergence":      kl,
        "non_adjacent_mass":  non_adj_mass,
        "bhattacharyya_approx": bhat_dist,
    }


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_event_distributions(
    logit_distributions: dict,   # {layer_idx: (n_tokens, vocab)}
    merge_layers:        list[int],
    plateau_layers:      list[int],
    window:              int = 2,
) -> tuple[dict, dict]:
    """
    For each merge event, collect distributions before, at, and after.
    Also collect matched plateau-layer distributions as the null comparison.

    Parameters
    ----------
    logit_distributions : layer_idx → (n_tokens, vocab) probability arrays
                          (averaged over tokens, or for a specific token position)
    merge_layers        : list of layer indices flagged as merge events by Phase 1
    plateau_layers      : list of layer indices in plateau windows
    window              : how many layers before/after to look for the surrounding state

    Returns
    -------
    merge_triples   : list of (p_before, p_at, p_after) for each merge event
    plateau_singles : list of p at plateau layers (not near a merge)
    """
    merge_set = set(merge_layers)
    merge_triples  = []
    plateau_singles = []

    for ml in merge_layers:
        # Find p_before: nearest plateau layer before ml
        before_layers = [l for l in plateau_layers if l < ml and ml - l <= window + 5]
        after_layers  = [l for l in plateau_layers if l > ml and l - ml <= window + 5]
        if not before_layers or not after_layers:
            continue
        l_before = max(before_layers)
        l_after  = min(after_layers)

        if ml not in logit_distributions: continue
        if l_before not in logit_distributions: continue
        if l_after not in logit_distributions: continue

        p_before = logit_distributions[l_before].mean(axis=0)
        p_at     = logit_distributions[ml].mean(axis=0)
        p_after  = logit_distributions[l_after].mean(axis=0)
        merge_triples.append((p_before, p_at, p_after))

    # Plateau singles: plateau layers not adjacent to any merge
    for pl in plateau_layers:
        if any(abs(pl - ml) <= window + 3 for ml in merge_layers):
            continue
        if pl not in logit_distributions:
            continue
        plateau_singles.append(logit_distributions[pl].mean(axis=0))

    return merge_triples, plateau_singles


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_merge_plateau(
    merge_scores:   dict,
    plateau_scores: dict,
    alpha:          float = 0.05,
) -> dict:
    """
    Statistical comparison of teleportation scores at merge vs plateau layers.

    Parameters
    ----------
    merge_scores   : dict with keys 'kl_divergence', 'non_adjacent_mass' (lists)
    plateau_scores : same structure

    Returns
    -------
    dict with means, p-values (Mann-Whitney U), and PASS/FAIL for P5b-C1, C3
    """
    results = {}

    for metric in ("kl_divergence", "non_adjacent_mass"):
        m_vals  = np.array(merge_scores.get(metric, []))
        p_vals  = np.array(plateau_scores.get(metric, []))

        if len(m_vals) < 2 or len(p_vals) < 2:
            results[f"{metric}_untestable"] = True
            results[f"{metric}_mean_merge"]   = float(m_vals.mean()) if len(m_vals) else float("nan")
            results[f"{metric}_mean_plateau"] = float(p_vals.mean()) if len(p_vals) else float("nan")
            results[f"{metric}_pvalue"]       = float("nan")
            continue

        stat, pval = mannwhitneyu(m_vals, p_vals, alternative="greater")
        results[f"{metric}_mean_merge"]   = float(m_vals.mean())
        results[f"{metric}_mean_plateau"] = float(p_vals.mean())
        results[f"{metric}_pvalue"]       = float(pval)
        results[f"{metric}_n_merge"]      = int(len(m_vals))
        results[f"{metric}_n_plateau"]    = int(len(p_vals))

    # Flatten for test compatibility
    results["kl_mean_merge"]    = results.get("kl_divergence_mean_merge", float("nan"))
    results["kl_mean_plateau"]  = results.get("kl_divergence_mean_plateau", float("nan"))
    results["kl_pvalue"]        = results.get("kl_divergence_pvalue", float("nan"))
    results["nam_mean_merge"]   = results.get("non_adjacent_mass_mean_merge", float("nan"))
    results["nam_mean_plateau"] = results.get("non_adjacent_mass_mean_plateau", float("nan"))
    results["nam_pvalue"]       = results.get("non_adjacent_mass_pvalue", float("nan"))

    kl_pass  = (results["kl_mean_merge"] > results["kl_mean_plateau"] and
                results["kl_pvalue"] < alpha)
    nam_pass = (results["nam_mean_merge"] > results["nam_mean_plateau"] and
                results["nam_pvalue"] < alpha)

    results["p5b_c1_pass"] = kl_pass
    results["p5b_c3_pass"] = nam_pass

    return results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_merge_teleportation(
    logit_distributions: dict,
    merge_layers:        list[int],
    plateau_layers:      list[int],
) -> dict:
    """Full Sub-experiment C pipeline."""
    triples, plateau_singles = extract_event_distributions(
        logit_distributions, merge_layers, plateau_layers
    )

    merge_scores   = {"kl_divergence": [], "non_adjacent_mass": []}
    plateau_scores = {"kl_divergence": [], "non_adjacent_mass": []}

    for p_before, p_at, p_after in triples:
        s = teleportation_score(p_before, p_at, p_after)
        merge_scores["kl_divergence"].append(s["kl_divergence"])
        merge_scores["non_adjacent_mass"].append(s["non_adjacent_mass"])

    # Null: consecutive plateau pairs — measures how much distributions
    # shift in a *stable* window so we have a non-trivial baseline.
    # plateau_singles is already in plateau_layers order (the order
    # extract_event_distributions iterated it in), so no re-sort is needed
    # here — it's the caller's job to pass plateau_layers pre-sorted if
    # "consecutive" is meant to track layer order.
    for idx in range(len(plateau_singles) - 1):
        p_a = plateau_singles[idx]
        p_b = plateau_singles[idx + 1]
        s = teleportation_score(p_a, p_b, p_b)   # p_after=p_b is a no-op stand-in
        plateau_scores["kl_divergence"].append(s["kl_divergence"])
        plateau_scores["non_adjacent_mass"].append(s["non_adjacent_mass"])

    comparison = compare_merge_plateau(merge_scores, plateau_scores)

    return {
        "n_merge_events":  len(triples),
        "n_plateau_refs":  len(plateau_singles),
        "merge_scores":    merge_scores,
        "plateau_scores":  plateau_scores,
        **comparison,
    }



