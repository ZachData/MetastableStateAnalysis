"""
p7_motifs/motif_stats.py — motif counts against matched nulls, and the
verdicts P-I1..P-I4 are adjudicated on.

The whole difficulty of this phase lives here, and it is not the counting.
It is that a motif count on its own is meaningless: on Pythia the attention
bilinear is M(Δ) = W_Q R(Δ) W_K^T, so anything that depends on relative
offset can be manufactured by an offset-distribution difference alone, and
induction pairs are not offset-matched against same-content pairs. A
`relay` count that clears "more than chance" without clearing an
offset-matched null is a rediscovery of the rotary embedding.

core/qk_offset_null.py already worked this out for P6-I2b, after that test
was found broken in exactly this way. The three nulls are reused rather
than re-derived:

  N1  rotary-only    closed form from the offsets, no weights involved
  N2  offset-matched same-content pairs at the SAME offsets
  N3  offset-shuffled  the motif's own pairs with offsets permuted

A PASS requires clearing N1 **and** N2. N3 is reported but does not gate:
it distinguishes "content and offset are jointly required" from "either
alone suffices", which is a statement about mechanism, not about whether
the effect is real.

Effective n
-----------
Edges within a head are not independent samples. Every significance call
here is made over **heads** (or head pairs, for relay), never over edge
counts — see PREDICTIONS.md's Phase 7 adjudication constraint 1. A z-score
computed over edges would be inflated by orders of magnitude, in the
direction that manufactures findings.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from core.interactions import InteractionTable
from core.nulls import nsigma_verdict, sigma_from_null
from p7_motifs.motif_alphabet import (
    ALPHABET_VERSION,
    DEFAULTS,
    THRESHOLD_STATUS,
    find_relays,
    motif_mask,
    relay_strength,
)

# Which nulls must be cleared for a PASS. N3 is informational.
GATING_NULLS = ("N1", "N2")
ALL_NULLS = ("N1", "N2", "N3")


class DegeneratePrompt(ValueError):
    """
    Raised when a prompt cannot carry a motif test at all.

    Refusing is the correct behaviour, not a fallback (standing rule 4):
    core/battery_structure.py's four degeneracy modes each describe a
    situation where the comparison is undefined rather than merely weak.
    `PROMPTS["repeated_tokens"]` is the worked case — every token is
    identical, so every causal pair is an induction pair, the
    same-content null set is empty, and there is nothing for N2 to be.
    """


def check_prompt_admissible(structure_report: dict, prompt_key: str) -> None:
    """
    Gate a prompt on core.battery_structure's own verdict before any
    counting happens.

    `structure_report` is what battery_structure.analyze_prompt /
    verify_battery_structure produced for this prompt. This function does
    not re-derive degeneracy — that logic lives in battery_structure and
    is tested there; duplicating the rule here is how the two drift apart.
    """
    modes = structure_report.get("degeneracy") or structure_report.get("degeneracy_modes")
    if modes:
        raise DegeneratePrompt(
            f"prompt {prompt_key!r} is degenerate for motif analysis "
            f"({', '.join(sorted(modes))}); refusing rather than returning "
            "a number. See core/battery_structure.py."
        )


# ---------------------------------------------------------------------------
# Per-head aggregation — the unit every significance call is made over
# ---------------------------------------------------------------------------

def per_head_motif_rate(
    t: InteractionTable,
    motif: str,
    thresholds: Optional[dict] = None,
) -> Dict[tuple, float]:
    """
    Fraction of each head's edges that participate in `motif`, keyed by
    (layer, head).

    A rate rather than a count, because heads differ in how many edges
    survive the retention cutoff and a raw count would rank heads by
    density rather than by structure. Heads with no edges are absent from
    the result rather than present with 0.0 — "this head had nothing to
    measure" and "this head had edges and none matched" are different
    facts, and averaging the first into the second is how a real effect
    gets diluted by empty heads.
    """
    res = motif_mask(motif, t, thresholds)
    mask = res["mask"]
    layers, heads = t.columns["layer"], t.columns["head"]

    out: Dict[tuple, float] = {}
    for key in {(int(l), int(h)) for l, h in zip(layers, heads)}:
        sel = (layers == key[0]) & (heads == key[1])
        n = int(sel.sum())
        if n == 0:
            continue
        out[key] = float(mask[sel].sum()) / n
    return out


def relay_rate_by_head_pair(
    t: InteractionTable,
    thresholds: Optional[dict] = None,
    normalize_by: Optional[Dict[tuple, int]] = None,
) -> Dict[tuple, float]:
    """
    Relay counts per (l1, h1, l2, h2), optionally normalized by the number
    of relays that were *possible* for that head pair.

    `normalize_by` is the denominator — how many compositions the prompt's
    induction-pair structure admits for each head pair. It is supplied by
    the caller rather than computed here on purpose: the denominator is a
    modelling choice (possible compositions? induction pairs present? tag
    particles available?) and baking one in would make it silently part of
    the motif's definition rather than part of the analysis.

    Without it, raw counts are returned and the caller must not compare
    them across prompts of different lengths.
    """
    counts = relay_strength(t, thresholds)
    if normalize_by is None:
        return {k: float(v) for k, v in counts.items()}
    out = {}
    for k, v in counts.items():
        denom = normalize_by.get(k, 0)
        out[k] = float(v) / denom if denom else float("nan")
    return out


# ---------------------------------------------------------------------------
# Null comparison
# ---------------------------------------------------------------------------

def compare_against_nulls(
    observed: float,
    nulls: Dict[str, Sequence[float]],
    sigma_threshold: float = 2.0,
) -> dict:
    """
    Adjudicate one observed motif statistic against N1/N2/N3.

    `nulls` maps null name -> that null's sampled values. A missing null is
    not treated as passed: `verdict` is REFUSED when a gating null is
    absent, because "we did not compute the offset-matched null" and "the
    offset-matched null was cleared" are the two readings that must never
    be confused. Phase 6's P6-I2 was broken for precisely this reason.

    Returns a dict with per-null summaries (from core.nulls.nsigma_verdict),
    the overall verdict, and the reason, so the artifact records what was
    read and whether it passed (standing rule 3).
    """
    per_null = {}
    for name in ALL_NULLS:
        if name in nulls and len(nulls[name]) > 0:
            per_null[name] = nsigma_verdict(observed, np.asarray(nulls[name]),
                                            sigma_threshold=sigma_threshold)

    missing = [n for n in GATING_NULLS if n not in per_null]
    if missing:
        verdict, reason = "REFUSED", f"gating null(s) not supplied: {missing}"
    else:
        failed = [n for n in GATING_NULLS
                  if not (per_null[n]["significant"] and per_null[n]["z_score"] > 0)]
        if failed:
            verdict = "FALSIFIED"
            reason = f"did not clear {failed} at {sigma_threshold}σ"
        else:
            verdict = "CONFIRMED"
            reason = f"cleared {list(GATING_NULLS)} at {sigma_threshold}σ"

    out = {
        "observed": float(observed),
        "nulls": per_null,
        "gating_nulls": list(GATING_NULLS),
        "verdict": verdict,
        "reason": reason,
        "sigma_threshold": sigma_threshold,
    }
    if "N3" in per_null:
        # Informational, never gating: separates "content and offset jointly
        # required" from "either alone suffices".
        out["n3_reading"] = (
            "content_and_offset_jointly_required"
            if per_null["N3"]["significant"] and per_null["N3"]["z_score"] > 0
            else "either_alone_suffices"
        )
    return out


# ---------------------------------------------------------------------------
# P-I3: the cross-head association, with its mandatory control arm
# ---------------------------------------------------------------------------

def cross_head_association(
    motif_rate: Dict[tuple, float],
    behavioral_score: Dict[tuple, float],
    is_induction_head: Dict[tuple, bool],
    independence_source: str,
) -> dict:
    """
    P-I3. Spearman correlation between per-head motif rate and behavioural
    induction score, reported separately for induction heads and
    non-induction heads.

    The control arm is not optional and this function will not run without
    it. Reporting the motif rate only among induction heads would read as
    confirmation whatever the number was — the same error the P-T1
    amendment was written to prevent, pre-empted here rather than corrected
    afterwards. If non-induction heads carry the motif at the same rate,
    the motif is a property of the activations rather than of the
    classification, and the bridge fails for this phenomenon.

    `independence_source` must name which of the three sources carries the
    association — "two_stage", "force_channel" or "particle_event" — per
    PREDICTIONS.md's Phase 7 adjudication constraint 2. It is a required
    argument, not a keyword with a default, because a result that cannot
    name one has measured the behavioural induction score twice and the
    correlation is tautological.
    """
    valid = {"two_stage", "force_channel", "particle_event"}
    if independence_source not in valid:
        raise ValueError(
            f"independence_source must be one of {sorted(valid)}; got "
            f"{independence_source!r}. A P-I3 result that cannot name the "
            "source of its independence from the behavioural score has "
            "measured one quantity twice — see PREDICTIONS.md, Phase 7 "
            "adjudication constraint 2."
        )

    present = sorted(set(motif_rate) & set(behavioral_score) & set(is_induction_head))
    if not present:
        return {"verdict": "REFUSED", "reason": "no heads present in all three inputs",
                "independence_source": independence_source}

    # NaN rates are real: relay_rate_by_head_pair returns NaN for a head pair
    # whose denominator was zero (no compositions were possible), which is
    # "undefined", not "zero". Feeding those to a correlation would silently
    # poison it, and imputing 0.0 would count an impossible composition as a
    # failed one. Drop them and say how many were dropped.
    shared = [k for k in present
              if np.isfinite(motif_rate[k]) and np.isfinite(behavioral_score[k])]
    n_undefined = len(present) - len(shared)
    if not shared:
        return {"verdict": "REFUSED",
                "reason": f"every shared head had an undefined rate (n={n_undefined})",
                "n_undefined_dropped": n_undefined,
                "independence_source": independence_source}

    def _spearman(keys):
        if len(keys) < 3:
            return float("nan"), len(keys)
        x = _rank(np.array([motif_rate[k] for k in keys], dtype=np.float64))
        y = _rank(np.array([behavioral_score[k] for k in keys], dtype=np.float64))
        if x.std() == 0 or y.std() == 0:
            return float("nan"), len(keys)
        return float(np.corrcoef(x, y)[0, 1]), len(keys)

    ind_keys = [k for k in shared if is_induction_head[k]]
    non_keys = [k for k in shared if not is_induction_head[k]]

    rho_ind, n_ind = _spearman(ind_keys)
    rho_non, n_non = _spearman(non_keys)
    mean_ind = float(np.mean([motif_rate[k] for k in ind_keys])) if ind_keys else float("nan")
    mean_non = float(np.mean([motif_rate[k] for k in non_keys])) if non_keys else float("nan")

    if not non_keys:
        verdict, reason = "REFUSED", "control arm empty: no non-induction heads to compare against"
    elif np.isnan(rho_ind):
        verdict, reason = "REFUSED", f"too few induction heads (n={n_ind}) or no rank variance"
    elif mean_non >= mean_ind:
        verdict = "FALSIFIED"
        reason = ("non-induction heads carry the motif at least as often "
                  f"({mean_non:.3f} vs {mean_ind:.3f}) — the motif is a property of "
                  "the activations, not of the classification")
    elif rho_ind > 0:
        verdict = "CONFIRMED"
        reason = f"rho={rho_ind:.3f} among induction heads, control arm lower"
    else:
        verdict, reason = "FALSIFIED", f"rho={rho_ind:.3f} among induction heads (not positive)"

    return {
        "spearman_induction_heads": rho_ind,
        "spearman_non_induction_heads": rho_non,
        "mean_motif_rate_induction": mean_ind,
        "mean_motif_rate_non_induction": mean_non,
        "n_induction_heads": n_ind,
        "n_non_induction_heads": n_non,
        "n_undefined_dropped": n_undefined,
        "independence_source": independence_source,
        "verdict": verdict,
        "reason": reason,
    }


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not distort Spearman."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # average tied ranks
    for v in np.unique(x):
        tie = x == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------

def motif_counts_payload(
    counts: Dict[str, dict],
    nulls: Dict[str, dict],
    verdicts: Dict[str, dict],
    degenerate_prompts: List[dict],
    force_cutoff: Optional[dict],
) -> dict:
    """
    Assemble motif_counts.json against its registered contract
    (core.artifacts REGISTRY["phase7"]["motif_counts"]).

    `degenerate_prompts` is a list, and an empty list is a claim — that
    every prompt was checked and none was degenerate — not an omission.
    `force_cutoff` records the retention threshold the counts were computed
    under together with whether it was placed or calibrated; None means no
    thinning was applied, which is different from "we did not record it".
    """
    return {
        "motif_alphabet_version": ALPHABET_VERSION,
        "counts": counts,
        "nulls": nulls,
        "verdicts": verdicts,
        "degenerate_prompts": degenerate_prompts,
        "force_cutoff": force_cutoff if force_cutoff is not None
                        else {"mode": "none", "status": "no_thinning_applied"},
        "thresholds": dict(DEFAULTS),
        "threshold_status": dict(THRESHOLD_STATUS),
    }
