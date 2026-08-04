"""
core/sink_audit.py — the measurement that decides P1, rather than the
argument that decides P1 (DESIGN_pythia_frames.md, policy item P1).

The question
------------
NeoX tokenizers do not prepend BOS, so position 0 becomes the attention
sink and carries a norm one to two orders above the bulk. `core.polar.
norm_stats` already SURFACES that. Nothing decides about it, and the
decision is not free: the trained-vs-random energy contrast is a Blog-1
continuity claim, and if the sink dominates E_beta then part of that
contrast is a sink contrast.

The statistic, and why a bare share is not it
---------------------------------------------
For any n x n pairwise matrix, the pairs touching index 0 are row 0 union
column 0: 2n - 1 of n^2 entries. On the ~264-token battery that is 0.76% by
construction, so "position 0 accounts for 3% of the energy" reads as small
and is in fact a 4x enrichment. Every share here is therefore reported
alongside its structural baseline and as an ENRICHMENT ratio. A sink that
is doing nothing sits at 1.0.

The decision rule, stated before the numbers (principle S2)
-----------------------------------------------------------
Three outcomes, checked in this order:

  policy_is_load_bearing  the set of energy-violation layers differs
                          between the two policies for any beta. The P1
                          choice then changes the headline claim (Blog 1's
                          monotonicity falsification), so exclusion becomes
                          the default for energy and clustering and BOTH
                          arms are reported everywhere.
  sink_dominates          no violation-layer set changes, but the energy
                          enrichment exceeds ENRICHMENT_THRESHOLD at some
                          layer. Exclusion is the default; inclusion is kept
                          for exactness checks only.
  policy_is_cosmetic      neither. Record the choice, report one arm, keep
                          the diff on file.

Checking the violation-layer sets first is deliberate: it asks whether the
policy moves the CONCLUSION, which is the only question that has to be
answered before the gated runs.

Note for the checkpoint sweep (item 11)
---------------------------------------
The sink EMERGES during training. A fixed exclusion rule removes a genuine
sink at step 143,000 and an ordinary token at step 0, so the rule is
constant while what it removes is not. Run this audit per checkpoint, not
once on the final model; `sink_emergence` reduces a sweep of audits to the
curve that shows it.
"""

from __future__ import annotations

import numpy as np

from core.frames import pos0_mask
from core.metrics import (
    _as_numpy,
    gram_matrix,
    interaction_energy,
    energy_violation_severity,
)
from core.polar import particle_norms, norm_stats, raw_gram

#: Enrichment above which one particle is judged to dominate a pairwise
#: quantity. 4.0 means the sink carries four times the mass its share of the
#: pairs would give it. Sweep it if a result sits near the line; do not
#: tune it after seeing the numbers.
ENRICHMENT_THRESHOLD: float = 4.0

SINK_INDEX: int = 0


# ---------------------------------------------------------------------------
# Pairwise mass attributable to one index
# ---------------------------------------------------------------------------

def structural_pair_share(n: int, index: int = SINK_INDEX) -> float:
    """
    Share of an n x n pairwise matrix's entries that touch `index`:
    (2n - 1) / n^2. The floor every share below must be read against.
    """
    if n <= 0:
        return float("nan")
    return float((2 * n - 1) / (n * n))


def index_mass_share(M, index: int = SINK_INDEX, absolute: bool = True) -> dict:
    """
    Share of a pairwise matrix's total mass carried by row/column `index`.

    absolute : take |M| first. Required for a raw Gram, whose off-diagonal
               entries change sign; harmless for exp(beta * G), which is
               positive everywhere.

    Returns dict(share, baseline, enrichment, total).
    """
    A = np.abs(_as_numpy(M).astype(np.float64, copy=False)) if absolute else \
        _as_numpy(M).astype(np.float64, copy=False)
    n = A.shape[0]
    if n == 0 or A.shape[0] != A.shape[1]:
        raise ValueError(f"index_mass_share: expected square matrix, got {A.shape}")
    total = float(A.sum())
    if total <= 0 or not np.isfinite(total):
        return dict(share=float("nan"), baseline=structural_pair_share(n, index),
                    enrichment=float("nan"), total=total)
    touched = float(A[index, :].sum() + A[:, index].sum() - A[index, index])
    share = touched / total
    baseline = structural_pair_share(n, index)
    return dict(share=share, baseline=baseline,
                enrichment=float(share / baseline) if baseline > 0 else float("nan"),
                total=total)


# ---------------------------------------------------------------------------
# Energy under a stated policy
# ---------------------------------------------------------------------------

def energy_under_policy(activations, beta: float,
                        policy: str = "included",
                        index: int = SINK_INDEX) -> float:
    """
    E_beta with the pos0 policy applied BEFORE the n^2 normalisation.

    This wrapper exists because `interaction_energy` divides by n^2, so
    masking after the fact is not the same quantity: excluding one token
    from a 264-token battery changes the denominator by 0.8% on its own.
    Any comparison across policies, arms, or checkpoints must mask first.
    """
    X = _as_numpy(activations).astype(np.float64, copy=False)
    keep = pos0_mask(X.shape[0], policy)
    if index != SINK_INDEX:
        keep = np.ones(X.shape[0], dtype=bool)
        if policy == "excluded":
            keep[index] = False
    return interaction_energy(X[keep], beta)


# ---------------------------------------------------------------------------
# Per-layer audit
# ---------------------------------------------------------------------------

def sink_audit_layer(activations, beta_values, index: int = SINK_INDEX,
                     top_k: int = 5) -> dict:
    """
    One layer's answer to "how much of this is position 0?".

    Reports the sink's weight in three places it can distort a result:
      norm       — polar.norm_stats, the direct magnitude view
      raw_gram   — where the sink enters through r_i * r_j
      exp(beta G)— where it enters through direction only, since E_beta is
                   defined on the sphere; the two need not agree, and a
                   large norm ratio with an enrichment near 1 means the sink
                   distorts the polar view and not the energy
    """
    X = _as_numpy(activations).astype(np.float64, copy=False)
    n = X.shape[0]
    norms = particle_norms(X)
    ns = norm_stats(norms, top_k=top_k)

    G = gram_matrix(X)
    out = {
        "n": int(n),
        "index": int(index),
        "norm_over_median": float(norms[index] / max(float(np.median(norms)), 1e-12)),
        "index_is_top_norm": bool(len(ns["top_outlier_indices"]) > 0
                                  and ns["top_outlier_indices"][0] == index),
        "norm_stats": ns,
        "raw_gram": index_mass_share(raw_gram(X), index, absolute=True),
        "energy": {},
    }
    for beta in beta_values:
        b = float(beta)
        W = np.exp(b * G)
        share = index_mass_share(W, index, absolute=False)
        share["energy_included"] = energy_under_policy(X, b, "included", index)
        share["energy_excluded"] = energy_under_policy(X, b, "excluded", index)
        denom = max(abs(share["energy_included"]), 1e-12)
        share["relative_shift"] = float(
            (share["energy_excluded"] - share["energy_included"]) / denom
        )
        out["energy"][b] = share
    return out


# ---------------------------------------------------------------------------
# Trajectory audit — the part that actually decides
# ---------------------------------------------------------------------------

def sink_audit_trajectory(acts_by_layer, beta_values,
                          index: int = SINK_INDEX,
                          rel_tol: float | None = None) -> dict:
    """
    Audit every layer, then ask the decisive question: does the pos0 policy
    change WHICH layers violate energy monotonicity?

    acts_by_layer : sequence of (n_tokens, d) arrays, one per layer, in
                    depth order. Same object Phase 1 already holds.

    Returns dict(per_layer, monotonicity, verdict, ...).
    """
    layers = [ _as_numpy(a) for a in acts_by_layer ]
    per_layer = [sink_audit_layer(X, beta_values, index) for X in layers]

    kwargs = {} if rel_tol is None else {"rel_tol": rel_tol}
    monotonicity = {}
    for beta in beta_values:
        b = float(beta)
        inc = [energy_under_policy(X, b, "included", index) for X in layers]
        exc = [energy_under_policy(X, b, "excluded", index) for X in layers]
        sev_i = energy_violation_severity(inc, **kwargs)
        sev_e = energy_violation_severity(exc, **kwargs)
        monotonicity[b] = {
            "included": {"violation_layers": sev_i["violation_layers"],
                         "n_violations": sev_i["n_violations"],
                         "sum_severity": sev_i["sum_severity"]},
            "excluded": {"violation_layers": sev_e["violation_layers"],
                         "n_violations": sev_e["n_violations"],
                         "sum_severity": sev_e["sum_severity"]},
            "violation_layers_agree": (sev_i["violation_layers"]
                                       == sev_e["violation_layers"]),
            "energies_included": [float(v) for v in inc],
            "energies_excluded": [float(v) for v in exc],
        }

    max_enrichment = float(np.nanmax([
        r["energy"][float(b)]["enrichment"]
        for r in per_layer for b in beta_values
    ])) if per_layer and len(beta_values) else float("nan")

    load_bearing = any(not m["violation_layers_agree"] for m in monotonicity.values())
    dominates = np.isfinite(max_enrichment) and max_enrichment >= ENRICHMENT_THRESHOLD

    if load_bearing:
        verdict = "policy_is_load_bearing"
    elif dominates:
        verdict = "sink_dominates"
    else:
        verdict = "policy_is_cosmetic"

    return {
        "per_layer": per_layer,
        "monotonicity": monotonicity,
        "max_energy_enrichment": max_enrichment,
        "max_raw_gram_enrichment": float(np.nanmax(
            [r["raw_gram"]["enrichment"] for r in per_layer]
        )) if per_layer else float("nan"),
        "max_norm_over_median": float(np.nanmax(
            [r["norm_over_median"] for r in per_layer]
        )) if per_layer else float("nan"),
        "enrichment_threshold": ENRICHMENT_THRESHOLD,
        "verdict": verdict,
        "recommended_default": ("included" if verdict == "policy_is_cosmetic"
                                else "excluded"),
    }


def sink_emergence(audits_by_step: dict) -> dict:
    """
    Reduce {training_step: sink_audit_trajectory(...)} to the curve that
    shows the sink forming.

    This is the item-11 interaction P1 needs stated explicitly: a constant
    exclusion rule removes different things at different points in training,
    so "does the verdict change across the sweep" is itself a result and not
    a robustness footnote. Sorted by step; pair the x-axis with
    `core.checkpoint_frames.step_x`.
    """
    steps = sorted(int(s) for s in audits_by_step)
    return {
        "steps": steps,
        "max_norm_over_median": [audits_by_step[s]["max_norm_over_median"] for s in steps],
        "max_energy_enrichment": [audits_by_step[s]["max_energy_enrichment"] for s in steps],
        "verdict": [audits_by_step[s]["verdict"] for s in steps],
        "verdict_changes": len({audits_by_step[s]["verdict"] for s in steps}) > 1,
    }


def sink_audit_summary_lines(audit: dict) -> list:
    lines = [
        "Position-0 audit (policy P1):",
        f"  max norm / median        : {audit['max_norm_over_median']:.1f}x",
        f"  max raw-Gram enrichment  : {audit['max_raw_gram_enrichment']:.2f}x",
        f"  max energy enrichment    : {audit['max_energy_enrichment']:.2f}x "
        f"(threshold {audit['enrichment_threshold']:.1f}x)",
    ]
    for beta, m in audit["monotonicity"].items():
        tag = "same" if m["violation_layers_agree"] else "DIFFERENT"
        lines.append(
            f"  beta={beta:g}: violation layers {tag} — "
            f"included {m['included']['violation_layers']} vs "
            f"excluded {m['excluded']['violation_layers']}"
        )
    lines.append(f"  verdict                  : {audit['verdict']}")
    lines.append(f"  recommended default      : {audit['recommended_default']}")
    return lines
