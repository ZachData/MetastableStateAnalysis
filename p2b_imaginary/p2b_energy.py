"""
p2b_imaginary/p2b_energy.py — the one place Phase 2b counts an energy
violation.

Why this module exists
----------------------
Before this, Phase 2b counted violations in three places with two different
rules, neither of which was the project's rule:

  - `run_2i.load_phase1_events` recomputed violations inline with an
    ABSOLUTE threshold (`e - e_prev < -1e-6`) and a rank gate of 3.0,
    reading `geometry.json["effective_rank"]` (raw mode).
  - `rotational_rescaled.rescaled_trajectory_component` and
    `.original_trajectory_metrics` each repeated the same absolute rule
    against their own locally-computed effective rank, which used
    UNSQUARED singular values where `core.metrics.effective_rank` uses
    squared ones.
  - `core.metrics.energy_violation_severity` — the project's actual rule —
    is RELATIVE (`(E_prev - E_curr)/|E_prev| > 1e-3`), and the gate is
    `core.config.DEGENERATE_RANK_THRESHOLD` (=2), not 3.0.

Consequence: Phase 2b's `n_original` was not Phase 1's count and not Phase
2's count for the same run, so no elimination rate it produced was
comparable to anything outside this phase. This is status-1's defect D7
("two different violation counters, both labelled violations") and D8
(the rank threshold) landing inside Phase 2b.

The gate quantity, and why it is `normed` here
----------------------------------------------
Phase 1 gates on RAW effective rank (`analysis_p1.py:216`). Phase 2b cannot:
the rescaled frames are constructed by applying a matrix to the residual
stream and re-projecting to the sphere, so their "raw" norms are an artifact
of the rescaling, not a property of the model. Raw rank is only defined for
the `original` frame.

So Phase 2b gates on NORMED rank throughout, and says so in every record it
writes (`gate_kind`). The consequence is stated rather than hidden: Phase
2b's `n_original` may differ from Phase 1's count for the same run, and
`cross_check_against_phase1` exists to surface that difference instead of
letting it sit unnoticed inside an elimination rate.

status-1 D1 and D8 both argue the project should move to a normed gate
anyway ("the gate should probably read normed rank"). If that happens,
`gate_kind="normed_rank"` becomes the project default and this note becomes
a historical one.

The denominator is a result, not a detail
-----------------------------------------
`n_transitions_scored` is returned next to `n_violations` on every frame.
Two frames that scored different numbers of transitions — because one of
them truncated (see `rotational_rescaled.n_valid_layers`) or because the
rank gate fired on different layers — have incomparable violation counts,
and any ratio between them is meaningless. `frames_comparable` is the guard
that refuses rather than warns.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from core.metrics import (
    ENERGY_VIOLATION_REL_TOL,
    effective_rank,
    gram_matrix,
    interaction_energies_batched,
    l2_normalize,
)

# Deferred, because core.config imports torch and transformers at module
# level and Phase 2b's math is otherwise torch-free (same pattern as
# p2_eigenspectra/weights.py's deferred core.pythia_weights import).
#
# The fallback is None, not 2.0. A silent numeric default here would
# reintroduce exactly the hardcoded-threshold problem this module exists to
# remove; `resolve_rank_gate` raises instead, so a torch-free session must
# pass the value explicitly and can never accidentally score against a
# different gate than the pipeline does.
try:  # pragma: no cover - depends on session
    from core.config import DEGENERATE_RANK_THRESHOLD as _CONFIG_RANK_GATE
except Exception:  # pragma: no cover
    _CONFIG_RANK_GATE = None


#: Which quantity the degeneracy gate reads.
GATE_KINDS = ("normed_rank", "raw_rank", "none")

#: Phase 2b's gate. See the module docstring for why it is not "raw_rank".
DEFAULT_GATE_KIND = "normed_rank"


def resolve_rank_gate(gate_threshold: Optional[float]) -> float:
    """
    The degeneracy threshold, from the caller or from core.config.

    Raises rather than defaulting when neither is available — a violation
    count scored against an unknown gate is not a measurement.
    """
    if gate_threshold is not None:
        return float(gate_threshold)
    if _CONFIG_RANK_GATE is None:
        raise RuntimeError(
            "p2b_energy: core.config.DEGENERATE_RANK_THRESHOLD is unavailable "
            "(torch-free session?) and no gate_threshold was passed. Pass it "
            "explicitly; do not let this fall back to a literal."
        )
    return float(_CONFIG_RANK_GATE)


# ---------------------------------------------------------------------------
# Per-layer scalars for one trajectory
# ---------------------------------------------------------------------------

def trajectory_scalars(
    normed: np.ndarray,
    beta_values: Sequence[float],
    n_valid_layers: Optional[int] = None,
) -> dict:
    """
    Energies, effective rank and inner-product summaries for one already-
    sphere-projected trajectory.

    Parameters
    ----------
    normed         : (n_layers, n_tokens, d), each row unit-norm.
    beta_values    : betas to evaluate.
    n_valid_layers : score only layers [0, n_valid_layers). Layers at or
                     beyond it are NaN — a truncated rescaling did not
                     produce them and must not be silently counted as
                     "no violation there". Defaults to all layers.

    Every quantity is computed with the canonical `core.metrics` function,
    not a local reimplementation:
      - energies       : interaction_energies_batched (the same
                         exp(beta*G).sum()/(2*beta*n*n) Phase 1 uses)
      - effective rank : effective_rank(..., mode="normed"), which uses
                         SQUARED singular values. The pre-rewrite Phase 2b
                         code used unsquared ones, which is a different
                         statistic that happens to have the same name.
    """
    normed = np.asarray(normed)
    n_layers, n_tokens, _ = normed.shape
    if n_valid_layers is None:
        n_valid_layers = n_layers
    n_valid_layers = int(max(0, min(n_valid_layers, n_layers)))

    betas = [float(b) for b in beta_values]
    energies = {b: np.full(n_layers, np.nan) for b in betas}
    eff_rank = np.full(n_layers, np.nan)
    ip_mean = np.full(n_layers, np.nan)
    ip_mass = np.full(n_layers, np.nan)

    iu = np.triu_indices(n_tokens, k=1)

    for L in range(n_valid_layers):
        X = np.asarray(normed[L], dtype=np.float64)
        G = X @ X.T
        ips = G[iu]
        ip_mean[L] = float(ips.mean()) if ips.size else float("nan")
        ip_mass[L] = float((ips > 0.9).mean()) if ips.size else float("nan")

        for b, e in interaction_energies_batched(G, betas).items():
            energies[float(b)][L] = e

        eff_rank[L] = effective_rank(X, mode="normed")

    return {
        "energies": energies,
        "effective_rank": eff_rank,
        "ip_mean": ip_mean,
        "ip_mass_near_1": ip_mass,
        "n_layers": n_layers,
        "n_valid_layers": n_valid_layers,
    }


# ---------------------------------------------------------------------------
# Violation counting
# ---------------------------------------------------------------------------

def count_violations(
    energies: Sequence[float],
    gate_values: Optional[Sequence[float]] = None,
    *,
    rel_tol: float = ENERGY_VIOLATION_REL_TOL,
    gate_kind: str = DEFAULT_GATE_KIND,
    gate_threshold: Optional[float] = None,
) -> dict:
    """
    Count energy-monotonicity violations along one beta-series, using the
    project's relative rule and the project's degeneracy gate.

    A transition L-1 -> L is SCORED when both energies are finite and the
    gate passes at L. It is a VIOLATION when, in addition,
    (E[L-1] - E[L]) / |E[L-1]| > rel_tol.

    Returns
    -------
    dict with:
      violation_layers      : list[int] — the L of each violating transition
      n_violations          : int
      n_transitions_scored  : int — the DENOMINATOR. Two frames with
                              different values here are not comparable; see
                              `frames_comparable`.
      n_transitions_gated   : int — scored-eligible transitions rejected by
                              the rank gate
      n_transitions_nan     : int — rejected for a non-finite energy, which
                              on a rescaled frame means truncation
      sum_severity          : float — sum of relative drops over violations
      max_severity          : float
      rel_drops             : list[float] — per-transition, NaN where unscored
      rule                  : dict — the exact rule applied, recorded so a
                              stale count can be identified rather than
                              inferred
    """
    if gate_kind not in GATE_KINDS:
        raise ValueError(
            f"count_violations: gate_kind must be one of {GATE_KINDS}, got {gate_kind!r}"
        )

    E = np.asarray(energies, dtype=np.float64)
    n = E.shape[0]

    if gate_kind == "none":
        thresh = None
        gate = None
    else:
        thresh = resolve_rank_gate(gate_threshold)
        if gate_values is None:
            raise ValueError(
                f"count_violations: gate_kind={gate_kind!r} requires gate_values. "
                "Pass gate_kind='none' to score without a gate, explicitly."
            )
        gate = np.asarray(gate_values, dtype=np.float64)
        if gate.shape[0] != n:
            raise ValueError(
                f"count_violations: gate_values has length {gate.shape[0]}, "
                f"energies has length {n}"
            )

    viol_layers: list = []
    rel_drops = np.full(max(n - 1, 0), np.nan)
    n_scored = 0
    n_gated = 0
    n_nan = 0

    for L in range(1, n):
        e_prev, e_curr = E[L - 1], E[L]
        if not (np.isfinite(e_prev) and np.isfinite(e_curr)):
            n_nan += 1
            continue
        if gate is not None:
            g = gate[L]
            if not np.isfinite(g):
                # A NaN gate value on a layer with finite energies means the
                # gate quantity was never computed there. Refuse rather than
                # pass: a missing gate is not a passing gate.
                n_nan += 1
                continue
            if g < thresh:
                n_gated += 1
                continue
        n_scored += 1
        ref = max(abs(e_prev), 1e-12)
        drop = -(e_curr - e_prev) / ref
        rel_drops[L - 1] = drop
        if drop > rel_tol:
            viol_layers.append(int(L))

    scored_drops = rel_drops[np.isfinite(rel_drops)]
    viol_drops = scored_drops[scored_drops > rel_tol]

    return {
        "violation_layers": viol_layers,
        "n_violations": int(len(viol_layers)),
        "n_transitions_scored": int(n_scored),
        "n_transitions_gated": int(n_gated),
        "n_transitions_nan": int(n_nan),
        "sum_severity": float(viol_drops.sum()) if viol_drops.size else 0.0,
        "max_severity": float(viol_drops.max()) if viol_drops.size else 0.0,
        "rel_drops": [None if not np.isfinite(v) else float(v) for v in rel_drops],
        "rule": {
            "rel_tol": float(rel_tol),
            "gate_kind": gate_kind,
            "gate_threshold": None if gate_kind == "none" else float(thresh),
            "criterion": "(E_prev - E_curr)/|E_prev| > rel_tol",
        },
    }


def count_violations_all_betas(
    scalars: dict,
    *,
    rel_tol: float = ENERGY_VIOLATION_REL_TOL,
    gate_kind: str = DEFAULT_GATE_KIND,
    gate_threshold: Optional[float] = None,
) -> dict:
    """`count_violations` over every beta in a `trajectory_scalars` result."""
    gate_values = (None if gate_kind == "none"
                   else scalars["effective_rank"])
    return {
        float(beta): count_violations(
            E, gate_values,
            rel_tol=rel_tol, gate_kind=gate_kind, gate_threshold=gate_threshold,
        )
        for beta, E in scalars["energies"].items()
    }


# ---------------------------------------------------------------------------
# Comparability guard
# ---------------------------------------------------------------------------

def frames_comparable(count_a: dict, count_b: dict) -> dict:
    """
    Whether two frames' violation counts can be divided by each other.

    They cannot when they scored different numbers of transitions. That
    happens for two reasons, both real on Pythia:

      1. Truncation. `expm(-S)` for a symmetric S with positive eigenvalues
         grows without bound; the cumulative product overflows and the
         rescaled trajectory stops early. `expm(-A)` for antisymmetric A is
         ORTHOGONAL and never overflows. So the signed-only frame is the one
         that can truncate and the rotation-only frame is the one that
         cannot — which means `elim_signed = 1.0` is precisely the value an
         early-truncating S-frame produces for free. This is Phase 2's
         verification item V1, in the phase where it does the most damage.
      2. The rank gate firing on different layers, because the rescaled
         trajectory has a different effective-rank profile. This one is not
         hypothetical and it scales with ||V||: applying `e^{-V}` to a
         trajectory contracts it directionally, so for a large-norm OV the
         rescaled frames drop below the degeneracy threshold at layers where
         the original frame does not. Measured on a synthetic sweep at d=12:
         with entries at N(0,1) the original frame scores 5 transitions and
         the signed frame scores 2, which the pre-rewrite code would have
         reported as `elim_signed = 0.75` produced entirely by the gate.
         Study A's OV spectral-norm confound (partial rho to -0.71,
         status-2.md blocker 2) is the same quantity, so this is the regime
         Phase 2 already knows the models are in.

    Returns dict(comparable, reason, n_scored_a, n_scored_b). `reason` is
    None when comparable.
    """
    na = int(count_a["n_transitions_scored"])
    nb = int(count_b["n_transitions_scored"])
    if count_a["rule"] != count_b["rule"]:
        return {"comparable": False, "reason": "different_counting_rule",
                "n_scored_a": na, "n_scored_b": nb}
    if na != nb:
        return {"comparable": False, "reason": "different_transitions_scored",
                "n_scored_a": na, "n_scored_b": nb}
    if na == 0:
        return {"comparable": False, "reason": "no_transitions_scored",
                "n_scored_a": na, "n_scored_b": nb}
    return {"comparable": True, "reason": None, "n_scored_a": na, "n_scored_b": nb}


def elimination_rate(count_orig: dict, count_rescaled: dict) -> dict:
    """
    Fraction of violations a rescaled frame removes, or an explicit refusal.

    Two refusals the pre-rewrite `_elim_rate` could not express, both of
    which it reported as the float 0.0:

      - `n_original == 0`. 90 of Phase 2 Study B's 243 Pythia runs are
        `no_violations`, and steps 8-64 are clean on all 9 prompts. Scoring
        those as `elim = 0.0` and feeding them to a majority vote makes the
        phase return `rotation_neutral` by vacuity at exactly the
        checkpoints where the theorem HOLDS. `rate` is None here.
      - Incomparable denominators (see `frames_comparable`).

    `rate` is NOT clipped at zero. A negative rate means the rescaling made
    monotonicity worse, which is the ALBERT overcorrection caveat in
    status-2b and the unclipped quantity Phase 2's verification item V2 asks
    for — `analysis_p2.py:153`'s `max(0, ...)` destroys it there, so Phase 2b
    is where it can be recovered.
    """
    comp = frames_comparable(count_orig, count_rescaled)
    n_orig = int(count_orig["n_violations"])
    n_resc = int(count_rescaled["n_violations"])

    if not comp["comparable"]:
        return {"rate": None, "status": comp["reason"],
                "n_original": n_orig, "n_rescaled": n_resc, **comp}
    if n_orig == 0:
        return {"rate": None, "status": "no_violations_to_eliminate",
                "n_original": n_orig, "n_rescaled": n_resc, **comp}

    return {
        "rate": float((n_orig - n_resc) / n_orig),
        "status": "ok",
        "n_original": n_orig,
        "n_rescaled": n_resc,
        **comp,
    }


# ---------------------------------------------------------------------------
# Cross-check against Phase 1's own count
# ---------------------------------------------------------------------------

def cross_check_against_phase1(
    p2b_counts: dict,
    phase1_violation_layers: dict,
) -> dict:
    """
    Compare Phase 2b's `original`-frame count against the count Phase 1
    wrote for the same run, per beta.

    They are expected to differ — Phase 1 gates on raw effective rank and
    Phase 2b on normed (module docstring). The point is that the difference
    is a recorded number rather than an invisible term inside every
    elimination rate downstream. A large disagreement means the gate, not
    the rescaling, is doing the work.

    Parameters
    ----------
    p2b_counts              : {beta: count_violations(...) result}
    phase1_violation_layers : {beta: list[int]} as recorded by Phase 1

    Returns {beta: {n_p2b, n_phase1, delta, only_p2b, only_phase1}}.
    """
    out = {}
    for beta, cnt in p2b_counts.items():
        b = float(beta)
        p1 = set(int(x) for x in (phase1_violation_layers.get(b)
                                  or phase1_violation_layers.get(str(b))
                                  or []))
        p2b = set(int(x) for x in cnt["violation_layers"])
        out[b] = {
            "n_p2b": len(p2b),
            "n_phase1": len(p1),
            "delta": len(p2b) - len(p1),
            "only_p2b": sorted(p2b - p1),
            "only_phase1": sorted(p1 - p2b),
        }
    return out


def sphere_project(activations) -> np.ndarray:
    """
    Row-normalize to the unit sphere, via `core.metrics.l2_normalize`.

    Phase 2b's inputs are already normed (`p1_io._save_activations` writes
    `layernorm_to_sphere(x)`, which is plain L2 normalization despite the
    name). This exists so that the one place Phase 2b re-normalizes after a
    rescaling calls the same function, rather than doing it inline — see
    `core/frames.py`'s design commitment 2, "a call site that normalizes
    inline is a call site that cannot be audited."
    """
    arr = np.asarray(activations)
    if arr.ndim == 3:
        return np.stack([l2_normalize(arr[i]) for i in range(arr.shape[0])])
    return l2_normalize(arr)


__all__ = [
    "GATE_KINDS",
    "DEFAULT_GATE_KIND",
    "resolve_rank_gate",
    "trajectory_scalars",
    "count_violations",
    "count_violations_all_betas",
    "frames_comparable",
    "elimination_rate",
    "cross_check_against_phase1",
    "sphere_project",
    "gram_matrix",
]
