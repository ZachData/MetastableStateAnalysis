"""
tools/dry_run_p_t1_p_m1.py — P-T1 and P-M1 on inputs whose answer is known.

`claims/EVALUABILITY.md`'s queue owes every converted row a run on an input
whose correct verdict is fixed a priori. `CLAIM-C` (2026-08-25), `P-ST1`
(2026-08-26), `P6-R2`/`P6-R4` (2026-08-26) and `CLAIM-B`/`P-I1` (2026-08-27)
have had theirs. These two are the seventh and eighth of the nine, and after
them only `P-S1` is owed one -- which has its own record,
`claims/audits/p_s1_dry_run.json`.

WHY THESE TWO IN ONE FILE, WHICH IS NOT THE REASON CLAIM-B AND P-I1 SHARED ONE

`CLAIM-B` and `P-I1` share a CONSTRUCTION, so one dry run covered both by
construction. `P-T1` and `P-M1` do not: one is a label permutation on a rate
difference over heads, the other a permutation of a violation series against a
regime score over layers. They are here together for two other reasons, and
both are findings rather than convenience.

**They have the same defect, reached independently.** Both report
`core.nulls.p_from_null`'s `resolution` -- 1/(n_perm + 1) -- as though it were
the smallest p the design can express. Both statistics are DISCRETE, so the
null puts a lump of mass exactly on the observed value and the real floor is
set by the data's own marginals. On a PERFECT input, P-T1 at five heads with
two candidates returns 0.109 and P-M1 at twelve layers with one violation
returns 0.091, both against a reported resolution of 0.0005. Two independently
written constructions, the same mistake, and no test was failing on either.

**And they are the only two entries whose shared instrument feeds ONE claim.**
Both are H-OPERATOR's, and both classify the same head's Wq, Wk and W_OV --
P-T1 on the eigenstructure of V and the QK form, P-M1 on M's symmetry and V's
alignment with it. A defect in which head's weights are which moves both, and
because they sit under the SAME claim their e-values multiply into one
product. `P6-R2`/`P6-R4` record their shared projector for this reason and
`CLAIM-B`/`P-I1` record their shared estimator; these two recorded nothing
until now, and they are the pair where it matters most.

WHAT THE FLOOR IS, FOR EACH, AND WHY IT IS EXACT

P-T1: the statistic is monotone in how many trimodal heads land in the
candidate arm, and the null holds both marginals fixed, so that count is
hypergeometric and the floor is the tail at the most extreme table the
marginals admit. Exact, by `math.comb`.

P-M1: permutations that only swap EQUAL violation values give the same
correlation, so the floor is `prod_v (multiplicity of v)! / n!` -- which for a
binary series with T violations in n layers is 1/C(n, T). A tied regime score
would make the true floor larger, so this is a LOWER BOUND and refusing on it
can never turn away a result that would have cleared alpha; the same shape of
argument as `P-ST1`'s 2m bound (`POPPER_PLAN.md` 6m).

Neither floor contains a draw count. More permutations do not move them; more
heads, more layers or more violations do -- which is what makes each of them a
requirement on the run rather than on the call.

FIVE ARMS

A. `known_answer`   -- both gates on inputs whose correct verdict is fixed a
                       priori, in both directions.
B. `the_floor`      -- the finding: the exact floor against the reported
                       resolution, and what a PERFECT input returns, over the
                       design sizes each entry can actually have.
C. `refusal_costs_nothing` -- every attainable p at a refused configuration,
                       ENUMERATED rather than sampled, so the claim is proved
                       rather than measured.
D. `validity`       -- H0 rejection rates CONDITIONAL ON EMISSION, before and
                       after the refusal, because changing what refuses changes
                       what that conditioning conditions on.
E. `shared_instrument` -- what moves both entries at once, and the arithmetic
                       of what that does to H-OPERATOR's product.

WHAT THIS DELIBERATELY DOES NOT DO

It adjudicates nothing: the heads and layers are synthetic, no Phase 2d run
artifact is in this repository, and `claims/adjudications/` stays empty. It
does not touch either statistic, either null or either alternative -- only the
floor each one reports and refuses on.

RUN IT

    python3 -m tools.dry_run_p_t1_p_m1 --write
    python3 -m tools.dry_run_p_t1_p_m1 --check
    python3 -m tools.dry_run_p_t1_p_m1 --summary

The generation cost is measured on every write and stored as `elapsed_seconds`
rather than quoted here, where it would go stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from p2d_operator_activation.gradient_flow_condition import (
    P_M1_ALTERNATIVE,
    p_m1_attainable_floor,
    p_value_p_m1,
)
from p2d_operator_activation.table1_predictions import (
    P_T1_ALTERNATIVE,
    P_T1_TARGET_MODES,
    p_t1_attainable_floor,
    p_value_p_t1,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "claims" / "audits" / "p_t1_p_m1_dry_run.json"

RECORD_SCHEMA_VERSION = 1

#: The two files every number here depends on.
P_T1_PATH = ROOT / "p2d_operator_activation" / "table1_predictions.py"
P_M1_PATH = ROOT / "p2d_operator_activation" / "gradient_flow_condition.py"

ALPHA = 0.05

#: Head counts P-T1 can actually be handed. A transformer layer has tens of
#: heads and Phase 2d classifies them per layer, so the small end is the live
#: case rather than a stress test -- which is the whole reason the floor
#: matters here.
T1_DESIGNS: Tuple[Tuple[int, int], ...] = (
    (2, 3), (3, 5), (3, 9), (4, 8), (5, 7), (6, 18), (8, 16), (12, 36))

#: (n_layers, n_violations). gpt2-large has 36 layers and pythia-1.4b has 24;
#: `UPDATE_PLAN.md` 5.9 makes the violation series a per-boundary INDICATOR, so
#: a model that mostly does not violate gives a small count -- again the live
#: case rather than a stress test.
M1_DESIGNS: Tuple[Tuple[int, int], ...] = (
    (6, 1), (12, 1), (12, 2), (12, 3), (24, 1), (24, 2), (36, 2), (49, 1))

N_PERM = 2000
N_REPS = 200

#: Family-wise level for this file's own numeric checks, tighter than the
#: registry's alpha for the reason POPPER_PLAN.md 6n records: these bounds are
#: applied to proportions in a REGENERATED artifact, and a bound that fails
#: once in twenty regenerations when nothing is wrong is one that gets re-run
#: rather than read.
CHECK_FAMILY_ALPHA = 0.01

_SEED = 20260827


def _two_sided_z(alpha_family: float, n_cells: int) -> float:
    """
    The z a one-sided per-cell bound needs to hold a family-wise error rate.
    Bonferroni over the cells, derived from that count rather than placed.
    """
    from math import erf, sqrt
    target = 1.0 - alpha_family / max(int(n_cells), 1)
    lo, hi = 0.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Synthetic inputs with a planted answer
# ---------------------------------------------------------------------------

def t1_heads(n_cand: int, n_ctrl: int, cand_trimodal: int,
             ctrl_trimodal: int) -> List[dict]:
    """Heads whose bandwidth scan says exactly what this call asks it to."""
    out = []
    for i in range(n_cand):
        out.append({"row2_candidate": True,
                    "stability": {"stable_n_modes":
                                  P_T1_TARGET_MODES if i < cand_trimodal else 2}})
    for i in range(n_ctrl):
        out.append({"row2_candidate": False,
                    "stability": {"stable_n_modes":
                                  P_T1_TARGET_MODES if i < ctrl_trimodal else 2}})
    return out


def m1_regimes(n_layers: int, heads_per_layer: int = 4, sep: float = 1.0,
               rng=None) -> List[dict]:
    """Per-head regime distances, monotone in the layer index."""
    rng = rng or np.random.default_rng(0)
    return [{"layer": l, "head": h,
             "regime_distance": float(l * sep + 0.01 * rng.standard_normal()),
             "in_gradient_flow_regime": False}
            for l in range(n_layers) for h in range(heads_per_layer)]


def m1_violations(n_layers: int, n_viol: int, where: str) -> np.ndarray:
    """`far` puts the violations on the highest-regime layers -- P-M1's H1."""
    v = np.zeros(n_layers, dtype=float)
    if where == "far":
        v[-n_viol:] = 1.0
    elif where == "near":
        v[:n_viol] = 1.0
    else:
        rng = np.random.default_rng(0)
        v[rng.choice(n_layers, size=n_viol, replace=False)] = 1.0
    return v


# ---------------------------------------------------------------------------
# A. the answer is known
# ---------------------------------------------------------------------------

def known_answer(alpha: float) -> dict:
    """
    Both gates on inputs whose correct verdict is fixed a priori, in both
    directions -- because a construction that cannot return p = 1 with the
    arms reversed is not testing the direction it claims to.
    """
    rows = []

    # P-T1 at a design big enough that the floor is not the binding thing.
    for label, cand_tri, ctrl_tri, expect in (
            ("every candidate trimodal, no control", 8, 0, "reject"),
            ("no candidate trimodal, every control", 0, 16, "p = 1"),
            ("both arms trimodal at the same rate", 4, 8, "no rejection"),
    ):
        r = p_value_p_t1(t1_heads(8, 16, cand_tri, ctrl_tri),
                         n_perm=N_PERM, seed=1)
        p = r["p_value"]
        rows.append({
            "entry": "P-T1", "input": label, "expected": expect,
            "p_value": p, "reason": r.get("reason"),
            "as_expected": bool(
                p is not None and p <= alpha if expect == "reject"
                else p is not None and p >= 0.99 if expect == "p = 1"
                else p is not None and p > alpha),
        })

    # P-M1 at a design big enough that the floor is not the binding thing.
    for label, where, expect in (
            ("violations on the highest-regime layers", "far", "reject"),
            ("violations on the lowest-regime layers", "near", "p = 1"),
    ):
        rng = np.random.default_rng(2)
        r = p_value_p_m1(m1_regimes(24, rng=rng), m1_violations(24, 4, where),
                         n_perm=N_PERM, seed=1)
        p = r["p_value"]
        rows.append({
            "entry": "P-M1", "input": label, "expected": expect,
            "p_value": p, "reason": r.get("reason"),
            "as_expected": bool(
                p is not None and p <= alpha if expect == "reject"
                else p is not None and p >= 0.99),
        })

    # And the aggregate-disagreement refusal, which predates this pass and is
    # checked here because a refusal no input reaches is a refusal nothing has
    # checked.
    rng = np.random.default_rng(3)
    reg = m1_regimes(12, heads_per_layer=3, rng=rng)
    for r_ in reg:                       # make min and max disagree with mean
        if r_["head"] == 0:
            r_["regime_distance"] = float(11 - r_["layer"])
    dis = p_value_p_m1(reg, m1_violations(12, 3, "far"), n_perm=200, seed=1)
    rows.append({
        "entry": "P-M1", "input": "the three aggregates disagree in sign",
        "expected": "refused", "p_value": dis["p_value"],
        "reason": dis.get("reason"),
        "as_expected": bool(dis["p_value"] is None
                            and "SIGN" in (dis.get("reason") or "")),
    })

    return {
        "_what": ("both gates on inputs whose correct verdict is fixed a "
                  "priori, in both directions, at design sizes where the "
                  "attainable floor is not the binding constraint."),
        "alpha": alpha,
        "rows": rows,
        "every_row_as_expected": bool(all(r["as_expected"] for r in rows)),
    }


# ---------------------------------------------------------------------------
# B. the finding
# ---------------------------------------------------------------------------

def the_floor(alpha: float) -> dict:
    """
    The exact attainable floor against the reported resolution, and what a
    PERFECT input returns at each design size.

    "Perfect" means the most extreme arrangement the marginals admit: every
    candidate trimodal and no control for P-T1, every violation on the
    highest-regime layers for P-M1. If the design cannot reject THERE it
    cannot reject anywhere, and the number it used to return instead was a
    p above alpha -- which on a prediction reads as evidence against it.
    """
    t1_rows = []
    for n_cand, n_ctrl in T1_DESIGNS:
        floor = p_t1_attainable_floor(n_cand, n_ctrl, n_cand)
        r = p_value_p_t1(t1_heads(n_cand, n_ctrl, n_cand, 0),
                         n_perm=N_PERM, seed=1)
        t1_rows.append({
            "n_candidates": n_cand, "n_controls": n_ctrl,
            "n_heads": n_cand + n_ctrl,
            "design_floor": floor["attainable_floor"],
            "sampling_resolution": 1.0 / (N_PERM + 1.0),
            "smallest_expressible_p": max(floor["attainable_floor"],
                                          1.0 / (N_PERM + 1.0)),
            "which_binds": ("the marginals"
                            if floor["attainable_floor"] > 1.0 / (N_PERM + 1.0)
                            else "the draw count"),
            "floor_over_resolution":
                floor["attainable_floor"] * (N_PERM + 1.0),
            "sufficient": floor["sufficient"],
            "perfect_input_p": r["p_value"],
            "perfect_input_refused": bool(r["p_value"] is None),
        })

    m1_rows = []
    for n_layers, n_viol in M1_DESIGNS:
        v = m1_violations(n_layers, n_viol, "far")
        floor = p_m1_attainable_floor(v)
        rng = np.random.default_rng(4)
        r = p_value_p_m1(m1_regimes(n_layers, rng=rng), v,
                         n_perm=N_PERM, seed=1)
        m1_rows.append({
            "n_layers": n_layers, "n_violations": n_viol,
            "design_floor": floor["attainable_floor"],
            "closed_form_1_over_C": 1.0 / comb(n_layers, n_viol),
            "sampling_resolution": 1.0 / (N_PERM + 1.0),
            "smallest_expressible_p": max(floor["attainable_floor"],
                                          1.0 / (N_PERM + 1.0)),
            "which_binds": ("the marginals"
                            if floor["attainable_floor"] > 1.0 / (N_PERM + 1.0)
                            else "the draw count"),
            "floor_over_resolution":
                floor["attainable_floor"] * (N_PERM + 1.0),
            "sufficient": floor["sufficient"],
            "perfect_input_p": r["p_value"],
            "perfect_input_refused": bool(r["p_value"] is None),
        })

    worst_t1 = max(r["floor_over_resolution"] for r in t1_rows)
    worst_m1 = max(r["floor_over_resolution"] for r in m1_rows)
    return {
        "_what": ("the exact attainable floor against the resolution "
                  "`core.nulls.p_from_null` reports, and what a PERFECT input "
                  "returns, at design sizes each entry can actually have."),
        "_the_finding": (
            "both entries reported 1/(n_perm+1) as their floor. Both "
            "statistics are DISCRETE, so the design's own floor is set by the "
            "marginals and no number of permutations moves it."),
        "_and_which_one_binds": (
            "the smallest p a RUN can express is the MAX of the two, and they "
            "bind at opposite ends. At a small design the marginals bind by a "
            "factor of hundreds and the draw count is irrelevant; at a large "
            "one the design floor falls BELOW 1/(n_perm+1) and the draw count "
            "binds again, which is why the reported resolution was never "
            "wrong everywhere. It was wrong exactly where these two entries "
            "live -- tens of heads, tens of layers, few violations."),
        "_why_a_perfect_input": (
            "a floor is a claim about what the design can express, and the "
            "way to check a claim like that is to hand the design the most "
            "extreme arrangement its marginals admit and look at what comes "
            "back. Where the floor exceeds alpha, what used to come back was "
            "a p ABOVE alpha -- which on a prediction reads as evidence "
            "against it rather than as a design that could not speak."),
        "alpha": alpha,
        "n_perm": N_PERM,
        "p_t1": t1_rows,
        "p_m1": m1_rows,
        "worst_floor_over_resolution_p_t1": float(worst_t1),
        "worst_floor_over_resolution_p_m1": float(worst_m1),
        "p_t1_has_designs_that_cannot_reject": bool(
            any(not r["sufficient"] for r in t1_rows)),
        "p_m1_has_designs_that_cannot_reject": bool(
            any(not r["sufficient"] for r in m1_rows)),
        "every_insufficient_design_is_refused": bool(
            all(r["perfect_input_refused"] for r in t1_rows + m1_rows
                if not r["sufficient"])
            and all(not r["perfect_input_refused"] for r in t1_rows + m1_rows
                    if r["sufficient"])),
    }


# ---------------------------------------------------------------------------
# C. what the refusal costs
# ---------------------------------------------------------------------------

def refusal_costs_nothing(alpha: float) -> dict:
    """
    Every attainable p at a refused configuration, ENUMERATED.

    `POPPER_PLAN.md` 6l had to MEASURE that CLAIM-C's informative-row refusal
    costs no power, because there the floor and the p come from different
    code. Here they do not: the floor IS the smallest value of the same
    discrete p, so the claim can be enumerated rather than sampled -- and
    6m's distinction between a refusal that costs nothing by construction and
    one measured to cost nothing is exactly what this arm records.

    For P-T1 the support is every attainable count of trimodal candidates; for
    P-M1 the smallest attainable p is the floor by construction, so the
    enumeration is over the P-T1 side and the P-M1 side is the bound.
    """
    rows = []
    for n_cand, n_ctrl in T1_DESIGNS:
        N, K, T = n_cand + n_ctrl, n_cand, n_cand
        floor = p_t1_attainable_floor(K, n_ctrl, T)
        if floor["sufficient"]:
            continue
        # Every table the marginals admit, and its exact one-sided p.
        total = comb(N, K)
        ps = []
        lo = max(0, K - (N - T))
        for m in range(lo, min(K, T) + 1):
            tail = sum(comb(T, j) * comb(N - T, K - j)
                       for j in range(m, min(K, T) + 1)
                       if 0 <= K - j <= N - T)
            ps.append(tail / total)
        rows.append({
            "entry": "P-T1", "n_candidates": K, "n_controls": n_ctrl,
            "attainable_floor": floor["attainable_floor"],
            "n_attainable_p_values": len(ps),
            "smallest_attainable_p": float(min(ps)),
            "any_clears_alpha": bool(any(p <= alpha for p in ps)),
        })

    m1_rows = []
    for n_layers, n_viol in M1_DESIGNS:
        floor = p_m1_attainable_floor(m1_violations(n_layers, n_viol, "far"))
        if floor["sufficient"]:
            continue
        m1_rows.append({
            "entry": "P-M1", "n_layers": n_layers, "n_violations": n_viol,
            "attainable_floor": floor["attainable_floor"],
            "any_clears_alpha": bool(floor["attainable_floor"] <= alpha),
        })

    all_rows = rows + m1_rows
    return {
        "_what": ("at every refused configuration, whether ANY arrangement of "
                  "the data could have cleared alpha."),
        "_by_construction_not_by_measurement": (
            "the floor is the smallest value of the same discrete p that the "
            "gate would report, so this is an enumeration and not a "
            "simulation. 6l's refusal for CLAIM-C had to be re-scored against "
            "a counterfactual because there the floor and the p come from "
            "different code; 6m's could not cost a verdict by construction. "
            "This one is the second of that kind, and saying which it is is "
            "the point -- a measured zero and a proved zero are not the same "
            "claim."),
        "_and_the_smallest_attainable_p_is_the_floor": (
            "for P-M1 there is nothing to enumerate: the floor is defined as "
            "the smallest attainable p, and it is a LOWER BOUND on it when the "
            "regime score has ties, so refusing on it can only under-refuse."),
        "alpha": alpha,
        "rows": all_rows,
        "n_refused_configurations": len(all_rows),
        "costs_no_verdict_anywhere": bool(
            len(all_rows) > 0 and not any(r["any_clears_alpha"] for r in all_rows)),
    }


# ---------------------------------------------------------------------------
# D. validity, conditional on emission
# ---------------------------------------------------------------------------

def validity(alpha: float, reps: Optional[int] = None) -> dict:
    """
    H0 rejection rates CONDITIONAL ON EMISSION, before and after the refusal.

    `POPPER_PLAN.md` 6g: a gate can look calibrated BY REFUSING, so the rate
    that governs a ledger is the one among runs that emitted. And 6m adds the
    part that bites here -- changing what refuses changes what that
    conditioning conditions on -- so both columns are reported rather than
    only the new one.
    """
    n = N_REPS if reps is None else int(reps)      # resolved here, not in the
    rng = np.random.default_rng(_SEED)             # signature; see 6h/6m/6n
    rows = []

    for n_cand, n_ctrl in T1_DESIGNS:
        N = n_cand + n_ctrl
        after = emitted = 0
        for _ in range(n):
            # H0: trimodality is independent of the row-2 label. Realised by
            # drawing the trimodal set uniformly at random, which is exactly
            # what the null permutes.
            tri_total = int(rng.integers(0, N + 1))
            idx = rng.permutation(N)[:tri_total]
            flags = np.zeros(N, dtype=bool)
            flags[idx] = True
            heads = []
            for i in range(N):
                heads.append({
                    "row2_candidate": i < n_cand,
                    "stability": {"stable_n_modes":
                                  P_T1_TARGET_MODES if flags[i] else 2}})
            r = p_value_p_t1(heads, n_perm=500, seed=int(rng.integers(1 << 30)))
            if r["p_value"] is not None:
                after += int(r["p_value"] <= alpha)
                emitted += 1
        rows.append({
            "entry": "P-T1", "design": f"{n_cand} candidates, {n_ctrl} controls",
            "n_draws": n, "n_emitted": emitted,
            "reject_conditional_on_emission":
                (after / emitted) if emitted else None,
            "emission_rate": emitted / n,
        })

    for n_layers, n_viol in M1_DESIGNS:
        emitted = hits = 0
        base = m1_regimes(n_layers, rng=np.random.default_rng(9))
        for _ in range(n):
            v = np.zeros(n_layers)
            v[rng.permutation(n_layers)[:n_viol]] = 1.0    # H0: placed at random
            r = p_value_p_m1(base, v, n_perm=500,
                             seed=int(rng.integers(1 << 30)))
            if r["p_value"] is not None:
                hits += int(r["p_value"] <= alpha)
                emitted += 1
        rows.append({
            "entry": "P-M1", "design": f"{n_layers} layers, {n_viol} violations",
            "n_draws": n, "n_emitted": emitted,
            "reject_conditional_on_emission":
                (hits / emitted) if emitted else None,
            "emission_rate": emitted / n,
        })

    rates = [r["reject_conditional_on_emission"] for r in rows
             if r["reject_conditional_on_emission"] is not None]
    se = float(np.sqrt(alpha * (1 - alpha) / n))
    z = _two_sided_z(CHECK_FAMILY_ALPHA, max(len(rates), 1))
    return {
        "_what": ("H0 rejection rate conditional on the gate emitting, at "
                  "every design size above. The H0 is exact in both: P-T1's "
                  "trimodal set is drawn uniformly at random, which is what "
                  "its null permutes, and P-M1's violations are placed "
                  "uniformly at random over layers."),
        "_why_conditional": (
            "a gate can look calibrated BY REFUSING -- 6g. The ledger only "
            "ever receives runs that emitted, so that is the rate which "
            "governs it, and the emission rate is reported beside it because "
            "the refusal added this pass is what moved it."),
        "alpha": alpha,
        "n_reps_per_cell": n,
        "standard_error_at_alpha": se,
        "rows": rows,
        "bound_in_standard_errors": z,
        "bound": float(alpha + z * se),
        "_the_bound": (
            f"alpha plus {z:.2f} standard errors: the Bonferroni allowance for "
            f"the cells this section tests at a family-wise "
            f"{CHECK_FAMILY_ALPHA}. A per-cell 1.96 fails once in twenty "
            f"regenerations with the null exactly nominal, which is a property "
            f"of the check rather than of the null (POPPER_PLAN.md 6n)."),
        "range": [min(rates), max(rates)] if rates else None,
        "holds": bool(max(rates) <= alpha + z * se) if rates else False,
        "some_designs_never_emit": bool(
            any(r["emission_rate"] == 0.0 for r in rows)),
    }


# ---------------------------------------------------------------------------
# E. what moves both at once
# ---------------------------------------------------------------------------

def shared_instrument(alpha: float) -> dict:
    """
    The common-cause dependence neither entry recorded, and the arithmetic of
    why it matters more here than anywhere else it has come up.

    This is structural rather than measured, and it is in the record because
    `EVALUABILITY.md`'s opening argument makes it non-optional: a claim's E is
    the product of its predictions' e-values, so two factors that a single
    defect moves together are not two factors.
    """
    return {
        "_what": ("what a single instrument defect moves, and what that does "
                  "to H-OPERATOR's product."),
        "entries": ["P-T1", "P-M1"],
        "claim": "H-OPERATOR",
        "shared_input": (
            "the same head's Wq, Wk and W_OV, and the same per-head extraction "
            "that decides which head those belong to. P-T1 classifies on the "
            "eigenstructure of V and the QK form; P-M1 on M = Q^T K's symmetry "
            "and V's alignment with it. The classifications differ; the "
            "objects classified do not."),
        "why_it_matters_more_here": (
            "P6-R2 and P6-R4 share a projector and CLAIM-B and P-I1 share an "
            "estimator, and both pairs record it. But CLAIM-B and P-I1 sit "
            "under DIFFERENT claims, so a shared defect does not multiply "
            "inside one product. P-T1 and P-M1 are both H-OPERATOR's. Two "
            "e-values that one defect moves together, multiplied into one "
            "claim's E, is the specific way a product inflates without "
            "anyone editing a number."),
        "precedent_for_the_defect_class": (
            "POPPER_PLAN.md 6h: a Schur block mislabelling swapping U_neg and "
            "U_A was a live alternative explanation for four months and took "
            "an audit to rule out. Which head's weights are which is the same "
            "class of question and has had no audit."),
        "not_measured_here": (
            "this arm asserts a dependence rather than measuring its size. "
            "Measuring it needs the real extraction path and a planted "
            "mislabelling -- 6h's arm S, one phase over -- and that is an "
            "audit rather than a dry run. It is named here so the registry "
            "can record it, which is what the other two pairs did."),
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Assembling
# ---------------------------------------------------------------------------

def build_record(seed: int = _SEED) -> dict:
    t0 = time.time()
    rec = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "_what": ("P-T1 and P-M1 run on inputs whose correct answer is known "
                  "a priori, and the measurement that gave both an attainable "
                  "floor."),
        "_why": ("EVALUABILITY.md's queue owes every converted row a run on an "
                 "input whose verdict is fixed in advance. These two are the "
                 "seventh and eighth of nine."),
        "_not": ("not evidence about any transformer and not an adjudication. "
                 "The heads and layers are synthetic; what is being checked is "
                 "the gate."),
        "generated_by": "python3 -m tools.dry_run_p_t1_p_m1 --write",
        "p_t1_file": str(P_T1_PATH.relative_to(ROOT)),
        "p_t1_sha256": _sha256(P_T1_PATH),
        "p_m1_file": str(P_M1_PATH.relative_to(ROOT)),
        "p_m1_sha256": _sha256(P_M1_PATH),
        "alpha": ALPHA,
        "alternatives": {"P-T1": P_T1_ALTERNATIVE, "P-M1": P_M1_ALTERNATIVE},
        "seed": int(seed),
        "known_answer": known_answer(ALPHA),
        "the_floor": the_floor(ALPHA),
        "refusal_costs_nothing": refusal_costs_nothing(ALPHA),
        "validity": validity(ALPHA),
        "shared_instrument": shared_instrument(ALPHA),
    }
    rec["elapsed_seconds"] = round(time.time() - t0, 1)
    return rec


def check_record(path: Path = RECORD_PATH) -> List[str]:
    """
    Is the committed record still about the files on disk, and does it still
    support the change it was the evidence for?
    """
    problems: List[str] = []
    if not path.exists():
        return [f"{path} is missing; regenerate with "
                f"`python3 -m tools.dry_run_p_t1_p_m1 --write`"]
    try:
        rec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path} is not valid JSON: {exc}"]

    if rec.get("schema_version") != RECORD_SCHEMA_VERSION:
        problems.append(
            f"schema_version {rec.get('schema_version')} != "
            f"{RECORD_SCHEMA_VERSION}; regenerate with --write")
    for key, described in (("p_t1", P_T1_PATH), ("p_m1", P_M1_PATH)):
        if not described.exists():
            problems.append(f"{described} is missing")
            continue
        if rec.get(f"{key}_sha256") != _sha256(described):
            problems.append(
                f"{described.name} has changed since this record was written "
                f"(sha256 {_sha256(described)[:12]} on disk vs "
                f"{str(rec.get(f'{key}_sha256'))[:12]} on record); rerun "
                f"--write rather than editing the hash")

    ka = rec.get("known_answer", {})
    if not ka.get("every_row_as_expected"):
        problems.append(
            "a gate did not return the known answer on an input whose correct "
            "verdict is fixed a priori; that is a criterion not meaning what "
            "it says rather than a calibration question")

    fl = rec.get("the_floor", {})
    if not (fl.get("p_t1") and fl.get("p_m1")):
        problems.append("the_floor is missing one of its two tables")
    else:
        for entry in ("p_t1", "p_m1"):
            worst = fl.get(f"worst_floor_over_resolution_{entry}")
            if worst is None or worst <= 1.0:
                problems.append(
                    f"{entry.upper().replace('_', '-')}'s attainable floor no "
                    f"longer exceeds the resolution it used to report "
                    f"(ratio {worst}). The floor added on 2026-08-27 rests on "
                    f"that gap; an artifact that no longer shows it does not "
                    f"support the change it is the evidence for")
        if not fl.get("every_insufficient_design_is_refused"):
            problems.append(
                "a design whose floor exceeds alpha still emitted a p-value, "
                "or a design whose floor clears alpha was refused. Either way "
                "the refusal is not the one this record describes")

    rc = rec.get("refusal_costs_nothing", {})
    if not rc.get("rows"):
        problems.append(
            "refusal_costs_nothing enumerated nothing. A sweep with no refused "
            "configuration reports success while being incapable of reporting "
            "anything else -- POPPER_PLAN.md 6l's `costs_no_power is None, "
            "never True`, in its other form")
    elif not rc.get("costs_no_verdict_anywhere"):
        problems.append(
            "some arrangement at a refused configuration WOULD have cleared "
            "alpha, so the refusal is turning away a reachable verdict")

    va = rec.get("validity", {})
    if not va.get("rows"):
        problems.append("validity has no rows")
    elif not va.get("holds"):
        problems.append(
            f"an H0 rejection rate conditional on emission reaches "
            f"{va.get('range')} against alpha {va.get('alpha')}")

    si = rec.get("shared_instrument", {})
    if si.get("claim") != "H-OPERATOR" or len(si.get("entries", [])) != 2:
        problems.append(
            "the shared-instrument block no longer names both entries and "
            "their claim; that dependence is the reason their e-values must "
            "not be read as two independent factors")
    return problems


def print_summary(rec: dict) -> None:
    print(f"P-T1: {rec['p_t1_file']}  sha256 {rec['p_t1_sha256'][:12]}")
    print(f"P-M1: {rec['p_m1_file']}  sha256 {rec['p_m1_sha256'][:12]}")
    print(f"alpha {rec['alpha']}   generated in {rec.get('elapsed_seconds')}s\n")

    ka = rec["known_answer"]
    print("=== A. the answer is known ===")
    for r in ka["rows"]:
        p = "REFUSED" if r["p_value"] is None else f"{r['p_value']:.4f}"
        print(f"  {r['entry']:6s} {r['input'][:44]:44s} expect "
              f"{r['expected']:14s} p {p:>8}  ok {r['as_expected']}")

    fl = rec["the_floor"]
    print(f"\n=== B. the attainable floor against the reported resolution "
          f"(n_perm {fl['n_perm']}) ===")
    print(f"  P-T1  {'cand':>5} {'ctrl':>5} {'design floor':>13} "
          f"{'resolution':>11} {'binds':>13} {'perfect input':>14}")
    for r in fl["p_t1"]:
        p = "REFUSED" if r["perfect_input_p"] is None else f"{r['perfect_input_p']:.4f}"
        print(f"        {r['n_candidates']:>5} {r['n_controls']:>5} "
              f"{r['design_floor']:>13.6f} {r['sampling_resolution']:>11.5f} "
              f"{r['which_binds']:>13} {p:>14}")
    print(f"  P-M1  {'layer':>5} {'viol':>5} {'design floor':>13} "
          f"{'resolution':>11} {'binds':>13} {'perfect input':>14}")
    for r in fl["p_m1"]:
        p = "REFUSED" if r["perfect_input_p"] is None else f"{r['perfect_input_p']:.4f}"
        print(f"        {r['n_layers']:>5} {r['n_violations']:>5} "
              f"{r['design_floor']:>13.6f} {r['sampling_resolution']:>11.5f} "
              f"{r['which_binds']:>13} {p:>14}")

    rc = rec["refusal_costs_nothing"]
    print(f"\n=== C. what the refusal costs ({rc['n_refused_configurations']} "
          f"refused configurations, enumerated) ===")
    for r in rc["rows"]:
        print(f"  {r['entry']:6s} floor {r['attainable_floor']:.4f}  "
              f"any arrangement clears alpha: {r['any_clears_alpha']}")
    print(f"  costs no verdict anywhere: {rc['costs_no_verdict_anywhere']}")

    va = rec["validity"]
    print(f"\n=== D. H0 rate conditional on emission ({va['n_reps_per_cell']} "
          f"draws a cell, SE {va['standard_error_at_alpha']:.3f}) ===")
    for r in va["rows"]:
        rate = ("n/a" if r["reject_conditional_on_emission"] is None
                else f"{r['reject_conditional_on_emission']:.3f}")
        print(f"  {r['entry']:6s} {r['design']:28s} emitted "
              f"{r['emission_rate']:.2f}  reject|emitted {rate:>5}")
    print(f"  holds against {va['bound']:.4f} "
          f"({va['bound_in_standard_errors']:.2f} SE): {va['holds']}   "
          f"range {va['range']}")

    si = rec["shared_instrument"]
    print(f"\n=== E. what moves both at once ===")
    print(f"  {' and '.join(si['entries'])} both under {si['claim']}")
    print(f"  {si['why_it_matters_more_here'][:200]}...")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--write", action="store_true",
                    help="run it and write the record; it prints and stores "
                         "its own elapsed time")
    ap.add_argument("--check", action="store_true",
                    help="verify the committed record without re-running")
    ap.add_argument("--summary", action="store_true",
                    help="print the committed record")
    ap.add_argument("--out", type=Path, default=RECORD_PATH)
    ap.add_argument("--seed", type=int, default=_SEED)
    args = ap.parse_args(argv)

    if args.write:
        rec = build_record(seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rec, indent=2, sort_keys=False) + "\n")
        print(f"wrote {args.out}")
        print_summary(rec)
        return 0
    if args.check:
        problems = check_record(args.out)
        for p in problems:
            print(f"PROBLEM: {p}")
        if not problems:
            print(f"{args.out.name}: clean, and describes the files on disk")
        return 1 if problems else 0
    if args.summary:
        if not args.out.exists():
            print(f"{args.out} is missing; run --write")
            return 1
        print_summary(json.loads(args.out.read_text()))
        return 0
    ap.error("nothing to do: pass --write, --check or --summary")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
