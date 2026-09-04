"""
tools/p_i1_attainable_floor.py — step 1 of the order, for P-I1's relay series.

`claims/EVALUABILITY.md` prescribes one order for every row that names a matched
control: *compute the attainable floor, name what the statistic degenerates on,
check what the measurement grid contributes, and only then build the control.*
`P-AB1` (`POPPER_PLAN.md` 6q) and `P-I3` (6s) were built that way and on both of
them the first three steps changed the design before any control existed.

For P-I1's relay-count null, HANDOFF.md's session of 2026-09-03 recorded steps 2
and 3 as done and step 1 as NOT done:

  * step 2 — what the statistic degenerates on: the prompt's own induction-pair
    supply, r = +0.9958 across the eight battery prompts at step 54000.
  * step 3 — what the measurement grid contributes: everything, on the
    twelve-step CLAIM-B grid, where all 116 heads' change centroids were ONE
    number. The five registered log-spaced fills inside (1000, 54000) fixed it:
    79 distinct centroids on the 19-step grid.

This is step 1, and it is deliberately run BEFORE the control. Nothing here
designs a relay-count null; what it does is say what such a null must leave
behind for a p-value to exist at all, which is a constraint the control has to
satisfy and is cheaper to know first.

WHAT IT FOUND, AND THE TWO ARE INDEPENDENT

**The gate cannot run on the axis the pipeline builds.** `formation_curve_
payload` takes its head axis from the BEHAVIOURAL series, which is dense over
all 384 heads (24 layers x 16), and zero-fills the relay side. But
`paired_colocation_arm` calls `change_profile` on every unit with no per-unit
skip, and `change_profile` REFUSES a series with no rise. On the real sweep 116
heads carry relays and 268 never do, so the arm refuses on the first all-zero
head and `p_value_p_i1` returns no p-value at all — for a reason that is a
property of the model's head count and the axis rule, not of the data. The
refusal message names no arm, no head and no unit count; it says "the series has
no rise anywhere in the sweep". Restricted to the 116 forming heads the same
input emits — which is what says the refusal is about the AXIS and not about the
relay series. Whether it emits at its FLOOR is arm C's question, and there it
does.

**The floor the arm reported was the wrong half.** `paired_colocation_arm`
reported `1 / n_draws`, the draw-count floor, alone. The statistic is
`-mean|ca - cb[p]|`, and permuting units WITHIN a class of equal locations
leaves it exactly unchanged, so every pairing ties a coset of order `prod(m!)`
and no input can express a p below `prod(m!) / n!`. Measured on the registered
19-step grid with nine of ten units sharing one location: reported floor
0.000500 against an attainable 0.100000 — a factor of 200, above alpha, and
emitted as a p-value with no refusal (the one realised sampled draw came back
0.101449). `core.changepoint_colocation.pairing_floor_report` now carries both
halves and says which binds; `POPPER_PLAN.md` 6m is the same defect in
`p7_motifs/steering_gate.py` and this is that lesson arriving in the shared
estimator.

On the REAL 116 heads the tie half is 10^-148 and does not bind: the fix costs
this row nothing today. It binds on the set a relay-count null LEAVES, which is
why it is step 1 rather than a footnote to step 4.

FIVE ARMS

A. `dense_axis`         — the wiring, on the real relay series.
B. `tie_structure`      — the multiplicity of the real per-head locations, per
                          relay owner. 79 distinct centroids is not 79 even
                          classes.
C. `perfect_input`      — what a perfect input returns on the real head sets,
                          beside what the arm says it can.
D. `the_defect`         — the reported floor against the realised floor, on the
                          grid and the tie structure that produce the gap.
E. `floor_vs_survivors` — the closed form over (n, k) and where the real
                          multiset sits: what a relay-count null may remove.

WHAT THIS DELIBERATELY DOES NOT DO

It builds no relay-count null and registers nothing. HANDOFF.md §5 records the
shape such a null would have — a degree-preserving rewiring within each
(context, layer, head) — and records that it is the author's decision. It stays
the author's decision.

It adjudicates nothing. `claims/adjudications/` stays empty, `P-I1` stays
`needs-null`, and the behavioural arm of the curve is still not computed: the
B-side series used here is synthetic and is only ever the side the refusals do
not come from. Every number about the RELAY side is measured on the 19 tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.changepoint_colocation import (
    ColocationRefused,
    REGISTERED_P_I1_SWEEP,
    _SEED,
    _alpha,
    change_profile,
    interval_midpoints,
    pairing_floor_report,
    paired_colocation_arm,
)
from p7_motifs.formation_curve import formation_curve_payload
from p7_motifs.formation_gate import P_I1_RELAY_OWNER, p_value_p_i1

REPO = Path(__file__).resolve().parent.parent
RECORD_PATH = REPO / "claims" / "audits" / "p_i1_attainable_floor.json"
RECORD_SCHEMA_VERSION = 1

CONSTRUCTION_PATH = REPO / "core" / "changepoint_colocation.py"
FORMATION_GATE_PATH = REPO / "p7_motifs" / "formation_gate.py"
FORMATION_CURVE_PATH = REPO / "p7_motifs" / "formation_curve.py"

#: Written by `tools/run/curve.py`. Generated bulk, git-ignored, so the record
#: carries their hashes rather than assuming a reader has them.
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
CURVE_JSON = DATA / "analysis" / "curve.json"
SERIES_JSON = DATA / "analysis" / "formation_series.json"

#: pythia-410m. Not read from a config: this arm is about what the axis rule
#: does at a KNOWN head count, and the count is the thing being varied against.
N_LAYERS, N_HEADS = 24, 16

OWNERS = ("tag_writer", "matcher", "both")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_series() -> Tuple[List[int], Dict[str, Dict[Tuple[int, int], List[float]]]]:
    """The per-head relay series `tools/run/curve.py` writes."""
    if not SERIES_JSON.exists():
        raise SystemExit(
            f"{SERIES_JSON} is missing. It is generated bulk: run\n"
            f"  python tools/run/curve.py\n"
            f"which writes it beside curve.json (~3 min over the 19 tables).")
    raw = json.loads(SERIES_JSON.read_text())
    steps = [int(s) for s in raw["steps"]]
    if tuple(steps) != tuple(REGISTERED_P_I1_SWEEP):
        raise SystemExit(
            f"{SERIES_JSON} is on a {len(steps)}-checkpoint grid that is not "
            f"the registered sweep {list(REGISTERED_P_I1_SWEEP)}. A floor "
            f"measured on the wrong grid is a floor for a different design.")
    series = {o: {tuple(int(x) for x in k.split(",")): [float(v) for v in vals]
                  for k, vals in raw["series"][o].items()}
              for o in OWNERS}
    return steps, series


def _b_side(a: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    A perfect input's B side: a DIFFERENT series with the SAME change location
    at every unit.

    `change_profile` normalises the change mass, so a positive rescaling leaves
    the centroid bit-identical while `np.array_equal` is False — which matters,
    because `p_value_p_i1` refuses two identical series as the tautology
    `PREDICTIONS.md`'s second adjudication constraint names. Perfect
    co-location is the input a floor is defined by; the tautology is a different
    thing and the gate must keep refusing it.
    """
    return [[2.0 * float(x) for x in v] for v in a]


# ---------------------------------------------------------------------------
# A. the wiring: the axis the pipeline builds
# ---------------------------------------------------------------------------

def dense_axis_arm(steps: Sequence[int],
                   series: Dict[Tuple[int, int], List[float]]) -> dict:
    """
    Hand `p_value_p_i1` the payload `formation_curve_payload` actually builds,
    and then the same input restricted to the heads that carry relays.

    The B side is synthetic. It has to be — the behavioural induction score is
    computed from phase 1's attention tensor and has not been computed for this
    sweep — and it is sound here for one reason that is checked rather than
    assumed: `paired_colocation_arm` profiles the A side FIRST, so the refusal
    this arm reports comes from the relay series, which is real. The arm records
    which side the refusal names.
    """
    forming = sorted(series)
    dense = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
    rng = np.random.default_rng(_SEED)

    # A behavioural series that rises everywhere, at a LOCATION that differs
    # across heads. The first version of this arm used a random monotone walk
    # and the forming row refused too — with the same numerical-noise backstop,
    # for the opposite reason: a walk spreads its change mass over every
    # interval, so every head's centroid sits at the grid's own midpoint, the
    # B side is a constant, and permuting it moves nothing. That is the
    # `diffuse_reference_profile` pull the construction exists to warn about,
    # arriving in this file's own fixture. A located change per head is what a
    # B side has to be for the arm to have two populations to pair.
    n_int = len(steps) - 1
    behav = {}
    for k in dense:
        v = np.zeros(len(steps), dtype=np.float64)
        v[int(rng.integers(n_int)) + 1:] = 1.0
        behav[k] = v.tolist()

    rows = []
    for label, axis in (("dense (every head the model has)", dense),
                        ("forming (heads carrying relays)", forming)):
        relay_by_step, score_by_step = [], []
        for i in range(len(steps)):
            relay_by_step.append({k: float(series[k][i])
                                  for k in axis if k in series})
            score_by_step.append({k: float(behav[k][i]) for k in axis})
        pay = formation_curve_payload(
            list(steps), relay_by_step, score_by_step,
            independence_source="two_stage", relay_owner=P_I1_RELAY_OWNER,
            above_null_excess=True)
        res = p_value_p_i1(pay["checkpoint_steps"], pay["motif_strength"],
                           pay["behavioral_induction_score"])
        n_zero = sum(1 for row in pay["motif_strength"]
                     if not any(v > 0.0 for v in row))
        rows.append({
            "axis": label,
            "n_heads_on_axis": int(pay["n_heads"]),
            "n_heads_with_no_relay_anywhere": int(n_zero),
            "p_value": res["p_value"],
            "refused": res["p_value"] is None,
            "reason": res["reason"],
            "attainable_floor": (res["arms"][0]["attainable_floor"]
                                 if res["p_value"] is not None else None),
        })

    dense_row, forming_row = rows
    return {
        "_what": ("`p_value_p_i1` on the payload `formation_curve_payload` "
                  "builds from the real relay series, and on the same input "
                  "restricted to the heads that carry relays."),
        "_why_it_is_here": (
            "a floor is the smallest p a design can express, and the first "
            "thing to establish is whether it can express one. On the axis the "
            "pipeline builds it cannot, and the reason is the axis rule rather "
            "than the data: the behavioural series is dense over all 384 heads, "
            "the relay side is zero on the 268 that never carry a relay, and "
            "`paired_colocation_arm` profiles every unit with no per-unit skip "
            "while `change_profile` refuses a series with no rise."),
        "_and_the_message_names_none_of_that": (
            "the refusal reads 'the series has no rise anywhere in the sweep'. "
            "It names no arm, no head index and no unit count, so a reader "
            "cannot tell that 268 of 384 units caused it, nor that no relay "
            "series whatever lifts it short of every head in the model gaining "
            "one."),
        "n_layers": N_LAYERS,
        "n_heads_per_layer": N_HEADS,
        "b_side_is_synthetic": True,
        "b_side_is_sound_because_the_a_side_is_profiled_first": True,
        "rows": rows,
        "dense_axis_emits_no_p_value": bool(dense_row["refused"]),
        "forming_axis_emits_a_p_value": bool(not forming_row["refused"]),
        "refusal_names_the_no_rise_condition": bool(
            dense_row["reason"] is not None
            and "no rise" in dense_row["reason"]),
        "refusal_names_the_unit_that_caused_it": bool(
            dense_row["reason"] is not None
            and ("head" in dense_row["reason"] or "unit" in dense_row["reason"])),
    }


# ---------------------------------------------------------------------------
# B. what "79 distinct centroids" is actually made of
# ---------------------------------------------------------------------------

def tie_structure_arm(steps: Sequence[int],
                      series: Dict[str, Dict[Tuple[int, int], List[float]]]) -> dict:
    """
    The multiplicity of the real per-head change locations, per relay owner.

    HANDOFF.md §3 reports 68 / 79 / 86 distinct centroids over 102 / 116 / 122
    heads and reads that as the degeneracy clearing. It does clear — one
    location was a floor of 1.000 and this is not — but the classes are not
    even, and the floor is set by the LARGEST class, not by the count of them.
    """
    rows = []
    for o in OWNERS:
        cents = []
        refused = 0
        for k in sorted(series[o]):
            try:
                cents.append(change_profile(steps, series[o][k], "rise")
                             ["centroid_log_step"])
            except ColocationRefused:
                refused += 1
        c = np.asarray(cents, dtype=np.float64)
        vals, counts = np.unique(np.round(c, 9), return_counts=True)
        order = np.argsort(-counts)
        log_ties = float(sum(math.lgamma(int(m) + 1) for m in counts))
        rows.append({
            "relay_owner": o,
            "registered": o == P_I1_RELAY_OWNER,
            "n_heads_scored": int(c.size),
            "n_heads_refused_no_rise": int(refused),
            "n_distinct_locations": int(vals.size),
            "largest_class_size": int(counts.max()),
            "largest_class_share": float(counts.max() / c.size),
            "n_singleton_classes": int((counts == 1).sum()),
            "class_size_histogram": {str(int(s)): int((counts == s).sum())
                                     for s in np.unique(counts)},
            "three_largest_classes": [
                {"location_log_step": float(vals[i]),
                 "location_step": float(10.0 ** vals[i] - 1.0),
                 "n_heads": int(counts[i])} for i in order[:3]],
            "log10_tying_subgroup_order": float(log_ties / math.log(10.0)),
            "log10_n_factorial": float(math.lgamma(int(c.size) + 1)
                                       / math.log(10.0)),
            "tie_floor": float(np.exp(min(
                log_ties - math.lgamma(int(c.size) + 1), 0.0))),
        })
    reg = next(r for r in rows if r["registered"])
    return {
        "_what": ("the class structure of the per-head change locations on the "
                  "registered 19-step grid, from the real relay series."),
        "_the_reading_it_corrects": (
            "'79 distinct centroids across 116 heads' is true and is not 79 "
            "classes of roughly 1.5. It is 77 singletons, one class of three "
            "and one class of thirty-six: 31% of the heads still put their "
            "change in a single interval, which is the twelve-step grid's "
            "degeneracy surviving in a third of the population."),
        "_why_the_largest_class_is_the_number_that_matters": (
            "the tying subgroup's order is prod(m!) over the classes, and a "
            "factorial is dominated by its largest term. Thirty-six of 116 is "
            "harmless — 10^42 against 10^190 — and thirty-six of forty would "
            "not be."),
        "rows": rows,
        "registered_owner": P_I1_RELAY_OWNER,
        "registered_largest_class_size": reg["largest_class_size"],
        "registered_tie_floor": reg["tie_floor"],
        "tie_half_does_not_bind_on_the_real_head_set": bool(
            reg["tie_floor"] < 1.0 / 2001.0),
    }


# ---------------------------------------------------------------------------
# C. what a perfect input returns, beside what the arm says it can
# ---------------------------------------------------------------------------

def perfect_input_arm(steps: Sequence[int],
                      series: Dict[str, Dict[Tuple[int, int], List[float]]],
                      alpha: float) -> dict:
    """
    Perfect co-location on the real head sets: does the arm land on its floor?

    `POPPER_PLAN.md` 6p's method — print what a perfect input returns beside
    what the arm claims it could — which is how the two halves of `P-AB1`'s
    floor were found and how 6m's were found in the steering gate.
    """
    rows = []
    for o in OWNERS:
        a = [series[o][k] for k in sorted(series[o])]
        arm = paired_colocation_arm(
            steps, a, "rise", _b_side(a), "rise",
            alpha=alpha, unit_name="head", arm_name=f"perfect::{o}")
        fl = arm["floor"]
        rows.append({
            "relay_owner": o,
            "registered": o == P_I1_RELAY_OWNER,
            "n_units": arm["n_units"],
            "n_pairings": arm["n_pairings"],
            "null_exhaustive": arm["null_exhaustive"],
            "observed_mean_distance_log_step": arm["mean_distance_log_step"],
            "perfect_input_p": arm["p_value"],
            "attainable_floor": arm["attainable_floor"],
            "tie_floor": fl["tie_floor"],
            "draw_count_floor": fl["draw_count_floor"],
            "binds": fl["binds"],
            "lands_on_the_floor": bool(
                abs(arm["p_value"] - arm["attainable_floor"]) <= 1e-12),
        })
    return {
        "_what": ("the mutual arm on the real relay locations against a B side "
                  "that co-locates with them exactly at every head."),
        "_why_it_is_here": (
            "the floor is defined as what a perfect input returns, so the way "
            "to check a reported floor is to build one and read the number "
            "back. It is what found the gap arm D reports."),
        "_the_b_side": (
            "twice the A series at every head. `change_profile` normalises the "
            "change mass, so the centroid is bit-identical while the series are "
            "not equal — which keeps `p_value_p_i1`'s tautology refusal, whose "
            "subject is identical series, out of a measurement about floors."),
        "alpha": alpha,
        "rows": rows,
        "every_owner_lands_on_its_floor": bool(
            all(r["lands_on_the_floor"] for r in rows)),
        "mean_distance_is_zero_at_a_perfect_input": bool(
            all(abs(r["observed_mean_distance_log_step"]) <= 1e-12
                for r in rows)),
    }


# ---------------------------------------------------------------------------
# D. the defect: the reported floor against the realised one
# ---------------------------------------------------------------------------

#: Tie structures on the registered grid, as (n_units, size of the one tied
#: class). Chosen to straddle alpha rather than to make the gap look large:
#: 9-of-10 is above it, 7-of-10 below, and the two differ by two heads.
DEFECT_CASES: Tuple[Tuple[int, int], ...] = ((10, 9), (10, 8), (10, 7),
                                             (12, 9), (20, 12), (36, 30))


def the_defect_arm(steps: Sequence[int], alpha: float) -> dict:
    """
    What `paired_colocation_arm` reported before 2026-09-03, and what a perfect
    input on the same locations actually returns.

    The old code's reported floor was `1 / n_draws` exactly, which is still
    computed and carried as `draw_count_floor`, so the before-column is read off
    the current record rather than reconstructed from git — there is nothing to
    get wrong in it.
    """
    n_int = interval_midpoints(np.asarray(steps, dtype=float)).size

    def series_at(j: int) -> List[float]:
        v = np.zeros(len(steps), dtype=np.float64)
        v[j + 1:] = 1.0                     # all change mass in interval j
        return v.tolist()

    rows = []
    for n, k in DEFECT_CASES:
        if k > n or (n - k) + 1 > n_int:
            continue
        classes = [0] * k + list(range(1, n - k + 1))
        a = [series_at(j) for j in classes]
        b = _b_side(a)
        ca = np.asarray([change_profile(steps, v, "rise")["centroid_log_step"]
                         for v in a])
        n_draws = math.factorial(n) if math.factorial(n) <= 5040 else 2001
        fl = pairing_floor_report(ca, ca.copy(), n_draws, alpha,
                                  math.factorial(n) <= 5040)
        closed = math.factorial(k) / math.factorial(n)
        try:
            arm = paired_colocation_arm(steps, a, "rise", b, "rise",
                                        alpha=alpha, unit_name="head",
                                        arm_name=f"defect::{n}::{k}")
            realised, refused, reason = arm["p_value"], False, None
        except ColocationRefused as exc:
            realised, refused, reason = None, True, str(exc)
        rows.append({
            "n_units": n,
            "largest_tied_class": k,
            "n_draws": n_draws,
            "reported_floor_before": fl["draw_count_floor"],
            "closed_form_tie_floor": closed,
            "computed_tie_floor": fl["tie_floor"],
            "closed_form_agrees": bool(
                abs(fl["tie_floor"] - closed) <= 1e-12 * max(closed, 1e-30)
                or abs(fl["tie_floor"] - closed) <= 1e-15),
            "attainable_floor_now": fl["attainable_floor"],
            "binds": fl["binds"],
            "perfect_input_p_if_emitted": realised,
            "refused_now": refused,
            "refusal_reason": (reason[:400] if reason else None),
            "understatement_factor": (
                fl["attainable_floor"] / fl["draw_count_floor"]),
        })
    emitted = [r for r in rows if not r["refused_now"]]
    worst = max(rows, key=lambda r: r["understatement_factor"])
    return {
        "_what": ("the arm's reported floor before this pass against the floor "
                  "a perfect input on the same locations actually reaches, on "
                  "the registered 19-step grid."),
        "_the_finding": (
            f"at {worst['n_units']} units with {worst['largest_tied_class']} "
            f"of them sharing one change location the arm reported "
            f"{worst['reported_floor_before']:.6f} and the attainable floor is "
            f"{worst['attainable_floor_now']:.6f} — a factor of "
            f"{worst['understatement_factor']:.0f}, above alpha={alpha}, and "
            f"emitted as a p-value with no refusal. It is 6m's defect in the "
            f"steering gate arriving in the shared estimator, and 6q's 'the "
            f"floor has two halves' reached by a third construction."),
        "_why_more_draws_do_not_fix_it": (
            "the tie fraction is prod(m!)/n!, which contains no draw count. "
            "Sampling the pairing group harder resolves a finer quantile of a "
            "distribution whose mass is already piled on the observation."),
        "_and_the_two_halves_cross_within_two_heads": (
            "9-of-10 refuses and 7-of-10 does not. The binding half is not a "
            "regime one is safely in or out of; it is set by the largest class "
            "and moves fast in it."),
        "alpha": alpha,
        "rows": rows,
        "closed_form_agrees_everywhere": bool(
            all(r["closed_form_agrees"] for r in rows)),
        "max_understatement_factor": float(worst["understatement_factor"]),
        "the_gap_reaches_above_alpha": bool(
            any(r["attainable_floor_now"] > alpha for r in rows)),
        "the_arm_now_refuses_exactly_those": bool(
            all(r["refused_now"] == (r["attainable_floor_now"] > alpha)
                for r in rows)),
        "and_emits_unchanged_below_it": bool(
            all(r["perfect_input_p_if_emitted"] is not None
                and r["perfect_input_p_if_emitted"] <= alpha
                for r in emitted)),
    }


# ---------------------------------------------------------------------------
# E. what a relay-count null may remove
# ---------------------------------------------------------------------------

def floor_vs_survivors_arm(alpha: float) -> dict:
    """
    The constraint step 1 puts on step 4, and the reason for running it first.

    A relay-count null turns the raw series into an above-null EXCESS, and a
    head whose excess no longer rises drops out of the scored set. So the null
    chooses `n_units`, and `n_units` with the tie structure chooses the floor.
    Two conditions, and they bind from opposite directions:

      * the draw-count half needs `n! >= 1/alpha` — four heads at alpha = 0.05.
      * the tie half needs `k! / n! <= alpha`, where k is the largest number of
        survivors sharing one change location.

    Reported as the largest admissible k per n, which is the form the author's
    decision needs: 'the null may leave n heads provided no more than k of them
    put their change in one interval'.
    """
    rows = []
    for n in range(2, 25):
        n_fact = math.factorial(n)
        n_draws = n_fact if n_fact <= 5040 else 2001
        draw_floor = 1.0 / n_draws
        k_max = 0
        for k in range(1, n + 1):
            if math.factorial(k) / n_fact <= alpha:
                k_max = k
        rows.append({
            "n_units": n,
            "n_draws": n_draws,
            "draw_count_floor": draw_floor,
            "draw_half_sufficient": bool(draw_floor <= alpha),
            "largest_admissible_tied_class": k_max,
            "tie_floor_at_that_class": (
                math.factorial(k_max) / n_fact if k_max else None),
            "share_of_units_that_may_be_tied": (k_max / n) if k_max else None,
        })
    ok = [r for r in rows if r["draw_half_sufficient"]]
    return {
        "_what": ("the two halves of the floor as functions of how many heads "
                  "survive the above-null excess, at the registered alpha."),
        "_why_it_is_here": (
            "EVALUABILITY.md's order puts the floor before the control because "
            "the floor is a constraint ON the control. A relay-count null that "
            "leaves three scored heads cannot produce a p-value at any alpha "
            "the registry uses, and one that leaves twelve of which nine share "
            "a location cannot either — and both are decidable now, with no "
            "null built and no draw taken."),
        "_what_it_does_not_say": (
            "nothing here is a null. It does not say how much a relay-count "
            "null SHOULD remove, only what it may remove and still leave a "
            "design that can reject. HANDOFF.md §5 records the shape the "
            "author has to choose and this does not choose it."),
        "alpha": alpha,
        "rows": rows,
        "min_units_for_the_draw_half": int(min(r["n_units"] for r in ok)),
        "the_two_halves_are_not_the_same_constraint": bool(
            any(r["largest_admissible_tied_class"] < r["n_units"] - 1
                for r in ok)),
        "_and_the_admissible_class_is_not_monotone_in_n": (
            "k = n - 1 gives a tie floor of exactly 1/n, so 'all but one head "
            "tied' clears alpha = 0.05 from n = 20 upward and fails below it. "
            "The column therefore jumps 17 -> 19 between 19 and 20 survivors. "
            "That is arithmetic and not a glitch, and it is why the table is "
            "reported as a table rather than as a rule of thumb: the "
            "constraint is on the largest class, and 'a big share of them may "
            "be tied' is true at one n and false at the n below it."),
        "_the_requirement": (
            "P-I1 needs a relay-count null leaving at least "
            f"{min(r['n_units'] for r in ok)} heads with a rising excess, and "
            "among them no more than k sharing one change location, k as "
            "tabulated. On the raw series the scored set is 116 heads with a "
            "largest class of 36, which clears both by a wide margin; what is "
            "not known, and cannot be until the null exists, is how much of "
            "that the excess subtraction keeps."),
    }


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

def build_record(seed: int = _SEED) -> dict:
    t0 = time.time()
    alpha = _alpha()
    steps, series = _load_series()
    reg = series[P_I1_RELAY_OWNER]

    rec: dict = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "_what_this_is": (
            "step 1 of EVALUABILITY.md's order — the attainable floor — for "
            "P-I1's relay-count null, run before the control exists."),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "alpha": alpha,
        "registered_sweep": list(REGISTERED_P_I1_SWEEP),
        "registered_relay_owner": P_I1_RELAY_OWNER,
        "construction_sha256": _sha256(CONSTRUCTION_PATH),
        "formation_gate_sha256": _sha256(FORMATION_GATE_PATH),
        "formation_curve_sha256": _sha256(FORMATION_CURVE_PATH),
        "inputs": {
            "series_json": str(SERIES_JSON.relative_to(REPO))
            if SERIES_JSON.is_relative_to(REPO) else str(SERIES_JSON),
            "series_json_sha256": _sha256(SERIES_JSON),
            "curve_json_sha256": (_sha256(CURVE_JSON)
                                  if CURVE_JSON.exists() else None),
            "_note": (
                "generated bulk under data/, git-ignored. The hashes are here "
                "so a reader can tell whether their tables are the ones these "
                "numbers came from; `--check` verifies them only when the "
                "files are present."),
        },
    }
    rec["dense_axis"] = dense_axis_arm(steps, reg)
    rec["tie_structure"] = tie_structure_arm(steps, series)
    rec["perfect_input"] = perfect_input_arm(steps, series, alpha)
    rec["the_defect"] = the_defect_arm(steps, alpha)
    rec["floor_vs_survivors"] = floor_vs_survivors_arm(alpha)
    rec["elapsed_seconds"] = round(time.time() - t0, 1)

    # Derived from the rows at write time, never typed. `EVALUABILITY.md`'s
    # thirty-second lesson (iv): three passes have now committed a summary
    # sentence carrying an earlier run's digits, because nothing in this
    # project compares a record's prose to its own fields.
    d, s, t = rec["the_defect"], rec["tie_structure"], rec["dense_axis"]
    p_reg = next(r for r in rec["perfect_input"]["rows"] if r["registered"])
    s_reg = next(r for r in s["rows"] if r["registered"])
    log_tie = (s_reg["log10_tying_subgroup_order"]
               - s_reg["log10_n_factorial"])
    rec["_the_finding"] = (
        f"P-I1's mutual arm cannot express a p-value at all on the axis "
        f"`formation_curve_payload` builds: {t['rows'][0]['n_heads_on_axis']} "
        f"heads, {t['rows'][0]['n_heads_with_no_relay_anywhere']} of them with "
        f"no relay anywhere, and `change_profile` refuses each one. On the "
        f"{t['rows'][1]['n_heads_on_axis']} heads that do form it emits "
        f"({t['rows'][1]['p_value']:.4g} against an arbitrary B side, "
        f"{p_reg['perfect_input_p']:.4g} against a perfectly co-locating one, "
        f"which is its floor). And the floor it reported was the draw-count "
        f"half alone: at "
        f"{d['rows'][0]['n_units']} units with "
        f"{d['rows'][0]['largest_tied_class']} sharing one location the "
        f"reported floor was {d['rows'][0]['reported_floor_before']:.6f} "
        f"against an attainable {d['rows'][0]['attainable_floor_now']:.6f}, a "
        f"factor of {d['max_understatement_factor']:.0f}. On the real "
        f"{s_reg['n_heads_scored']} heads the tie half is "
        f"10^{log_tie:.0f} and does not bind — it binds on the set a "
        f"relay-count null leaves, which is why this is step 1 and not a "
        f"footnote to step 4.")
    return rec


def check_record(path: Path = RECORD_PATH) -> List[str]:
    """
    Is the committed record still about the files on disk, and does it still
    support the change it was the evidence for?

    Four things can fail and each of them should: the record can describe a
    module that has moved; the dense-axis refusal can stop being there, in
    which case the wiring finding has nothing behind it; the two halves of the
    floor can stop disagreeing, in which case `pairing_floor_report` is
    reporting a distinction that no longer exists; and the arm can stop
    refusing exactly the cases whose floor is above alpha, which is the fix.
    """
    problems: List[str] = []
    if not path.exists():
        return [f"{path} is missing; regenerate with "
                f"`python3 -m tools.p_i1_attainable_floor --write`"]
    try:
        rec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path} is not valid JSON: {exc}"]

    if rec.get("schema_version") != RECORD_SCHEMA_VERSION:
        problems.append(
            f"schema_version {rec.get('schema_version')} != "
            f"{RECORD_SCHEMA_VERSION}; regenerate with --write")

    for key, described in (("construction", CONSTRUCTION_PATH),
                           ("formation_gate", FORMATION_GATE_PATH),
                           ("formation_curve", FORMATION_CURVE_PATH)):
        if not described.exists():
            problems.append(f"{described} is missing")
            continue
        if rec.get(f"{key}_sha256") != _sha256(described):
            problems.append(
                f"{described.name} has changed since this record was written "
                f"(sha256 {_sha256(described)[:12]} on disk vs "
                f"{str(rec.get(f'{key}_sha256'))[:12]} on record); rerun "
                f"--write rather than editing the hash")

    if list(rec.get("registered_sweep") or []) != list(REGISTERED_P_I1_SWEEP):
        problems.append(
            "the record was measured on a grid that is not the registered "
            "sweep; a floor on the wrong grid is a floor for another design")
    if rec.get("registered_relay_owner") != P_I1_RELAY_OWNER:
        problems.append(
            f"the record's relay owner {rec.get('registered_relay_owner')!r} "
            f"is not the registered {P_I1_RELAY_OWNER!r}")

    inp = rec.get("inputs") or {}
    if SERIES_JSON.exists() and inp.get("series_json_sha256") != _sha256(SERIES_JSON):
        problems.append(
            f"{SERIES_JSON.name} on disk is not the one these numbers came "
            f"from; rerun `python tools/run/curve.py` then --write")

    da = rec.get("dense_axis") or {}
    if not da.get("dense_axis_emits_no_p_value"):
        problems.append(
            "the record's dense-axis refusal is gone. Either the axis rule or "
            "the arm's per-unit handling changed, and the wiring finding no "
            "longer describes the code")
    if not da.get("forming_axis_emits_a_p_value"):
        problems.append(
            "the forming-head axis stopped emitting a p-value, so the record "
            "no longer shows that the dense-axis refusal is about the AXIS "
            "rather than about the data")

    ts = rec.get("tie_structure") or {}
    if not ts.get("tie_half_does_not_bind_on_the_real_head_set"):
        problems.append(
            "the tie half now binds on the real head set. That is not a "
            "record defect but it changes what P-I1 can express, and this "
            "record says it does not bind")

    de = rec.get("the_defect") or {}
    if not de.get("closed_form_agrees_everywhere"):
        problems.append(
            "prod(m!)/n! no longer reproduces `pairing_floor_report`'s tie "
            "floor; the closed form is the argument and the measurement is "
            "the check, and they have parted")
    if not de.get("the_gap_reaches_above_alpha"):
        problems.append(
            "no case in the record has an attainable floor above alpha, so "
            "the defect this pass fixed has nothing behind it")
    if not de.get("the_arm_now_refuses_exactly_those"):
        problems.append(
            "the arm no longer refuses exactly the cases whose attainable "
            "floor exceeds alpha — it refuses more or fewer, and either is a "
            "different design from the one this record measured")
    if not de.get("and_emits_unchanged_below_it"):
        problems.append(
            "a case below alpha stopped emitting; the refusal was measured to "
            "cost nothing where it does not fire and now costs something")

    fs = rec.get("floor_vs_survivors") or {}
    if not fs.get("rows"):
        problems.append("floor_vs_survivors has no rows")
    elif not fs.get("the_two_halves_are_not_the_same_constraint"):
        problems.append(
            "the tie half stopped being a separate constraint from the draw "
            "half, so the table this record hands the control's author is no "
            "longer telling it anything the unit count did not")

    pi = rec.get("perfect_input") or {}
    if not pi.get("every_owner_lands_on_its_floor"):
        problems.append(
            "a perfect input no longer lands on the arm's reported floor; "
            "that is the definition of a floor and it is failing")
    return problems


def print_summary(rec: dict) -> None:
    print(f"\n=== P-I1 attainable floor (alpha {rec['alpha']}, "
          f"{rec['elapsed_seconds']}s) ===")
    print(f"\n{rec['_the_finding']}\n")

    da = rec["dense_axis"]
    print("=== A. the axis the pipeline builds ===")
    print(f"  {'axis':36s} {'heads':>6} {'no relay':>9} {'p':>10}  reason")
    for r in da["rows"]:
        p = "REFUSED" if r["refused"] else f"{r['p_value']:.6g}"
        print(f"  {r['axis']:36s} {r['n_heads_on_axis']:>6d} "
              f"{r['n_heads_with_no_relay_anywhere']:>9d} {p:>10s}  "
              f"{(r['reason'] or '')[:52]}")
    print(f"  refusal names the unit that caused it: "
          f"{da['refusal_names_the_unit_that_caused_it']}")

    ts = rec["tie_structure"]
    print("\n=== B. what the distinct-centroid count is made of ===")
    print(f"  {'owner':12s} {'heads':>6} {'distinct':>9} {'largest':>8} "
          f"{'share':>7} {'singletons':>11} {'tie floor':>11}")
    for r in ts["rows"]:
        print(f"  {r['relay_owner']:12s} {r['n_heads_scored']:>6d} "
              f"{r['n_distinct_locations']:>9d} {r['largest_class_size']:>8d} "
              f"{r['largest_class_share']:>7.3f} "
              f"{r['n_singleton_classes']:>11d} {r['tie_floor']:>11.3g}")

    pi = rec["perfect_input"]
    print("\n=== C. what a perfect input returns ===")
    print(f"  {'owner':12s} {'units':>6} {'p':>11} {'floor':>11} "
          f"{'tie':>11} {'draws':>11} {'binds':>6}")
    for r in pi["rows"]:
        print(f"  {r['relay_owner']:12s} {r['n_units']:>6d} "
              f"{r['perfect_input_p']:>11.4g} {r['attainable_floor']:>11.4g} "
              f"{r['tie_floor']:>11.3g} {r['draw_count_floor']:>11.4g} "
              f"{r['binds']:>6s}")
    print(f"  every owner lands on its floor: "
          f"{pi['every_owner_lands_on_its_floor']}")

    de = rec["the_defect"]
    print("\n=== D. the reported floor against the realised one ===")
    print(f"  {'units':>6} {'tied':>5} {'reported':>10} {'attainable':>11} "
          f"{'closed form':>12} {'x':>6} {'binds':>6} {'now'}")
    for r in de["rows"]:
        print(f"  {r['n_units']:>6d} {r['largest_tied_class']:>5d} "
              f"{r['reported_floor_before']:>10.6f} "
              f"{r['attainable_floor_now']:>11.6f} "
              f"{r['closed_form_tie_floor']:>12.4g} "
              f"{r['understatement_factor']:>6.0f} {r['binds']:>6s} "
              f"{'REFUSED' if r['refused_now'] else 'emits'}")
    print(f"  closed form agrees everywhere: {de['closed_form_agrees_everywhere']}; "
          f"refuses exactly the cases above alpha: "
          f"{de['the_arm_now_refuses_exactly_those']}")

    fs = rec["floor_vs_survivors"]
    print("\n=== E. what a relay-count null may remove ===")
    print(f"  {'survivors':>10} {'draws':>7} {'draw floor':>11} "
          f"{'max tied':>9} {'tie floor there':>16}")
    for r in fs["rows"]:
        if not r["draw_half_sufficient"] and r["n_units"] > 6:
            continue
        tf = ("--" if r["tie_floor_at_that_class"] is None
              else f"{r['tie_floor_at_that_class']:.4g}")
        print(f"  {r['n_units']:>10d} {r['n_draws']:>7d} "
              f"{r['draw_count_floor']:>11.6f} "
              f"{r['largest_admissible_tied_class']:>9d} {tf:>16s}")
    print(f"  smallest survivor count the draw half allows: "
          f"{fs['min_units_for_the_draw_half']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--write", action="store_true",
                    help="run it and write the record")
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
