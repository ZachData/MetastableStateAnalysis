"""
tools/dry_run_p_st1.py — P-ST1's gate, run on inputs whose answer is known.

`claims/EVALUABILITY.md` closed its CLAIM-C dry-run section on a queue: seven
adjudicable rows had been validated by unit tests and none had been run on an
input whose correct verdict is fixed a priori, and *"the queue that used to
read 'convert the next needs-null row' now has a second entry ahead of it for
each row already converted."* This is that treatment for `P-ST1`, and it
commits what came back to `claims/audits/p_st1_dry_run.json`.

WHY THIS ENTRY WAS THE ONE TO DO NEXT

It can genuinely lose -- the particle and standard accounts make INCOMPATIBLE
predictions about the sign -- and its whole intervention is exact linear
algebra, so the gate can be exercised end to end on populations with a planted
answer and no model at all. `POPPER_PLAN.md` 6k also applied the dry-run
discipline at CONSTRUCTION time rather than after, which made this a real check
rather than a formality: the question was whether anything survives being run
on inputs the construction did not already have in mind.

Something did not. See 6m, and the second section below.

FIVE ARMS, AND WHAT EACH ONE'S ANSWER IS KNOWN TO BE

A. `sharp_input` -- the cloud occupies U_pos and nothing else, at dim U_pos =
   dim(occupied) so every draw from U_pos lands in it. Every pair is then
   informative and predicted, the statistic is at its maximum, and the correct
   verdict is TRACKS-DECOMPOSITION a priori. Mirrored, with the cloud in
   U_neg, the correct verdict is INVERTS -- the branch that would enter the
   ledger as a falsification, exercised here on an input whose answer is not in
   question rather than only as a rate.

   The sharp question it asks is whether a perfect input reaches the FLOOR,
   1/(draws + 1). It need not: the null is drawn on the observed population, so
   a null draw that happens to inform in the same direction ties the maximum
   and pushes p up. Whether that happens is a property of the layer, not of the
   draw count, and nothing had looked.

B. `exchangeable_input` -- the observed pair is itself drawn as a random
   re-split of a fixed union. The observed statistic is then exchangeable with
   the null draws BY CONSTRUCTION, so P(p <= alpha) <= alpha exactly, for any
   population whatsoever. This is the one arm whose correct answer needs no
   modelling assumption at all, and it is the direct analogue of CLAIM-C's
   self-comparison: an input where the right answer is not in question.

   It also measures how much of the conservatism is TIES rather than control,
   which is 6g's conditioning lesson asked of a rank statistic.

C. `verdict_band` -- the gate's verdict rates across its own input space,
   swept over the occupancy asymmetry between the two arms and the
   dim U_pos / dim(occupied) ratio the registry records as a precondition.
   CLAIM-C's dry run found an admissible band outside which its gate is a
   constant function; this asks the same question of this one. Every cell also
   re-scores a PERFECT input over several seeds, so a cell that reaches no
   verdict is separated into "the data was not strong enough here" and
   "nothing in this cell reached one at all".

   It stops short of CLAIM-C's claim, and the field names say so. That band
   was settled by enumerating every concordance count, which proves the gate
   is a constant function there. This statistic has no such enumeration, so
   what is recorded is a measured zero over a stated number of draws.

D. `occupancy_readout` -- whether the verdict is predictable from a quantity
   that needs no injection: each arm's share of the centred population's energy
   divided by the k/d a random subspace of that dimension would hold. Built
   from arm C's runs, so it costs nothing extra. If it predicts well, the pilot
   can read its answer off the activations and the two projectors before
   spending a sweep -- and a reader can see what a TRACKS verdict is made of.

E. `refusals_and_branches` -- every refusal kind and every verdict branch
   reached at least once, with each refusal re-scored to check it turns nothing
   away that could have cleared alpha. `POPPER_PLAN.md` 6h's arm-incapable-of-
   failing and 6l's refuses-what-could-have-passed are the same defect from
   opposite sides, and both directions are checked here. That includes the
   refusal this file's own arm A produced: a refusal no input in the record
   reaches is a refusal nothing has checked.

WHAT THIS DELIBERATELY DOES NOT DO

It adjudicates nothing. The populations are synthetic, no activations and no
Phase 2 projectors exist in this repository, and `claims/adjudications/` stays
empty. It adds no agreement axis: each one moves probability mass into
INSUFFICIENT, and 6g's caution against that is general even though it was
written about CLAIM-C.

RUN IT

    python3 -m tools.dry_run_p_st1 --write      # about ten minutes
    python3 -m tools.dry_run_p_st1 --check
    python3 -m tools.dry_run_p_st1 --summary

Committed rather than recomputed, for the reason `docs/CI_BASELINE.md` gives
for the other artifacts. `tests/test_p_st1_dry_run.py` pins the record together
with the sha256 of the gate and of the calibration it is read beside, so the
record going stale is a failure rather than a silence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "claims" / "audits" / "p_st1_dry_run.json"

#: Bump when the record's shape changes, so a reader at a different version
#: refuses rather than reinterpreting fields that may have moved.
RECORD_SCHEMA_VERSION = 1

#: The two files every verdict here depends on: the gate decides the verdicts,
#: and the calibration is what the gate's constants were set from. Either
#: changing makes this record a description of something that no longer exists.
GATE_PATH = ROOT / "p7_motifs" / "steering_gate.py"
CALIBRATION_PATH = ROOT / "claims" / "calibration" / "steering_sign.json"

#: Geometry. Smaller than the calibration's, because a dry run is a sweep over
#: verdicts rather than a rate measurement and the whole point is to run the
#: WHOLE gate many times. The occupied subspace is a fixed 8 dimensions, and
#: `dim U_pos` varies around it so the precondition ratio moves and nothing
#: else does -- the confound the calibration's own dimension arm had to fix.
D_MODEL, N_TOKENS = 80, 48
DIM_OCCUPIED = 8

#: Null draws here. Floor 1/100 = 0.01, well under alpha, and half the module's
#: 199 because this file runs the gate a few thousand times. Recorded in the
#: artifact, since the floor is a number the arms below assert against.
DRY_RUN_DRAWS = 99

#: Pair counts the sharp input is swept over. One pair is included on purpose:
#: the adjudicated null's floor does not depend on the pair count, so a single
#: informative pair should be able to reject, and that is a claim about the
#: gate that had never been run.
SHARP_N_PAIRS: Tuple[int, ...] = (1, 2, 4, 8, 16)
SHARP_SEEDS = 3

#: Independent draws of the exchangeable input. 200 resolves a rate to about
#: +/- 0.03, which is enough to see a rate that should be at or below 0.05
#: sitting at 0.2 -- the size of failure this arm exists to detect.
N_EXCHANGEABLE = 200

#: The band sweep: (energy into U_pos, energy into U_neg) in units of the
#: isotropic noise. The pairs are chosen to span from a symmetric H0 through to
#: a clean H1 in the SAME family, so the axis the verdict moves along is the
#: asymmetry and not the construction.
BAND_CONCENTRATIONS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0), (1.5, 1.5), (1.5, 1.2), (1.5, 0.9), (1.5, 0.0), (3.0, 0.0))
BAND_DIM_RATIOS: Tuple[float, ...] = (1.0, 2.0, 3.0)
BAND_N_PAIRS: Tuple[int, ...] = (8,)
N_BAND_DRAWS = 25

#: Perfect-input seeds per band cell. NOT one: the first version of this arm
#: read the counterfactual off a single draw and reported cells as reaching no
#: verdict while the cell's own draws reached one 28% of the time. A
#: counterfactual read off one realisation is a statement about that
#: realisation.
N_PERFECT_SEEDS = 5

_SEED = 20260826


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Populations with a planted answer
# ---------------------------------------------------------------------------

def planted_layer(rng: np.random.Generator, *, c_pos: float, c_neg: float,
                  dim_arm: int = DIM_OCCUPIED, dim_occupied: int = DIM_OCCUPIED,
                  d: int = D_MODEL, n: int = N_TOKENS,
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A population whose energy inside each arm is planted, and the two arms.

    `c_pos` and `c_neg` are the amplitudes of the energy injected into the
    first `dim_occupied` directions of U_pos and of U_neg, in units of the
    isotropic background. c_pos == c_neg is H0 with both arms occupied -- the
    two arms are then statistically identical, so a label swap is a
    distributional identity and the correct verdict is INSUFFICIENT. c_neg == 0
    is the planted H1.

    `dim_arm` sets both arms' dimension while `dim_occupied` stays fixed, so
    the registry's precondition ratio dim U_pos / dim(occupied) moves and
    nothing else does.
    """
    if dim_occupied > dim_arm:
        raise ValueError(
            f"the occupied subspace ({dim_occupied}) must fit inside an arm "
            f"({dim_arm}); otherwise the ratio being swept is not a ratio")
    Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    u_pos = Q[:, :dim_arm]
    u_neg = Q[:, dim_arm:2 * dim_arm]
    X = rng.normal(size=(n, d))
    for U, c in ((u_pos, c_pos), (u_neg, c_neg)):
        if c:
            X = X + (rng.normal(size=(n, dim_occupied))
                     * float(c)) @ U[:, :dim_occupied].T
    return X, u_pos, u_neg


def _gate(X, u_pos, u_neg, n_pairs, seed, n_draws: Optional[int] = None) -> dict:
    """One whole-gate run, with the reported diagnostics switched off.

    The alpha-profile and the two retired nulls are reported in a RECORD and
    read by a person; this file scores verdicts by the thousand and reads
    none of them, so paying for them here would buy nothing. The calibration
    measures the retired nulls, paired, on its own families.
    """
    from p7_motifs import steering_gate as sg

    # `n_draws` defaults to None and is resolved HERE rather than in the
    # signature: a module constant bound as a default argument is bound once
    # at definition time, which POPPER_PLAN.md 6h records as a live bug in
    # `attainable_floor_report` and which produced an inconsistent first run of
    # this very file -- the arms computed a floor from one draw count while the
    # gate ran at another.
    draws = DRY_RUN_DRAWS if n_draws is None else int(n_draws)
    prev = sg.MATCHED_DIMENSION_NULL_DIAGNOSTIC
    sg.MATCHED_DIMENSION_NULL_DIAGNOSTIC = False
    try:
        return sg.p_value_p_st1(X, u_pos, u_neg, n_pairs, seed=int(seed),
                                n_draws=draws, with_profile=False)
    finally:
        sg.MATCHED_DIMENSION_NULL_DIAGNOSTIC = prev


# ---------------------------------------------------------------------------
# A. the sharp input, and its mirror
# ---------------------------------------------------------------------------

def sharp_input(rng: np.random.Generator, alpha: float) -> dict:
    """
    The cloud in one arm and nothing in the other: the answer is known.

    Two things are checked that no unit test had. Does a PERFECT input reach
    the attainable floor -- and if not, is that because the null found the same
    direction, which is a property of the layer rather than of the draw count?
    And does one informative pair suffice, as the adjudicated null's
    draw-count-only floor says it must?
    """
    draw_floor = 1.0 / (DRY_RUN_DRAWS + 1.0)
    rows = []
    for planted, (c_pos, c_neg), want in (("H1", (3.0, 0.0), "TRACKS-DECOMPOSITION"),
                                          ("INVERTED", (0.0, 3.0), "INVERTS")):
        for m in SHARP_N_PAIRS:
            verdicts, ps, obs, informative = [], [], [], []
            at_attainable = at_draw_floor = refused = 0
            attainable, kinds = [], []
            for s in range(SHARP_SEEDS):
                X, u_pos, u_neg = planted_layer(rng, c_pos=c_pos, c_neg=c_neg)
                res = _gate(X, u_pos, u_neg, m, seed=_SEED + 97 * s)
                verdicts.append(res["verdict"])
                obs.append(float(res["observed"]))
                informative.append(int(res["n_informative_pairs"]))
                if res.get("p_value") is None:
                    refused += 1
                    kinds.append(res.get("refusal_kind"))
                    ps.append(None)
                    attainable.append(res.get("best_attainable_p"))
                    continue
                key = ("attainable_p_greater" if planted == "H1"
                       else "attainable_p_reciprocal")
                p = float(res["p_value"] if planted == "H1"
                          else res["p_reciprocal"])
                ps.append(p)
                attainable.append(float(res[key]))
                at_attainable += int(abs(p - float(res[key])) < 1e-12)
                at_draw_floor += int(abs(p - draw_floor) < 1e-12)
            rows.append({
                "planted": planted, "n_pairs": m,
                "expected_verdict": want,
                "verdicts": verdicts,
                "all_as_expected": all(v == want for v in verdicts),
                "n_refused": refused,
                "refusal_kinds": sorted({k for k in kinds if k}),
                "adjudicated_tail_p": ps,
                "attainable_floor_of_that_tail": attainable,
                "n_at_attainable_floor": at_attainable,
                "n_at_draw_count_floor": at_draw_floor,
                "mean_observed": float(np.mean(obs)),
                "max_observed": float(2 * m),
                "statistic_is_maximal":
                    bool(all(abs(o) == 2 * m for o in obs)),
                "mean_informative_pairs": float(np.mean(informative)),
                # The PLANTING is perfect -- the cloud is entirely inside one
                # arm and dim U_pos = dim(occupied), so every drawn direction
                # lies in it. The STATISTIC still is not deterministic: a
                # direction can land where the two arms' effective-rank changes
                # happen to share a sign, and then that pair contributes D = 0.
                # Measured here at about 1 pair in 48. The arm therefore reports
                # the rate rather than asserting exact maximality, because an
                # assertion that fails once in fifty runs is an assertion about
                # the draw and not about the gate.
                "informative_rate": (float(np.mean(informative)) / m
                                     if m else float("nan")),
            })
    emitted = [r for r in rows if r["n_refused"] == 0]
    return {
        "_what": ("the cloud planted entirely inside one arm, at "
                  "dim U_pos = dim(occupied) so every draw lands in it. The "
                  "correct verdict is TRACKS-DECOMPOSITION when it is U_pos "
                  "and INVERTS when it is U_neg, a priori."),
        "_the_sharp_question": (
            "whether a perfect input reaches the draw-count floor 1/(draws+1). "
            "It need not, and at small pair counts it does not: the null is "
            "drawn on the OBSERVED population, sum(D) cannot exceed 2m, and "
            "every re-split that already reaches 2m ties it. The attainable "
            "floor is therefore a property of the layer and the pair count. "
            "Finding that is what this arm was for, and it changed the gate: "
            "the attainable floor is now computed and the gate refuses when "
            "neither tail can reach alpha."),
        "draw_count_floor": draw_floor,
        "n_draws": DRY_RUN_DRAWS,
        "n_seeds_per_cell": SHARP_SEEDS,
        "alpha": alpha,
        "rows": rows,
        "smallest_pair_count_that_emits": min(
            [r["n_pairs"] for r in emitted], default=None),
        "worst_informative_rate": (min(r["informative_rate"] for r in rows)
                                   if rows else None),
        "every_pair_informative_everywhere": bool(
            all(r["statistic_is_maximal"] for r in rows)),
        "every_emitted_verdict_correct": bool(
            all(r["all_as_expected"] for r in emitted)),
        "every_planted_verdict_correct": bool(
            all(r["all_as_expected"] for r in rows)),
        "perfect_input_hits_its_attainable_floor_wherever_it_emits": bool(
            all(r["n_at_attainable_floor"] == SHARP_SEEDS for r in emitted)),
        "perfect_input_hits_the_draw_count_floor_everywhere": bool(
            emitted and all(r["n_at_draw_count_floor"] == SHARP_SEEDS
                            for r in emitted)),
    }


# ---------------------------------------------------------------------------
# B. the exchangeable input
# ---------------------------------------------------------------------------

def exchangeable_input(rng: np.random.Generator, alpha: float) -> dict:
    """
    The observed pair drawn from the null's own family: P(p <= alpha) <= alpha.

    Exactly, for any population, with no modelling assumption -- which is what
    makes this the analogue of CLAIM-C's self-comparison rather than one more
    simulated H0. If it does not hold, the null as IMPLEMENTED is not the null
    as described, and no amount of measuring families would localise that.

    The population deliberately occupies both arms strongly: that is where the
    retired matched-dimension null failed, and an exchangeability check run
    only on a bland population would be an arm incapable of failing.
    """
    from p7_motifs.steering_gate import occupancy, resplit_pair, union_basis

    reject_g = reject_l = ties_at_one = emitted = 0
    ps: List[float] = []
    occ = []
    for i in range(N_EXCHANGEABLE):
        X, u_pos, u_neg = planted_layer(rng, c_pos=1.5, c_neg=1.5)
        union = union_basis(u_pos, u_neg)
        a, b = resplit_pair(union, u_pos.shape[1], rng)
        res = _gate(X, a, b, 8, seed=_SEED + 3 * i)
        if res.get("p_value") is None:
            continue
        emitted += 1
        ps.append(float(res["p_value"]))
        reject_g += int(res["p_value"] <= alpha)
        reject_l += int(res["p_reciprocal"] <= alpha)
        ties_at_one += int(res["p_value"] == 1.0 and res["p_reciprocal"] == 1.0)
        occ.append((occupancy(X, a), occupancy(X, b)))
    occ_arr = np.asarray(occ, dtype=np.float64) if occ else np.zeros((1, 2))
    e = float(max(emitted, 1))
    return {
        "_what": ("the observed pair drawn as a random re-split of a fixed "
                  "union, so it is exchangeable with the null draws BY "
                  "CONSTRUCTION and P(p <= alpha) <= alpha exactly."),
        "_why_it_is_the_sharp_one": (
            "every other validity measurement in this project is a rate under "
            "a modelled H0 family. This one's answer follows from the "
            "construction, so a failure localises to the implementation "
            "rather than to the family."),
        "n_draws_of_the_input": N_EXCHANGEABLE,
        "n_emitted": emitted,
        "alpha": alpha,
        "reject_greater_given_emitted": reject_g / e,
        "reject_reciprocal_given_emitted": reject_l / e,
        "both_tails_at_one_given_emitted": ties_at_one / e,
        "mean_p": float(np.mean(ps)) if ps else None,
        "median_p": float(np.median(ps)) if ps else None,
        "mean_occupancy_of_each_arm": [float(occ_arr[:, 0].mean()),
                                       float(occ_arr[:, 1].mean())],
        "holds": bool(reject_g / e <= alpha + 1.96 * np.sqrt(
            alpha * (1 - alpha) / max(emitted, 1))),
    }


# ---------------------------------------------------------------------------
# C + D. the band, and what predicts it
# ---------------------------------------------------------------------------

def verdict_band(rng: np.random.Generator, alpha: float) -> dict:
    """
    Verdict rates across the gate's own input space, plus a perfect-input
    counterfactual in every cell.

    The counterfactual is what separates the two ways a cell can reach no
    verdict. CLAIM-C's dry run found a region where its gate is a CONSTANT
    FUNCTION -- refusing every input including a perfect one -- and a verdict
    table without the counterfactual cannot tell that from a region where the
    data simply was not strong enough.
    """
    from p7_motifs.steering_gate import occupancy

    cells, readout = [], []
    for ratio in BAND_DIM_RATIOS:
        dim_arm = int(round(ratio * DIM_OCCUPIED))
        for (c_pos, c_neg) in BAND_CONCENTRATIONS:
            for m in BAND_N_PAIRS:
                counts = {"TRACKS-DECOMPOSITION": 0, "INVERTS": 0,
                          "INSUFFICIENT": 0}
                refused = 0
                log_ratios, informative = [], []
                for i in range(N_BAND_DRAWS):
                    X, u_pos, u_neg = planted_layer(
                        rng, c_pos=c_pos, c_neg=c_neg, dim_arm=dim_arm)
                    res = _gate(X, u_pos, u_neg, m, seed=_SEED + 11 * i)
                    counts[res["verdict"]] += 1
                    refused += int(res.get("p_value") is None)
                    informative.append(int(res["n_informative_pairs"]))
                    lr = res["occupancy"]["occupancy_log_ratio"]
                    log_ratios.append(float(lr))
                    readout.append({"log_ratio": float(lr),
                                    "verdict": res["verdict"],
                                    "dim_ratio": float(ratio)})
                # The counterfactual: the same geometry with the cloud planted
                # entirely in U_pos, which is the strongest input this cell can
                # be handed. Several seeds, not one -- the first version of
                # this arm used a single draw and reported cells as reaching no
                # verdict while the cell's own 25 draws reached one 28% of the
                # time. A counterfactual read off one realisation is not a
                # statement about the cell, and printing the table is what
                # showed it.
                perfect = []
                for s in range(N_PERFECT_SEEDS):
                    X, u_pos, u_neg = planted_layer(
                        rng, c_pos=3.0, c_neg=0.0, dim_arm=dim_arm)
                    perfect.append(_gate(X, u_pos, u_neg, m,
                                         seed=_SEED + 7 + 13 * s))
                pv = [r["verdict"] for r in perfect]
                reached = (counts["TRACKS-DECOMPOSITION"] + counts["INVERTS"]
                           + sum(v != "INSUFFICIENT" for v in pv))
                cells.append({
                    "dim_ratio": float(ratio), "dim_arm": dim_arm,
                    "c_pos": c_pos, "c_neg": c_neg, "n_pairs": m,
                    "mean_occupancy_log_ratio": float(np.mean(log_ratios)),
                    "mean_informative_pairs": float(np.mean(informative)),
                    "tracks": counts["TRACKS-DECOMPOSITION"] / N_BAND_DRAWS,
                    "inverts": counts["INVERTS"] / N_BAND_DRAWS,
                    "insufficient": counts["INSUFFICIENT"] / N_BAND_DRAWS,
                    "refusal_rate": refused / N_BAND_DRAWS,
                    "perfect_input_verdicts": pv,
                    "perfect_input_reaches_a_verdict":
                        sum(v != "INSUFFICIENT" for v in pv),
                    "n_perfect_seeds": N_PERFECT_SEEDS,
                    "n_draws_reaching_a_verdict": int(reached),
                    "no_verdict_in_any_draw": bool(reached == 0),
                })
    dead = [c for c in cells if c["no_verdict_in_any_draw"]]
    return {
        "_what": ("verdict rates over the gate's input space: the occupancy "
                  "asymmetry between the arms, and the dim U_pos / "
                  "dim(occupied) ratio the registry records as a "
                  "precondition."),
        "_the_counterfactual": (
            "`perfect_input_verdicts` re-scores the strongest input each cell "
            "can be handed, over several seeds. A cell where nothing -- not "
            "one of its own draws and not one perfect input -- reaches a "
            "verdict is one where the gate returned no information, which is "
            "the shape of what CLAIM-C's dry run found for its own gate."),
        "_what_this_does_NOT_establish": (
            "that the gate is a CONSTANT FUNCTION there. CLAIM-C's band could "
            "be settled by enumerating every concordance count; this statistic "
            "has no such enumeration, so `no_verdict_in_any_draw` is a "
            "measured zero over a stated number of draws and not a proof. The "
            "distinction is kept in the field name."),
        "alpha": alpha,
        "n_draws_per_cell": N_BAND_DRAWS,
        "n_perfect_seeds": N_PERFECT_SEEDS,
        "cells": cells,
        "cells_with_no_verdict_in_any_draw": [
            {"dim_ratio": c["dim_ratio"], "c_pos": c["c_pos"],
             "c_neg": c["c_neg"], "n_pairs": c["n_pairs"]} for c in dead],
        "n_cells": len(cells),
        "n_cells_with_no_verdict": len(dead),
        "_readout": readout,
    }


def occupancy_readout(band: dict) -> dict:
    """
    Does a quantity needing NO injection predict the verdict?

    Built from the band's own runs, so it costs nothing. If the answer is yes,
    two things follow and they point opposite ways: the pilot can read its
    answer off the activations and the two projectors before spending a sweep,
    and a reader should know that a TRACKS verdict is a statement about which
    arm holds more of the cloud.
    """
    rows = band["_readout"]
    tr = [r["log_ratio"] for r in rows
          if r["verdict"] == "TRACKS-DECOMPOSITION"]
    inv = [r["log_ratio"] for r in rows if r["verdict"] == "INVERTS"]
    ins = [r["log_ratio"] for r in rows if r["verdict"] == "INSUFFICIENT"]
    sep = float(min(tr) - max(ins)) if (tr and ins) else None
    auc = None
    if tr and ins:
        a = np.asarray(tr)[:, None]
        b = np.asarray(ins)[None, :]
        auc = float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))
    return {
        "_what": ("each arm's chance-normalized occupancy is computable from "
                  "the activations and the two projectors with no injection "
                  "and no null. This asks how well their log ratio predicts "
                  "the verdict the whole gate returns."),
        "_how_to_read_the_auc": (
            "the probability that a TRACKS run has a larger occupancy log "
            "ratio than an INSUFFICIENT one, ties counted half. 0.5 is no "
            "information; 1.0 would mean the verdict is a readout of a "
            "quantity computable without running the gate at all, which would "
            "be worth knowing in both directions -- a free precondition for "
            "the pilot, and a warning about what a TRACKS verdict is made of."),
        "n_runs": len(rows),
        "n_tracks": len(tr), "n_inverts": len(inv),
        "n_insufficient": len(ins),
        "auc_tracks_vs_insufficient": auc,
        "smallest_log_ratio_that_tracked": (min(tr) if tr else None),
        "largest_log_ratio_that_was_insufficient": (max(ins) if ins else None),
        "largest_log_ratio_that_inverted": (max(inv) if inv else None),
        "separation": sep,
        "perfectly_separated": bool(sep is not None and sep > 0),
        "mean_log_ratio_by_verdict": {
            "TRACKS-DECOMPOSITION": (float(np.mean(tr)) if tr else None),
            "INVERTS": (float(np.mean(inv)) if inv else None),
            "INSUFFICIENT": (float(np.mean(ins)) if ins else None),
        },
    }


# ---------------------------------------------------------------------------
# E. refusals and branches
# ---------------------------------------------------------------------------

def refusals_and_branches(rng: np.random.Generator, alpha: float) -> dict:
    """
    Every refusal reached and re-scored; every verdict branch reached.

    A branch nothing can trigger is `POPPER_PLAN.md` 6h's arm incapable of
    failing; a refusal that turns away what could have passed is 6l's mirror
    image of it. Both are checked, and the refusals are re-scored rather than
    argued about: for each, whether ANY input could have cleared alpha under
    the configuration that was refused.
    """
    from p7_motifs.steering_gate import N_SUBSPACE_DRAWS

    X, u_pos, u_neg = planted_layer(rng, c_pos=3.0, c_neg=0.0)
    d = X.shape[1]
    rows = []

    # 1. too few null draws. Rescore: the floor is 1/(draws+1) and both tails
    #    share it, so no observed value whatsoever can clear alpha.
    few = 9
    res = _gate(X, u_pos, u_neg, 4, seed=_SEED, n_draws=few)
    rows.append({
        "refusal_kind": res.get("refusal_kind"),
        "reached": res.get("p_value") is None,
        "best_attainable_p": res.get("best_attainable_p"),
        "could_any_input_have_cleared_alpha":
            bool(res.get("best_attainable_p", 1.0) <= alpha),
        "how_rescored": ("the floor 1/(draws+1) bounds BOTH tails, so it is "
                         "read directly rather than searched for"),
        "remedy": f"raise n_draws to at least {int(np.ceil(1/alpha)) - 1}",
    })

    # 2. arms that overlap. Rescore: no orthogonal pair of those dimensions
    #    sits in this space, so there is no null to compare against at all --
    #    a refusal about the geometry and not about power.
    overlap = np.hstack([u_pos[:, :2], u_neg[:, :u_neg.shape[1] - 2]])
    res = _gate(X, u_pos, overlap, 4, seed=_SEED)
    rows.append({
        "refusal_kind": res.get("refusal_kind"),
        "reached": res.get("p_value") is None,
        "best_attainable_p": None,
        "could_any_input_have_cleared_alpha": False,
        "how_rescored": ("the union cannot hold the observed pair "
                         "orthogonally, so no re-split of it reproduces the "
                         "observed geometry and there is no p to compute"),
        "remedy": "supply arms the projector build's resolution order made "
                  "orthogonal, or fewer dimensions",
    })

    # 3. dimensions exceeding d_model, which is the same refusal reached from
    #    the other side. It is listed separately because it is a different
    #    fact about the run and a reader chasing it needs to see which.
    big = np.linalg.qr(rng.normal(size=(d, d - 4)))[0]
    res = _gate(X, big, big, 4, seed=_SEED)
    rows.append({
        "refusal_kind": res.get("refusal_kind"),
        "reached": res.get("p_value") is None,
        "best_attainable_p": None,
        "could_any_input_have_cleared_alpha": False,
        "how_rescored": "no pair of those dimensions fits in d_model at all",
        "remedy": "none available at this d_model",
    })

    # 4. the null ties the largest value the statistic can take, in BOTH
    #    directions. Rescore: 2m is an upper bound on the observation, so the
    #    floor computed at it is a lower bound on the attainable p and nothing
    #    the data could have done would have cleared alpha. This is the
    #    refusal the dry run itself produced (POPPER_PLAN.md 6m); it is
    #    exercised here because a refusal that no input in the record reaches
    #    is a refusal nothing has checked.
    X, u_pos, u_neg = planted_layer(rng, c_pos=1.5, c_neg=1.5)
    tie = None
    for s in range(12):
        cand = _gate(X, u_pos, u_neg, 1, seed=_SEED + 5 * s)
        if cand.get("refusal_kind") == "null_ties_the_maximum":
            tie = cand
            break
        X, u_pos, u_neg = planted_layer(rng, c_pos=1.5, c_neg=1.5)
    rows.append({
        "refusal_kind": (tie or cand).get("refusal_kind"),
        "reached": tie is not None,
        "best_attainable_p": (tie or cand).get("best_attainable_p"),
        "could_any_input_have_cleared_alpha": bool(
            tie is not None and tie["best_attainable_p"] <= alpha),
        "how_rescored": ("both tails' attainable floors are computed at the "
                         "largest value sum(D) can take, which is an upper "
                         "bound on any observation, so neither could have "
                         "reached alpha"),
        "remedy": "more pairs, or a union whose random re-splits inform less "
                  "often; raising n_draws does not help",
    })

    verdicts = {}
    for name, (c_pos, c_neg) in (("TRACKS-DECOMPOSITION", (3.0, 0.0)),
                                 ("INVERTS", (0.0, 3.0)),
                                 ("INSUFFICIENT", (0.0, 0.0))):
        X, u_pos, u_neg = planted_layer(rng, c_pos=c_pos, c_neg=c_neg)
        res = _gate(X, u_pos, u_neg, 8, seed=_SEED)
        verdicts[name] = res["verdict"]

    return {
        "_what": ("every refusal reached and re-scored, and every verdict "
                  "branch reached on an input whose answer is known."),
        "shipped_n_draws": int(N_SUBSPACE_DRAWS),
        "refusals": rows,
        "all_refusals_reached": all(r["reached"] for r in rows),
        "no_refusal_turned_away_a_clearable_input": not any(
            r["could_any_input_have_cleared_alpha"] for r in rows),
        "verdict_branches": verdicts,
        "all_branches_reached": all(k == v for k, v in verdicts.items()),
    }


# ---------------------------------------------------------------------------
# Assembling
# ---------------------------------------------------------------------------

def build_record(seed: int = _SEED) -> dict:
    from p7_motifs import steering_gate as sg

    rng = np.random.default_rng(seed)
    alpha = float(sg._alpha())
    band = verdict_band(rng, alpha)
    readout = occupancy_readout(band)
    band.pop("_readout")
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "_what": ("P-ST1's steering gate run on inputs whose correct verdict "
                  "is known a priori: a cloud planted entirely in one arm, an "
                  "observed pair drawn from the null's own family, and a sweep "
                  "of the gate's input space with a perfect-input "
                  "counterfactual in every cell."),
        "_why": ("claims/EVALUABILITY.md's queue: each converted row is owed a "
                 "run on an input whose answer is known, ahead of converting "
                 "the next one. CLAIM-C had it; this is P-ST1's."),
        "_not": ("not evidence about any model and not an adjudication. The "
                 "populations are synthetic; what is being checked is the "
                 "gate."),
        "generated_by": "python3 -m tools.dry_run_p_st1 --write",
        "gate_file": str(GATE_PATH.relative_to(ROOT)),
        "gate_sha256": _sha256(GATE_PATH),
        "calibration_file": str(CALIBRATION_PATH.relative_to(ROOT)),
        "calibration_sha256": _sha256(CALIBRATION_PATH),
        "alpha": alpha,
        "null_family": sg.NULL_FAMILY,
        "geometry": {"d_model": D_MODEL, "n_tokens": N_TOKENS,
                     "dim_occupied": DIM_OCCUPIED,
                     "n_subspace_draws": DRY_RUN_DRAWS,
                     "n_subspace_draws_shipped": int(sg.N_SUBSPACE_DRAWS)},
        "seed": int(seed),
        "sharp_input": sharp_input(rng, alpha),
        "exchangeable_input": exchangeable_input(rng, alpha),
        "verdict_band": band,
        "occupancy_readout": readout,
        "refusals_and_branches": refusals_and_branches(rng, alpha),
    }


# ---------------------------------------------------------------------------
# Staleness, summary, CLI
# ---------------------------------------------------------------------------

def check_record(path: Path = RECORD_PATH) -> List[str]:
    """
    Is the committed record still about the files on disk, and self-consistent?

    Deliberately does NOT re-run the gate: the record is committed precisely
    because ten minutes is too slow for a gate people wait on. What can go
    stale is the pair of files it describes, so that is what is hashed -- and
    what can be wrong is a headline that no longer follows from the numbers
    under it, so those are re-derived from the record itself.
    """
    problems: List[str] = []
    if not path.exists():
        return [f"{path} is missing; regenerate with "
                f"`python3 -m tools.dry_run_p_st1 --write`"]
    try:
        rec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path} is not valid JSON: {exc}"]

    from p7_motifs import steering_gate as sg

    if rec.get("schema_version") != RECORD_SCHEMA_VERSION:
        problems.append(
            f"schema_version {rec.get('schema_version')} != "
            f"{RECORD_SCHEMA_VERSION}; regenerate with --write")
    for key, described in (("gate", GATE_PATH),
                           ("calibration", CALIBRATION_PATH)):
        if not described.exists():
            problems.append(f"{described} is missing")
            continue
        if rec.get(f"{key}_sha256") != _sha256(described):
            problems.append(
                f"{described.name} has changed since the dry run was written "
                f"(sha256 {_sha256(described)[:12]} on disk vs "
                f"{str(rec.get(f'{key}_sha256'))[:12]} on record). The record "
                f"describes something that no longer exists in that form; "
                f"rerun --write rather than editing the hash.")
    if rec.get("null_family") != sg.NULL_FAMILY:
        problems.append(
            f"the dry run was made against null family "
            f"{rec.get('null_family')!r} and the module now adjudicates "
            f"{sg.NULL_FAMILY!r}; its verdicts are verdicts of a different "
            f"test")

    sharp = rec.get("sharp_input", {})
    if not sharp.get("every_emitted_verdict_correct"):
        wrong = [r for r in sharp.get("rows", [])
                 if r["n_refused"] == 0 and not r["all_as_expected"]]
        problems.append(
            f"the gate EMITTED a verdict other than the planted one on an "
            f"input whose answer is known a priori, in {len(wrong)} of "
            f"{len(sharp.get('rows', []))} cells. That is the criterion not "
            f"meaning what it says, not a calibration question")
    if not sharp.get("perfect_input_hits_its_attainable_floor_wherever_it_emits"):
        problems.append(
            "a perfect input did not land on its own attainable floor, so the "
            "statistic is not at its maximum where the arm says it is or the "
            "floor is not the floor")
    if sharp.get("smallest_pair_count_that_emits") is None:
        problems.append(
            "the sharp-input arm never emitted, so nothing in it can fail and "
            "its verdicts prove nothing")
    # NOT "the statistic was at its maximum": the planting is perfect but the
    # per-pair statistic is not deterministic, and an assertion that fails once
    # in fifty runs is an assertion about the draw. What has to hold is that
    # essentially every pair informs -- if that stops being true the input is
    # no longer the near-perfect one this arm is supposed to hand the gate.
    for r in sharp.get("rows", []):
        if r["informative_rate"] < 0.95:
            problems.append(
                f"planted {r['planted']} at {r['n_pairs']} pairs: only "
                f"{r['informative_rate']:.3f} of pairs informed, so the input "
                f"was not the near-perfect one this arm is supposed to hand "
                f"the gate")

    ex = rec.get("exchangeable_input", {})
    if not ex.get("n_emitted"):
        problems.append(
            "the exchangeable-input arm emitted nothing, so its rate is true "
            "by vacuity -- an arm incapable of failing")
    elif not ex.get("holds"):
        problems.append(
            f"the observed pair drawn from the NULL'S OWN FAMILY rejected at "
            f"{ex.get('reject_greater_given_emitted')} against alpha "
            f"{ex.get('alpha')}. Exchangeability there is by construction, so "
            f"this is the implementation not matching the description")

    rb = rec.get("refusals_and_branches", {})
    if not rb.get("all_refusals_reached"):
        problems.append(
            "a refusal listed in the record was not actually reached by the "
            "input built to trigger it, so nothing here says it works")
    if not rb.get("no_refusal_turned_away_a_clearable_input"):
        problems.append(
            "a refusal turned away a configuration that could have cleared "
            "alpha, which is POPPER_PLAN.md 6l's defect: a refusal must cost "
            "no verdict the gate could otherwise have reached")
    if not rb.get("all_branches_reached"):
        problems.append(
            f"not every verdict branch fired on the input built for it: "
            f"{rb.get('verdict_branches')}. A branch nothing can trigger is "
            f"POPPER_PLAN.md 6h's arm incapable of failing")

    band = rec.get("verdict_band", {})
    if not band.get("cells"):
        problems.append("verdict_band has no cells")
    return problems


def print_summary(rec: dict) -> None:
    print(f"gate:        {rec['gate_file']}  sha256 {rec['gate_sha256'][:12]}")
    print(f"calibration: {rec['calibration_file']}  sha256 "
          f"{rec['calibration_sha256'][:12]}")
    print(f"null:  {rec['null_family']}")
    print(f"alpha: {rec['alpha']}   geometry: {rec['geometry']}\n")

    s = rec["sharp_input"]
    print("=== A. the cloud planted entirely in one arm: the answer is known ===")
    print(f"  draw-count floor {s['draw_count_floor']:.4f} at "
          f"{s['n_draws']} draws -- and it is NOT the attainable one")
    print(f"  {'planted':>10} {'pairs':>6} {'as expected':>12} {'refused':>8} "
          f"{'at attainable':>13} {'at draw floor':>13} "
          f"{'adjudicated tail p':>26}")
    for r in s["rows"]:
        ps = ", ".join("refused" if p is None else f"{p:.4f}"
                       for p in r["adjudicated_tail_p"])
        n = len(r["adjudicated_tail_p"])
        print(f"  {r['planted']:>10} {r['n_pairs']:>6} "
              f"{str(r['all_as_expected']):>12} {r['n_refused']:>4}/{n:<3} "
              f"{r['n_at_attainable_floor']:>6}/{n:<6} "
              f"{r['n_at_draw_count_floor']:>6}/{n:<6} {ps:>26}")
    print(f"  smallest pair count that emits at all: "
          f"{s['smallest_pair_count_that_emits']}   worst informative rate "
          f"{s['worst_informative_rate']:.4f}  (every pair informative "
          f"everywhere: {s['every_pair_informative_everywhere']})")
    print(f"  every emitted verdict correct: "
          f"{s['every_emitted_verdict_correct']}   "
          f"perfect input lands on its attainable floor: "
          f"{s['perfect_input_hits_its_attainable_floor_wherever_it_emits']}   "
          f"on the draw-count floor: "
          f"{s['perfect_input_hits_the_draw_count_floor_everywhere']}")

    e = rec["exchangeable_input"]
    print("\n=== B. the observed pair drawn from the NULL'S OWN family ===")
    print(f"  {e['n_emitted']}/{e['n_draws_of_the_input']} emitted; arms at "
          f"occupancy {e['mean_occupancy_of_each_arm'][0]:.2f} and "
          f"{e['mean_occupancy_of_each_arm'][1]:.2f}")
    print(f"  P(TRACKS) {e['reject_greater_given_emitted']:.3f}   "
          f"P(INVERTS-tail) {e['reject_reciprocal_given_emitted']:.3f}   "
          f"alpha {e['alpha']}   holds: {e['holds']}")
    print(f"  both tails exactly 1.0 (conservative by TIE rather than by "
          f"control): {e['both_tails_at_one_given_emitted']:.3f}")
    print(f"  mean p {e['mean_p']:.3f}, median {e['median_p']:.3f}")

    b = rec["verdict_band"]
    print(f"\n=== C. the verdict band over {b['n_cells']} cells, "
          f"{b['n_draws_per_cell']} draws each ===")
    print(f"  {'dim ratio':>9} {'c_pos':>6} {'c_neg':>6} {'log occ ratio':>13} "
          f"{'TRACKS':>7} {'INVERTS':>8} {'INSUFF':>7} "
          f"{'perfect reaches':>21}")
    for c in b["cells"]:
        print(f"  {c['dim_ratio']:>9.1f} {c['c_pos']:>6.1f} {c['c_neg']:>6.1f} "
              f"{c['mean_occupancy_log_ratio']:>13.3f} "
              f"{c['tracks']:>7.3f} {c['inverts']:>8.3f} "
              f"{c['insufficient']:>7.3f} "
              f"{c['perfect_input_reaches_a_verdict']}/{c['n_perfect_seeds']}"
              f"{'':>17}")
    print(f"  cells where NOTHING reached a verdict -- not one of "
          f"{b['n_draws_per_cell']} draws, not one of "
          f"{b['n_perfect_seeds']} perfect inputs: "
          f"{b['n_cells_with_no_verdict']} of {b['n_cells']}")
    for c in b["cells_with_no_verdict_in_any_draw"]:
        print(f"    dim ratio {c['dim_ratio']}, c_pos {c['c_pos']}, "
              f"c_neg {c['c_neg']}, {c['n_pairs']} pairs")

    o = rec["occupancy_readout"]
    print("\n=== D. does a quantity needing NO injection predict the verdict? ===")
    print(f"  {o['n_runs']} runs: {o['n_tracks']} TRACKS, {o['n_inverts']} "
          f"INVERTS, {o['n_insufficient']} INSUFFICIENT")
    print(f"  mean log occupancy ratio by verdict: "
          f"{o['mean_log_ratio_by_verdict']}")
    print(f"  smallest that TRACKED {o['smallest_log_ratio_that_tracked']}, "
          f"largest that was INSUFFICIENT "
          f"{o['largest_log_ratio_that_was_insufficient']}")
    print(f"  AUC(TRACKS vs INSUFFICIENT): {o['auc_tracks_vs_insufficient']}  "
          f"(0.5 = no information, 1.0 = the verdict is a readout of it)")
    print(f"  perfectly separated: {o['perfectly_separated']}  "
          f"(separation {o['separation']})")

    r = rec["refusals_and_branches"]
    print("\n=== E. refusals re-scored, branches reached ===")
    for row in r["refusals"]:
        print(f"  {str(row['refusal_kind']):>22}  reached={row['reached']}  "
              f"could any input have cleared alpha="
              f"{row['could_any_input_have_cleared_alpha']}")
    print(f"  every branch fired on the input built for it: "
          f"{r['all_branches_reached']}  {r['verdict_branches']}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--write", action="store_true",
                    help="run the dry run and write the record (~10 minutes)")
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
