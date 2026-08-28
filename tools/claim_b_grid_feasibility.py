#!/usr/bin/env python3
"""
tools/claim_b_grid_feasibility.py — which checkpoint sweeps can carry
`CLAIM-B`'s anchor arms, computed before any data exists.

Run once, offline; the result is committed to
`claims/calibration/claim_b_grid_feasibility.json` and pinned by
`tests/test_claim_b_grid_feasibility.py`.

WHY THIS FILE EXISTS

`POPPER_PLAN.md` §6o left `CLAIM-B` with two grid failures and named them one
mechanism read at two geometries. The registered 25-checkpoint cheap sweep puts
a change-free series' location at step 955 — *inside* the 512–2000 anchor
window — so a series that changes nowhere attains the arm's maximum statistic
and `anchor_arm` refuses. Pythia's full every-1000 schedule puts it at step
31496, which clears that condition, and then the same pull drags a real
anchored change out of the window instead. §6o's closing line: *"the sweep that
satisfies both is neither of the two the project has"*, and it left choosing one
to the author.

**Which grids satisfy both is arithmetic on the grid and the window.** It needs
no checkpoints, no activations and no p-value, so it can be enumerated rather
than invented, and the author's decision becomes a choice from a computed set
rather than a design problem. That enumeration is this file. Choosing among the
grids it admits is still a pre-registered decision of `CLAIM-C`'s criterion's
class, and this file does not take it.

WHAT THE ARITHMETIC IS, AND THE THREE THINGS §6o DID NOT HAVE

`core.changepoint_colocation.grid_feasibility` computes five numbers per grid.
Two are §6o's; three came out of building this file and each changed the
answer.

  1. `reference_outside_window`   §6o's refusal, unchanged: the uniform
                                  profile's centroid against the window.
  2. `change_free_ceiling_rate`   condition 1 read as a RATE. §6o refuses on
                                  the NOISELESS reference; a realised
                                  change-free series scatters around it, so a
                                  grid clearing condition 1 by a hair still
                                  reaches the ceiling on half its draws.
  3. `retained_window_fraction`   §6o's drag, as a share of the window rather
                                  than a rate on one planted onset -- and
                                  computed from the change's OWN WIDTH, which
                                  is the third new thing and the one that
                                  mattered most.
  4. `false_anchor_fraction`      how much sweep OUTSIDE the window a grid
                                  reads INSIDE it.
  5. `window_read_span_in_change_free_sd`
                                  how far apart the locations it reports for
                                  the two ends of the window are, in the
                                  estimator's own scatter.

**Conditions 4 and 5 exist because the first enumeration without them returned
a grid that is optimal and useless.** Maximising retention alone picks a sweep
whose single wide interval swallows the whole window: every anchored change
reads inside it — retention 1.000 — and so does every change for a third of a
window-width above it, and it cannot say *where* in the window anything
happened. `(512, 3000, 8000, ...)` came top of the first run on all of §6o's
conditions. Condition 5 is the CLAIM's requirement and not the arm's, and it is
worth saying which: the anchor arm alone is happiest at a read span of zero.

**And the change's width is a required input, because the reading without it is
not a bound.** The first version located a change at the midpoint of the
interval containing it — the sharp limit, and the module docstring's own
"the sweep's resolution is its intervals". It looked conservative and is not: a
change of real width spreads mass into neighbouring intervals, so a coarse
interval just past the window collects it and the location leaves the window at
zero noise. A ten-checkpoint grid that reading scored at retention 1.000 put a
planted anchor inside the window on **0.017** of draws, and the prediction and
the measurement were both in a smoke run with nothing comparing them.
`closed_forms` now compares them on every row and `check_record` fails if the
sharp limit ever stops overstating, because that is the argument for requiring
the width.

WHAT THE ENUMERATION RANGES OVER

Only real checkpoints. Pythia publishes 11 log-spaced steps up to 512 and then
every 1000 to 143000, and a pilot can download a subset of those and nothing
else. The family enumerated is the SHAPE the schedules in this repository
already have — `core/pythia_registry.py`'s `PYTHIA_410M_PILOT_STEPS` is a
log-spaced head, a fine arithmetic stage and a coarse tail — with each stage's
knobs swept:

    grid = head + fine + tail
    head  a suffix of (0, 1, 2, 4, ..., 512), or (0, 512), or (512,), or none
    fine  range(1000, fine_hi, fine_gap)
    tail  range(fine_hi + tail_gap, tail_hi, tail_gap), or none

It is a family and not the power set, and the record says so: 2^154 subsets is
not enumerable and most of them are not schedules anyone would run. What the
record claims is that the family contains the shapes this project writes, and
that the answer inside it does not rest on an artificial bound of it.

WHAT IT FOUND

**The degeneracies are not the binding constraints.** Roughly two thirds of the
enumerated family clears them — 61,894 of 96,127 — leaving the reference outside
the window, some of the window retained and a read span above zero. What binds
is the graded conditions: **2,529 grids** meet all three (ceiling rate under
α/5, false anchors under a twentieth of a window-width, read span at least one
of the estimator's own standard deviations).

**And then the rule as first written admitted nothing, which is the finding.**
Retention was to be at least 0.95 across σ/R ∈ [0, 0.05] — across the range
rather than at one level, because σ/R is not known until the run happens and
retention is not monotone in it. **Zero grids of 96,127.** The honest reading is
not that the bound is too strict:

    the best worst-case retention any published Pythia schedule reaches is
    0.680, and no sweep holds the whole anchor window

So retention is **maximised** rather than thresholded, and the record stores the
achievable maximum and says that no grid reaches the reference. Tuning the
threshold until something passed would have hidden the one thing worth knowing.

**Where the window is lost is its upper end, and it is lost at zero noise.** A
change centred near step 2000 puts half its mass above 2000; the next published
checkpoint a feasible grid can afford there is tens of thousands of steps away,
so that mass is collected at an interval midpoint far outside the window. Adding
checkpoints between 2000 and 20000 fixes it and pulls the sweep's own midpoint
back INTO the window, which is §6o's refusal. That trade is between this claim's
registered anchor window and EleutherAI's release schedule, and no choice of
grid removes it.

**What the shortlist offers, and the axis it cannot decide.** The best-retention
grid is twelve checkpoints at 0.680 with a read span of 1.82 SDs; the largest
read span is 6.84 SDs at thirty-four checkpoints and 0.427 retention; the best
sweep reaching past step 100000 is fifteen checkpoints at 0.600, so serving the
predictions that need late checkpoints costs **0.080** of retention. How many
checkpoints a run can afford, and whether one sweep must do both jobs, is not in
this arithmetic.

**Measured, every grid this arithmetic picks discriminates at 1.000** between an
anchored change and no located change at all, and rejects on a change-free input
at 0.000, where every schedule this repository contains discriminates at 0.000.

WHAT IT DOES NOT DO

It adjudicates nothing and it does not choose a grid: the series are synthetic,
no Pythia sweep artifact is in this repository, `claims/adjudications/` is empty
and `null_construction` has not frozen. Choosing a sweep is a pre-registered
decision for the author; what is here is the set to choose from and the cost of
each choice. **The author took it on 2026-08-28** and
`core.changepoint_colocation.REGISTERED_CLAIM_B_SWEEP` carries the result — this
file is the evidence behind that decision and is not where it lives.

    python3 -m tools.claim_b_grid_feasibility --write
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "claims" / "calibration" / "claim_b_grid_feasibility.json"

import sys
if str(ROOT) not in sys.path:                    # pragma: no cover - entrypoint
    sys.path.insert(0, str(ROOT))

from core.changepoint_colocation import (        # noqa: E402
    CLAIM_B_ANCHOR_WINDOW,
    change_free_centroid_sd,
    change_profile,
    diffuse_reference_profile,
    grid_feasibility,
    interval_midpoints,
)
from core.checkpoint_frames import step_x        # noqa: E402

# The synthetic family, the control families and the anchor arm's arithmetic
# scored without the refusal all come from the dry run rather than being
# rewritten here. That is deliberate on both counts: the rates below are then
# rates under the SAME family §6o measured, so the two records are comparable
# row for row, and `_scored` is already pinned against `anchor_arm` itself by
# `claim_b_p_i1_dry_run.json`'s `module_agreement` — a third implementation of
# the arm's ranking would need its own guard and would earn nothing.
from tools.dry_run_claim_b_p_i1 import (         # noqa: E402
    CHEAP_SWEEP,
    DENSE_SWEEP,
    EARLY_DENSE_SWEEP,
    NOISE_SD,
    N_CONTROLS,
    N_UNITS,
    _refuses,
    _scored,
    controls,
    series,
)

SCHEMA_VERSION = 1

#: The file whose arithmetic this record is about.
CONSTRUCTION_PATH = ROOT / "core" / "changepoint_colocation.py"

#: Pythia's published checkpoints, and the only steps a pilot can sample. Copied
#: from the release schedule rather than taken from `core/pythia_registry.py`,
#: which imports `transformers` at module scope and so cannot be reached from a
#: tool that has to run without it. `check_record` reads the registry's own
#: assignments out of the SOURCE with `ast` and compares them, so the copy
#: cannot drift silently and the comparison needs no dependency.
PUBLISHED_LOG_STEPS: Tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
PUBLISHED_LINEAR_STEPS: Tuple[int, ...] = tuple(range(1000, 143001, 1000))

#: The enumerated family's knobs. PLACED, in the sense that they bound a search
#: rather than a measurement: every one is a range of real published steps, and
#: what would make them wrong is the answer sitting on one of their edges, which
#: `frontier_is_interior` checks rather than asserts.
FINE_GAPS: Tuple[int, ...] = (1000, 2000, 3000, 4000, 5000)
FINE_HIGHS: Tuple[int, ...] = tuple(range(1000, 40001, 1000))
TAIL_GAPS: Tuple[int, ...] = (5000, 10000, 15000, 20000, 25000, 30000, 40000,
                              50000, 60000)
TAIL_HIGHS: Tuple[int, ...] = (30000, 50000, 70000, 90000, 110000, 130000, 143000)

#: Which of those bounds are facts about Pythia and which are this file's. The
#: distinction is the whole content of the interior check: a frontier resting on
#: a REAL bound is an answer, a frontier resting on an ARTIFICIAL one means the
#: sweep was too narrow and the bound is the answer instead.
REAL_BOUNDS = {
    "smallest_fine_gap": (min(FINE_GAPS),
                          "Pythia's own release granularity above step 512"),
    "largest_tail_high": (max(TAIL_HIGHS), "the last published checkpoint"),
}
ARTIFICIAL_BOUNDS = {
    "largest_fine_high": max(FINE_HIGHS),
    "largest_tail_gap": max(TAIL_GAPS),
    "smallest_grid": None,                       # filled from MIN_GRID_SIZE
}

#: The smallest sweep this enumeration will consider. PLACED at the module's own
#: refusal plus one: `MIN_CHECKPOINTS` is 3, and a four-point grid is the first
#: that has an interior interval at all.
MIN_GRID_SIZE = 4

ALPHA = 0.05

#: Replicates per measured cell. 200 resolves a proportion to about +/-0.015,
#: which is what arm D needs: it rests on a DIFFERENCE between two rates
#: measured on the same cell, the same way `claim_b_p_i1_dry_run.json`'s arm B
#: does, rather than on either rate alone.
N_REPS = 200

_SEED = 20260828


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# A. the closed forms, against simulation
# ---------------------------------------------------------------------------

#: The grids the closed forms are checked on. Spanning n = 24 to 153 and three
#: different geometries, because a closed form that agrees on one grid has
#: agreed with one number.
_CHECK_GRIDS: Dict[str, Tuple[int, ...]] = {
    "cheap-25 (registered)": CHEAP_SWEEP,
    "early-dense-73": EARLY_DENSE_SWEEP,
    "dense-154": DENSE_SWEEP,
    "linear-143": tuple(range(1000, 143001, 1000)),
}


def closed_forms(rng, reps: int) -> dict:
    """
    The three closed forms `grid_feasibility` rests on, each measured beside
    its prediction.

    `change_free_centroid_sd` is the one worth stating, because its correction
    term looks like decoration and is not. Adjacent first differences of
    i.i.d. noise are correlated at exactly -1/2 — they share a checkpoint — and
    adjacent interval midpoints sit on the same side of the grid's mean, so the
    covariance term subtracts. Dropping it overstates the spread by about 75%,
    which is the difference between a change-free series being three standard
    deviations outside the window and being two.
    """
    sd_rows, retention_rows = [], []
    for name, grid in _CHECK_GRIDS.items():
        s = np.asarray(grid, dtype=float)
        mids = interval_midpoints(s)

        cs = np.array([
            change_profile(s, NOISE_SD * rng.standard_normal(s.size),
                           "rise")["centroid_log_step"]
            for _ in range(reps)])
        pred_sd = change_free_centroid_sd(s)
        # The same statistic with the adjacent-covariance term dropped, so the
        # record carries what the correction is worth rather than asserting it.
        c = mids - mids.mean()
        naive_sd = float(np.sqrt((np.pi - 1.0) * float(np.sum(c * c))) / mids.size)
        sd_rows.append({
            "grid": name,
            "n_intervals": int(mids.size),
            "predicted_mean_log_step": float(mids.mean()),
            "measured_mean_log_step": float(cs.mean()),
            "predicted_sd_log_step": pred_sd,
            "measured_sd_log_step": float(cs.std()),
            "sd_ratio_predicted_over_measured": float(pred_sd / cs.std()),
            "sd_without_the_adjacent_covariance_term": naive_sd,
            "that_naive_sd_ratio": float(naive_sd / cs.std()),
        })

        x_lo, x_hi = (float(step_x([CLAIM_B_ANCHOR_WINDOW[0]])[0]),
                      float(step_x([CLAIM_B_ANCHOR_WINDOW[1]])[0]))
        for sigma in (0.005, 0.02):
            feas = grid_feasibility(s, CLAIM_B_ANCHOR_WINDOW,
                                    noise_to_range=sigma,
                                    change_width_log_step=CHANGE_WIDTH_LOG_STEP)
            inside = []
            for _ in range(reps // 2):
                x_true = rng.uniform(x_lo, x_hi)
                v = _logistic(s, 10.0 ** x_true - 1.0, rng, sigma)
                cen = change_profile(s, v, "rise")["centroid_log_step"]
                inside.append(x_lo <= cen <= x_hi)
            retention_rows.append({
                "grid": name,
                "noise_to_range": sigma,
                "predicted_retained_fraction": feas["retained_window_fraction"],
                "predicted_retained_fraction_sharp_change_limit":
                    feas["sharp_change_retained_window_fraction"],
                "measured_share_of_planted_anchors_read_inside":
                    float(np.mean(inside)),
            })

    sd_ratios = [abs(r["sd_ratio_predicted_over_measured"] - 1.0) for r in sd_rows]
    naive_ratios = [r["that_naive_sd_ratio"] for r in sd_rows]
    ret_err = [abs(r["predicted_retained_fraction"]
                   - r["measured_share_of_planted_anchors_read_inside"])
               for r in retention_rows]
    sharp_optimistic = [
        r["predicted_retained_fraction_sharp_change_limit"]
        - r["measured_share_of_planted_anchors_read_inside"]
        for r in retention_rows]
    return {
        "_what": ("each closed form in `grid_feasibility`, measured beside its "
                  "prediction on four grids spanning n = 24 to 153."),
        "_why_it_is_here": (
            "the whole point of this file is that a grid can be judged with no "
            "data. A closed form that has only been derived is an argument; "
            "these are the measurements that say the arithmetic describes what "
            "the estimator actually does."),
        "n_reps": reps,
        "noise_sd": NOISE_SD,
        "change_free_centroid": sd_rows,
        "worst_sd_relative_error": float(max(sd_ratios)),
        "sd_closed_form_tracks_the_measurement": bool(max(sd_ratios) <= 0.05),
        "adjacent_covariance_term_is_load_bearing": bool(min(naive_ratios) > 1.4),
        "_adjacent_covariance_note": (
            "dropping the -1/2 adjacent correlation between first differences "
            "overstates the change-free spread by the ratio in "
            "`that_naive_sd_ratio` on every grid. It is not a refinement."),
        "retention": retention_rows,
        "worst_retention_absolute_error": float(max(ret_err)),
        "retention_tracks_the_measurement": bool(max(ret_err) <= 0.15),
        "largest_amount_the_sharp_limit_overstates_retention_by":
            float(max(sharp_optimistic)),
        "the_sharp_limit_is_not_a_bound": bool(max(sharp_optimistic) > 0.1),
        "_retention_note": (
            "TWO readings are predicted per row and only one of them tracks. "
            "The reading that uses the change's own width agrees with the "
            "measurement; the sharp-change limit -- all the mass in the "
            "interval containing the change -- OVERSTATES retention, by the "
            "amount stored here. That is why the width is a required input to "
            "`grid_feasibility` rather than a refinement, and the check fails "
            "if the sharp limit ever stops overstating, because then the "
            "argument for requiring the width has gone."),
    }


def _logistic(steps: np.ndarray, mid_step: float, rng, noise: float) -> np.ndarray:
    """The dry run's series, reached through its own `series` for the families
    that have one and inlined here for a single planted onset."""
    from tools.dry_run_claim_b_p_i1 import ONSET_WIDTH_LOG_STEP
    x = step_x(steps)
    v = 1.0 / (1.0 + np.exp(-(x - np.log10(mid_step + 1.0)) / ONSET_WIDTH_LOG_STEP))
    return v + noise * rng.standard_normal(x.size) if noise else v


# ---------------------------------------------------------------------------
# B. the grids the project already has
# ---------------------------------------------------------------------------

def catalogue() -> dict:
    """
    Every grid this project has written down, scored on all five conditions,
    with `grid_feasibility`'s condition 1 checked against `anchor_arm`'s own
    refusal on each.

    That check is the point of the section. Condition 1 is not a restatement of
    the refusal, it is meant to BE it, and two pieces of code computing the same
    condition is exactly the drift `POPPER_PLAN.md` §6g records on CLAIM-C's
    fast path.
    """
    rows = []
    for name, grid in (("cheap-25 (registered)", CHEAP_SWEEP),
                       ("early-dense-73", EARLY_DENSE_SWEEP),
                       ("dense-154", DENSE_SWEEP),
                       ("pythia-410m-pilot-27", _pilot_steps())):
        s = np.asarray(grid, dtype=float)
        feas = grid_feasibility(s, CLAIM_B_ANCHOR_WINDOW, noise_to_range=NOISE_SD,
                                change_width_log_step=CHANGE_WIDTH_LOG_STEP)
        rows.append({
            "grid": name,
            "steps": [int(v) for v in grid],
            "arm_refuses": bool(_refuses(s)),
            "feasibility": feas,
        })
    agree = all(r["arm_refuses"] != r["feasibility"]["reference_outside_window"]
                for r in rows)
    return {
        "_what": ("the grids this repository already contains, scored on the "
                  "five conditions."),
        "rows": rows,
        "condition_1_matches_the_arms_refusal": bool(agree),
        "_agreement_note": (
            "`grid_feasibility`'s `reference_outside_window` and `anchor_arm`'s "
            "refusal are the same condition computed twice. This asserts they "
            "agree on every catalogued grid, because a second implementation of "
            "a gate's arithmetic that nothing pins is a defect waiting."),
        "none_of_them_meets_the_hard_conditions": bool(
            not any(r["feasibility"]["reference_outside_window"]
                    and r["feasibility"]["change_free_ceiling_rate"] < MAX_CEILING_RATE
                    and r["feasibility"]["false_anchor_fraction"] < MAX_FALSE_ANCHOR_FRACTION
                    and r["feasibility"]["window_read_span_in_change_free_sd"] >= MIN_READ_SPAN_IN_SD
                    for r in rows)),
        "_why_each_one_fails": [
            {"grid": r["grid"],
             "reference_outside_window": r["feasibility"]["reference_outside_window"],
             "uniform_profile_centroid_step":
                 r["feasibility"]["uniform_profile_centroid_step"],
             "retained_window_fraction": r["feasibility"]["retained_window_fraction"],
             "false_anchor_fraction": r["feasibility"]["false_anchor_fraction"],
             "change_free_ceiling_rate": r["feasibility"]["change_free_ceiling_rate"],
             "window_read_span_in_change_free_sd":
                 r["feasibility"]["window_read_span_in_change_free_sd"]}
            for r in rows],
    }


def _pilot_steps() -> Tuple[int, ...]:
    """`core.pythia_registry.PYTHIA_410M_PILOT_STEPS`, reconstructed. That module
    imports `transformers` at scope, so it cannot be imported from a tool that
    must run in the pure tier's environment; `_check_the_schedule_copies` reads
    the registry's assignment out of the source and compares them."""
    return tuple(sorted(set(
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        + list(range(1000, 20001, 2000))
        + [40000, 60000, 80000, 100000, 120000, 143000])))


# ---------------------------------------------------------------------------
# C. the feasible set
# ---------------------------------------------------------------------------

def _enumerate_grids() -> Dict[Tuple[int, ...], dict]:
    """
    The family, deduplicated, each grid carrying the knobs that first produced
    it. Two knob settings often give the same grid; the knobs are kept so the
    interior check can ask whether an answer rests on a bound of the sweep
    rather than guessing from the steps.
    """
    heads: List[Tuple[int, ...]] = [
        tuple(PUBLISHED_LOG_STEPS[i:]) for i in range(len(PUBLISHED_LOG_STEPS) + 1)]
    heads += [(0, 512), (512,)]
    out: Dict[Tuple[int, ...], dict] = {}
    for head in heads:
        for fine_gap in FINE_GAPS:
            for fine_hi in FINE_HIGHS:
                fine = tuple(range(1000, fine_hi + 1, fine_gap))
                for tail_gap in (None,) + TAIL_GAPS:
                    if tail_gap is None:
                        tails: List[Tuple[Optional[int], Tuple[int, ...]]] = [
                            (None, ())]
                    else:
                        tails = [(tail_hi,
                                  tuple(range(fine_hi + tail_gap, tail_hi + 1,
                                              tail_gap)))
                                 for tail_hi in TAIL_HIGHS]
                    for tail_hi, tail in tails:
                        grid = tuple(sorted(set(head + fine + tail)))
                        if len(grid) < MIN_GRID_SIZE or grid in out:
                            continue
                        out[grid] = {
                            "head_from": head[0] if head else None,
                            "fine_gap": fine_gap,
                            "fine_high": fine_hi,
                            "tail_gap": tail_gap,
                            "tail_high": tail_hi,
                        }
    return out


#: The series properties the arithmetic needs, taken from the committed
#: calibration rather than re-chosen here. PLACED there
#: (`tools/calibrate_changepoint_colocation.py`) and cited here, so every rate
#: below is a rate under the same family §6o measured on.
CHANGE_WIDTH_LOG_STEP = 0.35

#: The shortlist rule, stated before the enumeration ran. Every number is
#: either derived or cited: the ceiling rate is compared against a fifth of
#: alpha, the false-anchor share against a twentieth of the window, the read
#: span against the estimator's OWN scatter (a location difference smaller than
#: that is not one the sweep can report), and the retained share against 0.95
#: because a grid losing a twentieth of the window is losing its edges.
MAX_CEILING_RATE = ALPHA / 5.0
MAX_FALSE_ANCHOR_FRACTION = 0.05
MIN_READ_SPAN_IN_SD = 1.0
MIN_RETAINED_FRACTION = 0.95

#: The retention condition is applied ACROSS the noise range, not at one level,
#: and that was a correction rather than a design. Applied at the calibration's
#: own sigma/R alone, the rule's cheapest pick was a twelve-checkpoint grid
#: retaining 1.000 there and 0.680 at zero noise -- retention is not monotone
#: in the noise (a location below the window is pulled INTO it), so a single
#: level can sit on a peak. sigma/R is not known until the run happens, so a
#: grid that works at one value of it is not a grid anyone can commit to in
#: advance. PLACED: it is the largest curve level the rule ranges over, chosen
#: to bracket the committed calibration's own 0.02 on the high side.
RULE_NOISE_RANGE_MAX = 0.05


def _dominates(a: dict, b: dict) -> bool:
    """
    Pareto domination on the five costs the author trades: the retained share
    and the read span up, the false-anchor share and the change-free ceiling
    rate down, and the number of checkpoints down because every one of them is
    a model to load and run.
    """
    ge = (a["retained"] >= b["retained"] and a["span"] >= b["span"]
          and a["false"] <= b["false"] and a["ceiling"] <= b["ceiling"]
          and a["n"] <= b["n"])
    gt = (a["retained"] > b["retained"] or a["span"] > b["span"]
          or a["false"] < b["false"] or a["ceiling"] < b["ceiling"]
          or a["n"] < b["n"])
    return ge and gt


def feasible_set(noise_to_range: float, change_width: float) -> dict:
    """
    The enumeration, and the set the author is being asked to choose from.

    The `feasible` flag is the DEGENERACIES only -- the reference outside the
    window, some of the window retained, a read span above zero. Everything
    else is reported as a number, because what counts as a small ceiling rate
    or a usable span is a decision rather than a measurement, and a tool that
    thresholds it has taken the decision.
    """
    grids = _enumerate_grids()
    scored: List[dict] = []
    counts = {"reference_inside_window": 0, "no_retention_or_no_span": 0,
              "feasible": 0}
    for grid, knobs in grids.items():
        s = np.asarray(grid, dtype=float)
        f = grid_feasibility(s, CLAIM_B_ANCHOR_WINDOW,
                             noise_to_range=noise_to_range,
                             change_width_log_step=change_width)
        if not f["reference_outside_window"]:
            counts["reference_inside_window"] += 1
            continue
        if not f["feasible"]:
            counts["no_retention_or_no_span"] += 1
            continue
        counts["feasible"] += 1
        scored.append({
            "grid": grid,
            "knobs": knobs,
            "n": f["n_checkpoints"],
            "retained": f["retained_window_fraction"],
            "false": f["false_anchor_fraction"],
            "ceiling": f["change_free_ceiling_rate"],
            "span": f["window_read_span_in_change_free_sd"],
            "side": f["margin_side"],
            "any_retention_budget": f["noise_to_range_for_any_retention"],
            "retention_curve": f["retention_curve"],
            "worst_retained_over_the_noise_range": min(
                p["retained_window_fraction"] for p in f["retention_curve"]
                if p["noise_to_range"] <= RULE_NOISE_RANGE_MAX),
        })

    ordered = sorted(scored, key=lambda r: (-r["retained"], -r["span"],
                                            r["false"], r["ceiling"], r["n"]))
    front: List[dict] = []
    for row in ordered:
        if not any(_dominates(f, row) for f in front):
            front.append(row)
    front.sort(key=lambda r: (r["n"], r["ceiling"]))

    best = _shortlist(scored, change_width, noise_to_range)
    return {
        "_what": ("every grid in the enumerated family, scored on the five "
                  "conditions and reduced to a Pareto frontier."),
        "_the_family": {
            "shape": "head + fine + tail, the shape core/pythia_registry.py "
                     "already writes",
            "heads": "suffixes of (0, 1, 2, 4, ..., 512), plus (0, 512), "
                     "(512,) and none",
            "fine_gaps": list(FINE_GAPS),
            "fine_highs": [min(FINE_HIGHS), max(FINE_HIGHS)],
            "tail_gaps": list(TAIL_GAPS),
            "tail_highs": list(TAIL_HIGHS),
            "min_grid_size": MIN_GRID_SIZE,
            "_not_the_power_set": (
                "2^154 subsets of Pythia's published schedule is not "
                "enumerable and almost none of them is a schedule anyone would "
                "run. What is claimed is that this family contains the shapes "
                "this project writes and that the answer is interior to it -- "
                "`frontier_is_interior_to_the_family` checks the second."),
        },
        "noise_to_range": noise_to_range,
        "change_width_log_step": change_width,
        "n_grids_enumerated": len(grids),
        "counts": counts,
        "published_checkpoints_strictly_inside_the_window": [
            int(v) for v in PUBLISHED_LINEAR_STEPS
            if CLAIM_B_ANCHOR_WINDOW[0] < v < CLAIM_B_ANCHOR_WINDOW[1]],
        "_what_that_one_checkpoint_costs": (
            "the window 512-2000 contains exactly ONE published checkpoint, "
            "step 1000, so in the sharp-change limit no sweep can distinguish "
            "more than two locations inside it. A real change is wider than an "
            "interval, which is what makes the read span continuous rather "
            "than a count -- but that single interior checkpoint is why the "
            "span is around two of the estimator's own standard deviations on "
            "the best grids rather than ten."),
        "n_on_the_frontier": len(front),
        "frontier": [_frontier_row(r) for r in front],
        "shortlist": best,
        "frontier_is_interior_to_the_family": _interior(
            best, [_frontier_row(r) for r in front]),
    }


def _frontier_row(r: dict) -> dict:
    return {
        "steps": [int(v) for v in r["grid"]],
        "knobs": r.get("knobs"),
        # What the grid leaves out below step 1000, and it is a cost rather
        # than a detail: adding those checkpoints pulls the sweep's midpoint
        # down into the window, so a grid that clears the conditions may be
        # one that drops checkpoints another prediction wants. "A denser sweep
        # is worse" (6o) reaching the early end of the schedule.
        "omits_published_log_steps": [
            int(v) for v in PUBLISHED_LOG_STEPS if v not in set(r["grid"])],
        "n_checkpoints": r["n"],
        "retained_window_fraction": r["retained"],
        "false_anchor_fraction": r["false"],
        "change_free_ceiling_rate": r["ceiling"],
        "window_read_span_in_change_free_sd": r["span"],
        "worst_retained_over_the_noise_range":
            r.get("worst_retained_over_the_noise_range"),
        "retention_curve": r.get("retention_curve"),
        "margin_side": r["side"],
        "noise_to_range_for_any_retention": r["any_retention_budget"],
    }


#: How close to the best achievable worst-case retention a grid has to be to
#: count as "as good as it gets". PLACED, and it is a tie-band rather than a
#: threshold: it exists so the shortlist can offer a cheaper grid that gives up
#: nothing worth having, not to admit or exclude anything.
RETENTION_TIE_BAND = 0.02


def _shortlist(scored: Sequence[dict], change_width: float,
               noise_to_range: float) -> dict:
    """
    The grids worth putting to the author.

    THREE HARD CONDITIONS AND THEN A MAXIMUM, NOT FOUR THRESHOLDS. The ceiling
    rate, the false-anchor share and the read span are compared against bounds
    fixed before the enumeration ran. Retention is **maximised** rather than
    thresholded, and that is a correction the enumeration forced: applied as a
    threshold at 0.95 across the noise range it admitted **nothing**, and the
    honest reading of that is not that the bound is too strict but that no
    published Pythia schedule holds the whole anchor window. Reporting the
    achievable maximum says so; tuning the threshold until something passed
    would have hidden it.

    Three or four grids rather than one, because the axis the author trades on
    is not in this arithmetic: how many checkpoints the run can afford, and
    whether the sweep must also serve the predictions that need late
    checkpoints.
    """
    hard = [r for r in scored
            if r["ceiling"] < MAX_CEILING_RATE
            and r["false"] < MAX_FALSE_ANCHOR_FRACTION
            and r["span"] >= MIN_READ_SPAN_IN_SD]
    picks: Dict[str, object] = {
        "_rule": {
            "hard_conditions": {
                "max_change_free_ceiling_rate": MAX_CEILING_RATE,
                "max_false_anchor_fraction": MAX_FALSE_ANCHOR_FRACTION,
                "min_read_span_in_change_free_sd": MIN_READ_SPAN_IN_SD,
            },
            "then_maximise": "worst_retained_over_the_noise_range",
            "retention_required_over_noise_to_range_up_to": RULE_NOISE_RANGE_MAX,
            "retention_tie_band": RETENTION_TIE_BAND,
            "reference_retention_that_no_grid_reaches": MIN_RETAINED_FRACTION,
            "at_noise_to_range": noise_to_range,
            "at_change_width_log_step": change_width,
            "_why_these": (
                "fixed before the enumeration ran. The ceiling rate is a fifth "
                "of alpha; the read span is measured in the estimator's OWN "
                "scatter rather than against a placed distance; the "
                "false-anchor bound is the one plainly placed number and is "
                "reported per grid so a reader can move it. Retention is "
                "maximised rather than bounded, because as a bound at "
                f"{MIN_RETAINED_FRACTION} it admitted nothing and that is a "
                "fact about Pythia's schedule rather than about the bound. It "
                "is taken as the WORST value over the noise range rather than "
                "the value at one level, because retention is not monotone in "
                "the noise -- a location below the window is pulled into it -- "
                "and sigma/R is not known until the run happens. Applied at "
                "the calibration's own level alone the rule's cheapest pick "
                "retained 1.000 there and 0.680 at zero noise."),
        },
        "_n_grids_meeting_the_hard_conditions": len(hard),
    }
    if not hard:                                 # pragma: no cover - defensive
        picks["_none"] = ("no grid in the family met the hard conditions; the "
                          "rule is in the record so it can be re-read rather "
                          "than guessed")
        return picks

    best_w = max(r["worst_retained_over_the_noise_range"] for r in hard)
    near = [r for r in hard
            if r["worst_retained_over_the_noise_range"] >= best_w - RETENTION_TIE_BAND]
    picks["best_achievable_worst_case_retention"] = float(best_w)
    picks["no_grid_reaches_the_reference_retention"] = bool(
        best_w < MIN_RETAINED_FRACTION)
    picks["_what_that_means"] = (
        "the anchor window's upper end is where retention is lost, and it is "
        "lost at ZERO noise as much as at high noise. A change centred near "
        "step 2000 puts half its mass above 2000, and the next published "
        "checkpoint a feasible grid can afford there is tens of thousands of "
        "steps away, so that mass is collected at an interval midpoint far "
        "outside the window. Adding checkpoints between 2000 and 20000 fixes "
        "it and pulls the sweep's own midpoint back INTO the window, which is "
        "the refusal. That is the trade this claim's anchor window and "
        "Pythia's release schedule make between them, and no choice of grid "
        "removes it.")
    picks["best_retention"] = _frontier_row(
        max(hard, key=lambda r: (r["worst_retained_over_the_noise_range"],
                                 -r["n"])))
    picks["cheapest_at_essentially_that_retention"] = _frontier_row(
        min(near, key=lambda r: (r["n"],
                                 -r["worst_retained_over_the_noise_range"])))
    picks["largest_read_span"] = _frontier_row(
        max(hard, key=lambda r: (r["span"],
                                 r["worst_retained_over_the_noise_range"])))
    long_run = [r for r in hard if max(r["grid"]) >= 100000]
    if long_run:
        best_long = max(long_run,
                        key=lambda r: (r["worst_retained_over_the_noise_range"],
                                       -r["n"]))
        picks["best_reaching_past_step_100000"] = _frontier_row(best_long)
        picks["_what_reaching_that_far_costs"] = float(
            best_w - best_long["worst_retained_over_the_noise_range"])
    return picks


def _interior(shortlist: dict, front: Sequence[dict]) -> dict:
    """
    Whether the answer rests on a bound of the enumerated family, which is the
    one way a family-based answer goes wrong: if the shortlist uses the largest
    `fine_high` on offer, the real optimum is outside the sweep and the bound
    is the answer rather than the arithmetic.

    Only the ARTIFICIAL bounds count. The smallest fine gap is Pythia's own
    release granularity above step 512 and the largest tail high is the last
    checkpoint EleutherAI published, so a grid resting on either rests on the
    data and not on this file. Distinguishing the two is the whole content of
    the check; a version that treated every bound alike reported a failure on a
    four-point grid at a resolution nobody would run.
    """
    picks = [row for key, row in shortlist.items()
             if not key.startswith("_") and isinstance(row, dict)]
    if not picks:
        return {"checked": False,
                "interior": False,
                "_why": "the shortlist is empty, so there is nothing to check"}
    on_artificial = []
    for row in picks:
        knobs = row.get("knobs") or {}
        touched = []
        if row["n_checkpoints"] <= MIN_GRID_SIZE:
            touched.append("smallest_grid")
        if knobs.get("fine_high") == max(FINE_HIGHS):
            touched.append("largest_fine_high")
        if knobs.get("tail_gap") == max(TAIL_GAPS):
            touched.append("largest_tail_gap")
        if touched:
            on_artificial.append({"steps": row["steps"], "bounds": touched})
    sizes = [len(row["steps"]) for row in front] or [0]
    tops = [max(row["steps"]) for row in front] or [0]
    return {
        "checked": True,
        "artificial_bounds": {
            "largest_fine_high": max(FINE_HIGHS),
            "largest_tail_gap": max(TAIL_GAPS),
            "smallest_grid": MIN_GRID_SIZE,
        },
        "real_bounds": {k: {"value": v[0], "why": v[1]}
                        for k, v in REAL_BOUNDS.items()},
        "shortlist_rows_resting_on_an_artificial_bound": on_artificial,
        "frontier_size_range": [int(min(sizes)), int(max(sizes))],
        "frontier_max_step_range": [int(min(tops)), int(max(tops))],
        "interior": bool(not on_artificial),
        "_note": (
            "the frontier spanning a range of sweep sizes and tops rather than "
            "piling on one place is the softer version of the same evidence, "
            "and it is reported rather than scored: a frontier legitimately "
            "reaches the last published checkpoint, because that is where "
            "training stopped."),
    }


def retention_curve_check(rng, grid: Sequence[int], reps: int) -> dict:
    """
    The curve a pilot actually reads, measured.

    `retention_curve` is what the author will check their own sigma/R against,
    so it is worth more than a derivation. A change planted uniformly inside
    the window is simulated at each level of the curve and the share whose
    location still READS inside the window is reported beside the prediction.

    This section exists because the first version of the arithmetic did not
    have it, and the defect it would have caught went out in a smoke run
    instead: a ten-checkpoint grid the sharp-change reading scored at retention
    1.000 put a planted anchor inside the window on 0.017 of draws. The
    prediction and the measurement were in the same record and nothing
    compared them.
    """
    s = np.asarray(grid, dtype=float)
    f = grid_feasibility(s, CLAIM_B_ANCHOR_WINDOW, noise_to_range=NOISE_SD,
                         change_width_log_step=CHANGE_WIDTH_LOG_STEP)
    x_lo, x_hi = (float(step_x([CLAIM_B_ANCHOR_WINDOW[0]])[0]),
                  float(step_x([CLAIM_B_ANCHOR_WINDOW[1]])[0]))
    rows = []
    for point in f["retention_curve"]:
        sigma = point["noise_to_range"]
        inside = []
        for _ in range(reps):
            x_true = rng.uniform(x_lo, x_hi)
            v = _logistic(s, 10.0 ** x_true - 1.0, rng, sigma)
            cen = change_profile(s, v, "rise")["centroid_log_step"]
            inside.append(x_lo <= cen <= x_hi)
        rows.append({
            "noise_to_range": sigma,
            "predicted_retained_fraction": point["retained_window_fraction"],
            "measured_share_read_inside": float(np.mean(inside)),
        })
    err = [abs(r["predicted_retained_fraction"] - r["measured_share_read_inside"])
           for r in rows]
    return {
        "_what": ("a change planted uniformly inside the window, on the "
                  "shortlist's cheapest grid, at every level of that grid's "
                  "own retention curve."),
        "grid": [int(v) for v in grid],
        "change_width_log_step": CHANGE_WIDTH_LOG_STEP,
        "n_reps_per_cell": reps,
        "rows": rows,
        "worst_absolute_error": float(max(err)),
        "curve_tracks_the_measurement": bool(max(err) <= 0.15),
        "_note": (
            "the curve is not a bound in either direction and is not presented "
            "as one. The NOISE half of it is bounded above -- rectified noise "
            "adds at most E[Z+] per interval -- but the WIDTH half is not: how "
            "much of a change's mass a coarse interval collects is a fact "
            "about the grid and the change together. What is claimed is "
            "agreement, and the worst disagreement is stored."),
    }


# ---------------------------------------------------------------------------
# D. the measurement, on the grids the arithmetic picks
# ---------------------------------------------------------------------------

def boundary_measurement(rng, reps: int, alpha: float,
                         picks: Dict[str, Tuple[str, Tuple[int, ...]]]) -> dict:
    """
    The anchor arm's actual rates on the shortlist and on the grids the project
    has, so the arithmetic is checked against the thing it is arithmetic about.

    The discriminating power is the DIFFERENCE between the rate on an anchored
    change and the rate on a series with no located change at all, measured on
    the same cell against the same control draws — `claim_b_p_i1_dry_run.json`
    arm B's construction, reused so the two records compare row for row. Either
    rate alone is a proportion over a few hundred draws; a discriminating arm
    cannot have the difference near zero however the sampling falls.

    Scored WITHOUT the refusal in front, because a refused grid has to reach a
    rate for the comparison to mean anything.
    """
    rows = []
    for name, (kind, grid) in picks.items():
        s = np.asarray(grid, dtype=float)
        feas = grid_feasibility(s, CLAIM_B_ANCHOR_WINDOW, noise_to_range=NOISE_SD,
                                change_width_log_step=CHANGE_WIDTH_LOG_STEP)
        for family in ("localized", "mixed"):
            rates: Dict[str, float] = {}
            # `input_kind` rather than `kind`: the outer loop's `kind` is the
            # grid's, and shadowing it here wrote "no-change" into every row's
            # grid kind, which emptied every summary list and turned three of
            # `check_record`'s findings into false alarms.
            for input_kind in ("anchor", "random", "no-change"):
                hits = 0
                for _ in range(reps):
                    under_test = series(input_kind, s, N_UNITS, rng, NOISE_SD)
                    ctrl, dirs = controls(family, s, N_UNITS, N_CONTROLS, rng,
                                          NOISE_SD)
                    hits += _scored(s, under_test, ctrl, dirs)["p_value"] <= alpha
                rates[input_kind] = hits / reps
            rows.append({
                "grid": name,
                "kind": kind,
                "n_checkpoints": int(s.size),
                "control_family": family,
                "arm_refuses": bool(_refuses(s)),
                "predicted_retained_window_fraction":
                    feas["retained_window_fraction"],
                "predicted_false_anchor_fraction": feas["false_anchor_fraction"],
                "predicted_window_read_span_in_change_free_sd":
                    feas["window_read_span_in_change_free_sd"],
                "predicted_change_free_ceiling_rate":
                    feas["change_free_ceiling_rate"],
                "reject_planted_at_anchor": rates["anchor"],
                "reject_h0_change_elsewhere": rates["random"],
                "reject_no_located_change": rates["no-change"],
                "discrimination": rates["anchor"] - rates["no-change"],
            })
    computed = [r for r in rows if r["kind"] == "computed"]
    project = [r for r in rows if r["kind"] == "project"]
    probe = [r for r in rows if r["kind"] == "probe"]
    localized = [r for r in project if r["control_family"] == "localized"]
    return {
        "_what": ("the anchor arm's measured rates on the shortlist and on the "
                  "grids this project has, under both control families."),
        "_the_three_kinds_of_grid": (
            "`project` are the schedules this repository contains -- the "
            "registered cheap sweep, `core/pythia_registry.py`'s 410M pilot "
            "schedule, and Pythia's full every-1000 release. `probe` is "
            "`early-dense-73`, which §6o INVENTED for its own sweep and nobody "
            "proposed running; it is here because the first version of this "
            "section lumped it in with the others and reported that no existing "
            "grid discriminates, which is false of it. `computed` are the "
            "shortlist."),
        "n_reps_per_cell": reps,
        "n_units": N_UNITS,
        "n_controls": N_CONTROLS,
        "noise_sd": NOISE_SD,
        "alpha": alpha,
        "rows": rows,
        "worst_computed_grid_discrimination":
            float(min(r["discrimination"] for r in computed)) if computed else None,
        "worst_computed_grid_change_free_rate":
            float(max(r["reject_no_located_change"] for r in computed))
            if computed else None,
        "best_project_grid_discrimination_on_localized_controls":
            float(max(r["discrimination"] for r in localized)) if localized else None,
        "probe_grid_change_free_rate":
            float(max(r["reject_no_located_change"] for r in probe))
            if probe else None,
        "computed_grids_discriminate_and_project_ones_do_not": bool(
            computed and localized
            and min(r["discrimination"] for r in computed) >= 0.9
            and max(r["discrimination"] for r in localized) <= 0.1),
        "computed_grids_reject_nothing_on_a_change_free_input": bool(
            computed and max(r["reject_no_located_change"] for r in computed) <= 0.02),
        "_the_finding": (
            "on the grids this arithmetic picks the anchor arm rejects on an "
            "anchored change and not on a change-free one. Every schedule this "
            "repository contains discriminates at 0.000 against the `localized` "
            "control family, the registered cheap sweep and the 410M pilot "
            "schedule because a change-free series attains the arm's ceiling "
            "there and the dense sweep because an anchored change is dragged "
            "out of the window. The registered sweep's failure is not that it "
            "is cheap: a sweep of the same size, spent on every-1000 from 512 "
            "rather than on a coarse tail out to 143000, discriminates at "
            "1.000."),
        "_and_what_the_probe_grid_says": (
            "§6o's `early-dense-73` DOES discriminate -- it is the grid §6o "
            "built to show that discrimination recovers off the registered "
            "sweep, and it does. What it does not do is work: at this noise "
            "level it retains about half the window, reads a full window-width "
            "of sweep outside the window as anchored, resolves the window into "
            "ONE location rather than two, and rejects on a change-free input "
            "on a fifth of draws. `discriminates` and `is a usable grid` are "
            "different questions, and only the second is what the enumeration "
            "answers."),
        "_and_the_residual_6o_left": (
            "§6o measured a change-free series clearing alpha on 0.245 of "
            "draws against the `localized` family on a grid whose midpoint sits "
            "just outside the window, and left it unrefused as a rate the "
            "analyst must discount. It is not irreducible. On a grid whose "
            "change-free reference sits FAR from the window it is 0.000 here, "
            "because the reference then ranks below controls that carry a real "
            "change rather than above them. What remains family-dependent is "
            "the `mixed` rows, where a quarter of the controls are themselves "
            "change-free: there the observed change-free series is competing "
            "against its own kind and no grid removes that."),
    }


# ---------------------------------------------------------------------------
# The staleness check
# ---------------------------------------------------------------------------

#: The two schedule constants in `core/pythia_registry.py` this file copies.
_REGISTRY_PATH = ROOT / "core" / "pythia_registry.py"
_REGISTRY_NAMES = ("PYTHIA_410M_PILOT_STEPS", "PYTHIA_ALL_STEPS")


def _registry_schedules() -> Dict[str, Tuple[int, ...]]:
    """
    `core/pythia_registry.py`'s checkpoint schedules, read out of the SOURCE
    rather than imported.

    That module imports `transformers` at scope, so it cannot be imported from
    the gating tier or from this tool -- which is why the schedule is copied
    here at all. A copy nobody compares is exactly the hand-synced constant the
    repo's own lint rule 3 exists to find, so the copy is checked against the
    original the only way that works without the dependency: the two assignments
    are located with `ast` and their right-hand sides evaluated with `sorted`,
    `set`, `list` and `range` and nothing else in scope. That is narrower than
    importing the module and it runs everywhere, which is the point.
    """
    tree = ast.parse(_REGISTRY_PATH.read_text(encoding="utf-8"))
    safe = {"__builtins__": {}, "sorted": sorted, "set": set, "list": list,
            "range": range}
    out: Dict[str, Tuple[int, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in _REGISTRY_NAMES:
                out[target.id] = tuple(
                    eval(compile(ast.Expression(node.value), "<registry>",  # noqa: S307
                                 "eval"), safe, {}))
    return out


def _check_the_schedule_copies() -> List[str]:
    """The published-schedule constants here against the registry's own."""
    try:
        reg = _registry_schedules()
    except Exception as exc:                     # pragma: no cover - defensive
        return [f"could not read core/pythia_registry.py's schedules: {exc}"]
    bad = []
    missing = [n for n in _REGISTRY_NAMES if n not in reg]
    if missing:
        return [f"core/pythia_registry.py no longer defines {missing}; the "
                f"schedule copy in this tool has nothing to check against"]
    if tuple(_pilot_steps()) != tuple(reg["PYTHIA_410M_PILOT_STEPS"]):
        bad.append("the 410M pilot schedule reconstructed here no longer "
                   "matches core/pythia_registry.PYTHIA_410M_PILOT_STEPS")
    published = set(PUBLISHED_LOG_STEPS) | set(PUBLISHED_LINEAR_STEPS)
    if published != set(reg["PYTHIA_ALL_STEPS"]):
        bad.append("the published-checkpoint copy here no longer matches "
                   "core/pythia_registry.PYTHIA_ALL_STEPS")
    return bad


def check_record(doc: dict) -> List[str]:
    """
    What must still be true of the committed record. Returns the failures.

    It fails if the FINDING stops being in the file, not only if a field goes
    missing: the point of this record is that a computed grid discriminates
    where the two the project has do not, and a record that no longer shows
    that does not support the section it is the evidence for. Same posture as
    `claim_b_p_i1_dry_run.json`'s, one pass later.
    """
    bad: List[str] = []
    if doc.get("schema_version") != SCHEMA_VERSION:
        bad.append(f"schema_version {doc.get('schema_version')} != {SCHEMA_VERSION}")
    if doc.get("construction_sha256") != _sha256(CONSTRUCTION_PATH):
        bad.append("construction_sha256 does not match core/changepoint_colocation.py")
    if list(doc.get("anchor_window_steps", [])) != list(CLAIM_B_ANCHOR_WINDOW):
        bad.append("anchor_window_steps do not match the registered window")

    cf = doc.get("closed_forms", {})
    if not cf.get("retention_tracks_the_measurement"):
        bad.append("the retention arithmetic no longer tracks the measurement")
    if not cf.get("the_sharp_limit_is_not_a_bound"):
        bad.append("the sharp-change limit stopped overstating retention, "
                   "which is the whole argument for requiring the change "
                   "width as an input")
    if not cf.get("sd_closed_form_tracks_the_measurement"):
        bad.append("the change-free spread closed form no longer tracks the "
                   "measurement")
    if not cf.get("adjacent_covariance_term_is_load_bearing"):
        bad.append("the adjacent-covariance term stopped mattering; if that is "
                   "real the simpler form should replace it rather than the "
                   "record carrying an argument for something else")

    cat = doc.get("catalogue", {})
    if not cat.get("condition_1_matches_the_arms_refusal"):
        bad.append("grid_feasibility's condition 1 and anchor_arm's refusal "
                   "disagree on a catalogued grid")
    if not cat.get("none_of_them_meets_the_hard_conditions"):
        bad.append("a grid this project already has now passes; 6o's premise "
                   "should be re-read rather than this check relaxed")

    fs = doc.get("feasible_set", {})
    if len(fs.get("published_checkpoints_strictly_inside_the_window", [])) != 1:
        bad.append("the window no longer contains exactly one published "
                   "checkpoint in its interior, which is what fixes the "
                   "resolution")
    if not fs.get("frontier_is_interior_to_the_family", {}).get("interior"):
        bad.append("the frontier sits on an edge of the enumerated family, so "
                   "the family's bound is the answer rather than the arithmetic")
    short = fs.get("shortlist", {})
    if not short.get("best_retention"):
        bad.append("the shortlist is empty")
    if short.get("no_grid_reaches_the_reference_retention") is not True:
        bad.append("some grid now holds 95% of the window across the noise "
                   "range; that is a better answer than this record gives and "
                   "the section should be rewritten rather than this check "
                   "relaxed")

    bc = doc.get("retention_curve_check", {})
    if not bc.get("curve_tracks_the_measurement"):
        bad.append("the retention curve no longer tracks the measurement, and "
                   "it is the curve a pilot would check its own series "
                   "against")

    bad.extend(_check_the_schedule_copies())

    bm = doc.get("boundary_measurement", {})
    if not bm.get("computed_grids_discriminate_and_project_ones_do_not"):
        bad.append("the finding is gone: either a computed grid stopped "
                   "discriminating or a schedule this repository contains "
                   "started")
    if not bm.get("computed_grids_reject_nothing_on_a_change_free_input"):
        bad.append("a computed grid now rejects on a series with no located "
                   "change, which is the residual §6o left unrefused coming "
                   "back")
    return bad


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=N_REPS,
                    help="replicates per measured cell (default %(default)s)")
    ap.add_argument("--seed", type=int, default=_SEED)
    ap.add_argument("--write", action="store_true",
                    help=f"write {OUT_PATH.relative_to(ROOT)}")
    ap.add_argument("--check", action="store_true",
                    help="check the committed record against the current code "
                         "and exit non-zero if it has gone stale")
    args = ap.parse_args(argv)

    if args.check:
        if not OUT_PATH.exists():
            print(f"missing {OUT_PATH.relative_to(ROOT)}")
            return 1
        problems = check_record(json.loads(OUT_PATH.read_text()))
        for p in problems:
            print(f"STALE: {p}")
        print("record is in step with the code" if not problems
              else f"{len(problems)} problem(s)")
        return 1 if problems else 0

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    print("closed forms ...", flush=True)
    cf = closed_forms(rng, max(400, args.reps * 2))
    for r in cf["change_free_centroid"]:
        print(f"  {r['grid']:22s} sd pred={r['predicted_sd_log_step']:.4f} "
              f"meas={r['measured_sd_log_step']:.4f} "
              f"ratio={r['sd_ratio_predicted_over_measured']:.3f} "
              f"(naive {r['that_naive_sd_ratio']:.2f})", flush=True)

    print("\ncatalogue ...", flush=True)
    cat = catalogue()
    for r in cat["rows"]:
        f = r["feasibility"]
        print(f"  {r['grid']:22s} n={f['n_checkpoints']:3d} "
              f"ref@step={f['uniform_profile_centroid_step']:9.0f} "
              f"{f['margin_side']:6s} rho={f['retained_window_fraction']:.3f} "
              f"false={f['false_anchor_fraction']:.3f} "
              f"ceil={f['change_free_ceiling_rate']:.4f} "
              f"span={f['window_read_span_in_change_free_sd']:.2f}", flush=True)

    print("\nfeasible set ...", flush=True)
    fs = feasible_set(NOISE_SD, CHANGE_WIDTH_LOG_STEP)
    print(f"  {fs['n_grids_enumerated']} grids enumerated; "
          f"{fs['counts']['feasible']} feasible", flush=True)
    print(f"  {fs['n_on_the_frontier']} on the Pareto frontier; "
          f"{fs['shortlist']['_n_grids_meeting_the_hard_conditions']} meet "
          f"the hard conditions; best worst-case retention "
          f"{fs['shortlist'].get('best_achievable_worst_case_retention')}",
          flush=True)
    for key in ("best_retention", "cheapest_at_essentially_that_retention",
                "largest_read_span", "best_reaching_past_step_100000"):
        row = fs["shortlist"].get(key)
        if row:
            print(f"  {key:38s} n={row['n_checkpoints']:3d} "
                  f"worst_rho={row['worst_retained_over_the_noise_range']:.3f} "
                  f"span={row['window_read_span_in_change_free_sd']:.2f} "
                  f"steps={row['steps'][:4]}...{row['steps'][-1]}", flush=True)

    picks: Dict[str, Tuple[str, Tuple[int, ...]]] = {
        "cheap-25 (registered)": ("project", CHEAP_SWEEP),
        "pythia-410m-pilot-27": ("project", _pilot_steps()),
        "dense-154": ("project", DENSE_SWEEP),
        "early-dense-73": ("probe", EARLY_DENSE_SWEEP),
    }
    for key in ("best_retention", "cheapest_at_essentially_that_retention",
                "largest_read_span", "best_reaching_past_step_100000"):
        row = fs["shortlist"].get(key)
        if not row:
            continue
        steps = tuple(row["steps"])
        # Deduplicated: the best-retention grid is often also the cheapest at
        # that retention, and measuring one grid twice buys nothing.
        if any(g == steps for _, g in picks.values()):
            continue
        picks[f"computed: {key}"] = ("computed", steps)

    print("\nretention curve check ...", flush=True)
    bc = retention_curve_check(
        rng, fs["shortlist"]["best_retention"]["steps"], args.reps)
    for r in bc["rows"]:
        print(f"  sigma/R={r['noise_to_range']:.4f} "
              f"pred={r['predicted_retained_fraction']:.3f} "
              f"meas={r['measured_share_read_inside']:.3f}", flush=True)

    print("\nboundary measurement ...", flush=True)
    bm = boundary_measurement(rng, args.reps, ALPHA, picks)
    for r in bm["rows"]:
        print(f"  {r['grid']:42s} {r['control_family']:10s} "
              f"anchored={r['reject_planted_at_anchor']:.3f} "
              f"h0={r['reject_h0_change_elsewhere']:.3f} "
              f"no-change={r['reject_no_located_change']:.3f} "
              f"disc={r['discrimination']:+.3f}", flush=True)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "_what": ("which checkpoint sweeps can carry CLAIM-B's anchor arms, "
                  "enumerated over Pythia's published schedule before any data "
                  "exists."),
        "_why": ("POPPER_PLAN.md 6o left CLAIM-B needing a grid neither of the "
                 "two the project has can supply, and said choosing one is the "
                 "author's decision. Which grids CLEAR the conditions is "
                 "arithmetic, so it is computed here and the decision becomes a "
                 "choice from a set."),
        "_not": ("it chooses no grid, registers nothing and adjudicates "
                 "nothing. The series are synthetic, no Pythia sweep artifact "
                 "is in this repository and claims/adjudications/ is empty."),
        "generated_by": "tools/claim_b_grid_feasibility.py",
        "construction_file": "core/changepoint_colocation.py",
        "construction_sha256": _sha256(CONSTRUCTION_PATH),
        "anchor_window_steps": list(CLAIM_B_ANCHOR_WINDOW),
        "alpha": ALPHA,
        "seed": args.seed,
        "closed_forms": cf,
        "catalogue": cat,
        "feasible_set": fs,
        "retention_curve_check": bc,
        "boundary_measurement": bm,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    problems = check_record(doc)
    doc["self_check"] = {"problems": problems, "clean": not problems}
    if problems:
        print("\nself-check problems:")
        for p in problems:
            print(f"  {p}")

    if not args.write:
        print("\n(not written: pass --write)")
        return 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)} in {doc['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":                       # pragma: no cover - entrypoint
    raise SystemExit(main())
