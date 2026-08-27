"""
tools/dry_run_claim_b_p_i1.py — CLAIM-B and P-I1 on inputs whose answer is known.

`claims/EVALUABILITY.md`'s queue: every converted row is owed a run on an input
whose correct verdict is fixed a priori, ahead of converting the next one.
`CLAIM-C` had it (2026-08-25), `P-ST1` (2026-08-26), `P6-R2`/`P6-R4`
(2026-08-26). These two are the fifth and sixth, and they share one estimator
-- `core/changepoint_colocation.py` -- so one dry run covers both, which is why
they were built together in the first place. The record is
`claims/audits/claim_b_p_i1_dry_run.json`; `POPPER_PLAN.md` 6o reads it.

WHAT IT FOUND, AND THE TWO ENTRIES COME BACK DIFFERENTLY AGAIN

A change location is the centroid of a change-mass profile: a weighted mean of
the sweep's interval midpoints. Mass spread evenly over the sweep therefore
lands on the grid's OWN midpoint -- exactly, by the definition of a mean, not
as an approximation. Per-checkpoint noise puts rectified mass in every
interval, so any real location is a mixture of where the series changed and
where the grid's midpoint is, weighted by the noise's share of total change
mass. That share grows with the interval count, so a DENSER sweep is worse.

`CLAIM-B`'s ANCHOR ARMS have it, and on the registered sweep it is total. The
20-30 checkpoint cheap-tier sweep the registry names as this prediction's
instrument has its midpoint at step 955 -- inside CLAIM-B's own 512-2000
anchor window. So a series that changes NOWHERE receives the arm's maximum
statistic, and measured against controls localized away from the window the
arm rejects on a change-free input at exactly the rate it rejects on a
perfectly anchored one. Its discriminating power there is zero.

`P-I1` does NOT have it, and the reason is structural rather than lucky. P-I1
is the MUTUAL arm alone: a difference of two locations, with a null that
permutes the pairing and so keeps both series' real locations on both sides of
every draw. The pull toward the grid's midpoint is common to both and cancels.
Measured on the same families that take the anchor arm to 1.000, the mutual arm
holds at nominal -- so `P-I1` is left alone, and that is a decision with a
number behind it, the precedent `P6-R4` set on 2026-08-26.

WHICH SHARPENS 6n's TAXONOMY RATHER THAN REPEATING IT

6n asked whether a statistic cancels a common ELEVATION of both arms, and put
`P6-R4` -- one subspace against matched controls -- in the safe column with
"nothing to mismatch". This is the counter-example that says what that column
actually requires: an absolute quantity against matched controls is safe only
when the controls are matched on the quantity the statistic DEGENERATES on.
`P6-R4`'s controls are matched on dimension, which is what drives its
statistic. The anchor arm's controls are matched on the sweep and the units,
and what drives its statistic is where the grid puts a profile that carries no
location. "Matched on what" has to name the statistic's degenerate input.

THE REFUSAL THIS ADDED, AND IT IS NOT FREE

`anchor_arm` now scores the change-free reference against the same controls by
the same ranking rule and refuses when the reference itself would clear alpha.
Nothing is placed: the reference comes from the step grid, the rank from the
controls, the cut from alpha. Unlike 6l's informative-row refusal (measured to
cost zero power) and 6m's attainable-floor refusal (costs none by
construction), THIS ONE COSTS VERDICTS -- including on inputs whose change
really is at the anchor. What it turns away is a verdict the design cannot
SUPPORT rather than one it could not REACH, which is a third category, and arm
D re-scores the counterfactual rather than asserting the cost is small.

FIVE ARMS

A. `known_answer`     -- both gates on inputs whose verdict is fixed a priori,
                         including every verdict branch, on a grid where the
                         anchor arm can discriminate.
B. `anchor_discrimination` -- the finding: reject rates on a planted-at-anchor
                         input and on a change-free one, per grid and per
                         control family, with their difference.
C. `grid_pull`        -- the mechanism, as a closed form checked against the
                         measurement: centroid = mixture of the true location
                         and the grid midpoint, weighted by the noise share.
D. `refusal_cost`     -- what the new refusal turns away, re-scored.
E. `p_i1_unaffected`  -- the mutual arm on the families that break the anchor
                         arm, which is what makes leaving P-I1 alone a decision.

WHAT THIS DELIBERATELY DOES NOT DO

It adjudicates nothing: the series are synthetic, no Pythia sweep artifact is
in this repository, and `claims/adjudications/` stays empty. It does not touch
the mutual arm, whose null is unchanged and now has a measurement behind that.
It does not redesign the estimator: correcting a centroid for its noise share
is a second pre-registered decision of the same class as CLAIM-C's criterion,
and this pass reports the quantity rather than dividing by it.

RUN IT

    python3 -m tools.dry_run_claim_b_p_i1 --write      # minutes, not tens
    python3 -m tools.dry_run_claim_b_p_i1 --check
    python3 -m tools.dry_run_claim_b_p_i1 --summary

The generation cost is measured on every write and stored as `elapsed_seconds`
in the record, rather than quoted here where it would go stale -- which is what
happened to `tools/dry_run_p6_r2_r4.py`, whose stated twenty minutes became
thirty-five when a section was added and had to be corrected in three places.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from math import erf, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.changepoint_colocation import (
    CLAIM_B_ANCHOR_CONTROL_FAMILY,
    CLAIM_B_ANCHOR_WINDOW,
    ColocationRefused,
    anchor_arm,
    anchor_statistic,
    change_profile,
    diffuse_reference_profile,
    grid_reference_report,
    interval_midpoints,
    p_value_claim_b,
    paired_colocation_arm,
)
from core.checkpoint_frames import step_x
from p7_motifs.formation_gate import p_value_p_i1

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "claims" / "audits" / "claim_b_p_i1_dry_run.json"

RECORD_SCHEMA_VERSION = 1

#: The two files every number here depends on: the shared construction, and
#: P-I1's thin half over it.
CONSTRUCTION_PATH = ROOT / "core" / "changepoint_colocation.py"
FORMATION_GATE_PATH = ROOT / "p7_motifs" / "formation_gate.py"

#: The sweep the REGISTRY names as CLAIM-B's instrument -- "20-30 checkpoint
#: cheap-tier sweep" -- and the one the committed calibration measured on. Its
#: uniform-profile midpoint is what this pass is about, so it is not a choice
#: made here.
CHEAP_SWEEP: Tuple[int, ...] = (
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000,
    8000, 13000, 23000, 33000, 43000, 63000, 83000, 103000, 123000, 143000)

#: Pythia's full every-1000 release schedule. `INDEX.md` records the dense
#: pilot sweep as not executed; it is here because it is the other grid the
#: project could actually run, not because it was tuned to pass.
DENSE_SWEEP: Tuple[int, ...] = tuple(sorted(set(
    [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 144000, 1000)))))

#: A third grid, between the two, whose midpoint sits just BELOW the window
#: rather than inside it or far above. It is what stops the finding from being
#: a two-point comparison with a line through it.
EARLY_DENSE_SWEEP: Tuple[int, ...] = tuple(sorted(set(
    list(range(0, 512, 8)) + [512, 1000, 2000, 4000, 8000, 16000, 32000,
                              64000, 143000])))

GRIDS: Dict[str, Tuple[int, ...]] = {
    "cheap-25 (registered)": CHEAP_SWEEP,
    "early-dense-73": EARLY_DENSE_SWEEP,
    "dense-154": DENSE_SWEEP,
}

#: Logistic width in log10-step and per-checkpoint noise, taken from
#: `tools/calibrate_changepoint_colocation.py` rather than re-chosen here, so
#: the rates below are rates under the family the committed calibration used.
#: PLACED there and placed here; that is why arm C sweeps the noise instead of
#: resting on this value.
ONSET_WIDTH_LOG_STEP = 0.35
NOISE_SD = 0.02

ALPHA = 0.05

#: Replicates per cell. 200 resolves a proportion to about +/-0.015, which
#: separates 0.05 from 0.15 and does not separate 0.05 from 0.08 -- so arm B
#: rests on a DIFFERENCE between two rates measured on the same cell rather
#: than on either one of them.
N_REPS = 200
N_UNITS = 16
N_CONTROLS = 19

#: Family-wise level for this file's own numeric checks, tighter than the
#: registry's alpha for the reason 6n records: these bounds are applied to
#: proportions in a REGENERATED artifact, and a bound that fails once in twenty
#: regenerations when nothing is wrong is one that gets re-run rather than read.
CHECK_FAMILY_ALPHA = 0.01

_SEED = 20260827


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _two_sided_z(alpha_family: float, n_cells: int) -> float:
    """Bonferroni z for a one-sided per-cell bound at a family-wise level."""
    target = 1.0 - alpha_family / max(int(n_cells), 1)
    lo, hi = 0.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Synthetic series with a planted answer
# ---------------------------------------------------------------------------

def _logistic(steps: np.ndarray, mid_step: float, rng, noise: float) -> np.ndarray:
    x = step_x(steps)
    v = 1.0 / (1.0 + np.exp(-(x - np.log10(mid_step + 1.0)) / ONSET_WIDTH_LOG_STEP))
    return v + noise * rng.standard_normal(x.size) if noise else v


def _onsets(kind: str, n: int, rng) -> np.ndarray:
    if kind == "anchor":                      # inside CLAIM-B's registered window
        return 10.0 ** rng.uniform(np.log10(600.0), np.log10(1800.0), n)
    if kind == "early":                       # "everything moves early in training"
        return 10.0 ** rng.uniform(np.log10(2.0), np.log10(2000.0), n)
    if kind == "late":                        # demonstrably past the anchor
        return 10.0 ** rng.uniform(np.log10(20000.0), np.log10(120000.0), n)
    return 10.0 ** rng.uniform(np.log10(2.0), np.log10(60000.0), n)


def series(kind: str, steps: np.ndarray, n_units: int, rng,
           noise: float) -> List[np.ndarray]:
    """
    `no-change` is the input this pass is about: per-checkpoint noise and no
    located change anywhere. Its correct verdict on an anchor arm is that there
    is nothing to anchor.
    """
    if kind == "no-change":
        return [noise * rng.standard_normal(steps.size) for _ in range(n_units)]
    return [_logistic(steps, m, rng, noise) for m in _onsets(kind, n_units, rng)]


def controls(family: str, steps: np.ndarray, n_units: int, n_controls: int,
             rng, noise: float) -> Tuple[dict, dict]:
    """
    `localized` is every control carrying a real change somewhere on the sweep.
    `mixed` puts a change-free series at every fourth control, which is what a
    real family of "other checkpoint-level metric series" looks like: not every
    metric on a sweep has a located change. `half-at-anchor` exists only for
    the RE-ANCHORS row: the reciprocal test needs controls that are CLOSE to
    the window for a late change to be demonstrably further away than.
    """
    c: Dict[str, List[np.ndarray]] = {}
    d: Dict[str, str] = {}
    for i in range(n_controls):
        if family == "mixed" and i % 4 == 0:
            kind = "no-change"
        elif family == "half-at-anchor" and i % 2 == 0:
            kind = "anchor"
        else:
            kind = "random"
        c[f"c{i}"] = series(kind, steps, n_units, rng, noise)
        d[f"c{i}"] = "rise"
    return c, d


# ---------------------------------------------------------------------------
# The anchor arm's arithmetic, scored WITHOUT the refusal
# ---------------------------------------------------------------------------
#
# Arm B has to measure what the arm did BEFORE this pass as well as after, so
# it needs the ranking without the refusal in front of it. That is a second
# implementation of the arm's arithmetic, which 6g records as a real risk -- so
# `module_agreement` pins it against `anchor_arm` itself on every input where
# the arm emits, rather than leaving the two to drift.

def _refuses(steps: np.ndarray, window=None) -> bool:
    """
    The condition `anchor_arm` refuses on, as this file models it: the
    change-free reference lands inside the registered window, so it attains
    the arm's maximum. It reads the GRID and the window and nothing else --
    no controls, no observation, no alpha -- which is what makes it decidable
    before a checkpoint is sampled.
    """
    w = CLAIM_B_ANCHOR_WINDOW if window is None else window
    return anchor_statistic(diffuse_reference_profile(steps), w) == 0.0


def _scored(steps: np.ndarray, under_test, ctrl: dict, dirs: dict,
            window=None) -> dict:
    w = CLAIM_B_ANCHOR_WINDOW if window is None else window
    obs = float(np.mean([anchor_statistic(change_profile(steps, v, "rise"), w)
                         for v in under_test]))
    cs = np.array([
        float(np.mean([anchor_statistic(change_profile(steps, v, dirs[name]), w)
                       for v in ctrl[name]]))
        for name in sorted(ctrl)], dtype=np.float64)
    ref = diffuse_reference_profile(steps)
    ref_stat = anchor_statistic(ref, w)
    return {
        "observed": obs,
        "p_value": float((np.sum(cs >= obs) + 1) / (cs.size + 1)),
        "p_reference": float((np.sum(cs >= ref_stat) + 1) / (cs.size + 1)),
        "reference_statistic": float(ref_stat),
    }


# ---------------------------------------------------------------------------
# A. the answer is known
# ---------------------------------------------------------------------------

def known_answer(rng, alpha: float) -> dict:
    """
    Both gates on inputs whose correct verdict is fixed a priori, and every
    verdict branch reached by the input built for it.

    CLAIM-B runs on `dense-154` rather than on the registered cheap sweep,
    because arm B is about to show that the cheap sweep's anchor arms cannot
    discriminate at all. Running the known-answer arm there would be asking a
    gate that refuses whether it returns the right verdict.
    """
    steps = np.asarray(DENSE_SWEEP, dtype=float)
    rows = []

    for label, expect, mutual_reversed, onset, family, seeds in (
            ("co-located at the anchor", "CO-LOCATES", False, "anchor",
             "localized", 5),
            # An H0 row for the anchor arms, so it asserts a RATE. Demanding
            # five INSUFFICIENTs out of five is an assertion about the draw,
            # and the first run of this arm duly produced a p at exactly the
            # floor on one seed of five. POPPER_PLAN.md 6m records the same
            # correction being needed for P-ST1's sharp input.
            ("co-located far from the anchor", "not CO-LOCATES", False,
             "random", "localized", 40),
            # The input built FOR the falsification branch, and it takes both
            # halves: the pairing anti-aligned AND both series demonstrably
            # further from the window than every control. Half the controls are
            # planted at the anchor, because a reciprocal test against controls
            # that are themselves far from the window has nothing to separate
            # from -- and a family planted ENTIRELY at the anchor is degenerate,
            # since `anchor_statistic` saturates at zero inside the window and
            # every such control returns the identical number.
            # 20 seeds and not 5, because this row's rate is the interesting
            # number rather than an incidental: see `re_anchors_margin` below.
            ("anti-aligned and demonstrably late", "RE-ANCHORS", True, "late",
             "half-at-anchor", 20),
    ):
        verdicts, ps, recips = [], [], []
        for _ in range(seeds):
            m = np.sort(_onsets(onset, N_UNITS, rng))
            energy = [_logistic(steps, x, rng, 0.0) for x in m]
            other = m[::-1] if mutual_reversed else m
            fiedler = [-_logistic(steps, x, rng, 0.0) for x in other]
            ctrl, dirs = controls(family, steps, N_UNITS, N_CONTROLS, rng, 0.0)
            res = p_value_claim_b(steps, energy, fiedler, ctrl, dirs,
                                  control_family=CLAIM_B_ANCHOR_CONTROL_FAMILY,
                                  alpha=alpha)
            verdicts.append(res["verdict"])
            ps.append(res["p_value"])
            recips.append(res["p_reciprocal"])
        hit = ([v != "CO-LOCATES" for v in verdicts]
               if expect == "not CO-LOCATES"
               else [v == expect for v in verdicts])
        rate = sum(hit) / len(hit)
        rows.append({
            "entry": "CLAIM-B", "input": label, "expected": expect,
            "verdicts": verdicts, "p_values": ps, "p_reciprocals": recips,
            "expected_rate": rate,
            # RE-ANCHORS is asserted to FIRE rather than to fire every time,
            # and the reason is arithmetic rather than sampling: both anchor
            # arms' reciprocal p is floored at 1/(n_controls + 1), which is
            # exactly alpha at nineteen controls, so the branch needs the
            # observed series to sit further from the window than EVERY control
            # in both arms at once. One control further out and the branch is
            # gone. `re_anchors_margin` records that.
            "as_expected": bool(
                rate > 0.0 if expect == "RE-ANCHORS"
                # Two verdict branches stay reachable under H0 at alpha each,
                # so an H0 row is allowed the two-sided alpha it is entitled
                # to. Derived from alpha and the seed count, not placed.
                else rate >= 1.0 - 2 * alpha
                - 3 * np.sqrt(2 * alpha * (1 - 2 * alpha) / len(hit))
                if expect == "not CO-LOCATES" else rate == 1.0),
        })

    # P-I1's rows. The independent-onset row asserts a RATE and not an
    # identity: under H0 the reciprocal tail fires at alpha by design, so a
    # row demanding INSUFFICIENT five times out of five is an assertion about
    # the draw rather than about the gate -- which is exactly what happened the
    # first time this arm was run, and POPPER_PLAN.md 6m records the same
    # correction being needed for P-ST1's sharp input.
    for label, expect, kind, seeds in (
            ("both curves rise together per head", "CO-LOCATES", "paired", 5),
            ("rise order reversed across heads", "RE-ANCHORS", "reversed", 5),
            ("independent onsets", "INSUFFICIENT", "independent", 40),
    ):
        verdicts = []
        for _ in range(seeds):
            m = np.sort(_onsets("random", N_UNITS, rng))
            relay = [_logistic(steps, x, rng, NOISE_SD) for x in m]
            if kind == "paired":
                b_onsets = m
            elif kind == "reversed":
                b_onsets = m[::-1]
            else:
                b_onsets = _onsets("random", N_UNITS, rng)
            behav = [_logistic(steps, x, rng, NOISE_SD) for x in b_onsets]
            verdicts.append(p_value_p_i1(steps, relay, behav,
                                         alpha=alpha)["verdict"])
        rate = sum(v == expect for v in verdicts) / len(verdicts)
        rows.append({
            "entry": "P-I1", "input": label, "expected": expect,
            "verdicts": verdicts, "p_values": None, "p_reciprocals": None,
            "expected_rate": rate,
            # Two verdict branches remain reachable under H0 at alpha each, so
            # the H0 row is allowed the two-sided alpha it is entitled to
            # rather than being asserted at 1.0. Derived from alpha, not placed.
            "as_expected": bool(rate == 1.0 if kind != "independent"
                                else rate >= 1.0 - 2 * alpha
                                - 3 * np.sqrt(2 * alpha * (1 - 2 * alpha)
                                              / len(verdicts))),
        })

    return {
        "_what": ("both gates on inputs whose correct verdict is fixed a "
                  "priori, on a grid where the anchor arm can discriminate."),
        "_why_not_the_registered_sweep": (
            "arm B shows the registered cheap sweep's anchor arms cannot tell "
            "an anchored change from no change at all, and the construction "
            "now refuses there. Asking a gate that refuses whether it returns "
            "the right verdict is not a check."),
        "grid": "dense-154",
        "alpha": alpha,
        "n_seeds_per_row": "5, except P-I1's H0 row at 40 -- see its note",
        "rows": rows,
        "every_row_as_expected": bool(all(r["as_expected"] for r in rows)),
        "branches_reached": sorted({v for r in rows for v in r["verdicts"]}),
        "re_anchors_margin": {
            "_what": ("CLAIM-B's FALSIFICATION branch fires on the input built "
                      "for it, and it fires with no margin."),
            "reciprocal_floor_at_n_controls": 1.0 / (N_CONTROLS + 1),
            "alpha": alpha,
            "floor_equals_alpha": bool(
                abs(1.0 / (N_CONTROLS + 1) - alpha) < 1e-12),
            "measured_rate_on_the_input_built_for_it": next(
                r["expected_rate"] for r in rows
                if r["entry"] == "CLAIM-B" and r["expected"] == "RE-ANCHORS"),
            "_reading": (
                "at nineteen controls the anchor arms' reciprocal floor is "
                "exactly alpha, so RE-ANCHORS requires the observed series to "
                "rank strictly worst of twenty in BOTH anchor arms at once. "
                "The nineteen-control requirement EVALUABILITY.md already "
                "records as the minimum for CO-LOCATES is therefore also the "
                "exact minimum for the falsification branch, with no margin in "
                "either direction -- a second reading of the same number, and "
                "one nothing in the registry had made."),
        },
    }


# ---------------------------------------------------------------------------
# B. the finding
# ---------------------------------------------------------------------------

def anchor_discrimination(rng, alpha: float, reps: Optional[int] = None) -> dict:
    """
    Does the anchor arm's rejection rate depend on whether there is a change?

    Three inputs per cell: a change planted INSIDE the registered window (the
    arm must reject), a change somewhere else (H0, the arm must hold at alpha),
    and NO located change at all (there is nothing to anchor, so a rejection is
    unsupported whatever it says). The finding is the DIFFERENCE between the
    first and the third, which is the arm's discriminating power, and it is
    measured on the same cell so the two rates share their control draws.
    """
    n = N_REPS if reps is None else int(reps)     # resolved here, not in the
    rows = []                                     # signature; see 6h/6m/6n
    for grid_name, grid in GRIDS.items():
        steps = np.asarray(grid, dtype=float)
        ref = grid_reference_report(steps)
        for family in ("localized", "mixed"):
            rates, refs = {}, []
            for kind in ("anchor", "random", "no-change"):
                hits = 0
                for _ in range(n):
                    ut = series(kind, steps, N_UNITS, rng, NOISE_SD)
                    ctrl, dirs = controls(family, steps, N_UNITS, N_CONTROLS,
                                          rng, NOISE_SD)
                    sc = _scored(steps, ut, ctrl, dirs)
                    hits += int(sc["p_value"] <= alpha)
                    refs.append(sc["p_reference"])
                rates[kind] = hits / n
            rows.append({
                "grid": grid_name,
                "n_checkpoints": int(steps.size),
                "control_family": family,
                "uniform_profile_centroid_step":
                    ref["uniform_profile_centroid_step"],
                "reference_inside_window": bool(
                    CLAIM_B_ANCHOR_WINDOW[0]
                    <= ref["uniform_profile_centroid_step"]
                    <= CLAIM_B_ANCHOR_WINDOW[1]),
                "reject_planted_at_anchor": rates["anchor"],
                "reject_h0_change_elsewhere": rates["random"],
                "reject_no_located_change": rates["no-change"],
                "discrimination": rates["anchor"] - rates["no-change"],
                "mean_p_of_the_change_free_reference": float(np.mean(refs)),
                "refusal_fires": bool(_refuses(steps)),
            })

    registered = [r for r in rows
                  if r["grid"] == "cheap-25 (registered)"
                  and r["control_family"] == "localized"][0]
    return {
        "_what": ("the anchor arm's rejection rate on a change planted at the "
                  "anchor, on a change elsewhere, and on NO located change -- "
                  "scored WITHOUT the refusal this pass added, which is what "
                  "the arm did before it."),
        "_the_finding": (
            "on the sweep the registry names as CLAIM-B's instrument, with "
            "controls that all carry a real change, the arm rejects on a "
            "change-free input at the same rate as on a perfectly anchored "
            "one. Its discriminating power there is the difference between "
            "those two numbers."),
        "_why_the_difference_and_not_either_rate": (
            "either rate alone is a proportion over a few hundred draws. The "
            "difference is measured on the same cell against the same control "
            "draws, and a discriminating arm cannot have it near zero however "
            "the sampling falls."),
        "alpha": alpha,
        "n_reps_per_cell": n,
        "n_units": N_UNITS,
        "n_controls": N_CONTROLS,
        "noise_sd": NOISE_SD,
        "rows": rows,
        "registered_sweep_discrimination": registered["discrimination"],
        "registered_sweep_rejects_a_change_free_input":
            registered["reject_no_located_change"],
        "discrimination_recovers_off_the_registered_grid": bool(
            max(r["discrimination"] for r in rows
                if not r["reference_inside_window"]) > 0.5),
    }


# ---------------------------------------------------------------------------
# C. the mechanism
# ---------------------------------------------------------------------------

def grid_pull(rng, reps: int = 300) -> dict:
    """
    Why it happens, as a closed form checked against the measurement.

    A centroid is a weighted mean of interval midpoints, so a profile is a
    mixture of the series' real location and the grid's own midpoint, weighted
    by the share of change mass that noise put there. Rectified Gaussian
    increments contribute `n_intervals * sigma * sqrt(2) * phi(0)` in
    expectation against the series' own range, which gives the share in closed
    form -- so this arm predicts the centroid before measuring it, and reports
    both.
    """
    lo_x, hi_x = float(step_x([CLAIM_B_ANCHOR_WINDOW[0]])[0]), \
        float(step_x([CLAIM_B_ANCHOR_WINDOW[1]])[0])
    rows = []
    for grid_name, grid in GRIDS.items():
        steps = np.asarray(grid, dtype=float)
        n_int = steps.size - 1
        mid = float(interval_midpoints(steps).mean())
        for sigma in (0.0, 0.005, 0.02, 0.05):
            cs, ests = [], []
            for _ in range(reps):
                m = float(_onsets("anchor", 1, rng)[0])
                v = _logistic(steps, m, rng, sigma)
                p = change_profile(steps, v, "rise")
                cs.append(p["centroid_log_step"])
                ests.append(p["noise_mass_share_estimate"])
            # E[max(N(0, sigma*sqrt2), 0)] = sigma*sqrt(2)/sqrt(2*pi) per
            # interval, against a logistic whose total rise is ~1.
            noise_mass = n_int * sigma * sqrt(2.0) / sqrt(2.0 * np.pi)
            share = noise_mass / (1.0 + noise_mass)
            true_loc = float(np.mean([np.log10(x + 1.0)
                                      for x in _onsets("anchor", 400, rng)]))
            rows.append({
                "grid": grid_name,
                "n_intervals": int(n_int),
                "grid_midpoint_log_step": mid,
                "noise_sd": sigma,
                "predicted_noise_mass_share": float(share),
                "measured_noise_mass_share_estimate": float(np.mean(ests)),
                "predicted_centroid_log_step":
                    float(share * mid + (1.0 - share) * true_loc),
                "measured_centroid_log_step": float(np.mean(cs)),
                "measured_centroid_inside_window":
                    float(np.mean((np.asarray(cs) >= lo_x)
                                  & (np.asarray(cs) <= hi_x))),
            })
    err = [abs(r["predicted_centroid_log_step"] - r["measured_centroid_log_step"])
           for r in rows]
    return {
        "_what": ("the centroid of a change planted INSIDE the window, against "
                  "the grid it was measured on and the per-checkpoint noise, "
                  "with a closed-form prediction beside every measurement."),
        "_why_a_closed_form": (
            "a rate that is only measured can be a coincidence of one family. "
            "A prediction made from the interval count and the noise alone, "
            "agreeing across three grids and four noise levels, is the "
            "mechanism -- and it is what says a DENSER sweep is worse rather "
            "than better, which is the opposite of what the module's power "
            "argument for change-mass weighting would suggest."),
        "_what_the_estimator_diagnostic_does_and_does_not_do": (
            "`noise_mass_share_estimate` is the reverse-direction mass over "
            "the forward mass, which needs no model of the noise and nothing "
            "placed. It tracks the predicted share where most intervals carry "
            "no signal and reads LOW on a coarse grid, where the signal's own "
            "increments suppress the reverse mass. Reported, never used to "
            "correct a centroid."),
        "n_reps_per_cell": reps,
        "anchor_window_steps": list(CLAIM_B_ANCHOR_WINDOW),
        "rows": rows,
        "max_absolute_centroid_prediction_error": float(max(err)),
        "closed_form_tracks_the_measurement": bool(max(err) <= 0.15),
    }


# ---------------------------------------------------------------------------
# C2. what the control family decides, and what it does not
# ---------------------------------------------------------------------------

#: How many of the nineteen controls carry no located change, swept. It is the
#: axis that decided the refusal's CONDITION, so the sweep is in the record.
CHANGE_FREE_CONTROL_COUNTS: Tuple[int, ...] = (0, 2, 5, 10, 15, 19)


def change_free_rate_vs_family(rng, alpha: float, reps: int = 200) -> dict:
    """
    The arm's rejection rate on a series with NO located change, against how
    many of its controls are themselves change-free.

    This arm exists because it corrected this pass's own first attempt. The
    refusal was first written to fire when the change-free reference OUTRANKED
    the controls -- which looked right and is not, because the reference is
    noiseless and a realised change-free series is not, so the reference
    outranks even the change-free members of a family and its rank pegs at the
    floor whatever it is handed. Measuring the family axis is what showed it:
    the rank is constant across this sweep and the rejection rate runs from
    1.000 to nominal, so the two cannot be the same condition.

    What the rate follows instead is exact and needs no fitting. A change-free
    series beats every control that has a located change, because on a grid
    whose midpoint is inside the window a located change is usually further
    from the window than the midpoint is. So its only real competition is the
    change-free controls, and it ranks first among k + 1 of them: 1/(k+1).
    """
    steps = np.asarray(CHEAP_SWEEP, dtype=float)
    rows = []
    for k in CHANGE_FREE_CONTROL_COUNTS:
        rates, refs = {}, []
        for kind in ("no-change", "anchor"):
            hits = 0
            for _ in range(reps):
                ut = series(kind, steps, N_UNITS, rng, NOISE_SD)
                c: Dict[str, List[np.ndarray]] = {}
                d: Dict[str, str] = {}
                for i in range(N_CONTROLS):
                    c[f"c{i}"] = series("no-change" if i < k else "random",
                                        steps, N_UNITS, rng, NOISE_SD)
                    d[f"c{i}"] = "rise"
                sc = _scored(steps, ut, c, d)
                hits += int(sc["p_value"] <= alpha)
                refs.append(sc["p_reference"])
            rates[kind] = hits / reps
        rows.append({
            "n_change_free_controls": int(k),
            "closed_form_1_over_k_plus_1": 1.0 / (k + 1.0),
            "reject_on_a_change_free_input": rates["no-change"],
            "reject_on_an_anchored_input": rates["anchor"],
            "mean_p_of_the_change_free_reference": float(np.mean(refs)),
        })
    err = [abs(r["reject_on_a_change_free_input"]
               - r["closed_form_1_over_k_plus_1"]) for r in rows]
    refs = [r["mean_p_of_the_change_free_reference"] for r in rows]
    return {
        "_what": ("on the REGISTERED cheap sweep: the anchor arm's rejection "
                  "rate on a series with no located change, against how many "
                  "of its nineteen controls are themselves change-free."),
        "_why_it_is_here": (
            "it corrected this pass's own first attempt at the refusal. A "
            "condition built on the change-free reference's RANK cannot see "
            "this axis -- the rank is flat across it, in the last column -- "
            "while the thing that matters runs from 1.000 to nominal. So the "
            "rank is reported and what is refused is the CEILING: the "
            "reference landing inside the window, which is a fact about the "
            "grid and decidable before a checkpoint is sampled."),
        "_and_what_it_costs_to_fix_by_the_family_instead": (
            "reaching nominal needs essentially every control to be a series "
            "with no located change, which is not a control family anyone "
            "would measure -- and the arm still has to rank the observation "
            "against it. That is why this is a requirement on WHICH "
            "CHECKPOINTS the pilot samples rather than on which metrics."),
        "alpha": alpha,
        "n_reps_per_cell": reps,
        "grid": "cheap-25 (registered)",
        "rows": rows,
        "max_absolute_error_against_the_closed_form": float(max(err)),
        "closed_form_holds": bool(max(err) <= 0.05),
        "reference_rank_is_flat_across_the_family_axis": bool(
            max(refs) - min(refs) <= 0.05),
    }


# ---------------------------------------------------------------------------
# D. what the refusal costs
# ---------------------------------------------------------------------------

def refusal_cost(rng, alpha: float, reps: int = 120) -> dict:
    """
    Re-score every cell with the refusal in front of it and without.

    `POPPER_PLAN.md` 6l's refusal removed no reachable verdict and was measured
    to cost zero power. 6m's could not cost one by construction. THIS ONE
    COSTS VERDICTS, so the counterfactual is run rather than an argument made:
    for each cell, P(CO-LOCATES) on a perfectly anchored input with and without
    the refusal, beside the same two numbers on a change-free input.
    """
    rows = []
    for grid_name, grid in GRIDS.items():
        steps = np.asarray(grid, dtype=float)
        for family in ("localized", "mixed"):
            cell = {"anchor": [0, 0], "no-change": [0, 0]}   # [with, without]
            refuses = _refuses(steps)
            for kind in ("anchor", "no-change"):
                for _ in range(reps):
                    ut = series(kind, steps, N_UNITS, rng, NOISE_SD)
                    ctrl, dirs = controls(family, steps, N_UNITS, N_CONTROLS,
                                          rng, NOISE_SD)
                    sc = _scored(steps, ut, ctrl, dirs)
                    cell[kind][1] += int(sc["p_value"] <= alpha)
                    if not refuses:
                        cell[kind][0] += int(sc["p_value"] <= alpha)
            rows.append({
                "grid": grid_name,
                "control_family": family,
                "anchored_input_verdict_rate_without_refusal":
                    cell["anchor"][1] / reps,
                "anchored_input_verdict_rate_with_refusal":
                    cell["anchor"][0] / reps,
                "change_free_input_verdict_rate_without_refusal":
                    cell["no-change"][1] / reps,
                "change_free_input_verdict_rate_with_refusal":
                    cell["no-change"][0] / reps,
                "refuses": bool(refuses),
            })
    cost = [r["anchored_input_verdict_rate_without_refusal"]
            - r["anchored_input_verdict_rate_with_refusal"] for r in rows]
    return {
        "_what": ("what the refusal turns away, re-scored on the inputs it "
                  "turns away -- 6l's discipline, asked of a refusal that does "
                  "NOT come out at zero."),
        "_the_honest_reading": (
            "the refusal removes verdicts on inputs whose change really is at "
            "the anchor. It removes them where a change-free input would have "
            "produced the same verdict, so what it costs is a verdict the "
            "design cannot support rather than one it could not reach -- but "
            "the cost is real and it is in this table rather than in a "
            "sentence. It is also why the refusal reads the CONTROLS: on the "
            "same grid, a control family that contains change-free series "
            "ranks the reference in the middle and nothing is refused."),
        "_and_what_it_never_removes": (
            "a verdict on a grid whose change-free reference falls outside the "
            "window. Those rows show zero cost by construction -- the refusal "
            "does not fire there -- and they are the rows a pilot should be "
            "aiming at."),
        "alpha": alpha,
        "n_reps_per_cell": reps,
        "rows": rows,
        "max_verdict_cost": float(max(cost)),
        "costs_nothing_where_it_does_not_fire": bool(all(
            abs(r["anchored_input_verdict_rate_without_refusal"]
                - r["anchored_input_verdict_rate_with_refusal"]) < 1e-9
            for r in rows if not r["refuses"])),
        "costs_verdicts_somewhere": bool(max(cost) > 0.0),
    }


# ---------------------------------------------------------------------------
# E. why P-I1 is left alone
# ---------------------------------------------------------------------------

def p_i1_unaffected(rng, alpha: float, reps: Optional[int] = None) -> dict:
    """
    The mutual arm on the families that take the anchor arm to 1.000.

    P-I1 is the mutual arm alone. Its null permutes the PAIRING, so both
    series keep their real per-head locations in every draw and the pull toward
    the grid's midpoint is present on both sides of every pairing. It should
    cancel. That is an argument, so it is measured -- on the REGISTERED cheap
    sweep, the grid where the anchor arm fails hardest, because measuring it
    anywhere else would be choosing the easy case.
    """
    n = N_REPS if reps is None else int(reps)
    steps = np.asarray(CHEAP_SWEEP, dtype=float)
    rows = []
    for label, a_kind, b_kind in (
            ("both series change nowhere", "no-change", "no-change"),
            ("one series changes nowhere", "no-change", "random"),
            ("both change, independent onsets", "random", "random"),
            ("both change early (common trend)", "early", "early"),
    ):
        hits = refused = 0
        for _ in range(n):
            a = series(a_kind, steps, N_UNITS, rng, NOISE_SD)
            b = series(b_kind, steps, N_UNITS, rng, NOISE_SD)
            try:
                r = paired_colocation_arm(steps, a, "rise", b, "rise",
                                          alpha=alpha, unit_name="head",
                                          arm_name="mutual")
            except ColocationRefused:
                refused += 1
                continue
            hits += int(r["p_value"] <= alpha)
        emitted = n - refused
        rows.append({
            "h0_family": label,
            "reject_conditional_on_emission":
                (hits / emitted) if emitted else None,
            "refused_rate": refused / n,
            "n_emitted": emitted,
        })
    se = float(np.sqrt(alpha * (1 - alpha) / n))
    z = _two_sided_z(CHECK_FAMILY_ALPHA, len(rows))
    rates = [r["reject_conditional_on_emission"] for r in rows
             if r["reject_conditional_on_emission"] is not None]
    return {
        "_what": ("the mutual arm -- which is the whole of P-I1's gate -- on "
                  "the H0 families that take CLAIM-B's anchor arm to 1.000, "
                  "measured on the REGISTERED cheap sweep where the anchor arm "
                  "fails hardest."),
        "_why_it_is_here": (
            "P-I1's null was NOT changed and CLAIM-B's anchor arms were. "
            "Leaving an entry alone is a decision, and this is the measurement "
            "behind it -- the precedent P6-R4 set on 2026-08-26. Without it "
            "the difference between the two entries would rest on an argument "
            "about their statistics rather than on a number."),
        "_the_mechanism": (
            "the pairing null keeps both series' real locations in every draw, "
            "so a pull that moves every location the same way is present in "
            "the null exactly as in the observation and cancels. The anchor "
            "arm's reference is a fixed window, and a fixed point does not "
            "move with the grid."),
        "alpha": alpha,
        "n_reps_per_cell": n,
        "grid": "cheap-25 (registered)",
        "standard_error_at_alpha": se,
        "bound_in_standard_errors": z,
        "bound": float(alpha + z * se),
        "rows": rows,
        "range": [min(rates), max(rates)] if rates else None,
        "holds": bool(max(rates) <= alpha + z * se) if rates else False,
    }


def module_agreement(rng, alpha: float, reps: int = 40) -> dict:
    """
    Arm B scores the anchor arm's arithmetic itself, to reach the rate the
    module now refuses. That is a second implementation, which 6g records as a
    real risk on CLAIM-C's fast path -- so it is pinned against `anchor_arm`
    cell by cell wherever the module emits.
    """
    worst, checked = 0.0, 0
    for grid in GRIDS.values():
        steps = np.asarray(grid, dtype=float)
        for family in ("localized", "mixed"):
            for _ in range(reps):
                ut = series("random", steps, N_UNITS, rng, NOISE_SD)
                ctrl, dirs = controls(family, steps, N_UNITS, N_CONTROLS,
                                      rng, NOISE_SD)
                sc = _scored(steps, ut, ctrl, dirs)
                try:
                    res = anchor_arm(steps, ut, "rise", CLAIM_B_ANCHOR_WINDOW,
                                     ctrl, dirs, alpha=alpha,
                                     unit_name="layer", arm_name="a")
                except ColocationRefused:
                    continue
                worst = max(worst, abs(res["p_value"] - sc["p_value"]),
                            abs(res["diffuse_reference"]["p_value"]
                                - sc["p_reference"]))
                checked += 1
    return {
        "_what": ("this file's own scoring of the anchor arm, checked against "
                  "`core.changepoint_colocation.anchor_arm` wherever the "
                  "module emits."),
        "n_compared": checked,
        "max_absolute_difference": float(worst),
        "agrees": bool(checked > 0 and worst == 0.0),
    }


# ---------------------------------------------------------------------------
# Assembling
# ---------------------------------------------------------------------------

def build_record(seed: int = _SEED) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    rec = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "_what": ("CLAIM-B and P-I1 run on inputs whose correct answer is "
                  "known a priori, and the measurement that changed CLAIM-B's "
                  "anchor arms."),
        "_why": ("EVALUABILITY.md's queue owes every converted row a run on an "
                 "input whose verdict is fixed in advance. These two share one "
                 "estimator, so one dry run covers both."),
        "_not": ("not evidence about Pythia and not an adjudication. The "
                 "series are synthetic; what is being checked is the gate."),
        "generated_by": "python3 -m tools.dry_run_claim_b_p_i1 --write",
        "construction_file": str(CONSTRUCTION_PATH.relative_to(ROOT)),
        "construction_sha256": _sha256(CONSTRUCTION_PATH),
        "formation_gate_file": str(FORMATION_GATE_PATH.relative_to(ROOT)),
        "formation_gate_sha256": _sha256(FORMATION_GATE_PATH),
        "alpha": ALPHA,
        "anchor_window_steps": list(CLAIM_B_ANCHOR_WINDOW),
        "grids": {name: {"n_checkpoints": len(g),
                         "uniform_profile_centroid_step":
                             grid_reference_report(np.asarray(g, dtype=float))[
                                 "uniform_profile_centroid_step"]}
                  for name, g in GRIDS.items()},
        "synthetic_family": {
            "curve": "logistic in log10(step+1)",
            "onset_width_log_step": ONSET_WIDTH_LOG_STEP,
            "noise_sd": NOISE_SD,
            "_placed": ("both taken from tools/calibrate_changepoint_"
                        "colocation.py rather than re-chosen, so these rates "
                        "are rates under the family the committed calibration "
                        "used. Arm C sweeps the noise for that reason."),
        },
        "seed": int(seed),
        "known_answer": known_answer(rng, ALPHA),
        "anchor_discrimination": anchor_discrimination(rng, ALPHA),
        "grid_pull": grid_pull(rng),
        "change_free_rate_vs_family": change_free_rate_vs_family(rng, ALPHA),
        "refusal_cost": refusal_cost(rng, ALPHA),
        "p_i1_unaffected": p_i1_unaffected(rng, ALPHA),
        "module_agreement": module_agreement(rng, ALPHA),
    }
    # The generation cost lives in the ARTIFACT and not only in a docstring.
    # POPPER_PLAN.md 6n's tool understated its own cost in three places and the
    # figure had to be chased across two files and a --help string; a number
    # the tool measures on every write cannot drift from the tool.
    rec["elapsed_seconds"] = round(time.time() - t0, 1)
    return rec


def check_record(path: Path = RECORD_PATH) -> List[str]:
    """
    Is the committed record still about the files on disk, and does it still
    support the change it was the evidence for?

    Four things can fail and each of them should: the record can describe a
    module that has moved; the finding can stop being there, in which case the
    refusal has nothing behind it; the refusal can stop costing what this
    record says it costs; and `P-I1`'s arm can stop holding, in which case
    leaving it alone is no longer a decision with a measurement behind it.
    """
    problems: List[str] = []
    if not path.exists():
        return [f"{path} is missing; regenerate with "
                f"`python3 -m tools.dry_run_claim_b_p_i1 --write`"]
    try:
        rec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path} is not valid JSON: {exc}"]

    if rec.get("schema_version") != RECORD_SCHEMA_VERSION:
        problems.append(
            f"schema_version {rec.get('schema_version')} != "
            f"{RECORD_SCHEMA_VERSION}; regenerate with --write")
    for key, described in (("construction", CONSTRUCTION_PATH),
                           ("formation_gate", FORMATION_GATE_PATH)):
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
    for branch in ("CO-LOCATES", "RE-ANCHORS", "INSUFFICIENT"):
        if branch not in (ka.get("branches_reached") or []):
            problems.append(
                f"no input in the record reaches the {branch} branch. A "
                f"verdict nothing can trigger is a defect (POPPER_PLAN.md 6h)")

    margin = ka.get("re_anchors_margin") or {}
    if not (margin.get("measured_rate_on_the_input_built_for_it") or 0.0) > 0.0:
        problems.append(
            "CLAIM-B's RE-ANCHORS branch did not fire on the input built for "
            "it. It is this claim's registered falsifier, and a verdict "
            "nothing can trigger is a defect (POPPER_PLAN.md 6h, 6i)")

    ad = rec.get("anchor_discrimination", {})
    if not ad.get("rows"):
        problems.append("anchor_discrimination has no rows")
    else:
        # `or 1.0` here would be a defect and was one: the value this guards
        # is a discrimination that SHOULD be 0.0, and 0.0 is falsy, so the
        # fallback fired on the healthy case and reported the finding missing.
        # Caught by running --check on the artifact rather than by a test.
        disc = ad.get("registered_sweep_discrimination")
        if disc is None or disc > 0.10:
            problems.append(
                f"the anchor arm's discriminating power on the registered "
                f"sweep is {disc}, not near zero. The refusal added on "
                f"2026-08-27 rests on that number; an artifact that no longer "
                f"shows it does not support the change it is the evidence for")
        free = ad.get("registered_sweep_rejects_a_change_free_input")
        if free is None or free < 0.5:
            problems.append(
                "a change-free input no longer clears alpha on the registered "
                "sweep; the defect this record documents is not in it")
        if not ad.get("discrimination_recovers_off_the_registered_grid"):
            problems.append(
                "no grid in the record recovers discriminating power, so this "
                "record says the arm is useless everywhere rather than that it "
                "is useless where its reference sits in the window -- which is "
                "a different and larger claim than the one it is used for")

    gp = rec.get("grid_pull", {})
    if not gp.get("rows"):
        problems.append("grid_pull has no rows")
    elif not gp.get("closed_form_tracks_the_measurement"):
        problems.append(
            f"the closed-form centroid prediction is off by "
            f"{gp.get('max_absolute_centroid_prediction_error')} in log10-step; "
            f"the mechanism this record claims is not the one it measured")

    cf = rec.get("change_free_rate_vs_family", {})
    if not cf.get("rows"):
        problems.append("change_free_rate_vs_family has no rows")
    else:
        if not cf.get("closed_form_holds"):
            problems.append(
                f"the change-free rejection rate no longer follows 1/(k+1) "
                f"(worst error {cf.get('max_absolute_error_against_the_closed_form')}); "
                f"the arithmetic this record uses to explain the finding is "
                f"not the arithmetic it measured")
        if not cf.get("reference_rank_is_flat_across_the_family_axis"):
            problems.append(
                "the change-free reference's rank now varies with the control "
                "family. That was the reason the refusal is not built on it, "
                "so the condition should be reconsidered rather than left")

    rc = rec.get("refusal_cost", {})
    if not rc.get("rows"):
        problems.append("refusal_cost has no rows")
    else:
        if not rc.get("costs_verdicts_somewhere"):
            problems.append(
                "the refusal is recorded as costing no verdict anywhere. That "
                "is a stronger claim than this pass makes and it would mean "
                "the counterfactual re-scoring reached nothing -- POPPER_PLAN "
                "6l's `costs_no_power is None, never True`, in its other form")
        if not rc.get("costs_nothing_where_it_does_not_fire"):
            problems.append(
                "the refusal removed a verdict in a cell where it never fired, "
                "which is arithmetically impossible and means the "
                "counterfactual is not measuring what it says")

    pi = rec.get("p_i1_unaffected", {})
    if not pi.get("rows"):
        problems.append("p_i1_unaffected has no rows")
    elif not pi.get("holds"):
        problems.append(
            f"the mutual arm's H0 rate reaches {pi.get('range')} against alpha "
            f"{pi.get('alpha')}. P-I1's null was left unchanged on the evidence "
            f"that it is unaffected; this record no longer says so")

    ma = rec.get("module_agreement", {})
    if not ma.get("agrees"):
        problems.append(
            f"this file's scoring of the anchor arm disagrees with the module "
            f"by {ma.get('max_absolute_difference')} over {ma.get('n_compared')} "
            f"comparisons; the rates above are then about a second "
            f"implementation rather than about the gate")
    return problems


def print_summary(rec: dict) -> None:
    print(f"construction:   {rec['construction_file']}  sha256 "
          f"{rec['construction_sha256'][:12]}")
    print(f"formation gate: {rec['formation_gate_file']}  sha256 "
          f"{rec['formation_gate_sha256'][:12]}")
    print(f"alpha {rec['alpha']}   anchor window {rec['anchor_window_steps']}   "
          f"generated in {rec.get('elapsed_seconds')}s")
    for name, g in rec["grids"].items():
        print(f"  grid {name:24s} n={g['n_checkpoints']:4d}  a series with no "
              f"located change lands at step {g['uniform_profile_centroid_step']:.0f}")

    ka = rec["known_answer"]
    print(f"\n=== A. the answer is known (grid {ka['grid']}) ===")
    for r in ka["rows"]:
        print(f"  {r['entry']:8s} {r['input']:36s} expect {r['expected']:15s} "
              f"as expected {r['as_expected']!s:>5}")
    print(f"  branches reached: {', '.join(ka['branches_reached'])}")

    ad = rec["anchor_discrimination"]
    print(f"\n=== B. the anchor arm's discriminating power "
          f"({ad['n_reps_per_cell']} reps, refusal NOT applied) ===")
    print(f"  {'grid':24s} {'controls':10s} {'anchored':>9} {'H0':>6} "
          f"{'no change':>10} {'discrim':>8} {'p_ref':>7}")
    for r in ad["rows"]:
        print(f"  {r['grid']:24s} {r['control_family']:10s} "
              f"{r['reject_planted_at_anchor']:>9.3f} "
              f"{r['reject_h0_change_elsewhere']:>6.3f} "
              f"{r['reject_no_located_change']:>10.3f} "
              f"{r['discrimination']:>8.3f} "
              f"{r['mean_p_of_the_change_free_reference']:>7.3f}")

    gp = rec["grid_pull"]
    print(f"\n=== C. the mechanism ({gp['n_reps_per_cell']} reps) ===")
    print(f"  {'grid':24s} {'sigma':>6} {'share pred':>11} {'share est':>10} "
          f"{'centroid pred':>14} {'measured':>9} {'in window':>10}")
    for r in gp["rows"]:
        print(f"  {r['grid']:24s} {r['noise_sd']:>6.3f} "
              f"{r['predicted_noise_mass_share']:>11.3f} "
              f"{r['measured_noise_mass_share_estimate']:>10.3f} "
              f"{r['predicted_centroid_log_step']:>14.3f} "
              f"{r['measured_centroid_log_step']:>9.3f} "
              f"{r['measured_centroid_inside_window']:>10.3f}")
    print(f"  closed form tracks the measurement: "
          f"{gp['closed_form_tracks_the_measurement']} "
          f"(worst {gp['max_absolute_centroid_prediction_error']:.3f})")

    cf = rec["change_free_rate_vs_family"]
    print(f"\n=== C2. what the control family decides "
          f"({cf['n_reps_per_cell']} reps, grid {cf['grid']}) ===")
    print(f"  {'k change-free':>14} {'1/(k+1)':>9} {'change-free in':>15} "
          f"{'anchored in':>12} {'p_ref':>7}")
    for r in cf["rows"]:
        print(f"  {r['n_change_free_controls']:>14d} "
              f"{r['closed_form_1_over_k_plus_1']:>9.3f} "
              f"{r['reject_on_a_change_free_input']:>15.3f} "
              f"{r['reject_on_an_anchored_input']:>12.3f} "
              f"{r['mean_p_of_the_change_free_reference']:>7.3f}")
    print(f"  closed form holds: {cf['closed_form_holds']}; the reference's "
          f"rank is flat across this axis: "
          f"{cf['reference_rank_is_flat_across_the_family_axis']}")

    rc = rec["refusal_cost"]
    print(f"\n=== D. what the refusal costs ({rc['n_reps_per_cell']} reps) ===")
    print(f"  {'grid':24s} {'controls':10s} {'anchored w/o':>13} "
          f"{'with':>6} {'change-free w/o':>16} {'with':>6}")
    for r in rc["rows"]:
        print(f"  {r['grid']:24s} {r['control_family']:10s} "
              f"{r['anchored_input_verdict_rate_without_refusal']:>13.3f} "
              f"{r['anchored_input_verdict_rate_with_refusal']:>6.3f} "
              f"{r['change_free_input_verdict_rate_without_refusal']:>16.3f} "
              f"{r['change_free_input_verdict_rate_with_refusal']:>6.3f}")
    print(f"  worst verdict cost {rc['max_verdict_cost']:.3f}; costs nothing "
          f"where it does not fire: "
          f"{rc['costs_nothing_where_it_does_not_fire']}")

    pi = rec["p_i1_unaffected"]
    print(f"\n=== E. P-I1, unchanged and why ({pi['n_reps_per_cell']} reps, "
          f"grid {pi['grid']}) ===")
    for r in pi["rows"]:
        print(f"  {r['h0_family']:36s} reject|emitted "
              f"{r['reject_conditional_on_emission']:.3f}  refused "
              f"{r['refused_rate']:.3f}")
    print(f"  holds against {pi['bound']:.4f} "
          f"({pi['bound_in_standard_errors']:.2f} SE): {pi['holds']}")

    ma = rec["module_agreement"]
    print(f"\n=== F. this file's scoring against the module ===")
    print(f"  {ma['n_compared']} comparisons, worst difference "
          f"{ma['max_absolute_difference']}, agrees: {ma['agrees']}")


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
