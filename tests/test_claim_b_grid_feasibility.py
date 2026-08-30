"""
tests/test_claim_b_grid_feasibility.py — the grids that can carry CLAIM-B's
anchor arms, and the arithmetic that decides which.

`tools/claim_b_grid_feasibility.py` enumerates Pythia checkpoint schedules
against `POPPER_PLAN.md` §6o's two grid failures and measures the anchor arm on
the grids it picks. The enumeration and the measurement together are minutes,
so the record is committed and this module pins it — the division every other
calibration under `claims/` already uses.

**Most of this module re-derives rather than compares.** A pinned number only
ever checked against itself is a number nobody has checked, and this record is
mostly closed forms, so most of it can be recomputed in milliseconds.
`TestTheArithmeticFromScratch` recomputes the change-free spread against a live
simulation, the rectified-normal covariance constant from its own integral, the
read map on a grid whose answer can be done by hand, and the fact the whole
design turns on: the sharp-change reading of a location is **not** a bound.

`TestTheFindingIsStillInTheRecord` is the one the section rests on: a grid this
arithmetic picks discriminates between an anchored change and no change at all,
and every schedule this repository contains discriminates at zero.

`TestTheShortlistIsRunnable` is the cheap one that would have hurt most to
miss: every step in every shortlisted grid has to be a checkpoint EleutherAI
actually published, or the answer is a schedule nobody can download.
"""

from __future__ import annotations

import json
import unittest
from math import erf, pi, sqrt

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pytest.ini [pytest] markers.
pytestmark = pytest.mark.pure

from core import changepoint_colocation as C
from core.checkpoint_frames import step_x
from tools.claim_b_grid_feasibility import (
    CHANGE_WIDTH_LOG_STEP,
    CONSTRUCTION_PATH,
    MAX_CEILING_RATE,
    MAX_FALSE_ANCHOR_FRACTION,
    MIN_READ_SPAN_IN_SD,
    MIN_RETAINED_FRACTION,
    OUT_PATH,
    PUBLISHED_LINEAR_STEPS,
    PUBLISHED_LOG_STEPS,
    SCHEMA_VERSION,
    _check_the_schedule_copies,
    _pilot_steps,
    _registry_schedules,
    _sha256,
    check_record,
)

WINDOW = C.CLAIM_B_ANCHOR_WINDOW


def _record() -> dict:
    return json.loads(OUT_PATH.read_text())


def _published() -> set:
    return set(PUBLISHED_LOG_STEPS) | set(PUBLISHED_LINEAR_STEPS)


class TestTheRecordIsPresentAndCurrent(unittest.TestCase):

    def test_it_exists_and_parses(self):
        self.assertTrue(OUT_PATH.exists(),
                        f"{OUT_PATH} is missing; run "
                        f"`python3 -m tools.claim_b_grid_feasibility --write`")
        self.assertEqual(_record()["schema_version"], SCHEMA_VERSION)

    def test_it_describes_the_construction_on_disk(self):
        # Every number here is a property of one module's arithmetic. Without
        # the hash the record could go on describing a file that no longer
        # exists in that form and nothing would notice.
        self.assertEqual(_record()["construction_sha256"],
                         _sha256(CONSTRUCTION_PATH))

    def test_check_record_is_clean(self):
        self.assertEqual(check_record(_record()), [])

    def test_it_adjudicates_nothing_and_says_so(self):
        rec = _record()
        self.assertIn("adjudicat", rec["_not"].lower())
        self.assertIn("chooses no grid", rec["_not"])

    def test_the_generation_cost_is_measured_rather_than_quoted(self):
        self.assertGreater(_record()["elapsed_seconds"], 0.0)


class TestTheArithmeticFromScratch(unittest.TestCase):
    """Re-derived here, not read back out of the record."""

    def test_the_change_free_spread_matches_a_live_simulation(self):
        # 400 draws on a 25-point grid is milliseconds and separates the closed
        # form from the one that drops the adjacent-covariance term, which is
        # off by about 75%.
        grid = np.asarray(
            [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000,
             8000, 13000, 23000, 33000, 43000, 63000, 83000, 103000, 123000,
             143000], dtype=float)
        rng = np.random.default_rng(11)
        cs = np.array([
            C.change_profile(grid, 0.02 * rng.standard_normal(grid.size),
                             "rise")["centroid_log_step"]
            for _ in range(400)])
        pred = C.change_free_centroid_sd(grid)
        self.assertAlmostEqual(pred / cs.std(), 1.0, delta=0.12)
        mids = C.interval_midpoints(grid)
        self.assertAlmostEqual(float(cs.mean()), float(mids.mean()), delta=0.05)

    def test_the_noise_scale_cancels_out_of_that_spread(self):
        # The claim that makes it grid arithmetic: the weights are normalised,
        # so sigma appears above and below and drops out. Asserted against the
        # estimator itself rather than against the formula.
        grid = np.asarray([0, 128, 512, 1000, 2000, 8000, 32000, 143000],
                          dtype=float)
        spreads = []
        for sigma in (0.002, 0.05):
            rng = np.random.default_rng(5)
            spreads.append(float(np.std([
                C.change_profile(grid, sigma * rng.standard_normal(grid.size),
                                 "rise")["centroid_log_step"]
                for _ in range(300)])))
        self.assertAlmostEqual(spreads[0] / spreads[1], 1.0, delta=0.05)

    def test_the_adjacent_covariance_constant_is_the_rectified_normal_one(self):
        # Cov(max(X,0), max(Y,0)) / (E max(X,0))^2 at correlation -1/2, from
        # E[X+ Y+] = (sqrt(1-r^2) + r (pi/2 + arcsin r)) / (2 pi). The -1/2 is
        # exact for adjacent first differences: they share a checkpoint.
        r = -0.5
        expected = (sqrt(1 - r * r) + r * (pi / 2 + np.arcsin(r))) / (2 * pi)
        expected = expected / (1.0 / (2 * pi)) - 1.0
        self.assertAlmostEqual(C._RECT_ADJACENT_COV_OVER_MEAN_SQ, expected,
                               places=12)
        self.assertAlmostEqual(C._RECT_VAR_OVER_MEAN_SQ, pi - 1.0, places=12)

    def test_the_sharp_change_reading_is_not_a_bound(self):
        # The defect the whole design turns on. On a grid whose next checkpoint
        # after the window is far away, a change of real width puts most of its
        # mass in that wide interval and reads well outside the window, while
        # the sharp reading -- all the mass in the interval containing the
        # change -- says the window is fully retained.
        grid = np.asarray([512, 1000, 2000, 17000, 32000, 47000, 62000, 77000,
                           92000, 107000], dtype=float)
        f = C.grid_feasibility(grid, WINDOW, noise_to_range=0.02,
                               change_width_log_step=CHANGE_WIDTH_LOG_STEP)
        self.assertGreater(f["sharp_change_retained_window_fraction"], 0.9)
        self.assertLess(f["retained_window_fraction"], 0.2)

    def test_a_width_and_a_noise_level_are_given_together_or_not_at_all(self):
        with self.assertRaises(C.ColocationRefused):
            C.grid_feasibility([0, 512, 1000, 2000], WINDOW, noise_to_range=0.02)
        with self.assertRaises(C.ColocationRefused):
            C.grid_feasibility([0, 512, 1000, 2000], WINDOW,
                               change_width_log_step=0.35)
        grid_only = C.grid_feasibility([0, 512, 1000, 2000], WINDOW)
        self.assertNotIn("retained_window_fraction", grid_only)
        self.assertIn("reference_outside_window", grid_only)

    def test_a_single_interval_swallowing_the_window_reads_one_place(self):
        # The grid the first enumeration returned before the span condition
        # existed: it cannot say where in the window anything happened, and it
        # calls a stretch above the window anchored too.
        wide = np.asarray([512, 3000, 8000, 20000, 60000, 143000], dtype=float)
        fine = np.asarray([0, 512, 1000] + list(range(2000, 30001, 1000)),
                          dtype=float)
        f_wide = C.grid_feasibility(wide, WINDOW, noise_to_range=0.02,
                                    change_width_log_step=CHANGE_WIDTH_LOG_STEP)
        f_fine = C.grid_feasibility(fine, WINDOW, noise_to_range=0.02,
                                    change_width_log_step=CHANGE_WIDTH_LOG_STEP)
        self.assertLess(f_wide["window_read_span_in_change_free_sd"],
                        f_fine["window_read_span_in_change_free_sd"])
        self.assertGreater(f_wide["sharp_change_false_anchor_fraction"], 0.2)

    def test_the_window_holds_exactly_one_published_checkpoint(self):
        # What fixes how much resolution is available at all: 512 and 2000 are
        # the edges and step 1000 is the only published checkpoint between them.
        inside = sorted(s for s in _published() if WINDOW[0] < s < WINDOW[1])
        self.assertEqual(inside, [1000])

    def test_the_noise_share_leans_the_safe_way(self):
        # E[(a+Z)+] - a is decreasing in a, so the closed form's noise mass
        # (the a = 0 case) is an upper bound on the PULL. Checked numerically
        # rather than taken from the docstring -- and it bounds only the pull,
        # which is why the width is still required.
        rng = np.random.default_rng(3)
        z = rng.standard_normal(200000)
        added = [float(np.mean(np.clip(a + z, 0.0, None)) - a)
                 for a in (0.0, 0.25, 1.0, 4.0)]
        self.assertTrue(all(x > y for x, y in zip(added[:-1], added[1:])),
                        f"not decreasing in the signal increment: {added}")
        self.assertAlmostEqual(added[0], 1.0 / sqrt(2 * pi), delta=0.01)

    def test_the_ceiling_rate_is_the_normal_tail_it_claims_to_be(self):
        grid = np.asarray([0, 512, 1000, 2000, 8000, 20000, 60000], dtype=float)
        f = C.grid_feasibility(grid, WINDOW)
        mid = f["uniform_profile_centroid_log_step"]
        sd = f["change_free_centroid_sd_log_step"]
        x_lo, x_hi = (float(step_x([WINDOW[0]])[0]), float(step_x([WINDOW[1]])[0]))
        expect = 0.5 * (erf((x_hi - mid) / sd / sqrt(2))
                        - erf((x_lo - mid) / sd / sqrt(2)))
        self.assertAlmostEqual(f["change_free_ceiling_rate"], expect, places=12)

    def test_retention_can_rise_with_noise_which_is_why_there_is_no_bisection(self):
        # The reason `_noise_budgets` is closed-form. A location sitting below
        # the window is pulled INTO it by a grid midpoint above, so retention is
        # not monotone in the noise and a bisection would have found only the
        # first crossing.
        grid = np.asarray([0, 512, 1000, 2000, 4000] + list(range(10000, 60001, 10000)),
                          dtype=float)
        curve = C.grid_feasibility(
            grid, WINDOW, noise_to_range=0.02,
            change_width_log_step=CHANGE_WIDTH_LOG_STEP)["retention_curve"]
        vals = [p["retained_window_fraction"] for p in curve]
        self.assertEqual(len(vals), len(C.RETENTION_CURVE_NOISE_LEVELS))
        # Not an assertion that it DOES rise on this grid -- only that nothing
        # in the module assumes it cannot.
        self.assertTrue(all(0.0 <= v <= 1.0 for v in vals))


class TestConditionOneIsTheArmsOwnRefusal(unittest.TestCase):
    """
    `grid_feasibility`'s `reference_outside_window` and `anchor_arm`'s refusal
    are the same condition computed twice, and `POPPER_PLAN.md` §6g records a
    second implementation of a gate's arithmetic as a real risk. Checked live
    on every grid the record names.
    """

    def _refuses(self, steps) -> bool:
        return C.anchor_statistic(C.diffuse_reference_profile(steps),
                                  WINDOW) == 0.0

    def test_on_every_grid_the_record_names(self):
        rec = _record()
        grids = [r["steps"] for r in rec["catalogue"]["rows"]]
        grids += [row["steps"] for row in rec["feasible_set"]["frontier"]]
        self.assertGreater(len(grids), 10)
        for grid in grids:
            s = np.asarray(grid, dtype=float)
            f = C.grid_feasibility(s, WINDOW)
            self.assertEqual(f["reference_outside_window"],
                             not self._refuses(s),
                             f"condition 1 disagrees with anchor_arm on "
                             f"{grid[:5]}")

    def test_the_registered_cheap_sweep_is_still_the_one_that_refuses(self):
        row = next(r for r in _record()["catalogue"]["rows"]
                   if r["grid"].startswith("cheap-25"))
        self.assertTrue(row["arm_refuses"])
        self.assertFalse(row["feasibility"]["reference_outside_window"])
        self.assertAlmostEqual(
            row["feasibility"]["uniform_profile_centroid_step"], 955.0, delta=1.0)

    def test_the_registry_pilot_schedule_fails_the_same_way(self):
        # New here, and it matters more than the registered cheap sweep does:
        # `PYTHIA_410M_PILOT_STEPS` is the schedule this repository would
        # actually run, and its change-free reference lands inside the window
        # too.
        row = next(r for r in _record()["catalogue"]["rows"]
                   if r["grid"].startswith("pythia-410m-pilot"))
        self.assertFalse(row["feasibility"]["reference_outside_window"])
        self.assertTrue(row["arm_refuses"])


class TestTheFindingIsStillInTheRecord(unittest.TestCase):

    def setUp(self):
        self.rec = _record()

    def test_no_grid_this_project_has_can_carry_the_arm(self):
        self.assertTrue(
            self.rec["catalogue"]["none_of_them_meets_the_hard_conditions"])

    def test_a_computed_grid_discriminates_where_the_project_ones_do_not(self):
        bm = self.rec["boundary_measurement"]
        self.assertTrue(
            bm["computed_grids_discriminate_and_project_ones_do_not"])
        computed = [r for r in bm["rows"] if r["kind"] == "computed"]
        project = [r for r in bm["rows"]
                   if r["kind"] == "project" and r["control_family"] == "localized"]
        self.assertGreaterEqual(min(r["discrimination"] for r in computed), 0.9)
        self.assertLessEqual(max(r["discrimination"] for r in project), 0.1)

    def test_a_computed_grid_rejects_nothing_on_a_change_free_input(self):
        bm = self.rec["boundary_measurement"]
        self.assertTrue(
            bm["computed_grids_reject_nothing_on_a_change_free_input"])

    def test_the_computed_grids_hold_the_h0_rate_too(self):
        # A grid that discriminates by rejecting on everything is not the
        # finding. The H0 row -- a real change somewhere other than the anchor
        # -- has to stay near alpha.
        for r in self.rec["boundary_measurement"]["rows"]:
            if r["kind"] != "computed":
                continue
            self.assertLessEqual(r["reject_h0_change_elsewhere"], 0.15,
                                 f"{r['grid']} / {r['control_family']}")

    def test_the_predicted_and_measured_columns_agree_in_direction(self):
        for r in self.rec["boundary_measurement"]["rows"]:
            if r["predicted_retained_window_fraction"] <= 0.0:
                self.assertLessEqual(r["reject_planted_at_anchor"], 0.1,
                                     f"{r['grid']} rejects an anchored change "
                                     f"on a grid predicted to retain none of "
                                     f"the window")

    def test_the_probe_grid_is_named_as_a_probe_rather_than_lumped_in(self):
        # §6o's `early-dense-73` DOES discriminate; it is not a schedule anyone
        # proposed running, and the first version of this section reported that
        # no existing grid discriminates, which is false of it.
        bm = self.rec["boundary_measurement"]
        self.assertIn("_and_what_the_probe_grid_says", bm)
        probe = [r for r in bm["rows"] if r["kind"] == "probe"]
        self.assertTrue(probe)
        self.assertGreater(max(r["discrimination"] for r in probe), 0.5)

    def test_the_record_says_which_part_of_6os_residual_is_the_grids(self):
        note = self.rec["boundary_measurement"]["_and_the_residual_6o_left"]
        self.assertIn("0.245", note)
        self.assertIn("mixed", note)

    def test_the_retention_curve_was_checked_against_a_simulation(self):
        rc = self.rec["retention_curve_check"]
        self.assertTrue(rc["curve_tracks_the_measurement"])
        self.assertGreaterEqual(len(rc["rows"]),
                                len(C.RETENTION_CURVE_NOISE_LEVELS))


class TestTheShortlistIsRunnable(unittest.TestCase):

    def setUp(self):
        self.short = _record()["feasible_set"]["shortlist"]
        self.picks = {k: v for k, v in self.short.items()
                      if not k.startswith("_") and isinstance(v, dict)}

    def test_there_is_a_shortlist_at_all(self):
        self.assertTrue(self.picks, "the shortlist is empty")

    def test_every_shortlisted_step_is_a_published_pythia_checkpoint(self):
        published = _published()
        for key, row in self.picks.items():
            missing = [s for s in row["steps"] if s not in published]
            self.assertEqual(missing, [],
                             f"{key} names steps EleutherAI never published: "
                             f"{missing}")

    def test_every_shortlisted_grid_meets_the_hard_conditions_when_recomputed(self):
        for key, row in self.picks.items():
            f = C.grid_feasibility(
                np.asarray(row["steps"], dtype=float), WINDOW,
                noise_to_range=self.short["_rule"]["at_noise_to_range"],
                change_width_log_step=CHANGE_WIDTH_LOG_STEP)
            self.assertTrue(f["reference_outside_window"], key)
            self.assertLess(f["false_anchor_fraction"],
                            MAX_FALSE_ANCHOR_FRACTION, key)
            self.assertLess(f["change_free_ceiling_rate"], MAX_CEILING_RATE, key)
            self.assertGreaterEqual(f["window_read_span_in_change_free_sd"],
                                    MIN_READ_SPAN_IN_SD, key)

    def test_the_retention_stored_with_each_pick_recomputes(self):
        # The number the author would choose on, re-derived rather than read
        # back: the WORST retained share over the noise range, not the value at
        # one level.
        cap = self.short["_rule"]["retention_required_over_noise_to_range_up_to"]
        for key, row in self.picks.items():
            f = C.grid_feasibility(
                np.asarray(row["steps"], dtype=float), WINDOW,
                noise_to_range=self.short["_rule"]["at_noise_to_range"],
                change_width_log_step=CHANGE_WIDTH_LOG_STEP)
            worst = min(p["retained_window_fraction"]
                        for p in f["retention_curve"]
                        if p["noise_to_range"] <= cap)
            self.assertAlmostEqual(worst,
                                   row["worst_retained_over_the_noise_range"],
                                   places=9, msg=key)

    def test_no_grid_reaches_the_reference_retention_and_the_record_says_so(self):
        # The finding, and the reason retention is maximised rather than
        # thresholded: applied as a bound it admitted nothing, and that is a
        # fact about Pythia's schedule rather than about the bound.
        self.assertTrue(self.short["no_grid_reaches_the_reference_retention"])
        self.assertLess(self.short["best_achievable_worst_case_retention"],
                        MIN_RETAINED_FRACTION)
        self.assertIn("_what_that_means", self.short)

    def test_the_rule_that_chose_them_is_stored_with_them(self):
        rule = self.short["_rule"]
        self.assertEqual(rule["hard_conditions"]["max_change_free_ceiling_rate"],
                         MAX_CEILING_RATE)
        self.assertEqual(
            rule["hard_conditions"]["min_read_span_in_change_free_sd"],
            MIN_READ_SPAN_IN_SD)
        self.assertEqual(rule["then_maximise"],
                         "worst_retained_over_the_noise_range")
        self.assertIn("before the enumeration ran", rule["_why_these"])

    def test_every_pick_reaches_below_the_windows_lower_edge(self):
        # Load-bearing and not obvious: a change centred at step 512 puts half
        # its mass below 512, so a sweep starting there cannot see it and reads
        # the change well above the window. Every pick carries the window's own
        # lower edge and checkpoints beneath it.
        for key, row in self.picks.items():
            self.assertIn(int(WINDOW[0]), row["steps"], key)
            below = [s for s in row["steps"] if s < WINDOW[0]]
            self.assertGreaterEqual(len(below), 1, f"{key}: {row['steps'][:6]}")

    def test_what_each_pick_gives_up_is_recorded(self):
        # Adding the early log-spaced checkpoints pulls the sweep's midpoint
        # down into the window, so a grid that passes may be one that drops
        # checkpoints another prediction wants. The omission is a cost and
        # belongs in the record rather than in the diff.
        for key, row in self.picks.items():
            self.assertIn("omits_published_log_steps", row)

    def test_the_cost_of_reaching_the_late_checkpoints_is_recorded(self):
        if "best_reaching_past_step_100000" in self.short:
            self.assertIn("_what_reaching_that_far_costs", self.short)
            self.assertGreaterEqual(
                self.short["_what_reaching_that_far_costs"], 0.0)


class TestTheEnumeratedFamily(unittest.TestCase):

    def test_the_answer_does_not_rest_on_an_artificial_bound(self):
        fs = _record()["feasible_set"]
        self.assertTrue(fs["frontier_is_interior_to_the_family"]["interior"],
                        fs["frontier_is_interior_to_the_family"]
                        .get("shortlist_rows_resting_on_an_artificial_bound"))

    def test_the_real_bounds_are_named_as_pythias_rather_than_this_files(self):
        interior = _record()["feasible_set"]["frontier_is_interior_to_the_family"]
        self.assertIn("real_bounds", interior)
        self.assertEqual(interior["real_bounds"]["largest_tail_high"]["value"],
                         max(PUBLISHED_LINEAR_STEPS))

    def test_the_family_contains_the_schedules_this_project_already_wrote(self):
        pilot = _pilot_steps()
        self.assertEqual(pilot[:11], PUBLISHED_LOG_STEPS)
        self.assertTrue(set(pilot) <= _published())

    def test_the_schedule_copies_match_the_registrys_own_source(self):
        # `core/pythia_registry.py` imports transformers at scope, so this tool
        # copies its schedules rather than importing them. A copy nobody
        # compares is the hand-synced constant lint rule 3 exists to find, so
        # the tool reads the registry's assignments out of the SOURCE with
        # `ast` -- which works with no dependency at all, here included.
        self.assertEqual(_check_the_schedule_copies(), [])
        reg = _registry_schedules()
        self.assertEqual(set(reg["PYTHIA_ALL_STEPS"]), _published())
        self.assertEqual(tuple(_pilot_steps()),
                         tuple(reg["PYTHIA_410M_PILOT_STEPS"]))

    def test_the_family_is_named_as_a_family_and_not_as_the_power_set(self):
        fam = _record()["feasible_set"]["_the_family"]
        self.assertIn("_not_the_power_set", fam)
        self.assertIn("2^154", fam["_not_the_power_set"])

    def test_the_degeneracies_are_not_what_binds(self):
        # Worth pinning because it is the shape of the answer: most of the
        # family clears the degeneracy conditions and a few percent of it meets
        # the graded ones. A record where the degeneracies did the work would
        # mean the graded conditions were decoration.
        fs = _record()["feasible_set"]
        total = sum(fs["counts"].values())
        self.assertGreater(fs["counts"]["feasible"] / total, 0.25)
        self.assertLess(
            fs["shortlist"]["_n_grids_meeting_the_hard_conditions"],
            0.1 * total)


class TestTheRefusals(unittest.TestCase):

    def test_a_zero_width_window_is_refused_rather_than_divided_by(self):
        with self.assertRaises(C.ColocationRefused):
            C.grid_feasibility([0, 512, 1000, 2000], (1000.0, 1000.0))

    def test_a_negative_noise_level_is_refused(self):
        with self.assertRaises(C.ColocationRefused):
            C.grid_feasibility([0, 512, 1000, 2000], WINDOW,
                               noise_to_range=-0.01,
                               change_width_log_step=0.35)

    def test_a_zero_change_width_is_refused_rather_than_read_as_the_sharp_limit(self):
        with self.assertRaises(C.ColocationRefused):
            C.grid_feasibility([0, 512, 1000, 2000], WINDOW,
                               noise_to_range=0.02, change_width_log_step=0.0)

    def test_the_grid_checks_are_the_modules_own(self):
        with self.assertRaises(C.ColocationRefused):
            C.grid_feasibility([1000, 512], WINDOW)


class TestTheAnchorArmCarriesIt(unittest.TestCase):
    """
    A reader discounting an anchor result needs the grid's numbers on the
    record, not only in a tool. `anchor_arm` refuses on condition 1, so every
    record that emits is one where the other conditions decide how much to
    believe.
    """

    def test_an_emitted_anchor_record_carries_the_grid_arithmetic(self):
        steps = np.asarray([0, 512, 1000] + list(range(2000, 29001, 1000)),
                           dtype=float)
        rng = np.random.default_rng(19)

        def logistic(mid_step, noise=0.02):
            x = step_x(steps)
            v = 1.0 / (1.0 + np.exp(-(x - np.log10(mid_step + 1.0)) / 0.35))
            return v + noise * rng.standard_normal(x.size)

        under_test = [logistic(1000.0) for _ in range(3)]
        ctrl = {f"c{i}": [logistic(10.0 ** rng.uniform(3.8, 4.6))
                          for _ in range(3)]
                for i in range(19)}
        dirs = {k: "rise" for k in ctrl}
        res = C.anchor_arm(steps, under_test, "rise", WINDOW, ctrl, dirs,
                           alpha=0.05, unit_name="layer", arm_name="test")
        self.assertIn("grid_feasibility", res)
        self.assertTrue(res["grid_feasibility"]["reference_outside_window"])
        # Grid-only: the arm does not know the series' noise or change width,
        # so the conditions that need them are absent rather than guessed.
        self.assertNotIn("retained_window_fraction", res["grid_feasibility"])


if __name__ == "__main__":                       # pragma: no cover
    unittest.main()
