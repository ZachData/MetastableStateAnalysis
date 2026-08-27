"""
tests/test_claim_b_p_i1_dry_run.py — CLAIM-B and P-I1 on inputs whose answer is
known.

`tools/dry_run_claim_b_p_i1.py` runs both gates on inputs whose correct verdict
is fixed a priori, measures what the anchor arm does when handed a series with
no located change, and re-scores what the refusal that answers it costs. That
takes about six minutes, so the record is committed and this module pins it —
the division `tests/test_claim_c_dry_run.py`, `tests/test_p_st1_dry_run.py` and
`tests/test_p6_r2_r4_dry_run.py` already use.

Two assertions carry the weight.

`TestTheFindingIsStillInTheRecord` is the one the change rests on. CLAIM-B's
anchor arms gained a refusal on the evidence that on the registered cheap-tier
sweep the arm rejects on a change-free input at the same rate as on a perfectly
anchored one. If the record stops showing that, the refusal has nothing behind
it — so the test fails rather than quietly agreeing with the module.
`POPPER_PLAN.md` §6h found an audit arm reporting PASS while incapable of
failing; this is the same question asked of a refusal.

`TestPI1WasLeftAlone` is the other half, and it is `P6-R4`'s precedent: leaving
an entry unchanged is a decision, and the measurement behind it belongs in the
gate rather than in an argument about the two statistics.
"""

from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pytest.ini [pytest] markers.
pytestmark = pytest.mark.pure

from core import changepoint_colocation as C
from tools.dry_run_claim_b_p_i1 import (
    CONSTRUCTION_PATH,
    FORMATION_GATE_PATH,
    RECORD_PATH,
    RECORD_SCHEMA_VERSION,
    check_record,
)


def _record() -> dict:
    return json.loads(RECORD_PATH.read_text())


class TestTheRecordIsPresentAndCurrent(unittest.TestCase):

    def test_it_exists_and_parses(self):
        self.assertTrue(RECORD_PATH.exists(),
                        f"{RECORD_PATH} is missing; run "
                        f"`python3 -m tools.dry_run_claim_b_p_i1 --write`")
        self.assertEqual(_record()["schema_version"], RECORD_SCHEMA_VERSION)

    def test_it_describes_the_files_on_disk(self):
        """
        The record is committed because six minutes is too long for the gating
        tier. The cost of that is that it can go stale, so the hash of every
        file it describes is pinned: a change to the construction fails here
        rather than leaving a record that quietly stops being about it.
        """
        rec = _record()
        for key, path in (("construction", CONSTRUCTION_PATH),
                          ("formation_gate", FORMATION_GATE_PATH)):
            with self.subTest(file=path.name):
                self.assertEqual(
                    rec[f"{key}_sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    f"{path.name} has changed since the record was written; "
                    f"rerun `--write` rather than editing the hash")

    def test_check_record_is_clean(self):
        problems = check_record()
        self.assertEqual(problems, [], "\n".join(problems))

    def test_it_adjudicates_nothing_and_says_so(self):
        rec = _record()
        self.assertIn("not an adjudication", rec["_not"])
        self.assertIn("synthetic", rec["_not"])


class TestTheKnownAnswers(unittest.TestCase):

    def test_every_row_returned_the_verdict_fixed_in_advance(self):
        ka = _record()["known_answer"]
        for row in ka["rows"]:
            with self.subTest(entry=row["entry"], input=row["input"]):
                self.assertTrue(
                    row["as_expected"],
                    f"{row['entry']} on {row['input']!r} expected "
                    f"{row['expected']} and returned {row['verdicts']}")

    def test_all_three_verdict_branches_are_reached(self):
        """A verdict nothing can trigger is a defect (`POPPER_PLAN.md` §6h)."""
        reached = _record()["known_answer"]["branches_reached"]
        for branch in ("CO-LOCATES", "RE-ANCHORS", "INSUFFICIENT"):
            self.assertIn(branch, reached)

    def test_claim_bs_falsification_branch_fires_with_no_margin(self):
        """
        RE-ANCHORS needs both anchor arms' reciprocal p at their floor, and
        that floor is exactly alpha at nineteen controls. So the control count
        `EVALUABILITY.md` records as the minimum for CO-LOCATES is also the
        exact minimum for the falsifier, with no margin either way.
        """
        m = _record()["known_answer"]["re_anchors_margin"]
        self.assertTrue(m["floor_equals_alpha"])
        self.assertGreater(m["measured_rate_on_the_input_built_for_it"], 0.0)
        self.assertLess(m["measured_rate_on_the_input_built_for_it"], 1.0,
                        "if the branch fires on every draw the margin claim "
                        "this record makes is not the one it measured")


class TestTheFindingIsStillInTheRecord(unittest.TestCase):
    """The refusal added on 2026-08-27 rests on exactly these numbers."""

    def test_the_registered_sweep_puts_a_change_free_series_in_the_window(self):
        grids = _record()["grids"]
        lo, hi = C.CLAIM_B_ANCHOR_WINDOW
        centre = grids["cheap-25 (registered)"]["uniform_profile_centroid_step"]
        self.assertTrue(lo < centre < hi,
                        f"a series with no located change lands at step "
                        f"{centre:.0f}, and the finding is that this is inside "
                        f"CLAIM-B's own {lo:.0f}-{hi:.0f} window")

    def test_the_arm_cannot_discriminate_there(self):
        ad = _record()["anchor_discrimination"]
        self.assertLessEqual(
            ad["registered_sweep_discrimination"], 0.10,
            "the anchor arm's rejection rate on a perfectly anchored input no "
            "longer equals its rate on a change-free one; the refusal is not "
            "supported by the artifact that supports it")
        self.assertGreaterEqual(
            ad["registered_sweep_rejects_a_change_free_input"], 0.5)

    def test_and_recovers_where_the_reference_sits_outside_the_window(self):
        """
        The other direction, and it is what stops this being a claim that the
        arm is useless everywhere: on a grid whose midpoint is outside the
        window the arm separates the two inputs.
        """
        self.assertTrue(_record()["anchor_discrimination"][
            "discrimination_recovers_off_the_registered_grid"])

    def test_the_mechanism_is_predicted_and_not_only_measured(self):
        gp = _record()["grid_pull"]
        self.assertTrue(gp["closed_form_tracks_the_measurement"],
                        f"worst centroid prediction error "
                        f"{gp['max_absolute_centroid_prediction_error']}")

    def test_a_denser_sweep_moves_the_centroid_further(self):
        """
        The part that is counterintuitive, and the reason the module's power
        argument for change-mass weighting does not settle it: the pull grows
        with the interval count, so more checkpoints make it worse.
        """
        rows = [r for r in _record()["grid_pull"]["rows"]
                if r["noise_sd"] == 0.02]
        by_n = sorted(rows, key=lambda r: r["n_intervals"])
        shares = [r["predicted_noise_mass_share"] for r in by_n]
        self.assertEqual(shares, sorted(shares))
        self.assertGreater(shares[-1], shares[0])


class TestWhatTheControlFamilyDecides(unittest.TestCase):
    """
    The arm that corrected this pass's own first attempt at the refusal, which
    is why it is pinned rather than left in the record as background.
    """

    def test_the_rate_follows_the_closed_form(self):
        cf = _record()["change_free_rate_vs_family"]
        self.assertTrue(
            cf["closed_form_holds"],
            f"worst error {cf['max_absolute_error_against_the_closed_form']} "
            f"against 1/(k+1)")

    def test_the_references_rank_cannot_see_that_axis(self):
        """
        Which is the reason the refusal is built on the ceiling and not on the
        rank. If this ever stops being true the condition should be revisited,
        so it fails rather than quietly agreeing.
        """
        self.assertTrue(_record()["change_free_rate_vs_family"][
            "reference_rank_is_flat_across_the_family_axis"])

    def test_the_rate_spans_from_certain_to_nominal(self):
        rows = _record()["change_free_rate_vs_family"]["rows"]
        rates = [r["reject_on_a_change_free_input"] for r in rows]
        self.assertGreaterEqual(max(rates), 0.9)
        self.assertLessEqual(min(rates), 0.10)

    def test_and_the_anchored_input_keeps_rejecting_throughout(self):
        """So the rate above is about the change-free input, not about power."""
        rows = _record()["change_free_rate_vs_family"]["rows"]
        self.assertGreaterEqual(
            min(r["reject_on_an_anchored_input"] for r in rows), 0.9)


class TestWhatTheRefusalCosts(unittest.TestCase):
    """
    §6l's counterfactual discipline, asked of a refusal that does NOT come out
    at zero. The point of the arm is that the cost is measured and recorded
    rather than asserted to be small.
    """

    def test_it_costs_verdicts_and_the_record_says_so(self):
        rc = _record()["refusal_cost"]
        self.assertTrue(rc["costs_verdicts_somewhere"])
        self.assertGreater(rc["max_verdict_cost"], 0.0)

    def test_and_costs_nothing_where_it_does_not_fire(self):
        self.assertTrue(_record()["refusal_cost"][
            "costs_nothing_where_it_does_not_fire"])


class TestPI1WasLeftAlone(unittest.TestCase):
    """
    P-I1's null is unchanged and CLAIM-B's anchor arms are not. Leaving an
    entry alone is a decision, and this is the measurement behind it — the
    precedent `P6-R4` set on 2026-08-26.
    """

    def test_the_mutual_arm_holds_on_the_families_that_break_the_anchor_arm(self):
        pi = _record()["p_i1_unaffected"]
        self.assertTrue(
            pi["holds"],
            f"the mutual arm's H0 rate reaches {pi['range']} against alpha "
            f"{pi['alpha']}; P-I1 was left unchanged on the evidence that it "
            f"is unaffected")

    def test_it_was_measured_where_the_anchor_arm_fails_hardest(self):
        """Measuring it on a friendlier grid would be choosing the easy case."""
        self.assertEqual(_record()["p_i1_unaffected"]["grid"],
                         "cheap-25 (registered)")

    def test_a_change_free_series_is_among_the_families_measured(self):
        families = {r["h0_family"] for r in _record()["p_i1_unaffected"]["rows"]}
        self.assertIn("both series change nowhere", families)


class TestTheSecondImplementationIsPinned(unittest.TestCase):
    """
    The dry run scores the anchor arm itself, because it has to reach the rate
    the module now refuses. `POPPER_PLAN.md` §6g records a second
    implementation of a gate's arithmetic as a real risk on CLAIM-C's fast
    path; this is the same check.
    """

    def test_it_agrees_with_the_module_wherever_the_module_emits(self):
        ma = _record()["module_agreement"]
        self.assertGreater(ma["n_compared"], 0)
        self.assertTrue(ma["agrees"],
                        f"worst difference {ma['max_absolute_difference']}")


class TestTheRecordIsReadableWithoutTheCode(unittest.TestCase):

    def test_every_arm_says_what_it_is(self):
        rec = _record()
        for arm in ("known_answer", "anchor_discrimination", "grid_pull",
                    "change_free_rate_vs_family", "refusal_cost",
                    "p_i1_unaffected", "module_agreement"):
            with self.subTest(arm=arm):
                self.assertIn("_what", rec[arm])
                self.assertGreater(len(rec[arm]["_what"]), 40)

    def test_the_generation_cost_is_measured_rather_than_quoted(self):
        """
        `POPPER_PLAN.md` §6n's tool understated its own cost in three places.
        A number the tool measures on every write cannot drift from the tool,
        so the docstrings point here instead of carrying a figure.
        """
        self.assertGreater(_record()["elapsed_seconds"], 0.0)

    def test_the_placed_constants_are_named_as_placed(self):
        fam = _record()["synthetic_family"]
        self.assertIn("_placed", fam)
        self.assertIn("calibrate_changepoint_", fam["_placed"])
