"""
tests/test_p_s1_dry_run.py — P-S1 on inputs whose answer is known.

`tools/dry_run_p_s1.py` runs the gate on inputs whose correct verdict is fixed
a priori and measures what it does when the two arms sit at different cluster
counts. The record is committed and this module pins it — the division the
other four dry-run tests already use.

`TestTheFindingIsStillInTheRecord` is the one the change rests on. P-S1 gained
an (m, d) refusal on the evidence that two I.I.D. arms — H0 realised exactly —
reject at 1.000 once the cluster counts differ by as little as two in
thirty-two. If the record stops showing that, the refusal has nothing behind
it.

`TestTheReImplementationIsPinned` is the guard on the arm that produces it.
The module now refuses every mismatched row, so the dry run reproduces the
arithmetic it used to run — and a second implementation of a gate's arithmetic
is a real risk (`POPPER_PLAN.md` §6g), so it is checked against the module
itself wherever the module still emits.
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

from tools.dry_run_p_s1 import (
    DESIGN_PATH,
    GATE_PATH,
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
                        f"`python3 -m tools.dry_run_p_s1 --write`")
        self.assertEqual(_record()["schema_version"], RECORD_SCHEMA_VERSION)

    def test_it_describes_the_files_on_disk(self):
        rec = _record()
        for key, path in (("gate", GATE_PATH), ("design", DESIGN_PATH)):
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

    def test_the_generation_cost_is_measured_rather_than_quoted(self):
        self.assertGreater(_record()["elapsed_seconds"], 0.0)


class TestTheKnownAnswers(unittest.TestCase):

    def test_every_row_returned_the_verdict_fixed_in_advance(self):
        for row in _record()["known_answer"]["rows"]:
            with self.subTest(input=row["input"]):
                self.assertTrue(
                    row["as_expected"],
                    f"{row['input']!r} expected {row['expected']} and rejected "
                    f"at {row['reject_rate']} with median p {row['median_p']}")

    def test_the_arms_reversed_give_a_high_p(self):
        """
        P-S1 is one-sided, and step-0 being better spread than trained is the
        prediction being WRONG rather than right. A construction that scores it
        as evidence is testing something other than what it claims.
        """
        row = [r for r in _record()["known_answer"]["rows"]
               if r["expected"] == "p high"][0]
        self.assertGreater(row["median_p"], 0.5)
        self.assertEqual(row["reject_rate"], 0.0)


class TestTheFindingIsStillInTheRecord(unittest.TestCase):
    """The (m, d) refusal added on 2026-08-27 rests on exactly these numbers."""

    def test_two_iid_arms_at_different_counts_reject_at_a_high_rate(self):
        ma = _record()["mismatched_arms"]
        self.assertGreaterEqual(
            ma["worst_mismatched_reject_rate"], 0.5,
            "a mismatched pair of I.I.D. arms no longer rejects; the refusal "
            "is not supported by the artifact that supports it")

    def test_and_matched_arms_do_not(self):
        """Otherwise the table separates nothing and the finding is the gate."""
        self.assertLessEqual(_record()["mismatched_arms"]["matched_reject_rate"],
                             0.15)

    def test_a_small_difference_is_enough(self):
        """
        The severity is that it does not take a wild mismatch. Cluster counts
        differ between a trained and a step-0 checkpoint as a matter of course.
        """
        smallest = _record()["mismatched_arms"]["smallest_difference_that_breaks_it"]
        self.assertIsNotNone(smallest)
        self.assertLessEqual(smallest, 8)

    def test_the_error_runs_in_both_directions(self):
        """
        Fewer step-0 clusters confirms the prediction; more makes it unwinnable.
        Half the finding is that neither is an answer an analyst would notice.
        """
        ma = _record()["mismatched_arms"]
        self.assertTrue(ma["a_mismatch_can_also_make_it_unwinnable"])
        signs = {np.sign(r["cluster_difference"]) for r in ma["rows"]}
        self.assertIn(-1, signs)
        self.assertIn(1, signs)


class TestTheRefusal(unittest.TestCase):

    def test_it_turns_away_every_mismatch(self):
        self.assertTrue(_record()["refusal"]["refuses_every_mismatch"])

    def test_and_no_matched_arm(self):
        """A refusal that fires on everything is not a check — §6h."""
        self.assertTrue(_record()["refusal"]["refuses_no_matched_arm"])

    def test_an_arm_that_cannot_be_checked_is_refused_too(self):
        rf = _record()["refusal"]
        self.assertTrue(rf["unverifiable_arm_refused"])
        self.assertIn("cannot be checked", rf["unverifiable_reason"])


class TestTheFallbackNoteWasCorrected(unittest.TestCase):
    """
    The module warned that its `Q_ratio` fallback leaves the p "mildly
    anticonservative" and cited a null-p mean of 0.40. That was measured on
    retired code; on the code that exists the two paths are indistinguishable.
    A rate that stopped describing its own path is `POPPER_PLAN.md` §6m's
    lesson about inlined figures, arriving again.
    """

    def test_the_two_paths_are_indistinguishable(self):
        self.assertTrue(
            _record()["the_fallback_note"]["the_two_paths_are_indistinguishable"])

    def test_and_neither_is_near_the_retired_number(self):
        fn = _record()["the_fallback_note"]
        self.assertTrue(fn["and_neither_path_is_near_it"])
        self.assertEqual(fn["the_retired_number_was"], 0.40)

    def test_the_module_no_longer_claims_it(self):
        from p1c_frames.centroids import p_value_p_s1
        from p1c_frames.design_test import design_report
        rng = np.random.default_rng(0)
        arms = []
        for _ in range(2):
            X = rng.normal(size=(16, 64))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            a = design_report(X, d=64, t_max=3)
            a.pop("Q")
            arms.append(a)
        note = p_value_p_s1(*arms, n_null=40, seed=0)["reference_note"]
        self.assertNotIn("mildly anticonservative", note)
        self.assertIn("cancels", note)


class TestTheFloorIsAttainableHere(unittest.TestCase):
    """
    P-S1's statistic is continuous, so 1/(n_null+1) really is the smallest p a
    run can express. That is the claim that FAILED for `P-ST1`, `P-T1` and
    `P-M1`, all of which have discrete statistics — so it is checked rather
    than assumed.
    """

    def test_a_perfect_input_reaches_the_reported_floor(self):
        fp = _record()["floor_and_power"]
        self.assertTrue(fp["the_floor_is_attainable"])
        self.assertGreater(fp["reached_the_floor_rate"], 0.0)

    def test_and_the_design_has_power_when_used_correctly(self):
        """A calibrated test that never rejects is also useless."""
        self.assertGreaterEqual(_record()["floor_and_power"]["power_at_alpha"], 0.8)


class TestTheReImplementationIsPinned(unittest.TestCase):

    def test_it_agrees_with_the_module_wherever_the_module_emits(self):
        agr = _record()["mismatched_arms"]["module_agreement"]
        self.assertGreater(agr["n_compared"], 0)
        self.assertTrue(agr["agrees"],
                        f"worst difference {agr['max_absolute_difference']}")
