"""
tests/test_p_st1_dry_run.py — P-ST1's gate run on inputs with known answers.

`tools/dry_run_p_st1.py` runs the shipped gate on a cloud planted entirely
inside one arm (correct verdict fixed a priori), on an observed pair drawn
from the NULL'S OWN family (P(p <= alpha) <= alpha by construction), and over a
sweep of the gate's input space with a perfect-input counterfactual in every
cell. That takes about ten minutes, so the record is committed and this module
pins it — the same division `tests/test_claim_c_dry_run.py` uses for CLAIM-C's
dry run and `tests/test_p6_projector_audit.py` for the projector audit.

Three things are pinned that a "does the JSON parse" test would not catch.

**The sha256 of both files the record describes.** The verdicts come from
`steering_gate.py` and its constants were set from the committed calibration;
either changing makes every number here a description of something that no
longer exists, and nothing else in the suite would notice.

**Exchangeability, re-derived in milliseconds rather than only stored.** The
one arm whose correct answer follows from the construction rather than from a
modelled H0 family is the observed pair drawn as a random re-split of a fixed
union. A handful of those are run here without reading the record, so the
committed rate is confirmed rather than trusted.

**The finding that changed the gate.** `TestTheAttainableFloorIsNotTheDrawCount`
pins what the dry run found: sum(D) cannot exceed 2m, null re-splits on an
occupied union reach 2m often, and the smallest expressible p is therefore a
property of the layer. Before this the gate reported 1/(draws+1) as its floor
and returned "not significant" from designs that could not have rejected —
`POPPER_PLAN.md` §6l's defect for CLAIM-C, here.
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

from p7_motifs.steering_gate import (
    NULL_FAMILY,
    p_value_p_st1,
    resplit_pair,
    union_basis,
)
from tools.dry_run_p_st1 import (
    CALIBRATION_PATH,
    GATE_PATH,
    RECORD_PATH,
    RECORD_SCHEMA_VERSION,
    check_record,
    planted_layer,
)


def _record() -> dict:
    return json.loads(RECORD_PATH.read_text())


class TestTheRecordIsPresentAndCurrent(unittest.TestCase):

    def test_it_exists_and_parses(self):
        self.assertTrue(RECORD_PATH.exists(),
                        f"{RECORD_PATH} is missing; run "
                        f"`python3 -m tools.dry_run_p_st1 --write`")
        self.assertEqual(_record()["schema_version"], RECORD_SCHEMA_VERSION)

    def test_check_record_is_clean(self):
        self.assertEqual(check_record(), [])

    def test_it_describes_the_files_on_disk(self):
        rec = _record()
        for key, path in (("gate", GATE_PATH),
                          ("calibration", CALIBRATION_PATH)):
            self.assertEqual(
                rec[f"{key}_sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
                f"{path.name} has changed since the dry run was written; "
                f"rerun --write rather than editing the hash")

    def test_it_is_about_the_null_the_module_adjudicates(self):
        self.assertEqual(_record()["null_family"], NULL_FAMILY)

    def test_it_is_not_an_adjudication(self):
        rec = _record()
        self.assertIn("not an adjudication", rec["_not"])
        self.assertIn("synthetic", rec["_not"])


class TestThePlantedAnswerComesBack(unittest.TestCase):
    """
    The arm that would say the criterion does not mean what it says.
    """

    def test_every_emitted_verdict_is_the_planted_one(self):
        s = _record()["sharp_input"]
        self.assertTrue(s["every_emitted_verdict_correct"])
        for row in s["rows"]:
            if row["n_refused"]:
                continue
            self.assertEqual(set(row["verdicts"]), {row["expected_verdict"]})

    def test_essentially_every_pair_informs_on_a_perfect_input(self):
        """
        NOT "the statistic is at its maximum". The PLANTING is perfect -- the
        cloud lies entirely in one arm and dim U_pos = dim(occupied), so every
        drawn direction is inside it -- but the statistic is not deterministic:
        a direction can land where both arms' effective-rank changes share a
        sign, and that pair contributes D = 0. Measured at about one pair in
        forty-eight. Asserting exact maximality would be asserting something
        about the draw rather than about the gate, which is how a test starts
        failing for the wrong reason.
        """
        for row in _record()["sharp_input"]["rows"]:
            self.assertGreaterEqual(
                row["informative_rate"], 0.95,
                f"{row['planted']} at {row['n_pairs']} pairs was not a "
                f"near-perfect input, so what it checks is not what it says")

    def test_both_directions_are_exercised(self):
        planted = {r["planted"] for r in _record()["sharp_input"]["rows"]}
        self.assertEqual(planted, {"H1", "INVERTED"})

    def test_it_still_returns_the_planted_verdict_when_run_here(self):
        """Re-derived rather than only stored, at one cell, in milliseconds."""
        rng = np.random.default_rng(2026)
        X, u_pos, u_neg = planted_layer(rng, c_pos=3.0, c_neg=0.0)
        res = p_value_p_st1(X, u_pos, u_neg, 8, n_draws=39, with_profile=False)
        self.assertEqual(res["verdict"], "TRACKS-DECOMPOSITION")


class TestTheAttainableFloorIsNotTheDrawCount(unittest.TestCase):
    """
    What the dry run found, and what it changed in the gate.

    The gate reported 1/(draws + 1) as the smallest p it could express. On a
    union the cloud occupies, many random re-splits already reach the largest
    value sum(D) can take, and every one of them ties an observation there. So
    a design could be incapable of rejecting and would report "not
    significant" -- which on an entry whose whole value is that it can lose
    reads as a loss. `POPPER_PLAN.md` 6l found the same defect in CLAIM-C's
    gate from the informative-row side.
    """

    def test_the_record_says_the_two_floors_differ_somewhere(self):
        s = _record()["sharp_input"]
        self.assertFalse(
            s["perfect_input_hits_the_draw_count_floor_everywhere"],
            "if a perfect input reached 1/(draws+1) at every pair count, the "
            "distinction this arm exists to establish would be empty")

    def test_a_perfect_input_lands_on_the_ATTAINABLE_floor(self):
        s = _record()["sharp_input"]
        self.assertTrue(
            s["perfect_input_hits_its_attainable_floor_wherever_it_emits"])

    def test_the_gate_refuses_where_neither_tail_can_reach_alpha(self):
        rng = np.random.default_rng(11)
        X, u_pos, u_neg = planted_layer(rng, c_pos=1.5, c_neg=1.5)
        res = p_value_p_st1(X, u_pos, u_neg, 1, n_draws=39, with_profile=False)
        if res["refusal_kind"] is None:          # one tail still reachable
            self.assertLessEqual(min(res["attainable_p_greater"],
                                     res["attainable_p_reciprocal"]), 0.05)
        else:
            self.assertEqual(res["refusal_kind"], "null_ties_the_maximum")
            self.assertGreater(res["best_attainable_p"], 0.05)


class TestExchangeabilityHolds(unittest.TestCase):
    """
    The one arm whose answer needs no modelling assumption at all.
    """

    def test_the_record_says_it_holds(self):
        ex = _record()["exchangeable_input"]
        self.assertTrue(ex["holds"])
        self.assertGreater(ex["n_emitted"], 0,
                           "an arm that emitted nothing cannot have failed")

    def test_the_arms_were_occupied_where_the_retired_null_failed(self):
        """
        Running this check on a bland population would be an arm incapable of
        failing: both arms at chance occupancy is exactly where the retired
        matched-dimension null was valid too.
        """
        ex = _record()["exchangeable_input"]
        for occ in ex["mean_occupancy_of_each_arm"]:
            self.assertGreater(occ, 1.3)

    def test_a_few_draws_re_derived_here_do_not_reject(self):
        """
        Confirmed rather than stored. Six draws cannot measure a rate; what
        they can catch is the implementation not being the null it describes,
        which shows up as rejection on most of them rather than on 5% of them.
        """
        rng = np.random.default_rng(7)
        rejects = 0
        for i in range(6):
            X, u_pos, u_neg = planted_layer(rng, c_pos=1.5, c_neg=1.5)
            union = union_basis(u_pos, u_neg)
            a, b = resplit_pair(union, u_pos.shape[1], rng)
            res = p_value_p_st1(X, a, b, 8, n_draws=39, with_profile=False,
                                seed=500 + i)
            if res.get("p_value") is None:
                continue
            rejects += int(res["p_value"] <= 0.05)
        self.assertLessEqual(rejects, 1)


class TestTheBandAndWhatPredictsIt(unittest.TestCase):

    def test_every_cell_carries_its_perfect_input_counterfactual(self):
        band = _record()["verdict_band"]
        for c in band["cells"]:
            self.assertEqual(len(c["perfect_input_verdicts"]),
                             band["n_perfect_seeds"])
            self.assertIsInstance(c["no_verdict_in_any_draw"], bool)

    def test_a_cell_marked_dead_had_no_verdict_in_ANY_of_its_draws(self):
        """
        The first version of this arm read the counterfactual off ONE draw and
        marked cells dead whose own draws reached a verdict 28% of the time.
        This is the consistency the field name now has to earn.
        """
        for c in _record()["verdict_band"]["cells"]:
            if c["no_verdict_in_any_draw"]:
                self.assertEqual(c["tracks"], 0.0)
                self.assertEqual(c["inverts"], 0.0)
                self.assertEqual(c["perfect_input_reaches_a_verdict"], 0)

    def test_a_symmetric_cell_reaches_no_verdict_and_that_is_correct(self):
        """
        c_pos == c_neg makes the two arms statistically identical, so a label
        swap is a distributional identity and INSUFFICIENT is the right answer.
        A cell like that showing a high TRACKS rate would be the invalidity
        this pass retired the previous null over, back again.
        """
        sym = [c for c in _record()["verdict_band"]["cells"]
               if c["c_pos"] == c["c_neg"] and c["c_pos"] > 0]
        self.assertTrue(sym, "no symmetric cell in the band")
        for c in sym:
            self.assertLessEqual(c["tracks"], 0.2)
            self.assertLessEqual(c["inverts"], 0.2)

    def test_the_occupancy_readout_is_reported_with_its_separation(self):
        o = _record()["occupancy_readout"]
        self.assertGreater(o["n_runs"], 0)
        self.assertIn("perfectly_separated", o)
        self.assertIn("TRACKS-DECOMPOSITION", o["mean_log_ratio_by_verdict"])


class TestRefusalsAndBranches(unittest.TestCase):

    def test_every_refusal_was_reached(self):
        self.assertTrue(_record()["refusals_and_branches"]
                        ["all_refusals_reached"])

    def test_no_refusal_turned_away_something_that_could_have_cleared_alpha(self):
        self.assertTrue(_record()["refusals_and_branches"]
                        ["no_refusal_turned_away_a_clearable_input"])

    def test_every_verdict_branch_fired_on_the_input_built_for_it(self):
        rb = _record()["refusals_and_branches"]
        self.assertTrue(rb["all_branches_reached"], rb["verdict_branches"])
