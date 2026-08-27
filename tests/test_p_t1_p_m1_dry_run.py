"""
tests/test_p_t1_p_m1_dry_run.py — P-T1 and P-M1 on inputs whose answer is known.

`tools/dry_run_p_t1_p_m1.py` runs both gates on inputs whose correct verdict is
fixed a priori and measures the attainable floor each design has against the
resolution each one used to report. The record is committed and this module
pins it — the division `tests/test_claim_c_dry_run.py`,
`tests/test_p_st1_dry_run.py`, `tests/test_p6_r2_r4_dry_run.py` and
`tests/test_claim_b_p_i1_dry_run.py` already use.

Two assertions carry the weight.

`TestTheFindingIsStillInTheRecord` is the one the change rests on: both entries
gained an attainable-floor refusal on the evidence that a PERFECT input returns
a p far above what they reported as their floor. If the record stops showing
that gap the refusal has nothing behind it, so the test fails rather than
quietly agreeing with the module.

`TestTheRefusalCostsNothing` is the other half, and it is an ENUMERATION rather
than a measurement — every arrangement the marginals admit, not a sample of
them. `POPPER_PLAN.md` §6l had to re-score a counterfactual because there the
floor and the p came from different code; here they are the same quantity, and
saying which kind of zero it is is the point.
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

from p2d_operator_activation.gradient_flow_condition import p_m1_attainable_floor
from p2d_operator_activation.table1_predictions import p_t1_attainable_floor
from tools.dry_run_p_t1_p_m1 import (
    P_M1_PATH,
    P_T1_PATH,
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
                        f"`python3 -m tools.dry_run_p_t1_p_m1 --write`")
        self.assertEqual(_record()["schema_version"], RECORD_SCHEMA_VERSION)

    def test_it_describes_the_files_on_disk(self):
        rec = _record()
        for key, path in (("p_t1", P_T1_PATH), ("p_m1", P_M1_PATH)):
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
            with self.subTest(entry=row["entry"], input=row["input"]):
                self.assertTrue(
                    row["as_expected"],
                    f"{row['entry']} on {row['input']!r} expected "
                    f"{row['expected']} and returned p={row['p_value']} "
                    f"reason={row['reason']}")

    def test_both_entries_can_return_p_equals_one_with_the_arms_reversed(self):
        """
        A construction that cannot do that is not testing the direction it
        claims to — `POPPER_PLAN.md` §6h's audit arm, asked of a one-sided
        alternative.
        """
        rows = [r for r in _record()["known_answer"]["rows"]
                if r["expected"] == "p = 1"]
        self.assertEqual({r["entry"] for r in rows}, {"P-T1", "P-M1"})
        for r in rows:
            self.assertGreaterEqual(r["p_value"], 0.99)


class TestTheFindingIsStillInTheRecord(unittest.TestCase):
    """The floor added on 2026-08-27 rests on exactly these numbers."""

    def test_both_entries_have_designs_whose_floor_exceeds_alpha(self):
        fl = _record()["the_floor"]
        self.assertTrue(fl["p_t1_has_designs_that_cannot_reject"])
        self.assertTrue(fl["p_m1_has_designs_that_cannot_reject"])

    def test_the_gap_against_the_reported_resolution_is_large(self):
        fl = _record()["the_floor"]
        for entry in ("p_t1", "p_m1"):
            with self.subTest(entry=entry):
                self.assertGreater(
                    fl[f"worst_floor_over_resolution_{entry}"], 10.0,
                    "the design floor no longer exceeds the resolution these "
                    "entries used to report; the change is not supported by "
                    "the artifact that supports it")

    def test_exactly_the_insufficient_designs_are_refused(self):
        """
        Both directions. A refusal that fires on everything is not a check, and
        one that fires on nothing is not either.
        """
        self.assertTrue(_record()["the_floor"]["every_insufficient_design_is_refused"])

    def test_the_record_says_which_constraint_binds_where(self):
        """
        The honest reading, and the reason the old resolution was not wrong
        everywhere: at a small design the marginals bind and at a large one the
        draw count does.
        """
        fl = _record()["the_floor"]
        binds = {r["which_binds"] for r in fl["p_t1"] + fl["p_m1"]}
        self.assertIn("the marginals", binds)
        self.assertIn("the draw count", binds)

    def test_the_perfect_input_tracks_the_design_floor(self):
        """
        Which is what makes the floor a measurement of the design rather than
        an argument about it: where the marginals bind, what a perfect input
        actually returns sits on the computed floor.
        """
        fl = _record()["the_floor"]
        for r in fl["p_t1"] + fl["p_m1"]:
            if r["perfect_input_p"] is None or r["which_binds"] != "the marginals":
                continue
            with self.subTest(row=r.get("n_candidates", r.get("n_layers"))):
                self.assertLess(
                    abs(r["perfect_input_p"] - r["design_floor"]),
                    max(0.25 * r["design_floor"], 0.003))


class TestTheRefusalCostsNothing(unittest.TestCase):

    def test_it_enumerated_something(self):
        """
        A sweep with no refused configuration would report success while being
        incapable of reporting anything else — `POPPER_PLAN.md` §6l's
        `costs_no_power is None, never True`, in its other form.
        """
        self.assertGreater(
            _record()["refusal_costs_nothing"]["n_refused_configurations"], 0)

    def test_no_arrangement_at_a_refused_design_clears_alpha(self):
        rc = _record()["refusal_costs_nothing"]
        self.assertTrue(rc["costs_no_verdict_anywhere"])
        for r in rc["rows"]:
            with self.subTest(entry=r["entry"]):
                self.assertFalse(r["any_clears_alpha"])


class TestValidity(unittest.TestCase):

    def test_h0_rates_hold_conditional_on_emission(self):
        va = _record()["validity"]
        self.assertTrue(
            va["holds"],
            f"an H0 rate conditional on emission reaches {va['range']} against "
            f"a bound of {va['bound']:.4f}")

    def test_the_bound_carries_the_cell_count(self):
        """
        A per-cell 1.96-sigma bound on a proportion in a REGENERATED artifact
        false-alarms once in twenty regenerations by construction, and did once
        already (`POPPER_PLAN.md` §6n).
        """
        va = _record()["validity"]
        self.assertGreater(va["bound_in_standard_errors"], 1.96)

    def test_some_designs_now_never_emit_and_that_is_the_requirement(self):
        """
        The designs whose floor exceeds alpha refuse on every draw. That is not
        a defect in the sweep — it is the pre-computed requirement on the run,
        in the form a reader can see.
        """
        self.assertTrue(_record()["validity"]["some_designs_never_emit"])


class TestTheSharedInstrumentIsRecorded(unittest.TestCase):
    """
    `P-T1` and `P-M1` classify the same head's weights and sit under the SAME
    claim, so a single extraction defect moves both factors of one product.
    `P6-R2`/`P6-R4` and `CLAIM-B`/`P-I1` both record their shared component;
    these two recorded nothing until this pass.
    """

    def test_it_names_both_entries_and_their_claim(self):
        si = _record()["shared_instrument"]
        self.assertEqual(sorted(si["entries"]), ["P-M1", "P-T1"])
        self.assertEqual(si["claim"], "H-OPERATOR")

    def test_it_says_what_it_did_not_measure(self):
        """An asserted dependence must not read as a measured one."""
        self.assertIn("not_measured_here", _record()["shared_instrument"])


class TestTheFloorsAreRederivableInMilliseconds(unittest.TestCase):
    """
    A pinned number only ever compared against itself is a number nobody has
    checked, so the two closed forms are re-derived here from scratch.
    """

    def test_p_t1_matches_the_hypergeometric_at_saturated_marginals(self):
        from math import comb
        for K, C in ((2, 3), (3, 5), (4, 8), (8, 16)):
            with self.subTest(K=K, C=C):
                f = p_t1_attainable_floor(K, C, K)
                self.assertAlmostEqual(f["attainable_floor"],
                                       1.0 / comb(K + C, K), places=12)

    def test_p_m1_matches_one_over_n_choose_t_for_a_binary_series(self):
        from math import comb
        for n, t in ((6, 1), (12, 2), (24, 3), (49, 1)):
            with self.subTest(n=n, t=t):
                v = np.zeros(n)
                v[:t] = 1.0
                f = p_m1_attainable_floor(v)
                self.assertAlmostEqual(f["attainable_floor"],
                                       1.0 / comb(n, t), places=12)

    def test_both_floors_are_derived_from_the_registry_alpha(self):
        """
        No constant is placed in either: the cut is the floor against alpha, so
        it moves when alpha does. That is how a test tells a derived cut from a
        placed one.
        """
        f = p_t1_attainable_floor(3, 5, 3)
        self.assertEqual(f["alpha"], _record()["alpha"])
        self.assertEqual(f["sufficient"], f["attainable_floor"] <= f["alpha"])

    def test_p_m1_says_its_floor_is_a_lower_bound(self):
        """
        A tied regime score makes the true floor larger, so refusing on this
        one can only under-refuse — the safe direction, and the same argument
        as `P-ST1`'s 2m bound.
        """
        f = p_m1_attainable_floor(np.array([0.0, 0.0, 1.0, 1.0]))
        self.assertTrue(f["is_a_lower_bound"])
