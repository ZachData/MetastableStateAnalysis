"""
tests/test_claim_c_dry_run.py — CLAIM-C's gate run on inputs with known answers.

`tools/dry_run_claim_c.py` runs the shipped gate on a self-comparison (one
model as BOTH arms, so every cell is concordant and the correct verdict is
fixed a priori) and on a power curve over the number of concordant cells. That
takes about five minutes, so the record is committed and this module pins it —
the same division `tests/test_p6_projector_audit.py` uses for the projector
audit and `tests/test_claim_c_homogeneity.py` for the calibration curve.

Two things are pinned that a plain "does the JSON parse" test would not catch.

**The sha256 of BOTH files the record describes.** The verdicts come from
`replication_gate.py` and the boundary comes from the committed homogeneity
curve; either changing makes every number here a description of something that
no longer exists, and nothing else in the suite would notice.

**The headline boundary, re-derived from scratch in milliseconds.** A pinned
number that is only ever compared against itself is a number nobody has
checked. `TestBoundaryReDerived` runs the gate on the self-comparison at eight
prompts and finds the boundary again without reading the record, so the stored
0.8125 is confirmed rather than stored. Three passes running, this project has
found defects in generated artifacts that no test failed on — see
`POPPER_PLAN.md` §6g item 5 and §6i — and the cheap half of the recomputation
is worth having in the gate.

The most consequential assertion here is `TestGateIsConstantAboveTheBoundary`.
Above homogeneity 0.8125 at eight prompts the gate returns INSUFFICIENT for
every possible input, including a perfect one, so its hard stop fires
unconditionally there. §6g's own caution is that a stop rule that always fires
carries no information; this pins where CLAIM-C's has one.
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

from p1_mstate_tracking.replication_gate import CLAIM_C_METRICS
from tools.dry_run_claim_c import (
    CURVE_PATH,
    GATE_PATH,
    N_PROMPTS_SWEPT,
    POWER_N_PROMPTS,
    RECORD_PATH,
    RECORD_SCHEMA_VERSION,
    build_arms,
    check_record,
    correction_is_monotone,
    homogeneity_of,
    independent_sign_reference,
    minority_counts,
    refusal_boundary_bins,
    refusal_kind,
    sign_table,
)


def _record() -> dict:
    return json.loads(RECORD_PATH.read_text())


def _self_compare(n_prompts: int, total_minority: int, seed: int = 3) -> dict:
    """The gate with one synthetic model as both reference and candidate."""
    from p1_mstate_tracking.replication_gate import p_value_claim_c

    rng = np.random.default_rng(seed)
    signs = sign_table(n_prompts,
                       minority_counts(n_prompts, total_minority,
                                       len(CLAIM_C_METRICS)))
    ref_t, ref_r = build_arms(signs, 36, rng)
    can_t, can_r = build_arms(signs, 24, rng)
    return p_value_claim_c(ref_t, ref_r, can_t, can_r, candidate_step0=None,
                           step0_absent_reason="test: no run exists")


class TestTheRecordExists(unittest.TestCase):

    def test_committed(self):
        self.assertTrue(
            RECORD_PATH.exists(),
            f"{RECORD_PATH} is missing. Regenerate with "
            f"`python3 -m tools.dry_run_claim_c --write` (~5 minutes). It is "
            f"committed rather than computed on demand for the same reason as "
            f"the other three artifacts in docs/CI_BASELINE.md.")

    def test_schema_version(self):
        self.assertEqual(_record()["schema_version"], RECORD_SCHEMA_VERSION)

    def test_check_record_is_clean(self):
        self.assertEqual(check_record(), [])

    def test_it_is_not_an_adjudication(self):
        """
        The record is a validation of the gate on synthetic inputs. If this
        ever reads as evidence about pythia-1.4b the whole point is lost.
        """
        rec = _record()
        self.assertIn("Not evidence about pythia-1.4b", rec["_not"])
        self.assertEqual(rec["metrics"], list(CLAIM_C_METRICS))


class TestDescribesTheFilesOnDisk(unittest.TestCase):
    """
    Every verdict in the record is a joint property of the gate and the
    committed homogeneity curve. Hashing only one of them would leave the other
    free to drift.
    """

    def test_gate_sha256_matches(self):
        self.assertEqual(_record()["gate_sha256"],
                         hashlib.sha256(GATE_PATH.read_bytes()).hexdigest(),
                         "replication_gate.py has changed since the dry run "
                         "was written; rerun --write rather than editing the "
                         "hash")

    def test_curve_sha256_matches(self):
        self.assertEqual(_record()["curve_sha256"],
                         hashlib.sha256(CURVE_PATH.read_bytes()).hexdigest(),
                         "the homogeneity curve has changed since the dry run "
                         "was written; the boundary it records may have moved")


class TestTheSelfComparisonIsOne(unittest.TestCase):
    """
    Structural facts that must hold at every prompt count, or the thing that
    ran was not a self-comparison and its verdicts mean nothing.
    """

    def test_every_cell_concordant(self):
        for n, row in _record()["self_comparison"]["per_n_prompts"].items():
            self.assertTrue(row["all_cells_concordant"], f"n_prompts={n}")

    def test_perfect_input_lands_exactly_on_the_attainable_floor(self):
        """
        Unanimity must not bite on a unanimous input. Every leave-one-out
        subset sees the same perfect table, so each returns the floor and the
        intersection-union max over the seven is that same floor.
        """
        for n, row in _record()["self_comparison"]["per_n_prompts"].items():
            self.assertTrue(row["perfect_input_hits_floor"], f"n_prompts={n}")

    def test_verdict_depends_on_homogeneity_and_not_on_placement(self):
        for n, row in _record()["self_comparison"]["per_n_prompts"].items():
            self.assertTrue(row["all_placements_invariant"], f"n_prompts={n}")

    def test_the_three_refusals_all_fire_somewhere(self):
        """
        Identical rows, a degenerate metric subset, and the derived homogeneity
        refusal. A refusal that never fires anywhere in the input space is one
        nothing has checked -- §6h's lesson about an arm incapable of failing,
        asked of a refusal instead of a PASS.

        The third entry changed on 2026-08-25 from `no-correction-available` to
        `subset-undefined`, and it is the reorder rather than a new behaviour
        (§6l). Near homogeneity 1.0 dropping a metric leaves the remaining sign
        rows identical WITHIN that subset, and the subsets are now scored before
        the curve is consulted -- so the specific degeneracy is reported where
        the generic 'the curve has no row for this' used to be. Both were true
        of those inputs; the more specific one is now in front.
        """
        row = _record()["self_comparison"]["per_n_prompts"][str(POWER_N_PROMPTS)]
        self.assertEqual(
            row["refusal_kinds_seen"],
            ["derived-homogeneity", "identical-rows", "subset-undefined"])


class TestBoundaryReDerived(unittest.TestCase):
    """
    The stored boundary, found again from scratch. Runs the gate about
    thirty times, which is milliseconds.
    """

    def test_boundary_at_eight_prompts(self):
        rec = _record()
        stored = rec["self_comparison"]["per_n_prompts"][
            str(POWER_N_PROMPTS)]["max_passing_homogeneity"]
        n_m = len(CLAIM_C_METRICS)
        passing = []
        for total in range((POWER_N_PROMPTS // 2) * n_m + 1):
            out = _self_compare(POWER_N_PROMPTS, total)
            if out["verdict"] == "TRANSFERS":
                passing.append(out["sign_homogeneity"])
        self.assertTrue(passing, "the self-comparison never returned TRANSFERS")
        self.assertAlmostEqual(max(passing), stored, places=10)
        # 0.8125 until 2026-08-25. The informative-row floor changed what
        # "conditional on emission" conditions on -- draws that could never
        # reject are now refused rather than counted as non-rejections -- so
        # the measured H0 rate rose, the correction got stronger, and the band
        # TIGHTENED. §6l records it; it is a real cost of the refusal and not a
        # regeneration artefact.
        self.assertAlmostEqual(max(passing), 0.7708333333333334, places=10)

    def test_minimum_dissenting_cells_to_pass(self):
        """
        The boundary restated in the unit a pilot can act on: at least this
        many of the 48 candidate cells must carry the minority sign for their
        metric before the gate can return TRANSFERS at all.
        """
        rec = _record()["self_comparison"]["per_n_prompts"][str(POWER_N_PROMPTS)]
        # 9 until 2026-08-25, 11 after -- the band tightened, so the pilot has
        # to clear a higher bar. See `test_boundary_at_eight_prompts`.
        self.assertEqual(rec["min_minority_cells_to_pass"], 11)
        self.assertEqual(_self_compare(POWER_N_PROMPTS, 11)["verdict"],
                         "TRANSFERS")
        self.assertEqual(_self_compare(POWER_N_PROMPTS, 10)["verdict"],
                         "INSUFFICIENT")


class TestGateIsConstantAboveTheBoundary(unittest.TestCase):
    """
    The consequence worth pinning: above the boundary the gate returns
    INSUFFICIENT for EVERY input, so its hard stop fires unconditionally and
    carries no information about the data.
    """

    def test_perfect_input_is_refused_just_above_the_boundary(self):
        out = _self_compare(POWER_N_PROMPTS, 10)
        self.assertIsNone(out["p_value"])
        self.assertEqual(out["verdict"], "INSUFFICIENT")
        self.assertTrue(out["hard_stop"])
        self.assertFalse(out["falsified"])
        self.assertEqual(refusal_kind(out), "derived-homogeneity")

    def test_the_power_curve_above_the_boundary_is_all_refusals(self):
        levels = _record()["power_curve"]["levels"]
        boundary = _record()["self_comparison"]["per_n_prompts"][
            str(POWER_N_PROMPTS)]["max_passing_homogeneity"]
        above = [lv for lv in levels if lv["homogeneity"] > boundary]
        self.assertTrue(above, "no power-curve level sits above the boundary")
        for lv in above:
            for row in lv["rows"]:
                self.assertEqual(row["refused"], 1.0,
                                 f"h={lv['homogeneity']} k={row['k_concordant']}")
                self.assertEqual(row["transfers"], 0.0)
                self.assertEqual(row["fails_to_transfer"], 0.0)


class TestTheDerivedRefusalIsTight(unittest.TestCase):
    """
    R(h, .) non-decreasing in p means `R(h, floor) > alpha` implies no
    attainable p clears alpha -- so the refusal never costs a verdict the gate
    could otherwise have reached. Re-derived from the committed curve rather
    than read off the record, because it is a property of the curve.
    """

    def test_monotone_in_the_committed_curve(self):
        mono = correction_is_monotone()
        self.assertTrue(mono["monotone"], mono["violations"])
        self.assertGreater(mono["n_quantile_vectors_checked"], 100)

    def test_record_agrees(self):
        self.assertTrue(_record()["correction_monotonicity"]["monotone"])


class TestThePowerCurve(unittest.TestCase):
    """
    How much concordance the gate needs, and the decomposition that says where
    the requirement comes from.
    """

    def test_thresholds_at_eight_prompts(self):
        # The levels are placed relative to the boundary, so they moved with it
        # on 2026-08-25 (§6l): 0.8125/0.75/0.625 became 0.7708/0.7083/0.5833.
        by_h = {round(lv["homogeneity"], 4): lv
                for lv in _record()["power_curve"]["levels"]}
        self.assertEqual(by_h[0.7708]["thresholds"]["k_transfers_half"], 38)
        self.assertEqual(by_h[0.7708]["thresholds"]["k_transfers_always"], 43)
        self.assertEqual(by_h[0.7083]["thresholds"]["k_transfers_half"], 35)
        self.assertEqual(by_h[0.7083]["thresholds"]["k_transfers_always"], 38)
        self.assertEqual(by_h[0.5833]["thresholds"]["k_transfers_half"], 35)
        self.assertEqual(by_h[0.5833]["thresholds"]["k_transfers_always"], 38)

    def test_the_requirement_tightens_as_homogeneity_rises(self):
        """
        The reading that matters: the closer the candidate's own contrast is to
        pointing one way on every prompt, the MORE concordance the gate demands
        before it will say so.
        """
        levels = sorted((lv for lv in _record()["power_curve"]["levels"]
                         if lv["thresholds"]["k_transfers_half"] is not None),
                        key=lambda lv: lv["homogeneity"])
        ks = [lv["thresholds"]["k_transfers_half"] for lv in levels]
        self.assertEqual(ks, sorted(ks))
        self.assertLess(ks[0], ks[-1])

    def test_unanimity_costs_less_than_the_correction(self):
        """
        The counterfactual rates separate the two costs. Dropping to the full
        set alone moves the 50% point by a cell or two; removing the
        homogeneity correction moves it by up to five, and only where the
        correction bites.
        """
        by_h = {round(lv["homogeneity"], 4): lv
                for lv in _record()["power_curve"]["levels"]}
        for h in (0.7708, 0.7083, 0.5833):
            t = by_h[h]["thresholds"]
            tool_axis_cost = t["k_iut_uncorrected_half"] - t["k_full_set_only_half"]
            correction_cost = t["k_transfers_half"] - t["k_iut_uncorrected_half"]
            self.assertLessEqual(tool_axis_cost, 3, f"h={h}")
            self.assertGreaterEqual(correction_cost, 0, f"h={h}")
        # Well inside the band the correction is free; at the boundary it is
        # what the requirement is made of.
        self.assertEqual(by_h[0.5833]["thresholds"]["k_transfers_half"]
                         - by_h[0.5833]["thresholds"]["k_iut_uncorrected_half"], 0)
        self.assertGreaterEqual(
            by_h[0.7708]["thresholds"]["k_transfers_half"]
            - by_h[0.7708]["thresholds"]["k_iut_uncorrected_half"], 4)

    def test_falsification_needs_as_much_discordance_as_transfer_needs_concordance(self):
        by_h = {round(lv["homogeneity"], 4): lv
                for lv in _record()["power_curve"]["levels"]}
        for h in (0.7708, 0.7083, 0.5833):
            t = by_h[h]["thresholds"]
            self.assertIsNotNone(t["k_fails_half"], f"h={h}")
            self.assertLessEqual(t["k_fails_half"], 14, f"h={h}")

    def test_the_insufficient_band_is_most_of_the_range(self):
        """
        Between the two branches sits a band where the gate says nothing and
        the hard stop fires. It is worth knowing how wide it is before spending
        forward passes.
        """
        by_h = {round(lv["homogeneity"], 4): lv
                for lv in _record()["power_curve"]["levels"]}
        lo, hi = by_h[0.7708]["thresholds"]["insufficient_band"]
        self.assertEqual([lo, hi], [11, 37])
        self.assertGreater((hi - lo + 1) / 49.0, 0.5)


class TestScaleReferences(unittest.TestCase):
    """
    What fixes the scale the boundary is read against: an independent-prompt
    candidate, and the boundary expressed as a curve bin.
    """

    def test_independent_prompts_are_comfortably_inside_the_band(self):
        """
        The refusal is not tight against chance. Under independent prompt signs
        homogeneity concentrates near 0.64 and the refusal essentially never
        fires -- which is what makes the finding a statement about UNIFORM
        effects rather than a claim that the gate is unusable.
        """
        for n in N_PROMPTS_SWEPT:
            ref = independent_sign_reference(n, len(CLAIM_C_METRICS))
            self.assertLess(ref["mean_homogeneity"], 0.70, f"n={n}")
            self.assertGreater(ref["mean_homogeneity"], 0.55, f"n={n}")
        # 1.7e-3 at eight prompts since 2026-08-25, against 8e-5 before: the
        # boundary moved down a bin (§6l) so slightly more of the
        # independent-prompt distribution sits above it. Two in a thousand is
        # still "essentially never", which is the claim being made.
        rec = _record()["independent_prompt_reference"]["per_n_prompts"]
        self.assertLess(rec[str(POWER_N_PROMPTS)]["p_above_refusal_boundary"],
                        5e-3)

    def test_the_boundary_does_not_move_with_prompt_count(self):
        """
        Expressed in the unit that is comparable across prompt counts, since
        the attainable homogeneities themselves lie on a grid of step
        1/(n_prompts * n_metrics). Six to twelve prompts move the boundary
        across two bins of 0.025, so more prompts is not the remedy for a
        refusal -- which is the point, and it survived the band tightening on
        2026-08-25 (§6l): every count moved down together.
        """
        bins = refusal_boundary_bins(0.05)
        los = [bins[str(n)]["first_refusing_bin"]["bin_lo"]
               for n in N_PROMPTS_SWEPT]
        self.assertLessEqual(max(los) - min(los), 0.05)
        self.assertGreaterEqual(min(los), 0.775)
        # Six, seven and eight prompts sit a bin BELOW nine and up: the
        # informative-row floor refuses more of the small-n H0 draws, so the
        # rate among the survivors is higher there.
        self.assertEqual(sorted(set(los)), [0.775, 0.825])
        self.assertEqual(_record()["refusal_boundary_bins"][
            str(POWER_N_PROMPTS)]["first_refusing_bin"]["bin_lo"], 0.775)


class TestTheInformativeRowFloorCostsNothing(unittest.TestCase):
    """
    The dual of `TestTheDerivedRefusalIsTight`, asked of the refusal added on
    2026-08-25 (POPPER_PLAN.md §6l) and answered by running the gate rather
    than by reading a curve's monotonicity.

    The informative-row floor refuses a table whose rows cannot move the
    statistic. It is only safe because both tails share that floor, so a
    refused table could not have cleared alpha in either direction. The record
    re-scores every table the refusal fired on and counts the ones that could
    have.
    """

    def setUp(self):
        self.floor = _record()["informative_row_floor"]

    def test_no_refused_table_could_have_cleared_alpha(self):
        """The assertion the whole section exists for."""
        for row in self.floor["rows"]:
            self.assertEqual(
                row["counterfactual_rejections"], 0,
                f"at H1 strength {row['h1_strength']} the floor refused "
                f"{row['counterfactual_rejections']} table(s) that could have "
                f"reached a verdict; the refusal takes power and has to go")
        self.assertIs(self.floor["costs_no_power"], True)

    def test_the_verdict_is_not_true_by_vacuity(self):
        """
        §6h found an audit arm reporting PASS while incapable of failing. A
        sweep in which the refusal never fires reports `costs_no_power` with
        nothing behind it, so the record stores None there rather than True --
        and this pins that the committed sweep actually had refusals to
        re-score.
        """
        self.assertGreater(self.floor["n_refusals_rescored"], 0)
        self.assertTrue(any(r["refusal_rate"] > 0 for r in self.floor["rows"]))

    def test_it_fires_hardest_under_the_null(self):
        """
        The shape that says it is a floor and not a filter on effects: under H0
        a prompt's six metrics split three and three with probability 20/64, so
        the refusal is common; as the effect grows the rows swing and it stops
        firing.
        """
        rows = sorted(self.floor["rows"], key=lambda r: r["h1_strength"])
        self.assertGreater(rows[0]["refusal_rate"], rows[-1]["refusal_rate"])


class TestTheHelpersAreWhatTheyClaim(unittest.TestCase):

    def test_homogeneity_is_linear_in_the_minority_count(self):
        """
        The property that makes sweeping `total` sweep every attainable
        homogeneity: the spread over metrics cancels.
        """
        n_m = len(CLAIM_C_METRICS)
        for n in (6, 8, 11):
            for total in range((n // 2) * n_m + 1):
                s = sign_table(n, minority_counts(n, total, n_m))
                self.assertAlmostEqual(homogeneity_of(s), 1 - total / (n * n_m),
                                       places=12)

    def test_sign_table_plants_exactly_the_requested_minority(self):
        n_m = len(CLAIM_C_METRICS)
        for n in (6, 7, 8, 9, 10, 11, 12):
            for total in range((n // 2) * n_m + 1):
                mino = minority_counts(n, total, n_m)
                s = sign_table(n, mino)
                self.assertEqual([int((s[:, j] < 0).sum()) for j in range(n_m)],
                                 mino, f"n={n} total={total}")

    def test_build_arms_produces_exactly_the_requested_contrast_signs(self):
        from p1_mstate_tracking.replication_gate import contrast

        rng = np.random.default_rng(11)
        signs = sign_table(8, minority_counts(8, 12, len(CLAIM_C_METRICS)))
        trained, random_ = build_arms(signs, 36, rng)
        for i, prompt in enumerate(sorted(trained)):
            for j, metric in enumerate(CLAIM_C_METRICS):
                d = contrast(trained[prompt][metric], random_[prompt][metric])
                self.assertEqual(np.sign(d), signs[i, j],
                                 f"{prompt}/{metric}")

    def test_refusal_kind_is_none_when_a_p_was_emitted(self):
        out = _self_compare(POWER_N_PROMPTS, 12)
        self.assertIsNotNone(out["p_value"])
        self.assertIsNone(refusal_kind(out))


if __name__ == "__main__":
    unittest.main()
