"""
tests/test_phase2b_report.py — cross-checkpoint aggregation.

The load-bearing classes are `TestFlatnessHasANoiseScale` (a range is not a
trajectory unless it is large next to the within-checkpoint scatter) and
`TestAlignmentRefusesVerdicts` (status-2's own warning: of the 13
`mixed_or_unattributed` runs in the 40000-100000 window, five sit at
`frac_repulsive` exactly 0.500 against a strict `> 0.5` guard, so "the verdict
label is an artifact of where the threshold happens to fall").
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from p2b_imaginary import p2b_report as rep

import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure

STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 5000,
         7000, 19000, 40000, 60000, 100000, 120000, 143000]


def fake_combined(value_fn, spread=0.01, statistic="henrici_relative",
                  steps=STEPS, n_layers=24, block1b=None, missing=None):
    """
    A `phase2b_results.json` whose Block 1a per-layer values are drawn around
    `value_fn(step)` with the given across-layer scatter.
    """
    rng = np.random.default_rng(0)
    results = {}
    for step in steps:
        target = value_fn(step)
        vals = target + rng.normal(scale=spread, size=n_layers)
        per_layer = [{
            "layer": f"layer_{i}",
            "henrici_relative": float(v),
            "complex_energy_fraction": float(v),
            "theta": float(v),
            "frac_repulsive_real_part": float(v),
        } for i, v in enumerate(vals)]
        results[f"pythia-410m-step{step}"] = {
            "model_stem": f"pythia-410m-step{step}",
            "checkpoint_step": step,
            "block1a": {
                "checkpoint_step": step,
                "per_layer": per_layer,
                "summary": {
                    f"{statistic}_mean": float(np.mean(vals)),
                    "complex_energy_fraction_mean": float(np.mean(vals)),
                    "henrici_relative_mean": float(np.mean(vals)),
                },
            },
            "block1b": (block1b or {}).get(step, {}),
        }
    return {
        "phase": "2b", "base": "pythia-410m",
        "n_checkpoints": len(steps),
        "missing_checkpoints": missing or [],
        "n_failed": 0,
        "steps": list(steps),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Flatness
# ---------------------------------------------------------------------------

class TestFlatnessHasANoiseScale(unittest.TestCase):
    """
    "Does the complex fraction have a developmental trajectory" is not
    answered by the range being nonzero. It is answered by the range being
    large next to how much the quantity varies across layers of a SINGLE
    checkpoint — which Block 1a supplies for free, 24 values per point.
    """

    def test_a_flat_statistic_reads_as_flat_despite_a_nonzero_range(self):
        c = fake_combined(lambda s: 0.98, spread=0.05)
        f = rep.flatness(rep.collect_trajectory(c, "complex_energy_fraction_mean"))
        self.assertGreater(f["range"], 0.0)                  # never exactly zero
        self.assertLess(f["range_excess_over_noise"], 1.5)   # but not a move

    def test_the_values_are_means_so_the_sampling_scale_is_se_not_spread(self):
        """
        `values` are means over `n_layers`, so their sampling noise is
        spread/sqrt(n_layers). Dividing a range by the raw layer spread
        understates movement by ~sqrt(24) ~ 4.9x.
        """
        c = fake_combined(lambda s: 0.5, spread=0.1, n_layers=25)
        f = rep.flatness(rep.collect_trajectory(c, "henrici_relative_mean"))
        self.assertAlmostEqual(f["median_n_layers"], 25.0)
        self.assertAlmostEqual(f["standard_error_of_mean"],
                               f["typical_across_layer_spread"] / 5.0, places=6)
        self.assertAlmostEqual(f["range_in_standard_errors"],
                               f["range_in_spreads"] * 5.0, places=4)

    def test_pure_noise_has_excess_near_one_by_construction(self):
        """
        A 21-point series of iid noise has a range of ~3.8 standard errors
        BEFORE any trend. Comparing a range against one standard error, the
        obvious move, calls almost every flat trajectory a transition.
        """
        c = fake_combined(lambda s: 0.5, spread=0.05)
        f = rep.flatness(rep.collect_trajectory(c, "henrici_relative_mean"))
        self.assertGreater(f["range_in_standard_errors"], 2.0)
        self.assertLess(f["range_excess_over_noise"], 1.6)

    def test_expected_range_matches_the_known_constants(self):
        self.assertAlmostEqual(rep.expected_range_under_noise(21), 3.78, delta=0.05)
        self.assertAlmostEqual(rep.expected_range_under_noise(27), 4.00, delta=0.05)
        self.assertLess(rep.expected_range_under_noise(5),
                        rep.expected_range_under_noise(50))

    def test_a_real_move_exceeds_the_scatter(self):
        c = fake_combined(lambda s: 0.2 if s < 512 else 0.9, spread=0.01)
        f = rep.flatness(rep.collect_trajectory(c, "complex_energy_fraction_mean"))
        self.assertGreater(f["range_in_spreads"], 10.0)

    def test_spread_comes_from_the_layers_not_from_the_trajectory(self):
        c = fake_combined(lambda s: 0.5, spread=0.2)
        t = rep.collect_trajectory(c, "henrici_relative_mean")
        self.assertEqual(t["n_layers"][0], 24)
        self.assertAlmostEqual(float(np.median(t["spread"])), 0.2, delta=0.06)

    def test_monotone_trend_is_reported(self):
        c = fake_combined(lambda s: np.log(s + 1) / 12.0, spread=0.001)
        f = rep.flatness(rep.collect_trajectory(c, "henrici_relative_mean"))
        self.assertGreater(f["monotone_rank_corr_with_log_step"], 0.95)

    def test_no_data_is_a_status(self):
        self.assertEqual(
            rep.flatness({"statistic": "x", "values": [], "spread": [],
                          "steps": []})["status"], "no_data")


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

class TestIntervalsUseLogStepWidth(unittest.TestCase):
    """
    Pythia's schedule is log-spaced then linear.
    `p1_mstate_tracking/visualization/checkpoints.py` already settled on
    `log(step+1)`; comparing 8->16 with 40000->60000 as equal-length intervals
    would rank every late interval first purely on width.
    """

    def test_early_and_late_intervals_are_not_compared_raw(self):
        c = fake_combined(lambda s: 0.0 if s < 16 else 1.0, spread=0.001)
        rows = rep.interval_deltas(rep.collect_trajectory(
            c, "henrici_relative_mean"))
        top = rows[0]
        self.assertEqual(top["span"], (8, 16))
        self.assertGreater(top["log_width"], 0.0)

    def test_ranking_is_by_spread_units(self):
        """A large raw delta on a noisy statistic must not outrank a small
        delta on a quiet one."""
        c = fake_combined(lambda s: 0.0 if s < 16 else 0.1, spread=0.0005)
        rows = rep.interval_deltas(rep.collect_trajectory(
            c, "henrici_relative_mean"))
        self.assertEqual(rows[0]["span"], (8, 16))
        self.assertGreater(abs(rows[0]["delta_in_spreads"]),
                           abs(rows[1]["delta_in_spreads"]))

    def test_one_interval_per_adjacent_pair(self):
        c = fake_combined(lambda s: 0.5)
        rows = rep.interval_deltas(rep.collect_trajectory(
            c, "henrici_relative_mean"))
        self.assertEqual(len(rows), len(STEPS) - 1)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

class TestAlignmentRefusesVerdicts(unittest.TestCase):

    def test_every_row_is_continuous_never_a_boolean(self):
        c = fake_combined(lambda s: np.log(s + 1) / 12.0, spread=0.01)
        rows = rep.align_to_transitions(rep.collect_trajectory(
            c, "henrici_relative_mean"))
        for r in rows:
            self.assertNotIn("hit", r)
            self.assertNotIn("aligned", r)
            if r["status"] == "scored":
                self.assertIn("delta_in_spreads", r)
                self.assertIn("interval_rank", r)

    def test_rank_guards_against_a_statistic_that_moves_everywhere(self):
        """
        A large delta across a dated span means little if every span has one.
        The rank is what separates "moved there" from "moves everywhere".
        """
        rng = np.random.default_rng(1)
        c = fake_combined(lambda s: float(rng.normal()), spread=0.01)
        rows = [r for r in rep.align_to_transitions(rep.collect_trajectory(
            c, "henrici_relative_mean")) if r["status"] == "scored"]
        ranks = [r["interval_rank"] for r in rows if r["interval_rank"]]
        self.assertTrue(any(rk > 3 for rk in ranks))

    def test_step_8_to_16_is_bracketed_and_scored(self):
        """status-1 open item 4: is the collapse a training event?"""
        c = fake_combined(lambda s: 0.0 if s < 16 else 1.0, spread=0.001)
        rows = {r["key"]: r for r in rep.align_to_transitions(
            rep.collect_trajectory(c, "henrici_relative_mean"))}
        r = rows["late_layer_collapse"]
        self.assertEqual(r["status"], "scored")
        self.assertEqual(r["interval_rank"], 1)
        self.assertGreater(r["delta_in_spreads"], 100)

    def test_an_unbracketed_span_says_so_rather_than_scoring_zero(self):
        c = fake_combined(lambda s: 0.5, steps=[0, 512, 143000])
        rows = {r["key"]: r for r in rep.align_to_transitions(
            rep.collect_trajectory(c, "henrici_relative_mean"))}
        self.assertEqual(rows["late_layer_collapse"]["status"], "not_bracketed")
        self.assertEqual(rows["effective_rank_peak"]["status"], "not_bracketed")

    def test_transition_table_carries_its_sources(self):
        for t in rep.KNOWN_TRANSITIONS:
            self.assertIn(t["source"], ("status-1.md", "status-2.md"))
            self.assertEqual(len(t["span"]), 2)
            self.assertLess(t["span"][0], t["span"][1])


# ---------------------------------------------------------------------------
# Co-movement
# ---------------------------------------------------------------------------

class TestCoMovement(unittest.TestCase):

    def test_external_series_can_be_compared(self):
        """Phase 2's `frac_repulsive`: 1.00 -> 0.50 -> 0.80 over ~90k steps."""
        c = fake_combined(lambda s: 1.0 if s < 7000 else 0.5, spread=0.01)
        henrici = rep.collect_trajectory(c, "henrici_relative_mean")
        frac_rep = rep.external_trajectory(
            "phase2_frac_repulsive",
            [0, 512, 7000, 19000, 40000, 100000, 120000, 143000],
            [0.5, 1.0, 1.0, 0.79, 0.58, 0.50, 0.80, 0.72],
        )
        res = rep.co_movement(henrici, frac_rep)
        self.assertEqual(res["status"], "ok")
        self.assertIn("spearman_deltas", res)
        self.assertIn("interval_agreement", res)

    def test_a_causal_reading_is_refused_in_the_output(self):
        c = fake_combined(lambda s: np.log(s + 1), spread=0.01)
        t = rep.collect_trajectory(c, "henrici_relative_mean")
        res = rep.co_movement(t, t)
        self.assertIn("caveat", res)
        self.assertIn("neither is a causal claim", res["caveat"])

    def test_insufficient_overlap_is_a_status(self):
        c = fake_combined(lambda s: 0.5)
        t = rep.collect_trajectory(c, "henrici_relative_mean")
        ext = rep.external_trajectory("x", [999999], [1.0])
        self.assertEqual(rep.co_movement(t, ext)["status"],
                         "insufficient_overlap")

    def test_external_series_carries_nan_spread_not_a_substituted_scale(self):
        ext = rep.external_trajectory("x", [0, 1], [0.0, 1.0])
        self.assertTrue(all(np.isnan(s) for s in ext["spread"]))
        self.assertTrue(np.isnan(rep.flatness(ext)["range_in_spreads"]))


# ---------------------------------------------------------------------------
# Block 1b
# ---------------------------------------------------------------------------

def fake_1b(verdict, elim_full=None, elim_signed=None, truncated=0,
            invariance="identity_holds"):
    frames = {f"f{i}": {"truncated": i < truncated} for i in range(3)}
    return {
        "interpretation": {"overall": verdict, "reference_beta": 1.0},
        "comparison": {"1.0": {
            "elim_full": {"rate": elim_full, "status": "ok"},
            "elim_signed": {"rate": elim_signed, "status": "ok"},
        }},
        "frames": frames,
        "invariance": {"status": invariance},
    }


class TestBlock1bTrajectory(unittest.TestCase):

    def test_refusals_are_counted_beside_the_verdicts(self):
        """
        A checkpoint where every run refused looks identical to one where
        every run said `both_frames_inert` if only the tally is read. Steps
        8-64 are clean on 9/9 prompts in Study B.
        """
        b1b = {
            8: {"p1": fake_1b("no_violations"), "p2": fake_1b("no_violations")},
            143000: {"p1": fake_1b("both_frames_inert", 0.0, 0.0),
                     "p2": fake_1b("signed_exceeds_full_v", 0.02, 0.95)},
        }
        t = rep.block1b_trajectory(fake_combined(lambda s: 0.5, block1b=b1b))
        by_step = {r["step"]: r for r in t["per_step"]}
        self.assertEqual(by_step[8]["n_refused"], 2)
        self.assertEqual(by_step[8]["n_runs"], 2)
        self.assertEqual(by_step[143000]["n_refused"], 0)

    def test_elimination_rates_aggregate_only_over_real_numbers(self):
        b1b = {143000: {"p1": fake_1b("no_violations"),
                        "p2": fake_1b("signed_exceeds_full_v", 0.02, 0.95)}}
        t = rep.block1b_trajectory(fake_combined(lambda s: 0.5, block1b=b1b))
        row = {r["step"]: r for r in t["per_step"]}[143000]
        self.assertEqual(row["elim_signed_n"], 1)
        self.assertAlmostEqual(row["elim_signed_mean"], 0.95)

    def test_truncated_frames_are_tallied(self):
        b1b = {512: {"p1": fake_1b("not_comparable", truncated=2)}}
        t = rep.block1b_trajectory(fake_combined(lambda s: 0.5, block1b=b1b))
        row = {r["step"]: r for r in t["per_step"]}[512]
        self.assertEqual(row["n_truncated_frames"], 2)

    def test_broken_invariance_is_surfaced(self):
        b1b = {512: {"p1": fake_1b("both_frames_inert", 0.0, 0.0,
                                   invariance="identity_broken")}}
        t = rep.block1b_trajectory(fake_combined(lambda s: 0.5, block1b=b1b))
        row = {r["step"]: r for r in t["per_step"]}[512]
        self.assertEqual(row["n_invariance_broken"], 1)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_report_is_serializable_and_finite(self):
        c = fake_combined(lambda s: np.log(s + 1) / 12.0, spread=0.01)
        rep.write_report(c, self.tmp)
        s = (self.tmp / "phase2b_report.json").read_text()
        json.loads(s)
        self.assertNotIn("NaN", s)
        self.assertNotIn("Infinity", s)

    def test_text_report_explains_both_noise_scales(self):
        c = fake_combined(lambda s: 0.98, spread=0.05)
        lines = rep.report_lines(rep.build_report(c))
        self.assertTrue(any("PURE NOISE" in l for l in lines))
        self.assertTrue(any("Below 1.0" in l for l in lines))
        self.assertTrue(any("across-LAYER scatter" in l for l in lines))

    def test_text_report_states_the_invariance_control_is_not_a_result(self):
        b1b = {512: {"p1": fake_1b("both_frames_inert", 0.0, 0.0)}}
        lines = rep.report_lines(rep.build_report(
            fake_combined(lambda s: 0.5, block1b=b1b)))
        self.assertTrue(any("not a result about rotation" in l for l in lines))

    def test_missing_checkpoints_propagate_to_the_report(self):
        c = fake_combined(lambda s: 0.5, missing=[2000])
        r = rep.build_report(c)
        self.assertEqual(r["missing_checkpoints"], [2000])
        self.assertTrue(any("2000" in l for l in rep.report_lines(r)[:4]))

    def test_load_combined_accepts_a_directory(self):
        c = fake_combined(lambda s: 0.5)
        (self.tmp / "phase2b_results.json").write_text(json.dumps(c))
        self.assertEqual(rep.load_combined(self.tmp)["base"], "pythia-410m")

    def test_every_tracked_statistic_documents_what_it_would_mean(self):
        for key, meaning in rep.TRACKED_STATISTICS.items():
            self.assertTrue(meaning and len(meaning) > 20, key)


# ---------------------------------------------------------------------------
# Rank correlation helper
# ---------------------------------------------------------------------------

class TestSpearman(unittest.TestCase):

    def test_perfect_monotone(self):
        self.assertAlmostEqual(rep._spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(rep._spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_ties_are_averaged(self):
        r = rep._rankdata(np.array([1.0, 2.0, 2.0, 3.0]))
        np.testing.assert_allclose(r, [1.0, 2.5, 2.5, 4.0])

    def test_too_few_points_is_nan(self):
        self.assertTrue(np.isnan(rep._spearman([1, 2], [1, 2])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
