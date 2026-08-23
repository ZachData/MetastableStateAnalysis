"""
tests/test_p5_tiers.py — p5_single_mstate_analysis/tiers.py.

Uses the real `core/nulls.py` (vendored into the sandbox), not a stub, so the
Nσ path is exercised end to end.

Load-bearing tests:

  TestNullSeparatesRealFromArbitrary — a planted cluster must clear the
      permutation null; an arbitrary size-matched subset of the same
      population must not. If both cleared, the null would be measuring
      nothing.

  TestPos0InNull — the sink must not be drawable into a null set, and the
      sphere-frame metrics must be *invariant* to its norm. The second half
      is the one that corrected the module: an earlier draft claimed norm
      inflation widened the null, and it does not, because these metrics
      L2-normalize first. What the policy actually protects is the raw-mode
      statistics and the definition of what the null is a null of.

  TestOrderingBookkeeping — "the claim failed" and "the claim could not be
      evaluated" must not collapse to the same value.
"""

import unittest

import numpy as np

from p5_single_mstate_analysis.tiers import (
    TIERS,
    resultant_of_labelled,
    separation_of_labelled,
    tier_contrast,
    tier_nulls,
    population_structure_null,
    ordering_consistency,
    sweep_tier_records,
    falsification_table_lines,
)
from p5_single_mstate_analysis.token_sets import TokenSet

import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure

def _population(n=40, d=16, seed=0, tight=(0, 8), tight_spread=0.03,
                loose=(8, 16), loose_spread=0.45):
    """A tight planted cluster, a looser one, and a diffuse remainder."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    for (lo, hi), spread in ((tight, tight_spread), (loose, loose_spread)):
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction)
        X[lo:hi] = direction + spread * rng.normal(size=(hi - lo, d))
    return X


def _ts(primary, sibling=(), control=(), n_tokens=40):
    return TokenSet(
        name="anchor_final", prompt_key="wiki_paragraph",
        anchor_model="pythia-410m-step143000", anchor_step=143000,
        anchor_run_dir="/runs/x",
        positions=tuple(sorted(primary)),
        sibling_positions=tuple(sorted(sibling)),
        control_positions=tuple(sorted(control)),
        n_tokens_prompt=n_tokens,
    )


# ---------------------------------------------------------------------------

class TestMetricForms(unittest.TestCase):

    def setUp(self):
        self.X = _population(seed=1)

    def test_resultant_matches_set_geometry(self):
        from p5_single_mstate_analysis.sweep_geometry import set_geometry
        pos = list(range(1, 8))
        labels = np.zeros(self.X.shape[0], dtype=int)
        labels[pos] = 1
        self.assertAlmostEqual(
            resultant_of_labelled(self.X, labels),
            set_geometry(self.X, pos, pos0_policy="included")["resultant_length"],
            places=10)

    def test_resultant_bounded(self):
        labels = np.zeros(40, dtype=int)
        labels[:8] = 1
        v = resultant_of_labelled(self.X, labels)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_empty_set_is_nan(self):
        self.assertTrue(np.isnan(
            resultant_of_labelled(self.X, np.zeros(40, dtype=int))))

    def test_separation_nan_for_singleton(self):
        labels = np.zeros(40, dtype=int)
        labels[3] = 1
        self.assertTrue(np.isnan(separation_of_labelled(self.X, labels)))

    def test_separation_nan_when_no_complement(self):
        self.assertTrue(np.isnan(
            separation_of_labelled(self.X, np.ones(40, dtype=int))))

    def test_separation_positive_for_planted_cluster(self):
        labels = np.zeros(40, dtype=int)
        labels[1:8] = 1
        self.assertGreater(separation_of_labelled(self.X, labels), 0.5)


class TestNullSeparatesRealFromArbitrary(unittest.TestCase):

    def setUp(self):
        self.X = _population(seed=2)
        self.rng = lambda: np.random.default_rng(11)

    def test_planted_cluster_clears_the_null(self):
        out = tier_nulls(self.X, range(1, 8), pos0_policy="included",
                         n_permutations=200, rng=self.rng())
        self.assertTrue(out["resultant_length"]["significant"])
        self.assertGreater(out["resultant_length"]["z_score"], 2.0)

    def test_arbitrary_subset_does_not(self):
        out = tier_nulls(self.X, range(25, 32), pos0_policy="included",
                         n_permutations=200, rng=self.rng())
        self.assertFalse(out["resultant_length"]["significant"])

    def test_separation_also_discriminates(self):
        real = tier_nulls(self.X, range(1, 8), metrics=("separation",),
                          pos0_policy="included", n_permutations=200,
                          rng=self.rng())
        arb = tier_nulls(self.X, range(25, 32), metrics=("separation",),
                         pos0_policy="included", n_permutations=200,
                         rng=self.rng())
        self.assertGreater(real["separation"]["z_score"],
                           arb["separation"]["z_score"])

    def test_verdict_string_is_reportable(self):
        out = tier_nulls(self.X, range(1, 8), pos0_policy="included",
                         n_permutations=100, rng=self.rng())
        self.assertIn("σ from null", out["resultant_length"]["verdict_str"])

    def test_tiny_set_is_not_evaluable_rather_than_false(self):
        out = tier_nulls(self.X, [3], pos0_policy="included")
        self.assertIn("not evaluable", out["resultant_length"]["verdict_str"])
        self.assertEqual(out["resultant_length"]["n_null"], 0)

    def test_pool_exhausted_is_not_evaluable(self):
        out = tier_nulls(self.X, range(40), pos0_policy="included")
        self.assertIn("not evaluable", out["resultant_length"]["verdict_str"])


class TestPos0InNull(unittest.TestCase):

    def setUp(self):
        self.X = _population(seed=3)
        self.X[0] *= 100.0        # the attention sink

    def test_sphere_metrics_are_invariant_to_the_sinks_norm(self):
        # The correction: L2-normalized metrics do not care how big the sink
        # is. Any argument for excluding it that rests on norm inflation is
        # wrong for these statistics.
        plain = _population(seed=3)
        spiked = plain.copy()
        spiked[0] *= 100.0
        a = tier_nulls(plain, range(1, 8), pos0_policy="included",
                       n_permutations=200, rng=np.random.default_rng(5))
        b = tier_nulls(spiked, range(1, 8), pos0_policy="included",
                       n_permutations=200, rng=np.random.default_rng(5))
        self.assertAlmostEqual(a["resultant_length"]["observed"],
                               b["resultant_length"]["observed"], places=12)
        self.assertAlmostEqual(a["resultant_length"]["null_std"],
                               b["resultant_length"]["null_std"], places=10)

    def test_raw_population_null_IS_sink_sensitive(self):
        # Where the policy actually earns its keep.
        inc = population_structure_null(self.X, n_shuffles=150,
                                        pos0_policy="included",
                                        rng=np.random.default_rng(5))
        exc = population_structure_null(self.X, n_shuffles=150,
                                        pos0_policy="excluded",
                                        rng=np.random.default_rng(5))
        self.assertNotAlmostEqual(inc["observed"], exc["observed"], places=6)

    def test_effect_survives_exclusion(self):
        exc = tier_nulls(self.X, range(1, 8), pos0_policy="excluded",
                         n_permutations=300, rng=np.random.default_rng(5))
        self.assertTrue(exc["resultant_length"]["significant"])

    def test_sink_member_is_dropped_from_the_observed_set_too(self):
        out = tier_nulls(self.X, [0, 1, 2, 3, 4], pos0_policy="excluded",
                         n_permutations=50, rng=np.random.default_rng(1))
        self.assertEqual(out["resultant_length"]["n_set_in_pool"], 4)
        self.assertEqual(out["resultant_length"]["n_dropped_by_pos0"], 1)

    def test_no_drop_when_included(self):
        out = tier_nulls(self.X, [0, 1, 2, 3, 4], pos0_policy="included",
                         n_permutations=50, rng=np.random.default_rng(1))
        self.assertEqual(out["resultant_length"]["n_set_in_pool"], 5)


class TestPopulationStructureNull(unittest.TestCase):

    def test_structured_population_is_significant(self):
        X = _population(seed=4)
        out = population_structure_null(X, n_shuffles=150,
                                        pos0_policy="included",
                                        rng=np.random.default_rng(3))
        self.assertTrue(np.isfinite(out["z_score"]))
        self.assertTrue(out["significant"])

    def test_isotropic_population_is_not(self):
        X = np.random.default_rng(9).normal(size=(40, 16))
        out = population_structure_null(X, n_shuffles=150,
                                        pos0_policy="included",
                                        rng=np.random.default_rng(3))
        self.assertFalse(out["significant"])

    def test_too_few_tokens_is_not_evaluable(self):
        out = population_structure_null(np.zeros((2, 5)))
        self.assertIn("not evaluable", out["verdict_str"])


class TestTierContrast(unittest.TestCase):

    def setUp(self):
        self.X = _population(seed=6)
        self.ts = _ts(primary=range(1, 8), sibling=range(8, 16),
                      control=range(25, 32))

    def test_all_tiers_present(self):
        out = tier_contrast(self.X, self.ts, pos0_policy="included")
        self.assertEqual(set(out["tiers"]), set(TIERS))

    def test_ordering_holds_on_planted_data(self):
        out = tier_contrast(self.X, self.ts, pos0_policy="included")
        self.assertTrue(out["ordering_holds"])

    def test_gaps_reported(self):
        out = tier_contrast(self.X, self.ts, pos0_policy="included")
        self.assertIn("primary_minus_sibling", out["gaps"])
        self.assertGreater(out["gaps"]["primary_minus_control"], 0)

    def test_missing_sibling_gives_partial_two_tier_claim(self):
        ts = _ts(primary=range(1, 8), control=range(25, 32))
        out = tier_contrast(self.X, ts, pos0_policy="included")
        self.assertTrue(out["ordering_holds"])
        self.assertIn("partial", out)

    def test_missing_control_gives_none_not_false(self):
        ts = _ts(primary=range(1, 8), sibling=range(8, 16))
        out = tier_contrast(self.X, ts, pos0_policy="included")
        self.assertIsNone(out["ordering_holds"])

    def test_ordering_fails_when_control_is_the_tight_cluster(self):
        ts = _ts(primary=range(25, 32), sibling=range(8, 16),
                 control=range(1, 8))
        out = tier_contrast(self.X, ts, pos0_policy="included")
        self.assertFalse(out["ordering_holds"])


class TestOrderingBookkeeping(unittest.TestCase):

    def _rec(self, holds, step=0, layer=0):
        return {"step": step, "layer": layer,
                "contrast": {"ordering_holds": holds, "gaps": {"g": 0.1}}}

    def test_not_evaluable_excluded_from_the_fraction(self):
        recs = [self._rec(True), self._rec(True), self._rec(None)]
        o = ordering_consistency(recs)
        self.assertEqual(o["n_evaluable"], 2)
        self.assertEqual(o["n_not_evaluable"], 1)
        self.assertEqual(o["fraction_held"], 1.0)
        self.assertEqual(o["n_cells"], 3)

    def test_failures_listed_with_gaps(self):
        o = ordering_consistency([self._rec(True), self._rec(False, step=512)])
        self.assertEqual(o["fraction_held"], 0.5)
        self.assertEqual(o["failures"][0]["step"], 512)
        self.assertIn("g", o["failures"][0]["gaps"])

    def test_all_unevaluable_gives_none_not_zero(self):
        o = ordering_consistency([self._rec(None), self._rec(None)])
        self.assertIsNone(o["fraction_held"])
        self.assertEqual(o["n_held"], 0)

    def test_empty(self):
        o = ordering_consistency([])
        self.assertEqual(o["n_cells"], 0)
        self.assertIsNone(o["fraction_held"])


class TestSweepTierRecords(unittest.TestCase):

    def setUp(self):
        self.ts = _ts(primary=range(1, 8), sibling=range(8, 16),
                      control=range(25, 32))
        self.acts = {
            0: np.random.default_rng(0).normal(size=(4, 40, 16)),
            512: _population(seed=7)[None].repeat(4, 0),
            143000: _population(seed=8)[None].repeat(4, 0),
        }

    def test_one_record_per_checkpoint_at_the_default_layer(self):
        out = sweep_tier_records(self.ts, self.acts, n_permutations=50,
                                 pos0_policy="included")
        self.assertEqual([r["step"] for r in out["records"]],
                         [0, 512, 143000])
        self.assertEqual(out["layers"], [3])

    def test_explicit_layers(self):
        out = sweep_tier_records(self.ts, self.acts, layers=[0, 2],
                                 n_permutations=30, pos0_policy="included")
        self.assertEqual(len(out["records"]), 6)

    def test_nulls_present_for_every_tier(self):
        out = sweep_tier_records(self.ts, self.acts, n_permutations=30,
                                 pos0_policy="included")
        self.assertEqual(set(out["records"][0]["nulls"]), set(TIERS))

    def test_ordering_holds_on_structured_checkpoints_not_isotropic_one(self):
        out = sweep_tier_records(self.ts, self.acts, n_permutations=50,
                                 pos0_policy="included")
        by_step = {r["step"]: r["contrast"]["ordering_holds"]
                   for r in out["records"]}
        self.assertTrue(by_step[512])
        self.assertTrue(by_step[143000])

    def test_reproducible(self):
        a = sweep_tier_records(self.ts, self.acts, n_permutations=40,
                               pos0_policy="included", seed=3)
        b = sweep_tier_records(self.ts, self.acts, n_permutations=40,
                               pos0_policy="included", seed=3)
        za = [r["nulls"]["primary"]["resultant_length"]["z_score"]
              for r in a["records"]]
        zb = [r["nulls"]["primary"]["resultant_length"]["z_score"]
              for r in b["records"]]
        np.testing.assert_allclose(za, zb)

    def test_different_seeds_give_different_nulls(self):
        a = sweep_tier_records(self.ts, self.acts, n_permutations=40,
                               pos0_policy="included", seed=1)
        b = sweep_tier_records(self.ts, self.acts, n_permutations=40,
                               pos0_policy="included", seed=2)
        za = a["records"][0]["nulls"]["primary"]["resultant_length"]["null_mean"]
        zb = b["records"][0]["nulls"]["primary"]["resultant_length"]["null_mean"]
        self.assertNotAlmostEqual(za, zb, places=10)

    def test_token_count_mismatch_skipped_with_reason(self):
        acts = dict(self.acts)
        acts[512] = np.random.default_rng(0).normal(size=(4, 41, 16))
        out = sweep_tier_records(self.ts, acts, n_permutations=20,
                                 pos0_policy="included")
        self.assertEqual([r["step"] for r in out["records"]], [0, 143000])
        self.assertTrue(any("same particles" in s for s in out["skipped"]))

    def test_layer_beyond_range_skipped(self):
        out = sweep_tier_records(self.ts, self.acts, layers=[99],
                                 n_permutations=20, pos0_policy="included")
        self.assertEqual(out["records"], [])
        self.assertTrue(any("beyond" in s for s in out["skipped"]))

    def test_no_activations(self):
        out = sweep_tier_records(self.ts, {}, n_permutations=10)
        self.assertEqual(out["records"], [])


class TestFalsificationTable(unittest.TestCase):

    def setUp(self):
        self.ts = _ts(primary=range(1, 8), sibling=range(8, 16),
                      control=range(25, 32))
        self.acts = {512: _population(seed=7)[None].repeat(3, 0),
                     143000: _population(seed=8)[None].repeat(3, 0)}

    def test_table_has_a_row_per_cell_and_a_summary(self):
        out = sweep_tier_records(self.ts, self.acts, n_permutations=50,
                                 pos0_policy="included")
        lines = falsification_table_lines(out)
        blob = "\n".join(lines)
        self.assertIn("Group G falsification", blob)
        self.assertIn("primary_z", blob)
        self.assertIn("512", blob)
        self.assertIn("ordering held", blob)

    def test_frame_and_policy_in_the_header(self):
        out = sweep_tier_records(self.ts, self.acts, n_permutations=20,
                                 pos0_policy="excluded")
        header = falsification_table_lines(out)[0]
        self.assertIn("l2_sphere", header)
        self.assertIn("pos0=excluded", header)

    def test_na_rows_are_printed_not_dropped(self):
        ts = _ts(primary=range(1, 8), sibling=range(8, 16))  # no control
        out = sweep_tier_records(ts, self.acts, n_permutations=20,
                                 pos0_policy="included")
        blob = "\n".join(falsification_table_lines(out))
        self.assertIn("n/a", blob)

    def test_empty_sweep_reports_skips(self):
        out = sweep_tier_records(self.ts, {}, n_permutations=10)
        blob = "\n".join(falsification_table_lines(out))
        self.assertIn("no cells evaluated", blob)


if __name__ == "__main__":
    unittest.main(verbosity=2)
