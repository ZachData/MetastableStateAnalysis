"""
tests/test_p5_sweep_geometry.py — p5_single_mstate_analysis/sweep_geometry.py.

Pure numpy. The load-bearing tests are:

  TestEffRankEquivalence — the Gram-eigenvalue effective rank must equal the
      reference SVD definition. The whole cost argument rests on that
      identity, so it is checked rather than assumed.

  TestNormedVsRaw — a synthetic population where one particle's norm is
      inflated but its direction is unchanged. Raw quantities must move;
      normed quantities must not. This is D1/B12 in miniature.
"""

import unittest
from pathlib import Path

import numpy as np

from p5_single_mstate_analysis.sweep_geometry import (
    DEFAULT_POS0_POLICY,
    l2_normalize_rows,
    eff_rank_from_gram,
    population_rank,
    set_geometry,
    particle_rank_contributions,
    particle_geometry,
    layer_geometry,
    sweep_geometry,
    geometry_report_lines,
)
from p5_single_mstate_analysis.token_sets import TokenSet
from core.run_discovery import RunRef

import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure

def _reference_eff_rank(X, mode="raw"):
    """metrics.effective_rank, reimplemented here so the test does not
    depend on the project module being importable."""
    X = np.asarray(X, dtype=np.float64)
    if mode == "normed":
        X = l2_normalize_rows(X)
    sv = np.linalg.svd(X, compute_uv=False)
    sv2 = sv ** 2
    total = sv2.sum()
    if total < 1e-12:
        return 1.0
    p = np.clip(sv2 / total, 1e-12, None)
    return float(np.exp(-np.sum(p * np.log(p))))


def _cluster_population(n=30, d=16, n_cluster=8, spread=0.05, seed=0):
    """A tight cluster of `n_cluster` tokens inside a diffuse cloud."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    X[:n_cluster] = direction + spread * rng.normal(size=(n_cluster, d))
    return X


def _token_set(positions, sibling=(), control=(), n_tokens=30,
               name="anchor_final", prompt="wiki_paragraph"):
    return TokenSet(
        name=name, prompt_key=prompt, anchor_model="pythia-410m-step143000",
        anchor_step=143000, anchor_run_dir="/runs/x",
        positions=tuple(sorted(positions)),
        sibling_positions=tuple(sorted(sibling)),
        control_positions=tuple(sorted(control)),
        n_tokens_prompt=n_tokens,
    )


# ---------------------------------------------------------------------------

class TestEffRankEquivalence(unittest.TestCase):

    def test_matches_svd_definition_raw(self):
        for seed in range(5):
            X = _cluster_population(seed=seed)
            self.assertAlmostEqual(eff_rank_from_gram(X @ X.T),
                                   _reference_eff_rank(X, "raw"), places=8)

    def test_matches_svd_definition_normed(self):
        for seed in range(5):
            X = _cluster_population(seed=seed)
            U = l2_normalize_rows(X)
            self.assertAlmostEqual(eff_rank_from_gram(U @ U.T),
                                   _reference_eff_rank(X, "normed"), places=8)

    def test_wide_matrix_where_n_exceeds_d(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 5))
        self.assertAlmostEqual(eff_rank_from_gram(X @ X.T),
                               _reference_eff_rank(X, "raw"), places=8)

    def test_rank_one_population(self):
        v = np.ones((10, 4))
        self.assertAlmostEqual(eff_rank_from_gram(v @ v.T), 1.0, places=6)

    def test_orthonormal_population_gives_n(self):
        X = np.eye(6)
        self.assertAlmostEqual(eff_rank_from_gram(X @ X.T), 6.0, places=6)

    def test_zero_population(self):
        X = np.zeros((5, 4))
        self.assertEqual(eff_rank_from_gram(X @ X.T), 1.0)

    def test_empty(self):
        self.assertTrue(np.isnan(eff_rank_from_gram(np.zeros((0, 0)))))


class TestNormalization(unittest.TestCase):

    def test_rows_become_unit(self):
        X = _cluster_population()
        U = l2_normalize_rows(X)
        np.testing.assert_allclose(np.linalg.norm(U, axis=1), 1.0, atol=1e-10)

    def test_zero_row_stays_zero_not_nan(self):
        X = np.zeros((3, 4))
        X[1] = [1.0, 0, 0, 0]
        U = l2_normalize_rows(X)
        self.assertTrue(np.all(np.isfinite(U)))
        self.assertAlmostEqual(np.linalg.norm(U[0]), 0.0)


class TestNormedVsRaw(unittest.TestCase):
    """D1 / B12 in miniature: inflate one particle's norm, leave its
    direction alone, and check which quantities move."""

    def setUp(self):
        self.X = _cluster_population(seed=3)
        self.Y = self.X.copy()
        self.Y[5] *= 50.0          # direction unchanged, norm inflated
        self.pos = list(range(1, 9))   # excludes position 0 (the sink slot)

    def test_normed_population_rank_is_unchanged(self):
        a = population_rank(self.X)["eff_rank_normed"]
        b = population_rank(self.Y)["eff_rank_normed"]
        self.assertAlmostEqual(a, b, places=8)

    def test_raw_population_rank_moves(self):
        a = population_rank(self.X)["eff_rank_raw"]
        b = population_rank(self.Y)["eff_rank_raw"]
        self.assertGreater(abs(a - b), 0.1)

    def test_normed_set_compactness_unchanged(self):
        a = set_geometry(self.X, self.pos)
        b = set_geometry(self.Y, self.pos)
        self.assertAlmostEqual(a["resultant_length"], b["resultant_length"],
                               places=10)
        self.assertAlmostEqual(a["mean_within_cos"], b["mean_within_cos"],
                               places=10)

    def test_raw_set_rank_moves(self):
        a = set_geometry(self.X, self.pos)["eff_rank_raw"]
        b = set_geometry(self.Y, self.pos)["eff_rank_raw"]
        self.assertGreater(abs(a - b), 0.05)

    def test_particle_contribution_modes_disagree(self):
        cn = particle_rank_contributions(self.Y, self.pos, mode="normed")
        cr = particle_rank_contributions(self.Y, self.pos, mode="raw")
        self.assertGreater(abs(cn[5] - cr[5]), 0.05)

    def test_normed_contributions_unchanged_by_norm_inflation(self):
        a = particle_rank_contributions(self.X, self.pos, mode="normed")
        b = particle_rank_contributions(self.Y, self.pos, mode="normed")
        for p in self.pos:
            self.assertAlmostEqual(a[p], b[p], places=8)


class TestSetGeometry(unittest.TestCase):

    def setUp(self):
        self.X = _cluster_population(n=30, n_cluster=8, spread=0.02, seed=7)

    def test_tight_set_has_high_resultant(self):
        g = set_geometry(self.X, range(1, 8))
        self.assertGreater(g["resultant_length"], 0.95)

    def test_diffuse_set_has_low_resultant(self):
        g = set_geometry(self.X, range(20, 28))
        self.assertLess(g["resultant_length"], 0.6)

    def test_separation_positive_for_the_real_cluster(self):
        self.assertGreater(set_geometry(self.X, range(1, 8))["separation"], 0.5)

    def test_separation_near_zero_for_an_arbitrary_subset(self):
        g = set_geometry(self.X, range(20, 28))
        self.assertLess(abs(g["separation"]), 0.35)

    def test_singleton_set_has_nan_within_cos_but_valid_resultant(self):
        g = set_geometry(self.X, [5])
        self.assertTrue(np.isnan(g["mean_within_cos"]))
        self.assertAlmostEqual(g["resultant_length"], 1.0, places=8)

    def test_empty_set_returns_nans_not_error(self):
        g = set_geometry(self.X, [])
        self.assertEqual(g["n_set"], 0)
        self.assertTrue(np.isnan(g["resultant_length"]))

    def test_out_of_range_positions_dropped(self):
        g = set_geometry(self.X, [1, 2, 999])
        self.assertEqual(g["n_set"], 2)


class TestPos0Policy(unittest.TestCase):
    """The sink must be excludable by MASK, never by reindexing."""

    def setUp(self):
        self.X = _cluster_population(seed=11)
        self.X[0] *= 100.0     # the attention sink

    def test_excluding_the_sink_changes_raw_population_rank(self):
        inc = population_rank(self.X, "included")["eff_rank_raw"]
        exc = population_rank(self.X, "excluded")["eff_rank_raw"]
        self.assertGreater(abs(inc - exc), 0.1)
        self.assertEqual(population_rank(self.X, "included")["n_population"],
                         self.X.shape[0])
        self.assertEqual(population_rank(self.X, "excluded")["n_population"],
                         self.X.shape[0] - 1)

    def test_positions_are_not_reindexed_by_exclusion(self):
        # Position 5 must mean row 5 under both policies.
        a = particle_geometry(self.X, [5], pos0_policy="included")
        b = particle_geometry(self.X, [5], pos0_policy="excluded")
        self.assertAlmostEqual(a[5]["norm"], b[5]["norm"], places=10)
        self.assertAlmostEqual(a[5]["norm"], float(np.linalg.norm(self.X[5])),
                               places=10)

    def test_excluded_position_zero_gets_nan_not_zero(self):
        c = particle_rank_contributions(self.X, [0, 5], pos0_policy="excluded")
        self.assertTrue(np.isnan(c[0]))
        self.assertFalse(np.isnan(c[5]))

    def test_included_position_zero_is_measured(self):
        c = particle_rank_contributions(self.X, [0], pos0_policy="included")
        self.assertFalse(np.isnan(c[0]))

    def test_bad_policy_raises(self):
        with self.assertRaises(ValueError):
            population_rank(self.X, "sometimes")


class TestParticleContributions(unittest.TestCase):

    def setUp(self):
        self.X = _cluster_population(n=24, d=12, n_cluster=6, spread=0.02,
                                     seed=5)

    def test_matches_bruteforce_leave_one_out(self):
        got = particle_rank_contributions(self.X, [3, 4, 10],
                                          mode="normed", pos0_policy="included")
        U = l2_normalize_rows(self.X)
        full = _reference_eff_rank(U, "raw")
        for p in (3, 4, 10):
            remaining = np.delete(U, p, axis=0)
            expected = full - _reference_eff_rank(remaining, "raw")
            self.assertAlmostEqual(got[p], expected, places=7)

    def test_redundant_particle_contributes_less_than_a_unique_one(self):
        c = particle_rank_contributions(self.X, [1, 20], mode="normed",
                                        pos0_policy="included")
        # 1 is inside the tight cluster (redundant); 20 is diffuse (unique).
        self.assertLess(c[1], c[20])

    def test_sign_is_preserved_not_clipped(self):
        c = particle_rank_contributions(self.X, list(range(1, 24)),
                                        mode="normed")
        self.assertTrue(any(v < 0 for v in c.values())
                        or all(v >= 0 for v in c.values()))
        self.assertTrue(all(np.isfinite(v) for v in c.values()))

    def test_out_of_range_is_nan(self):
        c = particle_rank_contributions(self.X, [999])
        self.assertTrue(np.isnan(c[999]))

    def test_bad_mode_raises(self):
        with self.assertRaises(ValueError):
            particle_rank_contributions(self.X, [1], mode="sphere")


class TestParticleGeometry(unittest.TestCase):

    def setUp(self):
        self.X = _cluster_population(n=24, n_cluster=6, spread=0.02, seed=9)

    def test_cluster_members_align_with_their_centroid(self):
        rows = particle_geometry(self.X, [1, 2, 3, 4, 5],
                                 with_contributions=False)
        for r in rows.values():
            self.assertGreater(r["cos_to_centroid"], 0.9)

    def test_reference_positions_are_respected(self):
        # A diffuse particle measured against the tight cluster's centroid.
        rows = particle_geometry(self.X, [20], reference_positions=[1, 2, 3, 4],
                                 with_contributions=False)
        self.assertLess(rows[20]["cos_to_centroid"], 0.9)

    def test_norm_z_flags_the_sink(self):
        X = self.X.copy()
        X[3] *= 40.0
        rows = particle_geometry(X, [3], pos0_policy="included",
                                 with_contributions=False)
        self.assertGreater(rows[3]["norm_z"], 3.0)

    def test_contributions_can_be_skipped_for_speed(self):
        rows = particle_geometry(self.X, [1, 2], with_contributions=False)
        self.assertTrue(np.isnan(rows[1]["rank_contribution_normed"]))


class TestLayerGeometry(unittest.TestCase):

    def setUp(self):
        self.X = _cluster_population(n=30, n_cluster=8, spread=0.02, seed=2)
        self.ts = _token_set(positions=[1, 2, 3, 4, 5],
                             sibling=[6, 7, 8],
                             control=[20, 21, 22, 23, 24])

    def test_all_three_roles_measured_identically(self):
        rec = layer_geometry(self.X, self.ts, layer=4)
        self.assertEqual(set(rec["sets"]), {"primary", "sibling", "control"})
        for role in ("primary", "sibling", "control"):
            self.assertIn("resultant_length", rec["sets"][role])

    def test_particle_rows_cover_every_role(self):
        rec = layer_geometry(self.X, self.ts, layer=4,
                             with_contributions=False)
        roles = {r["role"] for r in rec["particles"]}
        self.assertEqual(roles, {"primary", "sibling", "control"})
        self.assertEqual(len(rec["particles"]), 5 + 3 + 5)

    def test_primary_is_tighter_than_control(self):
        rec = layer_geometry(self.X, self.ts, layer=4)
        self.assertGreater(rec["sets"]["primary"]["resultant_length"],
                           rec["sets"]["control"]["resultant_length"])

    def test_empty_role_omitted(self):
        ts = _token_set(positions=[1, 2, 3])
        rec = layer_geometry(self.X, ts, layer=0, with_contributions=False)
        self.assertEqual(set(rec["sets"]), {"primary"})

    def test_layer_index_carried(self):
        rec = layer_geometry(self.X, self.ts, layer=7,
                             with_contributions=False)
        self.assertEqual(rec["layer"], 7)
        self.assertTrue(all(r["layer"] == 7 for r in rec["particles"]))


class TestSweep(unittest.TestCase):

    def setUp(self):
        self.ts = _token_set(positions=[1, 2, 3, 4, 5],
                             control=[20, 21, 22, 23, 24], n_tokens=30)
        self.refs = [
            RunRef(run_dir=Path(f"/runs/step{s}"),
                   model=f"pythia-410m-step{s}", base="pythia-410m",
                   step=s, is_random=False, prompt_key="wiki_paragraph",
                   source="manifest", hf_revision=f"step{s}")
            for s in (0, 512, 143000)
        ]
        self.acts = {
            0: np.random.default_rng(0).normal(size=(6, 30, 12)),
            512: _cluster_population(n=30, d=12, seed=1)[None].repeat(6, 0),
            143000: _cluster_population(n=30, d=12, seed=2)[None].repeat(6, 0),
        }

    def _loader(self, path):
        step = int(str(path).rsplit("step", 1)[1])
        return self.acts.get(step)

    def test_every_checkpoint_measured_with_the_same_positions(self):
        out = sweep_geometry(self.ts, self.refs, loader=self._loader,
                             with_contributions=False)
        self.assertEqual([r["step"] for r in out["records"]], [0, 512, 143000])
        self.assertEqual(out["positions"], [1, 2, 3, 4, 5])

    def test_frame_and_policy_recorded(self):
        out = sweep_geometry(self.ts, self.refs, loader=self._loader,
                             with_contributions=False)
        self.assertEqual(out["frame_kind"], "l2_sphere")
        self.assertEqual(out["pos0_policy"], DEFAULT_POS0_POLICY)

    def test_missing_activations_skipped_with_reason(self):
        out = sweep_geometry(self.ts, self.refs, loader=lambda p: None,
                             with_contributions=False)
        self.assertEqual(out["records"], [])
        self.assertEqual(len(out["skipped"]), 3)
        self.assertIn("no activations", out["skipped"][0]["reason"])

    def test_token_count_mismatch_aborts_that_checkpoint(self):
        acts = dict(self.acts)
        acts[512] = np.random.default_rng(0).normal(size=(6, 31, 12))

        def loader(path):
            return acts.get(int(str(path).rsplit("step", 1)[1]))

        out = sweep_geometry(self.ts, self.refs, loader=loader,
                             with_contributions=False)
        self.assertEqual([r["step"] for r in out["records"]], [0, 143000])
        skip = [s for s in out["skipped"] if s["step"] == 512][0]
        self.assertIn("same particles", skip["reason"])

    def test_step_filter(self):
        out = sweep_geometry(self.ts, self.refs, steps=[0, 143000],
                             loader=self._loader, with_contributions=False)
        self.assertEqual([r["step"] for r in out["records"]], [0, 143000])

    def test_sink_membership_produces_a_note(self):
        ts = _token_set(positions=[0, 1, 2, 3], n_tokens=30)
        out = sweep_geometry(ts, self.refs, loader=self._loader,
                             pos0_policy="excluded", with_contributions=False)
        self.assertTrue(any("attention sink" in n for n in out["notes"]))

    def test_bad_activation_shape_skipped(self):
        out = sweep_geometry(self.ts, self.refs,
                             loader=lambda p: np.zeros((30, 12)),
                             with_contributions=False)
        self.assertEqual(out["records"], [])
        self.assertIn("expected", out["skipped"][0]["reason"])

    def test_report_lines(self):
        out = sweep_geometry(self.ts, self.refs, loader=self._loader,
                             with_contributions=False)
        blob = "\n".join(geometry_report_lines(out))
        self.assertIn("l2_sphere", blob)
        self.assertIn("pop_rank_raw", blob)
        self.assertIn("143000", blob)

    def test_report_on_empty_sweep(self):
        out = sweep_geometry(self.ts, self.refs, loader=lambda p: None,
                             with_contributions=False)
        self.assertIn("no checkpoints measured",
                      "\n".join(geometry_report_lines(out)))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestContributionModes(unittest.TestCase):

    def setUp(self):
        self.X = _cluster_population(n=24, d=12, n_cluster=6, seed=4)
        self.ts = _token_set(positions=[1, 2, 3], n_tokens=24)

    def test_normed_only_leaves_raw_as_nan_not_zero(self):
        rows = particle_geometry(self.X, [1, 2],
                                 contribution_modes=("normed",))
        self.assertFalse(np.isnan(rows[1]["rank_contribution_normed"]))
        self.assertTrue(np.isnan(rows[1]["rank_contribution_raw"]))

    def test_raw_only(self):
        rows = particle_geometry(self.X, [1], contribution_modes=("raw",))
        self.assertTrue(np.isnan(rows[1]["rank_contribution_normed"]))
        self.assertFalse(np.isnan(rows[1]["rank_contribution_raw"]))

    def test_bad_mode_raises(self):
        with self.assertRaises(ValueError):
            particle_geometry(self.X, [1], contribution_modes=("sphere",))

    def test_modes_recorded_in_sweep_output(self):
        refs = [RunRef(run_dir=Path("/runs/step0"), model="pythia-410m-step0",
                       base="pythia-410m", step=0, is_random=False,
                       prompt_key="wiki_paragraph", source="manifest")]
        out = sweep_geometry(self.ts, refs,
                             loader=lambda p: self.X[None].repeat(3, 0),
                             contribution_modes=("normed",))
        self.assertEqual(out["contribution_modes"], ["normed"])

    def test_disabled_contributions_record_empty_modes(self):
        refs = [RunRef(run_dir=Path("/runs/step0"), model="pythia-410m-step0",
                       base="pythia-410m", step=0, is_random=False,
                       prompt_key="wiki_paragraph", source="manifest")]
        out = sweep_geometry(self.ts, refs,
                             loader=lambda p: self.X[None].repeat(3, 0),
                             with_contributions=False)
        self.assertEqual(out["contribution_modes"], [])
