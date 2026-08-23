"""
tests/test_p5_particle_join.py — p5_single_mstate_analysis/particle_join.py.

Uses the real `core/particles.py` (vendored into the sandbox), not a stub.

Load-bearing tests:

  TestTurnoverDiscriminates — two synthetic sweeps, one where the SAME
      particles cluster at every checkpoint but for fewer layers each time,
      one where DIFFERENT particles cluster at late checkpoints. Both produce
      the same falling mean lifespan. The decomposition must separate them,
      or it is not answering status-1's question.

  TestComplementRetained — tokens in no role and in no cluster must still
      have rows. That population is what design-5c is about.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.particles import ParticleTable
from core.run_discovery import RunRef
from p5_single_mstate_analysis.particle_join import (
    GEOMETRY_COLUMNS,
    build_layer_table,
    build_sweep_particle_table,
    particle_biography,
    clustered_set_overlap,
    turnover_decomposition,
    biography_report_lines,
    turnover_report_lines,
)
from p5_single_mstate_analysis.token_sets import TokenSet

import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure
N_TOKENS = 20
N_LAYERS = 10


def _ts(primary=(1, 2, 3, 4), sibling=(5, 6), control=(10, 11, 12, 13),
        n_tokens=N_TOKENS):
    return TokenSet(
        name="anchor_final", prompt_key="wiki_paragraph",
        anchor_model="pythia-410m-step143000", anchor_step=143000,
        anchor_run_dir="/runs/x",
        positions=tuple(sorted(primary)),
        sibling_positions=tuple(sorted(sibling)),
        control_positions=tuple(sorted(control)),
        n_tokens_prompt=n_tokens,
    )


def _ref(step):
    return RunRef(run_dir=Path(f"/runs/step{step}"),
                  model=f"pythia-410m-step{step}", base="pythia-410m",
                  step=step, is_random=False, prompt_key="wiki_paragraph",
                  source="manifest", hf_revision=f"step{step}")


def _labels_for(clustered_positions, n_layers_each, n_tokens=N_TOKENS,
                n_layers=N_LAYERS, label_stride=1):
    """Label arrays where each listed position is clustered for
    `n_layers_each` layers starting at layer 1, cycling through
    `label_stride` distinct labels."""
    labels = [np.full(n_tokens, -1, dtype=np.int64) for _ in range(n_layers)]
    for p in clustered_positions:
        for k in range(n_layers_each):
            li = 1 + k
            if li < n_layers:
                labels[li][p] = (k // max(1, n_layers_each // label_stride)) \
                    if label_stride > 1 else 0
    return labels


def _loader_from(labels_by_step, tokens=None):
    def loader(path):
        step = int(str(path).rsplit("step", 1)[1])
        labels = labels_by_step.get(step)
        if labels is None:
            return {}
        return {
            "hdbscan_labels": labels,
            "n_tokens": int(len(labels[0])),
            "tokens": tokens or [f"t{i}" for i in range(len(labels[0]))],
        }
    return loader


# ---------------------------------------------------------------------------

class TestBuildLayerTable(unittest.TestCase):

    def setUp(self):
        self.ts = _ts()
        self.labels = np.full(N_TOKENS, -1, dtype=np.int64)
        self.labels[[1, 2, 3, 4]] = 7

    def test_row_per_token(self):
        t = build_layer_table("pythia-410m-step512", 512, "wiki_paragraph", 3,
                              self.labels, self.ts, ParticleTable=ParticleTable)
        self.assertEqual(len(t), N_TOKENS)

    def test_roles_assigned(self):
        t = build_layer_table("m", 512, "p", 0, self.labels, self.ts,
                              ParticleTable=ParticleTable)
        roles = t.extra["token_set_role"]
        self.assertEqual(roles[1], "primary")
        self.assertEqual(roles[5], "sibling")
        self.assertEqual(roles[10], "control")
        self.assertEqual(roles[19], "none")

    def test_in_token_set_matches_primary(self):
        t = build_layer_table("m", 512, "p", 0, self.labels, self.ts,
                              ParticleTable=ParticleTable)
        self.assertEqual(sorted(np.where(t.extra["in_token_set"] == 1)[0]),
                         list(self.ts.positions))

    def test_geometry_columns_present_and_nan_without_rows(self):
        t = build_layer_table("m", 512, "p", 0, self.labels, self.ts,
                              ParticleTable=ParticleTable)
        for c in GEOMETRY_COLUMNS:
            self.assertIn(c, t.extra)
            self.assertTrue(np.all(np.isnan(t.extra[c])))

    def test_geometry_rows_merged_by_position(self):
        rows = [{"token_position": 2, "cos_to_centroid": 0.9, "norm": 12.0,
                 "norm_z": 1.5, "rank_contribution_normed": 0.03,
                 "rank_contribution_raw": 0.4}]
        t = build_layer_table("m", 512, "p", 0, self.labels, self.ts,
                              geometry_rows=rows, ParticleTable=ParticleTable)
        self.assertAlmostEqual(t.extra["cos_to_centroid"][2], 0.9)
        self.assertTrue(np.isnan(t.extra["cos_to_centroid"][3]))

    def test_unmeasured_tokens_get_nan_not_zero(self):
        rows = [{"token_position": 2, "cos_to_centroid": 0.9}]
        t = build_layer_table("m", 512, "p", 0, self.labels, self.ts,
                              geometry_rows=rows, ParticleTable=ParticleTable)
        self.assertTrue(np.isnan(t.extra["cos_to_centroid"][19]))
        self.assertFalse(t.extra["cos_to_centroid"][19] == 0.0)

    def test_out_of_range_geometry_row_ignored(self):
        rows = [{"token_position": 999, "cos_to_centroid": 0.9}]
        build_layer_table("m", 512, "p", 0, self.labels, self.ts,
                          geometry_rows=rows, ParticleTable=ParticleTable)

    def test_role_column_is_unicode_not_object(self):
        # ParticleTable.save refuses dtype=object.
        t = build_layer_table("m", 512, "p", 0, self.labels, self.ts,
                              ParticleTable=ParticleTable)
        self.assertNotEqual(t.extra["token_set_role"].dtype, object)

    def test_mismatched_token_str_dropped(self):
        t = build_layer_table("m", 512, "p", 0, self.labels, self.ts,
                              token_str=["a", "b"], ParticleTable=ParticleTable)
        self.assertEqual(len(t), N_TOKENS)


class TestComplementRetained(unittest.TestCase):

    def test_unclustered_and_unroled_tokens_have_rows(self):
        ts = _ts()
        labels = np.full(N_TOKENS, -1, dtype=np.int64)
        t = build_layer_table("m", 0, "p", 0, labels, ts,
                              ParticleTable=ParticleTable)
        self.assertEqual(len(t), N_TOKENS)
        self.assertTrue(np.all(t.columns["population"] == "unclustered"))
        self.assertIn("none", set(t.extra["token_set_role"]))

    def test_filter_recovers_the_complement(self):
        ts = _ts()
        labels = np.full(N_TOKENS, -1, dtype=np.int64)
        labels[[1, 2]] = 3
        t = build_layer_table("m", 0, "p", 0, labels, ts,
                              ParticleTable=ParticleTable)
        self.assertEqual(len(t.filter(population="unclustered")), N_TOKENS - 2)


class TestBuildSweep(unittest.TestCase):

    def setUp(self):
        self.ts = _ts()
        self.labels_by_step = {
            0: _labels_for([1, 2, 3, 4], 6),
            512: _labels_for([1, 2, 3, 4], 5),
            143000: _labels_for([1, 2, 3, 4], 3),
        }
        self.refs = [_ref(s) for s in (0, 512, 143000)]

    def test_rows_cover_every_checkpoint_layer_and_token(self):
        t, skipped = build_sweep_particle_table(
            self.ts, self.refs, _loader_from(self.labels_by_step),
            ParticleTable=ParticleTable)
        self.assertEqual(len(t), 3 * N_LAYERS * N_TOKENS)
        self.assertEqual(skipped, [])

    def test_checkpoint_step_column_populated(self):
        t, _ = build_sweep_particle_table(
            self.ts, self.refs, _loader_from(self.labels_by_step),
            ParticleTable=ParticleTable)
        self.assertEqual(set(t.columns["checkpoint_step"].tolist()),
                         {0, 512, 143000})

    def test_token_count_mismatch_skipped(self):
        labels = dict(self.labels_by_step)
        labels[512] = _labels_for([1], 3, n_tokens=N_TOKENS + 1)
        t, skipped = build_sweep_particle_table(
            self.ts, self.refs, _loader_from(labels),
            ParticleTable=ParticleTable)
        self.assertEqual(set(t.columns["checkpoint_step"].tolist()),
                         {0, 143000})
        self.assertIn("same particles", skipped[0]["reason"])

    def test_unloadable_run_skipped(self):
        t, skipped = build_sweep_particle_table(
            self.ts, self.refs, lambda p: {}, ParticleTable=ParticleTable)
        self.assertEqual(len(t), 0)
        self.assertEqual(len(skipped), 3)

    def test_step_filter(self):
        t, _ = build_sweep_particle_table(
            self.ts, self.refs, _loader_from(self.labels_by_step),
            steps=[0, 143000], ParticleTable=ParticleTable)
        self.assertEqual(set(t.columns["checkpoint_step"].tolist()),
                         {0, 143000})

    def test_geometry_joined_when_supplied(self):
        geometry = {"records": [{
            "step": 512,
            "layers": [
                {"particles": [{"token_position": 1, "cos_to_centroid": 0.77,
                                "norm": 5.0, "norm_z": 0.1,
                                "rank_contribution_normed": 0.02,
                                "rank_contribution_raw": 0.3}]}
                for _ in range(N_LAYERS)
            ],
        }]}
        t, _ = build_sweep_particle_table(
            self.ts, self.refs, _loader_from(self.labels_by_step),
            geometry=geometry, ParticleTable=ParticleTable)
        sub = t.filter(checkpoint_step=512, layer=0, token_position=1)
        self.assertAlmostEqual(float(sub.extra["cos_to_centroid"][0]), 0.77)
        other = t.filter(checkpoint_step=0, layer=0, token_position=1)
        self.assertTrue(np.isnan(other.extra["cos_to_centroid"][0]))

    def test_table_saves_and_loads(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            t, _ = build_sweep_particle_table(
                self.ts, self.refs, _loader_from(self.labels_by_step),
                ParticleTable=ParticleTable)
            p = tmp / "particles.npz"
            t.save(p)
            back = ParticleTable.load(p)
            self.assertEqual(len(back), len(t))
            self.assertIn("token_set_role", back.extra)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestBiography(unittest.TestCase):

    def setUp(self):
        self.ts = _ts()
        labels_by_step = {0: _labels_for([1, 2, 3, 4], 6),
                          512: _labels_for([1, 2, 3, 4], 3)}
        self.table, _ = build_sweep_particle_table(
            self.ts, [_ref(0), _ref(512)], _loader_from(labels_by_step),
            ParticleTable=ParticleTable)
        self.bio = particle_biography(self.table)

    def test_one_record_per_step_and_position(self):
        self.assertEqual(len(self.bio), 2 * N_TOKENS)

    def test_clustered_particle_dates(self):
        r = next(r for r in self.bio
                 if r["checkpoint_step"] == 0 and r["token_position"] == 1)
        self.assertEqual(r["first_clustered_layer"], 1)
        self.assertEqual(r["last_clustered_layer"], 6)
        self.assertEqual(r["n_layers_clustered"], 6)
        self.assertEqual(r["longest_run"], 6)

    def test_never_clustered_is_none_not_minus_one(self):
        r = next(r for r in self.bio
                 if r["checkpoint_step"] == 0 and r["token_position"] == 19)
        self.assertIsNone(r["first_clustered_layer"])
        self.assertIsNone(r["last_clustered_layer"])
        self.assertEqual(r["n_layers_clustered"], 0)

    def test_lifespan_falls_between_checkpoints(self):
        a = np.mean([r["n_layers_clustered"] for r in self.bio
                     if r["checkpoint_step"] == 0 and r["role"] == "primary"])
        b = np.mean([r["n_layers_clustered"] for r in self.bio
                     if r["checkpoint_step"] == 512 and r["role"] == "primary"])
        self.assertGreater(a, b)

    def test_role_carried(self):
        r = next(r for r in self.bio
                 if r["checkpoint_step"] == 0 and r["token_position"] == 5)
        self.assertEqual(r["role"], "sibling")

    def test_longest_run_distinguishes_flicker_from_persistence(self):
        labels = [np.full(N_TOKENS, -1, dtype=np.int64) for _ in range(N_LAYERS)]
        for li in (1, 2, 3):
            labels[li][1] = 0
        for li in (1, 5, 9):
            labels[li][2] = 0
        t, _ = build_sweep_particle_table(
            _ts(), [_ref(0)], _loader_from({0: labels}),
            ParticleTable=ParticleTable)
        bio = particle_biography(t)
        persistent = next(r for r in bio if r["token_position"] == 1)
        flicker = next(r for r in bio if r["token_position"] == 2)
        self.assertEqual(persistent["n_layers_clustered"],
                         flicker["n_layers_clustered"])
        self.assertGreater(persistent["longest_run"], flicker["longest_run"])

    def test_empty_table(self):
        self.assertEqual(particle_biography(ParticleTable.concat([])), [])

    def test_report_lines(self):
        blob = "\n".join(biography_report_lines(self.bio, role="primary"))
        self.assertIn("particle biography", blob)
        self.assertIn("mean_first", blob)

    def test_report_on_absent_role(self):
        self.assertIn("no particles",
                      "\n".join(biography_report_lines(self.bio, role="ghost")))


class TestTurnoverDiscriminates(unittest.TestCase):
    """The §7 item 1 test. Both scenarios have falling mean lifespan; only the
    per-particle decomposition separates them."""

    def _bio(self, labels_by_step):
        refs = [_ref(s) for s in sorted(labels_by_step)]
        t, _ = build_sweep_particle_table(
            _ts(), refs, _loader_from(labels_by_step),
            ParticleTable=ParticleTable)
        return particle_biography(t)

    def setUp(self):
        early = list(range(1, 9))
        late = list(range(11, 19))
        # (a) same particles, shorter memberships
        self.same = self._bio({0: _labels_for(early, 8),
                               512: _labels_for(early, 5),
                               143000: _labels_for(early, 3)})
        # (b) different particles at late checkpoints
        self.diff = self._bio({0: _labels_for(early, 8),
                               512: _labels_for(early[:4] + late[:4], 5),
                               143000: _labels_for(late, 3)})

    def test_both_scenarios_have_falling_mean_lifespan(self):
        for bio in (self.same, self.diff):
            means = [np.mean([r["n_layers_clustered"] for r in bio
                              if r["checkpoint_step"] == s])
                     for s in (0, 512, 143000)]
            self.assertGreater(means[0], means[-1])

    def test_jaccard_stays_high_for_same_particles(self):
        out = turnover_decomposition(self.same)
        j = out["first_vs_last"]["jaccard_by_threshold"][1]
        self.assertGreater(j, 0.95)

    def test_jaccard_collapses_for_different_particles(self):
        out = turnover_decomposition(self.diff)
        j = out["first_vs_last"]["jaccard_by_threshold"][1]
        self.assertLess(j, 0.2)

    def test_rank_correlation_separates_the_two(self):
        same = turnover_decomposition(self.same)["first_vs_last"][
            "rank_corr_layers_clustered"]
        diff = turnover_decomposition(self.diff)["first_vs_last"][
            "rank_corr_layers_clustered"]
        self.assertGreater(same, diff)

    def test_thresholds_reported(self):
        out = turnover_decomposition(self.same, thresholds=(1, 4))
        self.assertEqual(sorted(out["first_vs_last"]["jaccard_by_threshold"]),
                         [1, 4])

    def test_consecutive_pairs_present(self):
        out = turnover_decomposition(self.same)
        self.assertEqual([(p["step_a"], p["step_b"]) for p in out["pairs"]],
                         [(0, 512), (512, 143000)])

    def test_distinct_labels_measured_among_clustered_not_all_tokens(self):
        # The complement contributes a structural 0 and drags the plain mean
        # toward it; the claim is about particles that cluster.
        p = turnover_decomposition(self.same)["first_vs_last"]
        self.assertLess(p["mean_distinct_labels_a"],
                        p["mean_distinct_labels_among_clustered_a"])
        self.assertEqual(p["mean_distinct_labels_among_clustered_a"], 1.0)
        self.assertGreater(p["n_clustered_a"], 0)

    def test_layers_among_clustered_reported(self):
        p = turnover_decomposition(self.same)["first_vs_last"]
        self.assertGreater(p["mean_layers_among_clustered_a"],
                           p["mean_layers_clustered_a"])

    def test_degenerate_thresholds_flagged_in_the_report(self):
        blob = "\n".join(turnover_report_lines(
            turnover_decomposition(self.same, thresholds=(1, 99))))
        self.assertIn("Not evidence of turnover", blob)

    def test_no_verdict_field(self):
        # The two readings are not exhaustive; naming a winner would be the
        # premature collapse this rebuild keeps finding elsewhere.
        out = turnover_decomposition(self.same)
        self.assertNotIn("verdict", out)
        self.assertNotIn("conclusion", out)

    def test_single_checkpoint_returns_note(self):
        bio = self._bio({0: _labels_for([1, 2], 4)})
        out = turnover_decomposition(bio)
        self.assertEqual(out["pairs"], [])
        self.assertIn("note", out)

    def test_report_lines(self):
        blob = "\n".join(turnover_report_lines(turnover_decomposition(self.same)))
        self.assertIn("turnover decomposition", blob)
        self.assertIn("first vs last", blob)

    def test_report_on_uncomputable(self):
        bio = self._bio({0: _labels_for([1], 3)})
        self.assertIn("not computable",
                      "\n".join(turnover_report_lines(
                          turnover_decomposition(bio))))


class TestSetOverlap(unittest.TestCase):

    def setUp(self):
        refs = [_ref(0), _ref(512)]
        t, _ = build_sweep_particle_table(
            _ts(), refs,
            _loader_from({0: _labels_for([1, 2, 3], 6),
                          512: _labels_for([3, 4, 5], 6)}),
            ParticleTable=ParticleTable)
        self.bio = particle_biography(t)

    def test_jaccard(self):
        o = clustered_set_overlap(self.bio, 0, 512)
        self.assertEqual(o["n_a"], 3)
        self.assertEqual(o["n_b"], 3)
        self.assertEqual(o["n_intersection"], 1)
        self.assertAlmostEqual(o["jaccard"], 0.2, places=4)

    def test_threshold_raises_the_bar(self):
        o = clustered_set_overlap(self.bio, 0, 512, min_layers_clustered=99)
        self.assertEqual(o["n_a"], 0)
        self.assertIsNone(o["jaccard"])
        self.assertEqual(o["degenerate"], "both")

    def test_one_empty_side_is_degenerate_not_zero(self):
        # Threshold above the later checkpoint's ceiling: J would be a
        # mechanical 0.0 and read as complete turnover.
        refs = [_ref(0), _ref(512)]
        t, _ = build_sweep_particle_table(
            _ts(), refs,
            _loader_from({0: _labels_for([1, 2, 3], 8),
                          512: _labels_for([1, 2, 3], 3)}),
            ParticleTable=ParticleTable)
        bio = particle_biography(t)
        o = clustered_set_overlap(bio, 0, 512, min_layers_clustered=6)
        self.assertEqual(o["n_a"], 3)
        self.assertEqual(o["n_b"], 0)
        self.assertEqual(o["degenerate"], "b")
        self.assertIsNone(o["jaccard"])

    def test_non_degenerate_case_still_returns_a_number(self):
        o = clustered_set_overlap(self.bio, 0, 512, min_layers_clustered=1)
        self.assertIsNone(o["degenerate"])
        self.assertIsNotNone(o["jaccard"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
