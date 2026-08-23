"""
tests/test_phase5b_integration.py — End-to-end integration tests for Phase 5b.

Two fixture regimes:

  A. Structural (`test_phase5b_io._make_run_dir`, --skip-logits). Exercises
     Sub-exp A and the io machinery on disk with no model. Unchanged in
     spirit from before; these tests only ever claimed files appear, and
     that claim is still worth making.

  B. Ground truth (`tests/fixtures_p5b.py`). A synthetic world in which the
     activation/behavior isometry genuinely holds, so a correct pipeline
     must FIND it and a broken one must not. This replaces the old
     `_synthetic_logit_cache`, which assigned distributions by token index
     and therefore had no relationship to cluster membership — meaning the
     old assertions ("isometry.json exists", "r_manifold is finite") were
     satisfied equally by a correct pipeline and by one computing a
     meaningless number.

Regime B asserts, in order of strength:
  * the correspondence is recovered (r_frame high, P5b-B1/B2/B3 pass)
  * the read frame beats the raw stream (the frame-vs-raw control is real)
  * a fixture with the correspondence broken produces a low r (no false
    positive)
  * a fixture built the way the ORIGINAL BUG built it produces no verdict
    at all (regression guard)

Measured separation over 50 seeds is recorded in fixtures_p5b.py. The
thresholds below are loose relative to it on purpose: they should fail when
the pipeline is wrong, not when the RNG is unlucky.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from tests.test_phase5b_io import (
    _make_run_dir, _make_p2_projectors,
    N_LAYERS, N_TOKENS, D_MODEL,
)
from tests.test_fixtures_p5b import (
    make_ground_truth, make_coherent_run_dir, make_aligned_logit_cache,
    make_orthogonal_projectors,
    DEFAULT_CHAIN_LAYERS, DEFAULT_PLATEAU, DEFAULT_MERGE,
)

import pytest

# Tier: deps -- needs the heavy tier importable (torch / transformers /
# scikit-learn / matplotlib). No model download, no run artifacts.
# Measured, not assumed; see pyproject.toml markers.
pytestmark = pytest.mark.deps
N_TRAJ   = 6
VOCAB    = 128
LIFESPAN = 5


def _make_full_p1_dir(base: Path) -> tuple[Path, Path]:
    """Structural fixture (regime A). No cluster semantics."""
    p1_dir = base / "phase1"
    p2_dir = base / "phase2"
    p2_dir.mkdir(parents=True)
    _make_run_dir(
        p1_dir, stem="gpt2_large", prompt="wiki_paragraph",
        n_traj=N_TRAJ, lifespan=LIFESPAN, n_plateau=4, n_merge=2,
    )
    _make_p2_projectors(p2_dir, stem="gpt2_large", d=D_MODEL, k=8)
    return p1_dir, p2_dir


def _base_args(p1_dir: Path, p2_dir: Path, **over):
    """
    Namespace matching build_argparser's fields.

    Kept in one place because _run_one reads args attributes directly; a
    missing field is an AttributeError deep in the pipeline rather than an
    argparse error, so new CLI options must be mirrored here when added.
    """
    ns = types.SimpleNamespace(
        phase1_dir     = str(p1_dir),
        phase2_dir     = str(p2_dir),
        prompt         = None,
        pca_dim        = 4,
        geo_pts        = 20,
        min_lifespan   = 2,
        device         = "cpu",
        skip_logits    = True,
        behavior_space = "hellinger",
        min_coverage   = 0.0,
        fast           = True,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


# ===========================================================================
# A — structural fixture, --skip-logits
# ===========================================================================

class TestRunOneDiskOutputSkipLogits(unittest.TestCase):

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self.p1_dir, self.p2_dir = _make_full_p1_dir(self._tmp)
        self.out_dir = self._tmp / "out"
        self.out_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self._tmp)

    def _args(self):
        return _base_args(self.p1_dir, self.p2_dir)

    def test_fit_summary_written(self):
        from p5b_manifold_steering.run_5b import _run_one
        _run_one(self._args(), "gpt2-large", self.out_dir)
        self.assertTrue((self.out_dir / "fit_summary.json").exists())

    def test_mh_params_written(self):
        from p5b_manifold_steering.run_5b import _run_one
        _run_one(self._args(), "gpt2-large", self.out_dir)
        self.assertTrue((self.out_dir / "mh_params.npz").exists())

    def test_report_written(self):
        from p5b_manifold_steering.run_5b import _run_one
        _run_one(self._args(), "gpt2-large", self.out_dir)
        self.assertTrue((self.out_dir / "p5b_report.txt").exists())

    def test_report_contains_required_sections(self):
        from p5b_manifold_steering.run_5b import _run_one
        _run_one(self._args(), "gpt2-large", self.out_dir)
        text = (self.out_dir / "p5b_report.txt").read_text()
        for section in ("PHASE 5b", "SUB-EXP A", "SUB-EXP B", "FALSIFICATION"):
            self.assertIn(section, text, f"Report missing '{section}'")

    def test_fit_summary_valid_json(self):
        from p5b_manifold_steering.run_5b import _run_one
        _run_one(self._args(), "gpt2-large", self.out_dir)
        summary = json.loads((self.out_dir / "fit_summary.json").read_text())
        self.assertIn("pca_explained_var", summary)
        self.assertGreater(summary["pca_explained_var"], 0.0)
        self.assertLessEqual(summary["pca_explained_var"], 1.0 + 1e-6)

    def test_frame_diagnostics_present_without_a_model(self):
        """
        sphere_gap needs only activations, so the escalation trigger must be
        recorded even under --skip-logits. Runs already completed should not
        have to be redone to obtain it.
        """
        from p5b_manifold_steering.run_5b import _run_one
        _run_one(self._args(), "gpt2-large", self.out_dir)
        summary = json.loads((self.out_dir / "fit_summary.json").read_text())
        self.assertIn("frame_diagnostics", summary)
        self.assertIn("per_layer", summary["frame_diagnostics"])

    def test_pca_basis_shape(self):
        from p5b_manifold_steering.run_5b import _run_one
        _run_one(self._args(), "gpt2-large", self.out_dir)
        data = np.load(self.out_dir / "mh_params.npz")
        self.assertIn("pca_basis", data)
        self.assertEqual(data["pca_basis"].shape[0], D_MODEL)

    def test_returns_zero_on_success(self):
        from p5b_manifold_steering.run_5b import _run_one
        self.assertEqual(_run_one(self._args(), "gpt2-large", self.out_dir), 0)

    def test_returns_one_for_missing_runs(self):
        from p5b_manifold_steering.run_5b import _run_one
        args = self._args()
        args.phase1_dir = str(self._tmp / "nonexistent")
        self.assertEqual(_run_one(args, "gpt2-large", self.out_dir), 1)


# ===========================================================================
# B — ground-truth fixture
# ===========================================================================

class _GroundTruthBase(unittest.TestCase):

    BEHAVIOR = "aligned"

    def setUp(self):
        self._tmp   = Path(tempfile.mkdtemp())
        self.p1_dir = self._tmp / "phase1"
        self.p2_dir = self._tmp / "phase2"
        self.p2_dir.mkdir(parents=True)
        self.out_dir = self._tmp / "out"
        self.out_dir.mkdir()

        self.gt = make_ground_truth(behavior=self.BEHAVIOR, seed=0)
        make_coherent_run_dir(self.p1_dir, stem="gpt2_large",
                              prompt="wiki_paragraph", gt=self.gt)
        # NOT _make_p2_projectors: that builder sets U_A = U_pos (the same
        # columns), so U_A sits inside span(U_S_full) and Sub-exp D would be
        # comparing a space with a subspace of itself. make_orthogonal_
        # projectors writes disjoint, equal-dimension S/A blocks from the
        # SAME fixed frame Q that make_ground_truth placed the latent arc
        # and A-component in (both default to SUBSPACE_FRAME_SEED), so the
        # projectors on disk describe the subspaces the data actually lives
        # in. Regime A (TestRunOneDiskOutputSkipLogits) never reaches
        # Sub-exp D and keeps the original _make_p2_projectors.
        make_orthogonal_projectors(self.p2_dir, stem="gpt2_large", d=D_MODEL)

    def tearDown(self):
        shutil.rmtree(self._tmp)

    def _args(self, **over):
        return _base_args(self.p1_dir, self.p2_dir, skip_logits=False, **over)

    def _inject_cache(self, layers=None, cache=None):
        """
        Write a logit cache covering the chain layers.

        NOTE the layer set: the union of the surviving trajectories' CHAIN
        layers, not `plateau_layers + merge_layers`. The old fixture used the
        latter, which is not the set run_5b requests — a chain routinely
        spans layers outside it, and masking against a cache that omits them
        silently yields zero coverage for those steps.
        """
        from p5b_manifold_steering.logit_cache import save_logit_cache
        layers = list(layers if layers is not None else DEFAULT_CHAIN_LAYERS)
        cache = cache if cache is not None else make_aligned_logit_cache(self.gt, layers)
        save_logit_cache(cache, self.out_dir / "logit_cache.npz")

    def _run(self, **over):
        from p5b_manifold_steering.run_5b import _run_one
        return _run_one(self._args(**over), "gpt2-large", self.out_dir)

    def _iso(self):
        return json.loads((self.out_dir / "isometry.json").read_text())


class TestGroundTruthAligned(_GroundTruthBase):
    BEHAVIOR = "aligned"

    def test_isometry_json_written(self):
        self._inject_cache()
        self.assertEqual(self._run(), 0)
        self.assertTrue((self.out_dir / "isometry.json").exists(),
                        "isometry.json not written")

    def test_schema_and_alignment_recorded(self):
        self._inject_cache()
        self._run()
        iso = self._iso()
        self.assertEqual(iso["schema"], "p5b_isometry_v2")
        # traj_ids is the audit trail back to Phase 1; without it a reader
        # cannot tell which clusters a correlation was computed over.
        self.assertEqual(len(iso["traj_ids"]), iso["n_points"])
        self.assertEqual(iso["n_points"], self.gt["n_clusters"])
        self.assertEqual(iso["behavior_aggregation_space"], "hellinger")

    def test_correspondence_is_recovered(self):
        """The isometry is real in this fixture; the pipeline must find it."""
        self._inject_cache()
        self._run()
        v = self._iso()["verdict"]
        self.assertIsNotNone(v["r_frame"], "r_frame undefined — behavior "
                                           "distances likely collapsed")
        self.assertGreater(v["r_frame"], 0.70)
        self.assertTrue(v["P5b-B2"])

    def test_read_frame_beats_raw_stream(self):
        """
        P5b-B1's control must be doing work. The fixture gives clusters a 3x
        norm spread permuted against the latent coordinate, so the raw frame
        is genuinely worse — if r_raw came out equal to r_frame, the control
        would be degenerate (which is exactly what happened when d_linear was
        computed on already-L2-normalized centroids).
        """
        self._inject_cache()
        self._run()
        v = self._iso()["verdict"]
        self.assertIsNotNone(v["r_raw"])
        self.assertGreater(v["r_frame"], v["r_raw"])
        self.assertGreater(v["delta"], 0.10)
        self.assertTrue(v["P5b-B1"])
        self.assertTrue(v["P5b-B3"])

    def test_all_readings_reported_not_selected(self):
        """
        Every frame x metric combination is written, exactly one is marked
        primary. Reporting only the best would be the post-hoc selection
        design-5b.md prohibits.
        """
        self._inject_cache()
        self._run()
        iso = self._iso()
        self.assertGreaterEqual(len(iso["readings"]), 2)
        primaries = [r for r in iso["readings"] if r["is_primary"]]
        self.assertEqual(len(primaries), 1)
        self.assertEqual(primaries[0]["activation_frame"],
                         iso["primary"]["activation_frame"])
        metrics = {r["behavior_metric"] for r in iso["readings"]}
        self.assertIn("hellinger", metrics)
        self.assertIn("sym_kl", metrics)

    def test_ln_unavailability_is_recorded_not_silent(self):
        """
        No model is loaded here, so no LN frame exists. The artifact must say
        so rather than omitting the field — a sphere number under an 'ln'
        label is the failure mode this guards.
        """
        self._inject_cache()
        self._run()
        iso = self._iso()
        self.assertIn("ln_available", iso)
        self.assertFalse(iso["ln_available"])
        self.assertNotIn("ln", {r["activation_frame"] for r in iso["readings"]})

    def test_escalation_decision_is_recorded(self):
        self._inject_cache()
        self._run()
        esc = self._iso()["escalation"]
        for k in ("activation_frame", "triggered", "threshold", "reason"):
            self.assertIn(k, esc)
        self.assertIsInstance(esc["reason"], str)
        self.assertTrue(esc["reason"])

    def test_coverage_summarized(self):
        self._inject_cache()
        self._run()
        cov = self._iso()["coverage"]
        self.assertIsNotNone(cov)
        self.assertEqual(cov["n_trajectories"], self.gt["n_clusters"])
        self.assertAlmostEqual(cov["mean_frac"], 1.0, places=6)

    def test_partial_coverage_degrades_not_drops(self):
        """
        A chain layer missing from the cache must lower coverage, not remove
        the trajectory. This is the ordinary case on real runs.
        """
        self._inject_cache(layers=list(DEFAULT_CHAIN_LAYERS)[:-1])
        self._run()
        iso = self._iso()
        self.assertEqual(iso["n_points"], self.gt["n_clusters"])
        self.assertLess(iso["coverage"]["mean_frac"], 1.0)
        self.assertGreater(iso["verdict"]["r_frame"], 0.70)

    def test_subspace_isometry_written(self):
        self._inject_cache()
        self._run()
        path = self.out_dir / "subspace_isometry.json"
        self.assertTrue(path.exists(), "subspace_isometry.json not written")
        sub = json.loads(path.read_text())
        for k in ("r_S", "r_A", "r_full", "r_linear"):
            self.assertIn(k, sub)
            self.assertIsNotNone(sub[k], f"{k} undefined")

    def test_signal_subspace_beats_full_and_null_subspace(self):
        """
        The fixture puts the latent arc inside span(U_S) and gives each
        cluster a fixed-norm component in span(U_A) that is independent of
        the latent coordinate. So S must carry the correspondence, A must
        not, and the full space must sit between them (it contains the
        signal plus the distractor). Measured over 30 seeds: r_S >= +0.987,
        r_full ~ +0.64, r_A ~ -0.06, D1 holding every time.

        This replaces the old test_subspace_isometry_written body, which
        only checked r_S/r_A/r_full were finite — satisfied equally by a
        correct pipeline and by a degenerate comparison (see
        design-5b.md, Sub-exp D amendment).
        """
        self._inject_cache()
        self._run()
        sub = json.loads((self.out_dir / "subspace_isometry.json").read_text())
        self.assertGreater(sub["r_S"], 0.80)
        self.assertGreater(sub["r_S"], sub["r_full"])
        self.assertGreaterEqual(sub["r_full"], sub["r_A"])
        self.assertLess(sub["r_A"], 0.50)
        self.assertTrue(sub["p5b_d1_pass"])

    def test_subspace_dims_equal_and_control_is_real(self):
        """
        Two guards against the comparison being confounded rather than wrong.

        dim_S == dim_A: a larger subspace captures more of any geometry
        regardless of which subspace it is, so unequal dimensions would
        confound subspace identity with subspace size.

        r_linear_is_alias False: r_linear used to be assigned equal to
        r_full, making P5b-D2 (|r_A - r_linear| < 0.05) an unfalsifiable
        restatement of D1's premise. It must now come from the ambient
        un-normalized centroids (run_5b.py passes frames["raw"] as
        centroids_raw).
        """
        self._inject_cache()
        self._run()
        sub = json.loads((self.out_dir / "subspace_isometry.json").read_text())
        self.assertEqual(sub["dim_S"], sub["dim_A"])
        self.assertFalse(sub["r_linear_is_alias"],
                         "r_linear is still an alias of r_full — the linear "
                         "control is not independent")
        self.assertNotAlmostEqual(sub["r_linear"], sub["r_full"], places=6)

    def test_merge_teleportation_written(self):
        """
        Filename: `merge_teleportation.json`.

        CHANGED from `merge_teleportation_subspace.json`. run_5b.py writes
        the former and design-5b.md's file list names the former as
        canonical, so code and design agreed and this assertion was the
        outlier — most likely copied from the module filename
        `merge_teleportation_subspace.py`. Resolved in favour of the design
        of record. (WORKING-5b.md bug #4.)
        """
        self._inject_cache()
        self._run()
        self.assertTrue((self.out_dir / "merge_teleportation.json").exists())


class TestGroundTruthShuffled(_GroundTruthBase):
    """Negative control: correspondence broken, everything else identical."""
    BEHAVIOR = "shuffled"

    def test_no_false_positive(self):
        self._inject_cache()
        self._run()
        v = self._iso()["verdict"]
        self.assertIsNotNone(v["r_frame"])
        self.assertLess(v["r_frame"], 0.70)
        self.assertFalse(v["P5b-B2"])


class TestGlobalMeanRegression(_GroundTruthBase):
    """
    Regression guard for the original bug.

    Builds a cache the way the old pipeline effectively consumed one — every
    token at a layer carrying the same distribution, i.e. what a global mean
    over all tokens produces. Every cluster then decodes identically, all
    pairwise behavior distances are zero, the correlation is undefined, and
    no verdict may pass. If this test ever reports a passing verdict, the
    masking has been lost again.
    """
    BEHAVIOR = "aligned"

    def test_uniform_distributions_yield_no_verdict(self):
        layers = list(DEFAULT_CHAIN_LAYERS)
        n_tok  = self.gt["n_tokens"]
        flat   = np.tile(self.gt["cluster_p"].mean(axis=0), (n_tok, 1))
        self._inject_cache(
            layers=layers,
            cache={int(L): flat.astype(np.float32) for L in layers},
        )
        self._run()
        v = self._iso()["verdict"]
        self.assertIsNone(v["r_frame"],
                          "behavior distances did not collapse — the fixture "
                          "is no longer reproducing the global-mean bug")
        self.assertFalse(v["P5b-B1"])
        self.assertFalse(v["P5b-B2"])
        self.assertFalse(v["P5b-B3"])


# ===========================================================================
# C — teleportation plateau null is non-trivial
# ===========================================================================

class TestTeleportationNullNonTrivial(unittest.TestCase):
    """
    When plateau distributions are stable (near-identical), plateau KL
    should be near 0. When they differ (as in merges), KL should be high.
    This verifies the null comparison is real, not hardcoded zeros.
    """

    def _peaked(self, vocab: int, idx: int) -> np.ndarray:
        p = np.ones(vocab) * 1e-6 / vocab
        p[idx] = 0.95
        return p / p.sum()

    def test_plateau_kl_near_zero_for_stable_distributions(self):
        from p5b_manifold_steering.merge_teleportation_subspace import run_merge_teleportation

        vocab = 100
        logit_dists = {l: np.tile(self._peaked(vocab, 0), (N_TOKENS, 1))
                       for l in [1, 2, 3, 4, 5]}
        logit_dists[10] = np.tile(self._peaked(vocab, 50), (N_TOKENS, 1))

        result = run_merge_teleportation(
            logit_dists, merge_layers=[10], plateau_layers=[1, 2, 3, 4, 5],
        )
        plateau_kl = result["plateau_scores"]["kl_divergence"]
        merge_kl   = result["merge_scores"]["kl_divergence"]

        if plateau_kl:
            self.assertLess(max(plateau_kl), 0.5,
                            "Plateau KL should be near zero for stable distributions")
        if merge_kl:
            self.assertGreater(max(merge_kl), 1.0,
                               "Merge KL should be high when distribution jumps")


if __name__ == "__main__":
    unittest.main()
