"""
tests/test_p5b_integration.py — End-to-end integration tests for Phase 5b.

Builds a complete synthetic Phase 1 directory on disk (matching the real v2
layout), runs the full pipeline via run_5b._run_one with --skip-logits, and
checks that all expected output files are written and contain sensible values.

No model weights or GPU required: --skip-logits bypasses the forward pass so
Sub-exp B and C are skipped, but Sub-exp A and the io machinery are fully
exercised on disk.

A second test runs the full pipeline including synthetic logit distributions
injected via a patched logit_cache, covering Sub-exps B, C, and D.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Re-use the fixture builder from test_p5b_io
from tests.test_phase5b_io import (
    _make_run_dir, _make_p2_projectors,
    N_LAYERS, N_TOKENS, D_MODEL,
)

N_TRAJ  = 6
VOCAB   = 128
LIFESPAN = 5


def _make_full_p1_dir(base: Path) -> tuple[Path, Path]:
    """Build a phase1_dir with one gpt2_large run and return (p1_dir, p2_dir)."""
    p1_dir = base / "phase1"
    p2_dir = base / "phase2"
    p2_dir.mkdir(parents=True)

    _make_run_dir(
        p1_dir, stem="gpt2_large", prompt="wiki_paragraph",
        n_traj=N_TRAJ, lifespan=LIFESPAN,
        n_plateau=4, n_merge=2,
    )
    _make_p2_projectors(p2_dir, stem="gpt2_large", d=D_MODEL, k=8)
    return p1_dir, p2_dir


def _synthetic_logit_cache(layer_idxs: list[int], n_tok: int, vocab: int) -> dict:
    """
    Build a logit distribution cache where cluster i concentrates on token i.
    Designed to have structure that produces non-trivial My.
    """
    rng = np.random.default_rng(42)
    out = {}
    for idx, layer in enumerate(layer_idxs):
        p = np.ones((n_tok, vocab)) * 1e-4
        # Each token gets a peaked distribution on a different vocabulary item
        for t in range(n_tok):
            tok_class = (t + idx) % vocab
            p[t, tok_class]          = 0.80
            p[t, (tok_class+1)%vocab] = 0.10
            p[t, (tok_class-1)%vocab] = 0.10
        p /= p.sum(axis=1, keepdims=True)
        out[layer] = p.astype(np.float32)
    return out


# ===========================================================================
# A — output files written with skip-logits
# ===========================================================================

class TestRunOneDiskOutputSkipLogits(unittest.TestCase):

    def setUp(self):
        self._tmp  = Path(tempfile.mkdtemp())
        self.p1_dir, self.p2_dir = _make_full_p1_dir(self._tmp)
        self.out_dir = self._tmp / "out"
        self.out_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self._tmp)

    def _args(self):
        ns = types.SimpleNamespace(
            phase1_dir   = str(self.p1_dir),
            phase2_dir   = str(self.p2_dir),
            prompt       = None,
            pca_dim      = 4,
            geo_pts      = 20,
            min_lifespan = 2,
            device       = "cpu",
            skip_logits  = True,
            fast         = True,
        )
        return ns

    def test_fit_summary_written(self):
        from p5b_manifold_steering.run_5b import _run_one
        _run_one(self._args(), "gpt2-large", self.out_dir)
        path = self.out_dir / "fit_summary.json"
        self.assertTrue(path.exists(), "fit_summary.json not written")

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
        with open(self.out_dir / "fit_summary.json") as f:
            summary = json.load(f)
        self.assertIn("pca_explained_var", summary)
        self.assertGreater(summary["pca_explained_var"], 0.0)
        self.assertLessEqual(summary["pca_explained_var"], 1.0 + 1e-6)

    def test_pca_basis_shape(self):
        from p5b_manifold_steering.run_5b import _run_one
        _run_one(self._args(), "gpt2-large", self.out_dir)
        data = np.load(self.out_dir / "mh_params.npz")
        self.assertIn("pca_basis", data)
        basis = data["pca_basis"]
        self.assertEqual(basis.shape[0], D_MODEL)

    def test_returns_zero_on_success(self):
        from p5b_manifold_steering.run_5b import _run_one
        rc = _run_one(self._args(), "gpt2-large", self.out_dir)
        self.assertEqual(rc, 0)

    def test_returns_one_for_missing_runs(self):
        from p5b_manifold_steering.run_5b import _run_one
        args = self._args()
        args.phase1_dir = str(self._tmp / "nonexistent")
        rc = _run_one(args, "gpt2-large", self.out_dir)
        self.assertEqual(rc, 1)


# ===========================================================================
# B — full pipeline with injected logit cache
# ===========================================================================

class TestRunOneWithLogits(unittest.TestCase):
    """
    Injects a synthetic logit cache so Sub-exps B and C run without a model.
    Patches logit_cache.extract_layer_logits to return synthetic distributions.
    """

    def setUp(self):
        self._tmp  = Path(tempfile.mkdtemp())
        self.p1_dir, self.p2_dir = _make_full_p1_dir(self._tmp)
        self.out_dir = self._tmp / "out"
        self.out_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self._tmp)

    def _args(self):
        return types.SimpleNamespace(
            phase1_dir   = str(self.p1_dir),
            phase2_dir   = str(self.p2_dir),
            prompt       = None,
            pca_dim      = 4,
            geo_pts      = 20,
            min_lifespan = 2,
            device       = "cpu",
            skip_logits  = False,
            fast         = True,
        )

    def _inject_cache(self, out_dir: Path, plateau_layers: list[int],
                      merge_layers: list[int]) -> None:
        """Write a synthetic logit cache npz directly into out_dir."""
        from p5b_manifold_steering.logit_cache import save_logit_cache
        all_layers = list(set(plateau_layers + merge_layers))
        cache = _synthetic_logit_cache(all_layers, N_TOKENS, VOCAB)
        save_logit_cache(cache, out_dir / "logit_cache.npz")

    def test_isometry_json_written_when_cache_present(self):
        from p5b_manifold_steering.run_5b import _run_one
        from p5b_manifold_steering.p5b_io import find_phase1_runs, load_phase1_run, select_best_run

        # Pre-inject cache before running
        runs = find_phase1_runs(self.p1_dir, "gpt2_large")
        pk, rd = select_best_run(runs)
        p1 = load_phase1_run(rd)
        self._inject_cache(
            self.out_dir, p1["plateau_layers"], p1["merge_layers"]
        )

        rc = _run_one(self._args(), "gpt2-large", self.out_dir)
        self.assertEqual(rc, 0)
        self.assertTrue((self.out_dir / "isometry.json").exists(),
                        "isometry.json not written")

    def test_isometry_r_manifold_finite(self):
        from p5b_manifold_steering.run_5b import _run_one
        from p5b_manifold_steering.p5b_io import find_phase1_runs, load_phase1_run, select_best_run

        runs = find_phase1_runs(self.p1_dir, "gpt2_large")
        pk, rd = select_best_run(runs)
        p1 = load_phase1_run(rd)
        self._inject_cache(self.out_dir, p1["plateau_layers"], p1["merge_layers"])

        _run_one(self._args(), "gpt2-large", self.out_dir)
        with open(self.out_dir / "isometry.json") as f:
            iso = json.load(f)
        r = iso.get("r_manifold")
        self.assertIsNotNone(r)
        self.assertTrue(np.isfinite(r), f"r_manifold is not finite: {r}")
        self.assertGreaterEqual(r, -1.0)
        self.assertLessEqual(r,  1.0)

    def test_subspace_isometry_written_when_projectors_present(self):
        from p5b_manifold_steering.run_5b import _run_one
        from p5b_manifold_steering.p5b_io import find_phase1_runs, load_phase1_run, select_best_run

        runs = find_phase1_runs(self.p1_dir, "gpt2_large")
        pk, rd = select_best_run(runs)
        p1 = load_phase1_run(rd)
        self._inject_cache(self.out_dir, p1["plateau_layers"], p1["merge_layers"])

        _run_one(self._args(), "gpt2-large", self.out_dir)
        path = self.out_dir / "subspace_isometry.json"
        self.assertTrue(path.exists(), "subspace_isometry.json not written")
        with open(path) as f:
            sub = json.load(f)
        for k in ("r_S", "r_A", "r_full"):
            self.assertIn(k, sub)
            self.assertTrue(np.isfinite(sub[k]), f"{k} not finite")

    def test_merge_teleportation_written_when_merges_present(self):
        from p5b_manifold_steering.run_5b import _run_one
        from p5b_manifold_steering.p5b_io import find_phase1_runs, load_phase1_run, select_best_run

        runs = find_phase1_runs(self.p1_dir, "gpt2_large")
        pk, rd = select_best_run(runs)
        p1 = load_phase1_run(rd)
        self._inject_cache(self.out_dir, p1["plateau_layers"], p1["merge_layers"])

        _run_one(self._args(), "gpt2-large", self.out_dir)
        # Merge teleportation depends on having merge layers in events.json
        if p1["merge_layers"]:
            self.assertTrue(
                (self.out_dir / "merge_teleportation_subspace.json").exists()
            )


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
        # plateau layers: all near-identical distributions
        logit_dists = {l: np.tile(self._peaked(vocab, 0), (N_TOKENS, 1))
                       for l in [1, 2, 3, 4, 5]}
        # merge layer: jumps to a different distribution
        logit_dists[10] = np.tile(self._peaked(vocab, 50), (N_TOKENS, 1))

        result = run_merge_teleportation(
            logit_dists,
            merge_layers   = [10],
            plateau_layers = [1, 2, 3, 4, 5],
        )
        plateau_kl = result["plateau_scores"]["kl_divergence"]
        merge_kl   = result["merge_scores"]["kl_divergence"]

        if plateau_kl:
            self.assertLess(
                max(plateau_kl), 0.5,
                "Plateau KL should be near zero for stable distributions"
            )
        if merge_kl:
            self.assertGreater(
                max(merge_kl), 1.0,
                "Merge KL should be high when distribution jumps"
            )


if __name__ == "__main__":
    unittest.main()
