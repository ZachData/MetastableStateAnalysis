"""
tests/test_phase2b_io.py — Phase 2b discovery and loading.

The first class is the point of the module. `pythia-410m-step1` is a
substring of `pythia-410m-step16`, `-step128`, `-step1000` and
`-step128000`; eight of the twenty-seven Study B stems collide under the
`model_stem in d.name` predicate `run_2i.find_phase2_runs` used. On the
GPT-2 study the same predicate produced the `gpt2` aggregation entry
status-2b caveat 2 records as "a runner bug, not a result."
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from p2b_imaginary import p2b_io


STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 128000, 143000]
PROMPTS = ["wiki_paragraph", "short_heterogeneous", "repeated_tokens"]


def _write_ov(weights_dir: Path, stem: str, d=8, n_layers=3):
    rng = np.random.default_rng(abs(hash(stem)) % 2**31)
    arrays = {f"ov_total_layer_{i}": rng.normal(size=(d, d))
              for i in range(n_layers)}
    np.savez_compressed(weights_dir / f"ov_weights_{stem}.npz", **arrays)


def _write_p1_run(phase1_dir: Path, stem: str, prompt: str,
                  n_layers=4, n_tokens=6, d=8):
    run = phase1_dir / f"{stem}_{prompt}"
    run.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(run.name)) % 2**31)
    X = rng.normal(size=(n_layers, n_tokens, d))
    X = X / np.linalg.norm(X, axis=-1, keepdims=True)
    np.savez_compressed(run / "activations.npz", activations=X,
                        norms=np.ones((n_layers, n_tokens), dtype=np.float32))
    with open(run / "geometry.json", "w") as f:
        json.dump({
            "model": stem, "prompt": prompt,
            "n_tokens": n_tokens, "d_model": d,
            "layers": [{"layer": i, "effective_rank": 5.0,
                        "effective_rank_normed": 4.0} for i in range(n_layers)],
        }, f)
    with open(run / "energies.json", "w") as f:
        json.dump({
            "layers": [{"layer": i, "energies": {"1.0": 1.0 + 0.1 * i}}
                       for i in range(n_layers)],
            "violation_layers": {"1.0": [2]},
        }, f)
    return run


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.weights = self.tmp / "weights"
        self.phase1 = self.tmp / "phase1"
        self.weights.mkdir()
        self.phase1.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# The collision
# ---------------------------------------------------------------------------

class TestStemMatchingIsExact(_Tmp):

    def setUp(self):
        super().setUp()
        for step in STEPS:
            stem = f"pythia-410m-step{step}"
            _write_ov(self.weights, stem)
            for p in PROMPTS:
                _write_p1_run(self.phase1, stem, p)

    def test_step1_does_not_swallow_step16(self):
        runs = p2b_io.find_phase1_runs(self.phase1, "pythia-410m-step1")
        self.assertEqual(sorted(runs), sorted(PROMPTS))
        for path in runs.values():
            self.assertTrue(path.name.startswith("pythia-410m-step1_"))

    def test_every_checkpoint_resolves_to_its_own_prompt_count(self):
        """
        Under the old `stem in name` predicate, step1 would resolve to
        3 x (number of stems containing 'step1') runs. Each must see 3.
        """
        for step in STEPS:
            runs = p2b_io.find_phase1_runs(self.phase1, f"pythia-410m-step{step}")
            self.assertEqual(len(runs), len(PROMPTS),
                             f"step{step} resolved {len(runs)} runs")

    def test_substring_predicate_would_have_collided(self):
        """Pins the bug being fixed, so a regression is visible as a failure."""
        names = [d.name for d in self.phase1.iterdir()]
        naive = [n for n in names if "pythia-410m-step1" in n]
        self.assertGreater(len(naive), len(PROMPTS))

    def test_random_baseline_is_not_a_checkpoint_of_the_family(self):
        _write_ov(self.weights, "pythia-1.4b-random")
        found = p2b_io.discover_checkpoints(self.weights)
        self.assertNotIn("pythia-1.4b-random", [s for _, s in found])
        with_extra = p2b_io.discover_checkpoints(
            self.weights, include_non_checkpoints=True)
        self.assertIn("pythia-1.4b-random", [s for _, s in with_extra])


# ---------------------------------------------------------------------------
# Checkpoint axis
# ---------------------------------------------------------------------------

class TestCheckpointAxis(_Tmp):

    def setUp(self):
        super().setUp()
        for step in STEPS:
            _write_ov(self.weights, f"pythia-410m-step{step}")

    def test_returned_in_training_order_not_lexicographic(self):
        found = p2b_io.discover_checkpoints(self.weights)
        steps = [s for s, _ in found]
        self.assertEqual(steps, sorted(STEPS))
        self.assertEqual(steps[:4], [0, 1, 2, 4])

    def test_step_is_carried_on_the_loaded_ov_data(self):
        ov = p2b_io.load_ov_data(self.weights, "pythia-410m-step512")
        self.assertEqual(ov["checkpoint_step"], 512)
        self.assertTrue(ov["is_per_layer"])

    def test_families_do_not_mix(self):
        _write_ov(self.weights, "pythia-1.4b-step512")
        found = p2b_io.discover_checkpoints(self.weights, base="pythia-410m")
        self.assertNotIn("pythia-1.4b-step512", [s for _, s in found])

    def test_missing_weights_return_none_not_raise(self):
        self.assertIsNone(p2b_io.load_ov_data(self.weights, "pythia-410m-step99999"))


# ---------------------------------------------------------------------------
# OV loading
# ---------------------------------------------------------------------------

class TestOvLoading(_Tmp):

    def test_layer_keys_sort_numerically(self):
        """Lexicographic order puts layer_10 before layer_2 and looks fine."""
        d = 4
        arrays = {f"ov_total_layer_{i}": np.full((d, d), float(i))
                  for i in range(12)}
        np.savez_compressed(self.weights / "ov_weights_m.npz", **arrays)
        ov = p2b_io.load_ov_data(self.weights, "m")
        for i, M in enumerate(ov["ov_total"]):
            self.assertEqual(M[0, 0], float(i))

    def test_shared_weight_layout(self):
        np.savez_compressed(self.weights / "ov_weights_albert.npz",
                            ov_total_shared=np.eye(4))
        ov = p2b_io.load_ov_data(self.weights, "albert")
        self.assertFalse(ov["is_per_layer"])
        self.assertEqual(ov["layer_names"], ["shared"])

    def test_unrecognised_layout_raises_rather_than_returning_empty(self):
        np.savez_compressed(self.weights / "ov_weights_bad.npz",
                            something_else=np.eye(4))
        with self.assertRaises(KeyError):
            p2b_io.load_ov_data(self.weights, "bad")


# ---------------------------------------------------------------------------
# Phase 1 bundle
# ---------------------------------------------------------------------------

class TestPhase1Bundle(_Tmp):

    def test_phase1_violations_are_read_not_recomputed(self):
        run = _write_p1_run(self.phase1, "pythia-410m-step512", "wiki_paragraph")
        bundle = p2b_io.load_phase1_run_bundle(run)
        self.assertEqual(bundle["phase1_violation_layers"][1.0], [2])

    def test_both_rank_variants_are_carried(self):
        run = _write_p1_run(self.phase1, "pythia-410m-step512", "wiki_paragraph")
        bundle = p2b_io.load_phase1_run_bundle(run)
        self.assertEqual(bundle["phase1_effective_rank_raw"][0], 5.0)
        self.assertEqual(bundle["phase1_effective_rank_normed"][0], 4.0)

    def test_activations_load_through_the_canonical_reader(self):
        run = _write_p1_run(self.phase1, "pythia-410m-step512", "wiki_paragraph")
        acts = p2b_io.load_activations(run)
        self.assertEqual(acts.shape, (4, 6, 8))
        np.testing.assert_allclose(
            np.linalg.norm(acts, axis=-1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Contract, frame, refusal
# ---------------------------------------------------------------------------

class TestContractAndFrame(_Tmp):

    def test_ov_path_comes_from_the_registry(self):
        path = p2b_io.ov_weights_path(self.weights, "pythia-410m-step0")
        self.assertEqual(path.name, "ov_weights_pythia-410m-step0.npz")

    def test_subresult_names_match_the_registry(self):
        from core.artifacts import get_spec
        for name in p2b_io.SUBRESULT_NAMES:
            self.assertEqual(get_spec("phase2b", name).filename,
                             p2b_io.subresult_filename(name))

    def test_unknown_subresult_raises(self):
        with self.assertRaises(KeyError):
            p2b_io.subresult_filename("block9_nonexistent")

    def test_frame_spec_records_l2_sphere_and_the_revision(self):
        spec = p2b_io.frame_spec_for_activations("pythia-410m-step512")
        self.assertEqual(spec.kind, "l2_sphere")
        self.assertEqual(spec.model_rev, "pythia-410m-step512")
        self.assertFalse(spec.rope_applied)

    def test_two_checkpoints_cannot_be_silently_compared(self):
        from core.frames import verify_same_revision, FrameMismatch
        a = p2b_io.frame_spec_for_activations("pythia-410m-step512")
        b = p2b_io.frame_spec_for_activations("pythia-410m-step143000")
        with self.assertRaises(FrameMismatch):
            verify_same_revision(a, b, context="phase 2b")

    def test_legacy_run_dir_is_refused(self):
        (self.tmp / "phase2i_results.json").write_text("{}")
        with self.assertRaises(RuntimeError):
            p2b_io.refuse_legacy_run_dir(self.tmp)

    def test_clean_run_dir_is_accepted(self):
        p2b_io.refuse_legacy_run_dir(self.tmp)  # must not raise

    def test_subresult_write_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            p2b_io.write_subresult(
                self.tmp / "sub", "block1a_rotational_spectrum",
                {"x": float("nan")},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
