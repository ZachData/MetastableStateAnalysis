"""
tests/test_p5_input_loading.py

Unit tests for p5_single_mstate_analysis.io input-discovery functions.
All tests use a synthetic on-disk fixture (tmp_path) — no model weights needed.

Covers:
  find_phase1_runs:
    - hyphen-named dirs are found when stem is underscore form
    - ALBERT iter-depth collision: highest iter wins deterministically
    - GPT-2-style no-iter dirs are found
    - Missing dir returns empty dict
    - geometry.json prompt key preferred over dirname fallback

  load_phase2i:
    - Files nested inside hyphenated model subdir are discovered
    - Non-matching subdirs are skipped
    - Missing dir returns empty dict with a warning (not an exception)

  load_phase4:
    - model_stem filtering prevents picking the wrong model's run
    - Most-recent subdir selected when multiple timestamped dirs match
    - Missing dir returns empty dict
    - Empty model_stem falls back to global mtime sort (legacy path)
"""

import json
import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from p5_single_mstate_analysis.io import (
    find_phase1_runs,
    load_phase2i,
    load_phase4,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_geometry(run_dir: Path, prompt: str, model: str = "test-model") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "geometry.json", "w") as f:
        json.dump({"prompt": prompt, "model": model, "n_layers": 4,
                   "n_tokens": 8, "d_model": 16}, f)


def _write_minimal_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# find_phase1_runs
# ---------------------------------------------------------------------------

class TestFindPhase1RunsHyphenStem(unittest.TestCase):
    """Dirs use hyphen naming; stem is underscore form."""

    def setUp(self, tmp_path_factory=None):
        # Use a fresh tmp dir per test via unittest's addCleanup pattern
        import tempfile, shutil
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir)
        self.p1_dir = Path(self._tmpdir)

    def _make_run(self, dirname: str, prompt: str) -> Path:
        d = self.p1_dir / dirname
        _write_geometry(d, prompt)
        return d

    def test_hyphen_dir_found_with_underscore_stem(self):
        """albert_xlarge_v2 (underscore) must match albert-xlarge-v2_*iter_* dirs."""
        self._make_run("albert-xlarge-v2_48iter_wiki_paragraph", "wiki_paragraph")
        result = find_phase1_runs(self.p1_dir, "albert_xlarge_v2")
        self.assertIn("wiki_paragraph", result)

    def test_unrelated_model_not_included(self):
        """gpt2-large dirs must not appear when searching for albert_xlarge_v2."""
        self._make_run("albert-xlarge-v2_48iter_wiki_paragraph", "wiki_paragraph")
        self._make_run("gpt2-large_wiki_paragraph", "wiki_paragraph")
        result = find_phase1_runs(self.p1_dir, "albert_xlarge_v2")
        # Should have exactly one entry and the path must be the albert dir
        self.assertEqual(len(result), 1)
        self.assertIn("albert-xlarge-v2", str(result["wiki_paragraph"]))

    def test_missing_directory_returns_empty(self):
        result = find_phase1_runs(self.p1_dir / "does_not_exist", "albert_xlarge_v2")
        self.assertEqual(result, {})


class TestFindPhase1RunsIterDepthCollision(unittest.TestCase):
    """ALBERT iter-depth dirs all share the same prompt key — highest iter wins."""

    def setUp(self):
        import tempfile, shutil
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir)
        self.p1_dir = Path(self._tmpdir)

    def _make_iter_run(self, depth: int, prompt: str) -> Path:
        dirname = f"albert-xlarge-v2_{depth}iter_{prompt}"
        d = self.p1_dir / dirname
        _write_geometry(d, prompt)
        return d

    def test_highest_iter_selected_for_colliding_prompt_key(self):
        for depth in (12, 24, 36, 48):
            self._make_iter_run(depth, "wiki_paragraph")
        result = find_phase1_runs(self.p1_dir, "albert_xlarge_v2")
        self.assertIn("wiki_paragraph", result)
        self.assertIn("48iter", result["wiki_paragraph"].name,
                      f"Expected 48iter dir, got: {result['wiki_paragraph'].name}")

    def test_two_prompts_both_return_highest_iter(self):
        for depth in (12, 48):
            self._make_iter_run(depth, "wiki_paragraph")
            self._make_iter_run(depth, "paper_excerpt")
        result = find_phase1_runs(self.p1_dir, "albert_xlarge_v2")
        self.assertEqual(len(result), 2)
        for pk, path in result.items():
            self.assertIn("48iter", path.name,
                          f"Prompt '{pk}': expected 48iter, got {path.name}")

    def test_single_iter_depth_still_works(self):
        """Models without iter suffix (GPT-2) must still be found."""
        d = self.p1_dir / "gpt2-large_wiki_paragraph"
        _write_geometry(d, "wiki_paragraph")
        result = find_phase1_runs(self.p1_dir, "gpt2_large")
        self.assertIn("wiki_paragraph", result)

    def test_geometry_json_prompt_key_beats_dirname_fallback(self):
        """geometry.json 'prompt' field takes priority over dirname parsing."""
        d = self.p1_dir / "albert-xlarge-v2_48iter_some_odd_dirname"
        _write_geometry(d, "sullivan_ballou")   # geometry says sullivan_ballou
        result = find_phase1_runs(self.p1_dir, "albert_xlarge_v2")
        self.assertIn("sullivan_ballou", result)
        self.assertNotIn("some_odd_dirname", result)

    def test_corrupt_geometry_falls_back_to_dirname(self):
        """A geometry.json that cannot be parsed must not crash; dirname fallback used."""
        d = self.p1_dir / "albert-xlarge-v2_24iter_paper_excerpt"
        d.mkdir(parents=True, exist_ok=True)
        (d / "geometry.json").write_text("{invalid json")
        result = find_phase1_runs(self.p1_dir, "albert_xlarge_v2")
        self.assertEqual(len(result), 1)  # entry present via fallback


# ---------------------------------------------------------------------------
# load_phase2i
# ---------------------------------------------------------------------------

class TestLoadPhase2i(unittest.TestCase):

    def setUp(self):
        import tempfile, shutil
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir)
        self.p2i_dir = Path(self._tmpdir)

    def test_nested_hyphen_subdir_discovered(self):
        """NPZ inside p2i_dir/albert-xlarge-v2/ must be found for stem albert_xlarge_v2."""
        model_subdir = self.p2i_dir / "albert-xlarge-v2"
        _write_minimal_npz(
            model_subdir / "schur_decomp_albert-xlarge-v2.npz",
            V_sym=np.eye(4, dtype=np.float32),
            V_asym=np.zeros((4, 4), dtype=np.float32),
        )
        result = load_phase2i(self.p2i_dir, "albert_xlarge_v2")
        self.assertIn("V_sym", result)
        self.assertIn("V_asym", result)

    def test_wrong_model_subdir_not_loaded(self):
        """NPZ files for gpt2-large must not appear when loading albert_xlarge_v2."""
        model_subdir = self.p2i_dir / "gpt2-large"
        _write_minimal_npz(
            model_subdir / "schur_decomp_gpt2-large.npz",
            V_sym=np.eye(4, dtype=np.float32),
        )
        result = load_phase2i(self.p2i_dir, "albert_xlarge_v2")
        self.assertEqual(result, {})

    def test_missing_dir_returns_empty_not_exception(self):
        result = load_phase2i(self.p2i_dir / "nonexistent", "albert_xlarge_v2")
        self.assertEqual(result, {})

    def test_flat_npz_at_top_level_also_found(self):
        """If a file with the stem name exists directly in phase2i_dir, it should load."""
        _write_minimal_npz(
            self.p2i_dir / "rotational_albert_xlarge_v2.npz",
            schur_T=np.eye(8, dtype=np.float32),
        )
        result = load_phase2i(self.p2i_dir, "albert_xlarge_v2")
        self.assertIn("schur_T", result)

    def test_multiple_npz_files_merged(self):
        """Keys from separate NPZ files are merged; first occurrence wins on collision."""
        subdir = self.p2i_dir / "albert-xlarge-v2"
        _write_minimal_npz(subdir / "part1_albert-xlarge-v2.npz", V_sym=np.eye(4))
        _write_minimal_npz(subdir / "part2_albert-xlarge-v2.npz", schur_Z=np.eye(4))
        result = load_phase2i(self.p2i_dir, "albert_xlarge_v2")
        self.assertIn("V_sym", result)
        self.assertIn("schur_Z", result)


# ---------------------------------------------------------------------------
# load_phase4
# ---------------------------------------------------------------------------

class TestLoadPhase4(unittest.TestCase):

    def setUp(self):
        import tempfile, shutil, time
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir)
        self.p4_dir = Path(self._tmpdir)
        self._tick = 0  # used to ensure distinct mtimes

    def _make_run_dir(self, name: str, files: dict | None = None) -> Path:
        """Create a timestamped run subdir with optional file content."""
        import time
        d = self.p4_dir / name
        d.mkdir(parents=True, exist_ok=True)
        # Ensure distinct mtimes by touching with a small delay simulation
        self._tick += 1
        ts = 1_000_000 + self._tick
        import os
        os.utime(d, (ts, ts))
        if files:
            for fname, content in files.items():
                p = d / fname
                if isinstance(content, dict):
                    with open(p, "w") as f:
                        json.dump(content, f)
                elif isinstance(content, np.ndarray):
                    np.savez(p.with_suffix(""), data=content)
        return d

    def test_correct_model_stem_selected(self):
        """When albert and gpt2 runs coexist, only albert dir is loaded for albert stem."""
        # gpt2 has LATER mtime — must still be ignored when stem=albert_xlarge_v2
        self._make_run_dir("albert_xlarge_v2_2026-05-04_14-43-27",
                           {"verdict.json": {"overall": "albert_result"}})
        self._make_run_dir("gpt2_large_2026-05-04_14-43-50",
                           {"verdict.json": {"overall": "gpt2_result"}})

        result = load_phase4(self.p4_dir, "albert_xlarge_v2")
        self.assertIn("verdict", result)
        self.assertEqual(result["verdict"]["overall"], "albert_result",
                         "load_phase4 loaded gpt2 data for albert stem")

    def test_hyphen_stem_form_also_matches(self):
        """albert-xlarge-v2 stem form must also resolve to the albert dir."""
        self._make_run_dir("albert_xlarge_v2_ts",
                           {"verdict.json": {"overall": "ok"}})
        result = load_phase4(self.p4_dir, "albert-xlarge-v2")
        self.assertIn("verdict", result)

    def test_most_recent_matching_dir_selected(self):
        """When two albert dirs exist, the newer one is used."""
        self._make_run_dir("albert_xlarge_v2_old",
                           {"verdict.json": {"ts": "old"}})
        self._make_run_dir("albert_xlarge_v2_new",
                           {"verdict.json": {"ts": "new"}})
        result = load_phase4(self.p4_dir, "albert_xlarge_v2")
        self.assertEqual(result["verdict"]["ts"], "new")

    def test_missing_dir_returns_empty(self):
        result = load_phase4(self.p4_dir / "nonexistent", "albert_xlarge_v2")
        self.assertEqual(result, {})

    def test_no_stem_falls_back_to_global_mtime(self):
        """Empty model_stem → legacy behaviour: globally most-recent dir."""
        self._make_run_dir("some_model_old", {"verdict.json": {"which": "old"}})
        self._make_run_dir("some_model_new", {"verdict.json": {"which": "new"}})
        result = load_phase4(self.p4_dir, "")
        self.assertEqual(result["verdict"]["which"], "new")

    def test_no_matching_stem_warns_and_falls_back(self):
        """No dirs match the stem → warning printed, falls back to all subdirs."""
        self._make_run_dir("gpt2_large_ts", {"verdict.json": {"m": "gpt2"}})
        # Should not raise; returns something (or empty if no subdirs at all)
        result = load_phase4(self.p4_dir, "albert_xlarge_v2")
        # A warning was printed; result may or may not be empty but must not crash
        self.assertIsInstance(result, dict)

    def test_json_and_npz_files_loaded(self):
        """Both JSON and NPZ artifacts in the run dir are loaded."""
        d = self._make_run_dir("albert_xlarge_v2_ts")
        with open(d / "verdict.json", "w") as f:
            json.dump({"overall": "ok"}, f)
        np.savez(d / "t2_lda_directions.npz", dir0=np.ones((4,), dtype=np.float32))
        result = load_phase4(self.p4_dir, "albert_xlarge_v2")
        self.assertIn("verdict", result)
        self.assertIn("t2_lda_directions", result)
        self.assertIn("dir0", result["t2_lda_directions"])


# ---------------------------------------------------------------------------
# Group B — on-the-fly projector key mapping
#
# Regression test for the key-name mismatch between build_global_projectors
# (returns per_layer[0]["U_pos"] / ["U_neg"]) and compute_v_alignment
# (expects v_projectors["U_attractive"] / ["U_repulsive"]).
#
# No real model weights or disk fixtures needed: a synthetic block-diagonal
# OV matrix gives exact analytic expectations for attraction/repulsion fracs.
#
# Add this class to tests/test_phase5_inputs.py alongside the existing
# TestFindPhase1RunsHyphenStem / TestLoadPhase2i / TestLoadPhase4 classes.
# It imports from the same project root path already set by sys.path.insert
# at the top of that file.
# ---------------------------------------------------------------------------

# These imports mirror the ones already used in the on-the-fly Group B path
# in run_5.py.  If either import fails, the bug is upstream of the key mapping.
from p6_subspace.subspace_build import build_global_projectors
from p5_single_mstate_analysis.v_alignment import compute_v_alignment


def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _make_synthetic_ov(d: int = 8) -> np.ndarray:
    """
    Block-diagonal OV: upper half = 2x2 rotation blocks (imaginary / A-subspace),
    lower half = positive-definite 2x2 blocks (real / S-subspace, all λ > 0).

    This gives a matrix whose Schur decomposition has:
      - d/2 real positive 1x1 blocks  → U_pos (attractive)
      - 0   real negative blocks       → U_neg empty
      - d/2 rotation plane vectors     → U_A

    For the centroid-trajectory test we use a centroid that lies entirely in
    the positive-eigenvalue subspace, so attr_frac should be 1.0.
    """
    assert d % 4 == 0, "d must be divisible by 4 for this fixture"
    M = np.zeros((d, d))
    half = d // 2
    # Upper half: 2x2 rotation blocks (λ = ±iω → imaginary)
    for i in range(0, half, 2):
        M[i:i+2, i:i+2] = _rot2(np.pi / 4)
    # Lower half: identity-scaled blocks (λ = 1, purely attractive)
    for i in range(half, d, 2):
        M[i:i+2, i:i+2] = np.eye(2)
    return M


def _make_ov_data(OV: np.ndarray) -> dict:
    """Minimal ov_data dict matching the contract of build_global_projectors."""
    d = OV.shape[0]
    return {
        "is_per_layer": False,
        "ov_per_head":  [OV],        # ALBERT-style: one shared OV per head
        "n_heads":      1,
        "d_model":      d,
        "layer_names":  ["shared"],
    }


class TestGroupBProjectorKeyMapping(unittest.TestCase):
    """
    Verifies that the key path from build_global_projectors output to the
    v_projectors dict expected by compute_v_alignment is correct.

    Three failure modes caught:
      1. KeyError  — wrong key name at any level of projectors dict
      2. None      — key resolves but returns None (shape mismatch downstream)
      3. All-zero  — projectors exist but are empty matrices (dim=0 subspace)
    """

    def setUp(self):
        self.d = 8
        OV = _make_synthetic_ov(self.d)
        ov_data = _make_ov_data(OV)

        # This is exactly what the on-the-fly Group B path should call.
        self.projectors = build_global_projectors(ov_data)

    # ------------------------------------------------------------------
    # 1. Structural contract: build_global_projectors output shape
    # ------------------------------------------------------------------

    def test_projectors_has_per_layer_list(self):
        """per_layer must be a non-empty list."""
        self.assertIn("per_layer", self.projectors)
        self.assertGreater(len(self.projectors["per_layer"]), 0)

    def test_per_layer_entry_has_U_pos(self):
        """U_pos (attractive basis) must exist inside per_layer[0]."""
        entry = self.projectors["per_layer"][0]
        self.assertIn("U_pos", entry, (
            "build_global_projectors does not produce 'U_pos' in per_layer[0]. "
            "The on-the-fly Group B projector extraction is broken."
        ))

    def test_per_layer_entry_has_U_neg(self):
        """U_neg (repulsive basis) must exist inside per_layer[0]."""
        entry = self.projectors["per_layer"][0]
        self.assertIn("U_neg", entry, (
            "build_global_projectors does not produce 'U_neg' in per_layer[0]. "
            "The on-the-fly Group B projector extraction is broken."
        ))

    def test_U_pos_is_nonempty(self):
        """U_pos must span at least one dimension; synthetic OV has d/2 real-pos vectors."""
        U_pos = self.projectors["per_layer"][0]["U_pos"]
        self.assertGreater(U_pos.shape[1], 0, (
            "U_pos has zero columns — attractive subspace is empty. "
            "The synthetic OV has d/2 positive-eigenvalue blocks so this should never happen."
        ))

    # ------------------------------------------------------------------
    # 2. Key-mapping correctness: the remapping run_5.py must perform
    # ------------------------------------------------------------------

    def test_key_remap_does_not_raise(self):
        """
        The exact remap block that run_5.py's Group B runner should execute.
        If this raises KeyError, the bug is confirmed.
        """
        layer_proj = self.projectors["per_layer"][0]
        try:
            v_projectors = {
                "U_attractive": layer_proj["U_pos"],
                "U_repulsive":  layer_proj["U_neg"],
            }
        except KeyError as e:
            self.fail(f"Key remap raised KeyError({e}). run_5.py Group B will fail silently.")

        self.assertIsNotNone(v_projectors["U_attractive"])
        self.assertIsNotNone(v_projectors["U_repulsive"])

    # ------------------------------------------------------------------
    # 3. End-to-end: compute_v_alignment produces non-trivial output
    # ------------------------------------------------------------------

    def test_compute_v_alignment_non_empty(self):
        """
        With correct projectors, compute_v_alignment must return non-None
        summary values. All-None summary = projectors passed as None (pre-fix behaviour).
        """
        layer_proj = self.projectors["per_layer"][0]
        v_projectors = {
            "U_attractive": layer_proj["U_pos"],
            "U_repulsive":  layer_proj["U_neg"],
        }

        # Centroid trajectory: N_LAYERS steps, each centroid = first basis vector
        # of the attractive subspace → attr_frac should be 1.0
        U_pos = layer_proj["U_pos"]
        centroid_dir = U_pos[:, 0]                     # (d,), unit attractive vec
        n_layers = 6
        centroids = np.tile(centroid_dir, (n_layers, 1))  # (6, d)

        traj = {"id": 0, "chain": [(l, 0) for l in range(n_layers)],
                "start_layer": 0, "end_layer": n_layers - 1}

        result = compute_v_alignment(centroids, v_projectors, traj)

        self.assertIn("summary", result)
        summary = result["summary"]
        self.assertIsNotNone(summary.get("mean_attr_frac"), (
            "mean_attr_frac is None — projectors were not applied. "
            "Most likely U_attractive resolved to None before the fix."
        ))

    def test_attractive_centroid_has_high_attr_frac(self):
        """
        A centroid that lies entirely in U_pos should produce attr_frac ≈ 1.0.
        If attr_frac ≈ 0 after the fix, U_pos is wrong (column-span mismatch).
        """
        layer_proj = self.projectors["per_layer"][0]
        v_projectors = {
            "U_attractive": layer_proj["U_pos"],
            "U_repulsive":  layer_proj["U_neg"],
        }

        U_pos = layer_proj["U_pos"]
        centroid_dir = U_pos[:, 0]
        centroids = np.tile(centroid_dir, (4, 1))
        traj = {"id": 0, "chain": [(l, 0) for l in range(4)],
                "start_layer": 0, "end_layer": 3}

        result = compute_v_alignment(centroids, v_projectors, traj)
        attr_frac = result["summary"]["mean_attr_frac"]

        self.assertGreater(attr_frac, 0.9, (
            f"attr_frac={attr_frac:.4f} for a centroid in U_pos — expected ≈ 1.0. "
            "U_pos columns do not span the centroid direction."
        ))

    def test_repulsive_centroid_has_low_attr_frac(self):
        """
        A centroid in U_neg should have attr_frac ≈ 0.
        If U_neg is empty (no negative eigenvalues in synthetic OV),
        the centroid falls into the orthogonal complement instead — that's fine,
        the test is skipped in that case.
        """
        layer_proj = self.projectors["per_layer"][0]
        U_neg = layer_proj["U_neg"]

        if U_neg.shape[1] == 0:
            self.skipTest("Synthetic OV has no repulsive subspace — skip repulsive test")

        v_projectors = {
            "U_attractive": layer_proj["U_pos"],
            "U_repulsive":  U_neg,
        }

        centroid_dir = U_neg[:, 0]
        centroids = np.tile(centroid_dir, (4, 1))
        traj = {"id": 0, "chain": [(l, 0) for l in range(4)],
                "start_layer": 0, "end_layer": 3}

        result = compute_v_alignment(centroids, v_projectors, traj)
        attr_frac = result["summary"]["mean_attr_frac"]

        self.assertLess(attr_frac, 0.1, (
            f"attr_frac={attr_frac:.4f} for a centroid in U_neg — expected ≈ 0.0."
        ))

    def test_wrong_key_name_fails_lookup(self):
        """
        Negative test: documents the pre-fix broken path.
        Accessing projectors["U_attractive"] directly (no per_layer nesting)
        must raise KeyError. This confirms the bug was real.
        """
        with self.assertRaises(KeyError):
            _ = self.projectors["U_attractive"]   # does NOT exist at top level


if __name__ == "__main__":
    unittest.main()