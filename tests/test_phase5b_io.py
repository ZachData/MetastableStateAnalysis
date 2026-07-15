"""
tests/test_p5b_io.py — Contract tests for p5b_manifold/io.py.

Uses synthetic Phase 1 run directories that match the real v2 layout
so these tests pass without any actual model output on disk.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from p5b_manifold_steering.p5b_io import (
    find_phase1_runs,
    load_phase1_run,
    select_best_run,
    load_phase2_projectors,
)


# ---------------------------------------------------------------------------
# Synthetic fixture builder
# ---------------------------------------------------------------------------

N_LAYERS = 6
N_TOKENS = 10
D_MODEL  = 32
N_HEADS  = 4


def _make_run_dir(
    base:        Path,
    stem:        str  = "gpt2_large",
    prompt:      str  = "wiki_paragraph",
    n_traj:      int  = 5,
    lifespan:    int  = 4,
    n_plateau:   int  = 3,
    n_merge:     int  = 1,
) -> Path:
    """Write a minimal but structurally correct Phase 1 v2 run directory."""
    run_dir = base / f"{stem}_{prompt}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # geometry.json
    with open(run_dir / "geometry.json", "w") as f:
        json.dump({
            "model": stem.replace("_", "-"),
            "prompt": prompt,
            "n_layers": N_LAYERS,
            "n_tokens": N_TOKENS,
            "d_model":  D_MODEL,
        }, f)

    # trajectory.json — plateau_layers + trajectories
    plateau_layers = list(range(1, n_plateau + 1))
    trajectories = [
        {"id": i, "chain": [[j, i] for j in range(lifespan)]}
        for i in range(n_traj)
    ]
    with open(run_dir / "trajectory.json", "w") as f:
        json.dump({
            "plateau_layers": plateau_layers,
            "cluster_tracking": {
                "trajectories": trajectories,
                "events": [{"layer_from": 3, "n_merges": n_merge}],
                "summary": {"n_trajectories": n_traj},
            },
        }, f)

    # events.json — merge_layers
    with open(run_dir / "events.json", "w") as f:
        json.dump({
            "merge_layers": [3] if n_merge > 0 else [],
            "energy_violations": {"1.0": [2]},
        }, f)

    # centroid_trajectories.npz — key format: "traj_{id}"
    arrays = {}
    for i in range(n_traj):
        c = rng.standard_normal((lifespan, D_MODEL)).astype(np.float32)
        # L2-normalise each row
        norms = np.linalg.norm(c, axis=1, keepdims=True)
        c = c / np.maximum(norms, 1e-10)
        arrays[f"traj_{i}"] = c
    np.savez_compressed(run_dir / "centroid_trajectories.npz", **arrays)

    # activations.npz
    acts = rng.standard_normal((N_LAYERS, N_TOKENS, D_MODEL)).astype(np.float32)
    np.savez_compressed(run_dir / "activations.npz", activations=acts)

    return run_dir


def _make_p2_projectors(base: Path, stem: str, d: int = D_MODEL, k: int = 8) -> Path:
    """Write a minimal ov_projectors_{stem}.npz."""
    rng  = np.random.default_rng(1)
    U, _ = np.linalg.qr(rng.standard_normal((d, k * 2)))
    np.savez_compressed(
        base / f"ov_projectors_{stem}.npz",
        U_pos=U[:, :k].astype(np.float32),
        U_neg=U[:, k:].astype(np.float32),
        U_A=U[:, :k].astype(np.float32),
    )
    return base / f"ov_projectors_{stem}.npz"


# ===========================================================================
# find_phase1_runs
# ===========================================================================

class TestFindPhase1Runs:

    def setup_method(self):
        self._tmp = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self._tmp)

    def test_finds_matching_dirs(self):
        _make_run_dir(self._tmp, stem="gpt2_large", prompt="wiki_paragraph")
        _make_run_dir(self._tmp, stem="gpt2_large", prompt="paper_excerpt")
        runs = find_phase1_runs(self._tmp, "gpt2_large")
        assert "wiki_paragraph" in runs
        assert "paper_excerpt"  in runs

    def test_ignores_non_matching_stem(self):
        _make_run_dir(self._tmp, stem="gpt2_large",     prompt="wiki_paragraph")
        _make_run_dir(self._tmp, stem="bert_base", prompt="wiki_paragraph")
        runs = find_phase1_runs(self._tmp, "gpt2_large")
        assert all("gpt2_large" in str(v) for v in runs.values())

    def test_returns_empty_for_missing_stem(self):
        runs = find_phase1_runs(self._tmp, "nonexistent_model")
        assert runs == {}

    def test_values_are_paths(self):
        _make_run_dir(self._tmp, stem="gpt2_large", prompt="wiki_paragraph")
        runs = find_phase1_runs(self._tmp, "gpt2_large")
        for v in runs.values():
            assert isinstance(v, Path)
            assert v.is_dir()


# ===========================================================================
# load_phase1_run
# ===========================================================================

class TestLoadPhase1Run:

    def setup_method(self):
        self._tmp   = Path(tempfile.mkdtemp())
        self.run_dir = _make_run_dir(self._tmp)

    def teardown_method(self):
        shutil.rmtree(self._tmp)

    def test_plateau_layers_loaded(self):
        p1 = load_phase1_run(self.run_dir)
        assert isinstance(p1["plateau_layers"], list)
        assert len(p1["plateau_layers"]) == 3
        assert all(isinstance(l, int) for l in p1["plateau_layers"])

    def test_merge_layers_loaded_from_events_json(self):
        """merge_layers must come from events.json, NOT trajectory.json."""
        p1 = load_phase1_run(self.run_dir)
        assert isinstance(p1["merge_layers"], list)
        assert 3 in p1["merge_layers"]

    def test_centroid_trajs_integer_keyed(self):
        """centroid_trajs must be {int: ndarray}, not {'traj_0': ndarray}."""
        p1 = load_phase1_run(self.run_dir)
        ct = p1["centroid_trajs"]
        assert len(ct) == 5
        for k in ct:
            assert isinstance(k, int), f"Key {k!r} is not int"

    def test_centroid_traj_shape(self):
        p1 = load_phase1_run(self.run_dir)
        for tid, arr in p1["centroid_trajs"].items():
            assert arr.ndim == 2, f"traj {tid}: expected 2D, got {arr.ndim}D"
            assert arr.shape[1] == D_MODEL

    def test_activations_shape(self):
        p1 = load_phase1_run(self.run_dir)
        acts = p1["activations"]
        assert acts is not None
        assert acts.shape == (N_LAYERS, N_TOKENS, D_MODEL)

    def test_trajectories_list(self):
        p1 = load_phase1_run(self.run_dir)
        assert isinstance(p1["trajectories"], list)
        assert len(p1["trajectories"]) == 5
        for t in p1["trajectories"]:
            assert "id" in t
            assert "chain" in t

    def test_missing_centroid_file_gives_empty_dict(self):
        (self.run_dir / "centroid_trajectories.npz").unlink()
        p1 = load_phase1_run(self.run_dir)
        assert p1["centroid_trajs"] == {}

    def test_missing_events_gives_empty_merge_layers(self):
        (self.run_dir / "events.json").unlink()
        p1 = load_phase1_run(self.run_dir)
        assert p1["merge_layers"] == []

    def test_missing_trajectory_gives_empty_plateau(self):
        (self.run_dir / "trajectory.json").unlink()
        p1 = load_phase1_run(self.run_dir)
        assert p1["plateau_layers"] == []


# ===========================================================================
# select_best_run
# ===========================================================================

class TestSelectBestRun:

    def setup_method(self):
        self._tmp = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self._tmp)

    def test_prefers_explicit_prompt(self):
        runs = {
            "wiki_paragraph": self._tmp / "a",
            "paper_excerpt":  self._tmp / "b",
        }
        key, path = select_best_run(runs, preferred_prompt="paper_excerpt")
        assert key == "paper_excerpt"

    def test_falls_back_to_priority_order(self):
        # sullivan_ballou absent; should pick paper_excerpt over wiki_paragraph
        runs = {
            "wiki_paragraph": self._tmp / "a",
            "paper_excerpt":  self._tmp / "b",
        }
        key, path = select_best_run(runs, preferred_prompt=None)
        assert key == "paper_excerpt"

    def test_returns_none_for_empty(self):
        key, path = select_best_run({})
        assert key is None
        assert path is None

    def test_unknown_preferred_falls_back(self):
        runs = {"wiki_paragraph": self._tmp / "a"}
        key, path = select_best_run(runs, preferred_prompt="nonexistent")
        assert key == "wiki_paragraph"


# ===========================================================================
# load_phase2_projectors
# ===========================================================================

class TestLoadPhase2Projectors:

    def setup_method(self):
        self._tmp = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self._tmp)

    def test_loads_projectors(self):
        _make_p2_projectors(self._tmp, stem="gpt2_large")
        proj = load_phase2_projectors(self._tmp, "gpt2_large")
        assert proj is not None
        assert "U_S" in proj
        assert "U_A" in proj

    def test_projectors_are_arrays(self):
        _make_p2_projectors(self._tmp, stem="gpt2_large")
        proj = load_phase2_projectors(self._tmp, "gpt2_large")
        assert isinstance(proj["U_S"], np.ndarray)
        assert isinstance(proj["U_A"], np.ndarray)

    def test_u_s_full_built_when_both_pos_neg(self):
        _make_p2_projectors(self._tmp, stem="gpt2_large")
        proj = load_phase2_projectors(self._tmp, "gpt2_large")
        # U_S_full = concat(U_pos, U_neg)
        assert "U_S_full" in proj
        assert proj["U_S_full"].shape[1] == proj["U_S"].shape[1] + proj["U_S_neg"].shape[1]

    def test_returns_none_for_missing(self):
        proj = load_phase2_projectors(self._tmp, "nonexistent_model")
        assert proj is None

    def test_stem_dash_underscore_variant(self):
        """Should find file even when stem has dashes instead of underscores."""
        _make_p2_projectors(self._tmp, stem="gpt2_large")
        proj = load_phase2_projectors(self._tmp, "gpt2-large")
        assert proj is not None


# ===========================================================================
# load_plateau_centroids (integration with io)
# ===========================================================================

class TestLoadPlateauCentroids:

    def setup_method(self):
        self._tmp   = Path(tempfile.mkdtemp())
        self.run_dir = _make_run_dir(self._tmp, n_traj=5, lifespan=4)

    def teardown_method(self):
        shutil.rmtree(self._tmp)

    def test_returns_correct_shape(self):
        from p5b_manifold.manifold_fit import load_plateau_centroids
        p1 = load_phase1_run(self.run_dir)
        centroids, tids = load_plateau_centroids(
            p1["centroid_trajs"], p1["trajectories"], min_lifespan=2
        )
        assert centroids.ndim == 2
        assert centroids.shape[1] == D_MODEL
        assert len(tids) == centroids.shape[0]

    def test_all_five_trajectories_included(self):
        from p5b_manifold.manifold_fit import load_plateau_centroids
        p1 = load_phase1_run(self.run_dir)
        centroids, tids = load_plateau_centroids(
            p1["centroid_trajs"], p1["trajectories"], min_lifespan=2
        )
        assert len(tids) == 5

    def test_min_lifespan_filters(self):
        from p5b_manifold.manifold_fit import load_plateau_centroids
        p1 = load_phase1_run(self.run_dir)
        # lifespan=4 in fixture, min_lifespan=5 should exclude all
        with pytest.raises(ValueError):
            load_plateau_centroids(
                p1["centroid_trajs"], p1["trajectories"], min_lifespan=5
            )

    def test_centroids_l2_normalised(self):
        from p5b_manifold.manifold_fit import load_plateau_centroids
        import numpy.testing as npt
        p1 = load_phase1_run(self.run_dir)
        centroids, _ = load_plateau_centroids(
            p1["centroid_trajs"], p1["trajectories"], min_lifespan=2
        )
        norms = np.linalg.norm(centroids, axis=1)
        npt.assert_allclose(norms, 1.0, atol=1e-5)
