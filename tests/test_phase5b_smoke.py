"""
tests/test_phase5b_smoke.py — Tier 1 smoke test for p5b_manifold_steering
(run_5b.py).

Fully offline and self-contained: with skip_logits=True, _run_one never
needs a live model, so this test builds its own synthetic Phase 1 run on
disk (geometry.json / trajectory.json / events.json /
centroid_trajectories.npz, matching p5b_io._load_phase1_run_direct's
documented schema) rather than depending on tiny_phase1_dir. This makes it
the cheapest of the five new smoke tests to actually run.

Three real, previously-undiscovered bugs surfaced while building this test
— all fixed in the accompanying patched files, not just documented:

FIX-P5B-1 (run_5b.py, all three affected files)
    _run_one's local imports referenced a package/module naming scheme
    that never matches the actual repo: `p5b_manifold.io`,
    `p5b_manifold.merge_teleportation`, `p5b_manifold.report` (missing the
    "_steering" suffix on the package, and — for those three specifically —
    the module was itself renamed: io.py -> p5b_io.py, report.py ->
    p5b_report.py, merge_teleportation.py -> merge_teleportation_subspace.py).
    Every docstring header in this package still self-identifies as
    "p5b_manifold/<name>.py" — the renames evidently happened without
    updating any of the cross-references. This is a guaranteed
    ModuleNotFoundError the instant _run_one is actually called; main()
    happens to catch it per-model and report "FAILED: ..." rather than
    crashing outright, which is presumably why it went unnoticed. Fixed by
    correcting all 8 import statements to their real locations.

FIX-P5B-2 (manifold_fit.py::compute_fit_summary)
    _run_one called `compute_fit_summary(mh, my if my is not None else mh, evr)`
    whenever no behavior manifold could be fit (fewer than 4 logit-bearing
    plateau layers — the common case for any offline/no-model run).
    compute_fit_summary unconditionally read `my["vocab"]`, and
    fit_activation_manifold's return dict (substituted in for `my` here)
    has no such key — guaranteed KeyError. Fixed by making
    compute_fit_summary accept `my: dict | None` and report the
    behavior-side fields as None/False instead of crashing.

FIX-P5B-3 (p5b_report.py::write_report)
    Exposed only once FIX-P5B-2 was applied: `fs.get('my_spline_residual_rms',
    float('nan'))` formatted with `:.4f}` — `.get(key, default)` only
    substitutes its default for a *missing* key, and the key is now always
    present (per FIX-P5B-2) with an explicit `None` value when there's no
    behavior manifold. Formatting None with `:.4f` raises TypeError. Fixed
    with a small `_num()` helper that treats "present but None" the same
    as "missing".

Confirmed by actually running _run_one end-to-end against the synthetic
fixture below, before and after each fix — see the fix commit history / the
conversation this came from, not just asserted here.

Run with:
    pytest -m smoke tests/test_phase5b_smoke.py -v
(no SMOKE_REAL_DEPS needed — this test never imports torch/transformers)
"""
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def _write_synthetic_phase1_run(run_dir: Path, n_traj: int = 6, d: int = 16,
                                 lifespan: int = 5, seed: int = 0) -> None:
    """
    Matches p5b_io._load_phase1_run_direct's documented schema exactly.
    Centroids are placed on a clean ring in the first two dimensions (small
    noise elsewhere) purely so Sub-exp A's PCA/spline fit has something
    non-degenerate to work with — the geometry itself isn't under test
    here, only that the pipeline runs and reports on it correctly.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    angles  = np.linspace(0, 2 * np.pi, n_traj, endpoint=False)
    centers = np.zeros((n_traj, d))
    centers[:, 0] = np.cos(angles)
    centers[:, 1] = np.sin(angles)
    centers[:, 2:] += rng.standard_normal((n_traj, d - 2)) * 0.02

    (run_dir / "geometry.json").write_text(json.dumps({
        "n_layers": lifespan, "n_tokens": 20, "d_model": d,
        "prompt": "wiki_paragraph", "model": "dummy-model",
    }))

    trajectories = [
        {"id": i, "chain": [[L, i] for L in range(lifespan)]}
        for i in range(n_traj)
    ]
    (run_dir / "trajectory.json").write_text(json.dumps({
        "plateau_layers": [1, 2, 3],
        "cluster_tracking": {"trajectories": trajectories},
    }))

    (run_dir / "events.json").write_text(json.dumps({"merge_layers": []}))

    arrs = {}
    for i in range(n_traj):
        pts = centers[i][None, :] + rng.standard_normal((lifespan, d)) * 0.01
        arrs[f"traj_{i}"] = pts.astype(np.float32)
    np.savez_compressed(run_dir / "centroid_trajectories.npz", **arrs)


@pytest.fixture(scope="session")
def tiny_phase5b_result(tmp_path_factory):
    import argparse
    from p5b_manifold_steering.run_5b import _run_one

    phase1_root = tmp_path_factory.mktemp("phase5b_smoke_phase1")
    _write_synthetic_phase1_run(phase1_root / "dummy_model_wiki_paragraph")

    out_dir = tmp_path_factory.mktemp("phase5b_smoke_out")
    args = argparse.Namespace(
        phase1_dir=str(phase1_root),
        phase2_dir=str(tmp_path_factory.mktemp("phase5b_smoke_phase2_absent")),
        prompt=None,
        pca_dim=8,
        geo_pts=50,
        min_lifespan=3,
        device="cpu",
        skip_logits=True,
        fast=True,
        out=None,
    )
    rc = _run_one(args, "dummy_model", out_dir)
    return rc, out_dir


def test_phase5b_run_succeeds(tiny_phase5b_result):
    rc, out_dir = tiny_phase5b_result
    assert rc == 0, f"_run_one returned {rc} (skip/failure) under out_dir={out_dir}"


def test_phase5b_expected_outputs_written(tiny_phase5b_result):
    _, out_dir = tiny_phase5b_result
    for name in ("fit_summary.json", "mh_params.npz", "p5b_report.txt"):
        assert (out_dir / name).exists(), f"missing {name} under {out_dir}"


def test_phase5b_fit_summary_reflects_no_behavior_manifold(tiny_phase5b_result):
    """
    With skip_logits=True there are always 0 logit-bearing plateau layers,
    so My is never fit. This is exactly FIX-P5B-2/3's code path — pins it
    down as a real regression test rather than just "didn't crash".
    """
    _, out_dir = tiny_phase5b_result
    with open(out_dir / "fit_summary.json") as f:
        fs = json.load(f)

    assert fs["my_spline_residual_rms"] is None
    assert fs["vocab_size"] is None
    assert fs["p5b_a2_pass"] is False
    assert np.isfinite(fs["mh_spline_residual_rms"])
    assert np.isfinite(fs["pca_explained_var"])


def test_phase5b_report_has_no_behavior_manifold_line_without_crashing(tiny_phase5b_result):
    _, out_dir = tiny_phase5b_result
    text = (out_dir / "p5b_report.txt").read_text()
    assert "My spline residual RMS" in text
    assert "nan" in text.lower(), (
        "expected the None-safe _num() helper to render My's missing "
        "residual as nan, not crash or silently omit the line"
    )


def test_phase5b_subexperiments_b_c_d_gracefully_skipped(tiny_phase5b_result):
    """No logits -> B, C, D all skip rather than crash or silently no-op
    with wrong data. Confirmed via stdout in the fixture's own dev run;
    here we just confirm no artifacts got written for them."""
    _, out_dir = tiny_phase5b_result
    for name in ("isometry.json", "merge_teleportation.json", "subspace_isometry.json"):
        assert not (out_dir / name).exists(), (
            f"{name} was written despite no logits being available — "
            "one of Sub-exp B/C/D ran when it should have skipped"
        )
