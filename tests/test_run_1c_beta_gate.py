"""
`run_1c.py` must only demand a beta for the sub-experiments that use one.

Until 2026-09-19 the driver skipped every run whose `beta_used` was
non-finite, whatever `--subexp` asked for. Since no `geometry.json` in the
tree carries `beta_eff`, `--subexp E` against the real 410m sweep reported
8 runs, 8 SKIPs and 0 written — E reads activations and never looks at beta.
A and B integrate gamma_beta and genuinely cannot proceed without one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from p1c_frames.run_1c import BETA_SUBEXPERIMENTS, requires_beta, run_one

pytestmark = pytest.mark.pure


@pytest.fixture
def run_dir(tmp_path):
    """A minimal Phase 1 run directory with no `beta_eff`, as every run on
    disk is."""
    n_layers, n_tok, d = 4, 12, 8
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_layers, n_tok, d))
    X /= np.linalg.norm(X, axis=2, keepdims=True)
    np.savez(tmp_path / "activations.npz", activations=X,
             norms=np.ones((n_layers, n_tok)))
    (tmp_path / "geometry.json").write_text(json.dumps({
        "model": "m", "prompt": "p", "n_tokens": n_tok, "d_model": d,
        "n_layers": n_layers,
        "layers": [{"layer": i, "ip_mean": 0.1 * i} for i in range(n_layers)],
    }))
    return tmp_path


def test_only_A_and_B_require_beta():
    assert BETA_SUBEXPERIMENTS == {"A", "B"}
    for wanted in ({"A"}, {"B"}, {"A", "B"}, {"A", "E"}):
        assert requires_beta(wanted) is True
    for wanted in ({"E"}, {"F"}, {"C"}, {"D"}, {"E", "F"}, set()):
        assert requires_beta(wanted) is False


def test_E_runs_without_a_beta(run_dir):
    out = run_one(run_dir, {"E"}, float("nan"), causal=True, t_target=0.9)
    assert out["beta_required"] is False
    assert out["beta_source"] == "unavailable"
    # The point of the fix: E produced its measurement anyway.
    assert "E" in out and "min_margin" in out["E"]


def test_A_still_reports_that_it_needs_one(run_dir):
    out = run_one(run_dir, {"A"}, float("nan"), causal=True, t_target=0.9)
    assert out["beta_required"] is True
    assert not np.isfinite(out["beta_used"])


def test_a_supplied_fallback_is_still_named_as_one(run_dir):
    out = run_one(run_dir, {"A"}, 1.0, causal=True, t_target=0.9)
    assert out["beta_required"] is True
    assert out["beta_used"] == 1.0
    # A fallback must never be mistaken for a measured beta.
    assert out["beta_source"] == "fallback_flag"


def test_a_measured_beta_is_named_as_measured(run_dir):
    geo = json.loads((run_dir / "geometry.json").read_text())
    geo["beta_eff"] = 2.5
    (run_dir / "geometry.json").write_text(json.dumps(geo))
    out = run_one(run_dir, {"E"}, float("nan"), causal=True, t_target=0.9)
    assert out["beta_used"] == 2.5
    assert out["beta_source"] == "geometry.json"
