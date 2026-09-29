"""
tests/test_phase1d_beta_refit.py — the reproduction gate in `beta_refit`
refuses anything it could not check against #108's stored βs.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.deps

from p1d_cluster_ensemble.beta_refit import check_reproduction  # noqa: E402


def _stored(tmp_path, betas):
    rec = {"step": "step1", "prompt": "p", "layer": 1,
           "betas": [[{"beta": b} for b in betas]]}
    f = tmp_path / "null.json"
    f.write_text(json.dumps({"records": [rec]}))
    return f


def _fit(head, beta, layer=1):
    return {"step": "step1", "prompt": "p", "layer": layer, "head": head, "linear": beta}


def test_exact_match_passes(tmp_path):
    r = check_reproduction([_fit(0, 1.0), _fit(1, 2.0)], _stored(tmp_path, [1.0, 2.0]))
    assert r == {"n_compared": 2, "n_mismatched": 0, "max_abs_diff": 0.0}


def test_fitted_head_absent_from_the_record_is_mismatched(tmp_path):
    r = check_reproduction([_fit(0, 1.0), _fit(0, 5.0, layer=2)], _stored(tmp_path, [1.0]))
    assert r["n_mismatched"] == 1


def test_stored_head_not_refitted_is_mismatched(tmp_path):
    r = check_reproduction([_fit(0, 1.0)], _stored(tmp_path, [1.0, 2.0]))
    assert r["n_mismatched"] == 1


def test_duplicate_fitted_head_is_mismatched(tmp_path):
    r = check_reproduction([_fit(0, 1.0), _fit(0, 1.0)], _stored(tmp_path, [1.0]))
    assert r["n_mismatched"] == 1


def test_finite_on_one_side_is_mismatched(tmp_path):
    r = check_reproduction([_fit(0, float("nan"))], _stored(tmp_path, [1.0]))
    assert r["n_mismatched"] == 1 and r["n_compared"] == 0
