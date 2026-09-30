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


def test_summary_counts_a_refused_head_per_variant():
    """A head one variant refused (NaN) leaves that variant's count, not the row's."""
    from p1d_cluster_ensemble.beta_refit import VARIANTS, summarise
    heads = []
    for h in range(3):
        r = {"step": "step1", "prompt": "p", "layer": 5, "head": h, "dedupe": False,
             "linear_r2": 0.1, "fe_full_r2": 0.2, "fe_full_pr2": 0.1}
        for v in VARIANTS:
            r[v] = float(h + 1)
        heads.append(r)
    heads[0]["fe_w4"] = float("nan")
    row = next(r for r in summarise(heads) if r["band"] == "L1-8" and r["floor"] == 0.0)
    assert row["n"] == 3 and row["fe_w4"][3] == 2 and row["fe_full"][3] == 3


# --- resume only on matching settings and inputs (#118 review, finding 3) ---

from p1d_cluster_ensemble.beta_refit import needs_stored, part_settings, reuse_part  # noqa: E402


def _run(tmp_path, name="pythia-410m-step0_wiki_paragraph_long"):
    r = tmp_path / "runs" / name
    r.mkdir(parents=True)
    (r / "activations.npz").write_bytes(b"a")
    (r / "attentions.npz").write_bytes(b"b")
    return r


def _write(path, job, recs):
    path.write_text(json.dumps({"_settings": part_settings(job), "records": recs}))


def test_part_reused_only_under_its_own_settings(tmp_path):
    r = _run(tmp_path)
    job = (str(r), "/w", False, None)
    p = tmp_path / "part.json"
    _write(p, job, [{"x": 1}])
    assert reuse_part(p, job) == [{"x": 1}]
    assert reuse_part(p, (str(r), "/other", False, None)) is None      # weights
    assert reuse_part(p, (str(r), "/w", False, 465)) is None           # max offset
    (r / "attentions.npz").write_bytes(b"changed")                     # input
    assert reuse_part(p, job) is None


def test_pre_settings_part_is_refitted(tmp_path):
    r = _run(tmp_path)
    p = tmp_path / "part.json"
    p.write_text(json.dumps([{"x": 1}]))
    assert reuse_part(p, (str(r), "/w", False, None)) is None
    assert reuse_part(tmp_path / "missing.json", (str(r), "/w", False, None)) is None


def test_stored_required_unless_every_run_is_long(tmp_path):
    from pathlib import Path
    long = Path("x/pythia-410m-step0_wiki_paragraph_long")
    assert not needs_stored([long])
    assert needs_stored([long, Path("x/pythia-410m-step0_wiki_paragraph")])
