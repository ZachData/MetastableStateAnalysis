"""
tests/test_phase1d_position.py — the position check on admission's groups
(`p1d_cluster_ensemble/position_check.py`, `design-1d.md` build step 2):
a run of text is flagged, a random group is flagged at about alpha, and the
first-token and contiguity flags read the kept positions, not raw indices.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.deps

from p1d_cluster_ensemble.position_check import (
    ALPHA, _near_share, check_file, group_position, random_groups, summarise,
)

#: Kept first occurrences skew early, as in the real prompts.
KEPT = np.unique(np.concatenate([np.arange(60), np.arange(60, 400, 3)]))


def _null(size, n=2000, seed=0):
    return random_groups(KEPT, size, n, np.random.default_rng(seed))


def test_near_share_hand_example():
    assert _near_share(np.array([0, 2, 10])) == pytest.approx(1 / 3)
    assert _near_share(np.array([[0, 1], [0, 9]])).tolist() == [1.0, 0.0]


def test_a_run_of_text_is_positional_and_contiguous():
    g = group_position(range(100, 110), KEPT, _null(10))
    assert g["contiguous"] and not g["has_first_kept"]
    assert g["p_near"] <= ALPHA and g["p_span"] <= ALPHA


def test_first_and_contiguous_read_kept_indices():
    # Kept indices 0..4 are positions 0..4: the opening, contiguous.
    g = group_position([3, 0, 4, 1, 2], KEPT, _null(5))
    assert g["has_first_kept"] and g["contiguous"] and g["span"] == 4
    # Indices 60, 61 are positions 60 and 63: contiguous kept tokens, 3 apart.
    g = group_position([60, 61], KEPT, _null(2))
    assert g["contiguous"] and g["span"] == 3 and g["near_share"] == 1.0


@pytest.mark.parametrize("size", [2, 6, 20])
def test_random_groups_flagged_at_most_alpha(size):
    rng = np.random.default_rng(size)
    null = _null(size, seed=99)
    flags = [group_position(rng.choice(KEPT.size, size, replace=False), KEPT, null)
             for _ in range(400)]
    # Rank p on a discrete statistic is conservative: at most alpha, plus noise.
    assert np.mean([f["p_near"] <= ALPHA for f in flags]) <= ALPHA + 0.03
    assert np.mean([f["p_span"] <= ALPHA for f in flags]) <= ALPHA + 0.03


def test_check_file_and_summarise():
    rec = {"run_dir": "r", "step": "step0", "prompt": "p", "layer": 3, "keep": KEPT.tolist(),
           "info": {"frame": "centred"},
           "arms": {"2": {"groups": [
               {"size": 8, "members": list(range(8)), "admitted_excess": True,
                "admitted_log_life": True},
               {"size": 3, "members": [5, 90, 150], "admitted_excess": False,
                "admitted_log_life": False}]}}}
    rows = check_file({"records": [rec]}, n_draws=500)
    assert [r["has_first_kept"] for r in rows] == [True, False]
    summ = {(s["admitted"]): s for s in summarise(rows)}
    assert summ[True]["n"] == 1 and summ[True]["positional"] == 1.0
    assert summ[False]["has_first_kept"] == 0.0
    assert summ[True]["mostly_near"] == 1.0 and summ[False]["mostly_near"] == 0.0


def test_cli_writes_report_without_attentions(tmp_path):
    import json
    from p1d_cluster_ensemble.admit import _job
    from p1d_cluster_ensemble.position_check import main
    d = tmp_path / "2026-01-01_00-00-00" / "pythia-410m-step143000_wiki_paragraph"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 12))
    X[-8:] = 5 * np.eye(12)[1] + 0.1 * rng.standard_normal((8, 12))
    np.savez(d / "activations.npz", activations=np.stack([X, X]).astype(np.float32))
    (d / "geometry.json").write_text(json.dumps({"tokens": [f"t{i}" for i in range(40)]}))
    real = tmp_path / "real.json"
    real.write_text(json.dumps({"records": [_job((str(d), 1, "raw", 9, 0, False, 0))]}))
    out = tmp_path / "pos.json"
    assert main(["--real", str(real), "--out", str(out), "--n-draws", "50"]) == 0
    rep = json.loads(out.read_text())
    assert rep["runs"][str(d)]["attention_tv_to_uniform"].startswith("unavailable")
    assert rep["groups"] and out.with_suffix(".txt").exists()
