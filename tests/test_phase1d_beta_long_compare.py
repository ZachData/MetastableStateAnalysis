"""
tests/test_phase1d_beta_long_compare.py — long-vs-v1 β pairs heads by
(step, prompt, layer, head) and keeps only the long run's prompts.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.deps

from p1d_cluster_ensemble.beta_long_compare import compare, main  # noqa: E402


def _h(prompt, layer, head, beta, dedupe=False, step="step143000"):
    return {"dedupe": dedupe, "step": step, "prompt": prompt, "layer": layer,
            "head": head, "linear": beta, "fe_full": beta}


def test_pairs_by_head_and_drops_other_prompts():
    long = [_h("a_long", 2, 0, 3.0), _h("a_long", 2, 1, 1.0)]
    v1 = [_h("a", 2, 0, 2.0), _h("a", 2, 1, 2.0), _h("b", 2, 0, 100.0)]
    row = next(r for r in compare(long, v1) if r["band"] == "L1-8")
    c = row["fe_full"]
    assert c["v1"][3] == 2                     # prompt b left out
    assert c["paired_diff"][0] == 0.0 and c["n_up"] == 1
    assert sorted(r["band"] for r in compare(long, v1)) == ["L1-23", "L1-8"]


def test_nan_head_drops_from_its_side_and_the_pair_only():
    long = [_h("a_long", 10, 0, float("nan")), _h("a_long", 10, 1, 4.0)]
    v1 = [_h("a", 10, 0, 2.0), _h("a", 10, 1, 3.0)]
    c = next(r for r in compare(long, v1) if r["band"] == "L9-16")["fe_full"]
    assert c["long"][3] == 1 and c["v1"][3] == 2 and c["paired_diff"] == [1.0, 1.0, 1.0, 1]


def test_refuses_when_nothing_pairs(tmp_path):
    (tmp_path / "l.json").write_text(json.dumps({"heads": [_h("a_long", 2, 0, 1.0)]}))
    (tmp_path / "v.json").write_text(json.dumps({"heads": [_h("a", 2, 0, 1.0, step="step0")]}))
    assert main(["--long", str(tmp_path / "l.json"), "--v1", str(tmp_path / "v.json"),
                 "--out", str(tmp_path / "o.json")]) == 1
