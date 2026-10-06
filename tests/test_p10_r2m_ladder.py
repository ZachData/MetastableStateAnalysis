"""
`tools/run/p10_r2m_ladder.py` — R2m: F12 against its matched control (each step's labels on step
0's activations), the raw sign beside, and §1.5 on F1 ∧ matched below.
"""
import json

import pytest

from tools.run import p10_r2m_ladder as lad
from tools.run.p10_r1_ladder import LadderError

# Tier: stdlib only -- runs in `scripts/check.sh pure`.
pytestmark = pytest.mark.pure


def _u(layer, beta, stat, p=0.5):
    return {"layer": layer, "beta": beta, "stat": stat, "p": p}


def test_paired_matches_by_unit_and_refuses_a_missing_control_or_a_step_0_gap():
    t = {"0|a": [_u(1, 1.0, 0.3)], "64|a": [_u(1, 1.0, -0.2), _u(1, 2.0, 0.1)]}
    c = {"0|a": [_u(1, 1.0, 0.3)], "64|a": [_u(1, 2.0, 0.4, 0.01), _u(1, 1.0, 0.5, 0.02)]}
    got = lad.paired(t, c)
    assert got[64] == [("a", -0.2, 0.5, 0.02), ("a", 0.1, 0.4, 0.01)]
    with pytest.raises(LadderError, match="no control unit"):
        lad.paired(t, {"0|a": c["0|a"], "64|a": c["64|a"][:1]})
    with pytest.raises(LadderError, match="at step 0"):
        lad.paired(t, {**c, "0|a": [_u(1, 1.0, 0.31)]})


def test_step_cells_label_the_gap_against_the_control_and_keep_the_raw_sign_beside():
    units = {64: [("a", -0.3, 0.2, 0.01), ("a", -0.1, 0.0, 0.03), ("b", 0.4, 0.3, 0.5)]}
    x = lad.step_cells(units)[64]
    assert x["trained"] == 0.0 and x["control"] == pytest.approx(0.1667, abs=1e-4)
    assert x["delta"] == pytest.approx(-0.1667, abs=1e-4) and x["label"] == "below control"
    assert x["raw"] == "as rest"                    # trained mean 0: no raw sign
    assert (x["prompts_negative"], x["prompts"]) == (1, 2) and x["control_median_p"] == 0.03
    assert lad.step_cells({0: [("a", 0.2, 0.2, 0.01), ("b", 0.1, 0.1, 0.01)]})[0]["sign_p"] == 1.0
    assert lad.word(0.06, "control") == "above control" and lad.word(None, "rest") == "unavailable"


def test_parked_needs_f1_negative_and_f12m_below_control():
    def rec(stat_t, stat_c):
        return {"runs": {"0|a": [_u(1, 1.0, 0.0)], "64|a": [_u(1, 1.0, stat_t)]}}, \
               {"runs": {"0|a": [_u(1, 1.0, 0.0)], "64|a": [_u(1, 1.0, stat_c)]}}
    t, c = rec(-0.4, 0.2)
    f1 = {0: {"n": 1, "mean": 0.0, "median_p": 1.0}, 64: {"n": 1, "mean": -0.5, "median_p": 0.001}}
    data = {"recs": {"f12": {k: t for k in lad.COLUMNS}, "f12m": {k: c for k in lad.COLUMNS},
                     "f1": {k: f1 for k in lad.COLUMNS}}}
    cols = lad.columns(data)
    assert cols["§1.5"]["c3"][("§1.5", "all")] == {0: (None, "not"), 64: (None, "parked")}
    assert cols["F12 raw"]["c3"][("F12 raw", "all")][64] == (-0.4, "below rest")
    t2, c2 = rec(-0.4, -0.42)                        # below the rest, but as its control
    data["recs"]["f12"]["c3"], data["recs"]["f12m"]["c3"] = t2, c2
    assert lad.columns(data)["§1.5"]["c3"][("§1.5", "all")][64] == (None, "not")


def test_load_refuses_a_control_not_read_on_step_0(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "summary.json").write_text("{}")
    r2, m = tmp_path / "r2", tmp_path / "m"
    r2.mkdir(), m.mkdir()
    for col in lad.COLUMNS:
        ls = {"labels": str(labels), "column": col, "summary_sha256": "x"}
        base = {"label_source": ls, "records_readable": {"0": {"n": 1, "readable": 1}}, "by_step": {}, "runs": {}}
        for f in (r2 / f"f12_{col}.json", r2 / f"f1_{col}.json"):
            f.write_text(json.dumps(base))
        (m / f"f12m_{col}.json").write_text(json.dumps({**base, "activations_step": 0 if col != "c3" else 512}))
    with pytest.raises(LadderError, match="not 0"):
        lad.load(m, r2, labels)


def test_sign_p_is_the_exact_two_sided_binomial():
    assert lad.sign_p(0, 7) == pytest.approx(2 / 128)
    assert lad.sign_p(1, 7) == pytest.approx(2 * 8 / 128)       # 0.125, /challenge-pr on #149
    assert lad.sign_p(7, 7) == lad.sign_p(0, 7)
    assert lad.sign_p(3, 6) == 1.0 and lad.sign_p(0, 0) == 1.0
