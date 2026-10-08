"""
`tools/run/p10_r8x_exact.py` — R8x, the exact check, as `design-10.md` "R8x" fixed it.
"""
import numpy as np
import pytest

from p1d_cluster_ensemble import move_text as mt
from tools.run import p10_r8_floor as r8
from tools.run import p10_r8x_exact as r

# Tier: the module imports unit 1's runner, whose package imports scikit-learn (as move_text's tests).
pytestmark = pytest.mark.deps

CIDS = ("a|50|eod", "a|300|eod", "b|50|eod")


def _rec(js_by_cond, J0, stable=True):
    return {"groups": [{"stable": stable, "J0": J0, "opening": False}], "c_holds": False,
            "conditions": {cid: {"best_jaccard": [j]} for cid, j in js_by_cond.items()}}


def test_one_floor_per_condition_equals_r8s_bar():
    rec = _rec({"a|50|eod": 0.4, "a|300|eod": 0.6, "b|50|eod": 0.5, "a|50|nl2": 0.0}, J0=0.4)
    for bar in (0.4, 0.45, 0.55):
        assert r.classes_by_condition(rec, [{c: bar for c in CIDS}]) == r8.classes_at(rec, [bar])


def test_a_condition_with_its_own_chance_level_can_drop_a_group():
    rec = _rec({"a|50|eod": 0.4, "a|300|eod": 0.6, "b|50|eod": 0.5}, J0=0.4)
    assert r.classes_by_condition(rec, [{"a|50|eod": 0.4, "a|300|eod": 0.4, "b|50|eod": 0.4}]) == ["moves"]
    assert r.classes_by_condition(rec, [{"a|50|eod": 0.4, "a|300|eod": 0.4, "b|50|eod": 0.55}]) \
        == ["preamble-dependent"]


def test_unstable_and_floor_zero_kept():
    assert r.classes_by_condition(_rec({"a|50|eod": 1.0}, 0.4, stable=False), [{}]) == ["unstable"]
    assert r.classes_by_condition(_rec({"a|50|eod": 1.0}, 0.0), [{}]) == [mt.FLOOR_ZERO]


def test_eod_conditions_only():
    conds = [{"id": "P0", "P": 0, "join": None}, {"id": "a|50|eod", "P": 50, "join": "eod"},
             {"id": "a|50|nl2", "P": 50, "join": "nl2"}]
    assert [c["id"] for c in r.eod_conditions(conds)] == ["a|50|eod"]


def test_check_condition_refuses_count_and_jaccards():
    offs = [np.array([1, 2]), np.array([5, 6, 7])]
    gp = [np.array([1, 2]), np.array([5, 6])]
    ok = {"k": 2, "best_jaccard": [1.0, 2 / 3]}
    r.check_condition(ok, offs, gp, "x")
    with pytest.raises(r.ExactError, match="stored k"):
        r.check_condition({**ok, "k": 3}, offs, gp, "x")
    with pytest.raises(r.ExactError, match="1 of 2"):
        r.check_condition({**ok, "best_jaccard": [1.0, 0.5]}, offs, gp, "x")


def test_boot_keep_is_seeded_and_bounded():
    pp = {p: {"c3": 10, "c3x": k} for p, k in zip("abcdefg", (10, 9, 9, 8, 10, 3, 9))}
    a, b = r.boot_keep(pp, "c3x", n=500), r.boot_keep(pp, "c3x", n=500)
    assert a == b and 0.3 <= a[0] <= 58 / 70 <= a[1] <= 1.0


def _step(keep):
    return {"all": {"keep_c3x": keep, "c3": 1000, "c3x": round(1000 * keep)}}


def test_verdict_reports_the_pooled_window_beside():
    v = r.verdict({"step64": _step(0.95), "step128": _step(0.85), "step2000": _step(0.5)})
    assert v["pooled_keep_64_1000"] == pytest.approx(0.90) and v["drop_at_64_1000_holds"]


def test_chance_level_is_drawn_against_each_conditions_own_partition(monkeypatch):
    """Two conditions with different partitions of the same kept set get their own `Jc` and labels."""
    kept = np.arange(40)
    parts = {"fine": [np.array([2 * i, 2 * i + 1]) for i in range(20)], "coarse": [np.arange(20), np.arange(20, 40)]}
    monkeypatch.setattr(r.mt, "groups_of", lambda Y, frame, mcs: (None, parts[Y]))
    monkeypatch.setattr(r, "check_condition", lambda *a: None)
    rec = {"groups": [{"offsets": [0, 1]}], "conditions": {"a|50|eod": {}, "b|50|eod": {}}}
    r._G.clear()
    r._G.update(kept=kept, recs={1: rec}, sizes={1: [2]}, step="s", prompt="p", where="w",
                conds=[{"id": "a|50|eod"}, {"id": "b|50|eod"}],
                hidden={"a|50|eod": ["fine"], "b|50|eod": ["coarse"]})
    out = r._layer_job(1)
    r._G.clear()
    assert max(out["labels"]["a|50|eod"]) == 19 and max(out["labels"]["b|50|eod"]) == 1
    assert out["k"] == {"a|50|eod": 20, "b|50|eod": 2}
    want = {c: r8.chance_by_size(np.asarray(out["labels"][c]), [2], ("s", "p", 1, c))[2][0] for c in out["k"]}
    assert out["jc"]["a|50|eod"][2] == want["a|50|eod"] and out["jc"]["b|50|eod"][2] == want["b|50|eod"]
    assert want["a|50|eod"] != want["b|50|eod"]


def test_verdict_decides_on_64_to_1000():
    ok = {"step32": _step(0.5), "step64": _step(0.95), "step1000": _step(0.91), "step2000": _step(0.95)}
    v = r.verdict(ok)
    assert v["c3_stands"] and not v["drop_at_64_1000_holds"]
    v = r.verdict({**ok, "step128": _step(0.89)})
    assert v["drop_at_64_1000_holds"] and v["steps_below"] == ["step128"] and v["recommend"].startswith("(a)")
    v = r.verdict({**ok, "step4000": _step(0.85)})
    assert not v["drop_at_64_1000_holds"] and not v["c3_stands"]
