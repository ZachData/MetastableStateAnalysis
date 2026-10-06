"""
`tools/run/p10_r8_floor.py` — R8's own-floor check on "moves", as `design-10.md` "R8" fixed it.
"""
import numpy as np
import pytest

from p1d_cluster_ensemble import move_text as mt
from tools.run import p10_r8_floor as r

# Tier: the module imports unit 1's runner, whose package imports scikit-learn (as move_text's tests).
pytestmark = pytest.mark.deps


def test_best_jaccards_matches_move_text():
    lab = np.array([0, 0, 1, 1, 1, -1, -1, 2, 2])
    groups = [set(np.flatnonzero(lab == g).tolist()) for g in range(3)]
    rng = np.random.default_rng(0)
    draws = np.stack([rng.choice(lab.size, size=3, replace=False) for _ in range(200)])
    got = r.best_jaccards(draws, lab)
    want = [mt.best_jaccard(set(d.tolist()), groups) for d in draws]
    assert np.allclose(got, want)


def test_best_jaccards_all_noise_is_zero():
    assert r.best_jaccards(np.array([[0, 1]]), np.array([-1, -1, -1])).tolist() == [0.0]


def test_chance_bar_is_the_discrete_95():
    js = np.array([0.0] * 90 + [1 / 3] * 6 + [1.0] * 4)
    # share >= 1/3 is 0.10 > 0.05; share >= 1 is 0.04 <= 0.05
    assert r.chance_bar(js) == 1.0
    assert r.chance_bar(np.array([0.0] * 95 + [0.5] * 5)) == 0.5
    assert r.chance_bar(np.ones(10)) == float("inf")


def test_share_at_least():
    js = np.sort(np.array([0.0, 0.2, 0.2, 0.5]))
    assert r.share_at_least(js, 0.2) == 0.75 and r.share_at_least(js, 0.6) == 0.0


def test_chance_by_size_is_seeded_by_key():
    lab = np.array([0, 0, 1, 1, -1, -1, 2, 2, 2, -1] * 5)
    a = r.chance_by_size(lab, [2, 3], ("s", "p", 1), n=300)
    b = r.chance_by_size(lab, [2, 3], ("s", "p", 1), n=300)
    c = r.chance_by_size(lab, [2], ("s", "p", 2), n=300)
    assert a[2][0] == b[2][0] and np.array_equal(a[3][1], b[3][1])
    assert not np.array_equal(a[2][1], c[2][1])


def _rec(js_by_cond, J0, stable=True):
    return {"groups": [{"stable": stable, "J0": J0, "opening": False}], "c_holds": False,
            "conditions": {cid: {"best_jaccard": [j]} for cid, j in js_by_cond.items()}}


def test_classes_at_own_floor_equals_group_classes():
    rec = _rec({"a|50|eod": 0.4, "a|300|eod": 0.6, "b|50|eod": 0.5, "a|50|nl2": 0.0}, J0=0.4)
    assert r.classes_at(rec, [0.4]) == mt.group_classes(rec) == ["moves"]


def test_chance_aware_bar_can_only_remove():
    rec = _rec({"a|50|eod": 0.4, "a|300|eod": 0.6, "b|50|eod": 0.5}, J0=0.3)
    assert r.classes_at(rec, [0.3]) == ["moves"]
    assert r.classes_at(rec, [max(0.3, 0.45)]) == ["preamble-dependent"]
    assert r.classes_at(rec, [max(0.3, 0.7)]) == ["context-bound"]


def test_classes_at_keeps_unstable_and_floor_zero():
    assert r.classes_at(_rec({"a|50|eod": 1.0}, J0=0.5, stable=False), [0.5]) == ["unstable"]
    assert r.classes_at(_rec({"a|50|eod": 1.0}, J0=0.0), [0.9]) == [mt.FLOOR_ZERO]


def test_pool_and_verdict():
    rows = [{"c3": True, "c3c": True, "J0": 0.1, "Jc": 0.2, "q_J0": 0.3},
            {"c3": True, "c3c": False, "J0": 0.1, "Jc": 0.2, "q_J0": 0.4},
            {"c3": True, "c3c": True, "J0": 0.5, "Jc": 0.2, "q_J0": 0.0},
            {"c3": False, "c3c": False, "J0": 0.5, "Jc": 0.2, "q_J0": 0.0}]
    p = r.pool(rows)
    assert (p["c2"], p["c3"], p["c3c"], p["J0_below_Jc"], p["dropped_among_lenient"]) == (4, 3, 2, 2, 1)
    v = r.verdict({"step32": {"all": {"keep": 0.1}}, "step64": {"all": {"keep": 0.95}},
                   "step512": {"all": {"keep": 0.85}}})
    assert v["steps_below"] == ["step512"] and not v["c3_stands"] and v["min_keep"] == 0.85
