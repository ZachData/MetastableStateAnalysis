"""T2, V on Dodecahedron (`trackT_timaeus/t2.py`, design-T2.md, frozen 2026-10-29): pure parts."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackT_timaeus import t2


def test_the_frozen_constants():
    assert t2.REF == {"u": (256, 512), "c": (512, 1000)}
    assert len(t2.GRID) == 52 and t2.GRID[0] == 1 and t2.GRID[-1] == 8158 and t2.SCALE_STEP not in t2.GRID
    assert t2.PLATEAU_EARLY == (1253, 1465, 1713, 2002, 2340, 2735, 3198, 3738)
    assert t2.PLATEAU_LATE == (5108, 5970, 6979, 8158) and t2.EARLY == (1, 2)
    assert t2.TOKENS_524K is None and t2.scored_hyps() == ("instantaneous", "cumulative", "fraction", "step")
    assert t2.scored_hyps(True)[-1] == "tok524k"


def test_maps_at_a_point():
    assert abs(t2.to_dodeca("cumulative", 1000) - 244.8) < 0.05
    assert abs(t2.to_dodeca("instantaneous", 1000) - 59.94) < 0.01
    assert t2.to_dodeca("tok4M", 512) == 256 and t2.to_dodeca("tok524k", 512) == 2048
    assert abs(t2.to_dodeca("fraction", 1430) - 1000) < 1e-9


# design-T2.md's outcome table, written out from the design (not from `predicted`)
TABLE_U = {"instantaneous": {8, 10, 12, 14, 16, 19, 22, 25, 30, 35},
           "cumulative": {35, 40, 47, 55, 65, 75, 88, 103, 121, 141},
           "fraction": {103, 121, 141, 165, 193, 225, 263, 308, 359},
           "step": {141, 165, 193, 225, 263, 308, 359, 420, 491, 574},
           "tok4M": {65, 75, 88, 103, 121, 141, 165, 193, 225, 263},
           "tok524k": {574, 671, 785, 917, 1072, 1253, 1465, 1713, 2002, 2340}}
TABLE_C = {"instantaneous": {16, 19, 22, 25, 30, 35, 40, 47, 55, 65},
           "cumulative": {65, 75, 88, 103, 121, 141, 165, 193, 225, 263},
           "fraction": {193, 225, 263, 308, 359, 420, 491, 574, 671, 785},
           "step": {263, 308, 359, 420, 491, 574, 671, 785, 917, 1072},
           "tok4M": {141, 165, 193, 225, 263, 308, 359, 420, 491, 574},
           "tok524k": {1072, 1253, 1465, 1713, 2002, 2340, 2735, 3198, 3738, 4369}}


@pytest.mark.parametrize("hyp", t2.HYPS)
def test_predicted_is_the_designs_outcome_table(hyp):
    assert t2.predicted(t2.REF["u"], hyp) == frozenset(TABLE_U[hyp])
    assert t2.predicted(t2.REF["c"], hyp) == frozenset(TABLE_C[hyp])


def test_consistent():
    hyps = t2.scored_hyps(False)
    assert t2.consistent([141], "u", hyps) == {"cumulative", "fraction", "step"}
    assert t2.consistent([65], "u", hyps) == {"cumulative"}
    assert t2.consistent([1072], "u", hyps) == frozenset()
    assert t2.consistent([1072], "u", t2.scored_hyps(True)) == {"tok524k"}
    assert t2.tokens_beside([65], "u", hyps) == {"tok4M": True, "tok524k": False}


def _v(at, early=0.0, plateau=40.0, late=None):
    v = np.zeros(len(t2.GRID))
    v[t2.GRID.index(at):] = plateau
    if late is not None:
        v[t2.GRID.index(5108):] = late
    v[:2] = early
    return v


def test_count_reading_two_plateaus():
    r = t2.count_reading(_v(at=263), tok=False)
    assert r["primary"] == "early" and r["dates"] == [263] and not r["unreadable"]
    assert r["plateaus"]["early"]["sustained"] == 263
    v = _v(at=65)
    v[t2.GRID.index(263):] = 100.0                         # a second rise: the late plateau reads it
    v[t2.GRID.index(1253):t2.GRID.index(5108)] = 60.0      # early plateau 60, half 30: reads 65's 40
    r = t2.count_reading(v, tok=False)
    assert r["plateaus"]["early"]["step"] == 65 and r["plateaus"]["late"]["step"] == 263
    assert r["dates"] == [65, 263]                         # the union keeps both
    assert t2.count_reading(_v(at=263, early=20.0), tok=False)["unreadable"]
    assert t2.count_reading(np.zeros(len(t2.GRID)), tok=False)["unreadable"]
    v = _v(at=263, plateau=0.0, late=40.0)                 # early plateau 0: unreadable, late not consulted
    assert t2.count_reading(v, tok=False)["unreadable"] and not t2.count_reading(v, tok=True)["unreadable"]


def test_model_reading_and_joint():
    vu, gate = _v(at=263), {"0.0001": True, "0.0003": True, "0.001": False}
    vcs = {"0.0001": _v(at=574), "0.0003": _v(at=917), "0.001": _v(at=65)}
    r = t2.model_reading(vu, vcs, gate, tok=False)
    assert r["uncentred"]["consistent"] == ["fraction", "step"]
    assert r["centred"]["dates"] == [574, 917] and not r["centred"]["agree"]
    assert r["centred"]["consistent"] == ["fraction", "step"]
    j = t2.joint({"32m": r, "67m": r}, tok=False)
    assert j["label"] == "consistent with fraction and step" and j["readable_dates"] == 4
    assert j["with_3b"] == ["fraction", "step"] and j["reading_3c"] == "consistent with step"

    def m(u, c):
        return {"uncentred": {"consistent": u}, "centred": {"consistent": c}}

    assert t2.joint({"a": m(["instantaneous"], ["cumulative"])}, tok=False)["label"] == "neither"
    assert t2.joint({"a": m(None, None)}, tok=False)["label"] == "unreadable"
    j = t2.joint({"a": m(["instantaneous", "cumulative"], None)}, tok=False)
    assert j["unreadable"] == ["a centred"] and j["with_3b"] == ["cumulative"]


@pytest.mark.parametrize("tok", [False, True])
def test_planted_effect_passes(tok):
    r = t2.planted_effect(tok)
    assert r["pass"]
    assert r["per_schedule"]["cumulative"]["cells"]["uncentred"]["dates"] == [65]
    assert r["per_schedule"]["step"]["cells"]["uncentred"]["excluded"][:2] == ["instantaneous", "cumulative"]
    if tok:
        assert r["per_schedule"]["tok524k"]["joint"] == "consistent with tok524k"


def test_planted_null_is_mostly_neither():
    r = t2.planted_null(np.random.default_rng(0), n=500, tok=False)
    assert r["label_share"]["neither"] > 0.9 and abs(sum(r["label_share"].values()) - 1) < 1e-9


def test_null_matched_refuses_a_zero_uncentred_rate():
    with pytest.raises(t2.T2Refused):
        t2.null_matched_tol([-1e-2] * 10, [1e-3] * 10)
    assert t2.null_matched_tol([2e-3] * 5 + [-1e-2] * 5, list(np.linspace(0, 1e-3, 10))) > 0
    with pytest.raises(t2.T2Refused):                     # the quantile below 0
        t2.null_matched_tol([2e-3] * 5 + [-1e-2] * 5, list(np.linspace(-1e-3, -1e-4, 10)))
