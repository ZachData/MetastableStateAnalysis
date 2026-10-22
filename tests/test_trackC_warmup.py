"""C2 unit 3c, the warmup arm (`trackC_writeup/c2_warmup.py`, design-C2.md Blocked 92 (ii), 98 (c)): pure parts."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackC_writeup import c2_warmup as w


def test_the_reference_is_410m_seeds_dates_and_the_grid_is_shared():
    assert w.REF == {"u": (256, 512), "c": (512, 1000)}
    assert w.GRID[0] == 0 and 300 not in w.GRID and 30 not in w.GRID      # expanse lacks both
    assert w.PLATEAU == (4000, 6000, 8000, 12000, 16000) and w.EARLY == (0, 10)
    assert w.SCALE_STEP not in w.GRID
    assert w.TAU_ARMS == (2.45e-3, 1e-4, 3e-4, 3.42e-3)                  # frozen from the preflight


@pytest.mark.parametrize("kind,hyp,want", [
    ("u", "instantaneous", {20, 40}), ("u", "cumulative", {40, 60, 80, 100, 150}),
    ("u", "step", {150, 200, 250, 350, 400, 450, 500, 550}),
    ("u", "tokens", {550, 600, 700, 800, 900, 1000, 1100, 1200, 1400, 1600, 1800, 2000, 2400}),
    ("c", "cumulative", {60, 80, 100, 150, 200, 250, 350}),
    ("c", "step", {350, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 1100}),
    ("c", "fraction", {800, 900, 1000, 1100, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 3200}),
])
def test_predicted_grid_steps(kind, hyp, want):
    assert w.predicted(w.REF[kind], hyp) == frozenset(want)


def test_maps_at_a_point():
    assert w.to_gpt2("step", 512, 15) == 527 and w.to_gpt2("tokens", 512, 15) == 2048
    assert abs(w.to_gpt2("fraction", 1430, 0) - 4000) < 1e-9
    assert abs(w.to_gpt2("instantaneous", 1430 * 2, 0) - 4000) < 1e-6   # lr_p = 6e-4 at 2860 (outside, as a check)
    n = w.to_gpt2("cumulative", 512, 0)
    assert abs(w.g_cum(n) - w.p_cum(512)) < 1e-9


def test_consistent_and_joint():
    assert w.consistent([200], "u") == {"step"}
    assert w.consistent([80], "u") == {"cumulative"}
    assert w.consistent([1000], "u") == {"fraction", "tokens"}
    assert w.consistent([350, 1000], "c") == {"cumulative", "step", "fraction"}

    def seed(u, c):
        return {"uncentred": {"consistent": u}, "centred": {"consistent": c}}

    j = w.joint({"a": seed(["step"], ["cumulative", "step"]), "b": seed(["step", "fraction"], ["step"])})
    assert j["survivors"] == ["step"] and j["label"] == "consistent with step" and j["readable_dates"] == 4
    j = w.joint({"a": seed(["instantaneous", "cumulative"], None)})
    assert j["unreadable"] == ["a centred"] and j["with_3b"] == ["cumulative"]
    assert w.joint({"a": seed(["step"], ["tokens"])})["label"] == "neither"
    assert w.joint({"a": seed(None, None)})["label"] == "unreadable"


def _v(early=0.0, at=None, plateau=40.0):
    v = np.zeros(len(w.GRID))
    v[:2] = early
    i = w.GRID.index(at)
    v[i:] = plateau
    return v


def test_date_reading():
    r = w.date_reading(_v(at=450))
    assert r["step"] == 450 and not r["unreadable"] and r["plateau"] == 40
    assert r["beside"]["sustained"] == 450
    assert w.date_reading(_v(early=20.0, at=450))["unreadable"]        # an EARLY count at half
    assert w.date_reading(np.zeros(len(w.GRID)))["unreadable"]           # no plateau
    v = _v(at=450)
    v[w.GRID.index(100)] = 25.0                                            # one spike first
    r = w.date_reading(v)
    assert r["step"] == 100 and r["beside"]["sustained"] == 450


def test_seed_reading_reads_gated_arms_only():
    vu = _v(at=200)
    vcs = {"0.0001": _v(at=500), "0.0003": _v(at=1000), "0.001": _v(at=80)}
    r = w.seed_reading(vu, vcs, {"0.0001": True, "0.0003": True, "0.001": False})
    assert r["uncentred"]["consistent"] == ["step"]
    assert r["centred"]["dates"] == [500, 1000] and not r["centred"]["agree"]
    assert r["centred"]["consistent"] == ["fraction", "step"]
