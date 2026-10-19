"""A3's locator (`trackA_force/a3.py`, design-A3.md §4): pure parts, synthetic."""
import numpy as np
import pytest

from trackA_force import a3

pytestmark = pytest.mark.deps  # a1a's imports
from trackA_force.a1a import STEPS


def _step_at(i, amp=1.0, sign=1.0):
    x = np.zeros(len(STEPS)); x[i:] = sign * amp
    return x


def test_half_rise_places_first_step_past_half():
    i = STEPS.index(512)
    r = a3.place(_step_at(i), "up", sigma=0.01)
    assert r["reading"] == "located" and r["step"] == 512 and r["low_power"] is False


def test_down_row_is_signed():
    r = a3.place(_step_at(STEPS.index(64), sign=-1.0), "down", sigma=0.01)
    assert r["step"] == 64
    assert a3.place(_step_at(STEPS.index(64)), "down", sigma=0.01)["reading"] == "no change"


def test_floor_and_low_power():
    x = _step_at(STEPS.index(256))
    assert a3.place(x, "up", sigma=1 / 3.9)["reading"] == "no change"
    r = a3.place(x, "up", sigma=1 / 5.0)
    assert r["reading"] == "located" and r["low_power"] is True


def test_reference_ignores_steps_after_4000():
    x = _step_at(STEPS.index(8000), amp=10.0)
    x[STEPS.index(128):STEPS.index(8000)] = 1.0
    assert a3.place(x, "up", sigma=0.01)["step"] == 128


def test_turning_point_is_the_minimum():
    x = np.zeros(len(STEPS))
    x[STEPS.index(64):] = -1.0; x[STEPS.index(128)] = -2.0; x[STEPS.index(1000):] = -3.0
    r = a3.place(x, "down", sigma=0.01, kind="turn")
    assert r["step"] == 1000  # the minimum over <= 4000 is the late fall
    x[STEPS.index(1000):] = -0.5
    assert a3.place(x, "down", sigma=0.01, kind="turn")["step"] == 128


def test_lr_anchor_is_1000_and_d4a():
    e10 = np.array([a3.lr(s) for s in STEPS])
    assert a3.place(e10, "up", None, floor=None)["step"] == 1000
    v = np.zeros(len(STEPS)); v[STEPS.index(256)] = 10; v[STEPS.index(512):] = 57
    assert a3.d4a(v)["step"] == 512


def test_bootstrap_se_of_mean_delta():
    rng = np.random.default_rng(0)
    u = np.zeros((400, len(STEPS), 1)); u[:, 5, 0] = rng.standard_normal(400)
    se = a3.boot_se(u, a3.stat_mean, 5, rng)
    assert abs(se - 1 / np.sqrt(400)) < 0.01
