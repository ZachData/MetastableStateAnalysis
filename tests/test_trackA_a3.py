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


def test_synthetic_spread_matches_the_rows_own():
    """The gate's noise has the real row's per-step spread, in the statistic's units."""
    rng = np.random.default_rng(1)
    n, S = 35, len(STEPS)
    tau = np.geomspace(1e-6, 1e-2, S); tau[0] = 0
    w = rng.integers(400, 3000, size=(n, 1)).astype(float)
    score = rng.standard_normal((n, S)) * tau + 0.004
    pooled = np.stack([score * w, np.broadcast_to(w, (n, S))], axis=-1)
    for units, name in ((score[..., None], "mean"), (pooled, "pooled")):
        real = a3.unit_scores(units, name)
        want = np.std(real - real[:, :1], axis=0, ddof=1)
        syn = a3._synthetic(units, "up", name, rng, 4000)
        got = np.std(a3.unit_scores(syn.reshape((-1,) + syn.shape[2:]), name)
                     .reshape(4000, n, S), axis=(0, 1))
        assert np.allclose(got[1:], want[1:], rtol=0.05)


def test_gate_with_constant_noise_passes_like_the_calibration(monkeypatch):
    """With one noise size at every step the gate passes, as a3_locator.py's iid model does."""
    monkeypatch.setattr(a3, "N_GATE", 200)
    monkeypatch.setattr(a3, "N_BOOT", 200)
    rng = np.random.default_rng(2)
    u = rng.standard_normal((7, len(STEPS), 1))
    g = a3.gate_row("E4", u, rng)
    assert g["null_no_change"] >= 0.95
    assert min(e["placed_right"] for e in g["effect"]) >= 0.85


def test_bootstrap_se_of_mean_delta():
    rng = np.random.default_rng(0)
    u = np.zeros((400, len(STEPS), 1)); u[:, 5, 0] = rng.standard_normal(400)
    se = a3.boot_se(u, a3.stat_mean, 5, rng)
    assert abs(se - 1 / np.sqrt(400)) < 0.01


def test_boot_se_all_matches_one_step():
    u = np.random.default_rng(3).standard_normal((50, len(STEPS), 1))
    a = a3.boot_se_all(u, a3.stat_mean, np.random.default_rng(4), n_boot=4000)
    b = a3.boot_se(u, a3.stat_mean, 6, np.random.default_rng(5), n_boot=4000)
    assert a[0] == 0 and abs(a[6] - b) / b < 0.06


def test_planted_shape_shifts_each_statistic_exactly():
    """Why one bootstrap per noise draw serves every amplitude (gate_row's docstring)."""
    rng = np.random.default_rng(6)
    n, S = 35, len(STEPS)
    w = rng.integers(400, 3000, size=(n, 1)).astype(float)
    z = rng.standard_normal((n, S)); sh = np.linspace(0, 2, S)
    for name in ("mean", "median", "pooled"):
        if name == "pooled":
            mk = lambda y: np.stack([y * w, np.broadcast_to(w, (n, S))], axis=-1)
        else:
            mk = lambda y: y[..., None]
        st = a3.STATS[name]
        assert np.allclose(st(mk(z + sh)), st(mk(z)) + sh)


def test_censored_and_rederived_low_power():
    x = _step_at(STEPS.index(4000))
    r = a3.place(x, "up", sigma=0.01)
    assert a3.censored(r) is True
    r2 = a3.place(_step_at(STEPS.index(256)), "up", sigma=0.1)    # ref / σ = 10
    assert a3.censored(r2) is False
    gate = {"effect": [{"to_index": STEPS.index(256), "power": 12}]}
    assert a3.low_power(r2, gate) is True
    gate["effect"][0]["power"] = 6
    assert a3.low_power(r2, gate) is False
    assert a3.low_power({"reading": "no change"}, gate) is None


def test_gate_floor_is_rederived_and_checked(monkeypatch):
    monkeypatch.setattr(a3, "N_GATE", 300)
    monkeypatch.setattr(a3, "N_BOOT", 200)
    rng = np.random.default_rng(7)
    g = a3.gate_row("E4", rng.standard_normal((7, len(STEPS), 1)), rng)
    assert 1.5 < g["floor"] < 6 and g["null_no_change"] >= 0.9
    assert all(e["power"] is not None and e["power"] <= 18 for e in g["effect"])
    assert g["effect"][0]["curve"]["2"] < g["effect"][0]["curve"]["9"]   # power rises
