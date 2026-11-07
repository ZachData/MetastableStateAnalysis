"""T5′ U1 (`trackT_timaeus/t5k.py`): the scorer, the dating rule, the outcome table and the plant,
on synthetic inputs, written before any output (`trackT_timaeus/design-T5p.md`)."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.deps  # t5k imports a1a
from trackT_timaeus import t5k


def test_draws_share_first_half():
    d = t5k.draws(np.arange(10, 50), seed=1, r=3, t=8)
    for k in ("rep", "ctrl", "null"):
        assert d[k].shape == (3, 16) and d[k].min() >= 10
        assert (d[k][:, :8] == d["rep"][:, :8]).all()
    assert (d["rep"][:, 8:] == d["rep"][:, :8]).all()
    assert not (d["ctrl"][:, 8:] == d["null"][:, 8:]).all()


def test_g3_scorer():
    assert t5k.g3()["pass"]


def test_pm_offsets():
    """PM reads query t + j → key j + 1 for j = 0 … t − 2 and nothing else."""
    torch = pytest.importorskip("torch")
    t = 5
    a = torch.zeros(1, 1, 2 * t, 2 * t, dtype=torch.float64)
    a[0, 0, t - 1, 0] = 1.0               # i = 0: excluded
    a[0, 0, 2 * t - 1, t] = 1.0           # i = t: excluded
    assert float(t5k.pm_of(a, t)[0, 0]) == 0.0
    a[0, 0, t + 2, 3] = 1.0               # j = 2
    assert float(t5k.pm_of(a, t)[0, 0]) == pytest.approx(1 / (t - 1))
    b = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    b[0, 0, 1, 0] = b[0, 0, 2, 1] = 1.0
    assert float(t5k.prev_of(b)[0, 0]) == pytest.approx(2 / 3)


def test_summarize_signs():
    rng = np.random.default_rng(0)
    ctrl = rng.uniform(0, 0.01, (32, 6))
    rep = ctrl.copy()
    rep[:, 4] += 0.5
    a = {"pm_rep": rep, "pm_ctrl": ctrl, "pm_null": ctrl + 0.001, "prev_ctrl": ctrl, "loss_rep": np.ones(32),
         "loss_ctrl": 2 * np.ones(32), "icl_late": np.ones(64), "icl_early": 3 * np.ones(64)}
    s = t5k.summarize(a, n_boot=50)
    assert s["I"] == pytest.approx(0.5) and s["argmax"] == 4
    assert s["I_null"] == pytest.approx(0.001)
    assert s["B"] == pytest.approx(-1.0) and s["ICL"] == pytest.approx(-2.0)
    assert s["se"]["B"] == pytest.approx(0.0)


STEPS = [0, 10, 20, 30, 40, 50, 60, 70]


def test_date_series_onset_and_half():
    x = [0.2, 0.0, 0.0, 0.05, 0.3, 0.9, 1.0, 1.0]   # min before max at 10 (first), peak 1.0 at 60
    r = t5k.date_series(STEPS, x, floor=0.1)
    assert r["base"] == 10 and r["peak"] == 60 and r["rise"] == pytest.approx(1.0)
    assert r["on_0.1"] == 40 and r["on_0.5"] == 50
    assert not r["censored"] and not r["refused"]


def test_date_series_floor_refuses():
    r = t5k.date_series(STEPS, [0, 0.01, 0.02, 0.05, 0.08, 0.09, 0.09, 0.09], floor=0.1)
    assert r["refused"]


def test_censored():
    r = t5k.date_series(STEPS, [0, 0, 0, 0, 0.1, 0.3, 0.6, 1.0], floor=0.1)
    assert r["censored"] and r["peak"] == 70
    r2 = t5k.date_series(STEPS, [0, 0, 0, 0, 0.1, 0.5, 0.95, 1.0], floor=0.1)
    assert not r2["censored"]                       # last step carries 0.05 < 0.1 of the rise


def test_se_floor():
    x = [0, 0, 0, 0, 0.1, 0.2, 0.3, 0.3]
    assert t5k.date_series(STEPS, x, floor_se=[0.1] * 8)["refused"]          # 0.3 < 4·√2·0.1
    assert not t5k.date_series(STEPS, x, floor_se=[0.01] * 8)["refused"]


def test_rho_dates():
    rho = [0.5, 1.0, 0.7, 0.2, 0.1, 0.4, 0.8, 0.9]
    rd = t5k.rho_dates(STEPS, rho)
    assert rd["peak"] == 10 and rd["trough"] == 40
    assert rd["fall"]["on_0.5"] == 30                 # 1.0 − 0.2 = 0.8 ≥ 0.45
    assert rd["rise"]["on_0.1"] == 50 and rd["rise"]["on_0.5"] == 60
    assert not rd["fall"]["censored"]
    assert [t5k.phase(s, rd) for s in (0, 10, 40, 50)] == ["early rise", "fall", "fall", "rise"]
    rd2 = t5k.rho_dates(STEPS, [0.5, 1.0, 0.7, 0.2, 0.1, 0.11, 0.4, 0.9])
    assert t5k.phase(50, rd2) == "trough"


def _i(on, cens=False, refused=False):
    return {"on_0.1": on, "censored": cens, "refused": refused}


def test_label_table():
    rr, rc = {"censored": False}, {"censored": True}
    L = t5k.label
    assert L(_i(None, refused=True), rr, None, 4369) == "no induction formed by 4369"
    assert L(_i(10), rr, -2, 4369) == "induction leads the rise"
    assert L(_i(10), rc, -3, 4369) == "induction leads the rise"          # ρ's censoring cannot move it
    assert L(_i(10, cens=True), rr, -3, 4369) == "censored, not ordered"
    assert L(_i(10), rr, 1, 4369) == "induction with the rise"
    assert L(_i(10), rc, 0, 4369) == "censored, not ordered"
    assert L(_i(10, cens=True), rr, 2, 4369) == "induction after the rise"   # I's censoring only moves it later
    assert L(_i(10), rc, 2, 4369) == "censored, not ordered"
    assert L(_i(10), rr, None, 4369) == "censored, not ordered"


def test_gap_in_grid_steps():
    assert t5k.gap(STEPS, 20, 50) == -3 and t5k.gap(STEPS, None, 50) is None


def test_noise():
    n = t5k.noise({"A0": 0, "A0p": 2, "A0pp": 1, "A1": 4, "A2": 5, "A3": None})
    assert n["span"] == 2 and n["seed_dependent"] and n["range"] == [-2, 4]
    assert n["not_told_apart"] == {"A1": True, "A2": False, "A3": False}
    assert t5k.noise({"A0": None, "A0p": 0, "A0pp": 0, "A1": 0, "A2": 0, "A3": 0})["span"] is None


def test_plant_and_g5_on_a_clean_grid():
    from trackT_timaeus import t2
    steps = [0] + [s for s in t2.GRID if s <= 4369]
    p = t5k.plant(steps, 2002)
    assert p[0] == 0 and p[steps.index(2002)] == pytest.approx(0.1)
    assert p[-1] < 0.2
    for c in t5k.PLANT_C["dodeca"]:
        assert t5k.g5_one(steps, [0.0] * len(steps), c)["pass"]
    from trackA_force import a3
    s410 = [s for s in a3.STEPS if s <= 4000]
    for c in t5k.PLANT_C["410m"]:
        assert t5k.g5_one(s410, [0.0] * len(s410), c)["pass"]
