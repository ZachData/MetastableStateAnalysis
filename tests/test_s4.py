"""S4 U1's pure parts (`trackS_subspace/s4.py`): the date rule equals `a3.d4a` on Pythia's grid
(design-S4.md "Dates"), the grids and windows, the outcome rows, the gate's steps."""
import numpy as np
import pytest

pytestmark = pytest.mark.pure

from trackA_force import a3
from trackA_force.a1a import STEPS
from trackS_subspace import s4


@pytest.mark.parametrize("seed", range(200))
def test_half_rise_equals_a3_d4a_on_pythias_grid(seed):
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 60, size=len(STEPS)).astype(float)
    if seed % 5 == 0:
        v[STEPS.index(1000)] = v[STEPS.index(2000)] = 0       # zero plateaus too
    ref = a3.d4a(v)
    got = s4.half_rise(v, (1000, 2000, 4000), STEPS)
    assert got == ref


def test_toy_grid_and_window():
    g = s4.TOY_GRID
    assert g[:26] == tuple(range(1, 27))
    assert s4.TOY_WINDOW_U1 == (526, 1035, 2100)
    assert s4.TOY_WINDOW == (18540, 38140, 75000)           # Blocked 149 (ii) (a)
    assert set(s4.TOY_WINDOW) | set(s4.TOY_WINDOW_U1) <= set(g) and g[-1] == 75000
    assert s4.DATED_TOYS == ("R2",)                           # Blocked 149 (i) (a)
    assert set(g) <= set(range(1, 27)) | set(s4.LOG_GRID) | {2100}
    assert len(g) == 117                     # the design's "~120 per depth"
    assert s4.R1_GRID[0] == 0 and all(s % 10 == 0 for s in s4.R1_GRID)
    assert s4.R1_WINDOW == (530, 1050, 2100) and set(s4.R1_WINDOW) <= set(s4.R1_GRID)
    assert s4.H_GRID[0] == 100 and s4.H_GRID[-1] == 4000 and len(s4.H_GRID) == 21
    assert set(s4.H_WINDOW) <= set(s4.H_GRID)


def test_nearest_tie_goes_later():
    assert s4.nearest(525, range(0, 1000, 10)) == 530
    assert s4.nearest(1050, (1035, 1083)) == 1035


def test_date_reading_outcome_rows():
    g = (1, 2, 3, 4, 5)
    w = (3, 4, 5)
    assert s4.date_reading([0, 0, 0, 0, 0], w, g)["no_break"]                 # V = 0 everywhere
    assert s4.date_reading([10, 10, 10, 10, 10], w, g)["no_break"]            # at its plateau from the first
    r = s4.date_reading([0, 1, 6, 10, 10], w, g)
    assert not r["no_break"] and r["step"] == 3 and r["plateau"] == 10


def test_top_share():
    assert s4.top_share([1, 1, 2, 3]) == 0.5


def test_gate_steps():
    g = (1, 2, 4, 8, 16, 32, 64)
    assert s4.gate_steps(8, g, (16, 32, 64)) == (4, 16)
    assert s4.gate_steps(None, g, (16, 32, 64)) == (16, 64)
    assert s4.gate_steps(64, g, (16, 32, 64)) == (32, 64)


@pytest.mark.parametrize("arm", ["R3", "R4", "R5", "R6"])
def test_description_only_toys_refuse_the_gate(tmp_path, arm):
    class A:
        out = str(tmp_path)
    A.arm = [arm]
    with pytest.raises(s4.S4Refused, match="description only"):
        s4.gate(A)


def test_reading_refuses_a_gate_record_on_another_window(tmp_path):
    (tmp_path / "R2").mkdir()
    (tmp_path / "R2" / "gate.json").write_text('{"meta": {"window": [526, 1035, 2100]}}')
    with pytest.raises(s4.S4Refused, match="window"):
        s4.arm_reading(tmp_path, "R2")


def test_h_reserve_refuses_induction(tmp_path):
    with pytest.raises(s4.S4Refused):
        s4.induction_reading(tmp_path, "H_vanilla")


def test_rise_fall_rule():
    g = list(range(10, 110, 10))
    r = s4.rise_fall([0, 0, 1, 4, 5, 5, 4, 1, 0, 0], g)
    assert (r["peak"], r["rise"], r["fall"], r["peak_step"]) == (5.0, 40, 80, 50)
    assert s4.rise_fall([0] * 10, g)["no_rise"]
    assert s4.rise_fall([5] * 10, g)["no_rise"] and s4.rise_fall([5] * 10, g)["no_fall"]
    up = s4.rise_fall([0, 0, 0, 2, 4, 4, 4, 4, 4, 4], g)
    assert up["rise"] == 40 and up["no_fall"]           # m3 at 40 is median(0, 2, 4) = 2 = P/2


def test_b_ceiling_seen_caps_the_dose():
    sh = {"0.25": 0.1, "0.5": 0.3, "1": 0.45, "2": 0.9, "ceiling": 1.0}
    r = s4.b_ceiling_seen(0.2, sh, 0.0, 1.0, 0.4, 0.05)
    assert r["pass"] and r["dose_seen"] == 1.0
    assert not s4.b_ceiling_seen(0.2, {**sh, "1": 0.39}, 0.0, 1.0, 0.4, 0.05)["pass"]   # 2 and the ceiling don't count
    assert not s4.b_ceiling_seen(0.2, sh, 0.06, 1.0, 0.4, 0.05)["pass"]                 # the null fails
