"""C4's description beside (`trackC_writeup/c4_beside.py`), on planted series. No torch, no data."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports

from trackC_writeup import c4_beside as b
from trackC_writeup import c4_train as ct


def test_peak_half_fall_and_minimum_on_a_rise_fall_rise():
    grid = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    v = [0.2, 0.5, 0.9, 0.8, 0.6, 0.3, 0.25, 0.35, 0.4]
    s = b.shape_of(v, grid, "peak")
    assert s["peak"] == {"step": 2, "value": 0.9}
    assert s["half"]["step"] == 8                    # midpoint (0.9 + 0.4) / 2 = 0.65; 0.6 first below
    assert s["min_after"] == {"step": 32, "value": 0.25, "censored": False}
    assert s["min_all"]["step"] == 0
    assert s["end"] == 0.4


def test_minimum_at_the_end_is_censored_and_a_peak_at_the_end_has_no_half():
    grid = [0, 1, 2, 4]
    assert b.shape_of([0.1, 0.5, 0.3, 0.2], grid, "peak")["min_after"]["censored"]
    s = b.shape_of([0.1, 0.2, 0.3, 0.4], grid, "peak")
    assert s["half"] is None and s["peak"]["step"] == 4


def test_trough_is_the_mirror():
    grid = [0, 1, 2, 4, 8]
    s = b.shape_of([50, 20, 10, 40, 30], grid, "trough")
    assert s["trough"]["step"] == 2 and s["half"]["step"] == 4      # midpoint 20; 40 first above
    assert s["max_after"] == {"step": 4, "value": 40.0, "censored": False}
    assert s["max_all"]["step"] == 0


def test_unpopulated_series_refuses():
    with pytest.raises(b.C4BesideRefused):
        b.shape_of([0.1, np.nan], [0, 1], "peak")
    with pytest.raises(b.C4BesideRefused):
        b.shape_of([0.1], [0, 1], "peak")


def test_summed_map_is_t2_over_2w_on_the_warmup_and_shifts_by_w_over_2_after():
    for t in (2, 53, 256, 1000):
        assert b.summed_map(t) == pytest.approx(t * (t - 1) / (2 * ct.W), abs=1e-6)  # A0's iteration 0 runs at 0
    late = b.summed_map(3000)
    assert ct.summed("F", int(late)) <= ct.summed("A0", 3000) <= ct.summed("F", int(late) + 1)
    assert 3000 - ct.W / 2 - 2 < late < 3000 - ct.W / 2 + 2


def test_summarise_takes_states_1_to_5_on_seven_states():
    per = [{"rho": [0, 1, 2, 3, 4, 5, 9], "rho_no0": [0] * 7, "m": [0, 1, 1, 1, 1, 1, 0],
            "er": [9, 1, 2, 3, 4, 5, 0]}]
    r = b.summarise(per, 7)
    assert r["rho_mid"] == 3 and r["rho_15"] == 3 and r["er_mid"] == 3 and r["m_mid"] == 1
    with pytest.raises(b.C4BesideRefused):
        b.summarise(per, 8)
