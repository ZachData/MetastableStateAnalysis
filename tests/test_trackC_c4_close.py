"""C4's close-out readings (`trackC_writeup/c4_close.py`), on planted series. No torch, no data."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports

from trackC_writeup import c4_close as k
from trackC_writeup import c4_read as cr
from trackC_writeup import c4_train as ct


def _recs(profiles):
    """{step: {prompt: profile}} → records keyed as `c4_read.load_arm`'s."""
    return {(s, p): {"e1": e, "e1c": e} for s, per in profiles.items() for p, e in per.items()}


def test_per_transition_places_each_drop_and_sums_to_the_count():
    up, dip1, dip13 = [1.0, 1.1, 1.2, 1.3], [1.0, 0.9, 1.0, 1.1], [1.0, 0.9, 1.0, 0.8]
    recs = _recs({0: {"a": up, "b": up}, 8: {"a": dip1, "b": dip13}})
    mat = k.per_transition(recs, "u", [0, 8], prompts=("a", "b"))
    assert mat.tolist() == [[0, 0, 0], [2, 0, 1]]
    assert mat.sum(1).tolist() == [sum(cr.c.n_viol(recs[(s, p)]["e1"], k.REL_TOL) for p in ("a", "b")) for s in (0, 8)]


def test_per_transition_uses_the_arm_tolerance_on_the_centred_field():
    small = [1.0, 1.0 - 5e-4, 1.0]                       # a relative drop of 5e-4
    recs = _recs({0: {"a": small}})
    assert k.per_transition(recs, "0.0001", [0], prompts=("a",)).tolist() == [[1, 0]]
    assert k.per_transition(recs, "0.00478", [0], prompts=("a",)).tolist() == [[0, 0]]


def test_settled_is_the_first_step_the_majority_set_stays_the_last():
    grid = [0, 1, 2, 4, 8]
    mat = np.array([[0, 0, 0], [7, 7, 7], [0, 7, 7], [0, 7, 3], [0, 7, 0]])
    assert k.settled(mat, grid) == {"step": 4, "set": [1]}          # 3 of 7 is not a majority
    back = np.array([[0, 7, 0], [7, 7, 0], [0, 7, 0]])
    assert k.settled(back, [0, 1, 2])["step"] == 2                  # an early match does not count
    assert k.settled(np.array([[0, 7], [0, 7]]), [0, 1])["step"] == 0


def test_rho_star_puts_the_date_on_the_registered_band_edge():
    for t_a0, t_f in ((640, 48), (384, 24), (640, 400)):
        r = k.rho_star(t_a0, t_f, "cumulative")
        lo, hi = cr.m("cumulative", t_a0 / r), cr.m("cumulative", t_a0 * r)
        assert min(abs(lo - t_f), abs(hi - t_f)) < 1e-9 and r >= 1
        s = k.rho_star(t_a0, t_f, "step")
        assert min(abs(t_a0 / s - t_f), abs(t_a0 * s - t_f)) < 1e-9
    assert not cr.consistent(48, 640, 1.72)["cumulative"] and cr.consistent(48, 640, 1.73)["cumulative"]
    assert k.rho_star(640, cr.m("cumulative", 640), "cumulative") == pytest.approx(1.0)
    with pytest.raises(k.C4CloseRefused):
        k.rho_star(0, 48, "step")


def test_window_loss_and_the_maps():
    losses = [float(i) for i in range(1, 21)]                       # iteration t's loss is t
    assert k.window_loss(losses, 10, half=2) == pytest.approx(10.0)
    assert k.window_loss(losses, 1, half=4) == pytest.approx(3.0)   # iterations 1..5
    assert k.window_loss(losses, 40, half=4) is None
    assert k.mapped(640) == {"step": 640.0, "cumulative": 640 * 640 / (2.0 * ct.W)}
    assert k.mapped(2000)["cumulative"] is None and k.mapped(None)["step"] is None


def test_induction_of_dates_the_top_head_and_lists_heads_at_the_bar():
    grid = list(ct.GRID)
    per = {}
    for s in grid:
        m = np.full((6, 8), 0.01)
        if s >= 320:
            m[3, 6] = 0.9
        if s >= 1000:
            m[2, 1] = 0.6
        per[str(s)] = m.tolist()
    d = k.induction_of(per)
    assert d["half_rise"] == 320 and d["plateau"] == pytest.approx(0.9)
    assert [(h["layer"], h["head"]) for h in d["heads_at_bar_4000"]] == [(3, 6), (2, 1)]
    assert d["top_head"][grid.index(4000)] == [3, 6]
