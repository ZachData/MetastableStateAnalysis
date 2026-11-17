"""C4′ (b): is F A0 on another clock? (`trackC_writeup/c4p_path.py`, design-C4p.md "(b) as U2
runs it"): the reading's pure parts on planted K matrices."""
import math

import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackC_writeup import c4_train as ct
from trackC_writeup import c4p_path as p


def test_sets_match_the_design():
    assert len(p.A0_STEPS) == 29 and 0 in p.A0_STEPS and 3500 in p.A0_STEPS
    assert len(p.F_STEPS) == 28 and len(p.REL_STEPS) == 13 and len(p.SEED_STEPS) == 12
    w = p.windows(p.REL_STEPS)
    assert (len(w["early"]), len(w["late"])) == (9, 4)
    w = p.windows(p.SEED_STEPS)
    assert (len(w["early"]), len(w["late"])) == (8, 4)
    w = p.windows(p.F_STEPS)
    assert (len(w["early"]), len(w["late"])) == (16, 12)
    assert p.N_POS == 131072
    assert set(p.REL_STEPS) <= set(p.A0_STEPS)


def test_merge_states():
    assert p.merge_states([0, 1, 2, 4], ["a", "a", "b", "c"]) == [(0, 1), (2,), (4,)]
    assert p.merge_states([0, 1, 2], ["a", "b", "a"]) == [(0,), (1,), (2,)]


def test_deltas_take_the_nearer_neighbour():
    K = np.array([[0, 1, 9], [2, 0, 3], [9, 5, 0]], float)
    assert p.deltas(K).tolist() == [1, 2, 5]


def test_monotone_allows_one_step_back_from_the_running_max():
    assert p.monotone([0, 1, 2, 1, 2, 3])
    assert not p.monotone([0, 3, 1])
    assert not p.monotone([3, 2, 1])          # a slow descent fails


@pytest.mark.parametrize("n_on,want", [(4, "one path"), (3, "one path"), (2, "mixed"), (1, "another path"), (0, "another path")])
def test_thresholds_on_four_steps(n_on, want):
    assert p.window_reading([0, 1, 2, 3], [True] * n_on + [False] * (4 - n_on)) == want


def test_not_monotone_is_not_one_path():
    assert p.window_reading([3, 0, 1, 2], [True] * 4) == "mixed"


def _planted(pseudo_x, ref_x):
    """K between points on a line, (x_a − x_b)², with the reference first."""
    x = np.concatenate([ref_x, pseudo_x])
    return (x[:, None] - x[None, :]) ** 2


def _keys(pkind, psteps, rkind):
    return [p.key(rkind, t) for t in p.A0_STEPS] + [p.key(pkind, s) for s in psteps]


def _x(steps):
    return np.log1p(np.asarray(steps, float))


def test_identity_reads_one_path_with_identity():
    K = _planted(_x(p.REL_STEPS), _x(p.A0_STEPS))
    hs = {p.key("A0h", t): str(t) for t in p.A0_STEPS}
    rd = p.read_pseudo(K, _keys("rel", p.REL_STEPS, "A0h"), "rel", p.REL_STEPS, "A0h", hs)
    for w in ("early", "late"):
        assert rd["windows"][w]["reading"] == "one path"
        assert rd["windows"][w]["n_identity"] == rd["windows"][w]["n"]


def test_far_path_reads_another_path():
    K = _planted(_x(p.SEED_STEPS) + 100.0, _x(p.A0_STEPS))
    hs = {p.key("A0", t): str(t) for t in p.A0_STEPS}
    rd = p.read_pseudo(K, _keys("seed1", p.SEED_STEPS, "A0"), "seed1", p.SEED_STEPS, "A0", hs)
    assert all(rd["windows"][w]["reading"] == "another path" for w in ("early", "late"))


def test_bit_equal_steps_merge_and_identity_holds_on_either():
    hs = {p.key("A0h", t): ("z" if t in (0, 1) else str(t)) for t in p.A0_STEPS}
    x = _x(p.A0_STEPS).copy()
    x[1] = x[0]                                            # A0₁ = A0₀
    K = _planted(np.array([x[0]] + list(_x(p.REL_STEPS[1:]))), x)
    rd = p.read_pseudo(K, _keys("rel", p.REL_STEPS, "A0h"), "rel", p.REL_STEPS, "A0h", hs)
    assert rd["states"][0] == [0, 1] and rd["match"][0] == [0, 1] and rd["identity"][0]
    assert rd["delta"][0] > 0


def test_gate_needs_both_null_and_seeds():
    one = {"windows": {w: {"reading": "one path", "n": 4, "n_identity": 3} for w in ("early", "late")}}
    another = {"windows": {w: {"reading": "another path"} for w in ("early", "late")}}
    mixed = {"windows": {w: {"reading": "mixed"} for w in ("early", "late")}}
    seeds = {f"seed{k}": another for k in range(1, 9)} | {"seed9": mixed}
    assert all(v["readable"] for v in p.gate_of(one, seeds).values())
    seeds["seed8"] = mixed
    assert not any(v["readable"] for v in p.gate_of(one, seeds).values())
    low = {"windows": {w: {"reading": "one path", "n": 4, "n_identity": 2} for w in ("early", "late")}}
    assert not any(v["readable"] for v in p.gate_of(low, {f"seed{k}": another for k in range(1, 10)}).values())


def test_summed_match_is_sqrt_2ws_in_the_warmup():
    for s in (8, 64, 256, 512):
        assert abs(p.summed_match(s) - math.sqrt(2 * ct.W * s)) < 1.5
    assert p.summed_match(4000) is None
