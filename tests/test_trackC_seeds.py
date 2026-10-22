"""C2 unit 3s, the centred count on the seeds (`trackC_writeup/c2_seeds.py`, design-C2.md Blocked 96 (b), 97 (a)): pure parts."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackA_force.a1a import STEPS
from trackC_writeup import c2_seeds as s

SEED0_V = np.array([0, 0, 0, 0, 0, 0, 0, 0, 19, 56, 56, 61, 57, 59, 64, 76, 72, 70], float)
SEED0_VC_1E4 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 11, 55, 56, 55, 57, 56, 51, 58, 75], float)


def test_the_arms_are_frozen_and_ordered():
    assert s.TAU_ARMS is not None and list(s.TAU_ARMS) == sorted(s.TAU_ARMS)
    assert s.TAU_ARMS[1:3] == s.FIXED_ARMS


def test_median_matched_tol_is_the_same_multiple_of_each_scale():
    r_u = np.array([2e-2, -3e-2, 1e-2, 4e-2])
    assert np.isclose(s.median_matched_tol(r_u, r_u / 30), 1e-3 / 30)
    with pytest.raises(s.SeedsRefused):
        s.median_matched_tol([0.0, 0.0, 0.0], [1e-4, 2e-4, 3e-4])


def test_share_is_the_turned_part_of_the_unviolated():
    assert s.share(0.36, 0.36) == 0.0 and np.isclose(s.share(1.0, 0.36), 1.0)
    assert np.isclose(s.share(0.68, 0.36), 0.5)


def test_null_matched_tol_equalises_the_null_rates():
    rng = np.random.default_rng(0)
    nu, nc = rng.normal(0, 1e-2, 1000), rng.normal(0, 1e-5, 1000)
    t = s.null_matched_tol(nu, nc)
    assert abs(s.rate(nc, t) - s.rate(nu, 1e-3)) <= 2e-3


def test_b_matched_tol_hits_its_target_share():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1e-4, 2000)
    planted = base + rng.exponential(3e-4, 2000)
    t = s.b_matched_tol(base, planted, 0.5)
    assert abs(s.share(s.rate(planted, t), s.rate(base, t)) - 0.5) < 0.01


def test_b_matched_tol_refuses_a_target_it_cannot_reach():
    with pytest.raises(s.SeedsRefused):
        s.b_matched_tol(np.zeros(10), np.zeros(10), 0.5)


def _units(b_c, a_c, null_c):
    return {"trained": {"base": {"u": [0.0] * 10, "c": [0.0] * 10},
                        "plants": {"B_c": {"0.25": b_c, "0.1": b_c}, "A_c": {"0.5": a_c, "1": a_c}}},
            "reinit": {"base": {"c": null_c}}}


def test_gate_arms_pass_and_fail():
    ok = s.gate_arms(_units([1.0] * 10, [0.0] * 10, [0.0] * 100), (1e-4,))
    assert ok["pass"]
    assert not s.gate_arms(_units([0.0] * 10, [0.0] * 10, [0.0] * 100), (1e-4,))["pass"]   # B unseen
    assert not s.gate_arms(_units([1.0] * 10, [1.0] * 10, [0.0] * 100), (1e-4,))["pass"]   # A leaks
    assert not s.gate_arms(_units([1.0] * 10, [0.0] * 10, [1.0] * 100), (1e-4,))["pass"]   # null noisy


def test_seed0_at_1e4_reads_both_events():
    r = s.seed_reading(SEED0_V, {"1e-4": SEED0_VC_1E4})
    assert r["break_step"] == 512 and r["event1"] and r["event2"] and not r["unreadable"]


def test_an_event_must_hold_at_every_arm():
    r = s.seed_reading(SEED0_V, {"a": SEED0_VC_1E4, "b": SEED0_V})
    assert r["arms"]["a"]["event2"] and not r["arms"]["b"]["event2"] and not r["event1"] and not r["event2"]


def test_a_centred_date_at_2000_is_event1_not_event2():
    vc = SEED0_VC_1E4.copy()
    vc[STEPS.index(1000)] = 10
    r = s.seed_reading(SEED0_V, {"a": vc})
    assert r["arms"]["a"]["centred_step"] == 2000 and r["event1"] and not r["event2"]


def test_early_counts_make_a_seed_unreadable_and_count_against():
    vc = SEED0_VC_1E4.copy()
    vc[STEPS.index(8)] = 40
    r = s.seed_reading(SEED0_V, {"a": SEED0_VC_1E4, "b": vc})
    assert r["unreadable"] and not r["event1"] and not r["event2"]


def test_a_zero_centred_plateau_is_unreadable():
    assert s.seed_reading(SEED0_V, {"a": np.zeros(18)})["unreadable"]


def test_tally_labels():
    assert [s.tally_label(k) for k in (9, 7, 6, 3, 2, 0)] == [
        "reproduced", "reproduced", "mixed", "mixed", "seed 0's only", "seed 0's only"]
