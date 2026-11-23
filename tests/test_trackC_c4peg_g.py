"""C4′ (g) (`trackC_writeup/c4peg_g.py`, design-C4peg.md "(g) as U2 runs it"): the pure parts on
planted inputs."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackC_writeup import c4peg_g as g


def test_barrier_of_a_planted_bump():
    a = g.ALPHAS
    L = [1.0 + 0.5 * x + 0.3 * np.sin(np.pi * x) for x in a]
    b, am = g.barrier(a, L)
    assert np.isclose(b, 0.3) and am == 0.5
    assert g.barrier(a, [2.0 - x for x in a])[0] == pytest.approx(0.0, abs=1e-12)   # linear: no barrier
    assert g.refine_alphas(0.5) == [0.45, 0.55] and g.refine_alphas(0.0) == [0.05]


def test_gate_scale_and_labels():
    assert g.gate_g1(0.5, 0.1) and not g.gate_g1(0.5, 0.2) and not g.gate_g1(0.09, 0.0)
    assert g.scale_g1(0.0, 0.0, 0.0) == g.S_FLOOR
    S = g.scale_g1(0.02, 0.05, 0.01)
    assert S == 0.05
    assert g.label_g1(0.05, S, True) == "within the order-only spread"
    assert g.label_g1(0.08, S, True) == "between"
    assert g.label_g1(0.11, S, True) == "beyond it"
    assert g.label_g1(9.0, S, False) == "unreadable"
    assert g.lean_g1(0.05, 0.02, 0.0) == "held" and g.lean_g1(0.02, 0.05, 0.0) == "broken"
    assert g.lean_g1(0.025, 0.02, 0.0) == "not told apart"
    assert g.lean_g2("between") is True and g.lean_g2("within the order-only spread") is False
    assert g.lean_g2("unreadable") is None


def test_random_sets_avoid_the_union_and_strong_heads():
    rng = np.random.default_rng(0)
    sc = rng.uniform(0, 0.15, size=(6, 8))
    sc[3, :4] = 0.9
    sc[2, 6] = 0.3                                   # neither own nor eligible
    union = [f"L3H{h}" for h in range(4)] + ["L2H6", "L1H1"]
    sets = g.random_sets(sc, union, 4, 20, 7)
    assert len(sets) == 20 and all(len(s) == 4 and len(set(s)) == 4 for s in sets)
    assert not any(h in union or h == "L2H6" for s in sets for h in s)
    assert sets == g.random_sets(sc, union, 4, 20, 7)
    assert g.own_set(sc) == ("L3H0", "L3H1", "L3H2", "L3H3")


def test_g2_labels_and_induction_batch():
    lb = g.label_g2(1.0, 0.1, [0.2, 0.3])
    assert lb["own_carry"] and not lb["index_carries"] and lb["random_max"] == 0.3
    s = g.induction_batch(4, 8, 1)
    assert s.shape == (4, 16) and (s[:, :8] == s[:, 8:]).all()


def test_read_g_on_planted_records():
    def pair(B):
        return {"B": B}
    g1r = {"pairs": {"question@4000": pair(0.2), "scale_A0@4000": pair(0.01), "scale_F@4000": pair(0.05),
                     "null@4000": pair(0.001), "effect@4000": pair(2.0), "question_pair2@4000": pair(0.3)}}
    arms = {}
    for arm in g.ARMS:
        sets = {"own": {"d_ind": 2.0, "d_held": 0.01, "heads": ["L3H0"]},
                "other": {"d_ind": 0.01, "d_held": 0.0, "heads": ["L2H6"]}}
        sets.update({f"random{i}": {"d_ind": 0.05, "d_held": 0.0, "heads": ["L0H1"]} for i in range(g.N_RANDOM)})
        arms[arm] = {"sets": sets}
    r = g.read_g(g1r, {"arms": arms})
    assert r["g1"]["readable"] and r["g1"]["S"] == 0.05 and r["g1"]["label"] == "beyond it"
    assert r["leans"]["g1"] == "held" and r["leans"]["g2"]["held"] is True and r["leans"]["g3"]["held"]
