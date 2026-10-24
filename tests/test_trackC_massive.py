"""C5 unit 1, M2 (`trackC_writeup/c5_massive.py`, design-C5.md frozen on Blocked 104 (a)): pure parts."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports, torch
from trackA_force.a1a import STEPS
from trackC_writeup import c2_aniso as c
from trackC_writeup import c5_massive as m


def _cloud(seed=0, n=40, d=64, big=(3, 17)):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((n, d))
    H[:, list(big)] += 30.0 * rng.random((n, len(big))) + 20.0     # a common, large pair of coordinates
    return H


def _direct(H, idx):
    X = np.array(H, float)
    X[:, list(idx)] = 0.0
    U = c.unit(X)
    return c.energy(U), c.energy(c.centred(U))


@pytest.mark.parametrize("idx", [[], [3], [3, 17], [5, 9, 40]])
def test_the_gram_correction_is_the_direct_projection(idx):
    H = _cloud()
    got, want = m.proj_energies(H, idx), _direct(H, idx)
    assert np.allclose(got, want, rtol=1e-10, atol=0)


def test_state_quantities_match_proj_energies_and_the_mean():
    H = _cloud(1)
    sets = {"S": [3, 17], "M": [5, 9]}
    q = m.state_quantities(H, sets)
    for k, idx in sets.items():
        assert np.allclose(q["e"][k], m.proj_energies(H, idx), rtol=1e-12)
    U = c.unit(H)
    mu = U.mean(0)
    assert np.isclose(q["m2"], mu @ mu) and np.isclose(q["m2_set"]["S"], mu[[3, 17]] @ mu[[3, 17]])
    # E₁ = ½ + ½‖m‖² + R (math check M4): its first-order term is the anisotropy
    assert q["e"]["none"][0] >= 0.5 + 0.5 * q["m2"] - 1e-12


def test_a_projection_that_empties_a_token_refuses():
    H = np.zeros((5, 8))
    H[:, 2] = 1.0
    H[0, 4] = 1.0
    with pytest.raises(m.M2Refused):
        m.proj_energies(H, [2])


def test_ratios_and_top3_are_s1s_over_median():
    row = np.ones(11)
    row[4], row[7] = 500.0, 120.0
    assert np.allclose(m.ratios(row)[[4, 7]], [500.0, 120.0])
    assert m.top3(row)[:2] == [(4, 500.0), (7, 120.0)]


def _r(steps, d=1024, hits=()):
    out = {s: np.ones((m.N_STATES, d)) for s in steps}
    for s, state, coord, val in hits:
        out[s][state, coord] = val
    for s in steps:                                       # graded below-threshold ratios for M
        out[s][0, 900:910] = np.linspace(10, 19, 10)
    return out


def test_select_takes_the_union_over_steps_and_m_from_the_rest():
    r = _r(STEPS, hits=[(512, 6, 357, 101.0), (143000, 10, 130, 300.0), (143000, 10, 357, 400.0)])
    sel = m.select_sets(r, 0)
    assert sel["refused"] is None and sel["S"] == [130, 357]
    assert sel["R0b_step"] == 512 and sel["above_by_step"]["256"] == []
    assert sel["M"] == [908, 909]                         # the next two by their largest r
    assert len(sel["random"]) == m.N_RAND
    assert all(len(x) == 2 and not set(x) & {130, 357} for x in sel["random"])
    assert m.select_sets(r, 0)["random"] == sel["random"] != m.select_sets(r, 1)["random"]


def test_select_refuses_an_empty_or_oversized_set():
    assert m.select_sets(_r(STEPS), 0)["refused"]
    many = [(143000, 5, i, 200.0) for i in range(m.MAX_S + 1)]
    assert m.select_sets(_r(STEPS, hits=many), 0)["refused"]


def _v(break_at, plateau=56.0):
    return np.array([0.0 if s < break_at else plateau for s in STEPS])


def test_date_compare_and_label():
    a, b = m.date(_v(512)), m.date(_v(1000))
    assert a["step"] == 512 and not a["unreadable"]
    assert m.compare(a, a) == "kept" and m.compare(b, a) == "later" and m.compare(a, b) == "earlier"
    early = _v(512)
    early[STEPS.index(64)] = 40.0
    assert m.date(early)["unreadable"] and m.compare(m.date(early), a) == "unreadable"
    assert m.date(np.zeros(len(STEPS)))["unreadable"]
    assert m.label({"x": "kept", "y": "kept"})["label"] == "kept"
    assert m.label({"x": "later", "y": "later"})["label"] == "moved"
    lab = m.label({"x": "later", "y": "unreadable"})
    assert lab["label"] == "mixed" and lab["flagged"]


def test_specific_needs_both_controls():
    assert m.specific({"a": "kept", "b": "kept"}, 18)
    assert not m.specific({"a": "kept", "b": "later"}, 20)
    assert not m.specific({"a": "kept", "b": "kept"}, 17)


def test_tally_labels():
    kept = {s: {"label": "kept", "specific": False, "flagged": False} for s in range(1, 10)}
    assert m.tally(kept)["label"] == "kept"
    moved = {s: {"label": "moved", "specific": True, "flagged": False} for s in range(1, 10)}
    assert m.tally(moved)["label"] == "the coordinates'"
    moved[1].update(specific=False)
    moved[2].update(specific=False)
    moved[3].update(label="mixed", flagged=True)
    t = m.tally(moved)
    assert t["label"] == "mixed" and t["K_moved_specific"] == 6 and t["flagged"] == [3]


def test_shares_divide_the_falls_only_at_drops():
    e1 = np.array([1.0, 0.9, 0.95, 0.94999])               # a drop at t=0 only (the last is below 1e-3)
    m2 = np.array([0.8, 0.6, 0.7, 0.7])
    m2s = np.array([0.5, 0.35, 0.4, 0.4])
    v, skipped = m.shares(m2, m2s, e1)
    assert np.allclose(v, [0.75]) and skipped == 0
    v, skipped = m.shares(np.array([0.8, 0.8, 0.8, 0.8]), m2s, e1)
    assert v == [] and skipped == 1


def test_the_rule_freezes_before_pass2():
    if m.TAU_P is None:
        with pytest.raises(m.M2Refused):
            m._frozen()
    else:
        assert m.TAU_P > 0 and m.TAU_M > 0
