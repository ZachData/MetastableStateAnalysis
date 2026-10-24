"""C5 unit 2, M2's other half (`trackC_writeup/c5_mean.py`, design-C5.md "Unit 2"): pure parts."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports, torch
from trackA_force.a5_seeds import n_viol
from trackC_writeup import c5_mean as m


def test_k_half_by_hand():
    A = np.array([1.0, 1.0, 1.0, 0.0])
    B = np.array([0.0, 0.9, 1.0, 0.0])          # δ = (−1, −0.19, 0, 0), Δ = −1.19
    assert m.k_half(A, B).tolist() == [1]
    assert m.k_half_set(A, B) == [0]
    B2 = np.array([0.6, 0.6, 0.6, 0.0])         # three equal falls of 0.64: half needs two
    assert m.k_half(A, B2).tolist() == [2]


def test_k_half_refuses_a_rise():
    with pytest.raises(m.MeanRefused):
        m.k_half(np.zeros(4), np.ones(4))


def test_k_half_is_blind_to_the_order_of_coordinates():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 64))
    B = 0.9 * A + 0.01 * rng.standard_normal((5, 64))
    perm = rng.permutation(64)
    assert np.array_equal(m.k_half(A, B), m.k_half(A[:, perm], B[:, perm]))


def _spread_fall(T=30, d=256, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((T, d)) / np.sqrt(d)
    return A, 0.95 * A                          # every coordinate falls alike: maximally spread


def test_a_planted_fall_of_two_thirds_in_three_coordinates_reads_few():
    A, B = _spread_fall()
    D = (B * B - A * A).sum(1)
    coords = [[3, 70, 200]] * len(A)
    Ap, Bp = m.plant(A, B, coords, 2.0)
    assert np.allclose((Bp * Bp - Ap * Ap).sum(1), 3.0 * D)     # the total is (1 + f)Δ
    assert np.all(m.k_half(Ap, Bp) <= 3)
    assert np.all(m.k_half(*m.plant(A, B, coords, 0.0)) == m.k_half(A, B))
    assert np.all(m.k_half(A, B) > m.FEW)


def test_rotation_keeps_the_fall_and_spreads_a_one_coordinate_fall():
    d = 512
    A = np.zeros((1, d))
    A[0, 5] = 1.0
    B = A * 0.5
    Qs = m.rotations(20, d, 7)
    for Q in Qs[:3]:
        assert np.isclose(((B @ Q.T) ** 2 - (A @ Q.T) ** 2).sum(), (B ** 2 - A ** 2).sum())
    fr, kr = m.f_rot(A, B, Qs)
    assert m.k_half(A, B)[0] == 1 and kr.min() > m.FEW and fr[0] == 0.0
    # and a spread fall reads at chance against its own rotations
    A2, B2 = _spread_fall(T=20, d=d, seed=3)
    fr2, _ = m.f_rot(A2, B2, Qs)
    assert np.median(fr2) > 0.2                 # a Gaussian-shaped fall reads at chance, not concentrated


def test_level_statistics():
    one = np.zeros(10)
    one[2] = 0.3
    flat = np.full(10, 0.5)
    assert m.k_half_level(np.array([one, flat])).tolist() == [1, 5]
    assert np.allclose(m.participation(np.array([one, flat])), [1.0, 10.0])


def test_drops_counts_as_a5s_violations():
    rng = np.random.default_rng(4)
    for _ in range(20):
        e = 1.0 + np.cumsum(rng.normal(0.002, 0.004, 25))
        assert len(m.drops(e)) == n_viol(e)


def test_family_skips_rises_and_tiny_changes():
    d = 8
    base = np.zeros((25, d))
    base[:, 0] = 0.5
    base[4, 0], base[5, 0] = 0.6, 0.4           # transition 4 → 5: a fall of ‖m‖²
    base[10, 0], base[11, 0] = 0.4, 0.6         # 10 → 11: a rise
    e1 = np.linspace(1.0, 2.0, 25)
    e1[5] = e1[4] * 0.9                          # E₁ drops at 4 → 5
    e1[11] = e1[10] * 0.9                        # and at 10 → 11
    e1[16] = e1[15] * 0.9                        # and at 15 → 16, where m does not move
    ms = {(128, p): base for p in m.V1}
    e1s = {(128, p): e1 for p in m.V1}
    fam = m.family(ms, e1s, steps=(128,))
    assert [k[2] for k in fam["keys"]] == [4] * 7
    assert fam["skipped_rise"] == 7 and fam["skipped_small"] == 7


def test_recurring_and_tally():
    assert m.recurring([[1, 2], [1], [3]], 0.5) == [(1, 2 / 3)]
    assert m.tally({s: "not aligned" for s in range(1, 10)})["label"] == "not basis-aligned"
    labs = {s: ("aligned" if s <= 7 else "not aligned") for s in range(1, 10)}
    assert m.tally(labs)["label"] == "basis-aligned"
    labs = {s: ("aligned" if s <= 4 else "not aligned") for s in range(1, 10)}
    assert m.tally(labs)["label"] == "mixed"
    assert m.label_q2(0.5) == "aligned" and m.label_q2(0.49) == "not aligned"


def test_a_mixed_sign_fall_reads_at_chance_and_a_plant_reads_aligned():
    """The gate's lesson (`LESSONS.md` 6, 2026-10-24): coordinates moving both ways, as the data's do."""
    rng = np.random.default_rng(5)
    d, T = 512, 25
    A = rng.standard_normal((T, d)) / np.sqrt(d)
    B = 0.97 * A + 0.05 * rng.standard_normal((T, d)) / np.sqrt(d)
    assert np.all((B * B - A * A).sum(1) < 0)
    assert np.mean((B * B - A * A) > 0) > 0.2                   # mixed signs
    Qs = m.rotations(40, d, 11)
    fr, _ = m.f_rot(A, B, Qs)
    assert np.mean(fr <= m.F_ROT) <= 0.2
    coords = [rng.choice(d, size=3, replace=False).tolist() for _ in range(T)]
    frp, _ = m.f_rot(*m.plant(A, B, coords, 2.0), Qs)
    assert np.mean(frp <= m.F_ROT) >= 0.9
