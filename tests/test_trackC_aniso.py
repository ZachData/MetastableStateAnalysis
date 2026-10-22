"""C2 unit 3's anisotropy control (`trackC_writeup/c2_aniso.py`, design-C2.md 92 (i), 95 (a)): pure parts."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackA_force.a1a import STEPS
from trackC_writeup import c2_aniso as c

SEED0_V = np.array([0, 0, 0, 0, 0, 0, 0, 0, 19, 56, 56, 61, 57, 59, 64, 76, 72, 70], float)


def _cloud(n=64, d=32, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:, 0] += shift
    return X


def test_centring_removes_an_added_vector_on_the_sphere_exactly():
    U = c.unit(_cloud())
    v = np.random.default_rng(1).standard_normal(U.shape[1])
    assert np.allclose(c.centred(U + v), c.centred(U))


def test_centred_cloud_has_unit_rows():
    assert np.allclose(np.linalg.norm(c.centred(c.unit(_cloud(shift=3.0))), axis=1), 1.0)


def test_mean_cos_matches_the_gram():
    U = c.unit(_cloud(shift=2.0))
    G = U @ U.T
    n = len(U)
    assert np.isclose(c.mean_cos(U), (G.sum() - np.trace(G)) / (n * (n - 1)))


def test_a_common_direction_raises_mean_cos_and_energy():
    lo, hi = c.unit(_cloud()), c.unit(_cloud(shift=3.0))
    assert c.mean_cos(hi) > c.mean_cos(lo) + 0.15 and c.energy(hi) > c.energy(lo)


def test_degenerate_row_refuses():
    X = _cloud()
    X[3] = 0
    with pytest.raises(c.AnisoRefused):
        c.unit(X)


def test_rel_drops_sign():
    assert c.rel_drops([1.0, 2.0, 1.5]).tolist() == [-1.0, 0.25]


def _v(**at):
    v = np.zeros(len(STEPS))
    for s, x in at.items():
        v[STEPS.index(int(s[1:]))] = x
    return v


def test_seed0_shape_survives():
    r = c.verdict(SEED0_V, SEED0_V)
    assert r["label"] == "survives" and r["step"] == 512 and r["ratio"] == 1.0


def test_small_plateau_is_mostly_removed_not_survives():
    """95 (a): the reason it was added; 92 (i) alone reads this 'survives'."""
    vc = np.minimum(SEED0_V, 3.0)
    r = c.verdict(vc, SEED0_V)
    assert r["label"] == "mostly removed" and r["step"] == 256


def test_zero_plateau_is_removed():
    assert c.verdict(np.zeros(len(STEPS)), SEED0_V)["label"] == "removed"


def test_early_counts_unreadable():
    vc = SEED0_V.copy()
    vc[STEPS.index(4)] = 30
    assert c.verdict(vc, SEED0_V)["label"] == "unreadable"


def test_early_counts_below_half_are_readable():
    vc = SEED0_V.copy()
    vc[STEPS.index(4)] = 28        # < 57 / 2
    assert c.verdict(vc, SEED0_V)["label"] == "survives"


def test_late_rise_is_moved():
    vc = _v(s1000=40, s2000=40, s4000=40, s8000=40)
    r = c.verdict(vc, SEED0_V)
    assert r["label"] == "moved" and r["step"] == 1000


def test_size_at_exactly_half_is_kept():
    vc = SEED0_V * 0.5             # plateau 28.5, ratio 0.5: not below SIZE
    assert c.verdict(vc, SEED0_V)["label"] == "survives"


def test_zero_uncentred_plateau_refuses():
    with pytest.raises(c.AnisoRefused):
        c.verdict(SEED0_V, np.zeros(len(STEPS)))


def test_plant_common_moves_every_token_by_one_vector():
    X = _cloud()
    u = np.eye(X.shape[1])[1]
    d = c.plant_common(X, 0.5, u) - X
    assert np.allclose(d, d[0]) and np.isclose(np.linalg.norm(d[0]), 0.5 * np.median(np.linalg.norm(X, axis=1)))


def test_plant_spread_lowers_centred_energy():
    X = _cloud(shift=0.0)
    X[:32, 1] += 4.0               # two clusters
    rng = np.random.default_rng(2)
    e0 = c.energy(c.centred(c.unit(X)))
    e1 = c.energy(c.centred(c.unit(c.plant_spread(X, 1.0, rng))))
    assert e1 < e0
