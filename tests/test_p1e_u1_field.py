"""
`p1e_energy_field/u1_field.py` — U1 on synthetic clouds (`design-1e.md` "U1: the rule", tests):
three planted von Mises–Fisher groups give three wells and their labels; the leave-one-out
density against an explicit loop; the Gaussian draw's covariance; AMI 0 under permutation.
"""
import numpy as np
import pytest

from p1e_energy_field import u1_field as u1

pytestmark = pytest.mark.pure


def _vmf3(n_per=40, d=24, kappa=400.0, seed=0):
    """Three tight groups around orthogonal centres (vMF-like: Gaussian noise, then the sphere)."""
    rng = np.random.default_rng(seed)
    C = np.linalg.qr(rng.normal(size=(d, 3)))[0].T
    X = np.concatenate([c + rng.normal(size=(n_per, d)) / np.sqrt(kappa) for c in C])
    return X / np.linalg.norm(X, axis=1, keepdims=True), np.repeat(np.arange(3), n_per)


@pytest.mark.parametrize("exact", [True, False])
def test_three_planted_groups_give_three_wells(exact):
    X, lab = _vmf3()
    ms = u1.mean_shift(X, 10.0, exact=exact)
    assert u1.well_stats(ms["wells"])["k"] == 3
    assert u1.agreement(ms["wells"], lab) == 1.0
    assert ms["unconverged"] == 0


def test_float32_path_matches_float64_reference():
    X, _ = _vmf3(seed=3)
    a = u1.mean_shift(X, 10.0)["wells"]
    b = u1.mean_shift(X, 10.0, exact=True)["wells"]
    assert u1.agreement(a, b) == 1.0


def test_one_well_at_small_beta():
    X, _ = _vmf3()
    assert u1.well_stats(u1.mean_shift(X, 0.5)["wells"])["k"] == 1


def test_density_is_leave_one_out_and_centred():
    X, _ = _vmf3(n_per=10)
    beta = 3.5
    e = u1.density(X @ X.T, beta)
    lp = np.array([np.log(sum(np.exp(beta * X[i] @ X[j]) for j in range(len(X)) if j != i))
                   for i in range(len(X))])
    np.testing.assert_allclose(e, lp - lp.mean(), atol=1e-10)
    assert abs(e.sum()) < 1e-9


def test_causal_density_is_the_mean_over_earlier_positions():
    rng = np.random.default_rng(1)
    U = rng.normal(size=(12, 8))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    pos = np.array([1, 4, 7, 11])
    lp = np.array([np.log(np.mean([np.exp(2.0 * U[p] @ U[j]) for j in range(p)])) for p in pos])
    np.testing.assert_allclose(u1.causal_density(U, pos, 2.0), lp - lp.mean(), atol=1e-10)


def test_gaussian_draw_keeps_mean_and_covariance_before_the_sphere():
    rng = np.random.default_rng(2)
    U = rng.normal(size=(300, 6)) * [3, 2, 1, 1, .5, .5] + [4, 0, 0, 0, 0, 0]
    mu, C = U.mean(0), np.cov(U.T)
    draws = []
    for _ in range(200):
        G = rng.standard_normal((300, 300))
        draws.append(mu + G @ (U - mu) / np.sqrt(299))
    D = np.concatenate(draws)
    np.testing.assert_allclose(D.mean(0), mu, atol=0.05)
    np.testing.assert_allclose(np.cov(D.T), C, atol=0.15)
    Y = u1.gaussian_draw(U, rng)
    np.testing.assert_allclose(np.linalg.norm(Y, axis=1), 1.0)


def test_ami_and_purity_at_chance_under_permutation():
    from sklearn.metrics import adjusted_mutual_info_score as ami
    rng = np.random.default_rng(4)
    w = rng.integers(0, 3, 600)
    pos = np.arange(1, 601)
    vals = [ami(u1.position_bins(pos), rng.permutation(w)) for _ in range(50)]
    assert abs(np.mean(vals)) < 0.01
    d = u1.describe(np.repeat([0, 1, 2], 200), pos, rng.permutation(np.repeat(list("abc"), 200)),
                    [np.arange(0, 10), np.arange(300, 310)], rng)
    assert d["ami_pos"] > 0.5 and abs(d["ami_cls"]) < 0.02
    assert d["purity"] == 1.0 and d["purity_p"] < 0.01
    assert d["open_share"] == 1.0 and abs(d["open_well_share"] - 1 / 3) < 1e-12


def test_dedup_merges_only_within_tolerance():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([np.cos(1e-4), np.sin(1e-4), 0.0])
    c = np.array([0.0, 1.0, 0.0])
    lab, reps = u1.dedup(np.stack([a, b, c]), 1e-6)
    assert lab.tolist() == [0, 0, 1] and reps.tolist() == [0, 2]


def test_hidden_state_24_refused(tmp_path):
    n = 5
    acts = np.zeros((25, n, 4), dtype=np.float32)
    acts[..., 0] = 1
    np.savez(tmp_path / "activations.npz", activations=acts, norms=np.ones((25, n)))
    (tmp_path / "geometry.json").write_text('{"tokens": ["a", "b", "c", "d", "e"]}')
    ln1 = {"w": np.ones((24, 4)), "b": np.zeros((24, 4)), "eps": 1e-5}
    with pytest.raises(ValueError, match="refused"):
        u1.read_run(tmp_path, ln1, {"t12": np.arange(1, n)}, "k", "0", layers=[24])
