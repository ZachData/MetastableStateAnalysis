"""
`p1e_energy_field/u3_saddles.py` — U3 at β 10 on synthetic clouds (`design-1e.md` "U3 at β 10",
first checks): planted groups give k − 1 deaths with persistence ≥ 0; two mirror-image groups have
their pass on the mirror plane, where the graph and the band both find it; the path is resampled
on the sphere. The band needs torch (deps tier).
"""
import numpy as np
import pytest

from p1e_energy_field import u1_field as u1
from p1e_energy_field import u3_saddles as u3


def _groups(centres, n_per=40, kappa=200.0, seed=0, d=16):
    rng = np.random.default_rng(seed)
    C = u3.unit(np.asarray(centres, dtype=float))
    X = np.concatenate([c + rng.normal(size=(n_per, d)) / np.sqrt(kappa) for c in C])
    return u3.unit(X), np.repeat(np.arange(len(C)), n_per)


def _pair(theta=1.0, d=16, **kw):
    a, b = np.zeros(d), np.zeros(d)
    a[0], a[1], b[0], b[1] = np.cos(theta / 2), np.sin(theta / 2), np.cos(theta / 2), -np.sin(theta / 2)
    X, lab = _groups([a, b], d=d, **kw)
    X[len(X) // 2:] = X[:len(X) // 2] * np.r_[1, -1, np.ones(d - 2)]   # exact mirror in e_1
    return X, lab


@pytest.mark.pure
def test_three_groups_give_two_deaths_with_nonnegative_persistence():
    d = 16
    X, lab = _groups(np.eye(d)[:3], d=d)
    ms = u1.mean_shift(X, u3.BETA, modes=True)
    assert u1.agreement(ms["wells"], lab) == 1.0
    t = u3.merge_tree(X, ms["wells"], u3.heights(ms["modes"], X, u3.BETA), u3.BETA)
    assert len(t["deaths"]) == 2 and all(dd["p"] >= 0 for dd in t["deaths"])
    assert {dd["well"] for dd in t["deaths"]} | {dd["into"] for dd in t["deaths"]} == {0, 1, 2}


@pytest.mark.pure
def test_one_well_has_no_deaths_and_apart_wells_are_bridged():
    X, _ = _groups([np.eye(8)[0]], d=8)
    ms = u1.mean_shift(X, 2.0, modes=True)
    assert u3.merge_tree(X, ms["wells"], u3.heights(ms["modes"], X, 2.0), 2.0)["deaths"] == []
    Y, lab = _groups(np.eye(8)[:2], n_per=30, kappa=5000.0, d=8)       # far apart: 15-NN splits them
    t = u3.merge_tree(Y, lab, np.zeros(2), 10.0, k=2)
    assert t["bridges"] == 1 and len(t["deaths"]) == 1


def _plane_pass(X, m, beta=u3.BETA, iters=5000):
    """The pass of two mirror-image wells by brute force: the highest ``h`` on the mirror plane."""
    y = u3.unit(m[0] + m[1])
    for _ in range(iters):
        S = beta * (X @ y)
        w = np.exp(S - S.max())
        g = (w / w.sum()) @ X
        g[1] = 0.0
        y = u3.unit(y + 0.01 * (g - (g @ y) * y))
    return u3.heights(y[None], X, beta)[0]


@pytest.mark.pure
def test_mirror_pair_graph_pass_is_below_the_true_pass():
    X, _ = _pair()
    ms = u1.mean_shift(X, u3.BETA, modes=True)
    assert u1.well_stats(ms["wells"])["k"] == 2
    t = u3.merge_tree(X, ms["wells"], u3.heights(ms["modes"], X, u3.BETA), u3.BETA)
    (dd,) = t["deaths"]
    true = _plane_pass(X, ms["modes"])
    assert 0 < dd["p"] and dd["h"] <= true      # a chord's midpoint is a feasible path, not the ridge


@pytest.mark.pure
def test_slerp_path_is_on_the_sphere_and_evenly_spaced():
    P = u3._slerp_path(np.eye(3), 9)
    assert np.allclose(np.linalg.norm(P, axis=1), 1)
    ang = np.arccos(np.clip(np.sum(P[:-1] * P[1:], axis=1), -1, 1))
    assert np.allclose(ang, ang[0], atol=1e-9) and np.allclose(P[0], [1, 0, 0]) and np.allclose(P[-1], [0, 0, 1])


@pytest.mark.deps
def test_neb_finds_the_symmetric_pass():
    pytest.importorskip("torch")
    X, _ = _pair()
    ms = u1.mean_shift(X, u3.BETA, modes=True)
    m = ms["modes"]
    t = u3.merge_tree(X, ms["wells"], u3.heights(m, X, u3.BETA), u3.BETA)
    (dd,) = t["deaths"]
    r = u3.neb(m[dd["well"]], m[dd["into"]], [X[dd["i"]], X[dd["j"]]], X, u3.BETA)
    assert r["converged"]
    assert r["h"] == pytest.approx(_plane_pass(X, m), abs=1e-4)   # 12.8872 here, both ways
    assert r["h"] >= dd["h"]


@pytest.mark.pure
def test_skipped_well_ids_are_made_contiguous(monkeypatch):
    X, lab = _groups(np.eye(8)[:2], d=8)
    real = u1.mean_shift
    def gappy(*a, **k):
        r = real(*a, **k)
        return {**r, "wells": np.where(r["wells"] == 1, 2, r["wells"]),
                "modes": np.concatenate([r["modes"][:1], r["modes"][:1] * 0, r["modes"][1:]])}
    monkeypatch.setattr(u3, "mean_shift", gappy)
    ms = u3.wells_and_modes(X, u3.BETA, "cpu")
    assert set(ms["wells"].tolist()) == {0, 1} and len(ms["modes"]) == 2
    assert np.allclose(np.linalg.norm(ms["modes"], axis=1), 1)
    t = u3.merge_tree(X, ms["wells"], u3.heights(ms["modes"], X, u3.BETA), u3.BETA)
    assert len(t["deaths"]) == 1
