"""S1's instrument (`trackS_subspace/design-S1.md`): the low-rank singular and invariant bases
against dense ones, the mean projector's spectrum by Gram, the plant and its floor (math check
fact 3), the count, and G0's subspace distance."""

import numpy as np
import pytest

from trackS_subspace import s1

pytestmark = pytest.mark.pure  # numpy only

D, K, R = 96, 20, 6


def _AB(seed):
    g = np.random.default_rng(seed)
    return g.standard_normal((D, K)), g.standard_normal((D, K))


def test_singular_bases_span_the_dense_top_r():
    A, B = _AB(0)
    U, _, Vt = np.linalg.svd(A @ B.T)
    left, right = s1.singular_bases(A, B, R)
    assert s1.sin_max(left, U[:, :R]) < 1e-10 and s1.sin_max(right, Vt[:R].T) < 1e-10


def test_invariant_bases_are_invariant_and_typed():
    A, B = _AB(1)
    V = A @ B.T
    lam = np.linalg.eigvals(V)
    top = lam[np.argsort(-np.abs(lam))]
    out = s1.invariant_bases(A, B, R)
    for U in out.values():
        if U.shape[1]:
            assert np.linalg.norm(V @ U - U @ (U.T @ V @ U)) < 1e-9 * np.linalg.norm(V)
    n_real_neg = sum(1 for x in top[:R] if abs(x.imag) < 1e-12 and x.real < 0)
    n_dims = sum(U.shape[1] for U in out.values())
    assert out["ov_neg"].shape[1] <= n_real_neg + 1 and R <= n_dims <= R + 1   # a pair may straddle r


def test_mean_projector_eigs_by_gram_equal_dense_both_shapes():
    g = np.random.default_rng(2)
    for n in (3, 30):                      # Σ k below and above d
        bases = s1.haar_bases([5] * n, g, D)
        B = np.concatenate(bases, 1)
        dense = np.sort(np.linalg.eigvalsh(B @ B.T / n))[::-1]
        lo = s1.mean_projector_eigs(bases)
        assert np.allclose(lo, dense[:lo.size], atol=1e-12)
        assert np.isclose(lo.sum(), 5.0)                                    # fact 1: tr P̄ = mean k
        assert lo.max() <= 1 + 1e-12 and lo.min() >= -1e-12


def test_empty_pool_reads_zero():
    assert s1.mean_projector_eigs([np.zeros((D, 0))] * 3).size == 0


def test_plant_puts_u_inside_and_keeps_dimension():
    g = np.random.default_rng(3)
    bases = s1.haar_bases([4] * 8, g, D)
    u = g.standard_normal(D)
    u /= np.linalg.norm(u)
    out = s1.plant(bases, u, [1, 5])
    for h in (1, 5):
        assert out[h].shape == bases[h].shape
        assert np.linalg.norm(u - out[h] @ (out[h].T @ u)) < 1e-12
        assert np.allclose(out[h].T @ out[h], np.eye(4), atol=1e-12)
    assert out[0] is bases[0]


def test_planted_floor_k_over_n():
    g = np.random.default_rng(4)
    bases = s1.haar_bases([4] * 16, g, D)
    u = g.standard_normal(D)
    planted = s1.plant(bases, u / np.linalg.norm(u), range(6))
    assert s1.mean_projector_eigs(planted)[0] >= 6 / 16 - 1e-12             # fact 3


def test_two_plants_in_one_head_both_survive():
    g = np.random.default_rng(5)
    bases = s1.haar_bases([4], g, D)
    u, w = (x / np.linalg.norm(x) for x in g.standard_normal((2, D)))
    out = s1.plant(s1.plant(bases, u, [0]), w, [0])
    for x in (u, w):
        assert np.linalg.norm(x - out[0] @ (out[0].T @ x)) < 1e-12


def test_plant_many_draws_only_eligible_heads():
    g = np.random.default_rng(6)
    bases = s1.haar_bases([1, 4, 4, 2, 4], g, D)
    out = s1.plant_many(bases, 1, 3, g, min_dim=4)
    assert out[0] is bases[0] and out[3] is bases[3]


def test_count_and_reading():
    ev = np.array([0.5, 0.2, 0.1, 0.01])
    assert s1.count_above(ev, 0.1) == 2
    g = np.random.default_rng(7)
    rec = s1.reading(s1.haar_bases([4] * 4, g, D), threshold=2.0)
    assert rec["N"] == 0 and rec["m"] == 0.0 and np.isclose(sum(rec["eig"]), 4.0)


def test_sin_max_resolves_below_sqrt_eps():
    g = np.random.default_rng(8)
    U = np.linalg.qr(g.standard_normal((D, 4)))[0]
    W = np.linalg.qr(U + 1e-12 * g.standard_normal((D, 4)))[0]
    assert 0 < s1.sin_max(U, W) < 1e-9                        # √(1 − σ²) floors near 1e-8
    assert s1.sin_max(U, U[:, :3]) == float("inf")


# ---------------------------------------------------------------- unit 2: R1′ (Blocked 82 (b))

def test_pr0_closed_form_and_pr_of_against_eigenvalues():
    assert abs(s1.pr0([16] * 384) - 879.67) < 0.01
    rng = np.random.default_rng(5)
    bs = s1.haar_bases([4, 0, 6, 3, 5], rng, d=D)
    ev = s1.mean_projector_eigs(bs)
    assert abs(s1.pr_of(bs)["PR"] - ev.sum() ** 2 / np.sum(ev ** 2)) < 1e-10


def test_pr0_is_the_haar_mean_kernel():
    rng = np.random.default_rng(6)
    dims = [3, 5, 0, 4]
    k = np.array(dims, float)
    e2 = (k.sum() + (k.sum() ** 2 - (k ** 2).sum()) / 24) / len(dims) ** 2
    tr2 = [(k.sum() / len(dims)) ** 2 / s1.pr_of(s1.haar_bases(dims, rng, d=24))["PR"] for _ in range(4000)]
    assert abs(np.mean(tr2) - e2) < 4 * np.std(tr2) / np.sqrt(len(tr2))


def test_plant_add_keeps_every_column_and_holds_u():
    rng = np.random.default_rng(7)
    bs = s1.haar_bases([5] * 12, rng, d=D)
    u = rng.standard_normal(D)
    u /= np.linalg.norm(u)
    out = s1.plant_add(bs, u, [0, 3, 4])
    for h in (0, 3, 4):
        assert out[h].shape[1] == 6 and np.allclose(out[h][:, :5], bs[h])
        assert np.allclose(out[h].T @ out[h], np.eye(6), atol=1e-12)
        assert abs(np.linalg.norm(out[h].T @ u) - 1) < 1e-12
    tr2 = lambda b: (sum(x.shape[1] for x in b) / len(b)) ** 2 / s1.pr_of(b)["PR"]   # noqa: E731
    assert tr2(out) > tr2(bs)                                   # fact 5: tr P̄² cannot fall
    assert s1.pr_of(out)["chi"] > s1.pr_of(bs)["chi"]


def test_fold_ln1_scales_the_read_side_only():
    rng = np.random.default_rng(8)
    W = {k: rng.standard_normal(s) for k, s in (("Wq", (4, D)), ("Wk", (4, D)), ("Wv", (4, D)), ("Wo", (D, 4)))}
    W["g"] = rng.standard_normal(D)
    F = s1.fold_ln1(W)
    assert np.allclose(F["Wv"], W["Wv"] @ np.diag(W["g"])) and F["Wo"] is W["Wo"]
