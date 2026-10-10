"""
`tools/run/p10_e1p_project.py` — E1p, E1m on the moves less their group-blind part, as `design-10.md`
"E1p" fixed it, including the calibration the rule asks for before any real move.
"""
import numpy as np
import pytest

from tools.run import p10_e1_energy as e1
from tools.run import p10_e1m_match as m
from tools.run import p10_e1p_project as p

pytestmark = pytest.mark.deps


def _unit(A):
    return A / np.linalg.norm(A, axis=-1, keepdims=True)


def _tangent(U, V):
    return V - np.sum(V * U, axis=1, keepdims=True) * U


def _one_cloud(seed):
    """E1m's planted cloud: the group a random sample of a cloud as near as the members."""
    rng = np.random.default_rng(seed)
    c = _unit(rng.standard_normal(24))
    U = _unit(rng.standard_normal((90, 24)))
    U[20:30] = _unit(c + 0.5 * rng.standard_normal((10, 24)))
    U[30:60] = _unit(c + 0.5 * rng.standard_normal((30, 24)))
    return U, np.arange(20, 30)


def _blobs(seed):
    """Six density blobs of 10 among scattered rows; the group is one blob, after rows the causal view
    can pair with."""
    rng = np.random.default_rng(seed)
    U = _unit(rng.standard_normal((90, 24)))
    for b in range(6):
        U[30 + b * 10:40 + b * 10] = _unit(_unit(rng.standard_normal(24)) + 0.5 * rng.standard_normal((10, 24)))
    return U, np.arange(60, 70)


def _field(U, beta, vis="full"):
    """The idealised field over the rows each row sees (the first causal row: no move)."""
    D = np.zeros_like(U)
    for i in range(len(U)):
        v = m.visible(vis, len(U), i)
        if len(v):
            z = beta * (U[v] @ U[i])
            w = np.exp(z - z.max())
            D[i] = (w / w.sum()) @ U[v] - U[i]
    return D


def _nn(U, k, vis="full"):
    D = np.zeros_like(U)
    for i in range(len(U)):
        v = m.visible(vis, len(U), i)
        if len(v):
            D[i] = U[v[np.argsort(-(U[v] @ U[i]), kind="stable")[:k]]].mean(0) - U[i]
    return D


def _member(U, rows):
    D = np.zeros_like(U)
    for i in range(len(U)):
        D[i] = U[[j for j in rows if j != i]].mean(0) - U[i]
    return D


def _scores(U, rows, D, vis="full"):
    S = U @ U.T
    D = np.repeat(_tangent(U, D)[None], len(m.READS), axis=0)
    r = p.residual(D, p.basis(U, S, vis))
    fr = {"blk": p.blk_of(U, S, D)}
    res = {"blk": p.blk_of(U, S, r["R"]), "share": r["share"]}
    return p.score_group(fr, res, rows, vis)


def _mean_Xp(cloud, move, vis="full", n=20):
    out = []
    for seed in range(n):
        U, rows = cloud(seed)
        out.append(_scores(U, rows, move(U, rows, seed), vis)["reads"]["attn:r1out"]["3.5"]["X"])
    out = np.asarray(out)
    return np.nanmean(out), np.nanstd(out) / np.sqrt(max(np.isfinite(out).sum(), 1)), int(np.isfinite(out).sum())


def test_frame_moves_match_e1_block_frame():
    rng = np.random.default_rng(0)
    n, d = 30, 16
    x = rng.standard_normal((n, d)).astype(np.float32)
    comps = {k: 0.3 * rng.standard_normal((n, d)).astype(np.float32) for k in ("attn", "keys", "mlpx", "block")}
    w, b, t = 1 + 0.1 * rng.standard_normal(d), 0.1 * rng.standard_normal(d), np.array([0, 2, 3, 7, 9, 11, 20, 25])
    want = e1.block_frame(x, comps, w, b, 1e-5, t)
    fr = p.frame_moves(x, comps, w, b, 1e-5, t)
    blk = p.blk_of(fr["U"], fr["S"], fr["D"])
    assert np.allclose(blk["M"], want["M"], atol=1e-12) and np.allclose(blk["dn"], want["dn"], atol=1e-12)
    assert np.allclose(fr["S"], want["S"], atol=1e-12)


@pytest.mark.parametrize("vis", ["full", "causal"])
def test_residual_is_orthogonal_to_every_basis_vector_and_a_move_in_the_basis_vanishes(vis):
    U, rows = _blobs(2)
    S = U @ U.T
    Q = p.basis(U, S, vis)
    assert np.allclose(np.einsum("nkd,nd->nk", Q, U), 0, atol=1e-10)          # tangent at u_i (SVD noise ~1e-13)
    D = np.repeat(_tangent(U, np.random.default_rng(5).standard_normal(U.shape))[None], 2, axis=0)
    R = p.residual(D, Q)["R"]
    assert np.allclose(np.einsum("cnd,nkd->cnk", R, Q), 0, atol=1e-12)
    inside = 0.7 * _field(U, 3.5, vis) - 0.2 * _nn(U, 8, vis) + 0.1 * _field(U, 0.0, vis)
    Rin = p.residual(_tangent(U, inside)[None], Q)["R"]
    assert np.linalg.norm(Rin, axis=-1).max() < e1.TINY
    out = _scores(U, rows, inside, vis)
    assert np.isnan(out["reads"]["attn:r1out"]["3.5"]["X"])                    # no X_p: nothing is left
    assert np.isfinite(out["raw"]["attn:r1out"]["3.5"])                        # E1m's X is still there


def test_an_empty_basis_leaves_e1ms_score():
    U, rows = _one_cloud(3)
    D = np.repeat(_tangent(U, np.random.default_rng(9).standard_normal(U.shape))[None], len(m.READS), axis=0)
    S = U @ U.T
    Q = np.zeros((len(U), 5, U.shape[1]))
    r = p.residual(D, Q)
    assert np.array_equal(r["R"], D)
    a = m.score_group(p.blk_of(U, S, r["R"]), rows, "full")
    b = m.score_group(p.blk_of(U, S, D), rows, "full")
    assert a["reads"]["attn:r1out"]["3.5"]["X"] == b["reads"]["attn:r1out"]["3.5"]["X"]


def test_the_first_causal_row_has_no_basis_and_the_second_one_direction():
    U, _ = _one_cloud(1)
    Q = p.basis(U, U @ U.T, "causal")
    rank = (np.linalg.norm(Q, axis=-1) > 0.5).sum(axis=1)
    assert rank[0] == 0 and rank[1] == 1 and rank[20] == 5


def test_the_causal_basis_is_blind_to_later_rows():
    U, _ = _one_cloud(4)
    U2 = U.copy()
    U2[60:] = _unit(np.random.default_rng(1).standard_normal((30, U.shape[1])))
    Q1, Q2 = p.basis(U, U @ U.T, "causal")[:60], p.basis(U2, U2 @ U2.T, "causal")[:60]
    P1, P2 = np.einsum("nkd,nke->nde", Q1, Q1), np.einsum("nkd,nke->nde", Q2, Q2)    # the projectors
    assert np.allclose(P1, P2, atol=1e-10)


@pytest.mark.parametrize("vis", ["full", "causal"])
@pytest.mark.parametrize("cloud", [_one_cloud, _blobs])
@pytest.mark.parametrize("pull", ["field8", "nn4"])
def test_a_group_blind_pull_outside_the_basis_scores_zero(vis, cloud, pull):
    """The rule's calibration: E1m's pairing scored X ≈ +0.14 to +0.24 here; the residual does not.
    The pull is the view's own (a causal pull sees the rows before i): a full-view pull scored in the
    causal view is not group-blind there, it reaches rows the basis cannot."""
    move = {"field8": lambda U, r, s: _field(U, 8.0, vis), "nn4": lambda U, r, s: _nn(U, 4, vis)}[pull]
    mean, se, k = _mean_Xp(cloud, move, vis)
    assert k >= 15
    assert abs(mean) < 0.05 and abs(mean) < 4 * se + 0.02, (mean, se)


@pytest.mark.parametrize("vis", ["full", "causal"])
@pytest.mark.parametrize("cloud", [_one_cloud, _blobs])
def test_a_pull_towards_the_members_survives_and_a_random_move_does_not(vis, cloud):
    mean, _, _ = _mean_Xp(cloud, lambda U, r, s: _member(U, r), vis)
    assert mean > 0.3
    rnd, se, _ = _mean_Xp(cloud, lambda U, r, s: np.random.default_rng(1000 + s).standard_normal(U.shape), vis)
    assert abs(rnd) < 0.05 and abs(rnd) < 4 * se + 0.02


def test_the_removed_share_is_one_inside_the_basis_and_small_for_a_random_move():
    U, rows = _blobs(6)
    inside = _scores(U, rows, _field(U, 1.6))["removed"]["attn:r1out"]
    rnd = _scores(U, rows, np.random.default_rng(2).standard_normal(U.shape))["removed"]["attn:r1out"]
    assert inside == pytest.approx(1.0, abs=1e-9) and rnd < 0.5
