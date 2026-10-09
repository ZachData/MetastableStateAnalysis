"""
`tools/run/p10_e1m_match.py` — E1m, the membership-against-nearness re-run, as `design-10.md` "E1m" fixed it.
"""
import numpy as np
import pytest

from tools.run import p10_e1_energy as e1
from tools.run import p10_e1m_match as m

pytestmark = pytest.mark.deps


def _unit(A):
    return A / np.linalg.norm(A, axis=-1, keepdims=True)


def _tangent(U, V):
    return V - np.sum(V * U, axis=1, keepdims=True) * U


def _blk(U, D):
    D = np.asarray(D)
    dn = np.linalg.norm(D, axis=-1)
    ok = dn >= e1.TINY
    Dh = np.where(ok[..., None], D / np.maximum(dn, 1e-300)[..., None], 0.0)
    return {"U": U, "S": U @ U.T, "M": np.einsum("cnd,md->cnm", D, U), "dn": dn, "ok": ok,
            "Q": np.einsum("cnd,cmd->cnm", Dh, Dh)}


def test_match_member_pairs_equal_similarity_without_replacement():
    s_f = np.array([0.9, 0.5, 0.52, 0.1])
    s_n = np.array([0.51, 0.88, 0.3, 0.505, 0.9])
    fi, ni = m.match_member(s_f, s_n, 0.02)
    pairs = dict(zip(fi.tolist(), ni.tolist()))
    assert pairs[0] == 4 and pairs[2] == 0 and pairs[1] == 3        # 0.9↔0.9, 0.52↔0.51, 0.5↔0.505
    assert 3 not in pairs                                            # 0.1 has no non-member within eps
    assert len(set(ni.tolist())) == len(ni)
    assert np.all(np.abs(s_f[fi] - s_n[ni]) <= 0.02 + 1e-12)


def test_nothing_is_matched_when_every_member_is_nearer_than_every_non_member():
    fi, ni = m.match_member(np.array([0.9, 0.85]), np.array([0.3, 0.4, 0.5]), 0.02)
    assert fi.size == 0 and ni.size == 0


def test_pair_cosines_match_e1_set_cosines():
    rng = np.random.default_rng(0)
    U = _unit(rng.standard_normal((40, 16)))
    D = _tangent(U, rng.standard_normal(U.shape))[None]
    b = _blk(U, np.repeat(D, 2, axis=0))
    idx = np.array([3, 8, 20, 21])
    for i in (5, 8):
        a = m.pair_cosines(b["S"], b["M"], b["dn"], i, idx[idx != i], m.BETAS)
        for k, be in enumerate(m.BETAS):
            want = e1.set_cosines(b["S"], b["M"], b["dn"], np.array([i]), idx[None], be)[:, 0, 0]
            assert np.allclose(a[:, k], want, atol=1e-10)


def test_visibility():
    assert m.visible("full", 5, 2).tolist() == [0, 1, 3, 4]
    assert m.visible("causal", 5, 2).tolist() == [0, 1]
    assert m.visible("causal", 5, 0).size == 0


def _group_cloud(seed=1):
    """Group rows 20–29 around a centre; non-members spread, some on the group's flank at the same
    similarity to a member as its fellows (so the pairing has material)."""
    rng = np.random.default_rng(seed)
    c = _unit(rng.standard_normal(24))
    U = _unit(rng.standard_normal((90, 24)))
    U[20:30] = _unit(c + 0.5 * rng.standard_normal((10, 24)))
    U[30:60] = _unit(c + 0.5 * rng.standard_normal((30, 24)))        # a cloud as near as the members
    return U, np.arange(20, 30)


def test_a_pull_towards_the_members_scores_positive_and_away_negative():
    U, rows = _group_cloud()
    for sign, want in ((1, 1), (-1, -1)):
        D = np.zeros_like(U)
        for i in range(len(U)):
            others = [j for j in rows if j != i]
            D[i] = sign * (U[others].mean(axis=0) - U[i])
        D = np.repeat(_tangent(U, D)[None], len(m.READS), axis=0)
        out = m.score_group(_blk(U, D), rows, "full")
        cell = out["reads"]["attn:r1out"]["3.5"]
        assert np.sign(cell["X"]) == want and abs(cell["X"]) > 0.05, cell
        assert out["cover"]["members_paired"] >= m.MIN_MEMBERS


def test_a_pull_towards_near_tokens_whatever_they_are_scores_about_zero_where_the_pairs_exist():
    """Each token moves towards its own nearest 8 tokens, member or not: the matched non-members are as
    near as the matched fellows, so the pull does not separate them."""
    U, rows = _group_cloud(2)
    S = U @ U.T
    D = np.zeros_like(U)
    for i in range(len(U)):
        nn = np.argsort(-S[i])[1:9]
        D[i] = U[nn].mean(axis=0) - U[i]
    D = np.repeat(_tangent(U, D)[None], len(m.READS), axis=0)
    out = m.score_group(_blk(U, D), rows, "full")
    cell = out["reads"]["attn:r1out"]["3.5"]
    assert cell["members"] >= m.MIN_MEMBERS
    assert abs(cell["X"]) < 0.15 and abs(cell["X"]) < abs(cell["A_obs"])


def test_a_group_nearer_than_every_non_member_has_no_X_and_says_so():
    rng = np.random.default_rng(3)
    c = _unit(rng.standard_normal(24))
    U = _unit(rng.standard_normal((60, 24)))
    U[:8] = _unit(c + 0.05 * rng.standard_normal((8, 24)))           # a tight group
    D = np.repeat(_tangent(U, rng.standard_normal(U.shape))[None], len(m.READS), axis=0)
    out = m.score_group(_blk(U, D), np.arange(8), "full")
    assert out["cover"]["members_paired"] == 0 and out["cover"]["paired"] == 0
    assert all(np.isnan(out["reads"][k]["3.5"]["X"]) for k in out["reads"])


def test_causal_rows_see_only_earlier_rows():
    """Rows after every member are invisible to the causal view: replacing them changes nothing there
    (and changes the full view)."""
    U, rows = _group_cloud(4)
    rng = np.random.default_rng(11)
    D = _tangent(U, rng.standard_normal(U.shape))
    U2, D2 = U.copy(), D.copy()
    U2[60:] = _unit(rng.standard_normal((len(U) - 60, U.shape[1])))
    D2 = _tangent(U2, np.where(np.arange(len(U))[:, None] >= 60, rng.standard_normal(U.shape), D))
    D = np.repeat(_tangent(U, D)[None], len(m.READS), axis=0)
    D2 = np.repeat(D2[None], len(m.READS), axis=0)
    a = m.score_group(_blk(U, D), rows, "causal")["reads"]["attn:r1out"]["3.5"]
    b = m.score_group(_blk(U2, D2), rows, "causal")["reads"]["attn:r1out"]["3.5"]
    assert a["X"] == pytest.approx(b["X"], abs=1e-12) and a["A_all"] == pytest.approx(b["A_all"], abs=1e-12)
    fa = m.score_group(_blk(U, D), rows, "full")["reads"]["attn:r1out"]["3.5"]
    fb = m.score_group(_blk(U2, D2), rows, "full")["reads"]["attn:r1out"]["3.5"]
    assert abs(fa["A_cmp"] - fb["A_cmp"]) > 1e-6


def test_a_paired_nonmember_is_never_a_member_or_the_member_itself():
    U, rows = _group_cloud(5)
    S = U @ U.T
    inside = np.zeros(len(U), dtype=bool)
    inside[rows] = True
    for i in rows:
        v = m.visible("full", len(U), int(i))
        pf, pn = m.member_pairs(S, int(i), v[inside[v]], v[~inside[v]])
        assert not np.isin(pn, rows).any() and i not in pn and i not in pf
        assert len(set(pn.tolist())) == len(pn)
