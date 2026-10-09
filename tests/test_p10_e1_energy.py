"""
`tools/run/p10_e1_energy.py` — E1, energy against c3x's groups, as `design-10.md` "E1" fixed it.
"""
import numpy as np
import pytest

from tools.run import p10_e1_energy as e

# Tier: the module imports the label source, whose package imports scikit-learn (as R7's tests).
pytestmark = pytest.mark.deps


def _unit(A):
    return A / np.linalg.norm(A, axis=-1, keepdims=True)


def _tangent(U, V):
    return V - np.sum(V * U, axis=1, keepdims=True) * U


def _cloud(n=40, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal((n, d)))


def _blk(U, D):
    """`block_frame`'s outputs from given unit rows and moves (C, n, d)."""
    dn = np.linalg.norm(D, axis=-1)
    ok = dn >= e.TINY
    Dh = np.where(ok[..., None], D / np.maximum(dn, 1e-300)[..., None], 0.0)
    return {"U": U, "S": U @ U.T, "M": np.einsum("cnd,md->cnm", D, U), "dn": dn, "ok": ok,
            "Q": np.einsum("cnd,cmd->cnm", Dh, Dh)}


def _explicit(U, d, i, src, beta):
    """cos(d_i, P⊥ Σ_{j∈src∖{i}} softmax(β u_i·u_j) u_j), by vectors."""
    src = [j for j in src if j != i]
    z = np.array([beta * U[i] @ U[j] for j in src])
    w = np.exp(z - z.max())
    m = (w[:, None] * U[src]).sum(axis=0) / w.sum()
    g = m - (m @ U[i]) * U[i]
    return float(d[i] @ g / (np.linalg.norm(d[i]) * np.linalg.norm(g)))


@pytest.mark.parametrize("beta", [0.0, 1.6, 3.5, 5.6])
def test_set_cosines_match_the_vector_field(beta):
    U = _cloud()
    rng = np.random.default_rng(1)
    D = _tangent(U, rng.standard_normal(U.shape))[None]
    b = _blk(U, D)
    rows = np.array([2, 5, 9])
    sets = np.array([[2, 5, 9, 11], [1, 3, 4, 30]])
    a = e.set_cosines(b["S"], b["M"], b["dn"], rows, sets, beta)
    for k, s in enumerate(sets):
        for m, i in enumerate(rows):
            assert a[0, k, m] == pytest.approx(_explicit(U, D[0], i, s, beta), abs=1e-10)


def test_a_member_inside_its_own_set_is_left_out():
    """The tangent drops u_i, so S ∖ {i} and S ∪ {i} give the same cosine; the code leaves i out."""
    U = _cloud()
    D = _tangent(U, np.random.default_rng(2).standard_normal(U.shape))[None]
    b = _blk(U, D)
    rows = np.array([3, 7])
    with_self = e.set_cosines(b["S"], b["M"], b["dn"], rows, np.array([[3, 7, 12]]), 3.5)
    assert with_self[0, 0, 0] == pytest.approx(_explicit(U, D[0], 3, [7, 12], 3.5), abs=1e-10)
    assert with_self[0, 0, 1] == pytest.approx(_explicit(U, D[0], 7, [3, 12], 3.5), abs=1e-10)


def _planted(sign):
    """Members 0–5 near each other; each member's move points at (or away from) the others' mean."""
    rng = np.random.default_rng(3)
    c = _unit(rng.standard_normal(16))
    U = _cloud(60, 16, 4)
    U[:6] = _unit(c + 0.3 * rng.standard_normal((6, 16)))
    D = np.zeros_like(U)
    for i in range(60):
        pool = [j for j in range(6) if j != i] if i < 6 else []
        D[i] = (sign * (U[pool].mean(axis=0) - U[i]) if pool else rng.standard_normal(16))
    D = _tangent(U, D)
    D[:6] += 0.05 * _tangent(U[:6], rng.standard_normal((6, 16)))
    return U, np.repeat(D[None], len(e.READS), axis=0)


@pytest.mark.parametrize("sign,want", [(1, 1), (-1, -1)])
def test_a_planted_pull_gives_positive_X_and_a_push_negative(sign, want):
    U, D = _planted(sign)
    rows, pool = np.arange(6), np.arange(6, 60)
    out = e.score_group(_blk(U, D), rows, pool, e.draw_sets("t", pool, 6))
    X = out["reads"]["attn:r1out"]["3.5"]["X"]
    assert np.sign(X) == want and abs(X) > 0.2


def test_a_move_independent_of_position_gives_X_near_zero():
    U = _cloud(80, 16, 5)
    D = _tangent(U, np.random.default_rng(6).standard_normal(U.shape))
    D = np.repeat(D[None], len(e.READS), axis=0)
    Xs = []
    for g in range(10):
        rows = np.arange(8 * g, 8 * g + 8)
        pool = np.setdiff1d(np.arange(80), rows)
        Xs.append(e.score_group(_blk(U, D), rows, pool, e.draw_sets(f"z{g}", pool, 8))["reads"]["attn:r1out"]["3.5"]["X"])
    assert abs(np.mean(Xs)) < 0.1


def test_draws_never_hold_a_member_and_are_reproducible():
    pool = np.setdiff1d(np.arange(50), [4, 8, 15])
    a, b = e.draw_sets("k", pool, 3), e.draw_sets("k", pool, 3)
    assert a.shape == (e.N_DRAWS, 3) and np.array_equal(a, b)
    assert not np.isin(a, [4, 8, 15]).any()
    assert all(len(set(r)) == 3 for r in a.tolist())
    with pytest.raises(e.E1Error):
        e.draw_sets("k", np.arange(2), 3)


def test_coherence_is_the_mean_pairwise_cosine():
    rng = np.random.default_rng(7)
    D = rng.standard_normal((1, 10, 5))
    b = _blk(_cloud(10, 5), D)
    s = np.array([[0, 2, 3, 7]])
    Dh = _unit(D[0, s[0]])
    want = np.mean([Dh[i] @ Dh[j] for i in range(4) for j in range(4) if i != j])
    assert e.coherence(b["Q"], b["ok"], s)[0, 0] == pytest.approx(want)


def test_parallel_moves_score_coherence_but_not_the_force():
    """A shared tag (every member moves along one direction) is coherent, not towards the members."""
    U, _ = _planted(1)
    t = _unit(np.random.default_rng(8).standard_normal(16))
    D = _tangent(U, np.tile(t, (60, 1)) + 0.01 * np.random.default_rng(9).standard_normal((60, 16)))
    D[6:] = _tangent(U[6:], np.random.default_rng(10).standard_normal((54, 16)))
    D = np.repeat(D[None], len(e.READS), axis=0)
    rows, pool = np.arange(6), np.arange(6, 60)
    out = e.score_group(_blk(U, D), rows, pool, e.draw_sets("p", pool, 6))["reads"]["attn:r1out"]
    assert out["coherence"]["X"] > 0.5
    assert abs(out["3.5"]["X"]) < out["coherence"]["X"]


def test_sign_rule():
    assert e.sign_label([1, 2, 3, 4, 5, 6, 7]) == "pulls together"
    assert e.sign_label([1, 2, 3, 4, 5, 6, -7]) == "leans pulls"
    assert e.sign_label([-1, -2, -3, -4, -5, -6]) == "pushes apart"
    assert e.sign_label([1, 2, 3, 4, 5, float("nan"), float("nan")]) == "too few"
    assert e.sign_label([1, 2, 3, -4, -5, 6, 7]) == "mixed"


def test_finite_share_leaves_nan_out():
    assert e.finite_share_pos(np.array([1.0, -1.0, np.nan])) == 0.5
    assert np.isnan(e.finite_share_pos(np.array([np.nan])))


def test_report_refuses_a_missing_passage(tmp_path):
    (tmp_path / "records").mkdir()
    import json
    rec = {"step": 512, "passage": "wiki_paragraph", "code": "x", "groups": []}
    (tmp_path / "records" / "step512_wiki_paragraph.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="missing"):
        e.main(["report", "--out", str(tmp_path)])
