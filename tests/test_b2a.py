"""B2a unit 1's parts (`trackB_circuit/design-B2a.md`): the reference draw, V's ladder, the U_2
type, the |b| selection row, the offset bins, M1's convex fit and M2's exactness on the QK space."""

import math

import numpy as np
import pytest

from trackB_circuit import b2a

pytestmark = pytest.mark.deps  # torch (M1's fit)


def _V(S, d=40, seed=0):
    g = np.random.default_rng(seed)
    P, _ = np.linalg.qr(g.standard_normal((d, S.shape[0])))
    return P @ S, P.T, P


def _rot(nu, phi):
    return nu * np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])


def test_reference_heads_are_8_fixed_heads_outside_the_named_six():
    r = b2a.reference_heads()
    assert r == b2a.reference_heads() and len(set(r)) == 8
    assert not set(r) & set(b2a.NAMED)
    assert all(0 <= b2a.parse_head(h)[0] <= 22 for h in r)
    assert len(b2a.heads()) == 14


def test_ladder_orders_by_modulus_keeps_pairs_whole_and_is_invariant():
    S = np.diag([0.3, -0.2, 0.1, 0.05, 0.0, 0.0])
    S[:2, :2] = _rot(3.0, 2.0)                       # top: a complex pair
    S[2, 2] = -1.5
    left, right, P = _V(S)
    bl = b2a.blocks(left, right)
    assert [b["dim"] for b in bl] == [2, 1, 1]       # the two zeros are the kernel
    assert np.allclose([abs(b["lam"]) for b in bl], [3.0, 1.5, 0.05])
    lad = b2a.ladder(left, right)
    assert [r["D"] for r in lad] == [0, 2, 3, 4]     # rung 1 moved up into the pair; capped at rank 4
    assert max(r["residual"] for r in lad) < 1e-12
    # sin of the largest principal angle, exact to rounding; plane_angle's arccos near 1 resolves
    # only to sqrt(2 eps) ~ 1.5e-8, which BLAS rounding reaches on some runners (CI, 2026-10-16)
    U, P2 = lad[1]["U"], P[:, :2]
    assert np.linalg.norm(U - P2 @ (P2.T @ U), 2) < 1e-10
    assert b2a.plane_angle(U, P2) < 1e-7


def test_u2_type():
    def t(S):
        return b2a.u2_type(b2a.blocks(*_V(S)[:2]))["type"]
    assert t(np.diag([-2.0, -2.0, 0.3])) == "degenerate negative"
    assert t(np.diag([2.0, 1.95, 0.3])) == "degenerate positive"
    assert t(np.diag([-2.0, 1.0, 0.3])) == "mixed"
    S = np.diag([0.0, 0.0, 0.3])
    S[:2, :2] = _rot(2.0, 2.5)
    assert t(S) == "rotation"


def test_selection_by_b_and_the_kernel():
    left, right, _ = _V(np.diag([-2.0, 0.5, 0.0]))
    s = b2a.selection(b2a.blocks(left, right), 0.3, 40)
    # |b| = |0.7 − 0.6| = 0.1, |0.7 + 0.15| = 0.85; kernel 0.7
    assert s["b_top"] == pytest.approx(0.85) and s["sel_top_block"] == 1
    assert not s["sel_top_is_lam_top"] and s["kernel_beaten"]
    assert s["rho_perp"] == pytest.approx(0.7 / 0.85)
    left, right, _ = _V(np.diag([-1.0, -0.5, 0.0]))
    assert not b2a.selection(b2a.blocks(left, right), 0.3, 40)["kernel_beaten"]


def test_offset_bins():
    assert [b2a.bin_of(k) for k in (0, 15, 16, 31, 32, 63, 64, 1024, 2047)] == \
        [0, 15, 16, 16, 17, 17, 18, 22, 22]
    assert b2a.n_bins() == 23


def test_m1_fit_recovers_beta_and_profile_and_tv_is_zero_on_its_own_model():
    torch = pytest.importorskip("torch")
    g = np.random.default_rng(1)
    n, d = 80, 6
    theta = torch.zeros((1, b2a.n_bins() + 1), dtype=torch.float64)
    theta[0, 0], theta[0, 2], theta[0, 17] = 3.0, 2.0, -1.0
    us, seqs, cs = [], [], []
    for _ in range(4):
        u = g.standard_normal((n, d))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        ut = torch.as_tensor(u)
        probe = b2a.Seq(np.eye(n), "cpu")
        c = probe.off(b2a.cos_feat(ut, None))
        P = (b2a._logp(theta, c, probe).exp() * probe.valid)[0]
        A = np.zeros((n, n))                                                # back to key coords
        for i in range(n):
            for k in range(i + 1):
                A[i, i - k] = float(P[i, k])
        s = b2a.Seq(A, "cpu")
        us.append(ut), seqs.append(s), cs.append(s.off(b2a.cos_feat(ut, None)))
    th, info = b2a.fit(cs, seqs)
    assert th[0, 0].item() == pytest.approx(3.0, abs=1e-3)
    assert (th[0, 2] - th[0, 1]).item() == pytest.approx(2.0, abs=1e-3)   # φ up to a constant
    assert float(b2a.tv(th, cs[0], seqs[0])[0]) < 1e-5
    o = b2a.loo(us, seqs, None)
    assert o["tv"][0] < 1e-4 and o["beta"][0] == pytest.approx(3.0, abs=1e-2)


def test_m2_on_the_qk_read_space_is_exact_and_u_zero_is_position_only():
    g = np.random.default_rng(2)
    d, n = 200, 12
    W = {"Wq": g.standard_normal((64, d)), "Wk": g.standard_normal((64, d)),
         "bq": g.standard_normal(64), "bk": g.standard_normal(64)}
    y = g.standard_normal((n, d))
    U = b2a.qk_space(W)
    assert U.shape == (d, 128)
    L, Lm = b2a.head_logits(y, W), b2a.m2_logits(y, U, W)
    fin = np.isfinite(L)
    assert np.abs(L[fin] - Lm[fin]).max() < 1e-10
    L0 = b2a.m2_logits(y, np.zeros((d, 0)), W)
    L0b = b2a.m2_logits(g.standard_normal((n, d)), np.zeros((d, 0)), W)
    assert np.allclose(L0[fin], L0b[fin])                                   # reads no token


def test_the_rule_on_small_planted_heads(monkeypatch):
    """`read_head` end to end on CPU (d = 16, 3 prompts, 8 nulls): a planted plane reads D* = 2,
    an offset-only head D* = 0, strong random QK "outside" (/challenge-pr on #208, finding 5), weak
    random QK "outside" too (Blocked 51 (a))."""
    pytest.importorskip("torch")
    monkeypatch.setattr(b2a, "K_NULL", 8)
    monkeypatch.setattr(b2a, "NULL_CHUNK", 4)
    g = np.random.default_rng(3)
    d, n = 16, 60
    S = np.diag(g.uniform(-0.3, 0.3, 6))
    S[:2, :2] = -2.0 * np.eye(2)
    left, right, P = _V(S, d=d, seed=4)
    lad = b2a.ladder(left, right, rungs=(0, 1, 2, 4))

    def unit(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    plane = [unit(unit(g.standard_normal((n, 2))) @ P[:, :2].T + 0.05 * g.standard_normal((n, d)))
             for _ in range(3)]
    r = b2a.read_head(plane, [b2a.paper_attention(u) for u in plane], lad, "t-plane", "cpu")
    assert r["verdict"] == "D*" and r["D_star"] == 2 and r["nulls"]["2"]["beat"] == 0

    iso = [unit(g.standard_normal((n, d))) for _ in range(3)]
    phi = np.zeros(n)
    phi[1] = 5.0
    i = np.arange(n)
    Apos = b2a._causal_softmax(phi[np.clip(i[:, None] - i[None, :], 0, n - 1)])
    r = b2a.read_head(iso, [Apos] * 3, lad, "t-pos", "cpu")
    assert r["verdict"] == "D*" and r["D_star"] == 0

    G = g.standard_normal((d, d))
    r = b2a.read_head(iso, [b2a._causal_softmax(8.0 * (u @ G @ u.T)) for u in iso], lad, "t-out", "cpu")
    assert r["verdict"] == "outside"

    # Blocked 51 (a): weak random QK has TV(full) < 0.5 but M1's content term adds nothing
    r = b2a.read_head(iso, [b2a._causal_softmax(1.0 * (u @ G @ u.T)) for u in iso], lad, "t-weak", "cpu")
    assert r["tv_full"] < b2a.TV_OUT
    assert r["verdict"] == "outside" and r["outside_why"] == "content M1 cannot express"

    # Blocked 52 (a'): the same noise on a positional profile still reads D* = 0
    Ln = phi[np.clip(i[:, None] - i[None, :], 0, n - 1)]
    r = b2a.read_head(iso, [b2a._causal_softmax(Ln + 1.0 * (u @ G @ u.T)) for u in iso], lad, "t-posn", "cpu")
    assert r["tv_full"] > b2a.TV_TOL and r["verdict"] == "D*" and r["D_star"] == 0


def test_m1_attention_matches_the_fitted_model():
    """Unit 2's numpy Â (the beside rows) equals the torch M1 the rule fits, at a random θ."""
    torch = pytest.importorskip("torch")
    g = np.random.default_rng(5)
    n, d = 40, 12
    u = g.standard_normal((n, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    U = np.linalg.qr(g.standard_normal((d, 3)))[0]
    th = g.standard_normal(b2a.n_bins() + 1)
    A = b2a.m1_attention(u, U, th)
    s = b2a.Seq(A, "cpu")
    T = lambda x: torch.as_tensor(x, dtype=torch.float64)
    P = (b2a._logp(T(th)[None], s.off(b2a.cos_feat(T(u), T(U[None]))), s).exp() * s.valid)[0]
    assert float((P - s.A).abs().max()) < 1e-12


def test_renorm_refuses_a_row_with_all_mass_on_the_sink():
    """/challenge-pr on #209, finding 6: such a row would make TV NaN and the head read "none"."""
    A = np.tril(np.ones((4, 4)))
    A[2] = [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(b2a.B2aError):
        b2a.renorm(A / A.sum(1, keepdims=True))
