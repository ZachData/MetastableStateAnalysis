"""S2's arms in the core's coordinates (`trackS_subspace/design-S2.md` "The arms";
`tools/math_checks/s2_ov_arms.py`): the frame reproduces K, arm (a) is sym K, (b-schur) has real
eigenvalues, keeps the write space and edits by ‖S′ − S‖, (b-eig) has Re λ, the gain arm and the
doubly matched controls have the arm's norms, and the R1b overlap and R2 shuffle null read right."""

import numpy as np
import pytest

from trackS_subspace import s2

pytestmark = pytest.mark.pure  # numpy only

D, K = 48, 8


def _frame(seed):
    g = np.random.default_rng(seed)
    W_O, W_V = g.standard_normal((D, K)), g.standard_normal((K, D))
    gamma = 0.5 + g.random(D)
    return W_O, W_V, gamma, s2.head_frame(W_O, W_V, gamma)


def _dense(f, M):
    return f["Q"] @ M @ f["Q"].T


def test_frame_reproduces_the_centred_folded_map():
    W_O, W_V, gamma, f = _frame(0)
    Pi = np.eye(D) - np.ones((D, D)) / D
    want = Pi @ W_O @ W_V @ np.diag(gamma) @ Pi
    assert np.abs(_dense(f, f["K"]) - want).max() < 1e-10 * np.abs(want).max()
    assert np.allclose(f["Q"].T @ f["Q"], np.eye(2 * K), atol=1e-12)


def test_arm_a_is_the_symmetric_part():
    *_, f = _frame(1)
    M, _ = s2.arm_core(f, "a")
    Kd = _dense(f, f["K"])
    assert np.allclose(_dense(f, M), (Kd + Kd.T) / 2, atol=1e-10)


def test_b_schur_real_eigenvalues_write_space_and_size():
    for seed in range(20):
        *_, f = _frame(seed)
        M, info = s2.arm_core(f, "b")
        ev = np.linalg.eigvals(M)
        big = ev[np.abs(ev) > 1e-8 * np.abs(ev).max()]
        assert np.abs(big.imag).max() < 1e-7 * np.abs(big).max()
        assert s2.match_eigs(big, info["eig"]) < 1e-7 * np.linalg.norm(f["K"])
        Ub = np.linalg.qr(f["Bq"])[0]
        assert np.abs(Ub @ Ub.T @ M - M).max() < 1e-9 * np.abs(M).max()      # write space col(B)
        assert info["pairs"] >= 1


def test_b_schur_matches_the_math_check_formula():
    """K′ = U V S′ Vᵀ Uᵀ + U N (C7) equals the module's K + U V (S′ − S) Vᵀ Uᵀ."""
    import scipy.linalg as sla
    *_, f = _frame(3)
    U, R = np.linalg.qr(f["Bq"])
    T = R @ f["Cq"].T @ U
    N = R @ f["Cq"].T @ (np.eye(2 * K) - U @ U.T)
    S, V = sla.schur(T, output="real")
    S2 = S.copy()
    for i, m in s2.schur_blocks(S):
        if m == 2:
            S2[i:i + 2, i:i + 2] = s2.sym(S[i:i + 2, i:i + 2])
    want = U @ V @ S2 @ V.T @ U.T + U @ N
    M, _ = s2.arm_core(f, "b")
    assert np.allclose(M, want, atol=1e-10)
    assert np.isclose(np.linalg.norm(M - f["K"]), np.linalg.norm(S2 - S))


def test_b_eig_has_re_lambda_on_ks_eigenvectors():
    *_, f = _frame(4)
    M, info = s2.arm_core(f, "b_eig")
    lam = np.linalg.eigvals(f["Cq"].T @ f["Bq"])
    ev = np.linalg.eigvals(M)
    big = ev[np.abs(ev) > 1e-8 * np.abs(ev).max()]
    assert not info["kept"]
    assert s2.match_eigs(big, lam.real) < 1e-7 * np.abs(lam).max()


def test_gain_arm_and_controls_have_the_arms_norms_inside_the_span():
    *_, f = _frame(5)
    for arm in ("a", "b", "b_eig"):
        Kp, _ = s2.arm_core(f, arm)
        assert np.isclose(np.linalg.norm(s2.gain_core(f, Kp)), np.linalg.norm(Kp))
        Ww, Wr = s2.arm_span(f, arm)
        for i in range(5):
            C = s2.control_core(f, Kp, (Ww, Wr), np.random.default_rng(i))
            assert np.isclose(np.linalg.norm(C - f["K"]), np.linalg.norm(Kp - f["K"]), rtol=1e-10)
            assert np.isclose(np.linalg.norm(C), np.linalg.norm(Kp), rtol=1e-10)
            alpha = s2.ip(Kp - f["K"], f["K"]) / s2.ip(f["K"], f["K"])
            X = C - (1 + alpha) * f["K"]                    # the random part: in the span, ⟂ K
            assert np.abs(Ww @ Ww.T @ X @ Wr @ Wr.T - X).max() < 1e-9 * max(np.abs(X).max(), 1e-300)
            assert abs(s2.ip(X, f["K"])) < 1e-9 * np.linalg.norm(X) * np.linalg.norm(f["K"])


def test_controls_differ_by_draw():
    *_, f = _frame(6)
    Kp, _ = s2.arm_core(f, "a")
    sp = s2.arm_span(f, "a")
    a = s2.control_core(f, Kp, sp, np.random.default_rng(0))
    b = s2.control_core(f, Kp, sp, np.random.default_rng(1))
    assert np.linalg.norm(a - b) > 1e-3 * np.linalg.norm(a)


def test_schur_blocks():
    S = np.array([[1, 2, 0, 0], [-3, 1, 0, 0], [0, 0, 5, 1], [0, 0, 0, 6.]])
    assert s2.schur_blocks(S) == [(0, 2), (2, 1), (3, 1)]


def test_sizes_and_summary():
    Kc = np.eye(3)
    assert s2.sizes(Kc, 2 * Kc) == pytest.approx((1.0, 2.0))
    assert s2.size_summary([1, 2, 3]) == {"median": 2.0, "min": 1.0, "max": 3.0}


def test_overlap_and_its_haar_mean():
    g = np.random.default_rng(7)
    U = np.linalg.qr(g.standard_normal((64, 4)))[0]
    assert s2.overlap(U, U) == pytest.approx(1.0)
    ho = s2.haar_overlap(4, 400, 1, d=64)
    assert abs(ho.mean() - 4 / 64) < 0.01                  # E tr(P_U P_W) / k = k / d


def test_probe_first_eight_are_7ds():
    ids = s2.probe_ids()
    rng = np.random.default_rng(s2.EVAL_SEED)
    first = rng.integers(*s2.VOCAB, size=s2.N_REP)
    assert ids.shape == (16, 2 * s2.N_REP)
    assert np.array_equal(ids[0, :s2.N_REP], first) and np.array_equal(ids[0, s2.N_REP:], first)
