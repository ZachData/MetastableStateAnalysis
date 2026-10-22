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


# ---------------------------------------------------------------- unit 2: labels, readings, G2, G3′

def test_label_ties_go_to_as_its_controls():
    ctl = [3, 4, 5]
    assert s2.label(2, ctl) == "lower" and s2.label(6, ctl) == "higher"
    assert s2.label(3, ctl) == "as its controls" and s2.label(5, ctl) == "as its controls"


def test_gain_rule_reads_a_label_only_beyond_the_gain_arm():
    ctl = list(range(30, 40))
    assert not s2.cell(20, ctl, gain=25)["as_its_gain"]
    assert s2.cell(20, ctl, gain=20)["as_its_gain"]                   # a tie with the gain arm
    assert s2.cell(20, ctl, gain=10)["as_its_gain"]
    assert s2.cell(50, ctl, gain=55)["as_its_gain"]
    assert not s2.cell(35, ctl, gain=10)["as_its_gain"]               # unlabelled: nothing to withhold


def test_r1_moved_further_is_beside_and_the_frozen_guard_one_sided():
    ctl, dc = list(range(30, 40)), [0.5, 0.6]
    low_far = s2.r1_read(s2.cell(5, ctl, gain=50, d=0.9, d_ctl=dc))
    assert low_far == {"label": "lower, moved further", "read": True, "counts": True}
    frozen = s2.r1_read(s2.cell(5, ctl, gain=50, d=0.9, d_ctl=dc), guard=True)
    assert frozen == {"label": "lower, moved further", "read": False, "counts": False}
    assert s2.r1_read(s2.cell(60, ctl, gain=50, d=0.9, d_ctl=dc), guard=True)["read"]
    high_far = s2.r1_read(s2.cell(60, ctl, gain=50, d=0.9, d_ctl=dc))
    assert high_far["read"] and high_far["label"] == "higher, moved further"
    assert s2.r1_read(s2.cell(5, ctl, gain=50, d=0.55, d_ctl=dc))["counts"]
    assert not s2.r1_read(s2.cell(8, ctl, gain=50, d=0.55, d_ctl=dc))["counts"]        # V > 7
    less = s2.r1_read(s2.cell(5, ctl, gain=50, d=0.1, d_ctl=dc))
    assert less["counts"] and less["label"] == "lower, moved less"


def test_r2_outcomes_and_withholding():
    ctl, dc, gb = [8.0, 9.0, 10.0], [0.5, 0.6], 12.0
    assert s2.r2_read(s2.cell(4.0, ctl, gain=11, d=0.55, d_ctl=dc), gb)["label"] == "lost"
    assert s2.r2_read(s2.cell(7.0, ctl, gain=11, d=0.55, d_ctl=dc), gb)["label"] == "weakened"   # lower, G ≥ half
    assert s2.r2_read(s2.cell(9.0, ctl, gain=11, d=0.55, d_ctl=dc), gb)["label"] == "survives"
    assert s2.r2_read(s2.cell(5.0, [1.0, 2.0], gain=1.5, d=0.55, d_ctl=dc), gb)["label"] == "weakened"
    far = s2.r2_read(s2.cell(4.0, ctl, gain=11, d=0.9, d_ctl=dc), gb)
    assert far == {"label": "lost, moved further", "read": True, "outcome": "lost"}
    assert not s2.r2_read(s2.cell(4.0, ctl, gain=11, d=0.9, d_ctl=dc), gb, guard=True)["read"]
    assert s2.r2_read(s2.cell(9.0, ctl, gain=11, d=0.9, d_ctl=dc), gb, guard=True)["read"]   # survives is read
    g = s2.r2_read(s2.cell(4.0, ctl, gain=3, d=0.55, d_ctl=dc), gb)
    assert not g["read"] and g["label"] == "weakened, as its gain"


def test_g2_draws_layer_matched_without_members():
    from trackB_circuit.b2a import parse_head
    members = [parse_head(m) for m in s2.G2_MEMBERS]
    draws = s2.g2_draws()
    assert len(draws) == s2.N_CTL and draws == s2.g2_draws()
    for row in draws:
        assert [l for l, _ in row] == [l for l, _ in members]
        assert len(set(row)) == len(row) and not set(row) & set(members)
    assert len({tuple(r) for r in draws}) == s2.N_CTL


def test_zeroed_leaves_the_rest():
    cores = [np.ones((2, 3, 3)), np.ones((2, 3, 3))]
    z = s2.zeroed(cores, [(1, 0)])
    assert not z[1][0].any() and z[1][1].all() and z[0].all() and cores[1][0].all()


def test_g3p_frame_holds_both_steps_maps_and_matches_controls():
    W_O, W_V, gamma, fs = _frame(11)
    W_O2, W_V2, gamma2, ft = _frame(12)
    f4, Kt, span = s2.g3p_frame(fs, ft)
    Q = f4["Q"]
    assert np.allclose(Q.T @ Q, np.eye(4 * K), atol=1e-12)
    assert np.allclose(Q @ f4["K"] @ Q.T, _dense(fs, fs["K"]), atol=1e-10)
    assert np.allclose(Q @ Kt @ Q.T, _dense(ft, ft["K"]), atol=1e-10)
    C = s2.control_core(f4, Kt, span, np.random.default_rng(0))
    assert np.isclose(np.linalg.norm(C - f4["K"]), np.linalg.norm(Kt - f4["K"]), rtol=1e-10)
    assert np.isclose(np.linalg.norm(C), np.linalg.norm(Kt), rtol=1e-10)


def _gr(g1=True, g2=True, counts=(True,) * 5, mono=(False,) * 5):
    cells = {f"1.0@{s}": {"counts": c} for s, c in zip(s2.SCORED_STEPS, counts)}
    return {"G1": {"pass": g1}, "G2": {"pass": g2},
            "G3": {"pass": sum(counts) >= 4, "cells": cells,
                   "controls_monotone_at_1": {str(s): m for s, m in zip(s2.SCORED_STEPS, mono)}}}


def test_gate_refusals_follow_the_fallbacks():
    assert s2.gate_refusals(_gr()) == {"R1": None, "R2": None, "g3_mono": False}
    r = s2.gate_refusals(_gr(g1=False))
    assert r["R1"] and r["R2"]
    assert s2.gate_refusals(_gr(g2=False))["R2"] and not s2.gate_refusals(_gr(g2=False))["R1"]
    fail = (True, True, True, False, False)
    assert s2.gate_refusals(_gr(counts=fail))["R1"] == "G3 failed"                       # not by monotone controls
    r = s2.gate_refusals(_gr(counts=fail, mono=(False, False, False, True, True)))
    assert r["R1"] is None and r["g3_mono"]


def test_g3p_base_refuses_on_the_stored_record():
    assert s2.check_g3p_base(512, {"g3p_base": {"D": 5e-7}}) == 5e-7
    with pytest.raises(s2.S2Error, match="not the base"):
        s2.check_g3p_base(512, {"g3p_base": {"D": 2 * s2.G3P_BASE_TOL}})
