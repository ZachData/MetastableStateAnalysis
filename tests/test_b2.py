"""B2 unit 1's parts (`trackB_circuit/design-B2.md`): the exact low-rank spectra against dense
ones, the columns, 2d's regime rule, the percentiles and the layer-matched null, C6's overlap and
its planting, and C7's reader."""

import numpy as np
import pytest

from p2d_operator_activation.gradient_flow_condition import head_regime
from trackB_circuit import b2

pytestmark = pytest.mark.pure  # numpy only

D, K = 96, 20


def _head(seed=0, d=D, k=K):
    g = np.random.default_rng(seed)
    return {"Wq": g.standard_normal((k, d)), "Wk": g.standard_normal((k, d)),
            "Wv": g.standard_normal((k, d)), "Wo": g.standard_normal((d, k)),
            "g": 1 + 0.3 * g.standard_normal(d)}


def test_sym_eigs_are_the_dense_symmetric_parts_nonzero_spectrum():
    g = np.random.default_rng(1)
    A, B = g.standard_normal((D, K)), g.standard_normal((D, K))
    S = 0.5 * (A @ B.T + B @ A.T)
    dense = np.linalg.eigvalsh(S)
    lr = b2.sym_eigs(A, B)
    assert np.allclose(np.sort(lr), np.sort(dense[np.argsort(-np.abs(dense))[:2 * K]]), atol=1e-10)
    assert (lr > 0).sum() == K and (lr < 0).sum() == K          # fact 1


def test_sym_eigs_survive_zero_columns():
    g = np.random.default_rng(2)
    A, B = g.standard_normal((D, K)), g.standard_normal((D, K))
    E = np.linalg.qr(g.standard_normal((D, 2)))[0]
    ev0 = b2.sym_eigs(A, B)
    ev1 = b2.sym_eigs(np.concatenate([A, 0 * E], 1), np.concatenate([B, E], 1))
    assert np.isclose(b2.share(ev0), b2.share(ev1)) and np.isclose(b2.top_weight(ev0)[0], b2.top_weight(ev1)[0])


def test_columns_match_dense_and_head_regime():
    for s in range(5):
        W = _head(s)
        row = b2.head_row(W)
        A, B = b2.v_factors(W, "ln")
        V = A @ B.T
        assert np.allclose(V, (W["g"][:, None] * ((np.eye(D) - 1 / D) @ W["Wo"])) @ W["Wv"])
        ev = np.linalg.eigvalsh(0.5 * (V + V.T))
        assert np.isclose(row["C1"], b2.share(ev)) and np.isclose(row["C2"], b2.top_weight(ev)[0])
        hr = head_regime(W["Wq"].T, W["Wk"].T, V)
        assert np.isclose(row["asym"], hr["asymmetry"]) and np.isclose(row["C4"], hr["align"])
        assert row["regime"] == hr["regime"]
        hr_raw = head_regime(W["Wq"].T, W["Wk"].T, W["Wo"] @ W["Wv"])
        assert np.isclose(row["C4_raw"], hr_raw["align"])


def test_regime_label_on_planted_heads():
    W = _head(7)
    W["Wk"] = W["Wq"].copy()
    m = b2.m_row(W)
    Aq, Bk, _ = m["_m"]
    for sign, want in ((1, "gradient_flow"), (-1, "repulsive_aligned")):
        r = b2.v_row(sign * 0.5 * np.concatenate([Aq, Bk], 1), np.concatenate([Bk, Aq], 1), m)
        assert r["regime"] == want and np.isclose(abs(r["C4"]), 1)
    assert m["asym"] < 1e-12 and np.isclose(m["C3"], 1)


def test_top_weight_and_negative_pair():
    w, pair = b2.top_weight(np.array([-1.0, -0.9, 0.5, 0.1]))
    assert np.isclose(w, -1 / 2.5) and pair
    assert not b2.top_weight(np.array([-1.0, -0.7, 0.5]))[1]
    assert not b2.top_weight(np.array([-1.0, 0.9, 0.5]))[1]


def test_percentiles_midrank():
    p = b2.percentiles(np.array([3.0, 1.0, 2.0, 2.0]))
    assert np.allclose(p, [3.5 / 4, 0.5 / 4, 2 / 4, 2 / 4])


def test_layer_pools_and_null_exclude_the_member():
    heads = [b2.head_index(m) for m in b2.MEMBERS]
    for i, pool in zip(heads, b2.layer_pools(heads)):
        assert len(pool) == 15 and i not in pool and set(pool // 16) == {i // 16}
    S = b2.draw_sets(b2.layer_pools(heads), 1000, 0)
    assert all((S[:, j] // 16 == heads[j] // 16).all() for j in range(6))


def test_member_rank_finds_an_extreme_set_and_not_a_typical_one():
    g = np.random.default_rng(3)
    D_ = b2.dev(b2.percentiles(g.standard_normal(b2.N_POP)))
    heads = [b2.head_index(m) for m in b2.MEMBERS]
    D_[heads] = 0.5
    assert b2.member_rank(D_, heads, 1, 2000)["rank"] >= 0.99
    D_[heads] = 0.25
    assert b2.member_rank(D_, heads, 1, 2000)["rank"] < 0.99


def test_comp_mean_is_the_pairwise_overlap():
    g = np.random.default_rng(4)
    n = b2.N_POP
    Wb = np.stack([np.linalg.qr(g.standard_normal((D, 8)))[0] for _ in range(n)])
    Kb = np.stack([np.linalg.qr(g.standard_normal((D, 8)))[0] for _ in range(n)])
    heads = [b2.head_index(m) for m in b2.CIRCUIT]
    heads = sorted(heads, key=lambda i: i // 16)
    pairs = b2.ordered_pairs(heads)
    assert len(pairs) == 10
    want = np.mean([b2.overlap(Wb[heads[a]], Kb[heads[b_]], 4) for a, b_ in pairs])
    O = b2.comp_table(Wb, Kb, 4)
    assert np.isclose(O[heads[0], heads[1]], b2.overlap(Wb[heads[0]], Kb[heads[1]], 4))
    assert np.isclose(b2.comp_mean(O, np.array([heads]))[0], want)


def test_plant_keys_puts_the_directions_in_the_top_key_space():
    W = _head(5, d=D, k=16)
    g = np.random.default_rng(5)
    dirs = g.standard_normal((D, 4))
    W2 = b2.plant_keys(W, dirs)
    assert np.allclose(np.linalg.svd(W2["Wk"], compute_uv=False), np.linalg.svd(W["Wk"], compute_uv=False))
    _, Kb = b2.bases(W2, r=8)
    P = np.linalg.qr(dirs)[0]
    assert np.isclose(((Kb[:, :4].T @ P) ** 2).sum(), 4)


def test_dose_reaches_the_target():
    W = _head(6)
    m = b2.m_row(W)
    a0 = b2.v_row(*b2.v_factors(W, "ln"), m)["C1"]
    t, a = b2.dose_to(W, 0.9, m)
    assert t > 0 and a >= 0.9 > a0 and a - 0.9 < 1e-6
    assert b2.dose_to(W, a0 - 0.01, m)[0] == 0.0


def test_r90():
    c = [{"D": 0, "tv": 0.9}, {"D": 1, "tv": 0.5}, {"D": 4, "tv": 0.08}, {"D": 8, "tv": 0.01}]
    assert b2.r90(c) == 4 and b2.r90(c, 0.001) is None


def test_g3m_sees_a_consistent_set_and_dilutes_a_late_one():
    g = np.random.default_rng(0)
    F = b2.fake_sets([5, 7, 12, 8, 11, 10], 20, 1)
    heads = np.array(sorted(set(F.ravel().tolist())))
    parts = {}
    for st in b2.STAT_STEPS:
        c1 = g.random(b2.N_POP)
        hi = np.full(len(heads), 0.80)                       # every planted head at the 80th, every step
        parts[st] = {"C1": c1, "heads": heads, "planted": {0.5: c1[heads], 0.8: hi, 0.9: hi}}
    r = b2.g3m(parts, F, doses=(0.5, 0.8, 0.9))
    assert r["all_q0.80"]["pass_rate"] == 1.0                # 9 steps at the 80th: seen
    assert r["one143000_q0.80"]["pass_rate"] < 0.5           # one step at the 80th: mostly not
    assert r["late_q0.80"]["pass_rate"] < 0.5                # 5 of 9 steps: diluted
    assert r["all_q0.50"]["pass_rate"] <= 0.1                # unplanted values: the null's rate


def test_c5_row_is_classify_ov_row_in_low_rank():
    """Unit 2's C5: the low-rank Table 1 row equals the dense function, including planted rows."""
    from p2d_operator_activation.table1_predictions import classify_ov_row
    for s in range(8):
        W = _head(s)
        Aq, Bk = b2.m_factors(W)
        A, B = b2.v_factors(W, "ln")
        lo, de = b2.c5_row(A, B, Aq, Bk), classify_ov_row(A @ B.T, Aq @ Bk.T)
        assert lo["row"] == de["row"] and np.isclose(lo["lam1_real"], de["lam1_real"])
    g = np.random.default_rng(9)
    U = np.linalg.qr(g.standard_normal((D, K)))[0]
    lam = np.r_[3.0, np.linspace(1, 0.5, K - 1)]
    A, B = U * lam, U                                 # V symmetric PSD, simple top
    for Aq, Bk, want in ((U, U, "row2_three_hyperplanes"), (U, -U, "row2_eigen_only_qk_fails")):
        assert b2.c5_row(A, B, Aq, Bk)["row"] == classify_ov_row(A @ B.T, Aq @ Bk.T)["row"] == want
    I = np.eye(D)
    assert b2.c5_row(I, I, U, U)["row"] == "row1_V_identity"


def test_c5_row_with_no_positive_eigenvalue_is_unclassified():
    """`/challenge-pr` on #221, finding 4: every nonzero eigenvalue with Re < 0 puts a zero on top;
    the low-rank row is "unclassified" (the dense one decides on rounding noise there)."""
    g = np.random.default_rng(11)
    U = np.linalg.qr(g.standard_normal((D, K)))[0]
    A, B = -U * np.linspace(2, 1, K), U                 # V = −U diag Uᵀ, negative definite on U
    r = b2.c5_row(A, B, U, U)
    assert r["row"] == "unclassified" and r["lam1_real"] == 0.0
