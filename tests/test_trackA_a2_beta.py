"""A2 unit 2's pure parts: the layer βs, the bracket, the rule, the bands, M1 on a planted head
(`trackA_force/design-A2.md` "Unit 2")."""

import numpy as np
import pytest

from trackA_force import a2_beta as ab

pytestmark = pytest.mark.deps  # torch, scipy


def test_layer_betas_all_and_kept():
    hb = np.tile(np.arange(16, dtype=float), (24, 1))
    pr2 = np.zeros((24, 16))
    pr2[:, :5] = 0.2                      # 5 kept heads: β 0..4
    pr2[3, :5] = 0.0
    pr2[3, :3] = 0.2                      # layer 3: 3 kept, below MIN_KEPT
    hb[5, 0] = np.nan
    a = ab.layer_betas(hb, pr2, "all")
    k = ab.layer_betas(hb, pr2, "kept")
    assert a[0] == 7.5 and a[5] == 8.0
    assert k[0] == 2.0 and np.isnan(k[3]) and k[5] == 2.5    # NaN head dropped, 4 kept left


def test_bracket_whole_and_partial():
    lb = np.linspace(0, 23, 24)
    b = ab.bracket_or_none(lb)
    assert b["whole"] and b["n_layers"] == 23
    assert np.isclose(b["p50"], 11.0)     # blocks 0–22 only
    lb[4] = np.nan
    b = ab.bracket_or_none(lb)
    assert not b["whole"] and b["n_layers"] == 22
    assert ab.bracket_or_none(np.full(24, np.nan))["p50"] is None


def test_rule_counts_outside_the_a1_bracket():
    a1 = {"p10": 1.0, "p50": 2.0, "p90": 3.0}
    cells = [{"prompt": str(i), "a1": a1, "refit": {"m1/all": {"p50": v}}}
             for i, v in enumerate([0.5, 3.5, 2.0, 1.0, 3.0, 4.0, 0.9])]
    r = ab.rule(cells)
    assert r["n_outside"] == 4 and r["moves"] and r["outside"] == ["0", "1", "5", "6"]
    cells[6]["refit"]["m1/all"]["p50"] = 2.5
    assert not ab.rule(cells)["moves"]


def test_bands_pool_heads_and_prompts():
    hb = np.zeros((24, 16))
    hb[1:9] = 1.0
    hb[9:17] = 4.0
    hb[17:] = 9.0
    b = ab.band_betas([hb, hb])
    assert b["L1-8"]["p50"] == 1.0 and b["L1-8"]["n"] == 256 and b["L1-8"]["outside_convention"]
    assert b["L9-16"]["p50"] == 4.0 and not b["L9-16"]["outside_convention"]
    assert b["L17-23"]["outside_convention"] and b["L1-23"]["n"] == 23 * 32


def test_m1_recovers_a_planted_beta():
    """Attention drawn from M1 itself at β = 3 and a recency profile: the per-prompt fit returns 3."""
    from trackB_circuit.b2a import m1_attention
    rng = np.random.default_rng(0)
    n, d = 40, 16
    X = rng.standard_normal((n, d))
    ln = {"w": np.ones((1, d)), "b": np.zeros((1, d)), "eps": 1e-5}
    u = ab.rows_of(X[None], np.ones((1, n)), 0, ln)[1:]
    theta = np.zeros(1 + 23)
    theta[0] = 3.0
    theta[1:] = -0.3 * np.arange(23)
    Ah = m1_attention(u, None, theta)
    A = np.zeros((1, 1, n, n))
    A[0, 0, 1:, 1:] = 0.5 * Ah            # mass on position 0 is dropped by the renormalisation
    A[0, 0, :, 0] = 1.0 - A[0, 0].sum(1)
    r = ab.m1_heads(X[None], np.ones((1, n)), A, ln, "cpu")
    assert not r["failed"]
    assert abs(r["beta"][0, 0] - 3.0) < 1e-4     # b2a.RIDGE shrinks it by ~3e-6 at 39 rows


def test_fe_heads_matches_a1a():
    from trackA_force.a1a import head_betas
    rng = np.random.default_rng(1)
    n, d = 30, 8
    acts = rng.standard_normal((2, n, d))
    norms = np.ones((2, n))
    A = rng.random((2, 3, n, n)) * np.tril(np.ones((n, n)))
    A /= A.sum(-1, keepdims=True)
    ln = {"w": np.ones((2, d)), "b": np.zeros((2, d)), "eps": 1e-5}
    got = ab.fe_heads(acts, norms, A, ln)
    assert np.array_equal(got["beta"], head_betas(acts, norms, A, ln), equal_nan=True)
    assert np.all((got["partial_r2"] >= 0) & (got["partial_r2"] <= 1))
