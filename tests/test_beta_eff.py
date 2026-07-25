"""
tests/test_beta_eff.py — oracle tests for beta_eff.py.

The decisive test is `test_recovers_known_beta`: build a causal softmax with
a beta chosen in advance, and check the estimator returns it. The shipping
estimator returns ~0 on the same data, which is the point.
"""

import numpy as np
import pytest

from core.beta_eff import (
    beta_summary_lines,
    causal_pairs,
    estimate_beta_all_heads,
    estimate_beta_from_gram,
    legacy_beta,
    structural_zero_fraction,
)

N = 40
D = 16
BETA_TRUE = 6.0


def _sphere(n=N, d=D, seed=0):
    X = np.random.default_rng(seed).normal(size=(n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _causal_softmax(gram, beta, offset_coeff=0.0):
    """Attention generated from a known beta, with causal masking applied."""
    n = gram.shape[0]
    S = beta * gram
    if offset_coeff:
        d = np.arange(n)[None, :] - np.arange(n)[:, None]
        S = S + offset_coeff * d
    S = np.where(np.tril(np.ones((n, n), bool)), S, -np.inf)
    S = S - S.max(axis=1, keepdims=True)
    A = np.exp(S)
    return A / A.sum(axis=1, keepdims=True)


class TestPairSelection:

    def test_only_causal_pairs(self):
        rows, cols = causal_pairs(np.arange(5))
        assert np.all(cols < rows)

    def test_diagonal_optional(self):
        r, c = causal_pairs(np.arange(4), include_diagonal=True)
        assert np.any(r == c)

    def test_uses_original_positions_not_submatrix_order(self):
        """
        A cluster's indices need not be sorted. Causality is a property of
        the original positions; using submatrix order would silently change
        which pairs count.
        """
        idx = np.array([7, 2, 9])
        rows, cols = causal_pairs(idx)
        for r, c in zip(rows, cols):
            assert idx[c] < idx[r]

    def test_structural_zero_fraction(self):
        A = _causal_softmax(_sphere(6, 4) @ _sphere(6, 4).T, 3.0)
        assert structural_zero_fraction(A) > 0.3


class TestTheBug:

    def test_legacy_estimator_returns_zero_on_causal_attention(self):
        """
        triu_indices(k=1) selects exactly the entries causal masking zeroes.
        The clip turns them all into log(1e-12), so the regression fits a
        varying x against a constant y and the slope is numerically zero —
        regardless of the true beta.
        """
        X = _sphere()
        A = _causal_softmax(X @ X.T, BETA_TRUE)
        got = legacy_beta(A, X, np.arange(N))
        assert abs(got) < 1e-6
        assert abs(got - BETA_TRUE) > 5.0

    def test_legacy_is_insensitive_to_the_true_beta(self):
        """The strongest form: doubling beta does not move the estimate."""
        X = _sphere()
        a = legacy_beta(_causal_softmax(X @ X.T, 3.0), X, np.arange(N))
        b = legacy_beta(_causal_softmax(X @ X.T, 12.0), X, np.arange(N))
        assert abs(a - b) < 1e-6

    def test_upper_triangle_is_structurally_empty(self):
        X = _sphere()
        A = _causal_softmax(X @ X.T, BETA_TRUE)
        assert A[np.triu_indices(N, k=1)].max() == 0.0


class TestRecovery:

    def test_recovers_known_beta(self):
        X = _sphere()
        G = X @ X.T
        out = estimate_beta_from_gram(_causal_softmax(G, BETA_TRUE), G,
                                      np.arange(N))
        assert out["beta"] == pytest.approx(BETA_TRUE, abs=1e-6)
        assert out["r2"] > 0.99

    def test_recovers_across_a_range(self):
        X = _sphere(seed=3)
        G = X @ X.T
        for b in (1.0, 4.0, 15.0):
            out = estimate_beta_from_gram(_causal_softmax(G, b), G, np.arange(N))
            assert out["beta"] == pytest.approx(b, rel=1e-5)

    def test_row_fixed_effects_beat_pooling(self):
        """
        log A_ij = beta * s_ij - log Z_i. The normaliser is per query row and
        an intercept cannot absorb it; pooling biases the slope.
        """
        X = _sphere(seed=5)
        G = X @ X.T
        A = _causal_softmax(G, BETA_TRUE)
        fe = estimate_beta_from_gram(A, G, np.arange(N), row_fixed_effects=True)
        pooled = estimate_beta_from_gram(A, G, np.arange(N), row_fixed_effects=False)
        assert abs(fe["beta"] - BETA_TRUE) < abs(pooled["beta"] - BETA_TRUE)

    def test_offset_covariate_absorbs_positional_structure(self):
        """
        With rotary, offset structure loads onto the slope unless controlled.
        Simulated by adding a term linear in Delta to the logits.
        """
        X = _sphere(seed=7)
        G = X @ X.T
        A = _causal_softmax(G, BETA_TRUE, offset_coeff=0.25)
        controlled = estimate_beta_from_gram(A, G, np.arange(N),
                                             control_offset=True)
        uncontrolled = estimate_beta_from_gram(A, G, np.arange(N),
                                               control_offset=False)
        assert controlled["beta"] == pytest.approx(BETA_TRUE, abs=1e-5)
        assert abs(uncontrolled["beta"] - BETA_TRUE) > abs(
            controlled["beta"] - BETA_TRUE)
        assert controlled["offset_coeff"] == pytest.approx(0.25, abs=1e-5)

    def test_offset_coeff_absent_without_positional_structure(self):
        X = _sphere(seed=11)
        G = X @ X.T
        out = estimate_beta_from_gram(_causal_softmax(G, BETA_TRUE), G,
                                      np.arange(N))
        assert abs(out["offset_coeff"]) < 1e-6


class TestScale:

    def test_scale_divided_out(self):
        X = _sphere()
        G = X @ X.T
        raw = estimate_beta_from_gram(_causal_softmax(G, BETA_TRUE), G,
                                      np.arange(N))
        scaled = estimate_beta_from_gram(_causal_softmax(G, BETA_TRUE), G,
                                         np.arange(N),
                                         attn_scale=1.0 / np.sqrt(128))
        assert scaled["scale_applied"] is True
        assert scaled["beta"] == pytest.approx(raw["beta_raw"] * np.sqrt(128))

    def test_unscaled_flagged_as_not_comparable(self):
        X = _sphere()
        G = X @ X.T
        out = estimate_beta_from_gram(_causal_softmax(G, BETA_TRUE), G,
                                      np.arange(N))
        assert out["scale_applied"] is False
        text = "\n".join(beta_summary_lines({"per_head": [out],
                                             "n_valid_heads": 1,
                                             "cluster_mean_beta": out["beta"],
                                             "cluster_median_beta": out["beta"]}))
        assert "NOT applied" in text


class TestFrameDependence:

    def test_frame_changes_the_answer(self):
        """
        The Gram matrix is an argument precisely because beta is only defined
        relative to a frame. Two frames give two different numbers from the
        same attention.
        """
        X = _sphere()
        G_sphere = X @ X.T
        A = _causal_softmax(G_sphere, BETA_TRUE)
        rng = np.random.default_rng(2)
        Y = X * rng.uniform(0.5, 2.0, size=(N, 1))     # a different frame
        G_other = Y @ Y.T
        a = estimate_beta_from_gram(A, G_sphere, np.arange(N))["beta"]
        b = estimate_beta_from_gram(A, G_other, np.arange(N))["beta"]
        assert not np.isclose(a, b, rtol=1e-3)

    def test_record_states_a_frame_is_required(self):
        X = _sphere()
        G = X @ X.T
        A = _causal_softmax(G, BETA_TRUE)[None, :, :]
        assert estimate_beta_all_heads(A, G, np.arange(N))["frame_required"]


class TestGuards:

    def test_small_cluster(self):
        assert "too small" in estimate_beta_from_gram(
            np.eye(5), np.eye(5), np.arange(2))["note"]

    def test_too_few_causal_pairs(self):
        X = _sphere(6, 4)
        G = X @ X.T
        out = estimate_beta_from_gram(_causal_softmax(G, 3.0), G,
                                      np.array([0, 1, 2]))
        assert np.isnan(out["beta"])
        assert "causal pairs" in out["note"]

    def test_all_zero_attention_reports_why(self):
        A = np.zeros((N, N))
        X = _sphere()
        out = estimate_beta_from_gram(A, X @ X.T, np.arange(N))
        assert np.isnan(out["beta"])
        assert "no mass" in out["note"]

    def test_zero_variance_similarity(self):
        n = 20
        G = np.ones((n, n))
        A = _causal_softmax(G, 1.0)
        out = estimate_beta_from_gram(A, G, np.arange(n))
        assert np.isnan(out["beta"])

    def test_structural_zero_fraction_reported(self):
        X = _sphere()
        G = X @ X.T
        out = estimate_beta_from_gram(_causal_softmax(G, BETA_TRUE), G,
                                      np.arange(N))
        # Roughly half the submatrix is removed by causal masking...
        assert 0.0 < out["structural_zero_fraction"] < 1.0
        # ...and none of it reaches the regression.
        assert out["zero_among_causal_pairs"] == 0.0


class TestAllHeads:

    def test_shape_and_legacy_keys(self):
        X = _sphere()
        G = X @ X.T
        A = np.stack([_causal_softmax(G, b) for b in (2.0, 6.0, 10.0)])
        out = estimate_beta_all_heads(A, G, np.arange(N))
        assert len(out["per_head_beta"]) == 3
        assert out["cluster_mean_beta"] == pytest.approx(6.0, abs=1e-3)
        assert "cluster_median_beta" in out

    def test_invalid_heads_excluded_from_the_mean(self):
        X = _sphere()
        G = X @ X.T
        A = np.stack([_causal_softmax(G, 6.0), np.zeros((N, N))])
        out = estimate_beta_all_heads(A, G, np.arange(N))
        assert out["n_valid_heads"] == 1
        assert out["cluster_mean_beta"] == pytest.approx(6.0, abs=1e-5)
