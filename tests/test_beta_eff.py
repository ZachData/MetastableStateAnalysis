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

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure

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


# ---------------------------------------------------------------------------
# Per-offset fixed effects (status-1d.md "β refit")
# ---------------------------------------------------------------------------

from core.beta_eff import _two_way_demean, _two_way_demean_exact, estimate_beta_offset_fe  # noqa: E402


def _walk(n=60, d=D, step=0.4, seed=1):
    """Unit rows drifting along the sequence: similarity falls with offset."""
    rng = np.random.default_rng(seed)
    X = [rng.normal(size=d)]
    for _ in range(n - 1):
        X.append(X[-1] / np.linalg.norm(X[-1]) + step * rng.normal(size=d))
    X = np.array(X)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _softmax_with_profile(gram, beta, profile):
    """Causal softmax of ``beta * s_ij + profile(i - j)``."""
    n = gram.shape[0]
    off = np.arange(n)[:, None] - np.arange(n)[None, :]
    S = beta * gram + profile(np.maximum(off, 0))
    S = np.where(off >= 0, S, -np.inf)
    S = S - S.max(axis=1, keepdims=True)
    A = np.exp(S)
    return A / A.sum(axis=1, keepdims=True)


class TestOffsetFixedEffects:

    def test_recovers_beta_under_a_recency_bump(self):
        """
        A recency head whose similarity also falls with offset: the linear
        offset control leaves the bump's curvature on the slope, the
        per-offset dummies absorb it.
        """
        U = _walk()
        G = U @ U.T
        A = _softmax_with_profile(G, BETA_TRUE, lambda d: 4.0 * np.exp(-d / 2.0))
        idx = np.arange(G.shape[0])
        lin = estimate_beta_from_gram(A, G, idx)["beta_raw"]
        fe = estimate_beta_offset_fe(A, G, idx)
        assert abs(fe["beta_raw"] - BETA_TRUE) < 1e-6
        assert fe["partial_r2"] > 0.999
        assert abs(lin - BETA_TRUE) > 0.1          # the bias this fixes

    def test_window_exact_when_profile_is_linear_past_it(self):
        U = _walk(seed=2)
        G = U @ U.T
        W = 8
        prof = lambda d: np.where(d < W, 3.0 * np.exp(-d), -0.05 * d)  # noqa: E731
        A = _softmax_with_profile(G, BETA_TRUE, prof)
        fe = estimate_beta_offset_fe(A, G, np.arange(G.shape[0]), offset_window=W)
        assert abs(fe["beta_raw"] - BETA_TRUE) < 1e-6
        assert fe["n_offset_bins"] == W

    def test_positions_map_a_subset(self):
        """A deduped subset: offsets come from positions, not the subset's index."""
        U = _walk(seed=3)
        G = U @ U.T
        A = _softmax_with_profile(G, BETA_TRUE, lambda d: 2.0 * np.exp(-d / 3.0))
        keep = np.arange(0, G.shape[0], 2)
        sub = np.ix_(keep, keep)
        fe = estimate_beta_offset_fe(A[sub], G[sub], np.arange(keep.size), positions=keep)
        assert abs(fe["beta_raw"] - BETA_TRUE) < 1e-6

    def test_two_way_demean_equals_dummy_ols(self):
        rng = np.random.default_rng(0)
        rows = rng.integers(0, 7, 200)
        bins = rng.integers(0, 5, 200)
        v = rng.normal(size=200)
        D_ = np.column_stack([np.eye(7)[rows], np.eye(5)[bins]])
        coef, *_ = np.linalg.lstsq(D_, v, rcond=None)
        assert np.allclose(_two_way_demean(v, rows, bins, 7, 5, tol=1e-13), v - D_ @ coef,
                           atol=1e-9)
        V = np.column_stack([v, rng.normal(size=200)])
        assert np.allclose(_two_way_demean_exact(V, rows, bins, 7, 5)[:, 0], v - D_ @ coef,
                           atol=1e-9)

    def test_exact_demean_on_a_causal_design(self):
        """Rows see offsets 1..i: the design the iterative version is slow on."""
        n = 30
        i, j = np.tril_indices(n, -1)
        rows, bins = i - 1, (i - j) - 1
        v = np.random.default_rng(1).normal(size=rows.size)
        D_ = np.column_stack([np.eye(n - 1)[rows], np.eye(n - 1)[bins]])
        coef, *_ = np.linalg.lstsq(D_, v, rcond=None)
        assert np.allclose(_two_way_demean_exact(v, rows, bins, n - 1, n - 1), v - D_ @ coef,
                           atol=1e-9)


def test_offset_fe_refuses_similarity_collinear_with_the_tail():
    """offset_window=1 pools every offset; a Gram linear in offset is then the tail."""
    n = 30
    off = np.arange(n)[:, None] - np.arange(n)[None, :]
    G = 1.0 - 0.01 * np.abs(off)
    A = _softmax_with_profile(G, BETA_TRUE, lambda d: -0.1 * d)
    r = estimate_beta_offset_fe(A, G, np.arange(n), offset_window=1)
    assert np.isnan(r["beta_raw"]) and "collinear" in r["note"]


# --- max_offset: fit on near pairs only (#118 review, finding 1) ---

def _causal_head(n=40, d=6, seed=3):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    G = U @ U.T
    off = np.arange(n)[:, None] - np.arange(n)[None, :]
    logits = 2.0 * G - 0.1 * np.abs(off)
    logits = np.where(off >= 0, logits, -np.inf)
    A = np.exp(logits - logits.max(axis=1, keepdims=True))
    return A / A.sum(axis=1, keepdims=True), G, off


def test_max_offset_equals_zeroing_far_pairs():
    from core.beta_eff import estimate_beta_from_gram, estimate_beta_offset_fe
    A, G, off = _causal_head()
    idx = np.arange(1, A.shape[0])
    A_near = np.where(off <= 10, A, 0.0)          # zero attention is dropped by both fits
    for w in (4, None):
        a = estimate_beta_offset_fe(A, G, idx, offset_window=w, max_offset=10)
        b = estimate_beta_offset_fe(A_near, G, idx, offset_window=w)
        assert a["beta_raw"] == pytest.approx(b["beta_raw"], abs=1e-12)
        assert a["n_pairs"] == b["n_pairs"] < estimate_beta_offset_fe(A, G, idx, offset_window=w)["n_pairs"]
    a = estimate_beta_from_gram(A, G, idx, max_offset=10)
    b = estimate_beta_from_gram(A_near, G, idx)
    assert a["beta_raw"] == pytest.approx(b["beta_raw"], abs=1e-12) and a["n_pairs"] == b["n_pairs"]


def test_max_offset_past_the_prompt_changes_nothing():
    from core.beta_eff import estimate_beta_offset_fe
    A, G, _ = _causal_head()
    idx = np.arange(1, A.shape[0])
    assert (estimate_beta_offset_fe(A, G, idx, max_offset=10_000)["beta_raw"]
            == estimate_beta_offset_fe(A, G, idx)["beta_raw"])
