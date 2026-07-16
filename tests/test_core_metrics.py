"""
tests/test_core_metrics.py — Tests for core/metrics.py (core foundations
item 3: the single canonical metrics module).

Self-contained: fixtures are built locally rather than pulled from the
existing tests/config.py + conftest.py fixture set, so this file exercises
core.metrics in isolation from the rest of the (torch-dependent) test
session and can run wherever numpy/scipy are available, torch or not.

Covers:
  - Consolidation correctness: effective_rank(mode=...) matches the two old
    named functions it replaces; fiedler_and_eigengap matches the numbers
    spectral_eigengap_k used to produce inline.
  - The mass_near_1 duplicate-definition bug: analysis.py's inline
    `(ips > 0.9).mean()` and causal_tests.py's `_mass_near_1` (threshold
    0.95, mask-restricted) are both instances of the one function here.
  - Basic correctness of energy, Fiedler, and mass_near_1 on analytically
    known geometries (antipodal, uniform, collapsed clusters).
"""

import numpy as np
import pytest

from core.metrics import (
    l2_normalize,
    gram_matrix,
    pairwise_upper,
    pairwise_inner_products_from_gram,
    interaction_energy,
    interaction_energies_batched,
    energy_violation_severity,
    effective_rank,
    effective_rank_from_raw,
    effective_rank_from_normed,
    fiedler_and_eigengap,
    mass_near_1,
    MASS_NEAR_1_DEFAULT_THRESHOLD,
    attention_entropy,
    nearest_neighbor_indices,
    nearest_neighbor_stability,
    linear_cka,
    energy_drop_pairs_from_normed,
)

N_TOKENS = 40
D = 16
_rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def antipodal_normed():
    half = N_TOKENS // 2
    X = np.zeros((N_TOKENS, D), dtype=np.float64)
    X[:half, 0] = 1.0
    X[half:, 0] = -1.0
    noise = _rng.standard_normal((N_TOKENS, D)) * 0.05
    return l2_normalize(X + noise)


@pytest.fixture(scope="module")
def uniform_normed():
    X = _rng.standard_normal((N_TOKENS, D))
    return l2_normalize(X)


@pytest.fixture(scope="module")
def collapsed_normed():
    X = np.zeros((N_TOKENS, D), dtype=np.float64)
    X[:, 0] = 1.0
    noise = _rng.standard_normal((N_TOKENS, D)) * 0.05
    return l2_normalize(X + noise)


@pytest.fixture(scope="module")
def antipodal_gram(antipodal_normed):
    return antipodal_normed @ antipodal_normed.T


@pytest.fixture(scope="module")
def uniform_gram(uniform_normed):
    return uniform_normed @ uniform_normed.T


@pytest.fixture(scope="module")
def collapsed_gram(collapsed_normed):
    return collapsed_normed @ collapsed_normed.T


# ---------------------------------------------------------------------------
# Gram / pairwise
# ---------------------------------------------------------------------------

class TestGramAndPairwise:
    def test_pairwise_upper_count(self, antipodal_gram):
        ips = pairwise_upper(antipodal_gram)
        assert ips.shape == (N_TOKENS * (N_TOKENS - 1) // 2,)

    def test_backward_compat_alias_is_same_function(self):
        assert pairwise_inner_products_from_gram is pairwise_upper

    def test_antipodal_bimodal(self, antipodal_gram):
        ips = pairwise_upper(antipodal_gram)
        near_pos1 = (ips > 0.8).sum()
        near_neg1 = (ips < -0.8).sum()
        assert near_pos1 + near_neg1 == len(ips)

    def test_uniform_near_zero_mean(self, uniform_gram):
        ips = pairwise_upper(uniform_gram)
        assert abs(ips.mean()) < 0.15


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

class TestEnergy:
    BETAS = [0.1, 1.0, 2.0, 5.0]

    def test_all_betas_present_and_positive(self, antipodal_gram):
        result = interaction_energies_batched(antipodal_gram, self.BETAS)
        for b in self.BETAS:
            assert float(b) in result
            assert result[float(b)] > 0

    def test_collapsed_exceeds_antipodal_exceeds_uniform(
        self, collapsed_gram, antipodal_gram, uniform_gram
    ):
        beta = 1.0
        e_collapsed = interaction_energies_batched(collapsed_gram, [beta])[beta]
        e_antipodal = interaction_energies_batched(antipodal_gram, [beta])[beta]
        e_uniform = interaction_energies_batched(uniform_gram, [beta])[beta]
        assert e_collapsed > e_antipodal > e_uniform

    def test_batched_matches_scalar_interaction_energy(self, antipodal_normed, antipodal_gram):
        beta = 2.0
        batched = interaction_energies_batched(antipodal_gram, [beta])[beta]
        scalar = interaction_energy(antipodal_normed, beta)
        assert batched == pytest.approx(scalar, rel=1e-6)

    def test_violation_severity_detects_monotone_drop(self):
        energies = [1.0, 0.99, 0.5, 0.51]  # big relative drop at index 2
        out = energy_violation_severity(energies)
        assert 2 in out["violation_layers"]
        assert out["n_violations"] >= 1

    def test_violation_severity_monotone_series_has_no_violations(self):
        energies = [1.0, 1.1, 1.2, 1.3]
        out = energy_violation_severity(energies)
        assert out["n_violations"] == 0


# ---------------------------------------------------------------------------
# Effective rank — consolidation correctness
# ---------------------------------------------------------------------------

class TestEffectiveRank:
    def test_rank1_matrix_is_one(self):
        v = np.zeros(D)
        v[0] = 1.0
        X = np.tile(v, (N_TOKENS, 1))
        assert effective_rank(X, mode="raw") == pytest.approx(1.0, abs=1e-6)

    def test_orthonormal_columns_gives_full_rank(self):
        X = _rng.standard_normal((N_TOKENS, D))
        Q, _ = np.linalg.qr(X)
        assert effective_rank(Q, mode="normed") == pytest.approx(D, rel=0.05)

    def test_raw_alias_matches_mode_raw(self, antipodal_normed):
        assert effective_rank_from_raw(antipodal_normed) == effective_rank(antipodal_normed, mode="raw")

    def test_normed_alias_matches_mode_normed(self, antipodal_normed):
        assert effective_rank_from_normed(antipodal_normed) == effective_rank(antipodal_normed, mode="normed")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            effective_rank(np.zeros((3, 3)), mode="bogus")

    def test_raw_and_normed_differ_when_scale_varies(self):
        # Raw rank sees scale variation across tokens; normed rank doesn't.
        X = _rng.standard_normal((N_TOKENS, D))
        X[: N_TOKENS // 2] *= 100.0  # blow up half the tokens' norms
        raw = effective_rank(X, mode="raw")
        normed = effective_rank(X, mode="normed")
        # Not asserting a specific direction, just that consolidation didn't
        # silently make the two modes identical (that would mean one path
        # is dead code, defeating the point of keeping both names).
        assert raw != pytest.approx(normed, rel=1e-3)


# ---------------------------------------------------------------------------
# Fiedler / eigengap
# ---------------------------------------------------------------------------

class TestFiedlerAndEigengap:
    def test_two_clusters_detected(self, antipodal_gram):
        result = fiedler_and_eigengap(antipodal_gram, max_k=10)
        assert result["k_eigengap"] == 2

    def test_fiedler_value_present_and_finite(self, antipodal_gram):
        result = fiedler_and_eigengap(antipodal_gram, max_k=10)
        assert np.isfinite(result["fiedler_value"])

    def test_fiedler_vec_only_when_requested(self, antipodal_gram):
        without = fiedler_and_eigengap(antipodal_gram, max_k=10, return_fiedler_vec=False)
        assert "fiedler_vec" not in without
        with_vec = fiedler_and_eigengap(antipodal_gram, max_k=10, return_fiedler_vec=True)
        assert "fiedler_vec" in with_vec
        assert len(with_vec["fiedler_vec"]) == N_TOKENS

    def test_fiedler_vec_sign_partitions_the_two_clusters(self, antipodal_gram):
        result = fiedler_and_eigengap(antipodal_gram, max_k=10, return_fiedler_vec=True)
        vec = np.array(result["fiedler_vec"])
        half = N_TOKENS // 2
        # Every token in the first half should share sign, every token in
        # the second half the opposite sign (up to the vector's overall sign).
        signs = np.sign(vec)
        first_half_consistent = len(set(signs[:half])) == 1
        second_half_consistent = len(set(signs[half:])) == 1
        assert first_half_consistent and second_half_consistent
        assert signs[0] != signs[-1]

    def test_tiny_input_degenerate_case(self):
        result = fiedler_and_eigengap(np.array([[1.0]]), max_k=10)
        assert result["k_eigengap"] == 1
        assert np.isnan(result["fiedler_value"])

    def test_eigenvalues_non_negative(self, antipodal_gram, uniform_gram, collapsed_gram):
        for G in (antipodal_gram, uniform_gram, collapsed_gram):
            result = fiedler_and_eigengap(G, max_k=10)
            assert all(ev >= -1e-8 for ev in result["eigenvalues"])


# ---------------------------------------------------------------------------
# mass_near_1 — the duplicate-definition bug this module fixes
# ---------------------------------------------------------------------------

class TestMassNearOne:
    def test_default_threshold_is_0_9(self):
        assert MASS_NEAR_1_DEFAULT_THRESHOLD == 0.9

    def test_collapsed_cluster_is_near_1(self, collapsed_gram):
        assert mass_near_1(collapsed_gram) > 0.95

    def test_uniform_spread_is_near_0(self, uniform_gram):
        assert mass_near_1(uniform_gram) < 0.05

    def test_matches_old_analysis_py_inline_formula(self, antipodal_gram):
        # analysis.py computed: (pairwise_inner_products_from_gram(G) > 0.9).mean()
        expected = float((pairwise_upper(antipodal_gram) > 0.9).mean())
        assert mass_near_1(antipodal_gram, threshold=0.9) == pytest.approx(expected)

    def test_mask_restricts_to_cluster_matches_old_causal_tests_formula(self, antipodal_gram):
        # causal_tests.py's _mass_near_1(X, mask, thresh=0.95) restricted to
        # a boolean mask before computing the same fraction.
        mask = np.zeros(N_TOKENS, dtype=bool)
        mask[: N_TOKENS // 2] = True  # first cluster only

        sub = antipodal_gram[np.ix_(mask, mask)]
        expected = float((pairwise_upper(sub) > 0.95).mean())

        assert mass_near_1(antipodal_gram, threshold=0.95, mask=mask) == pytest.approx(expected)

    def test_mask_with_fewer_than_two_tokens_returns_zero(self, antipodal_gram):
        mask = np.zeros(N_TOKENS, dtype=bool)
        mask[0] = True
        assert mass_near_1(antipodal_gram, mask=mask) == 0.0

    def test_single_token_gram_returns_zero(self):
        assert mass_near_1(np.array([[1.0]])) == 0.0


# ---------------------------------------------------------------------------
# Remaining single-implementation metrics — smoke coverage
# ---------------------------------------------------------------------------

class TestRemainingMetrics:
    def test_attention_entropy_uniform_is_log_n(self):
        n_heads, n_tok = 3, 8
        attn = np.full((n_heads, n_tok, n_tok), 1.0 / n_tok)
        ent = attention_entropy(attn)
        assert ent.shape == (n_heads,)
        np.testing.assert_allclose(ent, np.log(n_tok), atol=1e-6)

    def test_attention_entropy_identity_is_zero(self):
        n_heads, n_tok = 2, 5
        eye = np.eye(n_tok)
        attn = np.stack([eye] * n_heads)
        ent = attention_entropy(attn)
        np.testing.assert_allclose(ent, 0.0, atol=1e-6)

    def test_nearest_neighbor_indices_excludes_self(self, antipodal_gram):
        nn = nearest_neighbor_indices(antipodal_gram)
        assert np.all(nn != np.arange(N_TOKENS))

    def test_nearest_neighbor_stability_identical_layers_is_one(self, antipodal_normed):
        stability = nearest_neighbor_stability(antipodal_normed, antipodal_normed)
        assert stability == pytest.approx(1.0)

    def test_linear_cka_identical_is_one(self, antipodal_normed):
        assert linear_cka(antipodal_normed, antipodal_normed) == pytest.approx(1.0, abs=1e-6)

    def test_linear_cka_bounded(self, antipodal_normed, uniform_normed):
        val = linear_cka(antipodal_normed, uniform_normed)
        assert 0.0 <= val <= 1.0 or np.isnan(val)

    def test_energy_drop_pairs_returns_top_k(self, antipodal_normed, uniform_normed):
        pairs = energy_drop_pairs_from_normed(antipodal_normed, uniform_normed, beta=1.0, top_k=5)
        assert len(pairs) == 5
        deltas = [d for (_, _, d) in pairs]
        assert deltas == sorted(deltas)  # most negative first
