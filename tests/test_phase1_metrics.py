"""
test_phase1_metrics.py — Pure-computation tests for p1_mstate_tracking metrics.

Modules under test (imported directly from /mnt/project/ via pytest pythonpath):
  metrics.py : pairwise_inner_products_from_gram, interaction_energies_batched,
               effective_rank_from_raw, attention_entropy, linear_cka,
               energy_drop_pairs
  sinkhorn.py: analyze_attention_sinkhorn

No model loading.  All inputs are constructed analytically in conftest.py.
"""

import numpy as np
import pytest
import torch

from metrics import (
    pairwise_inner_products_from_gram,
    interaction_energies_batched,
    effective_rank_from_raw,
    attention_entropy,
    linear_cka,
    energy_drop_pairs,
)
from sinkhorn import analyze_attention_sinkhorn

from tests.constants import N_TOKENS, D


# ============================================================================
# pairwise_inner_products_from_gram
# ============================================================================

class TestPairwiseInnerProducts:
    """Upper-triangle extraction from a pre-computed Gram matrix."""

    def test_returns_upper_triangle_count(self, antipodal_gram):
        """Number of returned values = n*(n-1)/2."""
        ips = pairwise_inner_products_from_gram(antipodal_gram)
        expected_count = N_TOKENS * (N_TOKENS - 1) // 2
        assert ips.shape == (expected_count,), (
            f"Expected {expected_count} pairs, got {ips.shape[0]}"
        )

    def test_antipodal_bimodal_distribution(self, antipodal_gram):
        """
        For tight antipodal clusters the upper-triangle values split into two
        groups: within-cluster ≈ +1 and between-cluster ≈ −1.

        With 20 tokens per cluster:
          within-cluster pairs : 2 × C(20,2) = 380  → ≈ +1
          between-cluster pairs: 20 × 20     = 400  → ≈ −1
        """
        ips = pairwise_inner_products_from_gram(antipodal_gram)
        near_pos1 = (ips > 0.8).sum()
        near_neg1 = (ips < -0.8).sum()
        total     = len(ips)
        assert near_pos1 + near_neg1 == total, (
            "Antipodal pairs should all be near ±1; "
            f"found {total - near_pos1 - near_neg1} values outside (−0.8, 0.8)"
        )

    def test_uniform_inner_products_near_zero(self, uniform_gram):
        """Random unit vectors in R^16 have mean cosine similarity ≈ 0."""
        ips = pairwise_inner_products_from_gram(uniform_gram)
        assert abs(ips.mean()) < 0.15, (
            f"Mean inner product for uniform spread should be near 0, got {ips.mean():.4f}"
        )


# ============================================================================
# interaction_energies_batched
# ============================================================================

class TestInteractionEnergiesBatched:
    """
    E_β = (1 / 2β n²) Σ_{i,j} exp(β ⟨xᵢ, xⱼ⟩)

    Analytical reference values (perfect clusters, noise → 0):
      E_β(collapsed) = exp(β) / (2β)
      E_β(antipodal) = cosh(β) / (2β)
      E_β(uniform)   ≈ 1 / (2β)
    Since exp(β) > cosh(β) > 1, we have collapsed > antipodal > uniform.
    """

    BETAS = [0.1, 1.0, 2.0, 5.0]

    def test_returns_dict_with_all_betas(self, antipodal_gram):
        result = interaction_energies_batched(antipodal_gram, self.BETAS)
        assert isinstance(result, dict)
        for b in self.BETAS:
            assert float(b) in result, f"Missing beta={b} in output dict"

    def test_all_energies_positive(self, antipodal_gram):
        """exp(·) is always positive and β > 0, so E_β > 0."""
        result = interaction_energies_batched(antipodal_gram, self.BETAS)
        for b, e in result.items():
            assert e > 0, f"E_β={b} should be positive, got {e}"

    def test_collapsed_energy_exceeds_antipodal(
        self, collapsed_gram, antipodal_gram
    ):
        """
        E_β(collapsed) = exp(β)/(2β) > cosh(β)/(2β) = E_β(antipodal) for all β>0,
        because exp(β) > cosh(β) = (exp(β)+exp(−β))/2 when β>0.

        This confirms the theory: maximal clustering maximises interaction energy.
        """
        beta = 1.0
        e_collapsed = interaction_energies_batched(collapsed_gram, [beta])[beta]
        e_antipodal = interaction_energies_batched(antipodal_gram, [beta])[beta]
        assert e_collapsed > e_antipodal, (
            f"E_β(collapsed)={e_collapsed:.4f} should exceed "
            f"E_β(antipodal)={e_antipodal:.4f} at β={beta}"
        )

    def test_antipodal_energy_exceeds_uniform(self, antipodal_gram, uniform_gram):
        """
        E_β(antipodal) ≈ cosh(β)/(2β) > 1/(2β) ≈ E_β(uniform) for β>0,
        because cosh(β) > 1.

        Within-cluster exp(+β) dominates between-cluster exp(−β), lifting the
        total above the unclustered baseline.
        """
        beta = 1.0
        e_antipodal = interaction_energies_batched(antipodal_gram, [beta])[beta]
        e_uniform   = interaction_energies_batched(uniform_gram,   [beta])[beta]
        assert e_antipodal > e_uniform, (
            f"E_β(antipodal)={e_antipodal:.4f} should exceed "
            f"E_β(uniform)={e_uniform:.4f} at β={beta}"
        )

    def test_decreasing_in_beta_small_range(self, antipodal_gram):
        """
        For antipodal geometry, E_β = cosh(β)/(2β).
        The derivative d/dβ = [β·sinh(β) − cosh(β)] / (2β²).
        This is negative when β·tanh(β) < 1, which holds for β ∈ (0, ~1.2).

        Using β ∈ {0.1, 0.5, 1.0} — all within the strictly decreasing regime —
        the energy sequence must be strictly monotone decreasing.
        """
        betas  = [0.1, 0.5, 1.0]
        result = interaction_energies_batched(antipodal_gram, betas)
        values = [result[float(b)] for b in betas]
        for i in range(len(values) - 1):
            assert values[i] > values[i + 1], (
                f"Energy not decreasing: E(β={betas[i]})={values[i]:.5f} "
                f"≤ E(β={betas[i+1]})={values[i+1]:.5f}"
            )

    def test_vectorised_matches_scalar(self, antipodal_gram):
        """
        Batched result must match iterating the scalar formula manually.
        E_β = sum(exp(β·G)) / (2β·n²)
        """
        betas  = [0.5, 2.0]
        result = interaction_energies_batched(antipodal_gram, betas)
        n      = antipodal_gram.shape[0]
        for b in betas:
            expected = float(np.exp(b * antipodal_gram).sum() / (2.0 * b * n * n))
            assert result[float(b)] == pytest.approx(expected, rel=1e-6), (
                f"Batched result mismatch at β={b}"
            )


# ============================================================================
# effective_rank_from_raw
# ============================================================================

class TestEffectiveRank:
    """
    Effective rank = exp(H(σ̄)) where σ̄ is the normalised singular-value
    distribution.

    Rank-1 matrix  → single σ > 0 → H = 0 → effective_rank = 1.0
    All-equal SVs  → H = log(d)   → effective_rank = d
    """

    def test_rank1_matrix_returns_one(self, rank1_tensor):
        """
        All rows of rank1_tensor are identical (outer product v⊗w).
        Only one singular value is non-zero → effective rank = 1.
        """
        rank = effective_rank_from_raw(rank1_tensor)
        assert rank == pytest.approx(1.0, abs=1e-4), (
            f"Rank-1 matrix should have effective rank 1.0, got {rank:.6f}"
        )

    def test_uniform_sv_returns_d(self, uniform_sv_tensor):
        """
        All 16 singular values equal 1 → uniform distribution over D components
        → entropy = log(D) → effective rank = exp(log(D)) = D.
        """
        rank = effective_rank_from_raw(uniform_sv_tensor)
        assert rank == pytest.approx(D, abs=0.5), (
            f"Uniform-SV tensor should have effective rank ≈ {D}, got {rank:.4f}"
        )

    def test_returns_float(self, uniform_sv_tensor):
        assert isinstance(effective_rank_from_raw(uniform_sv_tensor), float)

    def test_rank_bounded_between_1_and_d(self, uniform_sv_tensor, rank1_tensor):
        """Effective rank is always in [1, d] for any non-zero matrix."""
        r_low  = effective_rank_from_raw(rank1_tensor)
        r_high = effective_rank_from_raw(uniform_sv_tensor)
        assert 1.0 <= r_low  <= D
        assert 1.0 <= r_high <= D


# ============================================================================
# attention_entropy
# ============================================================================

class TestAttentionEntropy:
    """
    Shannon entropy of each attention row, averaged over tokens.

    Uniform attention (1/n for all entries):
      H = −Σ (1/n) log(1/n) = log(n)  — maximum entropy.

    Identity attention (each token attends only to itself):
      H = −1·log(1) = 0  — minimum entropy.
    """

    def test_uniform_entropy_is_log_n(self, uniform_attention):
        """Mean entropy over all heads and tokens = log(N_TOKENS)."""
        ent      = attention_entropy(uniform_attention)   # (n_heads,)
        expected = np.log(N_TOKENS)
        np.testing.assert_allclose(
            ent, expected, atol=1e-5,
            err_msg=f"Uniform attention entropy should be log({N_TOKENS})≈{expected:.4f}"
        )

    def test_identity_entropy_is_zero(self, identity_attention):
        """Each token attends exclusively to itself → H = 0 per row."""
        ent = attention_entropy(identity_attention)
        np.testing.assert_allclose(
            ent, 0.0, atol=1e-5,
            err_msg="Identity attention entropy should be 0"
        )

    def test_uniform_entropy_exceeds_identity(
        self, uniform_attention, identity_attention
    ):
        """Uniform > identity for all heads."""
        ent_uniform   = attention_entropy(uniform_attention)
        ent_identity  = attention_entropy(identity_attention)
        assert (ent_uniform > ent_identity).all()

    def test_output_shape_is_n_heads(self, uniform_attention):
        n_heads = uniform_attention.shape[0]
        ent = attention_entropy(uniform_attention)
        assert ent.shape == (n_heads,)

    def test_entropy_non_negative(self, uniform_attention):
        assert (attention_entropy(uniform_attention) >= 0.0).all()


# ============================================================================
# linear_cka
# ============================================================================

class TestLinearCKA:
    """
    Linear CKA(X, Y) = ‖Y^T X‖_F² / (‖X^T X‖_F · ‖Y^T Y‖_F).

    Identical representations at consecutive layers → CKA = 1.0.
    Orthogonal representations → CKA = 0.0.
    """

    def test_identical_activations_give_cka_one(self, uniform_normed):
        cka = linear_cka(uniform_normed, uniform_normed)
        assert cka == pytest.approx(1.0, abs=1e-6), (
            f"CKA(X, X) must equal 1.0, got {cka:.8f}"
        )

    def test_identical_antipodal_activations_give_cka_one(self, antipodal_normed):
        cka = linear_cka(antipodal_normed, antipodal_normed)
        assert cka == pytest.approx(1.0, abs=1e-6)

    def test_output_in_unit_interval(self, uniform_normed, antipodal_normed):
        """CKA is clipped to [0, 1] by the implementation."""
        cka = linear_cka(uniform_normed, antipodal_normed)
        assert 0.0 <= cka <= 1.0

    def test_returns_float(self, uniform_normed):
        assert isinstance(linear_cka(uniform_normed, uniform_normed), float)

    def test_symmetry(self, uniform_normed, antipodal_normed):
        """CKA is symmetric: CKA(X, Y) == CKA(Y, X)."""
        cka_xy = linear_cka(uniform_normed,  antipodal_normed)
        cka_yx = linear_cka(antipodal_normed, uniform_normed)
        assert cka_xy == pytest.approx(cka_yx, abs=1e-6)


# ============================================================================
# energy_drop_pairs
# ============================================================================

class TestEnergyDropPairs:
    """
    energy_drop_pairs identifies (i, j, delta) pairs where delta =
    [exp(β⟨xᵢ,xⱼ⟩_after) − exp(β⟨xᵢ,xⱼ⟩_before)] / (2β n²).

    If before == after, every delta must be 0.
    """

    def _make_raw_tensor(self):
        """Random (N_TOKENS, D) raw activation tensor (not normalised)."""
        return torch.randn(N_TOKENS, D, generator=torch.Generator().manual_seed(7))

    def test_identical_layers_all_deltas_zero(self):
        X = self._make_raw_tensor()
        pairs = energy_drop_pairs(X, X, beta=1.0, top_k=10)
        for i, j, delta in pairs:
            assert abs(delta) < 1e-6, (
                f"Pair ({i},{j}) has non-zero delta={delta:.2e} for identical layers"
            )

    def test_returns_list_of_triples(self):
        X = self._make_raw_tensor()
        pairs = energy_drop_pairs(X, X, beta=1.0, top_k=5)
        assert isinstance(pairs, list)
        for item in pairs:
            assert len(item) == 3, "Each entry should be (i, j, delta)"

    def test_top_k_respected(self):
        """At most top_k pairs are returned."""
        X     = self._make_raw_tensor()
        Y     = torch.randn_like(X)
        pairs = energy_drop_pairs(X, Y, beta=1.0, top_k=7)
        assert len(pairs) <= 7

    def test_pairs_sorted_ascending_delta(self):
        """Pairs are ordered by delta ascending (most negative first)."""
        X     = self._make_raw_tensor()
        Y     = torch.randn_like(X)
        pairs = energy_drop_pairs(X, Y, beta=1.0, top_k=10)
        deltas = [d for _, _, d in pairs]
        assert deltas == sorted(deltas), (
            "energy_drop_pairs should be sorted by delta ascending"
        )

    def test_indices_are_valid_upper_triangle(self):
        """All returned pairs satisfy 0 ≤ i < j < n_tokens."""
        X     = self._make_raw_tensor()
        Y     = torch.randn_like(X)
        pairs = energy_drop_pairs(X, Y, beta=1.0, top_k=15)
        for i, j, _ in pairs:
            assert 0 <= i < j < N_TOKENS, (
                f"Invalid pair indices ({i}, {j}); expected 0 ≤ i < j < {N_TOKENS}"
            )


# ============================================================================
# analyze_attention_sinkhorn — attention-tensor tests
# ============================================================================

class TestAnalyzeAttentionSinkhorn:
    """
    analyze_attention_sinkhorn performs per-head Sinkhorn normalisation and
    returns Fiedler values and cluster counts.

    Doubly stochastic input (uniform_attention) should yield
    row_col_balance_mean ≈ 0 (column sums already all equal 1).
    """

    EXPECTED_KEYS = {
        "fiedler_mean",
        "fiedler_per_head",
        "sinkhorn_cluster_count_mean",
        "sinkhorn_cluster_counts",
        "row_col_balance_mean",
    }

    def test_returns_all_expected_keys(self, uniform_attention):
        result = analyze_attention_sinkhorn(uniform_attention)
        assert self.EXPECTED_KEYS.issubset(result.keys()), (
            f"Missing keys: {self.EXPECTED_KEYS - set(result.keys())}"
        )

    def test_row_col_balance_near_zero_for_doubly_stochastic(
        self, uniform_attention
    ):
        """
        Uniform attention has all column sums = 1 → std(col_sums) = 0.
        row_col_balance measures deviation from doubly stochastic form.
        """
        result = analyze_attention_sinkhorn(uniform_attention)
        assert result["row_col_balance_mean"] == pytest.approx(0.0, abs=1e-6), (
            "Uniform (doubly stochastic) input should have row_col_balance=0"
        )

    def test_per_head_list_length_matches_n_heads(self, uniform_attention):
        n_heads = uniform_attention.shape[0]
        result  = analyze_attention_sinkhorn(uniform_attention)
        assert len(result["fiedler_per_head"])         == n_heads
        assert len(result["sinkhorn_cluster_counts"]) == n_heads

    def test_fiedler_mean_is_float(self, uniform_attention):
        result = analyze_attention_sinkhorn(uniform_attention)
        assert isinstance(result["fiedler_mean"], float)

    def test_identity_has_lower_fiedler_than_uniform(
        self, uniform_attention, identity_attention
    ):
        """
        Identity attention → each token is isolated → near-disconnected graph →
        small Fiedler value (λ₂ ≈ 0).
        Uniform attention → fully connected → large Fiedler value.
        """
        result_uniform   = analyze_attention_sinkhorn(uniform_attention)
        result_identity  = analyze_attention_sinkhorn(identity_attention)
        assert result_identity["fiedler_mean"] < result_uniform["fiedler_mean"], (
            "Identity attention should have lower Fiedler value than uniform"
        )
