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

from p1_mstate_tracking.metrics import (
    pairwise_inner_products_from_gram,
    interaction_energies_batched,
    effective_rank_from_raw,
    attention_entropy,
    linear_cka,
    energy_drop_pairs,
)
from p1_mstate_tracking.sinkhorn import analyze_attention_sinkhorn

from tests.config import N_TOKENS, D

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
    energy_drop_pairs(before, after, beta, top_k) → [(i, j, delta), ...]

    delta_ij = [exp(β⟨xᵢ,xⱼ⟩_after) − exp(β⟨xᵢ,xⱼ⟩_before)] / (2β n²)

    Contracts
    ---------
    - before == after  →  all deltas == 0
    - results sorted ascending (most negative first)
    - len(result) <= top_k
    - all (i, j) satisfy i < j  (upper triangle only)
    - n < 2  →  empty list

    Correctness
    -----------
    - known-geometry: pair that transitions from ⟨·⟩≈1 to ⟨·⟩≈0 is the
      most negative entry
    - energy-increasing transition (orthogonal → collapsed): no negative deltas
    - higher β amplifies the delta magnitude for a fixed geometry change
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raw(arr: np.ndarray) -> torch.Tensor:
        """Wrap a float32 ndarray as a torch Tensor."""
        return torch.tensor(arr.astype(np.float32))

    @staticmethod
    def _spread(n: int, seed: int = 0) -> np.ndarray:
        """n unit vectors spread roughly uniformly in R^D."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, D)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    @staticmethod
    def _with_pair_merged(n: int, i: int, j: int, seed: int = 0) -> np.ndarray:
        """
        Spread activations where tokens i and j are identical (⟨i,j⟩=1).
        All other pairs are approximately orthogonal.
        """
        rng = np.random.default_rng(seed)
        # Start from spread, then force i and j to be parallel
        X = rng.standard_normal((n, D)).astype(np.float32) * 0.05
        pole = np.zeros(D, dtype=np.float32)
        pole[0] = 1.0
        X[i] = pole
        X[j] = pole
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    # ------------------------------------------------------------------
    # Contract tests
    # ------------------------------------------------------------------

    def test_identical_inputs_all_deltas_zero(self):
        """
        before == after  →  G_before == G_after  →  every delta == 0.
        The function should return an empty list or a list of near-zero deltas.
        """
        X = self._spread(N_TOKENS, seed=1)
        t = self._raw(X)
        pairs = energy_drop_pairs(t, t, beta=1.0, top_k=N_TOKENS * N_TOKENS)
        for _, _, delta in pairs:
            assert abs(delta) < 1e-6, (
                f"Identical inputs: expected delta=0, got {delta:.2e}"
            )

    def test_upper_triangle_only(self):
        """Every returned pair satisfies i < j."""
        rng = np.random.default_rng(2)
        t1 = self._raw(rng.standard_normal((N_TOKENS, D)).astype(np.float32))
        t2 = self._raw(rng.standard_normal((N_TOKENS, D)).astype(np.float32))
        pairs = energy_drop_pairs(t1, t2, beta=1.0, top_k=50)
        for i, j, _ in pairs:
            assert i < j, f"Pair ({i}, {j}) violates upper-triangle constraint"

    def test_top_k_bounds_output_length(self):
        """len(result) <= top_k for all k."""
        rng = np.random.default_rng(3)
        t1 = self._raw(rng.standard_normal((N_TOKENS, D)).astype(np.float32))
        t2 = self._raw(rng.standard_normal((N_TOKENS, D)).astype(np.float32))
        for k in [1, 3, 10, 30]:
            assert len(energy_drop_pairs(t1, t2, beta=1.0, top_k=k)) <= k, (
                f"top_k={k} not respected"
            )

    def test_sorted_ascending(self):
        """Results are ordered by delta ascending (most negative first)."""
        rng = np.random.default_rng(4)
        t1 = self._raw(rng.standard_normal((N_TOKENS, D)).astype(np.float32))
        t2 = self._raw(rng.standard_normal((N_TOKENS, D)).astype(np.float32))
        pairs = energy_drop_pairs(t1, t2, beta=1.0, top_k=20)
        deltas = [d for _, _, d in pairs]
        assert deltas == sorted(deltas), (
            f"Pairs not sorted ascending: {deltas}"
        )

    def test_single_token_returns_empty(self):
        """n=1 → no pairs possible → empty list."""
        t = torch.randn(1, D)
        pairs = energy_drop_pairs(t, t, beta=1.0, top_k=5)
        assert pairs == [], f"Single token should yield [], got {pairs}"

    # ------------------------------------------------------------------
    # Correctness tests
    # ------------------------------------------------------------------

    def test_known_worst_pair_is_top_result(self):
        """
        Geometry
        --------
        before: tokens 0 and 1 are parallel (⟨0,1⟩ ≈ 1); all others spread.
        after:  all tokens spread uniformly (⟨0,1⟩ ≈ 0).

        delta_{01} = [exp(β·0) − exp(β·1)] / (2β n²)
                   = (1 − eᵝ) / (2β n²)  < 0   for all β > 0.

        All other pairs had ⟨·⟩_before ≈ 0 and ⟨·⟩_after ≈ 0, so their
        deltas are near zero.  Pair (0, 1) must be the most negative entry.
        """
        n = 8
        before = self._with_pair_merged(n, i=0, j=1, seed=7)
        after  = self._spread(n, seed=8)

        pairs = energy_drop_pairs(self._raw(before), self._raw(after),
                                  beta=1.0, top_k=n)

        assert len(pairs) > 0, "Expected at least one pair"
        worst_i, worst_j, worst_delta = pairs[0]
        assert worst_delta < 0, (
            f"Top pair delta should be negative, got {worst_delta:.4f}"
        )
        assert (worst_i, worst_j) == (0, 1), (
            f"Pair (0,1) should be worst; got ({worst_i},{worst_j}) "
            f"with delta={worst_delta:.4f}"
        )

    def test_energy_increasing_transition_no_negative_deltas(self):
        """
        Transition from spread (low energy) to collapsed (high energy) raises
        every pairwise exp-term.  No pair should have a negative delta.

        before: tokens spread uniformly → ⟨·⟩ ≈ 0
        after:  all tokens near +e₀ → ⟨·⟩ ≈ 1

        delta_ij = [exp(β·1) − exp(β·0)] / norm  > 0  for all (i,j).
        """
        n = 8
        spread    = self._spread(n, seed=10)
        collapsed = self._spread(n, seed=11) * 0.01  # tight cluster
        collapsed[:, 0] += 1.0                        # all near +e₀

        pairs = energy_drop_pairs(self._raw(spread), self._raw(collapsed),
                                  beta=1.0, top_k=n * n)
        for i, j, delta in pairs:
            assert delta >= -1e-6, (
                f"Spread→collapsed should have no negative deltas; "
                f"pair ({i},{j}) delta={delta:.2e}"
            )

    def test_delta_agrees_with_manual_formula(self):
        """
        For n=4 tokens where we control the Gram matrices exactly,
        verify that the returned delta matches the formula:
          delta_ij = [exp(β G_after[i,j]) − exp(β G_before[i,j])] / (2β n²)
        for the pair with the largest absolute change.
        """
        beta = 2.0
        n    = 4

        # before: tokens 0 and 1 are identical; 2 and 3 are orthogonal to everything
        before = np.zeros((n, D), dtype=np.float32)
        before[0, 0] = 1.0
        before[1, 0] = 1.0   # parallel to token 0
        before[2, 1] = 1.0
        before[3, 2] = 1.0

        # after: all tokens orthogonal to each other
        after = np.zeros((n, D), dtype=np.float32)
        for k in range(n):
            after[k, k] = 1.0   # standard basis vectors

        norm = 2.0 * beta * n * n
        # G_before[0,1] = ⟨e₀, e₀⟩ = 1.0
        # G_after[0,1]  = ⟨e₀, e₁⟩ = 0.0
        expected_delta_01 = (np.exp(beta * 0.0) - np.exp(beta * 1.0)) / norm

        pairs = energy_drop_pairs(self._raw(before), self._raw(after),
                                  beta=beta, top_k=n)
        # Pair (0, 1) must be present and match the formula
        matching = [(i, j, d) for i, j, d in pairs if (i, j) == (0, 1)]
        assert len(matching) == 1, "Pair (0,1) not found in results"
        _, _, actual_delta = matching[0]
        assert actual_delta == pytest.approx(expected_delta_01, rel=1e-4), (
            f"Expected delta={expected_delta_01:.6f}, got {actual_delta:.6f}"
        )

    def test_higher_beta_amplifies_negative_delta(self):
        """
        Larger β → larger exp contrast → more negative delta for a converging
        pair that subsequently diverges.

        For pair (0,1): before ⟨·⟩ = 1, after ⟨·⟩ = 0.
          delta(β) = (1 − eᵝ) / (2β n²)

        d/dβ [1 − eᵝ] = −eᵝ < 0, so delta becomes more negative as β grows.
        """
        n = 6
        before = self._with_pair_merged(n, i=0, j=1, seed=12)
        after  = self._spread(n, seed=13)

        b_lo, b_hi = self._raw(before), self._raw(after)
        delta_lo = energy_drop_pairs(b_lo, b_hi, beta=0.5, top_k=1)
        delta_hi = energy_drop_pairs(b_lo, b_hi, beta=4.0, top_k=1)

        assert len(delta_lo) == 1 and len(delta_hi) == 1
        assert delta_hi[0][2] < delta_lo[0][2], (
            f"Higher β should produce more negative delta: "
            f"β=0.5 → {delta_lo[0][2]:.4f}, β=4.0 → {delta_hi[0][2]:.4f}"
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
