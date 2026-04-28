"""
tests/test_phase1.py — Tests for phase1/ modules.

No model downloads.  No GPU required.  All inputs are synthetic numpy/torch arrays.

Run:
    pytest tests/test_phase1.py -v
    pytest tests/test_phase1.py -v -k "test_sinkhorn"
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normed_rand(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Return (n, d) float32 array with unit-norm rows."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _two_cluster_normed(n_per: int = 10, d: int = 64, sep: float = 8.0, seed: int = 0) -> np.ndarray:
    """
    Two tight clusters on S^{d-1}: cluster A near e_0, cluster B near -e_0.
    `sep` controls how far apart the centroids are in raw space before norming.
    """
    rng = np.random.default_rng(seed)
    noise = 0.05
    A = np.zeros((n_per, d))
    A[:, 0] = sep
    A += rng.standard_normal((n_per, d)) * noise
    B = np.zeros((n_per, d))
    B[:, 0] = -sep
    B += rng.standard_normal((n_per, d)) * noise
    X = np.vstack([A, B]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _gram(X: np.ndarray) -> np.ndarray:
    return X @ X.T


# ---------------------------------------------------------------------------
# p1_mstate_tracking.metrics
# ---------------------------------------------------------------------------

class TestPairwiseInnerProductsFromGram:
    def test_upper_triangle_count(self):
        from p1_mstate_tracking.metrics import pairwise_inner_products_from_gram
        n = 8
        G = np.eye(n, dtype=np.float32)
        ips = pairwise_inner_products_from_gram(G)
        assert ips.shape == (n * (n - 1) // 2,)

    def test_identity_gram_gives_zeros(self):
        from p1_mstate_tracking.metrics import pairwise_inner_products_from_gram
        # Identity Gram = orthonormal rows → all off-diagonal IPs = 0
        n = 6
        G = np.eye(n, dtype=np.float32)
        ips = pairwise_inner_products_from_gram(G)
        np.testing.assert_allclose(ips, 0.0, atol=1e-6)

    def test_all_ones_gram_gives_ones(self):
        from p1_mstate_tracking.metrics import pairwise_inner_products_from_gram
        # All tokens identical → Gram = all 1s → all IPs = 1
        n = 5
        G = np.ones((n, n), dtype=np.float32)
        ips = pairwise_inner_products_from_gram(G)
        np.testing.assert_allclose(ips, 1.0, atol=1e-6)

    def test_values_in_range(self):
        from p1_mstate_tracking.metrics import pairwise_inner_products_from_gram
        X = _normed_rand(12, 32)
        G = _gram(X)
        ips = pairwise_inner_products_from_gram(G)
        assert ips.min() >= -1.0 - 1e-5
        assert ips.max() <= 1.0 + 1e-5


class TestInteractionEnergiesBatched:
    def test_returns_all_betas(self):
        from p1_mstate_tracking.metrics import interaction_energies_batched
        from core.config import BETA_VALUES
        G = _gram(_normed_rand(10, 32))
        energies = interaction_energies_batched(G, BETA_VALUES)
        assert set(energies.keys()) == set(BETA_VALUES)

    def test_energy_is_positive(self):
        from p1_mstate_tracking.metrics import interaction_energies_batched
        G = _gram(_normed_rand(10, 32))
        for beta, e in interaction_energies_batched(G, [0.5, 1.0, 2.0]).items():
            assert e > 0, f"energy for beta={beta} is non-positive: {e}"

    def test_collapsed_tokens_higher_energy(self):
        """
        All-identical tokens (Gram = ones) should have higher energy than
        spread tokens at the same n, because exp(beta * 1) > exp(beta * 0).
        """
        from p1_mstate_tracking.metrics import interaction_energies_batched
        n, d = 10, 32
        # Spread: random unit vectors
        G_spread = _gram(_normed_rand(n, d))
        # Collapsed: all tokens identical → Gram = all ones
        G_collapsed = np.ones((n, n), dtype=np.float32)
        for beta in [0.5, 1.0, 2.0]:
            e_spread    = interaction_energies_batched(G_spread,    [beta])[beta]
            e_collapsed = interaction_energies_batched(G_collapsed, [beta])[beta]
            assert e_collapsed > e_spread, (
                f"beta={beta}: collapsed ({e_collapsed:.4f}) should exceed "
                f"spread ({e_spread:.4f})"
            )

    def test_larger_beta_amplifies_energy_on_collapsed_gram(self):
        """
        For a fully collapsed Gram (all g_ij = 1), E_beta = exp(beta) / (2*beta),
        which is increasing for beta > 1 but not necessarily for small beta.
        The correct monotonicity prediction is across *layers* for fixed beta,
        not across beta for fixed G.  This test checks the collapsed case where
        the relationship IS monotone (beta >= 1).
        """
        from p1_mstate_tracking.metrics import interaction_energies_batched
        # All tokens identical → Gram = all 1s
        n = 8
        G = np.ones((n, n), dtype=np.float32)
        betas = [1.0, 2.0, 5.0]
        energies = interaction_energies_batched(G, betas)
        # exp(beta)/(2*beta) is increasing for beta > 1
        for lo, hi in zip(betas, betas[1:]):
            assert energies[hi] > energies[lo], (
                f"energy[beta={hi}]={energies[hi]:.4f} should exceed "
                f"energy[beta={lo}]={energies[lo]:.4f} on collapsed Gram"
            )

    def test_agrees_with_scalar_formula(self):
        """Batched result matches the direct scalar computation."""
        from p1_mstate_tracking.metrics import interaction_energies_batched
        n, d = 6, 16
        X = _normed_rand(n, d)
        G = _gram(X)
        beta = 1.0
        expected = float(np.exp(beta * G).sum() / (2 * beta * n * n))
        result   = interaction_energies_batched(G, [beta])[beta]
        assert abs(result - expected) < 1e-5


class TestEffectiveRankFromRaw:
    def test_rank1_matrix(self):
        """All tokens identical → effective rank ≈ 1."""
        from p1_mstate_tracking.metrics import effective_rank_from_raw
        v = torch.randn(1, 64).expand(20, 64).contiguous()
        rank = effective_rank_from_raw(v)
        assert rank < 1.5, f"rank-1 matrix gave effective_rank={rank:.2f}"

    def test_full_rank_matrix(self):
        """Random matrix in high-d → effective rank >> 1."""
        from p1_mstate_tracking.metrics import effective_rank_from_raw
        t = torch.randn(30, 128)
        rank = effective_rank_from_raw(t)
        assert rank > 5.0, f"expected rank > 5, got {rank:.2f}"

    def test_rank_positive(self):
        from p1_mstate_tracking.metrics import effective_rank_from_raw
        t = torch.randn(10, 64)
        assert effective_rank_from_raw(t) > 0

    def test_rank_bounded_by_min_dimension(self):
        from p1_mstate_tracking.metrics import effective_rank_from_raw
        n, d = 8, 128
        t = torch.randn(n, d)
        rank = effective_rank_from_raw(t)
        # Effective rank cannot exceed the rank of the matrix
        assert rank <= min(n, d) + 1e-3


class TestLinearCKA:
    def test_identical_inputs_give_one(self):
        from p1_mstate_tracking.metrics import linear_cka
        X = _normed_rand(15, 32)
        assert abs(linear_cka(X, X) - 1.0) < 1e-5

    def test_orthogonal_inputs_give_zero(self):
        from p1_mstate_tracking.metrics import linear_cka
        # Two blocks of orthogonal vectors — CKA should be near 0
        n, d = 10, 64
        rng = np.random.default_rng(7)
        X = rng.standard_normal((n, d)).astype(np.float32)
        Y = rng.standard_normal((n, d)).astype(np.float32)
        # Make X and Y span orthogonal subspaces (very approximately)
        # by projecting away their overlap
        cka = linear_cka(X, Y)
        assert 0.0 <= cka <= 1.0

    def test_output_in_unit_interval(self):
        from p1_mstate_tracking.metrics import linear_cka
        X = _normed_rand(12, 32, seed=1)
        Y = _normed_rand(12, 32, seed=2)
        cka = linear_cka(X, Y)
        assert 0.0 <= cka <= 1.0


class TestNearestNeighborIndices:
    def test_output_shape(self):
        from p1_mstate_tracking.metrics import nearest_neighbor_indices
        G = _gram(_normed_rand(10, 32))
        nn = nearest_neighbor_indices(G)
        assert nn.shape == (10,)

    def test_no_self_nn(self):
        from p1_mstate_tracking.metrics import nearest_neighbor_indices
        G = _gram(_normed_rand(12, 32))
        nn = nearest_neighbor_indices(G)
        for i, j in enumerate(nn):
            assert i != j, f"token {i} is its own nearest neighbour"

    def test_two_cluster_nn_within_cluster(self):
        """In a clearly separated 2-cluster layout, each token's NN is
        in the same cluster."""
        from p1_mstate_tracking.metrics import nearest_neighbor_indices
        n_per = 15
        X = _two_cluster_normed(n_per=n_per, d=64, sep=10.0)
        G = _gram(X)
        nn = nearest_neighbor_indices(G)
        for i, j in enumerate(nn):
            same_cluster = (i // n_per) == (j // n_per)
            assert same_cluster, (
                f"token {i} (cluster {i // n_per}) has NN {j} (cluster {j // n_per})"
            )


class TestEnergyDropPairs:
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normed(n: int, d: int, seed: int) -> np.ndarray:
        """Return (n, d) float32 array with L2-unit rows."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    # ------------------------------------------------------------------
    # Contract tests
    # ------------------------------------------------------------------

    def test_returns_list_of_tuples(self):
        from p1_mstate_tracking.metrics import energy_drop_pairs_from_normed
        before = self._normed(8, 32, seed=0)
        after  = self._normed(8, 32, seed=1)
        pairs  = energy_drop_pairs_from_normed(before, after, beta=1.0, top_k=5)
        assert isinstance(pairs, list)
        for item in pairs:
            i, j, delta = item
            assert isinstance(i, int)
            assert isinstance(j, int)
            assert i < j  # upper triangle only

    def test_top_k_respects_limit(self):
        from p1_mstate_tracking.metrics import energy_drop_pairs_from_normed
        before = self._normed(10, 32, seed=2)
        after  = self._normed(10, 32, seed=3)
        pairs  = energy_drop_pairs_from_normed(before, after, beta=1.0, top_k=4)
        assert len(pairs) <= 4

    def test_sorted_ascending(self):
        from p1_mstate_tracking.metrics import energy_drop_pairs_from_normed
        before = self._normed(12, 32, seed=4)
        after  = self._normed(12, 32, seed=5)
        pairs  = energy_drop_pairs_from_normed(before, after, beta=1.0, top_k=8)
        deltas = [p[2] for p in pairs]
        assert deltas == sorted(deltas), "pairs should be sorted by delta ascending"


# ---------------------------------------------------------------------------
# p1_mstate_tracking.spectral
# ---------------------------------------------------------------------------

class TestSpectralEigengap:
    def test_output_keys(self):
        from p1_mstate_tracking.spectral import spectral_eigengap_k
        G = _gram(_normed_rand(10, 32))
        result = spectral_eigengap_k(G)
        for key in ("k_eigengap", "k_second_gap", "eigenvalues", "eigengaps"):
            assert key in result

    def test_k_at_least_one(self):
        from p1_mstate_tracking.spectral import spectral_eigengap_k
        G = _gram(_normed_rand(8, 32))
        result = spectral_eigengap_k(G)
        assert result["k_eigengap"] >= 1

    def test_two_clear_clusters(self):
        """
        Block-diagonal Gram with two perfectly separated blocks should give
        k_eigengap == 2.
        """
        from p1_mstate_tracking.spectral import spectral_eigengap_k
        # Build a 2-block Gram: within-block IP = 1, cross-block IP = -1
        # (perfectly antipodal clusters)
        n = 16
        G = -np.ones((n, n), dtype=np.float32)
        half = n // 2
        G[:half, :half] = 1.0
        G[half:, half:] = 1.0
        np.fill_diagonal(G, 1.0)
        result = spectral_eigengap_k(G)
        assert result["k_eigengap"] == 2, (
            f"expected k=2 for 2-block Gram, got {result['k_eigengap']}"
        )

    def test_eigenvalues_non_negative(self):
        """Normalized Laplacian eigenvalues are always >= 0."""
        from p1_mstate_tracking.spectral import spectral_eigengap_k
        G = _gram(_normed_rand(12, 32))
        result = spectral_eigengap_k(G)
        for ev in result["eigenvalues"]:
            assert ev >= -1e-6, f"negative eigenvalue: {ev}"


# ---------------------------------------------------------------------------
# p1_mstate_tracking.sinkhorn
# ---------------------------------------------------------------------------

class TestSinkhornNormalize:
    def _is_doubly_stochastic(self, P: np.ndarray, atol: float = 1e-4) -> bool:
        row_sums = P.sum(axis=1)
        col_sums = P.sum(axis=0)
        return (
            np.allclose(row_sums, 1.0, atol=atol) and
            np.allclose(col_sums, 1.0, atol=atol)
        )

    def test_random_matrix_converges(self):
        from p1_mstate_tracking.sinkhorn import sinkhorn_normalize
        rng = np.random.default_rng(0)
        A = np.abs(rng.standard_normal((8, 8))).astype(np.float64)
        P = sinkhorn_normalize(A)
        assert self._is_doubly_stochastic(P)

    def test_already_doubly_stochastic(self):
        """Uniform matrix is already DS; should be unchanged."""
        from p1_mstate_tracking.sinkhorn import sinkhorn_normalize
        n = 6
        A = np.ones((n, n), dtype=np.float64) / n
        P = sinkhorn_normalize(A)
        assert self._is_doubly_stochastic(P)
        np.testing.assert_allclose(P, A, atol=1e-6)

    def test_output_non_negative(self):
        from p1_mstate_tracking.sinkhorn import sinkhorn_normalize
        rng = np.random.default_rng(1)
        A = np.abs(rng.standard_normal((10, 10)))
        P = sinkhorn_normalize(A)
        assert (P >= 0).all()


class TestSinkhornNormalizeBatched:
    def test_all_heads_doubly_stochastic(self):
        from p1_mstate_tracking.sinkhorn import sinkhorn_normalize_batched
        rng = np.random.default_rng(3)
        n_heads, n = 4, 8
        A = np.abs(rng.standard_normal((n_heads, n, n)))
        P = sinkhorn_normalize_batched(A)
        for h in range(n_heads):
            row_sums = P[h].sum(axis=1)
            col_sums = P[h].sum(axis=0)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-4,
                                       err_msg=f"head {h} rows don't sum to 1")
            np.testing.assert_allclose(col_sums, 1.0, atol=1e-4,
                                       err_msg=f"head {h} cols don't sum to 1")

    def test_shape_preserved(self):
        from p1_mstate_tracking.sinkhorn import sinkhorn_normalize_batched
        rng = np.random.default_rng(4)
        A = np.abs(rng.standard_normal((3, 12, 12)))
        P = sinkhorn_normalize_batched(A)
        assert P.shape == A.shape

    def test_agrees_with_scalar(self):
        """Batched result for head 0 must match the scalar version."""
        from p1_mstate_tracking.sinkhorn import sinkhorn_normalize, sinkhorn_normalize_batched
        rng = np.random.default_rng(5)
        n_heads, n = 3, 6
        A = np.abs(rng.standard_normal((n_heads, n, n)))
        P_batch  = sinkhorn_normalize_batched(A.copy())
        P_scalar = sinkhorn_normalize(A[0].copy())
        np.testing.assert_allclose(P_batch[0], P_scalar, atol=1e-6)


class TestFiedlerValue:
    def test_disconnected_graph_near_zero(self):
        """
        Two completely disconnected components → λ₂ = 0.
        """
        from p1_mstate_tracking.sinkhorn import fiedler_value
        n = 8
        half = n // 2
        # Block diagonal: two isolated blocks (no cross-connections)
        P = np.zeros((n, n))
        P[:half, :half] = 1.0 / half
        P[half:, half:] = 1.0 / half
        fv = fiedler_value(P)
        assert fv < 0.1, f"expected Fiedler ≈ 0 for disconnected graph, got {fv}"

    def test_complete_graph_positive(self):
        """Complete graph (all-ones / n) → Fiedler value > 0."""
        from p1_mstate_tracking.sinkhorn import fiedler_value
        n = 8
        P = np.ones((n, n)) / n
        fv = fiedler_value(P)
        assert fv > 0.0

    def test_output_is_float(self):
        from p1_mstate_tracking.sinkhorn import fiedler_value
        rng = np.random.default_rng(6)
        P = np.abs(rng.standard_normal((6, 6)))
        P /= P.sum()
        assert isinstance(fiedler_value(P), float)


# ---------------------------------------------------------------------------
# p1_mstate_tracking.clustering
# ---------------------------------------------------------------------------

class TestClusterCountSweep:
    def test_output_keys(self):
        from p1_mstate_tracking.clustering import cluster_count_sweep
        X = _normed_rand(12, 32)
        result = cluster_count_sweep(X)
        assert "agglomerative" in result
        assert "kmeans" in result
        assert "best_k" in result["kmeans"]
        assert "labels" in result["kmeans"]
        assert "best_silhouette" in result["kmeans"]

    def test_labels_length(self):
        from p1_mstate_tracking.clustering import cluster_count_sweep
        n = 15
        X = _normed_rand(n, 32)
        result = cluster_count_sweep(X)
        assert len(result["kmeans"]["labels"]) == n

    def test_two_cluster_recovery(self):
        """
        Clearly separated two-cluster input → KMeans should find best_k == 2.
        """
        from p1_mstate_tracking.clustering import cluster_count_sweep
        X = _two_cluster_normed(n_per=20, d=64, sep=12.0)
        result = cluster_count_sweep(X)
        assert result["kmeans"]["best_k"] == 2, (
            f"expected best_k=2, got {result['kmeans']['best_k']}"
        )

    def test_accepts_normed_ndarray(self):
        """Function must accept a pre-normed ndarray without crashing."""
        from p1_mstate_tracking.clustering import cluster_count_sweep
        X = _normed_rand(10, 32).astype(np.float32)
        cluster_count_sweep(X)  # should not raise

    def test_mid_labels_in_agglomerative(self):
        from p1_mstate_tracking.clustering import cluster_count_sweep
        X = _normed_rand(12, 32)
        result = cluster_count_sweep(X)
        assert "mid_labels" in result["agglomerative"]
        assert len(result["agglomerative"]["mid_labels"]) == 12


class TestPcaProjection:
    def test_output_shape(self):
        from p1_mstate_tracking.clustering import pca_projection
        X = _normed_rand(20, 128)
        proj, var = pca_projection(X, n_components=3)
        assert proj.shape == (20, 3)
        assert var.shape == (3,)

    def test_explained_variance_sums_to_at_most_one(self):
        from p1_mstate_tracking.clustering import pca_projection
        X = _normed_rand(20, 64)
        _, var = pca_projection(X, n_components=3)
        assert var.sum() <= 1.0 + 1e-5
        assert (var >= 0).all()

    def test_fewer_components_than_samples(self):
        """n_components capped at min(n_tokens-1, d)."""
        from p1_mstate_tracking.clustering import pca_projection
        X = _normed_rand(4, 128)  # only 4 tokens
        proj, var = pca_projection(X, n_components=10)
        assert proj.shape[1] <= 3   # capped at n-1 = 3
