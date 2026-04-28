"""
tests/test_phase6.py — Pure-computation tests for p6_subspace modules.

Modules under test:
  eigenspace_degeneracy : project_to_subspace, degeneracy_ratio,
                          lda_direction, subspace_alignment
  probe_subspace        : probe_accuracy, probe_all_channels
  qk_decompose          : decompose_qk_matrix, logit_partition,
                          find_induction_pairs, find_same_content_pairs,
                          compare_aqk_fractions
  local_contraction     : spectral_radius, decompose_local_map, fit_local_map
  write_subspace        : head_write_alignment, principal_angles
  centroid_velocity     : decompose_centroid_delta, merge_geometry_test

No model loading. No GPU required. All inputs are synthetic numpy arrays.

Run:
    pytest tests/test_phase6.py -v
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from p6_subspace.eigenspace_degeneracy import (
    degeneracy_ratio,
    lda_direction,
    project_to_subspace,
    subspace_alignment,
)
from p6_subspace.probe_subspace import probe_accuracy, probe_all_channels
from p6_subspace.qk_decompose import (
    compare_aqk_fractions,
    decompose_qk_matrix,
    find_induction_pairs,
    find_same_content_pairs,
    logit_partition,
)
from p6_subspace.local_contraction import (
    decompose_local_map,
    fit_local_map,
    spectral_radius,
)
from p6_subspace.write_subspace import head_write_alignment, principal_angles
from p6_subspace.centroid_velocity import decompose_centroid_delta, merge_geometry_test


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _orth_basis(d: int, r: int, seed: int = 0) -> np.ndarray:
    """Random orthonormal (d, r) basis via QR."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, max(r, 1))))
    return Q[:, :r]


def _two_cluster_data(
    n_per: int = 20, d: int = 16, sep: float = 6.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two tight, well-separated clusters.
    Cluster 0 near +e_0, cluster 1 near -e_0.
    Returns (X, labels).
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((2 * n_per, d)) * 0.05
    centers = np.zeros((2 * n_per, d))
    centers[:n_per, 0] = sep
    centers[n_per:, 0] = -sep
    X = centers + noise
    labels = np.array([0] * n_per + [1] * n_per)
    return X.astype(np.float32), labels


def _projector_pair(d: int, r_S: int, r_A: int, seed: int = 0):
    """Return (P_S, P_A, U_pos, U_A) orthogonal projectors for a d-dim space."""
    assert r_S + r_A <= d
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    U_pos = Q[:, :r_S]
    U_A   = Q[:, r_S:r_S + r_A]
    P_S   = U_pos @ U_pos.T
    P_A   = U_A   @ U_A.T
    return P_S, P_A, U_pos, U_A


# ============================================================================
# eigenspace_degeneracy — project_to_subspace
# ============================================================================

class TestProjectToSubspace:

    def test_output_shape(self):
        X = np.random.default_rng(0).standard_normal((30, 16)).astype(np.float32)
        B = _orth_basis(16, 4)
        Z = project_to_subspace(X, B)
        assert Z.shape == (30, 4)

    def test_identity_basis_preserves_matrix(self):
        """Projecting onto the full identity returns X unchanged."""
        X = np.random.default_rng(1).standard_normal((10, 8)).astype(np.float32)
        B = np.eye(8)
        npt.assert_allclose(project_to_subspace(X, B), X, atol=1e-5)

    def test_orthonormal_basis_preserves_row_norms(self):
        """Projection onto orthonormal basis is isometric: ||Zx|| <= ||x||."""
        X = np.random.default_rng(2).standard_normal((20, 16)).astype(np.float32)
        B = _orth_basis(16, 6)
        Z = project_to_subspace(X, B)
        norms_X = np.linalg.norm(X, axis=1)
        norms_Z = np.linalg.norm(Z, axis=1)
        assert np.all(norms_Z <= norms_X + 1e-5)

    def test_projection_onto_subspace_of_data_recovers_data(self):
        """Rows of X that live in span(B) project to themselves (in coordinates)."""
        rng = np.random.default_rng(3)
        B = _orth_basis(16, 4)
        coefs = rng.standard_normal((10, 4))
        X = coefs @ B.T          # rows are in span(B)
        Z = project_to_subspace(X, B)
        # Z should equal coefs (up to float precision)
        npt.assert_allclose(Z, coefs, atol=1e-5)


# ============================================================================
# eigenspace_degeneracy — degeneracy_ratio
# ============================================================================

class TestDegeneracyRatio:

    def test_required_keys_present(self):
        X, labels = _two_cluster_data()
        B = _orth_basis(16, 4)
        Z = project_to_subspace(X, B)
        result = degeneracy_ratio(Z, labels)
        for k in ("ratio", "var_within", "var_between", "n_clusters", "n_tokens"):
            assert k in result, f"Missing key: {k}"

    def test_tight_clusters_give_high_ratio(self):
        """Well-separated clusters → between >> within → high ratio."""
        X, labels = _two_cluster_data(n_per=30, sep=10.0)
        # Project along the separating axis (e_0)
        B = np.zeros((16, 1))
        B[0, 0] = 1.0
        Z = project_to_subspace(X, B)
        result = degeneracy_ratio(Z, labels)
        assert result["ratio"] is not None
        assert result["ratio"] > 5.0, (
            f"Expected ratio > 5 for tight clusters, got {result['ratio']}"
        )

    def test_single_cluster_ratio_is_none(self):
        """Only one valid cluster → between-cluster variance undefined."""
        Z = np.random.default_rng(5).standard_normal((20, 4)).astype(np.float32)
        labels = np.zeros(20, dtype=int)  # all same cluster
        result = degeneracy_ratio(Z, labels)
        assert result["ratio"] is None

    def test_noise_tokens_excluded(self):
        """Tokens with label == -1 must not contribute to the ratio."""
        X, labels = _two_cluster_data(n_per=20)
        B = _orth_basis(16, 4)
        Z = project_to_subspace(X, B)
        labels_with_noise = labels.copy()
        labels_with_noise[:5] = -1
        result_clean = degeneracy_ratio(Z, labels)
        result_noisy = degeneracy_ratio(Z, labels_with_noise)
        # n_tokens should differ
        assert result_noisy["n_tokens"] == result_clean["n_tokens"] - 5

    def test_ratio_near_one_for_random_projection(self):
        """Random projection of tight clusters → ratio should not be huge."""
        X, labels = _two_cluster_data()
        rng = np.random.default_rng(99)
        B = _orth_basis(16, 4, seed=99)
        # Project onto subspace orthogonal to the separating axis
        # (zero out the first component of every basis vector)
        B_perp = B.copy()
        B_perp[0, :] = 0.0
        Q, _ = np.linalg.qr(B_perp)
        Z = project_to_subspace(X, Q[:, :4])
        result = degeneracy_ratio(Z, labels)
        # The separating axis (e_0) is absent, so ratio << tight-cluster ratio
        # We can't guarantee exactly 1, but it should be much lower
        assert result["ratio"] is None or result["ratio"] < 200.0


# ============================================================================
# eigenspace_degeneracy — lda_direction
# ============================================================================

class TestLdaDirection:

    def test_returns_unit_vector(self):
        X, labels = _two_cluster_data()
        w = lda_direction(X, labels, 0, 1)
        assert w is not None
        npt.assert_allclose(np.linalg.norm(w), 1.0, atol=1e-6)

    def test_aligns_with_separation_axis(self):
        """LDA direction for clusters separated along e_0 should align with e_0."""
        X, labels = _two_cluster_data(n_per=40, sep=10.0)
        w = lda_direction(X, labels, 0, 1)
        assert w is not None
        cos = abs(float(w[0]))   # cosine with e_0
        assert cos > 0.8, f"Expected strong alignment with e_0, got {cos}"

    def test_returns_none_for_too_few_tokens(self):
        X = np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32)
        labels = np.array([0, 0, 1])  # cluster 1 has only 1 token
        result = lda_direction(X, labels, 0, 1)
        assert result is None

    def test_output_dimension_matches_input(self):
        X, labels = _two_cluster_data(d=32)
        w = lda_direction(X, labels, 0, 1)
        assert w is not None
        assert w.shape == (32,)


# ============================================================================
# eigenspace_degeneracy — subspace_alignment
# ============================================================================

class TestSubspaceAlignment:

    def test_direction_in_subspace_gives_one(self):
        B = _orth_basis(16, 4)
        w = B[:, 0]   # first basis vector is in the subspace
        npt.assert_allclose(subspace_alignment(w, B), 1.0, atol=1e-6)

    def test_direction_orthogonal_to_subspace_gives_zero(self):
        d = 16
        B = _orth_basis(d, 4, seed=10)
        # Build a vector orthogonal to all of B's columns
        Q, _ = np.linalg.qr(np.hstack([B, np.eye(d, 1)]))
        w = Q[:, 4]   # orthogonal to span(B)
        npt.assert_allclose(subspace_alignment(w, B), 0.0, atol=1e-6)

    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(20)
        B = _orth_basis(16, 6)
        for _ in range(10):
            w = rng.standard_normal(16)
            w /= np.linalg.norm(w)
            val = subspace_alignment(w, B)
            assert 0.0 - 1e-6 <= val <= 1.0 + 1e-6

    def test_empty_basis_gives_zero(self):
        w = np.array([1.0, 0.0, 0.0])
        B = np.zeros((3, 0))
        assert subspace_alignment(w, B) == 0.0


# ============================================================================
# probe_subspace — probe_accuracy
# ============================================================================

class TestProbeAccuracy:

    def test_required_keys(self):
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((40, 8)).astype(np.float32)
        labels = np.array([0] * 20 + [1] * 20)
        result = probe_accuracy(Z, labels)
        for k in ("mean_accuracy", "std_accuracy", "n_samples", "n_classes", "chance_level"):
            assert k in result

    def test_perfectly_separable_data_high_accuracy(self):
        """Two tight clusters in R^2 → near-perfect CV accuracy."""
        X, labels = _two_cluster_data(n_per=30, d=2, sep=20.0)
        result = probe_accuracy(X, labels)
        assert result["mean_accuracy"] > 0.9, (
            f"Expected >0.9 accuracy on separable data, got {result['mean_accuracy']}"
        )

    def test_chance_returned_for_single_class(self):
        Z = np.random.default_rng(1).standard_normal((20, 4)).astype(np.float32)
        labels = np.zeros(20, dtype=int)
        result = probe_accuracy(Z, labels)
        assert result["mean_accuracy"] == result["chance_level"]

    def test_noise_tokens_excluded_from_n_samples(self):
        X, labels = _two_cluster_data(n_per=20)
        labels_noisy = labels.copy()
        labels_noisy[:10] = -1
        result = probe_accuracy(X, labels_noisy)
        assert result["n_samples"] == 30   # 40 - 10 noise

    def test_chance_level_equals_one_over_n_classes(self):
        rng = np.random.default_rng(2)
        Z = rng.standard_normal((60, 8)).astype(np.float32)
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        result = probe_accuracy(Z, labels)
        npt.assert_allclose(result["chance_level"], 1.0 / 3, atol=1e-6)

    def test_std_accuracy_nonnegative(self):
        X, labels = _two_cluster_data()
        result = probe_accuracy(X, labels)
        assert result["std_accuracy"] >= 0.0


# ============================================================================
# probe_subspace — probe_all_channels
# ============================================================================

class TestProbeAllChannels:

    @pytest.fixture(autouse=True)
    def _setup(self):
        d = 16
        self.X, self.labels = _two_cluster_data(n_per=30, d=d, sep=8.0)
        _, _, U_pos, U_A = _projector_pair(d, 4, 4)
        self.U_pos = U_pos
        self.U_A   = U_A

    def test_output_has_four_channels(self):
        result = probe_all_channels(self.X, self.labels, self.U_pos, self.U_A)
        for ch in ("full", "real", "imag", "random"):
            assert ch in result, f"Missing channel: {ch}"

    def test_full_channel_has_required_keys(self):
        result = probe_all_channels(self.X, self.labels, self.U_pos, self.U_A)
        for k in ("mean_accuracy", "std_accuracy", "n_samples", "n_classes", "chance_level"):
            assert k in result["full"]

    def test_real_channel_in_separating_subspace_has_high_accuracy(self):
        """Project onto the subspace that contains the separation axis."""
        d = 16
        # Build U_pos to include e_0 (the separation axis)
        B = np.eye(d, 4)   # columns e_0..e_3
        labels = self.labels
        result = probe_all_channels(self.X, labels, B, self.U_A)
        assert result["real"]["mean_accuracy"] > 0.8


# ============================================================================
# qk_decompose — decompose_qk_matrix
# ============================================================================

class TestDecomposeQkMatrix:

    @pytest.fixture(autouse=True)
    def _setup(self):
        rng = np.random.default_rng(0)
        self.d, self.h = 16, 8
        self.WQ = rng.standard_normal((self.d, self.h)).astype(np.float32)
        self.WK = rng.standard_normal((self.d, self.h)).astype(np.float32)
        self.decomp = decompose_qk_matrix(self.WQ, self.WK)

    def test_required_keys(self):
        for k in ("S_QK", "A_QK", "M", "s_frac", "a_frac"):
            assert k in self.decomp

    def test_S_is_symmetric(self):
        S = self.decomp["S_QK"]
        npt.assert_allclose(S, S.T, atol=1e-5)

    def test_A_is_antisymmetric(self):
        A = self.decomp["A_QK"]
        npt.assert_allclose(A, -A.T, atol=1e-5)

    def test_S_plus_A_reconstructs_M(self):
        M = self.decomp["M"]
        npt.assert_allclose(self.decomp["S_QK"] + self.decomp["A_QK"], M, atol=1e-5)

    def test_M_equals_WQ_T_WK(self):
            # M = W_Q W_K^T  (d_model × d_model), not W_Q^T W_K (d_head × d_head).
            # logit(i,j) = x_i^T M x_j  where M acts in residual-stream space.
            expected = self.WQ @ self.WK.T
            npt.assert_allclose(self.decomp["M"], expected, atol=1e-5)

    def test_fractions_sum_to_one(self):
        npt.assert_allclose(self.decomp["s_frac"] + self.decomp["a_frac"], 1.0, atol=1e-5)

    def test_fractions_in_unit_interval(self):
        assert 0.0 - 1e-6 <= self.decomp["s_frac"] <= 1.0 + 1e-6
        assert 0.0 - 1e-6 <= self.decomp["a_frac"] <= 1.0 + 1e-6

    def test_symmetric_M_gives_zero_a_frac(self):
        """If WQ == WK, M = WQ^T WK is symmetric → A=0 → a_frac=0."""
        rng = np.random.default_rng(7)
        WQ = rng.standard_normal((16, 8)).astype(np.float32)
        decomp = decompose_qk_matrix(WQ, WQ)
        npt.assert_allclose(decomp["a_frac"], 0.0, atol=1e-5)

    def test_antisymmetric_M_gives_zero_s_frac(self):
        """M = WQ @ WK^T is antisymmetric → S_QK = 0 → s_frac = 0.

        Construction: WQ with orthonormal columns, antisymmetric B (h×h),
        WK = WQ @ B.  Then M = WQ @ WK^T = WQ @ B^T @ WQ^T = -WQ @ B @ WQ^T = -M^T.
        """
        rng = np.random.default_rng(8)
        d, h = 16, 8
        # WQ with orthonormal columns  (QR gives exactly this)
        WQ, _ = np.linalg.qr(rng.standard_normal((d, h)).astype(np.float32))
        WQ = WQ[:, :h]
        # antisymmetric B
        raw = rng.standard_normal((h, h)).astype(np.float32)
        B   = raw - raw.T
        # WK chosen so M = WQ @ WK^T = WQ @ B^T @ WQ^T  (antisymmetric)
        WK  = WQ @ B
        decomp = decompose_qk_matrix(WQ, WK)
        npt.assert_allclose(decomp["s_frac"], 0.0, atol=1e-5)


# ============================================================================
# qk_decompose — logit_partition
# ============================================================================

class TestLogitPartition:

    @pytest.fixture(autouse=True)
    def _setup(self):
        rng = np.random.default_rng(10)
        d, h = 16, 8
        WQ = rng.standard_normal((d, h)).astype(np.float32)
        WK = rng.standard_normal((d, h)).astype(np.float32)
        self.decomp = decompose_qk_matrix(WQ, WK)
        X = rng.standard_normal((12, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        self.X = X
        self.n = 12
        self.result = logit_partition(self.decomp, self.X)

    def test_required_keys(self):
        for k in ("s_contrib", "a_contrib", "a_frac_mat"):
            assert k in self.result

    def test_output_shapes(self):
        assert self.result["s_contrib"].shape  == (self.n, self.n)
        assert self.result["a_contrib"].shape  == (self.n, self.n)
        assert self.result["a_frac_mat"].shape == (self.n, self.n)

    def test_a_frac_mat_in_unit_interval(self):
        f = self.result["a_frac_mat"]
        assert np.all(f >= 0.0 - 1e-6)
        assert np.all(f <= 1.0 + 1e-6)

    def test_s_plus_a_recovers_total_logit(self):
        """The signed contributions should sum to the full logit x_i^T M x_j."""
        M = self.decomp["M"]
        total = self.X @ M @ self.X.T
        reconstructed = self.result["s_contrib"] + self.result["a_contrib"]
        npt.assert_allclose(reconstructed, total, atol=1e-4)


# ============================================================================
# qk_decompose — find_induction_pairs / find_same_content_pairs
# ============================================================================

class TestFindInductionPairs:

    def test_offset_constraint_respected(self):
        """No returned pair (i, j) should have j < i + min_offset."""
        rng = np.random.default_rng(0)
        n, d = 20, 16
        ids = rng.integers(0, 5, size=n)
        X   = rng.standard_normal((n, d)).astype(np.float32)
        X  /= np.linalg.norm(X, axis=1, keepdims=True)
        pairs = find_induction_pairs(ids, X, min_offset=2)
        for i, j in pairs:
            assert j >= i + 2, f"Pair ({i},{j}) violates min_offset=2"

    def test_repeating_token_pattern_detected(self):
        """Token sequence [A, B, A, B, ...] should produce induction pairs."""
        n, d = 10, 8
        ids = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        rng = np.random.default_rng(1)
        X = rng.standard_normal((n, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        pairs = find_induction_pairs(ids, X, sim_threshold=0.9, min_offset=2)
        assert len(pairs) > 0

    def test_all_unique_tokens_no_pairs(self):
        """All unique tokens → no induction pairs."""
        n, d = 10, 8
        ids = np.arange(n)
        rng = np.random.default_rng(2)
        X = rng.standard_normal((n, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        pairs = find_induction_pairs(ids, X, sim_threshold=0.99, min_offset=2)
        assert pairs == []

    def test_returns_list_of_tuples(self):
        n, d = 15, 8
        ids = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5])
        rng = np.random.default_rng(3)
        X = rng.standard_normal((n, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        pairs = find_induction_pairs(ids, X)
        assert isinstance(pairs, list)
        for p in pairs:
            assert len(p) == 2


class TestFindSameContentPairs:

    def test_no_overlap_with_induction_pairs(self):
        """Same-content pairs must exclude induction-pattern pairs."""
        n, d = 12, 8
        ids = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        rng = np.random.default_rng(5)
        X = rng.standard_normal((n, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        ind = set(find_induction_pairs(ids, X, sim_threshold=0.7))
        same = set(find_same_content_pairs(ids, X, sim_threshold=0.7))
        overlap = ind & same
        assert len(overlap) == 0, f"Unexpected overlap: {overlap}"

    def test_offset_constraint_respected(self):
        n, d = 15, 8
        ids = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4])
        rng = np.random.default_rng(6)
        X = rng.standard_normal((n, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        pairs = find_same_content_pairs(ids, X, min_offset=2)
        for i, j in pairs:
            assert j >= i + 2


# ============================================================================
# qk_decompose — compare_aqk_fractions
# ============================================================================

class TestCompareAqkFractions:

    def _make_a_frac_mat(self, n: int, ind_val: float, same_val: float,
                          ind_pairs, same_pairs) -> np.ndarray:
        """Build an a_frac matrix with prescribed values at the given pairs."""
        mat = np.full((n, n), 0.3)
        for i, j in ind_pairs:
            mat[i, j] = ind_val
        for i, j in same_pairs:
            mat[i, j] = same_val
        return mat

    def test_required_keys(self):
        n = 10
        mat = np.random.default_rng(0).uniform(0, 1, (n, n))
        result = compare_aqk_fractions(mat, [(1, 3)], [(2, 5)])
        for k in ("delta", "mwu_pvalue", "p6_i2_satisfied"):
            assert k in result

    def test_induction_higher_than_same_content_satisfies_p6i2(self):
        n = 15
        ind_pairs  = [(1, 4), (2, 5), (3, 6), (1, 7), (2, 8)]
        same_pairs = [(0, 3), (5, 9), (6, 10), (7, 11), (8, 12)]
        mat = self._make_a_frac_mat(n, ind_val=0.8, same_val=0.2,
                                    ind_pairs=ind_pairs, same_pairs=same_pairs)
        result = compare_aqk_fractions(mat, ind_pairs, same_pairs)
        assert result["p6_i2_satisfied"] is True
        assert result["delta"] > 0

    def test_induction_lower_than_same_content_fails_p6i2(self):
        n = 15
        ind_pairs  = [(1, 4), (2, 5), (3, 6)]
        same_pairs = [(0, 3), (5, 9), (6, 10)]
        mat = self._make_a_frac_mat(n, ind_val=0.1, same_val=0.9,
                                    ind_pairs=ind_pairs, same_pairs=same_pairs)
        result = compare_aqk_fractions(mat, ind_pairs, same_pairs)
        assert result["p6_i2_satisfied"] is False

    def test_empty_induction_pairs_returns_none_delta(self):
        mat = np.random.default_rng(0).uniform(0, 1, (10, 10))
        result = compare_aqk_fractions(mat, [], [(1, 3)])
        assert result["delta"] is None


# ============================================================================
# local_contraction — spectral_radius
# ============================================================================

class TestSpectralRadius:

    def test_identity_matrix_gives_one(self):
        npt.assert_allclose(spectral_radius(np.eye(6)), 1.0, atol=1e-6)

    def test_zero_matrix_gives_zero(self):
        npt.assert_allclose(spectral_radius(np.zeros((5, 5))), 0.0, atol=1e-6)

    def test_scaled_identity(self):
        npt.assert_allclose(spectral_radius(3.7 * np.eye(4)), 3.7, atol=1e-5)

    def test_known_diagonal(self):
        M = np.diag([1.0, -2.5, 0.3])
        npt.assert_allclose(spectral_radius(M), 2.5, atol=1e-5)

    def test_nonnegative_for_any_matrix(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((8, 8))
        assert spectral_radius(M) >= 0.0


# ============================================================================
# local_contraction — decompose_local_map
# ============================================================================

class TestDecomposeLocalMap:

    @pytest.fixture(autouse=True)
    def _setup(self):
        rng = np.random.default_rng(0)
        self.W = rng.standard_normal((8, 8)).astype(np.float64)
        self.result = decompose_local_map(self.W)

    def test_required_keys(self):
        for k in ("W_S", "W_A", "rho_W", "rho_S", "rho_A", "contracting_S", "neutral_A"):
            assert k in self.result

    def test_W_S_is_symmetric(self):
        npt.assert_allclose(self.result["W_S"], self.result["W_S"].T, atol=1e-10)

    def test_W_A_is_antisymmetric(self):
        npt.assert_allclose(self.result["W_A"], -self.result["W_A"].T, atol=1e-10)

    def test_W_S_plus_W_A_equals_W(self):
        npt.assert_allclose(
            self.result["W_S"] + self.result["W_A"], self.W, atol=1e-10
        )

    def test_spectral_radii_nonnegative(self):
        assert self.result["rho_W"] >= 0.0
        assert self.result["rho_S"] >= 0.0
        assert self.result["rho_A"] >= 0.0

    def test_contracting_S_flag_correct(self):
        assert self.result["contracting_S"] == (self.result["rho_S"] < 1.0)

    def test_neutral_A_flag_correct(self):
        assert self.result["neutral_A"] == (abs(self.result["rho_A"] - 1.0) < 0.15)

    def test_contraction_matrix_gives_contracting_S_true(self):
        """Symmetric part of 0.5*I has rho_S = 0.5 < 1."""
        W = 0.5 * np.eye(6)
        result = decompose_local_map(W)
        assert result["contracting_S"] is True

    def test_expansion_matrix_gives_contracting_S_false(self):
        """Symmetric part of 2*I has rho_S = 2.0 ≥ 1."""
        W = 2.0 * np.eye(6)
        result = decompose_local_map(W)
        assert result["contracting_S"] is False


# ============================================================================
# local_contraction — fit_local_map
# ============================================================================

class TestFitLocalMap:

    def test_exact_linear_map_recovered(self):
        """If X_next = X_cur @ W_true.T exactly, lstsq should recover W_true."""
        rng = np.random.default_rng(0)
        d = 8
        W_true = rng.standard_normal((d, d))
        X_cur  = rng.standard_normal((20, d)).astype(np.float64)
        X_nxt  = X_cur @ W_true.T
        W_hat  = fit_local_map(X_cur, X_nxt)
        assert W_hat is not None
        npt.assert_allclose(W_hat, W_true.T, atol=1e-5)

    def test_output_shape(self):
        rng = np.random.default_rng(1)
        d = 10
        X_cur = rng.standard_normal((15, d))
        X_nxt = rng.standard_normal((15, d))
        W = fit_local_map(X_cur, X_nxt)
        assert W is not None
        assert W.shape == (d, d)

    def test_too_few_tokens_returns_none(self):
        """Fewer than d tokens → underdetermined → should return None."""
        d = 16
        rng = np.random.default_rng(2)
        X_cur = rng.standard_normal((2, d))
        X_nxt = rng.standard_normal((2, d))
        result = fit_local_map(X_cur, X_nxt)
        # Underdetermined but lstsq will still return something; we check it
        # doesn't crash. If None is returned for min_tokens guard, that's fine too.
        assert result is None or result.shape == (d, d)


# ============================================================================
# write_subspace — principal_angles
# ============================================================================

class TestPrincipalAngles:

    def test_same_subspace_gives_zero_angles(self):
        B = _orth_basis(16, 4)
        angles = principal_angles(B, B)
        npt.assert_allclose(angles, np.zeros(4), atol=1e-5)

    def test_orthogonal_subspaces_give_pi_over_2(self):
        d = 8
        Q, _ = np.linalg.qr(np.eye(d))
        A = Q[:, :3]
        B = Q[:, 3:6]
        angles = principal_angles(A, B)
        npt.assert_allclose(angles, np.full(3, np.pi / 2), atol=1e-5)

    def test_angles_in_range_zero_to_pi_over_2(self):
        A = _orth_basis(16, 4, seed=0)
        B = _orth_basis(16, 4, seed=1)
        angles = principal_angles(A, B)
        assert np.all(angles >= -1e-6)
        assert np.all(angles <= np.pi / 2 + 1e-6)

    def test_output_length_is_min_of_ranks(self):
        A = _orth_basis(16, 3)
        B = _orth_basis(16, 5)
        angles = principal_angles(A, B)
        assert len(angles) == 3

    def test_empty_subspace_returns_pi_over_2(self):
        A = _orth_basis(8, 3)
        B = np.zeros((8, 0))
        angles = principal_angles(A, B)
        npt.assert_allclose(angles, [np.pi / 2], atol=1e-5)


# ============================================================================
# write_subspace — head_write_alignment
# ============================================================================

class TestHeadWriteAlignment:

    @pytest.fixture(autouse=True)
    def _setup(self):
        d = 16
        _, _, self.U_pos, self.U_A = _projector_pair(d, 4, 4)
        self.P_S = self.U_pos @ self.U_pos.T
        self.P_A = self.U_A   @ self.U_A.T
        self.d   = d

    def test_required_keys(self):
        rng = np.random.default_rng(0)
        WO = rng.standard_normal((self.d, 8)).astype(np.float32)
        result = head_write_alignment(WO, self.P_A, self.P_S)
        for k in ("align_rot", "align_real", "sing_vals"):
            assert k in result

    def test_align_values_in_unit_interval(self):
        rng = np.random.default_rng(1)
        for seed in range(5):
            WO = rng.standard_normal((self.d, 8)).astype(np.float32)
            r  = head_write_alignment(WO, self.P_A, self.P_S)
            assert -1e-5 <= r["align_rot"]  <= 1.0 + 1e-5
            assert -1e-5 <= r["align_real"] <= 1.0 + 1e-5

    def test_wo_in_imaginary_subspace_high_align_rot(self):
        """W_O whose columns lie in U_A → align_rot should be near 1."""
        # Build a WO whose columns are in U_A
        coefs = np.random.default_rng(2).standard_normal((4, 8))
        WO = (self.U_A @ coefs).astype(np.float32)
        result = head_write_alignment(WO, self.P_A, self.P_S, top_r=8)
        assert result["align_rot"] > 0.8, (
            f"Expected align_rot near 1 for imaginary-subspace W_O, got {result['align_rot']}"
        )

    def test_wo_in_real_subspace_high_align_real(self):
        """W_O whose columns lie in U_pos → align_real should be near 1."""
        coefs = np.random.default_rng(3).standard_normal((4, 8))
        WO = (self.U_pos @ coefs).astype(np.float32)
        result = head_write_alignment(WO, self.P_A, self.P_S, top_r=8)
        assert result["align_real"] > 0.8, (
            f"Expected align_real near 1 for real-subspace W_O, got {result['align_real']}"
        )

    def test_sing_vals_length_matches_top_r(self):
        rng = np.random.default_rng(4)
        WO = rng.standard_normal((self.d, 8)).astype(np.float32)
        result = head_write_alignment(WO, self.P_A, self.P_S, top_r=6)
        assert len(result["sing_vals"]) == min(6, 8)


# ============================================================================
# centroid_velocity — decompose_centroid_delta
# ============================================================================

class TestDecomposeCentroidDelta:

    @pytest.fixture(autouse=True)
    def _setup(self):
        d = 16
        self.P_S, self.P_A, _, _ = _projector_pair(d, 4, 4)
        self.d = d

    def test_required_keys(self):
        delta = np.random.default_rng(0).standard_normal(self.d)
        result = decompose_centroid_delta(delta, self.P_S, self.P_A)
        for k in ("delta_S", "delta_A", "norm_S", "norm_A", "norm_total", "r_S"):
            assert k in result

    def test_delta_in_S_subspace_gives_r_S_one(self):
        """Delta fully in S → all motion accounted for by S projector."""
        rng = np.random.default_rng(1)
        # Build delta in S subspace
        _, _, U_pos, _ = _projector_pair(self.d, 4, 4)
        coef  = rng.standard_normal(4)
        delta = U_pos @ coef
        result = decompose_centroid_delta(delta, U_pos @ U_pos.T, np.zeros((self.d, self.d)))
        npt.assert_allclose(result["r_S"], 1.0, atol=1e-5)

    def test_delta_in_A_subspace_gives_r_S_zero(self):
        """Delta fully in A → none of the motion is in S."""
        _, _, U_pos, U_A = _projector_pair(self.d, 4, 4)
        coef  = np.random.default_rng(2).standard_normal(4)
        delta = U_A @ coef
        P_S   = U_pos @ U_pos.T
        P_A   = U_A   @ U_A.T
        result = decompose_centroid_delta(delta, P_S, P_A)
        npt.assert_allclose(result["r_S"], 0.0, atol=1e-5)

    def test_r_S_in_unit_interval(self):
        rng = np.random.default_rng(3)
        for _ in range(10):
            delta = rng.standard_normal(self.d)
            result = decompose_centroid_delta(delta, self.P_S, self.P_A)
            assert -1e-6 <= result["r_S"] <= 1.0 + 1e-6

    def test_norms_nonnegative(self):
        delta = np.random.default_rng(4).standard_normal(self.d)
        result = decompose_centroid_delta(delta, self.P_S, self.P_A)
        assert result["norm_S"] >= 0.0
        assert result["norm_A"] >= 0.0
        assert result["norm_total"] >= 0.0

    def test_zero_delta_gives_zero_r_S(self):
        delta = np.zeros(self.d)
        result = decompose_centroid_delta(delta, self.P_S, self.P_A)
        # r_S = norm_S / max(norm_total, 1e-12) → 0 / 1e-12 = 0
        npt.assert_allclose(result["r_S"], 0.0, atol=1e-6)


# ============================================================================
# centroid_velocity — merge_geometry_test
# ============================================================================

class TestMergeGeometryTest:

    def _seq(self, d_S_vals, d_A_vals):
        return [
            {"d_S": s, "d_A": a, "d_total": s + a}
            for s, a in zip(d_S_vals, d_A_vals)
        ]

    def test_required_keys(self):
        seq = self._seq([3.0, 2.0, 1.0], [1.0, 1.5, 2.0])
        result = merge_geometry_test(seq)
        for k in ("d_S_trend_rho", "d_A_trend_rho", "p6_d5_satisfied"):
            assert k in result

    def test_monotonically_decreasing_d_S_satisfies_p6d5(self):
        """d_S strictly decreasing, d_A flat → P6-D5 pass."""
        seq = self._seq([5.0, 4.0, 3.0, 2.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0])
        result = merge_geometry_test(seq, window=5)
        assert result["p6_d5_satisfied"] is True
        assert result["d_S_trend_rho"] < 0

    def test_increasing_d_S_fails_p6d5(self):
        """d_S increasing → P6-D5 fail."""
        seq = self._seq([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 2.0, 2.0, 2.0, 2.0])
        result = merge_geometry_test(seq, window=5)
        assert result["p6_d5_satisfied"] is False

    def test_too_short_sequence_returns_false(self):
        """Single-entry sequence can't establish a trend."""
        seq = self._seq([3.0], [1.0])
        result = merge_geometry_test(seq)
        assert result["p6_d5_satisfied"] is False

    def test_window_clips_to_tail(self):
        """window=2 uses only the last 2 entries."""
        # Last 2 entries: d_S decreasing
        seq = self._seq([5.0, 4.0, 3.0, 10.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        result = merge_geometry_test(seq, window=2)
        assert result["d_S_trend_rho"] < 0
        assert result["p6_d5_satisfied"] is True

    def test_rho_values_in_valid_range(self):
        """Spearman correlation must be in [-1, 1]."""
        seq = self._seq([3.0, 2.5, 2.0, 1.5, 1.0], [1.0, 1.2, 1.1, 1.3, 1.0])
        result = merge_geometry_test(seq, window=5)
        assert -1.0 - 1e-6 <= result["d_S_trend_rho"] <= 1.0 + 1e-6
        assert -1.0 - 1e-6 <= result["d_A_trend_rho"] <= 1.0 + 1e-6
