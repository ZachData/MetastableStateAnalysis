"""
tests/test_phase2_weights.py

Unit tests for p2_eigenspectra.weights pure-computation functions:

  eigendecompose(M)
    — returns dict with frac_attractive, frac_repulsive, schur_n_attractive,
      sym_frac_attractive, sym_frac_repulsive, eigenvalues, etc.

  build_subspace_projectors(decomp)
    — returns dict with schur_attract, schur_repulse, sym_attract, sym_repulse
      and dimension counts

  rescale_matrix(M)
    — returns scipy.linalg.expm(-M)

Run from the project root with:
    pytest tests/test_phase2_weights.py -v
"""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import expm

from p2_eigenspectra.weights import (
    build_subspace_projectors,
    eigendecompose,
    rescale_matrix,
)


# ---------------------------------------------------------------------------
# Synthetic matrix factories
# ---------------------------------------------------------------------------

def _pos_def(d: int = 6, seed: int = 0) -> np.ndarray:
    """Symmetric positive-definite matrix: all eigenvalues > 0."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    return A @ A.T + d * np.eye(d)  # strictly pos-def


def _neg_def(d: int = 6, seed: int = 0) -> np.ndarray:
    """Symmetric negative-definite matrix: all eigenvalues < 0."""
    return -_pos_def(d, seed)


def _mixed_block(d: int = 8, seed: int = 1) -> np.ndarray:
    """
    Symmetric block-diagonal with d//2 positive and d//2 negative eigenvalues.
    The symmetric part is constructed directly so frac_{attract,repulse} = 0.5.
    """
    assert d % 2 == 0
    half = d // 2
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((half, half))
    pos_block = A @ A.T + half * np.eye(half)

    B = rng.standard_normal((half, half))
    neg_block = -(B @ B.T + half * np.eye(half))

    M = np.zeros((d, d))
    M[:half, :half] = pos_block
    M[half:, half:] = neg_block
    return M


# ---------------------------------------------------------------------------
# eigendecompose — positive-definite input
# ---------------------------------------------------------------------------

class TestEigendecomposePosDef:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.d = 6
        self.M = _pos_def(self.d)
        self.dec = eigendecompose(self.M)

    def test_frac_attractive_is_one(self):
        npt.assert_allclose(self.dec["frac_attractive"], 1.0, atol=1e-5)

    def test_frac_repulsive_is_zero(self):
        npt.assert_allclose(self.dec["frac_repulsive"], 0.0, atol=1e-5)

    def test_sym_frac_attractive_is_one(self):
        npt.assert_allclose(self.dec["sym_frac_attractive"], 1.0, atol=1e-5)

    def test_sym_frac_repulsive_is_zero(self):
        npt.assert_allclose(self.dec["sym_frac_repulsive"], 0.0, atol=1e-5)

    def test_schur_n_attractive_equals_full_dim(self):
        assert self.dec["schur_n_attractive"] == self.d

    def test_all_eigenvalues_have_positive_real_part(self):
        # Symmetric pos-def → all eigenvalues are real and positive
        npt.assert_array_less(0.0, self.dec["eig_real"])

    def test_no_spurious_imaginary_parts(self):
        # Symmetric matrix → purely real eigenvalues
        npt.assert_allclose(self.dec["eig_imag"], 0.0, atol=1e-5)

    def test_methods_agree(self):
        assert self.dec["agree"] is True

    def test_sym_eigenvalues_all_positive(self):
        npt.assert_array_less(0.0, self.dec["sym_eigenvalues"])

    def test_return_shape_eigenvalues(self):
        assert self.dec["eigenvalues"].shape == (self.d,)

    def test_schur_Z_is_orthogonal(self):
        Z = self.dec["schur_Z"]
        npt.assert_allclose(Z @ Z.T, np.eye(self.d), atol=1e-5)


# ---------------------------------------------------------------------------
# eigendecompose — negative-definite input
# ---------------------------------------------------------------------------

class TestEigendecomposeNegDef:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.d = 6
        self.M = _neg_def(self.d)
        self.dec = eigendecompose(self.M)

    def test_frac_repulsive_is_one(self):
        npt.assert_allclose(self.dec["frac_repulsive"], 1.0, atol=1e-5)

    def test_frac_attractive_is_zero(self):
        npt.assert_allclose(self.dec["frac_attractive"], 0.0, atol=1e-5)

    def test_sym_frac_repulsive_is_one(self):
        npt.assert_allclose(self.dec["sym_frac_repulsive"], 1.0, atol=1e-5)

    def test_sym_frac_attractive_is_zero(self):
        npt.assert_allclose(self.dec["sym_frac_attractive"], 0.0, atol=1e-5)

    def test_schur_n_attractive_is_zero(self):
        assert self.dec["schur_n_attractive"] == 0

    def test_all_eigenvalues_have_negative_real_part(self):
        npt.assert_array_less(self.dec["eig_real"], 0.0)

    def test_sym_eigenvalues_all_negative(self):
        npt.assert_array_less(self.dec["sym_eigenvalues"], 0.0)

    def test_methods_agree(self):
        assert self.dec["agree"] is True


# ---------------------------------------------------------------------------
# eigendecompose — mixed (block diagonal, equal halves)
# ---------------------------------------------------------------------------

class TestEigendecomposeMixed:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.d = 8
        self.M = _mixed_block(self.d)
        self.dec = eigendecompose(self.M)

    def test_frac_attractive_nonzero(self):
        assert self.dec["frac_attractive"] > 0.0

    def test_frac_repulsive_nonzero(self):
        assert self.dec["frac_repulsive"] > 0.0

    def test_fracs_sum_to_one(self):
        # All eigenvalues of a symmetric matrix are real; sum should be 1
        total = self.dec["frac_attractive"] + self.dec["frac_repulsive"]
        npt.assert_allclose(total, 1.0, atol=1e-5)

    def test_sym_frac_attractive_is_half(self):
        # Block diagonal: d/2 positive eigenvalues
        npt.assert_allclose(self.dec["sym_frac_attractive"], 0.5, atol=1e-5)

    def test_sym_frac_repulsive_is_half(self):
        npt.assert_allclose(self.dec["sym_frac_repulsive"], 0.5, atol=1e-5)

    def test_sym_fracs_sum_to_one(self):
        total = self.dec["sym_frac_attractive"] + self.dec["sym_frac_repulsive"]
        npt.assert_allclose(total, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# build_subspace_projectors — positive-definite (attractive spans everything)
# ---------------------------------------------------------------------------

class TestProjectorsPosDef:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.d = 6
        self.M = _pos_def(self.d)
        self.proj = build_subspace_projectors(eigendecompose(self.M))

    def test_schur_attract_is_identity(self):
        npt.assert_allclose(self.proj["schur_attract"], np.eye(self.d), atol=1e-5)

    def test_schur_repulse_is_zero(self):
        npt.assert_allclose(
            self.proj["schur_repulse"], np.zeros((self.d, self.d)), atol=1e-5
        )

    def test_sym_attract_is_identity(self):
        npt.assert_allclose(self.proj["sym_attract"], np.eye(self.d), atol=1e-5)

    def test_sym_repulse_is_zero(self):
        npt.assert_allclose(
            self.proj["sym_repulse"], np.zeros((self.d, self.d)), atol=1e-5
        )

    def test_schur_dim_attract_equals_d(self):
        assert self.proj["schur_dim_attract"] == self.d

    def test_schur_dim_repulse_is_zero(self):
        assert self.proj["schur_dim_repulse"] == 0

    def test_sym_dim_attract_equals_d(self):
        assert self.proj["sym_dim_attract"] == self.d

    def test_sym_dim_repulse_is_zero(self):
        assert self.proj["sym_dim_repulse"] == 0


# ---------------------------------------------------------------------------
# build_subspace_projectors — negative-definite (repulsive spans everything)
# ---------------------------------------------------------------------------

class TestProjectorsNegDef:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.d = 6
        self.M = _neg_def(self.d)
        self.proj = build_subspace_projectors(eigendecompose(self.M))

    def test_schur_repulse_is_identity(self):
        npt.assert_allclose(self.proj["schur_repulse"], np.eye(self.d), atol=1e-5)

    def test_schur_attract_is_zero(self):
        npt.assert_allclose(
            self.proj["schur_attract"], np.zeros((self.d, self.d)), atol=1e-5
        )

    def test_sym_repulse_is_identity(self):
        npt.assert_allclose(self.proj["sym_repulse"], np.eye(self.d), atol=1e-5)

    def test_sym_attract_is_zero(self):
        npt.assert_allclose(
            self.proj["sym_attract"], np.zeros((self.d, self.d)), atol=1e-5
        )

    def test_schur_dim_repulse_equals_d(self):
        assert self.proj["schur_dim_repulse"] == self.d

    def test_schur_dim_attract_is_zero(self):
        assert self.proj["schur_dim_attract"] == 0


# ---------------------------------------------------------------------------
# build_subspace_projectors — mixed block diagonal
#   Key invariants regardless of which subspace decomposition method is used:
#   (i)  P_a + P_r = I  (partition of identity on their joint support)
#   (ii) P_a @ P_r = 0  (orthogonal subspaces)
#   (iii) P^2 = P       (idempotent)
#   (iv) symmetric      (P^T = P)
# ---------------------------------------------------------------------------

class TestProjectorsMixed:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.d = 8
        self.M = _mixed_block(self.d)
        self.proj = build_subspace_projectors(eigendecompose(self.M))

    # --- Schur projectors ---

    def test_schur_projectors_sum_to_identity(self):
        total = self.proj["schur_attract"] + self.proj["schur_repulse"]
        npt.assert_allclose(total, np.eye(self.d), atol=1e-5)

    def test_schur_projectors_are_orthogonal(self):
        cross = self.proj["schur_attract"] @ self.proj["schur_repulse"]
        npt.assert_allclose(cross, np.zeros((self.d, self.d)), atol=1e-5)

    def test_schur_attract_is_idempotent(self):
        P = self.proj["schur_attract"]
        npt.assert_allclose(P @ P, P, atol=1e-5)

    def test_schur_repulse_is_idempotent(self):
        P = self.proj["schur_repulse"]
        npt.assert_allclose(P @ P, P, atol=1e-5)

    def test_schur_attract_is_symmetric(self):
        P = self.proj["schur_attract"]
        npt.assert_allclose(P, P.T, atol=1e-5)

    def test_schur_repulse_is_symmetric(self):
        P = self.proj["schur_repulse"]
        npt.assert_allclose(P, P.T, atol=1e-5)

    # --- Symmetric-part projectors ---

    def test_sym_projectors_sum_to_identity(self):
        # No zero eigenvalues in _mixed_block, so P_a + P_r = I exactly
        total = self.proj["sym_attract"] + self.proj["sym_repulse"]
        npt.assert_allclose(total, np.eye(self.d), atol=1e-5)

    def test_sym_projectors_are_orthogonal(self):
        cross = self.proj["sym_attract"] @ self.proj["sym_repulse"]
        npt.assert_allclose(cross, np.zeros((self.d, self.d)), atol=1e-5)

    def test_sym_attract_is_idempotent(self):
        P = self.proj["sym_attract"]
        npt.assert_allclose(P @ P, P, atol=1e-5)

    def test_sym_repulse_is_idempotent(self):
        P = self.proj["sym_repulse"]
        npt.assert_allclose(P @ P, P, atol=1e-5)

    def test_sym_attract_is_symmetric(self):
        P = self.proj["sym_attract"]
        npt.assert_allclose(P, P.T, atol=1e-5)

    # --- Dimension accounting ---

    def test_schur_dims_sum_to_d(self):
        assert (
            self.proj["schur_dim_attract"] + self.proj["schur_dim_repulse"] == self.d
        )

    def test_sym_dims_are_equal_for_equal_block_sizes(self):
        # d=8, d//2=4 positive + 4 negative eigenvalues
        assert self.proj["sym_dim_attract"] == self.proj["sym_dim_repulse"]

    def test_sym_dims_sum_to_d(self):
        assert self.proj["sym_dim_attract"] + self.proj["sym_dim_repulse"] == self.d


# ---------------------------------------------------------------------------
# rescale_matrix
# ---------------------------------------------------------------------------

class TestRescaleMatrix:
    def test_zero_matrix_gives_identity(self):
        M = np.zeros((5, 5))
        npt.assert_allclose(rescale_matrix(M), np.eye(5), atol=1e-5)

    def test_diagonal_matrix_gives_elementwise_exp(self):
        lam = np.array([1.0, 2.0, -1.0, 0.5, 3.0])
        M = np.diag(lam)
        R = rescale_matrix(M)
        npt.assert_allclose(R, np.diag(np.exp(-lam)), atol=1e-5)

    def test_output_shape_preserved(self):
        for d in [4, 8, 16]:
            M = np.eye(d)
            assert rescale_matrix(M).shape == (d, d)

    def test_inverse_relation_expm_minus_M_times_expm_M_is_identity(self):
        # expm(-M) @ expm(M) = I  for any M
        rng = np.random.default_rng(99)
        M = rng.standard_normal((6, 6))
        R = rescale_matrix(M)
        npt.assert_allclose(R @ expm(M), np.eye(6), atol=1e-5)

    def test_scalar_multiple_scaling(self):
        # rescale_matrix(alpha * I) = exp(-alpha) * I
        alpha = 2.5
        d = 4
        M = alpha * np.eye(d)
        npt.assert_allclose(rescale_matrix(M), np.exp(-alpha) * np.eye(d), atol=1e-5)

    def test_symmetric_pos_def_input_gives_symmetric_output(self):
        # expm of a symmetric matrix is symmetric
        M = _pos_def(6)
        R = rescale_matrix(M)
        npt.assert_allclose(R, R.T, atol=1e-5)

    def test_output_is_float_array(self):
        M = np.eye(4)
        R = rescale_matrix(M)
        assert R.dtype.kind == "f"


#new tests


class TestProjectorComplementarity:
    """attract + repulse = I for both schur and sym projectors,
    regardless of input matrix signature."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_schur_complementarity_random(self, seed):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((8, 8))
        proj = build_subspace_projectors(eigendecompose(M))
        total = proj["schur_attract"] + proj["schur_repulse"]
        npt.assert_allclose(total, np.eye(8), atol=1e-8)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_sym_complementarity_random(self, seed):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((8, 8))
        proj = build_subspace_projectors(eigendecompose(M))
        total = proj["sym_attract"] + proj["sym_repulse"]
        npt.assert_allclose(total, np.eye(8), atol=1e-8)

    def test_complementarity_pos_def(self):
        proj = build_subspace_projectors(eigendecompose(_pos_def(6)))
        npt.assert_allclose(
            proj["schur_attract"] + proj["schur_repulse"], np.eye(6), atol=1e-8
        )

    def test_complementarity_neg_def(self):
        proj = build_subspace_projectors(eigendecompose(-_pos_def(6)))
        npt.assert_allclose(
            proj["schur_repulse"] + proj["schur_attract"], np.eye(6), atol=1e-8
        )


class TestEigendecomposeSignClassification:
    """eigendecompose correctly assigns eigenvalues to attract (positive)
    and repulse (negative) categories."""

    def test_pos_def_all_eigenvalues_attract(self):
        d = 6
        M = _pos_def(d)
        dec = eigendecompose(M)
        # All eigenvalues positive → attract count = d, repulse count = 0
        n_attract = (np.array(dec["eigenvalues"]) > 0).sum()
        assert n_attract == d

    def test_neg_def_all_eigenvalues_repulse(self):
        d = 6
        M = -_pos_def(d)
        dec = eigendecompose(M)
        n_repulse = (np.array(dec["eigenvalues"]) < 0).sum()
        assert n_repulse == d

    def test_mixed_block_equal_attract_repulse(self):
        # _mixed_block: d//2 positive + d//2 negative eigenvalues
        d = 8
        M = _mixed_block(d)
        dec = eigendecompose(M)
        eigs = np.array(dec["eigenvalues"])
        assert (eigs > 0).sum() == d // 2
        assert (eigs < 0).sum() == d // 2

    def test_eigenvalues_are_real(self):
        """Symmetric input → all eigenvalues are real."""
        M = _pos_def(6)
        dec = eigendecompose(M)
        for ev in dec["eigenvalues"]:
            assert abs(np.imag(ev)) < 1e-10