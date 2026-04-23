"""
tests/test_phase2_trajectory.py

Unit tests for p2_eigenspectra.trajectory pure-computation functions:

  step_size_trajectory(activations)
    — activations: (n_layers, n_tokens, d_model)
    → dict with step_norms (n_layers-1, n_tokens), step_mean, step_std, etc.

  subspace_activation(activations, projectors)
    → dict with {schur,sym}_{attract,repulse}_frac per layer

  self_interaction_trajectory(activations, OV)
    → dict with self_int (n_layers, n_tokens) = x @ OV @ x^T per token

Run from the project root with:
    pytest tests/test_phase2_trajectory.py -v
"""

import numpy as np
import numpy.testing as npt
import pytest

from p2_eigenspectra.trajectory import (
    self_interaction_trajectory,
    step_size_trajectory,
    subspace_activation,
)
from p2_eigenspectra.weights import build_subspace_projectors, eigendecompose


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pos_def(d: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    return A @ A.T + d * np.eye(d)


def _mixed_block(d: int = 8, seed: int = 1) -> np.ndarray:
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


def _mixed_projectors(d: int = 8) -> tuple:
    """Return (projectors dict, decomp dict) for a d×d mixed block matrix."""
    M = _mixed_block(d)
    dec = eigendecompose(M)
    proj = build_subspace_projectors(dec)
    return proj, dec


# ---------------------------------------------------------------------------
# step_size_trajectory
# ---------------------------------------------------------------------------

class TestStepSizeTrajectoryConstant:
    """Constant activation at every layer → all step sizes are zero."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        rng = np.random.default_rng(10)
        n_layers, n_tokens, d = 5, 4, 8
        base = rng.standard_normal((n_tokens, d))
        # Broadcast same frame to every layer
        self.acts = np.broadcast_to(base, (n_layers, n_tokens, d)).copy().astype(
            np.float32
        )
        self.result = step_size_trajectory(self.acts)

    def test_step_norms_are_zero(self):
        npt.assert_allclose(self.result["step_norms"], 0.0, atol=1e-5)

    def test_step_mean_is_zero(self):
        npt.assert_allclose(self.result["step_mean"], 0.0, atol=1e-5)

    def test_step_std_is_zero(self):
        npt.assert_allclose(self.result["step_std"], 0.0, atol=1e-5)

    def test_global_mean_is_zero(self):
        npt.assert_allclose(self.result["global_mean"], 0.0, atol=1e-5)

    def test_output_shape(self):
        n_layers, n_tokens = self.acts.shape[:2]
        assert self.result["step_norms"].shape == (n_layers - 1, n_tokens)
        assert self.result["step_mean"].shape == (n_layers - 1,)


class TestStepSizeTrajectoryMonotone:
    """
    Each layer adds an orthogonal perturbation whose magnitude grows linearly
    with layer index.  This gives step_norms that are non-decreasing.

    Construction for token 0 (single token for clarity):
      x[0] = e_0
      x[l] = x[l-1] + l * eps * e_l   (e_l orthonormal basis vector)

    Δx[l-1→l] = l * eps * e_l
    ||Δx|| grows as l * eps while ||x[l-1]|| ≈ 1 for small eps,
    so step_norms[l-1] ≈ l * eps — strictly increasing.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        n_layers, n_tokens, d = 7, 1, 16
        eps = 0.05

        acts = np.zeros((n_layers, n_tokens, d))
        acts[0, 0, 0] = 1.0  # start at e_0, unit norm
        for l in range(1, n_layers):
            acts[l] = acts[l - 1].copy()
            acts[l, 0, l] += l * eps  # grow perturbation along fresh axis

        self.acts = acts.astype(np.float32)
        self.result = step_size_trajectory(self.acts)

    def test_step_norms_are_nondecreasing(self):
        # step_mean (mean over tokens) should be non-decreasing
        step_mean = self.result["step_mean"]
        for i in range(len(step_mean) - 1):
            assert step_mean[i] <= step_mean[i + 1] + 1e-5, (
                f"step_mean[{i}]={step_mean[i]:.6f} > step_mean[{i+1}]={step_mean[i+1]:.6f}"
            )

    def test_first_step_is_positive(self):
        # There is movement from layer 0 to layer 1
        assert self.result["step_mean"][0] > 0.0

    def test_last_step_is_largest(self):
        step_mean = self.result["step_mean"]
        assert step_mean[-1] >= step_mean[0] - 1e-5


class TestStepSizeTrajectoryOutputContract:
    """Shape and value contracts independent of trajectory type."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        rng = np.random.default_rng(20)
        self.n_layers, self.n_tokens, d = 6, 5, 8
        acts = rng.standard_normal((self.n_layers, self.n_tokens, d)).astype(np.float32)
        self.result = step_size_trajectory(acts)

    def test_step_norms_shape(self):
        assert self.result["step_norms"].shape == (
            self.n_layers - 1, self.n_tokens
        )

    def test_step_mean_shape(self):
        assert self.result["step_mean"].shape == (self.n_layers - 1,)

    def test_step_std_shape(self):
        assert self.result["step_std"].shape == (self.n_layers - 1,)

    def test_step_norms_are_nonnegative(self):
        assert np.all(self.result["step_norms"] >= 0.0)

    def test_overshoot_threshold_above_global_mean(self):
        assert self.result["overshoot_threshold"] >= self.result["global_mean"]

    def test_overshoot_threshold_equals_mean_plus_two_std(self):
        expected = self.result["global_mean"] + 2.0 * self.result["global_std"]
        npt.assert_allclose(
            self.result["overshoot_threshold"], expected, atol=1e-5
        )


# ---------------------------------------------------------------------------
# subspace_activation
# ---------------------------------------------------------------------------

class TestSubspaceActivationInSubspace:
    """Activations that live entirely in the sym_attract subspace → frac = 1."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        d = 8
        n_layers, n_tokens = 5, 6
        rng = np.random.default_rng(30)

        proj, dec = _mixed_projectors(d)
        self.proj = proj

        # Get the eigenvectors spanning the attractive (positive-eigenvalue) subspace
        sym_vals = dec["sym_eigenvalues"]
        sym_vecs = dec["sym_eigenvectors"]  # columns are eigenvectors
        U_pos = sym_vecs[:, sym_vals > 0]   # (d, n_pos)

        # Activations = linear combinations of U_pos columns
        coefs = rng.standard_normal((n_layers, n_tokens, U_pos.shape[1]))
        acts = np.einsum("lnk,dk->lnd", coefs, U_pos)
        self.acts = acts.astype(np.float32)
        self.n_layers = n_layers
        self.result = subspace_activation(self.acts, self.proj)

    def test_sym_attract_frac_is_one(self):
        npt.assert_allclose(
            self.result["sym_attract_frac"], 1.0, atol=1e-5
        )

    def test_sym_repulse_frac_is_zero(self):
        npt.assert_allclose(
            self.result["sym_repulse_frac"], 0.0, atol=1e-5
        )

    def test_output_length_matches_n_layers(self):
        assert len(self.result["sym_attract_frac"]) == self.n_layers


class TestSubspaceActivationOrthogonal:
    """Activations orthogonal to sym_attract → attract frac = 0, repulse frac = 1."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        d = 8
        n_layers, n_tokens = 5, 6
        rng = np.random.default_rng(31)

        proj, dec = _mixed_projectors(d)
        self.proj = proj

        sym_vals = dec["sym_eigenvalues"]
        sym_vecs = dec["sym_eigenvectors"]
        U_neg = sym_vecs[:, sym_vals < 0]  # (d, n_neg)

        coefs = rng.standard_normal((n_layers, n_tokens, U_neg.shape[1]))
        acts = np.einsum("lnk,dk->lnd", coefs, U_neg)
        self.acts = acts.astype(np.float32)
        self.result = subspace_activation(self.acts, self.proj)

    def test_sym_attract_frac_is_zero(self):
        npt.assert_allclose(
            self.result["sym_attract_frac"], 0.0, atol=1e-5
        )

    def test_sym_repulse_frac_is_one(self):
        npt.assert_allclose(
            self.result["sym_repulse_frac"], 1.0, atol=1e-5
        )


class TestSubspaceActivationSchur:
    """Schur-based projectors: same in-subspace / orthogonal invariants."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        d = 8
        n_layers, n_tokens = 4, 5
        rng = np.random.default_rng(32)

        proj, dec = _mixed_projectors(d)
        self.proj = proj
        Z = dec["schur_Z"]
        n_a = dec["schur_n_attractive"]

        Z_attract = Z[:, :n_a]    # (d, n_a)
        Z_repulse = Z[:, n_a:]    # (d, d - n_a)

        coefs_a = rng.standard_normal((n_layers, n_tokens, n_a))
        acts_in = np.einsum("lnk,dk->lnd", coefs_a, Z_attract)
        self.acts_in = acts_in.astype(np.float32)

        coefs_r = rng.standard_normal((n_layers, n_tokens, Z_repulse.shape[1]))
        acts_out = np.einsum("lnk,dk->lnd", coefs_r, Z_repulse)
        self.acts_out = acts_out.astype(np.float32)

    def test_schur_attract_frac_is_one_when_in_subspace(self):
        result = subspace_activation(self.acts_in, self.proj)
        npt.assert_allclose(result["schur_attract_frac"], 1.0, atol=1e-5)

    def test_schur_attract_frac_is_zero_when_orthogonal(self):
        result = subspace_activation(self.acts_out, self.proj)
        npt.assert_allclose(result["schur_attract_frac"], 0.0, atol=1e-5)

    def test_schur_repulse_frac_is_one_when_in_repulse_subspace(self):
        result = subspace_activation(self.acts_out, self.proj)
        npt.assert_allclose(result["schur_repulse_frac"], 1.0, atol=1e-5)


class TestSubspaceActivationOutputContract:
    """Shape and key-existence checks."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        rng = np.random.default_rng(33)
        self.n_layers, n_tokens, d = 6, 5, 8
        acts = rng.standard_normal((self.n_layers, n_tokens, d)).astype(np.float32)
        proj, _ = _mixed_projectors(d)
        self.result = subspace_activation(acts, proj)

    def test_all_keys_present(self):
        for key in [
            "schur_attract_frac",
            "schur_repulse_frac",
            "sym_attract_frac",
            "sym_repulse_frac",
        ]:
            assert key in self.result, f"Missing key: {key}"

    def test_output_arrays_have_length_n_layers(self):
        for key in [
            "schur_attract_frac",
            "schur_repulse_frac",
            "sym_attract_frac",
            "sym_repulse_frac",
        ]:
            assert len(self.result[key]) == self.n_layers

    def test_fracs_are_in_unit_interval(self):
        for key in [
            "sym_attract_frac",
            "sym_repulse_frac",
            "schur_attract_frac",
            "schur_repulse_frac",
        ]:
            arr = self.result[key]
            assert np.all(arr >= -1e-6), f"{key} has values below 0"
            assert np.all(arr <= 1.0 + 1e-6), f"{key} has values above 1"


# ---------------------------------------------------------------------------
# self_interaction_trajectory
# ---------------------------------------------------------------------------

class TestSelfInteractionPositiveDefiniteOV:
    """
    Positive-definite OV → x @ OV @ x^T > 0 for any nonzero x.
    self_int[l, n] = x_{l,n} @ OV @ x_{l,n}^T (row convention).
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        d = 8
        rng = np.random.default_rng(40)
        self.OV = _pos_def(d, seed=40)

        n_layers, n_tokens = 5, 6
        acts = rng.standard_normal((n_layers, n_tokens, d))
        # Normalise so magnitudes don't obscure sign
        norms = np.linalg.norm(acts, axis=-1, keepdims=True)
        self.acts = (acts / np.maximum(norms, 1e-10)).astype(np.float32)
        self.result = self_interaction_trajectory(self.acts, self.OV)

    def test_all_self_interactions_positive(self):
        # Symmetric part of pos-def OV is pos-def → x^T OV x > 0 for all x ≠ 0
        assert np.all(self.result["self_int"] > 0), (
            "Expected all self-interactions > 0 for positive-definite OV"
        )

    def test_frac_negative_is_zero(self):
        npt.assert_allclose(self.result["frac_negative"], 0.0, atol=1e-5)


class TestSelfInteractionNegativeDefiniteOV:
    """Negative-definite OV → all self-interactions negative."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        d = 8
        rng = np.random.default_rng(41)
        self.OV = -_pos_def(d, seed=41)

        n_layers, n_tokens = 5, 6
        acts = rng.standard_normal((n_layers, n_tokens, d))
        norms = np.linalg.norm(acts, axis=-1, keepdims=True)
        self.acts = (acts / np.maximum(norms, 1e-10)).astype(np.float32)
        self.result = self_interaction_trajectory(self.acts, self.OV)

    def test_all_self_interactions_negative(self):
        assert np.all(self.result["self_int"] < 0)

    def test_frac_negative_is_one(self):
        npt.assert_allclose(self.result["frac_negative"], 1.0, atol=1e-5)


class TestSelfInteractionZeroActivation:
    """Zero activations → self-interaction is exactly zero."""

    def test_zero_activations_give_zero_self_int(self):
        d = 8
        acts = np.zeros((4, 5, d), dtype=np.float32)
        OV = _pos_def(d)
        result = self_interaction_trajectory(acts, OV)
        npt.assert_allclose(result["self_int"], 0.0, atol=1e-5)


class TestSelfInteractionOutputContract:
    """Shape and key-existence checks."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        rng = np.random.default_rng(42)
        self.n_layers, self.n_tokens, d = 6, 7, 8
        self.acts = rng.standard_normal(
            (self.n_layers, self.n_tokens, d)
        ).astype(np.float32)
        OV = _pos_def(d)
        self.result = self_interaction_trajectory(self.acts, OV)

    def test_self_int_shape(self):
        assert self.result["self_int"].shape == (self.n_layers, self.n_tokens)

    def test_self_int_mean_shape(self):
        assert self.result["self_int_mean"].shape == (self.n_layers,)

    def test_self_int_std_shape(self):
        assert self.result["self_int_std"].shape == (self.n_layers,)

    def test_frac_negative_shape(self):
        assert self.result["frac_negative"].shape == (self.n_layers,)

    def test_frac_negative_in_unit_interval(self):
        arr = self.result["frac_negative"]
        assert np.all(arr >= 0.0) and np.all(arr <= 1.0 + 1e-6)

    def test_mean_equals_per_layer_mean_of_self_int(self):
        expected = self.result["self_int"].mean(axis=1)
        npt.assert_allclose(self.result["self_int_mean"], expected, atol=1e-5)

    def test_std_equals_per_layer_std_of_self_int(self):
        expected = self.result["self_int"].std(axis=1)
        npt.assert_allclose(self.result["self_int_std"], expected, atol=1e-5)

    def test_frac_negative_matches_self_int_sign(self):
        expected = (self.result["self_int"] < 0).mean(axis=1)
        npt.assert_allclose(self.result["frac_negative"], expected, atol=1e-5)
