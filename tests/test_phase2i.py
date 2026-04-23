"""
tests/test_phase2i.py — Unit tests for Phase 2i modules.

Exercises four modules against analytically-known cases:
  rotational_schur.py   — Schur decomposition of rotation matrices
  rotational_rescaled.py — identity rescaling leaves activations unchanged
  fiedler_tracking.py   — Fiedler vector on block-diagonal Gram, stability
  rotation_hemisphere.py — plane_fiedler_alignment geometry

All inputs are synthetic numpy arrays. No model, no tokenizer.
Dimensions: d=8, n_tokens=20, n_layers=6.

Run:
    python -m unittest tests.test_phase2i -v
    python tests/test_phase2i.py
"""

from __future__ import annotations

import sys
import os
import unittest

import numpy as np
from scipy.linalg import expm

# ── allow running from project root or tests/ ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotational_schur import (
    extract_schur_blocks,
    rotation_energy_fractions,
    rotation_angle_stats,
    henrici_nonnormality,
    build_rotation_plane_projectors,
)
from rotational_rescaled import (
    decompose_symmetric_antisymmetric,
    rescaled_trajectory_component,
)
from fiedler_tracking import (
    extract_fiedler_per_layer,
    hemisphere_assignments,
    hemisphere_crossing_rate,
    fiedler_stability,
)
from rotation_hemisphere import (
    plane_fiedler_alignment,
)

# ── shared constants ──────────────────────────────────────────────────────────
D        = 8
N_TOKENS = 20
N_LAYERS = 6


# ── helpers ───────────────────────────────────────────────────────────────────

def _rot2(theta: float) -> np.ndarray:
    """Standard 2×2 rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _rot_d8(thetas) -> np.ndarray:
    """Block-diagonal 8×8 rotation matrix from four angles."""
    M = np.zeros((D, D))
    for i, th in enumerate(thetas):
        M[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = _rot2(th)
    return M


def _unit_acts(n_layers=N_LAYERS, n_tokens=N_TOKENS, d=D, seed=42) -> np.ndarray:
    """Random L2-normalised activations (n_layers, n_tokens, d)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_layers, n_tokens, d)).astype(np.float64)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(norms, 1e-10)


def _block_acts(n_tokens=N_TOKENS, d=D) -> np.ndarray:
    """
    (n_tokens, d) activations forming two perfectly separated clusters:
      tokens  0 .. n_tokens//2 - 1  → unit vector along e_0
      tokens n_tokens//2 .. end      → unit vector along e_1
    Gram matrix is block-diagonal with 1s in each block and 0s across.
    """
    X = np.zeros((n_tokens, d), dtype=np.float64)
    X[: n_tokens // 2, 0] = 1.0
    X[n_tokens // 2 :, 1] = 1.0
    return X


def _block_acts_multilayer(n_layers=N_LAYERS) -> np.ndarray:
    """Same block-structured layer repeated n_layers times."""
    X = _block_acts()
    return np.stack([X] * n_layers, axis=0)  # (n_layers, n_tokens, d)


# ── helpers for gram ip_mean ──────────────────────────────────────────────────

def _ip_means(acts: np.ndarray) -> np.ndarray:
    """Per-layer mean pairwise inner product (upper triangle of Gram)."""
    n_layers, n_tokens, _ = acts.shape
    out = np.empty(n_layers)
    for L in range(n_layers):
        G = acts[L] @ acts[L].T
        idx = np.triu_indices(n_tokens, k=1)
        out[L] = G[idx].mean()
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 1. rotational_schur — Schur decomposition of rotation matrices
# ═════════════════════════════════════════════════════════════════════════════

class TestSchurPureRotation2x2(unittest.TestCase):
    """extract_schur_blocks on a 2×2 rotation matrix R(θ)."""

    def _blocks(self, theta):
        return extract_schur_blocks(_rot2(theta))

    def test_single_block_structure(self):
        """R(π/4) must produce exactly one 2×2 block and zero 1×1 blocks."""
        result = self._blocks(np.pi / 4)
        self.assertEqual(result["n_complex"], 1)
        self.assertEqual(result["n_real"], 0)

    def test_theta_recovered_exactly(self):
        """
        The rotation angle extracted from the Schur block equals θ for θ ∈ [0, π/2].

        The implementation uses arctan2(sqrt(-bc), abs(a)) = arctan2(|sinθ|, |cosθ|),
        which folds all angles onto [0, π/2].  Angles > π/2 are mapped to their
        complement: recovered = arctan2(|sinθ|, |cosθ|) ≠ θ for θ > π/2.
        We therefore only test angles in the unambiguous range [0, π/2].
        """
        for theta in [0.1, 0.5, np.pi / 4, np.pi / 3, np.pi / 2 - 0.01]:
            with self.subTest(theta=theta):
                result = self._blocks(theta)
                block = result["blocks_2x2"][0]
                self.assertAlmostEqual(
                    block["theta"], theta, places=8,
                    msg=f"theta mismatch: input {theta}, got {block['theta']}"
                )

    def test_theta_folded_for_obtuse_angles(self):
        """
        For θ > π/2 the code returns arctan2(|sinθ|, |cosθ|) = π - θ (first quadrant fold).
        We verify the folded value rather than θ itself.
        """
        for theta in [2.0, 2.5, np.pi - 0.2]:
            with self.subTest(theta=theta):
                expected_folded = float(np.arctan2(abs(np.sin(theta)), abs(np.cos(theta))))
                result = self._blocks(theta)
                block = result["blocks_2x2"][0]
                self.assertAlmostEqual(
                    block["theta"], expected_folded, places=8,
                    msg=f"θ={theta}: expected folded={expected_folded}, got {block['theta']}"
                )

    def test_spectral_radius_unity(self):
        """Pure rotation matrix has unit spectral radius ρ = 1."""
        for theta in [0.3, 1.1, 2.7]:
            with self.subTest(theta=theta):
                block = self._blocks(theta)["blocks_2x2"][0]
                self.assertAlmostEqual(block["rho"], 1.0, places=8)

    def test_bc_negative_genuine_rotation(self):
        """bc = b*c < 0 confirms complex conjugate pair (genuine rotation)."""
        block = self._blocks(np.pi / 4)["blocks_2x2"][0]
        self.assertLess(block["bc"], 0.0)

    def test_dimension_consistent(self):
        result = self._blocks(np.pi / 3)
        self.assertEqual(result["d"], 2)
        self.assertEqual(result["n_real"] + 2 * result["n_complex"], 2)


class TestSchurBlockDiagonalD8(unittest.TestCase):
    """extract_schur_blocks on a block-diagonal 8×8 rotation matrix."""

    THETAS = [0.3, 0.7, 1.1, 1.5]

    def _result(self):
        return extract_schur_blocks(_rot_d8(self.THETAS))

    def test_four_complex_zero_real_blocks(self):
        result = self._result()
        self.assertEqual(result["n_complex"], 4)
        self.assertEqual(result["n_real"], 0)

    def test_all_angles_recovered(self):
        """All four rotation angles are found (order may differ)."""
        result = self._result()
        recovered = sorted(b["theta"] for b in result["blocks_2x2"])
        for r, e in zip(recovered, sorted(self.THETAS)):
            self.assertAlmostEqual(r, e, places=8)

    def test_all_radii_unity(self):
        result = self._result()
        for block in result["blocks_2x2"]:
            self.assertAlmostEqual(block["rho"], 1.0, places=8)

    def test_all_spectral_energy_in_rotation(self):
        """Rotation matrix → rotational_fraction = 1, signed_fraction = 0."""
        result = self._result()
        energy = rotation_energy_fractions(result)
        self.assertAlmostEqual(energy["rotational_fraction"], 1.0, places=8)
        self.assertAlmostEqual(energy["signed_fraction"],     0.0, places=8)
        self.assertEqual(energy["n_real"],    0)
        self.assertEqual(energy["n_complex"], 4)
        self.assertAlmostEqual(energy["frac_complex_dims"], 1.0, places=8)

    def test_rotation_angle_stats_mean_and_count(self):
        result = self._result()
        stats = rotation_angle_stats(result)
        self.assertEqual(len(stats["thetas"]), 4)
        self.assertAlmostEqual(
            stats["theta_mean"],
            float(np.mean(self.THETAS)), places=8
        )

    def test_henrici_zero_for_normal_matrix(self):
        """
        An orthogonal matrix is normal.
        Henrici departure from normality must be 0.

        For real Schur form of a rotation block [[a,b],[c,a]]:
          T_frob² per block = 2a² + b² + c² = 2cos²θ + 2sin²θ = 2
          eigenvalue_energy  = 2 * ρ²          = 2 * 1             = 2
          henrici            = 0
        """
        result = self._result()
        h = henrici_nonnormality(result)
        self.assertAlmostEqual(h["henrici_absolute"], 0.0, places=6)
        self.assertAlmostEqual(h["henrici_relative"], 0.0, places=6)

    def test_schur_vectors_orthonormal(self):
        """Schur vectors (columns of Z) form an orthonormal set."""
        result = self._result()
        Z = result["schur_Z"]
        self.assertAlmostEqual(
            float(np.max(np.abs(Z.T @ Z - np.eye(D)))),
            0.0, places=8
        )


class TestSchurIdentityMatrix(unittest.TestCase):
    """Identity matrix: all real eigenvalues (+1), no rotation blocks."""

    def test_all_real_blocks(self):
        result = extract_schur_blocks(np.eye(D))
        self.assertEqual(result["n_complex"], 0)
        self.assertEqual(result["n_real"], D)

    def test_eigenvalues_all_one(self):
        result = extract_schur_blocks(np.eye(D))
        for b in result["blocks_1x1"]:
            self.assertAlmostEqual(b["value"], 1.0, places=10)


# ═════════════════════════════════════════════════════════════════════════════
# 2. rotational_rescaled — identity rescaling + S/A decomposition
# ═════════════════════════════════════════════════════════════════════════════

class TestIdentityRescaling(unittest.TestCase):
    """
    apply_rescaling with expm_matrix = I (i.e. OV = zero_matrix):
    activations that are already L2-normalised should be unchanged.
    """

    def setUp(self):
        self.acts = _unit_acts()            # pre-normalised
        self.zero_M = np.zeros((D, D))      # expm(-0) = I

    def test_max_valid_layer_equals_n_layers(self):
        result = rescaled_trajectory_component(
            self.acts, self.zero_M, beta_values=[1.0]
        )
        self.assertEqual(result["max_valid_layer"], N_LAYERS)

    def test_ip_means_unchanged(self):
        """
        With R_cum = I at every layer, normed activations are returned as-is.
        Mean pairwise inner product must match the unrescaled trajectory.
        """
        expected = _ip_means(self.acts)
        result = rescaled_trajectory_component(
            self.acts, self.zero_M, beta_values=[1.0]
        )
        for L in range(N_LAYERS):
            with self.subTest(layer=L):
                self.assertAlmostEqual(
                    result["ip_mean"][L], expected[L], places=6,
                    msg=f"Layer {L}: ip_mean changed under identity rescaling"
                )

    def test_effective_rank_finite(self):
        result = rescaled_trajectory_component(
            self.acts, self.zero_M, beta_values=[1.0]
        )
        self.assertTrue(
            np.all(np.isfinite(result["effective_rank"])),
            "effective_rank should be finite for all layers"
        )

    def test_per_layer_mode_identity(self):
        """Per-layer mode with all-zero matrices: same result as shared mode."""
        zero_list = [np.zeros((D, D))] * N_LAYERS
        result = rescaled_trajectory_component(
            self.acts, None, beta_values=[1.0],
            is_per_layer=True, matrices_list=zero_list,
        )
        expected = _ip_means(self.acts)
        for L in range(N_LAYERS):
            self.assertAlmostEqual(result["ip_mean"][L], expected[L], places=6)


class TestSymmetricAntisymmetricDecomposition(unittest.TestCase):
    """decompose_symmetric_antisymmetric: algebraic identities."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.OV = rng.standard_normal((D, D))

    def test_reconstruction(self):
        sa = decompose_symmetric_antisymmetric(self.OV)
        self.assertTrue(
            np.allclose(sa["S"] + sa["A"], self.OV, atol=1e-12),
            "S + A must equal OV"
        )

    def test_S_is_symmetric(self):
        sa = decompose_symmetric_antisymmetric(self.OV)
        self.assertTrue(np.allclose(sa["S"], sa["S"].T, atol=1e-12))

    def test_A_is_antisymmetric(self):
        sa = decompose_symmetric_antisymmetric(self.OV)
        self.assertTrue(np.allclose(sa["A"], -sa["A"].T, atol=1e-12))

    def test_pythagorean_norm_identity(self):
        """
        Since S and A are orthogonal in the Frobenius inner product
        (tr(S^T A) = 0 for symmetric S and antisymmetric A), we have
        ||S||²_F + ||A||²_F = ||OV||²_F.
        """
        sa = decompose_symmetric_antisymmetric(self.OV)
        lhs = sa["S_frob"] ** 2 + sa["A_frob"] ** 2
        rhs = sa["V_frob"] ** 2
        self.assertAlmostEqual(lhs, rhs, places=8)

    def test_symmetric_ov_gives_zero_A(self):
        """A symmetric OV has A = 0."""
        OV_sym = self.OV + self.OV.T
        sa = decompose_symmetric_antisymmetric(OV_sym)
        self.assertAlmostEqual(sa["A_frob"], 0.0, places=10)
        self.assertAlmostEqual(sa["rotation_ratio"], 0.0, places=10)

    def test_antisymmetric_ov_gives_zero_S(self):
        """An antisymmetric OV has S = 0."""
        OV_anti = self.OV - self.OV.T
        sa = decompose_symmetric_antisymmetric(OV_anti)
        self.assertAlmostEqual(sa["S_frob"], 0.0, places=10)


# ═════════════════════════════════════════════════════════════════════════════
# 3. fiedler_tracking — Fiedler vector on block-diagonal Gram
# ═════════════════════════════════════════════════════════════════════════════

class TestFiedlerBlockDiagonalGram(unittest.TestCase):
    """
    Two-block Gram matrix (perfect block-diagonal):
      tokens  0..9  → e_0   (inner products: 1 within block, 0 across)
      tokens 10..19 → e_1

    Fiedler vector: +c for block A, -c for block B (up to global sign).
    Fiedler eigenvalue: ≈ 0 (two disconnected components).
    """

    def setUp(self):
        X = _block_acts()                           # (20, 8)
        self.acts1 = X[np.newaxis, :, :]            # (1, 20, 8) — single layer
        self.result = extract_fiedler_per_layer(self.acts1)

    def test_computation_succeeds(self):
        self.assertTrue(self.result["valid"][0], "Fiedler should succeed for n_tokens=20")

    def test_fiedler_eigenvalue_near_zero(self):
        """Disconnected graph → algebraic connectivity = 0."""
        val = self.result["fiedler_vals"][0]
        self.assertLess(
            abs(val), 1e-6,
            f"Fiedler eigenvalue={val} should be ≈0 for two disconnected components"
        )

    def test_fiedler_separates_blocks_by_sign(self):
        """
        All tokens in block A share one sign; all in block B share the opposite.
        This holds regardless of the global sign of the eigenvector.
        """
        fv = self.result["fiedler_vecs"][0]      # (20,)
        half = N_TOKENS // 2

        # Signs within each block
        signs_A = np.sign(fv[:half])
        signs_B = np.sign(fv[half:])

        self.assertTrue(
            np.all(signs_A == signs_A[0]),
            f"Block-A signs not uniform: {signs_A}"
        )
        self.assertTrue(
            np.all(signs_B == signs_B[0]),
            f"Block-B signs not uniform: {signs_B}"
        )
        self.assertNotEqual(
            int(signs_A[0]), int(signs_B[0]),
            "Block A and B must have opposite Fiedler signs"
        )

    def test_fiedler_values_equal_within_block(self):
        """
        By symmetry, all tokens in the same block have exactly the same
        Fiedler coordinate (±1/sqrt(n_block) after normalisation by eigh).
        """
        fv = self.result["fiedler_vecs"][0]
        half = N_TOKENS // 2
        self.assertAlmostEqual(float(np.std(fv[:half])),  0.0, places=8)
        self.assertAlmostEqual(float(np.std(fv[half:])), 0.0, places=8)

    def test_hemisphere_assignments_perfectly_split(self):
        """Hemisphere assignments reproduce the two-block structure exactly."""
        hemi = hemisphere_assignments(self.result)
        asgn = hemi["assignments"][0]              # (20,)
        half = N_TOKENS // 2

        # One hemisphere is entirely block A, the other entirely block B
        label_A = asgn[0]
        label_B = asgn[half]
        self.assertNotEqual(label_A, label_B)
        self.assertTrue(np.all(asgn[:half]  == label_A), "Block A not uniform")
        self.assertTrue(np.all(asgn[half:]  == label_B), "Block B not uniform")

    def test_hemisphere_sizes_equal(self):
        """Both hemispheres have exactly n_tokens // 2 = 10 members."""
        hemi = hemisphere_assignments(self.result)
        sizes = hemi["sizes"][0]                   # [size_0, size_1]
        self.assertEqual(sizes[0], N_TOKENS // 2)
        self.assertEqual(sizes[1], N_TOKENS // 2)


# ═════════════════════════════════════════════════════════════════════════════
# 4. fiedler_tracking — stability across layers
# ═════════════════════════════════════════════════════════════════════════════

class TestFiedlerStabilityIdenticalLayers(unittest.TestCase):
    """
    track_fiedler_across_layers with identical inputs every layer:
      - alignment scores (cosines) all = 1.0
      - zero sign flips needed
      - zero crossing rates
    """

    def setUp(self):
        self.acts = _block_acts_multilayer()        # 6 identical layers
        self.fiedler = extract_fiedler_per_layer(self.acts)

    def test_all_layers_valid(self):
        self.assertTrue(
            self.fiedler["valid"].all(),
            "Every layer should produce a valid Fiedler vector"
        )

    def test_alignment_scores_all_one(self):
        """
        Identical layers → consecutive Fiedler vectors are parallel →
        cosine similarity = 1.0 at every transition.
        """
        stability = fiedler_stability(self.fiedler)
        cosines = stability["fiedler_cosine"]         # (n_layers - 1,)

        self.assertEqual(len(cosines), N_LAYERS - 1)
        for L, c in enumerate(cosines):
            with self.subTest(transition=f"L{L}→L{L+1}"):
                self.assertAlmostEqual(
                    float(c), 1.0, places=10,
                    msg=f"Expected cosine=1.0, got {c}"
                )

    def test_zero_sign_flips(self):
        """No sign flips: hemisphere assignments identical across all layers."""
        hemi = hemisphere_assignments(self.fiedler)
        for L in range(N_LAYERS - 1):
            with self.subTest(transition=f"L{L}→L{L+1}"):
                self.assertTrue(
                    np.array_equal(hemi["assignments"][L], hemi["assignments"][L + 1]),
                    f"Assignments changed between L{L} and L{L+1}"
                )

    def test_zero_crossing_rate(self):
        """No tokens switch hemispheres when all layers are identical."""
        hemi = hemisphere_assignments(self.fiedler)
        crossing = hemisphere_crossing_rate(
            hemi["assignments"], self.fiedler["valid"]
        )
        rates = crossing["crossing_rate"]
        for L, r in enumerate(rates):
            with self.subTest(transition=f"L{L}→L{L+1}"):
                self.assertEqual(float(r), 0.0)

    def test_fiedler_vectors_identical_across_layers(self):
        """All n_layers Fiedler vectors are exactly equal."""
        vecs = self.fiedler["fiedler_vecs"]          # (n_layers, n_tokens)
        for L in range(1, N_LAYERS):
            with self.subTest(layer=L):
                self.assertTrue(
                    np.allclose(vecs[0], vecs[L], atol=1e-12) or
                    np.allclose(vecs[0], -vecs[L], atol=1e-12),
                    f"Fiedler vector at layer {L} differs from layer 0 "
                    f"(allowing global sign flip)"
                )

    def test_fiedler_eigenvalues_identical_across_layers(self):
        """Same activations → same algebraic connectivity at every layer."""
        vals = self.fiedler["fiedler_vals"]
        for L in range(1, N_LAYERS):
            self.assertAlmostEqual(float(vals[L]), float(vals[0]), places=8)


# ═════════════════════════════════════════════════════════════════════════════
# 5. rotation_hemisphere — plane_fiedler_alignment geometry
# ═════════════════════════════════════════════════════════════════════════════

class TestRotationHemisphereAlignment(unittest.TestCase):
    """
    plane_fiedler_alignment: pure geometry checks.

    We construct a rotation plane that is either:
      (a) orthogonal to the Fiedler axis  → alignment ≈ 0   (within-hemisphere)
      (b) aligned with the Fiedler axis   → alignment ≈ 1   (across-hemisphere)
    """

    def _make_projectors_and_fiedler(self, plane_col0, plane_col1):
        """
        Wrap a manually constructed rotation plane in the dict shape that
        plane_fiedler_alignment expects.
        """
        plane = np.column_stack([plane_col0, plane_col1])  # (d, 2)
        P = plane @ plane.T
        projectors = {
            "top_k_planes":      [plane],
            "top_k_projectors":  [P],
            "top_k_rhos":        [1.0],
            "top_k_thetas":      [0.5],
            "combined_rotation": P,
            "real_subspace":     np.zeros((D, D)),
            "dim_rotation":      2,
            "dim_real":          0,
        }
        return projectors

    def _block_fiedler_data_and_acts(self):
        acts1 = _block_acts()[np.newaxis, :, :]       # (1, 20, 8)
        fd = extract_fiedler_per_layer(acts1)
        return fd, acts1

    def test_within_hemisphere_plane_low_alignment(self):
        """
        A rotation plane spanned by {e_2, e_3} is orthogonal to the Fiedler axis,
        which lies in the {e_0, e_1} subspace (the centroid-difference direction).
        Expected alignment ≈ 0.
        """
        fd, acts1 = self._block_fiedler_data_and_acts()

        # Plane spanned by basis vectors e_2 and e_3 — orthogonal to e_0/e_1
        u1 = np.zeros(D); u1[2] = 1.0
        u2 = np.zeros(D); u2[3] = 1.0
        projectors = self._make_projectors_and_fiedler(u1, u2)

        result = plane_fiedler_alignment(
            projectors, fd["fiedler_vecs"], fd["valid"], acts1
        )
        self.assertIsNotNone(result["per_layer"][0])
        alignment = result["per_layer"][0]["mean_alignment"]
        self.assertLess(alignment, 0.05,
                        f"Within-hemisphere plane: expected alignment<0.05, got {alignment}")

    def test_across_hemisphere_plane_high_alignment(self):
        """
        A rotation plane spanned by {e_0, e_1} is PARALLEL to the Fiedler axis.
        Expected alignment ≈ 1.
        """
        fd, acts1 = self._block_fiedler_data_and_acts()

        # Plane spanned by e_0 and e_1 — exactly the Fiedler centroid-difference direction
        u1 = np.zeros(D); u1[0] = 1.0
        u2 = np.zeros(D); u2[1] = 1.0
        projectors = self._make_projectors_and_fiedler(u1, u2)

        result = plane_fiedler_alignment(
            projectors, fd["fiedler_vecs"], fd["valid"], acts1
        )
        self.assertIsNotNone(result["per_layer"][0])
        alignment = result["per_layer"][0]["mean_alignment"]
        self.assertGreater(alignment, 0.95,
                           f"Across-hemisphere plane: expected alignment>0.95, got {alignment}")

    def test_frac_within_hemisphere_orthogonal_plane(self):
        """frac_within = 1.0 for a purely within-hemisphere plane (alignment < 0.1)."""
        fd, acts1 = self._block_fiedler_data_and_acts()
        u1 = np.zeros(D); u1[2] = 1.0
        u2 = np.zeros(D); u2[3] = 1.0
        projectors = self._make_projectors_and_fiedler(u1, u2)

        result = plane_fiedler_alignment(
            projectors, fd["fiedler_vecs"], fd["valid"], acts1
        )
        self.assertEqual(result["per_layer"][0]["frac_within"], 1.0)
        self.assertEqual(result["per_layer"][0]["frac_across"],  0.0)

    def test_frac_across_hemisphere_aligned_plane(self):
        """frac_across = 1.0 for a fully across-hemisphere plane (alignment > 0.3)."""
        fd, acts1 = self._block_fiedler_data_and_acts()
        u1 = np.zeros(D); u1[0] = 1.0
        u2 = np.zeros(D); u2[1] = 1.0
        projectors = self._make_projectors_and_fiedler(u1, u2)

        result = plane_fiedler_alignment(
            projectors, fd["fiedler_vecs"], fd["valid"], acts1
        )
        self.assertEqual(result["per_layer"][0]["frac_across"],  1.0)
        self.assertEqual(result["per_layer"][0]["frac_within"], 0.0)

    def test_overall_mean_alignment_consistent(self):
        """overall.mean_alignment aggregates per_layer correctly."""
        fd, acts1 = self._block_fiedler_data_and_acts()
        u1 = np.zeros(D); u1[0] = 1.0
        u2 = np.zeros(D); u2[1] = 1.0
        projectors = self._make_projectors_and_fiedler(u1, u2)

        result = plane_fiedler_alignment(
            projectors, fd["fiedler_vecs"], fd["valid"], acts1
        )
        # Only one valid layer; overall mean must equal that layer's value
        layer_val = result["per_layer"][0]["mean_alignment"]
        self.assertAlmostEqual(
            result["overall"]["mean_alignment"], layer_val, places=10
        )


# ═════════════════════════════════════════════════════════════════════════════
# 6. Additional robustness / edge-case tests
# ═════════════════════════════════════════════════════════════════════════════

class TestFiedlerRandomActivations(unittest.TestCase):
    """Sanity checks on random unit activations — no known ground truth,
    but output shapes, ranges, and types must be correct."""

    def setUp(self):
        self.acts = _unit_acts()
        self.fiedler = extract_fiedler_per_layer(self.acts)

    def test_all_layers_valid(self):
        self.assertTrue(self.fiedler["valid"].all())

    def test_output_shapes(self):
        self.assertEqual(self.fiedler["fiedler_vecs"].shape, (N_LAYERS, N_TOKENS))
        self.assertEqual(self.fiedler["fiedler_vals"].shape, (N_LAYERS,))

    def test_fiedler_vals_nonnegative(self):
        """Laplacian eigenvalues are non-negative."""
        vals = self.fiedler["fiedler_vals"]
        self.assertTrue(np.all(vals[np.isfinite(vals)] >= -1e-10))

    def test_hemisphere_sizes_sum_to_n_tokens(self):
        hemi = hemisphere_assignments(self.fiedler)
        for L in range(N_LAYERS):
            if self.fiedler["valid"][L]:
                self.assertEqual(hemi["sizes"][L].sum(), N_TOKENS)

    def test_cosines_in_zero_one(self):
        """Absolute cosines are in [0, 1]."""
        stability = fiedler_stability(self.fiedler)
        cosines = stability["fiedler_cosine"]
        valid = cosines[np.isfinite(cosines)]
        self.assertTrue(np.all(valid >= -1e-10))
        self.assertTrue(np.all(valid <= 1.0 + 1e-10))

    def test_crossing_rates_in_zero_one(self):
        hemi = hemisphere_assignments(self.fiedler)
        crossing = hemisphere_crossing_rate(
            hemi["assignments"], self.fiedler["valid"]
        )
        rates = crossing["crossing_rate"]
        valid = rates[np.isfinite(rates)]
        self.assertTrue(np.all(valid >= 0.0))
        self.assertTrue(np.all(valid <= 0.5 + 1e-10))  # sign-flip corrected


class TestSchurEdgeCases(unittest.TestCase):
    """Edge cases for Schur decomposition."""

    def test_scaling_matrix_all_real_blocks(self):
        """Diagonal scaling matrix (no rotation) → all 1×1 blocks."""
        diag_vals = np.array([2.0, -1.0, 0.5, -3.0, 1.5, -0.5, 0.1, -2.5])
        M = np.diag(diag_vals)
        result = extract_schur_blocks(M)
        self.assertEqual(result["n_complex"], 0)
        self.assertEqual(result["n_real"], D)
        recovered = sorted(b["value"] for b in result["blocks_1x1"])
        for r, e in zip(recovered, sorted(diag_vals)):
            self.assertAlmostEqual(r, e, places=8)

    def test_total_dimensions_always_d(self):
        """n_real + 2*n_complex == d for any input matrix."""
        rng = np.random.default_rng(99)
        for _ in range(5):
            M = rng.standard_normal((D, D))
            result = extract_schur_blocks(M)
            total = result["n_real"] + 2 * result["n_complex"]
            self.assertEqual(total, D)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
