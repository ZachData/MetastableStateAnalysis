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
import numpy.testing as npt

# ── allow running from project root or tests/ ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2_eigenspectra.subresult import SubResult

from p2b_imaginary.rotational_schur import (
    extract_schur_blocks,
    rotation_energy_fractions,
    rotation_angle_stats,
    henrici_nonnormality,
    build_rotation_plane_projectors,
)
from p2b_imaginary.rotational_rescaled import (
    decompose_symmetric_antisymmetric,
    rescaled_trajectory_component,
)
from p2b_imaginary.fiedler_tracking import (
    extract_fiedler_per_layer,
    hemisphere_assignments,
    hemisphere_crossing_rate,
    fiedler_stability,
)
from p2b_imaginary.rotation_hemisphere import (
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
    """Stack n_layers identical copies of _block_acts → (n_layers, n_tokens, d)."""
    return np.stack([_block_acts()] * n_layers, axis=0)


# ── 0. Schur block dimension invariant ───────────────────────────────────────

class TestSchurBlockDimensions(unittest.TestCase):
    """n_real + 2 * n_complex == d for any d×d real matrix."""

    def test_pure_rotation_all_complex(self):
        M = _rot_d8([0.3, 0.7, 1.1, 1.5])
        result = extract_schur_blocks(M)
        self.assertEqual(result["n_real"], 0)
        self.assertEqual(result["n_complex"], D // 2)
        self.assertEqual(result["n_real"] + 2 * result["n_complex"], D)

    def test_diagonal_all_real(self):
        M = np.diag([1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 2.5, -0.5])
        result = extract_schur_blocks(M)
        self.assertEqual(result["n_complex"], 0)
        self.assertEqual(result["n_real"], D)

    def test_random_matrices_dim_invariant(self):
        rng = np.random.default_rng(99)
        for _ in range(5):
            M = rng.standard_normal((D, D))
            result = extract_schur_blocks(M)
            total = result["n_real"] + 2 * result["n_complex"]
            self.assertEqual(total, D)


# ── 1. Angle recovery ────────────────────────────────────────────────────────

class TestRotationAngleRecovery(unittest.TestCase):
    """rotation_angle_stats recovers the exact input angle for a pure
    block-rotation matrix whose all four 2×2 blocks share the same θ."""

    ANGLES = [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2 - 0.01]

    def _mean_angle(self, theta):
        M = _rot_d8([theta] * 4)
        blocks = extract_schur_blocks(M)
        # FIX: was ["mean_angle"], current key is "theta_mean"
        return rotation_angle_stats(blocks)["theta_mean"]

    def test_mean_angle_pi_over_6(self):
        self.assertAlmostEqual(self._mean_angle(np.pi / 6), np.pi / 6, places=6)

    def test_mean_angle_pi_over_4(self):
        self.assertAlmostEqual(self._mean_angle(np.pi / 4), np.pi / 4, places=6)

    def test_mean_angle_pi_over_3(self):
        self.assertAlmostEqual(self._mean_angle(np.pi / 3), np.pi / 3, places=6)

    def test_std_angle_zero_for_uniform_blocks(self):
        """All blocks at the same angle → angle std ≈ 0."""
        M = _rot_d8([np.pi / 4] * 4)
        stats = rotation_angle_stats(extract_schur_blocks(M))
        # FIX: was ["std_angle"], current key is "theta_std"
        self.assertAlmostEqual(stats["theta_std"], 0.0, places=6)

    def test_heterogeneous_angles_span(self):
        """max_angle - min_angle matches the spread of input angles."""
        thetas = [0.2, 0.5, 0.9, 1.3]
        M = _rot_d8(thetas)
        stats = rotation_angle_stats(extract_schur_blocks(M))
        # FIX: was ["max_angle"] / ["min_angle"], current keys are "theta_max" / "theta_min"
        span = stats["theta_max"] - stats["theta_min"]
        self.assertGreater(span, 0.9)   # 1.3 - 0.2 = 1.1 > 0.9


# ── 2. Henrici nonnormality — zero for normal matrices ───────────────────────

class TestHenriciNormal(unittest.TestCase):
    """Pure rotation matrices are normal (R^T R = I), so Henrici = 0."""

    def test_pure_rotation_is_normal(self):
        M = _rot_d8([0.3, 0.7, 1.1, 1.5])
        # FIX: henrici_nonnormality now takes block_data dict, returns dict.
        # Was: henrici_nonnormality(M) returning float.
        h = henrici_nonnormality(extract_schur_blocks(M))
        self.assertAlmostEqual(h["henrici_absolute"], 0.0, places=6)

    def test_identity_is_normal(self):
        # FIX: same — pass blocks dict, access "henrici_absolute"
        h = henrici_nonnormality(extract_schur_blocks(np.eye(D)))
        self.assertAlmostEqual(h["henrici_absolute"], 0.0, places=8)

    def test_diagonal_is_normal(self):
        M = np.diag([1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 2.5, -0.5])
        h = henrici_nonnormality(extract_schur_blocks(M))
        self.assertAlmostEqual(h["henrici_absolute"], 0.0, places=8)

    def test_nonnormal_matrix_has_positive_henrici(self):
        """Strictly upper-triangular matrix is highly non-normal."""
        M = np.triu(np.ones((D, D)), k=1)
        h = henrici_nonnormality(extract_schur_blocks(M))
        self.assertGreater(h["henrici_absolute"], 0.0)


# ── 3. Rotation energy fractions — extremes ──────────────────────────────────

class TestRotationEnergyFractionsExtreme(unittest.TestCase):
    """rotational_fraction = 1.0 for pure rotation; 0.0 for diagonal."""

    def test_pure_rotation_frac_is_one(self):
        M = _rot_d8([0.4, 0.8, 1.2, 1.6])
        fracs = rotation_energy_fractions(extract_schur_blocks(M))
        # FIX: was ["rot_frac"] / ["real_frac"], current keys are
        # "rotational_fraction" / "signed_fraction"
        self.assertAlmostEqual(fracs["rotational_fraction"], 1.0, places=6)
        self.assertAlmostEqual(fracs["signed_fraction"], 0.0, places=6)

    def test_diagonal_frac_is_zero(self):
        M = np.diag([2.0, -1.0, 0.5, -3.0, 1.5, -0.5, 0.1, -2.5])
        fracs = rotation_energy_fractions(extract_schur_blocks(M))
        self.assertAlmostEqual(fracs["rotational_fraction"], 0.0, places=6)
        self.assertAlmostEqual(fracs["signed_fraction"], 1.0, places=6)

    def test_fracs_sum_to_one(self):
        """rotational_fraction + signed_fraction = 1.0 for any input matrix."""
        rng = np.random.default_rng(17)
        for _ in range(5):
            M = rng.standard_normal((D, D))
            fracs = rotation_energy_fractions(extract_schur_blocks(M))
            self.assertAlmostEqual(
                fracs["rotational_fraction"] + fracs["signed_fraction"], 1.0, places=8
            )


# ── 4. Rotation plane projectors — idempotency and symmetry ──────────────────

class TestRotationPlaneProjectors(unittest.TestCase):
    """Each projector P returned by build_rotation_plane_projectors must
    satisfy P² = P (idempotent) and P = P^T (orthogonal projector)."""

    def _projectors(self, k=2):
        M = _rot_d8([0.3, 0.7, 1.1, 1.5])
        blocks = extract_schur_blocks(M)
        # FIX: was build_rotation_plane_projectors(blocks, k=k) returning a list.
        # Current API: keyword is "top_k", returns a dict; extract the projector list.
        return build_rotation_plane_projectors(blocks, top_k=k)["top_k_projectors"]

    def test_each_projector_is_idempotent(self):
        for i, P in enumerate(self._projectors(k=2)):
            npt.assert_allclose(P @ P, P, atol=1e-10,
                                err_msg=f"Projector {i} not idempotent")

    def test_each_projector_is_symmetric(self):
        for i, P in enumerate(self._projectors(k=2)):
            npt.assert_allclose(P, P.T, atol=1e-10,
                                err_msg=f"Projector {i} not symmetric")

    def test_projector_rank_is_two(self):
        """Each rotation plane is 2-dimensional."""
        for i, P in enumerate(self._projectors(k=2)):
            rank = int(np.round(np.trace(P)))
            self.assertEqual(rank, 2, msg=f"Projector {i} has rank {rank}, expected 2")

    def test_projectors_are_pairwise_orthogonal(self):
        """Distinct rotation planes in a block-diagonal matrix are orthogonal."""
        projs = self._projectors(k=4)   # request all 4 planes
        for i in range(len(projs)):
            for j in range(i + 1, len(projs)):
                cross = projs[i] @ projs[j]
                npt.assert_allclose(cross, np.zeros((D, D)), atol=1e-10,
                                    err_msg=f"Projectors {i},{j} not orthogonal")


# ── 5. Fiedler stability — identical consecutive layers → cosine = 1 ─────────

class TestFiedlerStabilityIdentical(unittest.TestCase):
    """When all layers are identical, consecutive Fiedler vectors are
    (up to sign) the same, so the absolute cosine similarity = 1.0."""

    def setUp(self):
        self.acts = _block_acts_multilayer(N_LAYERS)
        self.fd = extract_fiedler_per_layer(self.acts)
        self.stab = fiedler_stability(self.fd)

    def test_cosines_all_one(self):
        cosines = self.stab["fiedler_cosine"]
        valid = cosines[np.isfinite(cosines)]
        npt.assert_allclose(valid, np.ones_like(valid), atol=1e-8)

    def test_length_is_n_layers_minus_one(self):
        self.assertEqual(len(self.stab["fiedler_cosine"]), N_LAYERS - 1)


# ── 6. Crossing rate — zero for perfectly stable block activations ────────────

class TestCrossingRateStable(unittest.TestCase):
    """Tokens that never move between hemispheres produce crossing_rate = 0."""

    def setUp(self):
        acts = _block_acts_multilayer(N_LAYERS)
        fd = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fd)
        crossing = hemisphere_crossing_rate(hemi["assignments"], fd["valid"])
        self.rates = crossing["crossing_rate"]

    def test_all_crossing_rates_zero(self):
        valid = self.rates[np.isfinite(self.rates)]
        npt.assert_allclose(valid, 0.0, atol=1e-10)


# ── 7. Block-structured Fiedler — perfect bipartition ────────────────────────

class TestBlockFiedlerBipartition(unittest.TestCase):
    """_block_acts forms two perfectly separated clusters.
    The Fiedler vector must assign them to opposite hemispheres."""

    def setUp(self):
        acts = _block_acts_multilayer(N_LAYERS)
        self.fd = extract_fiedler_per_layer(acts)
        self.hemi = hemisphere_assignments(self.fd)

    def test_fiedler_val_near_zero(self):
        vals = self.fd["fiedler_vals"]
        self.assertTrue(np.all(vals[self.fd["valid"]] >= -1e-8))

    def test_hemisphere_sizes_are_equal(self):
        """Both hemispheres should contain exactly N_TOKENS // 2 tokens."""
        sizes = self.hemi["sizes"]
        for L in range(N_LAYERS):
            if self.fd["valid"][L]:
                self.assertEqual(sizes[L, 0], N_TOKENS // 2)
                self.assertEqual(sizes[L, 1], N_TOKENS // 2)

    def test_hemisphere_assignment_separates_clusters(self):
        hemi = self.hemi
        asgn = hemi["assignments"][0]
        half = N_TOKENS // 2
        label_A = asgn[0]
        label_B = asgn[half]
        self.assertNotEqual(label_A, label_B)
        self.assertTrue(np.all(asgn[:half]  == label_A), "Block A not uniform")
        self.assertTrue(np.all(asgn[half:]  == label_B), "Block B not uniform")

    def test_hemisphere_sizes_equal(self):
        hemi = hemisphere_assignments(self.fd)
        sizes = hemi["sizes"][0]
        self.assertEqual(sizes[0], N_TOKENS // 2)
        self.assertEqual(sizes[1], N_TOKENS // 2)


# ── 8. Fiedler tracking — detailed stability across identical layers ──────────

class TestFiedlerStabilityIdenticalLayers(unittest.TestCase):
    """
    Detailed checks: identical activations at every layer means the
    Fiedler vector, hemisphere assignments, and crossing rates are all
    constant across depth.
    """

    def setUp(self):
        self.acts   = _block_acts_multilayer()
        self.fiedler = extract_fiedler_per_layer(self.acts)

    def test_all_layers_valid(self):
        self.assertTrue(
            self.fiedler["valid"].all(),
            "Every layer should produce a valid Fiedler vector"
        )

    def test_alignment_scores_all_one(self):
        """Identical layers → consecutive Fiedler vectors are parallel →
        cosine similarity = 1.0 at every transition."""
        stability = fiedler_stability(self.fiedler)
        cosines = stability["fiedler_cosine"]
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
        crossing = hemisphere_crossing_rate(hemi["assignments"], self.fiedler["valid"])
        rates = crossing["crossing_rate"]
        for L, r in enumerate(rates):
            with self.subTest(transition=f"L{L}→L{L+1}"):
                self.assertEqual(float(r), 0.0)

    def test_fiedler_vectors_identical_across_layers(self):
        """All n_layers Fiedler vectors are exactly equal (up to global sign)."""
        vecs = self.fiedler["fiedler_vecs"]
        for L in range(1, N_LAYERS):
            with self.subTest(layer=L):
                self.assertTrue(
                    np.allclose(vecs[0], vecs[L], atol=1e-12) or
                    np.allclose(vecs[0], -vecs[L], atol=1e-12),
                    f"Fiedler vector at layer {L} differs from layer 0"
                )

    def test_fiedler_eigenvalues_identical_across_layers(self):
        """Same activations → same algebraic connectivity at every layer."""
        vals = self.fiedler["fiedler_vals"]
        for L in range(1, N_LAYERS):
            self.assertAlmostEqual(float(vals[L]), float(vals[0]), places=8)


# ── 9. rotation_hemisphere — plane_fiedler_alignment geometry ────────────────

class TestRotationHemisphereAlignment(unittest.TestCase):
    """
    plane_fiedler_alignment: pure geometry checks.

    We construct a rotation plane that is either:
      (a) orthogonal to the Fiedler axis  → alignment ≈ 0   (within-hemisphere)
      (b) aligned with the Fiedler axis   → alignment ≈ 1   (across-hemisphere)
    """

    def _make_projectors(self, plane_col0, plane_col1):
        """Wrap manually constructed rotation plane in the dict shape that
        plane_fiedler_alignment expects."""
        plane = np.column_stack([plane_col0, plane_col1])  # (d, 2)
        P = plane @ plane.T
        return {
            "top_k_planes":      [plane],
            "top_k_projectors":  [P],
            "top_k_rhos":        [1.0],
            "top_k_thetas":      [0.5],
            "combined_rotation": P,
            "real_subspace":     np.zeros((D, D)),
            "dim_rotation":      2,
            "dim_real":          0,
        }

    def _block_fiedler_data_and_acts(self):
        acts1 = _block_acts()[np.newaxis, :, :]   # (1, 20, 8)
        fd = extract_fiedler_per_layer(acts1)
        return fd, acts1

    def test_within_hemisphere_plane_low_alignment(self):
        """Plane spanned by {e_2, e_3} is orthogonal to the Fiedler axis
        (which lies in {e_0, e_1}). Expected alignment ≈ 0."""
        fd, acts1 = self._block_fiedler_data_and_acts()
        u1 = np.zeros(D); u1[2] = 1.0
        u2 = np.zeros(D); u2[3] = 1.0
        result = plane_fiedler_alignment(
            self._make_projectors(u1, u2),
            fd["fiedler_vecs"], fd["valid"], acts1,
        )
        alignment = result["per_layer"][0]["mean_alignment"]
        self.assertLess(alignment, 0.05,
                        f"Within-hemisphere plane: expected alignment<0.05, got {alignment}")

    def test_across_hemisphere_plane_high_alignment(self):
        """Plane spanned by {e_0, e_1} is parallel to the Fiedler axis.
        Expected alignment ≈ 1."""
        fd, acts1 = self._block_fiedler_data_and_acts()
        u1 = np.zeros(D); u1[0] = 1.0
        u2 = np.zeros(D); u2[1] = 1.0
        result = plane_fiedler_alignment(
            self._make_projectors(u1, u2),
            fd["fiedler_vecs"], fd["valid"], acts1,
        )
        alignment = result["per_layer"][0]["mean_alignment"]
        self.assertGreater(alignment, 0.95,
                           f"Across-hemisphere plane: expected alignment>0.95, got {alignment}")

    def test_frac_within_hemisphere_orthogonal_plane(self):
        """frac_within = 1.0 for a purely within-hemisphere plane (alignment < 0.1)."""
        fd, acts1 = self._block_fiedler_data_and_acts()
        u1 = np.zeros(D); u1[2] = 1.0
        u2 = np.zeros(D); u2[3] = 1.0
        result = plane_fiedler_alignment(
            self._make_projectors(u1, u2),
            fd["fiedler_vecs"], fd["valid"], acts1,
        )
        self.assertEqual(result["per_layer"][0]["frac_within"], 1.0)
        self.assertEqual(result["per_layer"][0]["frac_across"],  0.0)

    def test_frac_across_hemisphere_aligned_plane(self):
        """frac_across = 1.0 for a fully across-hemisphere plane (alignment > 0.3)."""
        fd, acts1 = self._block_fiedler_data_and_acts()
        u1 = np.zeros(D); u1[0] = 1.0
        u2 = np.zeros(D); u2[1] = 1.0
        result = plane_fiedler_alignment(
            self._make_projectors(u1, u2),
            fd["fiedler_vecs"], fd["valid"], acts1,
        )
        self.assertEqual(result["per_layer"][0]["frac_across"],  1.0)
        self.assertEqual(result["per_layer"][0]["frac_within"], 0.0)

    def test_overall_mean_alignment_consistent(self):
        """overall["mean_alignment"] aggregates per_layer correctly."""
        fd, acts1 = self._block_fiedler_data_and_acts()
        u1 = np.zeros(D); u1[0] = 1.0
        u2 = np.zeros(D); u2[1] = 1.0
        result = plane_fiedler_alignment(
            self._make_projectors(u1, u2),
            fd["fiedler_vecs"], fd["valid"], acts1,
        )
        layer_val = result["per_layer"][0]["mean_alignment"]
        self.assertAlmostEqual(
            result["overall"]["mean_alignment"], layer_val, places=10
        )


# ── 10. Additional robustness / edge-case tests ───────────────────────────────

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


if __name__ == "__main__":
    unittest.main()
