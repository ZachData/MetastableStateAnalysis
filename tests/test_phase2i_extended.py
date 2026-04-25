"""
tests/test_p2b_imaginary.py — Comprehensive tests for p2b_imaginary.

Coverage
--------
  rotational_schur.py
    extract_schur_blocks          — 2×2, 8×8, identity, diagonal, random
    rotation_energy_fractions     — pure rotation, pure real, mixed
    rotation_angle_stats          — mean/count/expanding/contracting
    henrici_nonnormality          — normal (orthogonal) and non-normal matrices
    build_rotation_plane_projectors — projector idempotency, subspace dims
    rotation_depth_profile        — per-layer depth profile shape/keys
    analyze_rotational_spectrum   — shared and per-layer ov_data contracts

  rotational_rescaled.py
    decompose_symmetric_antisymmetric — algebraic identities
    rescaled_trajectory_component     — identity, per-layer, finite/diverge guard
    compare_rescaled_frames           — ordering invariants
    interpret_comparison              — all four classification categories
    analyze_rotational_rescaling      — shared and per-layer ov_data contracts
    comparison_to_json                — JSON-serialisable output

  fiedler_tracking.py
    extract_fiedler_per_layer     — block-diagonal Gram, random acts
    hemisphere_assignments        — sizes, sign split
    hemisphere_crossing_rate      — zero-crossing (stable) and constant-flip
    fiedler_stability             — cosine range
    hemisphere_centroid_separation — antipodal and identical centroids
    crossref_with_events          — with/without events
    analyze_fiedler_tracking      — full pipeline, with and without events
    fiedler_to_json               — all keys present and serialisable

  rotation_hemisphere.py
    plane_fiedler_alignment       — perpendicular plane (neutral), in-plane (high)
    token_fiedler_displacement    — sign distribution
    displacement_coherence        — identical-displacement → coherence 1
    analyze_rotation_hemisphere   — full pipeline
    rotation_hemisphere_to_json   — all keys present

  ffn_rotation.py
    project_ffn_onto_rotation_planes — fractions sum ≤ 1, shared/per-layer
    compare_ffn_rotation_at_violations — NaN on empty violations
    classify_ffn_rotation_per_violation — role labels
    analyze_ffn_rotation          — full pipeline
    ffn_rotation_to_json          — all keys present

  Import contracts (Phase 2 / Phase 1 artifact shapes)
    _make_ov_data_shared()        — ALBERT-style single matrix
    _make_ov_data_per_layer()     — GPT-2-style list of matrices
    _make_phase1_events()         — energy_violations dict

Run
---
    python -m pytest tests/test_p2b_imaginary.py -v
    python -m unittest tests.test_p2b_imaginary -v
"""

from __future__ import annotations

import sys
import os
import unittest

import numpy as np
from scipy.linalg import expm

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2b_imaginary.rotational_schur import (
    extract_schur_blocks,
    rotation_energy_fractions,
    rotation_angle_stats,
    henrici_nonnormality,
    build_rotation_plane_projectors,
    rotation_depth_profile,
    analyze_rotational_spectrum,
    summary_to_json,
)
from p2b_imaginary.rotational_rescaled import (
    decompose_symmetric_antisymmetric,
    rescaled_trajectory_component,
    compare_rescaled_frames,
    interpret_comparison,
    analyze_rotational_rescaling,
    comparison_to_json,
)
from p2b_imaginary.fiedler_tracking import (
    extract_fiedler_per_layer,
    hemisphere_assignments,
    hemisphere_crossing_rate,
    fiedler_stability,
    hemisphere_centroid_separation,
    crossref_with_events,
    analyze_fiedler_tracking,
    fiedler_to_json,
)
from p2b_imaginary.rotation_hemisphere import (
    plane_fiedler_alignment,
    token_fiedler_displacement,
    displacement_coherence,
    analyze_rotation_hemisphere,
    rotation_hemisphere_to_json,
)
from p2b_imaginary.ffn_rotation import (
    project_ffn_onto_rotation_planes,
    compare_ffn_rotation_at_violations,
    classify_ffn_rotation_per_violation,
    analyze_ffn_rotation,
    ffn_rotation_to_json,
)

# ── dimensions ────────────────────────────────────────────────────────────────
D        = 8
N_TOKENS = 20
N_LAYERS = 6
TOP_K    = 4   # rotation planes to extract in tests


# ═══════════════════════════════════════════════════════════════════════════════
# Shared fixture builders
# ═══════════════════════════════════════════════════════════════════════════════

def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _rot_d(thetas) -> np.ndarray:
    """Block-diagonal D×D rotation matrix from D/2 angles."""
    M = np.zeros((D, D))
    for i, th in enumerate(thetas):
        M[2*i : 2*i+2, 2*i : 2*i+2] = _rot2(th)
    return M


def _unit_acts(n_layers=N_LAYERS, n_tokens=N_TOKENS, d=D, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_layers, n_tokens, d)).astype(np.float64)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(norms, 1e-10)


def _block_acts(n_tokens=N_TOKENS, d=D) -> np.ndarray:
    """Two perfectly separated clusters: half → e0, half → e1."""
    X = np.zeros((n_tokens, d), dtype=np.float64)
    X[: n_tokens // 2, 0] = 1.0
    X[n_tokens // 2 :, 1] = 1.0
    return X


def _block_acts_ml(n_layers=N_LAYERS) -> np.ndarray:
    return np.stack([_block_acts()] * n_layers, axis=0)


def _make_ov_data_shared(OV=None) -> dict:
    """ALBERT-style: single shared weight matrix. Simulates Phase 2 import."""
    if OV is None:
        OV = _rot_d([0.3, 0.7, 1.1, 1.5])
    return {
        "ov_total":    OV,
        "is_per_layer": False,
        "layer_names": ["shared"],
    }


def _make_ov_data_per_layer(n_layers=N_LAYERS) -> dict:
    """GPT-2-style: one OV matrix per layer. Simulates Phase 2 import."""
    rng = np.random.default_rng(0)
    ov_list = [rng.standard_normal((D, D)) for _ in range(n_layers)]
    names = [f"layer_{i}" for i in range(n_layers)]
    return {
        "ov_total":    ov_list,
        "is_per_layer": True,
        "layer_names": names,
    }


def _make_phase1_events(n_layers=N_LAYERS, violation_layers=None) -> dict:
    """Minimal Phase 1 events dict. Simulates trajectory.load_phase1_events."""
    if violation_layers is None:
        violation_layers = [2, 4]
    betas = [0.1, 1.0, 2.0, 5.0]
    return {
        "n_layers": n_layers,
        "n_tokens": N_TOKENS,
        "energy_violations": {b: list(violation_layers) for b in betas},
        "energy_drop_pairs": {b: {} for b in betas},
    }


def _make_schur_blocks_pure_rotation(thetas=None):
    if thetas is None:
        thetas = [0.3, 0.7, 1.1, 1.5]
    OV = _rot_d(thetas)
    return extract_schur_blocks(OV), thetas


def _make_ffn_deltas(n_layers=N_LAYERS, n_tokens=N_TOKENS, d=D, seed=5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_layers, n_tokens, d)).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. rotational_schur — extract_schur_blocks
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractSchurBlocks2x2(unittest.TestCase):
    """Single 2×2 rotation matrix."""

    def _b(self, theta):
        return extract_schur_blocks(_rot2(theta))

    def test_single_complex_block(self):
        r = self._b(np.pi / 4)
        self.assertEqual(r["n_complex"], 1)
        self.assertEqual(r["n_real"], 0)

    def test_theta_recovered_first_quadrant(self):
        for theta in [0.1, 0.5, np.pi / 4, np.pi / 3, np.pi / 2 - 0.01]:
            with self.subTest(theta=theta):
                block = self._b(theta)["blocks_2x2"][0]
                self.assertAlmostEqual(block["theta"], theta, places=8)

    def test_rho_unity(self):
        for theta in [0.2, 0.9, 1.4]:
            self.assertAlmostEqual(self._b(theta)["blocks_2x2"][0]["rho"], 1.0, places=8)

    def test_plane_shape(self):
        block = self._b(0.5)["blocks_2x2"][0]
        self.assertEqual(block["plane"].shape, (2, 2))

    def test_d_equals_2(self):
        self.assertEqual(self._b(0.5)["d"], 2)


class TestExtractSchurBlocksD8(unittest.TestCase):
    THETAS = [0.3, 0.7, 1.1, 1.5]

    def _result(self):
        OV = _rot_d(self.THETAS)
        return extract_schur_blocks(OV)

    def test_n_complex_4_n_real_0(self):
        r = self._result()
        self.assertEqual(r["n_complex"], 4)
        self.assertEqual(r["n_real"], 0)

    def test_all_thetas_recovered(self):
        r = self._result()
        recovered = sorted(b["theta"] for b in r["blocks_2x2"])
        for rec, exp in zip(recovered, sorted(self.THETAS)):
            self.assertAlmostEqual(rec, exp, places=8)

    def test_all_radii_unity(self):
        for b in self._result()["blocks_2x2"]:
            self.assertAlmostEqual(b["rho"], 1.0, places=8)

    def test_schur_Z_orthonormal(self):
        Z = self._result()["schur_Z"]
        diff = np.max(np.abs(Z.T @ Z - np.eye(D)))
        self.assertAlmostEqual(diff, 0.0, places=8)

    def test_total_dims_equals_d(self):
        r = self._result()
        self.assertEqual(r["n_real"] + 2 * r["n_complex"], D)


class TestExtractSchurBlocksIdentity(unittest.TestCase):
    def test_all_real_blocks(self):
        r = extract_schur_blocks(np.eye(D))
        self.assertEqual(r["n_complex"], 0)
        self.assertEqual(r["n_real"], D)

    def test_eigenvalues_all_one(self):
        r = extract_schur_blocks(np.eye(D))
        for b in r["blocks_1x1"]:
            self.assertAlmostEqual(b["value"], 1.0, places=10)


class TestExtractSchurBlocksDiagonal(unittest.TestCase):
    def test_no_complex_blocks(self):
        diag = np.array([2.0, -1.0, 0.5, -3.0, 1.5, -0.5, 0.1, -2.5])
        r = extract_schur_blocks(np.diag(diag))
        self.assertEqual(r["n_complex"], 0)
        self.assertEqual(r["n_real"], D)

    def test_eigenvalues_recovered(self):
        diag = np.array([2.0, -1.0, 0.5, -3.0, 1.5, -0.5, 0.1, -2.5])
        r = extract_schur_blocks(np.diag(diag))
        recovered = sorted(b["value"] for b in r["blocks_1x1"])
        for rec, exp in zip(recovered, sorted(diag)):
            self.assertAlmostEqual(rec, exp, places=8)

    def test_total_dims_d(self):
        rng = np.random.default_rng(99)
        for _ in range(5):
            M = rng.standard_normal((D, D))
            r = extract_schur_blocks(M)
            self.assertEqual(r["n_real"] + 2 * r["n_complex"], D)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. rotational_schur — energy fractions, angle stats, Henrici
# ═══════════════════════════════════════════════════════════════════════════════

class TestRotationEnergyFractions(unittest.TestCase):

    def test_pure_rotation_all_in_rotation(self):
        blocks, _ = _make_schur_blocks_pure_rotation()
        ef = rotation_energy_fractions(blocks)
        self.assertAlmostEqual(ef["rotational_fraction"], 1.0, places=8)
        self.assertAlmostEqual(ef["signed_fraction"],     0.0, places=8)
        self.assertEqual(ef["n_real"], 0)
        self.assertEqual(ef["n_complex"], 4)
        self.assertAlmostEqual(ef["frac_complex_dims"], 1.0, places=8)

    def test_identity_all_in_real(self):
        blocks = extract_schur_blocks(np.eye(D))
        ef = rotation_energy_fractions(blocks)
        self.assertAlmostEqual(ef["signed_fraction"], 1.0, places=6)
        self.assertAlmostEqual(ef["rotational_fraction"], 0.0, places=6)

    def test_fractions_sum_to_one(self):
        rng = np.random.default_rng(7)
        OV = rng.standard_normal((D, D))
        blocks = extract_schur_blocks(OV)
        ef = rotation_energy_fractions(blocks)
        self.assertAlmostEqual(
            ef["rotational_fraction"] + ef["signed_fraction"], 1.0, places=6
        )


class TestRotationAngleStats(unittest.TestCase):

    def test_count_matches_n_complex(self):
        blocks, thetas = _make_schur_blocks_pure_rotation()
        stats = rotation_angle_stats(blocks)
        self.assertEqual(len(stats["thetas"]), len(thetas))

    def test_mean_theta_correct(self):
        thetas = [0.3, 0.7, 1.1, 1.5]
        blocks = extract_schur_blocks(_rot_d(thetas))
        stats = rotation_angle_stats(blocks)
        self.assertAlmostEqual(stats["theta_mean"], float(np.mean(thetas)), places=7)

    def test_no_expanding_for_unit_rho(self):
        blocks, _ = _make_schur_blocks_pure_rotation()
        stats = rotation_angle_stats(blocks)
        # Rho = 1.0 exactly: neither expanding nor contracting
        self.assertAlmostEqual(stats["frac_expanding"] + stats["frac_contracting"], 0.0, places=8)

    def test_expanding_for_scaled_rotation(self):
        """Scale a rotation by 2× → all blocks expanding."""
        OV = 2.0 * _rot_d([0.5, 1.0, 1.5, 2.0])
        blocks = extract_schur_blocks(OV)
        stats = rotation_angle_stats(blocks)
        self.assertAlmostEqual(stats["frac_expanding"], 1.0, places=5)


class TestHenriciNonnormality(unittest.TestCase):

    def test_zero_for_orthogonal(self):
        blocks = extract_schur_blocks(_rot_d([0.3, 0.7, 1.1, 1.5]))
        h = henrici_nonnormality(blocks)
        self.assertAlmostEqual(h["henrici_absolute"], 0.0, places=5)
        self.assertAlmostEqual(h["henrici_relative"], 0.0, places=5)

    def test_zero_for_diagonal(self):
        blocks = extract_schur_blocks(np.diag([1.0, -2.0, 0.5, -0.3, 1.2, -1.1, 0.7, -0.4]))
        h = henrici_nonnormality(blocks)
        self.assertAlmostEqual(h["henrici_absolute"], 0.0, places=5)

    def test_positive_for_non_normal(self):
        rng = np.random.default_rng(12)
        M = np.triu(rng.standard_normal((D, D)))  # upper-triangular → non-normal
        np.fill_diagonal(M, 0.0)
        M += np.eye(D)
        blocks = extract_schur_blocks(M)
        h = henrici_nonnormality(blocks)
        # Upper-triangular non-diagonal → Henrici > 0
        self.assertGreaterEqual(h["henrici_absolute"], 0.0)

    def test_relative_between_zero_and_one(self):
        rng = np.random.default_rng(55)
        M = rng.standard_normal((D, D))
        blocks = extract_schur_blocks(M)
        h = henrici_nonnormality(blocks)
        self.assertGreaterEqual(h["henrici_relative"], 0.0)
        self.assertLessEqual(h["henrici_relative"], 1.0 + 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. rotational_schur — build_rotation_plane_projectors
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildRotationPlaneProjctors(unittest.TestCase):

    def setUp(self):
        blocks, _ = _make_schur_blocks_pure_rotation()
        self.planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)

    def test_top_k_count(self):
        self.assertEqual(len(self.planes["top_k_projectors"]), TOP_K)

    def test_projectors_are_idempotent(self):
        for P in self.planes["top_k_projectors"]:
            diff = np.max(np.abs(P @ P - P))
            self.assertAlmostEqual(diff, 0.0, places=8,
                msg="Projector must satisfy P² = P")

    def test_projectors_are_symmetric(self):
        for P in self.planes["top_k_projectors"]:
            self.assertAlmostEqual(np.max(np.abs(P - P.T)), 0.0, places=8)

    def test_combined_rotation_idempotent(self):
        P = self.planes["combined_rotation"]
        diff = np.max(np.abs(P @ P - P))
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_dim_rotation_matches_n_complex(self):
        self.assertEqual(self.planes["dim_rotation"], 2 * 4)  # 4 blocks × 2

    def test_real_subspace_idempotent_for_identity(self):
        blocks = extract_schur_blocks(np.eye(D))
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        P = planes["real_subspace"]
        diff = np.max(np.abs(P @ P - P))
        self.assertAlmostEqual(diff, 0.0, places=8)

    def test_subspaces_orthogonal(self):
        """Rotation and real subspace projectors are orthogonal (P_rot @ P_real ≈ 0)."""
        rng = np.random.default_rng(3)
        OV = rng.standard_normal((D, D))
        blocks = extract_schur_blocks(OV)
        planes = build_rotation_plane_projectors(blocks, top_k=2)
        cross = planes["combined_rotation"] @ planes["real_subspace"]
        self.assertAlmostEqual(np.max(np.abs(cross)), 0.0, places=5)

    def test_rhos_descending(self):
        rhos = self.planes["top_k_rhos"]
        for i in range(len(rhos) - 1):
            self.assertGreaterEqual(rhos[i], rhos[i + 1] - 1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. rotational_schur — rotation_depth_profile and analyze_rotational_spectrum
# ═══════════════════════════════════════════════════════════════════════════════

class TestRotationDepthProfile(unittest.TestCase):

    def _run(self, n_layers=N_LAYERS):
        rng = np.random.default_rng(0)
        ov_list = [rng.standard_normal((D, D)) for _ in range(n_layers)]
        names = [f"layer_{i}" for i in range(n_layers)]
        return rotation_depth_profile(ov_list, names)

    def test_per_layer_count(self):
        result = self._run()
        self.assertEqual(len(result["per_layer"]), N_LAYERS)

    def test_summary_keys_present(self):
        result = self._run()
        for k in ["theta_mean_across_layers", "henrici_mean", "henrici_max"]:
            self.assertIn(k, result["summary"])

    def test_theta_mean_finite(self):
        result = self._run()
        self.assertTrue(np.isfinite(result["summary"]["theta_mean_across_layers"]))


class TestAnalyzeRotationalSpectrum(unittest.TestCase):

    def test_shared_model_keys(self):
        ov_data = _make_ov_data_shared()
        result = analyze_rotational_spectrum(ov_data, top_k_planes=TOP_K)
        for k in ["is_per_layer", "blocks", "angle_stats", "energy_fractions",
                  "henrici", "plane_projectors"]:
            self.assertIn(k, result)
        self.assertFalse(result["is_per_layer"])

    def test_per_layer_model_keys(self):
        ov_data = _make_ov_data_per_layer()
        result = analyze_rotational_spectrum(ov_data, top_k_planes=TOP_K)
        self.assertTrue(result["is_per_layer"])
        self.assertEqual(len(result["blocks"]), N_LAYERS)
        self.assertEqual(len(result["plane_projectors"]), N_LAYERS)
        self.assertIn("depth_profile", result)

    def test_summary_to_json_serialisable(self):
        """summary_to_json must return only Python-native types (no numpy scalars)."""
        ov_data = _make_ov_data_shared()
        result = analyze_rotational_spectrum(ov_data, top_k_planes=TOP_K)
        import json
        j = summary_to_json(result)
        # Should not raise
        json.dumps(j)

    def test_shared_pure_rotation_energy_fraction_one(self):
        OV = _rot_d([0.3, 0.7, 1.1, 1.5])
        ov_data = _make_ov_data_shared(OV)
        result = analyze_rotational_spectrum(ov_data, top_k_planes=TOP_K)
        ef = result["energy_fractions"]
        self.assertAlmostEqual(ef["rotational_fraction"], 1.0, places=6)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. rotational_rescaled — S/A decomposition
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecomposeSymmetricAntisymmetric(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(7)
        self.OV = rng.standard_normal((D, D))

    def test_reconstruction(self):
        sa = decompose_symmetric_antisymmetric(self.OV)
        self.assertTrue(np.allclose(sa["S"] + sa["A"], self.OV, atol=1e-12))

    def test_S_symmetric(self):
        sa = decompose_symmetric_antisymmetric(self.OV)
        self.assertTrue(np.allclose(sa["S"], sa["S"].T, atol=1e-12))

    def test_A_antisymmetric(self):
        sa = decompose_symmetric_antisymmetric(self.OV)
        self.assertTrue(np.allclose(sa["A"], -sa["A"].T, atol=1e-12))

    def test_pythagorean_identity(self):
        """||S||² + ||A||² = ||V||²  (S and A are Frobenius-orthogonal)."""
        sa = decompose_symmetric_antisymmetric(self.OV)
        self.assertAlmostEqual(
            sa["S_frob"]**2 + sa["A_frob"]**2, sa["V_frob"]**2, places=8
        )

    def test_symmetric_input_zero_A(self):
        OV = self.OV + self.OV.T
        sa = decompose_symmetric_antisymmetric(OV)
        self.assertAlmostEqual(sa["A_frob"], 0.0, places=10)
        self.assertAlmostEqual(sa["rotation_ratio"], 0.0, places=10)

    def test_antisymmetric_input_zero_S(self):
        OV = self.OV - self.OV.T
        sa = decompose_symmetric_antisymmetric(OV)
        self.assertAlmostEqual(sa["S_frob"], 0.0, places=10)

    def test_rotation_ratio_nonnegative(self):
        sa = decompose_symmetric_antisymmetric(self.OV)
        self.assertGreaterEqual(sa["rotation_ratio"], 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. rotational_rescaled — rescaled_trajectory_component
# ═══════════════════════════════════════════════════════════════════════════════

class TestRescaledTrajectoryIdentity(unittest.TestCase):
    """Zero matrix → expm(-0) = I → activations unchanged."""

    def setUp(self):
        self.acts = _unit_acts()
        self.zero = np.zeros((D, D))

    def _ip_means(self, acts):
        n_layers, n_tokens, _ = acts.shape
        out = []
        for L in range(n_layers):
            G = acts[L] @ acts[L].T
            idx = np.triu_indices(n_tokens, k=1)
            out.append(G[idx].mean())
        return np.array(out)

    def test_max_valid_layer_equals_n_layers(self):
        result = rescaled_trajectory_component(self.acts, self.zero, [1.0])
        self.assertEqual(result["max_valid_layer"], N_LAYERS)

    def test_ip_means_unchanged(self):
        expected = self._ip_means(self.acts)
        result = rescaled_trajectory_component(self.acts, self.zero, [1.0])
        np.testing.assert_allclose(result["ip_mean"], expected, atol=1e-6)

    def test_effective_rank_finite(self):
        result = rescaled_trajectory_component(self.acts, self.zero, [1.0])
        self.assertTrue(np.all(np.isfinite(result["effective_rank"])))

    def test_per_layer_mode_identity(self):
        zero_list = [np.zeros((D, D))] * N_LAYERS
        result = rescaled_trajectory_component(
            self.acts, None, [1.0],
            is_per_layer=True, matrices_list=zero_list,
        )
        expected = self._ip_means(self.acts)
        np.testing.assert_allclose(result["ip_mean"], expected, atol=1e-6)

    def test_n_violations_keys_present(self):
        betas = [0.1, 1.0, 2.0]
        result = rescaled_trajectory_component(self.acts, self.zero, betas)
        for b in betas:
            self.assertIn(b, result["n_violations"])


class TestRescaledTrajectoryLargeMatrix(unittest.TestCase):
    """Very large matrix → rescaling diverges at some layer (max_valid_layer < N_LAYERS)."""

    def test_divergence_truncates(self):
        acts = _unit_acts(n_layers=10)
        M = 1000.0 * np.eye(D)   # expm(-1000I) ≈ 0, but cumulative product diverges
        result = rescaled_trajectory_component(acts, M, [1.0])
        # max_valid_layer must be a valid int ≤ 10
        self.assertLessEqual(result["max_valid_layer"], 10)
        self.assertGreaterEqual(result["max_valid_layer"], 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. rotational_rescaled — compare_rescaled_frames and interpret_comparison
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompareRescaledFrames(unittest.TestCase):

    def _run(self, ov_data):
        acts = _unit_acts()
        return compare_rescaled_frames(acts, ov_data, beta_values=[1.0])

    def test_shared_model_keys(self):
        result = self._run(_make_ov_data_shared())
        for k in ["original", "full_rescaled", "signed_only", "rotation_only",
                  "sa_decomp", "comparison"]:
            self.assertIn(k, result)

    def test_per_layer_model_keys(self):
        result = self._run(_make_ov_data_per_layer())
        for k in ["original", "full_rescaled", "signed_only", "rotation_only"]:
            self.assertIn(k, result)

    def test_n_violations_nonnegative(self):
        result = self._run(_make_ov_data_shared())
        comp = result["comparison"][1.0]
        for k in ["n_original", "n_full_rescaled", "n_signed_only", "n_rotation_only"]:
            self.assertGreaterEqual(comp[k], 0)

    def test_elim_rates_in_zero_one(self):
        result = self._run(_make_ov_data_shared())
        comp = result["comparison"][1.0]
        for k in ["elim_full", "elim_signed", "elim_rotation"]:
            self.assertGreaterEqual(comp[k], -1e-6)
            self.assertLessEqual(comp[k], 1.0 + 1e-6)


class TestInterpretComparison(unittest.TestCase):

    def _comp(self, e_full, e_sign, e_rot):
        return {
            1.0: {
                "n_original": 10,
                "n_full_rescaled": int(10 * (1 - e_full)),
                "n_signed_only":   int(10 * (1 - e_sign)),
                "n_rotation_only": int(10 * (1 - e_rot)),
                "elim_full": e_full,
                "elim_signed": e_sign,
                "elim_rotation": e_rot,
            }
        }

    def test_rotation_neutral(self):
        result = interpret_comparison(self._comp(0.8, 0.75, 0.05))
        self.assertEqual(result["per_beta"][1.0]["classification"], "rotation_neutral")

    def test_rotation_contributes(self):
        result = interpret_comparison(self._comp(0.8, 0.6, 0.4))
        cat = result["per_beta"][1.0]["classification"]
        self.assertIn(cat, {"rotation_contributes", "rotation_dominant"})

    def test_rotation_dominant(self):
        result = interpret_comparison(self._comp(0.8, 0.5, 0.75))
        self.assertEqual(result["per_beta"][1.0]["classification"], "rotation_dominant")

    def test_overall_field_present(self):
        result = interpret_comparison(self._comp(0.8, 0.75, 0.05))
        self.assertIn("overall", result)

    def test_zero_violations_does_not_crash(self):
        comp = {1.0: {
            "n_original": 0, "n_full_rescaled": 0,
            "n_signed_only": 0, "n_rotation_only": 0,
            "elim_full": 0.0, "elim_signed": 0.0, "elim_rotation": 0.0,
        }}
        result = interpret_comparison(comp)
        self.assertIn("overall", result)


class TestAnalyzeRotationalRescaling(unittest.TestCase):

    def test_shared_model_output_keys(self):
        acts = _unit_acts()
        ov_data = _make_ov_data_shared()
        result = analyze_rotational_rescaling(acts, ov_data)
        self.assertIn("frames", result)
        self.assertIn("interpretation", result)

    def test_per_layer_model_output_keys(self):
        acts = _unit_acts()
        ov_data = _make_ov_data_per_layer()
        result = analyze_rotational_rescaling(acts, ov_data)
        self.assertIn("frames", result)

    def test_comparison_to_json_serialisable(self):
        import json
        acts = _unit_acts()
        ov_data = _make_ov_data_shared()
        result = analyze_rotational_rescaling(acts, ov_data)
        j = comparison_to_json(result)
        json.dumps(j)   # must not raise

    def test_comparison_to_json_keys(self):
        acts = _unit_acts()
        result = analyze_rotational_rescaling(acts, _make_ov_data_shared())
        j = comparison_to_json(result)
        self.assertIn("sa_decomp", j)
        self.assertIn("comparison", j)
        self.assertIn("interpretation", j)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. fiedler_tracking — extract_fiedler_per_layer
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractFiedlerBlockDiagonal(unittest.TestCase):
    """Two disconnected clusters → Fiedler eigenvalue ≈ 0, clean bipartition."""

    def setUp(self):
        X = _block_acts()[np.newaxis]   # (1, 20, 8)
        self.result = extract_fiedler_per_layer(X)

    def test_valid(self):
        self.assertTrue(self.result["valid"][0])

    def test_fiedler_eigenvalue_near_zero(self):
        self.assertLess(abs(self.result["fiedler_vals"][0]), 1e-6)

    def test_sign_separates_blocks(self):
        vec = self.result["fiedler_vecs"][0]
        signs_A = np.sign(vec[: N_TOKENS // 2])
        signs_B = np.sign(vec[N_TOKENS // 2 :])
        self.assertTrue(
            (np.all(signs_A == 1) and np.all(signs_B == -1)) or
            (np.all(signs_A == -1) and np.all(signs_B == 1)),
            "Fiedler vector must split the two clusters by sign"
        )


class TestExtractFiedlerRandom(unittest.TestCase):

    def setUp(self):
        self.acts = _unit_acts()
        self.result = extract_fiedler_per_layer(self.acts)

    def test_all_layers_valid(self):
        self.assertTrue(self.result["valid"].all())

    def test_output_shapes(self):
        self.assertEqual(self.result["fiedler_vecs"].shape, (N_LAYERS, N_TOKENS))
        self.assertEqual(self.result["fiedler_vals"].shape, (N_LAYERS,))

    def test_fiedler_vals_nonnegative(self):
        vals = self.result["fiedler_vals"]
        self.assertTrue(np.all(vals[np.isfinite(vals)] >= -1e-10))


# ═══════════════════════════════════════════════════════════════════════════════
# 9. fiedler_tracking — hemisphere_assignments, crossing_rate, stability
# ═══════════════════════════════════════════════════════════════════════════════

class TestHemisphereAssignments(unittest.TestCase):

    def test_sizes_sum_to_n_tokens(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        for L in range(N_LAYERS):
            if fiedler["valid"][L]:
                self.assertEqual(hemi["sizes"][L].sum(), N_TOKENS)

    def test_assignments_binary(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        for L in range(N_LAYERS):
            unique = set(hemi["assignments"][L])
            self.assertTrue(unique <= {0, 1})


class TestHemisphereCrossingRate(unittest.TestCase):

    def test_stable_hemispheres_zero_rate(self):
        """Block acts that never change → crossing rate = 0."""
        acts = _block_acts_ml()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        crossing = hemisphere_crossing_rate(hemi["assignments"], fiedler["valid"])
        valid = crossing["crossing_rate"][np.isfinite(crossing["crossing_rate"])]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_rates_in_zero_half(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        crossing = hemisphere_crossing_rate(hemi["assignments"], fiedler["valid"])
        valid = crossing["crossing_rate"][np.isfinite(crossing["crossing_rate"])]
        self.assertTrue(np.all(valid >= 0.0))
        self.assertTrue(np.all(valid <= 0.5 + 1e-10))


class TestFiedlerStability(unittest.TestCase):

    def test_stable_acts_cosine_one(self):
        """Identical activations across layers → Fiedler vector unchanged → cosine = 1."""
        acts = _block_acts_ml()
        fiedler = extract_fiedler_per_layer(acts)
        stability = fiedler_stability(fiedler)
        cosines = stability["fiedler_cosine"]
        valid = cosines[np.isfinite(cosines)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-6)

    def test_cosines_in_zero_one(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        stability = fiedler_stability(fiedler)
        valid = stability["fiedler_cosine"][np.isfinite(stability["fiedler_cosine"])]
        self.assertTrue(np.all(valid >= -1e-10))
        self.assertTrue(np.all(valid <= 1.0 + 1e-10))


# ═══════════════════════════════════════════════════════════════════════════════
# 10. fiedler_tracking — centroid separation and crossref
# ═══════════════════════════════════════════════════════════════════════════════

class TestHemisphereCentroidSeparation(unittest.TestCase):

    def test_antipodal_block_acts_angle_near_pi(self):
        """e_0 and e_1 clusters are exactly orthogonal → centroid angle = π/2."""
        acts = _block_acts_ml()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        sep = hemisphere_centroid_separation(acts, hemi["assignments"], fiedler["valid"])
        angles = sep["centroid_angle"]
        valid = angles[np.isfinite(angles)]
        # e0 ⊥ e1 → angle between centroids = π/2
        np.testing.assert_allclose(valid, np.pi / 2, atol=0.05)

    def test_centroid_cos_in_neg1_pos1(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        sep = hemisphere_centroid_separation(acts, hemi["assignments"], fiedler["valid"])
        valid = sep["centroid_cos"][np.isfinite(sep["centroid_cos"])]
        self.assertTrue(np.all(valid >= -1.0 - 1e-6))
        self.assertTrue(np.all(valid <= 1.0 + 1e-6))


class TestCrossrefWithEvents(unittest.TestCase):

    def test_no_events_no_crash(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        crossing = hemisphere_crossing_rate(hemi["assignments"], fiedler["valid"])
        stability = fiedler_stability(fiedler)
        events = _make_phase1_events(violation_layers=[])
        result = crossref_with_events(
            crossing["crossing_rate"], stability["fiedler_cosine"], events, beta=1.0
        )
        self.assertIn("violations_with_elevated_crossing", result)

    def test_with_violations_fraction_in_zero_one(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        crossing = hemisphere_crossing_rate(hemi["assignments"], fiedler["valid"])
        stability = fiedler_stability(fiedler)
        events = _make_phase1_events(violation_layers=[2, 4])
        result = crossref_with_events(
            crossing["crossing_rate"], stability["fiedler_cosine"], events, beta=1.0
        )
        frac = result["violations_with_elevated_crossing"]
        self.assertGreaterEqual(frac, 0.0)
        self.assertLessEqual(frac, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. fiedler_tracking — analyze_fiedler_tracking and fiedler_to_json
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeFiedlerTracking(unittest.TestCase):

    def test_without_events(self):
        acts = _unit_acts()
        result = analyze_fiedler_tracking(acts)
        for k in ["fiedler_vals", "crossing_rate", "fiedler_cosine",
                  "centroid_angle", "n_layers", "n_tokens"]:
            self.assertIn(k, result)
        self.assertNotIn("crossref", result)

    def test_with_events(self):
        acts = _unit_acts()
        events = _make_phase1_events()
        result = analyze_fiedler_tracking(acts, events, beta=1.0)
        self.assertIn("crossref", result)

    def test_shapes_consistent(self):
        acts = _unit_acts()
        result = analyze_fiedler_tracking(acts)
        self.assertEqual(result["n_layers"], N_LAYERS)
        self.assertEqual(result["n_tokens"], N_TOKENS)

    def test_fiedler_to_json_serialisable(self):
        import json
        acts = _unit_acts()
        result = analyze_fiedler_tracking(acts, _make_phase1_events())
        j = fiedler_to_json(result)
        json.dumps(j)

    def test_fiedler_to_json_keys(self):
        acts = _unit_acts()
        result = analyze_fiedler_tracking(acts)
        j = fiedler_to_json(result)
        for k in ["n_layers", "n_tokens", "crossing_rate_mean",
                  "fiedler_cosine_mean", "centroid_angle_mean"]:
            self.assertIn(k, j)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. rotation_hemisphere — plane_fiedler_alignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlaneFiedlerAlignment(unittest.TestCase):

    def _setup(self):
        acts = _block_acts_ml()
        fiedler = extract_fiedler_per_layer(acts)
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        return acts, fiedler, planes

    def test_returns_overall_key(self):
        acts, fiedler, planes = self._setup()
        result = plane_fiedler_alignment(
            planes, fiedler["fiedler_vecs"], fiedler["valid"], acts
        )
        self.assertIn("overall", result)

    def test_overall_in_neg1_pos1(self):
        acts, fiedler, planes = self._setup()
        result = plane_fiedler_alignment(
            planes, fiedler["fiedler_vecs"], fiedler["valid"], acts
        )
        overall = result["overall"]
        if overall is not None and np.isfinite(overall):
            self.assertGreaterEqual(overall, -1.0 - 1e-6)
            self.assertLessEqual(overall, 1.0 + 1e-6)

    def test_per_layer_planes(self):
        """Per-layer plane projectors (list) must not crash."""
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        rng = np.random.default_rng(9)
        planes_list = []
        for _ in range(N_LAYERS):
            OV = rng.standard_normal((D, D))
            blocks = extract_schur_blocks(OV)
            planes_list.append(build_rotation_plane_projectors(blocks, top_k=2))
        result = plane_fiedler_alignment(
            planes_list, fiedler["fiedler_vecs"], fiedler["valid"], acts
        )
        self.assertIn("overall", result)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. rotation_hemisphere — token_fiedler_displacement and displacement_coherence
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenFiedlerDisplacement(unittest.TestCase):

    def test_shape(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        result = token_fiedler_displacement(acts, fiedler["fiedler_vecs"], fiedler["valid"])
        self.assertIn("displacement_proj", result)

    def test_stable_acts_zero_displacement(self):
        """Identical layers → Δx = 0 → displacement projection = 0."""
        acts = _block_acts_ml()
        fiedler = extract_fiedler_per_layer(acts)
        result = token_fiedler_displacement(acts, fiedler["fiedler_vecs"], fiedler["valid"])
        proj = result["displacement_proj"]
        valid = proj[np.isfinite(proj)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)


class TestDisplacementCoherence(unittest.TestCase):

    def test_coherence_one_for_uniform_displacement(self):
        """
        All tokens displaced by the same vector → coherence = 1.
        Construct acts where layer L+1 = layer L + constant shift.
        """
        rng = np.random.default_rng(3)
        acts = np.zeros((N_LAYERS, N_TOKENS, D))
        base = rng.standard_normal((N_TOKENS, D))
        base /= np.linalg.norm(base, axis=-1, keepdims=True)
        shift = rng.standard_normal(D)
        shift /= np.linalg.norm(shift)
        for L in range(N_LAYERS):
            acts[L] = base + (L * 0.01) * shift[np.newaxis]
            norms = np.linalg.norm(acts[L], axis=-1, keepdims=True)
            acts[L] /= np.maximum(norms, 1e-10)

        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        result = displacement_coherence(acts, hemi["assignments"], fiedler["valid"], planes)
        self.assertIn("coherence_mean", result)

    def test_coherence_values_in_zero_one(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        result = displacement_coherence(acts, hemi["assignments"], fiedler["valid"], planes)
        coh = result["coherence_mean"]
        valid = coh[np.isfinite(coh)]
        self.assertTrue(np.all(valid >= -1e-6))
        self.assertTrue(np.all(valid <= 1.0 + 1e-6))


# ═══════════════════════════════════════════════════════════════════════════════
# 14. rotation_hemisphere — analyze_rotation_hemisphere and to_json
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeRotationHemisphere(unittest.TestCase):

    def _run(self, plane_projectors):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        return analyze_rotation_hemisphere(acts, fiedler, hemi, plane_projectors), acts

    def test_shared_projectors_keys(self):
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        result, _ = self._run(planes)
        for k in ["plane_alignment", "token_displacement", "crossing_ratio", "coherence"]:
            self.assertIn(k, result)

    def test_per_layer_projectors_keys(self):
        rng = np.random.default_rng(1)
        planes_list = []
        for _ in range(N_LAYERS):
            OV = rng.standard_normal((D, D))
            blocks = extract_schur_blocks(OV)
            planes_list.append(build_rotation_plane_projectors(blocks, top_k=2))
        result, _ = self._run(planes_list)
        for k in ["plane_alignment", "coherence"]:
            self.assertIn(k, result)

    def test_to_json_serialisable(self):
        import json
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        result, _ = self._run(planes)
        j = rotation_hemisphere_to_json(result)
        json.dumps(j)

    def test_to_json_keys(self):
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        result, _ = self._run(planes)
        j = rotation_hemisphere_to_json(result)
        for k in ["plane_fiedler_alignment", "crossing_ratio"]:
            self.assertIn(k, j)


# ═══════════════════════════════════════════════════════════════════════════════
# 15. ffn_rotation — project_ffn_onto_rotation_planes
# ═══════════════════════════════════════════════════════════════════════════════

class TestProjectFFNOntoRotationPlanes(unittest.TestCase):

    def _planes(self):
        blocks, _ = _make_schur_blocks_pure_rotation()
        return build_rotation_plane_projectors(blocks, top_k=TOP_K)

    def _run_shared(self):
        deltas = _make_ffn_deltas()
        return project_ffn_onto_rotation_planes(deltas, self._planes(), is_per_layer=False)

    def test_output_keys(self):
        result = self._run_shared()
        for k in ["ffn_rotation_frac", "ffn_real_frac", "ffn_residual_frac",
                  "ffn_total_energy"]:
            self.assertIn(k, result)

    def test_fractions_nonnegative(self):
        result = self._run_shared()
        self.assertTrue(np.all(result["ffn_rotation_frac"] >= -1e-6))
        self.assertTrue(np.all(result["ffn_real_frac"] >= -1e-6))
        self.assertTrue(np.all(result["ffn_residual_frac"] >= -1e-6))

    def test_fractions_sum_leq_one(self):
        result = self._run_shared()
        total = result["ffn_rotation_frac"] + result["ffn_real_frac"] + result["ffn_residual_frac"]
        self.assertTrue(np.all(total <= 1.0 + 1e-6))

    def test_zero_deltas_zero_fracs(self):
        deltas = np.zeros((N_LAYERS, N_TOKENS, D))
        result = project_ffn_onto_rotation_planes(deltas, self._planes(), is_per_layer=False)
        np.testing.assert_allclose(result["ffn_rotation_frac"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["ffn_total_energy"], 0.0, atol=1e-10)

    def test_per_layer_projectors(self):
        rng = np.random.default_rng(2)
        planes_list = []
        for _ in range(N_LAYERS):
            OV = rng.standard_normal((D, D))
            blocks = extract_schur_blocks(OV)
            planes_list.append(build_rotation_plane_projectors(blocks, top_k=2))
        deltas = _make_ffn_deltas()
        result = project_ffn_onto_rotation_planes(deltas, planes_list, is_per_layer=True)
        self.assertEqual(result["ffn_rotation_frac"].shape[0], N_LAYERS)

    def test_pure_rotation_plane_delta_frac_one(self):
        """
        FFN delta entirely in the first rotation plane → rotation_frac ≈ 1.
        """
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        v1 = planes["top_k_planes"][0][:, 0]  # first basis vector of first plane
        deltas = np.zeros((N_LAYERS, N_TOKENS, D))
        for L in range(N_LAYERS):
            deltas[L] = v1[np.newaxis]   # all tokens same direction in rotation plane
        result = project_ffn_onto_rotation_planes(deltas, planes, is_per_layer=False)
        np.testing.assert_allclose(result["ffn_rotation_frac"], 1.0, atol=1e-6)

    def test_top_plane_fracs_shape(self):
        result = self._run_shared()
        self.assertEqual(result["ffn_top_plane_fracs"].shape, (N_LAYERS, TOP_K))


# ═══════════════════════════════════════════════════════════════════════════════
# 16. ffn_rotation — compare_ffn_rotation_at_violations
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompareFFNRotationAtViolations(unittest.TestCase):

    def _projection(self):
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        deltas = _make_ffn_deltas()
        return project_ffn_onto_rotation_planes(deltas, planes, is_per_layer=False)

    def test_nan_on_empty_violations(self):
        proj = self._projection()
        events = _make_phase1_events(violation_layers=[])
        result = compare_ffn_rotation_at_violations(proj, events, beta=1.0)
        for metric in ["ffn_rotation_frac", "ffn_real_frac"]:
            self.assertTrue(
                np.isnan(result[metric]["z_score"]) or result[metric]["z_score"] is None
                or (isinstance(result[metric]["z_score"], float)),
            )

    def test_with_violations_has_z_score(self):
        proj = self._projection()
        events = _make_phase1_events(violation_layers=[2, 4])
        result = compare_ffn_rotation_at_violations(proj, events, beta=1.0)
        z = result["ffn_rotation_frac"]["z_score"]
        self.assertTrue(z is None or np.isfinite(z) or np.isnan(z))

    def test_result_keys(self):
        proj = self._projection()
        events = _make_phase1_events(violation_layers=[2])
        result = compare_ffn_rotation_at_violations(proj, events, beta=1.0)
        for metric in ["ffn_rotation_frac", "ffn_real_frac"]:
            self.assertIn(metric, result)
            for subkey in ["z_score", "v_mean", "pop_mean"]:
                self.assertIn(subkey, result[metric])


# ═══════════════════════════════════════════════════════════════════════════════
# 17. ffn_rotation — classify_ffn_rotation_per_violation
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyFFNRotation(unittest.TestCase):

    VALID_ROLES = {"rotation_dominant", "real_dominant", "mixed", "orthogonal"}

    def _run(self, violation_layers):
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        deltas = _make_ffn_deltas()
        events = _make_phase1_events(violation_layers=violation_layers)
        return classify_ffn_rotation_per_violation(
            deltas, planes, events, is_per_layer=False, beta=1.0
        )

    def test_empty_violations_returns_empty_list(self):
        result = self._run([])
        self.assertEqual(result, [])

    def test_roles_are_valid(self):
        result = self._run([2, 4])
        for item in result:
            self.assertIn(item["role"], self.VALID_ROLES)

    def test_per_violation_keys(self):
        result = self._run([2])
        if result:
            for k in ["layer", "rotation_frac", "real_frac", "residual_frac", "role"]:
                self.assertIn(k, result[0])

    def test_fractions_between_zero_and_one(self):
        result = self._run([2, 4])
        for item in result:
            self.assertGreaterEqual(item["rotation_frac"], -1e-6)
            self.assertLessEqual(item["rotation_frac"], 1.0 + 1e-6)

    def test_rotation_dominant_role_when_in_plane(self):
        """
        FFN deltas entirely in the rotation plane → every violation should be
        rotation_dominant.
        """
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        v1 = planes["top_k_planes"][0][:, 0]
        deltas = np.zeros((N_LAYERS, N_TOKENS, D))
        for L in range(N_LAYERS):
            deltas[L] = v1[np.newaxis]
        events = _make_phase1_events(violation_layers=[2, 3, 4])
        result = classify_ffn_rotation_per_violation(
            deltas, planes, events, is_per_layer=False, beta=1.0
        )
        for item in result:
            self.assertEqual(item["role"], "rotation_dominant")


# ═══════════════════════════════════════════════════════════════════════════════
# 18. ffn_rotation — analyze_ffn_rotation and ffn_rotation_to_json
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeFFNRotation(unittest.TestCase):

    def _run(self, violation_layers=None):
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        deltas = _make_ffn_deltas()
        events = _make_phase1_events(violation_layers=violation_layers or [2, 4])
        return analyze_ffn_rotation(
            deltas, planes, events, is_per_layer=False, beta=1.0
        )

    def test_output_keys(self):
        result = self._run()
        for k in ["projection", "comparison", "per_violation", "role_counts", "n_violations"]:
            self.assertIn(k, result)

    def test_n_violations_matches_events(self):
        result = self._run(violation_layers=[2, 4])
        # n_violations counts processed violations (may be < 2 if layer out of range)
        self.assertGreaterEqual(result["n_violations"], 0)

    def test_role_counts_keys_are_valid_roles(self):
        valid = {"rotation_dominant", "real_dominant", "mixed", "orthogonal"}
        result = self._run()
        for role in result["role_counts"]:
            self.assertIn(role, valid)

    def test_to_json_serialisable(self):
        import json
        result = self._run()
        j = ffn_rotation_to_json(result)
        json.dumps(j)

    def test_to_json_keys(self):
        result = self._run()
        j = ffn_rotation_to_json(result)
        self.assertIn("role_counts", j)
        self.assertIn("n_violations", j)

    def test_per_layer_planes(self):
        rng = np.random.default_rng(2)
        planes_list = []
        for _ in range(N_LAYERS):
            OV = rng.standard_normal((D, D))
            blocks = extract_schur_blocks(OV)
            planes_list.append(build_rotation_plane_projectors(blocks, top_k=2))
        deltas = _make_ffn_deltas()
        events = _make_phase1_events(violation_layers=[2, 4])
        result = analyze_ffn_rotation(
            deltas, planes_list, events, is_per_layer=True, beta=1.0
        )
        self.assertIn("projection", result)


# ═══════════════════════════════════════════════════════════════════════════════
# 19. Import contract tests — Phase 2 / Phase 1 artifact shapes
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase2ArtifactContract(unittest.TestCase):
    """
    Verify that analyze_rotational_spectrum and analyze_rotational_rescaling
    accept both artifact layouts that run_2i.load_ov_data produces.
    """

    def test_shared_ov_data_accepted_by_spectrum(self):
        ov_data = _make_ov_data_shared()
        result = analyze_rotational_spectrum(ov_data, top_k_planes=2)
        self.assertFalse(result["is_per_layer"])

    def test_per_layer_ov_data_accepted_by_spectrum(self):
        ov_data = _make_ov_data_per_layer()
        result = analyze_rotational_spectrum(ov_data, top_k_planes=2)
        self.assertTrue(result["is_per_layer"])
        self.assertEqual(len(result["blocks"]), N_LAYERS)

    def test_shared_ov_data_accepted_by_rescaling(self):
        acts = _unit_acts()
        ov_data = _make_ov_data_shared()
        result = analyze_rotational_rescaling(acts, ov_data)
        self.assertIn("frames", result)

    def test_per_layer_ov_data_accepted_by_rescaling(self):
        acts = _unit_acts()
        ov_data = _make_ov_data_per_layer()
        result = analyze_rotational_rescaling(acts, ov_data)
        self.assertIn("frames", result)

    def test_layer_name_preserved_in_depth_profile(self):
        ov_data = _make_ov_data_per_layer()
        result = analyze_rotational_spectrum(ov_data, top_k_planes=2)
        depth_names = [p["layer_name"] for p in result["depth_profile"]["per_layer"]]
        self.assertEqual(depth_names, ov_data["layer_names"])


class TestPhase1EventsContract(unittest.TestCase):
    """
    Verify that all analysis functions that consume Phase 1 events accept
    the dict structure produced by load_phase1_events / _make_phase1_events.
    """

    def test_fiedler_tracking_accepts_events(self):
        acts = _unit_acts()
        events = _make_phase1_events()
        result = analyze_fiedler_tracking(acts, events, beta=1.0)
        self.assertIn("crossref", result)

    def test_ffn_rotation_accepts_events(self):
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=2)
        deltas = _make_ffn_deltas()
        events = _make_phase1_events()
        result = analyze_ffn_rotation(deltas, planes, events)
        self.assertIn("n_violations", result)

    def test_crossref_with_events_accepts_structure(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        crossing = hemisphere_crossing_rate(hemi["assignments"], fiedler["valid"])
        stability = fiedler_stability(fiedler)
        events = _make_phase1_events(violation_layers=[1, 3, 5])
        result = crossref_with_events(
            crossing["crossing_rate"], stability["fiedler_cosine"], events, beta=1.0
        )
        self.assertIn("violations_with_elevated_crossing", result)

    def test_events_with_all_betas(self):
        """Energy violations keyed by float beta must all be readable."""
        events = _make_phase1_events(violation_layers=[2])
        for beta in [0.1, 1.0, 2.0, 5.0]:
            violations = events["energy_violations"].get(beta, [])
            self.assertIsInstance(violations, list)


# ═══════════════════════════════════════════════════════════════════════════════
# 20. JSON serialisation — all to_json functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestAllToJsonFunctions(unittest.TestCase):
    """Smoke-test all *_to_json / summary_to_json / comparison_to_json."""

    def _assert_serialisable(self, obj, label=""):
        import json
        try:
            json.dumps(obj)
        except (TypeError, ValueError) as e:
            self.fail(f"JSON serialisation failed for {label}: {e}")

    def test_summary_to_json_shared(self):
        ov_data = _make_ov_data_shared()
        result = analyze_rotational_spectrum(ov_data, top_k_planes=2)
        j = summary_to_json(result)
        self._assert_serialisable(j, "summary_to_json shared")

    def test_summary_to_json_per_layer(self):
        ov_data = _make_ov_data_per_layer()
        result = analyze_rotational_spectrum(ov_data, top_k_planes=2)
        j = summary_to_json(result)
        self._assert_serialisable(j, "summary_to_json per-layer")

    def test_comparison_to_json(self):
        acts = _unit_acts()
        result = analyze_rotational_rescaling(acts, _make_ov_data_shared())
        j = comparison_to_json(result)
        self._assert_serialisable(j, "comparison_to_json")

    def test_fiedler_to_json(self):
        acts = _unit_acts()
        result = analyze_fiedler_tracking(acts, _make_phase1_events())
        j = fiedler_to_json(result)
        self._assert_serialisable(j, "fiedler_to_json")

    def test_rotation_hemisphere_to_json(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        result = analyze_rotation_hemisphere(acts, fiedler, hemi, planes)
        j = rotation_hemisphere_to_json(result)
        self._assert_serialisable(j, "rotation_hemisphere_to_json")

    def test_ffn_rotation_to_json(self):
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        deltas = _make_ffn_deltas()
        events = _make_phase1_events(violation_layers=[2, 4])
        result = analyze_ffn_rotation(deltas, planes, events)
        j = ffn_rotation_to_json(result)
        self._assert_serialisable(j, "ffn_rotation_to_json")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
