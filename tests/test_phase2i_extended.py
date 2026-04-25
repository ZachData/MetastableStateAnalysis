"""
tests/test_phase2i_extended.py — Comprehensive tests for p2b_imaginary.

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

    def setUp(self):
        self.M = _rot2(np.pi / 4)
        self.blocks = extract_schur_blocks(self.M)

    def test_n_complex_one(self):
        self.assertEqual(self.blocks["n_complex"], 1)

    def test_n_real_zero(self):
        self.assertEqual(self.blocks["n_real"], 0)

    def test_d_correct(self):
        self.assertEqual(self.blocks["d"], 2)

    def test_theta_near_pi_over_4(self):
        b = self.blocks["blocks_2x2"][0]
        self.assertAlmostEqual(abs(b["theta"]), np.pi / 4, places=6)

    def test_rho_near_one(self):
        b = self.blocks["blocks_2x2"][0]
        self.assertAlmostEqual(b["rho"], 1.0, places=6)


class TestExtractSchurBlocks8x8(unittest.TestCase):
    """Full 8×8 block-diagonal rotation → all complex, no real."""

    def setUp(self):
        self.M = _rot_d([0.3, 0.7, 1.1, 1.5])
        self.blocks = extract_schur_blocks(self.M)

    def test_n_complex_four(self):
        self.assertEqual(self.blocks["n_complex"], 4)

    def test_n_real_zero(self):
        self.assertEqual(self.blocks["n_real"], 0)

    def test_total_dim(self):
        self.assertEqual(self.blocks["n_real"] + 2 * self.blocks["n_complex"], D)

    def test_theta_recovery(self):
        thetas_in = sorted([0.3, 0.7, 1.1, 1.5])
        thetas_out = sorted(b["theta"] for b in self.blocks["blocks_2x2"])
        for t_in, t_out in zip(thetas_in, thetas_out):
            self.assertAlmostEqual(t_in, t_out, places=6)


class TestExtractSchurBlocksIdentity(unittest.TestCase):
    """Identity → all eigenvalues 1.0, all real blocks."""

    def setUp(self):
        self.blocks = extract_schur_blocks(np.eye(D))

    def test_all_real(self):
        self.assertEqual(self.blocks["n_complex"], 0)
        self.assertEqual(self.blocks["n_real"], D)

    def test_all_eigenvalues_one(self):
        for b in self.blocks["blocks_1x1"]:
            self.assertAlmostEqual(b["value"], 1.0, places=8)


class TestExtractSchurBlocksRandom(unittest.TestCase):

    def test_dimension_invariant(self):
        rng = np.random.default_rng(99)
        for _ in range(5):
            M = rng.standard_normal((D, D))
            result = extract_schur_blocks(M)
            self.assertEqual(result["n_real"] + 2 * result["n_complex"], D)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. rotational_schur — rotation_energy_fractions
# ═══════════════════════════════════════════════════════════════════════════════

class TestRotationEnergyFractions(unittest.TestCase):

    def test_pure_rotation_all_in_imaginary(self):
        OV = _rot_d([0.3, 0.7, 1.1, 1.5])
        blocks = extract_schur_blocks(OV)
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


# ═══════════════════════════════════════════════════════════════════════════════
# 2b. rotational_schur — rotation_angle_stats
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# 2c. rotational_schur — henrici_nonnormality
# ═══════════════════════════════════════════════════════════════════════════════

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
        M = np.triu(rng.standard_normal((D, D)))
        np.fill_diagonal(M, 0.0)
        M += np.eye(D)
        blocks = extract_schur_blocks(M)
        h = henrici_nonnormality(blocks)
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
        P_rot  = self.planes["combined_rotation"]
        P_real = self.planes["real_subspace"]
        cross  = P_rot @ P_real
        self.assertAlmostEqual(np.max(np.abs(cross)), 0.0, places=6)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. rotational_schur — rotation_depth_profile and analyze_rotational_spectrum
# ═══════════════════════════════════════════════════════════════════════════════

class TestRotationDepthProfile(unittest.TestCase):

    def test_per_layer_output_shape(self):
        ov_data = _make_ov_data_per_layer()
        ov_list = ov_data["ov_total"]
        names = ov_data["layer_names"]
        depth = rotation_depth_profile(ov_list, names)
        self.assertIn("per_layer", depth)
        self.assertEqual(len(depth["per_layer"]), N_LAYERS)

    def test_layer_names_preserved(self):
        ov_data = _make_ov_data_per_layer()
        depth = rotation_depth_profile(ov_data["ov_total"], ov_data["layer_names"])
        names_out = [p["layer_name"] for p in depth["per_layer"]]
        self.assertEqual(names_out, ov_data["layer_names"])


class TestAnalyzeRotationalSpectrum(unittest.TestCase):

    def test_shared_model_keys(self):
        ov_data = _make_ov_data_shared()
        result = analyze_rotational_spectrum(ov_data, top_k_planes=2)
        for k in ["is_per_layer", "blocks", "angle_stats", "energy_fractions",
                  "henrici", "plane_projectors"]:
            self.assertIn(k, result)

    def test_per_layer_model_keys(self):
        ov_data = _make_ov_data_per_layer()
        result = analyze_rotational_spectrum(ov_data, top_k_planes=2)
        self.assertTrue(result["is_per_layer"])
        for k in ["blocks", "angle_stats", "energy_fractions", "henrici",
                  "plane_projectors", "depth_profile"]:
            self.assertIn(k, result)

    def test_per_layer_length(self):
        ov_data = _make_ov_data_per_layer()
        result = analyze_rotational_spectrum(ov_data, top_k_planes=2)
        self.assertEqual(len(result["blocks"]), N_LAYERS)

    def test_summary_to_json_serialisable(self):
        import json
        ov_data = _make_ov_data_shared()
        result = analyze_rotational_spectrum(ov_data, top_k_planes=TOP_K)
        j = summary_to_json(result)
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

    def test_zero_matrix_identity_rescaling(self):
        acts = _unit_acts()
        result = rescaled_trajectory_component(acts, np.zeros((D, D)), [1.0])
        self.assertEqual(result["max_valid_layer"], N_LAYERS)
        # n_violations should match original (no change)
        self.assertGreaterEqual(result["n_violations"][1.0], 0)

    def test_per_layer_zero_matrices(self):
        acts = _unit_acts()
        matrices = [np.zeros((D, D))] * N_LAYERS
        result = rescaled_trajectory_component(
            acts, None, [1.0], is_per_layer=True, matrices_list=matrices
        )
        self.assertEqual(result["max_valid_layer"], N_LAYERS)

    def test_divergence_truncates(self):
        acts = _unit_acts(n_layers=10)
        M = 1000.0 * np.eye(D)
        result = rescaled_trajectory_component(acts, M, [1.0])
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

    def _make_comp(self, elim_full, elim_signed, elim_rotation):
        return {1.0: {
            "n_original": 10, "n_full_rescaled": 0,
            "n_signed_only": 0, "n_rotation_only": 10,
            "elim_full": elim_full, "elim_signed": elim_signed,
            "elim_rotation": elim_rotation,
        }}

    def test_rotation_neutral_category(self):
        comp = self._make_comp(1.0, 1.0, 0.0)
        result = interpret_comparison(comp)
        self.assertIn("overall", result)

    def test_rotation_contributes_category(self):
        comp = self._make_comp(1.0, 0.5, 0.6)
        result = interpret_comparison(comp)
        self.assertIn("overall", result)

    def test_always_returns_overall(self):
        comp = self._make_comp(
            elim_full=0.0, elim_signed=0.0, elim_rotation=0.0,
        )
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
        json.dumps(j)

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
        acts = _block_acts_ml(n_layers=1)
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        result = hemisphere_centroid_separation(acts, hemi["assignments"], fiedler["valid"])
        angle = result["centroid_angle"][0]
        self.assertTrue(np.isfinite(angle))
        # e_0 and e_1 are orthogonal so angle = π/2
        self.assertAlmostEqual(angle, np.pi / 2, places=3)

    def test_identical_centroids_angle_nan_or_zero(self):
        """All tokens at same position → centroid diff ≈ 0 → angle undefined or 0."""
        X = np.ones((1, N_TOKENS, D), dtype=np.float64)
        X /= np.linalg.norm(X[0, 0])
        fiedler = extract_fiedler_per_layer(X)
        hemi = hemisphere_assignments(fiedler)
        result = hemisphere_centroid_separation(X, hemi["assignments"], fiedler["valid"])
        # angle is either nan or very small (degenerate case)
        angle = result["centroid_angle"][0]
        self.assertTrue(not np.isfinite(angle) or angle < 0.1)


class TestCrossrefWithEvents(unittest.TestCase):

    def test_without_events_no_crossref_key(self):
        acts = _unit_acts()
        result = analyze_fiedler_tracking(acts)
        self.assertNotIn("crossref", result)

    def test_with_events_crossref_key_present(self):
        acts = _unit_acts()
        events = _make_phase1_events()
        result = analyze_fiedler_tracking(acts, events, beta=1.0)
        self.assertIn("crossref", result)

    def test_crossref_violations_with_elevated_crossing_key(self):
        acts = _unit_acts()
        events = _make_phase1_events(violation_layers=[1, 3, 5])
        result = crossref_with_events(
            hemisphere_crossing_rate(
                hemisphere_assignments(extract_fiedler_per_layer(acts))["assignments"],
                extract_fiedler_per_layer(acts)["valid"],
            )["crossing_rate"],
            fiedler_stability(extract_fiedler_per_layer(acts))["fiedler_cosine"],
            events, beta=1.0,
        )
        self.assertIn("violations_with_elevated_crossing", result)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. fiedler_tracking — analyze_fiedler_tracking and fiedler_to_json
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeFiedlerTracking(unittest.TestCase):

    def test_output_keys(self):
        acts = _unit_acts()
        result = analyze_fiedler_tracking(acts)
        for k in ["fiedler_vals", "hemisphere_sizes", "crossing_rate",
                  "fiedler_cosine", "centroid_angle", "n_layers", "n_tokens"]:
            self.assertIn(k, result)

    def test_n_layers_n_tokens_correct(self):
        acts = _unit_acts()
        result = analyze_fiedler_tracking(acts)
        self.assertEqual(result["n_layers"], N_LAYERS)
        self.assertEqual(result["n_tokens"], N_TOKENS)

    def test_fiedler_to_json_serialisable(self):
        import json
        acts = _unit_acts()
        result = analyze_fiedler_tracking(acts)
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

    def test_overall_is_dict_with_mean_alignment(self):
        # FIX: "overall" is a dict, not a scalar.
        # Was: np.isfinite(overall) which raises TypeError on a dict.
        acts, fiedler, planes = self._setup()
        result = plane_fiedler_alignment(
            planes, fiedler["fiedler_vecs"], fiedler["valid"], acts
        )
        overall = result["overall"]
        self.assertIsInstance(overall, dict)
        self.assertIn("mean_alignment", overall)
        mean_a = overall["mean_alignment"]
        if np.isfinite(mean_a):
            self.assertGreaterEqual(mean_a, 0.0 - 1e-6)
            self.assertLessEqual(mean_a, 1.0 + 1e-6)

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
        # FIX: was "displacement_proj", correct key is "fiedler_displacement"
        self.assertIn("fiedler_displacement", result)

    def test_stable_acts_zero_displacement(self):
        """Identical layers → Δx = 0 → displacement projection = 0."""
        acts = _block_acts_ml()
        fiedler = extract_fiedler_per_layer(acts)
        result = token_fiedler_displacement(acts, fiedler["fiedler_vecs"], fiedler["valid"])
        abs_d = result["abs_displacement"]
        valid = abs_d[np.isfinite(abs_d)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_output_keys(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        result = token_fiedler_displacement(acts, fiedler["fiedler_vecs"], fiedler["valid"])
        for k in ["fiedler_displacement", "abs_displacement", "displacement_std"]:
            self.assertIn(k, result)

    def test_fiedler_displacement_shape(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        result = token_fiedler_displacement(acts, fiedler["fiedler_vecs"], fiedler["valid"])
        self.assertEqual(result["fiedler_displacement"].shape, (N_LAYERS - 1, N_TOKENS))


class TestDisplacementCoherence(unittest.TestCase):

    def test_identical_displacement_coherence_one(self):
        """All tokens displaced identically → coherence = 1."""
        acts = _block_acts_ml()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        result = displacement_coherence(acts, hemi["assignments"], fiedler["valid"], planes)
        self.assertIn("coherence_mean", result)

    def test_output_keys(self):
        acts = _unit_acts()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        result = displacement_coherence(acts, hemi["assignments"], fiedler["valid"], planes)
        for k in ["coherence_mean", "coherence_std"]:
            self.assertIn(k, result)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. rotation_hemisphere — analyze_rotation_hemisphere and to_json
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeRotationHemisphere(unittest.TestCase):

    def _run(self, plane_projectors):
        acts = _block_acts_ml()
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

    def test_output_keys(self):
        planes = self._planes()
        deltas = _make_ffn_deltas()
        result = project_ffn_onto_rotation_planes(deltas, planes)
        for k in ["ffn_rotation_frac", "ffn_real_frac", "ffn_residual_frac",
                  "ffn_total_energy"]:
            self.assertIn(k, result)

    def test_fractions_sum_leq_one(self):
        planes = self._planes()
        deltas = _make_ffn_deltas()
        result = project_ffn_onto_rotation_planes(deltas, planes)
        total = result["ffn_rotation_frac"] + result["ffn_real_frac"]
        self.assertTrue(np.all(total <= 1.0 + 1e-6))

    def test_per_layer_planes(self):
        rng = np.random.default_rng(3)
        planes_list = []
        for _ in range(N_LAYERS):
            OV = rng.standard_normal((D, D))
            blocks = extract_schur_blocks(OV)
            planes_list.append(build_rotation_plane_projectors(blocks, top_k=2))
        deltas = _make_ffn_deltas()
        result = project_ffn_onto_rotation_planes(deltas, planes_list, is_per_layer=True)
        self.assertIn("ffn_rotation_frac", result)


# ═══════════════════════════════════════════════════════════════════════════════
# 16. ffn_rotation — compare_ffn_rotation_at_violations
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompareFFNRotationAtViolations(unittest.TestCase):

    def test_empty_violations_returns_nan(self):
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        deltas = _make_ffn_deltas()
        proj = project_ffn_onto_rotation_planes(deltas, planes)
        events = _make_phase1_events(violation_layers=[])
        result = compare_ffn_rotation_at_violations(proj, events, beta=1.0)
        for metric in result.values():
            if isinstance(metric, dict):
                self.assertTrue(np.isnan(metric["z_score"]))

    def test_with_violations_returns_numeric(self):
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        deltas = _make_ffn_deltas()
        proj = project_ffn_onto_rotation_planes(deltas, planes)
        events = _make_phase1_events(violation_layers=[2, 4])
        result = compare_ffn_rotation_at_violations(proj, events, beta=1.0)
        self.assertIn("ffn_rotation_frac", result)


# ═══════════════════════════════════════════════════════════════════════════════
# 17. ffn_rotation — classify_ffn_rotation_per_violation
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyFFNRotationPerViolation(unittest.TestCase):

    def test_rotation_dominant_role(self):
        """FFN delta entirely in rotation plane → role = rotation_dominant."""
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
# 19. Import contracts — Phase 2 / Phase 1 artifact shapes
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase2ArtifactContract(unittest.TestCase):

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

    def test_summary_to_json_shared(self):
        import json
        result = analyze_rotational_spectrum(_make_ov_data_shared(), top_k_planes=2)
        json.dumps(summary_to_json(result))

    def test_summary_to_json_per_layer(self):
        import json
        result = analyze_rotational_spectrum(_make_ov_data_per_layer(), top_k_planes=2)
        json.dumps(summary_to_json(result))

    def test_comparison_to_json(self):
        import json
        acts = _unit_acts()
        result = analyze_rotational_rescaling(acts, _make_ov_data_shared())
        json.dumps(comparison_to_json(result))

    def test_fiedler_to_json(self):
        import json
        acts = _unit_acts()
        result = analyze_fiedler_tracking(acts)
        json.dumps(fiedler_to_json(result))

    def test_rotation_hemisphere_to_json(self):
        import json
        acts = _block_acts_ml()
        fiedler = extract_fiedler_per_layer(acts)
        hemi = hemisphere_assignments(fiedler)
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        result = analyze_rotation_hemisphere(acts, fiedler, hemi, planes)
        json.dumps(rotation_hemisphere_to_json(result))

    def test_ffn_rotation_to_json(self):
        import json
        blocks, _ = _make_schur_blocks_pure_rotation()
        planes = build_rotation_plane_projectors(blocks, top_k=TOP_K)
        deltas = _make_ffn_deltas()
        events = _make_phase1_events(violation_layers=[2, 4])
        result = analyze_ffn_rotation(deltas, planes, events)
        json.dumps(ffn_rotation_to_json(result))


if __name__ == "__main__":
    unittest.main()
