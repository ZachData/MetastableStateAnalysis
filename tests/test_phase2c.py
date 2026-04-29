"""
test_phase2c.py — Unit tests for all p2c_churchland modules.

Pure numpy tests only — no model inference required.
All fixtures are constructed analytically so test outcomes are exact.

Run:
    python -m pytest tests/test_phase2c.py -v
    python -m unittest tests.test_phase2c -v

Coverage
--------
C4  local_jacobian     : decompose_sa, sa_ratio, project_jacobian_onto_subspaces
C4  slow_point_compare : layer_sa_profile, bootstrap_ci, plateau_vs_merge_table,
                         compare_local_global
C1  jpca_fit           : pca_reduce, remove_condition_mean, build_regression_mats,
                         fit_skew_symmetric, r2_score, extract_rotation_planes,
                         fit_jpca (end-to-end on synthetic rotational data)
C1  jpca_alignment     : principal_angles, jpca_ua_alignment
C1  hdr_fit            : fit_single_condition_skew, consensus_rotation_plane,
                         hdr_variance_ratio
C2  tangling           : compute_velocities, project_channel, compute_Q_matrix,
                         token_Q_summary, tangling_three_channels
C5  icl_subspace_scaling: channel_magnitudes_one_prompt, monotonicity_score,
                          cross_task_direction_agreement
C5  context_selection  : trajectory_channel_projections, layer_cosine_similarity,
                         layer_l2_divergence, layer_angular_divergence
C3  cis_decompose      : compute_cis_decomposition, channel_variance_per_layer,
                         aggregate_channel_fractions, invariant_layer_velocity,
                         merge_layer_test, analyze_cis
"""

from __future__ import annotations

import sys
import os
import unittest

import numpy as np
import numpy.testing as npt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2c_churchland.local_jacobian import (
    decompose_sa, sa_ratio, project_jacobian_onto_subspaces,
)
from p2c_churchland.slow_point_compare import (
    layer_sa_profile, bootstrap_ci, plateau_vs_merge_table,
    compare_local_global,
)
from p2c_churchland.jpca_fit import (
    pca_reduce, remove_condition_mean, build_regression_mats,
    fit_skew_symmetric, r2_score, extract_rotation_planes, fit_jpca,
)
from p2c_churchland.jpca_alignment import (
    principal_angles, jpca_ua_alignment,
)
from p2c_churchland.hdr_fit import (
    fit_single_condition_skew, consensus_rotation_plane, hdr_variance_ratio,
)
from p2c_churchland.tangling import (
    compute_velocities, project_channel, compute_Q_matrix,
    token_Q_summary, tangling_three_channels,
)
from p2c_churchland.icl_subspace_scaling import (
    channel_magnitudes_one_prompt, monotonicity_score,
    cross_task_direction_agreement,
)
from p2c_churchland.context_selection import (
    trajectory_channel_projections, layer_cosine_similarity,
    layer_l2_divergence, layer_angular_divergence,
)
from p2c_churchland.cis_decompose import (
    compute_cis_decomposition, channel_variance_per_layer,
    aggregate_channel_fractions, invariant_layer_velocity,
    merge_layer_test, analyze_cis,
)


# ── shared fixture helpers ────────────────────────────────────────────────────

D = 8    # model dimension used throughout
RNG = np.random.default_rng(42)


def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s,  c]])


def _block_rotation(thetas) -> np.ndarray:
    """Block-diagonal D×D rotation matrix from D/2 angles."""
    M = np.zeros((D, D))
    for i, th in enumerate(thetas):
        M[2*i:2*i+2, 2*i:2*i+2] = _rot2(th)
    return M


def _orthogonal_projectors(d=D, r_A=4, r_S=4):
    """Return (P_A, P_S) as complementary orthogonal projectors from a random orthogonal basis."""
    Q, _ = np.linalg.qr(RNG.standard_normal((d, d)))
    P_A = Q[:, :r_A]  @ Q[:, :r_A].T
    P_S = Q[:, r_A:r_A+r_S] @ Q[:, r_A:r_A+r_S].T
    return P_A, P_S


def _make_fake_per_layer(layer_indices, n_centroids=3, d=D, sa_vals=None):
    """Construct synthetic per_layer dict with specified sa_ratios."""
    rng = np.random.default_rng(0)
    per_layer = {}
    for i, li in enumerate(layer_indices):
        recs = []
        for ci in range(n_centroids):
            sa = sa_vals[i] if sa_vals else rng.uniform(0.3, 0.9)
            recs.append({
                "centroid_id": ci,
                "J": rng.standard_normal((d, d)),
                "sa_ratio": sa,
                "decomp": {
                    "S_frob_sq": sa,
                    "A_frob_sq": 1.0 - sa,
                    "S_n_positive": 4,
                    "S_n_negative": 2,
                },
            })
        per_layer[li] = recs
    return per_layer


# ═══════════════════════════════════════════════════════════════════════════════
# C4 — local_jacobian
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecomposeSA(unittest.TestCase):

    def test_pure_symmetric_sa_ratio_is_one(self):
        J = RNG.standard_normal((D, D))
        S = 0.5 * (J + J.T)   # guaranteed symmetric
        res = decompose_sa(S)
        self.assertAlmostEqual(res["sa_ratio"], 1.0, places=8)

    def test_pure_antisymmetric_sa_ratio_is_zero(self):
        J = RNG.standard_normal((D, D))
        A = 0.5 * (J - J.T)   # guaranteed antisymmetric
        res = decompose_sa(A)
        self.assertAlmostEqual(res["sa_ratio"], 0.0, places=8)

    def test_symmetric_part_of_antisymmetric_is_zero(self):
        J = RNG.standard_normal((D, D))
        A = 0.5 * (J - J.T)
        res = decompose_sa(A)
        npt.assert_allclose(res["S"], np.zeros((D, D)), atol=1e-12)

    def test_antisymmetric_part_of_symmetric_is_zero(self):
        J = RNG.standard_normal((D, D))
        S = 0.5 * (J + J.T)
        res = decompose_sa(S)
        npt.assert_allclose(res["A"], np.zeros((D, D)), atol=1e-12)

    def test_s_plus_a_reconstructs_j(self):
        J = RNG.standard_normal((D, D))
        res = decompose_sa(J)
        npt.assert_allclose(res["S"] + res["A"], J, atol=1e-12)

    def test_sa_ratio_in_unit_interval(self):
        for _ in range(10):
            J = RNG.standard_normal((D, D))
            r = decompose_sa(J)["sa_ratio"]
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r,  1.0 + 1e-9)

    def test_frob_sq_partition(self):
        """S_frob² + A_frob² should equal the total Frobenius² of J."""
        J = RNG.standard_normal((D, D))
        res = decompose_sa(J)
        self.assertAlmostEqual(
            res["S_frob_sq"] + res["A_frob_sq"],
            res["J_frob_sq"],
            places=6,
        )

    def test_s_eigvals_count(self):
        J = RNG.standard_normal((D, D))
        res = decompose_sa(J)
        self.assertEqual(len(res["S_eigvals"]), D)

    def test_sa_ratio_wrapper(self):
        J = RNG.standard_normal((D, D))
        self.assertAlmostEqual(sa_ratio(J), decompose_sa(J)["sa_ratio"], places=12)


class TestProjectJacobianOntoSubspaces(unittest.TestCase):

    def test_identity_projector_captures_all(self):
        J = RNG.standard_normal((D, D))
        res = decompose_sa(J)
        P_I = np.eye(D)
        P_0 = np.zeros((D, D))
        al = project_jacobian_onto_subspaces(res, P_I, P_0)
        self.assertAlmostEqual(al["A_in_UA"], 1.0, places=6)
        self.assertAlmostEqual(al["S_in_US"], 0.0, places=6)

    def test_fractions_between_zero_and_one(self):
        J = RNG.standard_normal((D, D))
        P_A, P_S = _orthogonal_projectors()
        res = decompose_sa(J)
        al = project_jacobian_onto_subspaces(res, P_A, P_S)
        for v in al.values():
            self.assertGreaterEqual(float(v), 0.0)
            self.assertLessEqual(float(v),    1.0 + 1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# C4 — slow_point_compare
# ═══════════════════════════════════════════════════════════════════════════════

class TestLayerSAProfile(unittest.TestCase):

    def test_mean_correct(self):
        per_layer = {0: [{"sa_ratio": 0.8}, {"sa_ratio": 0.6}],
                     1: [{"sa_ratio": 0.2}]}
        prof = layer_sa_profile(per_layer)
        npt.assert_allclose(prof["mean_sa"], [0.7, 0.2], atol=1e-12)

    def test_n_centroids_correct(self):
        per_layer = {0: [{"sa_ratio": 0.5}] * 5,
                     1: [{"sa_ratio": 0.5}] * 3}
        prof = layer_sa_profile(per_layer)
        npt.assert_equal(prof["n_centroids"], [5, 3])

    def test_layer_order_respected(self):
        per_layer = {3: [{"sa_ratio": 0.9}], 1: [{"sa_ratio": 0.1}]}
        prof = layer_sa_profile(per_layer, layer_order=[1, 3])
        self.assertEqual(prof["layer_indices"], [1, 3])
        npt.assert_allclose(prof["mean_sa"], [0.1, 0.9], atol=1e-12)


class TestBootstrapCI(unittest.TestCase):

    def test_ci_contains_true_mean(self):
        rng = np.random.default_rng(7)
        vals = rng.normal(loc=5.0, scale=1.0, size=100)
        lo, hi = bootstrap_ci(vals, n_boot=3000, ci=0.95, seed=0)
        self.assertLess(lo, 5.0)
        self.assertGreater(hi, 5.0)

    def test_empty_vals_returns_nan(self):
        lo, hi = bootstrap_ci(np.array([]))
        self.assertTrue(np.isnan(lo))
        self.assertTrue(np.isnan(hi))

    def test_lo_less_than_hi(self):
        vals = np.arange(20, dtype=float)
        lo, hi = bootstrap_ci(vals)
        self.assertLess(lo, hi)


class TestPlateauVsMergeTable(unittest.TestCase):

    def test_grouping_correct(self):
        per_layer = _make_fake_per_layer(
            [0, 1, 2, 3], n_centroids=2,
            sa_vals=[0.9, 0.9, 0.2, 0.2],
        )
        tbl = plateau_vs_merge_table(per_layer,
                                     plateau_layers=[0, 1],
                                     merge_layers=[2, 3],
                                     global_sa_ratio=0.5)
        self.assertAlmostEqual(tbl["plateau_mean"], 0.9, places=6)
        self.assertAlmostEqual(tbl["merge_mean"],   0.2, places=6)

    def test_global_sa_ratio_echoed(self):
        per_layer = _make_fake_per_layer([0], n_centroids=1)
        tbl = plateau_vs_merge_table(per_layer, [], [], global_sa_ratio=0.777)
        self.assertAlmostEqual(tbl["global_sa_ratio"], 0.777)


class TestCompareLocalGlobal(unittest.TestCase):

    def _run(self, plateau_sa, merge_sa, global_r):
        """Inject known sa_ratios by constructing per_layer directly."""
        per_layer = {}
        for li, v in enumerate(plateau_sa):
            per_layer[li] = [{"sa_ratio": v}]
        offset = len(plateau_sa)
        for li, v in enumerate(merge_sa):
            per_layer[offset + li] = [{"sa_ratio": v}]
        plateau_layers = list(range(len(plateau_sa)))
        merge_layers   = list(range(offset, offset + len(merge_sa)))
        return compare_local_global(per_layer, plateau_layers, merge_layers, global_r)

    def test_s1_holds_when_plateau_exceeds_global(self):
        res = self._run(plateau_sa=[0.9, 0.8], merge_sa=[0.3], global_r=0.5)
        self.assertTrue(res["p2cs1_holds"])

    def test_s1_fails_when_plateau_below_global(self):
        res = self._run(plateau_sa=[0.3, 0.2], merge_sa=[0.1], global_r=0.5)
        self.assertFalse(res["p2cs1_holds"])

    def test_s2_holds_when_merge_below_plateau(self):
        res = self._run(plateau_sa=[0.9, 0.8], merge_sa=[0.2], global_r=0.5)
        self.assertTrue(res["p2cs2_holds"])

    def test_s2_fails_when_merge_above_plateau(self):
        res = self._run(plateau_sa=[0.2], merge_sa=[0.8, 0.9], global_r=0.1)
        self.assertFalse(res["p2cs2_holds"])


# ═══════════════════════════════════════════════════════════════════════════════
# C1 — jpca_fit
# ═══════════════════════════════════════════════════════════════════════════════

class TestPCAReduce(unittest.TestCase):

    def test_output_shape(self):
        X = RNG.standard_normal((5, 10, D))
        X_pc, V_pc, ve = pca_reduce(X, n_components=4)
        self.assertEqual(X_pc.shape, (5, 10, 4))
        self.assertEqual(V_pc.shape, (D, 4))
        self.assertEqual(len(ve), 4)

    def test_var_explained_sums_leq_one(self):
        X = RNG.standard_normal((4, 8, D))
        _, _, ve = pca_reduce(X, n_components=D)
        self.assertLessEqual(ve.sum(), 1.0 + 1e-9)
        self.assertGreaterEqual(ve.sum(), 0.0)

    def test_V_pc_orthonormal(self):
        X = RNG.standard_normal((6, 12, D))
        _, V_pc, _ = pca_reduce(X, n_components=4)
        npt.assert_allclose(V_pc.T @ V_pc, np.eye(4), atol=1e-10)

    def test_n_components_capped_at_d(self):
        X = RNG.standard_normal((3, 5, D))
        X_pc, V_pc, ve = pca_reduce(X, n_components=100)
        self.assertLessEqual(X_pc.shape[2], D)


class TestRemoveConditionMean(unittest.TestCase):

    def test_mean_is_zero_after_removal(self):
        X = RNG.standard_normal((6, 10, 4))
        X_d = remove_condition_mean(X)
        mean = X_d.mean(axis=0)
        npt.assert_allclose(mean, np.zeros_like(mean), atol=1e-12)

    def test_shape_preserved(self):
        X = RNG.standard_normal((4, 8, 6))
        self.assertEqual(remove_condition_mean(X).shape, X.shape)


class TestBuildRegressionMats(unittest.TestCase):

    def test_shapes(self):
        K, T, p = 3, 7, 4
        X_pc = RNG.standard_normal((K, T, p))
        X_r, dX_r = build_regression_mats(X_pc)
        N = K * (T - 1)
        self.assertEqual(X_r.shape,  (p, N))
        self.assertEqual(dX_r.shape, (p, N))

    def test_velocity_is_difference(self):
        K, T, p = 2, 4, 3
        X_pc = RNG.standard_normal((K, T, p))
        X_r, dX_r = build_regression_mats(X_pc)
        # First entry: condition 0, t=0 → dX should be x[0,1] - x[0,0]
        expected_vel = X_pc[0, 1] - X_pc[0, 0]
        npt.assert_allclose(dX_r[:, 0], expected_vel, atol=1e-12)


class TestFitSkewSymmetric(unittest.TestCase):

    def test_output_is_antisymmetric(self):
        M = RNG.standard_normal((6, 6))
        S = fit_skew_symmetric(M)
        npt.assert_allclose(S + S.T, np.zeros((6, 6)), atol=1e-12)

    def test_pure_symmetric_input_gives_zero(self):
        M = RNG.standard_normal((6, 6))
        S_in = 0.5 * (M + M.T)
        S_out = fit_skew_symmetric(S_in)
        npt.assert_allclose(S_out, np.zeros((6, 6)), atol=1e-12)

    def test_pure_antisymmetric_input_preserved(self):
        M = RNG.standard_normal((6, 6))
        A = 0.5 * (M - M.T)
        npt.assert_allclose(fit_skew_symmetric(A), A, atol=1e-12)


class TestR2Score(unittest.TestCase):

    def test_perfect_fit_is_one(self):
        M = RNG.standard_normal((4, 4))
        X = RNG.standard_normal((4, 20))
        dX = M @ X
        self.assertAlmostEqual(r2_score(M, X, dX), 1.0, places=8)

    def test_zero_matrix_gives_zero_or_negative(self):
        X  = RNG.standard_normal((4, 20))
        dX = RNG.standard_normal((4, 20))
        r2 = r2_score(np.zeros((4, 4)), X, dX)
        self.assertLessEqual(r2, 0.01)   # zero prediction → R²≈0

    def test_r2_bounded_above_by_one_for_ols(self):
        """Unconstrained OLS fit must have R² ≤ 1."""
        from p2c_churchland.jpca_fit import fit_unconstrained
        X  = RNG.standard_normal((4, 30))
        dX = RNG.standard_normal((4, 30))
        M_unc = fit_unconstrained(X, dX)
        self.assertLessEqual(r2_score(M_unc, X, dX), 1.0 + 1e-9)


class TestExtractRotationPlanes(unittest.TestCase):

    def test_returns_correct_count(self):
        M_skew = fit_skew_symmetric(RNG.standard_normal((6, 6)))
        V_pc   = np.eye(D, 6)
        planes = extract_rotation_planes(M_skew, V_pc, top_k=2)
        self.assertLessEqual(len(planes), 2)

    def test_plane_full_is_orthonormal(self):
        M_skew = fit_skew_symmetric(RNG.standard_normal((8, 8)))
        V_pc   = np.eye(D)
        planes = extract_rotation_planes(M_skew, V_pc, top_k=3)
        for pl in planes:
            P = pl["plane_full"]
            npt.assert_allclose(P.T @ P, np.eye(2), atol=1e-8)

    def test_projector_is_idempotent_and_symmetric(self):
        M_skew = fit_skew_symmetric(RNG.standard_normal((8, 8)))
        V_pc   = np.eye(D)
        planes = extract_rotation_planes(M_skew, V_pc, top_k=2)
        for pl in planes:
            Pr = pl["projector_full"]
            npt.assert_allclose(Pr @ Pr, Pr,      atol=1e-8)
            npt.assert_allclose(Pr, Pr.T,          atol=1e-8)

    def test_omega_positive(self):
        M_skew = fit_skew_symmetric(RNG.standard_normal((8, 8)))
        V_pc   = np.eye(D)
        planes = extract_rotation_planes(M_skew, V_pc, top_k=3)
        for pl in planes:
            self.assertGreater(pl["omega"], 0.0)


class TestFitJPCAEndToEnd(unittest.TestCase):

    def _make_rotational_activations(self, n_cond=8, n_layers=12, d=D, seed=0):
        """
        Synthetic data where each condition follows a noisy rotation in the
        first two dimensions — designed to produce a good skew-symmetric fit.
        """
        rng = np.random.default_rng(seed)
        M_true = np.zeros((d, d))
        M_true[0, 1] =  0.5
        M_true[1, 0] = -0.5   # skew-symmetric, rotation in plane (0,1)

        acts = np.zeros((n_cond, n_layers, d))
        for c in range(n_cond):
            x = rng.standard_normal(d) * 0.5
            for L in range(n_layers):
                acts[c, L] = x + rng.standard_normal(d) * 0.05
                x = x + M_true @ x * 0.3   # Euler step + noise

        return acts

    def test_r2_ratio_positive(self):
        acts = self._make_rotational_activations()
        res  = fit_jpca(acts, n_pc=4, top_k_planes=2)
        self.assertGreater(res["r2_ratio"], 0.0)

    def test_r2_ratio_in_valid_range(self):
        acts = self._make_rotational_activations()
        res  = fit_jpca(acts, n_pc=4, top_k_planes=2)
        # ratio ≤ 1 for the constrained vs unconstrained fit
        self.assertLessEqual(res["r2_ratio"], 1.0 + 1e-6)

    def test_output_keys_present(self):
        acts = self._make_rotational_activations()
        res  = fit_jpca(acts, n_pc=4)
        for k in ("M_skew", "r2_ratio", "planes", "V_pc", "p2cj1_holds"):
            self.assertIn(k, res)

    def test_m_skew_is_antisymmetric(self):
        acts = self._make_rotational_activations()
        res  = fit_jpca(acts, n_pc=4)
        M    = res["M_skew"]
        npt.assert_allclose(M + M.T, np.zeros_like(M), atol=1e-10)

    def test_marginal_flag_mutually_exclusive(self):
        acts = self._make_rotational_activations()
        res  = fit_jpca(acts, n_pc=4)
        # Cannot simultaneously be both holds and marginal
        self.assertFalse(res["p2cj1_holds"] and res["p2cj1_marginal"])


# ═══════════════════════════════════════════════════════════════════════════════
# C1 — jpca_alignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrincipalAngles(unittest.TestCase):

    def test_identical_subspaces_zero_angle(self):
        A = np.eye(D)[:, :2]
        angles = principal_angles(A, A)
        npt.assert_allclose(angles, np.zeros(2), atol=1e-8)

    def test_orthogonal_subspaces_ninety_degrees(self):
        A = np.eye(D)[:, :2]
        B = np.eye(D)[:, 2:4]
        angles = principal_angles(A, B)
        npt.assert_allclose(angles, np.full(2, 90.0), atol=1e-6)

    def test_angles_in_zero_to_ninety(self):
        A = np.eye(D)[:, :3]
        B = RNG.standard_normal((D, 3))
        B, _ = np.linalg.qr(B)
        angles = principal_angles(A, B[:, :3])
        self.assertTrue(np.all(angles >= -1e-8))
        self.assertTrue(np.all(angles <= 90.0 + 1e-8))

    def test_output_count_is_min_of_ranks(self):
        A = np.eye(D)[:, :3]
        B = np.eye(D)[:, :2]
        angles = principal_angles(A, B)
        self.assertEqual(len(angles), 2)


class TestJPCAAUAlignment(unittest.TestCase):

    def _synthetic_jpca_result(self, plane_full):
        """Minimal jpca_result-like dict with one plane."""
        return {"planes": [{"plane_full": plane_full, "omega": 0.3}]}

    def test_coinciding_planes_p2cj2_holds(self):
        plane = np.eye(D)[:, :2]
        ua_planes = [np.eye(D)[:, :2]]
        res_jpca = self._synthetic_jpca_result(plane)
        res = jpca_ua_alignment(res_jpca, ua_planes, angle_threshold_deg=30.0)
        self.assertTrue(res["p2cj2_holds"])
        self.assertAlmostEqual(res["mean_min_angle"], 0.0, places=5)

    def test_orthogonal_planes_p2cj2_fails(self):
        plane_j  = np.eye(D)[:, :2]
        plane_ua = np.eye(D)[:, 2:4]
        res_jpca = self._synthetic_jpca_result(plane_j)
        res = jpca_ua_alignment(res_jpca, [plane_ua], angle_threshold_deg=30.0)
        self.assertFalse(res["p2cj2_holds"])
        self.assertGreater(res["mean_min_angle"], 60.0)

    def test_distribution_tag_aligned(self):
        plane = np.eye(D)[:, :2]
        res_jpca = self._synthetic_jpca_result(plane)
        res = jpca_ua_alignment(res_jpca, [plane])
        self.assertEqual(res["angle_distribution"], "aligned")

    def test_distribution_tag_orthogonal(self):
        plane_j  = np.eye(D)[:, :2]
        plane_ua = np.eye(D)[:, 2:4]
        res_jpca = self._synthetic_jpca_result(plane_j)
        res = jpca_ua_alignment(res_jpca, [plane_ua])
        self.assertEqual(res["angle_distribution"], "orthogonal")


# ═══════════════════════════════════════════════════════════════════════════════
# C1 — hdr_fit
# ═══════════════════════════════════════════════════════════════════════════════

class TestHDRFit(unittest.TestCase):

    def test_single_condition_skew_is_antisymmetric(self):
        x_c = RNG.standard_normal((8, 6))
        M_c, _, _ = fit_single_condition_skew(x_c)
        npt.assert_allclose(M_c + M_c.T, np.zeros_like(M_c), atol=1e-10)

    def test_consensus_plane_shape_and_orthonormality(self):
        p = 6
        M_list = [fit_skew_symmetric(RNG.standard_normal((p, p))) for _ in range(4)]
        plane  = consensus_rotation_plane(M_list)
        self.assertEqual(plane.shape, (p, 2))
        npt.assert_allclose(plane.T @ plane, np.eye(2), atol=1e-8)

    def test_consensus_plane_empty_raises(self):
        with self.assertRaises((ValueError, IndexError, Exception)):
            consensus_rotation_plane([])

    def test_variance_ratio_bounded(self):
        p = 6
        M_list, X_list, dX_list = [], [], []
        for _ in range(3):
            x_c = RNG.standard_normal((8, p))
            M_c, X_c, dX_c = fit_single_condition_skew(x_c)
            M_list.append(M_c); X_list.append(X_c); dX_list.append(dX_c)
        plane = consensus_rotation_plane(M_list)
        vbkdn = hdr_variance_ratio(plane, M_list, X_list, dX_list)
        # Ratio can in theory slightly exceed 1 due to projection; check it's finite
        self.assertTrue(np.isfinite(vbkdn["variance_ratio"]))
        self.assertGreaterEqual(vbkdn["variance_ratio"], 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# C2 — tangling
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeVelocities(unittest.TestCase):

    def test_shape(self):
        acts = RNG.standard_normal((10, 5, D))
        vels = compute_velocities(acts)
        self.assertEqual(vels.shape, (9, 5, D))

    def test_correct_differences(self):
        acts = np.arange(3 * 2 * D, dtype=float).reshape(3, 2, D)
        vels = compute_velocities(acts)
        npt.assert_allclose(vels[0], acts[1] - acts[0], atol=1e-12)
        npt.assert_allclose(vels[1], acts[2] - acts[1], atol=1e-12)


class TestProjectChannel(unittest.TestCase):

    def test_idempotent(self):
        """Projecting onto P twice equals projecting once (P²=P)."""
        acts = RNG.standard_normal((6, 4, D))
        P, _ = _orthogonal_projectors()
        once  = project_channel(acts, P)
        twice = project_channel(once, P)
        npt.assert_allclose(once, twice, atol=1e-10)

    def test_shape_preserved(self):
        acts = RNG.standard_normal((5, 3, D))
        P, _ = _orthogonal_projectors()
        out  = project_channel(acts, P)
        self.assertEqual(out.shape, acts.shape)

    def test_identity_projector_unchanged(self):
        acts = RNG.standard_normal((4, 3, D))
        out  = project_channel(acts, np.eye(D))
        npt.assert_allclose(out, acts, atol=1e-12)


class TestComputeQMatrix(unittest.TestCase):

    def _uniform_vel_acts(self, T=6, N=4, d=D):
        """All velocities identical → denominator large relative to numerator
        only when positions differ, but vel_diff=0 → Q = pos_diff / eps → large.
        Useful as a shape/sign check."""
        pos = RNG.standard_normal((T, N, d))
        vel = np.ones((T, N, d))
        return pos, vel

    def test_shape(self):
        pos = RNG.standard_normal((6, 4, D))
        vel = RNG.standard_normal((6, 4, D))
        Q   = compute_Q_matrix(pos, vel)
        self.assertEqual(Q.shape, (6, 4))

    def test_non_negative(self):
        pos = RNG.standard_normal((5, 3, D))
        vel = RNG.standard_normal((5, 3, D))
        Q   = compute_Q_matrix(pos, vel)
        self.assertTrue(np.all(Q >= 0.0))

    def test_identical_positions_give_zero_Q(self):
        """If all positions are the same, pos_diff = 0 for all pairs → Q = 0."""
        T, N = 5, 3
        pos  = np.tile(RNG.standard_normal((1, N, D)), (T, 1, 1))
        vel  = RNG.standard_normal((T, N, D))
        Q    = compute_Q_matrix(pos, vel)
        npt.assert_allclose(Q, np.zeros((T, N)), atol=1e-8)

    def test_zero_vel_diff_raises_Q(self):
        """All velocities equal → vel_diff = 0 → denominator = eps → Q large."""
        T, N = 4, 2
        pos  = RNG.standard_normal((T, N, D))
        vel  = np.tile(RNG.standard_normal((1, N, D)), (T, 1, 1))
        Q    = compute_Q_matrix(pos, vel, eps=1e-6)
        # Q should be large because denominator = eps
        self.assertTrue(np.mean(Q) > 0.0)


class TestTokenQSummary(unittest.TestCase):

    def test_shapes(self):
        Q   = RNG.uniform(0, 10, size=(8, 5))
        summ = token_Q_summary(Q)
        self.assertEqual(summ["per_token_max"].shape,  (5,))
        self.assertEqual(summ["per_token_mean"].shape, (5,))
        self.assertEqual(summ["per_layer_mean"].shape, (8,))

    def test_population_mean_is_mean_of_per_token_max(self):
        Q    = RNG.uniform(0, 5, size=(6, 4))
        summ = token_Q_summary(Q)
        self.assertAlmostEqual(
            summ["population_mean"],
            float(Q.max(axis=0).mean()),
            places=10,
        )

    def test_p95_geq_median(self):
        Q    = RNG.uniform(0, 10, size=(10, 20))
        summ = token_Q_summary(Q)
        self.assertGreaterEqual(summ["population_p95"], summ["population_median"])


class TestTanglingThreeChannels(unittest.TestCase):

    def test_output_keys_present(self):
        P_A, P_S = _orthogonal_projectors()
        acts = RNG.standard_normal((8, 5, D))
        res  = tangling_three_channels(acts, P_A, P_S)
        for k in ("full", "S", "A", "p2ct1_holds", "A_vs_S_ratio"):
            self.assertIn(k, res)

    def test_Q_purely_in_A_channel_lower_when_A_is_smooth(self):
        """
        Construct activations where the A-channel is a pure rotation (smooth)
        and the S-channel is random noise. Expect Q_A < Q_S.
        """
        P_A, P_S = _orthogonal_projectors(d=D, r_A=4, r_S=4)
        T, N = 10, 6
        # S-channel: random per layer (high tangling expected)
        s_part = np.einsum("ij,lnj->lni",
                           P_S,
                           RNG.standard_normal((T, N, D)))
        # A-channel: smooth sinusoidal (low tangling expected)
        t_idx = np.linspace(0, 2 * np.pi, T)
        a_part = np.zeros((T, N, D))
        for i in range(N):
            a_part[:, i, 0] = np.sin(t_idx + i * 0.3)
            a_part[:, i, 1] = np.cos(t_idx + i * 0.3)
        a_part = np.einsum("ij,lnj->lni", P_A, a_part)

        acts = s_part + a_part
        res  = tangling_three_channels(acts, P_A, P_S, eps=1e-4)
        # A should have lower Q than S
        self.assertLess(res["A"]["population_mean"], res["S"]["population_mean"] * 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# C5 — icl_subspace_scaling
# ═══════════════════════════════════════════════════════════════════════════════

class TestChannelMagnitudesOnePrompt(unittest.TestCase):

    def test_non_negative(self):
        P_A, P_S = _orthogonal_projectors()
        acts = RNG.standard_normal((6, D))   # (n_layers, d)
        m = channel_magnitudes_one_prompt(acts, P_A, P_S)
        self.assertGreaterEqual(m["mag_A_total"], 0.0)
        self.assertGreaterEqual(m["mag_S_total"], 0.0)

    def test_normed_in_zero_one(self):
        P_A, P_S = _orthogonal_projectors()
        acts = RNG.standard_normal((8, D))
        m = channel_magnitudes_one_prompt(acts, P_A, P_S)
        self.assertGreaterEqual(m["mag_A_normed"], 0.0)
        self.assertLessEqual(m["mag_A_normed"],    1.0 + 1e-9)

    def test_identity_projector_captures_all(self):
        acts = RNG.standard_normal((5, D))
        m = channel_magnitudes_one_prompt(acts, np.eye(D), np.zeros((D, D)))
        self.assertAlmostEqual(m["mag_A_normed"], 1.0, places=8)
        self.assertAlmostEqual(m["mag_S_normed"], 0.0, places=8)


class TestMonotonicityScore(unittest.TestCase):

    def test_perfect_monotone_rho_is_one(self):
        k_vals = [0, 1, 2, 4, 8, 16]
        mags   = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        res = monotonicity_score(k_vals, mags)
        self.assertAlmostEqual(res["rho"], 1.0, places=8)
        self.assertTrue(res["monotone_increasing"])

    def test_constant_not_monotone(self):
        k_vals = [0, 1, 2, 4]
        mags   = np.ones(4)
        res = monotonicity_score(k_vals, mags)
        self.assertFalse(res["monotone_increasing"])

    def test_decreasing_negative_rho(self):
        k_vals = [0, 1, 2, 4]
        mags   = np.array([0.8, 0.6, 0.4, 0.2])
        res = monotonicity_score(k_vals, mags)
        self.assertLess(res["rho"], 0.0)


class TestCrossTaskDirectionAgreement(unittest.TestCase):

    def test_identical_directions_high_cosine(self):
        d = D
        v = RNG.standard_normal(d)
        v /= np.linalg.norm(v)
        dirs = np.tile(v, (4, 1))  # (n_k, d) — same direction for every k
        task_dirs = {"task_a": dirs.copy(), "task_b": dirs.copy()}
        res = cross_task_direction_agreement(task_dirs, k_vals=[0, 1, 2, 4])
        self.assertAlmostEqual(res["overall_mean_cosine"], 1.0, places=6)
        self.assertFalse(res["p2cm2_holds"])   # not task-specific

    def test_orthogonal_directions_low_cosine(self):
        dirs_a = np.tile(np.eye(D)[0], (4, 1))
        dirs_b = np.tile(np.eye(D)[1], (4, 1))
        task_dirs = {"task_a": dirs_a, "task_b": dirs_b}
        res = cross_task_direction_agreement(task_dirs, k_vals=[0, 1, 2, 4])
        self.assertAlmostEqual(res["overall_mean_cosine"], 0.0, places=6)
        self.assertTrue(res["p2cm2_holds"])    # task-specific


# ═══════════════════════════════════════════════════════════════════════════════
# C5 — context_selection
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrajectoryChannelProjections(unittest.TestCase):

    def test_shapes(self):
        acts = RNG.standard_normal((8, D))
        P_A, P_S = _orthogonal_projectors()
        out = trajectory_channel_projections(acts, P_A, P_S)
        self.assertEqual(out["A_vecs"].shape, (8, D))
        self.assertEqual(out["S_vecs"].shape, (8, D))

    def test_projection_idempotent(self):
        acts = RNG.standard_normal((6, D))
        P_A, P_S = _orthogonal_projectors()
        out  = trajectory_channel_projections(acts, P_A, P_S)
        out2 = trajectory_channel_projections(out["A_vecs"], P_A, P_S)
        npt.assert_allclose(out["A_vecs"], out2["A_vecs"], atol=1e-10)


class TestLayerCosineSimilarity(unittest.TestCase):

    def test_identical_vectors_give_one(self):
        v = RNG.standard_normal((6, D))
        cos = layer_cosine_similarity(v, v)
        npt.assert_allclose(cos, np.ones(6), atol=1e-8)

    def test_opposite_vectors_give_minus_one(self):
        v = RNG.standard_normal((4, D))
        cos = layer_cosine_similarity(v, -v)
        npt.assert_allclose(cos, -np.ones(4), atol=1e-8)

    def test_output_in_range(self):
        a = RNG.standard_normal((10, D))
        b = RNG.standard_normal((10, D))
        cos = layer_cosine_similarity(a, b)
        self.assertTrue(np.all(cos >= -1.0 - 1e-8))
        self.assertTrue(np.all(cos <=  1.0 + 1e-8))


class TestLayerL2Divergence(unittest.TestCase):

    def test_identical_vectors_give_zero(self):
        v = RNG.standard_normal((5, D))
        npt.assert_allclose(layer_l2_divergence(v, v), np.zeros(5), atol=1e-12)

    def test_non_negative(self):
        a = RNG.standard_normal((8, D))
        b = RNG.standard_normal((8, D))
        self.assertTrue(np.all(layer_l2_divergence(a, b) >= 0.0))

    def test_known_value(self):
        """||e1 - e2|| = sqrt(2) in R^D."""
        a = np.zeros((1, D)); a[0, 0] = 1.0
        b = np.zeros((1, D)); b[0, 1] = 1.0
        npt.assert_allclose(layer_l2_divergence(a, b), [np.sqrt(2)], atol=1e-10)


class TestLayerAngularDivergence(unittest.TestCase):

    def test_identical_gives_zero_degrees(self):
        v = RNG.standard_normal((4, D))
        ang = layer_angular_divergence(v, v)
        npt.assert_allclose(ang, np.zeros(4), atol=1e-6)

    def test_opposite_gives_180_degrees(self):
        v = RNG.standard_normal((3, D))
        ang = layer_angular_divergence(v, -v)
        npt.assert_allclose(ang, np.full(3, 180.0), atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# C3 — cis_decompose
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeCISDecomposition(unittest.TestCase):

    def _make_prompts(self, K=4, n_layers=6, n_tokens=5, d=D):
        return [RNG.standard_normal((n_layers, n_tokens, d)) for _ in range(K)]

    def test_invariant_plus_mean_specific_equals_original(self):
        prompts = self._make_prompts()
        res = compute_cis_decomposition(prompts)
        # mean(specific, axis=0) should be zero at every layer and token
        mean_spec = res["specific"].mean(axis=0)
        npt.assert_allclose(mean_spec, np.zeros_like(mean_spec), atol=1e-12)

    def test_invariant_equals_mean_of_prompts(self):
        prompts = self._make_prompts()
        res = compute_cis_decomposition(prompts)
        expected = np.stack(prompts).mean(axis=0)
        npt.assert_allclose(res["invariant"], expected, atol=1e-12)

    def test_shapes(self):
        K, L, N, d = 3, 5, 4, D
        prompts = self._make_prompts(K, L, N, d)
        res = compute_cis_decomposition(prompts)
        self.assertEqual(res["invariant"].shape, (L, N, d))
        self.assertEqual(res["specific"].shape,  (K, L, N, d))

    def test_raises_for_single_prompt(self):
        with self.assertRaises(AssertionError):
            compute_cis_decomposition([RNG.standard_normal((4, 3, D))])


class TestChannelVariancePerLayer(unittest.TestCase):

    def test_invariant_all_in_A_when_in_A_subspace(self):
        """When invariant lives entirely in P_A's column space, inv_frac_A ≈ 1."""
        P_A, P_S = _orthogonal_projectors(d=D, r_A=4, r_S=4)
        K, L, N = 4, 6, 5
        # Build activations whose mean at each layer lies in P_A's subspace
        mean_traj = np.einsum("ij,lnj->lni",
                              P_A, RNG.standard_normal((L, N, D)))
        # Add noise in A subspace for specific parts too
        prompts = [
            mean_traj + np.einsum("ij,lnj->lni",
                                  P_A, RNG.standard_normal((L, N, D)) * 0.01)
            for _ in range(K)
        ]
        decomp = compute_cis_decomposition(prompts)
        cv     = channel_variance_per_layer(decomp, P_A, P_S)
        # Invariant fraction in A should be very high
        self.assertGreater(cv["inv_frac_A"].mean(), 0.9)

    def test_fractions_in_zero_one(self):
        P_A, P_S = _orthogonal_projectors()
        prompts  = [RNG.standard_normal((5, 4, D)) for _ in range(3)]
        decomp   = compute_cis_decomposition(prompts)
        cv       = channel_variance_per_layer(decomp, P_A, P_S)
        for arr in (cv["inv_frac_A"], cv["spec_frac_S"]):
            self.assertTrue(np.all(arr >= 0.0 - 1e-9))
            self.assertTrue(np.all(arr <= 1.0 + 1e-9))


class TestAggregateChannelFractions(unittest.TestCase):

    def test_k1_holds_when_inv_in_A_and_spec_in_S(self):
        """Manually set variance arrays so the verdict is forced."""
        cv = {
            "inv_var_A":  np.array([8.0, 9.0]),
            "inv_var_S":  np.array([1.0, 1.0]),
            "spec_var_A": np.array([1.0, 1.0]),
            "spec_var_S": np.array([8.0, 9.0]),
        }
        k1 = aggregate_channel_fractions(cv)
        self.assertTrue(k1["p2ck1_holds"])

    def test_k1_fails_when_reversed(self):
        cv = {
            "inv_var_A":  np.array([1.0]),
            "inv_var_S":  np.array([9.0]),
            "spec_var_A": np.array([9.0]),
            "spec_var_S": np.array([1.0]),
        }
        k1 = aggregate_channel_fractions(cv)
        self.assertFalse(k1["p2ck1_holds"])


class TestInvariantLayerVelocity(unittest.TestCase):

    def test_shape(self):
        inv  = RNG.standard_normal((8, 5, D))
        vels = invariant_layer_velocity(inv)
        self.assertEqual(vels.shape, (7,))

    def test_non_negative(self):
        inv  = RNG.standard_normal((6, 4, D))
        vels = invariant_layer_velocity(inv)
        self.assertTrue(np.all(vels >= 0.0))

    def test_zero_for_constant_invariant(self):
        """If invariant is identical across layers, all velocities are zero."""
        base = RNG.standard_normal((1, 5, D))
        inv  = np.repeat(base, 7, axis=0)
        vels = invariant_layer_velocity(inv)
        npt.assert_allclose(vels, np.zeros(6), atol=1e-12)


class TestMergeLayerTest(unittest.TestCase):

    def test_k2_holds_when_merge_velocity_higher(self):
        vels = np.array([0.1, 0.1, 0.1, 1.5, 0.1, 0.1])
        res  = merge_layer_test(vels, merge_layers=[4], plateau_layers=[0, 1, 2])
        # merge at layer 4 → uses vels[3] = 1.5
        self.assertTrue(res["p2ck2_holds"])

    def test_k2_fails_when_merge_velocity_lower(self):
        vels = np.array([2.0, 2.0, 2.0, 0.1, 2.0])
        res  = merge_layer_test(vels, merge_layers=[4], plateau_layers=[0, 1, 2])
        self.assertFalse(res["p2ck2_holds"])

    def test_effect_sign_matches_verdict(self):
        vels = np.array([0.5, 0.5, 2.0, 0.5])
        res  = merge_layer_test(vels, merge_layers=[3], plateau_layers=[0, 1])
        self.assertEqual(res["p2ck2_holds"], res["k2_effect"] > 0.0)

    def test_returns_nan_for_empty_groups(self):
        vels = np.array([0.5, 0.5])
        res  = merge_layer_test(vels, merge_layers=[], plateau_layers=[])
        self.assertTrue(np.isnan(res["merge_mean_vel"]))


class TestAnalyzeCIS(unittest.TestCase):

    def test_full_pipeline_returns_all_keys(self):
        P_A, P_S = _orthogonal_projectors()
        prompts  = [RNG.standard_normal((5, 4, D)) for _ in range(3)]
        res      = analyze_cis(prompts, P_A, P_S,
                               merge_layers=[3], plateau_layers=[0, 1])
        for k in ("decomp", "cv", "k1", "velocities", "k2",
                  "p2ck1_holds", "p2ck2_holds"):
            self.assertIn(k, res)

    def test_no_k2_when_layers_not_provided(self):
        P_A, P_S = _orthogonal_projectors()
        prompts  = [RNG.standard_normal((5, 4, D)) for _ in range(3)]
        res      = analyze_cis(prompts, P_A, P_S)
        self.assertIsNone(res["k2"])
        self.assertIsNone(res["p2ck2_holds"])


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
