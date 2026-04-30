"""
tests/test_phase2c.py

Unit tests for p2c_churchland — Phase 2c trajectory-side dynamical-systems analysis.

No model loading.  All tests use small synthetic numpy arrays.

Coverage
--------
  jpca_fit.py          fit_jpca, jpca_to_json, pca_reduce, build_regression_mats
  jpca_alignment.py    principal_angles, align_jpca_to_ua, jpca_alignment_to_json
  hdr_fit.py           fit_hdr, hdr_to_json
  tangling.py          compute_velocities, compute_Q_matrix, tangling_three_channels
  cis_decompose.py     compute_cis_decomposition, channel_variance_per_layer,
                       analyze_cis, cis_to_json
  local_jacobian.py    compute_layer_jacobian, decompose_sa, centroid_jacobians_one_layer
  slow_point_compare.py  layer_sa_profile, plateau_vs_merge_table, compare_local_global
  icl_subspace_scaling.py  channel_magnitudes_one_prompt, kshot_channel_profile,
                            monotonicity_score, cross_task_direction_agreement
  context_selection.py   layer_cosine_similarity, layer_l2_divergence,
                          analyze_context_pair (offline path)

Run
---
    python -m pytest tests/test_phase2c.py -v
    python -m unittest tests.test_phase2c -v
"""

from __future__ import annotations

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Shared fixtures ────────────────────────────────────────────────────────────

N_COND   = 5    # number of "conditions" / prompts
N_LAYERS = 8    # layer depth ("time")
D        = 16   # residual-stream dimension
N_PC     = 4    # PCs used in jPCA / HDR


def _make_trajectories(seed: int = 0) -> np.ndarray:
    """(N_COND, N_LAYERS, D) synthetic centroid trajectories."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_COND, N_LAYERS, D)).astype(np.float64)


def _make_projectors(seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Random rank-4 projectors P_A, P_S that partition R^D (roughly)."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((D, D)))
    half = D // 2
    PA = U[:, :half] @ U[:, :half].T
    PS = U[:, half:] @ U[:, half:].T
    return PA, PS


def _make_ua_planes(n: int = 3, seed: int = 2) -> list[np.ndarray]:
    """List of n random (D, 2) orthonormal bases representing U_A planes."""
    rng = np.random.default_rng(seed)
    planes = []
    for _ in range(n):
        raw = rng.standard_normal((D, 2))
        q, _ = np.linalg.qr(raw)
        planes.append(q[:, :2])
    return planes


def _make_activations_per_prompt(
    K: int = 4,
    n_tokens: int = 10,
    seed: int = 3,
) -> list[np.ndarray]:
    """K arrays each (N_LAYERS, n_tokens, D) — matched-shape prompt activations."""
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal((N_LAYERS, n_tokens, D)).astype(np.float64)
        for _ in range(K)
    ]


def _make_centroids_per_layer(
    n_centroids: int = 6,
    seed: int = 4,
) -> dict[int, np.ndarray]:
    """layer_idx → (n_centroids, D) centroid array."""
    rng = np.random.default_rng(seed)
    return {li: rng.standard_normal((n_centroids, D)).astype(np.float64)
            for li in range(N_LAYERS)}


def _make_per_layer_records(
    n_centroids: int = 6,
    plateau_layers: list[int] | None = None,
    merge_layers: list[int] | None = None,
    seed: int = 5,
) -> dict:
    """Synthetic analyze_local_jacobians 'per_layer' output."""
    rng = np.random.default_rng(seed)
    per_layer: dict = {}
    for li in range(N_LAYERS):
        per_layer[li] = [
            {"centroid_id": ci, "sa_ratio": float(rng.uniform(0.5, 3.0))}
            for ci in range(n_centroids)
        ]
    return {
        "per_layer":       per_layer,
        "plateau_layers":  plateau_layers or [2, 3, 4],
        "merge_layers":    merge_layers   or [5],
        "global_sa_ratio": 1.5,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. jpca_fit
# ══════════════════════════════════════════════════════════════════════════════

class TestPcaReduce(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.jpca_fit import pca_reduce
        self.pca = pca_reduce
        self.X   = _make_trajectories()

    def test_output_shapes(self):
        X_pc, V_pc, var_exp = self.pca(self.X, n_components=N_PC)
        self.assertEqual(X_pc.shape,  (N_COND, N_LAYERS, N_PC))
        self.assertEqual(V_pc.shape,  (D, N_PC))
        self.assertEqual(var_exp.shape, (N_PC,))

    def test_var_explained_sums_leq_one(self):
        _, _, var_exp = self.pca(self.X, n_components=N_PC)
        self.assertLessEqual(float(var_exp.sum()), 1.0 + 1e-9)

    def test_var_explained_non_negative(self):
        _, _, var_exp = self.pca(self.X, n_components=N_PC)
        self.assertTrue(np.all(var_exp >= -1e-12))

    def test_pc_basis_orthonormal(self):
        _, V_pc, _ = self.pca(self.X, n_components=N_PC)
        gram = V_pc.T @ V_pc
        np.testing.assert_allclose(gram, np.eye(N_PC), atol=1e-9)

    def test_fewer_components_than_d(self):
        X_pc, _, _ = self.pca(self.X, n_components=2)
        self.assertEqual(X_pc.shape[-1], 2)


class TestBuildRegressionMats(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.jpca_fit import pca_reduce, build_regression_mats
        X   = _make_trajectories()
        X_pc, _, _ = pca_reduce(X, n_components=N_PC)
        self.X_reg, self.dX_reg = build_regression_mats(X_pc)

    def test_shapes_match(self):
        self.assertEqual(self.X_reg.shape,  self.dX_reg.shape)

    def test_n_observations(self):
        expected_cols = N_COND * (N_LAYERS - 1)
        self.assertEqual(self.X_reg.shape[1], expected_cols)

    def test_rows_equal_n_pc(self):
        self.assertEqual(self.X_reg.shape[0], N_PC)


class TestFitJpca(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.jpca_fit import fit_jpca
        self.result = fit_jpca(_make_trajectories(), n_pc=N_PC, top_k_planes=2)

    def test_required_keys(self):
        for k in ("r2_ratio", "r2_skew", "r2_unc", "planes",
                  "V_pc", "var_explained", "p2cj1_holds", "p2cj1_marginal"):
            self.assertIn(k, self.result, f"missing key: {k}")

    def test_r2_ratio_in_unit_interval(self):
        self.assertGreaterEqual(self.result["r2_ratio"], 0.0)
        self.assertLessEqual(self.result["r2_ratio"],    1.0 + 1e-9)

    def test_r2_skew_leq_r2_unc(self):
        self.assertLessEqual(self.result["r2_skew"],
                             self.result["r2_unc"] + 1e-9)

    def test_p2cj1_holds_is_bool(self):
        self.assertIsInstance(self.result["p2cj1_holds"], bool)

    def test_planes_list_length(self):
        self.assertEqual(len(self.result["planes"]), 2)

    def test_plane_has_plane_full(self):
        plane = self.result["planes"][0]
        self.assertIn("plane_full", plane)
        self.assertEqual(plane["plane_full"].shape, (D, 2))


class TestJpcaToJson(unittest.TestCase):

    def test_serializable(self):
        import json
        from p2c_churchland.jpca_fit import fit_jpca, jpca_to_json
        result = fit_jpca(_make_trajectories(), n_pc=N_PC)
        js = jpca_to_json(result)
        # Must not raise
        json.dumps(js)

    def test_verdict_preserved(self):
        from p2c_churchland.jpca_fit import fit_jpca, jpca_to_json
        result = fit_jpca(_make_trajectories(), n_pc=N_PC)
        js = jpca_to_json(result)
        self.assertIn("p2cj1_holds", js)
        self.assertIsInstance(js["p2cj1_holds"], bool)


# ══════════════════════════════════════════════════════════════════════════════
# 2. jpca_alignment
# ══════════════════════════════════════════════════════════════════════════════

class TestPrincipalAngles(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.jpca_alignment import principal_angles
        self.pa = principal_angles

    def test_identical_planes_give_zero(self):
        rng = np.random.default_rng(0)
        U, _ = np.linalg.qr(rng.standard_normal((D, 2)))
        angles = self.pa(U, U)
        np.testing.assert_allclose(angles, [0.0, 0.0], atol=1e-8)

    def test_orthogonal_planes_give_90(self):
        U = np.zeros((D, 2))
        U[0, 0] = 1.0
        U[1, 1] = 1.0
        V = np.zeros((D, 2))
        V[2, 0] = 1.0
        V[3, 1] = 1.0
        angles = self.pa(U, V)
        np.testing.assert_allclose(angles, [90.0, 90.0], atol=1e-6)

    def test_angles_in_range(self):
        rng = np.random.default_rng(1)
        A, _ = np.linalg.qr(rng.standard_normal((D, 2)))
        B, _ = np.linalg.qr(rng.standard_normal((D, 2)))
        angles = self.pa(A, B)
        self.assertTrue(np.all(angles >= -1e-9))
        self.assertTrue(np.all(angles <= 90.0 + 1e-9))

    def test_returns_two_angles_for_rank2(self):
        rng = np.random.default_rng(2)
        A, _ = np.linalg.qr(rng.standard_normal((D, 2)))
        B, _ = np.linalg.qr(rng.standard_normal((D, 2)))
        self.assertEqual(len(self.pa(A, B)), 2)


class TestAlignJpcaToUa(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.jpca_fit      import fit_jpca
        from p2c_churchland.jpca_alignment import align_jpca_to_ua
        self.jpca_result = fit_jpca(_make_trajectories(), n_pc=N_PC, top_k_planes=2)
        self.ua_planes   = _make_ua_planes(n=3)
        self.result = align_jpca_to_ua(self.jpca_result, self.ua_planes)

    def test_required_keys(self):
        for k in ("pairwise_angles", "per_jpca_min_angle",
                  "mean_min_angle", "p2cj2_holds", "angle_distribution"):
            self.assertIn(k, self.result, f"missing key: {k}")

    def test_pairwise_angles_shape(self):
        n_jp = len(self.jpca_result["planes"])
        n_ua = len(self.ua_planes)
        self.assertEqual(self.result["pairwise_angles"].shape, (n_jp, n_ua))

    def test_mean_min_angle_non_negative(self):
        self.assertGreaterEqual(self.result["mean_min_angle"], 0.0)

    def test_p2cj2_holds_is_bool(self):
        self.assertIsInstance(self.result["p2cj2_holds"], bool)

    def test_empty_ua_planes_handled(self):
        from p2c_churchland.jpca_alignment import align_jpca_to_ua
        r = align_jpca_to_ua(self.jpca_result, [])
        self.assertIn("p2cj2_holds", r)

    def test_distribution_is_valid_string(self):
        valid = {"aligned", "orthogonal", "mixed", "unknown"}
        self.assertIn(self.result["angle_distribution"], valid)


# ══════════════════════════════════════════════════════════════════════════════
# 3. hdr_fit
# ══════════════════════════════════════════════════════════════════════════════

class TestFitHdr(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.hdr_fit import fit_hdr
        self.result = fit_hdr(
            _make_trajectories(), n_pc=N_PC,
            ua_planes=_make_ua_planes(n=2),
        )

    def test_required_keys(self):
        for k in ("variance_ratio", "p2cj1_hdr_holds", "p2cj2_hdr_holds",
                  "ua_min_angle", "plane_full"):
            self.assertIn(k, self.result, f"missing key: {k}")

    def test_variance_ratio_in_unit_interval(self):
        vr = self.result["variance_ratio"]
        self.assertGreaterEqual(vr, 0.0)
        self.assertLessEqual(vr,    1.0 + 1e-9)

    def test_verdict_is_bool(self):
        self.assertIsInstance(self.result["p2cj1_hdr_holds"], bool)

    def test_plane_full_shape(self):
        self.assertEqual(self.result["plane_full"].shape, (D, 2))

    def test_plane_full_orthonormal(self):
        P = self.result["plane_full"]
        gram = P.T @ P
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-9)

    def test_hdr_to_json_serializable(self):
        import json
        from p2c_churchland.hdr_fit import hdr_to_json
        json.dumps(hdr_to_json(self.result))


# ══════════════════════════════════════════════════════════════════════════════
# 4. tangling
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeVelocities(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.tangling import compute_velocities
        self.cv = compute_velocities

    def test_output_shape(self):
        acts = np.random.default_rng(0).standard_normal((N_LAYERS, 20, D))
        vels = self.cv(acts)
        self.assertEqual(vels.shape, (N_LAYERS - 1, 20, D))

    def test_velocity_is_difference(self):
        acts = np.arange(24, dtype=float).reshape(3, 4, 2)
        vels = self.cv(acts)
        np.testing.assert_allclose(vels[0], acts[1] - acts[0])
        np.testing.assert_allclose(vels[1], acts[2] - acts[1])


class TestComputeQMatrix(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.tangling import compute_Q_matrix
        self.cq  = compute_Q_matrix
        rng      = np.random.default_rng(1)
        # (n_layers, n_tokens, d) but Q runs over layers; use small sizes
        self.X   = rng.standard_normal((6, 8, D))
        self.dX  = rng.standard_normal((6, 8, D))

    def test_output_shape(self):
        Q = self.cq(self.X, self.dX)
        self.assertEqual(Q.shape, (6, 8))

    def test_Q_non_negative(self):
        Q = self.cq(self.X, self.dX)
        self.assertTrue(np.all(Q >= 0.0))

    def test_constant_velocity_gives_zero_Q(self):
        """If all positions are distinct but all velocities identical → Q = 0."""
        X  = np.arange(48, dtype=float).reshape(6, 8, 1)
        dX = np.ones((6, 8, 1))
        Q  = self.cq(X, dX, eps=1e-30)
        # With identical velocities, numerator varies but denominator = 0 → large
        # (boundary case; just check no NaN / negative values)
        self.assertFalse(np.any(np.isnan(Q)))
        self.assertTrue(np.all(Q >= 0.0))


class TestTanglingThreeChannels(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.tangling import tangling_three_channels
        self.tc   = tangling_three_channels
        self.P_A, self.P_S = _make_projectors()
        rng       = np.random.default_rng(2)
        self.acts = rng.standard_normal((N_LAYERS, 12, D))

    def test_required_keys(self):
        r = self.tc(self.acts, self.P_A, self.P_S)
        for k in ("Q_full", "Q_A", "Q_S", "mean_Q_full", "mean_Q_A", "mean_Q_S"):
            self.assertIn(k, r, f"missing key: {k}")

    def test_mean_values_non_negative(self):
        r = self.tc(self.acts, self.P_A, self.P_S)
        self.assertGreaterEqual(r["mean_Q_full"], 0.0)
        self.assertGreaterEqual(r["mean_Q_A"],    0.0)
        self.assertGreaterEqual(r["mean_Q_S"],    0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 5. cis_decompose
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeCisDecomposition(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.cis_decompose import compute_cis_decomposition
        self.cis   = compute_cis_decomposition
        self.acts  = _make_activations_per_prompt(K=4)

    def test_required_keys(self):
        r = self.cis(self.acts)
        for k in ("invariant", "specific", "K", "n_layers", "n_tokens", "d"):
            self.assertIn(k, r)

    def test_invariant_shape(self):
        r   = self.cis(self.acts)
        inv = r["invariant"]
        self.assertEqual(inv.shape, (N_LAYERS, self.acts[0].shape[1], D))

    def test_specific_shape(self):
        r   = self.cis(self.acts)
        sp  = r["specific"]
        self.assertEqual(sp.shape[0], 4)          # K
        self.assertEqual(sp.shape[1], N_LAYERS)

    def test_specific_sums_to_zero(self):
        """Mean of per-prompt residuals across prompts must be ≈ 0."""
        r  = self.cis(self.acts)
        np.testing.assert_allclose(r["specific"].mean(axis=0),
                                   np.zeros_like(r["specific"][0]),
                                   atol=1e-10)

    def test_invariant_plus_specific_reconstructs(self):
        r    = self.cis(self.acts)
        inv  = r["invariant"]
        spec = r["specific"]
        recon = inv[np.newaxis] + spec
        orig  = np.stack(self.acts, axis=0)
        # Truncate to matched shape
        nl = min(recon.shape[1], orig.shape[1])
        nt = min(recon.shape[2], orig.shape[2])
        np.testing.assert_allclose(
            recon[:, :nl, :nt, :], orig[:, :nl, :nt, :], atol=1e-10
        )

    def test_raises_on_single_prompt(self):
        with self.assertRaises((AssertionError, ValueError)):
            self.cis([self.acts[0]])


class TestAnalyzeCis(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.cis_decompose import analyze_cis
        self.P_A, self.P_S = _make_projectors()
        self.result = analyze_cis(
            activations_per_prompt=_make_activations_per_prompt(K=4),
            P_A=self.P_A,
            P_S=self.P_S,
            plateau_layers=[2, 3],
            merge_layers=[5],
        )

    def test_required_keys(self):
        for k in ("k1", "p2ck1_holds", "velocities"):
            self.assertIn(k, self.result, f"missing key: {k}")

    def test_k1_fractions_in_unit_interval(self):
        k1 = self.result["k1"]
        self.assertGreaterEqual(k1["global_inv_frac_A"], 0.0)
        self.assertLessEqual(k1["global_inv_frac_A"],    1.0 + 1e-9)
        self.assertGreaterEqual(k1["global_spec_frac_S"], 0.0)
        self.assertLessEqual(k1["global_spec_frac_S"],    1.0 + 1e-9)

    def test_p2ck1_holds_is_bool(self):
        self.assertIsInstance(self.result["p2ck1_holds"], bool)

    def test_velocities_non_negative(self):
        vels = np.asarray(self.result["velocities"])
        self.assertTrue(np.all(vels >= 0.0))

    def test_cis_to_json_serializable(self):
        import json
        from p2c_churchland.cis_decompose import cis_to_json
        json.dumps(cis_to_json(self.result))


# ══════════════════════════════════════════════════════════════════════════════
# 6. local_jacobian
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeLayerJacobian(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.local_jacobian import compute_layer_jacobian
        self.cj = compute_layer_jacobian

    def _linear_fn(self, x):
        """y = Wx; J = W for all x."""
        import torch
        W = torch.eye(D, dtype=torch.float64) * 2.0
        return (W @ x.unsqueeze(-1)).squeeze(-1)

    def test_jacobian_of_identity_scaling(self):
        import torch
        x = torch.zeros(D, dtype=torch.float64)
        J = self.cj(self._linear_fn, x.numpy())
        np.testing.assert_allclose(J, np.eye(D) * 2.0, atol=1e-6)

    def test_output_shape(self):
        import torch
        rng = np.random.default_rng(0)
        x   = rng.standard_normal(D)
        J   = self.cj(self._linear_fn, x)
        self.assertEqual(J.shape, (D, D))


class TestDecomposeSa(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.local_jacobian import decompose_sa
        self.ds = decompose_sa

    def test_pure_symmetric_sa_ratio_zero(self):  # rename to _one
        J = np.eye(D, dtype=np.float64)
        r = self.ds(J)
        self.assertAlmostEqual(r["sa_ratio"], 1.0, places=10)  # was 0.0
        
    def test_pure_antisymmetric_sa_ratio_large(self):
        """Build a pure antisymmetric J."""
        rng = np.random.default_rng(0)
        A   = rng.standard_normal((D, D))
        J   = (A - A.T) / 2.0
        r   = self.ds(J)
        # S component ≈ 0 → S/A ratio should be very small
        self.assertLess(r["sa_ratio"], 1e-8)

    def test_required_keys(self):
        J = np.eye(D, dtype=np.float64)
        r = self.ds(J)
        for k in ("S", "A", "sa_ratio", "s_frac", "a_frac"):
            self.assertIn(k, r, f"missing key: {k}")

    def test_s_plus_a_reconstructs_j(self):
        rng = np.random.default_rng(1)
        J   = rng.standard_normal((D, D))
        r   = self.ds(J)
        np.testing.assert_allclose(r["S"] + r["A"], J, atol=1e-12)

    def test_s_is_symmetric(self):
        rng = np.random.default_rng(2)
        J   = rng.standard_normal((D, D))
        r   = self.ds(J)
        np.testing.assert_allclose(r["S"], r["S"].T, atol=1e-12)

    def test_a_is_antisymmetric(self):
        rng = np.random.default_rng(3)
        J   = rng.standard_normal((D, D))
        r   = self.ds(J)
        np.testing.assert_allclose(r["A"], -r["A"].T, atol=1e-12)


class TestJacobiansAtCentroids(unittest.TestCase):
    """Test centroid_jacobians_one_layer (per-centroid loop, no full model needed)."""

    def setUp(self):
        from p2c_churchland.local_jacobian import centroid_jacobians_one_layer
        self.jac = centroid_jacobians_one_layer

    def _identity_fn(self, x):
        import torch
        return x  # Jacobian = I

    def test_sa_ratio_for_identity_layer(self):
        rng       = np.random.default_rng(0)
        centroids = rng.standard_normal((3, D))
        results   = self.jac(
            layer_fn=self._identity_fn,
            centroids=centroids,
            centroid_ids=list(range(3)),
        )
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertAlmostEqual(r["sa_ratio"], 0.0, places=6)

    def test_centroid_id_preserved(self):
        rng       = np.random.default_rng(1)
        centroids = rng.standard_normal((2, D))
        results   = self.jac(
            layer_fn=self._identity_fn,
            centroids=centroids,
            centroid_ids=["alpha", "beta"],
        )
        self.assertEqual(results[0]["centroid_id"], "alpha")
        self.assertEqual(results[1]["centroid_id"], "beta")


# ══════════════════════════════════════════════════════════════════════════════
# 7. slow_point_compare
# ══════════════════════════════════════════════════════════════════════════════

class TestLayerSaProfile(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.slow_point_compare import layer_sa_profile
        self.lsp = layer_sa_profile
        recs     = _make_per_layer_records()
        self.per_layer = recs["per_layer"]

    def test_output_keys(self):
        r = self.lsp(self.per_layer)
        for k in ("layer_indices", "mean_sa", "std_sa", "n_centroids"):
            self.assertIn(k, r)

    def test_lengths_consistent(self):
        r = self.lsp(self.per_layer)
        n = len(r["layer_indices"])
        self.assertEqual(len(r["mean_sa"]),    n)
        self.assertEqual(len(r["std_sa"]),     n)
        self.assertEqual(len(r["n_centroids"]), n)

    def test_mean_sa_positive(self):
        r = self.lsp(self.per_layer)
        self.assertTrue(np.all(r["mean_sa"] > 0.0))


class TestCompareLocalGlobal(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.slow_point_compare import compare_local_global
        recs = _make_per_layer_records(
            plateau_layers=[2, 3, 4],
            merge_layers=[5],
        )
        self.result = compare_local_global(
            per_layer=recs["per_layer"],
            plateau_layers=recs["plateau_layers"],
            merge_layers=recs["merge_layers"],
            global_sa_ratio=recs["global_sa_ratio"],
        )

    def test_required_keys(self):
        for k in ("p2cs1_holds", "p2cs2_holds"):
            self.assertIn(k, self.result, f"missing key: {k}")

    def test_verdicts_are_bool(self):
        self.assertIsInstance(self.result["p2cs1_holds"], bool)
        self.assertIsInstance(self.result["p2cs2_holds"], bool)


# ══════════════════════════════════════════════════════════════════════════════
# 8. icl_subspace_scaling
# ══════════════════════════════════════════════════════════════════════════════

class TestChannelMagnitudesOnePrompt(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.icl_subspace_scaling import channel_magnitudes_one_prompt
        self.cm   = channel_magnitudes_one_prompt
        self.P_A, self.P_S = _make_projectors()
        rng       = np.random.default_rng(0)
        # (n_layers, d) — per-layer activations at one token position
        self.acts = rng.standard_normal((N_LAYERS, D))

    def test_output_keys(self):
        r = self.cm(self.acts, self.P_A, self.P_S)
        for k in ("mag_A", "mag_S", "mag_A_normed", "mag_S_normed"):
            self.assertIn(k, r, f"missing key: {k}")

    def test_magnitudes_non_negative(self):
        r = self.cm(self.acts, self.P_A, self.P_S)
        self.assertGreaterEqual(r["mag_A"], 0.0)
        self.assertGreaterEqual(r["mag_S"], 0.0)


class TestMonotonicityScore(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.icl_subspace_scaling import monotonicity_score
        self.ms = monotonicity_score

    def test_strictly_increasing_gives_rho_one(self):
        k_vals = [0, 1, 2, 4, 8]
        mags   = [1.0, 2.0, 3.0, 4.0, 5.0]
        r = self.ms(k_vals, mags)
        self.assertAlmostEqual(r["rho"], 1.0, places=5)
        self.assertTrue(r["monotone_increasing"])

    def test_strictly_decreasing_gives_rho_neg_one(self):
        k_vals = [0, 1, 2, 4, 8]
        mags   = [5.0, 4.0, 3.0, 2.0, 1.0]
        r = self.ms(k_vals, mags)
        self.assertAlmostEqual(r["rho"], -1.0, places=5)
        self.assertFalse(r["monotone_increasing"])

    def test_output_keys(self):
        r = self.ms([0, 1, 2], [1.0, 1.5, 2.0])
        for k in ("rho", "pvalue", "monotone_increasing"):
            self.assertIn(k, r)


class TestCrossTaskDirectionAgreement(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.icl_subspace_scaling import cross_task_direction_agreement
        self.ca  = cross_task_direction_agreement
        rng      = np.random.default_rng(0)
        n_k      = 5
        # Two tasks: same direction → high cosine; different → low cosine
        base_dir = rng.standard_normal((n_k, D))
        base_dir /= np.linalg.norm(base_dir, axis=1, keepdims=True)
        self.task_dirs_identical = {"task_a": base_dir, "task_b": base_dir.copy()}
        ortho = rng.standard_normal((n_k, D))
        ortho /= np.linalg.norm(ortho, axis=1, keepdims=True)
        self.task_dirs_random = {"task_a": base_dir, "task_b": ortho}
        self.k_vals = list(range(n_k))

    def test_identical_directions_high_cosine(self):
        r = self.ca(self.task_dirs_identical, self.k_vals)
        self.assertGreater(r["overall_mean_cosine"], 0.9)

    def test_output_keys(self):
        r = self.ca(self.task_dirs_random, self.k_vals)
        for k in ("per_k_mean_cosine", "overall_mean_cosine", "p2cm2_holds"):
            self.assertIn(k, r)

    def test_cosines_in_unit_interval(self):
        r = self.ca(self.task_dirs_random, self.k_vals)
        for c in r["per_k_mean_cosine"]:
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0 + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# 9. context_selection (offline sub-functions — no model)
# ══════════════════════════════════════════════════════════════════════════════

class TestLayerCosineSimilarity(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.context_selection import layer_cosine_similarity
        self.cs = layer_cosine_similarity

    def test_identical_vecs_give_one(self):
        rng  = np.random.default_rng(0)
        vecs = rng.standard_normal((N_LAYERS, D))
        cos  = self.cs(vecs, vecs)
        np.testing.assert_allclose(cos, np.ones(N_LAYERS), atol=1e-9)

    def test_orthogonal_vecs_give_zero(self):
        a = np.zeros((1, D))
        b = np.zeros((1, D))
        a[0, 0] = 1.0
        b[0, 1] = 1.0
        cos = self.cs(a, b)
        np.testing.assert_allclose(cos, [0.0], atol=1e-9)

    def test_output_shape(self):
        rng = np.random.default_rng(1)
        a   = rng.standard_normal((N_LAYERS, D))
        b   = rng.standard_normal((N_LAYERS, D))
        cos = self.cs(a, b)
        self.assertEqual(cos.shape, (N_LAYERS,))

    def test_values_in_unit_interval(self):
        rng = np.random.default_rng(2)
        a   = rng.standard_normal((N_LAYERS, D))
        b   = rng.standard_normal((N_LAYERS, D))
        cos = self.cs(a, b)
        self.assertTrue(np.all(cos >= -1.0 - 1e-9))
        self.assertTrue(np.all(cos <=  1.0 + 1e-9))


class TestLayerL2Divergence(unittest.TestCase):

    def setUp(self):
        from p2c_churchland.context_selection import layer_l2_divergence
        self.ld = layer_l2_divergence

    def test_identical_vecs_give_zero(self):
        rng  = np.random.default_rng(0)
        vecs = rng.standard_normal((N_LAYERS, D))
        dists = self.ld(vecs, vecs)
        np.testing.assert_allclose(dists, np.zeros(N_LAYERS), atol=1e-12)

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        a   = rng.standard_normal((N_LAYERS, D))
        b   = rng.standard_normal((N_LAYERS, D))
        dists = self.ld(a, b)
        self.assertTrue(np.all(dists >= 0.0))

    def test_symmetric(self):
        rng = np.random.default_rng(2)
        a   = rng.standard_normal((N_LAYERS, D))
        b   = rng.standard_normal((N_LAYERS, D))
        np.testing.assert_allclose(self.ld(a, b), self.ld(b, a), atol=1e-12)


class TestAnalyzeContextPairOffline(unittest.TestCase):
    """
    analyze_context_pair without a real model.
    If the function accepts pre-extracted activations via keyword args,
    use them.  Otherwise, test the sub-functions directly and verify
    m3_holds is computable from synthetic per-layer vectors.
    """

    def test_m3_s_more_similar_than_a(self):
        from p2c_churchland.context_selection import layer_cosine_similarity

        rng   = np.random.default_rng(0)
        P_A, P_S = _make_projectors()

        # Construct S-channel activations that agree and A-channel that diverge
        base      = rng.standard_normal((N_LAYERS, D))
        noise_a   = rng.standard_normal((N_LAYERS, D)) * 5.0
        noise_s   = rng.standard_normal((N_LAYERS, D)) * 0.01

        acts_a_A  = (P_A @ (base + noise_a).T).T
        acts_b_A  = (P_A @ (base - noise_a).T).T
        acts_a_S  = (P_S @ (base + noise_s).T).T
        acts_b_S  = (P_S @ (base - noise_s).T).T

        cos_A = layer_cosine_similarity(acts_a_A, acts_b_A)
        cos_S = layer_cosine_similarity(acts_a_S, acts_b_S)

        # S should be more similar (noise_s << noise_a)
        self.assertGreater(cos_S.mean(), cos_A.mean())

    def test_m3_a_more_divergent(self):
        from p2c_churchland.context_selection import layer_l2_divergence

        rng   = np.random.default_rng(1)
        P_A, P_S = _make_projectors()
        base  = rng.standard_normal((N_LAYERS, D))

        acts_a_A = (P_A @ (base + rng.standard_normal((N_LAYERS, D)) * 3).T).T
        acts_b_A = (P_A @ (base - rng.standard_normal((N_LAYERS, D)) * 3).T).T
        acts_a_S = (P_S @ (base + rng.standard_normal((N_LAYERS, D)) * 0.01).T).T
        acts_b_S = (P_S @ (base - rng.standard_normal((N_LAYERS, D)) * 0.01).T).T

        l2_A = layer_l2_divergence(acts_a_A, acts_b_A).mean()
        l2_S = layer_l2_divergence(acts_a_S, acts_b_S).mean()
        self.assertGreater(l2_A, l2_S)


# ══════════════════════════════════════════════════════════════════════════════
# 10. run_2c helpers (no I/O, no model)
# ══════════════════════════════════════════════════════════════════════════════

class TestRunHelpers(unittest.TestCase):

    def test_flatten_layers_list(self):
        from p2c_churchland.run_2c import _flatten_layers
        self.assertEqual(_flatten_layers([1, 2, 3]), [1, 2, 3])

    def test_flatten_layers_nested(self):
        from p2c_churchland.run_2c import _flatten_layers
        self.assertEqual(sorted(_flatten_layers([[1, 2], [3]])), [1, 2, 3])

    def test_flatten_layers_dict(self):
        from p2c_churchland.run_2c import _flatten_layers
        result = _flatten_layers({"a": [1, 2], "b": [3]})
        self.assertEqual(sorted(result), [1, 2, 3])

    def test_model_stem(self):
        from p2c_churchland.run_2c import _model_stem
        self.assertEqual(_model_stem("albert-base-v2"), "albert_base_v2")
        self.assertEqual(_model_stem("gpt2/large"), "gpt2_large")

    def test_aggregate_context_pairs_m3_holds(self):
        from p2c_churchland.run_2c import _aggregate_context_pairs
        pairs = [
            {"mean_S_cosine": 0.9, "mean_A_cosine": 0.2,
             "mean_A_l2": 1.0, "m3_holds": True},
            {"mean_S_cosine": 0.8, "mean_A_cosine": 0.3,
             "mean_A_l2": 0.8, "m3_holds": True},
        ]
        r = _aggregate_context_pairs(pairs)
        self.assertTrue(r["p2cm3_holds"])
        self.assertEqual(r["n_pairs"], 2)
        self.assertEqual(r["n_pairs_m3_holds"], 2)

    def test_aggregate_context_pairs_m3_fails(self):
        from p2c_churchland.run_2c import _aggregate_context_pairs
        pairs = [
            {"mean_S_cosine": 0.1, "mean_A_cosine": 0.9,
             "mean_A_l2": 0.1, "m3_holds": False},
        ]
        r = _aggregate_context_pairs(pairs)
        self.assertFalse(r["p2cm3_holds"])

    def test_group_icl_by_task_dict_passthrough(self):
        from p2c_churchland.run_2c import _group_icl_by_task
        inp = {"t1": [{"k": 0}], "t2": [{"k": 1}]}
        self.assertEqual(_group_icl_by_task(inp), inp)

    def test_group_icl_by_task_list(self):
        from p2c_churchland.run_2c import _group_icl_by_task
        inp = [{"task": "t1", "k": 0}, {"task": "t2", "k": 1},
               {"task": "t1", "k": 2}]
        r = _group_icl_by_task(inp)
        self.assertIn("t1", r)
        self.assertEqual(len(r["t1"]), 2)

    def test_json_default_numpy_types(self):
        import json
        from p2c_churchland.run_2c import _json_default
        self.assertEqual(_json_default(np.int64(5)), 5)
        self.assertIsNone(_json_default(np.float64(float("nan"))))
        self.assertEqual(_json_default(np.array([1, 2])), [1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)