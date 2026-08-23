"""
tests/test_phase2b_schur.py — Block 1a after the rewrite.

Four defect classes are pinned here so a regression is a test failure rather
than another accidental discovery: the memory footprint, the two-conventions
problem, the folded rotation angle, and the absolute subdiagonal threshold.
"""

import json
import unittest

import numpy as np

from p2b_imaginary import rotational_schur as rs

import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure

def block_diag_rotations(thetas, rhos, d_pad=0):
    """Exact block-diagonal matrix of 2x2 scaled rotations, plus real padding."""
    blocks = []
    for th, rho in zip(thetas, rhos):
        c, s = np.cos(th), np.sin(th)
        blocks.append(rho * np.array([[c, -s], [s, c]]))
    d = 2 * len(thetas) + d_pad
    M = np.zeros((d, d))
    i = 0
    for B in blocks:
        M[i:i + 2, i:i + 2] = B
        i += 2
    return M


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestNoDenseProjectorsRetained(unittest.TestCase):
    """
    The previous version kept `top_k=32` dense (d, d) projectors plus two
    combined ones, per layer, on top of `schur_T` and `schur_Z`: ~7 GB at
    d=1024 x 24 layers, ~27 GB at d=2048.
    """

    def _big_arrays(self, obj, d, found=None):
        found = [] if found is None else found
        if isinstance(obj, np.ndarray):
            if obj.ndim == 2 and obj.shape == (d, d):
                found.append(obj.shape)
        elif isinstance(obj, dict):
            for v in obj.values():
                self._big_arrays(v, d, found)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                self._big_arrays(v, d, found)
        return found

    def test_layer_scalars_retains_no_d_by_d_array(self):
        d = 24
        V = np.random.default_rng(0).normal(size=(d, d)) / np.sqrt(d)
        rec = rs.layer_scalars(V, "layer_0", top_k=8)
        self.assertEqual(self._big_arrays(rec, d), [])

    def test_planes_are_bases_not_projectors(self):
        d = 24
        V = np.random.default_rng(1).normal(size=(d, d)) / np.sqrt(d)
        planes = rs.top_rotation_planes(rs.extract_schur_blocks(V), top_k=5)
        for B in planes["bases"]:
            self.assertEqual(B.shape, (d, 2))
            np.testing.assert_allclose(B.T @ B, np.eye(2), atol=1e-10)

    def test_schur_factors_dropped_unless_requested(self):
        V = np.random.default_rng(2).normal(size=(12, 12))
        self.assertNotIn("schur_T", rs.extract_schur_blocks(V))
        self.assertIn("schur_T", rs.extract_schur_blocks(V, keep_factors=True))

    def test_full_analysis_retains_no_d_by_d_array(self):
        d, n_layers = 20, 4
        rng = np.random.default_rng(3)
        ov = {"ov_total": [rng.normal(size=(d, d)) / np.sqrt(d)
                           for _ in range(n_layers)],
              "is_per_layer": True,
              "layer_names": [f"layer_{i}" for i in range(n_layers)]}
        res = rs.analyze_rotational_spectrum(ov)
        self.assertEqual(self._big_arrays(res, d), [])


# ---------------------------------------------------------------------------
# One energy convention
# ---------------------------------------------------------------------------

class TestEnergyConvention(unittest.TestCase):

    def test_real_plus_complex_is_an_identity(self):
        V = np.random.default_rng(4).normal(size=(30, 30))
        e = rs.complex_energy_fraction(rs.extract_schur_blocks(V))
        self.assertAlmostEqual(e["complex_energy"] + e["real_energy"],
                               e["eigenvalue_energy"], places=8)
        self.assertAlmostEqual(e["complex_energy_fraction"] +
                               e["real_energy_fraction"], 1.0, places=10)

    def test_eigenvalue_energy_matches_the_eigenvalues(self):
        """Per-eigenvalue accounting must equal sum |lambda|^2 over eigvals."""
        V = np.random.default_rng(5).normal(size=(24, 24))
        e = rs.complex_energy_fraction(rs.extract_schur_blocks(V))
        direct = float(np.sum(np.abs(np.linalg.eigvals(V)) ** 2))
        self.assertAlmostEqual(e["eigenvalue_energy"], direct, places=6)

    def test_symmetric_matrix_is_all_real(self):
        A = np.random.default_rng(6).normal(size=(16, 16))
        S = (A + A.T) / 2
        e = rs.complex_energy_fraction(rs.extract_schur_blocks(S))
        self.assertEqual(e["n_complex"], 0)
        self.assertAlmostEqual(e["complex_energy_fraction"], 0.0, places=10)

    def test_antisymmetric_matrix_is_all_complex(self):
        A = np.random.default_rng(7).normal(size=(16, 16))
        A = (A - A.T) / 2
        e = rs.complex_energy_fraction(rs.extract_schur_blocks(A))
        self.assertEqual(e["n_real"], 0)
        self.assertAlmostEqual(e["complex_energy_fraction"], 1.0, places=10)

    def test_legacy_convention_is_reproducible_and_lower(self):
        """
        The historical 84-97.5% figure counted rho^2 once per 2x2 block
        against lambda^2 once per 1x1 block, mixing a per-pair total with a
        per-eigenvalue one. Kept under its own name so a rerun can state both.
        """
        V = np.random.default_rng(8).normal(size=(32, 32)) / np.sqrt(32)
        blocks = rs.extract_schur_blocks(V)
        legacy = rs.rotational_fraction_per_block(blocks)
        current = rs.complex_energy_fraction(blocks)["complex_energy_fraction"]
        self.assertLess(legacy, current)

    def test_henrici_is_zero_for_a_normal_matrix(self):
        rng = np.random.default_rng(9)
        Q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
        S = Q @ np.diag(rng.normal(size=16)) @ Q.T      # symmetric => normal
        h = rs.henrici_nonnormality(rs.extract_schur_blocks(S))
        self.assertLess(h["henrici_relative"], 1e-10)

    def test_henrici_unclamped_is_reported(self):
        """A materially negative unclamped value is a block-parse bug signal."""
        V = np.random.default_rng(10).normal(size=(16, 16))
        h = rs.henrici_nonnormality(rs.extract_schur_blocks(V))
        self.assertIn("henrici_absolute_unclamped", h)
        self.assertGreater(h["henrici_absolute_unclamped"], -1e-6)


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------

class TestRotationAngles(unittest.TestCase):

    def test_known_angles_and_moduli_recovered(self):
        thetas = [0.3, 1.2, 2.7]
        rhos = [0.5, 1.0, 2.0]
        M = block_diag_rotations(thetas, rhos)
        blocks = rs.extract_schur_blocks(M)
        self.assertEqual(blocks["n_complex"], 3)
        got_t = sorted(b["theta"] for b in blocks["blocks_2x2"])
        got_r = sorted(b["rho"] for b in blocks["blocks_2x2"])
        np.testing.assert_allclose(got_t, sorted(thetas), atol=1e-8)
        np.testing.assert_allclose(got_r, sorted(rhos), atol=1e-8)

    def test_obtuse_angle_is_not_folded(self):
        """
        `arctan2(sqrt(-bc), abs(a))` mapped theta = 2.7 onto pi - 2.7 = 0.44.
        The sign field survived, so nothing was lost — but `theta_mean` was
        not the mean rotation angle, which matters as soon as theta is
        regressed against depth or against checkpoint step.
        """
        M = block_diag_rotations([2.7], [1.0])
        b = rs.extract_schur_blocks(M)["blocks_2x2"][0]
        self.assertAlmostEqual(b["theta"], 2.7, places=8)
        self.assertGreater(b["theta"], np.pi / 2)
        self.assertEqual(b["sign"], -1)

    def test_theta_spans_the_full_range(self):
        thetas = [0.2, 1.5, 3.0]
        M = block_diag_rotations(thetas, [1.0] * 3)
        st = rs.rotation_angle_stats(rs.extract_schur_blocks(M))
        self.assertGreater(st["theta_max"], np.pi / 2)
        self.assertAlmostEqual(st["theta_mean"], float(np.mean(thetas)), places=8)

    def test_repulsive_fraction_reads_the_real_part_not_the_modulus(self):
        """
        `e^{-V}` grows where Re lambda < 0. `rho > 1` is a threshold on a
        scale convention, not on a dynamical property; both are returned,
        under names that separate them.
        """
        M = block_diag_rotations([2.9, 2.9, 0.2], [0.1, 0.1, 5.0])
        st = rs.rotation_angle_stats(rs.extract_schur_blocks(M))
        self.assertAlmostEqual(st["frac_repulsive_real_part"], 2 / 3, places=8)
        self.assertAlmostEqual(st["frac_rho_above_one"], 1 / 3, places=8)


# ---------------------------------------------------------------------------
# Block parsing edge cases
# ---------------------------------------------------------------------------

class TestBlockParsing(unittest.TestCase):

    def test_blocks_partition_the_dimension(self):
        for seed in range(4):
            V = np.random.default_rng(seed).normal(size=(21, 21))
            b = rs.extract_schur_blocks(V)
            self.assertEqual(b["n_real"] + 2 * b["n_complex"], b["d"])

    def test_subdiagonal_threshold_is_scale_invariant(self):
        """
        The old test was `abs(T[i+1, i]) > 1e-10`, absolute, on a matrix whose
        norm varies by orders of magnitude across layers and checkpoints. At
        small scale every block reads 1x1 and the matrix looks entirely real.
        """
        V = np.random.default_rng(11).normal(size=(20, 20))
        base = rs.extract_schur_blocks(V)["n_complex"]
        for scale in (1e-11, 1e-10, 1e-8, 1e-4, 1e4, 1e8):
            self.assertEqual(rs.extract_schur_blocks(V * scale)["n_complex"],
                             base, f"scale {scale}")

    def test_absolute_threshold_would_have_failed(self):
        """
        Pins the bug being fixed. The total-loss case (scale 1e-11, every
        block read as 1x1, matrix looks entirely real) is the obvious one;
        the PARTIAL case at 1e-10 is worse, because it mis-parses some blocks
        and not others and the output still looks like a plausible spectrum.
        """
        from scipy.linalg import schur
        V = np.random.default_rng(11).normal(size=(20, 20))
        truth = rs.extract_schur_blocks(V)["n_complex"]
        self.assertEqual(truth, 8)

        def naive_count(M):
            T, _ = schur(M, output="real")
            return sum(1 for i in range(M.shape[0] - 1)
                       if abs(T[i + 1, i]) > 1e-10)

        self.assertEqual(naive_count(V * 1e-11), 0)         # total loss
        self.assertEqual(naive_count(V * 1e-10), 6)         # partial, plausible
        self.assertEqual(naive_count(V), truth)             # fine at unit scale

        for scale in (1e-11, 1e-10):
            self.assertEqual(rs.extract_schur_blocks(V * scale)["n_complex"],
                             truth)

    def test_odd_dimension_leaves_a_real_block(self):
        b = rs.extract_schur_blocks(np.random.default_rng(12).normal(size=(7, 7)))
        self.assertGreaterEqual(b["n_real"], 1)

    def test_identity_is_all_real(self):
        b = rs.extract_schur_blocks(np.eye(8))
        self.assertEqual(b["n_complex"], 0)
        self.assertEqual(b["n_real"], 8)


# ---------------------------------------------------------------------------
# Plane operations
# ---------------------------------------------------------------------------

class TestPlaneOperations(unittest.TestCase):

    def test_projection_matches_the_dense_projector(self):
        d, n = 20, 15
        rng = np.random.default_rng(13)
        V = rng.normal(size=(d, d))
        planes = rs.top_rotation_planes(rs.extract_schur_blocks(V), top_k=4)
        X = rng.normal(size=(n, d))
        got = rs.project_onto_planes(X, planes["bases"])
        for j, B in enumerate(planes["bases"]):
            P = B @ B.T                        # the (d, d) we refuse to store
            want = np.sum((X @ P) * X, axis=1)
            np.testing.assert_allclose(got[:, j], want, atol=1e-9)

    def test_plane_energy_matches_the_dense_form(self):
        d = 16
        rng = np.random.default_rng(14)
        V = rng.normal(size=(d, d))
        M = rng.normal(size=(d, d))
        planes = rs.top_rotation_planes(rs.extract_schur_blocks(V), top_k=3)
        got = rs.plane_energy(M, planes["bases"])
        for j, B in enumerate(planes["bases"]):
            self.assertAlmostEqual(got[j], float(np.sum((B.T @ M @ B) ** 2)),
                                   places=9)

    def test_subspace_fraction_is_one_inside_the_span(self):
        d = 16
        V = np.random.default_rng(15).normal(size=(d, d))
        blocks = rs.extract_schur_blocks(V)
        B = np.concatenate([b["plane"] for b in blocks["blocks_2x2"]], axis=1)
        X = (np.random.default_rng(16).normal(size=(5, B.shape[1])) @ B.T)
        self.assertAlmostEqual(rs.rotation_subspace_fraction(X, blocks), 1.0,
                               places=8)

    def test_subspace_fraction_is_zero_in_the_real_complement(self):
        d = 16
        V = np.random.default_rng(17).normal(size=(d, d))
        blocks = rs.extract_schur_blocks(V)
        if blocks["n_real"] == 0:
            self.skipTest("no real subspace in this draw")
        R = np.column_stack([b["schur_vec"] for b in blocks["blocks_1x1"]])
        X = (np.random.default_rng(18).normal(size=(5, R.shape[1])) @ R.T)
        self.assertLess(rs.rotation_subspace_fraction(X, blocks), 1e-12)

    def test_empty_plane_list_is_handled(self):
        S = np.eye(6)
        blocks = rs.extract_schur_blocks(S)
        self.assertEqual(rs.project_onto_planes(np.ones((3, 6)), []).shape, (3, 0))
        self.assertEqual(rs.rotation_subspace_fraction(np.ones((3, 6)), blocks), 0.0)


# ---------------------------------------------------------------------------
# Nulls
# ---------------------------------------------------------------------------

class TestNulls(unittest.TestCase):

    def test_null_matrices_are_norm_matched(self):
        V = np.random.default_rng(19).normal(size=(16, 16)) * 3.0
        target = np.linalg.norm(V, "fro")
        for M in rs.gaussian_null_matrices(V, n_draws=4):
            self.assertAlmostEqual(np.linalg.norm(M, "fro"), target, places=8)
            self.assertEqual(M.shape, V.shape)

    def test_complex_fraction_does_not_separate_gaussian_from_gaussian(self):
        """
        The control the phase never had. A Gaussian matrix is essentially all
        complex pairs, so a Gaussian observation must sit inside the Gaussian
        null — which is the point: if trained OV also sits there, the
        84-97% headline is a statement about square matrices.
        """
        V = np.random.default_rng(20).normal(size=(24, 24))
        res = rs.null_comparison(V, "complex_energy_fraction", n_draws=12)
        self.assertLess(abs(res["z_score"]), 4.0)
        self.assertEqual(res["null_construction"], "norm_matched_gaussian")

    def test_symmetric_matrix_is_far_from_the_null(self):
        """A statistic that DOES separate: a symmetric V has no rotation."""
        A = np.random.default_rng(21).normal(size=(24, 24))
        S = (A + A.T) / 2
        res = rs.null_comparison(S, "complex_energy_fraction", n_draws=12)
        self.assertEqual(res["observed"], 0.0)
        self.assertLess(res["percentile"], 5.0)

    def test_null_result_carries_the_sigma_from_null_schema(self):
        V = np.random.default_rng(22).normal(size=(12, 12))
        res = rs.null_comparison(V, "henrici_relative", n_draws=6)
        for k in ("observed", "null_mean", "null_std", "z_score",
                  "percentile", "n_null"):
            self.assertIn(k, res)


# ---------------------------------------------------------------------------
# Pipeline and serialization
# ---------------------------------------------------------------------------

class TestPipeline(unittest.TestCase):

    def _ov(self, d=16, n_layers=3, step=512):
        rng = np.random.default_rng(23)
        return {
            "ov_total": [rng.normal(size=(d, d)) / np.sqrt(d)
                         for _ in range(n_layers)],
            "is_per_layer": True,
            "layer_names": [f"layer_{i}" for i in range(n_layers)],
            "model_stem": f"pythia-410m-step{step}",
            "checkpoint_step": step,
        }

    def test_checkpoint_step_survives_to_the_artifact(self):
        """Without it nothing can place the result on the training axis."""
        js = rs.summary_to_json(rs.analyze_rotational_spectrum(self._ov()))
        self.assertEqual(js["checkpoint_step"], 512)
        self.assertEqual(js["model_stem"], "pythia-410m-step512")

    def test_shared_weight_model(self):
        d = 12
        ov = {"ov_total": np.random.default_rng(24).normal(size=(d, d)),
              "is_per_layer": False, "layer_names": ["shared"]}
        res = rs.analyze_rotational_spectrum(ov)
        self.assertEqual(len(res["per_layer"]), 1)

    def test_summary_reports_the_argmax_layer_by_name(self):
        res = rs.analyze_rotational_spectrum(self._ov())
        self.assertIn(res["summary"]["henrici_argmax_layer"], res["layer_names"])

    def test_planes_are_dropped_by_the_serializer(self):
        res = rs.analyze_rotational_spectrum(self._ov(), top_k_planes=4)
        self.assertIn("planes", res["per_layer"][0])
        js = rs.summary_to_json(res)
        self.assertNotIn("planes", js["per_layer"][0])
        json.dumps(js)          # must not raise

    def test_nulls_are_off_by_default(self):
        res = rs.analyze_rotational_spectrum(self._ov())
        self.assertNotIn("nulls", res["per_layer"][0])

    def test_nulls_run_when_requested(self):
        res = rs.analyze_rotational_spectrum(
            self._ov(d=10, n_layers=1), with_nulls=True, n_null_draws=4,
            null_statistics=("complex_energy_fraction",),
        )
        self.assertIn("complex_energy_fraction", res["per_layer"][0]["nulls"])

    def test_summary_lines_render(self):
        js = rs.summary_to_json(rs.analyze_rotational_spectrum(self._ov()))
        lines = rs.summary_lines(js)
        self.assertTrue(any("Complex energy fraction" in l for l in lines))
        self.assertTrue(any("Henrici" in l for l in lines))


# ---------------------------------------------------------------------------
# The third definition
# ---------------------------------------------------------------------------

class TestRelativeCriterion(unittest.TestCase):
    """
    `complex_energy_fraction_relative` is the eigenvalue-based criterion
    `layernorm_jacobian.rotational_fraction` uses and `core/precision_policy.py`
    imports. Given a home here so the phase has one place per definition.
    """

    def test_matches_the_layernorm_jacobian_definition(self):
        V = np.random.default_rng(25).normal(size=(20, 20))
        eigs = np.linalg.eigvals(V)
        is_cx = np.abs(np.imag(eigs)) > 0.01 * (np.abs(np.real(eigs)) + 1e-12)
        want = float(np.sum(np.abs(eigs[is_cx]) ** 2) /
                     np.sum(np.abs(eigs) ** 2))
        self.assertAlmostEqual(rs.complex_energy_fraction_relative(V), want,
                               places=12)

    def test_is_tolerance_sensitive_which_is_the_point_of_item_p2(self):
        """
        A relative criterion is what an fp16-epsilon split of a genuinely
        real eigenvalue pair defeats. If the value moves across the tolerance
        sweep, the headline is a threshold artifact.
        """
        rng = np.random.default_rng(26)
        Q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
        S = Q @ np.diag(rng.normal(size=20)) @ Q.T          # exactly real
        S_fp16 = S.astype(np.float16).astype(np.float64)    # the storage dtype
        loose = rs.complex_energy_fraction_relative(S_fp16, tol=1e-6)
        tight = rs.complex_energy_fraction_relative(S_fp16, tol=1.0)
        self.assertGreaterEqual(loose, tight)

    def test_signature_is_what_precision_policy_expects(self):
        import inspect
        sig = inspect.signature(rs.complex_energy_fraction_relative)
        self.assertEqual(list(sig.parameters)[:2], ["M", "tol"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
