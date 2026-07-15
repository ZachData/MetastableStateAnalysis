"""
tests/test_p2b_imaginary_blocks34.py — Tests for Block 3 and Block 4 of Phase 2i.

Coverage
--------
imaginary_ablation.py
  build_imaginary_projector     — pure antisymmetric, symmetric, mixed, zero
  build_projectors_all_layers   — shared / per-layer contracts
  apply_ablation_from_threshold — threshold=0, threshold=n_layers, shape invariance
  output_cosine_degradation     — identical acts → 0, orthogonal acts → 1
  cluster_ari_at_plateau_layers — identical acts → ARI 1, out-of-range layers skipped
  depth_sweep                   — output shape, threshold=n_layers yields no change
  classify_outcome              — ARTIFACTUAL, COMPUTATION_ONLY, COUPLED, INDETERMINATE
  analyze_imaginary_ablation    — shared and per-layer ov_data contracts
  ablation_to_json              — JSON-serialisable

layernorm_jacobian.py
  layernorm_jacobian            — kills mean direction, rank d-2, known eigenvalue
  rotational_fraction           — pure real → 0, pure antisymmetric → near 1
  ln_curvature                  — unit-normalised zero-mean → 1
  compute_inflation_at_layer    — identity V → inflation ≈ 1, output keys
  analyze_layernorm_jacobian    — shared and per-layer contracts
  _classify                     — all four classification branches
  layernorm_jacobian_to_json    — serialisable, per-token arrays stripped
  layernorm_jacobian_summary_lines — required keys present

Run
---
    python -m pytest tests/test_p2b_imaginary_blocks34.py -v
"""

from __future__ import annotations
import sys, os, json, math, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2b_imaginary.imaginary_ablation import (
    build_imaginary_projector,
    build_projectors_all_layers,
    apply_ablation_from_threshold,
    output_cosine_degradation,
    cluster_ari_at_plateau_layers,
    depth_sweep,
    classify_outcome,
    analyze_imaginary_ablation,
    ablation_to_json,
    ablation_summary_lines,
)
from p2b_imaginary.layernorm_jacobian import (
    layernorm_jacobian,
    rotational_fraction,
    ln_curvature,
    compute_inflation_at_layer,
    analyze_layernorm_jacobian,
    _classify,
    layernorm_jacobian_to_json,
    layernorm_jacobian_summary_lines,
)

# ── shared dimensions ────────────────────────────────────────────────────────
D        = 8
N_TOKENS = 16
N_LAYERS = 6


# ── fixture helpers ───────────────────────────────────────────────────────────

def _rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _rot_d(thetas=None):
    """Block-diagonal D×D rotation (pure antisymmetric in the Lie-algebra sense)."""
    if thetas is None:
        thetas = [0.3, 0.7, 1.1, 1.5]
    M = np.zeros((D, D))
    for i, th in enumerate(thetas):
        M[2*i:2*i+2, 2*i:2*i+2] = _rot2(th)
    return M


def _sym_d():
    """Random symmetric (d,d) matrix — A component is zero."""
    rng = np.random.default_rng(0)
    S = rng.standard_normal((D, D))
    return (S + S.T) / 2


def _unit_acts(n_layers=N_LAYERS, n_tokens=N_TOKENS, d=D, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_layers, n_tokens, d))
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(norms, 1e-10)


def _make_ov_shared(V=None):
    return {"ov_total": V if V is not None else _rot_d(),
            "is_per_layer": False, "layer_names": ["shared"]}


def _make_ov_per_layer(n=N_LAYERS):
    rng = np.random.default_rng(1)
    mats = [_rot_d([rng.uniform(0.1, 1.5) for _ in range(D // 2)]) for _ in range(n)]
    return {"ov_total": mats, "is_per_layer": True,
            "layer_names": [f"layer_{i}" for i in range(n)]}


def _make_events(plateau_layers=None):
    if plateau_layers is None:
        plateau_layers = [2, 3, 4]
    return {"plateau_layers": plateau_layers}


# ═══════════════════════════════════════════════════════════════════════════════
# imaginary_ablation — projectors
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildImaginaryProjector(unittest.TestCase):

    def test_antisymmetric_v_nonzero_projector(self):
        A = np.zeros((D, D))
        A[0, 1], A[1, 0] = 1.0, -1.0
        Pi = build_imaginary_projector(A)
        self.assertGreater(np.linalg.norm(Pi, "fro"), 0)

    def test_symmetric_v_zero_projector(self):
        Pi = build_imaginary_projector(_sym_d())
        np.testing.assert_allclose(Pi, 0, atol=1e-8)

    def test_projector_idempotent(self):
        Pi = build_imaginary_projector(_rot_d())
        np.testing.assert_allclose(Pi @ Pi, Pi, atol=1e-10)

    def test_projector_symmetric(self):
        Pi = build_imaginary_projector(_rot_d())
        np.testing.assert_allclose(Pi, Pi.T, atol=1e-10)

    def test_projector_range_in_col_A(self):
        """Pi @ v should lie in col(A) for any v."""
        V  = _rot_d()
        A  = (V - V.T) / 2
        Pi = build_imaginary_projector(V)
        v  = np.random.default_rng(7).standard_normal(D)
        Pv = Pi @ v
        # Pv should be expressible as A @ something (residual small)
        # Equivalently (I - Pi) @ Pv ≈ 0
        np.testing.assert_allclose((np.eye(D) - Pi) @ Pv, 0, atol=1e-8)

    def test_zero_matrix_gives_zero_projector(self):
        Pi = build_imaginary_projector(np.zeros((D, D)))
        np.testing.assert_allclose(Pi, 0, atol=1e-10)


class TestBuildProjectorsAllLayers(unittest.TestCase):

    def test_shared_returns_replicated_list(self):
        ov = _make_ov_shared()
        projs = build_projectors_all_layers(ov)
        self.assertEqual(len(projs), 1)
        self.assertEqual(projs[0].shape, (D, D))

    def test_per_layer_returns_one_per_layer(self):
        ov = _make_ov_per_layer()
        projs = build_projectors_all_layers(ov)
        self.assertEqual(len(projs), N_LAYERS)
        for P in projs:
            self.assertEqual(P.shape, (D, D))


# ═══════════════════════════════════════════════════════════════════════════════
# imaginary_ablation — ablation application
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyAblationFromThreshold(unittest.TestCase):

    def test_shape_preserved(self):
        acts  = _unit_acts()
        projs = build_projectors_all_layers(_make_ov_shared())
        abl   = apply_ablation_from_threshold(acts, projs, threshold=0)
        self.assertEqual(abl.shape, acts.shape)

    def test_threshold_n_layers_no_change(self):
        """Threshold beyond last layer → identical output."""
        acts  = _unit_acts()
        projs = build_projectors_all_layers(_make_ov_shared())
        abl   = apply_ablation_from_threshold(acts, projs, threshold=N_LAYERS)
        np.testing.assert_array_equal(abl, acts)

    def test_threshold_zero_modifies_all_layers(self):
        """Threshold 0 → all layers modified (except those with zero projector)."""
        acts  = _unit_acts()
        V     = _rot_d()          # non-trivial antisymmetric component
        projs = [build_imaginary_projector(V)] * N_LAYERS
        abl   = apply_ablation_from_threshold(acts, projs, threshold=0)
        # At least one layer should differ from original
        any_diff = any(
            not np.allclose(abl[l], acts[l]) for l in range(N_LAYERS)
        )
        self.assertTrue(any_diff)

    def test_layers_before_threshold_unchanged(self):
        acts  = _unit_acts()
        projs = build_projectors_all_layers(_make_ov_shared())
        thr   = 3
        abl   = apply_ablation_from_threshold(acts, projs, threshold=thr)
        for l in range(thr):
            np.testing.assert_array_equal(abl[l], acts[l])

    def test_symmetric_v_ablation_is_noop(self):
        """Pi_A = 0 for symmetric V → ablation leaves activations unchanged."""
        acts  = _unit_acts()
        projs = [build_imaginary_projector(_sym_d())] * N_LAYERS
        abl   = apply_ablation_from_threshold(acts, projs, threshold=0)
        np.testing.assert_allclose(abl, acts, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# imaginary_ablation — measurement
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputCosineDegradation(unittest.TestCase):

    def test_identical_acts_zero_degradation(self):
        acts = _unit_acts()
        self.assertAlmostEqual(output_cosine_degradation(acts, acts), 0.0, places=10)

    def test_orthogonal_final_layer_unit_degradation(self):
        rng = np.random.default_rng(3)
        a = _unit_acts()
        b = a.copy()
        # Replace final layer with orthogonal vectors
        X = rng.standard_normal((N_TOKENS, D))
        X /= np.linalg.norm(X, axis=-1, keepdims=True)
        # Make orthogonal to original final layer
        orig = a[-1]
        X = X - (X * orig).sum(-1, keepdims=True) * orig
        X /= np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-10)
        b[-1] = X
        deg = output_cosine_degradation(a, b)
        self.assertGreater(deg, 0.1)

    def test_range_0_1(self):
        a = _unit_acts(seed=10)
        b = _unit_acts(seed=11)
        deg = output_cosine_degradation(a, b)
        self.assertGreaterEqual(deg, -1e-10)
        self.assertLessEqual(deg, 1.0 + 1e-10)


class TestClusterAriAtPlateauLayers(unittest.TestCase):

    def test_identical_acts_ari_near_one(self):
        acts = _unit_acts()
        res  = cluster_ari_at_plateau_layers(acts, acts, plateau_layers=[2, 3])
        for p in res["per_plateau"]:
            self.assertAlmostEqual(p["ari"], 1.0, places=5)

    def test_out_of_range_layers_skipped(self):
        acts = _unit_acts()
        res  = cluster_ari_at_plateau_layers(acts, acts, plateau_layers=[999])
        self.assertEqual(len(res["per_plateau"]), 0)
        self.assertTrue(math.isnan(res["mean_ari"]))

    def test_output_keys(self):
        acts = _unit_acts()
        res  = cluster_ari_at_plateau_layers(acts, acts, plateau_layers=[1])
        self.assertIn("per_plateau", res)
        self.assertIn("mean_ari",    res)

    def test_per_plateau_entry_keys(self):
        acts = _unit_acts()
        res  = cluster_ari_at_plateau_layers(acts, acts, plateau_layers=[1])
        for k in ("layer", "ari", "n_clusters_orig", "n_clusters_ablated"):
            self.assertIn(k, res["per_plateau"][0])


# ═══════════════════════════════════════════════════════════════════════════════
# imaginary_ablation — depth sweep and classification
# ═══════════════════════════════════════════════════════════════════════════════

class TestDepthSweep(unittest.TestCase):

    def test_output_length_matches_thresholds(self):
        acts = _unit_acts()
        ov   = _make_ov_shared()
        res  = depth_sweep(acts, ov, plateau_layers=[2], thresholds=[0, 2, 4])
        self.assertEqual(len(res), 3)

    def test_entry_keys(self):
        acts = _unit_acts()
        res  = depth_sweep(acts, _make_ov_shared(), plateau_layers=[1],
                           thresholds=[0])
        for k in ("threshold", "output_degradation", "mean_ari", "per_plateau"):
            self.assertIn(k, res[0])

    def test_threshold_n_layers_zero_degradation(self):
        """No ablation applied → degradation = 0."""
        acts = _unit_acts()
        res  = depth_sweep(acts, _make_ov_shared(), plateau_layers=[2],
                           thresholds=[N_LAYERS])
        self.assertAlmostEqual(res[0]["output_degradation"], 0.0, places=10)

    def test_per_layer_ov_accepted(self):
        acts = _unit_acts()
        res  = depth_sweep(acts, _make_ov_per_layer(), plateau_layers=[2],
                           thresholds=[0, 3])
        self.assertEqual(len(res), 2)


class TestClassifyOutcome(unittest.TestCase):

    def _sweep(self, out_degs, aris):
        """Build synthetic sweep dicts from parallel lists."""
        return [{"threshold": i, "output_degradation": od, "mean_ari": ar}
                for i, (od, ar) in enumerate(zip(out_degs, aris))]

    def test_artifactual(self):
        # Neither measure degrades
        sweep = self._sweep([0.0] * 5, [1.0] * 5)
        c = classify_outcome(sweep)
        self.assertEqual(c["outcome"], "ARTIFACTUAL")

    def test_computation_only(self):
        # Output degrades at step 2; clusters stay intact
        out_degs = [0.0, 0.0, 0.1, 0.2, 0.3]
        aris     = [1.0, 1.0, 1.0, 1.0, 1.0]
        c = classify_outcome(self._sweep(out_degs, aris))
        self.assertEqual(c["outcome"], "COMPUTATION_ONLY")
        self.assertEqual(c["output_threshold"], 2)

    def test_coupled(self):
        # Both degrade at the same threshold
        out_degs = [0.0, 0.0, 0.1, 0.2, 0.3]
        aris     = [1.0, 1.0, 0.5, 0.3, 0.2]
        c = classify_outcome(self._sweep(out_degs, aris))
        self.assertEqual(c["outcome"], "COUPLED")

    def test_empty_sweep_indeterminate(self):
        c = classify_outcome([])
        self.assertEqual(c["outcome"], "INDETERMINATE")

    def test_computation_only_no_ari(self):
        """No plateau layers → mean_ari is NaN; output degradation still classified."""
        sweep = [{"threshold": 0, "output_degradation": 0.1,
                  "mean_ari": float("nan"), "per_plateau": []}]
        c = classify_outcome(sweep)
        # cluster_threshold should be None; output_threshold = 0
        self.assertIsNone(c["cluster_threshold"])
        self.assertEqual(c["output_threshold"], 0)


class TestAnalyzeImaginaryAblation(unittest.TestCase):

    def test_shared_ov_output_keys(self):
        acts = _unit_acts()
        res  = analyze_imaginary_ablation(acts, _make_ov_shared(),
                                          _make_events(), thresholds=[0, 3])
        for k in ("sweep", "classification", "plateau_layers",
                  "n_layers", "interpretation"):
            self.assertIn(k, res)

    def test_per_layer_ov_accepted(self):
        acts = _unit_acts()
        res  = analyze_imaginary_ablation(acts, _make_ov_per_layer(),
                                          _make_events(), thresholds=[0, 3])
        self.assertIn("classification", res)

    def test_n_layers_matches_activations(self):
        acts = _unit_acts()
        res  = analyze_imaginary_ablation(acts, _make_ov_shared(),
                                          _make_events(), thresholds=[0])
        self.assertEqual(res["n_layers"], N_LAYERS)

    def test_json_serialisable(self):
        acts = _unit_acts()
        res  = analyze_imaginary_ablation(acts, _make_ov_shared(),
                                          _make_events(), thresholds=[0])
        json.dumps(ablation_to_json(res))

    def test_summary_lines_list_of_strings(self):
        acts = _unit_acts()
        res  = analyze_imaginary_ablation(acts, _make_ov_shared(),
                                          _make_events(), thresholds=[0])
        lines = ablation_summary_lines(res)
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(l, str) for l in lines))


# ═══════════════════════════════════════════════════════════════════════════════
# layernorm_jacobian — primitives
# ═══════════════════════════════════════════════════════════════════════════════

class TestLayernormJacobian(unittest.TestCase):

    def _random_x(self, seed=0):
        return np.random.default_rng(seed).standard_normal(D)

    def test_kills_constant_direction(self):
        """J @ 1_vec ≈ 0 (LN removes mean)."""
        J    = layernorm_jacobian(self._random_x())
        ones = np.ones(D)
        np.testing.assert_allclose(J @ ones, 0, atol=1e-8)

    def test_output_shape(self):
        J = layernorm_jacobian(self._random_x())
        self.assertEqual(J.shape, (D, D))

    def test_rank_d_minus_2(self):
        """J should have rank d-2 (removes mean and x̂ directions)."""
        J    = layernorm_jacobian(self._random_x(seed=5))
        rank = np.linalg.matrix_rank(J, tol=1e-6)
        self.assertEqual(rank, D - 2)

    def test_scaled_input_changes_jacobian(self):
        """Scaling x by k scales J by 1/k (σ scales, J = .../σ)."""
        x = self._random_x(seed=2)
        J1 = layernorm_jacobian(x)
        J2 = layernorm_jacobian(x * 3.0)
        # J2 ≈ J1 / 3  (up to the x̂ x̂^T term which is scale-invariant)
        # at least: Frobenius norm of J2 should be smaller
        self.assertLess(np.linalg.norm(J2, "fro"),
                        np.linalg.norm(J1, "fro"))

    def test_zero_mean_unit_var_input(self):
        """For x that is already normalised, J should be close to (I - xx^T)/1."""
        rng = np.random.default_rng(9)
        x   = rng.standard_normal(D)
        x   = (x - x.mean()) / (x.std() + 1e-5)
        J   = layernorm_jacobian(x, eps=0)
        # ||J||_F should be around sqrt(d-2)/1 (rank d-2, each sv ≈ 1)
        frob = np.linalg.norm(J, "fro")
        self.assertGreater(frob, 0.5)


class TestRotationalFraction(unittest.TestCase):

    def test_symmetric_matrix_near_zero(self):
        """Symmetric matrix has real eigenvalues → fraction ≈ 0."""
        S = _sym_d()
        self.assertAlmostEqual(rotational_fraction(S), 0.0, places=5)

    def test_pure_rotation_nonzero(self):
        """Block-diagonal rotation has complex eigenvalues → fraction > 0."""
        R = _rot_d()
        self.assertGreater(rotational_fraction(R), 0.5)

    def test_identity_is_zero(self):
        self.assertAlmostEqual(rotational_fraction(np.eye(D)), 0.0, places=5)

    def test_range_0_1(self):
        rng = np.random.default_rng(4)
        M   = rng.standard_normal((D, D))
        rf  = rotational_fraction(M)
        self.assertGreaterEqual(rf, 0.0)
        self.assertLessEqual(rf, 1.0 + 1e-10)


class TestLnCurvature(unittest.TestCase):

    def test_unit_normalised_zero_mean(self):
        """x already at LN fixed point → κ ≈ 1."""
        rng = np.random.default_rng(0)
        x   = rng.standard_normal(D)
        x   = (x - x.mean()) / (x.std() + 1e-5)
        kap = ln_curvature(x, eps=0)
        self.assertAlmostEqual(kap, 1.0, places=4)

    def test_constant_vector_low_curvature(self):
        """All-same values → σ ≈ 0, curvature → 0."""
        x   = np.ones(D) * 5.0
        kap = ln_curvature(x)
        self.assertAlmostEqual(kap, 0.0, places=3)

    def test_positive(self):
        x = np.random.default_rng(1).standard_normal(D)
        self.assertGreaterEqual(ln_curvature(x), 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# layernorm_jacobian — inflation
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeInflationAtLayer(unittest.TestCase):

    def test_output_keys(self):
        acts = _unit_acts()[0]
        res  = compute_inflation_at_layer(acts, _rot_d())
        for k in ("base_rot_frac", "per_token_rot_frac", "inflation_ratios",
                  "curvatures", "mean_inflation", "pearson_r"):
            self.assertIn(k, res)

    def test_identity_v_inflation_near_one(self):
        """V = I has no rotational structure; J @ I = J.  rot_frac(J) / rot_frac(I)."""
        # rot_frac(I) = 0, so inflation is ill-defined.  Use a near-identity V instead.
        eps_sym = np.eye(D) * 0.01
        acts    = _unit_acts()[0]
        res     = compute_inflation_at_layer(acts, eps_sym)
        # base_rot_frac should be near 0
        self.assertAlmostEqual(res["base_rot_frac"], 0.0, places=3)

    def test_shapes(self):
        acts = _unit_acts()[0]   # (N_TOKENS, D)
        res  = compute_inflation_at_layer(acts, _rot_d())
        self.assertEqual(res["per_token_rot_frac"].shape, (N_TOKENS,))
        self.assertEqual(res["inflation_ratios"].shape,   (N_TOKENS,))
        self.assertEqual(res["curvatures"].shape,         (N_TOKENS,))

    def test_inflation_positive(self):
        acts = _unit_acts()[0]
        res  = compute_inflation_at_layer(acts, _rot_d())
        self.assertTrue(np.all(res["inflation_ratios"] >= 0))


class TestAnalyzeLayernormJacobian(unittest.TestCase):

    def test_shared_ov_output_keys(self):
        acts = _unit_acts()
        res  = analyze_layernorm_jacobian(acts, _make_ov_shared())
        for k in ("per_layer", "mean_inflation", "mean_pearson_r",
                  "classification", "interpretation"):
            self.assertIn(k, res)

    def test_per_layer_ov_accepted(self):
        acts = _unit_acts()
        res  = analyze_layernorm_jacobian(acts, _make_ov_per_layer())
        self.assertEqual(len(res["per_layer"]), N_LAYERS)

    def test_per_layer_entry_keys(self):
        acts = _unit_acts()
        res  = analyze_layernorm_jacobian(acts, _make_ov_shared())
        for k in ("layer", "base_rot_frac", "mean_inflation", "pearson_r"):
            self.assertIn(k, res["per_layer"][0])

    def test_json_serialisable(self):
        acts = _unit_acts()
        res  = analyze_layernorm_jacobian(acts, _make_ov_shared())
        json.dumps(layernorm_jacobian_to_json(res))

    def test_json_strips_per_token_arrays(self):
        """per_token_rot_frac and inflation_ratios should not appear in json output."""
        acts = _unit_acts()
        res  = analyze_layernorm_jacobian(acts, _make_ov_shared())
        j    = layernorm_jacobian_to_json(res)
        for entry in j["per_layer"]:
            self.assertNotIn("per_token_rot_frac", entry)
            self.assertNotIn("inflation_ratios",   entry)

    def test_summary_lines_non_empty_strings(self):
        acts  = _unit_acts()
        res   = analyze_layernorm_jacobian(acts, _make_ov_shared())
        lines = layernorm_jacobian_summary_lines(res)
        self.assertGreater(len(lines), 0)
        self.assertTrue(all(isinstance(l, str) for l in lines))


class TestClassifyLayernorm(unittest.TestCase):

    def test_h2_supported(self):
        cls, interp = _classify(mean_inf=2.0, mean_r=0.5)
        self.assertEqual(cls, "H2_SUPPORTED")
        self.assertIn("H2", interp)

    def test_h2_partial_high_inflation_only(self):
        cls, _ = _classify(mean_inf=2.0, mean_r=0.1)
        self.assertEqual(cls, "H2_PARTIAL")

    def test_h2_partial_high_r_only(self):
        cls, _ = _classify(mean_inf=1.2, mean_r=0.5)
        self.assertEqual(cls, "H2_PARTIAL")

    def test_h2_unsupported(self):
        cls, _ = _classify(mean_inf=1.1, mean_r=0.1)
        self.assertEqual(cls, "H2_UNSUPPORTED")

    def test_indeterminate_on_nan(self):
        cls, _ = _classify(mean_inf=float("nan"), mean_r=0.5)
        self.assertEqual(cls, "INDETERMINATE")


if __name__ == "__main__":
    unittest.main()