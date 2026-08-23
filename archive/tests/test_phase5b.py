"""
tests/test_phase5b.py — TDD contracts for Phase 5b modules.

Each test class specifies the input/output contract for one module.
Tests use synthetic data with known geometric properties so the expected
outputs are analytically checkable.

Run with: pytest tests/test_phase5b.py -v
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

# Tier: deps -- needs the heavy tier importable (torch / transformers /
# scikit-learn / matplotlib). No model download, no run artifacts.
# Measured, not assumed; see pyproject.toml markers.
pytestmark = pytest.mark.deps

# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
D   = 64    # activation dimension
N_C = 7     # number of clusters (one per "day of week" analog)
N_L = 12    # number of plateau layers
K   = 32    # PCA reduction dimension
N_V = 512   # vocabulary size for logit distributions


def _ring_centroids(n: int, d: int, radius: float = 1.0) -> np.ndarray:
    """n centroids arranged on a ring in the first 2 PCA dims, noise in remaining."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c = np.zeros((n, d))
    c[:, 0] = radius * np.cos(angles)
    c[:, 1] = radius * np.sin(angles)
    c[:, 2:] = RNG.standard_normal((n, d - 2)) * 0.05  # small noise
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    return c / norms


def _peaked_distributions(n: int, vocab: int) -> np.ndarray:
    """n distributions each peaked on a different token, with neighbor spillover."""
    p = np.ones((n, vocab)) * 1e-6
    for i in range(n):
        p[i, i % vocab] = 0.8
        p[i, (i + 1) % vocab] = 0.1
        p[i, (i - 1) % vocab] = 0.1
    return p / p.sum(axis=1, keepdims=True)


# ===========================================================================
# A — manifold_fit
# ===========================================================================

class TestPCAReduce:
    """pca_reduce(centroids, k) → (scores, basis, explained_var_ratio)"""

    def setup_method(self):
        from p5b_manifold_steering.manifold_fit import pca_reduce
        self.fn = pca_reduce
        self.centroids = _ring_centroids(N_C, D)

    def test_output_shape(self):
        scores, basis, evr = self.fn(self.centroids, K)
        assert scores.shape == (N_C, K)
        assert basis.shape  == (D, K)
        assert evr.shape    == (K,)

    def test_explained_variance_sums_leq_1(self):
        _, _, evr = self.fn(self.centroids, K)
        assert float(evr.sum()) <= 1.0 + 1e-6

    def test_explained_variance_monotone(self):
        _, _, evr = self.fn(self.centroids, K)
        assert np.all(np.diff(evr) <= 1e-8), "EVR should be non-increasing"

    def test_ring_structure_captured_in_2d(self):
        """Ring centroids: first 2 PCs should capture almost all variance."""
        scores, _, evr = self.fn(self.centroids, K)
        assert evr[:2].sum() > 0.90, "Ring structure should be in top-2 PCs"

    def test_basis_orthonormal(self):
        _, basis, _ = self.fn(self.centroids, K)
        npt.assert_allclose(basis.T @ basis, np.eye(K), atol=1e-5)


class TestFitActivationManifold:
    """fit_activation_manifold(centroids_pca, intrinsic_coords) → SplineManifold"""

    def setup_method(self):
        from p5b_manifold_steering.manifold_fit import fit_activation_manifold
        self.fn = fit_activation_manifold
        self.ring = _ring_centroids(N_C, K)   # already in PCA space
        self.u = np.linspace(0, 1, N_C)       # arc-length params

    def test_returns_dict_with_required_keys(self):
        mh = self.fn(self.ring, self.u, periodic=True)
        for key in ("spline", "u_knots", "control_pts", "residual_rms"):
            assert key in mh, f"Missing key: {key}"

    def test_spline_interpolates_control_points(self):
        """Spline evaluated at knot u values should recover control points."""
        from p5b_manifold_steering.manifold_fit import fit_activation_manifold, eval_manifold
        mh = self.fn(self.ring, self.u, periodic=True)
        reconstructed = eval_manifold(mh, self.u)
        npt.assert_allclose(reconstructed, self.ring, atol=1e-3)

    def test_residual_rms_small_for_smooth_data(self):
        mh = self.fn(self.ring, self.u, periodic=True)
        assert mh["residual_rms"] < 0.05

    def test_eval_returns_correct_shape(self):
        from p5b_manifold_steering.manifold_fit import fit_activation_manifold, eval_manifold
        mh = self.fn(self.ring, self.u, periodic=True)
        t = np.linspace(0, 1, 50)
        pts = eval_manifold(mh, t)
        assert pts.shape == (50, K)


class TestFitBehaviorManifold:
    """fit_behavior_manifold(distributions, intrinsic_coords) → SplineManifold"""

    def setup_method(self):
        from p5b_manifold_steering.manifold_fit import fit_behavior_manifold
        self.fn = fit_behavior_manifold
        self.dists = _peaked_distributions(N_C, N_V)
        self.u = np.linspace(0, 1, N_C)

    def test_returns_dict_with_required_keys(self):
        my = self.fn(self.dists, self.u, periodic=True)
        for key in ("spline", "u_knots", "sqrt_centroids", "residual_rms"):
            assert key in my, f"Missing key: {key}"

    def test_sqrt_centroids_on_unit_sphere(self):
        """√p vectors should be unit norm."""
        my = self.fn(self.dists, self.u, periodic=True)
        norms = np.linalg.norm(my["sqrt_centroids"], axis=1)
        npt.assert_allclose(norms, 1.0, atol=1e-4)

    def test_eval_returns_valid_distributions(self):
        """Decoded distributions must be non-negative and sum to 1."""
        from p5b_manifold_steering.manifold_fit import fit_behavior_manifold, eval_behavior_manifold
        my = self.fn(self.dists, self.u, periodic=True)
        t = np.linspace(0, 1, 20)
        decoded = eval_behavior_manifold(my, t)
        assert decoded.shape == (20, N_V)
        assert np.all(decoded >= -1e-6)
        npt.assert_allclose(decoded.sum(axis=1), 1.0, atol=1e-4)


# ===========================================================================
# B — isometry_test
# ===========================================================================

class TestGeodesicDistance:
    """geodesic_distance_manifold(mh, u_i, u_j, n_pts) → float"""

    def setup_method(self):
        from p5b_manifold_steering.manifold_fit import fit_activation_manifold, pca_reduce
        from p5b_manifold_steering.isometry_test import geodesic_distance_manifold
        self.geo_fn = geodesic_distance_manifold
        ring = _ring_centroids(N_C, K)
        u    = np.linspace(0, 1, N_C)
        self.mh = fit_activation_manifold(ring, u, periodic=True)
        self.u  = u

    def test_self_distance_zero(self):
        d = self.geo_fn(self.mh, 0.0, 0.0)
        assert abs(d) < 1e-6

    def test_distance_positive(self):
        d = self.geo_fn(self.mh, 0.0, 0.3)
        assert d > 0

    def test_distance_symmetric(self):
        d_fwd = self.geo_fn(self.mh, 0.1, 0.4)
        d_rev = self.geo_fn(self.mh, 0.4, 0.1)
        npt.assert_allclose(d_fwd, d_rev, rtol=1e-3)

    def test_triangle_inequality(self):
        d_ab = self.geo_fn(self.mh, 0.0, 0.3)
        d_bc = self.geo_fn(self.mh, 0.3, 0.6)
        d_ac = self.geo_fn(self.mh, 0.0, 0.6)
        assert d_ac <= d_ab + d_bc + 1e-6


class TestHellingerDistance:
    """hellinger_distance(p, q) → float in [0, 1]"""

    def setup_method(self):
        from p5b_manifold_steering.isometry_test import hellinger_distance
        self.fn = hellinger_distance

    def test_identical_distributions(self):
        p = np.array([0.5, 0.3, 0.2])
        npt.assert_allclose(self.fn(p, p), 0.0, atol=1e-8)

    def test_orthogonal_distributions(self):
        # dH([1,0,0],[0,1,0]) = ||[1,0,0]-[0,1,0]||/√2 = √2/√2 = 1.0
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        npt.assert_allclose(self.fn(p, q), 1.0, atol=1e-6)

    def test_range(self):
        # Hellinger distance ∈ [0, 1]; maximum at disjoint supports
        rng = np.random.default_rng(7)
        for _ in range(20):
            p = rng.dirichlet(np.ones(10))
            q = rng.dirichlet(np.ones(10))
            d = self.fn(p, q)
            assert 0.0 <= d <= 1.0 + 1e-6

    def test_symmetry(self):
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.6, 0.3])
        npt.assert_allclose(self.fn(p, q), self.fn(q, p), atol=1e-10)


class TestPairwiseDistanceMatrix:
    """pairwise_distances(mh, my, u_coords) → dict with d_manifold, d_linear, d_behavior"""

    def setup_method(self):
        from p5b_manifold_steering.manifold_fit import (
            fit_activation_manifold, fit_behavior_manifold, pca_reduce
        )
        from p5b_manifold_steering.isometry_test import pairwise_distances
        ring  = _ring_centroids(N_C, K)
        dists = _peaked_distributions(N_C, N_V)
        u     = np.linspace(0, 1, N_C)
        self.mh     = fit_activation_manifold(ring, u, periodic=True)
        self.my     = fit_behavior_manifold(dists, u, periodic=True)
        self.u      = u
        self.raw_c  = ring
        self.fn     = pairwise_distances

    def test_output_keys(self):
        out = self.fn(self.mh, self.my, self.u, self.raw_c)
        for k in ("d_manifold", "d_linear", "d_behavior", "n_pairs"):
            assert k in out, f"Missing key: {k}"

    def test_n_pairs_correct(self):
        out = self.fn(self.mh, self.my, self.u, self.raw_c)
        expected = N_C * (N_C - 1) // 2
        assert out["n_pairs"] == expected

    def test_distances_nonnegative(self):
        out = self.fn(self.mh, self.my, self.u, self.raw_c)
        for key in ("d_manifold", "d_linear", "d_behavior"):
            assert np.all(out[key] >= 0), f"{key} has negative values"

    def test_diagonal_zero(self):
        """Self-distances should be zero (not stored in flat array, but
        we can verify via d_manifold at u_i == u_j)."""
        from p5b_manifold_steering.isometry_test import geodesic_distance_manifold
        d_self = geodesic_distance_manifold(self.mh, 0.0, 0.0)
        assert abs(d_self) < 1e-5


class TestIsometryScore:
    """isometry_score(d_manifold, d_behavior, d_linear) → dict with r_manifold, r_linear"""

    def setup_method(self):
        from p5b_manifold_steering.isometry_test import isometry_score
        self.fn = isometry_score

    def test_perfect_correlation_returns_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self.fn(x, x, x * 0.5)
        npt.assert_allclose(result["r_manifold"], 1.0, atol=1e-6)

    def test_r_in_valid_range(self):
        rng = np.random.default_rng(13)
        a = rng.uniform(0, 10, 30)
        b = rng.uniform(0, 10, 30)
        c = rng.uniform(0, 10, 30)
        out = self.fn(a, b, c)
        assert -1.0 <= out["r_manifold"] <= 1.0
        assert -1.0 <= out["r_linear"]   <= 1.0

    def test_output_keys(self):
        x = np.arange(1, 11, dtype=float)
        out = self.fn(x, x, x)
        for k in ("r_manifold", "r_linear", "p_manifold", "p_linear", "n_pairs"):
            assert k in out, f"Missing key: {k}"

    def test_manifold_better_than_linear_for_ring(self):
        """For a ring geometry, manifold distances should correlate better
        with true arc-length than Euclidean chord distances."""
        # Chord distances for a ring of radius 1 underestimate arc-length
        angles = np.linspace(0, 2 * np.pi, N_C, endpoint=False)
        arc    = np.abs(np.subtract.outer(angles, angles))
        arc    = np.minimum(arc, 2 * np.pi - arc)  # shorter arc
        chord  = 2 * np.sin(arc / 2)
        # Use arc as behavior, chord as linear, arc as manifold (perfect)
        upper  = arc[np.triu_indices(N_C, 1)]
        c_flat = chord[np.triu_indices(N_C, 1)]
        out = self.fn(upper, upper, c_flat)
        assert out["r_manifold"] >= out["r_linear"], (
            "Manifold distances should not be worse than chord for ring"
        )


# ===========================================================================
# C — merge_teleportation
# ===========================================================================

class TestTeleportationScore:
    """teleportation_score(p_before, p_at, p_after) → dict"""

    def setup_method(self):
        from p5b_manifold_steering.merge_teleportation_subspace import teleportation_score
        self.fn = teleportation_score

    def _peaked(self, idx: int) -> np.ndarray:
        p = np.ones(N_V) * 1e-6 / N_V
        p[idx] = 0.95
        return p / p.sum()

    def test_no_jump_returns_low_score(self):
        p = self._peaked(0)
        out = self.fn(p, p, p)
        assert out["kl_divergence"] < 0.01
        assert out["non_adjacent_mass"] < 0.05

    def test_teleportation_returns_high_kl(self):
        p_before = self._peaked(0)
        p_jump   = self._peaked(50)   # far-away token
        p_after  = self._peaked(1)
        out = self.fn(p_before, p_jump, p_after)
        assert out["kl_divergence"] > 1.0

    def test_output_keys(self):
        p = self._peaked(0)
        out = self.fn(p, p, p)
        for k in ("kl_divergence", "non_adjacent_mass", "bhattacharyya_approx"):
            assert k in out, f"Missing key: {k}"

    def test_kl_nonnegative(self):
        p = self._peaked(0)
        q = self._peaked(1)
        out = self.fn(p, q, p)
        assert out["kl_divergence"] >= 0.0

    def test_non_adjacent_mass_in_range(self):
        p = self._peaked(0)
        q = self._peaked(3)
        out = self.fn(p, q, p)
        assert 0.0 <= out["non_adjacent_mass"] <= 1.0


class TestMergeVsPlateauComparison:
    """compare_merge_plateau(merge_scores, plateau_scores) → dict with statistics"""

    def setup_method(self):
        from p5b_manifold_steering.merge_teleportation_subspace import compare_merge_plateau
        self.fn = compare_merge_plateau

    def test_output_keys(self):
        merge   = {"kl_divergence": [2.0, 1.8, 2.2], "non_adjacent_mass": [0.4, 0.3, 0.5]}
        plateau = {"kl_divergence": [0.1, 0.05, 0.2], "non_adjacent_mass": [0.02, 0.01, 0.03]}
        out = self.fn(merge, plateau)
        for k in ("kl_mean_merge", "kl_mean_plateau", "kl_pvalue",
                  "nam_mean_merge", "nam_mean_plateau", "nam_pvalue",
                  "p5b_c1_pass", "p5b_c3_pass"):
            assert k in out, f"Missing key: {k}"

    def test_clearly_different_distributions_pass(self):
        merge   = {"kl_divergence": [2.0] * 10, "non_adjacent_mass": [0.4] * 10}
        plateau = {"kl_divergence": [0.1] * 10, "non_adjacent_mass": [0.02] * 10}
        out = self.fn(merge, plateau)
        assert out["p5b_c1_pass"] is True
        assert out["p5b_c3_pass"] is True

    def test_identical_distributions_fail(self):
        scores  = {"kl_divergence": [0.5] * 10, "non_adjacent_mass": [0.1] * 10}
        out = self.fn(scores, scores)
        assert out["p5b_c1_pass"] is False


# ===========================================================================
# D — subspace_isometry
# ===========================================================================

class TestSubspaceProjection:
    """project_centroids(centroids, U) → projected (n, k)"""

    def setup_method(self):
        from p5b_manifold_steering.subspace_isometry import project_centroids
        self.fn = project_centroids

    def test_output_shape(self):
        centroids = RNG.standard_normal((N_C, D))
        U = np.linalg.svd(RNG.standard_normal((D, 16)), full_matrices=False)[0]
        out = self.fn(centroids, U)
        assert out.shape == (N_C, 16)

    def test_projection_idempotent(self):
        """Projecting twice gives same result as projecting once."""
        centroids = RNG.standard_normal((N_C, D))
        U = np.linalg.svd(RNG.standard_normal((D, 16)), full_matrices=False)[0]
        P1 = self.fn(centroids, U)
        # Reconstruct back to ambient and project again
        back = P1 @ U.T
        P2 = self.fn(back, U)
        npt.assert_allclose(P1, P2, atol=1e-5)


class TestSubspaceIsometryScore:
    """subspace_isometry_score(centroids, U_S, U_A, d_behavior) → dict"""

    def setup_method(self):
        from p5b_manifold_steering.subspace_isometry import subspace_isometry_score
        self.fn = subspace_isometry_score

    def _orthogonal_basis(self, d: int, k: int) -> np.ndarray:
        return np.linalg.svd(RNG.standard_normal((d, k)), full_matrices=False)[0]

    def test_output_keys(self):
        ring = _ring_centroids(N_C, D)
        U_S  = self._orthogonal_basis(D, 16)
        U_A  = self._orthogonal_basis(D, 16)
        d_beh = np.abs(RNG.standard_normal(N_C * (N_C - 1) // 2))
        out = self.fn(ring, U_S, U_A, d_beh)
        for k in ("r_S", "r_A", "r_full", "r_linear",
                  "p5b_d1_pass", "p5b_d2_pass"):
            assert k in out, f"Missing key: {k}"

    def test_r_values_in_range(self):
        ring  = _ring_centroids(N_C, D)
        U_S   = self._orthogonal_basis(D, 16)
        U_A   = self._orthogonal_basis(D, 16)
        d_beh = np.abs(RNG.standard_normal(N_C * (N_C - 1) // 2))
        out   = self.fn(ring, U_S, U_A, d_beh)
        for k in ("r_S", "r_A", "r_full", "r_linear"):
            assert -1.0 <= out[k] <= 1.0, f"{k} out of range"

    def test_ring_in_S_subspace_passes_d1(self):
        """If Mh lives entirely in U_S, r_S should dominate r_A."""
        # Build centroids that live in a known 2D subspace (the S subspace)
        U_S = self._orthogonal_basis(D, 8)
        U_A = self._orthogonal_basis(D, 8)  # orthogonal to ring
        # Ring in S coordinates
        angles = np.linspace(0, 2 * np.pi, N_C, endpoint=False)
        s_coords = np.zeros((N_C, 8))
        s_coords[:, 0] = np.cos(angles)
        s_coords[:, 1] = np.sin(angles)
        # Lift to ambient D
        centroids = s_coords @ U_S.T
        # Behavior distances = arc lengths on ring
        arc = np.abs(np.subtract.outer(angles, angles))
        arc = np.minimum(arc, 2 * np.pi - arc)
        d_beh = arc[np.triu_indices(N_C, 1)]
        out = self.fn(centroids, U_S, U_A, d_beh)
        assert out["r_S"] > out["r_A"], (
            "When Mh is in U_S, S-projected distances should correlate better "
            "with behavior distances than A-projected distances"
        )


# ===========================================================================
# logit_cache
# ===========================================================================

class TestLogitCache:
    """
    logit_cache.extract_layer_logits(model, tokenizer, prompt, layers)
       → dict: {layer_idx: (n_tokens, vocab_size) float32}
    """

    def setup_method(self):
        from p5b_manifold_steering.logit_cache import validate_logit_output
        self.validate = validate_logit_output

    def test_validate_passes_for_valid_output(self):
        n_tok, vocab = 10, 1000
        # Use finite log-probabilities (not -inf)
        logits = {
            0: np.full((n_tok, vocab), -6.9, dtype=np.float32),
            3: np.full((n_tok, vocab), -6.9, dtype=np.float32),
        }
        self.validate(logits, expected_vocab=vocab)  # should not raise

    def test_validate_raises_for_inf(self):
        logits = {0: np.full((5, 10), -np.inf, dtype=np.float32)}
        with pytest.raises(ValueError, match="infinite|non-finite"):
            self.validate(logits, expected_vocab=10)

    def test_logits_to_distribution_shape(self):
        from p5b_manifold_steering.logit_cache import logits_to_distribution
        logits = RNG.standard_normal((8, 1000)).astype(np.float32)
        p = logits_to_distribution(logits)
        assert p.shape == logits.shape
        npt.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(p >= 0)

    def test_save_load_roundtrip(self, tmp_path):
        from p5b_manifold_steering.logit_cache import save_logit_cache, load_logit_cache
        dists = {
            0: RNG.dirichlet(np.ones(100), size=8).astype(np.float32),
            4: RNG.dirichlet(np.ones(100), size=8).astype(np.float32),
        }
        path = tmp_path / "cache.npz"
        save_logit_cache(dists, path)
        loaded = load_logit_cache(path)
        assert set(loaded.keys()) == {0, 4}
        npt.assert_allclose(loaded[0], dists[0], atol=1e-6)


# ===========================================================================
# report
# ===========================================================================

class TestReport:
    """write_report(out_dir, results) → str (path to report file)"""

    def test_report_contains_all_sections(self, tmp_path):
        from p5b_manifold_steering.p5b_report import write_report
        results = {
            "fit_summary":  {"pca_explained_var": 0.92, "spline_residual_rms": 0.03},
            "isometry":     {"r_manifold": 0.87, "r_linear": 0.52, "n_pairs": 21},
            "teleportation": {"kl_mean_merge": 1.8, "kl_mean_plateau": 0.12,
                              "p5b_c1_pass": True},
            "subspace":     {"r_S": 0.91, "r_A": 0.44, "r_full": 0.87,
                             "p5b_d1_pass": True},
        }
        path = write_report(tmp_path, results, model="gpt2-large",
                            prompt="wiki_paragraph")
        assert path.exists()
        text = path.read_text()
        for section in [
            "PHASE 5b", "SUB-EXP A", "SUB-EXP B", "SUB-EXP C", "SUB-EXP D",
            "FALSIFICATION", "r_manifold", "r_linear",
        ]:
            assert section in text, f"Report missing section: {section}"

    def test_report_encodes_pass_fail(self, tmp_path):
        from p5b_manifold_steering.p5b_report import write_report
        results = {
            "fit_summary":  {"pca_explained_var": 0.75, "spline_residual_rms": 0.25},
            "isometry":     {"r_manifold": 0.45, "r_linear": 0.50, "n_pairs": 21},
            "teleportation": {"kl_mean_merge": 0.15, "kl_mean_plateau": 0.12,
                              "p5b_c1_pass": False},
            "subspace":     {"r_S": 0.40, "r_A": 0.42, "r_full": 0.45,
                             "p5b_d1_pass": False},
        }
        path = write_report(tmp_path, results, model="gpt2-large",
                            prompt="wiki_paragraph")
        text = path.read_text()
        assert "[FAIL]" in text or "FAIL" in text
