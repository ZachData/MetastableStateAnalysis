"""
tests/test_oracle.py

Oracle tier (v2 plan, Testing section, Tier 2 — "highest-value investment").

These are correctness tests, not regression tests: the theory gives a
*proven* answer for a special case, independent of any particular model or
run. If the pipeline's own metric/clustering/decomposition code doesn't
reproduce that answer on the constructed input, the pipeline is wrong,
full stop — there is no "maybe the model just doesn't show the effect
here" escape hatch the way there is for a real trained model.

Three cases, as named in the plan:

  TestIdentityWeightsMonotoneEnergyAndCollapse
      Identity-weight interacting-particle dynamics (Geshkovski et al.)
      ⇒ monotonically non-decreasing interaction energy and eventual
      directional collapse (effective rank -> 1). Exercises
      core.metrics.interaction_energy and core.metrics.effective_rank
      directly on a hand-simulated identity-weight trajectory — no model
      loading, no HF checkpoint, so this runs in milliseconds and has zero
      external dependencies beyond the metrics module itself.

  TestPlantedClustersRecoveredByClusteringStack
      k well-separated synthetic clusters ⇒ recovered by
      p1_mstate_tracking.clustering.cluster_count_sweep (both the KMeans/
      silhouette selection and, when hdbscan is installed, the HDBSCAN
      path) at essentially ARI = 1.0.

  TestConstructedRotationRecoveredBySchurSplit
      A matrix built from one known 2x2 rotation block (planted theta,
      rho) plus known real eigenvalues, conjugated by a random orthogonal
      change of basis (so it is not axis-aligned) ⇒
      p2b_imaginary.rotational_schur.extract_schur_blocks recovers exactly
      one 2x2 block with the planted theta/rho and the correct count of
      1x1 real blocks. This generalizes the existing pure-rotation tests
      in test_phase2i.py (TestSchurBlockDimensions /
      TestRotationAngleRecovery, which use only axis-aligned block-
      rotation matrices) to a mixed real+complex spectrum under an
      arbitrary basis — the case the real OV matrices this function is
      actually run on will look like.

Run:
    pytest tests/test_oracle.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from core.metrics import interaction_energy, effective_rank
from p2b_imaginary.rotational_schur import extract_schur_blocks
from p6_subspace.eigenspace_degeneracy import (
    degeneracy_ratio,
    project_to_subspace,
    run_eigenspace_degeneracy,
)

# Tier: deps -- needs the heavy tier importable (torch / transformers /
# scikit-learn / matplotlib). No model download, no run artifacts.
# Measured, not assumed; see pyproject.toml markers.
pytestmark = pytest.mark.deps

try:
    from p1_mstate_tracking.clustering import cluster_count_sweep
    from sklearn.metrics import adjusted_rand_score
    _HAS_CLUSTERING_STACK = True
except ImportError:
    _HAS_CLUSTERING_STACK = False


# ============================================================================
# Oracle 1 — identity weights: monotone energy, eventual collapse
# ============================================================================

def _simulate_identity_weight_dynamics(
    n_tokens: int = 40, d: int = 16, beta: float = 4.0,
    step: float = 0.3, n_layers: int = 40, seed: int = 0,
) -> list[np.ndarray]:
    """
    Forward-Euler discretization of the Geshkovski et al. interacting-
    particle ODE with identity query/key/value maps (i.e. attention scores
    and the aggregated "value" are computed directly on x_i, x_j with no
    learned projection — the "identity weights" case named in the plan):

        x_{l+1,i} = normalize( x_{l,i} + step * sum_j softmax_j(beta <x_i,x_j>) x_j )

    Returns the list of per-layer (n_tokens, d) activations, x normalized
    to the unit sphere at every layer (mirrors core.models.layernorm_to_sphere).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_tokens, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    layers = [X.copy()]
    for _ in range(n_layers):
        G = X @ X.T
        W = np.exp(beta * G)
        W = W / W.sum(axis=1, keepdims=True)
        X = X + step * (W @ X)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        layers.append(X.copy())
    return layers


class TestIdentityWeightsMonotoneEnergyAndCollapse:

    BETA = 4.0

    @pytest.fixture(scope="class")
    def layers(self):
        return _simulate_identity_weight_dynamics(beta=self.BETA)

    def test_energy_is_monotonically_nondecreasing(self, layers):
        energies = [interaction_energy(L, self.BETA) for L in layers]
        diffs = np.diff(energies)
        # Forward-Euler with a fixed step can have a negligible numerical
        # wobble; the prediction is about the trend, not bit-exact monotonicity.
        assert (diffs >= -1e-6).all(), (
            f"energy decreased somewhere — pipeline's own energy function "
            f"disagrees with the theory on its own constructed input. "
            f"diffs={diffs}"
        )

    def test_energy_rises_substantially(self, layers):
        e_first = interaction_energy(layers[0], self.BETA)
        e_last  = interaction_energy(layers[-1], self.BETA)
        assert e_last > 5 * e_first, (
            f"expected a large energy rise under identity-weight dynamics, "
            f"got {e_first:.4f} -> {e_last:.4f}"
        )

    def test_effective_rank_collapses(self, layers):
        rank_first = effective_rank(layers[0], mode="raw")
        rank_last  = effective_rank(layers[-1], mode="raw")
        assert rank_last < 1.5, (
            f"expected near-total directional collapse (eff. rank -> 1) "
            f"at the final layer, got {rank_last:.4f}"
        )
        assert rank_last < rank_first / 5, (
            f"expected effective rank to collapse by more than 5x, "
            f"got {rank_first:.4f} -> {rank_last:.4f}"
        )

    def test_no_collapse_without_interaction(self):
        """
        Negative control: with step=0 (no interaction term at all), energy
        and effective rank must stay flat across layers. This is what rules
        out "the metric just always looks like this regardless of input."
        """
        layers = _simulate_identity_weight_dynamics(step=0.0, n_layers=10)
        energies   = [interaction_energy(L, self.BETA) for L in layers]
        eff_ranks  = [effective_rank(L, mode="raw") for L in layers]
        assert max(energies) - min(energies) < 1e-9
        assert max(eff_ranks) - min(eff_ranks) < 1e-6


# ============================================================================
# Oracle 2 — planted synthetic clusters recovered by the clustering stack
# ============================================================================

@pytest.mark.skipif(not _HAS_CLUSTERING_STACK,
                     reason="p1_mstate_tracking.clustering / sklearn not importable")
class TestPlantedClustersRecoveredByClusteringStack:

    def _planted_clusters(self, k: int = 4, n_per: int = 15, d: int = 12,
                           sep: float = 5.0, noise: float = 0.05, seed: int = 1):
        rng = np.random.default_rng(seed)
        centers = rng.standard_normal((k, d))
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * sep

        X, true_labels = [], []
        for ci in range(k):
            pts = centers[ci] + rng.standard_normal((n_per, d)) * noise
            X.append(pts)
            true_labels += [ci] * n_per
        return np.vstack(X).astype(np.float32), np.array(true_labels)

    def test_kmeans_recovers_planted_k_and_labels(self):
        k = 4
        X, true_labels = self._planted_clusters(k=k)
        result = cluster_count_sweep(X)
        assert result["kmeans"]["best_k"] == k, (
            f"KMeans/silhouette selection picked k={result['kmeans']['best_k']}, "
            f"planted k={k}"
        )
        ari = adjusted_rand_score(true_labels, result["kmeans"]["labels"])
        assert ari > 0.95, f"expected near-perfect ARI on well-separated planted clusters, got {ari:.3f}"

    def test_hdbscan_recovers_planted_labels_when_available(self):
        k = 4
        X, true_labels = self._planted_clusters(k=k)
        result = cluster_count_sweep(X)
        if "hdbscan" not in result:
            pytest.skip("hdbscan not installed in this environment")
        ari = adjusted_rand_score(true_labels, result["hdbscan"]["labels"])
        assert ari > 0.9, f"expected near-perfect HDBSCAN ARI on planted clusters, got {ari:.3f}"
        assert result["hdbscan"]["n_clusters"] == k, (
            f"HDBSCAN found {result['hdbscan']['n_clusters']} clusters, planted {k}"
        )

    def test_single_blob_does_not_overcluster(self):
        """
        Negative control: a single well-mixed Gaussian blob (k=1 planted)
        should not be sliced into several spurious clusters by the
        silhouette-selected KMeans.
        """
        rng = np.random.default_rng(7)
        X = rng.standard_normal((60, 12)).astype(np.float32) * 0.3
        result = cluster_count_sweep(X)
        assert result["kmeans"]["best_k"] <= 2, (
            f"expected k<=2 on an unstructured blob, got {result['kmeans']['best_k']}"
        )


# ============================================================================
# Oracle 3 — constructed rotation recovered by the Schur split
# ============================================================================

class TestConstructedRotationRecoveredBySchurSplit:

    def _planted_matrix(self, theta: float, rho: float,
                         real_eigs=(0.5, -0.3, 0.1, -0.7), seed: int = 2):
        """
        Block-diagonal [2x2 rotation block(theta, rho)] + [diag(real_eigs)],
        conjugated by a random orthogonal basis change so the planted
        structure is not axis-aligned (unlike test_phase2i.py's pure
        block-rotation constructions, which are already axis-aligned by
        construction).
        """
        a = rho * np.cos(theta)
        b = rho * np.sin(theta)
        c = -rho * np.sin(theta)
        d = 2 + len(real_eigs)

        M = np.zeros((d, d))
        M[:2, :2] = [[a, b], [c, a]]
        M[2:, 2:] = np.diag(real_eigs)

        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        return Q @ M @ Q.T

    @pytest.mark.parametrize("theta", [0.3, 0.7, 1.0, 1.4])
    @pytest.mark.parametrize("rho", [0.5, 2.0])
    def test_theta_and_rho_recovered_under_arbitrary_basis(self, theta, rho):
        OV = self._planted_matrix(theta, rho)
        result = extract_schur_blocks(OV)

        assert result["n_complex"] == 1, (
            f"expected exactly one 2x2 block, got {result['n_complex']} "
            f"(n_real={result['n_real']})"
        )
        assert result["n_real"] == 4, (
            f"expected exactly 4 real 1x1 blocks, got {result['n_real']}"
        )

        blk = result["blocks_2x2"][0]
        np.testing.assert_allclose(blk["theta"], theta, atol=1e-6)
        np.testing.assert_allclose(blk["rho"], rho, atol=1e-6)

    def test_dimension_invariant_holds(self):
        OV = self._planted_matrix(theta=0.9, rho=1.3)
        result = extract_schur_blocks(OV)
        assert result["n_real"] + 2 * result["n_complex"] == result["d"]

    def test_pure_real_spectrum_has_no_complex_blocks(self):
        """Negative control: a diagonal (already-real) matrix conjugated by
        a random orthogonal basis must still show zero 2x2 blocks."""
        d = 6
        rng = np.random.default_rng(3)
        M = np.diag(rng.uniform(-1, 1, size=d))
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        OV = Q @ M @ Q.T
        result = extract_schur_blocks(OV)
        assert result["n_complex"] == 0
        assert result["n_real"] == d


# ============================================================================
# Oracle 4 — planted-subspace degeneracy ratio (eigenspace_degeneracy.py)
#
# Doubles as the regression/coverage test status-6.md names for known bug 1
# ("eigenspace_degeneracy.py — NameError, `d` undefined"). That specific
# error does not reproduce against the current source (traced: `d` in
# degeneracy_sweep is a local assigned from X.shape[1] before any use; no
# bare `d` appears anywhere else in the module) — most likely already fixed
# in the same pass that produced this file's two documented fixes, without
# the fix being added to that list or to status-6.md. Either way, neither
# degeneracy_sweep nor run_eigenspace_degeneracy had any test coverage
# before this (test_phase6.py only covers project_to_subspace,
# degeneracy_ratio, lda_direction, subspace_alignment), so this closes that
# gap regardless of whether the original bug still exists. status-6.md
# itself flags this exact function as "a candidate for the oracle-tier
# suite since a degeneracy-ratio computation has a known-correct answer on
# planted synthetic clusters" — this is that test.
# ============================================================================

class TestPlantedSubspaceDegeneracyRatio:

    def _planted_data(self, d: int = 40, k_planted: int = 4,
                       n_clusters: int = 3, n_per: int = 25,
                       center_scale: float = 5.0, noise: float = 0.2,
                       seed: int = 0):
        """
        Cluster centroids live entirely inside a planted k_planted-dim
        subspace (U_pos); every other direction (U_neg, U_A) carries only
        i.i.d. noise with no cluster-dependent structure. Basis is a random
        orthogonal Q so the planted subspace is not axis-aligned.
        """
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        U_pos = Q[:, :k_planted]
        U_neg = Q[:, k_planted:2 * k_planted]
        U_A   = Q[:, 2 * k_planted:]

        centers = rng.standard_normal((n_clusters, k_planted)) * center_scale
        X, labels = [], []
        for ci in range(n_clusters):
            coeffs     = centers[ci] + rng.standard_normal((n_per, k_planted)) * noise
            noise_rest = rng.standard_normal((n_per, d - k_planted)) * noise
            pts = coeffs @ U_pos.T + noise_rest @ np.hstack([U_neg, U_A]).T
            X.append(pts)
            labels += [ci] * n_per
        X = np.vstack(X).astype(np.float32)
        return X, np.array(labels), U_pos, U_neg, U_A

    def test_ratio_high_in_planted_subspace_low_elsewhere(self):
        X, labels, U_pos, U_neg, _ = self._planted_data()
        ratio_pos = degeneracy_ratio(project_to_subspace(X, U_pos), labels)["ratio"]
        ratio_neg = degeneracy_ratio(project_to_subspace(X, U_neg), labels)["ratio"]

        assert ratio_pos >= 5.0, (
            f"expected R>=5 in the subspace clusters actually live in "
            f"(P6-R1's own threshold), got {ratio_pos:.2f}"
        )
        assert ratio_neg < 2.0, (
            f"expected near-null R in a same-dimension subspace with no "
            f"planted cluster structure, got {ratio_neg:.4f}"
        )
        assert ratio_pos > 50 * ratio_neg

    def test_full_pipeline_runs_without_error_and_p6r1_passes(self):
        """
        Guards the previously-reported NameError directly: this calls
        run_eigenspace_degeneracy end-to-end (degeneracy_sweep +
        run_eigenspace_degeneracy's LDA-alignment path together), which had
        no prior test at all.
        """
        X, labels, U_pos, U_neg, U_A = self._planted_data()
        ctx = {
            "activations_per_layer": [X, X],
            "labels_per_layer":      [labels, labels],
            "layer_type_labels":     ["plateau", "plateau"],
            "projectors": {
                "per_layer": [{"U_pos": U_pos, "U_neg": U_neg, "U_A": U_A}] * 2,
                "d_model": U_pos.shape[0],
            },
            "layer_names": ["L0", "L1"],
        }
        result = run_eigenspace_degeneracy(ctx)   # must not raise
        assert result.applicable is True
        assert result.payload["n_p6r1_pass"] == result.payload["n_plateau_layers"] == 2
