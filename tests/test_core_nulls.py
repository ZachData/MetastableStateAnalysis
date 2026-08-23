"""
tests/test_core_nulls.py — Tests for core/nulls.py (core foundations
item 7: null distributions as first-class outputs).
"""

import numpy as np
import pytest

from core.nulls import (
    shuffled_dimension_null, label_permutation_null,
    sigma_from_null, nsigma_verdict,
)

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure


def _mass_near_1(acts, threshold=0.9):
    n = acts.shape[0]
    G = acts @ acts.T
    iu = np.triu_indices(n, k=1)
    return float((np.abs(G[iu]) > threshold).mean())


def _joint_direction_clusters(n=40, d=8, noise=0.15, seed=0):
    """
    Two clusters along random (non-axis-aligned) unit directions. Deliberately
    NOT a single dominant coordinate (e.g. X[:, 0] = +-1): that construction
    gives mass_near_1 = 1.0 under any row permutation (abs-value pair-counting
    on a rank-1 structure is permutation-invariant), so it wouldn't distinguish
    "structure destroyed" from "structure trivially preserved." The joint
    multi-dimension direction below is what shuffled_dimension_null is
    actually meant to destroy: correlation *across* dimensions.
    """
    rng = np.random.default_rng(seed)
    u1 = rng.standard_normal(d); u1 /= np.linalg.norm(u1)
    u2 = rng.standard_normal(d); u2 /= np.linalg.norm(u2)
    X = np.zeros((n, d))
    X[: n // 2] = u1
    X[n // 2 :] = u2
    X += rng.standard_normal((n, d)) * noise
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


class TestShuffledDimensionNull:
    def test_shape(self):
        X = _joint_direction_clusters()
        null = shuffled_dimension_null(X, _mass_near_1, n_shuffles=50, rng=np.random.default_rng(1))
        assert null.shape == (50,)

    def test_destroys_joint_cluster_structure(self):
        X = _joint_direction_clusters()
        observed = _mass_near_1(X)
        null = shuffled_dimension_null(X, _mass_near_1, n_shuffles=100, rng=np.random.default_rng(1))
        assert null.mean() < observed

    def test_renormalize_false_skips_normalization(self):
        X = _joint_direction_clusters()
        null_norm = shuffled_dimension_null(
            X, _mass_near_1, n_shuffles=20, renormalize=True, rng=np.random.default_rng(2)
        )
        null_raw = shuffled_dimension_null(
            X, _mass_near_1, n_shuffles=20, renormalize=False, rng=np.random.default_rng(2)
        )
        # Same rng draws, different post-processing -> generally different values.
        assert not np.array_equal(null_norm, null_raw)


class TestLabelPermutationNull:
    def test_shape(self):
        X = _joint_direction_clusters()
        labels = np.array([0] * 20 + [1] * 20)

        def frac(acts, lbls):
            return float((lbls == 0).mean())

        null = label_permutation_null(X, labels, frac, n_permutations=30, rng=np.random.default_rng(3))
        assert null.shape == (30,)

    def test_real_labeling_beats_permuted_labels(self):
        X = _joint_direction_clusters()
        labels = np.array([0] * 20 + [1] * 20)

        def cluster_internal_fraction(acts, lbls):
            n = acts.shape[0]
            G = acts @ acts.T
            iu = np.triu_indices(n, k=1)
            high = np.abs(G[iu]) > 0.9
            if high.sum() == 0:
                return 0.0
            same = (lbls[iu[0]] == lbls[iu[1]])
            return float((high & same).sum() / high.sum())

        observed = cluster_internal_fraction(X, labels)
        null = label_permutation_null(
            X, labels, cluster_internal_fraction, n_permutations=200, rng=np.random.default_rng(4)
        )
        assert observed > null.mean()


class TestSigmaFromNull:
    def test_basic_fields(self):
        null = np.random.default_rng(0).standard_normal(500)
        summary = sigma_from_null(3.0, null)
        assert summary["observed"] == 3.0
        assert summary["n_null"] == 500
        assert summary["z_score"] > 0

    def test_observed_at_null_mean_is_near_50th_percentile(self):
        null = np.random.default_rng(0).standard_normal(2000)
        summary = sigma_from_null(float(np.mean(null)), null)
        assert 45 <= summary["percentile"] <= 55

    def test_degenerate_null_gives_nan_zscore_not_a_crash(self):
        null = np.full(50, 0.42)
        summary = sigma_from_null(0.9, null)
        assert np.isnan(summary["z_score"])


class TestNsigmaVerdict:
    def test_significant_flag_and_string(self):
        null = np.random.default_rng(0).standard_normal(500)
        verdict = nsigma_verdict(10.0, null, sigma_threshold=2.0)
        assert verdict["significant"] is True
        assert "σ from null" in verdict["verdict_str"]

    def test_not_significant_below_threshold(self):
        null = np.random.default_rng(0).standard_normal(500)
        verdict = nsigma_verdict(float(np.mean(null)), null, sigma_threshold=2.0)
        assert verdict["significant"] is False

    def test_degenerate_null_not_significant(self):
        null = np.full(50, 0.42)
        verdict = nsigma_verdict(0.9, null)
        assert verdict["significant"] is False
        assert verdict["verdict_str"] == "nanσ from null (not significant)"
