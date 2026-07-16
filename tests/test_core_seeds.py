"""
tests/test_core_seeds.py — Tests for core/seeds.py (core foundations
item 6: seed policy + stability reporting).
"""

import numpy as np
import pytest

from core.seeds import (
    set_all_seeds, cluster_stability_across_seeds, stability_report_per_layer,
    run_clustering_over_seeds,
)


class TestSetAllSeeds:
    def test_reports_python_and_numpy_always_set(self):
        flags = set_all_seeds(42)
        assert flags["python"] is True
        assert flags["numpy"] is True

    def test_reproducible_numpy_draw(self):
        set_all_seeds(123)
        a = np.random.rand(5)
        set_all_seeds(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_torch_flag_never_raises_when_torch_absent(self):
        # Whether torch is installed or not in the running environment,
        # this must return a dict, not raise.
        flags = set_all_seeds(0)
        assert isinstance(flags["torch"], bool)


class TestClusterStabilityAcrossSeeds:
    def test_identical_labelings_give_ari_one(self):
        labels = [np.array([0, 0, 1, 1, -1]) for _ in range(4)]
        result = cluster_stability_across_seeds(labels)
        assert result["n_seeds"] == 4
        assert len(result["pairwise_ari"]) == 6  # C(4, 2)
        assert result["mean_ari"] == pytest.approx(1.0)
        assert result["low_stability"] is False

    def test_relabeling_invariant(self):
        """ARI must not penalize an arbitrary cluster-id relabeling of an
        otherwise identical partition — HDBSCAN gives no guarantee that
        cluster 3 means the same thing across seeds."""
        a = np.array([1, 1, 0, 0, -1])
        b = np.array([0, 0, 1, 1, -1])
        result = cluster_stability_across_seeds([a, b])
        assert result["mean_ari"] == pytest.approx(1.0)

    def test_unrelated_labelings_flagged_low_stability(self):
        rng = np.random.default_rng(0)
        labels = [rng.integers(0, 4, size=200) for _ in range(5)]
        result = cluster_stability_across_seeds(labels, threshold=0.5)
        assert result["mean_ari"] < 0.3
        assert result["low_stability"] is True

    def test_requires_at_least_two_seeds(self):
        with pytest.raises(ValueError):
            cluster_stability_across_seeds([np.array([0, 1])])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            cluster_stability_across_seeds([np.array([0, 1]), np.array([0, 1, 2])])

    def test_threshold_is_echoed_back(self):
        labels = [np.array([0, 0, 1, 1]) for _ in range(2)]
        result = cluster_stability_across_seeds(labels, threshold=0.7)
        assert result["threshold"] == 0.7


class TestStabilityReportPerLayer:
    def test_reports_one_entry_per_layer(self):
        stable = [np.array([0, 0, 1, 1]) for _ in range(2)]
        rng = np.random.default_rng(1)
        unstable = [rng.integers(0, 4, size=100) for _ in range(2)]
        report = stability_report_per_layer({0: stable, 1: unstable}, threshold=0.5)
        assert set(report.keys()) == {0, 1}
        assert report[0]["low_stability"] is False
        assert report[1]["low_stability"] is True


class TestRunClusteringOverSeeds:
    def test_calls_cluster_fn_per_seed_and_reports_stability(self):
        def fake_cluster_fn(activations, seed):
            n = activations.shape[0]
            if seed in (1, 2):
                return np.array([0] * (n // 2) + [1] * (n - n // 2))
            return np.random.default_rng(seed).integers(0, 2, size=n)

        acts = np.zeros((10, 4))
        out = run_clustering_over_seeds(acts, seeds=[1, 2], cluster_fn=fake_cluster_fn)
        assert out["mean_ari"] == pytest.approx(1.0)
        assert len(out["labels_per_seed"]) == 2
