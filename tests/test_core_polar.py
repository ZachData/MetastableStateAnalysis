"""
tests/test_core_polar.py — Oracle tests for core/polar.py (frames item 1).

Pure numpy tier: runs under the stubbed (torch-free) session. Every test
checks an exact mathematical property, not a pipeline behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.polar import (
    particle_norms,
    polar_decompose,
    raw_gram,
    norm_stats,
    cluster_norm_profile,
    sphere_gap,
    polar_layer_record,
)

_rng = np.random.default_rng(1234)


def _rand_acts(n=12, d=32, scale_spread=0.0):
    """Random activations; scale_spread > 0 plants per-row norm variation."""
    X = _rng.normal(size=(n, d))
    if scale_spread > 0:
        scales = np.exp(_rng.normal(scale=scale_spread, size=(n, 1)))
        X = X * scales
    return X


# ---------------------------------------------------------------------------
# Polar coordinates — exact identities
# ---------------------------------------------------------------------------

class TestPolarDecompose:
    def test_reconstruction_identity(self):
        """x == r * x_hat exactly (up to float64 eps)."""
        X = _rand_acts(scale_spread=1.0)
        r, xh = polar_decompose(X)
        np.testing.assert_allclose(r[:, None] * xh, X, rtol=1e-12, atol=1e-12)

    def test_directions_unit_norm(self):
        X = _rand_acts(scale_spread=1.0)
        _, xh = polar_decompose(X)
        np.testing.assert_allclose(np.linalg.norm(xh, axis=-1), 1.0, atol=1e-12)

    def test_norms_match_numpy(self):
        X = _rand_acts()
        np.testing.assert_allclose(
            particle_norms(X), np.linalg.norm(X, axis=-1), rtol=1e-12
        )


class TestRawGramIdentity:
    def test_raw_equals_rrg(self):
        """raw_ij == r_i r_j g_ij — the decomposition the whole module
        rests on. Checked exactly."""
        X = _rand_acts(scale_spread=0.8)
        r = particle_norms(X)
        from core.metrics import gram_matrix
        G_cos = gram_matrix(X)
        expected = np.outer(r, r) * G_cos
        np.testing.assert_allclose(raw_gram(X), expected, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Norm stats
# ---------------------------------------------------------------------------

class TestNormStats:
    def test_planted_outlier_detected(self):
        """A single 50x-norm row must top top_outlier_indices and drive
        max_over_median."""
        X = _rand_acts(n=10, d=16)
        X = X / np.linalg.norm(X, axis=-1, keepdims=True)   # all norms = 1
        X[7] *= 50.0
        s = norm_stats(particle_norms(X))
        assert s["top_outlier_indices"][0] == 7
        assert s["max_over_median"] > 40.0

    def test_equal_norms_zero_log_std(self):
        X = _rand_acts(n=8, d=16)
        X = 3.0 * X / np.linalg.norm(X, axis=-1, keepdims=True)
        s = norm_stats(particle_norms(X))
        assert s["log_std"] < 1e-10
        assert abs(s["max_over_median"] - 1.0) < 1e-10

    def test_empty_input_safe(self):
        s = norm_stats(np.array([]))
        assert s["n"] == 0 and s["top_outlier_indices"] == []

    def test_json_serializable(self):
        import json
        json.dumps(norm_stats(particle_norms(_rand_acts())))


class TestClusterNormProfile:
    def test_planted_norm_cluster_coupling(self):
        """Clustered rows at norm 10, noise rows at norm 1 ->
        clustered_minus_noise_mean == 9 exactly."""
        X = _rand_acts(n=8, d=16)
        X = X / np.linalg.norm(X, axis=-1, keepdims=True)
        labels = np.array([0, 0, 0, 1, 1, -1, -1, -1])
        X[labels >= 0] *= 10.0
        prof = cluster_norm_profile(particle_norms(X), labels)
        assert abs(prof["clustered_minus_noise_mean"] - 9.0) < 1e-10
        assert prof["per_label"][0]["n"] == 3
        assert prof["per_label"][-1]["n"] == 3

    def test_no_noise_gives_nan(self):
        r = np.ones(4)
        prof = cluster_norm_profile(r, np.array([0, 0, 1, 1]))
        assert np.isnan(prof["clustered_minus_noise_mean"])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cluster_norm_profile(np.ones(4), np.zeros(3, dtype=int))


# ---------------------------------------------------------------------------
# Sphere gap
# ---------------------------------------------------------------------------

class TestSphereGap:
    def test_equal_norms_zero_gap(self):
        """All norms equal -> raw Gram is a positive multiple of the
        cosine Gram -> both gaps exactly 0."""
        X = _rand_acts(n=15, d=24)
        X = 2.5 * X / np.linalg.norm(X, axis=-1, keepdims=True)
        g = sphere_gap(X)
        assert g["pearson_gap"] < 1e-10
        assert g["spearman_gap"] < 1e-10
        assert g["norm_log_std"] < 1e-10

    def test_norm_spread_opens_gap(self):
        """Planted heavy norm spread must produce a positive Pearson gap
        and rank distortion (raw ordering != cosine ordering)."""
        X = _rand_acts(n=20, d=24, scale_spread=1.5)
        g = sphere_gap(X)
        assert g["pearson_gap"] > 0.01
        assert g["norm_log_std"] > 0.5
        # Rank distortion: with heavy spread the most-similar pair by raw
        # IP is (almost surely under this seed) not the most-similar by
        # cosine — spearman gap must register it.
        assert g["spearman_gap"] > 0.0

    def test_gap_monotone_in_spread(self):
        """Same directions, increasing norm spread -> non-decreasing gap.
        Directions held fixed so ONLY the norm channel moves."""
        n, d = 20, 24
        U = _rand_acts(n, d)
        U = U / np.linalg.norm(U, axis=-1, keepdims=True)
        base_scales = _rng.normal(size=(n, 1))
        gaps = []
        for spread in (0.0, 0.5, 1.5):
            X = U * np.exp(spread * base_scales)
            gaps.append(sphere_gap(X)["pearson_gap"])
        assert gaps[0] < 1e-10
        assert gaps[0] <= gaps[1] <= gaps[2]

    def test_single_particle_safe(self):
        g = sphere_gap(_rand_acts(n=1))
        assert np.isnan(g["pearson_gap"]) and g["n_pairs"] == 0


class TestPolarLayerRecord:
    def test_record_shape_and_serializability(self):
        import json
        X = _rand_acts(n=9, d=16, scale_spread=0.5)
        labels = np.array([0, 0, 0, 1, 1, 1, -1, -1, -1])
        rec = polar_layer_record(X, labels)
        assert len(rec["norms"]) == 9
        assert "sphere_gap" in rec and "norm_stats" in rec
        assert "cluster_norm_profile" in rec
        json.dumps(rec)

    def test_labels_optional(self):
        rec = polar_layer_record(_rand_acts())
        assert "cluster_norm_profile" not in rec
