"""
tests/test_phase1d_gaussian_null.py — the matched-covariance Gaussian null
(`p1d_cluster_ensemble/gaussian_null.py`): frames, the draw's moments, and
that the null separates planted clusters from a Gaussian cloud.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.deps

from p1d_cluster_ensemble.gaussian_null import (
    FRAMES, _step_prompt, compare, frame_vectors, gaussian_draw, mean_direction,
    null_record, rogue_dims, span_coordinates, summarise,
)


def _unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _offset_cloud(n=120, d=40, seed=0):
    """A shared direction plus two rogue coordinates, like 410m's late layers."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:, 7] += 6.0
    X[:, 3] += 3.0
    X += 0.5 * rng.standard_normal(d)
    return _unit(X)


def _two_blobs(n=80, d=30, sep=4.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[: n // 2, 0] += sep
    X[n // 2:, 0] -= sep
    X[:, 1] += 5.0
    return _unit(X)


class TestFrames:
    def test_rows_are_unit_in_every_frame(self):
        Y = _offset_cloud()
        for f in FRAMES:
            Z, info = frame_vectors(Y, f)
            assert np.allclose(np.linalg.norm(Z, axis=1), 1.0)
            assert info["frame"] == f and info["eff_dim"] > 0

    def test_rogue_dims_are_the_planted_offsets(self):
        assert list(rogue_dims(_offset_cloud(), 2)) == [7, 3]

    def test_centred_rows_are_orthogonal_to_the_mean_direction(self):
        Y = _offset_cloud()
        Z, _ = frame_vectors(Y, "centred")
        assert np.allclose(Z @ mean_direction(Y), 0.0, atol=1e-12)

    def test_norogue_zeroes_the_rogue_coordinates(self):
        Y = _offset_cloud()
        Z, info = frame_vectors(Y, "centred_norogue")
        assert np.allclose(Z[:, info["rogue"]], 0.0, atol=1e-12)

    def test_centring_lowers_the_mean_share(self):
        Y = _offset_cloud()
        Z, info = frame_vectors(Y, "centred")
        assert info["mean_share"] > 0.3
        assert float(((Z @ mean_direction(Z)) ** 2).mean()) < 0.05

    def test_unknown_frame_refuses(self):
        with pytest.raises(ValueError):
            frame_vectors(_offset_cloud(), "whitened")


class TestDraw:
    def test_span_coordinates_keep_the_gram_matrix(self):
        Y = _offset_cloud(n=30, d=60)
        S = span_coordinates(Y)
        assert S.shape[1] <= 30
        assert np.allclose(S @ S.T, Y @ Y.T, atol=1e-10)

    def test_draws_match_mean_covariance_and_norm(self):
        Y = _offset_cloud(n=20, d=8)
        rng = np.random.default_rng(1)
        X = np.concatenate([gaussian_draw(Y, rng, renorm=False) for _ in range(3000)])
        mu = Y.mean(axis=0)
        cov = np.cov(Y.T, ddof=0)
        assert np.allclose(X.mean(axis=0), mu, atol=0.02)
        assert np.allclose(np.cov(X.T, ddof=0), cov, atol=0.02)
        assert abs(float((X ** 2).sum(axis=1).mean()) - 1.0) < 0.02

    def test_renormed_draws_are_unit(self):
        X = gaussian_draw(_offset_cloud(), np.random.default_rng(0))
        assert np.allclose(np.linalg.norm(X, axis=1), 1.0)


class TestNull:
    def test_planted_blobs_beat_the_null(self):
        rec = null_record(_two_blobs(), "raw", n_draws=19, seed=0)
        assert rec["stats"]["ci2"]["p_lower"] <= 0.05
        assert rec["stats"]["mt_life"]["p_upper"] <= 0.05

    def test_a_gaussian_cloud_does_not(self):
        # Only the lumpier tail is asserted. The plug-in null is biased
        # towards lumpy (its draws' spectrum spreads more than the data's),
        # so at this small n a true Gaussian lands in the null's *upper*
        # ci2 tail; `--calibrate` measures the bias at real sizes.
        rng = np.random.default_rng(3)
        Y = _unit(rng.standard_normal((80, 30)) * np.linspace(3, 0.5, 30) + 4.0)
        rec = null_record(Y, "raw", n_draws=39, seed=0)
        assert rec["stats"]["ci2"]["p_lower"] > 0.05
        assert rec["stats"]["nn1"]["p_lower"] > 0.05

    def test_compare_counts_ties_in_both_tails(self):
        s = compare(1.0, np.array([0.0, 1.0, 2.0]))
        assert s["p_lower"] == pytest.approx(3 / 4) and s["p_upper"] == pytest.approx(3 / 4)

    def test_summary_counts_tails(self):
        rec = null_record(_two_blobs(), "raw", n_draws=19, seed=0, keep_draws=False)
        rec.update(step="step143000", layer=12)
        rows = [r for r in summarise([rec], alpha=0.05) if r["stat"] == "ci2"]
        assert rows == [dict(rows[0], n=1, below=1, above=0)]
        assert rows[0]["band"] == "L9-16"


def test_step_prompt_parses_run_names():
    from pathlib import Path
    assert _step_prompt(Path("x/pythia-410m-step0_wiki_paragraph")) == ("step0", "wiki_paragraph")


def test_calibrate_replaces_the_tokens():
    real = null_record(_two_blobs(), "raw", n_draws=9, seed=0, keep_draws=False)
    cal = null_record(_two_blobs(), "raw", n_draws=9, seed=0, keep_draws=False, calibrate=True)
    assert cal["info"]["calibrate"] is True
    assert cal["stats"]["ci2"]["obs"] > real["stats"]["ci2"]["obs"]


def test_first_occurrences_keep_order_and_the_first_copy():
    from p1d_cluster_ensemble.gaussian_null import first_occurrences
    assert first_occurrences(["a", "b", "a", "c", "b"]).tolist() == [0, 1, 3]
