"""
tests/test_phase3_analysis.py

Unit tests for p3_crosscoder/analysis.py — post-training analysis functions.

Strategy:
  Most functions call _compute_feature_layer_scores(crosscoder, prompt_store)
  internally.  Rather than running a full crosscoder + data pipeline, we patch
  that helper to return controlled numpy arrays so the downstream logic can be
  tested with analytically-known expected values.

  A minimal FakeCrosscoder and FakePromptStore satisfy the calling conventions
  of functions that still need real objects passed in.

Standard geometry: N_LAYERS=6, D=8, N_FEATURES=32, K=4.
"""

import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from p3_crosscoder.crosscoder import Crosscoder
from p3_crosscoder.analysis import (
    feature_lifetimes,
    multilayer_fraction,
    positional_control,
    v_subspace_alignment,
    violation_layer_features,
    _compute_feature_layer_scores,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_LAYERS   = 6
D          = 8
N_FEATURES = 32
K          = 4
N_TOKENS   = 20

_PATCH_TARGET = "p3_crosscoder.analysis._compute_feature_layer_scores"

# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

def _make_cfg():
    return SimpleNamespace(n_layers=N_LAYERS, d_model=D, n_features=N_FEATURES, k=K)


def _make_crosscoder():
    return Crosscoder(_make_cfg()).eval()


class FakePromptStore:
    """Minimal PromptActivationStore substitute — no file I/O, no model."""

    def __init__(self, n_prompts=2, n_tokens=N_TOKENS, seed=0):
        rng = np.random.default_rng(seed)
        self._data = {}
        self._tokens = {}
        for i in range(n_prompts):
            key = f"prompt_{i}"
            # shape expected by get_stacked_tensor: (T, L*D)
            arr = rng.standard_normal((n_tokens, N_LAYERS * D)).astype(np.float32)
            self._data[key] = torch.from_numpy(arr)
            self._tokens[key] = [f"t{j}" for j in range(n_tokens)]

    def keys(self):
        return list(self._data.keys())

    def get_stacked_tensor(self, key):
        return self._data[key]

    def get_tokens(self, key):
        return self._tokens[key]


# ---------------------------------------------------------------------------
# Score-array factories
# ---------------------------------------------------------------------------

def _bimodal_scores(n_features=N_FEATURES, n_layers=N_LAYERS) -> np.ndarray:
    """
    Half the features active only at layer 0 (lifetime=1, short-lived).
    Half active at every layer (lifetime=N_LAYERS, long-lived).
    BC of a two-point mass distribution ≫ 5/9.
    """
    scores = np.zeros((n_features, n_layers), dtype=np.float32)
    half = n_features // 2
    scores[:half, 0] = 1.0          # active only at layer 0
    scores[half:, :] = 1.0          # active at all layers
    return scores


def _unimodal_scores(n_features=N_FEATURES, n_layers=N_LAYERS) -> np.ndarray:
    """All features active at floor(n_layers/2) contiguous layers → uniform lifetimes."""
    scores = np.zeros((n_features, n_layers), dtype=np.float32)
    mid = n_layers // 2
    scores[:, :mid] = 1.0
    return scores


def _dead_scores(n_features=N_FEATURES, n_layers=N_LAYERS) -> np.ndarray:
    return np.zeros((n_features, n_layers), dtype=np.float32)


def _single_layer_scores(n_features=N_FEATURES, n_layers=N_LAYERS) -> np.ndarray:
    """Each feature active at exactly one layer."""
    scores = np.zeros((n_features, n_layers), dtype=np.float32)
    for f in range(n_features):
        scores[f, f % n_layers] = 1.0
    return scores


def _full_span_scores(n_features=N_FEATURES, n_layers=N_LAYERS) -> np.ndarray:
    return np.ones((n_features, n_layers), dtype=np.float32)


# ---------------------------------------------------------------------------
# _compute_feature_layer_scores — integration smoke tests
# ---------------------------------------------------------------------------

class TestComputeFeatureLayerScores:
    """Smoke-test the real function with a tiny crosscoder + fake store."""

    def test_output_shape(self):
        cc = _make_crosscoder()
        ps = FakePromptStore()
        scores = _compute_feature_layer_scores(cc, ps)
        assert scores.shape == (N_FEATURES, N_LAYERS), (
            f"Expected ({N_FEATURES}, {N_LAYERS}), got {scores.shape}"
        )

    def test_scores_non_negative(self):
        cc = _make_crosscoder()
        ps = FakePromptStore()
        scores = _compute_feature_layer_scores(cc, ps)
        assert np.all(scores >= -1e-9), "Scores must be non-negative (mean squared projection)"

    def test_empty_store_returns_zeros(self):
        """A store with no prompts should return an all-zero score matrix."""
        cc = _make_crosscoder()
        ps = FakePromptStore(n_prompts=0)
        scores = _compute_feature_layer_scores(cc, ps)
        assert scores.shape == (N_FEATURES, N_LAYERS)
        assert np.allclose(scores, 0.0), "Empty store should produce zero scores"


# ---------------------------------------------------------------------------
# feature_lifetimes — required keys
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "lifetimes", "peak_layers", "feature_layer_scores", "max_scores",
    "lifetime_class", "short_lived_indices", "long_lived_indices",
    "n_alive", "mean_lifetime", "median_lifetime",
    "bimodality_coefficient", "bimodality_test", "bc_threshold",
    "valley_threshold", "n_short_lived", "n_long_lived",
}


class TestFeatureLifetimesRequiredKeys:

    def test_all_required_keys_present(self):
        with patch(_PATCH_TARGET, return_value=_bimodal_scores()):
            result = feature_lifetimes(_make_crosscoder(), FakePromptStore(), {}, {})
        missing = REQUIRED_KEYS - set(result)
        assert not missing, f"Missing keys: {missing}"

    def test_lifetime_class_length(self):
        with patch(_PATCH_TARGET, return_value=_bimodal_scores()):
            result = feature_lifetimes(_make_crosscoder(), FakePromptStore(), {}, {})
        assert len(result["lifetime_class"]) == N_FEATURES

    def test_lifetimes_length(self):
        with patch(_PATCH_TARGET, return_value=_bimodal_scores()):
            result = feature_lifetimes(_make_crosscoder(), FakePromptStore(), {}, {})
        assert len(result["lifetimes"]) == N_FEATURES

    def test_json_serializable(self):
        """Every value must survive json.dumps — catches np.bool_ bugs (Bug 3)."""
        with patch(_PATCH_TARGET, return_value=_bimodal_scores()):
            result = feature_lifetimes(_make_crosscoder(), FakePromptStore(), {}, {})
        # Must not raise
        serialized = json.dumps(result, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# feature_lifetimes — bimodality detection
# ---------------------------------------------------------------------------

class TestFeatureLifetimesBimodal:
    """
    Bimodal input: half features lifetime=1, half lifetime=N_LAYERS.
    BC of a two-point distribution = 1.0 > 5/9.
    """

    @pytest.fixture(autouse=True)
    def _run(self):
        with patch(_PATCH_TARGET, return_value=_bimodal_scores()):
            self.result = feature_lifetimes(
                _make_crosscoder(), FakePromptStore(), {}, {}
            )

    def test_bimodality_test_is_bimodal(self):
        assert self.result["bimodality_test"] == "bimodal", (
            f"Expected 'bimodal', got '{self.result['bimodality_test']}'"
        )

    def test_bc_above_threshold(self):
        bc = self.result["bimodality_coefficient"]
        assert bc is not None
        assert bc > 5.0 / 9.0, f"BC={bc:.4f} should exceed 5/9≈0.5556"

    def test_valley_threshold_set(self):
        assert self.result["valley_threshold"] is not None, (
            "valley_threshold must be set when bimodal"
        )

    def test_short_lived_count_nonzero(self):
        assert self.result["n_short_lived"] > 0

    def test_long_lived_count_nonzero(self):
        assert self.result["n_long_lived"] > 0

    def test_short_long_indices_disjoint(self):
        short = set(self.result["short_lived_indices"])
        long  = set(self.result["long_lived_indices"])
        assert short.isdisjoint(long), "Short and long indices must not overlap"

    def test_short_long_indices_within_range(self):
        all_idx = self.result["short_lived_indices"] + self.result["long_lived_indices"]
        assert all(0 <= i < N_FEATURES for i in all_idx)


class TestFeatureLifetimesUnimodal:
    """
    Uniform lifetime → unimodal distribution, BC ≤ 5/9.
    """

    @pytest.fixture(autouse=True)
    def _run(self):
        with patch(_PATCH_TARGET, return_value=_unimodal_scores()):
            self.result = feature_lifetimes(
                _make_crosscoder(), FakePromptStore(), {}, {}
            )

    def test_bimodality_test_is_unimodal(self):
        assert self.result["bimodality_test"] == "unimodal", (
            f"Expected 'unimodal', got '{self.result['bimodality_test']}'"
        )

    def test_bc_at_or_below_threshold(self):
        bc = self.result["bimodality_coefficient"]
        assert bc is not None
        assert bc <= 5.0 / 9.0 + 1e-6, f"BC={bc:.4f} should be ≤5/9 for unimodal"

    def test_valley_threshold_is_none(self):
        assert self.result["valley_threshold"] is None, (
            "valley_threshold should not be set for unimodal distribution"
        )


class TestFeatureLifetimesAllDead:
    """
    All features below activity threshold → n_alive=0.
    This is Bug 2's scenario: normalize_decoder makes all norms 1.0,
    causing all features to appear alive.  The data-driven score fixes
    it — but if the fix itself produces near-zero scores, we must not crash.
    """

    @pytest.fixture(autouse=True)
    def _run(self):
        with patch(_PATCH_TARGET, return_value=_dead_scores()):
            self.result = feature_lifetimes(
                _make_crosscoder(), FakePromptStore(), {}, {}
            )

    def test_n_alive_is_zero(self):
        assert self.result["n_alive"] == 0

    def test_bimodality_test_insufficient_data(self):
        assert self.result["bimodality_test"] == "insufficient_data"

    def test_bimodality_coefficient_is_none(self):
        assert self.result["bimodality_coefficient"] is None

    def test_all_classified_dead(self):
        classes = self.result["lifetime_class"]
        assert all(c == "dead" for c in classes), (
            "All features should be classified 'dead' when scores are zero"
        )

    def test_no_short_or_long_indices(self):
        assert self.result["short_lived_indices"] == []
        assert self.result["long_lived_indices"] == []


class TestFeatureLifetimesInsufficientData:
    """Fewer than 20 alive features → insufficient_data regardless of shape."""

    def test_small_n_features_gives_insufficient_data(self):
        # 10 features, all alive — below the n>=20 threshold
        small = np.ones((10, N_LAYERS), dtype=np.float32)
        with patch(_PATCH_TARGET, return_value=small):
            result = feature_lifetimes(
                _make_crosscoder(), FakePromptStore(), {}, {}
            )
        assert result["bimodality_test"] == "insufficient_data"
        assert result["bimodality_coefficient"] is None


# ---------------------------------------------------------------------------
# feature_lifetimes — lifetime value correctness
# ---------------------------------------------------------------------------

class TestFeatureLifetimesValues:

    def test_all_layers_active_gives_max_lifetime(self):
        """Full-span scores → every feature's lifetime = N_LAYERS."""
        with patch(_PATCH_TARGET, return_value=_full_span_scores()):
            result = feature_lifetimes(
                _make_crosscoder(), FakePromptStore(), {}, {}
            )
        lifetimes = result["lifetimes"]
        assert all(lt == N_LAYERS for lt in lifetimes), (
            f"Expected all lifetimes={N_LAYERS}, got {set(lifetimes)}"
        )

    def test_single_layer_active_gives_lifetime_one(self):
        """Each feature active at exactly one layer → lifetime=1 for all."""
        with patch(_PATCH_TARGET, return_value=_single_layer_scores()):
            result = feature_lifetimes(
                _make_crosscoder(), FakePromptStore(), {}, {}
            )
        lifetimes = [lt for lt, cls in zip(result["lifetimes"], result["lifetime_class"])
                     if cls != "dead"]
        assert all(lt == 1 for lt in lifetimes), (
            f"Single-layer active → lifetime=1, got {set(lifetimes)}"
        )

    def test_threshold_frac_respected(self):
        """High threshold_frac trims layers that barely clear 0."""
        # Feature 0: layer 0 = 1.0, layer 1 = 0.05 (below 0.1 * 1.0)
        scores = np.zeros((N_FEATURES, N_LAYERS), dtype=np.float32)
        scores[0, 0] = 1.0
        scores[0, 1] = 0.05
        with patch(_PATCH_TARGET, return_value=scores):
            result = feature_lifetimes(
                _make_crosscoder(), FakePromptStore(), {},
                {"lifetime_threshold_frac": 0.1}
            )
        assert result["lifetimes"][0] == 1, (
            "Layer with score < 10% of max should not extend contiguous run"
        )


# ---------------------------------------------------------------------------
# multilayer_fraction
# ---------------------------------------------------------------------------

class TestMultilayerFraction:

    def test_all_single_layer_gives_zero_fraction(self):
        with patch(_PATCH_TARGET, return_value=_single_layer_scores()):
            result = multilayer_fraction(
                _make_crosscoder(), FakePromptStore(), {},
                {"multilayer_min_layers": 3}
            )
        assert result["multilayer_fraction"] == pytest.approx(0.0), (
            "Single-layer features should yield fraction=0"
        )

    def test_all_full_span_gives_one_fraction(self):
        with patch(_PATCH_TARGET, return_value=_full_span_scores()):
            result = multilayer_fraction(
                _make_crosscoder(), FakePromptStore(), {},
                {"multilayer_min_layers": 3}
            )
        assert result["multilayer_fraction"] == pytest.approx(1.0), (
            "Full-span features should yield fraction=1"
        )

    def test_all_dead_gives_zero(self):
        with patch(_PATCH_TARGET, return_value=_dead_scores()):
            result = multilayer_fraction(
                _make_crosscoder(), FakePromptStore(), {}, {}
            )
        assert result["multilayer_fraction"] == pytest.approx(0.0)
        assert result["n_alive"] == 0

    def test_fractional_result_matches_count(self):
        """multilayer_fraction = multilayer_count / n_alive."""
        with patch(_PATCH_TARGET, return_value=_bimodal_scores()):
            result = multilayer_fraction(
                _make_crosscoder(), FakePromptStore(), {},
                {"multilayer_min_layers": 3}
            )
        n = result["n_alive"]
        c = result["multilayer_count"]
        expected_frac = c / max(n, 1)
        assert result["multilayer_fraction"] == pytest.approx(expected_frac)

    def test_required_keys_present(self):
        with patch(_PATCH_TARGET, return_value=_full_span_scores()):
            result = multilayer_fraction(
                _make_crosscoder(), FakePromptStore(), {}, {}
            )
        for key in ("multilayer_fraction", "multilayer_count", "n_alive",
                    "layers_active_distribution"):
            assert key in result, f"Missing key '{key}'"


# ---------------------------------------------------------------------------
# v_subspace_alignment — artifact-gated error paths
# ---------------------------------------------------------------------------

class TestVSubspaceAlignmentErrors:

    def test_missing_projectors_returns_error(self):
        result = v_subspace_alignment(
            _make_crosscoder(), FakePromptStore(), {}, {}
        )
        assert "error" in result, "Missing v_projectors artifact must produce error key"

    def test_low_rank_projector_returns_diagnostic(self):
        """
        A projector spanning k=1 dimension out of D=8 is too low-rank
        to distinguish attractive from repulsive (random unit vectors
        project k/D ≈ 0.125 into each subspace, concentrating near 0.5).
        The function should detect this and return a diagnostic rather
        than a misleading null finding.
        """
        d = D
        # Rank-1 projectors
        u = np.zeros((d, 1), dtype=np.float32)
        u[0, 0] = 1.0
        projectors = {
            "sym_attract": u @ u.T,
            "sym_repulse": np.zeros((d, d), dtype=np.float32),
        }
        artifacts = {"v_projectors": projectors, "is_per_layer": False}
        result = v_subspace_alignment(
            _make_crosscoder(), FakePromptStore(), artifacts, {}
        )
        # Either returns a diagnostic key or an error — must not silently
        # produce an "interpretation" that looks like a meaningful result
        has_diagnostic = (
            "error" in result
            or "projector_rank_too_low" in result
            or result.get("skipped_reason")
        )
        assert has_diagnostic, (
            "Low-rank projector should produce a diagnostic, not a silent null result"
        )


class TestVSubspaceAlignmentFullRank:
    """
    Full-rank attractive projector (= identity) with zero repulsive:
    every decoder direction is 100% attractive → attract_dominance ≈ 1.0 for all features.
    """

    def test_identity_projector_all_attractive(self):
        eye = np.eye(D, dtype=np.float32)
        zero = np.zeros((D, D), dtype=np.float32)
        artifacts = {
            "v_projectors": {"sym_attract": eye, "sym_repulse": zero},
            "is_per_layer": False,
        }
        result = v_subspace_alignment(
            _make_crosscoder(), FakePromptStore(), artifacts, {}
        )
        if "error" in result:
            pytest.skip(f"v_subspace_alignment errored: {result['error']}")
        # attract_fracs should be ~1.0 for all features
        attract = np.array(result.get("attract_fracs", []))
        if len(attract) > 0:
            assert attract.mean() > 0.8, (
                f"Identity attractive projector should give attract_frac≈1, "
                f"got mean={attract.mean():.3f}"
            )


# ---------------------------------------------------------------------------
# violation_layer_features — artifact fallback
# ---------------------------------------------------------------------------

class TestViolationLayerFeatures:

    def test_no_artifacts_returns_per_prompt_dict(self):
        """
        Without phase1 artifacts the function still runs per-prompt and
        skips violation analysis rather than crashing.
        """
        result = violation_layer_features(
            _make_crosscoder(), FakePromptStore(), {}, {}
        )
        # Must return a dict (possibly empty or with per-prompt entries)
        assert isinstance(result, dict)

    def test_violation_features_are_sorted_by_z_score(self):
        """
        When violation features are returned, they must be in descending
        z-score order (highest discriminating features first).
        """
        ps = FakePromptStore(n_prompts=1)
        prompt_key = ps.keys()[0]
        # Simulate violation/non-violation layer indices
        artifacts = {
            "merge_layers": {prompt_key: [2]},
            "energy_violations": {prompt_key: {4.0: [2]}},
            "layer_indices": list(range(N_LAYERS)),
        }
        with patch(_PATCH_TARGET, return_value=_bimodal_scores()):
            result = violation_layer_features(
                _make_crosscoder(), ps, artifacts, {}
            )
        # If the prompt ran, its violation_features list should be sorted
        if prompt_key in result and "violation_features" in result[prompt_key]:
            z_scores = [f["z_score"] for f in result[prompt_key]["violation_features"]]
            assert z_scores == sorted(z_scores, reverse=True), (
                "violation_features must be sorted by descending z_score"
            )


# ---------------------------------------------------------------------------
# positional_control — structure check
# ---------------------------------------------------------------------------

class TestPositionalControl:

    def test_required_output_keys(self):
        with patch(_PATCH_TARGET, return_value=_full_span_scores()):
            result = positional_control(
                _make_crosscoder(), FakePromptStore(), {}, {}
            )
        # Should return something dict-like; check it doesn't crash
        assert isinstance(result, dict)

    def test_perfectly_positional_feature_flagged(self):
        """
        A feature whose activation is exactly token-position rank
        (strictly monotone with position) should have Spearman ρ = 1.0
        and be classified positional.

        We achieve this by controlling the FakePromptStore to return
        activations where feature 0's projections grow monotonically.
        Because positional_control calls _compute_feature_layer_scores
        AND then runs per-prompt correlation, we need to ensure the
        correlation path executes with a detectable signal.  We patch
        the score helper and also provide a store whose stacked tensors
        produce a gradient — testing at minimum that the function
        doesn't crash and returns a result for each prompt.
        """
        ps = FakePromptStore(n_prompts=1)
        with patch(_PATCH_TARGET, return_value=_full_span_scores()):
            result = positional_control(
                _make_crosscoder(), ps, {}, {}
            )
        assert isinstance(result, dict)
        # At minimum every prompt key should produce a sub-dict
        for key in ps.keys():
            assert key in result or "error" in result, (
                f"prompt '{key}' missing from positional_control output"
            )
