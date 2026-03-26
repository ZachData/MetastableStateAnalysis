"""
tests/test_core.py — Tests for core/ (config.py, models.py).

No model downloads.  No GPU required.  All tests use synthetic data.

Run:
    pytest tests/test_core.py -v
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# core.config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_import(self):
        from core.config import (
            BETA_VALUES, DISTANCE_THRESHOLDS, K_RANGE,
            SINKHORN_MAX_ITER, SINKHORN_TOL, SPECTRAL_MAX_K,
            ALBERT_MAX_ITERATIONS, ALBERT_SNAPSHOTS,
            PROMPTS, MODEL_CONFIGS, BASE_RESULTS_DIR, DEVICE,
        )

    def test_beta_values_are_positive_floats(self):
        from core.config import BETA_VALUES
        assert len(BETA_VALUES) >= 1
        for b in BETA_VALUES:
            assert isinstance(b, (int, float))
            assert b > 0

    def test_distance_thresholds_ordered(self):
        from core.config import DISTANCE_THRESHOLDS
        import numpy as np
        arr = np.asarray(DISTANCE_THRESHOLDS)
        assert arr.ndim == 1
        assert len(arr) >= 2
        assert np.all(np.diff(arr) > 0), "thresholds must be strictly increasing"

    def test_albert_snapshots_within_max(self):
        from core.config import ALBERT_MAX_ITERATIONS, ALBERT_SNAPSHOTS
        for s in ALBERT_SNAPSHOTS:
            assert s <= ALBERT_MAX_ITERATIONS, (
                f"snapshot {s} exceeds ALBERT_MAX_ITERATIONS {ALBERT_MAX_ITERATIONS}"
            )

    def test_prompts_non_empty(self):
        from core.config import PROMPTS
        assert len(PROMPTS) >= 1
        for key, text in PROMPTS.items():
            assert isinstance(key, str) and len(key) > 0
            assert isinstance(text, str) and len(text) > 0

    def test_model_configs_required_keys(self):
        from core.config import MODEL_CONFIGS
        required = {"model_class", "tokenizer_class", "is_albert"}
        for name, cfg in MODEL_CONFIGS.items():
            missing = required - set(cfg.keys())
            assert not missing, f"{name} missing keys: {missing}"

    def test_k_range_contains_at_least_two_values(self):
        from core.config import K_RANGE
        vals = list(K_RANGE)
        assert len(vals) >= 2
        assert min(vals) >= 2, "k must be >= 2 for clustering"


# ---------------------------------------------------------------------------
# core.models  (layernorm_to_sphere only — no model loading)
# ---------------------------------------------------------------------------

class TestLayernormToSphere:
    """layernorm_to_sphere should map any tensor to unit-norm rows."""

    def _norms(self, t: torch.Tensor) -> np.ndarray:
        from core.models import layernorm_to_sphere
        out = layernorm_to_sphere(t)
        return out.norm(dim=-1).numpy()

    def test_unit_norm_random(self):
        t = torch.randn(20, 64)
        norms = self._norms(t)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_unit_norm_single_token(self):
        t = torch.randn(1, 128)
        norms = self._norms(t)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_unit_norm_large_values(self):
        t = torch.randn(8, 32) * 1000
        norms = self._norms(t)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_output_shape_preserved(self):
        from core.models import layernorm_to_sphere
        t = torch.randn(15, 256)
        out = layernorm_to_sphere(t)
        assert out.shape == t.shape

    def test_already_normalized_unchanged(self):
        from core.models import layernorm_to_sphere
        t = torch.nn.functional.normalize(torch.randn(10, 64), dim=-1)
        out = layernorm_to_sphere(t)
        np.testing.assert_allclose(out.numpy(), t.numpy(), atol=1e-5)

    def test_zero_vector_graceful(self):
        """Zero vectors should not produce NaN — F.normalize clips to zero."""
        from core.models import layernorm_to_sphere
        t = torch.zeros(5, 32)
        out = layernorm_to_sphere(t)
        assert not torch.isnan(out).any()
