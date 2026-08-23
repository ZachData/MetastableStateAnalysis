"""
tests/test_core_intervention_pure.py — the torch-free pieces of
core/intervention.py: `_step_gated_hook` (pure Python control flow) and
`next_token_kl` / `next_token_kl_all_positions` / `_logsumexp` (pure
numpy). These ARE runtime-verified (no torch needed for any of them —
`import torch` in core/intervention.py is deferred inside
run_model_with_hook itself, which this file does not exercise).

run_model_with_hook itself needs a real HuggingFace model and is NOT
covered here — see tests/test_core_intervention_smoke.py (unverified,
torch+transformers required).
"""
from __future__ import annotations

import numpy as np
import numpy.testing as npt

from core.intervention import (
    _step_gated_hook,
    _logsumexp,
    next_token_kl,
    next_token_kl_all_positions,
)

import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure

# ---------------------------------------------------------------------------
# _step_gated_hook
# ---------------------------------------------------------------------------

class TestStepGatedHookForwardPre:
    """is_pre=True: hook_fn(module, inputs) -> modified inputs or None."""

    def test_steps_none_fires_every_call(self):
        calls = []
        def hook_fn(module, inputs):
            calls.append(inputs)
            return inputs
        gated = _step_gated_hook(hook_fn, None, is_pre=True)
        for i in range(4):
            gated("mod", (i,))
        assert calls == [(0,), (1,), (2,), (3,)]

    def test_steps_list_fires_only_at_those_indices(self):
        calls = []
        def hook_fn(module, inputs):
            calls.append(inputs)
            return inputs
        gated = _step_gated_hook(hook_fn, [1, 3], is_pre=True)
        for i in range(5):
            gated("mod", (i,))
        assert calls == [(1,), (3,)]

    def test_non_gated_call_returns_none(self):
        gated = _step_gated_hook(lambda m, i: ("modified",), [0], is_pre=True)
        result_at_0 = gated("mod", ("original",))
        result_at_1 = gated("mod", ("original",))
        assert result_at_0 == ("modified",)
        assert result_at_1 is None  # not gated at step 1 -> no modification

    def test_counter_is_independent_per_wrapped_hook(self):
        """Two separately-wrapped hooks (e.g. two different modules) must
        not share a counter."""
        gated_a = _step_gated_hook(lambda m, i: "A", [0], is_pre=True)
        gated_b = _step_gated_hook(lambda m, i: "B", [0], is_pre=True)
        assert gated_a("mod", ()) == "A"
        assert gated_b("mod", ()) == "B"  # would be None if counters were shared

    def test_empty_steps_list_never_fires(self):
        gated = _step_gated_hook(lambda m, i: "modified", [], is_pre=True)
        assert gated("mod", ("x",)) is None


class TestStepGatedHookForward:
    """is_pre=False: hook_fn(module, inputs, output) -> modified output or None."""

    def test_steps_none_fires_every_call(self):
        calls = []
        def hook_fn(module, inputs, output):
            calls.append(output)
            return output
        gated = _step_gated_hook(hook_fn, None, is_pre=False)
        for i in range(3):
            gated("mod", (), i)
        assert calls == [0, 1, 2]

    def test_steps_list_fires_only_at_those_indices(self):
        calls = []
        def hook_fn(module, inputs, output):
            calls.append(output)
            return output
        gated = _step_gated_hook(hook_fn, [2], is_pre=False)
        for i in range(4):
            gated("mod", (), i)
        assert calls == [2]


# ---------------------------------------------------------------------------
# next_token_kl / next_token_kl_all_positions / _logsumexp
# ---------------------------------------------------------------------------

class TestLogSumExp:

    def test_matches_naive_computation_when_stable(self):
        x = np.array([1.0, 2.0, 3.0])
        expected = np.log(np.sum(np.exp(x)))
        npt.assert_allclose(_logsumexp(x), expected)

    def test_numerically_stable_for_large_values(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        # naive np.log(np.sum(np.exp(x))) would overflow to inf
        result = _logsumexp(x)
        assert np.isfinite(result)
        # log-sum-exp shift identity: logsumexp(x) - max(x) is small and finite
        assert result - np.max(x) < 5


class TestNextTokenKL:

    def test_identical_logits_give_zero_kl(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((5, 20))
        kl = next_token_kl(logits, logits.copy())
        npt.assert_allclose(kl, 0.0, atol=1e-10)

    def test_default_position_is_last(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((5, 20))
        b = rng.standard_normal((5, 20))
        kl_default = next_token_kl(a, b)
        kl_explicit = next_token_kl(a, b, position=-1)
        assert kl_default == kl_explicit

    def test_kl_is_nonnegative(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            a = rng.standard_normal((3, 15))
            b = rng.standard_normal((3, 15))
            assert next_token_kl(a, b, position=0) >= -1e-10

    def test_kl_is_asymmetric_in_general(self):
        rng = np.random.default_rng(2)
        a = rng.standard_normal((3, 15))
        b = rng.standard_normal((3, 15))
        kl_ab = next_token_kl(a, b, position=0)
        kl_ba = next_token_kl(b, a, position=0)
        assert not np.isclose(kl_ab, kl_ba)

    def test_matches_manual_kl_computation(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal(10)
        b = rng.standard_normal(10)
        logits_a = a[None, :]
        logits_b = b[None, :]

        # manual softmax + KL, independent implementation path
        p = np.exp(a - np.max(a)); p /= p.sum()
        q = np.exp(b - np.max(b)); q /= q.sum()
        expected = float(np.sum(p * np.log(p / q)))

        result = next_token_kl(logits_a, logits_b, position=0)
        npt.assert_allclose(result, expected, rtol=1e-6)


class TestNextTokenKLAllPositions:

    def test_shape_matches_shorter_sequence(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((7, 20))
        b = rng.standard_normal((5, 20))
        result = next_token_kl_all_positions(a, b)
        assert result.shape == (5,)

    def test_matches_per_position_next_token_kl(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((4, 10))
        b = rng.standard_normal((4, 10))
        result = next_token_kl_all_positions(a, b)
        expected = np.array([next_token_kl(a, b, position=i) for i in range(4)])
        npt.assert_allclose(result, expected)

    def test_identical_sequences_all_zero(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((6, 12))
        result = next_token_kl_all_positions(a, a.copy())
        npt.assert_allclose(result, np.zeros(6), atol=1e-10)
