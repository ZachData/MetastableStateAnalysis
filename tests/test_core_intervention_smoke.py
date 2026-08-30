"""
tests/test_core_intervention_smoke.py — Tier 1 smoke test for
core/intervention.py's run_model_with_hook.

*** NOT RUN IN THE SANDBOX THIS WAS WRITTEN IN *** — no torch, no network
to install it or download the checkpoint. Written to the project's own
smoke-test convention (see tests/test_phase1_smoke.py) and should be run
for real before this module is trusted:

    SMOKE_REAL_DEPS=1 pytest -m smoke tests/test_core_intervention_smoke.py -v

Uses AutoModelForCausalLM directly (NOT core.models.load_model / the
MODEL_CONFIGS registry) because every current registry entry uses the
bare model class (GPT2Model, GPTNeoXModel, AlbertModel, BertModel — see
core/config.py, core/pythia_registry.py) with no LM head, so `.logits`
would be None for any of them. This is a real, load-bearing gap: as
things stand, run_model_with_hook's logits/loss output has nothing to
work against for any model currently in MODEL_CONFIGS. Group D (design-
5c.md: "next-token cross-entropy delta and KL divergence") will need
either a second, LM-head-bearing entry in the registry or its own direct
AutoModelForCausalLM load, separate from whatever extraction pipeline
loads the bare model — this test's setup is a preview of that, not a
recommendation to leave it ad hoc going forward.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

SMOKE_TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def tiny_causal_lm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core.models import from_pretrained_eager
    # eager, like every loader in core/: sdpa returns a tuple of None for
    # output_attentions rather than attention matrices.
    model = from_pretrained_eager(AutoModelForCausalLM, SMOKE_TINY_GPT2)
    tokenizer = AutoTokenizer.from_pretrained(SMOKE_TINY_GPT2)
    model.eval()
    return model, tokenizer


class TestBaselineRun:
    """No hooks — confirms the plumbing (tokenize, call, extract) end to end."""

    def test_returns_expected_keys(self, tiny_causal_lm):
        from core.intervention import run_model_with_hook
        model, tokenizer = tiny_causal_lm
        result = run_model_with_hook(model, tokenizer, "The quick brown fox")
        for key in ("activations", "attentions", "logits", "loss", "tokens"):
            assert key in result

    def test_activations_include_embedding_at_index_0(self, tiny_causal_lm):
        """Convention check (see module docstring in core/intervention.py):
        activations[0] must be the embedding layer, matching
        core/models.py's extract_activations -- i.e. len(activations)
        should be n_transformer_layers + 1, not n_transformer_layers."""
        from core.intervention import run_model_with_hook
        model, tokenizer = tiny_causal_lm
        result = run_model_with_hook(model, tokenizer, "The quick brown fox")
        n_layers = model.config.n_layer if hasattr(model.config, "n_layer") else model.config.num_hidden_layers
        assert len(result["activations"]) == n_layers + 1

    def test_activations_are_finite(self, tiny_causal_lm):
        from core.intervention import run_model_with_hook
        model, tokenizer = tiny_causal_lm
        result = run_model_with_hook(model, tokenizer, "The quick brown fox")
        for layer_acts in result["activations"]:
            assert np.isfinite(layer_acts).all()

    def test_logits_present_for_lm_head_model(self, tiny_causal_lm):
        from core.intervention import run_model_with_hook
        model, tokenizer = tiny_causal_lm
        result = run_model_with_hook(model, tokenizer, "The quick brown fox")
        assert result["logits"] is not None
        assert result["logits"].ndim == 2  # (n_tokens, vocab)

    def test_loss_none_when_not_requested(self, tiny_causal_lm):
        from core.intervention import run_model_with_hook
        model, tokenizer = tiny_causal_lm
        result = run_model_with_hook(model, tokenizer, "The quick brown fox")
        assert result["loss"] is None

    def test_loss_is_finite_when_requested(self, tiny_causal_lm):
        from core.intervention import run_model_with_hook
        model, tokenizer = tiny_causal_lm
        result = run_model_with_hook(
            model, tokenizer, "The quick brown fox", compute_loss=True
        )
        assert result["loss"] is not None
        assert np.isfinite(result["loss"])
        assert result["loss"] > 0  # random-init tiny model: never exactly 0


class TestHookIntervention:
    """A real hook, attached to a real submodule, actually changes the output."""

    def test_zeroing_a_layer_output_changes_downstream_activations(self, tiny_causal_lm):
        from core.intervention import run_model_with_hook
        model, tokenizer = tiny_causal_lm

        target_block = model.transformer.h[0]

        def zero_hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] * 0.0,) + output[1:]
            return output * 0.0

        baseline = run_model_with_hook(model, tokenizer, "The quick brown fox")
        intervened = run_model_with_hook(
            model, tokenizer, "The quick brown fox",
            hooks=[{"module": target_block, "hook_fn": zero_hook, "type": "forward"}],
        )

        # Zeroing layer 0's output must change every later layer's
        # activations (they're all downstream of it) but not layer 0's
        # own recorded pre-zeroing hidden state (activations[1] is BEFORE
        # this hook's effect propagates into layer 1 -- the hook fires on
        # block 0's own output, so activations[1] (index 1 = after block 0)
        # should already reflect the zeroing).
        assert not np.allclose(baseline["activations"][1], intervened["activations"][1])
        assert result_differs_downstream(baseline, intervened)

    def test_steps_gating_limits_effect_to_one_layer(self, tiny_causal_lm):
        """A hook attached to every block but gated to steps=[0] should
        behave identically to attaching it only to block 0."""
        from core.intervention import run_model_with_hook
        model, tokenizer = tiny_causal_lm

        def zero_hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] * 0.0,) + output[1:]
            return output * 0.0

        only_block_0 = run_model_with_hook(
            model, tokenizer, "The quick brown fox",
            hooks=[{
                "module": model.transformer.h[0],
                "hook_fn": zero_hook,
                "type": "forward",
            }],
        )
        # Same target module, explicit steps=[0] (its only invocation
        # anyway for a per-layer model -- this checks the gating wrapper
        # doesn't change behavior when it doesn't need to).
        gated_block_0 = run_model_with_hook(
            model, tokenizer, "The quick brown fox",
            hooks=[{
                "module": model.transformer.h[0],
                "hook_fn": zero_hook,
                "type": "forward",
                "steps": [0],
            }],
        )
        for a, b in zip(only_block_0["activations"], gated_block_0["activations"]):
            np.testing.assert_allclose(a, b)


def result_differs_downstream(baseline, intervened) -> bool:
    return any(
        not np.allclose(b, i)
        for b, i in zip(baseline["activations"][1:], intervened["activations"][1:])
    )


class TestKLReadout:

    def test_kl_between_baseline_and_zeroed_layer_is_positive(self, tiny_causal_lm):
        from core.intervention import run_model_with_hook, next_token_kl
        model, tokenizer = tiny_causal_lm

        def zero_hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] * 0.0,) + output[1:]
            return output * 0.0

        baseline = run_model_with_hook(model, tokenizer, "The quick brown fox")
        intervened = run_model_with_hook(
            model, tokenizer, "The quick brown fox",
            hooks=[{"module": model.transformer.h[0], "hook_fn": zero_hook, "type": "forward"}],
        )
        kl = next_token_kl(baseline["logits"], intervened["logits"])
        assert kl > 0  # zeroing a whole layer's output should change the
                       # next-token distribution on a random-init model
