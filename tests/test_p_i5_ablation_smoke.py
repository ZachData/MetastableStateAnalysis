"""
tests/test_p_i5_ablation_smoke.py — Tier smoke test for
p7_motifs/p_i5_ablation.py.

RUN AND VERIFIED IN THIS SANDBOX (unlike most smoke tests in this repo —
see tests/test_core_intervention_smoke.py's own caveat): pythia-70m is
cached offline here (`data/hf/`), so this uses the REAL target checkpoint
(`p_i5_ablation.MODEL_NAME`), not a tiny stand-in — a tiny random model
wouldn't have GPT-NeoX's architecture (`gpt_neox.layers`, `attention.dense`)
the hooks target, and 70m is already small enough (~7s per test on CPU)
that a stand-in buys nothing.

    SMOKE_REAL_DEPS=1 HF_HUB_OFFLINE=1 pytest -m smoke tests/test_p_i5_ablation_smoke.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def loaded_model():
    from core.lm_loading import load_causal_lm
    from p7_motifs.p_i5_ablation import MODEL_NAME
    return load_causal_lm(MODEL_NAME)


class TestDrawUnitDirection:

    def test_unit_norm(self):
        from p7_motifs.p_i5_ablation import draw_unit_direction
        rng = np.random.default_rng(0)
        v = draw_unit_direction(rng, 64)
        assert v.shape == (64,)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-10)

    def test_different_draws_differ(self):
        from p7_motifs.p_i5_ablation import draw_unit_direction
        rng = np.random.default_rng(1)
        a = draw_unit_direction(rng, 64)
        b = draw_unit_direction(rng, 64)
        assert not np.allclose(a, b)


class TestMatchedMagnitudeRandomAblation:

    def test_changes_the_target_layer_output(self, loaded_model):
        """The control hook must actually perturb the model -- a no-op
        control would make the whole gate vacuous."""
        import torch
        from p7_motifs.p_i5_ablation import (
            TARGET_HEAD, matched_magnitude_random_ablation, draw_unit_direction,
        )
        from tools.run.induction_rank_sweep import arch_dims, head_means

        model, tokenizer = loaded_model
        ids = tokenizer("the cat sat on the mat, and the cat sat again", return_tensors="pt")["input_ids"]
        means = head_means(model, ids, [TARGET_HEAD])
        _, d_head, _ = arch_dims(model)
        direction = draw_unit_direction(np.random.default_rng(2), d_head)

        with torch.no_grad():
            clean_logits = model(ids).logits
        with matched_magnitude_random_ablation(model, TARGET_HEAD, direction, means):
            with torch.no_grad():
                control_logits = model(ids).logits
        assert not torch.allclose(clean_logits, control_logits)

    def test_hook_is_removed_after_the_context_exits(self, loaded_model):
        import torch
        from p7_motifs.p_i5_ablation import (
            TARGET_HEAD, matched_magnitude_random_ablation, draw_unit_direction,
        )
        from tools.run.induction_rank_sweep import arch_dims, head_means

        model, tokenizer = loaded_model
        ids = tokenizer("a short prompt", return_tensors="pt")["input_ids"]
        means = head_means(model, ids, [TARGET_HEAD])
        _, d_head, _ = arch_dims(model)
        direction = draw_unit_direction(np.random.default_rng(3), d_head)

        with torch.no_grad():
            before = model(ids).logits.clone()
        with matched_magnitude_random_ablation(model, TARGET_HEAD, direction, means):
            pass
        with torch.no_grad():
            after = model(ids).logits
        torch.testing.assert_close(before, after)


class TestRunPrompt:

    def test_returns_expected_schema(self, loaded_model):
        from p7_motifs.p_i5_ablation import run_prompt
        model, tokenizer = loaded_model
        text = "the cat sat on the mat, and the cat sat again on the mat"
        rng = np.random.default_rng(4)
        result = run_prompt(model, tokenizer, text, rng)
        assert result is not None
        for key in ("n_tokens", "n_pairs", "delta_geometric", "delta_logit",
                    "geo_real_mean", "geo_control_mean",
                    "logit_real_mean", "logit_control_mean"):
            assert key in result
        assert result["n_pairs"] > 0
        # Both readouts are non-negative magnitudes by construction (module
        # docstring: absolute displacement / KL, never signed).
        assert result["geo_real_mean"] >= 0
        assert result["geo_control_mean"] >= 0
        assert result["logit_real_mean"] >= 0
        assert result["logit_control_mean"] >= 0

    def test_no_matched_pairs_returns_none(self, loaded_model):
        from p7_motifs.p_i5_ablation import run_prompt
        model, tokenizer = loaded_model
        rng = np.random.default_rng(5)
        # No token repeats at offset >= 2 -> induction_candidates finds nothing.
        result = run_prompt(model, tokenizer, "a b c d e f g", rng)
        assert result is None


class TestRunAll:

    def test_end_to_end_on_the_real_prompt_battery(self, loaded_model):
        """The actual gate, on the actual prompt battery -- slow (~8
        prompts x 3 forward passes), but this is exactly the pipeline
        claims/calibration/p_i5_real_ablation.json was produced by, so it
        is worth one real end-to-end run rather than only unit pieces."""
        from p7_motifs.p_i5_ablation import run_all
        result = run_all(seed=20260916)
        assert result["n_prompts"] == 8  # 9 prompts minus repeated_tokens
        assert result["gate"] is not None
        assert 0.0 < result["gate"]["p_value"] <= 1.0
        assert len(result["delta_geometric"]) == 8
        assert len(result["delta_logit"]) == 8
