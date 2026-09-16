"""
tests/test_p_i5_structured_control_smoke.py — Tier smoke test for
p7_motifs/p_i5_structured_control.py.

Real, committed evidence for the actual findings (both constructions here
also fail to discriminate; the constant-substitution diagnostic explains
why) is claims/calibration/p_i5_structured_control.json. This file checks
plumbing, kept lean.

    SMOKE_REAL_DEPS=1 HF_HUB_OFFLINE=1 pytest -m smoke tests/test_p_i5_structured_control_smoke.py -v
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


class TestDrawOtherHeadDirection:

    def test_never_returns_the_target(self, loaded_model):
        from p7_motifs.p_i5_structured_control import all_heads, draw_other_head_direction
        import torch
        model, tokenizer = loaded_model
        target = (3, 6)
        ids = tokenizer("a short prompt for head means", return_tensors="pt")["input_ids"]
        from tools.run.induction_rank_sweep import head_means
        pool = all_heads(model)
        means = head_means(model, ids, pool)

        rng = np.random.default_rng(0)
        for _ in range(20):
            direction, donor = draw_other_head_direction(rng, target, pool, means)
            assert donor != target
            np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-6)

    def test_all_heads_has_expected_count(self, loaded_model):
        from p7_motifs.p_i5_structured_control import all_heads
        model, _ = loaded_model
        heads = all_heads(model)
        assert len(heads) == model.config.num_hidden_layers * model.config.num_attention_heads
        assert len(set(heads)) == len(heads)  # no duplicates


class TestRunPrompt:

    def test_returns_expected_schema(self, loaded_model):
        from p7_motifs.p_i5_structured_control import run_prompt
        model, tokenizer = loaded_model
        text = "the cat sat on the mat, and the cat sat again on the mat"
        rng = np.random.default_rng(1)
        result = run_prompt(model, tokenizer, text, rng, (3, 6))
        assert result is not None
        for key in ("n_pairs", "donor_head", "delta_geometric", "delta_logit"):
            assert key in result
        assert result["donor_head"] != [3, 6]


class TestConstantSubstitutionDiagnostic:

    def test_geometric_delta_is_near_zero_by_construction(self, loaded_model):
        """The documented finding: full constant substitution makes the
        ablated slice's contribution to a pairwise distance exactly zero
        regardless of which constant is used, so delta_geometric should
        be numerically ~0 -- checked directly, not just asserted in prose."""
        from p7_motifs.p_i5_structured_control import run_constant_substitution_diagnostic
        result = run_constant_substitution_diagnostic((3, 6), seed=1)
        assert result["geometric_is_near_zero"] is True
        assert max(abs(v) for v in result["delta_geometric"]) < 1e-6
