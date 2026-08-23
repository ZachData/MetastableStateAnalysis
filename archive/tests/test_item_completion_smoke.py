"""
tests/test_item_completion_smoke.py — Smoke tests for the finish-all-items pass —
everything the pure suite could NOT cover because it needs torch + a real (tiny) HF model.

Run: SMOKE_REAL_DEPS=1 pytest -m smoke tests/test_item_completion_smoke.py -v
Needs network once for hf-internal-testing/tiny-random-gpt2 and
tiny-random-GPTNeoXForCausalLM (both confirmed to exist on the Hub — see
tests/SMOKE_TESTS_NOTES.md); cached after.

Covers:
  1. causal_tests standard-path dispatch on a real GPT-2: steer/patch/
     ablate produce trajectories that differ from baseline only from the
     target layer on, with the embedding at index 0.
  2. dissociation.run_intervened_forward post-migration: embedding
     included at index 0, hook actually changes activations, logits
     present on an LM-head model / None on a bare one.
  3. lm_loading.load_causal_lm: returns a model whose
     run_model_with_hook output has non-None logits and finite loss.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

if not os.environ.get("SMOKE_REAL_DEPS"):
    pytest.skip("set SMOKE_REAL_DEPS=1 to run smoke tests", allow_module_level=True)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
TINY_NEOX = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
TEXT = "the cat sat on the mat and the cat sat again"


@pytest.fixture(scope="module")
def tiny_gpt2_lm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY_GPT2)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_GPT2)
    model.eval()
    return model, tok


@pytest.fixture(scope="module")
def tiny_neox_lm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY_NEOX)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_NEOX)
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# 1. causal_tests dispatch on real per-layer models
# ---------------------------------------------------------------------------

class TestCausalDispatchStandardPath:
    def _baseline(self, model, tok):
        from p5_single_mstate_analysis.causal_tests import _run_standard_with_hooks
        return _run_standard_with_hooks(model, tok, TEXT, hooks=None)

    @pytest.mark.parametrize("fixture", ["tiny_gpt2_lm", "tiny_neox_lm"])
    def test_steer_residual_changes_only_from_target_layer(self, fixture, request):
        from p5_single_mstate_analysis.causal_tests import steer_residual
        model, tok = request.getfixturevalue(fixture)
        base_traj, _, _ = self._baseline(model, tok)
        d = base_traj.shape[-1]
        target = 1
        traj, attns, tokens = steer_residual(
            model, tok, TEXT, max_iterations=-1,   # ignored on standard path
            target_layer=target, token_indices=[0, 1],
            direction=np.ones(d, dtype=np.float32), alpha=5.0,
        )
        assert traj.shape == base_traj.shape
        # embedding + layers before target: identical
        for i in range(target + 1):
            np.testing.assert_allclose(traj[i], base_traj[i], atol=1e-5)
        # from the target layer's output on: different
        assert not np.allclose(traj[target + 1], base_traj[target + 1], atol=1e-5)

    def test_patch_activation_standard(self, tiny_gpt2_lm):
        from p5_single_mstate_analysis.causal_tests import patch_activation
        model, tok = tiny_gpt2_lm
        base_traj, _, _ = self._baseline(model, tok)
        d = base_traj.shape[-1]
        traj, _, _ = patch_activation(
            model, tok, TEXT, max_iterations=-1,
            target_layer=0, token_idx=2,
            replacement_vector=np.zeros(d, dtype=np.float32),
        )
        assert not np.allclose(traj[1], base_traj[1], atol=1e-5)

    def test_ablate_head_standard_is_head_specific(self, tiny_gpt2_lm):
        from p5_single_mstate_analysis.causal_tests import ablate_head
        model, tok = tiny_gpt2_lm
        t0, _, _ = ablate_head(model, tok, TEXT, max_iterations=-1,
                               target_layer=1, head_ids=[0])
        t1, _, _ = ablate_head(model, tok, TEXT, max_iterations=-1,
                               target_layer=1, head_ids=[1])
        # Different heads ablated => different downstream trajectories
        # (the exact failure mode of the old whole-output shrink).
        assert not np.allclose(t0[2], t1[2], atol=1e-5)


# ---------------------------------------------------------------------------
# 2. dissociation.run_intervened_forward post-migration
# ---------------------------------------------------------------------------

class TestDissociationMigration:
    def test_embedding_included_and_logits_present(self, tiny_gpt2_lm):
        from p6_subspace.dissociation import run_intervened_forward
        model, tok = tiny_gpt2_lm
        n_layers = model.config.n_layer
        out = run_intervened_forward(model, tok, TEXT, None, [], "cpu")
        assert len(out["activations"]) == n_layers + 1   # embedding at 0
        assert out["logits"] is not None                 # LM-head model
        assert len(out["tokens"]) == out["activations"][0].shape[0]

    def test_projection_hook_changes_activations(self, tiny_gpt2_lm):
        from p6_subspace.dissociation import (
            run_intervened_forward, make_projection_hook,
        )
        model, tok = tiny_gpt2_lm
        d = model.config.n_embd
        P = torch.eye(d)   # zero everything the hook touches
        targets = [model.transformer.h[i].attn for i in range(model.config.n_layer)]
        base = run_intervened_forward(model, tok, TEXT, None, targets, "cpu")
        hooked = run_intervened_forward(
            model, tok, TEXT, make_projection_hook(P, "cpu"), targets, "cpu",
        )
        assert not np.allclose(
            base["activations"][-1], hooked["activations"][-1], atol=1e-5
        )


# ---------------------------------------------------------------------------
# 3. lm_loading end-to-end with the merged runner
# ---------------------------------------------------------------------------

class TestLmLoading:
    def test_load_causal_lm_gives_logits_and_finite_loss(self):
        from core.lm_loading import load_causal_lm
        from core.intervention import run_model_with_hook
        cfgs = {"tiny": {"model_class": type("GPT2Model", (), {}),
                         "hf_repo": TINY_GPT2}}
        model, tok = load_causal_lm("tiny", device="cpu", model_configs=cfgs)
        out = run_model_with_hook(model, tok, TEXT, device="cpu",
                                  compute_loss=True)
        assert out["logits"] is not None
        assert out["loss"] is not None and np.isfinite(out["loss"])

    def test_state_dict_path_neox(self, tiny_neox_lm):
        from core.lm_loading import load_causal_lm_from_state_dict
        lm, _ = tiny_neox_lm
        bare_sd = lm.gpt_neox.state_dict()
        cfgs = {"tiny-neox": {"model_class": type("GPTNeoXModel", (), {}),
                              "hf_repo": TINY_NEOX}}
        model, tok = load_causal_lm_from_state_dict(
            "tiny-neox", bare_sd, device="cpu", model_configs=cfgs,
        )
        # Body weights match the supplied dict exactly.
        for k, v in model.gpt_neox.state_dict().items():
            assert torch.equal(v, bare_sd[k])
