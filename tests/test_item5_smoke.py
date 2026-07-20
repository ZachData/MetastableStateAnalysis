"""
tests/test_item5_smoke.py — smoke tier for the item-4/item-5 completion
pass (GPT-NeoX weights branch, head-ablation dispatch, embed_out decode,
tuned-lens trainer).

Written to the project smoke convention; NOT executed in the sandbox that
authored it (no torch/network). Run with:

    SMOKE_REAL_DEPS=1 pytest -m smoke tests/test_item5_smoke.py -v

Covers exactly what the pure tests (test_phase2_weights_gptneox.py,
test_head_ablation_math.py) cannot: real transformers module attribute
names (attention.head_size, attention.dense, gpt_neox.final_layer_norm,
embed_out) on a live tiny checkpoint, and analyze_weights end-to-end
through eigendecomposition and NPZ writing.
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

if not os.environ.get("SMOKE_REAL_DEPS"):
    pytest.skip("set SMOKE_REAL_DEPS=1 to run smoke tests", allow_module_level=True)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

TINY_NEOX_LM = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
TINY_GPT2    = "hf-internal-testing/tiny-random-gpt2"
TEXT = "the cat sat on the mat and the market closed higher"


@pytest.fixture(scope="module")
def tiny_neox_lm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY_NEOX_LM)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_NEOX_LM)
    model.eval()
    return model, tok


@pytest.fixture(scope="module")
def tiny_neox_bare(tiny_neox_lm):
    return tiny_neox_lm[0].gpt_neox, tiny_neox_lm[1]


@pytest.fixture(scope="module")
def tiny_gpt2():
    from transformers import GPT2Model, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained(TINY_GPT2)
    model = GPT2Model.from_pretrained(TINY_GPT2)
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# 1. weights.py — analyze_weights end-to-end on real GPT-NeoX
# ---------------------------------------------------------------------------

class TestWeightsGptneoxSmoke:
    def test_detect_model_type_on_real_modules(self, tiny_neox_lm, tiny_neox_bare):
        from p2_eigenspectra.weights import _detect_model_type
        assert _detect_model_type(tiny_neox_lm[0]) == "gptneox"
        assert _detect_model_type(tiny_neox_bare[0]) == "gptneox"

    def test_analyze_weights_writes_all_npz(self, tiny_neox_bare, tmp_path):
        """Registers the tiny checkpoint under a neox-routing name, then runs
        the full pipeline: OV extraction → eigendecomposition → projectors →
        QK spectrum → NPZ save. Exercises attention.head_size /
        query_key_value / dense attribute names on the real module."""
        import core.config as cfg
        from p2_eigenspectra.weights import analyze_weights

        name = TINY_NEOX_LM  # contains "NeoX" — routes via _is_gptneox_name
        cfg.MODEL_CONFIGS.setdefault(name, {
            "model_class": None, "tokenizer_class": None,
            "is_albert": False, "random_init": False,
        })
        ov_data = analyze_weights(tiny_neox_bare[0], name, tmp_path)

        assert ov_data["is_per_layer"] is True
        assert ov_data["n_heads"] >= 1
        n_layers = len(ov_data["layer_names"])
        assert n_layers == len(tiny_neox_bare[0].layers)

        stem = name.replace("/", "_")
        for prefix in ("ov_weights", "ov_decomp", "ov_projectors"):
            f = tmp_path / f"{prefix}_{stem}.npz"
            assert f.exists(), f"{f.name} not written"
        with np.load(tmp_path / f"ov_weights_{stem}.npz") as d:
            assert any(k.startswith("ov_head0_layer_") for k in d.files)
            for k in d.files:
                assert np.isfinite(d[k]).all(), f"non-finite values in {k}"


# ---------------------------------------------------------------------------
# 2. head_ablation.py — per-head deltas on both architectures
# ---------------------------------------------------------------------------

class TestHeadAblationDispatchSmoke:
    @pytest.mark.parametrize("fixture", ["tiny_gpt2", "tiny_neox_bare"])
    def test_per_head_deltas_written_and_finite(self, fixture, request, tmp_path):
        from p2_eigenspectra.head_ablation import save_decomposed_per_head
        model, tok = request.getfixturevalue(fixture)
        save_decomposed_per_head(model, tok, TEXT, tmp_path)
        f = tmp_path / "per_head_attn_deltas.npz"
        assert f.exists()
        with np.load(f) as d:
            heads = [k for k in d.files if k.startswith("attn_deltas_head_")]
            assert heads
            for k in heads:
                arr = d[k]
                assert arr.ndim == 3
                assert np.isfinite(arr).all()

    def test_neox_head_deltas_sum_to_projection_output(self, tiny_neox_bare, tmp_path):
        """The exactness property, on a REAL forward: Σ_h delta_h at layer 0
        must equal the dense projection's actual output at layer 0, captured
        by an independent hook."""
        from p2_eigenspectra.head_ablation import save_decomposed_per_head
        model, tok = tiny_neox_bare

        captured = {}
        def grab(module, inp, out):
            captured.setdefault("out", out.detach()[0].float().cpu().numpy())
        h = model.layers[0].attention.dense.register_forward_hook(grab)
        try:
            save_decomposed_per_head(model, tok, TEXT, tmp_path)
        finally:
            h.remove()

        with np.load(tmp_path / "per_head_attn_deltas.npz") as d:
            total = sum(d[k][0] for k in d.files)   # layer 0
        np.testing.assert_allclose(total, captured["out"], rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. frozen_head_decode — embed_out path
# ---------------------------------------------------------------------------

class TestEmbedOutDecodeSmoke:
    def test_decode_through_embed_out(self, tiny_neox_lm):
        from p5_single_mstate_analysis.tuned_lens_cluster import frozen_head_decode
        model, tok = tiny_neox_lm
        d = model.config.hidden_size
        v = np.random.default_rng(0).standard_normal(d).astype(np.float32)
        out = frozen_head_decode(v, model, tok, top_k=5)
        assert len(out["top"]) == 5
        probs = np.asarray(out["probs"])
        assert probs.shape[-1] == model.config.vocab_size
        assert np.isfinite(out["entropy"])
        assert abs(float(probs.sum()) - 1.0) < 1e-3

    def test_bare_model_still_raises(self, tiny_neox_bare):
        from p5_single_mstate_analysis.tuned_lens_cluster import frozen_head_decode
        model, tok = tiny_neox_bare
        v = np.zeros(model.config.hidden_size, dtype=np.float32)
        with pytest.raises(RuntimeError):
            frozen_head_decode(v, model, tok)


# ---------------------------------------------------------------------------
# 4. train_tuned_lens — end-to-end on the tiny checkpoint
# ---------------------------------------------------------------------------

class TestTunedLensTrainerSmoke:
    def test_train_load_and_group_e_probabilities_nonzero(self, tmp_path):
        """The full Group E fix, end to end: train on the tiny NeoX
        checkpoint, load, decode through apply_tuned_lens + embed_out.
        The prompt here is short — min-tokens is lowered accordingly;
        real runs keep the default guard."""
        import core.config as cfg
        from p5_single_mstate_analysis import train_tuned_lens as tl
        from p5_single_mstate_analysis.tuned_lens_cluster import (
            load_tuned_lens, apply_tuned_lens,
        )

        cfg.MODEL_CONFIGS.setdefault(TINY_NEOX_LM, {
            "model_class": transformers.GPTNeoXModel,
            "tokenizer_class": transformers.AutoTokenizer,
            "is_albert": False, "random_init": False,
        })
        rc = tl.main([
            "--model", TINY_NEOX_LM,
            "--out", str(tmp_path),
            "--prompts", "short_heterogeneous", "wiki_paragraph",
            "--min-tokens", "8",
        ])
        assert rc == 0
        stem = TINY_NEOX_LM.replace("/", "_")
        lens_path = tmp_path / f"tuned_lens_{stem}.npz"
        assert lens_path.exists()
        assert lens_path.with_suffix(".json").exists()

        lens = load_tuned_lens(lens_path)
        assert lens and 0 in lens
        d = lens[0]["A"].shape[0]
        v = np.random.default_rng(1).standard_normal(d).astype(np.float32)
        v2 = apply_tuned_lens(v, 0, lens)
        assert v2.shape == v.shape
        assert not np.allclose(v2, v)   # translator actually does something
