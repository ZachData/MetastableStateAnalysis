"""
tests/test_core_frame_card.py — oracle tests for core/frame_card.py.

Built against SimpleNamespace fakes, as core/ln_frame.py's tests are, so the
whole card round-trips with no torch, no transformers, no weights.

The load-bearing tests are the ones asserting that the card's frame
resolution AGREES with core.ln_frame.frame_for_hidden_state. If those two
ever diverge, the off-by-one has been duplicated and the single-home
constraint has been lost.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.frame_card import (
    CARD_JSON,
    CARD_NPZ,
    FrameCard,
    FrameCardError,
    LNStore,
    build_frame_card,
    load_frame_card,
    save_frame_card,
    verify_card_for_run,
)
from core.frames import FrameSpec, FrameMismatch, apply_frame, verify_same_revision
from core.ln_frame import frame_for_hidden_state, ln_transform

D_MODEL = 32
N_HEADS = 4
N_BLOCKS = 5
HEAD_SIZE = D_MODEL // N_HEADS          # 8
ROT_PCT = 0.25
ROT_NDIMS = int(HEAD_SIZE * ROT_PCT)    # 2


def _ln(seed):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        weight=rng.normal(size=D_MODEL),
        bias=rng.normal(size=D_MODEL),
        eps=1e-5,
    )


def _fake_model(rotary_pct=ROT_PCT, vocab=50304, parallel=True):
    layers = [
        SimpleNamespace(input_layernorm=_ln(100 + i),
                        post_attention_layernorm=_ln(200 + i))
        for i in range(N_BLOCKS)
    ]
    cfg = SimpleNamespace(
        hidden_size=D_MODEL,
        num_attention_heads=N_HEADS,
        rotary_pct=rotary_pct,
        rotary_emb_base=10000,
        layer_norm_eps=1e-5,
        vocab_size=vocab,
        use_parallel_residual=parallel,
        _name_or_path="EleutherAI/pythia-fake",
    )
    inner = SimpleNamespace(layers=layers, final_layer_norm=_ln(999))
    return SimpleNamespace(config=cfg, gpt_neox=inner)


class _FakeTokenizer:
    def __init__(self, size=50277, bos_id=None, prepend=False):
        self._size = size
        self.bos_token_id = bos_id
        self.name_or_path = "EleutherAI/pythia-fake"
        self._prepend = prepend

    def __len__(self):
        return self._size

    def __call__(self, text):
        ids = [7, 8, 9]
        if self._prepend and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}


def _card_and_store(**kw):
    return build_frame_card(_fake_model(**kw.pop("model_kw", {})),
                            model_name="pythia-1.4b",
                            revision="step143000",
                            tokenizer=_FakeTokenizer(),
                            **kw)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class TestBuild:

    def test_shape_fields(self):
        card, _ = _card_and_store()
        assert (card.d_model, card.n_heads, card.head_size, card.n_blocks) == \
               (D_MODEL, N_HEADS, HEAD_SIZE, N_BLOCKS)

    def test_rotary_pct_not_assumed_full(self):
        card, _ = _card_and_store()
        assert card.rotary_ndims == ROT_NDIMS
        assert card.rotary_ndims != card.head_size
        assert card.uses_rope

    def test_attn_scale_recorded(self):
        card, _ = _card_and_store()
        assert card.attn_scale == pytest.approx(1.0 / np.sqrt(HEAD_SIZE))

    def test_parallel_residual_recorded(self):
        card, _ = _card_and_store()
        assert card.parallel_residual is True

    def test_vocab_split_recorded(self):
        card, _ = _card_and_store()
        assert card.vocab_size_padded == 50304
        assert card.vocab_size_real == 50277
        assert card.n_padding_rows == 27

    def test_neox_bos_not_prepended(self):
        """Policy P1's precondition: position 0 is a content token, not BOS."""
        card, _ = _card_and_store()
        assert card.prepends_bos is False

    def test_bos_detected_when_present(self):
        card, _ = build_frame_card(
            _fake_model(), "m", "r",
            tokenizer=_FakeTokenizer(bos_id=0, prepend=True),
        )
        assert card.prepends_bos is True

    def test_extraction_conventions_not_guessed(self):
        card, _ = build_frame_card(_fake_model(), "m", "r",
                                   embedding_stripped=False,
                                   last_is_post_final_ln=True)
        assert card.embedding_stripped is False
        assert card.last_is_post_final_ln is True

    def test_ln_arrays_shapes(self):
        card, store = _card_and_store()
        for k in ("ln_gamma_attn", "ln_beta_attn", "ln_gamma_mlp", "ln_beta_mlp"):
            assert store.arrays[k].shape == (N_BLOCKS, D_MODEL)
        assert store.arrays["final_ln_gamma"].shape == (D_MODEL,)

    def test_attn_and_mlp_ln_are_distinct(self):
        """
        Parallel residual means both read the same input, but with DIFFERENT
        learned gamma/beta. Collapsing them is a live temptation.
        """
        _, store = _card_and_store()
        assert not np.allclose(store.arrays["ln_gamma_attn"],
                               store.arrays["ln_gamma_mlp"])

    def test_non_rope_model(self):
        cfg = SimpleNamespace(hidden_size=D_MODEL, num_attention_heads=N_HEADS,
                              layer_norm_eps=1e-5, vocab_size=50257,
                              _name_or_path="gpt2-large")
        m = SimpleNamespace(config=cfg, gpt_neox=SimpleNamespace(
            layers=[SimpleNamespace(input_layernorm=_ln(i),
                                    post_attention_layernorm=_ln(50 + i))
                    for i in range(N_BLOCKS)],
            final_layer_norm=_ln(9)))
        card, _ = build_frame_card(m, "gpt2-large", "main")
        assert card.rotary_ndims == 0 and not card.uses_rope


# ---------------------------------------------------------------------------
# Frame resolution — the single-home check
# ---------------------------------------------------------------------------

class TestFrameResolution:

    def test_agrees_with_frame_for_hidden_state(self):
        """
        The card resolves from artifacts, frame_for_hidden_state from the
        model. They must never disagree — divergence means the off-by-one
        has been duplicated.
        """
        model = _fake_model()
        card, _ = build_frame_card(model, "pythia-1.4b", "step1000")
        n_hidden = N_BLOCKS
        for L in range(n_hidden):
            for which in ("attn", "mlp"):
                res = frame_for_hidden_state(model, L, n_hidden, which=which)
                spec = card.frame_spec_for(L, n_hidden, which=which)
                if res["frame"] == "block":
                    assert spec.reader_block == res["block_idx"]
                    assert spec.kind == ("ln_attn" if which == "attn" else "ln_mlp")
                elif res["frame"] == "identity":
                    assert spec.kind == "identity"
                else:
                    assert dict(spec.extras).get("ln_source") == "final_layer_norm"

    def test_agrees_under_alternate_conventions(self):
        model = _fake_model()
        card, _ = build_frame_card(model, "m", "r", last_is_post_final_ln=True)
        n_hidden = N_BLOCKS
        for L in range(n_hidden):
            res = frame_for_hidden_state(model, L, n_hidden,
                                         last_is_post_final_ln=True)
            spec = card.frame_spec_for(L, n_hidden)
            assert (spec.kind == "identity") == (res["frame"] == "identity")

    def test_rope_applied_defaults_to_the_card(self):
        card, _ = _card_and_store()
        assert card.frame_spec_for(1, N_BLOCKS).rope_applied is True

    def test_rope_omission_must_be_deliberate(self):
        """Recording a legacy proxy is allowed, but only by saying so."""
        card, _ = _card_and_store()
        assert card.frame_spec_for(1, N_BLOCKS, rope_applied=False).rope_applied is False

    def test_revision_travels_into_the_spec(self):
        card, _ = _card_and_store()
        spec = card.frame_spec_for(1, N_BLOCKS)
        assert spec.model_rev == "pythia-1.4b@step143000"

    def test_specs_from_different_checkpoints_fail_revision_check(self):
        a, _ = build_frame_card(_fake_model(), "pythia-1.4b", "step1000")
        b, _ = build_frame_card(_fake_model(), "pythia-1.4b", "step143000")
        with pytest.raises(FrameMismatch):
            verify_same_revision(a.frame_spec_for(1, N_BLOCKS),
                                 b.frame_spec_for(1, N_BLOCKS))

    def test_deduped_is_a_different_revision(self):
        a, _ = build_frame_card(_fake_model(), "pythia-1.4b", "step1000")
        b, _ = build_frame_card(_fake_model(), "pythia-1.4b-deduped", "step1000")
        with pytest.raises(FrameMismatch):
            verify_same_revision(a.frame_spec_for(1, N_BLOCKS),
                                 b.frame_spec_for(1, N_BLOCKS))

    def test_out_of_range_index_raises(self):
        card, _ = _card_and_store()
        with pytest.raises(IndexError):
            card.frame_spec_for(99, N_BLOCKS)


# ---------------------------------------------------------------------------
# LNStore
# ---------------------------------------------------------------------------

class TestLNStore:

    def test_params_match_the_model(self):
        model = _fake_model()
        card, store = build_frame_card(model, "m", "r")
        spec = card.frame_spec_for(0, N_BLOCKS, which="attn")
        p = store.params_for(spec)
        assert np.allclose(p["gamma"],
                           model.gpt_neox.layers[spec.reader_block].input_layernorm.weight)
        assert np.allclose(p["beta"],
                           model.gpt_neox.layers[spec.reader_block].input_layernorm.bias)

    def test_mlp_params_come_from_post_attention_ln(self):
        model = _fake_model()
        card, store = build_frame_card(model, "m", "r")
        spec = card.frame_spec_for(0, N_BLOCKS, which="mlp")
        p = store.params_for(spec)
        b = spec.reader_block
        assert np.allclose(p["gamma"],
                           model.gpt_neox.layers[b].post_attention_layernorm.weight)

    def test_final_ln_routed_by_extras(self):
        model = _fake_model()
        card, store = build_frame_card(model, "m", "r")
        spec = card.frame_spec_for(N_BLOCKS - 1, N_BLOCKS)
        p = store.params_for(spec)
        assert np.allclose(p["gamma"], model.gpt_neox.final_layer_norm.weight)

    def test_non_ln_spec_returns_none(self):
        _, store = _card_and_store()
        assert store.params_for(FrameSpec.l2_sphere()) is None

    def test_end_to_end_apply_frame(self):
        """Card -> spec -> params -> apply_frame equals a direct ln_transform."""
        model = _fake_model()
        card, store = build_frame_card(model, "m", "r")
        X = np.random.default_rng(3).normal(size=(6, D_MODEL))
        spec = card.frame_spec_for(2, N_BLOCKS, which="attn")
        p = store.params_for(spec)
        got = apply_frame(X, spec, p)
        ln = model.gpt_neox.layers[spec.reader_block].input_layernorm
        assert np.allclose(got, ln_transform(X, ln.weight, ln.bias, ln.eps))

    def test_shape_mismatch_rejected(self):
        card, store = _card_and_store()
        bad = dict(store.arrays)
        bad["ln_gamma_attn"] = np.zeros((N_BLOCKS + 1, D_MODEL))
        with pytest.raises(FrameCardError):
            LNStore(bad, card=card)

    def test_missing_array_rejected(self):
        _, store = _card_and_store()
        bad = {k: v for k, v in store.arrays.items() if k != "ln_beta_mlp"}
        with pytest.raises(FrameCardError):
            LNStore(bad)

    def test_out_of_range_block_raises(self):
        card, store = _card_and_store()
        spec = FrameSpec(kind="ln_attn", layer_idx=0, reader_block=99)
        with pytest.raises(FrameCardError):
            store.params_for(spec)


# ---------------------------------------------------------------------------
# Vocabulary mask
# ---------------------------------------------------------------------------

class TestVocabMask:

    def test_masks_padding_rows(self):
        card, _ = _card_and_store()
        m = card.vocab_mask()
        assert m.shape == (50304,)
        assert m[:50277].all() and not m[50277:].any()

    def test_no_padding_is_all_true(self):
        card = FrameCard(model_name="m", vocab_size_padded=100, vocab_size_real=100)
        assert card.vocab_mask().all()

    def test_missing_vocab_raises(self):
        with pytest.raises(FrameCardError):
            FrameCard(model_name="m").vocab_mask()

    def test_real_exceeding_padded_raises(self):
        card = FrameCard(model_name="m", vocab_size_padded=10, vocab_size_real=20)
        with pytest.raises(FrameCardError):
            card.vocab_mask()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_round_trip(self, tmp_path=None):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            card, store = _card_and_store()
            save_frame_card(td, card, store)
            card2, store2 = load_frame_card(td)
            assert card2 == card
            for k in store.arrays:
                assert np.allclose(store2.arrays[k], store.arrays[k])

    def test_json_is_human_readable(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            card, store = _card_and_store()
            paths = save_frame_card(td, card, store)
            d = json.loads(open(paths["json"]).read())
            assert d["rotary_ndims"] == ROT_NDIMS
            assert d["embedding_stripped"] is True

    def test_missing_card_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FrameCardError):
                load_frame_card(td)

    def test_missing_npz_raises(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            card, store = _card_and_store()
            save_frame_card(td, card, store)
            os.remove(f"{td}/{CARD_NPZ}")
            with pytest.raises(FrameCardError):
                load_frame_card(td)

    def test_unknown_schema_field_rejected(self):
        """A card from a newer schema must not be read as if it were this one."""
        with pytest.raises(FrameCardError):
            FrameCard.from_dict({"model_name": "m", "some_future_field": 1})


class TestRunVerification:

    def test_matching_card_passes(self):
        card, _ = _card_and_store()
        verify_card_for_run(card, "pythia-1.4b", "step143000")

    def test_wrong_model_raises(self):
        card, _ = _card_and_store()
        with pytest.raises(FrameCardError):
            verify_card_for_run(card, "gpt2-large")

    def test_wrong_revision_raises(self):
        """Item 11: final-model LN parameters applied to a step-1000 checkpoint."""
        card, _ = _card_and_store()
        with pytest.raises(FrameCardError):
            verify_card_for_run(card, "pythia-1.4b", "step1000")

    def test_summary_flags_no_bos(self):
        card, _ = _card_and_store()
        text = "\n".join(card.summary_lines())
        assert "NOT prepended" in text
        assert "dead rows" in text
