"""
tests/test_qk_context.py — tests for qk_context.py.

The behaviour under test is mostly refusal: when the frame card is missing,
when the activations arrive pre-normalized, when the QK biases were never
persisted, the module must record the omission rather than produce a
plausible number. Those paths get more tests than the happy one.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.frame_card import build_frame_card
from core.frames import FrameSpec, frame_of
from core.qk_context import (
    build_qk_logit_context,
    compare_logit_sets,
    compute_head_logits,
    compute_legacy_logits,
    detect_activation_frame,
    qk_context_summary_lines,
)
from core.metrics import l2_normalize

D_MODEL = 32
N_HEADS = 4
HEAD_SIZE = D_MODEL // N_HEADS
N_BLOCKS = 4
N_TOK = 10


def _ln(seed):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(weight=rng.normal(size=D_MODEL) * 0.3 + 1.0,
                           bias=rng.normal(size=D_MODEL) * 0.1, eps=1e-5)


def _model(rotary_pct=0.25):
    layers = [SimpleNamespace(input_layernorm=_ln(10 + i),
                              post_attention_layernorm=_ln(60 + i))
              for i in range(N_BLOCKS)]
    cfg = SimpleNamespace(hidden_size=D_MODEL, num_attention_heads=N_HEADS,
                          rotary_pct=rotary_pct, rotary_emb_base=10000,
                          layer_norm_eps=1e-5, vocab_size=64,
                          use_parallel_residual=True,
                          _name_or_path="EleutherAI/pythia-fake")
    return SimpleNamespace(config=cfg, gpt_neox=SimpleNamespace(
        layers=layers, final_layer_norm=_ln(900)))


def _qk_pairs(seed=1):
    rng = np.random.default_rng(seed)
    return [(rng.normal(size=(D_MODEL, HEAD_SIZE)) * 0.1,
             rng.normal(size=(D_MODEL, HEAD_SIZE)) * 0.1)
            for _ in range(N_HEADS)]


def _biases(seed=2):
    rng = np.random.default_rng(seed)
    return [(rng.normal(size=HEAD_SIZE) * 0.1, rng.normal(size=HEAD_SIZE) * 0.1)
            for _ in range(N_HEADS)]


def _raw_X(seed=3):
    return np.random.default_rng(seed).normal(size=(N_TOK, D_MODEL)) * 2.0


def _ctx(X=None, qk=None, layer_idx=1):
    return {"token_activations": _raw_X() if X is None else X,
            "qk_matrices": _qk_pairs() if qk is None else qk,
            "layer_idx": layer_idx,
            "activations_per_layer": [None] * (N_BLOCKS + 1)}


class TestFrameDetection:

    def test_raw_detected(self):
        assert detect_activation_frame(_raw_X()) == "raw"

    def test_normalized_detected(self):
        assert detect_activation_frame(l2_normalize(_raw_X())) == "l2_sphere"

    def test_empty(self):
        assert detect_activation_frame(np.zeros((0, D_MODEL))) == "empty"


class TestComputation:

    def test_corrected_differs_from_legacy(self):
        card, store = build_frame_card(_model(), "m", "r")
        qk = _qk_pairs()
        X = _raw_X()
        corrected = compute_head_logits(X, qk, card.rotary_ndims, card.rope_base,
                                        card.attn_scale, _biases())
        legacy = compute_legacy_logits(X, qk)
        assert not np.allclose(corrected[0], legacy[0], atol=1e-4)

    def test_shapes_and_count(self):
        card, _ = build_frame_card(_model(), "m", "r")
        out = compute_head_logits(_raw_X(), _qk_pairs(), card.rotary_ndims,
                                  card.rope_base, card.attn_scale)
        assert len(out) == N_HEADS
        assert all(m.shape == (N_TOK, N_TOK) for m in out)

    def test_diff_is_causal_only(self):
        """
        The upper triangle is never softmaxed, so disagreement there must not
        register. Corrupting only the upper triangle must leave the diff
        showing perfect agreement.
        """
        rng = np.random.default_rng(0)
        a = [rng.normal(size=(6, 6))]
        b = [a[0].copy()]
        b[0][np.triu_indices(6, k=1)] += 1000.0
        d = compare_logit_sets(a, b, causal_only=True)
        assert d["worst_pearson"] == pytest.approx(1.0)
        assert d["per_head"][0]["max_abs_err"] == 0.0
        assert compare_logit_sets(a, b, causal_only=False)["worst_pearson"] < 0.99

    def test_diff_reports_per_head(self):
        card, _ = build_frame_card(_model(), "m", "r")
        qk, X = _qk_pairs(), _raw_X()
        c = compute_head_logits(X, qk, card.rotary_ndims, card.rope_base,
                                card.attn_scale)
        l = compute_legacy_logits(X, qk)
        d = compare_logit_sets(c, l)
        assert d["n_heads"] == N_HEADS
        assert d["worst_pearson"] <= d["median_pearson"] + 1e-12


class TestContextBuilder:

    def test_happy_path_applies_all_three(self):
        card, store = build_frame_card(_model(), "m", "r")
        ctx = build_qk_logit_context(_ctx(), card, store, qk_biases=_biases())
        assert ctx["qk_logit_corrections"] == {"frame": True, "rotary": True,
                                               "bias": True}
        assert ctx["qk_logit_notes"] == []
        assert len(ctx["qk_logit_matrices"]) == N_HEADS

    def test_frame_recorded_on_the_output(self):
        card, store = build_frame_card(_model(), "m", "r")
        ctx = build_qk_logit_context(_ctx(), card, store, qk_biases=_biases())
        spec = FrameSpec.from_dict(ctx["qk_logit_frame"])
        assert spec.rope_applied is True
        assert spec.is_ln()
        assert dict(spec.extras)["bias_applied"] == "True"

    def test_absent_inputs_preserve_the_old_contract(self):
        ctx = build_qk_logit_context({"qk_matrices": None,
                                      "token_activations": None})
        assert ctx["qk_logit_matrices"] is None

    def test_no_card_refuses_to_half_correct(self):
        """
        Partial correction is worse than none: it produces a number nobody can
        interpret. Without a card the module falls back wholesale and says so.
        """
        ctx = build_qk_logit_context(_ctx())
        assert ctx["qk_logit_corrections"] == {"frame": False, "rotary": False,
                                               "bias": False}
        assert "UNCORRECTED" in ctx["qk_logit_notes"][0]
        spec = FrameSpec.from_dict(ctx["qk_logit_frame"])
        assert spec.rope_applied is False
        assert dict(spec.extras)["uncorrected"] == "no_frame_card"

    def test_no_card_output_equals_legacy_exactly(self):
        c = _ctx()
        ctx = build_qk_logit_context(dict(c))
        legacy = compute_legacy_logits(np.asarray(c["token_activations"]),
                                       c["qk_matrices"])
        assert np.allclose(ctx["qk_logit_matrices"][0], legacy[0])

    def test_prenormalized_activations_block_the_frame_fix(self):
        """
        The LN frame is not recoverable from normalized activations — the
        per-token scale is gone and gamma/beta act on the unnormalized vector.
        Applying LN to a sphere would produce a confident wrong answer.
        """
        card, store = build_frame_card(_model(), "m", "r")
        ctx = build_qk_logit_context(_ctx(X=l2_normalize(_raw_X())), card, store,
                                     qk_biases=_biases())
        assert ctx["qk_logit_corrections"]["frame"] is False
        assert ctx["qk_logit_corrections"]["rotary"] is True
        assert any("L2-normalized" in n for n in ctx["qk_logit_notes"])

    def test_missing_biases_are_recorded_not_defaulted(self):
        card, store = build_frame_card(_model(), "m", "r")
        ctx = build_qk_logit_context(_ctx(), card, store, qk_biases=None)
        assert ctx["qk_logit_corrections"]["bias"] is False
        assert any("biases not supplied" in n for n in ctx["qk_logit_notes"])
        spec = FrameSpec.from_dict(ctx["qk_logit_frame"])
        assert dict(spec.extras)["bias_applied"] == "False"

    def test_non_rotary_model_marks_rotary_false(self):
        cfg = SimpleNamespace(hidden_size=D_MODEL, num_attention_heads=N_HEADS,
                              layer_norm_eps=1e-5, vocab_size=50257,
                              _name_or_path="gpt2-large")
        m = SimpleNamespace(config=cfg, gpt_neox=SimpleNamespace(
            layers=[SimpleNamespace(input_layernorm=_ln(i),
                                    post_attention_layernorm=_ln(40 + i))
                    for i in range(N_BLOCKS)],
            final_layer_norm=_ln(7)))
        card, store = build_frame_card(m, "gpt2-large", "main")
        ctx = build_qk_logit_context(_ctx(), card, store, qk_biases=_biases())
        assert ctx["qk_logit_corrections"]["rotary"] is False
        assert ctx["qk_logit_corrections"]["frame"] is True

    def test_legacy_kept_for_the_diff(self):
        card, store = build_frame_card(_model(), "m", "r")
        ctx = build_qk_logit_context(_ctx(), card, store, qk_biases=_biases())
        assert ctx["qk_logit_legacy"] is not None
        assert ctx["qk_logit_diff"]["n_heads"] == N_HEADS

    def test_legacy_can_be_suppressed(self):
        card, store = build_frame_card(_model(), "m", "r")
        ctx = build_qk_logit_context(_ctx(), card, store, qk_biases=_biases(),
                                     keep_legacy=False)
        assert ctx["qk_logit_legacy"] is None
        assert ctx["qk_logit_diff"] is None

    def test_diff_shows_the_correction_matters(self):
        """
        The S3 number. If the corrected and legacy matrices agreed, the fix
        would be cosmetic and Phase 6 would not need a re-run.
        """
        card, store = build_frame_card(_model(), "m", "r")
        ctx = build_qk_logit_context(_ctx(), card, store, qk_biases=_biases())
        assert ctx["qk_logit_diff"]["worst_pearson"] < 0.99

    def test_summary_names_each_omission(self):
        card, store = build_frame_card(_model(), "m", "r")
        ctx = build_qk_logit_context(_ctx(), card, store, qk_biases=None)
        text = "\n".join(qk_context_summary_lines(ctx))
        bias_line = [l for l in text.splitlines() if "bias" in l][0]
        rot_line = [l for l in text.splitlines() if "rotary" in l][0]
        assert "OMITTED" in bias_line
        assert "applied" in rot_line
        assert "!" in text          # the note is surfaced, not just the flag

    def test_pos0_policy_travels_into_the_ledger(self):
        card, store = build_frame_card(_model(), "m", "r")
        ctx = build_qk_logit_context(_ctx(), card, store, qk_biases=_biases(),
                                     pos0_policy="excluded")
        assert frame_of(ctx, strict=False) is None      # ctx itself isn't a record
        assert FrameSpec.from_dict(ctx["qk_logit_frame"]).pos0_policy == "excluded"
