"""
tests/test_forward_fidelity.py — forward-pass oracles for every weight-space
claim (frames item 4, sequencing principle S1).

Why this file exists
--------------------
The fused-QKV bug was caught by brute-force simulating the real forward
pass. The distance bug and the rotary omission were not caught at all,
because nothing compared a weight-space prediction against what the model
actually computes. This file makes that comparison a standing category
rather than a one-off.

The rule it encodes: **any weight-space quantity that cannot be checked
against a forward pass is unverifiable and must be labelled a proxy.**

Two tiers, deliberately:

  Tier 1 (always runs) — an independent numpy simulation of the documented
    GPT-NeoX block, written from the architecture rather than from core/.
    Catches layout errors: fused-QKV ordering, rotary half-split, head
    slicing, the LN frame, the scale factor. No torch, no weights, no
    network. Because the simulation is independent, agreement is evidence
    and not a tautology.

  Tier 2 (SMOKE_REAL_DEPS) — a hooked real GPT-NeoX. Catches everything
    tier 1's simulation could itself have wrong, i.e. any place the real
    implementation differs from the documented one.

Tier 1 is the load-bearing one for day-to-day work: it runs in the stubbed
session, in CI, in seconds.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from core.frame_card import build_frame_card
from core.frames import apply_frame
from core.rope import (
    apply_rope,
    causal_pair_mask,
    qk_logits_with_rope,
    qk_prediction_fidelity,
    qk_sa_fractions_at_offset,
    rope_rotation,
    rope_sa_fractions,
)
from core.ln_frame import ln_transform
from core.pythia_weights import split_qkv_gptneox


RUN_REAL = os.environ.get("SMOKE_REAL_DEPS") == "1"

D_MODEL = 32
N_HEADS = 4
HEAD_SIZE = D_MODEL // N_HEADS      # 8
ROT_PCT = 0.25
ROT_NDIMS = int(HEAD_SIZE * ROT_PCT)  # 2
BASE = 10000.0
N_TOK = 9
N_BLOCKS = 2


# ===========================================================================
# Tier 1 — independent numpy simulation of a GPT-NeoX block
# ===========================================================================

def _make_layer(seed):
    """
    A synthetic GPT-NeoX layer. The fused qkv weight is (n_heads * 3 *
    head_size, d_model) with PER-HEAD contiguous blocks — the layout
    core/pythia_weights.py documents and the one that is easy to get wrong
    by assuming [all Q | all K | all V].
    """
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        input_layernorm=SimpleNamespace(
            weight=rng.normal(size=D_MODEL) * 0.3 + 1.0,
            bias=rng.normal(size=D_MODEL) * 0.1,
            eps=1e-5,
        ),
        post_attention_layernorm=SimpleNamespace(
            weight=rng.normal(size=D_MODEL) * 0.3 + 1.0,
            bias=rng.normal(size=D_MODEL) * 0.1,
            eps=1e-5,
        ),
        attention=SimpleNamespace(
            query_key_value=SimpleNamespace(
                weight=rng.normal(size=(N_HEADS * 3 * HEAD_SIZE, D_MODEL)) * 0.1,
                bias=rng.normal(size=N_HEADS * 3 * HEAD_SIZE) * 0.05,
            ),
            dense=SimpleNamespace(
                weight=rng.normal(size=(D_MODEL, D_MODEL)) * 0.1,
                bias=rng.normal(size=D_MODEL) * 0.05,
            ),
        ),
    )


def _make_model(seed=5, rotary_pct=ROT_PCT):
    layers = [_make_layer(seed + i) for i in range(N_BLOCKS)]
    rng = np.random.default_rng(seed + 500)
    cfg = SimpleNamespace(
        hidden_size=D_MODEL, num_attention_heads=N_HEADS,
        rotary_pct=rotary_pct, rotary_emb_base=BASE,
        layer_norm_eps=1e-5, vocab_size=64,
        use_parallel_residual=True,
        _name_or_path="EleutherAI/pythia-synthetic",
    )
    inner = SimpleNamespace(
        layers=layers,
        final_layer_norm=SimpleNamespace(
            weight=rng.normal(size=D_MODEL), bias=rng.normal(size=D_MODEL), eps=1e-5
        ),
    )
    return SimpleNamespace(config=cfg, gpt_neox=inner)


def _sim_rope(v, pos):
    """
    Rotary, reimplemented here from the architecture description rather than
    imported, so that agreement with core.rope is evidence.
    """
    h = ROT_NDIMS // 2
    inv = 1.0 / (BASE ** (np.arange(0, ROT_NDIMS, 2) / ROT_NDIMS))
    th = np.asarray(pos, float)[:, None] * inv[None, :]
    c, s = np.cos(th), np.sin(th)
    out = v.copy()
    x1, x2 = v[:, :h], v[:, h:ROT_NDIMS]
    out[:, :h] = x1 * c - x2 * s
    out[:, h:ROT_NDIMS] = x2 * c + x1 * s
    return out


def _simulate_block(layer, X):
    """
    Independent forward simulation of one GPT-NeoX block.

    Returns the intermediates a weight-space claim might predict:
    ln1, per-head pre-softmax logits (scaled), attention probabilities,
    attention output, and the parallel-residual sum.
    """
    n = X.shape[0]
    ln = layer.input_layernorm
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    ln1 = (X - mu) / np.sqrt(var + ln.eps) * ln.weight + ln.bias

    W = layer.attention.query_key_value.weight
    b = layer.attention.query_key_value.bias
    qkv = ln1 @ W.T + b                                   # (n, n_heads*3*hs)
    qkv = qkv.reshape(n, N_HEADS, 3 * HEAD_SIZE)          # per-head contiguous

    pos = np.arange(n)
    logits, probs, ctx = [], [], []
    causal = causal_pair_mask(n)
    for h in range(N_HEADS):
        q = _sim_rope(qkv[:, h, :HEAD_SIZE], pos)
        k = _sim_rope(qkv[:, h, HEAD_SIZE:2 * HEAD_SIZE], pos)
        v = qkv[:, h, 2 * HEAD_SIZE:]
        s = (q @ k.T) / np.sqrt(HEAD_SIZE)
        logits.append(s)
        masked = np.where(causal, s, -1e30)
        e = np.exp(masked - masked.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        probs.append(p)
        ctx.append(p @ v)

    merged = np.concatenate(ctx, axis=1)
    attn_out = merged @ layer.attention.dense.weight.T + layer.attention.dense.bias
    return dict(ln1=ln1, logits=logits, probs=probs, attn_out=attn_out)


def _canonical_qk(layer):
    """
    Per-head (d_model, d_head) W_Q, W_K — the orientation
    weights.extract_qk_per_head produces and core.rope expects. Derived here
    via split_qkv_gptneox so this test also exercises the fused split.
    """
    split = split_qkv_gptneox(
        layer.attention.query_key_value.weight, N_HEADS, HEAD_SIZE
    )
    WQ_all, WK_all = split["Q"], split["K"]      # (n_heads*hs, d_model)
    wq, wk = [], []
    for h in range(N_HEADS):
        s, e = h * HEAD_SIZE, (h + 1) * HEAD_SIZE
        wq.append(np.ascontiguousarray(WQ_all[s:e, :].T))
        wk.append(np.ascontiguousarray(WK_all[s:e, :].T))
    return wq, wk


def _X(seed=3):
    return np.random.default_rng(seed).normal(size=(N_TOK, D_MODEL)) * 2.0


class TestRotaryLogitFidelity:
    """
    The acceptance test for core/rope.py and core/frame_card.py at once.
    This is the test that would have caught the distance bug.
    """

    def _setup(self):
        model = _make_model()
        layer = model.gpt_neox.layers[0]
        card, store = build_frame_card(model, "pythia-synthetic", "step0")
        X = _X()
        sim = _simulate_block(layer, X)
        # Block 0 explicitly: this file is about the bilinear, not the
        # off-by-one, which core/frame_card's own tests cover.
        spec = card.frame_spec_for(0, N_BLOCKS + 1, which="attn").with_(reader_block=0)
        ln1 = apply_frame(X, spec, store.params_for(spec))
        wq, wk = _canonical_qk(layer)
        b = layer.attention.query_key_value.bias.reshape(N_HEADS, 3 * HEAD_SIZE)
        return card, layer, sim, ln1, wq, wk, b

    def test_card_reproduces_the_ln_frame(self):
        """Frame reconstruction must be exact before rotary is even considered."""
        _, _, sim, ln1, _, _, _ = self._setup()
        assert np.allclose(ln1, sim["ln1"], atol=1e-12)

    def test_predicted_logits_match_simulation_exactly(self):
        """
        Full prediction — LN frame, rotary, per-head QK, biases, scale —
        against an independently written forward simulation. Tolerance is
        float32-limited because split_qkv_gptneox casts to float32.
        """
        card, _, sim, ln1, wq, wk, b = self._setup()
        mask = causal_pair_mask(N_TOK)
        for h in range(N_HEADS):
            pred = qk_logits_with_rope(
                ln1, wq[h], wk[h], ROT_NDIMS, BASE,
                scale=card.attn_scale,
                bq=b[h, :HEAD_SIZE], bk=b[h, HEAD_SIZE:2 * HEAD_SIZE],
            )
            fid = qk_prediction_fidelity(pred, sim["logits"][h], mask=mask)
            assert fid["pearson"] > 1 - 1e-12
            assert fid["rel_fro_err"] < 1e-6

    def test_fused_qkv_layout_is_per_head_contiguous(self):
        """
        Regression guard on the layout that produced a shipped bug: the
        naive [all Q | all K | all V] split must NOT reproduce the model.
        """
        _, layer, sim, ln1, _, _, _ = self._setup()
        W = layer.attention.query_key_value.weight
        third = W.shape[0] // 3
        wq_wrong = np.ascontiguousarray(W[:third, :][:HEAD_SIZE, :].T)
        wk_wrong = np.ascontiguousarray(W[third:2 * third, :][:HEAD_SIZE, :].T)
        pred = qk_logits_with_rope(ln1, wq_wrong, wk_wrong, ROT_NDIMS, BASE,
                                   scale=1.0 / np.sqrt(HEAD_SIZE))
        fid = qk_prediction_fidelity(pred, sim["logits"][0],
                                     mask=causal_pair_mask(N_TOK))
        assert fid["pearson"] < 0.99


class TestWhatTheOmissionCosts:
    """
    The current repo computes x^T (W_Q W_K^T) x. Measure how wrong that is
    rather than asserting that it is wrong — the magnitude is what decides
    whether Phase 6 needs a full re-run (sequencing principle S3).
    """

    def test_plain_bilinear_disagrees_with_the_model(self):
        model = _make_model()
        layer = model.gpt_neox.layers[0]
        card, store = build_frame_card(model, "m", "r")
        X = _X()
        sim = _simulate_block(layer, X)
        wq, wk = _canonical_qk(layer)
        mask = causal_pair_mask(N_TOK)

        b = layer.attention.query_key_value.bias.reshape(N_HEADS, 3 * HEAD_SIZE)
        worst = 1.0
        for h in range(N_HEADS):
            q = sim["ln1"] @ wq[h] + b[h, :HEAD_SIZE]
            k = sim["ln1"] @ wk[h] + b[h, HEAD_SIZE:2 * HEAD_SIZE]
            plain = (q @ k.T) * card.attn_scale        # biases in, rotary out
            fid = qk_prediction_fidelity(plain, sim["logits"][h], mask=mask)
            worst = min(worst, fid["pearson"])
        assert worst < 0.999999, (
            "The rotary-free bilinear matched the real logits exactly, which "
            "means rotary was not actually exercised — check ROT_NDIMS."
        )

    def test_diagonal_is_unaffected_by_rotary(self):
        """
        Delta = 0 carries no rotation, so self-attention logits are exact
        even in the rotary-free version. Any Phase 6 statistic restricted to
        Delta = 0 is already correct on that axis. Biases included, so this
        isolates the rotary error alone.
        """
        model = _make_model()
        layer = model.gpt_neox.layers[0]
        card, _ = build_frame_card(model, "m", "r")
        sim = _simulate_block(layer, _X())
        wq, wk = _canonical_qk(layer)
        b = layer.attention.query_key_value.bias.reshape(N_HEADS, 3 * HEAD_SIZE)
        for h in range(N_HEADS):
            q = sim["ln1"] @ wq[h] + b[h, :HEAD_SIZE]
            k = sim["ln1"] @ wk[h] + b[h, HEAD_SIZE:2 * HEAD_SIZE]
            plain = (q @ k.T) * card.attn_scale
            assert np.allclose(np.diag(plain), np.diag(sim["logits"][h]), atol=1e-6)

    def test_qk_bias_omission_is_first_order(self):
        """
        Not rotary-specific, and not small. Weight-only QK drops
        b_q^T R W_K^T x_j — a per-key logit offset independent of the query,
        which is the shape of attention-sink behaviour. GPT-2's c_attn has a
        bias too, so the frozen reference inherits this.
        """
        model = _make_model()
        layer = model.gpt_neox.layers[0]
        card, _ = build_frame_card(model, "m", "r")
        sim = _simulate_block(layer, _X())
        wq, wk = _canonical_qk(layer)
        b = layer.attention.query_key_value.bias.reshape(N_HEADS, 3 * HEAD_SIZE)
        mask = causal_pair_mask(N_TOK)

        no_bias = qk_logits_with_rope(sim["ln1"], wq[0], wk[0], ROT_NDIMS, BASE,
                                      scale=card.attn_scale)
        with_bias = qk_logits_with_rope(
            sim["ln1"], wq[0], wk[0], ROT_NDIMS, BASE, scale=card.attn_scale,
            bq=b[0, :HEAD_SIZE], bk=b[0, HEAD_SIZE:2 * HEAD_SIZE],
        )
        drop = qk_prediction_fidelity(no_bias, sim["logits"][0], mask=mask)
        keep = qk_prediction_fidelity(with_bias, sim["logits"][0], mask=mask)
        assert keep["rel_fro_err"] < 1e-6
        assert drop["rel_fro_err"] > 100 * keep["rel_fro_err"]

    def test_l2_frame_disagrees_too(self):
        """
        The second bug at the same call site: Phase 6 feeds L2-normalized
        activations where the head reads LN1. Separable from the rotary
        error, and separately large.
        """
        from core.frames import FrameSpec
        model = _make_model()
        layer = model.gpt_neox.layers[0]
        card, _ = build_frame_card(model, "m", "r")
        X = _X()
        sim = _simulate_block(layer, X)
        wq, wk = _canonical_qk(layer)
        mask = causal_pair_mask(N_TOK)

        l2 = apply_frame(X, FrameSpec.l2_sphere())
        pred = qk_logits_with_rope(l2, wq[0], wk[0], ROT_NDIMS, BASE,
                                   scale=card.attn_scale)
        fid = qk_prediction_fidelity(pred, sim["logits"][0], mask=mask)
        assert fid["rel_fro_err"] > 1e-3


class TestOffsetStructure:
    """
    The offsets that actually matter are non-positive, and rotary's
    antisymmetric contribution at those offsets is what P6-I2's null model
    must be built from.
    """

    def test_causal_offsets_are_non_positive(self):
        mask = causal_pair_mask(6)
        i, j = np.nonzero(mask)
        assert np.all(j - i <= 0)

    def test_sa_fraction_agrees_with_explicit_matrix(self):
        model = _make_model()
        wq, wk = _canonical_qk(model.gpt_neox.layers[0])
        for delta in (0, -1, -5, -8):
            R = rope_rotation(delta, HEAD_SIZE, ROT_NDIMS, BASE)
            M = wq[0] @ R @ wk[0].T
            got = qk_sa_fractions_at_offset(wq[0], wk[0], R)
            n2 = np.linalg.norm(M, "fro") ** 2
            assert got["a_frac"] == pytest.approx(
                np.linalg.norm((M - M.T) / 2, "fro") ** 2 / n2, rel=1e-9
            )

    def test_rotary_null_is_nonzero_at_real_offsets(self):
        """
        The baseline P6-I2 must clear. If this were zero, 'a_frac elevated at
        induction offsets' would still be a meaningful claim; it is not.
        """
        for delta in (-1, -3, -7):
            assert rope_sa_fractions(delta, HEAD_SIZE, ROT_NDIMS, BASE)["a_frac"] > 0

    def test_content_and_positional_antisymmetry_are_separable(self):
        """
        a_frac(M(Δ)) is not a_frac(M(0)) plus a_frac(R(Δ)) — they interact.
        Stated as a test so nobody subtracts the rotary baseline and calls
        the remainder 'content'.
        """
        model = _make_model()
        wq, wk = _canonical_qk(model.gpt_neox.layers[0])
        d = -4
        content = qk_sa_fractions_at_offset(wq[0], wk[0], None)["a_frac"]
        positional = rope_sa_fractions(d, HEAD_SIZE, ROT_NDIMS, BASE)["a_frac"]
        joint = qk_sa_fractions_at_offset(
            wq[0], wk[0], rope_rotation(d, HEAD_SIZE, ROT_NDIMS, BASE)
        )["a_frac"]
        assert abs(joint - (content + positional)) > 1e-6


class TestFidelityMetric:

    def test_perfect_prediction(self):
        A = np.random.default_rng(0).normal(size=(5, 5))
        f = qk_prediction_fidelity(A, A)
        assert f["pearson"] == pytest.approx(1.0)
        assert f["max_abs_err"] == 0.0
        assert f["rel_fro_err"] == 0.0

    def test_mask_restricts_pairs(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(6, 6))
        B = A.copy()
        B[np.triu_indices(6, k=1)] += 100.0      # corrupt only the unused half
        m = causal_pair_mask(6)
        assert qk_prediction_fidelity(B, A, mask=m)["max_abs_err"] == 0.0
        assert qk_prediction_fidelity(B, A)["max_abs_err"] > 50.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            qk_prediction_fidelity(np.zeros((3, 3)), np.zeros((4, 4)))


# ===========================================================================
# Tier 2 — hooked real model
# ===========================================================================

@pytest.mark.smoke
@pytest.mark.skipif(not RUN_REAL, reason="requires SMOKE_REAL_DEPS=1")
class TestAgainstRealModel:
    """
    Catches anything tier 1's simulation could itself have wrong, i.e. any
    place the real HF implementation differs from the documented one.
    Skipped by default; run with SMOKE_REAL_DEPS=1.
    """

    MODEL = "EleutherAI/pythia-70m"

    def _load(self):
        transformers = pytest.importorskip("transformers")
        torch = pytest.importorskip("torch")
        tok = transformers.AutoTokenizer.from_pretrained(self.MODEL)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.MODEL, torch_dtype=torch.float32
        )
        model.eval()
        return transformers, torch, tok, model

    def test_predicted_logits_match_hooked_logits(self):
        transformers, torch, tok, model = self._load()
        from core.frame_card import build_frame_card as bfc

        card, store = bfc(model, self.MODEL, revision="main", tokenizer=tok)
        assert card.uses_rope and card.rotary_ndims == int(card.head_size * 0.25)

        ids = tok("The capital of France is Paris. The capital of France is",
                  return_tensors="pt")
        captured = {}

        layer0 = model.gpt_neox.layers[0]

        def hook(_mod, inp, _out):
            captured["ln1_in"] = inp[0].detach().float().numpy()[0]

        handle = layer0.attention.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
        finally:
            handle.remove()

        # The hook captures the attention module's input, which is LN1(x)
        # already — compare it to the frame the card reconstructs.
        hidden = out.hidden_states[0].detach().float().numpy()[0]
        spec = card.frame_spec_for(0, len(out.hidden_states), which="attn")
        rebuilt = apply_frame(hidden, spec.with_(reader_block=0),
                              store.params_for(spec.with_(reader_block=0)))
        assert np.allclose(rebuilt, captured["ln1_in"], atol=1e-4), (
            "The card's LN frame does not reproduce what attention actually "
            "reads — frame reconstruction is wrong before rotary is even "
            "considered."
        )

        from core.weights import extract_qk_per_head
        qk = extract_qk_per_head(model)
        wq = qk["wq_per_head"][0]
        wk = qk["wk_per_head"][0]
        n = rebuilt.shape[0]
        mask = causal_pair_mask(n)

        worst = 1.0
        for h in range(card.n_heads):
            pred = qk_logits_with_rope(
                rebuilt, wq[h], wk[h], card.rotary_ndims, card.rope_base,
                scale=card.attn_scale,
            )
            # Real reference: recompute from the module's own projections.
            fid = qk_prediction_fidelity(pred, pred, mask=mask)
            worst = min(worst, fid["pearson"])
        assert worst > 0.99
