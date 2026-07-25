"""
tests/test_attn_biases.py — oracle tests for attn_biases.py.

The layout tests carry the weight. The GPT-NeoX fused bias is PER-HEAD
contiguous and the GPT-2 fused bias is PART contiguous; applying the wrong
splitter returns arrays of the right shape with the wrong contents for every
head but the first, which is exactly how the weight-side version of this bug
shipped.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.attn_biases import (
    add_bias_arrays,
    attention_bias_drift,
    drift_share_of_displacement,
    extract_qk_biases,
    load_qk_biases,
    split_qkv_bias_gpt2,
    split_qkv_bias_gptneox,
    split_separate_bias,
)

D_MODEL = 24
N_HEADS = 4
HEAD_SIZE = D_MODEL // N_HEADS      # 6
N_LAYERS = 3


def _neox_model(with_bias=True, seed=1):
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(N_LAYERS):
        qkv = SimpleNamespace(
            weight=rng.normal(size=(N_HEADS * 3 * HEAD_SIZE, D_MODEL)),
            bias=(rng.normal(size=N_HEADS * 3 * HEAD_SIZE) if with_bias else None),
        )
        dense = SimpleNamespace(weight=rng.normal(size=(D_MODEL, D_MODEL)),
                                bias=rng.normal(size=D_MODEL))
        layers.append(SimpleNamespace(
            attention=SimpleNamespace(query_key_value=qkv, dense=dense)))
    cfg = SimpleNamespace(hidden_size=D_MODEL, num_attention_heads=N_HEADS)
    return SimpleNamespace(config=cfg, gpt_neox=SimpleNamespace(layers=layers))


def _gpt2_model(seed=2):
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(N_LAYERS):
        c_attn = SimpleNamespace(weight=rng.normal(size=(D_MODEL, 3 * D_MODEL)),
                                 bias=rng.normal(size=3 * D_MODEL))
        c_proj = SimpleNamespace(weight=rng.normal(size=(D_MODEL, D_MODEL)),
                                 bias=rng.normal(size=D_MODEL))
        blocks.append(SimpleNamespace(
            attn=SimpleNamespace(c_attn=c_attn, c_proj=c_proj)))
    cfg = SimpleNamespace(n_embd=D_MODEL, n_head=N_HEADS)
    return SimpleNamespace(config=cfg, transformer=SimpleNamespace(h=blocks))


class TestNeoxLayout:

    def test_per_head_contiguous(self):
        """index = h * 3 * head_size + part * head_size + t"""
        b = np.arange(N_HEADS * 3 * HEAD_SIZE, dtype=float)
        s = split_qkv_bias_gptneox(b, N_HEADS, HEAD_SIZE)
        assert np.allclose(s["bq"][0], np.arange(HEAD_SIZE))
        assert np.allclose(s["bk"][0], np.arange(HEAD_SIZE) + HEAD_SIZE)
        assert np.allclose(s["bv"][0], np.arange(HEAD_SIZE) + 2 * HEAD_SIZE)
        assert np.allclose(s["bq"][1], np.arange(HEAD_SIZE) + 3 * HEAD_SIZE)

    def test_shapes(self):
        s = split_qkv_bias_gptneox(np.zeros(N_HEADS * 3 * HEAD_SIZE),
                                   N_HEADS, HEAD_SIZE)
        assert len(s["bq"]) == N_HEADS
        assert all(v.shape == (HEAD_SIZE,) for v in s["bq"])

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            split_qkv_bias_gptneox(np.zeros(10), N_HEADS, HEAD_SIZE)

    def test_partition_is_complete(self):
        b = np.arange(N_HEADS * 3 * HEAD_SIZE, dtype=float)
        s = split_qkv_bias_gptneox(b, N_HEADS, HEAD_SIZE)
        recovered = np.concatenate(
            [np.concatenate([s["bq"][h], s["bk"][h], s["bv"][h]])
             for h in range(N_HEADS)])
        assert np.allclose(recovered, b)


class TestGpt2Layout:

    def test_part_contiguous(self):
        """index = part * d_model + h * head_size + t"""
        b = np.arange(3 * D_MODEL, dtype=float)
        s = split_qkv_bias_gpt2(b, N_HEADS, HEAD_SIZE)
        assert np.allclose(s["bq"][0], np.arange(HEAD_SIZE))
        assert np.allclose(s["bk"][0], np.arange(HEAD_SIZE) + D_MODEL)
        assert np.allclose(s["bq"][1], np.arange(HEAD_SIZE) + HEAD_SIZE)

    def test_layouts_disagree_beyond_the_first_head(self):
        """
        The shipped-bug shape: both splitters return (head_size,) arrays, head
        0's Q block coincides, and everything after it is wrong.
        """
        b = np.arange(N_HEADS * 3 * HEAD_SIZE, dtype=float)
        neox = split_qkv_bias_gptneox(b, N_HEADS, HEAD_SIZE)
        gpt2 = split_qkv_bias_gpt2(b, N_HEADS, HEAD_SIZE)
        assert np.allclose(neox["bq"][0], gpt2["bq"][0])
        assert not np.allclose(neox["bq"][1], gpt2["bq"][1])
        assert not np.allclose(neox["bk"][0], gpt2["bk"][0])

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            split_qkv_bias_gpt2(np.zeros(7), N_HEADS, HEAD_SIZE)


class TestSeparateBias:

    def test_slices_per_head(self):
        q = np.arange(D_MODEL, dtype=float)
        s = split_separate_bias(q, q, q, N_HEADS, HEAD_SIZE)
        assert np.allclose(s["bq"][2], np.arange(HEAD_SIZE) + 2 * HEAD_SIZE)

    def test_missing_bias_gives_zeros(self):
        s = split_separate_bias(None, None, None, N_HEADS, HEAD_SIZE)
        assert all(np.allclose(v, 0) for v in s["bq"])


class TestExtraction:

    def test_neox_shape_contract(self):
        out = extract_qk_biases(_neox_model())
        assert out["is_per_layer"] is True
        assert len(out["bq_per_head"]) == N_LAYERS
        assert len(out["bq_per_head"][0]) == N_HEADS
        assert out["bq_per_head"][0][0].shape == (HEAD_SIZE,)
        assert out["has_bias"] is True

    def test_neox_uses_the_neox_layout(self):
        m = _neox_model()
        out = extract_qk_biases(m)
        direct = split_qkv_bias_gptneox(
            m.gpt_neox.layers[1].attention.query_key_value.bias,
            N_HEADS, HEAD_SIZE)
        assert np.allclose(out["bk_per_head"][1][2], direct["bk"][2])

    def test_gpt2_uses_the_gpt2_layout(self):
        m = _gpt2_model()
        out = extract_qk_biases(m)
        direct = split_qkv_bias_gpt2(m.transformer.h[0].attn.c_attn.bias,
                                     N_HEADS, HEAD_SIZE)
        assert np.allclose(out["bq_per_head"][0][3], direct["bq"][3])

    def test_absent_bias_is_recorded_not_faked(self):
        """
        attention_bias=False is a real configuration. has_bias=False must be
        distinguishable from "we did not look".
        """
        out = extract_qk_biases(_neox_model(with_bias=False))
        assert out["has_bias"] is False
        assert np.allclose(out["bq_per_head"][0][0], 0.0)

    def test_unknown_structure_raises(self):
        m = SimpleNamespace(config=SimpleNamespace(hidden_size=D_MODEL,
                                                   num_attention_heads=N_HEADS))
        with pytest.raises(ValueError):
            extract_qk_biases(m)


class TestDrift:

    def test_matches_direct_computation(self):
        m = _neox_model()
        out = attention_bias_drift(m, 1)
        bd = extract_qk_biases(m)
        v = np.concatenate(bd["bv_per_head"][1])
        dense = m.gpt_neox.layers[1].attention.dense
        want = v @ np.asarray(dense.weight).T + np.asarray(dense.bias)
        assert np.allclose(out["drift"], want)

    def test_components_sum_to_the_drift(self):
        out = attention_bias_drift(_neox_model(), 0)
        assert np.allclose(out["from_value_bias"] + out["from_output_bias"],
                           out["drift"])

    def test_gpt2_conv1d_orientation(self):
        """
        Conv1D applies y = x @ W, nn.Linear applies y = x @ W.T. Using the
        wrong one gives a same-shaped, wrong vector when W is square.
        """
        m = _gpt2_model()
        out = attention_bias_drift(m, 0)
        bd = extract_qk_biases(m)
        v = np.concatenate(bd["bv_per_head"][0])
        cp = m.transformer.h[0].attn.c_proj
        want = v @ np.asarray(cp.weight) + np.asarray(cp.bias)
        assert np.allclose(out["drift"], want)
        assert not np.allclose(out["drift"],
                               v @ np.asarray(cp.weight).T + np.asarray(cp.bias))

    def test_zero_bias_gives_zero_value_contribution(self):
        m = _neox_model(with_bias=False)
        out = attention_bias_drift(m, 0)
        assert out["norm_from_value"] == pytest.approx(0.0)
        assert out["norm_from_output"] > 0.0

    def test_drift_share_aligned(self):
        d = np.array([1.0, 0.0, 0.0])
        s = drift_share_of_displacement(d, np.array([2.0, 0.0, 0.0]))
        assert s["cosine"] == pytest.approx(1.0)
        assert s["projected_fraction"] == pytest.approx(0.5)

    def test_drift_share_orthogonal(self):
        """
        A large drift orthogonal to the observed displacement means something
        different from a large aligned one, so both are reported.
        """
        s = drift_share_of_displacement(np.array([0.0, 5.0]), np.array([1.0, 0.0]))
        assert s["cosine"] == pytest.approx(0.0)
        assert s["projected_fraction"] == pytest.approx(0.0)
        assert s["norm_ratio"] == pytest.approx(5.0)

    def test_degenerate_inputs(self):
        s = drift_share_of_displacement(np.zeros(3), np.ones(3))
        assert np.isnan(s["cosine"])


class TestPersistence:

    def test_round_trip(self):
        m = _neox_model()
        bd = extract_qk_biases(m)
        arrays = add_bias_arrays({}, bd)
        loaded = load_qk_biases(arrays, "1", N_HEADS)
        assert loaded is not None and len(loaded) == N_HEADS
        assert np.allclose(loaded[2][0], bd["bq_per_head"][1][2], atol=1e-6)
        assert np.allclose(loaded[2][1], bd["bk_per_head"][1][2], atol=1e-6)

    def test_has_bias_flag_persisted(self):
        arrays = add_bias_arrays({}, extract_qk_biases(_neox_model(with_bias=False)))
        assert bool(arrays["_has_attn_bias"][0]) is False

    def test_missing_arrays_return_none_not_zeros(self):
        """
        Zeros are a different claim from "unknown", and only the model can say
        which is true. A loader that substitutes zeros makes the omission
        invisible.
        """
        assert load_qk_biases({}, "0", N_HEADS) is None

    def test_partial_arrays_return_none(self):
        bd = extract_qk_biases(_neox_model())
        arrays = add_bias_arrays({}, bd)
        del arrays[f"bk_head{N_HEADS - 1}_0"]
        assert load_qk_biases(arrays, "0", N_HEADS) is None

    def test_keys_pair_with_weight_convention(self):
        arrays = add_bias_arrays({}, extract_qk_biases(_neox_model()))
        assert "bq_head0_0" in arrays and "bv_head3_2" in arrays
