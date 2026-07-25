"""
tests/test_pythia_qkv_split.py — Tier 2 oracle test for
core/pythia_weights.split_qkv_gptneox.

Construction, not real weights: a synthetic query_key_value matrix is
built by hand from known per-head Q/K/V blocks, laid out exactly the way
GPTNeoXAttention.forward interprets its own weight (per-head-contiguous
thirds), and the test asserts the helper recovers each block exactly.
This is a correctness test, not a regression test — if a constructed,
known-layout matrix doesn't split back into the blocks it was built from,
the splitting logic is wrong, independent of any real model.
"""
import numpy as np
import pytest

from core.pythia_weights import split_qkv_gptneox, extract_v_gptneox

NUM_HEADS  = 4
HEAD_SIZE  = 3
HIDDEN_IN  = 6          # deliberately != num_heads * head_size, so the
                         # test also covers non-square input dimensions


def _build_reference_blocks(seed=0):
    """One distinct random (head_size, hidden_in) block per head per Q/K/V."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((NUM_HEADS, HEAD_SIZE, HIDDEN_IN)).astype(np.float32)
    K = rng.standard_normal((NUM_HEADS, HEAD_SIZE, HIDDEN_IN)).astype(np.float32)
    V = rng.standard_normal((NUM_HEADS, HEAD_SIZE, HIDDEN_IN)).astype(np.float32)
    return Q, K, V


def _assemble_qkv_weight(Q, K, V):
    """
    Interleave per head, matching GPTNeoXAttention's own
    `.view(..., num_heads, 3 * head_size)` layout: for head h, the
    3*head_size output rows are [Q_h | K_h | V_h] contiguous.
    """
    rows = []
    for h in range(NUM_HEADS):
        rows.append(Q[h])
        rows.append(K[h])
        rows.append(V[h])
    return np.concatenate(rows, axis=0)   # (num_heads * 3 * head_size, hidden_in)


class TestSplitQKVGPTNeoX:

    def test_recovers_v_blocks_exactly(self):
        Q, K, V = _build_reference_blocks()
        weight  = _assemble_qkv_weight(Q, K, V)

        out = split_qkv_gptneox(weight, num_heads=NUM_HEADS, head_size=HEAD_SIZE)
        v_expected = V.reshape(NUM_HEADS * HEAD_SIZE, HIDDEN_IN)

        assert out["V"].shape == (NUM_HEADS * HEAD_SIZE, HIDDEN_IN)
        np.testing.assert_array_equal(out["V"], v_expected)

    def test_recovers_q_and_k_blocks_exactly(self):
        Q, K, V = _build_reference_blocks(seed=1)
        weight  = _assemble_qkv_weight(Q, K, V)
        out     = split_qkv_gptneox(weight, num_heads=NUM_HEADS, head_size=HEAD_SIZE)

        np.testing.assert_array_equal(out["Q"], Q.reshape(NUM_HEADS * HEAD_SIZE, HIDDEN_IN))
        np.testing.assert_array_equal(out["K"], K.reshape(NUM_HEADS * HEAD_SIZE, HIDDEN_IN))

    def test_blocks_are_disjoint(self):
        """A change to only the K block must not perturb the recovered V block —
        catches an off-by-one in the (3, head_size) sub-reshape that would
        otherwise silently blend Q/K/V rows together."""
        Q, K, V = _build_reference_blocks(seed=2)
        weight  = _assemble_qkv_weight(Q, K, V)
        out_before = split_qkv_gptneox(weight, num_heads=NUM_HEADS, head_size=HEAD_SIZE)

        K2 = K.copy()
        K2[1] += 100.0   # perturb only head 1's K block
        weight2 = _assemble_qkv_weight(Q, K2, V)
        out_after = split_qkv_gptneox(weight2, num_heads=NUM_HEADS, head_size=HEAD_SIZE)

        np.testing.assert_array_equal(out_after["V"], out_before["V"])
        np.testing.assert_array_equal(out_after["Q"], out_before["Q"])
        assert not np.allclose(out_after["K"], out_before["K"])

    def test_wrong_head_count_raises(self):
        """Shape mismatch must fail loudly, not silently reshape garbage —
        this is the check that catches a stale num_heads/head_size read
        from the wrong config object."""
        Q, K, V = _build_reference_blocks(seed=3)
        weight  = _assemble_qkv_weight(Q, K, V)
        with pytest.raises(ValueError):
            split_qkv_gptneox(weight, num_heads=NUM_HEADS + 1, head_size=HEAD_SIZE)

    def test_accepts_torch_tensor_input(self):
        """analyze_value_eigenspectrum passes real .weight tensors, not
        ndarrays — the helper must accept both without a manual .numpy() at
        every call site."""
        torch = pytest.importorskip("torch")
        Q, K, V = _build_reference_blocks(seed=4)
        weight_np = _assemble_qkv_weight(Q, K, V)
        weight_t  = torch.tensor(weight_np)

        out = split_qkv_gptneox(weight_t, num_heads=NUM_HEADS, head_size=HEAD_SIZE)
        np.testing.assert_allclose(
            out["V"], V.reshape(NUM_HEADS * HEAD_SIZE, HIDDEN_IN), rtol=1e-5
        )


class _FakeAttention:
    """Minimal stand-in for GPTNeoXAttention exposing exactly what
    extract_v_gptneox reads — weight, num_attention_heads, head_size."""
    def __init__(self, weight, num_heads, head_size):
        class _W:
            def __init__(self, arr): self._arr = arr
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self._arr
        self.query_key_value = type("QKV", (), {"weight": _W(weight)})()
        self.num_attention_heads = num_heads
        self.head_size = head_size


class _FakeLayer:
    def __init__(self, attention):
        self.attention = attention


def test_extract_v_gptneox_wrapper():
    """The convenience wrapper used directly in analyze_value_eigenspectrum's
    Pythia branch — one layer in, square V matrix out."""
    Q, K, V = _build_reference_blocks(seed=5)
    weight  = _assemble_qkv_weight(Q, K, V)
    layer   = _FakeLayer(_FakeAttention(weight, NUM_HEADS, HEAD_SIZE))

    v_out = extract_v_gptneox(layer)
    np.testing.assert_array_equal(v_out, V.reshape(NUM_HEADS * HEAD_SIZE, HIDDEN_IN))

    class TestAttributeContract:
    """The stub in this file uses the exact attribute names the old
    implementation hardcoded, so it could not catch an upstream rename.
    These cases vary the names."""

    def _weight(self, num_heads, head_size):
        import numpy as np
        d = num_heads * head_size
        return np.arange(3 * d * d, dtype=np.float32).reshape(3 * d, d)

    def _layer(self, attn):
        return type("Layer", (), {"attention": attn})()

    def test_resolves_head_dim_alias(self):
        from core.pythia_weights import split_qkv_from_layer
        attn = type("Attn", (), {
            "num_heads": 4,
            "head_dim":  8,                       # not head_size
            "query_key_value": type("L", (), {"weight": self._weight(4, 8)})(),
        })()
        parts = split_qkv_from_layer(self._layer(attn))
        assert parts["_geometry"]["num_heads"] == 4
        assert parts["V"].shape == (32, 32)

    def test_resolves_from_config_only(self):
        from core.pythia_weights import split_qkv_from_layer
        cfg  = type("Cfg", (), {"num_attention_heads": 4, "hidden_size": 32})()
        attn = type("Attn", (), {
            "config": cfg,
            "query_key_value": type("L", (), {"weight": self._weight(4, 8)})(),
        })()
        assert split_qkv_from_layer(self._layer(attn))["V"].shape == (32, 32)

    def test_geometry_weight_shape_mismatch_raises(self):
        """Stale attributes that disagree with the weight must raise, not
        produce a misaligned split."""
        import pytest
        from core.pythia_weights import split_qkv_from_layer
        attn = type("Attn", (), {
            "num_attention_heads": 8,             # wrong: weight is 4 heads
            "head_size": 8,
            "query_key_value": type("L", (), {"weight": self._weight(4, 8)})(),
        })()
        with pytest.raises(ValueError, match="output rows"):
            split_qkv_from_layer(self._layer(attn))

    def test_unresolvable_geometry_reports_available_names(self):
        import pytest
        from core.pythia_weights import split_qkv_from_layer
        attn = type("Attn", (), {
            "query_key_value": type("L", (), {"weight": self._weight(4, 8)})(),
        })()
        with pytest.raises(AttributeError, match="cannot resolve head geometry"):
            split_qkv_from_layer(self._layer(attn))
