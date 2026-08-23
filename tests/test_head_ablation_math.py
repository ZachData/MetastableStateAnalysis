"""
tests/test_head_ablation_math.py — pure-numpy tests for
p2_eigenspectra/head_ablation.py's head-slice math (item 5).

Previously also covered p5_single_mstate_analysis/train_tuned_lens.py.
That half moved to archive/tests/test_p5_tuned_lens_math.py when Phase 5
was archived; this file is p2-only now.

Exactness oracle for head_delta_from_projection: the per-head deltas
must sum EXACTLY (up to float) to the full projection output, in both
weight conventions — Conv1D (GPT-2, y = x @ W) and Linear (GPT-NeoX,
y = x @ W.T). If either slice orientation were wrong, the sum test
fails immediately.

No torch anywhere on these paths.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# head_delta_from_projection — decomposition exactness
# ---------------------------------------------------------------------------

class TestHeadDeltaExactness:
    def _setup(self, seed, n_heads=4, d_head=3, n_tokens=6):
        rng = np.random.default_rng(seed)
        d_model = n_heads * d_head
        pre = rng.standard_normal((n_tokens, d_model)).astype(np.float32)
        W = rng.standard_normal((d_model, d_model)).astype(np.float32)
        return pre, W, n_heads, d_head

    def test_conv1d_heads_sum_to_full_projection(self):
        from p2_eigenspectra.head_ablation import head_delta_from_projection
        pre, W, n_heads, d_head = self._setup(0)
        full = pre @ W                                       # Conv1D map
        total = sum(head_delta_from_projection(pre, W, h, d_head, conv1d=True)
                    for h in range(n_heads))
        np.testing.assert_allclose(total, full, rtol=1e-5, atol=1e-5)

    def test_linear_heads_sum_to_full_projection(self):
        from p2_eigenspectra.head_ablation import head_delta_from_projection
        pre, W, n_heads, d_head = self._setup(1)
        full = pre @ W.T                                     # nn.Linear map
        total = sum(head_delta_from_projection(pre, W, h, d_head, conv1d=False)
                    for h in range(n_heads))
        np.testing.assert_allclose(total, full, rtol=1e-5, atol=1e-5)

    def test_single_head_isolation(self):
        """Zeroing every other head's columns of `pre` reproduces that
        head's delta alone — the ablation semantics downstream relies on."""
        from p2_eigenspectra.head_ablation import head_delta_from_projection
        pre, W, n_heads, d_head = self._setup(2)
        h = 2
        masked = np.zeros_like(pre)
        s, e = h * d_head, (h + 1) * d_head
        masked[:, s:e] = pre[:, s:e]
        for conv1d in (True, False):
            full_masked = masked @ (W if conv1d else W.T)
            delta = head_delta_from_projection(pre, W, h, d_head, conv1d)
            np.testing.assert_allclose(delta, full_masked, rtol=1e-5, atol=1e-5)

    def test_conventions_differ_on_asymmetric_weight(self):
        """Guard against the two conventions silently collapsing into one
        code path: for a non-symmetric W they must disagree."""
        from p2_eigenspectra.head_ablation import head_delta_from_projection
        pre, W, _, d_head = self._setup(3)
        a = head_delta_from_projection(pre, W, 0, d_head, conv1d=True)
        b = head_delta_from_projection(pre, W, 0, d_head, conv1d=False)
        assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# _locate_attn_projection — dispatch on fake models (no forward needed)
# ---------------------------------------------------------------------------

class _M:
    pass


def _fake_gpt2(n_heads=2, d_model=8):
    import numpy as _np

    class _WObj:
        def __init__(self, shape):
            self.shape = shape

    block = _M()
    block.attn = _M()
    block.attn.num_heads = n_heads
    block.attn.c_proj = _M()
    block.attn.c_proj.weight = _WObj((d_model, d_model))
    m = _M()
    m.h = [block]
    return m


def _fake_neox(n_heads=2, head_size=4):
    block = _M()
    block.attention = _M()
    block.attention.num_attention_heads = n_heads
    block.attention.head_size = head_size
    block.attention.query_key_value = _M()
    block.attention.dense = _M()
    m = _M()
    m.layers = [block]
    return m


class TestLocateAttnProjection:
    def test_gpt2_route(self):
        from p2_eigenspectra.head_ablation import _locate_attn_projection
        m = _fake_gpt2()
        blocks, get_proj, get_heads, conv1d = _locate_attn_projection(m)
        assert conv1d is True
        assert get_proj(blocks[0]) is m.h[0].attn.c_proj
        assert get_heads(blocks[0]) == (2, 4)

    def test_neox_route(self):
        from p2_eigenspectra.head_ablation import _locate_attn_projection
        m = _fake_neox()
        blocks, get_proj, get_heads, conv1d = _locate_attn_projection(m)
        assert conv1d is False
        assert get_proj(blocks[0]) is m.layers[0].attention.dense
        assert get_heads(blocks[0]) == (2, 4)

    def test_unsupported_raises_not_implemented(self):
        from p2_eigenspectra.head_ablation import _locate_attn_projection
        with pytest.raises(NotImplementedError):
            _locate_attn_projection(object())
