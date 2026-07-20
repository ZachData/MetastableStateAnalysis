"""
tests/test_head_ablation_math.py — pure-numpy tests for
p2_eigenspectra/head_ablation.py's head-slice math (item 5) and
p5_single_mstate_analysis/train_tuned_lens.py's fitting logic (item 4,
Group E fix).

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


# ---------------------------------------------------------------------------
# train_tuned_lens — fitting logic
# ---------------------------------------------------------------------------

class TestFitAffineTranslator:
    def test_recovers_planted_affine_map(self):
        """Oracle: generate H_final = A* H + b* exactly; the fit must
        recover (A*, b*) to numerical precision when n >> d."""
        from p5_single_mstate_analysis.train_tuned_lens import fit_affine_translator
        rng = np.random.default_rng(0)
        d, n = 6, 400
        A_star = rng.standard_normal((d, d))
        b_star = rng.standard_normal(d)
        H = rng.standard_normal((n, d))
        H_final = H @ A_star.T + b_star
        A, b = fit_affine_translator(H, H_final, ridge=1e-8)
        np.testing.assert_allclose(A, A_star, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(b, b_star, rtol=1e-3, atol=1e-3)

    def test_identity_at_final_layer(self):
        from p5_single_mstate_analysis.train_tuned_lens import (
            fit_affine_translator, identity_deviation,
        )
        rng = np.random.default_rng(1)
        H = rng.standard_normal((300, 5))
        A, b = fit_affine_translator(H, H, ridge=1e-8)
        assert identity_deviation(A, b) < 1e-3

    def test_shape_mismatch_raises(self):
        from p5_single_mstate_analysis.train_tuned_lens import fit_affine_translator
        with pytest.raises(ValueError):
            fit_affine_translator(np.zeros((10, 4)), np.zeros((10, 5)))


class TestLensStackAndIO:
    def test_stack_fit_and_npz_roundtrip(self, tmp_path):
        """save_lens must write the exact A_L{i}/b_L{i} format
        tuned_lens_cluster.load_tuned_lens reads — the producer/consumer
        contract this whole fix exists for."""
        from p5_single_mstate_analysis.train_tuned_lens import (
            fit_lens_from_activation_stack, save_lens,
        )
        from p5_single_mstate_analysis.tuned_lens_cluster import (
            load_tuned_lens, apply_tuned_lens,
        )
        rng = np.random.default_rng(2)
        n_layers, n_tokens, d = 4, 200, 5
        acts = rng.standard_normal((n_layers, n_tokens, d)).astype(np.float32)
        lens = fit_lens_from_activation_stack(acts, ridge=1e-6)
        assert set(lens.keys()) == set(range(n_layers))
        # near-identity at final layer
        assert lens[n_layers - 1]["identity_deviation"] < 1e-2

        out = save_lens(lens, tmp_path / "tuned_lens_test.npz",
                        meta={"model": "test"})
        loaded = load_tuned_lens(out)
        assert loaded is not None
        assert set(loaded.keys()) == set(range(n_layers))
        for L in range(n_layers):
            np.testing.assert_allclose(loaded[L]["A"], lens[L]["A"])
            np.testing.assert_allclose(loaded[L]["b"], lens[L]["b"])

        # apply_tuned_lens consumes the loaded dict without error
        v = rng.standard_normal(d).astype(np.float32)
        out_v = apply_tuned_lens(v, 1, loaded)
        np.testing.assert_allclose(
            out_v, loaded[1]["A"] @ v + loaded[1]["b"], rtol=1e-5)
        # sidecar written
        assert (tmp_path / "tuned_lens_test.json").exists()

    def test_translated_vector_closer_to_final_than_raw(self):
        """The functional claim behind the fix: on structured data (an
        affine drift per layer), the translated layer-0 vector must land
        closer to the final state than the untranslated one — otherwise
        the lens adds nothing over the frozen-head fallback."""
        from p5_single_mstate_analysis.train_tuned_lens import (
            fit_lens_from_activation_stack,
        )
        rng = np.random.default_rng(3)
        n_tokens, d = 300, 6
        base = rng.standard_normal((n_tokens, d))
        M = np.eye(d) + 0.3 * rng.standard_normal((d, d))
        shift = rng.standard_normal(d)
        acts = np.stack([base, base @ M.T + shift,
                         (base @ M.T + shift) @ M.T + shift], axis=0)
        lens = fit_lens_from_activation_stack(acts, ridge=1e-6)
        A0, b0 = lens[0]["A"], lens[0]["b"]
        translated = acts[0] @ A0.T + b0
        err_translated = np.linalg.norm(translated - acts[-1])
        err_raw = np.linalg.norm(acts[0] - acts[-1])
        assert err_translated < 0.1 * err_raw
