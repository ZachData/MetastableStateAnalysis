"""
p1d_cluster_ensemble/neox_block.py — one GPT-NeoX block, applied to any
residual, so a null draw can be read by the model's own attention.

Item (3)'s null A (`status-1d.md` "Attention communities") feeds Gaussian
draws through layer ``l``'s LN1, QKV and rotary positions exactly as the
real tokens went through them. Nothing here is fitted: the weights are the
checkpoint's. The map is checked against the stored run
(`tests/test_phase1d_attention_null.py` on a toy block;
`attention_null.py --verify` on the real one): recomputing a stored layer's
attention from its stored residual must reproduce ``attentions.npz``, and a
block applied to ``hidden[l]`` must give ``hidden[l + 1]``.

Pythia's layout (transformers 4.44 ``GPTNeoXLayer``): parallel residual,
``x + attn(LN1 x) + mlp(LN2 x)``; QKV fused and interleaved per head;
rotary on the first ``rotary_pct * head_size`` dimensions, half-split
(``rotate_half``); logits scaled by ``1/sqrt(head_size)``; softmax in
float32. ``hidden_states[-1]`` is after the final LN, so block ``L-1``'s
output is not in the stored stream and cannot be checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np


@dataclass
class NeoxBlockWeights:
    """One block's parameters as float32 numpy arrays (torch-free to hold)."""
    ln1_w: np.ndarray
    ln1_b: np.ndarray
    ln2_w: np.ndarray
    ln2_b: np.ndarray
    qkv_w: np.ndarray       # (3 d, d)
    qkv_b: np.ndarray       # (3 d,)
    dense_w: np.ndarray     # (d, d)
    dense_b: np.ndarray
    fc_in_w: np.ndarray     # (4 d, d)
    fc_in_b: np.ndarray
    fc_out_w: np.ndarray    # (d, 4 d)
    fc_out_b: np.ndarray
    n_heads: int
    rotary_ndims: int
    rotary_base: float
    eps: float
    parallel: bool = True

    @property
    def d(self) -> int:
        return int(self.ln1_w.shape[0])

    @property
    def head_size(self) -> int:
        return self.d // self.n_heads


def _layer_norm(X, w, b, eps):
    import torch
    return torch.nn.functional.layer_norm(X, (X.shape[-1],), w, b, eps)


def _rotary(n_pos: Sequence[int], ndims: int, base: float, dtype):
    import torch
    inv = 1.0 / (base ** (torch.arange(0, ndims, 2, dtype=torch.float32) / ndims))
    t = torch.as_tensor(np.asarray(n_pos), dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    import torch
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _t(a, dtype):
    import torch
    return torch.as_tensor(a, dtype=dtype)


def _qkv(X, W: NeoxBlockWeights, positions, dtype):
    """Rotated q, k and v, each (heads, n, head_size), from the raw residual ``X``."""
    import torch
    n, hs, H = X.shape[0], W.head_size, W.n_heads
    h = _layer_norm(X, _t(W.ln1_w, dtype), _t(W.ln1_b, dtype), W.eps)
    qkv = h @ _t(W.qkv_w, dtype).T + _t(W.qkv_b, dtype)
    qkv = qkv.view(n, H, 3 * hs).permute(1, 0, 2)
    q, k, v = qkv[..., :hs], qkv[..., hs:2 * hs], qkv[..., 2 * hs:]
    r = W.rotary_ndims
    cos, sin = _rotary(positions, r, W.rotary_base, dtype)
    q = torch.cat([q[..., :r] * cos + _rotate_half(q[..., :r]) * sin, q[..., r:]], dim=-1)
    k = torch.cat([k[..., :r] * cos + _rotate_half(k[..., :r]) * sin, k[..., r:]], dim=-1)
    return q, k, v


def attention_probs(X: np.ndarray, W: NeoxBlockWeights,
                    positions: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Block ``W``'s attention matrices on residual rows ``X`` (n, d): (heads, n, n).

    ``positions`` are the rotary positions of the rows (default ``0..n-1``).
    The causal mask is by row order, which is position order in every caller.
    """
    import torch
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    n = X.shape[0]
    pos = np.arange(n) if positions is None else np.asarray(positions)
    with torch.no_grad():
        q, k, _ = _qkv(X, W, pos, torch.float32)
        s = (q @ k.transpose(-1, -2)) / np.sqrt(W.head_size)
        mask = torch.ones(n, n, dtype=torch.bool).triu(1)
        s = s.masked_fill(mask, torch.finfo(torch.float32).min)
        return torch.softmax(s, dim=-1).numpy()


def block_forward(X: np.ndarray, W: NeoxBlockWeights,
                  positions: Optional[Sequence[int]] = None) -> np.ndarray:
    """The block's output residual on rows ``X`` (n, d)."""
    import torch
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    n = X.shape[0]
    pos = np.arange(n) if positions is None else np.asarray(positions)
    with torch.no_grad():
        q, k, v = _qkv(X, W, pos, torch.float32)
        s = (q @ k.transpose(-1, -2)) / np.sqrt(W.head_size)
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1),
                          torch.finfo(torch.float32).min)
        ctx = (torch.softmax(s, dim=-1) @ v).permute(1, 0, 2).reshape(n, W.d)
        attn = ctx @ _t(W.dense_w, torch.float32).T + _t(W.dense_b, torch.float32)
        src = X if W.parallel else X + attn
        h2 = _layer_norm(src, _t(W.ln2_w, torch.float32), _t(W.ln2_b, torch.float32), W.eps)
        mid = torch.nn.functional.gelu(h2 @ _t(W.fc_in_w, torch.float32).T
                                       + _t(W.fc_in_b, torch.float32))
        mlp = mid @ _t(W.fc_out_w, torch.float32).T + _t(W.fc_out_b, torch.float32)
        return (X + attn + mlp).numpy()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def weights_from_hf_layer(layer, config) -> NeoxBlockWeights:
    """Extract one ``GPTNeoXLayer``'s parameters."""
    def a(p):
        return p.detach().cpu().float().numpy().copy()
    att, mlp = layer.attention, layer.mlp
    hs = config.hidden_size // config.num_attention_heads
    return NeoxBlockWeights(
        ln1_w=a(layer.input_layernorm.weight), ln1_b=a(layer.input_layernorm.bias),
        ln2_w=a(layer.post_attention_layernorm.weight),
        ln2_b=a(layer.post_attention_layernorm.bias),
        qkv_w=a(att.query_key_value.weight), qkv_b=a(att.query_key_value.bias),
        dense_w=a(att.dense.weight), dense_b=a(att.dense.bias),
        fc_in_w=a(mlp.dense_h_to_4h.weight), fc_in_b=a(mlp.dense_h_to_4h.bias),
        fc_out_w=a(mlp.dense_4h_to_h.weight), fc_out_b=a(mlp.dense_4h_to_h.bias),
        n_heads=int(config.num_attention_heads),
        rotary_ndims=int(hs * config.rotary_pct),
        rotary_base=float(config.rotary_emb_base),
        eps=float(config.layer_norm_eps),
        parallel=bool(config.use_parallel_residual))


@lru_cache(maxsize=2)
def load_blocks(repo_id: str, revision: str) -> tuple:
    """Every block of a cached checkpoint, as ``NeoxBlockWeights`` (one load per process)."""
    from transformers import GPTNeoXForCausalLM
    model = GPTNeoXForCausalLM.from_pretrained(repo_id, revision=revision,
                                               torch_dtype="float32")
    cfg = model.config
    if getattr(cfg, "rope_scaling", None):
        raise ValueError(f"{repo_id}@{revision}: rope_scaling {cfg.rope_scaling} is not implemented")
    blocks = tuple(weights_from_hf_layer(l, cfg) for l in model.gpt_neox.layers)
    del model
    return blocks


_FIELDS = tuple(f for f in NeoxBlockWeights.__dataclass_fields__)


def export_blocks(repo_id: str, revision: str, out_dir) -> int:
    """
    Write each block to ``out_dir/block_{l}.npz`` so a worker loads ~50 MB,
    not the whole model. Returns the block count. The directory is under
    ``data/`` and is never committed.
    """
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blocks = load_blocks(repo_id, revision)
    for i, b in enumerate(blocks):
        np.savez(out / f"block_{i}.npz", **{f: np.asarray(getattr(b, f)) for f in _FIELDS})
    (out / "source.txt").write_text(f"{repo_id}@{revision}\n")
    return len(blocks)


def read_block(path) -> NeoxBlockWeights:
    z = np.load(path)
    kw = {f: z[f] for f in _FIELDS}
    for f in ("n_heads", "rotary_ndims"):
        kw[f] = int(kw[f])
    kw["rotary_base"], kw["eps"] = float(kw["rotary_base"]), float(kw["eps"])
    kw["parallel"] = bool(kw["parallel"])
    return NeoxBlockWeights(**kw)
