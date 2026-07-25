"""
core/pythia_weights.py — GPT-NeoX fused query_key_value splitting.

New module. GPT-NeoX (Pythia) computes Q, K, V from one fused Linear layer
(`attention.query_key_value`, weight shape (3*hidden, hidden)) instead of
GPT-2's Conv1D `c_attn` (weight shape (hidden, 3*hidden)). Both are
"one matrix, three blocks" splits, but the memory layout differs:

  GPT-2 c_attn   : output blocks are contiguous ACROSS all heads —
                   weight[:, :d], weight[:, d:2*d], weight[:, 2*d:]
                   are Q_all_heads, K_all_heads, V_all_heads.
  GPT-NeoX qkv   : output blocks are contiguous PER HEAD — for head h,
                   the 3*head_size slice at offset h*3*head_size is
                   [Q_h (head_size) | K_h (head_size) | V_h (head_size)].
                   This is what GPTNeoXAttention.forward does with
                   `.view(..., num_heads, 3 * head_size)` before slicing
                   query/key/value — the split here mirrors that exactly,
                   just applied to the weight matrix instead of an
                   activation.

Consumers: p1_mstate_tracking/plots.py's analyze_value_eigenspectrum (V
matrix per layer, phase 1) and p2_eigenspectra/head_ablation.py (per-head
Q/K/V, phase 2) — same helper, per the transition plan's note to "reuse
the shape of GPT-2's c_attn splitting logic" for both.
"""

import numpy as np

try:                                    # torch is optional here
    import torch
except ModuleNotFoundError:             # stubbed test session
    torch = None


def _is_torch_tensor(x) -> bool:
    return torch is not None and torch.is_tensor(x)


def _to_numpy(weight) -> np.ndarray:
    """Accept a torch.Tensor, ndarray, or any tensor-like duck type exposing
    .detach()/.cpu()/.numpy(); return a detached float32 ndarray.

    The torch import is optional: everything below the is_tensor check is
    already duck-typed, so an unconditional import was the only thing
    keeping the fused-QKV layout — the source of a previously-shipped bug —
    from having a torch-free oracle.
    """
    if _is_torch_tensor(weight):
        weight = weight.detach().cpu().float().numpy()
    else:
        if hasattr(weight, "detach"):
            weight = weight.detach()
        if hasattr(weight, "cpu"):
            weight = weight.cpu()
        if hasattr(weight, "numpy"):
            weight = weight.numpy()
    return np.asarray(weight, dtype=np.float32)


def split_qkv_gptneox(weight, num_heads: int, head_size: int) -> dict:
    """
    Split a GPT-NeoX fused query_key_value weight into Q, K, V.

    Parameters
    ----------
    weight    : (3 * num_heads * head_size, hidden_in) — the raw
                `attention.query_key_value.weight` tensor or ndarray.
                hidden_in is whatever the layer's input dimension is
                (equal to num_heads * head_size for Pythia's square
                attention blocks, but not assumed equal here).
    num_heads : attention.num_attention_heads
    head_size : attention.head_size

    Returns
    -------
    dict with keys "Q", "K", "V", each (num_heads * head_size, hidden_in) —
    the same square-matrix shape GPT-2's c_attn split already produces, so
    existing eigenspectrum / eigenvalue code that assumes a square V
    matrix does not need to change, only the extraction branch that feeds
    it.
    """
    w = _to_numpy(weight)
    hidden_out, hidden_in = w.shape
    expected = num_heads * 3 * head_size
    if hidden_out != expected:
        raise ValueError(
            f"query_key_value weight has {hidden_out} output rows; "
            f"expected num_heads * 3 * head_size = {num_heads} * 3 * "
            f"{head_size} = {expected}. Check num_heads/head_size against "
            f"the model's actual config before trusting the split."
        )

    # (num_heads, 3, head_size, hidden_in) — matches GPTNeoXAttention's own
    # `.view(*shape[:-1], num_heads, 3 * head_size)` reshape, one level
    # further split so Q/K/V are separately indexable.
    reshaped = w.reshape(num_heads, 3, head_size, hidden_in)

    def _block(idx: int) -> np.ndarray:
        return reshaped[:, idx, :, :].reshape(num_heads * head_size, hidden_in)

    return {"Q": _block(0), "K": _block(1), "V": _block(2)}


def extract_v_gptneox(layer, model_name: str = "") -> np.ndarray:
    """
    Convenience wrapper for the phase 1 / phase 2 call sites: given one
    GPTNeoXLayer, return just the V matrix as a square (d, d) ndarray —
    the same shape analyze_value_eigenspectrum already expects from its
    ALBERT/BERT/GPT-2 branches.
    """
    attn = layer.attention
    qkv  = split_qkv_gptneox(
        attn.query_key_value.weight,
        num_heads=attn.num_attention_heads,
        head_size=attn.head_size,
    )
    return qkv["V"]
