"""
core/pythia_weights.py — GPT-NeoX fused query_key_value splitting.

GPT-NeoX (Pythia) computes Q, K, V from one fused Linear layer
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
Q/K/V, phase 2).

Attribute contract
------------------
head geometry used to be read as `attn.num_attention_heads` /
`attn.head_size` with no fallback. Those names are not stable across
transformers releases — GPTNeoXAttention has variously exposed head_size,
head_dim, and config-sourced values through the attention-interface
refactors. tests/test_pythia_qkv_split.py stubs the exact names, so it
validates the reshape arithmetic and not the contract: an attribute rename
upstream would pass the suite and raise at run time, on a machine holding
a 1.4B checkpoint, partway through a sweep.

_attn_geometry now resolves the geometry from four sources in decreasing
order of directness and cross-checks the result against the weight shape,
which is the one thing that cannot drift.
"""

import warnings

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


def _first_present(obj, *names):
    """First attribute in *names* that exists on *obj* and is not None."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value, name
    return None, None


def _attn_geometry(attn, weight_shape, model_name: str = "") -> tuple:
    """Resolve (num_heads, head_size) for a GPTNeoXAttention module.

    Sources, in order:
      1. the module's own attributes (num_attention_heads / num_heads)
      2. the module's .config
      3. the weight shape, given whichever of the two is known
      4. failure, with the attribute names actually present reported

    Returns (num_heads, head_size, provenance_string).
    """
    hidden_out, hidden_in = weight_shape
    cfg = getattr(attn, "config", None)

    num_heads, nh_src = _first_present(attn, "num_attention_heads", "num_heads")
    if num_heads is None and cfg is not None:
        num_heads, nh_src = _first_present(cfg, "num_attention_heads", "n_head")
        nh_src = f"config.{nh_src}" if nh_src else None

    head_size, hs_src = _first_present(attn, "head_size", "head_dim")
    if head_size is None and cfg is not None:
        hidden_size, _ = _first_present(cfg, "hidden_size", "n_embd")
        if hidden_size is not None and num_heads:
            head_size, hs_src = hidden_size // num_heads, "config.hidden_size//heads"

    # Derive whichever is still missing from the weight itself.
    if num_heads is None and head_size:
        num_heads, nh_src = hidden_out // (3 * head_size), "weight_shape"
    if head_size is None and num_heads:
        head_size, hs_src = hidden_out // (3 * num_heads), "weight_shape"

    if not num_heads or not head_size:
        present = sorted(
            n for n in dir(attn)
            if ("head" in n.lower() or "config" in n.lower()) and not n.startswith("__")
        )
        raise AttributeError(
            f"{model_name or 'GPTNeoXAttention'}: cannot resolve head geometry. "
            f"Tried num_attention_heads/num_heads and head_size/head_dim on the "
            f"module and its .config. Attributes present that look relevant: "
            f"{present}. Add the current name to _attn_geometry rather than "
            f"patching the call site."
        )

    # The cross-check that cannot drift: the fused matrix has exactly
    # 3 * num_heads * head_size output rows, by construction.
    expected = 3 * num_heads * head_size
    if hidden_out != expected:
        raise ValueError(
            f"{model_name or 'GPTNeoXAttention'}: head geometry "
            f"num_heads={num_heads} (from {nh_src}), head_size={head_size} "
            f"(from {hs_src}) implies {expected} output rows, but "
            f"query_key_value.weight has {hidden_out}. The attribute names "
            f"resolved to values from a different model or a stale config; "
            f"the Q/K/V split would be silently misaligned."
        )

    return num_heads, head_size, f"{nh_src}/{hs_src}"


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
    num_heads : attention head count
    head_size : per-head dimension

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


def split_qkv_from_layer(layer, model_name: str = "") -> dict:
    """Resolve head geometry from a GPTNeoXLayer and split its fused QKV."""
    attn = getattr(layer, "attention", None) or getattr(layer, "attn", None)
    if attn is None:
        raise AttributeError(
            f"{model_name or 'layer'}: no .attention or .attn submodule; "
            f"this is not a GPTNeoXLayer."
        )

    qkv_module = getattr(attn, "query_key_value", None)
    if qkv_module is None:
        raise AttributeError(
            f"{model_name or 'layer'}: attention has no .query_key_value. "
            f"Newer GPT-NeoX variants may use separate q/k/v projections; "
            f"if so this module needs an unfused branch, not a workaround "
            f"at the call site."
        )

    # Normalize once, before either use. _attn_geometry needs .shape and
    # split_qkv_gptneox needs an ndarray; the raw qkv_module.weight is only
    # guaranteed to be the duck type _to_numpy documents (.detach/.cpu/
    # .numpy), which doesn't include .shape. Reading .shape off the raw
    # object before converting worked for real torch tensors and bare
    # ndarrays but broke any other conformer of the documented duck type.
    # _to_numpy is idempotent on ndarrays, so calling it again inside
    # split_qkv_gptneox is harmless.
    weight = _to_numpy(qkv_module.weight)
    num_heads, head_size, provenance = _attn_geometry(
        attn, weight.shape, model_name
    )
    parts = split_qkv_gptneox(weight, num_heads=num_heads, head_size=head_size)
    parts["_geometry"] = {
        "num_heads":  int(num_heads),
        "head_size":  int(head_size),
        "provenance": provenance,
    }
    return parts


def extract_v_gptneox(layer, model_name: str = "") -> np.ndarray:
    """
    Given one GPTNeoXLayer, return just the V matrix as a square (d, d)
    ndarray — the same shape analyze_value_eigenspectrum already expects
    from its ALBERT/BERT/GPT-2 branches.
    """
    v = split_qkv_from_layer(layer, model_name)["V"]

    if v.shape[0] != v.shape[1]:
        # Not fatal: analyze_value_eigenspectrum has a non-square fallback
        # that skips eigenvalues and keeps singular values. But it means
        # every eigenvalue-based claim is unavailable for this model, which
        # should be visible rather than inferred from NaNs in the output.
        warnings.warn(
            f"{model_name or 'GPT-NeoX'}: V is {v.shape}, not square. "
            f"Eigenvalues are undefined, so eig_frac_pos_real / "
            f"eig_spectral_radius will be NaN for this model.",
            stacklevel=2,
        )
    return v