"""
attn_biases.py — Attention biases: extraction, persistence, and the
token-independent write (frames item 10).

Why this module exists
----------------------
Every OV/QK quantity in the project is weight-only. The attention biases are
not a rounding term.

**QK side.** Expanding the real bilinear with biases:

    logit = x_i^T W_Q R W_K^T x_j        <- the only term weight-only keeps
          + b_q^T R W_K^T x_j            <- per-key, query-independent
          + x_i^T W_Q R b_k              <- per-query, key-independent
          + b_q^T R b_k                  <- constant

The second term is a per-key logit offset applied regardless of what is
querying — structurally the shape of attention-sink behaviour (policy P1).
Measured against true logits on synthetic weights, dropping the biases costs
pearson 0.99 -> and it compounds with the frame and rotary omissions to 0.60
(DESIGN_pythia_frames.md item 10). This is not Pythia-specific: GPT-2's
`c_attn` carries a bias too.

**OV side.** Because attention rows sum to one, the value-path bias
contributes a *token-independent* write every layer:

    drift = b_V W_O^T + b_out

Identical for every token, so it displaces the whole cloud rather than
restructuring it. In Phase 5 it currently lands in `orth_frac`
unattributed. Interpretive rather than a correctness bug, but cheap to close
once the biases are persisted.

Layout, which is where the bugs live
------------------------------------
Three conventions, and they disagree:

  gpt_neox   fused `query_key_value.bias`, PER-HEAD contiguous:
             index = h * 3 * head_size + part * head_size + t
  gpt2       fused `c_attn.bias`, PART contiguous:
             index = part * d_model + h * head_size + t
  albert/bert separate `query.bias` / `key.bias`, each (d_model,)

The GPT-NeoX layout is the one that already produced a shipped bug on the
weight side (see core/pythia_weights.py). Assuming GPT-2's layout on a NeoX
model silently returns the wrong bias for every head but the first.

Pure numpy, torch-optional, duck-typed on module structure so the whole
thing is oracle-testable in the stubbed session.
"""

from __future__ import annotations

import numpy as np


def _np(x) -> np.ndarray:
    """Tensor-like -> float64 ndarray, without importing torch."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Layout-aware splitting
# ---------------------------------------------------------------------------

def split_qkv_bias_gptneox(bias, n_heads: int, head_size: int) -> dict:
    """
    Split a fused GPT-NeoX qkv bias into per-head (bq, bk, bv).

    Layout is PER-HEAD contiguous, mirroring
    core.pythia_weights.split_qkv_gptneox's `.reshape(n_heads, 3, head_size)`.
    Returns dict of lists, one entry per head, each (head_size,).
    """
    b = _np(bias).ravel()
    expected = n_heads * 3 * head_size
    if b.shape[0] != expected:
        raise ValueError(
            f"split_qkv_bias_gptneox: expected n_heads * 3 * head_size = "
            f"{n_heads} * 3 * {head_size} = {expected}, got {b.shape[0]}"
        )
    r = b.reshape(n_heads, 3, head_size)
    return {"bq": [r[h, 0].copy() for h in range(n_heads)],
            "bk": [r[h, 1].copy() for h in range(n_heads)],
            "bv": [r[h, 2].copy() for h in range(n_heads)]}


def split_qkv_bias_gpt2(bias, n_heads: int, head_size: int) -> dict:
    """
    Split a fused GPT-2 `c_attn` bias into per-head (bq, bk, bv).

    Layout is PART contiguous: [all Q | all K | all V], each of width d_model.
    Applying this splitter to a NeoX bias, or the NeoX one to a GPT-2 bias,
    returns plausible arrays of the right shape and the wrong contents for
    every head but the first — which is why each has its own function rather
    than a flag.
    """
    b = _np(bias).ravel()
    d_model = n_heads * head_size
    if b.shape[0] != 3 * d_model:
        raise ValueError(
            f"split_qkv_bias_gpt2: expected 3 * d_model = {3 * d_model}, "
            f"got {b.shape[0]}"
        )
    parts = [b[:d_model], b[d_model:2 * d_model], b[2 * d_model:]]
    out = {}
    for key, p in zip(("bq", "bk", "bv"), parts):
        out[key] = [p[h * head_size:(h + 1) * head_size].copy()
                    for h in range(n_heads)]
    return out


def split_separate_bias(q_bias, k_bias, v_bias, n_heads: int,
                        head_size: int) -> dict:
    """ALBERT/BERT: separate (d_model,) biases, sliced per head."""
    out = {}
    for key, b in (("bq", q_bias), ("bk", k_bias), ("bv", v_bias)):
        arr = _np(b)
        if arr is None:
            out[key] = [np.zeros(head_size) for _ in range(n_heads)]
            continue
        arr = arr.ravel()
        out[key] = [arr[h * head_size:(h + 1) * head_size].copy()
                    for h in range(n_heads)]
    return out


# ---------------------------------------------------------------------------
# Model-level extraction
# ---------------------------------------------------------------------------

def _neox_layers(model):
    inner = getattr(model, "gpt_neox", None)
    return list(getattr(inner, "layers", [])) if inner is not None else []


def _gpt2_blocks(model):
    inner = getattr(model, "transformer", None)
    return list(getattr(inner, "h", [])) if inner is not None else []


def extract_qk_biases(model, model_type: str | None = None) -> dict:
    """
    Per-layer, per-head (bq, bk) in the same shape contract as
    weights.extract_qk_per_head.

    Returns
    -------
    {
      "bq_per_head":  list[list[(head_size,)]] | list[(head_size,)],
      "bk_per_head":  same,
      "bv_per_head":  same,
      "is_per_layer": bool,
      "layer_names":  list[str],
      "has_bias":     bool,          # False means the model genuinely has none
    }

    `has_bias=False` is a real answer, not a failure: some configurations set
    `attention_bias=False`. It must be recorded so a downstream omission is
    distinguishable from an unavailable value.
    """
    cfg = getattr(model, "config", model)
    d_model = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
    n_heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)))
    if not d_model or not n_heads:
        raise ValueError("extract_qk_biases: could not read hidden_size / n_heads")
    head_size = d_model // n_heads

    neox = _neox_layers(model)
    gpt2 = _gpt2_blocks(model)

    bq_all, bk_all, bv_all, names = [], [], [], []
    found_any = False

    if neox:
        for i, layer in enumerate(neox):
            b = getattr(getattr(layer.attention, "query_key_value", None),
                        "bias", None)
            if b is None:
                s = {"bq": [np.zeros(head_size)] * n_heads,
                     "bk": [np.zeros(head_size)] * n_heads,
                     "bv": [np.zeros(head_size)] * n_heads}
            else:
                s = split_qkv_bias_gptneox(b, n_heads, head_size)
                found_any = True
            bq_all.append(s["bq"]); bk_all.append(s["bk"]); bv_all.append(s["bv"])
            names.append(str(i))
    elif gpt2:
        for i, block in enumerate(gpt2):
            b = getattr(getattr(block.attn, "c_attn", None), "bias", None)
            if b is None:
                s = {"bq": [np.zeros(head_size)] * n_heads,
                     "bk": [np.zeros(head_size)] * n_heads,
                     "bv": [np.zeros(head_size)] * n_heads}
            else:
                s = split_qkv_bias_gpt2(b, n_heads, head_size)
                found_any = True
            bq_all.append(s["bq"]); bk_all.append(s["bk"]); bv_all.append(s["bv"])
            names.append(str(i))
    else:
        raise ValueError(
            "extract_qk_biases: unrecognised model structure. Add a branch "
            "rather than defaulting to a layout — the layouts disagree."
        )

    return {
        "bq_per_head": bq_all,
        "bk_per_head": bk_all,
        "bv_per_head": bv_all,
        "is_per_layer": True,
        "layer_names": names,
        "has_bias": found_any,
    }


# ---------------------------------------------------------------------------
# The token-independent write
# ---------------------------------------------------------------------------

def attention_bias_drift(model, layer_idx: int,
                         bias_data: dict | None = None) -> dict:
    """
    The per-layer token-independent residual write from the value path.

        drift = concat_h(b_V_h) @ W_out^T + b_out

    Attention rows sum to one, so this vector is added to EVERY token's
    residual identically. It translates the cloud without restructuring it,
    which is why it lands in Phase 5's `orth_frac` unattributed rather than
    showing up as a recognisable direction.

    Returns dict(drift, norm, from_value_bias, from_output_bias) so the two
    contributions can be reported separately — they have different
    interpretations, the first mediated by W_out and the second not.
    """
    if bias_data is None:
        bias_data = extract_qk_biases(model)

    neox = _neox_layers(model)
    gpt2 = _gpt2_blocks(model)
    if neox:
        attn = neox[layer_idx].attention
        W_out = _np(getattr(attn.dense, "weight"))          # (d_model, d_model)
        b_out = _np(getattr(attn.dense, "bias", None))
        # nn.Linear: y = x @ W^T + b
        w_applied = lambda v: v @ W_out.T
    elif gpt2:
        attn = gpt2[layer_idx].attn
        W_out = _np(getattr(attn.c_proj, "weight"))          # Conv1D: (in, out)
        b_out = _np(getattr(attn.c_proj, "bias", None))
        w_applied = lambda v: v @ W_out
    else:
        raise ValueError("attention_bias_drift: unrecognised model structure")

    bv = bias_data["bv_per_head"][layer_idx]
    v_concat = np.concatenate([np.asarray(b, dtype=np.float64) for b in bv])
    from_value = w_applied(v_concat)
    from_output = b_out if b_out is not None else np.zeros_like(from_value)
    drift = from_value + from_output
    return {
        "drift": drift,
        "norm": float(np.linalg.norm(drift)),
        "from_value_bias": from_value,
        "from_output_bias": from_output,
        "norm_from_value": float(np.linalg.norm(from_value)),
        "norm_from_output": float(np.linalg.norm(from_output)),
    }


def drift_share_of_displacement(drift, displacement) -> dict:
    """
    How much of an observed centroid displacement the drift accounts for.

    Reports both the projection (signed, how much of the drift direction is
    present) and the norm ratio, because a large drift orthogonal to the
    observed displacement means something different from a large drift
    aligned with it.
    """
    d = np.asarray(drift, dtype=np.float64).ravel()
    m = np.asarray(displacement, dtype=np.float64).ravel()
    nd, nm = np.linalg.norm(d), np.linalg.norm(m)
    if nd == 0 or nm == 0:
        return {"cosine": float("nan"), "projected_fraction": 0.0,
                "norm_ratio": float("nan")}
    cos = float(d @ m / (nd * nm))
    return {
        "cosine": cos,
        "projected_fraction": float((d @ m) / (nm ** 2)),
        "norm_ratio": float(nd / nm),
    }


# ---------------------------------------------------------------------------
# Persistence — matching the Phase 2 npz convention
# ---------------------------------------------------------------------------

def add_bias_arrays(arrays: dict, bias_data: dict) -> dict:
    """
    Add per-head bias arrays to a Phase 2 npz payload, in place.

    Key convention mirrors the existing per-head weight keys so a reader can
    pair them without a lookup table:
        bq_head{h}_{layer_name}, bk_head{h}_{layer_name}, bv_head{h}_{layer_name}
    """
    arrays["_has_attn_bias"] = np.array([bool(bias_data.get("has_bias", False))])
    for li, name in enumerate(bias_data["layer_names"]):
        for prefix in ("bq", "bk", "bv"):
            for h, b in enumerate(bias_data[f"{prefix}_per_head"][li]):
                arrays[f"{prefix}_head{h}_{name}"] = np.asarray(b, dtype=np.float32)
    return arrays


def load_qk_biases(npz, layer_name: str, n_heads: int) -> list | None:
    """
    Read back per-head (bq, bk) for one layer, in the form
    qk_context.build_qk_logit_context expects.

    Returns None when the arrays are absent — which callers must record as an
    omission rather than substituting zeros. Zeros are a different claim from
    "unknown", and only the model can say which is true.
    """
    keys = set(npz.keys()) if hasattr(npz, "keys") else set(npz)
    first = f"bq_head0_{layer_name}"
    if first not in keys:
        return None
    out = []
    for h in range(n_heads):
        kq, kk = f"bq_head{h}_{layer_name}", f"bk_head{h}_{layer_name}"
        if kq not in keys or kk not in keys:
            return None
        out.append((np.asarray(npz[kq], dtype=np.float64),
                    np.asarray(npz[kk], dtype=np.float64)))
    return out


def bias_summary_lines(bias_data: dict, drift: dict | None = None) -> list:
    lines = [
        "Attention biases:",
        f"  present       {'yes' if bias_data.get('has_bias') else 'NO (attention_bias=False)'}",
        f"  layers        {len(bias_data.get('layer_names', []))}",
    ]
    if drift is not None:
        lines += [
            f"  drift norm    {drift['norm']:.4f}"
            f" (value {drift['norm_from_value']:.4f}"
            f" + output {drift['norm_from_output']:.4f})",
        ]
    return lines
