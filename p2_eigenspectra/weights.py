"""
weights.py — Phase 2: Weight extraction and QK decomposition.

Extracts per-head W_Q and W_K matrices in canonical (d_model, d_head) orientation.
The canonical orientation is essential for computing QK products and analyzing
induction patterns where the query-key coupling logit is:

  logit(i, j) = x_i^T (W_Q W_K^T) x_j

Convention: W_Q and W_K are both (d_model, d_head) where d_model >= d_head.
For nn.Linear(in_features=d_model, out_features=d_head), the weight matrix
is stored as (d_head, d_model) transposed in rows, so extraction requires
careful index arithmetic.

Functions
---------
extract_qk_per_head      : per-head W_Q, W_K in canonical orientation
_add_qk_arrays_to_decomposition : integrate into artifact save structure

Bugs fixed:
  #2 : Extract per-head W_Q, W_K matrices in canonical (d_model, d_head) orientation
"""

import numpy as np


# -----------
# Per-head QK extraction
# Bug #2: Canonical (d_model, d_head) orientation
# -----------

def extract_qk_per_head(model, model_type: str | None = None) -> dict:
    """
    FIX Bug #2: Per-head W_Q, W_K matrices in canonical (d_model, d_head) orientation.
    
    For ALBERT (shared): single set of per-head matrices.
    For GPT-2 or BERT (per-layer): list of lists, indexed [layer][head].
    
    The canonical orientation ensures:
      logit(i,j) = x_i^T (W_Q @ W_K^T) x_j
    
    where x_i, x_j are row vectors in residual stream space.

    Parameters
    ----------
    model       : transformer model instance
    model_type  : str — "albert", "gpt2", or "bert" (auto-detected if None)

    Returns
    -------
    dict with:
      wq_per_head : list[(d_model, d_head)] for shared models or
                    list[list[(d_model, d_head)]] for per-layer models
      wk_per_head : corresponding list of W_K matrices
      is_per_layer: bool indicating whether results are per-layer
      layer_names : list of layer name strings ("shared" for ALBERT)
    """
    # Auto-detect model type if not provided
    if model_type is None:
        if hasattr(model, 'albert'):
            model_type = "albert"
        elif hasattr(model, 'transformer'):
            model_type = "gpt2"
        elif hasattr(model, 'encoder'):
            model_type = "bert"
        else:
            raise ValueError("Cannot auto-detect model type. Provide model_type explicitly.")

    if model_type == "albert":
        return _extract_albert_qk(model)
    elif model_type == "bert":
        return _extract_bert_qk(model)
    elif model_type == "gpt2":
        return _extract_gpt2_qk(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _extract_albert_qk(model) -> dict:
    """Extract per-head QK for ALBERT (shared across layers)."""
    attn = model.encoder.albert_layer_groups[0].albert_layers[0].attention

    # For nn.Linear with weight shape (out_features, in_features):
    # We need to slice by output indices (head_id * d_head : (head_id+1) * d_head)
    # then transpose to get (d_model, d_head) orientation
    W_Q_full = attn.query.weight.detach().cpu().float().numpy()  # (d_model, d_model)
    W_K_full = attn.key.weight.detach().cpu().float().numpy()    # (d_model, d_model)

    d_model = W_Q_full.shape[0]
    n_heads = attn.num_attention_heads
    d_head  = d_model // n_heads

    wq_per_head = []
    wk_per_head = []
    
    for h in range(n_heads):
        s = h * d_head
        e = s + d_head
        # Per-head rows of the weight matrix, shaped (d_head, d_model)
        # Transpose to canonical (d_model, d_head)
        W_Q_h = W_Q_full[s:e, :].T       # (d_model, d_head)
        W_K_h = W_K_full[s:e, :].T       # (d_model, d_head)
        wq_per_head.append(W_Q_h)
        wk_per_head.append(W_K_h)

    return {
        "wq_per_head": wq_per_head,
        "wk_per_head": wk_per_head,
        "is_per_layer": False,
        "layer_names": ["shared"],
    }


def _extract_bert_qk(model) -> dict:
    """Extract per-head QK for BERT (per-layer)."""
    all_wq = []
    all_wk = []
    layer_names = []

    for i, layer in enumerate(model.encoder.layer):
        W_Q_full = layer.attention.self.query.weight.detach().cpu().float().numpy()
        W_K_full = layer.attention.self.key.weight.detach().cpu().float().numpy()

        d_model = W_Q_full.shape[0]
        n_heads = layer.attention.self.num_attention_heads
        d_head  = d_model // n_heads

        wq_heads = []
        wk_heads = []
        
        for h in range(n_heads):
            s = h * d_head
            e = s + d_head
            W_Q_h = W_Q_full[s:e, :].T
            W_K_h = W_K_full[s:e, :].T
            wq_heads.append(W_Q_h)
            wk_heads.append(W_K_h)

        all_wq.append(wq_heads)
        all_wk.append(wk_heads)
        layer_names.append(f"layer_{i}")

    return {
        "wq_per_head": all_wq,
        "wk_per_head": all_wk,
        "is_per_layer": True,
        "layer_names": layer_names,
    }


def _extract_gpt2_qk(model) -> dict:
    """Extract per-head QK for GPT-2 (per-layer, fused QKV)."""
    all_wq = []
    all_wk = []
    layer_names = []

    for i, block in enumerate(model.h):
        # Conv1D: weight shape is (in_features, out_features)
        c_attn_w = block.attn.c_attn.weight.detach().cpu().float().numpy()

        d_model = c_attn_w.shape[0]
        # Q, K, V split: each gets d_model // 3 or d_model // 3 width
        # Actually GPT-2 uses full d_model for each of Q, K, V
        d_qkv = d_model  # each of Q, K, V has this output dimension
        
        W_Q_full = c_attn_w[:, :d_qkv]          # (d_model, d_model) for Q
        W_K_full = c_attn_w[:, d_qkv:2*d_qkv]   # (d_model, d_model) for K

        n_heads = block.attn.num_heads
        d_head  = d_model // n_heads

        wq_heads = []
        wk_heads = []
        
        for h in range(n_heads):
            s = h * d_head
            e = s + d_head
            # For Conv1D, weight is (out, in), so columns are the inputs
            # Per-head: columns [s:e] are the d_head dimensions for this head
            W_Q_h = W_Q_full[:, s:e]       # (d_model, d_head)
            W_K_h = W_K_full[:, s:e]       # (d_model, d_head)
            wq_heads.append(W_Q_h)
            wk_heads.append(W_K_h)

        all_wq.append(wq_heads)
        all_wk.append(wk_heads)
        layer_names.append(f"layer_{i}")

    return {
        "wq_per_head": all_wq,
        "wk_per_head": all_wk,
        "is_per_layer": True,
        "layer_names": layer_names,
    }


# -----------
# Integration with decomposition workflow
# -----------

def _add_qk_arrays_to_decomposition(arrays: dict, qk_data: dict) -> None:
    """
    Mutate arrays dict to add wq_head/wk_head keys for save_weight_decomposition.
    
    FIX Bug #2: Each (Q, K) pair is stored in canonical (d_model, d_head) orientation.

    Parameters
    ----------
    arrays  : dict to mutate (maps string keys to numpy arrays)
    qk_data : output of extract_qk_per_head
    """
    if qk_data["is_per_layer"]:
        for i, lname in enumerate(qk_data["layer_names"]):
            for h, (Q, K) in enumerate(
                zip(qk_data["wq_per_head"][i], qk_data["wk_per_head"][i])
            ):
                arrays[f"wq_head{h}_{lname}"] = Q
                arrays[f"wk_head{h}_{lname}"] = K
    else:
        for h, (Q, K) in enumerate(
            zip(qk_data["wq_per_head"], qk_data["wk_per_head"])
        ):
            arrays[f"wq_head{h}_shared"] = Q
            arrays[f"wk_head{h}_shared"] = K


# -----------
# Verification utilities
# -----------

def verify_qk_orientation(WQ: np.ndarray, WK: np.ndarray) -> bool:
    """
    Verify that W_Q and W_K are in canonical (d_model, d_head) orientation.
    
    Returns True if both are 2D with first dimension >= second dimension.
    """
    if WQ.ndim != 2 or WK.ndim != 2:
        return False
    if WQ.shape != WK.shape:
        return False
    if WQ.shape[0] < WQ.shape[1]:
        return False
    return True


def compute_qk_product(WQ: np.ndarray, WK: np.ndarray) -> np.ndarray:
    """
    Compute the QK product matrix in canonical orientation.
    
    Parameters
    ----------
    WQ, WK : (d_model, d_head) each
    
    Returns
    -------
    M : (d_model, d_model) — the logit coupling matrix M = WQ @ WK^T
    """
    assert verify_qk_orientation(WQ, WK), "WQ and WK must be (d_model, d_head)"
    return WQ @ WK.T