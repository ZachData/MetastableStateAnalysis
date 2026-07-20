"""
core/ln_frame.py — Pairwise geometry in the network's own reading frame
(frames item 2).

The model never reads the raw residual stream. Every sub-layer reads
    LN(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
with learned per-channel gamma/beta. That operation is itself a
projection: mean-centering + variance scaling maps x onto a sphere of
radius sqrt(d) inside the (d-1)-dim zero-mean subspace, and gamma
stretches it into an ellipsoid. The Gram matrix of LN(x_i) is therefore
"the geometry attention actually operates in" — a second frame to sit
next to the existing L2-sphere frame, not a replacement for it.

Split pure/extraction, following core/pythia_weights.py's pattern:
  - ln_transform / ln_frame_gram : pure numpy, oracle-testable torch-free.
  - get_ln_params / get_final_ln_params / frame_for_hidden_state :
    weight extraction from a GPT-NeoX/Pythia model. Duck-typed on the
    module attribute structure (works against SimpleNamespace fakes in
    the stubbed test session); torch never imported.

Two Pythia-specific facts encoded here, both easy to get wrong:

1. Parallel residual => TWO LNs per block. `input_layernorm` is what
   attention reads; `post_attention_layernorm` is what the MLP reads —
   despite its name it is applied to the same pre-block input, not to a
   post-attention state (GPT-NeoX use_parallel_residual=True). Since
   attention is the particle-coupling mechanism, which="attn" is the
   default frame.

2. Off-by-one. hidden_states[L] under this project's extraction
   convention (Fix 4: embedding stripped, index 0 = block-0 output) is
   the OUTPUT of block L, and the frame it is about to be read in is
   block L+1's input_layernorm. The last hidden state's reader is
   final_layer_norm — unless the extraction path already applied it
   (core/models.py's standard path records the final entry post-ln_f,
   per status-5.md), in which case the correct frame is the identity.
   frame_for_hidden_state resolves all of this in one place so no call
   site re-derives it.

LN eps convention matches torch.nn.LayerNorm: biased variance, eps added
INSIDE the sqrt. GPT-NeoX uses config.layer_norm_eps (default 1e-5).
"""

from __future__ import annotations

import numpy as np

from core.metrics import _as_numpy, l2_normalize


DEFAULT_LN_EPS: float = 1e-5


# ---------------------------------------------------------------------------
# Pure transform
# ---------------------------------------------------------------------------

def ln_transform(X, gamma=None, beta=None, eps: float = DEFAULT_LN_EPS) -> np.ndarray:
    """
    Row-wise LayerNorm with learned affine, exactly torch.nn.LayerNorm:

        y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta

    var is the biased (ddof=0) variance over the feature dimension.
    gamma/beta default to 1/0 (plain LN). Shapes: X (n, d); gamma, beta
    (d,) or scalar. Returns float64 (n, d).

    Consistency note: with gamma=1, beta=0 this is the forward map whose
    Jacobian p2b_imaginary/layernorm_jacobian.py computes — the
    finite-difference agreement between the two is an oracle test
    (tests/test_core_ln_frame.py), so the two modules cannot drift.
    """
    arr = _as_numpy(X).astype(np.float64, copy=False)
    if arr.ndim == 1:
        arr = arr[None, :]
    mu = arr.mean(axis=-1, keepdims=True)
    var = ((arr - mu) ** 2).mean(axis=-1, keepdims=True)
    xhat = (arr - mu) / np.sqrt(var + eps)
    if gamma is not None:
        xhat = xhat * _as_numpy(gamma).astype(np.float64, copy=False)
    if beta is not None:
        xhat = xhat + _as_numpy(beta).astype(np.float64, copy=False)
    return xhat


def ln_frame_gram(X, gamma=None, beta=None, eps: float = DEFAULT_LN_EPS) -> np.ndarray:
    """
    Pairwise cosine Gram in the LN frame: ln_transform then l2_normalize
    then G = Y Y^T. The composition (rather than raw LN inner products)
    keeps this directly comparable to the existing sphere-frame Gram and
    feeds every existing metric unchanged:

        G_ln = ln_frame_gram(acts, **params)
        lr["ip_mean_ln"]        = pairwise_upper(G_ln).mean()
        lr["ip_mass_near_1_ln"] = mass_near_1(G_ln)
        lr["energies_ln"]       = interaction_energies_batched(G_ln, betas)

    The learned beta is included deliberately: the network reads
    gamma*xhat + beta, bias and all — that shared offset changes pairwise
    angles, and pretending it isn't there would measure a frame nothing
    in the model uses.
    """
    Y = ln_transform(X, gamma=gamma, beta=beta, eps=eps)
    Yn = l2_normalize(Y)
    return Yn @ Yn.T


# ---------------------------------------------------------------------------
# Parameter extraction (GPT-NeoX / Pythia)
# ---------------------------------------------------------------------------

_WHICH_TO_ATTR = {
    "attn": "input_layernorm",
    "mlp": "post_attention_layernorm",
}


def _neox_inner(model):
    """Unwrapped-or-wrapped resolution, same both-forms pattern as
    p2_eigenspectra/weights.py's _gptneox_layers."""
    return getattr(model, "gpt_neox", model)


def _ln_module_params(ln_module, eps_fallback: float = DEFAULT_LN_EPS) -> dict:
    """{gamma, beta, eps} from a LayerNorm-like module. Duck-typed:
    anything with .weight (and optionally .bias, .eps) works, including
    SimpleNamespace fakes under the stubbed test session."""
    gamma = _as_numpy(ln_module.weight).astype(np.float64, copy=False)
    bias = getattr(ln_module, "bias", None)
    beta = (_as_numpy(bias).astype(np.float64, copy=False)
            if bias is not None else None)
    eps = float(getattr(ln_module, "eps", eps_fallback))
    return dict(gamma=gamma, beta=beta, eps=eps)


def get_ln_params(model, layer_idx: int, which: str = "attn") -> dict:
    """
    Learned LN parameters for one block of a GPT-NeoX/Pythia model.

    which="attn" -> input_layernorm (the frame attention reads; default,
                    since attention is the particle coupling)
    which="mlp"  -> post_attention_layernorm (the frame the MLP reads —
                    of the same pre-block input; see module docstring)
    """
    if which not in _WHICH_TO_ATTR:
        raise ValueError(f"get_ln_params: which={which!r}, expected 'attn' or 'mlp'")
    layers = _neox_inner(model).layers
    if not (0 <= layer_idx < len(layers)):
        raise IndexError(
            f"get_ln_params: layer_idx {layer_idx} out of range for "
            f"{len(layers)} blocks"
        )
    return _ln_module_params(getattr(layers[layer_idx], _WHICH_TO_ATTR[which]))


def get_final_ln_params(model) -> dict:
    """final_layer_norm parameters — the unembedding's reading frame
    (this is the frame core/functional_distance.py's inputs live in)."""
    return _ln_module_params(_neox_inner(model).final_layer_norm)


def n_blocks(model) -> int:
    return len(_neox_inner(model).layers)


def frame_for_hidden_state(
    model,
    hidden_layer_idx: int,
    n_hidden_states: int,
    which: str = "attn",
    embedding_stripped: bool = True,
    last_is_post_final_ln: bool = False,
) -> dict:
    """
    Resolve which LN frame hidden_states[hidden_layer_idx] is about to be
    read in — the single place the off-by-one lives.

    embedding_stripped    : True under this project's Fix 4 convention
                            (index 0 = block-0 output). False if index 0
                            is the raw embedding (then index L is block
                            (L-1)'s output and is read by block L).
    last_is_post_final_ln : True when the extraction path already applied
                            final_layer_norm to the last entry
                            (core/models.py standard path per status-5.md)
                            — the correct frame is then the identity, and
                            applying final LN again would be wrong.

    Returns {"frame": "block"|"final"|"identity",
             "block_idx": int|None, "params": dict|None}.
    params is ready to splat into ln_transform/ln_frame_gram; None for
    the identity frame (use the activations as-is).
    """
    if not (0 <= hidden_layer_idx < n_hidden_states):
        raise IndexError(
            f"frame_for_hidden_state: index {hidden_layer_idx} out of range "
            f"for {n_hidden_states} hidden states"
        )
    is_last = hidden_layer_idx == n_hidden_states - 1
    if is_last:
        if last_is_post_final_ln:
            return dict(frame="identity", block_idx=None, params=None)
        return dict(frame="final", block_idx=None,
                    params=get_final_ln_params(model))

    reader_block = hidden_layer_idx + 1 if embedding_stripped else hidden_layer_idx
    nb = n_blocks(model)
    if reader_block >= nb:
        # More hidden states than blocks with no post-final-LN flag —
        # the caller's conventions are inconsistent; refuse to guess.
        raise IndexError(
            f"frame_for_hidden_state: resolved reader block {reader_block} "
            f">= n_blocks {nb}. Check embedding_stripped / "
            f"last_is_post_final_ln against the extraction path used."
        )
    return dict(frame="block", block_idx=reader_block,
                params=get_ln_params(model, reader_block, which=which))
