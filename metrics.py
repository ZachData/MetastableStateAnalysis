"""
metrics.py — Core per-layer scalar metrics.

All functions take a (n_tokens, d_model) activation tensor and return a
scalar or small array.  No plotting, no I/O.

Functions
---------
pairwise_inner_products       : upper-triangle ⟨xᵢ, xⱼ⟩ values on S^{d-1}
pairwise_inner_products_from_gram : same, from a pre-computed Gram matrix
gram_matrix                   : full pairwise inner-product matrix
interaction_energy            : E_beta as defined in Geshkovski et al.
interaction_energies_batched  : all beta values in one vectorised pass
effective_rank                : spectral entropy of singular values
effective_rank_from_normed    : same, accepts pre-normalised ndarray
attention_entropy             : per-head Shannon entropy of attention rows
"""

import numpy as np
import torch

from scipy.linalg import svdvals

from models import layernorm_to_sphere


# ---------------------------------------------------------------------------
# Inner products / Gram matrix
# ---------------------------------------------------------------------------

def pairwise_inner_products(activations: torch.Tensor) -> np.ndarray:
    """Return upper-triangle pairwise cosine similarities (L2-normed)."""
    normed = layernorm_to_sphere(activations)
    gram   = (normed @ normed.T).numpy()
    n      = gram.shape[0]
    idx    = np.triu_indices(n, k=1)
    return gram[idx]


def pairwise_inner_products_from_gram(G: np.ndarray) -> np.ndarray:
    """
    Return upper-triangle pairwise cosine similarities from a pre-computed
    Gram matrix.  Use this inside the analysis loop where G is already
    available to avoid recomputing the matrix multiply.
    """
    n   = G.shape[0]
    idx = np.triu_indices(n, k=1)
    return G[idx]


def gram_matrix(activations: torch.Tensor) -> np.ndarray:
    """Full n×n pairwise inner-product matrix on S^{d-1}."""
    normed = layernorm_to_sphere(activations)
    return (normed @ normed.T).numpy()


# ---------------------------------------------------------------------------
# Interaction energies
# ---------------------------------------------------------------------------

def interaction_energy(activations: torch.Tensor, beta: float) -> float:
    """
    E_beta = (1 / 2β n²) Σᵢⱼ exp(β ⟨xᵢ, xⱼ⟩)

    Theory predicts this is monotone increasing along the residual-stream
    trajectory for the idealized gradient-flow dynamics.
    """
    normed = layernorm_to_sphere(activations).numpy()
    G      = normed @ normed.T
    n      = G.shape[0]
    return float(np.exp(beta * G).sum() / (2 * beta * n * n))


def interaction_energies_batched(G: np.ndarray, beta_values: list) -> dict:
    """
    Compute E_beta for every beta in one vectorised pass over a pre-computed
    Gram matrix G.  Avoids re-normalising activations and re-computing G for
    each beta value.

    Parameters
    ----------
    G           : (n, n) pre-computed pairwise inner-product matrix
    beta_values : list of beta floats

    Returns
    -------
    dict  {beta: energy_float}
    """
    n          = G.shape[0]
    betas      = np.asarray(beta_values, dtype=np.float64)   # (B,)
    # broadcast: (B, n, n) — single exp call for all betas
    exp_G      = np.exp(betas[:, None, None] * G[None])      # (B, n, n)
    sums       = exp_G.sum(axis=(1, 2))                       # (B,)
    energies   = sums / (2.0 * betas * n * n)                 # (B,)
    return {float(beta): float(e) for beta, e in zip(beta_values, energies)}


# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------

def effective_rank(activations: torch.Tensor) -> float:
    """
    Effective rank = exp(H), where H is the Shannon entropy of the
    normalised singular value distribution.

    Near 1  → tokens nearly collinear (collapsed).
    Near d  → tokens spread across all dimensions.
    """
    sv      = svdvals(activations.numpy())
    sv      = sv[sv > 1e-10]
    sv_norm = sv / sv.sum()
    entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-12))
    return float(np.exp(entropy))


def effective_rank_from_raw(activations: torch.Tensor) -> float:
    """
    Effective rank from a raw (un-normalized) activation tensor.

    SVD must run on raw activations, not L2-normed ones.  L2 normalization
    sets every token's norm to 1, collapsing the inter-token scale variation
    that the singular values actually measure — so svdvals(normed) gives a
    different (wrong) answer.

    Named _from_raw to make the contract explicit at call sites.
    """
    sv      = svdvals(activations.numpy())
    sv      = sv[sv > 1e-10]
    sv_norm = sv / sv.sum()
    entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-12))
    return float(np.exp(entropy))


# ---------------------------------------------------------------------------
# Attention entropy
# ---------------------------------------------------------------------------

def attention_entropy(attn_matrix: torch.Tensor) -> np.ndarray:
    """
    Shannon entropy of each attention row, averaged over tokens.

    Parameters
    ----------
    attn_matrix : (n_heads, n_tokens, n_tokens)

    Returns
    -------
    (n_heads,) array — mean entropy per head
    """
    attn              = attn_matrix.numpy()
    log_attn          = np.log(attn + 1e-12)
    entropy_per_token = -(attn * log_attn).sum(axis=-1)   # (n_heads, n_tokens)
    return entropy_per_token.mean(axis=-1)                 # (n_heads,)
