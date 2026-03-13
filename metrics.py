"""
metrics.py — Core per-layer scalar metrics.

All functions take a (n_tokens, d_model) activation tensor and return a
scalar or small array.  No plotting, no I/O.

Functions
---------
pairwise_inner_products  : upper-triangle ⟨xᵢ, xⱼ⟩ values on S^{d-1}
gram_matrix              : full pairwise inner-product matrix
interaction_energy       : E_beta as defined in Geshkovski et al.
effective_rank           : spectral entropy of singular values
attention_entropy        : per-head Shannon entropy of attention rows
"""

import numpy as np
import torch

from scipy.linalg import svdvals

from models import layernorm_to_sphere


def pairwise_inner_products(activations: torch.Tensor) -> np.ndarray:
    """Return upper-triangle pairwise cosine similarities (L2-normed)."""
    normed = layernorm_to_sphere(activations)
    gram   = (normed @ normed.T).numpy()
    n      = gram.shape[0]
    idx    = np.triu_indices(n, k=1)
    return gram[idx]


def gram_matrix(activations: torch.Tensor) -> np.ndarray:
    """Full n×n pairwise inner-product matrix on S^{d-1}."""
    normed = layernorm_to_sphere(activations)
    return (normed @ normed.T).numpy()


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
