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
nearest_neighbor_indices      : argmax-NN for each token from Gram matrix
nearest_neighbor_stability    : fraction of tokens with unchanged NN vs prev layer
linear_cka                    : linear CKA between consecutive layer activations
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


def energy_drop_pairs(
    activations_before: torch.Tensor,
    activations_after: torch.Tensor,
    beta: float,
    top_k: int = 10,
) -> list:
    """
    Identify token pairs (i, j) responsible for an energy drop between layers.

    The per-pair contribution to E_beta is exp(β⟨xᵢ,xⱼ⟩) / (2β n²).
    A drop at layer L means Σᵢⱼ [exp(β⟨xᵢ,xⱼ⟩_after) - exp(β⟨xᵢ,xⱼ⟩_before)] < 0.
    This function returns the top_k pairs with the most negative contribution delta.

    Parameters
    ----------
    activations_before : (n_tokens, d_model) float tensor — layer L activations
    activations_after  : (n_tokens, d_model) float tensor — layer L+1 activations
    beta               : interaction energy beta parameter
    top_k              : number of most-negative pairs to return

    Returns
    -------
    list of (i, j, delta) tuples sorted by delta ascending (most negative first),
    where delta = [exp(β⟨xᵢ,xⱼ⟩_after) - exp(β⟨xᵢ,xⱼ⟩_before)] / (2β n²)
    """
    normed_before = layernorm_to_sphere(activations_before).numpy()
    normed_after  = layernorm_to_sphere(activations_after).numpy()
    n             = normed_before.shape[0]
    norm          = 2.0 * beta * n * n

    G_before = normed_before @ normed_before.T   # (n, n)
    G_after  = normed_after  @ normed_after.T    # (n, n)

    # Per-pair delta matrix: (exp(β⟨·⟩_after) - exp(β⟨·⟩_before)) / (2β n²)
    delta = (np.exp(beta * G_after) - np.exp(beta * G_before)) / norm  # (n, n)

    # Consider upper triangle only (i < j) to avoid double-counting and self-pairs
    rows, cols = np.triu_indices(n, k=1)
    pair_deltas = delta[rows, cols]

    # Find top_k most-negative pairs
    k = min(top_k, len(pair_deltas))
    # argpartition is O(n) vs O(n log n) full sort — sufficient for selection
    worst_idx = np.argpartition(pair_deltas, k)[:k]
    worst_idx = worst_idx[np.argsort(pair_deltas[worst_idx])]  # sort by value ascending

    return [
        (int(rows[idx]), int(cols[idx]), float(pair_deltas[idx]))
        for idx in worst_idx
    ]


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


# ---------------------------------------------------------------------------
# Nearest-neighbour trajectory tracking
# ---------------------------------------------------------------------------

def nearest_neighbor_indices(G: np.ndarray) -> np.ndarray:
    """
    For each token, return the index of its nearest neighbour by cosine
    similarity, excluding self.

    Parameters
    ----------
    G : (n, n) pre-computed pairwise inner-product (Gram) matrix on S^{d-1}

    Returns
    -------
    (n,) int array  —  nn[i] = argmax_{j≠i} G[i, j]
    """
    G_masked = G.copy()
    np.fill_diagonal(G_masked, -np.inf)
    return np.argmax(G_masked, axis=1).astype(np.int32)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two (n_tokens, d) centered activation matrices.

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Parameters
    ----------
    X, Y : (n_tokens, d) arrays — already L2-normed (from layernorm_to_sphere).
           Both are mean-centered internally.

    Returns
    -------
    float in [0, 1]
      1.0 = representations identical up to rotation
      0.0 = representations orthogonal

    The centering step is critical: without it, a large shared bias token
    (e.g. [CLS]) can inflate similarity regardless of structure.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # ||Y^T X||_F^2 = tr(X^T Y Y^T X) = ||X.T @ Y||_F^2
    YtX       = Y.T @ X                          # (d, d)
    numerator = float(np.sum(YtX ** 2))
    XtX_norm  = float(np.linalg.norm(X.T @ X, "fro"))
    YtY_norm  = float(np.linalg.norm(Y.T @ Y, "fro"))
    denom     = XtX_norm * YtY_norm
    if denom < 1e-12:
        return float("nan")
    return float(np.clip(numerator / denom, 0.0, 1.0))


def nearest_neighbor_stability(
    activations: torch.Tensor,
    prev_activations: torch.Tensor,
) -> float:
    """
    Fraction of tokens whose nearest neighbour (by cosine similarity) did not
    change between *prev_activations* (layer L-1) and *activations* (layer L).

    Returns a scalar in [0, 1].
      1.0 = every token's NN is identical — perfect metastable plateau.
      0.0 = every token's NN changed — tokens still reorganising.

    This is the public spec-compliant API.  The analysis loop uses
    ``nearest_neighbor_indices`` directly on the pre-computed Gram matrix to
    avoid redundant normalisation and matmul operations.
    """
    from models import layernorm_to_sphere  # local import avoids circular dep
    normed_curr = layernorm_to_sphere(activations).numpy()
    normed_prev = layernorm_to_sphere(prev_activations).numpy()
    nn_curr = nearest_neighbor_indices(normed_curr @ normed_curr.T)
    nn_prev = nearest_neighbor_indices(normed_prev @ normed_prev.T)
    return float(np.mean(nn_curr == nn_prev))
