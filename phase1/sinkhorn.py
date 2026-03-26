"""
sinkhorn.py — Sinkhorn-Knopp doubly stochastic normalization + Fiedler analysis.

Motivated by Sander et al. (Sinkformers) and Section 3.3 of Geshkovski et al.
A doubly stochastic attention matrix is the gradient-flow object; the gap
between raw attention and doubly stochastic form measures deviation from
idealized dynamics.

Functions
---------
sinkhorn_normalize         : iterative row/col normalisation (single head)
sinkhorn_normalize_batched : vectorised normalisation across all heads at once
fiedler_value              : λ₂ of the normalised Laplacian
sinkhorn_cluster_count     : eigenvalues near 1 ≈ cluster count
analyze_attention_sinkhorn : per-head summary dict for one attention layer
"""

import numpy as np
import torch

from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian

from core.config import SINKHORN_MAX_ITER, SINKHORN_TOL


def sinkhorn_normalize(
    A: np.ndarray,
    max_iter: int = SINKHORN_MAX_ITER,
    tol: float = SINKHORN_TOL,
) -> np.ndarray:
    """
    Iteratively row- and column-normalise *A* until it is doubly stochastic.

    Convergence is declared when the max elementwise change < *tol*.
    Single-head version — kept for external use and the Fiedler/cluster
    functions that operate on one matrix at a time.
    """
    P = np.clip(A.copy().astype(np.float64), 1e-12, None)
    for _ in range(max_iter):
        P_prev = P.copy()
        P      = P / P.sum(axis=1, keepdims=True)
        P      = P / P.sum(axis=0, keepdims=True)
        if np.abs(P - P_prev).max() < tol:
            break
    return P


def sinkhorn_normalize_batched(
    A: np.ndarray,
    max_iter: int = SINKHORN_MAX_ITER,
    tol: float = SINKHORN_TOL,
) -> np.ndarray:
    """
    Vectorised Sinkhorn-Knopp across all attention heads simultaneously.

    Parameters
    ----------
    A : (n_heads, n_tokens, n_tokens)  raw attention weights

    Returns
    -------
    P : (n_heads, n_tokens, n_tokens)  doubly stochastic matrices

    This replaces the Python for-loop over heads in analyze_attention_sinkhorn,
    reducing n_heads serial passes to a single batched numpy operation.
    Row and column axes are 2 and 1 respectively under the (H, n, n) layout.
    """
    P = np.clip(A.astype(np.float64), 1e-12, None)
    for _ in range(max_iter):
        P_prev = P.copy()
        P     /= P.sum(axis=2, keepdims=True)   # row-normalise all heads
        P     /= P.sum(axis=1, keepdims=True)   # col-normalise all heads
        if np.abs(P - P_prev).max() < tol:
            break
    return P


def fiedler_value(P: np.ndarray) -> float:
    """
    Second-smallest eigenvalue (λ₂) of the normalised Laplacian of P.

    Interpretation:
      λ₂ ≈ 0  → near-disconnected components → strong cluster separation
      λ₂ large → well-connected → tokens mixing freely

    A low Fiedler value at a given layer indicates attention routing
    consistent with a metastable state.
    """
    P_sym       = (P + P.T) / 2
    L           = laplacian(P_sym, normed=True)
    n           = L.shape[0]
    k           = min(3, n - 1)
    eigenvalues = eigh(L, eigvals_only=True, subset_by_index=[0, k - 1])
    return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0


def sinkhorn_cluster_count(P: np.ndarray) -> int:
    """
    Count eigenvalues of the doubly stochastic matrix P that exceed 0.5.

    Eigenvalues near 1 correspond to near-invariant subspaces (clusters).
    """
    eigenvalues = np.real(np.linalg.eigvals(P))
    eigenvalues = np.sort(eigenvalues)[::-1]
    return int((eigenvalues > 0.5).sum())


def analyze_attention_sinkhorn(attn_matrix: torch.Tensor) -> dict:
    """
    Run per-head Sinkhorn analysis for one attention layer.

    Parameters
    ----------
    attn_matrix : (n_heads, n_tokens, n_tokens) float tensor

    Returns
    -------
    dict with keys:
      fiedler_mean               float  — mean λ₂ across heads
      fiedler_per_head           list   — λ₂ for each head
      sinkhorn_cluster_count_mean float — mean cluster count across heads
      sinkhorn_cluster_counts    list   — count per head
      row_col_balance_mean       float  — mean std of raw attention column sums
                                          (0 = already doubly stochastic)
    """
    attn    = attn_matrix.numpy()              # (n_heads, n, n)
    n_heads = attn.shape[0]

    # Vectorised column-sum std — no per-head loop needed
    col_sums        = attn.sum(axis=1)                          # (n_heads, n)
    row_col_balance = np.std(col_sums, axis=1).tolist()         # (n_heads,)

    # All heads normalised in one batched call
    P_all = sinkhorn_normalize_batched(attn)                    # (n_heads, n, n)

    # Fiedler and cluster count still require per-head scipy calls
    fiedler_vals   = [fiedler_value(P_all[h])        for h in range(n_heads)]
    cluster_counts = [sinkhorn_cluster_count(P_all[h]) for h in range(n_heads)]

    return {
        "fiedler_mean":                float(np.mean(fiedler_vals)),
        "fiedler_per_head":            fiedler_vals,
        "sinkhorn_cluster_count_mean": float(np.mean(cluster_counts)),
        "sinkhorn_cluster_counts":     cluster_counts,
        "row_col_balance_mean":        float(np.mean(row_col_balance)),
    }
