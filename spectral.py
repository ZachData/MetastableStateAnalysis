"""
spectral.py — Eigengap heuristic applied to the Gram matrix Laplacian.

The Gram matrix of L2-normed token activations is the pairwise inner-product
matrix tracked in Geshkovski et al.  Its spectral structure encodes cluster
count without requiring a distance threshold:
  - k near-zero eigenvalues of L → k well-separated clusters
  - The largest gap between consecutive eigenvalues locates k.

This is threshold-free and directly motivated by the paper's geometry.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian

from config import SPECTRAL_MAX_K


def spectral_eigengap_k(G: np.ndarray, max_k: int = SPECTRAL_MAX_K) -> dict:
    """
    Estimate cluster count from the eigengap heuristic on the Gram matrix.

    Parameters
    ----------
    G     : (n, n) pairwise inner-product matrix (output of gram_matrix)
    max_k : maximum number of eigenvalues to inspect

    Returns
    -------
    dict with keys:
      k_eigengap  int   — estimated cluster count
      eigenvalues list  — first (max_k+1) Laplacian eigenvalues
      eigengaps   list  — consecutive differences of eigenvalues
    """
    G_pos = np.clip(G, 0, None)
    np.fill_diagonal(G_pos, 1.0)
    L          = laplacian(G_pos, normed=True)
    n          = G_pos.shape[0]
    k          = min(max_k + 1, n - 1)
    eigenvalues = np.real(
        eigh(L, eigvals_only=True, subset_by_index=[0, k - 1])
    )
    gaps       = np.diff(eigenvalues)
    k_eigengap = int(np.argmax(gaps) + 1)

    return {
        "k_eigengap":  k_eigengap,
        "eigenvalues": eigenvalues.tolist(),
        "eigengaps":   gaps.tolist(),
    }
