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

from core.config import SPECTRAL_MAX_K


def spectral_eigengap_k(G: np.ndarray, max_k: int = SPECTRAL_MAX_K,
                        return_fiedler_vec: bool = False) -> dict:
    """
    Estimate cluster count from the eigengap heuristic on the Gram matrix.

    Parameters
    ----------
    G                  : (n, n) pairwise inner-product matrix (output of gram_matrix)
    max_k              : maximum number of eigenvalues to inspect
    return_fiedler_vec : if True, also return the second Laplacian eigenvector
                         (the Fiedler vector) as "fiedler_vec".  Its sign partition
                         splits tokens into two hemispheres of the dominant bipartition.
                         Always None when k < 2 eigenvalues are available.

    Returns
    -------
    dict with keys:
      k_eigengap       int   — estimated cluster count
      eigenvalues      list  — first (max_k+1) Laplacian eigenvalues
      eigengaps        list  — consecutive differences of eigenvalues
      fiedler_vec      list  — second eigenvector (only if return_fiedler_vec=True,
                               else absent from the dict)
    """
    G_pos = np.clip(G, 0, None)
    np.fill_diagonal(G_pos, 1.0)
    L  = laplacian(G_pos, normed=True)
    n  = G_pos.shape[0]
    k  = min(max_k + 1, n - 1)

    # Need at least 2 eigenvalues to compute any gap.  With n <= 2,
    # k can be < 2 and np.argmax on an empty gaps array would crash.
    if k < 2:
        result = {
            "k_eigengap":       1,
            "k_second_gap":     1,
            "second_gap_ratio": 1.0,
            "eigenvalues":      [],
            "eigengaps":        [],
        }
        if return_fiedler_vec:
            result["fiedler_vec"] = None
        return result

    if return_fiedler_vec and k >= 2:
        eigenvalues, eigenvectors = eigh(
            L, eigvals_only=False, subset_by_index=[0, k - 1]
        )
        eigenvalues  = np.real(eigenvalues)
        fiedler_vec  = np.real(eigenvectors[:, 1]).tolist()
    else:
        eigenvalues = np.real(
            eigh(L, eigvals_only=True, subset_by_index=[0, k - 1])
        )
        fiedler_vec = None

    gaps       = np.diff(eigenvalues)
    k_eigengap = int(np.argmax(gaps) + 1)

    # k_second_gap: dominant gap ignoring the trivial first gap (λ₁=0 → λ₂).
    # λ₁=0 is always the zero mode of any Laplacian, so Δλ₁ is structurally
    # large regardless of clustering. Skipping it surfaces secondary structure:
    #   k_second_gap = 1 means no secondary structure (spectrum decays smoothly)
    #   k_second_gap > 1 means genuine cluster count from non-trivial geometry
    # When the spectrum is fully collapsed (all eigenvalues ≈ identical), tail
    # gaps are pure floating-point noise. Suppress by setting k_second_gap=1
    # when second_gap_ratio < 1.1 — gaps are indistinguishable from each other.
    if len(gaps) > 1:
        tail_gaps        = gaps[1:]
        k_second_gap     = int(np.argmax(tail_gaps) + 2)
        sorted_tail      = np.sort(tail_gaps)
        second_gap_ratio = float(sorted_tail[-1] / (sorted_tail[-2] + 1e-10)
                                 if len(sorted_tail) > 1 else 1.0)
        if second_gap_ratio < 1.1:
            k_second_gap = 1
    else:
        k_second_gap     = 1
        second_gap_ratio = 1.0

    result = {
        "k_eigengap":        k_eigengap,
        "k_second_gap":      k_second_gap,
        "second_gap_ratio":  second_gap_ratio,
        "eigenvalues":       eigenvalues.tolist(),
        "eigengaps":         gaps.tolist(),
    }
    if return_fiedler_vec:
        result["fiedler_vec"] = fiedler_vec
    return result
