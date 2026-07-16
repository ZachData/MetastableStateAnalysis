"""
core/metrics.py — Canonical per-layer scalar metrics (transition plan v2,
core infrastructure item 3).

Single implementation of every metric whose definition previously existed
in more than one place, or was at risk of drifting:

  energy          : interaction_energy / interaction_energies_batched
                    (previously only in p1_mstate_tracking/metrics.py)
  effective rank  : effective_rank (raw and normed variants)
                    (previously only in p1_mstate_tracking/metrics.py,
                    one variant used torch.linalg.svdvals, the other
                    scipy.linalg.svdvals — same math, two code paths)
  Fiedler         : fiedler_and_eigengap
                    (previously computed inline inside
                    p1_mstate_tracking/spectral.py's spectral_eigengap_k)
  mass-near-1     : mass_near_1
                    (previously duplicated: analysis.py computed
                    `(ips > 0.9).mean()` inline; causal_tests.py's
                    `_mass_near_1` re-implemented the same fraction at a
                    *different* threshold, 0.95, for cluster-restricted
                    cohesion. Both are the same quantity — "fraction of
                    pairs above a similarity threshold" — parameterized
                    by threshold and an optional population mask. 0.9 is
                    the canonical default: it matches Blog 1's published
                    definition and analysis.py's per-layer field
                    `ip_mass_near_1`. Callers wanting the stricter
                    cluster-cohesion reading pass threshold=0.95 and a
                    mask explicitly; there is no second implementation to
                    drift out of sync with this one.)

Design choice: every function here accepts EITHER a torch.Tensor or a
np.ndarray. torch is never imported at module level — see `_as_numpy`.
This makes the module importable and independently testable (numpy/scipy
only) regardless of whether torch is installed in a given environment,
which is not true of the two source modules this consolidates. Existing
call sites that pass torch.Tensor activations are unaffected.

p1_mstate_tracking/metrics.py becomes a re-export shim over this module
(see that file) so none of its ~15 existing importers need to change.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian


# ---------------------------------------------------------------------------
# torch-optional array coercion
# ---------------------------------------------------------------------------

def _as_numpy(x) -> np.ndarray:
    """
    Accept a torch.Tensor, a np.ndarray, or anything array-like, and return
    a float64 np.ndarray. Duck-typed on .detach()/.cpu()/.numpy() so this
    module never has to import torch to support torch callers.
    """
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def l2_normalize(activations) -> np.ndarray:
    """
    L2-normalize each row onto the unit sphere. torch-optional equivalent
    of core.models.layernorm_to_sphere for callers that only need the
    numpy result (e.g. every function in this module).
    """
    arr = _as_numpy(activations).astype(np.float64, copy=False)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return arr / norms


# ---------------------------------------------------------------------------
# Gram matrix / pairwise inner products
# ---------------------------------------------------------------------------

def gram_matrix(activations) -> np.ndarray:
    """Full n×n pairwise cosine-similarity matrix on S^{d-1}."""
    normed = l2_normalize(activations)
    return normed @ normed.T


def pairwise_upper(G: np.ndarray) -> np.ndarray:
    """Upper-triangle (k=1) values of a pre-computed Gram matrix G."""
    n = G.shape[0]
    idx = np.triu_indices(n, k=1)
    return G[idx]


# Backward-compatible alias — this was p1_mstate_tracking/metrics.py's name.
pairwise_inner_products_from_gram = pairwise_upper


def pairwise_inner_products(activations) -> np.ndarray:
    """Upper-triangle pairwise cosine similarities, computed from raw activations."""
    return pairwise_upper(gram_matrix(activations))


# ---------------------------------------------------------------------------
# Interaction energy (Geshkovski et al.)
# ---------------------------------------------------------------------------

def interaction_energy(activations, beta: float) -> float:
    """E_beta = (1 / 2*beta*n^2) * sum_ij exp(beta * <x_i, x_j>)."""
    G = gram_matrix(activations)
    n = G.shape[0]
    return float(np.exp(beta * G).sum() / (2.0 * beta * n * n))


def interaction_energies_batched(G: np.ndarray, beta_values) -> dict:
    """
    E_beta for every beta in beta_values, vectorised over a pre-computed
    Gram matrix G. Returns {beta: energy_float}.
    """
    n = G.shape[0]
    betas = np.asarray(list(beta_values), dtype=np.float64)
    exp_G = np.exp(betas[:, None, None] * G[None])
    sums = exp_G.sum(axis=(1, 2))
    energies = sums / (2.0 * betas * n * n)
    return {float(b): float(e) for b, e in zip(beta_values, energies)}


# Relative drop threshold for energy-violation detection. Single definition;
# p1_mstate_tracking/metrics.py re-exports this same object (not a copy).
ENERGY_VIOLATION_REL_TOL: float = 1e-3


def energy_violation_severity(energies: list, rel_tol: float = ENERGY_VIOLATION_REL_TOL) -> dict:
    """
    Relative-threshold energy-violation analysis for one beta-series.
    Violation criterion: (E_prev - E_curr) / |E_prev| > rel_tol.
    """
    arr = np.array(energies, dtype=np.float64)
    valid = ~np.isnan(arr)

    if valid.sum() < 2:
        return dict(violation_layers=[], rel_drops=[], sum_severity=0.0,
                    max_severity=0.0, n_violations=0,
                    total_rel_change=float("nan"))

    diffs = np.diff(arr)
    ref = np.maximum(np.abs(arr[:-1]), 1e-12)
    rel_drop = -diffs / ref

    valid_trans = valid[:-1] & valid[1:]
    rel_drop = np.where(valid_trans, rel_drop, 0.0)

    viol_mask = rel_drop > rel_tol
    viol_layers = [i + 1 for i, v in enumerate(viol_mask) if v]
    viol_drops = rel_drop[viol_mask]

    first_valid = arr[valid][0]
    last_valid = arr[valid][-1]
    total_rel = float((last_valid - first_valid) / max(abs(first_valid), 1e-12))

    return dict(
        violation_layers=viol_layers,
        rel_drops=rel_drop.tolist(),
        sum_severity=float(viol_drops.sum()) if len(viol_drops) else 0.0,
        max_severity=float(viol_drops.max()) if len(viol_drops) else 0.0,
        n_violations=int(viol_mask.sum()),
        total_rel_change=total_rel,
    )


# ---------------------------------------------------------------------------
# Effective rank — one function, explicit mode
# ---------------------------------------------------------------------------

def effective_rank(activations, mode: str = "raw") -> float:
    """
    Spectral-entropy effective rank: exp(-sum p_i log p_i), p_i the
    normalized squared singular values.

    mode="raw"    : SVD on raw (unnormalized) activations. Captures both
                    scale and directional collapse. Use for degeneracy
                    gates (matches the old effective_rank_from_raw).
    mode="normed" : SVD on L2-normed activations. Measures directional
                    spread on the sphere only, independent of residual-
                    stream norm growth (matches the old
                    effective_rank_from_normed).

    Both old names are kept below as thin wrappers so existing call sites
    (which encode the contract in the function name) don't need to change.
    """
    if mode == "raw":
        arr = _as_numpy(activations).astype(np.float64, copy=False)
    elif mode == "normed":
        arr = l2_normalize(activations)
    else:
        raise ValueError(f"effective_rank: unknown mode {mode!r}, expected 'raw' or 'normed'")

    sv = np.linalg.svd(arr, compute_uv=False)
    sv2 = sv ** 2
    total = sv2.sum()
    if total < 1e-12:
        return 1.0
    p = np.clip(sv2 / total, 1e-12, None)
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def effective_rank_from_raw(activations) -> float:
    """Backward-compatible alias for effective_rank(activations, mode='raw')."""
    return effective_rank(activations, mode="raw")


def effective_rank_from_normed(normed) -> float:
    """Backward-compatible alias for effective_rank(normed, mode='normed')."""
    return effective_rank(normed, mode="normed")


# ---------------------------------------------------------------------------
# Fiedler value / vector + eigengap cluster-count estimate
# ---------------------------------------------------------------------------

def fiedler_and_eigengap(G: np.ndarray, max_k: int = 15, return_fiedler_vec: bool = False) -> dict:
    """
    Canonical Fiedler / eigengap computation on the normalized Laplacian of
    a Gram matrix. Single implementation — previously computed inline
    inside p1_mstate_tracking/spectral.py's spectral_eigengap_k, which now
    delegates here.

    Parameters
    ----------
    G                  : (n, n) pairwise inner-product matrix
    max_k              : maximum number of eigenvalues to inspect
    return_fiedler_vec : if True, include "fiedler_vec" (the second
                         Laplacian eigenvector) in the result.

    Returns
    -------
    dict with keys: k_eigengap, k_second_gap, second_gap_ratio,
    fiedler_value (lambda_2), eigenvalues, eigengaps, and (optionally)
    fiedler_vec.
    """
    G_pos = np.clip(G, 0, None)
    np.fill_diagonal(G_pos, 1.0)
    L = laplacian(G_pos, normed=True)
    n = G_pos.shape[0]
    k = min(max_k + 1, n - 1)

    if k < 2:
        result = {
            "k_eigengap": 1,
            "k_second_gap": 1,
            "second_gap_ratio": 1.0,
            "fiedler_value": float("nan"),
            "eigenvalues": [],
            "eigengaps": [],
        }
        if return_fiedler_vec:
            result["fiedler_vec"] = None
        return result

    if return_fiedler_vec:
        eigenvalues, eigenvectors = eigh(L, eigvals_only=False, subset_by_index=[0, k - 1])
        eigenvalues = np.real(eigenvalues)
        fiedler_vec = np.real(eigenvectors[:, 1]).tolist()
    else:
        eigenvalues = eigh(L, eigvals_only=True, subset_by_index=[0, k - 1])
        eigenvalues = np.clip(np.real(eigenvalues), 0.0, None)
        fiedler_vec = None

    gaps = np.diff(eigenvalues)
    k_eigengap = int(np.argmax(gaps) + 1)

    if len(gaps) > 1:
        tail_gaps = gaps[1:]
        k_second_gap = int(np.argmax(tail_gaps) + 2)
        sorted_tail = np.sort(tail_gaps)
        second_gap_ratio = float(sorted_tail[-1] / (sorted_tail[-2] + 1e-10) if len(sorted_tail) > 1 else 1.0)
        if second_gap_ratio < 1.1:
            k_second_gap = 1
    else:
        k_second_gap = 1
        second_gap_ratio = 1.0

    result = {
        "k_eigengap": k_eigengap,
        "k_second_gap": k_second_gap,
        "second_gap_ratio": second_gap_ratio,
        "fiedler_value": float(eigenvalues[1]) if len(eigenvalues) > 1 else float("nan"),
        "eigenvalues": eigenvalues.tolist(),
        "eigengaps": gaps.tolist(),
    }
    if return_fiedler_vec:
        result["fiedler_vec"] = fiedler_vec
    return result


# ---------------------------------------------------------------------------
# Mass-near-1 — one function, one threshold policy
# ---------------------------------------------------------------------------

MASS_NEAR_1_DEFAULT_THRESHOLD: float = 0.9


def mass_near_1(G: np.ndarray, threshold: float = MASS_NEAR_1_DEFAULT_THRESHOLD, mask: np.ndarray = None) -> float:
    """
    Fraction of pairwise inner products exceeding `threshold`.

    mask : optional (n,) boolean array. If given, restricts to pairs (i, j)
           both inside the mask (the cluster-cohesion reading causal_tests.py
           needed) — pass threshold=0.95 explicitly for that reading, since
           that was the value in use there. threshold=0.9 with mask=None is
           the population-level reading (Blog 1's published definition,
           analysis.py's ip_mass_near_1).

    Returns 0.0 if fewer than 2 tokens are in scope.
    """
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() < 2:
            return 0.0
        G = G[np.ix_(mask, mask)]
    n = G.shape[0]
    if n < 2:
        return 0.0
    pairs = pairwise_upper(G)
    return float((pairs > threshold).mean())


# ---------------------------------------------------------------------------
# Attention entropy
# ---------------------------------------------------------------------------

def attention_entropy(attn_matrix) -> np.ndarray:
    """Shannon entropy of each attention row, averaged over tokens. (n_heads,) out."""
    attn = _as_numpy(attn_matrix)
    log_attn = np.log(attn + 1e-12)
    entropy_per_token = -(attn * log_attn).sum(axis=-1)
    return entropy_per_token.mean(axis=-1)


# ---------------------------------------------------------------------------
# Nearest-neighbour tracking
# ---------------------------------------------------------------------------

def nearest_neighbor_indices(G: np.ndarray) -> np.ndarray:
    """nn[i] = argmax_{j!=i} G[i, j]."""
    G_masked = G.copy()
    np.fill_diagonal(G_masked, -np.inf)
    return np.argmax(G_masked, axis=1).astype(np.int32)


def nearest_neighbor_stability(activations, prev_activations) -> float:
    """Fraction of tokens whose nearest neighbour is unchanged vs the previous layer."""
    normed_curr = l2_normalize(activations)
    normed_prev = l2_normalize(prev_activations)
    nn_curr = nearest_neighbor_indices(normed_curr @ normed_curr.T)
    nn_prev = nearest_neighbor_indices(normed_prev @ normed_prev.T)
    return float(np.mean(nn_curr == nn_prev))


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (n_tokens, d) L2-normed, internally-centered matrices."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    YtX = Y.T @ X
    numerator = float(np.sum(YtX ** 2))
    XtX_norm = float(np.linalg.norm(X.T @ X, "fro"))
    YtY_norm = float(np.linalg.norm(Y.T @ Y, "fro"))
    denom = XtX_norm * YtY_norm
    if denom < 1e-12:
        return float("nan")
    return float(np.clip(numerator / denom, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Energy-drop pair localization
# ---------------------------------------------------------------------------

def energy_drop_pairs(activations_before, activations_after, beta: float, top_k: int = 10) -> list:
    """Token pairs (i, j, delta) responsible for an energy drop between two layers."""
    normed_before = l2_normalize(activations_before)
    normed_after = l2_normalize(activations_after)
    return _energy_drop_pairs_core(normed_before, normed_after, beta, top_k)


def energy_drop_pairs_from_normed(normed_before: np.ndarray, normed_after: np.ndarray, beta: float, top_k: int = 10) -> list:
    """Same as energy_drop_pairs but accepts pre-normalized ndarrays directly."""
    return _energy_drop_pairs_core(normed_before, normed_after, beta, top_k)


def _energy_drop_pairs_core(normed_before, normed_after, beta, top_k):
    n = normed_before.shape[0]
    if n < 2:
        return []
    norm = 2.0 * beta * n * n

    G_before = normed_before @ normed_before.T
    G_after = normed_after @ normed_after.T
    delta = (np.exp(beta * G_after) - np.exp(beta * G_before)) / norm

    rows, cols = np.triu_indices(n, k=1)
    pair_deltas = delta[rows, cols]

    k = min(top_k, len(pair_deltas))
    if k >= len(pair_deltas):
        worst_idx = np.argsort(pair_deltas)
    else:
        worst_idx = np.argpartition(pair_deltas, k)[:k]
        worst_idx = worst_idx[np.argsort(pair_deltas[worst_idx])]

    return [(int(rows[idx]), int(cols[idx]), float(pair_deltas[idx])) for idx in worst_idx]
