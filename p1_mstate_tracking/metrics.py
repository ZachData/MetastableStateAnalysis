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
energy_violation_severity     : relative-threshold violation analysis (Fix 2)
energy_drop_pairs             : token pairs driving an energy drop
energy_drop_pairs_from_normed : same, accepts pre-normalised ndarrays
effective_rank_from_raw       : spectral entropy of singular values
attention_entropy             : per-head Shannon entropy of attention rows
nearest_neighbor_indices      : argmax-NN for each token from Gram matrix
nearest_neighbor_stability    : fraction of tokens with unchanged NN vs prev layer
linear_cka                    : linear CKA between consecutive layer activations
"""

import numpy as np
import torch

from scipy.linalg import svdvals

from core.models import layernorm_to_sphere


# ---------------------------------------------------------------------------
# Module-level constant — Fix 2
# ---------------------------------------------------------------------------

# Relative drop threshold for energy violation detection.
# A layer is a violation iff (E_prev - E_curr) / |E_prev| > ENERGY_VIOLATION_REL_TOL.
# Imported by analysis.py (drop localization gate) and reporting.py (display).
ENERGY_VIOLATION_REL_TOL: float = 1e-3


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
    n        = G.shape[0]
    betas    = np.asarray(beta_values, dtype=np.float64)   # (B,)
    exp_G    = np.exp(betas[:, None, None] * G[None])      # (B, n, n)
    sums     = exp_G.sum(axis=(1, 2))                       # (B,)
    energies = sums / (2.0 * betas * n * n)                 # (B,)
    return {float(beta): float(e) for beta, e in zip(beta_values, energies)}


# ---------------------------------------------------------------------------
# Energy violation severity — Fix 2
# ---------------------------------------------------------------------------

def energy_violation_severity(
    energies: list,
    rel_tol: float = ENERGY_VIOLATION_REL_TOL,
) -> dict:
    """
    Compute relative-threshold energy violation severity for one β-series.

    Violation criterion: (E_prev - E_curr) / |E_prev| > rel_tol
    i.e. energy dropped by more than rel_tol × |E_prev| in one step.

    This replaces the old absolute -1e-4 / -1e-6 gates, which were scale-blind
    and guaranteed to fire on float32 noise for large β (Fix 2).

    Parameters
    ----------
    energies : list of float — E_β values, one per layer (NaN ok, skipped)
    rel_tol  : relative drop threshold (default ENERGY_VIOLATION_REL_TOL = 1e-3)

    Returns
    -------
    dict with keys:
        violation_layers   : list[int]  — 1-indexed layers where violation fired
        rel_drops          : list[float] — relative drop at each transition
                             (positive = energy fell; 0.0 for non-violations)
        sum_severity       : float — sum of relative drops at violation layers
        max_severity       : float — worst single relative drop
        n_violations       : int
        total_rel_change   : float — (E_last - E_first) / |E_first|, signed
    """
    arr   = np.array(energies, dtype=np.float64)
    valid = ~np.isnan(arr)

    if valid.sum() < 2:
        return dict(violation_layers=[], rel_drops=[], sum_severity=0.0,
                    max_severity=0.0, n_violations=0,
                    total_rel_change=float("nan"))

    diffs = np.diff(arr)
    # Reference: |E| at the preceding layer; floor at 1e-12 to avoid /0
    ref      = np.maximum(np.abs(arr[:-1]), 1e-12)
    rel_drop = -diffs / ref  # positive means energy fell

    # Mask out transitions where either endpoint is NaN
    valid_trans = valid[:-1] & valid[1:]
    rel_drop    = np.where(valid_trans, rel_drop, 0.0)

    viol_mask  = rel_drop > rel_tol
    viol_layers = [i + 1 for i, v in enumerate(viol_mask) if v]
    viol_drops  = rel_drop[viol_mask]

    first_valid = arr[valid][0]
    last_valid  = arr[valid][-1]
    total_rel   = float((last_valid - first_valid) / max(abs(first_valid), 1e-12))

    return dict(
        violation_layers = viol_layers,
        rel_drops        = rel_drop.tolist(),
        sum_severity     = float(viol_drops.sum()) if len(viol_drops) else 0.0,
        max_severity     = float(viol_drops.max()) if len(viol_drops) else 0.0,
        n_violations     = int(viol_mask.sum()),
        total_rel_change = total_rel,
    )


# ---------------------------------------------------------------------------
# Energy drop pair localization
# ---------------------------------------------------------------------------

def energy_drop_pairs(
    activations_before: torch.Tensor,
    activations_after: torch.Tensor,
    beta: float,
    top_k: int = 10,
) -> list:
    """
    Identify token pairs (i, j) responsible for an energy drop between layers.

    Accepts raw (un-normalized) tensors.  If the caller already holds
    L2-normed ndarrays, use ``energy_drop_pairs_from_normed`` to skip the
    redundant normalization.

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
    return _energy_drop_pairs_core(normed_before, normed_after, beta, top_k)


def energy_drop_pairs_from_normed(
    normed_before: np.ndarray,
    normed_after: np.ndarray,
    beta: float,
    top_k: int = 10,
) -> list:
    """
    Identify token pairs responsible for an energy drop between layers.

    Accepts pre-normalized (L2-normed) ndarrays directly — no redundant
    normalization.  Use this inside the analysis loop where ``normed`` is
    already computed once per layer.

    Parameters
    ----------
    normed_before : (n_tokens, d_model) float32 ndarray — L2-normed layer L
    normed_after  : (n_tokens, d_model) float32 ndarray — L2-normed layer L+1
    beta          : interaction energy beta parameter
    top_k         : number of most-negative pairs to return

    Returns
    -------
    list of (i, j, delta) tuples sorted by delta ascending (most negative first)
    """
    return _energy_drop_pairs_core(normed_before, normed_after, beta, top_k)


def _energy_drop_pairs_core(
    normed_before: np.ndarray,
    normed_after: np.ndarray,
    beta: float,
    top_k: int,
) -> list:
    """
    Shared implementation for energy_drop_pairs and energy_drop_pairs_from_normed.
    """
    n = normed_before.shape[0]
    if n < 2:
        return []
    norm = 2.0 * beta * n * n

    G_before = normed_before @ normed_before.T   # (n, n)
    G_after  = normed_after  @ normed_after.T    # (n, n)

    delta = (np.exp(beta * G_after) - np.exp(beta * G_before)) / norm

    rows, cols   = np.triu_indices(n, k=1)
    pair_deltas  = delta[rows, cols]

    k = min(top_k, len(pair_deltas))
    if k >= len(pair_deltas):
        worst_idx = np.argsort(pair_deltas)
    else:
        worst_idx = np.argpartition(pair_deltas, k)[:k]
        worst_idx = worst_idx[np.argsort(pair_deltas[worst_idx])]

    return [
        (int(rows[idx]), int(cols[idx]), float(pair_deltas[idx]))
        for idx in worst_idx
    ]


# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------

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


def effective_rank_from_normed(normed: np.ndarray) -> float:
    """
    Effective rank of the L2-normed activation matrix.  Fix 8.

    Measures directional spread on the unit sphere, independent of
    residual-stream norm growth across layers.  Because every row already
    has unit L2 norm the singular values reflect only how many directions
    the token cloud spans, not scale differences between tokens.

    Contrast with effective_rank_from_raw, which captures both scale and
    directional collapse.  Raw rank is the right choice for the degeneracy
    gate (more conservative); normed rank is the right choice for the
    "spread on the sphere" characterisation stored as effective_rank_normed.

    Named _from_normed to make the contract explicit at call sites.

    Parameters
    ----------
    normed : (n_tokens, d_model) float32 ndarray, each row already L2-normed

    Returns
    -------
    effective_rank : float ≥ 1.0
    """
    X   = torch.from_numpy(np.asarray(normed, dtype=np.float32))
    sv2 = torch.linalg.svdvals(X) ** 2
    tot = sv2.sum()
    if tot < 1e-12:
        return 1.0
    p = torch.clamp(sv2 / tot, min=1e-12)
    return float(torch.exp(-(p * torch.log(p)).sum()).item())


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
    normed_curr = layernorm_to_sphere(activations).numpy()
    normed_prev = layernorm_to_sphere(prev_activations).numpy()
    nn_curr     = nearest_neighbor_indices(normed_curr @ normed_curr.T)
    nn_prev     = nearest_neighbor_indices(normed_prev @ normed_prev.T)
    return float(np.mean(nn_curr == nn_prev))


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------

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
    YtX       = Y.T @ X
    numerator = float(np.sum(YtX ** 2))
    XtX_norm  = float(np.linalg.norm(X.T @ X, "fro"))
    YtY_norm  = float(np.linalg.norm(Y.T @ Y, "fro"))
    denom     = XtX_norm * YtY_norm
    if denom < 1e-12:
        return float("nan")
    return float(np.clip(numerator / denom, 0.0, 1.0))