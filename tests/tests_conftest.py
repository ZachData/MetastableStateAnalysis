"""
conftest.py — Shared synthetic fixtures for phase-1 pure-computation tests.

No model loading.  All fixtures produce deterministic numpy/torch data that
satisfy known analytical properties, documented inline.

Geometry
--------
All token vectors live on S^{d-1} (unit sphere in R^d).  Fixtures are built
as L2-normalised float32 ndarrays; torch.Tensor wrappers are added where
required by function signatures.

Shape constants used throughout: n_layers=6, n_tokens=40, d=16.
"""

import numpy as np
import torch
import pytest

from tests.constants import N_LAYERS, N_TOKENS, D

_rng = np.random.default_rng(42)   # fixed seed → deterministic across runs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Activation geometry fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def antipodal_normed() -> np.ndarray:
    """
    Two tight antipodal clusters on S^{d-1}.

    Construction
    ------------
    Half the tokens are concentrated near the north pole +e₁; the other half
    near the south pole −e₁.  Small isotropic Gaussian noise (σ=0.05) is
    added before normalisation.

    Analytical properties (noise → 0 limit)
    ----------------------------------------
    Within-cluster inner products ⟨xᵢ, xⱼ⟩  ≈ +1
    Between-cluster inner products            ≈ −1
    Effective rank                            ≈  2  (two dominant directions)
    Interaction energy E_β                    > E_β(uniform)  for any β>0
      (cosh(β)/(2β) vs 1/(2β), since cosh(β) > 1)
    """
    half  = N_TOKENS // 2
    X     = np.zeros((N_TOKENS, D), dtype=np.float32)
    X[:half, 0]  =  1.0
    X[half:, 0]  = -1.0
    noise = _rng.standard_normal((N_TOKENS, D)).astype(np.float32) * 0.05
    return _l2_normalize(X + noise)


@pytest.fixture(scope="session")
def uniform_normed() -> np.ndarray:
    """
    Uniform spread on S^{d-1}: i.i.d. Gaussian vectors, L2-normalised.

    Analytical properties (d → ∞ limit)
    -------------------------------------
    ⟨xᵢ, xⱼ⟩  ≈ 0  for i ≠ j
    Effective rank ≈ d   (spectrum is flat)
    Interaction energy E_β ≈ 1/(2β)  (off-diagonal exp terms average to 1)
    """
    X = _rng.standard_normal((N_TOKENS, D)).astype(np.float32)
    return _l2_normalize(X)


@pytest.fixture(scope="session")
def collapsed_normed() -> np.ndarray:
    """
    Single tight cluster: all tokens concentrated near +e₁.

    Analytical properties (noise → 0 limit)
    ----------------------------------------
    All inner products ≈ +1
    Effective rank ≈ 1
    Interaction energy E_β = exp(β)/(2β)  — highest of the three geometries
      because exp(β) > cosh(β) > 1.
    """
    X     = np.zeros((N_TOKENS, D), dtype=np.float32)
    X[:, 0] = 1.0
    noise = _rng.standard_normal((N_TOKENS, D)).astype(np.float32) * 0.001
    return _l2_normalize(X + noise)


# ---------------------------------------------------------------------------
# Gram matrices (pre-computed once per session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def antipodal_gram(antipodal_normed) -> np.ndarray:
    """(n_tokens, n_tokens) float64 Gram matrix for antipodal activations."""
    return (antipodal_normed.astype(np.float64) @
            antipodal_normed.astype(np.float64).T)


@pytest.fixture(scope="session")
def uniform_gram(uniform_normed) -> np.ndarray:
    return (uniform_normed.astype(np.float64) @
            uniform_normed.astype(np.float64).T)


@pytest.fixture(scope="session")
def collapsed_gram(collapsed_normed) -> np.ndarray:
    return (collapsed_normed.astype(np.float64) @
            collapsed_normed.astype(np.float64).T)


# ---------------------------------------------------------------------------
# Synthetic activation tensors for effective-rank tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rank1_tensor() -> torch.Tensor:
    """
    (N_TOKENS, D) float32 tensor of rank exactly 1.

    All rows are identical (the all-ones vector), so only one singular value
    is non-zero.  effective_rank_from_raw must return 1.0.
    """
    v = torch.ones(N_TOKENS, 1, dtype=torch.float32)
    w = torch.ones(1, D,       dtype=torch.float32)
    return v @ w   # (40, 16), rank 1


@pytest.fixture(scope="session")
def uniform_sv_tensor() -> torch.Tensor:
    """
    (N_TOKENS, D) float32 tensor whose singular values are all equal to 1.

    Construction: draw a random (N_TOKENS, D) Gaussian matrix, compute its
    compact SVD, replace the singular-value diagonal with ones, reconstruct.
    By construction svdvals = [1]*D, so effective_rank = exp(H([1/D]*D)) = D.
    """
    rng_fixed = np.random.default_rng(0)
    A = rng_fixed.standard_normal((N_TOKENS, D)).astype(np.float64)
    U, _, Vh = np.linalg.svd(A, full_matrices=False)  # U: (40,16), Vh: (16,16)
    # Replace singular values with ones: X = U @ I @ Vh
    X = (U @ Vh).astype(np.float32)
    return torch.tensor(X)


# ---------------------------------------------------------------------------
# Attention tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def uniform_attention() -> torch.Tensor:
    """
    (n_heads=4, N_TOKENS, N_TOKENS) doubly stochastic attention:
    every entry = 1/N_TOKENS.

    Row sums = col sums = 1  (doubly stochastic).
    Shannon entropy per row = log(N_TOKENS)  — maximum possible.
    row_col_balance (std of column sums) = 0.
    """
    n_heads = 4
    val = 1.0 / N_TOKENS
    return torch.full((n_heads, N_TOKENS, N_TOKENS), val, dtype=torch.float32)


@pytest.fixture(scope="session")
def identity_attention() -> torch.Tensor:
    """
    (n_heads=4, N_TOKENS, N_TOKENS) attention: each token attends only to itself.

    Row sums = col sums = 1  (doubly stochastic).
    Shannon entropy per row = 0  — minimum possible.
    """
    n_heads = 4
    attn    = torch.zeros(n_heads, N_TOKENS, N_TOKENS, dtype=torch.float32)
    idx     = torch.arange(N_TOKENS)
    attn[:, idx, idx] = 1.0
    return attn


# ---------------------------------------------------------------------------
# Cluster-tracking results dicts
# ---------------------------------------------------------------------------

def _make_results(label_list_per_layer):
    """
    Build the results dict expected by track_clusters from a plain list of
    per-layer label arrays.

    track_clusters reads: results["layers"][i]["clustering"]["hdbscan"]["labels"]
    """
    return {
        "layers": [
            {"clustering": {"hdbscan": {"labels": list(labels)}}}
            for labels in label_list_per_layer
        ]
    }


@pytest.fixture(scope="session")
def stable_tracking_results():
    """
    Six layers with identical cluster assignments (20 tokens in cluster 0,
    20 in cluster 1).

    Expected: no births, no deaths, no merges across any transition.
    """
    labels = [0] * (N_TOKENS // 2) + [1] * (N_TOKENS // 2)
    return _make_results([labels] * N_LAYERS)


@pytest.fixture(scope="session")
def one_merge_tracking_results():
    """
    Layers 0-2: two clusters (0→tokens 0-19, 1→tokens 20-39).
    Layer  3:   single cluster (0→all 40 tokens).  ← merge event here.
    Layers 4-5: single cluster persists.

    At the layer-2 → layer-3 transition, match_layer_pair sees:
      overlap(prev=0, curr=0) = 20/40 = 0.5
      overlap(prev=1, curr=0) = 20/40 = 0.5
    The Hungarian algorithm matches prev-cluster-0 to curr-cluster-0; then
    the merge-detection loop finds prev-cluster-1 also overlaps curr-cluster-0
    and records exactly ONE merge event.

    Expected: summary["total_merges"] == 1.
    """
    two_clusters = [0] * (N_TOKENS // 2) + [1] * (N_TOKENS // 2)
    one_cluster  = [0] * N_TOKENS
    layers = (
        [two_clusters] * 3
        + [one_cluster] * 3
    )
    return _make_results(layers)
