"""
core/functional_distance.py — Distance as the readout perceives it
(frames item 4).

Two particles are functionally identical iff the LM head decodes them to
the same next-token distribution — regardless of where they sit
geometrically, at what norm, in which frame. This module turns the
per-layer decoded distributions (p5b_manifold/logit_cache.py's arrays,
or tuned_lens_cluster.py's frozen-head/tuned-lens outputs) into a
pairwise divergence matrix, clusters on it, and scores agreement between
this functional labeling and the geometric ones. It is the arbiter the
frame-comparison table calls when the L2-sphere view and the LN-frame
view (core/ln_frame.py) disagree about whether a merge "really"
happened.

Consumes cached log-probs / probs — never runs a forward pass. Pure
numpy except for the optional HDBSCAN import inside functional_clusters
(lazy, matching the project's heavy-dep convention).

The one implementation fact that makes this cheap: the full pairwise KL
matrix over the whole vocabulary is a single matmul, no pair loop. With
P = probs (n, V) and L = log-probs (n, V):

    KL(p_i || p_j) = sum_v P[i,v] * (L[i,v] - L[j,v])
                   = rowsum(P * L)[i]  -  (P @ L.T)[i, j]

n=512 tokens x V=50k is ~13 GFLOPs and ~100 MB — trivial; a chunk
parameter bounds memory for anything larger.

Symmetrization is (KL + KL.T)/2. Documented limitation: symmetrized KL
is NOT a metric (triangle inequality fails); it is used here as a
clustering affinity and comparison score, which is all the arbiter role
needs. True JSD would need a per-pair mixture (n^2 x V memory) for
little gain at that role.
"""

from __future__ import annotations

import numpy as np

from core.metrics import _as_numpy


# ---------------------------------------------------------------------------
# Log-prob preparation
# ---------------------------------------------------------------------------

def logprobs_from_logits(logits) -> np.ndarray:
    """Numerically stable row-wise log-softmax. (n, V) float64 out."""
    arr = _as_numpy(logits).astype(np.float64, copy=False)
    shifted = arr - arr.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def logprobs_from_probs(probs, eps: float = 1e-12) -> np.ndarray:
    """
    log of already-softmaxed rows (logit_cache.py stores probabilities).
    Rows are renormalized first so float32-storage drift doesn't leak
    into the KL identity; eps floors zeros before the log.
    """
    p = _as_numpy(probs).astype(np.float64, copy=False)
    p = p / p.sum(axis=-1, keepdims=True)
    return np.log(np.maximum(p, eps))


# ---------------------------------------------------------------------------
# Pairwise KL
# ---------------------------------------------------------------------------

def kl_matrix(logprobs, chunk: int | None = None) -> np.ndarray:
    """
    Full pairwise KL divergence matrix from row log-probs.

        K[i, j] = KL(p_i || p_j),  p = exp(logprobs)

    via the matmul identity in the module docstring. Diagonal is exactly
    zeroed; tiny negative values from float error are clipped to 0.

    chunk : rows per block for the P @ L.T product. None -> one matmul
            (fine for n<=1024, V<=64k). Set to e.g. 128 to bound peak
            memory at chunk*V floats for long sequences.
    """
    L = _as_numpy(logprobs).astype(np.float64, copy=False)
    if L.ndim != 2:
        raise ValueError(f"kl_matrix: expected (n, V) log-probs, got shape {L.shape}")
    P = np.exp(L)
    self_term = (P * L).sum(axis=-1)          # (n,)  = -H(p_i)

    n = L.shape[0]
    if chunk is None or chunk >= n:
        cross = P @ L.T                        # (n, n)
    else:
        cross = np.empty((n, n), dtype=np.float64)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            cross[s:e] = P[s:e] @ L.T

    K = self_term[:, None] - cross
    np.fill_diagonal(K, 0.0)
    return np.maximum(K, 0.0)


def sym_kl(K: np.ndarray) -> np.ndarray:
    """(K + K.T) / 2 — symmetric clustering affinity (not a metric)."""
    return 0.5 * (K + K.T)


def kl_matrix_from_probs(probs, chunk: int | None = None,
                         eps: float = 1e-12) -> np.ndarray:
    """Convenience: probability rows (logit_cache format) -> KL matrix."""
    return kl_matrix(logprobs_from_probs(probs, eps=eps), chunk=chunk)


# ---------------------------------------------------------------------------
# Functional clustering
# ---------------------------------------------------------------------------

def functional_clusters(D: np.ndarray, min_cluster_size: int = 3) -> np.ndarray:
    """
    HDBSCAN on a precomputed symmetric divergence matrix (sym_kl output).
    Returns (n,) integer labels, -1 = noise — same convention as the
    geometric pipeline, so labels feed frame_agreement directly.

    Lazy import: sklearn.cluster.HDBSCAN first (sklearn >= 1.3), the
    standalone hdbscan package as fallback — whichever the environment
    running the geometric clustering already has.
    """
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"functional_clusters: expected square matrix, got {D.shape}")
    if not np.allclose(D, D.T, atol=1e-8):
        raise ValueError(
            "functional_clusters: matrix is not symmetric — pass sym_kl(K), "
            "not the raw kl_matrix"
        )
    n = D.shape[0]
    if n < max(min_cluster_size, 2):
        return np.full(n, -1, dtype=np.int64)

    D = D.copy()
    np.fill_diagonal(D, 0.0)
    try:
        from sklearn.cluster import HDBSCAN as _HDBSCAN          # sklearn >= 1.3
        clusterer = _HDBSCAN(min_cluster_size=min_cluster_size,
                             metric="precomputed")
    except ImportError:
        import hdbscan as _hdbscan                                # standalone pkg
        clusterer = _hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                     metric="precomputed")
    return np.asarray(clusterer.fit_predict(D), dtype=np.int64)


# ---------------------------------------------------------------------------
# Agreement between labelings (pure-numpy ARI)
# ---------------------------------------------------------------------------

def adjusted_rand_index(labels_a, labels_b, ignore_noise: bool = False) -> float:
    """
    Adjusted Rand Index between two labelings. Pure numpy (no sklearn
    dependency at import time, per core's torch-free/lazy convention);
    verified against hand-computed contingency values in tests.

    ignore_noise : if True, drop every point labeled -1 in EITHER
                   labeling before scoring. Default False treats -1 as
                   its own class — noise-vs-cluster disagreement then
                   counts as disagreement, which is usually what the
                   arbiter question wants.

    Returns nan when fewer than 2 points remain in scope. Returns 1.0
    for the degenerate all-one-class/all-noise agreement case (both
    partitions trivial and identical).
    """
    a = _as_numpy(labels_a).astype(np.int64, copy=False).ravel()
    b = _as_numpy(labels_b).astype(np.int64, copy=False).ravel()
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"adjusted_rand_index: {a.shape[0]} vs {b.shape[0]} labels"
        )
    if ignore_noise:
        keep = (a >= 0) & (b >= 0)
        a, b = a[keep], b[keep]
    n = a.shape[0]
    if n < 2:
        return float("nan")

    # Contingency table via joint codes
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    n_a, n_b = ai.max() + 1, bi.max() + 1
    cont = np.zeros((n_a, n_b), dtype=np.int64)
    np.add.at(cont, (ai, bi), 1)

    def _comb2(x):
        return x * (x - 1) / 2.0

    sum_cells = _comb2(cont.astype(np.float64)).sum()
    sum_a = _comb2(cont.sum(axis=1).astype(np.float64)).sum()
    sum_b = _comb2(cont.sum(axis=0).astype(np.float64)).sum()
    total = _comb2(float(n))

    expected = sum_a * sum_b / total if total > 0 else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        # Both partitions trivial (single class) — identical by construction.
        return 1.0
    return float((sum_cells - expected) / denom)


def frame_agreement(labelings: dict, ignore_noise: bool = False) -> dict:
    """
    Pairwise ARI over named labelings, e.g.
        frame_agreement({"sphere": lab_l2, "ln": lab_ln, "functional": lab_fn})
    -> {"sphere|ln": ari, "sphere|functional": ari, "ln|functional": ari}

    This is the per-layer row of the three-frame comparison table: where
    all three agree, the clustering claim is frame-robust; where the
    functional labeling breaks from both geometric ones, the geometry is
    seeing structure the readout does not.
    """
    names = sorted(labelings)
    out = {}
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            out[f"{na}|{nb}"] = adjusted_rand_index(
                labelings[na], labelings[nb], ignore_noise=ignore_noise
            )
    return out
