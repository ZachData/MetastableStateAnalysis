"""
p1d_cluster_ensemble/attention_graph.py — tokens grouped by who attends to
whom, and three nulls to read the groups against.

Item (3) of 1d (`lit-1d.md` §2.1; `status-1d.md` "Attention communities").
Distance-based families group tokens by where they *are*; this groups them
by how they *interact*, which is the theory's own coupling and the one
family with no analyst-chosen frame.

The graph, per (run, layer, window ``w``):

1. Each layer's attention, head-averaged, with the sink (position 0: Pythia
   has no BOS, `attention-10.md` §2.1) dropped as a node and every row
   renormalised over what is left.
2. A window of ``w`` layers is the rollout ``prod (I/2 + M_l/2)`` (Abnar &
   Zuidema 2020; the ``I/2`` is the residual). ``w`` is Markov time on the
   layer chain.
3. Made undirected two ways, which answer different questions:
   ``mutual`` ``(R + R^T)/2`` (i and j attend to each other) and ``coattn``
   ``R R^T`` (i and j attend to the same tokens). Directed Markov stability
   is degenerate on a causal chain (`lit-1d.md` §2.1), so it is not used.
4. Communities: Leiden on weighted modularity, resolution 1 (igraph).

The nulls (the user's choice, 2026-09-26: all of them):

- **A** (`attention_null.py`): Gaussian draws with the tokens' covariance
  (`gaussian_null.py`'s null) read by the model's own LN1, QKV and rotary
  (`neox_block.py`). Same map, structureless input.
- **B** ``diagonal_shuffle``: each offset diagonal of the real matrix
  permuted across query rows, then rows renormalised. Keeps the attention
  mass at every offset (recency, positional heads) exactly; destroys which
  content it lands on.
- **C** ``kernel_attention``: the theory's idealised head,
  ``softmax(beta_h <u_i, u_j>)`` on the unit LN1 frame, causal, with each
  head's ``beta_h`` fitted to the real attention (`core/beta_eff.py`, row
  fixed effects, offset control, **no** ``attn_scale``). Fitting and applying
  in the same frame makes β's unit convention irrelevant here (STATE
  Blocked 9): whatever the slope is, it reproduces the fit's logits.

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .constants import SUBSTANTIAL_CLUSTER_SIZE

SINK = 0
#: PLACED. Markov times on the layer chain; the user's "interaction over 2-3 layers".
WINDOWS = (1, 2, 3)
SYMMETRISATIONS = ("mutual", "coattn")
#: PLACED. "Local" for the position statistics: #106 found 30 % of nearest
#: neighbours within 3 positions.
LOCAL = 3

#: Which tail is "more community structure than the null". The rest are
#: descriptive and read in both tails.
STRONGER = {"Q": "higher", "lam2": "lower"}


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------

def non_sink(n: int) -> np.ndarray:
    return np.arange(1, int(n))


def drop_sink(M: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """
    ``M`` (..., n, n) restricted to ``keep`` (which excludes the sink), rows
    renormalised. A row with no mass left keeps its self-loop only.
    """
    keep = np.asarray(keep, dtype=int)
    if SINK in set(keep.tolist()):
        raise ValueError("keep must exclude the sink (position 0)")
    S = np.asarray(M, dtype=np.float64)[..., keep[:, None], keep[None, :]]
    rs = S.sum(axis=-1, keepdims=True)
    eye = np.broadcast_to(np.eye(keep.size), S.shape)
    return np.where(rs > 0, S / np.where(rs > 0, rs, 1.0), eye)


def head_mean(A_layer: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """(heads, n, n) -> the sink-dropped, head-averaged (m, m) chain."""
    return drop_sink(A_layer, keep).mean(axis=0)


def rollout(mats: Sequence[np.ndarray]) -> np.ndarray:
    """``prod (I/2 + M/2)``, first layer rightmost: ``R[i, j]`` is flow from j (bottom) to i (top)."""
    m = mats[0].shape[0]
    R = np.eye(m)
    for M in mats:
        R = (0.5 * np.eye(m) + 0.5 * M) @ R
    return R


def symmetrise(R: np.ndarray, how: str) -> np.ndarray:
    if how == "mutual":
        S = 0.5 * (R + R.T)
    elif how == "coattn":
        S = R @ R.T
    else:
        raise ValueError(f"unknown symmetrisation {how!r}; use one of {SYMMETRISATIONS}")
    S = np.array(S, dtype=np.float64)
    np.fill_diagonal(S, 0.0)
    return S


def communities(S: np.ndarray, seed: int = 0) -> np.ndarray:
    """Leiden (weighted modularity, resolution 1, run to convergence) labels."""
    import igraph as ig
    random.seed(int(seed))
    ig.set_random_number_generator(random)
    g = ig.Graph.Weighted_Adjacency(np.asarray(S), mode="undirected", attr="weight",
                                    loops=False)
    part = g.community_leiden(objective_function="modularity", weights="weight",
                              n_iterations=-1)
    return np.asarray(part.membership, dtype=int)


def modularity(S: np.ndarray, labels: np.ndarray) -> float:
    """Newman's weighted modularity, computed directly (not the optimiser's own score)."""
    S = np.asarray(S, dtype=np.float64)
    k = S.sum(axis=1)
    two_m = k.sum()
    if two_m <= 0:
        return float("nan")
    same = labels[:, None] == labels[None, :]
    return float(((S - np.outer(k, k) / two_m) * same).sum() / two_m)


def lambda2(S: np.ndarray) -> float:
    """Second-smallest eigenvalue of the normalised Laplacian (small = a strong bottleneck)."""
    S = np.asarray(S, dtype=np.float64)
    d = S.sum(axis=1)
    inv = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    L = np.eye(S.shape[0]) - inv[:, None] * S * inv[None, :]
    ev = np.linalg.eigvalsh(0.5 * (L + L.T))
    return float(ev[1]) if ev.size > 1 else float("nan")


def graph_stats(S: np.ndarray, labels: np.ndarray, positions: np.ndarray) -> Dict[str, float]:
    """
    ``Q`` modularity; ``k`` / ``k4`` communities (all / of >= 4 tokens);
    ``lam2``; ``contig`` share of position-adjacent kept tokens in one
    community; ``local`` share of edge weight within ``LOCAL`` positions.
    """
    pos = np.asarray(positions)
    sizes = np.bincount(labels)
    near = np.abs(pos[:, None] - pos[None, :]) <= LOCAL
    tot = S.sum()
    return {"Q": modularity(S, labels),
            "k": float(sizes.size),
            "k4": float(np.sum(sizes >= SUBSTANTIAL_CLUSTER_SIZE)),
            "lam2": lambda2(S),
            "contig": float(np.mean(labels[1:] == labels[:-1])) if labels.size > 1 else float("nan"),
            "local": float(S[near].sum() / tot) if tot > 0 else float("nan")}


# ---------------------------------------------------------------------------
# Agreement with the distance families, per frame
# ---------------------------------------------------------------------------

def frame_agreement(labels: np.ndarray, frames: Dict[str, np.ndarray],
                    seed: int = 0) -> Dict[str, float]:
    """
    ARI between the attention communities and k-means at the same ``k`` in
    each frame (unit rows, `gaussian_null.frame_vectors`). Matched ``k``
    keeps the number of groups out of the comparison; nan when ``k < 2``.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    k = int(np.unique(labels).size)
    out = {}
    for name, Z in frames.items():
        if k < 2 or k >= Z.shape[0]:
            out[f"ari_{name}"] = float("nan")
            continue
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(Z)
        out[f"ari_{name}"] = float(adjusted_rand_score(labels, km.labels_))
    return out


# ---------------------------------------------------------------------------
# Null B: offset-preserving diagonal shuffle
# ---------------------------------------------------------------------------

def diagonal_shuffle(M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Permute each causal diagonal across its rows, then renormalise rows.

    What is permuted is attention *relative to uniform*, ``M[i, j] * (i + 1)``
    (row ``i`` of a causal chain has ``i + 1`` keys), so the row's own scale
    stays with the row. Permuting raw weights moves an early row's large
    ``1/(i+1)`` onto late rows and manufactures strong random edges: on
    step 0's near-uniform attention it scored *more* modular than the real
    matrix (smoke run, 2026-09-26). Uniform attention is a fixed point here.
    The multiset of relative weights at every offset is kept; which key a
    query's weight lands on, beyond its offset, is random.
    """
    M = np.asarray(M, dtype=np.float64)
    m = M.shape[0]
    scale = np.arange(1, m + 1, dtype=np.float64)
    E = M * scale[:, None]
    out = np.zeros_like(M)
    for d in range(m):
        rows = np.arange(d, m)
        vals = E[rows, rows - d]
        out[rows, rows - d] = vals[rng.permutation(vals.size)]
    out = out / scale[:, None]
    rs = out.sum(axis=1, keepdims=True)
    return np.where(rs > 0, out / np.where(rs > 0, rs, 1.0), np.eye(m))


# ---------------------------------------------------------------------------
# Null C: the idealised head
# ---------------------------------------------------------------------------

def unit_ln_rows(X: np.ndarray, gamma: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    """The rows the head reads, on the unit sphere: ``LN1(x) / |LN1(x)|``."""
    from core.ln_frame import ln_transform
    Y = ln_transform(np.asarray(X, dtype=np.float64), gamma=gamma, beta=bias, eps=eps)
    return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)


def fit_betas(A_layer: np.ndarray, U: np.ndarray, keep: np.ndarray,
              positions: np.ndarray) -> List[Dict]:
    """
    Each head's β on the unit LN1 frame, over the non-sink rows ``keep``.

    ``A_layer`` (heads, n, n) and ``U`` (n, d) share one row index;
    ``positions`` maps it to sequence positions, which set the offset
    regressor (a deduped index is not a position). `core/beta_eff.py`'s row
    fixed effects absorb the sink's share of each row's normaliser, so the
    full-softmax probabilities are used as stored.
    """
    from core.beta_eff import estimate_beta_from_gram
    G = U @ U.T
    pos = np.asarray(positions, dtype=np.float64)
    offsets = pos[None, :] - pos[:, None]
    out = []
    for h in range(A_layer.shape[0]):
        r = estimate_beta_from_gram(A_layer[h], G, keep, offsets=offsets)
        out.append({"beta": r["beta"], "r2": r["r2"], "offset_coeff": r["offset_coeff"],
                    "n_pairs": r["n_pairs"], "note": r["note"]})
    return out


def kernel_attention(U: np.ndarray, betas: Sequence[float]) -> np.ndarray:
    """
    ``softmax_j(beta_h <u_i, u_j>)`` over ``j <= i``, per head: (heads, m, m).

    ``U`` is the kept tokens only, in position order (no sink: the idealised
    model has none). A head with nan β contributes a uniform causal row.
    """
    U = np.asarray(U, dtype=np.float64)
    m = U.shape[0]
    G = U @ U.T
    mask = np.triu(np.ones((m, m), dtype=bool), 1)
    out = np.empty((len(betas), m, m))
    for h, b in enumerate(betas):
        logits = (0.0 if not np.isfinite(b) else float(b)) * G
        logits = np.where(mask, -np.inf, logits)
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        out[h] = e / e.sum(axis=1, keepdims=True)
    return out


# ---------------------------------------------------------------------------
# One graph set: every window and symmetrisation from a list of layer chains
# ---------------------------------------------------------------------------

def graph_set(mats: Sequence[np.ndarray], positions: np.ndarray,
              frames: Optional[Dict[str, np.ndarray]], windows: Iterable[int] = WINDOWS,
              seed: int = 0, keep_labels: bool = False) -> Dict[str, Dict]:
    """
    Statistics for each ``w{w}_{sym}`` from the per-layer chains ``mats``
    (first = layer l). Windows longer than ``len(mats)`` are skipped.
    """
    out = {}
    for w in windows:
        if w > len(mats):
            continue
        R = rollout(mats[:w])
        for sym in SYMMETRISATIONS:
            S = symmetrise(R, sym)
            lab = communities(S, seed)
            st = graph_stats(S, lab, positions)
            if frames is not None:
                st.update(frame_agreement(lab, frames, seed))
            if keep_labels:
                st["labels"] = lab.tolist()
            out[f"w{w}_{sym}"] = st
    return out
