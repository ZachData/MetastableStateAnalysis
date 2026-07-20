"""
core/polar.py — Polar bookkeeping for off-sphere particles (frames item 1).

The residual stream is not confined to S^{d-1}: a particle's state is
exactly the pair (norm, direction), x = r * x_hat. The existing pipeline
keeps x_hat (via l2_normalize -> gram_matrix) and silently discards r.
This module refuses to discard r, and quantifies per layer what the
sphere projection is throwing away.

Three concerns, all pure numpy:

  particle_norms / polar_decompose
      the (r, x_hat) coordinates themselves — lossless bookkeeping.

  norm_stats / cluster_norm_profile
      distribution of r across particles (attention-sink outliers show
      up here), and its coupling to cluster structure (do high-norm
      particles sit inside clusters or outside them?).

  sphere_gap
      the per-layer diagnostic for "how much does the sphere assumption
      distort this layer": agreement between raw-inner-product structure
      and cosine structure. Small gap  -> the paper's S^{d-1} framework
      applies nearly as-is at this layer; large gap -> this layer is
      where the toy model and the real model part ways.

Deliberately NOT here: interaction energies on raw (un-normalized)
activations. E_beta = sum exp(beta <x_i, x_j>) is only meaningful on the
sphere — off it, exp(beta * r_i * r_j * cos) overflows as norms grow and
the theory (Prop 3.4, monotonicity) is undefined anyway. Energies stay a
sphere-frame quantity; this module measures what the projection costs,
it does not redefine the energy.

Conventions match core/metrics.py: torch-optional inputs (anything
_as_numpy can coerce), float64 internally, plain floats/lists in returned
dicts so results are JSON-serializable as-is.
"""

from __future__ import annotations

import numpy as np

from core.metrics import _as_numpy, l2_normalize, gram_matrix, pairwise_upper


# ---------------------------------------------------------------------------
# Polar coordinates
# ---------------------------------------------------------------------------

def particle_norms(activations) -> np.ndarray:
    """Per-particle L2 norms r_i. (n,) float64."""
    arr = _as_numpy(activations).astype(np.float64, copy=False)
    return np.linalg.norm(arr, axis=-1)


def polar_decompose(activations) -> tuple[np.ndarray, np.ndarray]:
    """
    Lossless polar coordinates: (norms, directions) with
    activations == norms[:, None] * directions (up to the l2_normalize
    zero-norm guard). norms is (n,), directions is (n, d) on S^{d-1}.
    """
    return particle_norms(activations), l2_normalize(activations)


def raw_gram(activations) -> np.ndarray:
    """
    Un-normalized Gram matrix X X^T. Exact identity with the sphere view:
        raw_gram[i, j] == r_i * r_j * gram_matrix[i, j].
    Provided so callers never rebuild this with a different dtype/route.
    """
    arr = _as_numpy(activations).astype(np.float64, copy=False)
    return arr @ arr.T


# ---------------------------------------------------------------------------
# Norm distribution
# ---------------------------------------------------------------------------

def norm_stats(norms, top_k: int = 5) -> dict:
    """
    Summary of the per-particle norm distribution at one layer.

    max_over_median is the attention-sink indicator: a handful of
    positions (often BOS/delimiters) carrying norms far above the bulk
    shows up as max_over_median >> 1 while mean/std stay unremarkable.
    log_std = std(log r) is the scale-free dispersion used by
    sphere_gap; 0 iff all norms are equal (the only case where the
    sphere projection is exactly lossless up to a global scale).

    top_outlier_indices: token positions of the top_k largest norms,
    descending — join these against the token list to identify sinks.
    """
    r = _as_numpy(norms).astype(np.float64, copy=False).ravel()
    n = r.shape[0]
    if n == 0:
        return dict(n=0, mean=float("nan"), std=float("nan"),
                    median=float("nan"), max=float("nan"),
                    max_over_median=float("nan"), log_std=float("nan"),
                    top_outlier_indices=[])
    median = float(np.median(r))
    safe_r = np.maximum(r, 1e-12)
    k = min(top_k, n)
    top_idx = np.argsort(r)[::-1][:k]
    return dict(
        n=int(n),
        mean=float(r.mean()),
        std=float(r.std()),
        median=median,
        max=float(r.max()),
        max_over_median=float(r.max() / max(median, 1e-12)),
        log_std=float(np.log(safe_r).std()),
        top_outlier_indices=[int(i) for i in top_idx],
    )


def cluster_norm_profile(norms, labels) -> dict:
    """
    Coupling between norm and cluster membership.

    labels : (n,) integer cluster labels, HDBSCAN convention (-1 = noise).

    Returns per-label {mean, std, n} plus the headline comparison
    clustered_minus_noise_mean (nan when either side is empty). A large
    positive value means high-norm particles concentrate inside clusters;
    large negative means the sinks are the unclustered outliers.
    """
    r = _as_numpy(norms).astype(np.float64, copy=False).ravel()
    lab = _as_numpy(labels).astype(np.int64, copy=False).ravel()
    if r.shape[0] != lab.shape[0]:
        raise ValueError(
            f"cluster_norm_profile: {r.shape[0]} norms vs {lab.shape[0]} labels"
        )
    per_label = {}
    for lb in np.unique(lab):
        sel = r[lab == lb]
        per_label[int(lb)] = dict(
            mean=float(sel.mean()), std=float(sel.std()), n=int(sel.size)
        )
    clustered = r[lab >= 0]
    noise = r[lab == -1]
    diff = (float(clustered.mean() - noise.mean())
            if clustered.size and noise.size else float("nan"))
    return dict(per_label=per_label, clustered_minus_noise_mean=diff)


# ---------------------------------------------------------------------------
# Sphere-gap diagnostic
# ---------------------------------------------------------------------------

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ac, bc = a - a.mean(), b - b.mean()
    denom = np.sqrt(float((ac ** 2).sum()) * float((bc ** 2).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((ac * bc).sum() / denom)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho as Pearson of ranks (average-rank ties via argsort of
    argsort is fine for the continuous values seen here)."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return _pearson(ra, rb)


def sphere_gap(activations) -> dict:
    """
    How much the sphere projection distorts this layer's pairwise
    structure. Decomposition used throughout: raw_ij = r_i r_j g_ij,
    so the gap is entirely attributable to norm spread.

    Returns
    -------
    pearson_gap  : 1 - Pearson(raw upper-triangle, cosine upper-triangle).
                   0 when all norms are equal (raw is then a positive
                   multiple of cosine). This is the headline number.
    spearman_gap : 1 - Spearman of the same pairs — rank distortion:
                   nonzero means norm spread actually REORDERS which
                   pairs count as most similar, not just rescales them.
    norm_log_std : std(log r) — the scale-free driver of both gaps.
    n_pairs      : number of upper-triangle pairs the gaps are over.

    Interpretation contract: report gaps per layer next to the existing
    sphere metrics. Where gaps are ~0, sphere-frame conclusions transfer
    to the raw stream; where they spike, that layer's sphere-frame
    clustering claims need the LN-frame / functional arbiter
    (core/ln_frame.py, core/functional_distance.py) before being trusted.
    """
    arr = _as_numpy(activations).astype(np.float64, copy=False)
    n = arr.shape[0]
    if n < 2:
        return dict(pearson_gap=float("nan"), spearman_gap=float("nan"),
                    norm_log_std=float("nan"), n_pairs=0)
    r = particle_norms(arr)
    G_cos = gram_matrix(arr)
    G_raw = raw_gram(arr)
    cos_u = pairwise_upper(G_cos)
    raw_u = pairwise_upper(G_raw)
    return dict(
        pearson_gap=float(1.0 - _pearson(raw_u, cos_u)),
        spearman_gap=float(1.0 - _spearman(raw_u, cos_u)),
        norm_log_std=float(np.log(np.maximum(r, 1e-12)).std()),
        n_pairs=int(cos_u.size),
    )


def polar_layer_record(activations, labels=None, top_k: int = 5) -> dict:
    """
    One-call convenience for the analysis loop: everything this module
    produces for a single layer, JSON-ready. `labels` optional (pass the
    layer's HDBSCAN labels when available).

    Intended wiring (analysis_p1.py layer loop, which already holds
    `activations` raw and `normed` side by side):
        lr["polar"] = polar_layer_record(activations, hdb_labels)
    """
    r = particle_norms(activations)
    rec = dict(
        norms=[float(v) for v in r],
        norm_stats=norm_stats(r, top_k=top_k),
        sphere_gap=sphere_gap(activations),
    )
    if labels is not None:
        rec["cluster_norm_profile"] = cluster_norm_profile(r, labels)
    return rec
