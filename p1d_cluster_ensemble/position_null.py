"""
p1d_cluster_ensemble/position_null.py — admission's null with position kept.

Blocked 11″ option (a) (`status-1d.md` "Blocked 11″ decided, and where 1d
stands"; form fixed there before this code). `gaussian_null.gaussian_draw`
draws every token from one Gaussian, so a group of early tokens that share a
prefix average (#123's mechanism: near-uniform attention at init) beats it on
position alone. Here each token keeps a position-driven mean ``mu_t`` and only
the rest is drawn:

    x_t = mu_t + (G R / sqrt(n))_t,   R = Z - mu,   G ~ N(0, I_n),   unit rows

- ``prefix`` (primary): ``mu_t = mean(Z) + c (P_t - mean(P))``, ``P_t`` the
  causal running mean of the kept rows before ``t`` (the first kept row gets
  ``mean(P[1:])``), ``c`` one least-squares scalar on the centred regressor.
- ``smooth`` (sensitivity arm): ``mu_t`` a leave-one-out Gaussian kernel
  smoother over ``log(1 + position)``, bandwidth by leave-one-out CV on a grid
  that ends at "no position" (the leave-one-out mean).
- ``gaussian``: ``mu_t = mean(Z)``, which is `gaussian_draw` draw for draw.

``R`` is the regression residual (mean 0 for ``prefix`` by construction;
centred for ``smooth``), so the draw has the residual's covariance around a
mean that keeps position. Rows are in the frame's span coordinates, which
commute with all three fits (they act on rows). Tier 1, unregistered.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .gaussian_null import _unit_rows

NULLS = ("gaussian", "prefix", "smooth")
#: PLACED (`status-1d.md` "Blocked 11″ decided"): bandwidths on log(1 + position).
SMOOTH_BANDWIDTHS = tuple(float(h) for h in np.geomspace(0.02, 4.0, 16))


def prefix_mean(Z: np.ndarray) -> np.ndarray:
    """``P_i`` = mean of rows ``0..i-1`` (rows in position order); ``P_0`` = mean of the rest."""
    Z = np.asarray(Z, dtype=np.float64)
    csum = np.cumsum(Z, axis=0)
    P = np.empty_like(Z)
    P[1:] = csum[:-1] / np.arange(1, Z.shape[0])[:, None]
    P[0] = P[1:].mean(axis=0) if Z.shape[0] > 1 else Z[0]
    return P


def fit_prefix(Z: np.ndarray) -> Tuple[np.ndarray, Dict]:
    Z = np.asarray(Z, dtype=np.float64)
    zbar = Z.mean(axis=0)
    Pc = prefix_mean(Z)
    Pc = Pc - Pc.mean(axis=0)
    den = float((Pc * Pc).sum())
    c = float(((Z - zbar) * Pc).sum() / den) if den > 0 else 0.0
    mu = zbar + c * Pc
    return mu, {"c": c}


def _loo_kernel_mean(Z: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
    W = np.exp(-0.5 * ((u[:, None] - u[None, :]) / h) ** 2)
    np.fill_diagonal(W, 0.0)
    s = W.sum(axis=1)
    if np.any(s < 1e-300):
        return None
    return (W @ Z) / s[:, None]


def fit_smooth(Z: np.ndarray, positions: Sequence[int],
               bandwidths: Sequence[float] = SMOOTH_BANDWIDTHS) -> Tuple[np.ndarray, Dict]:
    Z = np.asarray(Z, dtype=np.float64)
    n = Z.shape[0]
    u = np.log1p(np.asarray(positions, dtype=np.float64))
    loo_mean = (Z.sum(axis=0)[None, :] - Z) / (n - 1)
    cands = [(float("inf"), loo_mean)]
    for h in bandwidths:
        mu = _loo_kernel_mean(Z, u, h)
        if mu is not None:
            cands.append((float(h), mu))
    errs = [float(((Z - mu) ** 2).sum()) for _, mu in cands]
    k = int(np.argmin(errs))
    return cands[k][1], {"h": cands[k][0], "cv": {str(h): e for (h, _), e in zip(cands, errs)}}


def fit(Z: np.ndarray, positions: Sequence[int], null: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """``(mu, R, info)`` for ``null``; ``info`` carries the fit and its R²."""
    if null not in NULLS:
        raise ValueError(f"unknown null {null!r}; use one of {NULLS}")
    Z = np.asarray(Z, dtype=np.float64)
    if len(positions) != Z.shape[0] or np.any(np.diff(np.asarray(positions)) <= 0):
        raise ValueError("positions must be one strictly increasing position per row")
    if null == "gaussian":
        mu, info = np.broadcast_to(Z.mean(axis=0), Z.shape).copy(), {}
    elif null == "prefix":
        mu, info = fit_prefix(Z)
    else:
        mu, info = fit_smooth(Z, positions)
    R = Z - mu
    if null == "smooth":
        R = R - R.mean(axis=0)
    tot = float(((Z - Z.mean(axis=0)) ** 2).sum())
    info = {"null": null, **info, "r2": 1.0 - float((R ** 2).sum()) / tot if tot > 0 else 0.0}
    return mu, R, info


def draw(mu: np.ndarray, R: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``mu + G R / sqrt(n)``, unit rows: `gaussian_draw`'s draw around a per-row mean."""
    n = R.shape[0]
    G = rng.standard_normal((n, n)) / np.sqrt(n)
    return _unit_rows(mu + G @ R)
