"""
p1d_cluster_ensemble/position_null.py — admission's null with position kept.

Blocked 11″ option (a), as revised by Blocked 11‴ (`status-1d.md` "Blocked
11″ decided, and where 1d stands", and "Position-keeping null"; form fixed
there before this code). `gaussian_null.gaussian_draw` draws every token
from one Gaussian, so a group of early tokens that share a prefix average
(#123's mechanism: near-uniform attention at init) beats it on position
alone. Here each token keeps a position-driven mean ``mu_t`` and only the
rest is drawn, **on the un-normalised activations**; the frame (unit rows,
centring) is applied to each draw afterwards, as it is to the data:

    x_t = mu_t + (G R / sqrt(n))_t,   R = Y - mu,   G ~ N(0, I_n)

Why before normalisation (11‴): on unit rows an early token's shared
component inflates its norm, so its own noise is a small share after
normalising, while a pooled draw gives every row the same residual
covariance and spreads the opening wider than it is. Even the true mean,
drawn on unit rows, admitted a synthetic opening in 7 of 10 seeds.

- ``smooth`` (primary): ``mu_t`` a leave-one-out Gaussian kernel smoother
  over ``log(1 + position)``, bandwidth by leave-one-out CV on a grid that
  ends at "no position" (the leave-one-out mean).
- ``prefix`` (sensitivity arm): ``mu_t = mean(Y) + c (P_t - mean(P))``,
  ``P_t`` the causal running mean of the kept rows before ``t`` (the first
  kept row gets ``mean(P[1:])``), ``c`` one least-squares scalar on the
  centred regressor. It under-fits the synthetic opening (c ~ 0.5).
- ``flat``: ``mu_t = mean(Y)``: `gaussian_draw` with ``renorm=False``,
  draw for draw (a test), i.e. #122's null moved before normalisation. The
  comparator that differs from ``smooth`` only in keeping no position
  (`/challenge-pr` on #128, finding 2).

``R`` is the regression residual, centred, so the draw has the residual's
covariance around a mean that keeps position. All fits act on rows, so they
commute with `span_coordinates`. ``info["resid_top_share"]`` is the largest
single row's share of ``sum |R_t|^2``, i.e. of every draw's noise: on trained
layers token 0's massive activation reaches 0.75-0.93 at L6-18 (step 0:
0.01), and such a draw is mostly that one direction. Tier 1, unregistered.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .gaussian_null import _unit_rows, mean_direction

NULLS = ("flat", "smooth", "prefix")
#: PLACED (`status-1d.md` "Blocked 11″ decided"): bandwidths on log(1 + position).
SMOOTH_BANDWIDTHS = tuple(float(h) for h in np.geomspace(0.02, 4.0, 16))


def prefix_mean(Y: np.ndarray) -> np.ndarray:
    """``P_i`` = mean of rows ``0..i-1`` (rows in position order); ``P_0`` = mean of the rest."""
    Y = np.asarray(Y, dtype=np.float64)
    csum = np.cumsum(Y, axis=0)
    P = np.empty_like(Y)
    P[1:] = csum[:-1] / np.arange(1, Y.shape[0])[:, None]
    P[0] = P[1:].mean(axis=0) if Y.shape[0] > 1 else Y[0]
    return P


def fit_prefix(Y: np.ndarray) -> Tuple[np.ndarray, Dict]:
    Y = np.asarray(Y, dtype=np.float64)
    ybar = Y.mean(axis=0)
    Pc = prefix_mean(Y)
    Pc = Pc - Pc.mean(axis=0)
    den = float((Pc * Pc).sum())
    c = float(((Y - ybar) * Pc).sum() / den) if den > 0 else 0.0
    return ybar + c * Pc, {"c": c}


def _loo_kernel_mean(Y: np.ndarray, u: np.ndarray, h: float) -> Optional[np.ndarray]:
    W = np.exp(-0.5 * ((u[:, None] - u[None, :]) / h) ** 2)
    np.fill_diagonal(W, 0.0)
    s = W.sum(axis=1)
    if np.any(s < 1e-300):
        return None
    return (W @ Y) / s[:, None]


def fit_smooth(Y: np.ndarray, positions: Sequence[int],
               bandwidths: Sequence[float] = SMOOTH_BANDWIDTHS) -> Tuple[np.ndarray, Dict]:
    """The leave-one-out CV pick; ``h`` is ``inf`` when "no position" wins."""
    Y = np.asarray(Y, dtype=np.float64)
    n = Y.shape[0]
    u = np.log1p(np.asarray(positions, dtype=np.float64))
    cands = [(float("inf"), (Y.sum(axis=0)[None, :] - Y) / (n - 1))]
    for h in bandwidths:
        mu = _loo_kernel_mean(Y, u, h)
        if mu is not None:
            cands.append((float(h), mu))
    errs = [float(((Y - mu) ** 2).sum()) for _, mu in cands]
    k = int(np.argmin(errs))
    return cands[k][1], {"h": cands[k][0], "cv": {str(h): e for (h, _), e in zip(cands, errs)}}


def fit(Y: np.ndarray, positions: Sequence[int], null: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """``(mu, R, info)`` for ``null`` on un-normalised rows ``Y`` in position order."""
    if null not in NULLS:
        raise ValueError(f"unknown null {null!r}; use one of {NULLS}")
    Y = np.asarray(Y, dtype=np.float64)
    if len(positions) != Y.shape[0] or np.any(np.diff(np.asarray(positions)) <= 0):
        raise ValueError("positions must be one strictly increasing position per row")
    if null == "flat":
        mu, info = np.broadcast_to(Y.mean(axis=0), Y.shape).copy(), {}
    elif null == "prefix":
        mu, info = fit_prefix(Y)
    else:
        mu, info = fit_smooth(Y, positions)
    R = Y - mu
    R = R - R.mean(axis=0)
    tot = float(((Y - Y.mean(axis=0)) ** 2).sum())
    r2 = (R ** 2).sum(axis=1)
    info = {"null": null, **info,
            "r2": 1.0 - float(r2.sum()) / tot if tot > 0 else 0.0,
            "resid_top_share": float(r2.max() / r2.sum()) if r2.sum() > 0 else 0.0,
            "resid_top_row": int(np.argmax(r2))}
    return mu, R, info


def draw(mu: np.ndarray, R: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``mu + G R / sqrt(n)``, not normalised: apply the frame (`apply_frame`) after."""
    n = R.shape[0]
    G = rng.standard_normal((n, n)) / np.sqrt(n)
    return mu + G @ R


def apply_frame(X: np.ndarray, frame: str) -> np.ndarray:
    """`gaussian_null.frame_vectors`' rows for ``raw`` / ``centred``, without its diagnostics."""
    Z = _unit_rows(X)
    if frame == "centred":
        zh = mean_direction(Z)
        Z = _unit_rows(Z - np.outer(Z @ zh, zh))
    elif frame != "raw":
        raise ValueError(f"position nulls take frame raw or centred, not {frame!r}")
    return Z
