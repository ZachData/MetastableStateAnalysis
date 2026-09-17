"""
core/isometric_path.py — the pure-math half of §2.5's isometric path
construction (`MATH_SPECTRAL_OT.md` §2.5). Split out of
`tools/run/isometric_path_sweep.py` (2026-09-16) specifically so it is
importable, and its properties runtime-checkable, with torch absent —
`tools/run/induction_rank_sweep.py`, the real-model half's dependency,
imports torch at module level, which would otherwise make even these
pure-numpy functions untestable in the pure tier (`pytest.ini`'s CI
gating tier).

Given `M = U Sigma V^T` (thin SVD, `U`, `V` Stiefel), `M(t) = gamma(t)
Sigma gamma(1-t)^T` for any smooth `gamma: [0,1] -> Stiefel` with
`gamma(0) = U`, `gamma(1) = V` is an EXACT ISOMETRY at every `t` — same
singular values, same Frobenius norm, same rank as `M`, with `M(0) = M`,
`M(1) = M^T`, `M(1/2)` symmetric PSD. `polar_frame`/`check_refusal`/
`build_M_t` implement the closed-form `gamma` (polar retraction of the
chord, §2.5.2) that needs no geodesic. See `tools/run/isometric_path_sweep.py`
for the real-model sweep this feeds, and its own docstring for the full
derivation and the target (`L7H8`, pythia-410m).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class RunRefused(RuntimeError):
    pass


def polar_frame(Y: np.ndarray) -> Optional[np.ndarray]:
    """`gamma = Y (Y^T Y)^{-1/2}`. `None` if `Y^T Y` is singular (the refusal
    condition, checked in bulk by `check_refusal` before any forward pass —
    this is the per-point primitive it calls)."""
    YtY = Y.T @ Y
    w, Q = np.linalg.eigh(YtY)
    if np.min(w) <= 1e-12:
        return None
    inv_sqrt = (Q * (w ** -0.5)) @ Q.T
    return Y @ inv_sqrt


def check_refusal(U: np.ndarray, V: np.ndarray, n_grid: int = 101) -> dict:
    """`sigma_min(Y(t)) > 0` on `[0, 1]`, per §2.5.2's refusal condition.
    Checked on a fine grid rather than only at the sweep's own `t` values,
    so a thin spike between two sampled points is not missed."""
    ts = np.linspace(0.0, 1.0, n_grid)
    sigma_mins = np.array([
        np.linalg.svd((1 - t) * U + t * V, compute_uv=False).min() for t in ts
    ])
    return {
        "t_grid": ts.tolist(),
        "sigma_min": sigma_mins.tolist(),
        "min_sigma_min": float(sigma_mins.min()),
        "ok": bool(np.all(sigma_mins > 1e-10)),
    }


def build_M_t(U: np.ndarray, S: np.ndarray, V: np.ndarray, t: float):
    """`M(t) = gamma(t) Sigma gamma(1-t)^T`, returned as `(A_t, B_t)` OV
    factors with `A_t @ B_t == M(t)` (matching `ov_factors`'/`write_ov`'s
    contract in `tools/run/induction_rank_sweep.py`). `None` if either
    `gamma(t)` or `gamma(1-t)` is refused."""
    gamma_t = polar_frame((1 - t) * U + t * V)
    gamma_1mt = polar_frame(t * U + (1 - t) * V)
    if gamma_t is None or gamma_1mt is None:
        return None
    A_t = gamma_t * S
    B_t = gamma_1mt.T
    return A_t, B_t
