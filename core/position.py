"""core/position.py — one shared position covariate for every per-token readout (Track B, B0;
`trackB_circuit/design-B.md` "B0", row "position").

Phase 10 had three ad hoc ones (`p10_cluster_function/questions-10.md` §7 item 4); from Track B
on, a per-token readout that needs position removed uses this one. The covariate is a cubic in
``u = log(i + 1)`` plus an indicator for position 0, fitted by least squares over every token
given, **membership-blind**: no label, cluster or arm enters the fit, so residualising cannot
carry the quantity a reading is about.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

DEGREE = 3          # the cubic in log(i + 1); fixed by the design, not tuned


def design(positions: Sequence[int]) -> np.ndarray:
    """``[1, u, u², u³, 1{i = 0}]`` per position, ``u = log(i + 1)``; float64, shape (n, 5)."""
    i = np.asarray(positions)
    if i.ndim != 1 or (i < 0).any():
        raise ValueError("positions must be a 1-d array of non-negative token indices")
    u = np.log1p(i.astype(np.float64))
    cols = [u ** p for p in range(DEGREE + 1)] + [(i == 0).astype(np.float64)]
    return np.stack(cols, axis=1)


def fit(v: np.ndarray, positions: Sequence[int]) -> np.ndarray:
    """Least-squares coefficients of ``v`` (n,) or (n, m) on :func:`design`. Rank-deficient
    designs (few distinct positions) get the minimum-norm solution, as ``lstsq`` gives."""
    X = design(positions)
    v = np.asarray(v, dtype=np.float64)
    if v.shape[0] != X.shape[0]:
        raise ValueError(f"{v.shape[0]} values for {X.shape[0]} positions")
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return beta


def residualise(v: np.ndarray, positions: Optional[Sequence[int]] = None) -> np.ndarray:
    """``v`` less its position fit, over every row given (membership-blind). ``positions``
    defaults to ``0 … n−1``."""
    v = np.asarray(v, dtype=np.float64)
    if positions is None:
        positions = np.arange(v.shape[0])
    return v - design(positions) @ fit(v, positions)
