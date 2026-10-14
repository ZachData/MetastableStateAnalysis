"""The identity-weight (SA) dynamics simulated on a cloud, with or without the causal mask.

    dx_i/dt = P_perp(x_i) sum_j softmax_j(beta <x_i, x_j>) x_j,   j <= i if causal

Eq. (6.9) (`p1c_frames/gamma_ode.py`) is the full-attention, orthogonal-start case; Pythia's
attention is causal and its clouds are not orthogonal, so A1's comparator t* is taken from this
simulator started from the measured cloud (`design-A.md` "t* before any output", after
`/challenge-pr` on #201, finding 2). Forward Euler with retraction to the sphere; `dt` is
checked by halving (`tstar(..., check_dt=True)`).

    python -m trackA_force.sa_sim            # the known-answer table in design-A.md
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _dev() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def field(U: torch.Tensor, beta: float, causal: bool) -> torch.Tensor:
    S = beta * (U @ U.T)
    if causal:
        n = U.shape[0]
        S = S.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool, device=U.device), 1), -math.inf)
    A = torch.softmax(S, dim=1)
    V = A @ U
    return V - (V * U).sum(1, keepdim=True) * U


def mean_cos(U: torch.Tensor) -> float:
    n = U.shape[0]
    s = U.sum(0)
    return float(((s @ s) - n) / (n * (n - 1)))


def time_to(U0: np.ndarray, beta: float, causal: bool, target: float = 0.9,
            dt: float = 0.01, t_max: float = 60.0) -> float:
    """Time for the mean pairwise cosine to reach `target`; inf if not by t_max.

    Refuses a start already at or above the target (the reading would be a silent 0)."""
    U = torch.as_tensor(U0, dtype=torch.float64, device=_dev())
    U = U / U.norm(dim=1, keepdim=True)
    g = mean_cos(U)
    if g >= target:
        raise ValueError(f"start mean cosine {g:.3f} >= target {target}")
    t = 0.0
    while t < t_max:
        U = U + dt * field(U, beta, causal)
        U = U / U.norm(dim=1, keepdim=True)
        t += dt
        if mean_cos(U) >= target:
            return t
    return math.inf


def tstar(U0: np.ndarray, beta: float, causal: bool = True, check_dt: bool = False, **kw) -> float:
    t = time_to(U0, beta, causal, **kw)
    if check_dt:
        dt = kw.pop("dt", 0.01)
        t2 = time_to(U0, beta, causal, dt=dt / 2, **kw)
        if math.isfinite(t) and abs(t2 - t) > 0.02 * t:
            raise RuntimeError(f"dt not converged: {t:.3f} at dt, {t2:.3f} at dt/2")
    return t


def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 467, 1024
    U0 = rng.standard_normal((n, d))
    print(f"n = {n}, d = {d}, Gaussian start (seed 0), target mean cosine 0.9, dt 0.01")
    print("beta   full    causal")
    for b in (0.0, 3.5, 5.6, 7.0):
        print(f"{b:<6} {tstar(U0, b, causal=False):<7.2f} {tstar(U0, b, causal=True):<7.2f}")


if __name__ == "__main__":
    main()
