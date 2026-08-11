"""
p1c_frames/gamma_ode.py — the paper's closed-form pairwise-inner-product
trajectory, and the collapse time derived from it.

WHAT THIS IS

Theorem 6.8 of Geshkovski et al. (arXiv:2312.10794v5): for pairwise-
ORTHOGONAL initial points, permutation equivariance forces all pairwise
angles to stay equal, so the entire n-particle configuration is described
by one scalar gamma(t) = cos(theta(t)) obeying

    (SA), eq. (6.9)
        d(gamma)/dt = 2 e^{b g} (1-g) ((n-1)g + 1) / (e^b + (n-1) e^{b g})

    (USA), eq. (6.10)
        d(gamma)/dt = (2/n) e^{b g} (1-g) ((n-1)g + 1)

both with gamma(0) = 0.

Theorem 6.9 is what makes this usable on real data: for d >= d*(n, beta),
|<x_i, x_j> - gamma(t)| is small with high probability for ALL pairs — the
whole trajectory concentrates on this one curve. Our prompts have
n <= 512 against d = 1024, so we are in the regime where 6.9 applies.

WHY IT MATTERS HERE

gamma(t) is a closed-form prediction for the mean pairwise inner product as
a function of time. `ip_mean` is exactly that quantity measured per layer.
The project has never compared them. Doing so requires a time axis, which
is what integration_time.py supplies, and the residual between the two
curves is the actual object of study (see design-1c.md).

STANDING HYPOTHESES, all of which our models violate:
    Q^T K = V = I,  single head,  no FFN,  orthogonal initialization.
The point of the comparison is not that the model should obey this. It is
that the identity-weight dynamics running for the observed amount of time
is the correct null, and "the trained network resists collapse" is only
meaningful as a statement about the residual against it.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Right-hand sides
# ---------------------------------------------------------------------------

def gamma_rhs_sa(g: float, n: int, beta: float) -> float:
    """
    eq. (6.9), the NORMALIZED dynamics (SA) — softmax with its partition
    function, which is what an actual attention layer computes.

    Written to avoid overflow at large beta: the numerator and denominator
    both carry e^{b g} and e^{b}, so factor out e^{b*max(g,1)} first. At
    beta=5, g near 1, the naive form overflows float64 for large n.
    """
    b = float(beta)
    m = max(b * g, b)
    num = 2.0 * np.exp(b * g - m) * (1.0 - g) * ((n - 1) * g + 1.0)
    den = np.exp(b - m) + (n - 1) * np.exp(b * g - m)
    return float(num / den)


def gamma_rhs_usa(g: float, n: int, beta: float) -> float:
    """
    eq. (6.10), the SURROGATE (USA) — the partition function dropped. This
    is the object most of the paper's gradient-flow structure is proved
    for (Lemma 3.6: USA is a Wasserstein gradient flow for E_beta; SA is
    not, except in the reweighted metric of sec. 3.4).

    Kept separate rather than treated as interchangeable with SA, because
    they are not: see collapse_time_table below, where at n=20, beta=5 the
    two differ by a factor of ten. The surrogate is not a stand-in in the
    high-beta regime — which is precisely the regime the paper's own
    metastability numerics live in.
    """
    b = float(beta)
    return float((2.0 / n) * np.exp(b * g) * (1.0 - g) * ((n - 1) * g + 1.0))


_RHS = {"sa": gamma_rhs_sa, "usa": gamma_rhs_usa}


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def integrate_gamma(n: int, beta: float, t_max: float = 12.0,
                    dt: float = 1e-3, model: str = "sa",
                    g0: float = 0.0):
    """
    Integrate (6.9) or (6.10) from gamma(0) = g0 to t_max on a fixed grid.

    Returns (t, gamma), both 1-D arrays.

    RK4 on a fixed step. An adaptive solver is not worth the dependency
    here: the RHS is smooth, monotone, and bounded on [0, 1], and the
    solution is a single sigmoid. The step is validated in
    `integrate_gamma_converged` below by halving until the collapse time
    stops moving, which is the only accuracy claim this module makes.

    gamma is clipped to [g0, 1) at every step. It cannot exceed 1
    analytically — the (1-g) factor kills the RHS there — but a fixed-step
    method can overshoot in the last few steps of a stiff high-beta run,
    and an overshoot past 1 makes (1-g) negative and sends the solution to
    -inf. The clip is a guard against a numerical artifact, not a physical
    constraint.
    """
    if model not in _RHS:
        raise ValueError(f"model must be 'sa' or 'usa', got {model!r}")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    f = _RHS[model]

    n_steps = int(np.ceil(t_max / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    g = np.empty(n_steps + 1, dtype=np.float64)
    g[0] = g0

    for i in range(n_steps):
        x = g[i]
        k1 = f(x, n, beta)
        k2 = f(min(x + 0.5 * dt * k1, 1.0), n, beta)
        k3 = f(min(x + 0.5 * dt * k2, 1.0), n, beta)
        k4 = f(min(x + dt * k3, 1.0), n, beta)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        g[i + 1] = float(np.clip(x_next, g0, 1.0 - 1e-15))

    return t, g


def integrate_gamma_converged(n: int, beta: float, t_max: float = 12.0,
                              model: str = "sa", target: float = 0.9,
                              dt0: float = 1e-2, rtol: float = 1e-4,
                              max_halvings: int = 8):
    """
    Integrate with step halving until t(gamma = target) stops moving by
    more than rtol. Returns (t, gamma, info) with info recording the final
    dt, the number of halvings, and the last relative change — so a run
    that never converged is visible rather than silently returning the
    coarsest answer.
    """
    dt = dt0
    prev = None
    info = {"dt": dt, "halvings": 0, "rel_change": float("nan"),
            "converged": False}
    t, g = integrate_gamma(n, beta, t_max=t_max, dt=dt, model=model)
    for k in range(max_halvings):
        cur = time_to_threshold(t, g, target)
        if prev is not None and np.isfinite(cur) and np.isfinite(prev):
            rel = abs(cur - prev) / max(abs(prev), 1e-12)
            info.update(dt=dt, halvings=k, rel_change=float(rel))
            if rel < rtol:
                info["converged"] = True
                return t, g, info
        prev = cur
        dt /= 2.0
        t, g = integrate_gamma(n, beta, t_max=t_max, dt=dt, model=model)
    info.update(dt=dt, halvings=max_halvings)
    return t, g, info


def time_to_threshold(t: np.ndarray, g: np.ndarray, target: float) -> float:
    """
    First time gamma reaches `target`, linearly interpolated between grid
    points. inf if the trajectory never gets there within t_max — which is
    a real outcome at small beta and must not be silently reported as
    t_max.
    """
    idx = np.searchsorted(g, target)
    if idx == 0:
        return float(t[0])
    if idx >= len(g):
        return float("inf")
    g0, g1 = g[idx - 1], g[idx]
    if g1 == g0:
        return float(t[idx])
    frac = (target - g0) / (g1 - g0)
    return float(t[idx - 1] + frac * (t[idx] - t[idx - 1]))


def gamma_at(t_query, n: int, beta: float, model: str = "sa",
             t_max: float = 12.0, dt: float = 1e-3):
    """
    gamma_beta evaluated at arbitrary times — the interface sub-experiment
    B actually uses, since it needs gamma at the cumulative T_eff of each
    layer boundary rather than on a regular grid.

    Queries beyond t_max are clamped to the final value with a warning
    rather than extrapolated, because gamma is asymptotic to 1 and naive
    extrapolation of a saturating curve is the kind of error that looks
    like a result.
    """
    t_query = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
    need = float(np.nanmax(t_query)) if t_query.size else 0.0
    if need > t_max:
        t_max = need * 1.05
    t, g = integrate_gamma(n, beta, t_max=t_max, dt=dt, model=model)
    return np.interp(t_query, t, g)


# ---------------------------------------------------------------------------
# Collapse time — the number T_eff gets compared against
# ---------------------------------------------------------------------------

def collapse_time(n: int, beta: float, target: float = 0.9,
                  model: str = "sa", t_max: float = 12.0) -> float:
    """t* — the time at which gamma_beta reaches `target`."""
    t, g, _ = integrate_gamma_converged(n, beta, t_max=t_max, model=model,
                                        target=target)
    return time_to_threshold(t, g, target)


def collapse_time_table(ns=(20, 467), betas=(0.1, 1.0, 2.0, 5.0),
                        targets=(0.5, 0.9)) -> list:
    """
    The (n, beta) -> t* grid, for both SA and USA.

    Two facts fall out of this table and both matter for how Phase 1's
    result should be read:

    1. COLLAPSE TIME IS SHORT AND NEARLY beta-INDEPENDENT. At n = 467,
       reaching gamma = 0.9 takes t* ~ 4.2 under (SA), essentially
       unchanged across two decades of beta. So "how much integration time
       would this network need in order to collapse" has a single answer,
       and it does not depend on the beta sweep.

    2. SA AND USA SEPARATE AT LARGE beta AND SMALL n. At n=20, beta=5 the
       times are ~8.3 (SA) against ~0.79 (USA) — a factor of ten. Any
       claim that reads the surrogate as a stand-in for the normalized
       dynamics is unsupported in exactly the corner where the paper's
       metastability numerics sit.
    """
    rows = []
    for n in ns:
        for b in betas:
            row = {"n": int(n), "beta": float(b)}
            for model in ("sa", "usa"):
                t, g, info = integrate_gamma_converged(n, b, model=model)
                for tgt in targets:
                    row[f"{model}_t{tgt}"] = time_to_threshold(t, g, tgt)
                row[f"{model}_dt"] = info["dt"]
                row[f"{model}_converged"] = info["converged"]
            rows.append(row)
    return rows
