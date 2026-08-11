"""
p1c_frames/gamma_null.py — sub-experiment B: the gamma_beta null model.

THE DELIVERABLE IS THE RESIDUAL, NOT THE FIT.

Phase 1 measures `ip_mean` per layer. Theorem 6.8 gives a closed form for
exactly that quantity as a function of time, and Theorem 6.9 says that at
d >> n the whole configuration concentrates on it. Sub-experiment A gives
the time axis. Putting the three together produces the first version of
"the trained network resists collapse" that is a measured quantity rather
than a comparison against an idealization:

    residual(l) = ip_mean(l) - gamma_{beta_eff}( T_eff(l) )

Read: the part of the layer-wise trajectory NOT explained by identity-weight
dynamics run for the amount of time the network actually runs them.

    residual < 0  the network is BEHIND the identity-weight prediction —
                  less clustered than pure attention would make it. This is
                  resistance, and its magnitude is now a number.
    residual ~ 0  the network is doing what the null does. "Resistance" was
                  the comparison against t = infinity, not a property of
                  the weights.
    residual > 0  the network clusters FASTER than the null.

P-γ1 predicts the residual is near zero at step 0 and grows monotonically
with training. Its falsifier is that the step-0 residual is already as
large as the step-143k residual, which would mean the gap is architectural
rather than learned.

WHAT THIS COMPARISON ASSUMES, stated because it is doing a lot of work:

1. beta_eff is measurable and roughly constant across layers. It is not
   exactly constant, so the null is evaluated per layer at that layer's own
   beta and the spread is reported. A null built on a single global beta
   would silently absorb the layer-wise variation into the residual.
2. Theorem 6.8 assumes ORTHOGONAL initialization. Real layer-0 embeddings
   are not orthogonal — they carry a large common mode. The null is
   therefore also integrated from the OBSERVED ip_mean at layer 0 rather
   than from gamma(0) = 0, as `gamma_null_from_observed`. If the two nulls
   disagree, the disagreement is an anisotropy effect and not resistance,
   which is a distinction the single-null version cannot make.
3. Causal masking. The paper's field is non-causal. Both are available in
   integration_time.sa_field; whichever was used for T_eff must be used
   here, and the choice is recorded.
"""

from __future__ import annotations

import numpy as np

from .gamma_ode import collapse_time, integrate_gamma, time_to_threshold


def gamma_null(t_grid, n: int, beta: float, model: str = "sa",
               g0: float = 0.0) -> np.ndarray:
    """
    gamma_beta evaluated on an arbitrary (possibly non-uniform) time grid —
    which is what T_eff(l) is, since the step size varies by layer.

    g0 : starting inner product. 0.0 is Theorem 6.8's orthogonal
         initialization; pass the observed layer-0 ip_mean for the
         anisotropy-matched null (see module docstring, point 2).
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    if t_grid.size == 0:
        return np.array([])
    t_max = max(float(np.nanmax(t_grid)) * 1.05, 1e-3)
    t, g = integrate_gamma(n, beta, t_max=t_max, dt=min(1e-3, t_max / 2000),
                           model=model, g0=g0)
    return np.interp(t_grid, t, g)


def residual_curve(ip_mean, t_eff_grid, n: int, beta, model: str = "sa",
                   match_initial: bool = True) -> dict:
    """
    The central object of sub-experiment B.

    ip_mean     : (n_layers,) measured mean pairwise inner product per layer
    t_eff_grid  : (n_layers,) cumulative integration time before each layer,
                  from integration_time.cumulative_time
    n           : token count for this prompt
    beta        : scalar beta_eff, or a per-layer array. A per-layer array
                  is preferred: beta_eff is not constant across depth, and
                  folding that variation into the residual would attribute
                  a property of the QK circuits to the weights' resistance.
    match_initial : also compute the null started from the OBSERVED layer-0
                  ip_mean rather than from 0. Theorem 6.8 assumes orthogonal
                  init; embeddings are not orthogonal.

    Returns dict with gamma_null, gamma_null_matched, residual,
    residual_matched, and summary scalars.
    """
    ip_mean = np.asarray(ip_mean, dtype=np.float64)
    t_grid = np.asarray(t_eff_grid, dtype=np.float64)
    if len(ip_mean) != len(t_grid):
        raise ValueError(
            f"ip_mean has {len(ip_mean)} entries and t_eff_grid has "
            f"{len(t_grid)}; cumulative_time returns n_layers entries for "
            f"n_layers-1 steps, so these must already align"
        )

    betas = (np.full(len(ip_mean), float(beta)) if np.isscalar(beta)
             else np.asarray(beta, dtype=np.float64))

    # Per-layer beta means the null is not one curve but a family; evaluate
    # each layer against its own. Layers with unusable beta_eff (nan from a
    # failed regression) fall back to the run's median rather than dropping
    # the layer, so the residual series has no holes — and the fallback is
    # counted.
    med = float(np.nanmedian(betas)) if np.any(np.isfinite(betas)) else 1.0
    n_fallback = int(np.sum(~np.isfinite(betas)))
    betas = np.where(np.isfinite(betas), betas, med)

    g_null = np.array([
        gamma_null([t_grid[i]], n, betas[i], model=model)[0]
        for i in range(len(t_grid))
    ])
    residual = ip_mean - g_null

    out = {
        "gamma_null": g_null,
        "residual": residual,
        "beta_per_layer": betas,
        "beta_median": med,
        "n_beta_fallback": n_fallback,
        "model": model,
        "t_eff_grid": t_grid,
        "ip_mean": ip_mean,
        "final_residual": float(residual[-1]),
        "max_abs_residual": float(np.nanmax(np.abs(residual))),
        "mean_residual": float(np.nanmean(residual)),
        # Sign convention spelled out in the artifact, because a residual
        # whose sign convention has to be recovered from source is a
        # result waiting to be misread.
        "sign_convention": "residual = observed - null; negative = less "
                           "clustered than identity-weight dynamics predict "
                           "= resistance",
    }

    if match_initial:
        g0 = float(ip_mean[0])
        g_m = np.array([
            gamma_null([t_grid[i]], n, betas[i], model=model, g0=g0)[0]
            for i in range(len(t_grid))
        ])
        out["gamma_null_matched"] = g_m
        out["residual_matched"] = ip_mean - g_m
        out["final_residual_matched"] = float((ip_mean - g_m)[-1])
        # If these two disagree materially, the gap between them is an
        # anisotropy effect (non-orthogonal embeddings), not resistance.
        out["anisotropy_gap"] = float(np.nanmean(np.abs(g_null - g_m)))

    return out


def collapse_fraction(result: dict, target: float = 0.9) -> dict:
    """
    How far along the null's own collapse the network actually gets.

    Two numbers, and they answer different questions:

      time_fraction   T_eff_total / t*(target)
                      "what fraction of the collapse TIME does the network
                      spend?" This is the P-γ2 quantity.

      gamma_fraction  gamma_null(T_eff_total) relative to target
                      "how far would identity-weight dynamics have gotten
                      in that time?" Because gamma is a sigmoid, a small
                      time fraction can still correspond to substantial
                      clustering, or to essentially none, depending on
                      where on the curve it lands. Reporting only the time
                      fraction invites reading a linear relationship into
                      a saturating one.
    """
    n = int(result.get("n_tokens", 0)) or 2
    beta = result.get("beta_median", 1.0)
    t_total = float(result["t_eff_grid"][-1])
    t_star = collapse_time(n, beta, target=target, model=result.get("model", "sa"))
    g_reached = float(result["gamma_null"][-1])
    return {
        "t_eff_total": t_total,
        "t_star": t_star,
        "time_fraction": t_total / t_star if np.isfinite(t_star) and t_star > 0 else float("nan"),
        "gamma_reached_by_null": g_reached,
        "gamma_fraction": g_reached / target if target else float("nan"),
        "ip_mean_final": float(result["ip_mean"][-1]),
    }


def adjudicate_p_gamma1(residuals_by_step: dict) -> dict:
    """
    P-γ1: the residual is near zero at step 0 and grows monotonically with
    training. Falsifier: the step-0 residual is already as large as the
    final one.

    residuals_by_step : {training_step (int): final_residual (float)}

    Deliberately reports the falsifier check separately from the
    monotonicity check. The prediction has two clauses and they can come
    apart — the residual can grow without being monotone, which would be a
    partial confirmation and should not be recorded as a pass.
    """
    steps = sorted(residuals_by_step)
    if len(steps) < 2:
        return {"verdict": "insufficient checkpoints", "n_steps": len(steps)}
    vals = [abs(residuals_by_step[s]) for s in steps]
    first, last = vals[0], vals[-1]

    grew = last > first
    monotone = all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
    falsified = first >= last

    if falsified:
        verdict = ("FALSIFIED — the step-0 residual is already as large as "
                   "the final one. The gap between the network and the "
                   "identity-weight null is architectural, not learned.")
    elif grew and monotone:
        verdict = "CONFIRMED — residual grows monotonically with training."
    elif grew:
        verdict = ("PARTIAL — the residual grows overall but not "
                   "monotonically. Report the non-monotonicity; Phase 1 "
                   "already found that nothing else in this sweep is "
                   "monotone either.")
    else:
        verdict = "UNCLEAR — residual neither grew nor shrank materially."

    return {"verdict": verdict, "steps": steps, "abs_residuals": vals,
            "first": first, "last": last, "grew": bool(grew),
            "monotone": bool(monotone), "falsified": bool(falsified)}


# ---------------------------------------------------------------------------
# The time-domain residual
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. gamma is a sigmoid asymptotic to 1, so once the null
# passes ~0.95 the vertical residual has almost no dynamic range: two
# trajectories that differ substantially in behaviour both read
# residual ~ 0 simply because both are pressed against the ceiling.
# Measured on synthetic runs, a perturbation that visibly changed the
# dynamics registered a final residual of +0.0000 for this reason.
#
# Inverting the null removes the compression. For each layer, ask: how much
# integration time would the identity-weight dynamics have needed to reach
# the inner product we actually observe? Compare that to how much time the
# network actually spent.
#
#     time_residual(l) = t_null^{-1}( ip_mean(l) )  -  T_eff(l)
#
#     negative  the network is BEHIND — it spent more time than the null
#               needs to reach this much clustering, i.e. resistance
#     ~ zero    on the null's own schedule
#     positive  ahead of the null
#
# Because the inverse of a saturating curve stretches near the ceiling, this
# measure keeps its resolution exactly where the vertical residual loses it.
# Report both: the vertical residual is directly interpretable in units of
# inner product, and the time residual is the one that stays honest late.

def time_residual_curve(ip_mean, t_eff_grid, n: int, beta,
                        model: str = "sa", g0: float = 0.0,
                        t_max: float = 30.0) -> dict:
    """
    Time-domain residual. Arguments as residual_curve.

    Layers whose observed ip_mean lies outside the null's reachable range
    are reported as nan rather than clipped — an observed value BELOW g0
    (the network de-clustered past its own starting point) has no
    corresponding null time at all, and clipping it to 0 would silently
    turn the strongest possible resistance signal into "on schedule".
    `n_unreachable` counts them, because on a trained model they are the
    interesting layers.
    """
    ip_mean = np.asarray(ip_mean, dtype=np.float64)
    t_grid = np.asarray(t_eff_grid, dtype=np.float64)
    betas = (np.full(len(ip_mean), float(beta)) if np.isscalar(beta)
             else np.asarray(beta, dtype=np.float64))
    med = float(np.nanmedian(betas)) if np.any(np.isfinite(betas)) else 1.0
    betas = np.where(np.isfinite(betas), betas, med)

    t_required = np.full(len(ip_mean), np.nan)
    for i, (target, b) in enumerate(zip(ip_mean, betas)):
        if not np.isfinite(target) or target <= g0 or target >= 1.0:
            continue
        t, g = integrate_gamma(n, b, t_max=t_max, dt=1e-3, model=model, g0=g0)
        tr = time_to_threshold(t, g, float(target))
        t_required[i] = tr if np.isfinite(tr) else np.nan

    resid = t_required - t_grid
    return {
        "t_required": t_required,
        "t_eff_grid": t_grid,
        "time_residual": resid,
        "final_time_residual": float(resid[-1]),
        "mean_time_residual": float(np.nanmean(resid)),
        "n_unreachable": int(np.sum(~np.isfinite(t_required))),
        "sign_convention": "time_residual = t_null_required - T_eff_spent; "
                           "negative = network spent more time than the null "
                           "needs for this much clustering = resistance",
    }
