"""
p1c_frames/beta_reduction.py — turning the beta_eff head-to-layer decision
from a blocker into a measurement.

THE BLOCKER

core/beta_eff.py returns beta per HEAD. The gamma_beta null needs one beta
per LAYER. Mean, median and attention-weighted give different answers, and
run_1c deliberately refuses to invent one — so 1c-A and 1c-B, the two
highest-value measurements in the project, were gated behind a choice that
had no principled basis and would have been made by whoever wrote the
driver.

WHY IT CANNOT BE DISSOLVED THE WAY THE CLUSTERER QUESTION WAS

The clusterer question dissolved because the quantity turned out not to
depend on the choice. This one does depend on it. Measured spread in
gamma_beta(T_eff) across beta in [0.5, 5]:

    n = 20,  T_eff = 3.0    0.9482 (b=0.5) .. 0.0577 (b=5)   spread 0.89
    n = 128, T_eff = 3.0    0.7545        .. 0.1386          spread 0.62
    n = 467, T_eff = 3.0    0.4610        .. 0.1989          spread 0.26

That is larger than any residual we could hope to measure. The reduction
matters.

THE WAY OUT: BRACKET IT

gamma_beta(t) is MONOTONE in beta at fixed t, verified numerically over
n in {5, 20, 64, 128, 467, 512}, t in [0, 8], beta in [0.01, 10] — 984,246
grid points per model:

    (SA)   monotone DECREASING in beta.  ZERO violations.
    (USA)  monotone INCREASING in beta.  ~35% of points increase.

The two respond in OPPOSITE directions, which is worth stating on its own:
the partition function is what reverses the sign, so using the surrogate as
a stand-in for the normalized dynamics gets the direction of the
beta-dependence backwards, not merely the magnitude. The envelope endpoints
below swap between models for exactly this reason.

Monotonicity means the per-head beta RANGE brackets the null without any
reduction being chosen:

    envelope(t) = [ gamma_{beta_max}(t), gamma_{beta_min}(t) ]      for SA

and the observed ip_mean either falls outside that envelope — in which case
the conclusion holds for EVERY reduction and the decision is moot — or
inside it, in which case the decision genuinely matters and must be made
deliberately, with the width of the envelope reported as the uncertainty it
is.

This is the same "refuse rather than degrade" pattern as elsewhere in the
phase: it does not answer the question, it makes the answer's dependence on
an unmade choice visible instead of burying it in a default.
"""

from __future__ import annotations

import numpy as np

from .gamma_ode import integrate_gamma


REDUCTIONS = ("mean", "median", "attention_weighted", "min", "max")


def reduce_beta(per_head, method: str = "median", weights=None) -> float:
    """
    One beta from a layer's per-head betas.

    attention_weighted : weight each head by `weights` — intended to be the
                         head's share of attention mass or output norm, so
                         a head that barely contributes does not pull the
                         null. Requires weights; raises without them rather
                         than silently degrading to the mean, which would
                         make the most-defensible reduction indistinguishable
                         from the laziest one in the output.

    Non-finite per-head betas are dropped and counted by `reduction_report`.
    A head whose beta_eff regression failed is not a head with beta = 0.
    """
    b = np.asarray(per_head, dtype=np.float64).ravel()
    m = np.isfinite(b)
    if not m.any():
        return float("nan")
    b = b[m]

    if method == "mean":
        return float(b.mean())
    if method == "median":
        return float(np.median(b))
    if method == "min":
        return float(b.min())
    if method == "max":
        return float(b.max())
    if method == "attention_weighted":
        if weights is None:
            raise ValueError(
                "attention_weighted requires per-head weights. Falling back "
                "to the mean would make the reduction unreadable from the "
                "output — pass the weights or choose another method."
            )
        w = np.asarray(weights, dtype=np.float64).ravel()[m]
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            return float("nan")
        return float((b * w).sum() / s)
    raise ValueError(f"method must be one of {REDUCTIONS}, got {method!r}")


def reduction_report(per_head, weights=None) -> dict:
    """Every reduction at once, plus the spread between them."""
    b = np.asarray(per_head, dtype=np.float64).ravel()
    finite = b[np.isfinite(b)]
    out = {"n_heads": int(b.size), "n_finite": int(finite.size),
           "n_dropped": int(b.size - finite.size), "values": {}}
    for m in REDUCTIONS:
        try:
            out["values"][m] = reduce_beta(b, m, weights=weights)
        except ValueError:
            out["values"][m] = float("nan")
    vals = [v for k, v in out["values"].items()
            if k not in ("min", "max") and np.isfinite(v)]
    out["reduction_spread"] = float(max(vals) - min(vals)) if len(vals) > 1 else 0.0
    out["beta_min"] = float(finite.min()) if finite.size else float("nan")
    out["beta_max"] = float(finite.max()) if finite.size else float("nan")
    out["beta_range"] = (out["beta_max"] - out["beta_min"]
                         if finite.size else float("nan"))
    return out


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------

def _gamma_on(t_grid, n, beta, model, g0):
    t_grid = np.asarray(t_grid, dtype=np.float64)
    t_max = max(float(np.nanmax(t_grid)) * 1.05, 1e-3)
    t, g = integrate_gamma(n, float(beta), t_max=t_max,
                           dt=min(1e-3, t_max / 2000), model=model, g0=g0)
    return np.interp(t_grid, t, g)


def beta_envelope(t_grid, n: int, beta_min: float, beta_max: float,
                  model: str = "sa", g0: float = 0.0,
                  verify: bool = True) -> dict:
    """
    The band the null occupies over a layer's whole per-head beta range.

    Endpoint order depends on the model, since (SA) and (USA) are monotone
    in OPPOSITE directions. Rather than hard-code that, both endpoints are
    computed and sorted pointwise — which is also what makes `verify`
    cheap: if the two curves cross, monotonicity has failed and the
    envelope is not a bound.

    verify : check that the two endpoint curves do not cross. They should
             not, given the numerical monotonicity result, but the result
             was established on a grid and this is a per-call guarantee
             rather than an appeal to it. A crossing raises.
    """
    lo_curve = _gamma_on(t_grid, n, beta_min, model, g0)
    hi_curve = _gamma_on(t_grid, n, beta_max, model, g0)

    if verify:
        diff = lo_curve - hi_curve
        pos = (diff > 1e-9).any()
        neg = (diff < -1e-9).any()
        if pos and neg:
            raise AssertionError(
                f"gamma curves at beta={beta_min} and beta={beta_max} cross "
                f"(model={model}, n={n}). Monotonicity in beta is what makes "
                f"this an envelope; without it the band is not a bound and "
                f"the reduction must be chosen explicitly."
            )

    lower = np.minimum(lo_curve, hi_curve)
    upper = np.maximum(lo_curve, hi_curve)
    return {
        "lower": lower, "upper": upper,
        "width": upper - lower,
        "mean_width": float(np.nanmean(upper - lower)),
        "max_width": float(np.nanmax(upper - lower)),
        "beta_min": float(beta_min), "beta_max": float(beta_max),
        "model": model, "n": int(n),
        # Which beta produced which edge, so the envelope can be read back
        # to a regime rather than only to a number.
        "upper_edge_beta": float(beta_min if model == "sa" else beta_max),
        "lower_edge_beta": float(beta_max if model == "sa" else beta_min),
    }


def envelope_verdict(ip_mean, envelope: dict, tol: float = 0.0) -> dict:
    """
    Does the reduction choice matter for THIS run?

    Returns per-layer position relative to the band and an overall verdict:

      below   observed is under the whole envelope at some layer — the
              network is less clustered than the identity-weight null for
              EVERY beta in the head range, so the resistance reading holds
              regardless of the reduction and the decision is moot.
      above   symmetric case.
      inside  observed sits within the band. The reduction genuinely
              matters here; report the band width as the uncertainty it is
              and make the choice deliberately.

    `frac_outside` is the fraction of layers where the verdict does not
    depend on the reduction, which is the number to quote when the answer
    is mixed — a run that is outside the band for the last eighteen layers
    and inside for the first six is not "inconclusive".
    """
    y = np.asarray(ip_mean, dtype=np.float64)
    lo = np.asarray(envelope["lower"], dtype=np.float64)
    hi = np.asarray(envelope["upper"], dtype=np.float64)
    k = min(len(y), len(lo))
    y, lo, hi = y[:k], lo[:k], hi[:k]

    below = y < lo - tol
    above = y > hi + tol
    inside = ~(below | above)

    if below.any() and not above.any():
        verdict = ("BELOW the envelope at some layer: the network is less "
                   "clustered than the identity-weight null for EVERY beta "
                   "in the per-head range. The reduction choice does not "
                   "change this conclusion.")
    elif above.any() and not below.any():
        verdict = ("ABOVE the envelope at some layer: more clustered than "
                   "the null for every beta in range. Reduction-independent.")
    elif inside.all():
        verdict = ("INSIDE the envelope everywhere: the residual's sign "
                   "depends on which beta the reduction picks. The choice "
                   "must be made deliberately and the band reported as "
                   "uncertainty — this is exactly the case run_1c refuses "
                   "to paper over with a default.")
    else:
        verdict = ("MIXED: outside the envelope at some layers, inside at "
                   "others. Quote frac_outside and the layer indices, not a "
                   "single verdict.")

    return {
        "verdict": verdict,
        "below": below.tolist(), "above": above.tolist(),
        "inside": inside.tolist(),
        "frac_outside": float((below | above).mean()),
        "layers_outside": [i for i in range(k) if below[i] or above[i]],
        "mean_band_width": float(np.nanmean(hi - lo)),
        "n_layers": int(k),
    }


def residual_bracket(ip_mean, envelope: dict) -> dict:
    """
    The residual computed at both envelope edges — a bracket rather than a
    point estimate.

    This is what sub-experiment B should report when the reduction has not
    been decided: not `residual`, but `[residual_min, residual_max]`, whose
    sign is unambiguous exactly when `envelope_verdict` says the run is
    outside the band. A single residual computed at an arbitrary reduction
    is a point estimate with an unstated error bar the size of the band.
    """
    y = np.asarray(ip_mean, dtype=np.float64)
    lo = np.asarray(envelope["lower"], dtype=np.float64)
    hi = np.asarray(envelope["upper"], dtype=np.float64)
    k = min(len(y), len(lo))
    r_at_upper = y[:k] - hi[:k]      # most negative residual
    r_at_lower = y[:k] - lo[:k]      # least negative
    return {
        "residual_min": r_at_upper,
        "residual_max": r_at_lower,
        "final_residual_min": float(r_at_upper[-1]),
        "final_residual_max": float(r_at_lower[-1]),
        "sign_unambiguous": bool(
            np.sign(r_at_upper[-1]) == np.sign(r_at_lower[-1])),
        "bracket_width_final": float(r_at_lower[-1] - r_at_upper[-1]),
    }
