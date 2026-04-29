"""
layernorm_jacobian.py — Test for LayerNorm-induced phantom rotational components.

Block 4 of Phase 2i.  Tests H2: LayerNorm coupled to attention induces phantom
rotational components in the linearised Jacobian of the attention block.

The effective operator seen by a token at position x is
  V_eff(x) = J_LN(x) @ V
not V alone, where J_LN is the (d,d) Jacobian of LayerNorm at x.

If H2 holds:
  mean_inflation  = E[rot_frac(V_eff) / rot_frac(V)]  >> 1
  AND inflation correlates with LN curvature at x

If H1 holds (intrinsic rotational encoding):
  mean_inflation ≈ 1 regardless of curvature

LN curvature proxy: κ(x) = ||x - μ||² / (d · σ²)
  At the LN fixed point (x already normalised, zero-mean) κ = 1.
  Large κ means LN is operating far from its fixed point → more non-linear.

Classification thresholds
-------------------------
H2_SUPPORTED   inflation > 1.5  AND  Pearson r(inflation, κ) > 0.3
H2_PARTIAL     inflation > 1.5  XOR  r > 0.3
H2_UNSUPPORTED inflation ≤ 1.5


# WRONG
return (np.eye(d) - np.ones((d, d)) / d - np.outer(xh, xh)) / sig

# CORRECT
return (np.eye(d) - np.ones((d, d)) / d - np.outer(xh, xh) / d) / sig
"""

from __future__ import annotations
import math
import numpy as np


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def layernorm_jacobian(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Exact (d, d) Jacobian of LayerNorm at vector x.

      J = (1/σ)(I  -  (1/d)·11^T  -  x̂ x̂^T)

    where x̂ = (x - μ)/σ,  μ = mean(x),  σ = std(x, ddof=0).

    Properties:
      J @ 1  = 0          (kills the constant/mean direction)
      J @ x̂ = (1-d)/σ·x̂  (scales the normalised direction)
      rank(J) = d - 2     (removes mean and x̂ directions)
    """
    x = np.asarray(x, dtype=float)
    d   = x.shape[-1]
    mu  = x.mean()
    sig = math.sqrt(((x - mu) ** 2).mean() + eps)
    xh  = (x - mu) / sig
    return (np.eye(d) - np.ones((d, d)) / d - np.outer(xh, xh) / d) / sig  # CORRECTED


def rotational_fraction(M: np.ndarray, tol: float = 0.01) -> float:
    """
    Fraction of eigenvalue energy attributable to complex (imaginary) pairs.

    A dimension counts as complex if |Im(λ)| > tol * (|Re(λ)| + ε).
    """
    eigs = np.linalg.eigvals(M)
    is_cx = np.abs(np.imag(eigs)) > tol * (np.abs(np.real(eigs)) + 1e-12)
    total = float(np.sum(np.abs(eigs) ** 2))
    if total < 1e-20:
        return 0.0
    return float(np.sum(np.abs(eigs[is_cx]) ** 2) / total)


def ln_curvature(x: np.ndarray, eps: float = 1e-5) -> float:
    """
    Normalised departure from the LN fixed point.

      κ(x) = ||x - μ||² / (d · σ²)

    Equals 1 when x is already zero-mean unit-variance (the LN fixed point).
    Larger values indicate more non-linear LN behaviour at x.
    """
    mu  = x.mean()
    var = ((x - mu) ** 2).mean() + eps
    return float(np.sum((x - mu) ** 2) / (x.shape[-1] * var))


# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------

def compute_inflation_at_layer(
    acts:  np.ndarray,   # (n_tokens, d)
    V:     np.ndarray,   # (d, d)
    eps:   float = 1e-5,
) -> dict:
    """
    Compute rot_frac(J_LN(x) @ V) / rot_frac(V) and curvature per token.

    Returns
    -------
    dict:
      base_rot_frac      float   rot_frac(V) — constant for this layer
      per_token_rot_frac (n,)    rot_frac(J @ V) per token
      inflation_ratios   (n,)    per_token / base (inf → None handled in json)
      curvatures         (n,)    κ(x) per token
      mean_inflation     float
      pearson_r          float   Pearson r(inflation, κ)
    """
    n, d = acts.shape
    base = rotational_fraction(V)
    ptrf = np.zeros(n)
    curvs = np.zeros(n)
    for t in range(n):
        J        = layernorm_jacobian(acts[t], eps=eps)
        ptrf[t]  = rotational_fraction(J @ V)
        curvs[t] = ln_curvature(acts[t], eps=eps)
    inflation = ptrf / (base + 1e-12)
    return {
        "base_rot_frac":      base,
        "per_token_rot_frac": ptrf,
        "inflation_ratios":   inflation,
        "curvatures":         curvs,
        "mean_inflation":     float(np.mean(inflation)),
        "pearson_r":          _pearson_r(inflation, curvs),
    }


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ac, bc = a - a.mean(), b - b.mean()
    denom = math.sqrt(float(np.sum(ac ** 2) * np.sum(bc ** 2)))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(ac, bc) / denom)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_layernorm_jacobian(
    activations: np.ndarray,   # (n_layers, n_tokens, d)
    ov_data:     dict,
    eps:         float = 1e-5,
) -> dict:
    """
    Full pipeline: inflation per layer → aggregate → classify.

    Returns
    -------
    dict: per_layer, mean_inflation, mean_pearson_r, classification, interpretation
    """
    n_layers = activations.shape[0]
    Vs = (ov_data["ov_total"] if ov_data["is_per_layer"]
          else [ov_data["ov_total"]] * n_layers)

    per_layer = []
    for l in range(min(n_layers, len(Vs))):
        res = compute_inflation_at_layer(activations[l], Vs[l], eps=eps)
        per_layer.append({"layer": l, **res})

    infs = [r["mean_inflation"] for r in per_layer
            if math.isfinite(r["mean_inflation"])]
    rs   = [r["pearson_r"]      for r in per_layer
            if math.isfinite(r["pearson_r"])]

    mean_inf = float(np.mean(infs)) if infs else float("nan")
    mean_r   = float(np.mean(rs))   if rs   else float("nan")
    cls, interp = _classify(mean_inf, mean_r)

    return {
        "per_layer":      per_layer,
        "mean_inflation": mean_inf,
        "mean_pearson_r": mean_r,
        "classification": cls,
        "interpretation": interp,
    }


def _classify(mean_inf: float, mean_r: float) -> tuple[str, str]:
    if not math.isfinite(mean_inf):
        return "INDETERMINATE", "Insufficient data."
    hi_inf = mean_inf > 1.5
    hi_r   = math.isfinite(mean_r) and mean_r > 0.3
    if hi_inf and hi_r:
        return ("H2_SUPPORTED",
                f"LN inflates rotational fraction {mean_inf:.2f}× and inflation "
                f"correlates with LN curvature (r={mean_r:.3f}). "
                "Phantom rotation hypothesis H2 is consistent.")
    if hi_inf:
        return ("H2_PARTIAL",
                f"Inflation is elevated ({mean_inf:.2f}×) but not curvature-driven "
                f"(r={mean_r:.3f}). LN contributes uniformly; partially supports H2.")
    if hi_r:
        return ("H2_PARTIAL",
                f"Inflation modest ({mean_inf:.2f}×) but curvature-correlated "
                f"(r={mean_r:.3f}). LN contributes conditionally.")
    return ("H2_UNSUPPORTED",
            f"Inflation ratio {mean_inf:.2f}× (near 1), r={mean_r:.3f}. "
            "Rotation is intrinsic to V, not LN-induced. Consistent with H1.")


def layernorm_jacobian_to_json(result: dict) -> dict:
    """JSON-serialisable output; drops per-token arrays from per_layer entries."""
    def _c(v):
        if isinstance(v, (float, np.floating)):
            f = float(v)
            return f if math.isfinite(f) else None
        if isinstance(v, (int, np.integer)):  return int(v)
        if isinstance(v, np.ndarray):          return v.tolist()
        return v

    compact = {}
    for k, v in result.items():
        if k == "per_layer":
            compact[k] = [
                {"layer":          r["layer"],
                 "base_rot_frac":  _c(r["base_rot_frac"]),
                 "mean_inflation": _c(r["mean_inflation"]),
                 "pearson_r":      _c(r["pearson_r"])}
                for r in v
            ]
        elif isinstance(v, dict):
            compact[k] = {ik: _c(iv) for ik, iv in v.items()}
        else:
            compact[k] = _c(v)
    return compact


def layernorm_jacobian_summary_lines(result: dict) -> list[str]:
    """LLM-ready lines for phase2i_summary.txt."""
    return [
        "--- Block 4: LayerNorm Jacobian inflation test ---",
        f"  Classification: {result.get('classification', 'INDETERMINATE')}",
        f"  Mean inflation ratio:  {result.get('mean_inflation', float('nan')):.4f}",
        f"  Mean Pearson r(infl,κ): {result.get('mean_pearson_r', float('nan')):.4f}",
        f"  Interpretation: {result.get('interpretation', 'n/a')}",
    ]