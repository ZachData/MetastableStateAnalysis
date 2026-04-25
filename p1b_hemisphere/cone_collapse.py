"""
cone_collapse.py — Block 3 of Phase 1h.

Tests whether all tokens at a given layer lie in an open hemisphere
(cone-collapse regime) or span both hemispheres (split regime).

Formulation
-----------
Maximize γ subject to:  x_i · w ≥ γ  for all tokens i
                        w ∈ [-1, 1]^d  (L∞ ball)

γ* > +tol  →  cone_collapse: a half-space enclosing all tokens exists.
|γ*| ≤ tol →  borderline:    tokens exactly span a half-space boundary.
γ* < -tol  →  split:         no enclosing hemisphere exists.

High-dimensional note
---------------------
When d ≥ 256 the LP is large.  pca_n_components (default 64) reduces d
by projecting onto top-k PCA components.  The cone question is invariant
under orthogonal projections when the full token directions are preserved;
PCA is an approximation — set pca_n_components=None for exactness on
small n.

LP status note
--------------
scipy linprog status=1 (iteration limit) is treated as solved=True so the
best-known bound is propagated rather than silently discarded.  The
per-layer JSON now carries lp_at_limit=True for these entries so callers
can weight them appropriately.

Functions
---------
cone_margin_lp        : solve the LP for one layer.
classify_cone_regime  : map γ* to a regime label.
analyze_cone_collapse : full pipeline across all layers.
cone_collapse_to_json : JSON-serializable per-layer + summary.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


CONE_BORDERLINE_TOL = 1e-4


def cone_margin_lp(
    X: np.ndarray,
    pca_n_components: int | None = None,
) -> dict:
    """
    Solve the L∞-normalized cone-margin LP for one layer.

    Returns
    -------
    dict with:
      cone_margin       : float — γ* (positive = cone_collapse).
      normalized_margin : cone_margin / max(‖x_i‖₂).
      w_opt             : (d_eff,) witness vector.
      solved            : bool — False on solver error or infeasible status.
      lp_at_limit       : bool — True when solver hit iteration limit (status=1).
                          In this case solved=True but the margin may not be
                          globally optimal.
      status_msg        : str.
      d_eff             : int — effective dimension after any PCA.
    """
    n, d = X.shape
    if n == 0 or d == 0:
        return _failed_lp(d, "empty input")

    if pca_n_components is not None and pca_n_components < d:
        k = min(pca_n_components, n - 1, d)
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            X = X @ Vt[:k].T
            d = k
        except np.linalg.LinAlgError:
            return _failed_lp(d, "SVD failed during PCA reduction")

    A_ub         = np.zeros((n, d + 1), dtype=np.float64)
    A_ub[:, :d]  = -X
    A_ub[:, d]   =  1.0
    b_ub         = np.zeros(n, dtype=np.float64)
    c            = np.zeros(d + 1, dtype=np.float64)
    c[d]         = -1.0
    bounds       = [(-1.0, 1.0)] * d + [(None, None)]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method="highs", options={"disp": False})
    except Exception as exc:
        return _failed_lp(d, str(exc))

    at_limit = (res.status == 1)
    if res.status not in (0, 1):
        return _failed_lp(d, res.message, status_msg=res.message)

    gamma          = float(-res.fun)
    w_opt          = np.asarray(res.x[:d], dtype=np.float64)
    row_norms      = np.linalg.norm(X, axis=1)
    max_norm       = float(row_norms.max()) if row_norms.size else 1.0
    norm_margin    = gamma / max_norm if max_norm > 1e-12 else float("nan")

    return {
        "cone_margin":        gamma,
        "normalized_margin":  norm_margin,
        "w_opt":              w_opt,
        "solved":             True,
        "lp_at_limit":        at_limit,
        "status_msg":         res.message,
        "d_eff":              d,
    }


def _failed_lp(d: int, msg: str, status_msg: str | None = None) -> dict:
    return {
        "cone_margin":       float("nan"),
        "normalized_margin": float("nan"),
        "w_opt":             np.zeros(d, dtype=np.float64),
        "solved":            False,
        "lp_at_limit":       False,
        "status_msg":        status_msg or msg,
        "d_eff":             d,
    }


def classify_cone_regime(
    cone_margin: float,
    tol: float = CONE_BORDERLINE_TOL,
) -> str:
    """
    Map a cone_margin to a regime label.

    "cone_collapse" : γ* > +tol
    "borderline"    : |γ*| ≤ tol
    "split"         : γ* < -tol
    "invalid"       : γ* is nan
    """
    if cone_margin != cone_margin:  # nan check
        return "invalid"
    if cone_margin > tol:
        return "cone_collapse"
    if cone_margin < -tol:
        return "split"
    return "borderline"


def analyze_cone_collapse(
    activations: np.ndarray,
    valid: np.ndarray | None = None,
    pca_n_components: int | None = 64,
    tol: float = CONE_BORDERLINE_TOL,
) -> dict:
    """
    Run Block 3 across all layers.

    Parameters
    ----------
    activations      : (n_layers, n_tokens, d) — L2-normed.
    valid            : (n_layers,) bool mask.  If None, all layers valid.
    pca_n_components : PCA reduction before LP.  Default 64.
    tol              : borderline tolerance.

    Returns
    -------
    dict with arrays: cone_margin, normalized_margin, cone_regime, solved,
    lp_at_limit, plus scalar fields n_layers, n_tokens, d, pca_n_components, tol.
    """
    n_layers, n_tokens, d = activations.shape
    if valid is None:
        valid = np.ones(n_layers, dtype=bool)

    cone_margin  = np.full(n_layers, np.nan, dtype=np.float64)
    norm_margin  = np.full(n_layers, np.nan, dtype=np.float64)
    cone_regime  = np.full(n_layers, "invalid", dtype=object)
    solved       = np.zeros(n_layers, dtype=bool)
    lp_at_limit  = np.zeros(n_layers, dtype=bool)

    for L in range(n_layers):
        if not valid[L]:
            cone_regime[L] = "invalid"
            continue

        res             = cone_margin_lp(activations[L], pca_n_components=pca_n_components)
        cone_margin[L]  = res["cone_margin"]
        norm_margin[L]  = res["normalized_margin"]
        cone_regime[L]  = classify_cone_regime(res["cone_margin"], tol=tol)
        solved[L]       = res["solved"]
        lp_at_limit[L]  = res["lp_at_limit"]

    return {
        "cone_margin":       cone_margin,
        "normalized_margin": norm_margin,
        "cone_regime":       cone_regime,
        "solved":            solved,
        "lp_at_limit":       lp_at_limit,
        "n_layers":          n_layers,
        "n_tokens":          n_tokens,
        "d":                 d,
        "pca_n_components":  pca_n_components,
        "tol":               tol,
    }


def cone_collapse_to_json(result: dict) -> dict:
    """Flat per-layer + summary dict for the aggregator."""
    n      = result["n_layers"]
    regime = result["cone_regime"]

    per_layer = [
        {
            "layer":             L,
            "cone_regime":       str(regime[L]),
            "cone_margin":       _f(result["cone_margin"][L]),
            "normalized_margin": _f(result["normalized_margin"][L]),
            "solved":            bool(result["solved"][L]),
            "lp_at_limit":       bool(result["lp_at_limit"][L]),  # FIX: new field
        }
        for L in range(n)
    ]

    regime_counts: dict[str, int] = {}
    for r in regime:
        regime_counts[str(r)] = regime_counts.get(str(r), 0) + 1

    split_layers      = [L for L in range(n) if str(regime[L]) == "split"]
    first_split       = split_layers[0] if split_layers else None
    n_cc_before_split = (
        sum(1 for L in range(first_split) if str(regime[L]) == "cone_collapse")
        if first_split is not None else
        regime_counts.get("cone_collapse", 0)
    )

    valid_margins = result["cone_margin"][result["solved"]]
    summary = {
        "n_layers":                     n,
        "n_tokens":                     result["n_tokens"],
        "regime_counts":                regime_counts,
        "cone_collapse_fraction":
            float(regime_counts.get("cone_collapse", 0) / n) if n else 0.0,
        "split_fraction":
            float(regime_counts.get("split", 0) / n) if n else 0.0,
        "first_split_layer":            first_split,
        "n_cone_collapse_before_split": n_cc_before_split,
        "mean_cone_margin":  _f(valid_margins.mean()) if valid_margins.size else None,
        "min_cone_margin":   _f(valid_margins.min())  if valid_margins.size else None,
        "max_cone_margin":   _f(valid_margins.max())  if valid_margins.size else None,
        "n_lp_at_limit":     int(result["lp_at_limit"].sum()),
        "pca_n_components":  result["pca_n_components"],
        "tol":               result["tol"],
    }

    return {"per_layer": per_layer, "summary": summary}


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x