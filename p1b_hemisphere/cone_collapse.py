"""
cone_collapse.py — Block 3 of Phase 1h.

Per-layer LP test for the paper's hemisphere condition vs. the empirical
split regime observed in Phase 1.

Theory
------
Geshkovski et al. (Theorem 6.3, Lemma 6.4) prove exponential cone-collapse
when all tokens start inside a single open hemisphere:

    ∃ w ∈ S^{d-1}  s.t.  min_i <w, x_i> > 0

The Phase 1 Fiedler finding (dominant bipartition, antipodal centroids) is
*not* automatically the opposite geometry.  These two conditions measure
different things:

  Fiedler bipartition  (Block 0) — measures internal density structure.
                                   Two dense, internally-compact groups
                                   separated by the sign boundary of v₂.
  Cone-collapse test   (Block 3) — measures global containment.  Do all
                                   tokens fit inside one open hemisphere
                                   under ANY unit vector w?

A key geometric fact: two clusters at angle θ < π apart ALWAYS admit a
common enclosing hemisphere (bisect the arc between the cluster centroids).
The paper's cone-collapse theorem can apply even when a strong Fiedler
bipartition exists, as long as both Fiedler clusters sit within the same
half-space along some other direction.

The split regime (γ* < 0) requires the token cloud to *surround* the
origin — i.e., the convex hull of the tokens contains 0.  This is strictly
stronger than having two antipodal-looking clusters:

  • Two tight clusters at 120° separation → cone_collapse (γ* > 0).
  • Two tight clusters at 170° separation → cone_collapse (γ* > 0,
    but small; bisecting hemisphere is tight).
  • Tokens at exactly ±θ with enough spread to span >180° → split.
  • High-dimensional token cloud: even if Fiedler centroids are near π,
    the tokens may not surround the origin along all directions.

Therefore Block 3 is a direct test of a stronger condition than Block 0.
Finding split layers is positive evidence against the paper's theorem
at those layers.  Finding cone_collapse does not contradict the Fiedler
bipartition — the bipartition can coexist with cone containment.

LP formulation
--------------
We solve the linear program:

    maximize   γ
    subject to  x_i^T w ≥ γ,   i = 1 … n
                ‖w‖_∞ ≤ 1

    variables: w ∈ R^d, γ ∈ R
    constraints: x_i^T w - γ ≥ 0 for all i; -1 ≤ w_j ≤ 1 for all j.

The optimal value γ* distinguishes three regimes:

    γ* > +tol   →  cone_collapse:  all tokens in one hemisphere under w*.
    |γ*| ≤ tol  →  borderline:     tokens exactly span a half-space boundary.
    γ* < -tol   →  split:          no enclosing hemisphere exists.

The L∞ ball replaces the L2 unit sphere for the LP.  The sign of γ* is
invariant to this choice.  The magnitude of γ* is on the L∞ scale.

High-dimensional note
---------------------
When d is large (≥ 256), the LP has d + 1 variables and n + 2d constraints.
For typical n ≈ 50–500 and d ≈ 768, this is manageable with HiGHS (scipy ≥
1.7) but can be slow.  The optional pca_n_components argument reduces d by
projecting activations onto the top-k PCA components before solving.  The
cone-membership question is invariant under orthogonal projections only when
the projection preserves all token directions; PCA is an approximation.  Use
pca_n_components=None when exact results matter and n is small.

Functions
---------
cone_margin_lp       : solve the LP for one layer, return γ* and w*.
classify_cone_regime : map γ* to "cone_collapse" / "borderline" / "split".
analyze_cone_collapse: full pipeline across all layers.
cone_collapse_to_json: JSON-serializable per-layer + summary.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


# Regime boundary: |γ*| below this → "borderline".
CONE_BORDERLINE_TOL = 1e-4


def cone_margin_lp(
    X: np.ndarray,
    pca_n_components: int | None = None,
) -> dict:
    """
    Solve the L∞-normalized cone-margin LP for one layer.

    Parameters
    ----------
    X                : (n_tokens, d) — L2-normed activation matrix.
    pca_n_components : if not None, project X onto its top-k PCA
                       components before solving.  Reduces LP size at
                       the cost of exactness.

    Returns
    -------
    dict with:
      cone_margin       : float — γ*, the optimal objective value.
                          Positive → cone_collapse, negative → split.
      normalized_margin : cone_margin / max(‖x_i‖₂).  Equals cone_margin
                          for L2-normed inputs.
      w_opt             : (d_eff,) the witness vector (in the PCA subspace
                          if pca_n_components is set, else in the original
                          space).
      solved            : bool — False if the LP solver hit an error or
                          returned an infeasible/unbounded status.
      status_msg        : str — scipy linprog message.
      d_eff             : int — effective dimension used (after any PCA).
    """
    n, d = X.shape

    if n == 0 or d == 0:
        return _failed_lp(d, "empty input")

    # Optional PCA reduction.
    if pca_n_components is not None and pca_n_components < d:
        k = min(pca_n_components, n - 1, d)
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            X = X @ Vt[:k].T    # (n, k)
            d = k
        except np.linalg.LinAlgError:
            return _failed_lp(d, "SVD failed during PCA reduction")

    # LP formulation:
    # Variables:  z = [w (d,), γ (1,)]   total d+1 variables.
    # Maximize γ  ↔  minimize -γ.
    # Constraint Xw ≥ γ·1  ↔  -Xw + γ·1 ≤ 0
    #   A_ub[i, :] = [-x_i^T  |  1]  (n rows)
    # Bounds: -1 ≤ w_j ≤ 1; γ unbounded.

    # A_ub has shape (n, d+1).
    A_ub = np.zeros((n, d + 1), dtype=np.float64)
    A_ub[:, :d] = -X
    A_ub[:, d]  = 1.0
    b_ub = np.zeros(n, dtype=np.float64)

    # Objective: minimize -γ (column d of z).
    c = np.zeros(d + 1, dtype=np.float64)
    c[d] = -1.0

    # Bounds: w_j ∈ [-1, 1], γ ∈ (-inf, +inf).
    bounds = [(-1.0, 1.0)] * d + [(None, None)]

    try:
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={"disp": False},
        )
    except Exception as exc:
        return _failed_lp(d, str(exc))

    if res.status not in (0, 1):
        # status 0 = optimal; 1 = iteration limit (use best known).
        return _failed_lp(d, res.message, status_msg=res.message)

    gamma   = float(-res.fun)          # negate back from minimization
    w_opt   = np.asarray(res.x[:d], dtype=np.float64)
    norm_w  = np.linalg.norm(w_opt)

    # Normalized margin: for L2-normed inputs, max ||x_i|| = 1.
    row_norms = np.linalg.norm(X, axis=1)
    max_norm  = float(row_norms.max()) if row_norms.size else 1.0
    normalized_margin = gamma / max_norm if max_norm > 1e-12 else float("nan")

    return {
        "cone_margin":        gamma,
        "normalized_margin":  normalized_margin,
        "w_opt":              w_opt,
        "solved":             True,
        "status_msg":         res.message,
        "d_eff":              d,
    }


def _failed_lp(d: int, msg: str, status_msg: str | None = None) -> dict:
    return {
        "cone_margin":       float("nan"),
        "normalized_margin": float("nan"),
        "w_opt":             np.zeros(d, dtype=np.float64),
        "solved":            False,
        "status_msg":        status_msg or msg,
        "d_eff":             d,
    }


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_cone_regime(
    cone_margin: float,
    tol: float = CONE_BORDERLINE_TOL,
) -> str:
    """
    Map a cone_margin to a regime label.

    Returns
    -------
    "cone_collapse"  : γ* > +tol — paper's theorem applies.
    "borderline"     : |γ*| ≤ tol — tokens exactly span a half-space.
    "split"          : γ* < -tol — no enclosing hemisphere exists.
    "invalid"        : γ* is nan (LP failed).
    """
    if cone_margin != cone_margin:   # nan check
        return "invalid"
    if cone_margin > tol:
        return "cone_collapse"
    if cone_margin < -tol:
        return "split"
    return "borderline"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

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
    valid            : (n_layers,) bool mask.  If None, all layers treated
                       as valid.
    pca_n_components : PCA reduction before LP.  Default 64 — trades
                       exactness for speed on high-d models.  Set to None
                       for exact computation (slow for d ≥ 256).
    tol              : borderline tolerance.

    Returns
    -------
    dict with:
      cone_margin       (n_layers,) float
      normalized_margin (n_layers,) float
      cone_regime       (n_layers,) str
      solved            (n_layers,) bool
    """
    n_layers, n_tokens, d = activations.shape
    if valid is None:
        valid = np.ones(n_layers, dtype=bool)

    cone_margin   = np.full(n_layers, np.nan, dtype=np.float64)
    norm_margin   = np.full(n_layers, np.nan, dtype=np.float64)
    cone_regime   = np.full(n_layers, "invalid", dtype=object)
    solved        = np.zeros(n_layers, dtype=bool)

    for L in range(n_layers):
        if not valid[L]:
            cone_regime[L] = "invalid"
            continue

        res = cone_margin_lp(activations[L], pca_n_components=pca_n_components)
        cone_margin[L] = res["cone_margin"]
        norm_margin[L] = res["normalized_margin"]
        cone_regime[L] = classify_cone_regime(res["cone_margin"], tol=tol)
        solved[L]      = res["solved"]

    return {
        "cone_margin":        cone_margin,
        "normalized_margin":  norm_margin,
        "cone_regime":        cone_regime,
        "solved":             solved,
        "n_layers":           n_layers,
        "n_tokens":           n_tokens,
        "d":                  d,
        "pca_n_components":   pca_n_components,
        "tol":                tol,
    }


# ---------------------------------------------------------------------------
# JSON serializer
# ---------------------------------------------------------------------------

def cone_collapse_to_json(result: dict) -> dict:
    """Flat per-layer + summary dict for the aggregator."""
    n = result["n_layers"]

    per_layer = []
    for L in range(n):
        per_layer.append({
            "layer":              L,
            "cone_regime":        str(result["cone_regime"][L]),
            "cone_margin":        _f(result["cone_margin"][L]),
            "normalized_margin":  _f(result["normalized_margin"][L]),
            "solved":             bool(result["solved"][L]),
        })

    regime = result["cone_regime"]
    regime_counts: dict[str, int] = {}
    for r in regime:
        regime_counts[str(r)] = regime_counts.get(str(r), 0) + 1

    # First split layer: earliest L where regime is "split".
    split_layers = [L for L in range(n) if str(regime[L]) == "split"]
    first_split  = split_layers[0] if split_layers else None

    # Crossover analysis: how many layers are cone_collapse before first split?
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
        "mean_cone_margin":             _f(valid_margins.mean()) if valid_margins.size else None,
        "min_cone_margin":              _f(valid_margins.min())  if valid_margins.size else None,
        "max_cone_margin":              _f(valid_margins.max())  if valid_margins.size else None,
        "pca_n_components":             result["pca_n_components"],
        "tol":                          result["tol"],
    }

    return {"per_layer": per_layer, "summary": summary}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x
