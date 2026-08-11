"""
cone_collapse.py — Block 3 of Phase 1b.

Tests whether all tokens at a given layer lie in an open hemisphere
(cone-collapse regime) or span both hemispheres (split regime), and — new in
this revision — whether that answer says anything about the model.

Formulation
-----------
Maximize gamma subject to:  x_i . w >= gamma  for all tokens i
                            w in [-1, 1]^d  (L-infinity ball)

gamma* > +tol   ->  cone_collapse: a half-space enclosing all tokens exists.
|gamma*| <= tol ->  borderline:    tokens exactly span a half-space boundary.
gamma* < -tol   ->  split:         no enclosing hemisphere exists.

Three corrections to the first version, all of which change how the result
should be read
--------------------------------------------------------------------------

**1. The binary regime was close to free, and was reported as a finding.**

n points in general position in d dimensions admit a strictly separating
witness whenever they fail to positively span the space, which for n not
much larger than d is the overwhelmingly typical case. The first version ran
with `pca_n_components=64` on prompts of order 100-200 tokens and reported
"100% cone-collapse, every model, every layer" as an empirical result. It is
not clear how much of that is transformer geometry and how much is n vs
d_eff, and the run as constructed could not tell the difference.

`analyze_cone_collapse` now takes `n_null` and computes, per layer, the same
margin on two matched controls:

  - shuffled-dimension (core/nulls.py): each feature dimension independently
    permuted across tokens, re-normalised to the sphere. Same per-dimension
    marginals, no cross-token geometry. Answers "is the containment more
    than the marginals?"
  - uniform-sphere: n points drawn uniformly on S^{d-1} at matched (n, d).
    The pure dimension-counting control. Answers "is the containment more
    than n vs d?"

The reported quantity is then a z-score against each null, not a boolean.
The regime label is retained because downstream code and the existing STATUS
table consume it, but `z_vs_uniform` is what a falsification table should
adjudicate against — a 100% cone-collapse fraction that sits at the uniform
null's median is a statement about dimension, not about transformers.

**2. PCA is safe in one direction only, and the docstring claimed both.**

The old note said the cone question "is invariant under orthogonal
projections". It is not. What is true, and is now documented and enforced:

  - A witness found in the reduced space lifts *exactly* to the full space.
    With X_r = X @ Vt[:k].T and w_r feasible there, w = Vt[:k].T @ w_r gives
    X @ w = X_r @ w_r, so every inner product is preserved. A cone_collapse
    verdict under PCA is therefore sound as it stands.
  - The converse fails: the component of a full-space witness orthogonal to
    the retained subspace is discarded, so a split verdict under PCA may be
    an artifact of the projection.

`escalate_on_split=True` (default) re-solves at full d whenever the reduced
problem reports split or borderline, and sets `escalated=True` on that layer.
The existing 100%-cone-collapse result stands unchanged under this; any
future split does not get reported without the full-d solve behind it.

**3. Which tokens bind the constraint was never recorded.**

On GPT-NeoX the position-0 attention sink is a high-norm outlier and a
plausible sole determinant of the enclosing half-space. `binding_tokens` now
records the indices tight at the optimum, and `drop_indices` lets a caller
re-run without them. Running both is what turns "a cone exists" into "a cone
exists, and it is / is not held up by one token".

LP status note
--------------
scipy linprog status=1 (iteration limit) is treated as solved=True so the
best-known bound is propagated rather than silently discarded. Per-layer
JSON carries lp_at_limit=True for these entries.

Functions
---------
cone_margin_lp        : solve the LP for one layer.
classify_cone_regime  : map gamma* to a regime label.
normalized_margin_of  : (n, d) -> normalized margin, the metric_fn nulls use.
cone_margin_nulls     : matched null margins for one layer.
analyze_cone_collapse : full pipeline across all layers.
cone_collapse_to_json : JSON-serializable per-layer + summary.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from core.nulls import shuffled_dimension_null, sigma_from_null


CONE_BORDERLINE_TOL = 1e-4

# Fraction of the optimal margin within which a constraint counts as binding.
BINDING_REL_TOL = 1e-6


def cone_margin_lp(
    X: np.ndarray,
    pca_n_components: int | None = None,
    drop_indices=None,
) -> dict:
    """
    Solve the L-infinity-normalized cone-margin LP for one layer.

    Parameters
    ----------
    X                : (n_tokens, d) activations.
    pca_n_components : project onto this many top right-singular directions
                       before solving. See the module docstring for the
                       one-directional soundness of this. None = exact.
    drop_indices     : token indices to exclude before solving — e.g. the
                       position-0 sink under a pos0_policy="dropped" frame.

    Returns
    -------
    dict with:
      cone_margin       : gamma* (positive = cone_collapse).
      normalized_margin : cone_margin / max(||x_i||_2). The scale-free
                          quantity; compare this across layers, models and
                          checkpoints, never cone_margin.
      w_opt             : (d_eff,) witness vector.
      binding_tokens    : ORIGINAL token indices whose constraint is tight
                          at the optimum.
      n_binding         : len(binding_tokens).
      degenerate_w      : True when the LP returned the trivial w=0.
      solved            : False on solver error or infeasible status.
      lp_at_limit       : True when the solver hit its iteration limit.
      status_msg        : str.
      d_eff             : effective dimension after any PCA.
      n_used            : rows actually constrained.
      kept_indices      : original indices of the rows used.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        return _failed_lp(0, "expected a 2-D (n_tokens, d) array")
    n_full, d_full = X.shape

    kept = np.arange(n_full)
    if drop_indices is not None and len(list(drop_indices)):
        mask = np.ones(n_full, dtype=bool)
        mask[np.asarray(list(drop_indices), dtype=int)] = False
        X = X[mask]
        kept = kept[mask]

    n, d = X.shape
    if n == 0 or d == 0:
        return _failed_lp(d, "empty input", kept_indices=kept)

    if pca_n_components is not None and pca_n_components < d:
        k = int(min(pca_n_components, max(n - 1, 1), d))
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            X = X @ Vt[:k].T
            d = k
        except np.linalg.LinAlgError:
            return _failed_lp(d, "SVD failed during PCA reduction",
                              kept_indices=kept)

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
        return _failed_lp(d, str(exc), kept_indices=kept)

    at_limit = (res.status == 1)
    if res.status not in (0, 1):
        return _failed_lp(d, res.message, status_msg=res.message,
                          kept_indices=kept)

    gamma     = float(-res.fun)
    w_opt     = np.asarray(res.x[:d], dtype=np.float64)
    row_norms = np.linalg.norm(X, axis=1)
    max_norm  = float(row_norms.max()) if row_norms.size else 1.0

    # Degenerate-w detection: the LP can achieve gamma=0 with w=0, which
    # trivially satisfies every constraint while supplying no separating
    # direction. That happens exactly when the token set spans both
    # hemispheres. Push gamma below -tol so classify_cone_regime says
    # "split" rather than "borderline".
    degenerate_w = bool(np.linalg.norm(w_opt) < 1e-4
                        and gamma < CONE_BORDERLINE_TOL)
    if degenerate_w:
        gamma = -(CONE_BORDERLINE_TOL + 1e-6)
        binding_orig = []
    else:
        slack    = (X @ w_opt) - gamma
        tol_bind = BINDING_REL_TOL * max(1.0, abs(gamma))
        binding  = np.where(np.abs(slack) <= tol_bind)[0]
        binding_orig = kept[binding].tolist()

    return {
        "cone_margin":       gamma,
        "normalized_margin": gamma / max_norm if max_norm > 1e-12 else float("nan"),
        "w_opt":             w_opt,
        "binding_tokens":    binding_orig,
        "n_binding":         len(binding_orig),
        "degenerate_w":      degenerate_w,
        "solved":            True,
        "lp_at_limit":       at_limit,
        "status_msg":        res.message,
        "d_eff":             int(d),
        "n_used":            int(n),
        "kept_indices":      kept,
    }


def _failed_lp(d: int, msg: str, status_msg=None, kept_indices=None) -> dict:
    return {
        "cone_margin":       float("nan"),
        "normalized_margin": float("nan"),
        "w_opt":             np.zeros(max(int(d), 0), dtype=np.float64),
        "binding_tokens":    [],
        "n_binding":         0,
        "degenerate_w":      False,
        "solved":            False,
        "lp_at_limit":       False,
        "status_msg":        status_msg or msg,
        "d_eff":             int(d),
        "n_used":            0,
        "kept_indices":      np.arange(0) if kept_indices is None else kept_indices,
    }


def classify_cone_regime(cone_margin: float, tol: float = CONE_BORDERLINE_TOL) -> str:
    """
    Map a cone_margin to a regime label.

    "cone_collapse" : gamma* > +tol
    "borderline"    : |gamma*| <= tol
    "split"         : gamma* < -tol
    "invalid"       : gamma* is nan
    """
    if cone_margin != cone_margin:  # nan check
        return "invalid"
    if cone_margin > tol:
        return "cone_collapse"
    if cone_margin < -tol:
        return "split"
    return "borderline"


# ---------------------------------------------------------------------------
# Null margins
# ---------------------------------------------------------------------------

def normalized_margin_of(X: np.ndarray, pca_n_components: int | None = 64) -> float:
    """
    (n_tokens, d) -> normalized cone margin, or nan if the LP fails.

    This is the `metric_fn` handed to core.nulls.shuffled_dimension_null, so
    the null draws are solved through the identical LP path (PCA setting
    included) as the observation. Like-for-like is the whole point: a null
    computed at a different d_eff would answer a different question.
    """
    r = cone_margin_lp(X, pca_n_components=pca_n_components)
    return float(r["normalized_margin"]) if r["solved"] else float("nan")


def uniform_sphere_null(
    n: int,
    d: int,
    n_draws: int,
    pca_n_components: int | None = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Normalized cone margins for `n_draws` sets of n uniform points on S^{d-1}.

    The dimension-counting control. There is no corresponding helper in
    core/nulls.py because this is not a shuffle of observed data — it is a
    from-scratch geometric baseline, and its whole purpose is to carry none
    of the observed cloud's structure, not even its marginals.
    """
    rng = np.random.default_rng() if rng is None else rng
    out = np.full(int(n_draws), np.nan, dtype=np.float64)
    for i in range(int(n_draws)):
        Z = rng.standard_normal((n, d))
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = Z / np.maximum(norms, 1e-12)
        out[i] = normalized_margin_of(Z, pca_n_components=pca_n_components)
    return out


def cone_margin_nulls(
    X: np.ndarray,
    n_null: int = 20,
    pca_n_components: int | None = 64,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Matched null distributions for one layer's normalized cone margin.

    Returns raw null arrays plus sigma_from_null's summary against each.
    """
    rng  = np.random.default_rng() if rng is None else rng
    X    = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    observed = normalized_margin_of(X, pca_n_components=pca_n_components)

    shuffled = shuffled_dimension_null(
        X,
        metric_fn=lambda Z: normalized_margin_of(Z, pca_n_components=pca_n_components),
        n_shuffles=int(n_null),
        renormalize=True,
        rng=rng,
    )
    uniform = uniform_sphere_null(n, d, int(n_null),
                                  pca_n_components=pca_n_components, rng=rng)

    shuffled_ok = shuffled[np.isfinite(shuffled)]
    uniform_ok  = uniform[np.isfinite(uniform)]

    # A uniform null at matched (n, d) is frequently DEGENERATE: every draw
    # lands on the same sentinel margin because every draw positively spans
    # the space, so null_std is ~0 and z_score is (correctly) nan. That is
    # not a failure of the control, it is the control's answer — but a nan
    # z-score is useless in a table, so the fraction of null draws that are
    # themselves cone-collapsed is reported alongside. "Observed collapses,
    # 0/N matched draws collapse" is the readable form of a nan z here.
    def _cone_frac(vals):
        v = vals[np.isfinite(vals)]
        return float((v > CONE_BORDERLINE_TOL).mean()) if v.size else None

    return {
        "observed":      observed,
        "shuffled_null": shuffled,
        "uniform_null":  uniform,
        "shuffled_cone_fraction": _cone_frac(shuffled),
        "uniform_cone_fraction":  _cone_frac(uniform),
        "vs_shuffled":   (sigma_from_null(observed, shuffled_ok)
                          if shuffled_ok.size else _empty_sigma(observed)),
        "vs_uniform":    (sigma_from_null(observed, uniform_ok)
                          if uniform_ok.size else _empty_sigma(observed)),
        "n_null":        int(n_null),
    }


def _empty_sigma(observed: float) -> dict:
    return {
        "observed": float(observed), "null_mean": float("nan"),
        "null_std": float("nan"), "z_score": float("nan"),
        "percentile": float("nan"), "n_null": 0,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_cone_collapse(
    activations: np.ndarray,
    valid: np.ndarray | None = None,
    pca_n_components: int | None = 64,
    tol: float = CONE_BORDERLINE_TOL,
    escalate_on_split: bool = True,
    n_null: int = 0,
    null_layers: list | None = None,
    drop_indices=None,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Run Block 3 across all layers.

    Parameters
    ----------
    activations       : (n_layers, n_tokens, d) — L2-normed.
    valid             : (n_layers,) bool mask. If None, all layers valid.
    pca_n_components  : PCA reduction before the LP. Default 64.
    tol               : borderline tolerance.
    escalate_on_split : re-solve at full d whenever the reduced problem
                        reports anything other than cone_collapse. See the
                        module docstring for why the asymmetry is correct.
    n_null            : null replicates per analyzed layer. 0 disables. The
                        observation LP is cheap; the nulls are 2*n_null LPs
                        per layer, which is why this defaults off and
                        `null_layers` exists to run it on a subset.
    null_layers       : which layers get nulls. None with n_null>0 means
                        every valid layer.
    drop_indices      : token indices excluded from every solve.
    """
    activations = np.asarray(activations)
    n_layers, n_tokens, d = activations.shape
    if valid is None:
        valid = np.ones(n_layers, dtype=bool)
    rng = np.random.default_rng() if rng is None else rng

    cone_margin = np.full(n_layers, np.nan, dtype=np.float64)
    norm_margin = np.full(n_layers, np.nan, dtype=np.float64)
    cone_regime = np.full(n_layers, "invalid", dtype=object)
    solved      = np.zeros(n_layers, dtype=bool)
    lp_at_limit = np.zeros(n_layers, dtype=bool)
    escalated   = np.zeros(n_layers, dtype=bool)
    n_binding   = np.zeros(n_layers, dtype=np.int32)
    d_eff       = np.zeros(n_layers, dtype=np.int32)
    binding: dict = {}

    for L in range(n_layers):
        if not valid[L]:
            cone_regime[L] = "invalid"
            continue

        res = cone_margin_lp(activations[L],
                             pca_n_components=pca_n_components,
                             drop_indices=drop_indices)
        regime = classify_cone_regime(res["cone_margin"], tol=tol)

        # Reduced-space non-collapse may be a projection artifact; a
        # reduced-space collapse verdict never is. Escalate only the
        # direction that can lie.
        if (escalate_on_split
                and pca_n_components is not None
                and res["solved"]
                and regime in ("split", "borderline")):
            full = cone_margin_lp(activations[L], pca_n_components=None,
                                  drop_indices=drop_indices)
            if full["solved"]:
                res    = full
                regime = classify_cone_regime(res["cone_margin"], tol=tol)
                escalated[L] = True

        cone_margin[L] = res["cone_margin"]
        norm_margin[L] = res["normalized_margin"]
        cone_regime[L] = regime
        solved[L]      = res["solved"]
        lp_at_limit[L] = res["lp_at_limit"]
        n_binding[L]   = res["n_binding"]
        d_eff[L]       = res["d_eff"]
        binding[L]     = res["binding_tokens"]

    nulls: dict = {}
    if n_null and n_null > 0:
        targets = (null_layers if null_layers is not None
                   else [L for L in range(n_layers) if valid[L]])
        for L in targets:
            L = int(L)
            if L < 0 or L >= n_layers or not valid[L]:
                continue
            nulls[L] = cone_margin_nulls(activations[L], n_null=n_null,
                                         pca_n_components=pca_n_components,
                                         rng=rng)

    return {
        "cone_margin":       cone_margin,
        "normalized_margin": norm_margin,
        "cone_regime":       cone_regime,
        "solved":            solved,
        "lp_at_limit":       lp_at_limit,
        "escalated":         escalated,
        "n_binding":         n_binding,
        "binding_tokens":    binding,
        "d_eff":             d_eff,
        "nulls":             nulls,
        "n_layers":          n_layers,
        "n_tokens":          n_tokens,
        "d":                 d,
        "pca_n_components":  pca_n_components,
        "escalate_on_split": escalate_on_split,
        "dropped_indices":   ([int(i) for i in drop_indices]
                              if drop_indices is not None else []),
        "tol":               tol,
    }


def cone_collapse_to_json(result: dict) -> dict:
    """Flat per-layer + summary dict for the aggregator."""
    n      = result["n_layers"]
    regime = result["cone_regime"]
    nulls  = result.get("nulls", {})

    per_layer = []
    for L in range(n):
        entry = {
            "layer":               L,
            "cone_regime":         str(regime[L]),
            "cone_margin":         _f(result["cone_margin"][L]),
            "normalized_margin":   _f(result["normalized_margin"][L]),
            "solved":              bool(result["solved"][L]),
            "lp_at_limit":         bool(result["lp_at_limit"][L]),
            "escalated_to_full_d": bool(result["escalated"][L]),
            "d_eff":               int(result["d_eff"][L]),
            "n_binding":           int(result["n_binding"][L]),
            "binding_tokens":      list(result.get("binding_tokens", {}).get(L, [])),
        }
        if L in nulls:
            nl = nulls[L]
            entry["z_vs_shuffled"]      = _f(nl["vs_shuffled"].get("z_score"))
            entry["z_vs_uniform"]       = _f(nl["vs_uniform"].get("z_score"))
            entry["pct_vs_uniform"]     = _f(nl["vs_uniform"].get("percentile"))
            entry["null_mean_shuffled"] = _f(nl["vs_shuffled"].get("null_mean"))
            entry["null_mean_uniform"]  = _f(nl["vs_uniform"].get("null_mean"))
            entry["shuffled_cone_fraction"] = nl.get("shuffled_cone_fraction")
            entry["uniform_cone_fraction"]  = nl.get("uniform_cone_fraction")
        per_layer.append(entry)

    regime_counts: dict = {}
    for r in regime:
        regime_counts[str(r)] = regime_counts.get(str(r), 0) + 1

    split_layers      = [L for L in range(n) if str(regime[L]) == "split"]
    first_split       = split_layers[0] if split_layers else None
    n_cc_before_split = (
        sum(1 for L in range(first_split) if str(regime[L]) == "cone_collapse")
        if first_split is not None else regime_counts.get("cone_collapse", 0)
    )

    valid_margins = result["cone_margin"][result["solved"]]
    valid_norm    = result["normalized_margin"][result["solved"]]
    valid_norm    = valid_norm[np.isfinite(valid_norm)]

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
        # normalized_margin is the reportable quantity — cone_margin carries
        # whatever scale the frame left behind and is not comparable across
        # models or checkpoints.
        "mean_normalized_margin": _f(valid_norm.mean()) if valid_norm.size else None,
        "min_normalized_margin":  _f(valid_norm.min())  if valid_norm.size else None,
        "max_normalized_margin":  _f(valid_norm.max())  if valid_norm.size else None,
        "mean_cone_margin": _f(valid_margins.mean()) if valid_margins.size else None,
        "min_cone_margin":  _f(valid_margins.min())  if valid_margins.size else None,
        "max_cone_margin":  _f(valid_margins.max())  if valid_margins.size else None,
        "n_lp_at_limit":    int(result["lp_at_limit"].sum()),
        "n_escalated":      int(result["escalated"].sum()),
        "mean_n_binding":   _f(float(np.mean(result["n_binding"]))) if n else None,
        "mean_z_vs_shuffled":
            _mean_opt([nl["vs_shuffled"].get("z_score") for nl in nulls.values()]),
        "mean_z_vs_uniform":
            _mean_opt([nl["vs_uniform"].get("z_score") for nl in nulls.values()]),
        "mean_shuffled_cone_fraction":
            _mean_opt([nl.get("shuffled_cone_fraction") for nl in nulls.values()]),
        "mean_uniform_cone_fraction":
            _mean_opt([nl.get("uniform_cone_fraction") for nl in nulls.values()]),
        "n_null_layers":    len(nulls),
        "pca_n_components": result["pca_n_components"],
        "dropped_indices":  result.get("dropped_indices", []),
        "tol":              result["tol"],
    }

    return {"per_layer": per_layer, "summary": summary}


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _mean_opt(vals):
    xs = [float(v) for v in vals if v is not None and float(v) == float(v)]
    return float(np.mean(xs)) if xs else None
