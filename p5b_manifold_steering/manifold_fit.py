"""
p5b_manifold_steering/manifold_fit.py — Sub-experiment A.

Fit activation manifold Mh and behavior manifold My to cluster centroids
and output distributions, following Wurgaft et al. (2026) §2.2.

The key difference from Wurgaft: we use unsupervised HDBSCAN cluster
centroids (from Phase 1) instead of concept-labeled data. The intrinsic
coordinates are arc-length parameterized from the centroid path across layers.

Imports from existing project scripts:
  - core.config  : model registry, prompt registry
  - core.polar   : sphere_gap (frame diagnostic, see compute_fit_summary)
  - cluster_tracking.py (Phase 1): centroid trajectory loading
  - p5b_io.py    (Phase 5b): load_phase1_run for Phase 1 artifacts

Scope note (2026-07-21): the splines fit here are consumed by Sub-exp A's
own residual metric (P5b-A2) and by future steering work. Sub-exp B no
longer routes through them — see design-5b.md, "Sub-exp B on direct
pairwise distances", and p5b_distances.py.
"""

from __future__ import annotations

from pathlib import Path
from typing  import Optional

import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline


# ---------------------------------------------------------------------------
# PCA reduction
# ---------------------------------------------------------------------------

def _complete_basis(V: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    Extend orthonormal columns V (d, r) to (d, k) with arbitrary orthonormal
    directions drawn from the orthogonal complement of span(V).

    Only reached when more components are requested than the data has rank
    for (k > min(n, d)). The padded columns carry ZERO variance and are not
    principal components in any meaningful sense — they exist so the
    returned basis satisfies its declared (d, k) shape contract and stays
    orthonormal. `pca_reduce` reports their explained-variance ratio as
    exactly 0.0 so a caller reading `evr` can always tell which columns are
    real.

    Bounded memory on purpose: the obvious fix for the old truncation bug
    was np.linalg.svd(..., full_matrices=True), but that materializes a
    (d, d) matrix — 134 MB at d=4096 — to use k << d columns of it. This
    route never allocates more than (d, k).

    `seed` is fixed rather than drawn from global state so two runs on the
    same data return the same basis.
    """
    d, r = V.shape
    if k <= r:
        return V[:, :k]
    if k > d:
        raise ValueError(
            f"_complete_basis: cannot produce {k} orthonormal columns in "
            f"{d} dimensions"
        )
    rng   = np.random.default_rng(seed)
    extra = rng.standard_normal((d, k - r))
    # Project out span(V), orthonormalize, then re-project for numerical
    # safety (one pass of Gram-Schmidt leaves ~1e-8 leakage at large d).
    extra -= V @ (V.T @ extra)
    Q, _   = np.linalg.qr(extra)
    Q     -= V @ (V.T @ Q)
    Q, _   = np.linalg.qr(Q)
    return np.concatenate([V, Q[:, : k - r]], axis=1)


def pca_reduce(
    centroids: np.ndarray,
    k: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA-reduce centroids to k dimensions.

    Parameters
    ----------
    centroids : (n, d) — cluster centroid vectors
    k         : target dimensionality
    seed      : determinism for the rank-deficient padding path only

    Returns
    -------
    scores          : (n, k) — projections onto top-k PCs
    basis           : (d, k) — orthonormal PC basis (columns)
    explained_var_ratio : (k,) — fraction of variance per PC, zero-padded
                          beyond the data's rank

    Bug fix (2026-07-21): the previous implementation did
    `k = min(k, len(s))` with `full_matrices=False`, so `s` had only
    min(n, d) entries and k silently collapsed to the sample count. With
    n=7 centroids, d=64, k=32 requested, it returned a (64, 7) basis — the
    caller asked for a 32-d space and got a 7-d one with no warning. Now k
    is clamped to the ambient dimension d only, and any shortfall in rank
    is padded (see _complete_basis) and reported as zero variance.

    Note this padding is a contract-compliance path, not a normal one:
    run_5b.py already clamps `pca_k = min(args.pca_dim, n_c - 1)` before
    calling, so the pipeline never triggers it. It exists because the
    declared return shape should not depend on the input's rank.
    """
    X = centroids - centroids.mean(axis=0, keepdims=True)
    n, d = X.shape

    k = int(min(k, d))
    if k < 1:
        raise ValueError(f"pca_reduce: k must be >= 1, got {k}")

    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T                                   # (d, r), r = min(n, d)
    r = V.shape[1]

    basis  = V[:, :k] if k <= r else _complete_basis(V, k, seed=seed)
    scores = X @ basis                         # (n, k)

    total_var = float((s ** 2).sum())
    evr = np.zeros(k, dtype=np.float64)
    m = min(k, len(s))
    evr[:m] = (s[:m] ** 2) / (total_var + 1e-12)

    return scores, basis, evr


# ---------------------------------------------------------------------------
# Arc-length parameterization
# ---------------------------------------------------------------------------

def arc_length_params(
    pts: np.ndarray,
    periodic: bool = False,
) -> np.ndarray:
    """
    Compute arc-length parameterization u ∈ [0, 1] for a sequence of points.

    Parameters
    ----------
    pts      : (n, d) — ordered sequence of points
    periodic : if True, close the loop by appending pts[0]

    Returns
    -------
    u : (n,) — normalized cumulative arc-length, u[0]=0, u[-1]=1

    ORDERING CAVEAT — read before using u for anything comparative.
    "Ordered sequence" is this function's precondition and nothing upstream
    currently establishes it. load_plateau_centroids iterates
    `trajectories` in list order, which track_clusters produces as
    `sorted(active_trajectories)` — trajectory id order, i.e. cluster BIRTH
    order, which has no relationship to the geometry. Wurgaft's control
    points carry an intrinsic sequence (Monday→Sunday); ours do not. A
    spline threaded in birth order is a curve through a zigzag, and its
    cumulative arc length is the length of that zigzag.

    This is fine for Sub-exp A's residual metric (P5b-A2 asks whether a
    smooth curve can be threaded through the control points at all, which
    is order-dependent but still a real question). It is NOT fine as a
    coordinate for cross-manifold comparison, which is why Sub-exp B moved
    off it. Do not reintroduce u into a comparative test without first
    solving the seriation problem.
    """
    if periodic:
        pts_ext = np.vstack([pts, pts[:1]])
    else:
        pts_ext = pts

    diffs = np.diff(pts_ext, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cumul = np.concatenate([[0.0], np.cumsum(dists)])
    total = float(cumul[-1])
    if total < 1e-12:
        return np.linspace(0, 1, len(pts))
    u = cumul[:len(pts)] / total
    return u


# ---------------------------------------------------------------------------
# Periodic knot handling
# ---------------------------------------------------------------------------

def _periodic_wrap_u(u: np.ndarray, period: float = 1.0) -> float:
    """
    Resolve the wrap-around knot for a periodic spline.

    CubicSpline(bc_type="periodic") requires a strictly increasing knot
    vector whose final entry closes the loop.

    Bug fix (2026-07-21): both fit_activation_manifold and
    fit_behavior_manifold previously hardcoded `np.append(u, 1.0)`. That
    assumes u stops short of 1.0 — but every caller passes either
    np.linspace(0, 1, n) (tests) or arc_length_params output (pipeline),
    and BOTH terminate at exactly 1.0. The result was a duplicated final
    knot and scipy raising "`x` must be strictly increasing sequence",
    which took out 14 tests across four classes in test_phase5b.py.

    If u ends short of `period`, `period` is the wrap point. If u already
    reaches it, extrapolate one mean step past the last knot instead.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or u.size < 2:
        raise ValueError(
            f"_periodic_wrap_u: need 1-D u with >= 2 knots, got shape {u.shape}"
        )
    if not np.all(np.diff(u) > 0):
        raise ValueError("_periodic_wrap_u: u must be strictly increasing")
    if u[-1] < period - 1e-12:
        return float(period)
    return float(u[-1] + float(np.diff(u).mean()))


# ---------------------------------------------------------------------------
# Activation manifold Mh
# ---------------------------------------------------------------------------

def fit_activation_manifold(
    centroids_pca: np.ndarray,
    u:             np.ndarray,
    periodic:      bool = False,
    smooth:        float = 0.0,
) -> dict:
    """
    Fit a 1D cubic spline through centroids in PCA space.

    Parameters
    ----------
    centroids_pca : (n, k) — centroids already projected to PCA space
    u             : (n,) — intrinsic coordinates in [0, 1]
    periodic      : if True, fit a periodic spline (cyclic concepts)
    smooth        : smoothing factor for UnivariateSpline; 0 = interpolating

    Returns
    -------
    dict with keys:
      spline       — list of per-dimension scipy spline objects
      u_knots      — (n,) intrinsic coordinates of control points
      control_pts  — (n, k) centroids (= spline control points)
      residual_rms — RMS distance from centroid to nearest spline point
      k_dim        — dimensionality k
      periodic     — bool
      u_wrap       — float | None; the closing knot used when periodic
    """
    n, k_dim = centroids_pca.shape
    splines  = []
    u_wrap   = None

    if periodic:
        u_wrap = _periodic_wrap_u(u)
        u_ext  = np.append(np.asarray(u, dtype=float), u_wrap)
        c_ext  = np.vstack([centroids_pca, centroids_pca[:1]])
        for d in range(k_dim):
            spl = CubicSpline(u_ext, c_ext[:, d], bc_type="periodic")
            splines.append(spl)
    else:
        for d in range(k_dim):
            spl = CubicSpline(u, centroids_pca[:, d])
            splines.append(spl)

    # Measure interpolation residual
    reconstructed = np.stack([spl(u) for spl in splines], axis=1)  # (n, k)
    residuals = np.linalg.norm(reconstructed - centroids_pca, axis=1)
    residual_rms = float(np.sqrt((residuals ** 2).mean()))

    return {
        "spline":       splines,
        "u_knots":      u,
        "control_pts":  centroids_pca,
        "residual_rms": residual_rms,
        "k_dim":        k_dim,
        "periodic":     periodic,
        "u_wrap":       u_wrap,
    }


def eval_manifold(mh: dict, t: np.ndarray) -> np.ndarray:
    """
    Evaluate Mh at intrinsic coordinates t.

    Parameters
    ----------
    mh : output of fit_activation_manifold
    t  : (m,) query coordinates in [0, 1]

    Returns
    -------
    pts : (m, k) — points on Mh in PCA space
    """
    return np.stack([spl(t) for spl in mh["spline"]], axis=1)


# ---------------------------------------------------------------------------
# Behavior manifold My
# ---------------------------------------------------------------------------

def _to_hellinger(p: np.ndarray) -> np.ndarray:
    """Map distribution p → √p (Hellinger embedding). Result is unit-norm."""
    sq = np.sqrt(np.clip(p, 0.0, None))
    norms = np.linalg.norm(sq, axis=-1, keepdims=True)
    return sq / np.maximum(norms, 1e-12)


def _from_hellinger(h: np.ndarray) -> np.ndarray:
    """Map √p back to probability: square and renormalize."""
    p = h ** 2
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=-1, keepdims=True)
    return p / np.maximum(s, 1e-12)


def _sphere_log_map(base: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Spherical log map: project pts onto the tangent plane at base.
    base : (d,) unit vector
    pts  : (n, d) unit vectors
    Returns tangent vectors (n, d).
    """
    dot   = (pts * base).sum(axis=1, keepdims=True).clip(-1 + 1e-7, 1 - 1e-7)
    theta = np.arccos(dot)                   # (n, 1)
    perp  = pts - dot * base                 # (n, d)
    perp_norm = np.linalg.norm(perp, axis=1, keepdims=True)
    unit_perp = perp / np.maximum(perp_norm, 1e-12)
    return theta * unit_perp


def _sphere_exp_map(base: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    """
    Spherical exp map: lift tangent vectors at base back to the sphere.
    base     : (d,) unit vector
    tangents : (n, d) tangent vectors
    Returns (n, d) unit vectors on the sphere.
    """
    theta = np.linalg.norm(tangents, axis=1, keepdims=True)
    safe  = np.maximum(theta, 1e-12)
    return np.cos(theta) * base + np.sin(theta) * (tangents / safe)


def fit_behavior_manifold(
    distributions: np.ndarray,
    u:             np.ndarray,
    periodic:      bool = False,
) -> dict:
    """
    Fit a 1D spline through output distributions in Hellinger space.

    Maps p → √p onto the unit sphere, fits spline in tangent plane at base
    point (mean of √p vectors), then lifts back via exp map at decode time.

    Parameters
    ----------
    distributions : (n, vocab) — probability distributions (each row sums to 1)
    u             : (n,) — intrinsic coordinates in [0, 1]
    periodic      : if True, fit a periodic spline

    Returns
    -------
    dict with keys:
      spline           — list of per-dimension spline objects (in tangent space)
      u_knots          — (n,)
      sqrt_centroids   — (n, vocab) — √p vectors used as control points
      base             — (vocab,) — base point on sphere
      residual_rms     — RMS Hellinger distance from centroid to spline
      vocab            — int
      periodic         — bool
      u_wrap           — float | None
    """
    sqrt_c = _to_hellinger(distributions)         # (n, vocab)
    base   = sqrt_c.mean(axis=0)
    base   = base / np.linalg.norm(base)
    tangents = _sphere_log_map(base, sqrt_c)       # (n, vocab)

    vocab   = distributions.shape[1]
    splines = []
    u_wrap  = None

    if periodic:
        u_wrap = _periodic_wrap_u(u)
        u_ext  = np.append(np.asarray(u, dtype=float), u_wrap)
        t_ext  = np.vstack([tangents, tangents[:1]])
        for dim in range(vocab):
            spl = CubicSpline(u_ext, t_ext[:, dim], bc_type="periodic")
            splines.append(spl)
    else:
        for dim in range(vocab):
            spl = CubicSpline(u, tangents[:, dim])
            splines.append(spl)

    # Residual: Hellinger distance from √c to decoded spline at same u
    reconstructed_tan = np.stack([spl(u) for spl in splines], axis=1)
    reconstructed_sq  = _sphere_exp_map(base, reconstructed_tan)   # (n, vocab)
    # Hellinger distance: 1/√2 * ‖√p - √q‖
    diff = sqrt_c - reconstructed_sq
    h_dist = np.linalg.norm(diff, axis=1) / np.sqrt(2)
    residual_rms = float(np.sqrt((h_dist ** 2).mean()))

    return {
        "spline":         splines,
        "u_knots":        u,
        "sqrt_centroids": sqrt_c,
        "base":           base,
        "residual_rms":   residual_rms,
        "vocab":          vocab,
        "periodic":       periodic,
        "u_wrap":         u_wrap,
    }


def eval_behavior_manifold(my: dict, t: np.ndarray) -> np.ndarray:
    """
    Evaluate My at intrinsic coordinates t, returning probability distributions.

    Parameters
    ----------
    my : output of fit_behavior_manifold
    t  : (m,) query coordinates

    Returns
    -------
    p : (m, vocab) — probability distributions
    """
    tangents = np.stack([spl(t) for spl in my["spline"]], axis=1)  # (m, vocab)
    sqrt_pts = _sphere_exp_map(my["base"], tangents)                # (m, vocab)
    return _from_hellinger(sqrt_pts)


# ---------------------------------------------------------------------------
# Load cluster centroids from Phase 1 artifacts
# ---------------------------------------------------------------------------

def load_plateau_centroids(
    centroid_trajs: dict,
    trajectories:   list[dict],
    min_lifespan:   int = 3,
) -> tuple[np.ndarray, list[int]]:
    """
    Stack per-trajectory mean centroids into a (n_clusters, d) array.

    Uses the integer-keyed dict returned by p5b_io.load_phase1_run, not the
    raw NPZ (which uses string keys "traj_{id}").

    Parameters
    ----------
    centroid_trajs : {int trajectory_id: (lifespan, d) float32}
                     — from load_phase1_run["centroid_trajs"]
    trajectories   : list of trajectory dicts [{id, chain}, ...]
                     — from load_phase1_run["trajectories"]
    min_lifespan   : skip trajectories shorter than this

    Returns
    -------
    centroids : (n_valid, d) — L2-normalised mean centroid per trajectory
    traj_ids  : list[int]   — trajectory IDs in the same order as rows

    `traj_ids` is the alignment key for the whole phase: every downstream
    per-trajectory quantity (behavior distributions, subspace projections)
    must be assembled by iterating THIS list, not by independently
    enumerating some other population and hoping the counts match. That
    hope is what broke Sub-exp B (see design-5b.md).

    Frame note: the returned centroids are L2-normalized, i.e. they live in
    the sphere frame — which is the frame Phase 1's clustering was
    performed in, and therefore the frame in which "these are our cluster
    centroids" is true by construction. It is NOT the frame the model
    reads in (that is LN; see core/ln_frame.py). Callers wanting the read
    frame should build centroids via p5b_distances.frame_centroids rather
    than post-hoc transforming this output — LN is not linear, so
    LN(mean of tokens) != mean of LN(tokens).
    """
    centroids = []
    traj_ids  = []

    for traj in trajectories:
        tid  = int(traj["id"])
        arr  = centroid_trajs.get(tid)
        if arr is None or arr.shape[0] < min_lifespan:
            continue
        mean_c = arr.mean(axis=0)
        norm   = float(np.linalg.norm(mean_c))
        centroids.append(mean_c / max(norm, 1e-12))
        traj_ids.append(tid)

    if not centroids:
        raise ValueError(
            f"No trajectories with lifespan ≥ {min_lifespan}. "
            f"Available IDs: {list(centroid_trajs.keys())}"
        )

    return np.stack(centroids, axis=0).astype(np.float32), traj_ids


# ---------------------------------------------------------------------------
# Fit summary
# ---------------------------------------------------------------------------

def compute_fit_summary(
    mh:          dict,
    my:          dict | None,
    pca_evr:     np.ndarray,
    k_threshold: float = 0.80,
    frame_diagnostics: dict | None = None,
) -> dict:
    """
    Compute and return fit quality metrics.

    `my` is None whenever no behavior manifold could be fit (fewer than 4
    logit-bearing plateau layers — the common case for any run without a
    live model, or with too few plateau layers overlapping the logit
    cache). Bug fix: the caller previously substituted `mh` for `my` in
    this case, which crashed on `my["vocab"]` — `fit_activation_manifold`'s
    return dict has no "vocab" key, only `fit_behavior_manifold`'s does.
    Behavior-side fields are now reported as None/False instead.

    `frame_diagnostics` : optional dict of per-layer core.polar.sphere_gap
    output, merged in under "frame_diagnostics". This is the pre-registered
    escalation trigger for the sphere-vs-LN frame question (design-5b.md):
    where the gaps are ~0 the sphere-frame reading transfers and the frame
    question is empirically moot; where they spike, the LN-frame reading is
    the one to trust. Recording it here rather than deciding post hoc is
    the point — see core/polar.py::sphere_gap's own interpretation
    contract.

    Returns dict suitable for fit_summary.json.
    """
    pca_cumvar = float(pca_evr.sum())
    cum = np.cumsum(pca_evr)
    # Clamp: if the spectrum never reaches k_threshold, searchsorted returns
    # len(cum) and the old `+ 1` reported one MORE dimension than exists.
    n_dims_80 = int(min(np.searchsorted(cum, k_threshold) + 1, len(pca_evr)))

    out = {
        "pca_explained_var":     pca_cumvar,
        "pca_n_dims_for_80pct":  n_dims_80,
        "mh_spline_residual_rms": mh["residual_rms"],
        "my_spline_residual_rms": my["residual_rms"] if my is not None else None,
        "n_control_points":      int(len(mh["u_knots"])),
        "k_pca_dim":             int(mh["k_dim"]),
        "vocab_size":            int(my["vocab"]) if my is not None else None,
        "p5b_a1_pass":           pca_cumvar >= k_threshold,
        "p5b_a2_pass": (
            my is not None
            and mh["residual_rms"] < 0.1
            and my["residual_rms"] < 0.1
        ),
    }
    if frame_diagnostics is not None:
        out["frame_diagnostics"] = frame_diagnostics
    return out


def sphere_gap_by_layer(
    activations,
    layers: list[int] | None = None,
) -> dict:
    """
    core.polar.sphere_gap for each requested layer, JSON-ready.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d) — load_phase1_run["activations"]
    layers      : which layer indices to score; None -> all

    Returns
    -------
    {"per_layer": {layer_idx: sphere_gap dict},
     "max_pearson_gap": float, "max_spearman_gap": float}

    The two maxima are the summary numbers the escalation rule reads. Kept
    as a separate function from compute_fit_summary so it can be called
    (and tested) without a fitted manifold in hand.
    """
    from core.polar import sphere_gap

    if activations is None:
        return {"per_layer": {}, "max_pearson_gap": None, "max_spearman_gap": None}

    arr = np.asarray(activations)
    idxs = list(range(arr.shape[0])) if layers is None else list(layers)

    per_layer: dict = {}
    for li in idxs:
        if 0 <= li < arr.shape[0]:
            per_layer[int(li)] = sphere_gap(arr[li])

    def _max(field: str):
        vals = [v[field] for v in per_layer.values()
                if v.get(field) is not None and np.isfinite(v[field])]
        return float(max(vals)) if vals else None

    return {
        "per_layer":        per_layer,
        "max_pearson_gap":  _max("pearson_gap"),
        "max_spearman_gap": _max("spearman_gap"),
    }
