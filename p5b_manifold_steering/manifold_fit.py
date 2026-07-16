"""
p5b_manifold/manifold_fit.py — Sub-experiment A.

Fit activation manifold Mh and behavior manifold My to cluster centroids
and output distributions, following Wurgaft et al. (2026) §2.2.

The key difference from Wurgaft: we use unsupervised HDBSCAN cluster
centroids (from Phase 1) instead of concept-labeled data. The intrinsic
coordinates are arc-length parameterized from the centroid path across layers.

Imports from existing project scripts:
  - core.config  : model registry, prompt registry
  - cluster_tracking.py (Phase 1): centroid trajectory loading
  - io_utils.py  (Phase 1): load_run for Phase 1 artifacts
"""

from __future__ import annotations

from pathlib import Path
from typing  import Optional

import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline


# ---------------------------------------------------------------------------
# PCA reduction
# ---------------------------------------------------------------------------

def pca_reduce(
    centroids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA-reduce centroids to k dimensions.

    Parameters
    ----------
    centroids : (n, d) — cluster centroid vectors
    k         : target dimensionality

    Returns
    -------
    scores          : (n, k) — projections onto top-k PCs
    basis           : (d, k) — orthonormal PC basis (columns)
    explained_var_ratio : (k,) — fraction of variance per PC
    """
    X  = centroids - centroids.mean(axis=0, keepdims=True)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    k  = min(k, len(s))
    basis  = Vt[:k].T                              # (d, k)
    scores = X @ basis                             # (n, k)
    total_var = float((s ** 2).sum())
    evr = (s[:k] ** 2) / (total_var + 1e-12)
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
    """
    n, k_dim = centroids_pca.shape
    splines = []

    if periodic:
        # Append first point to close the loop
        u_ext = np.append(u, 1.0)
        c_ext = np.vstack([centroids_pca, centroids_pca[:1]])
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
    """
    sqrt_c = _to_hellinger(distributions)         # (n, vocab)
    base   = sqrt_c.mean(axis=0)
    base   = base / np.linalg.norm(base)
    tangents = _sphere_log_map(base, sqrt_c)       # (n, vocab)

    vocab   = distributions.shape[1]
    splines = []

    if periodic:
        u_ext = np.append(u, 1.0)
        t_ext = np.vstack([tangents, tangents[:1]])
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

    Uses the integer-keyed dict returned by io.load_phase1_run, not the
    raw NPZ (which uses string keys "traj_{id}").

    Parameters
    ----------
    centroid_trajs : {int trajectory_id: (lifespan, d) float32}
                     — from io.load_phase1_run["centroid_trajs"]
    trajectories   : list of trajectory dicts [{id, chain}, ...]
                     — from io.load_phase1_run["trajectories"]
    min_lifespan   : skip trajectories shorter than this

    Returns
    -------
    centroids : (n_valid, d) — L2-normalised mean centroid per trajectory
    traj_ids  : list[int]   — trajectory IDs in the same order as rows
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

    Returns dict suitable for fit_summary.json.
    """
    pca_cumvar = float(pca_evr.sum())
    n_dims_80  = int(np.searchsorted(np.cumsum(pca_evr), k_threshold)) + 1

    return {
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
