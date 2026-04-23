"""
bipartition_detect.py — Block 0 of Phase 1h.

Asks whether the Fiedler bipartition at each layer is real geometric
structure or an eigengap artifact.  A k=2 eigengap on the normalized
Laplacian of the Gram matrix always exists; the question is whether
the second eigenvector partitions tokens into two populated, separated,
internally-compact sets.

Per-layer outputs
-----------------
bipartition_eigengap   : (λ₃ − λ₂) / λ₃ on the normalized Laplacian.
                         High means the k=2 partition dominates; low
                         means tertiary structure is comparable.
bipartition_sharpness  : centroid angle in activation space, radians.
                         Near 0: collapsed.  π/2: orthogonal.
                         Near π: antipodal.
within_half_ip         : (mean_a, mean_b) — mean pairwise cosine
                         similarity inside each half.  >= 0.3 in both
                         means each half is itself a tight cluster.
between_half_ip        : mean pairwise cosine similarity across the
                         two halves.  Negative → halves point in
                         opposite directions; near 0 → orthogonal
                         separation; positive → cosmetic partition
                         (both halves face the same direction).
separation_ratio       : between_half_ip / mean(within_half_ip).
                         Negative values indicate genuine separation;
                         near 1 indicates no structural contrast.
fiedler_boundary_frac  : fraction of tokens with |v[i]| <
                         boundary_threshold * std(v).  Near 0 = sharp
                         bimodal Fiedler distribution; near 1 = all
                         tokens cluster near the partition boundary.
clip_fraction          : fraction of off-diagonal Gram entries clipped
                         to 0 (only nonzero when clip_negative=True).
                         Large values mean the Laplacian geometry is
                         substantially altered by clipping.
hemisphere_sizes       : (|A|, |B|) with |A| + |B| = n.
minority_fraction      : min(|A|, |B|) / n.
fiedler_vec            : (n_tokens,) — the raw second eigenvector
                         before sign partitioning.  Kept so downstream
                         blocks can recompute sign-corrected quantities
                         without re-diagonalizing the Laplacian.
regime                 : one of
                           "collapsed"           minority < 0.05
                           "weak_bipartition"    minority ∈ [0.05, 0.1)
                                                 or centroid_angle < π/2
                           "strong_bipartition"  minority ≥ 0.1,
                                                 centroid_angle ≥ π/2,
                                                 within_half_ip ≥ 0.3
                                                 in both halves
                           "diffuse"             minority ≥ 0.1,
                                                 centroid_angle ≥ π/2,
                                                 but at least one half
                                                 has within_half_ip < 0.3
The precedence above is evaluated top-down and is a strict partition.

Functions
---------
extract_bipartition_spectrum : top-3 Laplacian eigenvalues + Fiedler vector per layer.
within_half_inner_products   : mean pairwise cosine within each hemisphere.
between_half_inner_products  : mean pairwise cosine across hemispheres.
separation_ratio             : between_half / mean(within_half) contrast ratio.
fiedler_boundary_fraction    : fraction of Fiedler values near zero.
centroid_angle               : angle between hemisphere centroids in activation space.
classify_regime              : the four-way regime label for one layer's metrics.
analyze_bipartition          : full pipeline across all layers.
bipartition_to_json          : JSON-serializable per-layer + summary block.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian


# ---------------------------------------------------------------------------
# Spectrum + Fiedler extraction
# ---------------------------------------------------------------------------

def extract_bipartition_spectrum(
    activations: np.ndarray,
    clip_negative: bool = True,
) -> dict:
    """
    Compute the first three eigenvalues of the normalized Gram-Laplacian and
    the Fiedler vector at every layer.

    Mirrors phase1.fiedler_tracking.extract_fiedler_per_layer but (a) also
    returns λ₃, required for the bipartition_eigengap, and (b) exposes the
    raw fiedler vector for downstream mechanism analysis.

    Parameters
    ----------
    activations   : (n_layers, n_tokens, d) — L2-normed.
    clip_negative : whether to clip negative Gram entries to 0 before
                    building the Laplacian.  Matches the convention used
                    by phase1.spectral.spectral_eigengap_k.

    Returns
    -------
    dict with:
      eigvals        (n_layers, 3) — first three Laplacian eigenvalues,
                                     nan where the computation fails.
      fiedler_vecs   (n_layers, n_tokens) — second eigenvector.  Sign is
                                     arbitrary; downstream code must
                                     sign-align if identity matters.
      valid          (n_layers,) bool — True when eigendecomposition
                                     returned finite values AND the three
                                     eigenvalues came back in ascending
                                     order.
    """
    n_layers, n_tokens, _ = activations.shape

    eigvals      = np.full((n_layers, 3), np.nan, dtype=np.float64)
    fiedler_vecs = np.zeros((n_layers, n_tokens), dtype=np.float64)
    valid        = np.zeros(n_layers, dtype=bool)

    if n_tokens < 4:
        # Too few tokens for λ₃ to exist independently of λ₂.
        return {
            "eigvals":      eigvals,
            "fiedler_vecs": fiedler_vecs,
            "valid":        valid,
        }

    for L in range(n_layers):
        X = activations[L]
        G = X @ X.T
        if clip_negative:
            G = np.clip(G, 0, None)
        np.fill_diagonal(G, 1.0)
        Lap = laplacian(G, normed=True)

        try:
            vals, vecs = eigh(Lap, subset_by_index=[0, 2])
            vals = np.real(vals)
            if not np.all(np.isfinite(vals)):
                continue
            if not (vals[0] <= vals[1] <= vals[2]):
                # scipy returns ascending order; out-of-order signals numerical
                # trouble (e.g. from the clip) and should invalidate the layer.
                continue
            eigvals[L]      = vals
            fiedler_vecs[L] = np.real(vecs[:, 1])
            valid[L]        = True
        except Exception:
            continue

    return {
        "eigvals":      eigvals,
        "fiedler_vecs": fiedler_vecs,
        "valid":        valid,
    }


# ---------------------------------------------------------------------------
# Within-hemisphere compactness
# ---------------------------------------------------------------------------

def within_half_inner_products(
    X: np.ndarray,
    assignment: np.ndarray,
) -> tuple[float, float]:
    """
    Mean pairwise cosine similarity within each half.

    For L2-normed rows of X, the pairwise cosine is <x_i, x_j>.  We
    average over the strict upper triangle of each half's self-Gram.

    Parameters
    ----------
    X          : (n_tokens, d) — L2-normed activation matrix at one layer.
    assignment : (n_tokens,) int in {0, 1} — hemisphere label.

    Returns
    -------
    (mean_in_A, mean_in_B).  A half with fewer than 2 tokens returns nan.
    """
    out = []
    for half in (0, 1):
        mask = assignment == half
        k = int(mask.sum())
        if k < 2:
            out.append(float("nan"))
            continue
        Xh = X[mask]
        G  = Xh @ Xh.T
        # Upper triangle, excluding the diagonal (self-IP = 1).
        iu = np.triu_indices(k, k=1)
        out.append(float(G[iu].mean()))
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Cross-hemisphere compactness (separation contrast)
# ---------------------------------------------------------------------------

def between_half_inner_products(
    X: np.ndarray,
    assignment: np.ndarray,
) -> float:
    """
    Mean pairwise cosine similarity between the two halves (cross-block).

    For L2-normed X, the pairwise cosine is <x_i, x_j>.  We average over
    all (i in A, j in B) pairs — the full off-diagonal cross block, not
    the upper triangle, since A×B is not symmetric.

    Returns nan when either half has fewer than 1 token.

    Interpretation
    --------------
    Negative  → halves point in opposite directions on the sphere.
                Strong geometric separation.
    Near 0    → halves are roughly orthogonal.  Separation exists but
                the halves are not antipodal.
    Positive  → both halves lean the same direction.  The bipartition
                is a local density split, not a global orientation split.

    Use together with within_half_inner_products to compute the
    separation_ratio: between / mean(within).  Ratio < 0 or << 1 is
    strong evidence of genuine separation.
    """
    mask_a = assignment == 0
    mask_b = assignment == 1
    na = int(mask_a.sum())
    nb = int(mask_b.sum())
    if na < 1 or nb < 1:
        return float("nan")

    Xa = X[mask_a]   # (na, d)
    Xb = X[mask_b]   # (nb, d)
    # Full cross-Gram: (na, nb), every entry is <x_i, x_j> for i in A, j in B.
    cross = Xa @ Xb.T
    return float(cross.mean())


def compute_separation_ratio(
    within_a: float,
    within_b: float,
    between: float,
) -> float:
    """
    between_half_ip / mean(within_half_ip).

    Values < 0  : halves are antipodal while each is internally coherent.
    Values ≈ 1  : no structural contrast — bipartition is cosmetic.
    Values > 1  : within-half similarity is *less* than cross-half (unusual;
                  indicates the Fiedler partition is cutting through a coherent
                  cluster rather than separating two of them).

    Returns nan when any input is nan or when the mean within-half ip is 0.
    """
    vals = (within_a, within_b, between)
    if any(v != v for v in vals):
        return float("nan")
    denom = 0.5 * (within_a + within_b)
    if abs(denom) < 1e-12:
        return float("nan")
    return float(between / denom)


# ---------------------------------------------------------------------------
# Fiedler value distribution sharpness
# ---------------------------------------------------------------------------

def fiedler_boundary_fraction(
    fiedler_vec: np.ndarray,
    boundary_threshold: float = 0.30,
) -> float:
    """
    Fraction of tokens that fall in the "boundary zone" of the Fiedler vector.

    The boundary zone is defined as: |v[i]| < boundary_threshold * std(v).
    These are tokens that are neither clearly in hemisphere A nor hemisphere B
    — they sit close to the sign-boundary and would switch sides with a small
    perturbation of the eigenvector.

    Parameters
    ----------
    fiedler_vec        : (n_tokens,) raw second Laplacian eigenvector.
    boundary_threshold : fraction of std(v) that defines the boundary width.
                         Default 0.30 — tokens within 30% of one std from 0.

    Returns
    -------
    float in [0, 1].  Near 0 = sharp bimodal partition (tokens cluster far
    from 0 on both sides).  Near 1 = all tokens are marginal (unimodal
    distribution centered near 0, or uniform).

    This is the primary measure of *assignment confidence* at the partition
    boundary.  A "strong_bipartition" regime layer with a high
    fiedler_boundary_frac (> 0.3) should be interpreted cautiously: the
    regime labels are formally correct but a large fraction of the tokens
    could be re-assigned by numerical noise.
    """
    v = np.asarray(fiedler_vec, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    s = float(np.std(v))
    if s < 1e-12:
        return float("nan")
    boundary_width = boundary_threshold * s
    return float(np.mean(np.abs(v) < boundary_width))


# ---------------------------------------------------------------------------
# Centroid angle
# ---------------------------------------------------------------------------

def centroid_angle(
    X: np.ndarray,
    assignment: np.ndarray,
) -> float:
    """
    Angle in radians between the mean vectors of the two halves.

    Returns nan if either half is empty or either centroid is zero.
    """
    mask_a = assignment == 0
    mask_b = assignment == 1
    if not mask_a.any() or not mask_b.any():
        return float("nan")

    ca = X[mask_a].mean(axis=0)
    cb = X[mask_b].mean(axis=0)
    na = np.linalg.norm(ca)
    nb = np.linalg.norm(cb)
    if na < 1e-10 or nb < 1e-10:
        return float("nan")

    cos = float(np.dot(ca, cb) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return float(np.arccos(cos))


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------

# Thresholds collected in one place so the cross-run aggregator can record
# them next to the results (interpretation depends on the values used).
REGIME_THRESHOLDS = {
    "collapsed_minority":   0.05,
    "weak_minority":        0.10,
    "strong_angle_rad":     np.pi / 2.0,
    "strong_within_ip":     0.30,
    "boundary_threshold":   0.30,   # for fiedler_boundary_fraction
}


def classify_regime(
    minority_fraction: float,
    centroid_angle_rad: float,
    within_half_a: float,
    within_half_b: float,
    thresholds: dict | None = None,
) -> str:
    """
    Strict top-down classification into the four regimes.

    The ordering is chosen so the rules form a partition:

      1. minority < 0.05                       → collapsed
      2. minority < 0.10 OR angle < π/2        → weak_bipartition
      3. within_A ≥ 0.3 AND within_B ≥ 0.3     → strong_bipartition
      4. otherwise                             → diffuse

    A nan in any input returns "collapsed" (we cannot argue the partition
    is geometrically meaningful without the measurements).
    """
    th = REGIME_THRESHOLDS if thresholds is None else thresholds

    # Defensive: any nan kills the classification.
    vals = (minority_fraction, centroid_angle_rad, within_half_a, within_half_b)
    if any(v != v for v in vals):
        return "collapsed"

    if minority_fraction < th["collapsed_minority"]:
        return "collapsed"
    if (minority_fraction < th["weak_minority"]
            or centroid_angle_rad < th["strong_angle_rad"]):
        return "weak_bipartition"
    if (within_half_a >= th["strong_within_ip"]
            and within_half_b >= th["strong_within_ip"]):
        return "strong_bipartition"
    return "diffuse"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_bipartition(
    activations: np.ndarray,
    clip_negative: bool = True,
) -> dict:
    """
    Run Block 0 across every layer.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d) — L2-normed.

    Returns
    -------
    dict with:
      eigvals               (n_layers, 3)
      fiedler_vecs          (n_layers, n_tokens)
      valid                 (n_layers,)  bool
      assignments           (n_layers, n_tokens)  int in {0, 1}
      hemisphere_sizes      (n_layers, 2)  int
      minority_fraction     (n_layers,)
      bipartition_eigengap  (n_layers,)  = (λ₃ − λ₂) / λ₃
      centroid_angle        (n_layers,)
      within_half_ip        (n_layers, 2)
      between_half_ip       (n_layers,)   — NEW: cross-hemisphere cosine mean
      separation_ratio      (n_layers,)   — NEW: between / mean(within)
      fiedler_boundary_frac (n_layers,)   — NEW: fraction of tokens near 0
      clip_fraction         (n_layers,)   — NEW: fraction of Gram entries clipped
      regime                (n_layers,)  str
    """
    n_layers, n_tokens, _ = activations.shape

    spec = extract_bipartition_spectrum(activations, clip_negative=clip_negative)
    eigvals      = spec["eigvals"]
    fiedler_vecs = spec["fiedler_vecs"]
    valid        = spec["valid"]

    assignments        = np.full((n_layers, n_tokens), -1, dtype=np.int8)
    hemisphere_sizes   = np.zeros((n_layers, 2),        dtype=np.int32)
    minority_fraction  = np.full (n_layers,            np.nan)
    bipart_eigengap    = np.full (n_layers,            np.nan)
    cen_angle          = np.full (n_layers,            np.nan)
    within_ip          = np.full ((n_layers, 2),       np.nan)
    between_ip         = np.full (n_layers,            np.nan)   # NEW
    sep_ratio          = np.full (n_layers,            np.nan)   # NEW
    boundary_frac      = np.full (n_layers,            np.nan)   # NEW
    clip_frac          = np.full (n_layers,            np.nan)   # NEW
    regime             = np.full (n_layers, "collapsed", dtype=object)

    boundary_threshold = REGIME_THRESHOLDS["boundary_threshold"]

    for L in range(n_layers):
        X = activations[L]

        # --- clip fraction (before valid check — useful for diagnostics) ---
        if clip_negative:
            G_raw = X @ X.T
            n_off = n_tokens * (n_tokens - 1)
            if n_off > 0:
                mask = np.triu(np.ones((n_tokens, n_tokens), dtype=bool), k=1)
                n_neg = int((G_raw[mask] < 0).sum())
                clip_frac[L] = float(n_neg / n_off)
            else:
                clip_frac[L] = 0.0

        if not valid[L]:
            regime[L] = "collapsed"
            continue

        f = fiedler_vecs[L]
        # Sign partition: >= 0 → 0 (A), < 0 → 1 (B).  Ties to A.
        a = (f >= 0).astype(np.int8)
        assignments[L] = a

        na = int((a == 0).sum())
        nb = int((a == 1).sum())
        hemisphere_sizes[L] = (na, nb)
        minority_fraction[L] = min(na, nb) / n_tokens

        l2, l3 = eigvals[L, 1], eigvals[L, 2]
        if l3 > 1e-12:
            bipart_eigengap[L] = float((l3 - l2) / l3)

        cen_angle[L]  = centroid_angle(activations[L], a)
        within_ip[L]  = within_half_inner_products(activations[L], a)
        between_ip[L] = between_half_inner_products(activations[L], a)  # NEW
        sep_ratio[L]  = compute_separation_ratio(                         # NEW
            within_ip[L, 0], within_ip[L, 1], between_ip[L]
        )
        boundary_frac[L] = fiedler_boundary_fraction(f, boundary_threshold)  # NEW

        regime[L] = classify_regime(
            minority_fraction[L],
            cen_angle[L],
            within_ip[L, 0],
            within_ip[L, 1],
        )

    return {
        "eigvals":              eigvals,
        "fiedler_vecs":         fiedler_vecs,
        "valid":                valid,
        "assignments":          assignments,
        "hemisphere_sizes":     hemisphere_sizes,
        "minority_fraction":    minority_fraction,
        "bipartition_eigengap": bipart_eigengap,
        "centroid_angle":       cen_angle,
        "within_half_ip":       within_ip,
        "between_half_ip":      between_ip,       # NEW
        "separation_ratio":     sep_ratio,        # NEW
        "fiedler_boundary_frac": boundary_frac,  # NEW
        "clip_fraction":        clip_frac,        # NEW
        "regime":               regime,
        "n_layers":             n_layers,
        "n_tokens":             n_tokens,
        "thresholds":           dict(REGIME_THRESHOLDS),
    }


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def bipartition_to_json(result: dict) -> dict:
    """
    Flat per-layer + summary dict for the aggregator.

    Per-layer entries omit the raw fiedler_vec (saved to npz by the
    caller) but carry everything a regime transition analysis needs.
    """
    n = result["n_layers"]
    regime = result["regime"]

    per_layer = []
    for L in range(n):
        per_layer.append({
            "layer":                 L,
            "valid":                 bool(result["valid"][L]),
            "regime":                str(regime[L]),
            "lambda2":               _f(result["eigvals"][L, 1]),
            "lambda3":               _f(result["eigvals"][L, 2]),
            "bipartition_eigengap":  _f(result["bipartition_eigengap"][L]),
            "centroid_angle":        _f(result["centroid_angle"][L]),
            "within_half_ip":        [_f(v) for v in result["within_half_ip"][L]],
            "between_half_ip":       _f(result["between_half_ip"][L]),
            "separation_ratio":      _f(result["separation_ratio"][L]),
            "fiedler_boundary_frac": _f(result["fiedler_boundary_frac"][L]),
            "clip_fraction":         _f(result["clip_fraction"][L]),
            "hemisphere_sizes":      [int(x) for x in result["hemisphere_sizes"][L]],
            "minority_fraction":     _f(result["minority_fraction"][L]),
        })

    # Regime counts.
    regime_counts: dict[str, int] = {}
    for r in regime:
        regime_counts[str(r)] = regime_counts.get(str(r), 0) + 1

    # Summary over valid layers only.
    valid = result["valid"]
    mf = result["minority_fraction"][valid]
    ca = result["centroid_angle"][valid]
    eg = result["bipartition_eigengap"][valid]
    bi = result["between_half_ip"][valid]
    sr = result["separation_ratio"][valid]
    bf = result["fiedler_boundary_frac"][valid]
    cf = result["clip_fraction"]  # all layers, not just valid

    summary = {
        "n_layers":                    n,
        "n_tokens":                    int(result["n_tokens"]),
        "n_valid_layers":              int(valid.sum()),
        "regime_counts":               regime_counts,
        "strong_bipartition_fraction":
            float(regime_counts.get("strong_bipartition", 0) / n) if n else 0.0,
        "mean_minority_fraction":      _mean(mf),
        "mean_centroid_angle":         _mean(ca),
        "mean_bipartition_eigengap":   _mean(eg),
        "mean_between_half_ip":        _mean(bi),
        "mean_separation_ratio":       _mean(sr),
        "mean_fiedler_boundary_frac":  _mean(bf),
        "mean_clip_fraction":          _mean(cf[np.isfinite(cf)]),
        "thresholds":                  result["thresholds"],
    }

    return {"per_layer": per_layer, "summary": summary}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _f(v) -> float | None:
    """None for nan, float otherwise.  Keeps JSON readable."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _mean(arr) -> float | None:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())
