"""
bipartition_detect.py — Block 0 of Phase 1b.

Asks whether the Fiedler bipartition at each layer is real geometric
structure or an eigengap artifact. A k=2 eigengap on the normalized
Laplacian of the Gram matrix always exists; the question is whether the
second eigenvector partitions tokens into two populated, separated,
internally-compact sets.

Changes in this revision, and why they change the reading of the result
----------------------------------------------------------------------

**1. One Laplacian, not two.**

`extract_bipartition_spectrum` used to build its own normalized Laplacian
with a hardcoded 1e-4/n connectivity floor. `core.metrics.fiedler_and_eigengap`
— the function Phase 1 actually ran, via analysis_p1 — builds one without.
Phase 1b's entire premise is explaining Phase 1's k=2 result, and it was
doing so on a different graph. The floor is now a recorded parameter of the
single shared implementation (`connectivity_floor`), and this module calls
that implementation. `CONNECTIVITY_FLOOR = 1e-4` preserves Phase 1b's prior
numerical behaviour; pass 0.0 to reproduce Phase 1's graph exactly, and note
that any comparison between a Phase 1 spectral.json and a Phase 1b run must
state which floor each used.

**2. The regime classifier presupposed the answer.**

`strong_bipartition` required centroid_angle >= pi/2. If cone-collapse holds
— which Block 3 reports it does, at every layer of every model — then two
centroids drawn from inside a single open half-space essentially cannot be
pi/2 apart. So the "0% strong bipartition" null and the "100% cone-collapse"
result were not two independent tests pointing the same way, which is how
design-1b.md framed them. They are close to the same test run twice.

The antipodal classifier is retained under `variant="antipodal"` because the
existing STATUS table is stated in its vocabulary. A second classifier,
`variant="relative"`, asks the question that survives cone-collapse: is
between-half similarity materially below within-half similarity, whatever
the absolute angle? `analyze_bipartition` computes both and reports them
side by side. A layer that is "diffuse" under the antipodal reading and
"separated" under the relative one is the interesting case, and the old code
could not express it.

**3. Frames.**

The Gram matrix is built through `core.frames.frame_gram`, and the FrameSpec
travels with the result. The L2 sphere keeps the cloud's mean offset; LN
centres it. If the k=2 axis is the anisotropy direction, it should attenuate
under an LN frame — a mechanistic check on Phase 1b's own conclusion, using
only code that already exists. `frame=None` defaults to l2_sphere, which is
what every prior run used.

Per-layer outputs
-----------------
bipartition_eigengap   : (lambda3 - lambda2) / lambda3 on the normalized
                         Laplacian. High means the k=2 partition dominates.
centroid_angle         : angle between hemisphere centroids, radians.
within_half_ip         : (mean_a, mean_b) — mean pairwise cosine inside each
                         half.
between_half_ip        : mean pairwise cosine across the two halves.
separation_ratio       : between_half_ip / mean(within_half_ip). Below 1 is
                         contrast; at or above 1, none.
fiedler_boundary_frac  : fraction of tokens with |v[i]| < threshold*std(v).
clip_fraction          : fraction of upper-triangle off-diagonal Gram entries
                         that were negative before clipping.
hemisphere_sizes       : (|A|, |B|), |A| + |B| = n.
minority_fraction      : min(|A|, |B|) / n.
fiedler_vec            : (n_tokens,) raw second eigenvector.
regime                 : four-way antipodal label (see classify_regime).
regime_relative        : three-way cone-compatible label (see
                         classify_regime_relative).

Functions
---------
extract_bipartition_spectrum : top-3 Laplacian eigenvalues + Fiedler vector.
within_half_inner_products   : mean pairwise cosine within each hemisphere.
between_half_inner_products  : mean pairwise cosine across hemispheres.
compute_separation_ratio     : between/within contrast ratio.
fiedler_boundary_fraction    : fraction of Fiedler values near zero.
centroid_angle               : angle between hemisphere centroids.
classify_regime              : antipodal four-way label (legacy vocabulary).
classify_regime_relative     : cone-compatible three-way label.
analyze_bipartition          : full pipeline across all layers.
bipartition_to_json          : JSON-serializable per-layer + summary block.
"""

from __future__ import annotations

import numpy as np

from core.metrics import fiedler_and_eigengap
from core.frames import FrameSpec, apply_frame


#: Connectivity floor this module has always used. Clipping negative Gram
#: entries can disconnect an antipodal graph entirely, leaving lambda2 = 0
#: and a degenerate Fiedler eigenspace; the floor keeps the graph connected.
#: Negligible against within-group cosines, but NOT zero, and therefore not
#: the same graph Phase 1 spectral.json was computed on.
CONNECTIVITY_FLOOR = 1e-4


# ---------------------------------------------------------------------------
# Spectrum + Fiedler extraction
# ---------------------------------------------------------------------------

def extract_bipartition_spectrum(
    activations: np.ndarray,
    clip_negative: bool = True,
    frame: FrameSpec | None = None,
    ln_params=None,
    connectivity_floor: float = CONNECTIVITY_FLOOR,
) -> dict:
    """
    First three eigenvalues of the normalized Gram-Laplacian and the Fiedler
    vector at every layer, via core.metrics.fiedler_and_eigengap.

    Parameters
    ----------
    activations        : (n_layers, n_tokens, d).
    clip_negative      : clip negative Gram entries before building the
                         Laplacian. True is what every run has used. False
                         builds the signed Laplacian, which is what this
                         module's own implementation did and which
                         core.metrics.fiedler_and_eigengap now also supports
                         — the option is carried through the delegation
                         rather than dropped by it.
    frame              : FrameSpec the Gram is built in. None => l2_sphere.
    ln_params          : required when frame.kind is an LN kind.
    connectivity_floor : see module docstring.

    Returns
    -------
    dict with eigvals (n_layers, 3), fiedler_vecs (n_layers, n_tokens),
    valid (n_layers,), and the FrameSpec used.
    """
    activations = np.asarray(activations)
    n_layers, n_tokens, _ = activations.shape
    spec = frame if frame is not None else FrameSpec(kind="l2_sphere")

    eigvals      = np.full((n_layers, 3), np.nan, dtype=np.float64)
    fiedler_vecs = np.zeros((n_layers, n_tokens), dtype=np.float64)
    valid        = np.zeros(n_layers, dtype=bool)

    if n_tokens < 4:
        return {"eigvals": eigvals, "fiedler_vecs": fiedler_vecs,
                "valid": valid, "frame": spec}

    for L in range(n_layers):
        Xf = apply_frame(activations[L], spec, ln_params)
        G  = Xf @ Xf.T
        try:
            # max_k=2 => k = 3 eigenvalues, which is exactly lambda1..lambda3.
            res = fiedler_and_eigengap(
                G, max_k=2, return_fiedler_vec=True,
                connectivity_floor=connectivity_floor,
                clip_negative=clip_negative,
            )
        except Exception:
            continue

        vals = np.asarray(res.get("eigenvalues", []), dtype=np.float64)
        fvec = res.get("fiedler_vec")
        if vals.size < 3 or fvec is None:
            continue
        vals = vals[:3]
        if not np.all(np.isfinite(vals)):
            continue
        if not (vals[0] <= vals[1] <= vals[2]):
            continue

        eigvals[L]      = vals
        fiedler_vecs[L] = np.asarray(fvec, dtype=np.float64)
        valid[L]        = True

    return {"eigvals": eigvals, "fiedler_vecs": fiedler_vecs,
            "valid": valid, "frame": spec}


# ---------------------------------------------------------------------------
# Within-hemisphere compactness
# ---------------------------------------------------------------------------

def within_half_inner_products(X: np.ndarray, assignment: np.ndarray):
    """
    Mean pairwise cosine similarity within each half.

    For L2-normed rows of X the pairwise cosine is <x_i, x_j>; the mean is
    taken over the strict upper triangle of each half's self-Gram. A half
    with fewer than 2 tokens returns nan.
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
        iu = np.triu_indices(k, k=1)
        out.append(float(G[iu].mean()))
    return out[0], out[1]


def between_half_inner_products(X: np.ndarray, assignment: np.ndarray) -> float:
    """
    Mean pairwise cosine similarity between the two halves.

    Negative  -> halves point in opposite directions (genuine separation).
    Near 0    -> roughly orthogonal halves.
    Positive  -> both halves lean the same direction; under cone-collapse
                 this is the expected case and is NOT by itself evidence
                 that the partition is cosmetic — see separation_ratio.
    """
    mask_a = assignment == 0
    mask_b = assignment == 1
    if int(mask_a.sum()) < 1 or int(mask_b.sum()) < 1:
        return float("nan")
    return float((X[mask_a] @ X[mask_b].T).mean())


def compute_separation_ratio(within_a: float, within_b: float,
                             between: float) -> float:
    """
    between_half_ip / mean(within_half_ip).

    Below 1 : cross-half pairs are less similar than same-half pairs — real
              contrast, whether or not the halves are antipodal.
    At ~1   : no structural contrast.
    Below 0 : antipodal separation.
    nan when any input is nan or the mean within-half ip is ~0.
    """
    if any(v != v for v in (within_a, within_b, between)):
        return float("nan")
    denom = 0.5 * (within_a + within_b)
    if abs(denom) < 1e-12:
        return float("nan")
    return float(between / denom)


def fiedler_boundary_fraction(fiedler_vec: np.ndarray,
                              threshold: float = 0.30) -> float:
    """
    Fraction of tokens with |v[i]| < threshold * std(v).

    Near 0 = bimodal (tokens sit deep in one half).
    Near 1 = every token hugs the partition boundary.
    """
    s = float(np.std(fiedler_vec))
    if s < 1e-12:
        return float("nan")
    return float((np.abs(fiedler_vec) < threshold * s).mean())


def centroid_angle(X: np.ndarray, assignment: np.ndarray) -> float:
    """
    Angle between the two hemisphere centroids in activation space (radians).
    nan if either half is empty or either centroid is degenerate.
    """
    mask_a = assignment == 0
    mask_b = assignment == 1
    if not mask_a.any() or not mask_b.any():
        return float("nan")

    ca, cb = X[mask_a].mean(axis=0), X[mask_b].mean(axis=0)
    na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
    if na < 1e-10 or nb < 1e-10:
        return float("nan")

    cos = float(np.dot(ca, cb) / (na * nb))
    return float(np.arccos(max(-1.0, min(1.0, cos))))


# ---------------------------------------------------------------------------
# Regime classifiers
# ---------------------------------------------------------------------------

REGIME_THRESHOLDS = {
    "collapsed_minority":   0.05,
    "weak_minority":        0.10,
    "strong_angle_rad":     np.pi / 2.0,
    "strong_within_ip":     0.30,
    "boundary_threshold":   0.30,
    # Relative-variant thresholds. separation_ratio below this counts as
    # contrast; the value is a reporting convention, and the continuous
    # separation_ratio is what a falsification table should use.
    "relative_separation":  0.90,
    "relative_weak":        0.98,
}


def classify_regime(minority_fraction: float, centroid_angle_rad: float,
                    within_half_a: float, within_half_b: float,
                    thresholds: dict | None = None) -> str:
    """
    Antipodal four-way classification — the legacy vocabulary.

      "collapsed"          minority < 0.05, or any input nan
      "weak_bipartition"   minority in [0.05, 0.1) or centroid_angle < pi/2
      "strong_bipartition" minority >= 0.1, centroid_angle >= pi/2, and
                           within_half_ip >= 0.3 in both halves
      "diffuse"            minority >= 0.1, centroid_angle >= pi/2, but at
                           least one half has within_half_ip < 0.3

    Read the pi/2 condition as what it is: a test for antipodality. Under
    cone-collapse it cannot be met, so "strong_bipartition" is close to
    unreachable and its absence is close to uninformative. Use
    classify_regime_relative alongside this, not instead of it.
    """
    th = REGIME_THRESHOLDS if thresholds is None else thresholds

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


def classify_regime_relative(minority_fraction: float, separation_ratio: float,
                             thresholds: dict | None = None) -> str:
    """
    Cone-compatible three-way classification.

      "collapsed"  minority < collapsed_minority, or any input nan
      "separated"  both halves populated AND separation_ratio <=
                   relative_separation — cross-half pairs are measurably
                   less similar than same-half pairs
      "graded"     both halves populated, separation_ratio between
                   relative_separation and relative_weak — an axis with a
                   shallow gradient rather than a partition
      "uniform"    separation_ratio >= relative_weak — the sign split
                   carries no similarity contrast at all

    This asks nothing about absolute angle, so it stays informative inside a
    single open hemisphere. The distinction it can draw that the antipodal
    classifier cannot: "separated" and "not antipodal" simultaneously, which
    is the geometry Phase 1b actually found and had no label for.
    """
    th = REGIME_THRESHOLDS if thresholds is None else thresholds

    if any(v != v for v in (minority_fraction, separation_ratio)):
        return "collapsed"
    if minority_fraction < th["collapsed_minority"]:
        return "collapsed"
    if separation_ratio <= th["relative_separation"]:
        return "separated"
    if separation_ratio < th["relative_weak"]:
        return "graded"
    return "uniform"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_bipartition(
    activations: np.ndarray,
    clip_negative: bool = True,
    frame: FrameSpec | None = None,
    ln_params=None,
    connectivity_floor: float = CONNECTIVITY_FLOOR,
) -> dict:
    """
    Run Block 0 across every layer.

    `activations` is (n_layers, n_tokens, d). All pairwise quantities are
    computed on the frame activations, not on the raw input — so a caller
    passing raw residuals with frame=FrameSpec(kind="l2_sphere") gets the
    historical behaviour, and a caller passing an LN frame gets a coherent
    LN-frame answer rather than a mixture.
    """
    activations = np.asarray(activations)
    n_layers, n_tokens, _ = activations.shape
    spec = frame if frame is not None else FrameSpec(kind="l2_sphere")

    spec_out = extract_bipartition_spectrum(
        activations, clip_negative=clip_negative, frame=spec,
        ln_params=ln_params, connectivity_floor=connectivity_floor,
    )
    eigvals      = spec_out["eigvals"]
    fiedler_vecs = spec_out["fiedler_vecs"]
    valid        = spec_out["valid"]

    assignments       = np.full((n_layers, n_tokens), -1, dtype=np.int8)
    hemisphere_sizes  = np.zeros((n_layers, 2), dtype=np.int32)
    minority_fraction = np.full(n_layers, np.nan)
    bipart_eigengap   = np.full(n_layers, np.nan)
    cen_angle         = np.full(n_layers, np.nan)
    within_ip         = np.full((n_layers, 2), np.nan)
    between_ip        = np.full(n_layers, np.nan)
    sep_ratio         = np.full(n_layers, np.nan)
    boundary_frac     = np.full(n_layers, np.nan)
    clip_frac         = np.full(n_layers, np.nan)
    regime            = np.full(n_layers, "collapsed", dtype=object)
    regime_rel        = np.full(n_layers, "collapsed", dtype=object)
    frame_acts        = np.zeros_like(activations, dtype=np.float64)

    boundary_threshold = REGIME_THRESHOLDS["boundary_threshold"]

    for L in range(n_layers):
        X = apply_frame(activations[L], spec, ln_params)
        frame_acts[L] = X

        # clip_fraction diagnostic, computed unconditionally: previously it
        # was only filled when clip_negative was True, so the "no clipping"
        # case reported nan and any assertion about it passed vacuously.
        # The fraction of negative off-diagonal entries is a property of the
        # geometry, not of what was subsequently done to it.
        G_raw = X @ X.T
        n_tri = n_tokens * (n_tokens - 1) // 2
        if n_tri > 0:
            mask = np.triu(np.ones((n_tokens, n_tokens), dtype=bool), k=1)
            clip_frac[L] = float(int((G_raw[mask] < 0).sum()) / n_tri)
        else:
            clip_frac[L] = 0.0

        if not valid[L]:
            regime[L]     = "collapsed"
            regime_rel[L] = "collapsed"
            continue

        f = fiedler_vecs[L]
        a = (f >= 0).astype(np.int8)   # sign partition: >=0 -> A(0), <0 -> B(1)
        assignments[L] = a

        na, nb = int((a == 0).sum()), int((a == 1).sum())
        hemisphere_sizes[L]  = (na, nb)
        minority_fraction[L] = min(na, nb) / n_tokens

        l2, l3 = eigvals[L, 1], eigvals[L, 2]
        if l3 > 1e-12:
            bipart_eigengap[L] = float((l3 - l2) / l3)

        cen_angle[L]  = centroid_angle(X, a)
        within_ip[L]  = within_half_inner_products(X, a)
        between_ip[L] = between_half_inner_products(X, a)
        sep_ratio[L]  = compute_separation_ratio(
            within_ip[L, 0], within_ip[L, 1], between_ip[L])
        boundary_frac[L] = fiedler_boundary_fraction(f, boundary_threshold)

        regime[L] = classify_regime(
            minority_fraction[L], cen_angle[L],
            within_ip[L, 0], within_ip[L, 1])
        regime_rel[L] = classify_regime_relative(
            minority_fraction[L], sep_ratio[L])

    return {
        "eigvals":               eigvals,
        "fiedler_vecs":          fiedler_vecs,
        "valid":                 valid,
        "assignments":           assignments,
        "hemisphere_sizes":      hemisphere_sizes,
        "minority_fraction":     minority_fraction,
        "bipartition_eigengap":  bipart_eigengap,
        "centroid_angle":        cen_angle,
        "within_half_ip":        within_ip,
        "between_half_ip":       between_ip,
        "separation_ratio":      sep_ratio,
        "fiedler_boundary_frac": boundary_frac,
        "clip_fraction":         clip_frac,
        "regime":                regime,
        "regime_relative":       regime_rel,
        "frame":                 spec,
        "frame_activations":     frame_acts,
        "connectivity_floor":    float(connectivity_floor),
        "n_layers":              n_layers,
        "n_tokens":              n_tokens,
        "thresholds":            dict(REGIME_THRESHOLDS),
    }


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def bipartition_to_json(result: dict) -> dict:
    """Flat per-layer + summary dict for the aggregator."""
    n      = result["n_layers"]
    regime = result["regime"]
    rel    = result.get("regime_relative", np.full(n, "collapsed", dtype=object))

    per_layer = []
    for L in range(n):
        per_layer.append({
            "layer":                 L,
            "valid":                 bool(result["valid"][L]),
            "regime":                str(regime[L]),
            "regime_relative":       str(rel[L]),
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

    regime_counts: dict = {}
    for r in regime:
        regime_counts[str(r)] = regime_counts.get(str(r), 0) + 1
    relative_counts: dict = {}
    for r in rel:
        relative_counts[str(r)] = relative_counts.get(str(r), 0) + 1

    valid = result["valid"]
    cf = result["clip_fraction"]

    summary = {
        "n_layers":                    n,
        "n_tokens":                    int(result["n_tokens"]),
        "n_valid_layers":              int(valid.sum()),
        "regime_counts":               regime_counts,
        "regime_relative_counts":      relative_counts,
        "strong_bipartition_fraction":
            float(regime_counts.get("strong_bipartition", 0) / n) if n else 0.0,
        "separated_fraction":
            float(relative_counts.get("separated", 0) / n) if n else 0.0,
        "mean_minority_fraction":      _mean(result["minority_fraction"][valid]),
        "mean_centroid_angle":         _mean(result["centroid_angle"][valid]),
        "mean_bipartition_eigengap":   _mean(result["bipartition_eigengap"][valid]),
        "mean_between_half_ip":        _mean(result["between_half_ip"][valid]),
        "mean_separation_ratio":       _mean(result["separation_ratio"][valid]),
        "mean_fiedler_boundary_frac":  _mean(result["fiedler_boundary_frac"][valid]),
        "mean_clip_fraction":          _mean(cf[np.isfinite(cf)]),
        "connectivity_floor":          result.get("connectivity_floor"),
        "frame":                       _frame_dict(result.get("frame")),
        "thresholds":                  result["thresholds"],
    }

    return {"per_layer": per_layer, "summary": summary}


def _frame_dict(spec) -> dict | None:
    if spec is None:
        return None
    return {
        "kind":         spec.kind,
        "model_rev":    spec.model_rev,
        "rope_applied": bool(spec.rope_applied),
        "pos0_policy":  spec.pos0_policy,
        "reader_block": spec.reader_block,
    }


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _mean(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None
