"""
axis_identity.py — What the Fiedler axis actually is, and whether it is the
same object twice.

Why this module exists
----------------------
Phase 1b's finding was that the k=2 eigengap marks a real, stable *axis* in
the Gram matrix rather than a separator between two populated half-spaces.
That conclusion was reached without ever asking the cheapest follow-up
question: is the axis anything other than the direction the token cloud
already points?

Transformer residual streams are strongly anisotropic. If the Fiedler axis
in activation space is (up to sign) the top principal component, then
"leading variance direction" is not an interpretation of the result — it is
the whole result, and every downstream use of the axis as a probe feature
(Phase 5's hemisphere centroids, Phase 6's Fiedler-difference vector) is
using a rediscovery of PC1 under a more expensive name. That is a one-matmul
question and it has never been asked.

A correction to the first version of this module
-----------------------------------------------
That version asked whether the axis is the MEAN token direction, and gave
`redundancy` a "mean_direction" branch. Measured across every fixture, that
branch is unreachable: |cos(axis, mean)| came out between 0.000 and 0.085.

The reason is structural, not empirical. The Fiedler vector is the second
eigenvector of the normalized Laplacian and is therefore orthogonal to the
first, which is D^(1/2)·1 — measured <f, D^(1/2)1> is 1e-16 to 1e-1 against
||f|| = 1. A near-mean-zero coefficient vector makes X^T f cancel whatever
component every token shares, so the axis comes out near-orthogonal to the
mean by construction.

Shipping that branch would have repeated exactly the defect this revision
flags in bipartition_detect.classify_regime, where `strong_bipartition`
requires a centroid angle that cone-collapse makes unreachable, and the
resulting 0% was read as evidence. So:

  - `cos_axis_mean` is retained as a DEGENERACY DIAGNOSTIC, not a verdict.
    It should be ~0. A materially non-zero value means the Fiedler vector is
    not cleanly orthogonal to the trivial eigenvector — a disconnected graph,
    a degenerate eigenspace, or poor convergence — and the axis at that layer
    should not be trusted.
  - The redundancy question is asked against CENTERED PC1 and the top-k PC
    subspace, both of which are mean-removed and therefore commensurable
    with a mean-orthogonal axis.
  - `cos_axis_pc1` is compared against the isotropic baseline 1/sqrt(d),
    which is what |cos| between a random direction and any fixed direction
    concentrates on in d dimensions. Without that, "0.3" is not obviously
    different from chance at d = 1024.

The Fiedler vector lives in token space (one coefficient per token). Its
image in activation space is the coefficient-weighted combination of token
directions:

    a = X^T f / ||X^T f||        f = Fiedler vector, X = (n_tokens, d)

This is the direction along which the bipartition's sign pattern is realised
geometrically, and it is the object that is comparable to a mean direction, a
principal component, an OV eigenvector, or the same axis at another
checkpoint. `fiedler_vecs` alone is not comparable across any of those: two
runs with different token counts have Fiedler vectors of different lengths,
and two checkpoints of the same prompt have Fiedler vectors whose token-space
coordinates only coincide by accident of tokenization.

Cross-checkpoint use
--------------------
`compute_axis_rotation` in hemisphere_tracking.py measures rotation between
*adjacent layers within one model*. The identical statistic across
*checkpoints at a fixed layer* is what PREDICTIONS.md claim (b) needs — the
Fiedler drop is one of the three markers whose co-location with circuit
formation the 410M pilot tests, and nothing else in the project tracks the
axis's direction (as opposed to lambda_2's magnitude) over training. The
functions here take activation-space axes precisely so that comparison is
well-posed across checkpoints with different weights.

Everything in this module is pure numpy.
"""

from __future__ import annotations

import numpy as np


#: |cos(axis, mean)| above which the axis is treated as numerically suspect
#: rather than informative. The Fiedler vector is orthogonal to the
#: Laplacian's trivial eigenvector, so the axis is mean-orthogonal by
#: construction; a large value means that construction broke down.
MEAN_ORTHOGONALITY_TOL = 0.5

#: |cos(axis, PC1)| at or above which the axis IS PC1 for reporting purposes.
PC1_TOL = 0.9

#: Fraction of the axis inside span(PC1..PCk) at or above which the axis is
#: inside the leading variance block.
PC_BLOCK_TOL = 0.9


# ---------------------------------------------------------------------------
# Token space -> activation space
# ---------------------------------------------------------------------------

def axis_in_activation_space(
    X: np.ndarray,
    fiedler_vec: np.ndarray,
    center: bool = False,
) -> np.ndarray:
    """
    Map a token-space Fiedler vector to a unit direction in activation space.

    Parameters
    ----------
    X           : (n_tokens, d) activations in whatever frame the caller
                  chose. Not re-normalised here — the frame is the caller's
                  recorded decision (core/frames.py).
    fiedler_vec : (n_tokens,) second Laplacian eigenvector.
    center      : if True, subtract the token mean from X before projecting.
                  The uncentered axis answers "which direction realises the
                  sign pattern"; the centered one answers "which direction
                  realises it once the cloud's own mean offset is removed".
                  Both are reported by `axis_alignment`.

    Returns
    -------
    (d,) unit vector, or a zero vector if the projection is degenerate.

    Sign convention
    ---------------
    An eigenvector's global sign is arbitrary, so the axis must be
    canonicalised or no two axes are comparable. The canonicalisation is
    applied to f BEFORE projecting: f is flipped so that its largest-magnitude
    component is positive.

    The obvious alternative — orient the axis by the centroid difference
    between the positive-f and negative-f groups — does not work, and the
    first version of this module used it. Under f -> -f the axis negates AND
    the two groups swap, so the centroid difference negates too and the test
    `axis . delta >= 0` gives the same answer both times. The rule is
    self-consistent and still lets the output flip with its input.

    Orienting on sum(f) would be worse still: sum(f) is approximately zero by
    construction, since the Fiedler vector is orthogonal to the Laplacian's
    trivial eigenvector.
    """
    X = np.asarray(X, dtype=np.float64)
    f = np.asarray(fiedler_vec, dtype=np.float64).ravel()
    if X.ndim != 2 or f.shape[0] != X.shape[0]:
        raise ValueError(
            f"shape mismatch: X is {X.shape}, fiedler_vec is {f.shape}"
        )

    f = canonical_sign(f)

    Xc = X - X.mean(axis=0, keepdims=True) if center else X
    a = Xc.T @ f
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        return np.zeros(X.shape[1], dtype=np.float64)
    return a / norm


def canonical_sign(f: np.ndarray) -> np.ndarray:
    """
    Flip an eigenvector so its largest-magnitude component is positive.

    Deterministic and invariant under f -> -f, which is the only property
    required. Ties (exactly equal magnitudes of opposite sign) resolve to the
    first such index via argmax, which is stable for a fixed input.
    """
    f = np.asarray(f, dtype=np.float64).ravel()
    if f.size == 0:
        return f
    i = int(np.argmax(np.abs(f)))
    return -f if f[i] < 0 else f


def mean_direction(X: np.ndarray) -> np.ndarray:
    """Unit vector along the token mean. Zero vector if the mean vanishes."""
    m = np.asarray(X, dtype=np.float64).mean(axis=0)
    n = float(np.linalg.norm(m))
    return m / n if n > 1e-12 else np.zeros_like(m)


def principal_components(X: np.ndarray, k: int = 3, center: bool = True) -> dict:
    """
    Top-k principal directions and their explained-variance fractions.

    center=True is the standard PCA convention. center=False gives the top
    right-singular vectors of the raw matrix, whose first component on an
    anisotropic cloud is essentially the mean direction — reported separately
    because that near-degeneracy is exactly what this module is testing for.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True) if center else X
    k = int(min(k, min(Xc.shape)))
    if k < 1:
        d = X.shape[1]
        return {"components": np.zeros((0, d)), "explained_variance_ratio": np.zeros(0)}

    try:
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        d = X.shape[1]
        return {"components": np.zeros((0, d)), "explained_variance_ratio": np.zeros(0)}

    var = S ** 2
    total = float(var.sum())
    ratio = (var / total) if total > 1e-12 else np.zeros_like(var)
    return {
        "components":               Vt[:k],
        "explained_variance_ratio": ratio[:k],
    }


def _abs_cos(u: np.ndarray, v: np.ndarray) -> float:
    """|cos| between two vectors; nan if either is degenerate."""
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    c = float(np.dot(u, v) / (nu * nv))
    return float(abs(max(-1.0, min(1.0, c))))


# ---------------------------------------------------------------------------
# The actual test
# ---------------------------------------------------------------------------

def axis_alignment(
    X: np.ndarray,
    fiedler_vec: np.ndarray,
    n_components: int = 3,
) -> dict:
    """
    Is the Fiedler axis distinguishable from the cloud's own leading geometry?

    Returns
    -------
    dict with:
      axis                    (d,) uncentered activation-space axis
      axis_centered           (d,) axis computed on mean-subtracted X
      cos_axis_mean           |cos| between axis and the mean token direction.
                              DIAGNOSTIC, not a verdict: expected ~0 because
                              the Fiedler vector is orthogonal to the
                              Laplacian's trivial eigenvector. See the module
                              docstring.
      cos_axis_pc1            |cos| between axis and centered PC1
      cos_axis_pc1_uncentered |cos| between axis and uncentered top right-
                              singular vector
      cos_axis_centered_pc1   |cos| between the centered axis and centered PC1
      cos_mean_pc1            |cos| between mean direction and centered PC1.
      pc_subspace_fraction    fraction of the axis's squared norm lying in
                              span(PC1..PCk). This is the question "is the
                              axis inside the leading variance block", which
                              a single cosine against PC1 cannot answer when
                              the top eigenvalues are close together and the
                              individual components are not identifiable.
      isotropic_cos           1/sqrt(d) — where |cos| between two unrelated
                              directions concentrates in d dimensions. Any
                              cosine below a small multiple of this is chance.
      pc_explained            (n_components,) explained-variance ratios
      redundancy              str verdict, see below

    `redundancy` is one of:
      "pc1"            cos_axis_pc1 >= 0.9 — the axis IS the leading variance
                       direction.
      "top_pc_block"   not pc1, but pc_subspace_fraction >= 0.9 — the axis
                       lives inside the top-k principal subspace without
                       being any single component of it. Still not new
                       information, but a weaker statement than "pc1".
      "distinct"       the axis leaves the top-k block — it carries structure
                       the cloud's leading variance geometry does not supply,
                       and is worth its own probe.
      "degenerate"     inputs too small, projections vanish, or the axis is
                       not cleanly mean-orthogonal (see cos_axis_mean).

    The 0.9 thresholds are reporting conveniences, not tests. The continuous
    quantities are the result; anything downstream that needs a decision
    should take them and a null (see nulls.py usage in cone_collapse.py for
    the pattern) rather than this string.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 3:
        return {
            "axis": None, "axis_centered": None,
            "cos_axis_mean": None, "cos_axis_pc1": None,
            "cos_axis_pc1_uncentered": None, "cos_axis_centered_pc1": None,
            "cos_mean_pc1": None, "pc_subspace_fraction": None,
            "isotropic_cos": None,
            "pc_explained": None,
            "redundancy": "degenerate",
        }

    axis      = axis_in_activation_space(X, fiedler_vec, center=False)
    axis_cen  = axis_in_activation_space(X, fiedler_vec, center=True)
    mu        = mean_direction(X)

    pca_c = principal_components(X, k=n_components, center=True)
    pca_u = principal_components(X, k=1,            center=False)

    pc1   = pca_c["components"][0] if pca_c["components"].shape[0] else np.zeros(X.shape[1])
    pc1_u = pca_u["components"][0] if pca_u["components"].shape[0] else np.zeros(X.shape[1])

    cos_mean = _abs_cos(axis, mu)
    cos_pc1  = _abs_cos(axis, pc1)

    # Fraction of the axis inside the top-k principal subspace. Robust where
    # a single cosine is not: when the leading eigenvalues are close, the
    # individual components are not identifiable but their span is.
    comps = pca_c["components"]
    if comps.shape[0] and np.linalg.norm(axis) > 1e-12:
        proj = comps @ axis
        pc_frac = float(np.clip(np.dot(proj, proj), 0.0, 1.0))
    else:
        pc_frac = float("nan")

    isotropic = float(1.0 / np.sqrt(X.shape[1])) if X.shape[1] else float("nan")

    if np.linalg.norm(axis) < 1e-12:
        redundancy = "degenerate"
    elif np.isfinite(cos_mean) and cos_mean >= MEAN_ORTHOGONALITY_TOL:
        # Not "the axis is the mean" — the axis is supposed to be
        # mean-orthogonal, so this says the Fiedler vector is not clean.
        redundancy = "degenerate"
    elif np.isfinite(cos_pc1) and cos_pc1 >= PC1_TOL:
        redundancy = "pc1"
    elif np.isfinite(pc_frac) and pc_frac >= PC_BLOCK_TOL:
        redundancy = "top_pc_block"
    else:
        redundancy = "distinct"

    return {
        "axis":                    axis,
        "axis_centered":           axis_cen,
        "cos_axis_mean":           cos_mean,
        "cos_axis_pc1":            cos_pc1,
        "cos_axis_pc1_uncentered": _abs_cos(axis, pc1_u),
        "cos_axis_centered_pc1":   _abs_cos(axis_cen, pc1),
        "cos_mean_pc1":            _abs_cos(mu, pc1),
        "pc_subspace_fraction":    pc_frac,
        "isotropic_cos":           isotropic,
        "pc_explained":            pca_c["explained_variance_ratio"],
        "redundancy":              redundancy,
    }


def analyze_axis_identity(
    activations: np.ndarray,
    fiedler_vecs: np.ndarray,
    valid: np.ndarray | None = None,
    n_components: int = 3,
) -> dict:
    """
    Run `axis_alignment` at every valid layer.

    Parameters
    ----------
    activations  : (n_layers, n_tokens, d)
    fiedler_vecs : (n_layers, n_tokens)
    valid        : (n_layers,) bool, or None for all-valid.

    Returns
    -------
    dict of per-layer arrays plus `axes` (n_layers, d) for cross-checkpoint
    comparison, and a summary carrying the modal redundancy verdict.
    """
    n_layers, n_tokens, d = activations.shape
    if valid is None:
        valid = np.ones(n_layers, dtype=bool)

    axes        = np.zeros((n_layers, d), dtype=np.float64)
    cos_mean    = np.full(n_layers, np.nan)
    cos_pc1     = np.full(n_layers, np.nan)
    cos_pc1_u   = np.full(n_layers, np.nan)
    cos_cen_pc1 = np.full(n_layers, np.nan)
    cos_mu_pc1  = np.full(n_layers, np.nan)
    pc_frac     = np.full(n_layers, np.nan)
    pc1_var     = np.full(n_layers, np.nan)
    redundancy  = np.full(n_layers, "degenerate", dtype=object)

    for L in range(n_layers):
        if not valid[L]:
            continue
        r = axis_alignment(activations[L], fiedler_vecs[L], n_components=n_components)
        if r["axis"] is not None:
            axes[L] = r["axis"]
        cos_mean[L]    = _nan(r["cos_axis_mean"])
        cos_pc1[L]     = _nan(r["cos_axis_pc1"])
        cos_pc1_u[L]   = _nan(r["cos_axis_pc1_uncentered"])
        cos_cen_pc1[L] = _nan(r["cos_axis_centered_pc1"])
        cos_mu_pc1[L]  = _nan(r["cos_mean_pc1"])
        pc_frac[L]     = _nan(r["pc_subspace_fraction"])
        pcv = r["pc_explained"]
        if pcv is not None and len(pcv):
            pc1_var[L] = float(pcv[0])
        redundancy[L] = r["redundancy"]

    counts: dict[str, int] = {}
    for r in redundancy:
        counts[str(r)] = counts.get(str(r), 0) + 1
    modal = max(counts, key=counts.get) if counts else "degenerate"

    return {
        "axes":                     axes,
        "cos_axis_mean":            cos_mean,
        "cos_axis_pc1":             cos_pc1,
        "cos_axis_pc1_uncentered":  cos_pc1_u,
        "cos_axis_centered_pc1":    cos_cen_pc1,
        "cos_mean_pc1":             cos_mu_pc1,
        "pc_subspace_fraction":     pc_frac,
        "pc1_explained_variance":   pc1_var,
        "redundancy":               redundancy,
        "redundancy_counts":        counts,
        "modal_redundancy":         modal,
        "n_layers":                 n_layers,
        "n_tokens":                 n_tokens,
        "d":                        d,
    }


# ---------------------------------------------------------------------------
# Across checkpoints
# ---------------------------------------------------------------------------

def cross_checkpoint_axis_rotation(
    axes_by_step: dict[int, np.ndarray],
    reference: str = "adjacent",
) -> dict:
    """
    Angle between activation-space axes at consecutive training checkpoints,
    at one fixed layer.

    Parameters
    ----------
    axes_by_step : {checkpoint_step: (d,) unit axis}. Steps need not be
                   sorted; they are sorted here.
    reference    : "adjacent" — angle between step_i and step_{i+1}.
                   "final"    — angle between step_i and the largest step,
                                which is the "when does the axis reach its
                                trained direction" reading.

    Returns
    -------
    dict with steps, rotation (radians), and the log-step axis positions the
    derivative should be taken against. Pythia's checkpoints are log-spaced
    to step 512 and linear after, so a derivative over checkpoint *index*
    puts its largest values wherever the release schedule changes spacing —
    core/checkpoint_frames.py makes the same point for scalar metrics and
    this is the vector-valued case of it.
    """
    steps = sorted(int(s) for s in axes_by_step)
    if len(steps) < 2:
        return {
            "steps": steps, "rotation": np.zeros(0), "pair_steps": [],
            "log_step": np.log10(np.array(steps, dtype=np.float64) + 1.0)
                        if steps else np.zeros(0),
            "reference": reference,
        }

    log_step = np.log10(np.array(steps, dtype=np.float64) + 1.0)

    rot: list[float] = []
    pairs: list[tuple[int, int]] = []
    if reference == "final":
        target = axes_by_step[steps[-1]]
        for s in steps[:-1]:
            rot.append(_angle(axes_by_step[s], target))
            pairs.append((s, steps[-1]))
    else:
        for a, b in zip(steps[:-1], steps[1:]):
            rot.append(_angle(axes_by_step[a], axes_by_step[b]))
            pairs.append((a, b))

    return {
        "steps":      steps,
        "log_step":   log_step,
        "rotation":   np.array(rot, dtype=np.float64),
        "pair_steps": pairs,
        "reference":  reference,
    }


def axis_settling_step(
    rotation_to_final: dict,
    tol_rad: float = np.pi / 8.0,
) -> int | None:
    """
    Earliest checkpoint from which the axis stays within `tol_rad` of its
    final direction. None if it never does.

    This is the quantity PREDICTIONS.md claim (b) is about, stated as a step
    rather than as a plot: if the axis settles around 512-2,000 it co-locates
    with the energy-monotonicity break and the Fiedler drop; if it settles at
    step 0 the axis is initialisation geometry and training does not move it;
    if it never settles the "stable axis" reading from the ALBERT/GPT-2 runs
    does not transfer to a checkpoint family.
    """
    steps = rotation_to_final.get("steps") or []
    rot   = np.asarray(rotation_to_final.get("rotation", []), dtype=np.float64)
    if len(steps) < 2 or rot.size == 0:
        return None
    # rotation[i] pairs steps[i] with the final step.
    within = rot <= float(tol_rad)
    for i in range(len(within)):
        if within[i] and bool(np.all(within[i:])):
            return int(steps[i])
    return None


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    c = _abs_cos(u, v)
    return float(np.arccos(c)) if np.isfinite(c) else float("nan")


def _nan(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def axis_identity_to_json(result: dict) -> dict:
    """Per-layer + summary dict. `axes` is dropped — it belongs in npz."""
    n = result["n_layers"]
    per_layer = [
        {
            "layer":                   L,
            "cos_axis_mean":           _f(result["cos_axis_mean"][L]),
            "cos_axis_pc1":            _f(result["cos_axis_pc1"][L]),
            "cos_axis_pc1_uncentered": _f(result["cos_axis_pc1_uncentered"][L]),
            "cos_axis_centered_pc1":   _f(result["cos_axis_centered_pc1"][L]),
            "cos_mean_pc1":            _f(result["cos_mean_pc1"][L]),
            "pc_subspace_fraction":    _f(result["pc_subspace_fraction"][L]),
            "pc1_explained_variance":  _f(result["pc1_explained_variance"][L]),
            "redundancy":              str(result["redundancy"][L]),
        }
        for L in range(n)
    ]
    return {
        "per_layer": per_layer,
        "summary": {
            "n_layers":           n,
            "modal_redundancy":   result["modal_redundancy"],
            "redundancy_counts":  result["redundancy_counts"],
            "mean_cos_axis_mean": _mean(result["cos_axis_mean"]),
            "mean_cos_axis_pc1":  _mean(result["cos_axis_pc1"]),
            "mean_cos_mean_pc1":  _mean(result["cos_mean_pc1"]),
            "mean_pc_subspace_fraction": _mean(result["pc_subspace_fraction"]),
        },
    }


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _mean(arr) -> float | None:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None
