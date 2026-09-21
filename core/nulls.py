"""
core/nulls.py — Null distributions as first-class outputs (transition
plan v2, core infrastructure item 7).

Formalizes the control pattern the project already uses informally
(e.g. Phase 1's random-weight controls, Phase 5's collapse/disperse
controls): permutation and shuffled-dimension nulls for graph and
clustering metrics, plus the "Nσ from null" summary a falsification
table adjudicates against.

Two null constructions are provided, matching the two things this
project actually asks "is this above chance":

  shuffled_dimension_null  — is this geometric structure (energy,
      Fiedler, mass-near-1, ...) more than what the same per-dimension
      marginals would produce with no cross-token correlation? Built by
      independently permuting each feature dimension across tokens,
      which destroys joint geometric structure while preserving each
      dimension's own value distribution, then re-normalizing onto the
      sphere (matching layernorm_to_sphere) since these metrics are
      defined for unit-norm activations.

  label_permutation_null    — is this cluster/graph assignment doing
      better than a random relabeling of the same tokens? Built by
      permuting which token holds which cluster label while activations
      stay fixed, for metrics of the form metric_fn(activations, labels).

Both return raw null-value arrays; `sigma_from_null` turns an observed
value plus a null array into the "Nσ from null" summary a STATUS.md
falsification table reports.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def _rng_or(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


# ---------------------------------------------------------------------------
# Null constructions
# ---------------------------------------------------------------------------

def shuffled_dimension_null(
    activations: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_shuffles: int = 200,
    renormalize: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Null distribution for a geometric metric (energy, Fiedler,
    mass_near_1, effective_rank, ...) computed on (n_tokens, d)
    activations.

    Each shuffle independently permutes the token axis *within each
    feature dimension separately* (not one shared permutation of rows) —
    this is what actually destroys the cross-token geometric structure
    that these metrics measure, since a single shared row-permutation is
    just a relabeling and leaves every pairwise inner product, and hence
    every metric here, exactly unchanged.

    Parameters
    ----------
    activations : (n_tokens, d) array — normally one layer's hidden states.
    metric_fn   : callable, (n_tokens, d) array -> float. Typically one of
                  core.metrics's functions composed with normalization,
                  e.g. `lambda X: interaction_energy(gram_matrix(l2_normalize(X)), beta)`.
    n_shuffles  : number of independent null draws.
    renormalize : re-apply L2 normalization after shuffling (default True,
                  matching layernorm_to_sphere) since every metric this
                  project defines assumes unit-norm rows; set False only
                  if metric_fn already normalizes internally.

    Returns
    -------
    (n_shuffles,) array of the metric evaluated on each shuffled draw.
    """
    rng = _rng_or(rng)
    activations = np.asarray(activations, dtype=np.float64)
    n_tokens, d = activations.shape

    out = np.empty(n_shuffles, dtype=np.float64)
    for i in range(n_shuffles):
        shuffled = np.empty_like(activations)
        for col in range(d):
            shuffled[:, col] = activations[rng.permutation(n_tokens), col]
        if renormalize:
            norms = np.linalg.norm(shuffled, axis=1, keepdims=True)
            shuffled = shuffled / np.maximum(norms, 1e-12)
        out[i] = metric_fn(shuffled)
    return out


def label_permutation_null(
    activations: np.ndarray,
    labels: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Null distribution for a label-conditioned metric (e.g. "fraction of
    mass-near-1 pairs that are cluster-internal", cluster_profile.py's
    `_mass_near_1_contribution`) by permuting which token holds which
    label while activations stay fixed. Answers "does this label
    assignment do better than a random relabeling of the same tokens
    into the same-sized groups," which is the relevant chance baseline
    for a cluster-conditioned statistic (as opposed to
    shuffled_dimension_null, which asks whether the geometry itself is
    structured at all).

    Parameters
    ----------
    activations    : (n_tokens, d) array, held fixed across permutations.
    labels         : (n_tokens,) int array, the real label assignment.
    metric_fn      : callable, (activations, labels) -> float.
    n_permutations : number of independent label permutations.

    Returns
    -------
    (n_permutations,) array of the metric evaluated on each permuted
    label assignment.
    """
    rng = _rng_or(rng)
    labels = np.asarray(labels)
    n = len(labels)

    out = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        permuted = labels[rng.permutation(n)]
        out[i] = metric_fn(activations, permuted)
    return out


# ---------------------------------------------------------------------------
# Nσ-from-null summary
# ---------------------------------------------------------------------------

def sigma_from_null(observed: float, null_values: np.ndarray) -> dict:
    """
    Turn an observed value and a null-value array into the "Nσ from
    null" summary a falsification table adjudicates against.

    Returns
    -------
    dict:
      observed    : float, echoed back for provenance
      null_mean   : float
      null_std    : float
      z_score     : (observed - null_mean) / null_std; inf-safe (nan if
                     null_std is 0 — a degenerate null, not a division
                     to paper over with a fallback number)
      percentile  : observed's percentile rank within the null distribution
                     (0-100; 50 = indistinguishable from the null's median)
      n_null      : len(null_values)
    """
    null_values = np.asarray(null_values, dtype=np.float64)
    null_mean = float(np.mean(null_values))
    null_std = float(np.std(null_values))
    # A degenerate (effectively constant) null has std ~1e-16 due to
    # floating-point subtraction, not exactly 0.0 — guard with a small
    # tolerance rather than `> 0`, or a near-constant null would produce
    # a wildly inflated z-score instead of the honest "undefined" signal.
    z_score = float((observed - null_mean) / null_std) if null_std > 1e-9 else float("nan")
    percentile = float(100.0 * np.mean(null_values <= observed))

    return {
        "observed": float(observed),
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "percentile": percentile,
        "n_null": int(len(null_values)),
    }


def nsigma_verdict(
    observed: float,
    null_values: np.ndarray,
    sigma_threshold: float = 2.0,
) -> dict:
    """
    `sigma_from_null` plus a significance call and a ready-to-paste
    verdict string for a STATUS.md falsification table, e.g.
    "3.4σ from null (significant)".
    """
    summary = sigma_from_null(observed, null_values)
    z = summary["z_score"]
    significant = (not np.isnan(z)) and abs(z) >= sigma_threshold
    summary["sigma_threshold"] = sigma_threshold
    summary["significant"] = bool(significant)
    tag = "significant" if significant else "not significant"
    z_str = "nan" if np.isnan(z) else f"{z:.1f}"
    summary["verdict_str"] = f"{z_str}σ from null ({tag})"
    return summary


# ---------------------------------------------------------------------------
# P-values from a sampled null (POPPER_PLAN.md item B6)
# ---------------------------------------------------------------------------

def p_from_null(
    observed: float,
    null_values: np.ndarray,
    alternative: str = "greater",
) -> dict:
    """
    A Monte-Carlo p-value for an observed statistic against a sampled null.

    This is the bridge between this module and `core/adjudication.py`: an
    e-value needs a p-value that is *valid under H0*, and a null sampled by
    the same procedure that produced the observation is the construction this
    project already uses everywhere (`shuffled_dimension_null`,
    `label_permutation_null`, `p1c_frames.centroids.random_band`). What was
    missing was the last step, so every such null stopped at "Nσ from null" --
    a summary that reads like significance without being calibrated as one.

    The `+1` in both numerator and denominator is not a rounding convenience.
    Without it a statistic more extreme than every draw returns p = 0, which
    calibrates to an infinite e-value and asserts more evidence than a finite
    sample can carry. With it the smallest attainable p is 1/(n+1), which is
    the honest resolution limit of `n` draws, and it makes the estimator
    conservatively valid (Phipson & Smyth 2010) rather than anticonservative.

    `resolution` IS THE DRAW COUNT'S LIMIT AND NOT THE DESIGN'S FLOOR, AND ON A
    DISCRETE STATISTIC THE TWO ARE DIFFERENT (2026-08-27)

    1/(n+1) is what `n` draws can resolve. It is the smallest p a run can
    actually express only when the statistic is continuous, so that ties with
    the observation have probability zero. When the statistic is discrete --
    a rate difference over tens of heads, a correlation against a binary
    series, a sign sum bounded by the pair count -- the null puts a lump of
    mass exactly on the observed value, and the smallest expressible p is set
    by the DATA'S OWN MARGINALS rather than by how many draws were taken.
    Taking more draws does not move it.

    The gap is not small. `P-T1` at five heads and two candidates has an exact
    floor of 0.100 while this function reports a resolution of 0.0005 -- two
    hundred times smaller than anything that design can express -- and `P-M1`
    at twelve layers with one violation has one of 0.083 against the same
    0.0005. Both were reporting "not significant" from designs that could not
    have rejected. `claims/audits/p_t1_p_m1_dry_run.json` measures it and
    `POPPER_PLAN.md` 6p reads it; `p_t1_attainable_floor` and
    `p_m1_attainable_floor` are the two worked examples, and a caller with a
    discrete statistic owes its design one of its own.

    So `at_resolution_floor` answers "should I draw more?" and never "could
    this design have rejected?". Nothing here can compute the second, because
    it depends on the statistic's support and this function only ever sees a
    sample of it.

    Parameters
    ----------
    observed : float
        The statistic computed on the real data.
    null_values : array-like
        Draws of the same statistic under H0, from the same procedure.
    alternative : {"greater", "less", "two-sided"}
        Direction the prediction claims. MUST be fixed before seeing the data:
        choosing it afterward is a one-bit selection that doubles the effective
        error rate, and it is the cheapest possible way to void the guarantee
        the e-value is supposed to provide. The registry's `null_construction`
        field is where that choice is recorded.

    Returns
    -------
    dict
        observed, p_value, n_null, resolution (the 1/(n+1) DRAW-COUNT limit --
        see the note above; it is not the design's attainable floor when the
        statistic is discrete), at_resolution_floor (True when p is the
        smallest this SAMPLE can express -- a signal to draw more, not a
        stronger result and not a statement about the design), plus the
        `sigma_from_null` summary for continuity with existing reports.

    Raises
    ------
    ValueError
        On an unknown alternative, an empty null, or a non-finite observation.
        Refusing rather than defaulting: a p-value silently computed against
        the wrong tail is indistinguishable in the artifact from a correct one.
    """
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError(
            f"alternative must be 'greater', 'less' or 'two-sided'; got {alternative!r}"
        )
    null_values = np.asarray(null_values, dtype=np.float64)
    if null_values.size == 0:
        raise ValueError("null_values is empty; a p-value needs a sampled null")
    if not np.isfinite(observed):
        raise ValueError(f"observed statistic is not finite: {observed!r}")

    finite = null_values[np.isfinite(null_values)]
    if finite.size == 0:
        raise ValueError("null_values contains no finite draws")
    n = finite.size

    if alternative == "greater":
        n_extreme = int(np.sum(finite >= observed))
    elif alternative == "less":
        n_extreme = int(np.sum(finite <= observed))
    else:
        centre = float(np.median(finite))
        n_extreme = int(np.sum(np.abs(finite - centre) >= abs(observed - centre)))

    p_value = (n_extreme + 1.0) / (n + 1.0)
    resolution = 1.0 / (n + 1.0)

    out = sigma_from_null(observed, finite)
    out.update({
        "p_value": float(p_value),
        "alternative": alternative,
        "n_null_finite": n,
        "n_null_dropped": int(null_values.size - n),
        "resolution": float(resolution),
        "at_resolution_floor": bool(p_value <= resolution + 1e-12),
        "_resolution_note": (
            "resolution is 1/(n_draws+1), the limit of THIS SAMPLE. On a "
            "discrete statistic the smallest p the DESIGN can express is set "
            "by the data's marginals and is often far larger; see this "
            "function's docstring and p_t1_attainable_floor / "
            "p_m1_attainable_floor."),
    })
    return out


def label_permutation_null_within(
    fixed: np.ndarray,
    labels: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 200,
    rng: Optional[np.random.Generator] = None,
    noise_label: int = -1,
) -> np.ndarray:
    """
    `label_permutation_null`, restricted to the NON-NOISE positions.

    The ordinary version permutes labels among all tokens, so it answers "is
    this labelling special among all relabellings of the same sizes". That is
    the right question for a cluster-conditioned statistic, and the wrong one
    the moment the clustered and unclustered populations differ systematically
    in whatever `fixed` holds -- because then any statistic computed over
    clusters inherits that difference whole.

    FOUND, not anticipated (2026-09-20). Phase 10's F0 asks whether cluster
    nuclei are early tokens. Against the ordinary null the answer came back
    "no, they are systematically LATE". But the same sweep shows clustered
    tokens sitting later than unclustered ones on average, and if the clustered
    population as a whole is late then its clusters' earliest members are late
    for that reason alone, with nothing said about nucleation.

    This null holds the clustered/noise split FIXED and shuffles only which
    clustered token carries which cluster id. It therefore asks the question
    F0 actually means: **given which tokens are clustered, are the cluster
    seeds early?** Run both; they answer different questions and disagreeing is
    informative rather than a problem.

    Parameters
    ----------
    fixed : array held constant across permutations and passed to `metric_fn`
        as its first argument (positions, activations, a received-attention
        vector -- whatever the statistic reads).
    labels : (n_tokens,) int array, ``noise_label`` for unclustered.
    metric_fn : callable, ``(fixed, labels) -> float``.
    n_permutations : number of independent restricted permutations.
    noise_label : the value marking unclustered tokens. Default -1, HDBSCAN's.

    Returns
    -------
    (n_permutations,) array of the metric under the restricted null. When
    fewer than two positions are non-noise there is nothing to permute and the
    array is filled with the observed value, so a caller's p-value comes back
    at 1 rather than at a floor -- the same refusal-to-invent-evidence rule
    `p_from_null_tolerant` follows.
    """
    rng = _rng_or(rng)
    labels = np.asarray(labels)
    idx = np.flatnonzero(labels != noise_label)

    out = np.empty(n_permutations, dtype=np.float64)
    if idx.size < 2:
        out[:] = float(metric_fn(fixed, labels))
        return out

    for i in range(n_permutations):
        permuted = labels.copy()
        permuted[idx] = labels[idx][rng.permutation(idx.size)]
        out[i] = metric_fn(fixed, permuted)
    return out


def p_from_null_tolerant(
    observed: float,
    null_values: np.ndarray,
    alternative: str = "greater",
    rtol: float = 1e-9,
) -> dict:
    """
    `p_from_null` with a tie tolerance, for statistics whose null can be
    DEGENERATE at floating-point resolution.

    WHY THIS EXISTS
    ---------------
    `p_from_null` compares draws to the observation with an exact `>=`. That is
    correct for a continuous statistic, and wrong in one specific and easily
    missed case: when the statistic cannot actually vary under the null, so
    every draw equals the observation up to the last few bits of the mantissa,
    and the comparison is then decided by rounding noise rather than by data.

    FOUND, not anticipated (2026-09-20). Phase 10's row A0 divides the causal
    mask's structural tilt out of received attention. On a content-free
    attention matrix the corrected value is exactly 1 at every token, so every
    permutation of the labels gives an enrichment of exactly 1 -- the honest
    answer is p = 1, "the labels explain nothing". `p_from_null` instead
    returned **0.0025, the resolution floor**, because ~1.0 differs from ~1.0
    in the sixteenth digit and half the draws fell on the convenient side. A
    perfectly explained layer read as the strongest possible evidence.

    THE RULE
    --------
    A draw within ``tol = rtol * scale`` of the observation counts as a TIE and
    is counted as NOT LESS EXTREME -- it goes into the numerator. That is the
    conservative direction in every `alternative`: ties raise the p-value, so
    the failure mode is missing a real effect rather than inventing one. A
    fully degenerate null snaps entirely onto the observation and returns
    exactly p = 1, which is the right answer and a loud one.

    ``scale`` is ``max(|observed|, max|draw|, 1.0)``, so the tolerance is
    relative for large statistics and absolute for ones near zero. The default
    ``rtol = 1e-9`` sits seven orders above float64 noise and far below any
    effect this project would call one.

    Adds ``n_ties`` and ``degenerate_null`` to `p_from_null`'s record, so a
    result that came back p = 1 because nothing could vary is distinguishable
    afterwards from one that came back p = 1 because the effect was absent.
    """
    if rtol < 0.0:
        raise ValueError(f"rtol must be non-negative; got {rtol!r}")

    draws = np.asarray(null_values, dtype=np.float64)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        # Let p_from_null raise its own message rather than duplicating it.
        return p_from_null(observed, draws, alternative=alternative)

    scale = max(abs(float(observed)), float(np.max(np.abs(finite))), 1.0)
    tol = rtol * scale
    near = np.abs(finite - float(observed)) <= tol

    snapped = draws.astype(np.float64, copy=True)
    finite_idx = np.flatnonzero(np.isfinite(draws))
    snapped[finite_idx[near]] = float(observed)

    out = p_from_null(observed, snapped, alternative=alternative)
    out["n_ties"] = int(near.sum())
    out["tie_tolerance"] = float(tol)
    out["degenerate_null"] = bool(near.all())
    return out
