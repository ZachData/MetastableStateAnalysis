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
