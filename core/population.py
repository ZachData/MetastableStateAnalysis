"""
core/population.py — Population selector (transition plan v2, core analysis
primitives, item 1 of 4).

Plan text: "a parameter (cluster id / 'unclustered' / 'all') for
displacement_projection, v_alignment.py, probe_subspace.py (currently
hard-drops unclustered tokens via `labels >= 0`), eigenspace_degeneracy.py,
centroid_velocity.py. With per-particle records this reduces to a filter;
the work is threading it through the five consumers."

Before this module, each of the five consumers expressed the same
token-selection logic ad hoc, inconsistently, and in a way that silently
discarded the unclustered population rather than treating it as a first-
class object of study (see the plan's "Framing: particles first"):

  - p6_subspace/probe_subspace.py::probe_accuracy       `labels >= 0`
  - p6_subspace/eigenspace_degeneracy.py::degeneracy_ratio  `labels >= 0`
    (a separate, independent copy of the same line — exactly the kind of
    drift a shared primitive is meant to prevent)
  - p6_subspace/centroid_velocity.py::centroid_velocity_profile
    `labels == cluster_id` — specific-cluster only; -1 already selects
    "unclustered" by HDBSCAN convention, but there was no way to ask for
    every token regardless of cluster.
  - p2_eigenspectra/trajectory.py::displacement_projection — no selection
    at all; every token, every time, with no way to isolate a population.
  - p5_single_mstate_analysis/v_alignment.py::cluster_energy_trajectory —
    selection only via a tracked (layer, cluster_id) chain; no
    "unclustered" option independent of tracking a specific cluster's
    identity across depth.

`resolve_population_mask` is the one function all five now call. When
core.particles.ParticleTable is the data source directly, its own
`.filter(population=...)` is the equivalent primitive (see that module's
docstring); this module exists for the many consumers that still take
raw `(activations, cluster_labels)` arrays rather than a ParticleTable,
which is every consumer as of this pass.

Population spec
----------------
A "population" argument accepted by every threaded consumer is one of:

  None            : every token, no filtering. Equivalent to "all".
                     This is every consumer's *default*, chosen so that
                     adding the parameter changes no existing call site's
                     behavior — the population selector is additive, not
                     a silent behavior change to code already relying on
                     "clustered only" (see below for the one exception).
  "all"           : every token, no filtering. Same as None; the explicit
                     spelling exists for call sites that want to say so
                     rather than rely on an implicit default.
  "clustered"     : cluster_labels >= 0. The pre-existing implicit
                     behavior of probe_accuracy and degeneracy_ratio
                     (both hard-dropped noise this way before this pass);
                     kept as an explicit, nameable option and used as
                     *their* default specifically, so that neither
                     function's behavior changes when called exactly as
                     it was before this pass (see each function's own
                     docstring for why "clustered" rather than "all" is
                     the right default there — a classification probe or
                     a within/between-cluster variance ratio needs at
                     least the *option* to exclude the population that
                     has no cluster identity, even though "all" is also
                     now available on request).
  "unclustered"   : cluster_labels < 0. The population Blog 1 identified
                     as the large, informative, previously-discarded
                     remainder — the reason this selector exists at all.
  int             : cluster_labels == population. A single specific
                     cluster id (including -1, which is "unclustered"
                     expressed as an explicit id rather than the string
                     alias — both resolve to the same mask).

`resolve_population_mask` never raises on a mask that ends up empty (no
unclustered tokens at some layer, e.g.) — that is a real, reportable
result, not an error, consistent with how the rest of core/ treats
missing data (core.artifacts.validate_artifact, core.io.write_manifest's
None-tolerant fields). Each consumer decides for itself how to handle a
too-small or empty population (most already had a "too few tokens" guard
for the clustered case; that guard now also protects the unclustered and
all-tokens cases with no special-casing needed, since resolve_population_
mask just returns a boolean array of whatever size the guard already
checks).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

# HDBSCAN noise / "unclustered" convention. Matches core.particles.
# default_population_tag and Phase 1's hdbscan_labels.json — one constant,
# not re-hardcoded at each of the (now five, previously N) call sites.
NOISE_LABEL = -1

POPULATION_ALL = "all"
POPULATION_CLUSTERED = "clustered"
POPULATION_UNCLUSTERED = "unclustered"

_STRING_POPULATIONS = (POPULATION_ALL, POPULATION_CLUSTERED, POPULATION_UNCLUSTERED)

# What a "population" argument may be, across every threaded consumer.
PopulationSpec = Optional[Union[str, int]]


def resolve_population_mask(
    cluster_labels: np.ndarray,
    population: PopulationSpec = None,
) -> np.ndarray:
    """
    Boolean mask selecting `population` out of `cluster_labels`.

    Parameters
    ----------
    cluster_labels : (n,) int array — HDBSCAN-style labels, -1 = noise.
    population     : None | "all" | "clustered" | "unclustered" | int.
                     See module docstring for the full spec. None and
                     "all" are equivalent (both select every token).

    Returns
    -------
    (n,) bool array. May be all-False (e.g. no unclustered tokens at this
    layer) — that's a valid result for the caller to handle, not an error
    raised here.

    Raises
    ------
    ValueError if `population` is a string other than the three
    recognized spellings, or anything that isn't None/str/int.
    """
    cluster_labels = np.asarray(cluster_labels)

    if population is None or population == POPULATION_ALL:
        return np.ones(cluster_labels.shape, dtype=bool)

    if population == POPULATION_CLUSTERED:
        return cluster_labels >= 0

    if population == POPULATION_UNCLUSTERED:
        return cluster_labels < 0

    if isinstance(population, (int, np.integer)) and not isinstance(population, bool):
        return cluster_labels == int(population)

    if isinstance(population, str):
        raise ValueError(
            f"resolve_population_mask: unrecognized population {population!r}. "
            f"Expected one of {_STRING_POPULATIONS}, or an int cluster id."
        )
    raise ValueError(
        f"resolve_population_mask: population must be None, a recognized "
        f"string {_STRING_POPULATIONS}, or an int cluster id; got "
        f"{type(population).__name__} ({population!r})."
    )


def population_label(population: PopulationSpec) -> str:
    """
    Human-readable tag for a population spec, for figure titles / report
    lines / manifest extras. Not used for selection — resolve_population_
    mask is the only function that interprets the spec semantically.
    """
    if population is None:
        return POPULATION_ALL
    if isinstance(population, (int, np.integer)) and not isinstance(population, bool):
        return f"cluster_{int(population)}"
    return str(population)
