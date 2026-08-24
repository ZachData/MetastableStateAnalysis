"""
core/changepoint_colocation.py — the changepoint co-location construction, and
CLAIM-B's gate.

TWO REGISTRY ENTRIES, ONE CONSTRUCTION

    CLAIM-B (H-EMERGE)  the energy-monotonicity break and the Fiedler drop
                        co-locate with steps ~512-2000.
    P-I1    (H-BRIDGE)  `relay` motif strength above N1 and N2 first rises in
                        the same checkpoint window as the behavioral induction
                        score.

`claims/EVALUABILITY.md` closes on these two: *"CLAIM-B is next by the same
reasoning, and it shares a construction with P-I1 -- the same changepoint
co-location across a checkpoint sweep -- so the two should be built together
rather than each inventing one."* This module is that construction; CLAIM-B's
gate is here because `CLAIMS.md` names `core/checkpoint_frames.py` as
H-EMERGE's instrument, and P-I1's is in `p7_motifs/formation_gate.py` because
`PREDICTIONS.md` names `p7_motifs/motif_stats.py` as its.

They sit under DIFFERENT claims, so unlike `P5b-B1`/`P5b-B3` there is no
double-counting problem. But one shared estimator is a common-cause failure
mode -- an estimator defect moves both -- and `P6-R2`/`P6-R4`'s precedent is to
record that in `null_construction` rather than leave it inferable.

THE ESTIMATOR IS NOT `detect_transitions`, AND THE REASON WAS MEASURED

`checkpoint_frames.detect_transitions` returns the INTERVALS of largest change
per unit log-step. Adopting it is the reuse this project prefers, and it fixes
the choice in advance for free. It was checked first and it cannot carry this
test, for a reason that only shows up when the floor is computed:

**A binary co-location statistic has no usable floor here.** The log-step
geometry is not uniform -- Pythia's every-1000 releases compress to
d log10(step+1) = 0.065 at the top of the sweep against 0.301 at the bottom --
and `interval_rates` divides by that spacing. On a 25-checkpoint sweep the null
argmax lands on the smallest-spacing interval **44.7%** of the time when the
value series is permuted against the fixed step grid. So "the two
top intervals coincide" has a best attainable p of ~0.29 typical and 0.45 worst
case: the design cannot reject at any sensible alpha however clean the data is.
That is `POPPER_PLAN.md` 6f's attainable-floor refusal reached before building
rather than after a null result, and 6h's sharper form -- check the floor
against the null you COULD build, not only the one the registry wording reached
for. `detect_transitions` also takes `n_top` and `min_abs`, both of which are
selections if set after seeing the sweep.

WHAT REPLACES IT: A CHANGE-MASS PROFILE AND ITS CENTROID

For a series v sampled at steps s, with a REGISTERED direction (a rise or a
drop -- CLAIM-B names a Fiedler *drop*, P-I1 names a *rise*):

    w_i  proportional to  max(direction * (v_{i+1} - v_i), 0)     sum_i w_i = 1

-- the share of the series' total registered-direction change that happened in
interval i -- and the location is that distribution's centroid on the log-step
axis, x_i = (log10(s_i + 1) + log10(s_{i+1} + 1)) / 2.

Three properties, and the second is the one that decided it:

1. **No placed constant anywhere.** No `n_top`, no `min_abs`, no tolerance on
   what counts as co-located, no smoothing bandwidth. `EVALUABILITY.md` asked
   whether there was an ordinal formulation that needs none, the way CLAIM-C's
   sign-concordance avoided a magnitude cut. This is the answer: a distance in
   log10-step, compared against a null, with nothing to place.

2. **It is NOT divided by the log-step spacing, and that is a departure from
   `checkpoint_frames`.** Weighting by `interval_rates` is equally VALID -- both
   weightings measure H0 rejection at 0.043-0.073 under the pairing null this
   module actually uses -- but their POWER diverges as the sweep densifies.
   Measured at 8 units, alpha = 0.05: change mass holds 1.000 from 20 to 143
   checkpoints while rate falls 0.995, 0.970, 0.685, 0.090 over 20, 35, 80 and
   143. Dividing by dx amplifies per-checkpoint noise exactly where the spacing
   is tight, and a denser sweep makes every dx tighter. The log-step
   axis is right for plotting a derivative, which is what `checkpoint_frames`
   built it for, and wrong for weighting a location. `spacing_change_steps`
   exists to warn that an index-based derivative places a peak at a spacing
   change by construction; a change-mass profile takes no derivative at all, so
   there is nothing for it to warn about -- and `spacing_change_report` is
   computed and reported anyway, so a reader can check that rather than take it.

3. **The centroid is a location, not a claim of single-step resolution.**
   `detect_transitions` reports intervals because "a single-step answer implies
   a resolution the data does not have". The centroid does not assert one
   either: it is a summary whose sampling variability is exactly what the null
   quantifies, and `dispersion` is reported beside it so a bimodal change
   profile -- whose centroid sits between two changes and means much less -- is
   visible rather than hidden.

THE NULL IS A MATCHED CONTROL SERIES, NOT A PERMUTATION OVER CHECKPOINT ORDER

Both registry entries say "a permutation null over checkpoint order gives a
valid p once the changepoint estimator is fixed in advance". Measured, it does
not, and the failure is large. Three permutation-family nulls were built and
their H0 rejection rate measured at alpha = 0.05 against a nominal 0.05:

    permute the value series against the fixed step grid      0.45
    permute the interval increments                           0.32
    circular shift of the increments, sampled                 0.13
    circular shift of the increments, ENUMERATED (m rotations) 0.065

The first three are anticonservative for one reason: **the statistic is built
on a concentrated profile and those nulls dissolve the concentration.** A
permuted series' change is scattered across every interval, so the null's
statistic has far too little variance, and any partial overlap between two real
concentrated profiles reads as significant. (The sampled circular shift is
additionally wrong in a way worth naming: m rotations are not m independent
draws, so sampling 199 of them and dividing by 200 understates p. Enumerated,
it is honest.)

The enumerated circular shift is valid *if* changepoints are uniform on the
interval grid. They are not, and the confound has a name: **everything moves
early in training.** With both series' onsets drawn from an early-concentrated
distribution -- the realistic case -- the enumerated shift null rejects at
**0.103**, twice nominal, because it asserts that B's change could equally have
been anywhere.

The construction that survives that is a MATCHED CONTROL SERIES null: compare
the observed pair's co-location against the co-location of the same series A
with a set of OTHER series measured on the SAME sweep. Under H0 the series
under test is exchangeable with its controls, so the p-value is exact by
construction -- and it is exact *whether or not* the whole family changes early,
because the controls carry that trend too. Measured, at n = 99 controls:

    | H0 family                        | shift null | control null |
    |----------------------------------|-----------|--------------|
    | onsets log-uniform over the sweep| 0.065     | 0.050        |
    | onsets both early (common trend) | 0.103     | 0.050        |

This is `POPPER_PLAN.md` 6h's lesson arriving a second time from a different
direction. There the floor moved from 0.667 to 0.0005 on the question of what
is randomised; here validity moves from 0.103 to 0.050 on the same question,
with the same data, the same estimator and the same claim. The rows
`EVALUABILITY.md` lists as naming "a matched control that is a subspace or a
magnitude rather than a unit" gain a third kind: **a matched control that is
another series.**

WHAT THAT NULL COSTS, STATED HERE AND NOT ONLY IN THE REGISTRY

The control set is a commitment, and validity rests entirely on the controls
being exchangeable with the series under test under H0. That is a scientific
assumption, not a property of the code, so:

* the control family is REGISTERED per prediction as a module constant and the
  caller must name the one that is registered -- passing a convenient control
  set is refused, the same way `P6-R2` refuses a caller-supplied exchangeable
  unit;
* the attainable floor is 1/(n_controls + 1), so alpha = 0.05 needs at least 19
  controls and the module refuses below that -- derived from alpha, not placed;
* power is honestly low. Against two logistic curves with the SAME onset,
  25 checkpoints and 99 controls, it is ~0.39. Controls drawn from the same
  family as the series under test really can co-locate by chance, and a design
  that says otherwise is a design whose null is too narrow.

WHAT NO NULL HERE CAN DO

The sweep's resolution is its intervals. Two changes inside one interval are
one change to this construction, and no choice of statistic recovers what was
not sampled. That is the honest content of `detect_transitions`' docstring and
it survives the change of estimator.

Nothing here produces a p-value: as in POPPER_PLAN.md 6e-6h, the apparatus
exists and the artifacts do not. `claims/adjudications/` is empty.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.checkpoint_frames import spacing_change_steps, step_x

#: The two change directions a registered prediction can name. Not a free-text
#: field and with no default: CLAIM-B names a Fiedler *drop* and P-I1 a *rise*,
#: and a default would silently score a rise as a drop's absence.
CHANGE_DIRECTIONS: Dict[str, int] = {"rise": +1, "drop": -1}

#: One-sided, fixed in advance. Both entries predict co-location, so the
#: predicted outcome is the SMALL distance and therefore the LARGE statistic.
ALTERNATIVE = "greater"

#: The reciprocal one-sided test. It decides the RE-ANCHORS branch -- the
#: change sits demonstrably further from where it was predicted than the
#: controls do -- and it is a stop-rule input only. It never enters a claim's
#: e-process: two one-sided tests on one statistic would double the claim's
#: Type-I rate for free. Same division CLAIM-C makes.
RECIPROCAL_ALTERNATIVE = "less"

#: A sweep needs at least three checkpoints for a centroid to be able to move
#: at all: two give one interval and every profile is the same point.
MIN_CHECKPOINTS = 3


class ColocationRefused(Exception):
    """
    Raised when a p-value would have to come from inputs that cannot support
    one. Standing rule 4: a number from mismatched inputs is worse than no
    number, because it is unfalsifiable from the output alone.
    """


# ---------------------------------------------------------------------------
# The change-mass profile
# ---------------------------------------------------------------------------

def _checked_steps(steps: Sequence[float]) -> np.ndarray:
    s = np.asarray(steps, dtype=np.float64)
    if s.ndim != 1:
        raise ColocationRefused(f"steps must be one-dimensional; got shape {s.shape}")
    if s.size < MIN_CHECKPOINTS:
        raise ColocationRefused(
            f"a co-location test needs at least {MIN_CHECKPOINTS} checkpoints; "
            f"got {s.size}. Two checkpoints give one interval, every profile is "
            f"the same point, and the distance between two locations is zero by "
            f"construction rather than by measurement.")
    if not np.all(np.isfinite(s)) or np.any(s < 0):
        raise ColocationRefused("steps must be finite and non-negative")
    if np.any(np.diff(s) <= 0):
        raise ColocationRefused(
            "steps must be strictly increasing. Refusing rather than sorting: a "
            "caller who hands these in unsorted has a series whose value order "
            "may or may not have been sorted with them, and guessing which is "
            "the defect this refusal exists to surface.")
    return s


def interval_midpoints(steps: Sequence[float]) -> np.ndarray:
    """Interval midpoints on the log-step axis, x_i = (x(s_i) + x(s_{i+1})) / 2."""
    x = step_x(_checked_steps(steps))
    return 0.5 * (x[:-1] + x[1:])


def change_profile(steps: Sequence[float], values: Sequence[float],
                   direction: str) -> dict:
    """
    Where the series' registered-direction change happened, as a distribution
    over the sweep's intervals, plus that distribution's location.

    `direction` is required and must be one of `CHANGE_DIRECTIONS`. Refuses on
    a series with no change in the registered direction: a series that only
    falls has no "rise" to locate, and returning a uniform profile there would
    report "the rise is spread evenly over training" for "there is no rise".
    """
    if direction not in CHANGE_DIRECTIONS:
        raise ColocationRefused(
            f"unknown change direction {direction!r}; expected one of "
            f"{sorted(CHANGE_DIRECTIONS)}. Refusing rather than defaulting: "
            f"CLAIM-B names a Fiedler DROP and P-I1 names a RISE, and a default "
            f"would score one as the other's absence.")
    s = _checked_steps(steps)
    v = np.asarray(values, dtype=np.float64)
    if v.shape != s.shape:
        raise ColocationRefused(
            f"change_profile: {v.shape} values against {s.shape} steps; these "
            f"index the same checkpoints and must match")
    if not np.all(np.isfinite(v)):
        raise ColocationRefused(
            "the series has non-finite values; refusing rather than dropping "
            "them, since a dropped checkpoint silently merges two intervals "
            "into one and moves the location it is trying to measure")

    sign = CHANGE_DIRECTIONS[direction]
    mass = np.clip(sign * np.diff(v), 0.0, None)
    total = float(mass.sum())
    if total <= 0.0:
        raise ColocationRefused(
            f"the series has no {direction} anywhere in the sweep (total "
            f"registered-direction change is {total}); there is no location to "
            f"measure. A uniform profile would report the change as spread "
            f"evenly over training rather than as absent.")

    w = mass / total
    x = interval_midpoints(s)
    centroid = float(np.sum(w * x))
    dispersion = float(np.sqrt(max(np.sum(w * (x - centroid) ** 2), 0.0)))
    return {
        "weights": w,
        "x": x,
        "steps": s,
        "direction": direction,
        "centroid_log_step": centroid,
        "centroid_step": float(10.0 ** centroid - 1.0),
        "dispersion_log_step": dispersion,
        "total_change": total,
        "n_intervals": int(w.size),
        "concentration": float(np.max(w)),
    }


def spacing_change_report(steps: Sequence[float]) -> dict:
    """
    The spacing changes `checkpoint_frames.spacing_change_steps` warns about,
    carried into every record.

    A change-mass profile takes no derivative, so an index-based derivative's
    "peak here by construction" artifact cannot reach it. That is a claim, and
    this reports the input a reader needs to check it rather than asking them
    to take it.
    """
    s = _checked_steps(steps)
    x = step_x(s)
    dx = np.diff(x)
    return {
        "spacing_change_steps": spacing_change_steps(s),
        "log_step_spacing_min": float(dx.min()),
        "log_step_spacing_max": float(dx.max()),
        "log_step_spacing_ratio": float(dx.max() / dx.min()) if dx.min() > 0 else float("inf"),
        "_note": (
            "reported, not acted on: the profile weights are change MASS and "
            "are never divided by this spacing. A rate-weighted profile is "
            "equally valid under H0 and loses power as the spacing ratio grows; "
            "see the module docstring."),
    }


# ---------------------------------------------------------------------------
# The two statistics. Greater is the predicted direction in both.
# ---------------------------------------------------------------------------

def colocation_statistic(profile_a: dict, profile_b: dict) -> float:
    """
    Minus the distance between two change locations, in log10-step.

    Negated so that "greater" is the predicted direction, matching every other
    one-sided test in this project and `core.nulls.p_from_null`'s convention.
    """
    return -abs(float(profile_a["centroid_log_step"])
                - float(profile_b["centroid_log_step"]))


def anchor_statistic(profile: dict, window: Tuple[float, float]) -> float:
    """
    Minus the distance from a change location to a pre-registered step window,
    in log10-step. Zero inside the window.

    The window is a pair of STEPS, converted here, and it is not a tolerance:
    CLAIM-B's registered statement names "steps ~512-2000", so the numbers come
    from the prediction rather than from this module. Standing rule 6 asks where
    a constant came from; the answer has to be a citation, and the gate that
    uses it carries one.
    """
    lo, hi = float(window[0]), float(window[1])
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo < 0 or hi < lo:
        raise ColocationRefused(
            f"anchor window {window!r} must be a finite, non-negative, "
            f"non-decreasing pair of steps")
    c = float(profile["centroid_log_step"])
    xlo, xhi = float(step_x([lo])[0]), float(step_x([hi])[0])
    if xlo <= c <= xhi:
        return 0.0
    return -(xlo - c if c < xlo else c - xhi)


# ---------------------------------------------------------------------------
# The matched-control null
# ---------------------------------------------------------------------------

def attainable_floor_report(n_controls: int, alpha: float) -> dict:
    """
    The smallest p this design can express, before any data is seen.

    `EVALUABILITY.md`: check the attainable floor BEFORE building the null, not
    after a result comes back null. The binary alternative is reported beside it
    because the comparison is the finding -- a co-location statistic that can
    only say "same interval or not" is floored by the null's chance of picking
    that interval, and the log-step geometry makes that chance large.
    """
    n = int(n_controls)
    a = float(alpha)
    floor = 1.0 / (n + 1.0)
    return {
        "n_controls": n,
        "alpha": a,
        "control_null_floor": floor,
        "min_controls_for_alpha": int(np.ceil(1.0 / a - 1.0)),
        "sufficient": bool(floor <= a),
        "_note": (
            "control_null_floor is 1/(n_controls + 1) and the caller controls "
            "it. A BINARY co-location statistic -- 'the two top intervals "
            "coincide' -- is floored instead by the null's probability of "
            "selecting that interval, measured at 0.447 for the "
            "smallest-spacing interval of a 25-checkpoint Pythia sweep with the "
            "value series permuted against the fixed step grid -- which no "
            "number of controls reduces. The floor is a property of the "
            "statistic as much as of the null."),
    }


def _control_stats(observed: float, control_stats: Sequence[float],
                   alpha: float) -> dict:
    c = np.asarray(list(control_stats), dtype=np.float64)
    if c.size == 0:
        raise ColocationRefused("the control set is empty; there is no null")
    if not np.all(np.isfinite(c)):
        raise ColocationRefused(
            f"{int(np.sum(~np.isfinite(c)))} of {c.size} control statistics are "
            f"not finite; a null thinned by failures is not the null that was "
            f"designed")
    if not np.isfinite(observed):
        raise ColocationRefused(
            f"the observed statistic is not finite ({observed!r})")

    floor = attainable_floor_report(c.size, alpha)
    if not floor["sufficient"]:
        raise ColocationRefused(
            f"attainable floor {floor['control_null_floor']:.4f} exceeds "
            f"alpha={alpha}: this design cannot reject on a perfect result, and "
            f"reporting 'not significant' on nothing is worse than reporting "
            f"nothing. At least {floor['min_controls_for_alpha']} controls are "
            f"needed.")

    if float(c.max()) == float(c.min()):
        raise ColocationRefused(
            f"every control gives the identical statistic ({float(c.min())!r}). "
            f"The controls then contribute ONE observation and ranking the "
            f"observed value against n copies of it is the wrong null, not a "
            f"conservative one. This is a degeneracy and not a tolerance -- the "
            f"control values are either all equal or they are not -- so no "
            f"threshold is placed here.")

    return {
        "observed": float(observed),
        "p_value": float((np.sum(c >= observed) + 1) / (c.size + 1)),
        "p_reciprocal": float((np.sum(c <= observed) + 1) / (c.size + 1)),
        "n_controls": int(c.size),
        "control_mean": float(c.mean()),
        "control_min": float(c.min()),
        "control_max": float(c.max()),
        "attainable_floor": floor,
        "alternative": ALTERNATIVE,
        "reciprocal_alternative": RECIPROCAL_ALTERNATIVE,
    }


# ---------------------------------------------------------------------------
# Arm 1 — the mutual arm, over paired units
# ---------------------------------------------------------------------------
#
# The matched control for series B at unit u is series B AT ANOTHER UNIT: same
# metric, same construction, same sweep. Combining those controls across units
# is a permutation of the PAIRING between the two series' units, which is what
# makes the arm one test over all units rather than one test per unit -- the
# "49 ALBERT layers are not 49 independent observations" error, in this design's
# clothing. Under H0 the two series' locations are independent, so which unit of
# A is paired with which unit of B is arbitrary, and the permutation is exact.
#
# It also disposes of the common-trend confound for free: both series keep their
# real per-unit locations under every permutation, so "everything moves early in
# training" is held fixed on both sides instead of being asserted away.

#: Sampled permutations when the pairing group is too large to enumerate. A
#: module constant and not a parameter, the convention P-S1, P-T1, P-M1,
#: CLAIM-C and P6-R2/R4 all follow: a per-run null size is a per-run choice.
N_PAIRING_PERMUTATIONS = 2000

#: Enumerate the pairing group exhaustively at or below this many permutations.
#: 5040 is 7 units. Above it the group is sampled -- with the +1 rule, which is
#: what keeps a sampled permutation p valid rather than anticonservative.
EXHAUSTIVE_PAIRING_LIMIT = 5040

_SEED = 20260824


def _factorial_capped(n: int, cap: int) -> int:
    """n! , stopped as soon as it exceeds `cap`. Avoids building 24!."""
    out = 1
    for k in range(2, n + 1):
        out *= k
        if out > cap:
            return cap + 1
    return out


def _pairing_permutations(n_units: int, seed: int) -> Tuple[List[np.ndarray], bool]:
    """
    The null's pairings, enumerated when the group is small and sampled when it
    is not. The identity is included either way: it reproduces the observed
    statistic, which is what makes the smallest attainable p 1/P rather than 0.
    """
    n_distinct = _factorial_capped(n_units, EXHAUSTIVE_PAIRING_LIMIT)
    if n_distinct <= EXHAUSTIVE_PAIRING_LIMIT:
        from itertools import permutations
        return [np.asarray(p, dtype=int) for p in permutations(range(n_units))], True
    rng = np.random.default_rng(seed)
    return ([np.arange(n_units)]
            + [rng.permutation(n_units) for _ in range(N_PAIRING_PERMUTATIONS)],
            False)


def _rank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(v))
    return order.astype(np.float64)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(np.dot(ra, rb) / denom) if denom > 0 else float("nan")


def _shared_unit_factor_diagnostic(ca: np.ndarray, cb: np.ndarray) -> dict:
    """
    The construction's measured limitation, made visible in every record.

    The pairing null tests ASSOCIATION between the two series' per-unit change
    locations. A common per-unit factor -- a head that forms late forming late
    in BOTH series, for a reason that has nothing to do with the claim -- is an
    association, and the calibration measures the rejection rate under exactly
    that at **1.00**, against 0.05 when the two are independent. No null over
    the pairing can separate the two, because a confound present at every unit
    is present under every permutation.

    So this is a diagnostic and never a correction: it reports each series'
    rank correlation with the unit index, which catches a confound that is
    MONOTONE in that index (depth, head order) and catches nothing else. A
    reader placing a run between the two measured rates has this and the
    analyst's stated independence source, and nothing further. Same shape as
    CLAIM-C reporting `sign_homogeneity` between its two measured endpoints --
    except that there the middle was later measured and here it cannot be from
    these two series alone.
    """
    idx = np.arange(ca.size, dtype=np.float64)
    return {
        "centroid_rank_corr_a_vs_unit_index": _spearman(ca, idx),
        "centroid_rank_corr_b_vs_unit_index": _spearman(cb, idx),
        "centroid_rank_corr_a_vs_b": _spearman(ca, cb),
        "_note": (
            "REPORTED, NEVER SCORED. Measured H0 rejection is 0.05 when the two "
            "series' per-unit locations are independent and 1.00 under a common "
            "per-unit factor unrelated to the claim; the gate cannot tell them "
            "apart and the analyst must name the independence source, as "
            "PREDICTIONS.md's Phase 7 adjudication constraint 2 already "
            "requires. Both correlations near zero rules out only a confound "
            "that is MONOTONE in the unit index."),
    }


def paired_colocation_arm(steps: Sequence[float],
                          series_a: Sequence[Sequence[float]],
                          direction_a: str,
                          series_b: Sequence[Sequence[float]],
                          direction_b: str,
                          *,
                          alpha: float,
                          unit_name: str,
                          arm_name: str,
                          seed: int = _SEED) -> dict:
    """
    Do A's change locations and B's change locations co-locate, unit by unit,
    more than an arbitrary pairing of the same two populations allows?

    `series_a[u]` and `series_b[u]` are the two series at unit u (layer for
    CLAIM-B, head for P-I1), both sampled at `steps`. The statistic is the mean
    over units of minus the log10-step distance between the two change
    centroids; the null repairs A's units with B's under a permutation.
    """
    s = _checked_steps(steps)
    a = [change_profile(s, v, direction_a) for v in series_a]
    b = [change_profile(s, v, direction_b) for v in series_b]
    if len(a) != len(b):
        raise ColocationRefused(
            f"{len(a)} units on the A side against {len(b)} on the B side; the "
            f"pairing null needs the same units on both")
    n_units = len(a)
    if n_units < 2:
        raise ColocationRefused(
            f"the pairing null needs at least two units; got {n_units}. With one "
            f"unit there is one pairing, the null is the observation, and the "
            f"only expressible p is 1.0.")

    ca = np.array([p["centroid_log_step"] for p in a], dtype=np.float64)
    cb = np.array([p["centroid_log_step"] for p in b], dtype=np.float64)
    perms, exhaustive = _pairing_permutations(n_units, seed)
    n_draws = len(perms)

    # 1/n_draws in BOTH regimes, and not 1/(n_draws + 1) in the sampled one:
    # `perms` already carries the identity pairing as its first entry, so
    # n_draws is N + 1 and the sampled p is (count over the N samples + 1) /
    # n_draws, whose minimum is 1/n_draws. Writing the usual +1 here made the
    # reported floor SMALLER than any p the arm can express -- the same class of
    # slip as P6-R2's default argument bound at definition time, found the same
    # way, by a test that asserted a perfect result lands exactly on the floor.
    floor = 1.0 / n_draws
    if floor > alpha:
        raise ColocationRefused(
            f"arm {arm_name!r}: attainable floor {floor:.4f} exceeds "
            f"alpha={alpha}. With {n_units} units there are only "
            f"{n_draws} distinct pairings, so this arm cannot reject on a "
            f"perfect result and 'not significant' would be a statement about "
            f"the design and not about the data.")

    stats = np.array([-np.mean(np.abs(ca - cb[p])) for p in perms],
                     dtype=np.float64)
    observed = float(stats[0])          # the identity pairing IS the observation
    if not np.isfinite(observed) or not np.all(np.isfinite(stats)):
        raise ColocationRefused(f"arm {arm_name!r}: a non-finite statistic")
    if float(stats.max()) == float(stats.min()):
        raise ColocationRefused(
            f"arm {arm_name!r}: every pairing gives the identical statistic. "
            f"The units then contribute one observation and permuting them is "
            f"the wrong null, not a conservative one.")

    p_greater = float(np.sum(stats >= observed) / n_draws) if exhaustive else \
        float((np.sum(stats[1:] >= observed) + 1) / (n_draws))
    p_less = float(np.sum(stats <= observed) / n_draws) if exhaustive else \
        float((np.sum(stats[1:] <= observed) + 1) / (n_draws))

    return {
        "arm": arm_name,
        "kind": "paired-unit-permutation",
        "shared_unit_factor_diagnostic": _shared_unit_factor_diagnostic(ca, cb),
        "unit": unit_name,
        "n_units": n_units,
        "observed": observed,
        "mean_distance_log_step": -observed,
        "p_value": p_greater,
        "p_reciprocal": p_less,
        "n_pairings": int(n_draws),
        "null_exhaustive": bool(exhaustive),
        "attainable_floor": floor,
        "alternative": ALTERNATIVE,
        "direction_a": direction_a,
        "direction_b": direction_b,
        "centroids_a_step": [float(10.0 ** c - 1.0) for c in ca],
        "centroids_b_step": [float(10.0 ** c - 1.0) for c in cb],
        "dispersion_a_log_step": [p["dispersion_log_step"] for p in a],
        "dispersion_b_log_step": [p["dispersion_log_step"] for p in b],
    }


# ---------------------------------------------------------------------------
# Arm 2 — the anchor arm, against a pre-registered step window
# ---------------------------------------------------------------------------

def anchor_arm(steps: Sequence[float],
               series: Sequence[Sequence[float]],
               direction: str,
               window: Tuple[float, float],
               controls: Dict[str, Sequence[Sequence[float]]],
               control_directions: Dict[str, str],
               *,
               alpha: float,
               unit_name: str,
               arm_name: str) -> dict:
    """
    Does this series' change sit closer to the pre-registered step window than
    other series measured on the same sweep do?

    There is no relabeling that realises "unrelated to the literature's
    anchors", so this arm's null CANNOT be a permutation: it needs a reference
    population of change locations, and the only honest one is other series.
    `controls` maps a control series' name to its per-unit series; each control
    is scored by the same statistic and the p is its rank.

    This is the arm most likely to REFUSE, and the arithmetic says so before any
    data exists: the floor is 1/(n_controls + 1), so alpha = 0.05 needs 19
    control series measured on the same sweep at the same units. A cheap-tier
    sweep that measures six metrics cannot adjudicate this arm however clean the
    result is. That is a statement about what the pilot must measure, made
    before it runs, which is the whole point of computing a floor first.
    """
    s = _checked_steps(steps)
    if not controls:
        raise ColocationRefused(
            f"arm {arm_name!r}: no control series. This arm's null IS the "
            f"control population; there is no permutation that substitutes for "
            f"it.")
    missing = sorted(set(controls) - set(control_directions))
    if missing:
        raise ColocationRefused(
            f"arm {arm_name!r}: no registered change direction for control "
            f"series {missing}. Every control needs one for the same reason the "
            f"series under test does.")

    prof = [change_profile(s, v, direction) for v in series]
    n_units = len(prof)
    if n_units < 1:
        raise ColocationRefused(f"arm {arm_name!r}: no units supplied")
    observed = float(np.mean([anchor_statistic(p, window) for p in prof]))

    control_stats: List[float] = []
    names: List[str] = []
    for name in sorted(controls):
        cs = list(controls[name])
        if len(cs) != n_units:
            raise ColocationRefused(
                f"arm {arm_name!r}: control {name!r} has {len(cs)} units "
                f"against {n_units} in the series under test; a control that "
                f"does not cover the same units is not matched")
        cp = [change_profile(s, v, control_directions[name]) for v in cs]
        control_stats.append(float(np.mean([anchor_statistic(p, window) for p in cp])))
        names.append(name)

    res = _control_stats(observed, control_stats, alpha)
    res.update({
        "arm": arm_name,
        "kind": "matched-control-series",
        "unit": unit_name,
        "n_units": n_units,
        "direction": direction,
        "window_steps": [float(window[0]), float(window[1])],
        "control_names": names,
        "distance_to_window_log_step": -observed,
        "centroids_step": [float(10.0 ** p["centroid_log_step"] - 1.0) for p in prof],
    })
    return res


# ---------------------------------------------------------------------------
# Combining arms, and the three-way verdict
# ---------------------------------------------------------------------------

def combine_arms(arms: Sequence[dict]) -> dict:
    """
    Intersection-union over the arms: the reported p is the MAX.

    A max of p-values is a valid p for a conjunction REGARDLESS of how the arms
    depend on each other, which is what makes it right here -- CLAIM-B's three
    arms share two series between them and any multiplicity correction over
    them would be absurd. Same reasoning, and the same precedent, as CLAIM-C's
    metric-leave-one-out axis.

    Both directions are combined the same way, so "co-locates" needs every arm
    to clear and "re-anchors" needs every arm to show the separation. Anything
    mixed is INSUFFICIENT.
    """
    if not arms:
        raise ColocationRefused("no arms to combine")
    for a in arms:
        if a.get("p_value") is None or a.get("p_reciprocal") is None:
            raise ColocationRefused(
                f"arm {a.get('arm')!r} carries no p-value; a max over a set with "
                f"an undefined member is undefined, and reporting the rest would "
                f"silently drop whichever arm was hardest to satisfy")
    return {
        "p_value": max(float(a["p_value"]) for a in arms),
        "p_reciprocal": max(float(a["p_reciprocal"]) for a in arms),
        "n_arms": len(arms),
        "binding_arm": max(arms, key=lambda a: float(a["p_value"]))["arm"],
        "rule": "intersection-union max (unanimity in both directions)",
        "arms": list(arms),
    }


def _alpha() -> float:
    from core.adjudication import load_registry
    from core.evalues import DEFAULT_ALPHA
    try:
        return float(load_registry().get("alpha", DEFAULT_ALPHA))
    except Exception:
        return float(DEFAULT_ALPHA)


def gate_verdict(p_greater: Optional[float], p_less: Optional[float],
                 alpha: Optional[float] = None) -> dict:
    """
    The three-way stop rule, CLAIM-C's shape.

    CLAIM-B's falsifier is unusual and the verdict lattice is built around it:
    *"No co-location. Itself a real result: it re-anchors the 1.4B schedule
    rather than invalidating the sweep."* So a demonstrated separation is
    recorded as a falsification -- positively shown, not inferred from a failure
    to reject -- while a test that shows neither returns INSUFFICIENT, because
    an e-process records insufficient evidence and never a null accepted.
    """
    a = _alpha() if alpha is None else float(alpha)
    if p_greater is None:
        return {"verdict": "INSUFFICIENT", "falsified": False, "alpha": a,
                "reading": "no p-value could be emitted; nothing is adjudicated"}
    if p_greater <= a:
        return {"verdict": "CO-LOCATES", "falsified": False, "alpha": a,
                "reading": "the two changes sit closer together on the log-step "
                           "axis than the matched controls allow"}
    if p_less is not None and p_less <= a:
        return {"verdict": "RE-ANCHORS", "falsified": True, "alpha": a,
                "reading": "the changes sit demonstrably FURTHER apart than the "
                           "matched controls. The falsifier, positively shown: a "
                           "real result that re-anchors the schedule rather than "
                           "invalidating the sweep"}
    return {"verdict": "INSUFFICIENT", "falsified": False, "alpha": a,
            "reading": "co-location was not shown and separation was not shown. "
                       "Nothing is falsified -- an e-process records insufficient "
                       "evidence, never a null accepted"}


# ---------------------------------------------------------------------------
# CLAIM-B's gate
# ---------------------------------------------------------------------------
#
# CLAIMS.md names `core/checkpoint_frames.py` as H-EMERGE's instrument, which is
# why CLAIM-B's gate is here and P-I1's is in `p7_motifs/formation_gate.py`.

#: The literature anchors, taken from CLAIM-B's REGISTERED STATEMENT -- "the
#: energy-monotonicity break and the Fiedler drop co-locate with steps
#: ~512-2000". PRE-REGISTERED, not placed: standing rule 6 asks where a constant
#: came from and the answer here is a citation to the prediction itself, made
#: before any sweep existed. Nothing in this module may choose it.
CLAIM_B_ANCHOR_WINDOW: Tuple[float, float] = (512.0, 2000.0)

#: Which series, and which direction each one's change is registered to have.
#: `sum_severity` rather than `n_violations`: the count is an integer with heavy
#: ties, and a tied series puts its change mass on whichever interval happens to
#: cross an integer boundary. Severity is the magnitude, and "break" is a
#: statement about magnitude. Same class of decision as CLAIM-C reading
#: `effective_rank_normed` rather than the raw field.
CLAIM_B_SERIES: Dict[str, Dict[str, str]] = {
    "energy_break": {
        "field": "core.metrics.energy_violation_severity()['sum_severity']",
        "direction": "rise",
        "why": "an energy-monotonicity BREAK is a rise in violation severity",
    },
    "fiedler_drop": {
        "field": "CHECKPOINT_METRICS['fiedler_mean']",
        "direction": "drop",
        "why": "CLAIM-B names a Fiedler DROP",
    },
}

#: The exchangeable unit of the mutual arm's null. The layer, because the layer
#: is what both series are measured at and what the pairing permutes. It is NOT
#: an assertion that layers are independent observations -- the arm is one test
#: over all layers, not one test per layer, and that distinction is the whole
#: reason the null permutes the PAIRING rather than resampling layers.
CLAIM_B_UNIT = "layer"

#: The anchor arm's control family, registered here so a caller cannot
#: substitute a convenient one. Validity rests entirely on these being
#: exchangeable with the series under test under H0, which is a scientific
#: assumption and not a property of the code.
CLAIM_B_ANCHOR_CONTROL_FAMILY = (
    "other checkpoint-level metric series measured on the same sweep at the "
    "same layers")


def claim_b_arms(steps: Sequence[float],
                 energy_break: Sequence[Sequence[float]],
                 fiedler: Sequence[Sequence[float]],
                 anchor_controls: Dict[str, Sequence[Sequence[float]]],
                 anchor_control_directions: Dict[str, str],
                 *,
                 control_family: str,
                 alpha: Optional[float] = None,
                 seed: int = _SEED) -> List[dict]:
    """CLAIM-B's three arms. See `claim_b_gate`."""
    if control_family != CLAIM_B_ANCHOR_CONTROL_FAMILY:
        raise ColocationRefused(
            f"control_family {control_family!r} is not the registered one "
            f"({CLAIM_B_ANCHOR_CONTROL_FAMILY!r}). Refusing rather than "
            f"accepting it: this null's validity is exactly the claim that the "
            f"controls are exchangeable with the series under test under H0, so "
            f"which population they come from is a pre-registered scientific "
            f"decision and not a per-run argument.")
    a = _alpha() if alpha is None else float(alpha)
    d_e = CLAIM_B_SERIES["energy_break"]["direction"]
    d_f = CLAIM_B_SERIES["fiedler_drop"]["direction"]
    return [
        paired_colocation_arm(
            steps, energy_break, d_e, fiedler, d_f,
            alpha=a, unit_name=CLAIM_B_UNIT, arm_name="mutual", seed=seed),
        anchor_arm(steps, energy_break, d_e, CLAIM_B_ANCHOR_WINDOW,
                   anchor_controls, anchor_control_directions,
                   alpha=a, unit_name=CLAIM_B_UNIT, arm_name="anchor:energy_break"),
        anchor_arm(steps, fiedler, d_f, CLAIM_B_ANCHOR_WINDOW,
                   anchor_controls, anchor_control_directions,
                   alpha=a, unit_name=CLAIM_B_UNIT, arm_name="anchor:fiedler_drop"),
    ]


def p_value_claim_b(steps: Sequence[float],
                    energy_break: Sequence[Sequence[float]],
                    fiedler: Sequence[Sequence[float]],
                    anchor_controls: Dict[str, Sequence[Sequence[float]]],
                    anchor_control_directions: Dict[str, str],
                    *,
                    control_family: str,
                    alpha: Optional[float] = None,
                    seed: int = _SEED) -> dict:
    """
    CLAIM-B's p-value: unanimity over the mutual arm and the two anchor arms.

    Returns `p_value` None with a `reason` rather than a number the design
    cannot support. The likeliest refusal is the anchor arms' attainable floor
    -- 19 control series are needed at alpha = 0.05 and a cheap-tier sweep
    measuring six metrics has six. That is a requirement on the pilot, computed
    before it runs; see `anchor_arm`.
    """
    a = _alpha() if alpha is None else float(alpha)
    out: dict = {
        "prediction_id": "CLAIM-B",
        "claim": "H-EMERGE",
        "alpha": a,
        "anchor_window_steps": list(CLAIM_B_ANCHOR_WINDOW),
        "series": CLAIM_B_SERIES,
        "unit": CLAIM_B_UNIT,
        "control_family": CLAIM_B_ANCHOR_CONTROL_FAMILY,
        "spacing": None,
        "p_value": None,
        "p_reciprocal": None,
        "reason": None,
    }
    try:
        out["spacing"] = spacing_change_report(steps)
        arms = claim_b_arms(steps, energy_break, fiedler, anchor_controls,
                            anchor_control_directions,
                            control_family=control_family, alpha=a, seed=seed)
        comb = combine_arms(arms)
    except ColocationRefused as exc:
        out["reason"] = str(exc)
        out.update(gate_verdict(None, None, a))
        return out

    out.update({k: comb[k] for k in
                ("p_value", "p_reciprocal", "n_arms", "binding_arm", "rule", "arms")})
    out.update(gate_verdict(comb["p_value"], comb["p_reciprocal"], a))
    return out


def adjudicate_claim_b(steps: Sequence[float],
                       energy_break: Sequence[Sequence[float]],
                       fiedler: Sequence[Sequence[float]],
                       anchor_controls: Dict[str, Sequence[Sequence[float]]],
                       anchor_control_directions: Dict[str, str],
                       *,
                       control_family: str,
                       alpha: Optional[float] = None,
                       seed: int = _SEED,
                       artifact_hashes: Sequence[str] = (),
                       run_manifest: Optional[dict] = None,
                       adjudicate: bool = False,
                       adjudications_dir=None) -> dict:
    """
    `p_value_claim_b` plus, optionally, an entry in the falsification ledger.

    Opt-in behind a flag for the reason it is everywhere else here: these
    functions are exercised by tests and `core.adjudication` refuses to
    overwrite an existing record, so one accidental fixture run would
    permanently occupy CLAIM-B's slot with a synthetic p-value.

    Only `p_value` is adjudicated. `p_reciprocal` decides the RE-ANCHORS branch
    and lands in the record's notes; calibrating both into H-EMERGE's product
    would be two one-sided tests on one statistic and would double the claim's
    Type-I rate for free.
    """
    res = p_value_claim_b(steps, energy_break, fiedler, anchor_controls,
                          anchor_control_directions,
                          control_family=control_family, alpha=alpha, seed=seed)
    res["adjudication"] = None
    if not (adjudicate and res.get("p_value") is not None):
        return res

    from core.adjudication import adjudicate_if_registered
    res["adjudication"] = adjudicate_if_registered(
        "CLAIM-B", res["p_value"],
        artifact_hashes=tuple(artifact_hashes), run_manifest=run_manifest,
        test_name=(
            f"changepoint co-location on the log-step axis; location = centroid "
            f"of the change-mass profile; one-sided '{ALTERNATIVE}'; reported as "
            f"the intersection-union MAX over {res['n_arms']} arms (mutual "
            f"pairing-permutation over {CLAIM_B_UNIT}s, plus one "
            f"matched-control-series arm per series against the pre-registered "
            f"window {CLAIM_B_ANCHOR_WINDOW[0]:.0f}-{CLAIM_B_ANCHOR_WINDOW[1]:.0f})"),
        notes=(
            f"verdict={res['verdict']} binding_arm={res['binding_arm']} "
            f"p_reciprocal={res['p_reciprocal']:.4f} (RE-ANCHORS input only, NOT "
            f"calibrated into E) "
            f"shares its estimator with P-I1 under H-BRIDGE: an estimator defect "
            f"moves both, so their e-values are not two independent factors"),
        adjudications_dir=adjudications_dir,
    )
    return res
