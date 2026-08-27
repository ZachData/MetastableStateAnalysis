"""
p7_motifs/patching_gate.py — P-AB1's gate (the patching entry's
recapture-vs-propagation question, as a growth exponent in remaining depth).

    P-AB1  Trajectory divergence following an ablation grows SUPERLINEARLY with
           the number of layers after the ablation point, because ablation
           removes a term from a FIELD rather than a value from a SUM.
    H0     Divergence is constant or linear in remaining depth, as an
           additive-contribution account predicts once the ablated contribution
           is removed.

`design-7.md` names this the one entry in its translation table "where the
particle account plausibly says something the mechinterp framing does not
already say", and `claims/EVALUABILITY.md` records it as the last unbuilt
Phase 7 bridge entry with a live instrument.

BUILT IN THE ORDER `EVALUABILITY.md` PRESCRIBES, WHICH IS NOT THE ORDER IT WAS
TEMPTING TO BUILD IN

After nine dry runs that each found something, that document closed with an
order to work in for every remaining row that names a matched control:
*"compute the attainable floor, name what the statistic degenerates on, check
what the measurement grid contributes to it, and only then build the control."*
This module was built in that order and each of the three steps changed the
design before any control existed. What follows is what each one returned.

1. THE ATTAINABLE FLOOR, AND THE REGISTERED NULL THAT CANNOT REACH ANY
--------------------------------------------------------------------------
The registry's `null_construction` reads: *"Permutation over ablation points
once the fitted exponent is the statistic."* Read literally -- permute which
ablation point's real exponent is compared against which ablation point's
control exponent -- it is **degenerate**, and the algebra says so with no data
at all. The natural statistic on a matched design is the mean paired
difference

    T = mean_l ( beta_real(l) - beta_control(l) )

and permuting the pairing gives mean_l beta_real(l) - mean_l beta_control(sigma(l))
= mean beta_real - mean beta_control, which is **the same number for every
permutation**. The null has zero spread, every draw ties the observation, and
the smallest p the design can express is **1.000**. Not a small floor: the
largest one there is. That is `POPPER_PLAN.md` 6p's seventeenth lesson -- a
floor is a claim about the DESIGN and not about the call -- arriving in the one
form where the design can never reject whatever it is handed.

The reading that works is the other one: the exchangeable object under H0 is
the **label** on the two directions at one ablation point. If the ablation
removes a value from a sum, the real direction and a structureless direction of
equal magnitude at the same layer are exchangeable, so swapping their labels is
an exact null. That is a SIGN-FLIP null and its floor is `2/(2^n + 1)` in the
number of informative units, so:

    n = 1  2  3  4  5     6      7
    f = .67 .40 .22 .12 .061  .031  .016   <- six is the first that clears 0.05

**Six units is the first design that can reject at all**, which is the same
shape as CLAIM-C's six prompts (6f), P-ST1's two pairs (6m) and CLAIM-C's five
informative rows (6l), and it is arithmetic on the design rather than on any
draw count.

**And there is a second floor, which binds at the other end.** Under the
per-ablation-point unit the group is 2^42 and is sampled, so the run also cannot
express anything below `2/(n_patterns + 1)`: the design floor there is 4.5e-13
and a PERFECT input returns 4.0e-4, nine orders of magnitude apart. The
attainable floor is the MAX of the two -- 6p's rule for `P-T1`, in a second
setting -- and reporting either alone is 6i's defect, where CLAIM-B's sampled
pairing regime reported a floor smaller than any p the arm could express.
`attainable_floor_report` returns both and says which one `binds`. It was found
by printing what a perfect input returns beside what the arm claimed it could
return, which is the tenth session running in which looking at an output found
something no test was failing on.

2. WHAT THE STATISTIC DEGENERATES ON: THE FIT WINDOW, WHICH THE ABLATION POINT
   SETS
-------------------------------------------------------------------------------
6o's rule is that "matched on what" has to name the quantity the statistic
degenerates on. A fitted growth exponent degenerates on the **window it was
fitted over**, and the ablation point fixes that window: ablating at layer `l`
of an `L`-layer model leaves `K = L - l` downstream layers and no more.

Divergence saturates -- two trajectories can only diverge so far -- so the
log-log slope of a saturating curve is a DECREASING function of K. Measured on
`D(k) = 1 - exp(-(k/tau)^beta)` with beta = 2.0 and tau = 4, the same dynamics
read

    K        =    3     4     6     8    12    16    24
    beta-hat =  1.79  1.71  1.53  1.35  1.07  0.88  0.66

(`claims/calibration/patching_exponent.json`, `window_dependence`, which is
deterministic and takes no replicates) so the exponent falls by a factor of
nearly three on nothing but where the measurement stopped. **"Superlinear" is not a window-free statement**, and the
registry's own reason for requiring a control ("later layers have more
opportunity to diverge for reasons unrelated to field structure") is right but
is not the binding one. The binding one is that beta has no meaning at all
until a window is named.

The pairing is what disposes of it: the real and control arms at the same
ablation point are fitted over the **same** window, so the window's contribution
is identical on both sides of the difference. That puts this construction in
6o's safe column -- and only because it pairs at the ablation point. The
comparison across ablation points, which the registered null's literal reading
performs, is between exponents fitted over different windows, and it is not a
comparison at all.

| statistic | the quantity it degenerates on | matched on it? |
|---|---|---|
| `P6-R4`: one subspace against matched controls | subspace dimension | yes, by construction |
| `CLAIM-B` anchor: one location against a fixed window | where the grid puts an unlocated profile | **no** (6o) |
| `P-AB1`: one exponent against a control at the same point | the fit window | yes, by pairing |

Every rate quoted below is a field of
`claims/calibration/patching_exponent.json`, named where it is quoted. None is
inlined from a scratch measurement: 6m found three stale rates in one docstring
that way, and the rule since is pointers rather than digits wherever an
artifact holds the number.

3. WHAT THE MEASUREMENT GRID CONTRIBUTES, AND THE COMMON WINDOW
----------------------------------------------------------------
The window cancels within a pair; it does not make the pairs commensurable.
The sampling spread of a log-log slope goes like `1/sqrt(Sxx)` and `Sxx` is
fixed by K alone, so the fitted exponent's sd runs a factor of five across one
model's ablation grid -- `sampling_spread` tabulates it exactly, per unit of
log noise, and needs no replicates to do so. A MEAN of paired differences is then dominated by
whichever ablation point sits nearest the output and carries the least
information.

Two consequences, both taken here:

**The gate fits every ablation point over one COMMON window**, the largest one
every included point can supply, so the exponents it compares are exponents of
the same thing. The window is not a placed constant: it is `min_l K_l` over the
points the caller supplies, read off the input. What IS a choice is the
caller's ablation grid, and it trades directly against the floor -- `n` points
and a common window of `W` need `n + W <= L`, so on a 12-layer model six
ablation points leave a window of six, and on a six-layer model there is no
design at all.

**The units are combined by a SIGN and not by a mean.** Each ablation point
contributes `sign(beta_real - beta_control)`, so a point fitted over a short
window contributes exactly as much as one fitted over a long window and no
more. It also places no constant: 6i's change-mass centroid, CLAIM-C's
sign concordance and P-ST1's sign-of-a-difference are the same escape three
times over. The cost was measured rather than assumed -- against a mean of
paired differences on a grid whose windows deliberately run 3 to 27, power at
a planted exponent gap of 0.10 is 0.910 against 0.872 and at 0.05 is 0.429
against 0.383, with the H0 rate lower too, 0.030 against 0.045
(`statistic_choice`).

4. AND ONLY THEN THE CONTROL -- WHERE THE EXCHANGEABLE UNIT IS THE OPEN
   QUESTION
-------------------------------------------------------------------------
The control is the registry's: a **matched random-direction ablation of equal
magnitude at the same layer**. Equal magnitude is checked rather than assumed
(see `magnitude_match_report`) -- 6p's `P-S1` is the precedent, where a null
drawn at one arm's configuration and applied to the other's rejected at 1.000
on an input whose correct verdict was "no difference", and nothing had checked.

What the sign-flip null needs beyond per-point exchangeability is that the
signs are **independently** flippable, and that is a claim about the units and
not about the pair. `status-6.md`'s "49 layers are not 49 independent
observations" applies to ablation points inside one run exactly as it applied to
ALBERT's layers, so both readings were measured, at 6 prompts x 6 ablation
points, against a per-prompt shared component (rho = the share of each
difference that is the prompt's):

| rho | unit = ablation point | unit = prompt |
|---|---|---|
| 0.0 (independent) | 0.050 | 0.018 |
| 0.5 | **0.141** | 0.016 |
| 1.0 | **0.235** | 0.029 |

The per-prompt unit holds across the range; the per-ablation-point unit reaches
0.235, a fourth independent arrival of the fourfold-plus inflation POPPER
reports at 0.082 -> 0.340 and 6f, 6h and 6k each measured. Every rate is
CONDITIONAL ON EMISSION and `validity` carries the emission count beside it,
because the refusal below changes what that conditions on. **Which unit may
enter an e-process is a scientific decision of the same class as CLAIM-C's
criterion and P6-R2's**, so `REGISTERED_EXCHANGEABLE_UNIT` is `None`,
`adjudicate_p_ab1` raises while it is, and passing `unit=` selects what to
COMPUTE and never what may be adjudicated -- 6h's construction, and for its
reason: taking the decision after seeing a p-value would void the guarantee.

THE LIMITATION NO LABEL-SWAP NULL REMOVES, MEASURED RATHER THAN DESCRIBED

A shared component that is a random per-prompt draw is what the prompt unit
disposes of. A shared component that is a **fixed offset** -- real ablation
directions are not isotropic and the control directions are, so every cell is
nudged the same way for a reason with nothing to do with the field account --
is not, by either unit:

| fixed offset | unit = ablation point | unit = prompt |
|---|---|---|
| none | 0.050 | 0.018 |
| 0.5 jitter | 0.675 | 0.372 |
| 1.0 jitter | **1.000** | **0.895** |

That is 6i's shared-per-unit-factor at 1.00 in this design's clothing: a
confound present in every cell is present under every sign pattern. Every
record therefore carries `shared_prompt_factor_diagnostic`, which estimates how
much of the spread is between-prompt rather than within-prompt, and the analyst
must state what makes the control's direction population comparable to the real
one beyond its magnitude. `magnitude_match_report` checks the one thing the
registry named; nothing here can check the rest.

THE FALSIFIER AS REGISTERED IS NOT ONE AN E-PROCESS CAN CARRY

*"Divergence flat in remaining depth, or growing linearly, across ablation
points"* describes the NULL, and an e-process records insufficient evidence and
never a null accepted -- 6k recorded the same for `P-ST1` and the resolution is
the same. Both clauses map to INSUFFICIENT. The falsification branch is
**RECAPTURES**: the real ablation's exponent demonstrably BELOW its matched
control's, the perturbation reabsorbed faster than a structureless one of equal
magnitude, which is `design-7.md`'s own other side of the question and a
reversal positively shown rather than inferred from a failure to reject.

`p_reciprocal` decides that branch and is never calibrated into a claim's E,
CLAIM-B's and CLAIM-C's division.

AND WHAT THAT DIVISION TURNED OUT TO BE PROTECTING AGAINST

A fitted exponent is **not monotone** in how strongly an ablation propagates.
Divergence is bounded, so the arm whose divergence is LARGER at every layer
reaches its ceiling sooner inside a fixed window and its log-log slope
FLATTENS: at one true exponent of 2.0 fitted over eight layers, tau = 4 returns
1.35 where tau = 16 returns 1.95, and tau = 4 is the arm that dominates
everywhere. On two arms carrying the SAME true exponent where only the real one
saturates sooner -- which is what a real ablation that propagates does -- the
gate returned RECAPTURES, its falsification branch, on 0.98 of draws under the
prompt unit and 1.00 under the other. An input on which the prediction holds,
scored as the prediction refuted.

`power_law_arm` is the refusal, and its docstring records the first attempt at
it that the measurement threw out. What this leaves is a pre-computed
requirement on a pilot that constrains the INSTRUMENT rather than the analysis:
**the ablation magnitude and the fit window together must keep both arms inside
the power-law regime.** That is checkable off the divergence curves before any
p-value is computed, and it is not obtainable any other way.
"""

from __future__ import annotations

import math
from itertools import product
from typing import Dict, Optional, Sequence

import numpy as np


class PatchingRefused(Exception):
    """The design cannot support a p-value on this input. Never a failure."""


#: One-sided and fixed in advance. P-AB1 predicts the real ablation's exponent
#: to EXCEED its matched control's, so the predicted outcome is the large
#: statistic. Recorded as a constant so the tail cannot be picked afterward.
P_AB1_ALTERNATIVE = "greater"

#: The reciprocal one-sided test. It decides the RECAPTURES branch and is a
#: stop-rule input only -- see the module docstring for why that division is
#: load-bearing here and not a convention.
P_AB1_RECIPROCAL_ALTERNATIVE = "less"

#: The two readings of the registry's "permutation over ablation points". Only
#: one of them is a null; the other leaves the statistic invariant.
P_AB1_UNITS = ("prompt", "ablation_point")

#: Which one may enter an e-process. `None` until the author registers it: the
#: measurement is unambiguous and the DECISION is not a measurement. See the
#: module docstring's table and POPPER_PLAN.md 6h for the precedent.
REGISTERED_EXCHANGEABLE_UNIT: Optional[str] = None

#: The smallest fit window on which a log-log slope has a residual at all. Two
#: points interpolate exactly, so the curvature that detects saturation -- the
#: thing that makes the exponent a property of the window -- cannot be computed
#: there. This is a property of least squares and not a placed threshold.
MIN_FIT_POINTS = 3

#: Enumerate the sign-flip group exhaustively at or below this many patterns.
#: 2^20 is a million and well inside the pure tier's budget; above it the group
#: is sampled, with the +1 rule that keeps a sampled permutation p valid.
EXHAUSTIVE_SIGNFLIP_LIMIT = 1 << 20

#: Sampled sign patterns when the group is too large to enumerate. A module
#: constant and not a parameter, the convention every other gate here follows.
N_SIGNFLIP_DRAWS = 5000

_SEED = 20260827

#: Numerical identity, not a scientific tolerance. "Equal magnitude" is
#: something the CALLER constructs -- a control scaled to the real ablation's
#: norm agrees to floating-point noise, one that was not agrees visibly -- so
#: this separates those two cases and places nothing.
_MAGNITUDE_RTOL = 1e-9


# ---------------------------------------------------------------------------
# The estimator: a growth exponent, and what it is a property of
# ---------------------------------------------------------------------------

def fit_growth_exponent(divergence: Sequence[float],
                        window: Optional[int] = None) -> dict:
    """
    The OLS slope of log D against log k, over k = 1 .. window.

    `divergence[i]` is the trajectory divergence measured `i + 1` layers after
    the ablation point. The exponent is what P-AB1's statement is about --
    "grows superlinearly with the number of layers after the ablation point" --
    and `window_sensitivity` is what says how much of it belongs to the window
    rather than to the data.

    `window_sensitivity` is the full-window exponent minus the exponent fitted
    over the first half of the same points. Zero on a pure power law at any
    window; negative when the curve is bending over, which is the regime in
    which the exponent is a decreasing function of how far the fit ran. It is
    reported, never scored, and never thresholded: what it is FOR is telling a
    reader whether the two arms of a pair bent by different amounts, and that
    comparison belongs to the pair rather than to either arm.
    """
    d = np.asarray(list(divergence), dtype=np.float64)
    n_native = int(d.size)
    k_max = n_native if window is None else int(window)
    if k_max > n_native:
        raise PatchingRefused(
            f"a window of {k_max} was asked of a curve with {n_native} points")
    if k_max < MIN_FIT_POINTS:
        raise PatchingRefused(
            f"window {k_max} is below MIN_FIT_POINTS={MIN_FIT_POINTS}: a slope "
            f"through fewer points interpolates exactly, so it carries no "
            f"residual and its window-sensitivity cannot be computed")
    d = d[:k_max]
    if not np.all(np.isfinite(d)):
        raise PatchingRefused(
            f"{int(np.sum(~np.isfinite(d)))} of {k_max} divergence values are "
            f"not finite; a curve thinned by failures is not the curve fitted")
    if np.any(d <= 0.0):
        raise PatchingRefused(
            f"{int(np.sum(d <= 0.0))} of {k_max} divergence values are <= 0. A "
            f"growth EXPONENT is a slope in log space and log 0 is not a small "
            f"number, so this is a degeneracy and not a tolerance: a divergence "
            f"that is exactly zero says the ablation changed nothing at that "
            f"layer, which is a fact about the run and not about the fit")

    k = np.arange(1, k_max + 1, dtype=np.float64)
    y = np.log(d)
    x = np.log(k)
    slope, resid = _ols_slope(x, y)
    half = max(2, int(np.ceil(k_max / 2.0)))
    slope_half, _ = _ols_slope(x[:half], y[:half])
    return {
        "exponent": float(slope),
        "window": int(k_max),
        "n_points_native": n_native,
        "residual_rms": float(resid),
        "exponent_first_half": float(slope_half),
        "half_window": int(half),
        "window_sensitivity": float(slope - slope_half),
        "bend_z": _bend_z(x, y, half),
    }


def _ols_weights(x: np.ndarray) -> np.ndarray:
    xc = x - x.mean()
    return xc / float((xc * xc).sum())


def _bend_z(x: np.ndarray, y: np.ndarray, half: int) -> float:
    """
    `window_sensitivity` divided by the spread this curve's OWN residual gives
    it, under the null that the curve is a power law over the window.

    Both slopes are linear in `y`, so their difference is `(w_full - w_half).y`
    and its variance is `sigma^2 ||w_full - w_half||^2` with `sigma^2` the OLS
    residual variance at `window - 2` degrees of freedom. Nothing is placed: the
    scale comes from the curve.
    """
    K = int(x.size)
    if K < 3:
        return 0.0
    wf = _ols_weights(x)
    wh = np.zeros(K, dtype=np.float64)
    wh[:half] = _ols_weights(x[:half])
    delta = wf - wh
    slope = float(wf @ y)
    xc = x - x.mean()
    resid = y - (y.mean() + slope * xc)
    s2 = float((resid * resid).sum()) / float(max(1, K - 2))
    #: An exactly-fitted curve has no residual, so the bend has no noise scale
    #: and the ratio is 0/0 rather than large. Comparing against the floating
    #: scale of `y` rather than against zero is the difference between
    #: reporting 0.0 and reporting whatever 1e-16/1e-32 happens to be -- which
    #: is what the first version of this function did on a noiseless power law.
    scale = float(np.finfo(np.float64).eps * max(1.0, float((y * y).mean())))
    if s2 <= scale:
        return 0.0
    var = s2 * float((delta * delta).sum())
    if var <= 0.0:
        return 0.0
    return float(float(delta @ y) / np.sqrt(var))


def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple:
    xc = x - x.mean()
    sxx = float((xc * xc).sum())
    if sxx <= 0.0:
        raise PatchingRefused(
            "every fitted point sits at the same k; there is no slope")
    slope = float((xc * (y - y.mean())).sum() / sxx)
    resid = y - (y.mean() + slope * xc)
    return slope, float(np.sqrt(float((resid * resid).mean())))


def window_reference_report(native_windows: Sequence[int]) -> dict:
    """
    What the ablation grid contributes, decidable before a divergence is
    measured: which points can carry a fit, what window they all share, and how
    many units are left once the short ones are dropped.

    The whole report reads the grid and `MIN_FIT_POINTS`. It sees no data, and
    that is the point -- `EVALUABILITY.md` records the grid's contribution as
    one of the three defect kinds that are checkable before any data exists,
    and all three were missed anyway on the nine rows that had them.
    """
    w = np.asarray(list(native_windows), dtype=int)
    if w.size == 0:
        raise PatchingRefused("no ablation points")
    common = int(w.min())
    short = np.flatnonzero(w < MIN_FIT_POINTS)
    return {
        "native_windows": [int(v) for v in w],
        "n_ablation_points": int(w.size),
        "common_window": common,
        "points_below_min_fit": [int(i) for i in short],
        "usable": bool(common >= MIN_FIT_POINTS),
        "min_fit_points": MIN_FIT_POINTS,
        "_note": (
            "Every point is fitted over `common_window` and not over its own "
            "native depth. A log-log slope on a saturating curve DECREASES with "
            "the window -- measured at 1.79 -> 0.66 from K=3 to K=24 on one "
            "fixed set of dynamics -- so exponents fitted over different "
            "windows are not comparable, and the ablation point fixes the "
            "window. The window cancels inside a pair because both arms share "
            "it; nothing makes it cancel across points."),
    }


def magnitude_match_report(magnitude_real: Sequence[float],
                           magnitude_control: Sequence[float]) -> dict:
    """
    The registry requires "a MATCHED RANDOM-DIRECTION ablation of EQUAL
    MAGNITUDE at the same layer". Nothing checked it until now.

    6p's `P-S1` is why this exists rather than being left to the caller: there
    a null drawn at one arm's configuration and applied to the other's rejected
    at 1.000 on two i.i.d. arms -- an input the design could not compare, scored
    anyway -- and the fix was a refusal, not a caveat. Here the mismatch is
    worse behaved, because divergence grows with the size of the perturbation
    and the exponent's saturation point moves with it, so an unmatched control
    biases the contrast in whichever direction the mismatch runs.
    """
    a = np.asarray(list(magnitude_real), dtype=np.float64)
    b = np.asarray(list(magnitude_control), dtype=np.float64)
    if a.shape != b.shape:
        raise PatchingRefused(
            f"{a.shape} real magnitudes against {b.shape} control magnitudes")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise PatchingRefused("a magnitude is not finite")
    scale = np.maximum(np.abs(a), np.abs(b))
    scale[scale == 0.0] = 1.0
    rel = np.abs(a - b) / scale
    worst = float(rel.max()) if rel.size else 0.0
    return {
        "n_cells": int(a.size),
        "worst_relative_mismatch": worst,
        "rtol": _MAGNITUDE_RTOL,
        "matched": bool(worst <= _MAGNITUDE_RTOL),
        "_note": (
            "rtol is numerical identity and not a scientific tolerance: a "
            "control the caller scaled to the real ablation's norm agrees to "
            "floating-point noise, and one that was not scaled disagrees "
            "visibly. Nothing is placed between those two cases."),
    }


# ---------------------------------------------------------------------------
# The floor, computed from the design
# ---------------------------------------------------------------------------

def attainable_floor_report(n_units: int, n_informative: int,
                            alpha: float,
                            n_patterns: Optional[int] = None) -> dict:
    """
    The smallest p a sign-flip over `n_units` can express, before any data.

    A unit whose contribution is zero adds the same number to the observation
    and to every sign pattern, so it is enumerated and never counted: with `k`
    of `n` informative the floor is `(2^(n-k) + 1)/(2^n + 1)`, which is
    `2/(2^n + 1)` exactly when k = n. That is CLAIM-C's rule (`POPPER_PLAN.md`
    6l) and not a new one -- the same arithmetic reaches this design because the
    null is the same group.

    TWO FLOORS, AND WHICH ONE BINDS DEPENDS ON THE UNIT

    The rule above is the DESIGN's floor and contains no draw count. When the
    group is too large to enumerate it is sampled, and then the run also cannot
    express anything below `2/(n_patterns + 1)` -- the identity is in the
    pattern list, so it always ties. **The smallest p a run can express is the
    MAX of the two, and they bind at opposite ends**, which is 6p's finding for
    `P-T1` in a second setting.

    The gap is not decorative. Under the per-ablation-point unit at six prompts
    and seven points there are 42 units, the design floor is 4.5e-13, the group
    is sampled, and a PERFECT input returns 4.0e-4 -- nine orders of magnitude
    apart. Reporting the design floor alone there would be 6i's defect exactly:
    a reported floor smaller than any p the arm can express. Under the per-prompt
    unit the group enumerates and the design floor is the whole story.

    The way a unit lands at zero here is worth naming because it is exact and
    avoidable. Under the PROMPT unit a prompt contributes the sum of its
    ablation points' signs, so a prompt with an EVEN number of usable points can
    split evenly and contribute nothing. Under H0 each sign is a fair coin, so
    at six prompts a six-point grid leaves the design able to reject at all on
    only 0.394 of draws and an eight-point grid on 0.479, while five, seven or
    nine points leave it able on 1.000. **An odd number of ablation points per
    prompt is free and an even number is not**, and that is arithmetic on the
    grid rather than a fact about any model.
    """
    n = int(n_units)
    k = int(n_informative)
    a = float(alpha)
    if k > n:
        raise PatchingRefused(f"{k} informative units of {n}")
    design = (2.0 ** (n - k) + 1.0) / (2.0 ** n + 1.0)
    sampling = (2.0 / (int(n_patterns) + 1.0)
                if n_patterns is not None else None)
    floor = design if sampling is None else max(design, sampling)
    min_units = None
    for m in range(1, 64):
        if 2.0 / (2.0 ** m + 1.0) <= a:
            min_units = m
            break
    return {
        "n_units": n,
        "n_informative": k,
        "alpha": a,
        "design_floor": float(design),
        "sampling_floor": (float(sampling) if sampling is not None else None),
        "attainable_floor": float(floor),
        "binds": ("design" if sampling is None or design >= sampling
                  else "sampling"),
        "floor_all_informative": float(2.0 / (2.0 ** n + 1.0)),
        "min_informative_units_for_alpha": min_units,
        "sufficient": bool(floor <= a),
        "_note": (
            "`design_floor` contains no draw count and `sampling_floor` is "
            "nothing but one; `attainable_floor` is the max and is what a "
            "refusal must read. The registry's literal 'permutation over "
            "ablation points' -- permuting which point's real exponent meets "
            "which point's control exponent -- leaves a mean paired difference "
            "INVARIANT, so its floor is 1.000 and no input whatever could "
            "reject. See the module docstring."),
    }


def registered_null_invariance_report() -> dict:
    """
    The registered null's literal reading, and why it is not a null.

    Recorded as a computation the caller can run rather than as a paragraph,
    because 6p's lesson is that a floor read off a draw count instead of the
    design is the commonest defect these nine passes found and the cheapest to
    check. This one is checkable with algebra alone: permuting the pairing
    leaves `mean(beta_real) - mean(beta_control)` unchanged.
    """
    return {
        "reading": "permute which ablation point's real exponent is compared "
                   "against which ablation point's control exponent",
        "statistic": "mean_l ( beta_real(l) - beta_control(l) )",
        "invariant_under_the_null": True,
        "design_floor": 1.0,
        "why": (
            "the permuted statistic is mean(beta_real) - mean(beta_control) for "
            "every permutation, so every draw ties the observation and p = 1 on "
            "every input the design can be handed"),
        "reading_used_instead": (
            "swap the LABELS of the real and the matched control at an ablation "
            "point. Under H0 -- the ablation removes a value from a sum -- the "
            "two directions are exchangeable at equal magnitude and the same "
            "layer, so the swap is exact by construction"),
    }


# ---------------------------------------------------------------------------
# The paired differences, and the sign-flip arm over them
# ---------------------------------------------------------------------------

def paired_exponents(divergence_real, divergence_control,
                     window: Optional[int] = None) -> dict:
    """
    Fit both arms of every (prompt, ablation point) cell over one common window.

    `divergence_real[p][l]` is the divergence curve for the real ablation at
    prompt `p`, ablation point `l`; `divergence_control[p][l]` is its matched
    control's. Returns the per-cell exponents, their difference, and the
    saturation diagnostic the pair -- and only the pair -- can carry.
    """
    real = [list(row) for row in divergence_real]
    ctrl = [list(row) for row in divergence_control]
    if len(real) != len(ctrl):
        raise PatchingRefused(
            f"{len(real)} prompts of real curves against {len(ctrl)} of control")
    if not real:
        raise PatchingRefused("no prompts")
    n_points = len(real[0])
    for p, (a, b) in enumerate(zip(real, ctrl)):
        if len(a) != len(b):
            raise PatchingRefused(
                f"prompt {p}: {len(a)} real ablation points against {len(b)} "
                f"control ones; they index the same points and must match")
        if len(a) != n_points:
            raise PatchingRefused(
                f"prompt {p} has {len(a)} ablation points against prompt 0's "
                f"{n_points}. The sign-flip null flips a whole prompt at once, "
                f"so a ragged grid would give prompts unequal weight without "
                f"anything in the record saying so")
    if n_points == 0:
        raise PatchingRefused("no ablation points")

    #: One native window per ABLATION POINT, the shortest curve any prompt
    #: supplies there. The report is about the grid and not about a cell.
    native = [min(min(len(real[p][l]), len(ctrl[p][l]))
                  for p in range(len(real)))
              for l in range(n_points)]
    grid = window_reference_report(native)
    w = int(window) if window is not None else grid["common_window"]
    if w > grid["common_window"]:
        raise PatchingRefused(
            f"a common window of {w} was asked of a grid whose shortest "
            f"ablation point supplies {grid['common_window']}: native windows "
            f"{grid['native_windows']}")
    if w < MIN_FIT_POINTS:
        raise PatchingRefused(
            f"the common fit window is {w}, below MIN_FIT_POINTS="
            f"{MIN_FIT_POINTS}: ablation point(s) {grid['points_below_min_fit']} "
            f"sit too close to the output for a slope to be fitted at all. "
            f"Nothing is dropped here -- dropping a point changes the unit "
            f"count and therefore the attainable floor, which is the caller's "
            f"decision about the ablation grid and not this gate's. Native "
            f"windows: {grid['native_windows']}")

    n_prompts = len(real)
    exp_real = np.zeros((n_prompts, n_points))
    exp_ctrl = np.zeros((n_prompts, n_points))
    sens_real = np.zeros((n_prompts, n_points))
    sens_ctrl = np.zeros((n_prompts, n_points))
    z_real = np.zeros((n_prompts, n_points))
    z_ctrl = np.zeros((n_prompts, n_points))
    for p in range(n_prompts):
        for l in range(n_points):
            fr = fit_growth_exponent(real[p][l], window=w)
            fc = fit_growth_exponent(ctrl[p][l], window=w)
            exp_real[p, l] = fr["exponent"]
            exp_ctrl[p, l] = fc["exponent"]
            sens_real[p, l] = fr["window_sensitivity"]
            sens_ctrl[p, l] = fc["window_sensitivity"]
            z_real[p, l] = fr["bend_z"]
            z_ctrl[p, l] = fc["bend_z"]

    diff = exp_real - exp_ctrl
    return {
        "window": w,
        "grid": grid,
        "n_prompts": n_prompts,
        "n_ablation_points": n_points,
        "exponent_real": exp_real,
        "exponent_control": exp_ctrl,
        "difference": diff,
        "sensitivity_real": sens_real,
        "sensitivity_control": sens_ctrl,
        "bend_contrast": sens_real - sens_ctrl,
        "bend_z_real": z_real,
        "bend_z_control": z_ctrl,
        "saturation_diagnostic": _saturation_diagnostic(sens_real, sens_ctrl),
    }


def _saturation_diagnostic(sens_real: np.ndarray,
                           sens_ctrl: np.ndarray) -> dict:
    """
    How much of each arm's exponent is the window, and whether the two arms
    bent by different amounts.

    A negative `paired_mean` says the real arm bends MORE than its control --
    the arm that diverges further saturates sooner and reads as less
    superlinear. That biases the contrast toward the null, which is conservative
    for PROPAGATES and ANTICONSERVATIVE for RECAPTURES. Reported and never
    scored; the branch it could manufacture is the reciprocal one, which by
    construction reaches no ledger.
    """
    paired = sens_real - sens_ctrl
    return {
        "mean_window_sensitivity_real": float(sens_real.mean()),
        "mean_window_sensitivity_control": float(sens_ctrl.mean()),
        "paired_mean": float(paired.mean()),
        "paired_fraction_negative": float(np.mean(paired < 0.0)),
        "_note": (
            "window_sensitivity is the full-window exponent minus the "
            "first-half one: 0 on a pure power law, negative on a bending "
            "curve. A negative paired_mean attenuates the PROPAGATES contrast "
            "and inflates the RECAPTURES one, so read it before reading a "
            "reciprocal result. It enters no p-value."),
    }


def shared_prompt_factor_diagnostic(diff: np.ndarray) -> dict:
    """
    How much of the per-cell difference is the prompt's rather than the cell's.

    6i's `shared_unit_factor_diagnostic` for this design, and it exists for the
    same reason: a component common to a whole prompt is exactly what makes the
    per-ablation-point unit anticonservative, measured at 0.357 against a
    nominal 0.05 when the whole difference is shared. It catches a shared factor
    that varies BETWEEN prompts and catches nothing else -- a FIXED offset
    common to every prompt leaves this estimate at zero and takes both units to
    1.000, which is the limitation stated in the module docstring and not
    removed by anything here.
    """
    d = np.asarray(diff, dtype=np.float64)
    n_p, n_l = d.shape
    within = float(d.var(axis=1, ddof=0).mean()) if n_l > 1 else 0.0
    between = float(d.mean(axis=1).var(ddof=0)) if n_p > 1 else 0.0
    total = within + between
    return {
        "between_prompt_variance": between,
        "within_prompt_variance": within,
        "shared_share_estimate": float(between / total) if total > 0 else 0.0,
        "prompt_sign_agreement": float(
            np.mean([np.abs(np.sign(row).sum()) / n_l for row in d])),
        "_note": (
            "shared_share_estimate is a naive between-over-total ratio, not a "
            "variance-components fit: it is a diagnostic to read and not a "
            "correction to apply, and nothing in this module divides by it. A "
            "fixed offset common to EVERY prompt does not show up here at all."),
    }


def _sign_patterns(n: int, seed: int):
    """The flip group, enumerated when small and sampled when not. The identity
    is in the list either way -- it reproduces the observation, which is what
    makes the smallest attainable p a floor rather than zero."""
    if 2 ** n <= EXHAUSTIVE_SIGNFLIP_LIMIT:
        return np.array(list(product((-1.0, 1.0), repeat=n))), True
    rng = np.random.default_rng(seed)
    pats = rng.choice((-1.0, 1.0), size=(N_SIGNFLIP_DRAWS, n))
    return np.vstack([np.ones((1, n)), pats]), False


def signflip_arm(diff: np.ndarray, unit: str, alpha: float,
                 seed: int = _SEED) -> dict:
    """
    The sign-flip null over the chosen unit.

    `unit="prompt"` gives one bit per prompt and flips the whole prompt's block
    of ablation points together; `unit="ablation_point"` gives one bit per
    (prompt, ablation point) cell. The two differ only in what is assumed
    independent, and the module docstring's table is the measurement of what
    that assumption costs.
    """
    d = np.asarray(diff, dtype=np.float64)
    if unit not in P_AB1_UNITS:
        raise PatchingRefused(f"unit={unit!r} is not one of {P_AB1_UNITS}")
    if not np.all(np.isfinite(d)):
        raise PatchingRefused("a fitted exponent difference is not finite")

    s = np.sign(d)
    if unit == "prompt":
        contrib = s.sum(axis=1)
    else:
        contrib = s.ravel()

    n_units = int(contrib.size)
    n_informative = int(np.sum(contrib != 0.0))
    pats, exhaustive = _sign_patterns(n_units, seed)
    n_pat = int(pats.shape[0])
    floor = attainable_floor_report(n_units, n_informative, alpha,
                                    n_patterns=(None if exhaustive else n_pat))
    if not floor["sufficient"]:
        raise PatchingRefused(
            f"attainable floor {floor['attainable_floor']:.4g} exceeds alpha="
            f"{alpha} (the {floor['binds']} floor binds): with "
            f"{n_informative} informative {unit} unit(s) of {n_units}, no input "
            f"whatever could reject, and reporting 'not significant' from a "
            f"design that could not have rejected reads as evidence against the "
            f"prediction. At least "
            f"{floor['min_informative_units_for_alpha']} informative units are "
            f"needed.")

    stats = pats @ contrib
    obs = float(contrib.sum())
    p_greater = float((np.sum(stats >= obs - 1e-12) + 1) / (n_pat + 1))
    p_less = float((np.sum(stats <= obs + 1e-12) + 1) / (n_pat + 1))
    return {
        "arm": "paired-exponent sign flip",
        "unit": unit,
        "n_units": n_units,
        "n_informative_units": n_informative,
        "observed": obs,
        "max_attainable": float(np.abs(contrib).sum()),
        "p_value": p_greater,
        "p_reciprocal": p_less,
        "n_patterns": n_pat,
        "exhaustive": bool(exhaustive),
        "attainable_floor": floor,
        "alternative": P_AB1_ALTERNATIVE,
        "reciprocal_alternative": P_AB1_RECIPROCAL_ALTERNATIVE,
    }


# ---------------------------------------------------------------------------
# The power-law arm, and the refusal it drives
# ---------------------------------------------------------------------------

def power_law_arm(bend_z_real: np.ndarray, bend_z_control: np.ndarray,
                  alpha: float) -> dict:
    """
    Is each arm's divergence a POWER LAW over the fit window, or has it reached
    its ceiling inside it?

    WHY THE GATE REFUSES ON THIS AND NOT ON A CAVEAT

    A fitted exponent is not monotone in how strongly an ablation propagates.
    Divergence is bounded, so an ablation whose divergence is LARGER at every
    layer reaches the ceiling sooner inside a fixed window and its log-log slope
    FLATTENS. Measured on `D(k) = 1 - exp(-(k/tau)^beta)` at one true exponent
    of 2.0 fitted over 8 layers, tau = 4 returns 1.35 and tau = 16 returns 1.95
    -- and the tau = 4 arm is the larger one at every k.

    So on two arms with the same true exponent where the real one saturates
    sooner -- which is what a real ablation that propagates does -- the gate
    returns `RECAPTURES`, its registered FALSIFICATION branch, on **0.98** of
    draws under the prompt unit and **1.00** under the other -- re-scored in
    `validity`'s counterfactual columns, on exactly the draws this arm now turns
    away. An input on which the prediction holds, scored as the prediction
    refuted. That is 6p's fourth defect kind ("an input the design cannot
    compare, scored anyway") in its worst form, because the verdict it produces
    is the one that would enter the ledger as a falsification.

    THE FIRST ATTEMPT AT THIS REFUSAL WAS THE PAIRED CONTRAST, AND MEASUREMENT
    THREW IT OUT

    The obvious condition is that the two arms bend by DIFFERENT amounts, since
    an equal bend cancels in the pair -- and it is testable with the gate's own
    sign-flip null, two-sided, on the per-cell paired window sensitivity. It is
    the right shape and it is too weak to do the job: `discarded_refusal`
    measures it on the differential-saturation family and it turned away 52 of
    100 draws under the prompt unit, while the 48 it let through still returned
    RECAPTURES on **0.979** of them; under the other unit 23 got through and
    **1.000** of those did. A refusal that thins a defect is not a refusal. It is kept as a reported diagnostic --
    `bend_contrast_arm` names the direction of the confound when there is one --
    and the refusal is this arm instead.

    WHAT THIS ONE ASKS

    The exponent is a growth exponent only where the curve is a power law over
    the window, and that is a property of ONE arm rather than of the pair.
    `fit_growth_exponent` returns each curve's `bend_z`: its
    `window_sensitivity` divided by the spread its own OLS residual gives it
    under "this curve is a power law". Pooling those over the arm's cells as
    `sum(z) / sqrt(n)` gives a standard normal under that null, and the arm
    refuses two-sided at `alpha / 2` -- Bonferroni over the two arms, so the
    refusal rate on a pure power law is alpha and not twice it. 6n's rule, which
    this project learned by putting a per-cell bound on a two-cell family and
    watching it fire once in twenty regenerations.

    Measured over this design's own 42 cells at multiplicative noise 0.20
    (`power_law_arm_operating_curve`): the refusal fires on 0.060 of pure power
    laws and of tau = 30, 0.360 at tau = 15, 0.970 at tau = 8, and 1.000 at
    tau = 5 and below. Nominal where the shape is the one it is meant to admit,
    and certain where it is not.

    IT COSTS VERDICTS, AND WHICH KIND OF COST IS PART OF THE CLAIM

    6p records three kinds of "this costs nothing" -- measured, proved and
    enumerated -- and 6o records a fourth category, a refusal that is right and
    costs verdicts anyway. This is the second of those. It turns away inputs the
    gate would have scored, including the symmetric case where both arms bend
    by the SAME amount and the contrast is nominal (measured). That case is
    refused deliberately: an exponent fitted through a ceiling is a property of
    the window, so a `PROPAGATES` verdict there would say the real arm's window
    artifact exceeded the control's, which is not what P-AB1 predicts. What is
    refused is a verdict the design cannot SUPPORT rather than one it could not
    REACH, and the calibration re-scores the counterfactual in every family
    instead of asserting the cost is small.
    """
    zr = np.asarray(bend_z_real, dtype=np.float64).ravel()
    zc = np.asarray(bend_z_control, dtype=np.float64).ravel()
    if zr.size == 0 or zc.size == 0:
        raise PatchingRefused("no curves to test for power-law shape")
    if not (np.all(np.isfinite(zr)) and np.all(np.isfinite(zc))):
        raise PatchingRefused("a curve's bend statistic is not finite")
    per_arm = float(alpha) / 2.0
    out = {}
    for name, z in (("real", zr), ("control", zc)):
        pooled = float(z.sum() / np.sqrt(float(z.size)))
        p_two = float(math.erfc(abs(pooled) / math.sqrt(2.0)))
        out[name] = {
            "pooled_z": pooled,
            "p_two_sided": p_two,
            "n_curves": int(z.size),
            "rejects": bool(p_two <= per_arm),
        }
    return {
        "arm": "pooled per-curve power-law test",
        "alpha": float(alpha),
        "per_arm_level": per_arm,
        "real": out["real"],
        "control": out["control"],
        "not_a_power_law": bool(out["real"]["rejects"]
                                or out["control"]["rejects"]),
        "_note": (
            "Pooling assumes the per-curve bends are independent across cells. "
            "That is an assumption and it is a much weaker one than the "
            "exponent contrast needs: a bend is a WITHIN-curve quantity driven "
            "by that curve's own residual, where the exponent contrast is "
            "driven by whatever the prompt and the layer share. Calibrated at "
            "0.045 on pure power laws over 42 cells."),
    }


# ---------------------------------------------------------------------------
# The bend contrast, reported and no longer refused on
# ---------------------------------------------------------------------------

def bend_contrast_arm(bend: np.ndarray, unit: str, alpha: float,
                      seed: int = _SEED) -> dict:
    """
    The same sign-flip null, run TWO-SIDED on the per-cell paired
    window-sensitivity instead of on the exponent: do the two arms bend by
    DIFFERENT amounts?

    THIS WAS THE REFUSAL AND IS NOW A DIAGNOSTIC, AND MEASUREMENT DECIDED WHICH

    An equal bend cancels inside the pair, so the confound that reverses the
    statistic's sign is a bend CONTRAST, and testing it with the gate's own
    exact null was the obvious refusal to reach for. It is the right shape and
    it is too weak: `discarded_refusal` measures it on the
    differential-saturation family, where it turns away 52 of 100 draws under
    the prompt unit and the 48 it lets through still return RECAPTURES on 0.979
    of them. `power_law_arm` refuses instead, on the stronger per-ARM question,
    and this arm stays because it names the DIRECTION of the confound when there
    is one -- which arm bends more, and so which verdict is being attenuated and
    which inflated. `confounded` is reported and nothing reads it.

    6o's first attempt at CLAIM-B's refusal was thrown out the same way and for
    the same reason: a condition that looks like the right shape, does not track
    the axis that matters, and is corrected by sweeping it rather than by
    reading it. Both sweeps are in their artifacts because they are what changed
    the design.
    """
    d = np.asarray(bend, dtype=np.float64)
    if unit not in P_AB1_UNITS:
        raise PatchingRefused(f"unit={unit!r} is not one of {P_AB1_UNITS}")
    if not np.all(np.isfinite(d)):
        raise PatchingRefused("a window-sensitivity contrast is not finite")
    s = np.sign(d)
    contrib = s.sum(axis=1) if unit == "prompt" else s.ravel()
    n_units = int(contrib.size)
    pats, exhaustive = _sign_patterns(n_units, seed)
    stats = np.abs(pats @ contrib)
    obs = abs(float(contrib.sum()))
    n_pat = int(pats.shape[0])
    p_two = float((np.sum(stats >= obs - 1e-12) + 1) / (n_pat + 1))
    floor = attainable_floor_report(
        n_units, int(np.sum(contrib != 0.0)), alpha,
        n_patterns=(None if exhaustive else n_pat))
    return {
        "arm": "paired window-sensitivity sign flip (two-sided)",
        "unit": unit,
        "n_units": n_units,
        "observed": float(contrib.sum()),
        "p_two_sided": p_two,
        "n_patterns": n_pat,
        "exhaustive": bool(exhaustive),
        "attainable_floor": floor,
        "confounded": bool(p_two <= alpha and floor["sufficient"]),
        "_note": (
            "`confounded` requires the arm to be able to reject at all: where "
            "its own attainable floor exceeds alpha it can never fire, and a "
            "refusal that is impossible by arithmetic must not be reported as "
            "a check that passed."),
    }


# ---------------------------------------------------------------------------
# The stop rule
# ---------------------------------------------------------------------------

def gate_verdict(p_greater: Optional[float], p_less: Optional[float],
                 alpha: Optional[float] = None) -> dict:
    """
    Three-way, and only one branch is a falsification -- CLAIM-C's and
    CLAIM-B's shape, for the reason 6k gives: the registered falsifier
    describes the null, and an e-process never accepts a null.
    """
    a = _alpha() if alpha is None else float(alpha)
    if p_greater is None:
        return {"verdict": "INSUFFICIENT", "falsified": False, "alpha": a,
                "reading": "no p-value could be emitted; nothing is adjudicated"}
    if p_greater <= a:
        return {"verdict": "PROPAGATES", "falsified": False, "alpha": a,
                "reading": "the real ablation's divergence grows with a larger "
                           "exponent in remaining depth than a structureless "
                           "perturbation of equal magnitude at the same layer"}
    if p_less is not None and p_less <= a:
        return {"verdict": "RECAPTURES", "falsified": True, "alpha": a,
                "reading": "the real ablation's exponent is demonstrably BELOW "
                           "its matched control's -- the perturbation reabsorbed "
                           "faster than a structureless one. The falsifier, "
                           "positively shown. Read saturation_diagnostic first: "
                           "an arm that diverges further bends sooner and can "
                           "manufacture this branch"}
    return {"verdict": "INSUFFICIENT", "falsified": False, "alpha": a,
            "reading": "neither propagation nor recapture was shown. Nothing is "
                       "falsified -- 'flat or linear in remaining depth' is the "
                       "NULL, and an e-process records insufficient evidence "
                       "rather than a null accepted"}


def _alpha() -> float:
    from core.adjudication import load_registry
    from core.evalues import DEFAULT_ALPHA
    try:
        return float(load_registry().get("alpha", DEFAULT_ALPHA))
    except Exception:
        return float(DEFAULT_ALPHA)


# ---------------------------------------------------------------------------
# P-AB1's gate
# ---------------------------------------------------------------------------

def p_value_p_ab1(divergence_real,
                  divergence_control,
                  magnitude_real,
                  magnitude_control,
                  *,
                  unit: str = "prompt",
                  window: Optional[int] = None,
                  alpha: Optional[float] = None,
                  seed: int = _SEED) -> dict:
    """
    P-AB1's p-value.

    `divergence_real[p][l][k]` is the trajectory divergence `k + 1` layers after
    ablation point `l` on prompt `p`; `divergence_control` is the matched
    random-direction ablation of equal magnitude at the same layer.
    `magnitude_real[p][l]` and `magnitude_control[p][l]` are the two ablations'
    magnitudes, which the registry requires to be equal and which nothing
    checked before this module.

    Refuses -- `p_value` None with a `reason` -- rather than returning a number
    the design cannot support.
    """
    a = _alpha() if alpha is None else float(alpha)
    out: dict = {
        "prediction_id": "P-AB1",
        "claim": "H-BRIDGE",
        "unit_computed": unit,
        "registered_unit": REGISTERED_EXCHANGEABLE_UNIT,
        "registered_null_reading": registered_null_invariance_report(),
        "window": None,
        "grid": None,
        "magnitude_match": None,
        "saturation_diagnostic": None,
        "shared_prompt_factor": None,
        "bend_contrast": None,
        "power_law": None,
        "arm": None,
        "p_value": None,
        "p_reciprocal": None,
        "reason": None,
    }
    try:
        pair = paired_exponents(divergence_real, divergence_control,
                                window=window)
        # Filled in before the magnitude refusal can fire: a refused record
        # that still says what grid it was handed is worth more than one that
        # says only that it refused.
        out["window"] = pair["window"]
        out["grid"] = pair["grid"]
        n_cells = pair["n_prompts"] * pair["n_ablation_points"]
        mr = np.asarray(magnitude_real, dtype=np.float64).ravel()
        mc = np.asarray(magnitude_control, dtype=np.float64).ravel()
        if mr.size != n_cells or mc.size != n_cells:
            raise PatchingRefused(
                f"{mr.size} real and {mc.size} control magnitudes were given "
                f"for a grid of {pair['n_prompts']} prompts x "
                f"{pair['n_ablation_points']} ablation points = {n_cells} "
                f"cells. Magnitudes that do not index the curves cannot say "
                f"whether the control was matched, and an unchecked match is "
                f"the thing this refusal exists for.")
        out["magnitude_match"] = magnitude_match_report(mr, mc)
        if not out["magnitude_match"]["matched"]:
            raise PatchingRefused(
                f"the control's magnitude does not match the real ablation's: "
                f"worst relative mismatch "
                f"{out['magnitude_match']['worst_relative_mismatch']:.3e}. The "
                f"registry's control is 'of EQUAL magnitude at the same layer', "
                f"and divergence grows with the size of the perturbation, so an "
                f"unmatched control biases the contrast in whichever direction "
                f"the mismatch runs. This is an input the design cannot "
                f"compare, and 6p's P-S1 is what happens when one is scored "
                f"anyway")

        out["saturation_diagnostic"] = pair["saturation_diagnostic"]
        out["shared_prompt_factor"] = shared_prompt_factor_diagnostic(
            pair["difference"])
        out["bend_contrast"] = bend_contrast_arm(
            pair["bend_contrast"], unit, a, seed=seed)
        pl = power_law_arm(pair["bend_z_real"], pair["bend_z_control"], a)
        out["power_law"] = pl
        if pl["not_a_power_law"]:
            raise PatchingRefused(
                f"at least one arm's divergence is not a power law over the "
                f"fit window (pooled bend z: real "
                f"{pl['real']['pooled_z']:.2f}, control "
                f"{pl['control']['pooled_z']:.2f}, refusing two-sided at "
                f"alpha/2={pl['per_arm_level']}). A fitted exponent is not "
                f"monotone in how strongly an ablation propagates -- the arm "
                f"whose divergence is larger at every layer reaches its ceiling "
                f"sooner inside the window and reads as LESS superlinear -- so "
                f"on such an input the gate returns RECAPTURES, its "
                f"falsification branch, at 1.000 where the correct verdict is "
                f"INSUFFICIENT. This refusal COSTS verdicts; see "
                f"`power_law_arm` for which kind of cost that is.")
        arm = signflip_arm(pair["difference"], unit, a, seed=seed)
        out["arm"] = arm
        out["p_value"] = arm["p_value"]
        out["p_reciprocal"] = arm["p_reciprocal"]
    except PatchingRefused as exc:
        out["reason"] = str(exc)
        out.update(gate_verdict(None, None, a))
        return out

    out.update(gate_verdict(out["p_value"], out["p_reciprocal"], a))
    return out


def adjudicate_p_ab1(result: dict,
                     *,
                     artifact_hashes: Sequence[str] = (),
                     run_manifest: Optional[dict] = None,
                     adjudicate: bool = False,
                     adjudications_dir=None) -> dict:
    """
    `p_value_p_ab1`'s result plus, optionally, an entry in the falsification
    ledger.

    Refuses while `REGISTERED_EXCHANGEABLE_UNIT` is `None`, and refuses a result
    computed under any other unit once it is set. `unit=` selects what to
    COMPUTE; the module constant decides what may enter an e-process. 6h's
    construction and its reason: registering a unit after seeing a p-value would
    void the guarantee the e-value is supposed to provide.
    """
    if REGISTERED_EXCHANGEABLE_UNIT is None:
        raise PatchingRefused(
            "REGISTERED_EXCHANGEABLE_UNIT is None. Which unit may enter an "
            "e-process is a scientific decision of the same class as CLAIM-C's "
            "criterion, and the measurement that informs it is in this module's "
            "docstring: the per-prompt unit holds at 0.007-0.019 across the "
            "shared-factor range where the per-ablation-point unit reaches "
            "0.357. Register it before adjudicating, not after.")
    if result.get("unit_computed") != REGISTERED_EXCHANGEABLE_UNIT:
        raise PatchingRefused(
            f"this result was computed under unit="
            f"{result.get('unit_computed')!r}; the registered unit is "
            f"{REGISTERED_EXCHANGEABLE_UNIT!r}")

    result = dict(result)
    result["adjudication"] = None
    if result.get("p_value") is None:
        return result

    from core.adjudication import adjudicate_if_registered
    sat = result.get("saturation_diagnostic") or {}
    shared = result.get("shared_prompt_factor") or {}
    result["adjudication"] = adjudicate_if_registered(
        "P-AB1", result["p_value"],
        artifact_hashes=tuple(artifact_hashes), run_manifest=run_manifest,
        test_name=(
            f"growth exponent of trajectory divergence in remaining depth, "
            f"fitted over a common window of {result.get('window')} layers; "
            f"paired against a matched random-direction ablation of equal "
            f"magnitude at the same layer; null = sign flip of the pair labels "
            f"with the {REGISTERED_EXCHANGEABLE_UNIT} as the unit; one-sided "
            f"'{P_AB1_ALTERNATIVE}'"),
        notes=(
            f"verdict={result['verdict']} "
            f"p_reciprocal={result['p_reciprocal']:.4f} (RECAPTURES input only, "
            f"NOT calibrated into E) "
            f"paired window sensitivity {sat.get('paired_mean')} -- negative "
            f"attenuates PROPAGATES and inflates RECAPTURES; "
            f"between-prompt share of the difference "
            f"{shared.get('shared_share_estimate')} -- a shared factor VARYING "
            f"between prompts. A FIXED offset between the real and control "
            f"direction populations is invisible to this estimate and to both "
            f"units, and takes either to 1.000 at one sd"),
        adjudications_dir=adjudications_dir,
    )
    return result
