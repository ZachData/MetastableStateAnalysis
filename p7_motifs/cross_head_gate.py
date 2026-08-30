"""
p7_motifs/cross_head_gate.py — P-I3's gate (the cross-head entry, as a
motif-rate contrast between induction heads and control heads matched on the
behavioural induction score).

    P-I3  Across heads at a fixed checkpoint, `relay` strength correlates with
          the behavioural induction score -- AND does not among non-induction
          heads.
    H0    Relay strength is unrelated to the behavioural induction
          classification.
    falsifier
          Non-induction heads carry the motif at the same rate => the motif is
          a property of the activations, not of the classification.

`PREDICTIONS.md` calls this "the prediction that can kill the bridge, and it is
stated in the direction that would hurt", and says of its control arm that it
"is mandatory, not optional: reporting the motif rate only among induction
heads would read as confirmation no matter what the number was".

BUILT IN THE ORDER `claims/EVALUABILITY.md` PRESCRIBES

That document's rule for every row that names a matched control: *"compute the
attainable floor, name what the statistic degenerates on, check what the
measurement grid contributes, and only then build the control."* `P-AB1`
(`POPPER_PLAN.md` 6q) was the first row built that way and `CLAIM-B`'s grid
(6r) reused the order; this is the third, and as on both of those the steps
before the control changed the design. What follows is what each returned.

1. THE ATTAINABLE FLOOR, AND WHY THE REGISTERED NULL HAS NO USABLE ONE
-----------------------------------------------------------------------
The registry's `null_construction` reads *"Correlation with a REQUIRED control
arm over non-induction heads ... Permutation over the head classification."*
Permuting the classification means drawing which `k` of the `n` heads are
labelled induction. The group is `C(n, k)` and its floor is `1/C(n, k)` --
9.2e-17 at 384 heads and 8 induction heads, which looks like the most
resolution any design in this project has had.

**It is not a floor of anything, because the permuted labels are impossible
labels.** An induction head is one whose behavioural induction score clears a
cutoff, so the classification is a THRESHOLD ON THE VARIABLE THE STATISTIC
CORRELATES AGAINST. Exactly one of the `C(n, k)` assignments the null draws
from is a threshold on the score -- the observed one -- so the null family is
1 possible configuration and 1.09e16 impossible ones. That is not an argument
about power; it is measurable in one line, and the record measures its
consequence in `registered_null`.`spread_under_permutation`: with the
classification a cutoff on the score, the observed induction group's
behavioural-score spread is about a THIRD of a permuted group's, and only a
fraction of a percent of permuted draws are as tightly clustered. A fraction,
and not none -- a permuted group CAN be tightly clustered somewhere else, and
an earlier draft of this docstring claimed "smaller than all 20,000 draws" on
the strength of one draw that happened to show it. The exact statement is the
combinatorial one; the study is its illustration.

What that does to a statistic depends on which statistic, and the record
carries both readings on the same draws
(`registered_null`.`rejection_rates`):

one for the contrast of within-group CORRELATIONS and one for the contrast of
within-group SLOPES. The per-cell rates are proportions over the record's own
replicate count and move a little on regeneration; what the record CHECKS is
that neither statistic's `discrimination` exceeds 0.10 in absolute value, and
that the slope reading's H0 rate stays at or above 0.10 -- **anti-conservative
at twice nominal or worse, while having no power, and the correlation reading
has neither.** Neither can tell the prediction's own effect from its absence.
That is 6o's arm-B discrimination reading -- the difference between the rate on
an input carrying the effect and the rate on one that does not, measured on the
same draws -- reaching a second construction, and it is why nothing below uses
a permutation over the classification.

The floor the design does have is the one the matched sets give it. Each
induction head is compared against `M` control heads matched on its own
behavioural score, and under H0 the induction head's rank among the `M + 1` is
uniform; the sets are disjoint, so the null ENUMERATES in closed form and

    design floor = 1 / (M + 1) ** n_informative_sets

with a set counted informative only when its `M + 1` motif rates are not all
equal -- CLAIM-C's informative-row structure (6l), `P-AB1`'s odd-vs-even
ablation points (6q), and now this, all the same arithmetic on different
groups. There is no second, sampling floor: nothing here is sampled, so 6p's
"take the max of the design floor and the sampling resolution" applies with one
of its two terms absent, and `attainable_floor_report` says so rather than
reporting a draw count that does not exist.

**And the floor is where the tautology gets caught, which is the finding.**
`PREDICTIONS.md`'s second Phase 7 adjudication constraint records the phase's
central methodological danger: the behavioural induction score is *mean
attention on induction pairs* and a motif defined as *an attentive edge on
induction pairs* is the same number, so "correlating them would produce a
beautiful result that means nothing". `P-I1`'s gate could only refuse the
degenerate case and leave the rest "a claim the analyst must make in the
record", because no permutation over a PAIRING detects a pair that is
tautological at every head. Here it is arithmetic. Matching on the behavioural
score is what removes the shared component, and when the classification IS the
thresholded score there is no head with a higher score to match against:

    perfect separation  =>  no induction head can be straddled
                        =>  0 informative sets
                        =>  design floor 1.000

so the design cannot reject on any input whatever, and it says so before a
single motif is counted. 6q's twenty-first lesson -- *a null that leaves the
statistic invariant is not a weak null; it is a floor of 1.000* -- reached by a
third construction, and this time the invariance IS the tautology rather than
merely resembling one.

2. WHAT THE STATISTIC DEGENERATES ON: THE WITHIN-GROUP SPREAD OF THE SCORE
---------------------------------------------------------------------------
6o's rule is that "matched on what" has to name the quantity the statistic
degenerates on. The registered wording's statistic is a within-group
CORRELATION, and a correlation degenerates on the spread of what it correlates
-- which the classification cuts to a thin slice at the top by construction.

Measured on ONE population carrying ONE relation, with no interaction of any
kind (`degeneracy`.`selection_attenuation`, a row per classification size): the
induction arm's correlation runs around a quarter to a third where the control
arm's runs around two thirds, and the record checks that the contrast at the
design's own eight induction heads stays at or below **-0.20**. In a world
where the two groups' relation is literally identical, the registered statistic
reads a contrast several times larger than any effect a real result would
carry. That is not a small bias and it is not in the direction that merely
loses power: it is the FALSIFIER's direction, so the registered statistic tends
to report "the motif is a property of the activations" whenever the
classification is a threshold, which it always is.

The within-group SLOPE is the same quantity with the bias removed -- selection
on a regressor leaves an OLS slope unbiased -- and the record checks that the
two arms' slopes agree to within three standard errors of the induction arm's
own spread. **It is unbiased and unusable**: the induction group's score spread
is what a slope's variance is inversely proportional to, and `slope_sd_ratio`
is checked to stay at or above **5** at eight induction heads, where it in fact
runs into the tens and, at four induction heads, the tens of tens. That is the
same fact the correlation shows as bias, and it is why the registered null's
slope reading is anti-conservative on plain H0 above.

So neither within-group association statistic survives, and both are reported
as diagnostics that reach no ledger (`correlation_contrast_report`). What
replaces them is the quantity the registered H0 and the registered FALSIFIER
both name -- *"relay strength is unrelated to the classification"*, *"carry the
motif at the same rate"* -- which is a LEVEL contrast at matched score. The
prediction's own wording names a third quantity, the within-group correlation,
and one number cannot carry both questions: `P-I1`'s gate reached the same
split from the other side, where it was the falsifier's second half that had to
become a reported precondition rather than a p-value.

3. WHAT THE MEASUREMENT GRID CONTRIBUTES: THE MATCHING, AND ITS OWN OPTIMUM
----------------------------------------------------------------------------
The grid here is the matching -- how many control heads each induction head
gets, and on what key -- and it contributes three things, all arithmetic.

**The straddle, which is not optional.** Matching each induction head to its
`M` NEAREST control heads by score is the obvious construction and it leaks:
the nearest controls are almost all BELOW an induction head in score, so any
curvature in the score-to-motif relation is read as an effect. Measured in
`grid`.`straddle`, a curved score-to-motif relation with no interaction at all
rejects several times over nominal under nearest-neighbour matching and at or
under nominal when each induction head's controls are required to straddle it,
`M/2` below and `M/2` above; the record checks that the discarded matching
still leaks above 0.10 and that the straddled one stays at or below it. The
straddle is also what makes the tautology floor above exact, so it is one
requirement doing two jobs rather than two.

**And the leak is proportional to the tautology, which is the point.**
`grid`.`tautology_leak` sweeps how hard the motif tracks the behavioural score
under a classification that is a cutoff on it: with no relation the discarded
matching is nominal, and it climbs to certainty as the relation strengthens,
while the straddled matching yields **zero** matched sets at every point on
that sweep. PREDICTIONS.md's constraint 2 warns about two variables that are
"the same number wearing a different name"; the curve says the danger is not
identity but ANY tracking, which is a larger family and the one `P-I1`'s gate
could not see.

**The count, whose optimum is not at an extreme, and which excludes the heads
the claim is most about as it grows.** `grid`.`n_controls_frontier` sweeps
`M` = 2, 4, 6, 8, 10 at a flat H0 rate: power against a planted effect rises
from two controls to four and then stops, and the record checks that no larger
`M` beats four by more than 0.05. What every further control DOES buy is a
smaller analysis -- the informative set count falls across that sweep, and the
heads it drops are the highest-scoring ones, because the top of the ranking has
nothing above it to straddle with. The record checks that the retained
induction heads' mean behavioural-score rank falls monotonically end to end.
**Chasing the arm's numbers moves the analysis down the induction ranking,
away from the heads the prediction is about** -- 6r's twenty-sixth lesson in a
second construction, and the reason `N_CONTROLS_PER_INDUCTION_HEAD` is fixed at
the point where power stops rising rather than at the point where it is
highest.

**The key, which is the one thing here the arithmetic does not decide.** See
`REGISTERED_CONTROL_MATCHING_KEY`.

4. AND ONLY THEN THE CONTROL, WHERE THE CHOICE IS WHAT IT IS MATCHED ON
------------------------------------------------------------------------
The control is the registry's -- non-induction heads -- and what this pass adds
is that they are matched, on the quantity step 2 named, and that the match is
CHECKED rather than assumed (`P-AB1`'s magnitude precedent, 6q, and 6p's `P-S1`
before it: a control that was not matched and nothing looked).

What matching on the score alone does not remove is a confound the literature
guarantees is present: induction heads CLUSTER in a band of layers, so anything
that elevates the motif rate across that band elevates it on the induction
heads and not on their controls. Measured in `limitation`.`layer_band`, with the
induction heads confined to layers 6-11 as the literature reports and a shared
elevation on those layers: a control matched on score alone climbs steadily
with the elevation, and the record checks it exceeds 0.15 at one standard
deviation, while a control drawn from the induction head's own layer is FLAT --
checked to stay at or under 0.15 at the same elevation, and in fact barely off
zero across the whole sweep.

That is 6i's shared-per-unit-factor at 1.00 and 6q's fixed offset at 1.000 in
this design's clothing -- and unlike both of those, it CAN be removed. It is
removed at a price `limitation`.`layer_band_power` carries beside it: roughly a
quarter of the informative sets, a run that emits a p-value on well under
two-thirds of draws instead of all of them, and power roughly halved at both
planted effects. The record checks that the trade IS a trade -- that the layer
key costs power and costs emissions -- rather than letting a free-improvement
reading stand.

**The author registered `"score_and_layer"` on 2026-08-30**, so a result
adjudicated for `P-I3` is one whose control heads came from the induction
head's own layer. `REGISTERED_CONTROL_MATCHING_KEY` carries it and
`adjudicate_p_i3` turns away a result computed under the other key, while
`p_value_p_i3` still computes under either and reports which -- 6h's
construction, the same division `patching_gate.py` makes between what `unit=`
computes and what may be adjudicated.

Every rate quoted above is a field of
`claims/calibration/cross_head_association.json`, named where it is quoted, and
none is inlined from a scratch measurement -- 6m found three stale rates in one
docstring that way, and 6r found two more in a committed record's own prose.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class CrossHeadRefused(Exception):
    """The design cannot support a p-value on this input. Never a failure."""


#: One-sided and fixed in advance. P-I3 predicts induction heads to carry MORE
#: of the motif than controls matched on their behavioural score, so the
#: predicted outcome is the large rank sum. A constant, so the tail cannot be
#: picked after the fact.
P_I3_ALTERNATIVE = "greater"

#: The reciprocal one-sided test. It decides the ACTIVATION_PROPERTY branch and
#: is a stop-rule input only, never a term in any claim's E -- CLAIM-B's and
#: P-AB1's division, and it does real work here for the same reason it does
#: there: the branch a confound can manufacture is the one that reaches no
#: ledger.
P_I3_RECIPROCAL_ALTERNATIVE = "less"

#: PREDICTIONS.md's first Phase 7 adjudication constraint, which is registered
#: rather than chosen here: "Effective n is the number of heads, not the number
#: of edges. Edges within a head are not independent samples." The matched sets
#: make the head the unit by construction -- an edge-level n has no way into
#: the arithmetic -- and the effective n is sharper than the constraint says:
#: it is the number of INDUCTION heads that can be matched, not the number of
#: heads.
REGISTERED_EXCHANGEABLE_UNIT = "head"

#: What the control heads may be matched on. `"score"` matches on the
#: behavioural induction score alone; `"score_and_layer"` additionally requires
#: every control to sit in the induction head's own layer.
CONTROL_MATCHING_KEYS = ("score", "score_and_layer")

#: Which key may enter an e-process. Registered by the author on 2026-08-30
#: after the measurement in section 4 was put to them, and NOT chosen here: the
#: two keys differ on a confound the literature guarantees is present (induction
#: heads cluster in a layer band) and on how much of the design survives, which
#: is a scientific decision of CLAIM-C's criterion's class. `P6-R2`'s `"model"`
#: (6l), `P-AB1`'s `"prompt"` (6q) and CLAIM-B's sweep (6r) are the precedent
#: for both the question and the timing -- safe to take because no p-value on
#: real data exists: `claims/adjudications/` is empty, no motif sweep artifact
#: is in this repository, and every number either key has produced came from
#: synthetic head tables. It is a TRADE and not free, and `limitation` carries
#: both sides. `key=` selects what to COMPUTE; this constant decides what may be
#: adjudicated -- 6h's construction, and its reason: registering after seeing a
#: p-value would void the guarantee.
REGISTERED_CONTROL_MATCHING_KEY: Optional[str] = "score_and_layer"

#: Control heads per induction head. DERIVED rather than placed: power against
#: a planted effect stops rising after four on `grid`.`n_controls_frontier`,
#: while every further control drops another induction head off the top of the
#: score ranking. Even, because the straddle needs M/2 controls below the
#: induction head's score and M/2 above.
N_CONTROLS_PER_INDUCTION_HEAD = 4

#: PREDICTIONS.md's second Phase 7 adjudication constraint: which of the three
#: independence sources carries the association. Required, never defaulted.
INDEPENDENCE_SOURCES = ("two_stage", "force_channel", "particle_event")

#: The smallest number of matched sets that can express anything: with M
#: controls each the floor is (M+1)^-n, so this is arithmetic on the design and
#: not a placed threshold. Reported by `attainable_floor_report`, which
#: computes it rather than reading it.
_MIN_SETS_SEARCH = 64


# ---------------------------------------------------------------------------
# What the registered null does, recorded as a computation rather than a claim
# ---------------------------------------------------------------------------

def registered_null_invariance_report(n_heads: int, n_induction: int) -> dict:
    """
    The registered null's group, and how much of it is a possible
    classification.

    Recorded as arithmetic the caller can run, because 6p's finding is that a
    floor read off a group size rather than off the design is the commonest
    defect these passes found and the cheapest to check. This one needs no
    data: an induction head is one whose behavioural induction score clears a
    cutoff, so of the `C(n, k)` label assignments the registered null draws
    from, exactly ONE is a threshold on that score.
    """
    from math import comb
    n, k = int(n_heads), int(n_induction)
    if k <= 0 or k >= n:
        raise CrossHeadRefused(
            f"{k} induction heads of {n}: the classification has one empty "
            f"side and there is no contrast to draw")
    size = comb(n, k)
    return {
        "reading": "draw which k of the n heads are labelled induction",
        "group_size": int(size),
        "nominal_floor": 1.0 / size,
        "assignments_that_are_a_threshold_on_the_score": 1,
        "possible_fraction_of_the_group": 1.0 / size,
        "invalid_because": (
            "the classification IS a cutoff on the behavioural induction "
            "score, so every draw but the observed one contradicts the "
            "definition of the label being permuted. The permuted groups carry "
            "the full score spread where the observed induction group carries "
            "a thin slice at the top, and every within-group association "
            "statistic is a function of that spread"),
        "measured_consequence": (
            "see claims/calibration/cross_head_association.json, "
            "registered_null.rejection_rates: neither the correlation contrast "
            "nor the slope contrast discriminates a genuine effect from its "
            "absence, and the slope contrast is anti-conservative on plain H0 "
            "besides. The record carries both and checks that the "
            "discrimination stays inside 0.10 either way"),
        "reading_used_instead": (
            "compare each induction head against control heads MATCHED on its "
            "own behavioural score, and permute the induction label within a "
            "matched set only. The score is then on both sides of every draw, "
            "which is what the registered null could not arrange"),
    }


# ---------------------------------------------------------------------------
# Separation: decidable before a single motif is counted
# ---------------------------------------------------------------------------

def separation_report(behavioral_score: np.ndarray,
                      is_induction: np.ndarray) -> dict:
    """
    How far the classification is from being a cutoff on the score.

    This reads the two input vectors and nothing else -- no motif rates, no
    controls, no alpha -- so it is decidable before a checkpoint's edges are
    counted, which is where a requirement on a pilot belongs. 6o's refusal has
    the same posture and the same reason.

    `overlap` is the count of control heads scoring above the lowest induction
    head. Zero means the classification is exactly a threshold on the score,
    the two arms cannot be matched, and P-I3 is not adjudicable on this input
    -- not because the estimator is weak but because the prediction's two
    variables are then one variable read twice, which is PREDICTIONS.md's
    Phase 7 adjudication constraint 2.
    """
    b = np.asarray(behavioral_score, dtype=np.float64)
    lab = np.asarray(is_induction, dtype=bool)
    if lab.sum() == 0 or (~lab).sum() == 0:
        raise CrossHeadRefused(
            f"{int(lab.sum())} induction and {int((~lab).sum())} control heads: "
            f"one arm is empty and PREDICTIONS.md makes the control arm "
            f"mandatory rather than optional")
    lo = b[lab].min()
    hi = b[lab].max()
    above = int((b[~lab] > lo).sum())
    inside = int(((b[~lab] > lo) & (b[~lab] < hi)).sum())
    # How many induction heads have ANY control head above them in score. This
    # is the quantity the matching needs, and it is not the same as `above`:
    # one very high-scoring control head makes `above` large while straddling
    # only the induction heads below it.
    straddleable = int(sum(1 for v in b[lab] if (b[~lab] > v).any()))
    return {
        "n_heads": int(b.size),
        "n_induction": int(lab.sum()),
        "n_control": int((~lab).sum()),
        "lowest_induction_score": float(lo),
        "control_heads_above_it": above,
        "control_heads_inside_the_induction_range": inside,
        "induction_heads_with_a_control_above_them": straddleable,
        "perfectly_separated": bool(above == 0),
        "_note": (
            "perfectly_separated means the classification is a cutoff on the "
            "score itself. No control can then be matched above an induction "
            "head, no matched set is informative, and the design floor is "
            "1.000 -- the tautology, caught by arithmetic rather than asserted"),
    }


# ---------------------------------------------------------------------------
# The control: matched sets, straddled, and checked rather than assumed
# ---------------------------------------------------------------------------

def matched_sets(behavioral_score: np.ndarray,
                 is_induction: np.ndarray,
                 layer: Optional[np.ndarray] = None,
                 *,
                 key: str = "score",
                 n_controls: int = N_CONTROLS_PER_INDUCTION_HEAD) -> dict:
    """
    One matched set per induction head: itself plus `n_controls` control heads
    matched on its behavioural induction score, half below it and half above.

    THE STRADDLE IS THE CONSTRUCTION, NOT A REFINEMENT OF IT. Taking the
    `n_controls` NEAREST control heads is the obvious reading and it leaks: an
    induction head sits high in the score ranking, so its nearest controls are
    almost all below it, and any curvature in the score-to-motif relation then
    reads as an effect several times over nominal (`grid`.`straddle`).
    Requiring `n_controls/2`
    below and `n_controls/2` above makes the residual score gap cancel to first
    order, and the same requirement is what turns a threshold classification
    into zero informative sets rather than into a plausible-looking answer.

    An induction head with nothing above it to straddle with is DROPPED rather
    than matched one-sided, and dropping it is enumerated into the floor: it is
    a unit that contributes nothing, which is CLAIM-C's informative rows (6l)
    and P-AB1's even ablation grids (6q) a third time. The drops are not
    arbitrary -- they are the highest-scoring induction heads -- so
    `dropped_score_ranks` is reported beside the sets and the analyst can see
    which part of the ranking the answer is about.

    Matching is greedy in ascending induction score and without replacement, so
    it is deterministic and no control head is used twice.
    """
    if key not in CONTROL_MATCHING_KEYS:
        raise CrossHeadRefused(
            f"key={key!r} is not one of {list(CONTROL_MATCHING_KEYS)}")
    m = int(n_controls)
    if m < 2 or m % 2:
        raise CrossHeadRefused(
            f"n_controls={m}: the straddle needs an even count, at least one "
            f"control below the induction head's score and at least one above")
    b = np.asarray(behavioral_score, dtype=np.float64)
    lab = np.asarray(is_induction, dtype=bool)
    lay = None if layer is None else np.asarray(layer)
    if key == "score_and_layer" and lay is None:
        raise CrossHeadRefused(
            "key='score_and_layer' needs a layer for every head; none was given")

    half = m // 2
    ind = np.flatnonzero(lab)
    available = list(np.flatnonzero(~lab))
    sets: List[Tuple[int, List[int]]] = []
    dropped: List[int] = []
    gaps: List[float] = []
    for i in ind[np.argsort(b[ind], kind="mergesort")]:
        pool = [j for j in available
                if lay is None or key == "score" or lay[j] == lay[i]]
        below = [j for j in pool if b[j] <= b[i]]
        above = [j for j in pool if b[j] > b[i]]
        if len(below) < half or len(above) < half:
            dropped.append(int(i))
            continue
        below = [below[t] for t in
                 np.argsort(np.abs(b[np.array(below)] - b[i]), kind="mergesort")[:half]]
        above = [above[t] for t in
                 np.argsort(np.abs(b[np.array(above)] - b[i]), kind="mergesort")[:half]]
        chosen = below + above
        for j in chosen:
            available.remove(j)
        sets.append((int(i), [int(j) for j in chosen]))
        gaps.append(float(np.max(np.abs(b[np.array(chosen)] - b[i]))))

    order = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort")
    return {
        "key": key,
        "n_controls": m,
        "sets": sets,
        "n_sets": len(sets),
        "dropped": dropped,
        "n_dropped": len(dropped),
        "dropped_score_ranks": [int(order[j]) for j in dropped],
        "retained_score_ranks": [int(order[i]) for i, _ in sets],
        "worst_within_set_score_gap": (max(gaps) if gaps else None),
        "median_within_set_score_gap": (float(np.median(gaps)) if gaps else None),
        "score_sd": float(b.std()),
    }


def matching_report(behavioral_score: np.ndarray, sets, n_controls: int) -> dict:
    """
    Whether the match holds, checked rather than assumed.

    Two things are checked and both are about the SCORES alone, so both are
    decidable before any motif rate is read. First, that every set really
    straddles. Second, that the induction head is not systematically the
    highest scorer inside its own set -- tested with the same exact null the
    arm itself uses, so it places no threshold and needs no simulation. The
    straddle makes gross failure impossible, which is the point: this reports
    what is left rather than re-deriving what the construction already
    guarantees.

    `P-AB1`'s equal-magnitude check is the precedent (6q), and 6p's `P-S1` is
    what happens without one: a null drawn at one arm's configuration and
    applied to the other rejected at 1.000 because nothing checked they matched.
    """
    b = np.asarray(behavioral_score, dtype=np.float64)
    half = int(n_controls) // 2
    straddles = []
    for i, ctrl in sets:
        c = b[np.array(ctrl)]
        straddles.append(bool((c <= b[i]).sum() >= half and (c > b[i]).sum() >= half))
    if not sets:
        return {"n_sets": 0, "all_sets_straddle": True,
                "score_rank_p_two_sided": None,
                "mean_induction_rank_in_set": None,
                "expected_rank_under_exchangeability": (n_controls + 1 + 1) / 2.0}
    arm = exact_rank_arm(b, sets)
    return {
        "n_sets": len(sets),
        "all_sets_straddle": bool(all(straddles)),
        "sets_that_do_not_straddle": int(len(straddles) - sum(straddles)),
        "score_rank_p_two_sided": float(min(1.0, 2 * min(arm["p_greater"],
                                                         arm["p_less"]))),
        "mean_induction_rank_in_set": arm["mean_rank"],
        "expected_rank_under_exchangeability": (int(n_controls) + 2) / 2.0,
        "_note": (
            "score_rank_p_two_sided tests the SCORES with the arm's own exact "
            "null. Small means the induction head sits systematically high (or "
            "low) inside its own matched set, so the match is not one and a "
            "monotone score-to-motif relation would be read as an effect"),
    }


# ---------------------------------------------------------------------------
# The arm: an exact rank test that enumerates, so it has one floor and not two
# ---------------------------------------------------------------------------

def _mid_ranks_doubled(values: np.ndarray) -> np.ndarray:
    """Mid-ranks times two, so ties keep the arithmetic in integers."""
    out = np.empty(values.size, dtype=np.int64)
    for t, u in enumerate(values):
        below = int((values < u).sum())
        tied = int((values == u).sum())
        out[t] = 2 * below + tied + 1
    return out


def exact_rank_arm(values: np.ndarray, sets, alpha: Optional[float] = None) -> dict:
    """
    The induction head's rank among its own matched set, summed over sets, with
    the null enumerated exactly.

    Under H0 the induction label is exchangeable within a matched set, so each
    of the `M + 1` members is equally likely to be the induction head and the
    contributed rank is uniform over that set's own mid-ranks. Sets are
    disjoint, so the sum's null distribution is the convolution of `n_sets`
    distributions with `M + 1` atoms each -- **enumerated, not sampled**, which
    is why this design has one floor rather than 6p's two. Mid-ranks make ties
    exact rather than approximate: a set whose motif rates are all equal
    contributes a point mass and is enumerated out of the floor, not counted
    into it.

    Combining by a RANK rather than by a mean of differences places no
    constant, which is the fifth construction here to escape that way after
    CLAIM-B's change-mass centroid, CLAIM-C's sign concordance, `P-ST1`'s
    sign-of-a-difference and `P-AB1`'s sign sum.
    """
    v = np.asarray(values, dtype=np.float64)
    if not sets:
        raise CrossHeadRefused("no matched sets")
    observed = 0
    atoms: List[np.ndarray] = []
    informative = 0
    for i, ctrl in sets:
        members = np.array([v[i]] + [v[j] for j in ctrl], dtype=np.float64)
        if not np.all(np.isfinite(members)):
            raise CrossHeadRefused(
                "a matched set carries a non-finite motif rate; an undefined "
                "rate is not a zero one and imputing either would be a choice "
                "the caller has to make")
        mr = _mid_ranks_doubled(members)
        observed += int(mr[0])
        atoms.append(mr)
        if mr.min() != mr.max():
            informative += 1

    dist = {0: 1.0}
    for mr in atoms:
        w = 1.0 / mr.size
        nxt: Dict[int, float] = {}
        for s, p in dist.items():
            for r in mr:
                nxt[s + int(r)] = nxt.get(s + int(r), 0.0) + p * w
        dist = nxt
    p_greater = sum(p for s, p in dist.items() if s >= observed)
    p_less = sum(p for s, p in dist.items() if s <= observed)

    floor = 1.0
    for mr in atoms:
        floor *= float((mr == mr.max()).sum()) / mr.size
    n_ctrl = atoms[0].size - 1
    return {
        "n_sets": len(sets),
        "n_informative_sets": informative,
        "n_controls": n_ctrl,
        "observed_rank_sum_doubled": int(observed),
        "mean_rank": observed / (2.0 * len(sets)),
        "p_greater": float(min(1.0, p_greater)),
        "p_less": float(min(1.0, p_less)),
        "p_value": float(min(1.0, p_greater)),
        "p_reciprocal": float(min(1.0, p_less)),
        "design_floor": float(floor),
        "enumerated": True,
        "alpha": (None if alpha is None else float(alpha)),
    }


def attainable_floor_report(n_sets: int, n_informative: int, n_controls: int,
                            alpha: float) -> dict:
    """
    The smallest p the matched sets can express, before any motif rate is read.

    `1/(M+1)**n_informative`, with a set counted informative only when its
    members' motif rates are not all equal -- an all-tied set adds the same
    number to the observation and to every draw, so it is enumerated and never
    counted. That is CLAIM-C's rule (6l) and `P-AB1`'s (6q) reached by a third
    group.

    **Only one floor binds here, and the report says which and why.** 6p's rule
    is that the smallest expressible p is the MAX of the design floor and the
    sampling resolution; this null enumerates, so the second term does not
    exist. Saying that is not decoration: 6i's defect and 6q's were both a
    reported floor smaller than any p the arm could actually express, and a
    report that simply omitted the sampling term would look the same as one
    that had forgotten it.
    """
    m, k, a = int(n_controls), int(n_informative), float(alpha)
    floor = (m + 1.0) ** (-k) if k > 0 else 1.0
    min_sets = None
    for s in range(1, _MIN_SETS_SEARCH):
        if (m + 1.0) ** (-s) <= a:
            min_sets = s
            break
    return {
        "n_sets": int(n_sets),
        "n_informative_sets": k,
        "n_controls": m,
        "alpha": a,
        "design_floor": float(floor),
        "sampling_floor": None,
        "attainable_floor": float(floor),
        "binds": "design",
        "min_informative_sets_for_alpha": min_sets,
        "sufficient": bool(floor <= a),
        "_note": (
            "sampling_floor is None because the null ENUMERATES: the induction "
            "label is uniform over each matched set's M+1 members and the sets "
            "are disjoint, so the p-value is exact and no draw count enters. "
            "6p's max-of-two rule applies with one term absent. A floor of "
            "1.000 means no set is informative -- with a classification that "
            "is a cutoff on the behavioural score, no induction head can be "
            "straddled and that is what the arithmetic returns"),
    }


# ---------------------------------------------------------------------------
# The registered wording's own statistic, reported and never adjudicated
# ---------------------------------------------------------------------------

def correlation_contrast_report(motif: np.ndarray, behavioral: np.ndarray,
                                is_induction: np.ndarray) -> dict:
    """
    The within-group Spearman correlations and OLS slopes, and the reason
    neither enters a p-value.

    `motif_stats.cross_head_association` computes the correlations and this
    keeps reporting them, because they are what P-I3's wording names and a
    reader will look for them. What is new is the attenuation beside them:
    the classification cuts the induction group to a thin slice of the score
    range, so its correlation is attenuated whatever the truth. On ONE
    population carrying ONE relation the record measures the induction arm at a
    small fraction of the control arm's -- a contrast several times larger than
    any effect a real result would carry, in a world with no interaction at
    all, and in the falsifier's direction.

    `induction_score_spread_ratio` is the quantity that drives both the
    correlation's bias and the slope's variance, and it is reported so the
    reader can see how thin the slice was rather than take the word for it.
    """
    x = np.asarray(motif, dtype=np.float64)
    b = np.asarray(behavioral, dtype=np.float64)
    lab = np.asarray(is_induction, dtype=bool)

    def _rank(v):
        # The tie loop runs only when there are ties, rather than once per
        # distinct value: most heads carry no relay at all, so the tied case is
        # the expected one and the untied case must not pay for it.
        o = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=np.float64)
        r[o] = np.arange(v.size, dtype=np.float64)
        s = v[o]
        if s.size > 1 and (s[1:] == s[:-1]).any():
            for u in np.unique(s[:-1][s[1:] == s[:-1]]):
                t = v == u
                r[t] = r[t].mean()
        return r

    def _pair(sel):
        if sel.sum() < 3:
            return float("nan"), float("nan")
        xs, bs = x[sel], b[sel]
        rx, rb = _rank(xs), _rank(bs)
        rho = (float("nan") if rx.std() == 0 or rb.std() == 0
               else float(np.corrcoef(rx, rb)[0, 1]))
        slope = (float("nan") if bs.std() == 0
                 else float(np.cov(bs, xs, bias=True)[0, 1] / bs.var()))
        return rho, slope

    rho_i, slope_i = _pair(lab)
    rho_c, slope_c = _pair(~lab)
    sd_i = float(b[lab].std()) if lab.sum() else float("nan")
    sd_c = float(b[~lab].std()) if (~lab).sum() else float("nan")
    return {
        "spearman_induction_heads": rho_i,
        "spearman_control_heads": rho_c,
        "spearman_contrast": rho_i - rho_c,
        "slope_induction_heads": slope_i,
        "slope_control_heads": slope_c,
        "slope_contrast": slope_i - slope_c,
        "induction_score_sd": sd_i,
        "control_score_sd": sd_c,
        "induction_score_spread_ratio": (sd_i / sd_c if sd_c else float("nan")),
        "mean_motif_rate_induction": (float(x[lab].mean()) if lab.sum()
                                      else float("nan")),
        "mean_motif_rate_control": (float(x[~lab].mean()) if (~lab).sum()
                                    else float("nan")),
        "adjudicated": False,
        "_why_not": (
            "a within-group correlation degenerates on the within-group spread "
            "of what it correlates, and the classification sets that spread. "
            "On one population with one relation the induction arm reads a "
            "small fraction of the control arm's -- a large negative contrast "
            "with no interaction present, in the falsifier's direction. The "
            "slope removes the bias and not the cause: its spread runs an "
            "order of magnitude above the control arm's. Neither is "
            "adjudicable and both are reported. See "
            "claims/calibration/cross_head_association.json, "
            "degeneracy.selection_attenuation"),
    }


def layer_concentration_diagnostic(layer: np.ndarray,
                                   is_induction: np.ndarray) -> dict:
    """
    How concentrated the induction heads are in a band of layers, which is the
    confound `key="score"` does not remove.

    Reported on every record and never used to correct anything. It catches
    concentration; it does not catch whether the band carries an elevation, and
    nothing in one checkpoint's head table can -- which is exactly why the
    matching key is a registered decision rather than a diagnostic threshold.
    """
    lay = np.asarray(layer)
    lab = np.asarray(is_induction, dtype=bool)
    if lab.sum() == 0:
        return {"n_layers_represented": 0, "share_in_the_modal_layer": None}
    vals, counts = np.unique(lay[lab], return_counts=True)
    return {
        "n_layers_represented": int(vals.size),
        "n_layers_total": int(np.unique(lay).size),
        "share_in_the_modal_layer": float(counts.max() / counts.sum()),
        "layer_span": [int(vals.min()), int(vals.max())],
        "_note": (
            "induction heads cluster in a band of layers, which the literature "
            "reports and this measures. A shared elevation across that band is "
            "invisible to a control matched on score alone and is removed by "
            "matching controls within the layer, at a cost in informative sets "
            "and in power. Both sides are in "
            "claims/calibration/cross_head_association.json, limitation. See "
            "REGISTERED_CONTROL_MATCHING_KEY"),
    }


# ---------------------------------------------------------------------------
# The stop rule
# ---------------------------------------------------------------------------

def gate_verdict(p_greater: Optional[float], p_less: Optional[float],
                 alpha: Optional[float] = None) -> dict:
    """
    Three-way, and only one branch is a falsification -- CLAIM-B's, CLAIM-C's
    and P-AB1's shape, for 6k's reason: the registered falsifier ("non-induction
    heads carry the motif at the SAME rate") describes the null, and an
    e-process records insufficient evidence rather than a null accepted.
    """
    a = _alpha() if alpha is None else float(alpha)
    if p_greater is None:
        return {"verdict": "INSUFFICIENT", "falsified": False, "alpha": a,
                "reading": "no p-value could be emitted; nothing is adjudicated"}
    if p_greater <= a:
        return {"verdict": "TRACKS_CLASSIFICATION", "falsified": False, "alpha": a,
                "reading": "induction heads carry more of the relay motif than "
                           "control heads matched on their own behavioural "
                           "induction score, so the motif is not the score read "
                           "twice"}
    if p_less is not None and p_less <= a:
        return {"verdict": "ACTIVATION_PROPERTY", "falsified": True, "alpha": a,
                "reading": "control heads matched on score carry MORE of the "
                           "motif than the induction heads they are matched to "
                           "-- the falsifier, positively shown, and the "
                           "direction PREDICTIONS.md names as fatal to the "
                           "bridge for this phenomenon. Read "
                           "layer_concentration first: a shared elevation on "
                           "the layers the induction heads are NOT in reaches "
                           "this branch the same way"}
    return {"verdict": "INSUFFICIENT", "falsified": False, "alpha": a,
            "reading": "neither direction was shown. Nothing is falsified -- "
                       "'non-induction heads carry the motif at the same rate' "
                       "is the NULL, and an e-process records insufficient "
                       "evidence rather than a null accepted"}


def _alpha() -> float:
    from core.adjudication import load_registry
    from core.evalues import DEFAULT_ALPHA
    try:
        return float(load_registry().get("alpha", DEFAULT_ALPHA))
    except Exception:
        return float(DEFAULT_ALPHA)


# ---------------------------------------------------------------------------
# P-I3's gate
# ---------------------------------------------------------------------------

def _as_arrays(motif_rate: Dict[tuple, float],
               behavioral_score: Dict[tuple, float],
               is_induction_head: Dict[tuple, bool]):
    """
    Align the three per-head dicts, dropping heads that any of them is missing
    or that carry an undefined rate.

    An undefined rate is real: `relay_rate_by_head_pair` returns NaN when a
    head pair admitted no compositions at all, which is "undefined" and not
    "zero". Imputing 0.0 counts an impossible composition as a failed one and
    imputing the mean invents a head; both are dropped and counted.
    """
    keys = sorted(set(motif_rate) & set(behavioral_score) & set(is_induction_head))
    kept = [k for k in keys
            if np.isfinite(motif_rate[k]) and np.isfinite(behavioral_score[k])]
    n_undefined = len(keys) - len(kept)
    if not kept:
        raise CrossHeadRefused(
            f"no head is present in all three inputs with a defined rate "
            f"({n_undefined} shared heads had an undefined one)")
    x = np.array([float(motif_rate[k]) for k in kept])
    b = np.array([float(behavioral_score[k]) for k in kept])
    lab = np.array([bool(is_induction_head[k]) for k in kept])
    layer = np.array([int(k[0]) for k in kept])
    return kept, x, b, lab, layer, n_undefined


def p_value_p_i3(motif_rate: Dict[tuple, float],
                 behavioral_score: Dict[tuple, float],
                 is_induction_head: Dict[tuple, bool],
                 independence_source: str,
                 *,
                 key: str = "score",
                 n_controls: int = N_CONTROLS_PER_INDUCTION_HEAD,
                 alpha: Optional[float] = None) -> dict:
    """
    P-I3's p-value.

    All three inputs are keyed by `(layer, head)`, which is what
    `motif_stats.per_head_motif_rate` returns. `independence_source` is
    required and not a keyword with a default, for the reason
    `cross_head_association` already gives: a result that cannot name which of
    the three sources carries its independence from the behavioural score has
    measured one quantity twice.

    Refuses -- `p_value` None with a `reason` -- rather than returning a number
    the design cannot support.
    """
    a = _alpha() if alpha is None else float(alpha)
    out: dict = {
        "prediction_id": "P-I3",
        "claim": "H-BRIDGE",
        "independence_source": independence_source,
        "matching_key_computed": key,
        "registered_matching_key": REGISTERED_CONTROL_MATCHING_KEY,
        "exchangeable_unit": REGISTERED_EXCHANGEABLE_UNIT,
        "n_controls": int(n_controls),
        "registered_null_reading": None,
        "separation": None,
        "matched_sets": None,
        "matching": None,
        "floor": None,
        "correlation_contrast": None,
        "layer_concentration": None,
        "arm": None,
        "n_undefined_dropped": None,
        "p_value": None,
        "p_reciprocal": None,
        "reason": None,
    }
    try:
        if independence_source not in INDEPENDENCE_SOURCES:
            raise CrossHeadRefused(
                f"independence_source must be one of "
                f"{list(INDEPENDENCE_SOURCES)}; got {independence_source!r}. A "
                f"P-I3 result that cannot name the source of its independence "
                f"from the behavioural score has measured one quantity twice "
                f"-- PREDICTIONS.md, Phase 7 adjudication constraint 2")

        keys, x, b, lab, layer, n_undef = _as_arrays(
            motif_rate, behavioral_score, is_induction_head)
        out["n_undefined_dropped"] = n_undef
        out["separation"] = separation_report(b, lab)
        out["registered_null_reading"] = registered_null_invariance_report(
            out["separation"]["n_heads"], out["separation"]["n_induction"])
        out["correlation_contrast"] = correlation_contrast_report(x, b, lab)
        out["layer_concentration"] = layer_concentration_diagnostic(layer, lab)

        ms = matched_sets(b, lab, layer, key=key, n_controls=n_controls)
        # Filled in before any refusal can fire: a refused record that still
        # says how much of the design survived is worth more than one that says
        # only that it refused.
        out["matched_sets"] = {k: v for k, v in ms.items() if k != "sets"}
        out["matching"] = matching_report(b, ms["sets"], n_controls)

        if not ms["sets"]:
            out["floor"] = attainable_floor_report(0, 0, n_controls, a)
            raise CrossHeadRefused(
                f"no induction head could be matched: {ms['n_dropped']} of "
                f"{int(lab.sum())} had no control head above them in "
                f"behavioural induction score. That is what a classification "
                f"defined as a cutoff on that score looks like -- the two "
                f"variables are one variable read twice, the design floor is "
                f"1.000, and no input whatever could reject. PREDICTIONS.md's "
                f"Phase 7 adjudication constraint 2, as arithmetic rather than "
                f"as a caution")

        arm = exact_rank_arm(x, ms["sets"], a)
        out["floor"] = attainable_floor_report(
            arm["n_sets"], arm["n_informative_sets"], arm["n_controls"], a)
        if not out["floor"]["sufficient"]:
            raise CrossHeadRefused(
                f"the design cannot express a p at alpha={a}: "
                f"{arm['n_informative_sets']} informative matched sets of "
                f"{arm['n_controls']} controls floor it at "
                f"{out['floor']['attainable_floor']:.3g}, and "
                f"{out['floor']['min_informative_sets_for_alpha']} are needed. "
                f"A floor is a claim about the design, not about the call")
        if not out["matching"]["all_sets_straddle"]:
            raise CrossHeadRefused(
                f"{out['matching']['sets_that_do_not_straddle']} matched sets "
                f"do not straddle their induction head's score. A one-sided "
                f"match reads any curvature in the score-to-motif relation as "
                f"an effect, several times over nominal")
        if out["matching"]["score_rank_p_two_sided"] is not None and \
                out["matching"]["score_rank_p_two_sided"] <= a:
            raise CrossHeadRefused(
                f"the induction head's behavioural score is systematically "
                f"extreme inside its own matched set "
                f"(p={out['matching']['score_rank_p_two_sided']:.4g} on the "
                f"arm's own exact null, mean rank "
                f"{out['matching']['mean_induction_rank_in_set']:.2f} against "
                f"{out['matching']['expected_rank_under_exchangeability']:.2f}). "
                f"The controls are then not matched on the quantity the "
                f"statistic degenerates on, and a monotone score-to-motif "
                f"relation would be read as an effect")

        out["arm"] = arm
        out["p_value"] = arm["p_value"]
        out["p_reciprocal"] = arm["p_reciprocal"]
    except CrossHeadRefused as exc:
        out["reason"] = str(exc)
        out.update(gate_verdict(None, None, a))
        return out

    out.update(gate_verdict(out["p_value"], out["p_reciprocal"], a))
    return out


def adjudicate_p_i3(result: dict,
                    *,
                    artifact_hashes: Sequence[str] = (),
                    run_manifest: Optional[dict] = None,
                    adjudicate: bool = False,
                    adjudications_dir=None) -> dict:
    """
    `p_value_p_i3`'s result plus, optionally, an entry in the falsification
    ledger.

    Refuses while `REGISTERED_CONTROL_MATCHING_KEY` is `None`, and refuses a
    result computed under any other key once it is set. `key=` selects what to
    COMPUTE; the module constant decides what may enter an e-process -- 6h's
    construction, used a fourth time, and for its reason: registering the
    control's matching key after seeing a p-value would void the guarantee the
    e-value is supposed to provide.

    While no key is registered this refusal is ALSO what keeps a synthetic
    p-value out of P-I3's ledger slot, which 6l recorded as a consequence of
    registering one and 6q found a defect behind. Every test here that asks to
    adjudicate passes an isolated `adjudications_dir`, and one asserts the real
    `claims/adjudications/` directory does not exist afterwards.
    """
    if REGISTERED_CONTROL_MATCHING_KEY is None:
        raise CrossHeadRefused(
            "REGISTERED_CONTROL_MATCHING_KEY is None. What the control heads "
            "may be matched on is a scientific decision of CLAIM-C's "
            "criterion's class: matching on score alone leaves a shared "
            "elevation across the induction heads' own layer band confounding "
            "the result, and matching within the layer removes it at a cost in "
            "informative sets and in power -- both measured in "
            "claims/calibration/cross_head_association.json, limitation. "
            "Register it before adjudicating, not after.")
    if result.get("matching_key_computed") != REGISTERED_CONTROL_MATCHING_KEY:
        raise CrossHeadRefused(
            f"this result was computed under key="
            f"{result.get('matching_key_computed')!r}; the registered key is "
            f"{REGISTERED_CONTROL_MATCHING_KEY!r}")

    result = dict(result)
    result["adjudication"] = None
    if not (adjudicate and result.get("p_value") is not None):
        return result

    from core.adjudication import adjudicate_if_registered
    ms = result.get("matched_sets") or {}
    floor = result.get("floor") or {}
    lc = result.get("layer_concentration") or {}
    cc = result.get("correlation_contrast") or {}
    result["adjudication"] = adjudicate_if_registered(
        "P-I3", result["p_value"],
        artifact_hashes=tuple(artifact_hashes), run_manifest=run_manifest,
        test_name=(
            f"relay motif rate of each induction head against "
            f"{result.get('n_controls')} control heads matched on its "
            f"behavioural induction score "
            f"(key={REGISTERED_CONTROL_MATCHING_KEY!r}, straddled); statistic "
            f"= the induction head's rank within its matched set, summed over "
            f"{floor.get('n_informative_sets')} informative sets; null = the "
            f"induction label exchangeable within a set, enumerated exactly; "
            f"one-sided '{P_I3_ALTERNATIVE}'"),
        notes=(
            f"verdict={result['verdict']} "
            f"p_reciprocal={result['p_reciprocal']:.4g} (ACTIVATION_PROPERTY "
            f"input only, NOT calibrated into E) "
            f"independence_source={result.get('independence_source')!r} "
            f"(PREDICTIONS.md Phase 7 constraint 2, the analyst's claim and "
            f"not this gate's) "
            f"{ms.get('n_dropped')} induction heads dropped as unstraddleable, "
            f"at score ranks {ms.get('dropped_score_ranks')} of "
            f"{result.get('separation', {}).get('n_heads')} heads -- the "
            f"analysis is about the retained ones. Induction heads occupy "
            f"{lc.get('n_layers_represented')} of {lc.get('n_layers_total')} "
            f"layers, {lc.get('share_in_the_modal_layer')} of them in one; a "
            f"shared elevation across that band is separated by the "
            f"score_and_layer key only. The within-group correlation contrast "
            f"is {cc.get('spearman_contrast')} and is NOT what this p-value "
            f"tests: it is attenuated by the classification's own cut at "
            f"-0.41 in a world with no interaction"),
        adjudications_dir=adjudications_dir,
    )
    return result
