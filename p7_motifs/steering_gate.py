"""
p7_motifs/steering_gate.py — P-ST1's null construction.

`P-ST1` is H-BRIDGE's cheapest entry and the only registered bridge prediction
where the particle and the standard accounts make INCOMPATIBLE rather than
merely different predictions, which is what makes it the only one that can
genuinely lose. The statement:

    Two steering vectors of EQUAL NORM whose V-eigenbasis decompositions are
    predominantly attractive (U_pos) versus predominantly repulsive (U_neg),
    injected at the same layer, produce OPPOSITE-SIGNED changes in the
    effective rank of the token population at that layer.

Everything below was fixed BEFORE any activation was steered, and is a module
constant rather than a parameter so it cannot be re-made per run.

WHAT THE REGISTERED WORDING FLAGGED, AND THE ONE IT DID NOT

`null_construction` flagged two things as needing pre-registration: how many
pairs, and how "predominantly" is thresholded. It did not flag **alpha, the
injection scale** -- and alpha turns out to decide whether the prediction is
readable at all. Measured before this module existed
(`claims/calibration/steering_sign.json`):

  * below about 0.05 x spread the two arms move effective rank the SAME way in
    every pair. The statistic is not noisy there, it is identically zero:
    per-pair informative rate 0.00 at every fraction from 1e-6 to 1e-2. So the
    tempting alpha-free formulation -- the sign of dER/dalpha at 0+ -- carries
    no information at all under this module's effective-rank mode, and there is
    no limit to take. (Under `normed` there IS a nonzero small-alpha signal,
    and it is worse than nothing: it flips under v -> -v, which is the separate
    reason `normed` is disqualified. See ER_MODE.)
  * above about 0.26 x spread the rank-1 spike n*alpha^2*v v^T dominates the
    Gram matrix and BOTH arms reduce effective rank for ANY direction. The
    per-pair rate falls from 1.000 at 0.24 to 0.025 at 0.26 and 0.000 at 0.30,
    including where the prediction is true by construction.
  * in between there is a PLATEAU, 0.17 to 0.24 x spread, where the per-pair
    rate is 1.000 under the planted H1 and 0.000 under H0 -- and the plateau is
    the SAME four fractions whether the cloud's own mean offset is 0, 2 or 5
    spreads, which is what scaling alpha by a derived quantity buys. It is
    narrower than a decade and both edges are sharp, so the profile is reported
    with every result rather than trusted from the calibration.

So `ALPHA_SPREAD_FRACTION = 0.2` is **`placed`**, in the sense Phase 7
adjudication constraint 4 fixes: a threshold is labelled placed until it is
derived from an observed distribution. What is NOT placed is the scale it
multiplies -- the population's own RMS deviation from its mean -- so the window
travels with the data instead of with the units. Every record carries the
alpha-profile as a diagnostic that enters no p-value, which is also what the
falsifier's second clause ("the effect tracks ||s||") deserves: at matched norm
||s|| cannot vary between the arms, so that clause is designed out rather than
tested, and the profile is where a reader sees the norm dependence it names.

"PREDOMINANTLY" IS NOT THRESHOLDED, IT IS REMOVED

The registered wording asks for vectors "predominantly" in one subspace, which
is a magnitude cut with as many values as there are cuts. Instead each arm is
drawn UNIFORMLY FROM THE SUBSPACE ITSELF -- 100% by construction, not 60% or
80% by threshold. That is the same ordinal escape CLAIM-C's sign-concordance
and CLAIM-B's change-mass centroid took, and it removes one of the two
constants the registry flagged.

STEERING IS A PURE MEAN EFFECT, AND THAT DECIDED THE OTHER OPEN QUESTION

Adding alpha*v to every token is exactly a shift of the population mean, so
re-centring after injection annihilates the intervention: ER(X' - mean X') =
ER(X - mean X) identically. That is algebra, not simulation, and it has a
consequence the registered wording could not have anticipated -- the
pre-existing mean offset of the token cloud competes directly with the injected
one. Measured per-pair P(D = +2), H1 against H0:

  | cloud mean norm | H1 | H0 |
  |---|---|---|
  | 0 x spread | 0.970 | 0.010 |
  | 2 x spread | 0.230 | 0.180 |
  | 5 x spread | 0.110 | **0.250** |

Each row is that configuration at ITS OWN best alpha, not at the one the
debiased design uses -- comparing them at a shared alpha would flatter the
choice actually made, since the readable window moves when the baseline mean is
left in. At five spreads the undebiased design rejects MORE often under H0 than
under H1.

A real residual stream sits at the bottom of that table, so the design as
literally worded has NO POWER on one. Removing the BASELINE mean before
injecting -- keeping the injected offset, dropping the pre-existing one --
restores it: 1.000 against 0.000 at every offset measured. That choice was
put to the author before this module was written and is recorded as
`DEBIAS_BASELINE_MEAN`. Its cost is stated rather than hidden: the criterion is
then about the injected direction relative to the CENTRED population, which is
a narrower object than "the effective rank of the token population".

THE STATISTIC, AND WHY IT IS ANTISYMMETRIC

Per pair, D = sign(dER_neg) - sign(dER_pos), in {-2, -1, 0, 1, 2}. H1 predicts
D = +2 (attractive reduces, repulsive raises); the statistic is sum(D) over
pairs, one-sided `greater`. Only signs enter, so there is no magnitude cut
anywhere in the criterion.

Swapping which arm is called attractive negates D exactly, which is what gives
the registered null -- "the same injection procedure with the decomposition
label permuted across pairs" -- a closed form: the null distribution of
sum(+/- D_i) follows by convolution, exactly rather than by sampling, at any
pair count. That null is computed and reported. It is not the one adjudicated,
for the reason in the next section.

THE REGISTERED NULL DOES NOT HOLD, AND WHAT REPLACES IT

The registry says the null is "the same injection procedure with the
decomposition label permuted across pairs". Measured, it is anticonservative,
and the inflation GROWS WITH THE PAIR COUNT: every pair at one layer sees the
same tokens and the same two subspaces, so a chance tilt of the cloud toward
one of them moves every pair together. More pairs shrink the permutation null's
spread like sqrt(m) and leave the shared tilt untouched. Rejection rate under a
noisy H0 at alpha = 0.05, conditional on the gate emitting:

  | pairs | 8 | 24 | 40 | 150 |
  |---|---|---|---|---|
  | label permutation, weakly concentrated H0 | 0.000 | 0.031 | 0.030 | 0.220 |
  | label permutation, slightly concentrated H0 | 0.000 | 0.012 | 0.082 | 0.170 |

That is `status-6.md`'s "49 layers are not 49 independent observations" for the
third time, and it is invisible in the clean regime, where under H0 every pair
is uninformative and the gate REFUSES: the unconditional rate there is 0.000,
and it is 0.000 by refusal rather than by control. Conditioning on emission is
what makes it visible, which is `POPPER_PLAN.md` 6g's lesson exactly.

The first replacement was 6h's construction, arriving for the fourth time:
randomise over SUBSPACES, not over units -- replace the two operator-derived
subspaces with random ones OF THE SAME DIMENSIONS, drawn mutually orthogonal
from one Stiefel draw because the real pair is orthogonal by the projector
build's resolution order and 6h measured the cost of forgetting that at 0.0875
against a nominal 0.05.

IT IS ALSO INVALID, AND THE FAMILY THAT SHOWS IT IS THE REALISTIC ONE

Matching the dimensions holds fixed everything the statistic could read off
dimension. It does not hold fixed how much of the population each subspace
CONTAINS -- and dER is driven by exactly that. Injecting along a direction the
cloud already occupies reinforces a large Gram eigenvalue and lowers effective
rank; injecting along one it does not adds a new eigenvalue and raises it. A
random subspace of dimension k captures k/d of the population's energy in
expectation; U_pos and U_neg are cut from the model's own OV eigenstructure and
the residual stream is not orthogonal to either, so both capture MORE than
that. Compared against random pairs, such a pair is unusual whichever arm is
called attractive -- and the sign of the observed difference is then whichever
way this layer's realized asymmetry happens to fall.

Measured on an H0 family in which BOTH arms are occupied above chance and the
two are statistically identical by construction -- so the correct verdict is
INSUFFICIENT, and P(TRACKS) must equal P(INVERTS) exactly -- it is
anticonservative, the inflation GROWS with the pair count, and the adjudicated
null below is at or under nominal on the identical draws. The rates are in
`claims/calibration/steering_sign.json` and are deliberately not repeated here:
a rate inlined in a docstring is a rate that goes stale silently, which is what
the artifact's sha256 pinning exists to prevent and what prose cannot do. The
first version of this paragraph quoted a sweep at a different geometry than the
one the calibration measures, which is the whole argument in miniature.

The calibration that shipped with 6k could not see it: all three of its H0
families put the cloud in a subspace ORTHOGONAL to both arms, which is the one
case where a matched-dimension random pair IS exchangeable with the observed
one. An H0 family that cannot express the failure is 6h's audit arm incapable
of failing, one level up. `check_record` now fails if that family is absent,
and fails again if the retired null stops coming back anticonservative.

WHAT IS ADJUDICATED: RANDOMISE THE SPLIT, NOT THE SUBSPACES

The diagnosis names the fix. The old null randomised the union and the split
TOGETHER, so it rejected on either -- and "this pair of subspaces is unlike a
random pair" is a statement about the union, which is not what P-ST1 claims.
The claim is about the LABELLED SPLIT: does calling one of them attractive
predict which way effective rank moves? So hold the union fixed and randomise
only the split. The null draws a uniformly random k_pos-dimensional subspace of
span(U_pos + U_neg) and takes its orthogonal complement WITHIN that union as
the other arm. Every property of the pair as a pair -- its dimensions, its
orthogonality, its occupancy, its whole spectral relationship to this layer's
cloud -- is held exactly fixed, and the observed split is one point of the same
Grassmannian the null draws from, so under H0 it is exchangeable with them by
construction rather than by measurement.

Measured on the same families and the same draws, it is at or below nominal
everywhere the retired one is not. It costs power, and the cost is stated
rather than hidden: as dim U_pos grows past the dimension the population
occupies, the two nulls' power separates, and `claims/audits/p_st1_dry_run.json`
carries the whole-gate version of that against the precondition ratio. Power
lost that way was never power about the decomposition -- it was the union's
unusualness being read as the split's.

This is `POPPER_PLAN.md` 6h's question -- what is being randomised? -- arriving
for the fifth time, and the first time the answer is to randomise LESS. 6h
moved P6-R2 from units to subspaces because units were too coarse; here
subspaces are too coarse, and the exchangeable object is the assignment.

BOTH RETIRED NULLS ARE STILL COMPUTED AND REPORTED beside every result, never
adjudicated: the registered label permutation, and 6k's matched-dimension pair.
Keeping them visible is what lets a reader see the size of the difference
between the null a wording names and the one that holds, rather than taking
this module's word for it -- and this pass is the second time that difference
turned out to be large.

THE OCCUPANCY OF EACH ARM IS REPORTED, AND IT COSTS NOTHING TO COMPUTE

`occupancy_pos` and `occupancy_neg` are the share of the centred population's
energy inside each arm, divided by the k/d a random subspace of that dimension
would capture -- 6h's chance normalization, applied to the population instead
of to a single vector. They need no injection and no null: the pilot can read
them off the activations and the two projectors before spending a sweep. Two
things to read them for. Both near 1 means neither arm is where the cloud
lives, the per-pair statistic is near-degenerate, and the gate will refuse or
return INSUFFICIENT. A large asymmetry between them is what a TRACKS verdict
is made of, so a reader who wants to know whether the verdict has a
non-particle explanation should look there first.

THE FLOOR, AND WHY REPLACING THE NULL REMOVED IT

Under the REGISTERED permutation the floor is not 2/(2^m + 1). A pair whose two
arms move effective rank the same way contributes D = 0, and a zero contributes
identically to the observed sum and to every null pattern, so with k of m pairs
informative the best attainable p is

    (2^(m - k) + 1) / (2^m + 1)     ~=  2^-k

-- set by the INFORMATIVE pairs and not by the pairs drawn. Five is the first k
that clears alpha = 0.05, at every m, so a hundred pairs at a 2% informative
rate buy two informative pairs and a best possible p of 0.25.

Neither subspace null has that property. The draw-count floor of both is
1/(draws + 1), fixed by how many null draws are taken and independent of the
data, so replacing the null removed a power requirement the registered design
could not meet: if re-splitting this very union essentially never informs, then
the operator's own split doing so is exactly the surprise the claim is about,
and a single informative pair can carry it.

BUT THE DRAW-COUNT FLOOR IS NOT THE ATTAINABLE ONE, and reporting it as though
it were was a defect this module carried until a dry run looked
(`tools/dry_run_p_st1.py`, `POPPER_PLAN.md` 6m). sum(D) cannot exceed 2m, so
the smallest p a run can express is what an observation of 2m would receive --
and every null re-split that already reaches 2m ties it. On a union the cloud
occupies, re-splits inform often, and the two floors part company -- at one pair
by an order of magnitude. The conditional in the previous paragraph is the whole
content of it, and the measured version is in
`claims/audits/p_st1_dry_run.json` rather than here.

So the gate computes the attainable floor from the null it already has, in both
directions, and REFUSES when neither tail can reach alpha -- the design was
then going to return INSUFFICIENT whatever the statistic came to, which on an
entry whose value is that it can lose reads as a loss. 2m is an upper bound on
the observation rather than an attainable value, so the floor computed at it is
a lower bound on what the run can express and the refusal can never turn away a
result that would have cleared alpha. It is CLAIM-C's informative-row refusal
(6l) arriving here, and it was found the same way: by looking at the output of
a gate run on an input whose answer was already known.

The informative-PAIR floor, (2^(m-k) + 1)/(2^m + 1), is still computed and
reported, because it is the retired permutation's floor and a reader comparing
the arms needs it.

THE PRECONDITION THAT REMAINS

A uniform draw from U_pos carries only dim(occupied)/dim(U_pos) of its energy
into the subspace the cloud actually lives in, so the per-pair informative rate
falls as U_pos grows past the population's own occupied dimension. `POPPER_PLAN.md`
6h measured that U_pos is the UN-shrunk bucket in the projector build's
resolution order, which is the unfavourable side of this. Measured, per-pair
informative rate against dim U_pos / dim occupied: 1.000 at ratio 1, 0.710 at
1.5, 0.320 at 2, 0.030 at 3, 0.005 at 4, 0.000 at 6.

The gate reports `dim_u_pos` and the informative rate in every record, and the
pilot should read them against the population's effective rank BEFORE spending
a sweep -- the same shape of pre-computed requirement as CLAIM-B's 19 control
series and CLAIM-C's 19% dissenting cells.

THE RECIPROCAL TAIL IS MEASURED SEPARATELY AND AT MORE REPLICATES, because it
is the branch that would enter the ledger as a falsification and because fifty
gate runs -- what the 2026-08-25 calibration could afford per cell -- resolve a
rate only to about +/- 0.03, which cannot separate nominal from twice nominal.
`claims/calibration/steering_sign.json` carries a dedicated section for it at a
higher replicate count and at one pair count, which is what buys the
replicates; the main validity table stays at fifty because it sweeps seven H0
and H1 families and two pair counts. Read the artifact for the rates rather
than this docstring: a number inlined here is a number that goes stale
silently, which is what the sha256 pinning exists to prevent for the artifact
and cannot do for prose.

The obvious fix is refused. Drawing from the intersection of U_pos with the
population's occupied subspace would restore the rate and would be circular:
a probe aligned with the cloud by construction concentrates it by construction.

THE VERDICT LATTICE, AND A REGISTERED FALSIFIER THAT CANNOT BE ONE

Three-way, CLAIM-C's and CLAIM-B's shape. TRACKS-DECOMPOSITION on
`p_greater <= alpha`; INVERTS when the reciprocal rejects -- attractive-dominant
steering demonstrably RAISES effective rank and repulsive-dominant lowers it,
a positively shown reversal; INSUFFICIENT otherwise.

The registered falsifier reads *"Both arms move effective rank the same way, or
the effect tracks ||s|| and is insensitive to the decomposition."* Both clauses
are the NULL, not a positively showable alternative, and an e-process records
insufficient evidence and never a null accepted -- so as written P-ST1's
falsifier is not one an e-value can carry. It maps to INSUFFICIENT, which is
recorded and not scored, and INVERTS is the branch that enters the ledger as a
falsification. That is stated here rather than discovered at the moment it
binds.

Only `p_greater` is calibrated into H-BRIDGE's product. `p_reciprocal` is a
verdict input and stays out, since two one-sided tests on one statistic would
double the claim's Type-I rate for free.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The exchangeable unit of the permutation: one matched-norm vector PAIR. The
#: label "attractive" attaches to a vector, so a swap moves that pair's whole
#: contribution and nothing else.
P_ST1_UNIT = "matched-norm vector pair"

#: CLAIM-C's tail convention: the prediction is that D is LARGE, recorded as a
#: constant so it cannot be picked after seeing the data.
ALTERNATIVE = "greater"

#: The reciprocal direction, which separates "the decomposition matters, but
#: backwards" from "nothing was shown". Never calibrated into an e-value.
RECIPROCAL_ALTERNATIVE = "less"

#: Author's call, 2026-08-25. Effective rank is measured on the population with
#: its BASELINE mean removed, and the injected offset kept:
#:     ER(X - xbar)  vs  ER(X - xbar + alpha v)
#: Steering is a pure mean effect, so re-centring AFTER injection would make
#: dER identically zero; not centring at all leaves the pre-existing residual
#: stream offset competing with the injected one, which was measured to remove
#: all power. See the module docstring's table.
DEBIAS_BASELINE_MEAN = True

#: PLACED (Phase 7 adjudication constraint 4). The injection scale as a
#: fraction of the population's RMS deviation from its own mean. The FRACTION
#: is placed; the SCALE it multiplies is derived from the data, which is what
#: makes the readable window travel with the population instead of with the
#: units -- measured, the window sits at the same fraction whether the cloud's
#: own mean offset is 0, 2 or 5 spreads.
#:
#: 0.2 is the centre of a measured PLATEAU, and the first value written here
#: was 0.1 because the plateau was missed on a coarse grid. Per-pair rate at
#: alpha/spread of 0.15, 0.17, 0.20, 0.22, 0.24, 0.26, 0.30: 0.85, 0.96, 1.00,
#: 1.00, 1.00, 0.03, 0.00 (claims/calibration/steering_sign.json; an earlier
#: revision of this comment carried 0.81 at 0.15 and a 0.28 the grid does not
#: contain, which is the reason the rates elsewhere in this module are now
#: pointers rather than digits). A grid of (0.03, 0.1, 0.3) reads 0.1 as the
#: peak because its neighbours are both zero; the plateau is 0.17-0.24 and 0.1
#: sits on the shoulder at a sixth of the informative rate -- which is 29 pairs
#: needed instead of 5. The upper cliff is sharp, so 0.2 is taken as the middle
#: of the plateau rather than its best single point.
ALPHA_SPREAD_FRACTION = 0.2
ALPHA_IS_PLACED = True

#: The alpha-profile reported as a diagnostic beside every result. It enters no
#: p-value: it exists so a reader can see where the readable window sits for
#: THIS population rather than taking the calibration's word for it, and so the
#: falsifier's "the effect tracks ||s||" clause has somewhere to be read off.
ALPHA_PROFILE_FRACTIONS: Tuple[float, ...] = (
    0.03, 0.07, 0.1, 0.15, 0.17, 0.2, 0.24, 0.26, 0.3, 1.0)

#: RAW, and this is structural rather than a preference. `status-1.md` defect
#: D1 is why CLAIM-C reads `effective_rank_normed`, and at alpha = 0.1 x spread
#: the two modes are indistinguishable here -- near-equal informative rates and
#: H0 exactly 0.000 for both -- which is exactly how this nearly shipped wrong.
#: Away from that one point they are not interchangeable:
#:
#:   With the baseline mean removed the centred population has zero mean, so
#:   the first-order term of the Gram perturbation, alpha*(base^T 1 v^T + v 1^T
#:   base), VANISHES IDENTICALLY. dER is then O(alpha^2) and EVEN in v, so the
#:   statistic does not depend on the arbitrary sign of the steering vector.
#:   Measured: sign(dER(v)) == sign(dER(-v)) in 60 of 60 draws at every alpha
#:   from 1e-6 to 0.3 x spread.
#:
#:   L2 row-normalization is not linear, and ||x_i + alpha v|| depends on
#:   x_i . v, which is ODD in v. `normed` therefore reintroduces exactly the
#:   antisymmetry debiasing removed: agreement falls to 0.00 at small alpha,
#:   and it manufactures inversions -- D = -2 in about a fifth of pairs at
#:   alpha <= 1e-4 where raw gives 0%, tabulated per alpha in
#:   claims/calibration/steering_sign.json.
#:
#: A criterion that answers differently for v and -v is not a criterion about a
#: steering DIRECTION, so `normed` is disqualified here for a reason CLAIM-C's
#: does not have to weigh.
ER_MODE = "raw"

#: THE NULL THAT IS ADJUDICATED, and it is not the one the registry names.
#:
#: The registered null -- "the same injection procedure with the decomposition
#: label permuted across pairs" -- was measured and it does not hold. Every
#: pair at one layer sees the SAME tokens and the SAME two subspaces, so a
#: chance tilt of the cloud toward one of them moves every pair together; the
#: permutation treats m pairs as m exchangeable units when they carry far fewer
#: than m independent pieces of information. Measured under a noisy H0 at
#: alpha = 0.05 and conditional on the gate emitting, the rate rises with the
#: pair count and is worst at 150; the numbers are in
#: claims/calibration/steering_sign.json, and an earlier revision of this
#: comment quoted a pair that section no longer contains, which is why they are
#: no longer inlined. That is `status-6.md`'s "49 layers are not 49 independent
#: observations" arriving a third time, and it is invisible in the clean regime
#: because there the gate refuses instead of emitting -- so the rate looks like
#: 0.000 unless it is conditioned on emission, which is POPPER_PLAN.md 6g's
#: lesson exactly.
#:
#: The FIRST replacement was POPPER_PLAN.md 6h's construction, arriving for the
#: fourth time: randomise over SUBSPACES, not over units, by replacing the two
#: operator-derived subspaces with RANDOM ones OF THE SAME DIMENSIONS. It is
#: also invalid, for a reason dimension matching cannot reach. dER is driven by
#: how much of the population each subspace CONTAINS, a random k-dimensional
#: subspace contains k/d of it, and U_pos and U_neg -- cut from the model's own
#: OV eigenstructure, on a residual stream orthogonal to neither -- contain
#: more. Such a pair is unusual against random pairs whichever arm is called
#: attractive, and the sign is then whichever way the layer's realized
#: asymmetry falls. Measured on an H0 family where both arms are occupied above
#: chance and the two are IDENTICAL by construction -- so a label swap is a
#: distributional identity and INSUFFICIENT is the only correct verdict -- it is
#: anticonservative and the inflation grows with the pair count. The rates live
#: in claims/calibration/steering_sign.json, whose sha256 pins them to this
#: module; inlining them here would put a number in prose that nothing can
#: check. 6k's calibration could not see the failure at all, because all three
#: of its H0 families put the cloud in a subspace ORTHOGONAL to both arms --
#: the one case where a matched-dimension random pair IS exchangeable with the
#: observed one.
#:
#: WHAT IS ADJUDICATED holds the union fixed and randomises only the SPLIT.
#: The old null randomised the union and the split together, so it rejected on
#: either, and "this pair is unlike a random pair" is a statement about the
#: union -- which is not what P-ST1 claims. The claim is about the labelled
#: split. So: draw a uniformly random k_pos-dimensional subspace of
#: span(U_pos + U_neg) and take its orthogonal complement WITHIN that union.
#: Dimensions, orthogonality, occupancy and the whole spectral relationship to
#: this layer's cloud are held exactly fixed; the observed split is one point
#: of the Grassmannian the null draws from, so exchangeability under H0 is by
#: construction rather than by measurement. Measured on the same draws, it is
#: at or below nominal everywhere the retired one is not. It costs power, and
#: that is stated rather than hidden: the two nulls' power separates as
#: dim U_pos grows past the dimension the population occupies, and
#: claims/audits/p_st1_dry_run.json carries the whole-gate version of it
#: against that ratio.
#:
#: The floor is 1/(draws + 1) either way, fixed by the draws and independent of
#: the pair count, which is the escape 6h found when P6-R2's floor moved from
#: 0.667 to 0.0005 on this same question of what is randomised. This is its
#: fifth arrival and the first time the answer is to randomise LESS.
NULL_FAMILY = "random re-split of the observed pair's union subspace"

#: 6k's matched-dimension pair, computed and reported beside every result and
#: never adjudicated -- the same standing this module already gives the
#: registered label permutation. Two retired nulls in the record is not
#: clutter: it is the only way a reader sees the size of the difference between
#: a null that was believed and the one that holds, and that difference has now
#: been large twice.
MATCHED_DIMENSION_NULL_DIAGNOSTIC = True

#: Draws of the null. 199 puts the floor at 1/200 = 0.005, well under alpha,
#: and it is enumerated-free: unlike a permutation there is no exhaustive set
#: to under-sample, so drawing is the only option and more draws only sharpen.
N_SUBSPACE_DRAWS = 199

#: The registered permutation null is still COMPUTED and reported beside the
#: result, never adjudicated. Keeping it visible is what lets a reader see the
#: size of the difference between the null the registry names and the one that
#: holds, rather than taking this module's word for it.
REPORT_LABEL_PERMUTATION_DIAGNOSTIC = True

#: Smallest number of INFORMATIVE pairs whose perfect result clears alpha=0.05.
#: Derived, not placed: 2^-4 = 0.0625 > 0.05 >= 0.03125 = 2^-5. Recomputed from
#: alpha by `min_informative_pairs`; this is the value at the registry's alpha
#: and exists so the number in the docstring is checkable.
MIN_INFORMATIVE_PAIRS_AT_5PCT = 5

_SEED = 20260825


# ---------------------------------------------------------------------------
# Subspaces
# ---------------------------------------------------------------------------

def subspace_rank(U) -> int:
    """
    The dimension of the subspace, for either shape a caller can hold.

    ||B||_F^2 is the rank for BOTH: for an orthonormal (d, r) basis it is
    trace(B^T B) = r, and for a symmetric idempotent (d, d) projector it is
    trace(P^2) = trace(P) = rank. Carried in every record because
    `POPPER_PLAN.md` 6h's whole finding is that a statistic over subspaces
    which does not report their dimensions is one step from being read wrong.
    """
    from core.interactions import _as_basis

    B = _as_basis(np.asarray(U, dtype=np.float64),
                  np.asarray(U).shape[0], name="steering subspace")
    return int(round(float(np.sum(B * B))))


def prepared_subspace(U) -> np.ndarray:
    """
    Validate a subspace ONCE and return the array the draws will use.

    `_as_basis` checks the shape and, for a (d, d) projector, that it really is
    symmetric idempotent -- which costs a (d, d) matmul. That is the right
    check and the wrong thing to repeat: a gate run draws thousands of vectors
    from the same two subspaces, and Phase 2 stores its attractive/repulsive
    channels AS (d, d) projectors, so re-validating per draw put ~1 ms on every
    one of them. Validated here once, and the array is handed back unchanged --
    no compact basis is extracted, because `B @ (B.T @ g)` is already the
    projection for both accepted shapes (P @ P @ g = P g for an idempotent
    projector, U U^T g for an orthonormal basis).
    """
    from core.interactions import _as_basis

    U = np.asarray(U, dtype=np.float64)
    return _as_basis(U, U.shape[0], name="steering subspace")


def draw_unit_in_subspace(U, rng: np.random.Generator,
                          prepared: bool = False) -> np.ndarray:
    """
    A unit vector uniform on the subspace's own unit sphere.

    Pass `prepared=True` when `U` already came from `prepared_subspace`, which
    is what the inner loops do. A subspace too degenerate to draw from is a
    refusal rather than a zero vector, which would silently contribute dER = 0
    and be counted as an uninformative pair -- relaxing the very floor the
    uninformative count is used to compute.
    """
    B = np.asarray(U, dtype=np.float64) if prepared else prepared_subspace(U)
    for _ in range(8):
        g = rng.normal(size=B.shape[0])
        y = B @ (B.T @ g)
        n = float(np.linalg.norm(y))
        if n > 1e-12:
            return y / n
    raise ValueError(
        "could not draw a unit vector from this subspace in 8 attempts; its "
        "rank is effectively zero. An empty channel is a real answer (a layer "
        "with no negative real eigenvalue), but it is a refusal here rather "
        "than a pair contributing D = 0, which would be counted as an "
        "uninformative pair and silently relax the attainable floor.")


# ---------------------------------------------------------------------------
# The per-pair statistic
# ---------------------------------------------------------------------------

def population_spread(activations: np.ndarray) -> float:
    """RMS deviation from the population mean -- the scale alpha multiplies."""
    X = np.asarray(activations, dtype=np.float64)
    C = X - X.mean(axis=0, keepdims=True)
    return float(np.sqrt((C ** 2).sum(axis=1).mean()))


def population_mean_ratio(activations: np.ndarray) -> float:
    """
    ||mean token|| / spread. Reported, never adjudicated.

    This is the number that decided `DEBIAS_BASELINE_MEAN`: undebiased, the
    design has no power once it exceeds about 2. With debiasing it no longer
    gates anything, and it is kept in every record because a reader comparing
    this run to the calibration's H0 families needs to know where the run sat.
    """
    X = np.asarray(activations, dtype=np.float64)
    s = population_spread(X)
    return float(np.linalg.norm(X.mean(axis=0)) / s) if s > 0 else float("nan")


def _baseline(activations: np.ndarray) -> np.ndarray:
    X = np.asarray(activations, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True) if DEBIAS_BASELINE_MEAN else X


def _delta_from_base(base: np.ndarray, er0: float, v: np.ndarray,
                     alpha: float, mode: str) -> float:
    """The inner half, with the baseline and its effective rank already in hand."""
    from core.metrics import effective_rank

    moved = base + float(alpha) * np.asarray(v, dtype=np.float64)[None, :]
    return float(effective_rank(moved, mode=mode) - er0)


def delta_effective_rank(activations: np.ndarray, v: np.ndarray,
                         alpha: float, mode: str = ER_MODE) -> float:
    """
    ER(baseline + alpha*v added to every token) - ER(baseline).

    `v` is added to every row: steering translates the whole cloud rigidly,
    which is why the effect is entirely a mean effect and why
    `DEBIAS_BASELINE_MEAN` is a decision rather than a detail.
    """
    from core.metrics import effective_rank

    base = _baseline(activations)
    return _delta_from_base(base, float(effective_rank(base, mode=mode)),
                            v, alpha, mode)


def pair_statistic(activations: np.ndarray, v_pos: np.ndarray, v_neg: np.ndarray,
                   alpha: float, mode: str = ER_MODE,
                   base: Optional[np.ndarray] = None,
                   er0: Optional[float] = None) -> dict:
    """
    One pair's D = sign(dER_neg) - sign(dER_pos), and the two deltas behind it.

    D is exactly negated by swapping which vector is called attractive, which
    is what makes the registered label-permutation null enumerable in closed
    form. The deltas themselves are reported and never adjudicated -- the
    criterion is ordinal, so admitting a magnitude here would introduce the cut
    that drawing from the subspaces outright was chosen to avoid needing.
    """
    from core.metrics import effective_rank

    if base is None:
        base = _baseline(activations)
    if er0 is None:
        er0 = float(effective_rank(base, mode=mode))
    d_pos = _delta_from_base(base, er0, v_pos, alpha, mode)
    d_neg = _delta_from_base(base, er0, v_neg, alpha, mode)
    return {"delta_er_pos": d_pos, "delta_er_neg": d_neg,
            "D": float(np.sign(d_neg) - np.sign(d_pos)),
            "informative": bool(np.sign(d_neg) != np.sign(d_pos))}


def draw_pairs(activations: np.ndarray, u_pos, u_neg, n_pairs: int,
               *, alpha: Optional[float] = None,
               seed: int = _SEED, mode: str = ER_MODE) -> dict:
    """
    Draw `n_pairs` matched-norm pairs at one layer and score each.

    Both arms of a pair get the SAME alpha, so the norms match exactly rather
    than approximately: matching is by construction, not by a tolerance. The
    layer, the population and the injection procedure are shared across the two
    arms of every pair, so the only thing that differs within a pair is which
    subspace the direction came from -- which is the whole content of the
    prediction.
    """
    X = np.asarray(activations, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError(
            f"activations must be (n_tokens, d) with at least 2 tokens; got "
            f"{X.shape}. Effective rank of fewer is not a measurement.")
    spread = population_spread(X)
    if not np.isfinite(spread) or spread <= 0:
        raise ValueError(
            "the token population has zero spread about its mean, so every "
            "token is identical and effective rank is 1 whatever is injected. "
            "Refusing rather than returning D = 0 for every pair.")
    a = ALPHA_SPREAD_FRACTION * spread if alpha is None else float(alpha)

    rng = np.random.default_rng(seed)
    from core.metrics import effective_rank
    base = _baseline(X)
    er0 = float(effective_rank(base, mode=mode))
    bp, bn = prepared_subspace(u_pos), prepared_subspace(u_neg)
    pairs = []
    for _ in range(int(n_pairs)):
        vp = draw_unit_in_subspace(bp, rng, prepared=True)
        vn = draw_unit_in_subspace(bn, rng, prepared=True)
        pairs.append(pair_statistic(X, vp, vn, a, mode, base=base, er0=er0))
    return {
        "pairs": pairs,
        "D": [p["D"] for p in pairs],
        "alpha": float(a),
        "alpha_spread_fraction": float(a / spread),
        "alpha_is_placed": ALPHA_IS_PLACED,
        "population_spread": spread,
        "population_mean_ratio": population_mean_ratio(X),
        "debias_baseline_mean": DEBIAS_BASELINE_MEAN,
        "er_mode": mode,
        "n_tokens": int(X.shape[0]),
        "d_model": int(X.shape[1]),
        "dim_u_pos": subspace_rank(u_pos),
        "dim_u_neg": subspace_rank(u_neg),
        "unit": P_ST1_UNIT,
    }


def alpha_profile(activations: np.ndarray, u_pos, u_neg, n_pairs: int,
                  *, seed: int = _SEED, mode: str = ER_MODE,
                  fractions: Sequence[float] = ALPHA_PROFILE_FRACTIONS) -> List[dict]:
    """
    The informative rate and mean D across the alpha window. DIAGNOSTIC.

    Enters no p-value. It is here because ALPHA_SPREAD_FRACTION is `placed` and
    a placed constant whose profile is not shown is a constant nobody can
    check, and because the falsifier's "the effect tracks ||s||" clause has no
    other place to be read: at matched norm ||s|| cannot vary WITHIN a pair, so
    the norm dependence lives across this profile rather than inside the test.
    """
    spread = population_spread(activations)
    out = []
    for f in fractions:
        drawn = draw_pairs(activations, u_pos, u_neg, n_pairs,
                           alpha=float(f) * spread, seed=seed, mode=mode)
        D = np.asarray(drawn["D"], dtype=np.float64)
        out.append({
            "alpha_spread_fraction": float(f),
            "alpha": float(f) * spread,
            "mean_D": float(D.mean()) if D.size else float("nan"),
            "informative_rate": float((D != 0).mean()) if D.size else float("nan"),
            "frac_predicted": float((D == 2).mean()) if D.size else float("nan"),
            "frac_inverted": float((D == -2).mean()) if D.size else float("nan"),
        })
    return out


# ---------------------------------------------------------------------------
# The null, and the floor that is not 2/(2^m + 1)
# ---------------------------------------------------------------------------

def attainable_floor(n_pairs: int, n_informative: int) -> float:
    """
    The smallest p the exhaustive label-permutation null can express.

    A pair with D = 0 contributes identically to the observed sum and to every
    null pattern, so it cannot separate them: with k informative pairs of m,
    exactly 2^(m-k) patterns tie the observed maximum and

        floor = (2^(m-k) + 1) / (2^m + 1)  ~=  2^-k.

    The floor is therefore a property of the INFORMATIVE pairs and drawing more
    pairs helps only in proportion to the rate at which they inform. Getting
    this wrong in the optimistic direction is the specific error `POPPER_PLAN.md`
    6f names for CLAIM-C at four prompts and 6i names for CLAIM-B's binary
    co-location statistic.
    """
    m, k = int(n_pairs), int(n_informative)
    if m <= 0 or k < 0 or k > m:
        raise ValueError(f"need 0 <= n_informative <= n_pairs; got {k}, {m}")
    return (2.0 ** (m - k) + 1.0) / (2.0 ** m + 1.0)


def min_informative_pairs(alpha: float) -> int:
    """
    Smallest k whose perfect result clears `alpha`, derived from alpha alone.

    Uses the large-m limit 2^-k, which is what the floor converges to and is
    never optimistic: the finite-m floor (2^(m-k)+1)/(2^m+1) exceeds 2^-k for
    every finite m, so a k chosen here is a lower bound on what a real design
    needs rather than a number that might just miss.
    """
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}")
    k = 1
    while 2.0 ** (-k) > a:
        k += 1
        if k > 64:
            break
    return k


def null_sums(D: Sequence[float]) -> np.ndarray:
    """
    Every value sum(D) takes under the 2^m label permutations, ENUMERATED.

    The label swap negates that pair's D exactly, so the null is the set of
    +/- sign patterns applied to the observed D vector. Exact and direct, and
    exponential -- it is the reference `null_distribution` is pinned against,
    not the thing the gate calls.
    """
    d = np.asarray(D, dtype=np.float64)
    m = d.size
    pats = np.array(list(itertools.product((1.0, -1.0), repeat=m)),
                    dtype=np.float64)
    return pats @ d


def null_distribution(D: Sequence[float]) -> Tuple[np.ndarray, List[int]]:
    """
    The same null as (values, exact integer counts), by convolution.

    D takes values in {-2, -1, 0, 1, 2}, so sum(+/- D_i) is an integer in
    [-2m, 2m] and the whole 2^m-point distribution follows from convolving m
    two-point distributions -- O(m^2) instead of O(2^m). Counts are Python
    integers rather than floats because they reach 2^m exactly and a float
    count would start rounding around m = 53, silently, in the direction that
    changes a p-value's last digits.

    This is what removes the enumeration ceiling, and removing it matters:
    the floor below needs about `min_informative_pairs / informative_rate`
    pairs, which at a measured rate near 0.17 is about thirty -- and 2^30
    patterns cannot be enumerated while 30 convolutions are instant. A gate
    that refused above twenty pairs would have refused every design its own
    floor calls for. Same fast-path-plus-pin arrangement as CLAIM-C's
    homogeneity curve (POPPER_PLAN.md 6g item 1), for the same reason: a second
    implementation of the null's arithmetic is a real risk and is pinned
    against the first rather than trusted.
    """
    d = [int(round(float(x))) for x in D]
    if any(abs(x) > 2 for x in d):
        raise ValueError(
            f"D must lie in [-2, 2]; got {sorted(set(d))}. D is a difference "
            f"of two signs and nothing else can produce it.")
    m = len(d)
    span = sum(abs(x) for x in d)
    counts = [0] * (2 * span + 1)
    counts[span] = 1                      # offset: index i means value i - span
    for x in d:
        nxt = [0] * (2 * span + 1)
        for i, c in enumerate(counts):
            if not c:
                continue
            nxt[i + x] += c
            nxt[i - x] += c
        counts = nxt
    values = np.arange(-span, span + 1, dtype=np.float64)
    return values, counts


def p_from_distribution(observed: float, values: np.ndarray,
                        counts: Sequence[int], alternative: str) -> float:
    """
    (n_extreme + 1) / (n_null + 1), formed exactly as `core.nulls.p_from_null`
    forms it, but over a weighted distribution rather than a sample.

    The +1 floor is kept: it is what makes the p valid rather than merely
    unbiased, and dropping it because the null here is exact would let a
    perfect result report p = 0.
    """
    obs = float(observed)
    if alternative == ALTERNATIVE:
        extreme = sum(c for v, c in zip(values, counts) if v >= obs)
    elif alternative == RECIPROCAL_ALTERNATIVE:
        extreme = sum(c for v, c in zip(values, counts) if v <= obs)
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    total = sum(counts)
    return (extreme + 1) / (total + 1)


def random_orthogonal_subspace_pair(d: int, k_pos: int, k_neg: int,
                                    rng: np.random.Generator
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two random subspaces of dimensions (k_pos, k_neg), MUTUALLY ORTHOGONAL.

    Orthogonal because the real ones are: the projector build's resolution
    order removes span(U_pos) from U_neg, so the observed pair is orthogonal by
    construction. `POPPER_PLAN.md` 6h measured what drawing the null pair
    INDEPENDENTLY costs on the same shape of comparison -- an H0 rejection rate
    of 0.0875 against a nominal 0.05, from comparing an orthogonal observed
    pair against overlapping null pairs -- and fixed it exactly this way: one
    Stiefel draw, split, so each half stays marginally uniform while the pair
    is orthogonal.
    """
    k = int(k_pos) + int(k_neg)
    if k > d:
        raise ValueError(
            f"dim U_pos + dim U_neg = {k} exceeds d_model = {d}; no orthogonal "
            f"pair of those dimensions exists, so the null cannot match the "
            f"observed pair's geometry")
    Q = np.linalg.qr(rng.normal(size=(d, k)))[0]
    return Q[:, :int(k_pos)], Q[:, int(k_pos):]


def _sum_D(base: np.ndarray, er0: float, u_pos, u_neg, n_pairs: int,
           alpha: float, mode: str, rng: np.random.Generator) -> float:
    total = 0.0
    for _ in range(int(n_pairs)):
        vp = draw_unit_in_subspace(u_pos, rng, prepared=True)
        vn = draw_unit_in_subspace(u_neg, rng, prepared=True)
        total += pair_statistic(None, vp, vn, alpha, mode, base=base, er0=er0)["D"]
    return total


def subspace_null(activations: np.ndarray, dim_pos: int, dim_neg: int,
                  n_pairs: int, alpha: float, *, mode: str = ER_MODE,
                  n_draws: int = N_SUBSPACE_DRAWS,
                  seed: int = _SEED) -> np.ndarray:
    """
    sum(D) under `n_draws` matched-dimension random orthogonal subspace pairs.

    RETIRED as the adjudicated null and kept as a reported diagnostic. It holds
    the population fixed across draws, which was the point of it, and it holds
    the DIMENSIONS fixed -- but not how much of the population each subspace
    contains, which is what dER is actually driven by. On an H0 family where
    both arms are occupied above chance and the two are identical by
    construction it is anticonservative and the inflation grows with the pair
    count; the rates are in claims/calibration/steering_sign.json rather than
    here. See NULL_FAMILY, and `resplit_null` for what replaced it.
    """
    from core.metrics import effective_rank

    base = _baseline(activations)
    er0 = float(effective_rank(base, mode=mode))
    rng = np.random.default_rng(seed)
    d = base.shape[1]
    return np.array([
        _sum_D(base, er0, *random_orthogonal_subspace_pair(d, dim_pos, dim_neg, rng),
               n_pairs, alpha, mode, rng)
        for _ in range(int(n_draws))], dtype=np.float64)


# ---------------------------------------------------------------------------
# The union, its occupancy, and the null that randomises only the split
# ---------------------------------------------------------------------------

def compact_basis(U) -> np.ndarray:
    """
    A (d, k) orthonormal basis, whichever of the accepted shapes came in.

    `prepared_subspace` deliberately does NOT extract one, because
    `B @ (B.T @ g)` projects correctly for an orthonormal basis and for a
    (d, d) symmetric idempotent projector alike. Forming the union of two
    subspaces needs the actual columns, so this is the one place that pays for
    the extraction: an SVD, with the singular values telling it how many
    columns to keep rather than a caller asserting a rank.
    """
    B = prepared_subspace(U)
    k = subspace_rank(B)
    if B.shape[1] == k and np.allclose(B.T @ B, np.eye(k), atol=1e-8):
        return B
    Q, s, _ = np.linalg.svd(B, full_matrices=False)
    return Q[:, :k]


def union_basis(u_pos, u_neg) -> np.ndarray:
    """
    An orthonormal basis for span(U_pos + U_neg), of dimension k_pos + k_neg.

    REFUSES if the union's numerical rank falls short of that -- which happens
    when the two arms overlap, and when their dimensions together exceed
    d_model. Both are the same fact about the data: there is no pair of
    subspaces of those dimensions sitting orthogonally inside this space, so
    the observed pair is not a point in the Grassmannian the null draws from
    and no re-split of it reproduces the observed geometry.

    Note that the module has ALWAYS assumed this orthogonality -- it is what
    the projector build's resolution order guarantees and what
    `random_orthogonal_subspace_pair` reproduced -- and never checked it. A
    caller passing overlapping arms used to get a null quietly drawn on a
    geometry the observed pair does not have.
    """
    A, B = compact_basis(u_pos), compact_basis(u_neg)
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"U_pos has d={A.shape[0]} and U_neg has d={B.shape[0]}; these "
            f"must match (frame mismatch)")
    want = A.shape[1] + B.shape[1]
    M = np.hstack([A, B])
    s = np.linalg.svd(M, compute_uv=False)
    tol = max(M.shape) * float(np.finfo(np.float64).eps) * (s[0] if s.size else 1.0)
    rank = int((s > max(tol, 1e-10)).sum())
    if rank < want:
        raise ValueError(
            f"span(U_pos + U_neg) has rank {rank}, not dim U_pos + dim U_neg = "
            f"{want} (d_model = {A.shape[0]}). The two arms overlap, or their "
            f"dimensions together exceed d_model. The adjudicated null draws a "
            f"random split OF THAT UNION, so a union that cannot hold the "
            f"observed pair orthogonally gives a null on a geometry the "
            f"observed pair does not have.")
    return np.linalg.qr(M)[0][:, :want]


def occupancy(activations: np.ndarray, U) -> float:
    """
    Share of the CENTRED population's energy inside U, divided by chance.

    Chance is `k / d`: a uniformly random k-dimensional subspace captures that
    fraction in expectation, which is `POPPER_PLAN.md` 6h's
    E[||P_U v||^2] = k/d with the population in place of a single vector. So
    1.0 means "no more of the cloud than a random subspace of this size would
    hold" and the number is comparable across arms of different dimension --
    which raw captured energy is not, and 6h's whole finding was a comparison
    read without that normalization.

    Costs one matmul and no injection, so it is available before a sweep runs.

    Always centred, whatever `DEBIAS_BASELINE_MEAN` says. The two agree at the
    module's own setting, and where they would not, an uncentred "share of the
    population's energy" is dominated by the mean offset rather than by the
    cloud -- so a diagnostic that followed the flag would silently mean two
    different things.
    """
    X = np.asarray(activations, dtype=np.float64)
    base = X - X.mean(axis=0, keepdims=True)
    B = compact_basis(U)
    total = float((base ** 2).sum())
    if total <= 0:
        return float("nan")
    inside = float(((base @ B) ** 2).sum())
    return (inside / total) / (B.shape[1] / B.shape[0])


def occupancy_report(activations: np.ndarray, u_pos, u_neg) -> dict:
    """
    Both arms' chance-normalized occupancy, and the asymmetry between them.

    DIAGNOSTIC: enters no p-value. It is here because the statistic is driven
    by this asymmetry, so a reader deciding whether a TRACKS verdict has a
    non-particle explanation should be able to look at the quantity the verdict
    is made of instead of inferring it. `log_ratio` is positive when U_pos is
    the better-occupied arm, which is the direction H1 predicts.
    """
    op, on = occupancy(activations, u_pos), occupancy(activations, u_neg)
    ratio = (op / on) if (np.isfinite(op) and np.isfinite(on) and on > 0) \
        else float("nan")
    return {
        "occupancy_pos": float(op),
        "occupancy_neg": float(on),
        "occupancy_log_ratio": float(np.log(ratio)) if ratio > 0 else float("nan"),
        "_what": ("share of the centred population's energy inside each arm, "
                  "divided by the k/d a random subspace of that dimension "
                  "would capture. 1.0 is chance. DIAGNOSTIC: enters no "
                  "p-value."),
    }


def resplit_pair(union: np.ndarray, k_pos: int,
                 rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    A uniformly random (k_pos, k - k_pos) split of the union, as two bases.

    Rotating the union's basis by a Haar-random k x k orthogonal matrix and
    cutting the columns gives a uniformly distributed k_pos-dimensional
    subspace of the union together with its orthogonal complement INSIDE the
    union -- so the pair is orthogonal, of the right dimensions, and spans
    exactly what the observed pair spans. Nothing about the union moves.
    """
    S = np.asarray(union, dtype=np.float64)
    k = S.shape[1]
    kp = int(k_pos)
    if not (0 < kp < k):
        raise ValueError(
            f"a split needs 0 < k_pos < k; got k_pos={kp}, k={k}. A union "
            f"assigned entirely to one arm has no second arm to draw from.")
    R = np.linalg.qr(rng.normal(size=(k, k)))[0]
    Z = S @ R
    return Z[:, :kp], Z[:, kp:]


def resplit_null(activations: np.ndarray, union: np.ndarray, k_pos: int,
                 n_pairs: int, alpha: float, *, mode: str = ER_MODE,
                 n_draws: int = N_SUBSPACE_DRAWS,
                 seed: int = _SEED) -> np.ndarray:
    """
    sum(D) under `n_draws` random re-splits of the observed union. ADJUDICATED.

    The population, the union, its dimensions and its whole spectral
    relationship to the cloud are held fixed; only which half is called
    attractive moves. That is H0-BRIDGE for this entry stated exactly -- the
    decomposition label carries no information about the sign -- and it makes
    the observed value one point of the same Grassmannian the null samples,
    so exchangeability is by construction rather than by measurement.
    """
    from core.metrics import effective_rank

    base = _baseline(activations)
    er0 = float(effective_rank(base, mode=mode))
    rng = np.random.default_rng(seed)
    return np.array([
        _sum_D(base, er0, *resplit_pair(union, k_pos, rng),
               n_pairs, alpha, mode, rng)
        for _ in range(int(n_draws))], dtype=np.float64)


def _alpha() -> float:
    from core.changepoint_colocation import _alpha as shared_alpha
    return shared_alpha()


def gate_verdict(p_greater: Optional[float], p_less: Optional[float],
                 alpha: Optional[float] = None) -> dict:
    """
    The three-way lattice, and the registered falsifier that cannot be one.

    P-ST1's falsifier reads "both arms move effective rank the same way, or the
    effect tracks ||s|| and is insensitive to the decomposition". Both clauses
    describe the NULL. An e-process records insufficient evidence and never a
    null accepted, so neither can enter the ledger as a falsification; they map
    to INSUFFICIENT. INVERTS -- attractive-dominant steering demonstrably
    RAISING effective rank while repulsive-dominant lowers it -- is a
    positively shown reversal and is the branch that is recorded as one.
    """
    a = _alpha() if alpha is None else float(alpha)
    if p_greater is None:
        return {"verdict": "INSUFFICIENT", "falsified": False, "alpha": a,
                "reading": "no p-value could be emitted; nothing is adjudicated"}
    if p_greater <= a:
        return {"verdict": "TRACKS-DECOMPOSITION", "falsified": False, "alpha": a,
                "reading": "at matched norm, attractive-dominant steering lowers "
                           "effective rank and repulsive-dominant raises it, more "
                           "often than label swaps allow"}
    if p_less is not None and p_less <= a:
        return {"verdict": "INVERTS", "falsified": True, "alpha": a,
                "reading": "the decomposition governs the sign and governs it "
                           "BACKWARDS: attractive-dominant steering raises "
                           "effective rank. A reversal positively shown, not "
                           "inferred from a failure to reject"}
    return {"verdict": "INSUFFICIENT", "falsified": False, "alpha": a,
            "reading": "neither direction was shown. The registered falsifier's "
                       "own wording -- both arms moving the same way -- lands "
                       "here, and an e-process records insufficient evidence "
                       "rather than a null accepted"}


def label_permutation_diagnostic(D: Sequence[float], alpha: float) -> dict:
    """
    The REGISTERED null, computed and reported, never adjudicated.

    It is kept because the size of the difference between the null the registry
    names and the one that holds should be visible in every record rather than
    asserted in a docstring. Its floor is the informative-pair one --
    (2^(m-k) + 1)/(2^m + 1), set by the k pairs with a nonzero D and not by the
    m drawn, since a D = 0 pair contributes identically to the observed sum and
    to every null pattern.
    """
    d = np.asarray(D, dtype=np.float64)
    m, k = int(d.size), int((d != 0).sum())
    out = {
        "null_family": "label permutation over pairs (REGISTERED, NOT ADJUDICATED)",
        "n_pairs": m, "n_informative_pairs": k,
        "informative_rate": (float(k / m) if m else float("nan")),
        "min_informative_pairs_for_alpha": min_informative_pairs(alpha),
        "measured_h0_rate_given_emission": (
            "anticonservative at alpha=0.05 and rising with the pair count; "
            "rates in claims/calibration/steering_sign.json, whose sha256 pins "
            "them to this module"),
        "why_not_adjudicated": (
            "every pair at one layer shares the tokens and both subspaces, so a "
            "chance tilt moves them together and m pairs are not m exchangeable "
            "units"),
    }
    if m < 1 or k == 0:
        out.update({"p_value": None, "best_attainable_p": None})
        return out
    out["best_attainable_p"] = float(attainable_floor(m, k))
    values, counts = null_distribution(d)
    out["p_value"] = float(p_from_distribution(float(d.sum()), values, counts,
                                               ALTERNATIVE))
    out["p_reciprocal"] = float(p_from_distribution(float(d.sum()), values,
                                                    counts,
                                                    RECIPROCAL_ALTERNATIVE))
    return out


def matched_dimension_diagnostic(activations: np.ndarray, observed: float,
                                 dim_pos: int, dim_neg: int, n_pairs: int,
                                 alpha: float, *, mode: str = ER_MODE,
                                 n_draws: int = N_SUBSPACE_DRAWS,
                                 seed: int = _SEED) -> dict:
    """
    6k's null, computed and reported, never adjudicated.

    It replaces both arms with random subspaces of the same DIMENSIONS, which
    randomises the union and the split together -- so it rejects when the pair
    is unusual as a pair, and "this pair of subspaces holds more of the cloud
    than a random pair would" is a fact about the union rather than about the
    decomposition P-ST1 names. Kept in the record for the same reason the
    registered permutation is: the size of the difference between a null that
    was believed and the one that holds belongs in the artifact, and it has now
    been large twice.
    """
    from core.nulls import p_from_null

    out = {
        "null_family": ("matched-dimension random orthogonal subspace pair "
                        "(RETIRED 2026-08-26, NOT ADJUDICATED)"),
        "n_draws": int(n_draws),
        "measured_h0_rate_given_emission": (
            "anticonservative where both arms are occupied above chance and "
            "are identical by construction, and the inflation grows with the "
            "pair count; rates in claims/calibration/steering_sign.json, whose "
            "sha256 pins them to this module"),
        "why_not_adjudicated": (
            "it randomises the union together with the split, so it rejects on "
            "a pair that holds more of the population than a random pair "
            "would -- a fact about the union, not about which half is called "
            "attractive"),
    }
    null = subspace_null(activations, dim_pos, dim_neg, n_pairs, alpha,
                         mode=mode, n_draws=int(n_draws), seed=seed)
    out["null_mean"] = float(null.mean())
    out["null_sd"] = float(null.std())
    out["p_value"] = float(
        p_from_null(float(observed), null, alternative=ALTERNATIVE)["p_value"])
    out["p_reciprocal"] = float(
        p_from_null(float(observed), null,
                    alternative=RECIPROCAL_ALTERNATIVE)["p_value"])
    return out


def p_value_p_st1(activations: np.ndarray, u_pos, u_neg, n_pairs: int,
                  *, alpha: Optional[float] = None,
                  gate_alpha: Optional[float] = None,
                  seed: int = _SEED, mode: str = ER_MODE,
                  n_draws: int = N_SUBSPACE_DRAWS,
                  with_profile: bool = True) -> dict:
    """
    P-ST1's calibrated p-value, or a refusal saying why there is none.

    The adjudicated null is a random re-split of the observed pair's union --
    see NULL_FAMILY for why it is neither the null the registry's wording names
    nor the matched-dimension pair that replaced that one, and what each of
    those two was measured to do. Both are computed beside the result as
    diagnostics.

    Refuses -- `p_value` None with a `reason` and a `refusal_kind` -- rather
    than returning a number the design cannot support. On a prediction whose
    whole value is that it can lose, a "not significant" produced by an
    underpowered draw reads as a loss, which is the wrong thing to put in the
    ledger.

    Emitting into the ledger is deliberately not this function's job; see
    `adjudicate_p_st1`.
    """
    from core.metrics import effective_rank
    from core.nulls import p_from_null

    a_gate = _alpha() if gate_alpha is None else float(gate_alpha)
    out = draw_pairs(activations, u_pos, u_neg, n_pairs,
                     alpha=alpha, seed=seed, mode=mode)
    out["alpha_gate"] = a_gate
    out["null_family"] = NULL_FAMILY
    D = np.asarray(out["D"], dtype=np.float64)
    m = int(D.size)
    out.update({
        "n_pairs": m,
        "n_informative_pairs": int((D != 0).sum()),
        "informative_rate": (float((D != 0).mean()) if m else float("nan")),
        "observed": float(D.sum()),
        "n_predicted": int((D == 2).sum()),
        "n_inverted": int((D == -2).sum()),
        "n_partial": int((np.abs(D) == 1).sum()),
        "n_subspace_draws": int(n_draws),
        "refusal_kind": None,
    })
    out["occupancy"] = occupancy_report(activations, u_pos, u_neg)
    if REPORT_LABEL_PERMUTATION_DIAGNOSTIC:
        out["label_permutation_diagnostic"] = label_permutation_diagnostic(D, a_gate)
    if with_profile:
        out["alpha_profile"] = alpha_profile(activations, u_pos, u_neg, n_pairs,
                                             seed=seed, mode=mode)

    def _refuse(kind: str, reason: str) -> dict:
        out["p_value"] = None
        out["refusal_kind"] = kind
        out["reason"] = reason
        out.update(gate_verdict(None, None, a_gate))
        return out

    # DATA refusals first, calibration refusals second, and the order is the
    # arrangement POPPER_PLAN.md 6l settled for CLAIM-C's gate: a run whose
    # geometry cannot carry the null should say so, rather than be turned away
    # for a draw count that could be raised. Both groups stay reachable --
    # `n_draws` is a caller's argument and the geometry is the data's, so
    # neither can hide the other.
    if m < 1:
        return _refuse("no_pairs",
                       "no pairs were drawn, so there is no statistic")

    try:
        union = union_basis(u_pos, u_neg)
    except ValueError as exc:
        return _refuse("union_rank_deficient", str(exc))
    out["dim_union"] = int(union.shape[1])

    floor = 1.0 / (int(n_draws) + 1.0)
    out["draw_count_floor"] = float(floor)
    # Provisional: replaced below by the floor the null's TIES set, which is
    # the one a run can actually reach. It is set here so a record that stops
    # at the refusal below still carries a floor rather than a missing key.
    out["best_attainable_p"] = float(floor)
    if floor > a_gate:
        return _refuse("draws_below_floor", (
            f"{n_draws} null draws can express no p smaller than "
            f"1/({n_draws}+1) = {floor:.4f}, above alpha={a_gate}. A test that "
            f"cannot reject on a perfect result is not a test. This one IS "
            f"fixed by the draws and is raised by drawing more -- unlike the "
            f"attainable floor computed below it, which is set by how many "
            f"null re-splits already reach the largest value the statistic "
            f"can take and is a fact about the layer."))

    null = resplit_null(activations, union, out["dim_u_pos"], m, out["alpha"],
                        mode=mode, n_draws=int(n_draws), seed=seed + 1)
    out["null_mean"] = float(null.mean())
    out["null_sd"] = float(null.std())
    out["null_is_degenerate"] = bool(np.all(null == null[0]))

    # THE FLOOR THAT TIES SET, and it is not 1/(draws + 1).
    #
    # sum(D) cannot exceed 2m, so the smallest p this run can express in the
    # `greater` direction is what a hypothetical observation of 2m would get:
    # every null draw that already reaches 2m ties it and is counted extreme.
    # On a union the cloud occupies, random re-splits inform often and many of
    # them reach the maximum, so the draw-count floor 1/(draws+1) can be
    # unreachable by a wide margin -- at one pair by an order of magnitude.
    # The measured version is in claims/audits/p_st1_dry_run.json.
    #
    # Reporting the draw-count floor as `best_attainable_p` was therefore
    # optimistic in exactly the way POPPER_PLAN.md 6i names for CLAIM-B's
    # sampled pairing regime, and leaving it uncorrected reproduced 6l's
    # defect for CLAIM-C: a design that COULD NOT have rejected returning "not
    # significant", which on an entry whose whole value is that it can lose
    # reads as a loss. Found by running the gate on an input whose answer was
    # known (tools/dry_run_p_st1.py), not by a failing test.
    #
    # 2m is an upper bound on the observation rather than an attainable value,
    # so this floor is a LOWER bound on what the run can express: refusing on
    # it can never turn away a table that would have cleared alpha. Both tails
    # are computed, and the refusal needs BOTH to be out of reach -- one
    # reachable tail means one reachable verdict, and the gate is then not a
    # constant function.
    reach_g = float((int((null >= 2.0 * m).sum()) + 1) / (int(n_draws) + 1.0))
    reach_l = float((int((null <= -2.0 * m).sum()) + 1) / (int(n_draws) + 1.0))
    out["attainable_p_greater"] = reach_g
    out["attainable_p_reciprocal"] = reach_l
    out["best_attainable_p"] = float(min(reach_g, reach_l))
    # Which VERDICTS this run could have reached, reported rather than refused
    # on. Refusing when only one tail is out of reach would cost the verdict
    # the other tail can still deliver, which is the thing 6l's rule forbids.
    # But the asymmetry matters to a reader: a run whose only reachable verdict
    # is INVERTS can produce a falsification and cannot produce its opposite,
    # and that belongs in the record rather than in an inference from two
    # floats.
    out["reachable_tails"] = (
        (["greater"] if reach_g <= a_gate else [])
        + (["reciprocal"] if reach_l <= a_gate else []))
    if min(reach_g, reach_l) > a_gate:
        return _refuse("null_ties_the_maximum", (
            f"with {m} pairs the statistic cannot exceed {2 * m}, and "
            f"{int((null >= 2.0 * m).sum())} of {n_draws} null re-splits "
            f"already reach it, so the smallest expressible p is "
            f"{reach_g:.4f} in the predicted direction and {reach_l:.4f} in "
            f"the reciprocal one -- both above alpha={a_gate}. Neither "
            f"TRACKS-DECOMPOSITION nor INVERTS is reachable however the data "
            f"falls, so the verdict was INSUFFICIENT before the statistic was "
            f"looked at. Raising n_draws does NOT fix this: the ties are the "
            f"layer's, and more pairs or a union the re-splits inform about "
            f"less often would. dim U_pos {out['dim_u_pos']}, dim U_neg "
            f"{out['dim_u_neg']}, occupancy "
            f"{out['occupancy']['occupancy_pos']:.2f}/"
            f"{out['occupancy']['occupancy_neg']:.2f}."))

    out["p_value"] = float(
        p_from_null(out["observed"], null, alternative=ALTERNATIVE)["p_value"])
    out["p_reciprocal"] = float(
        p_from_null(out["observed"], null,
                    alternative=RECIPROCAL_ALTERNATIVE)["p_value"])
    out["alternative"] = ALTERNATIVE
    if MATCHED_DIMENSION_NULL_DIAGNOSTIC:
        out["matched_dimension_diagnostic"] = matched_dimension_diagnostic(
            activations, out["observed"], out["dim_u_pos"], out["dim_u_neg"], m,
            out["alpha"], mode=mode, n_draws=int(n_draws), seed=seed + 2)
    out["statistic"] = (
        f"sum over {m} matched-norm vector pairs of D = sign(dER_neg) - "
        f"sign(dER_pos), where dER is the change in {mode} effective rank when "
        f"alpha*v is added to every token at one layer and alpha = "
        f"{out['alpha_spread_fraction']:.3g} x the population's RMS deviation "
        f"from its mean (PLACED); baseline mean "
        f"{'removed' if DEBIAS_BASELINE_MEAN else 'kept'} before injection; "
        f"null re-splits span(U_pos + U_neg), dimension {out['dim_union']}, "
        f"into {n_draws} random ({out['dim_u_pos']}, {out['dim_u_neg']}) "
        f"orthogonal pairs at the same layer and population, holding the union "
        f"fixed; one-sided '{ALTERNATIVE}'")
    out.update(gate_verdict(out["p_value"], out["p_reciprocal"], a_gate))
    return out


def adjudicate_p_st1(activations: np.ndarray, u_pos, u_neg, n_pairs: int,
                     *, alpha: Optional[float] = None,
                     gate_alpha: Optional[float] = None,
                     seed: int = _SEED, mode: str = ER_MODE,
                     n_draws: int = N_SUBSPACE_DRAWS,
                     artifact_hashes: Sequence[str] = (),
                     run_manifest: Optional[dict] = None,
                     adjudicate: bool = False,
                     adjudications_dir=None) -> dict:
    """
    `p_value_p_st1` plus, optionally, an entry in the falsification ledger.

    Opt-in behind a flag for the reason it is everywhere else here: these
    functions are exercised by tests and `core.adjudication` refuses to
    overwrite an existing record, so one accidental fixture run would
    permanently occupy P-ST1's slot with a synthetic p-value.
    """
    res = p_value_p_st1(activations, u_pos, u_neg, n_pairs, alpha=alpha,
                        gate_alpha=gate_alpha, seed=seed, mode=mode,
                        n_draws=n_draws)
    res["adjudication"] = None
    if not (adjudicate and res.get("p_value") is not None):
        return res

    from core.adjudication import adjudicate_if_registered
    res["adjudication"] = adjudicate_if_registered(
        "P-ST1", res["p_value"],
        artifact_hashes=tuple(artifact_hashes), run_manifest=run_manifest,
        test_name=res["statistic"],
        notes=(
            f"verdict={res['verdict']} p_reciprocal={res['p_reciprocal']:.4f} "
            f"(INVERTS input only, NOT calibrated into E) "
            f"null = {NULL_FAMILY} over {res['n_subspace_draws']} draws, which "
            f"is NEITHER of the two nulls this entry previously carried. The "
            f"registry's wording names a label permutation across pairs, "
            f"measured anticonservative at alpha=0.05 conditional on emission, "
            f"rising with the pair count; 2026-08-25 replaced it with a "
            f"matched-dimension random orthogonal pair, measured "
            f"anticonservative where both arms are occupied above chance and "
            f"are identical by construction -- the inflation growing with the "
            f"pair count -- because it randomises the union together with the "
            f"split; see claims/calibration/steering_sign.json. Both are "
            f"computed and "
            f"reported as diagnostics only; "
            f"{res['n_informative_pairs']}/{res['n_pairs']} pairs informative, "
            f"floor {res['best_attainable_p']:.4f}; dim U_pos "
            f"{res['dim_u_pos']}, dim U_neg {res['dim_u_neg']}, d_model "
            f"{res['d_model']}; attainable floors "
            f"{res['attainable_p_greater']:.4f} (greater) and "
            f"{res['attainable_p_reciprocal']:.4f} (reciprocal), set by how "
            f"many null re-splits already reach the largest value the "
            f"statistic can take and NOT by the draw count "
            f"({res['draw_count_floor']:.4f}); reachable verdicts "
            f"{res['reachable_tails']}; chance-normalized occupancy "
            f"{res['occupancy']['occupancy_pos']:.3f} (pos) against "
            f"{res['occupancy']['occupancy_neg']:.3f} (neg), DIAGNOSTIC and "
            f"the quantity a TRACKS verdict is made of; population mean/spread "
            f"{res['population_mean_ratio']:.3f}; alpha is PLACED at "
            f"{res['alpha_spread_fraction']:.3g} x spread and the alpha-profile "
            f"is reported as a diagnostic that enters no p-value; the "
            f"registered falsifier's wording describes the null and maps to "
            f"INSUFFICIENT, so only INVERTS is recorded as a falsification"),
        adjudications_dir=adjudications_dir,
    )
    return res
