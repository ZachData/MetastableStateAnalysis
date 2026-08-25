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

What replaces it is 6h's construction, arriving for the fourth time: randomise
over SUBSPACES, not over units. H0 for this entry is "the sign is independent
of the U_pos/U_neg decomposition", realised directly by replacing the two
operator-derived subspaces with random ones OF THE SAME DIMENSIONS, at the same
layer, on the same population. Every chance tilt is present in the null exactly
as it is in the observed value, so the confound the permutation cannot see is
what the null is made of. The pair is drawn MUTUALLY ORTHOGONAL from one
Stiefel draw, because the real pair is orthogonal by the projector build's
resolution order and 6h measured the cost of forgetting that at 0.0875 against
a nominal 0.05.

The registered permutation is still computed and reported beside every result
as a diagnostic, never adjudicated, so the difference between the null the
wording names and the one that holds is visible in the record rather than
asserted here.

THE FLOOR, AND WHY REPLACING THE NULL REMOVED IT

Under the REGISTERED permutation the floor is not 2/(2^m + 1). A pair whose two
arms move effective rank the same way contributes D = 0, and a zero contributes
identically to the observed sum and to every null pattern, so with k of m pairs
informative the best attainable p is

    (2^(m - k) + 1) / (2^m + 1)     ~=  2^-k

-- set by the INFORMATIVE pairs and not by the pairs drawn. Five is the first k
that clears alpha = 0.05, at every m, so a hundred pairs at a 2% informative
rate buy two informative pairs and a best possible p of 0.25.

The subspace null has no such property: its floor is 1/(draws + 1), fixed by
how many null draws are taken and independent of the data. A single informative
pair can reject, and correctly so -- if random subspaces of the same dimensions
essentially never inform, an operator-derived pair that does is exactly the
surprise the claim is about. So replacing the null did not only fix validity;
it removed a power requirement the registered design could not meet. The
informative-pair floor is still computed and reported, because it is the
diagnostic arm's floor and a reader comparing the two needs it.

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

WHAT IS STILL WEAKLY MEASURED, stated rather than left to be found. The
RECIPROCAL tail -- the INVERTS branch, the one that would enter the ledger as a
falsification -- is measured under H0 at 0.02 to 0.10 over fifty gate runs per
cell. That is consistent with nominal and it is not a tight bound: fifty runs
resolve a rate to about +/- 0.03, and the one cell at 0.10 is well inside that.
The adjudicated `greater` tail is measured at 0.000 to 0.040 across the same
cells. Before anything is adjudicated on the INVERTS branch specifically, that
cell wants more replicates than a committed artifact can afford to carry.

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
#: alpha/spread of 0.15, 0.17, 0.20, 0.22, 0.24, 0.26, 0.28: 0.81, 0.96, 1.00,
#: 1.00, 1.00, 0.03, 0.00. A grid of (0.03, 0.1, 0.3) reads 0.1 as the peak
#: because its neighbours are both zero; the plateau is 0.17-0.24 and 0.1 sits
#: on the shoulder at a sixth of the informative rate -- which is 29 pairs
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
#: the two modes are indistinguishable here (informative rate 0.153 raw against
#: 0.170 normed, H0 exactly 0.000 for both) -- which is exactly how this nearly
#: shipped wrong. Away from that one point they are not interchangeable:
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
#:   and it manufactures inversions -- D = -2 in 12-18% of pairs at
#:   alpha <= 1e-4 where raw gives 0%.
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
#: alpha = 0.05, conditional on the gate emitting: 0.143 to 0.172 at 40 and 150
#: pairs. That is `status-6.md`'s "49 layers are not 49 independent
#: observations" arriving a third time, and it is invisible in the clean regime
#: because there the gate refuses instead of emitting -- so the rate looks like
#: 0.000 unless it is conditioned on emission, which is POPPER_PLAN.md 6g's
#: lesson exactly.
#:
#: What replaces it is POPPER_PLAN.md 6h's construction, arriving for the
#: fourth time: randomise over SUBSPACES, not over units. H0-BRIDGE for this
#: entry is "the sign is independent of the U_pos/U_neg decomposition", which
#: is realised directly by replacing the two operator-derived subspaces with
#: RANDOM ones OF THE SAME DIMENSIONS. Everything the statistic could read off
#: dimension is held fixed, the layer's own geometry is held fixed, and a
#: chance tilt is present in every null draw exactly as it is in the observed
#: one -- so the confound the permutation cannot see is what the null is made
#: of. The floor becomes 1/(draws + 1) and stops depending on the pair count at
#: all, which is the same escape 6h found when P6-R2's floor moved from 0.667
#: to 0.0005 on this question.
NULL_FAMILY = "matched-dimension random orthogonal subspace pair"

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

    The POPULATION is held fixed across draws -- that is the whole point. A
    chance tilt of this layer's cloud toward one subspace produces the same
    tilt in every null draw, so the observed value is compared against the
    tilts chance supplies rather than against an independence assumption the
    pairs do not satisfy.
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
        "measured_h0_rate_given_emission": "0.143-0.172 at alpha=0.05; see "
                                           "claims/calibration/steering_sign.json",
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


def p_value_p_st1(activations: np.ndarray, u_pos, u_neg, n_pairs: int,
                  *, alpha: Optional[float] = None,
                  gate_alpha: Optional[float] = None,
                  seed: int = _SEED, mode: str = ER_MODE,
                  n_draws: int = N_SUBSPACE_DRAWS,
                  with_profile: bool = True) -> dict:
    """
    P-ST1's calibrated p-value, or a refusal saying why there is none.

    The adjudicated null is the matched-dimension random orthogonal subspace
    pair -- see NULL_FAMILY for why it is not the one the registry's wording
    names, and what that wording's null was measured to do. The registered
    permutation is computed beside it as a diagnostic.

    Refuses -- `p_value` None with a `reason` -- rather than returning a number
    the design cannot support. On a prediction whose whole value is that it can
    lose, a "not significant" produced by an underpowered draw reads as a loss,
    which is the wrong thing to put in the ledger.

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
    })
    if REPORT_LABEL_PERMUTATION_DIAGNOSTIC:
        out["label_permutation_diagnostic"] = label_permutation_diagnostic(D, a_gate)
    if with_profile:
        out["alpha_profile"] = alpha_profile(activations, u_pos, u_neg, n_pairs,
                                             seed=seed, mode=mode)

    if m < 1:
        out["p_value"] = None
        out["reason"] = "no pairs were drawn, so there is no statistic"
        out.update(gate_verdict(None, None, a_gate))
        return out

    floor = 1.0 / (int(n_draws) + 1.0)
    out["best_attainable_p"] = float(floor)
    if floor > a_gate:
        out["p_value"] = None
        out["reason"] = (
            f"{n_draws} null draws can express no p smaller than "
            f"1/({n_draws}+1) = {floor:.4f}, above alpha={a_gate}. A test that "
            f"cannot reject on a perfect result is not a test. Unlike the "
            f"registered permutation's floor this one is fixed by the DRAWS "
            f"and not by the data, so it is raised by drawing more.")
        out.update(gate_verdict(None, None, a_gate))
        return out

    d_model = int(np.asarray(activations).shape[1])
    if out["dim_u_pos"] + out["dim_u_neg"] > d_model:
        out["p_value"] = None
        out["reason"] = (
            f"dim U_pos ({out['dim_u_pos']}) + dim U_neg ({out['dim_u_neg']}) "
            f"exceeds d_model ({d_model}), so no random ORTHOGONAL pair of "
            f"matching dimensions exists. Drawing the null pair independently "
            f"instead would compare an orthogonal observed pair against "
            f"overlapping null pairs, which POPPER_PLAN.md 6h measured at an "
            f"H0 rate of 0.0875 against a nominal 0.05.")
        out.update(gate_verdict(None, None, a_gate))
        return out

    null = subspace_null(activations, out["dim_u_pos"], out["dim_u_neg"], m,
                         out["alpha"], mode=mode, n_draws=int(n_draws),
                         seed=seed + 1)
    out["null_mean"] = float(null.mean())
    out["null_sd"] = float(null.std())
    out["null_is_degenerate"] = bool(np.all(null == null[0]))
    out["p_value"] = float(
        p_from_null(out["observed"], null, alternative=ALTERNATIVE)["p_value"])
    out["p_reciprocal"] = float(
        p_from_null(out["observed"], null,
                    alternative=RECIPROCAL_ALTERNATIVE)["p_value"])
    out["alternative"] = ALTERNATIVE
    out["statistic"] = (
        f"sum over {m} matched-norm vector pairs of D = sign(dER_neg) - "
        f"sign(dER_pos), where dER is the change in {mode} effective rank when "
        f"alpha*v is added to every token at one layer and alpha = "
        f"{out['alpha_spread_fraction']:.3g} x the population's RMS deviation "
        f"from its mean (PLACED); baseline mean "
        f"{'removed' if DEBIAS_BASELINE_MEAN else 'kept'} before injection; "
        f"null replaces (U_pos, U_neg) with {n_draws} random ORTHOGONAL "
        f"subspace pairs of the same dimensions "
        f"({out['dim_u_pos']}, {out['dim_u_neg']}) at the same layer and "
        f"population; one-sided '{ALTERNATIVE}'")
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
            f"null = {NULL_FAMILY} over {res['n_subspace_draws']} draws, NOT "
            f"the label permutation the registry's wording names -- that one "
            f"was measured anticonservative (0.143-0.172 at alpha=0.05 "
            f"conditional on emission) and is reported as a diagnostic only; "
            f"{res['n_informative_pairs']}/{res['n_pairs']} pairs informative, "
            f"floor {res['best_attainable_p']:.4f}; dim U_pos "
            f"{res['dim_u_pos']}, dim U_neg {res['dim_u_neg']}, d_model "
            f"{res['d_model']}; population mean/spread "
            f"{res['population_mean_ratio']:.3f}; alpha is PLACED at "
            f"{res['alpha_spread_fraction']:.3g} x spread and the alpha-profile "
            f"is reported as a diagnostic that enters no p-value; the "
            f"registered falsifier's wording describes the null and maps to "
            f"INSUFFICIENT, so only INVERTS is recorded as a falsification"),
        adjudications_dir=adjudications_dir,
    )
    return res
