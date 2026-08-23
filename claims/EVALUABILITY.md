# EVALUABILITY.md — which predictions can carry an e-value, and which cannot

POPPER_PLAN.md item B5. Generated from `claims/registry.json`; regenerate with
`python tools/render_evaluability.py` rather than editing by hand.

## Why this document is the important one

The temptation, once an e-value kernel exists, is to attach one to every
prediction in the project and report a cumulative E per claim. That would be
worse than doing nothing, and the reason is arithmetic rather than
philosophical: **the product is only as valid as its weakest factor.** One
e-value derived from a null that is not actually valid under H0 voids the
Type-I guarantee for every other prediction on that claim, silently, and the
artifact gives a later reader no way to tell which factor was the bad one.

POPPER measures this directly. Its relevance checker exists solely to keep
sub-hypotheses whose nulls are not implied by the main hypothesis out of the
product, and removing it raises Type-I error from 0.082 to 0.340 on
TargetVal-IL2 -- a fourfold inflation, from nothing but admitting nulls that
should not have been there.

So each of the 30 registered predictions is classified into exactly one of
three states, and `core/adjudication.py` refuses to emit an e-value for the
latter two rather than emitting a neutral one. That refusal is an instance of
the project's own standing rule 4: *"Refuse rather than degrade. A number from
mismatched inputs is worse than no number: it is unfalsifiable from the output
alone."* (`UPDATE_PLAN.md` §6.)

## The three states

**`e-value`** — a valid null exists or is directly constructible, and the
p-value it yields is calibrated under H0. These may contribute to a claim's
product.

**`needs-null`** — the prediction is testable and worth testing, but the null
has to be built before a p-value means anything. Most of these are currently
*threshold comparisons*: a measured quantity against a number, with no
distribution behind the number. That is a decision rule, not a test. Each row
here is a queued chunk.

**`measurement`** — no valid null exists, and forcing one would manufacture
evidence. The honest output is a number with an interval. `P-H1` is the clearest
case: Wendel's theorem gives probability 1 for $d > n$, which all eight prompts
satisfy, and `UPDATE_PLAN.md` §5.7 already calls the boolean "nearly vacuous".
An e-value here would be evidence extracted from a theorem rather than from
data.

## The count

| state | n | may contribute to a claim's E |
|---|---|---|
| `e-value` | 7 | yes |
| `needs-null` | 28 | not yet |
| `measurement` | 3 | never |

## Three patterns worth naming, because they recur

**A threshold is not a null.** `P5b-A1` ("32d PCA retains ≥ 80% variance"),
`P6-R1` ("degeneracy ratio ≥ 5"), `P5b-C1`'s arc-length bound — each compares a
measurement to a constant. The fix is the same in every case and it is cheap:
sample the matched random baseline the prediction already implies, and read the
observed value against that distribution. `core/nulls.py` exists for exactly
this and already implements both constructions the project needs.

**An equivalence claim needs an equivalence test.** `P5b-D2` says
$r_A \approx r_\text{linear}$ — the imaginary subspace carries *no extra*
behavioural geometry. A conventional test cannot support that: failing to reject
is not evidence of equality, and a large p-value from an underpowered test looks
identical to a real null effect. It needs a TOST with a pre-stated margin, or it
stays a measurement. Registered as `measurement` until that margin is set.

**The same data cannot settle two entries.** `P5b-B1` ($r_\text{manifold} >
r_\text{linear}$) and `P5b-B3` (the same difference $> 0.1$) are one test with
two thresholds. Admitting both into the product would multiply one experiment's
evidence in twice. `core/evalues.EProcess.add` refuses a repeated prediction id,
but it cannot see that two *different* ids are the same experiment — so the
registry must name which of the two is adjudicated, and that decision belongs
in the chunk that constructs the null.

## Already-run predictions, and why they are still `needs-null`

`p6_subspace/status-6.md` records `P6-R2` and `P6-R4` as **run and inverted** —
mean LDA alignment 0.887 with the imaginary subspace $U_A$ against 0.067 with
the real repulsive $U_\text{neg}$, and **0 of 49 layers** in the predicted
direction. That is the strongest single result in the registry and it points
against the prediction.

It is still `needs-null`, for the reason `status-6.md` states itself: 49 ALBERT
layers are not 49 independent observations. Turning "0/49" into a p-value
requires deciding what the independent unit is, and that decision changes the
answer by orders of magnitude. Registering the inversion as a fact and the
p-value as outstanding is the correct state — not a hedge.

## The table

| id | claim | state | relevance | null construction, or why none exists |
|---|---|---|---|---|
| `CLAIM-A` | H-RESIST | **needs-null** | 1.0 | Needs a null over the three pass criteria jointly. Each is currently a threshold comparison; a matched random-weight baseline (core/nulls.shuffled_dimension_null) gives a per-criterion p, but combining three into one prediction needs the combination fixed in advance. |
| `P-gamma1` | H-RESIST | **needs-null** | 1.0 | beta_reduction.py reports a residual BRACKET across beta in [0.5,5] rather than a point estimate. A p-value needs the bracket converted to a test: bootstrap over prompts, with the reading restated per UPDATE_PLAN.md 5.3 (the calibrated step makes the residual rate-invariant, so it measures direction not magnitude). |
| `P-gamma2` | H-RESIST | **needs-null** | 1.0 | Point estimate against a constant. A bootstrap over the eight prompts gives a p-value; n=8 is small and the record must say so rather than bury it. UPDATE_PLAN.md 5.2: three step-size definitions are computed and verdict() refuses a call when they straddle t*, which must be honoured before any p is emitted. |
| `P-H1` | H-RESIST | **measurement** | 0.8 | NONE, and deliberately so. Wendel's theorem gives probability 1 for d > n, which all 8 prompts satisfy; UPDATE_PLAN.md 5.7 already records the boolean as 'nearly vacuous'. Emitting an e-value here would manufacture evidence out of a theorem. The reportable object is the MARGIN max_w min_i <x_i, w> and the layer at which it first crosses zero. |
| `P-S1` | H-RESIST | **e-value** | 1.0 | Monte-Carlo permutation against a matched i.i.d. baseline at the trained configuration's own (m, d). STATISTIC, fixed in advance: the sum over degrees k=1..3 of the (step0 - trained) Q_k ratio, each standardised by that degree's own null standard deviation. ALTERNATIVE: one-sided 'greater' (P-S1 predicts trained ratios SMALLER, so step0 - trained > 0). NULL: two independent i.i.d. configurations at the same (m, d), same statistic -- a pair per draw, so the null carries both arms' sampling variability. Resolution floor 1/(n_draws+1). Implemented in p1c_frames/centroids.py::p_value_p_s1. |
| `CLAIM-C` | H-TRANSFER | **needs-null** | 1.0 | Requires a stated equivalence criterion. 'Reproduces' is currently prose; a p-value needs the criterion fixed in advance (which statistics, what tolerance) or the prediction is unfalsifiable in practice. This is the highest-priority needs-null item because it is the one with a hard stop attached. |
| `CLAIM-B` | H-EMERGE | **needs-null** | 1.0 | Co-location of two changepoints across a checkpoint sweep. A permutation null over checkpoint order gives a valid p once the changepoint estimator is fixed in advance. |
| `P-T1` | H-OPERATOR | **e-value** | 1.0 | Two-proportion test (candidates vs the REQUIRED control arm of non-candidates), or Fisher's exact when counts are small. The control arm is mandatory per the dated addendum: without it a candidates-only number reads as confirmation. |
| `P-M1` | H-OPERATOR | **e-value** | 1.0 | Permutation test over layer boundaries: shuffle the per-boundary violation INDICATOR against the per-layer regime score. UPDATE_PLAN.md 5.9 - a violation is an event between two adjacent layers, so the series is an indicator, not a count; layer 0 is zero by construction and is reported rather than dropped so the two series stay aligned. |
| `P6-A2` | H-OPERATOR | **needs-null** | 0.8 | Classification agreement between f_rot and head type; needs a permutation null over the head-type labels. |
| `P6-I1` | H-OPERATOR | **e-value** | 1.0 | Already a Mann-Whitney U on f_rot(induction heads) vs f_rot(semantic heads). Valid as it stands; only needs threading through core.adjudication. |
| `P6-I2` | H-OPERATOR | **e-value** | 0.8 | Two-sample test over head pairs; same shape as P6-I1. |
| `P6-R1` | H-OPERATOR | **needs-null** | 0.8 | Threshold on a ratio (R >= 5) with a random-projection reference already named. That reference IS the null; it needs to be sampled rather than used as a single comparison value. |
| `P6-R2` | H-OPERATOR | **needs-null** | 1.0 | Currently a per-layer direction comparison. UPDATE ORDER MATTERS: status-6.md records this as ALREADY RUN AND INVERTED (0/49 layers show the predicted direction, alignment 0.887 with U_A vs 0.067 with U_neg). Any p-value must respect status-6.md's own caveat that 49 ALBERT layers are not 49 independent observations. |
| `P6-R3` | H-OPERATOR | **needs-null** | 0.8 | Directional dominance at merge events; permutation over merge vs non-merge steps. |
| `P6-R4` | H-OPERATOR | **needs-null** | 1.0 | Probe accuracy comparison with a real-only / imaginary-only / full contrast. Same inversion as P6-R2 per status-6.md. |
| `P6-R5` | H-OPERATOR | **needs-null** | 0.8 | Counts of contracting/rotating steps against a null of no directional preference; a binomial test once the per-step unit is fixed. |
| `P6-C1` | H-OPERATOR | **needs-null** | 0.8 | Alignment of write subspace with matching channel; needs a random-subspace null of matched dimension. |
| `P6-DD1` | H-OPERATOR | **needs-null** | 1.0 | Two thresholds (induction drop, ARI floor) on an intervention. Needs the intervention's own control arm sampled. |
| `P6-DD2` | H-OPERATOR | **needs-null** | 1.0 | Symmetric counterpart of P6-DD1, same construction. |
| `P6-D5` | H-OPERATOR | **needs-null** | 0.8 | Monotonicity of d_S approaching merge vs d_A; a trend test with a permutation null over the approach window. |
| `P5b-A1` | H-BRIDGE | **needs-null** | 0.8 | Variance-retention threshold. A null needs matched random data of the same shape (a Marchenko-Pastur or permutation reference), not a bare 80% cut. |
| `P5b-A2` | H-BRIDGE | **needs-null** | 0.8 | Ratio against a placed threshold; needs a matched-random residual distribution. |
| `P5b-B1` | H-BRIDGE | **e-value** | 1.0 | A difference of two dependent correlations on the same pairs. Steiger's test, or a paired bootstrap over pairs. isometry_test.py already computes p_manifold; the DIFFERENCE is what B1 asserts and it needs its own test. |
| `P5b-B2` | H-BRIDGE | **measurement** | 0.6 | A threshold against a value calibrated from Wurgaft's reported numbers (p5b_distances.py:31), not from a null distribution of our own. That makes it a calibrated cut, not a test; report r_manifold with a confidence interval. |
| `P5b-B3` | H-BRIDGE | **needs-null** | 0.8 | Effect-size floor on the same dependent-correlation difference as B1. Same test, different threshold; register which of B1/B3 is adjudicated so the same data does not support two entries. |
| `P5b-C1` | H-BRIDGE | **e-value** | 1.0 | Already a two-sample comparison with a stated alpha. Valid once the test is fixed (merge_teleportation_subspace.py computes it); only needs threading through core.adjudication. |
| `P5b-C3` | H-BRIDGE | **needs-null** | 0.8 | Two-sample comparison over layers; permutation over the merge/plateau labelling. |
| `P5b-D1` | H-BRIDGE | **needs-null** | 1.0 | A three-way ordering of dependent correlations. Needs the ordering tested jointly rather than as two pairwise comparisons, and needs a matched-dimension control subspace (subspace_isometry.py already provides one). |
| `P5b-D2` | H-BRIDGE | **measurement** | 1.0 | An EQUIVALENCE claim, not a difference. A conventional test cannot support it - failing to reject is not evidence of equality. Needs a TOST/equivalence bound with a pre-stated margin, or it stays a measurement. |
| `P-ST1` | H-BRIDGE | **needs-null** | 1.0 | Two-sample comparison of the signed effective-rank delta between the two arms, over a set of matched-norm vector pairs drawn per layer. The null is generated by the SAME injection procedure with the decomposition label permuted across pairs, so the norm, the layer and the injection machinery are all held fixed and only the label moves. Requires the pair-drawing scheme (how many pairs, how 'predominantly' is thresholded) fixed in advance. |
| `P-I5` | H-BRIDGE | **needs-null** | 1.0 | Permutation null over the matched-magnitude random-direction ablation arm, on a two-dimensional statistic (geometric delta, logit delta). The joint form matters: two separate one-dimensional tests would let the prediction be scored a partial pass in the configuration it is designed to rule out. REQUIRES an extension to core/dual_reading.py -- every current geometric field is per-point and this needs a pairwise one. |
| `P-AB1` | H-BRIDGE | **needs-null** | 0.8 | Growth-exponent comparison against a MATCHED RANDOM-DIRECTION ablation of equal magnitude at the same layer -- the same control design design-5c.md already requires for its force-collapse and force-disperse arms. The control is not optional: later layers have more opportunity to diverge for reasons unrelated to field structure, so a superlinear fit against no control measures remaining depth, not mechanism. Permutation over ablation points once the fitted exponent is the statistic. |
| `P-SA1` | H-BRIDGE | **needs-null** | 0.8 | Random-subspace null of matched dimension, comparing the observed mass fraction in U_neg against dictionaries of the same rank drawn isotropically. |
| `P-I1` | H-BRIDGE | **needs-null** | 1.0 | Changepoint co-location across a checkpoint sweep; permutation over checkpoint order once the changepoint estimator is fixed. Same construction CLAIM-B needs, and the two should share it rather than each inventing one. |
| `P-I2` | H-BRIDGE | **needs-null** | 1.0 | Two-sample comparison of channel mass between edge types, against the N1/N2 nulls motif_stats.py already gates on. |
| `P-I3` | H-BRIDGE | **needs-null** | 1.0 | Correlation with a REQUIRED control arm over non-induction heads; motif_stats.py makes independence_source a positional argument so the arm cannot be omitted by default. Permutation over the head classification. |
| `P-I4` | H-BRIDGE | **needs-null** | 1.0 | Matched-magnitude control on moved_fraction; permutation over which edges are labelled motif edges. |

## Adjudication order

Cheapest-first among the `e-value` rows, since those need no new construction —
`P6-I1` (already a Mann-Whitney U), `P5b-C1` (already a two-sample comparison
with a stated α), then `P-S1`, `P-T1`, `P-M1`, `P5b-B1`, `P6-I2`.

Then the `needs-null` rows in the order their claims matter: `CLAIM-C` first,
because it is the one with a hard stop attached and a stop rule that cannot be
adjudicated is a stop rule that gets argued with at the moment it binds.
