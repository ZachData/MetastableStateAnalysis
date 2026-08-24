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
| `e-value` | 10 | yes |
| `needs-null` | 25 | not yet |
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

## Already-run predictions: what happened to the strongest result in the registry

`p6_subspace/status-6.md` records `P6-R2` and `P6-R4` as **run and inverted** —
mean LDA alignment 0.887 with the imaginary subspace $U_A$ against 0.067 with
the real repulsive $U_\text{neg}$, and **0 of 49 layers** in the predicted
direction. That was described here as the strongest single result in the
registry, pointing against the prediction, and as `needs-null` because 49 ALBERT
layers are not 49 independent observations.

**Both halves of that were wrong, and the second one was the smaller error**
(2026-08-24, `POPPER_PLAN.md` §6h).

`p6_subspace/math-6.md` §7.2 names a third explanation neither `status-6.md`
nor this file listed: **the comparison is not dimension-normalized.** For a
random unit vector and a $k$-dimensional subspace, $\mathbb{E}[\lVert P_U v
\rVert^2] = k/d$ — alignment scales with the subspace's dimension — and the
projector build's own resolution order makes $U_\text{neg}$ the doubly-shrunk
bucket. `claims/audits/p6_projector_labels.json` measures
$\dim U_A / \dim U_\text{neg} = 24.9$ at `albert-xlarge-v2`'s exact shape,
against an observed alignment ratio of $0.887/0.067 = 13.2$. **The dimension
correction is larger than the effect it would explain.** So the recorded
inversion is not weak evidence against the prediction; it is not evidence either
way, and no choice of exchangeable unit would have rescued it.

The apparatus is nonetheless live. Phase 6's projector path was **rebuilt** in
`p6_subspace/` against `core/particles.py` (`archive/README.md` rule 2 — nothing
is salvaged by copying), which is what taking the two entries out of `dormant`
required, and both now carry a matched-dimension random-subspace null. They are
`e-value` and `active`. **No p-value is emitted**: no run artifacts exist here,
and no exchangeable unit is registered.

Three things generalise from it:

**A prerequisite is not a footnote.** `status-6.md` item 5 listed a projector
mislabelling as a live alternative explanation and `design-6.md` pre-registered
ruling it out *first*. Nothing had. `tools/audit_p6_projector_labels.py` does,
and reports RULED-OUT — but the audit's own sensitivity arm caught its first
labelling check being **incapable of failing** on one of the two bug classes,
which is the entire reason to build a sensitivity arm.

**The choice of null can dissolve a problem the choice of unit cannot.** Under a
`CLAIM-C`-style sign-flip enumeration the coarsest honest unit here is "one
model", $n = 1$, attainable floor $2/(2^1+1) = 0.667$ — the design refuses on a
perfect result. Randomising over **subspaces** instead of over units leaves
$n = 1$ untouched, because the resolution floor becomes $1/(\text{draws}+1)$.
Several queued rows are small-$n$ designs and should ask this before concluding
they are underpowered.

**The unit still decides validity, and the cost is measurable.** Measured at 400
replicates: with independent per-layer directions both units sit at 0.0525; as
the layers come to share one direction the per-layer unit rises to 0.0800,
0.2325 and 0.2800 while the per-model unit holds at 0.045–0.0575. Which one may
enter an e-process is left unregistered, and `adjudicate_p6_r2_r4` refuses while
it is.

## The table

| id | claim | state | relevance | null construction, or why none exists |
|---|---|---|---|---|
| `CLAIM-A` | H-RESIST | **needs-null** | 1.0 | Needs a null over the three pass criteria jointly. Each is currently a threshold comparison; a matched random-weight baseline (core/nulls.shuffled_dimension_null) gives a per-criterion p, but combining three into one prediction needs the combination fixed in advance. |
| `P-gamma1` | H-RESIST | **needs-null** | 1.0 | beta_reduction.py reports a residual BRACKET across beta in [0.5,5] rather than a point estimate. A p-value needs the bracket converted to a test: bootstrap over prompts, with the reading restated per UPDATE_PLAN.md 5.3 (the calibrated step makes the residual rate-invariant, so it measures direction not magnitude). |
| `P-gamma2` | H-RESIST | **needs-null** | 1.0 | Point estimate against a constant. A bootstrap over the eight prompts gives a p-value; n=8 is small and the record must say so rather than bury it. UPDATE_PLAN.md 5.2: three step-size definitions are computed and verdict() refuses a call when they straddle t*, which must be honoured before any p is emitted. |
| `P-H1` | H-RESIST | **measurement** | 0.8 | NONE, and deliberately so. Wendel's theorem gives probability 1 for d > n, which all 8 prompts satisfy; UPDATE_PLAN.md 5.7 already records the boolean as 'nearly vacuous'. Emitting an e-value here would manufacture evidence out of a theorem. The reportable object is the MARGIN max_w min_i <x_i, w> and the layer at which it first crosses zero. |
| `P-S1` | H-RESIST | **e-value** | 1.0 | Monte-Carlo permutation against a matched i.i.d. baseline at the trained configuration's own (m, d). STATISTIC, fixed in advance: the sum over degrees k=1..3 of the (step0 - trained) Q_k ratio, each standardised by that degree's own null standard deviation. ALTERNATIVE: one-sided 'greater' (P-S1 predicts trained ratios SMALLER, so step0 - trained > 0). NULL: two independent i.i.d. configurations at the same (m, d), same statistic -- a pair per draw, so the null carries both arms' sampling variability. Resolution floor 1/(n_draws+1). Implemented in p1c_frames/centroids.py::p_value_p_s1. |
| `CLAIM-C` | H-TRANSFER | **e-value** | 1.0 | Sign-concordance of the trained-minus-random CONTRAST, built 2026-08-24 in p1_mstate_tracking/replication_gate.py and fixed BEFORE any gate data exists. STATISTICS: the six per-layer series of CHECKPOINT_METRICS -- mass_near_1, effective_rank, cluster_membership, cluster_count, cka_prev, fiedler_mean -- each resampled onto a common normalized-depth grid of 32 points, since gpt2-large has 36 layers and pythia-1.4b has 24. CELL: for each (metric, prompt), delta = mean over normalized depth of (trained - random); the cell scores concordant when sign(delta_pythia) == sign(delta_gpt2). STATISTIC: the count of concordant cells, one-sided 'greater'. SECOND AGREEMENT AXIS (added 2026-08-24, still before any gate data exists): the whole test is re-run once per METRIC-LEAVE-ONE-OUT subset as well as on the full set, and the gate requires UNANIMITY in both directions -- TRANSFERS needs every subset to clear, FAILS-TO-TRANSFER needs every subset to show the inversion, anything mixed is INSUFFICIENT. The reported p is the INTERSECTION-UNION max over the seven runs, which is a valid p for the conjunction regardless of how dependent the runs are, so no multiplicity correction is needed -- and that matters because six leave-one-out runs share five sixths of their data. Prompt eligibility is decided once from the full six-metric requirement and leave-one-out drops COLUMNS only, so every subset shares one prompt set and one null size and the max compares like with like. The rule is 'no subset may fail', NOT 'no metric may dissent': five of six metrics inverting on every prompt survives leave-one-out and is correctly a falsification, because no single metric is carrying it. What the axis catches is a verdict that evaporates when one metric is dropped. The claim axis (gpt/pythia) and the instrument axis (metrics) are kept as separate factors over a verdict lattice rather than folded into one number, because a single number cannot distinguish 'the phenomenology does not transfer' from 'one of our six measurements is quirky' and those have opposite consequences for the sweep. The attainable floor is unchanged by the axis; the cost is power, and that is the point. FORM: a difference test, not an equivalence test -- see the cost below. NULL: CLAIM-C's own falsifier, realised by permuting the trained/random condition label on the candidate side with gpt2-large held fixed as the reference phenomenology. The label attaches to a RUN, so the exchangeable unit is the PROMPT and a swap negates all six metric contrasts together; six metrics on one prompt are not six independent observations. delta is antisymmetric in (trained, random), so the swap is an exact sign flip and the null is enumerated EXHAUSTIVELY while 2^n_prompts <= 65536 (256 patterns for the eight metastability prompts). TOLERANCE: none. The criterion is ordinal, so no magnitude cut and no per-metric standardisation is needed; cells with a non-finite or exactly-zero delta are dropped as sign-undefined and counted. STOP RULE: three-way. TRANSFERS when p_greater <= alpha; FAILS-TO-TRANSFER when the reciprocal 'less' test rejects, i.e. the contrast systematically inverts; INSUFFICIENT otherwise. The hard stop fires on both of the latter, but ONLY FAILS-TO-TRANSFER is a falsification -- an e-process records insufficient evidence, never a null accepted. Only p_greater is calibrated into an e-value; p_reciprocal is a stop-rule input recorded in the record's notes and never enters H-TRANSFER's product, since two one-sided tests on one statistic would double the claim's Type-I rate. REFUSES rather than degrades when fewer than two prompts survive, when no cell has a sign, or when the enumeration's best attainable p (2/(2^n_prompts + 1)) exceeds alpha -- at four prompts a perfect result gives p = 0.118, and a test that cannot reject on a perfect result reports 'not significant' on nothing, which on a hard-stop claim reads as evidence against transfer. Six prompts is the first workable gate. It also refuses when ANY leave-one-out subset cannot carry a p-value -- a max over a set with an undefined member is undefined, and reporting the rest would silently drop whichever subset was hardest to satisfy -- and when every usable prompt carries the SAME candidate sign pattern: the prompts then contribute one observation and enumerating 2^n patterns over it is the wrong null, not a conservative one. FOUR THINGS THE REGISTERED WORDING LEFT OPEN, decided here and recorded so a later reader is not left to infer them. (1) The criterion adjudicates the CONTRAST, not the two absolute reproductions the statement's words name: a pythia pair whose levels both sit far from gpt2-large's but whose difference has the same sign passes. That is deliberate -- Blog 1's phenomenology is a trained-vs-random contrast -- and its cost is that the criterion is scale-blind. The absolute per-arm profile distances are computed and reported as a diagnostic and enter no p-value. (2) The two-baseline policy PREDICTIONS.md attaches to this claim: the p-value runs on the norm-matched pythia-1.4b-random, which is what the statement names. The true step-0 init is a MANDATORY sensitivity arm, computed and reported beside the result and refused-on-omission, but it does not enter the p-value -- step 0 is CLAIM-A's object and one dataset must not settle two entries. Disagreement in direction between the two baselines is flagged in the record. (3) effective_rank is read from the artifact field effective_rank_normed, not the raw one: status-1.md defect D1 records that the raw field mixes directional collapse with residual-stream norm growth. (4) Full normalized depth, no band restriction -- Blog 1 quotes layers 5-30 of gpt2-large, but a depth band is a choice with as many options as there are bands. KNOWN LIMITATION, and the CALIBRATION CURVE that now bounds it (added 2026-08-24, still before any gate data exists and before any adjudication, so null_construction is not yet frozen). Prompts run on one model share that model's weights, so the rows are not fully independent either. A pythia-wide effect present in every prompt is invisible to the enumeration. The cost was measured rather than assumed: the rejection rate at alpha=0.05 is about 0.015 with independent rows and about 0.34 with identical ones -- the same fourfold-plus inflation POPPER reports when its relevance checker is removed. Those two numbers bounded the ends and left the whole MIDDLE uncontrolled, which is where a real run lands, so the middle is now measured too. CORRECTION: tools/calibrate_claim_c_homogeneity.py simulates H0 offline across the homogeneity range and stores R(h, p) = P(the gate reports a p at or below p GIVEN it reported one at all, under H0, at prompt sign-row homogeneity h) in claims/calibration/claim_c_homogeneity.json. The reported p is max(p_exact, R(sign_homogeneity, p_exact)) -- it ADJUSTS the p that enters the e-value rather than sitting beside it as a diagnostic, and it may only BLUNT the exhaustive enumeration's p, never sharpen it. The asymmetry is the choice: at the independent end the enumeration is genuinely conservative, so the max is a no-op and the exact conditional guarantee survives untouched; at the dependent end the reported number becomes the measured rate. Taking R unconditionally would recover the lost power but would trade an exact guarantee for a simulated one on the claim carrying the hard stop. BOTH DIRECTIONS are corrected -- p_reciprocal decides FAILS-TO-TRANSFER, the branch that writes a falsification into the ledger, so leaving it uncorrected would inflate the outcome it is worst to get wrong -- but only p_greater still enters E, unchanged. H0 FAMILY the rates are rates under: a per-metric candidate-wide sign propensity with prompt rows conditionally independent given it, which is literally the limitation above; three bias shapes are swept (uniform, k-of-six, graded ramp) and each homogeneity bin keeps the WORST-rejecting configuration that reached it, because one scalar summary cannot determine a distribution. Rates are CONDITIONAL ON EMISSION: not conditioning would let the gate look calibrated by refusing, since at high homogeneity most draws hit the identical-rows refusal. NEW REFUSALS, both derived rather than placed. (a) No correction available -- curve missing, at another schema version, measured on another metric set, tabulated for another prompt count, or with no measurement in the bin the run landed in -- refuses, because once the correction is what enters the e-value an uncorrected p is not a degraded answer but a Type-I guarantee asserted on a null already measured to be anticonservative. (b) The CORRECTED attainable floor: if R(h, 2/(2^n_prompts + 1)) already exceeds alpha then a PERFECT result does not survive its own correction and the gate cannot reject however clean the data is. That is the existing attainable-floor refusal one level up, derived from alpha the same way, and it settles the question of whether there is a homogeneity above which the gate refuses rather than corrects WITHOUT introducing a tolerance: no homogeneity constant appears anywhere in the module, and the cut moves when alpha does. Measured, the boundary lands near homogeneity 0.80-0.85 for every tabulated prompt count. VALIDATION: the corrected rejection rate returns to nominal in sample (worst fitted configuration 0.199 -> 0.046 at alpha=0.05) and stays at or below nominal OUT of sample, on a duplicate-prompt mixture family the curve was never fitted to, which is the check that indexing the correction by a scalar summary transfers to a different mechanism of dependence. WHAT THE CURVE DOES NOT COVER: every simulated draw has a complete (prompt x metric) table, so a real run that drops cells has a coarser statistic than anything tabulated and its correction is read off a table measured on a slightly different design; the gate reports n_cells_dropped beside the correction and every record says so. Also note that the 0.015 and 0.34 endpoints above were measured BEFORE the metric-leave-one-out axis existed; with the axis the independent-rows rate is about 0.003, since the reported p is a max over seven subsets. The endpoints are kept as history and the curve is what the code reads. The prompt remains the coarsest unit this design provides -- a coarser one would need independent training runs, which do not exist. |
| `CLAIM-B` | H-EMERGE | **needs-null** | 1.0 | Co-location of two changepoints across a checkpoint sweep. A permutation null over checkpoint order gives a valid p once the changepoint estimator is fixed in advance. |
| `P-T1` | H-OPERATOR | **e-value** | 1.0 | Label-permutation test over the row-2 classification. STATISTIC, fixed in advance: trimodal-rate(row-2 candidates) minus trimodal-rate(controls), with trimodality defined as stable_n_modes == 3. ALTERNATIVE: one-sided 'greater'. NULL: permute the row-2 labels across heads, holding both marginals fixed -- which is exactly the amended falsifier ('trimodality is a property of the activations rather than of the classification'). The control arm is therefore not an add-on, it IS the null. Implemented in p2d_operator_activation/table1_predictions.py::p_value_p_t1. |
| `P-M1` | H-OPERATOR | **e-value** | 1.0 | Permutation test over layers. STATISTIC, fixed in advance: the Pearson correlation between the per-layer MEAN head regime distance and the violation series. ALTERNATIVE: one-sided 'greater' (P-M1 predicts violations concentrate FAR from the gradient-flow condition). NULL: permute the violation series against the regime score, preserving both marginals exactly -- which matters because the violation series is heavily skewed and a parametric correlation test would lean on normality it does not have. Implemented in p2d_operator_activation/gradient_flow_condition.py::p_value_p_m1. |
| `P6-A2` | H-OPERATOR | **needs-null** | 0.8 | Classification agreement between f_rot and head type; needs a permutation null over the head-type labels. |
| `P6-I1` | H-OPERATOR | **e-value** | 1.0 | Already a Mann-Whitney U on f_rot(induction heads) vs f_rot(semantic heads). Valid as it stands; only needs threading through core.adjudication. |
| `P6-I2` | H-OPERATOR | **e-value** | 0.8 | Two-sample test over head pairs; same shape as P6-I1. |
| `P6-R1` | H-OPERATOR | **needs-null** | 0.8 | Threshold on a ratio (R >= 5) with a random-projection reference already named. That reference IS the null; it needs to be sampled rather than used as a single comparison value. |
| `P6-R2` | H-OPERATOR | **e-value** | 1.0 | Matched-dimension random-subspace null, built 2026-08-24 in p6_subspace/r2_r4_null.py and fixed BEFORE any p-value exists. INSTRUMENT: Phase 6's projector path was REBUILT live in p6_subspace/ against core/particles.py and core/nulls.py rather than lifted from archive/p6_subspace/subspace_build.py, per archive/README.md rule 2, which is what taking this entry out of `dormant` requires. PREREQUISITE, SETTLED FIRST: status-6.md item 5 records a projector-construction error (Schur block mislabelling swapping U_neg and U_A) as a live alternative explanation for the recorded inversion, and design-6.md pre-registered ruling it out BEFORE treating the hypothesis failure as established. tools/audit_p6_projector_labels.py does so and commits the record to claims/audits/p6_projector_labels.json: RULED-OUT on two independent routes -- planted structure recovered to 3.3e-08 rad, and bucket sizes matching a classification taken from np.linalg.eigvals without touching the Schur form -- with two deliberate mislabellings caught. STATISTIC, fixed in advance: the mean over layers of (chance-normalized alignment of the cluster-separating direction with U_neg minus the same with U_A). NORMALIZED ON BOTH ARMS, which is the substantive change from the archived comparison: p6_subspace/math-6.md 7.2 records that E[\|\|P_U v\|\|^2] = dim U / d, so raw alignment scales with subspace dimension, and the resolution order (span(U_pos) removed from U_neg, span(U_S) removed from U_A) makes U_neg the doubly-shrunk bucket. The audit measures dim(U_A)/dim(U_neg) = 24.9 at albert-xlarge-v2's exact shape against an observed alignment ratio of 13.2 -- the dimension correction is LARGER than the effect it would explain. ALTERNATIVE: one-sided 'greater' (P6-R2 predicts MORE alignment with the real repulsive channel). NULL: H0-OPERATOR realised directly -- replace the operator-derived subspaces with random subspaces OF THE SAME DIMENSION and recompute. Drawn MUTUALLY ORTHOGONAL, because U_neg and U_A are orthogonal by construction and independently drawn null pairs are not; that mismatch alone put the H0 rejection rate at 0.0875 against a nominal 0.05, in the anticonservative direction and invisible in any single result, and it was found by simulating rather than reasoning (the P-S1 defect of POPPER_PLAN.md 6d, second instance). Corrected, the measured rate is 0.045 at alpha=0.05 over 400 replicates. EXCHANGEABLE UNIT: NOT REGISTERED, and adjudication refuses while it is not. The construction is parameterized over it and computes either -- unit='model' draws one set of subspaces shared across layers, which is what ALBERT's weight-tying literally means (one OV matrix, one Schur decomposition, one projector pair, 49 activation snapshots); unit='layer' draws independently per layer, which is the error status-6.md names. The gap was MEASURED at 400 replicates rather than argued: with independent per-layer directions both units sit at 0.0525; as the layers come to share one direction the layer unit rises to 0.0800 (rho=0.5), 0.2325 (rho=0.9) and 0.2800 (rho=1.0) while the model unit stays at 0.045-0.0575 throughout. The mechanism is that the layer unit averages n independent null draws where the model unit averages n copies of one, so its null is narrower by sqrt(n). The evidence points unambiguously at 'model'; registering it is a separate decision and has not been made, so REGISTERED_EXCHANGEABLE_UNIT is None and adjudicate_p6_r2_r4 raises. Passing unit= does not route around that -- the argument selects what to COMPUTE, the module constant decides what may enter an e-process. ATTAINABLE FLOOR, checked before building rather than after a null result: 1/(N_NULL_DRAWS+1) = 0.0005, two orders below alpha, and the module refuses if it ever exceeds alpha. This REFRAMES the question the plan had posed. Under a CLAIM-C-style sign-flip enumeration the coarsest honest unit is 'one model', n=1, floor 2/(2^1+1) = 0.667, and the design could not reject on a perfect result. Under randomisation over SUBSPACES rather than over units, n=1 is no obstacle at all. The binding constraint was the choice of null, not the choice of exchangeable unit. REFUSES rather than degrades: on an unknown unit (no default, since the two differ by orders of magnitude); on an empty U_neg or U_A (normalized alignment is undefined there and 0.0 would read as 'orthogonal' for 'absent'); on fewer than two clusters after dropping HDBSCAN noise (label < 0); on a non-finite statistic; on a null thinned by failures; and on the attainable floor exceeding alpha. WHAT THIS DOES NOT DO: it does not adjudicate the 2026-04 ALBERT run. That run reported RAW alignments (0.887 with U_A, 0.067 with U_neg) and its statistic is not dimension-normalized, so those numbers are SUPERSEDED as evidence rather than turned into a p-value. Chance-normalized against the audit's measured dims they read 0.960 for U_A and 1.805 for U_neg -- the PREDICTED direction -- but the audit's dims come from random OV matrices at ALBERT's shape and not from ALBERT's trained weights, so that is a bound on the correction and not a result. The actual per-layer dims are computed by the projector build on every run and were never reported; recovering them is one number and it settles the reading. DEPENDENCE ON P6-R4: the two share one projector, so a projector defect moves both. They are NOT the P5b-B1/B3 pattern of one test with two thresholds -- R2 compares alignments, R4 compares probe accuracies, two statistics on two instruments -- and both are registered as adjudicable, but the common-cause dependence is recorded here so a reader does not read their product as two independent factors. |
| `P6-R3` | H-OPERATOR | **needs-null** | 0.8 | Directional dominance at merge events; permutation over merge vs non-merge steps. |
| `P6-R4` | H-OPERATOR | **e-value** | 1.0 | Matched-dimension random-subspace null, built 2026-08-24 in p6_subspace/r2_r4_null.py alongside P6-R2 and fixed BEFORE any p-value exists. Same rebuilt instrument, same prerequisite audit (claims/audits/p6_projector_labels.json), same exchangeable-unit position: NOT REGISTERED, adjudication refuses, and the measured cost of the wrong unit is recorded on the P6-R2 entry. STATISTIC, fixed in advance: the mean over layers of the cross-validated accuracy of a linear probe fit INSIDE U_S, on the projected coordinates rather than the ambient embedding -- projecting back into R^d would hand the probe the dimensions the projection was supposed to remove. ALTERNATIVE: one-sided 'greater'. NULL: the same probe fit inside a random subspace of THE SAME DIMENSION, which is what makes this a test of operator content rather than of capacity; a probe fit in a higher-dimensional subspace has more capacity, and math-6.md 7.2 records that the archived 'imaginary-only 0.564 vs real-only 0.152' comparison has exactly that confound. PROBE: a ridge one-vs-rest linear probe with stratified k-fold, in numpy. NOT sklearn's LogisticRegression, which archive/p6_subspace/probe_subspace.py used. Partly because sklearn is a heavy-tier dependency and this module is pure tier, but mainly because the archived accuracies are not comparable to anything computed here anyway -- they were measured on subspaces of unequal dimension -- so matching the classifier would buy a comparability that is not available. What matters is that the SAME probe scores both arms, and it does. The ridge coefficient is PLACED, not calibrated, and is applied identically to both arms so it cannot move the contrast. POWER CAVEAT, measured: this statistic separates only while U_S is a SMALL fraction of d_model. At dim U_S / d = 14/24 both arms saturate at accuracy 1.0 and the test reports p = 1.0 on a planted effect -- a random subspace of that size captures the signal about as well as the real one. ALBERT's ratio is roughly 150/2048 and at a comparable fraction the arms separate cleanly (planted in U_S: p = 0.016; planted in U_A: p = 1.000). A run where U_S is a large fraction of d_model needs that said, not a p-value. REFUSES rather than degrades: on an unknown unit; on an empty U_S (there is no subspace to probe); on fewer than two clusters after dropping HDBSCAN noise; when the smallest cluster has fewer than two members, since cross-validation cannot be stratified over it; and on the attainable floor exceeding alpha. DEPENDENCE ON P6-R2: recorded on that entry. The shared projector is a common-cause failure mode for both. |
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

Then the `needs-null` rows in the order their claims matter. `CLAIM-C` was
first, because it is the one with a hard stop attached and a stop rule that
cannot be adjudicated is a stop rule that gets argued with at the moment it
binds. **Built 2026-08-24** (`p1_mstate_tracking/replication_gate.py`,
`POPPER_PLAN.md` §6f) — it is now an `e-value` row. `P6-R2` and `P6-R4` followed
the same day (`p6_subspace/r2_r4_null.py`, §6h), which also took them out of
`dormant`.

Two lessons from building them generalise to the rows still queued, several of
which are small-n permutation designs.

**Check the attainable floor before building the null, not after the result
comes back null.** A permutation over n exchangeable units can express no p
smaller than `2/(2^n + 1)` when the null is enumerated exhaustively, so a
four-unit design cannot reject at α = 0.05 even on a perfect result. Reporting
that as "not significant" is worse than reporting nothing, and
`replication_gate` refuses instead.

**But check it against the null you could build, not only the one you reached
for.** `P6-R2`'s floor is 0.667 under a sign-flip enumeration over its one
honest unit and 0.0005 under randomisation over matched-dimension subspaces —
same data, same unit, same claim. `P6-R1`, `P6-C1`, `P5b-A1`, `P5b-A2`,
`P-SA1` and `P-I4` are all threshold or two-sample rows whose predictions
already name a matched control, and that control is a subspace or a magnitude
rather than a unit. The floor argument that retires a design should be made
after asking what is being randomised.

`CLAIM-B` is next by the same reasoning, and it shares a construction with
`P-I1` — the same changepoint co-location across a checkpoint sweep — so the
two should be built together rather than each inventing one.
