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
| `e-value` | 13 | yes |
| `needs-null` | 22 | not yet |
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
| `P-S1` | H-RESIST | **e-value** | 1.0 | Monte-Carlo permutation against a matched i.i.d. baseline at the trained configuration's own (m, d). STATISTIC, fixed in advance: the sum over degrees k=1..3 of the (step0 - trained) Q_k ratio, each standardised by that degree's own null standard deviation. ALTERNATIVE: one-sided 'greater' (P-S1 predicts trained ratios SMALLER, so step0 - trained > 0). NULL: two independent i.i.d. configurations at the same (m, d), same statistic -- a pair per draw, so the null carries both arms' sampling variability. Resolution floor 1/(n_draws+1). Implemented in p1c_frames/centroids.py::p_value_p_s1. DRY RUN 2026-08-27 (claims/audits/p_s1_dry_run.json, POPPER_PLAN.md 6p), AND THE GATE GAINED AN (m, d) REFUSAL. p_value_p_s1 takes m and d from the TRAINED arm, draws its null there and re-references BOTH arms against that one baseline -- 6d's fix, correct when the arms sit at the same configuration, and nothing checked that they did. E[Q_k] = 1/m for i.i.d. points, so the baseline scales like 1/m and a step-0 arm at a different cluster count is divided by a baseline that is not its own; its ratio is off by roughly m_trained/m_step0, which enters the statistic as a DIFFERENCE BETWEEN THE ARMS -- the exact shape of the effect P-S1 predicts. Measured on TWO I.I.D. ARMS, where the correct verdict is 'no difference' at every row: 32 against 30 clusters rejects at 1.000, as do 28 and 24; 32 against 36 and 40 return p = 1.000 so the design can never win. TWO CLUSTERS OUT OF THIRTY-TWO is enough, and the error runs both ways -- fewer step-0 clusters CONFIRMS the prediction. Unequal counts are the expected case, since clustering runs per checkpoint and a random-weight model's geometry is not a trained one's. THE REFUSAL: the gate refuses when the two arms report different (m, d), and refuses when the step-0 arm does not report them at all. A degeneracy and not a tolerance -- the counts are equal or they are not -- and it costs nothing BY CONSTRUCTION rather than by measurement, because Q_k's i.i.d. floor depends on m so 'closer to a spherical design' is not a comparison that exists across different m and no baseline choice rescues the row. SIXTH PRE-COMPUTED REQUIREMENT ON A RUN, and the first about how it is CLUSTERED: both arms must be clustered to the same count rather than each to its own best k. THE FALLBACK NOTE WAS ALSO CORRECTED: it warned the Q_ratio path is 'mildly anticonservative' and cited a null-p mean of 0.40. That was measured on pre-2026-08-24 code; the statistic is now a DIFFERENCE of two ratios formed against the SAME caller baseline, so a common per-degree factor cancels and the two paths measure 0.508 against 0.507 with identical rejection rates. The note stopped describing its own path and nothing noticed. WHAT DID NOT CHANGE, checked rather than assumed: the reported 1/(n_null+1) floor IS attainable here, because the statistic is continuous and ties have probability zero -- the claim that failed for P-ST1, P-T1 and P-M1, all discrete. |
| `CLAIM-C` | H-TRANSFER | **e-value** | 1.0 | Sign-concordance of the trained-minus-random CONTRAST, built 2026-08-24 in p1_mstate_tracking/replication_gate.py and fixed BEFORE any gate data exists. STATISTICS: the six per-layer series of CHECKPOINT_METRICS -- mass_near_1, effective_rank, cluster_membership, cluster_count, cka_prev, fiedler_mean -- each resampled onto a common normalized-depth grid of 32 points, since gpt2-large has 36 layers and pythia-1.4b has 24. CELL: for each (metric, prompt), delta = mean over normalized depth of (trained - random); the cell scores concordant when sign(delta_pythia) == sign(delta_gpt2). STATISTIC: the count of concordant cells, one-sided 'greater'. SECOND AGREEMENT AXIS (added 2026-08-24, still before any gate data exists): the whole test is re-run once per METRIC-LEAVE-ONE-OUT subset as well as on the full set, and the gate requires UNANIMITY in both directions -- TRANSFERS needs every subset to clear, FAILS-TO-TRANSFER needs every subset to show the inversion, anything mixed is INSUFFICIENT. The reported p is the INTERSECTION-UNION max over the seven runs, which is a valid p for the conjunction regardless of how dependent the runs are, so no multiplicity correction is needed -- and that matters because six leave-one-out runs share five sixths of their data. Prompt eligibility is decided once from the full six-metric requirement and leave-one-out drops COLUMNS only, so every subset shares one prompt set and one null size and the max compares like with like. The rule is 'no subset may fail', NOT 'no metric may dissent': five of six metrics inverting on every prompt survives leave-one-out and is correctly a falsification, because no single metric is carrying it. What the axis catches is a verdict that evaporates when one metric is dropped. The claim axis (gpt/pythia) and the instrument axis (metrics) are kept as separate factors over a verdict lattice rather than folded into one number, because a single number cannot distinguish 'the phenomenology does not transfer' from 'one of our six measurements is quirky' and those have opposite consequences for the sweep. The attainable floor is unchanged by the axis; the cost is power, and that is the point. FORM: a difference test, not an equivalence test -- see the cost below. NULL: CLAIM-C's own falsifier, realised by permuting the trained/random condition label on the candidate side with gpt2-large held fixed as the reference phenomenology. The label attaches to a RUN, so the exchangeable unit is the PROMPT and a swap negates all six metric contrasts together; six metrics on one prompt are not six independent observations. delta is antisymmetric in (trained, random), so the swap is an exact sign flip and the null is enumerated EXHAUSTIVELY while 2^n_prompts <= 65536 (256 patterns for the eight metastability prompts). TOLERANCE: none. The criterion is ordinal, so no magnitude cut and no per-metric standardisation is needed; cells with a non-finite or exactly-zero delta are dropped as sign-undefined and counted. STOP RULE: three-way. TRANSFERS when p_greater <= alpha; FAILS-TO-TRANSFER when the reciprocal 'less' test rejects, i.e. the contrast systematically inverts; INSUFFICIENT otherwise. The hard stop fires on both of the latter, but ONLY FAILS-TO-TRANSFER is a falsification -- an e-process records insufficient evidence, never a null accepted. Only p_greater is calibrated into an e-value; p_reciprocal is a stop-rule input recorded in the record's notes and never enters H-TRANSFER's product, since two one-sided tests on one statistic would double the claim's Type-I rate. REFUSES rather than degrades when fewer than two prompts survive, when no cell has a sign, or when the enumeration's best attainable p (2/(2^n_prompts + 1)) exceeds alpha -- at four prompts a perfect result gives p = 0.118, and a test that cannot reject on a perfect result reports 'not significant' on nothing, which on a hard-stop claim reads as evidence against transfer. Six prompts is the first workable gate. It also refuses when ANY leave-one-out subset cannot carry a p-value -- a max over a set with an undefined member is undefined, and reporting the rest would silently drop whichever subset was hardest to satisfy -- and when every usable prompt carries the SAME candidate sign pattern: the prompts then contribute one observation and enumerating 2^n patterns over it is the wrong null, not a conservative one. FOUR THINGS THE REGISTERED WORDING LEFT OPEN, decided here and recorded so a later reader is not left to infer them. (1) The criterion adjudicates the CONTRAST, not the two absolute reproductions the statement's words name: a pythia pair whose levels both sit far from gpt2-large's but whose difference has the same sign passes. That is deliberate -- Blog 1's phenomenology is a trained-vs-random contrast -- and its cost is that the criterion is scale-blind. The absolute per-arm profile distances are computed and reported as a diagnostic and enter no p-value. (2) The two-baseline policy PREDICTIONS.md attaches to this claim: the p-value runs on the norm-matched pythia-1.4b-random, which is what the statement names. The true step-0 init is a MANDATORY sensitivity arm, computed and reported beside the result and refused-on-omission, but it does not enter the p-value -- step 0 is CLAIM-A's object and one dataset must not settle two entries. Disagreement in direction between the two baselines is flagged in the record. (3) effective_rank is read from the artifact field effective_rank_normed, not the raw one: status-1.md defect D1 records that the raw field mixes directional collapse with residual-stream norm growth. (4) Full normalized depth, no band restriction -- Blog 1 quotes layers 5-30 of gpt2-large, but a depth band is a choice with as many options as there are bands. KNOWN LIMITATION, and the CALIBRATION CURVE that now bounds it (added 2026-08-24, still before any gate data exists and before any adjudication, so null_construction is not yet frozen). Prompts run on one model share that model's weights, so the rows are not fully independent either. A pythia-wide effect present in every prompt is invisible to the enumeration. The cost was measured rather than assumed: the rejection rate at alpha=0.05 is about 0.015 with independent rows and about 0.34 with identical ones -- the same fourfold-plus inflation POPPER reports when its relevance checker is removed. Those two numbers bounded the ends and left the whole MIDDLE uncontrolled, which is where a real run lands, so the middle is now measured too. CORRECTION: tools/calibrate_claim_c_homogeneity.py simulates H0 offline across the homogeneity range and stores R(h, p) = P(the gate reports a p at or below p GIVEN it reported one at all, under H0, at prompt sign-row homogeneity h) in claims/calibration/claim_c_homogeneity.json. The reported p is max(p_exact, R(sign_homogeneity, p_exact)) -- it ADJUSTS the p that enters the e-value rather than sitting beside it as a diagnostic, and it may only BLUNT the exhaustive enumeration's p, never sharpen it. The asymmetry is the choice: at the independent end the enumeration is genuinely conservative, so the max is a no-op and the exact conditional guarantee survives untouched; at the dependent end the reported number becomes the measured rate. Taking R unconditionally would recover the lost power but would trade an exact guarantee for a simulated one on the claim carrying the hard stop. BOTH DIRECTIONS are corrected -- p_reciprocal decides FAILS-TO-TRANSFER, the branch that writes a falsification into the ledger, so leaving it uncorrected would inflate the outcome it is worst to get wrong -- but only p_greater still enters E, unchanged. H0 FAMILY the rates are rates under: a per-metric candidate-wide sign propensity with prompt rows conditionally independent given it, which is literally the limitation above; three bias shapes are swept (uniform, k-of-six, graded ramp) and each homogeneity bin keeps the WORST-rejecting configuration that reached it, because one scalar summary cannot determine a distribution. Rates are CONDITIONAL ON EMISSION: not conditioning would let the gate look calibrated by refusing, since at high homogeneity most draws hit the identical-rows refusal. NEW REFUSALS, both derived rather than placed. (a) No correction available -- curve missing, at another schema version, measured on another metric set, tabulated for another prompt count, or with no measurement in the bin the run landed in -- refuses, because once the correction is what enters the e-value an uncorrected p is not a degraded answer but a Type-I guarantee asserted on a null already measured to be anticonservative. (b) The CORRECTED attainable floor: if R(h, 2/(2^n_prompts + 1)) already exceeds alpha then a PERFECT result does not survive its own correction and the gate cannot reject however clean the data is. That is the existing attainable-floor refusal one level up, derived from alpha the same way, and it settles the question of whether there is a homogeneity above which the gate refuses rather than corrects WITHOUT introducing a tolerance: no homogeneity constant appears anywhere in the module, and the cut moves when alpha does. Measured, the boundary lands near homogeneity 0.80-0.85 for every tabulated prompt count. VALIDATION: the corrected rejection rate returns to nominal in sample (worst fitted configuration 0.199 -> 0.046 at alpha=0.05) and stays at or below nominal OUT of sample, on a duplicate-prompt mixture family the curve was never fitted to, which is the check that indexing the correction by a scalar summary transfers to a different mechanism of dependence. THE CELL-DROP DIMENSION (2026-08-25): the curve no longer assumes a complete (prompt x metric) table. It used to, and a real run that dropped cells -- a non-finite or exactly-zero contrast in either architecture, which the ordinal criterion has to drop -- read its correction, and therefore its refusal, off a table measured on a design it does not have. Dropping cells is not the same statistic made noisier: the sum runs over fewer cells, the per-row null weights valid_i stop being equal, and a row can lose its SWING entirely. The curve is now indexed by (n_prompts, drop fraction, homogeneity). Drop bin 0 is n_cells_dropped == 0 EXACTLY, tested as integers rather than as a float against a tolerance; the remaining bins run to a tabulated ceiling above which the gate REFUSES rather than reading the nearest row. Three drop mechanisms reach each rate -- every cell independently, concentrated in as few metrics as the rate allows, and concentrated in as few prompts, which is the severe one since a rate above 1/n_prompts empties a row outright -- and each (homogeneity, drop) cell keeps the WORST configuration over the bias shapes and mechanisms together, which is the same rule the bias family already used. NOTHING IS FILLED ACROSS THE DROP DIMENSION, and that is a measurement rather than caution: coarsening pushes p-values up while selecting for tables that survived the informative-row floor pushes the conditional rate down, and measured, at eight prompts 93 of 117 adjacent drop-bin pairs at fixed homogeneity are neither non-decreasing nor non-increasing, and at six prompts 12 rise while 7 fall. A hole in that dimension is a refusal. WHAT THIS FAMILY ASSUMES: drops are independent of concordance GIVEN THE POSITION -- which cells go is modelled, whether a surviving cell agrees is not conditioned on it -- so a mechanism preferentially removing discordant cells is outside it. THE FLOOR IS SET BY THE ROWS THAT CAN MOVE (2026-08-25): a third refusal, and the one the drop dimension made pointed. Flipping prompt i's label swaps its concordant and discordant cells, so row i's SWING is \|valid_i - 2 conc_i\| and a row with swing 0 contributes the same number to the observed sum and to every one of the 2^n null patterns -- it is enumerated and never counted. With k rows that do move, the smallest expressible p is (2^(n-k) + 1)/(2^n + 1) ~ 2^-k, which is 2/(2^n + 1) exactly when k = n, so the floor the module already refused on is the special case rather than a second rule. Five informative rows is the first count that clears alpha = 0.05 AT EVERY PROMPT COUNT -- the same k >= 5 as P-ST1's informative-pair floor. Two ways a row lands there: all its cells dropped, or an EVEN number of usable cells splitting exactly half and half, which with six metrics is three concordant and three not and happens to 20/64 of rows under H0. The second was live in this gate from the day it was written and nothing looked for it. It does not bite on the five-metric leave-one-out subsets of a complete table -- five is odd and an odd swing cannot be zero -- so the full set is the binding subset until drops make a subset's usable count even or empty it. WHY A DATA-DEPENDENT REFUSAL IS SAFE HERE: the null is symmetric under a global flip, so both tails share the floor; when it exceeds alpha neither p_greater nor p_reciprocal can reach alpha, TRANSFERS and FAILS-TO-TRANSFER are both unreachable, and the verdict was INSUFFICIENT whatever the statistic came to. The refusal removes no verdict. It replaces a p above alpha -- which on this claim reads as evidence against CLAIM-C -- with a record saying the design could not have rejected, which is the tightness argument of (e) below applied to the refusal that was missing, and it is measured rather than restated. Also note that the 0.015 and 0.34 endpoints above were measured BEFORE the metric-leave-one-out axis existed; with the axis the independent-rows rate is about 0.003, since the reported p is a max over seven subsets. The endpoints are kept as history and the curve is what the code reads. The prompt remains the coarsest unit this design provides -- a coarser one would need independent training runs, which do not exist. MEASURED OPERATING RANGE (2026-08-25, dry run on inputs with known answers; still no gate data, still no adjudication, so null_construction is not yet frozen). The construction above is unchanged; what follows is what it DOES, measured by running it rather than reasoned about. tools/dry_run_claim_c.py puts one synthetic model in as BOTH the reference and the candidate -- every cell is then concordant and the correct verdict is TRANSFERS a priori -- and sweeps the candidate's own sign homogeneity across every attainable value at each tabulated prompt count. Record: claims/audits/claim_c_dry_run.json. (a) THE CRITERION IS SOUND ON A PERFECT INPUT: the observed statistic is the maximum of its own null in the full set and in all six leave-one-out subsets, so each returns exactly 2/(2^n + 1) and the intersection-union max is that same floor. The unanimity rule does not bite on a unanimous input. (b) THE GATE HAS AN ADMISSIBLE BAND, and outside it it is a CONSTANT FUNCTION. At eight prompts the band is sign_homogeneity <= 0.7708, which is at least 11 of the 48 candidate cells carrying the minority sign for their metric. (It was <= 0.8125 and 9 cells when first measured on 2026-08-25; the band TIGHTENED later the same day when the informative-row floor changed what 'conditional on emission' conditions on. A draw whose rows could not move the statistic used to be emitted with a p above alpha and counted as a non-rejection, diluting R downward; it is now refused, reaches no ledger and belongs in no denominator, so the rate among draws that do reach one is higher, the correction is stronger, and the derived refusal bites at a lower homogeneity. Both curves were internally consistent; only the new one describes the gate that exists. The refusal that costs no power on any individual table does cost BAND, and that is the price of this pass.) Above it the corrected-attainable-floor refusal fires on EVERY input including a perfect one (measured at every concordance count from 0 to 48), so neither TRANSFERS nor FAILS-TO-TRANSFER is reachable and the hard stop fires unconditionally. That is not a Type-I defect and not an argument for a weaker correction: sign_homogeneity is a within-candidate statistic, under H0 it measures the prompt redundancy the curve corrects for and under H1 the same number rises with the strength and UNIFORMITY of a real effect, and the correction cannot tell the two apart. The cost lands as power and it lands hardest where the effect is most uniform. (c) THE SCALE THE BAND IS READ AGAINST. Under independent prompt signs -- the most favourable candidate the design can be handed -- homogeneity concentrates at 0.637 at eight prompts and the refusal fires with probability 1e-4, so the band is not tight against chance. It is tight against a clean effect: a contrast pointing the same way on every prompt sits at exactly 1.0 and is refused with certainty. So this is A REQUIREMENT ON WHAT THE PILOT MUST MEASURE, computed before it runs -- the same shape as CLAIM-B's 19 control series: at least ~23% of the candidate's 48 cells must dissent in sign, and whether they do is an empirical fact nothing here yet knows. More prompts do not supply it: expressed as the curve bin the refusal starts in, the boundary is 0.775-0.800 at six, seven and eight prompts and 0.825-0.850 at nine and up -- two bins of 0.025, every count having moved down together when the band tightened. The share of the independent-prompt distribution above the boundary is 0.0017 at eight prompts and 0.026 at six, which is the clearest statement yet that six prompts is the marginal design. (d) THE DERIVED REFUSAL IS TIGHT. R(h, .) is non-decreasing in p in all 1914 tabulated bins, the drop dimension included, so R(h, floor) > alpha implies R(h, p) > alpha for every attainable p: whenever it fires, no input could have cleared alpha. It never costs a verdict the gate could otherwise have reached. (e) HOW MUCH CONCORDANCE THE GATE NEEDS, at eight prompts and 48 cells, over randomly placed arrangements at a fixed candidate sign table. TRANSFERS reaches 50% at 35 of 48 concordant cells at homogeneity 0.5833, 35 at 0.7083 and 38 at 0.7708, and becomes certain at 38, 38 and 43 respectively -- so the requirement TIGHTENS as the candidate's contrast becomes more uniform. FAILS-TO-TRANSFER reaches 50% at or below 14 concordant cells. Between them sits an INSUFFICIENT band of 27 of the 49 possible concordance counts at homogeneity 0.7708, and the hard stop fires across all of it. Decomposed against two counterfactual rates in the same record: the metric-leave-one-out axis moves the 50% point by 1-3 cells and the homogeneity correction by 0 well inside the band rising to 4 at its edge. (The levels are placed relative to the boundary, so they moved down with it.) |
| `CLAIM-B` | H-EMERGE | **e-value** | 1.0 | Changepoint co-location on the log-step axis, built 2026-08-24 in core/changepoint_colocation.py and fixed BEFORE any sweep data exists. SHARED CONSTRUCTION: EVALUABILITY.md named CLAIM-B and P-I1 as sharing one construction and said they should be built together rather than each inventing one; they are. They sit under DIFFERENT claims (H-EMERGE and H-BRIDGE) so there is no P5b-B1/B3 double-counting problem, but one shared ESTIMATOR is a common-cause failure mode -- an estimator defect moves both -- and it is recorded here rather than left inferable, the same way P6-R2 and P6-R4 record their shared projector. ESTIMATOR, and why it is NOT detect_transitions: the existing estimator in core/checkpoint_frames.py returns the INTERVALS of largest change per unit log-step. Adopting it is the reuse this project prefers and it was checked first. It cannot carry this test, for a reason that only appears once the attainable floor is computed: the log-step geometry is not uniform (Pythia's every-1000 releases compress to d log10(step+1) = 0.065 at the top of a 25-checkpoint sweep against 0.301 at the bottom) and interval_rates divides by that spacing, so under a permutation of the value series against the fixed step grid the argmax interval lands on the smallest-spacing interval 44.7% of the time. A BINARY co-location statistic -- 'the two top intervals coincide' -- therefore has a best attainable p of about 0.29 typical and 0.45 worst case and cannot reject at any sensible alpha however clean the data is. detect_transitions also takes n_top and min_abs, both selections if set after seeing the sweep. WHAT REPLACES IT: a CHANGE-MASS PROFILE. For a series v at steps s with a REGISTERED direction, w_i is proportional to max(direction * (v_{i+1} - v_i), 0), normalised over the sweep's intervals -- the share of the series' total registered-direction change that happened in interval i -- and the location is that distribution's centroid on the log-step axis. It carries NO placed constant: no n_top, no min_abs, no tolerance on what counts as co-located, no smoothing bandwidth. EVALUABILITY.md asked whether there was an ordinal formulation needing none, the way CLAIM-C's sign-concordance avoided a magnitude cut; this is it, a distance in log10-step compared against a null. The profile is NOT divided by the log-step spacing, which is a departure from checkpoint_frames: rate weighting is equally VALID (H0 rejection 0.043-0.073 either way, measured under the pairing null actually used) but their POWER diverges as the sweep densifies: at 8 units and alpha=0.05 change mass holds 1.000 from 20 to 143 checkpoints while rate falls 0.995, 0.970, 0.685, 0.090 over 20, 35, 80 and 143, because dividing by dx amplifies per-checkpoint noise exactly where the spacing is tight and a denser sweep makes every dx tighter. The log-step axis is right for plotting a derivative, which is what checkpoint_frames built it for, and wrong for weighting a location; a change-mass profile takes no derivative, so spacing_change_steps' 'an index-based derivative places a peak here by construction' cannot reach it, and the spacing report is emitted in every record anyway so a reader checks that rather than taking it. Dispersion is reported beside every centroid so a bimodal change profile -- whose centroid sits between two changes and means much less -- is visible. STATISTIC: minus the distance between two change centroids in log10-step (negated so 'greater' is the predicted direction), and for the anchor arms minus the distance from a centroid to the pre-registered step window, zero inside it. ALTERNATIVE: one-sided 'greater', fixed in advance. THE NULL IS NOT A PERMUTATION OVER CHECKPOINT ORDER, and the registered wording that reached for one was measured to be wrong. Four permutation-family nulls were built and their H0 rejection rate measured against a nominal 0.05: permuting the value series against the fixed step grid 0.45, permuting the interval increments 0.32, a SAMPLED circular shift of the increments 0.13, the same shift ENUMERATED over its m rotations 0.065. The first three are anticonservative for one reason -- the statistic is built on a concentrated profile and those nulls dissolve the concentration, so the null's variance is far too small and any partial overlap of two real profiles reads as significant. (The sampled circular shift is additionally wrong in a way worth naming: m rotations are not m independent draws, so sampling 199 of them and dividing by 200 understates p.) The enumerated shift is valid only if changepoints are uniform on the interval grid, and they are not: with both series' onsets drawn early -- 'everything moves early in training' -- it rejects at 0.103, twice nominal. NULL ACTUALLY USED: a MATCHED CONTROL SERIES, where the control for series B at unit u is series B AT ANOTHER UNIT -- same metric, same construction, same sweep -- and those controls are combined across units as a permutation of the PAIRING between the two series' units. Under H0 the two series' per-unit locations are independent, so which unit of A is paired with which unit of B is arbitrary and the permutation is exact. It also disposes of the common-trend confound for free, because both series keep their real per-unit locations under every permutation. Making it a permutation over PAIRINGS rather than one test per unit is what keeps this from being the 'n layers are not n independent observations' error status-6.md records. Enumerated exhaustively at or below 5040 pairings (7 units) and sampled with the +1 rule above it; the identity pairing is included either way, which is what makes the smallest attainable p 1/P rather than 0. EXCHANGEABLE UNIT: the LAYER. ARMS, and the unanimity rule: CLAIM-B's statement names two co-locations at once, so three arms are run -- the mutual arm (energy break against Fiedler drop, paired over layers) and one anchor arm per series against the window. The reported p is the INTERSECTION-UNION MAX over the three, which is a valid p for a conjunction regardless of dependence, so no multiplicity correction is needed and that matters because the arms share two series between them. Same precedent as CLAIM-C's metric-leave-one-out axis. Both directions are combined the same way: CO-LOCATES needs every arm to clear and RE-ANCHORS needs every arm to show the separation. A third axis is affordable here in a way it is not on CLAIM-C, because CLAIM-B carries no hard stop. SERIES AND DIRECTIONS, registered: the energy-monotonicity BREAK is read as a RISE in core.metrics.energy_violation_severity()['sum_severity'], not in n_violations -- the count is an integer with heavy ties and a tied series puts its change mass on whichever interval happens to cross an integer boundary, while severity is the magnitude and 'break' is a statement about magnitude (same class of decision as CLAIM-C reading effective_rank_normed rather than the raw field). The Fiedler DROP is read as a drop in CHECKPOINT_METRICS['fiedler_mean']. ANCHOR WINDOW: steps 512 to 2000, taken from CLAIM-B's own registered statement and not chosen by the module. Standing rule 6 asks where a constant came from and the answer is a citation to the prediction itself. ATTAINABLE FLOOR, checked before building rather than after a null result: the mutual arm's floor is 1/(pairings), 0.0005 at the sampled size, but each ANCHOR arm's floor is 1/(n_controls + 1), so alpha = 0.05 needs 19 control series measured on the same sweep at the same layers. A cheap-tier sweep measuring six metrics has six and the anchor arms REFUSE. That is a requirement on the pilot, computed before it runs, and it is the most likely reason this gate returns no number. STOP RULE: three-way. CO-LOCATES when p_greater <= alpha; RE-ANCHORS when the reciprocal 'less' test rejects, i.e. the changes sit demonstrably FURTHER apart than the matched controls; INSUFFICIENT otherwise. Only RE-ANCHORS is a falsification, and CLAIM-B's falsifier is why the branch exists at all -- 'No co-location. Itself a real result: it re-anchors the 1.4B schedule rather than invalidating the sweep' -- so it is recorded as positively shown rather than inferred from a failure to reject. Only p_greater is calibrated into H-EMERGE's product; p_reciprocal is a stop-rule input in the record's notes, since two one-sided tests on one statistic would double the claim's Type-I rate. REFUSES rather than degrades: on an unregistered change direction (no default, since CLAIM-B names a DROP and P-I1 a RISE and a default would score one as the other's absence); on a series with no change in the registered direction (a uniform profile would report the change as spread evenly over training rather than as absent); on non-finite or unsorted steps or values; on fewer than three checkpoints; on fewer than two units; on a control family that is not the registered one; on every pairing or every control giving the identical statistic (the units then contribute one observation, which is the wrong null and not a conservative one -- a degeneracy and not a tolerance, so no threshold is placed); on any arm that cannot carry a p-value, since a max over a set with an undefined member is undefined and reporting the rest would silently drop whichever arm was hardest to satisfy; and on any attainable floor exceeding alpha. CALIBRATION, measured offline and committed to claims/calibration/changepoint_colocation.json, pinned by the pure tier: at 8, 16 and 24 units the H0 rejection rate is at nominal both when the two series' onsets are independent AND under the common early trend that defeats every permutation-over-order null; a deliberately anti-aligned pairing returns p = 1.000 with the reciprocal test firing, which is what makes RE-ANCHORS a branch that can actually happen rather than a verdict that cannot fail. THE LIMITATION THAT DOES NOT GO AWAY, measured rather than described: the pairing null tests ASSOCIATION, and a common per-unit factor -- a layer that changes late changing late in BOTH series for a reason unrelated to the claim -- is an association. The measured rejection rate under exactly that is 1.00 against 0.05 when the two are independent. No null over the pairing separates them, because a confound present at every unit is present under every permutation. Every record therefore carries a shared_unit_factor_diagnostic -- each series' rank correlation with the unit index -- which catches a confound MONOTONE in that index and catches nothing else, and the analyst must name the independence source. The honest fix is a confound-control arm testing co-location against other per-unit series, and it needs the same 19 control series the anchor arms need; it is not built. WHAT NO NULL HERE CAN DO: the sweep's resolution is its intervals, two changes inside one interval are one change to this construction, and no choice of statistic recovers what was not sampled -- the honest content of detect_transitions' docstring, which survives the change of estimator. STILL NO DATA: the apparatus exists and the artifacts do not. INDEX.md records the dense pilot sweep as not executed, validation is on synthetic inputs with known answers, and claims/adjudications/ is empty. DRY RUN 2026-08-27 (claims/audits/claim_b_p_i1_dry_run.json, POPPER_PLAN.md 6o), shared with P-I1 because the estimator is shared, and THE ANCHOR ARMS CHANGED. A change location is the centroid of a change-mass profile, which is a weighted mean of the sweep's interval midpoints -- so mass spread evenly over the sweep lands on the GRID'S OWN MIDPOINT exactly, and per-checkpoint noise makes every real location a mixture of where the series changed and where that midpoint is, weighted by the noise's share of the mass. The share grows with the interval count, so a DENSER sweep is worse; the closed form predicts the measured centroid to 0.061 in log10-step over three grids and four noise levels. THE REGISTERED INSTRUMENT IS THE GRID ON WHICH THAT IS FATAL: a 20-30 checkpoint cheap-tier sweep puts the uniform-profile midpoint at step 955, INSIDE this prediction's own 512-2000 window, so a series that changes NOWHERE attains the anchor arm's maximum statistic. Against controls that all carry a located change the arm rejects on a change-free input at 1.000 -- exactly its rate on a perfectly anchored one, so its discriminating power is ZERO -- and in general at 1/(k+1) with k the number of controls that are themselves change-free, measured against that closed form at every k from 0 to 19. THE REFUSAL: anchor_arm refuses when the change-free reference lands inside the window. The condition reads the step grid against the registered window and nothing else -- no controls, no observation, no alpha -- so it is decidable before a checkpoint is sampled. It was FIRST written as the reference's RANK among the controls and that was wrong: the reference is noiseless and a realised change-free series is not, so the rank pegs at the floor whatever family it is handed (flat at 0.050-0.051 across the whole k sweep) while the rate runs 1.000 to nominal. UNLIKE CLAIM-C's informative-row refusal (measured at zero power cost) AND P-ST1's attainable-floor refusal (costs none by construction), THIS ONE COSTS VERDICTS -- it turns away inputs whose change really is at the anchor, and the dry run re-scores the counterfactual in every cell rather than asserting the cost is small. What it refuses is a verdict the design cannot SUPPORT rather than one it could not REACH. FIFTH PRE-COMPUTED REQUIREMENT ON THE PILOT, and the first about WHICH CHECKPOINTS rather than which metrics: the anchor arms need a sweep whose uniform-profile midpoint falls outside 512-2000. The registered cheap sweep puts it at 955 and fails; Pythia's full every-1000 schedule puts it at 31496 and clears the condition, but at that density the noise share reaches 0.63 and a real change at the anchor is dragged out of the window, so the arm has no power there either. It sits beside the 19-control requirement rather than replacing it. AND THE FALSIFICATION BRANCH FIRES WITH NO MARGIN: both anchor arms' reciprocal p is floored at 1/(n_controls+1), which is exactly alpha at nineteen controls, so RE-ANCHORS needs the observed series to rank strictly worst of twenty in both arms at once. Nineteen is therefore the exact minimum for the falsifier as well as for CO-LOCATES. |
| `P-T1` | H-OPERATOR | **e-value** | 1.0 | Label-permutation test over the row-2 classification. STATISTIC, fixed in advance: trimodal-rate(row-2 candidates) minus trimodal-rate(controls), with trimodality defined as stable_n_modes == 3. ALTERNATIVE: one-sided 'greater'. NULL: permute the row-2 labels across heads, holding both marginals fixed -- which is exactly the amended falsifier ('trimodality is a property of the activations rather than of the classification'). The control arm is therefore not an add-on, it IS the null. Implemented in p2d_operator_activation/table1_predictions.py::p_value_p_t1. DRY RUN 2026-08-27 (claims/audits/p_t1_p_m1_dry_run.json, POPPER_PLAN.md 6p), shared with P-M1, AND THE GATE GAINED AN ATTAINABLE FLOOR. It reported core.nulls.p_from_null's `resolution` = 1/(n_perm+1) as its floor. That is the SAMPLE's limit. This statistic is DISCRETE -- a rate difference over tens of heads -- so the null puts a lump of mass exactly on the observed value and the smallest expressible p is set by the MARGINALS. At five heads with two candidates the exact floor is 0.100 against a reported resolution of 0.0005, two hundred-fold, so a PERFECT input (every candidate trimodal, no control) could not have cleared alpha there. The gate was returning 'not significant' from a design that could not have rejected, which on a prediction reads as evidence against it. THE FLOOR IS EXACT: the statistic is monotone in how many trimodal heads land in the candidate arm and the null holds both marginals fixed, so that count is hypergeometric and the floor is the tail at the most extreme table the marginals admit, by math.comb, with no draw count in it. The gate refuses when it exceeds alpha, and the refusal costs nothing BY ENUMERATION rather than by measurement -- every attainable arrangement at every refused configuration is listed and none clears alpha. WHICH CONSTRAINT BINDS: the smallest p a run can express is the MAX of the design floor and the sampling resolution, and they bind at opposite ends -- at 12 candidates against 36 controls the design floor is below 1e-6 and the draw count binds again. The old resolution was wrong exactly where this entry lives, at tens of heads. A DESIGN THAT NEVER EMITS: two candidates against three controls now refuses on every draw, which is the pre-computed requirement in the form a reader can see. SHARED INSTRUMENT WITH P-M1, and it is recorded here for the first time: both classify the same head's Wq, Wk and W_OV -- this entry on V's eigenstructure and the QK form, P-M1 on M = Q^T K's symmetry and V's alignment with it -- and the extraction that decides which head's weights are which is shared. Unlike CLAIM-B and P-I1, WHICH SIT UNDER DIFFERENT CLAIMS, these two are both H-OPERATOR's, so two e-values that one defect moves together multiply into ONE claim's E. That is the specific way a product inflates without anyone editing a number. POPPER_PLAN.md 6h spent an audit ruling out exactly this defect class one phase over. |
| `P-M1` | H-OPERATOR | **e-value** | 1.0 | Permutation test over layers. STATISTIC, fixed in advance: the Pearson correlation between the per-layer MEAN head regime distance and the violation series. ALTERNATIVE: one-sided 'greater' (P-M1 predicts violations concentrate FAR from the gradient-flow condition). NULL: permute the violation series against the regime score, preserving both marginals exactly -- which matters because the violation series is heavily skewed and a parametric correlation test would lean on normality it does not have. Implemented in p2d_operator_activation/gradient_flow_condition.py::p_value_p_m1. DRY RUN 2026-08-27 (claims/audits/p_t1_p_m1_dry_run.json, POPPER_PLAN.md 6p), shared with P-T1, AND THE GATE GAINED AN ATTAINABLE FLOOR -- the same defect, reached independently. It reported core.nulls.p_from_null's `resolution` = 1/(n_perm+1) as its floor. This statistic is DISCRETE: a correlation against a violation series that UPDATE_PLAN.md 5.9 makes a per-boundary INDICATOR, so with few violations the permutation null has few distinct outcomes. Twelve layers with one violation have an exact floor of 0.083 and six layers with one have 0.167, against a reported resolution of 0.0005 -- so a PERFECT input (violations on the highest-regime layers) could not have cleared alpha at either. THE FLOOR IS EXACT AND IS A LOWER BOUND: permutations that only swap EQUAL violation values give the same correlation, so the floor is prod_v(multiplicity of v)! / n!, which for a binary series with T violations in n layers is 1/C(n, T). A tied REGIME SCORE would make the true floor larger, so refusing on this bound can only under-refuse and can never turn away a result that would have cleared alpha -- the same argument as P-ST1's 2m bound (6m). The gate refuses when it exceeds alpha. TWO DESIGNS THAT NEVER EMIT: six layers with one violation, and twelve with one. A 36-layer model with a single energy-monotonicity violation is not exotic -- it is what a mostly monotone trained model produces -- so this is a requirement on the sweep rather than a stress case. WHICH CONSTRAINT BINDS: the smallest p a run can express is the MAX of the design floor and the sampling resolution; here the marginals bind at every design size tabulated. SHARED INSTRUMENT WITH P-T1: both classify the same head's Wq, Wk and W_OV, and both are H-OPERATOR's -- so unlike CLAIM-B and P-I1 under different claims, a single extraction defect moves two factors of ONE claim's product. See P-T1's entry. |
| `P6-A2` | H-OPERATOR | **needs-null** | 0.8 | Classification agreement between f_rot and head type; needs a permutation null over the head-type labels. |
| `P6-I1` | H-OPERATOR | **e-value** | 1.0 | Already a Mann-Whitney U on f_rot(induction heads) vs f_rot(semantic heads). Valid as it stands; only needs threading through core.adjudication. |
| `P6-I2` | H-OPERATOR | **e-value** | 0.8 | Two-sample test over head pairs; same shape as P6-I1. |
| `P6-R1` | H-OPERATOR | **needs-null** | 0.8 | Threshold on a ratio (R >= 5) with a random-projection reference already named. That reference IS the null; it needs to be sampled rather than used as a single comparison value. |
| `P6-R2` | H-OPERATOR | **e-value** | 1.0 | Matched-dimension random-subspace null, built 2026-08-24 in p6_subspace/r2_r4_null.py and fixed BEFORE any p-value exists. INSTRUMENT: Phase 6's projector path was REBUILT live in p6_subspace/ against core/particles.py and core/nulls.py rather than lifted from archive/p6_subspace/subspace_build.py, per archive/README.md rule 2, which is what taking this entry out of `dormant` requires. PREREQUISITE, SETTLED FIRST: status-6.md item 5 records a projector-construction error (Schur block mislabelling swapping U_neg and U_A) as a live alternative explanation for the recorded inversion, and design-6.md pre-registered ruling it out BEFORE treating the hypothesis failure as established. tools/audit_p6_projector_labels.py does so and commits the record to claims/audits/p6_projector_labels.json: RULED-OUT on two independent routes -- planted structure recovered to 3.3e-08 rad, and bucket sizes matching a classification taken from np.linalg.eigvals without touching the Schur form -- with two deliberate mislabellings caught. STATISTIC, fixed in advance: the mean over layers of (chance-normalized alignment of the cluster-separating direction with U_neg minus the same with U_A). NORMALIZED ON BOTH ARMS, which is the substantive change from the archived comparison: p6_subspace/math-6.md 7.2 records that E[\|\|P_U v\|\|^2] = dim U / d, so raw alignment scales with subspace dimension, and the resolution order (span(U_pos) removed from U_neg, span(U_S) removed from U_A) makes U_neg the doubly-shrunk bucket. The audit measures dim(U_A)/dim(U_neg) = 24.9 at albert-xlarge-v2's exact shape against an observed alignment ratio of 13.2 -- the dimension correction is LARGER than the effect it would explain. ALTERNATIVE: one-sided 'greater' (P6-R2 predicts MORE alignment with the real repulsive channel). NULL: H0-OPERATOR realised directly -- replace the operator-derived subspaces with random subspaces OF THE SAME DIMENSION and recompute. Drawn MUTUALLY ORTHOGONAL, because U_neg and U_A are orthogonal by construction and independently drawn null pairs are not; that mismatch alone put the H0 rejection rate at 0.0875 against a nominal 0.05, in the anticonservative direction and invisible in any single result, and it was found by simulating rather than reasoning (the P-S1 defect of POPPER_PLAN.md 6d, second instance). Corrected, the measured rate is 0.045 at alpha=0.05 over 400 replicates. EXCHANGEABLE UNIT: REGISTERED as 'model' -- the AUTHOR'S decision, taken 2026-08-25 and recorded in POPPER_PLAN.md 6l. It was deliberately left unregistered for two passes: which unit may enter an e-process is a scientific decision of the same class as CLAIM-C's criterion, and taking it after seeing a p-value would void the guarantee. It was safe to take now because no p-value on real activations exists -- claims/adjudications/ is empty, no run artifact is in the repository, and every number either unit has produced came from synthetic populations. The construction is parameterized over it and computes either -- unit='model' draws one set of subspaces shared across layers, which is what ALBERT's weight-tying literally means (one OV matrix, one Schur decomposition, one projector pair, 49 activation snapshots); unit='layer' draws independently per layer, which is the error status-6.md names. The gap was MEASURED at 400 replicates rather than argued: with independent per-layer directions both units sit at 0.0525; as the layers come to share one direction the layer unit rises to 0.0800 (rho=0.5), 0.2325 (rho=0.9) and 0.2800 (rho=1.0) while the model unit stays at 0.045-0.0575 throughout. The mechanism is that the layer unit averages n independent null draws where the model unit averages n copies of one, so its null is narrower by sqrt(n). 'model' is therefore the conservative choice at every point of that range and not a trade: it costs nothing at rho=0, where the two agree, and it is the only one that holds at nominal where ALBERT actually sits. REGISTERED_EXCHANGEABLE_UNIT is 'model' and adjudicate_p6_r2_r4 now refuses a result computed under 'layer' rather than refusing everything. Passing unit= still does not route around that -- the argument selects what to COMPUTE, the module constant decides what may enter an e-process. ONE CONSEQUENCE, recorded because it is easy to miss: while no unit was registered that refusal was doubling as the safety catch keeping a synthetic p-value out of P6-R2's ledger slot, since adjudicate_p6_r2_r4 could not reach core.adjudication at all. It can now, so every test that both asks to adjudicate and uses the registered unit passes an isolated adjudications_dir. Registering the unit lifts one refusal and adjudicates nothing: there is still no run artifact and the ledger is still empty. ATTAINABLE FLOOR, checked before building rather than after a null result: 1/(N_NULL_DRAWS+1) = 0.0005, two orders below alpha, and the module refuses if it ever exceeds alpha. This REFRAMES the question the plan had posed. Under a CLAIM-C-style sign-flip enumeration the coarsest honest unit is 'one model', n=1, floor 2/(2^1+1) = 0.667, and the design could not reject on a perfect result. Under randomisation over SUBSPACES rather than over units, n=1 is no obstacle at all. The binding constraint was the choice of null, not the choice of exchangeable unit. REFUSES rather than degrades: on an unknown unit (no default, since the two differ by orders of magnitude); on an empty U_neg or U_A (normalized alignment is undefined there and 0.0 would read as 'orthogonal' for 'absent'); on fewer than two clusters after dropping HDBSCAN noise (label < 0); on a non-finite statistic; on a null thinned by failures; and on the attainable floor exceeding alpha. WHAT THIS DOES NOT DO: it does not adjudicate the 2026-04 ALBERT run. That run reported RAW alignments (0.887 with U_A, 0.067 with U_neg) and its statistic is not dimension-normalized, so those numbers are SUPERSEDED as evidence rather than turned into a p-value. Chance-normalized against the audit's measured dims they read 0.960 for U_A and 1.805 for U_neg -- the PREDICTED direction -- but the audit's dims come from random OV matrices at ALBERT's shape and not from ALBERT's trained weights, so that is a bound on the correction and not a result. The actual per-layer dims are computed by the projector build on every run and were never reported; recovering them is one number and it settles the reading. DEPENDENCE ON P6-R4: the two share one projector, so a projector defect moves both. They are NOT the P5b-B1/B3 pattern of one test with two thresholds -- R2 compares alignments, R4 compares probe accuracies, two statistics on two instruments -- and both are registered as adjudicable, but the common-cause dependence is recorded here so a reader does not read their product as two independent factors. THE NULL WAS CHANGED ON 2026-08-26 AND THE REASON CAME FROM ANOTHER ENTRY (POPPER_PLAN.md 6n; claims/audits/p6_r2_r4_dry_run.json). The matched-dimension random orthogonal subspace pair randomises the UNION of the two channels together with the SPLIT between them, so it rejects when the pair is unusual as a pair -- and 'span(U_neg + U_A) sits above chance against this layer's separating direction' is a fact about the union, not about which half the operator calls repulsive. Measured on an H0 whose split is uniformly random by construction, so that the correct answer is 'do not reject', the retired null's rejection rate is a monotone TREND in that alignment: 0.000 at chance and 0.155 at 3.9x chance, against a nominal 0.05. WHAT IS ADJUDICATED holds each layer's union fixed and re-splits it at the observed dimensions, so exchangeability under H0 is by construction rather than by measurement; measured 0.047 and 0.068 at the same two ends of the sweep over 600 replicates, pooling with an independent 1000-replicate run to about 0.056 at the aligned end -- at or marginally above nominal, flat in the union's alignment, and 9.7 standard errors below the retired null there -- with power unchanged at 1.000 against the union's content concentrated in U_neg. The retired null is computed beside every result and never adjudicated. The check was forced by P-ST1: POPPER_PLAN.md 6m retired the same construction there, and it is 6h's, introduced HERE -- a defect in a borrowed construction is a defect at its source. |
| `P6-R3` | H-OPERATOR | **needs-null** | 0.8 | Directional dominance at merge events; permutation over merge vs non-merge steps. |
| `P6-R4` | H-OPERATOR | **e-value** | 1.0 | Matched-dimension random-subspace null, built 2026-08-24 in p6_subspace/r2_r4_null.py alongside P6-R2 and fixed BEFORE any p-value exists. Same rebuilt instrument, same prerequisite audit (claims/audits/p6_projector_labels.json), same exchangeable-unit position: REGISTERED as 'model' by the author on 2026-08-25, before any p-value on real activations existed; adjudication now refuses a 'layer' result rather than refusing everything, and the measured cost of the wrong unit, together with what the decision was made against, is recorded on the P6-R2 entry and in POPPER_PLAN.md 6l. STATISTIC, fixed in advance: the mean over layers of the cross-validated accuracy of a linear probe fit INSIDE U_S, on the projected coordinates rather than the ambient embedding -- projecting back into R^d would hand the probe the dimensions the projection was supposed to remove. ALTERNATIVE: one-sided 'greater'. NULL: the same probe fit inside a random subspace of THE SAME DIMENSION, which is what makes this a test of operator content rather than of capacity; a probe fit in a higher-dimensional subspace has more capacity, and math-6.md 7.2 records that the archived 'imaginary-only 0.564 vs real-only 0.152' comparison has exactly that confound. PROBE: a ridge one-vs-rest linear probe with stratified k-fold, in numpy. NOT sklearn's LogisticRegression, which archive/p6_subspace/probe_subspace.py used. Partly because sklearn is a heavy-tier dependency and this module is pure tier, but mainly because the archived accuracies are not comparable to anything computed here anyway -- they were measured on subspaces of unequal dimension -- so matching the classifier would buy a comparability that is not available. What matters is that the SAME probe scores both arms, and it does. The ridge coefficient is PLACED, not calibrated, and is applied identically to both arms so it cannot move the contrast. POWER CAVEAT, measured: this statistic separates only while U_S is a SMALL fraction of d_model. At dim U_S / d = 14/24 both arms saturate at accuracy 1.0 and the test reports p = 1.0 on a planted effect -- a random subspace of that size captures the signal about as well as the real one. ALBERT's ratio is roughly 150/2048 and at a comparable fraction the arms separate cleanly (planted in U_S: p = 0.016; planted in U_A: p = 1.000). A run where U_S is a large fraction of d_model needs that said, not a p-value. REFUSES rather than degrades: on an unknown unit; on an empty U_S (there is no subspace to probe); on fewer than two clusters after dropping HDBSCAN noise; when the smallest cluster has fewer than two members, since cross-validation cannot be stratified over it; and on the attainable floor exceeding alpha. DEPENDENCE ON P6-R2: recorded on that entry. The shared projector is a common-cause failure mode for both. P6-R4'S NULL IS UNCHANGED, and that is a decision with a measurement behind it (POPPER_PLAN.md 6n). It compares ONE subspace against matched-dimension random ones, so it has no union and no split for the 2026-08-26 defect to reach; measured where a high-variance U_S captures 3.4x the population variance a random subspace of its dimension would, its rate holds at 0.040-0.048. Which of the two happens is decided by the STATISTIC and not by the claim: a sign of a difference saturates and fails hardest, a difference of chance-normalized alignments cancels a common elevation to first order and fails late, and a single subspace against matched controls has no common elevation to mismatch. |
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
| `P-ST1` | H-BRIDGE | **e-value** | 1.0 | Sign contrast of the effective-rank change between two matched-norm steering arms, built 2026-08-25 in p7_motifs/steering_gate.py and fixed BEFORE any activation was steered. Every choice below is a MODULE CONSTANT rather than a parameter, so it cannot be re-made per run. Evidence: claims/calibration/steering_sign.json; narrative: POPPER_PLAN.md 6k. STATISTIC: for each matched-norm pair, D = sign(dER_neg) - sign(dER_pos) in {-2,-1,0,1,2}, where dER is the change in RAW effective rank of the token population at the injection layer when alpha*v is added to every token; the statistic is sum(D) over pairs, one-sided 'greater' (H1 predicts D = +2 per pair: attractive-dominant lowers effective rank, repulsive-dominant raises it). Only SIGNS enter, so the criterion carries no magnitude cut. PAIR DRAWING: each arm is drawn UNIFORMLY FROM ITS SUBSPACE rather than thresholded as 'predominantly' in it -- 100% by construction, which retires one of the two constants the original wording flagged as needing pre-registration. Both arms of a pair receive the SAME alpha, so norm matching is by construction and not by a tolerance. STEERING IS A PURE MEAN EFFECT, and it decided what the original wording left most open. Adding alpha*v to every token is exactly a shift of the population mean, so re-centring AFTER injection annihilates the intervention identically -- algebra, not simulation. The cloud's PRE-EXISTING mean offset therefore competes with the injected one. Measured per-pair P(D=+2), H1 against H0, each configuration at its own best alpha: 0.970/0.010 at a mean offset of 0 spreads, 0.230/0.180 at 2, and 0.110/0.250 at 5 -- at five spreads the undebiased design rejects MORE often under H0 than under H1, and a real residual stream sits at that end. So the gate removes the BASELINE mean before injecting and keeps the injected offset: ER(X - xbar) against ER(X - xbar + alpha v), which restores 1.000 against 0.000 at every offset measured. Put to the author before the module was written. Its cost is stated: the criterion is then about the injected direction relative to the CENTRED population, a narrower object than the statement's 'effective rank of the token population'. ALPHA IS A THIRD PLACED CONSTANT THE ORIGINAL WORDING DID NOT FLAG, and it decides whether the prediction is readable at all. Below about 0.05 x spread both arms move the same way in every pair and the statistic is identically zero, so no alpha-free limit exists; above about 0.26 the rank-1 spike n*alpha^2*v v^T dominates the Gram matrix and both arms fall for ANY direction; between them is a PLATEAU at 0.17-0.24 where the per-pair rate is 1.000 under H1 and 0.000 under H0. ALPHA_SPREAD_FRACTION = 0.2, the middle of that plateau, labelled `placed` per Phase 7 adjudication constraint 4. The FRACTION is placed; the SCALE it multiplies -- the population's own RMS deviation from its mean -- is derived, which is what makes the plateau the SAME four fractions at mean offsets of 0, 2 and 5 spreads. The alpha-profile is reported with every result and enters no p-value; it is also where the falsifier's 'the effect tracks \|\|s\|\|' clause is read, since at matched norm \|\|s\|\| cannot vary WITHIN a pair and that clause is designed out rather than tested. EFFECTIVE RANK IS READ RAW, against CLAIM-C's use of effective_rank_normed, and the reason is structural rather than preferential. With the baseline mean removed the centred population has zero mean, so the first-order Gram term vanishes identically, dER is O(alpha^2) and EVEN in v. L2 row-normalization is not linear and reintroduces an odd term: measured, raw agrees with itself under v -> -v in 60 of 60 draws at every alpha from 1e-6 to 0.3, while normed falls to 0.00 at small alpha and MANUFACTURES D = -2 in 20-22% of pairs there. A criterion that answers differently for v and -v is not a criterion about a steering DIRECTION. The two modes are indistinguishable at the working alpha, which is why this is recorded rather than left as a preference. THE NULL IS NOT THE ONE THIS ENTRY'S EARLIER WORDING NAMED, and the earlier one is reported beside every result rather than deleted. Permuting the decomposition label across pairs treats m pairs as m exchangeable units; every pair at one layer sees the same tokens and the same two subspaces, so a chance tilt of the cloud moves them together and more pairs shrink the permutation null's spread like sqrt(m) while leaving the tilt untouched. Measured rejection rate under a noisy H0 at alpha=0.05, CONDITIONAL ON EMISSION, at 8/24/40/150 pairs: 0.000/0.031/0.030/0.220 and 0.000/0.012/0.082/0.170 in two H0 families. In the clean regime it is invisible, because there every H0 pair is uninformative and the gate REFUSES -- the unconditional rate reads 0.000, calibrated by refusing rather than by controlling, which is the conditioning error POPPER_PLAN.md 6g records for CLAIM-C. THAT REPLACEMENT WAS ALSO RETIRED, on 2026-08-26, and it is reported beside every result too (POPPER_PLAN.md 6m). It was a MATCHED-DIMENSION RANDOM ORTHOGONAL SUBSPACE PAIR, P6-R2/P6-R4's construction: replace the two operator-derived subspaces with random ones OF THE SAME DIMENSIONS, drawn mutually orthogonal from one Stiefel draw because the real pair is orthogonal by the projector build's resolution order. Matching the dimensions holds fixed everything the statistic could read off dimension and does NOT hold fixed how much of the population each subspace contains -- which is what dER is driven by, since injecting along a direction the cloud already occupies reinforces a large Gram eigenvalue and lowers effective rank while injecting along one it does not raises it. A random k-dimensional subspace holds k/d of the population; U_pos and U_neg are cut from the model's own OV eigenstructure and a residual stream is orthogonal to neither, so both hold more, and such a pair is unusual against random pairs whichever arm is called attractive. Measured on an H0 family where both arms are occupied above chance and the two are IDENTICAL by construction -- so a label swap is a distributional identity and the correct verdict is INSUFFICIENT -- it rejects at up to 0.20 against a nominal 0.05, the inflation growing with the pair count, in whichever direction the layer's realized asymmetry falls. The 2026-08-25 calibration could not see this: all three of its H0 families put the cloud in a subspace ORTHOGONAL to both arms, which is the one case where a matched-dimension random pair IS exchangeable with the observed one. WHAT IS ADJUDICATED holds the union fixed and randomises only the SPLIT. The old null randomised the union and the split together, so it rejected on either, and 'this pair of subspaces is unlike a random pair' is a statement about the union, which is not what this entry claims -- the claim is about the labelled split. So: draw a uniformly random k_pos-dimensional subspace of span(U_pos + U_neg) and take its orthogonal complement WITHIN that union. Dimensions, orthogonality, occupancy and the whole spectral relationship to the layer's cloud are held exactly fixed; the observed split is one point of the Grassmannian the null draws from, so exchangeability under H0 is by construction rather than by measurement. Measured: at or below nominal on every H0 family including the one that retired its predecessor. It costs no power where the cloud fills the whole arm and costs it as dim U_pos grows past the dimension the population occupies -- and power lost that way was never power about the decomposition. This is POPPER_PLAN.md 6h's question, what is being randomised, arriving for the FIFTH time and the first time the answer is to randomise LESS. Each arm's chance-normalized occupancy -- its share of the centred population's energy divided by the k/d a random subspace of that dimension holds -- is reported in every record as a DIAGNOSTIC that enters no p-value: it needs no injection, so a pilot can read it off the activations and the two projectors before spending a sweep, and a large asymmetry between the arms is what a TRACKS verdict is made of. REPLACING THE NULL ALSO REMOVED A FLOOR. Under the permutation a pair whose two arms move the same way contributes D = 0, and a zero contributes identically to the observed sum and to every null pattern, so with k of m pairs informative the best attainable p is (2^(m-k)+1)/(2^m+1) ~= 2^-k -- set by the INFORMATIVE pairs, needing k >= 5 at alpha=0.05 at every m. The draw-count floor of the subspace nulls is 1/(draws+1), fixed by the draws and independent of the data. IT IS NOT THE ATTAINABLE FLOOR, which is a defect this construction carried until a dry run on inputs with known answers looked (tools/dry_run_p_st1.py, POPPER_PLAN.md 6m): sum(D) cannot exceed 2m, and every null re-split that already reaches 2m ties an observation there, so on a union the cloud occupies the smallest expressible p is a fact about the LAYER -- 0.11-0.17 in both directions on a PERFECT input at one pair with 99 draws, where 1/(draws+1) says 0.01 (claims/audits/p_st1_dry_run.json). The gate now computes the attainable floor in both directions from the null it already has and REFUSES when neither tail can reach alpha, since the verdict was INSUFFICIENT before the statistic was looked at; 2m is an upper bound on the observation rather than an attainable value, so that floor is a lower bound on what the run can express and the refusal can never turn away a result that would have cleared alpha. It is CLAIM-C's informative-row refusal (6l) arriving here. When only ONE tail is out of reach the gate does not refuse -- one reachable tail is one reachable verdict -- and records `reachable_tails` instead, because a run whose only reachable verdict is the FALSIFICATION is one a reader must be told about. The informative-pair floor is still computed and reported as the diagnostic arm's. STOP RULE: three-way. TRACKS-DECOMPOSITION when p_greater <= alpha; INVERTS when the reciprocal 'less' test rejects, i.e. attractive-dominant steering demonstrably RAISES effective rank while repulsive-dominant lowers it; INSUFFICIENT otherwise. ONLY INVERTS is a falsification, and this needs saying because the registered falsifier's own wording -- 'both arms move effective rank the same way, or the effect tracks \|\|s\|\|' -- describes the NULL in both clauses, and an e-process records insufficient evidence and never a null accepted. Those map to INSUFFICIENT. Only p_greater is calibrated into H-BRIDGE's product; p_reciprocal is a verdict input, since two one-sided tests on one statistic would double the claim's Type-I rate for free. INVERTS was checked to be a branch that can actually fire (1.000 under a planted inversion), because POPPER_PLAN.md 6h found an audit arm reporting PASS while incapable of failing. REFUSES rather than degrades when no pairs are drawn, when span(U_pos + U_neg) has rank below dim U_pos + dim U_neg -- the two arms overlap, or their dimensions together exceed d_model, and either way the union cannot hold the observed pair orthogonally so no re-split of it reproduces the observed geometry -- when the draw count cannot express a p at or below alpha, when neither tail's attainable floor reaches alpha, when the population has zero spread about its mean, and when a subspace is too degenerate to draw from. Data refusals are grouped ahead of calibration refusals, POPPER_PLAN.md 6l's ordering: a run whose geometry cannot carry the null should say so rather than be turned away for a draw count that could be raised. The orthogonality of the two arms had been ASSUMED from the projector build's resolution order since the module was written and was never checked until 2026-08-26. PRECONDITION ON THE PILOT, computed before it runs. A uniform draw from U_pos carries only dim(occupied)/dim(U_pos) of its energy into the subspace the cloud occupies, so the per-pair informative rate falls with that ratio: 1.000, 0.710, 0.320, 0.030, 0.005, 0.000 at ratios 1, 1.5, 2, 3, 4, 6. claims/audits/p6_projector_labels.json already records U_pos as the UN-shrunk bucket in the projector build's resolution order, which is the unfavourable side. The pilot must report dim U_pos at the injection layer against the population's effective rank. The obvious fix is REFUSED: drawing from the intersection of U_pos with the occupied subspace would restore the rate and is circular, since a probe aligned with the cloud by construction concentrates it by construction. THE RECIPROCAL TAIL, the INVERTS branch that would enter the ledger as a falsification, is measured in its own section of claims/calibration/steering_sign.json at four times the main table's replicates, at one pair count, which is what pays for them: fifty gate runs resolve a rate only to about +/- 0.03 and cannot separate nominal from twice nominal, which POPPER_PLAN.md 6k recorded as this construction's weakest measurement. Every family in that section has arms that are exchangeable by construction, so the two tails must agree within sampling error and `tails_agree` says whether they do. STILL NO DATA: the gate needs activations and the Phase 2 attractive/repulsive projectors, and neither exists in this repository, so validation is on synthetic populations with planted answers and claims/adjudications/ remains empty. |
| `P-I5` | H-BRIDGE | **needs-null** | 1.0 | Permutation null over the matched-magnitude random-direction ablation arm, on a two-dimensional statistic (geometric delta, logit delta). The joint form matters: two separate one-dimensional tests would let the prediction be scored a partial pass in the configuration it is designed to rule out. REQUIRES an extension to core/dual_reading.py -- every current geometric field is per-point and this needs a pairwise one. |
| `P-AB1` | H-BRIDGE | **needs-null** | 0.8 | Growth-exponent comparison against a MATCHED RANDOM-DIRECTION ablation of equal magnitude at the same layer -- the same control design design-5c.md already requires for its force-collapse and force-disperse arms. The control is not optional: later layers have more opportunity to diverge for reasons unrelated to field structure, so a superlinear fit against no control measures remaining depth, not mechanism. Permutation over ablation points once the fitted exponent is the statistic. |
| `P-SA1` | H-BRIDGE | **needs-null** | 0.8 | Random-subspace null of matched dimension, comparing the observed mass fraction in U_neg against dictionaries of the same rank drawn isotropically. |
| `P-I1` | H-BRIDGE | **e-value** | 1.0 | Changepoint co-location on the log-step axis, built 2026-08-24. The construction is CLAIM-B's, in core/changepoint_colocation.py, and is deliberately NOT reinvented -- EVALUABILITY.md named these two as sharing one and said they should be built together. P-I1's gate is the thin half, in p7_motifs/formation_gate.py, because PREDICTIONS.md names p7_motifs/motif_stats.py as its instrument. See CLAIM-B's null_construction for the estimator, the null, the floor arithmetic, the refusals and the measured calibration; everything below is what differs. COMMON-CAUSE DEPENDENCE: CLAIM-B (H-EMERGE) and P-I1 (H-BRIDGE) sit under different claims, so there is no P5b-B1/B3 double-counting problem, but a defect in the shared estimator moves both and their e-values must not be read as two independent factors. ONE ARM, NOT THREE: P-I1 names no literature anchor -- it asks only that the two curves rise together -- so there is nothing for an anchor arm to test and none is invented. The mutual arm is the whole gate, and because the anchor arms are what would refuse on control-set size, P-I1 is the more likely of the two to return a number. SERIES AND DIRECTIONS, both registered as a RISE: `relay` motif strength MINUS the N1/N2 offset-null envelope (core/qk_offset_null.py), and the behavioral induction score. The series handed in must ALREADY be the above-null excess, since P-I1's wording is 'strength above N1 and N2' and clearing those nulls is motif_stats.py's job; the gate cannot check that and says so rather than implying it did. EXCHANGEABLE UNIT: the HEAD, and that is a registered constraint rather than a convenience -- PREDICTIONS.md's first Phase 7 adjudication constraint is 'Effective n is the number of heads, not the number of edges. Edges within a head are not independent samples. Any significance computed over edge counts is wrong by orders of magnitude, in the direction that manufactures findings.' The null permutes which head's motif curve is matched with which head's behavioral curve, so the head is the unit by construction and an edge-level n cannot enter. THE FALSIFIER'S SECOND HALF IS A PRECONDITION, NOT A P-VALUE: 'motif already above nulls at step 0, or absent at step 143,000 despite a high behavioral score' is a statement about the curve's ENDPOINTS and the statistic is about where it rises, and one number cannot carry both questions. endpoint_flags reports both as per-head counts beside the result and enters no p-value. Because the series is an above-null EXCESS, zero is the null envelope and neither endpoint check needs a placed threshold. THE TAUTOLOGY RISK, which this gate cannot discharge: PREDICTIONS.md's second Phase 7 adjudication constraint records that the behavioral induction score is 'mean attention on induction pairs' and a motif defined as 'attentive edge on induction pairs' is the same number. Two identical series co-locate perfectly and the gate would report p at its floor. No null detects it, because the null is over the PAIRING and a tautological pair is tautological at every head. The gate refuses on series that are exactly identical at any head, which catches the degenerate case and not the substantive one, and the independence source stays a claim the analyst must make in the record exactly as the constraint requires. This is the same failure mode as the measured shared-unit-factor limitation in CLAIM-B's entry -- rejection rate 1.00 against 0.05 -- reached from Phase 7's own direction. STILL NO DATA: no checkpoint sweep of motif strength exists, validation is on synthetic inputs with known answers, and claims/adjudications/ is empty. DRY RUN 2026-08-27 (claims/audits/claim_b_p_i1_dry_run.json, POPPER_PLAN.md 6o), shared with CLAIM-B, and P-I1 WAS NOT CHANGED -- a decision with a measurement behind it, the precedent P6-R4 set one pass earlier. The dry run found that a change location is partly a property of the sweep grid: mass spread evenly over the sweep lands on the grid's own midpoint exactly, and noise makes every location a mixture of the two. CLAIM-B's ANCHOR arms cannot cancel that pull, because their reference is a fixed window, and on the registered cheap sweep they reject on a series with NO located change at 1.000. P-I1 is the MUTUAL arm alone -- a difference of two locations whose null permutes the pairing and so keeps both series' real per-head locations on both sides of every draw -- so the pull is common to both and cancels. Measured on the REGISTERED cheap sweep, the grid where the anchor arm fails hardest rather than a friendlier one, the mutual arm holds at 0.045-0.065 across four H0 families including one in which neither series changes anywhere. WHICH SHARPENS POPPER_PLAN.md 6n's TAXONOMY: 6n put 'one subspace against matched-dimension controls' in the safe column with 'nothing to mismatch'. The anchor arm is the counter-example -- an absolute quantity against matched controls is safe only when the controls are matched on THE QUANTITY THE STATISTIC DEGENERATES ON. P6-R4's are matched on dimension, which drives its statistic; the anchor arm's are matched on the sweep and the units, and what drives its statistic is where the grid puts a profile carrying no location. GRID DIAGNOSTIC: every record, P-I1's included, now carries grid_reference_report -- where a series with no located change lands on this grid -- and each profile carries noise_mass_share_estimate, the reverse-direction mass over the forward mass, which needs no model of the noise and is reported and never used to correct a centroid. |
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

## `CLAIM-B` and `P-I1`: one construction, two entries (2026-08-24)

The line this document closed on — *"`CLAIM-B` is next, and it shares a
construction with `P-I1`, so the two should be built together rather than each
inventing one"* — is done. `core/changepoint_colocation.py` is the
construction; `p7_motifs/formation_gate.py` is P-I1's thin half. Both entries
are `e-value` and both emit nothing, because no checkpoint sweep exists here.
Eight predictions are now adjudicable in principle and `claims/adjudications/`
is still empty.

**Building them together was not a tidiness argument, and the registry says so.**
They sit under *different* claims — H-EMERGE and H-BRIDGE — so there is none of
the `P5b-B1`/`P5b-B3` double-counting problem. But one shared estimator is a
common-cause failure mode: a defect in it moves both, and their e-values are
therefore not two independent factors. That is recorded in both
`null_construction` fields and in both ledger records, the precedent `P6-R2`
and `P6-R4` set for their shared projector.

**The registered wording named a null that measurement showed to be invalid.**
Both entries said "a permutation null over checkpoint order gives a valid p once
the changepoint estimator is fixed in advance." Four permutation-family nulls
were built and their H0 rejection rate measured against a nominal 0.05:
permuting the value series against the fixed step grid, 0.45; permuting the
interval increments, 0.32; a *sampled* circular shift, 0.13; the same shift
*enumerated*, 0.065. The first three fail for one reason — the statistic is
built on a concentrated change profile and those nulls dissolve the
concentration, so the null's variance is far too small and any partial overlap
of two real profiles reads as significant. The enumerated shift is honest but
assumes changepoints are uniform on the interval grid, and with both series'
onsets drawn early — *everything moves early in training* — it rejects at 0.103.

What replaced it is a **matched control series**, where the control for series
B at unit *u* is series B at another unit, combined across units as a
permutation of the *pairing* between the two series' units. Measured at 8, 16
and 24 units, it holds nominal under the plain H0 **and** under the common
early trend that defeats every order-permutation. Making it a permutation over
pairings rather than one test per unit is also what keeps it clear of
`status-6.md`'s "n layers are not n independent observations".

So the third lesson to carry, after the two above:

**A matched control can be another series.** `EVALUABILITY.md` already noted
that several queued rows name a matched control that is a *subspace* or a
*magnitude* rather than a unit. This adds a kind: the control population for a
per-unit series is the same series at the other units, and the p-value is the
rank of the observed pairing among the arbitrary ones. It costs a commitment —
validity *is* the claim that the controls are exchangeable under H0 — and the
module refuses a control family that is not the registered one, the same way
`P6-R2` refuses a caller-supplied exchangeable unit.

**The floor was checked first, and it retired the obvious estimator.**
`checkpoint_frames.detect_transitions` returns intervals of largest change per
unit log-step. Because `interval_rates` divides by a spacing that varies 4.6×
across a Pythia sweep, the argmax of a permuted value series lands on the tightest-spacing interval
44.7% of the time when the value series is permuted against the fixed step
grid, so a binary "the two top intervals coincide" statistic has a
best attainable p of ~0.29 typical and 0.45 worst case. What replaced it — the
centroid of a change-*mass* profile — carries no placed constant at all: no
`n_top`, no `min_abs`, no tolerance on what counts as co-located. That is the
ordinal-style escape this document asked for, the way CLAIM-C's
sign-concordance avoided a magnitude cut.

**And a limitation that is severe, measured, and not fixed.** The pairing null
tests *association*, and a common per-unit factor — a layer that changes late
changing late in both series, for a reason unrelated to the claim — is an
association. Measured rejection rate under exactly that: **1.00**, against 0.05
when the two are independent. No null over the pairing separates them, because
a confound present at every unit is present under every permutation. Every
record carries a diagnostic that catches a confound *monotone in the unit
index* and catches nothing else, and the analyst must name the independence
source — which `PREDICTIONS.md`'s Phase 7 adjudication constraint 2 already
required, now with a number behind it. The honest fix is a confound-control arm
against other per-unit series; it needs the same 19 control series CLAIM-B's
anchor arms need, and it is not built.

**What the floor says the pilot must measure.** CLAIM-B's two anchor arms test
the change location against the pre-registered ~512–2000 window, and an anchor
arm has no permutation available — it needs a reference population of change
locations, so its floor is 1/(n_controls + 1) and α = 0.05 needs 19 control
series on the same sweep at the same layers. A cheap-tier sweep measuring six
metrics has six, and the arms refuse. Under the gate's unanimity rule that
refuses the whole gate. That is a requirement on the pilot, computed before it
runs, and it is the most likely reason CLAIM-B returns no number.

`P-I1` has no anchor arm — it names no literature anchor — so it is the more
likely of the two to produce one.

## The first dry run: `CLAIM-C` on an input whose answer is known (2026-08-25)

Five passes converted `needs-null` rows and the ledger stayed empty. This one
converted none. It ran the gate `CLAIM-C` already has on inputs whose correct
verdict is fixed *a priori* — one model as both the reference and the candidate,
so every cell is concordant — and recorded what came back
(`tools/dry_run_claim_c.py`, `claims/audits/claim_c_dry_run.json`,
`POPPER_PLAN.md` §6j).

**The fourth lesson, after the three above, and it is about order of work.**
*Validate a construction against a known answer before building the next one.*
The three defects the last three passes found — a rounding step that silently
disabled a refusal, a sensitivity arm that reported PASS while incapable of
failing, power figures measured under a null that had already been discarded —
were each found by *looking at an output*, and none of them failed a test. A dry
run is the cheap, systematic version of that: it generates the one output nobody
had generated, namely the gate's verdict on an input where the right answer is
not in question. It cost one session, needed no data, and found something no
synthetic unit test in the suite was failing on.

**What it found.** The criterion is sound — on a perfect input every
leave-one-out subset returns exactly the attainable floor, so the unanimity axis
does not bite on a unanimous input. But the gate has an **admissible band** in
its own input space: above `sign_homogeneity` 0.8125 at eight prompts (0.7708
since the band tightened later that day — see the cell-drop section below) the
derived refusal fires on *every* input including a perfect one, so the hard stop
fires unconditionally and carries no information about the data. The cost is
power rather than validity, and it falls hardest where the effect is most
uniform, because `sign_homogeneity` measures prompt redundancy under H0 and
effect uniformity under H1 and the correction cannot separate them.

Restated as the thing a pilot can act on — the §6i shape again, where CLAIM-B's
anchor arms needed 19 control series no six-metric sweep provides — **at least
9 of the candidate's 48 cells must dissent in sign** — 11 since the band
tightened — or CLAIM-C returns no
verdict at all. More prompts do not relax it.

**And one positive result, which is §6h's question asked of a refusal rather
than of a PASS.** §6h found an audit arm reporting PASS while incapable of
failing. The dual question is whether a refusal ever refuses something that
would have passed. R(h, ·) is non-decreasing in p in all 264 tabulated bins, so
whenever the derived refusal fires no input could have cleared α. It is tight:
it never costs a verdict the gate could otherwise have reached.

**What the remaining seven adjudicable rows are owed.** `P-S1`, `P-T1`, `P-M1`,
`P6-R2`, `P6-R4`, `CLAIM-B` and `P-I1` have all been validated on synthetic
inputs by unit tests and none has been run on an input whose answer is known in
the sense above. The queue that used to read "convert the next `needs-null` row"
now has a second entry ahead of it for each row already converted.


## `P-ST1`: the entry that can lose, and a fourth arrival of the same question (2026-08-25)

The row this document ranked as cheapest-among-the-bridge-entries is built:
`p7_motifs/steering_gate.py`, with `claims/calibration/steering_sign.json`
behind it and `POPPER_PLAN.md` §6k as the record. Nine predictions are now
adjudicable in principle and `claims/adjudications/` is still empty.

**It is the only registered prediction that can lose**, in the sense H-BRIDGE
needs: the particle and standard accounts make INCOMPATIBLE predictions about
the sign, not merely different ones. That is why it was worth building before
the rows with more apparatus behind them.

**Fifth lesson, and it is the third arrival of the fourth one.** *Before
concluding a design is valid, ask what is being randomised* — this document's
own third lesson, and §6h's. The registered null here permutes the
decomposition label across pairs. Measured, it is anticonservative, and the
inflation **grows with the pair count**: 0.000 at 8 pairs, 0.03 at 40, 0.17–0.22
at 150, because every pair at one layer shares the tokens and both subspaces
and more pairs shrink the null's spread while leaving the shared tilt
untouched. The replacement is the matched-dimension random orthogonal subspace
pair — `P6-R2`'s construction, now used by a third claim. The list of matched
controls this document keeps is now: a subspace, a magnitude, another series,
and a *dimension-matched random subspace* used as the null rather than as the
comparison.

**A refusal that only conditioning could see.** In the clean regime every H0
pair is uninformative, the gate refuses, and the unconditional Type-I rate
reads 0.000 — calibrated *by refusing* rather than by controlling. §6g
established that rates must be conditional on emission for CLAIM-C's
homogeneity curve; here the same conditioning is what made an invalid null
visible at all. It should now be the default for every rate this project
measures, not a per-entry decision.

**And the third pre-computed requirement on a pilot in three passes.**
`CLAIM-B` needs 19 control series a six-metric sweep does not have.
`CLAIM-C` needs at least 19% of its candidate cells to dissent in sign (23%
since the band tightened).
`P-ST1` needs dim `U_pos` to be comparable to the dimension the token
population actually occupies: the per-pair informative rate falls from 1.000 at
ratio 1 to 0.000 at ratio 6, and `claims/audits/p6_projector_labels.json`
already records `U_pos` as the un-shrunk bucket. Three entries, three
requirements, all computed before any sweep runs — which is what the
attainable-floor lesson looks like once it is applied habitually rather than
after a null result.

**One thing this pass did NOT get to do, stated because it is the honest
gap.** The dry-run discipline `CLAIM-C` was given on 2026-08-25 — run the gate
on an input whose answer is known and look at the verdict — was applied here at
construction time rather than after, which is better. But the reciprocal tail,
the INVERTS branch that would enter the ledger as a falsification, is measured
at 0.02–0.10 over fifty runs per cell. That is consistent with nominal and it
is not a tight bound. The `greater` tail that is actually adjudicated is
measured at 0.000–0.040 over the same cells.


## `CLAIM-C`'s cell-drop dimension, and a floor that was never tight (2026-08-25)

Two things, and the second was found by building the first
(`POPPER_PLAN.md` §6l).

**The completion.** §6g's homogeneity curve measured every draw on a *complete*
(prompt × metric) table, so a real run that dropped cells — a non-finite or
exactly-zero contrast, which an ordinal criterion has to drop — read its
correction off a table measured on a design it does not have. §6g named the
second curve dimension as the honest fix; §6j made it the binding gap. It is
built: the curve is indexed by `(n_prompts, drop fraction, homogeneity)`,
refuses above the drop rate it tabulates instead of reading the nearest row, and
interpolates **nothing** across that dimension — because coarsening pushes
p-values up while selection pushes the conditional rate down, and measured, at
every tabulated prompt count the large majority of adjacent drop-bin pairs go
neither way (93 of 117 at eight prompts, 116 of 118 at twelve), and at six
prompts 12 pairs rise while 7 fall.

**Sixth lesson, and it is about where a design's power actually lives.** *Count
the units that can carry information, not the units that were run.* CLAIM-C's
attainable-floor refusal was derived from the prompt count. But a prompt whose
label flip does not change the statistic — every cell dropped, or an **even**
number of usable cells splitting exactly half and half — is enumerated through
all 2^n patterns and never counted, so the real floor is `2^-k` in the *k*
prompts that can move. Five is the first *k* that clears α = 0.05 at every
prompt count, which is `P-ST1`'s informative-**pair** floor arriving at the same
number from the other direction.

The half-and-half case is the one worth carrying: it needs no dropped cells at
all. With six metrics a prompt splits 3–3 under H0 with probability 20/64, so
the gate could be handed a table **perfect on four prompts** and 3–3 on two,
return p = 0.0769 — exactly that table's own floor — and report it as "not
significant". Measured, 61% of H0 draws at six prompts could not have rejected
however the statistic fell; 22% at eight, 1.2% at twelve. It had been live since
the gate was written and no synthetic unit test was failing on it.

**The refusal costs nothing, and that is measured rather than argued.** Both
tails share the floor, so a table it refuses could not have cleared α in either
direction. Across five H1 strengths, P(TRANSFERS) is identical to four decimals
with the refusal and without it — including where it fires on 15% of draws. The
dry run re-scores every refused table rather than restating the argument, and
reports `costs_no_power` as **None** rather than True when the refusal never
fired, because a sweep with nothing to re-score would pass while being incapable
of failing.

**The list of pre-computed requirements on a pilot is now four**, all computed
before any sweep runs. `CLAIM-B` needs 19 control series. `CLAIM-C` needs at
least 23% of its candidate cells to dissent in sign, **and at least five prompts
whose usable metrics do not split evenly**. `P-ST1` needs `dim U_pos` comparable
to the dimension the token population occupies.

**Seventh lesson, and it is about the shape of the measurement rather than the
design.** *A rate conditional on emission needs a fixed number of EMITTED draws,
so a sweep sized in draws measures the refusing corners more coarsely than the
rest.* Six prompts emit on 39% of independent-row H0 draws and twelve on 99%,
and drawing the same 40000 everywhere left the six-prompt (0, 5%] drop slab with
**no measured bin at all** — a slab the gate refuses outright, whatever the
data. Nothing failed; it was found by printing the coverage of a generated file,
the fifth session running that has been how a defect surfaced. The draw count is
now derived from that emission probability, which closes it (every slab at every
prompt count carries 12 to 18 measured bins of 20), and every curve carries a
`coverage` block so the next hole is visible in the artifact rather than in a
run being turned away.

**And a category this document had not used.** `P6-R2` and `P6-R4` are the first
entries whose refusal was lifted by a **recorded decision** rather than by new
apparatus: the author registered `"model"` as the exchangeable unit, before any
p-value on real activations existed. Nothing is adjudicated — there is still no
run artifact — but the refusal moved from "no unit is registered" to "this
result was computed under the other one". The one thing that changed underneath
it is worth stating: while no unit was registered, that refusal was doubling as
the safety catch keeping a synthetic p-value out of `P6-R2`'s ledger slot, and
it no longer is.

**And the cost this pass paid, stated because it is the kind that is easy to
leave out.** The refusal that costs no power does cost *band*. Every rate in the
curve is conditional on emission; a draw whose rows could not move the statistic
used to be emitted with a p above α and counted as a non-rejection, diluting the
measured rate downward. It is now refused, reaches no ledger, and belongs in no
denominator — so the rate among the draws that do reach one is higher, the
correction is stronger, and the derived refusal bites at a lower homogeneity.
The band at eight prompts went from ≤ 0.8125 to ≤ 0.7708 and the pilot
requirement from 9 dissenting cells to 11. Both curves were internally
consistent; only the new one describes the gate that exists. `POPPER_PLAN.md`
§6g's caution that each thing added moves probability mass into INSUFFICIENT
applies to a refusal exactly as it applies to a robustness axis, and this pass
added a refusal.


## `P-ST1` run on inputs whose answer is known, and the null that did not hold (2026-08-26)

The queue this document set on 2026-08-25 — *"the queue that used to read
'convert the next `needs-null` row' now has a second entry ahead of it for each
row already converted"* — has its second entry done. `tools/dry_run_p_st1.py`
→ `claims/audits/p_st1_dry_run.json`, `POPPER_PLAN.md` §6m. Seven adjudicable
rows are still owed it. `claims/adjudications/` is still empty.

`P-ST1` was the one to do next for the reason it was built early: it can
genuinely lose, and its whole intervention is exact linear algebra, so the gate
runs end to end on populations with a planted answer and no model at all.
§6k applied the dry-run discipline at *construction* time rather than after,
which is better and which made this a real check rather than a formality — the
question was whether anything survives being run on inputs the construction did
not already have in mind.

**Eighth lesson, and it is the one this document has been circling since §6h.**
*The H0 families a calibration measures are part of the measurement, and their
absence is invisible.* The null this entry adjudicated until 2026-08-26 replaced
both operator-derived subspaces with random ones of the same dimensions. Every
H0 family the calibration measured put the token cloud in a subspace orthogonal
to **both** arms — which leaves both at chance occupancy, and is exactly the
case in which a matched-dimension random pair *is* exchangeable with the
observed one. On the family that was missing — both arms occupied above chance,
the two identical by construction, so a label swap is a distributional identity
and INSUFFICIENT is the only correct verdict — it rejects at up to **0.20 against a nominal
0.05**, and the inflation grows with the pair count. That is the realistic case: `U_pos` and `U_neg` are cut from
the model's own OV eigenstructure and a residual stream is orthogonal to
neither. A calibration whose families cannot express the failure it is meant to
rule out is §6h's audit arm incapable of failing, one level up, and
`check_record()` now fails if that family is absent — and fails again if the
retired null does *not* come back anticonservative.

**Ninth, and it is the fifth arrival of the third lesson above.** *Before
concluding a design is valid, ask what is being randomised* — and this is the
first time the answer was to randomise **less**. The old null moved the union
and the split together, so it rejected on either, and "this pair of subspaces is
unlike a random pair" is a statement about the union rather than about the
decomposition the entry names. What replaced it holds the union fixed and
randomises only the labelled split: a uniformly random *k*<sub>pos</sub>-
dimensional subspace of span(`U_pos` + `U_neg`), with its orthogonal complement
*inside* that union as the other arm. Every property of the pair as a pair is
held exactly fixed, and the observed split is one point of the same Grassmannian
the null samples — so exchangeability under H0 is **by construction rather than
by measurement**, which no other null in this project can say. It costs no power where the cloud
fills the whole arm and costs it as dim `U_pos` grows past the dimension the
population occupies; power lost that way was never power about the
decomposition.

**Tenth: a reported floor is a claim, and it can be wrong in the optimistic
direction.** The gate reported `1/(draws + 1)` as the smallest p it could
express. `sum(D)` cannot exceed 2*m*, and on a union the cloud occupies many
random re-splits already reach 2*m* and tie an observation there, so the
attainable floor is a fact about the **layer**: 0.11–0.17 on a perfect input at one pair with 99 draws, where the reported
floor said 0.01. Until this was found the gate
could return "not significant" from a design that could not have rejected —
`POPPER_PLAN.md` §6l's defect for `CLAIM-C`, arriving here from the other side,
and §6i's for `CLAIM-B`'s sampled pairing regime arriving for the second time.
It is fixed by computing both tails' floors from the null the gate already has
and refusing when neither reaches α; 2*m* is an upper bound on the observation,
so that floor is a lower bound on what the run can express and the refusal turns
away nothing that could have cleared α.

**And a category worth naming, because it will recur.** A run can have only its
**falsification** branch reachable. The two tails' floors are computed
separately and are not equal, so one can be out of reach while the other is not,
and where the reachable one is INVERTS the design can return a falsification or
nothing. The gate does not refuse there — one reachable tail is one reachable
verdict, and a refusal must cost none — but every record now carries
`reachable_tails`. A run whose only reachable verdict is the one that enters the
ledger as a falsification is a run a reader has to be told about.

**The dry run also found a defect in its own arm, which is the sixth session
running.** The band sweep re-scored a "perfect input" in every cell to separate
"the data was not strong enough" from "nothing could have been" — and read that
counterfactual off a *single* draw. It marked cells as reaching no verdict whose
own twenty-five draws reached one 28% of the time. It now runs several seeds and
calls the field `no_verdict_in_any_draw`, and says explicitly that this is a
measured zero over a stated number of draws rather than `CLAIM-C`'s enumerated
proof that its gate is a constant function there. Nothing failed; printing the
table is what showed it.

**What the list of pre-computed requirements on a pilot now says**, unchanged in
count at four but sharper on one of them. `CLAIM-B` needs 19 control series.
`CLAIM-C` needs at least 23% of its candidate cells to dissent in sign and at
least five prompts whose usable metrics do not split evenly. `P-ST1` needs
dim `U_pos` comparable to the dimension the token population occupies — and the
dry run turns that from a per-pair informative rate into a whole-gate statement:
at dim `U_pos` / dim(occupied) = 3, nothing in the sweep reached a verdict at
all. The quantity to read before spending a sweep is now in every record and
needs no injection to compute: each arm's share of the centred population's
energy, divided by the *k/d* a random subspace of that dimension would hold.


## The retired null, checked where it came from: `P6-R2` and `P6-R4` (2026-08-26)

The section above retired the null `P-ST1` adjudicated. That construction is
`POPPER_PLAN.md` §6h's and §6h introduced it for `P6-R2` and `P6-R4`, so it had
to be checked at its source before anything else was built —
`tools/dry_run_p6_r2_r4.py` → `claims/audits/p6_r2_r4_dry_run.json`,
`POPPER_PLAN.md` §6n. It doubles as the dry run this document's queue owed both
entries, which makes four of the nine adjudicable rows done and five still
owed: `P-S1`, `P-T1`, `P-M1`, `CLAIM-B` and `P-I1`.

**Eleventh lesson, and it is the one this document's opening argument implies
but had never been exercised.** *A defect in a borrowed construction is a defect
at its source.* This document opens by saying the product is only as valid as
its weakest factor; the operational form of that is that when a shared
construction fails for one entry, the entries it was borrowed from are not
"probably fine". Three entries across two claims shared this one. Checking took
one pass and changed one of them.

**`P6-R2` had it.** Measured on an H0 whose split is uniformly random by
construction — so the correct answer is *do not reject* — the retired null's
rejection rate is a monotone trend in how far span(`U_neg` + `U_A`) sits above
chance against the layer's separating direction: 0.000 at chance and 0.155 at
3.9× chance, against a nominal 0.05. The replacement holds the union fixed and
re-splits it at the observed dimensions and does not trend — 0.047 and 0.068 at
the same two ends, pooling with an independent 1000-replicate run to about
0.056 at the aligned end, which is at or marginally above nominal and 9.7
standard errors below the retired null there. Power is unchanged at 1.000.

**`P6-R4` did not, and it is left alone.** It compares one subspace against
matched-dimension random ones, so it has no union and no split for the defect to
reach; its rate holds at 0.040–0.048 where a high-variance `U_S` captures 3.4×
the variance chance would give. That measurement is in the record because
leaving an entry unchanged is a decision, and a decision with nothing behind it
is the position §6h's construction was in for two passes.

**Twelfth, and it is the useful one for the rows still queued.** *A matched
control is matched on something, and which statistic you build decides whether
that something is enough.* The same construction is valid in one of these three
entries, mildly invalid in another and badly invalid in the third, and what
separates them is whether the statistic cancels a common elevation of both arms:

| entry | statistic | under an elevated union |
|---|---|---|
| `P-ST1` | the **sign** of a difference | saturating — no cancellation, 0.20 at 1.27× |
| `P6-R2` | a **difference** of chance-normalized alignments | cancels to first order — 0.14, but only at 3.9× |
| `P6-R4` | a **single** subspace against matched controls | nothing to mismatch — 0.04–0.05 at 3.4× |

This document already lists `P6-R1`, `P6-C1`, `P5b-A1`, `P5b-A2`, `P-SA1` and
`P-I4` as rows whose predictions name a matched control that is a subspace or a
magnitude rather than a unit. **Matched on what** is the question to ask of each
before the control is built, and the table above is how to answer it: look at
whether the row's statistic is a sign, a difference, or an absolute quantity
against a control.

And a smaller one, recorded because writing it down did not prevent it. §6h
found a module constant bound as a **default argument**, so an override did not
reach it; §6m found the same bug again in `tools/dry_run_p_st1.py` and wrote a
comment about it; this pass's tool reproduced it a third time anyway, caught by
a smoke run taking implausibly long rather than by anything failing. A comment
is not a guard.


## `CLAIM-B` and `P-I1` run on inputs whose answer is known, and the location that is partly the grid's (2026-08-27)

`tools/dry_run_claim_b_p_i1.py` → `claims/audits/claim_b_p_i1_dry_run.json`,
`POPPER_PLAN.md` §6o. These two share one estimator, which is why this document
said in the first place that they should be built together rather than each
inventing one, and one dry run covers both. That makes **six of the nine
adjudicable rows done and three still owed: `P-S1`, `P-T1` and `P-M1`.** Every
one of the six changed something. `claims/adjudications/` is still empty.

**Thirteenth lesson, and it is the one this document's own table could not have
asked for.** *A statistic can be partly a property of the measurement grid
rather than of the data, and the grid is not in any H0 family.* A change
location is the centroid of a change-mass profile — a weighted mean of the
sweep's interval midpoints — so mass spread evenly over the sweep lands on the
grid's **own midpoint**, exactly. Per-checkpoint noise puts rectified mass in
every interval, so every real location is a mixture of where the series changed
and where the grid's midpoint is, weighted by the noise's share of the mass;
that share grows with the interval count, so a **denser** sweep is worse rather
than better. None of the five H0 families the committed calibration measures
contains this, because it is not a hypothesis about the data at all.

**On `CLAIM-B`'s registered instrument that is fatal, and the coincidence is
what hid it.** The registry names a "20-30 checkpoint cheap-tier sweep"; that
grid's uniform-profile midpoint is step 955, **inside** CLAIM-B's own 512–2000
anchor window. A series that changes *nowhere* therefore attains the anchor
arm's maximum statistic, and against controls that all carry a located change
the arm rejects on a change-free input at **1.000** — exactly its rate on a
perfectly anchored one, so its discriminating power there is **zero**. The
general rate is `1/(k+1)` in the number of controls that are themselves
change-free, checked against that closed form at every *k*. On the one grid the
construction was calibrated for, the bias points at the answer.

**Fourteenth, and it is a category of refusal this project had not used.** *A
refusal can be right and still cost verdicts.* §6l's informative-row refusal
removed no reachable verdict and was measured at zero power cost; §6m's
attainable-floor refusal could not cost one by construction. The refusal added
here turns away inputs that would have rejected, including inputs whose change
really is at the anchor — on the registered sweep it costs the whole arm. What
it refuses is a verdict the design cannot **support** rather than one it could
not **reach**, and the dry run re-scores the counterfactual in every cell rather
than asserting the cost is small. The condition itself reads the step grid
against the registered window and nothing else, so it is decidable before a
checkpoint is sampled.

**Fifteenth, and it corrected this pass's own first attempt.** *A condition
built on a reference's rank cannot see what the reference is not exposed to.*
The refusal was first written to fire when the change-free reference outranked
the controls — which looked like the right shape and is not, because the
reference is a **noiseless** profile and a realised change-free series is a
noisy one, so the reference outranks even the change-free members of a family.
Across the whole *k* = 0…19 sweep its rank is flat at 0.050–0.051 while the rate
it was meant to track runs 1.000 → 0.050. Sweeping the axis is what showed it;
the sweep is in the artifact because it is what changed the design.

**Sixteenth, and it is the useful one for the six queued rows.** *"Matched on
what" has to name the quantity the statistic degenerates on.* §6n's taxonomy put
"one subspace against matched-dimension controls" (`P6-R4`) in the safe column
with "nothing to mismatch". The anchor arm is the counter-example:

| entry | statistic | the quantity it degenerates on | controls matched on it? |
|---|---|---|---|
| `P6-R4` | one subspace against matched controls | subspace dimension | **yes**, by construction |
| `CLAIM-B` anchor | one location against a fixed window | where the grid puts an unlocated profile | **no** |
| `CLAIM-B`/`P-I1` mutual | a difference of two locations | — cancels under the pairing null | n/a |

An absolute quantity against matched controls is safe only when the controls are
matched on its degenerate input. For `P6-R1`, `P6-C1`, `P5b-A1`, `P5b-A2`,
`P-SA1` and `P-I4` — the rows this document lists as already naming a matched
control — that is a question to answer before the control is built.

**`P-I1` was not changed, and the measurement is why.** It is the mutual arm
alone, whose null permutes the pairing and therefore keeps both series' real
per-head locations on both sides of every draw, so a pull that moves every
location the same way cancels. Measured on the **registered cheap sweep** — the
grid where the anchor arm fails hardest, because measuring it on a friendlier one
would be choosing the easy case — it holds at 0.045–0.065 across four H0
families, including one in which neither series changes anywhere. That is
`P6-R4`'s precedent used a second time: leaving an entry alone is a decision, and
a decision with nothing behind it is the position §6h's construction was in for
two passes.

**What the list of pre-computed requirements on a pilot now says.** Five, and
the new one is the first that constrains **which checkpoints are sampled**
rather than what is measured at them. `CLAIM-B` needs 19 control series *and* a
sweep whose uniform-profile midpoint falls outside 512–2000 — the registered
cheap sweep puts it at step 955 and fails, and Pythia's full every-1000 schedule
clears that condition but pushes the noise share to 0.63, where a real change at
the anchor is dragged out of the window and the arm loses its power instead.
Neither sweep the project has satisfies both ends. `CLAIM-C` needs at least 23%
of its candidate cells to dissent in sign and at least five prompts whose usable
metrics do not split evenly. `P-ST1` needs dim `U_pos` comparable to the
dimension the token population occupies.

And a smaller one, recorded because it is the seventh session running.
`check_record` guarded this pass's headline with `(value or 1.0) > 0.10` — and
the value it guards is a discriminating power that *should* be 0.0, which is
falsy, so the fallback fired on the healthy artifact and reported the finding
missing. It was found by running `--check` on the file that had just been
generated, which is the same habit that found §6g's rounding defect, §6h's audit
arm, §6i's discarded-null power figures, §6k's α on a shoulder, §6l's empty drop
slab, §6m's single-draw counterfactual and §6n's default argument. Nothing was
failing.


## The queue closes: `P-S1`, `P-T1` and `P-M1` (2026-08-27)

`tools/dry_run_p_s1.py` → `claims/audits/p_s1_dry_run.json` and
`tools/dry_run_p_t1_p_m1.py` → `claims/audits/p_t1_p_m1_dry_run.json`,
`POPPER_PLAN.md` §6p. **All nine adjudicable rows have now been run on an input
whose correct verdict is fixed a priori, and every one of the nine changed
something.** The queue this document opened on 2026-08-25 — *"a second entry
ahead of it for each row already converted"* — is finished.
`claims/adjudications/` is still empty.

**Seventeenth lesson, and it is the one to ask of the next construction
first.** *A floor computed from a draw count is a claim about the call; a floor
computed from the data's marginals is a claim about the design, and only the
second one is what "could this have rejected?" means.* `core.nulls.p_from_null`
reports `resolution` = 1/(n_draws + 1), which is honest about the sample and
was being read as the design's floor by two entries at once. Both statistics
are discrete, so the null puts a lump of mass on the observed value: `P-T1` at
five heads with two candidates has an exact floor of **0.100** against a
reported 0.0005, and `P-M1` at twelve layers with one violation **0.083**
against the same — 200× and 167×. At those designs no input whatever could
clear α, and both were reporting "not significant" instead.

The floor is arithmetic on the design, so it is checkable **before any data
exists** — which makes it the cheapest of the four defect kinds these nine
passes found, and the one worth asking of a construction first.

**Eighteenth, and it is about the refusal rather than the floor.** *There are
three kinds of "this costs no verdict", and which one a record claims is part
of the claim.* §6l's informative-row refusal had to be **re-scored against a
counterfactual**, because the floor and the p came from different code. §6m's
attainable-floor refusal cost none **by construction**, from the 2m bound.
`P-T1`'s can be **enumerated** — every attainable arrangement at every refused
configuration, listed, none clearing α — and `P-S1`'s costs none because there
was no correct p to remove: `Q_k`'s i.i.d. floor depends on `m`, so the
comparison the statistic makes does not exist across different `m`. A measured
zero, a proved zero and an enumerated zero are not the same claim and the
records say which.

**Nineteenth, and it is the largest Type-I number this registry has produced.**
*An input a design cannot compare will be compared anyway unless something
refuses it.* `P-S1` draws its null at the **trained** arm's `(m, d)` and
re-references both arms against that baseline. Nothing checked the step-0 arm
matched. On **two i.i.d. arms** — H0 realised exactly — a difference of two
clusters in thirty-two rejects at **1.000**, and in the direction that confirms
the prediction; the other direction sends p to 1.000 and the design can never
win. Unequal cluster counts are the expected case, not a stress test.

**Twentieth, and it is the pair where this document's opening argument has the
most force.** *Two entries under the SAME claim that share an instrument are
not two factors.* `P6-R2`/`P6-R4` record a shared projector and
`CLAIM-B`/`P-I1` a shared estimator — but those two sit under different claims,
so a common defect does not multiply inside one product. `P-T1` and `P-M1` are
both H-OPERATOR's and both classify the same head's `Wq`, `Wk` and `W_OV`.
Neither recorded it until now. A claim's E is the product of its predictions'
e-values, and two of them that one defect moves together is precisely how that
product inflates without anyone editing a number.

**And one thing that did not change, recorded because leaving an entry alone is
a decision.** `P-S1`'s reported floor **is** attainable — its statistic is
continuous, so ties have probability zero and a perfect input lands on
1/(n_null + 1) every time. That is the claim that failed for `P-ST1`, `P-T1`
and `P-M1`, all three discrete, so it was checked rather than assumed.
`P6-R4`'s precedent, used a third time.

### What the nine found, sorted by what was wrong

| what was wrong | where | checkable before data? |
|---|---|---|
| a reported floor that was not the design's | `P-ST1`, `CLAIM-C`, `P-T1`, `P-M1` | **yes** — it is arithmetic on the design |
| a null that randomised more than the claim is about | `P-ST1`, `P6-R2` | no — it took an H0 family the calibration lacked |
| a statistic partly determined by the measurement grid | `CLAIM-B` | **yes** — the grid's own midpoint against the window |
| an input the design cannot compare, scored anyway | `P-S1` | **yes** — the two arms' configurations |

Three of the four are checkable with no data at all, and all three were missed
anyway. For the six rows this document lists as still naming a matched control
— `P6-R1`, `P6-C1`, `P5b-A1`, `P5b-A2`, `P-SA1`, `P-I4` — the order to work in
is now: compute the attainable floor, name what the statistic degenerates on,
check what the measurement grid contributes to it, and only then build the
control.

### What the pilot must produce, after nine passes

Six requirements, none of which any existing sweep satisfies, and they
constrain different things:

| claim | requirement | constrains |
|---|---|---|
| `CLAIM-B` | 19 control series | what the sweep measures |
| `CLAIM-B` | a grid whose uniform-profile midpoint falls outside 512–2000 | which checkpoints it samples |
| `CLAIM-C` | ≥ 23% of candidate cells dissenting in sign, and ≥ 5 prompts whose usable metrics do not split evenly | what the contrast looks like |
| `P-ST1` | dim `U_pos` comparable to the dimension the population occupies | the projector's shape |
| `P-T1` / `P-M1` | enough heads, and enough layers or violations, for the design floor to clear α | how large the run is |
| `P-S1` | both arms clustered to the same count | how the run is clustered |

That list is what stands ahead of converting the next `needs-null` row, and it
is the real output of nine passes of running gates on inputs whose answers were
already known.
