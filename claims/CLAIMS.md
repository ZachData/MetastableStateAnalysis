# CLAIMS.md — the main hypotheses, and what each one is adjudicated by

POPPER_PLAN.md item B2. Written 2026-08-23.

Every registered prediction names exactly one claim below. A claim's e-process
is the product of the e-values from *its* predictions and nothing else
(`core/evalues.py`), so the claim boundaries decide what evidence can support
what — this file is therefore load-bearing, not an index.

## Why the claim layer exists at all

Predictions in this project were registered per-phase: `PREDICTIONS.md` holds
the transition-project claims and Phase 1c's six, `p6_subspace/report_6.py`
holds twelve more, `p5b_manifold_steering/p5b_report.py` holds nine. Each set
is internally coherent. What was missing is the statement of *what larger
claim each set is evidence for*, which is exactly what an e-process needs:
aggregating evidence requires knowing what the evidence is about.

The phases stay as they are — they are **instruments**, and renaming twelve
directories to match a claim taxonomy would break every artifact stem, test
fixture and path in the project to buy nothing (POPPER_PLAN.md §C3). This file
is the layer above them.

## The six claims

### H-RESIST — trained weights actively resist the architecture's collapse dynamics

The project's founding observation. Under the Geshkovski et al. dynamics every
token is driven toward a single collapsed point; GPT-2-large does not get
there, and its energy is flat-to-slightly-decreasing across layers 5–30 under
trained weights while staying close to monotone under random ones.

**Null (H0-RESIST):** the observed departure from collapse is what the
architecture and initialization give for free — training adds nothing.

Instruments: `p1_mstate_tracking`, `p1b_hemisphere`, `p1c_frames`, `core/nulls.py`.
Predictions: `P-γ1`, `P-γ2`, `P-H1`, `P-S1`, `CLAIM-A`.

**`P-S1` was run on inputs whose answer is known on 2026-08-27 and gained an
`(m, d)` refusal** (`POPPER_PLAN.md` §6p, `claims/audits/p_s1_dry_run.json`).
`p_value_p_s1` takes `m` and `d` from the **trained** arm, draws its null
there, and re-references both arms against that one baseline — which is right
when the two arms sit at the same configuration, and nothing checked that they
did. Since `E[Q_k] = 1/m` for i.i.d. points, a step-0 arm at a different
cluster count is divided by a baseline that is not its own, and its ratio is
off by roughly `m_trained/m_step0` — which enters the statistic as a
*difference between the arms*, the exact shape of the effect P-S1 predicts.

Measured on **two i.i.d. arms**, where the correct verdict is "no difference"
at every row, a difference of **two clusters in thirty-two** takes the
rejection rate to **1.000**; and the error runs both ways, since more step-0
clusters sends p to 1.000 instead and the design can never win. Unequal counts
are the expected case rather than the exception — clustering runs per
checkpoint and a random-weight model's activation geometry is not a trained
one's.

The gate now refuses on a mismatch, and on a step-0 arm that does not report
its own `(m, d)` to be checked. It costs nothing **by construction**: `Q_k`'s
i.i.d. floor depends on `m`, so "closer to a spherical design" is not a
comparison that exists across different `m` and no baseline choice rescues the
row. **So this claim carries a pre-computed requirement too:** both arms must
be clustered to the same count rather than each to its own best `k`.

**The sharpest threat to this claim is internal, and already recorded.**
`UPDATE_PLAN.md` §5.2 found that `MATH.md` §8's step-size definition
understates $T_\text{eff}$ by ~5.7×, in the direction that would make the
headline an artifact of depth rather than of learned weights. P-γ2 is stated so
that the outcome that would hurt is the one predicted.

### H-TRANSFER — the phenomenology is a property of trained transformers, not of GPT-2-large

**Null (H0-TRANSFER):** the trained-vs-random contrast is architecture-specific;
`pythia-1.4b-random` does not reproduce `gpt2-large-random` phenomenology, nor
checkpoint 143,000 the trained one.

Instruments: the replication gate (`UPDATE_PLAN.md` execution-order item 6).
Predictions: `CLAIM-C`.

**This claim carries a hard stop.** `PREDICTIONS.md`: "If this fails, no
checkpoint-sweep work (items 9–11) proceeds past the gate." Item B8 wires that
sentence to this claim's e-process, so the stop is a check rather than a
sentence to be argued with at the moment it binds.

**`CLAIM-C` became the first constructed null (2026-08-24), and the stop is now
three-way.** `p1_mstate_tracking/replication_gate.py` reads "reproduces" as
sign-concordance of the trained-minus-random contrast over the six per-layer
series `CHECKPOINT_METRICS` registers, with an exhaustively enumerated
sign-flip null whose exchangeable unit is the prompt. The gate returns
TRANSFERS, FAILS-TO-TRANSFER or INSUFFICIENT; it stops the sweep on the latter
two but records only FAILS-TO-TRANSFER as a falsification, since an e-process
reports insufficient evidence and never a null accepted. Only the one-sided
`greater` p-value is calibrated into this claim's E — the reciprocal test that
separates the two stop branches is a decision input and stays out of the
product. `POPPER_PLAN.md` §6f records the four choices the registered wording
left open, including that the criterion adjudicates the *contrast* rather than
the two absolute reproductions the statement's words name.

**The p that enters this claim's E is homogeneity-corrected (2026-08-24).** The
gate's one real limitation is that its exchangeable unit — the prompt — is not
independent across draws, since every prompt runs on the same model. §6f
measured what that costs at the two ends of the range and left the middle,
where any real run lands, uncontrolled. `claims/calibration/claim_c_homogeneity.json`
now holds the measured H0 rejection rate across that whole range, and the
reported p is `max(p_exact, R(sign_homogeneity, p_exact))` — the exact
enumeration's p, blunted to the measured rate wherever the measured rate is
worse, and never sharpened. Above roughly homogeneity 0.80–0.85 the gate
refuses instead of correcting, on the derived ground that a *perfect* result
would not survive its own correction there. `POPPER_PLAN.md` §6g records the
two decisions this required and the out-of-sample validation.

**The gate was run on inputs whose answer is known, and it has an admissible
band (2026-08-25).** `tools/dry_run_claim_c.py` puts one model in as *both* the
reference and the candidate — every cell is then concordant and the correct
verdict is TRANSFERS a priori — and sweeps the candidate's own sign homogeneity.
`claims/audits/claim_c_dry_run.json` records the result and `POPPER_PLAN.md` §6j
reads it. Three things belong here rather than only there.

**The criterion is sound.** On a perfect input every leave-one-out subset
returns exactly the attainable floor and the intersection-union max is that same
floor, so the unanimity axis does not bite on a unanimous input. That had never
been checked.

**But above `sign_homogeneity` 0.7708 at eight prompts the gate is a constant
function** (0.8125 as first measured; the band tightened on 2026-08-25 when the
informative-row floor changed what "conditional on emission" conditions on —
`POPPER_PLAN.md` §6l). The corrected-attainable-floor refusal fires on *every* input,
including a perfect one, at every concordance count from 0 to 48 — so neither
TRANSFERS nor FAILS-TO-TRANSFER is reachable and this claim's hard stop fires
unconditionally there. That is not a Type-I defect: `sign_homogeneity` measures
prompt redundancy under H0 and effect *uniformity* under H1, and the correction
cannot separate them, so the cost lands as power and lands hardest where the
effect is most uniform. **This claim's gate is powered against a contrast with a
prompt-specific signature that transfers, and unpowered against a contrast with
one uniform direction that transfers.** Blog 1's phenomenology is the second
kind.

**Which makes it a requirement on the pilot, computed before it runs** — the
same shape as `CLAIM-B`'s 19 control series below. At least 11 of the candidate's
48 cells must dissent in sign, about 23%, or the gate cannot return a verdict at
all — 9 when first measured, raised with the band on 2026-08-25. A **second** requirement joined it on 2026-08-25 and it points the other
way: at least five prompts must have usable metrics that do **not** split evenly.
A prompt whose label flip does not change the statistic — every cell dropped, or
an even number of usable cells split exactly half and half, which with six
metrics is 3–3 and happens to 20/64 of rows under H0 — is enumerated through all
2^n null patterns and never counted, so the floor is `2^-k` in the *k* prompts
that can move. Before that was checked the gate could be handed a table
**perfect on four prompts** and 3–3 on two, return p = 0.0769 (exactly that
table's own floor) and report it as "not significant". Refusing there costs
nothing: both tails share the floor, so nothing is turned away that could have
cleared α, and measured across five H1 strengths P(TRANSFERS) is unchanged to
four decimals. `POPPER_PLAN.md` §6l. Under *independent* prompt signs homogeneity concentrates at 0.637 and the
refusal essentially never fires, so the band is not tight against chance; it is
tight against a clean effect, which sits at exactly 1.0. More prompts do not
move it. Whether pythia-1.4b's contrast dissents enough is an empirical fact
nothing here yet knows, and it is now a stated precondition rather than a
surprise at the far end of a sweep.

### H-EMERGE — collapse-resistance emerges at circuit-formation events

**Null (H0-EMERGE):** clustering dynamics and circuit formation are independent;
the energy-monotonicity break and Fiedler drop do not co-locate with the
literature's checkpoint anchors.

Instruments: the Pythia-410M pilot (item 8), `core/checkpoint_frames.py`,
`core/changepoint_colocation.py`.
Predictions: `CLAIM-B`.

Note the asymmetry, which `PREDICTIONS.md` states and which should survive into
the e-process: a failure here is *informative on its own terms* — it re-anchors
the 1.4B checkpoint schedule rather than invalidating the sweep. An e-process
records "insufficient evidence", never "null accepted", which is the right
shape for that.

**`CLAIM-B` became the third constructed null (2026-08-24), and the asymmetry is
now a verdict branch.** `core/changepoint_colocation.py` locates each series'
change as the centroid of its change-mass profile on the log-step axis and asks
whether two such locations sit closer together — and closer to the
pre-registered ~512–2000 window — than a matched control population allows. The
gate returns CO-LOCATES, RE-ANCHORS or INSUFFICIENT, and **RE-ANCHORS is the
falsification**: the separation is positively shown rather than inferred from a
failure to reject, which is exactly what this claim's falsifier says is worth
having. Only the one-sided `greater` p enters this claim's E; the reciprocal
test that separates the branches is a decision input.

Three things about it belong here rather than only in `POPPER_PLAN.md` §6i.

**The registered null was wrong, and measurement is what showed it.** The entry
said "a permutation null over checkpoint order". Measured, permuting the value
series rejects under H0 at 0.45 and permuting the interval increments at 0.32,
because both dissolve the concentrated change profile the statistic is built on
and leave the null with far too little variance. An enumerated circular shift
is honest at 0.065 but assumes changepoints are uniform on the interval grid,
and with both series changing early — as everything does early in training — it
reaches 0.103. What replaced it is a matched control series: the control for a
series at one layer is that series at another layer, combined as a permutation
of the *pairing* between the two series' layers. That holds nominal under both
families.

**The obvious estimator could not carry it.** `detect_transitions` divides by a
log-step spacing that varies 4.6× across a Pythia sweep, so under the null its
argmax of a permuted value series lands on the tightest-spacing interval 44.7%
of the time and a binary
"the two top intervals coincide" statistic is floored near 0.29. The
change-mass centroid that replaced it carries no placed constant at all.

**And the gate will most likely refuse, for a reason computed before the pilot
runs.** The two anchor arms have no permutation available — nothing relabels
"unrelated to the literature's anchors" — so each needs a reference population,
and at α = 0.05 that is 19 control series measured on the same sweep at the
same layers. A cheap-tier sweep measuring six metrics has six. Under the gate's
unanimity rule a refusing arm refuses the whole gate, so **this is a
requirement on what the pilot must measure**, not a result.

**`CLAIM-B` was run on inputs whose answer is known on 2026-08-27, and its
anchor arms changed** (`POPPER_PLAN.md` §6o,
`claims/audits/claim_b_p_i1_dry_run.json`). One dry run covered this entry and
`P-I1`, because they share one estimator. Five things belong here rather than
only in the plan.

**A change location is partly a property of the sweep grid.** The location is
the centroid of a change-mass profile — a weighted mean of the sweep's interval
midpoints — so mass spread evenly over the sweep lands on the grid's *own*
midpoint, exactly. Per-checkpoint noise puts rectified mass in every interval,
so every real location is a mixture of where the series changed and where the
grid's midpoint is, weighted by the noise's share of the mass. That share grows
with the interval count, which makes a **denser** sweep worse; the dry run
predicts the centroid in closed form from the interval count and the noise, and
the worst disagreement with the measurement across three grids and four noise
levels is 0.061 in log10-step.

**The registered instrument is the one grid on which that is fatal.** This
entry's instrument field names a "20-30 checkpoint cheap-tier sweep", and that
grid's uniform-profile midpoint is step **955** — inside this claim's own
registered 512–2000 anchor window. So a series that changes *nowhere* receives
the anchor arm's maximum statistic. Against controls that all carry a located
change, the arm rejects on a change-free input at **1.000**, which is exactly
its rate on a perfectly anchored one: its discriminating power there is
**zero**. The general rate is `1/(k+1)` with *k* the number of controls that
are themselves change-free, measured against that closed form at every *k*.
The coincidence that hid it is that the cheap sweep's midpoint sits almost
exactly where this claim's anchor is, so on the grid the construction was
calibrated for the bias points at the answer.

**`anchor_arm` now refuses there, and the refusal costs verdicts.** The
condition is the step grid against the registered window and nothing else — no
controls, no observation, no α — so it is decidable before a checkpoint is
sampled. Unlike `CLAIM-C`'s informative-row refusal (measured at zero power
cost) and `P-ST1`'s attainable-floor refusal (costs none by construction), this
one turns away inputs that would have rejected, including inputs whose change
really is at the anchor. What it refuses is a verdict the design cannot
**support** rather than one it could not **reach**, and the dry run re-scores
the counterfactual in every cell rather than asserting the cost is small.

**So the fifth pre-computed requirement on the pilot is about which checkpoints
it samples, not which metrics it measures.** The anchor arms need a sweep whose
uniform-profile midpoint falls outside 512–2000. The registered cheap sweep
puts it at step 955 and fails. Pythia's full every-1000 schedule puts it at step
31,496 and clears the condition — but at that density the noise share reaches
0.63 and a real change at the anchor is dragged out of the window, so the arm
has no power there either. Neither sweep the project has satisfies both ends,
and that is a design question for the pilot rather than a surprise waiting at
the end of one. It sits beside the 19-control requirement rather than replacing
it.

**One thing the dry run settled that had never been asked.** This claim's
falsification branch fires — RE-ANCHORS is reachable on the input built for it
— and it fires with no margin: both anchor arms' reciprocal p is floored at
`1/(n_controls + 1)`, which is *exactly* α at nineteen controls, so the branch
needs the observed series to rank strictly worst of twenty in both arms at once.
The 19 control series recorded above as the minimum for CO-LOCATES are therefore
also the exact minimum for the falsifier, in either direction.

### H-BUDGET — the network spends a bounded dimensionality budget on particles that must stay individuated

Phase 5c's object of study, promoted by plan v2 to be the project's. Motivated
by effective rank plateauing near ~200 across models whose $d_\text{model}$
spans 768–1600: if the network simply used what training gave it, effective
rank should scale with $d_\text{model}$, and it does not.

**Null (H0-BUDGET):** the unclustered population is inert overflow — the
architecture failing to collapse it, with no computational role.

Instruments: `p5c_unclustered`, `p5_single_mstate_analysis`, `core/particles.py`,
`core/population.py`.
Predictions: none registered yet. Phase 5c's Groups C and D are designed but
their predictions are not in the registry; that is a queued chunk.

**Design note carried forward:** `design-5c.md` records that the attention-flip
result (trained models route attention *toward* unclustered tokens at 1.6–2×,
sign-flipped from random weights) already reshapes this claim's null before any
Group D experiment runs — the null to design against is a moderate-to-large
effect, not "no preference either way". Any prediction registered here must be
powered accordingly.

### H-OPERATOR — collapse and anti-collapse are attributable to stated operator conditions

Not "something in the weights resists collapse" but: the resistance sits where
the theory says it must, in heads whose $V$ eigenstructure and $QK$ symmetry
place them outside the gradient-flow regime.

**Null (H0-OPERATOR):** energy-monotonicity violations are unrelated to where
the gradient-flow hypotheses fail; the operator classification carries no
information about activation geometry.

Instruments: `p2_eigenspectra`, `p2b_imaginary`, `p2d_operator_activation` (live);
`archive/p6_subspace` (archived 2026-08-22).
Predictions: `P-T1`, `P-M1` (active, Phase 2d); `P6-A2`, `P6-I1`, `P6-I2`,
`P6-R1`–`P6-R5`, `P6-C1`, `P6-DD1`, `P6-DD2`, `P6-D5` (**dormant** — see below).

**Ten of this claim's fourteen predictions are dormant.** `P-T1` and `P-M1` are
instrumented by Phase 2d, which is live; `P6-R2` and `P6-R4` came back on
2026-08-24 when their projector path was rebuilt in `p6_subspace/` against
`core/particles.py`, per `archive/README.md` rule 2 that nothing is salvaged by
copying. `archive/p6_subspace/` stays frozen.

**This claim was described here as already carrying a falsification, and it does
not.** P6-R2 predicted LDA alignment with the real repulsive subspace
$U_\text{neg}$; the 2026-04 ALBERT run gave 0.887 alignment with the *imaginary*
subspace $U_A$ against 0.067 with $U_\text{neg}$, and 0 of 49 layers in the
predicted direction. P6-R4 inverted the same way. The caveat recorded against
both was that 49 ALBERT layers are not 49 independent observations.

That caveat was real and it was not the binding one. `p6_subspace/math-6.md`
§7.2 names a third explanation: **the comparison is not dimension-normalized.**
$\mathbb{E}[\lVert P_U v\rVert^2] = \dim U / d$, so raw alignment scales with
the subspace's dimension, and the projector build's resolution order makes
$U_\text{neg}$ the doubly-shrunk bucket.
`claims/audits/p6_projector_labels.json` measures
$\dim U_A / \dim U_\text{neg} = 24.9$ at `albert-xlarge-v2`'s exact shape
against an observed ratio of 13.2 — **the correction is larger than the effect
it would explain.** Chance-normalized, the two numbers read 0.960 for $U_A$ and
1.805 for $U_\text{neg}$, which is the *predicted* direction; but those dims come
from random OV matrices at ALBERT's shape rather than ALBERT's trained weights,
so that bounds the correction and is not itself a result.

So the recorded inversion is **not evidence in either direction**, and no
exchangeable unit would have made it so. It is withdrawn as a falsification and
kept as a measurement whose reading is unresolved. The one number that settles
it — the actual per-layer $\dim U_A$ and $\dim U_\text{neg}$ — is computed by the
projector build on every run and was never reported.

**`P6-R2`'s null was changed on 2026-08-26 and `P6-R4`'s was not**
(`POPPER_PLAN.md` §6n, `claims/audits/p6_r2_r4_dry_run.json`). The check was
forced by §6m: it retired the matched-dimension random orthogonal subspace pair
for `P-ST1`, and that construction is §6h's, introduced here. It randomises the
union of the two channels together with the split between them, so it rejects
when the pair is unusual *as a pair* — and "span($U_\text{neg}$ + $U_A$) sits
above chance against this layer's separating direction" is a fact about the
union, not about which half the operator calls repulsive. Measured on an H0
whose split is uniformly random by construction, so that the correct answer is
*do not reject*, the retired null's rejection rate is a monotone trend in that
alignment — 0.000 at chance and **0.155** at 3.9× chance, against a nominal
0.05. The replacement holds the union fixed and re-splits it at the observed
dimensions and does not trend: 0.047 and 0.068 at the same two ends over 600
replicates, pooling with an independent 1000-replicate run to about 0.056 at
the aligned end — at or marginally above nominal, and 9.7 standard errors
below the retired null there. Power is unchanged at 1.000.

**`P6-R4` is untouched, and that is a decision with a measurement behind it.**
It compares *one* subspace against matched-dimension random ones, so it has no
union and no split for the defect to reach, and its rate holds at 0.040–0.048
where a high-variance $U_S$ captures 3.4× the variance chance would give.
Which of the two happens is decided by **the statistic, not the claim**: a sign
of a difference saturates and fails hardest, a difference of chance-normalized
alignments cancels a common elevation to first order and fails late, and a
single subspace against matched controls has no common elevation to mismatch.
`claims/EVALUABILITY.md` carries that taxonomy forward to the six queued rows
whose predictions already name a matched control.

**`P-T1` and `P-M1` were run on inputs whose answer is known on 2026-08-27,
and both gained an attainable floor** (`POPPER_PLAN.md` §6p,
`claims/audits/p_t1_p_m1_dry_run.json`). Both reported
`core.nulls.p_from_null`'s `resolution` — 1/(n_perm + 1) — as the smallest p
they could express. That is the **sample's** limit. Both statistics are
*discrete*, so the null puts a lump of mass exactly on the observed value and
the real floor comes from the data's marginals. P-T1 at five heads with two
candidates has an exact floor of **0.100** and P-M1 at twelve layers with one
violation **0.083**, both against a reported 0.0005 — so at those designs no
input whatever could clear α, and both were returning "not significant"
instead.

Both floors are now exact — hypergeometric for P-T1, `∏(multiplicity)!/n!` for
P-M1 — and neither contains a draw count, so more permutations do not move
them: more heads, more layers or more violations do. The refusal costs nothing
**by enumeration** rather than by measurement, since the floor is the smallest
value of the same discrete p. Three designs now never emit, which is the
pre-computed requirement in the form a reader can see, and a 36-layer model
with a single energy-monotonicity violation is one of them.

**And these two share an instrument, which matters more here than in either
pair that already records one.** Both classify the same head's `Wq`, `Wk` and
`W_OV`, and the extraction deciding which head's weights are which is shared —
the defect class §6h spent an audit ruling out one phase over. `P6-R2`/`P6-R4`
record their shared projector and `CLAIM-B`/`P-I1` their shared estimator, but
those two sit under *different* claims. **`P-T1` and `P-M1` are both this
claim's**, so two e-values that one defect moves together multiply into one
product — the specific way a claim's E inflates without anyone editing a
number, which is `EVALUABILITY.md`'s opening argument. Both entries now say so.

The same pass also settled `status-6.md` item 5, which `design-6.md` had
pre-registered as the prerequisite: the Schur block labelling is **correct**,
verified on planted structure and against an independently derived spectrum,
with two deliberate mislabellings caught. See `POPPER_PLAN.md` §6h.

### H-BRIDGE — natural-language-interpretability constructs are particle-dynamical objects

Not "the two vocabularies can be translated" — that is unfalsifiable — but: the
particle reading makes *differential* predictions, correct where the standard
account is silent or says something else.

**Null (H0-BRIDGE):** the particle-paradigm definition of each construct adds no
predictive content over its standard definition; where they differ observably,
the standard account is right.

**Phase 7 is this claim's instrument.** `p7_motifs/` opened on main 2026-08-22 as
"the mechinterp/particle bridge" and asks the same question this claim does, with
a sharper formalization: a named mechinterp phenomenon is a **motif** — a
recurring structure of typed particle interactions — that can be counted, tested
against a matched null, and tracked across training. Its seven-motif alphabet is
fixed in advance precisely so the phase cannot become a motif zoo, and `relay`
*is* the induction head restated as a particle motif.

This branch independently produced a `docs/PARTICLE_ONTOLOGY.md` covering the
same ground, down to the same constraint — Phase 7 states it as "the trap this
phase is designed against is producing a glossary"; the ontology stated it as "an
entry with no differential prediction is a glossary line and does not belong
here". Two documents describing one bridge is how doc drift starts, so the
ontology was folded into Phase 7 and deleted (2026-08-23). Its four differential
predictions survive as registry entries, renumbered into Phase 7's own `P-*`
scheme rather than kept under a second `PB-*` convention:

| was | now | where it sits in Phase 7 |
|---|---|---|
| `PB-IND1` | `P-I5` | the interventional arm `P-I1`–`P-I4` lack: ablate the relay motif's edges, require both a geometric effect at the matched positions and a logit effect |
| `PB-STEER1` | `P-ST1` | the steering entry's falsifiable half — sign of the effective-rank change tracks the V-decomposition at matched norm |
| `PB-ABL1` | `P-AB1` | the patching entry's recapture-vs-propagation question, as a growth exponent in remaining depth |
| `PB-SAE1` | `P-SA1` | the SAE entry, which asks the identical question independently. Instrument frozen; registered and unrun |

Predictions: `P-I1`–`P-I5`, `P-ST1`, `P-AB1`, `P-SA1` (active), and the nine
`P5b-*` (dormant — Phase 5b is archived).

**`P-I1` gained a null the same day as `CLAIM-B`, and it is the same one**
(2026-08-24). `EVALUABILITY.md` had named the two as sharing a construction and
said they should be built together rather than each inventing one;
`p7_motifs/formation_gate.py` is the thin half over
`core/changepoint_colocation.py`. The unit is the **head**, which is not a
convenience — `PREDICTIONS.md`'s first Phase 7 adjudication constraint fixes it,
and because the null permutes head pairings an edge-level *n* has no way in.
P-I1 gets one arm and no anchor arm: it names no literature anchor, and
inventing one would be the glossary error this phase is designed against.

**The two entries sit under different claims, and that is precisely why their
dependence had to be written down.** `CLAIM-B` is H-EMERGE's and `P-I1` is
H-BRIDGE's, so there is no `P5b-B1`/`P5b-B3` double-counting problem — but one
shared *estimator* is a common-cause failure mode, and both `null_construction`
fields and both ledger records say so, the precedent `P6-R2` and `P6-R4` set for
their shared projector.

**`P-I1` was run on the same inputs on 2026-08-27 and was NOT changed, and that
is a decision with a measurement behind it** (`POPPER_PLAN.md` §6o). The dry run
that changed `CLAIM-B`'s anchor arms covers this entry too, since the estimator
is shared. `P-I1` is the **mutual arm alone** — a difference of two locations,
with a null that permutes the pairing and so keeps both series' real per-head
locations on both sides of every draw. The pull toward the sweep grid's own
midpoint that takes `CLAIM-B`'s anchor arms to a rejection rate of 1.000 on a
series with no located change is common to both series here, and cancels.
Measured on the **registered cheap sweep**, the grid where the anchor arm fails
hardest, the mutual arm holds at 0.045–0.065 across four H0 families including
one in which neither series changes anywhere. Leaving an entry alone is a
decision, and the precedent is `P6-R4` one pass earlier: without the number the
difference between the two entries would rest on an argument about their
statistics.

**Which sharpens the taxonomy `POPPER_PLAN.md` §6n left.** §6n put "one
subspace against matched-dimension controls" in the safe column with "nothing
to mismatch". The anchor arm is the counter-example that says what that column
requires: an absolute quantity against matched controls is safe only when the
controls are matched on **the quantity the statistic degenerates on**.
`P6-R4`'s controls are matched on dimension, which is what drives its
statistic; the anchor arm's are matched on the sweep and the units, and what
drives its statistic is where the grid puts a profile carrying no location.
`claims/EVALUABILITY.md` carries that to the six queued rows whose predictions
already name a matched control.

**The construction cannot discharge this phase's tautology risk, and says so.**
Adjudication constraint 2 records that the behavioral induction score is "mean
attention on induction pairs" while a motif defined as "attentive edge on
induction pairs" is the same number. Two identical series co-locate perfectly,
and no null detects it, because the null is over the pairing and a tautological
pair is tautological at every head. The gate refuses exactly-identical series
per head — the degenerate case, not the substantive one — and the independence
source stays a claim the analyst must make. The measured version of the same
problem is that a common per-unit factor produces a rejection rate of **1.00**
against 0.05; see `POPPER_PLAN.md` §6i.

**`P-ST1` gained a construction on 2026-08-25, and it is the entry that can
lose.** `p7_motifs/steering_gate.py`; `POPPER_PLAN.md` §6k. It is H-BRIDGE's
cheapest entry and the only registered bridge prediction where the particle and
standard accounts make *incompatible* rather than merely different predictions.
Four things belong here rather than only in the plan.

**The registered falsifier is not one this claim's e-process can carry.** It
reads "both arms move effective rank the same way, or the effect tracks ‖s‖ and
is insensitive to the decomposition" — and both clauses describe the **null**.
An e-process records insufficient evidence and never a null accepted, so
neither can enter the ledger. They map to INSUFFICIENT. The falsification
branch is INVERTS: attractive-dominant steering demonstrably *raising*
effective rank while repulsive-dominant lowers it, a reversal positively shown,
and it was checked to be a branch that can fire.

**The registered null was measured and retired, the second in three passes.**
Permuting the decomposition label across pairs treats *m* pairs as *m*
exchangeable units when every pair at one layer shares the tokens and both
subspaces; the rejection rate under a noisy H0 grows from nominal at 8 pairs to
0.17–0.22 at 150. What replaced it was the matched-dimension random orthogonal
subspace pair — `P6-R2`/`P6-R4`'s construction, arriving for the fourth time.

**And that replacement was retired in turn on 2026-08-26, on an H0 family the
calibration did not have** (`POPPER_PLAN.md` §6m). Matching the dimensions holds
fixed everything the statistic could read off dimension. It does not hold fixed
how much of the population each subspace *contains*, which is what a change in
effective rank is driven by: injecting along a direction the cloud already
occupies reinforces a large Gram eigenvalue and lowers effective rank, and
injecting along one it does not raises it. A random *k*-dimensional subspace
holds *k/d* of the population; `U_pos` and `U_neg` are cut from the model's own
OV eigenstructure and a residual stream is orthogonal to neither, so both hold
more — and against random pairs such a pair is unusual whichever arm is called
attractive. Measured where both arms are occupied above chance and the two are
*identical by construction*, so that a label swap is a distributional identity
and INSUFFICIENT is the only correct verdict: **up to 0.20 against a nominal 0.05, the inflation growing with the pair count**, in whichever direction the layer's realized asymmetry happened to fall.
All three H0 families the 2026-08-25 calibration measured put the cloud in a
subspace orthogonal to *both* arms — the one case where a matched-dimension
random pair is exchangeable with the observed one — so nothing could have seen
it.

**What replaced it randomises the split rather than the subspaces**, which is
the first time this project's recurring question — *what is being randomised?*
— has been answered by randomising **less**. The old null moved the union and
the split together, so it rejected on either, and "this pair of subspaces is
unlike a random pair" is a statement about the union rather than about the
decomposition `P-ST1` names. The gate now draws a uniformly random
*k*<sub>pos</sub>-dimensional subspace of span(`U_pos` + `U_neg`) and takes its
orthogonal complement *within that union*: dimensions, orthogonality, occupancy
and the whole spectral relationship to the layer's cloud are held exactly
fixed, and the observed split is one point of the same Grassmannian the null
draws from, so exchangeability under H0 is by construction rather than by
measurement. It holds at or below nominal on every family including the one
that retired its predecessor. It costs no power where the cloud fills the whole arm
(both nulls reach 1.000 in both directions there) and costs it as
dim `U_pos` grows past the dimension the population occupies — and power lost
that way was never power about the decomposition.

**Steering is a pure mean effect, which decided the one thing the wording left
most open.** Re-centring after injection annihilates the intervention exactly,
so the cloud's pre-existing mean offset competes with the injected one; on a
realistic residual stream the design as literally worded has no power, and at a
mean offset of five spreads it rejects more often under H0 than under H1.
Removing the *baseline* mean before injecting restores it. That was put to the
author before the module was written.

**And the pilot has a precondition, computed before it runs** — the third in
three passes, after `CLAIM-B`'s 19 control series and `CLAIM-C`'s 19%
dissenting cells. A uniform draw from `U_pos` carries only
dim(occupied)/dim(`U_pos`) of its energy into the subspace the token cloud
occupies, and the per-pair informative rate falls from 1.000 at ratio 1 to
0.000 at ratio 6. `claims/audits/p6_projector_labels.json` already records that
`U_pos` is the *un-shrunk* bucket in the projector build's resolution order,
which is the unfavourable side of this.

**`P-ST1` was the second entry run on inputs whose answer is known
(2026-08-26).** `tools/dry_run_p_st1.py` → `claims/audits/p_st1_dry_run.json`,
`POPPER_PLAN.md` §6m. `claims/EVALUABILITY.md` had said each converted row is
owed that treatment ahead of converting the next one; `CLAIM-C` had it and this
is the second. Three things belong here rather than only in the plan.

**The gate's reported floor was not the attainable one, and a design that could
not reject was returning "not significant".** `sum(D)` cannot exceed 2*m*, so
the smallest p a run can express is what an observation of 2*m* would receive —
and on a union the cloud occupies, many random re-splits already reach 2*m* and
tie it. Measured, the attainable floors at one pair are 0.11–0.17 where
`1/(draws + 1)` says 0.01. The gate now computes both tails' attainable floors
from the null it already has and refuses when neither can reach α. 2*m* is an
upper bound on the observation rather than an attainable value, so that floor is
a lower bound on what the run can express and the refusal can never turn away a
result that would have cleared α — the same argument, and the same defect, as
`CLAIM-C`'s informative-row refusal one claim over.

**A run can have only its FALSIFICATION branch reachable, and the record now
says so.** The two tails' floors are computed separately and are not equal, so
one can be out of reach while the other is not — and where the reachable one is
INVERTS, the design can return a falsification or nothing and nothing else. The
gate does not refuse there: one reachable tail is one reachable verdict, and a
refusal must cost no verdict. It records `reachable_tails` instead, because a
run whose only reachable verdict is the one that enters the ledger as a
falsification is a run a reader must be told about, and nothing else in the
record would say it. **Two pairs is the smallest design that emits at all** —
at one pair, on an input planted perfectly in either arm, both tails' floors sit
at 0.11–0.17 and the gate refuses — and the binding quantity is how often a
re-split of the same union reaches the maximum, not the draw count.

**And the validity statement this entry can make is sharper than any other in
the registry.** Draw the observed pair as a random re-split of a fixed union and
it is exchangeable with the null draws *by construction*, so
P(p ≤ α) ≤ α exactly, for any population, with no modelling assumption at all.
Measured over 200 draws on a population that occupies both arms — the family
that retired the previous null — it holds. Every other validity number in this
project is a rate under a modelled H0 family; this one's answer follows from the
construction, so a failure would localise to the implementation rather than to
the choice of family.

**`P-AB1` gained a construction on 2026-08-27, and it is the last unbuilt
bridge entry with a live instrument.** `p7_motifs/patching_gate.py`;
`POPPER_PLAN.md` §6q. `design-7.md` calls the patching entry the one place in
its translation table "where the particle account plausibly says something the
mechinterp framing does not already say", and the particle question is
**recapture versus propagation**: superlinear divergence in remaining depth is
propagation, flat divergence is recapture. Three things belong here rather than
only in the plan.

**The registered null could not have rejected anything, and one line of algebra
says so.** *"Permutation over ablation points once the fitted exponent is the
statistic"* permutes which point's real exponent meets which point's control
exponent — and a mean paired difference is **invariant** under that, since it is
mean β_real − mean β_control for every permutation. The null has no spread and
the design's floor is **1.000**. What replaces it is the exact sign flip of the
two arms' labels at one ablation point, which is exchangeable under H0 by
construction: if ablation removes a value from a sum, a real direction and a
structureless one of equal magnitude at the same layer are the same kind of
object. Six informative units is then the first design that can reject, and
because a prompt contributes the *sum* of its points' signs, an **odd** number
of ablation points per prompt is free while an even one is not — at six prompts,
seven points leave the design able to reject on every H0 draw and six leave it
able on 0.394.

**The statistic is not monotone in the quantity the claim is about, which is the
finding and is new to this registry.** Divergence is bounded, so the arm whose
divergence is larger at every layer reaches its ceiling sooner inside a fixed
window and its log-log slope **flattens**. On two arms carrying the same true
exponent where only the real one saturates sooner — which is what a real
ablation that propagates does — the gate returned `RECAPTURES`, its falsification
branch, on 0.98–1.00 of draws. The gate now refuses when either arm's divergence
is not a power law over the window; the first attempt at that refusal tested the
*paired* bend and was thrown out by measurement, because it let 48 of 100 draws
through and 0.979 of those still returned the falsification. The refusal costs
verdicts — §6o's fourth category, used a second time — and the calibration
re-scores the counterfactual in every family rather than claiming the cost is
small.

**And the exchangeable unit is not registered, for `P6-R2`'s reason.** Measured
on the same draws, one bit per ablation point runs to 0.235 under a per-prompt
shared factor where one bit per prompt holds at 0.029.
`REGISTERED_EXCHANGEABLE_UNIT` is `None` and `adjudicate_p_ab1` raises while it
is. A **fixed** offset between the real and the control direction populations —
real ablation directions are not isotropic and the controls are — is separated
by neither unit and reaches 1.000, which is `P-I1`'s shared-per-unit-factor
limitation in a second construction: diagnosed, stated, not removed.

**Why the P5b predictions stay here rather than under H-OPERATOR.** Sub-experiment
D asks whether behavioural geometry is carried by the real/symmetric subspace and
*not* by the imaginary/antisymmetric one (`P5b-D1`, `P5b-D2`). That is a claim
about what a steering vector *is* in particle terms — a coordinate-system
statement, not an operator classification.

## Dormant predictions

Nineteen of the thirty-eight registered predictions are **dormant**: their
instrument moved to `archive/` on 2026-08-22 and nothing live can produce their
p-value. Ten of the twelve `P6-*` (H-OPERATOR) and the nine `P5b-*` (H-BRIDGE);
nothing else. Phase 1c, Phase 2d and Phase 7 are live, so `P-γ1`, `P-γ2`, `P-H1`,
`P-S1`, `P-T1`, `P-M1`, the three `CLAIM-*` entries and all eight Phase 7
predictions stay active — and `P6-R2` and `P6-R4` rejoined them on 2026-08-24
when their instrument was rebuilt, which is the reversal the next paragraph
says is available.

Dormant is a status, not a deletion, and the distinction is the point. The
prediction was pre-registered, its falsifier is unchanged, and it has **not** been
withdrawn — `core/adjudication.py` refuses it and it contributes nothing to any
claim's E, but it stays counted and visible. Deleting a pre-registered prediction
because its apparatus went away would leave the record as the flattering subset of
what was actually predicted, which is the specific failure the pre-registration
gate exists to prevent. It reverses if the instrument is rebuilt — per
`archive/README.md`'s second rule, rebuilt against `core/particles.py` rather than
lifted.

`H-OPERATOR` had **no live path to adjudication** while twelve of its fourteen
predictions were dormant. It now has four: `P-T1` and `P-M1` on Phase 2d, and
`P6-R2` and `P6-R4` on the rebuilt `p6_subspace/`. That is a live path in the
sense of apparatus, not of evidence — no run artifacts exist here.

**The exchangeable unit is registered, and it is `"model"` (2026-08-25).** It
was deliberately left unregistered for two passes, because which unit may enter
an e-process is a scientific decision of the same class as `CLAIM-C`'s criterion
and taking it after seeing a p-value would void the guarantee. The author took
it, before any p-value on real activations existed. What it was decided against
is on the record: measured at 400 replicates, as the layers come to share one
direction `"layer"` runs 0.0525 → 0.0800 → 0.2325 → 0.2800 while `"model"` holds
at 0.045–0.0575 throughout, and under ALBERT's weight-tying the layers are as far
from independent as they get. `"model"` is therefore the conservative choice at
every point of that range rather than a trade. `POPPER_PLAN.md` §6l records it.
`adjudicate_p6_r2_r4` now refuses a result computed under `"layer"` instead of
refusing everything; it still adjudicates nothing, because there is still no
run artifact.

## Claim boundaries, and the one that is not obvious

`P-T1` and `P-M1` sit under H-OPERATOR rather than H-RESIST even though both
appear in `PREDICTIONS.md` alongside the Phase 1c predictions, because both are
statements about *which heads* rather than about whether resistance exists.
Putting them under H-RESIST would let operator-level evidence accumulate toward
a claim it does not bear on, which is the specific way a claim taxonomy can
inflate an e-process without anyone editing a number.

## Status

No claim has been adjudicated. `claims/adjudications/` is empty by design: an
adjudication record may only be written by `core/adjudication.py` (item B4,
not yet built) and only for a prediction the evaluability audit (item B5)
classified as `e-value`.

Nine predictions are adjudicable as of 2026-08-25 — `P-S1`, `P-T1`, `P-M1`,
`CLAIM-C`, `P6-R2`, `P6-R4`, `CLAIM-B`, `P-I1` and `P-ST1`. What is missing is
data, not apparatus: no run artifacts exist in this repo, so all nine are
validated on synthetic inputs with known answers. **All nine have now been put
through a dry run on inputs whose answer is known a priori** rather than only
through unit tests, and every one of them changed something:

| entry | dry run | what it found |
|---|---|---|
| `CLAIM-C` | 2026-08-25, `claim_c_dry_run.json` | an admissible band in its own input space, outside which the gate is a constant function |
| `P-ST1` | 2026-08-26, `p_st1_dry_run.json` | the reported floor was not the attainable one — and, before it ran, that the null it adjudicated was invalid on the H0 family a real residual stream presents |
| `P6-R2`, `P6-R4` | 2026-08-26, `p6_r2_r4_dry_run.json` | the same null, checked where it came from: invalid for R2 and not for R4 |
| `CLAIM-B`, `P-I1` | 2026-08-27, `claim_b_p_i1_dry_run.json` | a change location is partly a property of the sweep grid; CLAIM-B's anchor arms cannot tell an anchored change from no change at all on the registered sweep, and P-I1's mutual arm cancels the same pull |

| `P-T1`, `P-M1` | 2026-08-27, `p_t1_p_m1_dry_run.json` | both reported the draw count's resolution as their floor; both statistics are discrete, so the design's floor is set by the marginals and a perfect input returns a p hundreds of times larger |
| `P-S1` | 2026-08-27, `p_s1_dry_run.json` | the null is drawn at the trained arm's (m, d) and nothing checked the step-0 arm matched; two i.i.d. arms two clusters apart in thirty-two reject at 1.000 |

**All nine have now had one, and every one of them changed something.** None
was failing a test. The queue `claims/EVALUABILITY.md` opened on 2026-08-25 is
closed; what stands ahead of converting the next `needs-null` row is the list
of pre-computed requirements those nine passes produced, none of which any
existing sweep satisfies. `P6-R2` and `P6-R4` no longer
carry the extra refusal they used to: their exchangeable unit is registered as
`"model"` (above), so `adjudicate_p6_r2_r4` now turns away only a result
computed under the other unit. `CLAIM-B` carries a different one: its two anchor
arms refuse below 19 control series, so the gate as a whole refuses on any sweep
that measures fewer.
