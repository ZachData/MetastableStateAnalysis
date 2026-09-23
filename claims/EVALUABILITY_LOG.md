# EVALUABILITY_LOG.md — how each null was built, in the order it happened

**Append-only, dated, never edited after the fact.** This is the construction
diary that `claims/EVALUABILITY.md` carried until 2026-09-17, moved here so
that the file a session reads first holds the current state and this one holds
how it was reached. Nothing below was reworded in the move. A citation of
"`EVALUABILITY.md`" dated before 2026-09-17, in `POPPER_PLAN.md` or a module
docstring, refers to a section that now lives here.

Each dated section is one pass — a null built, a dry run on an input whose
answer was known, a floor computed — and most close with a "Which rows are
next" and a "What the pilot must produce, after N" table. **Those tables are
superseded in order**; the current one is the single `## What is next` in
`EVALUABILITY.md`, which is replaced rather than appended. The numbered
lessons ("Seventeenth …") continue `POPPER_PLAN.md` §6's count.

`EVALUABILITY.md` keeps what is durable from these passes: the three states,
the recurring patterns, the order to build a null in, and the generated
tables. A new pass is recorded here and its consequences — a state change, a
new queue entry, a new pilot requirement — are edited into `EVALUABILITY.md`
in the same commit.

---
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
mislabelling as a live alternative explanation and `archive/p6_subspace/design-6.md` pre-registered
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

*Corrected 2026-08-28: four of those six are **dormant** — `P6-R1`, `P6-C1`,
`P5b-A1` and `P5b-A2`, whose instruments are under `archive/`, so converting
them buys no e-value and `core/adjudication.py` refuses them. The list above is
right about the STATISTICS and wrong as a queue, and it has been read as a
queue three times. The live `needs-null` bridge rows are `P-I2`, `P-I3`, `P-I4`
and `P-I5` (all active, relevance 1.0) and `P-SA1` (active, 0.8, instrument
frozen). See "Which rows are actually next" below.*

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
`P-I4` (four of them dormant — see "Which rows are actually next")
as rows whose predictions name a matched control that is a subspace or a
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
control — that is a question to answer before the control is built. *(Four of
those six are dormant; see "Which rows are actually next", 2026-08-28.)*

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
— `P6-R1`, `P6-C1`, `P5b-A1`, `P5b-A2`, `P-SA1`, `P-I4`, four of them dormant
and corrected under "Which rows are actually next" — the order to work in
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

*Superseded 2026-08-27: the list is now **eight**, and the two new rows are
`P-AB1`'s — see the next section. The table above stands as it was measured.*


## `P-AB1` built in the order this document prescribed, and the first row where that order was the method (2026-08-27)

`p7_motifs/patching_gate.py`, with
`tools/calibrate_patching_exponent.py` → `claims/calibration/patching_exponent.json`
behind it, `POPPER_PLAN.md` §6q. `P-AB1` was the last unbuilt Phase 7 bridge
entry with a live instrument, and `design-7.md` calls the patching entry the one
place in its translation table "where the particle account plausibly says
something the mechinterp framing does not already say". **Ten predictions are
adjudicable in principle and `claims/adjudications/` is still empty.**

The section above closed by naming an order for every row that already names a
matched control: *compute the attainable floor, name what the statistic
degenerates on, check what the measurement grid contributes, and only then build
the control.* This is the first row built that way, and **each of the first
three steps changed the design before any control existed.** That is the case
for the order, stated as a result rather than as a process note.

**Twenty-first lesson, and it is the strongest form the floor lesson has
taken.** *A null that leaves the statistic invariant is not a weak null; it is a
floor of 1.000.* The registered construction read "permutation over ablation
points once the fitted exponent is the statistic", and permuting the pairing
between the two arms' points leaves a mean paired difference **exactly**
unchanged — mean β_real − mean β_control for every permutation. Every draw ties
the observation and no input whatever can reject. §6p found floors of 0.100 and
0.083 reported as 0.0005; this one is 1.000, it needed no simulation to find,
and the arithmetic that finds it is one line. The null that replaces it is the
exact sign flip of the two arms' labels, whose floor is `(2^(n−k)+1)/(2^n+1)` —
CLAIM-C's rule from §6l reached by a second construction, because it is the same
group — so **six informative units is the first design that can reject**, and an
**odd** number of ablation points per prompt is free while an even one is not:
at six prompts, seven points leave the design able to reject on 1.000 of H0
draws and six leave it able on 0.394.

**And the floor has two halves here, which is 6p's rule reached from the other
side.** Under the per-ablation-point unit the sign-flip group is 2^42 and is
sampled, so `2/(n_patterns + 1)` binds instead: the design floor is 4.5e-13
where a perfect input returns 4.0e-4. The attainable floor is the max of the
two and the record says which binds. Found by printing what a perfect input
returns beside what the arm claimed it could — the **tenth session running** in
which looking at a generated output found something no test was failing on.

**Twenty-second, and it is what §6o's rule looks like when the answer comes out
"yes".** *An absolute quantity against matched controls is safe only when the
controls are matched on its degenerate input — and here the pairing is what does
the matching.* A fitted growth exponent degenerates on the **fit window**, and
the ablation point fixes it: ablating at layer ℓ of an L-layer model leaves
K = L − ℓ downstream layers. Divergence saturates, so a log-log slope falls with
K — on one fixed set of dynamics, 1.79 at K = 3 against **0.66** at K = 24, a
factor of nearly three on nothing but where the measurement stopped.
"Superlinear" is therefore not a window-free statement, and the registry's own
reason for requiring a control (later layers have more room to diverge) is right
but is not the binding one. Both arms of a pair share the ablation point and so
share the window, which is what makes the contrast legal; the comparison
**across** points, which the registered null performed, is between exponents
fitted over different windows and is not a comparison at all.

**Twenty-third, and it is the finding.** *A statistic can be non-monotone in the
quantity the prediction is about, and then the design returns the falsification
branch on inputs where the prediction holds.* Divergence is **bounded**, so the
arm whose divergence is larger at every layer reaches its ceiling sooner inside
a fixed window and its slope flattens: at one true exponent of 2.0 over eight
layers, τ = 4 fits 1.35 and τ = 16 fits 1.95, and τ = 4 is the arm that
dominates everywhere. On two arms carrying the **same** true exponent where only
the real one saturates sooner — which is what a real ablation that propagates
does — the gate returned `RECAPTURES`, its registered falsification branch, on
**0.98** of draws under one unit and **1.00** under the other. Nothing in the
number reported distinguishes the two readings, because it is the same number.

This is a fifth defect kind beside §6p's four, and it is the only one so far
whose failure lands on the branch that would enter the ledger as a
falsification:

| what was wrong | where | checkable before data? |
|---|---|---|
| a reported floor that was not the design's | `P-ST1`, `CLAIM-C`, `P-T1`, `P-M1`, `P-AB1` | **yes** |
| a null that randomised more than the claim is about | `P-ST1`, `P6-R2` | no |
| a statistic partly determined by the measurement grid | `CLAIM-B`, `P-AB1` | **yes** |
| an input the design cannot compare, scored anyway | `P-S1` | **yes** |
| **a statistic non-monotone in what the prediction is about** | **`P-AB1`** | **yes** — it is the shape of the curve, not a rate |

**Twenty-fourth, and it repeats §6o's correction one pass later.** *A refusal
that thins a defect is not a refusal.* The obvious condition here is that the
two arms bend by different amounts, since an equal bend cancels inside the pair,
and it is testable two-sided with the gate's own exact null. Right shape, too
weak: on the differential-saturation family it turned away 52 of 100 draws under
the prompt unit, and the 48 it let through **still returned RECAPTURES on
0.979** of them. What refuses instead asks the per-**arm** question — is this
curve a power law over the window at all — pooling each curve's own
residual-scaled curvature and testing each arm at α/2. Measured: 0.060 on pure
power laws, 0.360 at τ = 15, 0.970 at τ = 8, 1.000 at τ = 5. Nominal on the
shape it admits, certain on the shape it does not. The discarded sweep is in the
artifact because it is what corrected the design, exactly as §6o's rank sweep
is in its own.

That refusal is the **second** member of §6o's fourth category — a refusal that
is right and costs verdicts anyway. It turns away every draw of both saturating
families, including the symmetric one where the contrast measured nominal, and
every family in the calibration carries the counterfactual re-scored on the
refused draws rather than a claim that the cost was small.

**Twenty-fifth, and it is §6h's question a second time.** The exchangeable unit
is not registered. Measured on the same draws, the per-ablation-point reading —
the one the registry's wording implies — runs 0.050 → 0.141 → **0.235** as a
per-prompt factor takes over, where the per-prompt reading holds at
0.018–0.029. A fourth arrival of the fourfold-plus inflation POPPER reports at
0.082 → 0.340. Which unit may enter an e-process is a decision of CLAIM-C's
class, so it was put to the author, who registered **`"prompt"`** — safe to take
for §6l's reason, that no p-value on real data exists, and unlike §6l's it is a
**trade** rather than free: it costs resolution (six prompts, an odd number of
ablation points each) and power (0.503 against 0.856). Registering it also
removed the safety catch §6l named, and behind that catch was a third defect —
`adjudicate_p_ab1`'s opt-in flag was not being read at all, so every call would
have reached the ledger. Found by a test written because §6l said what
registering a unit removes. And a **fixed** offset between the real and control direction
populations — real ablation directions are not isotropic and the controls are —
is separated by neither unit, reaching 1.000 and 0.895: §6i's
shared-per-unit-factor at 1.00 in this design's clothing, diagnosed and not
removed, with the diagnostic's blindness to it pinned by a test so it cannot
later be mistaken for coverage.

### What the pilot must produce, after ten

Eight requirements now, and `P-AB1`'s second one constrains something none of
the previous six did — the **intervention**, rather than what is measured, where
it is sampled, how it is clustered or how large the run is:

| claim | requirement | constrains |
|---|---|---|
| `CLAIM-B` | 19 control series | what the sweep measures |
| `CLAIM-B` | a grid whose uniform-profile midpoint falls outside 512–2000 | which checkpoints it samples |
| `CLAIM-C` | ≥ 23% of candidate cells dissenting in sign, and ≥ 5 prompts whose usable metrics do not split evenly | what the contrast looks like |
| `P-ST1` | dim `U_pos` comparable to the dimension the population occupies | the projector's shape |
| `P-T1` / `P-M1` | enough heads, and enough layers or violations, for the design floor to clear α | how large the run is |
| `P-S1` | both arms clustered to the same count | how the run is clustered |
| `P-AB1` | six informative units, an ODD number of ablation points per prompt, and `n + W ≤ L` | the ablation grid |
| `P-AB1` | an ablation magnitude and fit window keeping BOTH arms inside the power-law regime | **the intervention itself** |

Still none of which any existing sweep satisfies.


## `CLAIM-B`'s grid, computed rather than chosen (2026-08-28)

`core/changepoint_colocation.grid_feasibility` with
`tools/claim_b_grid_feasibility.py` →
`claims/calibration/claim_b_grid_feasibility.json`, `POPPER_PLAN.md` §6r. §6o
left `CLAIM-B` needing a sweep neither grid this project has can supply and said
choosing one is a pre-registered decision for the author. **Which grids clear
the conditions is arithmetic, so it is computed and the decision becomes a
choice from a set.** Nothing here chooses a grid, and
`claims/adjudications/` is still empty.

**Twenty-sixth lesson, and it is the one this exercise keeps producing in new
clothes.** *The grid that maximises an arm's numbers can be the grid that
destroys what the claim is about.* Maximising the retained share of the anchor
window alone picks a sweep whose one wide interval swallows the window: every
anchored change reads inside it — retention 1.000 — and so does every change for
a third of a window-width above, and it cannot say *where* in the window
anything happened. Two conditions were added for it, and the second of them, the
read span inside the window, is the **claim's** requirement rather than the
arm's: the anchor arm alone is happiest at a span of zero. §6o's "a statistic
partly determined by the measurement grid" read from the other side — not what
the grid does to the statistic, but what optimising the statistic does to the
grid.

**Twenty-seventh, and it is the defect.** *A change's location depends on the
change's own WIDTH as much as on the grid, and the reading that ignores it is
not conservative.* The first version located a change at the midpoint of the
interval containing it — this construction's own stated resolution limit, and it
looks like a bound. It is not: a change of real width spreads mass into
neighbouring intervals, so a coarse interval just past the window collects it
and the location leaves the window at **zero** noise. A ten-checkpoint grid that
reading scored at retention 1.000 put a planted anchor inside the window on
**0.017** of draws. `grid_feasibility` now takes the width and σ/R together or
not at all — both are properties of the series, both are measurable from a
pilot's own data before any p-value is computed, and neither is defaulted. The
eleventh session running in which looking at a generated output found something
no test was failing on.

**Twenty-eighth, and it is what the enumeration is for.** *When a
pre-registered rule admits nothing, the reading is about the world and not about
the rule.* Requiring 0.95 of the anchor window to survive across the plausible
noise range admitted **zero** grids of 96,127. The best any published Pythia
schedule reaches is **0.680**, and the loss is at the window's upper end and at
zero noise: a change centred near step 2000 puts half its mass above 2000, the
next affordable published checkpoint is tens of thousands of steps away, and
adding checkpoints between 2000 and 20000 pulls the sweep's own midpoint back
*into* the window, which is §6o's refusal. That trade is between this claim's
registered anchor window and EleutherAI's release schedule. The rule now
maximises retention and records the achievable maximum; tuning the threshold
until something passed would have hidden the only thing worth knowing.

**And the registered cheap sweep is not the only schedule here that fails.**
`core/pythia_registry.PYTHIA_410M_PILOT_STEPS` — the schedule this repository
would actually run — puts its change-free reference at step **1191**, inside the
window, exactly as the registry's cheap-tier description does at step 955. §6o
read the failure off the registry's instrument field; the code's own schedule
has it too. Measured, both discriminate at **0.000** between an anchored change
and a series with no located change at all, where all three grids the arithmetic
picks discriminate at **1.000**.

**And the decision was taken.** Put to the author with three computed grids and
their costs, and **`(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 54000)` is
registered** — `REGISTERED_CLAIM_B_SWEEP`, with `adjudicate_claim_b` refusing a
result computed on any other grid while `p_value_claim_b` still computes on any
and reports which it was. The third registered decision of this class, after
`P6-R2`'s `"model"` and `P-AB1`'s `"prompt"`, safe to take for the same reason
as both: no p-value on real data exists. Unlike those two it **adds** a refusal
rather than lifting one — `adjudicate_claim_b` previously took a result from any
grid at all — so what carries over from them is the isolation rather than the
exposure, and there is now a test asserting the real `claims/adjudications/`
directory does not exist after the adjudicating tests run.

### Which rows are actually next

The list this document has carried since 2026-08-24 — `P6-R1`, `P6-C1`,
`P5b-A1`, `P5b-A2`, `P-SA1`, `P-I4` — is right about the **statistics** and
wrong as a **queue**, and it has been read as a queue three times. Four of the
six are **dormant**: `P6-R1` and `P6-C1`'s instrument is `archive/p6_subspace`,
`P5b-A1` and `P5b-A2`'s is `archive/p5b_manifold_steering`, and
`core/adjudication.py` refuses a dormant row, so converting one buys no e-value
at all.

The live `needs-null` rows a conversion could reach:

| row | claim | status | relevance | what its null needs |
|---|---|---|---|---|
| `P-I2` | H-BRIDGE | active | 1.0 | two-sample channel mass between edge types, against the N1/N2 nulls `motif_stats.py` already gates on |
| `P-I3` | H-BRIDGE | active | 1.0 | correlation with a **required** control arm over non-induction heads |
| `P-I4` | H-BRIDGE | active | 1.0 | matched-magnitude control on `moved_fraction`; permutation over which edges are labelled motif edges |
| `P-I5` | H-BRIDGE | active | 1.0 | permutation over a matched-magnitude random-direction ablation, on a **joint two-dimensional** statistic |
| `P-SA1` | H-BRIDGE | active | 0.8 | random-subspace null of matched dimension; instrument frozen |

`P-I3` and `P-I4` name a matched control and are the natural next, in the order
`P-AB1` established: compute the attainable floor, name what the statistic
degenerates on, check what the measurement grid contributes, and only then build
the control. `P-I5` is the first row here whose statistic is two-dimensional,
which no construction in this project has built.

### What the pilot must produce, after eleven

Nine requirements. `CLAIM-B`'s grid row is no longer a constraint to satisfy but
a set to choose from, and the new row constrains something none of the previous
eight did — how quiet the measurement has to be, relative to the series' own
change.

| claim | requirement | constrains |
|---|---|---|
| `CLAIM-B` | 19 control series | what the sweep measures |
| `CLAIM-B` | the registered sweep `(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 54000)`, chosen 2026-08-28 from the computed set — no schedule previously in this repository is in it, and none holds more than 0.680 of the anchor window | which checkpoints it samples |
| `CLAIM-B` | a σ/R and a change width that the chosen grid's retention curve still supports | **how quiet the measurement is** |
| `CLAIM-C` | ≥ 23% of candidate cells dissenting in sign, and ≥ 5 prompts whose usable metrics do not split evenly | what the contrast looks like |
| `P-ST1` | dim `U_pos` comparable to the dimension the population occupies | the projector's shape |
| `P-T1` / `P-M1` | enough heads, and enough layers or violations, for the design floor to clear α | how large the run is |
| `P-S1` | both arms clustered to the same count | how the run is clustered |
| `P-AB1` | six informative units, an ODD number of ablation points per prompt, and `n + W ≤ L` | the ablation grid |
| `P-AB1` | an ablation magnitude and fit window keeping BOTH arms inside the power-law regime | the intervention itself |

Still none of which any existing sweep satisfies — and for `CLAIM-B` that is now
a statement with a set behind it rather than a gap.

## `P-I3` converted, and a tautology that turned out to be a floor (2026-08-30)

`p7_motifs/cross_head_gate.py`, with
`tools/calibrate_cross_head_association.py` →
`claims/calibration/cross_head_association.json` behind it,
`POPPER_PLAN.md` §6s. The section above corrected this document's queue and
named `P-I3` and `P-I4` as the natural next rows; this is `P-I3`. **Eleven
predictions are adjudicable in principle and `claims/adjudications/` is still
empty.**

Third row built in the order this document prescribes — floor, then what the
statistic degenerates on, then the measurement grid, then the control — after
`P-AB1` (§6q) and `CLAIM-B`'s grid (§6r). On this one the first step changed
what the null is, the second changed what the statistic is, and the third
changed what the control is matched on.

**Twenty-ninth lesson.** *When the label a null permutes is a deterministic
function of the variable the statistic uses, the null draws one possible
configuration and every other draw is impossible.* An induction head is one
whose behavioural induction score clears a cutoff, so "permutation over the
head classification" — the registered null — draws which *k* of *n* heads are
labelled, and of its **1.09e16** members at 384 heads and eight induction heads
exactly **one** is a classification the definition permits. This is not §6q's
invariance, where the statistic does not move; the statistic moves freely, and
what it moves through is a family the data could not have produced. Measured on
both readings of the statistic on the same draws, the correlation contrast
discriminates a genuine effect from its absence at **-0.003** and the slope
contrast at **-0.010**, the latter from an H0 rate of 0.110 — **anti-conservative
and powerless at once**, which is one mechanism read in two directions: a
permuted group carries the full score spread and the observed group carries a
thin slice at the top.

**Thirtieth, and it is the finding.** *A phase's stated central danger can turn
out to be a floor rather than a caution.* `design-7.md` gives the tautology risk
its own section — the behavioural induction score is *mean attention on
induction pairs* and a motif defined as *an attentive edge on induction pairs*
is the same number — and `PREDICTIONS.md`'s Phase 7 adjudication constraint 2
makes naming the independence source mandatory. §6i built `P-I1`'s gate and
recorded that no null over its pairing could detect the substantive case, so it
stayed "a claim the analyst must make in the record". Here the matching removes
the shared component by construction, and when the classification **is** the
thresholded score, no induction head has a control above it, no matched set can
be straddled, and the design floor is **1.000** — decidable before a single edge
is counted. And the danger is larger than the constraint's wording: the leak is
proportional to how hard the motif *tracks* the score, not to a literal
identity — 0.027, 0.153, 0.400, 0.827, 1.000 across that sweep under the
matching this construction discarded, against zero matched sets throughout under
the one it uses.

**Thirty-first, and it is what §6o's rule looks like when the answer comes out
"cannot".** *An absolute quantity against matched controls is safe only when the
controls are matched on the quantity the statistic degenerates on — and
sometimes the classification makes that matching impossible, which is a
statement about the design and not about the estimator.* A within-group
correlation degenerates on the within-group spread of what it correlates. On ONE
population carrying ONE relation the two arms read **0.255** and **0.673** — a
contrast of **-0.417** with no interaction present anywhere, in the FALSIFIER's
direction. The within-group slope removes that bias and not its cause: its
sampling spread is **23.9×** the control arm's at eight induction heads. So
§6o's table gains a third column:

| entry | statistic | the quantity it degenerates on | matched on it? |
|---|---|---|---|
| `P6-R4` | one subspace against matched controls | subspace dimension | **yes**, by construction |
| `P-AB1` | one exponent against a control at the same point | the fit window | **yes**, by pairing |
| `CLAIM-B` anchor | one location against a fixed window | where the grid puts an unlocated profile | **no** (§6o) |
| `P-I3` correlation | one within-group correlation against another | the within-group score spread | **cannot be**, the classification sets it |

What is adjudicated instead is the quantity this entry's own `h0` and
`falsifier` both name — *"unrelated to the classification"*, *"carry the motif
at the same rate"* — a LEVEL contrast at matched score, combined as the
induction head's rank within its matched set. The `statement`'s "correlates" is
a third quantity and one number cannot carry both questions, which is `P-I1`'s
split reached from the other side: there it was the falsifier that had to become
a precondition, here it is the statement.

**Thirty-second, and it is four defects with one shape.** *A record's own
self-check is the thing that catches the pass's reasoning, and the reasoning is
what is usually wrong.* The twelfth session running, and all four were found by
reading the tool's output on the file it had just written.
(i) The module docstring claimed the observed induction group's score spread was
"smaller than all 20,000 label-permuted draws" — true of the one exploratory
draw it was written from, and not a bound: averaged, the permuted family's mean
spread is 2.92× the observed one and 0.00255 of draws are as tight. §6r's rule
against quoting a rate that moves on regeneration, reached one level up, as a
per-DRAW claim presented as a bound. (ii) The self-check asserted the two
within-group slopes agree to a fixed 0.20 — of a statistic whose enormous
sampling spread that same section exists to report — so it fired on a healthy
artifact; it now tests against three standard errors of the arm's own spread.
§6n's `precision_check` failure mode, third arrival. (iii) The counterfactual
"what the arm would return with no refusal" returned 0.000 on the tautology
family, because on that family the straddle yields no sets and there was no arm
to score. **A counterfactual has to be computable on the family the refusal
exists for, or it reports the refusal's success as the family's harmlessness.**
It is now the discarded matching, which returns 0.410 where the gate refuses at
1.000. (iv) `registered_null._the_finding` — the record's own summary
sentence — carried an exploratory run's digits (+0.010 from 0.065, −0.005 from
0.285) that the committed rows contradict (−0.003 from 0.020, −0.010 from
0.110), and nothing was failing because **nothing in this project compares a
record's prose to its own fields**. §6r found two rounds of exactly this by
re-reading a committed output, which makes this the third pass and a pattern
rather than an accident, so the fix is structural: the sentence is now derived
from the rows at write time and `check_record` fails if a measured rate it
should carry is missing. What all three passes' occurrences share is that the
code was already right and the prose beside it was written from an earlier run.

**And the decision was taken.** Induction heads cluster in a band of layers, and
a shared elevation across that band is invisible to a control matched on score
alone — 0.043, 0.107, 0.233, **0.440** as the elevation grows — while a control
drawn from the induction head's own layer is flat at 0.011 throughout. Unlike
§6i's shared unit factor and §6q's fixed offset, this one **can** be removed,
and the price is measured: 1.88 informative sets against 7.21, a p-value emitted
on 0.657 of draws against 1.000, power 0.203 against 0.410 and 0.528 against
0.883. Put to the author with both sides, and **`"score_and_layer"` is
registered** — the fourth decision of this class after `P6-R2`'s `"model"`,
`P-AB1`'s `"prompt"` and CLAIM-B's sweep, safe to take for the same reason as
all three (no p-value on real data exists), a trade rather than free as in the
last two, and like §6r's it **adds** a refusal rather than lifting one.

### Which rows are next

Unchanged from §6r's corrected list minus the row now built. The live
`needs-null` rows a conversion could reach:

| row | claim | status | relevance | what its null needs |
|---|---|---|---|---|
| `P-I2` | H-BRIDGE | active | 1.0 | two-sample channel mass between edge types, against the N1/N2 nulls `motif_stats.py` already gates on |
| `P-I4` | H-BRIDGE | active | 1.0 | matched-magnitude control on `moved_fraction`; permutation over which edges are labelled motif edges |
| `P-I5` | H-BRIDGE | active | 1.0 | permutation over a matched-magnitude random-direction ablation, on a **joint two-dimensional** statistic |
| `P-SA1` | H-BRIDGE | active | 0.8 | random-subspace null of matched dimension; instrument frozen |

`P-I4` is the remaining row that names a matched control and is the natural
next. `P-I5` is still the first row here whose statistic is two-dimensional,
which no construction in this project has built. The six-row list this document
carried until §6r — `P6-R1`, `P6-C1`, `P5b-A1`, `P5b-A2`, `P-SA1`, `P-I4` — is
still right about the statistics and still not a queue: four of those six are
dormant and `core/adjudication.py` refuses a dormant row.

### What the pilot must produce, after twelve

Ten requirements. The new one constrains something none of the previous nine
did — **how the two arms are defined**, rather than what is measured, where it
is sampled, how it is clustered, how large the run is, the ablation grid, the
intervention, or how quiet the measurement is.

| claim | requirement | constrains |
|---|---|---|
| `CLAIM-B` | 19 control series | what the sweep measures |
| `CLAIM-B` | the registered sweep `(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 54000)`, chosen 2026-08-28 from the computed set — no schedule previously in this repository is in it, and none holds more than 0.680 of the anchor window | which checkpoints it samples |
| `CLAIM-B` | a σ/R and a change width that the chosen grid's retention curve still supports | how quiet the measurement is |
| `CLAIM-C` | ≥ 23% of candidate cells dissenting in sign, and ≥ 5 prompts whose usable metrics do not split evenly | what the contrast looks like |
| `P-ST1` | dim `U_pos` comparable to the dimension the population occupies | the projector's shape |
| `P-T1` / `P-M1` | enough heads, and enough layers or violations, for the design floor to clear α | how large the run is |
| `P-S1` | both arms clustered to the same count | how the run is clustered |
| `P-AB1` | six informative units, an ODD number of ablation points per prompt, and `n + W ≤ L` | the ablation grid |
| `P-AB1` | an ablation magnitude and fit window keeping BOTH arms inside the power-law regime | the intervention itself |
| `P-I3` | a head classification that is NOT a cutoff on the behavioural induction score — specifically, ≥ 2 induction heads with two control heads below and two above them in that score, **within their own layer** | **how the two arms are defined** |

Still none of which any existing sweep satisfies — and `P-I3`'s is the first
that a sweep cannot satisfy at all, because it is a requirement on the
classification criterion rather than on the run.

## `P-I1`'s floor, computed before its null: a design that cannot express a p, and a floor with a half missing (2026-09-03)

`tools/p_i1_attainable_floor.py` ->
`claims/audits/p_i1_attainable_floor.json`, with
`core.changepoint_colocation.pairing_floor_report` behind it,
`POPPER_PLAN.md` §6t. **Eleven predictions are adjudicable in principle and
`claims/adjudications/` is still empty.**

Fourth row worked in the order this document prescribes — floor, then what the
statistic degenerates on, then the measurement grid, then the control — after
`P-AB1` (§6q), `CLAIM-B`'s grid (§6r) and `P-I3` (§6s). It is the first applied
to a control that **does not exist yet**: `P-I1`'s relay-count null has never
been built, `PROJECT.md` §3.4 records its shape as the author's decision, and
steps 2 and 3 were done on 2026-09-01 with step 1 left. `P-I1` is **not
converted** by this pass and stays `needs-null`.

**Thirty-third lesson, and it is the floor lesson arriving at a design that
cannot express a p at all.** *Before asking how small a p a design can return,
check that it can return one — and the answer can be no for a reason that is a
property of the axis rather than of the data.* `formation_curve_payload` takes
its head axis from the behavioural series, dense over all 384 heads, and
zero-fills the relay side; `paired_colocation_arm` profiles every unit with **no
per-unit skip**; `change_profile` refuses a series with no rise. On the real
sweep 116 heads carry relays and **268 never do**, so the gate returns no
p-value at all, and its message — *"the series has no rise anywhere in the
sweep"* — names no arm, no head and no unit count. The same input on the 116
forming heads emits. What that costs is not a p-value at the wrong number; it is
the absence of one, attributed to a series that is fine.

**Thirty-fourth, and it is the strongest evidence yet FOR the order, because it
is a lesson this repository already had.** *A null that leaves the statistic
partly invariant has a floor the draw count cannot reach, and the two halves can
sit two units apart.* The pairing statistic is `-mean|ca - cb[p]|`; permuting
units within a class of equal locations leaves it exactly unchanged, so every
pairing ties a coset of order `prod(m!)` and no input can express a p below
`prod(m!)/n!`. Measured on the registered 19-step grid with nine of ten units
sharing one location: reported floor **0.000500**, attainable **0.100000** — a
factor of **200**, above α, emitted with no refusal. Seven of ten tied is
0.00139 and emits legitimately. `p7_motifs/steering_gate.py` has carried
`draw_count_floor` beside `best_attainable_p` since §6m, with a test named
`test_the_attainable_floor_is_set_by_ties_and_not_by_the_draw_count` pinning it,
and **the shared estimator two gates over did not have it**. Nothing about
finding it needed data; it needed someone to run the floor step.

**Thirty-fifth, and it is §6o's rule at a fourth entry with the answer "not
yet".** *When the control does not exist, the floor is a constraint ON it, and
that is the entire reason for computing it first.* A relay-count null turns the
series into an above-null excess, and a head whose excess stops rising leaves
the scored set — so the null chooses `n_units`, and `n_units` with the tie
structure chooses the floor. The draw-count half needs four heads; the tie half
needs `k!/n! ≤ α` for the largest surviving class k, and is **not monotone in
n** (`k = n − 1` gives exactly `1/n`, so all-but-one-tied clears 0.05 at twenty
survivors and fails at nineteen). Both are computable with no null built and no
draw taken.

| entry | statistic | the quantity it degenerates on | matched on it? |
|---|---|---|---|
| `P6-R4` | one subspace against matched controls | subspace dimension | **yes**, by construction |
| `P-AB1` | one exponent against a control at the same point | the fit window | **yes**, by pairing |
| `CLAIM-B` anchor | one location against a fixed window | where the grid puts an unlocated profile | **no** (§6o) |
| `P-I3` correlation | one within-group correlation against another | the within-group score spread | **cannot be**, the classification sets it |
| `P-I1` mutual | two locations paired unit by unit | how many DISTINCT locations the grid resolves | **not yet** — the null that sets it is unbuilt |

**And one thing the pass declined to do, recorded because declining is a
decision.** Reducing the axis to the forming heads, or giving the arm a
per-unit skip with a count reported, would both make the gate return a number.
Both change what `P-I1`'s **unit** is, and `PREDICTIONS.md`'s first Phase 7
adjudication constraint fixes the unit at the head. So the diagnosability gap is
pinned as it is, in a test that says so, and the choice goes to the author with
the measurement beside it — §6r's and §6s's shape, a third time.

**Addendum, 2026-09-04 — the null was built, run, and `P-I1` scored
INSUFFICIENT.** `p7_motifs/relay_count_null.py`, `POPPER_PLAN.md` §6v: a
head-level payload shuffle, degree-preserving at the head (edge count and the
full force distribution held fixed, not per particle), holding `n_induction`
fixed per prompt automatically because the pool and candidate sets are
properties of the prompt's tokenisation alone. The requirement this document's
table above named — "a relay-count null leaving ≥ 4 heads with a rising
above-null excess, and among them no more than k sharing one change
location" — is satisfied by a wide margin: 116 heads scored, 0 skipped,
attainable floor 0.0005. The row `P-I1` mutual's "not yet — the null that sets
it is unbuilt" is stale as of this line; `tools/score_p_i1.py` returns **p =
0.1414**, verdict **INSUFFICIENT** — not falsified, not validated, and
`claims/adjudications/` is untouched. `PROJECT.md` §3.6 carries the full
number set.

### Which rows are next

Unchanged from §6s. `P-I1` is not on it: it was already `needs-null` and this
pass did not convert it — the pass that DID, 2026-09-04, produced a p-value
rather than an adjudication, and the addendum above is where it is recorded.

| row | claim | status | relevance | what its null needs |
|---|---|---|---|---|
| `P-I2` | H-BRIDGE | active | 1.0 | two-sample channel mass between edge types, against the N1/N2 nulls `motif_stats.py` already gates on |
| `P-I4` | H-BRIDGE | active | 1.0 | matched-magnitude control on `moved_fraction`; permutation over which edges are labelled motif edges |
| `P-I5` | H-BRIDGE | active | 1.0 | permutation over a matched-magnitude random-direction ablation, on a **joint two-dimensional** statistic |
| `P-SA1` | H-BRIDGE | active | 0.8 | random-subspace null of matched dimension; instrument frozen |

### What the pilot must produce, after thirteen

Eleven requirements. The new one constrains something none of the previous ten
did — **how aggressive the control may be**, rather than what is measured, where
it is sampled, how it is clustered, how large the run is, the ablation grid, the
intervention, how quiet the measurement is, or how the arms are defined.

| claim | requirement | constrains |
|---|---|---|
| `CLAIM-B` | 19 control series | what the sweep measures |
| `CLAIM-B` | the registered sweep `(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 54000)`, chosen 2026-08-28 from the computed set | which checkpoints it samples |
| `CLAIM-B` | a σ/R and a change width that the chosen grid's retention curve still supports | how quiet the measurement is |
| `CLAIM-C` | ≥ 23% of candidate cells dissenting in sign, and ≥ 5 prompts whose usable metrics do not split evenly | what the contrast looks like |
| `P-ST1` | dim `U_pos` comparable to the dimension the population occupies | the projector's shape |
| `P-T1` / `P-M1` | enough heads, and enough layers or violations, for the design floor to clear α | how large the run is |
| `P-S1` | both arms clustered to the same count | how the run is clustered |
| `P-AB1` | six informative units, an ODD number of ablation points per prompt, and `n + W ≤ L` | the ablation grid |
| `P-AB1` | an ablation magnitude and fit window keeping BOTH arms inside the power-law regime | the intervention itself |
| `P-I3` | a head classification that is NOT a cutoff on the behavioural induction score — ≥ 2 induction heads with two control heads below and two above them, within their own layer | how the two arms are defined |
| `P-I1` | a relay-count null leaving ≥ 4 heads with a rising above-null excess, and among them no more than k sharing one change location (k tabulated in `POPPER_PLAN.md` §6t) | **how aggressive the control may be** |

Still none of which any existing sweep satisfies. `P-I3`'s is a requirement on
the classification criterion and `P-I1`'s is a requirement on the **null**,
which is the first of either kind: it is not satisfiable or unsatisfiable by a
run at all, and it is checkable the moment the null is written and before it is
used. `P-I1`'s row IS now satisfied — 2026-09-04, 116 heads scored, 0 skipped,
see the addendum above — which is a fact about the null's design, not about
the sweep it was checkable against; the "still none" reading was always about
the OTHER ten rows and remains true for them.
