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

### H-EMERGE — collapse-resistance emerges at circuit-formation events

**Null (H0-EMERGE):** clustering dynamics and circuit formation are independent;
the energy-monotonicity break and Fiedler drop do not co-locate with the
literature's checkpoint anchors.

Instruments: the Pythia-410M pilot (item 8), `core/checkpoint_frames.py`.
Predictions: `CLAIM-B`.

Note the asymmetry, which `PREDICTIONS.md` states and which should survive into
the e-process: a failure here is *informative on its own terms* — it re-anchors
the 1.4B checkpoint schedule rather than invalidating the sweep. An e-process
records "insufficient evidence", never "null accepted", which is the right
shape for that.

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
sense of apparatus, not of evidence — no run artifacts exist here, and P6-R2/R4
additionally refuse until an exchangeable unit is registered.

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

Six predictions are adjudicable as of 2026-08-24 — `P-S1`, `P-T1`, `P-M1`,
`CLAIM-C`, `P6-R2` and `P6-R4`. What is missing is data, not apparatus: no run
artifacts exist in this repo, so all six are validated on synthetic inputs with
known answers. `P6-R2` and `P6-R4` carry one further refusal on top of that —
their exchangeable unit is not registered, and `adjudicate_p6_r2_r4` raises
rather than letting a caller supply it.
