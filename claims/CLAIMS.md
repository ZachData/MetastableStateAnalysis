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

Instruments: `p2_eigenspectra`, `p2b_imaginary`, `p2d_operator_activation`, `p6_subspace`.
Predictions: `P-T1`, `P-M1`, `P6-A2`, `P6-I1`, `P6-I2`, `P6-R1`, `P6-R2`,
`P6-R3`, `P6-R4`, `P6-R5`, `P6-C1`, `P6-DD1`, `P6-DD2`, `P6-D5`.

**This claim already carries a falsification, and it is the strongest single
result in the registry.** P6-R2 predicted LDA alignment with the real repulsive
subspace $U_\text{neg}$; the measurement gives 0.887 alignment with the
*imaginary* subspace $U_A$ against 0.067 with $U_\text{neg}$, and **0 of 49
layers** show the predicted direction. P6-R4 inverts the same way: real-only
probe accuracy 0.152 (chance), imaginary-only 0.564 against 0.590 for the full
activation. See `p6_subspace/status-6.md`, and its caveat that 49 layers of
ALBERT are not 49 independent observations — which is exactly the kind of thing
that has to be settled before a p-value from this can enter an e-process.

### H-BRIDGE — natural-language-interpretability constructs are particle-dynamical objects

The workstream-C claim. Not "the two vocabularies can be translated" — that is
unfalsifiable — but: the particle reading makes *differential* predictions,
correct where the standard account is silent or says something else.

**Null (H0-BRIDGE):** the particle-paradigm definition of each construct adds no
predictive content over its standard definition; where they differ observably,
the standard account is right.

Instruments: `core/dual_reading.py`, `p6_subspace/induction_ov.py`,
`p5b_manifold_steering`, `core/intervention.py`.
Predictions: `P5b-A1`, `P5b-A2`, `P5b-B1`, `P5b-B2`, `P5b-B3`, `P5b-C1`,
`P5b-C3`, `P5b-D1`, `P5b-D2`, plus the differential predictions from
POPPER_PLAN.md item C2 (not yet registered).

**Why P5b lives here rather than under H-OPERATOR.** Its sub-experiment D asks
whether behavioral geometry is carried by the real/symmetric subspace and *not*
by the imaginary/antisymmetric one (P5b-D1, P5b-D2). That is a claim about what
the standard "steering vector" is, in particle terms — a coordinate system
statement, not an operator-classification one.

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
