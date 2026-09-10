<!-- p7d_redundancy/design-7d.md -->
# Phase 7d — DESIGN

## Core question

Two heads in pythia-410m dominate the induction behaviour: `L5H2` and `L7H8`.
The natural reading of that pair is a **circuit** — a prev-token head writing a
tag, a matcher reading it — and the natural next step is to trace it stage by
stage.

**Two measurements killed that reading within hours of it being written down**,
and this phase exists because of what replaced it.

- **§3.12-O/P killed the three-stage reading.** `L7H8` is not a token-identity
  copier at any point in training; the real copiers sit in layers 9–20; and the
  mediation test found **no sub-additivity anywhere**, so `L7H8` does not route
  through them.
- **§3.12-S killed the serial reading of the pair itself.** Ablating both costs
  **+4.151 more** than the sum of ablating each — 2.2× the parts-sum. A serial
  two-stage circuit predicts the opposite sign: remove the prev-token head and
  the matcher has less to match on, so removing it too should cost *less*.

So the object is not a chain of stages. It is a **set of functionally redundant,
structurally distinct heads** that jointly hold a residual-stream regime, and
the unit of this phase is that set.

> Which heads hold the regime, when did each of them form, and is "the set" one
> class or several?

## The five questions

1. **How many members?** `L5H2` and `L7H8` may be a pair or the two visible
   members of a set.
2. **Did they form at the same time?**
3. **Did they form as early as the network could?** The literature puts
   induction-head emergence near step 1000 in Pythia, so this is checkable
   rather than rhetorical.
4. **Did they form the same way** — same structure, same job?
5. **One class or several?** Do members cluster by structure, by formation time,
   or not at all?

## Why the instrument is causal and never structural

Every question above could be asked of the weights, and the weights cannot
answer them. Three independent results say so:

- **§3.12-R** — `||OV||_F` explains **0.1 %** of the variance in causal effect,
  and the relationship runs *backwards*.
- **§3.12-G6** — no spectral field predicts causal effect either.
- **§3.12-S** — the pair's write subspaces overlap **at or below chance** (0.222
  against 0.250) while their residual effects are **87 % aligned**. Weight-space
  overlap and function-space overlap come apart.

A proxy screen would inherit exactly that failure, which is why Q1 was answered
by a full 384-head causal sweep rather than by ranking a structural quantity.

**§3.12-U added a fourth**: the *behavioural* proxy fails too, and for `L5H2` it
is inverted — its induction score falls twenty-fold across the very interval in
which its causal effect goes from +0.01 to +4.97.

## What the design must not lose

- **Membership is causal, by definition.** Every composition score in §3.12 is a
  weight-space measure and is blind to the redundancy the ablations show.
- **§3.13's report-both rule.** Report the mean view *and* the extremum view,
  and never choose between them after seeing the data. §3.12-J found the right
  instrument changes with training stage, so this applies per checkpoint too.
- **The spent-artifact rule** (`check_registry` rule 3). Everything here is
  pythia-410m and exploratory; it **cannot later be registered and adjudicated
  on the same data**. Any 7d claim intended for the registry needs its own
  unseen test site — which is why `P-I7` was registered against a model this
  project has not measured.
- **The particle-dynamics half is blocked** on `dual_reading`'s pairwise field
  (`P-I5`, §3.12-C). Until it exists, 7d can characterise structure and timing
  but not inter-particle geometry.

## The readout's ceiling, and when it stops being a ruler

`dNLL` on the second copy has a hard upper bound: uniform prediction over the
vocabulary, `ln 50304 = 10.83`. §3.12-U found the joint-ablation arm sitting
within **0.83 nats** of it at steps 1000–2000, and the `L5H2`×`L11H14` pilot
landing *past* it at 11.77.

In that regime the number is a floor, not a measurement, and — worse — the
compression biases interactions toward apparent **sub**-additivity (§3.12-M5),
which is the serial-circuit signature. **Any interaction measured where the
joint arm approaches the ceiling is uninterpretable at raw `dNLL`.** The graded
readout (§3.12-M's KL / λ scale) is the prerequisite, not an improvement.

Two instruments in this phase are immune and should carry the weight when the
readout cannot: a single-head arm far from the ceiling, and the residual-delta
cosine, which is geometric and independent of the readout entirely.

## Where the cross-case question lives

Once several circuits are catalogued — other 410m members, and pythia-70m under
§3.9's grid — ask whether their **dynamics correlate across cases**. A shared
signature over independent circuits is a population claim no single circuit can
make, and it is the natural home for anything §3.12 produced that wants to
generalise.
