# Phase 7 — DESIGN

## Core question

Mechanistic interpretability names things: induction heads, steering vectors, activation
patching, prompt injection, SAE features. Every one of those names is grounded, ultimately,
in a natural-language description of what a component "does" — *this head copies*, *this
feature means "Golden Gate Bridge"*.

This project has a different account of the same systems. Tokens are particles on the
sphere, attention is the interaction kernel, a residual block is a forward-Euler step of the
Geshkovski dynamics, and depth is integration time. Nothing in that account has a semantic
primitive. It says where particles are and what moves them.

Phase 7 asks whether the first vocabulary can be written in the second **without passing
through the first's semantics**:

> Is each named mechinterp phenomenon a **recurring structure of particle interactions** —
> a motif — that can be identified, counted, tested against a matched null, and tracked
> across training?

This is a translation plus an empirical bet on the translation. The translation is free: we
can always *define* an induction head as some pattern of forces. The bet is not. The pattern
has to be there, has to exceed a matched null, and has to be present in the components the
mechinterp name includes and absent from the ones it excludes. Any of those can fail, and
the failure is the result.

## Why this is not the same as "explain SAEs in particle language"

The trap this phase is designed against is producing a glossary — a document asserting that
steering "is" a translation of the particle cloud and patching "is" a teleportation, with no
measurement attached. That would be a restatement, not a finding, and it would be
unfalsifiable in exactly the way this project's methodology exists to prevent.

So every entry in the translation table below has to bottom out in a quantity computed from
an interaction table, compared against a null this project already knows how to build. If an
entry cannot be written that way, it does not go in the table.

## Why induction-head formation is the first study

Four reasons, in order of weight:

1. **It has a sharp behavioral metric that already exists.** "Mean post-softmax attention on
   induction pairs" is a number, not a description. The bridge has a well-defined thing to be
   tested against on the mechinterp side, which is not true of most named phenomena.
2. **It has a time axis.** Induction heads form abruptly, at a known-ish point in training.
   This project's whole Pythia transition is built on a checkpoint axis. A motif and a
   behavior can be compared not just on whether they co-occur but on *which comes first* —
   and all three answers (before, with, after) are informative.
3. **It bears directly on a pre-registered claim.** `PREDICTIONS.md` claim (b) says
   collapse-resistance emerges at circuit-formation events. Phase 7's formation curve is a
   direct measurement of one such event's location on the same axis.
4. **`core/` is already most of the way there.** `battery_structure.py` defines induction
   pairs (and knows when a prompt cannot carry the test); `qk_offset_null.py` implements the
   three nulls that matter; Phase 2/2b supply the projectors. Very little new machinery is
   required, which means the phase's risk is concentrated in its *reasoning*, where it should
   be, rather than in a pile of new code.

## The interaction object, and why the edge is typed by force rather than attention

A naive interaction graph uses post-softmax attention as the edge weight. That is the
interaction *kernel*, but it is not the interaction: attention says how much particle `i`
looks at particle `j`, not what looking does to `i`. Two heads with identical attention
patterns and opposite-signed OV circuits produce opposite motion.

So an edge carries the **force** — the displacement contribution `A_ij · V x_j` — and is
typed along four independent axes:

- **sign channel** (attractive / repulsive), through Phase 2's `U_pos` / `U_neg`;
- **rotational channel** (real / imaginary), through Phase 2b's Schur split;
- **offset class** `Δ = i − j`;
- **pair type** (induction / strict / same-content / neither), from `battery_structure`.

The sign channel is what makes the graph a *dynamical* object rather than a routing diagram.
It is also what keeps this phase from being a re-description of attention-pattern analysis,
which is the thing mechinterp already does well and which the particle account has no
advantage over.

## Why the motif alphabet is fixed in advance

The obvious implementation is to enumerate every typed subgraph up to size 3 or 4 and report
which are over-represented. This is a motif zoo. With millions of edges per checkpoint, the
number of nominally-significant patterns is limited only by patience, and the resulting table
is not falsifiable by anything.

Version 1 therefore pre-commits to seven named motifs, each with a written particle-dynamics
definition, chosen because each is either (a) the restatement of a specific mechinterp object
or (b) a structure the existing theory says should exist:

| Motif | Definition | Why it is in the alphabet |
|---|---|---|
| `prev_token` | attractive edge at offset −1 | stage 1 of the induction circuit |
| `match` | attractive edge to a position whose predecessor matches the target's content | stage 2 of the induction circuit |
| `sink` | edge into position 0 / a known attention sink | Phase 6 found the same-content null can collapse onto the sink column; sink behavior must be separable or it will masquerade as everything else |
| `relay` | a `prev_token` edge at ℓ₁ whose **target particle** is the **source** of a `match` edge at ℓ₂ > ℓ₁ | **the induction head, restated as a particle motif** |
| `mutual` | reciprocal attractive edges | a bound pair — the smallest metastable structure the dynamics admit |
| `hub` | in-degree far above the layer's distribution | a local attractor; the object Phase 1's clustering finds indirectly |
| `repulsor` | dominant edges in the repulsive channel | individuating pressure — the mechanism Phase 5c's unclustered population would need |

Open-ended motif discovery is deliberately **deferred to a later, explicitly exploratory
pass**, run only after this alphabet has been calibrated against nulls, and reported as
exploratory rather than confirmatory. That ordering is the whole difference between a
hypothesis and a fishing expedition.

## Why two levels — edges and events

An edge-level motif can be real and inconsequential. A head can attend, write a force, and
move nothing that matters, because the force was small, or orthogonal to anything the next
layer reads, or cancelled by another head.

So the second level is the **particle event**: what actually happened to the particle.
`capture`, `hold`, `escape`, `relay_target`, `moved_fraction` — written as `extra__` columns
on the existing `ParticleTable` rather than as a new artifact, because per the project's own
framing the particle table *is* the unit of analysis and this is exactly the kind of
annotation it was built to carry. (`hold` is also, incidentally, the primitive Phase 5c
specified as `noise_tracking.py` and never built: "this token has been unclustered for N
consecutive layers" is a groupby on this table.)

The edge level says who pushed whom. The event level says whether it mattered. A motif that
appears at the edge level and vanishes at the event level is a routing artifact, and P-I4 is
written to catch exactly that.

## The tautology risk, stated in full because it is the phase's central methodological danger

The behavioral induction score is *mean attention on induction pairs*. A motif defined as
*an attentive edge on induction pairs* is the same number wearing a different name.
Correlating them would produce a beautiful result that means nothing, and it would be very
easy to do by accident.

The `relay` motif escapes this only because it adds three things the single-head behavioral
score does not contain:

1. **Two-stage composition** — it requires the ℓ₁ `prev_token` edge. The behavioral score is
   computed per head and knows nothing about what wrote the tag it is matching on.
2. **Force decomposition** — which channel the edge writes into, not merely that attention
   was paid.
3. **The particle event** — whether the target actually moved.

**Any Phase 7 result claiming an association between motif and behavior must state which of
these three is carrying the independence.** If the answer is "none," the phase has measured
one thing twice. This is written into the smoke-tier verification as a hand check
(confirm the two numbers are not numerically identical) rather than left as a caution.

## Nulls: reused wholesale, and why that is the most important decision in the phase

`core/qk_offset_null.py` already implements the three nulls this phase needs, and it was
built for a closely related problem — P6-I2b — after that test was found to be broken in
precisely the way Phase 7 could break. The reasoning transfers without modification: on
Pythia the attention bilinear is `M(Δ) = W_Q R(Δ) W_K^T`, so anything that depends on Δ can
be manufactured by an offset-distribution difference alone. Induction pairs and same-content
pairs are *not* offset-matched. A motif result that does not control for this is worthless.

- **N1** rotary-only, closed form, no weights.
- **N2** offset-matched: same-content pairs at the *same* Δ.
- **N3** offset-shuffled: induction pairs with Δ permuted within the set.

A pass requires clearing **N1 and N2**. N3 separates "content and offset are jointly
required" from "either alone suffices." `core/nulls.py`'s `sigma_from_null` / `nsigma_verdict`
give the verdict format.

`battery_structure`'s four degeneracy modes (`uniform`, `empty_null`, `single_offset`,
`null_is_sink`) gate every prompt before it enters the analysis. On a degenerate prompt the
phase **refuses** rather than returning a number — standing rule 4. `PROMPTS["repeated_tokens"]`
is the worked case: every token identical, so every causal pair is an induction pair, the
null is empty, and no motif claim is evaluable there at all.

## The rest of the translation table — specified now, not run

Stated here so the phase has a destination and so each entry is on record before any of them
is measured. None of these is in scope for the first pass.

- **Steering.** Adding `α·v` to every particle at layer ℓ is a rigid translation of the cloud
  followed by renormalization — an external field added to the interaction dynamics. The
  particle questions: which edges change sign under it, and does a new `hub` form? Note that
  `archive/p3_crosscoder/steering.py` already implements a steering intervention with a
  merge-event readout and recorded a null; this design should be written having read it,
  rather than re-deriving an evaluation that has already been tried.
- **Activation patching.** Teleporting one particle to another run's position. The particle
  question is **recapture vs propagation**: is the patched particle pulled back toward its
  original attractor within *k* layers (the patch is absorbed by the dynamics), or does the
  surrounding interaction graph reorganize around it? This is a question about basins, and it
  is the one entry in the table where the particle account plausibly says something the
  mechinterp framing does not already say.
- **Prompt injection.** Inserting particles that acquire high in-degree attractive edges — a
  **hostile `hub` takeover**. Injection succeeds exactly when injected particles capture the
  force mass previously attracted by the instruction-carrying particles. Needs a new
  adversarial prompt battery, gated by `battery_structure.verify_battery_structure`: the same
  tokenizer-drift problem applies, and an injection battery that loses its structure under the
  NeoX tokenizer would produce a null that looks like a finding.
- **SAEs.** See below.

## SAEs: object of study, never instrument

The standing rule from `core/DESIGN_dual_reading.md` is unchanged and binding here: **no
SAE/LRAE features anywhere in a measurement path.** If a semantic reading is ever wanted from
a sparse dictionary, that is a different primitive with a different name.

What this phase *may* ask is the other question, which that rule never prohibited: **are SAE
dictionary directions aligned with the interaction structure?** — with the V-eigenbasis, with
`U_pos`/`U_neg`, with `U_S`/`U_A`, with attractor directions; and does a feature firing
coincide with a named motif. Here the SAE is the thing being described, exactly as steering
and patching are. This does not reopen Phases 3/4 and does not touch their reintroduction
trigger.

**And it starts from a real prior rather than from zero.** Phase 3 found sparse crosscoder
decoder directions aligned with V at chance (0.484 / 0.501, two models). Phase 4 Track 3
found that *removing* the sparsity penalty recovered the alignment for ALBERT — 33 bottleneck
directions on V-attractive against 0 for GPT-2 — and read it as "sparsity was the confound,
not absence of geometric structure."

The particle-terms restatement, and the thing neither frozen phase could test: a sparse
dictionary allocates capacity to features that are *independent*, while the interaction
structure organizes particles by a *shared attractor*. So the alignment gap between sparse
and dense-low-rank dictionaries should track how attractor-organized the layer is — and on a
checkpoint axis, should widen as attractor structure forms. That is a prediction with a
falsifier, on an axis Phases 3 and 4 did not have. It is the natural fourth study, not the
first: it needs the motif alphabet calibrated before "how attractor-organized" means anything.

## Known constraints carried in from other phases

- **Cone collapse is universal** (Phase 1b). Nothing here should be read as testing for
  antipodal separation; it does not happen, in any model tested, at any layer.
- **Effective *n* is heads, not edges.** Edges within a head are not independent samples. Any
  significance claim computed over edge counts is wrong by orders of magnitude.
- **Every gate records what it read and whether it passed** (standing rule 3), and any
  threshold not derived from a distribution is labelled *placed*, not calibrated, in the code
  next to the value (standing rule 6).
- **Anchors need a non-symmetric arm** (standing rule 5). `UPDATE_PLAN.md` §5.6 is the
  cautionary case: a wrong trace contraction that agreed with the truth at `M = I` and at
  every symmetric `M`, passing its anchor while being wrong for every real head.
