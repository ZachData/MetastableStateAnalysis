# Phase 5c — DESIGN

## Core question

Phase 5 treated HDBSCAN clusters as the unit of interest and reconstructed the mechanism
behind one, end to end. The ~40–50% of tokens HDBSCAN never assigns to a cluster were
implicitly "everything else," outside that phase's scope. Phase 5c asks whether that's the
wrong frame: the Geshkovski dynamics drive every token toward a single collapsed point, but
GPT-2-large doesn't reach total collapse. Is the unclustered population (a) overflow the
architecture fails to collapse, computationally inert relative to the clustered population,
or (b) the place where the network keeps representations individuated because collapsing
them would cost something on the actual task (induction, n-gram completion,
position-tracking — anything needing a specific token, not its cluster's attractor)?

The working hypothesis is that both stories are true of different tokens, and the relevant
resource-allocation unit isn't "in cluster" vs. "out of cluster" but something like a fixed
dimensionality budget the network spends on whatever still needs distinguishing — motivated
by effective rank plateauing near ~200 across models whose $d_\text{model}$ spans roughly
768–1600. If the network just used whatever space training gave it, effective rank should
scale with $d_\text{model}$. It doesn't — that's the first thing this phase tries to explain
rather than treat as a curiosity.

## Why the motivating evidence changed mid-design (kept visible, not silently edited)

An earlier draft cited ALBERT-base's random-vs-trained two-timescale ratio (13.44 vs. ≤1.25)
as evidence that random weights resist collapse *more* than trained weights. That's a
misreading — the ratio is *mean plateau width / collapse onset layer*, a relative measure of
plateau duration, not of whether or how fast total collapse is eventually reached.
Random-weight ALBERT-base collapses fully and reliably (MaxMass=1.0000 every seed); the
ratio is ALBERT-specific and says nothing about GPT-2. The stronger motivating comparison,
kept in its place: GPT-2-large's energy stays close to monotonic under random weights but is
flat-to-slightly-decreasing across layers 5–30 under trained weights — training actively
pushing against the architecture's default, not just a property already present
pre-training. This correction is preserved in the document rather than removed because the
reasoning behind it constrains how Group C and D results should be read later.

## Why the attention-flip finding reshapes Group D's prior before it's even run

`noise_importance_proxy.py`'s attention-flip result (trained models route attention *toward*
unclustered tokens at 1.6–2×, sign-flipped relative to random weights in every model
examined) is treated as the primary trained-vs-random evidence motivating this phase, ahead
of the energy-plateau asymmetry (corroborating context, not primary). This changes the null
hypothesis Group D should be designed against: not "unclustered tokens are inert overflow"
(a null that would predict no attention preference either way) but a moderate-to-large
effect. The document explicitly notes Group D should be powered accordingly rather than
sized for a null result.

## Why Group B is descoped rather than folded in

GPT-2's attention heads being universally content-independent (fixed at training, identical
across inputs — established in Phase 1) is flagged as possibly the project's largest single
finding, and investigating what that fixed routing actually computes (induction, n-gram
completion, skip-trigrams) is a real research thread on its own. Bundling it into a phase
already scoped around "does clustering exist and what's the unclustered population doing"
would force Phase 5c to carry two separate questions to publication. It's kept as a
"Deferred" section (candidate "Phase 7") with a concrete starting point (Phase 1's per-head
Fiedler tables, Phase 6's `induction_ov.py`/`head_classify.py`) rather than dropped
entirely.

## Why token-level selection needs new machinery, not a copy of Phase 5's

Phase 5's `select_cluster.py` scores cluster *trajectories*. Phase 5c's unit is a *token*,
and persistence means "stayed unclustered," not "stayed in the same cluster" — this is why
`noise_tracking.py` is new primitive work, not a wrapper around `cluster_tracking.py`
(which only matches and chains label ≠ −1 across layers). The five-criterion scoring
(persistence, attention-received rank, content-word status, matched-control availability,
cross-prompt positional consistency) deliberately mirrors Phase 5's `SCORE_WEIGHTS`
structure for comparability, with one new criterion (cross-prompt consistency) added
specifically to distinguish a structurally-fixed unclustered *position* (an architectural
role) from a content-flipping one (tracks specific information) — this distinction changes
how every downstream result should be read, so it's built into selection rather than added
as a post hoc interpretation.

## Why Group C tests three specific outcomes rather than one hypothesis

The effective-rank budget test (`rank_decomposition.py`) is designed to fail cleanly in
either direction rather than assume the "waste not, want not" framing going in: unclustered
rank could scale with $d_\text{model}$ while clustered rank doesn't (population-mixture
story), neither could scale (a genuinely fixed, model-independent budget — the strongest
form of the hypothesis), or both could scale proportionally (the plateau is a superposition
artifact and the motivating observation doesn't survive decomposition). All three are
pre-registered as legible outcomes, not just the favored one.

## Why Group D needs both directions, not just force-collapse

The design explicitly runs force-collapse (does the network need this token individuated)
and force-disperse (does the network rely on this token being collapsed) as symmetric
questions, with matched controls for each (collapse-control, random-displacement-control,
disperse-control) — collapsing only unclustered tokens and calling it a day would leave the
"clustering discards what's safe to discard" half of the hypothesis untested. Readout is
next-token cross-entropy delta and KL divergence, run on both the trained model and its
random-weight twin per arm, so every causal result has the same trained-vs-random contrast
that motivated the phase in the first place.

## v2 plan: this phase's frame becomes the project's frame

The transition plan's "Framing: particles first" section promotes this phase's central
object — every particle (token), clustered or not, and how it evolves — to be the unit of
analysis for the whole Pythia transition, not a Phase 5c-specific concern. Two direct
consequences worth tracking:

1. **The per-particle-record schema (core infrastructure) generalizes `noise_tracking.py`.**
   This phase already needed a primitive `cluster_tracking.py` doesn't have: "this specific
   token has been noise for N consecutive layers." The plan's canonical artifact shape — a
   long table keyed by (prompt, token position, layer, step) with columns for cluster label,
   population tag, V-projection, and dual-reading output — is that same idea, generalized
   across every phase rather than built once here and re-derived elsewhere. Once it exists,
   Group A's persistence tracking and the population selector (used by Phase 5's
   `v_alignment.py`, `probe_subspace.py`, etc.) are both just filters on the same table,
   rather than this phase needing its own bespoke tracking module.
2. **The two-random-baselines policy is a direct extension of this phase's own trained-vs-
   random methodology.** This phase's strongest result (the attention-flip sign reversal,
   present in trained models and absent in random-weight twins) is exactly the kind of
   contrast the plan's `pythia-1.4b-random` (norm-matched, not fresh-init) construction is
   designed to preserve correctly on the new architecture. The plan explicitly cites this
   phase's energy-plateau asymmetry alongside the attention flip as motivating evidence in
   its own framing section — this phase's evidence is now part of the project-level
   falsification record (see `PREDICTIONS.md`, claim (a): "Collapse-resistance is learned,
   not initial," tested directly against Pythia step 0 and step 8).

Nothing about this phase's own blockers changes because of this reframe — `noise_tracking.py`
still needs building, the GPT-2 hook-wiring bug in `causal_tests.py` still blocks Group D.
What changes is that this phase's eventual results carry weight for every other phase's
particle-level aggregations, not just for the unclustered-population question in isolation.

## Known constraints carried in from other phases (not re-derived here)

- Cone-collapse is universal (Phase 1h) — nothing in this phase should be read as testing
  for antipodal separation; it never happens, at any layer, in any model tested.
- Final-layer LM-head contamination in gpt2-small/medium (Phase 1, Known Gap 7) — excluded
  from any selection or readout treating late-layer geometry as meaningful dynamics.

## Code structure (proposed, not yet built as a package)

`noise_tracking.py` (Group A, new primitive), `particle_profile.py` (Group A, structural
profile), `select_particle.py` (candidate gates + scoring), `rank_decomposition.py` (Group
C), `causal_tests.py` (Group D — extends, does not duplicate,
`p5_single_mstate_analysis/causal_tests.py`), `report.py`, `run_5c.py`. Visualization
additions live in `p1_mstate_tracking/visualization/` (wired into `ALL_PLOTS`):
`cluster_orthogonality.py`, `ip_population_dynamics.py`, `noise_importance_proxy.py`,
`attractor_alignment.py` — these already exist and produced the preliminary findings above.
