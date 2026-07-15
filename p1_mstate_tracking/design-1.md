# Phase 1 — DESIGN

## Core question

Geshkovski et al. (*A Mathematical Perspective on Transformers*) prove that transformer
token representations, modeled as interacting particles on $\mathbb{S}^{d-1}$, converge to
a single cluster in the long-time limit, passing through metastable multi-cluster states
before merging. The proof holds for a simplified model ($Q^\top K = V = I_d$). Phase 1's
question: does metastability survive in trained architectures with learned weights,
multi-head attention, and FFN layers?

Falsification criterion: if no plateaus appear in cluster count or inner-product histograms
across consecutive layers, metastability doesn't survive at this scale and the project stops
here. It didn't — Phase 1 passed, and everything downstream is built on that result.

## Why this design

Seven theoretical predictions were checked directly against empirical trajectories rather
than testing the theorem's assumptions abstractly, because the paper's proof conditions
(identity Q^TK and V) never hold in a trained model — the only way to know if the
*qualitative* prediction (metastability) survives is to run it and look for plateaus.

Architecture coverage: 7 architectures (BERT, ALBERT-base, ALBERT-xlarge, GPT-2 through
GPT-2-xl) × 8 prompts, with ALBERT run at 4 iteration depths (12/24/36/48), because ALBERT's
shared-weight architecture makes "iteration depth" a free variable standard per-layer models
don't have — the paper's dynamics are literally an iterated map for ALBERT, and depth sweep
is the cleanest way to see convergence directly.

The 10-seed random-weight sweep of ALBERT-base exists to separate architecture-level
properties (present under any weight draw) from weight-level properties (specific to what
training found). This distinction — first established here — becomes the throughline for
every later phase's "is this architectural or learned" question, most directly Phase 5c's
attention-flip finding.

## Key findings and their downstream role

- **Metastability is architecture-determined, not weight-determined.** Every
  architecture-level quantity (MaxMass, per-head Fiedler classification, two-timescale
  ratio direction) is seed-invariant. Quantities that are seed-sensitive: HDBSCAN cluster
  counts, merge-event layer locations, high-β energy violation counts. This split is what
  motivated running a random-weight twin at every later phase that compares trained vs.
  untrained behavior (most directly Phase 5c).
- **GPT-2 attention heads are universally content-independent**; BERT's are content-driven;
  ALBERT's evolve with iteration depth. This is flagged in the plan as possibly the
  project's largest single finding and is explicitly descoped from Phase 5c into a future
  "Phase 7" routing/flow-analysis thread rather than diluting the current phase.
- **Energy violation** (Theorem 3.4 falsified universally) is the observation Phase 2 exists
  to explain mechanistically.
- **ALBERT-xlarge resists collapse** in a way Theorem 6.1 doesn't predict — governed by V's
  spectral radius, not dimension. This anomaly becomes a first-class object of study in
  Phase 5c (the "unclustered population").

## Module architecture

- `run_1.py` — CLI orchestrator. `--random-baseline` adds the untrained control (now part
  of the standard run); `--sublayer` captures post-attention/post-FFN streams separately
  (supplementary, excluded from cross-run comparison to avoid conflating decomposition
  granularities); ALBERT extended mode runs one forward pass to max depth and slices at
  snapshots rather than re-running per depth, since the iterated map is deterministic given
  the input.
- `analysis.py` — one pass per layer computing every metric off pre-normed activations and
  a single Gram matrix, to avoid recomputing the same pairwise structure per metric.
- `metrics.py` / `clustering.py` — scalar metrics and clustering/projection algorithms kept
  separate: metrics are cheap and always computed; clustering (HDBSCAN, multi-scale nesting,
  pair agreement) is heavier and independently interpretable.
- `reporting.py` — the cross-run report is the primary artifact for downstream analysis
  (not the per-run report), because every later phase's cross-referencing depends on
  comparing across the full model×prompt grid, not one run at a time.
- `io_utils.py` — v2 split-file format (separate JSON per metric family, each <100KB) rather
  than one `metrics.json`, so downstream phases can load only what they need. v1 back-compat
  is kept because early runs used the monolithic format.
- `core/config.py`, `core/models.py` — global registry and model loading factored out early
  because every subsequent phase re-imports the same model list and extraction logic rather
  than re-implementing it. This is also why the duplicate `config.py` in `p1_mstate_tracking/`
  vs. `core/` is flagged in the transition plan for collapsing into one.

## Output format (v2)

Per run: `geometry.json`, `energies.json`, `clustering.json`, `spectral.json`,
`activations.npz`, `attentions.npz`, `clusters.npz`, `centroid_trajectories.npz`,
`llm_report.txt`. Session-level: `llm_cross_run_report.txt`, `experiment.txt`. Everything
Phase 2 needs — activations, plateau windows, merge indices, violation layers, energy drop
token pairs — is saved here specifically so Phase 2 never needs to re-run a forward pass.

## v2 plan: what carries forward unchanged, what's upgraded

Phase 1's 10-seed random-weight sweep is the direct ancestor of the v2 plan's formalized
"seed policy + stability reporting" (core infrastructure item 6) and its distinction between
architecture-level and weight-level quantities is exactly the frame the plan's two-baseline
policy makes explicit for Pythia: a fresh Pythia step-0 init and a norm-matched
`pythia-1.4b-random` (weights of the final checkpoint, randomized to match trained norms) are
kept as two separate objects rather than one "random" condition, because GPT-NeoX's init
variance-scaling isn't comparable to GPT-2's by construction — attraction dynamics scale with
weight norms, so a same-architecture-different-scale random baseline would silently change
what "random" controls for. This phase doesn't need to change anything to accommodate that —
it's a downstream consequence of a distinction this phase already drew.

The energy-attribution figures in this phase's visualization extension
(`energy_decomposition.py`, `energy_attribution_aggregate.py`) previously had no real Pythia
story (parallel-residual has no post-attention/pre-FFN state to decompose the old way). The
v2 plan reframes this as an upgrade rather than a gap to skip around: Pythia's parallel
residual makes attn-vs-FFN decomposition *exact* (Δx = attn_out + ffn_out from the same
input, no ordering confound), which GPT-2's sequential architecture never offered. See Phase
2's design doc for the parallel-residual module this depends on.

## Relationship to Phase 2

Phase 1 measures the *outcome* of the attractive/repulsive tension the paper's framework
predicts (softmax attention pulls together; V's mixed-sign eigenspectrum pushes apart).
Phase 2 measures the tension itself — why the energy violations happen and what the weight
structure is doing. This division (outcome vs. mechanism) is why Phase 2 is a separate
directory rather than an extension of Phase 1's analysis loop.
