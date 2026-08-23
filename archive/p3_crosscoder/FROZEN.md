# Phase 3 — FROZEN

**Frozen 2026-08-22.** Status: frozen-for-deletion (transition plan v2, item 4).
Verdict: **null**, and robustly so. Full detail in `status-3.md` and `design-3.md`,
which stay authoritative; this file records the freeze and its trigger.

## What was run

A cross-layer sparse dictionary (crosscoder: one shared encoder
`(L_sampled * d_model) -> n_features`, per-layer decoders `n_features -> d_model`),
TopK / BatchTopK sparsity with `k = 64` by default, trained on activations from
**albert-xlarge-v2** and **gpt2-large**. Dictionary widths as configured per model
(2048 ALBERT / 5120 GPT-2 in the Phase 4 readouts that consumed these features).
Both models completed. Last verified 2026-04-29.

## What the null actually was

Not "the code didn't work" and not "we ran out of time" — the features trained fine
and had the expected temporal profile. They were not organized by V's eigenstructure:

| Test | ALBERT-xlarge-v2 | GPT-2-large |
|---|---|---|
| P2 — decoder → V alignment | 0.484 (attract_dominance; random is 0.5) | 0.501 |
| Lifetime × V alignment (Spearman) | ρ = 0.03 | ρ = 0.09, negligible at n = 1041 |
| FFN alignment (cosine) | 0.018 | 0.007 |
| Steering causal effect | null (mean Δmerge = 0) | unrunnable (no baseline merge event) |
| Chorus ARI | 0.000 | 0.000 |

P1 (feature lifetime bimodality) confirmed for ALBERT (BC = 0.622), not for GPT-2
(unimodal, BC = 0.514) — the one non-null row, and it is about lifetimes, not geometry.

The phase's own reading, kept: the crosscoder learned syntax / frequency / surface-form
features with the right temporal profile "but not organized by V's eigenstructure.
Phase 2's mechanism explains *why* energy drops; it doesn't organize *what* the model
represents at the feature level."

Two analyses never ran (`cross_term_feature_weighting`, `induction_feature_tagging`),
both for missing upstream artifacts. `status-3.md` judges that rerunning them is
unlikely to move a null this consistent across two models and four metrics.

## Reintroduction trigger

Unchanged from plan v2, and it is a conjunction, not a wish:

> Activation caches exist at **≥4 checkpoints** *and* a specific particle-dynamics
> question **requires a dictionary**.

Not "once more data becomes available," and not because dictionaries are interesting.
Until both hold, no work happens here.

## What is *not* blocked by this freeze

Asking what SAEs *are*, in particle terms, is a different question from using SAE
features as a measurement instrument. The standing rule (`core/DESIGN_dual_reading.md`)
prohibits the second and never prohibited the first. A future phase may ask whether
dictionary directions align with the interaction structure — with the V-eigenbasis,
with `U_pos`/`U_neg`, with `U_S`/`U_A` — treating the SAE as an object of study.

That question inherits this phase's result as a prior rather than starting from zero,
and it inherits Phase 4's alongside it: the sparse dictionary aligned at chance here,
while Phase 4's *dense* low-rank autoencoder recovered the alignment for ALBERT. See
`archive/p4_mstate_features/FROZEN.md`.
