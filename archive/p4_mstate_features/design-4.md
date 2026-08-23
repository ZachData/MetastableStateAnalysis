# Phase 4 — DESIGN

## Core question

Phase 3 confirmed crosscoder features split into short/long-lived populations (Prediction 1)
but their *decoder directions* are geometrically random with respect to V's eigensubspaces
(Prediction 2, clean null). Phase 4 asks the question Phase 3's null leaves open: do
crosscoder features track metastable cluster structure through their *activation patterns*,
even though their decoder directions don't align with V?

The distinction is load-bearing: a feature's decoder direction says how it contributes to
reconstruction; its activation pattern says what it detects. A feature can fire exactly on
one HDBSCAN cluster's tokens — a perfect cluster-identity feature — while pointing in an
arbitrary direction in $\mathbb{R}^d$. Phase 3's null rules out the first kind of alignment,
not the second.

## Why three parallel tracks, not one

Each track tests a structurally different hypothesis about *where* cluster-tracking
structure would live if it exists, and each has a distinct failure mode that the others
don't share:

- **Track 1 (crosscoder activation patterns)** — reuses the *existing* Phase 3 crosscoder
  rather than training a new model, since the question is specifically whether the sparse
  dictionary's activation patterns (not directions) carry signal. Its failure mode is
  low fire-rate on a narrow eval distribution — a property of the crosscoder's training
  distribution (C4), not of the hypothesis being tested.
- **Track 2 (direct geometric methods — PCA on Δx, LDA, linear probes)** — doesn't go
  through any learned dictionary at all. This is the control that establishes whether
  cluster structure is linearly accessible in the residual stream *independent of* any
  representation-learning choice. Its result (strong separability, both models) is what
  makes Track 1's low fire-rate diagnosable as a crosscoder-specific limitation rather than
  evidence that the structure isn't there.
- **Track 3 (non-sparse alternatives — low-rank AE primarily, plus k-means and ICA as
  described alternatives)** — directly tests whether *sparsity itself* was the confound in
  Phase 3. Bottleneck dimension is set to match Phase 1's cluster count (2–8), specifically
  so the basis has no sparsity pressure pushing it toward independent syntax/frequency
  atoms. This is the track that turns out to be the most informative, because it's the one
  that can distinguish "sparsity was the confound" from "the structure genuinely isn't
  V-aligned at the feature level."

## Why sparsity is treated as a prior, not a neutral default

Sparse coding assumes representations decompose into many independent atomic concepts.
Metastable clustering assumes representations live near a small number of attractors. These
priors actively conflict: sparsity pressure allocates dictionary capacity to
syntax/frequency/position features that dominate the training distribution, diluting any
cluster-tracking signal before it can be measured. This is the explicit rationale for Track
3 rather than treating Phase 3's null as final.

## Result and why the regime split reproduces

ALBERT: 33 attractive-dominant and 3 repulsive-dominant bottleneck directions —
`v_alignment_recovered`. GPT-2: both alignment values are exactly 0.0 — `v_alignment_still_null`
even with sparsity removed. This reproduces the Regime A/B split from Phases 2 and 3 at the
feature-decomposition level: ALBERT's attention-mediated dynamics produce recoverable
geometric structure once the wrong prior (sparsity) is removed; GPT-2's FFN-mediated dynamics
distribute the metastable signal in a way that doesn't concentrate into a low-rank
V-aligned subspace even without that constraint. The consistency of this split across three
independent phases (2, 3, 4) — each using a different method — is why it's treated as an
architectural property of the two regimes rather than an artifact of any one method.

## Cross-track interpretation (why the three results are read together, not separately)

1. Track 2 establishes cluster structure is geometrically real and linearly accessible —
   not an HDBSCAN artifact.
2. Track 3 establishes sparse coding is the wrong prior for recovering it in ALBERT
   specifically — the geometry is there, sparsity suppresses it.
3. Track 1's positive MI (but null ARI, low plateau count) is read as: whatever the
   crosscoder does fire on tracks clusters, but it's not a complete inventory of metastable
   feature identities — consistent with (2), not in tension with it.

This is why the phase's actionable output for Phase 5/6 is "use LRAE bottleneck directions
as the primary cluster-identity representation" rather than "use whichever crosscoder
features have high NMI" — the LRAE result is the one that survived the sparsity-prior
critique.

## Module structure

`activation_trajectories.py` (Track 1: per-token feature activation across layers, shape
`(n_features, n_tokens, n_layers)`), `chorus.py` (Track 1: co-activation cliques vs. cluster
identity), `geometric.py` (Track 2: PCA-on-Δx, LDA, linear probes), `low_rank_ae.py` (Track 3:
linear bottleneck replacing BatchTopK), `analysis.py` (cross-track comparison and alignment
tests), `run_4.py` (CLI). Phase 3's existing cross-phase analyses
(`coactivation_at_merges`, `feature_cluster_correlation`, `cluster_identity_diff`,
`plateau_clustering`) are imported from `phase3/analysis.py` rather than duplicated.

## Status per transition plan (v2: frozen-for-deletion, with a caveat specific to this file)

`low_rank_ae.py` is frozen-for-deletion alongside Phase 3, same wording ("candidate for
deletion; git history is the archive") and same reintroduction trigger (activation caches at
≥4 checkpoints **and** a specific particle-dynamics question requiring a dictionary).

Worth flagging explicitly here, though, since this is the one module in the freeze that
produced a *positive* result: Track 3's `v_alignment_recovered` for ALBERT is this phase's
headline finding, and it's a low-rank (not sparse) method. The freeze targets sparse
dictionary methods on the stated rationale that they underperform dense/low-rank
alternatives — which is exactly what Track 3 demonstrated, using this file. Freezing it isn't
inconsistent with that finding: the frozen status is about not doing *further* SAE/dictionary
work speculatively before the Pythia checkpoint data exists to motivate it, not a judgment
that this particular result was wrong. If the reintroduction trigger fires, this is the
method with the strongest track record to build on, not the crosscoder.
