# Phase 4 — Identifying Metastable Features

**Status:** Not started. Partially scaffolded by Phase 3 cross-phase analyses.

---

## Core Question

Phase 3 confirmed that crosscoder features split into short-lived and long-lived populations (Prediction 1), but their decoder directions are geometrically random with respect to V's eigensubspaces (Prediction 2 — clean null). Phase 4 asks: **do crosscoder features track metastable cluster structure through their activation patterns, even though their decoder directions don't align with V?**

The distinction matters. A feature's decoder direction says how it contributes to reconstruction. Its activation pattern says what it detects. A feature can fire on exactly the tokens in one HDBSCAN cluster — making it a perfect cluster identity feature — while pointing in an arbitrary direction in R^d. The Prediction 2 null rules out geometric alignment of decoder directions with V. It does not rule out functional alignment of activation patterns with cluster membership.

---

## What Phase 3 established

- Feature lifetime bimodality confirmed (ALBERT: 236 short-lived, 343 long-lived, 99.2% multilayer, 0% positional)
- Decoder directions indistinguishable from random w.r.t. V's top-64 eigenvectors (SNR 0.18×)
- Interpretation A (likely): crosscoder trained on C4 learned syntax/frequency features that happen to have the right temporal profile, not dynamical features organized by V
- Interpretation B (possible): 4-prompt eval set too narrow; metastability-rich prompts might recover signal

**Implication for Phase 4:** We cannot rely on decoder geometry to identify metastable features. We must work through activation patterns — which tokens a feature fires on, and whether those token sets correspond to Phase 1's cluster assignments.

---

## Approach: Three parallel tracks

### Track 1: Crosscoder activation pattern analysis (using existing crosscoder)

The crosscoder exists and produces meaningful features with bimodal lifetimes. Even though the directions are random w.r.t. V, the activation patterns may still track cluster structure. This track tests that.

1. **Per-token activation trajectories.** For each (feature, token) pair, compute activation strength across all sampled layers. Stack to get a tensor of shape `(n_features, n_tokens, n_layers)`. Identify features with low variance over a layer window followed by a spike — these are metastable feature candidates at the individual token level.

2. **Feature–cluster correspondence via activation patterns.** At each mid-plateau layer from Phase 1, check whether each feature's set of active tokens overlaps significantly with any HDBSCAN cluster. Use the F-statistic from `feature_cluster_correlation` (already implemented) and a new mutual-information measure between feature activation and cluster label. A feature doesn't need to point along V to be a cluster identity feature — it just needs to fire on cluster members.

3. **Co-activation chorus analysis.** Individual features may be random w.r.t. clusters, but *sets* of co-active features may not be. At each plateau layer, compute the co-activation matrix over features. Identify feature cliques (connected components at a co-activation threshold). Test whether tokens that activate the same clique belong to the same cluster. The hypothesis: cluster identity is encoded in the joint activation of a feature chorus, not in any single feature.

4. **Feature plateau–cluster plateau alignment.** The original Phase 4 falsification criterion: do feature activation plateaus align with cluster count plateaus from Phase 1? Compute per-feature plateau windows (rolling variance < threshold across layers), aggregate across tokens, and test overlap with Phase 1 plateau layer ranges.

5. **Coordinated reorganization events.** At merge layers from Phase 1, identify sets of features that change activation simultaneously across tokens. This is `coactivation_at_merges` (already implemented). Extend it: do the features that die at a merge correspond to the pre-merge cluster identity features from item 2?

### Track 2: Direct geometric methods (no learned dictionary)

These methods bypass the crosscoder entirely and work directly on the activation geometry. They serve as ground truth for Track 1 — if the crosscoder features track cluster structure, they should agree with these results.

1. **LDA / contrastive directions.** At each plateau layer, compute the linear discriminant direction that maximally separates HDBSCAN clusters. Track this direction's stability across layers (cosine similarity of LDA vectors at consecutive layers). Stable LDA direction = metastable window. Rotating LDA direction = merge event.

2. **PCA on layer-to-layer deltas.** Compute Δx = x^(L+1) - x^(L) at each layer. PCA on these deltas reveals which directions carry the most update variance. At violation layers, the top PC should point into V's repulsive subspace (testing the connection Track 1 can't make). At plateau layers, the update variance should be low overall.

3. **Supervised linear probes.** Train a linear classifier to predict cluster label from residual stream activations at each layer. The probe weight vector is the cluster identity direction. Accuracy vs. layer should mirror NN-stability from Phase 1: high during plateaus, dropping at merge events. This is the simplest test of whether cluster structure is linearly accessible, independent of any dictionary learning.

### Track 3: Non-sparse alternatives

Sparsity is a prior that says representations decompose into many independent atomic concepts. Metastable clustering is a prior that says representations live near a small number of attractors. These priors conflict — sparsity pressure allocates dictionary capacity to syntax/frequency/position features that dominate the training distribution, diluting any cluster-tracking signal.

1. **Low-rank autoencoder.** Replace BatchTopK with a linear bottleneck. Set bottleneck dimension to match the number of metastable clusters (2–8 from Phase 1). The bottleneck basis should align with cluster-separating directions because there's no sparsity pressure. If bottleneck dimensions align with V's eigensubspaces where sparse features didn't, sparsity was the confound.

2. **k-means in activation space per layer.** The simplest non-parametric approach. At each layer, k-means the residual stream activations with k from Phase 1's spectral eigengap. Track centroid identity across layers via Hungarian matching. Centroids that persist = metastable configurations. Centroids that merge = merge events. No learned features needed.

3. **ICA (Independent Component Analysis).** Finds maximally non-Gaussian directions. Unlike PCA (variance) or sparse coding (sparsity), ICA finds statistical independence. If cluster membership is encoded in independent components of the residual stream, ICA will find it without sparsity pressure.

---

## Falsification criteria

- **Track 1 null:** Feature activation patterns don't correspond to cluster membership at plateau layers → crosscoder features track syntax/frequency, not dynamical structure, even at the activation level. The Interpretation A from Phase 3 is confirmed as total (not just geometric).
- **Track 2 null:** LDA directions are unstable even within plateau windows → cluster structure is not linearly encoded in the residual stream. This would be surprising given Phase 1's HDBSCAN results.
- **Track 3 null:** Low-rank AE bottleneck directions also random w.r.t. V → the dissociation between mechanism (V eigenspectrum) and representation (learned features) is fundamental, not an artifact of sparsity.
- **Cross-track null:** If Tracks 1–3 all fail to connect features to clusters, the dynamical structure from Phases 1–2 is real but doesn't organize the representation at a level accessible to dictionary learning. The metastability is a property of the bulk geometry (pairwise inner products, cluster counts) that doesn't decompose into feature-level units.

The original falsification criterion stands: **feature plateaus don't align with cluster count plateaus → features aren't tracking the metastable configurations.** But it now applies to all three tracks independently, and failure in one doesn't invalidate the others.

---

## Dependencies

### Required (blocking)

- **Phase 1 results for albert-xlarge-v2** — HDBSCAN labels, merge layers, plateau windows. Without these, Tracks 1 and 2 return errors on every cross-phase analysis. This is the single most important blocker.
- **Phase 3 crosscoder checkpoint** — for Track 1. Already exists (verify: 2048 or 8192 features?).

### Required (Phase 2)

- V eigensubspace projectors — for testing whether Track 2's LDA directions or Track 3's bottleneck directions align with V. Already saved as `ov_projectors_{stem}.npz`.

### Optional

- Phase 1 results for gpt2-large — enables cross-model comparison (Regime A vs B).
- WandB integration — for logging Track 3 training runs.

---

## Code structure

```
phase4/
├── README.md
├── __init__.py
├── activation_trajectories.py  — Track 1: per-token feature activation across layers
├── chorus.py                   — Track 1: co-activation cliques and cluster correspondence  
├── geometric.py                — Track 2: LDA, PCA on deltas, linear probes
├── low_rank_ae.py              — Track 3: low-rank autoencoder (no sparsity)
├── analysis.py                 — cross-track comparison and alignment tests
└── run.py                      — CLI entry point
```

Phase 3's existing cross-phase analyses (`coactivation_at_merges`, `feature_cluster_correlation`, `cluster_identity_diff`, `plateau_clustering`) remain in `phase3/analysis.py` and are called from Phase 4 as imports. No duplication.

---

## Forward compatibility: Phases 5 and 6

Phase 4 outputs should be structured so Phases 5 and 6 can consume them without re-running:

- **For Phase 5 (cluster identity + merge characterization):** Save per-plateau-layer cluster identity feature sets (from Track 1) and LDA directions (from Track 2) as `.npz`. Phase 5 computes centroids in feature space and characterizes merge events — it needs to know which features/directions define each cluster at each window.

- **For Phase 6 (tuned lens backwards tracing):** Save cluster centroids in residual stream space at each plateau layer, keyed by `(prompt, layer, cluster_id)`. Phase 6 passes these through tuned lens probes to get token probability distributions. The centroid vectors must be in the model's native activation space (not crosscoder space, not LDA-projected space).

- **For future experiments (activation patching, linear probes as interpretability tools):** Track 2's linear probes are the direct bridge. The probe weight vectors are cluster identity directions. Activation patching along these directions during plateaus should preserve cluster membership; patching at merge events should have weaker effects. This prediction falls out of the metastability framework and can be tested without additional training.

---

## Open questions for this phase

1. **Is the chorus hypothesis viable?** If no individual feature tracks a cluster, but a clique of 5–10 co-active features does, what does that mean for SAE interpretability more broadly? It suggests features aren't atomic units of meaning — they're basis vectors in a distributed code, and the meaningful unit is the activation pattern, not the feature.

2. **Does the low-rank AE recover V-alignment?** If yes, the story is clean: sparsity was the wrong prior for this problem, and the right decomposition (low-rank, matching cluster count) recovers the theory-to-representation link. If no, the dissociation is deeper than sparsity.

3. **How do the three tracks compare quantitatively?** If Track 2 (direct geometry) finds strong cluster-direction alignment but Track 1 (crosscoder) doesn't, the crosscoder is adding noise. If Track 1 finds something Track 2 misses, the crosscoder is capturing nonlinear structure the linear methods can't see.
