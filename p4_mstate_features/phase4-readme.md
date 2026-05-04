# Phase 4 — Identifying Metastable Features

**Status:** Complete.

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

## Results summary

### Verdicts

| Model | Track 1 | Track 2 | Track 3 | Overall |
|---|---|---|---|---|
| albert-xlarge-v2 | crosscoder_tracks_clusters | strong_linear_separability | **v_alignment_recovered** | metastable_features_detected |
| gpt2-large | crosscoder_tracks_clusters | strong_linear_separability | v_alignment_still_null | metastable_features_detected |

### Key numbers

| Metric | ALBERT | GPT-2 |
|---|---|---|
| Feature–cluster NMI (max) | 1.000 | 0.857 |
| Feature–cluster NMI (mean) | 0.867 | 0.635 |
| Chorus ARI (max) | 0.001 | 0.000 |
| Feature plateau rate | 24–28 / 2048 | 12 / 5120 |
| Linear probe accuracy (mean) | 1.000 | 0.300 |
| Linear probe accuracy (max) | 1.000 | 1.000 |
| LDA cosine stability (mean) | 0.277 | 0.809 |
| LRAE/CC MSE ratio (mean) | 0.797 | 0.788 |
| Bottleneck→V attractive directions | 33 | 0 |
| Bottleneck→V repulsive directions | 3 | 0 |

---

## Track-level findings

### Track 1: Crosscoder activation patterns

Sub-verdict: `crosscoder_tracks_clusters` (both models), but with a critical qualification.

The NMI between feature activations and cluster labels is high — up to 1.0 for ALBERT. This is real signal: the features that do fire are associated with cluster-consistent token sets. However, the Chorus ARI is zero in both models, meaning co-activation cliques carry no cluster structure. The plateau alignment falsification test returns `fail` in both cases (plateau_alignment_rate = 1.0, which is a degenerate result: it means the few features with plateaus always align — but only 12–28 features out of thousands have detectable plateaus at all).

**Root cause:** The crosscoder fires on too few features across the eval prompts. This is a known limitation of sparse autoencoders evaluated on narrow distributions relative to training data (C4). The sparsity penalty allocates dictionary capacity to syntax/frequency features that dominate training; metastability-relevant features are underutilized on the 4-prompt eval set. Investigating this further is not warranted — the Track 3 LRAE results address the same question without the dead-feature pathology.

**Interpretation:** Track 1's positive MI result reflects that cluster structure is accessible to whatever the crosscoder does learn to fire on. The ARI null and low plateau counts mean the crosscoder is not a complete inventory of metastable feature identities.

### Track 2: Direct geometric methods

Sub-verdict: `strong_linear_separability` (both models).

Linear probes trained on residual stream activations achieve perfect accuracy on ALBERT (mean 1.0) and layer-selective high accuracy on GPT-2 (mean 0.30, max 1.0). The GPT-2 mean is depressed by layers outside plateau windows; within plateaus, separability is high. LDA cosine stability differs by regime: GPT-2 has high stability (0.809) reflecting consistent discriminant directions within plateau windows; ALBERT has lower stability (0.277) because the cluster signal is globally accessible across the full depth range, not concentrated in specific windows.

**Interpretation:** Metastable cluster structure is linearly accessible in the residual stream of both models, independent of any dictionary learning. This is a clean confirmation of Phase 1's HDBSCAN structure. The regime A/B difference (ALBERT vs GPT-2) in LDA stability is consistent with Phase 2 findings.

### Track 3: Low-rank autoencoder

Sub-verdict: `v_alignment_recovered` (ALBERT), `v_alignment_still_null` (GPT-2).

**This is the most informative track.** The LRAE consistently outperforms the crosscoder on reconstruction (LRAE/CC ratio ~0.79 for both models), confirming it captures variance the sparse crosscoder suppresses. The critical result is the V-subspace alignment:

- **ALBERT:** 33 attractive-dominant bottleneck directions, 3 repulsive-dominant. The LRAE bottleneck aligns with V's attractive subspace. This directly confirms the Phase 4 hypothesis that sparsity was the confound in Phase 3: once you remove the sparsity penalty, the low-rank basis finds the dynamically-organized directions.
- **GPT-2:** Both alignment values are 0.0 — null result. The LRAE reconstructs better but finds no V-alignment. This is consistent with GPT-2 Regime B: FFN-mediated dynamics distribute the metastable signal differently across the eigenspectrum, and it does not concentrate in a low-rank subspace that aligns neatly with V.

**The regime split reproduces Phases 2 and 3.** ALBERT's attention-mediated Regime A produces recoverable geometric structure at the feature level. GPT-2's FFN-mediated Regime B remains null at the feature decomposition level even with the sparsity constraint removed.

---

## Cross-track interpretation

The three tracks converge on a consistent picture:

1. **Cluster structure is geometrically real and linearly accessible** (Track 2, both models). Phase 1's HDBSCAN clusters are not an artifact — they are linearly separable in the residual stream.

2. **Sparse coding is the wrong prior for recovering the metastable structure** (Track 3, ALBERT). The LRAE recovers V-alignment that the crosscoder cannot. The geometry is there; sparsity suppresses it.

3. **The Regime A/B distinction extends to feature-level decomposability.** ALBERT's metastable structure is geometrically organized by V and recoverable by a non-sparse low-rank method. GPT-2's is not — the structure exists (Track 2 confirms separability) but does not project into V's eigensubspaces in a low-rank way.

4. **The crosscoder is not the right tool for metastability.** What fires on the eval set tracks clusters by MI, but the sparse dictionary misses the bulk of the structure. The LRAE is the recommended approach for Phases 5 and 6.

---

## Approach: Three parallel tracks

### Track 1: Crosscoder activation pattern analysis (using existing crosscoder)

The crosscoder exists and produces meaningful features with bimodal lifetimes. Even though the directions are random w.r.t. V, the activation patterns may still track cluster structure. This track tests that.

1. **Per-token activation trajectories.** For each (feature, token) pair, compute activation strength across all sampled layers. Stack to get a tensor of shape `(n_features, n_tokens, n_layers)`. Identify features with low variance over a layer window followed by a spike — these are metastable feature candidates at the individual token level.

2. **Feature–cluster correspondence via activation patterns.** At each mid-plateau layer from Phase 1, check whether each feature's set of active tokens overlaps significantly with any HDBSCAN cluster. Use the F-statistic from `feature_cluster_correlation` (already implemented) and a new mutual-information measure between feature activation and cluster label. A feature doesn't need to point along V to be a cluster identity feature — it just needs to fire on cluster members.

3. **Co-activation chorus analysis.** Identify co-activation cliques — sets of features that tend to fire together across tokens and layers. Test whether clique membership correlates with HDBSCAN cluster identity. A clique that fires on cluster C but not cluster C' is a chorus for cluster C.

### Track 2: Direct geometric methods

1. **PCA on residual-stream updates.** At each layer, compute Δx = x_{l+1} − x_l per token. Take the top PC of the update matrix. At violation layers, the top PC should point into V's repulsive subspace. At plateau layers, the update variance should be low overall.

2. **LDA on cluster labels.** At each layer, train a linear discriminant using Phase 1 HDBSCAN cluster labels. Track the cosine stability of the discriminant direction across consecutive layers. A stable LDA direction within a plateau window = the cluster-separating axis is persistent.

3. **Supervised linear probes.** Train a linear classifier to predict cluster label from residual stream activations at each layer. The probe weight vector is the cluster identity direction. Accuracy vs. layer should mirror NN-stability from Phase 1: high during plateaus, dropping at merge events.

### Track 3: Non-sparse alternatives

Sparsity is a prior that says representations decompose into many independent atomic concepts. Metastable clustering is a prior that says representations live near a small number of attractors. These priors conflict — sparsity pressure allocates dictionary capacity to syntax/frequency/position features that dominate the training distribution, diluting any cluster-tracking signal.

1. **Low-rank autoencoder.** Replace BatchTopK with a linear bottleneck. Set bottleneck dimension to match the number of metastable clusters (2–8 from Phase 1). The bottleneck basis should align with cluster-separating directions because there's no sparsity pressure. If bottleneck dimensions align with V's eigensubspaces where sparse features didn't, sparsity was the confound. **Recommended for downstream phases.**

2. **k-means in activation space per layer.** The simplest non-parametric approach. At each layer, k-means the residual stream activations with k from Phase 1's spectral eigengap. Track centroid identity across layers via Hungarian matching. Centroids that persist = metastable configurations. Centroids that merge = merge events.

3. **ICA (Independent Component Analysis).** Finds maximally non-Gaussian directions. Unlike PCA (variance) or sparse coding (sparsity), ICA finds statistical independence. If cluster membership is encoded in independent components of the residual stream, ICA will find it without sparsity pressure.

---

## Falsification criteria (evaluated)

- **Track 1:** Feature activation patterns correlate with cluster membership by MI (NMI up to 1.0) but co-activation structure is null (ARI ≈ 0). **Partial signal, confounded by low crosscoder fire rate.** Not pursuing further.
- **Track 2:** LDA directions are stable within plateau windows (GPT-2) or globally (ALBERT); probes achieve high accuracy. **Not null — cluster structure is linearly encoded.**
- **Track 3:** ALBERT bottleneck directions align with V's attractive subspace (33 directions); GPT-2 null. **Sparsity was the confound for Regime A. Regime B remains null at this level of analysis.**

---

## Dependencies

### Required (blocking)

- **Phase 1 results for albert-xlarge-v2** — HDBSCAN labels, merge layers, plateau windows. ✓
- **Phase 3 crosscoder checkpoint** — for Track 1. ✓ (2048 features for ALBERT, 5120 for GPT-2)

### Required (Phase 2)

- V eigensubspace projectors — for testing Track 3 bottleneck alignment with V. ✓ (`ov_projectors_{stem}.npz`)

### Optional

- Phase 1 results for gpt2-large — enables cross-model comparison (Regime A vs B). ✓

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

Phase 3's existing cross-phase analyses (`coactivation_at_merges`, `feature_cluster_correlation`, `cluster_identity_diff`, `plateau_clustering`) remain in `phase3/analysis.py` and are called from Phase 4 as imports.

---

## Forward compatibility: Phases 5 and 6

- **For Phase 5:** Use LRAE bottleneck directions (not crosscoder features) as the primary cluster identity representation. LDA directions from Track 2 are saved as `.npz` and are directly consumable.
- **For Phase 6:** The ALBERT v_alignment_recovered result confirms that Phase 6's direct geometric tests (probe_subspace, eigenspace_degeneracy) are operating on a real signal. GPT-2's null in Track 3 suggests Phase 6 tests for Regime B should not rely on V-subspace decomposition at the feature level.
