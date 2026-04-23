# Phase 5 — Single-Cluster Case Study

**Status:** Not started. Consumes artifacts from Phases 1, 2, 2i, 3, and 4.

-----

## Core Question

Every previous phase has worked in aggregate: 35 model×prompt verdicts, hundreds of features ranked by F-statistic, bulk V-alignment scores. Aggregate statistics established that metastability exists (Phase 1), that OV’s signed component drives energy violations (Phase 2/2i), that crosscoder features bifurcate by lifetime but miss V geometrically (Phase 3), and — pending Phase 4 results — whether features track cluster membership functionally.

Aggregate statistics have a ceiling: they answer “how often” and “how much,” not “what happened here.” Phase 5 asks: **take a single HDBSCAN cluster trajectory from Phase 1 and reconstruct, end to end, the mechanism that creates it, maintains it, and dissolves it.** The deliverable is an interpretable narrative of one piece of the model’s computation, cross-referenced to every framework the project has built.

This is the inverse of Phase 4. Phase 4 asks whether any features track any clusters across all (feature, cluster, layer) combinations. Phase 5 asks, for one cluster: which features, which heads, which FFN channels, which V subspace directions, and what this cluster means semantically.

-----

## What prior phases make available

- **Phase 1:** HDBSCAN labels per layer, Hungarian-matched trajectory chains (`cluster_tracking`), centroid trajectories (`centroid_trajectories.npz`), per-head Sinkhorn Fiedler, CKA, NN stability, merge/birth/death events with token-level accounting, P1-3 nesting, P1-4 semantic/artifact tagging.
- **Phase 2:** Composed $V_\text{eff}$ per layer (or shared for ALBERT), Schur and symmetric decompositions, attractive/repulsive subspace projectors `ov_projectors_{stem}.npz`, per-head OV with rep_frac, FFN subspace projections, per-layer violation classification.
- **Phase 2i:** S/A decomposition, confirmed globally `rotation_neutral`. For this phase: locally this still needs checking along a single trajectory.
- **Phase 3:** Crosscoder checkpoint, per-feature decoder directions, feature lifetime classes, prompt activation store, steering infrastructure.
- **Phase 4:** Per-token feature activation trajectories, feature–cluster MI, chorus cliques, LDA cluster-separation directions, linear probes, low-rank AE bottleneck directions.

All of this exists as a per-prompt, per-layer substrate. Phase 5 indexes into one cluster trajectory and pulls every matching slice.

-----

## Cluster selection

Not every cluster is informative. Selection is not the first interesting step, but getting it wrong wastes the rest of the phase.

Candidate scoring on Phase 1 tracked trajectories:

1. **Lifespan** — at least 6 layers of continuous identity. Below this, too few samples for trajectory statistics.
1. **Merge participation** — trajectory ends in a merge or contains one. A cluster that just persists and then the model collapses isn’t as informative as one that gets absorbed. Merge events are the interpretable unit of computation (per the plan).
1. **Semantic content** — P1-4 tags the cluster as semantic (not induction-artifact). Token strings inspected manually for coherence.
1. **Prompt context** — prefer `sullivan_ballou` or `paper_excerpt` for ALBERT-xlarge; these produced the clearest plateau structure in Phase 1. Avoid `repeated_tokens` (collapse control).
1. **Size** — at least 4 tokens while alive. Clusters of 2–3 are too close to HDBSCAN’s noise floor.
1. **Sibling availability** — the cluster it merges with (or its nearest non-merged neighbor) is also long-lived, to enable contrast analysis.

Scoring produces a ranked list. The selection step outputs one primary cluster plus a runner-up for replication. Manual override is allowed — if the top-scored cluster has uninteresting tokens, take the next.

Expected output: one primary trajectory with (prompt_key, trajectory_id, layer_start, layer_end, merges_with, sibling_trajectory_id).

-----

## Investigations

Seven thematic groups. Each produces a section of the final report.

### A. Structural profile across the cluster’s lifespan

Establish what the cluster *is* before asking why.

- Token membership per layer (the chain — stable within, flux at boundaries)
- Size, mean intra-cluster cosine similarity, cluster radius (max cosine distance from centroid), compactness vs isolation (silhouette against sibling and all-other)
- Centroid trajectory: angular step size per layer, cumulative arc length on $S^{d-1}$, layers where centroid velocity spikes
- Cluster mass contribution: fraction of the global `mass-near-1` metric attributable to this cluster
- CKA contribution: restrict CKA computation to cluster tokens vs complement
- Membership stability: fraction of tokens retained from layer $L$ to $L+1$; onset (birth) layer and dissolution (death/merge) layer with precise token-level accounting
- Nesting (P1-3): does the cluster contain stable sub-clusters, and do those sub-clusters survive the merge?

### B. Paper-theoretical alignment

Test the mathematical predictions of Geshkovski et al. at the level of this cluster.

- Effective $\beta$ estimate: fit the local attention softmax temperature by regressing attention weights on inner products within the cluster
- Intra-cluster mass-near-1 trajectory vs Theorem 6.3 prediction for $n = |\text{cluster}|$, $d = d_\text{model}$
- Energy contribution: $E_\beta$ restricted to cluster pairs, as a function of layer — does it match the monotonicity the theorem predicts for the simplified dynamics?
- V-eigenspace projection of the centroid: at each layer, decompose centroid into attractive and repulsive components (using Phase 2 projectors). The paper’s framework predicts attractive component should grow during stable phases, repulsive component at merge events.
- V-eigenspace projection of cluster-mean displacement $\Delta \bar{x} = \bar{x}^{(L+1)} - \bar{x}^{(L)}$: localizes where the update is pushing the cluster
- S/A decomposition local test (Phase 2i): does the cluster-level update stay `rotation_neutral`, or does antisymmetric energy become locally meaningful inside this cluster’s subspace?
- Rotational blocks intersecting the cluster direction: find the Schur blocks whose invariant 2D plane has highest overlap with cluster-centroid direction and with the cluster-separation (LDA) direction
- Merge-event geometry: at the merge layer, compute the angle between the two pre-merge centroids and the direction along which they fuse; project onto V’s repulsive subspace; test whether fusion proceeds along an attractive direction

### C. Attention mechanism

Per-head, per-layer causal structure.

- Per-head attention restricted to cluster: for each head $h$, extract $A^{(h)}[\text{cluster}, \text{cluster}]$ and $A^{(h)}[\text{cluster}, \text{complement}]$; classify each head as `inward` (attention concentrated inside the cluster), `outward` (cluster tokens attend to non-members), `ignoring`, or `gated-by-position`
- Sinkhorn Fiedler restricted to cluster: algebraic connectivity of the cluster-only attention subgraph
- Per-head contribution to cluster cohesion: scalar $\sum_{i \in C} \langle \Delta^{(h)}_i, \bar{x}_C \rangle$ where $\Delta^{(h)}_i$ is head $h$’s residual contribution at token $i$ — positive means the head is pulling the cluster together
- QK pattern analysis: the top-attended token pairs within the cluster, and whether these are semantically equivalent, syntactically equivalent, or positional
- Per-head OV × cluster direction: $\langle v_h, \bar{x}_C \rangle$ for each head’s dominant OV eigenvector, to identify which heads’ attractive (or repulsive) subspaces align with this cluster’s direction
- FFN projection: at each layer, FFN update projected onto (1) cluster centroid direction, (2) cluster-separating LDA direction. Tests whether FFN actively maintains the cluster or just passes through

### D. Feature-level signatures

Consume Phase 3/4 outputs for the cluster.

- Cluster identity features (from Phase 4 Track 1, MI-ranked): top-20 features by mutual information with cluster membership at each layer the cluster is alive
- Activation trajectories of those features: do they have plateau shapes aligned with the cluster’s lifespan?
- Feature chorus for the cluster: connected components in the co-activation graph restricted to cluster-member tokens; how many features, whether the clique is stable across layers
- At the cluster’s merge event: which features die, which are born, which survive; cross-reference to Phase 4’s `merge_dynamics`
- LDA direction (Phase 4 Track 2) between this cluster and its sibling at each layer: stability (cosine similarity across layers), alignment with V’s repulsive subspace at the merge
- Low-rank AE (Phase 4 Track 3) bottleneck directions: do any align with this cluster’s centroid direction?
- Decoder direction geometry: for the top cluster identity features, compute alignment with V eigensubspaces, with cluster centroid, with LDA direction. Phase 3 found decoder directions random in aggregate — is this still true for the specific features that fire on this cluster?

### E. Semantic content via tuned lens

What does the cluster *mean* at each depth. This is the Phase 6 preview, narrowed to one cluster.

- Train (or load) a tuned lens per layer for ALBERT
- Decode centroid at each layer → token probability distribution; top-20 tokens per layer
- Decode individual cluster members: consistency across tokens confirms that the cluster-semantic-content reading isn’t centroid-averaging artifact
- Shannon entropy of centroid distribution per layer: sharp distributions suggest concrete semantic content; diffuse distributions suggest abstract or pre-computational state
- Sibling-contrast at the pre-merge layers: decode both centroids, compute KL divergence between the two token distributions. The KL drops to zero at the merge — what semantic distinction was the model maintaining before fusing them?
- Distribution evolution: track the top-1 and top-5 token overlap from layer to layer. Stable top tokens during plateau = locked-in semantic content. Rotating top tokens = the cluster is doing something computational rather than representational.

### F. Causal interventions

Phase 1–4 establish correlations. This group tests causation.

- **Head ablation:** at the merge layer, zero the head(s) identified in group C as most attractive-to-cluster. Re-run the forward pass. Measure: does the merge layer shift later? Does cluster membership persist further?
- **Head ablation (control):** ablate an equally contributing but cluster-neutral head. Effect should be small.
- **Steering along cluster direction:** inject $+\alpha \bar{x}_C$ into residual stream of cluster tokens at an early mid-plateau layer. Measure: does cluster cohesion increase (mass-near-1 within cluster), merge delayed?
- **Steering along LDA direction:** inject $+\alpha w_\text{LDA}$ (cluster vs sibling). Measure: does membership flip for tokens near the boundary?
- **Activation patching from sibling cluster onto cluster member:** at a mid-plateau layer, replace token $i$’s residual stream with sibling-centroid. Does it switch cluster at the next layer? At which depth does the patching fail to propagate?
- **Feature ablation:** zero the top-5 cluster identity features from the crosscoder reconstruction and reinject the reconstructed activation. Measure cluster dissolution rate.
- **Cross-reference to Phase 3 steering results:** the existing steering code tests lifetime-class features. Apply it specifically to this cluster’s identity features.

### G. Sibling and contrast

Every interpretation in A–F is strengthened by a contrast baseline.

- Repeat structural profile (A) and attention mechanism (C) for the sibling cluster
- Compute the LDA direction between target and sibling at each layer; this is the most interpretable “distinguishing direction”
- Contrast the token distributions (E) at every layer — where they start nearly identical and diverge, vs where they merge
- Random control cluster: a random same-size subset of non-cluster tokens, tracked as a pseudo-cluster. Most of A–F should return null on this control. If they don’t, the signal in the real cluster is weaker than it looks.

-----

## Questions to answer

Consolidated list. Each maps to one or more groups above.

Paper-theoretical:

1. Does the cluster’s intra-mass approach the Theorem 6.3 prediction for its $n$ and $d$?
1. Is the cluster’s evolution consistent with the energy $E_\beta$ trajectory the paper predicts during stable phases?
1. What fraction of the centroid lies in V’s attractive vs repulsive subspace at each layer, and does this fraction predict the merge timing?
1. At the merge event, does fusion proceed along an attractive direction (confirming the framework) or a repulsive one (violating it)?
1. Is the cluster a Theorem 6.8-type single-cluster-attractor case, or a large-$\beta$ metastable configuration (Figure 4)?
1. Does the rotational component of $V$ have locally measurable effects on this cluster, even though Phase 2i found it globally neutral?
1. Is the effective $\beta$ needed to explain the cluster’s stability consistent with ALBERT’s actual attention temperature?

Mechanism:

1. Which attention heads pull tokens into the cluster, and which push them out?
1. Is cluster cohesion maintained primarily by attention, FFN, or a combination?
1. Does the cluster’s attention structure match any known head type (induction, positional, syntactic, semantic)?
1. Does the FFN actively maintain the cluster, or just pass it through?
1. What layer creates the cluster and how? What layer dissolves it and how?

Representation:

1. Is there a crosscoder feature (or small set) whose activation is bijective with cluster membership?
1. If no single feature, does a chorus of features encode the cluster?
1. Does the cluster-separating LDA direction remain stable across the cluster’s lifespan, or rotate?
1. Do the decoder directions of this cluster’s identity features align with the cluster centroid, or are they geometrically random (as Phase 3 found in aggregate)?
1. Does the low-rank AE recover the cluster direction where sparse features didn’t?

Semantic:

1. What does the cluster *mean* — what tokens does the tuned lens decode the centroid to at each layer?
1. Does the semantic content evolve (computation) or stay fixed (representation)?
1. At the merge: what semantic distinction is lost? What’s the KL divergence trajectory between the two pre-merge token distributions?

Causal:

1. Does ablating the top attractor head(s) dissolve the cluster or shift the merge?
1. Does steering along the cluster direction delay the merge?
1. Does steering along the LDA direction flip membership?
1. Does activation patching from sibling propagate forward?
1. Does zeroing the top identity features dissolve cluster cohesion?

Cross-cutting:

1. Does the mechanistic story (attention + FFN) agree with the feature-level story (identity features + chorus)? Where they disagree, which is right?
1. Does the paper’s mathematical framework predict what we actually see, or are there residual effects the framework doesn’t explain?

-----

## Code structure

```
phase5_case/
├── README.md                        (this file)
├── __init__.py
├── select_cluster.py                — rank and select primary + sibling trajectories
├── cluster_profile.py               — Group A: structural profile across lifespan
├── v_alignment.py                   — Group B: paper-theoretical alignment
├── head_contributions.py            — Group C.1: per-head attention + cohesion scalars
├── ffn_contributions.py             — Group C.2: FFN projection onto cluster directions
├── feature_signature.py             — Group D: identity features, chorus, LDA
├── tuned_lens_cluster.py            — Group E: train/apply tuned lens to this cluster
├── causal_tests.py                  — Group F: ablation, steering, patching
├── sibling_contrast.py              — Group G: sibling + random control
├── report.py                        — assemble all_results.txt (main deliverable)
└── run.py                           — CLI entry point
```

Each module is independent and writes a JSON fragment to the run directory. `report.py` consumes all fragments and produces the final text file.

-----

## Outputs

### Primary deliverable: `cluster_report.txt`

One flat text file, LLM-ingestible, containing every quantitative result the phase produces. Sections mirror the investigation groups. No plots, no images, no JSON nesting beyond what text handles naturally. Headers use plain ASCII (`===` and `---`) for robust parsing. Numbers formatted to fixed precision so diffs are meaningful across runs.

Example structure:

```
================================================================
PHASE 5 CASE STUDY — CLUSTER REPORT
================================================================
model: albert-xlarge-v2
prompt: sullivan_ballou
trajectory_id: 17
layers_alive: 18-41 (lifespan 24)
merges_with: trajectory_id 23 at layer 42
sibling_id_for_contrast: 23
selection_score: 8.4 (rank 1 of 47 candidates)

----------------------------------------------------------------
A. STRUCTURAL PROFILE
----------------------------------------------------------------
A.1 Token membership by layer
  Layer 18: [novelist, poet, writer, author]  (n=4)
  Layer 19: [novelist, poet, writer, author, essayist]  (n=5)
  ...
A.2 Compactness trajectory
  layer  size  ip_mean  radius  silhouette_vs_sibling  silhouette_vs_all
   18     4    0.842    0.18    0.61                    0.44
   ...

----------------------------------------------------------------
B. V-EIGENSPACE ALIGNMENT
----------------------------------------------------------------
B.1 Centroid attractive/repulsive decomposition
  layer  ||proj_attr||  ||proj_rep||  attr_fraction
   18     0.73          0.42          0.75
   ...
B.2 Merge-event geometry
  fusion_direction_vs_attractive_subspace: cos=0.81
  fusion_direction_vs_repulsive_subspace : cos=0.34
  verdict: fusion_attractive_dominant

...
```

All numbers include units or scale context. Every verdict-style claim names the metric and threshold it was computed from. No prose narrative, just structured data rows with minimal headers.

### Secondary outputs

- `cluster_metadata.json` — selection metadata, ranked candidate list
- `per_layer_arrays.npz` — centroid trajectory, cluster mask, subspace projections, feature activations for cluster tokens only (keeps the report text short)
- `causal_results.json` — counterfactual outcomes from Group F
- `tuned_lens_distributions.npz` — top-k tokens per layer for centroid and each member
- `plots/` — optional visualisations (centroid on S^{d-1} via PCA, feature-cluster heatmap, per-head contribution bars). Not required for the report.

-----

## Dependencies

Required (blocking):

- Phase 1 run for the chosen prompt on albert-xlarge-v2 with HDBSCAN labels, cluster tracking, and centroid trajectories
- Phase 2 V projectors (`ov_projectors_albert_xlarge_v2.npz`)
- Phase 3 crosscoder checkpoint
- Tuned lens training data (C4 subset used in Phase 3 is sufficient) — or skip Group E if tuned lens isn’t trained in time

Required (soft):

- Phase 4 Track 1 MI results and Track 2 LDA directions — Group D can partly reproduce these inline but reuses Phase 4 outputs if available
- Phase 2i rotational-spectrum per-layer arrays

Optional:

- GPT-2-large equivalents for cross-model replication of a matched cluster

-----

## Falsification criteria

The phase can fail cleanly in several ways, each informative:

1. **No cluster meets selection criteria** — metastable trajectories in ALBERT-xlarge are too short-lived or too noisy for case-study work. Implies the real metastability is at the bulk-geometry level (Phase 4 null) and not at the trajectory level.
1. **Structural profile is unremarkable** — the cluster just persists and dissolves without distinctive attention, FFN, or feature signatures. Implies the cluster is a statistical artifact of HDBSCAN, not a computational object.
1. **V-alignment is random at the cluster level** — the centroid has no meaningful attractive/repulsive decomposition trajectory. Confirms and strengthens Phase 3’s null: V’s eigenstructure doesn’t organize specific clusters, only the bulk.
1. **No attention heads or features are cluster-specific** — cohesion is distributed across all heads/features. Implies the representation is genuinely holographic and no atomic-unit interpretation is valid.
1. **Tuned lens distributions are incoherent** — the cluster doesn’t have decodable semantic content at the layers where it’s alive. Implies the cluster is mid-computation, not a stable representation.
1. **Causal tests return null** — ablation, steering, and patching don’t shift the merge or flip membership. Implies the cluster structure is an epiphenomenon of pre-merge-layer attractors rather than a load-bearing piece of the computation.

Each null result is a publishable constraint on the metastability-as-interpretability claim. A mixed pattern — some groups positive, some null — is the most likely outcome and the most informative about where the paper’s framework does and doesn’t predict real model behavior.

-----

## Open questions for this phase

1. **Does a cluster have a single “identity,” or is identity layer-dependent?** The cluster token membership is (mostly) stable across its lifespan — but the mechanism that maintains it might not be. If the attention heads responsible change from layer to layer, the cluster is being maintained by different circuitry at different depths, and there’s no unified “what this cluster is doing” answer.
1. **How much does the sibling cluster matter for interpretation?** A cluster’s identity might only be meaningful relative to what it’s being distinguished from. The LDA direction is always *between* two clusters. If we picked a different sibling (next-merged, or nearest-neighbor), would the whole interpretation shift?
1. **Is “merge” a single event or a distributed transition?** Phase 1 flags a merge at a specific layer. If causal tests show the dissolution is already underway two layers before the merge and completes one layer after, the “merge layer” is a summary statistic, not a boundary.
1. **Does the mechanism generalize?** If the runner-up cluster tells a completely different story, we have one case study, not a theory. The phase should always run the pipeline on at least two clusters and report the degree of agreement.

-----

## Forward compatibility

- **Phase 6 (tuned lens full):** Group E here is a preview. If the tuned lens infrastructure works on this cluster, Phase 6 scales it to all clusters.
- **Phase 7 (if pursued — generalization):** The pipeline here is per-cluster. Running it on 20+ clusters produces a distribution of mechanistic stories. Whether the distribution is unimodal (one story) or multimodal (multiple distinct mechanisms) is a Phase 7 question.
- **Publishable narrative unit:** this phase is the first point in the project where a single interpretable story — “here is what the model is doing at this cluster, cross-referenced to mathematics, mechanism, features, and semantics” — can be written. Prior phases were measurement; this is explanation.