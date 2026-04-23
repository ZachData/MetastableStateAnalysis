# Phase 1h — Hemispheric Structure Investigation

**Status:** Planned. Scaffolded by existing `fiedler_tracking.py`, `rotation_hemisphere.py`, and Phase 1 per-layer Fiedler vector extraction.

-----

## Core Question

Phase 1 found that the spectral eigengap heuristic applied to the normalized Laplacian of the token Gram matrix consistently reports $k = 2$ on long prompts — a dominant bipartition — while HDBSCAN simultaneously finds 30–60+ local clusters. The Fiedler vector (second Laplacian eigenvector) defines this bipartition; its sign partitions tokens into two “hemispheres.”

Phase 1h asks: **what is this bipartition, dynamically and semantically?** When does it exist, when does it rearrange, which tokens live where, what determines it, and how does it relate to the hemisphere condition that appears in the paper’s convergence theorems?

-----

## Relationship to the paper’s “hemisphere”

Geshkovski et al. use “hemisphere” as a containment condition, not a bipartition. Theorem 6.3 and Lemma 6.4 (cone collapse) state that if all tokens start in a single open hemisphere ${x : \langle w, x\rangle > 0}$, the dynamics converge exponentially to a single point. The paper’s hemisphere is the setting in which the theorems apply. Our spectral bipartition is the opposite geometry: two populated antipodal half-spaces.

This distinction is itself a result to verify and quantify. Phase 1h tests two regimes at each layer:

- **Cone-collapse regime (paper):** all tokens in one open hemisphere. Operationalized as: there exists a unit vector $w$ with $\min_i \langle w, x_i\rangle > 0$.
- **Split regime (Phase 1 observation):** Fiedler bipartition populated on both sides with compact within-hemisphere geometry.

These are mutually exclusive. The phase maps which regime holds at each depth for each (model, prompt) and identifies the crossover layers.

-----

## Motivation

Five things make the bipartition worth an investigation on its own:

1. It is the lowest-resolution structural statement the token geometry admits. HDBSCAN captures local density but varies with hyperparameters; the Laplacian bipartition is a threshold-free global partition of the graph into its two dominant components.
1. It is persistent across long-prompt runs and across models — the Phase 1 result is not prompt-specific.
1. It has clean causal infrastructure available (Phase 2’s OV eigendecomposition) to test whether any weight-level object predicts the Fiedler axis.
1. It is the simplest object Phases 4–6 would need to interpret if the cluster-identity features exist at the hemisphere level rather than at the HDBSCAN-cluster level.
1. It offers a direct empirical contrast to the paper’s main convergence hypothesis (cone collapse vs. persistent bipartition) at the same geometric granularity.

-----

## Approach

### Block 0 — Bipartition quality and existence

The eigengap result “k = 2” is not itself evidence of a real bipartition. Any graph’s Laplacian has a non-trivial second eigenvalue. The question is whether that second eigenvalue is small enough (relative to the third) and the associated eigenvector partitions the tokens into two geometrically separated, populated sets.

Per layer, record:

- **bipartition_eigengap** — $(\lambda_3 - \lambda_2) / \lambda_3$, the normalized gap that isolates the bipartition. High means bipartition dominates; low means tertiary structure is comparable.
- **bipartition_sharpness** — centroid angle between the two sign-partition halves, in radians. $\pi/2$ is orthogonal; $\pi$ is antipodal; near 0 is collapsed.
- **within_half_ip** — mean pairwise cosine similarity within each half. High means each half is itself a tight cluster.
- **hemisphere_sizes** — $(|A|, |B|)$ with $|A| + |B| = n$.
- **minority_fraction** — $\min(|A|, |B|) / n$. Zero means everything collapsed to one side; 0.5 means perfectly balanced.

Per-layer classification into four regimes:

|Regime              |Minority fraction|Centroid angle|Within-half IP|
|--------------------|-----------------|--------------|--------------|
|`collapsed`         |< 0.05           |—             |—             |
|`weak_bipartition`  |≥ 0.05           |< π/2         |any           |
|`strong_bipartition`|≥ 0.1            |≥ π/2         |≥ 0.3 in both |
|`diffuse`           |≥ 0.1            |≥ π/2         |< 0.3 in both |

Output: per-layer regime label and the raw quality metrics.

### Block 1 — Hemisphere identity tracking across layers

Fiedler vectors are defined up to a global sign flip. To match hemisphere A at layer $L$ to hemisphere A at layer $L+1$, compute Jaccard overlap on token membership with the sign-flip convention chosen to maximize overlap. This is a direct adaptation of `cluster_tracking.match_layer_pair` specialized to $k=2$.

Record per layer-transition:

- **crossing_count** — tokens that switch sides (after sign-flip correction). Already computed by `fiedler_tracking.hemisphere_crossing_rate`.
- **axis_rotation** — $\arccos(|\langle v_L, v_{L+1}\rangle|)$, the angle between consecutive Fiedler vectors. Already computed by `fiedler_tracking.fiedler_stability`.
- **identity_preserved** — bool, whether matched A–A and B–B overlap exceeds 0.5.

Detect hemisphere events:

- **birth** — first layer where regime transitions from `collapsed` → `strong_bipartition`.
- **collapse** — regime transitions from `strong_bipartition` → `collapsed` (minority fraction drops below 0.05).
- **swap** — identity mapping fails (matched overlap < 0.5) despite both layers being in `strong_bipartition`. Indicates the axis has rotated to a different structural partition.
- **shear** — crossing_count in the top 5% of the run, but identity preserved. The partition is the same object, but a specific subset of tokens is being re-routed across it.

Cross-reference every event with Phase 1’s merge events and energy-violation layers.

### Block 2 — Token-level traversal and HDBSCAN nesting

For each token, record its hemisphere label at every layer. Derive:

- **stability_score** — fraction of layer-transitions where the token stays in the same hemisphere (in the identity-corrected frame from Block 1).
- **border_index** — mean of $|v_L(i)|$ across layers, normalized by the per-layer mean. Values near zero mean the token sits near the boundary throughout; large values mean it’s deep in one side.
- **first_assignment_layer** — the earliest layer at which the token’s hemisphere matches its final-layer hemisphere and stays fixed.

HDBSCAN-nesting test. At each plateau layer from Phase 1, for each HDBSCAN cluster $c$, compute the fraction of its tokens in hemisphere A:

$$r_c = \frac{|{i \in c : h_L(i) = A}|}{|c|}$$

A cluster is **nested** if $r_c \in {0, 1}$ (up to a one-token tolerance), **mixed** if $r_c \approx 0.5$. Histogram the $r_c$ values. Full nesting across plateau layers means the token geometry is genuinely hierarchical: dominant bipartition containing the local density clusters found by HDBSCAN. Mixed clusters indicate the bipartition cuts through local density structure, which would argue against a nested-clustering interpretation.

### Block 3 — Paper cone-collapse vs. split regime

Per layer, compute:

- **cone_collapse_witness** — the SVM-like margin problem: does there exist $w \in S^{d-1}$ with $\min_i \langle w, x_i \rangle > 0$? Computed via the linear-programming test $\max_w \min_i \langle w, x_i\rangle$ subject to $|w| \le 1$ (solvable as a small LP or via `scipy.optimize.linprog`). Positive optimum → all-in-one hemisphere exists → cone-collapse regime holds.
- **cone_margin** — the optimum itself; size of the tightest cone containing all tokens. Zero means the tokens exactly span a half-space (borderline); positive means they are in its interior; negative means no enclosing hemisphere exists.

Classify each layer as `cone_collapse`, `borderline`, or `split`. Tabulate the regime transitions through depth. The prediction from Phase 1 is that long prompts enter `split` mid-depth and remain there until the final-layer collapse. Short prompts should be `cone_collapse` throughout.

### Block 4 — Size symmetry and asymmetry distribution

Compute across all (model, prompt, layer) tuples in `strong_bipartition`:

- **asymmetry** $= ||A| - |B|| / (|A| + |B|)$. 0 = perfect split; 1 = single token outlier.
- **mean_layerwise_asymmetry** — per-run, for ranking prompts and models.

Histogram across models. Test: does asymmetry correlate with prompt structure (content-heterogeneity, length, special-token count), with model scale, or with depth within a run?

### Block 5 — Mechanism: what determines the Fiedler axis?

For each (model, prompt, layer) with `strong_bipartition`, compute the absolute cosine of the Fiedler axis against:

- **OV symmetric top eigenvector** — from Phase 2’s saved OV decomposition, the dominant positive eigenvector of $S = (V_{\text{eff}} + V_{\text{eff}}^\top)/2$ at that layer. High alignment means the paper’s attractive subspace is the bipartition axis.
- **OV symmetric bottom eigenvector** — the most-negative eigenvector. High alignment means the repulsive subspace is the bipartition axis.
- **Activation PCA top axis** — top singular vector of the L2-normed activation matrix at that layer. High alignment means the bipartition is just the first principal component.
- **Per-head Sinkhorn–Fiedler axes** — the Fiedler vectors of individual attention heads’ Sinkhorn-normalized attention matrices, already computed in Phase 1. For each head, check whether its Fiedler axis aligns with the residual-stream Fiedler axis. Identifies heads that “drive” the bipartition.
- **Embedding-space separation axis** — top singular vector of the token-embedding matrix for this prompt’s tokens (layer 0). High alignment means the bipartition is inherited from lexical geometry rather than induced by the dynamics.
- **Axis at the previous layer** — tracks rotation of the bipartition axis through depth. Supplements the already-computed `fiedler_cosine`.

Output a per-layer alignment matrix. Read the columns: consistently high OV-S-bottom alignment across layers = mechanism is the repulsive subspace of $V$. Consistently high embedding alignment at all depths = the bipartition is imported from the embedding layer. Consistently high Activation-PCA alignment = spectral bipartition is trivially recovering the PC1, and the whole “hemisphere” framing is a relabeling of the dominant variance direction (in which case the Fiedler analysis adds no information over PCA).

### Block 6 — Token semantics and hemisphere membership

At each `strong_bipartition` layer, tabulate hemisphere assignments against token attributes:

- **Special tokens** — CLS, SEP, BOS, EOS, PAD. Which side, which layer. Do they migrate?
- **Punctuation vs. word tokens** — via regex on token strings.
- **Position** — absolute position, relative position (first third / middle / last third), distance from CLS.
- **Content vs. function words** — via a small POS heuristic (a stopword list is sufficient for a first pass; spaCy if needed).
- **Subword rank** — vocab ID as a proxy for training frequency.

For each attribute, compute the contingency table and a chi-square statistic against hemisphere label. Also compute the mutual information $I(\text{hemisphere}; \text{attribute})$ per layer. Attributes with consistently high MI across layers are the semantic content of the bipartition.

This block answers “what are these hemispheres” at the level a reader can interpret.

### Block 7 — Causal projection (optional / stretch)

At a chosen layer $L$, project the residual stream onto the orthogonal complement of the Fiedler axis, then re-run the model forward from layer $L$. Measure:

- downstream Fiedler axis reappearance (does the bipartition regrow?)
- output perplexity shift on the same prompt
- attention pattern changes at subsequent layers

If the bipartition regrows quickly, the axis is determined by the subsequent layers’ weights, not by residual-stream history. If it stays ablated, the axis is carried forward passively.

This is a forward-hooks experiment. Cheap to implement, but only informative if prior blocks identify a consistent axis; deferred to a follow-up if Block 5 returns inconclusive alignment.

-----

## Falsification criteria

- **Block 0 null:** no layer reaches `strong_bipartition` for any (model, prompt). The Phase 1 $k = 2$ finding was an eigengap artifact, not geometric structure. Phase 1h stops.
- **Block 1 null:** bipartition identity not preserved across consecutive layers within any run (high swap rate, low matched overlap). Hemispheres are not persistent objects; each layer has its own ad hoc bipartition. The “hemisphere” framing is retired.
- **Block 2 null:** HDBSCAN cluster nesting fraction near 0.5 uniformly. Dominant bipartition is orthogonal to local density structure. Nested-clustering interpretation fails.
- **Block 3 null:** cone-collapse regime holds at every layer of every long prompt. Phase 1’s $k = 2$ finding is compatible with all tokens in a single open hemisphere (it would mean the Fiedler split is cutting within a cone, not across one). This falsifies the interpretation that we’re observing antipodal bipartition and aligns results with the paper’s setting.
- **Block 5 null:** no consistent axis alignment — Fiedler axis cosine-distributed near zero against every candidate. The bipartition exists but has no weight-level or geometric explanation. Possible but unlikely if Blocks 0–2 pass.

Blocks 4 (size symmetry) and 6 (semantics) are descriptive rather than falsifying. They produce the phenomenology, not a pass/fail verdict.

-----

## Dependencies

### Required (blocking)

- **Phase 1 run outputs** per (model, prompt) — need saved per-layer L2-normed activations (`.npz`), HDBSCAN labels, plateau and merge event indices, per-head Sinkhorn Fiedler arrays, `fiedler_bipartition` if already saved, token lists. All present in Phase 1 artifact tree.
- **Phase 2 OV artifacts** per model — `ov_decomposition_*.npz` with symmetric eigenvectors and Schur vectors. Required for Block 5.

### Optional

- Phase 1 runs with the full Fiedler vector saved rather than just its sign partition. If the current Phase 1 output dropped the full vector after sign-partitioning, a small patch to `analysis.py` is required: keep `spectral_result["fiedler_vec"]` alongside `fiedler_bipartition`. Alternatively, Block 1 can recompute the Fiedler vectors from the saved activations via `fiedler_tracking.extract_fiedler_per_layer` — already implemented, no new infra.
- Cone-margin LP (Block 3) uses `scipy.optimize.linprog`. Available in the existing environment.

-----

## Code structure

```
phase1h/
├── README.md
├── __init__.py
├── bipartition_detect.py        — Block 0: per-layer regime classifier and quality metrics
├── hemisphere_tracking.py        — Block 1: identity matching, events, axis rotation
├── hemisphere_membership.py      — Block 2: per-token trajectories, HDBSCAN nesting
├── paper_alignment.py            — Block 3: cone-collapse LP test and regime classifier
├── size_symmetry.py              — Block 4: asymmetry distribution and correlations
├── hemisphere_mechanism.py       — Block 5: axis alignment against OV / PCA / embedding / heads
├── hemisphere_semantics.py       — Block 6: token-attribute contingency and MI
├── hemisphere_ablation.py        — Block 7: causal projection experiment (optional)
├── aggregator.py                 — LLM-friendly digest: single JSON + markdown per run + cross-run
└── run.py                        — CLI entry point
```

Reused existing modules (imported, not duplicated):

- `fiedler_tracking.py` — Fiedler vector extraction, hemisphere assignments, crossing rate, stability, centroid separation. Already complete; Block 0/1 calls it and adds classification.
- `rotation_hemisphere.py` — displacement coherence within hemispheres, already complete; feeds Block 5 for the “do hemispheres rotate rigidly” question.
- `cluster_tracking.match_layer_pair` — the identity-matching logic; Block 1 reuses it at $k = 2$.
- `spectral.spectral_eigengap_k` — already returns the full Fiedler vector when requested. Block 0 calls it with `return_fiedler_vec=True` and stores the vectors.

-----

## Aggregation and LLM-friendly output

The aggregator produces, per (model, prompt):

1. **`phase1h_{model}_{prompt}.json`** — single flat JSON for programmatic consumption. Schema:
   
   ```
   {
     "model": "...",
     "prompt": "...",
     "n_layers": int,
     "n_tokens": int,
     "per_layer": [
       {
         "layer": int,
         "regime": "collapsed" | "weak_bipartition" | "strong_bipartition" | "diffuse",
         "cone_regime": "cone_collapse" | "borderline" | "split",
         "bipartition_eigengap": float,
         "centroid_angle": float,
         "within_half_ip": [float, float],
         "hemisphere_sizes": [int, int],
         "minority_fraction": float,
         "asymmetry": float,
         "cone_margin": float,
         "fiedler_axis_alignments": {
           "ov_s_top":     float,
           "ov_s_bottom":  float,
           "activation_pc1": float,
           "embedding_pc1": float,
           "prev_layer_fiedler": float
         },
         "head_fiedler_alignments": [float, ...],  // one per attention head
         "hdbscan_nesting": {
           "mean_r_c_distance_from_half": float,
           "fully_nested_fraction": float,
           "mixed_fraction": float
         },
         "attribute_mi": {
           "position": float,
           "is_special": float,
           "is_punct": float,
           "is_function_word": float,
           "subword_rank_bin": float
         }
       },
       ...
     ],
     "events": [
       {"type": "birth"|"collapse"|"swap"|"shear", "layer": int, "detail": {...}},
       ...
     ],
     "per_token": [
       {
         "token_id": int,
         "token_str": "...",
         "position": int,
         "hemisphere_trajectory": [0|1|-1, ...],   // -1 = invalid layer
         "stability_score": float,
         "border_index": float,
         "first_assignment_layer": int | null
       },
       ...
     ],
     "summary": {
       "strong_bipartition_layer_fraction": float,
       "cone_collapse_layer_fraction": float,
       "mean_axis_rotation": float,
       "mean_asymmetry_strong": float,
       "dominant_mechanism": "ov_s_top" | "ov_s_bottom" | "activation_pc1" | "embedding_pc1" | "none",
       "dominant_semantic_attribute": "position" | "is_special" | ... | "none",
       "crossref_with_phase1": {
         "mean_crossing_at_merge_events": float,
         "mean_axis_rotation_at_violation_layers": float
       }
     }
   }
   ```
1. **`phase1h_{model}_{prompt}.md`** — human-readable summary: one short paragraph per block. Same content, prose form.
1. **`phase1h_cross_run.json`** — cross-run digest. For each model, aggregate the per-prompt summaries. For each prompt, aggregate the per-model summaries. Plus a global verdict section:
   
   ```
   {
     "by_model": { "<model>": {summary stats aggregated over prompts}, ... },
     "by_prompt": { "<prompt>": {summary stats aggregated over models}, ... },
     "global_verdict": {
       "bipartition_exists_universally": bool,
       "bipartition_identity_persistent": bool,
       "hdbscan_nested_in_bipartition": bool,
       "cone_collapse_regime_at_long_prompts": bool,
       "dominant_mechanism_consensus": "...",
       "dominant_semantic_attribute_consensus": "...",
       "paper_alignment": "cone_collapse" | "split" | "mixed"
     }
   }
   ```
1. **`phase1h_cross_run.md`** — one-page synthesis intended to be pasted into an LLM context window. Contains:
- The regime counts per model in one table.
- The axis-alignment medians per model in one table.
- The semantic-attribute MI medians per model in one table.
- The event counts per model in one table.
- Three paragraphs: what the bipartition is, what determines it, how it relates to the paper’s hemisphere.

The flat schema is intentional: one layer per row, no nested structures beyond the per-layer dict. Anything downstream (Phase 4 feature–cluster correspondence, Phase 6 tuned-lens probes on hemisphere centroids) can load `phase1h_{model}_{prompt}.json` and index into `per_layer[layer_idx]["hemisphere_sizes"]` without additional parsing.

-----

## Expected outputs by model, from Phase 1 priors

Prior from Phase 1: the $k = 2$ eigengap finding was on long prompts in ALBERT-base-v2 and GPT-2 families. The main bipartition persistence results should concentrate there. Predictions entering the phase:

- **Short prompts / repeated-token prompts** — expected `cone_collapse` at every layer. Block 3 result: paper regime holds. Nothing more to report.
- **ALBERT-base-v2 at 48 iterations** — expected `strong_bipartition` in the mid-iteration plateau, `collapsed` or single-hemisphere cone collapse at late iterations. Fiedler axis expected to align with OV symmetric top or bottom eigenvector (Block 5), since the OV matrix is shared and any structural axis must be a fixed function of it.
- **GPT-2-small / medium / large / xl** — expected `strong_bipartition` over a wide mid-depth window. Axis rotation across layers expected to be non-trivial (per-layer weights rotate it). GPT-2-xl’s Phase 1 oscillation in late layers may correspond to axis-rotation events without regime collapse — testable.
- **Hemisphere semantics** — the likely dominant attribute is absolute position (tokens early in the sequence vs. late), given what has been observed in similar residual-stream PCA analyses. A surprising result would be a content-based split (e.g., quantum-mechanics tokens vs. cooking tokens in the heterogeneous prompt).

If the predicted patterns hold, Phase 1h produces a clean characterization of the Phase 1 spectral finding that Phases 4–6 can consume at the hemisphere level before descending to HDBSCAN-cluster granularity. If they don’t — especially if Block 5 returns no consistent mechanism — the bipartition is a real phenomenon without an identified cause, which is itself a clean open problem for a standalone writeup.

-----

## Handoff to later phases

- **Phase 4.** The hemisphere-level activation pattern is a two-class label. Training a two-class linear probe on hemisphere membership at each plateau layer is a trivial special case of Phase 4’s Track 2 LDA. If the probe accuracy at plateau layers is high and its weight vector aligns with the Fiedler axis, the hemisphere is a linearly accessible representation — more compelling evidence than HDBSCAN-cluster probes, which have many classes.
- **Phase 5.** The two hemisphere centroids at each `strong_bipartition` layer are the first candidates for “cluster identity vectors” in Phase 5’s sense. Much simpler than HDBSCAN centroids: two per layer, trackable across layers via Block 1 identity matching, and the difference vector (the Fiedler axis itself) is the cluster-separation direction ready for steering experiments.
- **Phase 6.** The two hemisphere centroids passed through a tuned lens produce two token probability distributions per layer. Their KL divergence is the “semantic distance” between hemispheres at that depth. The layer at which this divergence peaks is a strong candidate for the layer “computing the distinction.”