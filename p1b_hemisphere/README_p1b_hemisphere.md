# Phase 1h — Hemispheric Structure Investigation

**Status:** Complete. Results below reflect a full cross-run synthesis over ALBERT-base-v2 (×4 iteration depths), ALBERT-xlarge-v2 (×4), GPT-2, GPT-2-medium, GPT-2-large, and GPT-2-xl.

Blocks 5 (mechanism) and 6 (semantic MI) were not run — they require Phase 2 OV decomposition artifacts, which were not supplied. They are attempted silently if `--phase1-dir` is provided and the relevant artifacts exist.

-----

## Core Question

Phase 1 found that the spectral eigengap heuristic applied to the normalized Laplacian of the token Gram matrix consistently reports $k = 2$ on long prompts — a dominant bipartition — while HDBSCAN simultaneously finds 30–60+ local clusters. The Fiedler vector (second Laplacian eigenvector) defines this bipartition; its sign partitions tokens into two "hemispheres."

Phase 1h asks: **what is this bipartition, dynamically and semantically?** When does it exist, when does it rearrange, which tokens live where, what determines it, and how does it relate to the hemisphere condition that appears in the paper's convergence theorems?

-----

## Relationship to the paper's "hemisphere"

Geshkovski et al. use "hemisphere" as a containment condition, not a bipartition. Theorem 6.3 and Lemma 6.4 (cone collapse) state that if all tokens start in a single open hemisphere $\{x : \langle w, x\rangle > 0\}$, the dynamics converge exponentially to a single point. The paper's hemisphere is the setting in which the theorems apply. The spectral bipartition is the opposite geometry: two populated antipodal half-spaces.

This distinction is itself a result to verify and quantify. Phase 1h tests two regimes at each layer:

- **Cone-collapse regime (paper):** all tokens in one open hemisphere. Operationalized as: there exists a unit vector $w$ with $\min_i \langle w, x_i\rangle > 0$.
- **Split regime:** Fiedler bipartition populated on both sides with compact within-hemisphere geometry.

These are mutually exclusive.

### What the run found

**The cone-collapse regime holds universally.** Block 3 (LP cone test) returned 100% cone-collapse at every layer for every model. The paper's precondition is satisfied throughout the residual stream — at all depths, for all tested prompts. The `split` regime does not appear. This is Block 3 null: the LP test never returns a negative cone margin.

**Strong bipartition is absent.** Block 0 returned 0.0% `strong_bipartition` layers across all models (gpt2-medium: 2.4%, effectively zero). The Fiedler split never achieves the quality threshold — compact, separated, internally coherent — required to be called a genuine bipartition.

**The Fiedler axis is nonetheless stable.** Despite no strong bipartition, `bipartition identity persistent: True` and mean axis rotation is low (0.121–0.642 rad across models). The Gram matrix has a persistent directional structure: the Fiedler vector tracks coherently across layers and hemisphere labels are preserved at most transitions. The axis exists; it just doesn't divide tokens cleanly.

These findings align the empirical geometry with the paper rather than against it. The Phase 1 $k=2$ eigengap finding reflects a real geometric axis in the Gram matrix, but the axis does not define antipodal clusters — all tokens remain on the same open half-space throughout.

-----

## Results

### Regime counts by model

| Model | Strong bipartition % | Cone-collapse % | Mean axis rotation (rad) |
|-------|---------------------|-----------------|--------------------------|
| albert-base-v2@12 | 0.0% | 100.0% | 0.395 |
| albert-base-v2@24 | 0.0% | 100.0% | 0.396 |
| albert-base-v2@36 | 0.0% | 100.0% | 0.479 |
| albert-base-v2@48 | 0.0% | 100.0% | 0.642 |
| albert-xlarge-v2@12 | 0.0% | 100.0% | 0.192 |
| albert-xlarge-v2@24 | 0.0% | 100.0% | 0.297 |
| albert-xlarge-v2@36 | 0.0% | 100.0% | 0.265 |
| albert-xlarge-v2@48 | 0.0% | 100.0% | 0.311 |
| gpt2 | 0.0% | 100.0% | 0.340 |
| gpt2-medium | 2.4% | 100.0% | 0.201 |
| gpt2-large | 0.0% | 100.0% | 0.134 |
| gpt2-xl | 0.0% | 100.0% | 0.121 |

GPT-2 axis rotation decreases monotonically with model scale. ALBERT-base rotation increases with iteration depth; ALBERT-xlarge is more stable throughout.

### Token stability by model

| Model | Mean stability score | Fraction never stable |
|-------|---------------------|-----------------------|
| albert-base-v2@12 | 0.864 | 47.7% |
| albert-base-v2@24 | 0.864 | 39.4% |
| albert-base-v2@36 | 0.839 | 30.0% |
| albert-base-v2@48 | 0.789 | 40.2% |
| albert-xlarge-v2@12 | 0.892 | 14.9% |
| albert-xlarge-v2@24 | 0.886 | 22.8% |
| albert-xlarge-v2@36 | 0.911 | 15.8% |
| albert-xlarge-v2@48 | 0.902 | 44.5% |
| gpt2 | 0.908 | 44.9% |
| gpt2-medium | 0.942 | 48.3% |
| gpt2-large | 0.966 | 36.3% |
| gpt2-xl | 0.966 | 41.5% |

High mean stability and high "never stable" fraction are not contradictory: a token that flips on every transition contributes to "never stable" while the majority that stay put drive the mean up. GPT-2-large and -xl achieve the highest per-token stability (0.966). ALBERT-base-v2@48 is the lowest (0.789), suggesting iteration depth degrades hemisphere-label coherence in the base model.

### Global verdict

| | Albert | GPT |
|---|---|---|
| Bipartition exists universally | False | False |
| Bipartition identity persistent across layers | True | True |
| HDBSCAN clusters nested in bipartition | — | True |
| Cone-collapse regime absent at long prompts | False | False |
| Paper alignment | cone_collapse | cone_collapse |

**Note on the "Cone-collapse regime absent at long prompts" field:** This field reports `False` when the split regime was NOT observed on long prompts — i.e., cone-collapse held throughout. Combined with 100% cone-collapse in the table, the interpretation is: the paper's regime holds on long prompts as well as short ones. The field name in the raw JSON (`cone_collapse_regime_at_long_prompts`) is ambiguous; the correct reading is "did we observe cone_collapse failing on long prompts?" → `False` = no, we did not.

**ALBERT vs. GPT nesting.** HDBSCAN nesting is `True` for GPT (local density clusters are contained within the Fiedler partition) and absent/not computed for ALBERT. This is a structural difference worth preserving: even with a weak bipartition, GPT's cluster structure respects the Fiedler axis; ALBERT's does not (or the test had insufficient data to resolve it).

-----

## Falsification outcomes

| Criterion | Result |
|---|---|
| Block 0 null: no strong bipartition anywhere | **Fired.** 0% strong bipartition across models. Run continued — other blocks remain informative. |
| Block 1 null: identity not preserved across layers | **Did not fire.** Identity persistent = True for both families. |
| Block 2 null: HDBSCAN nesting near 0.5 uniformly | **Partial.** GPT nesting confirmed. ALBERT inconclusive. |
| Block 3 null: cone-collapse holds at every layer | **Fired.** 100% cone-collapse. Split regime absent. |
| Block 5 null: no consistent axis alignment | **Not run** (no Phase 2 OV artifacts). |

The two most informative falsification criteria both fired. The primary finding is that the geometry is in the paper's regime, not the split regime that Phase 1 speculative narrative anticipated.

-----

## What the Fiedler axis is (revised interpretation)

The Phase 1 $k=2$ eigengap is real in the sense that Block 1 confirms the Fiedler vector is a stable, identity-preserving axis. But it is not a bipartition in the strong geometric sense — all tokens sit within the same open half-space, so there is no antipodal split. The axis reflects an anisotropy in the Gram matrix (some direction of maximum spread), not a cluster boundary. This is consistent with the cone-collapse regime: within a cone, the Fiedler direction is the axis of elongation, not a separator.

The corollary: Phases 4–6 should not treat the Fiedler split as a two-class label for hemisphere membership. The axis itself (as a direction) may still be useful — e.g., projection onto the Fiedler axis as a continuous feature, or its alignment with OV eigenvectors — but the binary partition it induces at 0 is not meaningful given that all tokens are on the same side of the separating hyperplane.

-----

## Code structure

```
p1b_hemisphere/
├── __init__.py
├── bipartition_detect.py        — Block 0: per-layer regime classifier and quality metrics
├── hemisphere_tracking.py        — Block 1: identity matching, events, axis rotation
├── hemisphere_membership.py      — Block 2: per-token trajectories, HDBSCAN nesting
├── cone_collapse.py              — Block 3: cone-collapse LP test and regime classifier
├── size_symmetry.py              — Block 4: asymmetry distribution
├── hemisphere_mechanism.py       — Block 5: axis alignment vs. OV / PCA / embedding / heads (requires Phase 2)
├── hemisphere_semantics.py       — Block 6: token-attribute contingency and MI (requires Phase 2)
└── run_1b.py                     — CLI entry point and aggregation
```

Reused existing modules (imported, not duplicated):

- `fiedler_tracking.py` — Fiedler vector extraction, hemisphere assignments, crossing rate, stability, centroid separation.
- `rotation_hemisphere.py` — displacement coherence within hemispheres; feeds Block 5.
- `cluster_tracking.match_layer_pair` — identity-matching logic; Block 1 reuses it at $k=2$.
- `spectral.spectral_eigengap_k` — returns the full Fiedler vector when requested.

### Entry point

```
python -m p1b_hemisphere.run_1b                              # all models, all prompts
python -m p1b_hemisphere.run_1b --fast                       # albert-base-v2 + wiki_paragraph
python -m p1b_hemisphere.run_1b --models albert-base-v2 gpt2
python -m p1b_hemisphere.run_1b --phase1-dir results/2026-04-23_18-30-06
python -m p1b_hemisphere.run_1b --no-cone                    # skip Block 3 LP
```

Block 5 and 6 are attempted if `--phase1-dir` is supplied and the corresponding OV decomposition artifacts are present; skipped silently otherwise.

### Known issue in aggregated output

The `_write_cross_run_md` function in `run_1b.py` contains hardcoded narrative text in the "Relationship to the paper's hemisphere" section that describes the split regime appearing at mid-depth. This text was written as a placeholder before the run. It is factually incorrect given actual results and should be replaced with dynamically generated text conditioned on `paper_alignment` from the global verdict.

-----

## Outputs

Per (model, prompt):

- `phase1h_{model}_{prompt}.json` — flat per-layer JSON. Schema: `per_layer[L]` with `regime`, `cone_regime`, `bipartition_eigengap`, `centroid_angle`, `within_half_ip`, `hemisphere_sizes`, `cone_margin`; plus `events`, `per_token`, `summary`.
- `phase1h_{model}_{prompt}.md` — prose summary per block.

Cross-run:

- `phase1h_cross_run.json` — aggregated by model and by prompt; global verdict.
- `phase1h_cross_run.md` — one-page synthesis.

-----

## Handoff to later phases

**Phase 4.** The bipartition-as-binary-label approach is not supported by results (Block 0 null). The Fiedler axis as a continuous projection direction remains a candidate feature. Train a one-dimensional linear probe on the Fiedler coordinate (not the sign) and check alignment with the OV eigenvectors once Block 5 runs.

**Phase 5.** The two hemisphere centroids are still well-defined even under cone-collapse (they are the centroids of the two sign-partitions of the Fiedler vector, which is a real axis even if not a separator). They remain usable as candidate cluster-identity vectors, but their interpretation changes: they are not antipodal cluster centers but the two extremes of an elongated cone.

**Phase 6.** The Fiedler axis difference vector passed through the tuned lens is still meaningful as a "what changes between the two extremes of token geometry" measurement. KL divergence between the two centroid distributions per layer remains a valid probe.
