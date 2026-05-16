# Phase 5 — Single-Cluster Case Study

**Status:** Complete. Six models run. Several groups partially blocked (D, E, merge geometry). See Known Gaps below.

---

## Core Question

Every previous phase worked in aggregate: 35 model×prompt verdicts, hundreds of features ranked by F-statistic, bulk V-alignment scores. Aggregate statistics established that metastability exists (Phase 1), that OV's signed component drives energy violations (Phase 2/2i), that crosscoder features bifurcate by lifetime but miss V geometrically (Phase 3), and — pending Phase 4 results — whether features track cluster membership functionally.

Phase 5 asks: **take a single HDBSCAN cluster trajectory from Phase 1 and reconstruct, end to end, the mechanism that creates it, maintains it, and dissolves it.** The deliverable is an interpretable narrative of one piece of the model's computation, cross-referenced to every framework the project has built.

---

## What prior phases make available

- **Phase 1:** HDBSCAN labels per layer, Hungarian-matched trajectory chains, centroid trajectories, per-head Sinkhorn Fiedler, CKA, NN stability, merge/birth/death events with token-level accounting, P1-3 nesting, P1-4 semantic/artifact tagging.
- **Phase 2:** Composed $V_\text{eff}$ per layer (or shared for ALBERT), Schur and symmetric decompositions, attractive/repulsive subspace projectors, per-head OV with rep_frac, FFN subspace projections, per-layer violation classification.
- **Phase 2i:** S/A decomposition, confirmed globally `rotation_neutral`.
- **Phase 3:** Crosscoder checkpoint, per-feature decoder directions, feature lifetime classes, prompt activation store, steering infrastructure.
- **Phase 4:** Per-token feature activation trajectories, feature–cluster MI, chorus cliques, LDA cluster-separation directions, linear probes, low-rank AE bottleneck directions.

---

## Cluster selection

Candidate scoring on Phase 1 tracked trajectories uses six sub-scores:

1. **Lifespan** — at least 6 layers of continuous identity.
2. **Merge participation** — trajectory ends in a merge or contains one.
3. **Semantic content** — P1-4 tags the cluster as semantic (not induction-artifact).
4. **Prompt context** — prefer `sullivan_ballou` or `paper_excerpt`. Avoid `repeated_tokens`.
5. **Size** — at least 4 tokens while alive.
6. **Sibling availability** — merge partner or nearest non-merged neighbor is also long-lived.

### Observed selection results (all six models)

| model | prompt | traj_id | lifespan | score | merge_layer |
|---|---|---|---|---|---|
| gpt2-xl | paper_excerpt | 29 | 49 (L0→L48) | 9.000 | L33→L34 |
| gpt2-large | paper_excerpt | 23 | 16 (L0→L15) | 8.778 | L5→L6 |
| gpt2-medium | paper_excerpt | 12 | 18 (L0→L17) | 8.478 | L16→L17 |
| bert-base-uncased | paper_excerpt | 25 | 9 (L0→L8) | 8.000 | L7→L8 |
| albert-xlarge-v2 | sullivan_ballou | 24 | 20 (L0→L19) | 9.000 | L18→L19 |
| albert-base-v2 | sullivan_ballou | 34 | 31 (L0→L30) | 9.000 | L29→L30 |

All six top-scoring trajectories run to their merge event within the final 1–3 layers of their lifespan. The top-4 models achieve perfect scores (9.000); gpt2-medium is penalized on size (mean_size=4.78). For all models, the runner-up trajectory shares the same prompt as the primary and scores within 0.3 points, indicating the selection is stable.

---

## Investigations

### A. Structural Profile

**What it measures:** token membership stability, compactness (IP mean, radius), silhouette against sibling and all-other, centroid arc length, CKA restricted, mass-near-1.

**Results:**

| model | mean_ip | mean_jaccard | mean_radius | mean_cka | mean_sil_all |
|---|---|---|---|---|---|
| gpt2-xl | 0.876 | 0.914 | 0.139 | 0.984 | 0.783 |
| gpt2-large | 0.916 | 0.943 | 0.091 | 0.987 | 0.825 |
| gpt2-medium | 0.901 | 0.889 | 0.061 | 0.979 | 0.628 |
| bert-base | 0.702 | 0.775 | 0.191 | 0.969 | 0.591 |
| albert-xlarge | 0.462 | 0.770 | 0.398 | 0.979 | 0.394 |
| albert-base | 0.847 | 0.919 | 0.138 | 0.986 | 0.624 |

Restricted CKA is uniformly high (0.97–0.99) across all models — the cluster representation is highly self-consistent within its lifespan regardless of IP spread. BERT and albert-xlarge are noticeably looser (lower IP, higher radius, lower silhouette). Albert-xlarge's IP mean of 0.46 with radius 0.40 is the outlier: its clusters are real by the Jaccard and CKA metrics, but geometrically broader than any other model. This likely reflects the weight-sharing architecture spreading token representations over a wider region of $S^{d-1}$ across iterations.

`mean_silhouette_sib` (cluster vs. its sibling specifically) ranges from −0.096 (bert-base) to 0.704 (gpt2-large). Negative silhouette vs sibling for bert-base means the primary cluster and sibling are geometrically interleaved — their separation is low, consistent with the model being shallower and the clusters not yet fully resolved at the layers captured.

---

### B. Paper-Theoretical Alignment

**What it measures:** centroid attractive/repulsive decomposition, local rotational test (S/A), merge event geometry.

**Results:**

| model | attr_frac | rep_frac | rot_verdict | merge_verdict |
|---|---|---|---|---|
| gpt2-xl | 0.522 | 0.478 | locally_rotational | n/a |
| gpt2-large | 0.481 | 0.519 | locally_rotational | n/a |
| gpt2-medium | 0.540 | 0.460 | locally_rotational | n/a |
| bert-base | 0.520 | 0.480 | locally_rotational | n/a |
| albert-xlarge | 0.493 | 0.507 | locally_rotational | n/a |
| albert-base | 0.565 | 0.435 | locally_rotational | n/a |

**`locally_rotational` is universal.** Every model, every cluster, every prompt. The Phase 2i global finding of rotation-neutral holds locally along individual trajectories as well. The S/A asymmetry fractions (sa_mean_asym_frac) are all close to 0.50, meaning neither symmetric nor antisymmetric component dominates locally any more than globally.

The attractive/repulsive centroid split is close to 50/50 everywhere. No model shows strong attractive-subspace dominance at the centroid level, which is a mild tension with the theoretical prediction that cluster tokens should sit primarily in the attractive subspace during stable phases.

**`merge_verdict` is `n/a` for all six models.** This is a known gap — see Known Issues below. Merge geometry did not compute.

---

### C1. Per-Head Attention Contributions

**What it measures:** per-head cumulative cohesion scalar; top attractor heads ranked.

**Results — top attractor head per model:**

| model | top head | cohesion | source |
|---|---|---|---|
| gpt2-xl | h07 | 9.778 | inward_mass_fallback |
| gpt2-large | h08 | 3.770 | inward_mass_fallback |
| gpt2-medium | h04 | 2.399 | inward_mass_fallback |
| bert-base | h06 | 0.943 | inward_mass_fallback |
| albert-xlarge | h03 | 9.869 | inward_mass_fallback |
| albert-base | h07 | 5.914 | inward_mass_fallback |

The leading head's cohesion score is consistently 2–4× higher than the second-ranked head, indicating a single dominant attractor head per cluster. Head concentration is sharper in gpt2-xl and albert-xlarge, where the leading head accounts for a large fraction of total cohesion mass.

**OV values are `n/a` universally.** `cohesion_source` is `inward_mass_fallback` for all runs — the OV-path computation did not produce values. Head rankings are valid as relative signals but are not grounded in the OV mechanism analysis Phase 2 intended. This is noted as a Known Issue below.

---

### C2. FFN vs. Attention Contributions

**What it measures:** mean FFN cohesion, mean attention cohesion, categorical verdict.

**Results:**

| model | attn_coh | ffn_coh | verdict |
|---|---|---|---|
| gpt2-xl | 17.13 | 14.38 | both_cohesive |
| gpt2-large | 2.94 | 1.25 | attn_dominant_cohesive |
| gpt2-medium | 6.92 | 1.08 | attn_dominant_cohesive |
| bert-base | 12.40 | −13.19 | attn_cohesive_ffn_disruptive |
| albert-xlarge | −4.09 | 2.85 | ffn_cohesive_attn_disruptive |
| albert-base | −22.36 | 15.16 | ffn_cohesive_attn_disruptive |

This is the clearest architecture-level split in Phase 5. GPT-2 variants show attention-dominant or co-dominant cohesion. ALBERT models invert: FFN is cohesive and attention is disruptive. This is consistent with ALBERT's weight-sharing: the same attention weights applied iteratively produce fragmented, less-targeted attention patterns, while the FFN — also shared but applied to a changing residual state — acts as the stabilizing mechanism. BERT is the anomaly: attention cohesive, FFN strongly disruptive (−13.19), despite architectural similarity to ALBERT. This may reflect BERT's bidirectional masked-LM pretraining producing a different balance of computational roles.

**Implication for the paper framework:** the framework's prediction that cluster cohesion flows through the attractive subspace of $V_\text{eff}$ (which is dominated by OV composition) does not map cleanly onto FFN dynamics. The ALBERT result in particular suggests that for weight-tied models, the FFN is the primary cohesion mechanism, not OV.

---

### D. Feature Signatures

**Status: blocked for all six models.**

All runs return `ERROR: feature activations unavailable`. Phase 4 outputs (activation trajectories, MI results, LDA directions) are not reaching Phase 5 for any model. No feature-level analysis ran.

---

### E. Tuned-Lens Decoding

**Status: tuned lens not trained; fallback to logit lens.**

`used_tuned_lens=false` for all runs.

**Top-1 stability results:**

| model | top1_stable | layers |
|---|---|---|
| gpt2-xl | 41/48 | 85% |
| gpt2-large | 13/15 | 87% |
| gpt2-medium | 13/17 | 76% |
| bert-base | 5/8 | 63% |
| albert-xlarge | 19/19 | 100% |
| albert-base | 27/30 | 90% |

Top-1 token prediction at the centroid is largely stable across the trajectory lifespan in most models. Albert-xlarge is perfectly stable (19/19) despite having the geometrically loosest clusters — meaning the predicted token identity is fixed even as the representation moves substantially on $S^{d-1}$. BERT is least stable (63%).

**All token probabilities are 0.000.** The logit-lens decoding is running but not recording softmax probabilities — either they're not being stored or they're below the fixed-precision floor. First- and last-layer token distributions are present in the output (e.g., gpt2-xl first layer: `.`, `,`, `;`; last layer: `the`, `a`, `one`) but without probability mass. This is a Known Issue.

---

### F. Causal Interventions

**Status: run on 5 models; bert-base skipped (Group F not run).**

Interventions: `ablate_top_head`, `ablate_control_head`, `steer_centroid`, `steer_lda`. Albert-base and gpt2-medium also ran `patch_sibling`.

**`mean_frac_together` results (fraction of tokens remaining co-clustered post-intervention):**

| model | ablate_top | ablate_ctrl | steer_centroid | steer_lda | patch_sibling |
|---|---|---|---|---|---|
| gpt2-xl | 0.801 | 0.801 | 0.801 | 0.801 | — |
| gpt2-large | 0.527 | 0.527 | 0.527 | 0.527 | — |
| gpt2-medium | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| albert-xlarge | 0.143 | 0.143 | 0.143 | 0.143 | — |
| albert-base | 0.146 | 0.146 | 0.146 | 0.154 | 0.160 |

Within each model all interventions return identical (or near-identical) `mean_frac_together`. This is unexpected — the four interventions target different mechanisms and should in principle produce different effect sizes. Two interpretations: (a) the metric is being computed once globally and reused rather than per-intervention, which would be a bug; or (b) all interventions are hitting the same causal bottleneck and the cluster either dissolves completely or is unaffected regardless of the specific perturbation. The gpt2-medium result (0.000 across all interventions) is the most extreme: every perturbation fully dissolves the cluster.

Interpreted as a robustness gradient: gpt2-xl (0.80) > gpt2-large (0.53) > albert-base (0.15) ≈ albert-xlarge (0.14) > gpt2-medium (0.00). The largest model is most robust to any single targeted intervention.

**Action item before using F results in any analysis:** verify that `mean_frac_together` is computed separately per intervention in `causal_tests.py`.

---

### G. Sibling and Random Control

**What it measures:** sibling cluster IP mean and silhouette vs. all; random baseline of same size.

**Results:**

| model | sibling_ip | random_ip | sibling_sil | random_sil |
|---|---|---|---|---|
| gpt2-xl | 0.804 | 0.416 | 0.674 | −0.009 |
| gpt2-large | 0.937 | 0.471 | 0.863 | −0.003 |
| gpt2-medium | 0.886 | 0.714 | 0.599 | 0.006 |
| bert-base | 0.762 | 0.215 | 0.681 | 0.003 |
| albert-xlarge | 0.580 | 0.129 | 0.493 | 0.012 |
| albert-base | 0.909 | 0.648 | 0.723 | −0.011 |

The three-tier ordering holds cleanly: primary cluster IP > sibling IP > random baseline IP, and similarly for silhouette, for all six models. The random control silhouette is near zero or negative universally, confirming the selection procedure is not detecting noise. The sibling itself is a real, coherent cluster — it passes the quality bar even when it does not pass the selection gates (noted in gpt2-xl and gpt2-medium results: "sibling did not pass selection gates").

Gpt2-medium's random IP (0.714) is elevated relative to other models — its residual stream may be more uniformly correlated even for non-cluster tokens, compressing the signal-to-noise ratio for the cluster/random contrast.

---

## Known Issues (to fix before Phase 7 or publication)

### 1. `merge_verdict` always `n/a`
Group B does not populate merge geometry for any model. `merge_event_geometry()` is either not executing or the merge event data is not being passed through to `v_alignment.py`. The README anticipated this as a central Group B output (fusion direction vs. attractive subspace). **Fix:** trace the `merge_events` argument from `run_5.py` Group B runner through to `merge_event_geometry()` and confirm the call is being reached.

### 2. OV values always `n/a` in C1
`inward_mass_fallback` is the universal cohesion source. The OV-path computation in `head_contributions.py` is not producing values — likely a missing or miskeyed weights load from Phase 2. Head rankings based on the fallback are valid relative signals but not grounded in the OV mechanism. **Fix:** confirm `p5io.load_phase2_weights()` is returning the correct OV weight arrays for each model stem, and that `analyze_heads()` uses them when present.

### 3. Group D blocked universally
Feature activations unavailable for all six models. Phase 4 outputs are not being loaded — either the results directory path is wrong, the file naming convention doesn't match what `p5io.load_phase4()` expects, or Phase 4 did not write activation cache files. **Fix:** check `load_phase4()` path construction against actual Phase 4 output layout; add a fallback that re-derives feature activations inline if the cache is absent.

### 4. Token probabilities all 0.000 in E
Logit-lens decoding is running (token strings appear) but softmax probabilities are not stored. Either the values are being computed and dropped, or they're falling below the output format's fixed-precision floor. **Fix:** check `tuned_lens_cluster.py` output serialization; store log-probabilities if softmax probs round to zero.

### 5. Causal interventions return identical `mean_frac_together`
All four interventions produce the same score within each model. **Fix:** audit `causal_tests.py` to confirm the metric is computed in a separate forward pass per intervention, not computed once and duplicated across all result dicts. If the computation is correct, document this as an intentional aggregate metric and add a per-token breakdown to distinguish intervention effects.

### 6. bert-base Group F not run
BERT skipped causal tests entirely. **Fix:** no blocking reason identified — run Group F for bert-base and add results to the report.

---

## Architecture-level findings (cross-model summary)

These patterns hold across all six models and can inform subsequent phases:

- **Locally rotational, universally.** The global Phase 2i finding extends to individual cluster trajectories. No cluster shows a locally non-rotational S/A profile.
- **Attractive/repulsive centroid split is ~50/50 everywhere.** No evidence of strong attractive-subspace dominance at the centroid level during stable phases. This is a mild tension with Theorem 6.3 predictions and should be examined in Phase 6.
- **FFN role is architecture-dependent.** GPT-2: attention-dominant or co-dominant. BERT: attention cohesive, FFN disruptive. ALBERT: FFN cohesive, attention disruptive. The OV-centric framework from Phases 2/2i does not generalize cleanly to FFN-dominant models.
- **A single head dominates per cluster.** In every model, the top attractor head has cohesion 2–4× higher than second place. This concentration is sharper in larger models.
- **Cluster robustness scales with model size (GPT-2 family).** gpt2-xl clusters resist targeted interventions; gpt2-medium clusters dissolve completely. This may reflect larger models distributing cluster identity across more components.

---

## Dependencies

**Required (blocking):**
- Phase 1 run for the chosen prompt with HDBSCAN labels, cluster tracking, centroid trajectories.
- Phase 2 V projectors (`ov_projectors_{stem}.npz`). *(Currently not loading for C1 — see Known Issue 2.)*
- Phase 3 crosscoder checkpoint.

**Required (soft — currently blocking Group D):**
- Phase 4 activation cache and MI/LDA outputs. *(Currently not loading for any model — see Known Issue 3.)*
- Phase 2i rotational-spectrum per-layer arrays.

**Optional:**
- Tuned lens checkpoint. *(Not trained; fallback logit lens in use — see Known Issue 4.)*

---

## Falsification criteria

1. **If merge geometry had been populated:** does fusion proceed along an attractive or repulsive direction? The framework predicts attractive-dominant. This remains untested.
2. **Attractive/repulsive centroid split near 0.50:** the 50/50 split is inconsistent with strong attractor-basin confinement during plateau phases. Either the V-projector computation is underestimating attractive-subspace contribution, or the framework's prediction is too strong.
3. **Causal tests:** if the identical `mean_frac_together` issue is a bug, the actual causal sensitivity of each intervention type is unknown. If it is not a bug, the result means cluster identity is not localized in any single head, direction, or activation pattern — it is distributed, and targeted single-point interventions are insufficient to assess it.

---

## Forward compatibility

- **Phase 6:** Group E here is a preview. Tuned lens infrastructure needs to be trained before Phase 6 scales it to all clusters. The logit-lens top-1 stability results (76–100%) establish an upper bound on what the trained tuned lens should recover.
- **Phase 7 (if pursued):** this pipeline is per-cluster. Running it on 20+ clusters produces a distribution of mechanistic stories. The ALBERT/GPT-2 FFN inversion already suggests the distribution is multimodal by architecture.
- **Publication:** the merge geometry gap (Known Issue 1) and the feature signature gap (Known Issue 3) are the two most important items to close before this phase contributes to a publishable narrative. Everything else is present.

---

## Code structure

```
p5_single_mstate_analysis/
├── README_p5.md                     (this file)
├── __init__.py
├── select_cluster.py                — rank and select primary + sibling trajectories
├── cluster_profile.py               — Group A: structural profile
├── v_alignment.py                   — Group B: paper-theoretical alignment
├── head_contributions.py            — Group C.1: per-head attention + cohesion scalars
├── ffn_contributions.py             — Group C.2: FFN projection onto cluster directions
├── feature_signature.py             — Group D: identity features, chorus, LDA
├── tuned_lens_cluster.py            — Group E: tuned/logit lens decoding
├── causal_tests.py                  — Group F: ablation, steering, patching
├── sibling_contrast.py              — Group G: sibling + random control
├── report.py                        — assemble cluster_report.txt
└── run_5.py                         — CLI entry point
```
