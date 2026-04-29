# Phase 1 — Empirical Verification of Metastability

**Status:** Complete. Extensions (cluster tracking, multi-scale nesting, pair agreement filtering, dense ALBERT sweep, V eigenspectrum extraction, sublayer stream analysis) implemented. Run across 7 architectures × 4 prompts, with ALBERT models run at 4 iteration depths (12, 24, 36, 48). `bert-large-uncased` and `albert-base-v2-random` are in config but excluded from the standard run; use `--models bert-large-uncased` or `--random-baseline` to include them.

---

## Core Question

Geshkovski et al. (*A Mathematical Perspective on Transformers*) prove that transformer token representations, modeled as interacting particles on $\mathbb{S}^{d-1}$, converge to a single cluster in the long-time limit. Before convergence, the dynamics pass through **metastable states** — multi-cluster configurations that persist across many consecutive layers before abruptly merging. The paper establishes this for a simplified model where $Q^\top K = V = I_d$. Phase 1 asks: **does this metastability survive in trained architectures with learned weight matrices, multi-head attention, and feed-forward layers?**

The falsification criterion: if no plateaus appear in cluster count or inner-product histograms across consecutive layers, metastability does not survive trained dynamics at this scale, and the project stops.

Phase 1 passed.

---

## Theoretical Predictions Tested

Seven predictions from the paper, checked against empirical trajectories:

1. **Tokens cluster over layers** — pairwise inner products $\langle x_i, x_j \rangle$ drift toward 1. ✓ Universal across all models and prompts.
2. **Two-timescale dynamics** — fast initial grouping, slow pairwise merging. ✓ Confirmed for BERT-base (ratio=8.0), GPT-2-large (ratio=8.2), GPT-2-xl (ratio=7.62). ✗ Not cleanly separated in ALBERT (metastable window ≤ collapse onset) or GPT-2-medium (ratio=0.45). GPT-2-small: degenerate initial distribution never collapses at all in the control (mass never reaches 0.9 in the repeated-tokens run).
3. **Metastable states** appear as plateaus in cluster count metrics. ✓ Universal.
4. **ALBERT** (shared weights) should show cleaner dynamics than BERT/GPT-2. ✓ Partially — ALBERT-base is cleanest and collapses fully (MaxMass=1.0 by 24 iterations for long prompts). ALBERT-xlarge resists collapse in a way the theory does not predict: MaxMass stays below 0.30 across all prompts even at 48 iterations, and MinRank stays above 55 for long prompts. The cross-run report flags this directly: "ALBERT reaches higher mass>0.9 than other models. Consistent with theory: shared weights = cleaner dynamics." This applies to ALBERT-base, not xlarge.
5. **Higher $\beta$** (sharper attention) → stronger metastability. ✓ Confirmed — violations and plateau counts increase with $\beta$.
6. **Higher dimension $d$** → faster convergence to single cluster (Theorem 6.1). ✗ Falsified. ALBERT-xlarge ($d=2048$) converges far slower than ALBERT-base ($d=768$): ALBERT-base collapses at 24 iterations, ALBERT-xlarge has not collapsed at 48. The governing variable is the spectral radius of $V$, not $d$.
7. **Interaction energy $E_\beta$** is monotone increasing along the trajectory (Theorem 3.4). ✗ Universally violated. Every model, every prompt, every $\beta$: EnergyOK = NO. Mechanism identified in Phase 2: $V$'s mixed-sign eigenspectrum introduces rotational dynamics incompatible with monotone energy flow.

---

## Key Findings

### Two-timescale separation is architecture-specific, with a depth threshold

Two-timescale separation (quantified as the ratio of mean plateau width to collapse onset layer in the repeated-tokens control) is confirmed above a depth threshold between GPT-2-medium (24 layers, ratio=0.45, no separation) and GPT-2-large (36 layers, ratio=8.2, confirmed). The full table:

| Model | Layers | Ratio | Verdict |
|---|---|---|---|
| BERT-base | 12 | 8.0 | TWO-TIMESCALE CONFIRMED |
| GPT-2-large | 36 | 8.2 | TWO-TIMESCALE CONFIRMED |
| GPT-2-xl | 48 | 7.62 | TWO-TIMESCALE CONFIRMED |
| GPT-2-medium | 24 | 0.45 | NO SEPARATION |
| ALBERT-base@36iter | 37 | 1.06 | WEAK SEPARATION |
| ALBERT-base@48iter | 49 | 1.25 | WEAK SEPARATION |
| ALBERT-base@12iter | 13 | 0.62 | NO SEPARATION |
| ALBERT-xlarge@36+iter | 37/49 | 0.25 | NO SEPARATION |
| GPT-2-small | 12 | — | No collapse in control |

GPT-2-small is anomalous: it never collapses a degenerate repeated-token input at all, yet regular prompts reach MaxMass of 0.87–0.97. This indicates its clustering is content/position-driven rather than a strong geometric attractor.

### GPT-2 attention heads are universally content-independent

Cross-prompt per-head Fiedler consistency shows a sharp architecture split. All GPT-2 family models (gpt2 through gpt2-xl) have 100% STABLE-CLUSTER heads across all four prompts: no head changes its cluster/mixing behavior with input content. BERT-base is the opposite — 11/12 heads are VARIABLE (change behavior by prompt). ALBERT-base transitions from mostly STABLE-CLUSTER at 12 iterations to all STABLE-MIXED at 36+ iterations. ALBERT-xlarge has many VARIABLE heads throughout.

The implication: GPT-2's routing structure is entirely fixed at training; the cluster geometry is a weight-level property. BERT's routing is content-driven. ALBERT's routing evolves with iteration depth — the shared weights effectively accumulate a content-adaptive transformation.

### Merge events are weight-level; plateau onset is mostly content-driven

Spectral k merge events occur at the same layers across all prompts for a given model (e.g., GPT-2-large always merges at layer 35, GPT-2-xl at layers 37 or 47 depending on prompt). The merge schedule is a property of the weights, not the input. Plateau onset, by contrast, is content-driven for most models (standard deviation across prompts: 5–9 layers for ALBERT-base, ALBERT-xlarge, GPT-2-xl, GPT-2-large). Exceptions: BERT-base (SD=0, weight-level) and GPT-2-medium (SD=1.7, effectively weight-level).

### ALBERT-xlarge resists collapse; spectral k is non-informative for it

ALBERT-xlarge MaxMass stays below 0.30 across all prompts at all tested iteration depths. MinRank stays above 55 for long prompts. The spectral k metric records zero merge events (nMerges=0) for all ALBERT-xlarge runs due to the zero-mode dominating the primary eigengap at $d=2048$. This is misleading: the cluster tracking section (P1-1) records 47–139 genuine HDBSCAN merge events per run. The two numbers measure different things and are both correct; the spectral k limitation is the known limitation, not a contradiction.

Energy violation count grows with iteration depth for ALBERT-xlarge (up to 17–18 violations at 48 iterations, beta=0.1–5.0, for short_heterogeneous), while total_delta stays bounded. This is the opposite of ALBERT-base, where total_delta grows sharply with depth (reaching 10.8 at 48iter, beta=5.0 for short_heterogeneous).

### Multi-scale nesting is sparse

Spectral eigengap within HDBSCAN clusters (P1-3) detects nesting at only layers 0–3 for ALBERT variants and layers 0–1 for BERT. No hierarchical layering is detected across deep layers. The clusters found at plateau windows are flat partitions, not hierarchies.

### Pair HDBSCAN agreement: artifact fraction is universally low

Mutual-NN cycle pairs tagged as artifacts (tokens in different clusters that are nearest neighbors) stay below 3% across all models and prompts. Semantic pairs (tokens in the same cluster) increase with prompt length (longer context → cleaner clusters) and with ALBERT iteration depth (42% at 12iter → 74% at 48iter for ALBERT-base + wiki_paragraph). ALBERT-xlarge's semantic fraction is consistently lower than ALBERT-base's (56% vs 74% for long prompts at 48 iterations), consistent with its lower MaxMass and more diffuse clustering.

### Token clusters carry semantic content; deeper models add syntactic structure

Mutual-NN cycle analysis at plateau layers recovers interpretable structure: `novelist ↔ poet`, `lancashire ↔ brussels`, `school ↔ lo` (Lowood). The clusters are not arbitrary geometric artifacts. They track co-reference and semantic similarity. Larger GPT-2 models and BERT add positional and syntactic groupings that smaller models do not show clearly.

### The transition between GPT-2 regimes is abrupt

Mean spectral radius drops from 3.21 (GPT-2-medium, 24 layers) to 1.38 (GPT-2-large, 36 layers), a 2.3× drop with no intermediate values. This coincides with the appearance of two-timescale separation. The governing variable is something learned during training rather than a direct function of depth: BERT-base (12 layers, mean spectral radius ≈0.94) clusters with the large regime despite matching GPT-2-small's depth.

---

## Known Limitations

1. **ALBERT-xlarge extended runs.** The spectral radius of V (1.278) predicts collapse at ~70–80 iterations: $\log(0.9)/\log(1.278) \approx 75$ passes needed. ALBERT-xlarge at 12–48 iterations is now run and confirms slow dynamics. Runs at 96–128 iterations are needed to determine whether collapse is merely delayed or structurally prevented by the complex eigenvalue structure.
2. **Spectral $k$ non-informative for ALBERT-xlarge.** Confirmed across all prompts and all iteration depths: spectral k = 1 everywhere, nMerges = 0 in summary table. The zero mode dominates. Genuine merges are recorded by HDBSCAN cluster tracking (P1-1) and are the correct measure for xlarge.
3. **Depth-conditioning regime shift unexplained.** The jump in mean spectral radius from GPT-2-medium to GPT-2-large is abrupt and unexplained by depth, dimension, or training objective alone. The governing factor — whether training data, learning rate schedule, parameter budget allocation, or something else — is unknown.
4. **BERT-large not in standard run.** The model is in config (`bert-large-uncased`) but was excluded from this run. Use `--models bert-large-uncased` to include it.
5. **Random-init ALBERT baseline not in standard run.** Available via `--random-baseline`. Tests whether metastability is a property of trained weights or just of the iterated-map architecture.
6. **Final-layer LM-head contamination.** GPT-2-small and GPT-2-medium final layers show extreme collapse that is the unembedding projection, not dynamics. These layers are flagged but not removed from trajectory plots.
7. **Induction head confound.** Mutual-NN pairs include subword completions that may be attention artifacts. Per-pair HDBSCAN agreement (P1-4) tags but does not remove them. Artifact fraction stays below 3%, suggesting the confound is small but present.
8. **GPT-2-small repeated-tokens anomaly.** Mass never reaches 0.9 in the control, making the two-timescale ratio undefined. This is not a code failure — regular prompts do reach high mass. The model's clustering appears to require semantic/positional diversity in the input.

---

## Modules

### `run_1.py` — CLI entry point
Orchestrates the full pipeline. Key arguments:
- `--models`, `--prompts` — subset selection
- `--fast` — albert-base-v2 + wiki_paragraph only, legacy snapshots
- `--no-extended` — disable ALBERT extended-iteration mode
- `--legacy-snapshots` — use [12,24,36,48] instead of the dense sweep
- `--random-baseline` — add albert-base-v2-random (untrained control)
- `--sublayer` — additionally run post-attention and post-FFN sublayer streams, saved as `{model}@attn` / `{model}@ffn` run directories (supplementary, excluded from cross-run comparison)
- `--length-sweep` — run wiki_paragraph truncated at each LENGTH_SWEEP_TOKENS target
- `--replot RUN_DIR`, `--summary RUN_DIR` — replot or summarize a saved run

ALBERT extended mode runs a single forward pass to `ALBERT_MAX_ITERATIONS` and slices the trajectory at each snapshot, saving one result directory per (prompt, depth) pair. Sublayer streams use forward hooks on `attn.c_proj` / `mlp` (GPT-2) or `attention.output` / `output` (BERT/ALBERT) to capture intermediate residual streams.

### `analysis.py` — Layer-wise analysis loop
Ingests hidden states and attentions, calls every metric/clustering/projection function, collects results into a single dict. Pre-computes normed activations and Gram matrix once per layer. Post-loop: cluster tracking (P1-1), plateau layer identification.

### `metrics.py` — Core per-layer scalar metrics
Pairwise inner products, interaction energies (batched over β), effective rank, attention entropy, nearest-neighbor indices and stability, linear CKA, energy drop pair localization.

### `clustering.py` — Clustering algorithms and projections
Agglomerative threshold sweep, KMeans silhouette, HDBSCAN, PCA, UMAP. Multi-scale nesting (P1-3): spectral eigengap within each HDBSCAN cluster. Per-pair HDBSCAN agreement (P1-4): mutual-NN cycles tagged as semantic vs artifact. Accepts pre-normed arrays.

### `cluster_tracking.py` — HDBSCAN trajectory accounting (P1-1)
Tracks token cluster membership across layers. Records births, deaths, and merge events at the token level. Reports trajectories, mean/max lifespan, and the full merge event sequence. This is the correct merge-detection method for ALBERT-xlarge (where spectral k is non-informative).

### `spectral.py` — Spectral eigengap clustering
Computes spectral cluster count k from the attention graph Fiedler value and eigengap structure. Per-head Fiedler values used in the cross-prompt consistency table.

### `sinkhorn.py` — Attention Sinkhorn analysis
Doubly-stochastic normalization of attention matrices, Fiedler value computation, per-head cluster/mixing classification.

### `metrics.py` → `energy_drop_pairs` — Energy violation localization
For each β, identifies token pairs responsible for non-monotone energy steps. Used in Phase 2 cross-referencing.

### `plots.py` — All visualization
Trajectory plots, IP histograms, PCA panels, Sinkhorn detail, spectral eigengap, eigenvalue spectra, ALBERT extended comparison, cross-model comparison, CKA trajectory. Also `analyze_value_eigenspectrum` (extracts V eigenvalues, feeds Phase 2).

### `reporting.py` — Text reports
`generate_llm_report`: single-run plain-text report with per-layer data table, trend descriptions, plateau locations, merge events, method agreement, energy trajectory, PCA variance, inner-product histogram summary.  
`generate_cross_run_report`: comparative report across all (model, prompt) combinations. Sections: summary table, plateau locations, prompt sensitivity (SD of onset), merge events, energy monotonicity, flagged cross-run patterns, per-head Fiedler consistency, cluster tracking summary (P1-1), multi-scale nesting (P1-3), pair HDBSCAN agreement (P1-4), collapse control runs.

### `io_utils.py` — Serialization
`save_run`: writes v2 split format — `geometry.json`, `energies.json`, `clustering.json`, `spectral.json`, `activations.npz`, `attentions.npz`, `clusters.npz`, `centroid_trajectories.npz`. Each JSON is <100KB. `load_run` is backward-compatible with v1 `metrics.json`. `replot_all`: regenerate plots from saved data without model reload.

### `core/config.py` — Global constants
Model registry (9 entries including `bert-large-uncased` and `albert-base-v2-random`), prompt variants (5: `short_heterogeneous`, `wiki_paragraph`, `sullivan_ballou`, `paper_excerpt`, `repeated_tokens`), β values, distance thresholds, Sinkhorn tolerances, ALBERT iteration sweep parameters. Device selection.

### `core/models.py` — Model loading and extraction
Standard forward pass extraction for BERT/GPT-2. ALBERT extended-iteration extraction (single pass to max depth, sliced at snapshot points). bfloat16 on CUDA, `torch.compile` when available. `layernorm_to_sphere` projects activations to $\mathbb{S}^{d-1}$ before metric computation.

---

## Output Format (v2)

Per-run directory `{model}_{prompt}/`:
```
geometry.json          — per-layer ip_mean, ip_std, effective_rank, nn_stability, cka_prev
energies.json          — per-layer energies (all β), violations, drop pairs
clustering.json        — per-layer spectral_k, hdbscan_k, mass>0.9, nesting, pair_agreement
spectral.json          — per-layer Fiedler, sinkhorn_k, attention_entropy
activations.npz        — (n_layers, n_tokens, d_model) float32, L2-normed
attentions.npz         — (n_layers, n_heads, n_tokens, n_tokens) float32
clusters.npz           — hdbscan_labels_L{i} per layer
centroid_trajectories.npz
llm_report.txt
```

Session-level: `llm_cross_run_report.txt`, `experiment.txt`.

---

## Transition to Phase 2

Phase 1 establishes that metastability exists and that the energy functional is not monotone. Phase 2 asks why.

The paper's framework attributes metastability to the tension between attractive dynamics (softmax attention pulls tokens together) and repulsive dynamics (mixed-sign eigenspectrum of V pushes them apart). Phase 1 measures the outcome of this tension. Phase 2 measures the tension itself.

Everything Phase 2 needs from Phase 1 is saved: activations at every layer as `.npz`, plateau layer windows, merge event indices, energy violation layers, energy drop token pairs, and token lists.
