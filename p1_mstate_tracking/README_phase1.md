# Phase 1 — Empirical Verification of Metastability

**Status:** Complete. Extensions (cluster tracking, multi-scale nesting, pair agreement filtering, dense ALBERT sweep, V eigenspectrum extraction, sublayer stream analysis, 10-seed random baseline sweep) implemented. Run across 7 architectures × 8 prompts, with ALBERT models run at 4 iteration depths (12, 24, 36, 48). `bert-large-uncased` is in config but excluded from the standard run; use `--models bert-large-uncased` to include it. `albert-base-v2-random` is now included in the standard run; the 10-seed sweep is complete.

---

## Core Question

Geshkovski et al. (*A Mathematical Perspective on Transformers*) prove that transformer token representations, modeled as interacting particles on $\mathbb{S}^{d-1}$, converge to a single cluster in the long-time limit. Before convergence, the dynamics pass through **metastable states** — multi-cluster configurations that persist across many consecutive layers before abruptly merging. The paper establishes this for a simplified model where $Q^\top K = V = I_d$. Phase 1 asks: **does this metastability survive in trained architectures with learned weight matrices, multi-head attention, and feed-forward layers?**

The falsification criterion: if no plateaus appear in cluster count or inner-product histograms across consecutive layers, metastability does not survive trained dynamics at this scale, and the project stops.

Phase 1 passed.

---

## Theoretical Predictions Tested

Seven predictions from the paper, checked against empirical trajectories:

1. **Tokens cluster over layers** — pairwise inner products $\langle x_i, x_j \rangle$ drift toward 1. ✓ Universal across all models and prompts.
2. **Two-timescale dynamics** — fast initial grouping, slow pairwise merging. ✓ Confirmed for BERT-base (ratio=8.0), GPT-2-large (ratio=8.2), GPT-2-xl (ratio=7.62). ✗ Not cleanly separated in ALBERT-base with trained weights (metastable window ≤ collapse onset) or GPT-2-medium (ratio=0.45). Under random weights, ALBERT-base shows confirmed two-timescale separation (ratio growing from 2.8 at 24iter to 13.5 at 48iter), with the trained-weight contrast still unquantified. GPT-2-small: degenerate initial distribution never collapses in the control (mass never reaches 0.9 in the repeated-tokens run).
3. **Metastable states** appear as plateaus in cluster count metrics. ✓ Universal.
4. **ALBERT** (shared weights) should show cleaner dynamics than BERT/GPT-2. ✓ Partially — ALBERT-base is cleanest and collapses fully (MaxMass=1.0 by 24 iterations for long prompts; MaxMass=1.0000 in every seed of the random baseline). ALBERT-xlarge resists collapse in a way the theory does not predict: MaxMass stays below 0.30 across all prompts even at 48 iterations, and MinRank stays above 55 for long prompts.
5. **Higher $\beta$** (sharper attention) → stronger metastability. ✓ Confirmed — violations and plateau counts increase with $\beta$. Energy violation counts at high $\beta$ are seed-sensitive in the random baseline but the directional trend is structural.
6. **Higher dimension $d$** → faster convergence to single cluster (Theorem 6.1). ✗ Falsified. ALBERT-xlarge ($d=2048$) converges far slower than ALBERT-base ($d=768$): ALBERT-base collapses at 24 iterations, ALBERT-xlarge has not collapsed at 48. The governing variable is the spectral radius of $V$, not $d$.
7. **Interaction energy $E_\beta$** is monotone increasing along the trajectory (Theorem 3.4). ✗ Universally violated. Every model, every prompt, every $\beta$, including random-weight runs. Mechanism identified in Phase 2: $V$'s mixed-sign eigenspectrum introduces rotational dynamics incompatible with monotone energy flow.

---

## Key Findings

### Metastability is architecture-determined, not weight-determined

The 10-seed random-initialization sweep of ALBERT-base establishes this cleanly. Every architecture-level quantity is perfectly invariant across seeds: MaxMass=1.0000 in every seed, every prompt, every iteration depth; per-head Fiedler classification is entirely stable (all MIXING, mean ~0.97); plateau onset SD stays in the 0–1.3 band (weight-level). The two-timescale ratio holds in every run and grows monotonically with iteration depth (2.8 at 24iter → 13.5 at 48iter). These are properties of the iterated-map architecture, not of any particular weight draw.

Quantities that are seed-sensitive and should be treated as draws from a distribution rather than reproducible facts: HDBSCAN cluster counts, merge-event layer locations, energy violation counts at high $\beta$.

The interpretability signal (ext_sem_frac) declines monotonically with depth under random weights (~0.99 at 12iter → ~0.30 at 48iter). High clustering coincides with low external-semantic agreement under random weights, consistent with the collapse being geometry-driven rather than semantics-driven. Whether trained weights produce a higher floor or different slope on this metric requires the trained runs at matched iteration depths — that comparison is not yet quantified.

### Two-timescale separation is architecture-specific, with a depth threshold

Two-timescale separation (ratio of mean plateau width to collapse onset layer in the repeated-tokens control) is confirmed above a depth threshold between GPT-2-medium (24 layers, ratio=0.45) and GPT-2-large (36 layers, ratio=8.2). The full table:

| Model | Layers | Ratio | Verdict |
|---|---|---|---|
| BERT-base | 12 | 8.0 | TWO-TIMESCALE CONFIRMED |
| GPT-2-large | 36 | 8.2 | TWO-TIMESCALE CONFIRMED |
| GPT-2-xl | 48 | 7.62 | TWO-TIMESCALE CONFIRMED |
| ALBERT-base-random@24iter | 25 | 2.78 | TWO-TIMESCALE CONFIRMED |
| ALBERT-base-random@36iter | 37 | 9.89 | TWO-TIMESCALE CONFIRMED |
| ALBERT-base-random@48iter | 49 | 13.44 | TWO-TIMESCALE CONFIRMED |
| GPT-2-medium | 24 | 0.45 | NO SEPARATION |
| ALBERT-base@36iter (trained) | 37 | 1.06 | WEAK SEPARATION |
| ALBERT-base@48iter (trained) | 49 | 1.25 | WEAK SEPARATION |
| ALBERT-base@12iter (trained) | 13 | 0.62 | NO SEPARATION |
| ALBERT-xlarge@36+iter | 37/49 | — | No collapse in control |
| GPT-2-small | 12 | — | No collapse in control |

GPT-2-small is anomalous: it never collapses a degenerate repeated-token input at all, yet regular prompts reach MaxMass of 0.87–0.97. Its clustering is content/position-driven rather than a strong geometric attractor.

Notably, ALBERT-base under random weights shows cleaner two-timescale separation (ratio up to 13.44 at 48iter) than under trained weights (ratio ≤ 1.25). This suggests training may alter the dynamics relative to the purely architectural baseline, though the direction and mechanism of this difference remain uncharacterized.

### GPT-2 attention heads are universally content-independent

Cross-prompt per-head Fiedler consistency shows a sharp architecture split. All GPT-2 family models (gpt2 through gpt2-xl) have 100% STABLE-CLUSTER heads across all eight prompts: no head changes its cluster/mixing behavior with input content. BERT-base is the opposite — 11/12 heads are VARIABLE. ALBERT-base transitions from mostly STABLE-CLUSTER at 12 iterations to all STABLE-MIXED at 36+ iterations, identically under trained and random weights. ALBERT-xlarge has many VARIABLE heads throughout.

GPT-2's routing structure is entirely fixed at training; the cluster geometry is a weight-level property. BERT's routing is content-driven. ALBERT's routing evolves with iteration depth — the shared weights accumulate a content-adaptive transformation, and this pattern is present even without training.

### Merge events are weight-level; plateau onset is mostly content-driven

Spectral k merge events occur at the same layers across all prompts for a given model (e.g., GPT-2-large always merges at layer 35; GPT-2-xl at layers 37 or 47 depending on prompt). The merge schedule is a property of the weights, not the input. Plateau onset is content-driven for most models (SD across prompts: 5–13 layers for ALBERT-base, ALBERT-xlarge, GPT-2-xl, GPT-2-large). Exceptions: BERT-base (SD=0.0, weight-level), GPT-2-small (SD=1.89, effectively weight-level), and ALBERT-base at 12 iterations (SD=1.70, weight-level — too shallow for content sensitivity to emerge).

### The transition between GPT-2 regimes is abrupt

Mean spectral radius drops from 3.21 (GPT-2-medium, 24 layers) to 1.38 (GPT-2-large, 36 layers), a 2.3× drop with no intermediate values. This coincides with the appearance of two-timescale separation. The governing variable is something learned during training rather than a direct function of depth or dimension: BERT-base (12 layers, mean spectral radius ≈0.94) clusters with the large regime despite matching GPT-2-small's depth.

### ALBERT-xlarge resists collapse; spectral k is non-informative for it

ALBERT-xlarge MaxMass stays below 0.30 across all prompts at all tested iteration depths. MinRank stays above 55 for long prompts. The spectral k metric records zero merge events (nMerges=0) for all ALBERT-xlarge runs — the zero-mode dominates the primary eigengap at $d=2048$. This is misleading: the cluster tracking (P1-1) records 47–139 genuine HDBSCAN merge events per run. The two numbers measure different things and are both correct; spectral k is the wrong tool for xlarge.

Energy violation count grows with iteration depth for ALBERT-xlarge (up to 17–18 violations at 48 iterations), while total_delta stays bounded. This is the opposite of ALBERT-base, where total_delta grows sharply with depth (reaching 10.8 at 48iter, $\beta$=5.0 for short_heterogeneous).

### Multi-scale nesting is sparse

Spectral eigengap within HDBSCAN clusters (P1-3) detects nesting only at layers 0–3 for ALBERT variants and layers 0–1 for BERT. No hierarchical layering is detected across deep layers. The clusters found at plateau windows are flat partitions, not hierarchies.

### Pair HDBSCAN agreement: artifact fraction is universally low

Mutual-NN cycle pairs tagged as artifacts stay below 3% across all models and prompts. Semantic pairs (tokens in the same cluster) increase with prompt length and with ALBERT iteration depth (42% at 12iter → 74% at 48iter for ALBERT-base + wiki_paragraph). ALBERT-xlarge's semantic fraction is consistently lower than ALBERT-base's (56% vs 74% for long prompts at 48 iterations), consistent with its lower MaxMass and more diffuse clustering. Under random weights, ext_sem_frac is initially high (≈0.99 at 12iter) and declines with depth, confirming the signal is not a structural artifact of the metric.

### Token clusters carry semantic content; deeper models add syntactic structure

Mutual-NN cycle analysis at plateau layers recovers interpretable structure: `novelist ↔ poet`, `lancashire ↔ brussels`, `school ↔ lo` (Lowood). The clusters are not arbitrary geometric artifacts. They track co-reference and semantic similarity. Larger GPT-2 models and BERT add positional and syntactic groupings that smaller models do not show clearly.

---

## Known Limitations

1. **ALBERT-xlarge extended runs.** The spectral radius of V (1.278) predicts collapse at ~70–80 iterations: $\log(0.9)/\log(1.278) \approx 75$ passes needed. ALBERT-xlarge at 12–48 iterations confirms slow dynamics. Runs at 96–128 iterations are needed to determine whether collapse is merely delayed or structurally prevented by the complex eigenvalue structure.
2. **Spectral $k$ non-informative for ALBERT-xlarge.** Confirmed across all prompts and all iteration depths: spectral k = 1 everywhere, nMerges = 0. The zero mode dominates. HDBSCAN cluster tracking (P1-1) is the correct measure for xlarge.
3. **Depth-conditioning regime shift unexplained.** The jump in mean spectral radius from GPT-2-medium to GPT-2-large is abrupt and unexplained by depth, dimension, or training objective alone. The governing factor is unknown.
4. **BERT-large not in standard run.** In config as `bert-large-uncased`. Use `--models bert-large-uncased` to include it.
5. **Trained-vs-random contrast on ext_sem_frac not yet quantified.** The 10-seed random sweep establishes that ext_sem_frac declines monotonically with iteration depth under random weights. Whether trained weights produce a higher floor, a different slope, or plateau-aligned semantic structure at matched iteration depths is unquantified — the trained runs would need to be run at the same iteration depths to support that comparison.
6. **Trained-vs-random two-timescale ratio discrepancy unexplained.** ALBERT-base under random weights shows stronger two-timescale separation (ratio up to 13.44) than under trained weights (ratio ≤ 1.25). The direction of this effect is unexpected and the mechanism is unknown. One candidate: training reshapes the plateau width distribution rather than the collapse onset, compressing the ratio without suppressing metastability.
7. **Final-layer LM-head contamination.** GPT-2-small and GPT-2-medium final layers show extreme collapse from the unembedding projection, not dynamics. These layers are flagged but not removed from trajectory plots.
8. **Induction head confound.** Mutual-NN pairs include subword completions that may be attention artifacts. Per-pair HDBSCAN agreement (P1-4) tags but does not remove them. Artifact fraction stays below 3%, suggesting the confound is small but present.
9. **GPT-2-small repeated-tokens anomaly.** Mass never reaches 0.9 in the control, making the two-timescale ratio undefined. This is not a code failure — regular prompts do reach high mass. The model's clustering appears to require semantic/positional diversity in the input.

---

## Modules

### `run_1.py` — CLI entry point
Orchestrates the full pipeline. Key arguments:
- `--models`, `--prompts` — subset selection
- `--fast` — albert-base-v2 + wiki_paragraph only, legacy snapshots
- `--no-extended` — disable ALBERT extended-iteration mode
- `--legacy-snapshots` — use [12,24,36,48] instead of the dense sweep
- `--random-baseline` — add albert-base-v2-random (untrained control); included in standard run
- `--sublayer` — additionally run post-attention and post-FFN sublayer streams, saved as `{model}@attn` / `{model}@ffn` run directories (supplementary, excluded from cross-run comparison)
- `--length-sweep` — run wiki_paragraph truncated at each LENGTH_SWEEP_TOKENS target
- `--replot RUN_DIR`, `--summary RUN_DIR` — replot or summarize a saved run

ALBERT extended mode runs a single forward pass to `ALBERT_MAX_ITERATIONS` and slices the trajectory at each snapshot, saving one result directory per (prompt, depth) pair. Sublayer streams use forward hooks on `attn.c_proj` / `mlp` (GPT-2) or `attention.output` / `output` (BERT/ALBERT) to capture intermediate residual streams.

### `analysis.py` — Layer-wise analysis loop
Ingests hidden states and attentions, calls every metric/clustering/projection function, collects results into a single dict. Pre-computes normed activations and Gram matrix once per layer. Post-loop: cluster tracking (P1-1), plateau layer identification.

### `metrics.py` — Core per-layer scalar metrics
Pairwise inner products, interaction energies (batched over $\beta$), effective rank, attention entropy, nearest-neighbor indices and stability, linear CKA, energy drop pair localization.

### `clustering.py` — Clustering algorithms and projections
Agglomerative threshold sweep, KMeans silhouette, HDBSCAN, PCA, UMAP. Multi-scale nesting (P1-3): spectral eigengap within each HDBSCAN cluster. Per-pair HDBSCAN agreement (P1-4): mutual-NN cycles tagged as semantic vs artifact.

### `reporting.py` — Report generation
Per-run `llm_report.txt` and session-level `llm_cross_run_report.txt`. The cross-run report is the primary artifact for downstream analysis — it contains the summary table, plateau locations, merge events, energy monotonicity by run, prompt sensitivity, per-head Fiedler consistency, cluster tracking summary, nesting summary, pair agreement summary, and collapse control runs.

### `io_utils.py` — Serialization
`save_run`: writes v2 split format — `geometry.json`, `energies.json`, `clustering.json`, `spectral.json`, `activations.npz`, `attentions.npz`, `clusters.npz`, `centroid_trajectories.npz`. Each JSON is <100KB. `load_run` is backward-compatible with v1 `metrics.json`. `replot_all`: regenerate plots from saved data without model reload.

### `core/config.py` — Global constants
Model registry (9 entries including `bert-large-uncased` and `albert-base-v2-random`), prompt variants (8), $\beta$ values, distance thresholds, Sinkhorn tolerances, ALBERT iteration sweep parameters. Device selection.

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

Phase 1 establishes that metastability exists, that the energy functional is not monotone, and that both properties are present under random weights — metastability is a structural consequence of the iterated-map architecture, not installed by training. Phase 2 asks why the energy violations occur and what the weight structure is doing.

The paper's framework attributes metastability to the tension between attractive dynamics (softmax attention pulls tokens together) and repulsive dynamics (mixed-sign eigenspectrum of V pushes them apart). Phase 1 measures the outcome of this tension. Phase 2 measures the tension itself.

Everything Phase 2 needs from Phase 1 is saved: activations at every layer as `.npz`, plateau layer windows, merge event indices, energy violation layers, energy drop token pairs, and token lists.
