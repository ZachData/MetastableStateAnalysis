# Phase 1 — Empirical Verification of Metastability

**Status:** Complete. Extensions (cluster tracking, multi-scale nesting, pair agreement filtering, BERT-large, dense ALBERT sweep, V eigenspectrum extraction) implemented. Run across 8 models × 5 prompts.

---

## Core Question

Geshkovski et al. (*A Mathematical Perspective on Transformers*) prove that transformer token representations, modeled as interacting particles on $\mathbb{S}^{d-1}$, converge to a single cluster in the long-time limit. Before convergence, the dynamics pass through **metastable states** — multi-cluster configurations that persist across many consecutive layers before abruptly merging. The paper establishes this for a simplified model where $Q^\top K = V = I_d$. Phase 1 asks: **does this metastability survive in trained architectures with learned weight matrices, multi-head attention, and feed-forward layers?**

The falsification criterion: if no plateaus appear in cluster count or inner-product histograms across consecutive layers, metastability does not survive trained dynamics at this scale, and the project stops.

Phase 1 passed.

---

## Theoretical Predictions Tested

Seven predictions from the paper, checked against empirical trajectories:

1. **Tokens cluster over layers** — pairwise inner products $\langle x_i, x_j \rangle$ drift toward 1. ✓ Universal.
2. **Two-timescale dynamics** — fast initial grouping, slow pairwise merging. ✓ Confirmed for BERT-base, GPT-2-large, GPT-2-xl. ✗ Not cleanly separated in ALBERT (metastable window ≤ collapse onset).
3. **Metastable states** appear as plateaus in cluster count metrics. ✓ Universal.
4. **ALBERT** (shared weights) should show cleaner dynamics than BERT/GPT-2. ✓ Partially — ALBERT-base is cleanest but degenerately collapses. ALBERT-xlarge resists collapse in a way the theory does not predict.
5. **Higher $\beta$** (sharper attention) → stronger metastability. ✓ Confirmed.
6. **Higher dimension $d$** → faster convergence to single cluster (Theorem 6.1). ✗ Falsified. ALBERT-xlarge ($d=2048$) converges slower than ALBERT-base ($d=768$). The governing variable is the spectral radius of $V$, not $d$.
7. **Interaction energy $E_\beta$** is monotone increasing along the trajectory (Theorem 3.4). ✗ Universally violated. Mechanism identified: $V$'s 96–98% complex eigenvalues introduce rotational dynamics; the near-50/50 real-part sign split means no dominant attractive or repulsive character.

---

## Experimental Setup

### Models

| Model | Layers | $d$ | Heads | Weight sharing | Role |
|-------|--------|-----|-------|----------------|------|
| ALBERT-base-v2 | 12 (iterated to 60) | 768 | 12 | Yes | Primary — direct analog of paper's recurrent dynamics |
| ALBERT-xlarge-v2 | 24 (iterated to 60) | 2048 | 16 | Yes | High-dimensional shared-weight control |
| BERT-base-uncased | 12 | 768 | 12 | No | Baseline non-shared encoder |
| BERT-large-uncased | 24 | 1024 | 16 | No | Fills gap between BERT-base and GPT-2-medium |
| GPT-2 | 12 | 768 | 12 | No | Autoregressive baseline |
| GPT-2-medium | 24 | 1024 | 16 | No | Intermediate depth |
| GPT-2-large | 36 | 1280 | 20 | No | Deep non-shared control |
| GPT-2-xl | 48 | 1600 | 25 | No | Maximum depth |

ALBERT is the primary model. Its weight sharing makes every plateau a genuine dynamical fixpoint of a single map, not an artifact of differing weight matrices at different depths. ALBERT is run once to `ALBERT_MAX_ITERATIONS` (60) and sliced at every even depth from 6 to 60, testing whether metastability onset is a sharp phase transition or gradual crossover.

### Prompts

Five prompts span a range of token counts and semantic structure:

- **short_heterogeneous** — two unrelated sentences (quantum mechanics + stock market). Tests whether semantically unrelated domains form distinct clusters.
- **wiki_paragraph** — Charlotte Brontë Wikipedia excerpt (~300 tokens). Rich co-reference structure.
- **sullivan_ballou** — Civil War letter (~300 tokens). Dense emotional/thematic content.
- **paper_excerpt** — passage from the Geshkovski paper itself. Self-referential control.
- **repeated_tokens** — 288 copies of "cat." Degenerate control measuring collapse speed. Excluded from metastability analyses; reported separately.

### Activation Extraction

At every layer, residual stream activations are projected onto the unit sphere via L2 normalization (`layernorm_to_sphere`), producing an $(n_\text{tokens}, d_\text{model})$ matrix of unit vectors. All geometric analyses operate on these sphere-projected representations, matching the paper's mathematical setting. Raw (pre-normalization) activations are retained for effective rank computation, where magnitude information is meaningful.

---

## Methods

### Geometry of the Token Cloud

**Mean pairwise cosine similarity** ($\text{ip\_mean}$) — average of $\langle x_i, x_j \rangle$ over all pairs $i \neq j$. Proxy for cloud collapse. Values near 0 indicate spread; near 1, convergence.

**Mass near 1** — fraction of token pairs with cosine similarity exceeding 0.9. Sharpest indicator of cluster formation: a jump from 0 to positive marks onset of tight grouping.

**Inner product histogram** — full pairwise cosine similarity distribution in 50 bins over $[-1, 1]$. Multimodal histograms indicate distinct clusters. Stable multimodal shape across consecutive layers is the most direct metastability signature.

**Effective rank** — $\exp(H)$ where $H$ is the Shannon entropy of the normalized singular value distribution of the raw activation matrix. Rank 1 = all tokens collinear. Rank near $d$ = full-space span. Drops correspond to collapse; plateaus between drops correspond to stable geometric configurations.

### Energy Functional

**Interaction energy** $E_\beta = \frac{1}{2\beta n^2} \sum_{i,j} e^{\beta \langle x_i, x_j \rangle}$ at $\beta \in \{0.1, 1.0, 2.0, 5.0\}$. The paper proves this functional increases monotonically under the simplified dynamics (Theorem 3.4). Tracking it across layers in trained models tests whether the gradient-flow structure is preserved or broken.

**Energy drop localization** — at layers where $E_\beta$ decreases (violating monotonicity), the per-pair contribution $\Delta_{ij} = e^{\beta \langle x_i, x_j \rangle_{L+1}} - e^{\beta \langle x_i, x_j \rangle_L}$ is decomposed. The top-10 token pairs with the most negative $\Delta$ are identified by token string, across all $\beta$ values. CLS, SEP, and punctuation tokens are flagged.

### Cluster Counting

Four independent methods, each capturing different structure:

- **KMeans** with silhouette score over $k \in [2, 10]$ — best-fitting partition. Suppressed when effective rank < 10 (geometry too degenerate for meaningful $k$).
- **HDBSCAN** — density-based, no assumed $k$, allows noise labeling.
- **Agglomerative** at 12 cosine distance thresholds in $[0.05, 0.6]$ — hierarchical merging.
- **Spectral eigengap** — cluster count implied by the largest gap in the normalized Laplacian eigenspectrum, with secondary gap detection for nested structure.

Method agreement is tracked layer-by-layer.

### Cluster Tracking Across Layers (P1-1)

HDBSCAN clusters are matched across adjacent layers by maximum Jaccard overlap of token membership, using the Hungarian algorithm for optimal assignment. Unmatched source clusters are deaths, unmatched targets are births, many-to-one matches are merges. This replaces spectral-$k$ drop counting with token-level accounting of what actually moved. Centroid trajectories of matched clusters are saved as `centroid_trajectories.npz`.

### Multi-Scale Cluster Nesting (P1-3)

Spectral eigengap is run within each HDBSCAN cluster to detect hierarchical organization. If global spectral $k \leq 3$ while individual HDBSCAN clusters show internal sub-structure ($k > 1$), the token cloud has two-scale organization: macro-bipartition containing many micro-clusters.

### Per-Pair Induction Head Filtering (P1-4)

Mutual nearest-neighbor pairs are tagged by HDBSCAN cluster agreement. Pairs sharing a cluster are tagged "semantic" (genuine clustering). Pairs that are mutual NNs but in different clusters are tagged "artifact" (likely induction-head-driven cross-position subword completions, e.g. `he ↔ ger` for "Heger").

### Representation Dynamics

**Linear CKA** between consecutive layers — $\text{CKA}(X, Y) = \|Y^\top X\|_F^2 / (\|X^\top X\|_F \cdot \|Y^\top Y\|_F)$ where $X, Y$ are centered activation matrices. 1.0 = identical up to rotation. A CKA plateau is the theoretically cleanest metastability signature: the representation is not changing.

**Nearest-neighbor stability** — fraction of tokens whose cosine-nearest neighbor is unchanged from the previous layer. A plateau near 1.0 means local neighborhood structure has locked in.

### Attention Structure

**Sinkhorn Fiedler value** — attention matrix at each head is Sinkhorn-normalized to a doubly stochastic matrix, then the algebraic connectivity ($\lambda_2$ of the Laplacian) is computed. Low Fiedler = near-disconnected components = attention concentrating within clusters.

**Per-head Fiedler profiling** — each head classified by mean Fiedler across the active (non-collapsed) phase as CLUSTER (< 0.3), MIXED (0.3–0.7), or MIXING (> 0.7). Cross-prompt consistency checked in the cross-run report.

### Plateau and Merge Detection

**Plateau detection** — sliding window on five metrics simultaneously (mass, rank, spectral $k$, HDBSCAN $k$, CKA) with per-metric tolerance. Plateaus that align across metrics are the strongest evidence.

**Merge event detection** — layer where cluster count drops according to $\geq 2$ independent methods simultaneously. Filters out single-method noise.

**Token cluster membership** — at plateau layers, mutual nearest-neighbor cycles extracted and classified as duplicate (same token string) or semantic (distinct strings).

### V Weight Spectrum

Singular values and eigenvalues of $V$ matrices extracted and saved per model (`v_eigenspectrum_{model}.json`). Per layer: `sv_max`, `sv_min`, `sv_mean`, `sv_std`, `spectral_gap` ($\sigma_1/\sigma_2$), `eig_frac_pos_real`, `eig_frac_neg_real`, `eig_frac_complex`, `eig_real_mean`, `eig_spectral_radius` (max $|\lambda|$). These are the primary Phase 2 inputs: spectral radius governs convergence rate under shared-weight iteration; the complex eigenvalue fraction and real-part sign split determine whether the gradient flow model applies. Note: `plots.analyze_value_eigenspectrum` extracts $W_V$ per head, not the composed OV circuit — eigenvalue sign analysis requires the full $V_\text{eff} = \sum_h W_O^{(h)} W_V^{(h)}$, which is computed in Phase 2.

---

## Results

### Metastability exists in trained ALBERT

ALBERT-base at 48 iterations produces clear plateau structure in cluster count and mass-near-1 across all natural language prompts. The two-timescale dynamics the paper predicts are visible: a fast initial phase (rising $\text{ip\_mean}$, falling rank), followed by a slow phase of pairwise cluster merging (step-drops in cluster count, step-jumps in mass). CKA plateaus align with cluster count plateaus, confirming that the representation genuinely stabilizes during metastable windows.

### Depth matters: 12 iterations is insufficient

At 12 iterations, ALBERT produces only the fast phase — weak clustering with $\text{ip\_mean}$ below 0.45, mass-near-1 near zero, no merge events. At 24, onset of merging. By 36, multiple merge events. At 48, near-collapse on most prompts. The metastable window is wide relative to ALBERT's practical operating depth. A 12-layer ALBERT (standard deployment) lives entirely within the fast phase and never reaches the metastable regime.

### Energy monotonicity is universally violated

Every run violates Theorem 3.4 — every $\beta$, every prompt, every model. Violations scale with $\beta$. The V eigenspectrum explains why: across all models and all layers, 96–98% of $V$'s eigenvalues have non-trivial imaginary parts, and the real parts split almost exactly 50/50 between positive and negative (within ±1% in every case measured). $V$ has no dominant attractive or repulsive character — it mixes both in roughly equal measure and introduces substantial rotation. The gradient flow framework assumes dynamics decompose along fixed eigenvector directions; rotational dynamics violate this at the level of the weight matrices themselves, not as an emergent effect of the forward pass.

Violation character differs by architecture:

1. **Degenerate-regime violations** (effective rank < 3) — floating-point noise in collapsed layers. Suppressed.
2. **Structural violations** at active layers — top contributing pairs are semantically coherent. Violation character differs substantially by architecture. GPT-2-xl produces many small violations (27–37 out of 48 layers, total $\Delta E \approx 0.1$–$0.2$), dominated by punctuation repulsion (commas repelling commas, periods repelling `My`). The structural/positional subspace fights semantic clustering throughout the network. ALBERT-base produces few large violations (1–7 per run, total $\Delta E$ up to 14.5), concentrated at specific transition layers (layer 11 consistently), with semantic token pairs (`joy ↔ country`, `revolution ↔ those`) as the top contributors. At "reset" layers (where $\text{ip\_mean}$ drops and rank increases), co-reference pairs are being broken apart. The Spearman correlation between $|\Delta E|$ and subsequent rank change is ρ = 0.83 (ALBERT-base), 0.54 (GPT-2-large), 0.41 (GPT-2-xl) — large energy drops reliably precede geometric reorganization.

### ALBERT-base collapses; ALBERT-xlarge resists — and dimension is not the cause

ALBERT-base with shared weights reaches mass>0.9 = 1.0 by iteration 24 on long prompts and stays collapsed (effective rank drops to ~1.4). ALBERT-xlarge, also shared weights but $d = 2048$, reaches only mass>0.9 ≈ 0.29 at iteration 48. This directly contradicts prediction (f) (Theorem 6.1): higher dimension should accelerate convergence, but the opposite is observed.

The V eigenspectrum resolves this. ALBERT-base's spectral radius is 1.506; ALBERT-xlarge's is 1.278. Under shared-weight iteration, the dominant eigenmode amplifies by this factor each pass. After 24 iterations: $1.506^{24} \approx 2700\times$ vs $1.278^{24} \approx 340\times$. ALBERT-xlarge also has `sv_mean` ≈ 1.02 (near-isometric on average), so the shared-weight iteration drifts slowly instead of collapsing. Dimension is not the governing variable — the spectral radius of $V$ is. ALBERT-xlarge learned a more conservative $V$.

ALBERT-xlarge has richer fine-grained cluster structure despite resisting global collapse: HDBSCAN finds 591 cluster trajectories and 108 merge events across layers vs GPT-2-xl's 442 trajectories and 38 merges. It also has a stronger Fiedler-cluster correlation (ρ = −0.62 vs ρ = −0.40 for GPT-2-xl). The structure never consolidates into a global partition visible to spectral methods (spectral $k = 1$ at every layer in ALBERT-xlarge; the zero mode dominates the eigengap), but it is genuine and geometrically rich.

### GPT-2-xl has a layer-wise spectral radius profile that explains its dynamics

GPT-2-xl's V matrices are not uniform across layers. Layers 0–2 are contractive (spectral radii 0.73, 0.82, 0.85) — these layers compress the token cloud, driving the initial sharp rise in $\text{ip\_mean}$ (0.10 → 0.64 in layers 0–3). The middle layers are increasingly expansive, peaking around layers 35–43 (radii 1.70–1.74). These are where energy violations concentrate. The very late layers return toward neutral (layer 47: 1.07), which is why the layer 48 reset (CKA drop 0.995 → 0.418, $\text{ip\_mean}$ drop 0.624 → 0.298) has a direct mechanism: near-neutral $V$ with complex eigenvalues applied to a highly clustered state rotates tokens out of alignment without re-amplifying them.

### Depth-conditioning of spectral radius is not monotone — there is a regime shift

With all models measured, the depth-conditioning picture is more structured than a simple inverse relationship:

| Model | $d$ | Layers | Spectral radius range | Mean |
|-------|-----|--------|-----------------------|------|
| GPT-2-small | 768 | 12 | 1.69 – 4.62 | 3.28 |
| GPT-2-medium | 1024 | 24 | 1.59 – 4.07 | 3.21 |
| GPT-2-large | 1280 | 36 | 0.94 – 1.81 | 1.38 |
| GPT-2-xl | 1600 | 48 | 0.73 – 1.74 | 1.33 |
| BERT-base | 768 | 12 | 0.79 – 1.09 | 0.94 |
| ALBERT-base | 768 | shared | 1.51 (single) | — |
| ALBERT-xlarge | 2048 | shared | 1.28 (single) | — |

GPT-2-small and GPT-2-medium are in a qualitatively different regime from GPT-2-large and GPT-2-xl. The transition is not gradual — mean spectral radius drops from 3.21 (medium, 24 layers) to 1.38 (large, 36 layers), a factor of 2.3×, with no intermediate values. Within each regime, the per-layer profiles share a shape: spectral radius increases with layer depth in both small/medium (1.59 → 4.07) and large/xl (0.73 → 1.74), with late-layer drops toward neutral.

Depth alone does not explain this. BERT-base has 12 layers and a mean spectral radius of 0.94 — similar to GPT-2-large/xl, not to GPT-2-small which also has 12 layers. The governing variable appears to be something learned during training rather than a direct function of depth. One candidate: GPT-2-small and GPT-2-medium were trained with a parameter budget concentrated in fewer layers, requiring each $V$ to carry more signal per layer. GPT-2-large and GPT-2-xl, having more layers, can afford conservative per-layer $V$ matrices. BERT's bidirectional training objective may independently select for conservative $V$ matrices regardless of depth.

GPT-2-medium layers 1 and 23 have anomalously high leading singular values (15.66 and 15.27) at near-average spectral radii — consistent with the Phase 2 finding of OV spectral norm spikes at the LM-head-adjacent layers in smaller models.

### Merge events are architecture-determined; plateau onset is content-driven

GPT-2's merge events occur at the same layers regardless of input. Different prompts produce different cluster contents but the same merge schedule. The layer-wise transition structure is a property of the weights, not the input.

Plateau onset, by contrast, is mostly content-driven. For GPT-2-xl, ALBERT-xlarge, and ALBERT-base (≥24 iterations), the standard deviation of plateau onset across prompts is 5–9 layers. Exceptions: BERT-base (SD = 0) and GPT-2-medium (SD = 1.7), where onset appears to be a weight-level property.

### Two-timescale separation is architecture-specific

Two-timescale separation (fast initial grouping, slow pairwise merging) is confirmed for BERT-base, GPT-2-large, and GPT-2-xl on the repeated-tokens control — the metastable window is substantially wider than the collapse onset. ALBERT models show no clean separation or weak separation: their metastable windows are narrower than or comparable to the collapse onset. The cluster content itself shows the two-timescale signature more clearly than the quantitative metrics: duplicate clusters (same token at different positions) dominate early layers, semantic clusters (distinct tokens with related meaning) grow as layers deepen, with duplicates dissolving into the semantic structure.

### Token clusters carry semantic content; deeper models add syntactic structure

Mutual-NN cycle analysis at plateau layers recovers interpretable structure: `novelist ↔ poet`, `lancashire ↔ brussels`, `school ↔ lo` (Lowood). The clusters are not arbitrary geometric artifacts. They track co-reference and semantic similarity. The same paradigmatic pairs appear across architectures at stable layers: `Ġcalmly ↔ Ġproudly`, `Ġstruggle ↔ Ġcontest`, `Ġblood ↔ Ġsuffering`, `Ġclosely ↔ Ġdiligently`. ALBERT-xlarge builds similar semantic pairs earlier (layer 7–9) with a higher duplicate-to-semantic ratio. Deeper models develop additional syntactic-discourse pairs not present in shallower ones: GPT-2-xl at layer 24 finds `conflict ↔ death`, `ĠBut ↔ ,`, `ĠI ↔ should`.

### Clustering methods disagree systematically; no hierarchical nesting found

Method agreement is poor across all runs. Spectral $k$, HDBSCAN $k$, agglomerative $k$, and $k$-means $k$ agree at essentially one layer per run. Spectral eigengap consistently reports $k = 2$ on long prompts; HDBSCAN finds 30–60+. These methods measure different structure: spectral captures the dominant bipartition, HDBSCAN captures local density neighborhoods. The disagreement is not a bug, but it means no single clustering method is sufficient.

Hierarchical nesting (global bipartition containing local sub-clusters) is near-universally absent. Only a handful of layers in short prompts show any nesting. Spectral $k$ is unreliable as the sole cluster-count metric for high-dimensional models — ALBERT-xlarge has spectral $k = 1$ at every layer despite 108 genuine merge events in HDBSCAN tracking.

### Prompt length scales clustering pressure

Max mass-near-1 increases monotonically with token count: 8 tokens → 0.68, 58 tokens → 0.97. Matches Theorem 6.3 ($d \geq n$ regime): more particles produce more clustering pressure.

### Repeated tokens are a degenerate case

High $\text{ip\_mean}$ from layer 0, merge events even at 12 iterations, unimodal histograms throughout. Tests collapse speed, not metastability. Excluded from cross-run aggregation and reported as a separate collapse control.

### Final-layer collapse in GPT-2 small and medium is an LM-head artifact

GPT-2-small's $\text{ip\_mean}$ goes from 0.64 to 0.97 in a single layer (11 → 12) with CKA dropping from 0.94 to 0.24. GPT-2-medium does the same at layer 23 → 24. This is the unembedding layer projecting into vocabulary space, not a metastable state. The mass>0.9 = 0.97 reading at GPT-2-small's final layer is the LM head collapsing representations for logit computation. It should not be interpreted as the model reaching the paper's single-cluster attractor. GPT-2-large and GPT-2-xl have analogous final-layer norm spikes but at much smaller ratios to their stack means, and their late-layer dynamics continue past the LM head.

### Head stratification is architecturally determined

Per-head Fiedler profiling reveals a clean three-way split across architectures. All GPT-2-family heads at every scale classify as STABLE-CLUSTER (mean Fiedler < 0.3, often < 0.05) across every prompt — attention routing is structurally fixed and content-insensitive. ALBERT-base heads all become STABLE-MIXED by 24 iterations. ALBERT-xlarge splits exactly 8 CLUSTER / 8 MIXED, with many heads classified VARIABLE (change classification with prompt content) — consistent with shared weights reusing the same heads in different functional modes at different iterations. GPT-2's universally low Fiedler means attention always partitions tokens into near-disconnected groups, but this partitioning does not track the actual geometric clustering in the residual stream (Fiedler-cluster correlation ρ ≈ 0 to −0.40 for mid-size GPT-2 models vs ρ ≈ −0.62 for ALBERT).

---

## Saved Artifacts

Each (model, prompt) run persists:

| File | Contents |
|------|----------|
| `metrics.json` | Full results dict (all per-layer metrics, JSON-serializable) |
| `activations.npz` | L2-normed hidden states `(n_layers, n_tokens, d)` |
| `attentions.npz` | Attention weights `(n_layers, n_heads, n, n)` |
| `clusters.npz` | Per-layer KMeans labels/centroids, HDBSCAN labels |
| `centroid_trajectories.npz` | HDBSCAN centroid coordinates for tracked cluster trajectories |
| `plateau_attentions.npz` | Raw attention matrices at plateau layers (for Phase 3) |
| `tokens.txt` | Decoded token strings with indices |
| `layer_metrics.csv` | Flat CSV of per-layer scalars |
| `llm_report.txt` | Self-contained plain-text analysis for LLM consumption |

Cross-run outputs: `llm_cross_run_report.txt`, cross-model comparison plots, V eigenspectrum JSON.

---

## Known Limitations

1. **ALBERT-xlarge extended runs.** The spectral radius of V (1.278) predicts collapse at ~70–80 iterations: $\log(0.9) / \log(1.278) \approx 75$ passes needed to match ALBERT-base's 24-iteration collapse. This has not been run. ALBERT-xlarge at 96–128 iterations would confirm whether collapse is delayed or genuinely prevented by the complex eigenvalue structure.
2. **Depth-conditioning regime shift unexplained.** The jump in mean spectral radius from GPT-2-medium (3.21, 24 layers) to GPT-2-large (1.38, 36 layers) is abrupt and not explained by depth, dimension, or training objective alone. BERT-base (0.94, 12 layers) clusters with the large models despite matching GPT-2-small's depth. The governing factor — whether training data, learning rate schedule, parameter budget allocation, or something else — is unknown.
3. **Spectral $k$ unreliable for high-dimensional models.** ALBERT-xlarge has spectral $k = 1$ at every layer despite 108 genuine merge events in HDBSCAN tracking. The zero mode dominates the primary eigengap. Secondary eigengap detection partially compensates; full resolution requires per-cluster spectral analysis.
4. **Induction head confound.** Mutual-NN pairs include subword completions that may be attention artifacts (`he ↔ ger` for "Heger"). Per-pair HDBSCAN agreement (P1-4) tags but does not remove them. Phase 3 crosscoder features should separate the two mechanisms.
5. **Final-layer LM-head contamination.** GPT-2-small and GPT-2-medium final layers show extreme collapse that is the unembedding projection, not dynamics. These layers are flagged but not removed from trajectory plots.

---

## Modules

### `analysis.py` — Layer-wise analysis loop
Ingests hidden states and attentions, calls every metric/clustering/projection function, collects results into a single dict. Pre-computes normed activations and Gram matrix once per layer to eliminate redundant matrix multiplies. Post-loop: cluster tracking (P1-1), plateau layer identification (P1-7).

### `metrics.py` — Core per-layer scalar metrics
Pairwise inner products, interaction energies (single and batched), effective rank, attention entropy, nearest-neighbor indices and stability, linear CKA, energy drop pair localization. No plotting, no I/O.

### `clustering.py` — Clustering algorithms and projections
Agglomerative threshold sweep, KMeans silhouette, HDBSCAN, PCA, UMAP. Multi-scale nesting analysis (P1-3). Per-pair HDBSCAN agreement for induction head filtering (P1-4). Accepts pre-normed arrays to avoid redundant normalization.

### `cluster_tracking.py` — HDBSCAN cluster tracking across layers
Jaccard overlap matching between adjacent layers via Hungarian algorithm. Births, deaths, merges. Centroid trajectory computation from normed activations.

### `spectral.py` — Eigengap heuristic on Gram matrix Laplacian
Threshold-free cluster count estimation from the spectral structure of the pairwise inner-product matrix. Primary and secondary gap detection.

### `sinkhorn.py` — Sinkhorn-Knopp normalization + Fiedler analysis
Doubly stochastic normalization (batched across all heads). Fiedler value ($\lambda_2$ of normalized Laplacian). Sinkhorn cluster count from eigenvalues near 1.

### `plots.py` — Figure generation
Trajectory summary panels, inner-product histograms (paper Figure 1 replication), PCA panels, Sinkhorn heatmaps, spectral eigengap plots, eigenvalue spectra, ALBERT extended comparison, cross-model comparison, CKA trajectory, V weight spectrum.

### `reporting.py` — Text reports
Plateau detection, merge event detection, method agreement, NN cycle extraction, per-head Fiedler profiling. Single-run LLM report with 12 sections. Cross-run comparative report with cluster tracking summaries, nesting summaries, pair agreement summaries, and collapse control section.

### `io_utils.py` — Persistence
Save/load runs, activations, attentions, clusters, centroid trajectories, plateau attentions. Backward-compatible loading with defaults for fields added in later versions. Replot from saved runs without model loading.

### `config.py` — Global constants
Model registry (8 models), prompt variants (5), numerical parameters ($\beta$ values, distance thresholds, Sinkhorn tolerances, ALBERT iteration sweep). Device selection.

### `models.py` — Model loading and extraction
Standard forward pass extraction. ALBERT extended-iteration extraction (single pass to max depth, sliced at snapshot points). bfloat16 on CUDA, torch.compile when available.

### `run.py` — CLI entry point
`--models`, `--prompts`, `--fast` (ALBERT-base + wiki_paragraph with legacy snapshots), `--no-extended`, `--legacy-snapshots` (revert to [12,24,36,48]), `--replot`, `--summary`. Excludes repeated_tokens from cross-run metastability aggregation (P1-2).

---

## Transition to Phase 2

Phase 1 establishes that metastability exists and that the energy functional is not monotone. Phase 2 asks why.

The paper's framework attributes metastability to the tension between attractive dynamics (softmax attention pulls tokens together) and repulsive dynamics (mixed-sign eigenspectrum of V pushes them apart). Phase 1 measures the outcome of this tension. Phase 2 measures the tension itself.

Everything Phase 2 needs from Phase 1 is saved: activations at every layer as `.npz`, plateau layer windows, merge event indices, energy violation layers, energy drop token pairs, and token lists.
