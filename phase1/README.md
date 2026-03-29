# Phase 1 — Empirical Verification of Metastability

**Status:** Complete. Extensions (cluster tracking, multi-scale nesting, pair agreement filtering, BERT-large, dense ALBERT sweep) implemented. Run across 8 models × 5 prompts.

---

## Core Question

Geshkovski et al. (*A Mathematical Perspective on Transformers*) prove that transformer token representations, modeled as interacting particles on $\mathbb{S}^{d-1}$, converge to a single cluster in the long-time limit. Before convergence, the dynamics pass through **metastable states** — multi-cluster configurations that persist across many consecutive layers before abruptly merging. The paper establishes this for a simplified model where $Q^\top K = V = I_d$. Phase 1 asks: **does this metastability survive in trained architectures with learned weight matrices, multi-head attention, and feed-forward layers?**

The falsification criterion: if no plateaus appear in cluster count or inner-product histograms across consecutive layers, metastability does not survive trained dynamics at this scale, and the project stops.

Phase 1 passed.

---

## Theoretical Predictions Tested

Seven predictions from the paper, checked against empirical trajectories:

1. **Tokens cluster over layers** — pairwise inner products $\langle x_i, x_j \rangle$ drift toward 1.
2. **Two-timescale dynamics** — fast initial grouping, slow pairwise merging.
3. **Metastable states** appear as plateaus in cluster count metrics.
4. **ALBERT** (shared weights) should show cleaner dynamics than BERT/GPT-2.
5. **Higher $\beta$** (sharper attention) → stronger metastability.
6. **Higher dimension $d$** → faster convergence to single cluster (Theorem 6.1).
7. **Interaction energy $E_\beta$** is monotone increasing along the trajectory (Theorem 3.4).

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

### V Weight Spectrum (Phase 2 prep)

Singular values and eigenvalues of V matrices extracted and saved per model. Sign distribution (fraction positive/negative real part) stored for cross-referencing with plateau locations.

---

## Results

### Metastability exists in trained ALBERT

ALBERT-base at 48 iterations produces clear plateau structure in cluster count and mass-near-1 across all natural language prompts. The two-timescale dynamics the paper predicts are visible: a fast initial phase (rising $\text{ip\_mean}$, falling rank), followed by a slow phase of pairwise cluster merging (step-drops in cluster count, step-jumps in mass). CKA plateaus align with cluster count plateaus, confirming that the representation genuinely stabilizes during metastable windows.

### Depth matters: 12 iterations is insufficient

At 12 iterations, ALBERT produces only the fast phase — weak clustering with $\text{ip\_mean}$ below 0.45, mass-near-1 near zero, no merge events. At 24, onset of merging. By 36, multiple merge events. At 48, near-collapse on most prompts. The metastable window is wide relative to ALBERT's practical operating depth. A 12-layer ALBERT (standard deployment) lives entirely within the fast phase and never reaches the metastable regime.

### Energy monotonicity is universally violated

Every run violates Theorem 3.4 — every $\beta$, every prompt, every model. Violations scale with $\beta$. Two types:

1. **Degenerate-regime violations** (effective rank < 3) — floating-point noise in collapsed layers. Suppressed.
2. **Structural violations** at active layers — top contributing pairs are semantically coherent. At "reset" layers (where $\text{ip\_mean}$ drops and rank increases), the largest negative contributions come from co-reference pairs being broken apart. Example: Jane Eyre prompt yields `novelist ↔ poet`, `lancashire ↔ school`, `cowan ↔ charlotte`.

### Larger models oscillate rather than collapse

GPT-2-xl (48 layers, 1600-dim) does not converge monotonically. In late layers, cluster count and $\text{ip\_mean}$ oscillate around a metastable density rather than settling. The larger model sustains a richer attractor geometry that resists full convergence within its depth.

### Merge events are architecture-determined

GPT-2's merge events occur at the same layers regardless of input. Different prompts produce different cluster contents but the same merge schedule. The layer-wise transition structure is a property of the weights, not the input.

### Token clusters carry semantic content

Mutual-NN cycle analysis at plateau layers recovers interpretable structure: `novelist ↔ poet`, `lancashire ↔ brussels`, `school ↔ lo` (Lowood). The clusters are not arbitrary geometric artifacts. They track co-reference and semantic similarity.

### Clustering methods disagree systematically

Spectral eigengap consistently reports $k = 2$ on long prompts; HDBSCAN finds 30–60+. Spectral captures the dominant bipartition. HDBSCAN captures local density neighborhoods. The disagreement reflects genuine multi-scale structure.

### Prompt length scales clustering pressure

Max mass-near-1 increases monotonically with token count: 8 tokens → 0.68, 58 tokens → 0.97. Matches Theorem 6.3 ($d \geq n$ regime): more particles produce more clustering pressure.

### Repeated tokens are a degenerate case

High $\text{ip\_mean}$ from layer 0, merge events even at 12 iterations, unimodal histograms throughout. Tests collapse speed, not metastability. Excluded from cross-run aggregation and reported as a separate collapse control.

### Head stratification is real but resolution-limited in ALBERT

Per-head Fiedler profiling classifies all ALBERT-base heads as MIXED (0.3–0.7). Convergence at 48 iterations is too aggressive for clean CLUSTER/MIXING separation. Head 3 sits at the boundary (mean 0.34). Prediction confirmed by Phase 2: models with slower trajectories (GPT-2, BERT) produce cleaner head stratification.

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

1. **ALBERT head resolution.** All heads classify as MIXED at 48 iterations — the dynamics converge too fast for clean CLUSTER/MIXING separation. The dense iteration sweep (6–60 step 2) may surface a window where stratification is visible.
2. **Spectral vs HDBSCAN disagreement.** The $k=2$ vs $k=30\text{--}60$ gap is characterized but not resolved. Multi-scale nesting analysis (P1-3) addresses this; full resolution requires the crosscoder in Phase 3.
3. **Induction head confound.** Mutual-NN pairs include subword completions that may be attention artifacts. Pair HDBSCAN agreement (P1-4) tags but does not remove them. Phase 3 crosscoder features should separate the two mechanisms.
4. **No causal mechanism.** Phase 1 identifies *where* energy drops and *which token pairs* are involved. It does not identify *why*. Phase 2 answers this.

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
