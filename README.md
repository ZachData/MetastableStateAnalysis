# Metastability in Transformers

An empirical investigation of metastable states in transformer residual streams, motivated by [Geshkovski et al. (2024), *A Mathematical Perspective on Transformers*](https://arxiv.org/abs/2312.10794).

The paper proves that token representations in idealized transformers collapse into a small number of clusters over depth, passing through long-lived *metastable* configurations before merging. This project asks: does that structure survive in real, trained models — and if so, can we read it interpretably?

---

## Research Plan

The project is structured as six sequential phases, each independently falsifiable.

**Phase 1 — Empirical metastability detection** *(implemented)*
Confirm the signal exists before anything else. Extract residual stream activations post-layernorm at every layer, replicate Figure 1 of the paper on real prompts, and look for plateaus in cluster count and inner-product histogram shape. ALBERT is the primary model because weight-sharing gives cleaner dynamics — the same layer block runs repeatedly, so metastable structure either emerges or it doesn't with no ambiguity from heterogeneous weights.

**Phase 2 — V eigenspectrum characterization**
Extract value matrices from each layer. Compute their eigenspectra and record sign distribution. Layers with mixed-sign spectra (repulsive and attractive directions coexisting) are candidate metastability-rich layers. Cross-reference locations with Phase 1 plateau windows.

**Phase 3 — Crosscoder training**
Train a crosscoder spanning all layers on the residual stream. Monitor whether features stratify naturally into short-lived and long-lived activation patterns. ALBERT's weight-sharing is critical here: a feature active across many consecutive layers is tracking a genuine dynamical plateau, not a coincidence of different weight matrices.

**Phase 4 — Metastable feature identification**
For each crosscoder feature, compute per-token activation trajectory variance across layers. Low variance over a window followed by a spike is a metastable feature candidate. Look for coordinated reorganization events — sets of features changing simultaneously across tokens. These are merge events. Cross-reference with Phase 1 cluster count plateaus.

**Phase 5 — Cluster identity and merge event characterization**
At each metastable window, compute cluster centroids in activation space and identify which crosscoder features are characteristically active. At each merge event, record which features die and which activate. This is the interpretable content of what that layer is computing.

**Phase 6 — Backwards tracing**
Train tuned lens probes per layer. Apply to cluster centroids at each metastable window. Unembedding gives token probability distributions — what each cluster means semantically at each depth. At merge events, compare pre-merge token distributions. The semantic distinction being erased is what that layer computes.

### Falsification criteria

Each phase has a clear stopping condition:

- **Phase 1**: no plateaus → metastability doesn't survive trained transformer dynamics at this scale
- **Phase 3**: crosscoder features don't stratify by lifetime → representations aren't organized along dynamical structure
- **Phase 4**: feature plateaus don't align with cluster count plateaus → features aren't tracking metastable configurations
- **Phase 6**: tuned lens distributions are incoherent at early layers → backwards tracing isn't reliable enough to interpret

Each phase is independently publishable even if later phases fail.

---

## Phase 1 — What's Implemented

### Metrics computed per layer

- **Pairwise inner products** on S^{d-1}: mean, std, histogram, fraction of pairs > 0.9
- **Interaction energy** E_β for β ∈ {1, 2, 5} — theory predicts monotone increase
- **Effective rank** — spectral entropy of singular values; collapse toward 1 indicates clustering
- **Agglomerative clustering** sweep across 12 cosine-distance thresholds
- **KMeans** best-k via silhouette score
- **HDBSCAN** cluster count (if installed)
- **Spectral eigengap** on the Gram matrix — threshold-free cluster count estimate directly motivated by the paper's geometry
- **Sinkhorn-Knopp** doubly stochastic normalization of attention + Fiedler value per head
- **Attention entropy** per head
- **PCA** projections (PC1–3) with explained variance trajectory
- **UMAP** projections (optional)

### Models tested

| Model | Notes |
|---|---|
| `albert-base-v2` | Primary. Weight-sharing lets you run 48+ iterations of the same block |
| `albert-xlarge-v2` | Higher-dimensional version of the same dynamics |
| `bert-base-uncased` | 12-layer baseline without weight sharing |
| `gpt2` | Decoder-only comparison |

### Prompt variants

Six prompts designed to stress-test different clustering regimes: a wiki paragraph, short homogeneous, short heterogeneous, long structured, repeated tokens, and a minimal 4-token sequence.

---

## Repository Structure

```
├── config.py        # All constants: BETA_VALUES, thresholds, prompts, model registry
├── models.py        # load_model, extract_activations, extract_albert_extended
├── metrics.py       # Pairwise IPs, Gram matrix, interaction energy, effective rank, attention entropy
├── sinkhorn.py      # Sinkhorn-Knopp normalization, Fiedler value, cluster count
├── spectral.py      # Eigengap heuristic on Gram matrix Laplacian
├── clustering.py    # Agglomerative sweep, KMeans, HDBSCAN, PCA, UMAP
├── analysis.py      # Layer-wise analysis loop — calls all of the above
├── plots.py         # All matplotlib figure generation
├── reporting.py     # Plateau detection, terminal summaries, LLM-ready text reports
├── io_utils.py      # Save/load run artifacts (metrics.json, activations.npz, CSV)
└── run.py           # Experiment orchestrator and CLI entry point
```

Each module has a single responsibility. To add a metric: write it in `metrics.py` or `sinkhorn.py`, call it in `analysis.py`. To change what gets plotted: edit only `plots.py`. To change models or prompts: edit only `config.py`.

---

## Installation

```bash
pip install torch transformers scikit-learn scipy tqdm matplotlib
```

Optional (recommended):
```bash
pip install hdbscan umap-learn
```

---

## Usage

**Run the default experiment** (ALBERT-base, all prompts, 48-iteration extended mode):
```bash
python run.py
```

**Fast single-model run:**
```bash
python run.py --fast
```

**Specific models and prompts:**
```bash
python run.py --models albert-base-v2 bert-base-uncased --prompts wiki_paragraph repeated_tokens
```

**Standard layer stack instead of extended iterations:**
```bash
python run.py --no-extended
```

**Regenerate all plots from a saved run without reloading the model:**
```bash
python run.py --replot metastability_results/2024-01-01_12-00-00/albert-base-v2_48iter_wiki_paragraph
```

**Print a text summary of a saved run:**
```bash
python run.py --summary metastability_results/2024-01-01_12-00-00/albert-base-v2_48iter_wiki_paragraph
```

---

## Output Structure

Each run creates a timestamped directory under `metastability_results/`:

```
metastability_results/
└── 2024-01-01_12-00-00/
    ├── experiment.txt                          # Full parameter manifest
    ├── cross_model_wiki_paragraph.png          # Cross-model comparison plots
    ├── llm_cross_run_report.txt                # Comparative LLM-ready report
    ├── V_spectrum_albert-base-v2.png           # Value matrix singular values
    └── albert-base-v2_48iter_wiki_paragraph/
        ├── metrics.json                        # Full results dict
        ├── activations.npz                     # L2-normed activations (layers, tokens, d)
        ├── attentions.npz                      # Attention weights (layers, heads, n, n)
        ├── tokens.txt                          # Token index list
        ├── layer_metrics.csv                   # Flat CSV of all per-layer scalars
        ├── llm_report.txt                      # Self-contained LLM analysis report
        ├── *_trajectory.png                    # 4×4 summary panel
        ├── *_histograms.png                    # Inner-product histograms (paper Fig. 1)
        ├── *_pca.png                           # Token PCA positions across layers
        ├── *_sinkhorn_heads.png                # Per-head Fiedler heatmap
        └── *_spectral.png                      # Laplacian eigenvalue spectrum
```

---

## What to Look For

The primary signal of metastability is **plateau structure** in:

1. Fraction of token pairs with inner product > 0.9 (should plateau, then jump)
2. Effective rank (should drop to a plateau, then drop again at merge events)
3. Spectral k from eigengap heuristic (should hold at k > 1, then step down)
4. Sinkhorn Fiedler value (should be low — near-disconnected clusters — during metastable windows)

When these plateau at the same layers simultaneously, and when the inner-product histograms show a stable multimodal or bimodal shape across those layers, that's the empirical signature the paper predicts.

The `llm_report.txt` in each run directory contains a pre-computed data table, trend descriptions, plateau locations, merge event detection, and flagged anomalies formatted for direct LLM analysis.

---

## Reference

Geshkovski, B., Letrouit, C., Polyanskiy, Y., & Rigollet, P. (2024). *A Mathematical Perspective on Transformers*. [arXiv:2312.10794](https://arxiv.org/abs/2312.10794)
