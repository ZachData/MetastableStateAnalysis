# Metastability in Transformers

An empirical investigation of metastable states in transformer residual streams, motivated by [Geshkovski et al. (2024), *A Mathematical Perspective on Transformers*](https://arxiv.org/abs/2312.10794).

The paper proves that token representations in idealized transformers collapse into a small number of clusters over depth, passing through long-lived *metastable* configurations before merging. This project asks three questions:

1. Does that structure survive in real, trained models?
2. What mechanism drives it?
3. Can we ground interpretability tools in the resulting dynamics?

The first two are answered. The third is in progress.

---

## Results summary

### Phase 1 — Metastability exists *(complete)*

Confirmed across 8 models × 5 prompts. ALBERT (shared weights, iterated to 60 layers) shows clean plateau structure: cluster count holds steady across layer windows, then drops in step-changes at merge events. The two-timescale dynamics the paper predicts — fast initial grouping, slow pairwise merging — are visible in every natural language prompt. GPT-2 shows the same structure with model-specific merge schedules. Energy monotonicity (Theorem 3.4) is violated at every scale, structurally, with the largest violations at semantically coherent token pairs.

### Phase 2 — V's eigenspectrum is the mechanism *(complete)*

The composed OV matrix has mixed-sign eigenvalues. Attractive directions sustain clusters; repulsive directions cause energy violations. Two dynamical regimes: in ALBERT (shared weights), attention mediates V's effect directly — 100% of violation-layer tokens have negative self-interaction $x_i^\top V_\text{eff} x_i < 0$. In GPT-2 (per-layer weights), the FFN is the proximal channel — it pushes token updates into V's repulsive subspace. Rescaled-frame analysis ($z_i(L) = (e^{-V})^L x_i(L)$) eliminates 100% of violations in GPT-2, confirming V as the distal cause.

### Phase 3 — Crosscoder features have the right temporal structure but wrong geometry *(complete)*

A sparse crosscoder trained across layers on ALBERT-xlarge and GPT-2-large produces features with bimodal lifetimes: 236 short-lived (≤3 layers), 343 long-lived (≥L/2), 99.2% multilayer, 0% positional contamination. The temporal structure matches the two-timescale dynamics.

The decoder directions are geometrically random with respect to V's eigensubspaces. Mean projection equals the random baseline (3.14% vs expected 3.12%), SNR 0.18×. This is a clean null after five independent bug fixes that each could have masked it.

**Interpretation:** The crosscoder found features with the right temporal profile — they persist across the right layer windows — but the wrong geometric content. What makes a feature long-lived is syntactic or frequency-based regularity that happens to persist, not alignment with V's attractive subspace. Mechanism (V eigenspectrum) and representation (learned sparse features) are dissociated under sparsity.

### Phase 4 — In progress

Revised to address the Phase 3 null. Three parallel tracks: crosscoder activation patterns (testing whether feature *firing patterns* correspond to cluster membership even though decoder *directions* don't), direct geometric methods (LDA, PCA on deltas, linear probes — no learned dictionary), and non-sparse alternatives (low-rank autoencoders, ICA — testing whether sparsity was the wrong prior).

### Phases 5–6 — Planned

Cluster identity characterization at merge events. Tuned lens backwards tracing to read out what each cluster means semantically at each depth.

---

## The larger aim

SAE features, activation patching, and linear probes are standard interpretability tools. They work, but the explanations for *why* they work are thin. The metastability framework offers a physical picture: the model's residual stream is a dynamical system with attractors, phase transitions, and a spectral structure that governs when representations are stable and when they reorganize.

If interpretability tools can be grounded in this framework, they become predictive rather than descriptive. Specific predictions that fall out of the dynamics and can be tested:

- Activation patches applied during metastable plateaus should have lasting downstream effects; patches at merge events should be transient (the clusters are about to reorganize anyway).
- Linear probes trained at one plateau layer should generalize to other layers in the same plateau window and fail outside it.
- Sparse features with long lifetimes should correspond to stable cluster membership; short-lived features should activate at merge/violation layers.
- Non-sparse decompositions (low-rank AE, ICA) whose bottleneck dimension matches the cluster count should recover alignment with V's eigensubspaces that sparse methods miss.

These are falsifiable. Phase 4 tests the last two. Future experiments will test the first two.

---

## Repository structure

```
.
├── core/                     # Shared infrastructure
│   ├── config.py             # Model registry, prompts, constants
│   └── models.py             # Model loading, activation extraction
├── phase1/                   # Empirical metastability detection
│   ├── analysis.py           # Layer-wise metric computation
│   ├── clustering.py         # KMeans, HDBSCAN, spectral, PCA, UMAP
│   ├── cluster_tracking.py   # Cross-layer cluster identity tracking
│   ├── metrics.py            # Pairwise IPs, energy, effective rank
│   ├── spectral.py           # Eigengap heuristic
│   ├── sinkhorn.py           # Doubly stochastic normalization, Fiedler value
│   ├── reporting.py          # Terminal summaries, LLM-ready reports
│   ├── io_utils.py           # Save/load run artifacts
│   ├── plots.py              # All figure generation
│   └── run.py                # CLI entry point
├── phase2/                   # V eigenspectrum and mechanism identification
│   ├── weights.py            # OV eigendecomposition, subspace projectors
│   ├── trajectory.py         # Trajectory analysis (shared-weight models)
│   ├── trajectory_perlayer.py # Trajectory analysis (per-layer models)
│   ├── decompose.py          # Attn/FFN energy attribution
│   ├── ffn_subspace.py       # FFN projection onto V's eigensubspaces
│   ├── analysis.py           # Violation classification, plateau characterization
│   ├── analysis_extended.py  # Continuous correlations, norm confound, attractive-zone
│   ├── verdict_v2.py         # Per-run mechanistic verdict
│   ├── reporting.py          # Summary generation
│   └── run.py                # CLI entry point
├── phase3/                   # Crosscoder training and feature analysis
│   ├── crosscoder.py         # nn.Module, activation functions
│   ├── data.py               # Extraction, streaming, caching, Dataset
│   ├── training.py           # Training loop, mixed precision, dead feature resampling
│   ├── analysis.py           # Feature lifetimes, V-alignment, cluster correlation,
│   │                         #   co-activation, cross-phase bridge analyses
│   ├── steering.py           # Causal steering experiments
│   └── run.py                # CLI entry point
├── phase4/                   # Metastable feature identification (in progress)
│   └── README.md
├── phase5/                   # Cluster identity and merge characterization (planned)
│   └── README.md
├── phase6/                   # Tuned lens backwards tracing (planned)
│   └── README.md
├── plan                      # Original phase plan
├── phase1_writeup.md         # Full Phase 1 experiment report
└── phase2_writeup.md         # Full Phase 2 experiment report
```

---

## Models

| Model | d_model | Layers | Weight sharing | Role |
|-------|---------|--------|----------------|------|
| ALBERT-base-v2 | 768 | 12 (iterated to 60) | Yes | Primary Phase 1 model |
| ALBERT-xlarge-v2 | 2048 | 24 (iterated to 60) | Yes | Primary Phase 2–4 model |
| BERT-base-uncased | 768 | 12 | No | Baseline encoder |
| BERT-large-uncased | 1024 | 24 | No | Intermediate encoder |
| GPT-2 | 768 | 12 | No | Autoregressive baseline |
| GPT-2-medium | 1024 | 24 | No | Intermediate depth |
| GPT-2-large | 1280 | 36 | No | Primary Phase 3–4 model (Regime B) |
| GPT-2-xl | 1600 | 48 | No | Maximum depth |

ALBERT is the primary model throughout. Its weight sharing makes every plateau a genuine dynamical fixpoint of a single map. GPT-2-large provides the per-layer-weight contrast (Regime B: FFN-mediated dynamics).

---

## Installation

```bash
pip install torch transformers scikit-learn scipy tqdm matplotlib
```

Optional:
```bash
pip install hdbscan umap-learn wandb
```

---

## Usage

```bash
# Phase 1: empirical metastability detection
python -m phase1.run --models albert-base-v2 --prompts wiki_paragraph

# Phase 2: mechanism identification (requires Phase 1 results)
python -m phase2.run --full
python -m phase2.run --offline results/2026-03-15_18-55-33

# Phase 3: crosscoder training and analysis
python -m phase3.run --albert-only --phase1-dir results/... --phase2-dir results/...
python -m phase3.run --albert-only --skip-cache --skip-train  # re-analyze only
```

---

## Falsification criteria

| Phase | Falsification | Status |
|-------|--------------|--------|
| 1 | No plateaus in cluster count → metastability doesn't survive trained dynamics | **Passed** |
| 3 | Features don't stratify by lifetime → representation not organized along dynamical structure | **Passed** |
| 3 | Decoder directions align with V's eigensubspaces | **Failed (clean null)** |
| 4 | Feature activation plateaus don't align with cluster count plateaus | Not yet tested |
| 4 | No feature or feature-chorus corresponds to cluster membership | Not yet tested |
| 6 | Tuned lens distributions incoherent at early layers | Not yet tested |

Each phase produces publishable results regardless of downstream outcomes. The Phase 3 null — mechanism and representation dissociated under sparsity — is itself a result about how sparse autoencoders relate to the model's internal dynamics.

Readme was written by Claude

---

## Reference

Geshkovski, B., Letrouit, C., Polyanskiy, Y., & Rigollet, P. (2024). *A Mathematical Perspective on Transformers*. [arXiv:2312.10794](https://arxiv.org/abs/2312.10794)
