# Phase 3: Crosscoder Training on Metastable Dynamics

## Quickstart

```bash
# Full pipeline, both models, C4 data (default)
python -m phase3.run

# ALBERT-xlarge only, TinyStories for fast iteration
python -m phase3.run --albert-only --data-source tinystories --n-texts 10000 --total-steps 10000

# GPT-2-large only, full scale
python -m phase3.run --gpt2-only --n-texts 50000

# With Phase 1/2 cross-referencing
python -m phase3.run --albert-only \
    --phase1-dir results/2024-xx-xx/albert-xlarge-v2_wiki_paragraph \
    --phase2-dir results/phase2/albert-xlarge-v2

# Re-analyze without retraining
python -m phase3.run --albert-only --skip-cache --skip-train
```

The script auto-detects existing cached activations and trained checkpoints. If they exist, it skips to the next stage. Use `--force-cache` or `--force-train` to override.

---

## What this does

Trains a **sparse crosscoder** (a sparse autoencoder whose input is the residual stream stacked across multiple layers) on ALBERT-xlarge and GPT-2-large. The crosscoder learns features that span depth, and we test whether those features align with the dynamical structure Phase 1 and Phase 2 identified.

Phase 1 found metastability: tokens cluster, clusters persist across layers (plateaus), then merge in step-drops. Phase 2 found the mechanism: V's mixed-sign eigenspectrum drives the dynamics — attractive directions sustain clusters, repulsive directions cause energy violations. Phase 3 asks: **does this dynamical structure organize the learned representation at the feature level?**

---

## Models

| Model | d_model | Layers sampled | Regime | Why |
|-------|---------|----------------|--------|-----|
| ALBERT-xlarge-v2 | 2048 | 0,6,12,18,24,30,36,42,46,48 | A (attention-mediated) | Shared weights, 100% negative self-interaction, cleanest dynamical signal |
| GPT-2-large | 1280 | 0,4,8,12,16,20,24,28,32,35 | B (FFN-mediated) | Per-layer weights, 100% rescaled-frame elimination, smooth repulsive gradient |

BERT-base is excluded — below the detection threshold on every Phase 2 test.

---

## Predictions

### Prediction 1: Feature lifetime bimodality
Features should split into short-lived (1–5 layers, active at violation/merge events) and long-lived (20+ layers, tracking stable cluster membership). This corresponds to the two-timescale dynamics from Phase 1.

**Metric:** `feature_lifetimes` — histogram of contiguous layer spans.

### Prediction 2: Decoder directions align with V's eigensubspaces
Long-lived features → attractive subspace. Short-lived features → repulsive subspace. V_eff's projectors from Phase 2 provide the test basis.

**Metric:** `v_subspace_alignment` + `lifetime_vs_alignment` (Spearman ρ between lifetime and attractive fraction).

### Prediction 3: Cluster identity features
Some features should fire on exactly the tokens in one HDBSCAN cluster. At merge events, pre-merge cluster features die and post-merge features activate.

**Metric:** `cluster_identity` — recall/FPR per feature vs HDBSCAN labels from Phase 1.

### Prediction 4: Violation-layer features
Features active at Phase 1's violation layers should have decoder directions in V's repulsive subspace. For ALBERT-xlarge, these should correlate with the self-interaction signal (x^T V x < 0).

**Metric:** `violation_layer_features` — z-score of decoder norms at violation vs non-violation layers.

---

## Failure modes (each is informative)

| What happens | What it means |
|---|---|
| Features don't stratify by lifetime | Representation isn't organized along the dynamical structure. Crosscoder found syntax/position instead. |
| Feature plateaus don't align with cluster count plateaus | Features track something other than the metastable configurations. |
| Decoder directions orthogonal to V's subspaces | V's eigenstructure explains energy violations but doesn't organize the learned features. |
| Most features are single-layer | Cross-layer superposition isn't a major factor. Per-layer SAEs would suffice. |
| Long-lived features correlate with position | "Persistence" is positional, not dynamical. |

---

## Controls

| Control | Purpose |
|---|---|
| `multilayer_fraction` | If <20% of features span 3+ layers, the crosscoder is just per-layer SAEs stapled together |
| `positional_control` | Flags features whose activation correlates with token position across prompts |
| Per-layer SAE baselines | Train single-layer SAEs at iterations 12, 24, 36, 48 and compare to crosscoder features |
| `repeated_tokens` prompt excluded | Tests collapse of degenerate distribution, not metastability |

---

## Architecture

```
Encoder:  (batch, L_sampled * d_model) → (batch, n_features)
          shared linear map + pre-encoder bias + ReLU + BatchTopK

Decoder:  (batch, n_features) → (batch, L_sampled, d_model)
          L_sampled separate linear maps from the same sparse code

Loss:     Σ_layers MSE(x_layer, x_hat_layer)
```

BatchTopK activation (not L1) per recent consensus. Decoder columns normalized to unit norm after each optimizer step. Dead features resampled from high-loss inputs. Mixed precision (fp16 forward, fp32 optimizer) enabled by default on CUDA.

### VRAM scaling

| VRAM | n_features | batch | grad_accum | Approx model+optim |
|------|------------|-------|------------|---------------------|
| 10GB (3080) | 8192 (4x) | 512 | 4 | ~4GB |
| 16GB (4080) | 16384 (8x) | 512 | 4 | ~8GB |
| 24GB (3090/4090) | 32768 (16x) | 1024 | 2 | ~16GB |

Default is 4x expansion (8192 for ALBERT-xlarge, 5120 for GPT-2-large). Override with `--n-features`.

---

## Data

**Primary: C4** (allenai/c4, English split, streamed). Diverse web text matching GPT-2's training distribution. Default 50k texts ≈ 1M+ tokens.

**Secondary: TinyStories** (roneneldan/TinyStories). Simple semantic structure. Cluster identity features should be obvious here (characters, settings, actions). Useful as a sanity check.

---

## File structure

```
phase3/
├── __init__.py
├── crosscoder.py       — nn.Module, activation functions, no I/O
├── data.py             — extraction, C4/TinyStories streaming, caching, Dataset
├── training.py         — loop, loss, mixed precision, dead feature resampling
├── analysis.py         — registry of metric functions, standard signature
└── run.py              — plug-and-play CLI, auto-detects existing work
```

Dependency graph: `run → {data, crosscoder, training, analysis}`. `training → crosscoder`. `analysis → crosscoder`. No cycles.

---

## Storage budget (~400GB available)

| Item | Size estimate |
|---|---|
| ALBERT-xlarge activations (10 layers, 1M tokens, fp16) | ~40 GB |
| GPT-2-large activations (10 layers, 1M tokens, fp16) | ~25 GB |
| Crosscoder checkpoints (2 models × ~2GB) | ~4 GB |
| Analysis artifacts | <1 GB |
| **Total** | **~70 GB** |

---

## Phase 2 cross-referencing

`run.py` loads Phase 2 projectors directly from Phase 2's native output format (`ov_projectors_{stem}.npz`). No export step needed — just point to the directory:

```bash
python -m phase3.run --albert-only --phase2-dir results/phase2/albert-xlarge-v2
```

---

## Findings

*This section is updated as experiments complete.*

### Run log

| Date | Model | Data | Steps | Features | Notes |
|------|-------|------|-------|----------|-------|
| | | | | | |

### Results

*(to be filled in)*

---

## Known issues / TODO

- [ ] Per-layer SAE baselines for comparison (single-layer SAEs at a few layers)
- [ ] Centroid trajectory correlation (requires `cluster_tracking.py` output format)
- [ ] GPT-2-large two-zone crosscoders (separate repulsive/attractive zone dictionaries)
- [ ] Merge event feature dynamics (feature birth/death at cluster merge transitions)
- [ ] Cross-term accommodation for ALBERT-xlarge (mixed attractive+repulsive projection)
- [ ] Visualization: feature activation heatmaps across layers × tokens
- [ ] Integration with Neuronpedia dashboard format for feature browsing
