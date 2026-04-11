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
    --phase1-dir results/2026-03-15_18-55-33 \
    --phase2-dir results/phase2_2026-03-27_05-24-25

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

Default is 4× expansion (8192 for ALBERT-xlarge, 5120 for GPT-2-large). Override with `--n-features`.

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

`run.py` loads Phase 2 projectors from Phase 2's native output format (`ov_projectors_{stem}.npz`). No export step needed:

```bash
python -m phase3.run --albert-only --phase2-dir results/phase2_2026-03-27_05-24-25
```

Projectors are loaded as **low-rank** (top-64 eigenvectors by absolute eigenvalue magnitude). Full-rank projectors are useless for crosscoder alignment: when `rep_frac ≈ 0.5`, any random unit vector projects ~0.5 into each subspace, so the classification carries no signal. Low-rank k=64 matches the crosscoder's sparsity parameter k. Tunable via `k_top` in `_load_artifacts` in `run.py`.

---

## Findings (2026-03-30)

### Run log

| Date | Model | Data | Steps | Features | Notes |
|------|-------|------|-------|----------|-------|
| 2026-03-30 | albert-xlarge-v2 | C4 50k | 100k | 2048 | batch_topk k=64. Note: preset is 4× (8192) — verify checkpoint config. |
| 2026-03-30 | gpt2-large | C4 50k | 100k | 5120 | batch_topk k=64 |

### Prediction 1: Feature lifetime bimodality — CONFIRMED

| Model | Mean lifetime | Short (≤3) | Long (≥L/2) | Bimodal score | Multilayer % |
|-------|--------------|-----------|------------|---------------|-------------|
| ALBERT-xlarge | 5.2 layers | 236 | 343 | 2 | 99.2% (660/665) |
| GPT-2-large | 3.4 layers | 627 | 225 | 1 | 97.8% (1018/1041) |

ALBERT shows a clear bimodal distribution with roughly equal numbers of short- and long-lived features. GPT-2 skews short-lived, consistent with Regime B dynamics concentrated in early repulsive layers. Both pass the multilayer control at >97%, confirming the crosscoder is learning genuine cross-layer features. Positional control is clean: 0% ALBERT, 1% GPT-2.

Lifetimes are computed using **data-driven activation scores** rather than decoder norms — see Bug 2 below for why this is necessary.

### Prediction 2: Decoder directions align with V's eigensubspaces — NULL

Decoder directions are statistically indistinguishable from random with respect to V's top-64 eigenvectors:

```
ALBERT-xlarge (d=2048, k=64, expected random projection = 64/2048 = 0.031):
  Rep energy: mean=0.0314, std=0.0057
  Att energy: mean=0.0316, std=0.0058
  Signal-to-noise: 0.18×
```

This is a clean null — mean equals the random expectation, std is 0.18× of mean. All known measurement artifacts were ruled out during debugging (Bugs 2–5 below). Two interpretations remain:

**Interpretation A (likely):** The crosscoder trained on C4 general web text learned features tracking syntax, frequency, or surface form rather than the dynamical V-eigenstructure. The mechanism (Phase 2) and the representation (Phase 3) are dissociated. V explains *why* energy drops but doesn't organize *what* the model represents.

**Interpretation B (possible):** The 4-prompt eval set is too narrow to activate features tied to metastable dynamics. Training with metastability-rich prompts overrepresented might recover the alignment.

**Diagnostic to distinguish A from B before retraining:**
```python
# Check whether short-lived vs long-lived features separate at all
# on per-layer decoder directions, even without reaching significance
lifetimes = np.array(lt_result["lifetimes"])
short_mask = lifetimes <= 3
long_mask  = lifetimes >= 5

# rep_by_layer shape: (L, F) — computed from decoder directions × top-k eigenvectors
print(f"Short-lived: rep={rep_by_layer[:, short_mask].mean():.4f}  att={att_by_layer[:, short_mask].mean():.4f}")
print(f"Long-lived:  rep={rep_by_layer[:, long_mask].mean():.4f}  att={att_by_layer[:, long_mask].mean():.4f}")
# Equal values → Interpretation A. Separation at specific layer subsets → B.
```

### Predictions 3 & 4: Cluster identity, violation-layer features — NOT RUN

Both require Phase 1 results for ALBERT-xlarge and GPT-2-large. The existing Phase 1 directory (`results/2026-03-15_18-55-33`) only contains ALBERT-base.

---

## Bugs found and fixed

All bugs below are fixed in the current `analysis.py`, `crosscoder.py`, and `run.py`. Documented to prevent reintroduction.

### Bug 1: CUDA tensor → numpy crash (crosscoder.py)

`decoder_norms()` and `decoder_directions()` returned tensors on the model's device. Analysis called `.numpy()` directly.

**Fix:** `.cpu()` added to both return values in `crosscoder.py`.

### Bug 2: All feature lifetimes equal n_layers (analysis.py)

`normalize_decoder()` runs after every optimizer step, making all `W_dec` column norms exactly 1.0. The lifetime threshold `norms[f] > max_norm * 0.1` became `1.0 > 0.1` — always true. Every feature got lifetime = 10. `spearmanr` on a constant array returns NaN.

**Fix:** Replaced `decoder_norms()` with `_compute_feature_layer_scores()`, a new helper computing the mean squared projection of actual residual stream activations onto each decoder direction at each layer, weighted by feature activity. Unaffected by norm normalization. Used in `feature_lifetimes`, `multilayer_fraction`, `positional_control`, and `v_subspace_alignment`.

### Bug 3: np.bool_ not JSON serializable (analysis.py)

`rho > 0.2 and pval < 0.05` with NumPy scalar inputs produces `np.bool_`. Python 3.10's JSON encoder rejects it — confusingly, `np.bool_.__name__` is `"bool"` but it isn't a native Python bool. `prediction_confirmed` in `lifetime_vs_alignment` hit this.

**Fix:** Added `isinstance(obj, np.bool_): return bool(obj)` before the `np.integer` check in `_convert()`. Must be first because `np.bool_` is a subclass of `np.integer` in some NumPy versions.

### Bug 4: V subspace alignment all-Mixed due to full-rank projectors (run.py + analysis.py)

Phase 2's `build_subspace_projectors` constructs `P_attract = U_plus @ U_plus.T` using all eigenvectors with positive eigenvalues. With `rep_frac ≈ 0.57` (ALBERT-xlarge), this is a projector onto 57% of R^2048. Any random unit vector projects ~0.57 into repulsive and ~0.43 into attractive — both below the hardcoded 0.6 threshold — so everything was classified Mixed regardless of true alignment.

**Fix (run.py):** Load low-rank projectors using only the top-k=64 eigenvectors by absolute eigenvalue magnitude. Random projection drops to ~3.1%, giving genuine discriminating power. k=64 matches the crosscoder sparsity parameter.

**Fix (analysis.py):** Replace absolute threshold with relative dominance: `attract_dominance = attract / (attract + repulse)`, classified attractive if >0.6, repulsive if <0.4.

After both fixes the all-Mixed result became the clean null above, confirming it is real.

### Bug 5: V subspace weighting used decoder norms (analysis.py)

Even after Bug 4 was fixed, `v_subspace_alignment` weighted per-layer projections by `decoder_norms()`, which `normalize_decoder()` makes all-ones. For per-layer GPT-2 where `rep_frac` ranges from 0.23 (late layers) to 0.69 (early layers), uniform weighting averages opposite signals and cancels any real alignment.

**Fix:** Replaced `decoder_norms()` weighting with `_compute_feature_layer_scores()` weighting, consistent with the lifetime computation.

### Bug 6: Phase 1 artifact loader expected flat directory (run.py)

`_load_artifacts` looked for `phase1_dir/metrics.json`. Phase 1 stores results in `phase1_dir/{model}_{N}iter_{prompt}/metrics.json`.

**Fix:** Now iterates subdirectories matching the model stem, extracts prompt names via `re.sub(r'^.*?\d+iter_', '', dir_name)` (handles multi-word names like `short_heterogeneous`), and accumulates artifacts across all matching prompts.

---

## Known issues / TODO

- [ ] **Run Phase 1 for albert-xlarge-v2 and gpt2-large** — required for Predictions 3 and 4. Use the same prompts as the current eval set.
- [ ] **Verify ALBERT feature count** — current checkpoint may be 1× (2048 features) rather than the 4× (8192) preset. Check `checkpoints/albert_xlarge_v2/final/config.json`.
- [ ] **Run the A/B diagnostic** before retraining — see the block in Prediction 2 findings. Cheap and determines whether retraining will help.
- [ ] **Retrain with metastability-rich prompts** — if diagnostic suggests Interpretation B, oversample Phase 1 eval prompts in the training mix.
- [ ] **Attach WandB before next run** — `metrics_history.json` exists for retroactive logging. Add live logging to `training.py` callback.
- [ ] **Per-layer SAE baselines** — train single-layer SAEs at layers 12, 24, 36, 48; compare lifetime distributions and V-alignment to crosscoder.
- [ ] **GPT-2-large two-zone crosscoders** — separate dictionaries for repulsive (early) and attractive (late) layer zones.
- [ ] **Merge event feature dynamics** — birth/death of features at cluster merge transitions (requires Phase 1 xlarge results).
- [ ] **Visualization** — feature activation heatmaps across layers × tokens.
- [ ] **Neuronpedia format** — integration with dashboard for feature browsing.
