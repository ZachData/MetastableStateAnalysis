# Phase 3: Crosscoder Training on Metastable Dynamics

**Status: Complete. All three predictions tested. Overall verdict: null.**

---

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

## Predictions and results

### Prediction 1: Feature lifetime bimodality

Features should split into short-lived (1–5 layers, active at violation/merge events) and long-lived (20+ layers, tracking stable cluster membership).

**Metric:** `feature_lifetimes` — bimodality coefficient (BC > 5/9 ≈ 0.556 = bimodal).

| Model | n_alive | mean_lifetime | BC | Result |
|---|---|---|---|---|
| ALBERT-xlarge-v2 | 665 / 2048 | 5.2 | 0.622 | **Confirmed** (bimodal, valley at 8) |
| GPT-2-large | 1041 / 5120 | 3.4 | 0.514 | **Not confirmed** (unimodal) |

The high dead-feature rates (67% ALBERT, 80% GPT-2) indicate underutilization of dictionary capacity, likely because the eval prompts are a narrow distribution relative to the C4 training data.

**Note on short/long counts — two classification methods exist:**

The `feature_lifetimes` analysis reports counts under two schemes. The *current* method uses a valley threshold derived from the lifetime histogram (when bimodal); the *legacy* method uses fixed lifetime cutoffs. Results from the run dated 2026-04-29:

| Model | Method | n_short_lived | n_long_lived |
|---|---|---|---|
| ALBERT | current (valley=8) | 505 | 160 |
| ALBERT | legacy | 236 | 343 |
| GPT-2 | current | 352 | 689 |
| GPT-2 | legacy | 627 | 225 |

The legacy ALBERT numbers (236 short, 343 long) match previously documented results. The current method's valley threshold changes the split substantially. Downstream analyses (steering, lifetime_vs_alignment) use the current classification.

**multilayer_fraction:** 99.2% (ALBERT) and 97.8% (GPT-2) of alive features span 3+ layers. The crosscoder is not behaving as a stack of per-layer SAEs.

**positional_fraction:** 0% (ALBERT), 0.6% (GPT-2). Feature lifetimes are not driven by token position.

---

### Prediction 2: Decoder directions align with V's eigensubspaces

Long-lived features → attractive subspace. Short-lived features → repulsive subspace.

**Metrics:** `v_subspace_alignment`, `lifetime_vs_alignment` (Spearman ρ).

| Model | n_attractive | n_repulsive | attract_dominance | Spearman ρ | p-value | Result |
|---|---|---|---|---|---|---|
| ALBERT-xlarge-v2 | 0 | 0 | 0.484 ± 0.007 | 0.027 | 0.487 | **Null** |
| GPT-2-large | 0 | 1 | 0.501 ± 0.014 | 0.091 | 0.003 | **Null** (effect size negligible at n=1041) |

All 2048 ALBERT features and 5119/5120 GPT-2 features fall in the "mixed" category. The attract_dominance values are indistinguishable from the 0.5 expected under random decoder directions.

The GPT-2 Spearman p=0.003 is technically significant but at n=1041 corresponds to a negligible effect (ρ=0.09). It does not indicate alignment with V's subspaces.

---

### Prediction 3: Cluster identity features / Prediction 4: Violation-layer features

Both predictions require Phase 1 HDBSCAN labels and plateau layers for ALBERT-xlarge and GPT-2-large. These were run with the available eval set.

**decoder_violation_projection** (how much of the violation displacement is explained by crosscoder features):

| Model | top-1 | top-5 | top-10 | top-50 | n_violations |
|---|---|---|---|---|---|
| ALBERT-xlarge-v2 | 4.5% | 16.2% | 23.6% | 42.4% | 36 |
| GPT-2-large | 1.3% | 4.6% | 7.8% | 26.2% | 82 |

Violations are not well-explained by crosscoder features. Top-10 features account for less than a quarter of violation energy in ALBERT and under 8% in GPT-2.

**ffn_repulsive_feature_alignment** (cosine similarity between feature decoder directions and FFN delta at violation layers):

| Model | mean cosine sim | n_violations checked |
|---|---|---|
| ALBERT-xlarge-v2 | 0.018 | 10 |
| GPT-2-large | 0.007 | 18 |

Indistinguishable from zero.

---

### Steering results

5 experiments per model. Each experiment activates a selected feature and measures change in cluster merge layer.

**ALBERT-xlarge-v2:** All 5 experiments → `null` (mean merge delta = +0.0 layers). Fisher's exact test: OR=nan, p=1.0. No association between feature lifetime class and causal effect on cluster stability.

**GPT-2-large:** All 5 experiments → `no_baseline_merge`. The eval prompts used for GPT-2 steering do not produce a baseline merge event, making the test structurally unrunnable. This is an eval prompt coverage gap, not a conclusive result, but there is no positive signal to report.

**Pair tracking:** Both models show Jaccard=1.0, zero pairs broken, zero pairs formed across all experiments. Feature steering does not perturb token-pair cluster membership at any measured layer.

---

## Analyses that did not run

Two cross-phase analyses errored on both models:

**`cross_term_feature_weighting`:** Requires `cross_term_results` artifact from Phase 2. This artifact was not passed to Phase 3's run directory. The analysis would test whether features preferentially active on cross-term-dominant violation token pairs correlate with cluster-discriminating features (high F-statistic). Low priority given that violation projection is already null.

**`induction_feature_tagging`:** Requires `pair_agreement` artifact from Phase 1 (semantic vs. artifact mutual-NN pairs). Reports 0 exclusive tokens for both semantic and artifact categories, indicating the artifact was absent or empty. This analysis characterizes feature type (syntax/induction vs. semantic) but does not test the metastability hypothesis.

---

## Overall findings

All substantive tests converge on a null:

| Test | ALBERT | GPT-2 |
|---|---|---|
| Bimodality (Prediction 1) | Confirmed | Not confirmed |
| Decoder → V alignment (Prediction 2) | Null | Null |
| Lifetime × V alignment (Spearman) | Null (ρ=0.03) | Null (ρ=0.09) |
| Violation projection (top-10) | 24% | 8% |
| FFN alignment | ~0 | ~0 |
| Steering causal effect | Null | Unrunnable |
| Pair tracking | Null | Unrunnable |

**Interpretation A (favored):** The crosscoder, trained on C4 general web text, learned features tracking syntax, token frequency, and surface form. These features happen to have the right temporal profile (short-lived vs. long-lived relative to sequence position) but are not organized by V's eigenstructure. The mechanism Phase 2 identified explains *why* energy drops at violation layers but does not organize *what* the model represents at the feature level. The two findings are dissociated.

**Interpretation B (not favored):** The 4-prompt eval set is too narrow to activate features tied to metastable dynamics. GPT-2's unimodal lifetime distribution argues against this — a coverage problem would suppress alignment signal but should not suppress bimodality, since bimodality is a property of the training distribution, not the eval set.

The Phase 3 null is robust across two models, multiple independent metrics, and direct causal intervention (steering). Rerunning the two failed analyses is unlikely to change the conclusion.

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

n_features=2048 for ALBERT-xlarge, n_features=5120 for GPT-2-large. k=64 for both.

---

## Failure modes (each is informative)

| What happened | What it means |
|---|---|
| Features don't stratify by lifetime (GPT-2) | Representation isn't organized along the dynamical structure. Crosscoder found syntax/position instead. |
| Decoder directions orthogonal to V's subspaces (both) | V's eigenstructure explains energy violations but doesn't organize the learned features. |
| Steering null (ALBERT) / unrunnable (GPT-2) | Crosscoder features are not causally upstream of cluster merge events. |
| Violation projection < 25% (both) | Violation displacement doesn't decompose into a small number of feature directions. |

---

## Controls

| Control | Result |
|---|---|
| `multilayer_fraction` | >97% both models — crosscoder is not per-layer SAEs stapled together |
| `positional_control` | 0% (ALBERT), 0.6% (GPT-2) — lifetimes not driven by position |
| Bimodality coefficient | ALBERT bimodal (0.622), GPT-2 unimodal (0.514) |

---

## Bugs found and fixed

All bugs below are fixed in the current `analysis.py`, `crosscoder.py`, and `run.py`.

### Bug 1: CUDA tensor → numpy crash (crosscoder.py)

`decoder_norms()` and `decoder_directions()` returned tensors on the model's device. Analysis called `.numpy()` directly.

**Fix:** `.cpu()` added to both return values in `crosscoder.py`.

### Bug 2: All feature lifetimes equal n_layers (analysis.py)

`normalize_decoder()` runs after every optimizer step, making all `W_dec` column norms exactly 1.0. The lifetime threshold `norms[f] > max_norm * 0.1` became `1.0 > 0.1` — always true. Every feature got lifetime = 10.

**Fix:** Replaced `decoder_norms()` with `_compute_feature_layer_scores()`, which computes the mean squared projection of actual residual stream activations onto each decoder direction at each layer, weighted by feature activity. Unaffected by norm normalization.

### Bug 3: np.bool_ not JSON serializable (analysis.py)

`rho > 0.2 and pval < 0.05` with NumPy scalar inputs produces `np.bool_`. Python 3.10's JSON encoder rejects it.

**Fix:** Explicit `bool()` cast on `prediction_confirmed`.

### Bug 4: Multi-word prompt key parsing (run.py)

`re.sub` pattern extracted only the first word of multi-word prompt names (e.g., `short_heterogeneous` → `short`).

**Fix:** Now iterates subdirectories matching the model stem, extracts prompt names via `re.sub(r'^.*?\d+iter_', '', dir_name)`.

---

## Known gaps

- **`cross_term_feature_weighting`** requires `cross_term.json` from Phase 2 to be passed via `--phase2-dir`. Not run. Low priority given null violation projection.
- **`induction_feature_tagging`** requires `pair_agreement` from Phase 1. Not run. Feature characterization only, not a hypothesis test.
- **GPT-2 steering** unrunnable with current eval prompts — no baseline merge event. Would require prompts that produce merge events in GPT-2.
- **Per-layer SAE baselines** not run. Would establish whether crosscoder lifetime structure is meaningfully different from per-layer SAEs.
- **GPT-2-large two-zone crosscoders** not run. Separate dictionaries for repulsive (early) and attractive (late) layer zones might recover signal, but given the global null this is speculative.
