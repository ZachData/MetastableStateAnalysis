# Phase 2 — Energy Violation Mechanism Identification

**Status:** Complete. Extensions (FFN subspace, continuous ΔE, norm confound, revised verdicts) implemented and run across all 35 model×prompt combinations.

---

## Core Question

Phase 1 established that the interaction energy $E_\beta$ drops at structurally active "reset" layers in every model tested, violating the monotonicity theorem of Geshkovski et al. (Theorem 3.4). The token pairs responsible are semantically coherent. Phase 2 asks: **what mechanism causes the energy to decrease?**

---

## Competing Explanations Tested

Four hypotheses, tested in order of parsimony:

1. **Discrete overshoot** — the Euler-step size exceeds the gradient-flow basin of attraction, producing a mechanical energy drop unrelated to any repulsive force.
2. **OV repulsive subspace** — negative eigenvalues of the composed OV matrix ($W_O W_V$) push tokens apart. This is the paper's framework: mixed-sign V creates a tension between attraction (softmax) and repulsion (V eigenstructure).
3. **Attention routing** — attention concentrates on dissimilar targets, creating effective repulsion even when V is net-attractive.
4. **Feed-forward layer** — the FFN drives the energy drop, entirely outside the paper's attention-only model.

---

## Methods

### OV eigendecomposition

The composed OV circuit $V_\text{eff} = \sum_{h=1}^{H} W_O^{(h)} W_V^{(h)}$ is eigendecomposed by two methods: ordered real Schur form (numerically stable for non-normal matrices) and symmetric part $(V + V^\top)/2$ (orthogonal eigenvectors, discards rotation). Methods agree for all models except GPT-2-small (large rotational component) and final layers of GPT-2-small/medium.

### Per-layer trajectory analysis

For per-layer models (GPT-2, BERT), each layer's own V and projectors are used — the initial implementation incorrectly used layer 0's projectors everywhere, which was fixed in `trajectory_perlayer.py`.

Metrics computed per layer: step-size norm, subspace activation energy, per-token self-interaction $x_i^\top V_\text{eff} x_i$, displacement projection onto V's eigensubspaces, and rescaled trajectory via $z_i(L) = (e^{-V})^L x_i(L)$.

### Attn vs FFN decomposition

`decompose.py` splits the residual stream update at each layer into attention and FFN components, computes the energy delta from each at violation layers, and classifies the proximal channel.

### FFN subspace projection (new)

`ffn_subspace.py` projects FFN updates at violation layers onto V's attractive and repulsive eigensubspaces. Per-violation classification: "amplifies_repulsive" (FFN pushes into V's repulsive subspace), "amplifies_attractive", or "orthogonal" (FFN operates outside V's eigenstructure).

### Continuous ΔE correlations (new)

`analysis_extended.py` replaces binary violation indicators with actual energy change magnitude per layer. Spearman correlation of rep_frac vs ΔE tests whether repulsive fraction predicts energy change magnitude, not just binary violation occurrence.

### OV spectral norm confound check (new)

Partial Spearman correlation of OV norm vs violation indicator, controlling for repulsive fraction. Tests whether spectral norm spikes independently predict violations after removing the effect of eigenvalue sign.

### Revised verdict logic (new)

`verdict_v2.py` separates causal question (is V responsible?) from channel question (does energy drop come through attention or FFN?). New categories: V_repulsive_local, V_repulsive_via_FFN, V_repulsive_via_FFN_confirmed, FFN_independent.

---

## Results

### Overshoot: ruled out universally

0% for ALBERT, 0–10% for GPT-2 across all 35 runs. Discrete step size is not the explanation.

### Two regimes of V-repulsive dynamics

**Regime A — Attention-mediated, locally detectable (ALBERT-xlarge).**

5/5 V_repulsive_local. Displacement test passes at 53–85%. Self-interaction $x^\top V x < 0$ at 100% of violation tokens on every run. FFN subspace confirms: 91% mean "amplifies_repulsive" across 5 prompts. Rescaled frame eliminates 37–73% of violations (partial, never 100%). Channel is attention or mixed (mean_attn_frac 45–75%).

Decomposition anomaly: on 3/5 prompts, neither attn nor FFN individually registers as dropping energy (both decompose_*_drop ≈ 0), yet violations occur. The additive decomposition misses the attn-FFN cross-term, which appears to be the dominant mechanism.

**Regime B — FFN-mediated, globally coherent (GPT-2 large/xl, partially medium).**

FFN drops energy at ~100% of violation layers (18/20 GPT-2 runs, exceptions are repeated_tokens prompts). Mean FFN energy fraction: 58–81%. Displacement repulsive fraction moderate (20–60% for large, 37–77% for xl). Rescaled frame eliminates 100% of violations across all 20 GPT-2 runs with no exceptions. FFN subspace: 63% amplifies_repulsive (large), 61% (xl). The causal chain is V → FFN → energy drop.

**Below threshold (ALBERT-base, BERT, GPT-2-small).**

Neither local nor global V-repulsive signal. Rescaling neutral or harmful (ALBERT-base: −3 to −8). Detection threshold: rep_frac × β ≈ 2.8.

### Key numbers across all models

| Model | OV rep frac | β (QK mean) | Rep×β | Rescaled | Channel | FFN amp rep | Regime |
|-------|------------|-------------|-------|----------|---------|-------------|--------|
| ALBERT-base | 0.467 | 1.86 | 0.87 | worse | attn | 0% | below |
| ALBERT-xlarge | 0.569 | 4.99 | 2.84 | partial | attn | 91% | local |
| BERT | 0.509 | 2.86 | 1.46 | ~0 | attn | 18% | below |
| GPT-2-small | 0.431 | 23.0† | — | 100% | mixed | 30% | below‡ |
| GPT-2-medium | 0.474 | 23.1† | — | 100% | FFN | 41% | global |
| GPT-2-large | 0.502 | 4.7 | — | 100% | FFN | 63% | global |
| GPT-2-xl | 0.525 | 4.2 | — | 100% | FFN | 61% | global |

† Layer-0 anomaly inflates mean; typical layers much lower.
‡ Small has 100% rescaled elimination and strong continuous ΔE correlations (ρ = −0.87) but fails the categorical tests. The mechanism is present but below the per-violation classification threshold.

### GPT-2 depth-dependent repulsive gradient

All GPT-2 models: early layers majority-repulsive, late layers majority-attractive. Layer 0 is anomalous everywhere (low rep_frac, extremely high QK norms — the embedding layer). QK norms decay monotonically with depth, creating doubly concentrated repulsive force in early layers (high rep_frac × high effective β).

Crossover layers: small L5, medium L9, large L16, xl L19. Violations concentrate in repulsive zones for large/xl (Spearman ρ(rep_frac, violation_indicator) positive in 18/20 GPT-2 runs, range 0.19–0.61).

### Continuous ΔE correlations

Repulsive fraction predicts energy change magnitude (not just binary violation). GPT-2-small: 4/5 significant (ρ up to −0.87). GPT-2-large: 4/5 significant (ρ ≈ −0.35). GPT-2-xl: 4/5 significant (ρ up to −0.60). GPT-2-medium: 0/5 significant — the weakest model on this test despite being intermediate in scale.

Coupling product (rep_frac × QK norm) adds nothing over rep_frac alone. Significant in only 2/20 GPT-2 runs. QK decay concentrates both factors in early layers, making the product redundant.

### OV spectral norm confound

OV spectral norm independently predicts violations after controlling for rep_frac in: GPT-2-medium 3/5 runs (partial ρ up to −0.71, p = 0.0001), GPT-2-large 3/5, GPT-2-xl 2/5. The norm spikes (medium L10 = 70.75, medium L23 = 98.2) produce large displacements in all directions that get misattributed to the repulsive subspace. The rescaled-frame result is immune to this confound. GPT-2-small: 0/5 confound (uniform norm profile). BERT: 0/5.

### Per-head OV × Fiedler cross-reference

Prediction: repulsive heads → low Fiedler, attractive heads → high Fiedler. GPT-2-medium: ρ = 0.64–0.77, p < 0.01 on 4/5 prompts (strong). BERT: significant on 2/5. GPT-2-large: trending (p = 0.06–0.22). GPT-2-xl: mostly null except short_heterogeneous (ρ = 0.46, p = 0.02).

### Final-layer anomaly

GPT-2-small L11: OV norm 174.3 (22× mean), rep_frac 0.198, methods disagree. GPT-2-medium L23: OV norm 98.2 (5× mean), same pattern. Large/xl have elevated final-layer norms but methods still agree. The final layer is doing something qualitatively different from the rest of the stack — likely the unembedding projection.

---

## Falsification Verdict Distribution (35 runs)

| Verdict | Count | Models |
|---------|-------|--------|
| V_repulsive_local | 9 | ALBERT-xlarge (5), GPT-2-large (2), GPT-2-xl (3) |
| V_repulsive_via_FFN | 13 | GPT-2-small (4), GPT-2-medium (4), GPT-2-large (2), GPT-2-xl (1), ALBERT-base (0†), BERT (0) |
| V_repulsive_via_FFN_confirmed | 7 | GPT-2-small (1‡), GPT-2-medium (1), GPT-2-large (2), GPT-2-xl (3) |
| FFN_independent | 2 | ALBERT-base (2) |
| mixed_or_unattributed | 7 | ALBERT-base (2), BERT (5) |
| overshoot_dominant | 0 | — |
| no_violations | 0 | — |

† ALBERT-base has no continuous ΔE or norm confound tests (shared weights, no per-layer profile).
‡ Bug: GPT-2-small short_heterogeneous gets _confirmed with channel = "attention." The upgrade should require channel ≠ "attention."

---

## Known Issues

1. **verdict_v2 bug**: _confirmed upgrade does not check channel consistency. GPT-2-small short_heterogeneous gets V_repulsive_via_FFN_confirmed despite channel = "attention."
2. **verdict_v2 bug**: FFN_independent triggers on n=1 violations (ALBERT-base paper_excerpt, sullivan_ballou). Should require n ≥ 3.
3. **ALBERT-xlarge decomposition**: additive Δ_attn + Δ_ffn accounting misses the cross-term. Need three-way decomposition.
4. **Coupling product**: confirmed useless. Can be dropped from future analysis.

---

## Modules

### `weights.py` — Weight extraction and decomposition
Composed OV circuit, eigendecomposition (Schur + symmetric), subspace projectors, QK spectral norms, per-head OV matrices. No inference.

### `trajectory.py` — Offline trajectory analysis (shared-weight path)
Step-size norms, subspace activation, self-interaction, displacement projection, centroid projection, rescaled trajectory. Uses single V for ALBERT.

### `trajectory_perlayer.py` — Per-layer-correct trajectory analysis
Drop-in replacement for per-layer models. Uses each layer's own V and projectors.

### `layer_v_events.py` — Per-layer V vs Phase 1 event overlay
Correlates depth-dependent repulsive fraction with violations, merges, cluster structure. Zone classification (repulsive/transition/attractive).

### `head_ov_analysis.py` — Per-head OV × Fiedler cross-reference
Per-head eigendecomposition, Fiedler loading from Phase 1, Spearman correlation.

### `decompose.py` — Attn/FFN forward-pass decomposition
Splits residual update, computes per-component energy deltas, saves attn_deltas.npz and ffn_deltas.npz.

### `ffn_subspace.py` — FFN update projection onto V subspaces
Projects FFN residual updates onto V's eigensubspaces. Per-violation classification. Z-score comparison of FFN-repulsive fraction at violation vs population layers.

### `analysis.py` — Cross-reference statistical tests
Violation classification, z-scores, plateau characterization, rescaled comparison, merge prediction.

### `analysis_extended.py` — Continuous ΔE, norm confound, adaptive zones
Continuous Spearman correlations, partial correlations controlling for rep_frac, adaptive zone thresholds (median ± 0.5×MAD).

### `reporting.py` — Terminal summaries
Per-run output and machine-readable verdict JSON.

### `verdict_v2.py` — Revised falsification logic
Separates causal (V responsible?) from channel (attn or FFN?). New verdict categories.

### `run.py` — CLI entry point
`--full` (model load + decompose), `--offline` (saved data only). Wires all modules including new extensions.
