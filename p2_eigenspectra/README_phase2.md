# Phase 2 — V Eigenspectrum Characterization

## What this does

Phase 1 identified energy violations: layers where the interaction energy $E_\beta$ fails to decrease monotonically, consistent with the V-repulsive mechanism predicted by Geshkovski et al. Phase 2 asks: **is V's mixed-sign eigenspectrum the actual cause?**

Three causal tests are applied:

1. **Displacement test (local):** At each violation layer, project the token displacement $\Delta x = x^{(l+1)} - x^{(l)}$ onto V's repulsive vs attractive eigensubspace. If $>50\%$ of violation-layer displacements are repulsive-dominant, V is locally detectable.

2. **Rescaled frame (global):** Apply the exponential rescaling $z = e^{-tV}x$ to remove V's effect. If violations disappear in the rescaled frame, V is globally causal regardless of local detection.

3. **FFN/attention decomposition (channel):** Split each layer's residual update into attention and FFN components; compute per-component energy deltas. Classifies whether V's causal effect is mediated through the attention path or the FFN path.

A continuous V-score combines these:

```
V-score = 0.40 × rescaled_frac
        + 0.25 × frac_repulsive_disp
        + 0.20 × frac_ffn_amplifies_repulsive
        - 0.15 × |ov_norm_partial_rho|
```

Scores above ~0.5 consistently correspond to `V_repulsive_via_FFN_confirmed` or `V_repulsive_local` verdicts.

---

## Running

```bash
# Full pipeline — loads model, runs decompose, runs all analyses
python -m p2_eigenspectra.run_2 --full

# Offline only — uses saved activations/deltas from a previous full run
python -m p2_eigenspectra.run_2 --offline --phase1-dir results/2026-04-23_18-30-06

# Single model
python -m p2_eigenspectra.run_2 --full --models albert-xlarge-v2

# Specific prompt
python -m p2_eigenspectra.run_2 --full --prompts repeated_tokens wiki_paragraph
```

Outputs are written to a timestamped directory under `results/`. The cross-run summary is written to `p2_eigenspectra_cross_run.summary.txt` and `.json`.

---

## Falsification verdict categories

| Verdict | Condition | Interpretation |
|---|---|---|
| `V_repulsive_local` | `frac_repulsive > 0.5` | V directly detectable via displacement test |
| `V_repulsive_via_FFN` | `rescaled_frac > 0.8` and `ffn_frac_drop > 0.5` | V globally causal, FFN is the proximal channel |
| `V_repulsive_via_FFN_confirmed` | as above, plus `frac_ffn_amplifies_repulsive > 0.5` and `channel == "FFN"` | Strongest evidence: FFN pushes into V's repulsive subspace |
| `V_repulsive_via_attn` | `rescaled_frac > 0.8` and `ffn_frac_drop ≤ 0.5` | V globally causal, attention is the proximal channel |
| `FFN_independent` | `ffn_frac_drop > 0.5` and `rescaled_frac < 0.2` and `n_decomposed ≥ 3` | FFN causes violations independently of V |
| `mixed_or_unattributed` | none of the above | No single mechanism dominates |
| `overshoot_dominant` | `frac_overshoot > 0.5` | Energy violations due to step-size overshoot, not V |

Note: `V_repulsive_via_attn` exists in the verdict logic and is distinct from `V_repulsive_via_FFN`, but fires on zero runs across the full 35-run evaluation. The condition (rescaling helps + attention dominates FFN drops) does not arise empirically: when the rescaled frame eliminates violations, FFN is always the proximal dropper.

Classification priority order: `V_repulsive_local` takes precedence over all rescaling-based verdicts. A run where `frac_repulsive > 0.5` gets `V_repulsive_local` regardless of what the rescaled frame shows.

---

## Results

### Overshoot: ruled out universally

`frac_overshoot = 0` for ALBERT, 0–10% for GPT-2 and BERT across all 35 runs. Discrete step size is not the explanation for energy violations.

### Two regimes of V-repulsive dynamics

**Regime A — locally detectable (ALBERT-xlarge, GPT-2-xl, GPT-2-large partial)**

The displacement test passes: `frac_repulsive > 0.5` at violation layers. V's repulsive eigenstructure is directly detectable in the token trajectory without needing the rescaled frame.

ALBERT-xlarge: 5/5 `V_repulsive_local`. v_scores 0.60–0.73. `frac_ffn_amplifies_repulsive` 0.71–1.00. Self-interaction $x^\top V x < 0$ at 100% of violation tokens on every run. Channel is attention-dominant (mean_attn_frac > mean_ffn_frac), consistent with ALBERT's shared-weight architecture where the OV circuit acts directly without layer-specific FFN amplification.

GPT-2-xl: 3/5 `V_repulsive_local` + 2/5 `V_repulsive_via_FFN_confirmed`. v_scores 0.51–0.74. The two `_confirmed` runs (short_heterogeneous, wiki_paragraph) have moderate `frac_repulsive` (0.371–0.441) just below threshold, with FFN confirmed as the channel.

GPT-2-large: 2/5 `V_repulsive_local` + 1/5 `V_repulsive_via_FFN_confirmed` + 2/5 `mixed_or_unattributed`. The two mixed runs (short_heterogeneous, wiki_paragraph) have v_scores 0.455–0.486 — borderline, not clean reversals. `frac_repulsive` (0.467–0.409) just misses the 0.5 threshold and `rescaled_frac` doesn't reach 0.8.

**Regime B — globally coherent, FFN-mediated (GPT-2-small, GPT-2-medium)**

The displacement test fails (`frac_repulsive < 0.5`), but the rescaled frame eliminates violations (`rescaled_frac > 0.8`). FFN is the proximal dropper. V is the cause; FFN is the channel.

GPT-2-small: 4/5 `V_repulsive_via_FFN` + 1/5 `mixed_or_unattributed` (repeated_tokens, v_score 0.28). v_scores 0.42–0.57 on the 4 detected runs. `frac_ffn_amplifies_repulsive` 0.17–0.60, lower than larger models.

GPT-2-medium: 4/5 `V_repulsive_via_FFN` + 1/5 `V_repulsive_local` (repeated_tokens, `frac_repulsive` = 0.545). All 5 runs have channel = FFN. v_scores 0.43–0.60. `frac_ffn_amplifies_repulsive` 0.29–0.55.

GPT-2-medium repeated_tokens (v_score 0.598, `frac_repulsive` = 0.545) sits at the boundary between regimes: the displacement test marginally passes and the FFN energy decomposition also supports V. Both regimes are consistent for this run.

**Below threshold (ALBERT-base, BERT)**

ALBERT-base: 2/5 `V_repulsive_local` (short_heterogeneous, repeated_tokens), 3/5 `mixed_or_unattributed`. The two positive runs have v_scores 0.167 and 0.250 — weak. Low violation counts (n=1 on 2 runs, n=6 on the positive runs) make the signal fragile. No per-layer decompose data (shared weights), so channel is always "attention" and the FFN path is unresolvable.

BERT-base: 4/5 `mixed_or_unattributed` + 1/5 `FFN_independent` (repeated_tokens). v_scores 0.039–0.107. `ov_norm_partial_rho` consistently negative (−0.34 to −0.49), indicating the OV norm confound is significant. The norm confound + low rep_frac means the V-repulsive signal is not credibly separable from spectral-norm effects.

### OV spectral norm confound

`ov_norm_partial_rho` measures Spearman correlation of OV norm vs violation indicator after controlling for `rep_frac`. Significant negative values mean norm spikes predict violations independently of the repulsive-fraction signal.

Present and significant across most GPT-2 models (partial ρ up to −0.71 for GPT-2-medium, with p < 0.05 on multiple runs). Norm spikes in early and final layers produce large displacements that partly get attributed to the repulsive subspace. The rescaled-frame result is immune to this confound; the `V_repulsive_local` verdict is more vulnerable to it.

Notable outliers: GPT-2-medium L23 OV norm ≈ 98.2 (5× mean); GPT-2-small L11 OV norm ≈ 174.3 (22× mean). These are likely the unembedding projections at final layers. ALBERT models: confound not computed (no per-layer norm profile under shared weights).

### Per-head OV × Fiedler cross-reference

Prediction: repulsive heads → low Fiedler, attractive heads → high Fiedler. GPT-2-medium: ρ = 0.64–0.77, p < 0.01 on 4/5 prompts. BERT: significant on 2/5. GPT-2-large: trending (p = 0.06–0.22). GPT-2-xl: mostly null except short_heterogeneous (ρ = 0.46, p = 0.02).

---

## Falsification Verdict Distribution (35 runs, current results)

| Verdict | Count | Models |
|---|---|---|
| `V_repulsive_local` | 13 | ALBERT-xlarge (5), GPT-2-xl (3), GPT-2-large (2), GPT-2-medium (1), ALBERT-base (2†) |
| `V_repulsive_via_FFN` | 8 | GPT-2-small (4), GPT-2-medium (4) |
| `V_repulsive_via_FFN_confirmed` | 3 | GPT-2-xl (2), GPT-2-large (1) |
| `FFN_independent` | 1 | BERT (1‡) |
| `mixed_or_unattributed` | 10 | ALBERT-base (3), BERT (4), GPT-2-small (1), GPT-2-large (2) |
| `overshoot_dominant` | 0 | — |
| `no_violations` | 0 | — |

† ALBERT-base positive runs (v_scores 0.167, 0.250) are weak signals. No per-layer decompose path available under shared weights.  
‡ BERT repeated_tokens: `ffn_frac_drop` = 0.400, `rescaled_frac` low, `n_decomposed` = 5. Borderline; the FFN signal is present but uncorroborated by the rescaled frame.

---

## Modules

### `weights.py` — Weight extraction and decomposition
Composed OV circuit, eigendecomposition (Schur + symmetric), subspace projectors, QK spectral norms, per-head OV matrices. No inference.

### `trajectory.py` — Offline trajectory analysis (shared-weight path)
Step-size norms, subspace activation, self-interaction, displacement projection, centroid projection, rescaled trajectory. Uses single V for ALBERT.

### `trajectory_perlayer.py` — Per-layer-correct trajectory analysis
Drop-in replacement for per-layer models (GPT-2, BERT). Uses each layer's own V and projectors.

### `decompose.py` — Attn/FFN forward-pass decomposition
Splits residual update into attention and FFN components at each layer, computes per-component energy deltas. Saves `attn_deltas.npz` and `ffn_deltas.npz`. Required for channel classification and `_confirmed` upgrade.

### `layer_v_events.py` — Per-layer V vs Phase 1 event overlay
Correlates depth-dependent repulsive fraction with violations, merges, cluster structure. Zone classification (repulsive/transition/attractive) using adaptive thresholds (median ± 0.5×MAD).

### `head_ov_analysis.py` — Per-head OV × Fiedler cross-reference
Per-head eigendecomposition, Fiedler loading from Phase 1, Spearman correlation between head repulsive fraction and Fiedler score.

### `ffn_subspace.py` — FFN update projection onto V subspaces
Projects FFN residual updates onto V's eigensubspaces at violation layers. Per-violation classification: amplifies_repulsive / amplifies_attractive / orthogonal. Z-score comparison of FFN-repulsive fraction at violation vs population layers.

### `cross_term_analysis.py` — Attn/FFN cross-term decomposition
Three-way decomposition of the residual update: `Δ_attn + Δ_ffn + Δ_cross`. The additive two-way decomposition misses the cross-term, which is the dominant energy-drop mechanism in ALBERT-xlarge on several prompts. This module computes the cross-term contribution explicitly.

### `analysis.py` — Core cross-reference statistical tests
Violation classification, z-scores, plateau characterization, rescaled comparison, merge prediction.

### `analysis_extended.py` — Continuous ΔE, norm confound, adaptive zones, verdict
Continuous Spearman correlations (rep_frac vs ΔE magnitude), partial correlations controlling for OV norm, adaptive zone thresholds, attractive-zone violation breakdown. **This is the single source of truth for the categorical falsification verdict** — `_classify()` in this module is what determines the final verdict string.

### `verdict_v2.py` — Verdict logic and V-score
Contains `build_verdict_v2` (used when calling from the monolithic analysis dict path) and `build_v_score`. The `_classify` function in `analysis_extended.py` is authoritative; `verdict_v2._classify` is kept for back-compat via a shim.

### `subresult.py` — Per-subexperiment result container
Typed container holding raw results and `verdict_contribution` dict for each subexperiment. Used by `subexperiments.py` to pass structured results to the verdict assembler.

### `subexperiments.py` — Subexperiment registry and orchestration
Registry pattern connecting each analysis module to the verdict assembler. Controls which subexperiments run in `--full` vs `--offline` mode, handles per-subexperiment errors without aborting the run.

### `subexp_wrappers.py` — Thin wrappers around analysis modules
One wrapper per subexperiment. Translates raw module outputs into the `SubResult` contract.

### `threshold_analysis.py` — Beta-sweep and threshold sensitivity
Sweeps β values to characterize how violation counts and verdict stability change with the energy threshold. Used to assess whether verdict assignments are robust to β choice.

### `head_ablation.py` — Per-head causal ablation
Ablates individual attention heads and measures the effect on violation counts and energy profiles. Identifies which heads are necessary for the violation pattern.

### `reporting.py` — Terminal summaries and machine-readable output
Per-run output and the `p2_eigenspectra_cross_run.summary.txt` / `.json` files.

### `run_2.py` — CLI entry point
`--full` (model load + decompose + all analyses), `--offline` (saved data only). Wires all subexperiments through the registry.

---

## Known Issues

All four bugs documented in the previous README version are resolved in the current code:

- **Bug 1** (`_confirmed` upgrade firing on attention channel): Fixed. `analysis_extended._classify` requires `channel == "FFN"` explicitly. Previously `channel != "attention"` allowed the upgrade when decompose coverage was zero and channel was `"unknown"`.
- **Bug 2** (`FFN_independent` on n=1 violations): Fixed. Guard now uses `n_decomposed ≥ 3` (count of violations that went through the decompose path) rather than `n_violations ≥ 3`.
- **Bug 3** (ALBERT-xlarge cross-term): Fixed. `cross_term_analysis.py` computes the three-way `Δ_attn + Δ_ffn + Δ_cross` decomposition. The cross-term is the dominant mechanism for ALBERT-xlarge on several prompts where neither attn nor FFN individually registers as the energy dropper.
- **Bug 4** (coupling product): Confirmed useless and removed from the analysis pipeline.

Open:

- **GPT-2-large borderline runs**: short_heterogeneous and wiki_paragraph have v_scores 0.455–0.486. Neither test passes cleanly. A finer-grained β sweep might resolve the ambiguity, or these may be genuine regime-boundary cases.
- **`V_repulsive_via_attn` never fires**: The code path exists and is correct. Empirically, when the rescaled frame helps (rescaled_frac > 0.8), FFN always dominates the energy drop (ffn_frac_drop > 0.5). If a future model or prompt type changes this, the verdict is ready.
- **ALBERT-base cross-term**: Shared weights mean no per-layer decompose. The FFN path is unresolvable for ALBERT-base. The two positive runs get `V_repulsive_local` but the channel remains "attention" by default — not because attention is confirmed, but because the decompose path doesn't differentiate.
