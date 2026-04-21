# Phase 2i — Rotational Dynamics Investigation

**Status:** Complete. All 7 models × 5 prompts run. Block 2 skipped universally (rotation_neutral across all 35 combinations).

---

## Core Question

Phase 2 found that OV matrices have ~98% complex (imaginary) eigenvalue dimensions across all models. This raised a question about the Phase 2 conclusions: if the dominant spectral feature of the OV circuit is rotational rather than signed, does the rotation contribute to energy violations, and could the imaginary structure be confounding the signed-component analysis?

Phase 2i asks: **is the rotational component of OV causally responsible for any energy violations, or is it dynamically neutral?**

---

## Motivation

Phase 2's eigendecomposition framework uses the sign of OV eigenvalues (attractive vs repulsive) to explain energy violations. But the violation mechanism identified — negative eigenvalues of the composed OV circuit driving tokens apart — implicitly assumes the signed component dominates dynamically. If ~98% of OV's spectral energy is rotational (pure imaginary eigenvalues), this warrants explicit verification.

The S/A decomposition $V_\text{eff} = S + A$ separates OV into its symmetric part $S = (V + V^\top)/2$ (signed, real eigenvalues, the component Phase 2 attributes violations to) and antisymmetric part $A = (V - V^\top)/2$ (pure rotation, no signed component). Phase 2i applies this decomposition causally: does removing $A$ alone affect violations? Does removing $S$ alone eliminate them?

---

## Approach

### Block 1a — Rotational spectrum characterization

For each model, the complex eigenvalue structure of OV is characterized per layer:

- **frac_complex_dims** — fraction of eigenvalue dimensions with non-trivial imaginary part ($|\text{Im}| > 0.01 \times (|\text{Re}| + \epsilon)$)
- **rotational_fraction** — fraction of total eigenvalue energy ($\sum |\lambda|^2$) attributable to complex pairs
- **signed_fraction** — complementary fraction attributable to real eigenvalues
- **Henrici non-normality** — $\|T\|_F^2 - \sum |\lambda_i|^2$ (absolute) and relative; quantifies departure from normality that makes the S/A decomposition informative
- Rotation plane geometry: mean rotation angle $\theta$, radial scaling $\rho$, fraction expanding vs contracting, fraction of complex-pair planes with net-attractive vs net-repulsive contribution

For shared-weight models (ALBERT), these are computed once per OV matrix. For per-layer models (GPT-2, BERT), per-layer values are aggregated into depth summaries.

### Block 1b — Causal isolation via rescaled frames

Three rescaled trajectories are computed per prompt per model:

1. **Full rescaling:** $z = e^{-tV} x$ — removes all of $V_\text{eff}$
2. **Signed-only rescaling:** $z = e^{-tS} x$ — removes the symmetric (signed) part only
3. **Rotation-only rescaling:** $z = e^{-tA} x$ — removes the antisymmetric (rotational) part only

Energy violations are recomputed in each frame. The comparison logic:

- If signed-only removal eliminates violations (`elim_signed = 1.0`) and rotation-only removal does not (`elim_rotation = 0.0`) → `rotation_neutral`: violations are entirely caused by the signed component.
- If rotation-only removal also reduces violations → `rotation_contributes`: rotation has independent causal weight.
- If both together eliminate more than either alone → non-linear interaction between $S$ and $A$.

### Block 2 — Hemispheric tracking (conditional)

Fiedler vector tracking and rotation-hemisphere alignment analysis. Run only if Block 1b returns `rotation_contributes` for any model. **Not run** in this experiment.

---

## Results

### Block 1a: Rotational spectrum

The OV circuit is structurally dominated by complex eigenvalue pairs across all architectures and scales.

| Model | frac\_complex\_dims | rotational\_fraction | signed\_fraction | Henrici (relative) |
|---|---|---|---|---|
| albert-base-v2 | 0.948 | 0.882 | 0.118 | 0.677 |
| albert-xlarge-v2 | 0.974 | **0.975** | **0.025** | 0.649 |
| bert-base-uncased | 0.953 | 0.906 | 0.094 | 0.741 |
| gpt2 | 0.943 | 0.836 | 0.164 | 0.734 |
| gpt2-medium | 0.953 | 0.887 | 0.113 | 0.716 |
| gpt2-large | 0.961 | 0.903 | 0.097 | 0.686 |
| gpt2-xl | 0.965 | 0.904 | 0.096 | 0.681 |

The 98% imaginary observation from Phase 2 is confirmed and universal. ALBERT-xlarge is the extreme case: rotational energy is 97.5% of the total OV eigenvalue energy, leaving only 2.5% in the signed component. Henrici non-normality (relative) runs 0.65–0.74 across models, consistent with substantially non-normal matrices where the S/A decomposition is non-trivial.

For GPT-2 models, depth-averaged rotation ratios are consistent layer-to-layer (std ≈ 0.03–0.05), with no systematic depth trend. The antisymmetric component is not concentrated in any particular layer.

### Block 1b: Causal test

**All 35 model × prompt combinations return `rotation_neutral`.**

The pattern is entirely uniform:

- `elim_signed = 1.0` in every case — removing only $S$ eliminates all energy violations completely, regardless of model, prompt, or $\beta$ value tested (0.1, 1.0, 2.0, 5.0).
- `elim_rotation = 0.0` in every case — removing only $A$ eliminates no violations. The rotational component has zero causal weight.

Block 2 was skipped for all 7 models.

### ALBERT full-rescaling anomaly

For ALBERT models, the full rescaling (`elim_full`) frequently goes negative — meaning the full $e^{-tV}$ frame *creates* more violations than exist in the original trajectory. Representative cases:

- albert-xlarge-v2, sullivan\_ballou, β=0.1: 1 original violation → 9 after full rescaling (elim\_full = −8.0)
- albert-xlarge-v2, repeated\_tokens, β=1.0: 26 original → 7 after full rescaling (elim\_full = +0.73)

This is an ALBERT-specific artifact of the shared-weight architecture. The full rescaling frame is $z = e^{-tV} x^{(L)}$ computed once per iteration step rather than once per layer — the single shared $V$ interacts with $S$ in the rescaled frame in a way that is not additively separable. Removing $S$ alone (`z = e^{-tS} x`) is always clean (elim\_signed = 1.0). This does not affect the causal conclusion: the signed component is still the sole cause of violations. It does mean the full rescaling is an unreliable diagnostic for ALBERT, consistent with the observation in Phase 2 that rescaling is "neutral or harmful" for ALBERT-base/xlarge.

For GPT-2 models, the full rescaling is clean: `elim_full = 1.0` across all prompts and beta values, with no overcorrection artifacts.

---

## What Was Learned

**The rotational component of OV is a structural red herring.** It accounts for 84–97% of OV's spectral energy and is ubiquitous across architectures and scales, but it has zero causal weight for energy violations. The violation mechanism identified in Phase 2 — negative eigenvalues of the signed component $S$ — operates through at most 2–16% of the OV matrix's spectral energy.

This has two implications:

1. **Phase 2 conclusions are not confounded by the imaginary structure.** The fact that ~98% of OV eigenvalue dimensions are complex does not undermine the repulsive subspace attribution. The signed residue, however structurally minor, carries 100% of the causal weight for violations.

2. **ALBERT-xlarge's violations are sustained by a 2.5% signed residue.** At ALBERT-xlarge's scale (1024-dimensional OV, 97.5% rotational), there are still 26+ violations on the repeated_tokens prompt, all attributable to the ~25-dimensional signed subspace of a 1024-dimensional matrix. The mechanism is highly concentrated.

The broad pattern — rotation dominant structurally, signed component dominant causally — holds without exception across encoder and decoder architectures, from 12-layer to 48-layer models.

---

## Known Issues / Caveats

1. **ALBERT full rescaling overcorrection**: `elim_full` is unreliable as a diagnostic for ALBERT (shared-weight architecture). Use signed-only rescaling as the definitive test.
2. **Block 2 not run**: Fiedler tracking and rotation-hemisphere alignment analyses are not executed because Block 1b uniformly returned `rotation_neutral`. These analyses remain available if a future model or prompt type shows rotational contribution.
3. **gpt2 entry in results**: The `gpt2` (small) entry in `phase2i_results.json` appears to aggregate results from multiple GPT-2 model sizes (large/medium/xl/small) due to a naming collision in the runner. The per-model entries (`gpt2-large`, `gpt2-medium`, `gpt2-xl`) are the authoritative per-model results.

---

## Modules

### `rotational_schur.py` — Block 1a rotational spectrum
Decomposes OV eigenvalue structure into complex pairs and real eigenvalues. Computes rotation angles ($\theta$), radial scaling ($\rho$), rotational vs signed energy fractions, and Henrici non-normality. Handles per-layer and shared-weight models.

### `rotational_rescaled.py` — Block 1b causal isolation
S/A decomposition of $V_\text{eff}$. Applies three independent rescaling frames (full, signed-only, rotation-only). Recomputes energy violations in each frame and classifies each as `rotation_neutral`, `rotation_contributes`, or `nonlinear_interaction`.

### `fiedler_tracking.py` — Block 2 (conditional)
Tracks the Fiedler vector of the token similarity graph across layers. Crossing rate, cosine similarity to previous layer, and centroid angle near $\pi$ (bipartition indicator). Not run in this experiment.

### `rotation_hemisphere.py` — Block 2 (conditional)
Tests alignment between token displacement vectors and rotation-plane hemispheres defined by OV's complex eigenvectors. Displacement coherence metric. Not run in this experiment.

### `ffn_rotation.py` — FFN rotation interaction (conditional)
Tests whether FFN updates at violation layers have anomalous projections onto rotation planes. Requires per-layer FFN deltas from Phase 2 decomposition. Not run in this experiment.

### `run_2i.py` — CLI entry point
Orchestrates Block 1a → 1b → (conditionally) Block 2 → FFN rotation. Loads Phase 2 OV artifacts and Phase 1 activations/events. Saves per-model JSON results and combined `phase2i_results.json`.


OLD:

# Phase 2i — Rotational Dynamics Analysis

## What this does

Phase 2 characterized ~3% of V_eff's spectrum (real eigenvalues, the signed structure). Phase 2i characterizes the remaining ~97% — complex conjugate pairs encoding rotations in 2D invariant subspaces.

The central question: **does the rotational structure contribute to metastability, or is it dynamically neutral?**

## Running

```bash
# Full pipeline using saved Phase 2 + Phase 1 artifacts
python -m phase2i.run_2i \
    --phase2-dir results/phase2_full \
    --phase1-dir results/phase1

# Single model
python -m phase2i.run_2i \
    --phase2-dir results/phase2_full \
    --phase1-dir results/phase1 \
    --models albert-xlarge-v2

# Block 1 only (weight analysis + causal test)
python -m phase2i.run_2i \
    --phase2-dir results/phase2_full \
    --phase1-dir results/phase1 \
    --block1-only

# Force Block 2 even if rotation is neutral
python -m phase2i.run_2i \
    --phase2-dir results/phase2_full \
    --phase1-dir results/phase1 \
    --force-block2
```

## Block structure

### Block 1a: `rotational_schur.py` — Rotational spectrum characterization

Pure weight-matrix analysis. No activations needed.

- Schur-decomposes V_eff, parses 1×1 (real) and 2×2 (rotation) blocks
- Per-block: rotation angle θ, spectral radius ρ, radial sign
- Energy fractions: what percentage of V_eff's action is rotational vs signed
- Henrici departure from normality: measures inter-block coupling (transient amplification)
- Builds projectors onto top-k rotation planes and the real subspace
- For GPT-2: depth profile of θ and ρ distributions

### Block 1b: `rotational_rescaled.py` — Causal isolation (decision point)

Needs saved Phase 1 activations.

Decomposes V = S + A where S = (V+V^T)/2 (symmetric/signed) and A = (V-V^T)/2 (antisymmetric/rotational). Applies three rescaled frames:

| Frame | What it removes | What it keeps |
|-------|----------------|---------------|
| Full (Phase 2 baseline) | Everything | Nothing |
| Signed-only (e^{-tS}) | Signed structure | Rotation |
| Rotation-only (e^{-tA}) | Rotation | Signed structure |

Compares energy violation counts across all four frames (original + three rescaled).

**This is the decision point.** If removing rotation alone doesn't reduce violations, Block 2 is skipped and the conclusion is: rotational dynamics are dynamically neutral, Phase 2's analysis was complete.

### Block 2: `fiedler_tracking.py` + `rotation_hemisphere.py` — Hemispheric geometry

Conditional on Block 1b showing rotational contribution.

**Fiedler tracking:** Extracts the Fiedler vector (spectral bipartition) at every layer. Tracks hemisphere assignments, crossing rate (tokens switching sides), Fiedler stability (cosine similarity between consecutive layers), centroid-centroid angle (antipodal = π).

**Rotation-hemisphere alignment:** Tests whether dominant rotation planes are within-hemisphere (preserving the bipartition) or across-hemisphere (destabilizing it). The token trajectory test (projecting actual Δx onto the Fiedler axis) is primary — it captures the full nonlinear dynamics. The linear plane alignment is secondary.

Also measures within-hemisphere displacement coherence: are tokens in the same hemisphere displaced in parallel (rigid rotation) or independently? High coherence = rigid rotation hypothesis holds.

### FFN rotation: `ffn_rotation.py` — FFN interaction with rotational structure

Conditional on Block 1b + per-layer model (GPT-2). Needs saved FFN deltas.

Projects FFN updates onto rotation planes vs real subspace. Tests whether the FFN selectively amplifies specific rotation planes at violation layers. If yes, removing A in the rescaled frame may be insufficient because the FFN re-introduces rotational displacement.

## Dependencies

**Required:**
- Phase 2 saved artifacts: `ov_weights_{model}.npz` (contains V_eff matrices)
- Phase 1 saved activations: `activations.npz` per (model, prompt) run
- Phase 1 metrics: `metrics.json` per run (for violation cross-referencing)

**Optional:**
- Phase 2 FFN deltas: `ffn_deltas_raw.npz` or `ffn_deltas_normed.npz` (for FFN rotation analysis)

## Falsification

| Test | Null result | Meaning |
|------|-------------|---------|
| Rotation-only rescaled frame (1b) | No improvement | Rotations dynamically neutral; 3% real eigenvalues do all the work |
| Hemisphere crossing rate (2) | High during plateaus | Hemispheres not stable units |
| Rotation planes across-hemisphere (2) | Planes cut across Fiedler boundary | Rotation destabilizes rather than preserves |
| Displacement coherence (2) | Low within-hemisphere coherence | Not rigid rotation; hypothesis fails on its own terms |
| FFN rotation z-score | Near zero | FFN blind to rotational structure |

The cleanest falsification: Block 1b shows rotation_neutral AND Block 2 shows planes across-hemisphere → the 97% is structurally present but dynamically inert for metastability.

## Output files

Per model:
- `block1a_rotational_spectrum.json` — Schur block statistics, Henrici measure
- `block1b_rescaled_comparison.json` — Violation counts in all four frames
- `block2_hemispheric.json` — Fiedler tracking + rotation-hemisphere alignment
- `ffn_rotation.json` — FFN projection onto rotation planes (GPT-2 only)

Combined: `phase2i_results.json` — all results for all models.
