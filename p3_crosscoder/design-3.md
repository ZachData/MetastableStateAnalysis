# Phase 3 — DESIGN

## Core question

Phase 1 found metastability (plateaus, merges); Phase 2 found the mechanism (V's mixed-sign
eigenspectrum). Phase 3 asks whether this dynamical structure organizes the *learned
representation* at the feature level — trains a sparse crosscoder (a sparse autoencoder
whose input is the residual stream stacked across multiple layers) and tests whether its
features align with Phase 1/2's dynamical structure.

## Model and regime selection

ALBERT-xlarge-v2 (Regime A, attention-mediated, 100% negative self-interaction — the
cleanest dynamical signal from Phase 2) and GPT-2-large (Regime B, FFN-mediated, 100%
rescaled-frame elimination, smooth repulsive gradient). BERT-base is excluded — it sits
below Phase 2's detection threshold on every test, so a crosscoder result there would have
no dynamical signal to explain regardless of outcome. This mirrors Phase 2's regime split
directly rather than re-running the full 7-model grid.

## Architecture choices and why

BatchTopK (not L1) activation, per current SAE-training consensus at the time this was
built. Decoder columns normalized to unit norm after each optimizer step — this choice is
also what caused Bug 2 (below): a lifetime metric that read raw decoder norm broke silently
once normalization made all norms identically 1.0. Dead features resampled from high-loss
inputs. Cross-layer stacking (not per-layer SAEs) is the entire premise of a *crosscoder* —
the `multilayer_fraction` control (>97% both models) exists specifically to confirm the
model isn't just behaving as independent per-layer SAEs stapled together.

## Four predictions, and why each is structured this way

1. **Feature lifetime bimodality** — tests whether features split into short-lived
   (violation/merge-associated) and long-lived (stable cluster membership) populations. Two
   classification schemes (valley-threshold "current" vs. fixed-cutoff "legacy") are reported
   side by side rather than the older one being silently replaced, because they disagree
   substantially on the ALBERT split and downstream analyses need to be traceable to which
   scheme they used.
2. **Decoder → V alignment** — direct geometric test: do long-lived features' decoder
   directions sit in V's attractive subspace, short-lived in repulsive? This is the test that
   returns a clean null in both models and is the load-bearing negative result of the phase.
3. **Cluster identity / violation-layer features** — whether violation displacement
   decomposes into a small number of feature directions (top-k projection coverage). Requires
   Phase 1 HDBSCAN labels and plateau layers, so it's necessarily downstream of both earlier
   phases rather than independent.
4. **Steering** — the causal complement to the correlational tests above; activates a
   selected feature and measures the effect on cluster merge layer, specifically to check
   whether the null in (2) also holds under intervention rather than only under passive
   correlation.

## Reading the null

Two interpretations were considered and the design favors one over the other for a specific,
falsifiable reason:

- **Favored — dissociation.** The crosscoder (trained on general web text, C4) learned
  syntax/frequency/surface-form features that happen to have the right *temporal* profile,
  but Phase 2's mechanism and Phase 3's feature-level organization are simply different
  things.
- **Not favored — eval-set coverage gap.** The 4-prompt eval set is too narrow to activate
  metastability-tied features. This is disfavored specifically because GPT-2's *unimodal*
  lifetime distribution argues against it: a coverage problem would suppress alignment signal
  (which it does) but shouldn't suppress bimodality, since bimodality is a property of the
  *training* distribution, not the eval set. This is the kind of falsifiable distinguishing
  test the plan's methodology section asks every phase to apply — the two interpretations
  aren't just asserted, one is ruled less likely by an internal control.

## Bugs fixed (kept for record, not re-litigated)

1. `decoder_norms()`/`decoder_directions()` returned CUDA tensors; analysis called `.numpy()`
   directly → crash. Fixed with `.cpu()`.
2. Decoder normalization after every optimizer step made all `W_dec` column norms exactly
   1.0, collapsing the lifetime threshold `norms[f] > max_norm * 0.1` to always-true → every
   feature got lifetime = n_layers. Fixed by replacing the norm-based metric with
   `_compute_feature_layer_scores()` — mean squared projection of actual activations onto each
   decoder direction, weighted by activity, which is unaffected by norm normalization.
3. `np.bool_` isn't JSON serializable under Python 3.10's encoder; explicit `bool()` cast
   added to `prediction_confirmed`.
4. Multi-word prompt names (`short_heterogeneous`) were truncated to their first word by a
   regex extracting run metadata; fixed to iterate subdirectories and extract the full prompt
   name.

## Why frozen-for-deletion, not just frozen (v2)

v1 treated this as a pause: relocate untouched, revisit once a checkpoint suite exists. v2
is more decisive: this is a *candidate for deletion*, with git history as the archive, not a
paused experiment awaiting more data. The reasoning shift matters — "more data might help"
is a weak reason to keep code alive indefinitely; "this method has a specific, checkable
reintroduction condition" is a real gate. That condition, as stated in the plan: activation
caches at ≥4 checkpoints *and* a specific particle-dynamics question that actually requires a
dictionary. Absent both, the code stays inert rather than becoming a maintenance burden that
looks active. The underlying empirical reason is unchanged from v1 — sparse dictionary
methods already underperform dense/low-rank alternatives here (chorus ARI = 0.000, both
models) — v2 just declines to treat "the Pythia sweep will eventually generate more data" as
sufficient justification on its own for keeping this alive in the meantime.

## Code structure

`crosscoder.py` (model), `training.py`, `data.py` (C4 default, TinyStories for fast
iteration), `analysis.py` (predictions 1–4 plus the two cross-phase analyses that didn't
run), `steering.py`, `run_3.py` / `run.py` (CLI — auto-detects cached activations/checkpoints,
`--skip-cache`/`--skip-train`/`--force-*` to override).
