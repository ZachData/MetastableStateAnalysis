# Phase 5b — Metastable States as Activation Manifold Control Points

**Status:** Not started.
**Reference:** Wurgaft et al. (2026), *Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior*. Code: https://github.com/goodfire-ai/causalab/tree/manifold_steering

---

## Core Hypothesis

Wurgaft et al. fit an activation manifold Mh by taking concept centroids and threading a spline
through them. They show Mh is approximately isometric to a behavior manifold My (fit to output
probability distributions), and that steering *along* Mh produces natural behavioral trajectories
while linear steering does not.

**This phase asks: are our metastable-state cluster centroids the same objects as Wurgaft's concept
centroids?**

If yes, several things follow immediately:
1. The activation manifold Mh in our setting is the *attractor landscape* of the Geshkovski
   dynamics — the metastable states are its control points.
2. Merge events are off-manifold excursions, producing the "teleportation" Wurgaft observes under
   linear steering.
3. The real/symmetric subspace S (Phase 2/6) determines the manifold geometry — S is the
   coordinate system in which Mh lives.
4. The Geshkovski framework provides the *causal upstream* Wurgaft lacks: Mh has its geometry
   because V's eigenstructure defines which states are metastable.

Combined causal chain:
```
V eigenstructure (S subspace) → metastable attractor landscape
                               → activation manifold Mh
                               ↔ (isometry) behavior manifold My
```

---

## What Prior Phases Provide

- **Phase 1:** HDBSCAN cluster labels per layer, cluster centroid trajectories
  (`centroid_trajectories.npz`), plateau windows, merge event layers,
  energy violation layers. All already computed.
- **Phase 2 / Phase 6:** S/A subspace projectors (`ov_projectors_{stem}.npz`),
  attractive/repulsive eigenbases U_pos, U_neg. Required for sub-experiment D.
- **Phase 5:** Per-cluster structural profile, V-alignment, merge geometry.
  Provides candidate trajectories already ranked by quality.

New requirement: **output distributions at each layer for the target model/prompt.**
These are not stored by Phase 1 (which only stores activations). We need to re-run
the forward pass with logit extraction, or cache logits during Phase 1.

---

## Sub-Experiments

### A — Manifold Fitting (`manifold_fit.py`)

Fit Mh and My to cluster centroids and output distributions respectively,
following Wurgaft §2.2 closely but using unsupervised cluster centroids
instead of concept-labeled data.

**Inputs:**
- `centroid_trajectories.npz` — pre-computed cluster centroid paths (Phase 1)
- `clustering.json` / `clusters.npz` — HDBSCAN labels per layer
- Model + tokenizer (for logit extraction)
- Plateau layer windows from Phase 1

**Procedure:**
1. Identify N_c clusters alive at plateau layers. Compute per-cluster centroid
   at each plateau layer (already in centroid_trajectories).
2. PCA-reduce centroids to 32d. Fit a 1D cubic spline through centroids in
   intrinsic-coordinate order (ordering determined by arc-length parameterization
   of the centroid path across layers).
3. For each cluster centroid at each plateau layer, extract logit distribution
   over the full vocabulary. Restrict to the top-K tokens (K=512) to keep My
   tractable. Map to Hellinger coordinates p → √p. Fit 1D smoothing spline.

**Outputs:**
- `mh_params.npz` — PCA basis (32d), spline knots and coefficients in PCA space
- `my_params.npz` — Hellinger-space spline knots and coefficients
- `fit_summary.json` — n_control_points, explained variance of PCA, spline residuals

**Predictions:**
- P5b-A1: The 32d PCA of cluster centroids retains ≥ 80% of variance.
- P5b-A2: Spline residuals (centroid-to-nearest-spline-point distance) are small
  relative to inter-centroid distances (ratio < 0.1).

---

### B — Isometry Test (`isometry_test.py`)

Test whether pairwise geodesic distances on Mh correlate with geodesic distances on My.
This is the central claim of Wurgaft §2.3 — replicated on unsupervised cluster structure.

**Inputs:** Mh and My from sub-experiment A.

**Procedure:**
1. For every pair of cluster centroids (i, j), compute:
   - `d_manifold(i,j)`: arc-length along Mh between the two points
     (cumulative Euclidean distance in PCA space, 150 waypoints).
   - `d_linear(i,j)`: Euclidean distance between raw centroid vectors.
   - `d_behavior(i,j)`: arc-length along My (cumulative Hellinger distance).
2. Compute Pearson r between d_manifold and d_behavior (main result).
3. Compute Pearson r between d_linear and d_behavior (baseline).
4. Visualize via MDS: embed each pairwise distance matrix in 2D,
   compare structures.

**Outputs:**
- `isometry.json` — r_manifold, r_linear, N_pairs, p-values
- `isometry_mds.npz` — 2D MDS embeddings of all three distance matrices

**Predictions (falsifiable):**
- P5b-B1: r_manifold > r_linear (main claim — manifold distances track behavior better).
- P5b-B2: r_manifold > 0.7 (Wurgaft reports 0.89–0.999 for concept-labeled tasks).
- P5b-B3: r_linear < r_manifold by at least 0.1.

**Failure mode:** If r_manifold ≈ r_linear, the cluster centroids are not the same
objects as concept centroids (or the output distributions at plateau layers don't
encode a structured concept space). This is informative — it would mean that
metastable states are dynamically meaningful but not semantically structured in
the Wurgaft sense.

---

### C — Merge-Event Teleportation (`merge_teleportation.py`)

Wurgaft shows linear steering produces "teleportation" — output probability mass
jumps to non-adjacent concepts at intermediate steps. Our merge events are the
model's own transitions between metastable states. This test checks whether
merge layers show the same signature in the output distribution.

**Inputs:**
- Phase 1 merge event layers, plateau layers, cluster labels
- Output distributions at each layer (from logit cache)

**Procedure:**
1. For each merge event: extract output distributions at (a) the plateau window
   just before the merge, (b) the merge layer itself, (c) the first post-merge
   plateau layer.
2. Compute "teleportation score" T = KL(p_merge || p_plateau_before) — how far
   does the distribution move at the merge layer.
3. Compute "neighbor score" N = fraction of probability mass on tokens that were
   NOT in the top-5 of p_plateau_before.
4. Compare T and N at merge layers vs. random non-merge layers.
5. Also compute: does the distribution at the merge layer pass THROUGH
   a region of behavior space far from My (Bhattacharyya distance to My)?

**Outputs:**
- `merge_teleportation.json` — per-merge T, N, dBC scores; plateau vs merge comparison
- `teleportation_summary.json` — mean/std/p-value for each metric

**Predictions:**
- P5b-C1: KL divergence (teleportation score T) is significantly higher at merge
  layers than at plateau layers (p < 0.05).
- P5b-C2: Bhattacharyya distance to My is higher at merge layers than plateau layers.
- P5b-C3: N (non-adjacent token mass) is higher at merge layers.

---

### D — S-Subspace Isometry (`subspace_isometry.py`)

Phase 6's prediction: metastable cluster structure lives in the S (real/symmetric)
subspace of V. Wurgaft's Mh is fit in a PCA subspace, which may or may not
align with S. This test asks: does Mh restricted to the S subspace have higher
isometry with My than full Mh or Mh restricted to A?

**Inputs:**
- Phase 2/6 S/A projectors (U_pos, U_neg, U_A from `ov_projectors_{stem}.npz`)
- Cluster centroids (raw, not PCA-reduced)
- My from sub-experiment A

**Procedure:**
1. Project each cluster centroid onto U_S = span(U_pos ∪ U_neg) and onto U_A.
2. Recompute pairwise distances in each projected space.
3. Correlate with d_behavior from sub-experiment B.
4. Compare: r(S-projected, My) vs r(full, My) vs r(A-projected, My).

**Outputs:**
- `subspace_isometry.json` — r_S, r_A, r_full, r_linear; all vs d_behavior

**Predictions:**
- P5b-D1: r_S > r_full ≥ r_A (S subspace is better at recovering behavior geometry).
- P5b-D2: r_A ≈ r_linear (A-subspace distances are no better than linear distances).

This directly cross-validates Phase 6's S/A division-of-labor hypothesis using
the Wurgaft isometry framework.

---

## What Each Test Tells Us

| Test | Connection Confirmed | What It Means |
|------|---------------------|---------------|
| B passes (r_manifold >> r_linear) | Cluster centroids = Mh control points | Metastable states are the manifold |
| C passes (merge = teleportation) | Merge events = off-manifold excursions | Merges violate manifold geometry |
| D passes (r_S > r_full) | S subspace = manifold coordinate system | V eigenstructure explains Mh geometry |
| B fails (r_manifold ≈ r_linear) | Cluster structure is dynamical, not semantic | Objects are different; Wurgaft Mh is concept-specific |
| C fails (no teleportation) | Merges are smooth, not jumps | Transition physics differs from linear steering |

---

## Code Structure

```
p5b_manifold/
├── README_p5b.md                   (this file)
├── __init__.py
├── manifold_fit.py                 — Sub-exp A: fit Mh and My
├── isometry_test.py                — Sub-exp B: geodesic distances + Pearson r
├── merge_teleportation.py          — Sub-exp C: output distributions at merge events
├── subspace_isometry.py            — Sub-exp D: S vs A vs full isometry
├── logit_cache.py                  — extract and cache output distributions per layer
├── report.py                       — emit p5b_report.txt
└── run_5b.py                       — CLI entry point
```

---

## Output Directory

```
results/phase5b/{model_stem}_{timestamp}/
├── run_config.json
├── fit_summary.json                — Sub-exp A
├── mh_params.npz
├── my_params.npz
├── isometry.json                   — Sub-exp B
├── isometry_mds.npz
├── merge_teleportation.json        — Sub-exp C
├── teleportation_summary.json
├── subspace_isometry.json          — Sub-exp D
└── p5b_report.txt                  — LLM-friendly flat summary
```

---

## Dependencies

### Required (blocking)
- Phase 1 results: `centroid_trajectories.npz`, `clustering.json`, `clusters.npz`,
  merge event annotations — ✓ complete for all models
- Phase 2 / Phase 6 projectors: `ov_projectors_{stem}.npz` — ✓ available

### New requirement
- Forward pass with logit extraction. Not stored in Phase 1. `logit_cache.py`
  provides this with a single re-forward on the same prompts used in Phase 1.

### Optional
- Phase 5 trajectory rankings, to prefer high-quality clusters for the case study.

---

## Falsification Criteria

| ID | Prediction | Failure |
|----|-----------|---------|
| P5b-A1 | 32d PCA retains ≥ 80% variance of centroids | < 70%: centroids are too diffuse; Mh is not low-dimensional |
| P5b-A2 | Spline residual / inter-centroid distance < 0.1 | > 0.2: centroids do not lie on a smooth manifold |
| P5b-B1 | r_manifold > r_linear | Falsified: cluster geometry not better than linear |
| P5b-B2 | r_manifold > 0.7 | 0.5–0.7: weak; < 0.5: not the same object as Wurgaft's Mh |
| P5b-C1 | KL(merge) > KL(plateau), p < 0.05 | No difference: merge events don't distort output distributions |
| P5b-C2 | dBC to My higher at merge layers | Falsified: merge layers are on-manifold |
| P5b-D1 | r_S > r_full ≥ r_A | Falsified: S subspace not the manifold coordinate system |
| P5b-D2 | r_A ≈ r_linear | Falsified: A subspace carries behavior-relevant geometry |

---

## Notes on Scope

We do NOT implement Wurgaft's manifold steering intervention (replacing activations
with spline-interpolated targets) in this phase. The goal is identity verification:
are these the same geometric objects? Steering experiments belong to a later phase.

We also do not attempt to replicate Wurgaft's pullback procedure (optimizing activation
paths to follow My), though sub-experiment D is a partial analog.
