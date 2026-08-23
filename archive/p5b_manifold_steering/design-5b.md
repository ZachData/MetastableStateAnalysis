# Phase 5b — DESIGN

## Core hypothesis

Wurgaft et al. (2026), *Manifold Steering Reveals the Shared Geometry of Neural Network
Representation and Behavior*, fit an activation manifold $M_h$ from concept centroids
threaded with a spline, show it's approximately isometric to a behavior manifold $M_y$ (fit
to output probability distributions), and that steering *along* $M_h$ produces natural
behavioral trajectories while linear steering doesn't.

Phase 5b asks: **are our metastable-state cluster centroids the same objects as Wurgaft's
concept centroids?** This reframes the entire project's cluster-tracking apparatus
(Phase 1's HDBSCAN labels and centroid trajectories) as unsupervised evidence for a claim
Wurgaft made with labeled concept data — if it holds, the project's cluster centroids give a
causal upstream Wurgaft's original paper doesn't have: $M_h$ has its geometry *because* V's
eigenstructure (Phase 2/6's S subspace) determines which states are metastable.

The proposed causal chain: V eigenstructure → metastable attractor landscape → activation
manifold $M_h$ ↔ (isometry) → behavior manifold $M_y$.

## Why four sub-experiments, each targeting one link in that chain

- **A (manifold fitting)** — necessary first step: without $M_h$/$M_y$ fit to *our* cluster
  centroids (not Wurgaft's labeled concepts), no other sub-experiment has anything to
  operate on. Uses unsupervised HDBSCAN centroids in place of concept-labeled data
  deliberately, since that substitution is the entire test.
- **B (isometry test)** — the central replication of Wurgaft's own claim (§2.3), on
  unsupervised structure instead of labeled concepts. This is the load-bearing test: if
  `r_manifold` isn't clearly greater than `r_linear`, the rest of the causal chain has
  nothing to attach to.
- **C (merge-event teleportation)** — a novel prediction not in the original Wurgaft paper:
  if merge events are the model's own transitions between metastable states, they should
  show the same "teleportation" signature Wurgaft found under *linear* steering
  (non-adjacent probability jumps). This is where the phase's own dynamical-systems framework
  (merge events from Phase 1) makes a prediction Wurgaft's framework alone wouldn't.
- **D (S-subspace isometry)** — directly cross-validates Phase 6's S/A division-of-labor
  hypothesis using the isometry framework: does restricting $M_h$ to the S (real/symmetric)
  subspace improve isometry with $M_y$ relative to full or A-restricted $M_h$? This is the
  test that would confirm V's eigenstructure is *the* coordinate system $M_h$ lives in, not
  just correlated with it.

## Why falsification criteria are specified per sub-experiment before running

Each sub-experiment has explicit pass/fail thresholds set in advance (e.g., P5b-B2:
`r_manifold > 0.7`, calibrated against Wurgaft's reported 0.89–0.999 for concept-labeled
tasks) rather than being evaluated post hoc — this matters more here than in most phases
because the entire premise (cluster centroids = concept centroids) is an identity claim that
could otherwise be argued into a positive result from ambiguous correlations. A `B fails`
outcome is explicitly informative, not just a null: it would mean cluster structure is
dynamically meaningful (established by Phases 1–4) but not semantically structured in
Wurgaft's specific sense.

## New requirement this phase introduces

Prior phases never needed output distributions (logits) at intermediate layers — only
activations. Phase 5b requires a re-forward pass with logit extraction (`logit_cache.py`) or
caching logits during a Phase-1-style run. This is called out explicitly as new
infrastructure rather than an existing artifact, since every other phase's dependency list
so far has been satisfiable from Phase 1/2/3/4 outputs alone.

## Explicit scope boundary

This phase does not implement Wurgaft's manifold-steering *intervention* (replacing
activations with spline-interpolated targets) — the goal here is identity verification (are
these the same geometric objects?), not steering. Steering experiments are deliberately left
to a later phase. Also not attempted: Wurgaft's pullback procedure (optimizing activation
paths to follow $M_y$) — sub-experiment D is only a partial analog.

## Code structure

`manifold_fit.py` (A), `isometry_test.py` (B), `merge_teleportation_subspace.py` (C),
`subspace_isometry_file.py` (D), `p5b_io.py` (logit caching + I/O), `run_5b.py` (CLI).
Output directory per model/timestamp: `run_config.json`, `fit_summary.json`, `mh_params.npz`,
`my_params.npz`, `isometry.json`, `isometry_mds.npz`, `merge_teleportation.json`,
`teleportation_summary.json`, `subspace_isometry.json`, `p5b_report.txt`.
