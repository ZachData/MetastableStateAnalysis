# Phase 1b (1h) — DESIGN

## Core question

Phase 1's spectral eigengap heuristic on the token Gram matrix's normalized Laplacian
consistently returns $k=2$ on long prompts — a dominant bipartition — while HDBSCAN
simultaneously finds 30–60+ local clusters at the same layer. The Fiedler vector (second
Laplacian eigenvector) defines this bipartition by sign. Phase 1b asks what this bipartition
is, dynamically and semantically: when it exists, when it rearranges, which tokens live
where, and how it relates to the "hemisphere" condition in the paper's convergence theorems.

## Why this distinction matters (and is itself a result)

Geshkovski et al.'s "hemisphere" is a *containment* condition (Theorem 6.3 / Lemma 6.4,
cone collapse): if all tokens start in one open hemisphere $\{x : \langle w,x\rangle > 0\}$,
dynamics converge exponentially to a single point. That's the setting in which the theorems
apply, not a claim about tokens splitting into two groups. The Phase 1 $k=2$ finding
suggested the opposite geometry — two populated antipodal half-spaces. These regimes are
mutually exclusive, so Phase 1b operationalizes both and tests which one holds:

- **Cone-collapse (paper):** $\exists\, w$ with $\min_i \langle w, x_i \rangle > 0$.
- **Split:** Fiedler bipartition populated on both sides with compact within-hemisphere
  geometry.

Running these as an exhaustive, mutually-exclusive pair (rather than testing "does a split
exist" alone) is what lets the result be a clean falsification rather than an ambiguous
partial finding: Block 3's null (cone-collapse holds everywhere) and Block 0's null (no
strong bipartition) are two independent tests that happen to point the same direction, which
is stronger evidence than either alone.

## What the result means for the project

The finding — cone-collapse universal, strong bipartition absent, but the Fiedler axis
itself stable and identity-preserving — reconciles Phase 1's empirical observation with the
paper's actual theorem rather than contradicting it. The axis is real (an anisotropy
direction in the Gram matrix) but isn't a separator. This reframing is why the module is
named for the finding it produced (`p1b_hemisphere`) rather than for the bipartition
hypothesis that motivated it — the directory predates knowing the answer, but the content
inside documents the correction, including the note that an earlier draft's narrative
(hardcoded in `run_1b.py`) was written before the run and is now wrong (tracked as a known
issue rather than silently patched, since the reasoning behind the correction has value).

## Module structure

- `bipartition_detect.py` (Block 0) — per-layer regime classifier and quality metrics.
- `hemisphere_tracking.py` (Block 1) — identity matching across layers, axis rotation.
  Reuses `cluster_tracking.match_layer_pair` at $k=2$ rather than duplicating
  identity-matching logic — the same Jaccard-chaining approach that tracks HDBSCAN clusters
  works unchanged for a 2-way Fiedler partition.
- `hemisphere_membership.py` (Block 2) — per-token trajectories, HDBSCAN nesting within the
  bipartition.
- `cone_collapse.py` (Block 3) — LP-based cone-collapse test and regime classifier. LP
  (rather than a heuristic) because the containment condition is exactly a linear
  feasibility question ($\exists w$ such that all inner products are positive).
- `hemisphere_mechanism.py` (Block 5, conditional) — axis alignment vs. OV/PCA/embedding/
  heads. Needs Phase 2 artifacts; designed to run silently if absent rather than fail, since
  Phase 1b is meant to be runnable standalone off Phase 1 output alone.
- `hemisphere_semantics.py` (Block 6, conditional) — token-attribute contingency and MI.
  Same Phase 2 dependency as Block 5.

Reused, not duplicated: `fiedler_tracking.py`, `rotation_hemisphere.py`,
`spectral.spectral_eigengap_k` (returns the full Fiedler vector on request).

## Forward dependencies this phase sets up

The corollary for every later phase: don't treat the bipartition sign as a two-class label.
The axis (continuous projection) is still a candidate feature — e.g. for a linear probe, or
for alignment against OV eigenvectors once Phase 2 artifacts are available. Phase 5's
hemisphere centroids and Phase 6's Fiedler-difference-vector probe both inherit this
"axis, not separator" framing directly from Phase 1b's result rather than from the original
Phase 1 speculative narrative.
