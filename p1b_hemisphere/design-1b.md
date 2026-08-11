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
  Delegates the $k=2$ flip decision to `cluster_tracking.match_layer_pair`
  (`matcher="hungarian"`, the default). At $k=2$ the Hungarian assignment over labels
  $\{0,1\}$ *is* the global-sign-flip decision — the only two assignments are identity
  and flip — so there is no hemisphere-specific matching logic to own. The reported
  *score* stays local: the mean of the two halves' Jaccards, which is what
  `IDENTITY_THRESHOLD` and every existing result are stated against.
  `matcher="local"` keeps the previous in-module comparison so the delegation stays
  checkable against it.
  (Until this revision this paragraph described reuse that did not exist; the module
  carried its own Jaccard pair. See Errata.)
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


---

# Errata and revisions

This section records where the design above was wrong or has been
superseded. It is appended rather than edited in place so the original
reasoning stays legible next to what replaced it. Full detail in
`CHANGES-1b.md`.

## The two-test framing of Block 0 and Block 3 was one test

The design presents "no strong bipartition" and "universal cone-collapse" as
independent results pointing the same way. They are close to the same test.
`classify_regime`'s `strong_bipartition` requires a centroid angle of at
least pi/2, and two centroids inside a single open half-space essentially
cannot be pi/2 apart. Under cone-collapse the antipodal classifier's positive
verdict is near-unreachable, so its absence is near-uninformative.

Block 0 now also runs `classify_regime_relative`, which asks whether
between-half similarity is materially below within-half similarity regardless
of absolute angle. Measured: two clusters 60 degrees apart with separation
ratio 0.45 read as `weak_bipartition` under the antipodal rule and
`separated` under the relative one. **"0% strong bipartition" is not the same
claim as "no bipartition"**, and the original run could not express the
difference.

`regime_key` selects which vocabulary drives Block 1's events. With the
antipodal default, birth/collapse/swap and persistence are all foreclosed
under cone-collapse — which is why the original run reported zero events.

## Block 3's PCA note was wrong in one direction

The design said the cone question is invariant under orthogonal projection.
It is not, and the asymmetry matters:

- A reduced-space witness lifts **exactly**: `w = Vt[:k].T @ w_r` gives
  `X @ w == X_r @ w_r`. A cone_collapse verdict under PCA is sound.
- A full-space witness's orthogonal component is discarded, so a **split**
  verdict under PCA may be a projection artifact.

`escalate_on_split=True` re-solves at full d in the one direction that can
lie. The existing 100%-cone-collapse result is unaffected.

Separately, the design never asked how much of that result is dimension
counting. n points in d dimensions separate for free unless they positively
span. Two matched nulls (shuffled-dimension, uniform-sphere) are now
available via `--n-null`, and `normalized_margin` — not the regime label — is
the quantity a falsification table should adjudicate.

## Block 1's reuse claim was aspirational

See the corrected text above. Wiring the delegation up surfaced a live
hazard: exact ties let the assignment solver return either pairing, on 4 of
500 random label pairs, and anchor chaining would propagate each flip through
the rest of a run.

## New blocks

**Block A — axis identity** (`axis_identity.py`). Maps the token-space
Fiedler vector into activation space so it is comparable across layers and
checkpoints, and asks whether it is distinguishable from the cloud's leading
variance geometry.

Note what this block does *not* ask. Its first version tested the axis
against the mean token direction; that is unreachable by construction,
because the Fiedler vector is orthogonal to the Laplacian's trivial
eigenvector and `X^T f` therefore cancels the shared mean component
(measured |cos| between 0.000 and 0.085 across all fixtures). Asking it would
have repeated the Block 0 defect above. Redundancy is asked against centered
PC1 and the top-k PC subspace, with `1/sqrt(d)` reported beside every cosine.

**Boundary vs unclustered** (`border_vs_noise` in Block 2). Crosses the
per-token distance from the Fiedler boundary against HDBSCAN's noise labels.
Both quantities already existed; nothing had crossed them. This is the Phase
5c question — whether the unclustered population is the boundary population —
answered with a rank AUC.

## Checkpoint axis

The design has no training-step axis anywhere, so a Pythia pilot renders as N
unrelated models. `aggregate_by_checkpoint` groups families and reports
against log10(step+1), and `cross_checkpoint_axis_rotation` /
`axis_settling_step` ask when the Fiedler axis reaches its trained direction
— the quantity PREDICTIONS.md claim (b) needs, and the one thing in this
phase that tracks the axis's *direction* rather than lambda_2's magnitude.
