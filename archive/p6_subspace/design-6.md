# Phase 6 — DESIGN

## Core question

Phase 2b established that the antisymmetric/imaginary component $A$ is dynamically neutral
— removing it from OV leaves energy violations unchanged; the symmetric/real component $S$
carries 100% of violation causality. That answers "does rotation drive clustering?" (no). It
doesn't answer **what the rotational subspace does** — a subspace orthogonal to clustering
dynamics is free to carry other computation without interfering with the attractor
structure. Phase 6 tests a specific division-of-labor hypothesis:

> **Real subspace ($S$):** semantic similarity, metastable-state organization. Tokens in the
> same cluster are degenerate in $S$'s eigenspace.
>
> **Imaginary subspace ($A$):** relational computation — operations that can't reduce to
> pairwise inner products on current positions (induction, previous-token heads, copy/
> name-mover heads, anti-similarity heads, coreference).

The unifying principle: self-similarity operations use real structure; relational operations
use imaginary structure. This is a stronger, more falsifiable claim than "rotation exists
and might do something" — it predicts a specific correlation structure (Part A) and a
specific geometric signature (Part B) and licenses a direct causal test (Part C).

## Why four parts, each targeting a different kind of evidence

- **Part A (head classification, CC/PC scores)** — behavioral evidence. Classifies heads by
  content-coupling (CC, attends by inner-product similarity) vs. positional-coupling (PC,
  attends by relative offset) and predicts rotational energy fraction correlates with this
  2D map. Includes anti-similarity heads as a *second* relational class distinct from
  induction — heads with high imaginary OV fraction but low induction score, negative CC —
  because collapsing all relational computation into "induction" would miss a mechanism that
  could drive merge events through the attention-routing channel specifically (complementary
  to, not the same as, Phase 2's V-repulsive energy mechanism).
- **Part B (eigenspace degeneracy)** — the direct, SAE-free, theory-grounded definition of
  a metastable cluster: tokens whose projections onto $S$'s dominant attractive eigenvectors
  are nearly identical are dynamically equivalent. This reframes "is HDBSCAN finding
  something real" as a testable geometric claim rather than trusting the clustering
  algorithm's output on faith — B.2's degeneracy ratio, B.3's LDA-vs-$S$ alignment, B.4's
  centroid velocity decomposition, and B.5's local contraction analysis are four independent
  angles on the same underlying claim.
- **Part C (residual stream subspace channels)** — mechanistic-interpretability-style
  per-head write-subspace analysis (C.1/C.2), culminating in **C.3, the double dissociation**
  — described as the single most falsifiable prediction in the phase: zero the imaginary
  channel (induction should drop, clusters preserved) and zero the real channel (clusters
  disrupted, induction preserved) as two surgical interventions, with a random-subspace
  control to confirm any effect is channel-specific rather than generic damage. This is more
  direct than Phase 2b's rescaled frames — it intervenes on activations during inference
  rather than removing the operator's contribution retroactively.
- **Part D (metastable states without SAEs)** — explicitly positions this phase's direct
  geometric tests as the answer to a limitation named back in Phase 3/4: sparsity pressure
  allocates dictionary capacity to frequent independent features, and metastable cluster
  membership may not decompose into independent sparse features — not because the structure
  is absent, but because sparse coding is the wrong prior. D.3 lays out three possible
  readings once both geometric and SAE results exist (geometric succeeds/SAE fails → SAE
  prior was wrong for this structure; both fail → clusters aren't linearly encoded, which
  would be surprising given HDBSCAN's success; both succeed → SAE features are proxies for
  $S$-projections).

## Why ALBERT is a strengthening test, not a confound, for the shared-weight case

For ALBERT, the same $W_Q, W_K, W_V, W_O$ are reused at every iteration, so Schur
decomposition and head classification are computed once. This is framed as a *cleaner* test
of the functional-separation hypothesis, not a weaker one in principle: if the same weights
implement both channels, functional separation must arise from which subspace the incoming
activation occupies, not from separate weight matrices — a cleaner test of whether the
residual stream itself is partitioned. In practice, though, the first-run result (below)
shows this cuts the other way empirically for ALBERT specifically, and the design document
itself flags that the 49-layer consistency isn't 49 independent measurements when the
projector is identical across all of them.

## First-run result and how it's being read

Track A (behavioral/causal) produced no data — blocked on wiring bugs (see STATUS.md),
not on a null finding. Track B/D's LDA alignment and linear-probe results are the
substantive first finding, and they invert the prediction: cluster-separating directions
align overwhelmingly with $U_A$ (imaginary, 0.887) not $U_\text{neg}$ (real repulsive, 0.067),
and imaginary-only probes (0.564) nearly match full-activation accuracy (0.590) while
real-only probes sit at chance (0.152). This is treated as a genuine open question with two
live explanations rather than a settled falsification of the phase's central hypothesis:
either a projector-construction/labeling bug in `subspace_build.py` swaps $U_\text{neg}$ and
$U_A$ (which would invert all four geometry tests together, exactly as observed), or the
hypothesis is wrong specifically under ALBERT's weight-tying. The design explicitly
prioritizes ruling out the first (a Schur sign/block-type convention check) before treating
the second as established, since a labeling bug would produce exactly this pattern of
results and is checkable in isolation. P6-R5's partial pass (contraction in $S$ during
plateau, near-neutral rotation in $A$, as predicted) is the one result with positive signal,
though the merge-destabilization half fails — possibly a layer-type-label mismatch between
Phase 1's per-layer plateau/merge classification and this phase's per-step dynamic labeling,
not necessarily a hypothesis failure.

## Module structure

`head_classify.py` (A.2), `induction_ov.py`/`induction_detect.py` (A.3), `qk_decompose.py`
(A.4), `eigenspace_degeneracy.py` (B.2 — has the known NameError bug), `centroid_velocity.py`
(B.4), `local_contraction.py` (B.5), `write_subspace.py` (C.1/C.2 — has the known `top_r`
kwarg bug), `dissociation.py` (C.3), `probe_subspace.py` (D.2.4), `run_6.py` (CLI),
`report_6.py`. External, read-only imports: `p2b_imaginary/rotational_schur.py` (Schur
blocks, $U_A$/$U_S$), `p2_eigenspectra/weights.py` (per-head OV, $W_Q$/$W_K$/$W_O$),
`p2_eigenspectra/decompose.py` (attn/FFN deltas for intervention setup).

## Why this phase, not Phase 5, owns the real/imaginary question

Phase 5's Group E (tuned-lens preview) originally anticipated absorbing some of this
scope; the current README corrects that — Phase 6 is specifically the real/imaginary
subspace decomposition question, a different question from Phase 5's semantic-decode
enrichment, and the two shouldn't be bundled behind one flag or one phase's report.
