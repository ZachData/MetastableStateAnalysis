# Phase 2c — DESIGN

## Placement: why a new directory, not an extension of Phase 2b

Phase 2b analyzed the operator: Schur decomposition of $V$, causal rescaling by $e^{-tS}$ and
$e^{-tA}$. It established $A$ is dynamically neutral *for clustering* — it did not test
whether $A$ is used by any other computation. Phase 2c analyzes the trajectory instead,
importing neuroscience population-recording methods (jPCA, tangling, CIS, slow points,
context-dependent subspace selection) and applying them to layer-by-layer activation
trajectories, independent of the operator-spectrum analysis.

Four reasons this is `p2c_churchland/`, not a `p2b_imaginary/` extension:

1. **Operator-side vs. data-side.** p2b decomposes weights; p2c fits dynamical models to
   activations — different objects, different machinery.
2. **p2b is closed.** It was a complete experiment with a unanimous null; reopening it for
   new metrics would blur that conclusion.
3. **Imports, not refactors.** p2c reads $U_A$/$U_S$ projectors from
   `p2b_imaginary/rotational_schur.py` and displacement helpers from `phase2/trajectory.py`
   as read-only inputs. No existing file needs modification.
4. **Falsifiable separation.** If p2c finds rotational signatures in trajectories that p2b's
   operator analysis doesn't predict, that misalignment is itself the result — mixing the
   two analyses would obscure it.

## Core question

The operator has 84–97% of its spectral energy in rotation blocks, but the antisymmetric
component doesn't drive clustering (Phase 2b). Three possibilities this phase distinguishes:

1. Trajectories use $V$'s rotational capacity for non-clustering computation (Δx projects
   onto $U_A$ planes; jPCA recovers the same planes).
2. Trajectories rotate, but in planes orthogonal to the operator's $U_A$ (rotation emerges
   from softmax+FFN composition, not from $V$ itself).
3. No rotational structure in trajectories at all — the operator's 97% is a representational
   accident of high-dimensional non-normal matrices, unused computationally.

## Five imported methods and why each was chosen

Each method is a specific, named neuroscience result with a precise transformer-side port
and a precommitted falsifier — the phase is built around avoiding the failure mode of
over-interpreting a confirming result (see Regime-Mismatch Caveat below).

- **C1 (jPCA, Churchland et al. 2012).** Fits $\dot X = MX$ with skew-symmetric $M$ to
  layer-to-layer Δx; the $R^2$ ratio (constrained vs. unconstrained) measures how rotational
  the *data* actually is, independent of the operator. Directly comparable to $U_A$ via
  principal angles — this comparison is the whole point of doing jPCA rather than just
  reporting whether the data looks rotational in isolation.
- **C2 (tangling, Russo et al. 2018).** $Q(t)$ penalizes nearby states with divergent
  derivatives. Chosen as a descriptive contrast across S/A-projected channels rather than as
  a standalone autonomy test, because the regime-mismatch caveat (below) means the absolute
  magnitude isn't theoretically grounded for a non-autonomous, layer-discrete system — only
  the relative comparison across channels and prompt types is.
- **C3 (CIS, Kaufman et al. 2016).** Condition-invariant vs. condition-specific variance
  decomposition. Directly tests the sharper, variance-decomposed version of a
  division-of-labor claim (invariant → A, specific → S) using the same projectors as C1.
- **C4 (slow points, Sussillo & Barak 2013).** Local Jacobian at Phase 1's known metastable
  centroids (no gradient-descent slow-point search needed, since the candidate points are
  already known). Chosen to run **first** in the recommended order — cheapest (reuses Phase 1
  centroids, no new prompts) and answers a live methodological gap: Phase 2b's global
  rescaling used the *global* $V$; if local Jacobians differ substantially, the global null
  is silent on what local linearization would show.
- **C5 (Mante et al. 2013, context-dependent subspace selection).** The cleanest available
  analog to in-context learning: does A-channel projection scale with k-shot count, and is
  the scaling direction task-specific? Requires purpose-built prompt grids (not the existing
  8-prompt set), since the original result depends on tightly controlled context/stimulus
  separation that arbitrary prompts don't give.

## Recommended order and why

C4 → C1 → C5 → C2 → C3, ranked by (cost, likelihood of overturning or refining an existing
conclusion), not by presentation order. C3 is last because it requires the most carefully
controlled prompt construction and is most informative once C1/C5 already point somewhere
specific.

## Regime-mismatch caveat (load-bearing, not a footnote)

All five methods assume continuous-time autonomous recurrent dynamics from population
spike-rate recordings. Transformers are layer-discrete, non-autonomous (attention re-weights
every layer based on full token state), have a *known* operator (unlike neuroscience, where
the operator is inferred), and "trajectory across depth" has no physical $dt$. Numerical
computability isn't the issue — theoretical grounding is. Every prediction should be read as
conditional: *if* transformer dynamics share the relevant structural features with
motor-cortex dynamics, *this* is what we'd observe. A disconfirming result is more
informative than a confirming one, since it rules out specific mechanisms regardless of
whether the broader analogy holds. This framing is why the actual result (mostly FAILS
across models, with scattered partial HOLDS) is read as "the clean division-of-labor story
does not cohere" rather than as a stronger claim that rotation is inert everywhere.

## Module structure

`jpca_fit.py` / `jpca_alignment.py` / `hdr_fit.py` (C1 + HDR fallback when jPCA is
borderline — HDR sidesteps jPCA's cross-condition-mean centering issue), `tangling.py` (C2),
`cis_decompose.py` (C3), `local_jacobian.py` / `slow_point_compare.py` (C4),
`icl_subspace_scaling.py` / `context_selection.py` (C5), `prompt_grids/` (purpose-built
prompt sets for C3/C5 — the existing 8-prompt set is unsuitable for controlled-variation
designs), `run_2c.py` (CLI), `report_2c.py` (flat report).

## ALBERT vs. per-layer models

For ALBERT (shared weights), the Schur decomposition gives one $U_A$ per model and "layer"
means iteration index — the same forward function is applied at every iteration, so any
divergence between local (C4) and global S/A ratio is purely an effect of *where* in state
space the linearization is taken, not a change in the operator. This is actually the
cleanest test bed for the local-vs-global comparison. Per-layer models (GPT-2, BERT) have
their own $V_L$ per layer and require per-layer Jacobians for C4.

## Current-status caveat

The `readme-phase2c.md` header text ("Not started") is stale relative to actual run output
(see STATUS.md) — treat the results file as ground truth until the header is corrected.
