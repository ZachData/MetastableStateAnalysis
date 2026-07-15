# Phase 2b — DESIGN

*(Documented in its own README and result filenames as "Phase 2b" — see status-2b.md for
the naming reconciliation. "Phase 2b" is used throughout below.)*

## Core question

Phase 2 found OV matrices are ~98% complex (imaginary eigenvalue dimensions) across all
models. This raises a direct question about Phase 2's own conclusions: if the dominant
spectral feature is rotational rather than signed, does rotation contribute to energy
violations, or could the imaginary structure be confounding the signed-component
attribution? Phase 2b asks: is the rotational component of OV causally responsible for any
energy violations, or is it dynamically neutral?

## Why the S/A decomposition

$V_\text{eff} = S + A$ splits OV into $S = (V+V^\top)/2$ (symmetric, real eigenvalues — the
component Phase 2 attributes violations to) and $A = (V-V^\top)/2$ (antisymmetric, pure
rotation). This is the natural decomposition because it isolates exactly the two
possibilities: either the signed residue (small fraction of spectral energy) does all the
causal work, or the rotational majority also contributes. Applying it *causally* (via
rescaled frames) rather than just characterizing the spectrum descriptively is what makes
this a test of Phase 2's conclusions rather than a restatement of them.

## Block design

- **Block 1a (`rotational_schur.py`)** — pure weight analysis, no activations needed.
  Characterizes the complex eigenvalue structure per model: fraction of complex dimensions,
  rotational vs. signed energy fraction, Henrici non-normality (quantifies how informative
  the S/A split actually is — high non-normality means S and A aren't just decorative).
- **Block 1b (`rotational_rescaled.py`)** — the decision point. Three rescaled frames (full,
  signed-only, rotation-only) recompute energy violations; comparing elimination rates
  across frames directly tests causal weight rather than mere presence.
- **Block 2 (conditional, not run)** — Fiedler tracking and rotation-hemisphere alignment,
  gated on Block 1b returning `rotation_contributes` for any model. Designed this way so
  that expensive hemispheric analysis is only run if there's a positive signal to explain;
  the uniform `rotation_neutral` result means it was correctly never triggered.
- **FFN rotation (`ffn_rotation.py`, conditional)** — tests whether FFN selectively amplifies
  rotation planes at violation layers, which would mean removing $A$ in the rescaled frame
  is insufficient because FFN re-introduces rotational displacement. Also gated on Block 1b.

## Interpretation of the result

The uniform `rotation_neutral` finding across all 35 combinations, all β values, and both
encoder/decoder architectures means:

1. Phase 2's conclusions are not confounded by the imaginary structure — the ~98%
   complex-dimension observation doesn't undermine the repulsive-subspace attribution; the
   signed residue carries 100% of causal weight regardless of how structurally minor it is.
2. ALBERT-xlarge's violations (26+ on `repeated_tokens`) are sustained by a ~2.5% signed
   residue in a 1024-dimensional matrix — the mechanism is highly concentrated, not diffuse.

This "rotation dominant structurally, signed dominant causally" pattern holding without
exception from 12-layer to 48-layer models, encoder and decoder alike, is what licenses
treating it as a closed finding rather than a per-model curiosity.

## Why this phase is closed, and why Phase 2c is separate rather than a reopening

Phase 2c (trajectory-side dynamical systems analysis) exists precisely because Phase 2b's
null answers one question — does rotation drive *clustering*? — without asking whether the
rotational capacity is used for anything else. A subspace orthogonal to clustering dynamics
isn't necessarily inert; it's free to carry other computation without perturbing the
attractor structure Phase 2b tested. Rather than adding new metrics to this phase's analysis
(which would blur a complete experiment with a unanimous null), Phase 2c is a new directory
that only *imports* this phase's Schur projectors ($U_A$, $U_S$) as read-only inputs. See
Phase 2c's own DESIGN.md for the full reasoning.

## Known caveats baked into the design

- ALBERT's full-rescaling overcorrection is architecture-specific: the single shared $V$
  applied once per iteration step (rather than once per layer) interacts with $S$ in the
  rescaled frame in a way that isn't additively separable. This doesn't change the causal
  conclusion (signed-only rescaling is still clean) — it's a diagnostic-reliability caveat
  on one of three frames, not on the result.
- The `gpt2` (small) naming collision in `phase2i_results.json` is a runner bug, not a
  result — per-model entries are authoritative and the aggregation isn't relied on anywhere
  downstream.

## Module structure

`rotational_schur.py` (1a), `rotational_rescaled.py` (1b), `fiedler_tracking.py` /
`rotation_hemisphere.py` (2, conditional), `ffn_rotation.py` (conditional, needs Phase 2 FFN
deltas), `run_2i.py` (orchestrates 1a → 1b → conditional 2 → conditional FFN rotation, loads
Phase 1/2 artifacts, writes per-model JSON plus combined `phase2i_results.json`).
