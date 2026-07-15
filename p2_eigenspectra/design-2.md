# Phase 2 — DESIGN

## Core question

Phase 1 found energy violations — layers where $E_\beta$ fails to decrease monotonically —
consistent with the paper's V-repulsive mechanism. Phase 2 asks whether V's mixed-sign
eigenspectrum is the actual cause, not just a consistent post-hoc story.

## Why three causal tests, not one

A single test (e.g., just checking whether displacement projects onto V's repulsive
subspace) can't distinguish "V is locally detectable" from "V is globally causal but the
local signal is masked by noise or channel routing." Three tests target different failure
modes of that ambiguity:

1. **Displacement test (local)** — direct: project $\Delta x$ onto attractive/repulsive
   eigensubspaces at violation layers.
2. **Rescaled frame (global)** — causal intervention in representation space: apply
   $z = e^{-tV}x$ and check whether violations disappear. This catches cases where V is
   causal but the effect is distributed rather than locally dominant.
3. **FFN/attention decomposition (channel)** — splits the residual update to identify
   *which* pathway (attention vs. FFN) carries V's effect into the trajectory.

The continuous V-score combining all three (rather than a single hard threshold) exists
because the two regimes found empirically — locally detectable vs. globally-coherent-but-
FFN-mediated — need a common scale to compare across models with very different
architectures (ALBERT's shared weights vs. GPT-2's per-layer weights).

## Verdict logic design

Classification priority (`V_repulsive_local` takes precedence over all rescaling-based
verdicts) exists because a run where the displacement test directly passes shouldn't be
downgraded by a noisier global measure — direct evidence outranks indirect evidence when
both are available. `analysis_extended.py`'s `_classify` is deliberately the single source
of truth for the categorical verdict; `verdict_v2.py`'s `_classify` is kept only as a
back-compat shim so old call sites don't break, not as a second authority.

`V_repulsive_via_attn` exists in the logic (rescaling helps, FFN doesn't dominate the drop)
but has never fired across 35 runs — kept rather than removed because a future model or
prompt type could exercise it, and removing an unused-but-reachable branch would silently
narrow the classifier's coverage.

## Two regimes, architecturally grounded

- **Regime A (locally detectable):** ALBERT-xlarge, GPT-2-xl, GPT-2-large (partial). V's
  repulsive structure is directly visible in the trajectory. ALBERT-xlarge's channel is
  attention-dominant, consistent with its shared-weight architecture — the OV circuit acts
  directly without layer-specific FFN amplification, so there's nothing else it could route
  through.
- **Regime B (globally coherent, FFN-mediated):** GPT-2-small, GPT-2-medium. The
  displacement test fails locally but the rescaled frame eliminates violations, with FFN as
  the proximal dropper. This regime split (A vs. B) becomes the organizing frame for Phase 3
  (model selection), Phase 4, and Phase 5's cross-track interpretation.

This regime distinction is why Phase 3 selects exactly ALBERT-xlarge (Regime A) and
GPT-2-large (Regime B) rather than running the full model grid again — the two regimes are
the meaningfully different conditions to test crosscoder behavior against.

## OV norm confound

`ov_norm_partial_rho` — Spearman correlation of OV norm vs. violation indicator after
controlling for `rep_frac` — exists because early/final-layer OV norm spikes (up to 22× mean
at GPT-2-small L11, likely the unembedding projection) can produce large displacements that
get misattributed to the repulsive subspace. This is why the rescaled-frame result, not the
raw displacement test, is treated as the more trustworthy signal where the two disagree.

## Module structure

- `weights.py` — pure weight decomposition (composed OV, Schur + symmetric eigendecomposition,
  subspace projectors, QK norms). No inference, so it's reusable by Phase 2b/2c/6 without
  re-deriving projectors.
- `trajectory.py` vs. `trajectory_perlayer.py` — split because ALBERT has one shared V and
  GPT-2/BERT need each layer's own V; keeping them separate avoids a single function branching
  on architecture internally.
- `decompose.py` — the GPT-2-only producer for attn/FFN residual splitting. This is the file
  frozen against Pythia's parallel-residual architecture (no post-attention/pre-FFN
  intermediate exists to decompose there).
- `cross_term_analysis.py` — added after finding the additive two-way (attn+FFN) decomposition
  misses a dominant cross-term for ALBERT-xlarge on several prompts; kept as a separate module
  rather than folded into `decompose.py` since it's a genuinely different decomposition
  (three-way, not two-way).
- `analysis_extended.py` — authoritative verdict classifier (see above).
- `subresult.py` / `subexperiments.py` / `subexp_wrappers.py` — registry pattern so each
  analysis module reports through a typed contract (`SubResult`) rather than the verdict
  assembler needing to know every module's raw output shape. This is also what lets `--full`
  vs. `--offline` mode skip subexperiments cleanly, and lets one subexperiment error without
  aborting the run.
- `head_ablation.py` / `threshold_analysis.py` — causal (per-head) and sensitivity (β-sweep)
  robustness checks on top of the main verdict, not required for the verdict itself.

## v2 plan: the parallel-residual decomposition is an upgrade, not a gap

The FFN/attention decomposition (`decompose.py` and its consumers) is sequential-architecture
specific: it relies on there being a real post-attention, pre-FFN intermediate state, which
only exists because GPT-2/ALBERT/BERT compute FFN *after* adding attention's output to the
residual. Pythia's GPT-NeoX architecture computes both from the same pre-block input and
sums them in parallel — there is no intermediate to extract, so this module's literal
approach cannot be ported.

The v2 plan reframes this rather than just freezing around it: a parallel-residual model
makes Δx = attn_out + ffn_out an *exact* identity with both terms computed from the same
input, which is a strictly cleaner decomposition than this phase's sequential one ever gave
— GPT-2's decomposition always carried an implicit ordering assumption (attention's output
already having modified the residual before FFN sees it) that Pythia's architecture doesn't
have. This is why a new, separate module belongs in `core/` rather than as a Pythia branch
inside `decompose.py`: it isn't the same computation with a conditional, it's a genuinely
different (and simpler) decomposition that happens to answer the same question. Once that
module exists, the FFN-vs-V question this phase asked for GPT-2 can be asked again, natively,
on Pythia — not just inherited as a frozen reference point.

## Bugs fixed (historical, kept for record)

1. `_confirmed` upgrade previously fired on `channel == "unknown"` (zero decompose coverage)
   because the guard checked `channel != "attention"` rather than `channel == "FFN"`
   explicitly. Fixed.
2. `FFN_independent` guard previously used raw violation count instead of decomposed-violation
   count, allowing an n=1 verdict. Fixed to require `n_decomposed ≥ 3`.
3. ALBERT-xlarge's dominant cross-term mechanism was invisible to the additive decomposition;
   `cross_term_analysis.py` added to address it.
4. The "coupling product" metric was confirmed useless and removed.
