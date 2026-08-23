# Phase 2 — DESIGN

> **Scope note (2026-08-04).** Everything from "Core question" through "Bugs fixed" is the
> design rationale for **Study A**, the pre-Pythia GPT-2/ALBERT/BERT run, and is unchanged.
> The sections under "Study B — what the Pythia sweep changed about this design" are new and
> cover why the same code behaves differently on a parallel-residual architecture. Read both
> before touching the classifier or the v-score. See `status-2.md` for results and for the
> open verification items that gate them.

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

---

# Study B — what the Pythia sweep changed about this design

Study A's design assumed a sequential-residual architecture and a working global
intervention. Pythia has neither. The code runs and produces well-formed output regardless,
which is the problem this section exists to document: **several columns degrade silently
into defaults rather than erroring, and the resulting values look like measurements.**

## Classifier reachability on a parallel-residual architecture

`_classify` (`analysis_extended.py:708`) has eight outcomes. On Pythia only three are
reachable, and the reason differs by branch:

| Branch | Reachable on Pythia? | Why |
|---|---|---|
| `no_violations` | yes | — |
| `overshoot_dominant` | yes | never fires empirically, same as Study A |
| `V_repulsive_local` | yes | the only positive verdict that fires |
| `V_repulsive_via_FFN` | **no** | requires `decompose_frac_ffn_drop > 0.5`; `decompose.py` frozen |
| `V_repulsive_via_FFN_confirmed` | **no** | additionally requires `channel == "FFN"` |
| `FFN_independent` | **no** | requires `decompose_frac_ffn_drop > 0.5` and `n_decomposed ≥ 3` |
| `V_repulsive_via_attn` | in principle | requires `rescaled_frac > 0.8`; never observed |
| `mixed_or_unattributed` | yes | absorbs everything the above miss |

The consequence for interpretation is sharp and easy to get wrong: **`mixed_or_unattributed`
on Pythia does not mean what it means on GPT-2.** On GPT-2 it was a residual bucket after
four substantive branches had a chance to fire. On Pythia it is the residual bucket after
*one*. A Pythia `mixed_or_unattributed` run means only `frac_repulsive ≤ 0.5` — it carries no
information about channel, and it is not evidence that the effect is unattributable.

This is why status-2.md's Study B section reports the continuous `frac_repulsive` curve as
the primary result and treats the verdict labels as secondary. The labels are a
coarse-graining calibrated for a classifier that isn't running.

## Why the degradation is silent, and what to do about it

Two independent fall-throughs produce plausible-looking values from missing data:

- **`channel`.** `subexp_wrappers.py:221` assigns `"FFN"` above 0.6, `"attention"` above 0.6,
  and `"mixed"` otherwise. An empty decompose result gives both fractions 0.0, which lands in
  the `else`. So `"mixed"` is what "no data" looks like, and it is indistinguishable in the
  output from a genuine even split. `verdict_v2.py:154` sets `"unknown"` as the initial value
  precisely to make missing data visible; the decompose wrapper then overwrites it with
  `"mixed"` on the way through. **Fix: make the `else` branch conditional on `n > 0` and emit
  `"unknown"` otherwise.** Until that lands, treat every Pythia `channel` value as absent.
- **`frac_ffn_amplifies_repulsive`.** Absent on Pythia, and `build_v_score` reads it with
  `verdict.get(..., 0.0)`. A missing term becomes a zero contribution rather than an
  incomparable score. See below.

The general lesson, and the reason this section exists rather than a one-line comment in the
code: **a metric that has a defined value when its input is missing will be plotted.** Every
scalar in the verdict contract needs either a sentinel that propagates or an explicit
applicability flag. The `SubResult.applicable` field already exists for this; the verdict
assembler doesn't consult it when flattening `verdict_contribution`.

## Why `v_score` is not portable across architectures

`build_v_score` (`verdict_v2.py:45`) is a fixed-weight sum:

```
0.40 · rescaled_frac  +  0.25 · frac_repulsive_disp  +  0.20 · frac_ffn_amp  −  0.15 · |ov_norm_partial_rho|
```

The weights were chosen to express a theory-motivated ordering of evidence strength — global
intervention beats local detection beats confirmatory channel evidence. That ordering is
sound. What the design did not anticipate is a model on which **two of the three positive
terms are structurally unavailable**, leaving a score whose ceiling is 0.65 and whose
observed variance is entirely explained by the remaining two columns.

Empirically on Pythia: `frac_ffn_amp` is 0 in all 243 runs, `rescaled_frac` is 0 in 134 of
the 153 runs with violations, and the score reduces to `0.25·fr − 0.15·|ρ|` to within 0.002.
It is a rearrangement of two columns that are already printed.

Three options, in preference order:

1. **Renormalize over available terms** and report coverage alongside the score, so a
   two-term score is on the same [0,1] scale as a four-term one and the reader knows which.
2. **Emit `None` when any term is unavailable**, forcing the comparison to happen on the
   component columns.
3. Leave as-is and never compare across architectures. This is the status quo and it has
   already produced one cross-study table that had to be caught by hand.

Until one of these lands: **Study A's calibration points — "above ~0.5 corresponds to
`_confirmed` or `_local`", and the 0.455–0.486 GPT-2-large borderline band — do not apply to
Pythia**, because Pythia cannot reach either number.

## The rescaled frame: two design flaws the Pythia result exposed

The global intervention is the highest-weighted term in the v-score and Study A treated it as
the more trustworthy of the two main tests. On Pythia it eliminates 2.1% of violations. Two
properties of the implementation make that number un-interpretable as reported, and both are
design problems rather than bugs:

**The failure is unobservable in the output.** `rescaled_trajectory_perlayer` builds
$R_L = e^{-OV_0} \cdots e^{-OV_{L-1}}$ as a running product and breaks out when it goes
non-finite or exceeds 1e15, recording how far it got in `n_valid_layers` (line 324). That
field is not propagated into the verdict contract and does not appear in the cross-run
summary. A run that truncated at layer 6 and a run that completed all 24 produce
indistinguishable summary rows. **`n_valid_layers` belongs in `verdict_contribution`**, and a
run where it is less than the layer count should be flagged the way
`decompose_coverage_warning` (`analysis_extended.py:671`) already flags its analogue.

The reason to expect trouble specifically here: the cumulative product is over 24 layers of
$e^{-OV}$, and its conditioning depends on OV's spectral norm in a way that has no reason to
be architecture-invariant. Study A's models had this path exercised at 12–48 layers on
weights with different scaling; that it stayed finite there is not evidence it stays finite
on GPT-NeoX.

**The metric is clipped at zero.** `improvement = max(0, n_phase1 − n_rescaled)`
(`analysis_p2.py:153`). The clip exists so that a noisier rescaled trajectory doesn't produce
a negative "improvement" that reads as evidence against V. But it also means **rescaling that
actively makes violations worse is reported as rescaling that does nothing** — and those are
opposite results. Overcorrection is a known real mode, documented for ALBERT in status-2b
caveat 1 (`elim_full` going negative under shared weights). The underlying
`n_rescaled_violations` is retained in the same dict; the clip should move to presentation,
not storage, and the raw signed difference should reach the verdict.

## What this means for the Regime A / Regime B frame

Regime B is defined by a conjunction: displacement test fails locally **and** rescaled frame
eliminates violations **and** FFN is the proximal dropper. On Pythia the second conjunct is
never satisfied and the third is not measurable. So no Pythia run can be classified as
Regime B — by construction.

**This is not evidence that Regime B is absent on Pythia.** It is an instrument that cannot
detect it. The distinction matters because the Regime A/B split is the organizing frame for
Phase 3's model selection and Phase 5's cross-track interpretation; a premature "Pythia is
Regime A only" would propagate into all of them.

The frame becomes testable on Pythia when two things land: the parallel-residual
decomposition module (restoring the FFN measurement) and a working global intervention
(either the numerical fix, or Phase 2b's signed-only rescaling substituted for full-V). Phase
2b's result — OV is 84–97% rotational energy but the signed component carries 100% of causal
weight — makes the substitution the more promising of the two, since the full-V matrix
exponential is exponentiating a matrix that is mostly rotation and mostly irrelevant.

## Why the checkpoint sweep changes the phase's unit of analysis

Study A's unit was a model×prompt run, and the verdict was the deliverable. Study B has 27
checkpoints of one model, and the interesting object is a **curve**, not a verdict: violation
count against training step, and `frac_repulsive` against training step, which turn out to
have different shapes and different timescales.

The categorical verdict actively obscures this. The 40000–100000 flip from
`V_repulsive_local` to `mixed_or_unattributed` is a smooth `frac_repulsive` decay crossing a
hard 0.5 guard, with five of the flipped runs sitting at exactly 0.500. Reported as a
verdict, it looks like a regime change. Reported as a curve, it is a monotone decay and a
partial rebound.

This is a reporting-layer change, not a classifier change — the verdict stays as the
cross-run comparable summary. But **`reporting_p2.py`'s cross-run summary should carry the
component columns as first-class output** (`rescaled_improvement`, `n_rescaled_violations`,
`n_valid_layers`, `decompose_n_violations`) rather than only the derived v-score and verdict.
Every substantive finding in Study B had to be reconstructed by algebra from the v-score
because the components weren't printed. That reconstruction happened to be exact; it should
not have been necessary.
