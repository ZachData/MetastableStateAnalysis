# Phase 2b — REWRITE PLAN (living document)

Updated as work lands. `status-2b.md` is the phase's verdict record; this file is
the work tracker. Anything marked DONE has passing tests named next to it.

**Last updated:** session 4, after the per-head redesign. 187 tests passing.
**Test command:** `python3 -m unittest discover -s tests -p "test_phase2b_*.py"`
(plain unittest — no torch, no pytest plugins; the degeneracy gate is passed
explicitly so `core.config` is never imported).

---

## Why this phase reopened

Block 1b's headline — `elim_rotation = 0.0` in 35/35 runs, read as "rotation is
dynamically neutral" — is an algebraic identity. `A = (V − V^T)/2` is real
antisymmetric, so `e^{−A}` is orthogonal, and every quantity Block 1b measures is
a function of `X X^T`, which an orthogonal map preserves exactly. Residual over 24
accumulated layers at d = 1024: ~1e-15, against a 1e-3 relative threshold.

Everything else in this plan follows from two facts: that finding must be
withdrawn, and the phase was written before the Pythia rebuild (checkpoint axis,
frame ledger, artifact contract, `core/nulls.py`, `core/precision_policy.py`,
`core/sublayer_streams.py`, `core/intervention.py`, `core/functional_distance.py`).

---

## Status board

| # | Item | State | Lands in |
|---|---|---|---|
| 1 | Withdraw `rotation_neutral` from the docs | **DONE** | `status-2b.md` |
| 2 | Single violation-counting rule | **DONE** | `p2b_imaginary/p2b_energy.py` |
| 3 | Block 1b rewrite — frames, truncation, refusals | **DONE** | `p2b_imaginary/rotational_rescaled.py` |
| 4 | Discovery / loading / contract / manifest | **DONE** | `p2b_imaginary/p2b_io.py` |
| 5 | Checkpoint name grammar out of the viz module | **DONE** | `core/model_family.py` |
| 6 | `PHASE2B` artifact registry | **DONE** | `core/artifacts.py` |
| 7 | Block 1a rewrite — memory, conventions, θ, nulls, precision | **DONE** | `p2b_imaginary/rotational_schur.py` |
| 8 | `run_2b.py` — checkpoint loop, wiring, manifests | **DONE** | `p2b_imaginary/run_2b.py` |
| 9 | Cross-checkpoint aggregation + trajectory report | **DONE** | `p2b_imaginary/p2b_report.py` |
| 10 | Block 3 redefined — forward-pass operator substitution | TODO | `imaginary_ablation.py` |
| 19 | Per-head circuit algebra, factored | **DONE** | `p2b_imaginary/head_circuits.py` |
| 11 | Block 4 redefined (curvature, saturation, γ, cost) | TODO | `layernorm_jacobian.py` |
| 12 | Block 2 gate rewritten (its trigger was a constant) | TODO | `fiedler_tracking.py`, `rotation_hemisphere.py` |
| 13 | FFN rotation on parallel-residual streams | TODO | `ffn_rotation.py` |
| 14 | The real rotation test (readout / weight-space) | TODO | new module |
| 15 | Patch `core/precision_policy.py`'s docstring | **DONE** | `core/precision_policy.py` |
| 16 | Point `precision_policy._default_frac_fn` at the single definition | **DONE** | `core/precision_policy.py` |
| 17 | `p1_mstate_tracking/visualization/checkpoints.py` re-exports the grammar | **DONE** | that file |
| 18 | Strike status-2.md's third explanation | TODO | `status-2.md` |

---

## Landed

### 2 — `p2b_energy.py`

The phase's only violation counter. Was three hardcoded copies of an absolute
`−1e-6` threshold with an `eff_rank >= 3.0` gate; the project's rule is
`core.metrics.ENERGY_VIOLATION_REL_TOL = 1e-3` relative with
`core.config.DEGENERATE_RANK_THRESHOLD = 2` (status-1 defects D7 and D8).
Effective rank now uses squared singular values via `core.metrics.effective_rank`;
the local version normalized unsquared ones — same name, different statistic,
used as a gate.

`n_transitions_scored` is a first-class output because it is the denominator.
`elimination_rate` is unclipped and returns `None` with a status for four
refusals: `no_violations_to_eliminate`, `different_transitions_scored`,
`no_transitions_scored`, `different_counting_rule`.

Gates on **normed** rank throughout and says so in every record (`gate_kind`).
Phase 1 gates on raw; Phase 2b cannot, because the rescaled frames' norms are an
artifact of the rescaling. `cross_check_against_phase1` surfaces the resulting
divergence as a number instead of leaving it inside every elimination rate.

Tests: `TestCountingRule`, `TestEliminationRate`.

### 3 — `rotational_rescaled.py`

Frames renamed by what they **remove**: `remove_full`, `remove_signed`,
`remove_rotation`. The last carries `is_invariance_control=True`, is excluded
from the causal comparison, and is audited by `_audit_invariance`
(`identity_holds` / `identity_broken`). Only `elim_full` vs `elim_signed` is a
measurement — `e^{−(S+A)} ≠ e^{−S}e^{−A}` unless S and A commute — and that
contrast is exactly status-2.md's "next experiments" item 2.

Per-frame `n_valid_layers` / `truncated` / `truncation_reason` survive
serialization (Phase 2's V1). `expm` cached per checkpoint via `rescaler_cache`;
it was being recomputed per prompt for a prompt-independent quantity.

Verdicts: `signed_carries_full_v`, `signed_exceeds_full_v`,
`full_v_exceeds_signed`, `both_frames_inert`, `no_violations`, `not_comparable`.
None names rotation. `overall` reads β = 1.0 rather than a majority vote (Phase 1
found counts β-independent after step 512; Study B ran β = 1.0 only), with
`beta_dispersion` alongside.

Tests: `TestRotationFrameIsAnIdentity`, `TestTruncationIsSurfaced`,
`TestInterpretation`, `TestGateDivergenceIsRefused`.

### 4 — `p2b_io.py`

Exact stem matching (never `stem in name`), checkpoint axis, delegation to
`p1_io.load_phase1_run`, artifact contract, `core.io.write_manifest`, FrameSpec,
`refuse_legacy_run_dir`.

Tests: `TestStemMatchingIsExact`, `TestCheckpointAxis`, `TestOvLoading`,
`TestPhase1Bundle`, `TestContractAndFrame`.

### 19 — `head_circuits.py`

The mathematical decision the rest of items 10, 11 and 14 rest on.

**The operator.** `ov_total = sum(ov_per_head)` (`weights.py:184`) is the
effective operator only under a counterfactual the model does not satisfy —
that every head shares an attention pattern. The real update is
`Σ_h α^h X W_OV^h`. So Phase 2b's headline is a statistic of an object the
model never forms. `summed_vs_per_head` reports both plus the gap and
`head_agreement`, because the disagreement is the finding; nothing here
adjudicates by fiat.

**The representation.** Per-head `W_OV` is rank `d_head` (64 of 1024), so the
dense `(d, d)` form `weights.py` saves is 16× larger than the information in
it. Three identities make the whole phase cheap, all pinned by tests:

| identity | consequence |
|---|---|
| `eig(W_O W_V) \ {0} = eig(W_V W_O)` (residual 0.0) | spectra are a `d_head²` problem, not `d_model²` |
| `S = B_S C`, `A = B_A C` with `B: (d, 2k)`, `C: (2k, d)` (exact to 2e-17) | `rank(S) ≤ 2·d_head`; applying it is O(n·d·2k) |
| `S` and `A` share `C` | swapping S for V costs one extra `(n, 2k)` matmul |

Per layer at 410m: `16 × 64³` against `1024³` — a 256× reduction, and the
ratio grows with model size, which is what makes pythia-1.4b reachable.

**A claim of mine that a test corrected.** The rank deficiency destroys
DIMENSION fractions (same head: 5.5% of dims rotating in `d`, 87.5% in its
core) but not ENERGY fractions — `|0|² = 0`, so the zeros contribute to
neither numerator nor denominator, and the two computations agree to 1e-6. The
published 84–97.5% is an energy fraction, so **the rank argument does not
overturn it.** The shared-attention argument does. `head_spectrum` no longer
carries a `complex_energy_fraction_ambient` field, since it would be the same
number.

**Block 4 falls out analytically.** `σ · J_LN` is an *orthogonal projector* of
rank `d−2` onto the complement of `span{1, x̂}` — `P² = P` to 1e-9, symmetric
exactly, trace exactly `d−2`. Both `1/d` terms are projectors because
`‖1‖² = ‖x̂‖² = d`. Therefore `1/σ` is a pure scale that cannot move any angle
or fraction, and the token-dependent content is exactly one rank-1 direction.
Everything else is `diag(γ)`: token-independent, one eigendecomposition per
**layer**, not per token per layer. Measured effect on the complex fraction:
0.944 → 0.940 against a shipped `H2_SUPPORTED` threshold of 1.5×.

Tests: `test_phase2b_head_circuits.py`.

### 9 — `p2b_report.py`

Cross-checkpoint aggregation. Phase 1 and Phase 2's dated events are
transcribed into `KNOWN_TRANSITIONS` as data, with their source document and
the quantity that moved, so a Block 1a trajectory can be laid against them
without re-reading a prose table.

Three refusals are built in:

- **No categorical change-point verdicts.** status-2's own headline warns that
  five of the 13 `mixed_or_unattributed` runs sit at `frac_repulsive` exactly
  0.500 against a strict `> 0.5` guard, so "the verdict label is an artifact of
  where the threshold happens to fall." Every function returns a continuous
  quantity and a rank.
- **`interval_rank` beside every alignment row.** A large move across a dated
  span means little if every span has one. Rank 1 of 20 with a large delta is
  co-location; rank 17 of 20 is not.
- **`not_bracketed` rather than a zero.** A span the sweep has fewer than two
  checkpoints inside is reported as unanswerable. status-1 notes the rank peak
  "sits unbracketed between 1000 and 3000."

`co_movement` exists for one question — does Henrici track Phase 2's
`frac_repulsive` decay — and says in its own output that it is the wrong tool
for a causal claim: two quantities that both drift with training will
correlate at the level, so `spearman_deltas` and `interval_agreement` are the
readings to look at.

Tests: `test_phase2b_report.py`.

### 15–17 — three small patches

`core/precision_policy.py`'s "What this does NOT reopen" section endorsed the
withdrawn `rotation_neutral` result; it now carries the correction and states
that `elim_signed` is being re-adjudicated separately. `_default_frac_fn` is
repointed from `layernorm_jacobian.rotational_fraction` to
`rotational_schur.complex_energy_fraction_relative` — same function, and the
repoint means `analyze_ov_precision` now runs without importing
`layernorm_jacobian`, which it could not before.
`p1_mstate_tracking/visualization/checkpoints.py` re-exports the step grammar
from `core/model_family.py` under the same leading-underscore names its four
figure consumers already call; its `import re` is gone.

### 8 — `run_2b.py`

Checkpoint axis is the outer loop; every artifact carries `checkpoint_step`
and the combined file is indexed by it. Weights-only work runs once per
checkpoint: Block 1a, and the `expm` rescalers, which are prompt-independent
and were being rebuilt inside every (model, prompt) pair — 27 × 9 × 3 × 24
exponentials of a 1024×1024 matrix where 27 × 3 × 24 suffice.

**Errors raise by default.** `run_2i.py` wrapped each prompt in a bare
`try/except Exception` that recorded `{"error": ...}` and continued; that is
how Block 4 shipped raising `NameError` on every prompt of every run while the
summary still wrote. `--continue-on-error` restores the old behaviour, but the
failure count is in the summary's first four lines and in `n_failed`.

**Blocks are not nested inside each other's gates.** `run_2i.py` placed Blocks
3 and 4 after `if not run_block2: return`, so on the (constant)
`rotation_neutral` verdict neither was reachable at all. Blocks 2–4 are absent
from `BLOCKS` until their maths is redefined; naming one is a `SystemExit`
pointing here.

Default is β = 1.0 alone, matching Study B, rather than four betas and a
majority vote.

Tests: `test_phase2b_runner.py` — `TestCheckpointAxisIsTheOuterLoop`,
`TestWeightsWorkHappensOncePerCheckpoint`, `TestErrorsAreNotSwallowed`,
`TestBlocksAreNotNested`, `TestArtifacts`, `TestSummary`, `TestCli`.

### 7 — `rotational_schur.py`

Memory: plane **bases** `(d, 2)` are stored; `(d, d)` projectors are never
materialized. `project_onto_planes` contracts through the basis. `schur_T` /
`schur_Z` are dropped unless `keep_factors=True`.

One energy convention: per-eigenvalue accounting (a 2×2 block contributes
`2ρ²`, matching what `henrici_nonnormality` already did in the same file).
`rotational_fraction_per_block` reproduces the old per-block number so the
84–97.5% figure stays checkable, under a name that says which it is.

θ on `[0, π]` — the old `arctan2(sqrt(−bc), abs(a))` folded repulsive rotations
onto their reflections. The 2×2 detection threshold is relative to `‖T‖_F`
rather than an absolute `1e-10`.

Nulls (`core/nulls.py` convention) and precision (`core/precision_policy.py`)
are wired in, since "84–97% complex" is not known to distinguish trained from
random and may be an fp16 threshold artifact.

Tests: `test_phase2b_schur.py`.

---

## The causal test, decided

Three candidates for replacing what the Block 1b withdrawal removed:

- **Weight surgery** — install `S` as 16 rank-64 heads by splitting its
  eigenvectors. Arithmetically possible (`rank(S) = 1024 = 16 × 64` exactly),
  but it redistributes eigenvectors across heads, so it changes which
  attention pattern acts on which subspace. That is a different model, not an
  ablation of `A`.
- **Post-hoc subspace projection of activations** — what Block 3 attempted.
  Not causal, and `col(A)` is the identity.
- **Forward-pass operator substitution** — hook attention, recompute
  `o' = Σ_h α^h X_LN S_h` through the factored form, continue the forward
  pass. Exact per head, no shared-attention assumption, no weight surgery,
  never forms `(d, d)`. **Chosen.**

This makes Block 3 (item 10) and the real rotation test (item 14) the same
instrument rather than two, which is why item 10 now names it.

## A scale error found while demonstrating the report

`collect_trajectory`'s `values` are MEANS over ~24 layers, so their sampling
noise is `spread / sqrt(n_layers)`, not `spread`. The first version divided
the trajectory's range by the raw layer spread — conservative, but the wrong
scale by a factor of ~4.9.

Worse, the obvious correction is also wrong: a 21-point series of iid noise has
a range of about **3.8 standard errors before any trend at all** (4.0 at
n = 27). Comparing a range against one standard error calls almost every flat
trajectory a transition. `flatness` now reports `range_excess_over_noise` —
range in standard errors divided by `expected_range_under_noise(n)`, simulated
rather than tabulated so the constant cannot drift from the n actually used.
Below 1.0, "it moves" is not supported. `range_in_spreads` is kept alongside as
the substantive comparison (trajectory against depth variation) rather than the
statistical one.

## Silent-absence failures found while building

Distinct from the manufactured-rate list below: these produce a result that is
*missing* rather than wrong, which is harder to notice.

- **A checkpoint with no OV weights vanished from the sweep.** Discovery is a
  glob over `ov_weights_*.npz`, so a step Phase 2 failed to write simply does
  not appear — 26 rows instead of 27, with nothing saying which. `run_sweep`
  now takes `expected_steps` (`--expect-registry-steps` pulls the canonical
  list from `core/pythia_registry.py`) and reports `missing_checkpoints` in the
  summary's first lines.
- **The absolute subdiagonal threshold in Block 1a had a partial regime.**
  Below `‖T‖ ~ 1e-11` every block reads 1×1 and the matrix looks entirely real;
  at ~1e-10 it mis-parses *some* blocks and the spectrum still looks plausible.
  Measured on a fixed 20×20 draw: true count 8, naive count 0 at scale 1e-11
  and 6 at 1e-10. Now relative to `‖T‖_F`.

## Three ways an elimination rate gets manufactured

Found while writing the guards; all three are now refusals rather than numbers.

1. **Overflow.** `e^{−S}` for S with negative eigenvalues diverges; the
   cumulative product is truncated. `e^{−A}` is orthogonal and cannot, so
   `elim_signed = 1.0` is precisely the value an early-truncating signed frame
   produces for free. → `truncation_reason="rescaler_overflow"`.
2. **Underflow (silent).** `e^{−S}` for positive-definite S contracts until rows
   fall below `l2_normalize`'s 1e-12 floor, after which it leaves them
   unnormalized, the Gram goes to ~0, every energy becomes the constant
   `1/(2β)`, and the frame reports zero violations. →
   `truncation_reason="rescaler_underflow"`.
3. **Rank-gate divergence, scaling with ‖V‖.** Rescaling contracts the trajectory
   directionally, so the rescaled frames fall below the degeneracy gate at layers
   the original passes. Measured at d = 12 with N(0,1) entries: original scores 5
   transitions, signed scores 2 — an `elim_signed = 0.75` produced entirely by the
   gate. Study A's OV spectral-norm confound (partial ρ to −0.71) is the same
   quantity. → `different_transitions_scored`.

---

## Ready to run

Blocks 1a and 1b are wired and testable end to end:

```
python -m p2b_imaginary.run_2b \
    --weights-dir <phase2 weights> --phase1-dir <phase1 runs> \
    --output-dir <new dir> --base pythia-410m \
    --expect-registry-steps --dry-run
```

`--dry-run` prints the checkpoint list and a cost estimate before committing.
Block 1a alone (`--blocks 1a`) needs no activations and is the cheapest way to
answer open question 1. Then:

```
python -c "from p2b_imaginary import p2b_report as r; \
           r.write_report(r.load_combined('<out dir>'), '<out dir>')"
```

## Open questions this phase can now ask

Ordered by cost. 1–3 are weights-only: no activations, no forward passes.

1. **Does the complex fraction have a developmental trajectory?** If it sits at
   ~0.98 from step 0, "84–97% complex" says nothing about training. If it moves at
   3000–5000 (Phase 1's effective-rank peak) or across 40000–100000 (Phase 2's
   `frac_repulsive` decay), it is the first candidate mechanism for either.
2. **Does Henrici non-normality track the `frac_repulsive` decay?** Phase 2 open
   item 5: attribution goes 1.00 → 0.50 → 0.80 over 90k steps with count flat.
   Henrici measures exactly how much S and A interact. Already computed, never
   plotted against step.
3. **Is the step 8→16 collapse rotational?** Phase 1 open item 4 — unpredicted,
   one interval, confined to layers 21–23.
4. **A null.** A norm-matched Gaussian is ~100% complex.
5. **Signed-only rescaling on Pythia.** Study B's discriminating test.
6. **The real rotation test.** Something not invariant under a global orthogonal
   map: weight-space ablation (`core/intervention.py`, set `W_OV := S` and re-run)
   or readout space (`core/functional_distance.py` — decoded distributions depend
   on the fixed `embed_out`, so rotating the residual stream does change them).
7. **Which frame.** Block 1b runs in `l2_sphere`; the claim is about what
   *attention* applies, so `core/ln_frame.py`'s LN frame is arguably correct and
   is currently unreachable from this phase.

---

## What Phase 2b closes for Phase 2

- **V1** — `n_valid_layers` per frame, serialized.
- **V2** — unclipped rescaled counts. `analysis_p2.py:153`'s
  `max(0, n_phase1 − n_rescaled)` destroys the sign that separates "no effect"
  from "made it worse".
- **Next-experiment 2** — signed-only rescaling on Pythia.

## What Phase 2 must give Phase 2b

- `ov_weights_{stem}.npz` per checkpoint (present — stems are
  `pythia-410m-step{N}`, so no overwrite).
- Parallel-residual attn/FFN streams for item 13. `core/sublayer_streams.py`
  supplies these natively for GPT-NeoX; Pythia's `Δx = attn_out + ffn_out` is
  exact, so this block is more tractable than it was on GPT-2.

---

## Deferred, with reasons

- **Renaming on-disk `phase2i_*` → `phase2b_*`.** Done for new output.
  `p2b_io.refuse_legacy_run_dir` raises on a directory holding the old names
  rather than parsing them, because the counting rule changed underneath them.
- **Block 2.** Its trigger was the withdrawn Block 1b verdict, i.e. a constant.
  Nothing to gate on until item 3 produces a real signal, so item 12 waits.
- **LN-frame Block 1b (item 7 in the question list).** Needs `ln_params` per
  checkpoint, which means touching the extraction path. After item 8.
