# Phase 2b — STATUS

**Naming.** The directory is `p2b_imaginary/`; "Phase 2b" is the name used in
documentation and cross-phase references. The pre-rewrite artifacts are named
`phase2i_*`. INDEX.md left that rename unscoped on the grounds that renaming a
frozen artifact bought nothing — true while the files were frozen, and no longer
true now that they are regenerated. New output is `phase2b_results.json` /
`phase2b_summary.txt`, registered in `core/artifacts.py::PHASE2B`.
`p2b_io.refuse_legacy_run_dir` raises on a directory containing the old names,
because the counting rule changed underneath them.

**Last verified:** 2026-04-29 (GPT-2/ALBERT/BERT study).
**Overall:** **Reopened.** The phase's headline finding is withdrawn — not
falsified by new data, but identified as an algebraic identity that was never
falsifiable. Block 1a's descriptive result stands with a caveat. Blocks 2, 3 and
4 have never produced a number.

---

## Withdrawal: `rotation_neutral` was not a measurement

Block 1b built three rescaled frames from `V = S + A` and compared violation
counts. `elim_rotation = 0.0` in 35/35 runs was reported as the phase's headline.

`A = (V - V^T)/2` is real antisymmetric, so `e^{-A}` is **orthogonal**, and so is
any cumulative product of such matrices. Every quantity Block 1b measures is a
function of the Gram matrix `X X^T`, and `(XR^T)(XR^T)^T = X X^T` exactly for
`RR^T = I`. Energies, effective rank, `ip_mean`, `ip_mass_near_1`: identical to
the unrescaled trajectory. Measured residual over 24 accumulated layers at
d = 1024 is ~1e-15, against a violation threshold of 1e-3 relative.

So `n_rotation_only == n_original` and `elim_rotation == 0.0` were forced by
construction, in every run, on every model, at every β, before any data was read.
The row is now pinned as an identity by
`tests/test_phase2b_rescaled.py::TestRotationFrameIsAnIdentity`, so if it ever
stops holding that is a numerical failure of the control rather than a finding
about rotation.

### What this changes elsewhere

- **status-2.md.** The third candidate explanation for Pythia's inert rescaled
  frame — "Phase 2b established that OV is 84–97% rotational energy but the
  *signed* component carries 100% of causal weight" — loses its evidence and must
  be struck. The other two (numerical truncation, clipped overcorrection) stand
  and are now directly testable here; see "What Phase 2b closes for Phase 2".
- **design-2b.md.** The "Interpretation of the result" section and the
  "why Phase 2c is separate rather than a reopening" argument both rest on the
  withdrawn finding. Phase 2c is out of scope as of 2026-07-18 regardless, but the
  question it was carved out to ask is now answerable inside this phase — see
  "New questions".
- **core/precision_policy.py.** Its docstring states that Phase 2b's causal
  conclusion is "unaffected by how the complex fraction is counted." True about
  the tolerance, and irrelevant: the conclusion fails for an unrelated reason.
  The paragraph needs a correction so a reader does not take it as a second
  endorsement.

### What survives

`elim_full` vs `elim_signed`. `e^{-(S+A)} != e^{-S} e^{-A}` unless S and A
commute, so those two frames genuinely differ. That contrast is exactly what
status-2.md's "next experiments" item 2 asks for and is the phase's remaining
question.

---

## Verdict table

| Test | Result |
|---|---|
| Block 1a — rotational spectrum | OV structurally dominated by complex pairs everywhere: 84–97.5% rotational energy across 7 architectures. **Stands as a description, with two caveats:** (i) it is not known to distinguish trained from random — a norm-matched Gaussian is ~100% complex and no null was run; (ii) `core/precision_policy.py` flags the relative `\|Im λ\| > 0.01·\|Re λ\|` criterion as possibly an fp16-storage artifact (item P2). Also, this phase contains three different definitions of "rotational fraction" and this number uses one of them. |
| Block 1b — causal isolation | **`elim_rotation` withdrawn** (identity, above). `elim_signed = 1.0` in 35/35 is **not yet re-adjudicated**: it was scored with an absolute 1e-6 threshold and a 3.0 rank gate rather than the project's relative 1e-3 and `DEGENERATE_RANK_THRESHOLD`, and per-frame truncation depth was computed and then discarded, so a truncating signed frame is indistinguishable from a working one. |
| Block 2 — hemispheric tracking | Never run. Its gate was the withdrawn Block 1b verdict, so "correctly never triggered" no longer holds; it was gated on a constant. |
| Block 3 — imaginary ablation | Never run (commented out of the runner) and degenerate as written: `build_imaginary_projector` projects onto `col(A)`, and a real antisymmetric matrix in even dimension is generically full rank. Measured at d = 1024: rank 1024, `‖Π − I‖ = 1.6e-15`. The ablation zeroes every activation at every depth threshold. |
| Block 4 — LayerNorm Jacobian | Never run. Raises `NameError` on every prompt (`analyze_layernorm_jacobian` and `layernorm_jacobian_to_json` are used but not imported), caught by the per-prompt handler. Degenerate on three counts besides: `ln_curvature` is identically 1 by algebra (`κ = ‖x−μ‖²/(d·σ²)` with `σ² = ‖x−μ‖²/d`; measured 0.9999995), so the regressor has zero variance and Pearson r is always NaN; `inflation ≤ ~1.02` because the base fraction is ~0.98, so the `> 1.5` classification thresholds are unreachable and `_classify` returns `H2_UNSUPPORTED` unconditionally; and the Jacobian omits Pythia's learned `diag(γ)`. |

---

## Known issues

1. **Counting rule divergence.** Phase 2b scored violations with an absolute
   `-1e-6` threshold and an `eff_rank >= 3.0` gate, in three separate hardcoded
   copies. The project's rule is `core.metrics.ENERGY_VIOLATION_REL_TOL = 1e-3`
   relative with `core.config.DEGENERATE_RANK_THRESHOLD = 2`. So no elimination
   rate this phase produced was comparable to any Phase 1 or Phase 2 number.
   This is status-1's D7 and D8 landing inside Phase 2b. Fixed in
   `p2b_imaginary/p2b_energy.py`, which is now the only place the phase counts a
   violation.
2. **Effective rank was a different statistic.** The local implementation
   normalized *unsquared* singular values; `core.metrics.effective_rank` uses
   squared ones. Same name, different quantity, used as a gate.
3. **Truncation was discarded.** `max_valid_layer` was computed and dropped by
   the serializer — Phase 2's verification item V1, in the phase where it does
   most damage: `e^{-A}` is orthogonal and cannot overflow while `e^{-S}` can, so
   `elim_signed = 1.0` is precisely the value an early-truncating signed frame
   produces for free. Now returned per frame, with `truncated` and
   `truncation_reason`.
4. **Three ways an elimination rate is manufactured**, all now refused rather
   than reported: overflow (`e^{-S}` diverging), underflow (`e^{-S}` contracting
   until rows fall below `l2_normalize`'s 1e-12 floor, after which every energy
   is the constant `1/(2β)` and the frame reports zero violations), and rank-gate
   divergence between frames. The third scales with `‖V‖`, which is Study A's OV
   spectral-norm confound (partial ρ to −0.71) — the regime the models are
   already known to be in.
5. **`elim = 0.0` on a clean run.** `_elim_rate` returned the float 0.0 when
   `n_original == 0`, indistinguishable from "rescaling did nothing", and that
   value then entered a β majority vote. 90 of Study B's 243 Pythia runs are
   `no_violations` and steps 8–64 are clean on all 9 prompts — the phase would
   have returned a verdict by vacuity at exactly the checkpoints where the
   theorem holds. Now `None` with an explicit status.
6. **Substring model matching.** `find_phase2_runs` matched `model_stem in
   d.name`. That produced the `gpt2` aggregation entry recorded here as caveat 2;
   on the Pythia sweep it makes `pythia-410m-step1` match `step16`, `step128`,
   `step1000` and `step128000` — eight of twenty-seven stems collide.
7. **No checkpoint axis, no manifest, no artifact contract, no frame ledger.**
   The runner was organised model × prompt, so 27 checkpoints arrive as 27
   unrelated "models" with no step and no ordering. Addressed in
   `p2b_imaginary/p2b_io.py`.
8. **ALBERT full-rescaling overcorrection** (original caveat 1) is now a
   *measurable* quantity rather than a caveat: `elimination_rate` is unclipped
   and returns negatives.

---

## What Phase 2b closes for Phase 2

- **V2 — unclipped rescaled violation counts.** `analysis_p2.py:153` applies
  `max(0, n_phase1 − n_rescaled)`, which destroys the sign that distinguishes
  "rescaling has no effect" from "rescaling makes it worse".
  `p2b_energy.elimination_rate` does not clip.
- **V1 — `n_valid_layers` per run.** Recorded per frame and serialized.
- **Next-experiment 2 — signed-only rescaling on Pythia.** This is Block 1b's
  remaining question, and it is the discriminating test for Study B's inert
  rescaled frame.

---

## New questions in scope

1. **Does the complex fraction have a developmental trajectory?** Block 1a is
   weights-only — no activations, no forward passes — so running it across all 27
   checkpoints is the cheapest item in the phase. If it sits at ~0.98 from step 0,
   "84–97% complex" says nothing about training. If it moves at 3000–5000 (Phase
   1's effective-rank peak) or across 40000–100000 (Phase 2's `frac_repulsive`
   decay), it is the first candidate mechanism for either.
2. **Does Henrici non-normality track the `frac_repulsive` decay?** Phase 2's open
   item 5: attribution goes 1.00 → 0.50 → 0.80 over 90k steps with violation count
   flat. Henrici is a weights-only per-layer scalar measuring exactly how much S
   and A interact. Already computed; never plotted against step.
3. **Is the step 8→16 collapse rotational?** Phase 1 open item 4 — unpredicted,
   unexplained, one interval, confined to layers 21–23.
4. **A null.** A norm-matched Gaussian is ~100% complex. Without
   `pythia-1.4b-random` and `core/nulls.py`, Block 1a's headline is not known to
   distinguish trained from random.
5. **The real rotation test.** Something not invariant under a global orthogonal
   map of the residual stream. Two are reachable: weight-space ablation through
   `core/intervention.py` (set `W_OV := S`, re-run the forward pass), or
   readout-space measurement through `core/functional_distance.py` — the decoded
   next-token distribution depends on `embed_out`, which is fixed, so rotating the
   residual stream does change it. That is the clean discriminator between
   "rotation is inert" and "rotation is orthogonal to the metric we chose."
6. **Which frame.** Block 1b runs in `l2_sphere`. The claim is about the operator
   *attention* applies, so `core/ln_frame.py`'s LN frame is arguably the right one
   and is currently unreachable from this phase.

---

## Not yet done

Blocks 1a, 2, 3 and 4 are not yet rewritten. Block 1a needs the memory fix
(`build_rotation_plane_projectors` materializes 32 dense (d,d) projectors plus 2
combined per layer, and `analyze_rotational_spectrum` retains `schur_T`/`schur_Z`
per layer — ~7 GB at d = 1024 × 24 layers, ~27 GB at d = 2048), one energy
convention (`rotation_energy_fractions` counts ρ² once per 2×2 block while
`henrici_nonnormality` counts 2ρ², in the same file), and θ on [0, π] rather than
folded to [0, π/2] by `abs(a)`. Blocks 3 and 4 need their math redefined before
their wiring is worth fixing.
