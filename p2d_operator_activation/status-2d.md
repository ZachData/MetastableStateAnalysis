# Phase 2d — STATUS

<!-- phase-card -->
## Card

- **Question:** Do the heads' operators sit where the paper's dynamics are a gradient flow ($Q^\top K$ symmetric, $V = Q^\top K$) or in one of Table 1's rows, and do the particles' energy-monotonicity violations fall in the layers whose heads leave that regime?
- **Inputs:** designed for Phase 2's `pythia-410m` 19-checkpoint weights (`wq_head*` / `wk_head*` / `ov_head*` in `ov_weights_*.npz`) joined to Phase 1 activations in the LN frame; validated on constructed operators only. One unrecorded pilot, 27 pilot-schedule checkpoints × 2 prompts, LN frame, run 2026-08-14 and rerun 2026-08-17, no manifest, **quarantined, values unopened and never seen** (user, 2026-09-23, confirmed 2026-09-24) — `p2d_operator_activation/status-2d.md` "The 2026-08 Pythia pilot (quarantined)"
- **Results:**
  - D1–D4 recover constructed regimes, rows and bounds, and every join guard fires — `p2d_operator_activation/status-2d.md` "Validation performed"
  - The operator/activation join runs on real artifacts (24 of 25 layers, 16 heads) and both its warnings fire — `p2d_operator_activation/status-2d.md` "E-value audit, Phase 2 / 2d (2026-09-19)"
  - Histogram peak-counting is not a modality test; a mode count must survive a bandwidth scan — `p2d_operator_activation/status-2d.md` "Findings from implementation"
  - A trace contraction wrong everywhere except at symmetric $M$, caught by a non-negativity check; anchors need a non-symmetric arm — `p2d_operator_activation/status-2d.md` "Findings from implementation"
  - Signed OV/QK alignment separates `repulsive_aligned` heads, which confirm the paper, from unstructured ones — `p2d_operator_activation/status-2d.md` "Findings from implementation"
- **Superseded / wrong:**
  - "Not run against Pythia": the pilot ran twice in August — `p2d_operator_activation/status-2d.md` "Corrections received"
  - The runner adjudicated and printed verdicts, and its D3 records would have starved `p_value_p_t1`; now measurement only, with a manifest — `p2d_operator_activation/status-2d.md` "The 2026-08 Pythia pilot (quarantined)"
  - The 2026-09-19 audit called `P-T1`'s wording an open defect; the amendment landed 2026-08-11, before any run — `PREDICTIONS.md` "Addendum — P-T1 amended"
  - `adjudicate_p_t1` reads a single-bandwidth mode count, against the amendment; `p_value_p_t1` implements the amendment — `POPPER_PLAN.md`
  - The reported p floors (0.100, 0.083) were the call's, not the design's, off by 200× and 167× — `claims/EVALUABILITY.md`
- **Registry:** `P-T1`, `P-M1` (`H-OPERATOR`), `active`, e-value gates calibrated on one shared dry run (`claims/audits/p_t1_p_m1_dry_run.json`), not adjudicated. Scored only on a fresh run with a manifest; the August pilot is not a scoring input (user, 2026-09-23)
- **Depends on:** 1@30c0d54cc5, 1c@11c2f3840c, 2@5e9fc59e62
- **Feeds:** none
- **Open threads:**
  - Blocked by design on Phase 1c-B's $T_{\rm eff}$, which waits on a β producer that does not exist — `p2d_operator_activation/design-2d.md`
  - Is the gradient-flow framing void for a causal decoder (2411.04990)? — `p9_metric_intervention/plan-9.md`
  - `simple_tol` / `align_tol` are placed, not derived; centred vs uncentred covariance — `p2d_operator_activation/status-2d.md` "Open before running"
  - `P-M1` refuses when head-to-layer aggregates disagree in sign: per-layer energies may not resolve a per-head claim
  - Which runs the registered gates are scored on (checkpoints, prompts, pooled or per run) — `p2d_operator_activation/status-2d.md` "The 2026-08 Pythia pilot (quarantined)"
- **After Phase 10:**
  - The fresh run on the 410m sweep in the LN frame, v1 prompt keys only, once 1c-B unblocks (free: weights and Phase 1 activations on disk)
  - Tolerance sensitivity scan, reclassifying from the saved per-head records (free)
  - Head ablation for `P-M1` (forward pass: one per head × checkpoint)
- **Reviewed:** 2026-09-24 · body `4b23c8427e`
<!-- /phase-card -->

**Registered predictions:** `P-T1` (e-value — label-permutation null in
`table1_predictions.py`) and `P-M1` (e-value — layer-permutation null in
`gradient_flow_condition.py`), both calibrated on one shared known-answer dry
run and **not run against real artifacts**. Both sit under `H-OPERATOR` and
classify the same head's `Wq`, `Wk`, `W_OV` — one defect moves them together,
which the claim's product would not show. Nulls and evidence paths are
`claims/registry.json`; the per-phase view is `claims/EXPERIMENTS.md`.

**State:** D1, D2, D3, D4 implemented and validated on constructed operators and synthetic
activations. **No scoring run against Pythia**; one unscored pilot ran in August and is
quarantined (below). P-M1 and P-T1 were registered in `PREDICTIONS.md` before this code
existed; `P-T1`'s amendment landed with the code, 2026-08-11 (`cfd7f5b`).

**Driver complete** as of this revision: LN frame resolution (`resolve_ln_params`) and P-M1's
violation counts (`violation_counts`) are wired, so `run_2d.py` runs end to end given Phase 2
weights and a Phase 1 run at a matching revision.

**Blocked on Phase 1c-B by design** — see design-2d.md. The $T_{\rm eff}$ result determines
whether the energy-monotonicity break is the right thing to attribute.

## The 2026-08 Pythia pilot (quarantined)

**Decided by the user, 2026-09-23:** nobody read these values; the files stay unopened and are
**not a scoring input**. `P-T1` / `P-M1` are scored only on a fresh run with a manifest (git
sha, battery hash, frame). The amendment the question below asked about had already landed on
2026-08-11 (see "Corrections received"), so both August runs came after it.

**Two facts found after the decision** (`/challenge-pr` on #78). Put to the user, who
**confirmed the decision on 2026-09-24: "I never looked at the output."** It stands:
- **The August runner printed a verdict.** `run_2d.py` at `6f01f6b` (2026-08-14, the code
  both runs used) prints `P-T1: <verdict>` to stdout for every run (line 369). Whoever
  launched the 54 runs may have seen 54 verdicts, even though nobody opened a file. Fixed
  2026-09-24 (below).
- **Part of the adjudication design came after the pilot.** The amendment's rules (both row-2
  conditions, a control arm, `stable_n_modes` only) date from 2026-08-11. The registry's
  fixed statistic and null (`null_construction`) were recorded 2026-08-23, after both runs.

**The runner, fixed 2026-09-24: measurement only.** Until then `run_2d.py` wrote no manifest,
scored `P-T1` with `adjudicate_p_t1` (a single-bandwidth mode count that contradicts the
amendment) and `P-M1` with `adjudicate_p_m1`, and printed the verdict. It also stored the
bandwidth scan under `modality.stability` and only with `--bw-scan`, but `p_value_p_t1` reads a
top-level `stability`. Fed those records, the gate would have skipped every head and reported
"need both arms: 0 candidates" instead of refusing. Now:

| | |
|---|---|
| scoring | none. No adjudicator or gate is called (an AST test pins it); scoring is a separate call on `p_value_p_t1` / `p_value_p_m1` over the runs the registration names |
| stdout | structural facts only: head count, frame, revision, output path |
| D3 | always runs the bandwidth scan, stored as top-level `stability`; a degenerate projection is stored as `stable_n_modes: None`, so the gate counts it `n_undetermined` |
| manifest | `core.io.write_manifest`: git sha, `git_dirty`, battery hash and prompt key **from the Phase 1 run's manifest**, activation revision, `scored: false`. The Phase 1 manifest is checked **before** any analysis (none or no hash → exit 4, nothing written), and `manifest.json` is written **last**, so it marks a complete run |
| `--bw-scan` | removed (always on) |

Tests: `tests/test_run_2d.py`. **Still open before scoring, both for the user:** (1) which runs
the gates are scored on (checkpoint(s), prompts, pooled or per run) is a registration question,
not settled here. (2) `p_value_p_t1` itself still skips a head with no `stability` and then
returns "need both arms: 0 candidates"; a run made without D3 (the default `--subexp` is D1 D2)
would still produce that. Making the gate refuse instead changes how a registered gate refuses,
not what it computes (`/challenge-pr` on #79). The
fresh run itself is still blocked on 1c-B by design.

**Found 2026-09-23: `run_2d.py` did run on Pythia, on 2026-08-17.** Found by
`docs/PHASE_REVIEW.md` Parked 5; nothing in the repo recorded it. Main tree
`results/p2d_pilot/` (untracked, 60 MB): 54 `p2d.json` files, `pythia-410m` × 27
pilot-schedule checkpoints × 2 prompts (`short_heterogeneous`, `wiki_paragraph`), written
2026-08-17 08:18 → 12:42 **over an earlier run**: the dirs were created 2026-08-14 07:39 →
12:01 and their files overwritten on 08-17 (`stat`; seen by `/challenge-pr` on #77), so the
statistics were computed at least twice. **LN frame** in all 54 (so not the raw-frame sensitivity check),
one warning each (24 operator vs 25 activation layers). No manifest, no `git_sha`, no battery
hash. Each file carries `p_m1` (verdict, per-layer regime lists, correlation aggregates) and
`p_t1` (verdict, trimodal / unimodal / equally-spaced rates on candidates and controls): the
**statistics** the two registered gates score, though not their e-values. It ran after
`P-T1`'s amendment (2026-08-11), not before it as first written here. **The values were not
opened** when this was recorded (only key names and provenance fields were read). Whether this
counted as the peek the design block exists to prevent was put to the user and decided above.

## Corrections received

Corrections to this phase written elsewhere, one line each (`CLAUDE.md` Stop
step 2; `docs/phase_card.md`).

- 2026-08-27 · the reported p floors were the call's (`1/(n_draws+1)`), not the design's, off by 200× (`P-T1`) and 167× (`P-M1`) · `POPPER_PLAN.md` "6p. The last three dry runs: P-S1, P-T1 and P-M1, and a floor that was never the design's (2026-08-27)"
- 2026-08 · `adjudicate_p_t1` reads a single-bandwidth mode count, contradicting the amendment's `stable_n_modes` rule; `p_value_p_t1` follows the amendment · `POPPER_PLAN.md`
- 2026-09-20 · the causally-masked system is not a mean-field gradient flow (2411.04990), which may void this phase's gradient-flow framing · `p9_metric_intervention/plan-9.md`
- 2026-09-23 · "not run against Pythia" was wrong: a pilot ran 2026-08-14 and 2026-08-17 · `docs/PHASE_REVIEW.md`
- 2026-09-23 · the 2026-09-19 audit (below; `PROJECT.md` §3.43) called `P-T1`'s wording an open registration defect. The addendum (no date in its text; "recorded when the Phase 2d code was written, before any run") landed with the code on 2026-08-11 (`cfd7f5b`) and the registry's statement already carries both row-2 conditions, so it was closed before any run · `PREDICTIONS.md` "Addendum — P-T1 amended"

## E-value audit, Phase 2 / 2d (2026-09-19)

Third unit of the audit (`PROJECT.md` §3.43), after Phase 1 and 1c. **Phase 2
registers nothing of its own** — `claims/EXPERIMENTS.md` calls it "a
measurement programme [that] supplies the artifacts other phases adjudicate
on", and that is accurate: its 19-step sweep is the input to 2d and to Phase
7, not a claim. So this unit is `P-T1`, `P-M1`, and `CLAIM-B`, which the
registry places in Phase 1 but whose instrument is this sweep.

**`P-T1` and `P-M1` are the first registered gates in the whole audit whose
inputs are all present.** Checked on disk rather than assumed: **19 of 19**
`p2_eigenspectra_*` directories carry `ov_weights_*.npz` with the per-head
`wq_head*`, `wk_head*` and `ov_head*` arrays that `load_operators` refuses
without — the "legacy behaviour, not recommended" path that omits them was not
taken. The join was then run for real on `step143000`: **24 operator layers
paired against 24 of 25 activation layers, 16 heads, d_head 64**, no
`JoinRefused`. Both of its warnings fired correctly — the 24-vs-25 layer-count
note, and `RAW FRAME`, because LN parameters were not supplied. That second one
is the standing instruction: a primary measurement resolves the frame with
`core.ln_frame.frame_for_hidden_state`, and the raw pairing is a sensitivity
check only.

**So what blocks 2d is not evidence. It is two things already written down.**
1. **The design block.** `design-2d.md` holds the phase behind Phase 1c-B:
   $T_{\rm eff}$ decides whether the energy-monotonicity break is the right
   thing to attribute. 1c-B is itself blocked on the β producer that does not
   exist (§3.40), so the dependency chain is β producer → 1c-B → 2d, and no
   part of it costs a forward pass.
2. **`P-T1`'s registered wording omits half of its own hypothesis** — finding 1
   below, unchanged since it was written: Table 1 row 2 requires
   $\langle Q\varphi_1, K\varphi_1\rangle > 0$ as well as $\lambda_1(V) > 0$
   simple. Running the gate as worded would falsify a claim the paper does not
   make. This is the audit's one live registration defect: the amendment is
   dated and additive, not a silent correction, and it must land **before** the
   gate is run, because afterwards it is indistinguishable from fitting the
   wording to the result.
   **Wrong (2026-09-23):** it had landed, 2026-08-11 — see "Corrections received".

**`CLAIM-B` cannot clear its own floor, and its gate says so before any data.**
`core/changepoint_colocation.py::p_value_claim_b`'s docstring states two
refusals as requirements computed before the pilot ran. Both now checked
against the sweep on disk:

| requirement | on disk | verdict |
|---|---|---|
| anchor arms need **19 control series** at α = 0.05 | the sweep measures **6** metrics | cannot clear the floor |
| registered instrument: a **20–30 checkpoint** cheap-tier sweep | the 410m sweep has **19** checkpoints (0 … 143000) | one short of its own lower bound |

And the grid's third problem is structural rather than countable: on this
schedule a series with **no** located change lands *inside* the 512–2000
window, so a null reads as co-location. That is the same shape as `CLAIM-C`'s
tied rows (§3.41) — a design that cannot come down on one side — arriving by a
different route. `CLAIM-B` has a built, calibrated gate, a `calibration_record`
and no path by which this instrument produces a p-value.

**What this unit did not do.** No statistic was computed for any of the three.
2d's phase block is a design decision recorded by its own author, and running
the gates to see what they say would be exactly the peek that block exists to
prevent; `CLAIM-B`'s refusal is established from the grid and the control
count, which are properties of the instrument, not of the outcome. The
plumbing `CLAIM-C` needed (`tools/score_claim_c.py`) has no counterpart here
and was not written, because a scorer that can only ever return "the floor is
unreachable" is a worse artifact than the two rows above.

## Validation performed

**D1 recovers constructed regimes.** A head built with $M$ symmetric and $V = M$ classifies as
`gradient_flow` (asymmetry 0.000, alignment $+1.000$); the same head with $V = -M$ classifies as
`repulsive_aligned`; a random head reads `outside` with asymmetry 0.693, which is the
$1/\sqrt2$ a generic matrix should give.

**D2's sanity anchor holds.** At $M = I$, $\mathrm{PR}_M$ equals $\mathrm{PR}_C$ to 1e-8. A
rank-1 operator gives $\mathrm{PR}_M = 1.000$ whether aligned with the cloud's top direction or
its tail — correct, and the discriminator between the two is $\mathrm{tr}(MC)$, not the rank.

**D3's row classifier is exact on constructed cases.** $V = +2I \to$ row 1; $V = -2I \to$ row 4;
a real simple $\lambda_1 = 3$ with $M = I \to$ row 2; the *same* $V$ with $M = -I \to$
`row2_eigen_only_qk_fails`; a complex top pair $\to$ `unclassified`. All four adjudicator
branches fire on constructed inputs.

**D4's overflow guard works.** At $\|M\|$ scaled 50×, $E_{\beta=5}$ evaluates to $1.3\times10^7$
rather than `inf`.

**The join guards all fire.** Revision mismatch, unknown revision, $d_{\rm model}$ mismatch and
missing $W_Q/W_K$ arrays each raise rather than proceeding. The raw-frame warning is attached
to every record when LN parameters are absent.

**P-M1's inputs are derived, not stubbed.** `violation_counts` reconstructs the per-boundary
violation indicator from `energies.json` using `ENERGY_VIOLATION_REL_TOL` — the same relative
rule the summary table and `checkpoint_scalars.py` now share, so the three cannot drift. Tested
on a constructed series: a 0.17% drop counts, a 34% drop counts, a flat segment does not.

**Both P-M1 adjudicator branches fire.** On a constructed series where violations sit in the
high-distance layers, all three aggregates give $r = 1.00$ → CONFIRMED. On uncorrelated inputs
they give $-0.25 / -0.09 / -0.54$ → FALSIFIED.

## Findings from implementation

1. **P-T1 as registered omits half of Table 1's row-2 hypothesis.** The row requires
   $\langle Q\varphi_1, K\varphi_1\rangle > 0$ in addition to $\lambda_1(V) > 0$ simple.
   Testing without it would falsify a claim the paper does not make — structurally the same
   error as the retracted "Thm 6.1" verdict row. The prediction's *wording* in `PREDICTIONS.md`
   should be amended with a dated addendum rather than silently corrected, since it was
   pre-registered. The code checks both conditions and labels the difference.

2. **Histogram peak-counting is not a modality test.** The first implementation of
   `projection_modality` scored a plain Gaussian cloud at **nine modes** and a genuinely
   trimodal one at four, at 60 bins on 500 points. Replaced with a KDE at Silverman bandwidth:
   the Gaussian now reads 1, the trimodal reads 3 with spacing ratio 1.000, and both are stable
   across a 4× bandwidth scan.

3. **A modality claim at a single unstated bandwidth is not a measurement.** Any distribution
   can be made unimodal by over-smoothing and multimodal by under-smoothing.
   `modality_stability` scans and reports `stable_n_modes` — the count holding over at least
   three consecutive bandwidths — and `None` when the data does not determine it. P-T1 should
   be adjudicated on the stable count only.

   (Note: `p1c_frames/design_test.inner_product_modes` remains histogram-based, which is
   correct *there* — a sharp configuration's pairwise cosines concentrate at a few exact
   values, so the histogram is nearly a set of deltas and validates exactly against the
   octahedron and icosahedron. The two are different regimes and the difference is deliberate.)

4. **The signed OV/QK cosine separates two regimes the plan treated as one.** Anti-alignment is
   not "far from the condition" — it is the $V = -I_d$ case where the paper itself predicts
   decreasing energy. `repulsive_aligned` heads should be scored as *confirming* the paper,
   not as violating it, and a distance-only score would have put them at the far end of the
   same axis as genuinely unstructured heads.

5. **A wrong trace contraction that the $M = I$ anchor could not catch.** D2's denominator
   $\mathrm{tr}(M^\top CMC)$ was implemented as `sum((C@M) * (C@M.T))`, which contracts to
   $\mathrm{tr}(CMMC)$ — a different quantity that **coincides at $M = I$ and at any symmetric
   $M$**, so the sanity anchor passed while the value was wrong for every real head. It was
   caught only because `coupled_mass` came out negative, which is impossible:
   $\mathrm{tr}(M^\top CMC) = \|C^{1/2}MC^{1/2}\|_F^2 \ge 0$. Measured on a generic $M$, the
   wrong form gave $-72.08$ against a true $+167.00$. The correct contraction is
   `sum((C@M) * (M@C))`; it appeared in three places (D2, `spectral_pairing`, and D4's
   second-order term) and all three were wrong. A non-negativity assertion now runs on every
   call, so the check is permanent rather than a one-off.

   The general lesson, which applies beyond this function: **an anchor that only tests the
   identity case tests almost nothing about a bilinear form.** Every anchor in this phase
   should have a non-symmetric arm.

6. **$\mathrm{PR}_M$'s numerator is a signed trace, so cancellation reads as absence.** A head
   that couples the cloud strongly with mixed signs has $\mathrm{tr}(MC) \approx 0$ and reads
   $\mathrm{PR}_M \approx 0$, identical to a head that couples nothing. A pure rotation
   (antisymmetric $M$) reads exactly 0, since the trace of symmetric × antisymmetric vanishes.
   `coupled_mass` — sign-blind, non-negative — is reported alongside, and the pair distinguishes
   them: low $\mathrm{PR}_M$ with low `coupled_mass` is a head pointed away from the tokens
   (the $\beta$-independence hypothesis); low $\mathrm{PR}_M$ with high `coupled_mass` is a
   rotation.

7. **A violation "count per layer" is a category error.** A violation is an event between two
   adjacent layers, so there is exactly one per boundary and the series is an indicator, not a
   count. Correlating a per-layer regime score against it is correlating against a boolean, and
   `violation_counts` returns it as one rather than letting a "count" name imply otherwise.
   Layer 0 is zero by construction (no preceding layer), which biases the correlation slightly
   toward zero; it is reported rather than dropped, since dropping it would misalign the regime
   series.

## Open before running

1. **Which activations feed D2/D3/D4.** They must be the LN'd states attention actually reads,
   in the right frame — `core/ln_frame.frame_for_hidden_state` resolves the off-by-one, and it
   must be used rather than re-derived. D2 and D4 on raw residual-stream activations would be
   measuring a different operator's action.
2. **Centred vs uncentred token covariance in D2.** Both are meaningful and they answer
   different questions; the uncentred $C$ is dominated by the common mode on an anisotropic
   cloud, so $\mathrm{PR}_M$ can read $\approx 1$ purely because every token shares a
   direction. Run both wherever $\kappa_1$ is large.
3. **The `simple_tol` and `align_tol` constants are placed, not derived.** Eigenvalues of a
   $d{=}1024$ non-normal OV circuit come in near-degenerate clusters, so "simple" needs a
   tolerance, and the classification counts move with it. Both are returned in every record so
   reclassification needs no recomputation — do the sensitivity scan before quoting any rate.
4. **fp32 is mandatory.** The row classification turns on the sign and multiplicity of
   $\lambda_1(V)$ near zero, which is exactly what `core/models.py`'s precision guard exists to
   protect (and which that guard's docstring, corrected this cycle, now names correctly).
5. **The extraction convention must be passed, not guessed.** `resolve_ln_params` exposes
   `--keep-embedding` and `--last-is-post-final-ln` because the off-by-one between hidden-state
   index and reading block depends on how the activations were extracted, and
   `core/ln_frame.resolve_frame_index` cannot infer it. The defaults follow this project's Fix 4
   convention (index 0 = block-0 output) and assume final LN was *not* pre-applied. If the
   extraction did apply it, the last state's correct frame is the identity and applying final LN
   again is wrong — the driver prints which indices resolved to identity so this is checkable
   against the extraction path rather than assumed.
6. **`--pm1-beta` is not swept, deliberately.** P-M1 is a claim about *where* violations sit, and
   different $\beta$ produce different violation sets; pooling them would mix the sets and the
   correlation would be against a union that corresponds to no single energy. Report per $\beta$
   by re-running.
