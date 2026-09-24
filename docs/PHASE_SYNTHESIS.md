# Phase synthesis — duplicates, what to run after Phase 10, the e-value plan

**Written:** 2026-09-24, phase review session 9 (`docs/PHASE_REVIEW.md`). **Inputs:**
the 19 cards as of `origin/main` `32b0b9b` (`docs/PHASES.md`), `claims/EXPERIMENTS.md`,
the calibration and dry-run records under `claims/`, and the code paths named below.
Each item points to the card or file that holds its numbers; nothing is restated here
except this doc's own arithmetic (`tools/math_checks/evalue_unit_count.py`).

## 1. Duplicate-question map

The same question asked in more than one phase. "Owner" is where the next unit of
work should land, so that it lands once.

| # | question | asked in | state | owner |
|---|---|---|---|---|
| D1 | Is the attention flip real or causal mask? | 5c; 10 A0 | answered on 410m (~94 % mask), 5c's GPT-2/ALBERT number stands for those models (`archive/p5c_unclustered/status-5c.md` card) | 10, closed |
| D2 | Carrying capacity (how many clusters fit) | 1 (50–55); 10 F14 / Lemma C.1 | 10 has the formula (`p10_cluster_function/math-10.md` §5.4); 1's number is its only data point | 10 F14 |
| D3 | Is an HDBSCAN partition reproducible? | 1d (archived ensemble); 10 §3 (ARI p5 0.347); `CLAIM-C`'s two HDBSCAN metrics; 9 E9 (timescales against the floor) | measured once in 10; `CLAIM-C` never compared against it | 10 (`status-10.md` §5 open item). Rebuild 1d only after (`archive/p1d_cluster_ensemble/`) |
| D4 | Does steering along a geometric direction move the output? | 3 (`archive/p3_crosscoder/steering.py`); 5b (manifold steering); 7 `P-ST1` | only `P-ST1` is registered and gated; 5b needs an ordered-concept positive control first | 7 `P-ST1`; 5b's control is a prerequisite, not a rival |
| D5 | Force clusters to form or dissolve | 5c Group D; 9 E3 (force-collapse, force-disperse); 10 F7 (centroid substitution only) | none run; E3's arms reached no phase (`p9_metric_intervention/status-9.md` card) | one intervention unit, after 10's F4 says what a cluster is (`p10_cluster_function/notes-10.md` §9) |
| D6 | Transport / displacement between layers | 9 E0; 10 F1 | ran once, as F1 (`p10_cluster_function/status-10.md` §1.3) | 10, closed |
| D7 | β: its scale and a per-head producer | 1c (`beta_eff_per_head`, A/B); 2d (T_eff waits on it); 9 §4.6 (γ_β predictions); 5 (effective-β numbers broken) | one undecided convention (factor 8 on 410m) blocks all four; code still offers both (§3.2 below) | 1c, after the user's decision |
| D8 | The rotational channel U_A | 2b (Schur blocks); 6 `P6-R2` (needs U_A); 7 (`U_S`/`U_A` absent, `real_frac` NaN) | no producer; `p7_io.rotational_channel_from_blocks` exists and is unwired (`p6_subspace/status-6.md` card) | one producer from saved OV weights, then 6 and 7 read it |
| D9 | Rank on the normed scale | 1 (D1/D3/D10 re-derive thresholds); 5c (rank plateau, Group C budget); `archive/UPDATE_PLAN.md` (`DEGENERATE_RANK_THRESHOLD`) | blocked on one normed-rank distribution from the sweep | 1, once Stage 0 lands |
| D10 | A matched random baseline per checkpoint | 10 (no norm-matched twin, F12 argued not controlled); 2b (complex fraction vs norm-matched null); 1b (cone collapse `--n-null`, relative classifier cutoff) | three phases, three separate requests, no shared construction | one null in `core/nulls.py`, then each phase calls it |
| D11 | Position and the attention sink as confounds | 5c (position 0 unclustered by construction); 10 (three separate position corrections, none shared) | no shared correction in `core/` (`p10_cluster_function/handoff-10.md` "Standing constraints on all of it") | 10 Stage 1, into `core/` |
| D12 | What clusters are made of by token class / frequency | 5c (regress membership on log frequency); 10 Stage 1 (token-composition table); 7d (`freq` probe) | 10 Stage 1 is next; 5c's regression is the same table with a frequency column | 10 Stage 1, with 5c's regression as one column |
| D13 | The cone margin | 1b (`normalized_margin`, not scale-free); 1c (`hull_min_norm`, `P-H1`); 9 (subspace cone condition, E10 at n > d; §3.50) | 1c's margin is the exact one; 1b has not adopted it | 1c's `hull_min_norm` everywhere |
| D14 | Sublayer streams (post-attention vs post-FFN) | 1c (`h_attn_only`); 1 (late severity: attention vs FFN); 2 (parallel-residual attribution) | 0/152 run dirs have them; one `run_1.py --sublayer` forward pass serves all three | 1 / 1c, one run |
| D15 | A graded readout (KL or λ) instead of raw ΔNLL | 7d (step-1000 pair); 8 (`L2H1`×`L3H6` cell) | both unmeasurable at raw ΔNLL | one readout in `p7d_redundancy/`, two forward passes |
| D16 | Overlap of the induction members' subspaces | 7d (`member_subspace_geometry.py`: principal angles and union rank on *effect* subspaces); 7e (principal angles on top-`r*` *OV weight* subspaces) | different objects, same arithmetic; 7e's is unrun | 7e, reusing 7d's angle and union-rank code |
| D17 | A dictionary on the residual stream | 3 (sparse crosscoder, archived); 4 (dense low-rank autoencoder) | 3's card says use 4's if any question needs one | 4 |
| D18 | `P-I5`'s battery and target | 7 (`P-I5`); 8 (`L3H6` not a matcher); `STATE.md` Blocked 2 | one user decision; the nightly smoke has been red on it since 2026-09-19 | the user |
| D19 | The 410m sweep's 8 v1 prompts, read by many phases | 1b, 1c E, 2d, 4, 5 Study B, 5c, 6 `P6-R4` | each card lists its own "free once Stage 0 lands" item | batch them behind the holdout guard (`docs/PHASE_REVIEW.md` Parked 1) |

## 2. After Phase 10, ranked

Ranked by: unblocks a registered prediction first; then serves several phases (the map
above); then cost, free before forward passes. Everything reading 410m Stage 0 dirs
reads the 8 v1 keys only, through the holdout guard.

| rank | item | serves | cost | blocked on |
|---|---|---|---|---|
| 1 | β's scale convention, then the `beta_eff_per_head` producer (D7) | 1c A/B (`P-gamma1`, `P-gamma2`), 2d (`P-T1`, `P-M1` fresh run), 9 | free, one model load per checkpoint | the user's decision (§3.2) |
| 2 | Extend `CLAIM-C`'s homogeneity calibration to 20 prompts and rescore the v2 arms. Ranked for being the only registered row with a real run waiting; expected outcome is a refusal or no rejection (§3.1) | `CLAIM-C` | free, ~45 min CPU (§3.46) | nothing: no 410m data (§3.3) |
| 3 | `P-S1` at matched k from `activations.npz`, with ≥ 1 600 null draws (§3.1) | `P-S1` | free | nothing |
| 4 | `P-I5`'s battery and target (D18) | `P-I5`, the nightly smoke | free | the user |
| 5 | The U_A producer (D8), then `P6-R2`; `P6-R4` after the unit memo (§3.2) | `P6-R2`, `P6-R4`, 7's tables | free (weights), then `run_7.py` ~16 min × 19 for 7 | Parked 6 memo |
| 6 | Check the 19 Phase 7 tables against `P-I3`'s and `P-AB1`'s pre-computed requirements, without scoring | `P-I3`, `P-AB1` | free | nothing |
| 7 | One matched random baseline per checkpoint in `core/nulls.py` (D10) | 10 F12, 2b, 1b | free (code), then CPU | nothing |
| 8 | The v1-prompt batch on the 410m sweep (D19) | 1b, 1c E, 2d, 4, 5, 5c, 6 | free, CPU hours | Stage 0 chunk 3; the holdout guard |
| 9 | Normed-rank thresholds (D9) | 1, 5c Group C | free | Stage 0 |
| 10 | Weights-only induction items: 7e `schur` vs `svd` on `L11H14`, OV principal angles (D16), `L11H14`'s step-1000 copying score, 8's formation-point equation vs recorded windows | 7d, 7e, 8 | free | nothing |
| 11 | 9's free items: E1 (γ dynamic range), the subspace cone condition as a math check, E9 | 9 | free | E9 needs 10's floor |
| 12 | `run_1.py --sublayer` on 410m (D14) | 1, 1c, 2 | forward pass, 410m | nothing |
| 13 | The graded readout (D15) | 7d, 8 | forward pass, two checkpoints | nothing |
| 14 | A registered invariant on `pythia-1.4b` (8), or `P-I7` | 8, `P-I7` | forward pass, matched grid on 1.4b | the user's rung call (`STATE.md` Blocked 7) |
| 15 | One cluster intervention unit (D5): 5c Group D, 9 E3, 10 F7/F20 | 5c, 9, 10 | forward passes | 10's F4 |
| 16 | Heavy single-phase items: 5b's positive control, 2b's `W_OV := S`, 2d's `P-M1` head ablation, 7e's consolidation surgery, `CLAIM-A` on 1.4b, 9 E6+E8, E10 | one phase each | forward passes | see each card |

Items not listed are single-phase, free, and low-stakes; each card's "After Phase 10"
holds them.

## 3. The e-value plan

### 3.1 Attainable E per adjudicable row

κ = 1/2 and rejection at E ≥ 20 mean **one prediction alone needs p ≤ 1/1600**: a
permutation null needs ≥ 1 599 draws. An exact sign-flip over n units, k of them
informative, has floor (2ⁿ⁻ᵏ + 1)/(2ⁿ + 1) (`p7_motifs/patching_gate.py`), so it needs
**all of 12** informative, or 11 of ≥ 13 (`tools/math_checks/evalue_unit_count.py`).
A claim's E may be the product over its predictions (`core/evalues.py` `combine`) only
when each is calibrated given the others, which fails for predictions scored on one
shared run (`P-T1` and `P-M1` will be); those merge by `average`, which never exceeds
its largest input. So a row below 20 can still count toward its claim, but not
always. The last column says whether a row can decide its claim **alone**. The calibration records'
"clears α" / "sufficient" fields test p ≤ 0.05, which is E ≈ 2.2, not E = 20.

**Only two rows are exchangeable over prompts** (`CLAIM-C`, `P-AB1`), so the 8 / 12 / 20
question matters only for them. The rest are set by heads, layers, draws or vector pairs.

| row | unit that sets the floor | floor and E_max | alone? | record |
|---|---|---|---|---|
| `CLAIM-C` | prompt (sign pattern, homogeneity-corrected). Arms: `pythia-1.4b`, `gpt2-large`; no 410m | 8 prompts: raw 0.0078, corrected 0.0661 → **E 1.94**. 12 (tabulated; the gate's own `homogeneity_correction`, no drops): E 22.6 below homogeneity 0.70, E 20.2 at 0.72, **E 11.5 at 0.725**, E 3.7 at 0.75, **E 1.67 at 0.875**. 20: untabulated | at 12 only if sign homogeneity < 0.725. The 20-prompt run recorded **0.875** and 54/120 concordant cells, so a rescore will most likely refuse or fail to reject | `claims/calibration/claim_c_homogeneity.json`, `claims/audits/claim_c_real_run.json` |
| `P-AB1` | prompt (sign over ablation points; odd counts only) | (2ⁿ⁻ᵏ + 1)/(2ⁿ + 1): 8 of 8 → E 5.67; 12 of 12 → E 22.6; 11 of 12 → E 18.5. The calibrated design has 6 prompts | only with every one of 12 prompts informative, or 11 of ≥ 13 | `claims/calibration/patching_exponent.json` |
| `P-ST1` | matched-norm vector pair | same formula over pairs; 8 informative of 20 → E 8.0 | at ≥ 11 informative pairs out of ≥ 13 | `claims/calibration/steering_sign.json` |
| `P-I1` | head, 2 000 draws | 1/2 001 → **E 22.4** | yes, 12 % margin | `claims/audits/p_i1_real_run.json` |
| `CLAIM-B` | mutual arm: 2 000 pairings; anchor arms: 1/(n_controls + 1) | mutual E 22.4; anchor arms at the 19 controls α needs: **E 2.24** | no: the anchor arms would need ~1 600 control series | `claims/audits/claim_b_p_i1_dry_run.json` |
| `P-T1` | label permutation over heads, 2 000 draws | max(1/C(heads, candidates), 1/2 001). On 410m's 384 heads with ≥ 2 candidates the draws bind: E 22.4 | yes, on a full-model design | `claims/audits/p_t1_p_m1_dry_run.json` |
| `P-M1` | permutation over layers, 2 000 draws | 1/C(layers, violations): 24 layers need 3–21 violation layers (C(24, 3) = 2 024); then E 22.4 | only with 3–21 violation layers | same |
| `P6-R2`, `P6-R4` | unit `model`, 2 000 subspace draws | 1/2 001 → E 22.4 | yes | `claims/audits/p6_r2_r4_dry_run.json` |
| `P-I3` | matched sets, exact enumeration | (1/(M+1))^sets; 4 controls: 4 sets → E 12.5, 5 sets → E 28.0 | at ≥ 5 sets (4 controls) | `claims/calibration/cross_head_association.json` |
| `P-S1` | Monte-Carlo, **default 500 draws** | 1/501 → **E 11.2** | **no, at the default**; 1 600 draws fix it | `p1c_frames/centroids.py` (`adjudicate_p_s1_from_reports`, `n_null=500`) |

What this changes:
- `CLAIM-C` at 8 prompts cannot matter (E ≤ 1.94). At 12 it can decide only below a
  homogeneity it has already been measured well above. So the ~45-min extension to
  20 prompts (§2 rank 2) most likely buys a refusal, not a decision. It is still worth
  running once, since only the extension can show whether that holds at 20.
- `P-S1`'s default is a defect: raise it before any real run (Parked 1 below).

### 3.2 The remaining decisions, checked against the code first (Parked 7)

| decision | code today | still open? | recommendation |
|---|---|---|---|
| β's scale convention (D7) | `core/beta_eff.py` `estimate_beta_from_gram`: `beta_raw` is the fitted slope of the logits on the reader-frame Gram; passing `attn_scale` returns `beta_raw / attn_scale` (×8 on 410m), which its docstring calls the one "comparable across architectures". No producer passes it | **yes**, and **the two sources name the ends oppositely**: `status-1c.md` finding 2 calls the ×8 number the "raw-slope β" that is *not* comparable across `CLAIM-C`'s arms, while the code calls the ×8 number the comparable one | First fix the vocabulary, in `core/beta_eff.py`'s docstring and `status-1c.md` together, by stating which number is the slope the particle dynamics sees (softmax of β⟨xᵢ, xⱼ⟩). Then decide. Lean, not settled: that slope, because it is what γ_β's reduction is a function of, with the head width recorded next to it. Record the choice in the γ_β rows' notes before the producer runs |
| `P-S1`'s matched k | no matched-k re-cluster in `p1c_frames/`; the gate refuses when the arms' (m, d) differ | **yes** | Re-cluster both arms offline at a k fixed before reading them (e.g. the trained arm's HDBSCAN count, chosen per layer from step 0 only), and record the k rule as an amendment before the run. The alternative, "record the gate as unfeedable", leaves `H-RESIST` with no adjudicable row except `CLAIM-C` |
| `P-I5` battery and target (D18) | `p7_motifs/p_i5_ablation.py` iterates the live battery | yes | Pin to v1's keys (the calibration record's battery) *and* re-register the target: `L3H6` fails the statement "an induction head" (`p7_motifs/status-7.md` "Corrections received"). Pinning alone would score a row whose target is known wrong |
| `P6-R4`'s unit `model` on Pythia (Parked 6) | `p6_subspace/r2_r4_null.py` `_draw_rngs`: one seed per draw, a fresh generator per layer; each layer draws `random_orthogonal_subspace_pair(d_model, dim U_neg, dim U_A, rng)` | yes, narrowed | Where two layers' subspace dimensions match, they get the same subspace (the tied reading). Where they differ, the same seed draws a different-shaped matrix, so the subspaces are effectively independent (argued from the draw's shape, not tested). A third case: equal `dim U_neg + dim U_A` split differently may draw the same union and split it differently. Unread in `random_orthogonal_subspace_pair`, and it should be read before the memo is final. On Pythia "model" is therefore at most "same subspace where the shapes agree". That is still conservative, since the recorded table keeps `model` at nominal at ρ = 0. Recommend: keep `model`, amend the notes to say this |

### 3.3 Phase 10's route into the registry

1. **Order.** `CLAIM-C` does not bear on "Open" 3: its arms are `pythia-1.4b` and
   `gpt2-large`, it never reads 410m, and its 20 prompts' contrasts are already
   computed (`claims/audits/claim_c_real_run.json`). Rescoring it spends nothing new on
   410m, so it can go in any order. "Open" 3 is still open for rows that *are* scored
   on 410m v2 runs (none registered today). Correction to this doc's first draft, which
   said otherwise, following `PHASE_REVIEW.md` "Open" 3's example.
2. **What graduates.** F14 (cluster count against Lemma C.1) is the named one, and it
   needs F13 first (`p10_cluster_function/status-10.md` §5.1). A cluster-count test is a
   quantity `CLAIM-C` never computed, so it is blind on the 12 under either reading of
   "Open" 1.
3. **Unit and floor before registering.** Say whether F14's unit is the prompt; if so it
   needs **all 12** prompts informative to decide alone (one uninformative prompt
   leaves E 18.5). That margin is zero, so a per-prompt sign design on the 12 cannot
   decide alone in practice. A permutation design with ≥ 1 600 draws over a finer unit
   can. Put `max_attainable_average_E` on the row's face, as the 10 A0
   runners do.
4. **Axis.** No F-row reads 7d's causal sweep so far, so Parked 8 does not bite yet. The
   check stays at drafting time.

### 3.4 Native e-values for new registrations (a proposal)

Betting e-values (testing by betting; likelihood-ratio e-values) avoid the calibrator's
1/1600 wall and can accrue prompt by prompt. **Not decided and not scanned:** `CLAUDE.md`
literature trigger 2 applies before any row lands. Existing rows keep the calibrator.

## 4. The routing rule, scored against the Superseded lists

`docs/PHASE_REVIEW.md` "Decisions" asked for this once several phases had cards. The
Superseded fields list 74 items. For 44 the pointer is outside the phase's own
directory. 37 of those have a line in `## Corrections received` under the same pointer.
Of the 7 others, 4 are Phase 10 correcting its own plan, 1 is Phase 1b's own `math-1b.md`,
1 (2d's p floors) is routed under a different pointer, and 1 (9's E4) is in 9's card
but not its Corrections section. 9's only reader, 10, does not use E4. **Reading:** the
backfill is complete. It cannot show whether the rule is followed going forward; that
is still Parked 3's check at the first three Stage 1 corrections.

## Parked

1. **`P-S1`'s default of 500 null draws gives E_max 11.2, below the E ≥ 20 wall.**
   Why: a real run at the default could not decide its claim alone. The dry run used 120
   draws and passed only because its test is p ≤ α. Cost: one default in
   `p1c_frames/centroids.py` plus a test pinning `max_attainable_average_E`. Changes: rank
   3 above. Classified as a defect, but not fixed here: this thread is docs-only.
2. **Calibration records say "sufficient" at p ≤ 0.05.** `steering_sign.json`,
   `cross_head_association.json` and the 2d floor tables. Why: a reader could take
   "clears α" as "can decide", and those are 80× apart in p (0.05 vs 1/1600, §3.1). Cost: add an
   `E_max` column when each is next regenerated. Changes: nothing decided.
