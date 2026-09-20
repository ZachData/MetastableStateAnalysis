# EXPERIMENTS.md — which phase carries which e-value

**Generated** by `tools/render_experiments.py` from `claims/registry.json`,
`claims/audits/`, `claims/calibration/` and `claims/adjudications/`. Do not edit
by hand; `--check` fails CI when this file disagrees with what it summarises.

`INDEX.md` maps phases to directories and `claims/FALSIFICATION.md` maps claims to
evidence. This file is the join between them: phase → experiment → prediction →
gate → what that gate is worth. It is the file to read when the question is
*"does this phase carry a falsifier, and has it been run?"*

α = 0.05 · κ = 0.5 · relevance floor r0 = 0.6 · a claim is supported at
**E ≥ 1/α = 20**.

## Where the evidence stands

- **39** registered predictions across **6** phases and **5** claims.
- **11** can carry an e-value right now (`e-value`, active, relevance ≥ r0).
- **19** are dormant — pre-registered, falsifier intact, instrument archived.
- **0 have been adjudicated.** Every E below is 1 and every decision is "not adjudicated". The apparatus is built; no p-value against a real artifact exists yet.

## Per phase

| phase | registered | e-value | adjudicable now | adjudicated | claims touched |
|---|---|---|---|---|---|
| `1` | 3 | 2 | 2 | 0 | `H-EMERGE`, `H-RESIST`, `H-TRANSFER` |
| `6` | 12 | 4 | 2 | 0 | `H-OPERATOR` |
| `7` | 9 | 4 | 4 | 0 | `H-BRIDGE` |
| `1c` | 4 | 1 | 1 | 0 | `H-RESIST` |
| `2d` | 2 | 2 | 2 | 0 | `H-OPERATOR` |
| `5b` | 9 | 2 | 0 | 0 | `H-BRIDGE` |

## The map

One section per phase, one row per registered prediction. `gate` is the
`module:function` that computes the p-value; an empty gate means no null is
built, which is what `needs-null` and `measurement` mean in code. The null
column is a pointer, not the construction — `claims/registry.json`'s
`null_construction` holds the whole of it, and `POPPER_PLAN.md` §6a–6zb holds
what each one cost to build and what it was wrong about first. `calibrated`
and `real run` are the registry's `calibration_record` and `real_run_record`:
each a git-tracked path or nothing, checked by `tools/check_registry.py`, so
an empty cell means no such evidence exists in the tree. A row can carry a
gate, a calibration and a real run and still read `needs-null` — that is a
null that was tried and found invalid, kept visible rather than dropped.

### Phase 1 — `p1_mstate_tracking/`

| experiment | prediction | claim | evaluable | status | gate | calibrated | real run | null (first line) |
|---|---|---|---|---|---|---|---|---|
| item-6 replication gate | `CLAIM-A` | `H-RESIST` | needs-null | active | — | — | — | Needs a null over the three pass criteria jointly. |
| item-6 replication gate | `CLAIM-C` | `H-TRANSFER` | e-value | active | `p1_mstate_tracking.replication_gate:p_value_claim_c` | `claims/audits/claim_c_dry_run.json` | `claims/audits/claim_c_real_run.json` | Sign-concordance of the trained-minus-random CONTRAST, built 2026-08-24 in p1_mstate_tracking/replication_gate.py and fixed BEFORE any gate data exis… |
| item-8 checkpoint pilot | `CLAIM-B` | `H-EMERGE` | e-value | active | `core.changepoint_colocation:p_value_claim_b` | `claims/audits/claim_b_p_i1_dry_run.json` | — | Changepoint co-location on the log-step axis, built 2026-08-24 in core/changepoint_colocation.py and fixed BEFORE any sweep data exists. |

### Phase 6 — `p6_subspace/`

| experiment | prediction | claim | evaluable | status | gate | calibrated | real run | null (first line) |
|---|---|---|---|---|---|---|---|---|
| — | `P6-A2` | `H-OPERATOR` | needs-null | dormant | — | — | — | Classification agreement between f_rot and head type; needs a permutation null over the head-type labels. |
| — | `P6-C1` | `H-OPERATOR` | needs-null | dormant | — | — | — | Alignment of write subspace with matching channel; needs a random-subspace null of matched dimension. |
| — | `P6-D5` | `H-OPERATOR` | needs-null | dormant | — | — | — | Monotonicity of d_S approaching merge vs d_A; a trend test with a permutation null over the approach window. |
| — | `P6-DD1` | `H-OPERATOR` | needs-null | dormant | — | — | — | Two thresholds (induction drop, ARI floor) on an intervention. |
| — | `P6-DD2` | `H-OPERATOR` | needs-null | dormant | — | — | — | Symmetric counterpart of P6-DD1, same construction. |
| — | `P6-I1` | `H-OPERATOR` | e-value | dormant | — | — | — | Already a Mann-Whitney U on f_rot(induction heads) vs f_rot(semantic heads). |
| — | `P6-I2` | `H-OPERATOR` | e-value | dormant | — | — | — | Two-sample test over head pairs; same shape as P6-I1. |
| — | `P6-R1` | `H-OPERATOR` | needs-null | dormant | — | — | — | Threshold on a ratio (R >= 5) with a random-projection reference already named. |
| — | `P6-R2` | `H-OPERATOR` | e-value | active | `p6_subspace.r2_r4_null:p_value_p6_r2` | `claims/audits/p6_r2_r4_dry_run.json` | — | Matched-dimension random-subspace null, built 2026-08-24 in p6_subspace/r2_r4_null.py and fixed BEFORE any p-value exists. |
| — | `P6-R3` | `H-OPERATOR` | needs-null | dormant | — | — | — | Directional dominance at merge events; permutation over merge vs non-merge steps. |
| — | `P6-R4` | `H-OPERATOR` | e-value | active | `p6_subspace.r2_r4_null:p_value_p6_r4` | `claims/audits/p6_r2_r4_dry_run.json` | — | Matched-dimension random-subspace null, built 2026-08-24 in p6_subspace/r2_r4_null.py alongside P6-R2 and fixed BEFORE any p-value exists. |
| — | `P6-R5` | `H-OPERATOR` | needs-null | dormant | — | — | — | Counts of contracting/rotating steps against a null of no directional preference; a binomial test once the per-step unit is fixed. |

### Phase 7 — `p7_motifs/`

| experiment | prediction | claim | evaluable | status | gate | calibrated | real run | null (first line) |
|---|---|---|---|---|---|---|---|---|
| 7-A | `P-I1` | `H-BRIDGE` | e-value | active | `p7_motifs.formation_gate:p_value_p_i1` | `claims/audits/claim_b_p_i1_dry_run.json` | `claims/audits/p_i1_real_run.json` | Changepoint co-location on the log-step axis, built 2026-08-24. |
| 7-A/7-D | `P-I5` | `H-BRIDGE` | needs-null | active | `p7_motifs.p_i5_gate:intersection_union_pvalue` | `claims/calibration/p_i5_joint_null.json` | `claims/calibration/p_i5_real_ablation.json` | Permutation null over the matched-magnitude random-direction ablation arm, on a two-dimensional statistic (geometric delta, logit delta). |
| 7-B | `P-I2` | `H-BRIDGE` | needs-null | active | — | — | — | Two-sample comparison of channel mass between edge types, against the N1/N2 nulls motif_stats.py already gates on. |
| 7-C | `P-I3` | `H-BRIDGE` | e-value | active | `p7_motifs.cross_head_gate:p_value_p_i3` | `claims/calibration/cross_head_association.json` | — | Correlation with a REQUIRED control arm over non-induction heads; motif_stats.py makes independence_source a positional argument so the arm cannot be… |
| 7-D | `P-I4` | `H-BRIDGE` | needs-null | active | — | — | — | Matched-magnitude control on moved_fraction; permutation over which edges are labelled motif edges. |
| 7-SAE | `P-SA1` | `H-BRIDGE` | needs-null | active | — | — | — | Random-subspace null of matched dimension, comparing the observed mass fraction in U_neg against dictionaries of the same rank drawn isotropically. |
| 7-patching | `P-AB1` | `H-BRIDGE` | e-value | active | `p7_motifs.patching_gate:p_value_p_ab1` | `claims/calibration/patching_exponent.json` | — | Paired growth-exponent comparison against a MATCHED RANDOM-DIRECTION ablation of EQUAL MAGNITUDE at the same layer -- the same control design design-… |
| 7-steering | `P-ST1` | `H-BRIDGE` | e-value | active | `p7_motifs.steering_gate:p_value_p_st1` | `claims/audits/p_st1_dry_run.json` | — | Sign contrast of the effective-rank change between two matched-norm steering arms, built 2026-08-25 in p7_motifs/steering_gate.py and fixed BEFORE an… |
| qk-symmetry-sweep | `P-I7` | `H-BRIDGE` | needs-null | active | — | — | — | DESIGN FIXED HERE; THE NULL ITSELF IS TO BE BUILT AND CALIBRATED IN THE CHUNK, which is why this registers as needs-null rather than claiming an e-va… |

### Phase 1c — `p1c_frames/`

| experiment | prediction | claim | evaluable | status | gate | calibrated | real run | null (first line) |
|---|---|---|---|---|---|---|---|---|
| 1c-A | `P-gamma2` | `H-RESIST` | needs-null | active | — | — | — | Point estimate against a constant. |
| 1c-B | `P-gamma1` | `H-RESIST` | needs-null | active | — | — | — | beta_reduction.py reports a residual BRACKET across beta in [0.5,5] rather than a point estimate. |
| 1c-E | `P-H1` | `H-RESIST` | measurement | active | — | — | — | NONE, and deliberately so. |
| 1c-F | `P-S1` | `H-RESIST` | e-value | active | `p1c_frames.centroids:p_value_p_s1` | `claims/audits/p_s1_dry_run.json` | — | Monte-Carlo permutation against a matched i.i.d. baseline at the trained configuration's own (m, d). |

### Phase 2d — `p2d_operator_activation/`

| experiment | prediction | claim | evaluable | status | gate | calibrated | real run | null (first line) |
|---|---|---|---|---|---|---|---|---|
| 2d-D1 | `P-M1` | `H-OPERATOR` | e-value | active | `p2d_operator_activation.gradient_flow_condition:p_value_p_m1` | `claims/audits/p_t1_p_m1_dry_run.json` | — | Permutation test over layers. |
| 2d-D3 | `P-T1` | `H-OPERATOR` | e-value | active | `p2d_operator_activation.table1_predictions:p_value_p_t1` | `claims/audits/p_t1_p_m1_dry_run.json` | — | Label-permutation test over the row-2 classification. |

### Phase 5b — `p5b_manifold_steering/`

| experiment | prediction | claim | evaluable | status | gate | calibrated | real run | null (first line) |
|---|---|---|---|---|---|---|---|---|
| — | `P5b-A1` | `H-BRIDGE` | needs-null | dormant | — | — | — | Variance-retention threshold. |
| — | `P5b-A2` | `H-BRIDGE` | needs-null | dormant | — | — | — | Ratio against a placed threshold; needs a matched-random residual distribution. |
| — | `P5b-B1` | `H-BRIDGE` | e-value | dormant | — | — | — | A difference of two dependent correlations on the same pairs. |
| — | `P5b-B2` | `H-BRIDGE` | measurement | dormant | — | — | — | A threshold against a value calibrated from Wurgaft's reported numbers (p5b_distances.py:31), not from a null distribution of our own. |
| — | `P5b-B3` | `H-BRIDGE` | needs-null | dormant | — | — | — | Effect-size floor on the same dependent-correlation difference as B1. |
| — | `P5b-C1` | `H-BRIDGE` | e-value | dormant | — | — | — | Already a two-sample comparison with a stated alpha. |
| — | `P5b-C3` | `H-BRIDGE` | needs-null | dormant | — | — | — | Two-sample comparison over layers; permutation over the merge/plateau labelling. |
| — | `P5b-D1` | `H-BRIDGE` | needs-null | dormant | — | — | — | A three-way ordering of dependent correlations. |
| — | `P5b-D2` | `H-BRIDGE` | measurement | dormant | — | — | — | An EQUIVALENCE claim, not a difference. |

## The adjudicable rows, and what stands behind each

Only these can move a claim's E. `dry run` is a run on inputs whose correct
verdict is fixed a priori (`claims/audits/`); `calibration` is the measured
behaviour of the construction (`claims/calibration/`).

| prediction | phase · experiment | claim | dry run | calibration | p | e | decision |
|---|---|---|---|---|---|---|---|
| `P-S1` | `1c` · 1c-F | `H-RESIST` | p_s1_dry_run.json | — | — | — | not adjudicated |
| `CLAIM-C` | `1` · item-6 replication gate | `H-TRANSFER` | claim_c_dry_run.json | claim_c_homogeneity.json | — | — | not adjudicated |
| `CLAIM-B` | `1` · item-8 checkpoint pilot | `H-EMERGE` | claim_b_p_i1_dry_run.json | changepoint_colocation.json, claim_b_grid_feasibility.json | — | — | not adjudicated |
| `P-T1` | `2d` · 2d-D3 | `H-OPERATOR` | p_t1_p_m1_dry_run.json | — | — | — | not adjudicated |
| `P-M1` | `2d` · 2d-D1 | `H-OPERATOR` | p_t1_p_m1_dry_run.json | — | — | — | not adjudicated |
| `P6-R2` | `6` · — | `H-OPERATOR` | p6_r2_r4_dry_run.json | — | — | — | not adjudicated |
| `P6-R4` | `6` · — | `H-OPERATOR` | p6_r2_r4_dry_run.json | — | — | — | not adjudicated |
| `P-ST1` | `7` · 7-steering | `H-BRIDGE` | p_st1_dry_run.json | steering_sign.json | — | — | not adjudicated |
| `P-AB1` | `7` · 7-patching | `H-BRIDGE` | **none** | patching_exponent.json | — | — | not adjudicated |
| `P-I1` | `7` · 7-A | `H-BRIDGE` | claim_b_p_i1_dry_run.json | — | — | — | not adjudicated |
| `P-I3` | `7` · 7-C | `H-BRIDGE` | **none** | cross_head_association.json | — | — | not adjudicated |

## Per claim

| claim | phases feeding it | registered | adjudicable now | E | decision |
|---|---|---|---|---|---|
| `H-BRIDGE` | `7`, `5b` | 18 | 4 | 1 | not adjudicated |
| `H-BUDGET` | **none** | 0 | 0 | 1 | not adjudicated |
| `H-EMERGE` | `1` | 1 | 1 | 1 | not adjudicated |
| `H-OPERATOR` | `6`, `2d` | 14 | 4 | 1 | not adjudicated |
| `H-RESIST` | `1`, `1c` | 5 | 1 | 1 | not adjudicated |
| `H-TRANSFER` | `1` | 1 | 1 | 1 | not adjudicated |

## Gaps

Three joins that no other tool here checks, each stated rather than left to be
noticed. None of them is automatically a fault; all of them are things a reader
would otherwise have to reconstruct from five files.

### Phases on disk with no registered prediction

| phase | directory | why |
|---|---|---|
| `2` | `p2_eigenspectra/` | eigenspectra; the 19-step Pythia sweep is a measurement programme and supplies the artifacts other phases adjudicate on. |
| `3` | `archive/p3_crosscoder/` | archived 2026-08-22, null result. `archive/p3_crosscoder/FROZEN.md`. |
| `4` | `archive/p4_mstate_features/` | archived 2026-08-22. `archive/p4_mstate_features/FROZEN.md`. |
| `5` | `p5_single_mstate_analysis/` | archived 2026-08-22; six code-level blockers, no falsifier registered. |
| `8` | `p8_scale_ladder/` | UNEXPLAINED — active phase, see the section below. |
| `10` | `p10_cluster_function/` | pre-design and deliberately unregistered. `notes-10.md` §13 states it: no construction frozen, no `P-*` id, `claims/registry.json` untouched. Its free rows (F0, F1, F11-A0, F12) have RUN, and their records under `data/analysis/` are tier 1, exploratory, and not quotable as adjudications. |
| `1b` | `p1b_hemisphere/` | hemisphere geometry; its findings feed P-H1 and CLAIM-A rather than carrying a falsifier of their own. |
| `2b` | `p2b_imaginary/` | imaginary/rotational decomposition; measurement, feeding H-OPERATOR's P-M1 through `rotational_schur.py`. |
| `5c` | `archive/p5c_unclustered/` | docs only, no code. Cited by `PREDICTIONS.md` claim (a). |
| `7d` | `p7d_redundancy/` | UNEXPLAINED — active phase, see the section below. |
| `7e` | `p7e_consolidation/` | UNEXPLAINED — active phase, see the section below. |

A phase outside the registry is outside the apparatus: `core/adjudication.py`
cannot refuse what was never registered, and a headline from such a phase
carries no Type-I guarantee whatever its control distribution looks like.

### Declared claims with no prediction

- **`H-BUDGET`** is a heading in `claims/CLAIMS.md` and no registered
  prediction names it, so its e-process has no factor that could ever
  enter and its E is 1 by construction rather than by result.

### Adjudicable gates with no known-answer dry run

- **`P-AB1`** (`7` · 7-patching, gate `p7_motifs.patching_gate:p_value_p_ab1`) — patching_exponent.json.
- **`P-I3`** (`7` · 7-C, gate `p7_motifs.cross_head_gate:p_value_p_i3`) — cross_head_association.json.

`POPPER_PLAN.md` §6p records the base rate on the nine rows that had one:
*"Nine for nine, every one of them changed something. … not one converted row
survived being run on an input whose answer was already known, and no test was
failing on any of them."* A calibration measures the construction's behaviour
on a synthetic family; a dry run asks the different question of whether the
gate returns the verdict that is correct a priori. The rows above have the
first and not the second.

### Gates that do not resolve

None — every `gate` names a module that exists and a function it defines.

