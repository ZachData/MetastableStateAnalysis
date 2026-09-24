<!-- p9_metric_intervention/status-9.md -->
# Phase 9 — STATUS (parked, pre-design, nothing run)

<!-- phase-card -->
## Card

- **Question:** Can the metric the particles are compared in, LayerNorm's learned diagonal `Γ`, be patched to form, dissolve or move clusters, and does such a patch mean anything before it is known what a cluster does?
- **Inputs:** none. No runner, no run dir, no forward pass under this phase's name. `notes-9.md` (2026-09-18) and `plan-9.md` (2026-09-20) are pre-design; no design file exists. Its E0 was run as Phase 10's F1 (see "Ladder state")
- **Results:**
  - The phase cannot start at the intervention: what a cluster does is unsettled, so tiers 0 and 1 moved to Phase 10 and Phase 9 is parked on it — §3.47, `p9_metric_intervention/plan-9.md` §1
  - A γ-patch is a congruence on the QK bilinear form, shared by every head in the block (derived, no math check; the same holds for the next three items and the write-side clock) — `p9_metric_intervention/plan-9.md` §4.1
  - It is not a read-side lever: Pythia's `input_layernorm` feeds `W_Q`, `W_K` and `W_V`, so a γ-patch moves the pattern and the displacement together, and the sign test needs a `W_V` arm — `p9_metric_intervention/plan-9.md` §4.1, `docs/readings/2411.04990.md`
  - On Pythia the MLP can be excluded exactly: two LayerNorms per block give a three-arm factorial (attention, MLP, both) — `p9_metric_intervention/plan-9.md` §4.2
  - Lemma 6.4's collapse survives every γ (proved from positivity alone, so mask-agnostic) — `p9_metric_intervention/plan-9.md` §4.3
  - The cone margin's first-order response to a rank-1 γ-patch is closed form and reads only the binding set; a patch aimed where no binding token sits does nothing at first order — `p10_cluster_function/math-10.md` §3, `tools/math_checks/cone_margin_gamma_gradient.py`
  - Scaling a branch's write is a subspace-restricted integration time — `p9_metric_intervention/plan-9.md` §4.6
  - Under the causal mask the ensemble Hessian framing is void and a per-token one survives; the transfer-operator readout is preferred — `p9_metric_intervention/plan-9.md` §5.1a, `p9_metric_intervention/plan-9.md` §5.1b
  - The novelty claim narrows to state vs operator and content vs geometry (GUARD-IT holds the genus); any forgetting result needs a relearning arm — `p9_metric_intervention/notes-9.md` §8, `p9_metric_intervention/plan-9.md` §6.1
  - The transport half of `core/dissipation.py` was built and never run; it became Phase 10's F1 — `p9_metric_intervention/notes-9.md` §9
- **Superseded / wrong:**
  - `notes-9.md` §7's three insertion points and its candidate list's "read-side" patch (item 3): γ cannot separate them — `docs/readings/2411.04990.md`, `p9_metric_intervention/plan-9.md` §4.1
  - `plan-9.md` §7 question 7 and §8's E0 call transport "free, unrun": it ran as F1 that day, and the identity coupling is almost always optimal, so displacement is motion of the measure, not shuffling — `p10_cluster_function/status-10.md` §1.3
  - `plan-9.md` §8's E5 lists `P6-R4`'s unit as a pending decision; it was registered `model` on 2026-08-25 — `p6_subspace/status-6.md` "Corrections received"
  - `plan-9.md` §8's E4 calls the 24.9-vs-13.2 ratio "measured": 24.9 is the chance ratio at ALBERT's shape from a planted construction, and the April run's per-layer dimensions were never reported — `claims/audits/p6_projector_labels.json`, `archive/p6_subspace/status-6.md`
  - `plan-9.md` §2.3's reason for the size-profile null: the ARI is already centred, the null is for its variance (amended inline) — `p10_cluster_function/math-10.md` §4
- **Registry:** none, because nothing is designed: no `P-*` id names anything here. `plan-9.md` §9 marks E7 (sign test) and E8 (repair channel) as the only tier-2 candidates and bars reserved rungs (1b, 1.4b)
- **Depends on:** 1c@b60160d2bd, 2@5e9fc59e62, 7d@19b7d835b7, 7e@0c1071db50, 8@8cc3fb223c, 10@b77b1adec9
- **Feeds:** 10
- **Open threads:**
  - β's unit convention (a factor of 8) gates every `gamma_beta` prediction the phase would make; it is a decision, not a computation — `p9_metric_intervention/plan-9.md` §4.6
  - `CLAUDE.md` literature trigger 1 is not discharged: ContraNorm (`2303.06562`) unread, APD/SPD attribution unconfirmed, Sinkformers and metric deformation as an intervention not searched — `p9_metric_intervention/plan-9.md` §12, `p9_metric_intervention/notes-9.md` §11
  - Whether the cone condition has a subspace form under `V ≠ I` at all — `p9_metric_intervention/plan-9.md` §4.5
  - Does self-repair engage against a change of cost geometry rather than an ablation? The field's routes are all ablations, on `[S]` evidence only — `p9_metric_intervention/plan-9.md` §6.2
  - E3's force-collapse and force-disperse arms went nowhere: Phase 10 carries only centroid substitution (F7) — `p10_cluster_function/notes-10.md` §8
  - Which tier-0 answer Phase 10 returns decides what "form a cluster" means here — `p10_cluster_function/notes-10.md` §9
- **After Phase 10:**
  - E1: Pythia's γ dynamic range per layer across training, `frame_table.py` sub-experiment D (free: LN weights of the cached checkpoints)
  - The subspace cone condition as a `tools/math_checks/` script, stating on its face what `n = 4` does not prove (free)
  - E9: implied timescales from `cluster_tracking.py`'s transitions, against Phase 10's reproducibility floor (free)
  - E10: the cone margin at `n > d` and its binding set on `pythia-70m` (forward pass: 70m at 2048 tokens)
  - E6 + E8: the γ-patch with a `W_V` arm and a matched random patch, then patch vs matched ablation for self-repair (forward pass: one LayerNorm on 70m or 410m, swept `D`; needs E1, β's convention and Phase 10's F4)
- **Reviewed:** 2026-09-24 · body `300c78784e`
<!-- /phase-card -->

This file exists so Phase 9 has a card (`docs/PHASE_REVIEW.md` session 7).
The phase's content is `notes-9.md` (read first) and `plan-9.md`; nothing
below restates them.

## Ladder state

`plan-9.md` §8's rows, checked against the tree on 2026-09-24.

| row | state |
|---|---|
| E0 transport | **run as Phase 10's F1** (`tools/run/transport.py`) — `p10_cluster_function/status-10.md` §1.3 |
| E1 γ calibration | not run |
| E2 three-frame agreement | moved to Phase 10 (F4, blocked on the J-lens) |
| E3 cluster-as-function battery | centroid substitution moved to Phase 10 (F7); force-collapse and force-disperse are not carried anywhere |
| E4 Phase 6 note | done: frozen 6's card carries the chance-normalised reading; live 6's "unresolved" agrees (no p-value was computed) |
| E5 `P6-R4`'s unit | **decided** 2026-08-25, `model` |
| E6–E8, E10–E13 | not run; E6 onward need a design |
| E9 implied timescales | not run |

## Corrections received

Corrections to this phase written elsewhere, one line each (`CLAUDE.md` Stop
step 2; `docs/phase_card.md`). Backfilled 2026-09-24.

- 2026-08-25 · E5's unit decision had been taken before `plan-9.md` listed it (found 2026-09-23) · `p6_subspace/status-6.md` "Corrections received"
- 2026-09-20 · a γ-patch is not read-side; `notes-9.md` §7's insertion points are not separable by γ; §4.7 needs a `W_V` arm · `docs/readings/2411.04990.md`
- 2026-09-20 · the Hessian-of-`E_β` framing holds only per token under the causal mask · `p10_cluster_function/notes-10.md` §10.1
- 2026-09-20 · the ARI null is for variance, not centring · `p10_cluster_function/math-10.md` §4
- 2026-09-20 · GUARD-IT occupies the genus; relearning arm required · §3.52
- 2026-09-20 · E0 run as F1: identity coupling optimal · `p10_cluster_function/status-10.md` §1.3
- 2026-09-20 · §4.4's margin derivative computed and checked · `p10_cluster_function/math-10.md` §3
