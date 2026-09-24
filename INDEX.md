<!-- INDEX.md -->
# Project index

**This file is the structural map: which phase lives in which directory, what is
archived, and what is referenced but absent. It is not the current state of the
work — that is `STATE.md`, which is the file to read first; `PROJECT.md` is the
long-form record behind it.**

## Live phases

| Phase | Directory | State |
|---|---|---|
| 1 | `p1_mstate_tracking/` | Complete |
| 1b | `p1b_hemisphere/` | Complete |
| 1c | `p1c_frames/` | Implemented and validated on synthetic data. **Audited 2026-09-19 (§3.40): E run against Pythia artifacts (`P-H1` measured), A/B/F blocked on inputs no run directory carries** |
| 2 | `p2_eigenspectra/` | Complete; the Pythia rerun is done and the 19-step sweep is on disk |
| 2b | `p2b_imaginary/` | Complete *(directory name is canonical; on-disk artifacts still say "2i")* |
| 2d | `p2d_operator_activation/` | Implemented and validated on constructed operators; **no scoring run** (an August pilot is quarantined, values unopened). **Audited 2026-09-19 (§3.43): inputs all present and the join verified on real artifacts** — blocked on Phase 1c-B by design. (`P-T1`'s wording was already amended, 2026-08-11: `status-2d.md` "Corrections received") |
| 6 | `p6_subspace/` (live) / `archive/p6_subspace/` (frozen) | Ten rows dormant; **`P6-R2` and `P6-R4` active**, their projector path rebuilt live 2026-08-24. **Audited 2026-09-19 (§3.44): `P6-R2` needs `U_A`, which no artifact carries; `P6-R4`'s inputs exist.** The audit said it also lacked an exchangeable unit; it had one, `model`, since 2026-08-25 (`status-6.md` "Corrections received") |
| 7 | `p7_motifs/` | New — mechinterp phenomena as particle motifs. **Audited 2026-09-19 (§3.45): `P-I1`'s run is recorded at last; `P-ST1`/`P-AB1`/`P-I3` built, calibrated, unrun; `P-I5` parked.** See "Phase 7" below |
| 7d | `p7d_redundancy/` | **Active.** The redundancy set — which heads hold the induction regime and when each formed. Q1/Q2/Q3 answered; results in `PROJECT.md` §3.12-S/T/U. Every runner now takes `--model` and `--ablation` |
| 7e | `p7e_consolidation/` | **Active.** Whether the set collapses into one head; `L11H14`. Gate measurement answered — `status-7e.md` |
| 8 | `p8_scale_ladder/` | **Active.** The same measurements across the Pythia ladder. 70m rung run 2026-09-11; `compare_rungs.py` is the cross-rung reader. **Read `literature-8.md` before the next measurement** — the verified scan reframes the phase. See `status-8.md`, `PROJECT.md` §3.15–§3.16 |
| 9 | `p9_metric_intervention/` | **Notes and a plan. PARKED, pre-design, nothing frozen, nothing registered** — intervening on the *metric* the dynamics is read in rather than on the weights. Read `notes-9.md` then `plan-9.md`. **Parked on Phase 10**, whose answer it needs: `plan-9.md` §2–§3 moved there; §2.1a says the functional partition is measurable per layer after all; **§5.1a is RESOLVED (2026-09-20) — the masked system is a *sequential* gradient flow, so the ensemble Hessian framing is void but the per-token curvature claim survives**; **§5.1b is new** — `2601.02932` has built the transfer-operator reduction, and its reversibility-constrained estimator must not be transported to a depth dynamics. §4 onward is unaffected. **Trigger 1 partly discharged via `lit-10.md` §14** — `2605.12765` and `2505.16831` read as primary text; `2303.06562` (ContraNorm) still unread and still blocking any spreading arm |
| 10 | `p10_cluster_function/` | **The current front. Four ladder rows RUN — read `status-10.md` first** (`PROJECT.md` §3.51). `notes-10.md` (the trash-collection hypothesis, the J-lens, the F0–F12 ladder), `attention-10.md` (the flip audited; **A0 has run, A1–A8 unblocked**), **`math-10.md`** (five derivations, four `tools/math_checks/` files, 28 checks — §1 and §2 now **confirmed on real data**; **§7 added 2026-09-20 from the paper itself and has no check file yet**), **`lit-10.md` (THREE scan passes — §11–§15 are five papers read as PRIMARY TEXT on 2026-09-20, including `2411.04990`; §15 is the remaining queue)**. **`handoff-10.md`** (the cluster-function thread's ordered plan) and **`questions-10.md`** (its hypotheses). Runners: `tools/run/p10_attention_baseline.py`, `p10_anchor.py`, `transport.py`, `p10_partition_function.py`, `p10_partition_stability.py`, `backfill_hdbscan.py`. **Trigger 1 discharged for the five papers that mattered** (`lit-10.md` §11–§15); `2303.06562` and `2607.15495` remain. `claims/registry.json` untouched |

Shared code lives in `core/`; one-off scripts in `tools/`; tests in `tests/`.

**Project-wide maps in `docs/`:** `docs/AXES.md` — questions to data (which axes exist, what is
populated, which producers do not exist, and what has never been asked). `docs/LITERATURE.md` —
the per-phase prior-work index, and §0.1's note on which hosts a scan can actually reach.
`docs/readings/` — papers read as primary text, one file per arXiv id, marked **[R]**.
`docs/PHASE_REVIEW.md` — the phase-by-phase review thread (one card per phase, the duplicate map, the e-value plan).
`docs/PHASES.md` — one row per `status-N.md`, generated from the card at the top of each (`docs/phase_card.md`, `tools/render_phases.py`).

`p7d_redundancy/` and `p7_motifs/` are different programmes and are easy to
confuse. `p7_motifs/` is the motif/relay line behind `P-I1`; `p7d_redundancy/`
is the causal-ablation line behind §3.12. In particular
`p7_motifs/formation_curve.py` is a **behavioural** relay curve and
`p7d_redundancy/member_formation_curves.py` is the **causal** ablation curve —
similar names, different instruments. `7a`/`7b`/`7c` remain labels in
`PROJECT.md` §3.14.1 with no directory of their own; only `7d` grew one, when it
went active and started producing results.

Read `status-N.md` for the current state of a phase. Read `design-N.md` for the reasoning
that is not visible from the code. Read **`lit-N.md`** for what has been done before,
where the phase is scooped, and what is left to grow — `docs/LITERATURE.md` indexes all
sixteen. `PREDICTIONS.md` is the project-level falsification record, separate from any
single phase's.

**Which phase carries which e-value: `claims/EXPERIMENTS.md`.** Generated from
`claims/registry.json`, it is the phase → experiment → prediction → gate join that this
index (phases to directories) and `claims/FALSIFICATION.md` (claims to evidence) each hold
half of. It is also where the three phases with a live instrument and no registered
prediction are named — **7d, 7e and 8, which are the active work** — together with the two
adjudicable gates that never had a known-answer dry run (`P-AB1`, `P-I3`) and the one
declared claim nothing feeds (`H-BUDGET`). See `POPPER_PLAN.md` §6zb.

## Archived phases

Moved to `archive/` on 2026-08-22. Reason in every case: the project moved to Pythia
checkpoints and to the "particles first" framing, and this code predates both. Not a
verdict on the work.

Each archived phase also carries a `lit-N.md`: `archive/p3_crosscoder/lit-3.md`,
`archive/p4_mstate_features/lit-4.md`, `archive/p5c_unclustered/lit-5c.md`. Phases 5,
5b and 6 keep their reviews beside their study notes in the live top-level directories
(`p5_single_mstate_analysis/lit-5.md`, `p5b_manifold_steering/lit-5b.md`,
`p6_subspace/lit-6.md`). **A `lit-N.md` is not a reintroduction trigger** — it records
what the field has done since, so that a trigger in `FROZEN.md` is evaluated against
2026-09, not against the date the phase was frozen.

| Phase | Directory | What it found |
|---|---|---|
| 3 | `archive/p3_crosscoder/` | **Null.** Sparse crosscoder decoder directions align with V at chance (0.484 / 0.501), both models. `FROZEN.md` carries the reintroduction trigger |
| 4 | `archive/p4_mstate_features/` | **Not null.** Track 3's dense low-rank AE recovered V-alignment for ALBERT (33 bottleneck directions on V-attractive vs 0 for GPT-2): "sparsity was the confound." `FROZEN.md` |
| 5 | `archive/p5_single_mstate_analysis/` | Complete for 6 models; 6 code-level blockers. Carries the tuned-lens skip-to-output note |
| 5b | `archive/p5b_manifold_steering/` | Built and tested, never run |
| 5c | `archive/p5c_unclustered/` | Docs only, no code. Its attention-flip result is cited in `PREDICTIONS.md` claim (a) |
| 6 | `archive/p6_subspace/` | Partial run, ALBERT only. The LDA-alignment inversion (0.887 imaginary vs 0.067 real repulsive) is unresolved: the labelling explanation is ruled out (`claims/audits/p6_projector_labels.json`), and the comparison is not dimension-normalised (`p6_subspace/math-6.md` §7.2) |

Phase 2c was described in earlier versions of this index and has never existed on disk.

## Referenced, not present

These files are cited by name in live code and docs and **do not exist in this repository**.
Recorded rather than quietly tolerated: the project's own standing rule 4 is "refuse rather
than degrade," and a documentation reference that silently resolves to nothing is the
documentation instance of that bug class. Nothing here has been invented to fill the gap.

| Referenced file | Cited by | Notes |
|---|---|---|
| `MATH.md` | 12 live files — `PREDICTIONS.md`, `archive/UPDATE_PLAN.md`, `core/metrics.py`, `p1_mstate_tracking/{design-1,status-1}.md`, `p1c_frames/*` | The most load-bearing absence. Cited for §3.2's collapse-time table, §8's step-size definition, §9.1/§9.3 — all of which Phase 1c validates against |
| `DESIGN_pythia_frames.md` | 11 live files, all in `core/` — `frames.py`, `rope.py`, `qk_offset_null.py`, `battery_structure.py`, `frame_card.py`, `sink_audit.py`, and others | Cited by item number ("see items 5, 8, 12"), so the numbering is load-bearing too |
| `CHANGES_jlens_adjacent.md` | `p2_eigenspectra/lens_band.py` | |
| A "2026-07-22 addendum" to `PREDICTIONS.md` | `core/qk_offset_null.py:12` | `PREDICTIONS.md` has no such addendum. The one it does carry (P-T1) is undated in the body and describes a different change |
| `POPPER_PLAN.md` §6x | `POPPER_PLAN.md:4030`, `PROJECT.md:1636` | Both forward-reference a section that stops at §6w. `PROJECT.md` calls it "where the [design] is" and `POPPER_PLAN.md` calls it "the proposal for getting [a fresh artifact]"; the audit pass that would otherwise have taken the next letter used §6y instead, so the reference stays open rather than silently resolving to the wrong section |

## Deleted code whose intent is kept

Both branches went in the 2026-09-23 cleanup (`LESSONS.md` lesson 12). The code
is let go; the reason it existed stays (user, 2026-09-24). Until someone deletes
them, the code can still be read from local, unpushed tags.

- **Phase 1d**, clusterer comparison (`010448c`, tag
  `dead/particle-methods-comparison-vpuads`): design, status, findings and its
  never-registered `P-C1`–`P-C4` are in `archive/p1d_cluster_ensemble/`
  (`FROZEN.md` first).
- **Cross-phase visualization CLI**, `tools/visualize_latest.py` (`d1c75ac`,
  tag `dead/visualize-mets-results-sl2ya5`). Why it existed: each phase's
  `visualization` package has its own entry point and input flag and refuses
  another phase's directory, and a `results/` tree mixes bare-timestamp Phase 1
  roots, `p2_eigenspectra_<ts>/` and hand-named pilots. It classified each
  directory by the marker file the phase's own loader discovers (not by name),
  took the newest per phase, and called that phase's entry point; it plotted
  nothing itself. Rebuild against `data/phase12/`'s manifests if wanted.

`archive/docs/deleted-branches-2026-09-10.md` records an earlier cleanup.

## Phase 7 — the mechinterp/particle bridge

New as of 2026-08-22. The goal is to describe mechinterp phenomena — induction heads,
steering, activation patching, prompt injection, SAEs — as statements about what particles
are doing, **without going through natural language**, and then to test whether recurring
structures (motifs) in the interaction graph are what those names actually pick out.

First study: **induction-head formation**, restated as a two-stage `relay` motif and tracked
across the checkpoint axis. It bears directly on `PREDICTIONS.md` claim (b) — that
collapse-resistance emerges at circuit-formation events.

SAEs are an **object of study, never an instrument**: the standing rule from
`core/DESIGN_dual_reading.md` (no SAE/LRAE features in any measurement path) is unchanged.

See `p7_motifs/design-7.md` for the translation table and the motif alphabet,
`p7_motifs/status-7.md` for state, and `PREDICTIONS.md` for the pre-registered predictions.

## Dates

Recovered from run-directory names / report timestamps where present: Phase 1 — 2026-04-23;
Phase 2 — 2026-04-28; Phase 2b, Phase 3 — 2026-04-29; Phase 4 — 2026-05-04; Phase 5 — not
recorded (after 2026-05-04); Phase 5b — never run; Phase 5c — not recorded; Phase 6 —
recorded only as "2026-04-xx", itself a gap. Every future run carries a real timestamp via
the run-manifest infrastructure, so this should stop recurring.
