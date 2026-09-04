<!-- INDEX.md -->
# Project index

**This file is the structural map: which phase lives in which directory, what is
archived, and what is referenced but absent. It is not the current state of the
work — that is `PROJECT.md`, which is the file to read first.**

## Current priority (updated 2026-09-03)

- **Active work: Phase 7** — the mechinterp/particle bridge, currently `P-I1`
  (induction-head formation as a two-stage `relay` motif). See `PROJECT.md` for
  where it stands and what is blocking, and `p7_motifs/design-7.md` for the
  translation table.
- **Phase 2's Pythia rerun is complete.** The 19-step registered sweep is on
  disk; `PROJECT.md` §1 has the layout.
- **Everything after Phase 2 has been archived.** Phases 3, 4, 5, 5b, 5c and 6 moved to
  `archive/` on 2026-08-22. They are not maintained, not imported by anything live, and
  their tests are not collected. Their findings stand and stay citable — see
  `archive/README.md`, which states the policy once so it does not have to be re-derived.
- **Checkpoint schedule: anchors only, pilot sweep not run.** Checkpoints at known event
  locations (the plan's provisional-anchor steps) are in use directly, not the dense item-8
  pilot sweep. Whether item 8 happens as its own pass is still undecided; not blocking.
- **SLT anchor and LLC: not current work.** The plan's "cheap SLT anchor" (per-checkpoint
  training-loss / weight-norm logging) is not implemented and is not being added. LLC stays
  dropped, with the same conditional revisit (good checkpoint data *and* an actual need).
  Flagged where it would otherwise get built: `p1_visualization/checkpoints.py` and
  `p2_eigenspectra/status-2.md`.

## Live phases

| Phase | Directory | State |
|---|---|---|
| 1 | `p1_mstate_tracking/` | Complete |
| 1b | `p1b_hemisphere/` | Complete |
| 1c | `p1c_frames/` | Implemented and validated on synthetic data; **not yet run against Pythia artifacts** |
| 2 | `p2_eigenspectra/` | Complete; the Pythia rerun is done and the 19-step sweep is on disk |
| 2b | `p2b_imaginary/` | Complete *(directory name is canonical; on-disk artifacts still say "2i")* |
| 2d | `p2d_operator_activation/` | Implemented and validated on constructed operators; **not run**. Blocked on Phase 1c-B by design |
| 7 | `p7_motifs/` | New — mechinterp phenomena as particle motifs. See "Phase 7" below |

Shared code lives in `core/`; one-off scripts in `tools/`; tests in `tests/`.

Read `status-N.md` for the current state of a phase. Read `design-N.md` for the reasoning
that is not visible from the code. `PREDICTIONS.md` is the project-level falsification
record, separate from any single phase's.

## Archived phases

Moved to `archive/` on 2026-08-22. Reason in every case: the project moved to Pythia
checkpoints and to the "particles first" framing, and this code predates both. Not a
verdict on the work.

| Phase | Directory | What it found |
|---|---|---|
| 3 | `archive/p3_crosscoder/` | **Null.** Sparse crosscoder decoder directions align with V at chance (0.484 / 0.501), both models. `FROZEN.md` carries the reintroduction trigger |
| 4 | `archive/p4_mstate_features/` | **Not null.** Track 3's dense low-rank AE recovered V-alignment for ALBERT (33 bottleneck directions on V-attractive vs 0 for GPT-2): "sparsity was the confound." `FROZEN.md` |
| 5 | `archive/p5_single_mstate_analysis/` | Complete for 6 models; 6 code-level blockers. Carries the tuned-lens skip-to-output note |
| 5b | `archive/p5b_manifold_steering/` | Built and tested, never run |
| 5c | `archive/p5c_unclustered/` | Docs only, no code. Its attention-flip result is cited in `PREDICTIONS.md` claim (a) |
| 6 | `archive/p6_subspace/` | Partial run, ALBERT only. The LDA-alignment inversion (0.887 imaginary vs 0.067 real repulsive) is unresolved, two live explanations |

Phase 2c was described in earlier versions of this index and has never existed on disk.

## Referenced, not present

These files are cited by name in live code and docs and **do not exist in this repository**.
Recorded rather than quietly tolerated: the project's own standing rule 4 is "refuse rather
than degrade," and a documentation reference that silently resolves to nothing is the
documentation instance of that bug class. Nothing here has been invented to fill the gap.

| Referenced file | Cited by | Notes |
|---|---|---|
| `MATH.md` | 12 live files — `PREDICTIONS.md`, `UPDATE_PLAN.md`, `core/metrics.py`, `p1_mstate_tracking/{design-1,status-1}.md`, `p1c_frames/*` | The most load-bearing absence. Cited for §3.2's collapse-time table, §8's step-size definition, §9.1/§9.3 — all of which Phase 1c validates against |
| `DESIGN_pythia_frames.md` | 11 live files, all in `core/` — `frames.py`, `rope.py`, `qk_offset_null.py`, `battery_structure.py`, `frame_card.py`, `sink_audit.py`, and others | Cited by item number ("see items 5, 8, 12"), so the numbering is load-bearing too |
| `CHANGES_jlens_adjacent.md` | `p2_eigenspectra/lens_band.py` | |
| A "2026-07-22 addendum" to `PREDICTIONS.md` | `core/qk_offset_null.py:12` | `PREDICTIONS.md` has no such addendum. The one it does carry (P-T1) is undated in the body and describes a different change |

## In flight on other branches

Not on `main`, and not reflected in the tables above:

- **Phase 1d** — clusterer comparison, `origin/claude/particle-methods-comparison-vpuads`.
- **Cross-phase visualization CLI** — `origin/claude/visualize-mets-results-sl2ya5`.

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
