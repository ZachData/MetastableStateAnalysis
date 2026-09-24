# Phase 5b — STATUS

<!-- phase-card -->
## Card

- **Question:** Are the metastable clusters' centroids the same objects as the concept manifolds a steering paper found with labelled data: does the centroid path fit a manifold isometric to the output distributions, and do merge events jump along it?
- **Inputs:** none; never run
- **Results:**
  - Four sub-experiments (manifold fit, isometry, merge teleportation, subspace isometry) built and tested, none run — `archive/p5b_manifold_steering/status-5b.md` "Verdict table"
  - The source paper exists, and its isometry holds on ordered, low-cardinality concepts (weekdays, months) — `p5b_manifold_steering/lit-5b.md` §1
- **Superseded / wrong:**
  - The main arm must not run before an ordered-concept positive control: unordered HDBSCAN centroids are a substitution the paper never tested, so a null would be uninterpretable — `PROJECT.md` "One design change this review proposes outright"
  - Blocker 1 (no logits): an LM-head loader at the pinned revision now exists — `archive/p5_single_mstate_analysis/status-5.md` "v2 follow-up: causal_tests.py migration — DONE (item 3 aftermath, closed)"
- **Registry:** nine `P5b-*` rows, all `dormant` since the phase was archived; `P5b-B2`'s threshold comes from the paper's reported numbers, not a null — `claims/EXPERIMENTS.md`
- **Depends on:** 1@6a6e6a3c1a, 2@27c0fbe55d
- **Feeds:** none
- **Open threads:**
  - The steering literature moved past the paper, and the causal-chain claim lost a link — `p5b_manifold_steering/lit-5b.md` §3
- **After Phase 10:**
  - The ordered-concept positive control on Pythia: one of the paper's four tasks through our pipeline (forward pass: the task's prompts at a few checkpoints)
- **Reviewed:** 2026-09-24 · body `777eb42b96`
<!-- /phase-card -->

## Corrections received

Corrections to this phase written elsewhere, one line each (`CLAUDE.md` Stop
step 2; `docs/phase_card.md`). Backfilled 2026-09-24.

- 2026-07-18 · an LM-head loader exists (`core/lm_loading.py`), which blocker 1 needed · `archive/p5_single_mstate_analysis/status-5.md` "v2 follow-up: causal_tests.py migration — DONE (item 3 aftermath, closed)"
- 2026-09-16 · a positive control on an ordered concept must come first · `PROJECT.md` "One design change this review proposes outright"

**Last verified:** not run. No results file exists for this phase (confirmed — no
`p5b`/`phase5b` output found in the project's results artifacts).
**Overall:** Not started (execution). Code and tests exist (`manifold_fit.py`,
`isometry_test.py`, `merge_teleportation_subspace.py`, `subspace_isometry_file.py`,
`p5b_io.py`, `run_5b.py`, plus `test_phase5b.py` / `test_phase5b_io.py` /
`test_p5b_integration.py`) — implementation is built, no run has been executed against it.

## Verdict table

No verdicts — nothing has been run. All four sub-experiments (A: manifold fitting, B:
isometry test, C: merge-event teleportation, D: S-subspace isometry) and all eight
falsification predictions (P5b-A1/A2, B1/B2/B3, C1/C2/C3, D1/D2) are outstanding.

## Known blockers

1. **New requirement not yet built at time of writing:** output distributions (logits) at
   each layer, for the target model/prompt. Phase 1 only stores activations. `logit_cache.py`
   is specified in the design to provide this via a single re-forward pass; check whether it
   exists as a working module before running sub-experiment A.
2. Sub-experiment D depends on Phase 2/6 S/A projectors (`ov_projectors_{stem}.npz`) —
   available, not itself a blocker.

## Not yet done

Everything. This is a fully-specified, unexecuted phase. First run should start with
sub-experiment A (manifold fitting) since B, C, and D all consume its output (`mh_params.npz`,
`my_params.npz`).
