# Phase 5b — STATUS

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
