# archive/MOVED.md — where a cited document went

Machine-read by `tools/lint_repo.py` (rule `cited-md-path`) and
`tools/rewrite_moved_refs.py`. Every `.md` path cited in a live file must
exist in the tree, or be in the **Moved** table (old → new), or be in the
**Absent** table (cited, never in the tree, and why). Paths are
repo-relative. A row with `§` names a section that moved out of a file that
stayed.

Batch of 2026-09-22 (user-approved): the one-offs, the finished plans and the
old startup blocks, so that the live docs are the ones a session reads.
`p8_scale_ladder/literature-8.md` was on the list conditionally ("check
whether it duplicates `lit-8.md` first"); it does not (it holds the phase's
fetched readings, 18 arXiv ids `lit-8.md` lacks), so it stayed.

## Moved

| old | new | when |
|---|---|---|
| `UPDATE_PLAN.md` | `archive/UPDATE_PLAN.md` | 2026-09-22 |
| `core/CHANGES.md` | `archive/core/CHANGES.md` | 2026-09-22 |
| `p1b_hemisphere/CHANGES-1b.md` | `archive/p1b_hemisphere/CHANGES-1b.md` | 2026-09-22 |
| `p2b_imaginary/PLAN_2b.md` | `archive/p2b_imaginary/PLAN_2b.md` | 2026-09-22 |
| `docs/literature_scan_2026-09-10.md` | `archive/docs/literature_scan_2026-09-10.md` | 2026-09-22 |
| `docs/literature_scan_2026-09-13.md` | `archive/docs/literature_scan_2026-09-13.md` | 2026-09-22 |
| `docs/results_provenance_audit_2026-09-05.md` | `archive/docs/results_provenance_audit_2026-09-05.md` | 2026-09-22 |
| `docs/deleted-branches-2026-09-10.md` | `archive/docs/deleted-branches-2026-09-10.md` | 2026-09-22 |
| `docs/CI_BASELINE.md` | `archive/docs/CI_BASELINE.md` | 2026-09-22 |
| `docs/agent_context_scan_2026-09-22.md` | `archive/docs/agent_context_scan_2026-09-22.md` | 2026-09-22 |
| `PROJECT.md §1` | `archive/PROJECT-start-here.md` | 2026-09-22 |
| `POPPER_PLAN.md §A0` | `archive/POPPER_PLAN-done.md §A0` | 2026-09-22 |
| `POPPER_PLAN.md §A1` | `archive/POPPER_PLAN-done.md §A1` | 2026-09-22 |
| `POPPER_PLAN.md §A2` | `archive/POPPER_PLAN-done.md §A2` | 2026-09-22 |
| `POPPER_PLAN.md §A3` | `archive/POPPER_PLAN-done.md §A3` | 2026-09-22 |
| `POPPER_PLAN.md §A4` | `archive/POPPER_PLAN-done.md §A4` | 2026-09-22 |
| `POPPER_PLAN.md §A5` | `archive/POPPER_PLAN-done.md §A5` | 2026-09-22 |
| `POPPER_PLAN.md §A6` | `archive/POPPER_PLAN-done.md §A6` | 2026-09-22 |
| `POPPER_PLAN.md §A7` | `archive/POPPER_PLAN-done.md §A7` | 2026-09-22 |
| `POPPER_PLAN.md §B1` | `archive/POPPER_PLAN-done.md §B1` | 2026-09-22 |
| `POPPER_PLAN.md §B2` | `archive/POPPER_PLAN-done.md §B2` | 2026-09-22 |
| `POPPER_PLAN.md §B3` | `archive/POPPER_PLAN-done.md §B3` | 2026-09-22 |
| `POPPER_PLAN.md §B4` | `archive/POPPER_PLAN-done.md §B4` | 2026-09-22 |
| `POPPER_PLAN.md §B5` | `archive/POPPER_PLAN-done.md §B5` | 2026-09-22 |
| `POPPER_PLAN.md §B7` | `archive/POPPER_PLAN-done.md §B7` | 2026-09-22 |
| `POPPER_PLAN.md §C1` | `archive/POPPER_PLAN-done.md §C1` | 2026-09-22 |
| `POPPER_PLAN.md §C2` | `archive/POPPER_PLAN-done.md §C2` | 2026-09-22 |
| `p6_subspace/design-6.md` | `archive/p6_subspace/design-6.md` | 2026-08-22 |

## Absent

| cited | why it is cited and not here |
|---|---|
| `MATH.md` | Never in this repo; the most load-bearing absence (`INDEX.md`, "Referenced files that do not exist") |
| `DESIGN_pythia_frames.md` | Never in this repo; `core/` cites it by item number (`INDEX.md`, same table) |
| `CHANGES_jlens_adjacent.md` | Never in this repo (`INDEX.md`, same table) |
| `design-9.md` | Not written yet: `plan-9.md` says it is "still not `design-9.md`" |
| `design-10.md` | Not written yet: cited as "when that file exists" |
| `docs/PARTICLE_ONTOLOGY.md` | Folded into Phase 7, never written as a file (`POPPER_PLAN.md`, "Workstream C folded into Phase 7") |
| `DESIGN.md` | A generic "the module's design doc" in `core/DESIGN_dual_reading.md`, `core/dual_reading.py` and `design-2b.md`; no file of that name |
| `STATUS.md` | Generic "a status doc" in `core/nulls.py` |
| `HANDOFF.md` | The sister project's file (`lora_ind`, `PROJECT.md` §3.9), not this repo's |
| `REVIEW.md` | The sister project's file, as above |
| `phase1b_cross_run.md` | A report `run_1b.py` generates into the run directory, not a tracked doc |
| `p2_eigenspectra/ISSUES_p2.md` | Cited by `core/pythia_registry.py`; never in git history |
| `README_phase6.md` | Cited as a stale header; deleted in `c3f2b73` (2026-07-14) |
| `readme-phase2c.md` | Cited as a stale header; deleted in `c3f2b73` (2026-07-14) |
