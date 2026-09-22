# STATE — read this first, and only this, to start

**Last updated:** 2026-09-22 · **Cap:** 150 lines (tier-0 lint enforces it).
Overwrite, never append: this file says what is true *now*. History goes to
`PROJECT.md` §3.x and `git log`; mistakes go to `LESSONS.md`.

## What this is

Measuring metastable clustering of tokens in transformer residual streams
(the Geshkovski et al. particle picture) on Pythia checkpoints, with
pre-registered predictions scored by e-values (`claims/registry.json`).
One solo researcher (the user) + Claude. Tier 1 = exploratory, unregistered.

## Now

| | |
|---|---|
| Active thread | Phase 10, cluster function — `p10_cluster_function/handoff-10.md` |
| Current stage | Stage 0: Phase-1 battery 8 → 20 prompts. **Blocked on the user** (below) |
| Next after it | Stage 1: `ext_sem_threshold` sweep, then the token-composition table |

## Blocked on the user

1. **Stage 0 battery shape.** 152 WDS dirs are battery v1 (`1e47918ef77a`);
   new runs write v2 (`06790b90dcfe`), which `core/prompts.py` says is not
   comparable. v1 texts ARE byte-identical inside v2 (hash reconstruction).
   Options: 12 new under v2 (~57 GB, ~12 CPU-h) or all 20 under v2
   (~100 GB, ~20 CPU-h). Detail: `handoff-10.md` §0.2.
2. **P-I5 runs on whatever battery is current.** `p7_motifs/p_i5_ablation.py`
   iterates `core.config.PROMPTS`; since v2 it gates on 20 prompts, not the
   8 its calibration record used. Nightly smoke red since 2026-09-19 because
   of it. 14 live files iterate the live battery. Needs a decision: pin
   P-I5 to v1's keys, or re-register.
3. **Branch protection on `main`** is off; red CI has been merged
   (#57, #58). Enabling "require CI to pass" is a GitHub setting.

## Open PRs and branches

| branch | PR | what | state |
|---|---|---|---|
| `claude/float32-tripwire-ci` | #61 | CI flake fix (float32 tripwire at d=1024) | green, awaiting merge |
| `claude/workflow-state-lessons` | #62 | this file, `LESSONS.md`, CLAUDE.md start/stop protocol, lint rule 6 | open |
| `claude/ci-matrix-and-alerts` | #63 | pure tier on py3.10 + py3.14; nightly-smoke failure opens an issue | CI was pending at handoff: check it |
| `claude/p10-stage0` | #64 | Stage 0 timing probe + battery finding in `handoff-10.md` | open, docs only |

Suggested merge order: #61, #62, #63, #64 (independent; #62 and #61 both touch
`PROJECT.md` in different hunks). After merging, confirm each with
`git merge-base --is-ancestor`.

Superseded remote branches, safe to delete: `claude/p10-free-rows`,
`claude/aca-phase-9-planning-rvzw3x`, `claude/attention-collapse-augmentation-qsxwg8`.

## Where things stand (one line each; detail behind the pointer)

- `CLAIM-C`: gate ran, **INSUFFICIENT**, p floor 0.0661 at 8 prompts — `PROJECT.md` §3.41, §3.46.
- e-value audit: complete, 39 registered predictions, zero e-values — §3.45.
- Phase 10 free rows (tier 1): attention flip ~94 % causal mask; F0 fails; identity coupling optimal 99.5 %; HDBSCAN partition not reproducible (ARI p5 0.347) — `status-10.md`.
- Literature: five papers read as primary text — `lit-10.md` §11–15, `PROJECT.md` §3.52.
- Registry: untouched since the audit. Nothing in Phase 10 is registered.

## Machine and environments

**Moving to a new machine (2026-09-22).** Everything below is the OLD local box
(Fedora, `/run/media/system/WDS_500`). On a new box: git has all code and docs;
**`data/` does not travel with git** — the HF cache (70 GB), the 152 WDS run
dirs, CLAIM-C's scoreable arms (hashed by `claims/audits/claim_c_real_run.json`),
the Stage 0 probe `data/phase12/2026-09-22_16-43-47`, and `data/analysis/`'s
Phase 10 records all live only on the old box (pilot sweep: HDD_1TB). Rebuild
the conda `mets` env with the exact versions in `p1_mstate_tracking/clustering.py`'s
toolchain note (py3.10.20, hdbscan 0.8.41, sklearn 1.7.2, numpy 2.2.6), or new
partitions will not match historical ones. Rewrite this section for the new box.


| | |
|---|---|
| Repo | `/run/media/system/WDS_500/Mets` (main tree, stays on `main`) |
| Task worktree | `../Mets-work` — at most one; remove after its PR merges |
| `.venv` | project interpreter, py3.14, CPU torch. Tests, tools |
| conda `mets` | `/run/media/system/WDS_500/miniforge3/envs/mets/bin/python`. **Anything that runs HDBSCAN** (only install that reproduces historical partitions). Has CUDA; set `CUDA_VISIBLE_DEVICES=""` to match every run on disk (CPU) |
| Env vars | `HF_HOME=<main>/data/hf HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 METS_RESULTS_DIR=<main>/data/phase12` |
| From a worktree | also `METS_REPO=$PWD METS_DATA=<main>/data`, else runners import the main tree's code |
| Gate | `./scripts/check.sh` (with `.venv` on PATH) → 2622 passed, ~55 s |
| Phase-1 run cost | 410m, CPU, ~400 tokens: ~200 s, ~260 MB per prompt × checkpoint |
| Disk | 164 GB free on WDS_500 (2026-09-22). `data/` is never `git add`ed |

## Hazards that bite (the full list with history is `LESSONS.md`)

- Another Claude session may be live in the main tree: work in `../Mets-work`, `git fetch` and recheck HEAD before every commit.
- Push with the default SSH key; never set `GIT_SSH_COMMAND`.
- Target PRs at `main`, never at another PR's branch. Check merges with `git merge-base --is-ancestor <tip> origin/main`.
- An instrument that returns zeros/empty on a missing dependency is a bug: check a populated output on the FIRST run before launching the rest.
- Before trusting any "exists / doesn't exist / is green" in a doc, check the tree (`ls`, `gh run list`, manifests). Docs have been wrong about each.

## Map (open only what the task needs)

| need | file |
|---|---|
| Phase → directory | `INDEX.md` |
| Phase detail, numbers, how to re-run | `<phase>/status-N.md` |
| Scoped plan for the active thread | `p10_cluster_function/handoff-10.md` |
| History, reasoning, the §3.x record | `PROJECT.md` (large — grep, don't read) |
| Registered predictions | `claims/registry.json`, `PREDICTIONS.md` |
| What went wrong and the rule it produced | `LESSONS.md` |
