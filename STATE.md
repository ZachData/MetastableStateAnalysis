# STATE — read this first, and only this, to start

**Last updated:** 2026-09-23 (#69 archive batch merged; /challenge-pr fix) · **Cap:** 150 lines (tier-0 lint enforces it).
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
| Current stage | Stage 0, option B: all 20 prompts × 19 checkpoints (380 runs) under v2, in 10-h chunks via `tools/run/stage0_chunk.py`, **3 chunks** planned. **Pin `64a4087`** (run tree `../Mets-stage0`). **Chunk 1 was started 2026-09-22 21:20 and killed: no runs done** (no `stage0_index.json`; 4 orphaned step143000 dirs in `data/phase12/2026-09-22_21-20-26`, which `status` counts and readers must not glob). Runbook `handoff-10.md` §0.3 |
| Next after it | Stage 1: `ext_sem_threshold` sweep, then the token-composition table |

## Blocked on the user

1. **Relaunch Stage 0 chunk 1** (`handoff-10.md` §0.3) from `../Mets-stage0`
   at pin `64a4087`.
2. **P-I5 runs on whatever battery is current.** `p7_motifs/p_i5_ablation.py`
   iterates `core.config.PROMPTS`; since v2 it gates on 20 prompts, not the
   8 its calibration record used. Nightly smoke red since 2026-09-19 because
   of it. 14 live files iterate the live battery. Needs a decision: pin
   P-I5 to v1's keys, or re-register.
3. **Branch protection on `main`** is off; red CI has been merged
   (#57, #58). Enabling "require CI to pass" is a GitHub setting.
4. **Register the large-read hook.** `scripts/hooks/guard_large_read.py` is built and
   tested; wiring it into `.claude/settings.json` (PreToolUse, matcher `Read`) was
   refused to Claude as self-modification. Snippet: PR #65's description.

## Open PRs and branches

#71 `claude/challenge-pr-jq`: `/challenge-pr` fix (it could not run since `d5d6218`). #69 (archive batch) merged
2026-09-23 unreviewed: run `/challenge-pr 69` once #71 merges. #67, #68 merged 2026-09-22.
Nightly smoke last red 2026-09-22 (Blocked item 2). Current state: `./scripts/status.sh`.

Superseded remote branches, safe to delete: `claude/p10-free-rows`,
`claude/aca-phase-9-planning-rvzw3x`, `claude/attention-collapse-augmentation-qsxwg8`.

## Where things stand (one line each; detail behind the pointer)

- `CLAIM-C`: gate ran, **INSUFFICIENT**, p floor 0.0661 at 8 prompts — `PROJECT.md` §3.41, §3.46.
- e-value audit: complete, 39 registered predictions, zero e-values — §3.45.
- Phase 10 free rows (tier 1): attention flip ~94 % causal mask; F0 fails; identity coupling optimal 99.5 %; HDBSCAN partition not reproducible (ARI p5 0.347) — `status-10.md`.
- Literature: five papers read as primary text — `lit-10.md` §11–15, `PROJECT.md` §3.52.
- Token cost (2026-09-22): scan of Claude Code docs + 5 papers (abstracts only) — `archive/docs/agent_context_scan_2026-09-22.md`. Built: `docs/index/` (section indexes), large-read hook (unregistered, Blocked item 4). Measured on the 2026-09-22 session's own transcript: 147 calls, ~19.0M context tokens re-read (avg ~129k/call, peak 214k) vs ~34k of tool output. Session length is the cost driver, not file size. Built since: `tools/session_cost.py` (calls, context, tool output off a transcript), `docs/cost_log.md` (one row per unit; Stop step 7; lesson 9's 2×-median rule), `scripts/status.sh` (Start step 2), and `CLAUDE.md` "While working" lines (one session per unit, batch calls, Edit not shell). Next: A3 (split `handoff-10.md` by stage) and A4 (Stop protocol → skill, claims rules → path rule).
- Archive batch (2026-09-22): PROJECT's §1, UPDATE_PLAN, POPPER_PLAN's DONE chunks, three CHANGES/PLAN files, six `docs/` one-offs → `archive/`; map `archive/MOVED.md`, lint rule `cited-md-path`, rewriter `tools/rewrite_moved_refs.py`. `literature-8.md` kept (not a duplicate of `lit-8.md`).
- **Decided 2026-09-22: no per-phase STATE/PROJECT files.** Instead: a header on each `status-N.md`, a generated phase table, and a dependency-staleness lint — **the next PR**.
- Carried from `archive/UPDATE_PLAN.md`: BLOCKED re-derive `DEGENERATE_RANK_THRESHOLD` / `FIEDLER_ACTIVE_RANK_THRESHOLD` on the normed scale (needs the sweep's normed-rank distribution); BLOCKED persist per-head Fiedler (`p1_io._save_sinkhorn` fixed, needs a rerun; fold into the next forward pass, `status-1.md`); BLOCKED `geometry.json` must carry `beta_eff_per_head` (`status-1c.md`); OPEN run Phase 1c/2d on artifacts (`tools/preflight_1c.py` first; `INDEX.md` 1c/2d rows).
- Registry: untouched since the audit. Nothing in Phase 10 is registered.

## Machine and environments

**Two boxes; check which one you are on (`pwd`).** Stage 0 and all data work
run on the **local box** (Fedora, `/run/media/system/WDS_500`; details at the
end of this section, 164 GB free, all 19 410m checkpoints cached). The table
below is the **Claude Code cloud container** (2026-09-22): ephemeral, reclaimed
when idle, so nothing outside a pushed commit survives. The local box
still holds every byte of `data/` (list: `handoff-10.md` §0.2, CLAIM-C arms
hashed by `claims/audits/claim_c_real_run.json`, `data/phase12/`, pilot sweep on HDD_1TB).

| | |
|---|---|
| Repo | `/home/user/MetastableStateAnalysis` (the task branch is checked out here; no second worktree needed, no other session shares this container) |
| Hardware | 4 cores, 15 GB RAM, no GPU, **22 GB free disk** (after the env) |
| Network | Allowed: PyPI, GitHub. **Blocked: `huggingface.co`, `conda.anaconda.org`, `download.pytorch.org`** (environment network policy) |
| `mets` env | venv `/home/user/mets`, **pip, not conda** (conda channels blocked): py3.10.20 (`/usr/bin/python3.10`), numpy 2.2.6, scikit-learn 1.7.2, hdbscan 0.8.41 (manylinux wheel), scipy 1.15.3, torch 2.14.0+cu130 (PyPI), transformers 4.57.6. The four pinned versions match `clustering.py`'s note; **whether it reproduces historical partitions is untested** (no activations here to replay) |
| `.venv` (py3.14) | not built; the `mets` venv runs the gate |
| Gate | `PATH=/home/user/mets/bin:$PATH ./scripts/check.sh` → 2622 passed, 5 skipped, 83 s |
| `data/` | only the tracked `data/analysis/*.py`. No HF cache, no run dirs |
| Can run here | tests, lint, docs, math checks, anything not needing weights or `data/` |
| Cannot run here | any forward pass (no weights reachable), Stage 0 (57–100 GB > 22 GB), anything reading the 152 WDS dirs or CLAIM-C arms |

Local box: conda `mets` at `/run/media/system/WDS_500/miniforge3/envs/mets/bin/python`
with `CUDA_VISIBLE_DEVICES=""`; env `HF_HOME=<main>/data/hf HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 METS_RESULTS_DIR=<main>/data/phase12`,
plus `METS_REPO=$PWD METS_DATA=<main>/data` from a worktree; 164 GB free; Phase-1 run ≈ 200 s, 260 MB per prompt × checkpoint (410m, CPU).

## Hazards that bite (the full list with history is `LESSONS.md`)

- On the old box another Claude session may be live in the main tree: work in `../Mets-work`. Everywhere: `git fetch` and recheck HEAD before every commit.
- Push with the default SSH key; never set `GIT_SSH_COMMAND`.
- Target PRs at `main`, never at another PR's branch. Check merges with `git merge-base --is-ancestor <tip> origin/main`.
- An instrument that returns zeros/empty on a missing dependency is a bug: check a populated output on the FIRST run before launching the rest.
- Before trusting any "exists / doesn't exist / is green" in a doc, check the tree (`ls`, `gh run list`, manifests). Docs have been wrong about each.
- A headless `claude -p "/challenge-pr N"` that exits 0 has not necessarily reviewed anything: check the PR has the comment (`LESSONS.md` lesson 2).

## Map (open only what the task needs)

| need | file |
|---|---|
| Phase → directory | `INDEX.md` |
| Phase detail, numbers, how to re-run | `<phase>/status-N.md` |
| Scoped plan for the active thread | `p10_cluster_function/handoff-10.md` |
| History, reasoning, the §3.x record | `PROJECT.md` via `docs/index/PROJECT.idx.md` (line ranges; never read whole) |
| Registered predictions | `claims/registry.json`, `PREDICTIONS.md` |
| What went wrong and the rule it produced | `LESSONS.md` |
