# STATE — read this first, and only this, to start

**Last updated:** 2026-09-24 (phase cards: a dependency hashes only the upstream `## Corrections received`, so a routed correction stales its readers and nothing else does; "Open" 7 revised after #85's review) · **Cap:** 150 lines (tier-0 lint enforces it).
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
| Current stage | Stage 0, option B: all 20 prompts × 19 checkpoints (380 runs) under v2, in 10-h chunks via `tools/run/stage0_chunk.py`, **3 chunks** planned. **Pin `64a4087`** (run tree `../Mets-stage0`). **Chunk 1 DONE** (2026-09-22 21:20 → 2026-09-23 10:31; survived an overnight suspend): 144/380 runs indexed, all populated. **Chunk 2 RUNNING** (started 2026-09-24 05:35:20, driver pid 23468 under `flock` + `systemd-inhibit`; plan 233 runs in 48 invocations, ~9.9 h after re-fitting, so **chunk 3 is the last**, 3 runs). Log `data/phase12/stage0_logs/chunk_2026-09-24_05-35-20.log`. Detail and the guarded launch block: `handoff-10.md` §0.3 (the runbook) |
| Next after it | Holdout guard (`docs/PHASE_REVIEW.md` Parked 1), then Stage 1 on the **8 v1 prompts only**: `ext_sem_threshold` sweep, then the token-composition table. **The 12 new v2 prompts are held out on 410m** as the confirmation set until registrations are frozen (user, 2026-09-23; `handoff-10.md` §0.4). Partly seen already via `CLAIM-C` on 1.4b/gpt2-large; scope, count, order and release are open for the user (`docs/PHASE_REVIEW.md` "Open") |
| Parallel thread | Phase review, docs-only: a card per phase, a generated phase table, duplicate map, after-Phase-10 list, e-value plan. 9 sessions — `docs/PHASE_REVIEW.md`; its sessions table holds each session's findings. **Sessions 1–7 done**: cards for 1 through 9 (`docs/PHASES.md`; 9's is in a new `status-9.md`: parked, nothing run under its name, its E0 ran as Phase 10's F1), template `docs/phase_card.md`, lint rule `phase-card`. Standing rules from it: a correction to an earlier phase adds a line to its `## Corrections received` (user, 2026-09-23); deleted code may go if its intent stays (user, 2026-09-24; `LESSONS.md` lesson 12). Decided: the 2d pilot is quarantined and unseen; `P-T1`/`P-M1` get a fresh manifested run (`status-2d.md`). For the user before Phase 10 registers: `docs/PHASE_REVIEW.md` Parked 8 (410m's induction axis is spent). Session 6 found `P-I5`'s 70m target `L3H6` is not a matcher (Phase 8, 2026-09-13; routed to `status-7.md`). Next: session 8 (10). Holdout guard not built: Stage 1 not close |

## Blocked on the user

1. **Launch Stage 0 chunk 3** once chunk 2 logs `chunk end` (≈ 15:30 on
   2026-09-24 if the box stays awake): `handoff-10.md` §0.3's guarded block.
   `pgrep -af` matches its own shell; use `pgrep -f 'python -m tools.run.stage0_chunk'`.
2. **P-I5 runs on whatever battery is current.** `p7_motifs/p_i5_ablation.py`
   iterates `core.config.PROMPTS`; since v2 it gates on 20 prompts, not the
   8 its calibration record used. Nightly smoke red since 2026-09-19 because
   of it. 14 live files iterate the live battery. Needs a decision: pin
   P-I5 to v1's keys, or re-register. **Also its target:** the registered
   statement says "an induction head", and its real run's `L3H6` (70m) has
   induction score 0.009; 70m's matcher is `L0H3` (`status-7.md` "Corrections
   received", 2026-09-24).
3. **Branch protection on `main`** is off; red CI has been merged
   (#57, #58). Enabling "require CI to pass" is a GitHub setting.
4. **Register the large-read hook.** `scripts/hooks/guard_large_read.py` is built and
   tested; wiring it into `.claude/settings.json` (PreToolUse, matcher `Read`) was
   refused to Claude as self-modification. Snippet: PR #65's description.
5. **`P6-R2`/`R4` registry `notes` say no unit is registered**; the
   structured `null_construction` and the code say `model` (2026-08-25).
   Amending the notes is a registry edit, so it is yours. The same goes for whether
   `model` fits Pythia before `P6-R4` runs (`docs/PHASE_REVIEW.md` Parked 6).
6. **Which runs `P-T1` / `P-M1` are scored on** (checkpoints, prompts, pooled or
   per run) is not in the registry; decide before the fresh 2d run. Also:
   should `p_value_p_t1` refuse, rather than report "0 candidates", when
   heads lack a bandwidth scan? (`status-2d.md` "The 2026-08 Pythia pilot
   (quarantined)".) And 10 `data/analysis/` scripts still default to the main
   tree; Claude does not `git add` under `data/`, so they're yours to change.
7. **Which rung for `P-I7` and the first Phase 8 registration.** 70m went to
   exploration without the recorded call `design-8.md` asked for; 1b is
   measured by outside groups; so both compete for 1.4b (`status-8.md` card).
8. **Who re-reads the reader cards after a routed correction** (#86 review,
   finding 1). A line routed to Phase 1 stales its 13 carded readers, and the
   gate stays red until each is re-stamped. Either the correcting PR does
   all of them, or the lint only warns and each reader's next review clears it.

## Open PRs and branches

`claude/corrections-scoped-deps` (in `../Mets-work-2`): each dependency hashes only the upstream Corrections section (follows #85, merged 2026-09-24, branch deleted). #84 (`claude/phase-cards-10`, card 10, in `../Mets-work`) open; whichever merges second needs a rebase and `--stamp 9`, because #84 gives `status-10.md` a Corrections section, and card 9 reads 10. #83 merged 2026-09-24. #82 merged 2026-09-24, its branch deleted. #81 merged 2026-09-24, its branch deleted. #80 merged 2026-09-24.
#79 (`run_2d.py` measurement only + worktree-gate fix), #78, #77, #76 merged;
`claude/p2d-runner-gates` deleted. **Branch cleanup done 2026-09-23 (user asked):** every remote and
local branch except `main` deleted, the 3 non-ancestors included, after checking them:
two held only merge commits with no hand resolution, and `cf5f7ee`'s 352 added
lines are in main verbatim except 4 that main has since rewritten.
`../Mets-work-3a` removed. The main tree's `main` is behind `origin/main`
(fast-forward refused to Claude). Nightly smoke red (Blocked item 2).
Current state: `./scripts/status.sh`.

## Where things stand (one line each; detail behind the pointer)

- `CLAIM-C`: gate ran, **INSUFFICIENT** twice: p floor 0.0661 at 8 prompts (§3.41); at 20 v2 prompts on 1.4b + gpt2-large the floor is 0.0002 but the homogeneity correction is untabulated past 12 (§3.46). Fix ~45 min of calibration, deferred by the user.
- e-value audit: complete, 39 registered predictions, zero e-values — §3.45.
- Phase 10 free rows (tier 1): attention flip ~94 % causal mask; F0 fails; identity coupling optimal 99.5 %; HDBSCAN partition not reproducible (ARI p5 0.347) — `status-10.md`.
- Literature: five papers read as primary text — `lit-10.md` §11–15, `PROJECT.md` §3.52.
- Token cost (2026-09-22): scan of Claude Code docs + 5 papers (abstracts only) — `archive/docs/agent_context_scan_2026-09-22.md`. Built: `docs/index/` (section indexes), large-read hook (unregistered, Blocked item 4). Measured on the 2026-09-22 session's own transcript: 147 calls, ~19.0M context tokens re-read (avg ~129k/call, peak 214k) vs ~34k of tool output. Session length is the cost driver, not file size. Built since: `tools/session_cost.py` (calls, context, tool output off a transcript), `docs/cost_log.md` (one row per unit; Stop step 7; lesson 9's 2×-median rule), `scripts/status.sh` (Start step 2), and `CLAUDE.md` "While working" lines (one session per unit, batch calls, Edit not shell). Next: A3 (split `handoff-10.md` by stage) and A4 (Stop protocol → skill, claims rules → path rule).
- Archive batch (2026-09-22): PROJECT's §1, UPDATE_PLAN, POPPER_PLAN's DONE chunks, three CHANGES/PLAN files, six `docs/` one-offs → `archive/`; map `archive/MOVED.md`, lint rule `cited-md-path`, rewriter `tools/rewrite_moved_refs.py`. `literature-8.md` kept (not a duplicate of `lit-8.md`).
- **Decided 2026-09-22: no per-phase STATE/PROJECT files.** Instead: a card on each `status-N.md`, a generated phase table, and a staleness lint — built (session 1); 17 of 19 rows carded (`docs/PHASES.md`); left: 1d (archived), 10, and 9 once it has a status file.
- Carried from `archive/UPDATE_PLAN.md`: BLOCKED re-derive `DEGENERATE_RANK_THRESHOLD` / `FIEDLER_ACTIVE_RANK_THRESHOLD` on the normed scale (needs the sweep's normed-rank distribution); BLOCKED persist per-head Fiedler (`p1_io._save_sinkhorn` fixed, needs a rerun; fold into the next forward pass, `status-1.md`); BLOCKED `geometry.json` must carry `beta_eff_per_head` (`status-1c.md`); OPEN run Phase 1c/2d on artifacts (`tools/preflight_1c.py` first; `INDEX.md` 1c/2d rows); OPEN regenerate the energy-trajectory PNGs (wrong citation baked into suptitles, `math-1c.md:869`).
- Also from `archive/UPDATE_PLAN.md`, **for a decision**: §0 left `status-2.md:74,231` and Phase 1b's "Theorem 6.3" for cone collapse (live in `p1b_hemisphere/p1b_report.py:496,503`; the plan says Lemma 6.4, `math-1b.md:38` says "Lemma 6.4, feeding Theorem 6.3", so possibly fine). §4 not doing yet: the 1.4B sweep (gated on claim (c)), new checkpoints, BBGKY, diffusive regularisation, the β→∞ limit.
- Registry: untouched since the audit. Nothing in Phase 10 is registered. `results/p2d_pilot` (main tree) is quarantined: do not open (user, 2026-09-23; `status-2d.md`).

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

- Worktree script runs: `METS_REPO` now defaults to the script's own checkout, so set `METS_DATA=<main>/data` or it reads the worktree's empty `data/`.
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
