# STATE — read this first, and only this, to start

**Last updated:** 2026-10-02 (1d unit 2's first check: the 10 real step-0 inits sit among 40 Pythia-σ re-inits in all 24 cells; #130 merged; this unit in a PR; trained cells not read) · **Cap:** 150 lines (tier-0 lint enforces it).
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
| Active thread | **Phase 1d, what a cluster is** (`p1d_cluster_ensemble/status-1d.md`). Phase 10 is on hold (user, 2026-09-25) until the project can say what a cluster is: every Phase 10 row reads one HDBSCAN partition (`min_cluster_size=2`), a confound in all of them. **Summary table, with a pointer per row: `status-1d.md` "Blocked 11″ decided, and where 1d stands"** (also the user's standing directions for 1d). Pages: https://claude.ai/artifact/XYbhuwAvMkfkWgv27v9qVR (null atlas, 2026-09-29; its "where it stands" box predates the units below); https://claude.ai/artifact/1C18uD3MGLiaJ1Ux8F1HVn (plain-language catch-up, 2026-10-01; no numbers of its own) |
| Current stage | **Programme of Blocked 11⁗ (`status-1d.md` "Blocked 11⁗ decided"): units 0, 1 merged (#129, #130); unit 2, the architecture null, first check run and in a PR** (`status-1d.md` "Unit 2: the architecture null — first check"; output `data/p1d/arch_null_2026-10-02/`). Unit 1 in one line: trained groups mostly move with the passage (centred size 2: 0.95 / 0.89 / 0.74 by band), step 0's do not ("Unit 1: move the text"). Unit 2: the 10 real step-0 inits (pythia-410m + PolyPythias seeds 1–9) are Pythia's init at float16 values, and that carries the claim that 40 Pythia-σ re-inits (now float16-rounded, `/challenge-pr` on #131 finding 3) stand in for them. The cloud check passes **all 24 first-check cells** (`hdb_k` 2 / 4, `nn1`, `ci2` × band × frame; largest share 0.143, bound 0.20) but fails only a gross mismatch (~1.5 SD), with a small late lean (`ci2`, `hdb_k_4`, L17–24) it cannot resolve. No massive token at step 0. Settled before any trained record is opened (`design-1d.md` "For the trained cells"; yours to overrule): the per-cloud rank p reads the lumpier tail (lower for `nn1` / `ci2`), and T2's union over all 60 models forces recomputing changed prompts and re-running the check. **Next: unit 2's trained cells.** PolyPythias `step143000` is cached, unopened. Earlier route (four constructed nulls, the M = 32 cut, the position-keeping null on v1, which failed step 0 in all 6 cells): one line each in "Where 1d stands". **HDBSCAN's tie order is also in scikit-learn** (`tools/hdbscan_tie_repro.py`: 26 of 30 row orders change it in both); reporting upstream is the user's call (draft `docs/upstream/hdbscan_ties.md`) |
| On hold | **Phase 10**, `p10_cluster_function/handoff-10.md` (its own "Last updated" line says where it stopped: Stage 0 380/380 at pin `64a4087`, Stage 1 done, §1.6–§1.13). Waiting there: the context-shuffle test, a cross-checkpoint cluster matcher, Stage 2. **The 12 new v2 prompts stay held out on 410m** (`core/holdout.py`; every `data/phase12` runner, `run_1d.py` included, refuses them or takes `--v1-only` / `--allow-holdout`); scope and release are the user's (`docs/PHASE_REVIEW.md` "Open") |
| Parallel thread | Phase review, docs-only — `docs/PHASE_REVIEW.md`; its sessions table holds each session's findings. **All 9 sessions done.** Session 9's synthesis is `docs/PHASE_SYNTHESIS.md`: 19 duplicate questions, a ranked after-Phase-10 list (top: β's convention, `CLAIM-C`'s calibration to 20, `P-S1` at matched k), attainable E per adjudicable row (only `CLAIM-C`, `P-AB1` depend on prompt count; `CLAIM-C` cannot decide at its measured sign homogeneity), and memos for the four open decisions. Found: `P-S1`'s gate defaults to 500 draws, too few for E ≥ 20 (Parked 1 there), and `core/beta_eff.py` and `status-1c.md` name β's ×8 ends oppositely. Sessions 1–8: cards for 1 through 10 (`docs/PHASES.md`; 9's is in a new `status-9.md`: parked, nothing run under its name, its E0 ran as Phase 10's F1; 10's routes 3 corrections and fixes `handoff-10.md` §0.4's stale "228 directories" done-line to option B's 380), template `docs/phase_card.md`, lint rule `phase-card`. Standing rules from it: a correction to an earlier phase adds a line to its `## Corrections received` (user, 2026-09-23); deleted code may go if its intent stays (user, 2026-09-24; `LESSONS.md` lesson 12). Decided: the 2d pilot is quarantined and unseen; `P-T1`/`P-M1` get a fresh manifested run (`status-2d.md`). For the user before Phase 10 registers: `docs/PHASE_REVIEW.md` Parked 8 (410m's induction axis is spent). Session 6 found `P-I5`'s 70m target `L3H6` is not a matcher (Phase 8, 2026-09-13; routed to `status-7.md`). Next in this thread: nothing scheduled; the user's decisions in `docs/PHASE_SYNTHESIS.md` §3.2–3.3. Holdout guard: built (row above) |

## Blocked on the user

1. *(Cleared 2026-09-24; kept so the numbers stay stable.)* Guard for a later driver:
   `pgrep -f '[p]ython -m tools.run.stage0_chunk'` (the bracket stops it matching itself).
2. **P-I5's target.** Its battery is pinned (2026-09-29, user's call): every
   P-I5 loop reads its calibrated 8 prompts (`p_i5_gate.p_i5_battery`, hash
   `e77b5528f536`) and refuses on a change, so the live battery can keep
   growing. The nightly smoke is green on it (2026-09-30, first since 09-19; issue #70 closed). **Still open, its target:** the registered
   statement says "an induction head", and its real run's `L3H6` (70m) has
   induction score 0.009; 70m's matcher is `L0H3` (`status-7.md` "Corrections
   received", 2026-09-24). **Same question for `P-I1`:** `tools/run/behavioural.py`
   takes every prompt with a full 19-step sweep, so after chunk 3 that would include the 12.
   Since #88 it refuses them until you decide (`docs/PHASE_REVIEW.md` "Open" 3).
3. **Branch protection on `main`** is off; red CI has been merged
   (#57, #58). Require `lint`, both `pure` legs and `deps`: tick "Require status checks", type `tier` in its search box.
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
9. **β's scale convention and `P-S1`'s matched-k rule**, both open in code;
   memos in `docs/PHASE_SYNTHESIS.md` §3.2. **β's measured recommendation**
   (`status-1d.md` "β refit"): `beta_raw`, unit LN1, per-offset fixed effects,
   all heads, per band; β = 3.5 [1.6, 5.6] at v1 length. Quote it with its
   offset range: on v1's offsets ~2000-token prompts give nearly the same β.
   No β-free way to pick "kernel heads" yet (Parked there). §3.1: `CLAIM-C`'s rescore will most likely refuse (sign homogeneity above what
   12 prompts could survive; not 410m). Its scorer reads every live metastability
   key (20 now): pin it like P-I5, or recalibrate per count (`docs/battery_consumers.md`).
10. *(Decided 2026-09-30: retire core/halo/contested; `design-1d.md` rewritten. Kept so the numbers above stay stable.)*
11. *(Decided 2026-10-01: (b), done; 11′: keep M = 32 if step 0 passes on the long prompts, it did not; 11″: option (a), the position-keeping null, per-prompt counts beside the per-record verdict; 11‴ (2026-10-02): fit and draw before normalisation, smoother primary; run, fails all 6 cells. `status-1d.md` "Position-keeping null"; 11⁗ (2026-10-02): stop modelling nulls, run the five-unit programme in `status-1d.md` "Blocked 11⁗ decided": 0 lit scan + design, 1 move the text, 2 architecture null (random inits / PolyPythias), 3 scale spectrum, 4 positive controls. The estimator fix and the write-up are not taken up.)* Unit 0's design is fixed once #129 merges (its "Worth challenging" names the calls to check before unit 1 runs: T1 drops position 0 at step 0 too; the merge tree, not Markov stability, for unit 3).
12. **Report HDBSCAN's tie order upstream?** Both `hdbscan` and scikit-learn's `HDBSCAN` change their clusters when the same points come in another row order (`status-1d.md` "Blocked 11″ decided"). Known upstream as a symptom (hdbscan #265, #409), never diagnosed or fixed. Draft: a comment on #265 and a new scikit-learn issue (`docs/upstream/hdbscan_ties.md`). Posting is public, so it is yours.
13. **PDFs for the mean-field reading list**, ranked in its §4 (`docs/readings/meanfield_reading_list_2026-10-01.md`): 2604.01978, 2605.09213, 2412.09080 first. Scholarly hosts are blocked from the session.

## Open PRs and branches

**Open: this unit (`claude/p1d-arch-null`, in `../Mets-work`).** Every PR through #130 is merged (2026-10-02; #130's branch and worktree deleted); main tree at `01b8642`, behind `origin/main` (not fast-forwarded: another session may use it). **Merged branches not yet deleted** (the delete was refused to Claude by the permission classifier, 2026-10-02): `claude/p1d-unit0-design`, `claude/meanfield-reading-list` (local and remote; both ancestors of `origin/main`). Remote branches left that no open PR uses: `claude/intelligent-fermat-w4ug56`, `claude/modest-cannon-c3t3j9`, `claude/p1d-attention-communities`, `rvm-onboard` (not checked; delete only after `git merge-base --is-ancestor`).
Branch cleanup of 2026-09-23 (user asked; every branch but `main`, after checking each): `git log` of that date. Nightly smoke green since 2026-09-30.
Current state: `./scripts/status.sh` (without `gh` it reads the public API; exit 1 = could not see).

## Where things stand (one line each; detail behind the pointer)

- **Triage 2026-09-29 (proposal, not decided): `docs/TRIAGE_2026-09.md`.** Flagship = the anti-collapse force on the training axis (energy, β, T_eff, attention vs MLP), gated on Blocked 9; second paper = 1d written up as a null-model audit; §6a ranks mechanistic methods (exact per-component dissipation first; own small models, n-gram milestones, LLC; SPD later); local-object phases (3/4, 5, 7–8) set aside for low value, 1c/2d only for being blocked. Weight side: `FUTURE_IDEAS.md` D1/D3/D4 (weights only) set against the energy break and the OV repulsive phase at 1000–2000 (`PROJECT.md` §3.12 A).
- `CLAIM-C`: gate ran, **INSUFFICIENT** twice: p floor 0.0661 at 8 prompts (§3.41); at 20 v2 prompts on 1.4b + gpt2-large the floor is 0.0002 but the homogeneity correction is untabulated past 12 (§3.46). Fix ~45 min of calibration, deferred by the user.
- e-value audit: complete, 39 registered predictions, zero e-values — §3.45.
- Mutation testing (`mutmut`, `.github/workflows/mutation.yml`; the one place this count lives): `core/evalues.py` **397/413 killed**, the 16 left in `tools/mutation_accepted.json`, each reason probed on the live mutant (6 equivalent, 10 accepted with the input that tells them apart). What was missing: `LESSONS.md` 6 and 11. The pure tier fails when the code an accepted reason rests on changes (no mutmut needed); the Mutation workflow itself is not required. **For the user:** (1) `EProcess.from_record` has no production caller (the ledger replay uses `EProcess` + `add`): keep it, make it the replay path, or delete it. (2) `verify_ledger` never compares a record's stored alpha with the registry's, so a changed alpha is caught only where it flips a decision (0.05 → 0.2 passes): add the check now, or in the `core/adjudication.py` unit? **Parked (unreachable today; fix when `core/evalues.py` is next edited):** `average` multiplies before normalising, so a multi-term sum past 1.8e308 raises `OverflowError` (`average([1e308, 1e308])`), and a single w·e past it returns `(inf, True)` silently (`average([2.0], weights=[1e308])`); fix: `w / total` first. `required_p_for_rejection(log_E_prior=nan)` returns 0.0 instead of refusing. Next: `core/adjudication.py`, then `core/nulls.py`.
- Phase 10 free rows (tier 1): attention flip ~94 % causal mask; F0 fails; identity coupling optimal 99.5 %; HDBSCAN's run-to-run floor (ARI p5 0.347) was float32 distances; on float64 it is 1.000, and stored `repeated_tokens` labels are rounding (`status-1d.md` "Float64 distances…") — `status-10.md`.
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
end of this section, 72 GB free on 2026-09-29, all 19 410m checkpoints cached; `HDD_1TB` 310 GB free, long-prompt runs in `HDD_1TB/mets_data`). The table
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
| Gate | `./scripts/check.sh` → 2820 passed, 5 skipped, 72 s (2026-09-30, system py3.11); deps tier: `check.sh deps` after `pip install torch -r requirements/heavy.txt` (PyPI CUDA wheel; CPU index blocked) |
| `data/` | only the tracked `data/analysis/*.py`. No HF cache, no run dirs |
| Can run here | tests, lint, docs, math checks, anything not needing weights or `data/` |
| Cannot run here | any forward pass (no weights reachable), Stage 0 (57–100 GB > 22 GB), anything reading the 152 WDS dirs or CLAIM-C arms |

**Fleet box** (research-vm-infra; EC2 t4g.small, aarch64, 2 vCPU, 1.8 GB,
no swap, repo at `/home/ubuntu/MetastableStateAnalysis`, config
`infra/rvm.env`). Set up 2026-09-27; gate and smoke ran on `c5b56fd` (+ docs-only #107):
| | |
|---|---|
| venv | `/home/ubuntu/venv`, py3.10 (uv), torch 2.14.0+cpu, transformers 4.57.6, numpy 2.2.6, scipy 1.15.3, sklearn 1.7.2, **hdbscan 0.8.44** (built from source; no aarch64 wheel; the `mets` env has 0.8.41). S3 key `envs/aarch64/py3.10/MetastableStateAnalysis/9c0773c95606ad6f.tar.zst` (275 MB); boots restore it |
| Gate | `PATH=~/venv/bin:$PATH ./scripts/check.sh` → 2773 passed, **4 failed** (aarch64 numerics; `LESSONS.md` 4), 145 s |
| Smoke | `SMOKE_REAL_DEPS=1 pytest -m smoke` → 52 passed, 1 failed (P-I5, Blocked 2). OOM without swap; 24 min with 4 GB swap |
| S3 | bucket holds only this project (Lora_inductionhead's objects deleted 2026-09-27 on the user's instruction); `caches/huggingface/` = pythia-70m + tiny smoke models |

Local box: conda `mets` at `/run/media/system/WDS_500/miniforge3/envs/mets/bin/python`
with `CUDA_VISIBLE_DEVICES=""`; env `HF_HOME=<main>/data/hf HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 METS_RESULTS_DIR=<main>/data/phase12`,
plus `METS_REPO=$PWD METS_DATA=<main>/data` from a worktree; 72 GB free (2026-09-29); Phase-1 run ≈ 200 s, 260 MB per prompt × checkpoint (410m, CPU).

## Hazards that bite (the full list with history is `LESSONS.md`)

- Worktree script runs: `METS_REPO` now defaults to the script's own checkout, so set `METS_DATA=<main>/data` or it reads the worktree's empty `data/`.
- On the old box another Claude session may be live in the main tree: work in `../Mets-work`. Everywhere: `git fetch` and recheck HEAD before every commit.
- Push with the default SSH key; never set `GIT_SSH_COMMAND`.
- Target PRs at `main`, never at another PR's branch. Check merges with `git merge-base --is-ancestor <tip> origin/main`.
- An instrument that returns zeros/empty on a missing dependency is a bug: check a populated output on the FIRST run before launching the rest.
- Before trusting any "exists / doesn't exist / is green" in a doc, check the tree (`ls`, `./scripts/status.sh`, manifests). Docs have been wrong about each. A red step skips the steps after it: "smoke red" hid the deps tier for 10 nights.
- A headless `claude -p "/challenge-pr N"` that exits 0 has not necessarily reviewed anything: check the PR has the comment (`LESSONS.md` lesson 2).

## Map (open only what the task needs)

| need | file |
|---|---|
| Phase → directory | `INDEX.md` |
| Phase detail, numbers, how to re-run | `<phase>/status-N.md` |
| Wall time / peak memory / disk per job (container sizing) | `docs/compute_profile.md` |
| Scoped plan for the active thread | `p10_cluster_function/handoff-10.md` |
| History, reasoning, the §3.x record | `PROJECT.md` via `docs/index/PROJECT.idx.md` (line ranges; never read whole) |
| Registered predictions | `claims/registry.json`, `PREDICTIONS.md` |
| What went wrong and the rule it produced | `LESSONS.md` |
