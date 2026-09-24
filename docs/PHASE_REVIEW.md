# Phase review — handoff

**Opened:** 2026-09-23 · **Owner:** the user + Claude · **Runs beside:** Phase 10
Stage 0 (compute on the local box; this thread is docs-only).

## Why

Seventeen phase directories plus two unmerged branches (`INDEX.md` "In flight").
No file says, per phase, in one screen: what it asked, what ran on which inputs,
what it found, what superseded it, and what it left open. Three costs follow:

1. **Doing the same thing twice.** Seen already: Phase 5c's attention flip vs
   Phase 10 A0; Phase 1's carrying capacity (50–55) vs F14 / Lemma C.1;
   HDBSCAN reproducibility (`status-10.md` §3) vs the Phase 1d clusterer
   branch; steering in 3 (`archive/p3_crosscoder/steering.py`) vs 5b vs 7 `P-ST1`.
2. **Losing leads.** Archived phases carry open results (Phase 6's LDA
   inversion, Phase 4's "sparsity was the confound") that no plan cites.
3. **Stale maps.** `INDEX.md` carried a "Current priority" block dated
   2026-09-20 that duplicated `STATE.md` (deleted in session 1; every line
   had another home).

## Decisions (user, 2026-09-23)

| decision | choice |
|---|---|
| Where a phase review lands | A **card** at the top of its `status-N.md`; a phase table in `docs/` generated from the cards (built in session 1); a lint keeps them current. This is the "next PR" decided 2026-09-22 (`STATE.md`) |
| The 12 v2 prompts new to the 410m sweep | **Held out** as Phase 10's confirmation set. Stage 0 still runs all 20; Stages 1–5 read only the 8 v1 prompts until registrations are frozen (`p10_cluster_function/handoff-10.md` §0.4). **Partly seen already**: `CLAIM-C` ran them on 1.4b and gpt2-large (`PROJECT.md` §3.46), so see "Open" |
| Order vs Stage 0 | In parallel |
| What makes a card stale ("Open" 5, settled) | **Route corrections to the corrected phase.** A correction to an earlier phase adds a line to that phase's `## Corrections received`, which the existing body hash already covers (`CLAUDE.md` Stop step 2, `docs/phase_card.md`). Rejected: hashing the sections the pointers name. That watches only sections a card already cites, and each of Phase 1's 7 corrections arrived in a section no card cited yet, so it would have caught none of them (`LESSONS.md` lesson 1). Later: a warning lint for unrouted corrections (Parked 3); score the rules against the Superseded lists once several phases have cards (session 9) |

## Open (for the user, from `/challenge-pr` on #74)

1. **Blind for what?** Every Phase 10 measure, or only measures `CLAIM-C`'s
   runs did not compute (cluster count, membership, effective rank, Fiedler)?
   This decides whether F14, a cluster-count test, can be registered on the 12.
2. **12 or 20?** Are Phase 10 predictions scored on the 12, or on all 20?
3. **Order.** Is any registered prediction (e.g. `CLAIM-C` once its calibration
   is extended) scored on the 410m v2 runs before Phase 10 registers?
4. **Release.** If nothing gets registered by some point, what frees the 12?
   Until then exploration stays at 8 prompts, only 7 of them natural text.
5. ~~What makes a card stale~~: settled 2026-09-23, see "Decisions".

## The card

Fields, format, what the lint checks and how to clear a stale card:
`docs/phase_card.md` (the template). The table the cards generate:
`docs/PHASES.md`. Numbers stay where they already live ("write once"); the
card points.

**Workflow per card session:** copy the skeleton, fill it from the status
file (and the files it cites), grep for the phase outside its directory and
route any correction found there into its `## Corrections received`,
look in the main tree's `results/` and `data/` for runs the status file does
not mention, `python3 tools/render_phases.py --stamp <id>`,
`python3 tools/render_phases.py`, `./scripts/check.sh lint`. Adding a card
whose Depends on names a carded phase means updating that phase's Feeds too;
the lint checks both ends.

## Sessions (one unit each; `CLAUDE.md` "one session per unit")

| # | unit | state |
|---|---|---|
| 1 | Card template `docs/phase_card.md`, generator `tools/render_phases.py` → `docs/PHASES.md`, lint rule `phase-card` (fields filled, pointers resolve, and **stale when the phase's status file or any file in its "Depends on" changed since the review**, by content hash rather than commit). Phase 1 card done; `INDEX.md`'s priority block deleted | **done** 2026-09-23 |
| 2 | Cards: 1b, 1c. Also: "Open" 5 settled (corrections routed to `## Corrections received`, Phase 1 backfilled); found an unrecorded post-revision Phase 1b Pythia run from 2026-08-17 and recorded it in `status-1b.md` | **done** 2026-09-23 |
| 3 | Cards: 2, 2b, 2d, 6 (live `status-6.md`; `P6-R2`/`R4` active, same claim `H-OPERATOR`) | open |
| 4 | Cards: archived 3, 4, 5, 5b, 5c and frozen 6 (`archive/p6_subspace/status-6.md`); the Phase 1d and viz branches → keep / revive / drop | open |
| 5 | Cards: 7, 7d, 7e | open |
| 6 | Card: 8 | open |
| 7 | Card: 9 (no status file yet: create one in `p9_metric_intervention/` for the card), checked against 10 (`notes-10.md` §9) | open |
| 8 | Card: 10 | open |
| 9 | Synthesis: the duplicate-question map, the ranked after-Phase-10 list, the e-value plan below | open |

Before the first Stage 1 read: **the holdout guard** (Parked item 1). It is not
a review session but it is due before Stage 1 starts, not after session 9.

## The e-value plan (session 9, or its own session)

The machinery exists: `core/evalues.py` (p→e calibrator, e-process; `average`
for dependent units, since a product over-rejected, `LESSONS.md` lesson 6),
`core/adjudication.py` (ledger), 39 registered, 11 adjudicable, 0 adjudicated
(`claims/EXPERIMENTS.md`). What is missing is **power and four decisions**, not code.

1. **Attainable-E table at 8, 12 and 20 prompts.** With κ = 0.5 one prediction
   alone needs p ≤ 1/1600 for E ≥ 20. `CLAIM-C`'s floor was 0.0661 at 8
   prompts (E ≈ 1.9) and is 0.0002 at 20 (§3.46). 12 is the confirmation-set
   size: a CLAIM-C-shaped design there needs ≥ 11 informative prompts, a
   margin of one. Per adjudicable row, via `max_attainable_average_E`; rows
   that cannot reach 20 say so before anyone runs them.
2. **The four blocking decisions**, one short memo each for the user: β's
   scale convention, `P-T1`'s wording, `P6-R2`/`R4`'s exchangeable unit,
   `P-S1`'s matched-k clustering. Plus `P-I5`'s battery (`STATE.md` Blocked 2).
3. **Phase 10's route into the registry.** Which tier-1 findings graduate,
   and in what order relative to existing rows scored on the 410m v2 runs
   ("Open" 3). F14 is already named the one to register (`handoff-10.md`
   "Standing constraints"), subject to "Open" 1.
4. **Native e-values for new registrations** (testing by betting, likelihood
   ratio) instead of calibrating a p-value: usually more power, and they fit
   accruing prompts one at a time. Literature scan first (trigger 2 in
   `CLAUDE.md`); existing rows keep the calibrator. A proposal, not a decision.

## Parked

1. **Holdout guard in code.** A list of held-out prompt keys in `core/` that
   Phase 10 readers refuse unless passed an explicit flag, covering the 410m
   Stage 0 dirs **and** the runs that already touched the 12
   (`data/phase12/2026-09-19_*`, `claims/audits/claim_c_real_run.json`).
   Why: a rule only in docs has been missed before (`LESSONS.md`). Cost:
   small, `core/` + tests. Changes: whether the confirmation set survives Stage 1.
   **Not built in session 1** (2026-09-23): Stage 0 has chunks 2 and 3 to go
   (about 20 h, chunk 2 not launched), so Stage 1 was not close.
2. **A gate run from `../Mets-work` printed a warning from the main tree's
   `p1c_frames/integration_time.py`** (`tests/test_run_1c_beta_gate.py`),
   though `import p1c_frames` resolves to the worktree. Why: if some test
   imports or spawns from the main tree, worktree gates test `main`'s code, not
   the branch's. Cost: one look at that test's subprocess/env. Changes: whether
   any worktree gate result so far can be trusted for package code.
3. **Warn on unrouted corrections.** A lint warning (not a failure) when a
   commit adds text outside a carded phase's directory that names the phase,
   by path **or in prose** ("Phase 1b", `p1b_`), and that phase's
   `## Corrections received` did not change. Why: the routing rule
   ("Decisions") depends on people remembering it, and prose rules get
   forgotten (`LESSONS.md` lesson 1). Prose matching would have found 4 of
   the 5 external sources routed into Phase 1's list; path matching alone
   finds fewer (`/challenge-pr` on #76). Cost: a diff scan in
   `tools/lint_repo.py`, noisy on prose. **When to decide:** the card
   sessions only backfill, so they cannot show whether the rule is followed.
   Check at the first 3 corrections written after 2026-09-23 (Phase 10
   Stage 1 will produce them) whether each was routed.
4. ~~Is step 2 also step 0's weights?~~ No: answered by `/challenge-pr` on
   #76 from the 1b pilot's saved axes (step 1 bitwise equal to step 0, step 2
   differs).
5. **Grep every phase's results/ for unrecorded runs.** The 1b pilot sat on
   disk for five weeks with its status file saying nothing had been rerun.
   `p2b_pilot` and `p2d_pilot` are there too. Why: a card built only from its
   status file inherits the same blind spot. Cost: `ls` plus one manifest each.
   Changes: session 3's 2b and 2d cards; do it at the start of that session.
