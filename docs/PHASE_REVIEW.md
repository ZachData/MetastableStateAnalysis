# Phase review — handoff

**Opened:** 2026-09-23 · **Owner:** the user + Claude · **Runs beside:** Phase 10
Stage 0 (compute on the local box; this thread is docs-only).

## Why

Seventeen phase directories plus two unmerged branches (`INDEX.md` "Off `main`").
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
| Do cards go stale when a phase they read changes? ("Open" 7, settled 2026-09-24, user) | **No, for every card, not only archived ones.** Depends on lists bare phase ids; a card goes stale only on its own status file. Why: the dependency hashes forced re-stamps that changed nothing (card 9 after card 10, session 8; five frozen cards on each `status-1.md` edit, session 4) and caught none of Phase 1's 7 real corrections, which arrived through routing (`LESSONS.md` lesson 1). Cost accepted: a change to phase A that nobody routes to A's readers no longer flags them |

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
6. ~~The Phase 1d and viz branches: keep, revive or drop?~~ **Settled
   2026-09-24 (user): let the code go, keep the intent.** Tooling has moved on,
   so a revival would rewrite anyway. 1d's design, status, findings and
   `P-C1`–`P-C4` text are now in `archive/p1d_cluster_ensemble/` (`FROZEN.md`
   says when to rebuild: when Phase 10 needs a partition it can trust, after
   first measuring whether tuning reduces the run-to-run drift of
   `p10_cluster_function/status-10.md` §3, which is not the noise 1d tuned
   against). The viz tool's purpose is in `INDEX.md` "Deleted code whose intent
   is kept". Pushing the local `dead/*` tags is optional. The deletion is
   `LESSONS.md` lesson 12.
7. ~~Should archived cards go stale when a live phase they read changes?~~
   **Settled 2026-09-24 (user), wider than proposed:** no card goes stale on
   a phase it reads, archived or live. See "Decisions".

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
| 3a | Parked 5 for 2 / 2b / 2d; cards 2, 2b. Found three unrecorded runs: the 2b pilot (run 2026-08-14 and rerun 2026-08-17, recorded in `status-2b.md`), the 2d pilot (same two dates; it computed the `P-M1` / `P-T1` statistics on real artifacts; recorded in `status-2d.md`, values not opened, for the user) and the 243-run Phase 2 sweep (2026-08-13) that holds Study B's numbers, whose decompose columns contradict "Degenerate columns" (`status-2.md` "Phase 2 runs on disk") | **done** 2026-09-23 |
| 3b | Cards 2d, 6. The user decided the 2d pilot (unseen, quarantined, fresh scoring run only). Found that two of the 2026-09-19 audit's four "blocking decisions" were already taken: `P-T1`'s amendment landed 2026-08-11 and the P6 unit was registered `model` 2026-08-25 (routed to both `## Corrections received`). 1b's Feeds now names `6-frozen`; live 6 reads 1, 2, 2b | **done** 2026-09-23 |
| 4 | Cards: archived 3, 4, 5, 5b, 5c and frozen 6; live 6 now depends on frozen 6; 1b's advice-only Feeds (4, 5, 5c, 6-frozen) removed. Found: Phase 3's only runs on disk reproduce 1 of its 7 verdict rows (bimodality), so the headline null's producer is not on this box (`archive/p3_crosscoder/status-3.md` "Runs on disk"); nothing for 4, 5, 5b, 5c or frozen 6 is on disk. Routed 12 corrections (5c: the flip is mostly causal mask on 410m). Fixed INDEX/README's stale "two live explanations" and branch lines. The branches went to the user as "Open" 6, archived-card staleness as "Open" 7 | **done** 2026-09-24 |
| 5 | Cards: 7, 7d, 7e. 7 depends on 1, 2 (1's Feeds gained 7; 1b's advice-only edge to 7 dropped, as in session 4); 7e on 7d. Found: `status-7.md` still said nothing had run, while all 19 `data/phase7/` tables (410m, v1 battery) and `P-I1`'s score existed; fixed in place and routed. `P-I5`'s real run was on 70m `L3H6`, not 410m. Routed 15 corrections, most from §3.15 / §3.17 / §3.26 and `status-8.md`'s `mean` reruns (410m holds under `mean` for the catalogue, `r*`, matrix, formation, geometry; FV, self-repair and MLP 6 were not rerun). New: Parked 8, 9 | **done** 2026-09-24 |
| 6 | Card: 8, depending on 7d, 7e; feeding 7 (`P-I5`'s target came from `status-8.md`, so 7 now depends on 8) and 10. Each of the 30 Phase 8 outputs on disk (`data/analysis/`, 2026-09-11 to 09-13) is a run `status-8.md` describes; nothing on disk measures 1b or 1.4b on the induction axis. Found: `P-I5`'s target, 70m `L3H6`, was chosen as "the `L7H8` analogue" (§3.32) three days after `status-8.md` found 70m has no relay-fed matcher (routed to `status-7.md`; `LESSONS.md` lesson 1). Fixed in place: "Reproducing" undercounted the runners that take `--probe`; the "Literature scan" section now points to its two successors; README's Phase 8 row. Routed 3 corrections (§3.16, `lit-8.md` §3) | **done** 2026-09-24 |
| 7 | Card: 9, in a new `p9_metric_intervention/status-9.md` (card, `plan-9.md` §8's ladder state, corrections received), depending on 1c, 2, 7d, 7e, 8, 10 and feeding 10; those five gained 9 in Feeds (1c's "no status file" note dropped). A status file makes a phase: `tools/render_experiments.py` needed a no-predictions entry for 9. Nothing has run under Phase 9's name and nothing on disk is its. Found: E5 was stale when written (unit decided 2026-08-25), E0 ran as Phase 10's F1 ten hours later and was never passed back (`LESSONS.md` lesson 1), E4's "measured" ratio is a planted-construction chance ratio, E3's force-collapse / force-disperse arms reached no phase, and `notes-9.md` §7 and `plan-9.md` §4.7 still called the γ-patch read-side after §4.1's correction. Pointers added in place; 7 corrections routed. `/challenge-pr` on #83 corrected three of these | **done** 2026-09-24 |
| 8 | Card: 10 | open |
| 9 | Synthesis: the duplicate-question map, the ranked after-Phase-10 list, the e-value plan below | open |

Before the first Stage 1 read: **the holdout guard** (Parked item 1). It is not
a review session but it is due before Stage 1 starts, not after session 9.

## The e-value plan (session 9, or its own session)

The machinery exists: `core/evalues.py` (p→e calibrator, e-process; `average`
for dependent units, since a product over-rejected, `LESSONS.md` lesson 6),
`core/adjudication.py` (ledger), 39 registered, 11 adjudicable, 0 adjudicated
(`claims/EXPERIMENTS.md`). What is missing is **power and at most two decisions**, not code.

1. **Attainable-E table at 8, 12 and 20 prompts.** With κ = 0.5 one prediction
   alone needs p ≤ 1/1600 for E ≥ 20. `CLAIM-C`'s floor was 0.0661 at 8
   prompts (E ≈ 1.9) and is 0.0002 at 20 (§3.46). 12 is the confirmation-set
   size: a CLAIM-C-shaped design there needs ≥ 11 informative prompts, a
   margin of one. Per adjudicable row, via `max_attainable_average_E`; rows
   that cannot reach 20 say so before anyone runs them.
2. **The remaining blocking decisions**, one short memo each for the user,
   after checking each against the code first (Parked 7): β's scale
   convention and `P-S1`'s matched-k clustering. Plus `P-I5`'s battery
   (`STATE.md` Blocked 2) and Parked 6. The audit's other two, `P-T1`'s
   wording and the P6 unit, had been decided in August (session 3b).
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
   **Found and fixed 2026-09-24:** 47 runners defaulted `METS_REPO` to the
   main tree and ran `sys.path.insert(0, REPO)` at import. Collecting
   `tests/test_backfill_hdbscan.py` put the main tree first, so **every
   worktree gate imported main's copy of each package not yet loaded**. Now
   each defaults to its own checkout; `tests/test_no_hardcoded_repo.py` pins
   it. Past worktree gates tested a mix of branch and main code; a PR merged
   green on one may not have been green on its own code. Rerunning old gates
   is not planned: main has the fix now, and CI (a single checkout) was never
   affected.
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
   **Done for 2 / 2b / 2d (session 3a, 2026-09-23)**: all three had unrecorded
   runs (row 3a). Main tree `results/` also holds two empty dirs
   (`2026-08-12_05-01-35` and `p2_eigenspectra_2026-08-13_05-13-52` read empty
   only without following their symlinks to HDD_1TB) and `phase3`. The later
   card sessions still check their own phase's dirs.
6. **What does `P6-R4`'s unit `model` mean on Pythia?** Validity is not the
   question: `r2_r4_null.py`'s recorded table has `model` at nominal across
   the whole dependence range, independent layers (ρ = 0) included, so it is
   conservative for Pythia's untied layers too (`/challenge-pr` on #78).
   Still open: `model` draws one set of subspaces shared across layers, and
   Pythia's per-layer channel dimensions differ, so what "the same draw" means
   there is unstated. Cost: read how `unit="model"` seeds per layer (line
   ~218), then a memo. It must be settled before any P6 run. Changes: e-value
   plan item 2.
7. **Check the audit's remaining "decisions" against the code before writing
   memos.** Two of its four (`P-T1` wording, P6 unit) had been settled in
   August; the audit read registry `notes` and prose, not code. β's scale
   convention and `P-S1`'s matched-k may be the same. Cost: a grep each.
   Changes: e-value plan item 2. Also for the user: both `P6-R2`/`R4` registry
   `notes` still say no unit is registered, and quarantining
   `results/p2d_pilot` could join Parked 1's guard list.
8. **Does any Phase 10 registration read the induction axis on 410m?**
   `p8_scale_ladder/design-8.md` marks `pythia-410m` spent on the induction
   axis (7d/7e; rung policy = `check_registry` rule 3 applied forward), and
   the Phase 10 holdout plans registered predictions scored on 410m. Cluster
   quantities are off that axis, so the two agree, unless a registration uses
   7d's 384-head causal sweep, which `attention-10.md` A3 joins to. Why: such a
   registration would be scored on an axis the policy calls spent. Cost: one
   line in `handoff-10.md` §0.4 when registrations are drafted. Changes: which
   Phase 10 rows may be registered on 410m.
9. **`claims/FALSIFICATION.md` shows `P-I5` as "null not yet constructed".**
   Its generator writes that for every needs-null row; `P-I5`'s null is built,
   calibrated and run, and fails for another reason (§3.33–§3.34). Why: the
   ledger understates where `P-I5` is. Cost: a branch in
   `tools/render_falsification.py` for rows with a gate. Changes: nothing
   decided; e-value plan item 2 reads this ledger.
