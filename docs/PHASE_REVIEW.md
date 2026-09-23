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
3. **Stale maps.** `INDEX.md` lines 9–163 are a "Current priority" block
   dated 2026-09-20 that duplicates `STATE.md`.

## Decisions (user, 2026-09-23)

| decision | choice |
|---|---|
| Where a phase review lands | A **card** at the top of its `status-N.md`; a phase table in `docs/` generated from the cards (built in session 1); a lint keeps them current. This is the "next PR" decided 2026-09-22 (`STATE.md`) |
| The 12 v2 prompts never run through Phase 1 | **Held out** as Phase 10's confirmation set. Stage 0 still runs all 20; nothing reads the 12 until registrations are frozen (`p10_cluster_function/handoff-10.md` §0.4) |
| Order vs Stage 0 | In parallel |

## The card (fixed fields, one line each, pointers not numbers)

| field | content |
|---|---|
| Question | the question the phase asked, in one sentence, in the particle/OT vocabulary |
| Inputs | model(s), checkpoints, battery hash, run dirs; "none" if never run |
| Results | one line per result, each with its pointer (`§`, file, audit JSON) |
| Superseded / wrong | what later work corrected, with the pointer |
| Registry | prediction ids and their state, or "none, because …" |
| Depends on / feeds | phase ids |
| Open threads | unanswered questions the phase itself raised |
| After Phase 10 | candidate experiments, each with cost (forward pass or free) |

Numbers stay where they already live ("write once"); the card points.

## Sessions (one unit each; `CLAUDE.md` "one session per unit")

| # | unit | state |
|---|---|---|
| 1 | Card template in `docs/`, generator `tools/render_phases.py` → the phase table, lint rule (card present, fields filled, pointers resolve, card newer than the phase's last result commit); pilot on Phase 1; delete `INDEX.md`'s stale priority block | open |
| 2 | Cards: 1b, 1c | open |
| 3 | Cards: 2, 2b, 2d | open |
| 4 | Cards: archived 3, 4, 5, 5b, 5c, 6; the Phase 1d and viz branches → keep / revive / drop | open |
| 5 | Cards: 7, 7d, 7e | open |
| 6 | Card: 8 | open |
| 7 | Card: 9, checked against 10 (`notes-10.md` §9) | open |
| 8 | Synthesis: the duplicate-question map, the ranked after-Phase-10 list, the e-value plan below | open |

Before the first Stage 1 read: **the holdout guard** (Parked item 1). It is not
a review session but it is due before Stage 1 starts, not after session 8.

## The e-value plan (session 8, or its own session)

The machinery exists: `core/evalues.py` (p→e calibrator, e-process; `average`
for dependent units, since a product over-rejected, `LESSONS.md` lesson 6),
`core/adjudication.py` (ledger), 39 registered, 11 adjudicable, 0 adjudicated
(`claims/EXPERIMENTS.md`). What is missing is **power and four decisions**, not code.

1. **Attainable-E table at 20 prompts.** With κ = 0.5 one prediction alone needs
   p ≤ 1/1600 for E ≥ 20; `CLAIM-C`'s floor at 8 prompts (0.0661) caps it near
   E ≈ 1.9. Per adjudicable row: its floor at 8 and at 20, via
   `max_attainable_average_E`. Rows that cannot reach 20 even at 20 prompts
   say so before anyone runs them.
2. **The four blocking decisions**, one short memo each for the user: β's
   scale convention, `P-T1`'s wording, `P6-R2`/`R4`'s exchangeable unit,
   `P-S1`'s matched-k clustering. Plus `P-I5`'s battery (`STATE.md` Blocked 2).
3. **Phase 10's route into the registry.** Which tier-1 findings graduate,
   registered before anything reads the 12 held-out prompts. F14 is already
   named the one to register (`handoff-10.md` "Standing constraints").
4. **Native e-values for new registrations** (testing by betting, likelihood
   ratio) instead of calibrating a p-value: usually more power, and they fit
   accruing prompts one at a time. Literature scan first (trigger 2 in
   `CLAUDE.md`); existing rows keep the calibrator. A proposal, not a decision.

## Parked

1. **Holdout guard in code.** A list of held-out prompt keys in `core/` that
   Phase 10 readers refuse unless passed an explicit flag. Why: a rule only in
   docs has been missed before (`LESSONS.md`). Cost: small, `core/` + tests.
   Changes: whether the confirmation set survives Stage 1.
