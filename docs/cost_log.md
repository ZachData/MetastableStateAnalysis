# Cost log: one row per unit of work

Appended by the Stop protocol (`CLAUDE.md` step 7):
`python tools/session_cost.py ~/.claude/projects/<project>/<session>.jsonl --row "<unit>" --pr "#N"`.

- **calls**: model API calls (distinct message ids).
- **peak ctx**: the largest single-call context (input + cache read + cache creation).
- **total ctx**: that summed over calls. This is the number the 2× rule reads.
- **tool out**: tool-result size, chars ÷ 4.

**Rule (`LESSONS.md` lesson 9):** a row whose total ctx is more than 2× the
running median of the rows above it gets a line under the table saying why.

| date | unit | calls | peak ctx | total ctx | tool out | PR |
|---|---|---|---|---|---|---|
| 2026-09-22 | Machine setup + doc indexes + large-read guard (baseline, measured by hand) | 147 | 214k | 19.0M | 34k | #65 |
| 2026-09-22 | This tool, cost log, `scripts/status.sh`, CLAUDE.md lines (measured before the commit/PR calls) | 9 | 58k | 428k | 7k | #66 |
| 2026-09-22 | Stage 0 chunk driver, option B (measured before the commit/PR calls) | 30 | 106k | 2.2M | 23k | #67 |
| 2026-09-22 | Archive batch + `cited-md-path` lint (measured before the commit/PR calls) | 63 | 188k | 7.9M | 40k | #69 |
| 2026-09-23 | Stage 0 chunk 1: found running, first-output check (measured before the commit/PR calls) | 24 | 75k | 1.4M | 9k | #73 |
| 2026-09-23 | Phase-review plan + 12-prompt holdout (measured before the commit/PR calls) | 23 | 93k | 1.7M | 20k | #74 |
| 2026-09-23 | Phase review session 1: card template, `render_phases.py`, lint `phase-card`, Phase 1 card (measured before the commit/PR calls) | 38 | 170k | 4.8M | 45k | #75 |
| 2026-09-23 | Phase review session 2: cards 1b, 1c, correction routing, unrecorded 1b run (measured before the challenge-pr calls) | 50 | 154k | 5.2M | 35k | #76 |
| 2026-09-23 | Phase review session 3a: Parked 5 for 2 / 2b / 2d, cards 2, 2b (measured before the commit/PR calls) | 46 | 157k | 4.7M | 41k | #77 |
| 2026-09-23 | Phase review session 3b: cards 2d, 6; 2d pilot decided; branch cleanup (measured before the commit/PR calls) | 42 | 147k | 3.7M | 37k | #78 |
| 2026-09-24 | `run_2d` measurement only + worktree-gate `sys.path` fix; chunk 2 launched (whole transcript, includes #78's later calls; measured before the commit/PR calls) | 116 | 280k | 20.0M | 69k | #79 |
| 2026-09-24 | Phase review session 4: cards 3, 4, 5, 5b, 5c, 6-frozen; the 1d/viz branches (measured before the commit/PR calls) | 57 | 181k | 7.2M | 47k | #80 |
| 2026-09-24 | Phase review session 5: cards 7, 7d, 7e (measured before the commit/PR calls) | 41 | 171k | 4.9M | 48k | #81 |
| 2026-09-24 | Phase review session 6: card 8; chunk 2 checked, still running (measured before the commit/PR calls) | 32 | 148k | 3.6M | 46k | #82 |
| 2026-09-24 | Phase review session 7: card 9; chunk 2 checked, still running (measured before the commit/PR calls) | 45 | 154k | 5.1M | 46k | #83 |

## Over 2× median, and why

- 2026-09-22 machine setup: the baseline; one long session covering setup, a
  literature scan and two builds. Why "one session per unit of work" exists.
- 2026-09-22 archive batch (7.9M, 3.6× the 2.2M median): one unit, but wide.
  129 citations across 76 files, 95 pre-existing dangling citations to triage
  for the new lint, a gate failure (the rewrite had touched three hash-pinned
  gate files) and two full gate runs. Context re-read on each of 63 calls,
  not tool output (40k), is the cost, as lesson 9 predicts.
- 2026-09-23 phase review session 1 (4.8M, 2.5× the 1.95M median): the unit
  asked for one card, which means reading all of `status-1.md` (725 lines,
  ~25k tokens) plus the INDEX block being deleted, and that stays in context
  for every later call. Cards 1b + 1c in one session will cost about the same.
- 2026-09-23 phase review session 2 (5.2M, 2.4× the 2.2M median): as
  predicted above, plus a rule decision taken mid-session ("Open" 5) and an
  unrecorded 1b run found and read. Two cards and a rule change per session
  is about the ceiling; session 3 has four cards and should be split.
- 2026-09-23 phase review session 3a (4.7M, about 2× the median): split as
  planned (2 cards), but Parked 5 found three unrecorded runs, each needing
  its own reads (JSON structure, provenance, a second disk). The ceiling holds:
  two cards plus a disk check per session.
- 2026-09-24 `run_2d` + gate fix (20.0M, about 5× the median): **not one
  unit.** The session was never cleared after #78 (its review answers, the
  branch cleanup), then took three more units at once on request: launch
  chunk 2, fix `run_2d.py`, and the worktree-gate defect it exposed, which
  took four gate runs and a bisect. Peak context 280k on each later call is
  the cost. The row covers the whole transcript, so #78's row is partly
  double-counted here.
| 2026-09-24 | Phase review session 8: card 10; chunk 2 checked, still running (measured before the commit/PR calls) | 28 | 130k | 2.7M | 40k | #84 |
| 2026-09-24 | Card dependencies without hashes ("Open" 7), same transcript as #84 (measured before the commit/PR calls) | 47 | 163k | 5.4M | 49k | #85 |
| 2026-09-24 | Card deps: hash only the upstream Corrections section (option A, after #85's review; measured before the commit/PR calls) | 23 | 85k | 1.5M | 18k | #86 |
| 2026-09-24 | Phase review session 9: synthesis (measured before the commit/PR calls) | 49 | 165k | 5.7M | 45k | #87 |
| 2026-09-24 | Holdout guard, core/holdout.py (measured before the commit/PR calls) | 37 | 115k | 3.2M | 25k | #88 |
| 2026-09-24 | Stage 1 step 1: ext_sem_threshold sweep (measured before the commit/PR calls) | 35 | 128k | 3.1M | 27k | #89 |
| 2026-09-24 | Stage 1 step 2: token-composition table (measured before the commit/PR calls) | 41 | 136k | 3.9M | 30k | #90 |
