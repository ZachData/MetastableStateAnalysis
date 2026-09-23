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
