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

## Over 2× median, and why

- 2026-09-22 machine setup: the baseline; one long session covering setup, a
  literature scan and two builds. Why "one session per unit of work" exists.
