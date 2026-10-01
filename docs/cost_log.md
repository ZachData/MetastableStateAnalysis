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
- 2026-09-25 Phase 1d revival (10.7M, about 3× the ~3.1M median): **two
  units in one session.** #97's cleanup and an investigation of what a cluster
  is (reading Phase 1, 1d and 10's cluster history, plus a scratch agreement
  read), then, on request, the 1d restore: two gate runs, a crashed first real
  run, a defect fix and a 7-minute re-run. Peak context ~200k on the later calls.
| 2026-09-24 | Phase review session 8: card 10; chunk 2 checked, still running (measured before the commit/PR calls) | 28 | 130k | 2.7M | 40k | #84 |
| 2026-09-24 | Card dependencies without hashes ("Open" 7), same transcript as #84 (measured before the commit/PR calls) | 47 | 163k | 5.4M | 49k | #85 |
| 2026-09-24 | Card deps: hash only the upstream Corrections section (option A, after #85's review; measured before the commit/PR calls) | 23 | 85k | 1.5M | 18k | #86 |
| 2026-09-24 | Phase review session 9: synthesis (measured before the commit/PR calls) | 49 | 165k | 5.7M | 45k | #87 |
| 2026-09-24 | Holdout guard, core/holdout.py (measured before the commit/PR calls) | 37 | 115k | 3.2M | 25k | #88 |
| 2026-09-24 | Stage 1 step 1: ext_sem_threshold sweep (measured before the commit/PR calls) | 35 | 128k | 3.1M | 27k | #89 |
| 2026-09-24 | Stage 1 step 2: token-composition table (measured before the commit/PR calls) | 41 | 136k | 3.9M | 30k | #90 |
| 2026-09-24 | Stage 1 step 2 parked checks: planted duplicates, count vs repeated types; chunk 3 launch (measured before the commit/PR calls) | 45 | 136k | 4.2M | 30k | #91 |
| 2026-09-24 | Stage 1 co-membership reader; step 1 re-run at 152 (measured before the commit/PR calls) | 22 | 123k | 1.9M | 29k | #92 |
| 2026-09-24 | §1.9 parked: lexical vs contextual (§1.10) (measured before the commit/PR calls) | 23 | 118k | 1.9M | 25k | #93 |
| 2026-09-25 | Stage 1 step 3: the pilot sweep (§1.11) (measured before the commit/PR calls) | 54 | 156k | 5.6M | 38k | #95 |
| 2026-09-25 | §1.9–§1.10 on the pilot sweep (§1.12) (measured before the commit/PR calls) | 34 | 124k | 3.0M | 32k | #96 |
| 2026-09-25 | F1/F12's 32–64 window on the pilot sweep (§1.13) (measured before the commit/PR calls) | 32 | 110k | 2.5M | 27k | #97 |
| 2026-09-25 | Phase 1d restored from its tag, first real run; Phase 10 on hold (measured before the commit/PR calls) | 83 | 197k | 10.7M | 55k | #98 |
| 2026-09-25 | Phase 1d literature scan: scale selection, attention-window and well definitions (`lit-1d.md`) (measured before the commit/PR calls) | 24 | 124k | 2.1M | 37k | #99 |
| 2026-09-25 | Phase 1d A0: singleton bound in the trivial filter, smoke re-run (measured before the commit/PR calls) | 27 | 96k | 2.0M | 23k | #100 |
| 2026-09-25 | Phase 1d float-noise drift, Stage 0 vs pilot, 72 layer-records (measured before the commit/PR calls) | 45 | 119k | 3.9M | 25k | #101 |
| 2026-09-25 | Phase 1d matched k and near-tie location on `repeated_tokens` (measured before the commit/PR calls) | 23 | 105k | 1.8M | 25k | #102 |
| 2026-09-26 | Phase 1d float64 cosine distances, Phase 10 §3 floor re-run on all 8 v1 prompts (measured before the commit/PR calls) | 40 | 117k | 3.2M | 29k | #103 |
| 2026-09-26 | Phase 1d merge tree over scales, cross-layer linking, F stage-gating fix, and its same-session revision after a second review (whole transcript; measured before the final commit) | 190 | 447k | 54.6M | 101k | #104 |

- 2026-09-26 Phase 1d merge tree (54.6M, about 15× the 3.65M running
  median): **one unit built twice in one session.** The first pass (24.8M
  at its PR) was a new module and tests, a worktree-vs-main-tree recovery
  and the F stage-gating defect. Its headline was then found to be an
  artefact (`LESSONS.md` lesson 6, 2026-09-26) and the unit was redone on
  the same transcript: new plateau rule, containment linking, tie fix,
  all 8 prompts re-run, write-up rewritten. Peak context 447k by the end.
  The revision should have been a fresh session with the review as its
  prompt (lesson 9).
| 2026-09-26 | 1d merge-tree link null + token 0 dropped, `/challenge-pr` answers, and the Phase 1d explainer page (whole transcript, before the final commit) | 65 | 207k | 8.1M | 205k | #105 |

- #105 is 2.1× the running median (3.8M). Two units shared one session.
  The null, its review answers, and a user-requested explainer page (an
  artifact, not in the repo: data extraction, design skills, one
  screenshot) all ran on the same context. The page should have been its
  own session (lesson 9).
| 2026-09-26 | 1d matched-covariance Gaussian null: three frames, calibration and dedupe runs (whole transcript, before challenge-pr) | 67 | 185k | 8.8M | 38k | #106 |

- #106 is 2.3× the running median (3.9M). The unit grew twice inside
  itself, both times on a confound of its own result: the calibration run
  (the null was off nominal level) and the dedupe run (token identity
  explained step 0). Three background runs were waited on across turns,
  each re-reading the context on return.
| 2026-09-27 | fleet onboarding: venv built and published to S3, HF cache pushed, `rvm.env` un-ignored, Lora objects deleted (whole transcript, before the commit) | 53 | 107k | 4.3M | 23k | #107 |
| 2026-09-28 | 1d item (3) finished: resumed the stopped run, reports, page, write-up, β recommendation (whole transcript, before challenge-pr) | 66 | 200k | 8.0M | 354k | #108 |

- #108 is 2.05× the running median (3.9M). Three parts: the report is 1 469
  lines per pair, condensed by hand into four scratch tables (most of the
  354k tool output); the page's first render found a layout defect, so the
  30 KB `index.html` was read whole and screenshotted twice; and the 7-hour
  run was waited on across turns, each return re-reading the context. A
  condensed-table mode in `attention_null_report.py` would have saved the
  first part.
| 2026-09-29 | FUTURE_IDEAS.md: literature scan, how training selects weights (docs only) | 23 | 147k | 2.6M | 33k | #109 |
| 2026-09-29 | 1d β refit: per-offset fixed effects + R² floor; #109 conflict resolution and merge; page read for explanation (whole transcript, before challenge-pr) | 49 | 165k | 5.3M | 44k | #110 |
| 2026-09-29 | #110 follow-up: CodeRabbit guards (collinearity, reproduction coverage) + page reading guide (whole transcript incl. #110, before challenge-pr) | 99 | 246k | 15.6M | 56k | #111 |
| 2026-09-29 | Triage of all phases, FUTURE_IDEAS tie-in, `docs/TRIAGE_2026-09.md` (docs only; before commit and challenge-pr) | 33 | 217k | 5.1M | 66k | #113 |
| 2026-09-29 | CI: deps tier on every push, status.sh without gh, pyproject-packages rule (whole transcript incl. the planning survey, before challenge-pr) | 71 | 199k | 9.9M | 38k | #114 |
| 2026-09-29 | P-I5 battery pin (whole transcript incl. #114 and its two review rounds, before challenge-pr) | 171 | 377k | 38.8M | 88k | #115 |
| 2026-09-30 | Battery-consumer audit + run_7 token check (after #115's merge wake, before challenge-pr) | 72 | 208k | 10.8M | 49k | #116 |
| 2026-09-30 | Mutation testing of core/evalues.py (whole transcript incl. #114–#116 and #116's CodeRabbit round, before challenge-pr) | 416 | 641k | 122.1M | 232k | #117 |

- #117's row is cumulative over the #114/#115 session, which kept being
  resumed: #116's CodeRabbit round and this unit both ran at 400–640k per
  call. Its share is roughly 122.1 − 38.8 (#115's row) − 10.8 (#116's row)
  ≈ 72M, ~17× the median, the worst in this log. Every call re-read two
  merged PRs' worth of context. Lesson 9, not optional: the next unit is a
  fresh session, started from the handoff prompt in #117's description.
- #116 is 2.5× the running median (4.3M). This was a fresh session after
  `/clear`, so none of it is carried context. The cost was the audit
  itself: slices of ~20 consumer files, STATE.md in full, and the Phase 7
  tests, each re-read by every later call (~150k average over 72 calls).
  An audit over 30 files could go to one `Explore` subagent that returns
  only the table.
- #115's row is cumulative: #114's review rounds (CodeRabbit, /challenge-pr,
  a pinned-env venv) and this unit both ran in #114's session, so per-call
  context reached 377k. This unit's share is ~28.9M, ~7× the median. Lesson
  9 at its clearest: the next unit starts a fresh session.
- #114 is 2.4× the running median (4.05M): one session held the user's
  open planning question (a survey of CI, tests, registry and lessons, with
  test runs) and then the implementation. Two units' reading in one context.
- #111's row is cumulative: it includes #110's 5.3M, so this unit's share is
  ~10.3M, ~2.6× the running median. The user continued in #110's session
  instead of clearing (their call, 2026-09-29), so every call re-read ~200k
  of #110's context, including the page's 30 KB source and a verbose
  explanation of it. Lesson 9 again: the cost is the carried context.
| 2026-09-29 | 1d long prompts, part 1: rule + 4 texts (whole transcript incl. #110, #111; in progress) | 180 | 360k | 40.4M | 79k | branch |

- Cumulative again: this part's share is 40.4M − 15.6M ≈ 24.8M, ~6× the
  running median, for a rule and four texts. The session carried #110 and
  #111 (peak context 360k), and each source fetch, tokenizer check and
  permission-classifier outage was a call re-reading all of it. Lesson 9:
  the unit should have started in a fresh session, as the user then chose.
| 2026-09-29 | 1d long prompts, part 2: extractor, 8 runs, prefix, merge tree, deduped Gaussian null, β speed-up | 153 | 254k | 26.2M | 64k | #118 |
| 2026-09-30 | 1d long prompts, part 3: all-token Gaussian null + β at length, reports, Stop, /challenge-pr fixes | 108 | 212k | 14.1M | 49k | #118 |

- Part 2: 26.2M, ~5.6× the running median (4.7M). One session held the
  extractor, 8 forward passes, the prefix check, the merge tree and its null,
  the deduped Gaussian null, and two defect fixes found mid-run (β's O(n²)
  estimator, the all-or-nothing drivers), with context growing to 254k and
  re-read on each of 153 calls. Each defect was its own unit (lesson 9).
- Part 3: 14.1M, ~3× the median. 65 calls (6.3M) to the PR; answering
  `/challenge-pr` (a new offset-matched β fit, the resume defect, a solver
  check) took 43 more at ~200k context each. Continuing in the same session
  kept the PR's reasoning in context but paid for re-reading it.

| 2026-09-30 | Answers to /challenge-pr on #117: 397/413 killed, reasons probed, accept list keyed on context (measured before #119's own challenge round) | 59 | 246k | 10.5M | 53k | #119 |

- #119 is 2.2× the running median. A fresh session after `/clear`, so none
  of it is carried. The first call was 63k; context grew to 250k because
  #117's five findings needed `core/evalues.py`, both test files, the
  adjudication replay and the accept list read before any edit, and the
  probe round's output (17 mutants) stayed in context for every later
  call. The probe output could have gone to a file with only the DIFF
  lines printed.

| 2026-09-30 | 1d vote rules: 9 rules x 48 records, scipy invalid-tree fix; also #119/#113 conflict resolution (whole transcript, before challenge-pr) | 108 | 224k | 16.5M | 55k | #120 |

- #120 is ~3.8× the running median. A fresh session after `/clear`, but it
  held two units: making #119 and #113 mergeable (two gate runs) and then
  the vote rules. The batch crashed halfway on a scipy defect, so the
  diagnosis (wrapper, re-run, matrix encoding, regression test) ran at
  ~150–220k per call. The PR fixes belonged in their own session, and the
  batch's failure would have been one call cheaper had `_job` named its
  input from the start.
