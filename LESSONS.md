# LESSONS — what we got wrong, and what we do differently now

Written for a human to read top to bottom. Each lesson is a **pattern** that
cost us more than once, with the concrete instances, what it cost, and the rule
it produced. The last column in each says **where the rule lives** — a rule that
lives only in prose gets forgotten, which is lesson 1.

Add to this file when something goes wrong (Claude does this as part of the
stop protocol in `CLAUDE.md`). Newest instances go at the top of their lesson.
If a new mistake fits no pattern, start a new numbered lesson.

**Status key:** ✅ enforced by a tool (lint/test/hook) · 📋 written in `CLAUDE.md` ·
⚠️ known, not yet enforced.

---

## 1. Handoff documents go stale, and the next session trusts them

**What happens.** A handoff says "X is open / green / doesn't exist", the
world moves, nobody updates the line, and the next session acts on it.

**Instances.**
- 2026-10-29 (Track T, T1): #282 flagged a mismatch on Dodecahedron's card (40B tokens against
  9536 × 524k = 5.0B), and still built T2's token range (~1024–2048) on 524k; `/challenge-pr` on
  #282 then made "the three ranges do not overlap" a reason for T2 and merged it. The card's
  524k is per GPU ("64 micro × 4 accum × 2048", 8 × H100): 8 × 524k × 9536 = 40.0B. T1's first
  fix then claimed "four disjoint ranges, no decision waits" from seed 0's bracket alone;
  `/challenge-pr` on #283 re-derived on A5's nine-seed (128, 512], where the tokens window
  overlaps both others (`trackT_timaeus/status-T.md` "T1, the inventory", Q3). No run was spent.
  Rule: a number a document has itself marked as conflicting is not used to justify a unit until
  the conflict is resolved or both readings are carried, and a mapped window starts from the
  registered spread, not one seed.
- 2026-10-22 (C2 unit 4): a range typed by hand after `/challenge-pr` on #256, "the centred count
  at the break is 0.17–0.50 of its own plateau", went into `status-C.md` (twice), `PROJECT.md`
  §3.67 and the STATE line, and was reviewed and merged. Computing it from the record for the
  draft gave 0.07–0.50: seed 4's B-matched arm (2 of 28) had been left out. No reading moved
  ("below half on every seed but the tie" holds). Caught only because the draft renders every
  number from its record; a post-hoc number typed into a status table had no such check.
- 2026-10-22 (C2 unit 3c): `lit-C2-unit1.md` §5 recorded GPT-2 small's warmup as "4000 steps,
  linear" from the trainer config ([R]), and §5's prediction table was built on it. The logged LR
  in every seed's `trainer_state.json` is logarithmic (DeepSpeed's default warmup type; half the
  peak by step 100). The config names the warmup's length, not its shape. Caught before the
  freeze, because the design re-derived the predictions from the log; it moved the dose's E₁
  band from (303, 605] to (29, 119]. In the same session the start hook printed the main tree's
  `STATE.md` (2026-10-20) while `origin/main` was at #257 (2026-10-22). Rule: **a schedule an
  analysis maps through is read from the run's own log, not its config**.
- 2026-10-22 (C2 unit 3s): unit 3's Parked 2 and Blocked 96 (b) costed the seeds unit as "a
  re-download of about 160 checkpoints (the seed checkpoints are no longer in the HF cache)".
  All 162 (9 seeds × 18 steps) are in `HDD_1TB/mets_data/hf`, where A5 put them and where
  `STATE.md`'s disk line says they are. "No longer in the HF cache" was true only of the default
  cache. The user chose (b) on that cost; it was a cheaper option than written, so the choice
  stands. In the same session the start hook printed the main tree's `STATE.md` at #240 while
  `origin/main` was at #255, the third time. Rule: **a claim that an input is absent names the
  place it looked, and an option's cost is checked against `STATE.md`'s disk line first**.
- 2026-10-19 (A3 opened): Claude put "A3 first, and start the 9-seed download now" to the user as
  the recommended answer to #231's finding 4, from the reviewer's wording alone; the user took
  it. The decided `design-A5.md` (step 3, and its line 19) registers *before* the seeds are
  downloaded, which is what makes A5 a held-out test. Caught one call later by reading the
  design before launching; the download half was dropped and said so, with a second reason
  that was wrong too ("131 GB would not fit, 50 GB free": that is WDS_500; A5's target
  `HDD_1TB` had 265 GB, as `design-A5.md` itself says; `/challenge-pr` on #232, finding 6). Again, the start hook printed `STATE.md` at #211 with `origin/main` at #231.
  Rule: an option offered to the user is checked against the decided text it touches first.
- 2026-10-19 (B3 unit 2): again, the start hook printed the main tree's `STATE.md` at #211 while
  `origin/main` was at #225 (14 PRs on): the main tree is never pulled, and the hook reads it.
  Caught by Start step 1's check; read `STATE.md` from the fresh branch instead. Fix not made
  (the hook could print `git show origin/main:STATE.md`); parked here as the cheapest one.
  2026-10-23 (C5 unit 0): again, at #239 with `origin/main` at #262 (124 commits behind).
  2026-10-25 (B7 unit 0): again, at #240 with `origin/main` at #270 (170 commits behind); the
  hook's output was 50 KB, so even the stale copy was read from a file, not from context.
  2026-10-25 (B7 unit 1b): again, at #240 with `origin/main` at #272.
- 2026-10-10 (roadmap session): in conversation Claude proposed registering a confirmatory test
  on 410m's held-out prompts. The rung policy (`p8_scale_ladder/design-8.md`, quoted on the 7d
  and 7e cards) keeps 410m exploration-only, while `status-10.md`'s card holds 12 prompts out on
  410m as a confirmation set (undecided, `docs/PHASE_REVIEW.md` "Open" 1–4): the docs are in
  tension, and the proposal followed neither. Caught by reading the cards before writing the
  plan; the roadmap's first draft then pinned the tension on `handoff-10.md`, which agrees with
  the policy (`/challenge-pr` on #185, finding 3; `docs/ROADMAP.md` §10.3). Rule: before
  proposing a registration, read the rung policy, the phase card and its standing constraints.
- 2026-10-10 (Stage 3 closed): `STATE.md` Blocked 36 (d) offered "take Stage 2 or Stage 4"
  from `handoff-10.md` alone, while the open roadmap #185 (same day) moves Stage 4 to Track A
  and never mentions Stage 2. The user picked (d) on that text. Caught at the next unit's
  start, by reading #185 before choosing. Rule: an option in Blocked names any open PR
  that re-homes or drops it. Also, again: the start hook's `STATE.md` was 15 commits old.
- 2026-10-10 (OV1s): #182 merged with `STATE.md` "Open PRs" still calling
  itself open and listing worktrees and branches that were already gone. The
  correct line was in `../Mets-work` as an **uncommitted** edit made after the
  PR's last commit (12:30, merge 12:34), so the merge never got it, and the
  worktree could not be removed without losing it. Also, again: the printed
  `STATE.md` was the main tree's, one merge behind. Rule: a Stop-protocol
  doc edit made after the push is pushed before the merge, or the PR is
  not ready; `git status` in the task worktree is part of "done".
- 2026-09-25 (Stage 1 step 3): `handoff-10.md` §1.1's table, the number that
  launched Stage 1, said "mean over 8 prompts". It was 9: the pilot has
  `short_heterogeneous`, which Stage 0 does not. Nothing broke, because §1.6
  re-derived from source; found only when a 9-prompt pilot run reproduced
  every value. Also: this session's printed `STATE.md` was the main tree's,
  12 merges behind `origin/main` (the fast-forward is refused to Claude), and
  the handoff header still said "launch chunk 3" a day after Stage 0
  finished. Rule: Start step 1's "older than `origin/main`" check reads
  `git show origin/main:STATE.md`, not the printed copy; a first-look table
  names its input set (Name the input).
- 2026-09-24 (phase review session 8): `handoff-10.md`, the active thread's
  runbook, still said "relaunch chunk 1" two days and one chunk later, and
  its Stage 0 done-line still counted option A's 228 directories after
  option B (380 runs) was decided in §0.3 of the same file. `STATE.md` had
  been kept current each time; the handoff header had not. Rule: Stop step 3
  covers the handoff's header and done-lines, not only its body.
- 2026-09-23 (phase review session 2): the card staleness lint (#75) watched
  a card's own status file and the phases it **depends on**, and those are
  upstream. Of the 7 corrections on Phase 1's card, 3 came from phases
  Phase 1 **feeds** (`status-1c.md`, `status-10.md`, `math-10.md`) and 1 from
  `PROJECT.md` §3.51. None of them touched `status-1.md`, so the lint could
  not see them. Hashing the sections the pointers name would have caught
  none either: that watches only sections a card already cites, and each
  correction arrived in a section nothing cited yet.
  `/challenge-pr` on #75 flagged the gap, and counting Phase 1's list settled
  it. The rule was designed by argument; the card's own data answered it.
  Fix: a correction to an earlier phase adds one line to that phase's
  `## Corrections received`, so the existing hash sees it.
- 2026-09-23 (same session): `status-1b.md` said "Nothing has been rerun"
  for five weeks while a populated post-revision Pythia run (243 runs,
  2026-08-17) sat in the main tree's `results/p1b_pilot`. The only mention was
  a disk-inventory line in `PROJECT.md`. Found by `find` while filling the
  card's Inputs field. Rule: a card session checks `results/` and `data/` for
  the phase's runs, not just its status file (`docs/PHASE_REVIEW.md` workflow).
- 2026-09-23 (phase review session 3a): the same batch (run 2026-08-14,
  rerun 2026-08-17 over the same dirs) also ran Phase 2b and Phase 2d on
  Pythia, and neither status file said so. `status-2d.md` said `P-T1` /
  `P-M1` were "not run against real artifacts", and the 2026-09-19 audit
  declined to compute their statistics to avoid a peek, while 54 files on
  disk already held those statistics in the LN frame. An audit that reads the
  docs and the registry cannot see a run nobody wrote down; `ls results/` can.
- 2026-09-24 (phase review session 5): the same pattern a fourth time, the
  other way round. `status-7.md` said "still nothing has executed against any
  model" and listed `P-I1` as "not run" for three weeks after all 19
  `data/phase7/` tables were written and `P-I1` scored on them. The runs *were*
  recorded, in `PROJECT.md` §2, §3.6 and the provenance audit; the status file
  was never told. Its header even contradicted its own audit section below it.
  Same fix: the phase's `## Corrections received`, and `ls data/phase<N>/`.
- 2026-09-23 (phase review session 3b): the 2026-09-19 e-value audit listed
  four "blocking decisions nobody has taken". Two had been taken in August:
  `P-T1`'s amendment landed 2026-08-11 with the code (`PREDICTIONS.md`
  addendum, registry statement), and the P6 unit was registered `model`
  2026-08-25 (`r2_r4_null.py`, registry `null_construction`). The audit read
  each entry's stale `notes` field and its own phase prose. The error then
  spread to README, `EVALUABILITY.md`, `INDEX.md`, `docs/PHASE_REVIEW.md` and
  STATE Blocked 5, which asked the user whether "the amendment can still
  land". Found while filling the cards' Registry field against the code. Rule:
  an "X is undecided" claim is checked against the code constant and the
  registry's structured field, not against a notes field.
- 2026-09-24 (phase review session 7): `plan-9.md` (2026-09-20) listed E5,
  `P6-R4`'s unit, as a decision for a human; it had been taken 2026-08-25.
  Its E0 (transport) ran as Phase 10's F1 about ten hours after the plan was
  written, and nobody passed that back, so the plan said "unrun" for four
  days. Same rule, plus: a result that answers another phase's ladder row
  gets a pointer in that row. (`/challenge-pr` on #83 corrected my first
  count here, which also charged the plan with a 5c claim that was right
  when written.)
- 2026-09-23 (phase review session 3a): I wrote that the 243-run Phase 2 sweep on disk
  "is not the Study B" `status-2.md` describes, from three columns that
  differed, into the card, `STATE.md` and this file. `/challenge-pr` on #77
  recomputed the per-step totals and found the same measurements, differing
  only in the decompose columns. Rule: before calling two runs different,
  compare the columns that should agree, not only the ones that differ.
- 2026-09-23 (phase review session 1): the SessionStart hook printed the main
  tree's `STATE.md`, and the main tree was 9 commits behind `origin/main`
  (nobody pulls it after a GitHub merge). It said chunk 1 was killed and
  `claude/archive-followup` was open; both were stale. Start step 1's
  "older than `origin/main`" check caught it. A fix that removes the check:
  have the hook `git fetch` and print `git show origin/main:STATE.md`
  (a `.claude/settings.json` edit, so it is the user's to make).
  **Again 2026-09-24:** 20 commits behind (#84–#90); the hook's `STATE.md`
  said chunk 3 was the user's to launch and listed parked items since done.
  The same Start check caught it; the hook fix is still unmade. **Third time,
  2026-09-24 (after #91):** the main tree sat at `21391c2`, 26 commits behind (#84–#91);
  the hook's `STATE.md` said chunk 2 was running and Stage 1 not started.
  **Fourth, 2026-09-25 (after #98):** the main tree was 3 commits behind, and the hook's
  `STATE.md` still named Phase 10 as the active thread; Start step 1 caught it.
  **Fifth, 2026-10-02 (after #130):** 5 commits behind; the hook's `STATE.md` said unit 0
  was in a PR and unit 1 next, when unit 1 had merged. Caught by Start step 1.
  **Sixth, 2026-10-02 (after #132):** 15 commits behind (#130–#132); the hook's `STATE.md`
  said unit 1 was next. Caught by Start step 1; the session worked from `origin/main`.
  **Again 2026-10-05, three sessions running (after #146, #147, #148):** one merge behind
  each time, so the hook printed the previous unit's in-PR `STATE.md`. Caught by Start step 1
  each time. Every session fast-forwards the main tree at its start, never at its end, so the
  next one always lags by the merge that came after. The hook fix is still the user's.
  **Again 2026-10-06 (after #152):** 5 commits behind (`6abbcb8`, #151's merge); the hook
  printed R6's in-PR `STATE.md`. Caught by Start step 1 (read `git show origin/main:STATE.md`).
  **Again 2026-10-15:** the main tree at #194, 9 merges behind (#195–#203); the hook printed
  Phase 10's close-out `STATE.md` while Track A sat at Blocked 47. Caught by Start step 1.
  **Again 2026-10-18 (C2 scan):** the main tree at #211, 15 merges behind (#212–#226); the hook
  printed B2a's close-out. Caught by Start step 1. Same session, a stale doc of another kind:
  `docs/LITERATURE.md` §0 says scholarly hosts are blocked, and `WebFetch` read eight arXiv
  abstract pages (`trackA_force/lit-C2.md` marks them [A]).
  **Again 2026-10-24 (C2p close-out):** the main tree at #240, 160 commits behind (#241–#268);
  the hook printed A5's register `STATE.md` while C2p sat at Blocked 109. Caught by Start step 1
  (`git show origin/main:STATE.md`). The largest lag so far; the hook fix is still the user's.
  **Again 2026-10-06 (after #156):** still at `6abbcb8`, now 5 merges behind (#152–#156): the
  five sessions in between each read `origin/main` and none fast-forwarded the main tree. The
  hook printed R6's `STATE.md` again; Start step 1 caught it, and the main tree was fast-forwarded.
  **Again 2026-10-11 (10.5, after #193):** 27 commits behind (`d7bebf7`, #189); the hook
  printed Stage 3's close, Blocked 37 open, when the user's prompt named Blocked 39. Caught by
  Start step 1 (`git show origin/main:STATE.md`); the hook fix is still the user's.
  **Again 2026-10-13 (1e close, after #194):** the same `d7bebf7`: the 10-11 session read
  `origin/main` and did not fast-forward the main tree, so the hook printed #189's `STATE.md` a
  second time. Caught by Start step 1; the main tree was fast-forwarded (`git pull --ff-only`) at
  the cleanup step. Rule: the session that removes a merged worktree also fast-forwards `main`.
  **Again 2026-10-14 (Track A's opening, after #200):** the main tree at `edc8e8c` (#194), 21
  commits behind; the hook printed Phase 10's close with Blocked 40 undecided. No session since
  10-13 removed a worktree (refused by the permission check), so the rule above never fired.
  Caught by Start step 1; the main tree was not fast-forwarded (another session may be in it).
  **Again 2026-10-15 (A1a's batch, after #204):** still `edc8e8c`, now 30+ commits behind; the hook
  printed 10-11's `STATE.md`. Caught by Start step 1 (`git show origin/main:STATE.md`). The
  cleanup command this time includes `git pull --ff-only` and goes to the user.
  **Again 2026-10-15 (A1b, after #205):** still `edc8e8c`; the hook printed 10-11's `STATE.md`.
  Caught by Start step 1 (`git show origin/main:STATE.md`); the pull is still the user's.
  **Again 2026-10-20 (A3's close, after #237):** the main tree at `f99d01b` (#211), 26 merges
  behind; the hook printed B2a's close (Blocked 55) when the prompt named Blocked 75. Caught by
  Start step 1; the session read `origin/main`'s `STATE.md` from the worktree. The main tree was
  not fast-forwarded (another session may be in it); the pull is still the user's.
- 2026-10-13 (B-pre): `claims/registry.json`'s `P-ST1` says twice that its activations and Phase 2's
  projectors "do not exist in this repository"; both have been on disk since 2026-08-31
  (`data/phase12/*/ov_projectors_*`, the v1 sweep). The registry is not edited outside a
  registration, so the stale line stays; `p7_motifs/status-7.md` "B-pre (2026-10-13)" is where
  the current fact lives. Rule: a registry note about what exists is checked against the disk
  before it is acted on.
- 2026-10-11 (10.5): a card open thread outlived the work that answered it. `status-10.md`'s
  card said "no row here is re-read on" 1d's tie-free level-set HDBSCAN, written 2026-10-01;
  R0–R9 (2026-10-05 to 10-09) read every row on c2a–c3x, which are built on that tree
  (`design-10.md` "c2a"), and each of their sessions re-stamped the card without touching the
  thread. Found only because the close-out had to give every thread a home. Rule: when a
  unit answers a card's open thread, the same commit says so on the thread; the stamp checks
  that someone looked, not what the threads say.
- 2026-09-24: `handoff-10.md` said its fixed guard `pgrep -f 'python -m
  tools.run.stage0_chunk'` "now matches only the python process". Run inside
  one `bash -c` with the rest of the block, it matched that shell and printed
  "DRIVER ALIVE" with no driver. A guard's own fix was not tested the way it is
  run. `[p]ython` in the pattern matches the driver, and the pattern's own
  text does not match it.
- 2026-09-23: STATE.md and `handoff-10.md` said Stage 0 chunk 1 "was killed:
  no runs done". The box had suspended; the process resumed and ran on. The
  next session was told to relaunch, and would have started a second chunk
  beside the live one. `pgrep` before launching found it. A process that went
  quiet is not a dead process: check `pgrep` and the log, not the last note.
- 2026-09-22 (archive batch): `INDEX.md` listed 3 cited-but-absent `.md` files,
  found by hand. A lint pass found 95 dangling citations of 15 names, and two
  of them were plain wrong paths (`MATH_INDEX.md` gave `p5c_unclustered/` for
  a file in `p5_single_mstate_analysis/`; a path split across a line break in
  `status-2.md`). Now lint rule `cited-md-path` + `archive/MOVED.md`.
- 2026-09-22: STATE.md's move note planned for a new *local* box (rebuild conda,
  re-download HF cache). The new box was a cloud container where conda channels
  and `huggingface.co` are blocked and 22 GB is free. Check the box before planning for it.
- 2026-09-22: the handoff said PR #60 was open (it had merged), that "no
  Phase-1 timing exists anywhere" (every `manifest.json` has
  `wall_time_seconds`), and set a battery-hash check that could not pass
  (the v1/v2 rule in `core/prompts.py` forbids it). Each cost an investigation.
- 2026-09-20: `7717fa3` "CI on main is red, and the handoff said it was green";
  `4025669` the handoff was wrong about where #59 landed; `69b0cfe` "the
  handoff was stale in three places, one of them a missing finding".
- 2026-09-12: §3.15 still ended on an open hedge the next commit had closed.
- 2026-09-09: `cf700f9` corrected a git line the commit before made stale.
- At least nine commits in the history exist only to correct a handoff.
- 2026-09-24: Phase 3's headline null (decoder→V 0.484 / 0.501) has been
  quoted in `INDEX.md`, `README.md` and `design-7.md` since April, and the only
  Phase 3 runs on disk hold an error where that number should be
  (`archive/p3_crosscoder/status-3.md` "Runs on disk"). A result without a named
  run directory cannot be re-checked. The card sessions have now found five
  runs that were unrecorded or cannot be produced from what is on disk.
- 2026-09-24 (phase review session 6): `P-I5`'s target (§3.32, 2026-09-16)
  was chosen from `status-8.md`'s 2026-09-11 table as "70m's `L7H8`
  analogue". A later section of the same file (2026-09-13) had already found
  70m has no relay-fed matcher. The reader cited one section of a file that
  its own later sections contradict. The same file's "Reproducing" also
  undercounted which runners take `--probe`, which one `grep` settles. Both
  were found only because the card session reads the whole status file.
- 2026-09-28 (1d item 3): the session prompt said the attention-null run
  "has finished (four JSONs)"; the directory held one, `calibrate` stopped at
  126/384, the other two never started. Start step 3 (check the tree) caught
  it before any report was written against a missing calibration.
- 2026-10-01 (1d Blocked 11″): option (a) was quoted at "~7 h" in `STATE.md`,
  `design-1d.md` and `status-1d.md` across four decisions (#108 to #126). The
  figure was `attention_null`'s (the model's block on every draw); the
  admission null it would replace is a Gaussian draw, ~12 min a pass on v1.
  Found only on opening `admit.py` to build it. The same session found that
  `STATE.md`'s "Now" cell (~30 KB, one table row) was the only record of the
  user's 2026-09-25 directions for 1d: a 150-line cap counted lines, not bytes.
  Moved to `status-1d.md`. Rule: a cost carried into a decision names the
  code path it was measured on.
- 2026-10-05 (Phase 10 re-read design): `design-1d.md` "Scope" (#143,
  written to answer a review finding) said F1 "needs a full partition" and
  put A0 and F12 out as well. Opening the readers showed all three use only a
  clustered / noise split (`label_permutation_null` over a binary label), so
  they can be re-read. Nobody had opened the readers. Same rule as the cost
  line above: a claim about what code needs names the function it read.
- 2026-10-05 (Phase 10 re-read, R0): `design-10.md` (#144) said the
  definition's 62 / 810 counts are on "T2's union over all 18 checkpoints".
  They are on unit 2's token set (`token_sets.json`, the union over 10 seeds
  and 40 re-inits), which also drops each prose passage's first `.`. Found by
  opening the run script and a record's `kept_from` before the first pass. No
  run was wasted. Rule: a claim about which input a number was computed on
  names the file and hash it read.
- 2026-10-21 (S1 unit 3, `/challenge-pr` on #245): unit 2's merged sentence "the OV families
  read the null through 256" was false for `ov_write` and `ov_cplx` (ρ 0.9935, 0.9975, both
  below their bands), and unit 3 built "QK leaves the null a step before OV" on it without
  reopening `r1.json`. The headline then reached the card, PROJECT, ROADMAP and STATE. Rule: an
  ordering claim between two records is checked against both records, not against one record
  and the other's prose.
- 2026-10-21 (S2 unit 1, the session's start): the user's prompt read "Blocked 86: <your
  letters>" with the placeholder unfilled. Looking for 86's options, `grep "^8[0-6]\. "` on
  `STATE.md` found nothing, and the session told the user that Blocked 86 "has no entry in STATE"
  and asked only arm (b)'s question. STATE packs decided entries onto one line with " · ", so
  80–87 begin mid-line; 86 was there, with a second part (scope) not asked. Corrected to the
  user within the hour, and the scope asked. Rule: **an absence claimed from a search is checked
  with a search that cannot miss the layout (`grep -o "86\. .\{0,200\}"`, or open the
  section) before it is said to the user.**
- **2026-10-07, the start hook again printed an old `STATE.md`** (#156's, which
  still said "Blocked 28 open, U2's rule unfixed"): the main tree was 8 merges
  behind `origin/main`, and the hook prints the main tree's file. Caught at
  Start step 1 by comparing with `git log origin/main`; read from
  `git show origin/main:STATE.md`. Third instance; the fix (the hook printing
  `origin/main`'s copy after a fetch) is a `.claude/settings.json` change, so
  it is the user's (Blocked 4's reason).
- **2026-10-07, a solo timing extrapolated to a pool** (1e U2): one long run
  took 143 s alone, ~700 s each with 14 workers (memory-bound n × n float64
  arrays), so the "~25 min" first estimate was 1 h 50 min. No run wasted.
  Rule: time the first pool wave, not the solo first cell, before quoting a
  wall time (`docs/compute_profile.md`).
- **2026-10-08, a card line simpler than its own table** (1e, building the
  page): `status-1e.md`'s card said the token-specific part "ascends in L1–16
  only from 16000"; the table beneath it (and `report.json`) has L9–16 from
  4000 and L1–8 at 64–128 too. `STATE.md` repeated the card. Found only
  because the page drew the grid from the report, not from the prose; and the
  first fix was itself incomplete (it missed L1–8's 64–128) until
  `/challenge-pr` on #165 counted the cells (finding 1). The same review found
  "most heads" for "more heads than not" (2 of 13 windows had a majority) and
  an R² range quoted from 6 of 18 steps. Rule: a summary line that names
  windows is checked against the table it summarises, and a page reads the
  report, never the card (`p1e_energy_field/page.py` refuses a count mismatch).
- **2026-10-10, the start hook's `STATE.md` five merges old again** (F13): the main tree sat at
  #181 while `origin/main` had #182–#186; caught at Start step 1 as on 2026-10-07. Fourth
  instance; the hook fix is still the user's (`.claude/settings.json`).
- **2026-10-16, the start hook's `STATE.md` 17 merges old** (A2's opening): the main tree sat at
  #194 while `origin/main` had #195–#211; caught at Start step 1 and the main tree fast-forwarded
  (`git pull --ff-only`, clean). Fifth instance; a closing session's last step could fast-forward
  the main tree, but the hook fix is still the user's.
- **2026-10-18, the start hook's `STATE.md` 14 merges old** (the user's check-in): the main tree
  sat at #211 while `origin/main` had #212–#225; caught at Start step 1, read from `origin/main`.
  Sixth instance. The main tree was not fast-forwarded, because B3 unit 2's batch was running
  from `../Mets-work` with data in the main tree. The hook fix is now a `docs/ROADMAP.md` §5 row.
- **2026-10-20, the start hook's `STATE.md` 27 merges old** (A5 unit 1): the main tree sat at
  #211 while `origin/main` had #212–#238; caught at Start step 1, read from `origin/main`, and
  the main tree fast-forwarded at the unit's end (`git pull --ff-only`; only untracked `data/`).
  Again. The same opening prompt said "Blocked 76 decided (<a/b/c>)", an unfilled
  placeholder: Claude asked rather than take the recommendation (the user chose (a)). Also a
  wasted call: the gate run from `../Mets-work` read `data/` under the worktree (empty) until
  `METS_DATA` pointed at the main tree's; `a1a.DATA` defaults to the code's own tree.
- **2026-10-20, the start hook's `STATE.md` one merge old** (A5 units 2–3): the main tree at
  #238, `origin/main` at #239; caught at Start step 1, read from `origin/main`. **Again the same
  day** (A5 close): main tree at #238, `origin/main` at #240, two merges; caught at Start step 1,
  and the main tree was fast-forwarded. A second stale kind in the same session: three tests
  asserted that the real ledger directory is empty or absent, the right guard only while it was;
  the first real record failed all three. Rule: a guard states its invariant (no record for *this*
  prediction), not the state of the world when it was written.
- **2026-10-18, a "blocked" that was a setting** (the same check-in): `docs/ROADMAP.md` §6 said
  Hugging Face downloads are blocked from this machine, so A5 on seeds needed steps "fetched
  elsewhere". One `curl` on the refs API answered (all 154 steps per seed); `HF_HUB_OFFLINE=1`
  is the scripts' setting. The cloud container's block had been carried over to the local box.
  Rule as before: check the tree (or the network) before planning around a doc's "cannot".
- **2026-10-10, a frame named for what it is not** (F13): `handoff-10.md` Stage 3 and
  `docs/readings/2411.04990.md` §8 said `core/ln_frame.py`'s Gram *is* the paper's `⟨Qx, Ky⟩`
  geometry. It is `LN(x)·LN(y)`: no `W_Q`, `W_K`, and in Pythia rotary makes `⟨Qx_i, Kx_j⟩`
  depend on `i − j`, so it is not a distance at all. Found by opening the module before writing
  the rule; the rule does not run it (Parked). Rule: as the 2026-10-05 lines above, a claim that
  a module computes X names the function and is checked against it.

**Why it keeps happening.** The same fact was written in 3–4 places
(`PROJECT.md`, `status-N.md`, `handoff-N.md`, `INDEX.md`); updating one left the
others wrong. `PROJECT.md` grew to 485 KB (~120k tokens) — too big to read, so
it got skimmed, so errors survived. The update was a thing the user had to ask
for ("update all the MD files"), and which files was never defined.

2026-10-23: **C2's close-out claimed "every Parked item carried or dropped" and missed four.**
The sweep grepped for "Parked"; three deferrals lived in `referee-C2.md`'s answers as "at the
close-out" (m15, m19, m25), one as "C5 or C2's close-out" (M5), and `lit-C2-unit1.md` Parked 2
had been sent to unit 3b, which never picked it up. Found by `/challenge-pr` on #262. Rule: a
close-out sweeps for every deferral form (`grep -n "close-out\|carried to\|Parked"` over the
phase's files) and checks that each item sent to a unit was closed there.

2026-10-30: **the startup hook printed a `STATE.md` nine days and ten merges old.** The hook reads
the main tree's file, and the main tree sat at `f22cf52` (2026-10-20) while `origin/main` was at
`0c3a0d4` (2026-10-29): work happens in `../Mets-work`, the user merges on GitHub, and nothing
pulls the main tree. Start step 1's age check caught it, and the session read `git show
origin/main:STATE.md` and then the open PR's branch. The same session appended a design section
with a heredoc and resolved a LESSONS conflict with a Python script, against the Edit-tool rule.
Rule: **the hook should print `git show origin/main:STATE.md` after a `git fetch` (a settings
change, so the user's), and until then the age check is not optional.**

**The rule now.** One startup file, `STATE.md`, capped at 150 lines and
**overwritten** each time (never appended). Each fact has one home; other files
point to it. Updating it is step 1 of the stop protocol, not a request.
A correction to an earlier phase is also routed to that phase's status file
(`## Corrections received`), where its card's staleness hash sees it.
Status: 📋 routing in `CLAUDE.md` Stop step 2; ⚠️ nothing checks that it was
done (a warning lint is proposed in `docs/PHASE_REVIEW.md`); 📋 protocol in `CLAUDE.md`; ✅ size cap by `tools/lint_repo.py`;
✅ `STATE.md` printed at session start by `.claude/settings.json`'s hook.

## 2. Instruments that degrade silently instead of refusing

**What happens.** A missing dependency or input makes code write a
well-formed but empty/zero result. It looks exactly like a real result.

**Instances.**
- 2026-10-14, Track A A1a re-gate, a progress log that hid the failure: each sweep line printed
  only the *min* T_proj/T, and every one read 0.913 or more, so the log read as a pass. The bar
  is two-sided, and the 7 gated cells outside it are all overshoots (to 1.344, step 143000,
  blocks 20–22); only the JSON showed them. Rule: a progress line prints the gate's own test
  (both ends and the count outside the bar), and the run ends by naming the failing cells; the
  JSON is the record, the log is never read as one (`a1a_gate.sweep` now does both).
- 2026-10-14, the same run, a pool that oversubscribed: 8 `ProcessPoolExecutor` workers each
  ran OpenBLAS at its default 16 threads on 16 cores (load ~110). Said to take ~10 minutes and
  run on the GPU; it took 1h52m on the CPU (the gate is synthetic numpy; only A1a's real pass
  is the GPU's). The first fix (set `OPENBLAS_NUM_THREADS` around the constructor, as
  `arch_null.py` does for `mp.Pool`) did nothing: the executor starts its workers at the first
  submit, after the variable is restored. Now `threadpool_limits(1)` in each worker's
  initializer: 8 jobs in 21 s against 124 s, values within 3e-11. Rule: a pool sets one BLAS
  thread *inside* the worker, and a pilot checks the load (≈ the worker count) and one job's
  time before the estimate is quoted; "GPU" in an estimate names the step that uses it.
- 2026-10-14, Track B B1, the converse: the populated check refused any σ = 0, and the first step
  (1000) was refused, rightly, but on a real zero. `L5H2`'s attention weight a few tokens back is
  ~1e-7–1e-10, so the ± change there is under float32 resolution and A is exactly 0. (The first
  write-up said "softmax underflow"; `/challenge-pr` on #200 asked, and the weights are not 0.) The rule then needed a
  substitution (zeros held at the rounding floor), set after one record and before any verdict.
  Rule: a populated check asserts what a dead pipeline gives (non-finite, or silent where the
  readout cannot be silent: here the nudged token's own output), not a property the model may
  legitimately have; any "> 0" in a check names why zero is impossible.
- 2026-10-10, Phase 10 F13r: `C_id` (merge rank, beside) read "too few" in every cell. The added
  literal-space tokens have no merge rank (NaN), one kept in a record made its bin means NaN, and
  `band_mean` drops non-finite values without a word. The runner's populated check named one
  statistic (`C_f`), which was fine. Fixed (finite tokens only) and re-run before writing up.
  Rule: a populated check counts non-null values for **every** statistic the report labels.
- 2026-10-10, Phase 10 OV1d: the rule carried OV1's "dissolves" flag (kept share < 0.2) to every
  arm of a dose curve. On a flipped or control arm, low kept meant death; on `w+1` / `z+1` it
  means *merged* (death 0.09–0.26), so the report printed "dissolves" for the arms that collapse
  the groups (`status-10.md` §1.39 (iii)). Also, the matched measure `dc` was frozen without one
  probe of its spread: every arm sat at ~1 stream-size, so most matches became interpolations
  across one dose gap. Rule: a flag reused in a new arm set names the kind it means (death, not
  "not kept"); a matching axis gets one record's values before the rule freezes.
- 2026-10-06, Phase 10 R6w: the rule named L12, L24 and the layer mean as readings and required
  records readable at every step of a span, without counting how many such records exist. On c3
  that leaves L12 with 5 and L24 with 2 over 512–143000, and L24 with none over 64–512. The first
  run refused on the empty level (good). The thin levels would have printed labels, four of them
  unlike R1's (`status-10.md` §1.21). Rule: when a design fixes a record set, count it per
  reading level before the rule freezes, and name a minimum.
- 2026-10-05, Phase 10 R2: the ladder's check of c0 against the published F1 / F12 records
  matched units by (timestamp, run dir) and printed "0 of 0 units identical", which reads as a
  pass. The published records were made on the earlier WDS sweep, not Stage 0, so no key
  matched. Caught on reading the line; matched by run-dir name, 11,936 of 11,936 identical.
  Rule: a comparison reports how many units it compared, and zero compared is a failure
  (`p10_r2_ladder.published_check` now refuses on zero).
- 2026-10-02, a pool that hangs instead of failing: 1d unit 2's runner
  (`arch_null.py`) copied unit 1's fork pool, forked after torch's forward
  pass, and added KMeans (`ci2`). Every worker sat at load 0 for 10 minutes
  with no output and no error; it looked like a slow run. Unit 1 never hit
  it (no OpenMP in its workers); `gaussian_null.py` had used spawn all along.
  Now a spawn pool with one BLAS / OpenMP thread per worker, each job given
  its own cloud: 77 s per model. Rule: a fork pool after torch is fine only
  while the workers use no OpenMP; otherwise spawn. A pilot on one input is
  also how to find a hang: watch the load, not just the output.
- 2026-10-01, a guard on the input that did not cover its null: 1d's
  identity simulator refuses snapshots whose smallest `1 − cos` is below a
  float floor, but a nearly collapsed snapshot's matched-covariance Gaussian
  draws are as tight as it is, and one fell below float32 resolution. The
  level-set code refused (good), but the batch died at its first record, and
  the pool's traceback again did not name the input (the 2026-09-30 rule,
  not applied to the new driver). It now records the snapshot as skipped
  with the reason. Rule: a floor on an input bounds everything derived from
  it at the same scale (its null draws, its calibration), or is checked
  there too.
- 2026-09-30, a library's output: scipy 1.15's average linkage returned a
  tree that merged a node with itself (a tied co-association, 1d's vote
  rules; `status-1d.md` "Vote rules"). `consensus_partition` never checked
  the tree; `fcluster` happened to, and raised. The same tree reaches the
  visualisation's `consensus_order`, whose `except ValueError` turns it
  into an unsorted heatmap without a word. The batch died 24 of 48 in, and
  the pool's traceback did not name the record, so finding the input took a
  re-run under a wrapper. Rule: validate what a library returns when its
  failure has a shape (`is_valid_linkage`), and a batch job's error names
  its input.
- 2026-09-30, `tools/mutation_check.py --write` (#119): a test failing in
  mutmut's clean run left all 413 mutants "not checked", and `--write`
  rewrote the accept list from that run as `{}`, discarding 16 reviewed
  reasons (restored from a copy). It now refuses to write while any mutant
  is untested.
- 2026-09-25, Phase 1d's gate: a null draw where a method builds no partition
  was dropped as NaN, which silently raised the p floor. On the first real
  run, agglomerative at fine thresholds had 0 of 20 usable draws and HDBSCAN
  at L18 had 18 (floor 0.053 > alpha 0.05). Both "abstained", and the first
  write-up reported those abstentions as findings about the data. Caught by
  `/challenge-pr` on #98, then confirmed from the kept output. Rule: a gate
  whose reachable p floor exceeds alpha refuses under its own branch; an
  outcome the method cannot fail is not data.
- 2026-09-24 (phase review session 6): `check.sh lint | tail -1 && git commit
  && git push` pushed `3240dfe` with 2 lint errors, because the pipe returns
  `tail`'s status, not the lint's. Fixed in `47e1820`. Capture the exit code
  (`> file; rc=$?`) before gating a commit on it.
- 2026-09-24: `run_2d.py` nested the bandwidth scan under `modality` and ran it
  only with `--bw-scan`, but `p_value_p_t1` reads a top-level `stability` and
  skips heads without it. Scored on the runner's records, `P-T1` would have
  come back "need both arms: 0 candidates", which reads as a finding about
  the model. The unit tests fed the gate hand-built records, never the
  runner's. Found by reading the runner against the gate while swapping in
  the gate. Rule: test a gate on its producer's actual output
  (`tests/test_run_2d.py`), not only on fixtures.
- 2026-09-24: every gate run from a worktree partly tested `main`. 47 runners
  defaulted `METS_REPO` to the main tree and inserted it at `sys.path[0]` on
  import; collecting one test that imported a runner switched every later
  first import to main's copy. It surfaced only because a new test failed
  against the old `run_2d.py` while passing alone. Parked 2 had seen one symptom
  (a main-tree warning) on 2026-09-23 and deferred it. Rule: a default path
  is the file's own checkout, never an absolute path to another tree
  (`tests/test_no_hardcoded_repo.py`).
- 2026-09-24 (same runner): it adjudicated its own registered gates and
  printed `P-T1: <verdict>` per run, so a pilot could put the outcome on a
  terminal before anyone decided to look. A runner for a registered
  prediction now measures only, and an AST test forbids it calling a gate.
- 2026-09-22, #69: `/challenge-pr` could not run after `d5d6218`. Its
  `gh pr view … | awk …` step fails the skill's own `allowed-tools` (no
  `awk`), and `claude -p "/challenge-pr N"` then exits 0 with no output and
  no PR comment. Fixed by hiding the section in `gh`'s `--jq` (no pipe).
  After any review run, check the PR actually has the comment.
- `pair_agreement` — the project's only semantic instrument — wrote zeros into
  all 152 WDS directories while HDBSCAN was missing (§3.54.1). Survived the
  outage, the backfill, an audit and two literature passes.
- `hdbscan_labels.json` was `{}` in 152/152 directories (§3.51); `docs/AXES.md`
  listed it as present.
- `b55375e`: CLAIM-C's metric set was silently missing a dependency and the
  writer forged the missing field.
- Runners launched from a worktree silently import the main tree's code
  (`METS_REPO` default).
- The cross-phase pattern (§ "the cross-phase finding"): three phases produced
  clean uniform nulls forced by the instrument — a scoring term that can
  silently be zero still returns a ranked list.
- 2026-09-26, `run_1d.py`'s sub-experiment flag: `_REQUIRES` said `F` (the new
  merge-tree stage) needs nothing, but `process_layer` ran the full tuning
  grid (stage A) unconditionally before checking `stages` at all — only stage
  B was ever actually gated. `--subexp F` on real data took 2m51s wall / 39
  min CPU, not the sub-second cost F's own cost claimed. The unit test that
  would have caught it used a 30-token synthetic fixture where the wasted
  tuning pass was too fast to notice. Caught by timing a real run before
  writing the number into `status-1d.md` (standing rule "Empirical numbers
  get their producer re-run"), not by the test suite. A flag that claims to
  scope work down needs a timing check against real-sized input, not just a
  correctness test against a fixture small enough to hide the waste.
- 2026-09-28, 1d's page and report (item 3): both were built before the data
  existed and never rendered on it. On the real outputs the attention grid
  overflowed to 8 800 px (a grid item's `min-width:auto`), the page had no
  charset, and `summarise` reported z ≈ 1e13 on null SDs of float noise.
  Caught by looking at the page once and reading the report before writing
  it up. A viewer or report is part of the instrument: open its first real
  output, as for a batch.
- 2026-09-29, #110 follow-up: one shell call ran `sed … && cat >> test.py
  <<EOF … && pytest`. The `sed` failed on its delimiter, so the `&&` chain
  skipped the append, and the pytest in a *separate* line still ran and
  passed (32 = the old 27 + the 5 in a new file). I told the user the new
  collinearity test "hits the guard" when it did not exist. Caught at
  `git status` (the test file unmodified). Rule: after adding a test, see it
  fail without the fix before quoting it; and never put an edit behind a
  `&&` whose failure is easy to miss.
- 2026-09-29, 1d long prompts: `core/models.py` tokenizes every prompt with
  `truncation=True, max_length=512`, silently. `homer_iliad` is 562 tokens,
  so every stored run of it is its first 512, and no doc said so (found
  only because the long prompts needed the cap raised). The same day,
  `STATE.md` said 164 GB free where 72 GB was, which would have sized the
  long runs wrong. Rule: a length cap in an extractor records or refuses,
  never trims; and a resource figure in `STATE.md` carries its date.
- 2026-09-29: `tools/lint_repo.py`'s `PACKAGE_DIRS` was a hand-kept tuple that
  stopped at `p7_motifs`, so four of its rules had never read `p1d`, `p7d`,
  `p7e` or `p8`, and `pyproject.toml` did not declare `p1d_cluster_ensemble`.
  Both lists are now checked against the directories with an `__init__.py`
  (rule `pyproject-packages`). A lint's scope is an input too.
- 2026-10-08, 1e U1 (#165, Blocked 30 (b)/(c)): U1's GPU phase merged rows
  within 1e-6 using a float32 dot, but a CUDA float32 dot of 1024-d unit rows
  is only good to about 1e-6. So a row near a ridge could join another
  basin. The first-cell check against CPU float64 passed on (a) and (b)
  because step 143000 has few wells. (b) at step 2000 had 185 wells where
  float64 has 189. It surfaced only when (c)'s check fell to 0.9984 and U3
  refused on a negative persistence. Rules: a tolerance must sit well above
  the precision it is compared at; a check against a reference runs on the
  hardest cell too (most wells), not only the first; and a cheap full audit
  (`u1_audit`, ~9 s a record) beats a one-cell gate. (a) is audited after
  the fact.
- 2026-10-08, 1e U1 → U3: U1's well labels can skip an id (merged rows
  leave a mode with no row). U1's counts never noticed; U3's merge tree
  read the skipped id as a well no edge could join and looped to its
  bridge cap. A label array consumed downstream is made contiguous, or the
  consumer checks it is.
- 2026-10-08, 1e and 10 in one session: R1's ladder ran with default BLAS
  threads (16) beside a GPU job whose float64 finish was also on the CPU,
  and the load reached ~80 on 16 cores. Both slowed down. Concurrent jobs
  set `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` explicitly.
- 2026-10-10, #181 (`/challenge-pr`): the runner-batch counter read only the
  idle shape of a `<task-notification>` (a `user` record); a mid-turn one is
  an `attachment` (146 of 308 tasks in this repo's transcripts), so half the
  batches would have read "open" and counted to the transcript's end. The
  hand-counted baseline beside it put E1 at 67 calls, its window running to
  the Monitor's expiry notice 30 min after the batch; under one rule it is
  21. Rule: a counter is tested on every record shape its event takes in real
  transcripts, and a baseline comes from the same function as what it is
  compared with.
- 2026-10-14 (B0): the first-step check was launched as `a && mkdir && (gpu) & (cpu) &`; `&`
  backgrounds the whole `&&` chain, so the CPU arm started before its directory existed, the log
  redirect failed and G2 never ran, unreported until the log was opened. Same session: STATE.md
  was edited by a Python script, against CLAUDE.md's Edit-tool rule. Rule: **create the output
  directory in its own command before launching parallel arms, and wait on each arm's exit.**
- 2026-10-21 (S1 unit 3), both again: `cd … && mkdir -p A B && … nohup … &` then the second arm's
  launch on the next line; the first `&` backgrounded the `mkdir` with it, and the second arm's log
  redirect failed. Caught at once (the shell printed the error), relaunched, nothing lost. The
  first edits to `status-S.md` were again a Python string replace. The rule lived only here, so a
  session that did not open this section repeated it. Rule unchanged; the `runner` template's
  launch line is the one to copy, with the `mkdir` as its own call.
- 2026-10-21 (S1 unit 3), a comparison against the wrong null: `r3` scored step 0's compressions
  (the trained S on step 0's V_l) against the *trained* step's Haar draws, so its "outside" counts
  read 19–24 of 24 at every step from 512 and looked like a finding. Found reading the output
  (the trained step's own Haar range was [0, 0] there). Not re-run: the design used step 0 for its
  medians only, and those are right; the counts are marked "not read" (`status-S.md` "S1 unit 3").
  Rule: a summary that takes a null as an argument names whose null it is, and a test feeds it a
  mismatched pair.
- 2026-10-21 (S2 unit 1), three GPU out-of-memory stops in one unit, each the 2026-10-16 rule
  again: `wiring` held 8 feeds' logits and hidden states on the GPU (the probe's alone 0.6 GB);
  in `g0` the name `hook` kept the previous step's model alive into the next load; the preflight's
  G3 held step 128's weights and a copy of 8000's on the GPU beside the model. Each cost a run
  (the first two a runner round trip). Also: `write_b_eig` read `W_V` through
  `.double().cpu().numpy()`, which on a float64 tensor is the tensor itself, so writing `g W_V`
  changed the `W_V` the bias was then computed from; the float64 smoke test caught it, float32
  could not have. Rule: **a reference kept for a comparison goes to the CPU; a loop over steps
  `del`s every name that holds the model (hooks included); a numpy view of a weight that is about
  to be written is copied.**
- 2026-10-21 (S2 unit 1): S2's hook took the attention weights as `out[2]` (4.44's
  `(out, present, weights)`); 4.57 returns `(out, weights)`, and the smoke test failed there,
  found by running it under 4.57 before the PR as the rule below says (`PYTHONPATH` the
  `tf457` scratch). Fixed by taking the weights by shape. The records (4.44) are unchanged.
- 2026-10-14 (B0, `/challenge-pr` on #198): `core.nudge.Stack` called GPT-NeoX layers the
  4.44.2 way (the conda `mets` env, where the gate and the smoke test ran); the nightly installs
  4.57.6, where the call crashes and the obvious fix silently drops the causal mask. Rule: **code
  that calls a library's internals passes its smoke test in both environments (`mets` and
  `.venv`) before the PR, and states the contract it relies on (here: the mask is passed, never
  assumed).**
- 2026-10-16 (A2 unit 4): the launch script had an apostrophe inside `${VAR:?msg}` and failed
  `bash -n` before any stage ran (one runner round trip); the batch then ran out of GPU memory at
  step 64, nine steps in, because each record was held across prompts (resumed). STATE Blocked 58
  quoted the 64-node gate at 25 min; it took 90 (the G2 redraws scale with the nodes). STATE.md was
  again edited by a Python script (one line deleted), against the Edit-tool rule. Rule: **`bash -n`
  every launch script before handing it to the runner; a per-record GPU loop frees its record and
  empties the cache each iteration; a wall-time quote scales every stage the change touches.**
- 2026-10-28 (figure 12): `numbers.py` lists `repo:trackC_writeup/p_a5_push_record.json` (GitHub's
  events dating P-A5's registration, which GitHub keeps ~90 days) as a tracked record, but the
  deny-by-default `.gitignore` never whitelisted it: its only copy was untracked in `../Mets-work`,
  so `produce` refused in any other worktree, and removing that worktree would have lost it. CI did
  not see it because CI never runs `produce`. Rule: **a `repo:` record is checked with `git
  ls-files` when it is added; `git check-ignore -v` on any new non-`.py`/`.md` file before commit.**
- 2026-10-29, Track T T2 unit 1, a derivation that returned nonsense quietly: on `dodecahedron-67m`
  `c2_seeds.null_matched_tol` gave a centred tolerance of −3.05e-4, which counts energy rises as
  violations; only the gate failing on other grounds kept it out (3c's Parked 0 had already named
  the same function's zero-rate case). The env also met a transformers-5 checkpoint: its
  `tokenizer.json` did not load under tokenizers 0.19, and its config carries `head_dim` and
  `rope_parameters`, which 4.44 ignores. Rule: **a derived threshold refuses outside its meaning
  (τ ≤ 0, a quantile set by one unit); a model saved by a newer library is loaded with a check of
  each field the old one ignores and a known-trained test (loss), not trusted because it loads**
  (`trackT_timaeus/t2.py` `null_matched_tol`, `check_llama_config`, `convention`). A tracked doc
  (`status-T.md`'s card) was again edited by a Python script, against the Edit-tool rule.

**The rule now.** Refuse rather than degrade (standing rule 4). Before
launching a batch, inspect the **first** output for populated content, not
just existence. Status: 📋 `CLAUDE.md`; ⚠️ no automatic run-directory
validator yet — candidate tool: `tools/verify_run_dir.py`.

## 3. The input changes underneath code that assumed it was fixed

**Instances.**
- 2026-10-23, C5 unit 1: `c5_massive.code_sha()` hashed the source files each time it was
  called, and pass 1 (a 27-minute run) stamped each record with it. Freezing τ_p and adding a
  `--seeds` flag mid-run changed the file, so the running process, still executing the code it
  loaded, labelled its later records with newer hashes (180 files under three hashes).
  `select`'s one-code check refused it, so no output was wrong; cost: the pass re-run under one
  hash (the first set aside as `2026-10-23-u1-v0`). Every earlier `code_sha` in the repo has the
  same shape (`c2_seeds`, `c2_aniso`, `a5_seeds`). Rule: **a run's code hash is taken once, when
  the process starts** (`functools.lru_cache`), and nothing the run imports is edited while it runs.
- 2026-10-20, `P-A5`'s registration: the first entry from a track directory. A5 unit 1 had
  taught `check_registry` the phase value `A` and written "accepted" in STATE, but two other
  readers of the registry still assumed every phase lives in `p<N>_*`: `check_registry`'s ID
  allowlist (`P-A5`'s mentions in the tree were invisible to the coverage check) and
  `render_experiments._phases_on_disk` (a test failed: phase `A` "has no directory"; once fixed,
  `trackB_circuit/` showed as an unexplained phase). Found by reading the allowlist and by
  running `check.sh` on the registration branch, both before any PR; no seed was affected. Rule: **a new value in a registry field is
  checked by every reader of that field** (`grep -rn '"phase"\|\["phase"\]' tools/`), not only the
  one that rejected it first.
- 2026-10-17, A2 unit 5: the per-component pass ran out of GPU memory on the 2,041-token
  passages twice; the second time because the shared loader sets `output_attentions=True`, so
  all 24 layers' (16, n, n) maps stayed alive through a pass that never read them (v1's n ≈ 511
  hid it). Rule: **a reader that moves to longer inputs passes the loader's memory-heavy flags
  explicitly** (here `output_attentions=False`) instead of inheriting them.
- 2026-10-09, Phase 10 E1: the rule borrowed 1e's `u2_attn.check_pass` for a pass read at
  blocks 0–23, but that check was written for 1e's blocks 0–22: it compares `x + attn + mlp`
  with the next stored hidden state, and `hs[24]` is after `final_layer_norm` (1e had refused
  block 23 for exactly this). The first record refused (off by 0.1), ~1 min lost; block 23 is
  now checked through the final norm, written into the rule as a departure before any reading.
  Rule: a borrowed check's range is part of its input; when the new reader's range is wider,
  read the check before writing "the check above" into a rule.
- 2026-10-09, Phase 10 R9 / R2: #166 added c3x's learned split to `p10_label_source.SPLITS`, but
  F1's, F12's and A0's readers checked a split column's steps against the older `LEARNED_SPLIT`
  (c3's two only), so `c3x_learned` would have been asked for all 18 steps and refused. Found by
  reading the step check before launching (no run lost); fixed to `SPLITS` with a test. Rule: a
  list of columns lives in one constant; a second, narrower tuple of the same thing is a stale copy.
- 2026-10-07, 1e U2's per-head arm: the rule's populated check (every cell finite on ≥ 90 % of
  targets) assumed every head reads some key besides key 0 and the token itself. Trained heads
  do not: at 143000 L18 head 3 puts all its weight on key 0 at 1,817 of 1,839 targets, and
  other heads attend only to the token itself, so the head's keys part or `V = I` field is 0
  there. The check refused three times on the first cell (~4 min); the rule was amended before
  any reading to accept exactly those gaps, verified per head (≥ 0.9999 on key 0 and the token
  itself), and still refuse any other, but only on the first run: the report then dropped every
  short cell on the other 269 without checking why, and `/challenge-pr` on #161 found 2 the
  explanation does not cover (v1's L18 head 3 at 143000 on a newline, a second sink). The report
  now checks every record and names them. Also: the CPU ascent step ran 12 workers each with
  multi-threaded BLAS (load 34) and crawled until restarted with `OMP_NUM_THREADS=1`, and a
  `pgrep -f "<cmd>" | xargs kill` in the same shell line killed its own restart. Rule: a check
  that "every unit is populated" names why a unit may legitimately be empty; a process pool
  pins BLAS to one thread; never `pgrep -f` a pattern the killing command line contains; a
  check the rule names for the first cell is run again wherever its outcome is used.
- 2026-10-07, 1e U2's attention arm: the first check ran on the rule's first cell
  (`wiki_paragraph_long`, 1,840 tokens) in a fresh process, and the batch died twice on the
  next runs with CUDA out of memory: the hooks held the previous checkpoint's model, and the
  captured key-0 column `w[..., 0]` was a view that kept every layer's (16, n, n) attention map
  alive (6.4 GB at 2,041 tokens, 5.2 GB at 1,840, which just fit). Cost: two relaunches, ~5 min,
  no record kept from the failed runs. Rule: a first check that also gates memory runs on the
  **largest** input and on a **second** pass in the same process, and a captured slice of a
  big tensor is cloned.
- 2026-10-06, the `φ_β` probe (#155, `tools/run/p10_phi_wells_probe.py`): it applied the
  measured β = 3.5 to raw and cloud-centred unit rows, but β was fitted on unit LN1 rows
  (`status-1d.md` "β refit"), and its "2–4 wells" went into a merged handoff line. Found
  while writing 1e's design, before anything was read on it; the number is marked as off-frame
  and not used (`design-1e.md` "Inputs, fixed here"). Rule: a measured parameter carries the
  frame it was measured in, and a reader that takes it reads in that frame or says it does not.
- 2026-10-05, Phase 10 R1: the batch driver set `OMP_NUM_THREADS=1` to run 14 readers in
  parallel. §1.9's and §1.10's readers rank float32 Gram entries, and copies tie exactly at
  layer 0, so the thread count (an environment setting, not an input anyone recorded) moved
  one record's kNN control by 0.27 and the c0 column stopped reproducing the published
  records. The c0 reproduction check caught it before anything was read; one 32-min re-run at
  the default threads. No reading label changed. Rule: a result that ranks float32 values is
  reproduced at the thread count it was made at, and a driver does not change BLAS threads
  without checking a reproduction first (`handoff-10.md` Parked: float64 Grams).
- 2026-10-05, 1d unit 4: two runners built for the 7 v1 prompts (512 tokens)
  were pointed at three short designed prompts (90–199 kept tokens). Unit 2's
  guard against a NaN group bar (#132) raised for the **whole run** when three
  raw / size-4 cells, a beside arm, had re-init groups their Gaussians never
  form (`s` = ∞): a refusal scoped wider than the defect. Now the cell refuses
  alone, listed. The same parametrisation renamed a variable to `keys` inside
  `scale_real diagnose`, which has its own `keys` (condition names), so the
  loop iterated over condition names (`KeyError: 'window'`); no test runs that
  command. Both found by the run, after the records, before any result was
  read; ~5 min lost. Rule: a guard's scope is the cell it protects, and a
  mechanical rename gets a grep for the new name in each function it enters.
- 2026-09-23, #69: the PR said "2661 passed"; CI said 1 failed, and `main`
  went red on merge (no branch protection, lesson 5). The gate ran before
  `git add`, so `tools/rewrite_moved_refs.py` was untracked and its new test
  (which reads `git ls-files`) never saw it; its docstring matched its own
  rule. The lint, meanwhile, walked the filesystem. Rule: run the gate after
  `git add`, and a check reads the tracked set, as CI does. Found by
  `/challenge-pr`, not by the author.
- 2026-09-22: battery v2 (8 → 20 metastability prompts) landed 2026-09-19.
  `p7_motifs/p_i5_ablation.py` iterates whatever `core.config.PROMPTS` holds,
  and its calibration record stores no battery hash — so the **registered**
  P-I5 gate now silently runs on 20 prompts, not the 8 it was calibrated on.
  The nightly smoke test caught it (`assert 20 == 8`) and has been red for
  four nights with nobody noticing. 14 live files iterate the live battery.
- `6de78d0`: a cache asserted by identity, which scipy 1.18 showed it never did.
- `0ef60d0`: transformers 5 broke the smoke tier; pinned `<5`.
- 2026-09-29, the audit: `run_7.py` (Phase 7's producer, read by four
  registered rows) rebuilt pair positions from the live text and recorded a
  key list as its "battery hash". A text edited under an unchanged key would
  have moved every pair with nothing failing. Latent (no text has changed);
  `status-7.md` finding 9. The input can change by *growing* or by an
  *edited text*; P-I5's pin only covered the first. `/challenge-pr` then
  found the audit's own blind spots: the degeneracy gate still judged the
  full text while the pairs read the run's prefix, and a grep for
  `PROMPTS` cannot see code that stamps the live hash without reading the
  battery (`p2b_io.py`) or that edits `PROMPTS` after hashing it
  (`run_1 --length-sweep`). An audit by grep covers only what it greps for.
- 2026-10-02, 1d unit 2 trained cells: `arch_null run` wrote `token_sets.json`
  into `<out>/<step>/` without creating it; in the first check, `norms` had
  always made that directory first. A new `--out` crashed the batch at launch
  (cost: one relaunch and a fix commit, `cddcba7`). A stage that writes
  makes its own directory; no test ran `run` on a fresh `--out`.
- 2026-10-15, A1a's batch (Track A unit 4): `a1a.ln_from` read `model.gpt_neox.layers`, but
  `core.models.load_model` returns the bare `GPTNeoXModel`; the gate's LN cache loads the LM
  (`b0_gate.load`), so the gate never touched the batch's loader. The batch crashed on its first
  load, before any record (cost: one gate re-run, 6 min, since the fix changed the code hash the
  gate checks). `Hooked` already had `getattr(model, "gpt_neox", model)`. Two loaders return two
  shapes. A batch whose gate is tied to its code hash cannot run before its gate, so whatever the
  batch takes from the model gets a test on the object `load_model` actually returns.

**The rule now.** Anything whose result is recorded names the exact input
set it ran on (a hash of the texts, not a key list) and refuses on a
mismatch. Code that rebuilds positions from the live text also checks them
against the run's `tokens.txt` (`core.battery_structure.verified_prompt_ids`).
Status per file: `docs/battery_consumers.md` (audit 2026-09-29; P-I5 and
`run_7.py` fixed, `CLAIM-C`'s growth is the user's call, three tier-1 files
not fixed).

## 4. Tests that assert a property of the machine, not of the code

**Instances.**
- 2026-10-16, B2a unit 1 (#208, blocked a merge): `test_b2a` asserted a planted plane's angle
  < 1e-10 through `plane_angle`, an arccos of the smallest singular value, which near 0 resolves
  only to √(2 eps) ≈ 1.5e-8. The same commit passed on the PR run and failed on the push run at
  1.49e-8 (BLAS rounding). Fixed by asserting sin of the angle, `‖U − PPᵀU‖₂`, exact to rounding
  (`e947a04`). Rule: **a tolerance below an instrument's resolution tests the machine**; check the
  resolution of the formula (arccos, sqrt near 0) before choosing the bound.
- 2026-10-06, Phase 10 R7: check (d) compared a 2-token pass with the passage's 512-token pass
  at 1e-5, a tolerance borrowed from re-running the *same* input against Stage 0. At step 143000
  float32 sequence-length noise is 1.5e-5 even unbatched, so the gate refused `latex_monograph`
  (1.06e-5). Caught at the gate, fixed as a departure (1e-4, `design-10.md` "R7"). Rule: a
  numeric tolerance is placed for one computation; reusing it for another (different shape,
  length, batch or device) needs that computation's noise measured first, at the latest checkpoint.
- 2026-09-22: two float32 tests compared roundoff at d=128 (itself ~1e-06)
  with a 1e-06 tolerance. Pass/fail was decided by the runner's CPU kernel;
  `main` was red from #57 to #60, and then went *green* on `086bdd1` without
  a fix. Fixed in #61 by testing at d=1024, where the margin is ≥ 3.3×.
- `238b503`: tier 0 went red because a module imported numpy at scope — passed
  on every dev machine, all of which have numpy.
- CI ran Python 3.11; the local envs are 3.14 (`.venv`) and 3.10 (conda). Fixed
  by #63 (matrix py3.10 + py3.14).
- 2026-09-24: `tests/conftest.py` replaces `core.config` with a stub whose
  `PROMPTS` has 2 keys, so a test that imports `PROMPTS` checks the stub, not the
  battery. `test_holdout.py` failed on that and now reads the keys from
  `core/config.py` with `ast`. Any test asserting on the real battery has to do the same.
- 2026-09-27: on the fleet box (aarch64, OpenBLAS `neoversen1`, py3.10) the
  gate has 4 failures that CI's x86 py3.10 passes: an eigenvalue count at
  `1e-10` (7 vs 6, `test_phase2b_schur.py`), an energy fraction on an exact
  tie (1.0 vs 0.0, `test_phase2b_head_circuits.py`), a corrected R² 0.18 vs
  < 0.05 (`test_p10_partition_function.py`), and a refusal whose branch
  (ties vs draws) flips (`test_p_i1_attainable_floor.py`). CI has no ARM leg.
- 2026-10-07: #160 and #161 were red in CI's deps tier from their first push.
  `tests/test_p1e_u2_attn.py` and `test_p1e_u2_heads.py` build a tiny real
  GPT-NeoX, marked `deps`. Outside `SMOKE_REAL_DEPS=1`, `conftest.py` makes transformers a
  MagicMock, so the model had no layers ("stack expects a non-empty TensorList";
  the parallel-residual refusal never raised). They had passed only with real
  transformers 4.44 on the box. Under CI's 4.57 the hooks broke as well:
  `GPTNeoXAttention._attn` no longer exists (it became the module-level
  `eager_attention_forward`). Fixed: both modules marked `smoke`, and `u2_attn.Hooked`
  supports both APIs. Rule: a test that needs a real model is `smoke`, and it is run
  with `SMOKE_REAL_DEPS=1` against CI's transformers before the push.
- 2026-10-10, Phase 10 OV1: the same again, three days later. `tests/test_p10_ov1_cut.py` built a
  tiny GPT-NeoX outside `smoke`; under the stub the cutter looped over zero layers and 4 of its
  model tests **passed vacuously** (no assertion ran). Caught only because a fifth indexed
  `cut.orig[0]`. Fixed: the model tests moved to `test_p10_ov1_cut_smoke.py`, which asserts the
  model has its layers, and `Cutter` refuses a model with none. The box's transformers 4.57 env
  cannot run the repo (no `hdbscan`, torch import fails), so the smoke tests ran on 4.44 only;
  the nightly covers 4.57. Rule added: a test that builds a model asserts it is real (a layer
  count) before asserting anything about it, so a stub fails instead of passing.
- 2026-10-11: the nightly went red on its first night with `test_p10_ov1s_sign_smoke.py`
  (`test_hook_output_is_the_map[norep]`). Since 4.5x, transformers takes GPT-NeoX's eager softmax in
  float32, so in a float64 tiny model P's rows sum to 1 only to ~1e-7, and the test's `want` added
  each head's value-bias term `c` once instead of `rowsum(P)·c`: 1.7e-7 off at 4.57.6, 4e-15 at
  4.44.2; `noatt` and OV1d's twin passed only through `allclose`'s default `rtol`. The hook and the
  model use the same P (bit-identical); nothing in `DeltaHook` / `Hooked` was wrong. Fixed in the
  tests: `rowsum(P)·c`, `rtol=0, atol=1e-9` (gap 5e-15 under both). OV1's cut test keeps `+ c`: the
  cut moves `c` into the dense bias, where it enters once. The PR's first diagnosis ("the hook adds
  with a float64 P") was a guess written as a cause, and its first fix loosened the tolerance to
  1e-6; `/challenge-pr` on #191 measured the real cause. Rule: before loosening a tolerance, find
  which term carries the gap. The runs on disk used 4.44.2 in float32, so no result moves. The rule's "against
  CI's transformers" is now cheap on the box: `pip install --no-deps --target <scratch>/tf457
  transformers==4.57.6 tokenizers==0.22.1 huggingface_hub==0.36.0 safetensors`, then run the
  `mets` python with `PYTHONPATH=<scratch>/tf457 SMOKE_REAL_DEPS=1`.
- 2026-10-21 (S2 unit 1, G0): the design gated "the read-back K′ has the arm's eigenvalues" at
  1e-6 ‖K‖ on the float32 core the hook holds. (b-schur) leaves near-repeated eigenvalues (gaps to
  7.5e-7 ‖K‖), which float32 rounding moves by √eps: 3.1e-4, while the map was exact to 3e-8 and
  the float64 core's eigenvalues to 2e-10. The gate asserted float32's eigen-sensitivity, not the
  code; one G0 run spent, diagnosed before any change. Revised before the freeze (Blocked 87): the
  map and the exact core gated, the float32 eigenvalues beside. Rule: **a gate on eigenvalues of
  a stored (rounded) non-normal matrix first checks the eigenvalue gaps; near-repeated ones are
  gated in float64 or through the map.**
- 2026-10-21 (S2 unit 1): the smoke bars for the hook were 1e-10 in float64, and under 4.57 the
  constant moved into the bias is off by `(1 − rowsum P)·c` (the float32 softmax, the 2026-10-11
  instance above). The bars now add 10× the measured row-sum gap, which is 0 under 4.44.

**The rule now.** A numerical threshold in a test gets its margin measured
across kernels (`OPENBLAS_CORETYPE=Prescott|Haswell|Zen`) and written next to
it. Status: 📋; ✅ CI matrix matches the local interpreters (#63).

## 5. Nobody watches CI, and nothing stops a red merge

**Instances.** `main` has **no branch protection**. #57 and #58 were merged
with CI red. Nightly smoke failed 2026-09-19 → 22 unnoticed.

2026-09-22: **CodeRabbit reviewed nothing from #50 to #68.** Every PR carried
its notice "does not receive automatic reviews because it has fewer than 10
stars", while `CLAUDE.md` said "CodeRabbit reviews PRs". Found by `/challenge-pr`
on #68. Rule: comment `@coderabbitai review` on every PR (Stop step 8).

2026-09-22: #63 (the py3.10 matrix) merged without the re-run STATE.md asked
for after #61; the first py3.10 run with #61 in was on `main` itself (green).

2026-09-29: **the deps tier ran in no CI for 10+ nights.** PR CI ran lint and
pure only; the deps tier (770 tests, every Phase 1d test among them) ran as the nightly smoke
job's last step, which GitHub skips when an earlier step fails, and the smoke
step had failed every night since 2026-09-19 on the known P-I5 red (Blocked
2). The issue said "smoke is red"; nothing said "and the tier behind it did
not run". Meanwhile `scripts/status.sh` printed "check by hand" and exited 0
in every cloud session (no `gh`), so start step 2 reported nothing. Found when
the user asked why PR CI was so fast. The tests passed once run (770 passed,
1 skipped, 176 s, CPU torch 2.14, cloud container, 2026-09-29). Fixed: a `deps` job in
`ci.yml` on every push, the nightly step runs `!cancelled()`, `status.sh`
falls back to the public API and exits 1 when it cannot see.

2026-10-23: **#261 was red in the deps tier from its first push, through `/challenge-pr` and
CodeRabbit.** Unit 5b added `fig5_count.svg`; `tests/test_trackC_figures.py` (marked `deps`)
still listed four figures. `./scripts/check.sh` deselects `deps`, so it passed locally, and the
user saw only "exit code 1". Fixed by adding the figure to the test. Rule: read `gh pr checks`
before Stop step 9, and a unit that adds an output its `deps` tests enumerate runs
`pytest -m deps` on those tests before pushing.

2026-10-24: **#267 red in the deps tier at its first push, again on that test** (five figures
added; it listed five), and the rule above was not run before Stop step 9: `/challenge-pr` found
it from CI. Worse, the test pointed `FIGDIR` at a temporary folder but not the new PDF folder, so
running it overwrote the tracked `tex/fig/*.pdf`. Fixed: the test reads the figure list from the
draft and sends both outputs to `tmp_path`. A list a test can derive from the tracked source is
derived, not typed.

**The rule now.** At session start, `./scripts/status.sh` for `main` and the
nightly smoke; a known red step must not gate the steps after it. Status: 📋
`CLAUDE.md` start protocol; ✅ deps tier on every push; ⚠️ branch protection
needs the user to enable it on GitHub.

## 6. Statistical designs that could not have rejected

**Instances.**
- 2026-10-25, B7 unit 2 (a planted gate homogeneous where the real input is not; found by
  `/challenge-pr` on #276 after the real grid was read): every planted arm put one partner map in
  every head of every prompt, at every row. On the real rows two prompts carry about 0.9 of
  mid-depth's condensed rows from 8000, and 0.85–0.94 of theirs sit on one newline (a second sink,
  known since this file's lesson 3 and never checked on B7's rows), so "depth-local persistence"
  mostly means "kept on the sink", and the 7-prompt bootstrap swings with which of those two prompts a
  draw holds (22 of 155 descriptions move with h\* held fixed). Claude first blamed the
  interval's re-choice of h\* and sized it with a filter that can only narrow an interval (wrong
  twice; the reviewer re-ran it). Rule: **before a gate freezes, read the real rows' composition
  at one step off the grid (each prompt's share of the rows, each prompt's most-attended key) and
  plant what it shows (a second sink, prompts that differ); and a diagnostic that changes two
  things at once (re-selection and draw size) sizes neither**. `status-B.md` "B7 unit 2".
- 2026-10-25, B7 unit 1b (a new arm pre-checked on its statistic, not its pass rule, on a
  simplified input; seen in the gate, before any real cell): G1 at the minimum entry was added for
  N0's bias. The PR's scratch numbers and this session's pre-gate GPU check both read N0 and r, and
  the GPU check used one constant sink (0.3) for every head, so every head pair kept the same 50
  thinned rows. With the real per-head sinks the thinned sets differ. Under G1's 16 identical
  heads, the count-based leave-one-out h\* then picks, for each held-out prompt, the head with the
  fewest rows there, so the matched pair sums to under 50 and 16–18 of 23 cells do not enter (8000,
  143000). N0's bias was fixed in every read cell. Rule: **a scratch pre-check of a new gate arm
  runs that arm's own pass check at the real design points (real sinks, real counts), not only the
  statistic it was added for**. Blocked 112.
- 2026-10-25, B7 unit 1 (a null estimated from the statistic's own rows; `/challenge-pr` on #272,
  before any real cell): N0's offset profiles are built from the same jointly condensed rows P is
  counted on, so each row's own agreeing pair enters Σ p q and N0 rises by ≈ P/m. Check 6 and the gate's
  null arms had P at chance, and the persistence arms had full rows (m ≈ 200), so the bias stayed
  under 1 %; at the 50-row entry it is 0.3 and G3 fails. Rule: **a null estimated from the same units
  as the statistic is checked under the planted effect at the smallest admitted count, not only under
  the null; a plug-in Σ p q over the same sample becomes a U-statistic over different units**.
- 2026-10-25, B7 unit 1 (a bias band applied per cell; seen on planted data, the gate passed):
  G2's "|P − N0| ≤ 0.25 · N0" was set to catch a biased null (the first proposal's N0 sat at half
  of chance), but applied to each of 23 cells it also counts each cell's sampling noise (P ≈ 0.008
  on 23k–45k rows). The gate's seed put the worst cell at 0.244; a scratch run on another seed, at
  0.287. The bias itself, pooled, is −3 to −5 %. The arm passed by seed, not by margin. Rule: **a
  band meant to test a null's bias is applied pooled, or set against the statistic's sampling
  spread at the real row counts; a per-cell extreme over a family is checked on more than one
  seed before the gate freezes**.
- 2026-10-22, C2 unit 3c (a tolerance set by one null unit): the null-matched centred arm matches
  the re-inits' uncentred violation rate at 1e-3. On GPT-2 small that rate is 0 (every re-init
  drop is negative), so the arm became the single largest of 252 re-init centred drops, 3.3× the
  next. The preflight gated it and it passed; the design table even noted it was "25× Pythia's".
  On 4 of 5 seeds the frozen label "consistent with step" survived only through that arm; at the
  next re-init value it reads "neither" (`/challenge-pr` on #258). Rule: **a tolerance matched to
  a null rate refuses when that rate is 0 (or rests on fewer than a handful of units), and a
  frozen widening rule is checked for which label it can hold up, not only which it keeps**.
- 2026-10-21, S2 unit 2 (a frozen gate that could not be built, and a guard that refused its own
  positive controls): G2's planted effect zeroes 10 heads, and its frozen controls were "doubly
  matched": the only edit with ‖D‖ = ‖K‖ and ‖K + D‖ = 0 is −K, so all 40 controls were the arm
  and G2 would have failed by construction. Found writing the code, before any run; the user
  chose a layer-matched head null. Then the gates showed the one-sided displacement guard
  withholding both planted effects (G2 at 3 of 4 steps, G3 at 3 of 5), because a structured
  edit moves the stream further than random edits of its weight norm; the guard became
  description (Blocked 88 (a)) before any arm ran. Rule: **before freezing, build each
  control for each planted effect on one real unit and check the matching constraints leave
  it any freedom; and a guard against a confound must be run on the planted effects too,
  since a guard that also fires on known effects reads nothing.**
- 2026-10-21, S1 unit 3 (a reading pinned at its bound; `/challenge-pr` on #245): R3's attractive
  share is ≥ 0, and at 1000–4000 every Haar subspace read exactly 0 in 18–20 of 24 layers, so
  "more repulsive than random" could not show and any attractive part counted as "less
  repulsive". Read first as a finding for QK. And S, drawn from the heads it compresses, made
  the OV families' Haar comparison biased by construction, which steps 0–128 cannot bound. Rule:
  before reading "outside the null's range", check the range is not at the statistic's bound,
  and that the object being compared was not selected from the data it is compared on.
- 2026-10-21, S1 unit 2 (a plant that changes what the data hold fixed): G4's additive plant
  raises each planted head's dimension, so tr P̄ rises and χ = (tr P̄² − E₀)/(tr P̄)² falls by
  2εχ₀ on that alone; on QK's concentrated pools (χ₀ 1.3–1.8e-3) that outweighs the plant at
  every dose to 64, and G4 failed there. Derivable before the freeze from unit 1's PR (344, 412);
  the design called χ "additive under a plant" without the first-order check. The singular
  families' data never change a head's dimension. Worse, `/challenge-pr` on #244 re-scored the
  saved draws without the dilution: 0.97–1.00 of the Haar response everywhere, so a random
  direction added to k heads overlaps any pool alike and the gate could not see the pool at all;
  only the dims-kept plant (G4r) depends on it. Rule: **a planted control changes only what
  the statistic is meant to see; list what else the plant moves (dimension, trace, norm) and
  expand the statistic to first order in it before fixing the bar; and check, on the stored
  draws of a dry run, that the plant's response differs between a null carrier and the real
  one, or the gate measures the plant, not the instrument on the data**
  (`trackS_subspace/status-S.md` "S1 unit 2"; math check fact 6).
- 2026-10-20, S1 unit 1 (a count at a threshold the spectrum is dense at; `/challenge-pr` on
  #243, finding 1): S1 counted P̄'s eigenvalues above the null's maximum, and G3 asked that 4
  planted directions raise the count by 4. At a trained step the eigenvalues sit about 1e-4
  apart through the threshold, and by interlacing a strong plant takes one slot while the
  remaining spectrum keeps N or N − 1 above it, a near coin flip, so no dose could pass. The
  pre-freeze read already showed 79–89 eigenvalues above the threshold at 8000; the preflight
  tool could not flag it, because R1 had one value per step, not a unit population. Rule: **a
  count above a threshold is gated only where the spectrum has a gap at the threshold; check the
  spacing there before freezing a count** (`trackS_subspace/status-S.md` "S1 unit 1").
- 2026-10-20, S1 unit 1 (a tolerance below its own metric's floor): G0 compared subspaces by
  sin θ_max = √(1 − σ_min²), which cannot resolve below √eps ≈ 1.5e-8, against a placed
  tolerance of 1e-8; a correct instrument read 5e-8 and failed. Replaced by ‖(I − UUᵀ)W‖₂ (1e-13),
  tolerance unchanged, before the full gate ran. Rule: **a tolerance is placed against the
  metric's own floor, checked on an exact case first** (`trackS_subspace/status-S.md` "S1 unit 1").
  Beside: the preflight tool's first design use (S1, same day) refused per-unit verdicts before
  the freeze, and the design changed to counts; the shape it was built for, caught in time.
- 2026-10-19, A3 unit 2c (a check that a correct answer fails half the time): the re-derived
  floor was the bare 95th percentile of the null's ref / σ, then checked at ≥ 95 % on fresh
  draws. A correct floor passes that about half the time; E2 failed it (0.936) and was refused,
  and the fix (the percentile's 95 % upper bound) came after seeing the failure. Rule: **a floor
  derived on draws and checked at the same level on fresh ones needs a margin (an upper
  confidence bound) chosen before the run** (`trackA_force/status-A.md` "A3 unit 2c").
- 2026-10-19, A3 unit 2 (a calibration on noise the data do not have): `a3_locator.py` calibrated
  the half-rise floor and power on iid noise of one size at every step. The real rows' spread
  grows 10⁴–10⁵-fold from step 2 to 4000, so at their own design the effect arm could not place
  early moves and E2–E5 were refused. Before a synthetic calibration fixes a constant, print the
  real rows' per-step spread and calibrate on that shape (`trackA_force/status-A.md` "A3 unit 2").
- 2026-10-18, B2's re-gate (a dose below the statistic's floor): `/challenge-pr` on #219 asked
  for G3's 9-step dose curve to start at the 70th percentile. With the two-sided |p − ½|, the
  null's mean is exactly ¼, so a set held at the 70th scores 0.20 and ranks below a typical
  random set. No number of steps could have seen it. A synthetic run before the batch showed this.
  The dose stayed in the run, labelled as below the floor (`tools/math_checks/b2_two_sided_floor.py`;
  `trackB_circuit/status-B.md` "B2 unit 1b"). Check a planted dose against the statistic's null
  mean before reading its 0 % as low power.
- 2026-10-17, B2 design (a null band that every trained value leaves): the proposed member
  statistic ranked "share of member cells outside step 0's 2.5–97.5 % band" against random head
  sets, and every gate arm ran at step 0. At d = 1024 a random head's attractive share
  concentrates: the band is 0.4963–0.5037, and 99.7 % of all heads at step 512 sit outside it, so
  members and random sets both score ≈ 1 and tie; the statistic could not have read "members
  differ" for any column (`/challenge-pr` on #218, finding 1; confirmed on the stored OVs). One
  read of the population at one trained step, before writing the rule, would have shown it.
  Rule: **before a null band is frozen, read where the population at a trained design point
  falls in it; a band that nearly everything leaves is a "moved" mark, not a null, and at least
  one gate arm runs at a trained step.**
- 2026-10-17, A2 unit 5 (a positive control that could not pass): the ablation rule needs the
  planted arm to *raise* the share of blocks where energy rises, but in the planted G1 stack that
  share is already 1.000 unablated, so (iii) fails by construction and the calibration fails
  with it (`trackA_force/status-A.md` "A2 unit 5"). One unablated pass, before the freeze, would
  have shown it. Rule: **before a rule with "raises X" freezes, read X on the planted control
  unablated and state its headroom**; a readout at its ceiling (or floor) is not a readout.
- 2026-10-16, B2a unit 2 (a rule fix gated on planted arms only, and a smoke test on real data):
  Blocked 51 (a) closed the reviewer's band with "positional only if TV(full) ≤ 0.05", checked
  against G2, whose planted profile M1 fits exactly. A real positional head is not fitted exactly:
  a one-head smoke test of `b2a.py real` (run to catch bugs before the 20-minute gate, since the
  gate binds the code hash) printed `L5H2`, B1's previous-token control, as "outside (content M1
  cannot express)" at TV(full) = TV(0) = 0.125, before the gate reran. Two rules: **a threshold
  that separates classes needs a planted arm at the realistic edge of each class** (an imperfect
  positional head, not only an exact one); and **a smoke test on real inputs prints no verdict**
  (run it with output to a file nobody reads, or on a planted input through the same path).
- 2026-10-14, Track B B0 (the other side: a rule that rejects a correct instrument): Blocked 43
  (a) made G4b an exact equality of rank-1 sets, nudged against linear, knowing a near-tie inside
  the linear error would fail it and leaving that rate "not derived". Over 18 steps it fired once,
  at 1 arm of 4,032 (margin 1.05e-4, linear error 1.4e-3, both of the arm's own projection), so a 3.5 h run ends in a waiver
  (Blocked 44). Rule: **an equality on ranks carries a tie clause sized by the error the same gate
  already measures, in one stated unit, written before the run**; a named failure mode left unbounded is a rule
  still open.
- 2026-10-11, Phase 10 LCo (caught by the calibration gate, before any real record): N0 failed at
  2.08 %, even on both sides, and no change to the statistic could fix it: the sign rule's own
  null rate is 2·2⁻ⁿ, 3.1 % with 6 readable passages, and 53 % of c3x's cells have 6. LC's N0 had
  passed (1.67 %) only because its floor removed about a quarter of the false labels under planted
  Exp noise, on the s scale; on the log scale the floor removes almost none. Rule: **a bar on
  false labels is checked against the label rule's own null rate at each cell's passage count,
  before any statistic is blamed**; here a full label now needs 7. Earlier rows' full labels on
  6-passage cells are parked with the skew audit (`handoff-10.md` "LCo done").
- 2026-10-11, Phase 10 LC (caught by the calibration gate, before any pass): F13's sign rule over
  passages (all same sign → a label) was run on a mean gap of a right-skewed quantity
  (surprisal). The smaller set's mean is the more skewed, so a zero-mean gap has a negative median
  and "all negative" fires about twice as often as "all positive": N0 gave 1.40 % lower against
  0.74 % higher, failing the 2 % bar. Residualising on position did not move it; a rank gap (a
  symmetric null) for the sign, the mean gap for the size, did (0.79 % / 0.88 %). Earlier rows
  that ran the same sign rule on mean gaps were not audited for this (parked, `handoff-10.md`
  "LC done"). Rule: **a sign rule takes its sign from a statistic whose null is symmetric; a
  mean difference of a skewed quantity between sets of unequal size is not one.** Same unit, a
  wrong assumption: the producer read `.logits` from `core.models.load_model`, which is the bare
  body; the runner refused at the first pass (no record written).
- 2026-10-10, Phase 10 OV1d (`/challenge-pr` on #184, finding 1): the matched-distance label
  interpolated `m` linearly along the far branch. Where `m` is convex in distance that line sits
  above the true curve, so a sign-blind `m(d)` on the run's own distances gives *merges* 7/7 at 4000
  L17–24; the rule had no null for its own matching. Taken by an amendment (distance-only curves
  beside every label, `status-10.md` §1.39). Rule: a matching or interpolating statistic is run once
  on a planted no-effect curve over the real design points before the rule freezes.
- 2026-10-05, Phase 10 R3: A0's "~94 % mask, learned residual from 2000–4000" (§1.1) was
  "confirmed" on a second sweep (§3.1). Both sweeps kept position 0 and one massive token per
  prompt in the unclustered population, and from step 4000 those two tokens carry the flip and
  the residual: out of the means, the late raw gap reverses (`status-10.md` §1.17). A replication
  that shares a confound cannot test it. Rule: before calling a population statistic confirmed,
  re-read it with the few highest-leverage tokens (the sink, massive tokens) out of the means.
- 2026-10-05, Phase 10 R5c: 5c's flip ("trained-specific in every model examined"), cited as
  motivating evidence for the v2 redesign, rested on random GPT-2 and ALBERT arms with no recorded
  run; neither `-random` entry could load through `run_1` before 2026-09-17 (`e191d77`). On the
  GPT-2 arm that now exists
  the raw flip is as large under random weights (all causal mask), and the trained-only part is
  position 0 (`status-10.md` §1.19). A control with no recorded run cannot be re-run, so its
  verdict was never checkable. Rule: a trained-vs-random contrast is quoted only with the random
  arm's run (model key, date, manifest); without one, it is a lead, not a result.
- 2026-10-05, Phase 10 R2: `design-10.md` froze F1's re-read rule as "negative iff median p ≤
  0.05 and the mean is negative" and stated §1.3's finding as "the 32–512 window". Applied to the
  published record itself, the rule gives negative from 32 to 54000: the window was a magnitude
  reading (−0.29 to −0.49 against −0.15 or smaller), so "does the row hold" was asked with a rule
  that could not reproduce the row on its own data. It surfaced only when c0 disagreed with the
  claim, after the run. Rule: before freezing a re-read rule, apply it to the published record
  and check it returns the published label (`status-10.md` §1.16).
- 2026-10-04, 1d unit 3, step 1 under option 4 (Blocked 19): the replacement gate (50
  Gaussians of trained clouds through (b) = `p_gauss` ∧ re-init rank) passed 0 of 50, but
  on the gating main arm it was also 0 of 50 **without** (b): no Gaussian held 3 stable
  points in a row (max 2), so the pass tests the plateau rule there, not (b); only the
  size-2 arm (29 → 0) tests (b). The option was written with the synthetic's 29 of 50
  in mind and no Gaussian of a trained cloud read first; it also costed the gate at
  "~1 min" before the batch, though its second term ranks against the batch's re-inits
  (found while fixing the design, before any run). Same rule as the next instance,
  applied to the replacement: **a gate placed to replace one that could not fail gets its
  own one-cloud read (event room, and the "without" arm) before it is chosen.**
- 2026-10-04, 1d unit 3, the real-input reader's step 1 (specificity on the step-0 inits):
  placed, reviewed (#138) and costed without looking at a step-0 cloud through the grid it
  would be read on. Step-0 tokens are nearly orthogonal (median cosine distance 1.006), so
  every merge falls in 2–3 of the 40 points of `r × median`: a 3-point plateau has room in
  0 of 56 clouds in L1–8, so "≤ 5 % of re-init clouds with a plateau" could not fail. The
  pilot the design's order row required showed it in 30 s (`z_G` at 1–4 of 40 points). Rule:
  **before a negative check's bound is placed, read one cloud of the reference set through
  the instrument and count how many of its points can carry the event; if few can, the
  bound tests the grid, not the rule.**
- 2026-10-04, 1d unit 3: the design's rule (b) read "more substantial clusters than the
  matched Gaussian at the same relative cut" as lumpier. On the multi-scale synthetic the
  Gaussian soon cuts into as many or more pieces (8–30) than the planted structure (3 or 10)
  once past the scales where it has none, so (b) could not have accepted a planted plateau
  at either scale. Caught by the
  first check before any real input, as it was built to do; the cost was a design fixed
  and reviewed (#129) with the tail never computed on a clustered cloud. Same rule as the
  2026-10-02 instance below: **a tail ("higher = lumpier") is computed on a known clustered
  cloud against its own Gaussian before a rule uses it.**
- 2026-10-02, 1d's position-keeping null, 11‴ form: the fix ("fit and draw on the
  un-normalised rows") was recommended on a synthetic whose rows all have similar norms.
  On trained Pythia, token 0's massive activation (~45× the median norm at L6–18) is then
  ~0.92 of every draw's noise, so trained L9–16 can never admit (0 of 56): a null that
  could not reject, on the cells it was built to read. The first real record showed it
  in seconds (per-draw maximum `excess` ~21 vs ~0.05). The same null passed its synthetic
  (0 / 10) and failed the real control (27–40 of 56 centred): the synthetic carried only
  the mechanism the null models. Rule: before recommending a change of space for a null,
  print the inputs' norm profile in that space on every checkpoint it will read; and a
  passed synthetic is a precondition, never evidence the real control will pass.
- 2026-10-01, 1d's position-keeping null (#127): its form was fixed, reviewed and given a
  pass/fail rule before anyone drew from it on data with a known answer. A synthetic opening
  showed that even the true positional mean, drawn as specified on unit rows, admits the
  opening in 7 of 10 seeds (the draw gives early tokens the pooled noise). The real run
  would have "failed the control" for a reason the rule could not tell apart from the
  mechanism. Rule: a null's form is frozen only after an oracle version of it passes a
  synthetic of the effect it is meant to absorb.
- 2026-10-01, 1d's identity-weights positive control (#125): the design put β = 16 and 64
  in the grid so the theory would have several clusters to recover, and kept one time grid
  (`t` ≤ 16) for every β, on the claim that the collapse time is "nearly free of β". The
  table cited for that stops at β = 5. The run's own (6.9) reference times were `inf` at
  16 and 64 and 21–24 at 8, so those trajectories could not move inside the grid, and the
  regime the control was built for was never reached. The numbers that showed it were
  computed and stored before the batch, and not read until the review (`/challenge-pr`
  on #125, finding 2). Rule: a time or sample grid shared across a parameter sweep is
  checked against the sweep's own reference scale at every value, before the batch.
- A0's first run: 400 permutation draws → largest possible merged e-value 10.01
  against a threshold of 20. It could not reject whatever the data said
  (§3.51.2).
- CLAIM-C: 8 prompts with ties → p cannot go below 0.0661 (§3.41).
- `core.evalues.combine` was a product and rejected 19.67 % under the null with
  dependent units (§3.51.2).
- Several registered nulls turned out invalid on construction: `73f566c`
  (CLAIM-B/P-I1), `a735c07` (P-ST1), `9510d8f` (P-I3), `98d168b` (F0).
- 2026-09-23: `handoff-10.md` §0.4 said Stages 1–5 re-run on all 20 prompts
  "for free" and, two lines later, that the point of Stage 0 was registering
  on prompts chosen blind. Now held out on 410m (`docs/PHASE_REVIEW.md`). The
  same PR then called the 12 "never seen"; `/challenge-pr` found `CLAIM-C` had
  already run them on 1.4b and gpt2-large (§3.46), so they are only partly blind.
  Rule: **name the confirmation set before new data lands, and list every run
  that has already touched it.** Status: 📋 until the guard (`PHASE_REVIEW.md`
  Parked 1) lands.
- 2026-09-24 (phase review session 9): the rule below reached the Phase 10
  runners, not the registered gates. `P-S1`'s gate defaults to 500 draws
  (too few for E ≥ 20), and its dry run said "the floor is attainable" because it
  tested p ≤ α = 0.05, which is E ≈ 2.2. `steering_sign.json` and
  `cross_head_association.json` use the same "clears α" test. Rule: a floor
  check compares against **E ≥ 20, i.e. p ≤ 1/1600**, not against α
  (`docs/PHASE_SYNTHESIS.md` §3.1).
- 2026-09-24 (Stage 1 step 1): `handoff-10.md` §1.1 called `ext_semantic` the
  project's "semantic instrument", and the threshold sweep's pre-stated verdict
  was built on a continuous cosine. On Pythia layer 0 the mutual pairs are a
  mixture: repeats of one token in a pile at cosine 1, the rest continuous and,
  through step 2000, below 0.32. So the 0.5 cut had nothing to act on early,
  and the all-pairs quantile cuts landed in the pile. The first write-up then
  overclaimed the other way ("says nothing about semantics") and misstated which
  cut drove each verdict. `/challenge-pr` on #89 caught both; the repeat /
  non-repeat split showed the remainder carries a signal. Rule: **before fixing
  a criterion on a statistic, histogram its input once**, on one run, and split
  a mixture before summarising it. A look at the distribution is not a look at
  the result.
- 2026-09-24 (Stage 1 step 2): the token-composition criterion was pre-stated
  on "non-repeat" tokens to drop lesson-6's cosine-1 pile, but "non-repeat"
  (no *earlier* copy) keeps first copies, which have twins *later* in the
  prompt. The criterion read "consistent" at 11 of 18 distinct checkpoints; on
  unique tokens it is "unclear" at all 18. Same rule as above, one level down: a confound
  defined by position in the sequence is not removed by a filter on order.
  Exclude on the symmetric property (copy count), not on "came first".
- 2026-09-24 (Stage 1 co-membership, #92): the pre-stated reading was
  Δ = lift(step) − lift(step 0), and the headline property scored co-members
  in the *trained* embedding. Step 0's clusters come from an unrelated random
  embedding, so their lift there is ~0 by construction; the Δ tracked the
  embedding converging, not the clusters. `/challenge-pr` caught it. Scored in
  each run's own embedding, step 0 is already +0.20, and the trained increment
  is +0.12, not +0.32. Rule: a baseline must be measured in a frame that could
  have shown the effect at that baseline.
- 2026-09-24 (§1.10, lexical vs contextual): the pre-stated control held
  embedding similarity to within a decile and read a positive "class beyond
  embedding" as not lexical. At trained layer 0, where the partition is built
  from the embedding, the same control read +0.18, as large as at depth. At 40
  bins it read +0.03, and the trained depth effect dropped from +0.15 to
  +0.04–0.07. Caught before the write-up, by looking at the L0 row. Rule: a
  control that claims to remove X needs a positive control, a place where the
  effect is X by construction. Run the reading there first. `/challenge-pr` on
  #93 then built the matched version (a kNN lexical cluster at the same layer:
  +0.21 under deciles) and found the carry table averaged 7 prompts in one
  column against 8 in the other (a "17 %" that was 5 %). A contrast between
  two groups is taken over the units that have both. The fix itself repeated
  it one level down (#94's review): the class-only reference returned None for
  small classes, so it averaged over fewer tokens than the value beside it
  (`homer_iliad` L24: 1 of 190). A test with 40 bins over a 40-member pool
  could not fail.
- 2026-09-25: Phase 10 §3's reproducibility floor (ARI p5 0.347) was pooled
  over 8 prompts and quoted for five days as the partition's floor. It was
  almost all one prompt, `repeated_tokens` (`status-1d.md` "Float-noise drift"). A
  tail percentile over pooled units reports the worst unit as the whole
  battery. Break a floor or tail down by prompt before quoting it.
- 2026-09-26 (#104, 1d merge tree, first version): "splits almost absent,
  merges at L0→L2" was produced by the instrument twice over. The
  longest-lived plateau at layers >= 2 is one cluster with 91–96 % of
  tokens plus outliers, and a Jaccard >= 0.1 link cannot register a piece
  leaving a large cluster (1/460), so every split it could have seen was
  a birth; the L0→L2 "merges" were the pick jumping from token identity to
  the blob. The PR's own 101-token test encoded the blindness (a straggler
  labelled "birth") and passed. `/challenge-pr` accepted the claim; a
  second read of the saved labels' cluster sizes caught it. Rule: before
  reporting counts of an event type, **print the sizes of the units the
  counts are over**, and build one fixture where the event must occur
  (here, a 4-token piece leaving 60) to check the instrument can see it.
- 2026-09-26 (1d link null, #105): #104's revised headlines, "merges
  outnumber splits at L0–L7" and "tangles are the most common change",
  were reported before any null and did not hold up against one
  (`status-1d.md` "The link counts against a size-preserving null…").
  The first write-up of #105 then overclaimed twice. It called beating an
  independent null "persistence survives", when neighbouring layers carry
  over by construction. It called the merge excess "caused by" the falling
  cluster count, when the null's event mix differs from the real one (292
  births vs 48) and only failed to separate the two. `/challenge-pr` caught
  both. Rule: **no depth pattern is reported before its null has run, and
  a null that anything with carry-over beats supports no positive claim.**
- 2026-09-26 (1d Gaussian null): three cases in one unit. (a) A matched-covariance
  Gaussian cannot make duplicate vectors, so any tokenised text beats it.
  The untrained model beat it harder than the trained one, which was a
  sign of token identity, not of structure. Deduplicating strings removed all of step 0's
  excess (`status-1d.md` "Matched-covariance Gaussian null"). (b) The
  module's docstring called the plug-in covariance's bias "conservative",
  reasoned but not measured. A calibration run (the null applied to its own
  Gaussian draws) found `ci2`'s lower tail firing 25 % in the centred
  frames, not 2.5 %. (c) The deduped results were then read against that
  calibration, which had run on all tokens pooled over 5 layers. It was on
  different inputs, and it fires 35 of 56 at L17–24. That hid a late 2-means excess and
  produced a wrong "nothing global" headline (`/challenge-pr` on #106).
  `gaussian_null_report.py` now refuses a calibration on other inputs.
  Rule: **a new null gets a calibration run on data where it is true, on
  the same inputs and bands as the result it will be read against, before
  any real result is read, and a negative control
  (here step 0) that should not beat it.**
- 2026-09-30, the tests themselves: mutation testing of `core/evalues.py`
  (the e-value core) killed 303 of 413 mutants on its first run (#117; the
  current count is in `STATE.md`). The test of the helper
  `simulate_type_i_error` asserted only `rate <= alpha`, so a helper that
  never rejects passed it; three mutants did exactly that.
  `EProcess.from_record` could ignore the stored alpha and kappa, and
  `next_p_needed` the evidence already accumulated, with every test green.
  The same pattern one level up: a check that cannot fail is not a check.
  Fixed by known-answer rates (`tests/test_core_evalues_contract.py`). #117
  pinned them only on the helper, which re-derives the calibrator in numpy.
  After `/challenge-pr` on #117 they are pinned through `EProcess` and
  `average_p`, the code that scores. #117 also said tier 0's ledger replay
  is built on `from_record`. It is not: `core/adjudication.py` rebuilds
  through `EProcess` + `add`, and no production code calls `from_record`.
- 2026-09-30, a kill by chance: `test_the_simulations_are_seeded` compared
  two rates of 2000 booleans. An unseeded helper matches itself about 4 % of
  the time, and in one run it survived. A mutation run is stochastic wherever
  a test is. Fixed with five seeds at alpha 0.5; two consecutive runs now
  give identical states. Rule: **a test that kills a mutant only most of
  the time is a flaky test, and gets fixed like one.**
- 2026-09-30, #121 (design only, caught before code): 1d's per-group test
  named hdbscan's `cluster_persistence_`, which divides by the largest λ in
  the whole tree. Real layers have far tighter nearest neighbours than their
  Gaussian null, so every real group would be scored down against null groups
  that are not: a test biased toward "nothing beats the null". The planned
  synthetic check (one planted group) could not have shown it, since that
  group sets the maximum itself. `/challenge-pr` found it; a planted pair
  confirmed it (0.75 alone, 0.012 beside a tighter group). Rule: **read how a
  library normalises a statistic before testing it against a null, and the
  synthetic check must include a second, unrelated structure.**
- 2026-10-01, `admit.py`: that rule's check paid off at once. With the second
  structure added, the looser group's `S_C` still moved (3.6 %), because
  hdbscan's tree orders *tied* edges by processing order, and with mutual
  reachability ties are the rule. The project had read hdbscan's labels as a
  function of the geometry since Phase 1; on deduped v1 tokens ~40 % of its
  groups exist only through tie order (`status-1d.md` "Admission"). Rule: **a
  library's output is a function of its input only where the input has no
  ties; check how ties are broken before scoring anything it builds.**
- 2026-10-01, `position_check.py` (#123): "about half of trained groups are
  runs of nearby tokens" came from a per-group significance flag (`p_near` ≤
  0.05), and that flag fires on slight tilts in large groups. Only 1–14 % of
  the flagged admitted groups were mostly near pairs. `/challenge-pr` caught
  it. Separately, the cut M = 32 was chosen on the step-0 control, so "the
  control passes at 32" held by construction. Rules: **a share of groups
  passing a test is not a share of groups that have the property; report an
  effect size beside it. A control used to tune a choice has not tested that
  choice.**
- 2026-10-01, the M = 32 cut on the long prompts: that rule's test, run with its pass rule
  committed first, failed. Step 0 admitted again at length, and the groups it admitted held
  position 32, the first token the cut kept. M = 8 on v1 had already shown the opening moving
  with the cut, and 32 was still read as "removes it" because at v1 length nothing was left to
  admit. Rule: **when an effect is anchored to a boundary the fix defines (the first kept
  token), moving the boundary moves the effect; check at the new boundary, on data the fix
  was not tuned on, before calling it removed.** (The first write-up then overstated the
  other way, "whatever M is"; the effect also weakens with M, `/challenge-pr` on #126.)
- 2026-10-02 (#129, unit 0 design, caught by `/challenge-pr` before any run): unit 2's
  fallback to the 10 real inits kept three "rank p ≤ 0.05" rules, whose floor at N = 10 is
  1/11 = 0.091. Every cell would have read "not learned". The same design compared raw
  `S_C / |C|` (units of 1 / distance) between trained and random-init clouds, so it would
  have measured tightness, not groups. Rule unchanged: **compute the floor of every rule
  under every fallback the design names**, and before comparing a statistic across
  models, write down its units.
- 2026-10-02 (#130, unit 1, move the text; caught by `/challenge-pr`): "moves" was "best
  Jaccard ≥ the group's own subsample floor J0", and a group could be stable (median ≥ 0.5)
  with J0 = 0, which no match can fail: 336 of step 0's 412 "moves" were such groups. The
  author then explained the inflated step-0 share with a story (the preamble barely moves
  the states, median cosine ≥ 0.96, so "moves" is the default) and wrote it into STATE and
  the PR as the main caveat. With the floor fixed, step 0 moves 0.08–0.28 by band against
  trained 0.74–0.95, at equal cosine. Rules: **a rule that compares against a data-driven
  threshold has a test for the threshold's degenerate value (0, 1, empty)**, and **a number
  that surprises gets its producer checked before it gets an explanation**. Same unit,
  frozen designed-prompt rule 1 measured purity over labelled members only, so step 0's
  ~20-token opening group, holding the three early "teacher" names, counted as 13 entity
  content groups: a purity rule names its denominator.
- 2026-10-02 (#132's answer, retracted by `/challenge-pr` on #133): "the replicating groups
  are tighter in position than chance, so they are not simply the embedding's classes". The
  embedding has no position, but text does: groups carried unchanged from L0 are tighter
  than chance too (median spread percentile 0.23). The test could not have told the two apart,
  and nobody ran it on a known embedding class first. Rule: **before reading a statistic as
  evidence against X, compute it on a case known to be X**.
- 2026-10-04 (#134's pinning test, found while re-running unit 3): `TestOnAKnownClusteredCloud`
  asserted "(b)'s count tail rejects both planted scales" with 10 Gaussian draws, so the
  smallest rank p was 1/11 = 0.09 > α: no tail could have accepted anything, and the test
  pinned the floor, not the direction. At 20 draws the count tail accepts the fine scale (where
  the Gaussian has no cluster) and rejects only the coarse one. Rule: **a test that asserts a
  rejection asserts first that acceptance was attainable** (`1/(N+1) <= ALPHA`).
- 2026-10-04 (unit 3, three fails in one day): the synthetic's spreads were placed "so each
  planted plateau spans ≥ 4 grid points" from construction distances, and `MIN_RUN` = 3 was
  set beside it; nobody measured how many cuts actually reproduce each planted partition.
  After the opening it is 3–4, so every rule since sat at the grid's resolution and a seed
  passed or failed on where grid points fell (seeds 2–11: 7 of 10). Rule: **a positive
  control's margin is measured in the instrument's own units (here, grid points over which
  the cut equals the planted partition) across seeds, before a run-length bar is placed**.
- 2026-10-07 (1e U2, #158): the frozen rule's token-shuffle null and its `Xt` "mean move out"
  were written as the control for "any update component common to all tokens", but neither
  can see one: a shared vector's tangent projections track the field's mean term, so a pure
  shift reads X = ±0.03–0.18, hundreds of null sds out. #158's headline (ascends 8–256,
  "repulsive" descents) rested on it. The tests checked only that a field step reads ±1
  (sensitivity), never that an unrelated move reads 0. Found by `/challenge-pr`; its own
  fix (the unit-frame mean out) was also unspecific (ascends at step 0 everywhere), caught by
  reading it at init. Rule: **every reading gets a specificity test (a move unrelated to the
  quantity reads ≈ 0) beside its sensitivity test, and a proposed control is first read at
  init, where the answer is known**.
- 2026-10-08 (1e U1): the rule asked "what are the 2–4 wells" (labels (2) and (4), c3x purity)
  from a probe the design itself said was on the wrong frame. In β's frame at β 1.6–5.6 the
  field has **one** well in every trained band (and its Gaussian too), so (2) reads `Xw = 0`,
  which the sign rule calls "mixed", indistinguishable from a real tie, and (4) and the purity
  are not read. One mean shift per step in the right frame (seconds) would have shown it before
  the rule was written. Also the rule asked β-robustness of a label whose null was drawn at one
  β only (caught after the first check, before any reading). Rule: **before a rule labels an
  object (wells, groups), check in the rule's own frame and parameters that the object exists,
  and give "the null and the cloud agree exactly" its own label, not "mixed"**. Same unit
  (`/challenge-pr` on #164, finding 2): the matched Gaussian, put on the sphere after drawing,
  scores a structureless cloud `Xe` up to +0.10 in trained cells, half the late raw value; the
  previous instance's rule (a specificity test) was not applied to the new null. Rule unchanged,
  applied: a structureless cloud reading ≈ 0 is now a test.
- 2026-10-09, 1e (b) and U3 at β 10 (`/challenge-pr` on #167, findings 1–2): `Xw` and `Xp` subtract
  a matched Gaussian's log well count or log persistence, which is 0 wherever the Gaussian has one
  well. That held in 2,504 of 3,128 and 336 of 408 cells, so "fewer" or "shallower" was nearly
  impossible, and the first write-up read "shallower in none" as a finding. The step-0 rule compares
  signs, so a stage that doubles a value already positive at init reads "not learned", while L20
  read "learned" only because half its passages sat at the floor at init. Rules: **count the cells
  where the null sits at its floor before reading a signed difference against it; and where step
  0 carries a sign, read the paired difference to init beside the label.**
- 2026-10-09, Phase 10 R9 / R6f: the random-drop reference matched c3x's drop count per record,
  so in the 72 records c3x empties every draw had to drop the same groups (185 of 382 drops).
  The 3 cells every draw changed were c3x's own changes, and the first write-up read them as
  labels unstable to the set's edges (`status-10.md` §1.32; `/challenge-pr` on #174). Rule:
  **before reading a reference, count what it holds fixed by construction (here: records where
  the matched count is all of them), and set the cells it forces apart.**
- 2026-10-15, A1b (`/challenge-pr` on #206): the note written before the output said "a drift can only
  add to `T_proj`", from the gate's one planted drift (sustained, toward a target). In the real
  stream removing the shared part raised `T_proj` in 19 cells. The same unit read 1e's question
  off (6.9), a model with one cluster by construction, while the design's own causal table
  already put the doubling β inside the measured range. Rule: **a sign or bound shown on one
  planted case is that case's; before it goes into a reading, check it on the data it will be
  read beside (here one line: `T_proj_c > T_proj`).**

**The rule now.** Compute the **attainable floor** (best possible p / max e)
of a design before running it, and print it on every record
(`max_attainable_E`). Status: ✅ in the runners since §3.51; for new designs, ✅ in part:
`tools/design_preflight.py` (2026-10-20) takes a statistic's values on its nulls, the per-unit
trained population at one step off the scored grid and a planted control at dose 0 and three
doses. It reports the floor and the detection curve, read at the p one factor needs for E ≥ 1/α
unless the design names its own level. It refuses on five of this lesson's shapes: a band
nearly everything leaves, a control at its ceiling or reading an effect unplanted, a dose not
past the null's mean, a floor below 1/α, and a largest dose seen < 80 %. Its tests replay B2's
band, A2's control, B2's re-gate dose and `P-S1`'s 500 draws, and A3 unit 2's *shape* (not its
cause). ⚠️ It does not check the others: calibration noise shaped unlike the data (A3 unit 2),
a floor checked at its own level (A3 unit 2c), a bound shown on one planted case (B2a), a
comparator's bias (E1m, E1p) (`/challenge-pr` on #242, finding 3). 📋 "A design cites its
report before it freezes" is a `CLAUDE.md` rule, so the user's (STATE Blocked 80).

- 2026-10-09 (E1m, `/challenge-pr` on #177): a comparator built to give "a pull towards near tokens
  scores 0" was never shown to do so. Its test planted a nearest-8 pull and asserted |X| < 0.15,
  ten times the headline cells' size; the planted X was +0.14 (and +0.24 for a pull towards the
  cloud's mean). The pairing matched similarity to 0.005 and still was not a null: tokens at equal
  similarity to i are not in the same direction from i. Found by the reviewer, after the run and
  the PR text had called the cells "membership". Rule: **a comparator's rule lists its planted
  null and the size of its bias (not a bound) before the run, and no label is worded as the effect
  until a pseudo-group or placebo of the same pairing has been scored.**
- 2026-10-09 (E1p, Blocked 31 (b)): the pseudo-group placebo that #177's write-up recommended as
  "the only way the pairing's labels can be read as membership" was itself never calibrated. Built
  on the planted clouds first, it scored group-blind pulls +0.15 to +0.51 (and +0.02 to +0.09 when
  the group is a density blob): a nearest-neighbour ball sits off its cloud's centre where a group
  need not. The user had picked it on that recommendation; it was replaced (the user's call) by a
  projection null before any real move was scored. In the projection's own tests, a full-view pull
  scored in the causal view gave +0.14 to +0.33, which is a wrong planted null for that view, not a
  bias in the instrument. Rule: **a null proposed to the user is first run on planted geometry (at
  least two shapes, each view's own pull) and its planted bias quoted beside the recommendation.**
  Then the projection's own calibration repeated the mistake one level down: both planted shapes
  had 2–5 % of fellow similarities above every non-member, while real c3x groups are tighter than
  their nearest non-members in 95–98 %. In that geometry the projection leaks (`X_p` up to +0.14),
  and the write-up had read E1m's cells as "the group-blind part" and an L1–8 push as surviving
  (`/challenge-pr` on #178, finding 1). Rule: **measure the real data's statistic that drives the
  bias (here: fellows above every non-member) and plant at least one shape that matches it.**
- 2026-10-10 (F13): three placed choices in the rule could not work on the real clouds, each
  found by a refusal before any statistic was read. (1) F15's primary `L_def` = 1: the scale check
  read every token's nearest-predecessor distance, repeats included, but the statistic reads the
  kept (unique) tokens, and at c = 2 all 213 of them are strong centres at L0–1, leaving nothing
  to cover. (2) A per-record 3-SE check on the null's mean, and then a 1 % pooled share: at
  ~56k records the first refuses ~150 correct ones by chance, the second ~5 of 252 cells. (3) Its
  SE floor of 1e-12: where the exact mean is 0.006, 200 draws are all 0 (sd 0) and a correct null
  reads z ≈ −6e9. Rule: **a scale check reads the population the statistic reads, and a check run
  many times sets its bar from its own false-refusal rate (a binomial quantile), with a floor
  for counts whose draws can all be zero.**
- 2026-10-10 (F13r): the rule froze "survives frequency" on `S_c` inside lf-tercile strata
  without a planted no-effect case, the OV1d rule above. The tests ran one after the freeze: with
  centres in the rarest 8 % and log ρ = lf alone, the strata gave −0.91 (negative in 20 of 20 draws)
  and would have read **survives**. Amended before any record was read (a linear `S_c~x` must
  agree). Rule: **the planted no-effect case for a matching statistic includes the covariate's
  tail (the matched set concentrated in one extreme), and runs before the rule is committed.**
- 2026-10-14 (B0, G4b): the gate's structured null (another prompt's loss gradient, ranked among
  32 random directions) assumed an unrelated on-distribution direction ranks like a random one.
  At 143000 it ranks first 17 times in 224 (bound 12), and the gradients alone give the same 17:
  loss gradients share a direction across prompts (mean |cos| 0.031 against 0.025). The arm
  measured the model, not the harness; it had no planted case of "a correct harness on aligned
  directions". Rule: **a null meant to test an instrument is checked against what the
  instrument's own linear theory predicts for that null before it is given a pass bar.**
- 2026-10-14 (A1a, G3): the design set 5 % on the recovered step for planted steps up to h = 0.5
  without computing the estimator's own bias: the tangent part of a retracted step is
  `V/√(1+|V|²)`, 6.4 % low on the largest. The second-order term was knowable from one line
  of algebra; the gate found it instead (`trackA_force/status-A.md` "A1a's gate"). The same
  instance as B0's G4b, one order up. Rule: **a recovery bar is set after the estimator's
  leading bias at the planted range is computed, and is quoted beside the bar.**
- 2026-10-15 (A1a, G2s, Blocked 47 (d)): a bar (|T_c| ≤ 0.1 T) was put on the shared-drift arm
  without asking what the drift does to the cloud over 23 steps. Held toward one direction, it
  collapses the cloud onto that direction, where the field points too, so the "null" becomes a
  planted collapse (first step 0.01 of h, last 0.80–0.99); 38 of 42 cells failed. A look at the
  arm's own trajectory (its mean cosine per step) would have shown it before the bar was set.
  Rule: **a planted null is checked to stay a null over its whole trajectory (the statistic
  per step, not only its sum) before it gets a bar.**
- 2026-10-15 (B2a, unit 1's gate): two planted arms were written without simulating them. G1
  ran the paper's own map to put tokens near a planted repulsive plane, but at the design's ν = 2
  the plane's `|b| = 0.1` is below the kernel's 0.7 (the paper's selection rule, read in the same
  scan), so the map moved tokens off it; at ν = 8 they collapsed and D\* = 0 was the right
  answer. G3 used isotropic tokens, whose attention at β = 4 is near uniform, so D = 0 passes on
  its TV clause and "none" was not reachable. Both were caught before any real token was read
  (a 20-line scratch run; the first G1 run) and rebuilt (`design-B2a.md` "Unit 1's gate, as
  built"). The same instance as A1a's G2s. Rule: **each planted arm is run once on a scratch
  slice, and its configuration checked to be the one the arm names, before the gate's table
  freezes.**
- 2026-10-16 (B2a unit 3, Blocked 53 (a)): Claude's recommendation placed a selection bar (a step
  is read where any head's U₂ share is above twice random) on a statistic whose value at 143000
  was already in hand: `L5H2` 5.3×, `L18H1` 29× (unit 2's own table). The bar was met there, where
  the rule had read no circle, so it could not tell "the premise holds" from "noise in one of 14
  heads"; it selected 7 of 18 steps, mostly at 2.0–3.5× in one head. Cost: about 35 GPU minutes
  that a bar set from the known values would have spent the same way or not at all; no verdict is
  wrong. Worse, `/challenge-pr` on #210 found the bar's "random" was the mean share D/d while the
  statistic is a median, whose Haar value is 0.69 × D/d (D = 2), so the 11 skipped steps were
  called "at chance" on the wrong baseline, and unit 2's "below random in 11 of 12" with them; step
  0 (random weights) read 0.79 and would have shown it. The rule was then run at all 18 steps
  (61 GPU min). Rule: **a bar placed on a statistic is checked against the values already recorded
  for it (does it pass where the answer is known to be negative?), and its chance level is the
  statistic's own under the null (simulated or read off step 0), not a formula for a different
  summary, before it goes to the user.**
- 2026-10-18 (B3 unit 1, Blocked 66): a design frozen after one `/challenge-pr` round failed its
  own gate on three things a type-score table would have shown before the freeze: the four
  "induction members" do not share an offset (two at 0, two at +1) and `L12H5` is not an induction
  head on 7d's probe at all (peak 0.011); R2's "bottom decile of its step" is met at step 2 by
  init structure; and C3′'s null (Claude's choice, a pool head per step, re-centred on each fake)
  was not exchangeable with the fakes (6 % false pass). Cost: one unit's gate run (about 6 min)
  and a decision round; no member α read. Rule: **before a design that names a head type freezes,
  tabulate each member's type score at every candidate offset; and a null built from "nearest"
  pools is gated with fakes drawn from the same fixed pool as the null, with the tested head kept
  in that pool, so the fake is exchangeable with it** (`/challenge-pr` on #224: leaving it out
  gave 3.0 % and 5.5 % false passes).
- 2026-10-18 (B3 unit 1b, #225): following that rule literally, G2 gave each C3′ fake the *member's*
  pool. The read gives the member a pool built around the member, so the gate tested a different
  construction from the one it gates. G2 recorded 0.5 % on 200 fakes. With each fake's pool built
  as the member's, 2,000 fakes give 1.5 %, still under the 2 % cap (`/challenge-pr` on #225, finding
  1). Also, a fake's R1 and R2 pool (its whole layer) held the real member. Cost: one fix round; no
  reading changed. Rule: **G2 calls the read's own pool builder on each fake (one function for
  both), and a false-pass rate near its cap is quoted from enough fakes to separate the two (200
  cannot tell 0.5 % from 1.5 %).**
- 2026-10-19 (A3 unit 3, #235): E11, "the simulator's final cluster count", was constant by
  theorem. With V = Id the causal dynamics end in one cluster from almost every start (Theorem
  4.1 of 2411.04990, read into `docs/readings/2411.04990.md` long before), so a final count is 1
  at any long horizon. Claude placed t = 200 (A1a's `T_MAX`, 30–80× A1a's t*) and blamed the
  horizon at first; `/challenge-pr` found the theorem. The count was 1 in all 126 cells, and 95
  GPU minutes went on a row that could not move. Its one signal (clouds stay split longer
  from 8000) was already in A1a's t* records. Rule: **before a row freezes, ask what the
  repo's own theorems say its value is in the limit; a statistic fixed by a theorem is not a
  row. Also, a quick slice that already shows the final value (the `--quick` cell read 1 from
  t = 20) stops the batch.**
- 2026-10-20 (A3 unit 5): #236's review re-worded E6 as "the step where the model's loss passes each
  predictor's own", and Claude adopted it on a check run on the same 2M tables (48 and 55 of 64
  blocks). That check could not reject the other reading, that the minimum belongs to the model's
  trajectory and not to the table. Blocked 74 (a)'s recount (a bigram 1.24 nats sharper) kept the
  minimum at 256 while the crossing moved to 512 (a match in 7 of 64 blocks), so the wording went
  into the card and was corrected one unit later. Cost: a correction line; no reading changed.
  Rule: **an explanation of where a statistic lands is checked against a variant that moves the
  proposed cause (here the table's quality) before it goes into the card; a check on the inputs it
  was built from is a description.**
- 2026-10-21 (S2's design, before any code): the controls were matched to each arm in edit size
  ‖K′ − K‖ only. A projection arm (K → sym K) shrinks the head to ≈ 0.71 ‖K‖, while an added random
  edit of the same size grows it to ≈ 1.2 ‖K‖, so arm and controls differed in output gain on all
  384 heads, in one direction. The planted null was a 41st control draw, exchangeable with the 40
  by construction, so it could not catch that. Also, λ → Re λ "keeps the radial part" only in a
  non-orthogonal eigenbasis, and on 410m at 512 it edits a head by 1.6× its own size. `/challenge-pr`
  on #246 found all three by loading 15 real heads. Cost: a design revision before the freeze; no
  run. Rule: **match a control to the arm on every norm the readout can feel (here the edit and
  the edited head), and make the planted null an arm-shaped edit that removes nothing specific
  (here the gain change alone). A null drawn from the control distribution only tests the
  labeller. Before a design names an edit's size, measure it on a few real units.**
- 2026-10-21 (S2 unit 1, the preflight): the first input gave `tools/design_preflight.py` the base
  model's 7 per-prompt violation counts (7–10) as `trained`, against a null of 7-prompt sums
  (30–60), so its band check refused arm b on a scale mismatch. Caught reading the refusal's band
  before writing it up; re-run offline on the records (`s2.py preflight-tool`). Arm a's refusal
  survived on the right scale and is real (base 59 outside its controls' [31, 39]). The same run
  showed both arms move the stream further than all 40 controls at 8000: as frozen, the
  displacement guard would have left R1 and R2 unread at every cell, found here before any scored
  step, which is the preflight's job (Blocked 87). Rule: **every population given to the
  preflight is on the statistic's own scale (the same sum, the same units) as its null.**
- 2026-10-22 (S2 unit 3, R3): the pairs "in S" were selected at a Haar plane's 99th-percentile
  share in R¹⁰²⁴. That was the wrong null, not only an unchecked one: a pair's plane lies in its
  head's 64-dim write space, which overlaps S, and a random plane there sits as high as the pairs
  (`/challenge-pr` on #249). So the rule took 37–69 % of the model's pairs, bR (as many of each
  head's *other* pairs) could not match the count, and the code degraded to fewer pairs instead of
  refusing; R3 was refused after review. The preflight ran on V, the reading's statistic, not on
  the selection that defines the arm. Second, R3 asked whether S's pairs carry what R1 and R2 read,
  and was run after R1 and R2 read nothing under arm b. Rule: **a selection's null is drawn from
  the space the selected objects already live in, it is run on the real population before the
  freeze, and its selected fraction is stated in the design; a control that cannot match its arm
  refuses; a reading conditional on an earlier effect is re-scoped when that effect is not shown.**
- 2026-10-22 (S2's close-out, `/challenge-pr` on #251, finding 3): the close-out dropped three R3
  variants because "a subset has no effect to carry" (arm b on every pair was as its controls in
  unit 2). The same reason held before unit 4, and Blocked 90's recommended (a) said so; unit 4
  ran by the user's (b) and read as predicted (as its controls in all 9 cells): one unit spent. The
  rule above stood. What it lacked was a place to bind: S2's design wrote Q3 unconditionally, so
  every re-scope of it was a separate decision. Rule: **a design states, at the freeze, which
  readings are conditional on which earlier ones, and what "not shown" upstream does to them**.
- 2026-10-22 (C2 unit 3, the anisotropy control): Blocked 95 (a), recommended by Claude and taken
  before any centred number, labels "mostly removed" when the centred plateau is below half the
  uncentred one, with both counts taken at D2's relative tolerance 1e-3. But centring compresses
  E₁'s range (a centred cloud's mean cosine is near 0), so the same change is a smaller relative
  drop, and the ratio is the tolerance's as much as the data's: 0.90 at 1e-4, 0.35 at 1e-3, 0 at
  1e-2 (labels "moved", "mostly removed", "removed"). The preflight had the populations to show
  this (step 7000's per-unit drops), but its table was read at 1e-3 only. The finding that holds
  at every tolerance (at 512 the centred count is 4–11 against 56–60; its date is 1000) came from
  the sensitivity rows, which the design kept beside. `/challenge-pr` on #255 then measured the
  scale directly (the centred per-transition change is about 30× smaller) and found that the
  preflight had no positive control for the ratio: plant B was read on the centred count only, so
  no plant had a known ratio of 1. Rule: **a label that compares two statistics is given, in the
  preflight, a plant read on both whose true ratio is known (1 for a change both should see), and
  their per-unit scales are stated before the freeze; a size condition reads a scale-free quantity
  or a tolerance matched to each statistic's own spread**.
- 2026-10-22 (C2 unit 3s, caught by the gate, before any seed record): the rule above did its job
  and showed its own second half was not enough. 96 (b)'s tolerance "matched to the centred
  statistic's own spread" (the median per-unit change, 2.81e-5) failed the plant read on both
  counts: plant B turned 2–5× more units centred than uncentred. A sweep from 1e-5 to 1e-3 found
  no tolerance that reads plant B alike at more than one dose, because the two counts' dose
  responses have different slopes. Matching spread, null rate or one plant dose gave step 7000
  ratios of 1.0, 0.93 and 0.44. The rule became dates only, each on its own count's plateau, at
  four arms (Blocked 97 (a), the user's). Rule: **two counts are comparable in size only if one
  tolerance reads a planted change alike through both at two or more doses; if none does, read
  each count on its own scale (a date, a half-rise), and give the matching criteria as arms, not
  one choice**.
- 2026-10-22 (C2 unit 3b, caught before any record, twice): 92 (iii)'s rule, frozen at unit 0,
  read "the same step on all three sizes" as "step-locked" and "earlier on the higher-LR sizes"
  as "dose-ordered". Computing each hypothesis's prediction on Pythia's grid showed they overlap:
  a break that waits for a cumulative LR dose moves 70m's crossing from 410m's (128, 512] to
  (70, 280], which the grid reads as 128, 256 or 512, and every date step-locked predicts,
  cumulative predicts too. The rule was re-cut as the schedules consistent with each date
  (Blocked 99 (a)). Then unit 3s's gate, moved to 70m unchanged, refused every arm: its plant was
  sized in token norms, and 70m's layers move about 5× more than 410m's, so the plant was small
  against them (Blocked 100 (a): the dose set where the reference count sees it). Rule: **a rule
  that labels outcomes by hypothesis is frozen with each hypothesis's predicted outcomes on the
  real grid (a check script), and an outcome two hypotheses both predict is labelled with both;
  a plant moved to a new model is sized against that model's own per-unit scale, not carried
  over in absolute units**.
- 2026-10-24 (C5 unit 2, caught by the gate, before any scored record): "a few coordinates hold
  half a fall in ‖m‖²" was frozen as an absolute bar, k½ ≤ 16 of 1024, on the *net* fall. The
  pure tests built their spread fall as a scaling, every coordinate falling, so they passed. Real
  falls at step 7000 turned to random bases read "few" at 0.65: coordinates move both ways, the
  net is small against them, and about 11 coordinates hold half of it in any basis. The
  rotation-relative reading beside it passed both sides (Blocked 106). Rule: **a concentration
  statistic on a signed sum is gated on the real vectors turned to random bases before its bar
  is chosen; a pure test's "spread" case must have mixed signs, as the data does**.
- 2026-10-25 (B7′ unit 1, caught by the code tests before the gate, and by the gate): the design's
  confound arm G10 (b) had to make B7's instrument read "depth-local" over a plant that put the
  sink in 2 prompts of 7, but the interval is a bootstrap over prompts: (5/7)^7 = 9.5 % of draws
  hold neither, so the 2.5 % bound sits at 0 whatever the sink does. A second arm, G11, required
  "frozen across depth" with a sub-threshold key confined to layers 7–20, and the two cannot both
  hold. Both were written and reviewed (#278) as text; neither reviewer computed what the plant
  gives under the instrument's own resampling. Rule: **before a design freezes, each planted arm's
  pass condition is checked against the instrument's resampling unit (here prompts): count the
  draws that miss the planted structure, and keep any added nuisance out of an arm whose
  condition has no noise to absorb it**. Same unit, found by #279's /challenge-pr: three arms
  whose condition is "rerouting with the drop" (G10 (c), G13, G14 (c)) would mostly have passed
  without the drop, since the instrument reads "rerouting" there anyway, and the write-up listed
  them as evidence for the rule. Rule: **an arm that passes with the treatment records its rate
  without it beside; an arm with no such contrast is a plant check, not a test of the rule**. And
  Blocked 115 shipped with 114's option text pasted on its end (two "Recommended" lines on one
  decision): **re-read the Blocked row you write as the user will, end to end**.

## 7. Literature found after the construction was built

**Instances.** §3.22's self-repair framing turned out to be CoAx (published
2 months earlier), corrected same day by §3.28. Phase 1's framing ignored a
causal-mask theory that existed since Nov 2024 (`2411.04990`), which Pythia
should have been compared against all along. §3.52: reading that paper changed
three constructions, and F0 had measured a proxy.
2026-10-15: `design-A.md` froze B2a's question on "2604.01978 §General value maps and effective
low-dimensional dynamics"; no such section exists there. It is 2609.28448 §6.1, whose LaTeX was
already in `docs/readings/`, and it puts the reduction on V's invariant subspaces with Q = K = I,
which changed B2a's primary object. Found at B2a's scan by fetching the cited paper before
designing on it. Rule: a design that quotes a section by title names the file it was read from;
a `grep` of `docs/readings/` for the title is the first step of the scan.
2026-10-24 (#270): the correction above, and B2a, B3, ROADMAP and C2's new citation after it,
numbered 2609.28448's sections wrongly (§6.1 for Sec. V A, §6.2 and §7 for V B, and four more);
the paper is REVTeX and numbers them in roman, and the repo's numbers match no rendering found.
A referee checking `paper/`'s `\citep[\S{}6.2]` would have found no such section. Found by
`/challenge-pr` reading the source's `\section` order. Rule: a section number is copied from the
PDF as printed, never derived; a reading note (`docs/readings/<id>.md`) holds the paper's section
map once, and citations point to it.
2026-10-22 (C2 unit 1): two rows written from summaries were wrong on the point a design leaned
on. `lit-S2.md` said the gradient-flow energy "decreases" (the texts: non-decreasing; no S2 rule
moved, since S2 counted drops as violations); `lit-C2.md` gave Bernas et al. as "anisotropy of
embeddings along Pythia ... checkpoints" from the abstract, and C2's F9 leaned on it, but the
paper's Pythia activations start at step 10 000, past the break's window. Found by reading the
texts before the draft cites them. The same unit then repeated the pattern one level up:
`/challenge-pr` on #253 found that the fetch summary of 2312.10794 §3.4.1 had dropped that the
monotone energy there is V-weighted, not E_β (the LaTeX was in `docs/readings/` all along), and
that "no suite" came from listing Hub branches when the checkpoints were tags. Rule: a fence or a
design line that leans on a cited claim names the claim's mark; a draft states an [A] row only
for what its abstract says and an [S] row never; a paper whose source is in `docs/readings/` is
read there, not through a fetch; "not available" is checked against every way a host stores it.
2026-10-23 (C5 unit 0): C5's scan, the first to search "massive activations" × "Pythia checkpoints",
found 2510.06477 (Oct 2025): σ₁²/‖X‖²_F, the BOS norm and the sink read on Pythia-410m's training
checkpoints, "emerge together around step 1k". C2's scans (2026-10-18, and unit 1's delta) searched
for anisotropy and the energy along training, not for massive activations or sinks, so C2's draft
closed with F9 saying published anisotropy along Pythia starts at step 10 000 and a §8 that does
not cite it. Found before anything was posted; carried to C2p. Rule: a scan for a statistic also
searches the named mechanisms a referee or fence offers for it (here §8's own "massive
activations and attention sinks"), each × the model family × "training checkpoints".
2026-10-24 (C2p unit 2): the same paper, one level down. `lit-C5.md` took its "around step 1k" as a
date and C5 stated a lean from it ("about 1000, not 512") and then reported the lean wrong. Its
appendix B.1 reads Pythia at steps 1, 1k, 2k, 4k, … on one GSM8K prompt, so "around 1k" is the
first checkpoint after step 1: a bound, (1, 1000], that contains both of C2's dates. Its training
figure reads entropy, sink rate and the BOS norm; σ₁²/‖X‖²_F is read in depth only. Found by
reading the paper's own checkpoint list before §8 cited it. In the same read a fetch summary said
the row's quote "in more than one stage" was not in the paper; the raw HTML had it (it is about
depth). Rule: a date taken from a paper names the paper's grid beside it; a fetch summary's
"not found" is checked against the raw text before it is acted on.

**The rule now.** Scan when a phase opens (before `design-N.md` freezes) and
before registering. Status: 📋 `CLAUDE.md` (already there; the misses predate it).

## 8. Git and PR mechanics

**Instances.**
- #59 targeted another PR's branch; after that base merged, #59 merged into a
  dead branch and its work never reached `main` (cost twice).
- PR sizes: #24 +618k lines, #17 +146k, 2026-09-12's merge 51 commits /
  ~9.5k lines — not reviewable.
- A merge mid-session captured the branch at that instant; later commits sat
  unmerged (#56 → #57).
- The repo's `github_key` is dead; `GIT_SSH_COMMAND` pointing at it fails.
- Two Claude sessions shared one working tree.
- A cleanup deleted branches it had been told to keep: lesson 12.
- 2026-09-24: an Edit deleting one card line returned a transient
  "no verdict" error; after the retry, two card lines were joined
  (`status-1c.md`; cause not established). The `phase-card` lint caught it.
  After an error on an edit, read the lines before retrying.
- 2026-09-26: Claude created `../Mets-work` per the Start protocol, then read
  and edited four files by their absolute `/Mets/...` path anyway — the
  worktree existed but every tool call still named the main tree. Caught only
  because an import check run from the worktree failed (the new module was
  not there); `git status` then showed the edits on the main tree. Bash cwd
  also resets between calls here, so a `cd` in one call does not carry to
  the next. Fixed by
  copying the changed/new files into the worktree and `git checkout --` on
  the main tree before continuing. Creating a worktree does not make it the
  default target of anything; every Read/Edit/Write/Bash path in the task
  must be re-anchored to it explicitly, and it is worth one `pwd`-equivalent
  check right after `git worktree add` to confirm before the first edit.
- 2026-09-27: the fleet-onboarding commit `cbff98f` ("Add infra/") contained
  only `infra/prompt.md`; the deny-by-default `.gitignore` dropped
  `infra/rvm.env` without a word, so a booted instance would have run on the
  fleet defaults (py3.12, `pip install -e '.[dev]'`, an extra that does not
  exist). After committing a new file type, `git show --stat HEAD` must list it.
- 2026-09-28 (#108): `./scripts/check.sh | tail -2 && git commit … && git
  push` committed and pushed `72ccbc0` over a red gate (3 phase-card errors):
  the pipeline's status is `tail`'s. Fixed in the next commit. Chain a commit
  on `check.sh`'s own exit code (redirect to a log, then grep it).
- 2026-10-24 (#266): the same chain again, a third time: `check.sh | tail -2 && … git commit
  && git push` pushed `08cc1fa` over one phase-card error (a bare `§4.1`, which the lint reads
  as `PROJECT.md`'s) and a stale doc index. Fixed in the next commit. The rule above was in this
  file, not in the habit: redirect to a log and test `$?` before any commit in the same call.
- 2026-09-29 (#115): `git log --diff-filter=A -1 -- <record>` in the cloud
  container named `2225beb` as the commit that added P-I5's real-run record,
  and the code comment cited it. The clone is shallow (`git rev-parse
  --is-shallow-repository` → true) and `2225beb` is its boundary: every file
  "is added" there. The record was added at `bff93d7`, before battery v2. The
  hash happened to match at both. Found by `/challenge-pr`. On a shallow
  clone, ask the API for history (`/commits?path=`), not `git log`.
- 2026-10-07 (#158): the PR was merged at its first commit while its `/challenge-pr`
  findings were being answered, so `main` carried a headline the review had overturned and
  the fixes went into a follow-up PR. Nothing on the PR said it was not ready. Rule: **open
  the PR as a draft; mark it ready only after every `/challenge-pr` finding is answered on
  it** (CLAUDE.md Stop steps 8–9).
- 2026-10-07 (#161): #59 again. The per-head arm was stacked on #160 on the user's direction
  (base `claude/p1e-u2-attn`). #160 merged at 22:01:29Z and #161 at 22:01:36Z, before GitHub
  retargeted it, so #161 merged into #160's dead branch (`2e11613`) and its work never reached
  `main`. `./scripts/status.sh` then said "open PRs: none" and `STATE.md` on the branch said
  "#160 open". Found the next session by `git log origin/main..origin/claude/p1e-u2-heads`;
  landed unchanged by #162. Rule: **a stacked PR stays a draft until its base has merged and
  its base reads `main`**; after any merge, run `git merge-base --is-ancestor <tip> origin/main`
  (already the rule; not run here).
- 2026-10-15 (A1b): an Edit to `status-A.md` and the `git commit` meant to carry it went out in
  one parallel batch; the Edit failed (file not yet read), the commit landed without the
  pre-output note. Caught from the commit's stat before push, amended, then pushed. Rule: **a
  commit is never in the same batch as the edits it records**; check `git show --stat` first.
- 2026-10-24 (C2p unit 2, #268): a new `trackC_writeup/paper_slots.json` that a new test reads was
  never committed. The whitelist `.gitignore` drops any `.json` not named, and `git add -A
  trackC_writeup/` said nothing; every local test passed on the untracked copy. `/challenge-pr`
  found it by unpacking the commit. Rule: **a PR that adds a file type or a new non-`.py`/`.md`
  file runs its new tests on `git archive HEAD | tar -x` before the PR opens**, and checks
  `git status --short --ignored` for the new paths.

**The rule now.** Target `main`; one coherent piece of work per PR; verify
with `git merge-base --is-ancestor`; worktree per task; fetch + recheck HEAD
before commit. After `git worktree add`, verify the very next file operation
actually landed under the worktree path before doing any more. Status: 📋 `CLAUDE.md`.

## 9. Token cost: reading to orient instead of working

**What happens.** Orienting meant reading `CLAUDE.md` → `PROJECT.md`'s resume
block → a handoff → a status file → a literature file: tens of thousands of
tokens before any work, and the files were too long to read carefully (lesson 1).
Our docs also mirror a verbose house style: bold-heavy paragraphs where a
table row would do, and results re-narrated in several places.

**The rule now.** `STATE.md` is the only startup read; everything else is
opened on demand. Numbers go in tables; one claim per line; a result is written
once and pointed to. Grep large files, don't read them. One session per unit
of work. Every unit's cost goes in `docs/cost_log.md` (`tools/session_cost.py`);
**a unit costing more than 2× the running median total context gets a line
saying why.** Status: 📋 + ✅ cap.
2026-09-22: `docs/index/` gives `PROJECT.md` and `POPPER_PLAN.md` a line-range
index (✅ `--check` in `check.sh`). `scripts/hooks/guard_large_read.py` refuses an
unbounded `Read` of any doc over 40 KB (⚠️ not yet registered in
`.claude/settings.json`; that needs the user). Basis: `archive/docs/agent_context_scan_2026-09-22.md`.
2026-10-01 (#122, #123): `STATE.md` is within its 150-line cap but 31 kB, because
the "Current stage" cell is one ~10 kB line. The startup hook's output goes over
the harness's inline limit, gets persisted to a file, and the session reads it
2–3 times to see it (wrapper, wrapper again, then `Read` in chunks). The line cap
does not bound bytes. ⚠️ Fix not done: a byte cap in the tier-0 lint, and the
stage history moved to `status-1d.md`.
2026-10-02 (#131, and again in the candidates unit): a `Monitor` that emits one line per
finished passage turns a ~20 min batch into ~20 notifications. Each one is a call that
re-reads the whole context. #131's cost row already said "one watcher on each batch's end
would have done", and the next unit made the same choice. Rule: after checking the first
output by hand, watch only the batch's end and its failure signatures (`refusing`,
`Traceback`, `Killed`), not per-item progress.
2026-10-06 (#154, R7): the same again, a third time: a `Monitor` matching each step's last
passage gave 15 notifications over a 40-min batch, while the background `Bash` already notified
on exit. The rule lived only here, and this file is not read at startup. ⚠️ Candidate: one line in
`CLAUDE.md` "While working" ("one notification per batch: its exit").
2026-10-09 (#176, E1): a fourth time, ~20 per-step notifications over a 15-min GPU batch, 17.1M
context (3× the median). 2026-10-10: the candidate is in `CLAUDE.md`, and run-and-watch moves
off the main session altogether: the `runner` agent (Haiku, small fresh context) runs, checks the
first output, waits on exit only, fixes at most 3 mechanical bugs and escalates the rest. ✅
Second gap, found while building it: `session_cost.py` did not read subagent transcripts, so
every `/challenge-pr` was uncounted (one in session `6f6e18c3`: 26 Opus calls, 1.9M context).
It now follows `<session>/subagents/` and splits context by model.
2026-10-16 (A2 unit 3): a gate run was handed to `runner` before a change the design's fences
required (the `ln_from` merge) was made; the change edits a file the gate's code stamp hashes,
so the run was killed a few minutes in and relaunched. Before a stamped run: make every change the
unit owes to a hashed file first, then launch.
2026-10-18 (the user's check-in): `STATE.md` at 149 lines and 46 kB, so the start hook's output was
persisted to a file and read again; the 2026-10-01 fix (a byte cap) was never built. Over the cost
log, context re-read is 1.54B tokens against 7.6M of tool output (200×; 127k per call on average
from 2026-10-13, read from `docs/cost_log.md`). The byte cap and a
per-read attribution in `tools/session_cost.py` are now `docs/ROADMAP.md` §5 rows.

## 10. Tangents: interesting findings that hijack the plan

**What happens.** A planned line item turns up something unplanned and
interesting; the session follows it; the plan and its handoff fall behind, and
the tangent's result lives only in prose.

**The rule now.** Triage every surprise the moment it appears:

| kind | test | action |
|---|---|---|
| **Defect** | could it make the current result wrong? | fix now; it blocks |
| **Confound** | does it change how the current item must be read? | attach to the current item as a check |
| **Discovery** | interesting, but the current item stands without it | park it: one line in the thread's handoff under "Parked", with *why*, *cost*, *which decision it could change* |

At each stage boundary, rank parked items by (chance it changes a decision) ÷
cost, and pull the top one or two into the plan. Status: 📋 `CLAUDE.md`.

**The same pattern one level up: phases.** 2026-10-10, named by the user ("we keep having
this problem of running through a phase and then finding a problem and then building a new
phase instead of just finishing the previous phase"). Phase 10 held for 1d; 1d paused while
Phase 10 re-read its definition; 1e opened and paused; each left Blocked and Parked items open
(over 20 Parked in `handoff-10.md`, 1e's U4 / U5, 1d's write-up), and Phase 10's own Stages 3–7
never ran. Rule: a phase or thread opens with an exit condition and a unit budget, closes out
(card, summary, every Parked item carried or dropped) before the next opens, and a mid-phase
idea goes into `docs/ROADMAP.md` (`CLAUDE.md` "Phases and tests"). Status: 📋.

## 11. The author's own review cannot see what the author believes

**What happens.** A design is checked by the session that built it, which
re-reads it through the reasoning that produced it. Where that reasoning holds
a wrong premise, the check inherits it.

**Instances.**
- 2026-10-14, Track A A1a re-gate (#203, `/challenge-pr` finding 1): the write-up read "G4 fails at
  blocks 20–22" as the reading's error there and recommended re-planting the start cosine. The
  sweep runs all 23 planted steps through one frame; the reading takes one step per block's frame
  and sums. The author never set the gate's statistic beside `block_steps`. Also miscounted (2,592
  gated cells for 2,484; "6 of 7" orthogonal for 7 of 7). Rule: before reading a gate's failure,
  write the reading's statistic and the gate's side by side; counts come from the record by code.
- 2026-10-10, Phase 10 F13r (#188, `/challenge-pr`): the write-up was drafted from a compact grid
  the author printed for f, and it called 2000 L1–8 *unclear* though the report's cls column read
  *carries* there (and at 512). It also measured the linear statistics' shrink against the binned
  `S_c4`, a different estimator, so "keeps its size" in L1–8 was partly the switch. Rule: the
  write-up quotes every reading the report prints for a cell, not the subset the author looked
  at, and a shrink is measured against a reference computed by the same estimator.
- 2026-10-06, Phase 10 R7 (#154, `/challenge-pr`): the rule computed three yardsticks (each
  group's own floor, the fixed bar 0.5, and the chance level as a count) and the headline gave the
  own floor's label, "mixed", as the result. On the fixed bar the same records read "context", and
  the own-floor share was lifted by floors below chance (68–91 % of them token-alone). The author
  had chosen the own floor for comparability with "moves" and read the rest as "beside". Rule: when
  a result's label changes across the yardsticks computed beside it, the headline gives the range,
  not the primary's label.
- 2026-10-06, Phase 10 R6w (#152, `/challenge-pr`): the headline said the drift "happens inside
  groups that persist". For statistics measured in each step's own layer 0, a shift shared by
  every group lands wholly in "within", because the entry and exit terms compare groups at the
  same step. So the result was "not replacement", not "lineage carries it". The same review found
  that the first output recorded the rule commit as its git while the runner was uncommitted.
  Rule: ask what a decomposition reads under "everything moved together"; record a dirty flag
  beside git.
- 2026-10-05, Phase 10 R6: the matcher's rule named 0 → 2 **and 2 → 4** as its no-learning
  floor ("weights barely move"), but R0's own table, read for that same rule, shows c1 at
  997 → 903 across 2 → 4; the run gave c2a stable 0.86 there. Caught at reading, before any
  claim rested on it; the rule now says only 0 → 2. Rule: a "nothing changes here" premise in
  a design is checked against a number already on record before it is written, not assumed
  from step size (`status-10.md` §1.20).
- 2026-10-05, Phase 10 R3: the author found that A0's flip was two tokens, then read the
  definition's leftover corrected gap as "a residual that persists, mostly on learned groups".
  The mask correction divides out uniform attention only, and the definition's members sit
  earlier than the rest, so a trained model's recency is left in the "corrected" gap: within
  position bins it is gone from 32000, and the learned groups' +0.18 goes to −0.05.
  `/challenge-pr` on #148 caught it. Rule: a "corrected" statistic is corrected only for the
  baseline it divides by; when the populations differ in the corrected axis (position bias ≠ 0),
  compare them inside bins of it before reading a residual (`status-10.md` §1.17).
- 2026-10-05, Phase 10 R2: the author read F1's labels going from "negative" (c0) to "none"
  (c3) at 1000–54000 as "the late tail goes". The label is a p-value and c3's records hold a
  fifth of c0's members (median 50 against 252); the effect sizes were the same (−0.02 to −0.13
  against −0.01 to −0.15). `/challenge-pr` on #147 caught it from the records. Rule: when a
  column changes the sample size, compare effect sizes before reading a significance label's
  change as an effect's (`status-10.md` §1.16).
- 2026-10-05, #142 (1d unit 4): the pass rule compared trained against step 0
  (seed 0's learned-content count above every step-0 init's), and the write-up
  read its pass as "the definition (moves ∧ learned) finds known content". The
  review recomputed content share by filter: learned 0.73 vs not learned 0.72;
  training supplies the content (0.72 vs 0.02 at step 0), not the filter. The
  rule was fixed before the run and could not separate the two. Rule: a pass
  says what its comparison compares; a conjunction's claim about one term needs
  that term against its complement (the ablation, minutes on stored records).
- 2026-10-02, #131 (1d unit 2's first check), two premises `/challenge-pr` broke:
  (1) the power reading ranked real inits against 40 re-inits and held-out ones
  against 30, then compared their outer-10 % shares, whose tie-driven baselines
  differ (0.098 vs 0.13); the write-up called it "the stronger evidence, and it
  agrees". Ranked against the same 30, real runs 0.08–0.17 against ~0.13:
  consistent, not tight. The analysis was an unsaved script, so nobody could
  re-run it; it is now `arch_null.power`. (2) The re-inits were float32 while
  every real init is float16-valued (`pythia-410m`'s float32 `step0` included);
  the session had read that checkpoint's dtype and checked only that it said
  float32. Rule: a comparison of rates states that both sides share the reference
  set; an "init matches init" claim checks precision, not just σ.
- 2026-09-30, #117's mutation accept list: 4 of 22 "equivalent" survivors
  were killable (`/challenge-pr`). One reason called `average`'s alpha check
  redundant; it is the only guard on the early return for an infinite
  e-value. Each reason had been argued, not run. Running every remaining
  reason's boundary against the live mutant (`MUTANT_UNDER_TEST`) killed 2
  more and turned 10 into "accepted, not equivalent", each naming the input
  that tells them apart. Rule: **an equivalence reason names the probe that
  was run**, and the list is keyed on the code the reason rests on
  (`tools/mutation_check.py`, `context`). The next round (`/challenge-pr` on
  #119) repeated the pattern: #119 said a changed registry alpha makes every
  stored decision fail `--verify`, and it does so only where a decision
  flips. The probe that would have shown it was one line.
- 2026-09-25, #95: I found Stage 0 bit-identical to the WDS backfill and
  wrote that §3's floor was "between the pilot and today", with "sensitivity or
  toolchain" parked as open. The evidence was already on disk: at steps 0–1000
  the two sweeps' activations differ by ≤ 2.5e-7 and partitions still differ,
  and §2's backfill had matched the pilot's labels with today's code.
  `/challenge-pr` caught it. I had looked only at the largest gap (8e-5, at
  step 143000) and not at the gap per step. Rule: before calling an
  explanation open, split the evidence by the variable it depends on.
- 2026-09-22, #67: the Stage 0 driver said a deadline kill re-runs only "its
  unfinished prompts". `run_1` writes `pair_agreement.json` once, after the
  whole invocation, so a kill threw away up to a checkpoint's worth (~1.4 h)
  of finished runs and left orphan directories matching the hash and sha. The
  author's gate was green; a fresh-context `/challenge-pr` found it on its
  first run, `file:line` included.
- 2026-09-26, 1d item (3): the same shape again, before any review could see
  it. `attention_null.py` collected every layer-record in `pool.map` and wrote
  once at the end; the user stopped the run after 3 h and all of it was lost.
  The #67 instance was in this file. Rule: any batch over ~10 min writes each
  finished unit as it lands and resumes from them, before its first launch
  (fixed: per-record parts, `imap_unordered`).
- 2026-09-29, 1d long prompts: the third time. The fix went into
  `attention_null.py` only; `gaussian_null.py` (built the same day) and
  `beta_refit.py` kept an all-or-nothing `pool.map` / `ex.map`, and Claude
  launched both at 2048 tokens (hours each) without checking them against
  the rule above. A shutdown then cost ~5 h of the Gaussian null and ~4 h of
  β. The estimate that made it seem safe was one record at one layer on a
  lightly loaded box (~2 h forecast; the run was on course for ~9 h). Rule
  addition: before launching a batch, grep its driver for `pool.map` /
  `ex.map` / a single write at the end; a fix to one driver is applied to
  every driver with the same shape (fixed here: both, per-record parts).
- 2026-09-30, 1d long prompts: resumable is not running. The chain was
  launched 2026-09-29 20:45 with nothing holding the box awake; KDE's idle
  suspend took it at 21:20 and it woke at 05:16: ~8 h of a ~18 h chain lost
  (45 of 600 records done by morning), and the 2026-09-23 instance (lesson 1)
  had already shown this box suspends under load. Rule addition: a batch
  expected to outlive the user's attention runs under
  `systemd-inhibit --what=sleep:idle` (here attached afterwards to the
  chain's PID with `tail --pid=<pid> -f /dev/null`, so it ends with the chain).
- 2026-10-21, S1 unit 2's re-gate: briefed to the `runner`, whose detached-run template had no
  inhibit, so the rule above never reached it; the box suspended 21:13–05:14. The runner read
  8.5 h of wall clock as a slow GPU and escalated; the author killed a healthy job (~30 min of
  compute lost) before checking `journalctl`. The same run also broke the per-unit-write rule
  above: `regate` writes nothing until every band is drawn (left running under the inhibit;
  `r1` was made to write and resume per step before its launch). Rule addition: the inhibit is
  in `.claude/agents/runner.md`'s template, and an overrun is checked against a suspend first.
- 2026-10-17, A2 unit 5: a "quick" synthetic dry check run on the GPU without the inhibit
  reported 40,283 s; the box had suspended mid-check. The rule above was read as "batches";
  it applies to **every** GPU call that can outlive a glance, dry runs included.
- 2026-09-30, #118: the author read "long − v1" as the length effect and
  quoted "L17–23 falls ~40 % in every prompt" for Blocked 9. `/challenge-pr`
  saw that the long fit adds pairs at offsets v1 never had (28–78 %). On v1's
  offsets the paired fall went from −0.49 to −0.17, and the headline from 2.80
  to 3.12 (v1 3.16). The "~40 %" was also a ratio of medians, not the paired
  change. The same review found `status-1d.md` saying `beta_refit` resumed
  only on matching settings, which it never checked. Rule: when a comparison
  changes n, name every other thing that changes with n (here the offset
  range, and in the Gaussian null the power) and hold it fixed before
  quoting the difference as length.
- 2026-09-24, #85: dropping the dependency hashes, the author wrote that a
  correction "reaches a reader through its own `## Corrections received`",
  though no rule puts a line there. The evidence also could not bear on the
  question: "the hashes caught none of Phase 1's 7 corrections" is true by
  construction, because Phase 1 depends on nothing. The user settled a
  binary question (hash the whole file or nothing) without being offered the
  middle option: hash only the Corrections section. `/challenge-pr` found
  all three. #85 had already merged by then, so the fix went into a follow-up PR.
- 2026-09-24, #91: the author parked "is the carrying capacity a repeat count?"
  as needing a pass over Phase 1's data, and called it "a competing reading".
  `max_alive` is the maximum over layers of the per-layer count that #91's
  own record stored, and 20 lines of reading it answered the question (it peaks
  at the embedding in 92 of 133 runs). The author also read `repeated_tokens`
  as evidence against, when it is a different mechanism. Before parking a
  test, check whether the data already written answers it.
- 2026-09-25, #96: the author wrote "every quoted cell is within 0.005" when
  the pilot lacks two of the quoted steps. "3 103 / 3 120 agree" also hid
  that 96% of run-layers were bit-identical, so the count mostly measured
  sameness. The reviewer's own step-32 count ("8 of 25 layers") was wrong
  too, and re-running it gave 29 run-layers in 16. A "holds" claim should
  state what it was compared on. Check a reviewer's numbers before quoting them.
- 2026-09-25, #97: the author called the pilot's below-baseline `Z` to ~7000
  "new". The WDS record's 1000–8000 rows, printed in the author's own
  comparison table, already showed it; §1.4's table had left those rows out,
  and the author compared against the doc, not the record. The same PR repeated
  #96's mistake: per-unit agreement counted identical cells. Before calling a
  result new, check the earlier *record*, not the earlier write-up.
- 2026-09-25, #99: the literature scan recommended "normalise stability by its
  null" as the textbook fix without trying it on the smoke run's stored
  candidates, which were already on disk. The reviewer did try it: the ratio is
  infinite where the null mean is 0, and it moved k to the grid's other end. A
  remedy taken from the literature gets checked against the data already in hand
  before it is recommended.
- Before `/challenge-pr` existed, the 2026-09-12 merge was 51 commits with no
  second reader at all (why the PR-size rule exists).
- 2026-09-25 (#101): the drift write-up said "the stable families are all
  coarse" from memory of the smoke run, and "tuned HDBSCAN carried the
  collapse" from one weight moving. The reviewer read the stored picks
  (agglomerative k median 162), and a swap test showed no single family moves
  it. A mechanism claim about a record gets its swap or ablation run before
  it is written down.
- 2026-09-25 (#102), the next unit: the matched-k write-up named the mechanism
  "methods that hang on distance order" from a check that only *located* the
  disagreements (in near-tie pairs, which is where every co-membership on that
  prompt sits anyway: the base rate). It read "k-means never moves" without a
  seed control. The PR's own "Worth challenging" said no intervention had been
  run. The reviewer ran the cheapest one, float64 distances, and the drift
  vanished: float32 cancellation in `1 - x·y`, upstream of every method. When
  the PR's own caveat names a missing intervention that costs minutes, run it
  before opening the PR.
- 2026-09-28 (#108): the write-up read "beats null A" as content evidence
  while its own step-0 row beat A too (59/161, 148/161 deduped): the negative
  control had failed and was written up as "nothing but position". It also
  called A "at level" from a calibration that is A against itself, and
  counted only Q's upper tail. When the negative control beats a null, the
  null is the finding.
- 2026-09-29 (#110): the β refit recommended 5.8, the median over heads with
  partial R² ≥ 0.05. With one regressor left, partial R² is `β̂²·Σs̃²/Σỹ²`
  and `Σs̃²` is shared within a layer, so the floor selected on |β̂|: the
  floored medians were truncation. The write-up read the arithmetic as two
  findings (steeper kernel heads, steadier depth profile), and "step 0 fails
  0.05" was one head below the cut. The PR's own "Worth challenging" named
  the floor without writing out what partial R² is made of. Before
  selecting units on a fit statistic, write the statistic in terms of the
  estimate; if the estimate is in it, the selection is on the outcome.
- 2026-10-01 (#124, reading list): the count of already-cited ids came from
  a script and was right; the "where" table beside it was typed from memory
  of an earlier grep and named the wrong file for 5 ids. Two rows set
  constant-free scalings (√β, `n^{2/(d−1)}`) next to measured numbers. A
  review found both. Generate a cross-reference table from the grep that
  produced its count, and put no number from an unread paper beside a
  measurement.
- 2026-10-02 (#129): the design said the long-prompt continuations were "long
  enough for 1000 tokens" without opening `provenance.json` (`sullivan_ballou`'s
  is 550). It also cited a 0.5 Jaccard bar as "the identity control's recall",
  which was 0.39. Both were written from memory of a file. A claim about a
  committed file's contents gets checked against the file in the same session
  it is written.
- 2026-10-04 (#137): Blocked 17's continuity rule was checked before the run only
  for what it was meant to fix (the synthetic's found scales on seeds 2–11: 8 of
  10). The same seen seeds' Gaussian clouds give 29 of 50 plateaus without (b)
  under it (0 under the old rule), so (b) became the only guard against false
  structure; nobody looked, though the design reports that reading, and the user
  chose without it. `/challenge-pr` found it. Rule: a rule change is replayed on
  the seen seeds for **every** reported reading, the negatives included, before
  the options table goes to the user.
- 2026-10-05 (#141): the trained reading found 1 plateau in 1,680 clouds, and a
  scratch script counting runs of admissible points (with no breakdown of which
  condition failed) led the write-up to "the grid is why", which then shaped Blocked
  21's options. `/challenge-pr` broke it down per point: the window's middle point (r ≈
  0.9) fails because the re-inits match the trained clouds there, not because of
  resolution. Rule: a diagnosis of *why* an instrument found nothing comes from a
  committed per-condition breakdown (here `scale_real diagnose`), not from a count of
  what passed.
- 2026-10-07, #157: one GPU-against-CPU pair at 2,048 tokens (7.2e-5) was written up as
  "~10× the 512-token gap", i.e. caused by length, and put into `compute_profile.md`.
  `/challenge-pr` asked whether it was length; all 8 pairs split by position quarter show
  the gap is step 143000's deep layers at every position (up to 1.8e-4) and absent at
  step 0. Rule: a cause named for a measurement is checked on the split that would
  separate it (here position), not inferred from one pair.
- 2026-10-08, R8x (#163, `/challenge-pr`): the write-up behind recommending (a) said "the
  drop's size is resolved" when the bootstrap intervals were 5–21 % (only its existence was),
  and "not one passage" when dropping `camus_letranger` lifts 128 over the bar. It also left
  out the pooled 64–1000 keep (0.905, which clears 0.90). Each slip leaned toward the
  recommendation already in `STATE.md`. Rule: a sentence supporting a recommendation states
  its interval, and the strongest number against it (here the pooled window) goes beside
  the verdict before the PR opens.
- 2026-10-21, S2 unit 1 (#247, `/challenge-pr`): Blocked 87's recommended option drew arm a's
  controls at 1.5–3× its size "as the fallback says", without checking that a control can be
  built there with both norms matched; for arm a it cannot (‖control‖ ≤ 1.73 ‖K‖ against 2.04),
  its feasible rungs flip or shrink the head, the interpolated label had no null, and the option
  was the one that could turn 8000's arm-a V into a reading. The reason given against the
  alternative ("it changes the controls after seeing 8000") applied to it too. Withdrawn before
  the user decided. Rule: **an option offered to the user is checked to be constructible with
  the design's own definitions (one draw, one head), and the argument against each alternative
  is tried against the recommended one.**

**The rule now.** Every PR gets `/challenge-pr` (fresh context, intent and
design, not lines) and the author answers each finding on the PR; the user
settles disagreements. Status: 📋 `CLAUDE.md` Stop step 9 (lands with #68).

- 2026-10-09 (E1m, `design-10.md` "E1m"): the reading function counted a *leans pulls* E1 label as
  "E1's pull" where the rule said *pulls together*; found when the reading table first printed,
  and fixed to the rule's letter before anything was quoted. The rule was also silent on *leans*
  (E1's rule had been silent on 1e's mixed windows): the report marks both. Rule: write the
  reading's code from the rule's sentences one at a time, and list in the rule which labels it
  does not name. Also this unit: a causal-view test planted a clustered group, so a pull towards
  *later* members legitimately scored > 0 (earlier members point the same way); the test, not the
  code, was wrong, and was replaced by one that changes the invisible rows. And one tracked doc
  (`status-10.md`, number fixes) was rewritten through a `python` heredoc against `CLAUDE.md`
  "While working" (use Edit); the rule exists for the re-inject cost.
  Again 2026-10-25 (B7 unit 0): `lit-B.md`'s B7 section appended with a `cat >>` heredoc.
- 2026-10-17, A2's close-out (#216's text, found while writing the one page): unit 5 wrote "1e's
  ≤ 0.05 holds to 1000 and fails in 17 of 24 blocks from 8000" and STATE carried it. A2's share
  is the sink's per-token norm over the update's per-token norm; 1e's is the sink's projection on
  the block's *shared* (mean) update (`u2_attn.shares`). Same name, different quantity; neither
  the author nor `/challenge-pr` on #216 opened 1e's code. Rule: a sentence that says an earlier
  phase's number "holds" or "fails" names both definitions, or it is not a correction.
- 2026-10-17, B2's freeze (found writing unit 1's code, before any output): `design-B2.md` put
  C1–C4 in "B2a's LN1 frame" with `M_0 = diag(g) C W_Qᵀ W_K C diag(g) / √64`, after the author
  and `/challenge-pr` on #218 (whose finding 4 made that frame primary) had both read it. B2a's
  tokens are LN1's rows with the gain in, and its logit is `yᵀ W_Qᵀ W_K y′ / √64`, so the formula
  applied the gain twice and set V and M_0 in different frames. Neither review opened
  `b2a.head_logits`. Rule: a design that reuses another unit's frame writes each matrix by
  reading that unit's code, not its prose.
- 2026-10-18, B2 unit 2 (found by `/challenge-pr` on #221, finding 4): the PR called the low-rank
  `c5_row` "exact up to eigenvector phase" against the dense `classify_ov_row`, from a check on 14
  heads and planted rows. The two differ whenever V has no eigenvalue with positive real part
  (the top is then one of the d − 64 zeros): 237 of 384 heads at 1000, so most of C5's
  "unclassified" rise. The 14 checked heads never met the case. Rule: before calling a rewrite
  equal to the original, list the branches where the original's answer is set by a degenerate
  value (ties, zeros, noise) and count how many inputs reach them.
- 2026-10-18, B2 close-out (found by `/challenge-pr` on #222, finding 1): the list carried to B4 as
  "every head that reads `L5H2`'s write space" came from `b2_show`'s table, which ranks only the
  members' layers. Over all 18 layers after 5, `L9H0` is 5th (27× chance) and `L6H3` 9×. Neither
  was listed; `L9H0` is also outside 7d's causal top 10, so it is the contrast B4 needs. Rule: a
  list labelled "every" or "all" is produced over the whole population, never copied from a view
  scoped to the members.
- 2026-10-19, B3 units 2 and 2b (found by `/challenge-pr` on the 2b commit, findings 2–3): #228's
  first review read ℓ ≈ 2.0 at `resid_pre` 7 as "most of the cone is in place after block 6"; the
  author took it, recommended Blocked 69 (c) on it, and 2b's first write-up said "later layers
  rebuild the cone". ℓ is a log ratio: the partner's σ there is 8 % of its final size, so nothing
  was in place and nothing was rebuilt. The same blindness put `L6H0`'s Δℓ (the neighbours rising)
  down as a route, and hid that half of L1's 0.54 is the neighbours' side. Both sides were on
  disk the whole time. Rule: a ratio statistic is read beside its numerator and denominator
  before any sentence about size, place or route; a Δ of a ratio says which side moved.
- 2026-10-19, B3 unit 2b and close-out (found by `/challenge-pr` on #230, finding 4): the
  per-block table is indexed by `resid_pre` b, block b − 1's output, and 2b's prose turned it into
  blocks one late ("flat through block 16, then 17–23"; the blocks are 6–8, 10–11, then 16–23);
  the close-out copied it into the card, ROADMAP's B4 row and `PROJECT.md`. #228's finding 1 was
  the same slip on the same table. Rule: a sentence naming blocks from a `resid_pre` table writes
  the subtraction out once (`resid_pre` b → block b − 1) beside the first block it names.
- 2026-10-22, S2 unit 2 (found at the next session's start): CodeRabbit posted two findings on
  #248 at 21:09Z, six minutes before the `/challenge-pr` answers were pushed, and the session ended
  with both unanswered. One was real: G3′'s base check sat inside `if not ps.has("g3p_base")`, so a
  resumed gates pass skipped it and `gate-read` never checked it (no recorded number reached it:
  `base_D` ≤ 1.6e-6). Rule: before the Stop protocol's last report, list the PR's inline review
  comments (`gh api repos/<repo>/pulls/<N>/comments`) and answer each; a refusal check runs on the
  stored record, not only on the branch that computes it.
- 2026-10-22, S2 unit 4 (found by `/challenge-pr` on #250, findings 1 and 3): the design chose m = 2
  from a scratch script outside the repo, and the write-up said bS's top 2 per head "are about
  half" of the pairs that lean into S. The committed records said the reverse: bS holds 84–94 % of
  them, but half of bS is filler, and 102–163 heads have none, so the null holds there by
  construction. When the scratch check was rerun as tracked code, it found a fact the scratch had
  missed: m = 1 would have been refused at every step. Rule: a number that justifies a design
  choice comes from a tracked command with its record's sha; a sentence about a selection's
  purity is read off the record in both directions (what share of the selection passes, what
  share of the passing is selected).
- 2026-10-22, C2 unit 0 (found by `/challenge-pr` on #252, findings 1 and 3): the design's
  opening sentence, the card's question and the generated phase table said the energy is
  "non-increasing in depth"; the repo counts a *drop* as a violation (`core/metrics.py`
  `energy_violation_severity`) and the oracle test expects a rise, so the paper's first line would
  have stated the prediction backwards. The same design quoted "≤ 8" step-0 violations, a bound
  its source gives for steps 0–4 (step 0's is 1–6). Both were written from memory of the one-pages,
  not from the definition or the record. Rule: a write-up's statement of the theory's prediction is
  checked against the code that counts its violations; a quoted bound names the steps it covers.
- 2026-10-22, C2 unit 2 (found by `/challenge-pr` on #254, finding 1): the render step checked
  that every printed number came from its record, and §5 still said "E8 (1000) < E3 = E4 = E10
  (1000)". The order was copied from A3's `reading.order`, which had placed E8 by its other date
  (512, from its peak) without printing it. The number was right and the "<" between two numbers
  was wrong, and no check looked at the "<". Rule: a rendered draft also checks what its sentences
  assert (orders, "every", "none", margins) against the same table (`numbers.CLAIMS`), and an
  order copied from a record is written with the date that placed each row.
- 2026-10-23, C2 unit 5 (the fresh-context referee read, `trackC_writeup/referee-C2.md`): six
  `/challenge-pr` rounds, each on one unit's diff, had passed the draft. A reader given only the
  draft found 9 major and 27 minor points. Among them: E₁'s first-order term is exactly ½‖m‖², the
  anisotropy the paper treats as a separate control, and no section said so. §1 called one-step
  gaps "after/before" while §5 counted them "aligned". "Nine other measured changes" did not
  match the table's rows. "The second energy agrees" hid that it differs on 2 of 9 seeds. Internal
  labels (tier 1, A3, v1, G3′) were never defined. Each passed review because each reviewer saw a
  diff in the context of its unit, not the whole document a stranger reads. Rule: a document
  meant for outside readers gets a whole-document read with no project context before it is
  called complete, and the statistic's simplest expansion is written down before the paper says
  what is new about it. The answers then repeated a stale belief: the draft said the ‖m‖² term
  was "not read", but unit 3s had read its layer profile on every seed (post hoc, in prose only,
  never saved as a record). `/challenge-pr` on #260 caught it. Rule: before answering "not
  measured", grep the phase's status file for the quantity; a post-hoc reading quoted in prose is
  saved as a record the same unit, or it gets lost.
- 2026-10-23, C2 unit 5b (follow-on to the instance above): the share of each E₁ drop that the
  mean's term carries was exact arithmetic on fields saved at unit 3 (`e1` and `cos` per state).
  Four units of text said "not measured", and the referee read listed it as a run past budget.
  One script with no forward pass read it: the mean's term is most of every drop at the break, and
  the abstract changed. Rule: when the statistic has an exact decomposition and the records hold
  its parts, the split is computed in the unit that first reads the statistic, before any sentence
  says how much one part carries.
- 2026-10-23, C2 unit 5b (found by `/challenge-pr` on #261, finding 1): the frozen reading R6 (a)
  compared "‖m‖² falls at every drop" with the share of *all* transitions at which ‖m‖² falls
  (0.20–0.35). Most transitions are rises of E₁, so the comparison was selection on the outcome:
  E₁ and ‖m‖² move together at 0.96–1.00 of transitions at every step. The draft first presented
  a property of E₁ everywhere along training as a property of the break. Rule: a conditional
  share ("X at every Y") is set beside the same share conditioned on not-Y, or on Y's complement
  of the same kind (here, rises), never beside the unconditional rate.
- 2026-10-25, B7 unit 0 (found by `/challenge-pr` on #271, finding 1, before any real value): the
  proposed design matched each head to the best of 16 next-layer heads on the prompts it read,
  and set it against an offset-profile null pooled over all positions. Both errors lean the same
  way: on a pure reshuffle P sat 40 % above chance and N0 at half of it, so every cell would have
  read "persistence beyond position". The design said "every null takes the same max" and N0 did
  not. Its own gate arm had an absolute tolerance (0.02) 2× the null's size, so it would have
  passed. Rule: a null is simulated at the real design points before the design is proposed,
  and a tolerance is stated relative to the null's size.
- 2026-10-25, B7 close (found by `/challenge-pr` on #277, finding 1): Blocked 113 recommended
  closing over a sink-excluded unit on an estimate "not computed": about 1/10 of a mid-depth cell's
  rows left, about 25 a head, under the 50-row entry, so "mostly not read". The estimate set the
  median head against the entry, but a cell is read once 2 heads enter, and the session's own sink
  table implied about 1/5 left. Scaled evenly in the records, 85 of 89 cells still read at 1/5
  and 37 at 1/10. It also missed the repo's unfenced long prompts. The user decided on that
  recommendation, and the close-out repeated the estimate in four files. Rule: an estimate that
  decides between a Blocked item's options is computed from the records when it is a read of
  them, under the rule's own read condition, before the option is recommended.
- 2026-10-25, B7′ unit 0 (found by `/challenge-pr` on #278, finding 1, before the freeze): the design
  placed its sink threshold (¼ of a layer's condensed rows over 16 heads) "in the gap" of B7's sink
  table, 0.50–0.94 against ≤ 0.06. That table pooled only the heads entering each cell
  (`b7_diag.py prompts`), so it was a different count from the rule's; at 2000 one prompt's key was
  the modal partner of 63 of 224 heads, so on a layer count its sink could sit under ¼. The rule
  now counts per head, ¼ is placed against the null, and the arms plant head-sparse and
  sub-threshold sinks. Rule: a threshold said to sit in a seen gap names the producer of that gap
  and shows it computes the rule's own statistic; otherwise it is placed against a null and
  labelled so.

## 12. Deleted, then needed

**What happens.** Something is deleted (a branch, a file, a run directory) and
a later session needs it. The code is usually the cheap part to lose, since
tooling has moved on and a revival would rewrite it anyway. The expensive part
is the *intent*: why it was built, what it was meant to test, and what building
it taught. When the only copy of that is inside the deleted thing, it goes too.

**Log every instance here**, newest first: what was deleted, when, what needed
it, and what was recovered.

**Instances.**
- 2026-09-25, the same Phase 1d, **code needed after all**: the user put
  Phase 10 on hold to settle what a cluster is. The tag still held all 5 069
  lines, so restoring took three small drift fixes and the gate stayed intact.
  A rewrite would have cost days. The first real layer then crashed
  (`separation_score` on a null draw with one cluster per token), a case no
  synthetic fixture produced (lesson 2's rule: run on real output first).
  Keep `dead/*` tags until the phase is either revived or formally dropped
  (`p1d_cluster_ensemble/status-1d.md` "Revived 2026-09-25").
- 2026-09-23 → needed 2026-09-24: the "delete every branch except `main`"
  cleanup took `claude/particle-methods-comparison-vpuads` (Phase 1d, 5 069
  lines) and `claude/visualize-mets-results-sl2ya5` (one 255-line tool), both
  listed in `INDEX.md` as "do not delete". The check ran on the 3 non-ancestor
  branches, not on that list. Needed next day: 1d is the obvious tool for Phase
  10's "HDBSCAN partition not reproducible" (`status-10.md` §3). Recovered from
  two unpushed local `dead/*` tags: 1d's design, status, findings and the text
  of its never-registered `P-C1`–`P-C4` now sit in `archive/p1d_cluster_ensemble/`,
  and the viz tool's purpose sits in `INDEX.md`. The code was let go (user,
  2026-09-24).
- 2026-09-24, found not needed yet: Phase 3's headline run (decoder→V
  0.484 / 0.501) is on no drive; only an earlier partial run survives
  (`archive/p3_crosscoder/status-3.md` "Runs on disk"). The intent survived in
  `status-3.md` and `design-3.md`; the evidence did not (lesson 1).

**The rule now.** Code may be deleted; intent may not. Before deleting
anything with work in it that exists nowhere else, move its design/status docs
(why it exists, what it tests, what it found, any predictions written) to
`main`, under `archive/` if frozen, with a `FROZEN.md` saying where the code
went and when to rebuild. A list that says "do not delete" is checked by the
cleanup itself, not by memory. Status: ⚠️ prose only; a cleanup is rare enough
that a lint is not proposed.
