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

**Why it keeps happening.** The same fact was written in 3–4 places
(`PROJECT.md`, `status-N.md`, `handoff-N.md`, `INDEX.md`); updating one left the
others wrong. `PROJECT.md` grew to 485 KB (~120k tokens) — too big to read, so
it got skimmed, so errors survived. The update was a thing the user had to ask
for ("update all the MD files"), and which files was never defined.

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

**The rule now.** Refuse rather than degrade (standing rule 4). Before
launching a batch, inspect the **first** output for populated content, not
just existence. Status: 📋 `CLAUDE.md`; ⚠️ no automatic run-directory
validator yet — candidate tool: `tools/verify_run_dir.py`.

## 3. The input changes underneath code that assumed it was fixed

**Instances.**
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

**The rule now.** Anything whose result is recorded names the exact input
set it ran on (a hash of the texts, not a key list) and refuses on a
mismatch. Code that rebuilds positions from the live text also checks them
against the run's `tokens.txt` (`core.battery_structure.verified_prompt_ids`).
Status per file: `docs/battery_consumers.md` (audit 2026-09-29; P-I5 and
`run_7.py` fixed, `CLAIM-C`'s growth is the user's call, three tier-1 files
not fixed).

## 4. Tests that assert a property of the machine, not of the code

**Instances.**
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

**The rule now.** At session start, `./scripts/status.sh` for `main` and the
nightly smoke; a known red step must not gate the steps after it. Status: 📋
`CLAUDE.md` start protocol; ✅ deps tier on every push; ⚠️ branch protection
needs the user to enable it on GitHub.

## 6. Statistical designs that could not have rejected

**Instances.**
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

**The rule now.** Compute the **attainable floor** (best possible p / max e)
of a design before running it, and print it on every record
(`max_attainable_E`). Status: ✅ in the runners since §3.51; 📋 for new designs.

## 7. Literature found after the construction was built

**Instances.** §3.22's self-repair framing turned out to be CoAx (published
2 months earlier), corrected same day by §3.28. Phase 1's framing ignored a
causal-mask theory that existed since Nov 2024 (`2411.04990`), which Pythia
should have been compared against all along. §3.52: reading that paper changed
three constructions, and F0 had measured a proxy.

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
- 2026-09-29 (#115): `git log --diff-filter=A -1 -- <record>` in the cloud
  container named `2225beb` as the commit that added P-I5's real-run record,
  and the code comment cited it. The clone is shallow (`git rev-parse
  --is-shallow-repository` → true) and `2225beb` is its boundary: every file
  "is added" there. The record was added at `bff93d7`, before battery v2. The
  hash happened to match at both. Found by `/challenge-pr`. On a shallow
  clone, ask the API for history (`/commits?path=`), not `git log`.

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

## 11. The author's own review cannot see what the author believes

**What happens.** A design is checked by the session that built it, which
re-reads it through the reasoning that produced it. Where that reasoning holds
a wrong premise, the check inherits it.

**Instances.**
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

**The rule now.** Every PR gets `/challenge-pr` (fresh context, intent and
design, not lines) and the author answers each finding on the PR; the user
settles disagreements. Status: 📋 `CLAUDE.md` Stop step 9 (lands with #68).

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
