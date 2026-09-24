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

**The rule now.** Anything whose result is recorded names the exact input
set it ran on (battery hash, key list) and refuses on a mismatch. Status:
⚠️ open; decision on P-I5 is with the user (`STATE.md`).

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

**The rule now.** At session start, check `gh run list` for `main` and the
nightly smoke. Status: 📋 `CLAUDE.md` start protocol; ⚠️ branch protection
needs the user to enable it on GitHub.

## 6. Statistical designs that could not have rejected

**Instances.**
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

**The rule now.** Target `main`; one coherent piece of work per PR; verify
with `git merge-base --is-ancestor`; worktree per task; fetch + recheck HEAD
before commit. Status: 📋 `CLAUDE.md`.

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
- 2026-09-22, #67: the Stage 0 driver said a deadline kill re-runs only "its
  unfinished prompts". `run_1` writes `pair_agreement.json` once, after the
  whole invocation, so a kill threw away up to a checkpoint's worth (~1.4 h)
  of finished runs and left orphan directories matching the hash and sha. The
  author's gate was green; a fresh-context `/challenge-pr` found it on its
  first run, `file:line` included.
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
- Before `/challenge-pr` existed, the 2026-09-12 merge was 51 commits with no
  second reader at all (why the PR-size rule exists).

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
