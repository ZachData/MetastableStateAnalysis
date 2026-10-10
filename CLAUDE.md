<!-- CLAUDE.md -->
# How we work in this repo

`STATE.md` is printed into context at session start (hook in
`.claude/settings.json`). It is the only file to read before starting; it says
what is true now, and `docs/ROADMAP.md` says what comes next. Open anything
else only when the task needs it. `LESSONS.md` explains why each rule exists.

## Start (every session, before any work)

1. Read `STATE.md` (already in context via the hook). If it is missing or
   older than the newest commit on `origin/main`, say so first.
2. `git fetch`, then `./scripts/status.sh`: one line each for `main` CI, the
   nightly smoke and open PRs. Report any red.
3. Verify any claim from `STATE.md` you are about to act on against the tree
   (merged? exists? populated?). Docs have been wrong about all three.
4. Work in the task worktree `../Mets-work` on a fresh branch from
   `origin/main`. The main tree stays on `main`; another session may be in it.

## While working

- **One session per unit of work.** After the Stop protocol, start a fresh
  session; context re-read every call is the cost driver (`LESSONS.md` 9).
- **Batch independent checks** into one call (one command, or parallel calls).
- **Edit tracked docs with the Edit tool**, not `sed`/heredoc (it re-injects the file).
- **Tangent triage.** Classify every surprise as a defect (fix now), confound
  (attach to the current item) or discovery (park one line under "Parked" in
  the thread's handoff: why / cost / which decision it could change). Do not
  follow a discovery without saying so to the user.
- **Refuse rather than degrade.** Before launching a batch, open the first
  output and check it is *populated*, not just present.
- **Run-and-watch goes to the `runner` agent** (Haiku; brief, NUMERICS,
  QUICK, escalation: `.claude/agents/runner.md`). Read its report, not the
  log; `git diff` its edits, `[NUMERIC]` ones before results. `TO: sonnet`
  → `runner` with `model: sonnet`, report as PRIOR; `TO: main` → you.
  Watching yourself: one notification, the exit. Calls per batch: step 7.
- **Name the input.** A recorded result states the exact input set (battery
  hash, key list, env) it ran on.
- **Write once.** A number lives in one file; others point to it. Tables over
  paragraphs, one claim per line. For `PROJECT.md` and `POPPER_PLAN.md`, open
  `docs/index/<name>.idx.md` and read one section by line range; never read them whole.
- **Math.** Closed-form claims get a `sympy` check in `tools/math_checks/`,
  noting what it does *not* prove. Empirical numbers get their producer re-run.

## Phases and tests (`docs/ROADMAP.md` §3; `LESSONS.md` 6, 10)

- **A phase or thread opens with an exit condition and a unit budget** in its
  design. At the budget it closes at what it knows, unless the user extends it
  on the record. A new idea mid-phase goes into `docs/ROADMAP.md`, not a new phase.
- **Close out before the next opens:** card current, a one-page "what it
  established", "not established" current, every Parked item *carried to X*
  or *dropped (why)*, every Blocked item decided or given a home.
- **Calibration gate.** Before a rule freezes, run the full pipeline on a
  planted null and a planted effect over the real design points. Quote effect
  sizes against a positive control where one exists.
- **Name the family.** The cells and the statistic that count are fixed before
  any output; leans are not readings. 410m and 70m are exploration rungs; a
  confirmatory claim names its rung or held-out set when it is registered.

## Stop (when a unit of work closes: a result, a defect, a decision)

Do these without being asked, in this order, in the same commit as the work:

1. **`STATE.md`**: overwrite what changed (Now, Blocked, Open PRs, Where
   things stand) and bump **Last updated**. Keep it ≤ 150 lines; move detail
   out rather than trimming facts.
2. **The phase's `status-N.md`**: numbers, caveats, how to re-run. If the
   unit corrects an *earlier* phase (a result, a definition, a citation),
   also add one line under that phase's `## Corrections received`: date,
   what changed, pointer. That line stales the earlier phase's card and every
   card that reads it (`docs/phase_card.md`).
3. **The active thread's handoff** (e.g. `p10_cluster_function/handoff-10.md`):
   stage state, Parked items.
4. **`LESSONS.md`**: if anything went wrong (a stale doc, a silent failure,
   a wrong assumption, a wasted run), add the instance under its pattern.
5. **`PROJECT.md`** only for a project-wide result, as a new §3.x section.
6. **`INDEX.md`** only if a directory or phase was added or moved;
   **`docs/ROADMAP.md`** if a phase closed, the order changed or a §10 decision was taken.
7. **`docs/cost_log.md`**: `python tools/session_cost.py <transcript> --row
   "<unit>" --pr "#N"` and append the row (transcript:
   `~/.claude/projects/<project>/<session>.jsonl`). Over 2× the running median
   total context → add a line saying why.
8. Run `./scripts/check.sh` (it includes the doc caps), commit, push, and
   **open a PR** if the unit is coherent on its own. Put a **"Worth
   challenging"** section in its description: the choices you are least sure
   of and the alternatives you rejected. Then comment **`@coderabbitai
   review`** on it (it does not review this repo on its own; `LESSONS.md` 5).
9. **Invoke `/challenge-pr <N>`** on the PR you just opened. It runs in a
   forked subagent with no conversation history, reviews intent and design
   choices, and posts one PR comment naming the commit it reviewed. Pass it
   only the number; do not brief it, because the point is that it has not
   seen your reasoning. When it returns, answer every finding **on the PR**:
   fix it (and push), or reply saying why not. Fixes pushed here re-open
   steps 1–3 if they change what those files say. Then tell the user the PR
   link, the verdict, and what is now blocked on them. **The user decides
   disagreements; neither side's argument settles one.** Re-take the step 7
   row after it returns (its tokens count). Terminal: `claude -p "/challenge-pr <N>"`.

Sessions end abruptly (context, watchdog, sleep), so do not batch these to
the end of the session.

## Git and PRs

- One coherent piece of work per PR (an invariant read, a defect fixed, a
  runner parametrised); the bound is reviewer attention, not commit count.
- Target `main` only, never another PR's branch. After a merge, confirm with
  `git merge-base --is-ancestor <tip> origin/main`.
- `git fetch` and recheck HEAD before every commit.
- Push with the default SSH key. Never set `GIT_SSH_COMMAND`.
- The user merges on GitHub. After a merge: `git worktree remove ../Mets-work`
  and delete the branch.
- Never `git add` under `data/`.

## Literature scans and the registry

- Scan when a phase or subphase opens (before its `design-N.md` freezes) and
  before an entry lands in `claims/registry.json`; at a phase's end only if it
  produced something publishable.
- `claims/registry.json` is untouched unless the user deliberately registers
  something, **before** looking at the data it will be scored on.
