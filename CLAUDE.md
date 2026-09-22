<!-- CLAUDE.md -->
# How we work in this repo

`STATE.md` is printed into context at session start (hook in
`.claude/settings.json`). It is the only file to read before starting. Open
anything else only when the task needs it. `LESSONS.md` explains why each rule
below exists.

## Start (every session, before any work)

1. Read `STATE.md` (already in context via the hook). If it is missing or
   older than the newest commit on `origin/main`, say so first.
2. `git fetch` and check: is `main` CI green, is the nightly smoke green,
   which PRs are open? (`gh run list --branch main --limit 3`,
   `gh run list --workflow smoke.yml --limit 1`, `gh pr list`). Report any red.
3. Verify any claim from `STATE.md` you are about to act on against the tree
   (merged? exists? populated?). Docs have been wrong about all three.
4. Work in the task worktree `../Mets-work` on a fresh branch from
   `origin/main`. The main tree stays on `main`; another session may be in it.

## While working

- **Tangent triage.** Classify every surprise as a defect (fix now), confound
  (attach to the current item) or discovery (park one line under "Parked"
  in the active thread's handoff, with why / cost / which decision it could
  change). Do not follow a discovery without saying so to the user.
- **Refuse rather than degrade.** Before launching a batch, open the first
  output and check it is *populated*, not just present.
- **Name the input.** A recorded result states the exact input set (battery
  hash, key list, env) it ran on.
- **Write once.** A number lives in one file; others point to it. Tables over
  paragraphs, one claim per line. Grep `PROJECT.md`; never read it whole.
- **Math.** Closed-form claims get a `sympy` check in `tools/math_checks/`,
  noting what it does *not* prove. Empirical numbers get their producer re-run
  (`PROJECT.md` §7).

## Stop (when a unit of work closes: a result, a defect, a decision)

Do these without being asked, in this order, in the same commit as the work:

1. **`STATE.md`**: overwrite what changed (Now, Blocked, Open PRs, Where
   things stand) and bump **Last updated**. Keep it ≤ 150 lines; move detail
   out rather than trimming facts.
2. **The phase's `status-N.md`**: numbers, caveats, how to re-run.
3. **The active thread's handoff** (e.g. `p10_cluster_function/handoff-10.md`):
   stage state, Parked items.
4. **`LESSONS.md`**: if anything went wrong (a stale doc, a silent failure,
   a wrong assumption, a wasted run), add the instance under its pattern.
5. **`PROJECT.md`** only for a project-wide result, as a new §3.x section.
   Never edit its "Resume here" block; `STATE.md` replaced it.
6. **`INDEX.md`** only if a directory or phase was added or moved.
7. Run `./scripts/check.sh` (it includes the doc caps), commit, push, and
   **open a PR** if the unit is coherent on its own. Then tell the user the
   PR link and what is now blocked on them.

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

## Literature scans: two triggers

1. When a phase or subphase opens, before its `design-N.md` freezes.
2. Before an entry lands in `claims/registry.json`.

An end-of-phase scan is for a phase that produced something publishable.

## Registry

`claims/registry.json` is untouched unless the user deliberately registers
something, **before** looking at the data it will be scored on.
