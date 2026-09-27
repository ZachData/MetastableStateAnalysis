You are running unattended on a research-vm fleet instance. Nobody is
watching this session; the user reads the PR you open.

Follow this repo's `CLAUDE.md` exactly: its Start protocol first (`STATE.md`
is already in context via the SessionStart hook), then its Stop protocol
when the unit closes. **One exception, the worktree rule:** this box runs one
session at a time, and the fleet's `wrapper.sh` checks for a clean tree,
`NEEDS_WORKERS` and CI in *this clone* on its current `HEAD`. So work here,
not in `../Mets-work`: `git checkout -b <branch> origin/main` in this clone,
and leave the clone on your pushed branch when you stop. (The venv's editable
install also points at this clone.)

Pick the next unit of work, in this order:

1. Open GitHub issues on this repo labelled `agent`
   (`gh issue list --label agent --state open`), oldest first.
2. The **Next** item of the active thread in `STATE.md`.

Do not start anything listed under "Blocked on the user" in `STATE.md`, do
not touch `claims/registry.json`, and do not release the held-out prompts.
If the next item needs one of those, or needs data this box does not have
under `data/`, say so **once**: first check for an open issue labelled
`agent-blocked` that already says it (`gh issue list --label agent-blocked`);
open one only if none does. Then stop without a PR and without edits.

Known on this box (aarch64), not yours to fix unless it is your unit:
`./scripts/check.sh` has 4 failures that pass on CI's x86 (`STATE.md`, fleet
box row; `LESSONS.md` §4). Your gate is "no failures beyond those 4"; list
any other failure in the PR.

Sweeps are not wired yet: workers launch from `main`, so a manifest on your
branch never reaches them. If the item is a sweep, write the plan in the PR
and stop; do not create `NEEDS_WORKERS`.

Never push to `main`. Commit, push a branch, open a PR (`gh pr create`) with
a "Worth challenging" section, for the user to review and merge. On this
box git pushes over HTTPS with the fleet PAT, not an SSH key.
