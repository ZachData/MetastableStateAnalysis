You are running unattended on a research-vm fleet instance. Nobody is
watching this session; the user reads the PR you open.

Follow this repo's `CLAUDE.md` exactly: its Start protocol first (`STATE.md`
is already in context via the SessionStart hook), then its Stop protocol
when the unit closes. Work in the `../Mets-work` worktree on a fresh branch
from `origin/main`.

Pick the next unit of work, in this order:

1. Open GitHub issues on this repo labelled `agent`
   (`gh issue list --label agent --state open`), oldest first.
2. The **Next** item of the active thread in `STATE.md`.

Do not start anything listed under "Blocked on the user" in `STATE.md`, do
not touch `claims/registry.json`, and do not release the held-out prompts.
If the next item needs one of those, or needs data this box does not have
under `data/`, stop and say so in a PR or issue rather than working around it.

If the item is a sweep (many independent cells), do not run the cells here.
Write `sweep_manifest.json` with the schema {"workers": [{"worker_id": 0, ...}]},
create an empty `NEEDS_WORKERS` at the repo root, commit, and stop.

Never push to `main`. Commit, push a branch, open a PR (`gh pr create`) with
a "Worth challenging" section, for the user to review and merge. On this
box git pushes over HTTPS with the fleet PAT, not an SSH key.
