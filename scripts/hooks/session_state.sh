#!/usr/bin/env bash
# SessionStart hook: print the newest STATE.md, not the main tree's checkout.
# The main tree often lags origin/main (another session may be using it, so it
# is not pulled), and the hook used to print its stale copy (LESSONS 1, six
# instances 2026-11-01 .. 11-03). This fetches, prints origin/main's STATE.md,
# and names any unmerged remote branch that changed STATE.md since branching.
# The provenance lines come FIRST: long hook output is cut to a short preview.
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" 2>/dev/null || exit 0

if timeout 20 git fetch -q origin 2>/dev/null; then fetched="fetched now"
else fetched="FETCH FAILED (offline?): origin/main may itself be stale"; fi

if git rev-parse -q --verify origin/main >/dev/null && git cat-file -e origin/main:STATE.md 2>/dev/null; then
  src="origin/main $(git log -1 --format='%h %cs' origin/main)"
  show() { git show origin/main:STATE.md; }
else
  src="the working tree (no origin/main:STATE.md)"
  show() { cat STATE.md 2>/dev/null || echo 'STATE.md missing: say so before starting work (CLAUDE.md, Start step 1).'; }
fi

echo "[session_state] STATE.md below is from $src; $fetched."
head=$(git rev-parse --short HEAD 2>/dev/null)
behind=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo "?")
if [ "$behind" != "0" ]; then
  echo "[session_state] this tree ($(git branch --show-current 2>/dev/null) $head) is $behind commits behind origin/main; its files are not what is printed."
fi
for b in $(git branch -r --no-merged origin/main --format='%(refname:short)' 2>/dev/null); do
  if ! git diff --quiet "origin/main...$b" -- STATE.md 2>/dev/null; then
    echo "[session_state] unmerged $b changed STATE.md ($(git log -1 --format='%cs: %s' "$b" | cut -c1-120)); read it with: git show $b:STATE.md"
  fi
done
echo
show
