#!/usr/bin/env bash
# SessionStart hook: print the newest STATE.md, not the main tree's checkout.
# The main tree often lags origin/main (another session may be using it, so it
# is not pulled), and the hook used to print its stale copy (LESSONS 1, six
# instances 2026-11-01 .. 11-03). This fetches, prints origin/main's STATE.md,
# and flags a merged PR whose branch kept commits after its merge (#306's
# freeze) apart from branches still in flight.
# The [session_state] lines come FIRST: long hook output is saved to a file and
# only a short preview reaches context (/challenge-pr on #307, finding 1).
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" 2>/dev/null || exit 0

if timeout 20 git fetch -q --prune origin 2>/dev/null; then fetched="fetched now"
else fetched="FETCH FAILED (offline?): origin/main is as old as the last fetch"; fi

if git rev-parse -q --verify origin/main >/dev/null && git cat-file -e origin/main:STATE.md 2>/dev/null; then
  src="origin/main $(git log -1 --format='%h %cs' origin/main)"
  show() { git show origin/main:STATE.md; }
  have_main=1
else
  src="the working tree (no origin/main:STATE.md)"
  show() { cat STATE.md 2>/dev/null || echo 'STATE.md missing: say so before starting work (CLAUDE.md, Start step 1).'; }
  have_main=0
fi

echo "[session_state] STATE.md below is from $src; $fetched."
echo "[session_state] If this output was cut to a preview, read the saved full output, NOT ./STATE.md (it may lag)."
if [ "$have_main" = 1 ]; then
  behind=$(git rev-list --count HEAD..origin/main 2>/dev/null)
  if [ -n "$behind" ] && [ "$behind" != "0" ]; then
    echo "[session_state] this tree ($(git branch --show-current 2>/dev/null) $(git rev-parse --short HEAD)) is $behind commits behind origin/main; its files are not what is printed."
  fi
  inflight=""
  for b in $(git branch -r --no-merged origin/main --format='%(refname:short)' 2>/dev/null); do
    name=${b#origin/}
    [ "$name" = "HEAD" ] && continue
    merge=$(git log origin/main --merges -1 --format='%h %s' --grep="from [^ ]*/$name\$" 2>/dev/null)
    if [ -n "$merge" ]; then
      n=$(git rev-list --count "origin/main..$b")
      echo "[session_state] MERGED WITHOUT ITS TIP: $b was merged ($merge) but $n of its commits never reached main; they may hold a decision main lacks: git log origin/main..$b"
    else
      inflight="$inflight $name"
    fi
  done
  [ -n "$inflight" ] && echo "[session_state] in flight, not on main (proposals, not state):$inflight"
fi
echo
show
