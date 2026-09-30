#!/usr/bin/env bash
# scripts/status.sh — the start protocol's step 2 in three lines:
# main CI, the nightly smoke, open PRs. Uses `gh` when it is installed and
# authenticated; otherwise GitHub's public REST API through curl (this repo is
# public, so no token is needed at ~3 calls per session).
#
# The fallback exists because cloud sessions have no `gh`: until 2026-09-29
# this script printed "check by hand" and exited 0 there, so step 2 of the
# start protocol degraded to nothing while the nightly smoke stayed red for 10
# nights (LESSONS.md lesson 5). A failed lookup now exits 1.
set -uo pipefail
cd "$(dirname "$0")/.."

api_status() {
  local slug
  slug=$(git remote get-url origin 2>/dev/null \
    | sed -E 's#/*$##; s#\.git$##; s#.*[:/]([^/:]+/[^/]+)$#\1#')
  if [ -z "$slug" ]; then
    echo "status: no origin remote and no gh — cannot check GitHub"
    return 1
  fi
  echo "status: gh unavailable; public API for $slug"
  python3 - "$slug" <<'PY'
import json, subprocess, sys

slug = sys.argv[1]


def get(path):
    r = subprocess.run(
        ["curl", "-sSf", "--connect-timeout", "10", "--max-time", "30",
         "-H", "Accept: application/vnd.github+json",
         f"https://api.github.com/repos/{slug}/{path}"],
        capture_output=True, text=True)
    if r.returncode:
        err = r.stderr.strip().splitlines()
        raise RuntimeError(err[-1] if err else f"curl exit {r.returncode}")
    return json.loads(r.stdout)


def run_line(label, path):
    try:
        runs = get(path)["workflow_runs"]
    except Exception as e:  # noqa: BLE001 - report, never guess
        print(f"{label} api error: {e}")
        return False
    if not runs:
        print(f"{label} no runs")
        return True
    r = runs[0]
    state = r["conclusion"] if r["status"] == "completed" else r["status"]
    print(f"{label} {state} {r['head_sha'][:7]} {r['created_at']} {r['html_url']}")
    return True


ok = run_line("main CI:     ", "actions/workflows/ci.yml/runs?branch=main&per_page=1")
ok &= run_line("nightly smoke:", "actions/workflows/smoke.yml/runs?event=schedule&per_page=1")
try:
    prs = get("pulls?state=open&per_page=50")
    print("open PRs:      " + ("; ".join(f"#{p['number']} {p['title']}" for p in prs) or "none"))
except Exception as e:  # noqa: BLE001
    print(f"open PRs:      api error: {e}")
    ok = False
sys.exit(0 if ok else 1)
PY
}

if ! command -v gh >/dev/null 2>&1 || ! gh auth status >/dev/null 2>&1; then
  api_status
  exit $?
fi

rc=0
run_line() {  # $1 label, rest: gh run list filters
  local label="$1"; shift
  local out
  out=$(gh run list "$@" --limit 1 \
    --json status,conclusion,headSha,createdAt,url \
    -q '.[0] | "\(if .status == "completed" then .conclusion else .status end) \(.headSha[0:7]) \(.createdAt) \(.url)"' 2>&1) \
    || { out="gh error: ${out%%$'\n'*}"; rc=1; }
  echo "$label ${out:-no runs}"
}

run_line "main CI:     " --branch main --workflow ci.yml
run_line "nightly smoke:" --workflow smoke.yml --event schedule
prs=$(gh pr list --state open --json number,title \
  -q 'map("#\(.number) \(.title)") | join("; ")' 2>&1) || { prs="gh error: ${prs%%$'\n'*}"; rc=1; }
echo "open PRs:      ${prs:-none}"
exit "$rc"
