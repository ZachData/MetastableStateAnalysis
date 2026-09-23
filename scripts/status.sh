#!/usr/bin/env bash
# scripts/status.sh — the start protocol's step 2 in three lines:
# main CI, the nightly smoke, open PRs. Needs `gh` (authenticated); without it,
# says so in one line rather than guessing.
set -uo pipefail
cd "$(dirname "$0")/.."

if ! command -v gh >/dev/null 2>&1; then
  echo "status: gh not installed — check CI, smoke and PRs on GitHub by hand"
  exit 0
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "status: gh not authenticated (gh auth login) — check GitHub by hand"
  exit 0
fi

run_line() {  # $1 label, rest: gh run list filters
  local label="$1"; shift
  local out
  out=$(gh run list "$@" --limit 1 \
    --json status,conclusion,headSha,createdAt,url \
    -q '.[0] | "\(if .status == "completed" then .conclusion else .status end) \(.headSha[0:7]) \(.createdAt) \(.url)"' 2>&1) \
    || out="gh error: ${out%%$'\n'*}"
  echo "$label ${out:-no runs}"
}

run_line "main CI:     " --branch main --workflow ci.yml
run_line "nightly smoke:" --workflow smoke.yml --event schedule
prs=$(gh pr list --state open --json number,title \
  -q 'map("#\(.number) \(.title)") | join("; ")' 2>&1) || prs="gh error: ${prs%%$'\n'*}"
echo "open PRs:      ${prs:-none}"
