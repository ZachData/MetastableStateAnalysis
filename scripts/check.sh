#!/usr/bin/env bash
# scripts/check.sh — what CI runs, runnable locally (POPPER_PLAN.md item A6).
#
# CI calls this script rather than reimplementing its commands, so "passes
# locally, fails in CI" cannot be a difference between two copies of the same
# command list.
#
#   ./scripts/check.sh          tier 0 + tier 1 — what gates a merge
#   ./scripts/check.sh lint     tier 0 only; needs no dependencies at all
#   ./scripts/check.sh pure     tier 1 only; needs requirements/test.txt
#   ./scripts/check.sh all      adds the deps tier; needs requirements/heavy.txt
#
# The pure tier runs in ~10 seconds against 1532 tests with torch,
# transformers, scikit-learn and matplotlib all absent. That speed is the
# point: a gate people wait on is a gate people route around.

set -euo pipefail

cd "$(dirname "$0")/.."
TARGET="${1:-gate}"

run_lint() {
  echo "=== tier 0: repo hygiene ==="
  python3 tools/lint_repo.py

  echo
  echo "=== tier 0: prediction registry + pre-registration gate ==="
  python3 tools/check_registry.py --summary

  echo
  echo "=== tier 0: EVALUABILITY.md in step with the registry ==="
  python3 tools/render_evaluability.py --check
}

run_pure() {
  echo
  echo "=== tier 1: pure tests (no torch) ==="
  python3 -m pytest -m pure -q
}

run_deps() {
  echo
  echo "=== tier 3: deps tests (needs torch/transformers/sklearn/matplotlib) ==="
  # `heavy` needs real run artifacts no runner has; `smoke` needs the HF Hub
  # and runs in its own workflow.
  python3 -m pytest -m "deps" -q
}

case "$TARGET" in
  lint) run_lint ;;
  pure) run_pure ;;
  gate) run_lint; run_pure ;;
  all)  run_lint; run_pure; run_deps ;;
  *)    echo "usage: $0 [lint|pure|gate|all]" >&2; exit 2 ;;
esac

echo
echo "check.sh: '$TARGET' OK"
