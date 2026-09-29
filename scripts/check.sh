#!/usr/bin/env bash
# scripts/check.sh — what CI runs, runnable locally (archive/POPPER_PLAN-done.md item A6).
#
# CI calls this script rather than reimplementing its commands, so "passes
# locally, fails in CI" cannot be a difference between two copies of the same
# command list.
#
#   ./scripts/check.sh          tier 0 + tier 1 — the fast part of the merge gate
#   ./scripts/check.sh lint     tier 0 only; needs no dependencies at all, and
#                               is RUN with them unimportable so that stays true
#   ./scripts/check.sh pure     tier 1 only; needs requirements/test.txt
#   ./scripts/check.sh iso      tier 1 with heavy deps forced absent (what CI has)
#   ./scripts/check.sh deps     tier 3 only; needs requirements/heavy.txt. CPU
#                               torch is enough: no test needs a GPU. CI runs it
#                               on every push (ci.yml job `deps`)
#   ./scripts/check.sh all      lint + iso + deps
#
# The pure tier ran 2782 tests in 72 s (2026-09-29) with torch, transformers,
# scikit-learn and matplotlib all absent. That speed is the point: a gate
# people wait on is a gate people route around.

set -euo pipefail

cd "$(dirname "$0")/.."
TARGET="${1:-gate}"

# Shadow a list of modules with packages that raise ImportError, so code which
# claims not to need them is actually run without them. Echoes the directory;
# the caller is responsible for removing it.
make_shadow() {
  local shadow m
  shadow="$(mktemp -d)"
  for m in "$@"; do
    mkdir -p "$shadow/$m"
    printf 'raise ImportError("%s blocked: dependency isolation, scripts/check.sh")\n' \
      "$m" > "$shadow/$m/__init__.py"
  done
  echo "$shadow"
}

run_lint() {
  # Tier 0's contract is "no dependencies at all", and CI installs none for it,
  # so tier 0 is run here with them genuinely unimportable. This is not
  # belt-and-braces: `python -m core.adjudication --verify` joined tier 0 with
  # `core/evalues.py` behind it, that module imported numpy at scope, and the
  # job went red on a runner that correctly had nothing installed -- while
  # passing on every developer machine, which all have numpy. Exactly the gap
  # `run_pure_isolated` exists to close, one tier up.
  local shadow
  shadow="$(make_shadow numpy scipy torch transformers sklearn matplotlib)"
  trap 'rm -rf "$shadow"' RETURN
  PYTHONPATH="$shadow${PYTHONPATH:+:$PYTHONPATH}" _lint_commands
}

_lint_commands() {
  echo "=== tier 0: repo hygiene (heavy deps AND numpy/scipy unimportable) ==="
  python3 tools/lint_repo.py

  echo
  echo "=== tier 0: prediction registry + pre-registration gate ==="
  python3 tools/check_registry.py --summary

  echo
  echo "=== tier 0: EVALUABILITY.md in step with the registry ==="
  python3 tools/render_evaluability.py --check

  echo
  echo "=== tier 0: EXPERIMENTS.md in step with the registry ==="
  # The phase -> experiment -> prediction -> gate join. Stale here means a phase
  # gained or lost a falsifier and the map still shows the old shape, which is
  # the one question this file exists to answer.
  python3 tools/render_experiments.py --check

  echo
  echo "=== tier 0: ledger recomputes to what it claims ==="
  # Replays every claim's e-process from the committed adjudication records,
  # recalibrating each e-value from its p-value rather than trusting the stored
  # number. Catches arithmetic drift and, more usefully, a decision word
  # updated by hand without its evidence.
  python3 -m core.adjudication --verify

  echo
  echo "=== tier 0: FALSIFICATION.md in step with the ledger ==="
  python3 tools/render_falsification.py --check

  echo
  echo "=== tier 0: docs/PHASES.md in step with the phase cards ==="
  # The cards themselves (fields, pointers, staleness) are lint rule phase-card.
  python3 tools/render_phases.py --check

  echo
  echo "=== tier 0: docs/index/ in step with PROJECT.md, POPPER_PLAN.md ==="
  # A stale index sends a session to the wrong line range; after editing either
  # document, run `python3 tools/build_doc_index.py`.
  python3 tools/build_doc_index.py --check
}

run_pure() {
  echo
  echo "=== tier 1: pure tests (no torch) ==="
  python3 -m pytest -m pure -q
}

run_pure_isolated() {
  # Run tier 1 with the heavy deps made genuinely unimportable, which is what
  # CI's runner actually has. Two red CI runs came from the gap this closes:
  # a developer machine with torch installed passes tests that a torch-free
  # runner fails, because the conftest's MagicMock stub only takes effect when
  # the real module is absent. `-m pure` on a machine that HAS torch does not
  # test the pure tier's central claim.
  #
  # Implemented by shadowing each module with a package that raises ImportError,
  # rather than by uninstalling anything.
  echo
  echo "=== tier 1 (isolated): pure tests with heavy deps genuinely absent ==="
  local shadow
  shadow="$(make_shadow torch transformers sklearn matplotlib)"
  trap 'rm -rf "$shadow"' RETURN
  PYTHONPATH="$shadow${PYTHONPATH:+:$PYTHONPATH}" python3 -m pytest -m pure -q
}

run_deps() {
  echo
  echo "=== tier 3: deps tests (needs torch/transformers/sklearn/matplotlib) ==="
  # `heavy` needs real run artifacts no runner has; `smoke` needs the HF Hub
  # and runs in its own workflow.
  #
  # Refuse rather than degrade: tests/conftest.py's pytest_ignore_collect drops
  # every deps-tier module when torch, sklearn or matplotlib is missing, and the
  # hdbscan/igraph tests importorskip. Either way the run would "pass" having
  # tested nothing, so check the imports first.
  python3 -c "import torch, transformers, sklearn, matplotlib, hdbscan, igraph" \
    || { echo "deps tier: requirements/heavy.txt not importable; refusing to run a tier that would test nothing" >&2; exit 1; }
  python3 -m pytest -m "deps" -q -rs
}

case "$TARGET" in
  lint) run_lint ;;
  pure) run_pure ;;
  gate) run_lint; run_pure_isolated ;;
  iso)  run_pure_isolated ;;
  deps) run_deps ;;
  all)  run_lint; run_pure_isolated; run_deps ;;
  *)    echo "usage: $0 [lint|pure|iso|deps|gate|all]" >&2; exit 2 ;;
esac

echo
echo "check.sh: '$TARGET' OK"
