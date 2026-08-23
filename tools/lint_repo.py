#!/usr/bin/env python3
"""
tools/lint_repo.py — tier-0 repo hygiene (POPPER_PLAN.md item A5).

Encodes the project's own standing rules (`UPDATE_PLAN.md` §6) as machine
checks. Every rule here exists because a defect of that shape already cost
real work in this repo; the rule stops it recurring rather than catching it
again by hand.

Design constraints, both deliberate:

* **Standard library only.** This runs as CI tier 0 with no `pip install` step
  at all, so a hygiene failure is reported in seconds rather than behind a
  dependency resolve. Nothing here may import numpy.
* **AST or text, never import.** Importing project modules to inspect them
  would need the heavy tier and would execute module-level code. Every check
  below reads files.

Exit code is 0 when no rule fails, 1 otherwise. Warnings do not fail the run;
they are printed so a heuristic rule can be useful without being a gate.

Usage
-----
    python tools/lint_repo.py            # all rules
    python tools/lint_repo.py --list     # what is checked, and why
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parent.parent

#: Directories that are Python packages of this project.
PACKAGE_DIRS = (
    "core", "tools", "tests",
    "p1_mstate_tracking", "p1b_hemisphere", "p1c_frames",
    "p2_eigenspectra", "p2b_imaginary", "p2d_operator_activation",
    "p3_crosscoder", "p4_mstate_features",
    "p5_single_mstate_analysis", "p5b_manifold_steering", "p5c_unclustered",
    "p6_subspace",
)

#: The tier markers pyproject.toml registers. Rule 2 requires exactly one.
TIER_MARKERS = ("pure", "smoke", "heavy")


@dataclass
class Finding:
    rule: str
    path: str
    line: int
    message: str
    severity: str = "error"      # "error" fails the run; "warning" does not

    def render(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"{self.severity.upper():7s} [{self.rule}] {loc}: {self.message}"


@dataclass
class Linter:
    findings: List[Finding] = field(default_factory=list)

    def error(self, rule: str, path: Path, line: int, message: str) -> None:
        self.findings.append(Finding(rule, self._rel(path), line, message, "error"))

    def warn(self, rule: str, path: Path, line: int, message: str) -> None:
        self.findings.append(Finding(rule, self._rel(path), line, message, "warning"))

    @staticmethod
    def _rel(path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(ROOT))
        except ValueError:                       # pragma: no cover - defensive
            return str(path)


# ---------------------------------------------------------------------------
# Rule 1 — no orphan modules
# ---------------------------------------------------------------------------

RULE_1_WHY = """\
A .py file inside a package directory whose name is not a valid Python
identifier can never be imported, so nothing type-checks it, no test covers it,
and it drifts silently. This project already carried one: `core/.py`, a 196-line
truncated copy of models.py whose docstring asserted bfloat16 model loading
while the live core/models.py asserts float32 and calls that choice
load-bearing. Two contradictory statements of a policy the project treats as
critical, with no mechanism able to notice."""


def rule_no_orphan_modules(lint: Linter) -> None:
    for pkg in PACKAGE_DIRS:
        d = ROOT / pkg
        if not d.is_dir():
            continue
        for py in sorted(d.glob("*.py")):
            stem = py.stem
            if not stem.isidentifier():
                lint.error(
                    "orphan-module", py, 0,
                    f"{py.name!r} is not importable (stem {stem!r} is not a valid "
                    f"Python identifier); delete it or give it a real name",
                )


# ---------------------------------------------------------------------------
# Rule 2 — every test module declares exactly one tier marker
# ---------------------------------------------------------------------------

RULE_2_WHY = """\
CI partitions the suite by marker: tier 1 runs `-m pure`, the smoke workflow
runs `-m smoke` and `-m "not heavy"`. A test module with no tier marker falls
into whichever half the partition happens to leave it in, which means it can
stop being run without anything failing. Before this rule the `smoke` marker
was not even registered (pyproject.toml did not exist and the pytest.ini
tests/SMOKE_TESTS_NOTES.md refers to was never written), so `-m smoke`
selected nothing at all and every marked module raised
PytestUnknownMarkWarning."""


def _module_markers(tree: ast.Module) -> set[str]:
    """Markers applied at module scope via `pytestmark = ...`."""
    found: set[str] = set()

    def collect(node: ast.AST) -> None:
        # pytest.mark.<name> / pytest.mark.<name>(...)
        if isinstance(node, ast.Call):
            collect(node.func)
            return
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "mark":
                found.add(node.attr)
            return
        if isinstance(node, (ast.List, ast.Tuple)):
            for elt in node.elts:
                collect(elt)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "pytestmark" in targets:
                collect(node.value)
    return found


def rule_test_tier_markers(lint: Linter) -> None:
    tests_dir = ROOT / "tests"
    if not tests_dir.is_dir():
        return
    for py in sorted(tests_dir.glob("test_*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            lint.error("test-tier-marker", py, exc.lineno or 0, f"cannot parse: {exc.msg}")
            continue
        markers = _module_markers(tree) & set(TIER_MARKERS)
        if len(markers) == 0:
            # Reported as a warning during the retrofit: ~100 existing modules
            # predate the taxonomy and marking them all is its own chunk
            # (POPPER_PLAN.md A3 follow-up). Promote to error once that lands.
            lint.warn(
                "test-tier-marker", py, 0,
                "no tier marker (pure/smoke/heavy); CI's -m partition cannot "
                "place it deterministically",
            )
        elif len(markers) > 1:
            lint.error(
                "test-tier-marker", py, 0,
                f"declares multiple tier markers {sorted(markers)}; exactly one",
            )


# ---------------------------------------------------------------------------
# Rule 3 — no hand-synced constants
# ---------------------------------------------------------------------------

RULE_3_WHY = """\
Standing rule from UPDATE_PLAN.md §4: a numeric constant duplicated across
modules with a comment asking editors to keep it in step is a defect, not a
convention. The project already hit it -- checkpoint_scalars.py carried a
hand-synced copy of ENERGY_VIOLATION_REL_TOL, fixed by parsing the constant out
of core/metrics.py with `ast` so a rename raises at import instead of silently
reading a stale value. This rule finds the next one."""

_SYNC_COMMENT = re.compile(
    r"#.*\b(keep\s+in\s+sync|kept\s+in\s+sync|must\s+match|mirror(?:s|ed)?\s+the|"
    r"remember\s+to\s+update|sync(?:ed)?\s+with|duplicate[sd]?\s+of)\b",
    re.IGNORECASE,
)
_ASSIGNS_LITERAL = re.compile(r"^\s*[A-Z_][A-Z0-9_]*\s*(?::[^=]+)?=\s*[-+]?[\d.]")


def rule_no_hand_synced_constants(lint: Linter) -> None:
    for py in _project_py_files():
        try:
            lines = py.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:               # pragma: no cover - defensive
            continue
        for i, line in enumerate(lines, start=1):
            if not _SYNC_COMMENT.search(line):
                continue
            # Only flag when the sync comment sits on, or directly above, an
            # actual constant assignment -- prose in a docstring describing the
            # pattern is not the defect.
            window = [line]
            if i < len(lines):
                window.append(lines[i])
            if any(_ASSIGNS_LITERAL.match(w) for w in window):
                lint.error(
                    "hand-synced-constant", py, i,
                    "constant kept in step by comment; import it from its "
                    "defining module, or parse it out with ast the way "
                    "checkpoint_scalars.py does",
                )


# ---------------------------------------------------------------------------
# Rule 4 — status-doc staleness
# ---------------------------------------------------------------------------

RULE_4_WHY = """\
INDEX.md records two live instances: readme-phase2c.md and README_phase6.md both
say 'Not started' while their own results data show partial or complete runs.
A status line that contradicts the phase's own status-N.md is worse than no
status line, because the reader has no way to know which one is current. This
rule fails when a phase directory contains a doc asserting 'Not started' while
its status-N.md says otherwise."""

_NOT_STARTED = re.compile(r"^\s*(?:#+\s*)?.*\bnot\s+started\b", re.IGNORECASE)


def rule_status_doc_staleness(lint: Linter) -> None:
    for pkg in PACKAGE_DIRS:
        d = ROOT / pkg
        if not d.is_dir():
            continue
        status_files = list(d.glob("status-*.md"))
        if not status_files:
            continue
        status_text = "\n".join(f.read_text(encoding="utf-8") for f in status_files)
        status_says_not_started = bool(
            re.search(r"\*\*Overall:\*\*\s*Not started", status_text, re.IGNORECASE)
        )
        for md in sorted(d.glob("*.md")):
            if md.name.startswith("status-"):
                continue
            head = md.read_text(encoding="utf-8").splitlines()[:15]
            for i, line in enumerate(head, start=1):
                if _NOT_STARTED.search(line) and not status_says_not_started:
                    lint.error(
                        "stale-status", md, i,
                        f"header says 'Not started' while {status_files[0].name} "
                        f"does not; the status-N.md is the source of truth "
                        f"(INDEX.md, 'Two things worth knowing')",
                    )
                    break


# ---------------------------------------------------------------------------
# Rule 5 — thresholds are labelled placed or calibrated
# ---------------------------------------------------------------------------

RULE_5_WHY = """\
Standing rule 6 (UPDATE_PLAN.md §6): 'A threshold that has not been derived from
a distribution is labelled as placed, not calibrated -- in the code, next to the
value.' §5.7 is why: Q_k cannot be compared against a fixed tolerance because
E[Q_k] = 1/n exactly for i.i.d. points, so every large-n configuration reads as
a spherical design under an absolute threshold. An unlabelled threshold gives a
reader no way to tell a measured cut from a guessed one. Warning-level: the
retrofit across existing constants is its own chunk."""

_THRESHOLD_NAME = re.compile(r"^\s*([A-Z_][A-Z0-9_]*(?:THRESHOLD|TOL|CUTOFF|_MIN|_MAX))\s*=")
_LABELLED = re.compile(r"\b(placed|calibrated|derived\s+from)\b", re.IGNORECASE)


def rule_threshold_provenance(lint: Linter) -> None:
    for py in _project_py_files():
        if py.parts and "tests" in py.parts:
            continue
        try:
            lines = py.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:               # pragma: no cover - defensive
            continue
        for i, line in enumerate(lines, start=1):
            m = _THRESHOLD_NAME.match(line)
            if not m:
                continue
            context = "\n".join(lines[max(0, i - 4): i + 3])
            if not _LABELLED.search(context):
                lint.warn(
                    "threshold-provenance", py, i,
                    f"{m.group(1)} has no 'placed' / 'calibrated' label within "
                    f"3 lines (standing rule 6)",
                )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _project_py_files() -> Iterable[Path]:
    for pkg in PACKAGE_DIRS:
        d = ROOT / pkg
        if not d.is_dir():
            continue
        for py in sorted(d.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            yield py


RULES = [
    ("orphan-module",         rule_no_orphan_modules,        RULE_1_WHY),
    ("test-tier-marker",      rule_test_tier_markers,        RULE_2_WHY),
    ("hand-synced-constant",  rule_no_hand_synced_constants, RULE_3_WHY),
    ("stale-status",          rule_status_doc_staleness,     RULE_4_WHY),
    ("threshold-provenance",  rule_threshold_provenance,     RULE_5_WHY),
]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="print each rule and the defect it exists to prevent")
    ap.add_argument("--warnings-as-errors", action="store_true",
                    help="fail the run on warnings too")
    args = ap.parse_args(argv)

    if args.list:
        for name, _, why in RULES:
            print(f"\n=== {name} ===\n{why}")
        return 0

    lint = Linter()
    for _, fn, _ in RULES:
        fn(lint)

    errors = [f for f in lint.findings if f.severity == "error"]
    warnings = [f for f in lint.findings if f.severity == "warning"]

    for f in errors:
        print(f.render())
    for f in warnings:
        print(f.render())

    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s) "
          f"across {len(RULES)} rules.")
    if errors or (args.warnings_as_errors and warnings):
        return 1
    print("repo hygiene OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
