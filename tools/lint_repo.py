#!/usr/bin/env python3
"""
tools/lint_repo.py — tier-0 repo hygiene (archive/POPPER_PLAN-done.md item A5).

Encodes the project's own standing rules as machine checks. Every rule here
exists because a defect of that shape already cost real work in this repo; the
rule stops it recurring rather than catching it again by hand.

The standing rules (this docstring is their home since 2026-09-22; they were
§6 of `archive/UPDATE_PLAN.md`, which keeps the history):

S1. **If a quantity appears in a report, it is persisted.** D2's per-head
   Fiedler existed only in the session that produced it.
S2. **Every data-dependent fallback records the branch it took.** On a model
   where no eigengap ever exists, the fallback *is* the metric.
S3. **Every gate records which quantity it read and whether it passed**, per
   layer. A gate reading a constant that may since have changed cannot be
   reconstructed from the artifact.
S4. **Refuse rather than degrade.** No unit-norm substitute for missing norms,
   no inferred revision, no invented beta, no silent raw-frame fallback. A
   number from mismatched inputs is worse than no number: it is unfalsifiable
   from the output alone.
S5. **Anchors need a non-symmetric arm.** A trace contraction that was wrong
   for every non-symmetric M passed its M = I anchor (`archive/UPDATE_PLAN.md`
   §5.6).
S6. **A threshold that has not been derived from a distribution is labelled as
   placed, not calibrated** -- in the code, next to the value.
S7. **No hand-synced constants** (`archive/UPDATE_PLAN.md` §4): a
   constant duplicated across modules with a comment asking editors to keep
   it in step is a defect; parse the one definition instead.

S1-S5 are review rules, not machine checks; S6 is `threshold-provenance` below
and S7 `hand-synced-constant`. "Rule N" further down numbers the checks
(`--list`), not these.

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
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parent.parent

#: Directories that are Python packages of this project.
#: LIVE packages only. Phases 3, 4, 5, 5b, 5c and 6 moved to archive/ on
#: 2026-08-22 and are deliberately absent: archive/README.md's first rule is
#: "not maintained, not imported, not collected", and linting them would
#: reintroduce exactly the maintenance the archive exists to end.
PACKAGE_DIRS = (
    "core", "tools", "tests",
    "p1_mstate_tracking", "p1b_hemisphere", "p1c_frames",
    "p2_eigenspectra", "p2b_imaginary", "p2d_operator_activation",
    # p6_subspace is live again as of 2026-08-24: the projector path was
    # REBUILT here against core/particles.py (archive/README.md rule 2 --
    # nothing is salvaged by copying) so that P6-R2 and P6-R4 could come out of
    # `dormant`. archive/p6_subspace/ stays frozen and stays out of this list.
    "p6_subspace",
    "p7_motifs",
)

#: The tier markers pytest.ini registers. Rule 2 requires exactly one.
#: `deps` was missing here while being a registered marker, so every
#: deps-tier module counted as unmarked and the rule under-reported.
TIER_MARKERS = ("pure", "deps", "smoke", "heavy")


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
            # An error since 2026-08-30, as the retrofit comment here always
            # said it should become. The last seven unmarked modules -- five of
            # them Phase 7's -- were marked `pure` that day, measured under the
            # heavy four made unimportable rather than assumed. They had been
            # invisible since e6d7dba (2026-08-22) opened Phase 7 one day
            # before 4fb460d introduced the taxonomy, and 221 tests, 8% of the
            # suite, ran in no tier at all: `-m pure` and `-m deps` between
            # them selected everything EXCEPT these, so nothing executed them
            # locally or in CI and nothing failed to say so. A warning did not
            # stop that, which is the argument for the promotion.
            lint.error(
                "test-tier-marker", py, 0,
                "no tier marker (pure/deps/smoke/heavy); CI's -m partition "
                "cannot place it deterministically, so it runs in no tier",
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
Standing rule S7 (module docstring): a numeric constant duplicated across
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
Standing rule S6 (module docstring): 'A threshold that has not been derived from
a distribution is labelled as placed, not calibrated -- in the code, next to the
value.' archive/UPDATE_PLAN.md §5.7 is why: Q_k cannot be compared against a fixed tolerance because
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


# ---------------------------------------------------------------------------
# Rule 6 — the startup docs stay small enough to be read
# ---------------------------------------------------------------------------

RULE_6_WHY = """\
LESSONS.md lessons 1 and 9. PROJECT.md grew to 485 KB (~120k tokens) while
calling itself the file to read first; too big to read carefully, so stale
lines in it survived and the next session acted on them. STATE.md replaced its
resume block as the one startup read, and is printed into every session by a
hook -- so its size is a per-session token cost, and a cap is what keeps it an
overwrite-in-place summary rather than a second append-only diary."""

#: (file, max lines, must carry a "Last updated" line)
DOC_CAPS = (
    ("STATE.md",  150, True),
    ("CLAUDE.md", 120, False),
)

_LAST_UPDATED = re.compile(r"\*\*Last updated:\*\*\s*\d{4}-\d{2}-\d{2}")


def rule_startup_doc_caps(lint: Linter) -> None:
    for name, max_lines, needs_date in DOC_CAPS:
        path = ROOT / name
        if not path.is_file():
            lint.error("startup-doc-cap", path, 0, "missing; CLAUDE.md's start protocol reads it")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > max_lines:
            lint.error(
                "startup-doc-cap", path, max_lines + 1,
                f"{len(lines)} lines, cap {max_lines}: move detail to the file it "
                f"belongs in and point to it, rather than trimming facts",
            )
        if needs_date and not any(_LAST_UPDATED.search(l) for l in lines[:10]):
            lint.error("startup-doc-cap", path, 1,
                       "no '**Last updated:** YYYY-MM-DD' in the first 10 lines")


# ---------------------------------------------------------------------------
# Rule 7 — every cited .md path resolves, or the archive map says where it went
# ---------------------------------------------------------------------------

RULE_7_WHY = """\
A citation that resolves to nothing is the documentation form of "refuse
rather than degrade": the reader follows it, finds nothing, and either stops
or guesses. INDEX.md already listed three such absences found by hand, and
the 2026-09-22 archive batch moved ten files and sixteen sections that live
docs and code cite. So: every `.md` path cited in a live file (outside
archive/, data/ and tests/, whose fixtures name files that do not exist on
purpose) must exist, or be in archive/MOVED.md's Moved table (old -> new), or
in its Absent table with the reason it is cited anyway. A bare file name
resolves if any file in the tree has that name; a path with a directory
resolves from the repo root or from the citing file's directory.
`tools/rewrite_moved_refs.py` applies the Moved table to live files."""

MOVED_MAP = "archive/MOVED.md"

#: Not walked: frozen, data, test fixtures, tool caches. archive/ is still
#: indexed for bare-name resolution.
_CITE_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "data", "archive", "tests", "__pycache__",
    "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache",
})
_CITE_SUFFIXES = (".md", ".py", ".sh", ".yml", ".yaml", ".toml", ".ini")
_CITED_MD = re.compile(r"(?<![\w./<{}-])((?:[\w.-]+/)*[\w-]+(?:\.[\w-]+)*\.md)(?![\w/-])")
#: `status-N.md`, `lit-N.md`: a pattern naming a family of files, not a file.
_PLACEHOLDER = re.compile(r"(?:^|[-_/])[NX](?:\.|/)")


def _map_rows(section: str) -> set[str]:
    path = ROOT / MOVED_MAP
    if not path.is_file():
        return set()
    out, inside = set(), False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.strip() == f"## {section}"
            continue
        m = re.match(r"^\|\s*`([^`]+)`\s*\|", line) if inside else None
        if m and "§" not in m.group(1):
            out.add(m.group(1))
    return out


def _tracked() -> list[Path] | None:
    """`git ls-files` under ROOT, or None outside a git checkout (test trees).
    The rule reads the tracked set so it sees what CI sees: an untracked or
    git-ignored file must not decide a local run (`LESSONS.md` lesson 3)."""
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                             check=True).stdout.decode("utf-8").split("\0")
    except (OSError, subprocess.CalledProcessError):
        return None
    if not (ROOT / ".git").exists():
        return None
    return [ROOT / f for f in out if f and (ROOT / f).is_file()]


def _walk(skip: frozenset) -> Iterable[Path]:
    tracked = _tracked()
    if tracked is not None:
        yield from (p for p in tracked
                    if not set(p.relative_to(ROOT).parts[:-1]) & skip)
        return
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(x for x in dirs if x not in skip and not x.endswith(".egg-info"))
        for f in sorted(files):
            yield Path(d) / f


def rule_cited_md_paths(lint: Linter) -> None:
    names = {p.name for p in _walk(frozenset({".git", ".venv", "venv", "data", "__pycache__"}))
             if p.suffix == ".md"}
    moved = _map_rows("Moved")
    known = moved | _map_rows("Absent")
    for old in sorted(moved):
        if "/" in old and (ROOT / old).exists():
            lint.error("cited-md-path", ROOT / MOVED_MAP, 0,
                       f"Moved row `{old}` exists again: its citations may mean the "
                       f"new file; delete the row or rename one of them")
    for path in _walk(_CITE_SKIP_DIRS):
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix not in _CITE_SUFFIXES or rel == MOVED_MAP:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for m in _CITED_MD.finditer(line):
                cited = m.group(1)[2:] if m.group(1).startswith("./") else m.group(1)
                head = cited.split("/", 1)[0]
                if _PLACEHOLDER.search(cited) or ("/" in cited and "." in head
                                                  and not head.startswith(".")):
                    continue                     # a family pattern, or a URL host
                if "/" in cited:
                    if (ROOT / cited).is_file() or (path.parent / cited).is_file():
                        continue
                elif cited in names:
                    continue
                if cited in known:
                    continue
                lint.error("cited-md-path", path, n,
                           f"`{cited}` does not exist and is not in {MOVED_MAP} "
                           f"(Moved: say where it went; Absent: say why it is cited)")


RULES = [
    ("orphan-module",         rule_no_orphan_modules,        RULE_1_WHY),
    ("test-tier-marker",      rule_test_tier_markers,        RULE_2_WHY),
    ("hand-synced-constant",  rule_no_hand_synced_constants, RULE_3_WHY),
    ("stale-status",          rule_status_doc_staleness,     RULE_4_WHY),
    ("threshold-provenance",  rule_threshold_provenance,     RULE_5_WHY),
    ("startup-doc-cap",       rule_startup_doc_caps,         RULE_6_WHY),
    ("cited-md-path",         rule_cited_md_paths,           RULE_7_WHY),
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
