#!/usr/bin/env python3
"""
tools/lit_ledger.py — the paper ledger, `docs/lit_ledger.md`: one row per arXiv
id, saying when the repo first cited it, every scan and file that cites it now,
and the strongest mark any scan has given it.

Why it exists. The scan files (`lit-*.md`) are written forwards: one per phase,
each listing the papers it used. The reverse view, "given a paper, where was it
used, since when, and how much of it did anyone actually read", was a grep with
no dates (C8 U0, 2026-11-27). The ledger only points; the verdicts stay in the
scan files it names.

It also holds the one definition of what a scan file is and what an index row
is, for `tools/lint_repo.py`'s `lit-index-row` rule (standard library only, so
tier 0 can import it).

What is read, and from where
----------------------------
* **Files**: everything `git add -A` would commit with suffix .md, .py, .tex or
  .bib, except `data/`, `tests/` (fixtures name ids on purpose), the ledger and
  this file.
* **Ids**: new-style arXiv ids `YYMM.NNNN` (to 1412) and `YYMM.NNNNN` (from 1501),
  version suffix dropped, YY 07-30 and MM 01-12. Old-style ids (`hep-th/9901001`)
  are not read.
* **Marks** (`docs/LITERATURE.md` §0): [N] < [S] < [A] < [H] < [R]. They are read
  only from scan files, `docs/LITERATURE.md` and `docs/readings/`, where that
  legend holds; elsewhere [R], [D], [W] ... are cost tags. On a line, an id takes
  a mark right after it (`ADJACENT`, same cell, before the next id). In a table row it
  may also take the row's mark if the row has one distinct mark, or the
  weakest if it has several (a tag column `[P] abstract; [S]` does not say
  whose is whose), or, if the row has no bracketed mark at all, the header
  row's mark if it has exactly one (`| work | what [S] | bearing |`). A mark
  after "not" is skipped. A mark on another line, or one given in prose ("every
  row is [S]", or not beside its id), is missed. Each choice
  errs low: the column can under-report a reading, not invent one.
  A file in `docs/readings/<id>.md` counts as [R]. File-local legends are in
  `LOCAL_MARKS`.
* **First cited**: the earliest author date of a commit whose diff adds a line
  holding the id (one `git log -p` pass; the whole build takes a few seconds). That is this repo's history: a
  paper cited before the history began dates from the import commit. Without git
  history (no `.git`, or a shallow clone) the column says so.

Usage
-----
    python3 tools/lit_ledger.py            # rewrite docs/lit_ledger.md
    python3 tools/lit_ledger.py --stdout   # print it instead
    python3 tools/lit_ledger.py --check    # tier 0 (scripts/check.sh)

Re-run it in the commit that adds or changes a scan file (CLAUDE.md "Literature
scans"). `--check` compares only the strongest and scans cells of ids a scan
file cites: they need no git history, and a citation added elsewhere does not
stale them. First cited and "elsewhere" are not checked.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = "docs/lit_ledger.md"
#: Not read for citations: the ledger, and this file (its comments quote ids).
NOT_READ = frozenset({LEDGER, "tools/lit_ledger.py"})
INDEX = "docs/LITERATURE.md"
INDEX_SECTION = "## 1. The files"
READINGS = "docs/readings/"

#: A scan file: a phase's `lit-N.md`, Phase 8's `literature-8.md`, or one of
#: the two pre-index scans in `archive/docs/`.
SCAN_RE = re.compile(r"(?:^|/)(?:lit-[^/]+|literature-[^/]+|literature_scan_[^/]+)\.md$")

SUFFIXES = (".md", ".py", ".tex", ".bib")
SKIP_TOP = frozenset({"data", "tests", ".git", ".venv", "venv"})

STRENGTH = {"N": 0, "S": 1, "A": 2, "H": 3, "R": 4}
#: Marks a scan file defines for itself, mapped onto the shared scale.
#: A file's [P], [M], [K] ... not listed here are not on the scale and are not
#: read (lit-1d's [P] is a pointer to an earlier reading, its [M] "measured").
LOCAL_MARKS: dict[str, dict[str, str | None]] = {
    # "[T] the LaTeX source read at the lines named; [P] the PDF text or a
    # figure read by eye" (its "Marks." line): both full text. Its [R] is "a
    # Hugging Face repository queried", not a paper read: dropped.
    "trackC_writeup/lit-C2p-fulltext.md": {"T": "R", "P": "R", "R": None},
    # "[F] full text read this session".
    "trackC_writeup/lit-C7.md": {"F": "R"},
    # "[P] primary text read (abstract page at least, as here)": the floor.
    "p1e_energy_field/lit-1e.md": {"P": "A"},
}

_ID = re.compile(r"(?<![\w.])(\d{2})(\d{2})\.(\d{4,5})(?:v\d+)?(?![\w])")
_MARK = re.compile(r"\[([A-Z])\]")
#: A mark starting within this many characters of its id's start, in the same
#: table cell, is that id's own (`**2509.23024** **[S]**` is 15, `2402.04362 [A]` 11).
ADJACENT = 16
_NEGATED = re.compile(r"\bnot[\s*]*$", re.I)


# ---------------------------------------------------------------------------
# Scan files and index rows (shared with tools/lint_repo.py)
# ---------------------------------------------------------------------------

def is_scan_file(rel: str) -> bool:
    return bool(SCAN_RE.search(rel))


def index_rows(root: Path) -> set[str]:
    """Backticked paths in the table rows of `docs/LITERATURE.md` §1."""
    path = root / INDEX
    if not path.is_file():
        return set()
    out, inside = set(), False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.strip().startswith(INDEX_SECTION)
            continue
        if inside and line.startswith("|"):
            out.update(re.findall(r"`([^`]+\.md)`", line))
    return out


def missing_index_rows(root: Path, rels: list[str]) -> list[str]:
    """Scan files among `rels` that no §1 row names by repo-relative path."""
    rows = index_rows(root)
    return sorted(r for r in rels if is_scan_file(r) and r not in rows)


# ---------------------------------------------------------------------------
# Reading ids and marks
# ---------------------------------------------------------------------------

def arxiv_ids(line: str) -> list[tuple[int, str]]:
    """(position, id) for each plausible new-style arXiv id on the line."""
    out = []
    for m in _ID.finditer(line):
        yy, mm, num = int(m.group(1)), int(m.group(2)), m.group(3)
        if not (7 <= yy <= 30 and 1 <= mm <= 12):
            continue
        if (len(num) == 5) != (yy * 100 + mm >= 1501):
            continue
        out.append((m.start(), f"{m.group(1)}{m.group(2)}.{num}"))
    return out


def line_marks(line: str, rel: str) -> list[tuple[int, str]]:
    local = LOCAL_MARKS.get(rel, {})
    out = []
    for m in _MARK.finditer(line):
        if _NEGATED.search(line[:m.start()]):
            continue                             # "not [R]"
        mark = local.get(m.group(1), m.group(1))
        if mark in STRENGTH:
            out.append((m.start(), mark))
    return out


def assign_marks(ids: list[tuple[int, str]], marks: list[tuple[int, str]],
                 default: str | None = None, row: bool = True,
                 line_: str = "") -> dict[str, str]:
    """Each id's mark on one line (module docstring, "Marks"). `default` is a
    table header's mark, used by a row with no bracketed mark at all. Off a
    table row (`row=False`) only an adjacent mark counts; a mark in another
    cell of `line_` is never adjacent. An id in a cell where another id has
    its own adjacent mark inherits nothing ("2604.01978 [T] ... Fedorov et al.
    2601.21942 (cited there)": the mark is the first paper's)."""
    out: dict[str, str] = {}
    distinct = {m for _, m in marks} if row else set()
    weakest = min(distinct, key=STRENGTH.__getitem__) if distinct else None
    own: list[str | None] = []
    for i, (pos, pid) in enumerate(ids):
        end = ids[i + 1][0] if i + 1 < len(ids) else 10 ** 9
        after = [m for p, m in marks if pos < p < min(end, pos + ADJACENT)
                 and "|" not in line_[pos:p]]
        own.append(after[0] if after else None)
    cell = [line_.count("|", 0, pos) for pos, _ in ids]
    for i, (pos, pid) in enumerate(ids):
        if own[i]:
            mark = own[i]
        elif any(own[j] and cell[j] == cell[i] for j in range(len(ids)) if j != i):
            mark = None                          # a sibling's mark, not this id's
        elif distinct:                           # one mark, or whose is whose unclear
            mark = weakest
        else:
            mark = default
        if mark and (pid not in out or STRENGTH[mark] > STRENGTH[out[pid]]):
            out[pid] = mark
    return out


def marks_apply(rel: str) -> bool:
    return is_scan_file(rel) or rel == INDEX or rel.startswith(READINGS)


# ---------------------------------------------------------------------------
# The tree and its history
# ---------------------------------------------------------------------------

def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True,
                              check=True).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.CalledProcessError):
        return None


def tree_files(root: Path) -> list[str]:
    out = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    if out is not None and (root / ".git").exists():
        rels = [r for r in out.split("\0") if r]
    else:
        rels = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    return sorted(r for r in rels
                  if r.endswith(SUFFIXES) and r not in NOT_READ
                  and r.split("/", 1)[0] not in SKIP_TOP and (root / r).is_file())


def first_cited(root: Path) -> tuple[dict[str, tuple[str, str, str]], str]:
    """id -> (date, commit, path) of the earliest added line holding it, and a
    note on the history read ("" when it is complete)."""
    if not (root / ".git").exists():
        return {}, "no git history"
    note = ""
    if (_git(root, "rev-parse", "--is-shallow-repository") or "").strip() == "true":
        note = "shallow clone: dates are not the first citation"
    log = _git(root, "log", "--reverse", "-p", "--no-renames", "--no-merges", "-U0",
               "--format=@@C %h %as", "--", *[f"*{s}" for s in SUFFIXES])
    if log is None:
        return {}, "git log failed"
    first: dict[str, tuple[str, str, str]] = {}
    commit = date = path = ""
    for line in log.splitlines():
        if line.startswith("@@C "):
            _, commit, date = line.split(" ", 2)
        elif line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else ""
        elif line.startswith("+") and path and path not in NOT_READ \
                and path.split("/", 1)[0] not in SKIP_TOP:
            for _, pid in arxiv_ids(line):
                if pid not in first or date < first[pid][0]:
                    first[pid] = (date, commit, path)
    return first, note


@dataclass
class Paper:
    pid: str
    scans: set[str] = field(default_factory=set)
    files: set[str] = field(default_factory=set)
    mark: str | None = None
    mark_at: set[str] = field(default_factory=set)

    def give(self, mark: str, rel: str) -> None:
        if self.mark is None or STRENGTH[mark] > STRENGTH[self.mark]:
            self.mark, self.mark_at = mark, {rel}
        elif mark == self.mark:
            self.mark_at.add(rel)


def collect(root: Path) -> dict[str, Paper]:
    papers: dict[str, Paper] = {}
    for rel in tree_files(root):
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scan, marked = is_scan_file(rel), marks_apply(rel)
        header, prev_row = None, False
        for line in text.splitlines():
            row = line.startswith("|")
            if row and not prev_row:             # a table's header row
                hm = {m for _, m in line_marks(line, rel)}
                header = hm.pop() if len(hm) == 1 else None
            elif not row:
                header = None
            prev_row = row
            ids = arxiv_ids(line)
            if not ids:
                continue
            for _, pid in ids:
                p = papers.setdefault(pid, Paper(pid))
                (p.scans if scan else p.files).add(rel)
            if marked:
                own = _MARK.search(line) is not None
                found = assign_marks(ids, line_marks(line, rel),
                                     None if own or not row else header, row, line)
                for pid, mark in found.items():
                    papers[pid].give(mark, rel)
    for rel in tree_files(root):
        m = re.fullmatch(r"docs/readings/(\d{4}\.\d{4,5})\.md", rel)
        if m and m.group(1) in papers:
            papers[m.group(1)].give("R", rel)
    return papers


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _short(rels: set[str], unique: set[str]) -> str:
    """A scan file by its name where that is unique; archive/ paths in full,
    since archive/MOVED.md maps their bare names to the old locations."""
    return ", ".join(f"`{Path(r).name if Path(r).name in unique and not r.startswith('archive/') else r}`"
                     for r in sorted(rels))


def render(root: Path) -> str:
    papers = collect(root)
    first, note = first_cited(root)
    head = (_git(root, "rev-parse", "--short", "HEAD") or "").strip() or "no git"
    unique = _unique_scan_names(root)

    counts = defaultdict(int)
    for p in papers.values():
        counts[p.mark or "-"] += 1
    in_scans = sum(1 for p in papers.values() if p.scans)
    unmarked_in_scans = sum(1 for p in papers.values() if p.scans and not p.mark)

    out = [
        f"<!-- {LEDGER}: GENERATED by tools/lit_ledger.py; do not edit by hand. -->",
        "# Paper ledger — every arXiv id the repo cites",
        "",
        f"**Built from** `HEAD` {head} and the working tree, by "
        "`python3 tools/lit_ledger.py`; re-run it in the commit that adds or changes a "
        "scan file. What is read and what is missed: the tool's docstring.",
        "",
        "Marks (`docs/LITERATURE.md` §0): [N] title only < [S] search summary < [A] "
        "abstract < [H] HTML full text through a summarising fetch < [R] primary text "
        "read. **Strongest** is the best mark any scan file, `docs/LITERATURE.md` or "
        "`docs/readings/` gives the id, with where. Rows at [S] or below that a draft "
        "cites are the reread candidates. Verdicts live in the scan files, not here.",
        "",
        "| | count |",
        "|---|---|",
        f"| ids cited | {len(papers)} |",
        f"| cited in a scan file | {in_scans} |",
        *[f"| strongest [{m}] | {counts[m]} |" for m in "RHASN"],
        f"| no mark read (in a scan file) | {unmarked_in_scans} |",
        f"| no mark read (cited outside scans only) | {counts['-'] - unmarked_in_scans} |",
        "",
    ]
    if note:
        out += [f"**First cited is incomplete:** {note}.", ""]
    out += [
        "| arXiv | first cited | strongest | scans | elsewhere |",
        "|---|---|---|---|---|",
    ]
    cols = scan_columns(papers, unique)
    for pid in sorted(papers):
        p = papers[pid]
        if pid in first:
            d, c, f = first[pid]
            fc = f"{d} `{c}` " + (_short({f}, unique) if (root / f).is_file()
                                  else "(a file since moved or deleted)")
        else:
            fc = "uncommitted" if not note else "—"
        mark, scans = cols[pid]
        out.append(f"| {pid} | {fc} | {mark} | {scans} | "
                   f"{_short(p.files, unique) or '—'} |")
    return "\n".join(out) + "\n"


def _unique_scan_names(root: Path) -> set[str]:
    names = [Path(r).name for r in tree_files(root) if is_scan_file(r)]
    return {n for n in names if names.count(n) == 1}


def scan_columns(papers: dict[str, Paper], unique: set[str]) -> dict[str, tuple[str, str]]:
    """id -> (strongest, scans) cells: the part of a row that only scan files,
    `docs/LITERATURE.md` and `docs/readings/` decide."""
    return {pid: (f"[{p.mark}] {_short(p.mark_at, unique)}" if p.mark else "—",
                  _short(p.scans, unique) or "—") for pid, p in papers.items()}


def check(root: Path) -> list[str]:
    """Rows of the committed ledger whose strongest or scans cell differs from
    the tree, for ids a scan file cites. Needs no git history, and a citation
    added outside the scan files does not trip it."""
    want = {pid: c for pid, c in scan_columns(collect(root), _unique_scan_names(root)).items()
            if c[1] != "—"}
    path = root / LEDGER
    have = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.split("|")]
            if len(cells) == 7 and re.fullmatch(r"\d{4}\.\d{4,5}", cells[1]) and cells[4] != "—":
                have[cells[1]] = (cells[3], cells[4])
    return [f"{pid}: ledger {have.get(pid)} != tree {want.get(pid)}"
            for pid in sorted(set(want) | set(have)) if want.get(pid) != have.get(pid)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    ap.add_argument("--check", action="store_true",
                    help="fail if the ledger's strongest/scans cells differ from the tree")
    args = ap.parse_args(argv)
    if args.check:
        diffs = check(ROOT)
        for d in diffs:
            print(d)
        if diffs:
            print(f"{LEDGER} is stale on {len(diffs)} scan-cited id(s): "
                  f"run python3 tools/lit_ledger.py")
            return 1
        print(f"{LEDGER} in step with the scan files")
        return 0
    text = render(ROOT)
    if args.stdout:
        sys.stdout.write(text)
    else:
        (ROOT / LEDGER).write_text(text, encoding="utf-8")
        rows = sum(1 for l in text.splitlines() if re.match(r"\| \d{4}\.", l))
        print(f"wrote {LEDGER}: {rows} ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
