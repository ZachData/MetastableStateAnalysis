#!/usr/bin/env python3
"""Rewrite citations of moved documents in live files, from ``archive/MOVED.md``.

``archive/MOVED.md``'s **Moved** table is the single record of what moved
where; this script applies it, so an archive batch updates every live
citation mechanically instead of by hand (and ``tools/lint_repo.py``'s
``cited-md-path`` rule catches what it misses).

Two kinds of row:

* a path (``UPDATE_PLAN.md`` -> ``archive/UPDATE_PLAN.md``): the repo-relative
  path is rewritten wherever it appears, and so is the bare file name when it
  is unique among moved rows and no live file shares it (``CHANGES-1b.md`` in
  ``p1b_hemisphere/design-1b.md``);
* a section (``POPPER_PLAN.md §B3`` -> ``archive/POPPER_PLAN-done.md §B3``):
  ``POPPER_PLAN.md §B3``, ``POPPER_PLAN §B3`` and ``POPPER_PLAN.md item B3``
  are rewritten, backticks kept. ``§1`` never matches ``§1.5`` or ``§10``.

Not touched: ``archive/`` (frozen), ``data/``, ``claims/**/*.json`` (the
registry, audit and calibration records are never edited after the fact),
the files the ``claims`` renderers generate (regenerate those instead), and
every source file a ``claims/`` record pins by hash (a ``*_file`` key such as
``gate_file``): a docstring edit there breaks the record's sha256 test, and the
record may not be edited. Their old citations stay valid through the Moved
table, which the lint accepts.
Idempotent: a rewritten path is preceded by ``archive/`` and no longer
matches.

    python tools/rewrite_moved_refs.py           # dry run: per-file counts
    python tools/rewrite_moved_refs.py --write   # apply
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOVED = "archive/MOVED.md"

#: Written by ``tools/render_*.py`` from the registry; editing them by hand
#: fails ``--check``. Rewrite the renderer and regenerate.
GENERATED = frozenset({
    "claims/EVALUABILITY.md", "claims/FALSIFICATION.md", "claims/EXPERIMENTS.md",
})
TEXT_SUFFIXES = (".md", ".py", ".sh", ".yml", ".yaml", ".toml", ".ini", ".txt")

_PINNED_KEY = re.compile(r'"[a-z0-9_]*_file"\s*:\s*"([^"]+)"')


def pinned_files(root: Path = ROOT) -> set[str]:
    """Source files a ``claims/`` JSON record names under a ``*_file`` key."""
    out = set()
    for rec in (root / "claims").rglob("*.json"):
        out.update(_PINNED_KEY.findall(rec.read_text(encoding="utf-8")))
    return out


_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|")


def moved_table(root: Path = ROOT) -> list[tuple[str, str]]:
    """``(old, new)`` rows of the Moved table, in file order."""
    rows, inside = [], False
    for line in (root / MOVED).read_text().splitlines():
        if line.startswith("## "):
            inside = line.strip() == "## Moved"
            continue
        m = _ROW.match(line) if inside else None
        if m:
            rows.append((m.group(1), m.group(2)))
    return rows


def absent_table(root: Path = ROOT) -> set[str]:
    """Cited paths the Absent table records as never in the tree."""
    out, inside = set(), False
    for line in (root / MOVED).read_text().splitlines():
        if line.startswith("## "):
            inside = line.strip() == "## Absent"
            continue
        m = re.match(r"^\|\s*`([^`]+)`\s*\|", line) if inside else None
        if m:
            out.add(m.group(1))
    return out


def live_files(root: Path = ROOT) -> list[str]:
    """Tracked text files a rewrite may touch."""
    tracked = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                             text=True, check=True).stdout.split("\n")
    pinned = pinned_files(root)
    return [f for f in tracked if f and f.endswith(TEXT_SUFFIXES)
            and not f.startswith(("archive/", "data/"))
            and not (f.startswith("claims/") and f.endswith(".json"))
            and f not in GENERATED and f not in pinned]


def _path_sub(pattern: str, new: str):
    rx = re.compile(r"(?<![\w./-])(?:\./)?" + re.escape(pattern) + r"(?![\w-])")
    return rx, new, Path(pattern).name


def build_rules(rows, live: list[str]):
    """Compile the Moved rows into ``(regex, replacement, needle)`` triples;
    a rule runs only on text containing its needle (a plain substring test,
    which keeps a whole-tree pass fast enough for the pure tier)."""
    rules = []
    live_names = Counter(Path(f).name for f in live)
    path_rows = [(o, n) for o, n in rows if "§" not in o]
    moved_names = Counter(Path(o).name for o, _ in path_rows)
    for old, new in path_rows:
        rules.append(_path_sub(old, new))
        name = Path(old).name
        if name != old and moved_names[name] == 1 and not live_names[name]:
            rules.append(_path_sub(name, new))
    for old, new in rows:
        if "§" not in old:
            continue
        ofile, osec = (s.strip() for s in old.split("§"))
        nfile, _, nsec = (s.strip() for s in new.partition("§"))
        stem = re.escape(ofile[:-3] if ofile.endswith(".md") else ofile)
        rx = re.compile(r"(?<![\w./-])(`?)" + stem + r"(?:\.md)?\1(\s*)(§\s*|item\s+)"
                        + re.escape(osec) + r"(?![\w]|\.\d)")

        def repl(m, nfile=nfile, nsec=nsec):
            tail = f"{m.group(2)}{m.group(3)}{nsec}" if nsec else ""
            return f"{m.group(1)}{nfile}{m.group(1)}{tail}"
        rules.append((rx, repl, ofile[:-3] if ofile.endswith(".md") else ofile))
    return rules


def rewrite(text: str, rules) -> tuple[str, int]:
    total = 0
    for rx, new, needle in rules:
        if needle not in text:
            continue
        text, n = rx.subn(new, text)
        total += n
    return text, total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="apply (default: dry run)")
    args = ap.parse_args(argv)
    live = live_files()
    rules = build_rules(moved_table(), live)
    grand = 0
    for f in live:
        path = ROOT / f
        try:
            old = path.read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        new, n = rewrite(old, rules)
        if n:
            grand += n
            print(f"{n:4d}  {f}")
            if args.write:
                path.write_text(new)
    print(f"{grand} citation(s) {'rewritten' if args.write else 'would be rewritten'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
