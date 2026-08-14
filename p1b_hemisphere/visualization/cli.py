"""
p1b_hemisphere/visualization/cli.py

The single command-line entry point:

    python -m p1b_hemisphere.visualization \\
        --p1b_dir results/<phase1b-output-dir> \\
        --out     blog_figures/p1b \\
        [--classes regime cone tracking membership axis curiosities
                   crossrun checkpoints] \\
        [--models gpt2-large] [--prompts wiki_paragraph] \\
        [--list_runs] [--fixture]

Takes a Phase 1b OUTPUT directory — the one holding `phase1b_*.json` — not a
Phase 1 run directory. The two are not interchangeable and pointing this at
the wrong one finds no runs and says so.

`--list_runs` prints what was discovered and what each run is missing,
without drawing anything. That is the fastest way to find out which of the
four data gaps in FIGURES-1b.md is biting a particular directory, and it is
worth running first against any directory older than the emission changes.

`--fixture` builds a synthetic Phase 1b directory and draws the whole
catalogue against it. The numbers are invented and no result should ever be
read off them; the shapes are the ones a real run writes.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from .loaders import describe_runs, discover_runs
from .pipeline import CLASSES, generate_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate every Phase 1b visualization from saved "
                    "artifacts. No model load, no torch.",
    )
    parser.add_argument(
        "--p1b_dir", type=Path, default=None,
        help="Directory holding phase1b_*.json (a Phase 1b output directory).")
    parser.add_argument(
        "--out", type=Path, default=Path("blog_figures/p1b"),
        help="Output directory (default: blog_figures/p1b). Per-run figures "
             "go in {out}/{stem}/, cross-run figures in {out}/_cross/.")
    parser.add_argument(
        "--classes", nargs="*", default=None, choices=list(CLASSES),
        help="Limit to a subset of figure classes (default: all). No class "
             "consumes another's output, so any subset is valid.")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Restrict to these model names (default: every model found).")
    parser.add_argument(
        "--prompts", nargs="*", default=None,
        help="Restrict to these prompt keys (default: every prompt found).")
    parser.add_argument(
        "--list_runs", action="store_true",
        help="Print discovered runs and what each is missing, then exit.")
    parser.add_argument(
        "--fixture", action="store_true",
        help="Draw the catalogue against a synthetic Phase 1b directory "
             "instead of a real one. Invented numbers, real shapes.")
    parser.add_argument(
        "--fixture_dir", type=Path, default=None,
        help="Where to build the fixture (default: a temporary directory). "
             "Useful for inspecting what the fixture actually writes.")
    args = parser.parse_args()

    p1b_dir = args.p1b_dir
    if args.fixture:
        from ._fixture import build_fixture
        p1b_dir = args.fixture_dir or Path(tempfile.mkdtemp(prefix="p1b_fixture_"))
        print(f"Building synthetic Phase 1b directory: {p1b_dir}")
        build_fixture(p1b_dir)
        print("  ⚠  fixture numbers are invented — shapes only, no results\n")
    elif p1b_dir is None:
        parser.error("one of --p1b_dir or --fixture is required")
    elif not p1b_dir.exists():
        print(f"ERROR: p1b_dir not found: {p1b_dir}", file=sys.stderr)
        sys.exit(1)

    if args.list_runs:
        runs = discover_runs(p1b_dir, models=args.models, prompts=args.prompts)
        print(f"{len(runs)} run(s) in {p1b_dir}:")
        print(describe_runs(runs))
        return

    generate_all(p1b_dir, args.out, classes=args.classes,
                 models=args.models, prompts=args.prompts)
    print(f"Figures written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
