"""
p2b_imaginary/visualization/cli.py

The single command-line entry point:

    python -m p2b_imaginary.visualization \\
        --p2b_dir results/<phase2b-output-dir> \\
        --out     blog_figures/p2b \\
        [--classes spectrum heads frames trajectory report verdicts
                   nulls curiosities] \\
        [--steps 512 3000] [--prompts wiki_paragraph] \\
        [--external phase2_frac_repulsive.json] \\
        [--list_runs] [--fixture]

Takes a Phase 2b OUTPUT directory — the one holding `phase2b_results.json`
and one subdirectory per checkpoint — not a Phase 2 weights directory and not
a Phase 1 run root. Pointing it at any of the others finds no results and
says so.

A PRE-REWRITE directory raises rather than drawing. That check is
`p2b_io.refuse_legacy_run_dir`, not a copy of it: those runs were scored with
an absolute 1e-6 threshold and a 3.0 rank gate, and their `elim_rotation`
column is an algebraic identity, so a figure of them would be a figure of two
incompatible counting rules with nothing on it saying which.

`--list_runs` prints what was discovered, what each checkpoint is missing,
and which data gaps are open in this particular directory, without drawing
anything. Run it first against any directory whose provenance is not obvious
— it is the fastest way to find out that a sweep was `--blocks 1a` before
wondering where the `frames` figures went.

`--fixture` builds a synthetic Phase 2b directory and draws the whole
catalogue against it. The weights are random and no result should ever be
read off the numbers; every key, verdict and refusal status is real, because
the phase's own `run_block_1a` and `run_block_1b` produce them.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .loaders import describe_sweep, load_sweep
from .pipeline import CLASSES, generate_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate every Phase 2b visualization from saved "
                    "artifacts. No model load, no weights, no torch.",
    )
    parser.add_argument(
        "--p2b_dir", type=Path, default=None,
        help="Phase 2b output directory (holds phase2b_results.json).")
    parser.add_argument(
        "--out", type=Path, default=Path("blog_figures/p2b"),
        help="Output directory (default: blog_figures/p2b). Per-checkpoint "
             "figures go in {out}/{stem}/, Block 1b figures in "
             "{out}/{stem}/{prompt}/, cross-checkpoint figures in "
             "{out}/_cross/.")
    parser.add_argument(
        "--classes", nargs="*", default=None, choices=list(CLASSES),
        help="Limit to a subset of figure classes (default: all). No class "
             "consumes another's output, so any subset is valid.")
    parser.add_argument(
        "--steps", nargs="*", type=int, default=None,
        help="Restrict to these checkpoint steps (default: every step found). "
             "Applies to the cross-checkpoint figures too, so a filtered run "
             "does not draw a trajectory over checkpoints it excluded.")
    parser.add_argument(
        "--prompts", nargs="*", default=None,
        help="Restrict Block 1b to these prompt keys (default: all found).")
    parser.add_argument(
        "--external", type=Path, default=None,
        help="JSON file {name, steps, values} holding a series from another "
             "phase — Phase 2's `frac_repulsive`, most usefully. Drawn "
             "against this phase's Henrici trajectory in the report class's "
             "co_movement figure, through p2b_report.external_trajectory so "
             "it carries NaN spread rather than borrowing Phase 2b's scale.")
    parser.add_argument(
        "--list_runs", action="store_true",
        help="Print the discovered sweep, what each checkpoint is missing, "
             "and which data gaps are open here, then exit.")
    parser.add_argument(
        "--fixture", action="store_true",
        help="Draw the catalogue against a synthetic Phase 2b directory "
             "instead of a real one. Invented weights, real shapes.")
    parser.add_argument(
        "--fixture_dir", type=Path, default=None,
        help="Where to build the fixture (default: a temporary directory). "
             "Useful for inspecting what the fixture actually writes.")
    args = parser.parse_args()

    p2b_dir = args.p2b_dir
    if args.fixture:
        from ._fixture import build_fixture
        p2b_dir = args.fixture_dir or Path(
            tempfile.mkdtemp(prefix="p2b_fixture_"))
        print(f"Building synthetic Phase 2b directory: {p2b_dir}")
        build_fixture(p2b_dir)
        print("  ⚠  fixture weights are random — shapes only, no results\n")
    elif p2b_dir is None:
        parser.error("one of --p2b_dir or --fixture is required")
    elif not p2b_dir.exists():
        print(f"ERROR: p2b_dir not found: {p2b_dir}", file=sys.stderr)
        sys.exit(1)

    if args.list_runs:
        sweep = load_sweep(p2b_dir, steps=args.steps, prompts=args.prompts)
        if sweep is None:
            sys.exit(1)
        print(f"Phase 2b sweep at {p2b_dir}:")
        print(describe_sweep(sweep))
        return

    external = None
    if args.external is not None:
        with open(args.external) as f:
            external = json.load(f)
        missing = [k for k in ("steps", "values") if k not in external]
        if missing:
            parser.error(f"--external file is missing {missing}; expected "
                         "{name, steps, values}")

    generate_all(p2b_dir, args.out, classes=args.classes, steps=args.steps,
                 prompts=args.prompts, external=external)
    print(f"Figures written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
