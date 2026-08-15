"""
p1c_frames/visualization/cli.py

The single command-line entry point:

    python -m p1c_frames.visualization \\
        --p1c_dir results_p1c/ \\
        --out     blog_figures/p1c \\
        [--classes integration null moments frames feasibility designs
                   curiosities theory crossrun checkpoints] \\
        [--models pythia-410m-step143000] [--prompts wiki_paragraph] \\
        [--list_runs] [--fixture] [--cheap]

Takes a Phase 1c OUTPUT directory — the one `run_1c --out` wrote, holding
one subdirectory per run with `p1c.json` in it — not a Phase 1 run
directory. The two are not interchangeable and pointing this at the wrong
one finds no runs and says so.

`--classes theory` needs no `--p1c_dir` at all. That class draws the null
model the phase compares against (the γ_β family, the collapse-time table,
Wendel, the Gegenbauer kernels, the effect-size floor) by calling
`p1c_frames`' own functions, so it works before a single Pythia checkpoint
has been analysed — which, per status-1c, is where the phase currently is.

`--list_runs` prints what was discovered, which sub-experiment blocks each
run carries, and what each is missing, without drawing anything. That is the
fastest way to find out which of the gaps in FIGURES-1c.md is biting a
particular directory, and it is worth running first against any directory
whose provenance you do not know.

`--fixture` builds a synthetic Phase 1c output directory and draws the whole
catalogue against it. The numbers are invented and no result should ever be
read off them; the shapes are the ones `save_p1c` actually writes.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from .loaders import describe_runs, discover_runs
from .pipeline import CLASSES, STANDALONE_CLASSES, generate_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate every Phase 1c visualization from saved "
                    "artifacts. No model load, no torch, no re-analysis.",
    )
    parser.add_argument(
        "--p1c_dir", type=Path, default=None,
        help="Phase 1c output directory (the one run_1c --out wrote). Not "
             "required when --classes is theory alone.")
    parser.add_argument(
        "--out", type=Path, default=Path("blog_figures/p1c"),
        help="Output directory (default: blog_figures/p1c). Per-run figures "
             "go in {out}/{stem}/, cross-run in {out}/_cross/, theory in "
             "{out}/_theory/.")
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
        help="Print discovered runs, the blocks each carries, and what each "
             "is missing, then exit.")
    parser.add_argument(
        "--fixture", action="store_true",
        help="Draw the catalogue against a synthetic Phase 1c directory "
             "instead of a real one. Invented numbers, real shapes.")
    parser.add_argument(
        "--fixture_dir", type=Path, default=None,
        help="Where to build the fixture (default: a temporary directory). "
             "Useful for inspecting what the fixture actually writes.")
    parser.add_argument(
        "--cheap", action="store_true",
        help="Reduce the simulation count in the one figure that simulates "
             "(T7, the Q_k random band). The caption says so, so a cheap "
             "figure cannot be mistaken for the phase's own measurement.")
    args = parser.parse_args()

    wanted = set(args.classes) if args.classes else set(CLASSES)
    theory_only = wanted <= set(STANDALONE_CLASSES)

    p1c_dir = args.p1c_dir
    if args.fixture:
        from ._fixture import build_fixture
        p1c_dir = args.fixture_dir or Path(tempfile.mkdtemp(prefix="p1c_fixture_"))
        print(f"Building synthetic Phase 1c directory: {p1c_dir}")
        build_fixture(p1c_dir)
        print("  ⚠  fixture numbers are invented — shapes only, no results\n")
    elif p1c_dir is None:
        if not theory_only:
            parser.error("one of --p1c_dir or --fixture is required "
                         "(except for --classes theory, which needs neither)")
    elif not p1c_dir.exists():
        print(f"ERROR: p1c_dir not found: {p1c_dir}", file=sys.stderr)
        sys.exit(1)

    if args.list_runs:
        if p1c_dir is None:
            parser.error("--list_runs needs --p1c_dir or --fixture")
        runs = discover_runs(p1c_dir, models=args.models, prompts=args.prompts)
        print(f"{len(runs)} run(s) in {p1c_dir}:")
        print(describe_runs(runs))
        return

    generate_all(p1c_dir, args.out, classes=args.classes, models=args.models,
                 prompts=args.prompts, cheap=args.cheap)
    print(f"Figures written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
