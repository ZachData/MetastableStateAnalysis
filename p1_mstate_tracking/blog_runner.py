"""
p1_mstate_tracking/blog_runner.py
CLI to generate all Phase 1 blog figures from saved results.

Usage:
    python -m p1_mstate_tracking.blog_runner \
        --results_dir results/phase1 \
        --out         blog_figures \
        [--groups A B C D E]

Discovers runs once from results_dir, then calls the requested group functions.
Default: all groups.
"""

import argparse
import sys
from pathlib import Path

# Relative import when called as a module; fall back to direct import for
# running as a standalone script during development.
try:
    from .plots_blog import (
        discover_runs,
        generate_group_A,
        generate_group_B,
        generate_group_C,
        generate_group_D,
        generate_group_E,
    )
except ImportError:
    from plots_blog import (
        discover_runs,
        generate_group_A,
        generate_group_B,
        generate_group_C,
        generate_group_D,
        generate_group_E,
    )

GROUP_FUNCS = {
    "A": generate_group_A,
    "B": generate_group_B,
    "C": generate_group_C,
    "D": generate_group_D,
    "E": generate_group_E,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 1 blog figures from saved results.",
    )
    parser.add_argument(
        "--results_dir", type=Path, default=Path("results/phase1"),
        help="Directory containing per-run subdirs (default: results/phase1)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("blog_figures"),
        help="Output directory for figures (default: blog_figures)",
    )
    parser.add_argument(
        "--groups", nargs="*", default=list(GROUP_FUNCS.keys()),
        choices=list(GROUP_FUNCS.keys()),
        help="Which groups to generate (default: all)",
    )
    parser.add_argument(
        "--list_runs", action="store_true",
        help="Print discovered runs and exit without generating figures.",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"ERROR: results_dir not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Discovering runs in: {args.results_dir}")
    runs = discover_runs(args.results_dir)

    if not runs:
        print("No runs found. Check that geometry.json exists in subdirs.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} runs:")
    for (model, prompt) in sorted(runs.keys()):
        print(f"  {model:<40} {prompt}")
    print()

    if args.list_runs:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out}\n")

    for g in args.groups:
        GROUP_FUNCS[g](runs, args.out)

    print(f"Done. Figures written to: {args.out}")


if __name__ == "__main__":
    main()
