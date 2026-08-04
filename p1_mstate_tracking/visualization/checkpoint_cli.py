"""
visualization/checkpoint_cli.py

Standalone entry point for the checkpoint figure classes only — useful
when the sweep results already exist and only these figures need
regenerating (e.g. re-picking filmstrip snapshots after inspecting
transitions.json), without re-running the whole per-model circuit:

    python -m p1_mstate_tracking.visualization.checkpoint_cli \\
        --results_dir results/pythia_410m_pilot \\
        --random_seed_dirs results/pythia_random \\
        --out blog_figures \\
        [--prompt wiki_paragraph] [--all_prompts] \\
        [--classes scalars heatmaps] \\
        [--filmstrip_k 6] [--filmstrip_layer 20]

The main `python -m p1_mstate_tracking.visualization` entry point also
runs everything here (pipeline.generate_all calls
generate_checkpoint_figures) — this CLI exists for the second pass of
the two-pass workflow, not as a required extra step.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

from core.style import DEFAULT_PROMPT
from .loaders import discover_runs
from .pipeline import _discover_random_dir
from .random_aggregate import build_aggregate
from .checkpoint_pipeline import generate_checkpoint_figures


def main():
    parser = argparse.ArgumentParser(
        description="Generate checkpoint-sweep figures (scalars / heatmaps / "
                    "sweeps / filmstrips) for every '-step{N}' family found.",
    )
    parser.add_argument("--results_dir", type=Path, default=Path("results/phase1"))
    parser.add_argument(
        "--random_seed_dirs", "--random_dirs", dest="random_dirs",
        nargs="*", type=Path, default=[],
        help="Multi-seed random-control run dirs (same semantics as the main CLI).",
    )
    parser.add_argument("--out", type=Path, default=Path("blog_figures"))
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--all_prompts", action="store_true")
    parser.add_argument(
        "--classes", nargs="*", default=None,
        choices=["scalars", "heatmaps", "sweeps", "filmstrips"],
        help="Limit to a subset of figure classes (default: all four). "
             "Running 'filmstrips' alone reuses an existing transitions.json.",
    )
    parser.add_argument("--filmstrip_k", type=int, default=6,
                        help="Max snapshot checkpoints per filmstrip (default 6).")
    parser.add_argument("--filmstrip_layer", type=int, default=None,
                        help="Fixed layer for filmstrips (default: deepest cached).")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"ERROR: results_dir not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    runs = discover_runs(args.results_dir)
    print(f"Discovered {len(runs)} runs in {args.results_dir}")

    random_agg: Dict[Tuple[str, str], dict] = {}
    for rd in args.random_dirs:
        if not rd.exists():
            print(f"WARNING: random_dir not found, skipping: {rd}", file=sys.stderr)
            continue
        extra = _discover_random_dir(rd)
        runs.update(extra)
        agg = build_aggregate(rd)
        random_agg.update(agg)
        print(f"  + {len(extra)} runs, {len(agg)} aggregate group(s) from {rd}")

    prompts = (sorted({p for (_, p) in runs.keys()})
               if args.all_prompts else [args.prompt])

    for prompt in prompts:
        generate_checkpoint_figures(
            runs, args.out, prompt,
            random_agg=random_agg or None,
            filmstrip_k=args.filmstrip_k,
            filmstrip_layer=args.filmstrip_layer,
            classes=args.classes,
        )


if __name__ == "__main__":
    main()
