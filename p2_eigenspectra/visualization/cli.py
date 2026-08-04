"""
p2_eigenspectra/visualization/cli.py

Entry point:

    python -m p2_eigenspectra.visualization \\
        --p2_dir results/p2_eigenspectra_2026-08-03_12-00-00 \\
        --out blog_figures/p2 \\
        [--classes scalars spectra clouds] \\
        [--metrics rep_frac non_normality] \\
        [--cloud_layers 0 12 23] [--filmstrip_k 6] \\
        [--prompts wiki_paragraph short_heterogeneous]

Takes a Phase 2 OUTPUT directory, not a Phase 1 one. The two are not
interchangeable: Phase 1 run dirs hold geometry.json per (model, prompt);
Phase 2's hold ov_summary_*.json per model plus per-run stem directories.
Pointing this at a Phase 1 directory finds no summaries and says so.
"""

import argparse
import sys
from pathlib import Path

from .pipeline import generate_p2_figures, CLASSES
from .spectra import WEIGHT_METRICS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Phase 2 checkpoint figures (eigenspectrum "
                    "scalars / sweeps+heatmaps / eigenvalue clouds) for every "
                    "'-step{N}' family in a Phase 2 output directory.",
    )
    parser.add_argument("--p2_dir", type=Path, required=True,
                        help="A p2_eigenspectra_<timestamp>/ directory.")
    parser.add_argument("--out", type=Path, default=Path("blog_figures/p2"))
    parser.add_argument("--classes", nargs="*", default=None, choices=list(CLASSES),
                        help="Limit to a subset of figure classes (default: all). "
                             "'clouds' alone reuses an existing "
                             "transitions_p2_*.json.")
    parser.add_argument("--metrics", nargs="*", default=None,
                        choices=list(WEIGHT_METRICS.keys()),
                        help="Limit the weight-side sweeps/heatmaps to these "
                             "metrics (default: all).")
    parser.add_argument("--cloud_layers", nargs="*", type=int, default=None,
                        help="Layers to draw eigenvalue clouds for "
                             "(default: most-changed layer plus first/mid/last).")
    parser.add_argument("--filmstrip_k", type=int, default=6,
                        help="Max checkpoints per cloud filmstrip (default 6).")
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="Restrict the verdict-side spread to these prompts "
                             "(default: every prompt found).")
    args = parser.parse_args()

    if not args.p2_dir.exists():
        print(f"ERROR: p2_dir not found: {args.p2_dir}", file=sys.stderr)
        sys.exit(1)

    generate_p2_figures(
        args.p2_dir, args.out,
        classes=args.classes,
        metrics=args.metrics,
        cloud_layers=args.cloud_layers,
        filmstrip_k=args.filmstrip_k,
        prompts=args.prompts,
    )
    print(f"\nFigures written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
