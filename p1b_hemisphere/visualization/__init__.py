"""
p1b_hemisphere/visualization — Phase 1b figures.

Loads exclusively from a Phase 1b output directory (`phase1b_*.json`,
`phase1b_*_particles.npz`, `phase1b_*_axes.npz`, `phase1b_cross_run.json`).
No model weights, no torch, no recomputation. Run as a single command:

    python -m p1b_hemisphere.visualization \\
        --p1b_dir results/<phase1b-output-dir> \\
        --out     blog_figures/p1b \\
        [--classes regime cone tracking membership axis curiosities
                   crossrun checkpoints] \\
        [--models …] [--prompts …] [--list_runs] [--fixture]

or imported:

    from pathlib import Path
    from p1b_hemisphere.visualization import discover_runs, generate_all
    generate_all(Path("results/phase1b"), Path("blog_figures/p1b"))

Eight figure classes. See FIGURES-1b.md in this directory for the full
catalogue — every figure, what it shows, which artifact it reads, and its
status — plus the four data gaps where a quantity is computed by a block and
dropped before writing.

    style.py           palette and drawing primitives
    loaders.py         every disk read; optional inputs reported, never assumed
    _fixture.py        synthetic Phase 1b directory (test aid)
    regime.py          Block 0 — bipartition quality, both classifiers
    cone.py            Block 3 — containment as a null-referenced quantity
    tracking.py        Block 1 — identity, rotation, events, Phase 1 crossref
    membership.py      Block 2 — per-token, HDBSCAN nesting, boundary vs noise
    axis.py            Block A — axis identity against PCA
    curiosities.py     exploratory figures off the particle table
    cross_run.py       model × prompt aggregation and the global verdict
    checkpoints_1b.py  the training-step axis
    pipeline.py        generate_all / class orchestration
    cli.py             argparse entry point (main())

Two conventions worth knowing before reading the code.

**Analysis logic is imported, never copied.** Thresholds come from
`bipartition_detect`, `hemisphere_tracking`, and `axis_identity`; aggregation
and the verdict come from `p1b_report`; checkpoint step-axis conventions come
from `p1_mstate_tracking/visualization/checkpoints.py`. This package decides
how something is drawn and nothing about what it means, so a figure that
disagrees with `phase1b_cross_run.md` is a bug here by construction.

**The continuous quantity is the figure; the label is an annotation.**
status-1b's R1 and R3 are both the same correction — a binary regime label
reported where a null-referenced continuous quantity was available — and the
figures are built so that reading only the labels is hard.
"""

from .loaders import Run, describe_runs, discover_runs, load_cross_run
from .pipeline import CLASSES, CROSS_CLASSES, PER_RUN_CLASSES, generate_all
from .style import BLOG_STYLE, CATEGORICAL, FIEDLER_CMAP, save_figure

__all__ = [
    "Run",
    "discover_runs",
    "describe_runs",
    "load_cross_run",
    "generate_all",
    "CLASSES",
    "PER_RUN_CLASSES",
    "CROSS_CLASSES",
    "BLOG_STYLE",
    "CATEGORICAL",
    "FIEDLER_CMAP",
    "save_figure",
]
