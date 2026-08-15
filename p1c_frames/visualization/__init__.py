"""
p1c_frames/visualization — Phase 1c figures.

Loads exclusively from a Phase 1c output directory (`{run}/p1c.json`,
`{run}/p1c_curves.npz`). No model weights, no torch, no Phase 1 artifacts,
no recomputation of a sub-experiment. Run as a single command:

    python -m p1c_frames.visualization \\
        --p1c_dir results_p1c/ \\
        --out     blog_figures/p1c \\
        [--classes integration null moments frames feasibility designs
                   curiosities theory crossrun checkpoints] \\
        [--models …] [--prompts …] [--list_runs] [--fixture] [--cheap]

or imported:

    from pathlib import Path
    from p1c_frames.visualization import discover_runs, generate_all
    generate_all(Path("results_p1c"), Path("blog_figures/p1c"))

Ten figure classes. See FIGURES-1c.md in this directory for the full
catalogue — every figure, what it shows, which artifact key it reads, and
its status — plus the four data gaps where a quantity is computed and
dropped, or never computed by the driver at all.

    style.py           palette and drawing primitives
    loaders.py         every disk read; the json/npz size rule resolved once
    _fixture.py        synthetic Phase 1c directory (test aid)
    integration.py     A — effective integration time, three definitions
    null_model.py      B — the gamma_beta residual, vertical and in time
    moments_fig.py     C — cumulant ladder, rank, the sink adjudication
    frames_fig.py      D — the four-frame table (needs a driver branch)
    feasibility.py     E — Lemma 6.4's cone condition as a margin
    designs.py         F — spherical designs, always against the band
    curiosities.py     exploratory figures off the trajectory
    theory.py          the null model itself — no artifacts needed
    cross_run.py       model × prompt, and what each run could answer
    checkpoints_1c.py  the training-step axis, P-γ1 and P-S1
    pipeline.py        generate_all / class orchestration
    cli.py             argparse entry point (main())

Three conventions worth knowing before reading the code.

**Analysis logic is imported, never copied.** Verdicts come from
`integration_time.verdict`, `moments.adjudicate_sink_hypothesis`,
`beta_reduction.envelope_verdict`, `gamma_null.adjudicate_p_gamma1` and
`centroids.adjudicate_p_s1_banded`; the null model itself comes from
`gamma_ode`; checkpoint step-axis conventions come from
`p1_mstate_tracking/visualization/checkpoints.py`. This package decides how
something is drawn and nothing about what it means, so a figure that
disagrees with a run's own artifact is a bug here by construction.

**The residual is the deliverable; the fit is not.** design-1c's first rule.
Every figure in the `null` class draws the gap rather than the agreement,
and the two nulls (orthogonal-init and observed-matched) are always drawn
together, because the distance between them is anisotropy and not
resistance.

**A NaN is a result here.** `time_residual` is NaN exactly where the network
de-clustered past its own starting point — the strongest resistance signal
the phase can produce. Those layers are drawn as marked bands and counted,
never clipped and never dropped.
"""

from .loaders import (
    SUBEXPS, Run, checkpoint_families, describe_runs, discover_runs,
)
from .pipeline import (
    CLASSES, CROSS_CLASSES, PER_RUN_CLASSES, STANDALONE_CLASSES, generate_all,
)
from .style import BLOG_STYLE, CATEGORICAL, RESIDUAL_CMAP, save_figure

__all__ = [
    "Run",
    "SUBEXPS",
    "discover_runs",
    "describe_runs",
    "checkpoint_families",
    "generate_all",
    "CLASSES",
    "PER_RUN_CLASSES",
    "CROSS_CLASSES",
    "STANDALONE_CLASSES",
    "BLOG_STYLE",
    "CATEGORICAL",
    "RESIDUAL_CMAP",
    "save_figure",
]
