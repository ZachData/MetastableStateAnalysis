"""
p2b_imaginary/visualization — Phase 2b figures.

Loads exclusively from a Phase 2b output directory (`phase2b_results.json`,
`{stem}/block1a_rotational_spectrum.json`,
`{stem}/{prompt}/block1b_rescaled_comparison.json`). No model weights, no
`ov_weights_*.npz`, no torch, no Schur decomposition, no recomputation. Run
as a single command:

    python -m p2b_imaginary.visualization \\
        --p2b_dir results/<phase2b-output-dir> \\
        --out     blog_figures/p2b \\
        [--classes spectrum heads frames trajectory report verdicts
                   nulls curiosities] \\
        [--steps …] [--prompts …] [--external …] [--list_runs] [--fixture]

or imported:

    from pathlib import Path
    from p2b_imaginary.visualization import load_sweep, generate_all
    generate_all(Path("results/phase2b"), Path("blog_figures/p2b"))

Eight figure classes. See FIGURES-2b.md in this directory for the full
catalogue — every figure, what it shows, which artifact it reads, and its
status — plus the seven data gaps, all of which have now landed in
`p2b_imaginary/` and are detected from the artifact rather than assumed.

    style.py        palette, step axis, drawing primitives
    loaders.py      every disk read; gaps DETECTED, never assumed
    _fixture.py     synthetic Phase 2b directory built by the phase's own code
    spectrum.py     Block 1a at one checkpoint, on the depth axis
    heads.py        per-head circuits — is the headline about any head?
    frames.py       Block 1b at one (checkpoint, prompt)
    trajectory.py   Block 1a across checkpoints — the training axis
    report_fig.py   p2b_report drawn: flatness, intervals, dated events
    verdicts.py     Block 1b across checkpoints and prompts
    nulls.py        the norm-matched Gaussian control (opt-in)
    curiosities.py  the exploratory half
    pipeline.py     generate_all / class orchestration
    cli.py          argparse entry point (main())

Four conventions worth knowing before reading the code.

**Analysis logic is imported, never restated.** The verdict vocabulary,
equivalence band, frame keys and tolerances come from
`rotational_rescaled`; the tracked statistics, dated events, flatness,
interval and alignment statistics come from `p2b_report`; the legacy-artifact
refusal and the checkpoint-step grammar come from `p2b_io`. A figure that
disagrees with `phase2b_summary.txt` is a bug here by construction.

**A refusal is never drawn at zero.** `elimination_rate` returns `None` with
a status for four distinct refusals, because the pre-rewrite code returned
the float `0.0` for all of them and that value entered a majority vote —
which would have returned a verdict by vacuity at exactly the checkpoints
where the theorem holds. Refusals get their own marker, their own gray, and
their own count.

**The invariance control is drawn as a control.** `remove_rotation` applies
an orthogonal map and reproduces the original frame by construction; reading
it as a measurement is the withdrawal `status-2b` opens with. It appears in
every frame figure, always hatched and labelled, and never in a comparison.

**Data gaps are detected from the artifact, not hardcoded.** All seven
emissions have since landed in `p2b_imaginary/`, and the figures that need
them started drawing against new directories with no change here — while old
directories keep skipping, which is the only behaviour that lets both stay
readable. That is what the detection was for.
"""

from .loaders import Checkpoint, Sweep, describe_sweep, load_sweep
from .pipeline import CLASSES, generate_all
from .style import BLOG_STYLE, CATEGORICAL, FRAME_COLORS, VERDICT_COLORS, save_figure

__all__ = [
    "Sweep",
    "Checkpoint",
    "load_sweep",
    "describe_sweep",
    "generate_all",
    "CLASSES",
    "BLOG_STYLE",
    "CATEGORICAL",
    "FRAME_COLORS",
    "VERDICT_COLORS",
    "save_figure",
]
