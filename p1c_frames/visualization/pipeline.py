"""
p1c_frames/visualization/pipeline.py

One entry point for every figure class, mirroring
`p1b_hemisphere/visualization/pipeline.py` and
`p2_eigenspectra/visualization/pipeline.py`.

Layout: per-run classes write into `{out}/{stem}/`, one directory per run,
so a 27-checkpoint family does not put 27 files named `residual_curve.png`
in one folder. Cross-run classes write into `{out}/_cross/` and the theory
class into `{out}/_theory/` — both underscore-prefixed so they sort above
the run directories and are obviously not runs.

`theory` is the one class that takes no runs at all. It is therefore the one
class that still draws when `--p1c_dir` finds nothing, which is deliberate:
Phase 1c has not been run against Pythia artifacts yet (status-1c), and the
null model it will be compared against is drawable today.

Order is not arbitrary but it is also not load-bearing: no class consumes
another's output, so `--classes` can name any subset and each will produce
the same figures it would have produced in a full pass.

Missing inputs are reported once per run at the top rather than per figure,
because a directory of runs that all predate the `norms` fix would otherwise
print the same four lines forty times.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .checkpoints_1c import generate_checkpoint_figures
from .cross_run import generate_cross_run_figures
from .curiosities import (
    generate_curiosity_cross_figures, generate_curiosity_figures,
)
from .designs import generate_design_figures
from .feasibility import generate_feasibility_figures
from .frames_fig import generate_frame_figures
from .integration import generate_integration_figures
from .loaders import Run, discover_runs
from .moments_fig import generate_moment_figures
from .null_model import generate_null_figures
from .theory import generate_theory_figures

__all__ = ["CLASSES", "PER_RUN_CLASSES", "CROSS_CLASSES", "STANDALONE_CLASSES",
           "generate_all"]

#: Per-run classes: one output directory per run stem.
PER_RUN_CLASSES = ("integration", "null", "moments", "frames", "feasibility",
                   "designs", "curiosities")

#: Cross-run classes: one shared `_cross/` directory.
CROSS_CLASSES = ("crossrun", "checkpoints")

#: Needs no runs at all — draws the phase's own null model.
STANDALONE_CLASSES = ("theory",)

CLASSES = PER_RUN_CLASSES + CROSS_CLASSES + STANDALONE_CLASSES

_PER_RUN_FNS = {
    "integration": generate_integration_figures,
    "null":        generate_null_figures,
    "moments":     generate_moment_figures,
    "frames":      generate_frame_figures,
    "feasibility": generate_feasibility_figures,
    "designs":     generate_design_figures,
    "curiosities": generate_curiosity_figures,
}


def generate_all(
    p1c_dir: Optional[Path],
    out_dir: Path,
    classes: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    prompts: Optional[Sequence[str]] = None,
    runs: Optional[Sequence[Run]] = None,
    cheap: bool = False,
) -> Dict[str, List[Path]]:
    """
    Draw every requested class over every discovered run.

    Returns `{class: [paths]}` so a caller — the smoke test, most of all —
    can assert on what was produced rather than on a printed line.

    `p1c_dir` may be None when only `theory` is requested; that is the whole
    point of the standalone class and the CLI relies on it.
    """
    out_dir = Path(out_dir)
    wanted = set(classes) if classes else set(CLASSES)
    unknown = wanted - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown figure class(es): {sorted(unknown)}; "
                         f"known: {list(CLASSES)}")

    produced: Dict[str, List[Path]] = {c: [] for c in sorted(wanted)}
    out_dir.mkdir(parents=True, exist_ok=True)

    needs_runs = wanted & set(PER_RUN_CLASSES + CROSS_CLASSES)
    if needs_runs:
        if runs is None:
            if p1c_dir is None:
                print("⚠  no --p1c_dir given — only `theory` can be drawn")
                runs = []
            else:
                runs = discover_runs(Path(p1c_dir), models=models,
                                     prompts=prompts)
        if not runs:
            print(f"⚠  no Phase 1c runs found under {p1c_dir} — "
                  f"nothing to plot")
        else:
            print(f"Discovered {len(runs)} run(s) in {p1c_dir}")
            for run in runs:
                if run.missing:
                    print(f"  {run.stem}: {len(run.missing)} absent input(s)"
                          f" — {run.missing[0].split(':')[0]}"
                          + (f" (+{len(run.missing) - 1} more; --list_runs "
                             f"for all)" if len(run.missing) > 1 else ""))
    runs = list(runs or [])

    # -- per-run ------------------------------------------------------------
    for run in runs:
        run_out = out_dir / run.stem
        drew = False
        for cls in PER_RUN_CLASSES:
            if cls not in wanted:
                continue
            if not drew:
                print(f"\n── {run.label} ──")
                drew = True
            paths = _PER_RUN_FNS[cls](run, run_out)
            produced[cls].extend(paths)
            print(f"  {cls:<12} {len(paths)} figure(s)")

    # -- cross-run ----------------------------------------------------------
    cross_out = out_dir / "_cross"
    if "crossrun" in wanted and runs:
        print("\n── cross-run ──")
        paths = generate_cross_run_figures(runs, cross_out)
        # The pooled curiosities belong to their class but need every run.
        paths += generate_curiosity_cross_figures(runs, cross_out)
        produced["crossrun"].extend(paths)
        print(f"  crossrun     {len(paths)} figure(s)")

    if "checkpoints" in wanted and runs:
        print("\n── checkpoints ──")
        paths = generate_checkpoint_figures(runs, cross_out)
        produced["checkpoints"].extend(paths)
        print(f"  checkpoints  {len(paths)} figure(s)")

    # -- standalone ---------------------------------------------------------
    if "theory" in wanted:
        print("\n── theory (no artifacts) ──")
        paths = generate_theory_figures(out_dir / "_theory", cheap=cheap)
        produced["theory"].extend(paths)
        print(f"  theory       {len(paths)} figure(s)")

    total = sum(len(v) for v in produced.values())
    print(f"\n{total} figure(s) written under {out_dir}")
    return produced
