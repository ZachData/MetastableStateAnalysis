"""
p1b_hemisphere/visualization/pipeline.py

One entry point for every figure class, mirroring
`p2_eigenspectra/visualization/pipeline.py`.

Layout: per-run classes write into `{out}/{stem}/`, one directory per run, so
a 27-checkpoint family does not put 27 files named `cone_margin_depth.png`
in one folder. Cross-run classes write into `{out}/_cross/`, underscore-
prefixed so it sorts above the run directories and is obviously not a run.

Order is not arbitrary but it is also not load-bearing: no class consumes
another's output (unlike Phase 2, where clouds read the scalars' transition
file), so `--classes` can name any subset and each will produce the same
figures it would have produced in a full pass.

Missing inputs are reported once per run at the top rather than per figure,
because a directory of runs that all predate an emission would otherwise
print the same line forty times.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .axis import generate_axis_cross_figures, generate_axis_figures
from .checkpoints_1b import generate_checkpoint_figures
from .cone import generate_cone_cross_figures, generate_cone_figures
from .cross_run import generate_cross_run_figures
from .curiosities import generate_curiosity_figures
from .loaders import Run, discover_runs, load_cross_run
from .membership import generate_membership_figures
from .regime import generate_regime_figures
from .tracking import generate_tracking_figures

__all__ = ["CLASSES", "PER_RUN_CLASSES", "CROSS_CLASSES", "generate_all"]

#: Per-run classes: one output directory per run stem.
PER_RUN_CLASSES = ("regime", "cone", "tracking", "membership", "axis",
                   "curiosities")

#: Cross-run classes: one shared `_cross/` directory.
CROSS_CLASSES = ("crossrun", "checkpoints")

CLASSES = PER_RUN_CLASSES + CROSS_CLASSES

_PER_RUN_FNS = {
    "regime":      generate_regime_figures,
    "cone":        generate_cone_figures,
    "tracking":    generate_tracking_figures,
    "membership":  generate_membership_figures,
    "axis":        generate_axis_figures,
    "curiosities": generate_curiosity_figures,
}


def generate_all(
    p1b_dir: Path,
    out_dir: Path,
    classes: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    prompts: Optional[Sequence[str]] = None,
    runs: Optional[Sequence[Run]] = None,
) -> Dict[str, List[Path]]:
    """
    Draw every requested class over every discovered run.

    Returns {class: [paths]} so a caller — the smoke test, most of all — can
    assert on what was produced rather than on a printed line.
    """
    p1b_dir, out_dir = Path(p1b_dir), Path(out_dir)
    wanted = set(classes) if classes else set(CLASSES)
    unknown = wanted - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown figure class(es): {sorted(unknown)}; "
                         f"known: {list(CLASSES)}")

    if runs is None:
        runs = discover_runs(p1b_dir, models=models, prompts=prompts)
    if not runs:
        print(f"⚠  no Phase 1b runs found under {p1b_dir} — nothing to plot")
        return {}

    print(f"Discovered {len(runs)} run(s) in {p1b_dir}")
    for run in runs:
        if run.missing:
            print(f"  {run.stem}: missing {len(run.missing)} optional input(s)"
                  f" — {run.missing[0]}"
                  + (f" (+{len(run.missing) - 1} more)"
                     if len(run.missing) > 1 else ""))

    produced: Dict[str, List[Path]] = {c: [] for c in sorted(wanted)}
    out_dir.mkdir(parents=True, exist_ok=True)

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
    cross_run = load_cross_run(p1b_dir)

    if "crossrun" in wanted:
        print("\n── cross-run ──")
        paths = generate_cross_run_figures(runs, cross_out, cross_run=cross_run)
        # The two pooled figures that belong to a block but need every run.
        paths += generate_cone_cross_figures(runs, cross_out)
        paths += generate_axis_cross_figures(runs, cross_out)
        produced["crossrun"].extend(paths)
        print(f"  crossrun     {len(paths)} figure(s)")

    if "checkpoints" in wanted:
        print("\n── checkpoints ──")
        paths = generate_checkpoint_figures(runs, cross_out, cross_run=cross_run)
        produced["checkpoints"].extend(paths)
        print(f"  checkpoints  {len(paths)} figure(s)")

    total = sum(len(v) for v in produced.values())
    print(f"\n{total} figure(s) written under {out_dir}")
    return produced
