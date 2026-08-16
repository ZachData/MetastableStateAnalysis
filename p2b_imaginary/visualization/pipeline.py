"""
p2b_imaginary/visualization/pipeline.py

One entry point for every figure class, mirroring
`p1b_hemisphere/visualization/pipeline.py` and
`p2_eigenspectra/visualization/pipeline.py`.

Layout differs from those two in one way worth stating. Phase 1b's unit is a
(model, prompt) run and its figures split cleanly into per-run and cross-run;
Phase 2b's unit is a CHECKPOINT, and three of its seven classes straddle the
split — `nulls` draws per checkpoint and once across the sweep, `curiosities`
does the same, and `trajectory`'s coverage figure is meaningful for a
one-checkpoint directory while the rest of the class is not. So each class
here takes the whole `Sweep` and decides its own layout through
`loaders.checkpoint_out` / `prompt_out` / `cross_out`, rather than the
pipeline sorting classes into two buckets. The output tree still mirrors the
input tree: `{out}/{stem}/`, `{out}/{stem}/{prompt}/`, `{out}/_cross/`.

Order is not load-bearing: no class consumes another's output, so `--classes`
can name any subset and each will produce the same figures it would have
produced in a full pass.

Skips are printed by the class that skips, once, with the reason — a sweep
run with `--blocks 1a` has no Block 1b anywhere, and printing that per
checkpoint per figure would bury everything else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .curiosities import generate_curiosity_figures
from .frames import generate_frame_figures
from .heads import generate_head_figures
from .loaders import Sweep, load_sweep
from .nulls import generate_null_figures
from .report_fig import generate_report_figures
from .spectrum import generate_spectrum_figures
from .trajectory import generate_trajectory_figures
from .verdicts import generate_verdict_figures

__all__ = ["CLASSES", "generate_all"]

#: Every figure class, in the order a reader would want them: what the
#: operator looks like, whether that description holds per head, what the
#: rescaling did, how both move over training, what the report concludes,
#: what the controls say, and then the speculative half.
CLASSES = ("spectrum", "heads", "frames", "trajectory", "report", "verdicts",
           "nulls", "curiosities")

_FNS = {
    "spectrum":    generate_spectrum_figures,
    "heads":       generate_head_figures,
    "frames":      generate_frame_figures,
    "trajectory":  generate_trajectory_figures,
    "report":      generate_report_figures,
    "verdicts":    generate_verdict_figures,
    "nulls":       generate_null_figures,
    "curiosities": generate_curiosity_figures,
}


def generate_all(
    p2b_dir: Path,
    out_dir: Path,
    classes: Optional[Sequence[str]] = None,
    steps: Optional[Sequence[int]] = None,
    prompts: Optional[Sequence[str]] = None,
    sweep: Optional[Sweep] = None,
    external: Optional[dict] = None,
) -> Dict[str, List[Path]]:
    """
    Draw every requested class over one Phase 2b output directory.

    Returns {class: [paths]} so a caller — the smoke test, most of all — can
    assert on what was produced rather than on a printed line.
    """
    p2b_dir, out_dir = Path(p2b_dir), Path(out_dir)
    wanted = set(classes) if classes else set(CLASSES)
    unknown = wanted - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown figure class(es): {sorted(unknown)}; "
                         f"known: {list(CLASSES)}")

    if sweep is None:
        sweep = load_sweep(p2b_dir, steps=steps, prompts=prompts)
    if sweep is None:
        print(f"⚠  no Phase 2b sweep found under {p2b_dir} — nothing to plot")
        return {}

    print(f"Phase 2b sweep: {len(sweep.checkpoints)} checkpoint(s), "
          f"{len(sweep.prompts)} prompt(s), blocks "
          f"{', '.join(sweep.blocks) or 'none'}   [read from {sweep.source}]")
    if sweep.missing_checkpoints:
        print(f"  ⚠  {len(sweep.missing_checkpoints)} checkpoint(s) have no "
              f"OV weights: {sweep.missing_checkpoints}")
    if sweep.n_failed:
        print(f"  ⚠  {sweep.n_failed} prompt(s) failed during the sweep — "
              "results below are incomplete")
    for gap in sweep.gaps:
        print(f"  ·  {gap['id']} open: {gap['what']}")

    produced: Dict[str, List[Path]] = {c: [] for c in sorted(wanted)}
    out_dir.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        if cls not in wanted:
            continue
        print(f"\n── {cls} ──")
        kwargs = {"external": external} if cls == "report" else {}
        paths = _FNS[cls](sweep, out_dir, **kwargs)
        produced[cls].extend(paths)
        print(f"  {cls:<12} {len(paths)} figure(s)")

    total = sum(len(v) for v in produced.values())
    print(f"\n{total} figure(s) written under {out_dir}")
    return produced
