"""
p2_eigenspectra/visualization/pipeline.py

One entry point for the Phase 2 figure classes, mirroring
`p1_mstate_tracking/visualization/checkpoint_pipeline.py`.

Order encodes the two-pass workflow and is not arbitrary:

  1. scalars  — first, because it writes transitions_p2_{base}.json.
  2. spectra  — independent of the others.
  3. clouds   — last, consuming the transitions dict to choose which
                checkpoints to draw. Running clouds alone later still
                works: `load_transitions` rereads the file from out_dir.

No-ops cleanly when the Phase 2 directory contains no '-step{N}' family,
so pointing it at the existing GPT-2/ALBERT results does nothing rather
than producing a one-point sweep.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from core.naming import _safe_model_name

from .loaders import discover_weight_summaries, discover_p2_runs
from .spectra import weight_families, generate_spectra_figures
from .p2_scalars import generate_scalar_figures
from .eigen_clouds import generate_eigen_cloud_figures

CLASSES = ("scalars", "spectra", "clouds")


def load_transitions(out_dir: Path, base: str) -> Optional[dict]:
    """Reread transitions_p2_{base}.json — lets `--classes clouds` reuse a
    previous pass's snapshot selection without recomputing the scalars."""
    p = Path(out_dir) / f"transitions_p2_{_safe_model_name(base)}.json"
    if not p.exists():
        return None
    try:
        return json.load(open(p)).get("per_metric")
    except Exception:
        return None


def generate_p2_figures(
    p2_dir: Path,
    out_dir: Path,
    classes: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    cloud_layers: Optional[List[int]] = None,
    filmstrip_k: int = 6,
    prompts: Optional[List[str]] = None,
) -> None:
    p2_dir, out_dir = Path(p2_dir), Path(out_dir)
    wanted = set(classes) if classes else set(CLASSES)

    summaries = discover_weight_summaries(p2_dir)
    runs = discover_p2_runs(p2_dir)
    if not summaries:
        print(f"⚠  no ov_summary_*.json under {p2_dir} — nothing to plot")
        return
    print(f"Discovered {len(summaries)} weight summaries, {len(runs)} runs "
          f"in {p2_dir}")

    families = weight_families(summaries)
    if not families:
        print("⚠  no '-step{N}' checkpoint family present — "
              "P2 checkpoint figures need a sweep, skipping")
        return

    for base, family in families.items():
        print(f"\n── {base}  ({len(family)} checkpoints) ──")
        transitions: Optional[dict] = None

        if "scalars" in wanted:
            transitions = generate_scalar_figures(
                summaries, runs, out_dir, base, family, prompts=prompts,
            )
        if transitions is None:
            transitions = load_transitions(out_dir, base)

        if "spectra" in wanted:
            generate_spectra_figures(summaries, out_dir, base, family,
                                     metrics=metrics)

        if "clouds" in wanted:
            generate_eigen_cloud_figures(
                p2_dir, summaries, out_dir, base, family,
                layers=cloud_layers, k=filmstrip_k, transitions=transitions,
            )
