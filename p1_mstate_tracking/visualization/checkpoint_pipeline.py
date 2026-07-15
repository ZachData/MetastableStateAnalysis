"""
visualization/checkpoint_pipeline.py

Orchestration for the checkpoint figure classes — mirrors pipeline.py's
role for the per-model/pair figures. generate_checkpoint_figures is the
one entry point; pipeline.generate_all calls it after the existing
overview/pair circuit, and it no-ops when no '-step{N}' families are
present, so nothing about non-Pythia runs changes.

Order matters and encodes the two-pass workflow: scalars run FIRST so
transitions.json exists, heatmaps and sweeps second (independent), and
filmstrips LAST, consuming the transitions dict to pick their snapshot
steps. Passing --filmstrips-only later still works because
checkpoint_filmstrip.load_transitions rereads transitions.json from
out_dir.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .checkpoints import checkpoint_families
from .checkpoint_scalars import generate_scalar_figures
from .checkpoint_heatmaps import generate_heatmap_figures
from .checkpoint_sweep import generate_sweep_figures
from .checkpoint_filmstrip import generate_filmstrip_figures


def generate_checkpoint_figures(
    runs: dict, out_dir: Path, prompt: str,
    random_agg: Optional[dict] = None,
    filmstrip_k: int = 6,
    filmstrip_layer: Optional[int] = None,
    classes: Optional[List[str]] = None,
) -> None:
    """
    Every checkpoint-sweep figure for every '-step{N}' family present at
    `prompt`. `classes` limits which figure classes run
    (subset of {'scalars', 'heatmaps', 'sweeps', 'filmstrips'}; default
    all). Figures land in out_dir/checkpoints/<base>/.
    """
    classes = set(classes or ("scalars", "heatmaps", "sweeps", "filmstrips"))
    models = sorted({m for (m, p) in runs.keys() if p == prompt})
    fams = checkpoint_families(models)
    if not fams:
        return

    for base, family in fams.items():
        fam_dir = out_dir / "checkpoints" / base.replace("/", "_")
        print(f"\nCheckpoint figures — {base} ({len(family)} checkpoints) | {prompt}")

        transitions = None
        if "scalars" in classes:
            transitions = generate_scalar_figures(
                runs, fam_dir, prompt, base, family, random_agg=random_agg,
            )
        if "heatmaps" in classes:
            generate_heatmap_figures(runs, fam_dir, prompt, base, family)
        if "sweeps" in classes:
            generate_sweep_figures(runs, fam_dir, prompt, base, family,
                                   random_agg=random_agg)
        if "filmstrips" in classes:
            generate_filmstrip_figures(
                runs, fam_dir, prompt, base, family,
                transitions=transitions, k=filmstrip_k, layer=filmstrip_layer,
            )
