"""
visualization/pipeline.py

Orchestration only — no plotting code of its own. ALL_PLOTS is the
per-model dispatch list (generate_model_figures runs every entry against
one run_dir); generate_all is the single function that produces every
figure this package knows how to make, given discovered runs and an
output directory. cli.py is the only caller that should need to know
about every submodule below — everything else imports through here.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .style import LayerSpec, DEFAULT_PROMPT
from .naming import filter_iteration_depths, _safe_model_name
from .loaders import discover_runs
from .overview import generate_overview_figures
from .pair_comparisons import generate_noise_comparison_figures
from .ablation import generate_ablation_figures
from .energy_decomposition import generate_energy_decomposition_figures
from .per_model_figures import (
    plot_ip_histogram_migration,
    plot_hdbscan_pca,
    plot_projection_comparison,
)
from .cluster_reality import (
    plot_cluster_size_histogram,
    plot_sorted_gram_heatmap,
    plot_within_between_ip_histogram,
    plot_pca_scatter_by_cluster,
    plot_cluster_persistence_river,
    plot_cluster_colored_text,
)
from .cluster_orthogonality import plot_cluster_orthogonality_trajectory
from .ip_population_dynamics import (
    plot_ip_histogram_depth_heatmap,
    plot_ip_population_trajectory,
)
from .noise_importance_proxy import plot_noise_importance_proxies
from .attractor_alignment import (
    plot_attractor_alignment_overview,
    plot_unclustered_pull_decomposition,
)

# ═════════════════════════════════════════════════════════════════════════════
# Dispatch + driver
# ═════════════════════════════════════════════════════════════════════════════

# Every entry shares the same signature: fn(run_dir, out_dir, layers=None).
# No blog/reality distinction — this is just "every figure we make per model".
# energy_trajectory is NOT here — it moved to the trained/random pair circuit
# (see plot_energy_trajectory_pair, called from generate_noise_comparison_figures)
# so trained and random show up together on each β panel instead of as two
# separate per-model figures.


def _ignore_layers(fn, **fixed_kwargs):
    """
    Adapter for figures whose analysis is inherently full-depth and takes
    no layers= argument. Keeps ALL_PLOTS's "every entry takes
    fn(run_dir, out_dir, layers=None)" invariant intact instead of adding
    an unused layers param to modules that don't use one.
    """
    def wrapped(run_dir, out_dir, layers=None):
        return fn(run_dir, out_dir, **fixed_kwargs)
    wrapped.__name__ = fn.__name__   # generate_model_figures prints fn.__name__ on error
    return wrapped

def generate_model_figures(
    run_dir: Path, out_dir: Path, layers: Optional[List[LayerSpec]] = None,
) -> None:
    """Run every per-model plot function against one (model, prompt) run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn in ALL_PLOTS:
        try:
            fn(run_dir, out_dir, layers=layers)
        except Exception as e:
            print(f"  ✗  {fn.__name__} raised: {e}")


def generate_all(
    runs: dict, out_dir: Path,
    prompts: Optional[List[str]] = None,
    layers: Optional[List[LayerSpec]] = None,
    random_agg: Optional[dict] = None,
    phase2_dir: Optional[Path] = None,
) -> None:
    """
    Generate every visualization: the cross-model overview figures for each
    prompt, then every per-model figure for every (model, prompt) pair that
    actually has a saved run — each model variant gets its own out_dir
    subfolder, named after its full literal model string (so a random
    control and an iteration-depth checkpoint are never folded into the
    same folder as their "parent" model).

    random_agg, if given, is the {(model, prompt): aggregate_dict} built by
    random_aggregate.py (or build_aggregate() on the fly) from every seed
    under results/p1_random/ — it makes every "random" line in the
    overview and trained/random pair figures a multi-seed mean ± std band
    instead of whichever single seed ended up in `runs`. Per-model figures
    (PCA scatters, etc.) are unaffected — those only ever show one run.

    phase2_dir, if given and found on disk, additionally cross-references
    every (model, prompt) here against a matching Phase 2 `--full` decompose
    run, generating the energy-decomposition figures (attention structure
    vs. attn/FFN energy attribution). Models with no Phase 2 run are skipped
    silently — this never blocks the rest of the figures.
    """
    runs = filter_iteration_depths(runs)
    prompts = prompts or [DEFAULT_PROMPT]

    print("Overview figures (cross-model)")
    for prompt in prompts:
        generate_overview_figures(runs, out_dir, prompt, random_agg=random_agg)
        generate_noise_comparison_figures(runs, out_dir, prompt, random_agg=random_agg)
        generate_ablation_figures(runs, out_dir, prompt)
        generate_energy_decomposition_figures(runs, phase2_dir, out_dir, prompt)

    models = sorted({m for (m, p) in runs.keys()})
    print(f"\nPer-model figures — {len(models)} model variant(s)")
    for model in models:
        model_dir = out_dir / _safe_model_name(model)
        for prompt in prompts:
            run_dir = runs.get((model, prompt))
            if run_dir is None:
                continue
            print(f"\n  {model} | {prompt}")
            generate_model_figures(run_dir, model_dir, layers=layers)



def _discover_random_dir(rd: Path) -> Dict[Tuple[str, str], Path]:
    """
    Discover runs under a --random_seed_dirs entry.

    Tries rd itself as a run directory first (per-run subdirs with
    geometry.json directly inside). If that finds nothing, treats rd as a
    parent containing several timestamped run directories and merges
    discover_runs() across all of its immediate subdirectories.
    """
    direct = discover_runs(rd)
    if direct:
        return direct

    merged: Dict[Tuple[str, str], Path] = {}
    if not rd.is_dir():
        return merged
    for sub in sorted(rd.iterdir()):
        if sub.is_dir():
            merged.update(discover_runs(sub))
    return merged

ALL_PLOTS = [
    plot_ip_histogram_migration,
    plot_hdbscan_pca,
    plot_projection_comparison,
    plot_cluster_size_histogram,
    plot_sorted_gram_heatmap,
    plot_within_between_ip_histogram,
    plot_pca_scatter_by_cluster,
    plot_cluster_persistence_river,
    plot_cluster_colored_text,
    _ignore_layers(plot_cluster_orthogonality_trajectory, top_k=5, rank_by="persistence"),
    _ignore_layers(plot_ip_histogram_depth_heatmap, beta=1.0),
    _ignore_layers(plot_ip_population_trajectory, beta=1.0),
    _ignore_layers(plot_noise_importance_proxies),
    _ignore_layers(plot_attractor_alignment_overview),
    _ignore_layers(plot_unclustered_pull_decomposition),
]