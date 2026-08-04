"""
p1_mstate_tracking/visualization/

Loads exclusively from saved artifacts in results/phase1/ — no model
weights. Run as a single command:

    python -m p1_mstate_tracking.visualization \\
        --results_dir results/phase1 \\
        --random_seed_dirs results/p1_random \\
        --out         blog_figures \\
        [--phase2_dir results/phase2] [--no_energy_decomposition] \\
        [--all_prompts] [--list_runs] \\
        [--random_run results/p1_random/20250101_120000]

or imported directly:

    from p1_mstate_tracking.visualization import discover_runs, generate_all
    from pathlib import Path
    runs = discover_runs(Path("results/phase1"))
    generate_all(runs, Path("blog_figures"), phase2_dir=Path("results/phase2"))

See README.md in this directory for the full output layout and figure
catalogue (per-model figures, overview figures, trained/random pair
figures, ablation, energy decomposition, energy-attribution prompt
aggregate). See each submodule's docstring for what it owns:

    style.py                       constants, colors, BLOG_STYLE
    naming.py                      model-name conventions, iteration filter
    loaders.py                     all disk reads
    plot_utils.py                  generic plot helpers (spans, projections)
    series.py                      shared per-layer extractors
    overview.py                    cross-model overview figures
    pair_comparisons.py            trained/random pair figures
    ablation.py                    sublayer-ablation figures
    energy_decomposition.py        single-prompt energy decomposition
    energy_attribution_aggregate.py  multi-prompt energy attribution
    per_model_figures.py           ip_histogram_migration, hdbscan_pca, projection_comparison
    cluster_reality.py             cluster_reality_* figure set
    random_aggregate.py            multi-seed random-control aggregation
    pipeline.py                    generate_all / ALL_PLOTS orchestration
    cli.py                         argparse entry point (main())
"""

from core.style import (
    DEFAULT_PROMPT, DEFAULT_LAYERS, MIN_CLUSTER_SIZE, LayerSpec,
    MODEL_COLORS, UNTRAINED_COLOR, RANDOM_COLOR_OVERRIDES, BLOG_STYLE,
)
from core.naming import filter_iteration_depths

from .loaders import discover_runs
from .pipeline import generate_all, generate_model_figures, ALL_PLOTS
from .random_aggregate import build_aggregate
from .energy_attribution_aggregate import (
    aggregate_energy_attribution_across_prompts,
    generate_energy_attribution_prompt_aggregate_figures,
)

__all__ = [
    "discover_runs",
    "generate_all",
    "generate_model_figures",
    "ALL_PLOTS",
    "filter_iteration_depths",
    "build_aggregate",
    "aggregate_energy_attribution_across_prompts",
    "generate_energy_attribution_prompt_aggregate_figures",
    "DEFAULT_PROMPT",
    "DEFAULT_LAYERS",
    "MIN_CLUSTER_SIZE",
    "LayerSpec",
    "MODEL_COLORS",
    "UNTRAINED_COLOR",
    "RANDOM_COLOR_OVERRIDES",
    "BLOG_STYLE",
]
