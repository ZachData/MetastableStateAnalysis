"""
visualization/overview.py

The cross-model overview figures — the only charts with more than one
model variant on the same axes: mass_near_1 (+ log-scale), effective_rank,
cluster_membership, cluster_count. One line per model variant; "-random"
controls render dashed/gray (or their RANDOM_COLOR_OVERRIDES color) with
a mean ± std band when random_agg has a multi-seed entry for them.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from core.style import BLOG_STYLE, DEGENERATE_RANK
from core.naming import _is_untrained, _random_color, _color, _label
from .series import (
    _series_or_aggregate,
    _mass_near_1_series,
    _effective_rank_series,
    _cluster_membership_series,
    _cluster_count_series,
)

def _overview_line_plot(
    runs: dict, out_dir: Path, prompt: str, *,
    value_fn, ylabel: str, title: str, fname_prefix: str,
    yscale: str = "linear", ylim: Optional[Tuple[float, float]] = None,
    ref_line: Optional[float] = None,
    agg_key: Optional[str] = None, random_agg: Optional[dict] = None,
) -> None:
    """
    One chart, one line per model variant present at `prompt`. Shared by
    every overview_* figure — they differ only in which per-layer series
    value_fn pulls out and how the axes are styled.

    When agg_key is given and random_agg has a matching multi-seed entry
    for a given "-random" model, that model's line is the across-seed mean
    with a shaded ±1 std band, instead of one seed's raw series.
    """
    plt.rcParams.update(BLOG_STYLE)
    models = sorted(m for (m, p) in runs.keys() if p == prompt)
    if not models:
        print(f"  ⚠  {fname_prefix}: no runs for prompt {prompt!r}, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    legend_handles: Dict[str, mpatches.Patch] = {}

    for model in models:
        run_dir = runs.get((model, prompt))
        mean_vals, std_vals, n_seeds = _series_or_aggregate(
            model, prompt, run_dir, value_fn, agg_key, random_agg,
        )
        if mean_vals is None:
            continue
        n     = len(mean_vals)
        x     = np.linspace(0, 1, n)
        utr   = _is_untrained(model)
        color = _random_color(model) if utr else _color(model)
        ax.plot(
            x, mean_vals, color=color, linewidth=2.0 if utr else 2.4,
            alpha=0.75 if utr else 0.9, linestyle="--" if utr else "-", zorder=3,
        )
        if std_vals is not None and std_vals.size == mean_vals.size:
            ax.fill_between(
                x, mean_vals - std_vals, mean_vals + std_vals,
                color=color, alpha=0.15, zorder=1, linewidth=0,
            )
        label = _label(model) + (f"  (n={n_seeds} seeds, ±1σ)" if n_seeds > 1 else "")
        legend_handles[model] = mpatches.Patch(color=color, label=label)

    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  ·  prompt: {prompt}", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yscale != "linear":
        ax.set_yscale(yscale)
    if ref_line is not None:
        ax.axhline(ref_line, color="#EF4444", linewidth=0.9, linestyle="--", alpha=0.6, zorder=5)
    if legend_handles:
        ax.legend(handles=list(legend_handles.values()), loc="best", fontsize=8, ncol=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{fname_prefix}_{prompt}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  ({len(legend_handles)} model variants)")


def generate_overview_figures(
    runs: dict, out_dir: Path, prompt: str, random_agg: Optional[dict] = None,
) -> None:
    """All cross-model comparison figures for one prompt."""
    _overview_line_plot(
        runs, out_dir, prompt,
        value_fn=_mass_near_1_series, agg_key="mass_near_1", random_agg=random_agg,
        ylabel="Fraction of token pairs with ⟨xᵢ, xⱼ⟩ > 0.9",
        title="Mass-near-1 across all models",
        fname_prefix="overview_mass_near_1",
        ylim=(-0.02, 1.06),
    )
    _overview_line_plot(
        runs, out_dir, prompt,
        value_fn=_mass_near_1_series, agg_key="mass_near_1", random_agg=random_agg,
        ylabel="Fraction of pairs  ⟨xᵢ,xⱼ⟩ > 0.9  (log scale)",
        title="Mass-near-1 across all models (log scale)",
        fname_prefix="overview_mass_near_1_logscale",
        yscale="log", ylim=(1e-3, 1.5),
    )
    _overview_line_plot(
        runs, out_dir, prompt,
        value_fn=_effective_rank_series, agg_key="effective_rank", random_agg=random_agg,
        ylabel="Effective rank",
        title="Effective rank collapses as tokens cluster",
        fname_prefix="overview_effective_rank",
        ref_line=DEGENERATE_RANK,
    )
    _overview_line_plot(
        runs, out_dir, prompt,
        value_fn=_cluster_membership_series, agg_key="cluster_membership", random_agg=random_agg,
        ylabel="Fraction of tokens assigned to a cluster\n(1 − HDBSCAN noise fraction)",
        title="Particles in clusters vs. labeled noise",
        fname_prefix="overview_cluster_membership",
        ylim=(-0.02, 1.06),
    )
    _overview_line_plot(
        runs, out_dir, prompt,
        value_fn=_cluster_count_series, agg_key="cluster_count", random_agg=random_agg,
        ylabel="HDBSCAN cluster count (k)",
        title="Cluster count across all models",
        fname_prefix="overview_cluster_count",
    )
