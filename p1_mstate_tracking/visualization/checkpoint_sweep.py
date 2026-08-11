"""
visualization/checkpoint_sweep.py

Class-1 figures, line variant: every checkpoint's depth profile on one
axes, colored by log(training step) — the reader reads the envelope (does
the family fan smoothly from random-like to trained, or jump), not
individual lines. The two baselines render per the checkpoints.py
conventions: '{base}-random' gray/dashed (multi-seed band when
random_agg has one), step 0 near-black/dotted, never from the colormap.

One figure per metric in CHECKPOINT_METRICS, plus one per β for the
interaction energy.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.style import BLOG_STYLE
from core.naming import _safe_model_name
from .loaders import _available_betas, _energy_series
from .series import _series_or_aggregate
from .checkpoints import (
    CHECKPOINT_METRICS, STEP0_STYLE, RANDOM_BASELINE_STYLE,
    step_norm, step_color, add_step_colorbar, family_baselines,
)


def _profile(run_dir: Optional[Path], fn) -> Optional[np.ndarray]:
    if run_dir is None:
        return None
    try:
        vals = fn(run_dir)
    except Exception:
        vals = None
    if not vals:
        return None
    return np.asarray(vals, dtype=float)


def plot_profile_sweep(
    runs: dict, out_dir: Path, prompt: str, base: str,
    family: List[Tuple[int, str]], metric_name: str,
    random_agg: Optional[dict] = None,
) -> None:
    """One metric, every checkpoint of one family, log-step colormap."""
    spec = CHECKPOINT_METRICS[metric_name]
    fn = spec["fn"]
    models = sorted({m for (m, p) in runs.keys() if p == prompt})
    baselines = family_baselines(base, models)

    plt.rcParams.update(BLOG_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    steps = [s for s, _ in family]
    norm = step_norm(steps)
    plotted = False

    for step, model in family:
        series = _profile(runs.get((model, prompt)), fn)
        if series is None:
            continue
        x = np.linspace(0, 1, len(series))
        if step == 0:
            ax.plot(x, series, **STEP0_STYLE, zorder=4,
                    label="step 0 (init)")
        else:
            ax.plot(x, series, color=step_color(step, norm),
                    linewidth=1.8, alpha=0.85, zorder=3)
        plotted = True

    rand = baselines["random"]
    if rand is not None:
        mean_r, std_r, n_seeds = _series_or_aggregate(
            rand, prompt, runs.get((rand, prompt)), fn,
            spec["agg_key"], random_agg,
        )
        if mean_r is not None:
            x_r = np.linspace(0, 1, len(mean_r))
            ax.plot(x_r, mean_r, **RANDOM_BASELINE_STYLE, zorder=4,
                    label="norm-matched random"
                          + (f" (n={n_seeds} seeds)" if n_seeds > 1 else ""))
            if std_r is not None and std_r.size == mean_r.size:
                ax.fill_between(x_r, mean_r - std_r, mean_r + std_r,
                                color=RANDOM_BASELINE_STYLE["color"],
                                alpha=0.15, zorder=1, linewidth=0)
            plotted = True

    if not plotted:
        plt.close(fig)
        print(f"  ⚠  sweep_{metric_name}: nothing to plot for {base!r} @ {prompt!r}")
        return

    add_step_colorbar(fig, ax, steps, norm)
    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel(spec["ylabel"])
    if spec.get("ylim") is not None:
        ax.set_ylim(*spec["ylim"])
    ax.set_xlim(0, 1)
    ax.set_title(
        f"{spec['title']} vs. depth across training  ·  {base}  ·  {prompt}",
        fontsize=12, fontweight="bold",
    )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="best")

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"sweep_{metric_name}_{_safe_model_name(base)}_{prompt}.png"
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


def plot_energy_sweep(
    runs: dict, out_dir: Path, prompt: str, base: str,
    family: List[Tuple[int, str]],
) -> None:
    """Interaction energy E_β vs. layer across all checkpoints — one panel
    per β found in any family member's energies.json. The monotone-increase
    prediction — eq. (3.6) for (SA), Lemma 3.7 for (USA), both under
    V = +I_d — reads directly as whether any color band dips. A dip is the
    V = -I_d repulsive regime (§3.2, §9.1). Formerly mis-cited as Thm 3.4."""
    betas: List[float] = []
    for _, model in family:
        rd = runs.get((model, prompt))
        if rd is not None:
            betas = _available_betas(rd)
            if betas:
                break
    if not betas:
        print(f"  ⚠  sweep_energy: no energies.json found for {base!r} @ {prompt!r}")
        return

    models = sorted({m for (m, p) in runs.keys() if p == prompt})
    baselines = family_baselines(base, models)
    steps = [s for s, _ in family]
    norm = step_norm(steps)

    plt.rcParams.update(BLOG_STYLE)
    ncol = min(len(betas), 2)
    nrow = int(np.ceil(len(betas) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 4.4 * nrow),
                             squeeze=False)
    plotted_any = False

    for bi, beta in enumerate(betas):
        ax = axes[bi // ncol][bi % ncol]
        for step, model in family:
            rd = runs.get((model, prompt))
            if rd is None:
                continue
            series = _energy_series(rd, beta)
            if series is None:
                continue
            series = np.asarray(series, dtype=float)
            x = np.linspace(0, 1, len(series))
            if step == 0:
                ax.plot(x, series, **STEP0_STYLE, zorder=4)
            else:
                ax.plot(x, series, color=step_color(step, norm),
                        linewidth=1.6, alpha=0.85, zorder=3)
            plotted_any = True

        rand = baselines["random"]
        if rand is not None and (rand, prompt) in runs:
            series = _energy_series(runs[(rand, prompt)], beta)
            if series is not None:
                series = np.asarray(series, dtype=float)
                ax.plot(np.linspace(0, 1, len(series)), series,
                        **RANDOM_BASELINE_STYLE, zorder=4)
                plotted_any = True

        ax.set_title(f"β = {beta:g}", fontsize=11)
        ax.set_xlabel("Normalized layer depth")
        ax.set_ylabel(r"$E_\beta$")
        ax.set_xlim(0, 1)

    for bi in range(len(betas), nrow * ncol):
        axes[bi // ncol][bi % ncol].axis("off")

    if not plotted_any:
        plt.close(fig)
        print(f"  ⚠  sweep_energy: nothing to plot for {base!r} @ {prompt!r}")
        return

    add_step_colorbar(fig, axes[0][-1], steps, norm)
    fig.suptitle(
        f"Interaction energy across training  ·  {base}  ·  {prompt}\n"
        "dotted black = step 0 (init)  ·  gray dashed = norm-matched random",
        fontsize=12, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"sweep_energy_{_safe_model_name(base)}_{prompt}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


def generate_sweep_figures(
    runs: dict, out_dir: Path, prompt: str, base: str,
    family: List[Tuple[int, str]], random_agg: Optional[dict] = None,
) -> None:
    if len(family) < 2:
        print(f"  ⚠  sweep: family {base!r} has <2 checkpoints, skipping")
        return
    for metric_name in CHECKPOINT_METRICS:
        plot_profile_sweep(runs, out_dir, prompt, base, family, metric_name,
                           random_agg=random_agg)
    plot_energy_sweep(runs, out_dir, prompt, base, family)
