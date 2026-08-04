"""
visualization/pair_comparisons.py

Trained/random pair figures — one matched (trained, random) control per
chart, never more: noise_persistence, membership_decoupling (+ the merged
fiedler/cka trajectory variants), and energy_trajectory. Also responsible
for finding which '-random' variant belongs to which trained model
(_trained_random_pairs) and for generating the whole noise-comparison set
for one prompt (generate_noise_comparison_figures).

CKA_YLIM is fixed to [0.9, 1.0]: linear CKA against the previous layer
sits in that band once depth increases, and a [0, 1] axis flattens it to
an invisible line at the top.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.style import BLOG_STYLE
from core.naming import _is_untrained, _color, _random_color, _label, _safe_model_name
from .loaders import _trajectory, _available_betas, _energy_series
from core.plot_utils import _shade_plateaus
from .series import (
    _series_or_aggregate,
    _cluster_membership_series,
    _cluster_count_series,
    _cka_prev_series,
    _fiedler_mean_series,
    _token_in_cluster_fraction,
)

def _trained_random_pairs(models: List[str]) -> List[Tuple[str, str]]:
    """
    For every '-random' variant present, find its trained counterpart.
    Handles the '-random@Niter' ALBERT convention (random control run at
    the same iteration depth) as well as plain '<model>-random'.
    Returns [(trained, random), ...] for pairs where both sides exist.
    """
    randoms = sorted(m for m in models if _is_untrained(m))
    pairs = []
    for r in randoms:
        if "-random@" in r:
            trained_guess = r.replace("-random@", "@")
        elif r.endswith("-random"):
            trained_guess = r[: -len("-random")]
        else:
            trained_guess = r.replace("-random", "")
        if trained_guess in models and trained_guess != r:
            pairs.append((trained_guess, r))
    return pairs

def plot_noise_persistence_pair(
    runs: dict, out_dir: Path, prompt: str, trained_model: str, random_model: str,
    random_agg: Optional[dict] = None,
) -> None:
    """
    Left: per-token "fraction of layers spent in a real cluster" density,
    trained vs. random overlaid — the token-level answer to "what fraction
    of particles are ever in a cluster, and is that different once trained."
    Right: the aggregate membership trajectory for this specific pair, with
    the trained/random gap shaded so it reads as area, not just two lines.

    When random_agg has a multi-seed entry for random_model, the random
    side of both panels is built from every seed instead of one: the
    histogram pools every seed's per-token fractions together, and the
    trajectory line becomes the across-seed mean with a shaded ±1 std
    band. Falls back to the single discovered run for random_model when no
    aggregate entry exists, so this still works without random_aggregate.py.
    """
    plt.rcParams.update(BLOG_STYLE)
    key_t, key_r = (trained_model, prompt), (random_model, prompt)
    if key_t not in runs:
        print(f"  ⚠  noise_persistence_pair: missing run for {trained_model!r} @ {prompt!r}")
        return

    agg_entry = random_agg.get((random_model, prompt)) if random_agg else None
    has_random_run = key_r in runs

    frac_t = _token_in_cluster_fraction(runs[key_t])
    mem_t  = _cluster_membership_series(runs[key_t])

    pooled = agg_entry.get("token_in_cluster_fraction_pooled") if agg_entry else None
    if pooled:
        frac_r = np.asarray(pooled, dtype=float)
    elif has_random_run:
        frac_r = _token_in_cluster_fraction(runs[key_r])
    else:
        frac_r = None

    mem_entry = agg_entry.get("cluster_membership") if agg_entry else None
    if mem_entry and mem_entry.get("mean"):
        mem_r      = np.asarray(mem_entry["mean"], dtype=float)
        mem_r_std  = np.asarray(mem_entry["std"], dtype=float) if mem_entry.get("std") else None
        n_seeds_r  = agg_entry.get("n_runs", 1)
    elif has_random_run:
        mem_r      = _cluster_membership_series(runs[key_r])
        mem_r      = np.asarray(mem_r, dtype=float) if mem_r else None
        mem_r_std  = None
        n_seeds_r  = 1
    else:
        mem_r, mem_r_std, n_seeds_r = None, None, 0

    if frac_t is None or frac_r is None or mem_t is None or mem_r is None:
        print(f"  ⚠  noise_persistence_pair: insufficient data for "
              f"{trained_model} vs {random_model}")
        return
    mem_t = np.asarray(mem_t, dtype=float)

    fig, (ax_hist, ax_traj) = plt.subplots(1, 2, figsize=(13, 5.2))
    t_color = _color(trained_model)
    r_color = _random_color(random_model)
    random_label = f"random ({_label(random_model)})" + (
        f", n={n_seeds_r} seeds pooled" if n_seeds_r > 1 else ""
    )

    bins = np.linspace(0, 1, 21)
    ax_hist.hist(frac_t, bins=bins, density=True, alpha=0.55,
                 color=t_color, label=f"trained ({_label(trained_model)})")
    ax_hist.hist(frac_r, bins=bins, density=True, alpha=0.55,
                 color=r_color, label=random_label)
    ax_hist.axvline(frac_t.mean(), color=t_color, linestyle="--", linewidth=1.2)
    ax_hist.axvline(frac_r.mean(), color=r_color, linestyle="--", linewidth=1.2)
    ax_hist.set_xlabel("Fraction of layers a token spends in a real cluster")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Per-token clustered-time distribution", fontsize=11, fontweight="bold")
    ax_hist.legend(fontsize=8)

    n_t, n_r = len(mem_t), len(mem_r)
    x_t, x_r = np.linspace(0, 1, n_t), np.linspace(0, 1, n_r)
    ax_traj.plot(x_t, mem_t, color=t_color, linewidth=2.2, label="trained")
    ax_traj.plot(x_r, mem_r, color=r_color, linewidth=2.0,
                 linestyle="--", label=random_label)
    if mem_r_std is not None and mem_r_std.size == mem_r.size:
        ax_traj.fill_between(
            x_r, mem_r - mem_r_std, mem_r + mem_r_std,
            color=r_color, alpha=0.18, zorder=1, linewidth=0,
        )
    common_x = np.linspace(0, 1, max(n_t, n_r))
    yt = np.interp(common_x, x_t, mem_t)
    yr = np.interp(common_x, x_r, mem_r)
    ax_traj.fill_between(common_x, yt, yr, color="#A7F3D0", alpha=0.4,
                          where=(yt >= yr), interpolate=True, label="trained > random")
    ax_traj.fill_between(common_x, yt, yr, color="#FCA5A5", alpha=0.4,
                          where=(yt < yr), interpolate=True, label="trained < random")
    ax_traj.set_ylim(-0.02, 1.06)
    ax_traj.set_xlabel("Normalized layer depth")
    ax_traj.set_ylabel("Fraction of tokens in a cluster")
    ax_traj.set_title("Cluster membership vs. depth", fontsize=11, fontweight="bold")
    ax_traj.legend(fontsize=7, loc="lower right")

    seed_note = f" (random: n={n_seeds_r} seeds, mean ± std)" if n_seeds_r > 1 else ""
    fig.suptitle(
        f"Clustered vs. noise — {trained_model} vs {random_model} | {prompt}{seed_note}\n"
        f"final layer: trained {mem_t[-1]*100:.0f}% clustered, "
        f"random {mem_r[-1]*100:.0f}% clustered   ·   "
        f"per-token mean time-in-cluster: trained {frac_t.mean()*100:.0f}%, "
        f"random {frac_r.mean()*100:.0f}%",
        fontsize=12, fontweight="bold",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (f"noise_persistence_{_safe_model_name(trained_model)}"
              f"_vs_{_safe_model_name(random_model)}_{prompt}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


def _shared_row_ylim(
    *series_lists: Optional[List[float]], pad: float = 0.06,
) -> Optional[Tuple[float, float]]:
    """
    Common y-limits across every series passed in (NaNs dropped), so a row
    of side-by-side panels reads off the same scale left to right instead
    of each column silently auto-scaling to its own data.
    """
    vals = [v for s in series_lists if s for v in s if v is not None and v == v]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


CKA_YLIM: Tuple[float, float] = (0.9, 1.0)


def _trailing_undefined_layer(series: np.ndarray) -> Optional[int]:
    """
    Index of the last layer with a defined (non-NaN) CKA value, if the
    series goes permanently NaN from some point on and never recovers.

    cka_prev is computed in analysis.py only while effective_rank stays
    at or above DEGENERATE_RANK_THRESHOLD; once a model collapses below
    that, cka_prev is NaN for every layer after, by construction — not a
    plotting bug, but a line that just stops drawing looks like one.
    Returns None if the series ends on a defined value (nothing to flag).
    """
    n = len(series)
    if n == 0 or not np.isnan(series[-1]):
        return None
    for i in range(n - 1, -1, -1):
        if not np.isnan(series[i]):
            return i
    return None


def plot_membership_decoupling(
    runs: dict, out_dir: Path, prompt: str, trained_model: str, random_model: str,
    random_agg: Optional[dict] = None,
) -> None:
    """
    Three stacked rows (HDBSCAN k + noise fraction, CKA vs. previous layer,
    mean Fiedler value), one column per model — for checking whether
    "high/stable CKA + volatile k + flat Fiedler" is architecture-level
    (shows up under random weights too) or training-induced, and whether
    it lines up with the same layers in both columns.

    Each row shares one y-axis scale across both columns (k row and
    Fiedler row are matched here; the CKA row is fixed to CKA_YLIM =
    [0.9, 1.0]), so scanning left-to-right compares the actual values,
    not two independently auto-scaled axes that happen to look similarly
    tall. Once a column's effective rank collapses below
    DEGENERATE_RANK_THRESHOLD, cka_prev goes permanently NaN for the rest
    of that run — the CKA panel shades that tail gray rather than letting
    the line silently stop.

    When random_agg has a multi-seed entry for random_model, every row in
    the random column is the across-seed mean with a shaded ±1 std band
    instead of one seed's raw series.
    """
    plt.rcParams.update(BLOG_STYLE)
    cols = [trained_model, random_model]
    keys = [(m, prompt) for m in cols]
    if keys[0] not in runs:
        print(f"  ⚠  membership_decoupling: missing run for {trained_model!r} @ {prompt!r}")
        return

    agg_entry = random_agg.get((random_model, prompt)) if random_agg else None
    if keys[1] not in runs and not agg_entry:
        print(f"  ⚠  membership_decoupling: missing run for {random_model!r} @ {prompt!r}")
        return

    def _pick(model: str, agg_key: str, value_fn, run_dir: Optional[Path]):
        """(mean, std) for one metric/model — std is None off the fast path."""
        if model == random_model and agg_entry and agg_key in agg_entry and agg_entry[agg_key].get("mean"):
            d = agg_entry[agg_key]
            mean = np.asarray(d["mean"], dtype=float)
            std  = np.asarray(d["std"], dtype=float) if d.get("std") else None
            return mean, std
        if run_dir is None:
            return None, None
        vals = value_fn(run_dir)
        return (np.asarray(vals, dtype=float) if vals else None), None

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex="col")

    # Pull every series first so row-shared y-limits can be fixed before
    # any plotting/styling happens.
    per_col = []
    for model in cols:
        run_dir = runs.get((model, prompt))
        k_mean,   k_std   = _pick(model, "cluster_count",      _cluster_count_series,     run_dir)
        mem_mean, mem_std = _pick(model, "cluster_membership", _cluster_membership_series, run_dir)
        cka_mean, cka_std = _pick(model, "cka_prev",            _cka_prev_series,           run_dir)
        fie_mean, fie_std = _pick(model, "fiedler_mean",        _fiedler_mean_series,       run_dir)
        n_seeds = agg_entry.get("n_runs", 1) if (model == random_model and agg_entry) else 1
        per_col.append({
            "k": k_mean, "k_std": k_std,
            "mem": mem_mean, "mem_std": mem_std,
            "cka": cka_mean, "cka_std": cka_std,
            "fie": fie_mean, "fie_std": fie_std,
            "plateau": _trajectory(run_dir).get("plateau_layers", []) if run_dir else [],
            "color":   _random_color(model) if _is_untrained(model) else _color(model),
            "n_seeds": n_seeds,
        })

    k_ylim   = _shared_row_ylim(*[c["k"].tolist() if c["k"] is not None else None for c in per_col])
    fie_ylim = _shared_row_ylim(*[c["fie"].tolist() if c["fie"] is not None else None for c in per_col])

    for col, (model, data) in enumerate(zip(cols, per_col)):
        k_series, k_std   = data["k"], data["k_std"]
        mem, mem_std      = data["mem"], data["mem_std"]
        cka_series, cka_std = data["cka"], data["cka_std"]
        fie_series, fie_std = data["fie"], data["fie_std"]
        noise_series = (1.0 - mem) if mem is not None else None
        plateau, line_color = data["plateau"], data["color"]
        seed_suffix = f"\n(n={data['n_seeds']} seeds, mean ± std)" if data["n_seeds"] > 1 else ""

        ax_k = axes[0, col]
        if k_series is not None:
            xs = np.arange(len(k_series))
            _shade_plateaus(ax_k, plateau)
            ax_k.plot(xs, k_series, color=line_color, linewidth=2.0,
                      marker="o", markersize=2.5, zorder=3)
            if k_std is not None and k_std.size == k_series.size:
                ax_k.fill_between(xs, k_series - k_std, k_series + k_std,
                                  color=line_color, alpha=0.18, zorder=1, linewidth=0)
        if k_ylim:
            ax_k.set_ylim(*k_ylim)
        ax_k.set_ylabel("HDBSCAN k" if col == 0 else "")
        if noise_series is not None and k_series is not None:
            ax_k2 = ax_k.twinx()
            xs_n = np.arange(len(noise_series))
            ax_k2.plot(xs_n, noise_series,
                       color="#9333EA", linewidth=1.4, linestyle=":", alpha=0.85, zorder=2)
            if mem_std is not None and mem_std.size == noise_series.size:
                ax_k2.fill_between(xs_n, noise_series - mem_std, noise_series + mem_std,
                                   color="#9333EA", alpha=0.12, zorder=1, linewidth=0)
            ax_k2.set_ylim(-0.02, 1.06)
            if col == 1:
                ax_k2.set_ylabel("noise fraction", color="#9333EA", fontsize=9)
            ax_k2.tick_params(axis="y", colors="#9333EA", labelsize=7)
        ax_k.set_title(_label(model) + seed_suffix, fontsize=10, fontweight="bold")

        ax_c = axes[1, col]
        if cka_series is not None:
            xs = np.arange(len(cka_series))
            _shade_plateaus(ax_c, plateau)
            ax_c.plot(xs, cka_series, color=line_color, linewidth=2.0, zorder=3)
            if cka_std is not None and cka_std.size == cka_series.size:
                ax_c.fill_between(xs, cka_series - cka_std, cka_series + cka_std,
                                   color=line_color, alpha=0.18, zorder=1, linewidth=0)
            cutoff = _trailing_undefined_layer(cka_series)
            if cutoff is not None:
                ax_c.axvspan(cutoff + 0.5, len(cka_series) - 1 + 0.5,
                             color="#E5E7EB", alpha=0.7, zorder=0)
                ax_c.annotate(
                    "rank collapsed —\nCKA undefined",
                    xy=((cutoff + 0.5 + len(cka_series) - 1) / 2, CKA_YLIM[0] + 0.015),
                    ha="center", va="bottom", fontsize=6.5, style="italic", color="#6B7280",
                )
        ax_c.set_ylim(*CKA_YLIM)
        ax_c.set_ylabel("CKA(layer, layer−1)" if col == 0 else "")

        ax_f = axes[2, col]
        if fie_series is not None:
            xs = np.arange(len(fie_series))
            _shade_plateaus(ax_f, plateau)
            ax_f.plot(xs, fie_series, color=line_color, linewidth=2.0, zorder=3)
            if fie_std is not None and fie_std.size == fie_series.size:
                ax_f.fill_between(xs, fie_series - fie_std, fie_series + fie_std,
                                   color=line_color, alpha=0.18, zorder=1, linewidth=0)
        if fie_ylim:
            ax_f.set_ylim(*fie_ylim)
        ax_f.set_ylabel("Fiedler value (mean)" if col == 0 else "")
        ax_f.set_xlabel("Layer")

    fig.suptitle(
        f"Cluster-count volatility vs. CKA / Fiedler stability — "
        f"{trained_model} vs {random_model} | {prompt}\n"
        f"dotted purple = noise fraction (right axis, top row); "
        f"yellow band = plateau window; CKA row fixed to [{CKA_YLIM[0]:.2g}, {CKA_YLIM[1]:.2g}]; "
        f"gray band = rank collapsed, CKA undefined; "
        f"k/Fiedler rows share one y-scale across both columns",
        fontsize=12, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (f"membership_decoupling_{_safe_model_name(trained_model)}"
              f"_vs_{_safe_model_name(random_model)}_{prompt}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


def _plot_depth_metric_pairs_merged(
    runs: dict, out_dir: Path, prompt: str, pairs: List[Tuple[str, str]],
    metric_fn, metric_key: str, ylabel: str, metric_label: str, fname_stem: str,
    random_agg: Optional[dict] = None, ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    One merged chart, every (trained, random) pair overlaid on the same
    axes — trained solid in the model's own color, random dashed (gray by
    default, see RANDOM_COLOR_OVERRIDES), ± std band when random_agg has
    a multi-seed entry for that random_model. Used for both the Fiedler
    and CKA "vs. depth" charts so e.g. gpt2-large, its random control,
    albert-base-v2, and its random control all read off one figure
    instead of one chart per architecture.

    `ylim`, if given, fixes the y-axis range instead of letting it
    auto-scale to the plotted data.
    """
    plt.rcParams.update(BLOG_STYLE)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    plotted = False

    for trained_model, random_model in pairs:
        key_t = (trained_model, prompt)
        if key_t not in runs:
            print(f"  ⚠  {fname_stem}: missing run for {trained_model!r} @ {prompt!r}")
            continue
        series_t = metric_fn(runs[key_t])
        if not series_t:
            print(f"  ⚠  {fname_stem}: no {metric_key} series for {trained_model!r} @ {prompt!r}")
            continue
        series_t = np.asarray(series_t, dtype=float)

        series_r, std_r, n_seeds_r = _series_or_aggregate(
            random_model, prompt, runs.get((random_model, prompt)), metric_fn,
            metric_key, random_agg,
        )
        if series_r is None:
            print(f"  ⚠  {fname_stem}: missing run for {random_model!r} @ {prompt!r}")
            continue

        t_color, r_color = _color(trained_model), _random_color(random_model)
        x_t, x_r = np.linspace(0, 1, len(series_t)), np.linspace(0, 1, len(series_r))
        random_label = f"random ({_label(random_model)})" + (
            f", n={n_seeds_r} seeds" if n_seeds_r > 1 else ""
        )

        ax.plot(x_t, series_t, color=t_color, linewidth=2.2, zorder=3,
                label=f"trained ({_label(trained_model)})")
        ax.plot(x_r, series_r, color=r_color, linewidth=2.0, linestyle="--", zorder=3,
                label=random_label)
        if std_r is not None and std_r.size == series_r.size:
            ax.fill_between(x_r, series_r - std_r, series_r + std_r,
                             color=r_color, alpha=0.18, zorder=1, linewidth=0)
        plotted = True

    if not plotted:
        plt.close(fig)
        print(f"  ⚠  {fname_stem}: nothing to plot for prompt {prompt!r}")
        return

    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(
        f"{metric_label} vs. depth — trained vs. random | {prompt}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="best")

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{fname_stem}_{prompt}.png"
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


def plot_fiedler_merged(
    runs: dict, out_dir: Path, prompt: str, pairs: List[Tuple[str, str]],
    random_agg: Optional[dict] = None,
) -> None:
    """Mean Fiedler value vs. depth — every trained/random pair on one chart."""
    _plot_depth_metric_pairs_merged(
        runs, out_dir, prompt, pairs, _fiedler_mean_series, "fiedler_mean",
        "Fiedler value (mean across heads)", "Fiedler value",
        "fiedler_trajectory_merged", random_agg=random_agg,
    )


def plot_cka_merged(
    runs: dict, out_dir: Path, prompt: str, pairs: List[Tuple[str, str]],
    random_agg: Optional[dict] = None,
) -> None:
    """CKA(layer, layer−1) vs. depth — every trained/random pair on one chart."""
    _plot_depth_metric_pairs_merged(
        runs, out_dir, prompt, pairs, _cka_prev_series, "cka_prev",
        "CKA(layer, layer−1)", "CKA vs. previous layer",
        "cka_trajectory_merged", random_agg=random_agg, ylim=(0.90, 1.01),
    )


def generate_noise_comparison_figures(
    runs: dict, out_dir: Path, prompt: str, random_agg: Optional[dict] = None,
) -> None:
    """Every trained/random pair figure, for every matched pair at `prompt`."""
    models = sorted({m for (m, p) in runs.keys() if p == prompt})
    if random_agg:
        models = sorted(set(models) | {m for (m, p) in random_agg.keys() if p == prompt})
    pairs = _trained_random_pairs(models)
    if not pairs:
        print(f"  ⚠  no trained/random pairs found for prompt {prompt!r}")
        return
    for trained_model, random_model in pairs:
        plot_noise_persistence_pair(runs, out_dir, prompt, trained_model, random_model, random_agg=random_agg)
        plot_membership_decoupling(runs, out_dir, prompt, trained_model, random_model, random_agg=random_agg)
        plot_energy_trajectory_pair(runs, out_dir, prompt, trained_model, random_model)
    plot_fiedler_merged(runs, out_dir, prompt, pairs, random_agg=random_agg)
    plot_cka_merged(runs, out_dir, prompt, pairs, random_agg=random_agg)



# ─────────────────────────────────────────────────────────────────────────────
# Energy trajectory — trained vs. random, one panel per β
# ─────────────────────────────────────────────────────────────────────────────

def plot_energy_trajectory_pair(
    runs: dict, out_dir: Path, prompt: str, trained_model: str, random_model: str,
) -> None:
    """
    Interaction energy E_β vs. layer, one panel per β found in
    energies.json — trained and random plotted together on each panel
    (trained solid, random dashed) instead of as two separate per-model
    figures, so the same β panel shows directly whether the dip Theorem
    3.4 predicts away is training-induced or already present at
    initialization.
    """
    plt.rcParams.update(BLOG_STYLE)
    key_t, key_r = (trained_model, prompt), (random_model, prompt)
    if key_t not in runs:
        print(f"  ⚠  energy_trajectory_pair: missing run for {trained_model!r} @ {prompt!r}")
        return
    run_t, run_r = runs[key_t], runs.get(key_r)
    if run_r is None:
        print(f"  ⚠  energy_trajectory_pair: missing run for {random_model!r} @ {prompt!r}")

    betas = _available_betas(run_t)
    if not betas:
        print(f"  ⚠  energy_trajectory_pair: no energies.json data for {trained_model}/{prompt}")
        return

    plateau_layers = _trajectory(run_t).get("plateau_layers", [])
    t_color, r_color = _color(trained_model), _random_color(random_model)

    fig, axes = plt.subplots(1, len(betas), figsize=(4.3 * len(betas), 4.6))
    if len(betas) == 1:
        axes = [axes]

    for ax, beta in zip(axes, betas):
        _shade_plateaus(ax, plateau_layers)

        series_t = _energy_series(run_t, beta)
        if series_t is not None:
            series_t = np.asarray(series_t, dtype=float)
            ax.plot(np.arange(len(series_t)), series_t, color=t_color, linewidth=2.2,
                     zorder=3, label=f"trained ({_label(trained_model)})")

        series_r = _energy_series(run_r, beta) if run_r is not None else None
        if series_r is not None:
            series_r = np.asarray(series_r, dtype=float)
            ax.plot(np.arange(len(series_r)), series_r, color=r_color, linewidth=2.0,
                     linestyle="--", zorder=3, label=f"random ({_label(random_model)})")

        ax.set_title(f"β = {beta:g}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer")
        if ax is axes[0]:
            ax.set_ylabel("Interaction energy  E_β")
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Interaction energy vs. depth — {trained_model} vs {random_model} | {prompt}\n"
        f"Theorem 3.4 predicts monotone increase  ·  yellow band = plateau window",
        fontsize=12, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (f"energy_trajectory_{_safe_model_name(trained_model)}"
              f"_vs_{_safe_model_name(random_model)}_{prompt}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")
