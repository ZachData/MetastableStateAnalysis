"""
visualization/ip_population_dynamics.py

Two figures aimed at a specific question: when the global pairwise-IP
histogram changes shape across depth (broadens and shifts right, then
narrows and shifts left into a right-skewed tail), is that being driven by
the clustered population tightening/loosening, or by the ~40-50% of
tokens HDBSCAN calls noise drifting on their own — and how does either
story square with interaction energy staying flat?

plot_ip_histogram_depth_heatmap
    Full-depth view of geometry.json's per-layer ip_histogram (already
    computed, no activations reload needed) as a layer x bin heatmap,
    with the per-layer mean and mode traced on top, HDBSCAN cluster
    count / noise fraction below it, and E_beta below that. Replaces
    eyeballing 8 sparse snapshot panels (plot_ip_histograms) with the
    continuous picture.

plot_ip_population_trajectory
    Same x-axis, but splits every layer's pairs into within-cluster /
    between-cluster / noise-involving (same partition
    plot_within_between_ip_histogram uses at 3 reference layers) and
    tracks each population's mean IP and mass>0.9 across all layers,
    plus each population's share of total pairs. Needs activations.npz
    and hdbscan_labels.json (not free, unlike the histogram heatmap).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .style import BLOG_STYLE
from .naming import _safe_model_name
from .loaders import (
    _geo, _clustering, _energy_series, _hdbscan_labels, _load_activations,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Full-depth IP histogram heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_ip_histogram_depth_heatmap(
    run_dir: Path, out_dir: Path, beta: float = 1.0,
) -> None:
    plt.rcParams.update(BLOG_STYLE)
    geo = _geo(run_dir)
    model, prompt, n_layers = geo["model"], geo["prompt"], geo["n_layers"]
    layers = geo["layers"]

    hist = np.array([lr.get("ip_histogram", []) for lr in layers], dtype=np.float64)
    if hist.size == 0 or hist.shape[1] == 0:
        print(f"  [skip] no ip_histogram stored for {model}/{prompt}")
        return

    n_bins = hist.shape[1]
    edges = np.linspace(-1, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    # Row-normalize to a density (each layer's histogram sums to 1) so layers
    # are comparable regardless of total pair count, which is fixed per run
    # anyway but this keeps the colorbar meaningful if min_size filtering
    # ever changes that.
    row_sums = hist.sum(axis=1, keepdims=True)
    density = np.divide(hist, row_sums, out=np.zeros_like(hist), where=row_sums > 0)

    mean_ip = np.array([lr.get("ip_mean", np.nan) for lr in layers])
    mass_near_1 = np.array([lr.get("ip_mass_near_1", np.nan) for lr in layers])
    mode_ip = np.array([
        centers[int(np.argmax(row))] if row.sum() > 0 else np.nan
        for row in hist
    ])

    clustering = _clustering(run_dir)
    clayers = clustering.get("layers", [])
    n_clusters = np.full(n_layers, np.nan)
    noise_frac = np.full(n_layers, np.nan)
    for lr in clayers:
        L = lr["layer"]
        if L < n_layers:
            hdb = lr.get("clustering", {}).get("hdbscan", {})
            n_clusters[L] = hdb.get("n_clusters", np.nan) or np.nan
            noise_frac[L] = hdb.get("noise_fraction", np.nan)

    energy = _energy_series(run_dir, beta)
    energy = np.array(energy) if energy is not None else np.full(n_layers, np.nan)

    x = np.arange(n_layers)
    fig, axes = plt.subplots(
        4, 1, figsize=(11, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1.2]},
    )

    # Panel 1 — heatmap
    ax = axes[0]
    norm = mcolors.LogNorm(vmin=max(density[density > 0].min(), 1e-6), vmax=density.max()) \
        if (density > 0).any() else None
    im = ax.pcolormesh(
        x, centers, density.T, shading="nearest", cmap="magma", norm=norm,
    )
    ax.plot(x, mean_ip, color="cyan", linewidth=1.5, label="mean ⟨xᵢ,xⱼ⟩")
    ax.plot(x, mode_ip, color="white", linewidth=1.2, linestyle="--", label="mode (histogram peak)")
    ax.axhline(0.9, color="lime", linewidth=0.8, linestyle=":", label="0.9 threshold")
    ax.set_ylabel("⟨xᵢ, xⱼ⟩")
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=7, loc="upper right")
    fig.colorbar(im, ax=ax, label="density (log scale)", pad=0.01)
    ax.set_title(
        "Full pairwise-IP histogram across depth — log-density, row-normalized per layer",
        fontsize=10,
    )

    # Panel 2 — mass>0.9 and noise fraction
    ax = axes[1]
    ax.plot(x, mass_near_1, color="firebrick", linewidth=1.6, label="mass(IP>0.9)")
    ax.set_ylabel("mass > 0.9", color="firebrick")
    ax.tick_params(axis="y", colors="firebrick")
    ax2 = ax.twinx()
    ax2.plot(x, noise_frac, color="#4B5563", linewidth=1.4, linestyle="--",
             label="HDBSCAN noise fraction")
    ax2.set_ylabel("noise fraction", color="#4B5563")
    ax2.tick_params(axis="y", colors="#4B5563")
    ax.set_title("Tail mass above 0.9 vs. fraction of tokens unclustered", fontsize=9)

    # Panel 3 — cluster count
    ax = axes[2]
    ax.plot(x, n_clusters, color="navy", linewidth=1.6)
    ax.set_ylabel("HDBSCAN k")
    ax.set_title("HDBSCAN cluster count", fontsize=9)

    # Panel 4 — energy
    ax = axes[3]
    ax.plot(x, energy, color="darkorange", linewidth=1.6, label=f"E(β={beta})")
    ax.set_ylabel(f"E(β={beta})")
    ax.set_xlabel("Layer")
    ax.set_title("Interaction energy — flat or decreasing contradicts the paper's monotonicity prediction", fontsize=9)

    fig.suptitle(f"IP histogram dynamics vs. cluster count and energy — {model} | {prompt}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ip_histogram_depth_{_safe_model_name(model)}_{prompt}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Within / between / noise population trajectory
# ─────────────────────────────────────────────────────────────────────────────

def _population_stats(vals: np.ndarray) -> dict:
    if vals.size == 0:
        return dict(mean=np.nan, mass_near_1=np.nan)
    return dict(mean=float(vals.mean()), mass_near_1=float((vals > 0.9).mean()))


def plot_ip_population_trajectory(
    run_dir: Path, out_dir: Path, beta: float = 1.0,
) -> None:
    plt.rcParams.update(BLOG_STYLE)
    geo = _geo(run_dir)
    model, prompt, n_layers = geo["model"], geo["prompt"], geo["n_layers"]
    hdb = _hdbscan_labels(run_dir)
    activations = _load_activations(run_dir)

    if not hdb or activations is None:
        print(f"  [skip] missing hdbscan labels / activations for {model}/{prompt}")
        return

    energy = _energy_series(run_dir, beta)
    energy = np.array(energy) if energy is not None else np.full(n_layers, np.nan)

    pops = ["within", "between", "noise_clustered", "noise_noise"]
    mean_by_pop = {p: np.full(n_layers, np.nan) for p in pops}
    mass_by_pop = {p: np.full(n_layers, np.nan) for p in pops}
    share_by_pop = {p: np.full(n_layers, np.nan) for p in pops}
    global_mean = np.full(n_layers, np.nan)

    for layer in range(min(n_layers, activations.shape[0])):
        labels = np.array(hdb.get(layer, []))
        if labels.size == 0:
            continue
        acts = activations[layer]
        n = len(labels)
        gram = acts @ acts.T
        iu, ju = np.triu_indices(n, k=1)
        li, lj = labels[iu], labels[ju]
        vals = gram[iu, ju]
        global_mean[layer] = float(vals.mean())

        both_clustered = (li != -1) & (lj != -1)
        both_noise = (li == -1) & (lj == -1)
        # "noisy" as a single bucket (li != -1 & lj != -1 is False) conflates
        # two different things: a pair where ONE side is unclustered and the
        # other is a real cluster member (mixed), vs. a pair where BOTH sides
        # are unclustered (pure). A token sitting just outside a cluster
        # boundary shows up identically to one nowhere near any cluster under
        # the single-bucket definition. Split them.
        groups = {
            "within": vals[both_clustered & (li == lj)],
            "between": vals[both_clustered & (li != lj)],
            "noise_clustered": vals[(~both_clustered) & (~both_noise)],
            "noise_noise": vals[both_noise],
        }
        total = max(len(vals), 1)
        for p, g in groups.items():
            stats = _population_stats(g)
            mean_by_pop[p][layer] = stats["mean"]
            mass_by_pop[p][layer] = stats["mass_near_1"]
            share_by_pop[p][layer] = len(g) / total

    x = np.arange(n_layers)
    colors = {
        "within": "#DC2626", "between": "#2563EB",
        "noise_clustered": "#D97706", "noise_noise": "#6B7280",
    }

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 1.4]})

    ax = axes[0]
    for p in pops:
        ax.plot(x, mean_by_pop[p], color=colors[p], linewidth=1.8, label=f"{p}-pair mean IP")
    ax.plot(x, global_mean, color="black", linewidth=1.2, linestyle=":", label="global mean IP")
    ax2 = ax.twinx()
    ax2.plot(x, energy, color="darkorange", linewidth=1.4, linestyle="--", label=f"E(β={beta})")
    ax2.set_ylabel(f"E(β={beta})", color="darkorange")
    ax2.tick_params(axis="y", colors="darkorange")
    ax.set_ylabel("mean ⟨xᵢ,xⱼ⟩")
    ax.set_title("Mean inner product by pair population, vs. energy", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")

    ax = axes[1]
    for p in pops:
        ax.plot(x, mass_by_pop[p], color=colors[p], linewidth=1.8, label=f"{p}-pair mass(IP>0.9)")
    ax.set_ylabel("mass(IP>0.9)")
    ax.set_title("Mass above 0.9, by pair population", fontsize=9)
    ax.legend(fontsize=7, loc="best")

    ax = axes[2]
    ax.stackplot(x, [share_by_pop[p] for p in pops],
                 colors=[colors[p] for p in pops], labels=[f"{p} pairs" for p in pops],
                 alpha=0.85)
    ax.set_ylabel("share of all pairs")
    ax.set_xlabel("Layer")
    ax.set_ylim(0, 1)
    ax.set_title("Population sizes — context for how much each line above can move the global mean", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(f"Within / between / noise IP decomposition — {model} | {prompt}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ip_population_{_safe_model_name(model)}_{prompt}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")
