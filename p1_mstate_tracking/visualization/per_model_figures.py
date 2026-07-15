"""
visualization/per_model_figures.py

Per-model PCA/projection figures, generated once per (model, prompt) run:
ip_histogram_migration (the paper Fig 1 replica, now run for every model),
hdbscan_pca (PCA scatter at 4 depths + cluster-count-vs-layer panel), and
projection_comparison (PCA / t-SNE / UMAP side by side at 3 depths).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .style import BLOG_STYLE, LayerSpec, PLATEAU_TINT, PLATEAU_BORDER
from .naming import _color, _safe_model_name
from .loaders import _geo, _clustering, _trajectory, _hdbscan_labels, _pca_trajs, _load_activations
from .plot_utils import _spans, _shade_plateaus, _annotation_box, _scatter_hdbscan, _project_2d, UMAP_AVAILABLE

def plot_ip_histogram_migration(
    run_dir: Path, out_dir: Path, layers: Optional[List[LayerSpec]] = None,
    n_panels: int = 8,
) -> None:
    """
    8-panel inner-product histogram evolution. Each panel shows ⟨xᵢ,xⱼ⟩ at
    one layer. Bars left of 0.9 are light blue; bars at or above 0.9 are
    dark blue (the clustering signal). The red dashed line at 0.9 marks the
    mass-near-1 threshold. (`layers` is unused here — panels are always 8
    evenly spaced depths — kept for interface uniformity with the other
    per-model plot functions.)
    """
    plt.rcParams.update(BLOG_STYLE)
    try:
        geo = _geo(run_dir)
    except Exception as e:
        print(f"  ✗  ip_histogram_migration: {e}")
        return
    model, prompt = geo.get("model", run_dir.name), geo.get("prompt", "")

    layers_data = geo.get("layers", [])
    n_layers    = len(layers_data)
    if n_layers == 0:
        print(f"  ✗  ip_histogram_migration: no layer data for {model}/{prompt}")
        return
    indices = np.linspace(0, n_layers - 1, n_panels, dtype=int)

    bins        = np.linspace(-1, 1, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_w       = bins[1] - bins[0]

    cols = 4
    rows = n_panels // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6.5), sharey=False)
    axes = axes.flatten()
    fig.suptitle(
        f"⟨xᵢ,xⱼ⟩ histogram migration  ·  {model} | {prompt}\n"
        "Each panel: one layer.  Spike moves right as tokens cluster.",
        fontsize=11, fontweight="bold",
    )

    for ax, li in zip(axes, indices):
        lr    = layers_data[li]
        mass  = lr.get("ip_mass_near_1", np.nan)
        hist  = lr.get("ip_histogram", [])
        if not hist:
            ax.text(0.5, 0.5, f"Layer {li}\n(no data)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="#9CA3AF")
            continue

        counts = np.array(hist, dtype=float)
        total  = counts.sum()
        if total > 0:
            counts /= total

        bar_colors = ["#93C5FD" if c < 0.88 else "#1D4ED8"
                      for c in bin_centers]
        ax.bar(bin_centers, counts, width=bin_w,
               color=bar_colors, edgecolor="none", alpha=0.85)
        ax.axvline(0.9, color="#EF4444", linewidth=0.9,
                   linestyle="--", alpha=0.8, label="0.9 threshold")

        mass_str = f"{mass:.2f}" if not np.isnan(mass) else "n/a"
        ax.set_title(f"Layer {li}  (mass>{0.9:.1f} = {mass_str})", fontsize=8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("⟨xᵢ, xⱼ⟩", fontsize=7)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)

    last_ax = axes[len(indices) - 1]
    last_ax.annotate(
        "← spike migrates\n   toward +1",
        xy=(0.65, 0.75), xycoords="axes fraction",
        fontsize=7, color="#1D4ED8", ha="left",
        arrowprops=None,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ip_histogram_migration_{_safe_model_name(model)}_{prompt}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


def plot_hdbscan_pca(
    run_dir: Path, out_dir: Path, layers: Optional[List[LayerSpec]] = None,
) -> None:
    """
    Two-panel layout:
      Top row: PCA scatter at 4 selected layers, colored by HDBSCAN
               assignment. Panels inside a detected plateau window are
               tinted pale yellow with an amber border and bold amber title.
      Bottom:  HDBSCAN cluster count k vs layer with plateau windows shaded.

    NOTE: this panel uses the cached 3-D PCA projection (pca_trajectories.npz).
    If clusters don't separate visibly here, see plot_projection_comparison,
    which recomputes t-SNE/UMAP directly from the raw activations.
    (`layers` is unused — panel depths are derived from the plateau window —
    kept for interface uniformity with the other per-model plot functions.)
    """
    plt.rcParams.update(BLOG_STYLE)
    try:
        geo = _geo(run_dir)
    except Exception as e:
        print(f"  ✗  hdbscan_pca: {e}")
        return
    model, prompt = geo.get("model", run_dir.name), geo.get("prompt", "")

    pca         = _pca_trajs(run_dir)
    hdb_labels  = _hdbscan_labels(run_dir)
    clust_data  = _clustering(run_dir)
    traj_data   = _trajectory(run_dir)

    layers_geo     = geo.get("layers", [])
    n_layers       = len(layers_geo)
    if n_layers == 0:
        print(f"  ✗  hdbscan_pca: no layer data for {model}/{prompt}")
        return
    tokens         = geo.get("tokens", [])
    plateau_layers = traj_data.get("plateau_layers", [])
    plateau_set    = set(plateau_layers)

    clust_layers = clust_data.get("layers", [])
    hdb_k = {}
    for cl in clust_layers:
        k = cl.get("clustering", {}).get("hdbscan", {}).get("n_clusters")
        if k is not None:
            hdb_k[cl["layer"]] = k

    plat_spans = _spans(plateau_layers)
    if plat_spans:
        pre   = max(0, plat_spans[0][0] - 1)
        p_mid = (plat_spans[0][0] + plat_spans[0][1]) // 2
        post  = min(n_layers - 1, plat_spans[-1][1] + 3)
    else:
        pre   = n_layers // 4
        p_mid = n_layers // 2
        post  = n_layers - 1
    selected = [0, pre, p_mid, post]
    panel_labels = [
        "Layer 0\n(embedding)",
        "Pre-plateau",
        "Plateau midpoint",
        "Post-plateau",
    ]

    fig = plt.figure(figsize=(16, 7.5))
    gs  = gridspec.GridSpec(2, 4, height_ratios=[3.2, 1.4],
                            hspace=0.38, wspace=0.30)
    axes_top = [fig.add_subplot(gs[0, c]) for c in range(4)]
    ax_bot   = fig.add_subplot(gs[1, :])
    fig.suptitle(
        f"HDBSCAN cluster structure (PCA)  ·  {model} | {prompt}",
        fontsize=12, fontweight="bold",
    )

    for col, (li, plbl) in enumerate(zip(selected, panel_labels)):
        ax = axes_top[col]
        proj   = pca.get(li)
        labels = hdb_labels.get(li)

        if proj is None or labels is None or len(proj) == 0:
            ax.text(0.5, 0.5, f"Layer {li}\n(no PCA data)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#9CA3AF")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(plbl, fontsize=9)
            continue

        in_pl = li in plateau_set
        n_cl  = _scatter_hdbscan(ax, proj[:, :2], labels)

        for ti, (lb, (x, y)) in enumerate(zip(labels, proj[:, :2])):
            if lb == -1:
                continue
            tok = tokens[ti][:4] if ti < len(tokens) else str(ti)
            ax.annotate(tok, (x, y), fontsize=4, alpha=0.55,
                        ha="center", va="bottom")

        if in_pl:
            ax.set_facecolor(PLATEAU_TINT)
            for spine_name in ("left", "bottom"):
                ax.spines[spine_name].set_color(PLATEAU_BORDER)
                ax.spines[spine_name].set_linewidth(1.8)
            title_color, title_weight, suffix = "#92400E", "bold", "\n(plateau window)"
        else:
            title_color, title_weight, suffix = "black", "normal", ""

        ax.set_title(f"{plbl}\nLayer {li}  k={n_cl}{suffix}", fontsize=8.5,
                     color=title_color, fontweight=title_weight)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("PC1", fontsize=7)
        if col == 0:
            ax.set_ylabel("PC2", fontsize=7)
            _annotation_box(
                ax, "✕ noise (unclustered)\n● cluster member",
                xy=(0.03, 0.04), fontsize=6, xycoords="axes fraction",
            )

    if hdb_k:
        k_series = [hdb_k.get(i, np.nan) for i in range(n_layers)]
        _shade_plateaus(ax_bot, plateau_layers, alpha=0.22)
        ax_bot.plot(range(n_layers), k_series,
                    color=_color(model), linewidth=2.2,
                    marker="o", markersize=3.5, alpha=0.88, zorder=3)
        for li in selected:
            if not np.isnan(k_series[li]):
                ax_bot.axvline(li, color="#6B7280", linewidth=0.8,
                               linestyle=":", alpha=0.7)
        ax_bot.set_xlabel("Layer")
        ax_bot.set_ylabel("HDBSCAN k")
        ax_bot.set_title(
            "Cluster count across layers  (yellow = plateau window, "
            "same shading as the panels above; dotted lines = panels above)",
            fontsize=9,
        )
        ax_bot.set_xlim(-0.5, n_layers - 0.5)
    else:
        ax_bot.text(0.5, 0.5, "No HDBSCAN k data",
                    ha="center", va="center", transform=ax_bot.transAxes,
                    fontsize=10, color="#9CA3AF")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"hdbscan_pca_{_safe_model_name(model)}_{prompt}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    final_mass = layers_geo[-1].get("ip_mass_near_1", float("nan"))
    print(f"  ✓  {path.name}  (final mass-near-1: {final_mass:.2f})")


def plot_projection_comparison(
    run_dir: Path, out_dir: Path, layers: Optional[List[LayerSpec]] = None,
) -> None:
    """
    Diagnostic / decision figure: at 3 depths (pre-plateau, plateau-mid,
    post-plateau), project the SAME token activations with PCA, t-SNE,
    and UMAP (if installed), colored identically by HDBSCAN label using
    the same noise-vs-cluster marker split as plot_hdbscan_pca.

    Projections are recomputed directly from activations.npz — the real
    high-dimensional normed activations — rather than the cached 3-D PCA
    projection used by plot_hdbscan_pca. (`layers` is unused — depths are
    fixed fractions of n_layers — kept for interface uniformity.)
    """
    plt.rcParams.update(BLOG_STYLE)

    try:
        geo = _geo(run_dir)
        model, prompt = geo.get("model", run_dir.name), geo.get("prompt", "")
    except Exception:
        model, prompt = run_dir.name, ""

    activations = _load_activations(run_dir)
    if activations is None:
        print(f"  ✗  projection_comparison: no activations.npz for {model}/{prompt}")
        return

    hdb_labels = _hdbscan_labels(run_dir)
    n_layers   = activations.shape[0]

    norm_depths = [0.07, 0.6, 1.0]
    selected    = [round(d * (n_layers - 1)) for d in norm_depths]
    row_labels  = [f"depth {d:.2f}  (layer {li})"
                   for d, li in zip(norm_depths, selected)]

    methods = ["pca", "tsne"] + (["umap"] if UMAP_AVAILABLE else [])

    fig, axes = plt.subplots(
        len(selected), len(methods),
        figsize=(5.0 * len(methods), 4.3 * len(selected)),
        constrained_layout=True,
    )
    if len(selected) == 1:
        axes = axes[np.newaxis, :]
    if len(methods) == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"Projection comparison  ·  {model} | {prompt}\n"
        "Same HDBSCAN labels, same colors — only the projection changes",
        fontsize=13, fontweight="bold",
    )

    for row, (li, rlabel) in enumerate(zip(selected, row_labels)):
        X      = activations[li]
        labels = hdb_labels.get(li)

        if labels is None or len(labels) != X.shape[0]:
            for col in range(len(methods)):
                ax = axes[row, col]
                ax.text(0.5, 0.5, "no labels", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#9CA3AF")
                ax.set_xticks([]); ax.set_yticks([])
            continue

        for col, method in enumerate(methods):
            ax = axes[row, col]
            try:
                proj = _project_2d(X, method=method, seed=42)
            except Exception as e:
                ax.text(0.5, 0.5, f"{method} failed:\n{e}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=7, color="#EF4444")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            n_cl = _scatter_hdbscan(ax, proj, labels, s_cluster=42, s_noise=20)
            if row == 0:
                ax.set_title(method.upper(), fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{rlabel}\n(k={n_cl})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0 and col == 0:
                _annotation_box(
                    ax, "✕ noise (unclustered)\n● cluster member",
                    xy=(0.03, 0.04), fontsize=7, xycoords="axes fraction",
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"projection_comparison_{_safe_model_name(model)}_{prompt}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    note = "" if UMAP_AVAILABLE else "  (umap-learn not installed — PCA/t-SNE only)"
    print(f"  ✓  {path.name}{note}")


