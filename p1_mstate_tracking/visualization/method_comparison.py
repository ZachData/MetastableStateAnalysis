"""
visualization/method_comparison.py

The "cluster_methods_*" figure set — per-model, written to each model's
own folder. Everything here asks one question in different ways: how much
of the cluster structure Phase 1 reports is a property of the token
geometry, and how much is a property of HDBSCAN.

Six figures:
  count_agreement   every per-layer cluster-count estimate on one axis,
                    with the +/-1 agreement band shaded
  threshold_sweep   the full agglomerative distance-threshold sweep as a
                    heatmap, plus the scale-plateau width per layer
  ari               pairwise ARI between the four persisted partitions
  consensus         co-association heatmap, sorted onto the diagonal
  scatter_grid      the same PCA projection colored four ways, one row
                    per method
  noise_audit       what the other methods do with HDBSCAN's noise tokens

All six read only artifacts run_1 already writes. Nothing here needs a
rerun; nothing here recomputes a clustering.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from core.style import BLOG_STYLE, LayerSpec, MIN_CLUSTER_SIZE, NOISE_COLOR
from core.naming import _safe_model_name
from .loaders import (
    _geo, _trajectory, _pca_trajs, _agglom_threshold_counts,
)
from core.plot_utils import _resolve_layers, _shade_plateaus, _scatter_hdbscan, _annotation_box
from .cluster_methods import (
    METHOD_ORDER, METHOD_LABELS, METHOD_COLORS,
    method_labels, common_layers, cluster_count_table,
    agreement_trajectory, co_association, consensus_order, consensus_strength,
    plateau_widths, noise_audit,
)

# Series drawn by the count-agreement figure, in legend order. Fiedler is
# absent on purpose — it is k=2 by construction.
_COUNT_SERIES = (
    ("hdbscan",       "HDBSCAN"),
    ("agglomerative", "Agglomerative @ mid threshold"),
    ("kmeans",        "KMeans (silhouette k)"),
    ("spectral_k",    "Spectral eigengap k"),
    ("sinkhorn_k",    "Sinkhorn (attention graph)"),
)


def _plateaus(run_dir: Path) -> List[int]:
    return _trajectory(run_dir).get("plateau_layers", []) or []


def _header(run_dir: Path):
    geo = _geo(run_dir)
    return geo["model"], geo["prompt"], geo["n_layers"]


def _finish(fig, out_dir: Path, fname: str, note: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}{'  ' + note if note else ''}")


def _aligned_colorbar(fig, im, ax_with, ax_without, label: str):
    """
    Colorbar for the upper panel of a shared-x stacked figure, without
    shrinking that panel relative to the lower one.

    fig.colorbar(ax=ax_top) steals width from the top axes only, which
    silently misaligns the two x-axes — the shared tick positions stop
    lining up with the bars underneath, and every reading off the figure
    is off by the colorbar's width. Carving an equal-width slot out of
    both panels and hiding the lower one keeps them registered.

    Shared with spectral_structure.py, which builds the same two-panel
    layout.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cax = make_axes_locatable(ax_with).append_axes("right", size="2%", pad=0.12)
    spacer = make_axes_locatable(ax_without).append_axes("right", size="2%", pad=0.12)
    spacer.axis("off")
    return fig.colorbar(im, cax=cax, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cluster-count agreement across methods
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_count_agreement(run_dir: Path, out_dir: Path) -> None:
    """
    Five cluster-count estimates against depth, on one axis.

    If the count is a property of the geometry, the curves track each
    other. If they diverge, "k clusters at layer L" is a statement about
    the algorithm. KMeans is drawn hollow at layers where its silhouette
    or the effective rank fails the trust gate — at those depths best_k=2
    is the floor of K_RANGE rather than a finding, and it is excluded from
    the agreement band for the same reason.
    """
    plt.rcParams.update(BLOG_STYLE)
    model, prompt, _ = _header(run_dir)
    table = cluster_count_table(run_dir)
    layers = table["layers"]
    if not layers:
        print(f"  [skip] no clustering.json layers for {model}/{prompt}")
        return

    counts = table["counts"]
    trusted = table["kmeans_trusted"]
    x = np.asarray(layers, dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5.6))
    _shade_plateaus(ax, _plateaus(run_dir), alpha=0.16)

    # Agreement band: vertical stripes where every trusted series is within +/-1.
    agree = np.asarray(table["agreement"])
    for li, ok in zip(layers, agree):
        if ok:
            ax.axvspan(li - 0.5, li + 0.5, color="#86EFAC", alpha=0.30, zorder=0)

    for key, label in _COUNT_SERIES:
        y = counts.get(key)
        if y is None or not np.isfinite(y).any():
            continue
        color = METHOD_COLORS.get(key, "#374151")
        if key == "kmeans":
            ax.plot(x, y, color=color, linewidth=1.6, alpha=0.75, zorder=3)
            ax.scatter(x[trusted], y[trusted], s=34, color=color,
                       edgecolors="white", linewidths=0.6, zorder=4)
            ax.scatter(x[~trusted], y[~trusted], s=30, facecolors="none",
                       edgecolors=color, linewidths=1.1, zorder=4)
        else:
            ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=3.4,
                    markevery=max(1, len(x) // 24), label=label, zorder=3)

    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=METHOD_COLORS["kmeans"], marker="o",
                          markersize=5, linewidth=1.6,
                          label="KMeans (filled = passes trust gate)"))
    handles.append(Line2D([], [], color="#86EFAC", linewidth=8, alpha=0.6,
                          label="all trusted estimates within ±1"))

    ax.set_xlabel("Layer")
    ax.set_ylabel("Estimated cluster count k")
    mid_t = table["mid_threshold"]
    thresh_txt = f"{mid_t:.3f}" if mid_t is not None else "n/a"
    ax.set_title(
        f"Cluster count is method-dependent — {model} | {prompt}\n"
        f"agglomerative threshold = {thresh_txt}; "
        f"yellow = plateau window, green = cross-method agreement",
        fontsize=12, fontweight="bold",
    )
    # Below the axes rather than inside: the agglomerative series routinely
    # runs an order of magnitude above the others, so an in-axes legend
    # lands on top of exactly the divergence the figure exists to show.
    ax.legend(handles=handles, fontsize=8, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.tight_layout()
    _finish(fig, out_dir,
            f"cluster_methods_count_agreement_{_safe_model_name(model)}_{prompt}.png",
            f"({int(agree.sum())}/{len(layers)} layers agree)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Agglomerative threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def plot_agglomerative_threshold_heatmap(run_dir: Path, out_dir: Path) -> None:
    """
    The full cosine-distance sweep, which HDBSCAN structurally cannot show
    because it picks the scale for you.

    Top panel: cluster count as a function of (threshold, layer). Bottom:
    the longest run of consecutive thresholds returning the same count.
    A wide plateau means the partition survives a range of cut distances —
    genuine scale separation. Width 1 everywhere means the reported k is
    an artifact of where the cut was placed.
    """
    plt.rcParams.update(BLOG_STYLE)
    model, prompt, _ = _header(run_dir)
    thresholds, layers, counts = _agglom_threshold_counts(run_dir)
    if counts.size == 0:
        print(f"  [skip] no agglomerative sweep for {model}/{prompt}")
        return

    widths = plateau_widths(counts)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.15], "hspace": 0.12},
    )

    finite = counts[np.isfinite(counts)]
    vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
    im = ax_top.imshow(
        counts, aspect="auto", origin="lower", cmap="viridis",
        vmin=1, vmax=max(vmax, 2),
        extent=[layers[0] - 0.5, layers[-1] + 0.5, -0.5, len(thresholds) - 0.5],
    )
    ax_top.set_yticks(range(len(thresholds)))
    ax_top.set_yticklabels([f"{t:.2f}" for t in thresholds], fontsize=7)
    ax_top.set_ylabel("Cosine-distance threshold")
    ax_top.grid(False)

    mid_row = len(thresholds) // 2
    ax_top.axhline(mid_row, color="white", linestyle="--", linewidth=1.4, alpha=0.9)
    ax_top.text(
        layers[0], mid_row + 0.15, " mid threshold (the one whose labels are saved)",
        color="white", fontsize=8, va="bottom",
    )
    _aligned_colorbar(fig, im, ax_top, ax_bot, "cluster count k")

    ax_bot.bar(layers, widths, color="#059669", alpha=0.85, width=0.85)
    ax_bot.axhline(1, color="#DC2626", linestyle="--", linewidth=1.0)
    ax_bot.set_ylabel("Plateau width\n(thresholds)", fontsize=9)
    ax_bot.set_xlabel("Layer")
    _shade_plateaus(ax_bot, _plateaus(run_dir), alpha=0.16)
    _annotation_box(
        ax_bot,
        "width 1 (red line) = k depends entirely on the cut distance",
        xy=(0.015, 0.82), xycoords="axes fraction", fontsize=8,
    )

    ax_top.set_title(
        f"Agglomerative cluster count across every cut distance — {model} | {prompt}\n"
        f"flat vertical bands = the partition survives a range of scales",
        fontsize=12, fontweight="bold",
    )
    _finish(fig, out_dir,
            f"cluster_methods_threshold_sweep_{_safe_model_name(model)}_{prompt}.png",
            f"(max plateau width {int(widths.max()) if widths.size else 0})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pairwise ARI trajectory
# ─────────────────────────────────────────────────────────────────────────────

def plot_method_ari_trajectory(
    run_dir: Path, out_dir: Path, noise_policy: str = "singleton", min_used: int = 8,
) -> None:
    """
    Adjusted Rand Index between every pair of persisted partitions, by
    depth. Two panels, one per noise policy, because they answer different
    questions and can disagree sharply:

      left   noise as singletons — do the methods agree about all tokens?
      right  noise excluded — do they agree about the tokens HDBSCAN was
             willing to commit on?

    Under "exclude", layers where fewer than min_used tokens survive are
    masked: an ARI of 1.0 over four tokens is not evidence of anything.
    """
    plt.rcParams.update(BLOG_STYLE)
    model, prompt, _ = _header(run_dir)
    per_method = method_labels(run_dir)
    if len(per_method) < 2:
        print(f"  [skip] fewer than two label families for {model}/{prompt}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    policies = [
        ("singleton", "noise → singletons\n(agreement over all tokens)"),
        ("exclude",   "noise excluded\n(agreement over HDBSCAN-assigned tokens)"),
    ]

    for ax, (policy, subtitle) in zip(axes, policies):
        layers, scores, counts = agreement_trajectory(per_method, noise_policy=policy)
        if not layers:
            ax.set_title("(no shared layers)", fontsize=10)
            ax.axis("off")
            continue
        x = np.asarray(layers, dtype=float)
        _shade_plateaus(ax, _plateaus(run_dir), alpha=0.16)

        stacked = []
        for (m1, m2), s in scores.items():
            y = s.copy()
            if policy == "exclude":
                y[counts[(m1, m2)] < min_used] = np.nan
            stacked.append(y)
            ax.plot(x, y, linewidth=1.5, alpha=0.75,
                    label=f"{m1[:5]} ↔ {m2[:5]}")
        if stacked:
            mean_y = np.nanmean(np.vstack(stacked), axis=0)
            ax.plot(x, mean_y, color="#111827", linewidth=2.6, zorder=5,
                    label="mean over pairs")

        ax.axhline(0.0, color="#9CA3AF", linewidth=1.0)
        ax.set_ylim(-0.15, 1.05)
        ax.set_xlabel("Layer")
        ax.set_title(subtitle, fontsize=10)
        ax.legend(fontsize=7, ncol=2, loc="lower left")

    axes[0].set_ylabel("Adjusted Rand Index")
    fig.suptitle(
        f"Do the clustering methods agree? — {model} | {prompt}\n"
        f"ARI 1 = identical partitions, 0 = no better than chance",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _finish(fig, out_dir,
            f"cluster_methods_ari_{_safe_model_name(model)}_{prompt}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Consensus co-association matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_consensus_matrix(
    run_dir: Path, out_dir: Path, layers: Optional[List[LayerSpec]] = None,
) -> None:
    """
    Fraction of methods placing each token pair together, sorted onto the
    diagonal by average linkage.

    This is the method-independent cluster definition. Block structure
    here can't be attributed to any one algorithm's inductive bias, which
    makes it a stronger claim than the HDBSCAN-sorted Gram heatmap in
    cluster_reality.py — that figure sorts by the very labels whose
    reality is in question, and will show blocks even when the underlying
    partition is arbitrary.
    """
    plt.rcParams.update(BLOG_STYLE)
    model, prompt, n_layers = _header(run_dir)
    per_method = method_labels(run_dir)
    if len(per_method) < 2:
        print(f"  [skip] fewer than two label families for {model}/{prompt}")
        return
    shared = common_layers(per_method)
    if not shared:
        print(f"  [skip] no shared layers for {model}/{prompt}")
        return

    resolved = [l for l in _resolve_layers(layers, n_layers) if l in shared]
    if not resolved:
        resolved = [shared[len(shared) // 2], shared[-1]]

    present = [m for m in METHOD_ORDER if m in per_method]
    fig, axes = plt.subplots(1, len(resolved), figsize=(5.6 * len(resolved), 5.6))
    axes = np.atleast_1d(axes)
    im = None

    for ax, layer in zip(axes, resolved):
        arrays = [per_method[m][layer] for m in present]
        C = co_association(arrays, noise_policy="singleton")
        order = consensus_order(C)
        im = ax.imshow(C[order][:, order], cmap="magma", vmin=0, vmax=1, aspect="equal")
        ax.set_title(
            f"layer {layer}\nunanimous on {consensus_strength(C):.0%} of pairs",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    fig.suptitle(
        f"Consensus across {len(present)} clustering methods — {model} | {prompt}\n"
        f"entry = fraction of methods putting the pair together",
        fontsize=12, fontweight="bold",
    )
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.75, label="co-association")
    _finish(fig, out_dir,
            f"cluster_methods_consensus_{_safe_model_name(model)}_{prompt}.png",
            f"(layers={resolved})")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Same projection, four colorings
# ─────────────────────────────────────────────────────────────────────────────

def plot_method_scatter_grid(
    run_dir: Path, out_dir: Path, layers: Optional[List[LayerSpec]] = None,
) -> None:
    """
    One cached PCA projection per layer, colored by each method in turn:
    rows are methods, columns are depths.

    The point of holding the projection fixed is that any visible
    difference between rows is a difference in the partition, not in the
    embedding. Where the rows look alike, the clusters are real; where
    HDBSCAN shows a field of gray crosses and KMeans shows confident
    blocks over the same points, the disagreement is the finding.
    """
    plt.rcParams.update(BLOG_STYLE)
    model, prompt, n_layers = _header(run_dir)
    per_method = method_labels(run_dir)
    if not per_method:
        print(f"  [skip] no cluster labels for {model}/{prompt}")
        return
    pca = _pca_trajs(run_dir)
    if not pca:
        print(f"  [skip] no PCA trajectories for {model}/{prompt}")
        return

    shared = common_layers(per_method)
    resolved = [l for l in _resolve_layers(layers, n_layers) if l in pca]
    if shared:
        resolved = [l for l in resolved if l in shared] or shared[-3:]
    if not resolved:
        print(f"  [skip] no usable layers for {model}/{prompt}")
        return

    present = [m for m in METHOD_ORDER if m in per_method]
    fig, axes = plt.subplots(
        len(present), len(resolved),
        figsize=(3.6 * len(resolved), 3.6 * len(present)),
        squeeze=False,
    )

    for r, method in enumerate(present):
        for c, layer in enumerate(resolved):
            ax = axes[r][c]
            proj = pca.get(layer)
            labels = per_method[method].get(layer)
            if proj is None or labels is None or labels.size != proj.shape[0]:
                ax.axis("off")
                continue
            k = _scatter_hdbscan(ax, proj, labels, s_cluster=26, s_noise=16)
            n_noise = int((labels == -1).sum())
            note = f"k={k}" + (f", {n_noise} noise" if n_noise else "")
            ax.set_title(note, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(METHOD_LABELS[method], fontsize=9)
            if r == 0:
                ax.text(0.5, 1.16, f"layer {layer}", transform=ax.transAxes,
                        ha="center", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"One projection, {len(present)} partitions — {model} | {prompt}\n"
        f"rows differ only in how the same points were labeled",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _finish(fig, out_dir,
            f"cluster_methods_scatter_grid_{_safe_model_name(model)}_{prompt}.png",
            f"(layers={resolved})")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Noise audit
# ─────────────────────────────────────────────────────────────────────────────

def plot_noise_label_audit(
    run_dir: Path, out_dir: Path, min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> None:
    """
    What the other methods do with the tokens HDBSCAN refuses to assign.

    Three series per alternative method, against depth:
      noise fraction        how much of the token set is at stake
      rescued               fraction of noise tokens the other method puts
                            in a cluster of at least min_cluster_size
      rescued into shared   of those, the fraction landing in a cluster
                            that also holds HDBSCAN-assigned tokens

    The gap between the last two is the informative quantity. High rescue
    with low shared means the noise tokens form their own coherent group
    that HDBSCAN's density criterion split off — a different claim from
    "noise is unstructured", and directly relevant to what p5c_unclustered
    is treating as unclustered.
    """
    plt.rcParams.update(BLOG_STYLE)
    model, prompt, _ = _header(run_dir)
    per_method = method_labels(run_dir)
    if "hdbscan" not in per_method:
        print(f"  [skip] no HDBSCAN labels for {model}/{prompt}")
        return
    others = [m for m in METHOD_ORDER if m in per_method and m != "hdbscan"]
    if not others:
        print(f"  [skip] no alternative partitions for {model}/{prompt}")
        return

    layers = common_layers(per_method)
    if not layers:
        print(f"  [skip] no shared layers for {model}/{prompt}")
        return
    x = np.asarray(layers, dtype=float)

    noise_frac = np.array([
        float((per_method["hdbscan"][li] == -1).mean()) for li in layers
    ])

    fig, ax = plt.subplots(figsize=(11, 5.6))
    _shade_plateaus(ax, _plateaus(run_dir), alpha=0.16)

    ax.fill_between(x, 0, noise_frac, color=NOISE_COLOR, alpha=0.55, zorder=1,
                    label="HDBSCAN noise fraction")

    for method in others:
        rescued = np.full(len(layers), np.nan)
        shared = np.full(len(layers), np.nan)
        for row, li in enumerate(layers):
            stats = noise_audit(
                per_method["hdbscan"][li], per_method[method][li],
                min_cluster_size=min_cluster_size,
            )
            rescued[row] = stats["rescued_fraction"]
            shared[row] = stats["rescued_into_shared"]
        color = METHOD_COLORS[method]
        ax.plot(x, rescued, color=color, linewidth=2.0, zorder=3,
                label=f"{method}: noise placed in a cluster ≥ {min_cluster_size}")
        ax.plot(x, shared, color=color, linewidth=1.6, linestyle="--", alpha=0.85,
                zorder=3, label=f"{method}: …and shared with assigned tokens")

    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction")
    ax.set_title(
        f"Is HDBSCAN noise a category or a refusal? — {model} | {prompt}\n"
        f"solid − dashed = noise tokens pooled into their own group",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.tight_layout()
    _finish(fig, out_dir,
            f"cluster_methods_noise_audit_{_safe_model_name(model)}_{prompt}.png")
