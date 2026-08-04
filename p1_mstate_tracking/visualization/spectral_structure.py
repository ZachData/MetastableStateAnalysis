"""
visualization/spectral_structure.py

The "spectral_*" figure set — per-model. Where method_comparison.py asks
whether the methods agree, this asks what the geometry supports before any
clustering algorithm is applied at all.

Three figures:
  eigenvalue_ladder   the normalized-Laplacian spectrum of the token Gram
                      matrix across depth, with the selected eigengap
                      marked — how many clusters the spectrum allows,
                      independent of any assignment
  fiedler             the k=2 bipartition: PCA colored by sign, and the
                      raw eigenvector distribution underneath, which is
                      what decides whether the split is two lobes or one
                      lobe cut near zero
  nesting             per-HDBSCAN-cluster internal spectral k — the
                      hierarchy that multiscale_nesting computes and only
                      reporting_p1 has ever printed

One naming caution. The Fiedler quantities here come from the *token Gram
matrix* (spectral.json, fiedler_vecs.npz). Every other "fiedler" in this
package — series._fiedler_mean_series, the pair_comparisons trajectory,
checkpoint_scalars.min_fiedler — is sinkhorn.json's fiedler_mean, a
property of the attention graph. Same name, different object, and they
are not expected to track each other.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

from core.style import BLOG_STYLE, LayerSpec, MIN_CLUSTER_SIZE
from core.naming import _safe_model_name
from .loaders import (
    _geo, _clustering, _spectral, _trajectory, _pca_trajs, _fiedler_vecs,
)
from core.plot_utils import _resolve_layers, _shade_plateaus, _annotation_box
from .cluster_methods import bipartition_separation
from .method_comparison import _aligned_colorbar

POS_COLOR = "#2563EB"
NEG_COLOR = "#DC2626"


def _plateaus(run_dir: Path) -> List[int]:
    return _trajectory(run_dir).get("plateau_layers", []) or []


def _finish(fig, out_dir: Path, fname: str, note: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}{'  ' + note if note else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Eigenvalue ladder
# ─────────────────────────────────────────────────────────────────────────────

def plot_eigenvalue_ladder(
    run_dir: Path, out_dir: Path, n_eig: int = 12,
) -> None:
    """
    The first n_eig eigenvalues of the normalized Laplacian, per layer.

    Read it as: eigenvalues near zero are connected components the
    geometry actually separates. The number of near-zero eigenvalues below
    the largest gap is the cluster count the spectrum supports — computed
    without assigning a single token, so it is not downstream of any
    algorithm's parameters. The white markers trace k_eigengap; the bottom
    panel shows the size of that gap, which is what makes the selection
    trustworthy or not. A k with a gap barely above its neighbours is a
    tie broken by noise.
    """
    plt.rcParams.update(BLOG_STYLE)
    geo = _geo(run_dir)
    model, prompt = geo["model"], geo["prompt"]
    layers_raw = _spectral(run_dir).get("layers", [])
    if not layers_raw:
        print(f"  [skip] no spectral.json for {model}/{prompt}")
        return

    layers = [int(lr["layer"]) for lr in layers_raw]
    grid = np.full((n_eig, len(layers)), np.nan)
    k_sel = np.full(len(layers), np.nan)
    gap_at_k = np.full(len(layers), np.nan)
    gap_margin = np.full(len(layers), np.nan)

    for col, lr in enumerate(layers_raw):
        ev = np.asarray(lr.get("eigenvalues", []), dtype=float)
        if ev.size:
            grid[: min(n_eig, ev.size), col] = ev[:n_eig]
        gaps = np.asarray(lr.get("eigengaps", []), dtype=float)
        k = lr.get("k_eigengap")
        if k is not None:
            k_sel[col] = float(k)
        if gaps.size:
            gi = int(k) - 1 if k is not None and 0 < int(k) <= gaps.size else int(np.argmax(gaps))
            gap_at_k[col] = gaps[gi]
            rest = np.delete(gaps, gi)
            if rest.size:
                gap_margin[col] = gaps[gi] - rest.max()

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.2], "hspace": 0.12},
    )

    im = ax_top.imshow(
        grid, aspect="auto", origin="lower", cmap="viridis",
        extent=[layers[0] - 0.5, layers[-1] + 0.5, 0.5, n_eig + 0.5],
    )
    ax_top.plot(layers, k_sel, color="white", linewidth=1.6, marker="o",
                markersize=4, markeredgecolor="black", markeredgewidth=0.5,
                label="k at largest eigengap")
    ax_top.set_ylabel("Eigenvalue index")
    ax_top.set_ylim(0.5, n_eig + 0.5)
    ax_top.grid(False)
    ax_top.legend(fontsize=8, loc="upper right")
    _aligned_colorbar(fig, im, ax_top, ax_bot, "λ (normalized Laplacian)")

    ax_bot.plot(layers, gap_at_k, color="#7C3AED", linewidth=2.0, label="gap at selected k")
    ax_bot.plot(layers, gap_margin, color="#9CA3AF", linewidth=1.5, linestyle="--",
                label="margin over the runner-up gap")
    ax_bot.axhline(0.0, color="#DC2626", linewidth=1.0)
    _shade_plateaus(ax_bot, _plateaus(run_dir), alpha=0.16)
    ax_bot.set_xlabel("Layer")
    ax_bot.set_ylabel("Eigengap", fontsize=9)
    ax_bot.legend(fontsize=8, loc="upper right")
    _annotation_box(
        ax_bot, "margin near 0 = k was a coin flip between two gaps",
        xy=(0.015, 0.08), xycoords="axes fraction", fontsize=8,
    )

    ax_top.set_title(
        f"How many clusters does the spectrum support? — {model} | {prompt}\n"
        f"dark band at the bottom = eigenvalues the geometry separates",
        fontsize=12, fontweight="bold",
    )
    _finish(fig, out_dir,
            f"spectral_eigenvalue_ladder_{_safe_model_name(model)}_{prompt}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fiedler bipartition
# ─────────────────────────────────────────────────────────────────────────────

def plot_fiedler_bipartition(
    run_dir: Path, out_dir: Path, layers: Optional[List[LayerSpec]] = None,
) -> None:
    """
    The k=2 partition, shown twice per layer: as a coloring of the PCA
    projection, and as the distribution of the raw Fiedler values the sign
    cut was taken from.

    The scatter always looks like two groups — sign always produces two
    groups. The histogram is the actual test. Two separated lobes mean a
    real bipartition; a single mode straddling zero means the cut is
    assigning near-zero tokens essentially at random, and the reported
    "two hemispheres" is an artifact of taking a sign. The printed
    statistics quantify exactly that: separation in pooled standard
    deviations, how balanced the two sides are, and what fraction of
    tokens sit close enough to zero that the assignment is arbitrary.
    """
    plt.rcParams.update(BLOG_STYLE)
    geo = _geo(run_dir)
    model, prompt, n_layers = geo["model"], geo["prompt"], geo["n_layers"]
    fvecs = _fiedler_vecs(run_dir)
    if not fvecs:
        print(f"  [skip] no fiedler_vecs.npz for {model}/{prompt}")
        return
    pca = _pca_trajs(run_dir)

    resolved = [l for l in _resolve_layers(layers, n_layers) if l in fvecs]
    if not resolved:
        resolved = sorted(fvecs.keys())[-3:]

    fig, axes = plt.subplots(
        2, len(resolved), figsize=(4.4 * len(resolved), 8.2), squeeze=False,
        gridspec_kw={"height_ratios": [1.5, 1.0]},
    )

    for c, layer in enumerate(resolved):
        v = np.asarray(fvecs[layer], dtype=float)
        stats = bipartition_separation(v)
        pos = v > 0

        ax_s = axes[0][c]
        proj = pca.get(layer)
        if proj is not None and proj.shape[0] == v.size:
            ax_s.scatter(proj[pos, 0], proj[pos, 1], s=26, color=POS_COLOR,
                         edgecolors="white", linewidths=0.5, zorder=3)
            ax_s.scatter(proj[~pos, 0], proj[~pos, 1], s=26, color=NEG_COLOR,
                         edgecolors="white", linewidths=0.5, zorder=3)
            ax_s.set_xticks([])
            ax_s.set_yticks([])
        else:
            ax_s.axis("off")
        ax_s.set_title(
            f"layer {layer}\nbalance {stats['balance']:.2f}, "
            f"separation {stats['separation']:.2f}σ",
            fontsize=10,
        )

        ax_h = axes[1][c]
        bins = np.linspace(float(v.min()), float(v.max()), 41)
        ax_h.hist(v[pos], bins=bins, color=POS_COLOR, alpha=0.7)
        ax_h.hist(v[~pos], bins=bins, color=NEG_COLOR, alpha=0.7)
        ax_h.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        vmax = np.abs(v).max()
        if vmax > 0:
            ax_h.axvspan(-0.1 * vmax, 0.1 * vmax, color="#9CA3AF", alpha=0.25, zorder=0)
        ax_h.set_xlabel("Fiedler value", fontsize=9)
        if c == 0:
            ax_h.set_ylabel("Tokens", fontsize=9)
        nz = stats["near_zero"]
        ax_h.set_title(
            f"{nz:.0%} of tokens within 10% of zero" if np.isfinite(nz) else "",
            fontsize=8,
        )

    fig.suptitle(
        f"Fiedler bipartition — {model} | {prompt}\n"
        f"the scatter always splits; the histogram is what says whether the split is real",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _finish(fig, out_dir,
            f"spectral_fiedler_bipartition_{_safe_model_name(model)}_{prompt}.png",
            f"(layers={resolved})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-scale nesting
# ─────────────────────────────────────────────────────────────────────────────

def plot_nesting_overview(
    run_dir: Path, out_dir: Path, min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> None:
    """
    Hierarchy: a global bipartition at the top, HDBSCAN's density clusters
    beneath it, and whatever spectral structure survives inside each of
    those.

    Left panel — every (HDBSCAN cluster, layer) pair as a point: size
    against its own internal spectral k. Points above k=1 are clusters
    that are themselves splittable, which is the nesting claim. Left of
    the size gate, internal k is unreliable and drawn faint.

    Right panel — against depth: the global spectral k, and how many
    clusters carry substructure. Nesting is the regime where the global k
    is small (a macro-bipartition) while local clusters remain internally
    structured; the two lines crossing marks where that stops holding.
    """
    plt.rcParams.update(BLOG_STYLE)
    geo = _geo(run_dir)
    model, prompt = geo["model"], geo["prompt"]
    layers_raw = _clustering(run_dir).get("layers", [])
    if not layers_raw:
        print(f"  [skip] no clustering.json for {model}/{prompt}")
        return

    sizes, sub_k, layer_of = [], [], []
    layers, global_k, n_sub, n_tot = [], [], [], []

    for lr in layers_raw:
        nest = lr.get("nesting", {}) or {}
        li = int(lr["layer"])
        layers.append(li)
        gk = nest.get("global_spectral_k")
        global_k.append(float(gk) if gk is not None else np.nan)
        n_sub.append(float(nest.get("n_clusters_with_substructure", 0)))
        per_cluster = nest.get("per_cluster", {}) or {}
        n_tot.append(float(len(per_cluster)))
        for info in per_cluster.values():
            if not isinstance(info, dict):
                continue
            sizes.append(float(info.get("n_tokens", 0)))
            sub_k.append(float(info.get("spectral_k", 1)))
            layer_of.append(li)

    if not sizes:
        print(f"  [skip] no nesting records for {model}/{prompt}")
        return

    sizes = np.asarray(sizes)
    sub_k = np.asarray(sub_k)
    layer_of = np.asarray(layer_of, dtype=float)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.4))

    big = sizes >= min_cluster_size
    jitter = np.random.default_rng(42).normal(0, 0.06, size=sub_k.size)
    sc = ax_l.scatter(
        sizes[big], sub_k[big] + jitter[big], c=layer_of[big],
        cmap="plasma", s=42, edgecolors="white", linewidths=0.5, zorder=3,
    )
    ax_l.scatter(
        sizes[~big], sub_k[~big] + jitter[~big], c="#D1D5DB", s=20,
        alpha=0.6, zorder=2, label=f"cluster < {min_cluster_size} tokens",
    )
    ax_l.axhline(1.5, color="#DC2626", linestyle="--", linewidth=1.2)
    ax_l.axvline(min_cluster_size, color="#9CA3AF", linestyle=":", linewidth=1.0)
    ax_l.set_xscale("log")
    ax_l.set_xlabel("Cluster size (tokens)")
    ax_l.set_ylabel("Internal spectral k")
    ax_l.set_title("Which clusters are themselves splittable?", fontsize=11)
    ax_l.legend(fontsize=8, loc="upper left")
    fig.colorbar(sc, ax=ax_l, shrink=0.85, label="layer")

    _shade_plateaus(ax_r, _plateaus(run_dir), alpha=0.16)
    ax_r.plot(layers, global_k, color="#7C3AED", linewidth=2.2, marker="o",
              markersize=3.4, markevery=max(1, len(layers) // 24),
              label="global spectral k")
    ax_r.plot(layers, n_sub, color="#059669", linewidth=2.0,
              label="HDBSCAN clusters with substructure")
    ax_r.plot(layers, n_tot, color="#9CA3AF", linewidth=1.4, linestyle="--",
              label="HDBSCAN clusters total")
    ax_r.set_xlabel("Layer")
    ax_r.set_ylabel("Count")
    ax_r.set_title("Global bipartition vs. local substructure", fontsize=11)
    ax_r.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Multi-scale cluster nesting — {model} | {prompt}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _finish(fig, out_dir,
            f"spectral_nesting_{_safe_model_name(model)}_{prompt}.png")
