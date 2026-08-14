"""
p1b_hemisphere/visualization/regime.py — Block 0, what the bipartition is.

Seven per-run figures (R1-R7 in FIGURES-1b.md). The organizing constraint is
status-1b R1: "0% strong bipartition" and "100% cone-collapse" were reported
as two independent findings when they are close to the same test, because
`strong_bipartition` requires a centroid angle of at least pi/2 and two
centroids inside one open half-space essentially cannot be that far apart.

So every figure here is built to make the near-unreachability visible rather
than to restate the verdict:

  * R1 draws both classifiers as parallel bands — the antipodal one can be
    uniformly `collapsed` while the relative one finds `separated`.
  * R3 draws the centroid angle against the pi/2 bar it has to clear, so a
    reader sees how far away it is instead of reading a 0%.
  * R2 draws `separation_ratio` with the two inner products it is built from,
    since the ratio alone can move because either term moved.

Thresholds come from `bipartition_detect.REGIME_THRESHOLDS`, imported, never
retyped — a threshold change must move the lines in these figures or the
figures are lying.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from p1b_hemisphere.bipartition_detect import REGIME_THRESHOLDS

from .loaders import Run, layer_pair_field
from .style import (
    BLOG_STYLE, CATEGORICAL, INVALID_COLOR, REGIME_COLORS, REGIME_ORDER,
    REGIME_REL_COLORS, REGIME_REL_ORDER, class_strip, depth_axis,
    legend_from_classes, reference_line, save_figure,
)

__all__ = ["generate_regime_figures"]


def _separation_thresholds(ax) -> None:
    """
    The two cut points `classify_regime_relative` actually uses.

    Both, not just one: `relative_separation` divides `separated` from
    `graded` and `relative_weak` divides `graded` from `uniform`, and a
    figure showing only the first makes a `graded` layer look like a near
    miss on the only line drawn.
    """
    sep = REGIME_THRESHOLDS.get("relative_separation")
    weak = REGIME_THRESHOLDS.get("relative_weak")
    if sep is not None:
        reference_line(ax, float(sep), f"separated at ≤ {float(sep):g}",
                       side="left")
    if weak is not None:
        reference_line(ax, float(weak), f"uniform at ≥ {float(weak):g}",
                       side="right")
    ax.margins(y=0.14)


def generate_regime_figures(run: Run, out_dir: Path) -> List[Path]:
    """Every Block 0 figure for one run."""
    with plt.rc_context(BLOG_STYLE):
        return [
            _regime_strip(run, out_dir),
            _bipartition_quality(run, out_dir),
            _centroid_angle(run, out_dir),
            _eigengap_depth(run, out_dir),
            _hemisphere_balance(run, out_dir),
            _boundary_fraction(run, out_dir),
            _asymmetry_depth(run, out_dir),
        ]


# ---------------------------------------------------------------------------
# R1 — the two classifiers, side by side
# ---------------------------------------------------------------------------

def _regime_strip(run: Run, out_dir: Path) -> Path:
    """
    Both regime vocabularies as parallel per-layer bands.

    This is status-1b R1 as one image. The two rows are the same layers under
    two classifiers; if the top row is uniformly `collapsed` while the bottom
    shows `separated`, the phase's headline null is a property of the
    antipodal rule rather than of the geometry.
    """
    antipodal = run.strings("regime")
    relative = run.strings("regime_relative")
    n = run.n_layers

    fig, axes = plt.subplots(3, 1, figsize=(11, 4.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.9]})

    class_strip(axes[0], antipodal, REGIME_COLORS, label="antipodal\n(`regime`)")
    class_strip(axes[1], relative, REGIME_REL_COLORS,
                label="relative\n(`regime_relative`)")

    # The continuous quantity the relative classifier actually thresholds,
    # underneath both labels — the label is the report, this is the evidence.
    sep = run.field("separation_ratio")
    ax = axes[2]
    ax.plot(range(n), sep, color=CATEGORICAL[0], linewidth=2.0,
            marker="o", markersize=3.5, label="separation ratio")
    _separation_thresholds(ax)
    depth_axis(ax, n)
    ax.set_ylabel("between / within IP")
    ax.legend(loc="lower left")

    handles_ax = axes[0]
    legend_from_classes(handles_ax, list(REGIME_ORDER) + ["invalid"],
                        REGIME_COLORS, loc="upper left",
                        bbox_to_anchor=(1.005, 1.9), fontsize=8)
    legend_from_classes(axes[1], list(REGIME_REL_ORDER) + ["invalid"],
                        REGIME_REL_COLORS, loc="upper left",
                        bbox_to_anchor=(1.005, 1.25), fontsize=8)

    strong = sum(1 for r in antipodal if r == "strong_bipartition")
    seps = sum(1 for r in relative if r == "separated")
    fig.suptitle(
        f"{run.label} — two classifiers on the same layers\n"
        f"strong bipartition {strong}/{n} layers · separated {seps}/{n} layers",
        fontsize=12, y=1.04)
    return save_figure(fig, out_dir, "regime_strip")


# ---------------------------------------------------------------------------
# R2 — the ratio and its two components
# ---------------------------------------------------------------------------

def _bipartition_quality(run: Run, out_dir: Path) -> Path:
    """
    `separation_ratio` with the inner products it is a ratio of.

    A ratio moving tells you nothing about which term moved. Between-half IP
    falling and within-half IP rising are different geometries with the same
    ratio, and only one of them is a bipartition getting cleaner.
    """
    n = run.n_layers
    within = layer_pair_field(run.per_layer, "within_half_ip")
    between = run.field("between_half_ip")
    ratio = run.field("separation_ratio")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1]})

    ax = axes[0]
    ax.plot(range(n), within[:, 0], color=CATEGORICAL[0], linewidth=2.0,
            label="within half A")
    ax.plot(range(n), within[:, 1], color=CATEGORICAL[0], linewidth=2.0,
            linestyle="--", label="within half B")
    ax.plot(range(n), between, color=CATEGORICAL[1], linewidth=2.0,
            label="between halves")
    ax.set_ylabel("mean inner product")
    ax.legend(loc="best")
    ax.set_title(f"{run.label} — bipartition quality", fontsize=12)

    ax = axes[1]
    ax.plot(range(n), ratio, color=CATEGORICAL[2], linewidth=2.0,
            marker="o", markersize=3.5)
    _separation_thresholds(ax)
    ax.set_ylabel("separation ratio\n(between / within)")
    depth_axis(ax, n)
    return save_figure(fig, out_dir, "bipartition_quality")


# ---------------------------------------------------------------------------
# R3 — the bar `strong_bipartition` has to clear
# ---------------------------------------------------------------------------

def _centroid_angle(run: Run, out_dir: Path) -> Path:
    """
    Centroid angle vs depth against pi/2.

    Draws the unreachability instead of asserting it: the shaded region above
    pi/2 is the only place the antipodal classifier can return
    `strong_bipartition`, and under cone-collapse the curve does not enter it.
    An empty region is the result.
    """
    n = run.n_layers
    angle = run.field("centroid_angle")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.axhspan(np.pi / 2, np.pi, color=REGIME_COLORS["strong_bipartition"],
               alpha=0.10, zorder=0, linewidth=0)
    ax.plot(range(n), angle, color=CATEGORICAL[0], linewidth=2.2,
            marker="o", markersize=4)
    reference_line(ax, np.pi / 2, "π/2 — antipodal requirement")

    top = float(np.nanmax(angle)) if np.isfinite(angle).any() else 1.0
    ax.set_ylim(0, max(np.pi * 1.02, top * 1.1))
    ax.set_yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_yticklabels(["0", "π/4", "π/2", "3π/4", "π"])
    ax.set_ylabel("angle between hemisphere centroids")
    depth_axis(ax, n)

    reached = int(np.nansum(angle >= np.pi / 2))
    ax.set_title(
        f"{run.label} — centroid angle vs. the antipodal bar\n"
        f"{reached}/{n} layers reach π/2 "
        f"(max {np.nanmax(angle):.2f} rad)", fontsize=12)
    ax.annotate("`strong_bipartition` is reachable only in this band",
                xy=(0.02, np.pi / 2 + 0.06), xycoords=("axes fraction", "data"),
                fontsize=8.5, color="#6B7280")
    return save_figure(fig, out_dir, "centroid_angle")


# ---------------------------------------------------------------------------
# R4 — how sharply k=2 is preferred
# ---------------------------------------------------------------------------

def _eigengap_depth(run: Run, out_dir: Path) -> Path:
    """`bipartition_eigengap` vs depth, with the regime band beneath."""
    n = run.n_layers
    gap = run.field("bipartition_eigengap")

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True,
                             gridspec_kw={"height_ratios": [4, 1]})
    ax = axes[0]
    ax.plot(range(n), gap, color=CATEGORICAL[0], linewidth=2.2,
            marker="o", markersize=3.5)
    ax.fill_between(range(n), 0, gap, color=CATEGORICAL[0], alpha=0.12)
    ax.set_ylabel("λ₃ − λ₂")
    ax.set_title(f"{run.label} — how sharply k=2 is preferred, by depth",
                 fontsize=12)

    class_strip(axes[1], run.strings("regime_relative"), REGIME_REL_COLORS,
                label="regime")
    depth_axis(axes[1], n)
    return save_figure(fig, out_dir, "eigengap_depth")


# ---------------------------------------------------------------------------
# R5 — who is on each side
# ---------------------------------------------------------------------------

def _hemisphere_balance(run: Run, out_dir: Path) -> Path:
    """
    Hemisphere sizes as a stacked band plus the minority fraction.

    The `collapsed_minority` threshold is drawn because it is the line below
    which both classifiers stop reporting a bipartition at all — a layer just
    under it and a layer just over it are labeled as different regimes while
    being nearly the same geometry.
    """
    n = run.n_layers
    sizes = layer_pair_field(run.per_layer, "hemisphere_sizes")
    minority = run.field("minority_fraction")
    total = np.nansum(sizes, axis=1)
    total[total == 0] = np.nan

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True)

    ax = axes[0]
    frac_a = sizes[:, 0] / total
    ax.fill_between(range(n), 0, frac_a, color=CATEGORICAL[0], alpha=0.85,
                    linewidth=0, label="hemisphere 0")
    ax.fill_between(range(n), frac_a, 1.0, color=CATEGORICAL[1], alpha=0.85,
                    linewidth=0, label="hemisphere 1")
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of tokens")
    ax.legend(loc="upper right", ncol=2)
    ax.set_title(f"{run.label} — hemisphere occupancy by depth", fontsize=12)
    ax.grid(False)

    ax = axes[1]
    ax.plot(range(n), minority, color=CATEGORICAL[2], linewidth=2.2,
            marker="o", markersize=3.5)
    th = REGIME_THRESHOLDS.get("collapsed_minority")
    if th is not None:
        reference_line(ax, float(th),
                       f"`collapsed` below {float(th):g}")
    ax.set_ylabel("minority fraction")
    ax.set_ylim(0, max(0.55, float(np.nanmax(minority)) * 1.15
                       if np.isfinite(minority).any() else 0.55))
    depth_axis(ax, n)
    return save_figure(fig, out_dir, "hemisphere_balance")


# ---------------------------------------------------------------------------
# R6 — the population the sign label means least for
# ---------------------------------------------------------------------------

def _boundary_fraction(run: Run, out_dir: Path) -> Path:
    """
    `fiedler_boundary_frac` vs depth.

    Tokens near zero on the axis get a hemisphere label from an arbitrarily
    small quantity. This is the size of that population, and the reason every
    downstream phase is told to use the axis as a projection rather than the
    sign as a class.
    """
    n = run.n_layers
    frac = run.field("fiedler_boundary_frac")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(range(n), frac, color=CATEGORICAL[0], width=0.78, alpha=0.9)
    ax.set_ylabel("fraction of tokens near the boundary")
    depth_axis(ax, n)
    mean = float(np.nanmean(frac)) if np.isfinite(frac).any() else float("nan")
    reference_line(ax, mean, f"mean {mean:.3f}")
    ax.set_title(
        f"{run.label} — tokens whose hemisphere label is a coin flip\n"
        f"(|v| within the boundary band of zero)", fontsize=12)
    return save_figure(fig, out_dir, "boundary_fraction")


# ---------------------------------------------------------------------------
# R7 — Block 4
# ---------------------------------------------------------------------------

def _asymmetry_depth(run: Run, out_dir: Path) -> Path:
    """
    Block 4's |A−B|/(A+B) vs depth.

    The phase reports a mean of this over `strong_bipartition` layers only.
    When there are none — the expected case — that mean is None, and the
    figure says so rather than showing an unlabeled curve whose summary
    statistic silently does not exist.
    """
    n = run.n_layers
    asym = run.field("asymmetry")
    regime = run.strings("regime")
    strong = [L for L in range(n) if regime[L] == "strong_bipartition"]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(range(n), asym, color=CATEGORICAL[0], linewidth=2.2,
            marker="o", markersize=3.5)
    if strong:
        ax.scatter(strong, asym[strong], s=64, facecolor="none",
                   edgecolor=REGIME_COLORS["strong_bipartition"], linewidth=1.8,
                   zorder=5, label="strong-bipartition layer")
        ax.legend(loc="best")
        note = (f"mean over strong-bipartition layers: "
                f"{np.nanmean(asym[strong]):.3f}")
    else:
        note = ("no strong-bipartition layer in this run — the phase's "
                "`mean_asymmetry_strong` is undefined here, not zero")
    ax.set_ylabel("|A − B| / (A + B)")
    ax.set_ylim(0, 1)
    depth_axis(ax, n)
    ax.set_title(f"{run.label} — hemisphere size asymmetry\n{note}", fontsize=12)
    return save_figure(fig, out_dir, "asymmetry_depth")
