"""
p1b_hemisphere/visualization/axis.py — Block A, is the axis anything new.

Four figures (A1-A4). Block A maps the token-space Fiedler vector into
activation space and asks whether it is distinguishable from the cloud's
leading variance geometry. The handoff note in status-1b is the reason this
matters more than it looks: on synthetic data the axis is frequently PC1 to
within |cos| >= 0.9, and if that reproduces on real runs then Phase 5's
hemisphere centroids are PC1 under a more expensive name.

Two constraints these figures keep.

**Chance is not zero.** In d dimensions two unrelated unit vectors have
|cos| concentrating around 1/sqrt(d), so a cosine of 0.1 in 64 dimensions is
not "nearly orthogonal", it is chance. A1 draws that floor beside every
curve. When d is not recoverable from the artifacts the floor is omitted
rather than guessed, and the panel says so.

**The mean row is a control, not a finding.** cos(axis, token mean) is
unreachable by construction — the Fiedler vector is orthogonal to the
Laplacian's trivial eigenvector, so X^T f cancels the shared mean component.
Block A's first version tested exactly that and would have repeated the
Block 0 defect. It stays on the figure, labeled as the control it is, because
a control that is never drawn is a control nobody checks.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from p1b_hemisphere.axis_identity import PC1_TOL, PC_BLOCK_TOL

from .loaders import Run, layer_field
from .style import (
    BLOG_STYLE, CATEGORICAL, INVALID_COLOR, REDUNDANCY_COLORS,
    REDUNDANCY_ORDER, class_strip, depth_axis, legend_from_classes,
    model_color, no_data, reference_line, save_figure,
)

__all__ = ["generate_axis_figures", "generate_axis_cross_figures"]


def generate_axis_figures(run: Run, out_dir: Path) -> List[Path]:
    if run.axis_identity is None:
        print(f"    axis: skipped for {run.stem} — axis_identity not in artifacts")
        return []
    with plt.rc_context(BLOG_STYLE):
        paths = [
            _axis_cosines(run, out_dir),
            _redundancy_strip(run, out_dir),
            _pc_subspace_fraction(run, out_dir),
        ]
    return [p for p in paths if p is not None]


def generate_axis_cross_figures(runs: Sequence[Run], out_dir: Path) -> List[Path]:
    """A4 — the one axis figure that pools every layer of every run."""
    with plt.rc_context(BLOG_STYLE):
        p = _axis_vs_pc1_scatter(runs, out_dir)
    return [p] if p is not None else []


# ---------------------------------------------------------------------------
# A1 — every cosine, against chance
# ---------------------------------------------------------------------------

def _axis_cosines(run: Run, out_dir: Path) -> Path:
    """
    |cos| to centered PC1, uncentered PC1, and the token mean, vs depth.

    Centered and uncentered PC1 are both drawn because they are different
    questions on a unit sphere: uncentered PC1 of a cloud inside one open
    hemisphere is largely the mean direction, so an axis that is
    mean-orthogonal by construction can look unrelated to it while being
    exactly centered PC1.
    """
    per_layer = (run.axis_identity or {}).get("per_layer") or []
    n = run.n_layers

    series = [
        ("cos_axis_centered_pc1", "axis · centered PC1", CATEGORICAL[0], "-"),
        ("cos_axis_pc1_uncentered", "axis · uncentered PC1", CATEGORICAL[1], "--"),
        ("cos_mean_pc1", "token mean · PC1  (context)", CATEGORICAL[3], ":"),
        ("cos_axis_mean", "axis · token mean  (control — chance by construction)",
         "#6B7280", "-."),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.0))
    for key, label, color, ls in series:
        vals = layer_field(per_layer, key)
        if not np.isfinite(vals).any():
            continue
        ax.plot(range(len(vals)), vals, color=color, linestyle=ls,
                linewidth=2.2 if ls == "-" else 1.8, marker="o",
                markersize=3.2, label=label)

    reference_line(ax, PC1_TOL, f"|cos| ≥ {PC1_TOL:g} → verdict `pc1`")

    iso = _isotropic_cos(run)
    if iso is not None:
        ax.axhspan(0, iso * 2, color="#9AA0A6", alpha=0.15, linewidth=0,
                   zorder=0)
        reference_line(ax, iso, f"1/√d = {iso:.3f} — chance", side="left")

    ax.set_ylim(0, 1.02)
    ax.set_ylabel("|cos|")
    depth_axis(ax, n)
    ax.legend(loc="best", fontsize=8.5)

    summary = (run.axis_identity or {}).get("summary") or {}
    ax.set_title(
        f"{run.label} — is the Fiedler axis distinguishable from PC1?\n"
        f"modal verdict: {summary.get('modal_redundancy', 'n/a')}", fontsize=12)
    return save_figure(fig, out_dir, "axis_cosines")


# ---------------------------------------------------------------------------
# A2 — the Phase 5 caveat, as a picture
# ---------------------------------------------------------------------------

def _redundancy_strip(run: Run, out_dir: Path) -> Path:
    """
    Per-layer redundancy verdict, with the cosine it thresholds beneath.

    If this band is `pc1` at every layer, Phase 5's hemisphere centroids are
    PC1 wearing a more expensive name and should say so. The continuous
    cosine sits below the band for the usual reason: the verdict is a
    threshold crossing and the reader should see how close the crossing was.
    """
    per_layer = (run.axis_identity or {}).get("per_layer") or []
    verdicts = [str(e.get("redundancy", "degenerate")) for e in per_layer]
    n = run.n_layers

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 3]})

    class_strip(axes[0], verdicts, REDUNDANCY_COLORS, label="verdict")
    legend_from_classes(axes[0], list(REDUNDANCY_ORDER) + ["degenerate"],
                        REDUNDANCY_COLORS, loc="upper left",
                        bbox_to_anchor=(1.005, 1.8), fontsize=8)

    ax = axes[1]
    cos_pc1 = layer_field(per_layer, "cos_axis_pc1")
    ax.plot(range(len(cos_pc1)), cos_pc1, color=CATEGORICAL[0], linewidth=2.2,
            marker="o", markersize=3.5)
    reference_line(ax, PC1_TOL, f"`pc1` at ≥ {PC1_TOL:g}")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("|cos(axis, PC1)|")
    depth_axis(ax, n)

    counts = Counter(verdicts)
    tally = " · ".join(f"{k}: {v}" for k, v in counts.most_common())
    axes[0].set_title(
        f"{run.label} — what the axis is redundant with\n{tally}", fontsize=12)
    return save_figure(fig, out_dir, "redundancy_strip")


# ---------------------------------------------------------------------------
# A3 — redundant with PC1, or with everything
# ---------------------------------------------------------------------------

def _pc_subspace_fraction(run: Run, out_dir: Path) -> Path:
    """
    Fraction of the axis inside the top-k principal subspace, over PC1's
    explained variance.

    These separate two ways of being uninteresting. An axis can be redundant
    because it IS PC1, or because PC1 explains most of the cloud and
    everything is inside the top block. The subspace fraction is also the
    robust statement when leading eigenvalues are close and individual
    components are not identifiable.
    """
    per_layer = (run.axis_identity or {}).get("per_layer") or []
    n = run.n_layers

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)

    ax = axes[0]
    frac = layer_field(per_layer, "pc_subspace_fraction")
    ax.plot(range(len(frac)), frac, color=CATEGORICAL[0], linewidth=2.2,
            marker="o", markersize=3.5)
    ax.fill_between(range(len(frac)), 0, frac, color=CATEGORICAL[0], alpha=0.10)
    reference_line(ax, PC_BLOCK_TOL, f"`top_pc_block` at ≥ {PC_BLOCK_TOL:g}")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("axis inside the\ntop-k PC subspace")
    ax.set_title(f"{run.label} — redundant with PC1, or with the whole top block?",
                 fontsize=12)

    ax = axes[1]
    var = layer_field(per_layer, "pc1_explained_variance")
    ax.bar(range(len(var)), var, color=CATEGORICAL[2], width=0.78, alpha=0.9)
    ax.set_ylabel("PC1 explained\nvariance ratio")
    ax.set_ylim(0, max(0.2, float(np.nanmax(var)) * 1.15)
                if np.isfinite(var).any() else 1.0)
    depth_axis(ax, n)
    return save_figure(fig, out_dir, "pc_subspace_fraction")


# ---------------------------------------------------------------------------
# A4 — pooled: is redundancy a property of the axis or of the cloud
# ---------------------------------------------------------------------------

def _axis_vs_pc1_scatter(runs: Sequence[Run], out_dir: Path) -> Optional[Path]:
    """
    cos(axis, PC1) against PC1's explained variance, every layer of every run.

    If the points climb with PC1's variance share, "the axis is PC1" is
    mostly a statement about how anisotropic the cloud is at that depth — the
    axis has little room to be anything else. If they do not, the redundancy
    is a property of the axis itself. The distinction changes what Phase 5
    should do about it.
    """
    xs, ys, cs, labels = [], [], [], []
    for run in runs:
        per_layer = (run.axis_identity or {}).get("per_layer") or []
        if not per_layer:
            continue
        var = layer_field(per_layer, "pc1_explained_variance")
        cos = layer_field(per_layer, "cos_axis_pc1")
        ok = np.isfinite(var) & np.isfinite(cos)
        if not ok.any():
            continue
        xs.append(var[ok])
        ys.append(cos[ok])
        cs.append(model_color(run.model))
        labels.append(run.label)

    if not xs:
        print("    axis: A4 skipped — no run carries axis_identity")
        return None

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    for x, y, c, lab in zip(xs, ys, cs, labels):
        ax.scatter(x, y, s=26, color=c, alpha=0.72, edgecolor="white",
                   linewidth=0.5, label=lab)
    reference_line(ax, PC1_TOL, f"verdict `pc1` at ≥ {PC1_TOL:g}")
    ax.set_xlabel("PC1 explained-variance ratio (how anisotropic the cloud is)")
    ax.set_ylabel("|cos(axis, PC1)|")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", fontsize=7.5)

    allx, ally = np.concatenate(xs), np.concatenate(ys)
    r = (float(np.corrcoef(allx, ally)[0, 1])
         if allx.size >= 3 and np.ptp(allx) > 0 else float("nan"))
    ax.set_title("Is the axis PC1, or is the cloud just anisotropic?\n"
                 f"pooled Pearson r = {r:+.2f} over {allx.size} layers",
                 fontsize=12)
    return save_figure(fig, out_dir, "axis_vs_pc1_scatter")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _isotropic_cos(run: Run) -> Optional[float]:
    """
    1/sqrt(d), from the saved axes when they exist.

    d is not in the per-run JSON — `axis_identity` reports cosines, not the
    dimension they were taken in — so this reads the axes npz. When that is
    absent the chance floor is omitted rather than guessed from n_tokens or
    the model name, either of which would put a confident wrong line on the
    figure.
    """
    axes = run.axes()
    if not axes or axes.get("axes") is None:
        return None
    d = int(np.asarray(axes["axes"]).shape[-1])
    return float(1.0 / np.sqrt(d)) if d > 0 else None
