"""
p1c_frames/visualization/designs.py — sub-experiment F, spherical designs.

Five per-run figures (F1-F5 in FIGURES-1c.md). Two rules from the phase's
own docs govern all of them.

**Never draw the raw Q_k.** For i.i.d. points E[Q_k] = 1/n exactly, so every
large-m configuration looks like a design under an absolute threshold, and a
raw comparison between checkpoints with different centroid counts would be
reading the cluster count rather than the geometry (status-1c finding 6).
The plotted quantity is always the ratio against the matched-(m, d)
baseline, which the measurement in `centroids.py` shows is flat at 1 across
a 32× range in m.

**Never read the ratio against a fixed tolerance.** Both the deviation and
the noise shrink with degree, at different rates — the 2σ band goes 0.17,
0.015, 0.002 at k = 1, 2, 3 — so a single tolerance is wrong in a different
direction at every degree. F1 shades each degree's own band and F2 draws
`outside_band` as the phase recorded it, rather than re-thresholding here.

A third, quieter one: **F's per-layer list is not one entry per layer.**
Layers whose centroids could not be loaded are skipped and recorded in
`errors`, so every series here is scattered back onto the depth axis by each
record's own `layer` key. Compressing the gaps would draw a 24-layer profile
as an 18-layer one and move every feature left.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .loaders import Run, record_field, record_matrix, records
from .style import (
    BLOG_STYLE, CATEGORICAL, INVALID_COLOR, SEQ_CMAP, caption, degree_color,
    depth_axis, no_data, reference_line, save_figure,
)

__all__ = ["generate_design_figures"]


def _skip(name: str, reason: str) -> None:
    print(f"    skip {name}: {reason}")


def _t_max(run: Run, recs: List[dict]) -> int:
    t = int(run.scalar("F", "t_max", 0) or 0)
    if t:
        return t
    for r in recs:
        v = r.get("Q_ratio")
        if isinstance(v, (list, tuple)):
            return len(v)
    return 0


def generate_design_figures(run: Run, out_dir: Path) -> List[Path]:
    """F1-F5 for one run."""
    if not run.has("F"):
        _skip("designs", next((m for m in run.missing if m.startswith("F ")),
                              "no F block"))
        return []

    recs = records(run, "F")
    if not recs:
        errs = run.block("F").get("errors") or []
        _skip("designs", f"F ran but produced no per-layer rows"
                         + (f" — first error: {errs[0]}" if errs else ""))
        return []

    t_max = _t_max(run, recs)
    n_layers = run.n_layers
    paths: List[Optional[Path]] = [
        _f1_q_ratio_depth(run, recs, t_max, n_layers, out_dir),
        _f2_outside_band_strip(run, recs, t_max, n_layers, out_dir),
        _f3_design_order_depth(run, recs, n_layers, out_dir),
        _f4_mode_structure(run, recs, n_layers, out_dir),
        _f5_sharp_score_depth(run, recs, n_layers, out_dir),
    ]
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# F1 — the headline quantity
# ---------------------------------------------------------------------------

def _f1_q_ratio_depth(run: Run, recs: List[dict], t_max: int, n_layers: int,
                      out_dir: Path) -> Path:
    """
    Q_k / Q_k^random per degree vs depth, each degree's own band shaded.

    One panel per degree rather than one axis with three curves: the bands
    differ by two orders of magnitude (0.17 at k = 1, 0.002 at k = 3), so a
    shared axis would render the k = 3 band as a line and invite exactly the
    fixed-tolerance reading the phase warns against.
    """
    ratio = record_matrix(recs, "Q_ratio", t_max, n=n_layers)
    band = record_matrix(recs, "random_band", t_max, n=n_layers)

    with plt.rc_context(BLOG_STYLE):
        fig, axes = plt.subplots(t_max, 1, figsize=(9.2, 2.1 * t_max + 1.2),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        x = np.arange(n_layers)
        for k in range(t_max):
            ax = axes[k]
            b = band[:, k]
            ax.fill_between(x, 1 - b, 1 + b, color=INVALID_COLOR, alpha=0.35,
                            linewidth=0,
                            label="2σ random band" if k == 0 else None)
            ax.axhline(1.0, color="#374151", lw=1.0, zorder=2)
            ax.plot(x, ratio[:, k], color=degree_color(k + 1, t_max), lw=2.4,
                    marker="o", ms=3.5, zorder=4)
            ax.set_ylabel(f"$Q_{k+1}$ ratio")
            if k == 0:
                ax.legend(loc="best", fontsize=8)
                ax.set_title(f"F1 · sharpness against the matched-(m, d) "
                             f"baseline — {run.label}")
        depth_axis(axes[-1], n_layers)
        caption(fig, (
            "The ratio, never the raw Q_k: for i.i.d. points E[Q_k] = 1/n "
            "exactly, so a raw curve reads the cluster count. Each degree "
            "carries its own band because both the signal and the noise "
            "shrink with k, at different rates."))
    return save_figure(fig, out_dir, "q_ratio_depth")


# ---------------------------------------------------------------------------
# F2 — the effect-size floor as a picture
# ---------------------------------------------------------------------------

def _f2_outside_band_strip(run: Run, recs: List[dict], t_max: int,
                           n_layers: int, out_dir: Path) -> Path:
    """
    Layer × degree matrix of `outside_band`, as the run recorded it.

    Read as a picture of where this run has any signal at all. Everything
    inside the band is inside the sampling noise of the baseline itself —
    "whatever the unbanded verdict says, this is not a detection"
    (`adjudicate_p_s1_banded`). Layers F could not compute are drawn in the
    invalid grey rather than as False, because "no centroids" and "no
    signal" are different statements.
    """
    outside = record_matrix(recs, "outside_band", t_max, n=n_layers)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.2, 2.4 + 0.22 * t_max))
        grid = np.where(np.isfinite(outside), outside, np.nan)
        # `with_extremes` rather than `set_bad`: the latter mutates a shared
        # colormap object and is deprecated in current matplotlib.
        cmap = plt.matplotlib.colors.ListedColormap(
            ["#EEF2F7", CATEGORICAL[2]]).with_extremes(bad=INVALID_COLOR)
        ax.imshow(grid.T, aspect="auto", origin="lower", cmap=cmap,
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(t_max))
        ax.set_yticklabels([f"k = {k+1}" for k in range(t_max)])
        ax.set_xlabel("layer")
        ax.grid(False)
        ax.set_title(f"F2 · where the signal clears its own noise — "
                     f"{run.label}")
        handles = [plt.matplotlib.patches.Patch(facecolor=c, label=l)
                   for c, l in ((CATEGORICAL[2], "outside the 2σ random band"),
                                ("#EEF2F7", "inside — not a detection"),
                                (INVALID_COLOR, "layer not computed"))]
        ax.legend(handles=handles, loc="upper center", ncol=3,
                  bbox_to_anchor=(0.5, -0.28), fontsize=8.5, frameon=False)
        errs = run.block("F").get("errors") or []
        caption(fig, (f"{len(errs)} layer(s) produced no centroids and are "
                      f"grey, not False." if errs else
                      "Every layer produced centroids."), y=-0.10)
    return save_figure(fig, out_dir, "outside_band_strip")


# ---------------------------------------------------------------------------
# F3 — the design order, and the m it was measured at
# ---------------------------------------------------------------------------

def _f3_design_order_depth(run: Run, recs: List[dict], n_layers: int,
                           out_dir: Path) -> Path:
    """
    `t_design_vs_random` and `t_design_strict` vs depth, with n_centroids.

    Two orders, because they answer different questions: the strict one asks
    whether Q_k actually vanishes (a real design), the vs-random one whether
    it beats 95% of random draws at the same (m, d). The centroid count is
    drawn beneath because the design order is bounded by it — a 4-centroid
    layer cannot be a high-order design, and reading the top panel without
    the bottom one invites treating that ceiling as a finding.
    """
    vs_rand = record_field(recs, "t_design_vs_random", n=n_layers)
    strict = record_field(recs, "t_design_strict", n=n_layers)
    m = record_field(recs, "n_centroids", n=n_layers)
    x = np.arange(n_layers)

    with plt.rc_context(BLOG_STYLE):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9.2, 5.8), sharex=True,
            gridspec_kw=dict(height_ratios=[1.4, 1], hspace=0.12))
        ax.step(x, vs_rand, where="mid", color=CATEGORICAL[0], lw=2.2,
                label="t-design vs the random p95")
        ax.step(x, strict, where="mid", color=CATEGORICAL[1], lw=2.0, ls="--",
                label="t-design, strict (Q_k ≈ 0)")
        ax.set_ylabel("design order t")
        ax.legend(loc="best", fontsize=8.5)
        ax.set_title(f"F3 · design order by depth — {run.label}")

        ax2.bar(x, m, color=INVALID_COLOR, width=0.75)
        ax2.set_ylabel("centroids m")
        depth_axis(ax2, n_layers)
        caption(fig, "The order is bounded by the centroid count, so the two "
                     "panels have to be read together. The ratio itself is "
                     "m-comparable (centroids.py); the ORDER is not.")
    return save_figure(fig, out_dir, "design_order_depth")


# ---------------------------------------------------------------------------
# F4 — Definition 9.1's other half
# ---------------------------------------------------------------------------

def _f4_mode_structure(run: Run, recs: List[dict], n_layers: int,
                       out_dir: Path) -> Path:
    """
    Mode count, mass at the modes, and every mode's location.

    A sharp configuration wants BOTH halves of Definition 9.1: few distinct
    inner products, and near-vanishing low-order moments. F1 draws the
    second; this draws the first. The mode locations panel is the one that
    distinguishes "two tight clusters of directions" from "one broad blob
    the peak-finder split in half" — mass_at_modes is what separates them
    and is drawn as the marker size.
    """
    modes = [r.get("modes") if isinstance(r.get("modes"), dict) else {}
             for r in recs]
    n_modes = record_field(modes, "n_modes")
    mass = record_field(modes, "mass_at_modes")
    layers = [int(r.get("layer", i)) for i, r in enumerate(recs)]

    with plt.rc_context(BLOG_STYLE):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9.2, 6.2), sharex=True,
            gridspec_kw=dict(height_ratios=[1, 1.4], hspace=0.12))
        ax.bar(layers, n_modes, color=CATEGORICAL[0], width=0.75, alpha=0.85)
        ax.set_ylabel("distinct inner-product\nmodes")
        ax.set_title(f"F4 · few distinct inner products? — {run.label}")

        for lay, rec, mm in zip(layers, modes, mass):
            locs = rec.get("mode_locations") or []
            if not isinstance(locs, (list, tuple)):
                continue
            size = 20 + 180 * (float(mm) if np.isfinite(mm) else 0.2)
            ax2.scatter([lay] * len(locs), [float(v) for v in locs],
                        s=size, color=CATEGORICAL[1], alpha=0.7,
                        edgecolor="white", linewidth=0.7)
        ax2.axhline(0.0, **{"color": "#6B7280", "ls": ":", "lw": 1.2})
        ax2.set_ylabel("mode location\n(inner product)")
        ax2.set_ylim(-1.05, 1.05)
        depth_axis(ax2, n_layers)
        caption(fig, "Marker size is `mass_at_modes` — the fraction of pairs "
                     "within a bin of some peak. Two modes carrying 15% of "
                     "the mass is a peak-finder artefact; two carrying 95% is "
                     "a sharp configuration.")
    return save_figure(fig, out_dir, "mode_structure")


# ---------------------------------------------------------------------------
# F5 — one number per layer
# ---------------------------------------------------------------------------

def _f5_sharp_score_depth(run: Run, recs: List[dict], n_layers: int,
                          out_dir: Path) -> Path:
    """
    `sharp_score` (the mean ratio across degrees) vs depth.

    The summary of F1, drawn once so a cross-run figure has a single column
    to aggregate — and captioned with the clusterer arm, because if the
    three arms disagree about P-S1 then the design signal is a property of
    the clustering rather than of the geometry, and a figure with no arm
    printed on it cannot be checked for that.
    """
    score = record_field(recs, "sharp_score", n=n_layers)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.2, 4.4))
        x = np.arange(n_layers)
        ax.fill_between(x, 1.0, score, where=score <= 1.0, color=CATEGORICAL[2],
                        alpha=0.18, linewidth=0)
        ax.fill_between(x, 1.0, score, where=score > 1.0, color="#D6483F",
                        alpha=0.15, linewidth=0)
        ax.plot(x, score, color="#12406F", lw=2.4, marker="o", ms=3.5)
        reference_line(ax, 1.0, "i.i.d. uniform at the same (m, d)",
                       side="left")
        depth_axis(ax, n_layers)
        ax.set_ylabel("sharp_score (mean $Q_k$ ratio)")
        method = run.text("F", "method", "?")
        ax.set_title(f"F5 · sharpness summary — {run.label}")
        caption(fig, (
            f"Clusterer arm: {method}. Fixed per sweep — the ratio is "
            f"m-comparable, so arms need not match on m, but if the three "
            f"arms disagree about P-S1 the signal is a property of the "
            f"clustering rather than of the geometry."))
    return save_figure(fig, out_dir, "sharp_score_depth")
