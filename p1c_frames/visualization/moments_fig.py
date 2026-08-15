"""
p1c_frames/visualization/moments_fig.py — sub-experiment C.

Four per-run figures (C1-C4 in FIGURES-1c.md; C5 waits on gap G2). The
question is not "how much rank was lost" but "was the rank collapse a sink
count".

`moments.py`: raw effective rank is 1/<s²> with norm-squared weights, and in
the near-orthogonal limit it degenerates to the participation ratio of the
NORM distribution alone — carrying zero directional content. So status-1's
"MinRank → 2.3 by step 143000" is not yet a geometric claim, and C1/C2/C3
are three views of the one comparison that settles it: does `shannon_raw`
track `norm_pr` or `shannon_normed`?

C3 draws the adjudicator rather than restating it. Both correlations are
computed by `adjudicate_sink_hypothesis` and read off the artifact; this
module decides only how they are drawn.

C4 is a different question sharing the block: how well the three-term
cumulant ladder reconstructs the measured interaction energy at each β.
Finding 8 is that it is accurate to 0.00 / 0.07 / 0.80% at β = 0.1 / 1 / 2
and **26.6% at β = 5** — so three of the four energy columns are redundant
and the fourth is not. A heatmap is the right form because the answer is
per-(layer, β) and the interesting cell is a whole column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .loaders import Run, record_field, records
from .style import (
    BLOG_STYLE, CATEGORICAL, DEPTH_CMAP, INVALID_COLOR, SEQ_CMAP, caption,
    depth_axis, no_data, reference_line, save_figure, verdict_box,
)

__all__ = ["generate_moment_figures"]

#: The four rank-like quantities, in the order `rank_panel` documents them:
#: raw (direction and scale mixed), normed (direction only), PR (the energy
#: expansion's rank), norm-PR (scale only, zero directional content).
RANK_KEYS = ("shannon_raw", "shannon_normed", "pr_rank", "norm_pr")
RANK_COLORS: Dict[str, str] = {
    "shannon_raw":    "#12406F",
    "shannon_normed": CATEGORICAL[2],
    "pr_rank":        CATEGORICAL[3],
    "norm_pr":        CATEGORICAL[1],
}
RANK_LABELS: Dict[str, str] = {
    "shannon_raw":    "shannon_raw — what status-1's MinRank reported",
    "shannon_normed": "shannon_normed — direction only (frame-correct)",
    "pr_rank":        "pr_rank — 1/⟨G²⟩ of the normed Gram",
    "norm_pr":        "norm_pr — the norm distribution alone",
}


def _skip(name: str, reason: str) -> None:
    print(f"    skip {name}: {reason}")


def generate_moment_figures(run: Run, out_dir: Path) -> List[Path]:
    """C1-C4 for one run."""
    if not run.has("C"):
        _skip("moments", next((m for m in run.missing if m.startswith("C ")),
                              "no C block"))
        return []

    panels = records(run, "C", "panels")
    paths: List[Optional[Path]] = []
    if panels:
        paths.append(_c1_rank_panel_depth(run, panels, out_dir))
        paths.append(_c2_sink_ratio_depth(run, panels, out_dir))
        paths.append(_c3_sink_adjudication(run, panels, out_dir))
    else:
        _skip("rank figures", "C block carries no per-layer panels")

    checks = records(run, "C", "moment_identity")
    if checks:
        paths.append(_c4_moment_identity_error(run, checks, out_dir))
    else:
        _skip("moment_identity_error",
              "no energies.json for this run, so the ladder was never "
              "checked against a measured E_beta")
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# C1 — four ranks on one axis
# ---------------------------------------------------------------------------

def _c1_rank_panel_depth(run: Run, panels: List[dict], out_dir: Path) -> Path:
    """
    All four rank-like quantities vs depth, log-y.

    One axis, not four panels: the whole content of the figure is which
    curves lie on top of each other. Log-y because they span an order of
    magnitude and the interesting agreement is multiplicative — raw tracking
    norm-PR means their RATIO is flat, which is a constant offset here and a
    converging pair on a linear axis.
    """
    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.2, 5.0))
        for key in RANK_KEYS:
            v = record_field(panels, key)
            if not np.isfinite(v).any():
                continue
            ax.plot(np.arange(v.size), v, color=RANK_COLORS[key],
                    label=RANK_LABELS[key], lw=2.2,
                    ls="-" if key in ("shannon_raw", "shannon_normed") else "--")
        ax.set_yscale("log")
        reference_line(ax, 2.0, "rank 2 — the degenerate floor", side="left")
        depth_axis(ax, run.n_layers)
        ax.set_ylabel("effective rank")
        ax.set_title(f"C1 · four ranks, one axis — {run.label}")
        ax.legend(loc="best", fontsize=8.5)
        caption(fig, "If shannon_raw lies on norm_pr, the reported rank "
                     "collapse is a statement about outlier token norms. If "
                     "it lies on shannon_normed, it survives the frame "
                     "correction.")
    return save_figure(fig, out_dir, "rank_panel_depth")


# ---------------------------------------------------------------------------
# C2 — the diagnostic ratio
# ---------------------------------------------------------------------------

def _c2_sink_ratio_depth(run: Run, panels: List[dict], out_dir: Path) -> Path:
    """
    sink_ratio = shannon_raw / norm_pr vs depth, with norm_max/median beneath.

    Near 1 means the raw rank is being set by the norm distribution and
    carries no geometric information. The second panel is the mechanism: a
    ratio near 1 with a flat norm distribution would be a coincidence, and
    with a few tokens 6× the median norm it is a sink count.
    """
    ratio = record_field(panels, "sink_ratio")
    nmax = record_field(panels, "norm_max_over_median")

    with plt.rc_context(BLOG_STYLE):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9.2, 6.0), sharex=True,
            gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.12))
        ax.plot(np.arange(ratio.size), ratio, color="#12406F", lw=2.4)
        ax.fill_between(np.arange(ratio.size), 1.0, ratio, color="#12406F",
                        alpha=0.12)
        reference_line(ax, 1.0, "raw rank IS the norm distribution",
                       side="left")
        ax.set_ylabel("sink_ratio\n(raw / norm-PR)")
        ax.set_title(f"C2 · is the rank collapse a sink count? — {run.label}")

        ax2.plot(np.arange(nmax.size), nmax, color=CATEGORICAL[1], lw=2.0)
        reference_line(ax2, 1.0, "uniform norms")
        ax2.set_ylabel("max/median\ntoken norm")
        depth_axis(ax2, run.n_layers)
        caption(fig, "The top panel is the diagnostic; the bottom is the "
                     "mechanism it would have to come from.")
    return save_figure(fig, out_dir, "sink_ratio_depth")


# ---------------------------------------------------------------------------
# C3 — the adjudicator as a scatter
# ---------------------------------------------------------------------------

def _c3_sink_adjudication(run: Run, panels: List[dict], out_dir: Path) -> Path:
    """
    shannon_raw against norm_pr and against shannon_normed, side by side.

    A single layer can agree by coincidence; a whole depth profile cannot,
    which is why the adjudicator uses the across-layer correlation and why
    this figure is two scatters over depth rather than two curves. Both
    correlations and the verdict are read from the artifact — nothing here
    recomputes them, so a figure that disagrees with the phase's verdict is
    a bug in this module by construction.
    """
    raw = record_field(panels, "shannon_raw")
    npr = record_field(panels, "norm_pr")
    nrm = record_field(panels, "shannon_normed")
    sv = run.block("C.sink_verdict")
    colors = DEPTH_CMAP(np.linspace(0.08, 0.95, max(raw.size, 1)))

    with plt.rc_context(BLOG_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
        for ax, other, name, corr_key in (
                (axes[0], npr, "norm_pr (scale only)", "corr_raw_vs_norm_pr"),
                (axes[1], nrm, "shannon_normed (direction only)",
                 "corr_raw_vs_normed")):
            ax.scatter(other, raw, c=colors[:other.size], s=48,
                       edgecolor="white", linewidth=0.8, zorder=4)
            lim = np.nanmax([np.nanmax(other) if np.isfinite(other).any() else 1,
                             np.nanmax(raw) if np.isfinite(raw).any() else 1])
            ax.plot([0, lim * 1.05], [0, lim * 1.05], color="#6B7280",
                    ls=":", lw=1.2, zorder=1)
            ax.set_xlabel(name)
            ax.set_xlim(0, lim * 1.05)
            ax.set_ylim(0, lim * 1.05)
            c = sv.get(corr_key)
            ax.set_title(f"corr = {float(c):.3f}" if c is not None
                         else "corr = n/a", fontsize=11)
        axes[0].set_ylabel("shannon_raw")
        fig.suptitle(f"C3 · which one does the raw rank track? — {run.label}",
                     fontsize=12)
        verdict_box(axes[0], sv.get("verdict", ""), loc="lower right")
        frac = sv.get("frac_layers_close_to_norm_pr")
        caption(fig, (f"Point colour is depth (dark = early). "
                      f"{float(frac):.0%} of layers have raw within 25% of "
                      f"norm-PR." if frac is not None else
                      "Point colour is depth (dark = early)."), y=0.0)
    return save_figure(fig, out_dir, "sink_adjudication")


# ---------------------------------------------------------------------------
# C4 — where the ladder can stand in for the energy sweep
# ---------------------------------------------------------------------------

def _c4_moment_identity_error(run: Run, checks: List[dict],
                              out_dir: Path) -> Path:
    """
    Layer × β heatmap of the two-term reconstruction's relative error.

    Not a pass/fail gate (`verify_moment_identity`'s own docstring): it
    quantifies where the cumulant ladder can stand in for the four-β energy
    sweep and where it cannot, which is what decides which columns the
    re-report can drop. The 1% contour is drawn because that is the
    threshold `ladder_sufficient` uses, imported implicitly by reading the
    same field rather than by re-thresholding here.
    """
    betas = _beta_columns(checks)
    if not betas:
        return None

    err = np.full((len(checks), len(betas)), np.nan)
    for i, row in enumerate(checks):
        for j, b in enumerate(betas):
            cell = row.get(b) or row.get(str(b)) or row.get(f"{b}")
            if isinstance(cell, dict):
                v = cell.get("rel_err_two")
                if v is not None:
                    err[i, j] = float(v)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        im = ax.imshow(err * 100.0, aspect="auto", origin="lower",
                       cmap=SEQ_CMAP, interpolation="nearest",
                       norm=plt.matplotlib.colors.LogNorm(
                           vmin=max(np.nanmin(err[err > 0]) * 100, 1e-3)
                           if np.isfinite(err).any() and (err > 0).any() else 1e-3,
                           vmax=max(np.nanmax(err) * 100, 1e-2)
                           if np.isfinite(err).any() else 1.0))
        ax.set_xticks(range(len(betas)))
        ax.set_xticklabels([f"β = {b:g}" for b in betas])
        ax.set_ylabel("layer")
        ax.set_xlabel("interaction-energy column")
        ax.set_title(f"C4 · which energy columns the ladder replaces — "
                     f"{run.label}")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("two-term reconstruction error (%)")

        # Mark the cells the phase's own `ladder_sufficient` accepts, so the
        # threshold is read off the artifact rather than re-applied here.
        for i, row in enumerate(checks):
            for j, b in enumerate(betas):
                cell = row.get(b) or row.get(str(b))
                if isinstance(cell, dict) and cell.get("ladder_sufficient"):
                    ax.plot(j, i, marker="o", color="white", ms=4,
                            mec="#374151", mew=0.6)

        sources = {str(r.get("source")) for r in checks}
        caption(fig, (
            f"White dots are `ladder_sufficient` (under 1%) as the run "
            f"recorded it — three of the four columns are redundant and the "
            f"fourth is not (status-1c finding 8). Ladder source(s): "
            f"{', '.join(sorted(sources))}."))
    return save_figure(fig, out_dir, "moment_identity_error")


def _beta_columns(checks: List[dict]) -> List[float]:
    """
    The β values present, as floats.

    `verify_moment_identity` keys its output by float β; after a round trip
    through JSON those keys are strings ("1.0"). Both are accepted here —
    this is the same key-convention hazard `loaders.py` normalizes for layer
    indices, in the one place it survives into a nested dict.
    """
    found = set()
    for row in checks:
        for k, v in row.items():
            if not isinstance(v, dict) or "rel_err_two" not in v:
                continue
            try:
                found.add(float(k))
            except (TypeError, ValueError):
                continue
    return sorted(found)
