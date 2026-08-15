"""
p1c_frames/visualization/feasibility.py — sub-experiment E, Lemma 6.4.

Four per-run figures (E1-E4 in FIGURES-1c.md). The organizing constraint is
status-1c finding 5: **the cone condition is nearly vacuous as a boolean and
the margin is not.** Wendel gives probability 1 whenever d > n, which every
prompt in the sweep satisfies, so P-H1 is close to guaranteed as registered
— deliberately, since the prediction was stated in the direction that is
boring if true so that the informative outcome is the failure.

So the reportable quantity is the margin and the layer at which it first
crosses zero, and every figure here is built to make a boolean impossible to
read off:

  * E1 is the margin profile with zero drawn, not a feasible/infeasible
    strip.
  * E2 draws Wendel's own probability against n at this model's d, so
    "probability 1" is visible as a property of the regime rather than as a
    verdict about the model.
  * E4 puts the margin against the measured i.i.d. reference (0.221 at
    n = 20, 0.030 at n = 512, both at d = 1024), because a margin of 0.03 is
    a small number and an unremarkable one at n = 512.

E1 also hatches the layers whose optimizer did not converge. That asymmetry
matters and is worth stating in the figure: an unconverged run can only
OVERSTATE the margin, so an unconverged "feasible" is not trustworthy while
an unconverged "infeasible" still is (`hemisphere_test`).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from p1c_frames.hemisphere_feasibility import wendel_probability

from .loaders import Run, record_field, records
from .style import (
    BLOG_STYLE, CATEGORICAL, INVALID_COLOR, NULL_BAND, UNREACHABLE_COLOR,
    caption, depth_axis, no_data, reference_line, save_figure,
)

__all__ = ["generate_feasibility_figures"]

#: Measured margins for i.i.d. uniform clouds at d = 1024 (status-1c
#: finding 5). Two points, so E4 draws them as reference levels rather than
#: interpolating a curve nobody measured.
IID_REFERENCE = {20: 0.221, 512: 0.030}


def _skip(name: str, reason: str) -> None:
    print(f"    skip {name}: {reason}")


def generate_feasibility_figures(run: Run, out_dir: Path) -> List[Path]:
    """E1-E4 for one run."""
    if not run.has("E"):
        _skip("feasibility", next((m for m in run.missing
                                   if m.startswith("E ")), "no E block"))
        return []

    per_layer = records(run, "E")
    margins = run.series("E.margins")
    if not margins.size and per_layer:
        margins = record_field(per_layer, "margin")
    if not margins.size:
        _skip("feasibility", "E block carries no margins")
        return []

    paths: List[Optional[Path]] = [
        _e1_margin_depth(run, per_layer, margins, out_dir),
        _e2_wendel_reference(run, per_layer, out_dir),
    ]
    if per_layer:
        paths.append(_e3_support_and_min_ip(run, per_layer, out_dir))
    else:
        _skip("support_and_min_ip", "E block carries no per-layer records")
    paths.append(_e4_margin_shrinkage(run, margins, out_dir))
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# E1 — the reportable object
# ---------------------------------------------------------------------------

def _e1_margin_depth(run: Run, per_layer: List[dict], margins: np.ndarray,
                     out_dir: Path) -> Path:
    """
    The hull margin vs depth, with zero, the minimum, and the first
    infeasible layer marked.

    The margin is `dist(0, conv{x_i})` — an exact convex QP optimum, not a
    feasibility heuristic — so zero is a real boundary and not a tolerance.
    Layers whose optimizer did not converge are hatched: their margin is an
    upper bound, which can only make the cloud look more feasible than it
    is.
    """
    x = np.arange(margins.size)
    conv = (record_field(per_layer, "converged") if per_layer
            else np.ones(margins.size))

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        ax.fill_between(x, 0, margins, where=margins > 0, color=CATEGORICAL[0],
                        alpha=0.16, linewidth=0)
        ax.fill_between(x, 0, margins, where=margins <= 0, color="#D6483F",
                        alpha=0.20, linewidth=0)
        ax.plot(x, margins, color="#12406F", lw=2.6, zorder=4)
        ax.axhline(0.0, color="#374151", lw=1.2, zorder=3)

        bad = np.nonzero((conv == 0) | ~np.isfinite(conv))[0]
        for i in bad:
            ax.axvspan(i - 0.5, i + 0.5, facecolor="none", edgecolor="#6B7280",
                       hatch="\\\\", linewidth=0.0, zorder=1)
        if bad.size:
            ax.axvspan(np.nan, np.nan, facecolor="none", edgecolor="#6B7280",
                       hatch="\\\\", linewidth=0.0,
                       label=f"optimizer unconverged ({bad.size}) — margin is "
                             f"an upper bound")

        i_min = int(run.scalar("E", "min_margin_layer", -1))
        if 0 <= i_min < margins.size:
            ax.plot([i_min], [margins[i_min]], marker="v", ms=10,
                    color=CATEGORICAL[1], zorder=6,
                    label=f"minimum margin {margins[i_min]:.4f} at layer {i_min}")
        first_bad = int(run.scalar("E", "first_infeasible_layer", -1))
        if first_bad >= 0:
            reference_line(ax, first_bad,
                           f"first infeasible layer ({first_bad})", axis="x")

        depth_axis(ax, run.n_layers)
        ax.set_ylabel("margin  $\\max_w \\min_i \\langle x_i, w\\rangle$")
        ax.set_title(f"E1 · the cone condition, as a quantity — {run.label}")
        ax.legend(loc="best", fontsize=8.5)
        caption(fig, (
            "P-H1 is registered in the direction that is boring if true. The "
            "informative outcome is a margin near zero, or the depth at which "
            "it crosses — which is the number this sub-experiment exists to "
            "produce."))
    return save_figure(fig, out_dir, "margin_depth")


# ---------------------------------------------------------------------------
# E2 — why the boolean is nearly vacuous
# ---------------------------------------------------------------------------

def _e2_wendel_reference(run: Run, per_layer: List[dict],
                         out_dir: Path) -> Path:
    """
    Wendel's probability against n at this model's d, with this prompt marked.

    `wendel_probability` is imported and evaluated over a range of n — the
    one place this package calls a phase function on arguments the run did
    not use, and it is drawing the reference the run is read against rather
    than measuring anything about the run. The prompt's own n sits on the
    curve, and for every prompt in the sweep it sits on the flat part at
    p = 1.
    """
    d_model = int(per_layer[0].get("d_model") or 0) if per_layer else 0
    if not d_model:
        d_model = 1024
    n_run = run.n_tokens or 2

    n_grid = np.unique(np.concatenate([
        np.linspace(2, max(3 * d_model, 4 * n_run), 160).astype(int),
        np.array([n_run])]))
    p = np.array([wendel_probability(int(n), d_model) for n in n_grid])

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        ax.plot(n_grid, p, color="#12406F", lw=2.4)
        ax.fill_between(n_grid, 0, p, **NULL_BAND)
        ax.axvline(d_model, **{"color": "#6B7280", "ls": ":", "lw": 1.2})
        ax.annotate(f"d = {d_model}", xy=(d_model, 0.5), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, color="#4B5563",
                    rotation=90, va="center")
        ax.plot([n_run], [wendel_probability(n_run, d_model)], marker="o",
                ms=10, color=CATEGORICAL[1], zorder=5,
                label=f"this prompt: n = {n_run}")
        ax.set_xlabel("n (tokens)")
        ax.set_ylabel("P(all in one open hemisphere)")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"E2 · P-H1 is near-certain by construction — {run.label}")
        ax.legend(loc="lower left")
        caption(fig, (
            "Theorem 6.7 gives probability exactly 1 for d > n, and every "
            "prompt in the sweep has n < d. That is why the margin, and not "
            "the boolean, is what E reports."))
    return save_figure(fig, out_dir, "wendel_reference")


# ---------------------------------------------------------------------------
# E3 — who breaks the cone
# ---------------------------------------------------------------------------

def _e3_support_and_min_ip(run: Run, per_layer: List[dict],
                           out_dir: Path) -> Path:
    """
    Support size and the minimum pairwise inner product vs depth.

    When 0 is in the hull, the support is the subset of tokens that spans it
    — the tokens responsible for breaking the cone condition, which is worth
    reporting rather than just the failure (`hemisphere_test`). The minimum
    pairwise inner product is drawn beside it because a support of 2 with a
    minimum IP near −1 is an antipodal pair and a support of 8 with a
    minimum near 0 is a spread cloud, and those are different stories with
    the same margin.
    """
    support = record_field(per_layer, "support_size")
    min_ip = record_field(per_layer, "min_pairwise_ip")

    with plt.rc_context(BLOG_STYLE):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9.2, 5.8), sharex=True,
            gridspec_kw=dict(height_ratios=[1, 1], hspace=0.12))
        ax.bar(np.arange(support.size), support, color=CATEGORICAL[0],
               width=0.75, alpha=0.85)
        ax.set_ylabel("support size\n(tokens on the certificate)")
        ax.set_title(f"E3 · how many tokens hold the hemisphere — {run.label}")

        ax2.plot(np.arange(min_ip.size), min_ip, color=CATEGORICAL[1], lw=2.2)
        reference_line(ax2, -1.0, "antipodal pair", side="left")
        reference_line(ax2, 0.0, "orthogonal")
        ax2.set_ylabel("min pairwise\ninner product")
        depth_axis(ax2, run.n_layers)
        caption(fig, "Same margin, different geometry: a support of 2 at "
                     "min IP ≈ −1 is an antipodal pair; a large support at "
                     "min IP ≈ 0 is a spread cloud.")
    return save_figure(fig, out_dir, "support_and_min_ip")


# ---------------------------------------------------------------------------
# E4 — is this margin small?
# ---------------------------------------------------------------------------

def _e4_margin_shrinkage(run: Run, margins: np.ndarray, out_dir: Path) -> Path:
    """
    This run's margins against the measured i.i.d. reference at d = 1024.

    status-1c finding 5: the margin SHRINKS as n → d — 0.221 at n = 20 and
    0.030 at n = 512 for i.i.d. uniform clouds. So "the margin is 0.03" is
    not a small number on its own; it is a small number only against the
    reference at the same n. Both measured reference levels are drawn, and
    the one matching this prompt's length is highlighted.
    """
    n_run = run.n_tokens or 0
    nearest = min(IID_REFERENCE, key=lambda n: abs(n - n_run)) if n_run else None

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        ax.plot(np.arange(margins.size), margins, color="#12406F", lw=2.6,
                label="this run's margin", zorder=4)
        for n_ref, lvl in sorted(IID_REFERENCE.items()):
            live = (n_ref == nearest)
            ax.axhline(lvl, color=CATEGORICAL[1] if live else INVALID_COLOR,
                       ls="--" if live else ":", lw=1.8 if live else 1.2,
                       zorder=2)
            ax.annotate(f"i.i.d. uniform at n = {n_ref}, d = 1024: {lvl:.3f}"
                        + ("  ← nearest to this prompt" if live else ""),
                        xy=(0.01, lvl), xycoords=("axes fraction", "data"),
                        fontsize=8, va="bottom",
                        color=CATEGORICAL[1] if live else "#6B7280")
        ax.axhline(0.0, color="#374151", lw=1.2, zorder=3)
        depth_axis(ax, run.n_layers)
        ax.set_ylabel("margin")
        ax.set_title(f"E4 · is this margin small? — {run.label} "
                     f"(n = {n_run})")
        ax.legend(loc="upper right", fontsize=8.5)
        caption(fig, "The margin shrinks as n → d for random clouds too, so a "
                     "raw value means nothing without the matched reference "
                     "(status-1c finding 5).")
    return save_figure(fig, out_dir, "margin_shrinkage")
