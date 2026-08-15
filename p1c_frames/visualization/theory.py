"""
p1c_frames/visualization/theory.py — the null model itself, with no data.

Seven figures (T1-T7 in FIGURES-1c.md), and the one class in this package
that reads no run artifacts at all. Every figure calls a `p1c_frames`
function directly on a grid of arguments:

    integrate_gamma / collapse_time / collapse_time_table   T1, T2, T3, T6
    wendel_probability                                      T4
    gegenbauer_normalized                                   T5
    random_band                                             T7

**No new math lives here.** That is the ground rule this class is closest to
breaking and therefore the one worth stating twice: if a figure needs a
quantity `p1c_frames` does not already export, the quantity belongs in
`p1c_frames`. Drawing the phase's own validated functions over a grid is a
picture of the null; recomputing them differently would be a second,
unvalidated implementation of the null living in a figures folder.

Why the class exists at all: Phase 1c compares measurements against a model
almost nobody reading the results has looked at. T1 and T2 are what
$\\gamma_\\beta$ actually does — the sigmoid whose saturation is the reason
the time-domain residual had to be invented, and the collapse-time table the
phase reproduces to 0.005. T4 is why P-H1 is near-vacuous as a boolean. T7
is the effect-size floor P-S1 was missing. Each of those is an argument the
status doc makes in prose and nowhere in a picture.

Everything here is cheap — the whole class draws in a few seconds — except
T7, which simulates. `--cheap` cuts its trial count and says so in the
caption rather than silently drawing a noisier band.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from p1c_frames.centroids import random_band
from p1c_frames.design_test import gegenbauer_normalized
from p1c_frames.gamma_ode import (
    collapse_time, collapse_time_table, integrate_gamma, time_to_threshold,
)
from p1c_frames.hemisphere_feasibility import wendel_probability

from .style import (
    BLOG_STYLE, CATEGORICAL, DEGREE_CMAP, INVALID_COLOR, MODEL_ODE_COLORS,
    SEQ_CMAP, caption, degree_color, reference_line, save_figure,
)

__all__ = ["generate_theory_figures"]

#: The grid the phase's own validation uses (`MATH.md` §3.2's table), so T1
#: and T2 are drawing the same corners the 0.005 max-deviation claim was
#: made on rather than a prettier selection.
THEORY_NS = (20, 467)
THEORY_BETAS = (0.1, 1.0, 2.0, 5.0)
#: Our prompt range, marked wherever an n axis appears.
PROMPT_N_RANGE = (20, 512)
PYTHIA_D = 1024


def generate_theory_figures(out_dir: Path, cheap: bool = False) -> List[Path]:
    """T1-T7. Takes no runs — only an output directory."""
    out_dir = Path(out_dir)
    return [
        _t1_gamma_family(out_dir),
        _t2_collapse_time_table(out_dir),
        _t3_beta_monotonicity(out_dir),
        _t4_wendel_surface(out_dir),
        _t5_gegenbauer_kernels(out_dir),
        _t6_sigmoid_compression(out_dir),
        _t7_random_band_by_degree(out_dir, cheap=cheap),
    ]


# ---------------------------------------------------------------------------
# T1 — what the null actually looks like
# ---------------------------------------------------------------------------

def _t1_gamma_family(out_dir: Path) -> Path:
    """
    gamma_beta(t) for a family of β at two n, (SA) and (USA) side by side.

    The two models are drawn in different hues and never share a panel:
    they are monotone in β in OPPOSITE directions, and a reader who takes
    one for the other gets the sign of the β-dependence backwards — which
    `beta_reduction.py` names as the specific error the envelope's swapped
    endpoints come from.
    """
    with plt.rc_context(BLOG_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
        for ax, model in zip(axes, ("sa", "usa")):
            for j, n in enumerate(THEORY_NS):
                for i, beta in enumerate(THEORY_BETAS):
                    t, g = integrate_gamma(n, beta, t_max=12.0, dt=1e-3,
                                           model=model)
                    ax.plot(t, g, color=MODEL_ODE_COLORS[model],
                            alpha=0.35 + 0.6 * i / (len(THEORY_BETAS) - 1),
                            ls="-" if j == 0 else "--", lw=1.9,
                            label=(f"β = {beta:g}" if j == 0 else None))
                    t_star = time_to_threshold(t, g, 0.9)
                    if np.isfinite(t_star) and t_star < 12:
                        ax.plot([t_star], [0.9], marker="|", ms=9,
                                color=MODEL_ODE_COLORS[model])
            reference_line(ax, 0.9, "γ = 0.9 — the collapse threshold",
                           side="left")
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("t (ODE time)")
            ax.set_title(f"({model.upper()})  —  solid n = {THEORY_NS[0]}, "
                         f"dashed n = {THEORY_NS[1]}")
            ax.legend(loc="lower right", fontsize=8.5, title="β")
        axes[0].set_ylabel("$\\gamma_\\beta(t)$")
        fig.suptitle("T1 · the null Phase 1c compares against", fontsize=12.5)
        caption(fig, "A sigmoid asymptotic to 1. Once it passes ~0.95 the "
                     "vertical residual has almost no dynamic range left — "
                     "which is the whole reason `time_residual_curve` exists "
                     "(status-1c finding 3, drawn in T6).", y=0.0)
    return save_figure(fig, out_dir, "gamma_family")


# ---------------------------------------------------------------------------
# T2 — the paper's own table
# ---------------------------------------------------------------------------

def _t2_collapse_time_table(out_dir: Path) -> Path:
    """
    t* over (n, β) for both models at both thresholds, as a heatmap.

    `MATH.md` §3.2's table, which `gamma_ode.py` reproduces to a maximum
    absolute deviation of 0.005 — the phase's single strongest validation
    claim, and one that has only ever existed as prose. Cell values are
    printed as well as coloured: the point is the numbers, and the colour is
    there to make the SA/USA divergence at large β and small n visible at a
    glance.
    """
    rows = collapse_time_table(ns=THEORY_NS, betas=THEORY_BETAS,
                               targets=(0.5, 0.9))
    with plt.rc_context(BLOG_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), sharex=True,
                                 sharey=True)
        for r, target in enumerate((0.5, 0.9)):
            for c, model in enumerate(("sa", "usa")):
                ax = axes[r, c]
                grid = np.full((len(THEORY_NS), len(THEORY_BETAS)), np.nan)
                for row in rows:
                    i = THEORY_NS.index(int(row["n"]))
                    j = THEORY_BETAS.index(float(row["beta"]))
                    grid[i, j] = row.get(f"{model}_t{target}", np.nan)
                im = ax.imshow(grid, aspect="auto", origin="lower",
                               cmap=SEQ_CMAP, interpolation="nearest")
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        v = grid[i, j]
                        ax.text(j, i, "∞" if not np.isfinite(v) else f"{v:.2f}",
                                ha="center", va="center", fontsize=9,
                                color="#111827")
                ax.set_xticks(range(len(THEORY_BETAS)))
                ax.set_xticklabels([f"β={b:g}" for b in THEORY_BETAS])
                ax.set_yticks(range(len(THEORY_NS)))
                ax.set_yticklabels([f"n={n}" for n in THEORY_NS])
                ax.grid(False)
                ax.set_title(f"({model.upper()})  γ = {target}", fontsize=10.5)
        fig.suptitle("T2 · collapse time t*(n, β) — MATH.md §3.2 reproduced "
                     "to 0.005", fontsize=12.5)
        caption(fig, "Two facts the table carries: at n = 467 the collapse "
                     "time is short and nearly β-independent, so \"how much "
                     "time would this network need\" has one answer; and SA "
                     "and USA separate by a factor of ten at n = 20, β = 5, "
                     "which is exactly the corner the paper's metastability "
                     "numerics sit in.", y=0.0)
    return save_figure(fig, out_dir, "collapse_time_table")


# ---------------------------------------------------------------------------
# T3 — why the envelope's endpoints swap
# ---------------------------------------------------------------------------

def _t3_beta_monotonicity(out_dir: Path) -> Path:
    """
    gamma at a fixed t against β, for several n, both models.

    `beta_reduction.py` establishes monotonicity numerically over 984,246
    grid points per model — (SA) decreasing with zero violations, (USA)
    increasing — and that result is what licenses bracketing the null by the
    per-head β range instead of choosing a reduction. This is that result at
    one t, which is as much as a figure can honestly show, and the caption
    says so.
    """
    t_fixed = 3.0
    betas = np.geomspace(0.1, 10.0, 24)
    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        for model in ("sa", "usa"):
            for j, n in enumerate((20, 128, 467)):
                g_at = []
                for b in betas:
                    t, g = integrate_gamma(n, float(b), t_max=t_fixed * 1.05,
                                           dt=1e-3, model=model)
                    g_at.append(float(np.interp(t_fixed, t, g)))
                ax.plot(betas, g_at, color=MODEL_ODE_COLORS[model],
                        alpha=0.45 + 0.25 * j, lw=2.1,
                        ls=("-", "--", ":")[j],
                        label=f"({model.upper()}) n = {n}")
        ax.set_xscale("log")
        ax.set_xlabel("β")
        ax.set_ylabel(f"$\\gamma_\\beta(t = {t_fixed:g})$")
        ax.set_ylim(0, 1.02)
        ax.set_title("T3 · monotone in β — in opposite directions")
        ax.legend(loc="best", fontsize=8.5, ncol=2)
        caption(fig, "(SA) decreases, (USA) increases. That is why the "
                     "envelope's upper edge is β_min under SA and β_max under "
                     "USA, and why using the surrogate as a stand-in gets the "
                     "direction of the β-dependence backwards rather than "
                     "merely the magnitude. One t here; the phase's own check "
                     "is a 984,246-point grid.")
    return save_figure(fig, out_dir, "beta_monotonicity")


# ---------------------------------------------------------------------------
# T4 — where P-H1 has any content
# ---------------------------------------------------------------------------

def _t4_wendel_surface(out_dir: Path) -> Path:
    """
    Wendel's probability over (n, d), with our prompt range and d = 1024.

    Theorem 6.7 gives exactly 1 whenever d > n. The whole sweep lives in
    that region, which is why P-H1 is registered in the direction that is
    boring if true — and why the margin, not the boolean, is what E reports.
    The transition band is narrow and sits at n ≈ 2d, so this figure is also
    the answer to "how much longer would a prompt have to be before the
    boolean meant anything".
    """
    ns = np.unique(np.geomspace(2, 8192, 120).astype(int))
    ds = (16, 64, 256, PYTHIA_D)
    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for i, d in enumerate(ds):
            p = [wendel_probability(int(n), d) for n in ns]
            ax.plot(ns, p, lw=2.2, color=SEQ_CMAP(0.35 + 0.6 * i / (len(ds) - 1)),
                    label=f"d = {d}" + (" (Pythia-410M)" if d == PYTHIA_D else ""))
            ax.plot([2 * d], [wendel_probability(2 * d, d)], marker="o", ms=5,
                    color=SEQ_CMAP(0.35 + 0.6 * i / (len(ds) - 1)))
        ax.axvspan(*PROMPT_N_RANGE, color="#FEF3C7", zorder=0,
                   label=f"our prompts (n = {PROMPT_N_RANGE[0]}–"
                         f"{PROMPT_N_RANGE[1]})")
        ax.set_xscale("log")
        ax.set_xlabel("n (tokens)")
        ax.set_ylabel("P(all in one open hemisphere)")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("T4 · Theorem 6.7 — probability 1 wherever d > n")
        ax.legend(loc="lower left", fontsize=8.5)
        caption(fig, "The dots mark n = 2d, where the probability passes ½. "
                     "Every prompt in the sweep sits on the flat part at 1, "
                     "so an infeasible layer would be a statement about what "
                     "the network did to the cloud, not about sampling.")
    return save_figure(fig, out_dir, "wendel_surface")


# ---------------------------------------------------------------------------
# T5 — what Q_k actually weights
# ---------------------------------------------------------------------------

def _t5_gegenbauer_kernels(out_dir: Path) -> Path:
    """
    Normalized Gegenbauer polynomials at d = 3 and d = 1024.

    Q_k is a weighted average of C_k(<x_i, x_j>) over all pairs, so what the
    design test is sensitive to is entirely the shape of these kernels. At
    high d they are flat across most of the inner-product range and steep
    only near ±1 — so Q_k at d = 1024 is dominated by the near-parallel and
    near-antipodal pairs, and a cloud can be reshaped substantially in the
    middle of the range without moving it. That is worth knowing before
    reading F.
    """
    t = np.linspace(-1, 1, 601)
    with plt.rc_context(BLOG_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
        for ax, d in zip(axes, (3, PYTHIA_D)):
            for k in range(1, 5):
                ax.plot(t, gegenbauer_normalized(t, k, d), lw=2.1,
                        color=degree_color(k, 4), label=f"k = {k}")
            ax.axhline(0.0, color="#374151", lw=1.0)
            ax.set_xlabel("inner product ⟨x_i, x_j⟩")
            ax.set_title(f"d = {d}")
            ax.set_ylim(-1.1, 1.1)
        axes[0].set_ylabel("$C_k^\\lambda(t)$, normalized to $C_k(1) = 1$")
        axes[1].legend(loc="upper left", fontsize=9)
        fig.suptitle("T5 · what the design test is sensitive to", fontsize=12.5)
        caption(fig, "At d = 1024 the kernels are flat through the middle and "
                     "steep only near ±1: Q_k is dominated by the "
                     "near-parallel and near-antipodal pairs. The recurrence "
                     "in `design_test.py` is what makes these computable at "
                     "all at this d — scipy's coefficient form overflows.",
                y=0.0)
    return save_figure(fig, out_dir, "gegenbauer_kernels")


# ---------------------------------------------------------------------------
# T6 — the argument for the time-domain residual
# ---------------------------------------------------------------------------

def _t6_sigmoid_compression(out_dir: Path) -> Path:
    """
    What a fixed lag in TIME looks like as a vertical residual, by γ level.

    The argument status-1c finding 3 makes in prose: a trajectory running a
    fixed amount of integration time behind the null shows a vertical gap
    that collapses toward zero as the null saturates, so late in the stack
    two visibly different dynamics both read residual ≈ 0. The same lag
    read in the time domain is, by construction, constant. Left panel: the
    null with three equal time lags marked. Right: the vertical gap they
    produce, against the null level.
    """
    n, beta, lag = 467, 1.0, 0.5
    t, g = integrate_gamma(n, beta, t_max=12.0, dt=1e-3)
    g_lag = np.interp(t - lag, t, g, left=g[0])
    gap = g - g_lag

    with plt.rc_context(BLOG_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        ax = axes[0]
        ax.plot(t, g, color="#12406F", lw=2.4, label="$\\gamma_\\beta(t)$")
        ax.plot(t, g_lag, color=CATEGORICAL[1], lw=2.0, ls="--",
                label=f"the same curve, {lag:g} time units behind")
        for t_mark in (1.0, 2.5, 5.0):
            i = int(np.searchsorted(t, t_mark))
            ax.vlines(t_mark, g_lag[i], g[i], color="#374151", lw=1.4)
            ax.annotate(f"{g[i] - g_lag[i]:+.3f}", xy=(t_mark, g[i]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8,
                        color="#374151")
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("t")
        ax.set_ylabel("γ")
        ax.legend(loc="lower right", fontsize=8.5)
        ax.set_title("one lag, three depths")

        ax2 = axes[1]
        ax2.plot(g, gap, color="#12406F", lw=2.4)
        ax2.axhline(0.0, color="#374151", lw=1.0)
        reference_line(ax2, 0.0, "", side="left")
        ax2.axvspan(0.95, 1.0, color="#FEF3C7", zorder=0,
                    label="null > 0.95 — no range left")
        ax2.set_xlabel("null level γ at this layer")
        ax2.set_ylabel(f"vertical residual from a {lag:g}-unit lag")
        ax2.set_xlim(0, 1)
        ax2.legend(loc="upper right", fontsize=8.5)
        ax2.set_title("the same lag, as a vertical residual")

        fig.suptitle("T6 · why the residual is also reported in the time "
                     "domain", fontsize=12.5)
        caption(fig, f"A constant {lag:g}-unit lag reads as a vertical "
                     f"residual of {np.nanmax(gap):.3f} at the steep part and "
                     f"under 0.01 near the ceiling. In the time domain it "
                     f"reads {lag:g} everywhere, by construction — which is "
                     f"the resolution status-1c finding 3 is about.", y=0.0)
    return save_figure(fig, out_dir, "sigmoid_compression")


# ---------------------------------------------------------------------------
# T7 — the effect-size floor P-S1 was missing
# ---------------------------------------------------------------------------

def _t7_random_band_by_degree(out_dir: Path, cheap: bool = False) -> Path:
    """
    The 2σ band of the Q_k ratio against degree, at several m.

    The only figure in this package that simulates. It draws the measurement
    `centroids.py` reports — bands of 0.164 / 0.015 / 0.002 at k = 1, 2, 3 —
    and the reason it is worth a figure is that the raw ratios point the
    other way: the simplex gives Q_2 ratio 0.977, which looks like no
    signal, while the band at k = 2 is 0.015, so a deviation of 0.023 is
    outside it. Higher degrees are MORE sensitive in relative terms, and a
    fixed absolute tolerance is wrong in a different direction at every one.
    """
    d, t_max = 256, 3
    n_trials = 60 if cheap else 200
    ms = (8, 32, 128)
    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        ks = np.arange(1, t_max + 1)
        for i, m in enumerate(ms):
            band = random_band(m, d, t_max=t_max, n_trials=n_trials)["band"]
            ax.plot(ks, band, marker="o", ms=6, lw=2.2,
                    color=SEQ_CMAP(0.35 + 0.55 * i / (len(ms) - 1)),
                    label=f"m = {m} centroids")
        ax.set_yscale("log")
        ax.set_xticks(ks)
        ax.set_xlabel("Gegenbauer degree k")
        ax.set_ylabel("2σ band of the $Q_k$ ratio")
        ax.set_title("T7 · the effect-size floor P-S1 was registered without")
        ax.legend(loc="best", fontsize=8.5)
        caption(fig, (
            f"d = {d}, {n_trials} trials"
            + (" (--cheap: fewer trials than the phase's own 200, so the "
               "bands are noisier here than the measured 0.164 / 0.015 / "
               "0.002)" if cheap else "")
            + ". Both the deviation and the noise shrink with k, at different "
              "rates — which is why P-S1 is adjudicated on the banded test "
              "and never on a fixed tolerance."))
    return save_figure(fig, out_dir, "random_band_by_degree")
