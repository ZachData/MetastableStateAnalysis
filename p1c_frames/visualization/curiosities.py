"""
p1c_frames/visualization/curiosities.py — the speculative half.

Eleven figures (X1-X11 in FIGURES-1c.md). None is a verdict figure and none
is in the falsification table. They exist because Phase 1c's unit of work is
the whole trajectory, which makes trajectory-shaped questions cheap to look
at, and looking is how the next question gets found. **A figure here that
shows nothing is a result worth having drawn once**, and each carries a
one-line "what would be interesting here" note in its own docstring so that
outcome is recognizable rather than mistaken for a broken figure.

Two of them exist because a quantity in this phase turns out to be an axis
nobody has drawn:

  X1/X2  T_eff is the network's own clock, and every Phase 1 figure plots
         against layer index instead. If the clock runs unevenly, "layer 12"
         is not half way through anything.
  X3     the observed trajectory has a phase portrait, and so does the ODE.
         Putting them on one plane asks whether the network is on the
         paper's flow, off it, or on a different one — which is the
         qualitative version of the residual.

Seven are per-run; four (X5, X6, X9, and the pooled half of X8) need every
run and are drawn by `generate_curiosity_cross_figures`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from p1c_frames.gamma_ode import collapse_time, integrate_gamma

from .loaders import Run, record_field, record_matrix, records
from .style import (
    BLOG_STYLE, CATEGORICAL, DEPTH_CMAP, INVALID_COLOR, NULL_LINE,
    RESIDUAL_CMAP, SEQ_CMAP, caption, depth_axis, model_color, no_data,
    reference_line, residual_norm, save_figure,
)

__all__ = ["generate_curiosity_figures", "generate_curiosity_cross_figures"]


def _skip(name: str, reason: str) -> None:
    print(f"    skip {name}: {reason}")


def generate_curiosity_figures(run: Run, out_dir: Path) -> List[Path]:
    """The per-run curiosities: X1-X4, X7, X8, X10, X11."""
    paths: List[Optional[Path]] = []
    if run.has("A"):
        paths.append(_x1_depth_clock(run, out_dir))
        paths.append(_x2_teff_budget(run, out_dir))
    else:
        _skip("depth_clock / teff_budget", "no A block — no clock to draw")

    if run.has("A") and run.has("B"):
        paths.append(_x3_phase_portrait(run, out_dir))
        paths.append(_x4_field_feedback(run, out_dir))
    else:
        _skip("phase_portrait / field_feedback", "needs both A and B")

    if run.has("C"):
        paths.append(_x7_sink_gallery(run, out_dir))
    if run.has("E") and run.has("B"):
        paths.append(_x8_margin_vs_clustering(run, out_dir))
    if run.block("B.beta_reduction"):
        paths.append(_x10_beta_fan(run, out_dir))
    paths.append(_x11_run_fingerprint(run, out_dir))
    return [p for p in paths if p is not None]


def generate_curiosity_cross_figures(runs: Sequence[Run],
                                     out_dir: Path) -> List[Path]:
    """The pooled curiosities: X5, X6, X9."""
    runs = list(runs)
    if not runs:
        return []
    paths: List[Optional[Path]] = [
        _x5_residual_barcode(runs, out_dir),
        _x6_t_star_landscape(runs, out_dir),
        _x9_design_vs_residual(runs, out_dir),
    ]
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# X1 — is depth a fair time axis?
# ---------------------------------------------------------------------------

def _x1_depth_clock(run: Run, out_dir: Path) -> Path:
    """
    Layer index against T_eff, with a uniform clock as the diagonal.

    Interesting if it bends: every Phase 1 figure plots against layer index,
    which is a fair proxy for integration time only if the network spends
    the same amount of it per block. A clock that runs fast early and slow
    late means the mid-network plateau occupies far less of the ODE's time
    than of the stack, and any "the plateau lasts eight layers" claim is a
    statement about depth rather than about the dynamics.
    """
    h = run.series("A.h_calibrated")
    cum = np.concatenate([[0.0], np.nancumsum(h)]) if h.size else np.zeros(0)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        if cum.size < 2:
            no_data(ax, "no step sizes for this run")
        else:
            x = np.arange(cum.size)
            uniform = cum[-1] * x / max(x[-1], 1)
            ax.plot(x, cum, color="#12406F", lw=2.6, marker="o", ms=3.5,
                    label="the network's clock")
            ax.plot(x, uniform, color="#6B7280", ls=":", lw=1.6,
                    label="a uniform clock")
            ax.fill_between(x, uniform, cum, color=CATEGORICAL[0], alpha=0.12)
            lead = cum - uniform
            i = int(np.nanargmax(np.abs(lead)))
            ax.annotate(f"largest deviation at layer {i}: {lead[i]:+.3f}",
                        xy=(i, cum[i]), xytext=(8, -18),
                        textcoords="offset points", fontsize=8.5,
                        color=CATEGORICAL[1],
                        arrowprops=dict(arrowstyle="->", color=CATEGORICAL[1],
                                        lw=1.0))
            ax.set_xlabel("layer")
            ax.set_ylabel("$T_{\\rm eff}$ elapsed")
            ax.legend(loc="upper left", fontsize=8.5)
        ax.set_title(f"X1 · is depth a fair time axis? — {run.label}")
        caption(fig, "If this bends, a plateau measured in layers is not a "
                     "plateau measured in the ODE's time, and every "
                     "depth-axis figure in Phase 1 inherits the distortion.")
    return save_figure(fig, out_dir, "depth_clock")


# ---------------------------------------------------------------------------
# X2 — where the budget goes
# ---------------------------------------------------------------------------

def _x2_teff_budget(run: Run, out_dir: Path) -> Path:
    """
    Each layer's share of the total integration time, sorted and cumulative.

    Interesting if it is concentrated: a stack whose top three blocks carry
    half the integration time is doing something quite different from one
    that spends evenly, and it would mean T_eff is a statement about a few
    layers rather than about depth. The Lorenz-style cumulative curve is
    drawn against the even-split diagonal so concentration is visible
    without computing a Gini.
    """
    h = run.series("A.h_calibrated")
    with plt.rc_context(BLOG_STYLE):
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
        if not np.isfinite(h).any():
            no_data(ax, "no step sizes")
            no_data(ax2, "")
        else:
            total = float(np.nansum(h))
            share = h / total if total else h
            colors = DEPTH_CMAP(np.linspace(0.08, 0.95, h.size))
            ax.bar(np.arange(h.size), share * 100, color=colors, width=0.8)
            ax.set_xlabel("layer boundary ℓ → ℓ+1")
            ax.set_ylabel("share of $T_{\\rm eff}$ (%)")
            ax.set_title("per layer")

            order = np.argsort(share)[::-1]
            cum = np.nancumsum(share[order]) * 100
            frac = np.arange(1, cum.size + 1) / cum.size * 100
            ax2.plot(frac, cum, color="#12406F", lw=2.4, marker="o", ms=3.5)
            ax2.plot([0, 100], [0, 100], color="#6B7280", ls=":", lw=1.4)
            half = int(np.searchsorted(cum, 50.0)) + 1
            ax2.annotate(f"half the clock in {half} of {cum.size} layers",
                         xy=(frac[min(half - 1, frac.size - 1)], 50),
                         xytext=(10, -22), textcoords="offset points",
                         fontsize=8.5, color=CATEGORICAL[1],
                         arrowprops=dict(arrowstyle="->", color=CATEGORICAL[1],
                                         lw=1.0))
            ax2.set_xlabel("layers, ranked by step size (%)")
            ax2.set_ylabel("cumulative share of $T_{\\rm eff}$ (%)")
            ax2.set_title("concentration")
        fig.suptitle(f"X2 · where the integration budget goes — {run.label}",
                     fontsize=12)
    return save_figure(fig, out_dir, "teff_budget")


# ---------------------------------------------------------------------------
# X3 — the trajectory as a flow
# ---------------------------------------------------------------------------

def _x3_phase_portrait(run: Run, out_dir: Path) -> Path:
    """
    (ip_mean, d ip_mean / d T_eff) for the observed trajectory, with the
    ODE's own (γ, γ̇) curve behind it.

    Interesting whichever way it comes out. The null's phase portrait is a
    single arch — γ̇ is a function of γ alone, which is what makes (6.9) an
    ODE in one variable. If the observed points lie on that arch, the
    network is running the paper's flow at its own rate. If they lie
    consistently BELOW it, the network is on a slower flow — the same shape,
    damped. If they leave the arch entirely, the network's dynamics are not
    a rescaling of the paper's at all, and the residual is measuring a
    different vector field rather than a different speed. That distinction
    is exactly what design-1c says the calibrated step makes visible.
    """
    ip = run.series("B.ip_mean")
    t = run.series("B.t_eff_grid")
    k = min(ip.size, t.size)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        if k < 3:
            no_data(ax, "not enough layers for a phase portrait")
        else:
            dt = np.diff(t[:k])
            dg = np.diff(ip[:k])
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = dg / np.where(dt > 0, dt, np.nan)
            mid = 0.5 * (ip[:k][1:] + ip[:k][:-1])

            n = run.n_tokens or 2
            beta = run.scalar("B", "beta_median", run.beta)
            if np.isfinite(beta) and n >= 2:
                tt, gg = integrate_gamma(int(n), float(beta), t_max=12.0,
                                         dt=1e-3)
                null_rate = np.gradient(gg, tt)
                ax.plot(gg, null_rate, label="the ODE's own flow $\\dot\\gamma"
                                             "(\\gamma)$", **NULL_LINE)

            colors = DEPTH_CMAP(np.linspace(0.08, 0.95, mid.size))
            ax.plot(mid, rate, color="#9CA3AF", lw=1.0, zorder=2)
            ax.scatter(mid, rate, c=colors, s=54, edgecolor="white",
                       linewidth=0.8, zorder=4, label="observed trajectory")
            ax.axhline(0.0, color="#374151", lw=1.0, zorder=1)
            ax.set_xlabel("mean pairwise inner product")
            ax.set_ylabel("rate of change per unit $T_{\\rm eff}$")
            ax.legend(loc="best", fontsize=8.5)
        ax.set_title(f"X3 · on the paper's flow, or a different one? — "
                     f"{run.label}")
        caption(fig, "On the arch: the paper's dynamics at the network's own "
                     "rate. Below it: damped — slower integration, which the "
                     "calibrated step already absorbs. Off it: a different "
                     "vector field, which is what the residual is for.")
    return save_figure(fig, out_dir, "phase_portrait")


# ---------------------------------------------------------------------------
# X4 — the feedback loop
# ---------------------------------------------------------------------------

def _x4_field_feedback(run: Run, out_dir: Path) -> Path:
    """
    Field magnitude against clustering, one point per layer.

    Interesting because it should be increasing, and by how much. ‖X‖ is
    bounded by 1 with equality only for a fully collapsed cloud, so as the
    cloud clusters the field that clusters it gets stronger — collapse is an
    instability, not a drift. A trained network that holds ‖X‖ down while
    ip_mean rises is resisting in a way the residual measures only
    indirectly, and it would show up here as a flat cloud where the null
    would give a rising one.
    """
    fmag = run.series("A.field_mag")
    ip = run.series("B.ip_mean")
    k = min(fmag.size, ip.size)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        if k < 2:
            no_data(ax, "needs both field_mag (A) and ip_mean (B)")
        else:
            colors = DEPTH_CMAP(np.linspace(0.08, 0.95, k))
            ax.plot(ip[:k], fmag[:k], color="#9CA3AF", lw=1.0, zorder=2)
            sc = ax.scatter(ip[:k], fmag[:k], c=np.arange(k), cmap=DEPTH_CMAP,
                            s=58, edgecolor="white", linewidth=0.8, zorder=4)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("layer")
            reference_line(ax, 1.0, "‖X‖ = 1 — a fully collapsed cloud",
                           side="left")
            ax.set_xlabel("mean pairwise inner product")
            ax.set_ylabel("mean ‖X(x)‖")
            ax.set_ylim(0, 1.05)
        ax.set_title(f"X4 · the loop that makes collapse an instability — "
                     f"{run.label}")
        caption(fig, "More clustering means a stronger field means more "
                     "clustering. A flat cloud here is a network holding the "
                     "field down while the cloud tightens — resistance the "
                     "residual only sees second-hand.")
    return save_figure(fig, out_dir, "field_feedback")


# ---------------------------------------------------------------------------
# X7 — which layers are sink-dominated
# ---------------------------------------------------------------------------

def _x7_sink_gallery(run: Run, out_dir: Path) -> Path:
    """
    norm_max/median against sink_ratio, one point per layer.

    Interesting if the sink-dominated layers cluster in depth. C2 draws both
    quantities against layer; this draws them against each other, so a
    diagonal band means "the more outlying the norms, the more the raw rank
    is just counting them" — the mechanism — and a vertical band means the
    ratio moves for some other reason, which would be worth chasing.
    """
    panels = records(run, "C", "panels")
    ratio = record_field(panels, "sink_ratio")
    nmax = record_field(panels, "norm_max_over_median")

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        if not np.isfinite(ratio).any():
            no_data(ax, "no rank panels for this run")
        else:
            sc = ax.scatter(nmax, ratio, c=np.arange(ratio.size),
                            cmap=DEPTH_CMAP, s=60, edgecolor="white",
                            linewidth=0.8, zorder=4)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("layer")
            reference_line(ax, 1.0, "raw rank IS the norm distribution",
                           side="left")
            ax.set_xlabel("max / median token norm")
            ax.set_ylabel("sink_ratio (raw rank / norm-PR)")
        ax.set_title(f"X7 · sink domination by layer — {run.label}")
        caption(fig, "A diagonal band is the mechanism working as expected. A "
                     "vertical one means the ratio moves for a reason the "
                     "norm outliers do not explain.")
    return save_figure(fig, out_dir, "sink_gallery")


# ---------------------------------------------------------------------------
# X8 — does clustering buy containment?
# ---------------------------------------------------------------------------

def _x8_margin_vs_clustering(run: Run, out_dir: Path) -> Path:
    """
    Cone margin against ip_mean, one point per layer.

    Interesting because the sign is not obvious. Clustering pulls tokens
    together, which should make a common witness direction easier to find
    and push the margin up. But Lemma 6.4's hypothesis is about containment
    in a halfspace, not about tightness, and Phase 1 finds mid-network
    de-clustering; if the margin falls where clustering falls, the two are
    one phenomenon and E is partly re-measuring B. If they move
    independently, the cone condition is carrying information neither B nor
    Phase 1b has.
    """
    margins = run.series("E.margins")
    if not margins.size:
        margins = record_field(records(run, "E"), "margin")
    ip = run.series("B.ip_mean")
    k = min(margins.size, ip.size)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        if k < 2:
            no_data(ax, "needs both E margins and B ip_mean")
        else:
            sc = ax.scatter(ip[:k], margins[:k], c=np.arange(k),
                            cmap=DEPTH_CMAP, s=60, edgecolor="white",
                            linewidth=0.8, zorder=4)
            ax.plot(ip[:k], margins[:k], color="#9CA3AF", lw=1.0, zorder=2)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("layer")
            ax.axhline(0.0, color="#374151", lw=1.2)
            corr = _corr(ip[:k], margins[:k])
            ax.set_xlabel("mean pairwise inner product")
            ax.set_ylabel("hull margin")
            ax.annotate(f"corr = {corr:+.2f}" if np.isfinite(corr) else
                        "corr = n/a", xy=(0.02, 0.95), xycoords="axes fraction",
                        fontsize=9, color="#4B5563")
        ax.set_title(f"X8 · does clustering buy containment? — {run.label}")
        caption(fig, "If these track, E is partly re-measuring B. If they do "
                     "not, the cone condition carries information neither B "
                     "nor Phase 1b has.")
    return save_figure(fig, out_dir, "margin_vs_clustering")


# ---------------------------------------------------------------------------
# X10 — the undecided choice, drawn
# ---------------------------------------------------------------------------

def _x10_beta_fan(run: Run, out_dir: Path) -> Path:
    """
    Every β reduction against the per-head range.

    Interesting as a picture of an open decision. `reduce_beta` offers five
    reductions and the phase refuses to pick one; this draws all five inside
    the min-max range so the size of the disagreement is visible next to the
    size of the spread it is drawn from. A run where mean, median and
    attention-weighted sit on top of each other is one where the decision is
    nearly free; one where they scatter across the range is a run whose
    residual has an unstated error bar the size of the fan.
    """
    rr = run.block("B.beta_reduction")
    values = run.block("B.beta_reduction.values")
    if not values:
        values = rr.get("values") or {}
    order = [k for k in ("min", "mean", "median", "attention_weighted", "max")
             if k in values]

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        b_min, b_max = rr.get("beta_min"), rr.get("beta_max")
        if b_min is not None and b_max is not None:
            ax.axvspan(float(b_min), float(b_max), color="#D3E3F3", alpha=0.55,
                       label=f"per-head range over {rr.get('n_finite', '?')} "
                             f"heads")
        for i, key in enumerate(order):
            try:
                v = float(values[key])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                ax.text(0.01, i, f"{key}: not computable (needs weights)",
                        transform=ax.get_yaxis_transform(), fontsize=8.5,
                        color=INVALID_COLOR, va="center")
                continue
            ax.plot([v], [i], marker="D", ms=9, color=CATEGORICAL[0], zorder=5)
            ax.annotate(f"{v:.3f}", xy=(v, i), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=8.5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel("$\\beta_{\\rm eff}$")
        ax.legend(loc="lower right", fontsize=8.5)
        ax.set_title(f"X10 · the reduction nobody has chosen — {run.label}")
        n_drop = rr.get("n_dropped")
        caption(fig, (
            f"Spread across reductions: {_f(rr.get('reduction_spread'))}; "
            f"{n_drop} head(s) dropped for a failed regression. A head whose "
            f"β_eff regression failed is not a head with β = 0."))
    return save_figure(fig, out_dir, "beta_fan")


# ---------------------------------------------------------------------------
# X11 — the whole per-layer table as one image
# ---------------------------------------------------------------------------

def _x11_run_fingerprint(run: Run, out_dir: Path) -> Path:
    """
    Layer × metric heatmap, each metric z-scored down its own column.

    Not a result — a way of spotting which layers are odd before deciding
    what to plot. Every per-layer series this run carries, on one image, in
    units of its own standard deviation. A vertical stripe is a layer that
    is unusual in many quantities at once, which is the thing worth chasing;
    a horizontal one is a metric with an outlier, which is usually a
    reminder that its scale is not what you assumed.
    """
    series = {}
    for key, label in (("B.ip_mean", "ip_mean"),
                       ("B.residual", "residual"),
                       ("B.gamma_null", "gamma_null"),
                       ("B.time_domain.time_residual", "time_residual"),
                       ("B.beta_per_layer", "beta"),
                       ("E.margins", "cone margin")):
        v = run.series(key)
        if v.size and np.isfinite(v).any():
            series[label] = v

    for key, label in (("A.h_calibrated", "h_calibrated"),
                       ("A.field_mag", "field_mag")):
        v = run.series(key)
        if v.size and np.isfinite(v).any():
            # Transition-indexed: pad to layer count so the depth axis of
            # this image means the same thing in every row.
            series[label] = np.concatenate([v, [np.nan]])

    panels = records(run, "C", "panels")
    for key in ("shannon_raw", "shannon_normed", "sink_ratio"):
        v = record_field(panels, key)
        if v.size and np.isfinite(v).any():
            series[key] = v

    f_recs = records(run, "F")
    if f_recs:
        v = record_field(f_recs, "sharp_score", n=run.n_layers)
        if np.isfinite(v).any():
            series["sharp_score"] = v

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.6, 0.42 * max(len(series), 3) + 2.4))
        if not series:
            no_data(ax, "this run carries no per-layer series")
        else:
            n_layers = max(v.size for v in series.values())
            grid = np.full((len(series), n_layers), np.nan)
            for i, (label, v) in enumerate(series.items()):
                z = (v - np.nanmean(v)) / (np.nanstd(v) or 1.0)
                grid[i, :z.size] = z
            im = ax.imshow(grid, aspect="auto", origin="upper",
                           cmap=RESIDUAL_CMAP, norm=residual_norm(grid, 3.0),
                           interpolation="nearest")
            ax.set_yticks(range(len(series)))
            ax.set_yticklabels(list(series), fontsize=9)
            ax.set_xlabel("layer")
            ax.grid(False)
            cb = fig.colorbar(im, ax=ax)
            cb.set_label("z-score within the row")
        ax.set_title(f"X11 · one run's whole per-layer table — {run.label}")
        caption(fig, "A vertical stripe is a layer that is unusual in many "
                     "quantities at once. A horizontal one is a metric with "
                     "an outlier.")
    return save_figure(fig, out_dir, "run_fingerprint")


# ---------------------------------------------------------------------------
# X5 — the sweep as one image
# ---------------------------------------------------------------------------

def _x5_residual_barcode(runs: Sequence[Run], out_dir: Path) -> Optional[Path]:
    """
    Layer × run heatmap of the residual, diverging at zero.

    Interesting as a shape: if resistance is a depth phenomenon the image
    has vertical structure, if it is a model phenomenon it has horizontal
    structure, and if it is neither the image is noise — which would itself
    settle something.
    """
    rows = [(r, r.series("B.residual")) for r in runs]
    rows = [(r, v) for r, v in rows if v.size and np.isfinite(v).any()]
    if not rows:
        _skip("residual_barcode", "no run carries a residual series")
        return None

    n_layers = max(v.size for _, v in rows)
    grid = np.full((len(rows), n_layers), np.nan)
    for i, (_, v) in enumerate(rows):
        grid[i, :v.size] = v

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(10.2, 0.38 * len(rows) + 2.6))
        im = ax.imshow(grid, aspect="auto", origin="upper", cmap=RESIDUAL_CMAP,
                       norm=residual_norm(grid), interpolation="nearest")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r.stem for r, _ in rows], fontsize=8)
        ax.set_xlabel("layer")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("residual (blue = behind the null = resistance)")
        ax.set_title("X5 · the whole sweep's residual, as one image")
        caption(fig, "Runs of different depths are drawn on a shared axis and "
                     "padded, not stretched: layer 12 is layer 12 in every "
                     "row.")
    return save_figure(fig, out_dir, "residual_barcode")


# ---------------------------------------------------------------------------
# X6 — t* is not one number
# ---------------------------------------------------------------------------

def _x6_t_star_landscape(runs: Sequence[Run], out_dir: Path) -> Path:
    """
    t*(n, β) as a contour field, with every run at its own (n, β_eff).

    status-1c open item 4 as a picture: t* is n-dependent and the prompts
    span 20-512 tokens, so a pooled t* would compare short-prompt runs
    against a collapse time that is not theirs. This draws the surface the
    runs are scattered on, which also makes visible how much of the sweep's
    spread in "distance to t*" is prompt length rather than model.
    """
    # 7x7 rather than a finer grid: each cell is a step-halved ODE solve, and
    # this is a reference surface to place runs on, not a table to read
    # values off — T2 is where the numbers live.
    ns = np.unique(np.geomspace(16, 600, 7).astype(int))
    betas = np.geomspace(0.3, 6.0, 7)
    grid = np.zeros((ns.size, betas.size))
    for i, n in enumerate(ns):
        for j, b in enumerate(betas):
            grid[i, j] = collapse_time(int(n), float(b), target=0.9)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        cs = ax.contourf(betas, ns, grid, levels=14, cmap=SEQ_CMAP)
        lines = ax.contour(betas, ns, grid, levels=7, colors="#374151",
                           linewidths=0.7)
        ax.clabel(lines, inline=True, fontsize=7.5, fmt="%.1f")
        cb = fig.colorbar(cs, ax=ax)
        cb.set_label("t* — time to γ = 0.9, (SA)")

        for r in runs:
            n, b = r.n_tokens, r.beta
            if not (n and np.isfinite(b)):
                continue
            ax.plot([np.clip(b, betas[0], betas[-1])],
                    [np.clip(n, ns[0], ns[-1])], marker="o", ms=7,
                    color=model_color(r.model), mec="white", mew=1.0,
                    zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("β")
        ax.set_ylabel("n (tokens)")
        ax.grid(False)
        ax.set_title("X6 · every run sits on its own t*")
        caption(fig, "One dot per run at its measured (n, β_eff). Points "
                     "outside the grid are clamped to the edge, so a dot on "
                     "the boundary means the run is off the drawn range, not "
                     "at its limit.")
    return save_figure(fig, out_dir, "t_star_landscape")


# ---------------------------------------------------------------------------
# X9 — where P-S1 and P-γ1 meet
# ---------------------------------------------------------------------------

def _x9_design_vs_residual(runs: Sequence[Run],
                           out_dir: Path) -> Optional[Path]:
    """
    Q_1 ratio against the residual, one point per layer per run.

    Interesting because the two predictions have never been compared. P-S1
    says the trained model's centroids approach a sharp configuration; P-γ1
    says the trajectory departs from the identity-weight null. If the layers
    with the most negative residual are also the sharpest, resistance has a
    target geometry and the two predictions are one. If they are unrelated,
    the repulsive-limit story and the resistance story are about different
    things, which is worth knowing before either is written up.
    """
    xs, ys, cs = [], [], []
    for r in runs:
        recs = records(r, "F")
        resid = r.series("B.residual")
        if not recs or not resid.size:
            continue
        t_max = max((len(rec.get("Q_ratio") or []) for rec in recs), default=0)
        if not t_max:
            continue
        q1 = record_matrix(recs, "Q_ratio", t_max, n=r.n_layers)[:, 0]
        k = min(q1.size, resid.size)
        for i in range(k):
            if np.isfinite(q1[i]) and np.isfinite(resid[i]):
                xs.append(q1[i])
                ys.append(resid[i])
                cs.append(model_color(r.model))

    if not xs:
        _skip("design_vs_residual", "no run carries both an F block and a "
                                    "residual series")
        return None

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        ax.scatter(xs, ys, c=cs, s=42, alpha=0.75, edgecolor="white",
                   linewidth=0.6)
        ax.axhline(0.0, color="#374151", lw=1.2)
        reference_line(ax, 1.0, "i.i.d. uniform", axis="x")
        corr = _corr(np.array(xs), np.array(ys))
        ax.annotate(f"corr = {corr:+.2f}  (n = {len(xs)} layers)"
                    if np.isfinite(corr) else f"n = {len(xs)} layers",
                    xy=(0.02, 0.03), xycoords="axes fraction", fontsize=9,
                    color="#4B5563")
        ax.set_xlabel("$Q_1$ ratio — sharper to the left")
        ax.set_ylabel("residual — resistance downward")
        ax.set_title("X9 · are the sharp layers the resistant ones?")
        caption(fig, "Point colour is the model. The lower-left quadrant is "
                     "\"sharp and resisting\"; a cloud with no structure means "
                     "P-S1 and P-γ1 are about different things.")
    return save_figure(fig, out_dir, "design_vs_residual")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _f(v, fmt: str = "{:.4f}") -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return fmt.format(x) if np.isfinite(x) else "n/a"
