"""
p1c_frames/visualization/null_model.py — sub-experiment B, the residual.

Eight per-run figures (B1-B8 in FIGURES-1c.md). The organizing constraint is
design-1c's first rule: **the residual is the deliverable, the fit is not.**
A figure showing `ip_mean` and $\\gamma_\\beta$ lying pleasantly on top of
each other is the wrong figure — it invites the reader to grade the fit,
which is not the question. B1 therefore shades the gap and B2 plots the gap
alone on a zero-centred axis.

Four things this module is careful about, each because the phase's own docs
say the naive version misleads:

**Two nulls, always.** Theorem 6.8 assumes orthogonal initialization; real
embeddings carry a large common mode. The observed-matched null is drawn
alongside, and where the two disagree the disagreement is an anisotropy
effect and not resistance — a distinction the single-null version cannot
make (design-1c).

**The vertical residual loses its range and the time residual does not.**
Once the null passes ~0.95 both curves are against the ceiling and a
perturbation that visibly changed the dynamics reads +0.0000 (status-1c
finding 3). B3 draws the time-domain measure with equal billing rather than
as an appendix, and B2's caption says where its own range ran out.

**A NaN here is the strongest signal in the phase, not missing data.**
`time_residual` is NaN exactly where the observed inner product sits below
the null's own starting point — the network de-clustered past where it
began. B3 draws those layers as marked bands and counts them; a gap in the
line would render the loudest result as nothing at all.

**The envelope is a bracket, not an error bar.** B5 and B6 draw the β
envelope hatched and quote `envelope_verdict`'s own wording, because inside
the band the residual's SIGN depends on a reduction nobody has chosen —
which is exactly the case `run_1c` refuses to paper over with a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .loaders import Run
from .style import (
    BLOG_STYLE, CATEGORICAL, ENVELOPE_BAND, INVALID_COLOR, NULL_BAND,
    NULL_LINE, RESIDUAL_CMAP, UNREACHABLE_COLOR, caption, depth_axis,
    mark_nan_spans, no_data, reference_line, residual_norm, save_figure,
    verdict_box,
)

__all__ = ["generate_null_figures"]

OBSERVED = dict(color="#12406F", lw=2.6, zorder=4)
MATCHED = dict(color=CATEGORICAL[2], lw=1.8, ls="-.", zorder=3)


def _skip(name: str, reason: str) -> None:
    print(f"    skip {name}: {reason}")


def generate_null_figures(run: Run, out_dir: Path) -> List[Path]:
    """B1-B8 for one run."""
    paths: List[Optional[Path]] = []
    if not run.has("B"):
        _skip("null", next((m for m in run.missing if m.startswith("B ")),
                           "no B block"))
        return []

    ip = run.series("B.ip_mean")
    if not ip.size:
        _skip("null", "B block carries no ip_mean series")
        return []

    paths.append(_b1_residual_curve(run, out_dir))
    paths.append(_b2_residual_depth(run, out_dir))
    if run.has_series("B.time_domain.time_residual"):
        paths.append(_b3_time_residual(run, out_dir))
    else:
        _skip("time_residual", "no B.time_domain in this run")
    paths.append(_b4_trajectory_in_time(run, out_dir))

    if run.has_series("B.envelope_lower"):
        paths.append(_b5_beta_envelope(run, out_dir))
        paths.append(_b6_residual_bracket(run, out_dir))
    else:
        note = (run.text("B", "envelope_note") or run.text("B", "envelope_error")
                or "no per-head beta_eff (FIGURES-1c.md G4)")
        _skip("beta_envelope", note)
        _skip("residual_bracket", note)

    if run.block("B.collapse_fraction"):
        paths.append(_b7_collapse_fraction(run, out_dir))
    else:
        _skip("collapse_fraction", "no B.collapse_fraction in this run")
    paths.append(_b8_beta_fallback_audit(run, out_dir))
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# B1 — the central image
# ---------------------------------------------------------------------------

def _b1_residual_curve(run: Run, out_dir: Path) -> Path:
    """
    ip_mean with both nulls, and the gap between observed and null shaded.

    The shading is the subject. It is drawn signed — one tint where the
    network is behind the null (resistance) and another where it is ahead —
    so the sign convention is legible without reading the axis, and the
    convention string the artifact carries is printed underneath rather than
    restated in this module's own words.
    """
    ip = run.series("B.ip_mean")
    null = run.series("B.gamma_null")
    matched = run.series("B.gamma_null_matched")
    x = np.arange(ip.size)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        k = min(ip.size, null.size) if null.size else 0
        if k:
            ax.fill_between(x[:k], ip[:k], null[:k], where=ip[:k] <= null[:k],
                            color="#2A78D6", alpha=0.20, linewidth=0,
                            label="behind the null — resistance")
            ax.fill_between(x[:k], ip[:k], null[:k], where=ip[:k] > null[:k],
                            color="#D6483F", alpha=0.18, linewidth=0,
                            label="ahead of the null")
            ax.plot(x[:k], null[:k], label="$\\gamma_\\beta(T_{\\rm eff})$ — "
                                           "orthogonal init (Thm 6.8)",
                    **NULL_LINE)
        if matched.size:
            ax.plot(x[:matched.size], matched,
                    label="null from the observed layer-0 value", **MATCHED)
        ax.plot(x, ip, label="observed ip_mean", **OBSERVED)

        depth_axis(ax, run.n_layers)
        ax.set_ylabel("mean pairwise inner product")
        ax.set_title(f"B1 · the residual, not the fit — {run.label}")
        ax.legend(loc="upper left", fontsize=8.5)

        gap = run.scalar("B", "anisotropy_gap")
        bits = [run.text("B", "sign_convention")]
        if np.isfinite(gap):
            bits.append(f"Mean distance between the two nulls is {gap:.4f} — "
                        f"that part of any disagreement is anisotropy "
                        f"(non-orthogonal embeddings), not resistance.")
        caption(fig, "  ".join(b for b in bits if b))
    return save_figure(fig, out_dir, "residual_curve")


# ---------------------------------------------------------------------------
# B2 — the gap alone
# ---------------------------------------------------------------------------

def _b2_residual_depth(run: Run, out_dir: Path) -> Path:
    """
    Residual and matched residual vs depth, on a zero-centred axis.

    Zero-centred and symmetric on purpose: the quantity's meaning is its
    sign relative to zero ("on the null's own schedule"), and an axis
    auto-scaled to an all-negative series puts the visual centre of the
    figure somewhere inside resistance.

    The caption reports where the null saturates, because above ~0.95 this
    figure has almost no dynamic range and B3 is the one to read instead.
    """
    resid = run.series("B.residual")
    matched = run.series("B.residual_matched")
    null = run.series("B.gamma_null")
    x = np.arange(resid.size)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        ax.axhline(0.0, color="#374151", lw=1.2, zorder=2)
        ax.fill_between(x, 0, resid, where=resid <= 0, color="#2A78D6",
                        alpha=0.22, linewidth=0)
        ax.fill_between(x, 0, resid, where=resid > 0, color="#D6483F",
                        alpha=0.20, linewidth=0)
        ax.plot(x, resid, label="residual = observed − null", **OBSERVED)
        if matched.size:
            ax.plot(np.arange(matched.size), matched,
                    label="residual vs the observed-matched null", **MATCHED)

        # Where the null is saturated this figure cannot say much, and the
        # honest thing is to shade that region rather than let a flat line
        # near zero read as "no effect".
        if null.size:
            sat = null >= 0.95
            for i in np.nonzero(sat)[0]:
                ax.axvspan(i - 0.5, i + 0.5, color="#FEF3C7", zorder=0,
                           linewidth=0)
            if sat.any():
                ax.axvspan(np.nan, np.nan, color="#FEF3C7",
                           label="null > 0.95 — vertical residual compressed")

        m = float(np.nanmax(np.abs(resid))) if np.isfinite(resid).any() else 1.0
        ax.set_ylim(-m * 1.25, m * 1.25)
        depth_axis(ax, run.n_layers)
        ax.set_ylabel("residual")
        ax.set_title(f"B2 · what the identity-weight dynamics do not explain "
                     f"— {run.label}")
        ax.legend(loc="best", fontsize=8.5)
        final = run.scalar("B", "final_residual")
        caption(fig, f"Final residual {final:+.4f}. Negative is resistance. "
                     f"Where the null is above 0.95 read B3 instead — both "
                     f"curves are against the ceiling and the vertical "
                     f"measure has no range there (status-1c finding 3).")
    return save_figure(fig, out_dir, "residual_depth")


# ---------------------------------------------------------------------------
# B3 — the measure that keeps its range
# ---------------------------------------------------------------------------

def _b3_time_residual(run: Run, out_dir: Path) -> Path:
    """
    How long the null would need, against how long the network spent.

    Two panels sharing a depth axis: the two times above, their difference
    below. The unreachable layers — observed below the null's starting
    point, so no null time exists at all — are drawn as marked bands in both
    panels and counted in the caption. `gamma_null.py` calls those layers
    the strongest possible resistance signal, and Phase 1 already found
    mid-network mass a factor of 20 below the embedding floor, so they are
    expected and are the interesting ones.
    """
    t_req = run.series("B.time_domain.t_required")
    t_eff = run.series("B.time_domain.t_eff_grid")
    if not t_eff.size:
        t_eff = run.series("B.t_eff_grid")
    resid = run.series("B.time_domain.time_residual")
    x = np.arange(max(t_req.size, resid.size))

    with plt.rc_context(BLOG_STYLE):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9.2, 6.4), sharex=True,
            gridspec_kw=dict(height_ratios=[1.5, 1], hspace=0.12))

        n_bad = mark_nan_spans(ax, t_req)
        mark_nan_spans(ax2, t_req, label="")
        ax.plot(np.arange(t_req.size), t_req, color="#12406F", lw=2.6,
                label="$t_{\\rm null}^{-1}({\\rm ip\\_mean})$ — time the null "
                      "would need")
        ax.plot(np.arange(t_eff.size), t_eff, color=CATEGORICAL[1], lw=2.2,
                ls="--", label="$T_{\\rm eff}$ — time the network spent")
        ax.set_ylabel("ODE time")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.set_title(f"B3 · the residual in the time domain — {run.label}")

        ax2.axhline(0.0, color="#374151", lw=1.2, zorder=2)
        ax2.fill_between(np.arange(resid.size), 0, resid, where=resid <= 0,
                         color="#2A78D6", alpha=0.22, linewidth=0)
        ax2.fill_between(np.arange(resid.size), 0, resid, where=resid > 0,
                         color="#D6483F", alpha=0.20, linewidth=0)
        ax2.plot(np.arange(resid.size), resid, color="#12406F", lw=2.2)
        ax2.set_ylabel("time residual")
        depth_axis(ax2, run.n_layers)

        final = run.scalar("B.time_domain", "final_time_residual")
        caption(fig, (
            f"Final time residual {final:+.3f}; negative means the network "
            f"spent longer than the null needs for this much clustering. "
            f"{n_bad} layer(s) shaded purple are UNREACHABLE — the observed "
            f"inner product is below the null's own starting point, so no "
            f"null time exists. Counted, never clipped to zero."))
    return save_figure(fig, out_dir, "time_residual")


# ---------------------------------------------------------------------------
# B4 — the trajectory in the ODE's own clock
# ---------------------------------------------------------------------------

def _b4_trajectory_in_time(run: Run, out_dir: Path) -> Path:
    """
    ip_mean against T_eff rather than against layer index.

    Depth is not time. This is the same trajectory as B1 re-plotted on the
    axis the theory is written in, with layer indices annotated along the
    path — so the places where the network spends a lot of integration time
    on very little clustering (or the reverse) show up as stretches and
    bunching rather than as slope changes on a uniform grid.
    """
    ip = run.series("B.ip_mean")
    t = run.series("B.t_eff_grid")
    null = run.series("B.gamma_null")
    k = min(ip.size, t.size)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        if k < 2:
            no_data(ax, "no T_eff grid for this run")
        else:
            if null.size:
                ax.plot(t[:k], null[:k], label="$\\gamma_\\beta(t)$ at this "
                                               "run's β", **NULL_LINE)
                ax.fill_between(t[:k], ip[:k], null[:k], color="#2A78D6",
                                alpha=0.14, linewidth=0)
            ax.plot(t[:k], ip[:k], marker="o", ms=4, **OBSERVED,
                    label="observed trajectory")
            stride = max(1, k // 8)
            for i in range(0, k, stride):
                ax.annotate(f"ℓ{i}", xy=(t[i], ip[i]), xytext=(5, -11),
                            textcoords="offset points", fontsize=7.5,
                            color="#4B5563")
            t_star = run.scalar("B.collapse_fraction", "t_star")
            if np.isfinite(t_star) and t_star < t[k - 1] * 4:
                reference_line(ax, t_star, f"t* = {t_star:.2f}", axis="x")
            ax.set_xlabel("$T_{\\rm eff}$ — integration time elapsed")
            ax.set_ylabel("mean pairwise inner product")
            ax.legend(loc="upper left", fontsize=8.5)
        ax.set_title(f"B4 · the trajectory in the ODE's own clock — {run.label}")
        caption(fig, "Same trajectory as B1, on the axis the theory is "
                     "written in. Bunched labels are layers that spend little "
                     "integration time; stretched ones spend a lot.")
    return save_figure(fig, out_dir, "trajectory_in_time")


# ---------------------------------------------------------------------------
# B5 / B6 — the β envelope
# ---------------------------------------------------------------------------

def _b5_beta_envelope(run: Run, out_dir: Path) -> Path:
    """
    The band the null occupies over the layer's whole per-head β range.

    Hatched, because it is a bracket over an undecided reduction and not a
    confidence interval. Layers where the observed curve leaves the band are
    marked: there the conclusion holds for EVERY reduction and the decision
    is moot. Where it is inside, the reduction decides the sign — and that
    is the case status-1c says must be reported as uncertainty rather than
    defaulted away.
    """
    ip = run.series("B.ip_mean")
    lo = run.series("B.envelope_lower")
    hi = run.series("B.envelope_upper")
    k = min(ip.size, lo.size, hi.size)
    x = np.arange(k)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.2, 5.0))
        ax.fill_between(x, lo[:k], hi[:k], label="null over the per-head β "
                                                 "range", **ENVELOPE_BAND)
        ax.plot(x, ip[:k], label="observed ip_mean", **OBSERVED)

        outside = (ip[:k] < lo[:k]) | (ip[:k] > hi[:k])
        if outside.any():
            ax.plot(x[outside], ip[:k][outside], "o", ms=7, mfc="white",
                    mec="#12406F", mew=2, zorder=6,
                    label=f"outside the band ({int(outside.sum())} layers) — "
                          f"reduction-independent")
        depth_axis(ax, run.n_layers)
        ax.set_ylabel("mean pairwise inner product")
        ax.set_title(f"B5 · the β envelope — {run.label}")
        ax.legend(loc="upper left", fontsize=8.5)
        verdict_box(ax, run.text("B.envelope_verdict", "verdict"),
                    loc="lower right")

        rr = run.block("B.beta_reduction")
        bits = []
        if rr:
            bits.append(f"β over {rr.get('n_finite', '?')} heads: "
                        f"[{_f(rr.get('beta_min'))}, {_f(rr.get('beta_max'))}], "
                        f"spread across reductions {_f(rr.get('reduction_spread'))}.")
        bits.append("γ_β is monotone in β, so the per-head range brackets the "
                    "null without any reduction being chosen.")
        caption(fig, " ".join(bits))
    return save_figure(fig, out_dir, "beta_envelope")


def _b6_residual_bracket(run: Run, out_dir: Path) -> Path:
    """
    [residual_min, residual_max] vs depth — the residual as the interval it
    actually is.

    A single residual computed at an arbitrary reduction is a point estimate
    with an unstated error bar the size of the band (`beta_reduction.py`).
    Where the band clears zero the sign holds for every reduction; where it
    straddles zero, the phase has no verdict and this figure says so.
    """
    lo = run.series("B.residual_bracket.residual_min")
    hi = run.series("B.residual_bracket.residual_max")
    point = run.series("B.residual")
    k = min(lo.size, hi.size)
    x = np.arange(k)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        ax.axhline(0.0, color="#374151", lw=1.2, zorder=3)
        ax.fill_between(x, lo[:k], hi[:k], label="residual over the per-head "
                                                 "β range", **ENVELOPE_BAND)
        if point.size:
            ax.plot(np.arange(min(point.size, k)), point[:k],
                    label="residual at the run's chosen β", **OBSERVED)
        straddles = (lo[:k] <= 0) & (hi[:k] >= 0)
        for i in np.nonzero(straddles)[0]:
            ax.axvspan(i - 0.5, i + 0.5, color="#FEF3C7", zorder=0,
                       linewidth=0)
        if straddles.any():
            ax.axvspan(np.nan, np.nan, color="#FEF3C7",
                       label=f"sign depends on the reduction "
                             f"({int(straddles.sum())} layers)")

        depth_axis(ax, run.n_layers)
        ax.set_ylabel("residual")
        ax.set_title(f"B6 · the residual as the interval it is — {run.label}")
        ax.legend(loc="best", fontsize=8.5)
        br = run.block("B.residual_bracket")
        sign = ("unambiguous" if br.get("sign_unambiguous")
                else "AMBIGUOUS — the reduction decides it")
        caption(fig, (
            f"Final bracket [{_f(br.get('final_residual_min'))}, "
            f"{_f(br.get('final_residual_max'))}]; sign {sign}."))
    return save_figure(fig, out_dir, "residual_bracket")


# ---------------------------------------------------------------------------
# B7 — two fractions that are not the same fraction
# ---------------------------------------------------------------------------

def _b7_collapse_fraction(run: Run, out_dir: Path) -> Path:
    """
    time_fraction and gamma_fraction, drawn on the curve they are read off.

    Because γ is a sigmoid, a small time fraction can correspond to
    substantial clustering or to essentially none depending on where on the
    curve it lands, and reporting the time fraction alone invites reading a
    linear relationship into a saturating one (`collapse_fraction`'s own
    docstring). So the two numbers are drawn as positions on the null curve
    rather than as two bars.
    """
    cf = run.block("B.collapse_fraction")
    t_grid = run.series("B.t_eff_grid")
    null = run.series("B.gamma_null")
    t_total = float(cf.get("t_eff_total") or np.nan)
    t_star = float(cf.get("t_star") or np.nan)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        if t_grid.size and null.size:
            ax.plot(t_grid, null, label="$\\gamma_\\beta(t)$ — the null's own "
                                        "schedule", **NULL_LINE)
            ax.fill_between(t_grid, 0, null, **NULL_BAND)
        g_reached = float(cf.get("gamma_reached_by_null") or np.nan)
        if np.isfinite(t_total) and np.isfinite(g_reached):
            ax.plot([t_total], [g_reached], marker="o", ms=10,
                    color=CATEGORICAL[1], zorder=5,
                    label="where the network's clock stops")
            ax.vlines(t_total, 0, g_reached, color=CATEGORICAL[1], lw=1.2,
                      ls=":")
            ax.hlines(g_reached, 0, t_total, color=CATEGORICAL[1], lw=1.2,
                      ls=":")
        ip_final = float(cf.get("ip_mean_final") or np.nan)
        if np.isfinite(ip_final) and np.isfinite(t_total):
            ax.plot([t_total], [ip_final], marker="*", ms=16,
                    color="#12406F", zorder=6, label="where the network "
                                                     "actually is")
        if np.isfinite(t_star):
            reference_line(ax, t_star, f"t* = {t_star:.2f}", axis="x")
            ax.set_xlim(0, max(t_star * 1.1, t_total * 1.1
                               if np.isfinite(t_total) else t_star))
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("ODE time")
        ax.set_ylabel("mean pairwise inner product")
        ax.set_title(f"B7 · two fractions, one curve — {run.label}")
        ax.legend(loc="lower right", fontsize=8.5)
        caption(fig, (
            f"time fraction {_f(cf.get('time_fraction'))} of t*, but the null "
            f"only reaches γ = {_f(cf.get('gamma_reached_by_null'))} in that "
            f"time (γ fraction {_f(cf.get('gamma_fraction'))}). The gap "
            f"between those two numbers is the sigmoid, and it is why P-γ2 "
            f"and P-γ1 are separate predictions."))
    return save_figure(fig, out_dir, "collapse_fraction")


# ---------------------------------------------------------------------------
# B8 — which β each layer's null was evaluated at
# ---------------------------------------------------------------------------

def _b8_beta_fallback_audit(run: Run, out_dir: Path) -> Path:
    """
    Per-layer β, with the fallback layers marked.

    β_eff is not constant across layers, and the null is evaluated per layer
    at that layer's own β precisely so the QK circuits' variation is not
    folded into the residual and attributed to the weights' resistance
    (design-1c). Layers whose regression failed fall back to the run median
    — the residual there is a different measurement wearing the same label,
    and `n_beta_fallback` counts them.
    """
    betas = run.series("B.beta_per_layer")
    med = run.scalar("B", "beta_median")
    n_fb = run.scalar("B", "n_beta_fallback", 0)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.0, 4.0))
        if not betas.size:
            no_data(ax, "no per-layer β recorded for this run")
        else:
            x = np.arange(betas.size)
            fell_back = np.isclose(betas, med, rtol=1e-12, atol=1e-12)
            ax.bar(x[~fell_back], betas[~fell_back], color=CATEGORICAL[0],
                   width=0.75, label="β from the layer's own regression")
            if fell_back.any() and n_fb:
                ax.bar(x[fell_back], betas[fell_back], color=INVALID_COLOR,
                       width=0.75, hatch="//", edgecolor="#6B7280",
                       label=f"possible median fallback (run reports "
                             f"{int(n_fb)})")
            if np.isfinite(med):
                reference_line(ax, med, f"run median β = {med:.3f}")
            ax.set_ylabel("$\\beta_{\\rm eff}$")
            ax.legend(loc="best", fontsize=8.5)
            depth_axis(ax, run.n_layers)
        ax.set_title(f"B8 · which β each layer's null used — {run.label}")
        caption(fig, (
            f"β source: {run.beta_source}. A layer whose β_eff regression "
            f"failed is not a layer with β = 0 — it is one evaluated against "
            f"a different null, and the count is the honest error bar on the "
            f"series."))
    return save_figure(fig, out_dir, "beta_fallback_audit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v, fmt: str = "{:+.4f}") -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return fmt.format(x) if np.isfinite(x) else "n/a"
