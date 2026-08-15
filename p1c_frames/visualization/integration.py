"""
p1c_frames/visualization/integration.py — sub-experiment A, the clock.

Six per-run figures (A1-A6 in FIGURES-1c.md). The organizing constraint is
status-1c finding 1: `MATH.md` §8's step-size definition understates
$T_{\\rm eff}$ by ~5.7x on the validation trajectory, because it omits the
$\\|\\mathcal{X}\\|$ denominator and the field runs at ~0.18 rather than at
its bound of 1 — and the bias points toward "the network never integrates
far enough", which is the direction that would make Blog 1's headline an
artifact of depth.

So no figure here plots one definition alone:

  * A1 and A2 draw all three, with linestyle as well as hue, because the
    5.7x gap is legible at thumbnail size and the hue difference is not.
  * A3 draws the mechanism — the field magnitude against its own bound —
    rather than the correction factor, so a reader can see WHY the two
    definitions differ instead of being told by how much.
  * A4 is the honesty figure: three dots and a line at t*. If the dots land
    on both sides, `verdict.robust` is False and the phase's own answer is
    "report the spread, not a verdict". That string is quoted, not
    paraphrased.

`t*` is per-prompt, never pooled: it is n-dependent and the prompts span
20-512 tokens (status-1c open item 4), so every figure that draws it takes
it from the run's own `A.t_star`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .loaders import Run
from .style import (
    BLOG_STYLE, CATEGORICAL, DEPTH_CMAP, INVALID_COLOR, STEP_COLORS,
    STEP_DEFS, STEP_LABELS, STEP_STYLES, VERDICT_COLORS, caption, depth_axis,
    no_data, reference_line, save_figure, verdict_box,
)

__all__ = ["generate_integration_figures"]


def _steps(run: Run) -> dict:
    """The three step series, by their artifact names. Absent -> empty."""
    return {k: run.series(f"A.{k}") for k in STEP_DEFS}


def _skip(name: str, reason: str) -> None:
    print(f"    skip {name}: {reason}")


def generate_integration_figures(run: Run, out_dir: Path) -> List[Path]:
    """A1-A6 for one run."""
    paths: List[Path] = []
    if not run.has("A"):
        _skip("integration", next((m for m in run.missing
                                   if m.startswith("A ")), "no A block"))
        return paths

    h = _steps(run)
    n_layers = run.n_layers
    t_star = run.scalar("A", "t_star")

    paths.append(_a1_step_size_depth(run, h, out_dir))
    paths.append(_a2_teff_accumulation(run, h, t_star, out_dir))
    if run.has_series("A.field_mag"):
        paths.append(_a3_field_magnitude(run, h, out_dir))
    else:
        _skip("field_magnitude", "no A.field_mag in this run")
    paths.append(_a4_definition_straddle(run, t_star, out_dir))
    paths.append(_a5_calibration_scatter(run, h, out_dir))
    if run.has_series("A.h_attn_only"):
        paths.append(_a6_attn_vs_full_step(run, h, out_dir))
    else:
        _skip("attn_vs_full_step",
              "h_attn_only is nan — no sublayer streams (G3). This is the "
              "frame-correct definition, so its absence is a limit on what "
              "this run can say, not a missing figure.")
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# A1 — the three definitions, per layer
# ---------------------------------------------------------------------------

def _a1_step_size_depth(run: Run, h: dict, out_dir: Path) -> Path:
    """
    h_l under all three definitions vs depth, log-y.

    Log-y because the definitions differ by a factor, not an offset: on a
    linear axis `h_displacement` is a flat line near zero and the shape it
    shares with `h_calibrated` — which is the evidence that they measure the
    same motion in different units — is invisible.
    """
    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        drawn = 0
        for key in STEP_DEFS:
            v = h[key]
            if not v.size or not np.isfinite(v).any():
                continue
            ax.plot(np.arange(v.size), v, color=STEP_COLORS[key],
                    label=STEP_LABELS[key], **STEP_STYLES[key])
            drawn += 1

        ax.set_yscale("log")
        depth_axis(ax, run.n_layers, xlabel="layer boundary ℓ → ℓ+1")
        ax.set_ylabel("step size $h_\\ell$  (ODE time units)")
        ax.set_title(f"A1 · step size per block — {run.label}")
        ax.legend(loc="best")

        ratio = _mean_ratio(h)
        caption(fig, (
            "All three definitions, always. §8 as written omits the ‖X‖ "
            f"denominator; here it runs {ratio:.1f}× below the calibrated "
            "step, against the 5.7× measured on the validation trajectory "
            "(status-1c finding 1)." if np.isfinite(ratio) else
            "All three definitions, always (status-1c finding 1)."))
        if not drawn:
            no_data(ax, "no finite step sizes in this run")
    return save_figure(fig, out_dir, "step_size_depth")


# ---------------------------------------------------------------------------
# A2 — the P-γ2 figure
# ---------------------------------------------------------------------------

def _a2_teff_accumulation(run: Run, h: dict, t_star: float,
                          out_dir: Path) -> Path:
    """
    Cumulative T_eff(l) for all three definitions against t*.

    The comparison P-γ2 is registered on. Each curve's crossing of t* is
    marked where it happens, and the absence of a crossing is itself the
    result: a 24-layer stack that never reaches t* has not run the paper's
    dynamics long enough to collapse, and "resistance" would be partly
    depth.
    """
    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5.0))
        for key in STEP_DEFS:
            v = h[key]
            if not v.size or not np.isfinite(v).any():
                continue
            cum = np.concatenate([[0.0], np.nancumsum(v)])
            ax.plot(np.arange(cum.size), cum, color=STEP_COLORS[key],
                    label=STEP_LABELS[key], **STEP_STYLES[key])
            cross = _first_crossing(cum, t_star)
            if cross is not None:
                ax.plot([cross], [t_star], marker="o", ms=7, mfc="white",
                        mec=STEP_COLORS[key], mew=2, zorder=5)
                ax.annotate(f"crosses t* at layer {cross}",
                            xy=(cross, t_star), xytext=(4, -12),
                            textcoords="offset points", fontsize=8,
                            color=STEP_COLORS[key])

        if np.isfinite(t_star):
            # Right edge: the legend lives at upper left and a t* label there
            # is silently overdrawn by it.
            reference_line(ax, t_star,
                           f"t* = {t_star:.2f}  (γ = 0.9 at n = {run.n_tokens})",
                           side="right")
        depth_axis(ax, run.n_layers)
        ax.set_ylabel("cumulative $T_{\\rm eff}(\\ell)$")
        ax.set_title(f"A2 · does the network integrate far enough? — {run.label}")
        ax.legend(loc="upper left")
        # `reading` begins "T_eff …" and carries no keyword of its own; the
        # classification is the separate `robust` flag, so it is passed in.
        robust = run.block("A.verdict").get("robust")
        verdict_box(ax, run.text("A.verdict", "reading"), loc="lower right",
                    word=("ROBUST" if robust else
                          "STRADDLES" if robust is False else "UNCLEAR"))
        caption(fig, "P-γ2. t* is this prompt's own — it is n-dependent, and "
                     "the sweep's prompts span 20–512 tokens.")
    return save_figure(fig, out_dir, "teff_accumulation")


# ---------------------------------------------------------------------------
# A3 — why the definitions differ
# ---------------------------------------------------------------------------

def _a3_field_magnitude(run: Run, h: dict, out_dir: Path) -> Path:
    """
    Mean ‖X(x_l)‖ per layer against the paper's bound of 1.

    The bound is reached only by a fully collapsed cloud; for a spread cloud
    the field is far weaker, and 1/‖X‖ is exactly the factor by which §8's
    definition understates the step. Drawing the mechanism rather than the
    factor is deliberate — the factor is a number to be trusted, the
    magnitude is a measurement to be read.
    """
    fmag = h and run.series("A.field_mag")
    with plt.rc_context(BLOG_STYLE):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9, 6.0), sharex=True,
            gridspec_kw=dict(height_ratios=[2, 1.2], hspace=0.12))

        ax.plot(np.arange(fmag.size), fmag, color=CATEGORICAL[0], lw=2.4,
                label="mean ‖X(x)‖ per layer")
        ax.fill_between(np.arange(fmag.size), 0, fmag, color=CATEGORICAL[0],
                        alpha=0.12)
        reference_line(ax, 1.0, "‖X‖ ≤ 1 (paper §2) — equality only at full "
                                "collapse", side="left")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("field magnitude")
        ax.legend(loc="upper left")
        ax.set_title(f"A3 · the field runs far below its bound — {run.label}")

        with np.errstate(divide="ignore", invalid="ignore"):
            factor = 1.0 / np.where(fmag > 0, fmag, np.nan)
        ax2.plot(np.arange(factor.size), factor, color=CATEGORICAL[1], lw=2.0)
        reference_line(ax2, 5.67, "5.67× measured on the validation "
                                  "trajectory", side="left")
        ax2.set_ylabel("understatement\nfactor 1/‖X‖")
        depth_axis(ax2, run.n_layers, xlabel="layer boundary ℓ → ℓ+1")

        caption(fig, "The whole content of status-1c finding 1: §8's step "
                     "size is the numerator of the Euler step, and this is "
                     "the denominator it is missing.")
    return save_figure(fig, out_dir, "field_magnitude")


# ---------------------------------------------------------------------------
# A4 — is the answer a measurement or a definition?
# ---------------------------------------------------------------------------

def _a4_definition_straddle(run: Run, t_star: float, out_dir: Path) -> Path:
    """
    The three T_eff totals against t*, with `verdict.robust` quoted.

    `integration_time.verdict` reports `robust=False` when the definitions
    straddle t*, "in which case the answer is a definition, not a
    measurement, and must be reported that way" (design-1c). This figure is
    that sentence, drawn — and the shaded half is the side of t* that P-γ2
    is registered on.
    """
    keys = ("T_eff_displacement", "T_eff_calibrated", "T_eff_attn_only")
    vals = [run.scalar("A.verdict", k, run.scalar("A", k)) for k in keys]
    robust = run.block("A.verdict").get("robust")

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        ys = np.arange(len(keys))[::-1]
        for y, key, v in zip(ys, STEP_DEFS, vals):
            color = STEP_COLORS[key]
            if not np.isfinite(v):
                ax.text(0.02, y, f"{STEP_LABELS[key]} — not computed",
                        transform=ax.get_yaxis_transform(), va="center",
                        fontsize=9, color=INVALID_COLOR)
                continue
            ax.plot([0, v], [y, y], color=color, lw=3, alpha=0.35,
                    solid_capstyle="round")
            ax.plot([v], [y], marker="o", ms=11, color=color, zorder=5)
            ax.annotate(f"{v:.2f}", xy=(v, y), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=color, fontweight="bold")

        if np.isfinite(t_star):
            ax.axvspan(0, t_star, color="#F3F4F6", zorder=0)
            reference_line(ax, t_star, f"t* = {t_star:.2f}", axis="x",
                           side="left")
        ax.set_yticks(ys)
        ax.set_yticklabels([STEP_LABELS[k].split(" (")[0] for k in STEP_DEFS])
        ax.set_xlabel("$T_{\\rm eff}$ (ODE time units)")
        ax.set_ylim(-0.6, len(keys) - 0.4)
        word = ("ROBUST" if robust else
                ("STRADDLES" if robust is False else "UNCLEAR"))
        ax.set_title(f"A4 · measurement or definition? — {run.label}",
                     color=VERDICT_COLORS.get(word, "#111827"))
        caption(fig, run.text("A.verdict", "reading") or
                "no verdict recorded for this run")
    return save_figure(fig, out_dir, "definition_straddle")


# ---------------------------------------------------------------------------
# A5 — the two definitions against each other
# ---------------------------------------------------------------------------

def _a5_calibration_scatter(run: Run, h: dict, out_dir: Path) -> Path:
    """
    h_calibrated against h_displacement, one point per layer boundary.

    A scatter rather than a ratio curve because the interesting failure is
    not "the ratio is large" but "the ratio varies with depth": a constant
    ratio means the two definitions are one measurement in two units, and a
    depth-varying one means the correction re-orders which layers spend the
    most time. The identity line and the run's own mean ratio are both
    drawn, so which of those two worlds this run is in is legible.
    """
    x, y = h["h_displacement"], h["h_calibrated"]
    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        if x.size and y.size and np.isfinite(x).any():
            k = min(x.size, y.size)
            colors = DEPTH_CMAP(np.linspace(0.08, 0.95, k))
            ax.scatter(x[:k], y[:k], c=colors, s=52, edgecolor="white",
                       linewidth=0.8, zorder=4)
            for i in (0, k // 2, k - 1):
                ax.annotate(f"ℓ{i}", xy=(x[i], y[i]), xytext=(6, 4),
                            textcoords="offset points", fontsize=8,
                            color="#4B5563")
            hi = float(np.nanmax([np.nanmax(x[:k]), np.nanmax(y[:k])])) * 1.1
            ax.plot([0, hi], [0, hi], **{**dict(color="#6B7280",
                                                linestyle=":", lw=1.2),
                                         "zorder": 1})
            ax.annotate("y = x (‖X‖ = 1: a fully collapsed cloud)",
                        xy=(hi * 0.55, hi * 0.55), fontsize=8, rotation=38,
                        color="#6B7280", ha="center")
            r = _mean_ratio(h)
            if np.isfinite(r):
                ax.plot([0, hi], [0, hi * r], color=CATEGORICAL[1], lw=1.6,
                        ls="--", label=f"this run's mean ratio ({r:.1f}×)")
                ax.legend(loc="lower right")
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
        else:
            no_data(ax, "no finite step sizes in this run")

        ax.set_xlabel("$h_{\\rm displacement}$  (§8 as written)")
        ax.set_ylabel("$h_{\\rm calibrated}$  (the Euler step)")
        ax.set_title(f"A5 · one motion, two units — {run.label}")
        caption(fig, "Point colour is depth (dark = early). A constant ratio "
                     "means the definitions differ only in units; a "
                     "depth-varying one means the correction re-orders which "
                     "layers spend the time.")
    return save_figure(fig, out_dir, "calibration_scatter")


# ---------------------------------------------------------------------------
# A6 — how much of the block the paper's model contains
# ---------------------------------------------------------------------------

def _a6_attn_vs_full_step(run: Run, h: dict, out_dir: Path) -> Path:
    """
    h_attn_only as a fraction of h_calibrated, per layer.

    The paper writes the feed-forward layer down in §2 and then excludes
    it; every theorem in Parts 1-2 is single-head, no-FFN (design-1c §2).
    So this fraction is the share of each block's motion that the compared
    model actually contains, and the rest is time the ODE is being credited
    with but does not produce. Pythia's parallel residual makes the split
    exact, which is the one place GPT-2's sequential architecture could not
    have supported this figure at all.
    """
    full, attn = h["h_calibrated"], h["h_attn_only"]
    k = min(full.size, attn.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = attn[:k] / np.where(full[:k] > 0, full[:k], np.nan)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.4))
        ax.bar(np.arange(k), frac, color=STEP_COLORS["h_attn_only"],
               alpha=0.85, width=0.75)
        reference_line(ax, 1.0, "all of the block's motion is attention",
                       side="left")
        ax.axhspan(0, 0, color="none")
        ax.set_ylim(0, max(1.15, float(np.nanmax(frac)) * 1.1
                           if np.isfinite(frac).any() else 1.15))
        depth_axis(ax, run.n_layers, xlabel="layer boundary ℓ → ℓ+1")
        ax.set_ylabel("$h_{\\rm attn}/h_{\\rm calibrated}$")
        ax.set_title(f"A6 · how much of the block is in the paper's model — "
                     f"{run.label}")
        share = float(np.nanmean(frac)) if np.isfinite(frac).any() else np.nan
        caption(fig, f"Mean {share:.0%} of the calibrated step is the "
                     f"attention branch. The remainder is motion the ODE "
                     f"would be credited with and does not contain."
                if np.isfinite(share) else
                "h_attn_only is nan here — see FIGURES-1c.md G3.")
    return save_figure(fig, out_dir, "attn_vs_full_step")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_ratio(h: dict) -> float:
    """Mean per-layer h_calibrated / h_displacement, NaN when unavailable."""
    x, y = h.get("h_displacement"), h.get("h_calibrated")
    if x is None or y is None or not x.size or not y.size:
        return float("nan")
    k = min(x.size, y.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = y[:k] / np.where(x[:k] > 0, x[:k], np.nan)
    return float(np.nanmean(r)) if np.isfinite(r).any() else float("nan")


def _first_crossing(cum: np.ndarray, level: float):
    """First index at which a cumulative series reaches `level`, or None."""
    if not np.isfinite(level):
        return None
    idx = np.nonzero(np.asarray(cum) >= level)[0]
    return int(idx[0]) if idx.size else None
