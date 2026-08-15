"""
p1c_frames/visualization/cross_run.py — the sweep, and what it could answer.

Six figures (V1-V6 in FIGURES-1c.md). Two of them exist because Phase 1c is
a re-analysis phase whose coverage is uneven by construction: a run without
`norms` cannot answer A or C at all, one without `beta_eff` cannot answer A
or B, and one without `clusters.npz` cannot answer F. V5 draws that as a
matrix and V6 audits where each run's β came from, because a sweep reported
as "27 checkpoints" that silently ran B on nine of them is the failure mode
this phase is most exposed to.

**No pooled residual figure, deliberately.** t* is n-dependent and the
prompts span 20-512 tokens (status-1c open item 4), so every figure here
keeps runs separate and draws each against its own t*. V4 is the picture of
why: it puts n against both T_eff and t* on one axis, so the reader can see
how much of the sweep's spread in "distance to t*" is prompt length rather
than model.

Verdicts are read from the artifacts, never recomputed. V1 tints each tile
by the leading word of the phase's own verdict string; a word the palette
does not know renders grey and says what it says, which is the visible
failure this package prefers to a silent mislabel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .loaders import SUBEXPS, SUBEXP_TITLES, Run, records
from .style import (
    BLOG_STYLE, CATEGORICAL, INVALID_COLOR, RESIDUAL_CMAP, SEQ_CMAP,
    STEP_COLORS, STEP_DEFS, STEP_LABELS, VERDICT_COLORS, caption, model_color,
    no_data, reference_line, residual_norm, save_figure, verdict_word,
)

__all__ = ["generate_cross_run_figures"]


def _skip(name: str, reason: str) -> None:
    print(f"    skip {name}: {reason}")


def generate_cross_run_figures(runs: Sequence[Run],
                               out_dir: Path) -> List[Path]:
    """V1-V6 over every discovered run."""
    runs = list(runs)
    if not runs:
        return []
    paths: List[Optional[Path]] = [
        _v1_verdict_card(runs, out_dir),
        _v2_teff_across_runs(runs, out_dir),
        _v3_residual_heatmap(runs, out_dir),
        _v4_length_vs_time(runs, out_dir),
        _v5_availability_matrix(runs, out_dir),
        _v6_beta_audit(runs, out_dir),
    ]
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# V1 — every verdict the phase reached
# ---------------------------------------------------------------------------

def _v1_verdict_card(runs: Sequence[Run], out_dir: Path) -> Path:
    """
    Run × verdict tiles, each tinted by the phase's own leading word.

    Four columns, one per adjudicated question, and each cell carries the
    supporting number as well as the word — a tile that is only a colour is
    one somebody will read as a result without its magnitude. Runs that
    could not answer a question get the invalid grey and the word "not run",
    which is a different statement from any verdict and must not look like
    one.
    """
    cols = ("P-γ2 (A)", "sinks (C)", "envelope (B)", "P-H1 (E)")

    def cell(run: Run, col: str):
        if col == "P-γ2 (A)":
            if not run.has("A"):
                return "not run", "", INVALID_COLOR
            robust = run.block("A.verdict").get("robust")
            ratio = run.scalar("A.verdict", "ratio_calibrated")
            word = "ROBUST" if robust else ("STRADDLES" if robust is False
                                            else "UNCLEAR")
            val = f"T_eff/t* = {ratio:.2f}" if np.isfinite(ratio) else ""
            return word, val, VERDICT_COLORS.get(word, INVALID_COLOR)
        if col == "sinks (C)":
            if not run.has("C"):
                return "not run", "", INVALID_COLOR
            v = run.text("C.sink_verdict", "verdict")
            word = verdict_word(v) or "?"
            c = run.scalar("C.sink_verdict", "corr_raw_vs_norm_pr")
            return word, (f"corr {c:.2f}" if np.isfinite(c) else ""), \
                VERDICT_COLORS.get(word, INVALID_COLOR)
        if col == "envelope (B)":
            v = run.text("B.envelope_verdict", "verdict")
            if not v:
                return "no per-head β", "", INVALID_COLOR
            word = verdict_word(v) or "?"
            frac = run.scalar("B.envelope_verdict", "frac_outside")
            return word, (f"{frac:.0%} outside" if np.isfinite(frac) else ""), \
                VERDICT_COLORS.get(word, INVALID_COLOR)
        if not run.has("E"):
            return "not run", "", INVALID_COLOR
        all_feas = run.block("E").get("all_feasible")
        m = run.scalar("E", "min_margin")
        word = "CONFIRMED" if all_feas else "FALSIFIED"
        return word, (f"min margin {m:.4f}" if np.isfinite(m) else ""), \
            VERDICT_COLORS.get(word, INVALID_COLOR)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(11.2, 0.55 * len(runs) + 2.2))
        ax.set_xlim(0, len(cols))
        ax.set_ylim(0, len(runs))
        ax.invert_yaxis()
        ax.set_xticks(np.arange(len(cols)) + 0.5)
        ax.set_xticklabels(cols)
        ax.xaxis.set_ticks_position("top")
        ax.set_yticks(np.arange(len(runs)) + 0.5)
        ax.set_yticklabels([r.stem for r in runs], fontsize=8)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i, run in enumerate(runs):
            for j, col in enumerate(cols):
                word, val, color = cell(run, col)
                ax.add_patch(mpl.patches.FancyBboxPatch(
                    (j + 0.03, i + 0.08), 0.94, 0.84,
                    boxstyle="round,pad=0.0,rounding_size=0.04",
                    facecolor=color, alpha=0.16, edgecolor=color,
                    linewidth=1.2))
                ax.text(j + 0.5, i + 0.36, word, ha="center", va="center",
                        fontsize=8.5, color="#111827", fontweight="bold")
                if val:
                    ax.text(j + 0.5, i + 0.68, val, ha="center", va="center",
                            fontsize=7.5, color="#4B5563")
        ax.set_title("V1 · every verdict the phase reached, per run", pad=28)
        caption(fig, "Verdict words are the phase's own, read from each "
                     "artifact. \"not run\" is grey and is not a verdict — a "
                     "run that could not answer a question has said nothing "
                     "about it.")
    return save_figure(fig, out_dir, "verdict_card")


# ---------------------------------------------------------------------------
# V2 — P-γ2 for the whole sweep
# ---------------------------------------------------------------------------

def _v2_teff_across_runs(runs: Sequence[Run], out_dir: Path) -> Optional[Path]:
    """
    T_eff under all three definitions per run, each against its own t*.

    Plotted as T_eff / t* rather than as T_eff, because the runs have
    different t* and a raw comparison would mostly be reading prompt length.
    The 1.0 line is then the prediction's boundary for every run at once,
    which is the only honest way to put a sweep with eight prompt lengths on
    one axis.
    """
    rows = [r for r in runs if r.has("A")]
    if not rows:
        _skip("teff_across_runs", "no run carries an A block")
        return None

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.4, 0.42 * len(rows) + 2.4))
        ys = np.arange(len(rows))
        for i, run in enumerate(rows):
            t_star = run.scalar("A", "t_star")
            for key, art in zip(STEP_DEFS, ("T_eff_displacement",
                                            "T_eff_calibrated",
                                            "T_eff_attn_only")):
                v = run.scalar("A", art)
                if not (np.isfinite(v) and np.isfinite(t_star) and t_star > 0):
                    continue
                ax.plot([v / t_star], [i], marker="o", ms=8,
                        color=STEP_COLORS[key], zorder=4,
                        label=STEP_LABELS[key] if i == 0 else None)
        reference_line(ax, 1.0, "T_eff = t*", axis="x")
        ax.axvspan(0, 1, color="#F3F4F6", zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels([r.stem for r in rows], fontsize=8)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel("$T_{\\rm eff}$ / t*   (each run against its own t*)")
        ax.set_title("V2 · P-γ2 across the sweep")
        ax.legend(loc="lower right", fontsize=8)
        caption(fig, "Normalized by each run's own t* — the raw T_eff would "
                     "mostly be reading prompt length, since t* is "
                     "n-dependent. A run whose three markers straddle 1.0 has "
                     "a definitional answer, not a measured one.")
    return save_figure(fig, out_dir, "teff_across_runs")


# ---------------------------------------------------------------------------
# V3 — model × prompt
# ---------------------------------------------------------------------------

def _v3_residual_heatmap(runs: Sequence[Run], out_dir: Path) -> Optional[Path]:
    """
    Final residual and final time residual over model × prompt.

    Both, because they are the same statement at two resolutions and the
    vertical one loses its range where the null saturates. A cell present in
    one panel and blank in the other is a run whose observed value fell
    outside the null's reachable range — which is information, not a gap.
    """
    rows = [r for r in runs if r.has("B")]
    if not rows:
        _skip("residual_heatmap", "no run carries a B block")
        return None

    models = sorted({r.model for r in rows})
    prompts = sorted({r.prompt for r in rows})
    grids = {}
    for key, section, field in (("final residual", "B", "final_residual"),
                                ("final time residual", "B.time_domain",
                                 "final_time_residual")):
        g = np.full((len(models), len(prompts)), np.nan)
        for r in rows:
            g[models.index(r.model), prompts.index(r.prompt)] = \
                r.scalar(section, field)
        grids[key] = g

    with plt.rc_context(BLOG_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.42 * len(models) + 3.2))
        for ax, (title, g) in zip(np.atleast_1d(axes), grids.items()):
            im = ax.imshow(g, aspect="auto", cmap=RESIDUAL_CMAP,
                           norm=residual_norm(g), interpolation="nearest")
            ax.set_xticks(range(len(prompts)))
            ax.set_xticklabels(prompts, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models, fontsize=8)
            ax.grid(False)
            ax.set_title(title, fontsize=11)
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    if np.isfinite(g[i, j]):
                        ax.text(j, i, f"{g[i, j]:+.3f}", ha="center",
                                va="center", fontsize=7.5, color="#111827")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("V3 · residual by model and prompt", fontsize=12.5)
        caption(fig, "Blue is behind the null — resistance. The two panels "
                     "are one statement at two resolutions; where the null "
                     "saturates only the right-hand one has range.", y=0.0)
    return save_figure(fig, out_dir, "residual_heatmap")


# ---------------------------------------------------------------------------
# V4 — why t* cannot be pooled
# ---------------------------------------------------------------------------

def _v4_length_vs_time(runs: Sequence[Run], out_dir: Path) -> Optional[Path]:
    """
    n_tokens against T_eff and t*, on one axis.

    status-1c open item 4 says per-prompt t*, never a pooled one. This is
    the evidence: t* moves with n across the sweep's prompt range, so the
    distance between a run's T_eff and its t* is partly a property of the
    prompt. If the two series are parallel, prompt length is a nuisance
    variable that cancels; if they diverge, it is a confound and every
    cross-prompt comparison in the phase has to be read against this figure.
    """
    rows = [r for r in runs if r.has("A") and r.n_tokens]
    if not rows:
        _skip("length_vs_time", "no run carries both A and a token count")
        return None

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        for r in rows:
            n = r.n_tokens
            ax.plot([n], [r.scalar("A", "T_eff_calibrated")], marker="o",
                    ms=8, color=model_color(r.model), zorder=4)
            ax.plot([n], [r.scalar("A", "t_star")], marker="^", ms=8,
                    mfc="white", mec=model_color(r.model), mew=1.6, zorder=4)
            ax.plot([n, n], [r.scalar("A", "T_eff_calibrated"),
                             r.scalar("A", "t_star")],
                    color=model_color(r.model), lw=1.0, alpha=0.5, zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("n (tokens in the prompt)")
        ax.set_ylabel("ODE time")
        handles = [plt.Line2D([], [], marker="o", ls="", color="#374151",
                              label="$T_{\\rm eff}$ (calibrated)"),
                   plt.Line2D([], [], marker="^", ls="", mfc="white",
                              mec="#374151", label="t* for this prompt")]
        ax.legend(handles=handles, loc="best", fontsize=8.5)
        ax.set_title("V4 · prompt length moves the target, not just the shot")
        caption(fig, "The vertical bar is each run's distance to its own t*. "
                     "If t* slopes with n, a pooled collapse time would "
                     "compare short-prompt runs against a number that is not "
                     "theirs.")
    return save_figure(fig, out_dir, "length_vs_time")


# ---------------------------------------------------------------------------
# V5 — what the sweep could actually answer
# ---------------------------------------------------------------------------

def _v5_availability_matrix(runs: Sequence[Run], out_dir: Path) -> Path:
    """
    Run × sub-experiment, from what landed rather than from what was asked.

    `tools/preflight_1c.py` predicts this before a sweep; this draws it
    after, which is the check that the prediction held. Three states, and
    they are genuinely different: ran, skipped-with-a-reason (the driver
    refused — no `norms`, no `beta_eff`, no `clusters.npz`), and never
    selected (`--subexp` did not include it). Collapsing the last two would
    turn a deliberate scope choice into a data problem.
    """
    state = np.zeros((len(runs), len(SUBEXPS)))
    reasons: Dict[tuple, str] = {}
    for i, r in enumerate(runs):
        for j, sub in enumerate(SUBEXPS):
            if r.has(sub):
                state[i, j] = 2
            elif sub in r.skipped:
                state[i, j] = 1
                reasons[(i, j)] = str(r.skipped[sub])
            else:
                state[i, j] = 0

    cmap = mpl.colors.ListedColormap([INVALID_COLOR, "#F5B98D", "#1BAF7A"])
    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(runs) + 2.8))
        ax.imshow(state, aspect="auto", cmap=cmap, vmin=0, vmax=2,
                  interpolation="nearest")
        ax.set_xticks(range(len(SUBEXPS)))
        ax.set_xticklabels([f"{s}\n{SUBEXP_TITLES[s].split(' ')[0]}"
                            for s in SUBEXPS], fontsize=8.5)
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels([r.stem for r in runs], fontsize=8)
        ax.grid(False)
        for (i, j) in reasons:
            ax.text(j, i, "✕", ha="center", va="center", fontsize=9,
                    color="#7A1F1D")
        handles = [mpl.patches.Patch(facecolor=c, label=l) for c, l in (
            ("#1BAF7A", "ran"),
            ("#F5B98D", "skipped — the driver refused (✕, reason recorded)"),
            (INVALID_COLOR, "not selected (--subexp)"))]
        ax.legend(handles=handles, loc="upper center", ncol=3,
                  bbox_to_anchor=(0.5, -0.06), fontsize=8, frameon=False)
        ax.set_title("V5 · what each run could actually answer")
        first = next(iter(reasons.values()), "")
        caption(fig, (f"Example skip reason: “{first[:150]}”" if first else
                      "No run was skipped by the driver."), y=-0.02)
    return save_figure(fig, out_dir, "availability_matrix")


# ---------------------------------------------------------------------------
# V6 — where β came from
# ---------------------------------------------------------------------------

def _v6_beta_audit(runs: Sequence[Run], out_dir: Path) -> Path:
    """
    β per run, coloured by source, with the envelope width where it exists.

    β is a measured property of a trained head (paper footnote 2), not a
    convention, which is why `run_1c` refuses to invent one and skips runs
    without it. A run using `--beta-fallback` is answering a slightly
    different question from one reading `geometry.json`, and the two must
    not sit in a table looking identical. The second panel is the envelope
    width — the error bar the point estimate does not carry — drawn only
    where per-head betas existed.
    """
    with plt.rc_context(BLOG_STYLE):
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(12, 0.42 * len(runs) + 2.4),
            gridspec_kw=dict(width_ratios=[2, 1]))
        ys = np.arange(len(runs))
        for i, r in enumerate(runs):
            from_geo = r.beta_source == "geometry.json"
            ax.barh(i, r.beta if np.isfinite(r.beta) else 0,
                    color=CATEGORICAL[0] if from_geo else "#F5B98D",
                    height=0.7,
                    edgecolor="#7A1F1D" if not from_geo else "none",
                    hatch="" if from_geo else "//")
            rr = r.block("B.beta_reduction")
            if rr.get("beta_min") is not None:
                ax.plot([rr["beta_min"], rr["beta_max"]], [i, i],
                        color="#374151", lw=1.4, zorder=5)
        ax.set_yticks(ys)
        ax.set_yticklabels([r.stem for r in runs], fontsize=8)
        ax.set_xlabel("$\\beta_{\\rm eff}$ used")
        ax.invert_yaxis()
        handles = [mpl.patches.Patch(facecolor=CATEGORICAL[0],
                                     label="from geometry.json"),
                   mpl.patches.Patch(facecolor="#F5B98D", hatch="//",
                                     edgecolor="#7A1F1D",
                                     label="--beta-fallback"),
                   plt.Line2D([], [], color="#374151", lw=1.4,
                              label="per-head range")]
        ax.legend(handles=handles, loc="lower right", fontsize=8)
        ax.set_title("β used, and where it came from", fontsize=11)

        widths = [r.scalar("B.envelope_verdict", "mean_band_width")
                  for r in runs]
        have = [i for i, w in enumerate(widths) if np.isfinite(w)]
        if have:
            ax2.barh([i for i in have], [widths[i] for i in have],
                     color=CATEGORICAL[3], height=0.7)
            ax2.set_yticks(ys)
            ax2.set_yticklabels([])
            ax2.invert_yaxis()
            ax2.set_xlabel("mean envelope width")
            ax2.set_title("the error bar the point\nestimate does not carry",
                          fontsize=10)
        else:
            no_data(ax2, "no run carries per-head β,\nso no envelope exists "
                         "anywhere in this sweep")
        fig.suptitle("V6 · the β audit", fontsize=12.5)
        caption(fig, "β is a measured property of a trained head, not a "
                     "convention — a run on --beta-fallback is answering a "
                     "slightly different question and is marked as such.",
                y=0.0)
    return save_figure(fig, out_dir, "beta_audit")
