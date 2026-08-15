"""
p1c_frames/visualization/checkpoints_1c.py — the training-step axis.

Six figures (K1-K6 in FIGURES-1c.md), and the two that adjudicate a
registered prediction across training:

  K1  P-γ1 — the residual is near zero at step 0 and grows with training.
      `adjudicate_p_gamma1` is imported and its verdict printed. Note the
      adjudicator reports the falsifier check separately from the
      monotonicity check on purpose: the prediction has two clauses and they
      can come apart, and "grew but not monotonically" is a PARTIAL, not a
      pass.
  K5  P-S1 — trained centroids closer to a spherical design than step-0
      ones, adjudicated with `adjudicate_p_s1_banded` against the family's
      own step-0 run. Banded, always: the registered falsifier is "no
      difference", which carries no threshold, and three degrees of pure
      sampling noise give a coin-flip's worth of "improvements" without one.

The step axis, colormap and family grouping come from
`p1_mstate_tracking/visualization/checkpoints.py` rather than being restated,
so Phase 1c's checkpoint figures cannot drift from Phase 1's, 1b's and 2's.
That package is the only cross-phase import in this one, and it is optional:
if it cannot be imported, this module reports the missing dependency against
itself and returns no figures rather than taking the rest of the package
down with it.

Families are grouped by (base model, prompt), never pooled across prompts.
t* is n-dependent and the sweep's prompts span 20-512 tokens, so a step axis
pooled over prompts would compare each checkpoint against a different
collapse time and call the difference training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .loaders import Run, checkpoint_families, record_matrix, records
from .style import (
    BLOG_STYLE, CATEGORICAL, INVALID_COLOR, RESIDUAL_CMAP, STEP_COLORS,
    STEP_DEFS, STEP_LABELS, caption, degree_color, no_data, reference_line,
    residual_norm, save_figure, verdict_box,
)

__all__ = ["generate_checkpoint_figures"]


def _skip(name: str, reason: str) -> None:
    print(f"    skip {name}: {reason}")


def _step_helpers():
    """
    Phase 1's step-axis helpers, or None.

    Imported lazily and defensively: reaching them pulls in that package's
    whole figure surface, and only this class needs it. A `None` here is a
    reported skip, not a crash — the other nine classes must draw on a
    machine where Phase 1's visualization deps are unavailable.
    """
    try:
        from p1_mstate_tracking.visualization.checkpoints import (
            _step_x, add_step_colorbar, format_step_axis, step_color,
            step_norm,
        )
        return dict(step_x=_step_x, step_norm=step_norm, step_color=step_color,
                    format_step_axis=format_step_axis,
                    add_step_colorbar=add_step_colorbar)
    except Exception as exc:
        _skip("checkpoints", f"Phase 1's step-axis helpers are unavailable "
                             f"({type(exc).__name__}: {exc})")
        return None


def generate_checkpoint_figures(runs: Sequence[Run],
                                out_dir: Path) -> List[Path]:
    """K1-K6, one set per (base model, prompt) family."""
    helpers = _step_helpers()
    if helpers is None:
        return []

    families = checkpoint_families(runs)
    if not families:
        _skip("checkpoints", "no run carries a training step in its model "
                             "name — nothing to place on a step axis")
        return []

    paths: List[Optional[Path]] = []
    for base, by_prompt in sorted(families.items()):
        for prompt, by_step in sorted(by_prompt.items()):
            if len(by_step) < 2:
                _skip(f"{base}/{prompt}",
                      f"only {len(by_step)} checkpoint — a step axis needs 2")
                continue
            tag = f"{base}_{prompt}"
            paths.append(_k1_residual_vs_step(by_step, tag, helpers, out_dir))
            paths.append(_k2_teff_vs_step(by_step, tag, helpers, out_dir))
            paths.append(_k3_residual_depth_by_step(by_step, tag, helpers,
                                                    out_dir))
            paths.append(_k4_margin_vs_step(by_step, tag, helpers, out_dir))
            paths.append(_k5_design_vs_step(by_step, tag, helpers, out_dir))
            paths.append(_k6_sink_ratio_vs_step(by_step, tag, helpers, out_dir))
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# K1 — P-γ1
# ---------------------------------------------------------------------------

def _k1_residual_vs_step(by_step: Dict[int, Run], tag: str, H: dict,
                         out_dir: Path) -> Optional[Path]:
    """
    |final residual| vs training step, with `adjudicate_p_gamma1`'s verdict.

    P-γ1's falsifier is that the step-0 residual is already as large as the
    final one, which would mean the gap between the network and the
    identity-weight null is architectural rather than learned. Both the
    vertical and the time-domain residual are drawn: the vertical one is
    what the prediction was registered on, and the time-domain one is the
    measure that survives late saturation, so a disagreement between the two
    panels is itself worth seeing.
    """
    from p1c_frames.gamma_null import adjudicate_p_gamma1

    steps = sorted(by_step)
    vert = {s: by_step[s].scalar("B", "final_residual") for s in steps}
    time = {s: by_step[s].scalar("B.time_domain", "final_time_residual")
            for s in steps}
    usable = {s: v for s, v in vert.items() if np.isfinite(v)}
    if len(usable) < 2:
        _skip(f"residual_vs_step ({tag})",
              "fewer than two checkpoints carry a final residual")
        return None

    adj = adjudicate_p_gamma1(usable)

    with plt.rc_context(BLOG_STYLE):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9.0, 6.4), sharex=True,
            gridspec_kw=dict(height_ratios=[1.3, 1], hspace=0.14))
        x = H["step_x"](steps)
        ax.plot(x, [abs(vert[s]) if np.isfinite(vert[s]) else np.nan
                    for s in steps],
                color="#12406F", lw=2.4, marker="o", ms=6)
        ax.set_ylabel("|final residual|")
        ax.set_title(f"K1 · P-γ1 — does the residual grow with training? — "
                     f"{tag}")
        verdict_box(ax, adj.get("verdict", ""), loc="upper left")

        ax2.plot(x, [time[s] for s in steps], color=CATEGORICAL[1], lw=2.2,
                 marker="s", ms=5)
        ax2.axhline(0.0, color="#374151", lw=1.2)
        ax2.set_ylabel("final time residual")
        H["format_step_axis"](ax2, steps)
        caption(fig, (
            f"Falsifier check and monotonicity check are reported separately "
            f"by the adjudicator, because the prediction has two clauses and "
            f"they can come apart. grew={adj.get('grew')}, "
            f"monotone={adj.get('monotone')}, falsified={adj.get('falsified')}."))
    return save_figure(fig, out_dir, f"residual_vs_step_{tag}")


# ---------------------------------------------------------------------------
# K2 — does the clock itself change?
# ---------------------------------------------------------------------------

def _k2_teff_vs_step(by_step: Dict[int, Run], tag: str, H: dict,
                     out_dir: Path) -> Optional[Path]:
    """
    T_eff under all three definitions vs step, against t*.

    The question P-γ2 does not ask: training could change how far the
    network integrates, or only what it does with the time. If T_eff is flat
    across training and the residual is not, the resistance is in the
    direction of motion rather than in its amount — which is precisely the
    reading design-1c says the calibrated step forces.
    """
    steps = sorted(by_step)
    if not any(by_step[s].has("A") for s in steps):
        _skip(f"teff_vs_step ({tag})", "no checkpoint carries an A block")
        return None

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        x = H["step_x"](steps)
        for key, art in zip(STEP_DEFS, ("T_eff_displacement",
                                        "T_eff_calibrated",
                                        "T_eff_attn_only")):
            v = [by_step[s].scalar("A", art) for s in steps]
            if not np.isfinite(v).any():
                continue
            ax.plot(x, v, color=STEP_COLORS[key], lw=2.3, marker="o", ms=5,
                    label=STEP_LABELS[key])
        t_stars = [by_step[s].scalar("A", "t_star") for s in steps]
        if np.isfinite(t_stars).any():
            ax.plot(x, t_stars, color="#6B7280", ls=":", lw=1.8, marker="^",
                    ms=5, label="t* (this prompt, per checkpoint)")
        H["format_step_axis"](ax, steps)
        ax.set_ylabel("ODE time")
        ax.set_title(f"K2 · does training change the clock? — {tag}")
        ax.legend(loc="best", fontsize=8.5)
        caption(fig, "t* is drawn per checkpoint rather than as one line: it "
                     "depends on n and on the measured β, and β moves with "
                     "training.")
    return save_figure(fig, out_dir, f"teff_vs_step_{tag}")


# ---------------------------------------------------------------------------
# K3 — depth and training on one image
# ---------------------------------------------------------------------------

def _k3_residual_depth_by_step(by_step: Dict[int, Run], tag: str, H: dict,
                               out_dir: Path) -> Optional[Path]:
    """
    Layer × step heatmap of the residual, diverging at zero.

    The one figure in the phase where both of Phase 1c's axes are visible at
    once. Vertical structure means resistance is a property of depth that
    training amplifies; horizontal structure means it arrives at a
    particular point in training and applies everywhere.
    """
    steps = sorted(by_step)
    series = [by_step[s].series("B.residual") for s in steps]
    if not any(v.size for v in series):
        _skip(f"residual_depth_by_step ({tag})", "no residual series")
        return None

    n_layers = max(v.size for v in series)
    grid = np.full((len(steps), n_layers), np.nan)
    for i, v in enumerate(series):
        grid[i, :v.size] = v

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.6, 0.42 * len(steps) + 2.6))
        im = ax.imshow(grid, aspect="auto", origin="upper", cmap=RESIDUAL_CMAP,
                       norm=residual_norm(grid), interpolation="nearest")
        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels([f"step {s}" for s in steps], fontsize=8)
        ax.set_xlabel("layer")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("residual (blue = resistance)")
        ax.set_title(f"K3 · depth × training — {tag}")
        caption(fig, "Steps are rows in order, not on a log axis: this is a "
                     "categorical stack of checkpoints, and spacing them by "
                     "log-step would imply an interpolation between rows that "
                     "the data does not support.")
    return save_figure(fig, out_dir, f"residual_depth_by_step_{tag}")


# ---------------------------------------------------------------------------
# K4 — P-H1 across training
# ---------------------------------------------------------------------------

def _k4_margin_vs_step(by_step: Dict[int, Run], tag: str, H: dict,
                       out_dir: Path) -> Optional[Path]:
    """
    Minimum margin and the first infeasible layer vs step.

    P-H1 is registered per checkpoint; this is the whole family at once. The
    interesting outcome is a margin trending toward zero with training,
    which would mean the embedding layer is learning to leave the regime
    Lemma 6.4 forces exponential collapse in — the informative failure the
    prediction was stated to expose.
    """
    steps = sorted(by_step)
    mins = [by_step[s].scalar("E", "min_margin") for s in steps]
    firsts = [by_step[s].scalar("E", "first_infeasible_layer", -1)
              for s in steps]
    if not np.isfinite(mins).any():
        _skip(f"margin_vs_step ({tag})", "no checkpoint carries an E block")
        return None

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        x = H["step_x"](steps)
        ax.plot(x, mins, color="#12406F", lw=2.4, marker="o", ms=6,
                label="minimum margin over layers")
        ax.axhline(0.0, color="#374151", lw=1.2)
        ax.set_ylabel("margin")
        H["format_step_axis"](ax, steps)

        ax2 = ax.twinx()
        drawn = [(xi, f) for xi, f in zip(x, firsts) if np.isfinite(f) and f >= 0]
        if drawn:
            ax2.plot([d[0] for d in drawn], [d[1] for d in drawn], ls="",
                     marker="v", ms=9, color=CATEGORICAL[1],
                     label="first infeasible layer")
            ax2.set_ylabel("first infeasible layer")
        else:
            ax2.set_yticks([])
        ax2.grid(False)
        lines, labels = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines + l2, labels + lab2, loc="best", fontsize=8.5)
        ax.set_title(f"K4 · P-H1 across training — {tag}")
        caption(fig, "No infeasible layer at any checkpoint is the expected "
                     "outcome (Wendel gives probability 1 for d > n) and is "
                     "why the margin, not the boolean, is plotted.")
    return save_figure(fig, out_dir, f"margin_vs_step_{tag}")


# ---------------------------------------------------------------------------
# K5 — P-S1, banded
# ---------------------------------------------------------------------------

def _k5_design_vs_step(by_step: Dict[int, Run], tag: str, H: dict,
                       out_dir: Path) -> Optional[Path]:
    """
    Q_k ratio vs step per degree, with the banded P-S1 verdict against
    step 0.

    The ratio is m-comparable (the measurement in `centroids.py`), so
    checkpoints whose clusterings disagree on the centroid count can be
    compared directly — which is what makes this figure possible at all.
    Each degree is summarised by the layer-median ratio, because P-S1 is a
    claim about the configuration and not about a particular depth; F1 is
    where depth structure lives.

    The verdict is `adjudicate_p_s1_banded`, which requires a
    trained-minus-step-0 improvement larger than the random band at the
    trained configuration's own (m, d). Unbanded, random-vs-random returns
    PARTIAL on nothing.
    """
    from p1c_frames.centroids import adjudicate_p_s1_banded

    steps = sorted(by_step)
    per_step = {}
    for s in steps:
        recs = records(by_step[s], "F")
        if not recs:
            continue
        t_max = max((len(r.get("Q_ratio") or []) for r in recs), default=0)
        if not t_max:
            continue
        ratios = record_matrix(recs, "Q_ratio", t_max)
        bands = record_matrix(recs, "random_band", t_max)
        per_step[s] = (np.nanmedian(ratios, axis=0),
                       np.nanmedian(bands, axis=0),
                       int(np.nanmedian(_col(recs, "n_centroids"))))
    if len(per_step) < 2:
        _skip(f"design_vs_step ({tag})",
              "fewer than two checkpoints carry an F block")
        return None

    t_max = max(len(v[0]) for v in per_step.values())
    ordered = sorted(per_step)

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        x = H["step_x"](ordered)
        for k in range(t_max):
            y = [per_step[s][0][k] if k < len(per_step[s][0]) else np.nan
                 for s in ordered]
            b = [per_step[s][1][k] if k < len(per_step[s][1]) else np.nan
                 for s in ordered]
            color = degree_color(k + 1, t_max)
            ax.plot(x, y, color=color, lw=2.3, marker="o", ms=5,
                    label=f"$Q_{k+1}$ ratio (layer median)")
            ax.fill_between(x, np.array(y) - np.array(b),
                            np.array(y) + np.array(b), color=color, alpha=0.14,
                            linewidth=0)
        reference_line(ax, 1.0, "i.i.d. uniform at the same (m, d)",
                       side="left")
        H["format_step_axis"](ax, ordered)
        ax.set_ylabel("$Q_k / Q_k^{\\rm random}$")
        ax.set_title(f"K5 · P-S1 across training — {tag}")
        ax.legend(loc="best", fontsize=8.5)

        # The earliest checkpoint in the family is the baseline. When it is
        # not literally step 0 the comparison is still the right one — but
        # the caption names the step it used rather than saying "step 0",
        # because a family starting at step 1000 would otherwise report a
        # baseline it does not have.
        baseline, trained = ordered[0], ordered[-1]
        t_rec = {"Q_ratio": per_step[trained][0].tolist(),
                 "random_band": per_step[trained][1].tolist(),
                 "n_centroids": per_step[trained][2]}
        z_rec = {"Q_ratio": per_step[baseline][0].tolist(),
                 "n_centroids": per_step[baseline][2]}
        adj = adjudicate_p_s1_banded(t_rec, z_rec)
        verdict_box(ax, adj.get("verdict", ""), loc="lower left")
        caption(fig, (
            f"Banded verdict, step {trained} against step {baseline}: "
            f"{adj.get('banded_verdict', '')} Bands are each degree's own 2σ "
            f"random band — the registered falsifier (\"no difference\") "
            f"carries no threshold, and without one three degrees of sampling "
            f"noise read as a PARTIAL."))
    return save_figure(fig, out_dir, f"design_vs_step_{tag}")


# ---------------------------------------------------------------------------
# K6 — does the rank story survive?
# ---------------------------------------------------------------------------

def _k6_sink_ratio_vs_step(by_step: Dict[int, Run], tag: str, H: dict,
                           out_dir: Path) -> Optional[Path]:
    """
    Minimum normed rank, minimum raw rank and the sink ratio vs step.

    status-1's headline rank row is "MinRank → 2.3 by step 143000". If the
    raw minimum falls across training while the normed minimum does not, the
    row is a statement about outlier token norms and has to be rewritten on
    the frame-correct quantity — which is `adjudicate_sink_hypothesis`'s
    SINKS verdict, drawn here as a trend rather than as a per-run label.
    """
    steps = sorted(by_step)
    raw = [by_step[s].scalar("C.sink_verdict", "min_shannon_raw") for s in steps]
    nrm = [by_step[s].scalar("C.sink_verdict", "min_shannon_normed")
           for s in steps]
    npr = [by_step[s].scalar("C.sink_verdict", "min_norm_pr") for s in steps]
    if not np.isfinite(raw).any():
        _skip(f"sink_ratio_vs_step ({tag})", "no checkpoint carries a C block")
        return None

    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        x = H["step_x"](steps)
        ax.plot(x, raw, color="#12406F", lw=2.4, marker="o", ms=5,
                label="min shannon_raw — status-1's MinRank")
        ax.plot(x, nrm, color=CATEGORICAL[2], lw=2.2, marker="s", ms=5,
                label="min shannon_normed — frame-correct")
        ax.plot(x, npr, color=CATEGORICAL[1], lw=2.0, ls="--", marker="^",
                ms=5, label="min norm_pr — scale only")
        reference_line(ax, 2.0, "the degenerate floor", side="left")
        ax.set_yscale("log")
        H["format_step_axis"](ax, steps)
        ax.set_ylabel("minimum effective rank over layers")
        ax.set_title(f"K6 · does the rank collapse survive the frame "
                     f"correction? — {tag}")
        ax.legend(loc="best", fontsize=8.5)
        verdicts = {by_step[s].text("C.sink_verdict", "verdict").split()[0]
                    for s in steps
                    if by_step[s].text("C.sink_verdict", "verdict")}
        caption(fig, f"Per-checkpoint verdicts in this family: "
                     f"{', '.join(sorted(verdicts)) or 'none recorded'}. A raw "
                     f"curve falling onto the norm-PR curve is the SINKS "
                     f"reading, drawn as a trend rather than as a label.")
    return save_figure(fig, out_dir, f"sink_ratio_vs_step_{tag}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(recs: Sequence[dict], name: str) -> np.ndarray:
    out = np.full(len(recs), np.nan)
    for i, r in enumerate(recs):
        try:
            out[i] = float(r.get(name))
        except (TypeError, ValueError):
            pass
    return out
