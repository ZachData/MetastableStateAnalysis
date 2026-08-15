"""
p2b_imaginary/visualization/trajectory.py — Block 1a across checkpoints.

The axis the Pythia rerun exists for. `run_2i.py` was organised model ×
prompt, so 27 checkpoints arrived as 27 unrelated "models" with no step and
no ordering, and the trajectory that is Phase 1's and Phase 2's actual
headline result could not be expressed at all. Every figure here is on the
step axis.

The class answers `PLAN_2b.md`'s open questions 1 and 3 directly:

  1. Does the complex fraction have a developmental trajectory? If it sits
     at ~0.98 from step 0, "84-97% complex" says nothing about training.
     T1 is that figure, and it draws the noise reference rather than leaving
     the reader to eyeball a wiggle — a 27-point series of pure noise has a
     range of about four standard errors by construction.
  3. Is the step 8->16 collapse rotational? Phase 1 open item 4, confined to
     layers 21-23. T5 is that figure, and T3 is the version that would show
     it without anyone having to know which layers to look at.

Every statistic here comes from `p2b_report`. `flatness` in particular is
CALLED, not reimplemented: its scale was got wrong twice while it was being
written (see `PLAN_2b.md`, "A scale error found while demonstrating the
report") and a second implementation in a figure module would be a third
chance to get it wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from p2b_imaginary import p2b_report

from .loaders import Sweep, cross_out, depth_matrix
from .style import (
    BLOG_STYLE, CATEGORICAL, EVENT_SPAN, REFUSAL_COLOR, SEQ_CMAP,
    STEP0_STYLE, add_depth_colorbar, depth_color, depth_norm,
    format_step_axis, no_data, note, reference_line, save_figure, step_x,
    subtitle,
)

__all__ = ["generate_trajectory_figures", "FIGURES", "PER_LAYER_SCALARS",
           "LATE_LAYERS"]

FIGURES = ("complex_fraction_trajectory", "tracked_statistics_panel",
           "depth_step_heatmap", "layer_race", "late_layer_zoom",
           "sweep_coverage", "henrici_hotspot", "angle_modulus_trajectory")

#: The per-layer scalars with a trajectory question attached, and the label
#: each is drawn under. Keys are `per_layer` keys; the summary-level names
#: they aggregate into are `p2b_report.TRACKED_STATISTICS`' business, not
#: this module's.
PER_LAYER_SCALARS = {
    "complex_energy_fraction": "complex energy fraction",
    "henrici_relative":        "Henrici (relative)",
    "theta_mean":              "mean θ  (rad)",
    "frac_repulsive_real_part": "repulsive fraction",
}

#: Phase 1's step 8->16 collapse was confined to these layers (status-1 open
#: item 4). Named here as data rather than typed into a figure, so a model
#: with a different depth gets a printed skip rather than an off-by-many
#: slice.
LATE_LAYERS = (21, 22, 23)


def generate_trajectory_figures(sweep: Sweep, out_dir: Path) -> List[Path]:
    """Every `trajectory` figure. Needs ≥ 2 stepped checkpoints with 1a."""
    out = cross_out(out_dir)
    paths: List[Path] = []

    with plt.rc_context(BLOG_STYLE):
        # Coverage draws even for one checkpoint — "this sweep has one point"
        # is exactly what it is for.
        paths.append(_sweep_coverage(sweep, out))

        if not sweep.has_trajectory:
            print(f"  trajectory: skipping — {len(sweep.stepped)} stepped "
                  "checkpoint(s) with Block 1a; a trajectory needs 2")
            return paths

        paths.append(_complex_fraction_trajectory(sweep, out))
        paths.append(_tracked_statistics_panel(sweep, out))
        paths.append(_depth_step_heatmap(sweep, out))
        paths.append(_layer_race(sweep, out))
        late = _late_layer_zoom(sweep, out)
        if late is not None:
            paths.append(late)
        paths.append(_henrici_hotspot(sweep, out))
        paths.append(_angle_modulus_trajectory(sweep, out))
    return paths


# ---------------------------------------------------------------------------
# Shared drawing
# ---------------------------------------------------------------------------

def _draw_trajectory(ax, traj: dict, color: str, label: str = "",
                     with_band: bool = True) -> None:
    """
    One `collect_trajectory` result on a step axis, with its spread band.

    The band is the ACROSS-LAYER standard deviation at each checkpoint —
    the only error bar Block 1a supplies without re-running anything, and
    the right scale for "did this move": a change between checkpoints
    smaller than the layer-to-layer scatter within one checkpoint is not a
    transition.
    """
    steps = np.asarray(traj["steps"], dtype=float)
    vals = np.asarray(traj["values"], dtype=float)
    spread = np.asarray(traj["spread"], dtype=float)
    x = step_x(steps)

    if with_band and np.isfinite(spread).any():
        ax.fill_between(x, vals - spread, vals + spread, color=color,
                        alpha=0.16, linewidth=0,
                        label="± across-layer sd" if label else None)
    ax.plot(x, vals, color=color, marker="o", markersize=3.8, linewidth=2.0,
            label=label or None)

    # Step 0 is the developmental origin, not the pale end of the sweep.
    if steps.size and steps[0] == 0:
        ax.axvline(x[0], **STEP0_STYLE)


def _mark_events(ax, transitions: Sequence[dict] = p2b_report.KNOWN_TRANSITIONS,
                 label: bool = True) -> None:
    """
    Phase 1's and Phase 2's dated events as spans on the step axis.

    Transcribed in `p2b_report.KNOWN_TRANSITIONS` with their source document,
    and read from there — if one of those dates moves, that table is what
    moves, and every figure follows.
    """
    for i, ev in enumerate(transitions):
        lo, hi = ev["span"]
        ax.axvspan(step_x([lo])[0], step_x([hi])[0], **EVENT_SPAN)
        if label:
            # Two of the seven dated spans share endpoints (energy_break and
            # plateau_onset_flip are both 256->512) and several more sit
            # within a few tenths of a decade of each other, so labels at one
            # height overlap into an unreadable stack. Cycling three heights
            # separates them without a collision solver.
            ax.annotate(ev["key"], xy=(step_x([lo])[0], 0.98 - 0.13 * (i % 3)),
                        xycoords=("data", "axes fraction"), rotation=90,
                        fontsize=6.0, color="#B45B5B", va="top", ha="right")


def _flatness_caption(flat: dict) -> str:
    """
    `flatness`' verdict as a sentence, with the number that decides it.

    `range_excess_over_noise` below 1.0 means the trajectory's range is no
    larger than a series of this length drawn from pure noise would give —
    "it moves" is not supported however nonzero the range happens to be.
    """
    if flat.get("status") == "no_data":
        return "no data"
    exc = flat.get("range_excess_over_noise", float("nan"))
    ins = flat.get("range_in_spreads", float("nan"))
    verdict = ("range is within what pure noise gives at this n — "
               "'it moves' is NOT supported"
               if np.isfinite(exc) and exc < 1.0 else
               "range exceeds the pure-noise expectation at this n")
    return (f"range {flat.get('range', float('nan')):.4g}   ·   "
            f"{exc:.2f}× the noise range   ·   {ins:.2f}× the across-layer "
            f"spread\n{verdict}")


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

def _complex_fraction_trajectory(sweep: Sweep, out: Path) -> Path:
    """
    T1 — the complex energy fraction vs training step. Open question 1.

    If this is flat from step 0, the phase's headline is a fact about square
    matrices and not about training, and everything downstream that treats
    "84-97% complex" as a property of the trained network loses its footing.
    The figure therefore draws the flatness verdict on its face rather than
    leaving it to the report.
    """
    traj = p2b_report.collect_trajectory(sweep.combined_view,
                                         "complex_energy_fraction_mean")
    flat = p2b_report.flatness(traj)

    fig, ax = plt.subplots(figsize=(9, 5.0))
    _mark_events(ax)
    _draw_trajectory(ax, traj, CATEGORICAL[0],
                     label="complex energy fraction (mean over layers)")
    reference_line(ax, 1.0, "a norm-matched Gaussian ≈ 1.0", side="left")

    format_step_axis(ax, traj["steps"])
    ax.set_ylabel("complex energy fraction")
    ax.set_title("Does the complex fraction have a developmental trajectory?")
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   "
                  f"{len(traj['steps'])} checkpoints   ·   Block 1a, "
                  "weights only")
    ax.legend(loc="best", fontsize=8.5)
    note(ax, _flatness_caption(flat))
    return save_figure(fig, out, "complex_fraction_trajectory")


# ---------------------------------------------------------------------------
# T2
# ---------------------------------------------------------------------------

def _tracked_statistics_panel(sweep: Sweep, out: Path) -> Path:
    """
    T2 — every `TRACKED_STATISTICS` entry as a small multiple.

    The registry is `p2b_report`'s, including the one-line "what it would
    mean if this moved" attached to each; the panel titles are those lines,
    truncated. Adding a statistic there adds a panel here.
    """
    stats = [s for s in p2b_report.TRACKED_STATISTICS
             if p2b_report.collect_trajectory(sweep.combined_view, s)["steps"]]
    n = len(stats)
    if not n:
        fig, ax = plt.subplots(figsize=(8, 3))
        no_data(ax, "no Block 1a summary statistics found in this sweep")
        return save_figure(fig, out, "tracked_statistics_panel")

    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.9 * nrows),
                             squeeze=False)
    for i, stat in enumerate(stats):
        ax = axes[i // ncols][i % ncols]
        traj = p2b_report.collect_trajectory(sweep.combined_view, stat)
        flat = p2b_report.flatness(traj)
        _mark_events(ax, label=False)
        _draw_trajectory(ax, traj, CATEGORICAL[i % len(CATEGORICAL)])
        format_step_axis(ax, traj["steps"], max_ticks=5)
        ax.set_xlabel("")
        ax.set_title(stat, fontsize=9.5)
        exc = flat.get("range_excess_over_noise", float("nan"))
        ax.annotate(f"{exc:.2f}× noise range",
                    xy=(0.98, 0.04), xycoords="axes fraction", ha="right",
                    fontsize=7.5,
                    color=("#B45B5B" if np.isfinite(exc) and exc >= 1.0
                           else "#6B7280"))
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    fig.suptitle("Every tracked Block 1a statistic on the training axis",
                 y=1.005)
    subtitle(fig, "shaded spans are the dated events in "
                  "p2b_report.KNOWN_TRANSITIONS   ·   band is ± across-layer sd")
    return save_figure(fig, out, "tracked_statistics_panel")


# ---------------------------------------------------------------------------
# T3
# ---------------------------------------------------------------------------

def _depth_step_heatmap(sweep: Sweep, out: Path) -> Path:
    """
    T3 — layer × step heatmaps for the four per-layer scalars.

    The only figure in the class where a LOCALIZED event can appear at all.
    A scalar trajectory averages over depth, so an event confined to three
    layers out of twenty-four moves the mean by an eighth of its size and
    looks like noise; here it is a stripe.
    """
    keys = list(PER_LAYER_SCALARS)
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 3.0 * len(keys)),
                             squeeze=False)
    for i, key in enumerate(keys):
        ax = axes[i][0]
        steps, mat = depth_matrix(sweep.checkpoints, key)
        if not steps:
            no_data(ax, f"no {key} in this sweep")
            continue
        im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap=SEQ_CMAP,
                       extent=(-0.5, len(steps) - 0.5, -0.5,
                               mat.shape[1] - 0.5))
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right",
                           fontsize=7.5)
        ax.set_ylabel("OV layer")
        ax.set_title(PER_LAYER_SCALARS[key], fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, pad=0.015)
    axes[-1][0].set_xlabel("training step  (checkpoints in order, not to scale)")

    fig.tight_layout()
    fig.suptitle("Depth and training on the same picture", y=1.002)
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   columns are checkpoints in "
                  "training order — the x axis is ordinal here, not log-spaced")
    return save_figure(fig, out, "depth_step_heatmap")


# ---------------------------------------------------------------------------
# T4
# ---------------------------------------------------------------------------

def _layer_race(sweep: Sweep, out: Path) -> Path:
    """
    T4 — every layer's own trajectory, coloured by depth, mean bold.

    Distinguishes "the model moved" from "three layers moved and the mean
    followed them". Which of those is true changes what the trajectory is
    evidence FOR, and the summary scalar cannot tell them apart.
    """
    steps, mat = depth_matrix(sweep.checkpoints, "complex_energy_fraction")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if not steps:
        no_data(ax, "no Block 1a per-layer data in this sweep")
        return save_figure(fig, out, "layer_race")

    x = step_x(steps)
    norm = depth_norm(mat.shape[1])
    for L in range(mat.shape[1]):
        ax.plot(x, mat[:, L], color=depth_color(L, norm), linewidth=1.0,
                alpha=0.75)
    ax.plot(x, np.nanmean(mat, axis=1), color="#111827", linewidth=2.6,
            marker="o", markersize=4, label="mean over layers", zorder=5)

    format_step_axis(ax, steps)
    ax.set_ylabel("complex energy fraction")
    ax.set_title("Do the layers move together?")
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   one line per OV layer, "
                  "coloured by depth")
    ax.legend(loc="best", fontsize=8.5)
    add_depth_colorbar(fig, ax, mat.shape[1])
    return save_figure(fig, out, "layer_race")


# ---------------------------------------------------------------------------
# T5
# ---------------------------------------------------------------------------

def _late_layer_zoom(sweep: Sweep, out: Path) -> Optional[Path]:
    """
    T5 — layers 21-23 against every other layer's band. Open question 3.

    Phase 1's step 8->16 collapse (raw effective rank 6.5 -> 2.1) was
    unpredicted, confined to layers 21-23, and fully recovered by step 512;
    status-1 open item 4 asks whether it is a training event or an LR-warmup
    artifact. If the OV spectrum does nothing there, that is evidence for the
    artifact reading — which makes a flat figure here a result, not a
    non-result.

    Skips on a model too shallow to have those layers rather than slicing
    whatever is at the end of the array.
    """
    steps, mat = depth_matrix(sweep.checkpoints, "complex_energy_fraction")
    if not steps:
        return None
    n_layers = mat.shape[1]
    if max(LATE_LAYERS) >= n_layers:
        print(f"  trajectory: skipping late_layer_zoom — Phase 1's collapse "
              f"is at layers {LATE_LAYERS} and this model has {n_layers} "
              "OV layers")
        return None

    fig, ax = plt.subplots(figsize=(9, 5.0))
    x = step_x(steps)
    others = [L for L in range(n_layers) if L not in LATE_LAYERS]
    band = mat[:, others]
    ax.fill_between(x, np.nanmin(band, axis=1), np.nanmax(band, axis=1),
                    color=REFUSAL_COLOR, alpha=0.35, linewidth=0,
                    label=f"every other layer (min–max, n={len(others)})")
    for L in LATE_LAYERS:
        ax.plot(x, mat[:, L], marker="o", markersize=4, linewidth=2.0,
                label=f"layer {L}")

    ev = next((e for e in p2b_report.KNOWN_TRANSITIONS
               if e["key"] == "late_layer_collapse"), None)
    if ev is not None:
        lo, hi = ev["span"]
        ax.axvspan(step_x([lo])[0], step_x([hi])[0], **EVENT_SPAN)
        ax.annotate(f"{ev['key']}  ({lo}→{hi})\n{ev['quantity']}",
                    xy=(step_x([lo])[0], 0.02), xycoords=("data", "axes fraction"),
                    fontsize=7.5, color="#B45B5B", va="bottom")

    format_step_axis(ax, steps)
    ax.set_ylabel("complex energy fraction")
    ax.set_title("Is Phase 1's step 8→16 collapse rotational?")
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   Phase 1 open item 4   ·   "
                  "status-1.md")
    ax.legend(loc="best", fontsize=8.5)
    note(ax, "A flat figure here is evidence for the LR-warmup reading, "
             "not an absence of result.")
    return save_figure(fig, out, "late_layer_zoom")


# ---------------------------------------------------------------------------
# T6
# ---------------------------------------------------------------------------

def _sweep_coverage(sweep: Sweep, out: Path) -> Path:
    """
    T6 — every step the sweep has, and every step it does not.

    `run_sweep`'s `expected_steps` exists because discovery is a glob over
    `ov_weights_*.npz`: a checkpoint Phase 2 failed to write simply does not
    appear, giving 26 rows instead of 27 with nothing saying which. This
    draws the gap. It also draws per-checkpoint prompt counts and failures,
    because a checkpoint present with zero scored prompts is a different
    failure from an absent one and both look like "no data" downstream.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6]})

    present = sweep.steps
    missing = sweep.missing_checkpoints
    all_steps = sorted(set(present) | set(missing))
    if not all_steps:
        for ax in axes:
            no_data(ax, "no stepped checkpoints in this sweep")
        return save_figure(fig, out, "sweep_coverage")

    ax = axes[0]
    for s in present:
        ax.plot([step_x([s])[0]], [0], marker="|", markersize=22,
                color=CATEGORICAL[0], markeredgewidth=2.2)
    for s in missing:
        ax.plot([step_x([s])[0]], [0], marker="x", markersize=9,
                color="#B45B5B", markeredgewidth=2.2)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.grid(False)
    ax.set_title(f"Checkpoint coverage — {len(present)} present, "
                 f"{len(missing)} with no OV weights")

    ax = axes[1]
    scored, failed = [], []
    by_step = {c.step: c for c in sweep.checkpoints if c.step is not None}
    for s in all_steps:
        c = by_step.get(s)
        scored.append(0 if c is None else len(c.block1b_scored()))
        failed.append(0 if c is None else len(c.failures))
    xs = step_x(all_steps)
    width = (np.diff(xs).min() * 0.6) if len(xs) > 1 else 0.4
    ax.bar(xs, scored, width=width, color=CATEGORICAL[0],
           label="prompts with a Block 1b comparison")
    ax.bar(xs, failed, width=width, bottom=scored, color="#B45B5B",
           label="prompts that failed")
    ax.set_ylabel("prompts")
    format_step_axis(ax, all_steps)
    ax.legend(loc="best", fontsize=8.5)

    subtitle(fig, f"{sweep.base or 'sweep'}   ·   read from "
                  f"{sweep.source}   ·   "
                  f"blocks: {', '.join(sweep.blocks) or 'none'}")
    if sweep.source == "subdirectories":
        note(axes[0], "Reconstructed from per-checkpoint subdirectories — no "
                      "combined file, so 'missing' cannot be known here.")
    return save_figure(fig, out, "sweep_coverage")


# ---------------------------------------------------------------------------
# T7
# ---------------------------------------------------------------------------

def _henrici_hotspot(sweep: Sweep, out: Path) -> Path:
    """
    T7 — where the non-normality maximum sits, checkpoint by checkpoint.

    Phase 2 open item 5: attribution goes 1.00 -> 0.50 -> 0.80 over ~90k
    steps while the violation COUNT stays flat, so something reorganises
    WHICH subspace the violations occupy without changing how many there
    are. Henrici measures how much S and A interact; if its hotspot migrates
    through depth during training, that is a mechanism candidate no scalar
    trajectory can show.
    """
    steps, mat = depth_matrix(sweep.checkpoints, "henrici_relative")
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    if not steps:
        for ax in axes:
            no_data(ax, "no henrici_relative in this sweep")
        return save_figure(fig, out, "henrici_hotspot")

    x = step_x(steps)
    with np.errstate(invalid="ignore"):
        argmax = np.array([np.nan if not np.isfinite(r).any()
                           else float(np.nanargmax(r)) for r in mat])
        peak = np.array([np.nan if not np.isfinite(r).any()
                         else float(np.nanmax(r)) for r in mat])
        median = np.nanmedian(mat, axis=1)

    axes[0].plot(x, argmax, color=CATEGORICAL[4], marker="o", markersize=5,
                 linewidth=1.8)
    axes[0].set_ylabel("argmax layer")
    axes[0].set_ylim(-0.5, mat.shape[1] - 0.5)
    axes[0].set_title("Does the non-normality hotspot migrate through depth?")

    axes[1].plot(x, peak, color=CATEGORICAL[4], marker="^", markersize=4,
                 label="max over layers")
    axes[1].plot(x, median, color="#374151", marker="s", markersize=3.4,
                 linestyle="--", label="median over layers")
    axes[1].set_ylabel("henrici_relative")
    axes[1].legend(loc="best", fontsize=8.5)
    format_step_axis(axes[1], steps)

    subtitle(fig, f"{sweep.base or 'sweep'}   ·   Phase 2 open item 5")
    note(axes[1], "A peak that moves in depth while the median stays flat is "
                  "reorganisation without a change in amount.")
    return save_figure(fig, out, "henrici_hotspot")


# ---------------------------------------------------------------------------
# T8
# ---------------------------------------------------------------------------

def _angle_modulus_trajectory(sweep: Sweep, out: Path) -> Path:
    """
    T8 — mean θ and mean ρ vs step.

    The two quantities a norm-matched Gaussian null has a real opinion about.
    The complex FRACTION is ~1.0 for any square Gaussian, so a z near zero
    there is the expected result and says nothing; a Gaussian's angles are
    near-uniform on [0, π] and its ρ distribution is set by its norm, so
    these two are where a trained matrix can actually differ from noise.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.0), sharex=True)

    theta = p2b_report.collect_trajectory(sweep.combined_view,
                                          "theta_mean_across_layers")
    _mark_events(axes[0], label=False)
    _draw_trajectory(axes[0], theta, CATEGORICAL[0], label="mean θ")
    reference_line(axes[0], np.pi / 2, "π/2  (a Gaussian's mean)")
    axes[0].set_ylabel("θ  (rad)")
    axes[0].set_title("Rotation angle and modulus on the training axis")
    axes[0].legend(loc="best", fontsize=8.5)

    # rho has no summary-level entry in TRACKED_STATISTICS, so it is built
    # from the per-layer column here — the mean over layers, matching how
    # every other trajectory in the report is defined.
    steps, mat = depth_matrix(sweep.checkpoints, "rho_mean")
    if steps:
        x = step_x(steps)
        mean = np.nanmean(mat, axis=1)
        sd = np.nanstd(mat, axis=1)
        axes[1].fill_between(x, mean - sd, mean + sd, color=CATEGORICAL[2],
                             alpha=0.16, linewidth=0,
                             label="± across-layer sd")
        axes[1].plot(x, mean, color=CATEGORICAL[2], marker="o", markersize=3.8,
                     linewidth=2.0, label="mean ρ")
        reference_line(axes[1], 1.0, "ρ = 1")
        axes[1].legend(loc="best", fontsize=8.5)
        format_step_axis(axes[1], steps)
    else:
        no_data(axes[1], "no rho_mean in this sweep")
    axes[1].set_ylabel("ρ")

    subtitle(fig, f"{sweep.base or 'sweep'}   ·   the two statistics the "
                  "Gaussian null is informative about")
    return save_figure(fig, out, "angle_modulus_trajectory")
