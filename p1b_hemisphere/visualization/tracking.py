"""
p1b_hemisphere/visualization/tracking.py — Block 1, does the axis hold.

Six figures (T1-T6). Block 1 asks whether the bipartition is the same object
from layer to layer: how far the axis turns per transition, how much of each
half's membership survives, how many tokens cross, and where the phase's
event vocabulary fires.

Two things these figures are built to keep visible.

**The event panel is expected to be empty, and that is the finding.**
Birth/collapse/swap were hardcoded to the `strong_bipartition` label, which
is near-unreachable under cone-collapse, so every persistence length was 0 by
construction while appearing measured (status-1b R4). T4 therefore prints the
`regime_key` the run used and states the foreclosure when the vocabulary is
the antipodal one — an empty panel with no caption is indistinguishable from
a bug.

**A crossref number without its baseline is not a number.** T5 draws axis
rotation at merge transitions beside rotation off them, and crossings at
violation layers beside crossings off them, with n annotated on each bar. The
old summary published the first of each pair without the second; the figure
cannot, because a single bar has nowhere to put the comparison.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from p1b_hemisphere.hemisphere_tracking import IDENTITY_THRESHOLD

from .loaders import Run
from .style import (
    BLOG_STYLE, CATEGORICAL, EVENT_COLORS, EVENT_MARKERS, EVENT_ORDER,
    INVALID_COLOR, REGIME_REL_COLORS, class_strip, depth_axis, no_data,
    reference_line, save_figure,
)

__all__ = ["generate_tracking_figures"]


def generate_tracking_figures(run: Run, out_dir: Path) -> List[Path]:
    with plt.rc_context(BLOG_STYLE):
        paths = [
            _axis_rotation(run, out_dir),
            _match_overlap(run, out_dir),
            _crossing_counts(run, out_dir),
            _event_timeline(run, out_dir),
            _crossref_events(run, out_dir),
            _persistence_length(run, out_dir),
        ]
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# T1 — rotation, per transition and cumulative
# ---------------------------------------------------------------------------

def _axis_rotation(run: Run, out_dir: Path) -> Path:
    """
    Per-transition axis rotation and its running sum.

    The cumulative panel is the one that answers the question. A sequence of
    small rotations all in one direction is an axis that has quietly ended up
    somewhere else; the same rotations alternating in sign are an axis that
    holds. The per-transition panel alone cannot tell those apart.
    """
    n = run.n_layers
    rot = run.field("axis_rotation")
    finite = np.nan_to_num(rot, nan=0.0)
    cumulative = np.cumsum(finite)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True)

    ax = axes[0]
    ax.bar(range(n), rot, color=CATEGORICAL[0], width=0.78, alpha=0.9)
    ax.set_ylabel("rotation per\ntransition (rad)")
    mean = float(np.nanmean(rot)) if np.isfinite(rot).any() else float("nan")
    reference_line(ax, mean, f"mean {mean:.3f} rad")
    ax.set_title(f"{run.label} — how far the Fiedler axis turns with depth",
                 fontsize=12)

    ax = axes[1]
    ax.plot(range(n), cumulative, color=CATEGORICAL[1], linewidth=2.4)
    ax.fill_between(range(n), 0, cumulative, color=CATEGORICAL[1], alpha=0.12)
    reference_line(ax, np.pi / 2, "π/2 — a right angle's worth, in total")
    ax.set_ylabel("cumulative\nrotation (rad)")
    depth_axis(ax, n)
    return save_figure(fig, out_dir, "axis_rotation")


# ---------------------------------------------------------------------------
# T2 — identity persistence
# ---------------------------------------------------------------------------

def _match_overlap(run: Run, out_dir: Path) -> Path:
    """
    Mean per-half Jaccard across each transition, against IDENTITY_THRESHOLD.

    The threshold is imported from `hemisphere_tracking`, because the verdict
    "identity persistent = True" is stated against exactly this constant and a
    figure with its own copy would keep drawing the old line after a change.
    """
    n = run.n_layers
    overlap = run.field("match_overlap")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.axhspan(IDENTITY_THRESHOLD, 1.0, color=CATEGORICAL[2], alpha=0.07,
               zorder=0, linewidth=0)
    ax.plot(range(n), overlap, color=CATEGORICAL[0], linewidth=2.2,
            marker="o", markersize=4)
    reference_line(ax, IDENTITY_THRESHOLD,
                   f"IDENTITY_THRESHOLD = {IDENTITY_THRESHOLD:g}")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("mean per-half Jaccard")
    depth_axis(ax, n)

    below = int(np.nansum(overlap < IDENTITY_THRESHOLD))
    mean = float(np.nanmean(overlap)) if np.isfinite(overlap).any() else float("nan")
    ax.set_title(f"{run.label} — does each half stay the same set?\n"
                 f"mean {mean:.3f}; {below} transition(s) below threshold",
                 fontsize=12)
    return save_figure(fig, out_dir, "match_overlap")


# ---------------------------------------------------------------------------
# T3 — who moves
# ---------------------------------------------------------------------------

def _crossing_counts(run: Run, out_dir: Path) -> Path:
    """
    Tokens changing hemisphere per transition, as a fraction of the prompt.

    Absolute counts are not comparable between a 64-token prompt and a
    148-token one, and both appear in every battery, so the fraction is the
    plotted quantity and the count is on the second axis label only.
    """
    n = run.n_layers
    crossing = run.field("crossing_count")
    frac = crossing / float(run.n_tokens) if run.n_tokens else crossing

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(range(n), frac, color=CATEGORICAL[0], width=0.78, alpha=0.9)
    ax.set_ylabel(f"fraction of the {run.n_tokens} tokens crossing")
    mean = float(np.nanmean(frac)) if np.isfinite(frac).any() else float("nan")
    reference_line(ax, mean, f"mean {mean:.3f}")
    depth_axis(ax, n)
    peak = int(np.nanargmax(frac)) if np.isfinite(frac).any() else -1
    ax.set_title(f"{run.label} — hemisphere crossings per transition"
                 + (f"\nbusiest transition: {peak} → {peak + 1}" if peak >= 0 else ""),
                 fontsize=12)
    return save_figure(fig, out_dir, "crossing_counts")


# ---------------------------------------------------------------------------
# T4 — the event vocabulary
# ---------------------------------------------------------------------------

def _event_timeline(run: Run, out_dir: Path) -> Path:
    """
    Every Block 1 event on the layer axis, one row per type.

    One row per type, with a per-type marker as well as a per-type color, so
    the five-way distinction survives both a colorblind reader and a
    grayscale print. Rows are drawn even when empty: "no swaps" is a
    statement, and a figure that omits the row makes it unsayable.
    """
    n = run.n_layers
    events = run.events
    by_type = {t: [] for t in EVENT_ORDER}
    for ev in events:
        by_type.setdefault(str(ev.get("type", "unknown")), []).append(
            int(ev.get("layer", 0)))

    types = list(EVENT_ORDER) + [t for t in by_type if t not in EVENT_ORDER]
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(types) + 2.4))

    for i, t in enumerate(types):
        layers = by_type.get(t, [])
        ax.axhline(i, color="#E5E7EB", linewidth=1.0, zorder=0)
        if layers:
            ax.scatter(layers, [i] * len(layers),
                       marker=EVENT_MARKERS.get(t, "o"), s=90,
                       color=EVENT_COLORS.get(t, INVALID_COLOR),
                       edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(-0.6, i, f"{t}  ({len(layers)})", ha="right", va="center",
                fontsize=9, color="#374151")

    ax.set_yticks([])
    ax.set_ylim(-0.8, len(types) - 0.2)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel("layer")
    ax.grid(False)

    regime_key = run.data.get("regime_key")
    if regime_key == "regime":
        note = ("regime_key = 'regime' (antipodal) — birth/collapse/swap are "
                "foreclosed under cone-collapse, so an empty row here is the "
                "classifier's, not the geometry's (status-1b R4)")
    elif regime_key:
        note = f"regime_key = {regime_key!r}"
    else:
        note = "regime_key not recorded in this run's artifacts"
    ax.set_title(f"{run.label} — Block 1 events\n{note}", fontsize=11)
    fig.subplots_adjust(left=0.16)
    return save_figure(fig, out_dir, "event_timeline")


# ---------------------------------------------------------------------------
# T5 — the Phase 1 cross-reference, with its baselines
# ---------------------------------------------------------------------------

def _crossref_events(run: Run, out_dir: Path) -> Optional[Path]:
    """
    Rotation at merges vs off them; crossings at violations vs off them.

    Paired bars with n annotated. A mean over three merge transitions and a
    mean over twenty-one others are not the same kind of estimate, and the
    figure says which is which rather than letting two equal-width bars imply
    equal weight.
    """
    x = run.summary.get("crossref_with_phase1") or {}
    if not x:
        return None

    pairs = [
        ("axis rotation (rad)",
         x.get("mean_axis_rotation_at_merge"), x.get("mean_axis_rotation_off_merge"),
         "at merge", "off merge", x.get("n_merges_in_run")),
        ("crossings per transition",
         x.get("mean_crossing_at_violation"), x.get("mean_crossing_off_violation"),
         "at violation", "off violation", x.get("n_violations_in_run")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (ylabel, on, off, on_lab, off_lab, n_on) in zip(axes, pairs):
        vals = [on, off]
        if all(v is None for v in vals):
            no_data(ax, f"{ylabel}: not recorded")
            continue
        vals = [np.nan if v is None else float(v) for v in vals]
        ax.bar([0, 1], vals, color=[CATEGORICAL[1], "#9AA0A6"], width=0.6,
               alpha=0.92)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{on_lab}\n(n={n_on if n_on is not None else '?'})",
                            off_lab])
        ax.set_ylabel(ylabel)

    fig.suptitle(f"{run.label} — do Phase 1's events line up with Phase 1b's?",
                 fontsize=12, y=1.02)
    return save_figure(fig, out_dir, "crossref_events")


# ---------------------------------------------------------------------------
# T6 — persistence, and which vocabulary produced it
# ---------------------------------------------------------------------------

def _persistence_length(run: Run, out_dir: Path) -> Optional[Path]:
    """
    Regime persistence length per layer, labeled with the regime_key.

    Under the antipodal key this is 0 everywhere by construction. That is
    worth drawing exactly once, with the reason in the title, so nobody reads
    a floor of zeros as a measurement — which is the mistake status-1b R4
    records having been made.
    """
    persistence = run.persistence_length
    if persistence is None:
        print(f"    tracking: T6 skipped for {run.stem} — "
              f"persistence_length not in artifacts")
        return None

    n = min(run.n_layers, len(persistence))
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True,
                             gridspec_kw={"height_ratios": [4, 1]})

    ax = axes[0]
    ax.bar(range(n), persistence[:n], color=CATEGORICAL[0], width=0.78,
           alpha=0.9)
    ax.set_ylabel("persistence length\n(layers)")

    regime_key = run.data.get("regime_key") or "regime"
    total = float(np.nansum(persistence[:n]))
    if total == 0.0:
        note = (f"every length is 0 under regime_key={regime_key!r} — under "
                f"cone-collapse the antipodal label never fires, so this is "
                f"foreclosed rather than measured")
    else:
        note = (f"regime_key={regime_key!r}; mean "
                f"{np.nanmean(persistence[:n]):.2f} layers")
    ax.set_title(f"{run.label} — how long a regime label survives\n{note}",
                 fontsize=11)

    class_strip(axes[1], run.strings("regime_relative")[:n], REGIME_REL_COLORS,
                label="regime")
    depth_axis(axes[1], n)
    return save_figure(fig, out_dir, "persistence_length")
