"""
p2b_imaginary/visualization/report_fig.py — the cross-checkpoint report,
drawn.

Every figure here calls a `p2b_report` function and draws its return value.
None computes a statistic, and that is load-bearing rather than tidy: the
module they call is deliberately conservative about what a 27-point series
with no repeats can support, and reimplementing any of it in a figure would
be reimplementing exactly the caution.

Three refusals are built into `p2b_report` and are visible in these figures:

  - **no categorical change-point verdicts.** status-2's own headline warns
    that five of its 13 `mixed_or_unattributed` runs sit at `frac_repulsive`
    exactly 0.500 against a strict `> 0.5` guard, so "the verdict label is an
    artifact of where the threshold happens to fall". Every function returns
    a continuous quantity and a rank; so does every figure.
  - **`interval_rank` beside every alignment row.** A large move across a
    dated span means little if every span has one. R3 annotates the rank in
    the cell.
  - **`not_bracketed` rather than a zero.** A span the sweep has fewer than
    two checkpoints inside is unanswerable, and R3 hatches those cells
    rather than colouring them at the zero end of the map — which would be a
    claim.

`flatness`' own scale was got wrong twice while it was being written (see
`PLAN_2b.md`, "A scale error found while demonstrating the report"): first
against the raw layer spread, which is the wrong scale by ~sqrt(n_layers),
then against one standard error, which calls almost every flat trajectory a
transition because a 21-point noise series has a range of ~3.8 standard
errors before any trend at all. R1 draws the third and correct version,
`range_excess_over_noise`, with the 1.0 line that decides it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from p2b_imaginary import p2b_report

from .loaders import Sweep, cross_out
from .style import (
    BLOG_STYLE, CATEGORICAL, ELIM_CMAP, REFERENCE_LINE, REFUSAL_COLOR,
    format_step_axis, no_data, note, reference_line, save_figure, signed_norm,
    step_x, subtitle,
)

__all__ = ["generate_report_figures", "FIGURES", "DEFAULT_CO_MOVEMENT"]

FIGURES = ("flatness_bars", "interval_ranking", "event_alignment",
           "co_movement", "known_transitions_map")

#: R4's default pair. Henrici against the repulsive fraction is the INTERNAL
#: version of Phase 2 open item 5 — the real question is Henrici against
#: Phase 2's own `frac_repulsive`, which is measured on violations after a
#: forward pass and is not in this phase's artifacts. `--external` supplies
#: it; this pair is what can be drawn without one, and R4's caption says so.
DEFAULT_CO_MOVEMENT = ("henrici_relative_mean", "frac_repulsive_real_part_mean")


def generate_report_figures(sweep: Sweep, out_dir: Path,
                            external: Optional[dict] = None) -> List[Path]:
    """
    Every `report` figure.

    `external` is an optional {name, steps, values} dict — a series from
    another phase, wrapped through `p2b_report.external_trajectory` so it
    carries NaN spread rather than silently borrowing Phase 2b's scale.
    """
    out = cross_out(out_dir)
    paths: List[Path] = []

    with plt.rc_context(BLOG_STYLE):
        # The coverage map is drawable from the step list alone, so it draws
        # even when there is no trajectory to speak of — "this sweep cannot
        # address any of the dated events" is the most useful thing it can
        # say and it needs one checkpoint to say it.
        paths.append(_known_transitions_map(sweep, out))

        if not sweep.has_trajectory:
            print(f"  report: skipping — {len(sweep.stepped)} stepped "
                  "checkpoint(s) with Block 1a; the report needs 2")
            return paths

        report = p2b_report.build_report(sweep.combined_view)
        paths.append(_flatness_bars(sweep, report, out))
        paths.append(_interval_ranking(sweep, report, out))
        paths.append(_event_alignment(sweep, report, out))
        paths.append(_co_movement(sweep, out, external=external))
    return paths


# ---------------------------------------------------------------------------
# R1
# ---------------------------------------------------------------------------

def _flatness_bars(sweep: Sweep, report: dict, out: Path) -> Path:
    """
    R1 — does anything move? Two comparisons, both drawn, neither conflated.

    `range_excess_over_noise` is the STATISTICAL comparison: the trajectory's
    range in standard errors of the per-checkpoint mean, divided by the range
    a series of this length drawn from pure noise would give. Below 1.0 the
    trajectory is indistinguishable from noise however nonzero its range is.

    `range_in_spreads` is the SUBSTANTIVE one: the same range against the
    across-LAYER scatter within a checkpoint. A statistic can clear the noise
    floor comfortably and still move less across 143000 steps than it varies
    across depth at any one of them, and that is worth knowing separately.
    """
    flat = report.get("flatness") or {}
    stats = [s for s in flat if flat[s].get("status") != "no_data"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(stats) + 3.0),
                             sharey=True)
    if not stats:
        for ax in axes:
            no_data(ax, "no trajectories in this sweep")
        return save_figure(fig, out, "flatness_bars")

    y = np.arange(len(stats))
    exc = np.array([flat[s].get("range_excess_over_noise", np.nan)
                    for s in stats], dtype=float)
    ins = np.array([flat[s].get("range_in_spreads", np.nan)
                    for s in stats], dtype=float)

    ax = axes[0]
    ax.barh(y, np.nan_to_num(exc), color=[
        CATEGORICAL[0] if np.isfinite(v) and v >= 1.0 else REFUSAL_COLOR
        for v in exc], height=0.62)
    ax.axvline(1.0, color="#B45B5B", linewidth=2.0)
    ax.annotate("1.0 — pure-noise range at this n", xy=(1.0, -0.85),
                fontsize=8, color="#B45B5B", va="bottom", ha="left")
    ax.set_xlabel("range ÷ pure-noise range   (statistical)")
    ax.set_yticks(y)
    ax.set_yticklabels(stats, fontsize=8.5)
    ax.set_title("does it move more than noise?")

    ax = axes[1]
    ax.barh(y, np.nan_to_num(ins), color=CATEGORICAL[2], height=0.62)
    reference_line(ax, 1.0, "", axis="x")
    ax.annotate("1.0 — one across-layer spread", xy=(1.0, -0.85),
                fontsize=8, color="#6B7280", va="bottom", ha="left")
    ax.set_xlabel("range ÷ across-layer spread   (substantive)")
    ax.set_title("does it move more than depth does?")

    # A NaN here is not "it did not move" — it is "there is no dispersion
    # scale for this statistic", which is FIGURES-2b.md NOTE-1 and affects two
    # of the seven. `nan_to_num` would draw both as an empty bar, so the
    # missing ones are labelled in place on both panels.
    for panel, vals in ((axes[0], exc), (axes[1], ins)):
        for yi, v in zip(y, vals):
            if not np.isfinite(v):
                panel.annotate("  no dispersion scale — see NOTE-1",
                               xy=(0, yi), fontsize=7, color="#B45309",
                               va="center")

    n_ck = report.get("n_checkpoints")
    expected = next((flat[s].get("expected_range_in_se_under_noise")
                     for s in stats), float("nan"))
    fig.tight_layout()
    fig.suptitle("Does anything in Block 1a move?", y=1.02)
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   {n_ck} checkpoints   ·   "
                  f"a noise series of this length has a range of ≈"
                  f"{expected:.2f} standard errors by construction")
    return save_figure(fig, out, "flatness_bars")


# ---------------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------------

def _interval_ranking(sweep: Sweep, report: dict, out: Path) -> Path:
    """
    R2 — every interval's change in spread units, ranked, per statistic.

    `interval_deltas` ranks by |delta_in_spreads| rather than by raw delta,
    because the sweep's intervals are wildly unequal in log-step width and an
    interval that moves less than the within-checkpoint layer scatter is not
    a transition however large the raw number looks. The dated spans are
    marked in place so co-location is visible without a second figure.

    `build_report` keeps the top 5 intervals per statistic, so this figure is
    a leaderboard rather than a full profile — which is the right shape for
    the question "which interval, if any" and the wrong one for "how does it
    move", which is T1's.
    """
    intervals = report.get("intervals") or {}
    stats = [s for s in intervals if intervals[s]]
    fig, axes = plt.subplots(len(stats), 1,
                             figsize=(9.5, 1.9 * max(len(stats), 1)),
                             squeeze=False)
    if not stats:
        no_data(axes[0][0], "no intervals in this sweep")
        return save_figure(fig, out, "interval_ranking")

    dated = {tuple(ev["span"]) for ev in p2b_report.KNOWN_TRANSITIONS}
    for i, stat in enumerate(stats):
        ax = axes[i][0]
        rows = intervals[stat]
        labels = [f"{r['span'][0]}→{r['span'][1]}" for r in rows]
        vals = np.array([r.get("delta_in_spreads", np.nan) for r in rows],
                        dtype=float)
        colors = [CATEGORICAL[1] if tuple(r["span"]) in dated
                  else CATEGORICAL[0] for r in rows]
        ax.barh(np.arange(len(rows)), np.nan_to_num(vals), color=colors,
                height=0.6)
        ax.axvline(0.0, **REFERENCE_LINE)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{stat}   ·   top {len(rows)} intervals", fontsize=9.5)
    axes[-1][0].set_xlabel("Δ in across-layer spreads")

    fig.tight_layout()
    fig.suptitle("Which intervals move, and by how much relative to depth "
                 "scatter", y=1.005)
    subtitle(fig, "orange bars are intervals whose endpoints exactly match a "
                  "dated event in p2b_report.KNOWN_TRANSITIONS")
    return save_figure(fig, out, "interval_ranking")


# ---------------------------------------------------------------------------
# R3
# ---------------------------------------------------------------------------

def _event_alignment(sweep: Sweep, report: dict, out: Path) -> Path:
    """
    R3 — statistic × dated event, with the rank in the cell.

    Two numbers per cell, and they answer different questions. The COLOUR is
    `delta_in_spreads`: how much this statistic moved across this event's
    span, in units of within-checkpoint layer scatter. The ANNOTATION is
    `interval_rank`: whether that was the sharpest interval in the sweep or
    the seventeenth. A statistic that moves everywhere produces a strong
    colour and a weak rank, and only the pair distinguishes co-location from
    coincidence.

    `not_bracketed` cells are hatched, never coloured. The sweep has fewer
    than two checkpoints inside those spans, so nothing can be said about
    them at all — and drawing "nothing can be said" at the zero end of a
    diverging map would be a claim that the statistic did not move.
    """
    alignment = report.get("alignment") or {}
    stats = list(alignment)
    if not stats:
        fig, ax = plt.subplots(figsize=(8, 3))
        no_data(ax, "no alignment rows in this sweep")
        return save_figure(fig, out, "event_alignment")

    events = [ev["key"] for ev in p2b_report.KNOWN_TRANSITIONS]
    grid = np.full((len(stats), len(events)), np.nan)
    ranks: Dict[tuple, str] = {}
    unbracketed: List[tuple] = []

    no_scale: List[tuple] = []
    for i, stat in enumerate(stats):
        by_key = {r["key"]: r for r in alignment[stat]}
        for j, key in enumerate(events):
            row = by_key.get(key) or {}
            if row.get("status") != "scored":
                unbracketed.append((i, j))
                continue
            delta = row.get("delta_in_spreads", np.nan)
            if delta is None or not np.isfinite(delta):
                # The event WAS bracketed and the statistic DID move; what is
                # missing is the dispersion scale to express the move in. See
                # NOTE-1 in FIGURES-2b.md: two of the seven tracked statistics
                # map to a per-layer key that does not exist, so their spread
                # is NaN and every spread-relative number for them is too.
                # Drawn as its own state — neither a zero nor an unanswerable
                # span.
                no_scale.append((i, j))
                ranks[(i, j)] = (f"{row.get('interval_rank')}/"
                                 f"{row.get('n_intervals')}")
                continue
            grid[i, j] = delta
            ranks[(i, j)] = (f"{row.get('interval_rank')}/"
                             f"{row.get('n_intervals')}")

    fig, ax = plt.subplots(figsize=(1.35 * len(events) + 4.5,
                                    0.55 * len(stats) + 3.2))
    im = ax.imshow(grid, aspect="auto", cmap=ELIM_CMAP,
                   norm=signed_norm(grid[np.isfinite(grid)]))
    for (i, j), text in ranks.items():
        if (i, j) in no_scale:
            continue
        ax.annotate(f"{grid[i, j]:+.2f}\nrank {text}", xy=(j, i), ha="center",
                    va="center", fontsize=7, color="#111827")
    for i, j in unbracketed:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor="white", edgecolor="#9AA0A6",
                                   hatch="///", linewidth=0.8))
        ax.annotate("not\nbracketed", xy=(j, i), ha="center", va="center",
                    fontsize=6.5, color="#6B7280")
    for i, j in no_scale:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor="white", edgecolor="#D97706",
                                   hatch="..", linewidth=0.8))
        ax.annotate(f"no spread\nrank {ranks[(i, j)]}", xy=(j, i),
                    ha="center", va="center", fontsize=6.5, color="#B45309")

    ax.set_xticks(range(len(events)))
    ax.set_xticklabels(events, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels(stats, fontsize=8.5)
    ax.grid(False)
    ax.set_title("Block 1a statistics against Phase 1's and Phase 2's dated "
                 "events")
    fig.colorbar(im, ax=ax, pad=0.02, label="Δ in across-layer spreads")
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   colour is size, annotation "
                  "is rank among ALL intervals — a statistic that moves "
                  "everywhere ranks low")
    note(ax, "/// = unanswerable: fewer than two checkpoints inside the "
             "span.    ⋯ = bracketed and moved, but with no dispersion scale "
             "to express it in (FIGURES-2b.md NOTE-1).    Neither is a zero.",
         outside=True)
    return save_figure(fig, out, "event_alignment")


# ---------------------------------------------------------------------------
# R4
# ---------------------------------------------------------------------------

def _co_movement(sweep: Sweep, out: Path,
                 external: Optional[dict] = None) -> Path:
    """
    R4 — two trajectories on their shared step grid, and their agreement.

    `co_movement` exists for one question — does Henrici non-normality track
    Phase 2's `frac_repulsive` decay — and says in its own output that it is
    the wrong tool for a causal claim: with 27 points on a shared monotone-ish
    schedule, two quantities that both drift with training will correlate at
    the LEVEL. `spearman_deltas` and `interval_agreement` are the readings to
    look at, and both are printed on the figure rather than left in the JSON.

    Without `--external`, the second series is Phase 2b's own
    `frac_repulsive_real_part_mean`, which is the WEIGHTS-side analogue of
    Phase 2's quantity rather than the quantity itself — measured on the
    operator instead of on violations after a forward pass. The caption says
    so, because the two are easy to conflate and only one of them makes the
    figure an answer to open item 5.
    """
    a_name, b_name = DEFAULT_CO_MOVEMENT
    traj_a = p2b_report.collect_trajectory(sweep.combined_view, a_name)
    if external:
        traj_b = p2b_report.external_trajectory(
            external.get("name", "external"), external["steps"],
            external["values"])
        provenance = f"external series: {traj_b['statistic']}"
    else:
        traj_b = p2b_report.collect_trajectory(sweep.combined_view, b_name)
        provenance = ("both series are Phase 2b's own — the weights-side "
                      "analogue of Phase 2's frac_repulsive, not that "
                      "quantity. Pass --external for the real comparison.")

    co = p2b_report.co_movement(traj_a, traj_b)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    if traj_a["steps"] and traj_b["steps"]:
        ax.plot(step_x(traj_a["steps"]), traj_a["values"], color=CATEGORICAL[0],
                marker="o", markersize=4, linewidth=2.0,
                label=traj_a["statistic"])
        ax2 = ax.twinx()
        ax2.plot(step_x(traj_b["steps"]), traj_b["values"],
                 color=CATEGORICAL[1], marker="s", markersize=4, linewidth=2.0,
                 linestyle="--", label=traj_b["statistic"])
        ax2.set_ylabel(traj_b["statistic"], color=CATEGORICAL[1], fontsize=9)
        ax2.tick_params(axis="y", colors=CATEGORICAL[1])
        ax2.grid(False)
        format_step_axis(ax, traj_a["steps"])
        ax.set_ylabel(traj_a["statistic"], color=CATEGORICAL[0], fontsize=9)
        ax.tick_params(axis="y", colors=CATEGORICAL[0])
        ax.set_title("two trajectories, shared step axis")
        # A twin axis is used HERE and nowhere else in the package: the
        # comparison is explicitly about SHAPE, the two series have different
        # units, and the figure's own summary is the rank statistics beside
        # it rather than the crossing point. Every other figure keeps one
        # scale per axis.
    else:
        no_data(ax, "one of the two trajectories is empty")

    ax = axes[1]
    if co.get("status") != "ok":
        no_data(ax, f"co_movement: {co.get('status')} "
                    f"(n_shared = {co.get('n_shared')})")
    else:
        items = [("spearman_levels", co["spearman_levels"]),
                 ("spearman_deltas", co["spearman_deltas"]),
                 ("interval_agreement", co["interval_agreement"])]
        y = np.arange(len(items))
        ax.barh(y, [v for _, v in items],
                color=[REFUSAL_COLOR, CATEGORICAL[0], CATEGORICAL[2]],
                height=0.55)
        ax.axvline(0.0, **REFERENCE_LINE)
        ax.set_yticks(y)
        ax.set_yticklabels([k for k, _ in items], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(f"agreement over {co['n_shared']} shared steps",
                     fontsize=10)
        for yi, (_, v) in zip(y, items):
            ax.annotate(f"{v:+.3f}", xy=(v, yi), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8.5)
        note(ax, co["caveat"], outside=True)

    fig.tight_layout()
    fig.suptitle("Does one statistic track the other?", y=1.02)
    subtitle(fig, provenance)
    return save_figure(fig, out, "co_movement")


# ---------------------------------------------------------------------------
# R5
# ---------------------------------------------------------------------------

def _known_transitions_map(sweep: Sweep, out: Path) -> Path:
    """
    R5 — the dated events, and which of them this sweep can address at all.

    Read before any statistic. `align_to_transitions` reports `not_bracketed`
    for a span containing fewer than two checkpoints, and this is the figure
    of that rule: each event is a bar on the step axis, coloured by how many
    checkpoints fall inside it, with the sweep's own checkpoints drawn
    beneath. status-1 notes that the effective-rank peak "sits unbracketed
    between 1000 and 3000" — with the anchor schedule rather than a dense
    sweep, that is a normal state of affairs and not a defect.
    """
    events = list(p2b_report.KNOWN_TRANSITIONS)
    steps = sweep.steps
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(events) + 3.0))

    if not events:
        no_data(ax, "no known transitions")
        return save_figure(fig, out, "known_transitions_map")

    for i, ev in enumerate(events):
        lo, hi = ev["span"]
        inside = [s for s in steps if lo <= s <= hi]
        x0, x1 = step_x([lo])[0], step_x([hi])[0]
        color = (REFUSAL_COLOR if len(inside) < 2 else CATEGORICAL[0])
        ax.barh([i], [x1 - x0], left=x0, height=0.5, color=color,
                hatch=("///" if len(inside) < 2 else None),
                edgecolor="#9AA0A6" if len(inside) < 2 else "none")
        ax.annotate(f"  {len(inside)} checkpoint(s) inside"
                    + ("   — NOT BRACKETED" if len(inside) < 2 else ""),
                    xy=(x1, i), fontsize=7.5, va="center",
                    color=("#B45B5B" if len(inside) < 2 else "#6B7280"))

    for s in steps:
        ax.axvline(step_x([s])[0], color="#374151", linewidth=0.7, alpha=0.45,
                   zorder=0)

    ax.set_yticks(range(len(events)))
    ax.set_yticklabels([f"{ev['key']}\n({ev['source']})" for ev in events],
                       fontsize=7.5)
    ax.invert_yaxis()
    format_step_axis(ax, steps or [0, 143000])
    ax.set_title("What this sweep can say anything about")
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   thin vertical lines are "
                  f"this sweep's {len(steps)} checkpoints   ·   spans from "
                  "p2b_report.KNOWN_TRANSITIONS")
    note(ax, "Fewer than two checkpoints inside a span makes it unanswerable "
             "— reported as `not_bracketed`, never as a zero.", outside=True)
    return save_figure(fig, out, "known_transitions_map")
