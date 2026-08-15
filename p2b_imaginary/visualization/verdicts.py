"""
p2b_imaginary/visualization/verdicts.py — Block 1b across the sweep.

`frames.py` draws one (checkpoint, prompt) pair; this draws the table. The
aggregation is `p2b_report.block1b_trajectory`'s, called rather than
reimplemented, and its central design decision drives every figure here:
`n_refused` is reported next to `n_scored` at every step, because a
checkpoint where every run REFUSED looks identical to one where every run
said "inert" if only the verdict tally is read.

That is not hypothetical. 90 of Phase 2 Study B's 243 Pythia runs are
`no_violations`, and steps 8-64 are clean on all 9 prompts — so at the early
end of any real sweep the expected picture is a wall of refusals, and it must
not read as a wall of findings. Every figure in this class draws refusals in
the gray family and counts them separately.

The verdict vocabulary is `rotational_rescaled.VERDICTS`, imported. It
deliberately contains no verdict naming rotation: `rotation_neutral`,
`rotation_contributes` and `rotation_dominant` were all verdicts about the
rotation-only frame, which is an identity and cannot support any of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from p2b_imaginary import p2b_report
from p2b_imaginary.rotational_rescaled import EQUIVALENCE_BAND

from .loaders import Sweep, cross_out, elim_row
from .style import (
    BLOG_STYLE, CATEGORICAL, REFERENCE_LINE, REFUSAL_COLOR, REFUSAL_VERDICTS,
    SEQ_CMAP, STATUS_COLORS, STATUS_MARKERS, VERDICT_COLORS, VERDICT_ORDER,
    format_step_axis, legend_from_classes, no_data, note, save_figure, step_x,
    subtitle,
)

__all__ = ["generate_verdict_figures", "FIGURES"]

FIGURES = ("verdict_matrix", "verdict_composition", "elim_trajectory",
           "refusal_reasons", "truncation_map")


def generate_verdict_figures(sweep: Sweep, out_dir: Path) -> List[Path]:
    """Every `verdicts` figure. Needs at least one scored Block 1b record."""
    if not sweep.with_1b:
        print("  verdicts: skipping — no Block 1b comparison in this sweep "
              "(--blocks 1a, or no Phase 1 run carried activations)")
        return []

    out = cross_out(out_dir)
    with plt.rc_context(BLOG_STYLE):
        return [
            _verdict_matrix(sweep, out),
            _verdict_composition(sweep, out),
            _elim_trajectory(sweep, out),
            _refusal_reasons(sweep, out),
            _truncation_map(sweep, out),
        ]


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def _grid(sweep: Sweep):
    """(steps, prompts) — the axes every matrix figure in this class uses."""
    cks = [c for c in sweep.checkpoints if c.step is not None and c.block1b]
    cks.sort(key=lambda c: c.step)
    prompts = sweep.prompts
    return cks, prompts


def _verdict_of(js: dict) -> str:
    """A record's overall verdict, or the refusal that stands in for one."""
    interp = js.get("interpretation")
    if not interp:
        # `{"status": "failed"}` / `{"status": "no_activations"}` — a record
        # that never produced a comparison. Rendered as "missing" rather than
        # dropped: an absent cell and a refused cell mean different things.
        return "missing"
    return str(interp.get("overall", "missing"))


# ---------------------------------------------------------------------------
# V1
# ---------------------------------------------------------------------------

def _verdict_matrix(sweep: Sweep, out: Path) -> Path:
    """
    V1 — step × prompt, every verdict as a cell.

    The three measurement verdicts take real hues; `both_frames_inert`,
    `no_violations`, `not_comparable` and `missing` take the gray family. A
    reader scanning the image sees at a glance what fraction of the sweep
    produced a number at all, which is the first thing to know about it and
    the thing a verdict TALLY hides.
    """
    cks, prompts = _grid(sweep)
    fig, ax = plt.subplots(figsize=(max(7, 1.0 + 0.85 * len(cks)),
                                    max(3.2, 0.8 + 0.55 * len(prompts)) + 1.6))
    if not cks or not prompts:
        no_data(ax, "no stepped checkpoint with a Block 1b record")
        return save_figure(fig, out, "verdict_matrix")

    for i, ck in enumerate(cks):
        for j, prompt in enumerate(prompts):
            js = ck.block1b.get(prompt)
            verdict = "missing" if js is None else _verdict_of(js)
            ax.add_patch(plt.Rectangle(
                (i - 0.5, j - 0.5), 1, 1,
                facecolor=VERDICT_COLORS.get(verdict, REFUSAL_COLOR),
                edgecolor="white", linewidth=1.5))
            if verdict in REFUSAL_VERDICTS:
                ax.plot([i], [j], marker="x", markersize=7, color="#6B7280",
                        markeredgewidth=1.6)

    ax.set_xlim(-0.5, len(cks) - 0.5)
    ax.set_ylim(-0.5, len(prompts) - 0.5)
    ax.set_xticks(range(len(cks)))
    ax.set_xticklabels([str(c.step) for c in cks], rotation=45, ha="right",
                       fontsize=8)
    ax.set_yticks(range(len(prompts)))
    ax.set_yticklabels(prompts, fontsize=9)
    ax.set_xlabel("training step")
    ax.grid(False)
    ax.set_title("Block 1b verdicts")
    legend_from_classes(ax, list(VERDICT_ORDER) + ["missing"], VERDICT_COLORS,
                        title="verdict", loc="upper left",
                        bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   × marks a cell that is a "
                  "refusal, not a finding")
    note(ax, "No verdict names rotation. Those were verdicts about the "
             "rotation-only frame, which is an identity.", outside=True)
    return save_figure(fig, out, "verdict_matrix")


# ---------------------------------------------------------------------------
# V2
# ---------------------------------------------------------------------------

def _verdict_composition(sweep: Sweep, out: Path) -> Path:
    """
    V2 — verdict composition vs step, with the refusal count beside it.

    `block1b_trajectory` counts `n_refused` separately from the verdict tally
    for exactly the reason this figure exists: at the early end of a real
    sweep every run is `no_violations`, and a stacked bar without the refusal
    band would show that as a solid block of "verdict" the same size as a
    solid block of findings.
    """
    traj = p2b_report.block1b_trajectory(sweep.combined_view)
    rows = traj.get("per_step") or []
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.7, 1]})
    if not rows:
        for ax in axes:
            no_data(ax, "no stepped Block 1b records")
        return save_figure(fig, out, "verdict_composition")

    steps = [r["step"] for r in rows]
    x = step_x(steps)
    width = (float(np.diff(x).min()) * 0.62) if len(x) > 1 else 0.4

    bottoms = np.zeros(len(rows))
    for verdict in VERDICT_ORDER:
        vals = np.array([r["verdicts"].get(verdict, 0) for r in rows],
                        dtype=float)
        if not vals.any():
            continue
        axes[0].bar(x, vals, bottom=bottoms, width=width,
                    color=VERDICT_COLORS.get(verdict, REFUSAL_COLOR),
                    label=verdict,
                    hatch=("//" if verdict in REFUSAL_VERDICTS else None),
                    edgecolor=("#9AA0A6" if verdict in REFUSAL_VERDICTS
                               else "none"))
        bottoms += vals
    axes[0].set_ylabel("prompts")
    axes[0].set_title("Verdict composition over training")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)

    axes[1].bar(x, [r["n_refused"] for r in rows], width=width,
                color=REFUSAL_COLOR, label="refused")
    axes[1].plot(x, [r["n_runs"] for r in rows], color="#374151", marker="o",
                 markersize=4, linewidth=1.6, label="runs at this step")
    axes[1].set_ylabel("prompts")
    axes[1].legend(loc="best", fontsize=8)
    format_step_axis(axes[1], steps)

    subtitle(fig, f"{sweep.base or 'sweep'}   ·   hatched verdicts are "
                  "refusals, counted again below")
    note(axes[1], "A step where every run refused looks identical to one "
                  "where every run said 'inert' — unless the refusals are "
                  "counted separately.")
    return save_figure(fig, out, "verdict_composition")


# ---------------------------------------------------------------------------
# V3
# ---------------------------------------------------------------------------

def _elim_trajectory(sweep: Sweep, out: Path) -> Path:
    """
    V3 — `elim_full` and `elim_signed` over training. The phase's question.

    `e^{−(S+A)} ≠ e^{−S}e^{−A}` unless S and A commute, so these two frames
    genuinely differ and their contrast is the one thing Block 1b measures
    after the rotation-only withdrawal — and it is exactly status-2.md's
    "next experiments" item 2.

    The n behind each point is annotated because the mean of one admissible
    run and the mean of nine are the same marker otherwise, and at the early
    end of a real sweep it is usually the former.
    """
    traj = p2b_report.block1b_trajectory(sweep.combined_view)
    rows = traj.get("per_step") or []
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    if not rows:
        no_data(ax, "no stepped Block 1b records")
        return save_figure(fig, out, "elim_trajectory")

    steps = [r["step"] for r in rows]
    x = step_x(steps)
    drew = False
    for key, color in (("elim_full", CATEGORICAL[0]),
                       ("elim_signed", CATEGORICAL[1])):
        mean = np.array([r.get(f"{key}_mean") if r.get(f"{key}_mean")
                         is not None else np.nan for r in rows], dtype=float)
        lo = np.array([r.get(f"{key}_min") if r.get(f"{key}_min") is not None
                       else np.nan for r in rows], dtype=float)
        hi = np.array([r.get(f"{key}_max") if r.get(f"{key}_max") is not None
                       else np.nan for r in rows], dtype=float)
        if not np.isfinite(mean).any():
            continue
        drew = True
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, marker="o", markersize=4.5,
                linewidth=2.0, label=key)
        for xi, m, r in zip(x, mean, rows):
            if np.isfinite(m):
                ax.annotate(f"n={r.get(f'{key}_n', 0)}", xy=(xi, m),
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=6.5, color=color)

    if not drew:
        no_data(ax, "every elimination rate in this sweep is a refusal — "
                    "see refusal_reasons")
        return save_figure(fig, out, "elim_trajectory")

    ax.axhspan(-EQUIVALENCE_BAND, EQUIVALENCE_BAND, color="#9AA0A6",
               alpha=0.12, linewidth=0)
    ax.axhline(0.0, **REFERENCE_LINE)
    format_step_axis(ax, steps)
    ax.set_ylabel("elimination rate  (unclipped)")
    ax.set_title("Removing all of V against removing only its signed part")
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   band is min–max over "
                  "prompts   ·   refusals contribute nothing, not zero")
    ax.legend(loc="best", fontsize=8.5)
    note(ax, "Points are means over the ADMISSIBLE runs only. A step whose "
             "runs all refused has no point here — check verdict_composition.")
    return save_figure(fig, out, "elim_trajectory")


# ---------------------------------------------------------------------------
# V4
# ---------------------------------------------------------------------------

def _refusal_reasons(sweep: Sweep, out: Path) -> Path:
    """
    V4 — every elimination-rate status in the sweep, counted.

    How much of the table is a refusal, by kind. `different_transitions_
    scored` is the one to watch: it is the rank-gate divergence, it scales
    with ‖V‖, and Study A's OV spectral-norm confound (partial ρ to −0.71) is
    the same quantity — so a sweep dominated by it is a sweep measuring its
    own gate.
    """
    tally: Dict[str, Dict[str, int]] = {}
    for ck in sweep.with_1b:
        for js in ck.block1b_scored().values():
            for name, res in elim_row(js).items():
                status = str((res or {}).get("status", "missing"))
                tally.setdefault(status, {}).setdefault(name, 0)
                tally[status][name] += 1

    fig, ax = plt.subplots(figsize=(9, 4.6))
    if not tally:
        no_data(ax, "no elimination-rate rows in this sweep")
        return save_figure(fig, out, "refusal_reasons")

    statuses = sorted(tally, key=lambda s: (s != "ok", s))
    names = sorted({n for row in tally.values() for n in row})
    x = np.arange(len(statuses))
    bottoms = np.zeros(len(statuses))
    for i, name in enumerate(names):
        vals = np.array([tally[s].get(name, 0) for s in statuses], dtype=float)
        ax.bar(x, vals, bottom=bottoms, width=0.62,
               color=CATEGORICAL[i % len(CATEGORICAL)], label=name)
        bottoms += vals

    for i, s in enumerate(statuses):
        ax.plot([i], [bottoms[i] + 0.15], marker=STATUS_MARKERS.get(s, "v"),
                markersize=9, color=STATUS_COLORS.get(s, REFUSAL_COLOR),
                markeredgecolor="#6B7280", linestyle="none")

    ax.set_xticks(x)
    ax.set_xticklabels(statuses, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("(checkpoint, prompt) rows")
    ax.set_title("How every elimination rate in this sweep resolved")
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   "
                  f"counting rule: {sweep.counting_rule}")
    ax.legend(loc="best", fontsize=8.5)
    note(ax, "`different_transitions_scored` is the rank-gate divergence. It "
             "scales with ‖V‖ — a sweep dominated by it is measuring its "
             "own gate.")
    return save_figure(fig, out, "refusal_reasons")


# ---------------------------------------------------------------------------
# V5
# ---------------------------------------------------------------------------

def _truncation_map(sweep: Sweep, out: Path) -> Path:
    """
    V5 — truncated frames per (step, prompt), and invariance failures.

    Truncation is NOT uniform across prompts: it depends on the trajectory
    the rescaler is applied to, not only on the weights, so the same
    checkpoint can truncate on one prompt and not another. That matters
    because `elim_signed = 1.0` is precisely the value an early-truncating
    signed frame produces for free — `e^{−S}` can overflow and `e^{−A}`, being
    orthogonal, cannot.
    """
    cks, prompts = _grid(sweep)
    fig, ax = plt.subplots(figsize=(max(7, 1.0 + 0.85 * len(cks)),
                                    max(3.2, 0.8 + 0.55 * len(prompts)) + 1.6))
    if not cks or not prompts:
        no_data(ax, "no stepped checkpoint with a Block 1b record")
        return save_figure(fig, out, "truncation_map")

    grid = np.full((len(cks), len(prompts)), np.nan)
    for i, ck in enumerate(cks):
        for j, prompt in enumerate(prompts):
            js = ck.block1b.get(prompt)
            if not js or "frames" not in js:
                continue
            grid[i, j] = sum(1 for f in js["frames"].values()
                             if f.get("truncated"))

    im = ax.imshow(grid.T, aspect="auto", origin="lower", cmap=SEQ_CMAP,
                   vmin=0, extent=(-0.5, len(cks) - 0.5, -0.5,
                                   len(prompts) - 0.5))
    for i, ck in enumerate(cks):
        for j, prompt in enumerate(prompts):
            js = ck.block1b.get(prompt) or {}
            if np.isnan(grid[i, j]):
                ax.plot([i], [j], marker="x", markersize=8, color="#9AA0A6")
                continue
            reasons = sorted({f.get("truncation_reason")
                              for f in (js.get("frames") or {}).values()
                              if f.get("truncated")})
            if reasons:
                ax.annotate("\n".join(str(r) for r in reasons), xy=(i, j),
                            ha="center", va="center", fontsize=6.5,
                            color="#111827")
            if (js.get("invariance") or {}).get("status") == "identity_broken":
                ax.plot([i], [j], marker="*", markersize=12, color="#B45B5B")

    ax.set_xticks(range(len(cks)))
    ax.set_xticklabels([str(c.step) for c in cks], rotation=45, ha="right",
                       fontsize=8)
    ax.set_yticks(range(len(prompts)))
    ax.set_yticklabels(prompts, fontsize=9)
    ax.set_xlabel("training step")
    ax.grid(False)
    ax.set_title("Truncated frames per run")
    fig.colorbar(im, ax=ax, pad=0.02, label="frames truncated")
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   × is a run with no Block 1b "
                  "record   ·   ★ is a broken invariance control")
    note(ax, "e^{−S} can overflow; e^{−A} is orthogonal and cannot. So the "
             "signed frame is the one that truncates, and elim_signed = 1.0 "
             "is what that produces for free.", outside=True)
    return save_figure(fig, out, "truncation_map")
