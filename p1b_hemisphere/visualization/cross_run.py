"""
p1b_hemisphere/visualization/cross_run.py — across models and prompts.

Five figures (V1-V5). These read `phase1b_cross_run.json` where it exists and
fall back to aggregating the per-run summaries in-process where it does not,
using `p1b_report.aggregate` — the phase's own aggregator — so a figure and
`phase1b_cross_run.md` cannot disagree about a mean.

V1 is the odd one out: it renders `global_verdict` as tiles rather than as a
chart. That is deliberate. The verdict is eight booleans and three scalars,
and the right form for eleven labeled values is not a bar chart — it is the
values, laid out, each with the number that supports it and the sample size
it was computed on. `p1b_report.global_verdict` names every boolean for what
True means; the tiles keep those names verbatim rather than paraphrasing
them into something friendlier and less exact.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from p1b_hemisphere.p1b_report import (
    AGGREGATED_FIELDS, LONG_PROMPT_TOKENS, aggregate,
)

from .loaders import Run
from .style import (
    BLOG_STYLE, CATEGORICAL, EVENT_COLORS, EVENT_ORDER, INVALID_COLOR,
    SEQ_CMAP, model_color, no_data, reference_line, save_figure,
)

__all__ = ["generate_cross_run_figures"]

#: The scalars worth a heatmap. A subset of AGGREGATED_FIELDS: the ones whose
#: value is comparable across models and prompts on a shared 0-1 scale.
_HEATMAP_FIELDS = (
    "separated_layer_fraction",
    "cone_collapse_layer_fraction",
    "mean_normalized_cone_margin",
    "mean_axis_rotation",
    "mean_stability_score",
    "fraction_never_stable",
    "border_vs_noise_mean_auc",
    "mean_cos_axis_pc1",
)

#: Verdict keys in the order they should be read: what the bipartition is,
#: then whether containment holds, then what the axis is.
_VERDICT_ORDER = (
    "antipodal_bipartition_present_universally",
    "separated_under_relative_classifier",
    "bipartition_identity_persistent",
    "hdbscan_nested_in_bipartition",
    "cone_collapse_regime_at_long_prompts",
    "cone_collapse_above_dimension_null",
    "axis_redundancy",
    "paper_alignment",
)


def generate_cross_run_figures(runs: Sequence[Run], out_dir: Path,
                               cross_run: Optional[dict] = None) -> List[Path]:
    if not runs:
        return []
    with plt.rc_context(BLOG_STYLE):
        paths = [
            _verdict_card(runs, out_dir, cross_run),
            _model_prompt_heatmap(runs, out_dir),
            _scalar_spread(runs, out_dir),
            _event_counts_by_model(runs, out_dir),
            _prompt_length_vs_cone(runs, out_dir),
        ]
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# V1 — the verdict, as tiles
# ---------------------------------------------------------------------------

def _verdict_card(runs: Sequence[Run], out_dir: Path,
                  cross_run: Optional[dict]) -> Optional[Path]:
    """
    `global_verdict` as labeled tiles, each with its supporting number.

    Recomputed from the runs in view when no digest is on disk, through
    `p1b_report.global_verdict` — so pointing this at a subset of a
    directory gives the verdict for that subset rather than silently
    reporting the whole directory's.
    """
    verdict = (cross_run or {}).get("global_verdict")
    subtitle = "from phase1b_cross_run.json"
    if verdict is None:
        from p1b_hemisphere.p1b_report import global_verdict
        verdict = global_verdict([r.data for r in runs])
        subtitle = f"recomputed over the {len(runs)} run(s) in view"

    support = {
        "cone_collapse_regime_at_long_prompts":
            f"mean cone fraction "
            f"{_num(verdict.get('mean_cone_collapse_fraction_long_prompts'))} "
            f"over {verdict.get('n_long_prompt_runs', 0)} long-prompt run(s), "
            f"threshold {verdict.get('long_prompt_token_threshold', LONG_PROMPT_TOKENS)} tokens",
        "axis_redundancy": _counts_str(verdict.get("axis_redundancy_counts")),
    }

    items = [(k, verdict.get(k)) for k in _VERDICT_ORDER if k in verdict]
    n = len(items)
    ncol = 2
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 1.5 * nrow + 1.0))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, value) in zip(axes, items):
        ax.axis("off")
        color, text = _verdict_style(value)
        ax.add_patch(plt.Rectangle((0.01, 0.08), 0.98, 0.84, transform=ax.transAxes,
                                   facecolor=color, edgecolor="#E5E7EB",
                                   linewidth=1.0, alpha=0.22))
        ax.text(0.035, 0.70, key, transform=ax.transAxes, fontsize=9.5,
                color="#111827", va="center", family="monospace")
        ax.text(0.035, 0.42, text, transform=ax.transAxes, fontsize=13,
                color="#111827", va="center", weight="bold")
        note = support.get(key)
        if note:
            ax.text(0.035, 0.19, note, transform=ax.transAxes, fontsize=7.5,
                    color="#6B7280", va="center")

    for ax in axes[len(items):]:
        ax.axis("off")

    fig.suptitle(f"Phase 1b — global verdict\n{subtitle}", fontsize=13, y=1.0)
    fig.text(0.5, -0.01,
             "Every boolean is named for what True means "
             "(p1b_report.global_verdict). `None` is 'not answerable from "
             "these runs', never a silent False.",
             ha="center", fontsize=8, color="#6B7280")
    return save_figure(fig, out_dir, "verdict_card")


def _verdict_style(value):
    """Tile color and display text. Gray for None — unanswerable, not false."""
    if value is None:
        return INVALID_COLOR, "not answerable"
    if isinstance(value, bool):
        return (CATEGORICAL[2] if value else CATEGORICAL[1]), str(value)
    return CATEGORICAL[0], str(value)


# ---------------------------------------------------------------------------
# V2 — every scalar, model × prompt
# ---------------------------------------------------------------------------

def _model_prompt_heatmap(runs: Sequence[Run], out_dir: Path) -> Optional[Path]:
    """
    One heatmap per aggregated scalar, models down, prompts across.

    Sequential single-hue per panel, each on its own scale with the value
    printed in the cell — the scalars share a 0-1 range but not a meaning,
    and a shared colorbar would invite comparing a stability score to a
    cosine.
    """
    models = sorted({r.model for r in runs})
    prompts = sorted({r.prompt for r in runs})
    if len(models) * len(prompts) < 2:
        return None

    fields = [f for f in _HEATMAP_FIELDS
              if any(r.summary.get(f) is not None for r in runs)]
    if not fields:
        return None

    ncol = 2
    nrow = int(np.ceil(len(fields) / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(3.1 * len(prompts) * ncol + 3.0,
                                      1.0 * len(models) * nrow + 2.2))
    axes = np.atleast_1d(axes).ravel()

    by_key = {(r.model, r.prompt): r for r in runs}
    for ax, field in zip(axes, fields):
        grid = np.full((len(models), len(prompts)), np.nan)
        for i, m in enumerate(models):
            for j, p in enumerate(prompts):
                run = by_key.get((m, p))
                if run is None:
                    continue
                v = run.summary.get(field)
                if v is not None:
                    grid[i, j] = float(v)

        ax.imshow(grid, cmap=SEQ_CMAP, aspect="auto")
        ax.set_xticks(range(len(prompts)))
        ax.set_xticklabels(prompts, fontsize=7.5, rotation=20, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=7.5)
        ax.grid(False)
        ax.set_title(field, fontsize=9)
        for i in range(len(models)):
            for j in range(len(prompts)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            fontsize=7.5, color="#111827")

    for ax in axes[len(fields):]:
        ax.axis("off")

    fig.suptitle("Phase 1b — summary scalars across models and prompts",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    return save_figure(fig, out_dir, "model_prompt_heatmap")


# ---------------------------------------------------------------------------
# V3 — per-model spread over prompts
# ---------------------------------------------------------------------------

def _scalar_spread(runs: Sequence[Run], out_dir: Path) -> Optional[Path]:
    """
    Per-model distribution of each aggregated scalar over prompts.

    A median with its full range rather than a mean bar: with three to nine
    prompts per model the spread across the battery is often the more
    interesting number, and a bar chart of means hides a scalar that is
    prompt-dependent — which for a phase whose verdict has a token-count
    threshold in it is exactly the failure to avoid.
    """
    models = sorted({r.model for r in runs})
    fields = [f for f in AGGREGATED_FIELDS
              if any(r.summary.get(f) is not None for r in runs)]
    if not fields or len(models) < 1:
        return None

    ncol = 3
    nrow = int(np.ceil(len(fields) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.6 * nrow),
                             sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, field in zip(axes, fields):
        for i, m in enumerate(models):
            vals = np.array([r.summary.get(field) for r in runs
                             if r.model == m and r.summary.get(field) is not None],
                            dtype=float)
            if not vals.size:
                continue
            c = model_color(m)
            ax.plot([i, i], [vals.min(), vals.max()], color=c, linewidth=2.0,
                    alpha=0.55, solid_capstyle="round")
            ax.scatter([i], [np.median(vals)], s=42, color=c, zorder=4)
            ax.scatter(np.full(vals.size, i), vals, s=10, color=c, alpha=0.5,
                       zorder=3)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=6.5, rotation=25, ha="right")
        ax.set_title(field, fontsize=9)

    for ax in axes[len(fields):]:
        ax.axis("off")

    fig.suptitle("Phase 1b — per-model spread over the prompt battery\n"
                 "dot = median, bar = full range across prompts",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    return save_figure(fig, out_dir, "scalar_spread")


# ---------------------------------------------------------------------------
# V4 — the event vocabulary, per model
# ---------------------------------------------------------------------------

def _event_counts_by_model(runs: Sequence[Run], out_dir: Path) -> Path:
    """
    Event-type counts per model, stacked.

    Expected to be empty or near-empty under the antipodal `regime_key`. The
    figure states the total on the title so an all-zero chart reads as a
    measurement rather than as a rendering failure.
    """
    models = sorted({r.model for r in runs})
    counts: Dict[str, Counter] = {m: Counter() for m in models}
    for r in runs:
        for ev in r.events:
            counts[r.model][str(ev.get("type", "unknown"))] += 1

    types = list(EVENT_ORDER) + sorted(
        {t for c in counts.values() for t in c if t not in EVENT_ORDER})

    fig, ax = plt.subplots(figsize=(max(7.0, 1.3 * len(models) + 4.0), 4.6))
    bottom = np.zeros(len(models))
    total = 0
    for t in types:
        vals = np.array([counts[m].get(t, 0) for m in models], dtype=float)
        total += int(vals.sum())
        if vals.sum() == 0:
            continue
        ax.bar(range(len(models)), vals, bottom=bottom, width=0.6,
               color=EVENT_COLORS.get(t, INVALID_COLOR), label=t)
        bottom += vals

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("events")
    if total:
        ax.legend(loc="best", fontsize=8.5)
    ax.set_title(f"Phase 1b — Block 1 events by model\n"
                 f"{total} event(s) across {len(runs)} run(s)"
                 + ("" if total else
                    " — foreclosed under the antipodal regime_key (status-1b R4)"),
                 fontsize=12)
    return save_figure(fig, out_dir, "event_counts_by_model")


# ---------------------------------------------------------------------------
# V5 — the verdict's own threshold, on the data it partitions
# ---------------------------------------------------------------------------

def _prompt_length_vs_cone(runs: Sequence[Run], out_dir: Path) -> Optional[Path]:
    """
    n_tokens against cone-collapse fraction, with LONG_PROMPT_TOKENS marked.

    The verdict computes its containment claim over runs above a token-count
    threshold. That threshold is tokenizer-dependent — the same battery under
    the NeoX BPE does not produce the same counts as under GPT-2 BPE — so
    which runs are "long" changes with the model family. This puts the line
    on the data so a reader can see how much of the verdict's sample sits
    near it.
    """
    xs, ys, cs, labs = [], [], [], []
    for r in runs:
        v = r.summary.get("cone_collapse_layer_fraction")
        if v is None:
            continue
        xs.append(r.n_tokens)
        ys.append(float(v))
        cs.append(model_color(r.model))
        labs.append(r.label)
    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    seen = set()
    for x, y, c, lab in zip(xs, ys, cs, labs):
        model = lab.split(" / ")[0]
        ax.scatter([x], [y], s=52, color=c, alpha=0.85, edgecolor="white",
                   linewidth=0.6, label=model if model not in seen else None)
        seen.add(model)
    reference_line(ax, 0.5, "0.5 — the verdict's split/collapse cut")
    ax.axvline(LONG_PROMPT_TOKENS, color="#6B7280", linestyle=":", linewidth=1.2)
    ax.annotate(f"LONG_PROMPT_TOKENS = {LONG_PROMPT_TOKENS}",
                xy=(LONG_PROMPT_TOKENS, 0.02), xycoords=("data", "axes fraction"),
                rotation=90, fontsize=8, color="#6B7280", ha="right", va="bottom")
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("cone-collapse layer fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="best", fontsize=8)

    n_long = sum(1 for x in xs if x > LONG_PROMPT_TOKENS)
    ax.set_title(f"Phase 1b — containment vs. prompt length\n"
                 f"{n_long} of {len(xs)} run(s) count as long", fontsize=12)
    return save_figure(fig, out_dir, "prompt_length_vs_cone")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(v, dp: int = 3) -> str:
    return "n/a" if v is None else f"{float(v):.{dp}f}"


def _counts_str(counts: Optional[dict]) -> str:
    if not counts:
        return ""
    return " · ".join(f"{k}: {v}" for k, v in
                      sorted(counts.items(), key=lambda kv: -kv[1]))
