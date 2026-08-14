"""
p1b_hemisphere/visualization/checkpoints_1b.py — the training-step axis.

Four figures (K1-K4). The design had no training-step axis anywhere, so a
Pythia pilot rendered as N unrelated models; `aggregate_by_checkpoint` fixed
that on the reporting side and this is the figure side of it.

Every step-axis convention — `log10(step+1)` positions, the viridis step
colormap, real-step tick labels, family grouping — is imported from
`p1_mstate_tracking/visualization/checkpoints.py` rather than restated, for
the reason that module gives: Pythia's checkpoints are log-spaced to step 512
and linear after, so anything differenced over checkpoint *index* peaks
wherever the release schedule changes spacing rather than wherever training
does. Three phases sharing one implementation is also the only way their
checkpoint figures stay comparable.

K4 is the one that matters most and the one most likely to skip. Angle to the
final-checkpoint axis vs step is what `axis_settling_step` reduces to a
number, and it is the only quantity in this phase tracking the axis's
*direction* rather than λ₂'s magnitude — the thing PREDICTIONS.md claim (b)
is about. It needs the saved activation-space axes, which older runs do not
have.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from p1b_hemisphere.axis_identity import axis_settling_step, cross_checkpoint_axis_rotation
from p1b_hemisphere.p1b_report import AGGREGATED_FIELDS, aggregate

from .loaders import Run, checkpoint_families
from .style import (
    BLOG_STYLE, CATEGORICAL, FIEDLER_CMAP, REGIME_REL_COLORS, REGIME_REL_ORDER,
    SEQ_CMAP, fiedler_norm, no_data, reference_line, save_figure,
)

__all__ = ["generate_checkpoint_figures"]


def _step_conventions():
    """
    Phase 1's step-axis helpers, imported at call time rather than at module
    scope.

    They are the canonical implementation and this module deliberately does
    not carry its own — but reaching them runs
    `p1_mstate_tracking.visualization.__init__`, which pulls in that
    package's whole figure surface (sklearn included). Deferring the import
    means the other seven Phase 1b figure classes do not inherit that
    dependency to draw a bar chart, and a missing one is reported against
    this class alone instead of failing the package import.
    """
    from p1_mstate_tracking.visualization.checkpoints import (
        _step_x, format_step_axis,
    )
    return _step_x, format_step_axis

#: Scalars worth a step-axis panel. A subset of AGGREGATED_FIELDS — the ones
#: whose movement through training is a claim someone might make.
_STEP_FIELDS = (
    "separated_layer_fraction",
    "cone_collapse_layer_fraction",
    "mean_normalized_cone_margin",
    "mean_axis_rotation",
    "mean_stability_score",
    "border_vs_noise_mean_auc",
    "mean_cos_axis_pc1",
    "fraction_never_stable",
)

#: Per-layer quantities worth a layer × step heatmap.
_DEPTH_STEP_FIELDS = ("normalized_margin", "separation_ratio")


def generate_checkpoint_figures(runs: Sequence[Run], out_dir: Path,
                                cross_run: Optional[dict] = None) -> List[Path]:
    """
    Every checkpoint figure, one set per family.

    No-ops cleanly when no run carries a step — pointing this at a GPT-2 /
    ALBERT directory produces nothing rather than a one-point sweep, which is
    the same contract `p2_eigenspectra/visualization` works under.
    """
    try:
        _step_conventions()
    except Exception as exc:
        print(f"  checkpoints: skipped — Phase 1's step-axis conventions are "
              f"unavailable ({exc})")
        return []

    families = checkpoint_families(runs)
    if not families:
        print("  checkpoints: no '-step{N}' family present — skipped")
        return []

    paths: List[Path] = []
    with plt.rc_context(BLOG_STYLE):
        for base, by_step in sorted(families.items()):
            if len(by_step) < 2:
                print(f"  checkpoints: {base} has one step — skipped")
                continue
            print(f"  checkpoints: {base} ({len(by_step)} steps)")
            for fn in (_checkpoint_scalars, _regime_by_step,
                       _checkpoint_depth_heatmap, _axis_settling):
                p = fn(base, by_step, out_dir)
                if isinstance(p, list):
                    paths.extend(x for x in p if x is not None)
                elif p is not None:
                    paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# K1 — every scalar against log-step
# ---------------------------------------------------------------------------

def _checkpoint_scalars(base: str, by_step: Dict[int, List[Run]],
                        out_dir: Path) -> Optional[Path]:
    """
    Each aggregated scalar vs log10(step+1), one panel per scalar.

    Aggregated per step through `p1b_report.aggregate` — the same function
    the cross-run digest uses — so a point here is the same number the
    digest reports for that step, not a second mean computed a second way.
    """
    _step_x, format_step_axis = _step_conventions()
    steps = sorted(by_step)
    aggs = {s: aggregate([r.data for r in by_step[s]]) for s in steps}

    fields = [f for f in _STEP_FIELDS
              if any(aggs[s].get(f"mean_{f}") is not None for s in steps)]
    if not fields:
        return None

    ncol = 2
    nrow = int(np.ceil(len(fields) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 2.9 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()

    x = _step_x(steps)
    for ax, field in zip(axes, fields):
        y = np.array([_f(aggs[s].get(f"mean_{field}")) for s in steps])
        ax.plot(x, y, color=CATEGORICAL[0], linewidth=2.2, marker="o",
                markersize=4)
        ax.set_title(field, fontsize=9.5)
        if np.isfinite(y).any():
            ax.set_ylim(_pad(y))

    for ax in axes[len(fields):]:
        ax.axis("off")
    for ax in axes[max(0, len(fields) - ncol):len(fields)]:
        format_step_axis(ax, steps)

    n_runs = sum(len(v) for v in by_step.values())
    fig.suptitle(f"{base} — Phase 1b scalars through training\n"
                 f"{len(steps)} checkpoints, {n_runs} run(s)", fontsize=13, y=1.0)
    fig.tight_layout()
    return save_figure(fig, out_dir, f"checkpoint_scalars_{_safe(base)}")


# ---------------------------------------------------------------------------
# K2 — the classifier's verdict through training
# ---------------------------------------------------------------------------

def _regime_by_step(base: str, by_step: Dict[int, List[Run]],
                    out_dir: Path) -> Optional[Path]:
    """
    Relative-regime composition vs step, stacked.

    The relative classifier rather than the antipodal one, because under
    cone-collapse the antipodal composition is one flat band at every step
    and would draw a rectangle. Composition rather than a single fraction
    because `graded` and `uniform` are different failures to be separated and
    a single "separated fraction" line cannot tell them apart.
    """
    _step_x, format_step_axis = _step_conventions()
    steps = sorted(by_step)
    classes = list(REGIME_REL_ORDER) + ["invalid"]
    shares = {c: [] for c in classes}

    for s in steps:
        counter: Counter = Counter()
        total = 0
        for run in by_step[s]:
            for label in run.strings("regime_relative"):
                counter[label if label in classes else "invalid"] += 1
                total += 1
        for c in classes:
            shares[c].append(counter[c] / total if total else np.nan)

    x = _step_x(steps)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    bottom = np.zeros(len(steps))
    for c in classes:
        vals = np.nan_to_num(np.array(shares[c]), nan=0.0)
        if vals.sum() == 0:
            continue
        ax.fill_between(x, bottom, bottom + vals, label=c,
                        color=REGIME_REL_COLORS[c], alpha=0.92, linewidth=0)
        bottom += vals

    ax.set_ylim(0, 1)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel("share of layers")
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8.5)
    ax.grid(False)
    format_step_axis(ax, steps)
    ax.set_title(f"{base} — the relative classifier through training",
                 fontsize=12)
    return save_figure(fig, out_dir, f"regime_by_step_{_safe(base)}")


# ---------------------------------------------------------------------------
# K3 — depth and training on one image
# ---------------------------------------------------------------------------

def _checkpoint_depth_heatmap(base: str, by_step: Dict[int, List[Run]],
                              out_dir: Path) -> Optional[Path]:
    """
    Layer × step heatmaps for the two per-layer quantities worth it.

    The step axis is drawn at even spacing with real-step labels rather than
    at log10 positions: an image's columns are categorical, and pretending
    otherwise by stretching cells would misrepresent the sampling. The
    labels carry the real spacing.
    """
    steps = sorted(by_step)
    fields = [f for f in _DEPTH_STEP_FIELDS
              if any(np.isfinite(r.field(f)).any()
                     for s in steps for r in by_step[s])]
    if not fields:
        return None

    n_layers = max(r.n_layers for s in steps for r in by_step[s])
    fig, axes = plt.subplots(len(fields), 1,
                             figsize=(max(8.0, 0.55 * len(steps) + 4.0),
                                      3.2 * len(fields) + 0.8))
    axes = np.atleast_1d(axes)

    for ax, field in zip(axes, fields):
        grid = np.full((n_layers, len(steps)), np.nan)
        for j, s in enumerate(steps):
            vals = [r.field(field) for r in by_step[s]]
            vals = [v for v in vals if np.isfinite(v).any()]
            if not vals:
                continue
            stacked = np.full((len(vals), n_layers), np.nan)
            for i, v in enumerate(vals):
                stacked[i, :len(v)] = v[:n_layers]
            with np.errstate(invalid="ignore"):
                grid[:, j] = np.nanmean(stacked, axis=0)

        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=SEQ_CMAP,
                       interpolation="nearest")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([_fmt(s) for s in steps], fontsize=7.5, rotation=45,
                           ha="right")
        ax.set_ylabel("layer")
        ax.grid(False)
        fig.colorbar(im, ax=ax, pad=0.012, label=field)
        ax.set_title(field, fontsize=10)

    axes[-1].set_xlabel("training step (columns evenly spaced; labels carry "
                        "the real spacing)")
    fig.suptitle(f"{base} — depth × training", fontsize=13, y=1.0)
    fig.tight_layout()
    return save_figure(fig, out_dir, f"checkpoint_depth_heatmap_{_safe(base)}")


# ---------------------------------------------------------------------------
# K4 — when does the axis reach its trained direction
# ---------------------------------------------------------------------------

def _axis_settling(base: str, by_step: Dict[int, List[Run]],
                   out_dir: Path) -> Optional[Path]:
    """
    Angle to the final-checkpoint axis vs step, per layer, with the settling
    step marked.

    Both `cross_checkpoint_axis_rotation` and `axis_settling_step` are
    imported from `axis_identity` — this figure computes nothing. The
    settling step is a real quantity with a real interpretation
    (PREDICTIONS.md claim (b)): settle around 512-2,000 and the axis
    co-locates with the energy-monotonicity break; settle at step 0 and the
    axis is initialisation geometry training never moves; never settle and
    the "stable axis" reading from the GPT-2/ALBERT runs does not transfer.
    """
    _step_x, format_step_axis = _step_conventions()
    steps = sorted(by_step)
    axes_by_step_layer: Dict[int, Dict[int, np.ndarray]] = {}
    n_layers = 0
    for s in steps:
        for run in by_step[s]:
            payload = run.axes()
            if not payload or payload.get("axes") is None:
                continue
            arr = np.asarray(payload["axes"])
            valid = payload.get("valid")
            n_layers = max(n_layers, arr.shape[0])
            for L in range(arr.shape[0]):
                if valid is not None and not bool(np.asarray(valid)[L]):
                    continue
                if np.linalg.norm(arr[L]) < 1e-12:
                    continue
                axes_by_step_layer.setdefault(L, {})[int(s)] = arr[L]
            break   # one run per step: the axis is a property of the model

    usable = {L: d for L, d in axes_by_step_layer.items() if len(d) >= 2}
    if not usable:
        print(f"  checkpoints: K4 skipped for {base} — no saved axes "
              f"(phase1b_*_axes.npz absent; runs predate that emission)")
        return None

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    norm = None
    settle_steps: List[Optional[int]] = []

    for L in sorted(usable):
        rot = cross_checkpoint_axis_rotation(usable[L], reference="final")
        pair_steps = [a for a, _ in rot["pair_steps"]]
        if not pair_steps:
            continue
        color = SEQ_CMAP(0.25 + 0.7 * (L / max(1, n_layers - 1)))
        ax.plot(_step_x(pair_steps), rot["rotation"], color=color,
                linewidth=1.6, alpha=0.85)
        settle_steps.append(axis_settling_step(rot))

    tol = np.pi / 8.0
    ax.axhspan(0, tol, color=CATEGORICAL[2], alpha=0.10, linewidth=0, zorder=0)
    reference_line(ax, tol, "π/8 — the settling tolerance")

    settled = [s for s in settle_steps if s is not None]
    if settled:
        median_settle = int(np.median(settled))
        ax.axvline(_step_x([median_settle])[0], color=CATEGORICAL[1],
                   linewidth=1.8, linestyle="--")
        ax.annotate(f"median settling step: {_fmt(median_settle)}",
                    xy=(_step_x([median_settle])[0], 0.96),
                    xycoords=("data", "axes fraction"), fontsize=9,
                    color=CATEGORICAL[1], ha="left", va="top")
        note = (f"{len(settled)}/{len(settle_steps)} layers settle; "
                f"median step {_fmt(median_settle)}")
    else:
        note = ("no layer's axis stays within π/8 of its final direction — "
                "the stable-axis reading does not transfer to this family")

    all_steps = sorted({s for d in usable.values() for s in d})
    format_step_axis(ax, all_steps)
    ax.set_ylabel("angle to the final-checkpoint axis (rad)")
    ax.set_ylim(0, np.pi / 2)
    ax.set_yticks([0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
    ax.set_yticklabels(["0", "π/8", "π/4", "3π/8", "π/2"])
    ax.set_title(f"{base} — when does the Fiedler axis reach its trained "
                 f"direction?\n{note}  (one line per layer, pale = early)",
                 fontsize=12)
    return save_figure(fig, out_dir, f"axis_settling_{_safe(base)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _pad(y: np.ndarray) -> tuple:
    finite = y[np.isfinite(y)]
    if not finite.size:
        return (0.0, 1.0)
    lo, hi = float(finite.min()), float(finite.max())
    pad = 0.08 * (hi - lo) if hi > lo else max(0.05, abs(hi) * 0.1)
    return (lo - pad, hi + pad)


def _fmt(step: int) -> str:
    return f"{step // 1000}k" if step >= 1000 and step % 1000 == 0 else str(step)


def _safe(name: str) -> str:
    from core.naming import _safe_model_name
    return _safe_model_name(name)
