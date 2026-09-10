"""
p2b_imaginary/visualization/heads.py — the operator the model actually applies.

The whole class exists to draw one distinction. Phase 2b's headline —
"OV structurally dominated by complex pairs everywhere: 84-97.5% rotational
energy" — is a statistic of `ov_total = Σ_h W_OV^h`, and that sum is the
effective operator only under a counterfactual the model does not satisfy:
that every head shares an attention pattern. The real update is
`Σ_h α^h X W_OV^h`. So the published number describes a matrix the model
never forms, and whether it also describes any HEAD is a separate question.

`head_circuits.summed_vs_per_head` answers it and does not adjudicate: it
reports the summed value, the per-head distribution, and the gap, because
the disagreement is the finding. `head_agreement` — the fraction of heads
within 0.05 of the summed value — is the one number to read first. Low
agreement means the summed statistic describes no head in the layer, and a
sentence about "OV" was never a sentence about the attention head that
applies it.

Three things make this cheap enough to run everywhere, all from
`head_circuits`' own algebra:

  - `eig(W_O W_V) \\ {0} = eig(W_V W_O)`, so every per-head spectrum is a
    `d_head²` problem rather than `d_model²` — 16 × 64³ against 1024³ per
    layer at 410m,
  - the energy fraction is rank-invariant (|0|² = 0 contributes to neither
    numerator nor denominator), so the per-head and ambient computations
    agree and only the DIMENSION fraction collapses,
  - `S = B_S C` and `A = B_A C` share `C`, so the Frobenius split costs one
    extra `(n, 2k)` matmul.

Absent from a sweep whose OV npz carries no `ov_head{h}_*` arrays, which is
what a weights file written before `weights.py` saved them looks like.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .loaders import Checkpoint, Sweep, checkpoint_out, cross_out
from .style import (
    BLOG_STYLE, CATEGORICAL, REFERENCE_LINE, depth_axis, format_step_axis,
    no_data, note, reference_line, save_figure, step_x, subtitle,
)

__all__ = ["generate_head_figures", "FIGURES", "AGREEMENT_BAND"]

FIGURES = ("summed_vs_per_head", "head_agreement_depth", "head_spectrum_spread",
           "head_agreement_trajectory")

#: `summed_vs_per_head`'s own tolerance for "this head agrees with the summed
#: value". Restated here only as a drawn reference; the number that matters is
#: `head_agreement`, which the phase computes.
AGREEMENT_BAND: float = 0.05


def generate_head_figures(sweep: Sweep, out_dir: Path) -> List[Path]:
    """Every `heads` figure, per checkpoint and across the sweep."""
    with_heads = [c for c in sweep.checkpoints if c.head_circuits]
    if not with_heads:
        print("  heads: skipping — no checkpoint carries per-head circuits "
              "(the OV npz has no ov_head{h}_* arrays, or the sweep ran "
              "--no-heads)")
        return []

    paths: List[Path] = []
    with plt.rc_context(BLOG_STYLE):
        for ck in with_heads:
            d = checkpoint_out(out_dir, ck)
            paths.append(_summed_vs_per_head(ck, d))
            paths.append(_head_agreement_depth(ck, d))
            paths.append(_head_spectrum_spread(ck, d))
        stepped = [c for c in with_heads if c.step is not None]
        if len(stepped) >= 2:
            paths.append(_head_agreement_trajectory(sweep, stepped,
                                                    cross_out(out_dir)))
        else:
            print("  heads: skipping head_agreement_trajectory — "
                  f"{len(stepped)} stepped checkpoint(s) with heads; need 2")
    return paths


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def _rows(ck: Checkpoint) -> List[dict]:
    return list((ck.head_circuits or {}).get("per_layer") or [])


def _col(rows: List[dict], *path) -> np.ndarray:
    """One field per layer, walking a dotted path into each row."""
    out = np.full(len(rows), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        node = row
        for key in path:
            node = (node or {}).get(key) if isinstance(node, dict) else None
        if node is None:
            continue
        try:
            out[i] = float(node)
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# H1
# ---------------------------------------------------------------------------

def _summed_vs_per_head(ck: Checkpoint, out: Path) -> Path:
    """
    H1 — the published statistic against the per-head one, layer by layer.

    The summed curve is what "OV is 84-97.5% complex" measures. The band is
    the per-head min–max and the second curve is their mean. Where the summed
    value sits outside the band, the number describes no head in that layer
    at all — which is a statement about the counterfactual, not about the
    model, and it is the reason this class exists.
    """
    rows = _rows(ck)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    if not rows:
        for ax in axes:
            no_data(ax, "no per-layer head circuits in this checkpoint")
        return save_figure(fig, out, "summed_vs_per_head")

    x = np.arange(len(rows))
    summed = _col(rows, "summed", "complex_energy_fraction")
    mean = _col(rows, "per_head", "complex_energy_fraction_mean")
    lo = _col(rows, "per_head", "complex_energy_fraction_min")
    hi = _col(rows, "per_head", "complex_energy_fraction_max")

    ax = axes[0]
    ax.fill_between(x, lo, hi, color=CATEGORICAL[2], alpha=0.18, linewidth=0,
                    label="per-head min – max")
    ax.plot(x, mean, color=CATEGORICAL[2], marker="s", markersize=3.4,
            linewidth=1.8, label="per-head mean (each head's own core)")
    ax.plot(x, summed, color=CATEGORICAL[1], marker="o", markersize=3.8,
            linewidth=2.2, label="Σ_h W_OV^h  —  the published statistic")
    outside = np.isfinite(summed) & ((summed > hi) | (summed < lo))
    if outside.any():
        ax.plot(x[outside], summed[outside], marker="*", markersize=13,
                linestyle="none", color="#B45B5B",
                label="summed value outside the per-head range")
    ax.set_ylabel("complex energy fraction")
    ax.set_title("Does the published number describe any head?")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    gap = _col(rows, "gap")
    ax.bar(x, gap, color=[CATEGORICAL[1] if g >= 0 else CATEGORICAL[0]
                          for g in np.nan_to_num(gap)], width=0.8)
    ax.axhline(0.0, **REFERENCE_LINE)
    ax.set_ylabel("summed − per-head mean")
    depth_axis(ax, len(rows))

    n_heads = int((ck.head_circuits or {}).get("summary", {}).get("n_heads", 0))
    subtitle(fig, f"{ck.label}   ·   {n_heads} heads per layer   ·   "
                  "the sum is the operator only if every head shares an "
                  "attention pattern")
    note(axes[1], "Neither curve is adjudicated here. The gap is the finding: "
                  "it is how much of the headline survives dropping the "
                  "shared-attention counterfactual.", outside=True)
    return save_figure(fig, out, "summed_vs_per_head")


# ---------------------------------------------------------------------------
# H2
# ---------------------------------------------------------------------------

def _head_agreement_depth(ck: Checkpoint, out: Path) -> Path:
    """
    H2 — what fraction of heads the summed value actually describes, vs depth.

    `head_agreement` is the fraction within `AGREEMENT_BAND` of the summed
    number. At 1.0 the summed statistic is a fair summary of the layer; at 0
    it is a number no head in the layer holds. The spread beneath is the
    other half of the same question — a layer can have high agreement because
    its heads are genuinely alike, or low spread and a summed value displaced
    from all of them.
    """
    rows = _rows(ck)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
    if not rows:
        for ax in axes:
            no_data(ax, "no per-layer head circuits in this checkpoint")
        return save_figure(fig, out, "head_agreement_depth")

    x = np.arange(len(rows))
    agree = _col(rows, "head_agreement")
    spread = _col(rows, "head_spread")

    axes[0].bar(x, agree, color=CATEGORICAL[0], width=0.8)
    reference_line(axes[0], 1.0, "every head agrees")
    reference_line(axes[0], 0.0, "no head agrees", side="left")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel(f"fraction of heads\nwithin ±{AGREEMENT_BAND}")
    axes[0].set_title("How much of each layer the summed statistic describes")

    axes[1].plot(x, spread, color=CATEGORICAL[4], marker="o", markersize=3.4,
                 linewidth=1.8)
    axes[1].set_ylabel("head-to-head sd")
    depth_axis(axes[1], len(rows))

    subtitle(fig, f"{ck.label}   ·   head_agreement from "
                  "head_circuits.summed_vs_per_head")
    note(axes[1], "High spread with low agreement: the heads differ and the "
                  "sum describes none of them. Low spread with low "
                  "agreement: they agree with each other and not with the "
                  "sum.", outside=True)
    return save_figure(fig, out, "head_agreement_depth")


# ---------------------------------------------------------------------------
# H3
# ---------------------------------------------------------------------------

def _head_spectrum_spread(ck: Checkpoint, out: Path) -> Path:
    """
    H3 — the per-head complex fraction, every head drawn, layer by layer.

    H1's band as its individual points. `layer_head_spectra`'s docstring puts
    the case plainly: "sixteen heads with complex fractions from 0.1 to 0.9
    sum to something whose spectrum reports a single middling number." This
    is where that shape is visible rather than inferred from an sd — two
    clusters of heads and one outlier all give the same standard deviation as
    a smooth spread, and they are three different pictures of a layer.
    """
    rows = _rows(ck)
    fig, ax = plt.subplots(figsize=(10, 5.0))
    if not rows:
        no_data(ax, "no per-layer head circuits in this checkpoint")
        return save_figure(fig, out, "head_spectrum_spread")

    drew_points = False
    for i, row in enumerate(rows):
        heads = ((row.get("per_head_values") or [])
                 or _per_head_values(row))
        if heads:
            drew_points = True
            jitter = (np.linspace(-0.22, 0.22, len(heads))
                      if len(heads) > 1 else np.zeros(1))
            ax.plot(i + jitter, heads, marker="o", markersize=4.5,
                    linestyle="none", color=CATEGORICAL[2], alpha=0.8)
        else:
            lo = _col([row], "per_head", "complex_energy_fraction_min")[0]
            hi = _col([row], "per_head", "complex_energy_fraction_max")[0]
            ax.plot([i, i], [lo, hi], color=CATEGORICAL[2], linewidth=3,
                    alpha=0.5, solid_capstyle="butt")

    summed = _col(rows, "summed", "complex_energy_fraction")
    ax.plot(np.arange(len(rows)), summed, color=CATEGORICAL[1], marker="_",
            markersize=14, markeredgewidth=2.5, linestyle="none",
            label="Σ_h W_OV^h")

    depth_axis(ax, len(rows))
    ax.set_ylabel("complex energy fraction (head core)")
    ax.set_title("Every head, against the number that stands for all of them")
    ax.legend(loc="best", fontsize=8.5)
    subtitle(fig, f"{ck.label}   ·   "
             + ("one point per head" if drew_points else
                "per-head min–max only — this record carries no per-head list"))
    return save_figure(fig, out, "head_spectrum_spread")


def _per_head_values(row: dict) -> List[float]:
    """
    Each head's core complex fraction, when the record carries the full list.

    `summed_vs_per_head` summarises to mean/sd/min/max, so this is present
    only when a caller kept `layer_head_spectra`'s `per_head`. Returning an
    empty list rather than raising is what lets H3 fall back to the range.
    """
    heads = row.get("per_head")
    if isinstance(heads, dict):
        heads = heads.get("per_head")
    if not isinstance(heads, list):
        return []
    out = []
    for h in heads:
        v = (h or {}).get("complex_energy_fraction_core")
        if v is not None and np.isfinite(float(v)):
            out.append(float(v))
    return out


# ---------------------------------------------------------------------------
# H4
# ---------------------------------------------------------------------------

def _head_agreement_trajectory(sweep: Sweep, stepped: List[Checkpoint],
                               out: Path) -> Path:
    """
    H4 — does the summed statistic become a better or worse summary over
    training?

    The interesting version of the question, and one no per-checkpoint figure
    asks. If `head_agreement` falls as training proceeds, the heads are
    differentiating and the published number is describing less of the model
    as it gets better — which would make "84-97% complex" a statement whose
    meaning depends on the checkpoint it was measured at.
    """
    stepped = sorted(stepped, key=lambda c: c.step)
    steps = [c.step for c in stepped]
    x = step_x(steps)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
    agree = [float((c.head_circuits or {}).get("summary", {})
                   .get("head_agreement_mean", np.nan)) for c in stepped]
    worst = [float((c.head_circuits or {}).get("summary", {})
                   .get("head_agreement_min", np.nan)) for c in stepped]
    gap = [float((c.head_circuits or {}).get("summary", {})
                 .get("gap_mean", np.nan)) for c in stepped]
    spread = [float((c.head_circuits or {}).get("summary", {})
                    .get("head_spread_mean", np.nan)) for c in stepped]

    axes[0].fill_between(x, worst, agree, color=CATEGORICAL[0], alpha=0.16,
                         linewidth=0, label="worst layer – mean")
    axes[0].plot(x, agree, color=CATEGORICAL[0], marker="o", markersize=4.5,
                 linewidth=2.0, label="mean head agreement")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("head agreement")
    axes[0].set_title("Is the summed statistic a better summary over training?")
    axes[0].legend(loc="best", fontsize=8.5)

    axes[1].plot(x, gap, color=CATEGORICAL[1], marker="o", markersize=4,
                 linewidth=2.0, label="mean gap (summed − per-head)")
    axes[1].plot(x, spread, color=CATEGORICAL[4], marker="s", markersize=3.6,
                 linewidth=1.6, linestyle="--", label="mean head-to-head sd")
    axes[1].axhline(0.0, **REFERENCE_LINE)
    axes[1].set_ylabel("fraction")
    axes[1].legend(loc="best", fontsize=8.5)
    format_step_axis(axes[1], steps)

    subtitle(fig, f"{sweep.base or 'sweep'}   ·   "
                  f"{len(stepped)} checkpoints with per-head circuits")
    note(axes[1], "Falling agreement means the heads are differentiating and "
                  "the published number describes less of the model as "
                  "training proceeds.", outside=True)
    return save_figure(fig, out, "head_agreement_trajectory")
