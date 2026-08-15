"""
p2b_imaginary/visualization/nulls.py — the norm-matched Gaussian control.

status-2b's first caveat on the phase's surviving descriptive result: "it is
not known to distinguish trained from random — a norm-matched Gaussian is
~100% complex and no null was run." `rotational_schur.null_comparison` closed
that hole in the code; these figures are where the answer becomes readable.

The construction is `core/nulls.py`'s and the baseline scheme is
`core/pythia_registry.build_pythia_random_baseline`'s — structure destroyed,
Frobenius norm preserved — so this is the project's continuity control rather
than a second notion of "random" invented here.

WHAT TO EXPECT, which matters more here than in most classes. A z near zero
on `complex_energy_fraction` is the EXPECTED result: a Gaussian matrix is
essentially all complex pairs, so the observed fraction sitting inside the
null means "84-97% complex" is a statement about square matrices. That is
not a null result — it is the answer. The live comparisons are `theta_mean`
(a Gaussian's angles are near-uniform on [0, π]) and `henrici_relative` (a
Gaussian is strongly non-normal, so a trained matrix sitting BELOW the null
means training has made V more normal). N4 exists to say this on the figure
rather than in a caption nobody reads.

Nulls are opt-in (`--with-nulls`) and cost `n_null_draws` extra Schur
decompositions per layer, so a real sweep runs them at a subset of
checkpoints. Every figure here handles a partly-populated sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .loaders import Checkpoint, Sweep, checkpoint_out, cross_out, null_stat
from .style import (
    BLOG_STYLE, CATEGORICAL, NULL_BAND, REFUSAL_COLOR, depth_axis,
    format_step_axis, no_data, note, reference_line, save_figure, step_x,
    subtitle,
)

__all__ = ["generate_null_figures", "FIGURES", "EXPECTATIONS"]

FIGURES = ("null_z_depth", "null_percentile_depth", "null_z_trajectory",
           "gaussian_expectation")

#: What a norm-matched Gaussian does to each statistic, and therefore how to
#: read a z near zero. Written down because the same number means opposite
#: things across these three rows, and the phase's headline is the row where
#: "no difference from the null" is the expected outcome.
EXPECTATIONS = {
    "complex_energy_fraction": (
        "A Gaussian is ≈ all complex pairs. z ≈ 0 is EXPECTED, and it means "
        "the 84–97% headline is about square matrices, not about training."),
    "theta_mean": (
        "A Gaussian's angles are near-uniform on [0, π], so its mean sits "
        "near π/2. A real difference here is informative."),
    "henrici_relative": (
        "A Gaussian is strongly non-normal. Sitting BELOW the null means "
        "training has made V more normal — the interesting direction."),
}


def generate_null_figures(sweep: Sweep, out_dir: Path) -> List[Path]:
    """Every `nulls` figure, per checkpoint and across the sweep."""
    paths: List[Path] = []
    with_nulls = [c for c in sweep.checkpoints if c.has_nulls]

    if not with_nulls:
        print("  nulls: skipping — no checkpoint in this sweep carries a "
              "null (Block 1a was run without --with-nulls)")
        return paths

    if len(with_nulls) < len(sweep.with_1a):
        print(f"  nulls: {len(with_nulls)} of {len(sweep.with_1a)} "
              "checkpoints carry a null — the rest ran without --with-nulls")

    with plt.rc_context(BLOG_STYLE):
        for ck in with_nulls:
            d = checkpoint_out(out_dir, ck)
            paths.append(_null_z_depth(ck, d))
            paths.append(_null_percentile_depth(ck, d))
            paths.append(_gaussian_expectation(ck, d))
        if len(with_nulls) >= 2:
            paths.append(_null_z_trajectory(sweep, with_nulls,
                                            cross_out(out_dir)))
        else:
            print("  nulls: skipping null_z_trajectory — one checkpoint with "
                  "a null; a trajectory needs 2")
    return paths


# ---------------------------------------------------------------------------
# N1
# ---------------------------------------------------------------------------

def _null_z_depth(ck: Checkpoint, out: Path) -> Path:
    """
    N1 — observed against the null band, per layer, one panel per statistic.

    Drawn as the observed value on top of a mean ± sd band rather than as a
    z, because a z hides both the magnitude and the null's width, and at 16
    draws the width is the part worth seeing.
    """
    stats = ck.null_statistics()
    fig, axes = plt.subplots(len(stats), 1, figsize=(9, 2.9 * len(stats)),
                             sharex=True, squeeze=False)
    n = len(ck.per_layer)
    x = np.arange(n)

    for i, stat in enumerate(stats):
        ax = axes[i][0]
        obs = null_stat(ck, stat, "observed")
        mu = null_stat(ck, stat, "null_mean")
        sd = null_stat(ck, stat, "null_std")
        ax.fill_between(x, mu - sd, mu + sd, label="null mean ± sd",
                        **NULL_BAND)
        ax.plot(x, mu, color="#9AA0A6", linewidth=1.2, linestyle="--")
        ax.plot(x, obs, color=CATEGORICAL[i % len(CATEGORICAL)], marker="o",
                markersize=3.6, linewidth=2.0, label="observed")
        ax.set_ylabel(stat, fontsize=9)
        ax.legend(loc="best", fontsize=8)
        if stat in EXPECTATIONS:
            note(ax, EXPECTATIONS[stat])

    depth_axis(axes[-1][0], n)
    fig.tight_layout()
    fig.suptitle("Observed against the norm-matched Gaussian null", y=1.003)
    subtitle(fig, f"{ck.label}   ·   "
                  f"{_n_draws(ck)} draws per layer   ·   core/nulls.py")
    return save_figure(fig, out, "null_z_depth")


# ---------------------------------------------------------------------------
# N2
# ---------------------------------------------------------------------------

def _null_percentile_depth(ck: Checkpoint, out: Path) -> Path:
    """
    N2 — the same comparison as a rank rather than a z.

    At 16 draws a z-score is a ratio of two noisy numbers and a percentile is
    not, so this is the honest reading of the two. The 2.5 / 97.5 lines are
    drawn as orientation, NOT as a significance test: with 16 draws the
    percentile is quantized to ~6-point steps and cannot resolve either tail.
    """
    stats = ck.null_statistics()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    n = len(ck.per_layer)
    x = np.arange(n)

    for i, stat in enumerate(stats):
        ax.plot(x, null_stat(ck, stat, "percentile"),
                color=CATEGORICAL[i % len(CATEGORICAL)], marker="o",
                markersize=3.4, linewidth=1.8, label=stat)
    reference_line(ax, 50.0, "50th percentile — inside the null")
    reference_line(ax, 97.5, "97.5", side="left")
    reference_line(ax, 2.5, "2.5", side="left")

    depth_axis(ax, n)
    ax.set_ylabel("percentile within the null")
    ax.set_ylim(-3, 103)
    ax.set_title("Where each layer sits inside its own null")
    subtitle(fig, f"{ck.label}   ·   {_n_draws(ck)} draws per layer")
    ax.legend(loc="best", fontsize=8.5)
    note(ax, f"At {_n_draws(ck)} draws the percentile is quantized to "
             f"~{100.0 / max(_n_draws(ck), 1):.0f}-point steps. The 2.5 / "
             "97.5 lines are orientation, not a test.")
    return save_figure(fig, out, "null_percentile_depth")


# ---------------------------------------------------------------------------
# N3
# ---------------------------------------------------------------------------

def _null_z_trajectory(sweep: Sweep, with_nulls: List[Checkpoint],
                       out: Path) -> Path:
    """
    N3 — z against the null, per statistic, on the training axis.

    The interesting version of open question 1. The raw complex fraction can
    be flat across training while DISTINGUISHABILITY FROM RANDOM is not:
    the null is norm-matched, so as ‖OV‖ changes the null moves with it, and
    a constant observed fraction against a moving null is a real trajectory
    that T1 cannot show.
    """
    stepped = sorted((c for c in with_nulls if c.step is not None),
                     key=lambda c: c.step)
    fig, ax = plt.subplots(figsize=(9, 5.0))
    if len(stepped) < 2:
        no_data(ax, "fewer than two stepped checkpoints carry a null")
        return save_figure(fig, out, "null_z_trajectory")

    steps = [c.step for c in stepped]
    x = step_x(steps)
    stats = stepped[0].null_statistics()
    for i, stat in enumerate(stats):
        z = [float(np.nanmean(null_stat(c, stat, "z_score"))) for c in stepped]
        ax.plot(x, z, color=CATEGORICAL[i % len(CATEGORICAL)], marker="o",
                markersize=4.5, linewidth=2.0, label=stat)
    reference_line(ax, 0.0, "z = 0 — indistinguishable from the null")

    format_step_axis(ax, steps)
    ax.set_ylabel("mean z over layers")
    ax.set_title("Distinguishability from a norm-matched Gaussian, over training")
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   "
                  f"{len(stepped)} checkpoints carry a null")
    ax.legend(loc="best", fontsize=8.5)
    note(ax, "The null is norm-matched, so it moves with ‖OV‖. A flat "
             "observed fraction against a moving null is still a trajectory.")
    return save_figure(fig, out, "null_z_trajectory")


# ---------------------------------------------------------------------------
# N4
# ---------------------------------------------------------------------------

def _gaussian_expectation(ck: Checkpoint, out: Path) -> Path:
    """
    N4 — observed vs null mean per statistic, with what each comparison means.

    The card that stops the headline being misread in either direction. On
    the complex fraction, agreement with the null IS the finding; on θ and
    Henrici, disagreement is. Both readings are printed beside their own
    row, because the alternative is a reader carrying one interpretation
    across three panels.
    """
    stats = ck.null_statistics()
    fig, axes = plt.subplots(1, len(stats), figsize=(4.2 * len(stats), 4.6),
                             squeeze=False)
    for i, stat in enumerate(stats):
        ax = axes[0][i]
        obs = np.nanmean(null_stat(ck, stat, "observed"))
        mu = np.nanmean(null_stat(ck, stat, "null_mean"))
        sd = np.nanmean(null_stat(ck, stat, "null_std"))
        ax.bar([0], [mu], yerr=[sd], width=0.55, color=REFUSAL_COLOR,
               capsize=6, label="null (Gaussian)")
        ax.bar([1], [obs], width=0.55,
               color=CATEGORICAL[i % len(CATEGORICAL)], label="observed")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["null", "observed"])
        ax.set_title(stat, fontsize=9.5)
        ax.annotate(EXPECTATIONS.get(stat, ""), xy=(0.5, -0.28),
                    xycoords="axes fraction", ha="center", va="top",
                    fontsize=7.5, color="#6B7280", wrap=True)

    fig.tight_layout()
    fig.suptitle("Is this a fact about training or about square matrices?",
                 y=1.02)
    subtitle(fig, f"{ck.label}   ·   means over layers   ·   "
                  f"{_n_draws(ck)} draws")
    return save_figure(fig, out, "gaussian_expectation")


def _n_draws(ck: Checkpoint) -> int:
    for rec in ck.per_layer:
        for res in (rec.get("nulls") or {}).values():
            n = res.get("n_draws") or res.get("n_null")
            if n:
                return int(n)
    return 0
