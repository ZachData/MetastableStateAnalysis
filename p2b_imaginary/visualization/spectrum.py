"""
p2b_imaginary/visualization/spectrum.py — Block 1a at one checkpoint.

Eight figures on the depth axis, all of them reading `block1a.per_layer` and
none of them reading anything else. Block 1a is weights-only — no
activations, no forward pass, no prompts — so this whole class is available
from a `--blocks 1a` sweep, which is the cheapest thing the phase can run.

The class exists to answer, at one checkpoint, the question `status-2b`'s
verdict table states as a single number: "OV structurally dominated by
complex pairs everywhere: 84-97.5% rotational energy". Three things about
that number are visible here and are not visible in the number:

  - it is one of THREE definitions of "rotational fraction" in this phase,
    and S1 draws all three,
  - a fraction hides its denominator, and S2 separates "rotation is large"
    from "the real part is tiny",
  - a norm-matched Gaussian scores ~1.0 on it, which S1 draws as the
    reference line the observed value has to beat to be about training
    rather than about square matrices. The actual null is the `nulls`
    class; this is the free version of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .loaders import Checkpoint, Sweep, checkpoint_out
from .style import (
    BLOG_STYLE, CATEGORICAL, REFERENCE_LINE, REFUSAL_COLOR, depth_axis,
    depth_color, depth_norm, no_data, note, reference_line, save_figure,
    subtitle,
)

__all__ = ["generate_spectrum_figures", "FIGURES"]

FIGURES = ("complex_fraction_depth", "energy_budget_depth", "theta_depth",
           "rho_depth", "repulsive_depth", "henrici_depth",
           "normality_budget", "dims_vs_energy", "plane_spectrum",
           "precision_surface")

#: A norm-matched Gaussian is essentially all complex pairs. Drawn as the
#: reference on every fraction panel, because "97% complex" against a null of
#: ~100% is a different statement from "97% complex" against a null of 50%,
#: and the phase shipped the first one reading like the second.
GAUSSIAN_COMPLEX_FRACTION = 1.0


def generate_spectrum_figures(sweep: Sweep, out_dir: Path) -> List[Path]:
    """Every `spectrum` figure for every checkpoint with Block 1a."""
    paths: List[Path] = []
    for ck in sweep.checkpoints:
        if not ck.block1a:
            print(f"  spectrum: skipping {ck.stem} — "
                  f"{'; '.join(ck.missing) or 'no block1a'}")
            continue
        d = checkpoint_out(out_dir, ck)
        with plt.rc_context(BLOG_STYLE):
            paths.append(_complex_fraction_depth(ck, d))
            paths.append(_energy_budget_depth(ck, d))
            paths.append(_theta_depth(ck, d))
            paths.append(_rho_depth(ck, d))
            paths.append(_repulsive_depth(ck, d))
            paths.append(_henrici_depth(ck, d))
            paths.append(_normality_budget(ck, d))
            paths.append(_dims_vs_energy(ck, d))
            for fn in (_plane_spectrum, _precision_surface):
                p = fn(ck, d)
                if p is not None:
                    paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# S1
# ---------------------------------------------------------------------------

def _complex_fraction_depth(ck: Checkpoint, out: Path) -> Path:
    """
    S1 — all three "rotational fraction" definitions on one depth axis.

    `complex_energy_fraction` counts per eigenvalue (a 2x2 block contributes
    2*rho^2); `complex_energy_fraction_legacy_per_block` counts rho^2 once
    per block, mixing a per-pair total with a per-eigenvalue one, and is the
    convention that produced the published 84-97.5%;
    `dim_complex_fraction` counts DIMENSIONS, which `head_circuits` showed is
    a different question again — rank deficiency destroys it and leaves the
    energy fractions intact.

    Drawing them together is the cheapest guard against the failure this
    phase already had: three definitions in one file, reported under one
    name.
    """
    fig, ax = plt.subplots(figsize=(9, 4.6))
    n = len(ck.per_layer)
    x = np.arange(n)

    ax.plot(x, ck.field("complex_energy_fraction"), color=CATEGORICAL[0],
            marker="o", markersize=3.4, linewidth=2.0,
            label="per-eigenvalue energy  (the corrected convention)")
    ax.plot(x, ck.field("complex_energy_fraction_legacy_per_block"),
            color=CATEGORICAL[1], marker="s", markersize=3.2, linewidth=1.8,
            linestyle="--",
            label="legacy per-block energy  (the published 84-97.5%)")
    ax.plot(x, ck.field("dim_complex_fraction"), color=CATEGORICAL[2],
            marker="^", markersize=3.2, linewidth=1.8, linestyle=":",
            label="dimension fraction  (2·n_complex / d)")

    reference_line(ax, GAUSSIAN_COMPLEX_FRACTION,
                   "a norm-matched Gaussian ≈ 1.0", side="left")
    depth_axis(ax, n)
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.02, 1.08)
    ax.set_title("Rotational fraction vs depth — three definitions")
    subtitle(fig, f"{ck.label}   ·   Block 1a, weights only")
    ax.legend(loc="lower right", fontsize=8.5)
    note(ax, "The three lines answer three different questions. The gap "
             "between the first two is a counting convention, not a result.")
    return save_figure(fig, out, "complex_fraction_depth")


# ---------------------------------------------------------------------------
# S2
# ---------------------------------------------------------------------------

def _energy_budget_depth(ck: Checkpoint, out: Path) -> Path:
    """
    S2 — complex and real eigenvalue energy in absolute terms, plus ‖OV‖_F.

    A fraction near 1.0 has two very different causes and the fraction cannot
    tell them apart: rotation is large, or the real part is tiny. Two panels
    rather than two y-scales on one — a twin axis makes a crossing look like
    a fact about the data when it is a fact about the scaling.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    n = len(ck.per_layer)
    x = np.arange(n)

    cplx = ck.field("complex_energy")
    real = ck.field("real_energy")
    axes[0].plot(x, cplx, color=CATEGORICAL[0], marker="o", markersize=3.2,
                 label="complex energy  Σ 2ρ²")
    axes[0].plot(x, real, color=CATEGORICAL[1], marker="s", markersize=3.0,
                 label="real energy  Σ λ²")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("eigenvalue energy (log)")
    axes[0].legend(loc="best", fontsize=8.5)
    axes[0].set_title("What the rotational fraction is a fraction of")

    axes[1].plot(x, ck.field("ov_frob"), color="#374151", marker="D",
                 markersize=3.0)
    axes[1].set_ylabel("‖OV‖_F")
    depth_axis(axes[1], n)

    subtitle(fig, f"{ck.label}   ·   Block 1a, weights only")
    note(axes[0], "A fraction near 1 can mean rotation is large or the real "
                  "part is small. Only the absolute panel separates them.")
    return save_figure(fig, out, "energy_budget_depth")


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def _theta_depth(ck: Checkpoint, out: Path) -> Path:
    """
    S3 — rotation angle vs depth, with the min-max band.

    θ is on [0, π] since the rewrite. The previous version computed
    `arctan2(sqrt(-bc), abs(a))`, folding repulsive rotations onto their
    reflections and pinning the range to [0, π/2] — so a mean near π/2 is
    both what a Gaussian genuinely gives AND what the bug used to produce,
    and the band is what distinguishes them: a folded distribution cannot
    exceed π/2 at all.
    """
    fig, ax = plt.subplots(figsize=(9, 4.6))
    n = len(ck.per_layer)
    x = np.arange(n)

    lo, hi = ck.field("theta_min"), ck.field("theta_max")
    ax.fill_between(x, lo, hi, color=CATEGORICAL[0], alpha=0.16, linewidth=0,
                    label="min – max over planes")
    ax.plot(x, ck.field("theta_mean"), color=CATEGORICAL[0], marker="o",
            markersize=3.4, linewidth=2.0, label="mean θ")
    ax.plot(x, ck.field("theta_median"), color=CATEGORICAL[0], linewidth=1.2,
            linestyle="--", alpha=0.75, label="median θ")

    reference_line(ax, np.pi / 2, "π/2  (a Gaussian's mean)")
    reference_line(ax, np.pi, "π  (reachable only since the θ fix)",
                   side="left")
    depth_axis(ax, n)
    ax.set_ylabel("rotation angle θ  (rad)")
    ax.set_ylim(0, np.pi * 1.06)
    ax.set_title("Rotation angle vs depth")
    subtitle(fig, f"{ck.label}   ·   θ on [0, π]")
    ax.legend(loc="best", fontsize=8.5)
    note(ax, "Any band reaching above π/2 could not have been produced by "
             "the pre-rewrite folded angle.")
    return save_figure(fig, out, "theta_depth")


# ---------------------------------------------------------------------------
# S4
# ---------------------------------------------------------------------------

def _rho_depth(ck: Checkpoint, out: Path) -> Path:
    """
    S4 — eigenvalue modulus vs depth, with the ρ > 1 fraction beneath.

    `frac_rho_above_one` is a threshold on a SCALE convention (how OV was
    normalized), not on a dynamical property. It is drawn here, named for
    exactly what it is, and kept in a separate panel from the dynamical
    quantity — which is S5's `frac_repulsive_real_part`. The previous
    version reported the first under a name that implied the second.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    n = len(ck.per_layer)
    x = np.arange(n)

    mean, sd = ck.field("rho_mean"), ck.field("rho_std")
    axes[0].fill_between(x, mean - sd, mean + sd, color=CATEGORICAL[2],
                         alpha=0.18, linewidth=0, label="± sd over planes")
    axes[0].plot(x, mean, color=CATEGORICAL[2], marker="o", markersize=3.4,
                 linewidth=2.0, label="mean ρ")
    reference_line(axes[0], 1.0, "ρ = 1")
    axes[0].set_ylabel("eigenvalue modulus ρ")
    axes[0].legend(loc="best", fontsize=8.5)
    axes[0].set_title("Eigenvalue modulus vs depth")

    axes[1].bar(x, ck.field("frac_rho_above_one"), color=CATEGORICAL[3],
                width=0.8)
    axes[1].set_ylabel("frac ρ > 1")
    axes[1].set_ylim(0, 1)
    depth_axis(axes[1], n)

    subtitle(fig, f"{ck.label}   ·   Block 1a, weights only")
    note(axes[1], "ρ > 1 is a threshold on the normalization convention, not "
                  "a dynamical property — see repulsive_depth for that.")
    return save_figure(fig, out, "rho_depth")


# ---------------------------------------------------------------------------
# S5
# ---------------------------------------------------------------------------

def _repulsive_depth(ck: Checkpoint, out: Path) -> Path:
    """
    S5 — the fraction of rotation planes with Re λ < 0, vs depth.

    THE dynamical quantity in Block 1a: `e^{−V}` grows in the directions
    where Re λ < 0, so this is what a rescaled frame actually responds to,
    and it is the weights-side analogue of Phase 2's `frac_repulsive` —
    which is measured on violations, on the other side of the forward pass.
    Whether these two track each other is `p2b_report.co_movement`'s
    question and the `report` class's R4.
    """
    fig, ax = plt.subplots(figsize=(9, 4.6))
    n = len(ck.per_layer)
    x = np.arange(n)

    rep = ck.field("frac_repulsive_real_part")
    att = ck.field("frac_attractive_real_part")
    ax.plot(x, rep, color=CATEGORICAL[1], marker="o", markersize=3.4,
            linewidth=2.0, label="repulsive  (Re λ < 0)")
    ax.plot(x, att, color=CATEGORICAL[0], marker="s", markersize=3.0,
            linewidth=1.6, linestyle="--", alpha=0.8,
            label="attractive  (Re λ > 0)")
    reference_line(ax, 0.5, "0.5  (no preference)")

    depth_axis(ax, n)
    ax.set_ylabel("fraction of rotation planes")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Repulsive rotation planes vs depth")
    subtitle(fig, f"{ck.label}   ·   the directions e^{{−V}} grows in")
    ax.legend(loc="best", fontsize=8.5)
    return save_figure(fig, out, "repulsive_depth")


# ---------------------------------------------------------------------------
# S6
# ---------------------------------------------------------------------------

def _henrici_depth(ck: Checkpoint, out: Path) -> Path:
    """
    S6 — Henrici departure from normality vs depth.

    Zero for a normal matrix; otherwise how much the Schur blocks interact,
    i.e. whether the S/A split is informative or decorative. The lower panel
    draws the UNCLAMPED absolute value against zero: the previous version
    clamped at zero silently, and a materially negative value there means the
    block parse disagrees with T — a bug signal, not numerical noise.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    n = len(ck.per_layer)
    x = np.arange(n)

    rel = ck.field("henrici_relative")
    axes[0].plot(x, rel, color=CATEGORICAL[4], marker="o", markersize=3.4,
                 linewidth=2.0)
    if np.isfinite(rel).any():
        k = int(np.nanargmax(rel))
        axes[0].plot([k], [rel[k]], marker="*", markersize=13,
                     color=CATEGORICAL[4], linestyle="none")
        axes[0].annotate(f"max at {ck.layer_names[k]}\n{rel[k]:.4f}",
                         xy=(k, rel[k]), xytext=(6, 8),
                         textcoords="offset points", fontsize=8,
                         color="#4B5563")
    axes[0].set_ylabel("henrici_relative")
    axes[0].set_title("Departure from normality vs depth")

    unc = ck.field("henrici_absolute_unclamped")
    axes[1].plot(x, unc, color="#374151", marker="s", markersize=3.0,
                 linewidth=1.5)
    axes[1].axhline(0.0, **REFERENCE_LINE)
    axes[1].set_ylabel("henrici_absolute\n(unclamped)", fontsize=9)
    depth_axis(axes[1], n)

    subtitle(fig, f"{ck.label}   ·   ‖T‖²_F − Σ|λ|²")
    note(axes[1], "A materially negative unclamped value is a block-parse "
                  "disagreement with T, not noise. The previous version "
                  "clamped it away.")
    return save_figure(fig, out, "henrici_depth")


# ---------------------------------------------------------------------------
# S7
# ---------------------------------------------------------------------------

def _normality_budget(ck: Checkpoint, out: Path) -> Path:
    """
    S7 — ‖T‖²_F split into eigenvalue energy and the Henrici departure.

    The same two numbers S6 draws as a ratio, drawn as a budget: how much of
    the operator's Frobenius mass lives in its eigenvalues, and how much in
    the interaction between Schur blocks. The second is what makes the S/A
    split worth doing at all — if it were zero everywhere, S and A would
    commute and `e^{−(S+A)} = e^{−S}e^{−A}`, which would make the whole
    remaining Block 1b question empty.
    """
    fig, ax = plt.subplots(figsize=(9, 4.6))
    n = len(ck.per_layer)
    x = np.arange(n)

    eig = ck.field("eigenvalue_energy")
    hen = ck.field("henrici_absolute")
    total = eig + hen
    with np.errstate(invalid="ignore", divide="ignore"):
        f_eig = eig / total
        f_hen = hen / total

    ax.bar(x, f_eig, width=0.85, color=CATEGORICAL[0],
           label="eigenvalue energy  Σ|λ|²")
    ax.bar(x, f_hen, width=0.85, bottom=f_eig, color=CATEGORICAL[3],
           label="Henrici departure  (block interaction)")

    depth_axis(ax, n)
    ax.set_ylabel("fraction of ‖T‖²_F")
    ax.set_ylim(0, 1)
    ax.set_title("Where the operator's Frobenius mass sits")
    subtitle(fig, f"{ck.label}   ·   Block 1a, weights only")
    ax.legend(loc="lower right", fontsize=8.5)
    note(ax, "With no interaction term S and A commute and "
             "e^{−(S+A)} = e^{−S}e^{−A} — which would make Block 1b's "
             "remaining question empty.")
    return save_figure(fig, out, "normality_budget")


# ---------------------------------------------------------------------------
# S8
# ---------------------------------------------------------------------------

def _dims_vs_energy(ck: Checkpoint, out: Path) -> Path:
    """
    S8 — dimension fraction against energy fraction, one point per layer.

    `head_circuits`' correction as a figure. Per-head `W_OV` is rank d_head
    (64 of 1024), so a head can rotate 5.5% of the ambient dimensions and
    87.5% of its own core — the DIMENSION fraction is destroyed by rank
    deficiency while the ENERGY fraction is not, because a zero eigenvalue
    contributes to neither numerator nor denominator. The published
    84-97.5% is an energy fraction, so the rank argument does not overturn
    it; the distance of these points from y = x is the size of the
    distinction.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    dim = ck.field("dim_complex_fraction")
    eng = ck.field("complex_energy_fraction")
    norm = depth_norm(len(ck.per_layer))

    for i, (a, b) in enumerate(zip(dim, eng)):
        ax.plot([a], [b], marker="o", markersize=6.5, linestyle="none",
                color=depth_color(i, norm), alpha=0.9)

    lim = [0.0, 1.05]
    ax.plot(lim, lim, **REFERENCE_LINE)
    ax.annotate("y = x", xy=(0.86, 0.9), fontsize=8, color="#6B7280")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("dimension fraction  (2·n_complex / d)")
    ax.set_ylabel("energy fraction  (per-eigenvalue)")
    ax.set_title("How many dimensions rotate vs how much energy rotates")
    subtitle(fig, f"{ck.label}   ·   point colour = depth")
    note(ax, "Rank deficiency destroys the x axis and leaves the y axis "
             "intact — the two are different questions.")
    return save_figure(fig, out, "dims_vs_energy")


# ---------------------------------------------------------------------------
# S9
# ---------------------------------------------------------------------------

def _plane_spectrum(ck: Checkpoint, out: Path) -> Optional[Path]:
    """
    S9 — every rotation plane in the checkpoint, in the complex plane.

    The spectrum itself, not a summary of it. S3 draws θ's mean, sd, median
    and extremes; those five numbers are compatible with a single tight
    cluster, with two clusters at either end, and with a uniform smear, and
    those are three different claims about what the operator does. A mean of
    1.5 rad over a bimodal spectrum at 0.2 and 2.8 describes no plane in the
    layer, and every comparison built on that mean inherits the problem.

    Left panel: the upper half-plane, one point per (layer, plane), radius ρ
    and angle θ, coloured by depth. The unit circle is drawn because ρ = 1 is
    where a rotation neither grows nor shrinks, and the imaginary axis
    because Re λ < 0 — everything left of it — is the direction `e^{−V}`
    grows in, i.e. what S5 counts.

    Right panel: the pooled θ histogram, which is where bimodality shows.
    """
    planes = ck.planes()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    if not planes:
        for ax in axes:
            no_data(ax, "no planes.npz for this checkpoint — the run passed "
                        "--no-planes, or predates the emission")
        return save_figure(fig, out, "plane_spectrum")

    rho = ck.plane_column("rho")
    theta = ck.plane_column("theta")
    depth = ck.plane_layer_index()
    norm = depth_norm(len(ck.per_layer))

    ax = axes[0]
    re, im = rho * np.cos(theta), rho * np.sin(theta)
    ax.scatter(re, im, s=14, alpha=0.75, edgecolors="none",
               c=[depth_color(i, norm) for i in depth])
    r_max = float(np.nanmax(rho)) if rho.size else 1.0
    # The unit circle is drawn only when the spectrum reaches it. On a
    # small-norm OV every plane sits near the origin, and a |λ| = 1 arc
    # would then set the axis limits and squash the actual cloud into a
    # line — a reference that hides the data is worse than none.
    if r_max > 0.35:
        arc = np.linspace(0, np.pi, 200)
        ax.plot(np.cos(arc), np.sin(arc), **REFERENCE_LINE)
        ax.annotate("|λ| = 1", xy=(np.cos(2.4), np.sin(2.4)), fontsize=8,
                    color="#6B7280")
        span = 1.08 * max(r_max, 1.0)
        ax.set_xlim(-span, span)
        ax.set_ylim(-0.04 * span, span)
    else:
        pad = 1.15 * max(float(np.nanmax(np.abs(re))), 1e-9)
        ax.set_xlim(-pad, pad)
        ax.set_ylim(-0.04 * r_max, 1.12 * r_max)
        ax.annotate(f"|λ| ≤ {r_max:.3g} — the unit circle is off-scale",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8,
                    color="#6B7280")
    ax.axvline(0.0, **REFERENCE_LINE)
    ax.set_xlabel("Re λ        ← repulsive | attractive →")
    ax.set_ylabel("Im λ")
    ax.set_title(f"{rho.size} rotation planes, coloured by depth")

    ax = axes[1]
    ax.hist(theta, bins=40, color=CATEGORICAL[0])
    reference_line(ax, 0, "", axis="y")
    ax.axvline(np.pi / 2, **REFERENCE_LINE)
    ax.annotate("π/2  (a Gaussian's mean)", xy=(np.pi / 2, 0.98),
                xycoords=("data", "axes fraction"), rotation=90, va="top",
                ha="left", fontsize=8, color="#6B7280")
    mean = float(np.nanmean(theta)) if theta.size else float("nan")
    ax.axvline(mean, color=CATEGORICAL[1], linewidth=2.0)
    ax.annotate(f"observed mean {mean:.2f}", xy=(mean, 0.62),
                xycoords=("data", "axes fraction"), rotation=90, va="top",
                ha="right", fontsize=8, color=CATEGORICAL[1])
    ax.set_xlim(0, np.pi)
    ax.set_xlabel("θ  (rad)")
    ax.set_ylabel("planes")
    ax.set_title("pooled over every layer")

    fig.tight_layout()
    fig.suptitle("The spectrum, not a summary of it")
    subtitle(fig, f"{ck.label}   ·   from planes.npz")
    note(axes[1], "If the mean line sits in a trough, every statistic built "
                  "on θ_mean describes no plane in this checkpoint.",
         outside=True)
    return save_figure(fig, out, "plane_spectrum")


# ---------------------------------------------------------------------------
# S10
# ---------------------------------------------------------------------------

def _precision_surface(ck: Checkpoint, out: Path) -> Optional[Path]:
    """
    S10 — is "84-97% complex" a fact about OV or about how it was counted?

    Precision-policy item P2. The complex fraction here uses a RELATIVE
    criterion, `|Im λ| > tol·(|Re λ| + eps)`, and a relative criterion is
    exactly what an fp16-epsilon split of a genuinely real eigenvalue pair
    defeats: the split is small in absolute terms and unbounded in ratio when
    `|Re λ|` is also small. The checkpoints went through an fp16 round trip.
    So the honest answer is a surface over (tolerance, perturbation), and
    this draws it — baseline against perturbed, swept over tolerance, with
    the shipped 0.01 marked.

    A flat pair of curves means the headline is a property of the operator. A
    baseline that slopes means it is a property of the counting rule and must
    be quoted with its tolerance. A gap between the curves means it is a
    property of fp16 storage and should be re-derived from float64. The
    verdict strip beneath is `precision_verdict`'s, per layer.
    """
    precision = ck.precision
    # Generous hspace: the two panels have DIFFERENT x axes — tolerance above,
    # depth below — so they must not read as a shared-axis pair.
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.6),
                             gridspec_kw={"height_ratios": [3, 1],
                                          "hspace": 0.62})
    if not precision:
        for ax in axes:
            no_data(ax, "no precision surface — the sweep ran without "
                        "--with-precision (it costs ~10 dense "
                        "eigendecompositions per layer)")
        return save_figure(fig, out, "precision_surface")

    per_layer = precision.get("per_layer") or {}
    names = [n for n in ck.layer_names if n in per_layer] or list(per_layer)
    norm = depth_norm(len(names))

    ax = axes[0]
    tols = None
    for i, name in enumerate(names):
        surface = (per_layer[name] or {}).get("surface") or {}
        tols = surface.get("tols") or tols
        base = surface.get("baseline")
        pert = surface.get("perturbed_mean")
        if not base:
            continue
        ax.plot(surface["tols"], base, color=depth_color(i, norm),
                linewidth=1.4, alpha=0.85)
        if pert:
            ax.plot(surface["tols"], pert, color=depth_color(i, norm),
                    linewidth=1.0, linestyle="--", alpha=0.7)
    if tols:
        ax.set_xscale("log")
        shipped = (per_layer[names[0]] or {}).get(
            "surface", {}).get("shipped_tol")
        if shipped:
            reference_line(ax, float(shipped),
                           f"shipped tol = {shipped:g}", axis="x")
    ax.set_xlabel("relative tolerance for |Im λ| > tol·|Re λ|  (log)")
    ax.set_ylabel("complex energy fraction")
    ax.set_title("Does the headline survive its own counting rule?")
    ax.plot([], [], color="#374151", linewidth=1.4, label="float64 baseline")
    ax.plot([], [], color="#374151", linewidth=1.0, linestyle="--",
            label="after an fp16 round trip")
    ax.legend(loc="best", fontsize=8.5)

    ax = axes[1]
    verdicts = [(per_layer[n] or {}).get("verdict", {}).get("verdict",
                                                            "unknown")
                for n in names]
    colors = {"stable": CATEGORICAL[2],
              "threshold_sensitive": CATEGORICAL[3],
              "precision_sensitive": CATEGORICAL[1],
              "both": "#B45B5B", "unknown": REFUSAL_COLOR}
    for i, v in enumerate(verdicts):
        ax.axvspan(i - 0.5, i + 0.5, color=colors.get(v, REFUSAL_COLOR),
                   linewidth=0)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.grid(False)
    depth_axis(ax, len(names))
    ax.set_ylabel("verdict", rotation=0, ha="right", va="center", fontsize=9)
    seen = [v for v in dict.fromkeys(verdicts)]
    ax.legend(handles=[plt.matplotlib.patches.Patch(
        facecolor=colors.get(v, REFUSAL_COLOR), label=v) for v in seen],
        loc="upper left", bbox_to_anchor=(1.005, 1.2), fontsize=7.5)

    spans = [(per_layer[n] or {}).get("verdict", {}) for n in names]
    worst_tol = max((v.get("tol_span", 0.0) or 0.0) for v in spans) if spans else 0.0
    worst_prec = max((v.get("precision_span", 0.0) or 0.0) for v in spans) if spans else 0.0
    subtitle(fig, f"{ck.label}   ·   overall: "
                  f"{precision.get('overall_verdict', '?')}   ·   worst "
                  f"tolerance span {worst_tol:.2e}, worst precision span "
                  f"{worst_prec:.2e}   ·   item P2")
    note(ax, "This verdict is about Block 1a's descriptive fraction. Block "
             "1b's frames are scored on energies and do not consult this "
             "tolerance.", outside=True)
    return save_figure(fig, out, "precision_surface")
