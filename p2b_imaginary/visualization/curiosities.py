"""
p2b_imaginary/visualization/curiosities.py — the speculative half.

Twelve figures that are not verdict figures. None of them is in a
falsification table, none decides anything, and none should be cited as
evidence for a claim the phase makes elsewhere. They exist because Block 1a
produces a per-layer × per-checkpoint table for free — it reads no
activations and runs no forward pass — and that table is cheap enough to look
at from angles nobody has a hypothesis for yet. Looking is how the next
question gets found, and a figure that shows nothing is a result worth having
drawn once.

Each carries its own "what would be interesting here" line in its docstring.
Four are worth naming up front because they are the ones most likely to
produce a real question:

  X6  ov_frob against the complex fraction, pooled over every layer and
      checkpoint. Study A's OV spectral-norm confound (partial ρ to −0.71) is
      a scatter this phase can draw for nothing, and it is the same quantity
      the rank-gate divergence scales with.
  X7  the layer × layer correlation of trajectories. Whether depth moves as
      one block, in adjacent groups, or independently is three different
      mechanisms and one image.
  X9  each layer's drift in (dimension fraction, energy fraction) from the
      first checkpoint to the last. Training moving layers ALONG y = x is a
      different story from moving them off it.
  X10 measured wall time against `run_2b.estimate_cost`. The estimator is
      calibrated at d = 1024 and its d³ scaling has never been checked
      against a real sweep, which is the first thing anyone planning
      pythia-1.4b needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from p2b_imaginary import p2b_report, run_2b

from .loaders import (
    Checkpoint, Sweep, checkpoint_out, cross_out, depth_matrix, frame_counts,
)
from .style import (
    BLOG_STYLE, CATEGORICAL, ELIM_CMAP, EVENT_SPAN, FRAME_COLORS, FRAME_ORDER,
    REFERENCE_LINE, REFUSAL_COLOR, SEQ_CMAP, add_step_colorbar, depth_color,
    depth_norm, format_step_axis, no_data, note, reference_line, save_figure,
    signed_norm, step_color, step_norm, step_x, subtitle,
)

__all__ = ["generate_curiosity_figures", "FIGURES", "FINGERPRINT_METRICS"]

FIGURES = ("spectrum_fingerprint", "rotation_clock", "spectral_annulus",
           "training_ribbon", "convention_divergence", "norm_vs_spectrum",
           "depth_coupling", "theta_coherence", "spectral_drift_arrows",
           "sweep_cost", "violation_depth_density", "rho_theta_joint")

#: The columns X1 puts in its fingerprint. Every one is a per-layer scalar
#: Block 1a already writes; the point of the figure is the whole table at
#: once, so the list is broad rather than curated.
FINGERPRINT_METRICS = (
    "complex_energy_fraction", "dim_complex_fraction", "theta_mean",
    "theta_std", "rho_mean", "rho_std", "frac_repulsive_real_part",
    "frac_rho_above_one", "henrici_relative", "ov_frob",
)


def generate_curiosity_figures(sweep: Sweep, out_dir: Path) -> List[Path]:
    """Every `curiosities` figure — per checkpoint and across the sweep."""
    paths: List[Path] = []
    cross = cross_out(out_dir)

    with plt.rc_context(BLOG_STYLE):
        for ck in sweep.with_1a:
            d = checkpoint_out(out_dir, ck)
            paths.append(_spectrum_fingerprint(ck, d))
            paths.append(_rotation_clock(ck, d))
            paths.append(_spectral_annulus(ck, sweep, d))

        if sweep.with_1a:
            paths.append(_norm_vs_spectrum(sweep, cross))
            paths.append(_rho_theta_joint(sweep, cross))
            paths.append(_convention_divergence(sweep, cross))

        if sweep.has_trajectory:
            paths.append(_training_ribbon(sweep, cross))
            paths.append(_depth_coupling(sweep, cross))
            paths.append(_theta_coherence(sweep, cross))
            paths.append(_spectral_drift_arrows(sweep, cross))
        else:
            print("  curiosities: skipping the four cross-checkpoint figures "
                  f"— {len(sweep.stepped)} stepped checkpoint(s), need 2")

        paths.append(_sweep_cost(sweep, cross))
        vdd = _violation_depth_density(sweep, cross)
        if vdd is not None:
            paths.append(vdd)
    return paths


# ---------------------------------------------------------------------------
# X1
# ---------------------------------------------------------------------------

def _spectrum_fingerprint(ck: Checkpoint, out: Path) -> Path:
    """
    X1 — one checkpoint's whole Block 1a table as a single image.

    Every metric z-scored down its OWN column, so the image shows which
    layers are unusual for each quantity rather than which quantities are
    large. Interesting when a layer lights up across several unrelated
    columns at once: that is a layer worth looking at directly, and it is
    hard to notice in ten separate depth profiles.
    """
    keys = [k for k in FINGERPRINT_METRICS
            if np.isfinite(ck.field(k)).any()]
    fig, ax = plt.subplots(figsize=(max(8, 0.36 * len(ck.per_layer) + 3),
                                    0.42 * len(keys) + 2.6))
    if not keys:
        no_data(ax, "no per-layer scalars in this checkpoint")
        return save_figure(fig, out, "spectrum_fingerprint")

    grid = np.full((len(keys), len(ck.per_layer)), np.nan)
    for i, key in enumerate(keys):
        col = ck.field(key)
        mu, sd = np.nanmean(col), np.nanstd(col)
        grid[i] = (col - mu) / sd if sd > 1e-12 else 0.0

    im = ax.imshow(grid, aspect="auto", cmap=ELIM_CMAP,
                   norm=signed_norm(grid[np.isfinite(grid)]))
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=8)
    ax.set_xticks(range(0, len(ck.per_layer),
                        1 if len(ck.per_layer) <= 16 else 2))
    ax.set_xlabel("OV layer")
    ax.grid(False)
    ax.set_title("One checkpoint's per-layer table, each metric z-scored")
    fig.colorbar(im, ax=ax, pad=0.02, label="z within the column")
    subtitle(fig, f"{ck.label}   ·   a layer lit across several unrelated "
                  "columns is worth looking at directly")
    return save_figure(fig, out, "spectrum_fingerprint")


# ---------------------------------------------------------------------------
# X2
# ---------------------------------------------------------------------------

def _rotation_clock(ck: Checkpoint, out: Path) -> Path:
    """
    X2 — each layer's mean rotation, on polar axes.

    Angle is θ, radius is ρ, colour is depth: "how fast does this layer
    rotate, and how hard". Interesting when the layers fan out with depth
    rather than clustering — a coherent angle shared across depth would be a
    strong structural claim, and the figure makes the absence of one obvious.
    """
    theta = ck.field("theta_mean")
    rho = ck.field("rho_mean")
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(111, projection="polar")

    norm = depth_norm(len(ck.per_layer))
    for i, (t, r) in enumerate(zip(theta, rho)):
        if not (np.isfinite(t) and np.isfinite(r)):
            continue
        ax.plot([t], [r], marker="o", markersize=7, linestyle="none",
                color=depth_color(i, norm), alpha=0.9)
    finite = np.isfinite(theta) & np.isfinite(rho)
    if finite.any():
        ax.plot(theta[finite], rho[finite], color="#9AA0A6", linewidth=0.9,
                alpha=0.6, zorder=0)

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    if finite.any():
        # Without this the radial axis auto-ranges to a round number well
        # above the data and every layer sits in a knot at the centre — the
        # spread BETWEEN layers is the whole figure, so the axis has to be
        # scaled to it.
        ax.set_rmax(float(np.nanmax(rho[finite])) * 1.15)
    ax.set_xlabel("θ  (rad, 0 … π)")
    ax.set_title("Rotation clock — angle θ, radius ρ, colour depth", pad=18)
    subtitle(fig, f"{ck.label}   ·   the gray path connects layers in depth "
                  "order")
    return save_figure(fig, out, "rotation_clock")


# ---------------------------------------------------------------------------
# X3
# ---------------------------------------------------------------------------

def _spectral_annulus(ck: Checkpoint, sweep: Sweep, out: Path) -> Path:
    """
    X3 — where each layer's eigenvalues live in the complex plane.

    A SUMMARY, not the spectrum. The per-plane (ρ, θ) lists are data gap G3 —
    `top_rotation_planes` computes them and `summary_to_json` drops them with
    the plane bases — so each layer is drawn as the sector its summary
    statistics imply: radius ρ_mean ± sd, angle θ_min … θ_max. If G3 ever
    closes, this figure should be replaced by the real scatter and not
    extended.

    Interesting when the sectors are narrow and separated by depth: that
    would mean each layer has its own characteristic rotation, which the
    summary statistics hint at and cannot establish.
    """
    theta_lo, theta_hi = ck.field("theta_min"), ck.field("theta_max")
    rho, rho_sd = ck.field("rho_mean"), ck.field("rho_std")
    fig = plt.figure(figsize=(6.8, 6.4))
    ax = fig.add_subplot(111, projection="polar")

    norm = depth_norm(len(ck.per_layer))
    for i in range(len(ck.per_layer)):
        if not np.isfinite(theta_lo[i]) or not np.isfinite(rho[i]):
            continue
        arc = np.linspace(theta_lo[i], theta_hi[i], 40)
        inner = max(rho[i] - (rho_sd[i] if np.isfinite(rho_sd[i]) else 0), 0)
        outer = rho[i] + (rho_sd[i] if np.isfinite(rho_sd[i]) else 0)
        ax.fill_between(arc, inner, outer, color=depth_color(i, norm),
                        alpha=0.28, linewidth=0)
        ax.plot(arc, np.full_like(arc, rho[i]), color=depth_color(i, norm),
                linewidth=1.4)

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_title("Where each layer's eigenvalues sit — a summary, not the "
                 "spectrum", pad=18, fontsize=11)
    subtitle(fig, f"{ck.label}   ·   sector = θ_min…θ_max × (ρ_mean ± sd)")
    note(ax, "The real per-plane (ρ, θ) list is data gap G3 — computed by "
             "top_rotation_planes and dropped at serialization.")
    return save_figure(fig, out, "spectral_annulus")


# ---------------------------------------------------------------------------
# X4
# ---------------------------------------------------------------------------

def _training_ribbon(sweep: Sweep, out: Path) -> Path:
    """
    X4 — the whole phase as one image: layer × step, with the dated events.

    Everything else in the `trajectory` class is a projection of this. The
    thing to look for is a stripe that is vertical (a training event hitting
    every layer at once) against one that is horizontal (a layer that is
    simply different at every checkpoint) — those are opposite mechanisms and
    a mean over either axis destroys the distinction.
    """
    steps, mat = depth_matrix(sweep.checkpoints, "complex_energy_fraction")
    fig, ax = plt.subplots(figsize=(11, 5.0))
    if not steps:
        no_data(ax, "no per-layer complex fraction in this sweep")
        return save_figure(fig, out, "training_ribbon")

    x = step_x(steps)
    # pcolormesh on the real log-spaced axis rather than imshow on an ordinal
    # one, so interval WIDTHS are honest — 8→16 and 40000→60000 are not the
    # same interval and T3's ordinal heatmap draws them as if they were.
    edges_x = np.concatenate([[x[0] - 0.25],
                              (x[1:] + x[:-1]) / 2.0,
                              [x[-1] + 0.25]])
    edges_y = np.arange(mat.shape[1] + 1) - 0.5
    mesh = ax.pcolormesh(edges_x, edges_y, mat.T, cmap=SEQ_CMAP,
                         shading="flat")
    for ev in p2b_report.KNOWN_TRANSITIONS:
        lo, hi = ev["span"]
        ax.axvspan(step_x([lo])[0], step_x([hi])[0], **EVENT_SPAN)

    format_step_axis(ax, steps)
    ax.set_ylabel("OV layer")
    ax.grid(False)
    ax.set_title("The whole sweep as one image")
    fig.colorbar(mesh, ax=ax, pad=0.02, label="complex energy fraction")
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   shaded spans are the dated "
                  "events   ·   x is log10(step+1), so interval widths are "
                  "honest")
    note(ax, "Vertical stripe = a training event across depth. Horizontal = "
             "a layer that is simply different. A mean destroys the "
             "difference.", outside=True)
    return save_figure(fig, out, "training_ribbon")


# ---------------------------------------------------------------------------
# X5
# ---------------------------------------------------------------------------

def _convention_divergence(sweep: Sweep, out: Path) -> Path:
    """
    X5 — the two energy conventions against each other, and their ratio.

    `complex_energy_fraction` counts 2ρ² per 2×2 block; the legacy
    `rotational_fraction_per_block` counts ρ² once, mixing a per-pair total
    with a per-eigenvalue one and understating rotational energy by roughly a
    factor of two. Both shipped under the name "rotational fraction". The
    published 84-97.5% is the second.

    Interesting for a reason beyond bookkeeping: the RATIO is not a constant
    2, because it depends on how much of the energy is in 1×1 blocks at all.
    Where the ratio deviates most is where the real spectrum matters most.
    """
    xs, ys, cs = [], [], []
    for ck in sweep.with_1a:
        new = ck.field("complex_energy_fraction")
        old = ck.field("complex_energy_fraction_legacy_per_block")
        m = np.isfinite(new) & np.isfinite(old)
        xs.extend(old[m])
        ys.extend(new[m])
        cs.extend([ck.step if ck.step is not None else 0] * int(m.sum()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if not xs:
        for ax in axes:
            no_data(ax, "no per-layer fractions in this sweep")
        return save_figure(fig, out, "convention_divergence")

    xs, ys, cs = np.array(xs), np.array(ys), np.array(cs)
    ax = axes[0]
    norm = step_norm(sorted(set(int(c) for c in cs)))
    ax.scatter(xs, ys, s=22, c=[step_color(int(c), norm) for c in cs],
               alpha=0.85, edgecolors="none")
    lim = [0, 1.05]
    ax.plot(lim, lim, **REFERENCE_LINE)
    ax.annotate("y = x", xy=(0.86, 0.9), fontsize=8, color="#6B7280")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("legacy per-block  (the published number)")
    ax.set_ylabel("per-eigenvalue  (the corrected number)")
    ax.set_title("two conventions, one name")

    ax = axes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ys / xs
    ratio = ratio[np.isfinite(ratio)]
    ax.hist(ratio, bins=30, color=CATEGORICAL[0])
    reference_line(ax, 0, "", axis="y")
    ax.axvline(2.0, color="#B45B5B", linewidth=2.0)
    ax.annotate("×2 — what a pure-rotation spectrum would give",
                xy=(2.0, ax.get_ylim()[1] * 0.95), fontsize=8,
                color="#B45B5B", rotation=90, va="top")
    ax.set_xlabel("corrected ÷ legacy")
    ax.set_ylabel("layers")
    ax.set_title("how far apart, and where")

    fig.tight_layout()
    fig.suptitle("The factor between two numbers that shipped under one name",
                 y=1.02)
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   point colour is training "
                  "step   ·   pooled over every layer of every checkpoint")
    return save_figure(fig, out, "convention_divergence")


# ---------------------------------------------------------------------------
# X6
# ---------------------------------------------------------------------------

def _norm_vs_spectrum(sweep: Sweep, out: Path) -> Path:
    """
    X6 — ‖OV‖_F against the complex fraction, pooled over the whole sweep.

    Study A's OV spectral-norm confound (partial ρ to −0.71, status-2 blocker
    2) is a scatter this phase can draw for free, and it is the same quantity
    the rank-gate divergence scales with — so a sweep whose Block 1b
    comparisons all refuse with `different_transitions_scored` should show
    its cause here.

    Interesting when the cloud has structure in the STEP colour rather than
    in position: that would mean the norm and the fraction move together over
    training while being unrelated across depth, which is two different
    relationships wearing one scatter.
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    xs, ys, cs = [], [], []
    for ck in sweep.with_1a:
        f, n = ck.field("complex_energy_fraction"), ck.field("ov_frob")
        m = np.isfinite(f) & np.isfinite(n)
        xs.extend(n[m])
        ys.extend(f[m])
        cs.extend([ck.step if ck.step is not None else 0] * int(m.sum()))
    if not xs:
        no_data(ax, "no per-layer data in this sweep")
        return save_figure(fig, out, "norm_vs_spectrum")

    steps = sorted(set(int(c) for c in cs))
    norm = step_norm(steps)
    ax.scatter(xs, ys, s=26, c=[step_color(int(c), norm) for c in cs],
               alpha=0.85, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("‖OV‖_F  (log)")
    ax.set_ylabel("complex energy fraction")
    ax.set_title("Is the spectrum a function of the norm?")
    if len(steps) > 1:
        add_step_colorbar(fig, ax, steps, norm)
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   one point per (layer, "
                  "checkpoint)   ·   Study A's spectral-norm confound")
    note(ax, "The rank-gate divergence that refuses Block 1b comparisons "
             "scales with this x axis.")
    return save_figure(fig, out, "norm_vs_spectrum")


# ---------------------------------------------------------------------------
# X7
# ---------------------------------------------------------------------------

def _depth_coupling(sweep: Sweep, out: Path) -> Path:
    """
    X7 — layer × layer correlation of trajectories across checkpoints.

    Three readings, three mechanisms. A uniformly high matrix means the whole
    model moves as one and any single layer's trajectory is the model's. A
    block-diagonal one means depth-adjacent groups move together — the
    interesting case, and the one Phase 1's layers-21-23 event would produce.
    A featureless one means the layers are independent and the across-layer
    mean is averaging over unrelated things.
    """
    steps, mat = depth_matrix(sweep.checkpoints, "complex_energy_fraction")
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    if len(steps) < 3:
        no_data(ax, f"{len(steps)} checkpoints — a correlation across "
                    "checkpoints needs at least 3")
        return save_figure(fig, out, "depth_coupling")

    # Columns with no variation produce a divide-by-zero and a NaN row, which
    # is the correct answer for a layer that never moves — masked rather than
    # filled, so "constant" does not read as "uncorrelated".
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(np.nan_to_num(mat.T, nan=0.0))
    im = ax.imshow(corr, cmap=ELIM_CMAP, vmin=-1, vmax=1, origin="lower")
    ax.set_xlabel("OV layer")
    ax.set_ylabel("OV layer")
    ax.grid(False)
    ax.set_title("Do layers move together?")
    fig.colorbar(im, ax=ax, pad=0.02, label="correlation across checkpoints")
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   complex energy fraction "
                  f"over {len(steps)} checkpoints")
    note(ax, "Uniform = the model moves as one. Block-diagonal = depth "
             "groups. Featureless = the across-layer mean averages unrelated "
             "things.", outside=True)
    return save_figure(fig, out, "depth_coupling")


# ---------------------------------------------------------------------------
# X8
# ---------------------------------------------------------------------------

def _theta_coherence(sweep: Sweep, out: Path) -> Path:
    """
    X8 — θ_std / θ_mean per layer per step.

    Where in depth × training the rotation is COHERENT (one angle shared by
    the planes) versus SCATTERED (a bag of unrelated planes that happens to
    average somewhere). The mean angle alone cannot distinguish these, and
    they are different claims about what the operator does.
    """
    steps, mean = depth_matrix(sweep.checkpoints, "theta_mean")
    _, sd = depth_matrix(sweep.checkpoints, "theta_std")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    if not steps:
        no_data(ax, "no theta statistics in this sweep")
        return save_figure(fig, out, "theta_coherence")

    with np.errstate(divide="ignore", invalid="ignore"):
        cv = sd / mean
    im = ax.imshow(cv.T, aspect="auto", origin="lower", cmap=SEQ_CMAP,
                   extent=(-0.5, len(steps) - 0.5, -0.5, cv.shape[1] - 0.5))
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right",
                       fontsize=8)
    ax.set_xlabel("training step  (ordinal)")
    ax.set_ylabel("OV layer")
    ax.grid(False)
    ax.set_title("Coherent rotation or a bag of planes?")
    fig.colorbar(im, ax=ax, pad=0.02, label="θ_std / θ_mean")
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   low = one shared angle, "
                  "high = unrelated planes averaging somewhere")
    return save_figure(fig, out, "theta_coherence")


# ---------------------------------------------------------------------------
# X9
# ---------------------------------------------------------------------------

def _spectral_drift_arrows(sweep: Sweep, out: Path) -> Path:
    """
    X9 — each layer's drift from the first checkpoint to the last.

    In (dimension fraction, energy fraction) coordinates, so the arrow's
    direction relative to y = x is the content: motion ALONG the diagonal is
    the whole spectrum becoming more or less rotational, motion OFF it is the
    rank structure changing without the energy following, or vice versa —
    which is `head_circuits`' distinction, on the training axis.
    """
    stepped = sweep.stepped
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    if len(stepped) < 2:
        no_data(ax, "need two stepped checkpoints for a drift")
        return save_figure(fig, out, "spectral_drift_arrows")

    first, last = stepped[0], stepped[-1]
    x0, y0 = first.field("dim_complex_fraction"), first.field("complex_energy_fraction")
    x1, y1 = last.field("dim_complex_fraction"), last.field("complex_energy_fraction")
    n = min(x0.size, x1.size)
    norm = depth_norm(n)

    for i in range(n):
        if not all(np.isfinite([x0[i], y0[i], x1[i], y1[i]])):
            continue
        ax.annotate("", xy=(x1[i], y1[i]), xytext=(x0[i], y0[i]),
                    arrowprops=dict(arrowstyle="->", color=depth_color(i, norm),
                                    linewidth=1.4, alpha=0.85))
        ax.plot([x0[i]], [y0[i]], marker="o", markersize=3.5,
                color=depth_color(i, norm), linestyle="none")

    lim = [0, 1.05]
    ax.plot(lim, lim, **REFERENCE_LINE)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("dimension fraction")
    ax.set_ylabel("energy fraction")
    ax.set_title(f"Where training moved each layer\nstep {first.step} → "
                 f"step {last.step}")
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   arrow colour is depth   ·   "
                  "tail = first checkpoint, head = last")
    note(ax, "Along the diagonal: the spectrum became more rotational. Off "
             "it: rank structure moved without energy following.")
    return save_figure(fig, out, "spectral_drift_arrows")


# ---------------------------------------------------------------------------
# X10
# ---------------------------------------------------------------------------

def _sweep_cost(sweep: Sweep, out: Path) -> Path:
    """
    X10 — measured wall time against `run_2b.estimate_cost`'s prediction.

    The estimator is calibrated against a 1024×1024 float64 factorisation at
    roughly one second and assumes d³ scaling, and nothing has ever checked
    it against a real sweep. Anyone planning pythia-1.4b (d = 2048, so 8×
    per factorisation) is relying on that constant, which makes this the
    cheapest useful figure in the class.

    The prediction is drawn as a single horizontal line rather than per
    checkpoint, because `estimate_cost` is a function of the SWEEP shape —
    checkpoints, layers, d, blocks — and gives one per-checkpoint number for
    all of them. A measured series that slopes is therefore already
    interesting.
    """
    cks = [c for c in sweep.checkpoints
           if c.wall_time_seconds is not None and c.step is not None]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    if not cks:
        no_data(ax, "no wall_time_seconds recorded in this sweep")
        return save_figure(fig, out, "sweep_cost")

    cks.sort(key=lambda c: c.step)
    steps = [c.step for c in cks]
    x = step_x(steps)
    ax.plot(x, [c.wall_time_seconds for c in cks], color=CATEGORICAL[0],
            marker="o", markersize=4.5, linewidth=2.0, label="measured")

    n_layers = max((c.n_ov_layers for c in cks), default=0)
    d_model = _d_model(cks)
    predicted = None
    if n_layers and d_model:
        est = run_2b.estimate_cost(len(cks), len(sweep.prompts) or 1,
                                   n_layers, d_model, sweep.blocks or ["1a"])
        predicted = est["estimated_seconds"] / max(len(cks), 1)
        ax.axhline(predicted, color="#B45B5B", linewidth=2.0, linestyle="--",
                   label="run_2b.estimate_cost ÷ n_checkpoints")

    format_step_axis(ax, steps)
    ax.set_ylabel("seconds per checkpoint")
    ax.set_title("What the sweep cost, against what it was predicted to cost")
    ax.legend(loc="best", fontsize=8.5)
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   d = {d_model or '?'}   ·   "
                  f"{n_layers} OV layers   ·   "
                  f"blocks {', '.join(sweep.blocks) or '?'}")
    if sweep.combined.get("fixture"):
        note(ax, "This sweep is the synthetic fixture — the measured times "
                 "are invented and the comparison means nothing.")
    else:
        note(ax, "estimate_cost assumes d³ scaling calibrated at d = 1024. "
                 "pythia-1.4b is 8× per factorisation on that assumption.")
    return save_figure(fig, out, "sweep_cost")


def _d_model(cks) -> Optional[int]:
    """`d` from any Block 1a per-layer record — the phase writes it there."""
    for c in cks:
        for rec in c.per_layer:
            if rec.get("d"):
                return int(rec["d"])
    return None


# ---------------------------------------------------------------------------
# X11
# ---------------------------------------------------------------------------

def _violation_depth_density(sweep: Sweep, out: Path) -> Optional[Path]:
    """
    X11 — where in depth violations land, per frame, pooled over the sweep.

    Phase 2's open item 5 says attribution reorganises while the count stays
    flat: the violations move to a different subspace without changing in
    number. This is the depth-side version of that claim and the histogram it
    implies — if the count is flat and the DEPTH distribution is not, the
    reorganisation is visible without any subspace machinery at all.

    Interesting when the original and rescaled frames put their violations at
    different depths while agreeing on the count: that is a rescaling doing
    something the elimination rate is blind to.
    """
    frames_seen: Dict[str, List[int]] = {}
    n_records = 0
    for ck in sweep.with_1b:
        for js in ck.block1b_scored().values():
            n_records += 1
            for key in (js.get("frames") or {}):
                layers = frame_counts(js, key).get("violation_layers") or []
                frames_seen.setdefault(key, []).extend(int(L) for L in layers)

    if not n_records:
        return None

    fig, ax = plt.subplots(figsize=(10, 4.6))
    keys = [k for k in FRAME_ORDER if frames_seen.get(k)]
    if not keys:
        no_data(ax, f"{n_records} Block 1b record(s), and no violations in "
                    "any frame — nothing to place in depth")
        return save_figure(fig, out, "violation_depth_density")

    max_layer = max(max(v) for v in frames_seen.values() if v)
    bins = np.arange(-0.5, max_layer + 1.5, 1.0)
    for key in keys:
        ax.hist(frames_seen[key], bins=bins, histtype="step", linewidth=2.0,
                color=FRAME_COLORS.get(key, REFUSAL_COLOR), label=key)

    ax.set_xlabel("transition (layer L)")
    ax.set_ylabel("violations, pooled over the sweep")
    ax.set_title("Where violations sit in depth, per frame")
    ax.legend(loc="best", fontsize=8.5)
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   pooled over {n_records} "
                  "(checkpoint, prompt) records")
    note(ax, "Frames agreeing on the COUNT while disagreeing on the DEPTH is "
             "a rescaling the elimination rate is blind to.")
    return save_figure(fig, out, "violation_depth_density")


# ---------------------------------------------------------------------------
# X12
# ---------------------------------------------------------------------------

def _rho_theta_joint(sweep: Sweep, out: Path) -> Path:
    """
    X12 — every (layer, checkpoint) as a point in (θ_mean, ρ_mean).

    Where the operator's rotation lives, as a cloud that training moves
    through. Interesting when the cloud has a spine — a one-dimensional
    relationship between angle and modulus would mean the two are not free
    parameters, which is a structural claim about OV that no per-layer
    profile makes visible.
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    xs, ys, cs = [], [], []
    for ck in sweep.with_1a:
        t, r = ck.field("theta_mean"), ck.field("rho_mean")
        m = np.isfinite(t) & np.isfinite(r)
        xs.extend(t[m])
        ys.extend(r[m])
        cs.extend([ck.step if ck.step is not None else 0] * int(m.sum()))
    if not xs:
        no_data(ax, "no angle/modulus statistics in this sweep")
        return save_figure(fig, out, "rho_theta_joint")

    steps = sorted(set(int(c) for c in cs))
    norm = step_norm(steps)
    ax.scatter(xs, ys, s=26, c=[step_color(int(c), norm) for c in cs],
               alpha=0.85, edgecolors="none")
    reference_line(ax, 1.0, "ρ = 1")
    ax.axvline(np.pi / 2, **REFERENCE_LINE)
    ax.annotate("π/2", xy=(np.pi / 2, 0.98), xycoords=("data", "axes fraction"),
                fontsize=8, color="#6B7280", va="top")
    ax.set_xlabel("mean θ  (rad)")
    ax.set_ylabel("mean ρ")
    ax.set_title("Where the operator's rotation lives")
    if len(steps) > 1:
        add_step_colorbar(fig, ax, steps, norm)
    subtitle(fig, f"{sweep.base or 'sweep'}   ·   one point per (layer, "
                  "checkpoint)")
    note(ax, "A spine in this cloud would mean θ and ρ are not free "
             "parameters — a structural claim no depth profile shows.")
    return save_figure(fig, out, "rho_theta_joint")
