"""
p1c_frames/visualization/frames_fig.py — sub-experiment D, the four frames.

Five per-run figures (D1-D5 in FIGURES-1c.md). **Nothing on disk carries a
`D` block today**: `frame_table.py` is implemented and validated, and
`run_1c --subexp` accepts `D`, but `run_one` has no `D` branch (gap G1). So
against a real directory every figure here skips with that reason printed,
and the fixture is what exercises them.

That is deliberate rather than premature. The block's shape is decided by
`frame_table()`'s return value, which exists and is tested; writing the
figures against it now means the driver branch has a target, and if it lands
in a different shape these break immediately instead of silently drawing the
wrong frame. `_fixture._fake_frame_block` is the written-down version of
that contract.

The organizing constraint is status-1c's D findings:

  9.  The dispersion statistic matters more than the mean. A γ of all 0.44
      and a γ of all 4.4 both leave the manifold a sphere — a uniform
      rescaling changes nothing. D3 plots the coefficient of variation
      against ALBERT's 0.018, with the condition number beside it because
      that is what bounds the metric distortion.
  10. "Constant across layers" is a second, separate condition. A model
      uniform within each layer and different between them is on a
      DIFFERENT sphere at each depth, so every cross-layer trajectory metric
      — which is all of Phase 1 — inherits a rescaling. D3 prints
      `cross_layer_mean_cv` for exactly this.
  11. Symmetric KL is not a metric, so the Torgerson Gram is not guaranteed
      PSD. D5 draws the negative-eigenvalue mass rather than clipping it: a
      frame whose Gram carries substantial negative mass is not one in which
      "effective rank" means what it means elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .loaders import Run, record_field, records
from .style import (
    BLOG_STYLE, CATEGORICAL, FRAME_COLORS, FRAME_ORDER, INVALID_COLOR,
    caption, depth_axis, no_data, reference_line, save_figure, verdict_box,
)

__all__ = ["generate_frame_figures"]

#: ALBERT XLarge v2's LN gain dispersion, from the paper's §2.2 and restated
#: in `frame_table.ALBERT_GAMMA_*`. Imported rather than typed, so a change
#: there moves this line.
try:
    from p1c_frames.frame_table import ALBERT_GAMMA_MEAN, ALBERT_GAMMA_SD
    ALBERT_CV = ALBERT_GAMMA_SD / ALBERT_GAMMA_MEAN
except Exception:                                    # pragma: no cover
    ALBERT_CV = 0.008 / 0.44

FRAME_LABELS = {
    "l2": "l2 (the sphere Phase 1 measures on)",
    "ln_plain": "plain LayerNorm (exact sphere projection)",
    "ln_learned": "learned LN (γ, β as trained)",
    "functional": "functional (symmetrized-KL, Torgerson)",
}


def _skip(name: str, reason: str) -> None:
    print(f"    skip {name}: {reason}")


def _frame_series(per_layer: List[dict], frame: str, key: str) -> np.ndarray:
    """One key out of one frame's sub-dict, per layer."""
    out = np.full(len(per_layer), np.nan)
    for i, entry in enumerate(per_layer):
        f = entry.get(frame)
        if isinstance(f, dict):
            v = f.get(key)
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def generate_frame_figures(run: Run, out_dir: Path) -> List[Path]:
    """D1-D5 for one run."""
    if not run.has("D"):
        _skip("frames", next((m for m in run.missing if m.startswith("D ")),
                             "no D block — run_1c has no D branch (G1)"))
        return []

    per_layer = records(run, "D")
    if not per_layer:
        _skip("frames", "D block carries no per-layer records")
        return []

    paths: List[Optional[Path]] = [
        _d1_frame_ip_mean(run, per_layer, out_dir),
        _d2_frame_rank(run, per_layer, out_dir),
    ]
    if any(isinstance(e.get("gamma_stats"), dict) for e in per_layer):
        paths.append(_d3_sphere_license(run, per_layer, out_dir))
    else:
        _skip("sphere_license", "no gamma_stats — LN gains were not read")
    if any(isinstance(e.get("bias_floor"), dict) for e in per_layer):
        paths.append(_d4_bias_energy_floor(run, per_layer, out_dir))
    else:
        _skip("bias_energy_floor", "no bias_floor — LN bias was not read")
    paths.append(_d5_neg_eigen_mass(run, per_layer, out_dir))
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# D1 — footnote or confound?
# ---------------------------------------------------------------------------

def _d1_frame_ip_mean(run: Run, per_layer: List[dict], out_dir: Path) -> Path:
    """
    ip_mean in every available frame, with the per-layer spread beneath.

    The spread is the decision: if it is 0.02 the frame choice is a
    footnote; if it is 0.4 then every Phase 1 statement about clustering is
    a statement about the l2 frame specifically and has to be labelled as
    one (`frame_disagreement`).
    """
    with plt.rc_context(BLOG_STYLE):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9.2, 6.2), sharex=True,
            gridspec_kw=dict(height_ratios=[1.7, 1], hspace=0.12))
        stack = []
        for frame in FRAME_ORDER:
            v = _frame_series(per_layer, frame, "ip_mean")
            if not np.isfinite(v).any():
                continue
            stack.append(v)
            ax.plot(np.arange(v.size), v, color=FRAME_COLORS[frame],
                    lw=2.2, label=FRAME_LABELS[frame])
        ax.set_ylabel("mean pairwise inner product")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.set_title(f"D1 · does the frame move the answer? — {run.label}")

        if stack:
            arr = np.vstack(stack)
            spread = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
            ax2.fill_between(np.arange(spread.size), 0, spread,
                             color=CATEGORICAL[4], alpha=0.25, linewidth=0)
            ax2.plot(np.arange(spread.size), spread, color=CATEGORICAL[4],
                     lw=2.0)
            reference_line(ax2, 0.02, "footnote territory", side="left")
            ax2.set_ylabel("spread across\nframes")
        depth_axis(ax2, run.n_layers)
        caption(fig, "Reported side by side rather than reduced to a "
                     "preferred frame: which frame is correct is the question "
                     "D exists to answer, and a figure that picked one would "
                     "be assuming its conclusion.")
    return save_figure(fig, out_dir, "frame_ip_mean")


# ---------------------------------------------------------------------------
# D2 — the rank the frame decides
# ---------------------------------------------------------------------------

def _d2_frame_rank(run: Run, per_layer: List[dict], out_dir: Path) -> Path:
    """
    pr_rank per frame vs depth, with the un-framed raw effective rank.

    status-1c's D finding: on a cloud with three sink tokens `pr_rank` reads
    144.7 in the l2 frame and 70.7 in the learned-LN frame, while the raw
    effective rank on the same cloud is 4.99. Those three numbers describe
    one cloud, and putting them on one axis is the only honest way to
    present a "rank collapse".
    """
    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.2, 5.0))
        for frame in FRAME_ORDER:
            v = _frame_series(per_layer, frame, "pr_rank")
            if not np.isfinite(v).any():
                continue
            ax.plot(np.arange(v.size), v, color=FRAME_COLORS[frame], lw=2.2,
                    label=FRAME_LABELS[frame])
        raw = _frame_series(per_layer, "l2", "raw_effective_rank")
        if np.isfinite(raw).any():
            ax.plot(np.arange(raw.size), raw, color="#111827", lw=2.0, ls=":",
                    label="raw effective rank (no frame at all)")
        ax.set_yscale("log")
        reference_line(ax, 2.0, "degenerate floor", side="left")
        depth_axis(ax, run.n_layers)
        ax.set_ylabel("participation-ratio rank")
        ax.set_title(f"D2 · one cloud, four ranks — {run.label}")
        ax.legend(loc="best", fontsize=8.5)
        caption(fig, "Frame choice moves reported quantities materially — "
                     "144.7 (l2) against 70.7 (learned LN) on the same "
                     "measured cloud, with a raw rank of 4.99.")
    return save_figure(fig, out_dir, "frame_rank")


# ---------------------------------------------------------------------------
# D3 — is the sphere licensed?
# ---------------------------------------------------------------------------

def _d3_sphere_license(run: Run, per_layer: List[dict], out_dir: Path) -> Path:
    """
    Per-layer γ coefficient of variation against ALBERT's, condition number
    beside it, and `cross_layer_mean_cv` printed.

    Three quantities because the paper's one-line justification hides three
    conditions: dispersion within a layer (the cv), the distortion that
    dispersion can cause (the condition number, which is what actually
    bounds it), and constancy across layers (finding 10 — a model on a
    different sphere at each depth still breaks every cross-layer metric).
    """
    stats = [e.get("gamma_stats") if isinstance(e.get("gamma_stats"), dict)
             else {} for e in per_layer]
    cv = record_field(stats, "cv")
    cond = record_field(stats, "condition_number")
    lic = run.block("D.sphere_license")

    with plt.rc_context(BLOG_STYLE):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9.2, 6.2), sharex=True,
            gridspec_kw=dict(height_ratios=[1.4, 1], hspace=0.12))
        ax.plot(np.arange(cv.size), cv, color="#12406F", lw=2.4)
        ax.axhline(ALBERT_CV, color=CATEGORICAL[2], ls="--", lw=1.8)
        ax.annotate(f"ALBERT: cv = {ALBERT_CV:.3f} — what \"essentially "
                    f"constant\" means", xy=(0.01, ALBERT_CV),
                    xycoords=("axes fraction", "data"), fontsize=8,
                    va="bottom", color=CATEGORICAL[2])
        ax.axhline(2 * ALBERT_CV, color=INVALID_COLOR, ls=":", lw=1.4)
        ax.annotate("2× ALBERT — the licensing cut", xy=(0.99, 2 * ALBERT_CV),
                    xycoords=("axes fraction", "data"), fontsize=8, ha="right",
                    va="bottom", color="#6B7280")
        ax.set_ylabel("γ coefficient\nof variation")
        ax.set_title(f"D3 · is the sphere licensed here? — {run.label}")
        verdict_box(ax, lic.get("verdict", ""), loc="upper right")

        ax2.plot(np.arange(cond.size), cond, color=CATEGORICAL[1], lw=2.2)
        reference_line(ax2, 1.0, "condition 1 — a true sphere", side="left")
        ax2.set_ylabel("γ condition\nnumber (max/min)")
        depth_axis(ax2, run.n_layers)

        x_cv = lic.get("cross_layer_mean_cv")
        caption(fig, (
            f"Cross-layer mean cv = {float(x_cv):.4f} — the second, separate "
            f"condition: uniform WITHIN each layer and different BETWEEN them "
            f"is a different sphere at every depth, and every cross-layer "
            f"trajectory metric inherits the rescaling."
            if x_cv is not None else
            "The condition number is what bounds the metric distortion; the "
            "cv is what the paper quotes."))
    return save_figure(fig, out_dir, "sphere_license")


# ---------------------------------------------------------------------------
# D4 — the floor that does not depend on the tokens
# ---------------------------------------------------------------------------

def _d4_bias_energy_floor(run: Run, per_layer: List[dict],
                          out_dir: Path) -> Path:
    """
    The share of measured interaction energy contributed by the LN bias.

    The bias is a common mode added to every token identically, so whatever
    energy it produces is not a property of the token cloud at all.
    status-1c measured 17.9% of E_{β=1} at a bias/signal norm ratio of 0.5,
    and 53.6% at 1.86 — which is why the norm ratio is drawn beside the
    fraction rather than left in the artifact.
    """
    floors = [e.get("bias_floor") if isinstance(e.get("bias_floor"), dict)
              else {} for e in per_layer]
    betas = sorted({float(b) for f in floors
                    for b in (f.get("energy_floor_frac") or {})})

    with plt.rc_context(BLOG_STYLE):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9.2, 6.2), sharex=True,
            gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.12))
        for j, b in enumerate(betas):
            v = np.full(len(floors), np.nan)
            for i, f in enumerate(floors):
                frac = f.get("energy_floor_frac") or {}
                cell = frac.get(b, frac.get(str(b), frac.get(f"{b}")))
                try:
                    v[i] = float(cell)
                except (TypeError, ValueError):
                    pass
            ax.plot(np.arange(v.size), v * 100.0, lw=2.2,
                    color=plt.cm.plasma(0.15 + 0.65 * j / max(len(betas) - 1, 1)),
                    label=f"β = {b:g}")
        ax.set_ylabel("energy from the LN bias (%)")
        ax.legend(loc="best", fontsize=8.5, ncol=2)
        ax.set_title(f"D4 · the LN bias energy floor — {run.label}")

        ratio = record_field(floors, "bias_norm_ratio")
        shift = record_field(floors, "kappa1_shift")
        ax2.plot(np.arange(ratio.size), ratio, color=CATEGORICAL[1], lw=2.2,
                 label="‖β_LN‖ / ‖signal‖")
        ax2.plot(np.arange(shift.size), shift, color=CATEGORICAL[0], lw=2.0,
                 ls="--", label="κ₁ shift (the mechanism)")
        ax2.legend(loc="best", fontsize=8.5)
        ax2.set_ylabel("bias size")
        depth_axis(ax2, run.n_layers)
        caption(fig, "An energy shift with no κ₁ shift would mean something "
                     "other than the common mode is responsible — which is "
                     "why both are drawn.")
    return save_figure(fig, out_dir, "bias_energy_floor")


# ---------------------------------------------------------------------------
# D5 — where "effective rank" stops meaning the same thing
# ---------------------------------------------------------------------------

def _d5_neg_eigen_mass(run: Run, per_layer: List[dict], out_dir: Path) -> Path:
    """
    Negative-eigenvalue mass of each frame's Gram, per layer.

    Zero for the l2 and LN frames by construction; nonzero for the
    functional frame, because symmetric KL violates the triangle inequality
    and the Torgerson Gram is therefore not guaranteed PSD. Reported rather
    than clipped: a frame carrying substantial negative mass is not one in
    which "effective rank" means what it means elsewhere, and D2's
    functional curve has to be read through this figure.
    """
    with plt.rc_context(BLOG_STYLE):
        fig, ax = plt.subplots(figsize=(9.2, 4.4))
        any_drawn = False
        for frame in FRAME_ORDER:
            v = _frame_series(per_layer, frame, "neg_eigen_mass")
            if not np.isfinite(v).any():
                continue
            any_drawn = True
            ax.plot(np.arange(v.size), v * 100.0, color=FRAME_COLORS[frame],
                    lw=2.2, label=FRAME_LABELS[frame])
        if not any_drawn:
            no_data(ax, "no neg_eigen_mass recorded")
        else:
            dropped = _frame_series(per_layer, "functional", "n_dropped_rows")
            if np.isfinite(dropped).any() and np.nansum(dropped) > 0:
                ax.annotate(f"{int(np.nansum(dropped))} row(s) dropped for "
                            f"non-positive diagonal", xy=(0.02, 0.92),
                            xycoords="axes fraction", fontsize=8,
                            color="#4B5563")
            ax.set_ylabel("negative eigenvalue mass (%)")
            ax.legend(loc="best", fontsize=8.5)
        depth_axis(ax, run.n_layers)
        ax.set_title(f"D5 · where the Gram stops being a Gram — {run.label}")
        caption(fig, "Symmetric KL is not a metric, so the Torgerson Gram is "
                     "not guaranteed PSD. Measured at 1.7% on synthetic "
                     "Dirichlet distributions; anything much larger means D2's "
                     "functional rank is not comparable to the others.")
    return save_figure(fig, out_dir, "neg_eigen_mass")
