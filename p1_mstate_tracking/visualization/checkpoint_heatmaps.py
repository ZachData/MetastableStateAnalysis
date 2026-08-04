"""
visualization/checkpoint_heatmaps.py

Class-1 figures, heatmap variant: the developmental picture. x = layer,
y = checkpoint (rows ordered by step, labeled in real steps), color =
metric value. A transition reads as a horizontal discontinuity — a band
where the whole depth profile changes character between adjacent rows.

Rows are ORDINAL (one row per checkpoint), not spaced by log(step):
Pythia's schedule is log-then-linear, and ordinal rows keep every
inter-checkpoint comparison one row apart, which is exactly the
comparison the pilot (item 8) adjudicates. The step labels carry the
spacing information.

The energy heatmap additionally overlays violation-layer markers (the
relative-threshold criterion from metrics.energy_violation_severity,
reimplemented here on numpy only — metrics.py imports torch, which this
package deliberately never does).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.style import BLOG_STYLE
from core.naming import _safe_model_name
from .loaders import _available_betas, _energy_series
from .checkpoints import CHECKPOINT_METRICS, _fmt_step
from .checkpoint_scalars import _violation_layers_np


def _profile_matrix(
    runs: dict, prompt: str, family: List[Tuple[int, str]], fn,
) -> Tuple[np.ndarray, List[int]]:
    """
    Stack one metric's depth profiles into (n_checkpoints, max_layers),
    NaN-padded on the right. Rows follow `family` order (ascending step).
    Returns (matrix, steps_present) — checkpoints with no series are
    dropped rather than left as all-NaN rows.
    """
    rows, steps = [], []
    for step, model in family:
        rd = runs.get((model, prompt))
        if rd is None:
            continue
        try:
            vals = fn(rd)
        except Exception:
            vals = None
        if not vals:
            continue
        rows.append(np.asarray(vals, dtype=float))
        steps.append(step)
    if not rows:
        return np.empty((0, 0)), []
    width = max(r.size for r in rows)
    mat = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        mat[i, : r.size] = r
    return mat, steps


def _draw_heatmap(
    mat: np.ndarray, steps: List[int], *, ax, cmap: str = "magma",
    vlabel: str = "",
) -> None:
    masked = np.ma.masked_invalid(mat)
    pc = ax.pcolormesh(
        np.arange(mat.shape[1] + 1) - 0.5,
        np.arange(mat.shape[0] + 1) - 0.5,
        masked, cmap=cmap, shading="flat",
    )
    cbar = plt.colorbar(pc, ax=ax, pad=0.02)
    cbar.set_label(vlabel, fontsize=9)
    ax.set_yticks(np.arange(len(steps)))
    ax.set_yticklabels([_fmt_step(s) for s in steps], fontsize=7)
    ax.set_ylabel("Training step  (one row per checkpoint)")
    ax.set_xlabel("Layer")
    ax.invert_yaxis()   # earliest checkpoint on top — reads downward as training


def plot_metric_heatmap(
    runs: dict, out_dir: Path, prompt: str, base: str,
    family: List[Tuple[int, str]], metric_name: str,
) -> None:
    spec = CHECKPOINT_METRICS[metric_name]
    mat, steps = _profile_matrix(runs, prompt, family, spec["fn"])
    if mat.size == 0 or len(steps) < 2:
        print(f"  ⚠  heatmap_{metric_name}: <2 checkpoints with data for {base!r}")
        return

    plt.rcParams.update(BLOG_STYLE)
    fig, ax = plt.subplots(figsize=(9.5, max(4.2, 0.28 * len(steps) + 1.8)))
    _draw_heatmap(mat, steps, ax=ax, vlabel=spec["ylabel"])
    ax.set_title(
        f"{spec['title']}: layer × training step  ·  {base}  ·  {prompt}\n"
        "a transition = a horizontal band where adjacent rows change character",
        fontsize=11, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"heatmap_{metric_name}_{_safe_model_name(base)}_{prompt}.png"
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


def plot_energy_heatmap(
    runs: dict, out_dir: Path, prompt: str, base: str,
    family: List[Tuple[int, str]], beta: Optional[float] = None,
) -> None:
    """E_β layer × checkpoint heatmap with violation layers marked (white
    ×). Uses β=1.0 when present, else the first available β."""
    # Resolve β from the first run that has energies
    if beta is None:
        for _, model in family:
            rd = runs.get((model, prompt))
            if rd is None:
                continue
            betas = _available_betas(rd)
            if betas:
                beta = 1.0 if 1.0 in betas else betas[0]
                break
    if beta is None:
        print(f"  ⚠  heatmap_energy: no energies.json for {base!r} @ {prompt!r}")
        return

    mat, steps = _profile_matrix(
        runs, prompt, family, lambda rd: _energy_series(rd, beta),
    )
    if mat.size == 0 or len(steps) < 2:
        print(f"  ⚠  heatmap_energy: <2 checkpoints with data for {base!r}")
        return

    plt.rcParams.update(BLOG_STYLE)
    fig, ax = plt.subplots(figsize=(9.5, max(4.2, 0.28 * len(steps) + 1.8)))
    _draw_heatmap(mat, steps, ax=ax, cmap="viridis", vlabel=rf"$E_\beta$ (β={beta:g})")

    # Violation overlay — same relative-drop criterion Phase 1 adjudicated
    # Theorem 3.4 with.
    for i in range(mat.shape[0]):
        for layer in _violation_layers_np(mat[i]):
            ax.plot(layer, i, marker="x", color="white", markersize=6,
                    markeredgewidth=1.6, zorder=5)

    ax.set_title(
        f"Interaction energy (β={beta:g}): layer × training step  ·  {base}  ·  {prompt}\n"
        "white × = energy-monotonicity violation (relative-drop criterion)",
        fontsize=11, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"heatmap_energy_b{beta:g}_{_safe_model_name(base)}_{prompt}.png"
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


def generate_heatmap_figures(
    runs: dict, out_dir: Path, prompt: str, base: str,
    family: List[Tuple[int, str]],
) -> None:
    if len(family) < 2:
        print(f"  ⚠  heatmaps: family {base!r} has <2 checkpoints, skipping")
        return
    for metric_name in CHECKPOINT_METRICS:
        plot_metric_heatmap(runs, out_dir, prompt, base, family, metric_name)
    plot_energy_heatmap(runs, out_dir, prompt, base, family)
