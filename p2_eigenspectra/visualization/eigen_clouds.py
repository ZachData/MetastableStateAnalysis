"""
p2_eigenspectra/visualization/eigen_clouds.py

Class-3 figures: the raw OV spectrum in the complex plane, as a row of
small multiples across selected checkpoints. Everything else in this
package plots a REDUCTION of the spectrum (a fraction, a norm, a
dimension); this plots the thing itself, which is the only figure that can
show a shape nobody thought to reduce.

Why a filmstrip and not a sweep: 27 overlaid point clouds of d_model
points each is a solid blob. The snapshot steps come from the same
transitions.json the Phase 1 filmstrips consume, via p1's
`select_snapshot_steps` — so the checkpoints shown here are the ones some
scalar actually changed sharply between, not an arbitrary even spread.

What to look for, and what each would mean:

  A disk at step 0. The composed OV at init is a sum of products of
  independent Gaussian blocks; its spectrum fills a disk roughly
  symmetrically about both axes. This is the null the whole phase is
  measured against, and it should be visible without any statistics.

  Condensation onto the real axis. Eigenvalues collapsing towards Im λ = 0
  means the learned map is approaching a symmetric one — the Schur and
  symmetric-part decompositions converge, `methods_agree` goes true, and
  Phase 2's projectors stop being method-dependent.

  Real-axis outliers. A few eigenvalues detaching from the bulk along the
  real axis is a low-rank structure appearing on top of the noise floor.
  Whether they detach on the negative (repulsive) or positive side is
  exactly the question `frac_repulsive` answers in aggregate — here you
  can see how MANY carry the mass, which a fraction cannot tell you.

  Bulk radius growth. Read against the spectral-radius panel in the
  scalar grid; a growing disk with an unchanged sign split is a scale
  change, not a structural one.

The symmetric-part spectrum is drawn as a marginal histogram along the
real axis of the same panel. The horizontal gap between that marginal and
the projection of the full spectrum IS the non-normality, shown rather
than summarized.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.style import BLOG_STYLE
from core.naming import _safe_model_name
from p1_mstate_tracking.visualization.checkpoints import _fmt_step
from p1_mstate_tracking.visualization.checkpoint_filmstrip import (
    select_snapshot_steps,
)

from .loaders import eigen_cloud, layer_field, layer_keys

# Point styling: at d_model = 1024 per panel a plain scatter is a blob, so
# points are small, translucent, and sign-colored — the sign split is the
# only categorical information in the cloud and it should be readable
# without counting.
_NEG_COLOR = "#DC2626"   # Re λ < 0, repulsive
_POS_COLOR = "#2563EB"   # Re λ > 0, attractive


def _panel(
    ax, re_: np.ndarray, im_: np.ndarray, sym: Optional[np.ndarray],
    lim: float, show_marginal: bool = True,
) -> None:
    neg = re_ < 0
    ax.scatter(re_[neg], im_[neg], s=3.0, alpha=0.35, linewidths=0,
               color=_NEG_COLOR, zorder=3)
    ax.scatter(re_[~neg], im_[~neg], s=3.0, alpha=0.35, linewidths=0,
               color=_POS_COLOR, zorder=3)
    ax.axvline(0.0, color="#111827", linewidth=0.9, zorder=4)
    ax.axhline(0.0, color="#9CA3AF", linewidth=0.6, zorder=2)

    if show_marginal and sym is not None and sym.size:
        # Symmetric-part spectrum as a rug along the bottom of the panel.
        counts, edges = np.histogram(sym, bins=48, range=(-lim, lim))
        if counts.max() > 0:
            h = 0.18 * (2 * lim) * counts / counts.max()
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.bar(centers, h, bottom=-lim, width=(edges[1] - edges[0]),
                   color="#6B7280", alpha=0.55, linewidth=0, zorder=1)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")


def plot_eigen_cloud_filmstrip(
    p2_dir: Path, summaries: Dict[str, dict], out_dir: Path, base: str,
    family: List[Tuple[int, str]], layer_idx: int,
    snapshot_steps: Optional[List[int]] = None,
    k: int = 6,
    transitions: Optional[dict] = None,
) -> None:
    """
    One row of complex-plane panels for a fixed layer across checkpoints.

    Panels share axis limits (the largest radius across the row) so the
    bulk growing is a real visual change and not an artifact of per-panel
    autoscaling. That shared scale is the whole point of the figure — the
    single most common way to make this plot lie is to let each panel
    rescale itself.
    """
    if snapshot_steps is None:
        snapshot_steps = select_snapshot_steps(
            [s for s, _ in family], transitions=transitions, k=k,
        )

    by_step = {s: m for s, m in family}
    panels: List[Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    for step in snapshot_steps:
        model = by_step.get(step)
        if model is None:
            continue
        cloud = eigen_cloud(p2_dir, model, layer_idx)
        if cloud is None:
            continue
        re_, im_, sym = cloud
        panels.append((step, re_, im_, sym))

    if len(panels) < 2:
        print(f"  ⚠  eigen_cloud L{layer_idx}: <2 checkpoints with "
              f"ov_decomp arrays for {base!r}")
        return

    lim = max(
        float(np.nanmax(np.abs(np.concatenate([re_, im_]))))
        for _, re_, im_, _ in panels
    )
    lim = lim * 1.08 if np.isfinite(lim) and lim > 0 else 1.0

    plt.rcParams.update(BLOG_STYLE)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.5), squeeze=False)

    for i, (step, re_, im_, sym) in enumerate(panels):
        ax = axes[0][i]
        _panel(ax, re_, im_, sym, lim)
        summary = summaries.get(by_step[step], {})
        rep = layer_field(summary, "frac_repulsive")
        rep_v = rep[layer_idx] if layer_idx < rep.size else float("nan")
        cplx = layer_field(summary, "frac_complex")
        cplx_v = cplx[layer_idx] if layer_idx < cplx.size else float("nan")
        ax.set_title(
            f"step {_fmt_step(step)}\n"
            f"rep={rep_v:.3f}  cplx={cplx_v:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("Re λ", fontsize=8)
        if i == 0:
            ax.set_ylabel("Im λ", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"OV eigenvalue spectrum, layer {layer_idx}  ·  {base}\n"
        "red = Re λ < 0 (repulsive) · blue = Re λ > 0 · gray rug = "
        "symmetric-part spectrum · shared axis limits across panels",
        fontsize=11, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"eigen_cloud_L{layer_idx:02d}_{_safe_model_name(base)}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


def available_decomp_layers(p2_dir: Path, model: str) -> List[int]:
    """Layer indices whose eigenvalue arrays are present in ov_decomp."""
    from .loaders import decomp_path
    import re as _re

    p = decomp_path(p2_dir, model)
    if p is None:
        return []
    with np.load(p) as z:
        idxs = sorted(
            int(m.group(1))
            for m in (_re.match(r"^eig_real_layer_(\d+)$", f) for f in z.files)
            if m
        )
    return idxs


def default_cloud_layers(
    p2_dir: Path, summaries: Dict[str, dict], family: List[Tuple[int, str]],
    n_want: int = 3,
) -> List[int]:
    """
    Which layers to draw clouds for when the caller doesn't say.

    Preference order: the layer whose `frac_repulsive` moves most between
    the first and last checkpoint (where the developmental signal is), then
    an early and a late layer for context. Falls back to an even spread
    when no summaries are usable — the point is never to default to layer 0
    alone, which on a pre-LN model is the least informative depth.
    """
    if not family:
        return []
    avail = available_decomp_layers(p2_dir, family[-1][1])
    if not avail:
        return []
    first, last = family[0][1], family[-1][1]
    a = layer_field(summaries.get(first, {}), "frac_repulsive")
    b = layer_field(summaries.get(last, {}), "frac_repulsive")
    picks: List[int] = []
    if a.size and a.size == b.size:
        delta = np.abs(b - a)
        for idx in np.argsort(-delta):
            if int(idx) in avail:
                picks.append(int(idx))
                break
    for cand in (avail[0], avail[len(avail) // 2], avail[-1]):
        if cand not in picks:
            picks.append(cand)
    return sorted(picks[:n_want])


def generate_eigen_cloud_figures(
    p2_dir: Path, summaries: Dict[str, dict], out_dir: Path, base: str,
    family: List[Tuple[int, str]],
    layers: Optional[List[int]] = None,
    k: int = 6,
    transitions: Optional[dict] = None,
) -> None:
    if len(family) < 2:
        print(f"  ⚠  eigen_clouds: family {base!r} has <2 checkpoints, skipping")
        return
    if layers is None:
        layers = default_cloud_layers(p2_dir, summaries, family)
    if not layers:
        print(f"  ⚠  eigen_clouds: no ov_decomp_*.npz eigenvalue arrays for {base!r}")
        return
    for layer_idx in layers:
        plot_eigen_cloud_filmstrip(
            p2_dir, summaries, out_dir, base, family, layer_idx,
            k=k, transitions=transitions,
        )
