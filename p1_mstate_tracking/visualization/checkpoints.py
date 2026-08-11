"""
p1_mstate_tracking/visualization/checkpoints.py

Checkpoint-sweep conventions (transition plan v2, item 7). Pure string /
dict / axis logic — no file I/O beyond what series.py extractors already
do, no figures of its own. Everything the four checkpoint figure modules
(checkpoint_sweep, checkpoint_heatmaps, checkpoint_scalars,
checkpoint_filmstrip) share lives here so the conventions can't drift
apart:

  - the '-step{N}' name grammar (core/pythia_registry.py keys:
    'pythia-410m-step2000') and family grouping. The grammar itself was
    moved to core/model_family.py and is re-exported here; this module
    imports matplotlib, so analysis code that only needed to parse a step
    was pulling in a plotting dependency to get it,
  - baseline resolution: '{base}-random' (norm-matched continuity
    control) and step 0 (developmental origin) are kept as two separate
    objects, per the plan's two-baseline policy — never folded into one
    "random" condition,
  - the log(step+1) x-axis convention (Pythia's schedule is log-spaced
    then linear; a pure log axis can't place step 0, symlog buries the
    log-spaced early steps),
  - the sequential colormap keyed to log(step+1), with step 0 pulled OUT
    of the colormap and drawn in its own near-black dotted style so the
    origin never reads as "just the earliest checkpoint".

The metric registry at the bottom is the single source of truth for
which per-layer series get swept/heatmapped/distance-measured — one
entry here adds the metric to all three figure classes at once, the same
way SERIES_EXTRACTORS works in random_aggregate.py.

Maybe later, not current work (2026-07-18): the plan's "cheap SLT anchor"
(per-checkpoint training loss + per-layer weight norms) would show up
here as two more METRIC registry entries if it's ever added — scalar,
same shape as everything else this file already sweeps. Not added now;
current checkpoint runs use known-event anchors directly rather than the
dense pilot sweep this was meant to support. See status-2.md.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from core.style import UNTRAINED_COLOR
from .series import (
    _mass_near_1_series,
    _effective_rank_series,
    _cluster_membership_series,
    _cluster_count_series,
    _cka_prev_series,
    _fiedler_mean_series,
)

# ─────────────────────────────────────────────────────────────────────────────
# Name grammar — '{base}-step{N}', additive to naming.py's existing
# '-random' / '@attn' / '@ffn' / '@Niter' grammar. Kept out of naming.py
# so nothing about the existing suffix handling changes.
# ─────────────────────────────────────────────────────────────────────────────

# The grammar itself now lives in `core/model_family.py`, which is
# stdlib-only. It was defined here, in a module that imports matplotlib and
# `.series`, so every ANALYSIS module that needed to know a run's checkpoint
# step either acquired a plotting dependency or re-typed the regex —
# `p2b_imaginary` was about to be the third copy. `core/model_family.py`
# already exists to stop exactly this ("two idioms that agree on registry
# keys and disagree elsewhere"), so the grammar moved there rather than to
# `core/naming.py`, which imports `core.style` and so imports matplotlib too.
#
# Re-exported under the leading-underscore names this module's four figure
# consumers already call, so nothing downstream changes.

from core.model_family import (
    CHECKPOINT_STEP_RE as _STEP_RE,
    checkpoint_step as _checkpoint_step,
    checkpoint_base as _checkpoint_base,
    checkpoint_families,
    is_checkpoint,
    sort_by_step,
)


def family_baselines(base: str, models: List[str]) -> Dict[str, Optional[str]]:
    """
    The two baseline objects for one checkpoint family, resolved against
    what's actually present in `models`:

      'random' : '{base}-random' if present — the norm-matched continuity
                 control. Also accepts the registry's cross-size control
                 name only on exact match (e.g. 'pythia-1.4b-random' is
                 NOT silently used for a 'pythia-410m' family — a
                 different-size random control is a different object).
      'step0'  : '{base}-step0' if present — the developmental origin.

    Either may be None. Callers draw whichever exist and say nothing
    about the other — no cross-family substitution.
    """
    return {
        "random": f"{base}-random" if f"{base}-random" in models else None,
        "step0":  f"{base}-step0" if f"{base}-step0" in models else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step axis + colormap conventions
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_CMAP = cm.viridis

# Step 0 is a separate object (developmental origin), not the low end of
# the training sweep — drawn dotted near-black, never from the colormap.
STEP0_STYLE = dict(color="#111827", linestyle=":", linewidth=2.0, alpha=0.9)

# Norm-matched random baseline keeps the package-wide untrained
# convention: gray, dashed.
RANDOM_BASELINE_STYLE = dict(color=UNTRAINED_COLOR, linestyle="--",
                             linewidth=2.0, alpha=0.85)

TRANSITION_SPAN_COLOR = "#FCA5A5"   # shaded step interval of sharpest change
TRANSITION_SPAN_ALPHA = 0.25


def _step_x(steps) -> np.ndarray:
    """Axis position for training steps: log10(step + 1). Monotone, places
    step 0 at x=0, keeps Pythia's log-spaced early steps evenly spread."""
    return np.log10(np.asarray(steps, dtype=float) + 1.0)


def step_norm(steps: List[int]) -> Normalize:
    """Colormap normalization over log10(step+1) for the NONZERO steps of
    a sweep (step 0 is styled separately — see STEP0_STYLE)."""
    nz = [s for s in steps if s > 0] or [1]
    return Normalize(vmin=_step_x([min(nz)])[0], vmax=_step_x([max(nz)])[0])


def step_color(step: int, norm: Normalize):
    return CHECKPOINT_CMAP(norm(_step_x([step])[0]))


def format_step_axis(ax, steps: List[int], axis: str = "x",
                     max_ticks: int = 9) -> None:
    """
    Put real training-step labels on a log10(step+1) axis. Ticks are the
    actual checkpoint steps, thinned to at most max_ticks (endpoints always
    kept) — no synthetic decades, so every label is a step that exists.
    """
    steps = sorted(set(int(s) for s in steps))
    if len(steps) > max_ticks:
        idx = np.unique(np.linspace(0, len(steps) - 1, max_ticks).astype(int))
        steps = [steps[i] for i in idx]
    pos = _step_x(steps)
    labels = [_fmt_step(s) for s in steps]
    if axis == "x":
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Training step  (log-spaced axis)")
    else:
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.set_ylabel("Training step  (log-spaced axis)")


def _fmt_step(step: int) -> str:
    if step >= 1000 and step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def add_step_colorbar(fig, ax, steps: List[int], norm: Normalize) -> None:
    """Colorbar labeled in real training steps for a colormap sweep."""
    sm = cm.ScalarMappable(norm=norm, cmap=CHECKPOINT_CMAP)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    nz = sorted(s for s in set(steps) if s > 0)
    if len(nz) > 7:
        idx = np.unique(np.linspace(0, len(nz) - 1, 7).astype(int))
        nz = [nz[i] for i in idx]
    cbar.set_ticks(_step_x(nz))
    cbar.set_ticklabels([_fmt_step(s) for s in nz])
    cbar.set_label("Training step", fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# Metric registry — one entry per per-layer series that participates in
# the checkpoint figure classes. agg_key matches random_aggregate.py's
# SERIES_EXTRACTORS names so the random baseline can render as a
# multi-seed mean ± std band through series._series_or_aggregate.
# Energies are handled separately (per-β, from energies.json).
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_METRICS: Dict[str, dict] = {
    "mass_near_1": dict(
        fn=_mass_near_1_series, agg_key="mass_near_1",
        ylabel="Fraction of pairs  ⟨xᵢ,xⱼ⟩ > 0.9",
        title="Mass-near-1", ylim=None,
    ),
    "effective_rank": dict(
        fn=_effective_rank_series, agg_key="effective_rank",
        ylabel="Effective rank",
        title="Effective rank", ylim=None,
    ),
    "cluster_membership": dict(
        fn=_cluster_membership_series, agg_key="cluster_membership",
        ylabel="Fraction of tokens in a cluster",
        title="Cluster membership", ylim=(-0.02, 1.06),
    ),
    "cluster_count": dict(
        fn=_cluster_count_series, agg_key="cluster_count",
        ylabel="HDBSCAN cluster count (k)",
        title="Cluster count", ylim=None,
    ),
    "cka_prev": dict(
        fn=_cka_prev_series, agg_key="cka_prev",
        ylabel="CKA(layer, layer−1)",
        title="CKA vs. previous layer", ylim=(0.90, 1.01),
    ),
    "fiedler_mean": dict(
        fn=_fiedler_mean_series, agg_key="fiedler_mean",
        ylabel="Fiedler value (mean across heads)",
        title="Fiedler value", ylim=None,
    ),
}
