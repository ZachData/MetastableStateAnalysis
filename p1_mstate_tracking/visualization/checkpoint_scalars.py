"""
visualization/checkpoint_scalars.py

Class-2 figures — the class that doesn't exist in Phase 1 because there
was nothing to plot against: one number per checkpoint, plotted vs.
log(training step). These are the pilot's (item 8) actual deliverable:
the knees in these curves are what locate the energy-monotonicity break,
Fiedler drop, and effective-rank collapse, and what places the 1.4B
adaptive slots.

Three things live here:

  1. SCALAR_EXTRACTORS — run_dir -> float reductions of each depth
     profile (violation counts/severity per β, min Fiedler, terminal and
     min effective rank, plateau count/length, peak cluster membership).
  2. distance-from-random — per per-layer metric, the L2 distance between
     checkpoint k's depth profile and the norm-matched random baseline's
     (both resampled to a common normalized-depth grid). "Smooth
     transition from random to trained" is literally this curve: smooth
     if training moves the model smoothly, kinked at a formation event.
  3. Transition detection — per scalar series, the consecutive-checkpoint
     interval with the largest normalized jump; written to
     transitions.json so the adaptive-slot placement (and the filmstrip
     module's snapshot selection) consumes a file, not a figure.

Violation criterion is metrics.energy_violation_severity's relative-drop
rule, reimplemented on numpy only (metrics.py imports torch; this
package deliberately never does). REL_TOL is READ FROM metrics.py's source
at import time rather than duplicated as a literal — see the note above its
definition below.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.style import BLOG_STYLE
from core.naming import _safe_model_name
from .loaders import _available_betas, _energy_series, _trajectory, _sinkhorn
from .series import _series_or_aggregate
from .checkpoints import (
    CHECKPOINT_METRICS, TRANSITION_SPAN_COLOR, TRANSITION_SPAN_ALPHA,
    _step_x, format_step_axis, family_baselines,
)
from core.plot_utils import _spans

# ---------------------------------------------------------------------------
# REL_TOL — kept in sync with metrics.ENERGY_VIOLATION_REL_TOL AT IMPORT TIME
# ---------------------------------------------------------------------------
# This used to be a hand-synced literal with a comment asking future editors
# to remember. That is not a mechanism: the two constants define the same
# criterion, this module's numbers appear in the checkpoint figures and
# metrics.py's appear in the text report, and a divergence would show up as
# two different violation counts for the same run with nothing to indicate
# which was stale.
#
# metrics.py imports torch and this package deliberately never does, so the
# module cannot simply be imported. It is parsed instead: the constant is
# read out of the source with ast, which costs one file read at import and
# cannot execute the torch import. If the constant is ever renamed or moved,
# this raises at import rather than silently falling back — a stale literal
# that keeps working is the failure this is designed to prevent.

def _rel_tol_from_metrics() -> float:
    import ast as _ast
    from pathlib import Path as _Path
    src = _Path(__file__).resolve().parents[2] / "core" / "metrics.py"
    if not src.exists():
        raise ImportError(
            f"checkpoint_scalars: cannot locate {src} to read "
            f"ENERGY_VIOLATION_REL_TOL. This module's violation criterion "
            f"must match metrics.py's; refusing to guess."
        )
    tree = _ast.parse(src.read_text())
    for node in tree.body:
        if isinstance(node, (_ast.Assign, _ast.AnnAssign)):
            targets = ([node.target] if isinstance(node, _ast.AnnAssign)
                       else node.targets)
            for t in targets:
                if isinstance(t, _ast.Name) and t.id == "ENERGY_VIOLATION_REL_TOL":
                    return float(_ast.literal_eval(node.value))
    raise ImportError(
        "checkpoint_scalars: ENERGY_VIOLATION_REL_TOL not found in "
        "core/metrics.py. It was renamed or moved; update this reader "
        "rather than restoring a literal."
    )


REL_TOL: float = _rel_tol_from_metrics()


# ─────────────────────────────────────────────────────────────────────────────
# Violation logic (numpy port of metrics.energy_violation_severity)
# ─────────────────────────────────────────────────────────────────────────────

def _violation_layers_np(energies, rel_tol: float = REL_TOL) -> List[int]:
    """1-indexed layers where (E_prev - E_curr)/|E_prev| > rel_tol."""
    arr = np.asarray(energies, dtype=np.float64)
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return []
    diffs = np.diff(arr)
    ref = np.maximum(np.abs(arr[:-1]), 1e-12)
    rel_drop = np.where(valid[:-1] & valid[1:], -diffs / ref, 0.0)
    return [i + 1 for i, v in enumerate(rel_drop > rel_tol) if v]


def _violation_severity_np(energies, rel_tol: float = REL_TOL) -> Tuple[int, float, float]:
    """(n_violations, sum_severity, first_violation_layer_or_nan)."""
    arr = np.asarray(energies, dtype=np.float64)
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return 0, 0.0, float("nan")
    diffs = np.diff(arr)
    ref = np.maximum(np.abs(arr[:-1]), 1e-12)
    rel_drop = np.where(valid[:-1] & valid[1:], -diffs / ref, 0.0)
    mask = rel_drop > rel_tol
    layers = [i + 1 for i, v in enumerate(mask) if v]
    return (int(mask.sum()),
            float(rel_drop[mask].sum()) if mask.any() else 0.0,
            float(layers[0]) if layers else float("nan"))


# ─────────────────────────────────────────────────────────────────────────────
# Scalar extractors — run_dir -> float (NaN when unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def _nan_reduce(series, red) -> float:
    if series is None:
        return float("nan")
    arr = np.asarray(series, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(red(arr))


def _terminal(series) -> float:
    if series is None:
        return float("nan")
    arr = np.asarray(series, dtype=float)
    valid = arr[~np.isnan(arr)]
    return float(valid[-1]) if valid.size else float("nan")


def _pick_beta(run_dir: Path) -> Optional[float]:
    betas = _available_betas(run_dir)
    if not betas:
        return None
    return 1.0 if 1.0 in betas else betas[0]


def _s_n_violations(run_dir: Path) -> float:
    beta = _pick_beta(run_dir)
    if beta is None:
        return float("nan")
    n, _, _ = _violation_severity_np(_energy_series(run_dir, beta))
    return float(n)


def _s_violation_severity(run_dir: Path) -> float:
    beta = _pick_beta(run_dir)
    if beta is None:
        return float("nan")
    _, sev, _ = _violation_severity_np(_energy_series(run_dir, beta))
    return sev


def _s_first_violation_layer(run_dir: Path) -> float:
    beta = _pick_beta(run_dir)
    if beta is None:
        return float("nan")
    _, _, first = _violation_severity_np(_energy_series(run_dir, beta))
    return first


def _s_min_fiedler(run_dir: Path) -> float:
    layers = _sinkhorn(run_dir).get("layers", [])
    vals = [lr.get("fiedler_mean") for lr in layers]
    return _nan_reduce([v for v in vals if v is not None] or None, np.nanmin)


def _s_terminal_effective_rank(run_dir: Path) -> float:
    return _terminal(CHECKPOINT_METRICS["effective_rank"]["fn"](run_dir))


def _s_min_effective_rank(run_dir: Path) -> float:
    return _nan_reduce(CHECKPOINT_METRICS["effective_rank"]["fn"](run_dir), np.nanmin)


def _s_peak_cluster_membership(run_dir: Path) -> float:
    return _nan_reduce(CHECKPOINT_METRICS["cluster_membership"]["fn"](run_dir), np.nanmax)


def _s_plateau_count(run_dir: Path) -> float:
    layers = _trajectory(run_dir).get("plateau_layers", [])
    return float(len(_spans(layers))) if layers else 0.0


def _s_plateau_mean_length(run_dir: Path) -> float:
    layers = _trajectory(run_dir).get("plateau_layers", [])
    spans = _spans(layers)
    if not spans:
        return 0.0
    return float(np.mean([e - s + 1 for s, e in spans]))


# name -> (extractor, ylabel). Names are the keys transitions.json uses.
SCALAR_EXTRACTORS: Dict[str, Tuple[Callable[[Path], float], str]] = {
    "n_energy_violations":    (_s_n_violations,          "Energy violations (count)"),
    "violation_sum_severity": (_s_violation_severity,    "Violation severity (Σ rel. drop)"),
    "first_violation_layer":  (_s_first_violation_layer, "First violation layer"),
    "min_fiedler":            (_s_min_fiedler,           "Min Fiedler value"),
    "terminal_effective_rank": (_s_terminal_effective_rank, "Terminal effective rank"),
    "min_effective_rank":     (_s_min_effective_rank,    "Min effective rank"),
    "peak_cluster_membership": (_s_peak_cluster_membership, "Peak cluster membership"),
    "plateau_count":          (_s_plateau_count,         "Plateau window count"),
    "plateau_mean_length":    (_s_plateau_mean_length,   "Mean plateau length (layers)"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Distance from random — the "smooth or not" curve
# ─────────────────────────────────────────────────────────────────────────────

def _resampled_l2(a: np.ndarray, b: np.ndarray, n: int = 64) -> float:
    """RMS distance between two depth profiles on a common
    normalized-depth grid — NaN-aware, NaN if <2 valid points on either."""
    def _resample(v):
        v = np.asarray(v, dtype=float)
        m = ~np.isnan(v)
        if m.sum() < 2:
            return None
        x = np.linspace(0, 1, v.size)
        return np.interp(np.linspace(0, 1, n), x[m], v[m])
    ra, rb = _resample(a), _resample(b)
    if ra is None or rb is None:
        return float("nan")
    return float(np.sqrt(np.mean((ra - rb) ** 2)))


def compute_distance_from_random(
    runs: dict, prompt: str, base: str, family: List[Tuple[int, str]],
    random_agg: Optional[dict] = None,
) -> Dict[str, Tuple[List[int], List[float]]]:
    """
    {metric_name: (steps, distances)} — per per-layer metric, RMS distance
    of each checkpoint's profile from the norm-matched random baseline's
    (multi-seed mean when random_agg has one). Empty dict when the family
    has no '{base}-random' baseline.
    """
    models = sorted({m for (m, p) in runs.keys() if p == prompt})
    if random_agg:
        models = sorted(set(models) | {m for (m, p) in random_agg.keys() if p == prompt})
    rand = family_baselines(base, models)["random"]
    if rand is None:
        return {}

    out: Dict[str, Tuple[List[int], List[float]]] = {}
    for name, spec in CHECKPOINT_METRICS.items():
        mean_r, _, _ = _series_or_aggregate(
            rand, prompt, runs.get((rand, prompt)), spec["fn"],
            spec["agg_key"], random_agg,
        )
        if mean_r is None:
            continue
        steps, dists = [], []
        for step, model in family:
            rd = runs.get((model, prompt))
            if rd is None:
                continue
            try:
                series = spec["fn"](rd)
            except Exception:
                series = None
            if not series:
                continue
            steps.append(step)
            dists.append(_resampled_l2(series, mean_r))
        if steps:
            out[name] = (steps, dists)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Transition detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_transitions(steps: List[int], values: List[float]) -> Optional[dict]:
    """
    The consecutive-checkpoint interval with the largest jump, normalized
    by the series' robust scale (IQR, falling back to ptp). None when <3
    finite points or zero scale. 'normalized_jump' is comparable across
    metrics; ~>0.5 means one inter-checkpoint interval carries half the
    metric's whole dynamic range.
    """
    s = np.asarray(steps, dtype=float)
    v = np.asarray(values, dtype=float)
    m = np.isfinite(v)
    s, v = s[m], v[m]
    if v.size < 3:
        return None
    q75, q25 = np.nanpercentile(v, [75, 25])
    scale = q75 - q25
    if scale <= 0:
        scale = np.ptp(v)
    if scale <= 0:
        return None
    jumps = np.abs(np.diff(v)) / scale
    i = int(np.argmax(jumps))
    return {
        "step_lo": int(s[i]), "step_hi": int(s[i + 1]),
        "value_lo": float(v[i]), "value_hi": float(v[i + 1]),
        "normalized_jump": float(jumps[i]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _scalar_series(
    runs: dict, prompt: str, family: List[Tuple[int, str]], fn,
) -> Tuple[List[int], List[float]]:
    steps, vals = [], []
    for step, model in family:
        rd = runs.get((model, prompt))
        if rd is None:
            continue
        try:
            vals.append(float(fn(rd)))
        except Exception:
            vals.append(float("nan"))
        steps.append(step)
    return steps, vals


def _draw_scalar_axis(ax, steps, vals, ylabel, transition=None):
    x = _step_x(steps)
    ax.plot(x, vals, marker="o", markersize=4.5, linewidth=1.8,
            color="#2563EB", zorder=3)
    if transition is not None and transition["normalized_jump"] > 0:
        lo, hi = _step_x([transition["step_lo"], transition["step_hi"]])
        ax.axvspan(lo, hi, color=TRANSITION_SPAN_COLOR,
                   alpha=TRANSITION_SPAN_ALPHA, zorder=0)
    format_step_axis(ax, steps)
    ax.set_ylabel(ylabel)


def plot_scalar_grid(
    runs: dict, out_dir: Path, prompt: str, base: str,
    family: List[Tuple[int, str]],
) -> Dict[str, Optional[dict]]:
    """
    Every scalar in SCALAR_EXTRACTORS vs. training step, one panel each,
    with the sharpest inter-checkpoint interval shaded per panel. Returns
    {scalar_name: transition_or_None} for transitions.json.
    """
    plt.rcParams.update(BLOG_STYLE)
    names = list(SCALAR_EXTRACTORS.keys())
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow),
                             squeeze=False)
    transitions: Dict[str, Optional[dict]] = {}
    plotted = False

    for i, name in enumerate(names):
        ax = axes[i // ncol][i % ncol]
        fn, ylabel = SCALAR_EXTRACTORS[name]
        steps, vals = _scalar_series(runs, prompt, family, fn)
        finite = [v for v in vals if np.isfinite(v)]
        if len(finite) < 2:
            ax.axis("off")
            transitions[name] = None
            continue
        tr = detect_transitions(steps, vals)
        transitions[name] = tr
        _draw_scalar_axis(ax, steps, vals, ylabel, transition=tr)
        ax.set_title(name, fontsize=10)
        plotted = True

    for i in range(len(names), nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")

    if not plotted:
        plt.close(fig)
        print(f"  ⚠  scalar_grid: nothing to plot for {base!r} @ {prompt!r}")
        return transitions

    fig.suptitle(
        f"Per-checkpoint scalars vs. training step  ·  {base}  ·  {prompt}\n"
        "shaded span = sharpest inter-checkpoint change (candidate adaptive-slot interval)",
        fontsize=12, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"scalars_vs_step_{_safe_model_name(base)}_{prompt}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")
    return transitions


def plot_distance_from_random(
    runs: dict, out_dir: Path, prompt: str, base: str,
    family: List[Tuple[int, str]], random_agg: Optional[dict] = None,
) -> Dict[str, Optional[dict]]:
    """
    Departure-from-random per metric vs. training step, all metrics on one
    figure (own panel each — units differ). Returns
    {f"dist_from_random__{metric}": transition_or_None}.
    """
    dist = compute_distance_from_random(runs, prompt, base, family,
                                        random_agg=random_agg)
    transitions: Dict[str, Optional[dict]] = {}
    if not dist:
        print(f"  ⚠  distance_from_random: no '{base}-random' baseline @ {prompt!r}")
        return transitions

    plt.rcParams.update(BLOG_STYLE)
    names = list(dist.keys())
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow),
                             squeeze=False)

    for i, name in enumerate(names):
        ax = axes[i // ncol][i % ncol]
        steps, dists = dist[name]
        tr = detect_transitions(steps, dists)
        transitions[f"dist_from_random__{name}"] = tr
        _draw_scalar_axis(ax, steps, dists,
                          "RMS distance from random profile", transition=tr)
        ax.set_title(name, fontsize=10)

    for i in range(len(names), nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")

    fig.suptitle(
        f"Departure from the norm-matched random baseline  ·  {base}  ·  {prompt}\n"
        "smooth curve = smooth random→trained transition; a knee = a formation event",
        fontsize=12, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"distance_from_random_{_safe_model_name(base)}_{prompt}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")
    return transitions


def write_transitions(
    out_dir: Path, prompt: str, base: str,
    transitions: Dict[str, Optional[dict]],
) -> Path:
    """
    transitions.json — the machine-readable deliverable item 8's
    adaptive-slot placement consumes. Metrics ranked by normalized jump;
    'consensus_intervals' counts how many metrics put their sharpest
    change in the same step interval.
    """
    ranked = sorted(
        ((k, v) for k, v in transitions.items() if v is not None),
        key=lambda kv: -kv[1]["normalized_jump"],
    )
    consensus: Dict[str, int] = {}
    for _, tr in ranked:
        key = f"{tr['step_lo']}→{tr['step_hi']}"
        consensus[key] = consensus.get(key, 0) + 1

    payload = {
        "base_model": base,
        "prompt": prompt,
        "per_metric": {k: v for k, v in transitions.items()},
        "ranked_by_jump": [
            {"metric": k, **v} for k, v in ranked
        ],
        "consensus_intervals": dict(
            sorted(consensus.items(), key=lambda kv: -kv[1])
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"transitions_{_safe_model_name(base)}_{prompt}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  ✓  {path.name}")
    return path


def generate_scalar_figures(
    runs: dict, out_dir: Path, prompt: str, base: str,
    family: List[Tuple[int, str]], random_agg: Optional[dict] = None,
) -> Dict[str, Optional[dict]]:
    """Both class-2 figure sets + transitions.json. Returns the merged
    transitions dict (the filmstrip module's snapshot selector reads it)."""
    if len(family) < 3:
        print(f"  ⚠  scalars: family {base!r} has <3 checkpoints, skipping")
        return {}
    transitions = plot_scalar_grid(runs, out_dir, prompt, base, family)
    transitions.update(
        plot_distance_from_random(runs, out_dir, prompt, base, family,
                                  random_agg=random_agg)
    )
    write_transitions(out_dir, prompt, base, transitions)
    return transitions
