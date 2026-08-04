"""
p2_eigenspectra/visualization/p2_scalars.py

Class-2 figures: one number per checkpoint, plotted against
log(training step). Same role as p1's `checkpoint_scalars` — the knees in
these curves are what date the phenomenon — but Phase 2 splits the class in
two, because its two artifact families have different denominators.

  WEIGHT_SCALARS  reduce `ov_summary_*.json`. One value per checkpoint,
                  no prompt, no error bar to draw and none needed: the
                  quantity is a deterministic function of the weights.

  RUN_SCALARS     reduce `phase2_verdict.json`. One value per checkpoint
                  PER PROMPT, so nine values per point in the 410M pilot.
                  Drawn as a median with an interquartile band and the
                  individual prompts faint underneath. This is information
                  Phase 1's checkpoint figures don't have and it changes
                  what a knee means: a knee in the median with a tight band
                  is a property of the model, a knee that only appears in
                  two prompts is a property of those prompts.

The v-score caveat is enforced in the figure rather than left to the
reader. `verdict_v2.build_v_score` weights `frac_ffn_amplifies_repulsive`
at 0.20, and that term comes from `ffn_subspace`, which needs
`ffn_deltas.npz`, which `decompose.py` cannot produce for a
parallel-residual model — its hook dispatch matches "gpt2"/"bert" in the
model name and Pythia matches neither, so it returns empty delta lists
without raising. On a Pythia sweep the term is therefore identically zero
and the score is capped at 0.80. Any comparison against status-2.md's
GPT-2 v-score table is invalid until `core/sublayer_streams.py` is wired
in as the decompose provider. The v_score panel says so on its face.

transitions_p2_{base}.json is written in the same schema
`checkpoint_scalars.write_transitions` uses, so `select_snapshot_steps`
consumes it unchanged and the eigen-cloud filmstrips land on the same
checkpoints the scalars flagged.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.style import BLOG_STYLE
from core.naming import _safe_model_name
from p1_mstate_tracking.visualization.checkpoints import (
    _step_x, format_step_axis, TRANSITION_SPAN_COLOR, TRANSITION_SPAN_ALPHA,
)
from p1_mstate_tracking.visualization.checkpoint_scalars import detect_transitions

from .loaders import layer_field, verdict as _verdict
from .spectra import _non_normality, zone_counts

# ─────────────────────────────────────────────────────────────────────────────
# Weight-side scalar extractors: ov_summary dict -> float
# ─────────────────────────────────────────────────────────────────────────────

def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def _w_mean_rep_frac(s: dict) -> float:
    v = _finite(layer_field(s, "frac_repulsive"))
    return float(v.mean()) if v.size else float("nan")


def _w_max_rep_deviation(s: dict) -> float:
    """max_L |rep_frac(L) − 0.5|: departure from the untrained expectation
    at the single most-changed layer. Insensitive to depth structure
    cancelling in the mean, which `mean_rep_frac` is not — an early-
    repulsive / late-attractive model has a mean of exactly 0.5."""
    v = _finite(layer_field(s, "frac_repulsive"))
    return float(np.max(np.abs(v - 0.5))) if v.size else float("nan")


def _w_rep_depth_range(s: dict) -> float:
    """max_L rep_frac − min_L rep_frac: how much depth structure exists at
    all. Flat at init, grows if training differentiates layers."""
    v = _finite(layer_field(s, "frac_repulsive"))
    return float(v.max() - v.min()) if v.size else float("nan")


def _w_crossover_depth(s: dict) -> float:
    v = layer_field(s, "frac_repulsive")
    if not np.isfinite(v).any():
        return float("nan")
    z = zone_counts(v)
    c = z["crossover_layer"]
    return float("nan") if c is None else float(c) / max(v.size, 1)


def _w_frac_repulsive_layers(s: dict) -> float:
    v = layer_field(s, "frac_repulsive")
    if not np.isfinite(v).any():
        return float("nan")
    return zone_counts(v)["n_repulsive"] / max(v.size, 1)


def _w_mean_non_normality(s: dict) -> float:
    v = _finite(_non_normality(s))
    return float(v.mean()) if v.size else float("nan")


def _w_max_non_normality(s: dict) -> float:
    v = _finite(_non_normality(s))
    return float(v.max()) if v.size else float("nan")


def _w_mean_frac_complex(s: dict) -> float:
    v = _finite(layer_field(s, "frac_complex"))
    return float(v.mean()) if v.size else float("nan")


def _w_max_spectral_radius(s: dict) -> float:
    v = _finite(layer_field(s, "ov_spectral_radius"))
    return float(v.max()) if v.size else float("nan")


def _w_max_spectral_norm(s: dict) -> float:
    v = _finite(layer_field(s, "ov_spectral_norm"))
    return float(v.max()) if v.size else float("nan")


def _w_mean_qk_norm(s: dict) -> float:
    v = _finite(layer_field(s, "qk_spectral_norm_mean"))
    return float(v.mean()) if v.size else float("nan")


def _w_frac_methods_disagree(s: dict) -> float:
    v = _finite(layer_field(s, "methods_agree"))
    return float(1.0 - v.mean()) if v.size else float("nan")


WEIGHT_SCALARS: Dict[str, Tuple[Callable[[dict], float], str]] = {
    "mean_rep_frac":          (_w_mean_rep_frac,        "Mean repulsive fraction"),
    "max_rep_deviation":      (_w_max_rep_deviation,    "max |rep_frac − 0.5|"),
    "rep_depth_range":        (_w_rep_depth_range,      "rep_frac depth range"),
    "crossover_depth":        (_w_crossover_depth,      "Crossover layer / depth"),
    "frac_repulsive_layers":  (_w_frac_repulsive_layers, "Fraction of repulsive layers"),
    "mean_non_normality":     (_w_mean_non_normality,   "Mean ‖OV‖₂/ρ(OV)"),
    "max_non_normality":      (_w_max_non_normality,    "Max ‖OV‖₂/ρ(OV)"),
    "mean_frac_complex":      (_w_mean_frac_complex,    "Mean complex fraction"),
    "max_spectral_radius":    (_w_max_spectral_radius,  "Max spectral radius"),
    "max_spectral_norm":      (_w_max_spectral_norm,    "Max spectral norm"),
    "mean_qk_norm":           (_w_mean_qk_norm,         "Mean QK spectral norm"),
    "frac_methods_disagree":  (_w_frac_methods_disagree, "Fraction of layers where\nSchur ≠ symmetric"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Run-side scalar extractors: verdict dict -> float
# ─────────────────────────────────────────────────────────────────────────────

def _v(verdict: dict, key: str) -> float:
    val = verdict.get(key)
    return float("nan") if val is None else float(val)


def _r_rescaled_frac(verdict: dict) -> float:
    """Fraction of violations the rescaled frame eliminates. Undefined
    rather than 1.0 when there are no violations — a checkpoint with
    nothing to explain should be a gap in the curve, not a perfect score."""
    n = _v(verdict, "beta1.0_n_violations")
    if not np.isfinite(n) or n <= 0:
        return float("nan")
    return float(np.clip(_v(verdict, "rescaled_improvement_beta1.0") / n, 0.0, 1.0))


def _r_zone_contrast(verdict: dict) -> float:
    """violation rate in the repulsive zone minus the attractive zone.
    Positive = violations concentrate where V is repulsive, which is
    layer_v_events' prediction 1."""
    return (_v(verdict, "violation_rate_repulsive_zone")
            - _v(verdict, "violation_rate_attractive_zone"))


RUN_SCALARS: Dict[str, Tuple[Callable[[dict], float], str]] = {
    "v_score":              (lambda v: _v(v, "v_score"),                     "V-score"),
    "n_violations":         (lambda v: _v(v, "beta1.0_n_violations"),        "Energy violations (β=1)"),
    "frac_repulsive_disp":  (lambda v: _v(v, "beta1.0_frac_repulsive"),      "Displacement test:\nfraction repulsive"),
    "frac_overshoot":       (lambda v: _v(v, "beta1.0_frac_overshoot"),      "Fraction overshoot"),
    "rescaled_frac":        (_r_rescaled_frac,                               "Violations removed by\nrescaled frame"),
    "zone_contrast":        (_r_zone_contrast,                               "Violation rate:\nrepulsive − attractive zone"),
    "rho_rep_vs_violations": (lambda v: _v(v, "rho_repulsive_vs_violations"), "ρ(rep_frac, violation)"),
    "rho_rep_vs_deltaE":    (lambda v: _v(v, "continuous_repfrac_vs_deltaE_rho"), "ρ(rep_frac, ΔE)"),
    "head_ov_fiedler_rho":  (lambda v: _v(v, "head_ov_fiedler_rho"),         "ρ(head rep_frac, head Fiedler)"),
    "ov_norm_partial_rho":  (lambda v: _v(v, "ov_norm_partial_rho"),         "Partial ρ(‖OV‖₂, violation)\ncontrolling rep_frac"),
}

# Panels whose reading changes when the FFN channel is unavailable.
_FFN_DEPENDENT = {"v_score"}


def ffn_channel_available(runs: Dict[Tuple[str, str], Path]) -> bool:
    """
    True if ANY run reports a nonzero `frac_ffn_amplifies_repulsive`.

    False on every parallel-residual sweep until the decompose provider is
    replaced — see the module docstring. Checked from data rather than
    assumed from the model name, so the annotation disappears by itself
    once the wiring lands.
    """
    for stem in runs.values():
        val = _verdict(stem).get("frac_ffn_amplifies_repulsive")
        if val:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Series construction
# ─────────────────────────────────────────────────────────────────────────────

def weight_scalar_series(
    summaries: Dict[str, dict], family: List[Tuple[int, str]], fn,
) -> Tuple[List[int], List[float]]:
    steps, vals = [], []
    for step, model in family:
        s = summaries.get(model)
        if s is None:
            continue
        try:
            vals.append(float(fn(s)))
        except Exception:
            vals.append(float("nan"))
        steps.append(step)
    return steps, vals


def run_scalar_series(
    runs: Dict[Tuple[str, str], Path], family: List[Tuple[int, str]],
    prompts: List[str], fn,
) -> Tuple[List[int], np.ndarray]:
    """
    (steps, values[n_steps, n_prompts]) with NaN for absent runs. Kept as
    the full matrix rather than pre-reduced so the caller can draw both
    the spread and the individual prompt lines from one read.
    """
    steps: List[int] = []
    rows: List[List[float]] = []
    for step, model in family:
        row = []
        for prompt in prompts:
            stem = runs.get((model, prompt))
            if stem is None:
                row.append(float("nan"))
                continue
            try:
                row.append(float(fn(_verdict(stem))))
            except Exception:
                row.append(float("nan"))
        if np.all(np.isnan(row)):
            continue
        steps.append(step)
        rows.append(row)
    return steps, (np.asarray(rows, dtype=float) if rows else np.empty((0, 0)))


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def _draw_scalar_axis(ax, steps, vals, ylabel, transition=None,
                      color="#2563EB") -> None:
    x = _step_x(steps)
    ax.plot(x, vals, marker="o", markersize=4.5, linewidth=1.8,
            color=color, zorder=3)
    if transition is not None and transition["normalized_jump"] > 0:
        lo, hi = _step_x([transition["step_lo"], transition["step_hi"]])
        ax.axvspan(lo, hi, color=TRANSITION_SPAN_COLOR,
                   alpha=TRANSITION_SPAN_ALPHA, zorder=0)
    format_step_axis(ax, steps)
    ax.set_ylabel(ylabel, fontsize=9)


def _draw_spread_axis(ax, steps, mat, ylabel, prompts, transition=None) -> None:
    x = _step_x(steps)
    with np.errstate(all="ignore"):
        med = np.nanmedian(mat, axis=1)
        q25 = np.nanpercentile(mat, 25, axis=1)
        q75 = np.nanpercentile(mat, 75, axis=1)
    for j in range(mat.shape[1]):
        ax.plot(x, mat[:, j], color="#9CA3AF", linewidth=0.7, alpha=0.55,
                zorder=2)
    ax.fill_between(x, q25, q75, color="#2563EB", alpha=0.18, linewidth=0,
                    zorder=3)
    ax.plot(x, med, marker="o", markersize=4.0, linewidth=1.9,
            color="#2563EB", zorder=4)
    if transition is not None and transition["normalized_jump"] > 0:
        lo, hi = _step_x([transition["step_lo"], transition["step_hi"]])
        ax.axvspan(lo, hi, color=TRANSITION_SPAN_COLOR,
                   alpha=TRANSITION_SPAN_ALPHA, zorder=0)
    format_step_axis(ax, steps)
    ax.set_ylabel(ylabel, fontsize=9)


def _grid(n_panels: int, ncol: int = 3):
    nrow = int(np.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow),
                             squeeze=False)
    return fig, axes, nrow, ncol


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_weight_scalar_grid(
    summaries: Dict[str, dict], out_dir: Path, base: str,
    family: List[Tuple[int, str]],
) -> Dict[str, Optional[dict]]:
    """Every WEIGHT_SCALAR vs. training step. Returns {name: transition}."""
    plt.rcParams.update(BLOG_STYLE)
    names = list(WEIGHT_SCALARS.keys())
    fig, axes, nrow, ncol = _grid(len(names))
    transitions: Dict[str, Optional[dict]] = {}
    plotted = False

    for i, name in enumerate(names):
        ax = axes[i // ncol][i % ncol]
        fn, ylabel = WEIGHT_SCALARS[name]
        steps, vals = weight_scalar_series(summaries, family, fn)
        if sum(np.isfinite(vals)) < 2:
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
        print(f"  ⚠  weight_scalars: nothing to plot for {base!r}")
        return transitions

    fig.suptitle(
        f"Eigenspectrum scalars vs. training step  ·  {base}  (weights only)\n"
        "shaded span = sharpest inter-checkpoint change",
        fontsize=12, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"p2_weight_scalars_{_safe_model_name(base)}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")
    return transitions


def plot_run_scalar_grid(
    runs: Dict[Tuple[str, str], Path], out_dir: Path, base: str,
    family: List[Tuple[int, str]], prompts: List[str],
    ffn_available: Optional[bool] = None,
) -> Dict[str, Optional[dict]]:
    """
    Every RUN_SCALAR vs. training step, median over prompts with an IQR
    band and the individual prompts drawn faint. Transitions are detected
    on the MEDIAN — a knee that only one prompt shows is not a transition
    of the model.
    """
    if ffn_available is None:
        ffn_available = ffn_channel_available(runs)

    plt.rcParams.update(BLOG_STYLE)
    names = list(RUN_SCALARS.keys())
    fig, axes, nrow, ncol = _grid(len(names))
    transitions: Dict[str, Optional[dict]] = {}
    plotted = False

    for i, name in enumerate(names):
        ax = axes[i // ncol][i % ncol]
        fn, ylabel = RUN_SCALARS[name]
        steps, mat = run_scalar_series(runs, family, prompts, fn)
        if mat.size == 0 or len(steps) < 2:
            ax.axis("off")
            transitions[name] = None
            continue
        with np.errstate(all="ignore"):
            med = np.nanmedian(mat, axis=1)
        if np.sum(np.isfinite(med)) < 2:
            ax.axis("off")
            transitions[name] = None
            continue
        tr = detect_transitions(steps, list(med))
        transitions[name] = tr
        _draw_spread_axis(ax, steps, mat, ylabel, prompts, transition=tr)
        title = name
        if name in _FFN_DEPENDENT and not ffn_available:
            title += "  ⚠ FFN term unavailable"
            ax.text(0.02, 0.03,
                    "no ffn_subspace on this architecture:\n"
                    "0.20·frac_ffn_amplifies_repulsive ≡ 0, score capped at 0.80",
                    transform=ax.transAxes, fontsize=6.5, color="#B45309",
                    va="bottom", ha="left")
        ax.set_title(title, fontsize=10)
        plotted = True

    for i in range(len(names), nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")

    if not plotted:
        plt.close(fig)
        print(f"  ⚠  run_scalars: nothing to plot for {base!r}")
        return transitions

    fig.suptitle(
        f"Verdict scalars vs. training step  ·  {base}  ·  "
        f"median over {len(prompts)} prompts\n"
        "band = interquartile range across prompts · gray = individual prompts · "
        "shaded span = sharpest change in the median",
        fontsize=12, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"p2_run_scalars_{_safe_model_name(base)}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")
    return transitions


# ─────────────────────────────────────────────────────────────────────────────
# transitions_p2_{base}.json
# ─────────────────────────────────────────────────────────────────────────────

def write_transitions(
    out_dir: Path, base: str, transitions: Dict[str, Optional[dict]],
    n_prompts: Optional[int] = None,
) -> Path:
    """
    Same schema as `checkpoint_scalars.write_transitions`, so
    `select_snapshot_steps` reads it unchanged. Written per BASE MODEL, not
    per prompt: the weight scalars have no prompt, and the run scalars are
    already reduced over prompts before detection.
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
        "phase": "p2_eigenspectra",
        "n_prompts_reduced": n_prompts,
        "per_metric": dict(transitions),
        "ranked_by_jump": [{"metric": k, **v} for k, v in ranked],
        "consensus_intervals": dict(
            sorted(consensus.items(), key=lambda kv: -kv[1])
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"transitions_p2_{_safe_model_name(base)}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  ✓  {path.name}")
    return path


def generate_scalar_figures(
    summaries: Dict[str, dict], runs: Dict[Tuple[str, str], Path],
    out_dir: Path, base: str, family: List[Tuple[int, str]],
    prompts: Optional[List[str]] = None,
) -> Dict[str, Optional[dict]]:
    """
    Both scalar grids plus transitions_p2_{base}.json. Returns the merged
    transitions dict — the eigen-cloud filmstrips consume it to pick their
    snapshot steps, which is why this runs first in the pipeline.
    """
    if len(family) < 3:
        print(f"  ⚠  p2 scalars: family {base!r} has <3 checkpoints, skipping")
        return {}
    transitions = plot_weight_scalar_grid(summaries, out_dir, base, family)
    if prompts is None:
        prompts = sorted({p for (_, p) in runs})
    if runs and prompts:
        run_tr = plot_run_scalar_grid(runs, out_dir, base, family, prompts)
        transitions.update({f"run__{k}": v for k, v in run_tr.items()})
    write_transitions(out_dir, base, transitions, n_prompts=len(prompts or []))
    return transitions
