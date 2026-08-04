"""
p2_eigenspectra/visualization/spectra.py

Class-W figures: the OV eigenspectrum as a developmental object. Every
metric here is a function of WEIGHTS ONLY, read out of `ov_summary_*.json`,
so these figures exist at 27 checkpoints rather than 27 × 9 runs and carry
no prompt dependence to average over or argue about.

That independence is the point. Phase 2's activation-side tests inherit
every ambiguity of the prompt, the tokenizer, and Phase 1's violation
criterion. The eigenspectrum inherits none of them: whatever
`frac_repulsive` does over training, it did to the weights, and a prompt
cannot explain it away.

Reference values, which is why several of these plot against a null line
rather than against zero:

  frac_repulsive       At step 0 the composed OV is a sum over heads of
                       products of two independent Gaussian blocks. Its
                       spectrum is symmetric about the imaginary axis, so
                       the expected fraction with Re λ < 0 is 0.5, flat
                       across depth, with O(d^-1/2) scatter. Every
                       departure from the 0.5 line is training. Drawn on a
                       diverging colormap CENTERED at 0.5 for the same
                       reason — "which side of chance" is the readable
                       quantity, not the absolute value.

  frac_complex         The opposite intuition from the one people expect.
                       A real Ginibre matrix has O(sqrt(d)) real
                       eigenvalues out of d, so frac_complex starts near 1
                       and can only fall. A DECREASE is the signal:
                       eigenvalues condensing onto the real axis means the
                       learned map is becoming closer to symmetric.

  non_normality        ‖OV‖₂ / ρ(OV), spectral norm over spectral radius.
                       Exactly 1 for a normal matrix and > 1 otherwise, so
                       the null line is 1.0. Both terms are already in the
                       summary: `_build_summary` records the true spectral
                       norm via svdvals and the radius separately,
                       precisely because they diverge on non-normal layers.
                       This is the quantity Phase 2b's rotational analysis
                       is about, measured here for free.

  methods_agree        Whether the Schur and symmetric-part decompositions
                       agree on the sign split within 10%. As a step×layer
                       map it says WHERE the antisymmetric part is large
                       enough to matter for the attractive/repulsive
                       classification — i.e. where Phase 2's projectors are
                       method-dependent and the verdict should be read with
                       care.

Zone bands call `layer_v_events.classify_layers` directly rather than
reimplementing its 0.5 ± 0.05 thresholds. Those thresholds are analysis
logic and live in the phase package; a local copy here would silently
disagree with the verdict reading the same numbers. This package holds
figure logic only and imports everything else.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.style import BLOG_STYLE
from core.naming import _safe_model_name
from p1_mstate_tracking.visualization.checkpoints import (
    STEP0_STYLE, RANDOM_BASELINE_STYLE,
    step_norm, step_color, add_step_colorbar, family_baselines,
    checkpoint_families, _fmt_step,
)

from .loaders import layer_field, n_layers

# ─────────────────────────────────────────────────────────────────────────────
# Derived per-layer quantities
# ─────────────────────────────────────────────────────────────────────────────

def _non_normality(summary: dict) -> np.ndarray:
    """‖OV‖₂ / ρ(OV) per layer. 1.0 iff normal; larger = more rotational."""
    norm = layer_field(summary, "ov_spectral_norm")
    radius = layer_field(summary, "ov_spectral_radius")
    with np.errstate(divide="ignore", invalid="ignore"):
        return norm / np.where(np.abs(radius) < 1e-30, np.nan, radius)


def _method_gap(summary: dict) -> np.ndarray:
    """sym_frac_repulsive − frac_repulsive: how much discarding the
    antisymmetric part moves the sign split. Zero when rotation is
    irrelevant to the classification."""
    return (layer_field(summary, "sym_frac_repulsive")
            - layer_field(summary, "frac_repulsive"))


def _dim_repulse_frac(summary: dict) -> np.ndarray:
    """Schur repulsive invariant-subspace dimension as a fraction of d.
    Tracks frac_repulsive closely by construction — a divergence between
    the two means the eigenvalue count and the invariant-subspace
    dimension disagree, i.e. a near-degenerate 2×2 block on the axis."""
    a = layer_field(summary, "schur_dim_attract")
    r = layer_field(summary, "schur_dim_repulse")
    with np.errstate(divide="ignore", invalid="ignore"):
        return r / (a + r)


def _log_schur_cond(summary: dict) -> np.ndarray:
    """log10 of the Schur block-norm ratio. Raw values span decades and
    are `inf` whenever one subspace is empty, which no linear axis
    survives."""
    c = layer_field(summary, "schur_cond")
    c = np.where(np.isfinite(c) & (c > 0), c, np.nan)
    return np.log10(c)


def _agree(summary: dict) -> np.ndarray:
    return layer_field(summary, "methods_agree")


def _field(name: str) -> Callable[[dict], np.ndarray]:
    return lambda s: layer_field(s, name)


# ─────────────────────────────────────────────────────────────────────────────
# Metric registry — single source of truth for the weight-side figures.
# One entry adds the metric to the sweep AND the heatmap, the same way
# p1's CHECKPOINT_METRICS does for the activation-side ones.
#
#   null   : reference value drawn as a horizontal line on sweeps and used
#            as the diverging-colormap center on heatmaps. None = no
#            meaningful reference, sequential colormap.
#   cmap   : heatmap colormap. Diverging ones are only correct with a null.
# ─────────────────────────────────────────────────────────────────────────────

WEIGHT_METRICS: Dict[str, dict] = {
    "rep_frac": dict(
        fn=_field("frac_repulsive"),
        title="Repulsive eigenvalue fraction",
        ylabel="fraction of eigenvalues with Re λ < 0",
        ylim=(0.0, 1.0), null=0.5, cmap="coolwarm",
        note="0.5 = the untrained expectation for a product of Gaussian blocks",
    ),
    "sym_rep_frac": dict(
        fn=_field("sym_frac_repulsive"),
        title="Repulsive fraction (symmetric part)",
        ylabel="fraction of eig((OV+OVᵀ)/2) < 0",
        ylim=(0.0, 1.0), null=0.5, cmap="coolwarm",
        note="rotation-free reading of the same split",
    ),
    "method_gap": dict(
        fn=_method_gap,
        title="Schur vs. symmetric-part disagreement",
        ylabel="sym_frac_repulsive − frac_repulsive",
        ylim=None, null=0.0, cmap="PuOr",
        note="nonzero = the antisymmetric part changes the sign classification",
    ),
    "frac_complex": dict(
        fn=_field("frac_complex"),
        title="Complex eigenvalue fraction",
        ylabel="fraction with |Im λ| > 0.01·|Re λ|",
        ylim=(0.0, 1.05), null=None, cmap="magma",
        note="starts near 1 at init; a fall = condensation onto the real axis",
    ),
    "non_normality": dict(
        fn=_non_normality,
        title="Departure from normality",
        ylabel="‖OV‖₂ / ρ(OV)",
        ylim=None, null=1.0, cmap="viridis",
        note="1.0 = normal matrix; growth = learned rotational structure",
    ),
    "ov_spectral_norm": dict(
        fn=_field("ov_spectral_norm"),
        title="OV spectral norm",
        ylabel="‖OV‖₂ (largest singular value)",
        ylim=None, null=None, cmap="magma",
        note="the confound Phase 2 controls for; watch early/final layers",
    ),
    "ov_spectral_radius": dict(
        fn=_field("ov_spectral_radius"),
        title="OV spectral radius",
        ylabel="max |λ|",
        ylim=None, null=None, cmap="magma",
    ),
    "qk_norm_mean": dict(
        fn=_field("qk_spectral_norm_mean"),
        title="QK spectral norm (head mean)",
        ylabel="mean_h ‖W_Q^T W_K‖₂",
        ylim=None, null=None, cmap="cividis",
        note="the β-like scale in the rep_frac × QK detection threshold",
    ),
    "dim_repulse_frac": dict(
        fn=_dim_repulse_frac,
        title="Repulsive invariant-subspace dimension",
        ylabel="schur_dim_repulse / d_model",
        ylim=(0.0, 1.0), null=0.5, cmap="coolwarm",
    ),
    "log_schur_cond": dict(
        fn=_log_schur_cond,
        title="Schur block-norm ratio",
        ylabel="log₁₀(‖T₊₊‖ / ‖T₋₋‖)",
        ylim=None, null=0.0, cmap="PuOr",
    ),
    "methods_agree": dict(
        fn=_agree,
        title="Decomposition methods agree",
        ylabel="1 = Schur and symmetric agree within 10%",
        ylim=(-0.05, 1.05), null=None, cmap="RdYlGn",
        note="where this is 0, the projectors are method-dependent",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Family construction from weight summaries
# ─────────────────────────────────────────────────────────────────────────────

def weight_families(summaries: Dict[str, dict]) -> Dict[str, List[Tuple[int, str]]]:
    """{base: [(step, model), ...]} over checkpoints that have a summary."""
    return checkpoint_families(list(summaries.keys()))


def _profile(summaries: Dict[str, dict], model: str, fn) -> Optional[np.ndarray]:
    s = summaries.get(model)
    if s is None or n_layers(s) == 0:
        return None
    try:
        arr = np.asarray(fn(s), dtype=float)
    except Exception:
        return None
    return arr if arr.size and not np.all(np.isnan(arr)) else None


def _matrix(
    summaries: Dict[str, dict], family: List[Tuple[int, str]], fn,
) -> Tuple[np.ndarray, List[int]]:
    """(n_checkpoints, max_layers) NaN-padded, rows in ascending step order."""
    rows, steps = [], []
    for step, model in family:
        prof = _profile(summaries, model, fn)
        if prof is None:
            continue
        rows.append(prof)
        steps.append(step)
    if not rows:
        return np.empty((0, 0)), []
    width = max(r.size for r in rows)
    mat = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        mat[i, : r.size] = r
    return mat, steps


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_weight_sweep(
    summaries: Dict[str, dict], out_dir: Path, base: str,
    family: List[Tuple[int, str]], metric_name: str,
) -> None:
    """One metric's depth profile for every checkpoint, colored by log step."""
    spec = WEIGHT_METRICS[metric_name]
    baselines = family_baselines(base, list(summaries.keys()))

    plt.rcParams.update(BLOG_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    steps = [s for s, _ in family]
    norm = step_norm(steps)
    plotted = False

    for step, model in family:
        prof = _profile(summaries, model, spec["fn"])
        if prof is None:
            continue
        x = np.linspace(0, 1, prof.size)
        if step == 0:
            ax.plot(x, prof, **STEP0_STYLE, zorder=4, label="step 0 (init)")
        else:
            ax.plot(x, prof, color=step_color(step, norm),
                    linewidth=1.8, alpha=0.85, zorder=3)
        plotted = True

    rand = baselines["random"]
    if rand is not None:
        prof_r = _profile(summaries, rand, spec["fn"])
        if prof_r is not None:
            ax.plot(np.linspace(0, 1, prof_r.size), prof_r,
                    **RANDOM_BASELINE_STYLE, zorder=4,
                    label="norm-matched random")
            plotted = True

    if not plotted:
        plt.close(fig)
        print(f"  ⚠  spectra_sweep_{metric_name}: no summaries for {base!r}")
        return

    if spec.get("null") is not None:
        ax.axhline(spec["null"], color="#6B7280", linestyle="-.",
                   linewidth=1.2, zorder=2,
                   label=f"null = {spec['null']:g}")

    add_step_colorbar(fig, ax, steps, norm)
    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel(spec["ylabel"])
    if spec.get("ylim") is not None:
        ax.set_ylim(*spec["ylim"])
    ax.set_xlim(0, 1)
    subtitle = spec.get("note", "")
    ax.set_title(
        f"{spec['title']} vs. depth across training  ·  {base}  (weights only)"
        + (f"\n{subtitle}" if subtitle else ""),
        fontsize=12, fontweight="bold",
    )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="best")

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"spectra_sweep_{metric_name}_{_safe_model_name(base)}.png"
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


def plot_weight_heatmap(
    summaries: Dict[str, dict], out_dir: Path, base: str,
    family: List[Tuple[int, str]], metric_name: str,
) -> None:
    """
    layer × checkpoint heatmap. Rows are ORDINAL (one per checkpoint), the
    same convention as p1's checkpoint_heatmaps and for the same reason:
    the comparison the pilot adjudicates is between ADJACENT checkpoints,
    so adjacent rows must be one row apart regardless of step spacing.

    Metrics with a null get a diverging colormap symmetric about it, so
    "above chance" and "below chance" are visually distinct rather than
    two shades of the same ramp.
    """
    spec = WEIGHT_METRICS[metric_name]
    mat, steps = _matrix(summaries, family, spec["fn"])
    if mat.size == 0 or len(steps) < 2:
        print(f"  ⚠  spectra_heatmap_{metric_name}: <2 checkpoints for {base!r}")
        return

    plt.rcParams.update(BLOG_STYLE)
    fig, ax = plt.subplots(figsize=(9.5, max(4.2, 0.28 * len(steps) + 1.8)))

    masked = np.ma.masked_invalid(mat)
    vmin = vmax = None
    if spec.get("null") is not None and masked.count():
        span = float(np.nanmax(np.abs(mat - spec["null"])))
        if np.isfinite(span) and span > 0:
            vmin, vmax = spec["null"] - span, spec["null"] + span

    pc = ax.pcolormesh(
        np.arange(mat.shape[1] + 1) - 0.5,
        np.arange(mat.shape[0] + 1) - 0.5,
        masked, cmap=spec["cmap"], shading="flat", vmin=vmin, vmax=vmax,
    )
    cbar = plt.colorbar(pc, ax=ax, pad=0.02)
    cbar.set_label(spec["ylabel"], fontsize=9)
    ax.set_yticks(np.arange(len(steps)))
    ax.set_yticklabels([_fmt_step(s) for s in steps], fontsize=7)
    ax.set_ylabel("Training step  (one row per checkpoint)")
    ax.set_xlabel("Layer")
    ax.invert_yaxis()

    subtitle = spec.get("note", "a transition = a horizontal band where "
                                "adjacent rows change character")
    ax.set_title(
        f"{spec['title']}: layer × training step  ·  {base}  (weights only)"
        f"\n{subtitle}",
        fontsize=11, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"spectra_heatmap_{metric_name}_{_safe_model_name(base)}.png"
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Zone composition
# ─────────────────────────────────────────────────────────────────────────────

def zone_counts(rep_frac: np.ndarray) -> dict:
    """
    Thin adapter over `layer_v_events.classify_layers`.

    Imported, not reimplemented: the zone thresholds are analysis logic and
    belong in the phase package, so a change there must move these figures
    with it. The adapter exists only to accept a bare array where
    classify_layers wants the v_profile dict it gets from
    build_v_profile_from_ov.
    """
    from p2_eigenspectra.layer_v_events import classify_layers

    return classify_layers({"repulsive_frac": np.asarray(rep_frac, dtype=float)})


def zone_series(
    summaries: Dict[str, dict], family: List[Tuple[int, str]],
) -> Tuple[List[int], Dict[str, List[float]], List[float]]:
    """
    Per checkpoint: the fraction of layers in each of the three zones,
    plus the crossover layer as a fraction of depth (NaN when there is no
    repulsive zone to cross out of).
    """
    steps: List[int] = []
    fracs: Dict[str, List[float]] = {"repulsive": [], "transition": [], "attractive": []}
    crossover: List[float] = []

    for step, model in family:
        prof = _profile(summaries, model, WEIGHT_METRICS["rep_frac"]["fn"])
        if prof is None:
            continue
        z = zone_counts(prof)
        n = max(prof.size, 1)
        steps.append(step)
        fracs["repulsive"].append(z["n_repulsive"] / n)
        fracs["transition"].append(z["n_transition"] / n)
        fracs["attractive"].append(z["n_attractive"] / n)
        c = z["crossover_layer"]
        crossover.append(float("nan") if c is None else c / n)
    return steps, fracs, crossover


def plot_zone_bands(
    summaries: Dict[str, dict], out_dir: Path, base: str,
    family: List[Tuple[int, str]],
) -> None:
    """
    The depth partition into repulsive / transition / attractive zones, as
    a stacked area over training, with the crossover depth overlaid.

    This is the figure that answers "does the model's repulsive region
    move, grow, or appear" in one panel. At init all three bands should be
    near their chance composition and the crossover should be undefined or
    unstable; a stable crossover that migrates and then locks is the
    developmental claim.
    """
    from p1_mstate_tracking.visualization.checkpoints import (
        _step_x, format_step_axis,
    )

    steps, fracs, crossover = zone_series(summaries, family)
    if len(steps) < 2:
        print(f"  ⚠  zone_bands: <2 checkpoints for {base!r}")
        return

    plt.rcParams.update(BLOG_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5.0))
    x = _step_x(steps)

    ax.stackplot(
        x, fracs["repulsive"], fracs["transition"], fracs["attractive"],
        labels=["repulsive (rep_frac > 0.55)",
                "transition (0.45–0.55)",
                "attractive (rep_frac < 0.45)"],
        colors=["#DC2626", "#E5E7EB", "#2563EB"], alpha=0.85,
    )

    ax2 = ax.twinx()
    ax2.plot(x, crossover, color="#111827", marker="o", markersize=4,
             linewidth=1.6, linestyle="--", label="crossover depth")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Crossover layer / depth")

    format_step_axis(ax, steps)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of layers in zone")
    ax.set_title(
        f"Depth partition into V-repulsive and V-attractive zones  ·  {base}\n"
        "thresholds are layer_v_events.classify_layers' 0.5 ± 0.05",
        fontsize=12, fontweight="bold",
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"zone_bands_{_safe_model_name(base)}.png"
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_spectra_figures(
    summaries: Dict[str, dict], out_dir: Path, base: str,
    family: List[Tuple[int, str]],
    metrics: Optional[List[str]] = None,
) -> None:
    """Every weight-side figure for one checkpoint family."""
    if len(family) < 2:
        print(f"  ⚠  spectra: family {base!r} has <2 checkpoints, skipping")
        return
    names = metrics if metrics is not None else list(WEIGHT_METRICS.keys())
    for name in names:
        if name not in WEIGHT_METRICS:
            print(f"  ⚠  unknown weight metric {name!r}, skipping")
            continue
        plot_weight_heatmap(summaries, out_dir, base, family, name)
        plot_weight_sweep(summaries, out_dir, base, family, name)
    plot_zone_bands(summaries, out_dir, base, family)
