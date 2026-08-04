"""
visualization/energy_attribution_aggregate.py

Multi-prompt counterpart to energy_decomposition.py's single-prompt
figures: aggregates per-layer attn/ffn/cross energy-attribution fraction
and raw delta norm across every prompt available for a model (mean ± std
per layer), and the same for one hand-picked random-init control, then
plots both as a mean ± std band. A series that's tight across prompts is
a property of the model; a series that's wide is prompt-dependent, and any
single-prompt figure of it should be read with that in mind.

The random side is deliberately NOT auto-aggregated across every seed
under a p1_random tree the way random_aggregate.py aggregates seeds for
the overview/pair figures — here it's exactly one hand-picked run
(random_run_dir), so a known-degenerate seed can be avoided, aggregated
only across that run's own prompts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.style import BLOG_STYLE
from core.naming import _color, _random_color, _safe_model_name
from .loaders import discover_runs
from .energy_decomposition import _phase2_decomposed, _find_phase2_run_dir, _per_layer_energy_fracs


# ─────────────────────────────────────────────────────────────────────────────
# Magnitude (norm) companion to the per-layer energy fraction series
# ─────────────────────────────────────────────────────────────────────────────

def _per_layer_energy_norms(decomp: dict) -> dict:
    """
    Per-layer mean L2 norm (averaged over tokens) of the raw attn and FFN
    residual-stream deltas. Uses the raw/unnormalized arrays — same
    convention as decompose.py's save_decomposed: scale carries information
    here, so this reads attn_deltas_raw.npz / ffn_deltas_raw.npz via the
    same `decomp` dict _phase2_decomposed already loads, not the
    direction-only normed copies.
    """
    n_layers = len(decomp["attn_deltas"])
    out = {"attn_norm": [], "ffn_norm": []}
    for i in range(n_layers):
        a = decomp["attn_deltas"][i]
        f = decomp["ffn_deltas"][i]
        out["attn_norm"].append(float(np.linalg.norm(a, axis=-1).mean()))
        out["ffn_norm"].append(float(np.linalg.norm(f, axis=-1).mean()))
    return out


def _energy_attribution_per_prompt(
    run_dir: Path,
    phase2_run_dir: Path,
    beta: float,
) -> Optional[dict]:
    """
    One prompt's full per-layer energy-attribution profile: attn/ffn/cross
    fraction (_per_layer_energy_fracs) plus attn/ffn raw delta norm
    (_per_layer_energy_norms). Returns None if Phase 2 data is missing.
    """
    decomp = _phase2_decomposed(phase2_run_dir)
    if decomp is None:
        return None
    fracs, n_layers = _per_layer_energy_fracs(decomp, beta)
    if n_layers == 0:
        return None
    norms = _per_layer_energy_norms(decomp)
    return {
        "n_layers":  n_layers,
        "attn_frac": np.asarray(fracs["attn"]),
        "ffn_frac":  np.asarray(fracs["ffn"]),
        "cross_frac": np.asarray(fracs["cross"]),
        "attn_norm": np.asarray(norms["attn_norm"][:n_layers]),
        "ffn_norm":  np.asarray(norms["ffn_norm"][:n_layers]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-prompt aggregation
# ─────────────────────────────────────────────────────────────────────────────

ENERGY_ATTRIBUTION_SERIES: Tuple[str, ...] = (
    "attn_frac", "ffn_frac", "cross_frac", "attn_norm", "ffn_norm",
)


def aggregate_energy_attribution_across_prompts(
    runs: Dict[Tuple[str, str], Path],
    phase2_dir: Path,
    model: str,
    prompts: Optional[List[str]] = None,
    beta: float = 1.0,
) -> Optional[dict]:
    """
    Aggregate per-layer energy attribution over every prompt found for
    `model` in `runs` that also has a matching Phase 2 decompose run under
    phase2_dir — mean and std across prompts, per layer, per series in
    ENERGY_ATTRIBUTION_SERIES. This is the multi-prompt counterpart to the
    single-prompt (DEFAULT_PROMPT-only) energy_decomposition figure.

    `prompts`, if given, restricts to that subset (still requires a Phase 2
    match for each). Leaving it None uses every prompt `runs` has for `model`.

    Returns None if no prompt for `model` has a matching Phase 2 run.
    """
    available_prompts = sorted({p for (m, p) in runs.keys() if m == model})
    if prompts is not None:
        available_prompts = [p for p in available_prompts if p in prompts]

    per_prompt: Dict[str, dict] = {}
    for prompt in available_prompts:
        run_dir = runs[(model, prompt)]
        phase2_run_dir = _find_phase2_run_dir(phase2_dir, model, prompt)
        if phase2_run_dir is None:
            continue
        profile = _energy_attribution_per_prompt(run_dir, phase2_run_dir, beta)
        if profile is None:
            continue
        per_prompt[prompt] = profile

    if not per_prompt:
        return None

    n_layers = min(p["n_layers"] for p in per_prompt.values())
    stacked = {
        s: np.stack([p[s][:n_layers] for p in per_prompt.values()])
        for s in ENERGY_ATTRIBUTION_SERIES
    }  # each: (n_prompts, n_layers)

    out: dict = {
        "model":     model,
        "beta":      beta,
        "n_prompts": len(per_prompt),
        "prompts":   sorted(per_prompt.keys()),
        "n_layers":  n_layers,
    }
    for s in ENERGY_ATTRIBUTION_SERIES:
        out[s] = {
            "mean": stacked[s].mean(axis=0),
            "std":  stacked[s].std(axis=0),
            "values_per_prompt": stacked[s],  # (n_prompts, n_layers) — kept for inspection
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot: trained (every prompt) vs one hand-picked random run (every prompt)
# ─────────────────────────────────────────────────────────────────────────────

def plot_energy_attribution_prompt_aggregate(
    trained_agg: dict,
    random_agg: Optional[dict],
    out_dir: Path,
    series: str = "ffn_frac",
) -> bool:
    """
    Mean ± std across prompts, per layer, for one energy-attribution series
    — trained model vs. one hand-picked random-init control, both
    aggregated the same way (aggregate_energy_attribution_across_prompts).
    The shaded band is prompt-to-prompt variance at that layer: a narrow
    trained band next to a wide (or absent) random band is the signal that
    the pattern is a trained-weight property, not something that would show
    up at random init too.
    """
    if series not in ENERGY_ATTRIBUTION_SERIES or series not in trained_agg:
        return False

    plt.rcParams.update(BLOG_STYLE)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    x = np.arange(trained_agg["n_layers"])
    mean_t, std_t = trained_agg[series]["mean"], trained_agg[series]["std"]
    ax.plot(x, mean_t, color=_color(trained_agg["model"]), lw=2,
             label=f"{trained_agg['model']}  (n={trained_agg['n_prompts']} prompts)")
    ax.fill_between(x, mean_t - std_t, mean_t + std_t,
                     color=_color(trained_agg["model"]), alpha=0.2)

    if random_agg is not None and series in random_agg:
        xr = np.arange(random_agg["n_layers"])
        mean_r, std_r = random_agg[series]["mean"], random_agg[series]["std"]
        ax.plot(xr, mean_r, color=_random_color(random_agg["model"]), lw=2, linestyle="--",
                 label=f"{random_agg['model']}  (n={random_agg['n_prompts']} prompts)")
        ax.fill_between(xr, mean_r - std_r, mean_r + std_r,
                         color=_random_color(random_agg["model"]), alpha=0.2)

    if series.endswith("_frac"):
        ax.set_yscale("log")
        ax.set_ylabel(f"{series}  (share of |ΔE_β|, log scale)")
        ax.axhline(1.0, color="black", lw=0.6, alpha=0.4)
    else:
        ax.set_ylabel(f"{series}  (mean per-token L2 norm)")

    ax.set_xlabel("Layer")
    ax.set_title(
        f"{series} across prompts (β={trained_agg['beta']:g}) — mean ± std band over prompts",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="best")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{_safe_model_name(trained_agg['model'])}_{series}_prompt_aggregate.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname.name}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Combined figure: all five series on one chart with twin y-axes
# ─────────────────────────────────────────────────────────────────────────────

# Fixed per-series colors, independent of model — the series identity is the
# primary visual dimension here, not which model is being compared.
_COMBINED_COLORS: Dict[str, str] = {
    "attn_frac":  "#2563EB",  # blue
    "ffn_frac":   "#059669",  # green
    "cross_frac": "#7C3AED",  # purple
    "attn_norm":  "#EA580C",  # orange
    "ffn_norm":   "#DC2626",  # red
}
_COMBINED_LABELS: Dict[str, str] = {
    "attn_frac":  "attn fraction",
    "ffn_frac":   "FFN fraction",
    "cross_frac": "cross fraction",
    "attn_norm":  "attn norm",
    "ffn_norm":   "FFN norm",
}
_FRAC_SERIES = ("attn_frac", "ffn_frac", "cross_frac")
_NORM_SERIES = ("attn_norm", "ffn_norm")


def plot_energy_attribution_combined(
    trained_agg: dict,
    random_agg: Optional[dict],
    out_dir: Path,
) -> bool:
    """
    All five energy-attribution series in one figure with twin y-axes:
      left  — attn_frac / ffn_frac / cross_frac  (share of |ΔE_β|, linear)
      right — attn_norm / ffn_norm                (mean per-token L2 norm)

    Each series is a mean line with ±1σ fill across prompts. Trained model
    uses solid lines; random control (if given) uses dashed lines of the
    same colors. Both y-axes drop slightly below zero so a series that goes
    flat at zero reads clearly at the floor rather than sitting on the edge.
    """
    if not any(s in trained_agg for s in (*_FRAC_SERIES, *_NORM_SERIES)):
        return False

    plt.rcParams.update(BLOG_STYLE)
    fig, ax_l = plt.subplots(figsize=(12, 5.5))
    ax_r = ax_l.twinx()

    def _draw_agg(agg: dict, linestyle: str, alpha_band: float, suffix: str) -> None:
        x = np.arange(agg["n_layers"])
        for s in _FRAC_SERIES:
            if s not in agg:
                continue
            c    = _COMBINED_COLORS[s]
            mean = np.asarray(agg[s]["mean"], dtype=float)
            std  = np.asarray(agg[s]["std"],  dtype=float)
            ax_l.plot(x, mean, color=c, lw=2.0, linestyle=linestyle,
                      label=f"{_COMBINED_LABELS[s]}{suffix}")
            ax_l.fill_between(x, mean - std, mean + std,
                              color=c, alpha=alpha_band, linewidth=0)
        for s in _NORM_SERIES:
            if s not in agg:
                continue
            c    = _COMBINED_COLORS[s]
            mean = np.asarray(agg[s]["mean"], dtype=float)
            std  = np.asarray(agg[s]["std"],  dtype=float)
            ax_r.plot(x, mean, color=c, lw=2.0, linestyle=linestyle,
                      label=f"{_COMBINED_LABELS[s]}{suffix}")
            ax_r.fill_between(x, mean - std, mean + std,
                              color=c, alpha=alpha_band, linewidth=0)

    n_t = trained_agg.get("n_prompts", 1)
    _draw_agg(trained_agg, linestyle="-", alpha_band=0.15, suffix=f"  (n={n_t})")
    if random_agg is not None and any(
        s in random_agg for s in (*_FRAC_SERIES, *_NORM_SERIES)
    ):
        n_r = random_agg.get("n_prompts", 1)
        _draw_agg(random_agg, linestyle="--", alpha_band=0.10,
                  suffix=f"  random (n={n_r})")

    # Drop both y-axes slightly below zero so zero-valued series sit visibly
    # at the floor rather than disappearing into the axis edge.
    for ax in (ax_l, ax_r):
        lo, hi = ax.get_ylim()
        pad = 0.04 * max(abs(hi), 1e-6)
        ax.set_ylim(-pad, hi + 0.5 * pad)

    ax_l.set_xlabel("Layer")
    ax_l.set_ylabel("Share of |ΔE_β|  (fraction, linear)")
    ax_r.set_ylabel("Mean per-token L2 norm")
    ax_l.set_title(
        f"{trained_agg['model']}  —  energy attribution, all series  "
        f"(β={trained_agg['beta']:g})\n"
        "solid = trained · dashed = random · shaded band = ±1σ across prompts",
        fontsize=10, fontweight="bold",
    )

    # Merge legends from both axes into one box in the upper right.
    h_l, lbl_l = ax_l.get_legend_handles_labels()
    h_r, lbl_r = ax_r.get_legend_handles_labels()
    ax_l.legend(
        h_l + h_r, lbl_l + lbl_r,
        fontsize=8, loc="upper right", ncol=2, framealpha=0.92,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / (
        f"{_safe_model_name(trained_agg['model'])}_energy_attribution_combined.png"
    )
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname.name}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper: discover the random run, aggregate both sides, plot every series
# ─────────────────────────────────────────────────────────────────────────────

def generate_energy_attribution_prompt_aggregate_figures(
    runs: Dict[Tuple[str, str], Path],
    phase2_dir: Optional[Path],
    out_dir: Path,
    model: str,
    random_run_dir: Optional[Path] = None,
    random_model: Optional[str] = None,
    random_phase2_dir: Optional[Path] = None,
    beta: float = 1.0,
    series: Tuple[str, ...] = ENERGY_ATTRIBUTION_SERIES,
) -> None:
    """
    Multi-prompt counterpart to generate_energy_decomposition_figures.
    Aggregates over every prompt `runs` has for `model` (mean ± std per
    layer) and, if random_run_dir is given, does the same over every prompt
    found in that one hand-picked p1_random seed directory for
    random_model — not auto-aggregated across seeds; exactly the run you
    point it at. Plots both as a mean ± std band per series, plus one
    combined figure with all five series on twin axes.

    random_phase2_dir defaults to phase2_dir — only pass it separately if
    the random model's Phase 2 decompose runs live in a different tree than
    the trained ones.
    """
    if phase2_dir is None:
        print(f"  ⚠  no phase2_dir given — skipping prompt-aggregate energy attribution for {model}")
        return

    trained_agg = aggregate_energy_attribution_across_prompts(runs, phase2_dir, model, beta=beta)
    if trained_agg is None:
        print(f"  ⚠  no Phase 2 runs found for any prompt of {model} — "
              f"skipping prompt-aggregate energy attribution")
        return
    print(f"  trained aggregate: {trained_agg['n_prompts']} prompt(s) for {model}: "
          f"{trained_agg['prompts']}")

    rmodel       = random_model or f"{model}-random"
    r_phase2_dir = Path(random_phase2_dir) if random_phase2_dir else phase2_dir

    if random_run_dir is not None:
        random_run_dir = Path(random_run_dir)
        if not random_run_dir.exists():
            print(f"  ⚠  --random_run not found: {random_run_dir} — falling back to runs")
            src_random_runs = runs
        else:
            src_random_runs = discover_runs(random_run_dir)
    else:
        # --random_run was not supplied.  `runs` already contains any
        # random-model entries added by --random_seed_dirs in cli.py, so
        # use it directly rather than requiring a separate flag.
        src_random_runs = runs

    random_agg = aggregate_energy_attribution_across_prompts(
        src_random_runs, r_phase2_dir, rmodel, beta=beta,
    )
    if random_agg is None:
        print(f"  ⚠  no Phase 2 data found for {rmodel!r} — random comparison skipped\n"
              f"     (looked in phase2_dir: {r_phase2_dir})")
    else:
        src_label = random_run_dir.name if random_run_dir else "runs"
        print(f"  random aggregate ({src_label}): "
              f"{random_agg['n_prompts']} prompt(s) for {rmodel}: "
              f"{random_agg['prompts']}")

    for s in series:
        plot_energy_attribution_prompt_aggregate(trained_agg, random_agg, out_dir, series=s)

    # Combined figure: all five series, twin axes, one file.
    plot_energy_attribution_combined(trained_agg, random_agg, out_dir)
