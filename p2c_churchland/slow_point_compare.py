"""
slow_point_compare.py — C4: local vs global S/A ratio, layer-wise comparison.

Takes the output of local_jacobian.analyze_local_jacobians and compares:
  - local S/A ratio at metastable (plateau) states vs global V's S/A ratio
  - local S/A at plateau layers vs merge layers
  - layer-depth profile of S/A ratio to identify where rotation concentrates

Produces the falsification verdicts for P2c-S1 and P2c-S2, plus plots.

Functions
---------
layer_sa_profile       : mean S/A ratio per layer across all centroids
plateau_vs_merge_table : tabulated comparison (plateau / merge / global)
compare_local_global   : full S1/S2 verdict with effect sizes
sa_ratio_by_layer_type : separate per-layer-type distributions for stats
bootstrap_ci           : 95% CI for mean sa_ratio in a group (bootstrap)
print_comparison       : terminal summary
plot_sa_profile        : matplotlib figure (optional)
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


# ---------------------------------------------------------------------------
# Layer-level aggregation
# ---------------------------------------------------------------------------

def layer_sa_profile(
    per_layer: dict,
    layer_order: list[int] | None = None,
) -> dict:
    """
    Compute mean and std of sa_ratio per layer.

    Parameters
    ----------
    per_layer   : dict layer_idx → list of per-centroid records
                  (output of analyze_local_jacobians["per_layer"])
    layer_order : optional list defining layer display order (default: sorted keys)

    Returns
    -------
    dict with:
      layer_indices : list of layer indices (in order)
      mean_sa       : (n_layers,) array of mean sa_ratio
      std_sa        : (n_layers,) array of std sa_ratio
      n_centroids   : (n_layers,) array of centroid counts per layer
    """
    if layer_order is None:
        layer_order = sorted(per_layer.keys())

    means, stds, ns = [], [], []
    for li in layer_order:
        vals = [r["sa_ratio"] for r in per_layer[li]]
        arr = np.array(vals, dtype=float)
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr, ddof=0)))
        ns.append(len(arr))

    return {
        "layer_indices": layer_order,
        "mean_sa":       np.array(means),
        "std_sa":        np.array(stds),
        "n_centroids":   np.array(ns, dtype=int),
    }


# ---------------------------------------------------------------------------
# Group comparisons
# ---------------------------------------------------------------------------

def plateau_vs_merge_table(
    per_layer: dict,
    plateau_layers: Sequence[int],
    merge_layers: Sequence[int],
    global_sa_ratio: float,
) -> dict:
    """
    Collect sa_ratio values split by layer type and compute descriptive stats.

    Parameters
    ----------
    per_layer       : dict layer_idx → list of per-centroid records
    plateau_layers  : layer indices classified as plateau (Phase 1)
    merge_layers    : layer indices classified as merge events (Phase 1)
    global_sa_ratio : V's global S/A ratio from p2b

    Returns
    -------
    dict with:
      plateau_vals    : flat array of sa_ratio at plateau layers
      merge_vals      : flat array of sa_ratio at merge layers
      other_vals      : flat array of sa_ratio at neither
      plateau_mean    : float
      merge_mean      : float
      other_mean      : float
      global_sa_ratio : echo
    """
    plateau_set = set(plateau_layers)
    merge_set   = set(merge_layers)

    plateau_vals, merge_vals, other_vals = [], [], []

    for li, recs in per_layer.items():
        vals = [r["sa_ratio"] for r in recs]
        if li in plateau_set:
            plateau_vals.extend(vals)
        elif li in merge_set:
            merge_vals.extend(vals)
        else:
            other_vals.extend(vals)

    def _stats(v):
        a = np.array(v, dtype=float)
        return float(np.mean(a)) if len(a) else float("nan")

    return {
        "plateau_vals":    np.array(plateau_vals),
        "merge_vals":      np.array(merge_vals),
        "other_vals":      np.array(other_vals),
        "plateau_mean":    _stats(plateau_vals),
        "merge_mean":      _stats(merge_vals),
        "other_mean":      _stats(other_vals),
        "global_sa_ratio": global_sa_ratio,
    }


def bootstrap_ci(
    vals: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Nonparametric bootstrap confidence interval for the mean.

    Parameters
    ----------
    vals   : 1-D array of observations
    n_boot : number of bootstrap resamples
    ci     : confidence level
    seed   : RNG seed

    Returns
    -------
    (lo, hi) : lower and upper bounds
    """
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        np.mean(rng.choice(vals, size=len(vals), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.quantile(boot_means, alpha)),
        float(np.quantile(boot_means, 1.0 - alpha)),
    )


def compare_local_global(
    per_layer: dict,
    plateau_layers: Sequence[int],
    merge_layers: Sequence[int],
    global_sa_ratio: float,
    n_boot: int = 2000,
) -> dict:
    """
    Full S1/S2 comparison with effect sizes and bootstrap CIs.

    P2c-S1: plateau_mean_sa > global_sa_ratio
      Effect size: (plateau_mean - global_sa_ratio)

    P2c-S2: merge_mean_sa < plateau_mean_sa
      Effect size: (plateau_mean - merge_mean)

    Parameters
    ----------
    per_layer       : dict layer_idx → list of per-centroid records
    plateau_layers  : Phase 1 plateau layer indices
    merge_layers    : Phase 1 merge layer indices
    global_sa_ratio : V's global S/A ratio from p2b
    n_boot          : bootstrap resamples for CIs

    Returns
    -------
    dict with full comparison, CIs, and boolean verdicts
    """
    tbl = plateau_vs_merge_table(
        per_layer, plateau_layers, merge_layers, global_sa_ratio
    )

    pv = tbl["plateau_vals"]
    mv = tbl["merge_vals"]

    p_ci = bootstrap_ci(pv, n_boot=n_boot)
    m_ci = bootstrap_ci(mv, n_boot=n_boot)

    plateau_mean = tbl["plateau_mean"]
    merge_mean   = tbl["merge_mean"]

    # Effect sizes (raw difference; sign carries the prediction)
    s1_effect = plateau_mean - global_sa_ratio   # >0 → P2c-S1 holds
    s2_effect = plateau_mean - merge_mean         # >0 → P2c-S2 holds

    return {
        "plateau_mean":    plateau_mean,
        "plateau_ci_95":   p_ci,
        "merge_mean":      merge_mean,
        "merge_ci_95":     m_ci,
        "global_sa_ratio": global_sa_ratio,

        # Prediction verdicts
        "p2cs1_holds":     plateau_mean > global_sa_ratio,
        "s1_effect":       s1_effect,   # plateau excess over global V
        "p2cs2_holds":     merge_mean < plateau_mean,
        "s2_effect":       s2_effect,   # plateau−merge gap

        # Raw group values for downstream use
        "plateau_vals":    pv,
        "merge_vals":      mv,
        "other_vals":      tbl["other_vals"],
        "other_mean":      tbl["other_mean"],

        # Layer profile (for plot)
        "profile":         layer_sa_profile(per_layer),
    }


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def print_comparison(result: dict) -> None:
    """Print a flat text summary of compare_local_global output."""
    p = result
    sep = "-" * 60
    print(sep)
    print("C4 — Local Jacobian S/A Comparison")
    print(sep)
    print(f"  Global V S/A ratio (p2b):   {p['global_sa_ratio']:.4f}")
    print()

    pm, (plo, phi) = p["plateau_mean"], p["plateau_ci_95"]
    mm, (mlo, mhi) = p["merge_mean"],   p["merge_ci_95"]
    om               = p["other_mean"]

    print(f"  Plateau layers  mean S/A:   {pm:.4f}  [95% CI {plo:.4f}–{phi:.4f}]  "
          f"(n={len(p['plateau_vals'])})")
    print(f"  Merge layers    mean S/A:   {mm:.4f}  [95% CI {mlo:.4f}–{mhi:.4f}]  "
          f"(n={len(p['merge_vals'])})")
    if not np.isnan(om):
        print(f"  Other layers    mean S/A:   {om:.4f}")
    print()

    s1 = "HOLDS" if p["p2cs1_holds"] else "FAILS"
    s2 = "HOLDS" if p["p2cs2_holds"] else "FAILS"
    print(f"  P2c-S1 (plateau more symmetric than V):  {s1}  "
          f"(effect {p['s1_effect']:+.4f})")
    print(f"  P2c-S2 (merge less symmetric than plateau): {s2}  "
          f"(effect {p['s2_effect']:+.4f})")
    print(sep)


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------

def plot_sa_profile(
    result: dict,
    plateau_layers: Sequence[int],
    merge_layers: Sequence[int],
    title: str = "Local Jacobian S/A ratio by layer",
    save_path: str | None = None,
) -> None:
    """
    Plot mean S/A ratio per layer with shaded ±1 std, annotated by layer type.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    prof = result["profile"]
    xs   = prof["layer_indices"]
    ys   = prof["mean_sa"]
    err  = prof["std_sa"]

    plateau_set = set(plateau_layers)
    merge_set   = set(merge_layers)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ys, color="steelblue", linewidth=1.5, label="mean S/A")
    ax.fill_between(xs, ys - err, ys + err, alpha=0.2, color="steelblue")

    # Shade plateau and merge regions
    for li, x in zip(xs, xs):
        if li in plateau_set:
            ax.axvspan(x - 0.5, x + 0.5, alpha=0.12, color="green", linewidth=0)
        elif li in merge_set:
            ax.axvspan(x - 0.5, x + 0.5, alpha=0.20, color="red", linewidth=0)

    ax.axhline(
        result["global_sa_ratio"],
        linestyle="--", color="black", linewidth=1.0, label="global V S/A"
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("S/A ratio")
    ax.set_title(title)

    patches = [
        mpatches.Patch(color="green",  alpha=0.4, label="plateau"),
        mpatches.Patch(color="red",    alpha=0.4, label="merge"),
        mpatches.Patch(color="black",  alpha=0.0, label="─── global V"),
    ]
    ax.legend(handles=[ax.get_lines()[0], *patches], fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def slow_point_compare_to_json(result: dict) -> dict:
    """
    JSON-serializable summary of compare_local_global output.
    Drops raw value arrays (plateau_vals, merge_vals, etc.).
    """
    prof = result["profile"]
    return {
        "plateau_mean":    result["plateau_mean"],
        "plateau_ci_95":   list(result["plateau_ci_95"]),
        "merge_mean":      result["merge_mean"],
        "merge_ci_95":     list(result["merge_ci_95"]),
        "other_mean":      result["other_mean"],
        "global_sa_ratio": result["global_sa_ratio"],
        "p2cs1_holds":     result["p2cs1_holds"],
        "s1_effect":       result["s1_effect"],
        "p2cs2_holds":     result["p2cs2_holds"],
        "s2_effect":       result["s2_effect"],
        "layer_profile": {
            "layer_indices": [int(x) for x in prof["layer_indices"]],
            "mean_sa":       prof["mean_sa"].tolist(),
            "std_sa":        prof["std_sa"].tolist(),
            "n_centroids":   prof["n_centroids"].tolist(),
        },
    }
