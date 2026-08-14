"""
p1b_hemisphere/visualization/membership.py — Block 2, tokens and the boundary.

Six figures (M1-M6). Three are about individual tokens (how stable, how close
to the boundary, when they settle) and three are about the two crossings
Block 2 makes with Phase 1's HDBSCAN labels:

  * `hdbscan_nesting` — does each local cluster sit inside one half, or
    straddle the boundary. M4 draws the r_c distribution against the two
    nesting poles, because a mean "fully nested fraction" hides whether the
    non-nested clusters are near-nested or split down the middle.
  * `border_vs_noise` — are the tokens HDBSCAN calls noise the tokens near
    the Fiedler boundary. M5 is Phase 5c's question, drawn against the 0.5
    line that means "no relationship". M6 draws the two |v| distributions
    behind that AUC, because an AUC of 0.62 is compatible with a large
    separation of a few tokens and a small separation of many.

M4-M6 need the per-layer tables. Runs written before that emission landed
carry only the one-number summary, and these three skip with that reason
rather than plotting a single point as if it were a depth profile.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .loaders import Run, token_field, token_strings
from .style import (
    BLOG_STYLE, CATEGORICAL, HEMI_COLORS, INVALID_COLOR, NOISE_COLOR,
    depth_axis, no_data, reference_line, save_figure,
)

__all__ = ["generate_membership_figures"]


def generate_membership_figures(run: Run, out_dir: Path) -> List[Path]:
    with plt.rc_context(BLOG_STYLE):
        paths = [
            _stability_hist(run, out_dir),
            _border_vs_stability(run, out_dir),
            _first_stable_layer_hist(run, out_dir),
            _nesting_r_c(run, out_dir),
            _border_vs_noise_auc(run, out_dir),
            _noise_vs_clustered_margin(run, out_dir),
        ]
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# M1 — how stable is a token's side
# ---------------------------------------------------------------------------

def _stability_hist(run: Run, out_dir: Path) -> Optional[Path]:
    """
    Distribution of per-token stability, split by dominant hemisphere.

    Split by hemisphere because an asymmetric cone has an asymmetric
    boundary: the minority half is smaller, so its members sit closer to the
    dividing plane on average and should be less stable. If the two
    distributions coincide, that expectation is wrong and worth knowing.
    """
    stability = token_field(run.per_token, "stability_score")
    if not np.isfinite(stability).any():
        return None
    dom = token_field(run.per_token, "dominant_hemisphere")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bins = np.linspace(0, 1, 26)
    for h in (0, 1):
        vals = stability[dom == h]
        if vals.size:
            ax.hist(vals, bins=bins, alpha=0.72, color=HEMI_COLORS[h],
                    label=f"hemisphere {h}  (n={vals.size})")
    reference_line(ax, 0.0, "")  # keeps the axis anchored at zero
    ax.axvline(0.5, **{"color": "#6B7280", "linestyle": ":", "linewidth": 1.2})
    ax.annotate("0.5", xy=(0.5, 0.98), xycoords=("data", "axes fraction"),
                fontsize=8, color="#6B7280", ha="left", va="top")
    ax.axvline(float(np.nanmean(stability)), color=CATEGORICAL[1],
               linewidth=1.6)
    ax.annotate(f"mean {np.nanmean(stability):.3f}",
                xy=(float(np.nanmean(stability)), 0.86),
                xycoords=("data", "axes fraction"), fontsize=8.5,
                color=CATEGORICAL[1], ha="right")
    ax.set_xlabel("stability score (fraction of layers in the dominant half)")
    ax.set_ylabel("tokens")
    ax.legend(loc="upper left")
    ax.set_title(f"{run.label} — how firmly tokens hold a side", fontsize=12)
    return save_figure(fig, out_dir, "stability_hist")


# ---------------------------------------------------------------------------
# M2 — the obvious hypothesis, tested
# ---------------------------------------------------------------------------

def _border_vs_stability(run: Run, out_dir: Path) -> Optional[Path]:
    """
    `border_index` against `stability_score`, one point per token.

    The obvious hypothesis is that unstable tokens are boundary tokens. It is
    obvious enough that it tends to get assumed; this is the scatter that
    either shows it or does not, with the rank correlation printed so the
    answer is a number and not an impression.
    """
    border = token_field(run.per_token, "border_index")
    stability = token_field(run.per_token, "stability_score")
    ok = np.isfinite(border) & np.isfinite(stability)
    if ok.sum() < 3:
        return None

    dom = token_field(run.per_token, "dominant_hemisphere")
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for h in (0, 1):
        m = ok & (dom == h)
        if m.any():
            ax.scatter(border[m], stability[m], s=26, alpha=0.75,
                       color=HEMI_COLORS[h], edgecolor="white", linewidth=0.4,
                       label=f"hemisphere {h}")
    ax.set_xlabel("border index (higher = further from the boundary)")
    ax.set_ylabel("stability score")
    ax.legend(loc="best")

    rho = _spearman(border[ok], stability[ok])
    ax.set_title(f"{run.label} — are unstable tokens boundary tokens?\n"
                 f"Spearman ρ = {rho:+.2f} over {int(ok.sum())} tokens",
                 fontsize=12)
    return save_figure(fig, out_dir, "border_vs_stability")


# ---------------------------------------------------------------------------
# M3 — when tokens settle
# ---------------------------------------------------------------------------

def _first_stable_layer_hist(run: Run, out_dir: Path) -> Optional[Path]:
    """
    First layer from which a token keeps its final side, never-stable
    included as its own terminal bar.

    Never-stable is a category, not a missing value. Dropping those tokens
    would make the histogram describe only the tokens that settled — which is
    the population whose settling time is least interesting.
    """
    first = token_field(run.per_token, "first_assignment_layer")
    n_never = int(np.sum(~np.isfinite(first)))
    settled = first[np.isfinite(first)]
    if settled.size == 0 and n_never == 0:
        return None

    n = run.n_layers
    counts = np.zeros(n + 1)
    for v in settled:
        counts[int(np.clip(v, 0, n - 1))] += 1
    counts[n] = n_never

    fig, ax = plt.subplots(figsize=(10, 4.4))
    colors = [CATEGORICAL[0]] * n + [NOISE_COLOR]
    ax.bar(range(n + 1), counts, color=colors, width=0.8)
    ax.set_xticks(list(range(0, n, max(1, n // 12))) + [n])
    ax.set_xticklabels([str(i) for i in range(0, n, max(1, n // 12))] + ["never"])
    ax.set_xlabel("first layer holding the final hemisphere")
    ax.set_ylabel("tokens")

    frac_never = n_never / max(1, len(first))
    ax.set_title(f"{run.label} — when does a token pick its side?\n"
                 f"{n_never} of {len(first)} never settle ({frac_never:.1%})",
                 fontsize=12)
    return save_figure(fig, out_dir, "first_stable_layer_hist")


# ---------------------------------------------------------------------------
# M4 — nesting
# ---------------------------------------------------------------------------

def _nesting_r_c(run: Run, out_dir: Path) -> Optional[Path]:
    """
    r_c distribution against the nesting poles, plus nesting fraction by depth.

    r_c is a cluster's share of members in half A. A fully nested cluster sits
    at 0 or 1; one split evenly straddles at 0.5. The distribution is the
    result — the "fully nested fraction" is a count of how much of it fell
    inside a tolerance, and two very different distributions can produce the
    same count.
    """
    nesting = run.nesting
    if nesting is None:
        print(f"    membership: M4 skipped for {run.stem} — "
              f"hdbscan_nesting not in artifacts")
        return None

    per_layer = nesting["per_layer"]
    r_c, sizes = [], []
    for L in sorted(per_layer):
        for c in per_layer[L].get("clusters", []):
            if c.get("r_c") is not None:
                r_c.append(float(c["r_c"]))
                sizes.append(float(c.get("size", 1)))
    if not r_c:
        return None
    r_c = np.asarray(r_c)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.2]})

    ax = axes[0]
    ax.hist(r_c, bins=np.linspace(0, 1, 26), color=CATEGORICAL[0], alpha=0.9)
    for x, lab in ((0.0, "nested in A"), (1.0, "nested in B")):
        ax.axvline(x, color="#374151", linewidth=1.2)
    ax.axvline(0.5, **{"color": "#6B7280", "linestyle": ":", "linewidth": 1.2})
    ax.annotate("straddling", xy=(0.5, 0.96), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=8.5, color="#6B7280")
    ax.set_xlabel("r_c — share of a cluster's members in half A")
    ax.set_ylabel("clusters")
    overall = nesting["overall"]
    ax.set_title(f"fully nested {_pct(overall.get('fully_nested_fraction'))} · "
                 f"mixed {_pct(overall.get('mixed_fraction'))}", fontsize=11)

    ax = axes[1]
    layers = sorted(per_layer)
    nested = [per_layer[L].get("summary", {}).get("fully_nested_fraction")
              for L in layers]
    nested = [np.nan if v is None else float(v) for v in nested]
    ax.plot(layers, nested, color=CATEGORICAL[2], linewidth=2.2, marker="o",
            markersize=3.5)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fully nested fraction")
    depth_axis(ax, run.n_layers)
    ax.set_title("nesting by depth", fontsize=11)

    fig.suptitle(f"{run.label} — do HDBSCAN clusters sit inside one half?",
                 fontsize=12, y=1.03)
    return save_figure(fig, out_dir, "nesting_r_c")


# ---------------------------------------------------------------------------
# M5 — Phase 5c's question
# ---------------------------------------------------------------------------

def _border_vs_noise_auc(run: Run, out_dir: Path) -> Optional[Path]:
    """
    AUC vs depth against 0.5.

    The probability that an HDBSCAN-noise token sits nearer the Fiedler
    boundary than a clustered one. 0.5 is no relationship, and it is drawn as
    a line rather than implied by the axis, because the whole question is
    whether the curve is distinguishable from it.
    """
    bvn = run.border_vs_noise
    if bvn is None:
        print(f"    membership: M5 skipped for {run.stem} — "
              f"border_vs_noise not in artifacts")
        return None

    per_layer = bvn["per_layer"]
    layers = sorted(per_layer)
    auc = np.array([float(per_layer[L].get("auc", np.nan)) for L in layers])

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axhspan(0.5, 1.0, color=CATEGORICAL[0], alpha=0.06, zorder=0,
               linewidth=0)
    ax.plot(layers, auc, color=CATEGORICAL[0], linewidth=2.4, marker="o",
            markersize=4)
    reference_line(ax, 0.5, "0.5 — no relationship")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("AUC — noise nearer the boundary")
    depth_axis(ax, run.n_layers)

    overall = bvn["overall"]
    ax.set_title(
        f"{run.label} — is the unclustered population the boundary population?\n"
        f"mean {_num(overall.get('mean_auc'))} · "
        f"{_pct(overall.get('fraction_layers_auc_above_0.6'))} of layers above 0.6",
        fontsize=12)
    ax.annotate("Phase 5c's object of study, from two quantities that already existed",
                xy=(0.01, 0.02), xycoords="axes fraction", fontsize=8,
                color="#6B7280")
    return save_figure(fig, out_dir, "border_vs_noise_auc")


# ---------------------------------------------------------------------------
# M6 — the two distributions behind the AUC
# ---------------------------------------------------------------------------

def _noise_vs_clustered_margin(run: Run, out_dir: Path) -> Optional[Path]:
    """
    Mean |v| for noise vs clustered tokens per layer, with the populations'
    sizes underneath.

    An AUC is a rank statistic and says nothing about magnitude. These are
    the two means it ranks; if they are 0.18 and 0.36 the effect is large,
    and if they are 0.28 and 0.30 the same AUC describes something almost
    invisible.
    """
    bvn = run.border_vs_noise
    if bvn is None:
        return None

    per_layer = bvn["per_layer"]
    layers = sorted(per_layer)
    if not layers:
        return None
    noise = np.array([_get(per_layer[L], "mean_abs_v_noise") for L in layers])
    clus = np.array([_get(per_layer[L], "mean_abs_v_clustered") for L in layers])
    n_noise = np.array([_get(per_layer[L], "n_noise") for L in layers])
    n_clus = np.array([_get(per_layer[L], "n_clustered") for L in layers])

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})

    ax = axes[0]
    ax.plot(layers, noise, color=NOISE_COLOR, linewidth=2.6,
            marker="o", markersize=4, label="HDBSCAN noise")
    ax.plot(layers, clus, color=CATEGORICAL[0], linewidth=2.4,
            marker="s", markersize=4, label="clustered")
    ax.fill_between(layers, noise, clus, color=CATEGORICAL[0], alpha=0.08)
    ax.set_ylabel("mean |Fiedler value|")
    ax.legend(loc="best")
    ax.set_title(f"{run.label} — how far each population sits from the boundary",
                 fontsize=12)

    ax = axes[1]
    ax.bar(layers, n_noise, color=NOISE_COLOR, width=0.78, label="noise")
    ax.bar(layers, n_clus, bottom=n_noise, color=CATEGORICAL[0], width=0.78,
           alpha=0.9, label="clustered")
    ax.set_ylabel("tokens")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    depth_axis(ax, run.n_layers)
    return save_figure(fig, out_dir, "noise_vs_clustered_margin")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy import for one number."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.size < 2 or np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _get(d: dict, key: str) -> float:
    v = d.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _pct(v) -> str:
    return "n/a" if v is None else f"{float(v) * 100:.0f}%"


def _num(v, dp: int = 3) -> str:
    return "n/a" if v is None else f"{float(v):.{dp}f}"
