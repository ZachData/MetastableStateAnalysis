"""
p1b_hemisphere/visualization/curiosities.py — the speculative half.

Eleven figures (X1-X11). None of these is a verdict figure and none belongs
in a falsification table. They exist because the particle table makes
per-(layer, token) structure cheap to look at, and looking is how the next
question gets found — which is the whole premise of the v2 plan's
"particles first" reframe.

The rule they follow: a curiosity figure has to be able to show nothing.
Several of these are drawn precisely so that a flat, structureless result is
visible as such — X4 (does the split track sequence position), X5 (does it
track token surface class), X10 (is the dwell distribution bimodal). A flat
X10 would say the hemisphere sign is noise; a bimodal one would say there
are two genuine populations. Both are worth one PNG.

X1 is the one to look at first. The Fiedler barcode is the whole run in one
image — every Block 0, 1, and 2 quantity in the package is some projection
of it, and it takes about two seconds to see whether a run has depth
structure at all.

Everything here reads the particle table (`phase1b_{stem}_particles.npz`),
falling back to `per_token.hemisphere_trajectory` where the same information
exists in the JSON, so a run written without particles still gets most of
the set.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .loaders import Run, particle_grid, token_field, token_strings, token_trajectories
from .style import (
    BLOG_STYLE, CATEGORICAL, FIEDLER_CMAP, HEMI_COLORS, INVALID_COLOR,
    NOISE_COLOR, SEQ_CMAP, depth_axis, fiedler_norm, no_data, reference_line,
    save_figure,
)

__all__ = ["generate_curiosity_figures", "TOKEN_CLASSES", "classify_token"]


def generate_curiosity_figures(run: Run, out_dir: Path) -> List[Path]:
    with plt.rc_context(BLOG_STYLE):
        paths = [
            _fiedler_barcode(run, out_dir),
            _hemisphere_ribbon(run, out_dir),
            _token_flow(run, out_dir),
            _position_vs_hemisphere(run, out_dir),
            _token_class_split(run, out_dir),
            _most_volatile_tokens(run, out_dir),
            _cone_opening_polar(run, out_dir),
            _border_dwellers(run, out_dir),
            _stability_landscape(run, out_dir),
            _hemisphere_dwell_histogram(run, out_dir),
            _run_fingerprint(run, out_dir),
        ]
    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# X1 — the whole run in one image
# ---------------------------------------------------------------------------

def _fiedler_barcode(run: Run, out_dir: Path) -> Optional[Path]:
    """
    Layer × token heatmap of the signed Fiedler value.

    Interesting if: the vertical banding survives depth (one axis, stable
    identity), or it reorganizes at a particular layer (the axis is a
    different object above and below it), or the whole image pales toward the
    last layers (the bipartition dissolving into the collapse the paper's
    theorem predicts).

    The colormap is anchored at zero by `fiedler_norm`, not at the data
    midpoint: with an 80/20 split matplotlib's default would put the neutral
    color inside the majority hemisphere and the image would show a boundary
    that is not there.
    """
    grid = particle_grid(run, "fiedler_value")
    if grid is None:
        return None

    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=FIEDLER_CMAP,
                   norm=fiedler_norm(grid), interpolation="nearest")
    ax.set_xlabel("token position")
    ax.set_ylabel("layer")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.012)
    cb.set_label("Fiedler value (sign = hemisphere)")

    ax.set_title(f"{run.label} — the axis as a barcode\n"
                 f"every per-layer and per-token quantity in this phase is a "
                 f"projection of this image", fontsize=12)
    return save_figure(fig, out_dir, "fiedler_barcode")


# ---------------------------------------------------------------------------
# X2 — the same thing, as membership, sorted
# ---------------------------------------------------------------------------

def _hemisphere_ribbon(run: Run, out_dir: Path) -> Optional[Path]:
    """
    Layer × token hemisphere membership, tokens sorted by stability.

    Sorting is what makes it readable: unsorted, a mixing band looks like
    noise everywhere. Sorted by stability, the turbulent population collects
    at one edge and the laminar one at the other, and the question becomes
    how wide the turbulent band is rather than whether there is one.
    """
    grid = particle_grid(run, "hemisphere")
    if grid is None:
        traj = token_trajectories(run)
        grid = traj.T if traj is not None else None
    if grid is None:
        return None

    stability = token_field(run.per_token, "stability_score")
    if stability.size == grid.shape[1] and np.isfinite(stability).any():
        order = np.argsort(np.nan_to_num(stability, nan=-1.0))
        sort_note = "sorted by stability (least stable at left)"
    else:
        order = np.arange(grid.shape[1])
        sort_note = "in token order (no stability scores in this run)"

    cmap = ListedColormap([INVALID_COLOR, HEMI_COLORS[0], HEMI_COLORS[1]])
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.imshow(np.clip(grid[:, order] + 1, 0, 2), aspect="auto", origin="lower",
              cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.set_xlabel(f"token — {sort_note}")
    ax.set_ylabel("layer")
    ax.grid(False)

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          color=HEMI_COLORS[h], label=f"hemisphere {h}")
               for h in (0, 1)]
    handles.append(plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                              color=INVALID_COLOR, label="invalid layer"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              fontsize=8.5)
    ax.set_title(f"{run.label} — laminar and turbulent tokens", fontsize=12)
    return save_figure(fig, out_dir, "hemisphere_ribbon")


# ---------------------------------------------------------------------------
# X3 — where the mixing happens
# ---------------------------------------------------------------------------

def _token_flow(run: Run, out_dir: Path) -> Optional[Path]:
    """
    The 2×2 hemisphere transition matrix at each layer pair, as flow bands.

    Interesting if: the off-diagonal mass concentrates at a few depths (the
    axis reorganizes there and holds elsewhere) or is spread evenly (tokens
    drift across the boundary continuously, which would make "membership" a
    much weaker notion than the sign label suggests).
    """
    grid = particle_grid(run, "hemisphere")
    if grid is None:
        traj = token_trajectories(run)
        grid = traj.T if traj is not None else None
    if grid is None or grid.shape[0] < 2:
        return None

    n_L = grid.shape[0]
    stay = np.zeros(n_L - 1)
    zero_to_one = np.zeros(n_L - 1)
    one_to_zero = np.zeros(n_L - 1)
    for L in range(n_L - 1):
        a, b = grid[L], grid[L + 1]
        valid = (a >= 0) & (b >= 0)
        n = max(1, int(valid.sum()))
        stay[L] = float(((a == b) & valid).sum()) / n
        zero_to_one[L] = float(((a == 0) & (b == 1) & valid).sum()) / n
        one_to_zero[L] = float(((a == 1) & (b == 0) & valid).sum()) / n

    x = np.arange(n_L - 1)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(x, 0, stay, color="#D8DEE6", linewidth=0, label="stayed")
    ax.fill_between(x, stay, stay + zero_to_one, color=HEMI_COLORS[1],
                    alpha=0.9, linewidth=0, label="0 → 1")
    ax.fill_between(x, stay + zero_to_one, stay + zero_to_one + one_to_zero,
                    color=HEMI_COLORS[0], alpha=0.9, linewidth=0, label="1 → 0")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, n_L - 2)
    ax.set_xlabel("transition (layer L → L+1)")
    ax.set_ylabel("share of tokens")
    ax.legend(loc="lower left", ncol=3, fontsize=8.5)
    ax.grid(False)

    busiest = int(np.argmax(zero_to_one + one_to_zero))
    ax.set_title(f"{run.label} — where tokens change sides\n"
                 f"busiest transition {busiest} → {busiest + 1} "
                 f"({(zero_to_one + one_to_zero)[busiest]:.1%} of tokens)",
                 fontsize=12)
    return save_figure(fig, out_dir, "token_flow")


# ---------------------------------------------------------------------------
# X4 — is the split just "position 0 vs everything else"
# ---------------------------------------------------------------------------

def _position_vs_hemisphere(run: Run, out_dir: Path) -> Optional[Path]:
    """
    Hemisphere membership against sequence position, with position 0 called out.

    Interesting if: the majority-hemisphere share varies smoothly with
    position (the axis is partly positional, which would make it a much less
    semantic object than "hemisphere" suggests), or if position 0 alone sits
    apart — the attention-sink token is a known outlier in exactly this kind
    of geometry, and a bipartition that is really "sink vs rest" would show
    here first and nowhere else.
    """
    grid = particle_grid(run, "hemisphere")
    if grid is None:
        traj = token_trajectories(run)
        grid = traj.T if traj is not None else None
    if grid is None:
        return None

    valid = grid >= 0
    share = np.where(valid.sum(0) > 0,
                     np.where(valid, grid, 0).sum(0) / np.maximum(1, valid.sum(0)),
                     np.nan)
    pos = np.arange(share.size)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6),
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.scatter(pos[1:], share[1:], s=22, color=CATEGORICAL[0], alpha=0.75,
               edgecolor="white", linewidth=0.4)
    if share.size:
        ax.scatter([0], [share[0]], s=150, marker="*", color=CATEGORICAL[1],
                   zorder=5, label="position 0 (attention sink)")
        ax.legend(loc="best")
    reference_line(ax, 0.5, "0.5")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("share of layers in hemisphere 1")
    ax.set_xlabel("token position")

    rho = _spearman(pos[1:].astype(float), share[1:])
    ax.set_title(f"{run.label} — does the split track sequence position?\n"
                 f"Spearman ρ = {rho:+.2f} (position 0 excluded)", fontsize=12)

    # A binned view, because 148 scattered points can hide a monotone trend
    # that 10 bins make obvious.
    ax = axes[1]
    nb = min(12, max(3, share.size // 10))
    edges = np.linspace(0, share.size, nb + 1)
    centers, means = [], []
    for i in range(nb):
        m = (pos >= edges[i]) & (pos < edges[i + 1]) & np.isfinite(share)
        if m.any():
            centers.append(float(pos[m].mean()))
            means.append(float(share[m].mean()))
    ax.plot(centers, means, color=CATEGORICAL[2], linewidth=2.2, marker="o")
    reference_line(ax, 0.5, "0.5")
    ax.set_ylim(0, 1)
    ax.set_ylabel("binned mean")
    ax.set_xlabel("token position")
    return save_figure(fig, out_dir, "position_vs_hemisphere")


# ---------------------------------------------------------------------------
# X5 — a Block 6 proxy that needs no Phase 2
# ---------------------------------------------------------------------------

#: Surface classes, in the order they are tested. Crude on purpose: this is a
#: tokenizer-level split, not a semantic one, and calling it "semantics"
#: would be exactly the overclaim Block 6 exists to do properly.
TOKEN_CLASSES = ("punctuation", "numeric", "leading-space word",
                 "subword continuation", "other")


def classify_token(tok: str) -> str:
    """Surface class of one token string. See TOKEN_CLASSES."""
    s = str(tok)
    core = s.strip()
    if not core:
        return "other"
    if re.fullmatch(r"[^\w\s]+", core):
        return "punctuation"
    if re.fullmatch(r"[\d.,]+", core):
        return "numeric"
    if s[:1] in (" ", "Ġ", "▁"):
        return "leading-space word"
    if re.fullmatch(r"\w+", s):
        return "subword continuation"
    return "other"


def _token_class_split(run: Run, out_dir: Path) -> Optional[Path]:
    """
    Hemisphere membership by token surface class.

    Interesting if: one class lands disproportionately on one side. That
    would be the cheapest possible evidence that the axis carries something
    lexical, and would tell Block 6 — which needs Phase 2 artifacts it does
    not have — where to look when it can finally run. A flat result is
    equally informative and much more likely.

    Explicitly a proxy. Tokenizer surface form is not semantics, and this
    figure's caption says so on the figure, not just here.
    """
    grid = particle_grid(run, "hemisphere")
    if grid is None:
        traj = token_trajectories(run)
        grid = traj.T if traj is not None else None
    if grid is None:
        return None

    toks = token_strings(run.per_token)
    if not toks or len(toks) != grid.shape[1]:
        cols = run.particles()
        if not cols or "token_str" not in cols:
            return None
        toks = [""] * grid.shape[1]
        for pos, s in zip(np.asarray(cols["token_position"]),
                          np.asarray(cols["token_str"])):
            if 0 <= int(pos) < len(toks):
                toks[int(pos)] = str(s)

    classes = [classify_token(t) for t in toks]
    valid = grid >= 0
    share = np.where(valid.sum(0) > 0,
                     np.where(valid, grid, 0).sum(0) / np.maximum(1, valid.sum(0)),
                     np.nan)

    present = [c for c in TOKEN_CLASSES if any(x == c for x in classes)]
    if not present:
        return None

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    data, labels = [], []
    for c in present:
        vals = np.array([share[i] for i, x in enumerate(classes)
                         if x == c and np.isfinite(share[i])])
        if vals.size:
            data.append(vals)
            labels.append(f"{c}\n(n={vals.size})")

    parts = ax.violinplot(data, showmeans=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor(CATEGORICAL[0])
        body.set_alpha(0.55)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("#374151")
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(vals.size, i) + np.random.default_rng(0)
                   .normal(0, 0.045, vals.size), vals, s=12,
                   color=CATEGORICAL[1], alpha=0.55, zorder=3)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8.5)
    reference_line(ax, 0.5, "0.5")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("share of layers in hemisphere 1")
    ax.set_title(f"{run.label} — does the axis care what a token IS?\n"
                 f"surface form, not semantics — a stand-in for Block 6, "
                 f"which needs Phase 2 artifacts", fontsize=12)
    return save_figure(fig, out_dir, "token_class_split")


# ---------------------------------------------------------------------------
# X6 — the tokens that cannot decide
# ---------------------------------------------------------------------------

def _most_volatile_tokens(run: Run, out_dir: Path, k: int = 24) -> Optional[Path]:
    """
    The least-stable tokens, with their trajectories as mini-ribbons.

    Interesting if: the volatile set is lexically coherent (function words,
    punctuation, the sink) rather than arbitrary. `per_token` is already
    sorted by stability ascending, so this is the head of that list made
    legible — which is a thing a JSON file cannot be.
    """
    traj = token_trajectories(run)
    if traj is None:
        grid = particle_grid(run, "hemisphere")
        traj = grid.T if grid is not None else None
    if traj is None:
        return None

    stability = token_field(run.per_token, "stability_score")
    toks = token_strings(run.per_token)
    if stability.size != traj.shape[0]:
        return None

    order = np.argsort(np.nan_to_num(stability, nan=2.0))[:k]
    cmap = ListedColormap([INVALID_COLOR, HEMI_COLORS[0], HEMI_COLORS[1]])

    fig, ax = plt.subplots(figsize=(11, 0.32 * len(order) + 2.2))
    ax.imshow(np.clip(traj[order] + 1, 0, 2), aspect="auto", cmap=cmap,
              vmin=0, vmax=2, interpolation="nearest")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [f"{repr(toks[i])[1:-1] if i < len(toks) else ''}  @{i}  "
         f"({stability[i]:.2f})" for i in order], fontsize=7.5)
    ax.set_xlabel("layer")
    ax.grid(False)
    ax.set_title(f"{run.label} — the {len(order)} least stable tokens\n"
                 f"label is token, position, stability score", fontsize=12)
    return save_figure(fig, out_dir, "most_volatile_tokens")


# ---------------------------------------------------------------------------
# X7 — the cone as a shape
# ---------------------------------------------------------------------------

def _cone_opening_polar(run: Run, out_dir: Path) -> Optional[Path]:
    """
    The cone's half-angle vs depth, on polar axes.

    The normalized margin is the sine of an angle: how far inside the open
    half-space the tightest token sits. Drawing it as an angle rather than a
    number makes "the cone tightens with depth" a shape closing, which is the
    thing the paper's convergence theorem is about.

    Decorative in the sense that it carries no information C1 lacks. Kept
    because it is the only figure in the package where the geometry looks
    like the geometry.
    """
    margin = run.field("normalized_margin")
    if not np.isfinite(margin).any():
        return None

    n = len(margin)
    half_angle = np.arcsin(np.clip(margin, -1.0, 1.0))
    depth = np.arange(n)

    fig = plt.figure(figsize=(6.8, 6.8))
    ax = fig.add_subplot(projection="polar")
    cmap = SEQ_CMAP
    for L in range(n):
        if not np.isfinite(half_angle[L]):
            continue
        color = cmap(0.25 + 0.7 * (L / max(1, n - 1)))
        ax.plot([0, half_angle[L]], [0, 1], color=color, linewidth=2.0,
                alpha=0.85)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_title(f"{run.label} — how tight is the containment\n"
                 f"each ray is one layer's cone half-angle "
                 f"(pale = early, dark = late)", fontsize=11, pad=22)
    return save_figure(fig, out_dir, "cone_opening_polar")


# ---------------------------------------------------------------------------
# X8 — who lives on the boundary
# ---------------------------------------------------------------------------

def _border_dwellers(run: Run, out_dir: Path, k: int = 20) -> Optional[Path]:
    """
    Per layer, the tokens nearest the Fiedler boundary, named.

    M5 measures this population in aggregate as an AUC. This lists it. If the
    same handful of tokens sits on the boundary at every depth, "the
    unclustered population" has a concrete membership and Phase 5c can go
    look at it rather than at a statistic.
    """
    grid = particle_grid(run, "fiedler_value")
    if grid is None:
        return None

    toks = _positions_to_tokens(run)
    near = np.argsort(np.abs(grid), axis=1)[:, :k]
    counts = np.bincount(near.ravel(), minlength=grid.shape[1])
    order = np.argsort(-counts)[:24]
    order = [int(t) for t in order if counts[t] > 0]
    if not order:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    ax = axes[0]
    mask = np.zeros_like(grid, dtype=float)
    for L in range(grid.shape[0]):
        mask[L, near[L]] = 1.0
    ax.imshow(mask, aspect="auto", origin="lower", cmap="Purples",
              interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel("token position")
    ax.set_ylabel("layer")
    ax.grid(False)
    ax.set_title(f"the {k} tokens nearest the boundary, per layer", fontsize=11)

    ax = axes[1]
    ax.barh(range(len(order)), [counts[t] for t in order],
            color=CATEGORICAL[4], alpha=0.9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{repr(toks[t])[1:-1]}  @{t}" for t in order],
                       fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"layers in the nearest-{k}")
    ax.set_title("the persistent boundary dwellers", fontsize=11)

    fig.suptitle(f"{run.label} — who lives on the Fiedler boundary",
                 fontsize=12, y=1.01)
    return save_figure(fig, out_dir, "border_dwellers")


# ---------------------------------------------------------------------------
# X9 — the same scatter, as density
# ---------------------------------------------------------------------------

def _stability_landscape(run: Run, out_dir: Path) -> Optional[Path]:
    """
    (border_index, stability) as a hexbin, with the extremes labeled.

    M2's scatter saturates on a long prompt — 148 points at 26 px is a blob.
    The density shows where the population actually is, and the labeled
    extremes name the four corners: stable-and-interior, stable-and-marginal,
    volatile-and-interior (the interesting one, if it is populated), and
    volatile-and-marginal.
    """
    border = token_field(run.per_token, "border_index")
    stability = token_field(run.per_token, "stability_score")
    ok = np.isfinite(border) & np.isfinite(stability)
    if ok.sum() < 12:
        return None

    toks = token_strings(run.per_token)
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    hb = ax.hexbin(border[ok], stability[ok], gridsize=18, cmap=SEQ_CMAP,
                   mincnt=1, linewidths=0.2, edgecolors="white")
    fig.colorbar(hb, ax=ax, label="tokens")

    # Label the extreme of each corner, not every point.
    corners = {
        "most volatile": int(np.nanargmin(np.where(ok, stability, np.nan))),
        "closest to boundary": int(np.nanargmin(np.where(ok, border, np.nan))),
        "most interior": int(np.nanargmax(np.where(ok, border, np.nan))),
    }
    for label, i in corners.items():
        if not (0 <= i < len(toks)):
            continue
        ax.annotate(f"{label}: {repr(toks[i])[1:-1]}",
                    xy=(border[i], stability[i]), fontsize=8,
                    xytext=(6, 6), textcoords="offset points",
                    color="#111827",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#D1D5DB", alpha=0.9))
        ax.scatter([border[i]], [stability[i]], s=40, facecolor="none",
                   edgecolor=CATEGORICAL[1], linewidth=1.6, zorder=5)

    ax.set_xlabel("border index (higher = further from the boundary)")
    ax.set_ylabel("stability score")
    ax.set_title(f"{run.label} — the token population, as a landscape",
                 fontsize=12)
    return save_figure(fig, out_dir, "stability_landscape")


# ---------------------------------------------------------------------------
# X10 — two populations, or one
# ---------------------------------------------------------------------------

def _hemisphere_dwell_histogram(run: Run, out_dir: Path) -> Optional[Path]:
    """
    Layers spent in hemisphere 1, per token.

    The cleanest test in the package of whether the sign means anything.
    Bimodal at 0 and n_layers: two genuine populations that keep their sides.
    Unimodal at n_layers/2: the sign is a coin flip re-tossed at every layer,
    and every hemisphere-based quantity downstream is measuring noise. The
    two outcomes look nothing alike, which is what makes it worth one PNG.
    """
    grid = particle_grid(run, "hemisphere")
    if grid is None:
        traj = token_trajectories(run)
        grid = traj.T if traj is not None else None
    if grid is None:
        return None

    valid = grid >= 0
    dwell = np.where(valid, grid, 0).sum(0)
    n_L = grid.shape[0]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.hist(dwell, bins=np.arange(-0.5, n_L + 1.5, 1), color=CATEGORICAL[0],
            alpha=0.9)
    ax.axvline(n_L / 2.0, **{"color": "#6B7280", "linestyle": ":",
                             "linewidth": 1.2})
    ax.annotate("half the layers — where a coin flip would pile up",
                xy=(n_L / 2.0, 0.96), xycoords=("data", "axes fraction"),
                fontsize=8, color="#6B7280", ha="center", va="top")
    ax.set_xlabel(f"layers spent in hemisphere 1 (of {n_L})")
    ax.set_ylabel("tokens")

    # A crude bimodality read: mass in the outer thirds vs the middle third.
    outer = float(np.mean((dwell < n_L / 3) | (dwell > 2 * n_L / 3)))
    ax.set_title(f"{run.label} — two populations, or one?\n"
                 f"{outer:.0%} of tokens sit in the outer thirds",
                 fontsize=12)
    return save_figure(fig, out_dir, "hemisphere_dwell_histogram")


# ---------------------------------------------------------------------------
# X11 — the run's per-layer table, as one image
# ---------------------------------------------------------------------------

#: The scalars worth putting on one z-scored grid. Deliberately the
#: continuous ones only — a regime label has no z-score and does not belong
#: on a heatmap of standard deviations.
_FINGERPRINT_FIELDS = (
    "bipartition_eigengap", "centroid_angle", "between_half_ip",
    "separation_ratio", "fiedler_boundary_frac", "minority_fraction",
    "asymmetry", "crossing_count", "axis_rotation", "match_overlap",
    "normalized_margin", "cone_n_binding", "cos_axis_pc1",
)


def _run_fingerprint(run: Run, out_dir: Path) -> Optional[Path]:
    """
    Layer × metric heatmap, each metric z-scored down its own column.

    Not a result — a triage tool. Thirteen depth profiles in one image, each
    on its own scale, so an odd layer shows up as a vertical stripe across
    several metrics and you know which figure to open next. Diverging, since
    the plotted quantity after z-scoring is a signed deviation.
    """
    rows, labels = [], []
    for f in _FINGERPRINT_FIELDS:
        vals = run.field(f)
        if not np.isfinite(vals).any():
            continue
        finite = vals[np.isfinite(vals)]
        sd = float(np.std(finite))
        z = (vals - float(np.mean(finite))) / sd if sd > 1e-12 else vals * 0.0
        rows.append(z)
        labels.append(f)
    if not rows:
        return None

    grid = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 2.2))
    im = ax.imshow(grid, aspect="auto", cmap=FIEDLER_CMAP,
                   norm=fiedler_norm(grid), interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("layer")
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.012, label="z (within metric)")
    ax.set_title(f"{run.label} — the run's per-layer table as one fingerprint\n"
                 f"a vertical stripe is a layer that is unusual in several "
                 f"metrics at once", fontsize=12)
    return save_figure(fig, out_dir, "run_fingerprint")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _positions_to_tokens(run: Run) -> List[str]:
    toks = token_strings(run.per_token)
    if toks and len(toks) >= run.n_tokens:
        return toks
    cols = run.particles()
    out = [""] * run.n_tokens
    if cols and "token_str" in cols and "token_position" in cols:
        for pos, s in zip(np.asarray(cols["token_position"]),
                          np.asarray(cols["token_str"])):
            if 0 <= int(pos) < run.n_tokens:
                out[int(pos)] = str(s)
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
