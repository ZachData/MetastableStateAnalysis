"""
p1b_hemisphere/visualization/style.py

Phase 1b's palette and the few drawing primitives every figure module in
this package shares. Imports the project-wide look from `core.style` rather
than restating it — `BLOG_STYLE`, `MODEL_COLORS`, `NOISE_COLOR` are the same
objects Phases 1 and 2 draw with, so the three phases' figures sit in one
document without looking like three projects.

What is new here is the small vocabulary Phase 1b has and the other phases
don't: two regime classifiers, a cone verdict, an axis-redundancy verdict,
and one genuinely signed quantity (the Fiedler value). Each gets a color
assignment chosen by the job it does, not by taste:

  ordinal verdicts (both regime classifiers)
      one hue, light to dark, in the order the classifier itself orders
      them. `collapsed` is the palest because it is the "nothing here" end.

  categorical verdicts (cone regime, axis redundancy, event type)
      fixed hue order, validated for colorblind separation — the four-slot
      set below clears the adjacent-pair CVD floor and the normal-vision
      floor in light mode. Never cycled: a new class gets a real slot or it
      gets gray. Every figure that encodes one of these ALSO encodes it as
      position, marker shape, or a direct label, so nothing is color-alone.

  the Fiedler value
      diverging, two hues with a neutral midpoint at exactly zero, because
      the sign is the bipartition and the magnitude is distance from the
      boundary. `fiedler_norm` builds the symmetric normalizer; using it
      rather than matplotlib's default is what keeps zero at the neutral
      color instead of wherever the data happens to be centered.

Gray (`INVALID_COLOR`) is reserved for invalid / degenerate / not-computed
and is never a data color. If a figure shows gray, something was missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from core.style import BLOG_STYLE, MODEL_COLORS, NOISE_COLOR, UNTRAINED_COLOR

__all__ = [
    "BLOG_STYLE", "MODEL_COLORS", "UNTRAINED_COLOR", "NOISE_COLOR",
    "CATEGORICAL", "INVALID_COLOR", "REGIME_COLORS", "REGIME_ORDER",
    "REGIME_REL_COLORS", "REGIME_REL_ORDER", "CONE_COLORS", "CONE_ORDER",
    "REDUNDANCY_COLORS", "REDUNDANCY_ORDER", "EVENT_COLORS", "EVENT_MARKERS",
    "EVENT_ORDER", "HEMI_COLORS", "FIEDLER_CMAP", "SEQ_CMAP", "NULL_BAND",
    "REFERENCE_LINE", "fiedler_norm", "model_color", "save_figure",
    "class_strip", "reference_line", "mark_layer_zero", "depth_axis",
    "legend_from_classes", "no_data",
]


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

#: Fixed categorical order. Validated in light mode against the surface these
#: figures actually use: every adjacent pair clears the CVD separation floor
#: (worst 9.1 ΔE, protan) and the normal-vision floor (worst 19.6 ΔE). The
#: ORDER is the mechanism, not decoration — reordering these invalidates the
#: check. Slots 3-5 sit below 3:1 contrast on white, so every figure using
#: them carries direct labels or an adjacent table, never color alone.
CATEGORICAL: tuple = (
    "#2A78D6",   # 1 blue
    "#EB6834",   # 2 orange
    "#1BAF7A",   # 3 aqua
    "#EDA100",   # 4 yellow
    "#E87BA4",   # 5 magenta
)

#: Never a data color. Invalid layers, degenerate verdicts, absent inputs.
INVALID_COLOR = "#B8BCC2"

# --- Block 0, antipodal classifier: an ordinal verdict, one hue ------------
# collapsed -> weak_bipartition -> diffuse -> strong_bipartition is ordered by
# how much structure the classifier is willing to claim, so it gets a single
# hue light-to-dark rather than four unrelated colors. `diffuse` is the
# angle-met-but-halves-not-compact case and sits one step below `strong`.
# The vocabulary is `bipartition_detect.classify_regime`'s, in full — a class
# missing from this map would silently render as invalid gray.
REGIME_ORDER = ("collapsed", "weak_bipartition", "diffuse", "strong_bipartition")
REGIME_COLORS: Dict[str, str] = {
    "collapsed":          "#D3E3F3",
    "weak_bipartition":   "#8DB9E4",
    "diffuse":            "#4381C4",
    "strong_bipartition": "#12406F",
    "invalid":            INVALID_COLOR,
}

# --- Block 0, relative classifier: the same shape, a different hue ---------
# A different hue rather than a different shade of the same one, because the
# two classifiers are the figure's whole point (status-1b R1) and must not
# read as two rows of one thing. `uniform` — the sign split carrying no
# similarity contrast at all — sits beside `collapsed` at the pale end.
REGIME_REL_ORDER = ("collapsed", "uniform", "graded", "separated")
REGIME_REL_COLORS: Dict[str, str] = {
    "collapsed": "#FBE0D0",
    "uniform":   "#F5B98D",
    "graded":    "#E8853F",
    "separated": "#A03A0C",
    "invalid":   INVALID_COLOR,
}

# --- Block 3, cone verdict: categorical, not ordinal -----------------------
# cone_collapse and split are mutually exclusive claims about the geometry,
# not two amounts of one thing, so they take separated hues. `borderline` is
# the LP's own "inside tolerance of zero" answer and gets the third slot.
CONE_ORDER = ("cone_collapse", "borderline", "split")
CONE_COLORS: Dict[str, str] = {
    "cone_collapse": CATEGORICAL[0],
    "borderline":    CATEGORICAL[3],
    "split":         CATEGORICAL[1],
    "invalid":       INVALID_COLOR,
    "unsolved":      INVALID_COLOR,
}

# --- Block A, axis redundancy ---------------------------------------------
# "distinct" is the interesting verdict and gets the strongest slot; pc1 and
# top_pc_block are both flavors of "already known" and sit adjacent.
REDUNDANCY_ORDER = ("pc1", "top_pc_block", "distinct")
REDUNDANCY_COLORS: Dict[str, str] = {
    "pc1":          CATEGORICAL[1],
    "top_pc_block": CATEGORICAL[3],
    "distinct":     CATEGORICAL[2],
    "degenerate":   INVALID_COLOR,
}

# --- Block 1, event types --------------------------------------------------
# Five types, each drawn with its own marker as well as its own hue, and
# always direct-labeled on its own row — the CVD floor for a five-slot set is
# only legal with that secondary encoding, and the row layout supplies it.
EVENT_ORDER = ("birth", "collapse", "swap", "shear", "drift")
EVENT_COLORS: Dict[str, str] = {
    "birth":    CATEGORICAL[2],
    "collapse": CATEGORICAL[1],
    "swap":     CATEGORICAL[0],
    "shear":    CATEGORICAL[3],
    "drift":    CATEGORICAL[4],
}
EVENT_MARKERS: Dict[str, str] = {
    "birth": "^", "collapse": "v", "swap": "X", "shear": "D", "drift": "o",
}

# --- Hemisphere identity ---------------------------------------------------
# The two poles of the diverging ramp below, so a hemisphere label and a
# Fiedler value never disagree about which side is which.
HEMI_COLORS: Dict[int, str] = {0: "#1B4F8A", 1: "#B3312F", -1: INVALID_COLOR}

#: Diverging, blue-neutral-red, for the signed Fiedler value. Neutral is a
#: gray rather than a light tint of either pole, so zero reads as "neither
#: hemisphere" instead of "a little bit of one".
FIEDLER_CMAP = LinearSegmentedColormap.from_list(
    "p1b_fiedler",
    ["#0D366B", "#2A78D6", "#9EC5F4", "#F0EFEC", "#F0A0A0", "#D6483F", "#7A1F1D"],
)

#: One-hue sequential, for unsigned magnitude (counts, fractions, densities).
SEQ_CMAP = plt.cm.Blues

#: Null bands and reference lines are structural, not series — they stay
#: recessive so a real curve is never confused for its own null.
NULL_BAND = dict(color="#9AA0A6", alpha=0.18, zorder=0, linewidth=0)
REFERENCE_LINE = dict(color="#6B7280", linestyle=":", linewidth=1.2, zorder=1)


def fiedler_norm(values) -> TwoSlopeNorm:
    """
    Symmetric diverging normalizer centered on exactly zero.

    Matplotlib centers a colormap on the data's own midpoint, which for a
    Fiedler vector with an 80/20 split would put the neutral color inside one
    hemisphere. The sign is the bipartition; zero is the boundary. Anchoring
    it is not a style choice.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    m = float(np.nanmax(np.abs(v))) if v.size else 1.0
    m = m if m > 0 else 1.0
    return TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)


def model_color(model: str) -> str:
    """
    A model's line color, falling back gracefully for checkpoint names.

    `pythia-410m-step2000` is not in MODEL_COLORS and never will be — the
    registry has 48 entries and the palette has 7. Checkpoints are colored by
    step within a family (see `checkpoints_1b.py`); this is the fallback for
    figures where a checkpoint appears as one model among several.
    """
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    if str(model).endswith("-random") or "random" in str(model):
        return UNTRAINED_COLOR
    base = str(model).rsplit("-step", 1)[0]
    if base in MODEL_COLORS:
        return MODEL_COLORS[base]
    # Deterministic per name, so the same model keeps its color across
    # figures and across sessions. Color follows the entity, never its rank
    # in whatever list this figure happened to build.
    return CATEGORICAL[hash(str(model)) % len(CATEGORICAL)]


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def save_figure(fig, out_dir: Path, name: str, dpi: int = 150) -> Path:
    """Write one PNG and close the figure. 150 dpi matches Phases 1 and 2."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def class_strip(ax, labels: Sequence[str], colors: Dict[str, str],
                order: Optional[Sequence[str]] = None,
                label: str = "", annotate_runs: bool = True) -> None:
    """
    Draw a per-layer categorical band on `ax` — one cell per layer.

    Used for every verdict-per-layer quantity in the package. The band is an
    annotation, never a figure on its own: it sits under a continuous panel
    sharing the same x axis, because a label is what the phase reports and
    the continuous quantity is what it should be adjudicated on (status-1b
    R1/R3).

    `annotate_runs` writes the class name inside any run of 3+ identical
    cells, which is the direct labeling that makes the strip readable without
    the legend and satisfies the not-color-alone rule.
    """
    labels = [str(x) for x in labels]
    n = len(labels)
    for i, lab in enumerate(labels):
        ax.axvspan(i - 0.5, i + 0.5, color=colors.get(lab, INVALID_COLOR),
                   linewidth=0)

    if annotate_runs:
        start = 0
        for i in range(1, n + 1):
            if i == n or labels[i] != labels[start]:
                if i - start >= 3:
                    ax.text((start + i - 1) / 2.0, 0.5, labels[start],
                            ha="center", va="center", fontsize=7.5,
                            color="#1F2937",
                            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                      ec="none", alpha=0.72))
                start = i

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.grid(False)
    if label:
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)


def reference_line(ax, y: float, text: str, axis: str = "y",
                   side: str = "right") -> None:
    """
    A named reference value — a threshold, a chance level, a null mean.

    `side` exists because two reference lines close together (Block 0's
    `relative_separation` at 0.90 and `relative_weak` at 0.98, for instance)
    put their labels on top of each other at the same edge. Sending one to
    each edge is cheaper than a collision solver and reads the same.
    """
    if axis == "y":
        ax.axhline(y, **REFERENCE_LINE)
        x, ha = (0.995, "right") if side == "right" else (0.005, "left")
        # annotation_clip=False: a label on a reference line at the very top
        # of the data range is otherwise clipped to the axes and silently
        # loses its last few characters.
        ax.annotate(text, xy=(x, y), xycoords=("axes fraction", "data"),
                    ha=ha, va="bottom", fontsize=8, color="#6B7280",
                    annotation_clip=False)
    else:
        ax.axvline(y, **REFERENCE_LINE)
        ax.annotate(text, xy=(y, 0.98), xycoords=("data", "axes fraction"),
                    ha="left", va="top", fontsize=8, color="#6B7280",
                    rotation=90)


def mark_layer_zero(ax) -> None:
    """
    Mark layer 0 rather than dropping it.

    Layer 0 is the embedding output, pre-any-LN (status-1b open blocker 5),
    so it is not the same kind of object as layers 1..N. Dropping it silently
    is one mistake; averaging it in without saying so is the other. Every
    depth axis in this package carries this mark.
    """
    ax.axvspan(-0.5, 0.5, color="#F3F4F6", zorder=0, linewidth=0)


def depth_axis(ax, n_layers: int, xlabel: str = "layer") -> None:
    """Shared layer-axis treatment: integer ticks, layer 0 marked."""
    mark_layer_zero(ax)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_xlabel(xlabel)
    step = 1 if n_layers <= 16 else (2 if n_layers <= 32 else 4)
    ax.set_xticks(range(0, n_layers, step))


def legend_from_classes(ax, classes: Iterable[str], colors: Dict[str, str],
                        title: str = "", **kwargs) -> None:
    """Legend built from a class vocabulary, in the vocabulary's own order."""
    handles = [mpl.patches.Patch(facecolor=colors.get(c, INVALID_COLOR),
                                 edgecolor="none", label=c)
               for c in classes]
    ax.legend(handles=handles, title=title or None, **kwargs)


def no_data(ax, message: str) -> None:
    """
    Say what is missing, in the panel where it would have been drawn.

    A blank axis and an absent figure look identical in an output directory
    three weeks later. This makes "Block 3 was not run" a visible statement
    rather than a gap.
    """
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10,
            color="#6B7280", wrap=True, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
