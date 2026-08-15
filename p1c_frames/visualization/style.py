"""
p1c_frames/visualization/style.py

Phase 1c's palette and the drawing primitives every figure module here
shares. Imports the project-wide look from `core.style` rather than
restating it — `BLOG_STYLE`, `MODEL_COLORS`, `UNTRAINED_COLOR` are the same
objects Phases 1, 1b and 2 draw with, so four phases' figures sit in one
document without looking like four projects.

What is new here is the vocabulary Phase 1c has and the others do not.

  the three step definitions
      `h_displacement`, `h_calibrated`, `h_attn_only` are not three series
      of one kind: one is MATH.md §8 as written, one is the actual Euler
      step, one is the frame-correct variant with no FFN in it. They get
      three distinct hues AND three distinct linestyles, because the whole
      point of drawing them together (status-1c finding 1) is telling them
      apart, and because A2 is often read at thumbnail size where a hue
      difference at 5.7x separation is not enough.

  the two residuals
      Both are signed with a meaningful zero — "on the null's own schedule"
      — so both take a diverging ramp with a neutral midpoint anchored at
      exactly zero by `residual_norm`. Matplotlib centres on the data's own
      midpoint, which for an all-negative residual would put the neutral
      colour inside "resistance" and draw a boundary that is not there.
      Negative is BLUE and is resistance, in both; the two share a ramp
      deliberately, since the whole argument for the time residual is that
      it says the same thing with more range.

  the null itself
      recessive gray, never a series colour. A null drawn in a data hue is
      one somebody will eventually cite as a measurement. Same for the
      envelope band, which is a bracket over an undecided reduction rather
      than an error bar, and is drawn as a hatched band so it cannot be
      mistaken for a confidence interval.

Gray (`INVALID_COLOR`) is reserved for invalid / unreachable / not-computed
and is never a data colour. `UNREACHABLE_COLOR` is its one loud cousin: the
layers where the observed inner product falls below the null's own starting
point are the strongest resistance signal in the phase, and they must be
visible as a marked category rather than absent as a NaN.
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
    "CATEGORICAL", "INVALID_COLOR", "UNREACHABLE_COLOR",
    "STEP_DEFS", "STEP_COLORS", "STEP_STYLES", "STEP_LABELS",
    "FRAME_ORDER", "FRAME_COLORS", "MODEL_ODE_COLORS", "DEGREE_CMAP",
    "RESIDUAL_CMAP", "SEQ_CMAP", "DEPTH_CMAP", "NULL_LINE", "NULL_BAND",
    "ENVELOPE_BAND", "REFERENCE_LINE", "VERDICT_COLORS",
    "residual_norm", "model_color", "degree_color", "depth_colors",
    "save_figure", "reference_line", "mark_layer_zero", "depth_axis",
    "legend_from_classes", "no_data", "caption", "verdict_box",
    "verdict_word", "mark_nan_spans", "tile_grid",
]


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

#: Fixed categorical order, shared with Phase 1b's package so a Phase 1b and
#: a Phase 1c figure side by side do not assign the same hue to unrelated
#: things in different orders. The ORDER is the mechanism: every adjacent
#: pair clears the CVD separation floor in light mode, and reordering
#: invalidates that check.
CATEGORICAL: tuple = (
    "#2A78D6",   # 1 blue
    "#EB6834",   # 2 orange
    "#1BAF7A",   # 3 aqua
    "#EDA100",   # 4 yellow
    "#E87BA4",   # 5 magenta
)

#: Never a data colour. Not-computed, degenerate, absent input.
INVALID_COLOR = "#B8BCC2"

#: Layers whose observed ip_mean falls below the null's starting point — no
#: corresponding null time exists, so `time_residual` is nan there. Loud on
#: purpose: `gamma_null.py` calls these "the strongest possible resistance
#: signal", and clipping or dropping them renders them as "on schedule".
UNREACHABLE_COLOR = "#7C3AED"

# --- The three step-size definitions ---------------------------------------
# Hue AND linestyle, because these are read together by construction and
# often at thumbnail size. `h_displacement` takes the palest slot: it is the
# definition status-1c finding 1 says understates by ~5.7x, and drawing it
# recessive is the one place this package expresses an opinion — the phase's
# own, stated in its own docs, not this package's.
STEP_DEFS = ("h_displacement", "h_calibrated", "h_attn_only")
STEP_COLORS: Dict[str, str] = {
    "h_displacement": "#9EC5F4",
    "h_calibrated":   "#12406F",
    "h_attn_only":    CATEGORICAL[1],
}
STEP_STYLES: Dict[str, dict] = {
    "h_displacement": dict(linestyle=":",  linewidth=1.8, marker=None),
    "h_calibrated":   dict(linestyle="-",  linewidth=2.4, marker=None),
    "h_attn_only":    dict(linestyle="--", linewidth=2.0, marker="^",
                           markersize=4, markevery=3),
}
STEP_LABELS: Dict[str, str] = {
    "h_displacement": "h_displacement (MATH.md §8 as written)",
    "h_calibrated":   "h_calibrated (the Euler step)",
    "h_attn_only":    "h_attn_only (no FFN — the paper's model)",
}

# --- Sub-experiment D's four frames ----------------------------------------
# Categorical, not ordinal: these are four different claims about which
# manifold the tokens live on, not four amounts of one thing.
FRAME_ORDER = ("l2", "ln_plain", "ln_learned", "functional")
FRAME_COLORS: Dict[str, str] = {
    "l2":         CATEGORICAL[0],
    "ln_plain":   CATEGORICAL[2],
    "ln_learned": CATEGORICAL[1],
    "functional": CATEGORICAL[4],
}

# --- The two ODE models ----------------------------------------------------
# (SA) and (USA) are monotone in beta in OPPOSITE directions
# (beta_reduction.py), so they must never share a colour: a reader who takes
# one for the other gets the sign of the beta-dependence backwards, which is
# the specific error that module exists to prevent.
MODEL_ODE_COLORS: Dict[str, str] = {"sa": "#12406F", "usa": "#A03A0C"}

#: Gegenbauer degree k — an ordinal, one hue light to dark.
DEGREE_CMAP = plt.cm.viridis

#: Diverging, blue-neutral-red, anchored at zero by `residual_norm`. Blue is
#: negative is resistance, in both the vertical and the time-domain
#: residual.
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "p1c_residual",
    ["#0D366B", "#2A78D6", "#9EC5F4", "#F0EFEC", "#F5B98D", "#D6483F",
     "#7A1F1D"],
)

#: One-hue sequential, for unsigned magnitude.
SEQ_CMAP = plt.cm.Blues

#: Depth. Used wherever layer index is encoded as colour rather than as
#: position — the scatters, mostly — and deliberately NOT viridis, so a
#: depth-coloured point cloud is never confused with a degree-coloured one.
DEPTH_CMAP = plt.cm.cividis

#: Nulls, references and brackets are structural, not series. They stay
#: recessive so a measured curve is never confused for its own null.
NULL_LINE = dict(color="#6B7280", linestyle="--", linewidth=1.8, zorder=1)
NULL_BAND = dict(color="#9AA0A6", alpha=0.18, zorder=0, linewidth=0)
#: Hatched, because the beta envelope is a bracket over an undecided
#: reduction (status-1c open item 1) and a plain band reads as an error bar.
ENVELOPE_BAND = dict(facecolor="#D3E3F3", edgecolor="#6B7280", alpha=0.45,
                     hatch="///", linewidth=0.0, zorder=0)
REFERENCE_LINE = dict(color="#6B7280", linestyle=":", linewidth=1.2, zorder=1)

#: Verdict tiles. Keyed by the leading word of the phase's own verdict
#: strings, which is why the vocabulary is imported rather than invented:
#: an unrecognized verdict renders gray and is visibly unclassified rather
#: than silently coloured as something else.
VERDICT_COLORS: Dict[str, str] = {
    "CONFIRMED":  "#1BAF7A",
    "FALSIFIED":  "#D6483F",
    "PARTIAL":    "#EDA100",
    "UNCLEAR":    INVALID_COLOR,
    "MIXED":      "#EDA100",
    "SINKS":      "#EB6834",
    "DIRECTIONAL": "#2A78D6",
    "BELOW":      "#2A78D6",
    "ABOVE":      "#D6483F",
    "INSIDE":     "#EDA100",
    "ROBUST":     "#1BAF7A",
    "STRADDLES":  "#D6483F",
    "LICENSED":   "#1BAF7A",
    "MARGINAL":   "#EDA100",
    "NOT":        "#D6483F",
}


def residual_norm(values, vmax: Optional[float] = None) -> TwoSlopeNorm:
    """
    Symmetric diverging normalizer centred on exactly zero.

    Zero is "on the null's own schedule" — the whole content of the
    quantity. Matplotlib would centre on the data's midpoint, which for an
    all-negative residual puts the neutral colour inside resistance and
    draws a boundary that is not there.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    m = float(vmax) if vmax else (float(np.max(np.abs(v))) if v.size else 1.0)
    m = m if m > 0 else 1.0
    return TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)


def model_color(model: str) -> str:
    """
    A model's line colour, falling back gracefully for checkpoint names.

    `pythia-410m-step2000` is not in MODEL_COLORS and never will be — the
    registry has dozens of entries and the palette has seven. Checkpoints
    are coloured by step within a family (`checkpoints_1c.py`); this is the
    fallback for figures where a checkpoint appears as one model among
    several. Colour follows the entity, never its rank in whatever list this
    figure happened to build.
    """
    model = str(model)
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    if "random" in model:
        return UNTRAINED_COLOR
    base = model.rsplit("-step", 1)[0]
    if base in MODEL_COLORS:
        return MODEL_COLORS[base]
    return CATEGORICAL[hash(model) % len(CATEGORICAL)]


def degree_color(k: int, k_max: int = 3):
    """Colour for Gegenbauer degree k (1-indexed), dark to light."""
    return DEGREE_CMAP(0.15 + 0.7 * (k - 1) / max(k_max - 1, 1))


def depth_colors(n_layers: int):
    """One colour per layer, for depth-encoded scatters."""
    return DEPTH_CMAP(np.linspace(0.08, 0.95, max(int(n_layers), 1)))


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def save_figure(fig, out_dir: Path, name: str, dpi: int = 150) -> Path:
    """Write one PNG and close the figure. 150 dpi matches Phases 1, 1b, 2."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def reference_line(ax, y: float, text: str, axis: str = "y",
                   side: str = "right", **kw) -> None:
    """
    A named reference value — t*, zero, the field's bound of 1, chance.

    `side` exists because two reference lines close together put their
    labels on top of each other at the same edge; sending one to each edge
    is cheaper than a collision solver and reads the same.
    """
    style = dict(REFERENCE_LINE)
    style.update(kw)
    if axis == "y":
        ax.axhline(y, **style)
        x, ha = (0.995, "right") if side == "right" else (0.005, "left")
        # annotation_clip=False: a label on a reference line at the very top
        # of the data range is otherwise clipped and silently loses its last
        # few characters.
        ax.annotate(text, xy=(x, y), xycoords=("axes fraction", "data"),
                    ha=ha, va="bottom", fontsize=8, color="#4B5563",
                    annotation_clip=False)
    else:
        ax.axvline(y, **style)
        va = "top" if side == "right" else "bottom"
        ax.annotate(text, xy=(y, 0.98 if side == "right" else 0.02),
                    xycoords=("data", "axes fraction"),
                    ha="left", va=va, fontsize=8, color="#4B5563",
                    rotation=90, annotation_clip=False)


def mark_layer_zero(ax) -> None:
    """
    Mark layer 0 rather than dropping it.

    Layer 0 is the embedding output, pre-any-LN, so it is not the same kind
    of object as layers 1..N — and in Phase 1c it is also the only layer
    where the null's initial condition is a choice rather than a
    consequence (Theorem 6.8 assumes orthogonal init; embeddings are not).
    Dropping it silently is one mistake; averaging it in without saying so
    is the other.
    """
    ax.axvspan(-0.5, 0.5, color="#F3F4F6", zorder=0, linewidth=0)


def depth_axis(ax, n_layers: int, xlabel: str = "layer") -> None:
    """Shared layer-axis treatment: integer ticks, layer 0 marked."""
    mark_layer_zero(ax)
    ax.set_xlim(-0.5, max(n_layers - 0.5, 0.5))
    ax.set_xlabel(xlabel)
    step = 1 if n_layers <= 16 else (2 if n_layers <= 32 else 4)
    ax.set_xticks(range(0, max(n_layers, 1), step))


def mark_nan_spans(ax, values, color: str = UNREACHABLE_COLOR,
                   label: str = "unreachable") -> int:
    """
    Draw the NaN layers as marked bands, and return how many there were.

    A NaN in this phase is not missing data. `time_residual` is NaN exactly
    where the observed inner product is below the null's own starting point
    — the network de-clustered past where it began — which `gamma_null.py`
    calls the strongest possible resistance signal. A gap in the line would
    render that as nothing at all.
    """
    v = np.asarray(values, dtype=np.float64)
    bad = ~np.isfinite(v)
    for i in np.nonzero(bad)[0]:
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.16, linewidth=0,
                   zorder=0)
    if bad.any() and label:
        ax.axvspan(np.nan, np.nan, color=color, alpha=0.16, linewidth=0,
                   label=f"{label} ({int(bad.sum())})")
    return int(bad.sum())


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
    three weeks later. This makes "sub-experiment D was never wired" a
    visible statement rather than a gap.
    """
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10,
            color="#6B7280", wrap=True, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def caption(fig, text: str, y: float = -0.02) -> None:
    """
    A figure-level caption, for the phase's own verdict strings.

    Every verdict in this package is quoted from `p1c_frames`, never
    paraphrased: a figure that summarises a verdict in its own words is one
    that can drift from it.
    """
    if not text:
        return
    fig.text(0.5, y, text, ha="center", va="top", fontsize=8.5,
             color="#4B5563", wrap=True)


def verdict_word(text: str) -> str:
    """
    The classifying word in one of the phase's verdict strings.

    Not simply the first token. `sphere_license` emits both "SPHERE
    LICENSED" and "SPHERE NOT LICENSED", which share a first word and mean
    opposite things — so the first few tokens are scanned and the first
    RECOGNIZED one wins. "NOT" preceding "LICENSED" is what makes the
    negative case land on the negative colour.

    Returns "" when nothing is recognized, which renders grey: an
    unclassified verdict is visibly unclassified rather than silently
    coloured as something else.
    """
    for token in str(text).split()[:4]:
        word = token.strip(":—-,.()").upper()
        if word in VERDICT_COLORS:
            return word
    return ""


def verdict_box(ax, text: str, loc: str = "upper left",
                word: Optional[str] = None) -> None:
    """
    A verdict quoted inside the axes, tinted by its classifying word.

    `word` overrides the scan for the one verdict whose text carries no
    keyword at all: `integration_time.verdict`'s `reading` begins "T_eff"
    and its classification lives in the separate `robust` flag.
    """
    if not text:
        return
    color = VERDICT_COLORS.get((word or verdict_word(text)).upper(),
                               INVALID_COLOR)
    xy = {"upper left": (0.02, 0.97, "left", "top"),
          "upper right": (0.98, 0.97, "right", "top"),
          "lower left": (0.02, 0.03, "left", "bottom"),
          "lower right": (0.98, 0.03, "right", "bottom")}[loc]
    ax.text(xy[0], xy[1], _wrap(str(text), 58), transform=ax.transAxes,
            ha=xy[2], va=xy[3], fontsize=8, color="#1F2937",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=color, linewidth=1.4, alpha=0.92))


def tile_grid(ax, tiles: Sequence[tuple], n_cols: int = 3) -> None:
    """
    Labelled status tiles — `(title, value, verdict_word)` each.

    Used by the verdict cards. The verdict word decides the tile's edge
    colour and the value is printed inside it, so a tile is never a colour
    alone: an unrecognized verdict is gray AND says what it says.
    """
    n = len(tiles)
    n_rows = int(np.ceil(n / n_cols)) or 1
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, (title, value, word) in enumerate(tiles):
        r, c = divmod(i, n_cols)
        color = VERDICT_COLORS.get(str(word).upper(), INVALID_COLOR)
        ax.add_patch(mpl.patches.FancyBboxPatch(
            (c + 0.04, r + 0.06), 0.92, 0.88,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            facecolor=color, alpha=0.13, edgecolor=color, linewidth=1.6))
        ax.text(c + 0.5, r + 0.30, str(title), ha="center", va="center",
                fontsize=8.5, color="#4B5563")
        ax.text(c + 0.5, r + 0.56, str(value), ha="center", va="center",
                fontsize=13, color="#111827", fontweight="bold")
        ax.text(c + 0.5, r + 0.79, str(word), ha="center", va="center",
                fontsize=8, color=color, fontweight="bold")


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width))
