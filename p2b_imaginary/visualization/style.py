"""
p2b_imaginary/visualization/style.py

Phase 2b's palette and the drawing primitives every figure module in this
package shares. Imports the project-wide look from `core.style` rather than
restating it — `BLOG_STYLE`, `MODEL_COLORS`, `UNTRAINED_COLOR` are the same
objects Phases 1, 1b, 1c and 2 draw with, so five phases' figures sit in one
document without looking like five projects.

What is new here is the vocabulary Phase 2b has and the other phases do not,
and each colour assignment is chosen by the job it does:

  the four frames
      `original` is the reference and takes the neutral dark slot.
      `remove_full` and `remove_signed` are the two measurements and take
      two separated hues. `remove_rotation` is an ALGEBRAIC IDENTITY, not a
      measurement, and takes gray plus a hatch plus the word "control" in
      every legend it appears in — reading it as a result is how
      `rotation_neutral` became a headline, and the rendering is built to
      make that hard.

  the six verdicts
      three measurement verdicts in real hues, three refusals
      (`no_violations`, `not_comparable`, and the absent case) in the gray
      family. A verdict table that is mostly refusals must not read as a
      table that is mostly findings.

  the five elimination-rate statuses
      `ok` is the only one with a colour. The four refusals share the
      refusal gray and are distinguished by marker, because they are four
      ways of having no number rather than four numbers.

  the elimination rate itself
      diverging, two hues with a neutral midpoint at exactly zero. The rate
      is UNCLIPPED by design (Phase 2's verification item V2): negative means
      the rescaling made monotonicity worse, which is ALBERT's overcorrection
      and a real reading. Centring the colormap anywhere but zero would put
      the neutral colour inside one sign.

Gray (`REFUSAL_COLOR`) is reserved for refusals, controls and absent inputs
and is never a data colour. If a figure shows gray, either something was
missing or something is not a measurement.

THE STEP AXIS. `log10(step+1)`, real checkpoint steps as ticks, step 0 in its
own near-black dotted style — the convention
`p1_mstate_tracking/visualization/checkpoints.py` settled and `p2b_report`
already computes its interval widths in. That module imports `.series`, which
reaches `core/plot_utils.py` and so `sklearn`, for figures Phase 2b has no
interest in; so the helpers are imported lazily here and mirrored locally when
that chain is unavailable. The mirror is pinned by
`tests/test_p2b_viz_smoke.py`, which asserts the two agree elementwise
whenever both are importable — a fallback that can silently disagree is worse
than no fallback. The NAME GRAMMAR is not mirrored: `core/model_family.py` is
stdlib-only and is imported directly, which is why it was moved there
(`PLAN_2b.md` item 5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

from core.style import BLOG_STYLE, MODEL_COLORS, NOISE_COLOR, UNTRAINED_COLOR

__all__ = [
    "BLOG_STYLE", "MODEL_COLORS", "UNTRAINED_COLOR", "NOISE_COLOR",
    "CATEGORICAL", "REFUSAL_COLOR", "CONTROL_HATCH",
    "FRAME_COLORS", "FRAME_ORDER", "FRAME_LABELS", "frame_style",
    "VERDICT_COLORS", "VERDICT_ORDER", "REFUSAL_VERDICTS",
    "STATUS_COLORS", "STATUS_MARKERS",
    "ELIM_CMAP", "SEQ_CMAP", "DIVERGING_CMAP", "signed_norm",
    "NULL_BAND", "REFERENCE_LINE", "EVENT_SPAN", "TRUNCATED_SPAN",
    "save_figure", "reference_line", "depth_axis", "class_strip",
    "legend_from_classes", "no_data", "note", "subtitle",
    "step_x", "step_norm", "step_color", "format_step_axis",
    "add_step_colorbar", "STEP0_STYLE", "CHECKPOINT_CMAP",
    "depth_color", "depth_norm", "add_depth_colorbar",
]


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

#: Fixed categorical order, shared with `p1b_hemisphere/visualization`. Every
#: adjacent pair clears the CVD separation floor and the normal-vision floor
#: in light mode. The ORDER is the mechanism, not decoration — reordering
#: these invalidates the check. Figures using slots 3-5 carry direct labels
#: or an adjacent table, never colour alone.
CATEGORICAL: tuple = (
    "#2A78D6",   # 1 blue
    "#EB6834",   # 2 orange
    "#1BAF7A",   # 3 aqua
    "#EDA100",   # 4 yellow
    "#E87BA4",   # 5 magenta
)

#: Refusals, controls, absent inputs. Never a data colour.
REFUSAL_COLOR = "#B8BCC2"

#: The invariance control's hatch. Applied wherever `remove_rotation` is
#: drawn as a bar or a band, so it is distinguishable from a measurement
#: without reading the legend.
CONTROL_HATCH = "///"


# --- the four frames -------------------------------------------------------
# `FRAME_KEYS` is imported rather than restated: a frame the phase can emit
# and this map has no entry for would render as refusal gray, i.e. a real
# frame displayed as "not a measurement".
def _frame_keys() -> tuple:
    try:
        from p2b_imaginary.rotational_rescaled import FRAME_KEYS
        return tuple(FRAME_KEYS)
    except Exception:      # pragma: no cover - the phase should be importable
        return ("original", "remove_full", "remove_signed", "remove_rotation")


FRAME_ORDER = _frame_keys()

FRAME_COLORS: Dict[str, str] = {
    "original":        "#374151",     # the reference, neutral dark
    "remove_full":     CATEGORICAL[0],
    "remove_signed":   CATEGORICAL[1],
    "remove_rotation": REFUSAL_COLOR,  # identity, not a measurement
}

#: Frame names describe what is REMOVED. Spelling that out in every legend is
#: the cheapest guard against the pre-rewrite reading, where the names
#: described what was kept and `n_rotation_only` sounded like a measurement of
#: rotation.
FRAME_LABELS: Dict[str, str] = {
    "original":        "original (no rescaling)",
    "remove_full":     "remove_full  —  e^{−V}",
    "remove_signed":   "remove_signed  —  e^{−S}",
    "remove_rotation": "remove_rotation  —  e^{−A}  [invariance control]",
}


def frame_style(key: str, is_control: Optional[bool] = None) -> dict:
    """
    Bar/patch kwargs for one frame.

    The control gets the hatch and a lighter edge in every figure that draws
    it, without each figure module deciding that for itself. `is_control` is
    read from the artifact when the caller has it (`is_invariance_control`),
    and falls back to the key — the artifact is authoritative because a future
    frame could be added as a control too.
    """
    control = (key == "remove_rotation") if is_control is None else bool(is_control)
    style = dict(color=FRAME_COLORS.get(key, REFUSAL_COLOR), linewidth=0)
    if control:
        style.update(hatch=CONTROL_HATCH, edgecolor="#6B7280", linewidth=0.6)
    return style


# --- Block 1b verdicts -----------------------------------------------------
# The vocabulary is `rotational_rescaled.VERDICTS`, in full. Three of the six
# are refusals and take the gray family; a verdict missing from this map would
# render as refusal gray, which for a MEASUREMENT verdict would be a silent
# demotion, so the smoke test asserts the map covers the vocabulary.
VERDICT_ORDER = (
    "signed_carries_full_v",
    "signed_exceeds_full_v",
    "full_v_exceeds_signed",
    "both_frames_inert",
    "no_violations",
    "not_comparable",
)

#: The three that are not findings. Drawn in the gray family everywhere.
REFUSAL_VERDICTS = ("no_violations", "not_comparable", "missing")

VERDICT_COLORS: Dict[str, str] = {
    "signed_carries_full_v": CATEGORICAL[0],
    "signed_exceeds_full_v": CATEGORICAL[2],
    "full_v_exceeds_signed": CATEGORICAL[1],
    "both_frames_inert":     "#C9CDD3",
    "no_violations":         "#DDE0E4",
    "not_comparable":        "#9AA0A6",
    "missing":               REFUSAL_COLOR,
}


# --- elimination-rate statuses ---------------------------------------------
# `ok` is the only status with a number behind it. The four refusals share one
# colour and are told apart by marker, because they are four ways of having no
# number rather than four different numbers.
STATUS_COLORS: Dict[str, str] = {
    "ok":                          CATEGORICAL[0],
    "no_violations_to_eliminate":  REFUSAL_COLOR,
    "different_transitions_scored": REFUSAL_COLOR,
    "no_transitions_scored":       REFUSAL_COLOR,
    "different_counting_rule":     REFUSAL_COLOR,
    "missing":                     REFUSAL_COLOR,
}
STATUS_MARKERS: Dict[str, str] = {
    "ok":                          "o",
    "no_violations_to_eliminate":  "s",
    "different_transitions_scored": "X",
    "no_transitions_scored":       "P",
    "different_counting_rule":     "D",
    "missing":                     "v",
}


# --- colormaps -------------------------------------------------------------

#: Diverging, for the UNCLIPPED elimination rate and for signed trajectory
#: deltas. Neutral is a light gray rather than a tint of either pole, so zero
#: reads as "no effect" instead of "a little bit of one".
ELIM_CMAP = LinearSegmentedColormap.from_list(
    "p2b_elim",
    ["#7A1F1D", "#D6483F", "#F0A0A0", "#F0EFEC", "#9EC5F4", "#2A78D6", "#0D366B"],
)
DIVERGING_CMAP = ELIM_CMAP

#: One-hue sequential, for unsigned magnitude (fractions, counts, densities).
SEQ_CMAP = plt.cm.Blues

#: Null bands and reference lines are structural, not series — they stay
#: recessive so a real curve is never confused for its own reference.
NULL_BAND = dict(color="#9AA0A6", alpha=0.18, zorder=0, linewidth=0)
REFERENCE_LINE = dict(color="#6B7280", linestyle=":", linewidth=1.2, zorder=1)

#: A dated event from `p2b_report.KNOWN_TRANSITIONS`, drawn on a step axis.
EVENT_SPAN = dict(color="#FCA5A5", alpha=0.22, zorder=0, linewidth=0)

#: Depth past a frame's `n_valid_layers`. Not "no violation here" — not
#: scored at all, which is a different statement and the one that made
#: `elim_signed = 1.0` free.
TRUNCATED_SPAN = dict(color="#E5E7EB", alpha=0.75, zorder=0, linewidth=0,
                      hatch="\\\\")


def signed_norm(values, vmax: Optional[float] = None) -> TwoSlopeNorm:
    """
    Symmetric diverging normalizer centred on exactly zero.

    Matplotlib centres a colormap on the data's own midpoint, which for a
    table of elimination rates that happen to be mostly positive would put
    the neutral colour at some positive rate — i.e. would draw "no effect"
    somewhere it is not. The rate is unclipped precisely so that its sign
    carries meaning; anchoring the colormap to zero is not a style choice.
    """
    v = np.asarray([x for x in np.ravel(values) if x is not None], dtype=np.float64)
    v = v[np.isfinite(v)]
    m = float(vmax) if vmax is not None else (float(np.max(np.abs(v))) if v.size else 1.0)
    m = m if m > 1e-12 else 1.0
    return TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)


# ---------------------------------------------------------------------------
# The step axis
# ---------------------------------------------------------------------------

def _load_p1_checkpoint_helpers():
    """
    Phase 1's step-axis module, or None.

    Reaching it imports `.series` -> `core/plot_utils.py` -> sklearn, which
    Phase 2b's figures have no use for. Absence is not an error; it selects
    the mirror below.
    """
    try:
        from p1_mstate_tracking.visualization import checkpoints as ck
        return ck
    except Exception:
        return None


_CK = _load_p1_checkpoint_helpers()

#: Step 0 is a separate object (the developmental origin), not the low end of
#: the training sweep. Near-black dotted, never from the colormap.
STEP0_STYLE = (dict(_CK.STEP0_STYLE) if _CK is not None
               else dict(color="#111827", linestyle=":", linewidth=2.0, alpha=0.9))

CHECKPOINT_CMAP = _CK.CHECKPOINT_CMAP if _CK is not None else cm.viridis


def step_x(steps) -> np.ndarray:
    """Axis position for training steps: log10(step+1). Places step 0 at 0."""
    if _CK is not None:
        return _CK._step_x(steps)
    return np.log10(np.asarray(steps, dtype=float) + 1.0)


def step_norm(steps: Sequence[int]) -> Normalize:
    """Colormap normalization over log10(step+1) for the NONZERO steps."""
    if _CK is not None:
        return _CK.step_norm(list(steps))
    nz = [s for s in steps if s > 0] or [1]
    return Normalize(vmin=step_x([min(nz)])[0], vmax=step_x([max(nz)])[0])


def step_color(step: int, norm: Normalize):
    if _CK is not None:
        return _CK.step_color(step, norm)
    return CHECKPOINT_CMAP(norm(step_x([step])[0]))


def format_step_axis(ax, steps: Sequence[int], axis: str = "x",
                     max_ticks: int = 9) -> None:
    """Real training-step labels on a log10(step+1) axis — no synthetic
    decades, so every tick is a checkpoint that exists."""
    if _CK is not None:
        _CK.format_step_axis(ax, list(steps), axis=axis, max_ticks=max_ticks)
        return
    steps = sorted(set(int(s) for s in steps))
    if len(steps) > max_ticks:
        idx = np.unique(np.linspace(0, len(steps) - 1, max_ticks).astype(int))
        steps = [steps[i] for i in idx]
    pos = step_x(steps)
    labels = [_fmt_step(s) for s in steps]
    if axis == "x":
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Training step  (log-spaced axis)")
    else:
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.set_ylabel("Training step  (log-spaced axis)")


def add_step_colorbar(fig, ax, steps: Sequence[int], norm: Normalize) -> None:
    """Colorbar labelled in real training steps."""
    if _CK is not None:
        _CK.add_step_colorbar(fig, ax, list(steps), norm)
        return
    sm = cm.ScalarMappable(norm=norm, cmap=CHECKPOINT_CMAP)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    nz = sorted(s for s in set(steps) if s > 0)
    if len(nz) > 7:
        idx = np.unique(np.linspace(0, len(nz) - 1, 7).astype(int))
        nz = [nz[i] for i in idx]
    cbar.set_ticks(step_x(nz))
    cbar.set_ticklabels([_fmt_step(s) for s in nz])
    cbar.set_label("Training step", fontsize=9)


def _fmt_step(step: int) -> str:
    if step >= 1000 and step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


# ---------------------------------------------------------------------------
# The depth axis
# ---------------------------------------------------------------------------

def depth_norm(n_layers: int) -> Normalize:
    return Normalize(vmin=0, vmax=max(int(n_layers) - 1, 1))


def depth_color(layer: int, norm: Normalize):
    """
    One layer's colour on a depth ramp.

    Plasma rather than viridis so a depth-coloured figure and a
    step-coloured one are never confused for each other — this package draws
    both, sometimes on the same page.
    """
    return cm.plasma(norm(int(layer)))


def add_depth_colorbar(fig, ax, n_layers: int) -> None:
    sm = cm.ScalarMappable(norm=depth_norm(n_layers), cmap=cm.plasma)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("layer", fontsize=9)


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
                   side: str = "right") -> None:
    """
    A named reference value — a threshold, a chance level, a null mean.

    `side` exists because two references close together put their labels on
    top of each other at the same edge; sending one to each edge is cheaper
    than a collision solver and reads the same.
    """
    if axis == "y":
        ax.axhline(y, **REFERENCE_LINE)
        x, ha = (0.995, "right") if side == "right" else (0.005, "left")
        # annotation_clip=False: a label on a reference at the very top of the
        # data range is otherwise clipped and silently loses characters.
        ax.annotate(text, xy=(x, y), xycoords=("axes fraction", "data"),
                    ha=ha, va="bottom", fontsize=8, color="#6B7280",
                    annotation_clip=False)
    else:
        ax.axvline(y, **REFERENCE_LINE)
        ax.annotate(text, xy=(y, 0.98), xycoords=("data", "axes fraction"),
                    ha="left", va="top", fontsize=8, color="#6B7280",
                    rotation=90)


def depth_axis(ax, n_layers: int, xlabel: str = "OV layer") -> None:
    """
    Shared layer-axis treatment: integer ticks, sensible thinning.

    Phase 2b's depth axis indexes OV MATRICES, one per attention layer —
    unlike Phase 1b's, which indexes hidden states and therefore has an
    embedding output at index 0 that is not the same kind of object. There
    is no layer-0 special case here, and saying so is worth a line because
    the two packages' depth axes look identical and are not.
    """
    ax.set_xlim(-0.5, max(int(n_layers) - 1, 0) + 0.5)
    ax.set_xlabel(xlabel)
    n = max(int(n_layers), 1)
    step = 1 if n <= 16 else (2 if n <= 32 else 4)
    ax.set_xticks(range(0, n, step))


def class_strip(ax, labels: Sequence[str], colors: Dict[str, str],
                label: str = "", annotate_runs: bool = True) -> None:
    """
    A per-position categorical band — one cell per layer or per checkpoint.

    Always an annotation beneath a continuous panel sharing the same x axis,
    never a figure on its own. `annotate_runs` writes the class name inside
    any run of 3+ identical cells, which is the direct labelling that makes
    the strip readable without the legend.
    """
    labels = [str(x) for x in labels]
    n = len(labels)
    for i, lab in enumerate(labels):
        ax.axvspan(i - 0.5, i + 0.5, color=colors.get(lab, REFUSAL_COLOR),
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


def legend_from_classes(ax, classes: Iterable[str], colors: Dict[str, str],
                        title: str = "", labels: Optional[Dict[str, str]] = None,
                        hatched: Sequence[str] = (), **kwargs) -> None:
    """
    Legend built from a class vocabulary, in the vocabulary's own order.

    `hatched` names the entries drawn with `CONTROL_HATCH` so the legend swatch
    matches the patch — a control that looks like a measurement in the legend
    is exactly the confusion this package exists to prevent.
    """
    handles: List[mpl.patches.Patch] = []
    for c in classes:
        kw = dict(facecolor=colors.get(c, REFUSAL_COLOR), edgecolor="none",
                  label=(labels or {}).get(c, c))
        if c in hatched:
            kw.update(hatch=CONTROL_HATCH, edgecolor="#6B7280")
        handles.append(mpl.patches.Patch(**kw))
    ax.legend(handles=handles, title=title or None, **kwargs)


def no_data(ax, message: str) -> None:
    """
    Say what is missing, in the panel where it would have been drawn.

    A blank axis and an absent figure look identical in an output directory
    three weeks later. This makes "the sweep was --blocks 1a" a visible
    statement rather than a gap.
    """
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10,
            color="#6B7280", wrap=True, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def note(ax, text: str, loc: str = "lower left", outside: bool = False) -> None:
    """
    A caveat printed on the figure rather than left in a docstring.

    Used for the ones a reader must not miss: the frame Block 1b actually
    ran in, the fact that a control is a control, and `co_movement`'s own
    warning about correlating two quantities that both drift with training.

    `outside` moves it below the axes. Every figure whose axes are a filled
    grid — a heatmap, a verdict matrix, a strip — needs it, because an
    in-axes note there does not cover empty space, it covers a cell, and a
    covered cell reads as a missing one.
    """
    if outside:
        # Figure coordinates, below everything. Axes coordinates would need a
        # per-figure offset guess: the gap under an axes is whatever its tick
        # labels, rotation and xlabel happen to occupy, and getting it wrong
        # puts the caveat on top of the axis it is explaining. `bbox_inches=
        # "tight"` at save time grows the canvas to include this.
        ax.figure.text(0.01, -0.015, text, ha="left", va="top", fontsize=7.5,
                       color="#6B7280", wrap=True)
        return
    x, ha = (0.01, "left") if "left" in loc else (0.99, "right")
    y, va = (0.02, "bottom") if "lower" in loc else (0.98, "top")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7.5,
            color="#6B7280", wrap=True,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E5E7EB",
                      alpha=0.85))


def subtitle(fig, text: str, y: Optional[float] = None) -> None:
    """
    One-line provenance under the figure title.

    A figure with a `suptitle` needs both lines placed together, and a
    multi-panel figure's suptitle sits at whatever y its own call chose. So
    when one is present it is lifted to a fixed height and the provenance
    goes just beneath it; otherwise the provenance takes the usual slot above
    a single panel's own title. `_suptitle` is private, hence the `getattr`
    and the default — a matplotlib that renames it loses the alignment, not
    the figure.
    """
    if y is None:
        sup = getattr(fig, "_suptitle", None)
        if sup is not None:
            # The gap between the two lines is fixed in INCHES, not in figure
            # fractions: this package draws figures from 4 to 20 inches tall,
            # and one fraction that clears a 13pt title on the short ones
            # leaves two inches of white space on the tall ones.
            x0, _ = sup.get_position()
            y = 1.005
            sup.set_position((x0, y + 0.34 / max(fig.get_figheight(), 1.0)))
        else:
            y = 0.965
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8.5,
             color="#6B7280")
