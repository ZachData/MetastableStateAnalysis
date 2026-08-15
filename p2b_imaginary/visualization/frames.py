"""
p2b_imaginary/visualization/frames.py — Block 1b at one (checkpoint, prompt).

The rescaled-frame comparison, drawn. Four frames go in — `original`,
`remove_full`, `remove_signed`, and `remove_rotation` — and exactly two
comparisons come out, because the fourth is an algebraic identity.

THE RULE THIS MODULE ENFORCES. `remove_rotation` applies `e^{−A}` for real
antisymmetric A, which is ORTHOGONAL, and every quantity Block 1b measures is
a function of `X Xᵀ`, which an orthogonal map preserves exactly. So that
frame reproduces the original by construction, `elim_rotation = 0.0` in 35/35
runs was forced before any data was read, and reading it as "rotation is
dynamically neutral" is the withdrawal `status-2b` opens with. It is drawn in
every figure here — hidden controls are worse than visible ones — always
hatched, always labelled "invariance control", and never in a comparison.
F5 exists to show the residual it actually holds to.

THE OTHER RULE. A refusal is never plotted at zero.
`p2b_energy.elimination_rate` returns `None` with a status for four distinct
refusals precisely because the pre-rewrite `_elim_rate` returned the float
`0.0` for all of them, and that value then entered a β majority vote — 90 of
Study B's 243 Pythia runs are `no_violations`, so the phase would have
returned a verdict by vacuity at exactly the checkpoints where the theorem
holds. Every refusal in these figures is a labelled marker on its own row.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from p2b_imaginary.rotational_rescaled import (
    EQUIVALENCE_BAND, ORTHOGONALITY_TOL, RESCALE_OVERFLOW_LIMIT,
)

from .loaders import (
    Checkpoint, Sweep, elim_row, frame_counts, prompt_out, reference_beta,
)
from .style import (
    BLOG_STYLE, CATEGORICAL, FRAME_COLORS, FRAME_ORDER, REFERENCE_LINE,
    REFUSAL_COLOR, STATUS_COLORS, STATUS_MARKERS, TRUNCATED_SPAN, depth_axis,
    frame_style, no_data, note, reference_line, save_figure, subtitle,
)

__all__ = ["generate_frame_figures", "FIGURES"]

FIGURES = ("frame_counts", "elimination_rates", "violation_layers_strip",
           "truncation_ladder", "invariance_control", "sa_decomposition_depth",
           "phase1_cross_check")

def generate_frame_figures(sweep: Sweep, out_dir: Path) -> List[Path]:
    """Every `frames` figure for every scored (checkpoint, prompt) pair."""
    paths: List[Path] = []
    any_scored = False

    for ck in sweep.checkpoints:
        for prompt, js in sorted(ck.block1b.items()):
            if "interpretation" not in js:
                print(f"  frames: skipping {ck.stem} / {prompt} — "
                      f"status {js.get('status', 'missing')!r}")
                continue
            any_scored = True
            d = prompt_out(out_dir, ck, prompt)
            with plt.rc_context(BLOG_STYLE):
                paths.append(_frame_counts(js, ck, prompt, d))
                paths.append(_elimination_rates(js, ck, prompt, d))
                paths.append(_violation_layers_strip(js, ck, prompt, d))
                paths.append(_truncation_ladder(js, ck, prompt, d))
                paths.append(_invariance_control(js, ck, prompt, d))
                paths.append(_sa_decomposition_depth(js, ck, prompt, d))
                paths.append(_phase1_cross_check(js, ck, prompt, d))

    if not any_scored:
        print("  frames: skipping — no Block 1b comparison in this sweep "
              "(--blocks 1a, or no Phase 1 run carried activations)")
    return paths


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def _header(js: dict, ck: Checkpoint, prompt: str) -> str:
    """
    The provenance line every figure in this class carries.

    The FRAME KIND is on it deliberately. Block 1b runs on `activations.npz`,
    which holds L2-normalized vectors — the `l2_sphere` frame, not the LN
    frame attention actually reads. The claim being tested is about the
    operator attention applies, so the frame is a live caveat on every number
    here and `PLAN_2b.md` carries it as a deferred item.
    """
    frame = (js.get("frame") or {}).get("kind", "?")
    beta = reference_beta(js)
    return (f"{ck.label}   ·   prompt {prompt}   ·   β = {beta}   ·   "
            f"frame: {frame}")


def _ordered_frames(js: dict) -> List[str]:
    """Frames present in this record, in `FRAME_KEYS` order then any extras."""
    present = list((js.get("frames") or {}).keys())
    ordered = [k for k in FRAME_ORDER if k in present]
    return ordered + [k for k in present if k not in ordered]


def _is_control(js: dict, key: str) -> bool:
    return bool((js.get("frames") or {}).get(key, {})
                .get("is_invariance_control", key == "remove_rotation"))


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------

def _frame_counts(js: dict, ck: Checkpoint, prompt: str, out: Path) -> Path:
    """
    F1 — violations per frame, and the transitions they are a fraction of.

    `n_transitions_scored` is a first-class output because it is the
    DENOMINATOR: two frames that scored different numbers of transitions are
    not comparable at all, and `frames_comparable` refuses the division.
    Drawing the counts without their denominators is how a rate produced
    entirely by the rank gate looks like a rate produced by the rescaling.
    """
    frames = _ordered_frames(js)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    x = np.arange(len(frames))

    ax = axes[0]
    for i, key in enumerate(frames):
        c = frame_counts(js, key)
        ax.bar([i], [c.get("n_violations", 0)],
               **frame_style(key, _is_control(js, key)))
    ax.set_xticks(x)
    ax.set_xticklabels(frames, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("violations")
    ax.set_title("violations")

    ax = axes[1]
    bottoms = np.zeros(len(frames))
    parts = [("n_transitions_scored", CATEGORICAL[0], "scored"),
             ("n_transitions_gated", CATEGORICAL[3], "gated (rank)"),
             ("n_transitions_nan", REFUSAL_COLOR, "NaN (truncated / no gate)")]
    for key, color, label in parts:
        vals = np.array([frame_counts(js, f).get(key, 0) for f in frames],
                        dtype=float)
        ax.bar(x, vals, bottom=bottoms, color=color, width=0.72, label=label)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(frames, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("transitions")
    ax.set_title("what each frame's count is out of")
    ax.legend(loc="best", fontsize=8)

    scored = {f: frame_counts(js, f).get("n_transitions_scored") for f in frames}
    disagree = len(set(v for k, v in scored.items()
                       if not _is_control(js, k))) > 1
    note(axes[1],
         ("The causal frames scored DIFFERENT numbers of transitions — "
          "elimination_rate refuses this comparison."
          if disagree else
          "The causal frames scored the same transitions — the comparison "
          "is admissible."))

    fig.tight_layout()
    fig.suptitle("Violation counts and their denominators", y=1.01)
    subtitle(fig, _header(js, ck, prompt))
    return save_figure(fig, out, "frame_counts")


# ---------------------------------------------------------------------------
# F2
# ---------------------------------------------------------------------------

def _elimination_rates(js: dict, ck: Checkpoint, prompt: str,
                       out: Path) -> Path:
    """
    F2 — `elim_full` and `elim_signed`, unclipped, with refusals as refusals.

    Two things this figure will not do. It will not clip the rate at zero:
    `analysis_p2.py:153` applies `max(0, n_phase1 − n_rescaled)`, which
    destroys the sign that separates "rescaling had no effect" from
    "rescaling made it worse", and recovering that sign is Phase 2's
    verification item V2 and ALBERT's overcorrection caveat. And it will not
    draw a refusal at zero: a `None` rate gets a marker on the refusal row,
    labelled with which of the four refusals it is.
    """
    row = elim_row(js)
    interp = js.get("interpretation") or {}
    fig, ax = plt.subplots(figsize=(9, 4.4))

    names = list(row)
    ok_any = False
    ax.axhspan(-EQUIVALENCE_BAND, EQUIVALENCE_BAND, color="#9AA0A6",
               alpha=0.14, linewidth=0, zorder=0)
    ax.axhline(0.0, **REFERENCE_LINE)

    for i, name in enumerate(names):
        res = row[name] or {}
        rate, status = res.get("rate"), str(res.get("status", "missing"))
        if rate is None:
            ax.plot([i], [0], marker=STATUS_MARKERS.get(status, "v"),
                    markersize=11, color=STATUS_COLORS.get(status,
                                                           REFUSAL_COLOR),
                    linestyle="none", markeredgecolor="#6B7280", zorder=3)
            ax.annotate(f"REFUSED\n{status}", xy=(i, 0), xytext=(0, 18),
                        textcoords="offset points", ha="center", fontsize=8,
                        color="#6B7280")
        else:
            ok_any = True
            ax.bar([i], [rate], width=0.5, color=CATEGORICAL[i % 2], zorder=2)
            ax.annotate(f"{rate:+.3f}", xy=(i, rate),
                        xytext=(0, 6 if rate >= 0 else -14),
                        textcoords="offset points", ha="center", fontsize=9)

    # The denominators go under the tick label rather than at a fixed offset
    # from y = 0: an offset there collides with the refusal marker on a
    # refused row and with the bar on an admissible one, and which of those
    # happens depends on the data.
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [f"{name}\nn_orig {(row[name] or {}).get('n_original')} → "
         f"n_resc {(row[name] or {}).get('n_rescaled')}" for name in names],
        fontsize=9)
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.annotate(f"±{EQUIVALENCE_BAND} equivalence band",
                xy=(0.01, EQUIVALENCE_BAND),
                xycoords=("axes fraction", "data"), ha="left", va="bottom",
                fontsize=8, color="#6B7280")
    ax.set_ylabel("elimination rate  (unclipped)")
    if not ok_any:
        ax.set_ylim(-1, 1)
    else:
        # Room for the value labels, which sit outside the bar on the side
        # the sign puts them — otherwise a rate at the axis limit loses its
        # own number to the frame.
        ax.margins(y=0.18)
    ax.set_title(f"Verdict: {interp.get('overall', '?')}")
    subtitle(fig, _header(js, ck, prompt))
    note(ax, "Negative means the rescaling made monotonicity WORSE — the "
             "sign Phase 2's max(0, …) destroys. remove_rotation is absent "
             "by design: it is an identity, not a comparison.", outside=True)
    return save_figure(fig, out, "elimination_rates")


# ---------------------------------------------------------------------------
# F3
# ---------------------------------------------------------------------------

def _violation_layers_strip(js: dict, ck: Checkpoint, prompt: str,
                            out: Path) -> Path:
    """
    F3 — which layers violated, per frame, with the unscored depth shaded.

    A frame that truncated at layer 3 and a frame that found no violations
    after layer 3 produce the same count and the same rate. They do not
    produce the same picture: everything past `n_valid_layers` is hatched as
    NOT SCORED, which is a different statement from "no violation here" and
    is the one that makes `elim_signed = 1.0` free.
    """
    frames = _ordered_frames(js)
    depth = max((js["frames"][f].get("n_valid_layers") or 0) for f in frames)
    depth = max(depth, 1)

    fig, ax = plt.subplots(figsize=(10, 0.75 * len(frames) + 2.2))
    for i, key in enumerate(frames):
        fr = js["frames"][key]
        valid = int(fr.get("n_valid_layers") or 0)
        control = _is_control(js, key)
        if valid < depth:
            ax.axhspan(i - 0.4, i + 0.4, xmin=0, xmax=1, color="none")
            ax.add_patch(plt.Rectangle(
                (valid - 0.5, i - 0.4), depth - valid, 0.8, **TRUNCATED_SPAN))
        for L in frame_counts(js, key).get("violation_layers") or []:
            ax.plot([L], [i], marker="|", markersize=20, markeredgewidth=3,
                    color=FRAME_COLORS.get(key, REFUSAL_COLOR))
        ax.annotate(("  [control]" if control else ""), xy=(depth - 0.4, i),
                    fontsize=7.5, color="#6B7280", va="center")

    ax.set_yticks(range(len(frames)))
    ax.set_yticklabels(frames, fontsize=9)
    ax.set_ylim(-0.6, len(frames) - 0.4)
    depth_axis(ax, depth, xlabel="transition (layer L, scored L−1 → L)")
    ax.grid(False)
    ax.set_title("Where each frame's violations are")
    subtitle(fig, _header(js, ck, prompt))
    note(ax, "Hatched depth is NOT SCORED — the frame truncated there. Not "
             "the same as 'no violation'.", outside=True)
    return save_figure(fig, out, "violation_layers_strip")


# ---------------------------------------------------------------------------
# F4
# ---------------------------------------------------------------------------

def _truncation_ladder(js: dict, ck: Checkpoint, prompt: str,
                       out: Path) -> Path:
    """
    F4 — how far each frame got, and how large its cumulative rescaler grew.

    The three ways an elimination rate gets manufactured, separated. Overflow
    (`e^{−S}` diverging) and underflow (`e^{−S}` contracting until rows fall
    below the normalizer's floor, after which every energy is the constant
    1/(2β) and the frame reports zero violations) both show as a short bar
    with a reason; gate divergence shows as a full-length bar with different
    scored counts in F1. `e^{−A}` is orthogonal and can do neither, which is
    why the control is always at full depth.
    """
    frames = _ordered_frames(js)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(frames))

    ax = axes[0]
    depth = max((js["frames"][f].get("n_valid_layers") or 0) for f in frames)
    for i, key in enumerate(frames):
        fr = js["frames"][key]
        ax.barh([i], [fr.get("n_valid_layers") or 0],
                **frame_style(key, _is_control(js, key)))
        if fr.get("truncated"):
            ax.annotate(f"  TRUNCATED — {fr.get('truncation_reason')}",
                        xy=(fr.get("n_valid_layers") or 0, i), fontsize=8,
                        color="#B45B5B", va="center")
    ax.axvline(depth, **REFERENCE_LINE)
    ax.annotate("full depth", xy=(depth, len(frames) - 0.5), fontsize=8,
                color="#6B7280", rotation=90, va="top")
    ax.set_yticks(x)
    ax.set_yticklabels(frames, fontsize=9)
    ax.set_xlabel("layers produced  (n_valid_layers)")
    ax.set_title("how far each frame got")

    ax = axes[1]
    vals = [js["frames"][f].get("r_cum_max_abs_final") for f in frames]
    plotted = [(i, v) for i, v in enumerate(vals) if v is not None and v > 0]
    if plotted:
        ax.bar([i for i, _ in plotted], [v for _, v in plotted],
               color=[FRAME_COLORS.get(frames[i], REFUSAL_COLOR)
                      for i, _ in plotted], width=0.6)
        ax.set_yscale("log")
        ax.axhline(RESCALE_OVERFLOW_LIMIT, **REFERENCE_LINE)
        ax.annotate("overflow limit", xy=(0.99, RESCALE_OVERFLOW_LIMIT),
                    xycoords=("axes fraction", "data"), ha="right",
                    va="bottom", fontsize=8, color="#6B7280")
    else:
        no_data(ax, "no r_cum_max_abs recorded")
    ax.set_xticks(x)
    ax.set_xticklabels(frames, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("max |R_cum|  (log)")
    ax.set_title("how large the rescaler grew")

    fig.tight_layout()
    fig.suptitle("Truncation — the mechanism that makes elim = 1.0 free",
                 y=1.02)
    subtitle(fig, _header(js, ck, prompt))
    note(axes[1], "Only the maximum survives serialization; the growth CURVE "
                  "is data gap G2.", outside=True)
    return save_figure(fig, out, "truncation_ladder")


# ---------------------------------------------------------------------------
# F5
# ---------------------------------------------------------------------------

def _invariance_control(js: dict, ck: Checkpoint, prompt: str,
                        out: Path) -> Path:
    """
    F5 — the identity, measured.

    status-2b's withdrawal in one image. `A = (V − Vᵀ)/2` is real
    antisymmetric, so `e^{−A}` is orthogonal and so is any cumulative product
    of such matrices; every quantity Block 1b measures is a function of
    `X Xᵀ`, and `(XRᵀ)(XRᵀ)ᵀ = X Xᵀ` exactly for `RRᵀ = I`. The residual sits
    around 1e-15 over 24 accumulated layers at d = 1024, against a violation
    threshold of 1e-3 relative — so `elim_rotation = 0.0` was forced in every
    run, on every model, at every β, before any data was read.

    A `identity_broken` status here is a numerical failure of the CONTROL
    (`expm`, or the accumulation), not a finding about rotation.
    """
    inv = js.get("invariance") or {}
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    if not inv:
        no_data(ax, "this record has no invariance control "
                    "(include_invariance_control=False)")
        return save_figure(fig, out, "invariance_control")

    ortho = inv.get("orthogonality") or {}
    items = [
        ("orthogonality residual\nmax |RRᵀ − I|",
         ortho.get("max_residual"), ORTHOGONALITY_TOL,
         "ORTHOGONALITY_TOL"),
        ("worst relative\nenergy difference",
         inv.get("max_relative_energy_difference"), 1e-3,
         "violation threshold (1e-3)"),
    ]
    xs = np.arange(len(items))
    for i, (label, value, thresh, tname) in enumerate(items):
        v = float(value) if value is not None else np.nan
        ax.bar([i], [max(v, 1e-18)], width=0.45, color=CATEGORICAL[0])
        ax.plot([i - 0.35, i + 0.35], [thresh, thresh], color="#B45B5B",
                linewidth=2.0)
        ax.annotate(tname, xy=(i + 0.36, thresh), fontsize=8, color="#B45B5B",
                    va="center")
        ax.annotate(f"{v:.2e}", xy=(i, max(v, 1e-18)), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=9)

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for lab, *_ in items], fontsize=9)
    ax.set_ylabel("magnitude (log)")
    status = inv.get("status", "?")
    ax.set_title(f"The rotation-only frame is an identity — {status}")
    subtitle(fig, _header(js, ck, prompt))
    ax.annotate(f"violation counts match: {inv.get('violation_counts_match')}",
                xy=(0.5, 0.9), xycoords="axes fraction", ha="center",
                fontsize=9, color="#374151")
    note(ax, "A match here is an ARITHMETIC CHECK, not evidence that rotation "
             "is dynamically neutral. That reading is the withdrawn finding.")
    return save_figure(fig, out, "invariance_control")


# ---------------------------------------------------------------------------
# F6
# ---------------------------------------------------------------------------

def _sa_decomposition_depth(js: dict, ck: Checkpoint, prompt: str,
                            out: Path) -> Path:
    """
    F6 — ‖S‖_F, ‖A‖_F and their ratio per layer.

    The structural claim in the norm the rescaled frames actually act in, as
    against S1's spectral one. Both are "how much of OV is rotation" and they
    are not the same number: the Frobenius ratio weighs the operator's
    entries and the energy fraction weighs its eigenvalues, and the gap
    between them is exactly the non-normality S6 measures.
    """
    sa = js.get("sa_decomp") or {}
    ratio = np.asarray(sa.get("per_layer_rotation_ratio_frobenius") or [],
                       dtype=float)
    s_frob = np.asarray(sa.get("per_layer_S_frob") or [], dtype=float)
    a_frob = np.asarray(sa.get("per_layer_A_frob") or [], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1]})
    if not ratio.size:
        for ax in axes:
            no_data(ax, "no sa_decomp in this record")
        return save_figure(fig, out, "sa_decomposition_depth")

    x = np.arange(ratio.size)
    axes[0].plot(x, s_frob, color=CATEGORICAL[1], marker="s", markersize=3.2,
                 label="‖S‖_F  (symmetric)")
    axes[0].plot(x, a_frob, color=CATEGORICAL[0], marker="o", markersize=3.2,
                 label="‖A‖_F  (antisymmetric)")
    axes[0].set_ylabel("Frobenius norm")
    axes[0].legend(loc="best", fontsize=8.5)
    axes[0].set_title("The S / A split in Frobenius norm")

    axes[1].plot(x, ratio, color=CATEGORICAL[2], marker="D", markersize=3.2,
                 linewidth=2.0)
    reference_line(axes[1], 0.5, "0.5 — equal parts")
    axes[1].set_ylabel("‖A‖ / (‖S‖+‖A‖)")
    axes[1].set_ylim(0, 1)
    depth_axis(axes[1], ratio.size)

    mean = sa.get("mean_rotation_ratio_frobenius")
    subtitle(fig, _header(js, ck, prompt)
             + (f"   ·   mean ratio {mean:.4f}" if mean is not None else ""))
    note(axes[1], "This is not the spectral fraction. Entries vs eigenvalues; "
                  "the gap between them is the non-normality.")
    return save_figure(fig, out, "sa_decomposition_depth")


# ---------------------------------------------------------------------------
# F7
# ---------------------------------------------------------------------------

def _phase1_cross_check(js: dict, ck: Checkpoint, prompt: str,
                        out: Path) -> Path:
    """
    F7 — Phase 2b's own count against Phase 1's, on the same run.

    These are EXPECTED to differ: Phase 2b gates on normed effective rank and
    Phase 1 on raw (status-1 defect D1), and the divergence is written into
    the artifact rather than left inside every elimination rate. What the
    figure is for is the size of it — a large disagreement means the GATE is
    doing the work the rescaling is being credited with, which is the third
    of the three ways an elimination rate gets manufactured and the one that
    scales with ‖V‖.
    """
    cross = js.get("phase1_cross_check") or {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1, 1.2]})
    if not cross:
        for ax in axes:
            no_data(ax, "no phase1_cross_check in this record — the run "
                        "bundle carried no energies.json")
        return save_figure(fig, out, "phase1_cross_check")

    # `cross_check_against_phase1` returns {beta: {n_p2b, n_phase1, delta,
    # only_p2b, only_phase1}}; `run_2b` re-keys beta to a string on the way
    # out. The layer SETS are the informative half — two counts can agree
    # while disagreeing about every layer.
    betas = sorted(cross, key=lambda b: float(b))
    x = np.arange(len(betas))

    ax = axes[0]
    ax.bar(x - 0.19, [(cross[b] or {}).get("n_p2b", 0) for b in betas],
           width=0.36, color=CATEGORICAL[0],
           label="Phase 2b  (gates on NORMED effective rank)")
    ax.bar(x + 0.19, [(cross[b] or {}).get("n_phase1", 0) for b in betas],
           width=0.36, color=CATEGORICAL[3],
           label="Phase 1  (gates on RAW effective rank)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"β = {b}" for b in betas])
    ax.set_ylabel("violations on the original frame")
    ax.set_title("how many")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    for i, b in enumerate(betas):
        row = cross[b] or {}
        for L in row.get("only_p2b") or []:
            ax.plot([L], [i + 0.12], marker="|", markersize=16,
                    markeredgewidth=2.6, color=CATEGORICAL[0])
        for L in row.get("only_phase1") or []:
            ax.plot([L], [i - 0.12], marker="|", markersize=16,
                    markeredgewidth=2.6, color=CATEGORICAL[3])
        ax.annotate(f"Δ = {row.get('delta')}", xy=(0.99, i),
                    xycoords=("axes fraction", "data"), ha="right",
                    fontsize=8, color="#6B7280")
    ax.set_yticks(x)
    ax.set_yticklabels([f"β = {b}" for b in betas])
    ax.set_ylim(-0.6, len(betas) - 0.4)
    ax.set_xlabel("layer")
    ax.set_title("which layers each found and the other did not")
    ax.grid(False)

    fig.tight_layout()
    fig.suptitle("Phase 2b's count against Phase 1's, same run", y=1.02)
    subtitle(fig, _header(js, ck, prompt))
    note(axes[1], "A difference is expected — two different gates. A LARGE "
                  "one means the gate is doing the work, not the rescaling.",
         outside=True)
    return save_figure(fig, out, "phase1_cross_check")
