"""
p1b_hemisphere/visualization/cone.py — Block 3, containment as a quantity.

Six figures (C1-C6). status-1b R3 is the brief: "100% cone-collapse, every
layer, every model" is a binary regime label, and n points in d dimensions
admit a separating witness for free unless they positively span. The label
was never the reportable quantity — `normalized_margin` against a matched
null is.

So C1 is the headline and everything else supports it:

  * C1 plots the margin with the null means overlaid, when `--n-null` was
    run. When it was not, the figure still draws but says in the panel that
    there is nothing to compare against — which is the honest rendering of
    the recorded result.
  * C4 plots `n_binding` and `d_eff` because "how many tokens hold the
    witness, in how many effective dimensions" is the dimension-counting
    question stated directly.
  * C6 pools every run and puts the margin against n_tokens/d_eff, which is
    where a dimension artifact would appear as a trend rather than as a
    verdict.

The LP's own tolerance and the escalation flag come from the artifacts, not
from constants here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .loaders import Run, layer_field
from .style import (
    BLOG_STYLE, CATEGORICAL, CONE_COLORS, CONE_ORDER, INVALID_COLOR,
    NULL_BAND, class_strip, depth_axis, legend_from_classes, model_color,
    no_data, reference_line, save_figure,
)

__all__ = ["generate_cone_figures", "generate_cone_cross_figures"]


def generate_cone_figures(run: Run, out_dir: Path) -> List[Path]:
    """Every per-run Block 3 figure, skipping what the run does not carry."""
    if run.cone is None:
        print(f"    cone: skipped for {run.stem} — "
              f"{_reason(run, 'cone (Block 3')}")
        return []

    with plt.rc_context(BLOG_STYLE):
        paths = [_cone_margin_depth(run, out_dir),
                 _cone_regime_strip(run, out_dir),
                 _cone_binding(run, out_dir)]
        if run.cone_per_layer:
            paths.append(_cone_null_z(run, out_dir))
            paths.append(_cone_witness_tokens(run, out_dir))
        else:
            print(f"    cone: C3/C5 skipped for {run.stem} — "
                  f"{_reason(run, 'cone_per_layer')}")
    return [p for p in paths if p is not None]


def generate_cone_cross_figures(runs: Sequence[Run], out_dir: Path) -> List[Path]:
    """C6 — the one cone figure that needs every run at once."""
    with plt.rc_context(BLOG_STYLE):
        p = _cone_vs_dimension(runs, out_dir)
    return [p] if p is not None else []


# ---------------------------------------------------------------------------
# C1 — the quantity R3 asks for
# ---------------------------------------------------------------------------

def _cone_margin_depth(run: Run, out_dir: Path) -> Path:
    """
    `normalized_margin` vs depth, with the regime band and the nulls.

    The margin is signed: positive is a witness (all tokens in one open
    half-space), negative is a genuine split. Zero is drawn because it is the
    decision boundary, and the null means are drawn as bands because "the
    margin is large" means nothing without "and a shuffled draw's is this
    large".
    """
    n = run.n_layers
    margin = run.field("normalized_margin")
    per_layer = run.cone_per_layer or []

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [4, 1]})
    ax = axes[0]

    null_shuffled = layer_field(per_layer, "null_mean_shuffled") if per_layer else None
    null_uniform = layer_field(per_layer, "null_mean_uniform") if per_layer else None
    has_nulls = (null_uniform is not None and np.isfinite(null_uniform).any())

    if has_nulls:
        ax.fill_between(range(n), 0, null_uniform, label="uniform-sphere null (mean)",
                        **NULL_BAND)
        ax.plot(range(n), null_shuffled, color="#6B7280", linestyle="--",
                linewidth=1.4, label="shuffled-dimension null (mean)")

    ax.plot(range(n), margin, color=CATEGORICAL[0], linewidth=2.4,
            marker="o", markersize=4, label="normalized margin", zorder=4)
    reference_line(ax, 0.0, "0 — split below, containment above")
    ax.set_ylabel("normalized cone margin")
    ax.legend(loc="best", fontsize=8.5)

    if has_nulls:
        note = "matched nulls present (--n-null)"
    else:
        note = ("no matched null in this run — the margin is unreferenced, "
                "which is exactly what status-1b R3 flags")
    ax.set_title(f"{run.label} — containment as a continuous quantity\n{note}",
                 fontsize=12)

    class_strip(axes[1], run.strings("cone_regime"), CONE_COLORS,
                label="cone regime")
    legend_from_classes(axes[1], list(CONE_ORDER) + ["invalid"], CONE_COLORS,
                        loc="upper left", bbox_to_anchor=(1.005, 1.6),
                        fontsize=8)
    depth_axis(axes[1], n)
    return save_figure(fig, out_dir, "cone_margin_depth")


# ---------------------------------------------------------------------------
# C2 — the label, with the caveat attached
# ---------------------------------------------------------------------------

def _cone_regime_strip(run: Run, out_dir: Path) -> Path:
    """
    Regime per layer, with escalations marked.

    A reduced-space cone_collapse verdict lifts exactly to full d and is
    sound; a reduced-space split may be a projection artifact, which is why
    `escalate_on_split` re-solves those at full d. The escalation marks are
    therefore the only places in this strip where the PCA reduction could
    have changed the answer, and they belong on the same picture as the
    labels they qualify.
    """
    n = run.n_layers
    regimes = run.strings("cone_regime")
    escalated = run.field("cone_escalated")

    fig, ax = plt.subplots(figsize=(10, 2.4))
    class_strip(ax, regimes, CONE_COLORS)
    marks = [L for L in range(n) if escalated[L] == 1.0]
    if marks:
        ax.scatter(marks, [0.5] * len(marks), marker="*", s=120,
                   color="#111827", zorder=6,
                   label="re-solved at full d (split under PCA)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8)
    depth_axis(ax, n)

    counts = Counter(regimes)
    tally = " · ".join(f"{k}: {v}" for k, v in counts.most_common())
    ax.set_title(f"{run.label} — cone regime by layer\n{tally}", fontsize=12)
    return save_figure(fig, out_dir, "cone_regime_strip")


# ---------------------------------------------------------------------------
# C3 — the nulls, as z-scores
# ---------------------------------------------------------------------------

def _cone_null_z(run: Run, out_dir: Path) -> Optional[Path]:
    """
    z vs the shuffled-dimension and uniform-sphere nulls, two panels.

    Two panels rather than two y-axes on one — the two nulls answer different
    questions (does the margin survive destroying the geometry within
    dimensions; does it survive an isotropic cloud of the same size) and
    putting them on a shared scale invites reading a crossing that means
    nothing.
    """
    per_layer = run.cone_per_layer or []
    z_shuf = layer_field(per_layer, "z_vs_shuffled")
    z_unif = layer_field(per_layer, "z_vs_uniform")
    if not (np.isfinite(z_shuf).any() or np.isfinite(z_unif).any()):
        return None

    n = run.n_layers
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    for ax, z, label, color in (
            (axes[0], z_unif, "vs uniform-sphere null", CATEGORICAL[0]),
            (axes[1], z_shuf, "vs shuffled-dimension null", CATEGORICAL[1])):
        ax.bar(range(n), z, color=color, width=0.78, alpha=0.9)
        reference_line(ax, 2.0, "2σ")
        ax.set_ylabel(f"z\n({label})")
        ax.axhline(0, color="#374151", linewidth=0.9)

    depth_axis(axes[1], n)
    axes[0].set_title(
        f"{run.label} — is the margin more than dimension counting?\n"
        f"the shuffled null keeps the marginal geometry; the uniform null "
        f"keeps only n and d", fontsize=12)
    return save_figure(fig, out_dir, "cone_null_z")


# ---------------------------------------------------------------------------
# C4 — the witness's support
# ---------------------------------------------------------------------------

def _cone_binding(run: Run, out_dir: Path) -> Path:
    """
    `n_binding` and `d_eff` vs depth, with n_tokens for scale.

    A witness held by three tokens in sixty-four effective dimensions is a
    different object from one held by half the prompt, and the regime label
    calls both `cone_collapse`.
    """
    n = run.n_layers
    binding = run.field("cone_n_binding")
    per_layer = run.cone_per_layer or []
    d_eff = layer_field(per_layer, "d_eff") if per_layer else np.full(n, np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)

    ax = axes[0]
    ax.bar(range(n), binding, color=CATEGORICAL[0], width=0.78, alpha=0.9)
    ax.set_ylabel("tokens binding\nthe witness")
    reference_line(ax, float(np.nanmean(binding)),
                   f"mean {np.nanmean(binding):.1f} of {run.n_tokens} tokens")

    ax = axes[1]
    if np.isfinite(d_eff).any():
        ax.plot(range(n), d_eff, color=CATEGORICAL[2], linewidth=2.2,
                marker="o", markersize=3.5, label="effective dimension")
        reference_line(ax, float(run.n_tokens), f"n_tokens = {run.n_tokens}")
        ax.set_ylabel("d_eff")
        ax.legend(loc="best")
    else:
        no_data(ax, "d_eff not in this run's artifacts "
                    "(predates the per-layer cone emission)")
    depth_axis(ax, n)

    axes[0].set_title(f"{run.label} — how much of the prompt holds the cone",
                      fontsize=12)
    return save_figure(fig, out_dir, "cone_binding")


# ---------------------------------------------------------------------------
# C5 — who holds it
# ---------------------------------------------------------------------------

def _cone_witness_tokens(run: Run, out_dir: Path) -> Optional[Path]:
    """
    Which token positions are binding, and how often across layers.

    The LP's binding set is the support of the containment condition: these
    are the tokens whose removal would change the answer. If the same handful
    binds at every layer, "all tokens lie in one open hemisphere" is a
    statement about those tokens and everything else is interior.
    """
    per_layer = run.cone_per_layer or []
    if not per_layer:
        return None

    n_L, n_T = run.n_layers, run.n_tokens
    grid = np.zeros((n_L, n_T), dtype=float)
    any_binding = False
    for entry in per_layer:
        L = int(entry.get("layer", -1))
        if not (0 <= L < n_L):
            continue
        for t in entry.get("binding_tokens") or []:
            if 0 <= int(t) < n_T:
                grid[L, int(t)] = 1.0
                any_binding = True
    if not any_binding:
        return None

    counts = grid.sum(0)
    order = np.argsort(-counts)
    top = [int(t) for t in order[:24] if counts[t] > 0]
    tokens = _token_strings(run)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4),
                             gridspec_kw={"height_ratios": [1.5, 1]})

    ax = axes[0]
    ax.imshow(grid, aspect="auto", cmap="Blues", origin="lower",
              interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel("token position")
    ax.set_ylabel("layer")
    ax.grid(False)
    ax.set_title(f"{run.label} — the cone's support, layer by layer\n"
                 f"a filled cell is a token binding the LP witness at that layer",
                 fontsize=12)

    ax = axes[1]
    if top:
        labels = [f"{repr(tokens[t])[1:-1]}  @{t}" if t < len(tokens) else f"@{t}"
                  for t in top]
        ax.barh(range(len(top)), [counts[t] for t in top],
                color=CATEGORICAL[0], alpha=0.9)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("layers where this token binds")
    else:
        no_data(ax, "no binding tokens recorded")
    return save_figure(fig, out_dir, "cone_witness_tokens")


# ---------------------------------------------------------------------------
# C6 — the dimension question, pooled
# ---------------------------------------------------------------------------

def _cone_vs_dimension(runs: Sequence[Run], out_dir: Path) -> Optional[Path]:
    """
    Margin against n_tokens / d_eff, every layer of every run.

    If containment is geometry, the margin should not care much about how
    crowded the space is. If it is dimension counting, it falls as n/d rises,
    and the cloud tilts. This is the question status-1b R3 says was never
    established, in the one form that pools every run's evidence.
    """
    xs, ys, colors, labels = [], [], [], []
    for run in runs:
        per_layer = run.cone_per_layer or []
        if not per_layer:
            continue
        d_eff = layer_field(per_layer, "d_eff")
        margin = layer_field(per_layer, "normalized_margin")
        ok = np.isfinite(d_eff) & np.isfinite(margin) & (d_eff > 0)
        if not ok.any():
            continue
        xs.append(run.n_tokens / d_eff[ok])
        ys.append(margin[ok])
        colors.append(model_color(run.model))
        labels.append(run.label)

    if not xs:
        print("    cone: C6 skipped — no run carries per-layer d_eff")
        return None

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for x, y, c, lab in zip(xs, ys, colors, labels):
        ax.scatter(x, y, s=26, color=c, alpha=0.75, edgecolor="white",
                   linewidth=0.5, label=lab)
    reference_line(ax, 0.0, "0 — split below")
    ax.set_xlabel("n_tokens / d_eff  (how crowded the space is)")
    ax.set_ylabel("normalized cone margin")
    ax.legend(loc="best", fontsize=7.5, ncol=1)

    allx = np.concatenate(xs)
    ally = np.concatenate(ys)
    if allx.size >= 3 and np.ptp(allx) > 0:
        r = float(np.corrcoef(allx, ally)[0, 1])
        note = f"pooled Pearson r = {r:+.2f} over {allx.size} layers"
    else:
        note = f"{allx.size} layers"
    ax.set_title("Is containment dimension counting?\n" + note, fontsize=12)
    return save_figure(fig, out_dir, "cone_vs_dimension")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_strings(run: Run) -> List[str]:
    cols = run.particles()
    if cols and "token_str" in cols and "token_position" in cols:
        toks = [""] * run.n_tokens
        for pos, s in zip(np.asarray(cols["token_position"]),
                          np.asarray(cols["token_str"])):
            if 0 <= int(pos) < run.n_tokens:
                toks[int(pos)] = str(s)
        return toks
    return [str(e.get("token_str") or "") for e in run.per_token]


def _reason(run: Run, prefix: str) -> str:
    for m in run.missing:
        if m.startswith(prefix):
            return m
    return "input absent"
