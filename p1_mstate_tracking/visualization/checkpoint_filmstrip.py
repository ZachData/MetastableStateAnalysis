"""
visualization/checkpoint_filmstrip.py

Class-3 figures: per-model snapshots (PCA-by-cluster scatter, sorted
cosine-Gram heatmap) don't sweep — 27 copies of each is noise. This
module implements the two-pass workflow's second pass:

  1. select_snapshot_steps() reads the transitions dict produced by
     checkpoint_scalars (or transitions.json from disk) and picks ~4-6
     checkpoints: both endpoints, plus the steps bracketing the
     top-consensus transition intervals.
  2. The filmstrip renderers draw one row of small multiples across
     those steps, shared style, so the before/during/after contrast is a
     single figure instead of N per-model folders.

Reuses plot_utils._scatter_hdbscan and the loaders' cached artifacts
(pca_trajectories.npz, hdbscan_labels.json, activations.npz) — no new
computation beyond a cosine Gram at one layer.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.style import BLOG_STYLE
from core.naming import _safe_model_name
from .loaders import _pca_trajs, _hdbscan_labels, _load_activations
from core.plot_utils import _scatter_hdbscan, _resolve_layers
from .checkpoints import _fmt_step


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot selection
# ─────────────────────────────────────────────────────────────────────────────

def select_snapshot_steps(
    steps: List[int],
    transitions: Optional[Dict[str, Optional[dict]]] = None,
    k: int = 6,
) -> List[int]:
    """
    Pick <=k snapshot steps from `steps` (the family's available
    checkpoints): both endpoints always; then the (step_lo, step_hi)
    pairs of transition intervals in descending consensus (how many
    metrics agree) then descending jump, until k is reached. Without
    transitions, fall back to endpoints + even spread in rank order.
    """
    steps = sorted(set(int(s) for s in steps))
    if len(steps) <= k:
        return steps
    chosen = {steps[0], steps[-1]}

    if transitions:
        intervals: Dict[Tuple[int, int], List[float]] = {}
        for tr in transitions.values():
            if tr is None:
                continue
            key = (tr["step_lo"], tr["step_hi"])
            intervals.setdefault(key, []).append(tr["normalized_jump"])
        ranked = sorted(
            intervals.items(),
            key=lambda kv: (-len(kv[1]), -max(kv[1])),
        )
        for (lo, hi), _ in ranked:
            for s in (lo, hi):
                if len(chosen) < k and s in steps:
                    chosen.add(s)
            if len(chosen) >= k:
                break

    if len(chosen) < k:   # fill with an even spread over rank order
        idx = np.linspace(0, len(steps) - 1, k).astype(int)
        for i in idx:
            if len(chosen) >= k:
                break
            chosen.add(steps[i])
    return sorted(chosen)


def load_transitions(out_dir: Path, base: str, prompt: str) -> Optional[dict]:
    """Read transitions.json written by checkpoint_scalars, if present."""
    p = out_dir / f"transitions_{_safe_model_name(base)}_{prompt}.json"
    if not p.exists():
        return None
    with open(p) as f:
        payload = json.load(f)
    return payload.get("per_metric")


# ─────────────────────────────────────────────────────────────────────────────
# Filmstrips
# ─────────────────────────────────────────────────────────────────────────────

def _snapshot_runs(
    runs: dict, prompt: str, family: List[Tuple[int, str]],
    snapshot_steps: List[int],
) -> List[Tuple[int, Path]]:
    by_step = {s: m for s, m in family}
    out = []
    for s in snapshot_steps:
        model = by_step.get(s)
        if model is None:
            continue
        rd = runs.get((model, prompt))
        if rd is not None:
            out.append((s, rd))
    return out


def plot_pca_filmstrip(
    runs: dict, out_dir: Path, prompt: str, base: str,
    family: List[Tuple[int, str]], snapshot_steps: List[int],
    layer: Optional[int] = None,
) -> None:
    """
    One row: PCA scatter colored by HDBSCAN labels at a fixed layer, one
    panel per snapshot step. Layer defaults to the deepest layer every
    snapshot has cached ("post-collapse" view). Uses the cached 3-D PCA
    projection (first two components) — per-checkpoint PCA bases differ,
    so panels compare cluster STRUCTURE, not absolute coordinates; the
    caption says so.
    """
    snaps = _snapshot_runs(runs, prompt, family, snapshot_steps)
    if len(snaps) < 2:
        print(f"  ⚠  pca_filmstrip: <2 snapshot runs for {base!r} @ {prompt!r}")
        return

    per_snap = []
    common_layers: Optional[set] = None
    for step, rd in snaps:
        trajs = _pca_trajs(rd)
        labels = _hdbscan_labels(rd)
        avail = set(trajs.keys()) & set(labels.keys())
        if not avail:
            continue
        per_snap.append((step, trajs, labels, avail))
        common_layers = avail if common_layers is None else (common_layers & avail)

    if not per_snap or not common_layers:
        print(f"  ⚠  pca_filmstrip: no common cached layer across snapshots for {base!r}")
        return

    if layer is None or layer not in common_layers:
        layer = max(common_layers)

    plt.rcParams.update(BLOG_STYLE)
    n = len(per_snap)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), squeeze=False)
    for i, (step, trajs, labels, _) in enumerate(per_snap):
        ax = axes[0][i]
        k = _scatter_hdbscan(ax, trajs[layer], labels[layer])
        ax.set_title(f"step {_fmt_step(step)}\nk={k}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)

    fig.suptitle(
        f"Cluster structure across training — PCA at layer {layer}  ·  {base}  ·  {prompt}\n"
        "per-panel PCA basis (structure comparable, coordinates not)  ·  gray × = HDBSCAN noise",
        fontsize=11, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"filmstrip_pca_L{layer}_{_safe_model_name(base)}_{prompt}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


def plot_gram_filmstrip(
    runs: dict, out_dir: Path, prompt: str, base: str,
    family: List[Tuple[int, str]], snapshot_steps: List[int],
    layer: Optional[int] = None,
) -> None:
    """
    One row: token cosine-Gram heatmap at a fixed layer, tokens sorted by
    HDBSCAN label (noise last), one panel per snapshot step. Block
    structure forming along the row is cluster formation made literal.
    Needs activations.npz per snapshot; snapshots without it are skipped.
    """
    snaps = _snapshot_runs(runs, prompt, family, snapshot_steps)
    panels = []
    for step, rd in snaps:
        acts = _load_activations(rd)
        if acts is None:
            continue
        n_layers = acts.shape[0]
        li = _resolve_layers([layer if layer is not None else "final"], n_layers)[0]
        X = np.asarray(acts[li], dtype=float)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-12)
        gram = X @ X.T
        labels = _hdbscan_labels(rd).get(li)
        if labels is not None and len(labels) == gram.shape[0]:
            lab = np.asarray(labels)
            order = np.lexsort((np.arange(lab.size), np.where(lab == -1, 10**6, lab)))
            gram = gram[np.ix_(order, order)]
        panels.append((step, li, gram))

    if len(panels) < 2:
        print(f"  ⚠  gram_filmstrip: <2 snapshots with activations for {base!r}")
        return

    plt.rcParams.update(BLOG_STYLE)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8), squeeze=False)
    im = None
    for i, (step, li, gram) in enumerate(panels):
        ax = axes[0][i]
        im = ax.imshow(gram, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"step {_fmt_step(step)}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    if im is not None:
        cbar = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.01)
        cbar.set_label("cosine similarity", fontsize=9)

    li = panels[0][1]
    fig.suptitle(
        f"Token Gram matrix across training — layer {li}, sorted by cluster  ·  {base}  ·  {prompt}",
        fontsize=11, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"filmstrip_gram_L{li}_{_safe_model_name(base)}_{prompt}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


def generate_filmstrip_figures(
    runs: dict, out_dir: Path, prompt: str, base: str,
    family: List[Tuple[int, str]],
    transitions: Optional[Dict[str, Optional[dict]]] = None,
    k: int = 6, layer: Optional[int] = None,
) -> None:
    steps = [s for s, _ in family]
    if len(steps) < 2:
        print(f"  ⚠  filmstrips: family {base!r} has <2 checkpoints, skipping")
        return
    if transitions is None:
        transitions = load_transitions(out_dir, base, prompt)
    snapshot_steps = select_snapshot_steps(steps, transitions, k=k)
    print(f"  filmstrip snapshots for {base!r}: {snapshot_steps}")
    plot_pca_filmstrip(runs, out_dir, prompt, base, family, snapshot_steps, layer=layer)
    plot_gram_filmstrip(runs, out_dir, prompt, base, family, snapshot_steps, layer=layer)
