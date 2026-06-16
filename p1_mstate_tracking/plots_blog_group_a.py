"""
p1_mstate_tracking/plots_blog.py
Blog-quality figure generation for Phase 1 results.
Loads exclusively from saved artifacts in results/phase1/ — no model weights.

Usage via CLI (blog_runner.py):
    python -m p1_mstate_tracking.blog_runner --results_dir results/phase1 --out blog_figures

Or imported directly:
    from p1_mstate_tracking.plots_blog import generate_group_A, discover_runs
    from pathlib import Path
    runs = discover_runs(Path("results/phase1"))
    generate_group_A(runs, Path("blog_figures"))

Outputs written so far (Group A):
    group_A_glance.png        mass-near-1 universality across all models/prompts
    group_A1_histograms.png   IP histogram migration — paper Fig 1 replica
    group_A1_mass_all.png     mass-near-1 vs layer, all models
    group_A2_rank.png         effective rank collapse, trained vs untrained
    group_A3_hdbscan.png      PCA scatter + k-count, ALBERT-xlarge wiki_paragraph
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

MODEL_COLORS: Dict[str, str] = {
    "albert-base-v2":    "#2563EB",
    "albert-xlarge-v2":  "#7C3AED",
    "bert-base-uncased": "#059669",
    "gpt2-xl":           "#DC2626",
    "gpt2-large":        "#EA580C",
    "gpt2-medium":       "#D97706",
    "gpt2":              "#CA8A04",
}
UNTRAINED_COLOR = "#9CA3AF"
PLATEAU_COLOR   = "#FEF08A"
DEGENERATE_RANK = 2.0
EXCLUDE_PROMPTS = {"repeated_tokens"}

BLOG_STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E5E7EB",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.7,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    12,
    "legend.fontsize":   9,
    "legend.framealpha": 0.92,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
}


# ─────────────────────────────────────────────────────────────────────────────
# Model-name helpers
# ─────────────────────────────────────────────────────────────────────────────

def _color(model: str) -> str:
    mn = model.replace("-", "_").lower()
    for k, c in MODEL_COLORS.items():
        if k.replace("-", "_") in mn:
            return c
    return "#374151"

def _is_untrained(model: str) -> bool:
    return "random" in model.lower()

def _base(model: str) -> str:
    """'albert-base-v2@48iter' → 'albert-base-v2'"""
    return model.split("@")[0]

def _iter_depth(model: str) -> Optional[int]:
    if "@" in model and "iter" in model:
        try:
            return int(model.split("@")[1].replace("iter", ""))
        except ValueError:
            pass
    return None

def _label(model: str) -> str:
    base = _base(model)
    depth = _iter_depth(model)
    suffix = f"@{depth}iter" if depth else ""
    tag = " [untrained]" if _is_untrained(model) else ""
    return f"{base}{suffix}{tag}"


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loaders
# ─────────────────────────────────────────────────────────────────────────────

def discover_runs(results_dir: Path) -> Dict[Tuple[str, str], Path]:
    """
    Scan results_dir and return {(model, prompt): run_dir} for every saved run.
    Reads geometry.json from each subdir; skips silently on errors.
    """
    runs: Dict[Tuple[str, str], Path] = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        geo_file = d / "geometry.json"
        if not geo_file.exists():
            continue
        try:
            with open(geo_file) as f:
                geo = json.load(f)
            model  = geo.get("model", d.name)
            prompt = geo.get("prompt", "")
            runs[(model, prompt)] = d
        except Exception:
            continue
    return runs

def _geo(run_dir: Path) -> dict:
    with open(run_dir / "geometry.json") as f:
        return json.load(f)

def _clustering(run_dir: Path) -> dict:
    p = run_dir / "clustering.json"
    return json.load(open(p)) if p.exists() else {}

def _trajectory(run_dir: Path) -> dict:
    p = run_dir / "trajectory.json"
    return json.load(open(p)) if p.exists() else {}

def _energies(run_dir: Path) -> dict:
    p = run_dir / "energies.json"
    return json.load(open(p)) if p.exists() else {}

def _sinkhorn(run_dir: Path) -> dict:
    p = run_dir / "sinkhorn.json"
    return json.load(open(p)) if p.exists() else {}

def _hdbscan_labels(run_dir: Path) -> Dict[int, List[int]]:
    """Returns {layer_idx: [int labels]} from hdbscan_labels.json."""
    p = run_dir / "hdbscan_labels.json"
    if not p.exists():
        return {}
    raw = json.load(open(p))
    return {int(k): v for k, v in raw.items()}

def _pca_trajs(run_dir: Path) -> Dict[int, np.ndarray]:
    """Returns {layer_idx: (n_tokens, 3)} from pca_trajectories.npz."""
    p = run_dir / "pca_trajectories.npz"
    if not p.exists():
        return {}
    data = np.load(p)
    out  = {}
    for key in data.files:
        parts = key.split("_")
        if len(parts) >= 2:
            try:
                out[int(parts[-1])] = data[key]
            except ValueError:
                pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared plot utilities
# ─────────────────────────────────────────────────────────────────────────────

def _spans(layer_list: List[int]) -> List[Tuple[int, int]]:
    if not layer_list:
        return []
    ll = sorted(set(layer_list))
    spans, start = [], ll[0]
    for i in range(1, len(ll)):
        if ll[i] != ll[i - 1] + 1:
            spans.append((start, ll[i - 1]))
            start = ll[i]
    spans.append((start, ll[-1]))
    return spans

def _shade_plateaus(ax, plateau_layers: List[int], alpha: float = 0.18):
    for s, e in _spans(plateau_layers):
        ax.axvspan(s - 0.5, e + 0.5, color=PLATEAU_COLOR, alpha=alpha, zorder=0)

def _annotation_box(ax, text: str, xy, fontsize: int = 10):
    ax.annotate(
        text, xy=xy, fontsize=fontsize, color="#374151",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#D1D5DB", alpha=0.92),
    )

def _pick_run(runs, base_model: str, prompt: str,
              untrained: bool = False) -> Optional[Tuple[str, Path]]:
    """
    Find the run for (base_model, prompt), preferring highest iter depth.
    Returns (model_name, run_dir) or None.
    """
    candidates = [
        (model, run_dir)
        for (model, p), run_dir in runs.items()
        if _base(model) == base_model and p == prompt
        and _is_untrained(model) == untrained
    ]
    if not candidates:
        return None
    # Prefer highest iter depth; fall back to mtime order (already sorted)
    candidates.sort(key=lambda t: _iter_depth(t[0]) or 0, reverse=True)
    return candidates[0]


# ─────────────────────────────────────────────────────────────────────────────
# Group A — Do clusters form?
# ─────────────────────────────────────────────────────────────────────────────

def group_A_glance(runs: dict, out_dir: Path) -> None:
    """
    mass-near-1 vs normalized layer depth for every (model, prompt) run.
    Trained = solid colored; untrained = dashed gray.
    Bold representative line for wiki_paragraph per model.

    Finding: clustering is universal. Untrained controls cluster harder
    than their trained counterparts.
    """
    plt.rcParams.update(BLOG_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5.2))

    bold_plotted: set = set()
    legend_handles: Dict[str, mpatches.Patch] = {}

    for (model, prompt), run_dir in sorted(runs.items()):
        if prompt in EXCLUDE_PROMPTS:
            continue
        try:
            geo = _geo(run_dir)
        except Exception:
            continue
        layers = geo.get("layers", [])
        if not layers:
            continue

        mass   = [lr.get("ip_mass_near_1", np.nan) for lr in layers]
        n      = len(mass)
        x      = np.linspace(0, 1, n)
        utr    = _is_untrained(model)
        color  = UNTRAINED_COLOR if utr else _color(model)
        base_m = _base(model)

        # Thin background line for every prompt
        ax.plot(x, mass,
                color=color, linewidth=0.8, alpha=0.22,
                linestyle="--" if utr else "-", zorder=2)

        # Thicker representative line for wiki_paragraph (one per base model)
        if prompt == "wiki_paragraph" and base_m not in bold_plotted:
            lw    = 2.6 if not utr else 2.0
            alpha = 0.92 if not utr else 0.75
            ax.plot(x, mass,
                    color=color, linewidth=lw, alpha=alpha,
                    linestyle="--" if utr else "-", zorder=4)
            bold_plotted.add(base_m)
            tag = " (untrained)" if utr else ""
            legend_handles[base_m + tag] = mpatches.Patch(
                color=color,
                label=f"{base_m}{tag}",
                linestyle="--" if utr else "-",
            )

    ax.set_xlabel("Normalized layer depth  (0 = embedding, 1 = final layer)")
    ax.set_ylabel("Fraction of token pairs with ⟨xᵢ, xⱼ⟩ > 0.9")
    ax.set_title(
        "Group A — Token clustering is universal and architectural",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.06)
    ax.axhline(0, color="#D1D5DB", linewidth=0.6)

    # Key finding annotation
    _annotation_box(
        ax,
        "Every model clusters.\nUntrained controls collapse\nfaster and harder.",
        xy=(0.68, 0.12),
    )

    if legend_handles:
        ax.legend(
            handles=list(legend_handles.values()),
            loc="upper left", ncol=2, fontsize=8,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "group_A_glance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


def group_A1_histograms(
    runs: dict,
    out_dir: Path,
    focus_model: str = "albert-xlarge-v2",
    focus_prompt: str = "wiki_paragraph",
    n_panels: int = 8,
) -> None:
    """
    8-panel inner-product histogram evolution.
    Default: ALBERT-xlarge@48iter / wiki_paragraph — direct paper Figure 1 replica.

    Each panel shows ⟨xᵢ,xⱼ⟩ at one layer.  Bars left of 0.9 are light blue;
    bars at or above 0.9 are dark blue (the clustering signal).
    The red dashed line at 0.9 marks the mass-near-1 threshold.
    """
    plt.rcParams.update(BLOG_STYLE)

    result = _pick_run(runs, focus_model, focus_prompt, untrained=False)
    if result is None:
        print(f"  ✗  group_A1_histograms: no run for {focus_model}/{focus_prompt}")
        return
    model_name, run_dir = result

    try:
        geo = _geo(run_dir)
    except Exception as e:
        print(f"  ✗  group_A1_histograms: {e}")
        return

    layers_data = geo.get("layers", [])
    n_layers    = len(layers_data)
    indices     = np.linspace(0, n_layers - 1, n_panels, dtype=int)

    bins        = np.linspace(-1, 1, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_w       = bins[1] - bins[0]

    cols = 4
    rows = n_panels // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6.5), sharey=False)
    axes = axes.flatten()
    fig.suptitle(
        f"A.1 — ⟨xᵢ,xⱼ⟩ histogram migration  ·  {model_name} | {focus_prompt}\n"
        "Each panel: one layer.  Spike moves right as tokens cluster.",
        fontsize=11, fontweight="bold",
    )

    for ax, li in zip(axes, indices):
        lr    = layers_data[li]
        mass  = lr.get("ip_mass_near_1", np.nan)
        hist  = lr.get("ip_histogram", [])
        if not hist:
            ax.text(0.5, 0.5, f"Layer {li}\n(no data)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="#9CA3AF")
            continue

        counts = np.array(hist, dtype=float)
        total  = counts.sum()
        if total > 0:
            counts /= total

        bar_colors = ["#93C5FD" if c < 0.88 else "#1D4ED8"
                      for c in bin_centers]
        ax.bar(bin_centers, counts, width=bin_w,
               color=bar_colors, edgecolor="none", alpha=0.85)
        ax.axvline(0.9, color="#EF4444", linewidth=0.9,
                   linestyle="--", alpha=0.8, label="0.9 threshold")

        mass_str = f"{mass:.2f}" if not np.isnan(mass) else "n/a"
        ax.set_title(f"Layer {li}  (mass>{0.9:.1f} = {mass_str})", fontsize=8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("⟨xᵢ, xⱼ⟩", fontsize=7)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)

    # Arrow annotation on the last occupied panel
    last_ax = axes[len(indices) - 1]
    last_ax.annotate(
        "← spike migrates\n   toward +1",
        xy=(0.65, 0.75), xycoords="axes fraction",
        fontsize=7, color="#1D4ED8", ha="left",
        arrowprops=None,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "group_A1_histograms.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


def group_A1_mass_all(runs: dict, out_dir: Path) -> None:
    """
    mass-near-1 vs raw layer index, one line per (model, prompt).
    Intended as a companion panel to A1_histograms: shows the same
    signal numerically across all runs, making universality explicit.
    """
    plt.rcParams.update(BLOG_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"wspace": 0.35})
    ax_tr, ax_un = axes
    fig.suptitle(
        "A.1b — mass-near-1 across all runs  "
        "(left: trained  |  right: trained vs. untrained)",
        fontsize=11, fontweight="bold",
    )

    legend_tr: Dict[str, mpatches.Patch] = {}
    untrained_lines_plotted: set = set()

    # Left panel: trained models only, all prompts
    for (model, prompt), run_dir in sorted(runs.items()):
        if prompt in EXCLUDE_PROMPTS or _is_untrained(model):
            continue
        try:
            layers = _geo(run_dir).get("layers", [])
        except Exception:
            continue
        if not layers:
            continue
        mass  = [lr.get("ip_mass_near_1", np.nan) for lr in layers]
        color = _color(model)
        base  = _base(model)
        ax_tr.plot(range(len(mass)), mass,
                   color=color, linewidth=1.1, alpha=0.4, zorder=2)
        # Bold line for wiki_paragraph
        if prompt == "wiki_paragraph":
            ax_tr.plot(range(len(mass)), mass,
                       color=color, linewidth=2.4, alpha=0.9, zorder=3,
                       label=_label(model))
        if base not in legend_tr:
            legend_tr[base] = mpatches.Patch(color=color, label=base)

    ax_tr.set_xlabel("Layer")
    ax_tr.set_ylabel("Fraction of pairs  ⟨xᵢ,xⱼ⟩ > 0.9")
    ax_tr.set_title("Trained models — all prompts", fontsize=10)
    ax_tr.set_ylim(-0.02, 1.06)
    if legend_tr:
        ax_tr.legend(handles=list(legend_tr.values()),
                     loc="upper left", fontsize=8, ncol=1)

    # Right panel: trained vs untrained for ALBERT-base and gpt2-large
    TARGET_PAIRS = [
        ("albert-base-v2",   "wiki_paragraph"),
        ("gpt2-large",       "wiki_paragraph"),
    ]
    for base_m, tgt_prompt in TARGET_PAIRS:
        for utr in (False, True):
            res = _pick_run(runs, base_m, tgt_prompt, untrained=utr)
            if res is None:
                continue
            model_name, run_dir = res
            try:
                layers = _geo(run_dir).get("layers", [])
            except Exception:
                continue
            if not layers:
                continue
            mass  = [lr.get("ip_mass_near_1", np.nan) for lr in layers]
            color = UNTRAINED_COLOR if utr else _color(base_m)
            ls    = "--" if utr else "-"
            lbl   = f"{base_m} {'[untrained]' if utr else ''}"
            ax_un.plot(range(len(mass)), mass,
                       color=color, linewidth=2.2, alpha=0.88,
                       linestyle=ls, label=lbl, zorder=3)

    ax_un.set_xlabel("Layer")
    ax_un.set_ylabel("Fraction of pairs  ⟨xᵢ,xⱼ⟩ > 0.9")
    ax_un.set_title("Trained vs untrained (ALBERT-base, GPT-2-large)", fontsize=10)
    ax_un.set_ylim(-0.02, 1.06)
    ax_un.legend(loc="upper left", fontsize=9)
    _annotation_box(
        ax_un,
        "Untrained collapses\nfaster — clustering\nis architectural",
        xy=(0, 0.55),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "group_A1_mass_all.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


def group_A2_rank(runs: dict, out_dir: Path) -> None:
    """
    Effective rank vs layer.
    Left: all trained models on wiki_paragraph.
    Right: ALBERT-base trained vs untrained (normalized depth).
    Shows rank collapsing in proportion with clustering; untrained faster.
    """
    plt.rcParams.update(BLOG_STYLE)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5),
                                     gridspec_kw={"wspace": 0.35})
    fig.suptitle("A.2 — Effective rank collapses as tokens cluster",
                 fontsize=13, fontweight="bold", pad=12)

    TARGET_PROMPT = "wiki_paragraph"
    legend_l: Dict[str, mpatches.Patch] = {}

    # ── Left: all trained, wiki_paragraph ───────────────────────────────────
    for (model, prompt), run_dir in sorted(runs.items()):
        if prompt != TARGET_PROMPT or _is_untrained(model):
            continue
        try:
            layers = _geo(run_dir).get("layers", [])
        except Exception:
            continue
        if not layers:
            continue
        rank  = [lr.get("effective_rank", np.nan) for lr in layers]
        color = _color(model)
        base  = _base(model)
        depth = _iter_depth(model)
        lbl   = f"{base}@{depth}iter" if depth else base
        ax_l.plot(range(len(rank)), rank, color=color, linewidth=2.0,
                  alpha=0.85, label=lbl)
        if base not in legend_l:
            legend_l[base] = mpatches.Patch(color=color, label=lbl)

    ax_l.axhline(DEGENERATE_RANK, color="#EF4444", linewidth=0.9,
                 linestyle="--", alpha=0.6, zorder=5,
                 label=f"Degeneracy gate (rank={DEGENERATE_RANK:.0f})")
    ax_l.set_xlabel("Layer")
    ax_l.set_ylabel("Effective rank")
    ax_l.set_title(f"All trained models — {TARGET_PROMPT}", fontsize=10)
    ax_l.legend(fontsize=8, loc="upper right")

    # ── Right: ALBERT-base trained vs untrained (normalized depth) ──────────
    for base_m in ("albert-base-v2",):
        for utr in (False, True):
            res = _pick_run(runs, base_m, TARGET_PROMPT, untrained=utr)
            if res is None:
                continue
            model_name, run_dir = res
            try:
                layers = _geo(run_dir).get("layers", [])
            except Exception:
                continue
            if not layers:
                continue
            rank   = [lr.get("effective_rank", np.nan) for lr in layers]
            n      = len(rank)
            x_norm = np.linspace(0, 1, n)
            color  = UNTRAINED_COLOR if utr else _color(base_m)
            ls     = "--" if utr else "-"
            lw     = 2.4
            lbl    = ("Untrained (random weights)"
                      if utr else f"{_label(model_name)}")
            ax_r.plot(x_norm, rank, color=color, linewidth=lw,
                      linestyle=ls, alpha=0.9, label=lbl, zorder=3)

    ax_r.axhline(DEGENERATE_RANK, color="#EF4444", linewidth=0.9,
                 linestyle="--", alpha=0.6, zorder=5,
                 label=f"Degeneracy gate (rank={DEGENERATE_RANK:.0f})")
    ax_r.set_xlabel("Normalized layer depth")
    ax_r.set_ylabel("Effective rank")
    ax_r.set_title("ALBERT-base: trained vs untrained", fontsize=10)
    ax_r.legend(fontsize=9, loc="upper right")
    _annotation_box(
        ax_r,
        "Untrained reaches rank ≈ 1\nby ~20% depth.\nTraining resists collapse.",
        xy=(0.38, ax_r.get_ylim()[1] * 0.55 if ax_r.get_ylim()[1] > 1 else 5),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "group_A2_rank.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


def group_A3_hdbscan(
    runs: dict,
    out_dir: Path,
    focus_model: str = "albert-xlarge-v2",
    focus_prompt: str = "wiki_paragraph",
) -> None:
    """
    Two-panel layout:
      Top row: PCA scatter at 4 selected layers, colored by HDBSCAN assignment.
      Bottom:  HDBSCAN cluster count k vs layer with plateau windows shaded.

    Shows that the clustering signal is spatially organized discrete groups,
    not a diffuse cloud — the clusters are real and visually distinct in PCA.
    """
    plt.rcParams.update(BLOG_STYLE)

    result = _pick_run(runs, focus_model, focus_prompt, untrained=False)
    if result is None:
        print(f"  ✗  group_A3_hdbscan: no run for {focus_model}/{focus_prompt}")
        return
    model_name, run_dir = result

    try:
        geo = _geo(run_dir)
    except Exception as e:
        print(f"  ✗  group_A3_hdbscan: {e}")
        return

    pca         = _pca_trajs(run_dir)
    hdb_labels  = _hdbscan_labels(run_dir)
    clust_data  = _clustering(run_dir)
    traj_data   = _trajectory(run_dir)

    layers_geo     = geo.get("layers", [])
    n_layers       = len(layers_geo)
    tokens         = geo.get("tokens", [])
    plateau_layers = traj_data.get("plateau_layers", [])

    # k per layer from clustering.json
    clust_layers = clust_data.get("layers", [])
    hdb_k = {}
    for cl in clust_layers:
        k = cl.get("clustering", {}).get("hdbscan", {}).get("n_clusters")
        if k is not None:
            hdb_k[cl["layer"]] = k

    # Select 4 representative layers
    plat_spans = _spans(plateau_layers)
    if plat_spans:
        pre   = max(0, plat_spans[0][0] - 1)
        p_mid = (plat_spans[0][0] + plat_spans[0][1]) // 2
        post  = min(n_layers - 1, plat_spans[-1][1] + 3)
    else:
        pre   = n_layers // 4
        p_mid = n_layers // 2
        post  = n_layers - 1
    selected = [0, pre, p_mid, post]
    panel_labels = [
        "Layer 0\n(embedding)",
        "Pre-plateau",
        "Plateau midpoint",
        "Post-plateau",
    ]

    CLUSTER_PAL = plt.cm.tab10(np.linspace(0, 1, 10))

    fig = plt.figure(figsize=(16, 7.5))
    gs  = gridspec.GridSpec(2, 4, height_ratios=[3.2, 1.4],
                            hspace=0.38, wspace=0.30)
    axes_top = [fig.add_subplot(gs[0, c]) for c in range(4)]
    ax_bot   = fig.add_subplot(gs[1, :])
    fig.suptitle(
        f"A.3 — HDBSCAN cluster structure  ·  {model_name} | {focus_prompt}",
        fontsize=12, fontweight="bold",
    )

    for col, (li, plbl) in enumerate(zip(selected, panel_labels)):
        ax = axes_top[col]
        proj   = pca.get(li)
        labels = hdb_labels.get(li)

        if proj is None or labels is None or len(proj) == 0:
            ax.text(0.5, 0.5, f"Layer {li}\n(no PCA data)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#9CA3AF")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(plbl, fontsize=9)
            continue

        # Assign colors: noise = light gray, clusters = tab10
        uniq = sorted(set(labels))
        ci   = 0
        cmap = {}
        for ul in uniq:
            if ul == -1:
                cmap[-1] = "#D1D5DB"
            else:
                cmap[ul] = CLUSTER_PAL[ci % 10]
                ci += 1

        ptc    = np.array([cmap[lb] for lb in labels])
        n_cl   = len([u for u in uniq if u >= 0])
        in_pl  = li in set(plateau_layers)
        pl_tag = " ● plateau" if in_pl else ""

        ax.scatter(proj[:, 0], proj[:, 1], c=ptc, s=45, zorder=3,
                   edgecolors="white", linewidths=0.5)

        # Light token labels for non-noise tokens (first 4 chars)
        for ti, (lb, (x, y)) in enumerate(zip(labels, proj[:, :2])):
            if lb == -1:
                continue
            tok = tokens[ti][:4] if ti < len(tokens) else str(ti)
            ax.annotate(tok, (x, y), fontsize=4, alpha=0.55,
                        ha="center", va="bottom")

        ax.set_title(f"{plbl}\nLayer {li}  k={n_cl}{pl_tag}", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("PC1", fontsize=7)
        if col == 0:
            ax.set_ylabel("PC2", fontsize=7)

    # Bottom: k count vs layer
    if hdb_k:
        k_series = [hdb_k.get(i, np.nan) for i in range(n_layers)]
        _shade_plateaus(ax_bot, plateau_layers, alpha=0.22)
        ax_bot.plot(range(n_layers), k_series,
                    color=_color(focus_model), linewidth=2.2,
                    marker="o", markersize=3.5, alpha=0.88, zorder=3)
        # Mark the 4 selected layers
        for li in selected:
            if not np.isnan(k_series[li]):
                ax_bot.axvline(li, color="#6B7280", linewidth=0.8,
                               linestyle=":", alpha=0.7)
        ax_bot.set_xlabel("Layer")
        ax_bot.set_ylabel("HDBSCAN k")
        ax_bot.set_title(
            "Cluster count across layers  (yellow = plateau window, "
            "dotted lines = panels above)", fontsize=9,
        )
        ax_bot.set_xlim(-0.5, n_layers - 0.5)
    else:
        ax_bot.text(0.5, 0.5, "No HDBSCAN k data",
                    ha="center", va="center", transform=ax_bot.transAxes,
                    fontsize=10, color="#9CA3AF")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "group_A3_hdbscan.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Group entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_group_A(runs: dict, out_dir: Path) -> None:
    """Generate all Group A figures."""
    print("Group A — Do clusters form?")
    group_A_glance(runs, out_dir)
    group_A1_histograms(runs, out_dir)
    group_A1_mass_all(runs, out_dir)
    group_A2_rank(runs, out_dir)
    group_A3_hdbscan(runs, out_dir)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Stubs (to be filled in subsequent passes)
# ─────────────────────────────────────────────────────────────────────────────

def generate_group_B(runs: dict, out_dir: Path) -> None:
    """Group B — Do clusters persist? (B-glance, B1 CKA, B2 NN, B3 multisignal, B4 trajectories)"""
    print("Group B — placeholder (not yet implemented)")

def generate_group_C(runs: dict, out_dir: Path) -> None:
    """Group C — Does attention know? (C-glance, C1 entropy, C2 Fiedler, C3 heads)"""
    print("Group C — placeholder (not yet implemented)")

def generate_group_D(runs: dict, out_dir: Path) -> None:
    """Group D — Does the energy hold? (D-glance, D1 violations, D2 pairs)"""
    print("Group D — placeholder (not yet implemented)")

def generate_group_E(runs: dict, out_dir: Path) -> None:
    """Group E — Two timescales? (E-glance, E1 ratio, E2 spectrum)"""
    print("Group E — placeholder (not yet implemented)")
