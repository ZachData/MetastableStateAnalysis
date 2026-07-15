"""
visualization/cluster_reality.py

The "cluster_reality_*" figure set — per-model, written to each model's
own folder: sizes, sorted Gram heatmap, within/between/noise IP histogram,
PCA scatter by cluster id, persistence river, and the colored-text view
(top-k tracked clusters with full roster at first appearance, then
join/leave diffs at later reference layers instead of relisting).
"""

import string
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .style import (
    BLOG_STYLE, LayerSpec, MIN_CLUSTER_SIZE, CLUSTER_PAL, NOISE_COLOR,
    ADDED_COLOR, REMOVED_COLOR,
)
from .naming import _safe_model_name
from .loaders import _geo, _clustering, _trajectory, _hdbscan_labels, _pca_trajs, _load_activations
from .plot_utils import _resolve_layers

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cluster size histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_size_histogram(
    run_dir: Path, out_dir: Path, layers: Optional[List[LayerSpec]] = None,
) -> None:
    plt.rcParams.update(BLOG_STYLE)
    geo = _geo(run_dir)
    model, prompt, n_tokens, n_layers = geo["model"], geo["prompt"], geo["n_tokens"], geo["n_layers"]
    hdb = _hdbscan_labels(run_dir)
    if not hdb:
        print(f"  [skip] no HDBSCAN labels for {model}/{prompt}")
        return
    resolved = _resolve_layers(layers, n_layers)

    fig, axes = plt.subplots(1, len(resolved), figsize=(5.2 * len(resolved), 5))
    axes = np.atleast_1d(axes)

    for ax, layer in zip(axes, resolved):
        labels = np.array(hdb.get(layer, []))
        if labels.size == 0:
            ax.set_title(f"layer {layer}\n(no data)", fontsize=10)
            ax.axis("off")
            continue
        counts = Counter(labels.tolist())
        noise_n = counts.pop(-1, 0)
        sizes = sorted(counts.values(), reverse=True)
        x = np.arange(len(sizes))
        ax.bar(x, sizes, color=[CLUSTER_PAL[i % len(CLUSTER_PAL)] for i in range(len(sizes))],
               edgecolor="white", linewidth=0.6)
        if noise_n:
            ax.bar([len(sizes)], [noise_n], color=NOISE_COLOR, edgecolor="white",
                   linewidth=0.6, hatch="//")
            ax.set_xticks(list(x) + [len(sizes)])
            ax.set_xticklabels([str(i + 1) for i in x] + ["noise"], fontsize=7)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([str(i + 1) for i in x], fontsize=7)

        frac_clustered = 1 - noise_n / n_tokens if n_tokens else 0.0
        ax.set_xlabel("Cluster (ranked by size)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Token count", fontsize=9)
        ax.set_title(
            f"layer {layer}\n{len(sizes)} clusters, {frac_clustered * 100:.0f}% clustered",
            fontsize=10,
        )

    fig.suptitle(f"Cluster size distribution — {model} | {prompt}", fontsize=12, fontweight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cluster_reality_sizes_{_safe_model_name(model)}_{prompt}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  (layers={resolved})")
# ─────────────────────────────────────────────────────────────────────────────
# 2. Sorted Gram-matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_sorted_gram_heatmap(
    run_dir: Path, out_dir: Path, layers: Optional[List[LayerSpec]] = None,
) -> None:
    plt.rcParams.update(BLOG_STYLE)
    geo = _geo(run_dir)
    model, prompt, n_layers = geo["model"], geo["prompt"], geo["n_layers"]
    hdb = _hdbscan_labels(run_dir)
    if not hdb:
        print(f"  [skip] no HDBSCAN labels for {model}/{prompt}")
        return
    resolved = _resolve_layers(layers, n_layers)

    activations = _load_activations(run_dir)
    if activations is None:
        print(f"  [skip] no activations for {model}/{prompt}")
        return

    fig, axes = plt.subplots(1, len(resolved), figsize=(5.6 * len(resolved), 5.5))
    axes = np.atleast_1d(axes)
    im = None

    for ax, layer in zip(axes, resolved):
        if layer >= activations.shape[0]:
            ax.set_title(f"layer {layer}\n(out of range)", fontsize=10)
            ax.axis("off")
            continue
        labels = np.array(hdb.get(layer, []))
        acts = activations[layer]
        gram = acts @ acts.T
        sort_key = np.where(labels == -1, np.iinfo(np.int64).max, labels)
        order = np.lexsort((np.arange(len(labels)), sort_key))
        gram_sorted = gram[order][:, order]
        labels_sorted = labels[order]

        im = ax.imshow(gram_sorted, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        boundaries = np.where(np.diff(labels_sorted) != 0)[0] + 0.5
        for b in boundaries:
            ax.axhline(b, color="white", linewidth=0.4, alpha=0.6)
            ax.axvline(b, color="white", linewidth=0.4, alpha=0.6)

        n_clusters = len(set(labels.tolist()) - {-1})
        ax.set_title(f"layer {layer}\n{n_clusters} clusters", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Pairwise inner products, sorted by cluster — {model} | {prompt}\n"
        f"block structure on the diagonal = real separation",
        fontsize=12, fontweight="bold",
    )
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.75, label="⟨xᵢ, xⱼ⟩")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cluster_reality_gram_{_safe_model_name(model)}_{prompt}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  (layers={resolved})")
# ─────────────────────────────────────────────────────────────────────────────
# 3. Within- vs between-cluster IP histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_within_between_ip_histogram(
    run_dir: Path, out_dir: Path, layers: Optional[List[LayerSpec]] = None, bins: int = 51,
) -> None:
    plt.rcParams.update(BLOG_STYLE)
    geo = _geo(run_dir)
    model, prompt, n_layers = geo["model"], geo["prompt"], geo["n_layers"]
    hdb = _hdbscan_labels(run_dir)
    if not hdb:
        print(f"  [skip] no HDBSCAN labels for {model}/{prompt}")
        return
    resolved = _resolve_layers(layers, n_layers)

    activations = _load_activations(run_dir)
    if activations is None:
        print(f"  [skip] no activations for {model}/{prompt}")
        return

    bin_edges = np.linspace(-1, 1, bins)
    fig, axes = plt.subplots(1, len(resolved), figsize=(5.4 * len(resolved), 5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, layer in zip(axes, resolved):
        if layer >= activations.shape[0]:
            ax.set_title(f"layer {layer}\n(out of range)", fontsize=10)
            ax.axis("off")
            continue
        labels = np.array(hdb.get(layer, []))
        acts = activations[layer]
        gram = acts @ acts.T
        n = len(labels)
        iu, ju = np.triu_indices(n, k=1)
        li, lj = labels[iu], labels[ju]
        vals = gram[iu, ju]
        both_clustered = (li != -1) & (lj != -1)
        within = vals[both_clustered & (li == lj)]
        between = vals[both_clustered & (li != lj)]
        noisy = vals[~both_clustered]

        for data, lab, color in [
            (noisy, "noise-involved", NOISE_COLOR),
            (between, "between-cluster", "#2563EB"),
            (within, "within-cluster", "#DC2626"),
        ]:
            if data.size == 0:
                continue
            ax.hist(data, bins=bin_edges, density=True, histtype="step",
                    linewidth=2.0, color=color, label=f"{lab} (n={data.size:,})")

        ax.axvline(0.9, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("⟨xᵢ, xⱼ⟩", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"layer {layer}", fontsize=10)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Within- vs between-cluster inner products — {model} | {prompt}",
        fontsize=12, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cluster_reality_ip_split_{_safe_model_name(model)}_{prompt}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  (layers={resolved})")
# ─────────────────────────────────────────────────────────────────────────────
# 4. PCA scatter colored by cluster
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_scatter_by_cluster(
    run_dir: Path, out_dir: Path, layers: Optional[List[LayerSpec]] = None,
) -> None:
    plt.rcParams.update(BLOG_STYLE)
    geo = _geo(run_dir)
    model, prompt, n_layers = geo["model"], geo["prompt"], geo["n_layers"]
    hdb = _hdbscan_labels(run_dir)
    if not hdb:
        print(f"  [skip] no HDBSCAN labels for {model}/{prompt}")
        return
    resolved = _resolve_layers(layers, n_layers)
    pca = _pca_trajs(run_dir)

    fig, axes = plt.subplots(1, len(resolved), figsize=(6 * len(resolved), 6))
    axes = np.atleast_1d(axes)

    for ax, layer in zip(axes, resolved):
        proj = pca.get(layer)
        if proj is None:
            ax.set_title(f"layer {layer}\n(no PCA)", fontsize=10)
            ax.axis("off")
            continue
        labels = np.array(hdb.get(layer, []))
        for cid in sorted(set(labels.tolist())):
            mask = labels == cid
            if cid == -1:
                ax.scatter(proj[mask, 0], proj[mask, 1], marker="x", s=16,
                           color=NOISE_COLOR, linewidths=0.8, zorder=2)
            else:
                color = CLUSTER_PAL[cid % len(CLUSTER_PAL)]
                ax.scatter(proj[mask, 0], proj[mask, 1], marker="o", s=28,
                           facecolor=color, edgecolor="white", linewidths=0.5, zorder=3)
        n_clusters = len(set(labels.tolist()) - {-1})
        ax.set_title(f"layer {layer}\n{n_clusters} clusters", fontsize=10)
        ax.set_xlabel("PC1", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("PC2", fontsize=9)

    fig.suptitle(f"PCA projection by cluster — {model} | {prompt}", fontsize=12, fontweight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cluster_reality_pca_{_safe_model_name(model)}_{prompt}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  (layers={resolved})")
# ─────────────────────────────────────────────────────────────────────────────
# 5. Cluster persistence river plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_persistence_river(
    run_dir: Path, out_dir: Path, layers: Optional[List[LayerSpec]] = None,
    top_k: int = 8, min_lifespan: int = 4,
) -> None:
    plt.rcParams.update(BLOG_STYLE)
    geo = _geo(run_dir)
    model, prompt, n_layers, n_tokens = geo["model"], geo["prompt"], geo["n_layers"], geo["n_tokens"]
    hdb = _hdbscan_labels(run_dir)
    traj_data = _trajectory(run_dir)
    trajectories = traj_data.get("cluster_tracking", {}).get("trajectories", [])
    if not hdb or not trajectories:
        print(f"  [skip] missing labels/trajectories for {model}/{prompt}")
        return
    ref_layers = _resolve_layers(layers, n_layers)

    ranked = sorted(trajectories, key=lambda t: t["lifespan"], reverse=True)
    kept = [t for t in ranked if t["lifespan"] >= min_lifespan][:top_k]
    if not kept:
        print(f"  [skip] no trajectories with lifespan >= {min_lifespan} for {model}/{prompt}")
        return

    kept_ids = {t["id"] for t in kept}
    chain_lookup = {t["id"]: dict((l, c) for l, c in t["chain"]) for t in kept}
    series = {t["id"]: np.zeros(n_layers) for t in kept}
    other_series = np.zeros(n_layers)
    noise_series = np.zeros(n_layers)

    for layer in range(n_layers):
        labels = np.array(hdb.get(layer, []))
        if labels.size == 0:
            continue
        noise_series[layer] = int((labels == -1).sum())
        accounted = np.zeros(len(labels), dtype=bool)
        for tid in kept_ids:
            cid = chain_lookup[tid].get(layer)
            if cid is None:
                continue
            mask = labels == cid
            series[tid][layer] = int(mask.sum())
            accounted |= mask
        clustered = labels != -1
        other_series[layer] = int((clustered & ~accounted).sum())

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(n_layers)
    ys = [series[t["id"]] for t in kept] + [other_series, noise_series]
    labels_legend = [f"trajectory {t['id']} (lifespan {t['lifespan']})" for t in kept] + \
        ["other tracked clusters", "noise / unclustered"]
    colors = [CLUSTER_PAL[t["id"] % len(CLUSTER_PAL)] for t in kept] + ["#9CA3AF", NOISE_COLOR]

    ax.stackplot(x, ys, labels=labels_legend, colors=colors, edgecolor="white", linewidth=0.3)
    for rl in ref_layers:
        ax.axvline(rl, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(rl, n_tokens * 1.01, f"L{rl}", fontsize=8, ha="center")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Token count")
    ax.set_ylim(0, n_tokens)
    ax.set_title(
        f"Cluster persistence across layers — {model} | {prompt}\n"
        f"dashed lines mark the reference layers used in the other figures",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cluster_reality_river_{_safe_model_name(model)}_{prompt}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  ({len(kept)} trajectories tracked, reference layers={ref_layers})")
# ─────────────────────────────────────────────────────────────────────────────
# 6. Top cluster token tracking (membership diff across layers)
# ─────────────────────────────────────────────────────────────────────────────

def _is_punct_token(tok: str) -> bool:
    """
    True if `tok` carries no alphanumeric content once common subword
    affixes are stripped — i.e. it's punctuation, a bare subword marker,
    or whitespace. Catches plain ASCII punctuation as well as Unicode
    punctuation (em dashes, smart quotes, etc. — Unicode category "P*"),
    so it's not just a hardcoded ASCII set.
    """
    core = tok
    for affix in ("Ġ", "▁", "##"):
        core = core.replace(affix, "")
    core = core.strip()
    if not core:
        return True
    return all(ch in string.punctuation or unicodedata.category(ch).startswith("P") for ch in core)


def _wrap_badges(
    items: List[Tuple[str, object]], chars_per_line: int = 90,
) -> List[List[Tuple[str, object]]]:
    """Greedy line-wrap a sequence of (display_text, facecolor) badges."""
    rows, cur_row, cur_len = [], [], 0
    for disp, color in items:
        w = len(disp) + 1
        if cur_len + w > chars_per_line and cur_row:
            rows.append(cur_row)
            cur_row, cur_len = [], 0
        cur_row.append((disp, color))
        cur_len += w
    if cur_row:
        rows.append(cur_row)
    return rows


def plot_cluster_colored_text(
    run_dir: Path, out_dir: Path, layers: Optional[List[LayerSpec]] = None,
    chars_per_line: int = 90, top_k: int = 3, min_size: int = MIN_CLUSTER_SIZE,
    exclude_punct_clusters: bool = True, punct_threshold: float = 0.6,
) -> None:
    """
    Track the `top_k` most-populated clusters (by mean member count across
    the reference layers) instead of recoloring the whole prompt at every
    layer. Each cluster gets its own block: a full token roster at the
    first reference layer it's active, then — at every later reference
    layer — only the *diff* against the previously shown layer (tokens
    that joined in green, tokens that left in red). A 4-50 token roster
    only gets printed once; everything after that is just what changed.

    Cluster identity across layers comes from cluster_tracking's
    trajectory chains (same source as plot_cluster_persistence_river),
    since raw HDBSCAN cluster ids aren't stable from one layer to the next.

    Punctuation/whitespace tokens (periods, commas, quotes, subword
    markers, ...) are frequent enough that they tend to dominate the
    biggest clusters on raw population alone, which crowds out clusters
    that are actually semantically grounded. If `exclude_punct_clusters`
    is set, any cluster whose membership is at least `punct_threshold`
    punctuation (by token type, over the union of everything it ever
    contains across the reference layers) is skipped when ranking by
    size — it still exists in the data, it just doesn't compete for a
    `top_k` slot.
    """
    plt.rcParams.update(BLOG_STYLE)
    geo = _geo(run_dir)
    model, prompt, tokens, n_layers = geo["model"], geo["prompt"], geo.get("tokens", []), geo["n_layers"]
    hdb = _hdbscan_labels(run_dir)
    traj_data = _trajectory(run_dir)
    trajectories = traj_data.get("cluster_tracking", {}).get("trajectories", [])
    if not hdb or not tokens or not trajectories:
        print(f"  [skip] missing labels/tokens/trajectories for {model}/{prompt}")
        return
    resolved = _resolve_layers(layers, n_layers)

    chain_lookup = {t["id"]: dict(t["chain"]) for t in trajectories}
    label_arrays = {l: np.array(hdb.get(l, [])) for l in resolved}

    def _members(tid: int, layer: int) -> Optional[set]:
        cid = chain_lookup[tid].get(layer)
        labels = label_arrays.get(layer)
        if cid is None or labels is None or labels.size == 0:
            return None
        return set(np.nonzero(labels == cid)[0].tolist())

    members_by_traj = {
        t["id"]: {l: _members(t["id"], l) for l in resolved} for t in trajectories
    }

    def _mean_size(tid: int) -> float:
        sizes = [len(m) for m in members_by_traj[tid].values() if m is not None]
        return float(np.mean(sizes)) if sizes else 0.0

    def _punct_fraction(tid: int) -> float:
        union: set = set()
        for m in members_by_traj[tid].values():
            if m:
                union |= m
        if not union:
            return 0.0
        n_punct = sum(1 for i in union if _is_punct_token(tokens[i]))
        return n_punct / len(union)

    ranked = sorted(trajectories, key=lambda t: _mean_size(t["id"]), reverse=True)
    candidates = [t for t in ranked if _mean_size(t["id"]) >= min_size]
    n_punct_skipped = 0
    if exclude_punct_clusters:
        n_before = len(candidates)
        candidates = [t for t in candidates if _punct_fraction(t["id"]) < punct_threshold]
        n_punct_skipped = n_before - len(candidates)
    kept = candidates[:top_k]
    if not kept:
        reason = "punctuation-dominated or " if exclude_punct_clusters else ""
        print(f"  [skip] no {reason}cluster reaches min_size={min_size} at the reference layers for {model}/{prompt}")
        return

    def _disp(idx: int) -> str:
        tok = tokens[idx]
        return tok if tok.strip() else "·"

    # First pass: build line items; second pass renders them. Each item is
    # ("cluster_header" | "layer_header" | "note", text) or ("rows", rows).
    HEIGHTS = {"cluster_header": 1.6, "layer_header": 1.3, "note": 1.0, "row": 1.0}
    items: List[Tuple[str, object]] = []

    for rank, t in enumerate(kept):
        tid = t["id"]
        color = tuple(CLUSTER_PAL[rank % len(CLUSTER_PAL)])
        items.append((
            "cluster_header",
            f"Cluster {rank + 1}  —  trajectory {tid}  "
            f"(mean {_mean_size(tid):.0f} tokens across the layers shown below)",
        ))

        prev_members: Optional[set] = None
        for layer in resolved:
            members = members_by_traj[tid][layer]
            if members is None:
                items.append(("note", f"  layer {layer} — cluster not active (outside its tracked lifespan)"))
                prev_members = None
                continue

            if prev_members is None:
                ordered = sorted(members)
                items.append(("layer_header", f"  layer {layer} — full roster, {len(ordered)} tokens"))
                badges = [(_disp(i), color) for i in ordered]
                items.append(("rows", _wrap_badges(badges, chars_per_line)))
            else:
                added = sorted(members - prev_members)
                removed = sorted(prev_members - members)
                unchanged = len(members & prev_members)
                items.append((
                    "layer_header",
                    f"  layer {layer} — {unchanged} unchanged, "
                    f"+{len(added)} joined, −{len(removed)} left "
                    f"(now {len(members)} tokens)",
                ))
                badges = [(f"+{_disp(i)}", ADDED_COLOR) for i in added] + \
                         [(f"−{_disp(i)}", REMOVED_COLOR) for i in removed]
                if badges:
                    items.append(("rows", _wrap_badges(badges, chars_per_line)))
            prev_members = members

    total_h = sum(
        (HEIGHTS["row"] * len(content)) if kind == "rows" else HEIGHTS.get(kind, 0)
        for kind, content in items
    ) or 3.0
    fig_h = max(3.0, 0.32 * total_h + 1.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.set_xlim(0, chars_per_line)
    ax.axis("off")

    y = 0.0
    for kind, content in items:
        if kind == "cluster_header":
            y += 0.4
            ax.text(0, y + 0.5, content, fontsize=11, fontweight="bold")
            y += HEIGHTS["cluster_header"]
        elif kind == "layer_header":
            ax.text(0, y + 0.5, content, fontsize=9.5, fontweight="bold", color="#374151")
            y += HEIGHTS["layer_header"]
        elif kind == "note":
            ax.text(0, y + 0.5, content, fontsize=9.5, style="italic", color="#6B7280")
            y += HEIGHTS["note"]
        elif kind == "rows":
            for row in content:
                x = 0.0
                for disp, badge_color in row:
                    ax.text(
                        x, y + 0.5, disp, fontsize=9, fontfamily="monospace",
                        va="center", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=badge_color,
                                  edgecolor="white", linewidth=0.4, alpha=0.9),
                    )
                    x += len(disp) + 1
                y += HEIGHTS["row"]

    ax.set_ylim(y, 0)  # top to bottom

    punct_note = f"; {n_punct_skipped} punctuation-dominated cluster(s) excluded from ranking" \
        if n_punct_skipped else ""
    fig.suptitle(
        f"Top {len(kept)} clusters by population, tracked across depth — {model} | {prompt}\n"
        f"first layer shown = full roster; later layers = diff only (green = joined, red = left)"
        f"{punct_note}",
        fontsize=12, fontweight="bold",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cluster_reality_text_{_safe_model_name(model)}_{prompt}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  ({len(kept)} clusters tracked, {n_punct_skipped} punct skipped, layers={resolved})")


