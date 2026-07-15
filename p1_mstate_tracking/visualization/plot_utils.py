"""
visualization/plot_utils.py

Generic plotting helpers with no model/metric semantics of their own:
layer-span/plateau shading, annotation boxes, layer-spec resolution
(int / "final" / negative index -> concrete layer index), the shared
HDBSCAN scatter renderer, and the PCA/t-SNE/UMAP 2-D projection dispatcher.
Anything here is reused across several of the per-model figure modules.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from .style import (
    LayerSpec, DEFAULT_LAYERS, CLUSTER_PAL, NOISE_COLOR,
    PLATEAU_COLOR, PLATEAU_TINT, PLATEAU_BORDER,
)

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

def _annotation_box(ax, text: str, xy, fontsize: int = 10, xycoords: str = "data"):
    ax.annotate(
        text, xy=xy, xycoords=xycoords, fontsize=fontsize, color="#374151",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#D1D5DB", alpha=0.92),
    )

def _resolve_layers(layers: Optional[List[LayerSpec]], n_layers: int) -> List[int]:
    """
    Resolve a requested layer list against n_layers: "final"/None -> last
    layer, clamp into [0, n_layers-1], dedupe while preserving order.
    """
    requested = layers if layers is not None else list(DEFAULT_LAYERS)
    resolved: List[int] = []
    for l in requested:
        if l is None or l == "final":
            l = n_layers - 1
        l = max(0, min(int(l), n_layers - 1))
        if l not in resolved:
            resolved.append(l)
    return resolved

def _scatter_hdbscan(
    ax, proj: np.ndarray, labels, s_cluster: float = 45, s_noise: float = 22,
) -> int:
    """
    Scatter points colored by HDBSCAN label, with a hard visual split:
      - noise (-1):  small light-gray "x", reduced opacity, no edge
      - clusters:    filled circle, white edge, full color, cycling through
                      CLUSTER_PAL (20 distinct hues)
    Noise never renders in a color a real cluster could also use, so a
    field of gray points can't be mistaken for "a cluster nobody colored."

    proj may have more than 2 columns (e.g. a cached 3-D PCA projection);
    only the first two are used. Returns the number of real (non-noise)
    clusters found.
    """
    labels   = np.asarray(labels)
    is_noise = labels == -1
    uniq_clusters = sorted(set(labels[~is_noise].tolist())) if (~is_noise).any() else []
    cmap = {cid: tuple(CLUSTER_PAL[i % len(CLUSTER_PAL)])
            for i, cid in enumerate(uniq_clusters)}

    if is_noise.any():
        ax.scatter(
            proj[is_noise, 0], proj[is_noise, 1],
            marker="x", s=s_noise, linewidths=0.8,
            color=NOISE_COLOR, alpha=0.45, zorder=2,
        )
    if (~is_noise).any():
        colors = np.array([cmap[c] for c in labels[~is_noise]])
        ax.scatter(
            proj[~is_noise, 0], proj[~is_noise, 1],
            c=colors, s=s_cluster, zorder=3,
            edgecolors="white", linewidths=0.6,
        )
    return len(uniq_clusters)

def _project_2d(
    X: np.ndarray, method: str = "pca", seed: int = 42, metric: str = "cosine",
) -> np.ndarray:
    """
    Project (n_tokens, d) activations to 2D via 'pca', 'tsne', or 'umap'.
    PCA is computed via SVD directly (no sklearn dependency). t-SNE and UMAP
    use cosine metric by default, matching the HDBSCAN distance metric used
    upstream, so the projection reflects the same notion of similarity that
    produced the cluster labels being visualized.
    """
    n = X.shape[0]
    if method == "pca":
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        return U[:, :2] * S[:2]

    elif method == "tsne":
        perplexity = max(2, min(30, (n - 1) // 3))
        ts = TSNE(
            n_components=2, metric=metric, perplexity=perplexity,
            init="pca", random_state=seed,
        )
        return ts.fit_transform(X)

    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise RuntimeError("umap-learn is not installed")
        n_neighbors = max(2, min(15, n - 1))
        reducer = umap.UMAP(
            n_components=2, metric=metric, n_neighbors=n_neighbors,
            random_state=seed,
        )
        return reducer.fit_transform(X)

    else:
        raise ValueError(f"unknown projection method: {method}")


