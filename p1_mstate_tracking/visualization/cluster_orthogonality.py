"""
visualization/cluster_orthogonality.py

Tracks the top-k clusters (cluster_tracking.py trajectories), selected by
mean population or by lifespan, and plots two things against layer depth
that the existing cluster_reality figures don't separate:

  1. cohesion   — mean within-cluster pairwise inner product, per cluster.
                  Falling cohesion = the cluster is dissolving internally.
  2. separation — mean off-diagonal inner product between cluster centroids,
                  plus the effective rank of the (k_active, d) centroid
                  matrix. Falling separation / rising centroid rank =
                  clusters are tight individually but spreading into more
                  mutually independent directions.

These two axes are orthogonal claims. "Clusters fall apart" (1) and
"clusters get placed in more orthogonal subspaces" (2) are different
mechanisms that can produce the same global statistic (e.g. a drop in
ip_mass_near_1) and this figure is built to tell them apart rather than
assume one.

Drop-in for the existing visualization/ package: same fn(run_dir, out_dir,
layers=None) signature is NOT used here because this figure is inherently
a single trajectory-vs-all-layers plot, not a per-reference-layer panel —
call signature is fn(run_dir, out_dir, top_k=5, rank_by="persistence",
min_lifespan=4).
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from .style import BLOG_STYLE, CLUSTER_PAL, MIN_CLUSTER_SIZE
from .naming import _safe_model_name
from .loaders import _geo, _trajectory, _hdbscan_labels, _load_activations


# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────────────

def _trajectory_mean_size(traj: dict, hdb: Dict[int, List[int]]) -> float:
    """Mean token count of this trajectory's cluster across its own chain
    (not the fixed reference layers cluster_reality.py uses — every layer
    the cluster is actually alive)."""
    sizes = []
    for layer, cid in traj["chain"]:
        labels = hdb.get(layer)
        if labels is None:
            continue
        sizes.append(sum(1 for l in labels if l == cid))
    return float(np.mean(sizes)) if sizes else 0.0


def _select_top_clusters(
    trajectories: List[dict],
    hdb: Dict[int, List[int]],
    top_k: int,
    rank_by: str,
    min_lifespan: int,
    min_size: int,
) -> List[dict]:
    candidates = [t for t in trajectories if t["lifespan"] >= min_lifespan]
    sized = [(t, _trajectory_mean_size(t, hdb)) for t in candidates]
    sized = [(t, s) for t, s in sized if s >= min_size]

    if rank_by == "persistence":
        sized.sort(key=lambda ts: ts[0]["lifespan"], reverse=True)
    elif rank_by == "size":
        sized.sort(key=lambda ts: ts[1], reverse=True)
    else:
        raise ValueError(f"rank_by must be 'persistence' or 'size', got {rank_by!r}")

    return [t for t, _ in sized[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer geometry
# ─────────────────────────────────────────────────────────────────────────────

def _unit_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return X / norms


def _effective_rank(C: np.ndarray) -> float:
    """Participation-ratio effective rank (same definition as
    metrics.effective_rank_from_normed) reimplemented locally so this
    module has no dependency on the parent package's import path."""
    if C.shape[0] < 1:
        return 0.0
    sv2 = np.linalg.svd(C, compute_uv=False) ** 2
    tot = sv2.sum()
    if tot < 1e-12:
        return 1.0
    p = np.clip(sv2 / tot, 1e-12, None)
    return float(np.exp(-(p * np.log(p)).sum()))


def _all_cluster_stats(
    activations: np.ndarray,
    hdb: Dict[int, List[int]],
    n_layers: int,
) -> dict:
    """
    Same two quantities as _layer_stats, but computed over every HDBSCAN
    cluster present at each layer, not just the tracked top_k. This is the
    "general population" context for the top-k lines: mean ± std of
    within-cluster cohesion across all clusters (not pooled pairs — the
    mean/std of each cluster's own cohesion score), and mean ± std of
    pairwise centroid IP across all cluster pairs.

    Returns per-layer arrays:
      cohesion_mean, cohesion_std   : across all clusters' individual cohesion scores
      cohesion_n                   : how many clusters had >=2 members (denominator)
      between_mean, between_std    : across all off-diagonal centroid-pair IPs
      n_clusters                   : total HDBSCAN cluster count that layer
    """
    cohesion_mean = np.full(n_layers, np.nan)
    cohesion_std = np.full(n_layers, np.nan)
    cohesion_n = np.zeros(n_layers, dtype=int)
    between_mean = np.full(n_layers, np.nan)
    between_std = np.full(n_layers, np.nan)
    n_clusters = np.zeros(n_layers, dtype=int)

    for layer in range(n_layers):
        if layer >= activations.shape[0]:
            break
        labels = np.array(hdb.get(layer, []))
        if labels.size == 0:
            continue
        acts = activations[layer]
        cluster_ids = sorted(set(labels.tolist()) - {-1})
        n_clusters[layer] = len(cluster_ids)

        cohesions = []
        centroids = []
        for cid in cluster_ids:
            mask = labels == cid
            n_members = int(mask.sum())
            if n_members < 2:
                continue
            sub = acts[mask]
            gram = sub @ sub.T
            iu, ju = np.triu_indices(n_members, k=1)
            cohesions.append(float(gram[iu, ju].mean()))
            centroids.append(sub.mean(axis=0))

        if cohesions:
            cohesion_mean[layer] = float(np.mean(cohesions))
            cohesion_std[layer] = float(np.std(cohesions))
            cohesion_n[layer] = len(cohesions)

        if len(centroids) >= 2:
            C = _unit_rows(np.array(centroids))
            G = C @ C.T
            iu, ju = np.triu_indices(len(centroids), k=1)
            off_diag = G[iu, ju]
            between_mean[layer] = float(off_diag.mean())
            between_std[layer] = float(off_diag.std())

    return dict(
        cohesion_mean=cohesion_mean, cohesion_std=cohesion_std, cohesion_n=cohesion_n,
        between_mean=between_mean, between_std=between_std, n_clusters=n_clusters,
    )


def _layer_stats(
    activations: np.ndarray,
    hdb: Dict[int, List[int]],
    chains: Dict[int, Dict[int, int]],
    n_layers: int,
) -> dict:
    """
    Returns per-layer arrays, all indexed [layer][cluster_rank]:
      intra_ip      : mean within-cluster pairwise IP (NaN if cluster
                      inactive or has < 2 members at that layer)
      centroid_ip   : (k, k) matrix, NaN rows/cols for inactive clusters
      sizes         : token count per cluster, 0 if inactive
    Plus per-layer scalars:
      n_active      : number of selected clusters alive at that layer
      inter_mean    : mean off-diagonal centroid IP among active clusters
      inter_max     : max off-diagonal centroid IP (closest pair)
      centroid_rank : effective rank of the active centroid matrix
    """
    traj_ids = sorted(chains.keys())
    k = len(traj_ids)

    intra_ip = np.full((n_layers, k), np.nan)
    sizes = np.zeros((n_layers, k), dtype=int)
    inter_mean = np.full(n_layers, np.nan)
    inter_max = np.full(n_layers, np.nan)
    centroid_rank = np.full(n_layers, np.nan)

    for layer in range(n_layers):
        if layer >= activations.shape[0]:
            break
        labels = np.array(hdb.get(layer, []))
        if labels.size == 0:
            continue
        acts = activations[layer]

        centroids = []
        for rank_idx, tid in enumerate(traj_ids):
            cid = chains[tid].get(layer)
            if cid is None:
                continue
            mask = labels == cid
            n_members = int(mask.sum())
            sizes[layer, rank_idx] = n_members
            if n_members < 2:
                continue
            sub = acts[mask]
            gram = sub @ sub.T
            iu, ju = np.triu_indices(n_members, k=1)
            intra_ip[layer, rank_idx] = float(gram[iu, ju].mean())
            centroid = sub.mean(axis=0)
            centroids.append(centroid)

        if len(centroids) >= 2:
            C = _unit_rows(np.array(centroids))
            G = C @ C.T
            iu, ju = np.triu_indices(len(centroids), k=1)
            off_diag = G[iu, ju]
            inter_mean[layer] = float(off_diag.mean())
            inter_max[layer] = float(off_diag.max())
            centroid_rank[layer] = _effective_rank(C)

    n_active = (sizes > 0).sum(axis=1)

    return dict(
        traj_ids=traj_ids,
        intra_ip=intra_ip,
        sizes=sizes,
        inter_mean=inter_mean,
        inter_max=inter_max,
        centroid_rank=centroid_rank,
        n_active=n_active,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_orthogonality_trajectory(
    run_dir: Path,
    out_dir: Path,
    top_k: int = 5,
    rank_by: str = "persistence",
    min_lifespan: int = 4,
    min_size: int = MIN_CLUSTER_SIZE,
) -> None:
    """
    Cohesion (within-cluster IP) and separation (between-centroid IP +
    centroid effective rank) for the top_k tracked clusters, one line per
    cluster, full depth.
    """
    plt.rcParams.update(BLOG_STYLE)
    geo = _geo(run_dir)
    model, prompt, n_layers = geo["model"], geo["prompt"], geo["n_layers"]
    hdb = _hdbscan_labels(run_dir)
    traj_data = _trajectory(run_dir)
    trajectories = traj_data.get("cluster_tracking", {}).get("trajectories", [])
    activations = _load_activations(run_dir)

    if not hdb or not trajectories or activations is None:
        print(f"  [skip] missing labels/trajectories/activations for {model}/{prompt}")
        return

    kept = _select_top_clusters(trajectories, hdb, top_k, rank_by, min_lifespan, min_size)
    if not kept:
        print(f"  [skip] no cluster reaches lifespan>={min_lifespan}, size>={min_size} "
              f"for {model}/{prompt}")
        return

    chains = {t["id"]: dict(t["chain"]) for t in kept}
    stats = _layer_stats(activations, hdb, chains, n_layers)
    all_stats = _all_cluster_stats(activations, hdb, n_layers)
    layers_x = np.arange(n_layers)

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1.3, 2, 1.3]})

    # Panel 1 — within-cluster cohesion, top-k only
    ax = axes[0]
    for rank_idx, tid in enumerate(stats["traj_ids"]):
        color = CLUSTER_PAL[rank_idx % len(CLUSTER_PAL)]
        traj = next(t for t in kept if t["id"] == tid)
        ax.plot(layers_x, stats["intra_ip"][:, rank_idx], color=color, linewidth=1.8,
                label=f"cluster {rank_idx + 1} (traj {tid}, lifespan {traj['lifespan']})")
    ax.axhline(0.9, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("mean within-cluster ⟨xᵢ,xⱼ⟩")
    ax.set_title(f"Cohesion — top {len(kept)} tracked clusters", fontsize=11)
    ax.legend(fontsize=7, loc="lower left")

    # Panel 2 — cohesion context: mean +/- std across EVERY HDBSCAN cluster
    # that layer, not just the tracked top-k. Tells you whether the top-k
    # lines above are representative of the general cluster population or
    # outliers within it.
    ax = axes[1]
    cm, cs = all_stats["cohesion_mean"], all_stats["cohesion_std"]
    ax.plot(layers_x, cm, color="#374151", linewidth=1.8, label="all-cluster mean cohesion")
    ax.fill_between(layers_x, cm - cs, cm + cs, color="#9CA3AF", alpha=0.4, label="± 1 std")
    ax.set_ylabel("mean within-cluster\n⟨xᵢ,xⱼ⟩ (all clusters)")
    ax.set_title(
        "Cohesion context — all HDBSCAN clusters that layer "
        "(n_clusters shown as faint dotted line, right axis)",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="lower left")
    ax2 = ax.twinx()
    ax2.plot(layers_x, all_stats["n_clusters"], color="#1F2937", linewidth=0.8,
             linestyle=":", alpha=0.6)
    ax2.set_ylabel("n clusters", fontsize=8, color="#1F2937")
    ax2.tick_params(axis="y", labelsize=7, colors="#1F2937")

    # Panel 3 — between-cluster separation, top-k only
    ax = axes[2]
    ax.plot(layers_x, stats["inter_mean"], color="#1F2937", linewidth=2.0,
            label="mean centroid-centroid ⟨c_a,c_b⟩ (top-k)")
    ax.plot(layers_x, stats["inter_max"], color="#9CA3AF", linewidth=1.4, linestyle="--",
            label="max centroid-centroid ⟨c_a,c_b⟩ (closest pair, top-k)")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("centroid inner product")
    ax.set_title(
        f"Separation — top {len(kept)} tracked clusters. Toward 0 means centroids "
        "orthogonalize; toward 1 means clusters are merging back together",
        fontsize=11,
    )
    ax.legend(fontsize=7, loc="upper right")

    ax2 = ax.twinx()
    ax2.plot(layers_x, stats["centroid_rank"], color="#B91C1C", linewidth=1.4,
             linestyle="-.", label=f"centroid effective rank (max={len(kept)})")
    ax2.axhline(len(kept), color="#B91C1C", linestyle=":", linewidth=0.7, alpha=0.4)
    ax2.set_ylabel("centroid eff. rank", color="#B91C1C")
    ax2.tick_params(axis="y", colors="#B91C1C")
    ax2.set_ylim(0, len(kept) + 0.5)

    # Panel 4 — separation context: mean +/- std centroid IP across EVERY
    # cluster pair that layer, not just the tracked top-k.
    ax = axes[3]
    bm, bs = all_stats["between_mean"], all_stats["between_std"]
    ax.plot(layers_x, bm, color="#1F2937", linewidth=1.8, label="all-cluster mean centroid IP")
    ax.fill_between(layers_x, bm - bs, bm + bs, color="#9CA3AF", alpha=0.4, label="± 1 std")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("centroid IP\n(all cluster pairs)")
    ax.set_xlabel("Layer")
    ax.set_title("Separation context — every HDBSCAN cluster pair that layer", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Cluster cohesion vs. separation across depth — {model} | {prompt}\n"
        f"top {len(kept)} clusters by {rank_by}, against the full cluster population",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cluster_orthogonality_{_safe_model_name(model)}_{prompt}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  ({len(kept)} clusters, rank_by={rank_by})")
