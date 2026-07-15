"""
visualization/attractor_alignment.py

Geshkovski et al.'s dynamics have a single asymptotic fixed point x* — full
collapse pulls every token toward it. Phase 1 confirms GPT-2/ALBERT never
fully reach it, but "not reached yet" and "not being pulled toward it" are
different claims. This module tests the second one directly, and asks the
specific question raised in conversation: are unclustered tokens more
attracted to x* (the eventual single-point consensus) than to their
nearest cluster centroid, or to each other?

x* proxy: since the true asymptotic fixed point isn't observable from a
finite-depth forward pass, x* is approximated as the L2-normalized mean of
every token's final-layer representation — the empirical center of mass
the dynamics have moved closest to by the end of the stack. This is a
single fixed vector per (model, prompt) run, computed once and used as the
reference for every layer, not re-estimated per layer — the question is
"how far has each layer moved toward where things end up," which requires
a fixed target, not a moving one.

Two figures:
  plot_attractor_alignment_overview   — population means (clustered vs.
                                         unclustered vs. overall) of
                                         ⟨x_i, x*⟩ across depth.
  plot_unclustered_pull_decomposition — for the unclustered population
                                         only: mean IP to x*, mean IP to
                                         nearest cluster centroid, and mean
                                         IP to other unclustered tokens,
                                         three lines on one axis so the
                                         relative pull is read directly off
                                         which line sits highest.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from .style import BLOG_STYLE
from .naming import _safe_model_name
from .loaders import _geo, _hdbscan_labels, _load_activations


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _compute_xstar(activations: np.ndarray) -> np.ndarray:
    """L2-normalized mean of every token's representation at the last
    available layer — the empirical proxy for the asymptotic fixed point."""
    final = activations[-1]
    return _unit(final.mean(axis=0))


def _per_layer_alignment(
    activations: np.ndarray,
    hdb: Dict[int, List[int]],
    x_star: np.ndarray,
    n_layers: int,
) -> dict:
    ip_xstar_clustered = np.full(n_layers, np.nan)
    ip_xstar_noise = np.full(n_layers, np.nan)
    ip_xstar_overall = np.full(n_layers, np.nan)

    noise_to_xstar = np.full(n_layers, np.nan)
    noise_to_best_cluster = np.full(n_layers, np.nan)
    noise_to_noise = np.full(n_layers, np.nan)

    for layer in range(min(n_layers, activations.shape[0])):
        labels = np.array(hdb.get(layer, []))
        if labels.size == 0:
            continue
        acts = activations[layer]
        ip_all = acts @ x_star  # (n,) cosine sim to x*, rows already unit norm

        noise_mask = labels == -1
        clustered_mask = ~noise_mask

        ip_xstar_overall[layer] = float(ip_all.mean())
        if clustered_mask.any():
            ip_xstar_clustered[layer] = float(ip_all[clustered_mask].mean())
        if noise_mask.any():
            ip_xstar_noise[layer] = float(ip_all[noise_mask].mean())
            noise_to_xstar[layer] = ip_xstar_noise[layer]

        # noise-noise mean IP (pure pairs, both unclustered)
        n_noise = int(noise_mask.sum())
        if n_noise >= 2:
            sub = acts[noise_mask]
            gram = sub @ sub.T
            iu, ju = np.triu_indices(n_noise, k=1)
            noise_to_noise[layer] = float(gram[iu, ju].mean())

        # noise -> nearest cluster centroid (max IP over all cluster centroids)
        cluster_ids = sorted(set(labels.tolist()) - {-1})
        if cluster_ids and noise_mask.any():
            centroids = []
            for cid in cluster_ids:
                m = labels == cid
                if m.sum() >= 1:
                    centroids.append(_unit(acts[m].mean(axis=0)))
            if centroids:
                C = np.array(centroids)  # (k, d)
                sims = acts[noise_mask] @ C.T  # (n_noise, k)
                best = sims.max(axis=1)
                noise_to_best_cluster[layer] = float(best.mean())

    return dict(
        ip_xstar_clustered=ip_xstar_clustered,
        ip_xstar_noise=ip_xstar_noise,
        ip_xstar_overall=ip_xstar_overall,
        noise_to_xstar=noise_to_xstar,
        noise_to_best_cluster=noise_to_best_cluster,
        noise_to_noise=noise_to_noise,
    )


def _load_common(run_dir: Path):
    geo = _geo(run_dir)
    hdb = _hdbscan_labels(run_dir)
    activations = _load_activations(run_dir)
    return geo, hdb, activations


def plot_attractor_alignment_overview(run_dir: Path, out_dir: Path) -> None:
    plt.rcParams.update(BLOG_STYLE)
    geo, hdb, activations = _load_common(run_dir)
    model, prompt, n_layers = geo["model"], geo["prompt"], geo["n_layers"]

    if not hdb or activations is None:
        print(f"  [skip] missing hdbscan labels / activations for {model}/{prompt}")
        return

    x_star = _compute_xstar(activations)
    stats = _per_layer_alignment(activations, hdb, x_star, n_layers)
    x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x, stats["ip_xstar_clustered"], color="#2563EB", linewidth=1.8,
            label="clustered tokens")
    ax.plot(x, stats["ip_xstar_noise"], color="#DC2626", linewidth=1.8,
            label="unclustered (noise) tokens")
    ax.plot(x, stats["ip_xstar_overall"], color="#374151", linewidth=1.2,
            linestyle=":", label="overall population")
    ax.set_ylabel("⟨x_i, x*⟩")
    ax.set_xlabel("Layer")
    ax.set_title(
        f"Alignment to the global attractor proxy x* — {model} | {prompt}\n"
        f"x* = L2-normalized mean of all tokens at the final layer; "
        f"higher = closer to where the dynamics end up",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="best")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"attractor_alignment_{_safe_model_name(model)}_{prompt}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


def plot_unclustered_pull_decomposition(run_dir: Path, out_dir: Path) -> None:
    """
    The direct test: for unclustered tokens only, which pulls harder —
    the global attractor x*, the nearest cluster centroid, or each other?
    """
    plt.rcParams.update(BLOG_STYLE)
    geo, hdb, activations = _load_common(run_dir)
    model, prompt, n_layers = geo["model"], geo["prompt"], geo["n_layers"]

    if not hdb or activations is None:
        print(f"  [skip] missing hdbscan labels / activations for {model}/{prompt}")
        return

    x_star = _compute_xstar(activations)
    stats = _per_layer_alignment(activations, hdb, x_star, n_layers)
    x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x, stats["noise_to_xstar"], color="#7C3AED", linewidth=2.0,
            label="mean IP to x* (global attractor)")
    ax.plot(x, stats["noise_to_best_cluster"], color="#059669", linewidth=2.0,
            label="mean IP to nearest cluster centroid (best of all clusters)")
    ax.plot(x, stats["noise_to_noise"], color="#DC2626", linewidth=2.0,
            label="mean IP to other unclustered tokens")
    ax.set_ylabel("mean inner product")
    ax.set_xlabel("Layer")
    ax.set_title(
        f"What pulls on unclustered tokens — {model} | {prompt}\n"
        f"whichever line sits highest is the stronger attractor for this "
        f"population at that depth",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="best")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"unclustered_pull_{_safe_model_name(model)}_{prompt}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")
