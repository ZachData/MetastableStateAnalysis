"""
plots.py — All figure-generation functions.

Each function takes a results dict (from analysis.analyze_trajectory) and
a save_dir Path.  Figures are written to disk; nothing is returned.

Functions
---------
plot_trajectory          : 4×4 summary panel for one run
plot_ip_histograms       : replicate paper Figure 1
plot_pca_panels          : token PCA positions at selected layers
plot_sinkhorn_detail     : per-head Fiedler heatmap
plot_spectral_eigengap   : eigenvalue spectrum + eigengap k per layer
plot_eigenvalue_spectra       : per-layer eigenvalue bar charts
plot_albert_extended          : multi-iteration comparison (ALBERT only)
plot_cross_model_comparison   : side-by-side metric curves across models
analyze_value_eigenspectrum   : singular value histograms of V matrices
plot_cka_trajectory           : CKA per layer with degenerate-region shading
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.linalg import svdvals

from config import BETA_VALUES, DISTANCE_THRESHOLDS, SPECTRAL_MAX_K
from models import layernorm_to_sphere
from metrics import pairwise_inner_products, effective_rank, effective_rank_from_raw


# ---------------------------------------------------------------------------
# Main trajectory summary panel
# ---------------------------------------------------------------------------

def plot_trajectory(results: dict, save_dir: Path):
    """4×4 panel: inner products, rank, energy, clustering, Sinkhorn, PCA, spectral."""
    model    = results["model"]
    prompt   = results["prompt"]
    n_layers = results["n_layers"]
    layers   = list(range(n_layers))

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        f"{model} | {prompt}\n"
        f"({results['n_tokens']} tokens, d={results['d_model']})",
        fontsize=10,
    )
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.38)

    def ax_plot(pos, y, title, color, ylabel=None):
        ax = fig.add_subplot(gs[pos])
        ax.plot(layers[:len(y)], y, color=color, linewidth=1.2)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Layer", fontsize=7)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=7)
        return ax

    # Mean ± std of pairwise inner products
    ip_means = [r["ip_mean"] for r in results["layers"]]
    ip_stds  = [r["ip_std"]  for r in results["layers"]]
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(layers, ip_means, color="steelblue")
    ax.fill_between(
        layers,
        np.array(ip_means) - np.array(ip_stds),
        np.array(ip_means) + np.array(ip_stds),
        alpha=0.2, color="steelblue",
    )
    ax.set_title("Mean ⟨xᵢ, xⱼ⟩ ± std", fontsize=8)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Layer", fontsize=7)

    ax_plot((0, 1), [r["ip_mass_near_1"]  for r in results["layers"]],
            "Fraction pairs > 0.9", "firebrick")
    ax_plot((0, 2), [r["effective_rank"]  for r in results["layers"]],
            "Effective rank", "darkgreen")

    # Interaction energy per beta
    ax = fig.add_subplot(gs[0, 3])
    for beta, c in zip(BETA_VALUES, ["purple", "darkorange", "teal", "steelblue"]):
        ax.plot(layers, [r["energies"][beta] for r in results["layers"]],
                label=f"β={beta}", color=c)
    ax.set_title("Interaction energy Eβ", fontsize=8)
    ax.legend(fontsize=6)
    ax.set_xlabel("Layer", fontsize=7)

    mid_thresh = float(DISTANCE_THRESHOLDS[len(DISTANCE_THRESHOLDS) // 2])
    ax_plot(
        (1, 0),
        [r["clustering"]["agglomerative"].get(mid_thresh, np.nan)
         for r in results["layers"]],
        f"Agglomerative k\n(t={mid_thresh:.2f})", "brown",
    )
    ax = fig.add_subplot(gs[1, 1])
    ax.step(layers, [r["spectral"]["k_eigengap"]              for r in results["layers"]],
            color="navy",       where="mid", label="k")
    ax.step(layers, [r["spectral"].get("k_second_gap", 1)     for r in results["layers"]],
            color="darkorange", where="mid", label="k₂", linestyle="--")
    ax.set_title("Spectral k (eigengap)\nsolid=full  dashed=skip Δλ₁", fontsize=8)
    ax.set_xlabel("Layer", fontsize=7)
    ax.legend(fontsize=6)

    has_sk    = [r for r in results["layers"] if "sinkhorn" in r]
    sk_layers = [r["layer"] for r in has_sk]

    if has_sk:
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(sk_layers, [r["sinkhorn"]["fiedler_mean"] for r in has_sk],
                color="darkmagenta", linewidth=1.2)
        ax.set_title("Sinkhorn Fiedler value\n(↓ = cluster-separated)", fontsize=8)
        ax.set_xlabel("Layer", fontsize=7)

        ax = fig.add_subplot(gs[1, 3])
        ax.plot(sk_layers,
                [r["sinkhorn"]["sinkhorn_cluster_count_mean"] for r in has_sk],
                color="darkolivegreen", linewidth=1.2)
        ax.set_title("Sinkhorn cluster count\n(from attention eigenvalues)", fontsize=8)
        ax.set_xlabel("Layer", fontsize=7)

    # Attention entropy
    if has_sk:
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(sk_layers, [r["attention_entropy_mean"] for r in has_sk],
                color="darkcyan")
        ax.set_title("Mean attention entropy", fontsize=8)
        ax.set_xlabel("Layer", fontsize=7)

        ax = fig.add_subplot(gs[2, 1])
        ax.plot(sk_layers, [r["sinkhorn"]["row_col_balance_mean"] for r in has_sk],
                color="sienna")
        ax.set_title("Attn col-sum std\n(0 = doubly stochastic)", fontsize=8)
        ax.set_xlabel("Layer", fontsize=7)

    # PCA explained variance
    ax = fig.add_subplot(gs[2, 2])
    for ci, color in enumerate(["red", "blue", "green"]):
        var = [
            r["pca_explained_variance"][ci]
            if ci < len(r["pca_explained_variance"]) else 0
            for r in results["layers"]
        ]
        ax.plot(layers, var, color=color, label=f"PC{ci+1}", linewidth=0.9)
    ax.set_title("PCA explained variance", fontsize=8)
    ax.legend(fontsize=6)
    ax.set_xlabel("Layer", fontsize=7)

    # Spectral eigenvalue ladder
    ax = fig.add_subplot(gs[2, 3])
    for ei in range(min(5, SPECTRAL_MAX_K)):
        eig_vals = [
            r["spectral"]["eigenvalues"][ei]
            if ei < len(r["spectral"]["eigenvalues"]) else np.nan
            for r in results["layers"]
        ]
        ax.plot(layers, eig_vals, linewidth=0.8, label=f"λ{ei+1}", alpha=0.8)
    ax.set_title("Laplacian eigenvalues", fontsize=8)
    ax.legend(fontsize=5)
    ax.set_xlabel("Layer", fontsize=7)

    # Threshold × layer heatmap
    ax         = fig.add_subplot(gs[3, :])
    thresh_list = sorted(DISTANCE_THRESHOLDS)
    heatmap     = np.zeros((len(thresh_list), n_layers))
    for li, r in enumerate(results["layers"]):
        for ti, t in enumerate(thresh_list):
            heatmap[ti, li] = r["clustering"]["agglomerative"].get(float(t), np.nan)
    im = ax.imshow(heatmap, aspect="auto", origin="lower",
                   cmap="viridis", interpolation="nearest")
    ax.set_title("Agglomerative cluster count: threshold × layer", fontsize=8)
    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Threshold index", fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)

    fname = save_dir / f"{model.replace('/', '_')}_{prompt}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Inner-product histograms (paper Figure 1)
# ---------------------------------------------------------------------------

def plot_ip_histograms(results: dict, save_dir: Path, n_panels: int = 8):
    """Replicate paper Figure 1: ⟨xᵢ, xⱼ⟩ histograms at selected layers."""
    model    = results["model"]
    prompt   = results["prompt"]
    n_layers = results["n_layers"]
    indices  = np.linspace(0, n_layers - 1, n_panels, dtype=int)

    bins        = np.linspace(-1, 1, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(2, n_panels // 2, figsize=(16, 5))
    axes = axes.flatten()
    fig.suptitle(f"{model} | {prompt} — ⟨xᵢ, xⱼ⟩ histograms", fontsize=9)

    for ax, li in zip(axes, indices):
        counts = np.array(results["layers"][li]["ip_histogram"])
        ax.bar(bin_centers, counts, width=bins[1] - bins[0],
               color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_title(f"Layer {li}", fontsize=7)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("⟨xᵢ, xⱼ⟩", fontsize=6)

    fname = save_dir / f"{model.replace('/', '_')}_{prompt}_histograms.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# PCA panels
# ---------------------------------------------------------------------------

def plot_pca_panels(results: dict, save_dir: Path, n_panels: int = 8):
    """Token positions in PC1–PC2 space at selected layers."""
    model    = results["model"]
    prompt   = results["prompt"]
    tokens   = results["tokens"]
    n_tokens = results["n_tokens"]
    n_layers = results["n_layers"]

    indices = np.linspace(0, n_layers - 1, n_panels, dtype=int)
    colors  = plt.cm.tab20(np.linspace(0, 1, n_tokens))

    fig, axes = plt.subplots(2, n_panels // 2, figsize=(18, 6))
    axes = axes.flatten()
    fig.suptitle(f"{model} | {prompt} — PCA projections (PC1 vs PC2)", fontsize=9)

    for ax, li in zip(axes, indices):
        proj = np.array(results["pca_trajectories"][li])
        ax.scatter(proj[:, 0], proj[:, 1], c=colors[:n_tokens], s=30, zorder=3)
        for ti, (x, y) in enumerate(zip(proj[:, 0], proj[:, 1])):
            label = tokens[ti][:5] if ti < len(tokens) else str(ti)
            ax.annotate(label, (x, y), fontsize=4, alpha=0.7)
        ax.set_title(f"Layer {li}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fname = save_dir / f"{model.replace('/', '_')}_{prompt}_pca.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Sinkhorn per-head detail
# ---------------------------------------------------------------------------

def plot_sinkhorn_detail(results: dict, save_dir: Path):
    """Per-head Fiedler value heatmap: heads × layers."""
    model  = results["model"]
    prompt = results["prompt"]

    has_sk = [r for r in results["layers"] if "sinkhorn" in r]
    if not has_sk:
        return

    n_heads   = len(has_sk[0]["sinkhorn"]["fiedler_per_head"])
    sk_layers = [r["layer"] for r in has_sk]

    heatmap = np.zeros((n_heads, len(sk_layers)))
    for li, r in enumerate(has_sk):
        for h, fv in enumerate(r["sinkhorn"]["fiedler_per_head"]):
            heatmap[h, li] = fv

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(heatmap, aspect="auto", cmap="coolwarm_r",
                   interpolation="nearest")
    ax.set_title(
        f"{model} | {prompt}\n"
        f"Fiedler value per head × layer  (blue = cluster-separated)",
        fontsize=9,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    plt.colorbar(im, ax=ax, shrink=0.8)

    fname = save_dir / f"{model.replace('/', '_')}_{prompt}_sinkhorn_heads.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Spectral eigengap
# ---------------------------------------------------------------------------

def plot_spectral_eigengap(results: dict, save_dir: Path):
    """Eigenvalue spectrum per layer + eigengap-derived k estimate."""
    model    = results["model"]
    prompt   = results["prompt"]
    n_layers = results["n_layers"]

    max_eigs = SPECTRAL_MAX_K
    heatmap  = np.zeros((max_eigs, n_layers))
    for li, r in enumerate(results["layers"]):
        evs = r["spectral"]["eigenvalues"][:max_eigs]
        heatmap[:len(evs), li] = evs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle(f"{model} | {prompt} — Spectral structure (Gram matrix)", fontsize=9)

    im = ax1.imshow(heatmap, aspect="auto", cmap="plasma",
                    interpolation="nearest", origin="lower")
    ax1.set_title("Laplacian eigenvalues across layers", fontsize=8)
    ax1.set_xlabel("Layer", fontsize=7)
    ax1.set_ylabel("Eigenvalue index", fontsize=7)
    plt.colorbar(im, ax=ax1, shrink=0.8)

    spectral_k   = [r["spectral"]["k_eigengap"]   for r in results["layers"]]
    spectral_k2  = [r["spectral"].get("k_second_gap", 1) for r in results["layers"]]
    ax2.step(range(n_layers), spectral_k,  color="navy",       where="mid", label="k (incl. Δλ₁)")
    ax2.step(range(n_layers), spectral_k2, color="darkorange", where="mid", label="k₂ (skip Δλ₁)", linestyle="--")
    ax2.set_title("k estimated from eigengap heuristic\n(dashed = skipping trivial zero-mode gap)", fontsize=8)
    ax2.set_xlabel("Layer", fontsize=7)
    ax2.set_ylabel("k", fontsize=7)
    ax2.legend(fontsize=6)

    fname = save_dir / f"{model.replace('/', '_')}_{prompt}_spectral.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Eigenvalue spectrum per layer — is the eigengap real?
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectra(results: dict, save_dir: Path, n_panels: int = 8):
    """
    Plot the raw Laplacian eigenvalue spectrum at selected layers.

    Two rows per panel:
      Row 1 — bar chart of eigenvalues.  Bars ≤ k are blue; bars > k are coral.
               A dashed vertical line marks the chosen gap location.
      Row 2 — bar chart of eigengaps (Δλ between consecutive eigenvalues).
               The dominant gap (used to pick k) is highlighted in gold.

    Reading this plot:
      Sharp spike in the gaps row  →  genuine cluster structure.
      Smooth, monotone gaps row    →  eigengap heuristic is picking noise;
                                      the k=2 plateau is an artifact.
    """
    model    = results["model"]
    prompt   = results["prompt"]
    n_layers = results["n_layers"]
    indices  = np.linspace(0, n_layers - 1, n_panels, dtype=int)

    fig, axes = plt.subplots(2, n_panels, figsize=(n_panels * 2.5, 6))
    fig.suptitle(
        f"{model} | {prompt}\n"
        "Eigenvalue spectra per layer  —  blue ≤ k  |  coral > k  |  gold = dominant gap",
        fontsize=9,
    )

    for col, li in enumerate(indices):
        r    = results["layers"][li]
        evs  = np.array(r["spectral"]["eigenvalues"])
        gaps = np.array(r["spectral"]["eigengaps"])
        k    = r["spectral"]["k_eigengap"]
        n_ev = len(evs)

        # ── Row 0: eigenvalue magnitudes ──────────────────────────────────
        ax0 = axes[0, col]
        bar_colors = ["steelblue" if i < k else "lightcoral" for i in range(n_ev)]
        ax0.bar(range(n_ev), evs, color=bar_colors, edgecolor="none", width=0.75)
        if 0 < k < n_ev:
            ax0.axvline(k - 0.5, color="black", lw=1.0, ls="--", alpha=0.65)
        ax0.set_title(f"L{li}  k={k}", fontsize=7)
        ax0.set_xticks(range(n_ev))
        ax0.set_xticklabels([str(i + 1) for i in range(n_ev)], fontsize=4)
        ax0.tick_params(axis="y", labelsize=5)
        ax0.set_xlabel("λ index", fontsize=5)
        if col == 0:
            ax0.set_ylabel("λ value", fontsize=6)

        # ── Row 1: eigengaps ───────────────────────────────────────────────
        ax1 = axes[1, col]
        if len(gaps) > 0:
            dominant = int(np.argmax(gaps))
            gap_colors = ["gold" if i == dominant else "steelblue"
                          for i in range(len(gaps))]
            ax1.bar(range(len(gaps)), gaps, color=gap_colors,
                    edgecolor="none", width=0.75)
        ax1.set_title("Δλ (gaps)", fontsize=7)
        ax1.set_xticks(range(len(gaps)))
        ax1.set_xticklabels([str(i + 1) for i in range(len(gaps))], fontsize=4)
        ax1.tick_params(axis="y", labelsize=5)
        ax1.set_xlabel("gap index", fontsize=5)
        if col == 0:
            ax1.set_ylabel("Δλ", fontsize=6)

    # Shared legend on the first eigenvalue panel
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="steelblue", label="≤ k"),
        Patch(facecolor="lightcoral", label="> k"),
        Patch(facecolor="gold",       label="dominant gap"),
    ]
    axes[0, 0].legend(handles=legend_elems, fontsize=4, loc="upper right")

    fname = (
        save_dir
        / f"{model.replace('/', '_')}_{prompt}_eigenvalue_spectra.png"
    )
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# ALBERT extended-iteration comparison
# ---------------------------------------------------------------------------

def plot_albert_extended(trajectories: dict, save_dir: Path):
    """
    Multi-iteration comparison plot for ALBERT.

    Parameters
    ----------
    trajectories : {n_iter: list of (n_tokens, d_model) tensors}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("ALBERT: extended iterations (shared weights)", fontsize=10)
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(trajectories)))

    for ax, metric in zip(axes, ["ip_mean", "ip_mass_near_1", "effective_rank"]):
        for (n_iter, traj), color in zip(trajectories.items(), colors):
            normed = [layernorm_to_sphere(h) for h in traj]
            if metric == "ip_mean":
                vals = [pairwise_inner_products(h).mean() for h in normed]
                ax.set_title("Mean ⟨xᵢ, xⱼ⟩", fontsize=8)
            elif metric == "ip_mass_near_1":
                vals = [(pairwise_inner_products(h) > 0.9).mean() for h in normed]
                ax.set_title("Fraction pairs > 0.9", fontsize=8)
            else:
                vals = [effective_rank_from_raw(h) for h in traj]
                ax.set_title("Effective rank", fontsize=8)
            ax.plot(range(len(vals)), vals, label=f"{n_iter} iters", color=color)
        ax.set_xlabel("Iteration", fontsize=7)
        ax.legend(fontsize=7)

    fname = save_dir / "albert_extended_iterations.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

def plot_cross_model_comparison(all_results: list, save_dir: Path):
    """Side-by-side metric curves for all models on each prompt."""
    from collections import defaultdict
    by_prompt = defaultdict(list)
    for r in all_results:
        by_prompt[r["prompt"]].append(r)

    for prompt_key, rlist in by_prompt.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Cross-model | {prompt_key}", fontsize=9)
        colors = plt.cm.tab10(np.linspace(0, 1, len(rlist)))

        for r, color in zip(rlist, colors):
            layers = list(range(r["n_layers"]))
            label  = r["model"].split("/")[-1]
            axes[0].plot(layers, [l["ip_mass_near_1"] for l in r["layers"]],
                         label=label, color=color)
            axes[1].plot(layers, [l["effective_rank"] for l in r["layers"]],
                         label=label, color=color)
            axes[2].step(layers, [l["spectral"]["k_eigengap"] for l in r["layers"]],
                         label=label, color=color, where="mid")

        for ax, title in zip(axes, [
            "Fraction pairs > 0.9", "Effective rank", "Spectral k (eigengap)"
        ]):
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("Layer", fontsize=7)
            ax.legend(fontsize=6)

        fname = save_dir / f"cross_model_{prompt_key}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# CKA trajectory
# ---------------------------------------------------------------------------

def plot_cka_trajectory(results: dict, save_dir: Path):
    """
    Two-panel figure showing CKA layer-to-layer similarity.

    Panel 1 — CKA per layer
      Each point is CKA(layer L, layer L-1).  Layer 0 is nan (no predecessor).
      Degenerate layers (effective_rank < 3) are shown as a grey shaded region
      — CKA is suppressed there to avoid noise-dominated ratios.
      A horizontal dashed line at y=1 marks perfect layer-to-layer identity.
      A vertical marker flags the sharpest CKA drop (end of metastable plateau).

    Panel 2 — Effective rank (reference)
      Drawn on the same x-axis so the rank collapse can be compared directly
      with the layer where CKA values become nan.  The rank=3 suppression
      threshold is marked with a dashed line.
    """
    model    = results["model"]
    prompt   = results["prompt"]
    layers_r = results["layers"]
    n_layers = results["n_layers"]
    xs       = list(range(n_layers))

    cka_vals  = [r.get("cka_prev", float("nan")) for r in layers_r]
    erank     = [r["effective_rank"]              for r in layers_r]

    # Find valid (non-nan) CKA pairs for drop detection
    valid_pairs = [(i, v) for i, v in enumerate(cka_vals) if not np.isnan(v)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(
        f"{model} | {prompt}\n"
        f"CKA layer-to-layer similarity  ({results['n_tokens']} tokens, d={results['d_model']})",
        fontsize=9,
    )

    # --- Panel 1: CKA ---
    valid_xs = [i for i, v in enumerate(cka_vals) if not np.isnan(v)]
    valid_vs = [v for v in cka_vals if not np.isnan(v)]

    ax1.plot(valid_xs, valid_vs, color="steelblue", linewidth=1.5, zorder=3)
    ax1.scatter(valid_xs, valid_vs, color="steelblue", s=18, zorder=4)
    ax1.axhline(1.0, color="gray", lw=0.7, ls="--", alpha=0.6, label="CKA=1 (identical)")

    # Shade degenerate (suppressed) regions
    in_degen = False
    degen_start = None
    for i, v in enumerate(cka_vals):
        if np.isnan(v) and not in_degen:
            in_degen = True
            degen_start = i - 0.5
        elif not np.isnan(v) and in_degen:
            ax1.axvspan(degen_start, i - 0.5, alpha=0.12, color="gray",
                        label="degenerate (rank<3, CKA suppressed)")
            in_degen = False
    if in_degen:  # still degenerate at end
        ax1.axvspan(degen_start, n_layers - 0.5, alpha=0.12, color="gray",
                    label="degenerate (rank<3, CKA suppressed)")

    # Mark sharpest drop
    if len(valid_pairs) >= 2:
        idxs  = [i for i, _ in valid_pairs]
        vals  = [v for _, v in valid_pairs]
        diffs = np.diff(vals)
        drop_pos = int(np.argmin(diffs))
        if diffs[drop_pos] < -0.05:
            drop_layer = idxs[drop_pos + 1]
            severity = "SHARP" if diffs[drop_pos] < -0.15 else "MILD"
            ax1.axvline(drop_layer, color="firebrick", lw=1.2, ls=":",
                        label=f"{severity} drop  (L{idxs[drop_pos]}→L{drop_layer}, "
                              f"Δ={diffs[drop_pos]:.3f})")

    ax1.set_ylabel("CKA(L, L-1)", fontsize=8)
    ax1.set_ylim(-0.05, 1.08)
    ax1.legend(fontsize=6, loc="lower left")
    ax1.set_title("Linear CKA between consecutive layers  "
                  "(1 = identical representations, 0 = orthogonal)", fontsize=8)

    # --- Panel 2: Effective rank ---
    ax2.plot(xs, erank, color="darkgreen", linewidth=1.3)
    ax2.axhline(3.0, color="gray", lw=0.7, ls="--", alpha=0.7,
                label="rank=3 suppression threshold")
    # Match degenerate shading to panel 1
    in_degen = False
    degen_start = None
    for i, v in enumerate(cka_vals):
        if np.isnan(v) and i > 0 and not in_degen:
            in_degen = True
            degen_start = i - 0.5
        elif not np.isnan(v) and in_degen:
            ax2.axvspan(degen_start, i - 0.5, alpha=0.12, color="gray")
            in_degen = False
    if in_degen:
        ax2.axvspan(degen_start, n_layers - 0.5, alpha=0.12, color="gray")

    ax2.set_xlabel("Layer", fontsize=8)
    ax2.set_ylabel("Effective rank", fontsize=8)
    ax2.set_title("Effective rank (reference — CKA suppressed where rank < 3)", fontsize=8)
    ax2.legend(fontsize=6, loc="upper right")

    plt.tight_layout()
    fname = save_dir / f"{model.replace('/', '_')}_{prompt}_cka.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Value matrix singular value spectrum
# ---------------------------------------------------------------------------

def analyze_value_eigenspectrum(model, model_name: str, save_dir: Path):
    """Histogram of singular values of V weight matrices per layer."""
    v_matrices = []

    if "albert" in model_name:
        try:
            attn     = model.encoder.albert_layer_groups[0].albert_layers[0].attention
            v_weight = attn.value.weight.detach().cpu().float().numpy()
            v_matrices = [("shared", v_weight)]
        except AttributeError:
            print(f"  Could not extract V from {model_name}")
            return
    elif "bert" in model_name:
        for i, layer in enumerate(model.encoder.layer):
            v = layer.attention.self.value.weight.detach().cpu().float().numpy()
            v_matrices.append((f"layer_{i}", v))
    elif "gpt2" in model_name:
        for i, block in enumerate(model.h):
            d = block.attn.c_attn.weight.shape[1]
            v = block.attn.c_attn.weight[:, 2*d//3:].detach().cpu().float().numpy()
            v_matrices.append((f"layer_{i}", v))

    if not v_matrices:
        return

    n_plot = min(len(v_matrices), 4)
    fig, axes = plt.subplots(1, n_plot, figsize=(4 * n_plot, 4))
    if n_plot == 1:
        axes = [axes]
    fig.suptitle(f"V singular values — {model_name}", fontsize=9)

    for ax, (name, V) in zip(axes, v_matrices[:n_plot]):
        sv = svdvals(V)
        ax.hist(sv, bins=30, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_title(f"{name}\nmax={sv.max():.2f} min={sv.min():.2f}", fontsize=8)
        ax.set_xlabel("Singular value", fontsize=7)

    fname = save_dir / f"V_spectrum_{model_name.replace('/', '_')}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
