"""
io_utils.py — Save and load run artifacts.

Functions
---------
save_run        : persist metrics, activations, attentions, tokens, CSV
load_run        : restore results dict from a run directory
load_activations: load raw L2-normed activation array
load_attentions : load raw attention array
replot_all      : regenerate all plots from a saved run (no model needed)
"""

import csv
import json
import numpy as np
import torch
from pathlib import Path

from config import BETA_VALUES
from models import layernorm_to_sphere


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_run(
    results: dict,
    hidden_states: list,
    attentions: list,
    run_dir: Path,
) -> None:
    """
    Persist everything needed to reproduce plots and reports later.

    Written files:
      metrics.json         — full results dict (JSON-serialisable)
      activations.npz      — L2-normed hidden states (n_layers, n_tokens, d)
      attentions.npz       — attention weights (n_layers, n_heads, n, n)
      tokens.txt           — one token per line with index
      layer_metrics.csv    — flat CSV of key per-layer scalars
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- metrics.json ---
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- activations.npz ---
    # Stack all layers into a single tensor, normalize in one fused op on
    # whatever device the tensors live on, then transfer once.
    stacked   = torch.stack(hidden_states)                  # (n_layers, n_tokens, d)
    act_stack = layernorm_to_sphere(stacked).numpy()        # single normalize call
    np.savez_compressed(run_dir / "activations.npz", activations=act_stack)

    # --- attentions.npz ---
    if attentions:
        attn_stack = np.stack([a.numpy() for a in attentions])
        np.savez_compressed(run_dir / "attentions.npz", attentions=attn_stack)

    # --- tokens.txt ---
    with open(run_dir / "tokens.txt", "w") as f:
        for i, tok in enumerate(results["tokens"]):
            f.write(f"{i:3d}  {tok}\n")

    # --- layer_metrics.csv ---
    csv_rows = []
    for layer in results["layers"]:
        row = {
            "layer":             layer["layer"],
            "ip_mean":           layer["ip_mean"],
            "ip_std":            layer["ip_std"],
            "ip_mass_near_1":    layer["ip_mass_near_1"],
            "effective_rank":    layer["effective_rank"],
            "spectral_k":        layer["spectral"]["k_eigengap"],
            "hdbscan_k":         layer["clustering"].get("hdbscan", {}).get("n_clusters", ""),
            "kmeans_k":          layer["clustering"]["kmeans"]["best_k"],
            "kmeans_silhouette": layer["clustering"]["kmeans"]["best_silhouette"],
            "nn_stability":      layer.get("nn_stability", ""),
            "cka":               layer.get("cka_prev", ""),
        }
        for beta in BETA_VALUES:
            row[f"energy_beta{beta}"] = layer["energies"][beta]
        if "sinkhorn" in layer:
            row["fiedler_mean"]      = layer["sinkhorn"]["fiedler_mean"]
            row["sinkhorn_k_mean"]   = layer["sinkhorn"]["sinkhorn_cluster_count_mean"]
            row["attn_entropy_mean"] = layer.get("attention_entropy_mean", "")
            row["row_col_balance"]   = layer["sinkhorn"]["row_col_balance_mean"]
        csv_rows.append(row)

    if csv_rows:
        with open(run_dir / "layer_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> dict:
    """
    Restore a results dict from a saved run directory.

    Parameters
    ----------
    run_dir : path to e.g. metastability_results/albert-base-v2_wiki_paragraph/

    Returns
    -------
    results dict in the same format produced by analysis.analyze_trajectory
    """
    run_dir = Path(run_dir)
    with open(run_dir / "metrics.json") as f:
        results = json.load(f)

    if "pca_trajectories" not in results:
        results["pca_trajectories"] = []

    # Backward compatibility: runs saved before CKA was added have no cka_prev.
    # Default to nan so all downstream code (plotting, reporting) handles it
    # the same way it handles the suppressed-degenerate case.
    for layer in results.get("layers", []):
        if "cka_prev" not in layer:
            layer["cka_prev"] = float("nan")

    print(f"Loaded: {results['model']} | {results['prompt']}")
    print(f"  {results['n_layers']} layers, {results['n_tokens']} tokens, "
          f"d={results['d_model']}")
    return results


def load_activations(run_dir: Path) -> np.ndarray:
    """
    Load raw L2-normed activations.

    Returns
    -------
    (n_layers, n_tokens, d_model) float32 array
    """
    data = np.load(Path(run_dir) / "activations.npz")
    return data["activations"]


def load_attentions(run_dir: Path) -> np.ndarray:
    """
    Load raw attention weights.

    Returns
    -------
    (n_layers, n_heads, n_tokens, n_tokens) float32 array
    """
    data = np.load(Path(run_dir) / "attentions.npz")
    return data["attentions"]


# ---------------------------------------------------------------------------
# Replot from saved run
# ---------------------------------------------------------------------------

def replot_all(run_dir: Path, out_dir: Path = None) -> None:
    """
    Recreate every plot from a saved run directory.
    No model loading required — everything comes from saved files.

    Usage:
        python run.py --replot metastability_results/albert-base-v2_wiki_paragraph
    """
    from plots import (
        plot_trajectory,
        plot_ip_histograms,
        plot_pca_panels,
        plot_sinkhorn_detail,
        plot_spectral_eigengap,
        plot_cka_trajectory,
    )
    from reporting import print_summary

    run_dir = Path(run_dir)
    out_dir = out_dir or run_dir
    results = load_run(run_dir)

    print("Regenerating plots...")
    plot_trajectory(results, out_dir)
    plot_ip_histograms(results, out_dir)
    plot_pca_panels(results, out_dir)
    plot_sinkhorn_detail(results, out_dir)
    plot_spectral_eigengap(results, out_dir)
    plot_cka_trajectory(results, out_dir)
    print_summary(results)
    print(f"Done. Plots written to {out_dir}/")
