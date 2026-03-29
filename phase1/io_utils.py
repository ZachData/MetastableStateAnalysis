"""
io_utils.py — Save and load run artifacts.

Functions
---------
save_run        : persist metrics, activations, attentions, clusters, tokens, CSV
load_run        : restore results dict from a run directory
load_activations: load raw L2-normed activation array
load_attentions : load raw attention array
load_clusters   : load per-layer cluster labels and KMeans centroids
replot_all      : regenerate all plots from a saved run (no model needed)
"""

import csv
import json
import numpy as np
import torch
from pathlib import Path

from core.config import BETA_VALUES
from core.models import layernorm_to_sphere


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
      clusters.npz         — per-layer cluster labels and KMeans centroids
      centroid_trajectories.npz — P1-1: HDBSCAN centroid trajectories across layers
      plateau_attentions.npz    — P1-7: raw attention matrices at plateau layers
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

    # --- clusters.npz ---
    # Per-layer cluster labels and KMeans centroids, keyed by layer index.
    # Labels are (n_tokens,) int32 arrays.
    # KMeans centroids are (k, d) float32 arrays on S^{d-1}.
    # HDBSCAN labels use -1 for noise tokens.
    # Arrays are named: kmeans_labels_L{i}, kmeans_centroids_L{i}, hdbscan_labels_L{i}.
    # This file is the primary input for Phase 5 cluster identity analysis.
    cluster_arrays = {}
    for layer in results["layers"]:
        i  = layer["layer"]
        km = layer["clustering"]["kmeans"]
        if "labels" in km:
            cluster_arrays[f"kmeans_labels_L{i}"] = np.array(
                km["labels"], dtype=np.int32
            )
        if "cluster_centroids_kmeans" in layer:
            cluster_arrays[f"kmeans_centroids_L{i}"] = np.array(
                layer["cluster_centroids_kmeans"], dtype=np.float32
            )
        hdb = layer["clustering"].get("hdbscan", {})
        if "labels" in hdb:
            cluster_arrays[f"hdbscan_labels_L{i}"] = np.array(
                hdb["labels"], dtype=np.int32
            )
    if cluster_arrays:
        np.savez_compressed(run_dir / "clusters.npz", **cluster_arrays)

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

    # --- centroid_trajectories.npz (P1-1) ---
    # Compute HDBSCAN centroid coordinates for each tracked trajectory.
    # Requires the normed activations and HDBSCAN label arrays.
    tracking = results.get("cluster_tracking", {})
    if tracking.get("trajectories"):
        from .cluster_tracking import compute_centroid_trajectories

        # Build label arrays from results
        label_arrays = []
        for layer in results["layers"]:
            hdb = layer["clustering"].get("hdbscan", {})
            if "labels" in hdb:
                label_arrays.append(np.array(hdb["labels"], dtype=np.int32))
            else:
                label_arrays.append(np.zeros(results["n_tokens"], dtype=np.int32))

        centroid_trajs = compute_centroid_trajectories(
            tracking, hidden_states, label_arrays,
        )
        if centroid_trajs:
            ct_arrays = {}
            for tid, coords in centroid_trajs.items():
                ct_arrays[f"traj_{tid}"] = coords
            np.savez_compressed(run_dir / "centroid_trajectories.npz", **ct_arrays)

    # --- plateau_attentions.npz (P1-7) ---
    # Save raw attention matrices at plateau layers for Phase 3 crosscoder
    # interpretation.  Plateau layers are identified during analysis.
    plateau_layers = results.get("plateau_layers", [])
    if plateau_layers and attentions:
        plateau_attn_arrays = {}
        for li in plateau_layers:
            if li < len(attentions):
                a = attentions[li]
                plateau_attn_arrays[f"attn_L{li}"] = (
                    a.numpy() if hasattr(a, 'numpy') else np.asarray(a)
                )
        if plateau_attn_arrays:
            np.savez_compressed(
                run_dir / "plateau_attentions.npz", **plateau_attn_arrays,
            )


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

    # v_spectrum was added in Phase 2 prep; old runs won't have it
    if "v_spectrum" not in results:
        results["v_spectrum"] = {}

    # Backward compatibility: runs saved before CKA was added have no cka_prev.
    # Default to nan so all downstream code (plotting, reporting) handles it
    # the same way it handles the suppressed-degenerate case.
    for layer in results.get("layers", []):
        if "cka_prev" not in layer:
            layer["cka_prev"] = float("nan")
        # energy_drop_pairs was added later; default to empty dict for old runs.
        # Runs saved before the all-betas extension stored a flat list for
        # beta=1.0 only — migrate those to the new {beta: [...]} dict format.
        if "energy_drop_pairs" not in layer:
            layer["energy_drop_pairs"] = {}
        elif isinstance(layer["energy_drop_pairs"], list):
            # Old format: flat list for beta=1.0 only — migrate to dict
            old = layer["energy_drop_pairs"]
            layer["energy_drop_pairs"] = {1.0: old} if old else {}
        elif isinstance(layer["energy_drop_pairs"], dict):
            # JSON stringifies float keys — rehydrate {"1.0": [...]} → {1.0: [...]}
            layer["energy_drop_pairs"] = {
                float(k): v for k, v in layer["energy_drop_pairs"].items()
            }
        # JSON serialization converts float dict keys to strings ("1.0" etc.).
        # Rehydrate back to float so downstream float-keyed lookups don't silently
        # return nan or raise KeyError (e.g. layer["energies"][1.0]).
        if "energies" in layer:
            layer["energies"] = {float(k): v for k, v in layer["energies"].items()}

        # P1-3: nesting — default to empty for old runs
        if "nesting" not in layer:
            layer["nesting"] = {
                "global_spectral_k": layer.get("spectral", {}).get("k_eigengap", 1),
                "per_cluster": {},
                "has_nesting": False,
                "nesting_summary": "not computed (old run)",
                "n_clusters_with_substructure": 0,
            }
        # P1-4: pair agreement — default to empty for old runs
        if "pair_agreement" not in layer:
            layer["pair_agreement"] = {
                "mutual_pairs": [],
                "n_semantic": 0,
                "n_artifact": 0,
                "n_noise": 0,
                "artifact_fraction": 0.0,
            }

    # P1-1: cluster tracking — default to empty for old runs
    if "cluster_tracking" not in results:
        results["cluster_tracking"] = {
            "events": [],
            "trajectories": [],
            "summary": {"total_births": 0, "total_deaths": 0, "total_merges": 0,
                         "max_alive": 0, "n_trajectories": 0,
                         "mean_lifespan": 0.0, "max_lifespan": 0},
        }
    # P1-7: plateau layers — default to empty for old runs
    if "plateau_layers" not in results:
        results["plateau_layers"] = []

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


def load_clusters(run_dir: Path) -> dict:
    """
    Load per-layer cluster labels and centroids from clusters.npz.

    Returns
    -------
    dict with keys:
      kmeans_labels    : list of (n_tokens,) int32 arrays, one per layer
      kmeans_centroids : list of (k, d) float32 arrays, one per layer
      hdbscan_labels   : list of (n_tokens,) int32 arrays, one per layer
                         (-1 = noise); empty list if HDBSCAN was not run

    Arrays are ordered by layer index (layer 0 first).  Missing layers
    (e.g. old runs without clusters.npz) raise FileNotFoundError.
    """
    path = Path(run_dir) / "clusters.npz"
    data = np.load(path)

    # Determine layer count from the kmeans_labels keys
    layer_indices = sorted(
        int(k.split("_L")[1])
        for k in data.files
        if k.startswith("kmeans_labels_L")
    )

    kmeans_labels    = [data[f"kmeans_labels_L{i}"]    for i in layer_indices]
    kmeans_centroids = [
        data[f"kmeans_centroids_L{i}"]
        for i in layer_indices
        if f"kmeans_centroids_L{i}" in data.files
    ]
    hdbscan_labels   = [
        data[f"hdbscan_labels_L{i}"]
        for i in layer_indices
        if f"hdbscan_labels_L{i}" in data.files
    ]

    return {
        "kmeans_labels":    kmeans_labels,
        "kmeans_centroids": kmeans_centroids,
        "hdbscan_labels":   hdbscan_labels,
    }


def load_centroid_trajectories(run_dir: Path) -> dict:
    """
    Load HDBSCAN centroid trajectory coordinates from centroid_trajectories.npz.

    Returns
    -------
    dict mapping trajectory_id (int) -> (lifespan, d) float32 array
    Raises FileNotFoundError if file does not exist (old runs).
    """
    path = Path(run_dir) / "centroid_trajectories.npz"
    data = np.load(path)
    result = {}
    for key in data.files:
        tid = int(key.split("_")[1])
        result[tid] = data[key]
    return result


def load_plateau_attentions(run_dir: Path) -> dict:
    """
    Load raw attention matrices at plateau layers from plateau_attentions.npz.

    Returns
    -------
    dict mapping layer_index (int) -> (n_heads, n_tokens, n_tokens) float32 array
    Raises FileNotFoundError if file does not exist (old runs or no plateaus).
    """
    path = Path(run_dir) / "plateau_attentions.npz"
    data = np.load(path)
    result = {}
    for key in data.files:
        li = int(key.split("_L")[1])
        result[li] = data[key]
    return result


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
    from .plots import (
        plot_trajectory,
        plot_ip_histograms,
        plot_pca_panels,
        plot_sinkhorn_detail,
        plot_spectral_eigengap,
        plot_cka_trajectory,
    )
    from .reporting import print_summary

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
