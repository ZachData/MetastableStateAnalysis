"""
analysis.py — Layer-wise analysis loop.

analyze_trajectory ingests a list of per-layer hidden states and attentions
and calls every metric/clustering/projection function, collecting results
into a single dict that all downstream plotting and reporting functions accept.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import BETA_VALUES, DISTANCE_THRESHOLDS
from models import layernorm_to_sphere
from metrics import (
    pairwise_inner_products,
    gram_matrix,
    interaction_energy,
    effective_rank,
    attention_entropy,
)
from sinkhorn import analyze_attention_sinkhorn
from spectral import spectral_eigengap_k
from clustering import cluster_count_sweep, pca_projection, umap_projection, HAS_UMAP


def analyze_trajectory(
    hidden_states: list,
    attentions: list,
    prompt_key: str,
    model_name: str,
    tokens: list,
    beta_values: list = BETA_VALUES,
    thresholds: np.ndarray = DISTANCE_THRESHOLDS,
    umap_dir: Path = None,
) -> dict:
    """
    Compute all per-layer metrics for one (model, prompt) run.

    Parameters
    ----------
    hidden_states : list of (n_tokens, d_model) float tensors, one per layer
    attentions    : list of (n_heads, n_tokens, n_tokens) float tensors
    prompt_key    : string key from PROMPTS
    model_name    : model identifier string
    tokens        : list of decoded token strings
    beta_values   : β values for interaction energy
    thresholds    : cosine-distance thresholds for agglomerative sweep
    umap_dir      : if provided, UMAP projections are saved here as .npy files

    Returns
    -------
    results dict consumed by plots.py, reporting.py, and io_utils.py
    """
    n_layers = len(hidden_states)
    results  = {
        "model":            model_name,
        "prompt":           prompt_key,
        "tokens":           tokens,
        "n_layers":         n_layers,
        "n_tokens":         hidden_states[0].shape[0],
        "d_model":          hidden_states[0].shape[1],
        "layers":           [],
        "pca_trajectories": [],   # (n_layers, n_tokens, 3) nested list
    }

    for layer_idx, activations in enumerate(tqdm(
        hidden_states,
        desc=f"{model_name[:20]} | {prompt_key}",
        leave=False,
    )):
        lr = {"layer": layer_idx}

        # --- Inner products ---
        ips                    = pairwise_inner_products(activations)
        lr["ip_mean"]          = float(ips.mean())
        lr["ip_std"]           = float(ips.std())
        lr["ip_histogram"]     = np.histogram(ips, bins=50, range=(-1, 1))[0].tolist()
        lr["ip_mass_near_1"]   = float((ips > 0.9).mean())

        # --- Gram matrix (shared by spectral methods) ---
        G = gram_matrix(activations)

        # --- Interaction energies ---
        lr["energies"] = {
            beta: interaction_energy(activations, beta)
            for beta in beta_values
        }

        # --- Effective rank ---
        lr["effective_rank"] = effective_rank(activations)

        # --- Standard clustering ---
        lr["clustering"] = cluster_count_sweep(activations, thresholds)

        # --- Spectral eigengap on Gram matrix ---
        lr["spectral"] = spectral_eigengap_k(G)

        # --- PCA projection ---
        proj, var_ratio              = pca_projection(activations, n_components=3)
        lr["pca_explained_variance"] = var_ratio.tolist()
        results["pca_trajectories"].append(proj.tolist())

        # --- UMAP (optional) ---
        if HAS_UMAP and umap_dir is not None and activations.shape[0] >= 4:
            umap_proj = umap_projection(activations, n_components=2)
            if umap_proj is not None:
                np.save(
                    umap_dir / (
                        f"{model_name.replace('/', '_')}"
                        f"_{prompt_key}_layer{layer_idx:02d}.npy"
                    ),
                    umap_proj,
                )

        # --- Attention: entropy + Sinkhorn ---
        if layer_idx < len(attentions):
            attn                             = attentions[layer_idx]
            ent                              = attention_entropy(attn)
            lr["attention_entropy_per_head"] = ent.tolist()
            lr["attention_entropy_mean"]     = float(ent.mean())
            lr["sinkhorn"]                   = analyze_attention_sinkhorn(attn)

        results["layers"].append(lr)

    return results
