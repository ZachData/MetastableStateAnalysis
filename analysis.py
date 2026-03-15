"""
analysis.py — Layer-wise analysis loop.

analyze_trajectory ingests a list of per-layer hidden states and attentions
and calls every metric/clustering/projection function, collecting results
into a single dict that all downstream plotting and reporting functions accept.

Performance notes
-----------------
normed (L2-normalised activations) and G (Gram matrix) are computed ONCE per
layer and threaded through to every downstream function that previously
recomputed them independently.  This eliminates ~8 redundant matrix multiplies
per layer (inner products, ×4 interaction energies, effective rank, clustering,
PCA, UMAP).
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import BETA_VALUES, DISTANCE_THRESHOLDS
from models import layernorm_to_sphere
from metrics import (
    pairwise_inner_products_from_gram,
    interaction_energies_batched,
    effective_rank_from_raw,
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

        # ------------------------------------------------------------------
        # Pre-compute normed activations and Gram matrix ONCE per layer.
        # Every downstream call receives these directly, avoiding ~8
        # redundant layernorm_to_sphere + matmul calls per layer.
        # ------------------------------------------------------------------
        normed = layernorm_to_sphere(activations).numpy()   # (n_tokens, d)
        G      = normed @ normed.T                          # (n_tokens, n_tokens)

        # --- Inner products ---
        ips                  = pairwise_inner_products_from_gram(G)
        lr["ip_mean"]        = float(ips.mean())
        lr["ip_std"]         = float(ips.std())
        lr["ip_histogram"]   = np.histogram(ips, bins=50, range=(-1, 1))[0].tolist()
        lr["ip_mass_near_1"] = float((ips > 0.9).mean())

        # --- Interaction energies (all betas, one vectorised exp call) ---
        lr["energies"] = interaction_energies_batched(G, beta_values)

        # --- Effective rank (must use raw activations, not L2-normed) ---
        lr["effective_rank"] = effective_rank_from_raw(activations)

        # --- Standard clustering (accepts pre-normed ndarray) ---
        lr["clustering"] = cluster_count_sweep(normed, thresholds)

        # --- Spectral eigengap on Gram matrix ---
        lr["spectral"] = spectral_eigengap_k(G)

        # --- PCA projection (accepts pre-normed ndarray) ---
        proj, var_ratio              = pca_projection(normed, n_components=3)
        lr["pca_explained_variance"] = var_ratio.tolist()
        results["pca_trajectories"].append(proj.tolist())

        # --- UMAP (optional, accepts pre-normed ndarray) ---
        if HAS_UMAP and umap_dir is not None and normed.shape[0] >= 4:
            umap_proj = umap_projection(normed, n_components=2)
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
