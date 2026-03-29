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

from core.config import BETA_VALUES, DISTANCE_THRESHOLDS
from core.models import layernorm_to_sphere
from .metrics import (
    pairwise_inner_products_from_gram,
    interaction_energies_batched,
    effective_rank_from_raw,
    attention_entropy,
    nearest_neighbor_indices,
    linear_cka,
    energy_drop_pairs,
)
from .sinkhorn import analyze_attention_sinkhorn
from .spectral import spectral_eigengap_k
from .clustering import (
    cluster_count_sweep, pca_projection, umap_projection, HAS_UMAP,
    multiscale_nesting, pair_hdbscan_agreement,
)
from .cluster_tracking import track_clusters


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

    prev_nn: np.ndarray = None   # NN index array from previous layer
    prev_normed: np.ndarray = None  # L2-normed activations from previous layer
    prev_activations = None  # raw activations from previous layer (for energy drop pairs)

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

        # --- Effective rank (must use raw activations, not L2-normed) ---
        # Computed early so the CKA block can use it for degeneracy gating.
        lr["effective_rank"] = effective_rank_from_raw(activations)

        # --- CKA vs previous layer ---
        # Suppress when effective_rank < 3: all tokens are a near-point-mass,
        # centering produces noise-dominated vectors, and the Frobenius norms
        # collapse to near-zero — the ratio is numerically meaningless.
        if prev_normed is not None and lr["effective_rank"] >= 3.0:
            lr["cka_prev"] = linear_cka(normed, prev_normed)
        else:
            lr["cka_prev"] = float("nan")
        prev_normed = normed

        # --- Inner products ---
        ips                  = pairwise_inner_products_from_gram(G)
        lr["ip_mean"]        = float(ips.mean())
        lr["ip_std"]         = float(ips.std())
        lr["ip_histogram"]   = np.histogram(ips, bins=50, range=(-1, 1))[0].tolist()
        lr["ip_mass_near_1"] = float((ips > 0.9).mean())

        # --- Nearest-neighbour trajectory tracking ---
        # nn[i] = index of token i's nearest neighbour at this layer (excl. self).
        # nn_stability = fraction of tokens with unchanged NN vs the previous layer.
        # Layer 0 has no predecessor, so stability is undefined (stored as None).
        nn                   = nearest_neighbor_indices(G)          # (n_tokens,)
        lr["nn_indices"]     = nn.tolist()
        if prev_nn is not None:
            lr["nn_stability"] = float(np.mean(nn == prev_nn))
        else:
            lr["nn_stability"] = None
        prev_nn = nn

        # --- Interaction energies (all betas, one vectorised exp call) ---
        lr["energies"] = interaction_energies_batched(G, beta_values)

        # --- Energy drop localization (all betas, violation layers only) ---
        # A violation is when E_beta decreases from the previous layer.
        # Gate on effective_rank >= 3 to suppress degenerate-regime noise.
        # Output format: {beta: [(i, j, delta), ...]} — empty list per beta
        # when no violation or not enough context.
        if prev_activations is not None and lr["effective_rank"] >= 3.0:
            drop_pairs_by_beta = {}
            for beta in beta_values:
                e_curr = lr["energies"].get(beta, float("nan"))
                e_prev = (
                    results["layers"][-1]["energies"].get(beta, float("nan"))
                    if results["layers"] else float("nan")
                )
                is_nan = lambda v: isinstance(v, float) and v != v
                if not is_nan(e_curr) and not is_nan(e_prev) and e_curr - e_prev < -1e-6:
                    drop_pairs_by_beta[beta] = energy_drop_pairs(
                        prev_activations, activations, beta=beta, top_k=10
                    )
                else:
                    drop_pairs_by_beta[beta] = []
            lr["energy_drop_pairs"] = drop_pairs_by_beta
        else:
            lr["energy_drop_pairs"] = {beta: [] for beta in beta_values}
        prev_activations = activations

        # --- Standard clustering (accepts pre-normed ndarray) ---
        lr["clustering"] = cluster_count_sweep(normed, thresholds)

        # --- KMeans centroids (Phase 5 cluster identity analysis) ---
        # Computed here where normed is available; stored as a nested list
        # so they survive JSON serialization in metrics.json and are also
        # persisted to clusters.npz by save_run.
        # Centroid[c] = mean of all normed token vectors assigned to cluster c,
        # re-normalized to S^{d-1} so inner products remain meaningful.
        km_labels = np.array(lr["clustering"]["kmeans"]["labels"], dtype=np.int32)
        best_k    = lr["clustering"]["kmeans"]["best_k"]
        centroids = np.zeros((best_k, normed.shape[1]), dtype=np.float32)
        for c in range(best_k):
            mask = km_labels == c
            if mask.any():
                mean_vec = normed[mask].mean(axis=0)
                norm     = np.linalg.norm(mean_vec)
                centroids[c] = mean_vec / norm if norm > 1e-10 else mean_vec
        lr["cluster_centroids_kmeans"] = centroids.tolist()

        # --- Spectral eigengap on Gram matrix ---
        lr["spectral"] = spectral_eigengap_k(G)

        # --- Multi-scale cluster nesting (P1-3) ---
        # Run spectral eigengap within each HDBSCAN cluster to detect
        # hierarchical organization (global bipartition nesting inside
        # local density structure).
        hdb_data = lr["clustering"].get("hdbscan", {})
        if "labels" in hdb_data and normed.shape[0] >= 4:
            hdb_labels = np.array(hdb_data["labels"], dtype=np.int32)
            n_real_clusters = len(set(hdb_labels) - {-1})
            if n_real_clusters >= 2:
                lr["nesting"] = multiscale_nesting(normed, hdb_labels)
            else:
                lr["nesting"] = {
                    "global_spectral_k": lr["spectral"]["k_eigengap"],
                    "per_cluster": {},
                    "has_nesting": False,
                    "nesting_summary": "fewer than 2 HDBSCAN clusters",
                    "n_clusters_with_substructure": 0,
                }
        else:
            lr["nesting"] = {
                "global_spectral_k": lr["spectral"]["k_eigengap"],
                "per_cluster": {},
                "has_nesting": False,
                "nesting_summary": "HDBSCAN not available",
                "n_clusters_with_substructure": 0,
            }

        # --- Per-pair HDBSCAN agreement / induction head filtering (P1-4) ---
        if "labels" in hdb_data:
            lr["pair_agreement"] = pair_hdbscan_agreement(
                nn, hdb_labels, tokens,
            )
        else:
            lr["pair_agreement"] = {
                "mutual_pairs": [],
                "n_semantic": 0,
                "n_artifact": 0,
                "n_noise": 0,
                "artifact_fraction": 0.0,
            }

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

    # ------------------------------------------------------------------
    # Post-loop: HDBSCAN cluster tracking across layers (P1-1)
    # ------------------------------------------------------------------
    results["cluster_tracking"] = track_clusters(results)

    # ------------------------------------------------------------------
    # Post-loop: identify plateau layers for attention saving (P1-7)
    # Plateau detection uses mass-near-1 with the same parameters as
    # reporting.detect_plateaus.  Layers in any plateau window are
    # flagged so that save_run can persist their raw attention matrices.
    # ------------------------------------------------------------------
    from .reporting import detect_plateaus
    mass1 = [r["ip_mass_near_1"] for r in results["layers"]]
    plateaus = detect_plateaus(mass1, window=2, tol=0.10)
    plateau_layer_set = set()
    for s, e, _ in plateaus:
        for l in range(s, e + 1):
            plateau_layer_set.add(l)
    results["plateau_layers"] = sorted(plateau_layer_set)

    return results
