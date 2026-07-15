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

Fix 1  — emb_gram pre-computed from hidden_states[0] (block-0 output after
          Fix 4) as an external semantic signal for pair_hdbscan_agreement.
Fix 2  — Energy drop localization uses a relative threshold
          diff(E)/|E| < -ENERGY_VIOLATION_REL_TOL instead of absolute -1e-6.
Fix 3  — is_causal flag threads through to analyze_attention_sinkhorn so
          causal-mask baseline subtraction is applied for GPT-2 family.
Fix 4  — extraction_meta kwarg stores lm_head_excluded, n_layers_total,
          n_layers_analyzed; GPT-2 final layer excluded upstream.
Fix 8  — DEGENERATE_RANK_THRESHOLD (=2) replaces all hardcoded 3.0/2.0 gates;
          effective_rank_normed stored per layer for sphere-spread reporting.

Bug fix (prev_normed order)
---------------------------
prev_normed was previously updated inside the CKA block, BEFORE the
energy-drop computation.  This caused energy_drop_pairs_from_normed to
receive (current, current) instead of (previous, current).  The update
is now deferred to after the energy-drop block.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

from core.config import BETA_VALUES, DISTANCE_THRESHOLDS, DEGENERATE_RANK_THRESHOLD
from core.models import layernorm_to_sphere
from .metrics import (
    pairwise_inner_products_from_gram,
    interaction_energies_batched,
    effective_rank_from_raw,
    effective_rank_from_normed,
    attention_entropy,
    nearest_neighbor_indices,
    linear_cka,
    energy_drop_pairs_from_normed,
    energy_violation_severity,
    ENERGY_VIOLATION_REL_TOL,
)
from .sinkhorn import analyze_attention_sinkhorn
from .spectral import spectral_eigengap_k
from .clustering import (
    cluster_count_sweep, pca_projection, umap_projection, HAS_UMAP,
    multiscale_nesting, pair_hdbscan_agreement,
)
from .cluster_tracking import track_clusters


# ---------------------------------------------------------------------------
# Fix 3 — causal-model detection
# ---------------------------------------------------------------------------

_CAUSAL_MODEL_PREFIXES = ("gpt2", "pythia", "gpt-neox", "gptneox")   # extend for other decoder models


def _is_causal_model(model_name: str) -> bool:
    """Return True for decoder-only (causally-masked) models."""
    return any(model_name.lower().startswith(p) for p in _CAUSAL_MODEL_PREFIXES)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_trajectory(
    hidden_states: list,
    attentions: list,
    prompt_key: str,
    model_name: str,
    tokens: list,
    beta_values: list = BETA_VALUES,
    thresholds: np.ndarray = DISTANCE_THRESHOLDS,
    umap_dir: Path = None,
    apply_causal_control: bool = False,   # Fix 3b: mask control for BERT
    extraction_meta: dict = None,          # Fix 4: from extract_activations
) -> dict:
    """
    Compute all per-layer metrics for one (model, prompt) run.

    Parameters
    ----------
    hidden_states        : list of (n_tokens, d_model) float32 Tensors
    attentions           : list of (n_heads, n_tokens, n_tokens) float32 Tensors
    prompt_key           : string key from PROMPTS
    model_name           : model identifier string
    tokens               : list of decoded token strings
    beta_values          : β values for interaction energy
    thresholds           : cosine-distance thresholds for agglomerative sweep
    umap_dir             : if provided, UMAP projections saved here as .npy
    apply_causal_control : Fix 3b — run causal-mask control on BERT to
                           test whether masking alone produces low Fiedler
    extraction_meta      : Fix 4 — dict with lm_head_excluded, n_layers_total,
                           n_layers_analyzed.  None → safe defaults.

    Returns
    -------
    results dict consumed by plots.py, reporting.py, and io_utils.py
    """
    # Fix 4 — safe defaults for legacy callers
    if extraction_meta is None:
        extraction_meta = {
            "lm_head_excluded":  False,
            "n_layers_total":    len(hidden_states),
            "n_layers_analyzed": len(hidden_states),
        }

    n_layers = len(hidden_states)
    results  = {
        "model":             model_name,
        "prompt":            prompt_key,
        "tokens":            tokens,
        "n_layers":          n_layers,
        "n_layers_total":    extraction_meta["n_layers_total"],
        "n_layers_analyzed": extraction_meta["n_layers_analyzed"],
        "lm_head_excluded":  extraction_meta["lm_head_excluded"],
        "n_tokens":          hidden_states[0].shape[0],
        "d_model":           hidden_states[0].shape[1],
        "layers":            [],
        "pca_trajectories":  [],
    }

    # Fix 3 — constant across layers for this model/run
    is_causal = _is_causal_model(model_name)

    # Rolling state — prev_normed updated AFTER energy-drop block (bug fix)
    prev_nn:       np.ndarray | None = None
    prev_normed:   np.ndarray | None = None
    prev_energies: dict | None       = None

    # ------------------------------------------------------------------
    # Fix 1 — pre-compute input-embedding Gram matrix.
    # After Fix 4, hidden_states[0] is the first block output (not the raw
    # embedding), since extract_activations strips the pre-transformer
    # embedding at index 0.  Still independent of the HDBSCAN pipeline.
    # ------------------------------------------------------------------
    _hs0_normed = layernorm_to_sphere(hidden_states[0]).numpy()
    emb_gram    = (_hs0_normed @ _hs0_normed.T).astype(np.float32)

    for layer_idx, activations in enumerate(tqdm(
        hidden_states,
        desc=f"{model_name[:20]} | {prompt_key}",
        leave=False,
    )):
        lr = {"layer": layer_idx}

        # Pre-compute normed activations and Gram matrix once per layer
        normed = layernorm_to_sphere(activations).numpy()   # (n_tokens, d)
        G      = normed @ normed.T                          # (n_tokens, n_tokens)

        # --- Fix 8: both rank variants ---
        # Raw: captures scale + directional collapse; used for all gates.
        # Normed: measures directional spread on the sphere only; for reporting.
        lr["effective_rank"]        = effective_rank_from_raw(activations)
        lr["effective_rank_normed"] = effective_rank_from_normed(normed)

        # --- CKA vs previous layer ---
        # Fix 8: gate uses DEGENERATE_RANK_THRESHOLD (=2) unified constant.
        # Note: prev_normed is NOT updated here — deferred until after energy drops.
        if prev_normed is not None and lr["effective_rank"] >= DEGENERATE_RANK_THRESHOLD:
            lr["cka_prev"] = linear_cka(normed, prev_normed)
        else:
            lr["cka_prev"] = float("nan")

        # --- Pairwise inner products + mass-near-1 ---
        ips                  = pairwise_inner_products_from_gram(G)
        lr["ip_mean"]        = float(ips.mean())
        lr["ip_std"]         = float(ips.std())
        lr["ip_histogram"]   = np.histogram(ips, bins=50, range=(-1, 1))[0].tolist()
        lr["ip_mass_near_1"] = float((ips > 0.9).mean())

        # --- Nearest-neighbour trajectory tracking ---
        nn               = nearest_neighbor_indices(G)
        lr["nn_indices"] = nn.tolist()
        if prev_nn is not None:
            lr["nn_stability"] = float(np.mean(nn == prev_nn))
        else:
            lr["nn_stability"] = None
        prev_nn = nn

        # --- Interaction energies ---
        lr["energies"] = interaction_energies_batched(G, beta_values)

        # --- Energy drop localization ---
        # Fix 2 + Fix 8: relative threshold + unified gate.
        # prev_normed still points to the PREVIOUS layer here (bug fix: the
        # update is deferred to after this block).
        if prev_energies is not None and lr["effective_rank"] >= DEGENERATE_RANK_THRESHOLD:
            drops = {}
            for beta in beta_values:
                e_prev = prev_energies.get(beta, float("nan"))
                e_curr = lr["energies"].get(beta, float("nan"))
                if np.isnan(e_prev) or np.isnan(e_curr):
                    drops[beta] = []
                    continue
                ref      = max(abs(e_prev), 1e-12)
                rel_drop = -(e_curr - e_prev) / ref
                if rel_drop > ENERGY_VIOLATION_REL_TOL:
                    drops[beta] = energy_drop_pairs_from_normed(
                        prev_normed, normed, beta
                    )
                else:
                    drops[beta] = []
            lr["energy_drop_pairs"] = drops
        else:
            lr["energy_drop_pairs"] = {beta: [] for beta in beta_values}

        prev_energies = lr["energies"]
        prev_normed   = normed   # deferred update — AFTER energy-drop block

        # --- Clustering (agglomerative + KMeans + HDBSCAN) ---
        lr["clustering"] = cluster_count_sweep(normed, thresholds)

        # --- Spectral eigengap + Fiedler vector ---
        spectral_result = spectral_eigengap_k(G, return_fiedler_vec=True)
        lr["spectral"]  = spectral_result
        fvec = spectral_result.get("fiedler_vec")
        if fvec is not None:
            lr["fiedler_bipartition"] = [int(np.sign(v)) if v != 0.0 else 1 for v in fvec]
        else:
            lr["fiedler_bipartition"] = None

        # --- Multi-scale cluster nesting (P1-3) ---
        hdb_data = lr["clustering"].get("hdbscan", {})
        if "labels" in hdb_data and normed.shape[0] >= 4:
            hdb_labels      = np.array(hdb_data["labels"], dtype=np.int32)
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

        # --- Per-pair agreement / induction-head filtering (P1-4) ---
        # Fix 1: emb_gram provides the external semantic axis.
        if "labels" in hdb_data:
            hdb_labels = np.array(hdb_data["labels"], dtype=np.int32)
            lr["pair_agreement"] = pair_hdbscan_agreement(
                nn, hdb_labels, tokens,
                emb_gram=emb_gram,
            )
        else:
            lr["pair_agreement"] = {
                "mutual_pairs":              [],
                "n_same_cluster":            0,
                "n_diff_cluster":            0,
                "n_noise":                   0,
                "artifact_fraction":         0.0,
                "n_ext_semantic":            0,
                "n_ext_non_semantic":        0,
                "n_ext_unknown":             0,
                "ext_semantic_fraction":     None,
                "ext_sem_same_cluster_frac": None,
                "n_semantic":                0,
                "n_artifact":                0,
            }

        # --- PCA projection ---
        proj, var_ratio              = pca_projection(normed, n_components=3)
        lr["pca_explained_variance"] = var_ratio.tolist()
        results["pca_trajectories"].append(proj.tolist())

        # --- UMAP (optional) ---
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

        # --- Attention: entropy + Sinkhorn (Fix 3) ---
        if layer_idx < len(attentions):
            attn                             = attentions[layer_idx]
            ent                              = attention_entropy(attn)
            lr["attention_entropy_per_head"] = ent.tolist()
            lr["attention_entropy_mean"]     = float(ent.mean())
            lr["sinkhorn"]                   = analyze_attention_sinkhorn(
                                                   attn,
                                                   is_causal=is_causal,
                                                   apply_causal_control=apply_causal_control,
                                               )

        results["layers"].append(lr)

    # ------------------------------------------------------------------
    # Post-loop: HDBSCAN cluster tracking (P1-1)
    # ------------------------------------------------------------------
    results["cluster_tracking"] = track_clusters(results)

    # ------------------------------------------------------------------
    # Post-loop: identify plateau layers for attention saving (P1-7)
    # Uses the multi-signal joint criterion: a layer is included when
    # it falls inside the plateau window of 2+ independent signals.
    # This matches what generate_llm_report prints, so results on disk
    # and the text report are now consistent.
    # ------------------------------------------------------------------
    from .reporting import compute_plateau_layers
    results["plateau_layers"] = compute_plateau_layers(results)

    return results