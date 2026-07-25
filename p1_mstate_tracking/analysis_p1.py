"""
p1_mstate_tracking/analysis_p1.py — Layer-wise analysis loop.

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

Fix 1  — emb_gram pre-computed from hidden_states[0] as an external semantic
          signal for pair_hdbscan_agreement.
Fix 2  — Energy drop localization uses a relative threshold
          diff(E)/|E| < -ENERGY_VIOLATION_REL_TOL instead of absolute -1e-6.
Fix 3  — is_causal flag threads through to analyze_attention_sinkhorn so
          causal-mask baseline subtraction is applied for decoder-only models.
          Detection now lives in core/model_family.py, which is also what
          plots.py dispatches on — the two used to disagree.
Fix 4  — extraction_meta is supplied by the caller (run_1.py, via
          core.models.describe_extraction). It was previously a parameter no
          caller ever passed, so the None-default branch ran on every run and
          the recorded metadata was a constant.
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
from core.model_family import is_causal_model, model_family
from core.metrics import (
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
    fiedler_and_eigengap as spectral_eigengap_k,
)
from .sinkhorn import analyze_attention_sinkhorn
from .clustering import (
    cluster_count_sweep, pca_projection, umap_projection, HAS_UMAP,
    multiscale_nesting, pair_hdbscan_agreement,
)
from .cluster_tracking import track_clusters


# ---------------------------------------------------------------------------
# Fix 3 — causal-model detection
# ---------------------------------------------------------------------------
# Kept as a module-level name for callers that imported it, but it is now a
# thin alias. The implementation moved to core/model_family.py because this
# module and plots.py each had their own version and they disagreed on any
# name that didn't start with the architecture (e.g. the smoke checkpoint
# "hf-internal-testing/tiny-random-GPTNeoXForCausalLM").
_is_causal_model = is_causal_model


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def _default_extraction_meta(hidden_states, attentions, model_name) -> dict:
    """Fallback meta for callers that don't have a model object to hand.

    _run_sublayer_analysis is the real one: it builds its hidden states from
    forward hooks, not from extract_activations, so there is no
    describe_extraction call to make. Everything it can honestly assert is
    asserted; everything it can't is None rather than a plausible-looking
    default.
    """
    return {
        "lm_head_excluded":              False,
        "n_layers_total":                len(hidden_states),
        "n_layers_analyzed":             len(hidden_states),
        "n_attention_layers":            len(attentions),
        "hidden_state_0_is_embedding":   None,
        "final_hidden_state_is_post_ln": None,
        "model_family":                  model_family(model_name),
        "weight_dtype":                  None,
        "autocast":                      None,
        "device":                        None,
    }


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
    extraction_meta: dict = None,          # Fix 4: from core.models.describe_extraction
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
    extraction_meta      : Fix 4 — from core.models.describe_extraction.
                           None → a fallback with the unknowable fields set
                           to None instead of to a guess.

    Returns
    -------
    results dict consumed by plots.py, reporting_p1.py, and p1_io.py
    """
    if extraction_meta is None:
        extraction_meta = _default_extraction_meta(
            hidden_states, attentions, model_name
        )

    n_layers = len(hidden_states)
    results  = {
        "model":             model_name,
        "prompt":            prompt_key,
        "tokens":            tokens,
        "n_layers":          n_layers,
        "n_tokens":          hidden_states[0].shape[0],
        "d_model":           hidden_states[0].shape[1],
        "layers":            [],
        "pca_trajectories":  [],
        # Everything below comes straight from extraction_meta. Recorded on
        # the results dict (and so into geometry.json) so a downstream phase
        # can tell what it is reading without re-deriving it from the model
        # name — in particular whether the last layer is post-LN and what
        # dtype the forward pass ran in.
        "n_layers_total":                extraction_meta.get("n_layers_total", n_layers),
        "n_layers_analyzed":             extraction_meta.get("n_layers_analyzed", n_layers),
        "lm_head_excluded":              extraction_meta.get("lm_head_excluded", False),
        "hidden_state_0_is_embedding":   extraction_meta.get("hidden_state_0_is_embedding"),
        "final_hidden_state_is_post_ln": extraction_meta.get("final_hidden_state_is_post_ln"),
        "model_family":                  extraction_meta.get("model_family"),
        "weight_dtype":                  extraction_meta.get("weight_dtype"),
        "autocast":                      extraction_meta.get("autocast"),
        "hf_repo":                       extraction_meta.get("hf_repo"),
        "revision":                      extraction_meta.get("revision"),
        "checkpoint_step":               extraction_meta.get("checkpoint_step"),
        "random_init":                   extraction_meta.get("random_init", False),
        "sublayer_semantics":            extraction_meta.get("sublayer_semantics"),
        "parallel_residual":             extraction_meta.get("parallel_residual"),
    }

    # Fix 3 — constant across layers for this model/run
    is_causal = is_causal_model(model_name)

    # Rolling state — prev_normed updated AFTER energy-drop block (bug fix)
    prev_nn:       np.ndarray | None = None
    prev_normed:   np.ndarray | None = None
    prev_energies: dict | None       = None

    # ------------------------------------------------------------------
    # Fix 1 — pre-compute input-embedding Gram matrix.
    # hidden_states[0] is the pre-transformer embedding output: nothing in
    # extract_activations strips it, and nothing downstream does either.
    # (A previous comment here asserted the opposite. It was describing an
    # intended Fix 4 behaviour that was never implemented — the meta dict
    # said lm_head_excluded=True nowhere, and no layer was ever dropped.)
    # Independent of the HDBSCAN pipeline either way.
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
    from .reporting_p1 import compute_plateau_layers
    results["plateau_layers"] = compute_plateau_layers(results)

    return results