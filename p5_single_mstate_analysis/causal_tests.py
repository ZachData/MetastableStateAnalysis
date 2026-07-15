"""
causal_tests.py — Group F: causal interventions.

Intervenes on a live forward pass and measures the effect on:
  - Cluster cohesion (mass-near-1 within cluster)
  - Cluster membership (token-to-cluster assignment at later layers)
  - Merge timing (layer at which the primary fuses with the sibling)

Interventions implemented:
  (1) Head ablation         : zero one or more attention heads at a target layer
  (2) Steering — centroid   : add + α * centroid to cluster-token residuals
  (3) Steering — LDA        : add + α * lda_direction to cluster-token residuals
  (4) Activation patching   : overwrite a cluster member's residual with sibling's
  (5) Feature ablation      : zero top identity features in crosscoder reconstruction

The forward-pass mechanism is model-specific. For ALBERT we use the extended
iterations helper (extract_albert_extended) and insert a hook at the target
layer. For per-layer models we route through the standard forward with an
nn.Module hook.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Callable

# Deferred torch
_TORCH = None

def _torch():
    global _TORCH
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH


# ---------------------------------------------------------------------------
# Shared measurement utilities
# ---------------------------------------------------------------------------

def _mass_near_1(X: np.ndarray, mask: np.ndarray, thresh: float = 0.95) -> float:
    """Fraction of pairs (i, j) in the cluster with <x_i, x_j> > thresh."""
    if mask.sum() < 2:
        return 0.0
    Xc = X[mask]
    G = Xc @ Xc.T
    iu = np.triu_indices(Xc.shape[0], k=1)
    if iu[0].size == 0:
        return 0.0
    return float((G[iu] > thresh).mean())


def measure_cluster_state(
    activations: np.ndarray,
    hdb_labels: list,
    chain: list,
) -> dict:
    """Cohesion + size across the trajectory's lifespan."""
    out = []
    for layer, cid in chain:
        if layer >= activations.shape[0]:
            continue
        mask = hdb_labels[layer] == cid
        out.append({
            "layer":        int(layer),
            "size":         int(mask.sum()),
            "mass_near_1":  round(_mass_near_1(activations[layer], mask), 4),
        })
    return out


# ---------------------------------------------------------------------------
# Hook infrastructure (ALBERT single-layer path)
# ---------------------------------------------------------------------------

def _run_albert_with_hook(
    model, tokenizer, prompt_text: str,
    max_iterations: int,
    hook_fn: Callable,
    ):
    """
    Run ALBERT's shared layer or GPT-2's block layers, giving hook_fn the chance
    to modify the attention tensor or hidden state at every iteration.

    hook_fn(step, hidden, layer) -> optionally-modified hidden

    `layer` is the GPT2Block or AlbertLayer module being about to run at
    this step; both branches now call hook_fn before the layer executes, so
    interventions that only touch `hidden` (steering, patching) work
    unmodified on GPT-2, and interventions that need the layer's attention
    submodule (head ablation) can dispatch on `layer`'s type.
    """
    torch = _torch()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt",
                       truncation=True, max_length=512).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    traj, attns = [], []
    model_cls = type(model).__name__
    _is_gpt2  = "GPT2" in model_cls

    with torch.no_grad():
        if _is_gpt2:
            transformer = model.transformer
            ids  = inputs["input_ids"]
            pos  = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            hidden = transformer.wte(ids) + transformer.wpe(pos)
            hidden = transformer.drop(hidden)
            traj.append(hidden[0].float().cpu().numpy())

            for step, block in enumerate(transformer.h):
                hidden = hook_fn(step, hidden, block)
                out = block(hidden, attention_mask=None, use_cache=False,
                            output_attentions=True)
                hidden = out[0]
                traj.append(hidden[0].float().cpu().numpy())
                
                # Check if attention weights exist
                if len(out) > 1 and out[1] is not None:
                    attns.append(out[1][0].float().cpu().numpy())
            
            hidden = transformer.ln_f(hidden)

        else:
            # ALBERT path
            base           = getattr(model, "albert", model)
            embeddings_mod = base.embeddings
            encoder_mod    = base.encoder
            
            embed = embeddings_mod(
                input_ids=inputs["input_ids"],
                token_type_ids=inputs.get("token_type_ids"),
            )
            hidden = encoder_mod.embedding_hidden_mapping_in(embed)
            traj.append(hidden[0].float().cpu().numpy())

            albert_layer = encoder_mod.albert_layer_groups[0].albert_layers[0]
            extended_mask = model.get_extended_attention_mask(
                inputs["attention_mask"], inputs["input_ids"].shape,
            )

            for step in range(max_iterations):
                hidden = hook_fn(step, hidden, albert_layer)
                out = albert_layer(hidden, attention_mask=extended_mask,
                                   output_attentions=True)
                hidden = out[0]
                traj.append(hidden[0].float().cpu().numpy())
                if len(out) > 1 and out[1] is not None:
                    attns.append(out[1][0].float().cpu().numpy())

    return np.stack(traj), np.stack(attns) if attns else None, tokens

# ---------------------------------------------------------------------------
# Intervention implementations (ALBERT; per-layer models TODO)
# ---------------------------------------------------------------------------

def _attn_submodule(layer):
    """Attention submodule for either a GPT2Block or an AlbertLayer."""
    if hasattr(layer, "attn"):
        return layer.attn          # GPT2Block
    return layer.attention          # AlbertLayer


def _attn_output_projection(attn_module):
    """
    (projection_module, head_dim) for the linear/conv layer that maps the
    concatenated per-head context back to hidden_size.

    Both ALBERT's `attention.dense` and GPT-2's `attn.c_proj` consume a
    (..., n_heads * head_dim) tensor in head-major order, so zeroing the
    columns belonging to `head_ids` immediately before this projection is
    architecture-agnostic and head-specific: it removes exactly that head's
    contribution to the layer's attention output, rather than scaling the
    whole output by a head-count ratio.
    """
    if hasattr(attn_module, "c_proj"):
        return attn_module.c_proj, attn_module.head_dim          # GPT2Attention
    return attn_module.dense, attn_module.attention_head_size    # AlbertAttention


def ablate_head(
    model, tokenizer, prompt_text: str, max_iterations: int,
    target_layer: int, head_ids: list,
):
    """
    Run the forward pass with head(s) zeroed at the target layer only.

    Implemented as a forward pre-hook on the attention's output projection,
    installed only while the hook_fn step equals target_layer. The hook
    zeroes the head_ids' column slices of the projection's input — the
    concatenated per-head context — which is equivalent to zeroing those
    heads before W_O / c_proj is applied. This distinguishes which head was
    targeted, unlike the previous whole-output scalar shrink which was
    identical for any single head.

    Returns (trajectory, attentions, tokens) where trajectory has same shape
    as the unmodified extended run.
    """
    state = {"handle": None}

    def _make_zero_hook(ids, head_dim):
        def _hook(module, inputs):
            (x,) = inputs
            x = x.clone()
            for h in ids:
                x[..., h * head_dim:(h + 1) * head_dim] = 0.0
            return (x,)
        return _hook

    def hook_fn(step, hidden, layer):
        if step == target_layer:
            if state["handle"] is None:
                attn_module = _attn_submodule(layer)
                proj, head_dim = _attn_output_projection(attn_module)
                state["handle"] = proj.register_forward_pre_hook(
                    _make_zero_hook(head_ids, head_dim)
                )
        elif state["handle"] is not None:
            state["handle"].remove()
            state["handle"] = None
        return hidden

    traj, attns, tokens = _run_albert_with_hook(
        model, tokenizer, prompt_text, max_iterations, hook_fn,
    )
    if state["handle"] is not None:
        state["handle"].remove()
        state["handle"] = None
    return traj, attns, tokens


def steer_residual(
    model, tokenizer, prompt_text: str, max_iterations: int,
    target_layer: int, token_indices: list, direction: np.ndarray,
    alpha: float,
):
    """
    Add + alpha * direction to the residual stream of specified tokens at
    target_layer (done BEFORE the layer computes its update).
    """
    torch = _torch()
    device = next(model.parameters()).device
    dir_t = torch.from_numpy(direction.astype(np.float32)).to(device)

    def hook_fn(step, hidden, albert_layer):
        if step == target_layer:
            hidden = hidden.clone()
            for t in token_indices:
                hidden[0, t] = hidden[0, t] + alpha * dir_t
        return hidden

    return _run_albert_with_hook(
        model, tokenizer, prompt_text, max_iterations, hook_fn,
    )


def patch_activation(
    model, tokenizer, prompt_text: str, max_iterations: int,
    target_layer: int, token_idx: int, replacement_vector: np.ndarray,
):
    """
    Replace token_idx's residual vector with replacement_vector at target_layer.
    """
    torch = _torch()
    device = next(model.parameters()).device
    rep_t = torch.from_numpy(replacement_vector.astype(np.float32)).to(device)

    def hook_fn(step, hidden, albert_layer):
        if step == target_layer:
            hidden = hidden.clone()
            hidden[0, token_idx] = rep_t
        return hidden

    return _run_albert_with_hook(
        model, tokenizer, prompt_text, max_iterations, hook_fn,
    )


# ---------------------------------------------------------------------------
# Re-cluster after intervention and compare to baseline
# ---------------------------------------------------------------------------

def recluster_after_intervention(
    modified_activations: np.ndarray,   # (n_layers, n, d)
    baseline_hdb_labels: list,          # for alignment of cluster IDs
    trajectory_chain: list,
    ) -> dict:
    """
    After intervention, re-run HDBSCAN at each layer of the trajectory's
    lifespan and measure:
      - how many baseline cluster members remain together
      - whether the cluster persists beyond its original merge layer
    """
    try:
        import hdbscan
    except ImportError:
        return {"error": "hdbscan not installed"}

    baseline_chain = dict(trajectory_chain)   # {layer: cluster_id}

    per_layer = []
    for layer, cid in trajectory_chain:
        if layer >= modified_activations.shape[0]:
            continue

        # L2-norm activations for HDBSCAN (mirrors Phase 1 preprocessing)
        X = modified_activations[layer]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X / np.maximum(norms, 1e-12)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(3, int(0.05 * Xn.shape[0])),
            metric="euclidean",
        )
        new_labels = clusterer.fit_predict(Xn)

        # Which baseline-cluster members are now in the same new cluster?
        baseline_mask = baseline_hdb_labels[layer] == cid
        if not baseline_mask.any():
            continue
        baseline_members = np.where(baseline_mask)[0]
        new_assignments = new_labels[baseline_members]
        # Dominant new cluster for these tokens
        # Noise (-1) counted separately
        valid = new_assignments[new_assignments >= 0]
        if valid.size == 0:
            dominant, frac_same = -1, 0.0
        else:
            vals, counts = np.unique(valid, return_counts=True)
            dominant = int(vals[np.argmax(counts)])
            frac_same = float(counts.max() / baseline_members.size)

        per_layer.append({
            "layer":       int(layer),
            "baseline_size": int(baseline_members.size),
            "frac_together": round(frac_same, 4),
            "dominant_new_label": dominant,
            "fraction_noise": round(float((new_assignments == -1).mean()), 4),
        })
    return {"per_layer": per_layer}


# ---------------------------------------------------------------------------
# Public entry point — runs all interventions and aggregates
# ---------------------------------------------------------------------------

def _get_input_embeds(model, input_ids, token_type_ids=None):
    """
    Model-agnostic token embedding lookup.

    BERT/ALBERT expose _get_input_embeds(model, input_ids, token_type_ids).
    GPT-2 / causal LMs expose model.transformer.wte(input_ids).
    All HuggingFace models expose model.get_input_embeddings() as a fallback.
    """
    if hasattr(model, "embeddings"):
        # BERT / ALBERT path
        return _get_input_embeds(model,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        # GPT-2 path
        return model.transformer.wte(input_ids)
    # Generic HuggingFace fallback (works for any architecture)
    return model.get_input_embeddings()(input_ids)


def run_causal_tests(
    model,
    tokenizer,
    prompt_text: str,
    max_iterations: int,
    baseline_activations: np.ndarray,
    baseline_hdb_labels: list,
    trajectory: dict,
    sibling_trajectory: dict,
    top_heads: list,          # from Group C.1
    lda_direction: np.ndarray,
    centroid_direction: np.ndarray,
    interventions_to_run: list = None,
    steering_alpha: float = 2.0,
) -> dict:
    """
    Execute requested interventions, re-cluster, and compare to baseline.

    interventions_to_run : subset of {"ablate_top_head", "ablate_control_head",
                                       "steer_centroid", "steer_lda",
                                       "patch_sibling"}
    """
    if interventions_to_run is None:
        interventions_to_run = [
            "ablate_top_head", "ablate_control_head",
            "steer_centroid", "steer_lda", "patch_sibling",
        ]

    chain = trajectory["chain"]
    chain_tuples = [(int(l), int(c)) for l, c in chain]
    merge_ev = trajectory.get("merge_event")
    target_layer = (
        merge_ev["layer_from"] if merge_ev is not None
        else chain_tuples[len(chain_tuples) // 2][0]
    )
    mid_layer = chain_tuples[len(chain_tuples) // 2][0]

    # Cluster-member token indices at target layer
    member_idxs = np.where(
        baseline_hdb_labels[target_layer] ==
        dict(chain_tuples).get(target_layer, -999)
    )[0].tolist()

    results = {"target_layer": int(target_layer), "interventions": {}}

    # ---- Ablation: top attractor head ----
    if "ablate_top_head" in interventions_to_run and top_heads:
        top_h = top_heads[0]["head"]
        traj, _, _ = ablate_head(
            model, tokenizer, prompt_text, max_iterations,
            target_layer=target_layer, head_ids=[top_h],
        )
        results["interventions"]["ablate_top_head"] = {
            "head":   int(top_h),
            "target_layer": int(target_layer),
            "recluster": recluster_after_intervention(
                traj, baseline_hdb_labels, chain_tuples,
            ),
        }

    # ---- Ablation: control head (middle of cohesion ranking) ----
    if "ablate_control_head" in interventions_to_run and len(top_heads) > 2:
        ctrl_h = top_heads[len(top_heads) // 2]["head"]
        traj, _, _ = ablate_head(
            model, tokenizer, prompt_text, max_iterations,
            target_layer=target_layer, head_ids=[ctrl_h],
        )
        results["interventions"]["ablate_control_head"] = {
            "head":   int(ctrl_h),
            "target_layer": int(target_layer),
            "recluster": recluster_after_intervention(
                traj, baseline_hdb_labels, chain_tuples,
            ),
        }

    # ---- Steering: + centroid ----
    if "steer_centroid" in interventions_to_run and centroid_direction is not None:
        traj, _, _ = steer_residual(
            model, tokenizer, prompt_text, max_iterations,
            target_layer=mid_layer, token_indices=member_idxs,
            direction=centroid_direction, alpha=steering_alpha,
        )
        results["interventions"]["steer_centroid"] = {
            "target_layer": int(mid_layer),
            "alpha": steering_alpha,
            "recluster": recluster_after_intervention(
                traj, baseline_hdb_labels, chain_tuples,
            ),
        }

    # ---- Steering: + LDA ----
    if "steer_lda" in interventions_to_run and lda_direction is not None \
            and float(np.linalg.norm(lda_direction)) > 1e-6:
        traj, _, _ = steer_residual(
            model, tokenizer, prompt_text, max_iterations,
            target_layer=mid_layer, token_indices=member_idxs,
            direction=lda_direction, alpha=steering_alpha,
        )
        results["interventions"]["steer_lda"] = {
            "target_layer": int(mid_layer),
            "alpha": steering_alpha,
            "recluster": recluster_after_intervention(
                traj, baseline_hdb_labels, chain_tuples,
            ),
        }

    # ---- Activation patching from sibling ----
    if "patch_sibling" in interventions_to_run and sibling_trajectory is not None:
        sib_chain = dict(sibling_trajectory["chain"])
        sib_cid = sib_chain.get(mid_layer)
        if sib_cid is not None:
            sib_mask = baseline_hdb_labels[mid_layer] == sib_cid
            if sib_mask.any():
                sib_centroid = baseline_activations[mid_layer][sib_mask].mean(axis=0)
                # Patch the first cluster member's residual
                target_token = member_idxs[0] if member_idxs else 0
                traj, _, _ = patch_activation(
                    model, tokenizer, prompt_text, max_iterations,
                    target_layer=mid_layer, token_idx=target_token,
                    replacement_vector=sib_centroid,
                )
                results["interventions"]["patch_sibling"] = {
                    "target_layer": int(mid_layer),
                    "patched_token_idx": int(target_token),
                    "recluster": recluster_after_intervention(
                        traj, baseline_hdb_labels, chain_tuples,
                    ),
                }

    # Baseline for reference
    results["baseline_state"] = measure_cluster_state(
        baseline_activations, baseline_hdb_labels, chain_tuples,
    )

    return results


def save_causal(result: dict, out_dir: Path) -> None:
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "group_F_causal.json", "w") as f:
        json.dump(result, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")
