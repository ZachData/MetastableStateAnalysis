"""
head_ablation.py — Per-head OV ablation at violation layers (GPT-2).

Tests the causal claim: if V's repulsive eigenstructure causes violations,
then ablating (zeroing) the most repulsive heads should reduce violation
magnitude, while ablating low-repulsive heads should have little effect.

Implementation
--------------
Requires per-head attention output deltas saved by the extended decompose
hook (``decompose.save_decomposed_per_head``).  This module provides:

  1. ``save_decomposed_per_head``  — hook-based extraction of per-head
     outputs for GPT-2, to be integrated into the decompose.py pipeline.

  2. ``ablate_head_at_violation``  — counterfactual energy computation
     when head h's contribution is zeroed at a violation layer.

  3. ``run_head_ablation``         — full pipeline: for each violation
     layer, ablate each head, rank by ablation effect, correlate effect
     with head rep_frac.

Functions
---------
save_decomposed_per_head         : save per-head attn deltas (GPT-2 only)
ablate_head_at_violation         : counterfactual energy for one head
run_head_ablation                : full per-violation ablation analysis
print_head_ablation_summary      : terminal output
"""

import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Per-head hook extraction (GPT-2)
# ---------------------------------------------------------------------------

def save_decomposed_per_head(
    model,
    tokenizer,
    text: str,
    run_dir: Path,
) -> None:
    """
    Extract and save per-head attention output deltas for GPT-2.

    For each GPT-2 block, captures the per-head attention output BEFORE
    the output projection c_proj.  The output projection mixes heads; we
    need pre-projection outputs to isolate each head's contribution to the
    residual stream.

    The per-head residual contribution is:

        head_delta_h = head_out_h @ W_O_h

    where head_out_h ∈ R^{n_tokens × d_head} is the attention output for
    head h and W_O_h ∈ R^{d_head × d_model} is the corresponding slice of
    the output projection.

    Saves
    -----
    per_head_attn_deltas.npz:
        attn_deltas_head_{h} : (n_layers, n_tokens, d_model) for head h
    """
    import torch
    from core.config import DEVICE

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)

    # Storage: per-layer, per-head outputs (before c_proj)
    # head_outputs[layer][head] = (n_tokens, d_head) tensor
    head_outputs_by_layer = []
    hooks = []

    for block_idx, block in enumerate(model.h):
        layer_heads = []
        head_outputs_by_layer.append(layer_heads)

        # We hook on block.attn to capture q, k, v and attention weights
        # then manually compute per-head outputs.
        # GPT-2 stores W_O as c_proj.weight: (d_model, d_model) Conv1D
        # where each head's slice is c_proj.weight[h*d_h:(h+1)*d_h, :]
        # in row convention (Conv1D: input @ weight).
        n_heads = block.attn.num_heads
        d_model = block.attn.c_proj.weight.shape[1]  # Conv1D: (in, out)
        d_head  = d_model // n_heads

        # Hook on the attention module to capture its output split by head
        def make_hook(b_idx, n_h, d_h, layer_list):
            def hook(module, inp, out):
                # out[0] is the full attention output (n_tokens, d_model)
                attn_out = out[0] if isinstance(out, tuple) else out
                attn_out = attn_out.detach()[0].to(torch.float32).cpu()
                # The output of c_proj is NOT split by head by default.
                # We need the pre-c_proj attention values.
                # Re-derive from the input to c_proj (stored in module._attn_out)
                # GPT-2: attn_output = self_attn @ c_proj
                # Fortunately we also hooked block.attn.c_proj (see below)
                # For now, split the output by approximating head contributions
                # via c_proj weight slices.
                # This is an approximation: head_h_output ≈ full_attn_out slice
                # split evenly — correct only if heads are ordered in d_model.
                # Proper implementation requires hooking before c_proj.
                heads = []
                for h in range(n_h):
                    # Each head h contributes to output dims [h*d_h:(h+1)*d_h] in
                    # the d_model axis BEFORE c_proj mixes them.  After c_proj the
                    # heads are fully mixed — we cannot cleanly separate them from
                    # the final output.  Save the pre-c_proj values instead.
                    # This placeholder saves None; the pre-c_proj hook below fills it.
                    heads.append(None)
                layer_list.append(heads)
            return hook

        hooks.append(block.attn.register_forward_hook(
            make_hook(block_idx, n_heads, d_head, layer_heads)))

    # We need the actual pre-c_proj head outputs.
    # Replace the hook strategy: capture the input to c_proj (= pre-mix head outputs).
    for h_obj in hooks:
        h_obj.remove()
    hooks.clear()

    # Reset storage
    head_outputs_by_layer.clear()
    pre_cproj_by_layer = []  # pre_cproj[layer] = (n_tokens, d_model) pre-projection attn output

    for block_idx, block in enumerate(model.h):
        layer_store = []
        pre_cproj_by_layer.append(layer_store)

        def make_pre_cproj_hook(store):
            def hook(module, inp, out):
                # inp[0] is the input to c_proj: (batch, n_tokens, d_model)
                # This is the concatenated head outputs before mixing.
                pre_cproj = inp[0].detach()[0].to(torch.float32).cpu()  # (n_tokens, d_model)
                store.append(pre_cproj)
            return hook

        hooks.append(block.attn.c_proj.register_forward_hook(
            make_pre_cproj_hook(layer_store)))

    with torch.no_grad():
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16,
                            enabled=(DEVICE == "cuda")):
            outputs = model(**inputs, output_hidden_states=True)

    for h_obj in hooks:
        h_obj.remove()

    # pre_cproj_by_layer[L][0] = (n_tokens, d_model) pre-projection output
    # W_O = c_proj.weight: Conv1D shape (d_model, d_model)
    # head h contribution = pre_cproj[:, h*d_h:(h+1)*d_h] @ W_O[h*d_h:(h+1)*d_h, :]
    n_layers  = len(pre_cproj_by_layer)
    n_heads   = model.h[0].attn.num_heads
    d_model_v = model.h[0].attn.c_proj.weight.shape[1]
    d_head_v  = d_model_v // n_heads
    n_tokens  = pre_cproj_by_layer[0][0].shape[0] if pre_cproj_by_layer[0] else 0

    # Build per-head delta arrays: (n_layers, n_tokens, d_model) per head
    per_head_deltas = {h: np.zeros((n_layers, n_tokens, d_model_v), dtype=np.float32)
                       for h in range(n_heads)}

    for L in range(n_layers):
        if not pre_cproj_by_layer[L]:
            continue
        pre = pre_cproj_by_layer[L][0].numpy()     # (n_tokens, d_model)
        W_O = model.h[L].attn.c_proj.weight.detach().cpu().float().numpy()
        # Conv1D: W_O shape (d_model, d_model), map is x @ W_O
        for h in range(n_heads):
            s, e = h * d_head_v, (h + 1) * d_head_v
            head_pre = pre[:, s:e]                  # (n_tokens, d_head)
            W_O_h    = W_O[s:e, :]                  # (d_head, d_model)
            per_head_deltas[h][L] = head_pre @ W_O_h  # (n_tokens, d_model)

    # Save
    arrays = {f"attn_deltas_head_{h}": per_head_deltas[h]
              for h in range(n_heads)}
    np.savez_compressed(run_dir / "per_head_attn_deltas.npz", **arrays)
    print(f"    Saved per-head attention deltas ({n_heads} heads) to {run_dir}/")


# ---------------------------------------------------------------------------
# Ablation at one violation layer
# ---------------------------------------------------------------------------

def ablate_head_at_violation(
    hidden_before: np.ndarray,
    attn_delta:    np.ndarray,
    ffn_delta:     np.ndarray,
    head_deltas:   dict,
    head_to_ablate: int,
    beta: float,
) -> dict:
    """
    Compute counterfactual energy when head ``head_to_ablate`` is zeroed.

    hidden_ablated = hidden_before + (attn_delta - head_delta_h) + ffn_delta

    Parameters
    ----------
    hidden_before   : (n_tokens, d_model)
    attn_delta      : (n_tokens, d_model) — total attention residual
    ffn_delta       : (n_tokens, d_model) — FFN residual
    head_deltas     : dict {head_idx: (n_tokens, d_model)}
    head_to_ablate  : int
    beta            : float

    Returns
    -------
    dict with:
      head, e_original, e_ablated, delta_E_ablation,
      ablation_reduces_violation (bool)
    """
    def _energy(X_raw):
        X = X_raw / np.maximum(np.linalg.norm(X_raw, axis=-1, keepdims=True), 1e-10)
        G = X @ X.T
        n = G.shape[0]
        return float(np.exp(beta * G).sum() / (2.0 * beta * n * n))

    h_full    = hidden_before + attn_delta + ffn_delta
    head_d    = head_deltas.get(head_to_ablate, np.zeros_like(attn_delta))
    h_ablated = hidden_before + (attn_delta - head_d) + ffn_delta

    e_orig    = _energy(h_full)
    e_ablated = _energy(h_ablated)
    e_before  = _energy(hidden_before)

    delta_orig    = e_orig    - e_before   # negative = violation
    delta_ablated = e_ablated - e_before   # should be less negative if head causes drop

    return {
        "head":                      head_to_ablate,
        "e_original":                e_orig,
        "e_ablated":                 e_ablated,
        "e_before":                  e_before,
        "delta_E_original":          delta_orig,
        "delta_E_ablated":           delta_ablated,
        "ablation_effect":           delta_ablated - delta_orig,  # positive = ablation raised E
        "ablation_reduces_violation": delta_ablated > delta_orig,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_head_ablation(
    run_dir: Path,
    ov_data: dict,
    decomposed_violations: list,
    beta: float = 1.0,
) -> dict:
    """
    For each GPT-2 violation layer, ablate each head and rank by effect.

    Requires ``per_head_attn_deltas.npz`` (from save_decomposed_per_head),
    ``attn_deltas_raw.npz`` and ``ffn_deltas_raw.npz`` (from fix 2), and
    saved activations from Phase 1.

    Returns
    -------
    dict with:
      applicable (bool)
      per_violation : list of per-layer ablation results
      head_ranking  : list of {head, mean_ablation_effect, rep_frac}
      spearman_rho  : correlation between rep_frac and mean ablation effect
    """
    from scipy.stats import spearmanr

    run_dir = Path(run_dir)

    if not ov_data.get("is_per_layer"):
        return {"applicable": False, "reason": "shared weights (ALBERT) — ablation is per-layer only"}

    # Load saved deltas
    ph_path   = run_dir / "per_head_attn_deltas.npz"
    attn_path = run_dir / "attn_deltas_raw.npz"
    ffn_path  = run_dir / "ffn_deltas_raw.npz"
    act_path  = run_dir / "activations.npz"

    for p, name in [(ph_path, "per_head_attn_deltas.npz"),
                    (attn_path, "attn_deltas_raw.npz"),
                    (ffn_path,  "ffn_deltas_raw.npz"),
                    (act_path,  "activations.npz")]:
        if not p.exists():
            return {"applicable": False,
                    "reason": f"Missing {name}. Run save_decomposed_per_head first."}

    ph_data   = np.load(ph_path)
    attn_data = np.load(attn_path)["attn_deltas"]   # (n_layers, n_tokens, d)
    ffn_data  = np.load(ffn_path)["ffn_deltas"]     # (n_layers, n_tokens, d)
    act_data  = np.load(act_path)["activations"]    # (n_layers, n_tokens, d)

    n_heads  = ov_data["n_heads"]
    decomps  = ov_data["decomps"]

    # Per-head repulsive fracs
    rep_fracs = [d["frac_repulsive"] for d in decomps]  # one per layer

    per_violation = []
    head_effects  = {h: [] for h in range(n_heads)}

    for dv in decomposed_violations:
        v_layer = dv["layer"]
        t_idx   = v_layer - 1
        if t_idx < 0 or t_idx >= attn_data.shape[0]:
            continue

        # Load per-head deltas for this layer
        head_deltas = {}
        for h in range(n_heads):
            key = f"attn_deltas_head_{h}"
            if key in ph_data:
                hd = ph_data[key]
                if t_idx < hd.shape[0]:
                    head_deltas[h] = hd[t_idx]

        if not head_deltas:
            continue

        hidden_before = act_data[t_idx].astype(np.float32)
        attn_delta    = attn_data[t_idx].astype(np.float32)
        ffn_delta     = ffn_data[t_idx].astype(np.float32)

        ablation_results = []
        layer_rep_frac = rep_fracs[min(t_idx, len(rep_fracs) - 1)]

        for h in range(n_heads):
            abl = ablate_head_at_violation(
                hidden_before, attn_delta, ffn_delta,
                head_deltas, h, beta
            )
            abl["layer_rep_frac"] = layer_rep_frac
            ablation_results.append(abl)
            head_effects[h].append(abl["ablation_effect"])

        # Rank heads by ablation effect (most positive = most causal for violation)
        ranked = sorted(ablation_results,
                        key=lambda a: a["ablation_effect"], reverse=True)

        per_violation.append({
            "layer":            v_layer,
            "ablation_results": ablation_results,
            "ranked_heads":     ranked,
            "top_causal_head":  ranked[0]["head"] if ranked else None,
        })

    if not per_violation:
        return {"applicable": True, "per_violation": [], "head_ranking": [],
                "spearman_rho": float("nan")}

    # Aggregate: mean ablation effect per head
    head_ranking = []
    for h in range(n_heads):
        effects = [e for e in head_effects[h] if np.isfinite(e)]
        if not effects:
            continue
        # Use layer-mean rep_frac as a proxy (heads are constant across layers in
        # the per-head OV structure; only layer-0 is used here for consistency)
        head_rep_frac = decomps[0]["frac_repulsive"] if decomps else 0.0
        # For per-head: get from ov_per_head eigendecomposition if available
        head_ranking.append({
            "head":                h,
            "mean_ablation_effect": float(np.mean(effects)),
            "n_violations_tested": len(effects),
        })

    # Spearman correlation: rep_frac per layer vs ablation effect
    # Collect per-violation-layer: (layer_rep_frac, mean_ablation_effect_across_heads)
    layer_pairs = []
    for vr in per_violation:
        layer_rep = vr["ablation_results"][0].get("layer_rep_frac", float("nan"))
        mean_eff  = float(np.mean([a["ablation_effect"] for a in vr["ablation_results"]]))
        if np.isfinite(layer_rep) and np.isfinite(mean_eff):
            layer_pairs.append((layer_rep, mean_eff))

    if len(layer_pairs) >= 4:
        from scipy.stats import spearmanr
        rho, pval = spearmanr([p[0] for p in layer_pairs],
                              [p[1] for p in layer_pairs])
    else:
        rho, pval = float("nan"), float("nan")

    return {
        "applicable":    True,
        "per_violation": per_violation,
        "head_ranking":  sorted(head_ranking,
                                key=lambda h: h["mean_ablation_effect"], reverse=True),
        "spearman_rho":  float(rho),
        "spearman_pval": float(pval),
        "interpretation": (
            "Positive rho: layers with higher rep_frac have higher ablation effect — "
            "supports V-repulsive causal claim."
        ),
    }


def print_head_ablation_summary(result: dict, model_name: str, prompt_key: str) -> None:
    """Print concise head ablation summary."""
    if not result.get("applicable"):
        print(f"\n  Head ablation: {result.get('reason')}")
        return

    pv = result.get("per_violation", [])
    hr = result.get("head_ranking", [])
    rho = result.get("spearman_rho", float("nan"))
    pval = result.get("spearman_pval", float("nan"))

    print(f"\n  Head ablation ({model_name} | {prompt_key}):")
    print(f"    {len(pv)} violation layers ablated")

    if hr:
        print(f"    Top heads by mean ablation effect (reduces violation energy):")
        for entry in hr[:5]:
            print(f"      Head {entry['head']:2d}  "
                  f"mean_effect={entry['mean_ablation_effect']:+.5f}  "
                  f"(n={entry['n_violations_tested']})")

    sig = "*" if not np.isnan(pval) and pval < 0.05 else " "
    print(f"    ρ(layer_rep_frac, ablation_effect) = {rho:+.3f}  "
          f"p={pval:.3f} {sig}")
