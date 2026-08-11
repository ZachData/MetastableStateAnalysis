"""
head_ablation.py — Per-head OV ablation at violation layers
(GPT-2 and GPT-NeoX/Pythia).

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
save_decomposed_per_head         : save per-head attn deltas (GPT-2, GPT-NeoX)
ablate_head_at_violation         : counterfactual energy for one head
run_head_ablation                : full per-violation ablation analysis
print_head_ablation_summary      : terminal output
"""

import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Per-head hook extraction (GPT-2)
# ---------------------------------------------------------------------------

def _locate_attn_projection(model):
    """
    Per-architecture locator for the attention blocks and the output
    projection whose INPUT is the concatenated per-head attention output.

    Returns
    -------
    (blocks, get_proj, get_heads, conv1d) where
      blocks    : iterable of transformer blocks
      get_proj  : block -> output-projection module (hook target)
      get_heads : block -> (n_heads, d_head)
      conv1d    : True  -> GPT-2 Conv1D convention, weight (in, out),
                           map y = x @ W;   head h rows  W[s:e, :]
                  False -> nn.Linear convention, weight (out, in),
                           map y = x @ W.T; head h slice W[:, s:e].T
    In both conventions the pre-projection tensor has head h at columns
    [h*d_head:(h+1)*d_head] — standard contiguous concat — so only the
    weight-slice orientation differs. Mirrors the dispatch
    p5_single_mstate_analysis/causal_tests.py's _block_attn_projection
    already uses for the same two architectures.
    """
    # GPT-2: unwrapped (GPT2Model, blocks at model.h) or wrapped
    # (model.transformer.h)
    transformer = getattr(model, "transformer", model)
    if hasattr(transformer, "h"):
        blocks = transformer.h

        def get_proj(block):
            return block.attn.c_proj

        def get_heads(block):
            n_h = block.attn.num_heads
            d_m = block.attn.c_proj.weight.shape[1]   # Conv1D: (in, out)
            return n_h, d_m // n_h

        return blocks, get_proj, get_heads, True

    # GPT-NeoX / Pythia: unwrapped (GPTNeoXModel, blocks at model.layers)
    # or wrapped (model.gpt_neox.layers)
    inner = getattr(model, "gpt_neox", model)
    if hasattr(inner, "layers") and len(inner.layers) \
            and hasattr(inner.layers[0], "attention") \
            and hasattr(inner.layers[0].attention, "query_key_value"):
        blocks = inner.layers

        def get_proj(block):
            return block.attention.dense

        def get_heads(block):
            attn = block.attention
            return attn.num_attention_heads, attn.head_size

        return blocks, get_proj, get_heads, False

    raise NotImplementedError(
        f"save_decomposed_per_head: unsupported architecture "
        f"{type(model).__name__} — GPT-2 and GPT-NeoX/Pythia only. "
        f"ALBERT/BERT are out of the forward path per the transition "
        f"plan's Pythia-only scope decision."
    )


def head_delta_from_projection(pre: "np.ndarray", W: "np.ndarray",
                               h: int, d_head: int, conv1d: bool) -> "np.ndarray":
    """
    Head h's residual-stream contribution from the pre-projection tensor.

    pre : (n_tokens, d_model) concatenated head outputs (input to the
          output projection), head h at columns [h*d_head:(h+1)*d_head].
    W   : the output projection's raw .weight —
          Conv1D (d_model, d_model), map x @ W          (conv1d=True)
          Linear (d_model, d_model), map x @ W.T        (conv1d=False)

    Pure numpy — the exactness property Σ_h head_delta_h == full
    projection output is verified in tests/test_head_ablation_math.py.
    """
    s, e = h * d_head, (h + 1) * d_head
    head_pre = pre[:, s:e]                       # (n_tokens, d_head)
    if conv1d:
        W_O_h = W[s:e, :]                        # (d_head, d_model)
    else:
        W_O_h = W[:, s:e].T                      # (d_head, d_model)
    return head_pre @ W_O_h                      # (n_tokens, d_model)


def save_decomposed_per_head(
    model,
    tokenizer,
    text: str,
    run_dir: Path,
) -> None:
    """
    Extract and save per-head attention output deltas (GPT-2, GPT-NeoX).

    For each block, captures the per-head attention output BEFORE the
    output projection (c_proj for GPT-2, attention.dense for GPT-NeoX).
    The output projection mixes heads; pre-projection outputs are needed
    to isolate each head's contribution to the residual stream:

        head_delta_h = head_out_h @ W_O_h

    where head_out_h ∈ R^{n_tokens × d_head} is head h's slice of the
    projection input and W_O_h is the matching slice of the projection
    weight (orientation per architecture — see head_delta_from_projection).

    Saves
    -----
    per_head_attn_deltas.npz:
        attn_deltas_head_{h} : (n_layers, n_tokens, d_model) for head h
    """
    import torch

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # The model's OWN device, not core.config.DEVICE. A caller that hands us
    # a CPU model on a CUDA box (any from_pretrained without an explicit
    # .to(), which is what the smoke fixtures do) otherwise gets cuda inputs
    # against cpu weights and an index_select device mismatch inside the
    # embedding. This is the idiom the rest of the project already uses
    # (core/models.py, core/sublayer_streams.py, causal_tests.py).
    device = next(model.parameters()).device

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    blocks, get_proj, get_heads, conv1d = _locate_attn_projection(model)

    # Capture the input to each block's output projection — the
    # concatenated pre-mix head outputs.
    pre_proj_by_layer = []   # pre_proj[layer] = [(n_tokens, d_model)]
    hooks = []

    for block in blocks:
        layer_store = []
        pre_proj_by_layer.append(layer_store)

        def make_pre_proj_hook(store):
            def hook(module, inp, out):
                # inp[0]: (batch, n_tokens, d_model) projection input
                store.append(inp[0].detach()[0].to(torch.float32).cpu())
            return hook

        hooks.append(get_proj(block).register_forward_hook(
            make_pre_proj_hook(layer_store)))

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            model(**inputs, output_hidden_states=True)

    for h_obj in hooks:
        h_obj.remove()

    n_layers = len(pre_proj_by_layer)
    n_heads, d_head = get_heads(blocks[0])
    first = next((s[0] for s in pre_proj_by_layer if s), None)
    if first is None:
        raise RuntimeError(
            "save_decomposed_per_head: no projection inputs captured — "
            "hook target never fired; check _locate_attn_projection "
            "against this transformers version's module names."
        )
    n_tokens = first.shape[0]
    d_model  = first.shape[1]

    per_head_deltas = {h: np.zeros((n_layers, n_tokens, d_model), dtype=np.float32)
                       for h in range(n_heads)}

    for L, (block, store) in enumerate(zip(blocks, pre_proj_by_layer)):
        if not store:
            continue
        pre = store[0].numpy()                                     # (n_tokens, d_model)
        W   = get_proj(block).weight.detach().cpu().float().numpy()
        for h in range(n_heads):
            per_head_deltas[h][L] = head_delta_from_projection(
                pre, W, h, d_head, conv1d
            )

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
