"""
decompose.py — Model-required forward-pass decomposition.

Splits the residual update at each layer into attention and FFN components,
captures pre-LayerNorm states, and computes energy contributions from each.

Requires model loading.  For offline analysis on pre-saved data, use
trajectory.py instead.

Functions
---------
extract_decomposed_albert   : ALBERT extended with attn/FFN split
extract_decomposed_standard : standard models via forward hooks
energy_by_component         : energy delta from attn-only vs FFN-only
save_decomposed             : persist component trajectories for Phase 3
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from core.config import DEVICE, MODEL_CONFIGS
from core.models import layernorm_to_sphere


# ---------------------------------------------------------------------------
# ALBERT decomposed extraction
# ---------------------------------------------------------------------------

def extract_decomposed_albert(
    model,
    tokenizer,
    text: str,
    snapshots: list,
    max_iterations: int,
) -> dict:
    """
    Run ALBERT's shared layer with attn/FFN component capture.

    At each iteration, captures:
      - hidden_pre      : input to the layer
      - attn_delta      : attention output - hidden_pre (attention contribution)
      - ffn_delta        : FFN output - post-attention (FFN contribution)
      - hidden_post     : full layer output

    Parameters
    ----------
    snapshots      : depths at which to record slices
    max_iterations : total iterations

    Returns
    -------
    dict keyed by snapshot depth n:
      {n: {
        "trajectory":  list of tensors (n+1 states, step 0..n),
        "attn_deltas": list of tensors (n deltas),
        "ffn_deltas":  list of tensors (n deltas),
        "attentions":  list of tensors (n attention matrices),
        "tokens":      list of str,
      }}
    """
    if any(n > max_iterations for n in snapshots):
        raise ValueError(f"Snapshots must be <= max_iterations ({max_iterations})")

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.bfloat16,
            enabled=(DEVICE == "cuda"),
        ):
            embedding_output = model.embeddings(
                input_ids=inputs["input_ids"],
                token_type_ids=inputs.get("token_type_ids"),
            )
            hidden = embedding_output
            attention_mask = model.get_extended_attention_mask(
                inputs["attention_mask"], inputs["input_ids"].shape
            )
            hidden = model.encoder.embedding_hidden_mapping_in(hidden)

            full_trajectory = [hidden[0].to(torch.float32).cpu()]
            full_attn_deltas = []
            full_ffn_deltas  = []
            full_attentions  = []

            albert_layer = model.encoder.albert_layer_groups[0].albert_layers[0]

            for _ in range(max_iterations):
                hidden_pre = hidden

                # --- Attention sub-step ---
                attn_out = albert_layer.attention(
                    hidden,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
                # attn_out: (context_layer, attention_probs)
                # context_layer already has the dense projection applied
                # The full attention block includes: self-attn → dense → dropout → LayerNorm(residual + output)
                # We need to separate the attention residual from the FFN.
                #
                # ALBERT's forward: attention_output → ffn → ffn_output → LayerNorm(hidden + ffn_output)
                # But the actual structure is:
                #   attention.output = LayerNorm(attention_dense(self_attn) + input)
                #   ffn = activation(intermediate_dense(attention_output))
                #   hidden = LayerNorm(ffn_dense(ffn) + attention_output)

                # The attention call returns the post-LayerNorm attention output
                if isinstance(attn_out, tuple):
                    attention_output = attn_out[0]
                    attn_probs = attn_out[1] if len(attn_out) > 1 else None
                else:
                    attention_output = attn_out
                    attn_probs = None

                attn_delta = (attention_output - hidden_pre)[0].to(torch.float32).cpu()
                full_attn_deltas.append(attn_delta)

                if attn_probs is not None:
                    full_attentions.append(attn_probs[0].to(torch.float32).cpu())

                # --- FFN sub-step ---
                # AlbertLayer forward: ffn (linear) → activation (gelu) → ffn_output (linear)
                # then full_layer_layer_norm(ffn_result + attention_output)
                ffn_out = albert_layer.ffn(attention_output)
                ffn_out = albert_layer.activation(ffn_out)
                ffn_out = albert_layer.ffn_output(ffn_out)
                hidden  = albert_layer.full_layer_layer_norm(ffn_out + attention_output)

                ffn_delta = (hidden - attention_output)[0].to(torch.float32).cpu()
                full_ffn_deltas.append(ffn_delta)

                full_trajectory.append(hidden[0].to(torch.float32).cpu())

    # Slice at each snapshot
    return {
        n: {
            "trajectory":  full_trajectory[:n + 1],
            "attn_deltas": full_attn_deltas[:n],
            "ffn_deltas":  full_ffn_deltas[:n],
            "attentions":  full_attentions[:n],
            "tokens":      tokens,
        }
        for n in snapshots
    }


# ---------------------------------------------------------------------------
# Standard models via forward hooks
# ---------------------------------------------------------------------------

def extract_decomposed_standard(
    model,
    tokenizer,
    text: str,
    model_name: str,
) -> dict:
    """
    Extract attn/FFN decomposition for non-ALBERT models using hooks.

    For BERT: hooks on attention.output.dense and output.dense
    For GPT-2: hooks on attn.c_proj and mlp.c_proj

    Returns
    -------
    dict with:
      trajectory  : list of (n_tokens, d) tensors — hidden states per layer
      attn_deltas : list of (n_tokens, d) tensors — attention contribution
      ffn_deltas  : list of (n_tokens, d) tensors — FFN contribution
      tokens      : list of str
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    attn_outputs = []
    ffn_outputs  = []
    hooks        = []

    if "bert" in model_name and "albert" not in model_name:
        # BERT: capture attention output (pre-residual) and FFN output (pre-residual)
        for layer in model.encoder.layer:
            # Attention output (after dense, before residual add + LayerNorm)
            def make_attn_hook(layer_ref):
                def hook(module, input, output):
                    # input[0] is the attention dense output
                    attn_outputs.append(output.detach()[0].to(torch.float32).cpu())
                return hook

            def make_ffn_hook(layer_ref):
                def hook(module, input, output):
                    ffn_outputs.append(output.detach()[0].to(torch.float32).cpu())
                return hook

            h1 = layer.attention.output.register_forward_hook(make_attn_hook(layer))
            h2 = layer.output.register_forward_hook(make_ffn_hook(layer))
            hooks.extend([h1, h2])

    elif "gpt2" in model_name:
        for block in model.h:
            def make_attn_hook(block_ref):
                def hook(module, input, output):
                    # GPT-2 attn returns (attn_output, present, attn_weights)
                    if isinstance(output, tuple):
                        attn_outputs.append(output[0].detach()[0].to(torch.float32).cpu())
                    else:
                        attn_outputs.append(output.detach()[0].to(torch.float32).cpu())
                return hook

            def make_ffn_hook(block_ref):
                def hook(module, input, output):
                    ffn_outputs.append(output.detach()[0].to(torch.float32).cpu())
                return hook

            h1 = block.attn.register_forward_hook(make_attn_hook(block))
            h2 = block.mlp.register_forward_hook(make_ffn_hook(block))
            hooks.extend([h1, h2])

    # Forward pass
    with torch.no_grad():
        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.bfloat16,
            enabled=(DEVICE == "cuda"),
        ):
            outputs = model(**inputs, output_hidden_states=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Hidden states
    hidden_states = [h[0].to(torch.float32).cpu() for h in outputs.hidden_states]

    # Compute deltas from hook captures
    # For GPT-2: each block does hidden = hidden + attn(ln1(hidden)) then hidden = hidden + mlp(ln2(hidden))
    # The hook on attn captures attn(ln1(hidden)) (the residual delta)
    # The hook on mlp captures mlp(ln2(hidden)) (the residual delta)
    # So attn_delta = attn_outputs[i] and ffn_delta = ffn_outputs[i]

    # For BERT: attention.output does LayerNorm(attn_dense + input), so the hook
    # captures the post-LN output, not the delta. Similarly for output.
    # The deltas need to be computed differently for BERT.

    attn_deltas = []
    ffn_deltas  = []

    if "gpt2" in model_name:
        # GPT-2 hooks capture the residual stream deltas directly
        attn_deltas = attn_outputs
        ffn_deltas  = ffn_outputs
    elif "bert" in model_name:
        # BERT hooks capture post-LN outputs; compute deltas from hidden states
        # attention.output = LayerNorm(attn_dense(self_attn) + input)
        # output = LayerNorm(ffn(attention.output) + attention.output)
        # So attn_delta ≈ attention.output - hidden[L], ffn_delta ≈ hidden[L+1] - attention.output
        for i in range(len(attn_outputs)):
            if i < len(hidden_states) - 1:
                attn_deltas.append(attn_outputs[i] - hidden_states[i])
                ffn_deltas.append(hidden_states[i + 1] - attn_outputs[i])

    return {
        "trajectory":  hidden_states,
        "attn_deltas": attn_deltas,
        "ffn_deltas":  ffn_deltas,
        "tokens":      tokens,
    }


# ---------------------------------------------------------------------------
# Energy attribution by component
# ---------------------------------------------------------------------------

def energy_by_component(
    hidden_before: np.ndarray,
    attn_delta: np.ndarray,
    ffn_delta: np.ndarray,
    beta: float,
) -> dict:
    """
    Compute energy change from attention-only and FFN-only updates.

    hidden_after = hidden_before + attn_delta + ffn_delta
    hidden_attn_only = hidden_before + attn_delta
    hidden_ffn_only  = hidden_before + ffn_delta

    Energy is computed on L2-normed versions of each.

    Parameters
    ----------
    hidden_before : (n_tokens, d) float
    attn_delta    : (n_tokens, d) float
    ffn_delta     : (n_tokens, d) float
    beta          : float

    Returns
    -------
    dict with:
      e_before, e_after, e_attn_only, e_ffn_only : float
      delta_total, delta_attn, delta_ffn          : float
      attn_frac, ffn_frac                         : float (of total delta)
    """
    def _energy(X_raw):
        X = X_raw / np.maximum(np.linalg.norm(X_raw, axis=-1, keepdims=True), 1e-10)
        G = X @ X.T
        n = G.shape[0]
        return float(np.exp(beta * G).sum() / (2.0 * beta * n * n))

    h_after     = hidden_before + attn_delta + ffn_delta
    h_attn_only = hidden_before + attn_delta
    h_ffn_only  = hidden_before + ffn_delta

    e_before    = _energy(hidden_before)
    e_after     = _energy(h_after)
    e_attn_only = _energy(h_attn_only)
    e_ffn_only  = _energy(h_ffn_only)

    delta_total = e_after - e_before
    delta_attn  = e_attn_only - e_before
    delta_ffn   = e_ffn_only - e_before

    # Attribution fraction (how much of the total delta each component explains)
    denom = abs(delta_attn) + abs(delta_ffn)
    if denom < 1e-12:
        attn_frac = 0.5
        ffn_frac  = 0.5
    else:
        attn_frac = abs(delta_attn) / denom
        ffn_frac  = abs(delta_ffn) / denom

    return {
        "e_before":    e_before,
        "e_after":     e_after,
        "e_attn_only": e_attn_only,
        "e_ffn_only":  e_ffn_only,
        "delta_total": delta_total,
        "delta_attn":  delta_attn,
        "delta_ffn":   delta_ffn,
        "delta_cross": delta_total - delta_attn - delta_ffn,
        "attn_frac":   attn_frac,
        "ffn_frac":    ffn_frac,
        # Sign: negative = energy-decreasing (violation direction)
        "attn_sign":   "drop" if delta_attn < -1e-6 else "rise",
        "ffn_sign":    "drop" if delta_ffn < -1e-6 else "rise",
        "cross_sign":  "drop" if (delta_total - delta_attn - delta_ffn) < -1e-6 else "rise",
    }


# ---------------------------------------------------------------------------
# Full decomposed analysis for violation layers
# ---------------------------------------------------------------------------

def analyze_violations_decomposed(
    decomposed: dict,
    phase1_events: dict,
    beta: float = 1.0,
) -> list:
    """
    Run energy_by_component at each violation layer.

    Parameters
    ----------
    decomposed    : dict from extract_decomposed_albert or _standard
    phase1_events : dict from trajectory.load_phase1_events

    Returns
    -------
    list of dicts (one per violation layer) with energy attribution
    """
    violations = phase1_events["energy_violations"].get(beta, [])
    trajectory = decomposed["trajectory"]
    attn_deltas = decomposed["attn_deltas"]
    ffn_deltas  = decomposed["ffn_deltas"]

    results = []
    for v_layer in violations:
        # Violation at layer v_layer: energy dropped from v_layer-1 to v_layer
        # The update happens at step v_layer-1 → v_layer
        t_idx = v_layer - 1
        if t_idx < 0 or t_idx >= len(attn_deltas):
            continue

        h_before = trajectory[t_idx].numpy() if hasattr(trajectory[t_idx], 'numpy') else trajectory[t_idx]
        a_delta  = attn_deltas[t_idx].numpy() if hasattr(attn_deltas[t_idx], 'numpy') else attn_deltas[t_idx]
        f_delta  = ffn_deltas[t_idx].numpy() if hasattr(ffn_deltas[t_idx], 'numpy') else ffn_deltas[t_idx]

        attribution = energy_by_component(h_before, a_delta, f_delta, beta)
        attribution["layer"] = v_layer
        results.append(attribution)

    return results


# ---------------------------------------------------------------------------
# Save for Phase 3
# ---------------------------------------------------------------------------

def save_decomposed(
    decomposed: dict,
    run_dir: Path,
) -> None:
    """
    Save attn and FFN deltas for Phase 3 crosscoder training.

    Writes:
      attn_deltas.npz  — (n_layers, n_tokens, d) L2-normed attention deltas
      ffn_deltas.npz   — (n_layers, n_tokens, d) L2-normed FFN deltas
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def _stack_and_norm(deltas):
        if not deltas:
            return np.array([])
        stacked = np.stack([
            d.numpy() if hasattr(d, 'numpy') else d for d in deltas
        ])
        # L2-normalize for consistency with Phase 1's activation storage
        norms = np.linalg.norm(stacked, axis=-1, keepdims=True)
        return stacked / np.maximum(norms, 1e-10)

    attn_normed = _stack_and_norm(decomposed["attn_deltas"])
    ffn_normed  = _stack_and_norm(decomposed["ffn_deltas"])

    if attn_normed.size > 0:
        np.savez_compressed(run_dir / "attn_deltas.npz", attn_deltas=attn_normed)
    if ffn_normed.size > 0:
        np.savez_compressed(run_dir / "ffn_deltas.npz", ffn_deltas=ffn_normed)

    print(f"    Saved decomposed deltas to {run_dir}/")
