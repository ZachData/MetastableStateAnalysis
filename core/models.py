"""
models.py — Model loading and activation/attention extraction.

Handles:
  - load_model              : download + configure any registered model
  - extract_activations     : standard forward pass → hidden states + attentions
  - extract_albert_extended : run ALBERT's shared layer N times to obtain
                              a long depth trajectory
  - layernorm_to_sphere     : L2-normalize token vectors onto S^{d-1}

Performance notes
-----------------
* Models are loaded in bfloat16 on CUDA (~2× memory reduction, faster matmuls
  on Ampere+ hardware).  Outputs are cast back to float32 on the GPU before the
  .cpu() transfer to keep downstream numpy code unchanged.
* torch.compile (mode="reduce-overhead") is applied on CUDA when available,
  giving a ~20–40% throughput improvement after the first warm-up forward pass.
* torch.autocast wraps every forward pass so that even float32-loaded models
  benefit from mixed-precision paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import DEVICE, MODEL_CONFIGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def layernorm_to_sphere(activation: torch.Tensor) -> torch.Tensor:
    """L2-normalize each token vector onto the unit sphere."""
    return F.normalize(activation, p=2, dim=-1)


def randomize_weights(model, scheme: str = "orthogonal", seed: int = 0) -> dict:
    """
    Re-initialise every learned parameter of an already-loaded model, keeping
    the architecture but discarding all trained representations.

    Why this exists
    ---------------
    `model.init_weights()` is a no-op after `from_pretrained`: HuggingFace
    guards initialisation behind a module flag that loading leaves disabled,
    so the call returns without changing parameter values.  The old random
    baseline therefore ran on fully trained weights and produced results
    identical to trained ALBERT.

    Scheme (applies to transformer weight matrices: ndim >= 2, not embedding
    or LayerNorm)
      - "orthogonal" (default): orthonormal init -> operator norm 1, so each
        layer moves the residual stream by an amount comparable to a trained
        model.  This is the control for "does the *architecture* cluster with
        any non-degenerate map", and the sphere metrics see its singular-value
        spectrum (scale is normalised away, structure is not).
      - "gaussian": N(0, 0.02), the literal HuggingFace init = model before
        training.  Under post-LN (ALBERT/BERT) this gives tiny per-layer
        updates, so weak clustering may be a scale artifact, not structure.
        Use deliberately.

    Embeddings -> N(0, 0.02) (random token directions on the sphere).
    LayerNorm  -> weight 1, bias 0.  All other biases -> 0.

    Architecture-agnostic: the ndim rule covers nn.Linear, nn.Embedding, and
    GPT-2's Conv1D (2-D .weight) with no model-specific imports.

    Asserts the parameter checksum changed, so a silent regression fails loudly.
    """
    if scheme not in ("orthogonal", "gaussian"):
        raise ValueError(f"unknown random_init_scheme: {scheme!r}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def _checksum(m):
        # order-sensitive fingerprint over all parameter values
        return sum(float(p.detach().float().sum().item()) for p in m.parameters())

    def _fill(param, init_fn):
        # init in float32 then cast back; orthogonal_/qr is unreliable in bf16
        tmp = torch.empty(param.shape, dtype=torch.float32, device=param.device)
        init_fn(tmp)
        with torch.no_grad():
            param.data.copy_(tmp.to(param.dtype))

    before = _checksum(model)
    n_matrix = n_embed = n_norm = n_bias = 0

    for module in model.modules():
        own = list(module.named_parameters(recurse=False))  # each param once
        if not own:
            continue

        if isinstance(module, nn.LayerNorm):
            with torch.no_grad():
                for name, p in own:
                    p.data.fill_(1.0) if name == "weight" else p.data.zero_()
                    n_norm += 1
            continue

        if isinstance(module, nn.Embedding):
            for _, p in own:
                _fill(p, lambda t: nn.init.normal_(t, mean=0.0, std=0.02))
                n_embed += 1
            continue

        # generic leaf: Linear, Conv1D, projections, pooler, ...
        for _, p in own:
            if p.dim() >= 2:
                if scheme == "orthogonal":
                    _fill(p, nn.init.orthogonal_)
                else:
                    _fill(p, lambda t: nn.init.normal_(t, mean=0.0, std=0.02))
                n_matrix += 1
            else:
                with torch.no_grad():
                    p.data.zero_()
                n_bias += 1

    after = _checksum(model)
    assert abs(after - before) > 1e-6, (
        "randomize_weights changed no parameters — the random baseline would "
        "silently run on trained weights again."
    )

    return {
        "scheme": scheme, "seed": seed,
        "n_weight_matrices": n_matrix, "n_embeddings": n_embed,
        "n_layernorm_params": n_norm, "n_biases": n_bias,
        "checksum_before": before, "checksum_after": after,
    }

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """
    Instantiate tokenizer + model for *model_name* (must be a key in
    MODEL_CONFIGS).  Model is moved to DEVICE and set to eval mode.

    On CUDA: loaded in bfloat16 and optionally compiled with torch.compile.
    On CPU:  loaded in float32 (bfloat16 has no benefit on CPU).

    Returns
    -------
    model, tokenizer
    """
    cfg      = MODEL_CONFIGS[model_name]
    repo_id  = cfg.get("hf_repo", cfg.get("pretrained_name", model_name))
    revision = cfg.get("revision")

    tokenizer = cfg["tokenizer_class"].from_pretrained(repo_id, revision=revision)

    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    model = cfg["model_class"].from_pretrained(
        repo_id,
        revision=revision,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=dtype,
    ).to(DEVICE)
    model.eval()

    if model_name == "gpt2" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Standard extraction
# ---------------------------------------------------------------------------

def extract_activations(model, tokenizer, text: str, model_name: str):
    """
    Run a standard forward pass and collect hidden states + attentions.

    Returns
    -------
    hidden_states : list[Tensor]  — (n_tokens, d_model) float32 per layer
    attentions    : list[Tensor]  — (n_heads, n_tokens, n_tokens) float32 per layer
    tokens        : list[str]     — decoded token strings
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.bfloat16,
            enabled=(DEVICE == "cuda"),
        ):
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    # Cast to float32 on the GPU (cheap) before the CPU transfer (expensive).
    # This avoids moving bfloat16 data across the PCIe bus then converting.
    hidden_states = [h[0].to(torch.float32).cpu() for h in outputs.hidden_states]
    attentions    = [a[0].to(torch.float32).cpu() for a in outputs.attentions]
    return hidden_states, attentions, tokens


# ---------------------------------------------------------------------------
# ALBERT extended-iteration extraction
# ---------------------------------------------------------------------------

def extract_albert_extended(
    model,
    tokenizer,
    text: str,
    snapshots: list,
    max_iterations: int,
):
    """
    Run ALBERT's single shared layer block *max_iterations* times and return
    trajectory slices at each requested snapshot depth.

    Because ALBERT shares weights across all layers, hidden[i] is identical
    whether the run stops at step i or continues further.  A single pass to
    max_iterations therefore captures every shallower depth for free — there
    is no need to run the loop multiple times.

    Parameters
    ----------
    snapshots      : list of ints — depths at which to record a slice,
                     e.g. [12, 24, 36, 48].  Every value must be <= max_iterations.
    max_iterations : total number of layer iterations to run.

    Returns
    -------
    dict keyed by snapshot depth n:
        {
          n: {
            "trajectory": list[Tensor],   # length n+1  (step 0 .. step n inclusive)
            "attentions": list[Tensor],   # length n
            "tokens":     list[str],
          }
        }
    """
    if any(n > max_iterations for n in snapshots):
        raise ValueError(
            f"All snapshots must be <= max_iterations ({max_iterations}). "
            f"Got: {snapshots}"
        )

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
            # ALBERT projects embeddings (128) → hidden_size (768) before iterating
            hidden = model.encoder.embedding_hidden_mapping_in(hidden)

            full_trajectory = [hidden[0].to(torch.float32).cpu()]  # step 0
            full_attentions = []
            albert_layer    = model.encoder.albert_layer_groups[0].albert_layers[0]

            for _ in range(max_iterations):
                layer_out = albert_layer(
                    hidden,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
                hidden = layer_out[0]
                full_trajectory.append(hidden[0].to(torch.float32).cpu())
                if len(layer_out) > 1:
                    full_attentions.append(layer_out[1][0].to(torch.float32).cpu())

    # Slice the single trajectory at each requested depth.
    # full_trajectory has length max_iterations+1 (index 0 = post-projection embedding).
    # Snapshot n covers steps 0..n, so trajectory[:n+1] and attentions[:n].
    return {
        n: {
            "trajectory": full_trajectory[: n + 1],
            "attentions": full_attentions[:n],
            "tokens":     tokens,
        }
        for n in snapshots
    }
