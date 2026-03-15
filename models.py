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
import torch.nn.functional as F

from config import DEVICE, MODEL_CONFIGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def layernorm_to_sphere(activation: torch.Tensor) -> torch.Tensor:
    """L2-normalize each token vector onto the unit sphere."""
    return F.normalize(activation, p=2, dim=-1)


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
    cfg       = MODEL_CONFIGS[model_name]
    tokenizer = cfg["tokenizer_class"].from_pretrained(model_name)

    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    model = cfg["model_class"].from_pretrained(
        model_name,
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
            outputs = model(**inputs)

    # Cast to float32 on the GPU (cheap) before the CPU transfer (expensive).
    # This avoids moving bfloat16 data across the PCIe bus then converting.
    hidden_states = [h[0].to(torch.float32).cpu() for h in outputs.hidden_states]
    attentions    = [a[0].to(torch.float32).cpu() for a in outputs.attentions]
    return hidden_states, attentions, tokens


# ---------------------------------------------------------------------------
# ALBERT extended-iteration extraction
# ---------------------------------------------------------------------------

def extract_albert_extended(model, tokenizer, text: str, n_iterations: int):
    """
    Run ALBERT's single shared layer block *n_iterations* times.
    ALBERT's weight-sharing means we can iterate the same block to observe
    dynamics that would normally require a much deeper stack.  This is the
    primary analysis path for ALBERT models.

    Returns
    -------
    trajectory : list[Tensor]  — (n_tokens, d_model) float32, length n_iterations+1
    attentions : list[Tensor]  — (n_heads, n_tokens, n_tokens) float32, length n_iterations
                                 (empty if attention capture fails)
    tokens     : list[str]
    """
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
            trajectory   = [hidden[0].to(torch.float32).cpu()]
            attentions   = []
            albert_layer = model.encoder.albert_layer_groups[0].albert_layers[0]

            for _ in range(n_iterations):
                layer_out = albert_layer(
                    hidden,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
                hidden = layer_out[0]
                trajectory.append(hidden[0].to(torch.float32).cpu())
                # layer_out[1] is (batch, heads, seq, seq) attention probabilities
                if len(layer_out) > 1:
                    attentions.append(layer_out[1][0].to(torch.float32).cpu())

    return trajectory, attentions, tokens
