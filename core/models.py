"""
core/models.py — Model loading and activation/attention extraction.

Handles:
  - load_model              : download + configure any registered model
  - extract_activations     : standard forward pass → hidden states + attentions
  - extract_albert_extended : run ALBERT's shared layer N times to obtain
                              a long depth trajectory
  - describe_extraction     : the extraction_meta dict analyze_trajectory records
  - layernorm_to_sphere     : L2-normalize token vectors onto S^{d-1}

Precision
---------
Default dtype is float32 (core.config.MODEL_DTYPE), not bfloat16. This is a
correctness choice, not a conservatism:

* Pythia checkpoints are stored as float16 on the Hub. Loading them as
  float32 is an exact upcast — every stored bit survives. Loading them as
  bfloat16 is a lossy re-quantisation, since fp16 carries 10 mantissa bits
  and bf16 carries 7.
* The quantity that suffers is the V eigenspectrum. eigvals() on a
  non-normal matrix has no backward-stability guarantee tied to the input
  perturbation, so eig_frac_pos_real / eig_frac_neg_real are unreliable near
  the zero crossing at bf16 input precision. Those, with eig_spectral_radius,
  are what the attractive/repulsive regime classification rests on — §3.2 and
  §9.1 (sign of V sets the sign of dE_beta/dt) and Table 1 in §9.2 (the sign
  and multiplicity of lambda_1(V) sets the predicted limit geometry). They
  were formerly described here as supporting a "Thm 6.1" falsification;
  Thm 6.1 is qualitative (d>=3, any beta) and V's spectrum says nothing
  about it.
* effective_rank feeds DEGENERATE_RANK_THRESHOLD, which gates CKA and
  NN-stability. A gate driven by a low-precision rank estimate flips
  silently.

Set MODEL_DTYPE = "auto" (or pass --dtype auto) to restore the previous
behaviour: bfloat16 on CUDA, float32 on CPU. Whatever is chosen is recorded
in experiment.txt and in every v_eigenspectrum JSON, so the dtype can never
again be an invisible term in a cross-run comparison.

autocast is enabled only when the model's own parameters are already in a
reduced-precision dtype. It used to be gated on `DEVICE == "cuda"` alone,
which meant a float32-loaded model still ran its forward pass through
bfloat16 kernels — reintroducing exactly the loss the fp32 load avoids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import DEVICE, MODEL_CONFIGS, MODEL_DTYPE
from core.model_family import model_family


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def layernorm_to_sphere(activation: torch.Tensor) -> torch.Tensor:
    """L2-normalize each token vector onto the unit sphere."""
    return F.normalize(activation, p=2, dim=-1)


_DTYPE_ALIASES = {
    "float32":  torch.float32,
    "fp32":     torch.float32,
    "32":       torch.float32,
    "float64":  torch.float64,
    "fp64":     torch.float64,
    "64":       torch.float64,
    "bfloat16": torch.bfloat16,
    "bf16":     torch.bfloat16,
    "float16":  torch.float16,
    "fp16":     torch.float16,
    "16":       torch.float16,
}

# dtypes for which mixed-precision autocast is meaningful on the forward pass
_REDUCED_PRECISION = (torch.bfloat16, torch.float16)


def resolve_dtype(spec=None) -> torch.dtype:
    """Turn a dtype spec into a torch.dtype.

    spec may be None (use core.config.MODEL_DTYPE), a torch.dtype, "auto",
    or any key of _DTYPE_ALIASES. "auto" means bfloat16 on CUDA and float32
    on CPU — the pre-fix behaviour, kept reachable so old runs can be
    reproduced deliberately rather than by accident.
    """
    if spec is None:
        spec = MODEL_DTYPE
    if isinstance(spec, torch.dtype):
        return spec

    key = str(spec).strip().lower()
    if key == "auto":
        return torch.bfloat16 if DEVICE == "cuda" else torch.float32
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"unknown dtype spec {spec!r}; expected 'auto' or one of "
            f"{sorted(set(_DTYPE_ALIASES))}"
        )
    return _DTYPE_ALIASES[key]


def model_dtype(model) -> torch.dtype:
    """dtype of the model's first parameter — what it was actually loaded as."""
    return next(model.parameters()).dtype


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _autocast_enabled(model) -> bool:
    return DEVICE == "cuda" and model_dtype(model) in _REDUCED_PRECISION


def _autocast_dtype(model) -> torch.dtype:
    dt = model_dtype(model)
    return dt if dt in _REDUCED_PRECISION else torch.bfloat16


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
        Use deliberately.  GPT-NeoX does *not* init at N(0, 0.02) — its
        variance scaling differs from GPT-2's — so on Pythia this scheme is
        not "the model before training" and should not be read as such.
      - "norm_matched": Gaussian base rescaled per parameter to that
        parameter's *trained* Frobenius norm.  Structure destroyed, scale
        preserved.  This is the continuity control for the trained-vs-random
        contrast (PREDICTIONS.md claim (c)), and the only scheme that
        transfers that contrast across architectures whose init variance
        scaling differs.

    Why "norm_matched" is not redundant with the other two
    ------------------------------------------------------
    "orthogonal" fixes every singular value at 1, so it matches operator norm
    but flattens the spectrum — and the spectrum is exactly what Phase 2
    measures.  "gaussian" fixes the base std, which is architecture-specific
    and therefore does not transfer.  "norm_matched" fixes ||W||_F and leaves
    the spectrum to be whatever a structureless matrix of that scale gives.

    The rescale makes the result independent of the base std (verified to
    machine precision), which is the property that makes it architecture-
    portable: two models with different init conventions get the same
    construction.

    Frobenius, not spectral, is matched.  For a token vector in generic
    position ||Wx|| ~ ||W||_F/sqrt(d) * ||x||, so Frobenius governs typical
    residual displacement — which is what E_beta and the trajectory metrics
    see.  The cost is explicit: against a trained matrix's heavy-tailed
    spectrum, a Frobenius-matched Gaussian has a *smaller* operator norm
    (measured ~0.48x on a synthetic heavy-tailed stand-in).  A run that
    depends on worst-case rather than typical displacement wants a different
    control, and should say so rather than reusing this one.

    Embeddings -> N(0, 0.02) under "orthogonal"/"gaussian"; norm-matched to
    the trained embedding matrix under "norm_matched", because the particle
    cloud's initial radius sets layer-0 geometry and core/polar.py now
    measures that radius directly.
    LayerNorm  -> weight 1, bias 0 under every scheme, including
    "norm_matched".  LN gamma is a diagonal rescale that core/ln_frame.py
    already tracks as its own object; identity is the conventional null and
    keeps continuity with the gpt2-large-random baseline Blog 1 established.
    All other biases -> 0 under every scheme (this discards GPT-NeoX's
    trained attention biases deliberately; see core/attn_biases.py).

    Architecture-agnostic: the ndim rule covers nn.Linear, nn.Embedding, and
    GPT-2's Conv1D (2-D .weight) with no model-specific imports.

    Asserts the parameter checksum changed, so a silent regression fails loudly.
    """
    if scheme not in ("orthogonal", "gaussian", "norm_matched"):
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

    def _fill_norm_matched(param):
        """
        Overwrite `param` with a Gaussian draw rescaled to its own current
        (trained) Frobenius norm.

        The target norm is read *before* the overwrite, in float32, so a
        bf16/fp16 parameter is matched against its own value rather than a
        rounded one. A parameter whose trained norm is 0 stays 0 — rescaling
        would divide by the draw's norm to reach a target of 0 anyway, and
        the explicit branch keeps that from depending on float behaviour.
        """
        target = float(param.detach().float().norm())
        tmp = torch.empty(param.shape, dtype=torch.float32, device=param.device)
        nn.init.normal_(tmp, mean=0.0, std=0.02)
        cur = float(tmp.norm())
        if target > 0.0 and cur > 0.0:
            tmp.mul_(target / cur)
        else:
            tmp.zero_()
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
                if scheme == "norm_matched":
                    _fill_norm_matched(p)
                else:
                    _fill(p, lambda t: nn.init.normal_(t, mean=0.0, std=0.02))
                n_embed += 1
            continue

        # generic leaf: Linear, Conv1D, projections, pooler, ...
        for _, p in own:
            if p.dim() >= 2:
                if scheme == "orthogonal":
                    _fill(p, nn.init.orthogonal_)
                elif scheme == "norm_matched":
                    _fill_norm_matched(p)
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

def load_model(model_name: str, dtype=None):
    """
    Instantiate tokenizer + model for *model_name* (must be a key in
    MODEL_CONFIGS).  Model is moved to DEVICE and set to eval mode.

    Parameters
    ----------
    dtype : dtype spec (see resolve_dtype). None -> core.config.MODEL_DTYPE,
            which defaults to float32. See the module docstring for why the
            default is not bfloat16.

    attn_implementation is pinned to "eager". GPT-NeoX's sdpa path returns
    no attention weights and only falls back to eager via a deprecation
    shim; when that shim is removed, output_attentions=True would start
    yielding empty attentions and Phase 1's entire sinkhorn/Fiedler/entropy
    family would go quiet without raising.

    Returns
    -------
    model, tokenizer
    """
    cfg      = MODEL_CONFIGS[model_name]
    repo_id  = cfg.get("hf_repo", cfg.get("pretrained_name", model_name))
    revision = cfg.get("revision")

    # Falls back to `revision` when the key is absent, so models whose
    # tokenizer genuinely varies by revision keep the old behaviour. Pythia
    # sets it to None explicitly: one tokenizer, 37 checkpoints.
    tok_revision = cfg.get("tokenizer_revision", revision)
    tokenizer = cfg["tokenizer_class"].from_pretrained(repo_id, revision=tok_revision)

    torch_dtype = resolve_dtype(dtype)

    load_kwargs = dict(
        revision=revision,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    try:
        model = cfg["model_class"].from_pretrained(repo_id, **load_kwargs)
    except (TypeError, ValueError):
        # Older transformers releases don't accept attn_implementation.
        # Falling back is safe for BERT/ALBERT/GPT-2 (eager is their only
        # path); for GPT-NeoX it restores the deprecation-shim behaviour.
        load_kwargs.pop("attn_implementation")
        model = cfg["model_class"].from_pretrained(repo_id, **load_kwargs)

    model = model.to(DEVICE)
    model.eval()

    if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None):
        # Previously gated on model_name == "gpt2"; every decoder-only
        # tokenizer in the registry now needs it, and none of them pad
        # during Phase 1 anyway, so an unconditional default is safer than
        # a per-model list that silently misses new entries.
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Extraction metadata
# ---------------------------------------------------------------------------

def _has_final_layernorm(model) -> bool:
    """True when the last hidden state has passed a stack-final LayerNorm.

    GPT-2 exposes it as .ln_f, GPT-NeoX as .final_layer_norm. BERT and
    ALBERT are post-LN per block with no stack-final norm, so their last
    hidden state is the same kind of object as every earlier one.

    This matters because it is the honest version of the claim
    `lm_head_excluded` was standing in for: on GPT-2 and Pythia the final
    entry of hidden_states is already shaped toward the output head, so it
    is not comparable to layers 1..L-1. status-1 blocker #4.
    """
    return any(hasattr(model, attr) for attr in ("ln_f", "final_layer_norm"))

def describe_extraction(model, model_name: str, hidden_states, attentions) -> dict:
    """Build the extraction_meta dict analyze_trajectory records.

    This exists as a separate function rather than a fourth return value of
    extract_activations because run_1b.py and train_tuned_lens.py both
    unpack that call as a 3-tuple.

    Checkpoint provenance is included so geometry.json is self-describing
    along the developmental axis. Without it, the only record of which
    training step produced a run is the model name, and checkpoints.py
    recovers the step by parsing it back out of the directory name — which
    means a rename silently re-labels the x-axis of every developmental
    plot.
    """
    # Strip the @attn / @ffn / @48iter variant suffix before the lookup.
    base_name = str(model_name).split("@")[0]
    cfg = MODEL_CONFIGS.get(base_name, {})

    return {
        # No registry entry loads a model with an LM head (GPT2Model,
        # GPTNeoXModel, AlbertModel, BertModel are all base classes), and
        # nothing downstream strips a layer, so this is False and honest.
        "lm_head_excluded":              False,
        "n_layers_total":                len(hidden_states),
        "n_layers_analyzed":             len(hidden_states),
        "n_attention_layers":            len(attentions),
        "hidden_state_0_is_embedding":   True,
        "final_hidden_state_is_post_ln": _has_final_layernorm(model),
        "model_family":                  model_family(model_name),
        "weight_dtype":                  dtype_name(model_dtype(model)),
        "autocast":                      _autocast_enabled(model),
        "device":                        DEVICE,
        # --- checkpoint provenance ---
        "hf_repo":                       cfg.get("hf_repo"),
        "revision":                      cfg.get("revision"),
        "tokenizer_revision":            cfg.get("tokenizer_revision", cfg.get("revision")),
        "checkpoint_step":               cfg.get("checkpoint_step"),
        "random_init":                   cfg.get("random_init", False),
        "random_init_scheme":            cfg.get("random_init_scheme"),
    }

# ---------------------------------------------------------------------------
# Standard extraction
# ---------------------------------------------------------------------------

def extract_activations(model, tokenizer, text: str, model_name: str):
    """
    Run a standard forward pass and collect hidden states + attentions.

    Returns
    -------
    hidden_states : list[Tensor]  — (n_tokens, d_model) float32 per layer,
                                    index 0 = embedding output, index L =
                                    final block output (post-LN on GPT-2 /
                                    GPT-NeoX)
    attentions    : list[Tensor]  — (n_heads, n_tokens, n_tokens) float32 per layer
    tokens        : list[str]     — decoded token strings
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        with torch.autocast(
            device_type=DEVICE,
            dtype=_autocast_dtype(model),
            enabled=_autocast_enabled(model),
        ):
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    if not outputs.attentions:
        raise RuntimeError(
            f"{model_name}: output_attentions=True returned no attention "
            "weights. The attention implementation is almost certainly sdpa "
            "or flash; load_model pins attn_implementation='eager' for this "
            "reason, so this means the pin was dropped by the fallback path."
        )

    # Cast to float32 on the GPU (cheap) before the CPU transfer (expensive).
    # This avoids moving reduced-precision data across the PCIe bus then
    # converting. It does not recover precision already lost at load time —
    # see the module docstring on MODEL_DTYPE.
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
            dtype=_autocast_dtype(model),
            enabled=_autocast_enabled(model),
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