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
    Extract the attention/FFN split of each residual update, by hooks.

    Dispatch is on `core.model_family.model_family`, and the block and
    submodule lookups come from `core.sublayer_streams`. Previously this
    function branched on substrings of the model name — `"bert" in
    model_name`, `"gpt2" in model_name` — and a GPT-NeoX name matched
    neither. The consequence was silent: no hooks were registered, the
    forward pass ran, and the function returned a well-formed dict whose
    `attn_deltas` and `ffn_deltas` were empty lists. `save_decomposed`
    skips empty arrays without complaint, so on a Pythia run
    `decomposed_violations` reported channel "mixed" over zero violations,
    `ffn_subspace` found no `ffn_deltas.npz` and was marked inapplicable,
    `attractive_zone_violations` was skipped with it, and the v-score's
    0.20 FFN term was identically zero — capping every Pythia v-score at
    0.80 while looking like a real measurement.

    Residual semantics per family, which is what makes the deltas mean
    different things even though the capture looks identical:

      pre-LN sequential (GPT-2)
          x1 = x + attn(ln1(x));  x2 = x1 + mlp(ln2(x1))
          Hooking the submodules captures the two deltas directly. The FFN
          delta is computed from the POST-ATTENTION stream, so the two
          channels are not symmetric and an FFN attribution is partly
          conditional on what attention just did.

      pre-LN parallel (GPT-NeoX / Pythia)
          out = x + attn(ln1(x)) + mlp(ln2(x))
          Both branches read the same block input. The attribution is
          symmetric and neither channel is downstream of the other, which
          makes Pythia a cleaner test of the attn-vs-FFN question than
          GPT-2 was. The flag is read from the config rather than assumed:
          GPT-NeoX supports both modes.

      post-LN (BERT)
          Each sub-block ends in add-then-LayerNorm, so its output is the
          stream, not the delta. Deltas are differences of streams, as
          before.

    Returns
    -------
    dict with:
      trajectory   : list of (n_tokens, d) tensors — hidden states per layer
      attn_deltas  : list of (n_tokens, d) tensors — attention contribution
      ffn_deltas   : list of (n_tokens, d) tensors — FFN contribution
      attentions   : list of (n_heads, seq, seq) tensors
      tokens       : list of str
      semantics    : "pre-ln-parallel" | "pre-ln-sequential" | "post-ln"
      parallel_residual : bool or None (None for post-LN)
      residual_identity : {"max_abs_err", "rel_err", "checked"} — see below

    Raises
    ------
    UnsupportedArchitecture
        When no branch exists, instead of returning empty deltas. An
        architecture with no decomposition and a decomposition whose hooks
        misfired should not produce the same artifact.
    """
    from core.model_family import model_family
    from core.sublayer_streams import (
        blocks_of, attn_module, ffn_module, uses_parallel_residual,
        UnsupportedArchitecture,
    )

    family = model_family(model_name)

    # Checked before block discovery: ALBERT HAS discoverable blocks, so a
    # discovery-first order would only produce this message for a
    # malformed ALBERT and give the generic "no branch" text for a
    # well-formed one. The actionable message should not depend on whether
    # the model is intact.
    if family == "albert":
        raise UnsupportedArchitecture(
            f"{model_name}: use extract_decomposed_albert, which unrolls the "
            f"shared layer across iterations."
        )

    blocks = blocks_of(model, family)
    if not blocks:
        raise UnsupportedArchitecture(
            f"{model_name}: no attn/FFN decomposition for family {family!r}. "
            f"Add a branch to core/sublayer_streams.blocks_of — the residual "
            f"structure is architecture-specific and cannot be guessed from "
            f"module names."
        )

    post_ln  = family == "bert"
    parallel = None if post_ln else uses_parallel_residual(model, blocks)

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    attn_outputs, ffn_outputs, hooks = [], [], []

    def _capture(sink):
        def hook(_module, _inp, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            sink.append(t.detach()[0].to(torch.float32).cpu())
        return hook

    try:
        for block in blocks:
            a_mod = attn_module(block, family)
            f_mod = ffn_module(block, family)
            if family == "bert":
                # The delta-bearing modules are the post-LN wrappers, not
                # the attention/MLP submodules themselves.
                a_mod = block.attention.output
                f_mod = block.output
            if a_mod is None or f_mod is None:
                raise UnsupportedArchitecture(
                    f"{model_name}: block {type(block).__name__} is missing an "
                    f"attention or FFN submodule for family {family!r}."
                )
            hooks.append(a_mod.register_forward_hook(_capture(attn_outputs)))
            hooks.append(f_mod.register_forward_hook(_capture(ffn_outputs)))

        # output_attentions=True so outputs.attentions carries the per-layer
        # (batch, n_heads, seq, seq) tensors the dynamic head test needs.
        with torch.no_grad():
            with torch.autocast(
                device_type=DEVICE,
                dtype=torch.bfloat16,
                enabled=(DEVICE == "cuda"),
            ):
                outputs = model(**inputs,
                                output_hidden_states=True,
                                output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    hidden_states = [h[0].to(torch.float32).cpu() for h in outputs.hidden_states]

    attn_matrices = []
    if getattr(outputs, "attentions", None) is not None:
        for a in outputs.attentions:
            attn_matrices.append(a[0].to(torch.float32).cpu())

    if len(attn_outputs) != len(ffn_outputs):
        raise RuntimeError(
            f"{model_name}: captured {len(attn_outputs)} attention and "
            f"{len(ffn_outputs)} FFN outputs. The hooks fired an unequal "
            f"number of times, so the channels cannot be aligned."
        )
    if not attn_outputs:
        raise RuntimeError(f"{model_name}: no decomposition hooks fired.")

    if post_ln:
        # Hooks captured post-LN streams; the deltas are their differences.
        attn_deltas, ffn_deltas = [], []
        for i in range(len(attn_outputs)):
            if i < len(hidden_states) - 1:
                attn_deltas.append(attn_outputs[i] - hidden_states[i])
                ffn_deltas.append(hidden_states[i + 1] - attn_outputs[i])
        semantics = "post-ln"
    else:
        # Pre-LN: the submodule outputs ARE the residual deltas, in both the
        # sequential and the parallel arrangement. What differs is which
        # stream the FFN branch read, not what was captured.
        attn_deltas = attn_outputs
        ffn_deltas  = ffn_outputs
        semantics = "pre-ln-parallel" if parallel else "pre-ln-sequential"

    return {
        "trajectory":  hidden_states,
        "attn_deltas": attn_deltas,
        "ffn_deltas":  ffn_deltas,
        "attentions":  attn_matrices,
        "tokens":      tokens,
        "semantics":   semantics,
        "parallel_residual": parallel,
        "residual_identity": _residual_identity(
            hidden_states, attn_deltas, ffn_deltas
        ),
    }


def _residual_identity(trajectory, attn_deltas, ffn_deltas) -> dict:
    """
    Check x_{L+1} - x_L == attn_delta_L + ffn_delta_L.

    Free, and it validates the whole capture end to end: if the hooks are
    on the wrong modules, or one family's outputs are streams where another's
    are deltas, this identity breaks even though every array still has the
    right shape. That is exactly the failure mode that made the old
    substring dispatch invisible.

    Reported, not asserted. Under bf16 autocast the deltas come back through
    a lower-precision path than the hidden states, so a relative error around
    1e-2 is expected on GPU and means nothing is wrong; an error near 1.0
    means the capture is wrong. Post-LN models are exempt — LayerNorm sits
    between the deltas and the stream, so no such identity exists there.
    """
    import numpy as _np

    n = min(len(attn_deltas), len(ffn_deltas), len(trajectory) - 1)
    if n <= 0:
        return {"checked": 0, "max_abs_err": None, "rel_err": None}

    max_abs, max_rel = 0.0, 0.0
    for t in range(n):
        actual = (trajectory[t + 1] - trajectory[t]).numpy()
        predicted = (attn_deltas[t] + ffn_deltas[t]).numpy()
        err = float(_np.abs(actual - predicted).max())
        scale = float(_np.abs(actual).max()) + 1e-12
        max_abs = max(max_abs, err)
        max_rel = max(max_rel, err / scale)
    return {"checked": n, "max_abs_err": max_abs, "rel_err": max_rel}


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

    Attribution fractions
    ---------------------
    The previous implementation used abs(delta_attn) / (abs(delta_attn) +
    abs(delta_ffn)), which is insensitive to sign.  When one component drops
    energy and the other raises it — which is common at violation layers where
    attention and FFN partially oppose each other — both components received
    positive fractions summing to 1, masking the opposition.

    The corrected fractions measure each component's contribution to the
    *realised* energy drop:

      attn_frac = max(0, -delta_attn) / max(|delta_total|, ε)
      ffn_frac  = max(0, -delta_ffn)  / max(|delta_total|, ε)

    Interpretation:
      - attn_frac > 1 means attention drops more than the total realised drop
        (FFN is actively opposing it).
      - attn_frac = 0 means attention raised energy (not a contributor to the
        violation).
      - attn_frac + ffn_frac can exceed 1 when the components partially cancel.

    The boolean flags attn_opposes and ffn_opposes mark when a component is
    working against the violation direction (delta > 0 while total delta < 0).

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
      delta_cross                                 : float
      attn_frac, ffn_frac                         : float (signed, see above)
      attn_opposes, ffn_opposes                   : bool
      attn_sign, ffn_sign, cross_sign             : str ("drop" | "rise")
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

    delta_total = e_after     - e_before
    delta_attn  = e_attn_only - e_before
    delta_ffn   = e_ffn_only  - e_before
    delta_cross = delta_total - delta_attn - delta_ffn

    # Signed attribution: fraction of the realised drop each component explains.
    # max(0, -delta) picks out only the energy-decreasing (violation) direction.
    denom = max(abs(delta_total), 1e-12)
    attn_frac = max(0.0, -delta_attn) / denom
    ffn_frac  = max(0.0, -delta_ffn)  / denom

    # Opposition flags: component is raising energy while total is dropping.
    is_violation = delta_total < -1e-6
    attn_opposes = bool(is_violation and delta_attn > 1e-6)
    ffn_opposes  = bool(is_violation and delta_ffn  > 1e-6)

    return {
        "e_before":    e_before,
        "e_after":     e_after,
        "e_attn_only": e_attn_only,
        "e_ffn_only":  e_ffn_only,
        "delta_total": delta_total,
        "delta_attn":  delta_attn,
        "delta_ffn":   delta_ffn,
        "delta_cross": delta_cross,
        "attn_frac":   attn_frac,
        "ffn_frac":    ffn_frac,
        "attn_opposes": attn_opposes,
        "ffn_opposes":  ffn_opposes,
        # Sign: negative = energy-decreasing (violation direction)
        "attn_sign":   "drop" if delta_attn  < -1e-6 else "rise",
        "ffn_sign":    "drop" if delta_ffn   < -1e-6 else "rise",
        "cross_sign":  "drop" if delta_cross < -1e-6 else "rise",
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
    Save attn and FFN deltas for Phase 3 crosscoder training and Phase 2
    FFN subspace analysis.

    Two files are written per component:

      {attn,ffn}_deltas_raw.npz
        (n_layers, n_tokens, d) float32 — unnormalised residual-stream deltas.
        Use these for energy magnitude analysis (ffn_total_energy, cross-term
        analysis) where scale carries information.

      {attn,ffn}_deltas_normed.npz
        (n_layers, n_tokens, d) float32 — per-token-vector L2-normalised.
        Use these when only direction matters (subspace projection fractions,
        cosine-based similarity).  Kept for back-compat with any code that
        already loads the old single normed file.

    The previous implementation only saved normed deltas, causing
    ffn_total_energy z-scores in ffn_subspace.py to be meaningless (all
    norms ≈ 1.0 after normalisation).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def _stack(deltas):
        """Stack a list of (n_tokens, d) tensors/arrays into (n_layers, n_tokens, d)."""
        if not deltas:
            return np.array([])
        return np.stack([
            d.numpy() if hasattr(d, "numpy") else d for d in deltas
        ]).astype(np.float32)

    def _l2_norm_rows(arr):
        """Per-token-vector L2 normalisation: each (d,) row becomes unit length."""
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / np.maximum(norms, 1e-10)

    for name, deltas in [("attn", decomposed["attn_deltas"]),
                          ("ffn",  decomposed["ffn_deltas"])]:
        raw = _stack(deltas)
        if raw.size == 0:
            continue

        # Raw: preserve scale for energy analysis.
        np.savez_compressed(run_dir / f"{name}_deltas_raw.npz",
                            **{f"{name}_deltas": raw})

        # Normed: direction-only for subspace projection.
        normed = _l2_norm_rows(raw)
        np.savez_compressed(run_dir / f"{name}_deltas_normed.npz",
                            **{f"{name}_deltas": normed})

    print(f"    Saved decomposed deltas (raw + normed) to {run_dir}/")
