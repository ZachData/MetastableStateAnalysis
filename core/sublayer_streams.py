"""
core/sublayer_streams.py — post-attention and post-FFN residual streams.

Replaces the hook logic that lived inside run_1._run_sublayer_analysis.
That version had three defects, of which the missing GPT-NeoX branch was
the mildest:

1. GPT-2 captured deltas, not streams. It hooked `block.attn` and
   `block.mlp` and labelled their outputs "@attn" / "@ffn". GPT2Block adds
   the residual *after* calling those submodules, so what was captured was
   the attention update and the MLP update — not x + update. Every metric
   downstream (inner-product distribution, effective rank, interaction
   energy, Fiedler) was therefore computed on the geometry of the update
   while being compared side by side with ALBERT runs that captured actual
   streams. Nothing raised; the arrays had the right shape.

2. ALBERT never ran at all. It looked for `layer.output`, which BertLayer
   has and AlbertLayer does not (ALBERT ends the block at
   `full_layer_layer_norm`). The AttributeError was swallowed by a bare
   try/except, leaving ffn_mods empty, which triggered the
   "Architecture not recognised" skip. The failure looked identical to an
   unsupported model.

3. Weight-shared layers were overwritten. Capture indexed into a
   preallocated list by module position, so ALBERT's single shared layer —
   called once per iteration — wrote all N captures into slot 0. Had (2)
   not skipped it first, the result would have been a one-layer
   "trajectory" holding only the final iteration.

Capture is now append-ordered, so a module called N times yields N entries
in call order. That makes weight-shared and distinct-layer architectures
the same case.

Stream semantics per family
---------------------------
post-LN (BERT, ALBERT): each sub-block ends in add-then-LayerNorm, so its
    output *is* the residual stream. Captured directly.

pre-LN sequential (GPT-2): x1 = x + attn(ln1(x));  x2 = x1 + mlp(ln2(x1)).
    Both streams are recoverable from the block input and the two deltas,
    and x2 equals the block output. The ordering confound is real here:
    the FFN stream is downstream of the attention stream.

pre-LN parallel (GPT-NeoX / Pythia, use_parallel_residual=True):
    out = x + attn(ln1(x)) + mlp(ln2(x)). Both branches read the same
    input, so post_attn = x + attn_delta and post_ffn = x + mlp_delta are
    symmetric and neither is downstream of the other. This is the
    decomposition design-1 flags as an upgrade over the GPT-2 case, and
    the reason the GPT-NeoX branch is worth having rather than aliasing to
    the GPT-2 one.

    Pythia sets use_parallel_residual=True at every scale, but the flag is
    read from the config rather than assumed — GPT-NeoX supports both.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from core.model_family import model_family


class UnsupportedArchitecture(RuntimeError):
    """No sublayer decomposition is defined for this model."""


@dataclass
class SublayerStreams:
    post_attn: List[torch.Tensor] = field(default_factory=list)
    post_ffn:  List[torch.Tensor] = field(default_factory=list)
    tokens:    List[str]          = field(default_factory=list)
    family:    Optional[str]      = None
    # "post-ln" | "pre-ln-sequential" | "pre-ln-parallel"
    semantics: str                = ""
    parallel_residual: Optional[bool] = None
    n_layers: int = 0

    def meta_overrides(self) -> dict:
        """Fields describe_extraction cannot infer for a hook capture."""
        return {
            "hidden_state_0_is_embedding":   False,
            "final_hidden_state_is_post_ln": self.semantics == "post-ln",
            "sublayer_semantics":            self.semantics,
            "parallel_residual":             self.parallel_residual,
        }


# ---------------------------------------------------------------------------
# Block discovery
# ---------------------------------------------------------------------------

def _blocks(model, family):
    """Return the per-layer block modules for *family*, or None."""
    if family == "gpt2":
        base = getattr(model, "transformer", model)
        return list(base.h) if hasattr(base, "h") else None

    if family == "gptneox":
        base = getattr(model, "gpt_neox", model)
        return list(base.layers) if hasattr(base, "layers") else None

    if family == "bert":
        enc = getattr(model, "encoder", None)
        return list(enc.layer) if enc is not None and hasattr(enc, "layer") else None

    if family == "albert":
        enc = getattr(model, "encoder", None)
        if enc is None or not hasattr(enc, "albert_layer_groups"):
            return None
        # One entry per distinct module. ALBERT shares it across all
        # iterations; append-ordered capture recovers the full trajectory.
        blocks = []
        for group in enc.albert_layer_groups:
            blocks.extend(group.albert_layers)
        return blocks

    return None


def _attn_module(block, family):
    if family == "gpt2":
        return block.attn
    if family == "gptneox":
        return getattr(block, "attention", None) or getattr(block, "attn", None)
    return block.attention          # bert, albert


def _ffn_module(block, family):
    if family in ("gpt2", "gptneox"):
        return block.mlp
    if family == "bert":
        return block.output
    if family == "albert":
        # AlbertLayer has no .output; the block ends at this LayerNorm,
        # which is applied to (ffn_output + attention_output).
        return block.full_layer_layer_norm
    return None


def _uses_parallel_residual(model, blocks) -> bool:
    cfg = getattr(model, "config", None)
    value = getattr(cfg, "use_parallel_residual", None) if cfg else None
    if value is None and blocks:
        value = getattr(blocks[0], "use_parallel_residual", None)
    # GPT-NeoX's own default is True; say so explicitly rather than
    # silently assuming it when neither source answers.
    return True if value is None else bool(value)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def _tensor_of(out):
    t = out[0] if isinstance(out, (tuple, list)) else out
    return t.detach().to(torch.float32).cpu().squeeze(0)


def _append_hook(sink):
    def hook(_module, _inp, out):
        sink.append(_tensor_of(out))
    return hook


def _append_pre_hook(sink):
    def hook(_module, inp):
        sink.append(_tensor_of(inp))
    return hook


def extract_sublayer_streams(
    model, tokenizer, text: str, model_name: str, max_length: int = 512,
) -> SublayerStreams:
    """Run one forward pass and return both sublayer residual streams.

    Raises UnsupportedArchitecture when no decomposition is defined —
    deliberately, rather than returning empty lists. The previous version
    returned empty and printed a note, which made "this architecture has no
    branch" indistinguishable from "the branch exists and the hooks
    misfired".
    """
    family = model_family(model_name)
    blocks = _blocks(model, family)
    if not blocks:
        raise UnsupportedArchitecture(
            f"{model_name}: no sublayer decomposition for family {family!r}. "
            f"Add a branch to core/sublayer_streams.py — the streams are "
            f"architecture-specific and cannot be guessed from module names."
        )

    post_ln  = family in ("bert", "albert")
    parallel = None if post_ln else _uses_parallel_residual(model, blocks)

    x_ins, attn_out, ffn_out = [], [], []
    handles = []

    try:
        for block in blocks:
            attn_mod = _attn_module(block, family)
            ffn_mod  = _ffn_module(block, family)
            if attn_mod is None or ffn_mod is None:
                raise UnsupportedArchitecture(
                    f"{model_name}: block {type(block).__name__} is missing an "
                    f"attention or FFN submodule for family {family!r}."
                )
            if not post_ln:
                handles.append(block.register_forward_pre_hook(_append_pre_hook(x_ins)))
            handles.append(attn_mod.register_forward_hook(_append_hook(attn_out)))
            handles.append(ffn_mod.register_forward_hook(_append_hook(ffn_out)))

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    if len(attn_out) != len(ffn_out):
        raise RuntimeError(
            f"{model_name}: captured {len(attn_out)} attention and "
            f"{len(ffn_out)} FFN outputs. The hooks fired an unequal number "
            f"of times, so the two streams cannot be aligned layer-for-layer."
        )
    n = len(attn_out)
    if n == 0:
        raise RuntimeError(f"{model_name}: no sublayer hooks fired.")

    if post_ln:
        # Each submodule output is already the residual stream.
        post_attn, post_ffn, semantics = attn_out, ffn_out, "post-ln"
    else:
        if len(x_ins) != n:
            raise RuntimeError(
                f"{model_name}: captured {len(x_ins)} block inputs for {n} "
                f"sublayer outputs. Streams cannot be reconstructed."
            )
        post_attn = [x + a for x, a in zip(x_ins, attn_out)]
        if parallel:
            # Both branches read the same block input — symmetric, no
            # ordering confound.
            post_ffn  = [x + f for x, f in zip(x_ins, ffn_out)]
            semantics = "pre-ln-parallel"
        else:
            # FFN branch reads the post-attention stream.
            post_ffn  = [p + f for p, f in zip(post_attn, ffn_out)]
            semantics = "pre-ln-sequential"

    return SublayerStreams(
        post_attn=post_attn, post_ffn=post_ffn, tokens=tokens,
        family=family, semantics=semantics,
        parallel_residual=parallel, n_layers=n,
    )