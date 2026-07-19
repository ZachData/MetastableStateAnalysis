"""
core/lm_loading.py — Registry-consistent *ForCausalLM loading for causal
work (Group D, Phase 5b readout, Phase 6 dissociation loss/KL).

Why this exists
---------------
Every entry in core/config.py's MODEL_CONFIGS and core/pythia_registry.py
loads the bare model class (GPT2Model, GPTNeoXModel, AlbertModel,
BertModel). None produce `.logits`, so core/intervention.py's
run_model_with_hook returns logits=None / loss=None for everything
loadable through the registry — a real, load-bearing gap between what
design-5c.md's Group D readout needs ("next-token cross-entropy delta and
KL divergence") and what the registry loads. The smoke test for the
intervention runner previewed the workaround (AutoModelForCausalLM
directly, bypassing the registry); this module is the real version: same
registry keys, same repo id + pinned HF revision as the bare load, so
"the LM-head model at pythia-1.4b-step1000" is provably the same
checkpoint the extraction pipeline analyzed, not a second, unpinned load.

Deliberately separate from core/models.py's load_model (per status-5.md /
the item-3 changelog: "separately from whatever the main extraction
pipeline uses"). The extraction pipeline keeps bare-model semantics —
its hidden_states/attentions contract, its bfloat16 policy, its
random-init handling — untouched. Causal readouts opt in to this loader
explicitly.

Two hard scope rules:

1. Causal-LM heads only. Masked-LM models (BERT/ALBERT) are refused, not
   silently loaded with the wrong head: run_model_with_hook's loss is the
   SHIFTED next-token cross-entropy (causal convention), documented there
   as *not* the right loss for a masked LM. Loading a ForMaskedLM here
   would make it one keystroke to compute a wrong-by-construction number.
   Per the plan's scope decision, no forward-going causal work targets
   BERT/ALBERT anyway.

2. No randomization here. `random_init` entries (gpt2-large-random,
   pythia-1.4b-random) are refused: randomize_weights lives in the
   extraction path (core/models.py / pythia_weights.py) and re-applying
   it here would create a *second* random model whose weights differ from
   the one the geometric results describe (different module tree =>
   different RNG consumption order, even at the same seed). Group D's
   random-twin readout must reuse the exact randomized weights the
   extraction produced — see load_causal_lm_from_state_dict below for
   that path.

torch/transformers are deferred inside functions (matching
core/metrics.py / core/intervention.py's pattern) so the module imports
cleanly in a torch-free environment and the pure-logic pieces
(resolve_lm_entry) are testable there.
"""

from __future__ import annotations

from typing import Optional, Tuple

# Architectures whose ForCausalLM variant is the correct LM head for the
# shifted next-token loss run_model_with_hook computes. Checked against
# model_class.__name__ so no transformers import is needed to resolve.
_CAUSAL_MODEL_CLASS_NAMES = ("GPT2Model", "GPTNeoXModel")
_MASKED_MODEL_CLASS_NAMES = ("BertModel", "AlbertModel")


def resolve_lm_entry(model_name: str, model_configs: Optional[dict] = None) -> dict:
    """
    Pure-logic half of load_causal_lm: resolve a registry key to the
    (repo_id, revision) pair a ForCausalLM load must use, refusing
    entries this loader must not serve.

    Parameters
    ----------
    model_name    : key into MODEL_CONFIGS (optionally merged with
                    build_pythia_model_configs() output by the caller).
    model_configs : the registry dict. Default: core.config.MODEL_CONFIGS
                    merged with core.pythia_registry.build_pythia_model_configs().
                    Injectable for tests (this is what makes the function
                    testable without transformers installed).

    Returns
    -------
    dict: {repo_id, revision, checkpoint_step (or None), model_name}

    Raises
    ------
    KeyError   : unknown model_name (valid keys listed, matching
                 core.artifacts.get_spec's error convention).
    ValueError : masked-LM architecture, or random_init entry — see
                 module docstring for why each is refused rather than
                 accommodated.
    """
    if model_configs is None:
        from core.config import MODEL_CONFIGS
        merged = dict(MODEL_CONFIGS)
        try:
            from core.pythia_registry import build_pythia_model_configs
            merged.update(build_pythia_model_configs())
        except Exception:
            pass
        model_configs = merged

    try:
        cfg = model_configs[model_name]
    except KeyError:
        raise KeyError(
            f"Unknown model {model_name!r}. Known models: "
            f"{sorted(model_configs)[:20]}{' ...' if len(model_configs) > 20 else ''}"
        )

    cls_name = getattr(cfg.get("model_class"), "__name__", str(cfg.get("model_class")))

    if cls_name in _MASKED_MODEL_CLASS_NAMES:
        raise ValueError(
            f"{model_name!r} is a masked-LM architecture ({cls_name}). "
            "load_causal_lm computes/serves the shifted next-token loss "
            "convention only (see core/intervention.py's masked-LM caveat); "
            "loading a ForMaskedLM here would silently produce the wrong "
            "loss. No forward-going causal work targets BERT/ALBERT (plan "
            "scope decision)."
        )
    if cls_name not in _CAUSAL_MODEL_CLASS_NAMES:
        raise ValueError(
            f"{model_name!r} has unrecognized model_class {cls_name!r}; "
            f"causal-LM loading is defined for {_CAUSAL_MODEL_CLASS_NAMES}."
        )

    if cfg.get("random_init"):
        raise ValueError(
            f"{model_name!r} is a random_init entry. Re-randomizing inside "
            "the LM-head loader would produce different weights than the "
            "extraction pipeline's randomized model (different module tree "
            "=> different RNG consumption at the same seed). Use "
            "load_causal_lm_from_state_dict with the extraction model's "
            "state_dict instead, so the causal readout runs on the exact "
            "weights the geometric results describe."
        )

    return {
        "repo_id":         cfg.get("hf_repo", cfg.get("pretrained_name", model_name)),
        "revision":        cfg.get("revision"),
        "checkpoint_step": cfg.get("checkpoint_step"),
        "model_name":      model_name,
    }


def load_causal_lm(
    model_name: str,
    device: Optional[str] = None,
    model_configs: Optional[dict] = None,
) -> Tuple[object, object]:
    """
    Load the *ForCausalLM* variant of a registry entry, same repo id and
    pinned revision as the bare load, in eval mode on `device`.

    dtype policy mirrors core/models.py's load_model (bfloat16 on cuda,
    float32 on cpu). Tokenizer comes from AutoTokenizer at the same
    revision; gpt2-family pad_token fallback matches load_model's.

    Returns
    -------
    (model, tokenizer) — model has `.logits` output, so
    run_model_with_hook's logits/loss/next_token_kl paths are live.
    """
    entry = resolve_lm_entry(model_name, model_configs=model_configs)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        from core.config import DEVICE as device  # noqa: F811

    tokenizer = AutoTokenizer.from_pretrained(
        entry["repo_id"], revision=entry["revision"]
    )
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        entry["repo_id"],
        revision=entry["revision"],
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_causal_lm_from_state_dict(
    model_name_for_arch: str,
    bare_state_dict: dict,
    device: Optional[str] = None,
    model_configs: Optional[dict] = None,
) -> Tuple[object, object]:
    """
    The random-twin path: build the ForCausalLM architecture for a
    registry entry, then overwrite its *transformer body* with the
    supplied bare-model state_dict (e.g. the norm-matched-randomized
    weights the extraction pipeline actually analyzed), leaving the LM
    head at its pretrained values.

    Head caveat, stated rather than hidden: for untied-embedding models
    (Pythia's embed_out) the head weights are NOT part of the bare
    state_dict, so they stay *trained* while the body is random. For a
    Group D trained-vs-random KL/loss contrast this is the correct
    control — both arms decode through the identical head, so every
    difference in the readout is attributable to the body. It is NOT "a
    fully random causal LM"; don't report it as one.

    Key mapping: HF's ForCausalLM wrappers prefix the body — GPT2LMHeadModel
    stores the body under `transformer.*`, GPTNeoXForCausalLM under
    `gpt_neox.*`. Bare-model keys are remapped with the detected prefix;
    unmatched keys raise rather than silently load-partial.
    """
    # resolve_lm_entry refuses random_init entries; for this path the
    # caller passes the *base* (non-random) entry name for architecture,
    # so resolution goes through unchanged.
    entry = resolve_lm_entry(model_name_for_arch, model_configs=model_configs)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        from core.config import DEVICE as device  # noqa: F811

    tokenizer = AutoTokenizer.from_pretrained(
        entry["repo_id"], revision=entry["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        entry["repo_id"],
        revision=entry["revision"],
        torch_dtype=torch.float32,   # state_dict override → keep full precision
    ).to(device)

    body_prefix = None
    for prefix in ("transformer", "gpt_neox"):
        if hasattr(model, prefix):
            body_prefix = prefix
            break
    if body_prefix is None:
        raise ValueError(
            f"Could not locate the transformer body on {type(model).__name__}; "
            "expected a .transformer (GPT-2) or .gpt_neox (GPT-NeoX) attribute."
        )

    remapped = {f"{body_prefix}.{k}": v for k, v in bare_state_dict.items()}
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # Missing keys are expected (the head + anything not in the bare body);
    # *unexpected* keys mean the bare dict didn't map onto the body — a
    # real mismatch, not a benign partial load.
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} bare-model keys did not map onto "
            f"{body_prefix}.*: {sorted(unexpected)[:5]} ... "
            "Check that the state_dict came from the matching bare class."
        )

    model.eval()
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
