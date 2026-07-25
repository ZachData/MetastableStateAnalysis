"""
core/model_family.py — single source of truth for architecture detection.

Two idioms were previously in use and disagreed:

    p1_mstate_tracking/analysis_p1.py   startswith(("gpt2","pythia","gpt-neox","gptneox"))
    p1_mstate_tracking/plots.py         substring `in`, with .lower() on only one branch

They agree on registry keys ("pythia-410m-step0") and disagree on anything
else — notably the smoke checkpoint
"hf-internal-testing/tiny-random-GPTNeoXForCausalLM", which is causal but
starts with none of the prefixes. The failure was silent rather than loud:
analyze_attention_sinkhorn skipped causal-baseline subtraction, so every
per-head Fiedler value in that run was classified against the wrong
baseline, and analyze_value_eigenspectrum took the correct branch for the
same model in the same run. Nothing raised.

Deliberately imports nothing beyond stdlib — analysis_p1.py depends on this
and must not acquire a matplotlib or torch edge.
"""

from typing import Optional

__all__ = [
    "model_family",
    "is_causal_model",
    "is_albert",
    "is_bert",
    "is_gpt2",
    "is_gptneox",
    "FAMILIES",
]

FAMILIES = ("albert", "bert", "gpt2", "gptneox")

# Order matters twice over:
#   - "albert" must be tested before "bert", since "albert" contains "bert".
#   - "gptneox" must be tested before "gpt2" for the same reason in reverse
#     (a hypothetical "gpt2-neox" would otherwise resolve to gpt2).
_FAMILY_MARKERS = (
    ("albert",  ("albert",)),
    ("gptneox", ("pythia", "gpt-neox", "gptneox", "neox")),
    ("gpt2",    ("gpt2", "gpt-2")),
    ("bert",    ("bert",)),
)

_CAUSAL_FAMILIES = frozenset({"gpt2", "gptneox"})


def _canon(model_name: str) -> str:
    """Lowercase, with '_' folded to '-' so 'gpt_neox' and 'gpt-neox' agree.

    Variant suffixes run_1.py appends ('@attn', '@ffn', '@48iter') and repo
    prefixes ('EleutherAI/', 'hf-internal-testing/') are left in place —
    every check below is a substring test, so they are harmless.
    """
    return str(model_name).lower().replace("_", "-")


def model_family(model_name: str) -> Optional[str]:
    """Return one of FAMILIES, or None if the name matches no known family.

    None is a real answer, not an error: callers decide whether an
    unrecognised architecture is fatal (weight extraction) or merely means
    "no special handling" (causal-mask subtraction).
    """
    canon = _canon(model_name)
    for family, markers in _FAMILY_MARKERS:
        if any(m in canon for m in markers):
            return family
    return None


def is_causal_model(model_name: str) -> bool:
    """True for decoder-only (causally-masked) architectures.

    Unknown families are treated as non-causal, matching the previous
    default for BERT/ALBERT. If that is wrong for a model you have added,
    the fix is a marker in _FAMILY_MARKERS, not a special case at the call
    site — one more call-site idiom is how this module came to exist.
    """
    return model_family(model_name) in _CAUSAL_FAMILIES


def is_albert(model_name: str) -> bool:
    return model_family(model_name) == "albert"


def is_bert(model_name: str) -> bool:
    return model_family(model_name) == "bert"


def is_gpt2(model_name: str) -> bool:
    return model_family(model_name) == "gpt2"


def is_gptneox(model_name: str) -> bool:
    return model_family(model_name) == "gptneox"