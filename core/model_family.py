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

# ---------------------------------------------------------------------------
# Checkpoint-name grammar  '{base}-step{N}'
# ---------------------------------------------------------------------------
#
# Added for Phase 2b's Pythia rerun. The grammar already existed, in
# `p1_mstate_tracking/visualization/checkpoints.py` — but that module imports
# matplotlib, so every analysis module that needed to know a run's checkpoint
# step either acquired a plotting dependency or re-typed the regex. Phase 2b
# was about to be the third copy.
#
# This is the same failure this module's header describes: two idioms that
# agree on registry keys and disagree elsewhere. Kept here rather than in
# core/naming.py because naming.py imports core.style, which imports
# matplotlib — the exact edge this move exists to avoid.
#
# `p1_mstate_tracking/visualization/checkpoints.py` should re-export these
# rather than keep its own `_STEP_RE`:
#
#     from core.model_family import (
#         checkpoint_step as _checkpoint_step,
#         checkpoint_base as _checkpoint_base,
#         checkpoint_families,
#     )
#
# Its `family_baselines` stays where it is: it is about which lines a figure
# draws, not about the name grammar.

import re as _re

#: 'pythia-410m-step2000' -> base 'pythia-410m', step 2000.
#: `pythia-1.4b-random` deliberately does NOT match — it is not a point on
#: the training trajectory and must never land on the step axis. See
#: core/pythia_registry.py.
CHECKPOINT_STEP_RE = _re.compile(r"^(?P<base>.+)-step(?P<step>\d+)$")


def checkpoint_step(model_name: str) -> Optional[int]:
    """'pythia-410m-step2000' -> 2000; None for non-checkpoint names."""
    m = CHECKPOINT_STEP_RE.match(str(model_name))
    return int(m.group("step")) if m else None


def checkpoint_base(model_name: str) -> Optional[str]:
    """'pythia-410m-step2000' -> 'pythia-410m'; None for non-checkpoint names."""
    m = CHECKPOINT_STEP_RE.match(str(model_name))
    return m.group("base") if m else None


def is_checkpoint(model_name: str) -> bool:
    return CHECKPOINT_STEP_RE.match(str(model_name)) is not None


def checkpoint_families(model_names) -> dict:
    """
    Group '-step{N}' variants by base model, ascending by step:

        {'pythia-410m': [(0, 'pythia-410m-step0'), (1, ...), ...]}

    Non-checkpoint names are dropped. Single-checkpoint families are kept —
    the caller decides its own minimum.
    """
    fams: dict = {}
    for m in model_names:
        step = checkpoint_step(m)
        if step is None:
            continue
        fams.setdefault(checkpoint_base(m), []).append((step, m))
    return {b: sorted(v) for b, v in sorted(fams.items())}


def sort_by_step(model_names) -> list:
    """Checkpoint names in training order. Non-checkpoint names sort last."""
    return sorted(
        model_names,
        key=lambda m: (checkpoint_step(m) is None, checkpoint_step(m) or 0, str(m)),
    )


__all__ += [
    "CHECKPOINT_STEP_RE",
    "checkpoint_step",
    "checkpoint_base",
    "is_checkpoint",
    "checkpoint_families",
    "sort_by_step",
]
