"""
core/naming.py (moved from p1_mstate_tracking/visualization/naming.py,
unmodified — transition plan v2, core foundations item 2)

Model-name conventions: color lookup, the '-random' / '@attn' / '@ffn' /
'@Niter' suffix grammar, and the ALBERT iteration-depth filter. No file
I/O — pure string/dict logic on the model-variant string as it appears in
geometry.json.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .style import MODEL_COLORS, UNTRAINED_COLOR, RANDOM_COLOR_OVERRIDES, ENERGY_BETA_COLORS

# ─────────────────────────────────────────────────────────────────────────────
# Model-name helpers
# ─────────────────────────────────────────────────────────────────────────────

def _color(model: str) -> str:
    mn = model.replace("-", "_").lower()
    for k, c in MODEL_COLORS.items():
        if k.replace("-", "_") in mn:
            return c
    return "#374151"

def _random_color(model: str) -> str:
    """
    Color for a randomly-initialized control line. Every random control
    defaults to UNTRAINED_COLOR (gray) so the family reads as one thing at
    a glance; only the entries in RANDOM_COLOR_OVERRIDES get a distinct
    color, for when two random controls land on the same chart and need
    to be told apart.
    """
    mn = model.replace("-", "_").lower()
    for k, c in RANDOM_COLOR_OVERRIDES.items():
        if k.replace("-", "_") in mn:
            return c
    return UNTRAINED_COLOR

def _is_untrained(model: str) -> bool:
    return "random" in model.lower()

def _beta_color(beta: float) -> str:
    """Color for one β's energy line — see ENERGY_BETA_COLORS."""
    for key in (str(beta), f"{beta:g}", f"{beta:.1f}"):
        if key in ENERGY_BETA_COLORS:
            return ENERGY_BETA_COLORS[key]
    return "#374151"

def _is_sublayer_variant(model: str) -> Optional[Tuple[str, str]]:
    """
    (base_model, 'attn'|'ffn') if `model` is a post-sublayer residual
    stream saved by --sublayer (run_1.py names these '{model}@attn' /
    '{model}@ffn'), else None. Deliberately checks only these two literal
    suffixes — ALBERT's '@Niter' iteration-depth suffix must not match.
    """
    if model.endswith("@attn"):
        return model[: -len("@attn")], "attn"
    if model.endswith("@ffn"):
        return model[: -len("@ffn")], "ffn"
    return None

def _sublayer_groups(models: List[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    For every base model with at least one of its @attn / @ffn streams
    present in `models` (and the full-block run itself present), return
    (base_model, attn_variant_or_None, ffn_variant_or_None).
    """
    sub_map: Dict[str, Dict[str, str]] = {}
    for m in models:
        parsed = _is_sublayer_variant(m)
        if parsed:
            base, kind = parsed
            sub_map.setdefault(base, {})[kind] = m
    groups = []
    for base, d in sorted(sub_map.items()):
        if base in models:
            groups.append((base, d.get("attn"), d.get("ffn")))
    return groups

def _iter_depth(model: str) -> Optional[int]:
    if "@" in model and "iter" in model:
        try:
            return int(model.split("@")[1].replace("iter", ""))
        except ValueError:
            pass
    return None

# ALBERT checkpoints were swept at several iteration depths (12/24/36/48).
# Keeping all of them makes the output noisy, so only one depth survives
# per architecture below — this never adds runs, only drops them. Applies
# to both the trained and "-random" control of each architecture (matched
# by prefix, so "albert-base-v2-random@12iter" is dropped too).
ITERATION_KEEP: Dict[str, int] = {
    "albert-base-v2":   24,
    "albert-xlarge-v2": 48,
}

def _passes_iteration_filter(model: str) -> bool:
    depth = _iter_depth(model)
    if depth is None:
        return True  # no iteration suffix on this model — nothing to filter
    arch = model.split("@")[0]
    for prefix, keep_depth in ITERATION_KEEP.items():
        if arch.startswith(prefix):
            return depth == keep_depth
    return True  # not one of the architectures this filter governs

def filter_iteration_depths(
    runs: Dict[Tuple[str, str], Path],
) -> Dict[Tuple[str, str], Path]:
    """Drop (model, prompt) runs whose ALBERT iteration depth isn't in ITERATION_KEEP."""
    return {(m, p): rd for (m, p), rd in runs.items() if _passes_iteration_filter(m)}

def _label(model: str) -> str:
    """Display label for legends — model is already the full variant string."""
    return model

def _safe_model_name(model: str) -> str:
    """
    Filesystem-safe identifier that preserves every distinguishing part of
    a model variant (iteration depth, random/untrained suffix) — each
    variant is treated as fully independent, so this never collapses two
    different variants onto the same name.
        'albert-base-v2@12iter'  -> 'albert-base-v2_12iter'
        'gpt2-large-random'      -> 'gpt2-large-random'
    """
    return model.replace("/", "_").replace(".", "_").replace("@", "_")
