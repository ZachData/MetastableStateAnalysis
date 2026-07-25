"""
checkpoint_frames.py — Binding frames to checkpoints (frames item 11).

Two jobs, both about the sweep.

**1. A frame card belongs to exactly one checkpoint.**

LayerNorm gamma/beta, and therefore every LN-frame quantity, change during
training. A card built from the final model and applied to a step-1000
checkpoint produces numbers that look entirely reasonable and describe
nothing. Extraction was never the risk here — caching is. Anything holding
LN parameters, projectors, or QK matrices across a family must key on the
revision, and the binding must be *checked* at consumption rather than
assumed at construction.

`CheckpointFrameCache` therefore refuses a hit whose revision does not match
the request, rather than returning the entry it has.

**2. Transitions must be located in log-step, not checkpoint index.**

Pythia's checkpoints are log-spaced to step 512 and then every 1000. A
derivative taken over checkpoint *index* places its largest values wherever
the spacing changes, which is an artifact of the release schedule rather
than of training. `checkpoints._step_x` already uses log10(step + 1) for
plotting; this module supplies the matching derivative and transition
detector so the analysis agrees with the axis it is drawn on.

Note on policy P4
-----------------
Checked during implementation and **not** an issue.
`checkpoints._STEP_RE` is `^(?P<base>.+)-step(\\d+)$`, so
`pythia-1.4b-deduped-step1000` yields base `pythia-1.4b-deduped`, distinct
from `pythia-1.4b`. Deduped and non-deduped families cannot silently merge.
The guard added here is the revision string carrying the base name, so the
distinction also survives into every result record.

See DESIGN_pythia_frames.md item 11 and policies P3, P4.
"""

from __future__ import annotations

import re

import numpy as np

from core.frame_card import FrameCardError, verify_card_for_run


_STEP_RE = re.compile(r"^(?P<base>.+)-step(?P<step>\d+)$")

#: Metrics cheap enough to run at every checkpoint. `polar` belongs here
#: because norm outliers and attention sinks EMERGE during training — sphere
#: validity is a per-checkpoint question, not a property of the final model.
CHEAP_TIER_METRICS = (
    "polar_layer_record",
    "sphere_gap",
    "norm_stats",
    "effective_rank",
    "centroid_norm",
)


# ---------------------------------------------------------------------------
# Revision identity
# ---------------------------------------------------------------------------

def parse_checkpoint(model_name: str) -> dict:
    """
    Split a checkpoint model name into base and step.

    'pythia-1.4b-step1000' -> base 'pythia-1.4b', step 1000
    'pythia-1.4b-deduped-step1000' -> base 'pythia-1.4b-deduped', step 1000
    'pythia-1.4b' -> base 'pythia-1.4b', step None
    """
    m = _STEP_RE.match(model_name)
    if not m:
        return {"base": model_name, "step": None, "is_checkpoint": False}
    return {"base": m.group("base"), "step": int(m.group("step")),
            "is_checkpoint": True}


def checkpoint_revision(model_name: str) -> str:
    """
    Canonical revision string for a checkpoint, matching the HF revision tag.

    'pythia-1.4b-step1000' -> 'step1000'; a non-checkpoint name -> 'main'.
    Used so `FrameSpec.model_rev` and the frame card agree everywhere, which
    is what makes `verify_same_revision` able to say anything.
    """
    p = parse_checkpoint(model_name)
    return f"step{p['step']}" if p["is_checkpoint"] else "main"


def revision_key(model_name: str) -> str:
    """`base@revision` — the key the cache and the ledger both use."""
    p = parse_checkpoint(model_name)
    return f"{p['base']}@{checkpoint_revision(model_name)}"


# ---------------------------------------------------------------------------
# Cache that refuses to serve the wrong checkpoint
# ---------------------------------------------------------------------------

class CheckpointFrameCache:
    """
    Frame cards keyed by revision, with a hard refusal on mismatch.

    A conventional cache returns whatever it holds. That is the exact
    behaviour that would apply the final model's LayerNorm parameters to a
    step-1000 checkpoint, so `get` raises on a base-name mismatch rather than
    serving a plausible wrong answer.
    """

    def __init__(self):
        self._cards = {}

    def put(self, model_name: str, card, store) -> str:
        key = revision_key(model_name)
        expected = parse_checkpoint(model_name)["base"]
        if card.model_name != expected:
            raise FrameCardError(
                f"CheckpointFrameCache.put: card is for {card.model_name!r} "
                f"but the model name implies base {expected!r}"
            )
        want_rev = checkpoint_revision(model_name)
        if card.revision != want_rev:
            raise FrameCardError(
                f"CheckpointFrameCache.put: card revision {card.revision!r} "
                f"!= {want_rev!r} implied by {model_name!r}. LN parameters "
                f"must come from the checkpoint being analysed."
            )
        self._cards[key] = (card, store)
        return key

    def get(self, model_name: str):
        key = revision_key(model_name)
        if key not in self._cards:
            raise FrameCardError(
                f"CheckpointFrameCache: no card for {key!r}. Build one per "
                f"checkpoint; do not reuse another checkpoint's."
            )
        card, store = self._cards[key]
        verify_card_for_run(card, parse_checkpoint(model_name)["base"],
                            checkpoint_revision(model_name),
                            context="checkpoint frame cache")
        return card, store

    def has(self, model_name: str) -> bool:
        return revision_key(model_name) in self._cards

    def keys(self) -> list:
        return sorted(self._cards)

    def __len__(self) -> int:
        return len(self._cards)


def assert_family_is_homogeneous(model_names) -> str:
    """
    Every name in a sweep must share one base. Returns that base.

    Guards the case `checkpoint_families` cannot: a caller assembling a list
    by hand and mixing deduped with non-deduped, or two model sizes.
    """
    bases = {parse_checkpoint(m)["base"] for m in model_names}
    if len(bases) != 1:
        raise ValueError(
            f"assert_family_is_homogeneous: a sweep must be one family, got "
            f"{sorted(bases)}. Deduped and non-deduped are separate training "
            f"runs and their checkpoints are not comparable."
        )
    return bases.pop()


# ---------------------------------------------------------------------------
# Log-step derivatives
# ---------------------------------------------------------------------------

def step_x(steps) -> np.ndarray:
    """log10(step + 1). Matches checkpoints._step_x; duplicated deliberately
    so this module carries no plotting dependency."""
    return np.log10(np.asarray(steps, dtype=float) + 1.0)


def log_step_derivative(steps, values) -> dict:
    """
    d(value) / d(log10(step + 1)), by central differences on unequal spacing.

    Taking the derivative over checkpoint *index* instead places its largest
    magnitudes wherever Pythia's release spacing changes — at step 512, where
    log-spacing gives way to every-1000 — which is an artifact of the release
    schedule, not of training.

    Returns dict(x, derivative, steps).
    """
    s = np.asarray(steps, dtype=float)
    v = np.asarray(values, dtype=float)
    if s.shape != v.shape:
        raise ValueError(f"log_step_derivative: shape mismatch {s.shape} vs {v.shape}")
    if s.size < 3:
        return {"x": step_x(s), "derivative": np.full(s.shape, np.nan),
                "steps": s}
    order = np.argsort(s)
    x, y = step_x(s[order]), v[order]
    d = np.gradient(y, x)          # handles non-uniform spacing
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return {"x": step_x(s), "derivative": d[inv], "steps": s}


def index_derivative(values) -> np.ndarray:
    """
    The naive derivative over checkpoint index, kept so the difference can be
    measured rather than argued about.
    """
    v = np.asarray(values, dtype=float)
    return np.gradient(v) if v.size >= 2 else np.full(v.shape, np.nan)


def interval_rates(steps, values) -> dict:
    """
    Forward differences between consecutive checkpoints, per unit log-step.

    Central differences (`log_step_derivative`) smooth a jump across its two
    neighbours, so a change occurring between samples i and i+1 shows up at
    i-1 and i+1 and the peak can land on the wrong side. For LOCATING a
    transition the honest object is the interval, not a point: a sweep can
    only ever say "between these two checkpoints".

    Returns dict(lo, hi, rate) with one entry per consecutive pair.
    """
    s = np.asarray(steps, dtype=float)
    v = np.asarray(values, dtype=float)
    if s.shape != v.shape:
        raise ValueError(f"interval_rates: shape mismatch {s.shape} vs {v.shape}")
    order = np.argsort(s)
    s, v = s[order], v[order]
    if s.size < 2:
        return {"lo": np.zeros(0), "hi": np.zeros(0), "rate": np.zeros(0)}
    x = step_x(s)
    dx = np.diff(x)
    dx[dx == 0] = np.nan
    return {"lo": s[:-1], "hi": s[1:], "rate": np.diff(v) / dx}


def detect_transitions(steps, values, n_top: int = 1,
                       min_abs: float = 0.0) -> dict:
    """
    Locate the INTERVALS of largest change, in log-step.

    Reports intervals rather than points: a sweep samples training, so the
    strongest available statement is "between step A and step B". A
    single-step answer implies a resolution the data does not have.

    `index_intervals` is what an index-based rate would have picked, kept in
    the record so a transition claim can be checked against the release-
    schedule artifact it might be.
    """
    s = np.asarray(steps, dtype=float)
    ir = interval_rates(s, values)
    if ir["rate"].size == 0:
        return {"intervals": [], "magnitudes": [], "index_intervals": [],
                "derivative": log_step_derivative(s, values)["derivative"]}

    mag = np.abs(ir["rate"])
    ok = ~np.isnan(mag)
    cand = np.argsort(-np.where(ok, mag, -np.inf))[:n_top]
    cand = [int(i) for i in cand if ok[i] and mag[i] >= min_abs]

    # Index-based rate: the same forward differences with dx = 1 per sample,
    # i.e. no notion of how much training separates two checkpoints.
    v = np.asarray(values, dtype=float)[np.argsort(s)]
    imag = np.abs(np.diff(v))
    icand = np.argsort(-imag)[:n_top]

    lo, hi = ir["lo"], ir["hi"]
    return {
        "intervals": [(int(lo[i]), int(hi[i])) for i in cand],
        "magnitudes": [float(mag[i]) for i in cand],
        "index_intervals": [(int(lo[i]), int(hi[i])) for i in icand],
        "derivative": log_step_derivative(s, values)["derivative"],
    }


def spacing_change_steps(steps) -> list:
    """
    Steps where the checkpoint spacing itself changes.

    Any transition reported at one of these deserves scrutiny: an
    index-based derivative will place a peak here by construction.
    """
    s = np.sort(np.asarray(steps, dtype=float))
    if s.size < 3:
        return []
    gaps = np.diff(s)
    out = []
    for i in range(1, gaps.size):
        if gaps[i] != gaps[i - 1]:
            out.append(int(s[i]))
    return out


def transition_summary_lines(result: dict, steps) -> list:
    changes = set(spacing_change_steps(steps))
    lines = ["Transitions (log-step intervals):"]
    for (lo, hi), mag in zip(result["intervals"], result["magnitudes"]):
        warn = "  <-- spans a spacing change" if hi in changes else ""
        lines.append(f"  {lo} -> {hi:<8d} rate = {mag:.4f} per log10-step{warn}")
    if result["index_intervals"] != result["intervals"]:
        lines.append(
            f"  index-based rate would report {result['index_intervals']} "
            f"— disagreement means the choice of x-axis is load-bearing"
        )
    return lines
