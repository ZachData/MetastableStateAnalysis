"""
core/frames.py — The frame ledger (frames item 7).

Why this module exists
----------------------
The distance-measurement bug was not a coding error. Every line was
correct for the object it named; the object was the wrong one for the
model. It survived because **no result record stated which frame its
numbers lived in**, so nothing could contradict anything else. The rotary
omission (core/rope.py) is the same class, found the same accidental way.

The patch for each instance is local. The fix for the class is this: every
metric record carries a FrameSpec, and any cross-record comparison passes
through a guard that *refuses* rather than warns. One dataclass and one
assert converts a silent failure mode into a loud one.

Design commitments
------------------
1. **Convention vs data.** `kind`, `rope_applied`, and `pos0_policy` are
   conventions — two records that disagree on them are not comparable, full
   stop. `layer_idx` and `reader_block` are data — records legitimately
   differ there. `verify_same_frame` compares only the conventions;
   `verify_same_revision` handles model identity separately, because
   comparing checkpoints is the point of the sweep.

2. **One place applies a frame.** `apply_frame` is the only function that
   turns raw activations into frame activations. A call site that
   normalizes inline is a call site that cannot be audited.

3. **Frozen and hashable.** A FrameSpec that a downstream function can
   mutate is a FrameSpec that stops describing the numbers it is attached
   to.

4. **No torch.** Pure numpy on top of core.ln_frame and core.metrics, so
   the whole ledger is oracle-testable in the stubbed session.

Usage
-----
    from core.frames import FrameSpec, apply_frame, attach_frame, verify_same_frame

    res  = frame_for_hidden_state(model, L, n_hidden, which="attn")
    spec = FrameSpec.from_ln_resolution(res, layer_idx=L, model_rev=rev,
                                        rope_applied=True)
    Xf   = apply_frame(X, spec, ln_params=res["params"])
    ...
    attach_frame(record, spec)              # record["frame"] = spec.to_dict()

    verify_same_frame(rec_a, rec_b, context="trained vs random energy")

See DESIGN_pythia_frames.md, item 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from core.metrics import _as_numpy, l2_normalize
from core.ln_frame import ln_transform, DEFAULT_LN_EPS


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Frames activations can live in.
#:   raw        — residual stream, untouched
#:   l2_sphere  — row-normalized to the unit sphere (the original particle frame)
#:   ln_attn    — LN1(x): what attention reads. Pythia's coupling frame.
#:   ln_mlp     — LN2(x): what the MLP reads. Under parallel residual this is
#:                the SAME input as ln_attn, differently scaled — not a
#:                post-attention state, despite the module name.
#:   identity   — already normalized upstream (post-final-LN extraction path)
FRAME_KINDS = ("raw", "l2_sphere", "ln_attn", "ln_mlp", "identity")

POS0_POLICIES = ("included", "excluded")

#: Fields that are *conventions*. Disagreement here means two numbers are
#: not measuring the same thing, regardless of how close they look.
CONVENTION_FIELDS = ("kind", "rope_applied", "pos0_policy")

UNKNOWN_REV = "unknown"

#: reader_block sentinel for the final_layer_norm frame, which has no block.
FINAL_LN_BLOCK = -1


class FrameMismatch(ValueError):
    """Raised when records in different frames are about to be compared."""


# ---------------------------------------------------------------------------
# FrameSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameSpec:
    """
    The frame a set of numbers lives in.

    Parameters
    ----------
    kind         : one of FRAME_KINDS
    layer_idx    : hidden-state index the numbers came from (data, not convention)
    reader_block : block whose LN defines the frame; None for l2/raw/identity
    model_rev    : checkpoint revision string. NOT the model name — the whole
                   point of the sweep is comparing revisions, so this must be
                   specific enough to distinguish step1000 from step143000,
                   and pythia-1.4b from pythia-1.4b-deduped.
    rope_applied : whether rotary was included in whatever bilinear produced
                   these numbers. False on a rotary model is a live claim that
                   the quantity is a proxy, not an accident.
    pos0_policy  : whether position 0 (the NeoX attention sink) was included.
    ln_eps       : recorded so an eps change cannot silently move results.
    extras       : free-form, hashed by repr; use for one-off qualifiers.
    """

    kind: str
    layer_idx: int | None = None
    reader_block: int | None = None
    model_rev: str = UNKNOWN_REV
    rope_applied: bool = False
    pos0_policy: str = "included"
    ln_eps: float = DEFAULT_LN_EPS
    extras: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if self.kind not in FRAME_KINDS:
            raise ValueError(
                f"FrameSpec: kind must be one of {FRAME_KINDS}, got {self.kind!r}"
            )
        if self.pos0_policy not in POS0_POLICIES:
            raise ValueError(
                f"FrameSpec: pos0_policy must be one of {POS0_POLICIES}, "
                f"got {self.pos0_policy!r}"
            )
        if self.kind in ("ln_attn", "ln_mlp") and self.reader_block is None:
            raise ValueError(
                f"FrameSpec: kind={self.kind!r} requires reader_block. The "
                f"off-by-one between hidden-state index and reading block is "
                f"exactly what this field exists to pin down; resolve it with "
                f"core.ln_frame.frame_for_hidden_state rather than passing None."
            )

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "layer_idx": self.layer_idx,
            "reader_block": self.reader_block,
            "model_rev": self.model_rev,
            "rope_applied": bool(self.rope_applied),
            "pos0_policy": self.pos0_policy,
            "ln_eps": float(self.ln_eps),
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "FrameSpec":
        extras = d.get("extras") or {}
        return cls(
            kind=d["kind"],
            layer_idx=d.get("layer_idx"),
            reader_block=d.get("reader_block"),
            model_rev=d.get("model_rev", UNKNOWN_REV),
            rope_applied=bool(d.get("rope_applied", False)),
            pos0_policy=d.get("pos0_policy", "included"),
            ln_eps=float(d.get("ln_eps", DEFAULT_LN_EPS)),
            extras=tuple(sorted(dict(extras).items())),
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_ln_resolution(
        cls,
        resolution: Mapping,
        layer_idx: int,
        model_rev: str = UNKNOWN_REV,
        which: str = "attn",
        rope_applied: bool = False,
        pos0_policy: str = "included",
        ln_eps: float = DEFAULT_LN_EPS,
    ) -> "FrameSpec":
        """
        Build from core.ln_frame.frame_for_hidden_state's return value.

        That function is the single home of the off-by-one; this constructor
        is the single bridge from it into the ledger, so no call site
        re-derives either.
        """
        frame = resolution["frame"]
        if frame == "identity":
            kind, reader_block = "identity", None
        elif frame == "final":
            # final_layer_norm has no reading block. reader_block = -1 is the
            # sentinel for it, so the "LN kinds need a reader_block" invariant
            # still holds and the source is recorded explicitly in extras.
            return cls(
                kind="ln_attn" if which == "attn" else "ln_mlp",
                layer_idx=layer_idx,
                reader_block=FINAL_LN_BLOCK,
                model_rev=model_rev,
                rope_applied=rope_applied,
                pos0_policy=pos0_policy,
                ln_eps=ln_eps,
                extras=(("ln_source", "final_layer_norm"),),
            )
        elif frame == "block":
            kind = "ln_attn" if which == "attn" else "ln_mlp"
            reader_block = resolution["block_idx"]
        else:
            raise ValueError(f"FrameSpec.from_ln_resolution: unknown frame {frame!r}")
        return cls(
            kind=kind,
            layer_idx=layer_idx,
            reader_block=reader_block,
            model_rev=model_rev,
            rope_applied=rope_applied,
            pos0_policy=pos0_policy,
            ln_eps=ln_eps,
        )

    @classmethod
    def l2_sphere(
        cls, layer_idx: int | None = None, model_rev: str = UNKNOWN_REV,
        pos0_policy: str = "included",
    ) -> "FrameSpec":
        """The original particle frame. Kept as a first-class option, not a default."""
        return cls(kind="l2_sphere", layer_idx=layer_idx, model_rev=model_rev,
                   pos0_policy=pos0_policy)

    @classmethod
    def raw(
        cls, layer_idx: int | None = None, model_rev: str = UNKNOWN_REV,
        pos0_policy: str = "included",
    ) -> "FrameSpec":
        return cls(kind="raw", layer_idx=layer_idx, model_rev=model_rev,
                   pos0_policy=pos0_policy)

    def with_(self, **kw) -> "FrameSpec":
        """Derived spec; frozen, so this returns a new object."""
        return replace(self, **kw)

    # -- reporting ----------------------------------------------------------

    def describe(self) -> str:
        bits = [self.kind]
        if self.reader_block is not None:
            bits.append(f"reader=block{self.reader_block}")
        if self.layer_idx is not None:
            bits.append(f"hidden={self.layer_idx}")
        bits.append("rope" if self.rope_applied else "no-rope")
        bits.append(f"pos0={self.pos0_policy}")
        bits.append(self.model_rev)
        return " | ".join(bits)

    def is_ln(self) -> bool:
        return self.kind in ("ln_attn", "ln_mlp")


# ---------------------------------------------------------------------------
# Applying a frame — the only place activations get transformed
# ---------------------------------------------------------------------------

def apply_frame(X, spec: FrameSpec, ln_params: Mapping | None = None) -> np.ndarray:
    """
    Transform raw residual-stream activations into `spec`'s frame.

    Parameters
    ----------
    X         : (n, d) raw residual stream for one hidden-state index
    spec      : FrameSpec
    ln_params : {"gamma":…, "beta":…, "eps":…} for ln_attn/ln_mlp, as
                returned by core.ln_frame.get_ln_params. Required for LN
                kinds — refusing to default to plain LN is deliberate:
                silently dropping gamma is precisely the failure this
                module exists to prevent.

    pos0_policy is NOT applied here. Dropping a row must happen in lockstep
    with token_ids, attention matrices, and labels, so it is a caller
    concern served by `pos0_mask`.
    """
    arr = _as_numpy(X).astype(np.float64, copy=False)
    if arr.ndim == 1:
        arr = arr[None, :]

    if spec.kind in ("raw", "identity"):
        return arr.copy()
    if spec.kind == "l2_sphere":
        return np.asarray(l2_normalize(arr), dtype=np.float64)
    if spec.is_ln():
        if ln_params is None:
            raise ValueError(
                f"apply_frame: kind={spec.kind!r} requires ln_params "
                f"(gamma/beta/eps). Pass core.ln_frame.get_ln_params(...) or "
                f"frame_for_hidden_state(...)['params']."
            )
        return ln_transform(
            arr,
            gamma=ln_params.get("gamma"),
            beta=ln_params.get("beta"),
            eps=float(ln_params.get("eps", spec.ln_eps)),
        )
    raise ValueError(f"apply_frame: unhandled kind {spec.kind!r}")


def frame_gram(X, spec: FrameSpec, ln_params: Mapping | None = None) -> np.ndarray:
    """Gram matrix of the frame activations. The pairwise geometry, in one call."""
    Xf = apply_frame(X, spec, ln_params)
    return Xf @ Xf.T


# ---------------------------------------------------------------------------
# Position-0 policy
# ---------------------------------------------------------------------------

def pos0_mask(n: int, policy: str = "included") -> np.ndarray:
    """
    Boolean keep-mask of length n implementing a pos0 policy.

    NeoX tokenizers do not prepend BOS, so position 0 becomes the attention
    sink and can carry a norm one to two orders above every other token.
    That single particle can dominate the raw Gram, dominate E_beta through
    exp(beta * <.,.>), and dominate clustering. Whether it is in or out must
    be one explicit decision applied identically across GPT-2, Pythia, and
    every checkpoint — otherwise a trained-vs-random energy contrast is
    partly a sink contrast.

    Apply the same mask to activations, token_ids, labels, and attention
    matrices (both axes). See DESIGN_pythia_frames.md, policy item P1.
    """
    if policy not in POS0_POLICIES:
        raise ValueError(f"pos0_mask: policy must be one of {POS0_POLICIES}, got {policy!r}")
    keep = np.ones(n, dtype=bool)
    if policy == "excluded" and n > 0:
        keep[0] = False
    return keep


def apply_pos0_policy(arrays: Sequence, policy: str = "included", axes=None) -> list:
    """
    Apply pos0_mask to several aligned arrays at once.

    `axes` is a per-array tuple of axes to filter (default: axis 0 only).
    Pass (0, 1) for an (n, n) attention or Gram matrix. Doing this in one
    call is the point — a mask applied to activations but not to token_ids
    is a silent misalignment.
    """
    if not arrays:
        return []
    axes = axes or [(0,)] * len(arrays)
    if len(axes) != len(arrays):
        raise ValueError("apply_pos0_policy: axes must be the same length as arrays")

    # One reference length for the whole call. Every filtered axis of every
    # array must match it: a mask applied to activations but not to token_ids
    # is silent corruption, and per-array lengths would hide exactly that.
    n_ref = int(_as_numpy(arrays[0]).shape[axes[0][0]])
    keep = pos0_mask(n_ref, policy)

    out = []
    for i, (arr, ax) in enumerate(zip(arrays, axes)):
        a = _as_numpy(arr)
        for axis in ax:
            if a.shape[axis] != n_ref:
                raise ValueError(
                    f"apply_pos0_policy: array {i} axis {axis} has length "
                    f"{a.shape[axis]}, expected {n_ref} — arrays are not aligned, "
                    f"so a shared position mask would misalign them."
                )
            a = np.compress(keep, a, axis=axis)
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# Attaching and guarding
# ---------------------------------------------------------------------------

def attach_frame(record: dict, spec: FrameSpec) -> dict:
    """Write the ledger entry into a result record, in place. Returns the record."""
    record["frame"] = spec.to_dict()
    return record


def frame_of(record: Mapping, strict: bool = True) -> FrameSpec | None:
    """
    Read a record's FrameSpec.

    strict=True (default) raises on a record with no ledger entry. A record
    without a frame is a record whose numbers cannot be checked; treating
    that as an error is the whole mechanism.
    """
    d = record.get("frame") if isinstance(record, Mapping) else None
    if d is None:
        if strict:
            raise FrameMismatch(
                "frame_of: record carries no frame ledger. Every metric record "
                "must record the frame its numbers live in — see "
                "core.frames.attach_frame."
            )
        return None
    return FrameSpec.from_dict(d)


def verify_same_frame(
    a: Mapping | FrameSpec,
    b: Mapping | FrameSpec,
    fields: Sequence[str] = CONVENTION_FIELDS,
    context: str = "",
) -> None:
    """
    Refuse to let two records be compared across differing conventions.

    Compares CONVENTION_FIELDS only: `layer_idx` and `reader_block` are data
    and legitimately differ. Model revision is checked by
    verify_same_revision, since comparing revisions is the sweep's purpose.

    Raises FrameMismatch with the specific disagreement — a message that
    names the field is the difference between a two-minute fix and another
    accidental discovery.
    """
    sa = a if isinstance(a, FrameSpec) else frame_of(a)
    sb = b if isinstance(b, FrameSpec) else frame_of(b)
    bad = [f for f in fields if getattr(sa, f) != getattr(sb, f)]
    if bad:
        detail = "; ".join(
            f"{f}: {getattr(sa, f)!r} vs {getattr(sb, f)!r}" for f in bad
        )
        where = f" [{context}]" if context else ""
        raise FrameMismatch(
            f"Refusing to compare records in different frames{where}. "
            f"Disagreement on {detail}. "
            f"A={sa.describe()} | B={sb.describe()}"
        )


def verify_same_revision(a, b, context: str = "") -> None:
    """
    Assert two records came from the same checkpoint.

    Separate from verify_same_frame because cross-revision comparison is
    intended in the sweep and forbidden everywhere else — e.g. a frame built
    from the final model's LN gamma/beta applied to a step-1000 checkpoint is
    a real and easy mistake that this catches.
    """
    sa = a if isinstance(a, FrameSpec) else frame_of(a)
    sb = b if isinstance(b, FrameSpec) else frame_of(b)
    if sa.model_rev != sb.model_rev:
        where = f" [{context}]" if context else ""
        raise FrameMismatch(
            f"Records are from different revisions{where}: "
            f"{sa.model_rev!r} vs {sb.model_rev!r}"
        )
    if sa.model_rev == UNKNOWN_REV:
        raise FrameMismatch(
            f"Both records carry model_rev={UNKNOWN_REV!r}{' [' + context + ']' if context else ''}. "
            f"An unrecorded revision cannot be verified; set it at extraction time."
        )


def verify_all_same_frame(records: Sequence, context: str = "") -> FrameSpec:
    """Guard a whole collection; returns the common spec. Empty input raises."""
    if not records:
        raise FrameMismatch(f"verify_all_same_frame: empty record set [{context}]")
    first = records[0] if isinstance(records[0], FrameSpec) else frame_of(records[0])
    for r in records[1:]:
        verify_same_frame(first, r, context=context)
    return first


def frame_summary_lines(spec: FrameSpec) -> list:
    """Report block for the phase reports, so the frame is visible in output."""
    return [
        "Frame:",
        f"  kind          {spec.kind}",
        f"  reader block  {spec.reader_block}",
        f"  hidden index  {spec.layer_idx}",
        f"  rotary        {'applied' if spec.rope_applied else 'OMITTED (proxy)'}",
        f"  position 0    {spec.pos0_policy}",
        f"  revision      {spec.model_rev}",
    ]
