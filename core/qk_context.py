"""
qk_context.py — Phase 6's QK logit matrices, computed correctly.

Replaces run_6.py::_compute_qk_logit_matrices, which computes

    logit_h[i, j] = (X @ (WQ_h @ WK_h.T) @ X.T)[i, j]

That expression carries three independent first-order errors on Pythia,
measured per head against true logits (DESIGN_pythia_frames.md item 10):

    frame omitted (L2 sphere instead of LN1)     pearson 0.86
    rotary omitted                               pearson 0.72
    QK biases omitted                            pearson 0.99
    all three — what ships today                 pearson 0.60

This module is the single choke point for all of them. Everything
downstream in Phase 6 reads ctx["qk_logit_matrices"], so head_classify,
qk_decompose, and the induction comparisons become correct without being
touched.

Dual-run by design
------------------
`build_qk_logit_context` computes the corrected matrices AND, on request,
the legacy ones, and reports the per-head divergence between them. The
side-by-side diff that decides whether Phase 6 needs a full re-run
(sequencing principle S3) is therefore a byproduct of the fix rather than a
separate exercise, and the frame ledger keeps the two sets from ever being
compared as if they were the same quantity.

What this module refuses to do
------------------------------
It does not guess. If the incoming activations are not in a recognisable
frame, if the frame card is absent, or if the QK biases were never
persisted, it records the gap in the ledger and proceeds with the omission
marked — rather than silently producing a plausible number. Every
correction that is off is off *on the record*.

See DESIGN_pythia_frames.md items 5, 5a, 10.
"""

from __future__ import annotations

import numpy as np

from core.frames import FrameSpec, apply_frame
from core.metrics import _as_numpy
from core.rope import (
    causal_pair_mask,
    qk_logits_with_rope,
    qk_prediction_fidelity,
)


# ---------------------------------------------------------------------------
# Frame detection
# ---------------------------------------------------------------------------

def detect_activation_frame(X, tol: float = 1e-3) -> str:
    """
    What frame are these activations already in?

    Phase 1's extract_activations returns RAW hidden states, so the LN frame
    is reconstructible in Phase 6 — but whether the saved artifact was
    normalized before persistence is a property of the writer, not of this
    call site. Detect rather than assume: unit row norms mean the sphere,
    anything else means raw.

    Returns "l2_sphere", "raw", or "empty". The answer goes into the ledger,
    so a mistaken assumption becomes a visible field instead of a silent
    transformation.
    """
    arr = _as_numpy(X)
    if arr is None or arr.size == 0:
        return "empty"
    norms = np.linalg.norm(np.asarray(arr, dtype=np.float64), axis=1)
    if np.allclose(norms, 1.0, atol=tol):
        return "l2_sphere"
    return "raw"


# ---------------------------------------------------------------------------
# The corrected computation
# ---------------------------------------------------------------------------

def compute_head_logits(
    X_frame,
    qk_pairs,
    rotary_ndims: int,
    rope_base: float,
    attn_scale: float | None,
    qk_biases=None,
    positions=None,
    dtype=np.float32,
) -> list:
    """
    Per-head pre-softmax logits from activations already in the reader frame.

    Parameters
    ----------
    X_frame   : (n, d_model) IN THE READER'S FRAME — this function does not
                normalize, and passing the wrong frame will not raise
    qk_pairs  : list of (WQ, WK), each (d_model, d_head) canonical
    qk_biases : list of (bq, bk) per head, or None. None is a real omission,
                not a default; the caller must record it.
    """
    out = []
    for h, (WQ, WK) in enumerate(qk_pairs):
        bq = bk = None
        if qk_biases is not None and h < len(qk_biases):
            bq, bk = qk_biases[h]
        L = qk_logits_with_rope(
            X_frame, WQ, WK, rotary_ndims, rope_base,
            positions=positions, scale=attn_scale, bq=bq, bk=bk,
        )
        out.append(np.asarray(L, dtype=dtype))
    return out


def compute_legacy_logits(X, qk_pairs, dtype=np.float32) -> list:
    """
    The shipping computation, preserved verbatim for the diff.

    Kept as its own function so the comparison is against what actually ran,
    not against a reconstruction of it.
    """
    return [np.asarray(X @ (WQ @ WK.T) @ X.T, dtype=dtype) for WQ, WK in qk_pairs]


def compare_logit_sets(corrected, legacy, causal_only: bool = True) -> dict:
    """
    Per-head divergence between the corrected and legacy matrices.

    This is the S3 diff. `worst_pearson` across heads is the number that
    decides whether a Phase 6 conclusion survives the fix or needs a re-run;
    a head-level list is returned as well, because a conclusion resting on
    one head is only as good as that head's agreement.
    """
    if not corrected or not legacy or len(corrected) != len(legacy):
        return {"per_head": [], "worst_pearson": None, "median_pearson": None,
                "n_heads": 0}
    n = corrected[0].shape[0]
    mask = causal_pair_mask(n) if causal_only else None
    per_head = []
    for h, (c, l) in enumerate(zip(corrected, legacy)):
        fid = qk_prediction_fidelity(l, c, mask=mask)
        per_head.append({"head": h, **fid})
    pear = [p["pearson"] for p in per_head if not np.isnan(p["pearson"])]
    return {
        "per_head": per_head,
        "worst_pearson": float(min(pear)) if pear else None,
        "median_pearson": float(np.median(pear)) if pear else None,
        "n_heads": len(per_head),
    }


# ---------------------------------------------------------------------------
# Context builder — the drop-in replacement
# ---------------------------------------------------------------------------

def build_qk_logit_context(
    ctx: dict,
    card=None,
    ln_store=None,
    qk_biases=None,
    n_hidden_states: int | None = None,
    which: str = "attn",
    pos0_policy: str = "included",
    keep_legacy: bool = True,
) -> dict:
    """
    Populate ctx["qk_logit_matrices"] correctly, in place. Returns ctx.

    Drop-in for run_6.py::_compute_qk_logit_matrices. Same contract on
    absence: leaves the key None when qk_matrices or token_activations are
    missing.

    Additional keys written, all of them so that a downstream reader can tell
    what was and was not corrected:

        qk_logit_frame          FrameSpec.to_dict() for the matrices
        qk_logit_corrections    {"frame","rotary","bias"} -> bool
        qk_logit_legacy         the old matrices, when keep_legacy
        qk_logit_diff           per-head corrected-vs-legacy divergence
        qk_logit_notes          human-readable list of what was skipped

    Degradation is explicit. With no frame card this falls back to the legacy
    computation and says so in `notes` and in the ledger's `rope_applied`
    field — it does not half-correct.
    """
    qk = ctx.get("qk_matrices")
    X_in = ctx.get("token_activations")
    notes = []

    if not qk or X_in is None:
        ctx["qk_logit_matrices"] = None
        ctx["qk_logit_frame"] = None
        ctx["qk_logit_corrections"] = {"frame": False, "rotary": False, "bias": False}
        ctx["qk_logit_notes"] = ["qk_matrices or token_activations absent"]
        return ctx

    X = np.asarray(_as_numpy(X_in), dtype=np.float64)
    incoming = detect_activation_frame(X)
    layer_idx = int(ctx.get("layer_idx", 0))

    legacy = compute_legacy_logits(X, qk) if keep_legacy else None

    # ---- no card: refuse to half-correct -------------------------------
    if card is None or ln_store is None:
        ctx["qk_logit_matrices"] = legacy if legacy is not None else \
            compute_legacy_logits(X, qk)
        ctx["qk_logit_frame"] = FrameSpec(
            kind="l2_sphere" if incoming == "l2_sphere" else "raw",
            layer_idx=layer_idx, rope_applied=False, pos0_policy=pos0_policy,
            extras=(("incoming_frame", incoming), ("uncorrected", "no_frame_card")),
        ).to_dict()
        ctx["qk_logit_corrections"] = {"frame": False, "rotary": False, "bias": False}
        ctx["qk_logit_legacy"] = None
        ctx["qk_logit_diff"] = None
        ctx["qk_logit_notes"] = [
            "No frame card supplied — QK logits are UNCORRECTED. Expected "
            "correlation with true logits ~0.6 on Pythia. Write a frame card "
            "at extraction time (core/frame_card.py) to enable the fix."
        ]
        return ctx

    # ---- frame ----------------------------------------------------------
    n_hidden = n_hidden_states if n_hidden_states is not None else \
        len(ctx.get("activations_per_layer") or []) or (card.n_blocks + 1)
    spec = card.frame_spec_for(layer_idx, n_hidden, which=which,
                               pos0_policy=pos0_policy)

    if incoming == "l2_sphere":
        # The LN frame is not recoverable from normalized activations: the
        # per-token scale was discarded and LN's gamma/beta act on the
        # unnormalized vector. Say so rather than applying LN to a sphere.
        X_frame = X
        frame_ok = False
        notes.append(
            "token_activations are already L2-normalized; the LN frame cannot "
            "be reconstructed from them. Re-extract raw hidden states (Phase 1 "
            "extract_activations returns them raw) to enable the frame fix."
        )
        spec = spec.with_(kind="l2_sphere", reader_block=None) \
            if spec.is_ln() else spec
    else:
        X_frame = apply_frame(X, spec, ln_store.params_for(spec))
        frame_ok = True

    # ---- biases ---------------------------------------------------------
    bias_ok = qk_biases is not None
    if not bias_ok:
        notes.append(
            "QK biases not supplied — b_q^T R W_K^T x_j is a per-key logit "
            "offset and its omission is first-order. Persist "
            "query_key_value.bias in the Phase 2 writer."
        )

    # ---- rotary ---------------------------------------------------------
    rotary_ok = card.uses_rope
    corrected = compute_head_logits(
        X_frame, qk,
        rotary_ndims=card.rotary_ndims,
        rope_base=card.rope_base,
        attn_scale=card.attn_scale,
        qk_biases=qk_biases,
    )

    extras = [("incoming_frame", incoming),
              ("bias_applied", str(bias_ok)),
              ("frame_applied", str(frame_ok))]
    spec = spec.with_(rope_applied=rotary_ok, extras=tuple(sorted(extras)))

    ctx["qk_logit_matrices"] = corrected
    ctx["qk_logit_frame"] = spec.to_dict()
    ctx["qk_logit_corrections"] = {
        "frame": frame_ok, "rotary": rotary_ok, "bias": bias_ok,
    }
    ctx["qk_logit_legacy"] = legacy
    ctx["qk_logit_diff"] = compare_logit_sets(corrected, legacy) if legacy else None
    ctx["qk_logit_notes"] = notes
    return ctx


def qk_context_summary_lines(ctx: dict) -> list:
    """Report block, so what was corrected is visible in Phase 6's output."""
    corr = ctx.get("qk_logit_corrections") or {}
    diff = ctx.get("qk_logit_diff") or {}
    lines = ["QK logit matrices:"]
    for k in ("frame", "rotary", "bias"):
        state = "applied" if corr.get(k) else "OMITTED"
        lines.append(f"  {k:8s} {state}")
    if diff.get("worst_pearson") is not None:
        lines.append(
            f"  vs legacy: median r={diff['median_pearson']:.4f}, "
            f"worst r={diff['worst_pearson']:.4f} over {diff['n_heads']} heads"
        )
    for note in ctx.get("qk_logit_notes") or []:
        lines.append(f"  ! {note}")
    return lines
