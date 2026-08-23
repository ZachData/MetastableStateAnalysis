"""
p7_motifs/interaction_graph.py — build a typed InteractionTable from one
head's activations, attention and OV circuit.

This is Phase 7's producer. Everything downstream (motif counts, nulls,
formation curves) is an aggregation over what this module emits, so its
two decisions matter more than anything in the analysis layer.

Decision 1: force, not attention
---------------------------------
The displacement particle i receives from particle j through head h is

    f_ij = A_ij * (x_j @ OV_h)

where OV_h = W_V @ W_O is the composed circuit (not W_O alone — see
`archive/p6_subspace/induction_ov.py`'s Bug 10 note, which is the same
distinction). Attention alone cannot distinguish two heads with identical
patterns and opposite-signed OV circuits, which move particles in opposite
directions.

Decision 2: how the n^2 x d tensor is never built
--------------------------------------------------
Materializing every f_ij is (n_tokens^2, d_model) per head per layer per
checkpoint — at n=512, d=2048 that is ~4 GB for ONE head in float64.

It is not necessary. Because A_ij >= 0 after softmax,

    ||f_ij|| = A_ij * ||x_j @ OV_h||

so every edge's force MAGNITUDE follows from one (n, d) matrix product and
a row-norm — O(n d^2) once, then an outer product. Selection by magnitude
therefore happens before any force vector exists, and only the retained
edges' vectors are ever formed. This is the same reasoning
`p2b_imaginary/rotational_schur.top_rotation_planes` used to stop building
(d, d) projectors, applied on the other axis.

Frame discipline
----------------
The activations must be in the same frame as the projectors. Phase 2's OV
projectors live in the residual-stream basis, so L2-normalized activations
are the wrong input — and nothing about the shapes reveals it. The frame
is detected (`core.qk_context.detect_activation_frame`), compared against
what the caller declares, and stamped into the output. A mismatch refuses.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from core.interactions import InteractionTable, classify_pair_types
from core.qk_context import detect_activation_frame

# Placed, not calibrated (standing rule 6): no observed force distribution
# exists yet to derive it from.
DEFAULT_TOP_K_PER_TARGET = 16


def edge_force_magnitudes(
    X: np.ndarray,
    attention: np.ndarray,
    OV: np.ndarray,
    causal: bool = True,
) -> tuple:
    """
    Every edge's force magnitude, without forming a single force vector.

    Returns (magnitudes, moved) where
      magnitudes : (n, n) — magnitudes[i, j] = ||f_ij||
      moved      : (n, d) — moved[j] = x_j @ OV, the direction particle j
                   pushes along, computed once and reused for whichever
                   edges are retained.

    causal=True zeroes the j > i upper triangle. A causal LM cannot have
    those edges, and leaving them in would put impossible interactions into
    every count and every null.
    """
    X = np.asarray(X, dtype=np.float64)
    attention = np.asarray(attention, dtype=np.float64)
    OV = np.asarray(OV, dtype=np.float64)

    n, d = X.shape
    if attention.shape != (n, n):
        raise ValueError(
            f"attention is {attention.shape}, expected ({n}, {n}) to match "
            f"{n} activation rows"
        )
    if OV.shape != (d, d):
        raise ValueError(
            f"OV is {OV.shape}, expected ({d}, {d}) to match d_model={d} "
            "(frame mismatch, or W_O passed instead of the composed W_V @ W_O)"
        )

    moved = X @ OV                       # (n, d) — one matmul, O(n d^2)
    push = np.linalg.norm(moved, axis=1)  # (n,)
    magnitudes = attention * push[None, :]

    if causal:
        magnitudes = np.tril(magnitudes)

    return magnitudes, moved


def select_edges(
    magnitudes: np.ndarray,
    top_k_per_target: Optional[int] = DEFAULT_TOP_K_PER_TARGET,
    min_magnitude: float = 0.0,
) -> tuple:
    """
    Which edges to retain, and the retention record that must travel with
    them.

    Selection is top-k PER TARGET, not globally. A global top-k would let a
    few high-norm particles consume the whole budget and leave other
    particles with no incoming edges at all — which does not read as "this
    particle was not moved much", it reads as "this particle was not
    moved", and every per-target motif (`hub`, `mutual`, both relay stages)
    would then be counted against a denominator that silently varies by
    particle.

    top_k_per_target=None retains everything above `min_magnitude`.

    Returns (targets, sources, retention) with retention describing the
    cutoff, so an absent edge is never mistaken for a zero-force edge.
    """
    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    n = magnitudes.shape[0]
    keep = magnitudes > min_magnitude

    if top_k_per_target is not None:
        k = int(top_k_per_target)
        if k < 1:
            raise ValueError(f"top_k_per_target must be >= 1 or None; got {k}")
        limited = np.zeros_like(keep)
        for i in range(n):
            cand = np.flatnonzero(keep[i])
            if cand.size == 0:
                continue
            if cand.size > k:
                cand = cand[np.argsort(magnitudes[i, cand])[::-1][:k]]
            limited[i, cand] = True
        keep = limited

    targets, sources = np.nonzero(keep)
    retention = {
        "mode": "top_k_per_target" if top_k_per_target is not None else "threshold_only",
        "k": int(top_k_per_target) if top_k_per_target is not None else -1,
        "min_magnitude": float(min_magnitude),
        # No observed force distribution exists to derive these from yet.
        "status": "placed",
    }
    return targets, sources, retention


def build_head_edges(
    model: str,
    prompt_key: str,
    layer: int,
    head: int,
    X: np.ndarray,
    attention: np.ndarray,
    OV: np.ndarray,
    U_pos=None,
    U_neg=None,
    U_S=None,
    U_A=None,
    induction_pairs: Optional[Sequence] = None,
    strict_pairs: Optional[Sequence] = None,
    same_content_pairs: Optional[Sequence] = None,
    checkpoint_step: Optional[int] = None,
    top_k_per_target: Optional[int] = DEFAULT_TOP_K_PER_TARGET,
    min_magnitude: float = 0.0,
    causal: bool = True,
    declared_frame: str = "raw",
) -> InteractionTable:
    """
    One head's typed edges.

    X          : (n_tokens, d_model) activations entering this layer, in the
                 SAME frame as the projectors.
    attention  : (n_tokens, n_tokens) post-softmax weights for this head.
    OV         : (d_model, d_model) composed W_V @ W_O for this head.
    declared_frame : what the caller believes X is in ("raw" or
                 "l2_sphere"). Checked against what X actually looks like;
                 a mismatch refuses rather than proceeding, because
                 applying residual-basis projectors to normalized
                 activations is invisible in the shapes and produces
                 numbers that look fine.

    The pair sets come from core.battery_structure and are passed through
    to classify_pair_types unchanged — this module does not re-derive which
    pairs are induction pairs, since that logic lives there together with
    the degeneracy checks that decide whether a prompt can carry the test
    at all.
    """
    actual = detect_activation_frame(X)
    if actual != "empty" and actual != declared_frame:
        raise ValueError(
            f"frame mismatch: activations look like {actual!r} but the caller "
            f"declared {declared_frame!r}. Phase 2's projectors live in the "
            "residual-stream basis, so L2-normalized activations are the wrong "
            "input here and nothing about the shapes would reveal it. Refusing "
            "rather than producing numbers that look fine."
        )

    magnitudes, moved = edge_force_magnitudes(X, attention, OV, causal=causal)
    targets, sources, retention = select_edges(
        magnitudes, top_k_per_target=top_k_per_target, min_magnitude=min_magnitude
    )
    retention["frame"] = actual

    if targets.size == 0:
        # A head that moves nothing is a real observation, not an error.
        # Return an empty table carrying the same retention record so it
        # concatenates with its siblings instead of breaking the merge.
        empty = InteractionTable.concat([])
        empty.retention = retention
        return empty

    # Only now are force vectors formed, and only for retained edges.
    force = attention[targets, sources][:, None] * moved[sources]

    pair_type = classify_pair_types(
        targets, sources,
        induction_pairs=induction_pairs,
        strict_pairs=strict_pairs,
        same_content_pairs=same_content_pairs,
    )

    return InteractionTable.from_head(
        model=model, prompt_key=prompt_key, layer=layer, head=head,
        targets=targets, sources=sources,
        weight=attention[targets, sources],
        force=force,
        U_pos=U_pos, U_neg=U_neg, U_S=U_S, U_A=U_A,
        pair_type=pair_type,
        checkpoint_step=checkpoint_step,
        retention=retention,
    )
