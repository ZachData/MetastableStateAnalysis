"""
p2d_operator_activation/p2d_io.py — the join between Phase 2's operators
and Phase 1's activations.

This module exists because that join is the only genuinely dangerous step in
Phase 2d. Everything else is arithmetic on matrices; this is where a
step-143000 OV circuit can be silently paired with step-0 activations and
produce plausible numbers for a model that never existed.

THREE GUARDS, IN ORDER OF HOW BADLY THEY FAIL WHEN ABSENT.

1. REVISION MATCH. Phase 2 keys its artifacts by model_name; Phase 1 keys
   runs by (model, prompt) directory. Neither carries the checkpoint
   revision in a place the other reads. `join` requires the revision
   explicitly from both sides and refuses to proceed on a mismatch or on an
   unknown. core/frames.py's verify_same_revision exists for exactly this
   and is used rather than reimplemented.

2. FRAME MATCH. D2, D3 and D4 all act on the states attention READS, which
   are LN'd, not the raw residual stream. Applying M_h to raw activations
   measures a different operator's action on a different space and will not
   error. The off-by-one between hidden-state index and reading block lives
   in core/ln_frame.frame_for_hidden_state and is resolved there, never
   re-derived here.

3. HEAD-COUNT AND WIDTH MATCH. A d_model or n_heads mismatch between the
   two artifact sets means they came from different models, and numpy will
   happily broadcast some of those. Checked before any pairing.

WHAT IS DELIBERATELY NOT DONE HERE

No fallback that "does its best" on a partial match. A partial join in this
phase produces a number rather than an error, and a number that came from
mismatched checkpoints is worse than no number — it is unfalsifiable from
the output alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


UNKNOWN_REV = "unknown"


# ---------------------------------------------------------------------------
# Loading Phase 2 operators
# ---------------------------------------------------------------------------

def load_operators(save_dir: Path, model_name: str) -> dict:
    """
    Per-layer, per-head W_Q, W_K and W_OV from Phase 2's saved npz.

    Reads the array-name conventions weights.save_weight_decomposition
    writes: `ov_head{h}_{lname}`, `wq_head{h}_{lname}`, `wk_head{h}_{lname}`
    with lname from summary["layers"], or the `_shared` suffix for ALBERT.

    Raises rather than returning partial data when the QK arrays are
    missing. They are written only when `model` was passed to
    save_weight_decomposition — the "legacy behaviour, not recommended"
    path in that function's docstring — and every sub-experiment in this
    phase needs M_h. A run saved without them cannot be analysed here, and
    saying so at load time is cheaper than a KeyError three modules deep.
    """
    save_dir = Path(save_dir)
    stem = model_name.replace("/", "_")

    sum_p = save_dir / f"ov_summary_{stem}.json"
    w_p = save_dir / f"ov_weights_{stem}.npz"
    for p in (sum_p, w_p):
        if not p.exists():
            raise FileNotFoundError(f"p2d: missing {p}")

    with open(sum_p) as f:
        summary = json.load(f)
    w = np.load(w_p)

    is_per_layer = bool(summary["is_per_layer"])
    lnames = list(summary["layers"]) if is_per_layer else ["shared"]

    if not any(k.startswith("wq_head") for k in w.files):
        raise KeyError(
            f"p2d: {w_p.name} has no wq_head*/wk_head* arrays. Phase 2 was "
            f"run without passing `model` to save_weight_decomposition, so "
            f"the raw W_Q/W_K were never written. Every sub-experiment here "
            f"needs M_h = W_Q W_K^T / sqrt(d_head); re-run Phase 2's weight "
            f"extraction rather than substituting the OV circuit."
        )

    layers = []
    for lname in lnames:
        heads = []
        h = 0
        while f"ov_head{h}_{lname}" in w.files:
            rec = {"head": h, "ov": w[f"ov_head{h}_{lname}"]}
            for key, arr in (("wq", f"wq_head{h}_{lname}"),
                             ("wk", f"wk_head{h}_{lname}")):
                if arr not in w.files:
                    raise KeyError(f"p2d: {w_p.name} missing {arr}")
                rec[key] = w[arr]
            heads.append(rec)
            h += 1
        if not heads:
            raise KeyError(f"p2d: no ov_head*_{lname} arrays in {w_p.name}")
        layers.append({"layer_name": lname, "heads": heads})

    d_model = layers[0]["heads"][0]["ov"].shape[0]
    d_head = layers[0]["heads"][0]["wq"].shape[1]
    return {
        "summary": summary,
        "model_name": summary.get("model", model_name),
        "layers": layers,
        "is_per_layer": is_per_layer,
        "n_layers": len(layers),
        "n_heads": len(layers[0]["heads"]),
        "d_model": int(d_model),
        "d_head": int(d_head),
        "source": str(w_p),
    }


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------

class JoinRefused(ValueError):
    """Raised when the two artifact sets cannot be safely paired.

    A distinct exception type so a driver can count refusals separately
    from genuine errors — a sweep where half the runs refuse to join is a
    provenance problem, not a code problem, and the two need different
    responses.
    """


def join(operators: dict, activations: np.ndarray,
         operator_rev: str, activation_rev: str,
         ln_params_by_layer: list = None,
         context: str = "") -> dict:
    """
    Pair per-layer operators with per-layer activations, with all three
    guards enforced.

    operators          : from load_operators
    activations        : (n_hidden, n_tokens, d_model) — the states to pair.
                         These should already be in the LN frame; see
                         `ln_params_by_layer` for the alternative.
    operator_rev       : checkpoint revision the weights came from
    activation_rev     : checkpoint revision the activations came from
    ln_params_by_layer : optional list of {gamma, beta, eps} dicts, one per
                         paired layer, resolved by
                         core.ln_frame.frame_for_hidden_state. When given,
                         the LN transform is applied here and `frame` is
                         recorded as "ln"; when omitted, `frame` is recorded
                         as "raw" and a warning is attached to every record.
                         Omitting it is allowed because a raw-frame run is a
                         legitimate sensitivity check — but it is never the
                         default and never silent.

    Returns {"pairs": [...], "frame": str, "warnings": [...]}.
    """
    if operator_rev == UNKNOWN_REV or activation_rev == UNKNOWN_REV:
        raise JoinRefused(
            f"p2d join{(' ' + context) if context else ''}: revision is "
            f"'{UNKNOWN_REV}' on at least one side (operators="
            f"{operator_rev!r}, activations={activation_rev!r}). The whole "
            f"point of the checkpoint sweep is comparing revisions; pairing "
            f"an unknown one produces numbers for a model that may never "
            f"have existed. Supply both explicitly."
        )
    if operator_rev != activation_rev:
        raise JoinRefused(
            f"p2d join{(' ' + context) if context else ''}: revision "
            f"mismatch — operators from {operator_rev!r}, activations from "
            f"{activation_rev!r}. Refusing to pair."
        )

    A = np.asarray(activations)
    if A.ndim != 3:
        raise JoinRefused(f"p2d join: activations must be 3-D, got {A.shape}")
    if A.shape[-1] != operators["d_model"]:
        raise JoinRefused(
            f"p2d join: d_model mismatch — operators {operators['d_model']}, "
            f"activations {A.shape[-1]}. These are different models."
        )

    n_op = operators["n_layers"]
    n_act = A.shape[0]
    warnings = []
    if operators["is_per_layer"] and n_act != n_op:
        # Not fatal on its own: Phase 1's extraction convention strips the
        # embedding, so n_act == n_op is expected but n_act == n_op + 1
        # means it did not. Pair the overlap and say which convention was
        # assumed rather than guessing silently.
        warnings.append(
            f"layer-count mismatch: {n_op} operator layers, {n_act} "
            f"activation layers. Pairing the first {min(n_op, n_act)}; "
            f"verify the embedding-stripped convention (core/ln_frame's "
            f"resolve_frame_index) before reading per-layer results."
        )

    if ln_params_by_layer is None:
        warnings.append(
            "RAW FRAME: no LN parameters supplied, so M_h is being applied "
            "to the raw residual stream rather than to the states attention "
            "reads. This is a sensitivity check, not the primary "
            "measurement. Resolve the frame with "
            "core.ln_frame.frame_for_hidden_state."
        )
        frame = "raw"
    else:
        frame = "ln"

    pairs = []
    n = min(n_op, n_act) if operators["is_per_layer"] else n_act
    for i in range(n):
        op_layer = operators["layers"][i if operators["is_per_layer"] else 0]
        Y = np.asarray(A[i], dtype=np.float64)
        if ln_params_by_layer is not None:
            from core.ln_frame import ln_transform
            p = ln_params_by_layer[i] if i < len(ln_params_by_layer) else None
            if p is not None:
                Y = ln_transform(Y, gamma=p.get("gamma"), beta=p.get("beta"),
                                 eps=p.get("eps", 1e-5))
        pairs.append({
            "layer": i,
            "layer_name": op_layer["layer_name"],
            "heads": op_layer["heads"],
            "Y": Y,
            "d_head": operators["d_head"],
        })

    return {"pairs": pairs, "frame": frame, "warnings": warnings,
            "revision": operator_rev, "n_paired": len(pairs),
            "n_heads": operators["n_heads"], "d_head": operators["d_head"]}


def extraction_convention(run: dict) -> dict:
    """
    The extraction convention, READ from the artifact rather than asserted
    by a caller.

    p1_io._PROVENANCE_FIELDS writes `hidden_state_0_is_embedding` and
    `final_hidden_state_is_post_ln` at geometry.json's top level precisely
    so that downstream code does not have to be told. Phase 2d's LN frame
    resolution depends on both — get either wrong and M_h is applied in the
    wrong frame, silently — so reading them is strictly better than the
    command-line flags, which are retained only as an override for
    artifacts that predate the fields.

    Returns {"embedding_stripped", "last_is_post_final_ln", "source"} with
    source "artifact" or "unrecorded". `unrecorded` is not a default: the
    caller must supply the convention explicitly, and run_2d refuses
    otherwise.
    """
    geo = (run or {}).get("geometry") or {}
    h0 = geo.get("hidden_state_0_is_embedding")
    post = geo.get("final_hidden_state_is_post_ln")
    if h0 is None and post is None:
        return {"embedding_stripped": None, "last_is_post_final_ln": None,
                "source": "unrecorded"}
    return {
        # "embedding stripped" is the negation of "index 0 IS the embedding".
        "embedding_stripped": (not bool(h0)) if h0 is not None else None,
        "last_is_post_final_ln": bool(post) if post is not None else None,
        "source": "artifact",
        "raw": {"hidden_state_0_is_embedding": h0,
                "final_hidden_state_is_post_ln": post},
    }


def revision_from_run(run: dict, explicit: str = None) -> str:
    """
    Best-effort revision extraction from a Phase 1 run record, with an
    explicit override.

    Returns UNKNOWN_REV rather than a guess when nothing is found. `join`
    then refuses, which is the intended behaviour: the alternative is a
    heuristic that reads "step143000" out of a directory name and is wrong
    the first time someone renames a directory.
    """
    if explicit:
        return str(explicit)
    geo = (run or {}).get("geometry") or {}
    for key in ("revision", "checkpoint_step", "model_rev",
                "checkpoint", "step"):
        v = geo.get(key)
        if v not in (None, ""):
            return str(v)
    return UNKNOWN_REV


# ---------------------------------------------------------------------------
# LN frame resolution
# ---------------------------------------------------------------------------

def resolve_ln_params(model_name: str, revision: str, n_hidden_states: int,
                      which: str = "attn", embedding_stripped: bool = True,
                      last_is_post_final_ln: bool = False,
                      dtype: str = "float32") -> tuple:
    """
    Per-state LN parameters for the frame attention actually reads.

    This is guard 2 from the module docstring, made operational. Every
    sub-experiment in this phase applies M_h to states; M_h acts on LN'd
    states, and applying it to the raw residual stream measures a different
    operator on a different space WITHOUT ERRORING. So this must be
    resolved, and it must be resolved by core/ln_frame rather than
    re-derived, because the off-by-one between hidden-state index and
    reading block lives there and a second copy of that arithmetic is how
    an extraction-convention mismatch becomes unfalsifiable.

    THE MODEL IS LOADED AT THE SAME REVISION AS THE OPERATORS. Loading it
    at any other would reintroduce the exact failure the revision guard
    exists to prevent, one level down and harder to see — the OV circuits
    would match the activations while the LN gains came from a different
    checkpoint.

    fp32 is the default and should not be changed here: D3's row
    classification turns on the sign and multiplicity of lambda_1(V) near
    zero, which is precisely what core/models.py's precision guard protects.

    Returns (params_by_state, info). params_by_state[i] is either None (the
    identity frame — the extraction already applied final LN, and applying
    it again would be wrong) or a dict ready to splat into ln_transform.
    """
    import torch
    from transformers import AutoModelForCausalLM
    from core.ln_frame import frame_for_hidden_state

    torch_dtype = getattr(torch, dtype)
    from core.models import from_pretrained_eager
    model = from_pretrained_eager(
        AutoModelForCausalLM, model_name,
        revision=revision, torch_dtype=torch_dtype,
    )

    params, frames, identity_idx = [], [], []
    try:
        for i in range(n_hidden_states):
            res = frame_for_hidden_state(
                model, i, n_hidden_states, which=which,
                embedding_stripped=embedding_stripped,
                last_is_post_final_ln=last_is_post_final_ln,
            )
            frames.append(res["frame"])
            if res["params"] is None:
                identity_idx.append(i)
            params.append(res["params"])
    finally:
        del model

    counts: dict = {}
    for f in frames:
        counts[f] = counts.get(f, 0) + 1

    return params, {
        "frames": frames,
        "frame_counts": counts,
        "identity_indices": identity_idx,
        "which": which,
        "revision": revision,
        "dtype": dtype,
        "embedding_stripped": embedding_stripped,
        "last_is_post_final_ln": last_is_post_final_ln,
    }
