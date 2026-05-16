"""
run_6.py — Phase 6 orchestrator with comprehensive bug fixes.

Fixes applied
-------------
Fix  #1  : _find_p2_weights_path — try hyphen stem then underscore stem.
Fix  #2  : _load_ov_weights — match ov_head* prefix, numeric head sort,
           detect per-layer vs shared.
Fix  #3  : _infer_d_model — raise KeyError on miss, no silent 768.
Fix  #4  : _build_or_load_projectors — cross-check d_model via _infer_d_model;
           remove manual layer_names / is_per_layer overrides.
Bug  W1  : _compute_qk_logit_matrices — compute X @ M_h @ X.T per head.
Bug  W3  : _normalise_empty_lists — coerce empty lists to None.
Bug  W2  : _classify_layer_types / _layer_idx_from_name — normalise
           layer name forms; add plateau_windows param.
Bug  10  : _attach_phase2_artifacts_to_ctx — rename wo_matrices →
           head_write_matrices; back-compat alias; per-layer clamping.
Bug  11  : (see W2)
FIX-R1   : _get_attention_output_modules — hook dense before residual add,
           not whole SelfOutput.
FIX-R2   : _INDUCTION_CANDIDATE_TEXTS + _select_input_text — live induction
           score check; replace static fallback sentence.
"""

import argparse
import json
import re
import re as _re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from p6_subspace.p6_io import (
    SubexperimentSpec,
    run_phase6,
    _jsonify,
)
from p6_subspace.subspace_build import (
    build_global_projectors,
    save_projectors,
    load_projectors,
    print_projector_summary,
)
from p6_subspace.head_classify import classify_heads, head_map_data
from p6_subspace.qk_decompose import run_qk_decompose
from p6_subspace.induction_ov import run_induction_ov
from p6_subspace.eigenspace_degeneracy import run_eigenspace_degeneracy
from p6_subspace.centroid_velocity import run_centroid_velocity
from p6_subspace.local_contraction import run_local_contraction
from p6_subspace.probe_subspace import run_probe_subspace
from p6_subspace.write_subspace import run_write_subspace
import p6_subspace.report_6 as report_6


# =============================================================================
# Fix #1 — Path resolution
# =============================================================================

def _find_p2_weights_path(phase2_dir: Path, model_name: str) -> Path | None:
    """
    Locate the Phase 2 ov_weights NPZ for ``model_name``.

    Phase 2's writer uses ``stem = model_name.replace("/", "_")`` — hyphens
    preserved.  Tries hyphen form first (canonical), then underscore form
    (legacy).  Returns None when neither file exists; caller decides how to
    fail.
    """
    hyphen_stem     = model_name.replace("/", "_")
    underscore_stem = hyphen_stem.replace("-", "_")
    for stem in (hyphen_stem, underscore_stem):
        candidate = phase2_dir / f"ov_weights_{stem}.npz"
        if candidate.exists():
            return candidate
    return None


# =============================================================================
# Fix #2 — _load_ov_weights
# =============================================================================

def _load_ov_weights(path: Path) -> dict:
    """
    Load per-head OV (and optional QK) matrices from a Phase 2 NPZ.

    Phase 2's writer (weights.py::save_weight_decomposition) stores:

        Shared-weight models (ALBERT):
            ov_total_shared             — (d, d) summed OV
            ov_head{h}_shared           — (d, d) per head h

        Per-layer models (GPT-2, BERT):
            ov_total_layer_{i}          — (d, d) summed OV for layer i
            ov_head{h}_layer_{i}        — (d, d) per (layer i, head h)

    The matrices are the *composed* OV = W_V_h @ W_O_h, not W_O alone.
    ``ctx["wo_matrices"]`` is misnamed for backward compatibility but holds
    OV.  See note in _attach_phase2_artifacts_to_ctx.

    Returns
    -------
    dict with keys:
        is_per_layer : bool
        ov_per_head  : list[(d, d)]            — shared
                       list[list[(d, d)]]      — per-layer, indexed [layer][head]
        layer_names  : list[str]
        n_heads      : int
        d_model      : int
        qk_per_head  : optional list[(W_Q, W_K)]
        rot_energy_fracs : optional list[float]

    Raises
    ------
    KeyError if no ov_head* keys are present.
    """
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    is_per_layer = any(k.startswith("ov_total_layer_") for k in keys)

    out: dict = {"is_per_layer": is_per_layer}

    if is_per_layer:
        layer_indices = sorted({
            int(k.split("_layer_")[1])
            for k in keys
            if k.startswith("ov_head") and "_layer_" in k
        })
        if not layer_indices:
            raise KeyError(
                f"{path}: ov_total_layer_* present but no ov_head*_layer_* keys"
            )

        layer_names: list[str] = [f"layer_{i}" for i in layer_indices]
        per_layer_heads: list[list[np.ndarray]] = []

        for i in layer_indices:
            hkeys = [
                k for k in keys
                if k.startswith("ov_head") and k.endswith(f"_layer_{i}")
            ]
            # Sort by integer head index, NOT lex (head 10 must follow head 9)
            hkeys.sort(key=lambda k: int(k[len("ov_head"):k.index("_layer_")]))
            per_layer_heads.append([data[k].astype(np.float64) for k in hkeys])

        if not per_layer_heads or not per_layer_heads[0]:
            raise KeyError(f"{path}: per-layer keys present but empty after parse")

        ov_per_head = per_layer_heads
        n_heads     = len(per_layer_heads[0])
        d_model     = per_layer_heads[0][0].shape[0]

    else:
        head_keys = [
            k for k in keys
            if k.startswith("ov_head") and k.endswith("_shared")
        ]
        if not head_keys:
            raise KeyError(
                f"{path}: no ov_head*_shared keys found. "
                f"Sample keys: {sorted(keys)[:8]}"
            )
        head_keys.sort(key=lambda k: int(k[len("ov_head"):k.index("_shared")]))
        ov_per_head = [data[k].astype(np.float64) for k in head_keys]
        layer_names = ["shared"]
        n_heads     = len(ov_per_head)
        d_model     = ov_per_head[0].shape[0]

    out["ov_per_head"] = ov_per_head
    out["layer_names"] = layer_names
    out["n_heads"]     = n_heads
    out["d_model"]     = d_model

    # Optional QK pairs
    wq_keys = sorted(k for k in keys if "wq_head" in k or "W_Q_head" in k)
    wk_keys = sorted(k for k in keys if "wk_head" in k or "W_K_head" in k)
    if wq_keys and wk_keys and len(wq_keys) == len(wk_keys):
        out["qk_per_head"] = [
            (data[q], data[k]) for q, k in zip(wq_keys, wk_keys)
        ]

    if "rot_energy_fracs" in keys:
        out["rot_energy_fracs"] = data["rot_energy_fracs"].tolist()

    return out


# =============================================================================
# Fix #3 — _infer_d_model
# =============================================================================

def _infer_d_model(p2_weights: Path) -> int:
    """
    Infer d_model from a Phase 2 ov_weights NPZ.

    Probes (in order):
        1. ov_total_shared             — ALBERT
        2. ov_total_layer_{lowest}     — per-layer
        3. ov_head0_shared             — fallback if ov_total absent
        4. ov_head0_layer_{lowest}     — same, per-layer

    Raises
    ------
    KeyError when none of the above are present.  The previous fallback of
    ``return 768`` silently propagated the wrong shape into every projector
    when the file was malformed.
    """
    data = np.load(p2_weights, allow_pickle=True)
    keys = set(data.keys())

    if "ov_total_shared" in keys:
        return int(data["ov_total_shared"].shape[0])

    layer_total_keys = sorted(
        (k for k in keys if k.startswith("ov_total_layer_")),
        key=lambda k: int(k.split("_layer_")[1]),
    )
    if layer_total_keys:
        return int(data[layer_total_keys[0]].shape[0])

    if "ov_head0_shared" in keys:
        return int(data["ov_head0_shared"].shape[0])

    layer_head0 = sorted(
        (k for k in keys if k.startswith("ov_head0_layer_")),
        key=lambda k: int(k.split("_layer_")[1]),
    )
    if layer_head0:
        return int(data[layer_head0[0]].shape[0])

    raise KeyError(
        f"{p2_weights}: cannot infer d_model — no ov_total_* or ov_head0_* keys. "
        f"Sample keys: {sorted(keys)[:8]}"
    )


# =============================================================================
# Fix #4 — _build_or_load_projectors
# =============================================================================

def _build_or_load_projectors(
    out_dir: Path, phase2_dir: Path, model_name: str
) -> dict | None:
    """
    Build (or load from cache) global S/A projectors.

    Cross-checks d_model between ov_total_* and per-head keys via
    _infer_d_model.  Returns the projector dict, or None when Phase 2
    weights are missing.
    """
    proj_path = out_dir / "projectors.npz"
    if proj_path.exists():
        print("Loading cached projectors...")
        return load_projectors(proj_path)

    print("Building global S/A projectors...")
    p2_weights = _find_p2_weights_path(phase2_dir, model_name)
    if p2_weights is None:
        print(
            f"  ERROR: Phase 2 weights not found in {phase2_dir} for "
            f"{model_name!r}. Tried both hyphen and underscore stem forms."
        )
        return None

    try:
        ov_data = _load_ov_weights(p2_weights)
    except KeyError as e:
        print(f"  ERROR: {e}")
        return None

    if not ov_data.get("ov_per_head"):
        print(f"  ERROR: no per-head OV matrices in {p2_weights}")
        return None

    # Cross-check d_model against ov_total when present
    try:
        d_total = _infer_d_model(p2_weights)
        if d_total != ov_data["d_model"]:
            print(
                f"  WARNING: d_model mismatch in {p2_weights.name} — "
                f"per-head says {ov_data['d_model']}, ov_total says {d_total}. "
                f"Using per-head value."
            )
    except KeyError:
        pass

    print(
        f"  loaded {ov_data['n_heads']} heads "
        f"× {len(ov_data['layer_names'])} layer(s) "
        f"@ d_model={ov_data['d_model']} "
        f"({'per-layer' if ov_data['is_per_layer'] else 'shared'})"
    )

    projectors = build_global_projectors(ov_data)
    save_projectors(projectors, proj_path)
    print_projector_summary(projectors)
    return projectors


# =============================================================================
# Bug 10 — _attach_phase2_artifacts_to_ctx
# =============================================================================

def _attach_phase2_artifacts_to_ctx(
    ctx: dict, phase2_dir: Path, model_name: str, layer_idx: int
) -> None:
    """
    Populate ctx["head_write_matrices"] (and back-compat ctx["wo_matrices"]),
    ctx["qk_matrices"], ctx["qk_logit_matrices"], ctx["rot_energy_fracs"]
    from Phase 2 outputs.

    NB: ctx["wo_matrices"] is a misnomer kept for back-compat — it holds the
    composed OV = W_V_h @ W_O_h, not W_O alone.  Consumers that compute
    column-space metrics (write_subspace, induction_ov) get the same answer
    up to the rank of W_V_h.  If a consumer ever needs raw W_O, add a
    separate ctx key rather than redefining this one.

    For per-layer models, slices out the matrices for layer_idx (clamped to
    valid range).  For shared models, uses the single shared set directly.
    """
    p2_weights = _find_p2_weights_path(phase2_dir, model_name)

    if p2_weights is None:
        ctx["head_write_matrices"] = None
        ctx["wo_matrices"]         = None
        ctx["qk_matrices"]         = None
        ctx["qk_logit_matrices"]   = None
        ctx["rot_energy_fracs"]    = None
        return

    try:
        ov_data = _load_ov_weights(p2_weights)
    except KeyError as e:
        print(f"  WARNING: could not load Phase 2 weights ({e})")
        ctx["head_write_matrices"] = None
        ctx["wo_matrices"]         = None
        ctx["qk_matrices"]         = None
        ctx["qk_logit_matrices"]   = None
        ctx["rot_energy_fracs"]    = None
        return

    if ov_data["is_per_layer"]:
        n_layers = len(ov_data["ov_per_head"])
        li = max(0, min(layer_idx, n_layers - 1))
        if li != layer_idx:
            print(
                f"  WARNING: layer_idx={layer_idx} out of range "
                f"[0, {n_layers}); clamped to {li}."
            )
        ctx["head_write_matrices"] = ov_data["ov_per_head"][li]

        qk = ov_data.get("qk_per_head")
        if isinstance(qk, list) and qk and isinstance(qk[0], list):
            qk_li = max(0, min(layer_idx, len(qk) - 1))
            ctx["qk_matrices"] = qk[qk_li]
        else:
            ctx["qk_matrices"] = qk or None
    else:
        ctx["head_write_matrices"] = ov_data["ov_per_head"]
        ctx["qk_matrices"]         = ov_data.get("qk_per_head") or None

    # Back-compat alias — remove in the next refactor pass
    ctx["wo_matrices"]       = ctx["head_write_matrices"]
    ctx["qk_logit_matrices"] = None   # populated downstream by _compute_qk_logit_matrices
    ctx["rot_energy_fracs"]  = ov_data.get("rot_energy_fracs")


# =============================================================================
# Bug W1 — _compute_qk_logit_matrices
# =============================================================================

def _compute_qk_logit_matrices(ctx: dict) -> dict:
    """
    Compute per-head QK logit matrices from qk_matrices + token_activations.

    For each head h with weight pair (WQ_h, WK_h):
        M_h        = WQ_h @ WK_h.T          # (d, d) in residual-stream space
        logit_h[i,j] = x_i.T @ M_h @ x_j   = (X @ M_h @ X.T)[i, j]

    Sets ctx["qk_logit_matrices"] in-place and returns ctx.
    Leaves ctx["qk_logit_matrices"] = None when qk_matrices or
    token_activations are absent.
    """
    qk = ctx.get("qk_matrices")
    X  = ctx.get("token_activations")

    if not qk or X is None:
        ctx["qk_logit_matrices"] = None
        return ctx

    logit_mats = []
    for WQ, WK in qk:
        M = WQ @ WK.T                                    # (d, d)
        logit_mats.append((X @ M @ X.T).astype(np.float32))  # (n, n)

    ctx["qk_logit_matrices"] = logit_mats
    return ctx


# =============================================================================
# Bug W3 — _normalise_empty_lists
# =============================================================================

def _normalise_empty_lists(ctx: dict) -> dict:
    """
    Coerce empty-list weight keys to None so prerequisites_met gates them.

    SubexperimentSpec.prerequisites_met checks ``ctx.get(k) is None``.
    An empty list [] is not None, passes the gate, and causes sub-experiments
    to receive zero matrices — either silent failure or a crash.
    """
    for key in ("qk_matrices", "wo_matrices", "head_write_matrices",
                "qk_logit_matrices", "attn_matrices"):
        if key in ctx and isinstance(ctx[key], list) and len(ctx[key]) == 0:
            ctx[key] = None
    return ctx


# =============================================================================
# Bug W2 / Bug 11 — _layer_idx_from_name + _classify_layer_types
# =============================================================================

_LAYER_PREFIX_RE = _re.compile(r"^(?:iter|layer)[_-]?(\d+)$")


def _layer_idx_from_name(name) -> int | None:
    """
    Extract a canonical integer layer index from any of the common forms:
        "iter_2", "iter-2", "layer_2", "layer-2", "2", 2, "02"
    Returns None if the name doesn't decode.
    """
    if name is None:
        return None
    if isinstance(name, (int, np.integer)):
        return int(name)
    s = str(name).strip()
    if not s:
        return None
    m = _LAYER_PREFIX_RE.match(s)
    if m is not None:
        return int(m.group(1))
    try:
        return int(s)
    except ValueError:
        return None


def _classify_layer_types(
    layer_names:     list,
    events:          list[dict],
    trajectories:    list[dict] | None = None,
    plateau_windows: list[dict] | None = None,
) -> list[str]:
    """
    Label each layer as one of: "merge", "plateau", "transition", "other".

    Both event names and layer_names are normalised to integer indices via
    _layer_idx_from_name before comparison.  This fixes the silent
    name-mismatch bug where "iter_2" never matched "2" in merge_layers.

    When plateau_windows is supplied, ONLY layers inside an inclusive window
    range receive the "plateau" label; everything non-merge outside a window
    is labelled "transition".  When omitted, falls back to legacy behaviour
    (all non-merge → "plateau") so existing tests keep passing.
    """
    # Build merge index set from events
    merge_indices: set[int] = set()
    for ev in events or []:
        if ev.get("type") != "merge":
            continue
        idx = _layer_idx_from_name(ev.get("layer_from"))
        if idx is None:
            idx = _layer_idx_from_name(ev.get("layer_name"))
        if idx is not None:
            merge_indices.add(idx)

    # Build plateau index set from windows (if given)
    plateau_indices: set[int] | None
    if plateau_windows:
        plateau_indices = set()
        for w in plateau_windows:
            try:
                start = int(w["start"])
                end   = int(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            plateau_indices.update(range(start, end + 1))
    else:
        plateau_indices = None   # signal: legacy fallback

    # Classify each layer
    types: list[str] = []
    for list_pos, lname in enumerate(layer_names):
        name_idx      = _layer_idx_from_name(lname)
        canonical_idx = name_idx if name_idx is not None else list_pos

        if canonical_idx in merge_indices:
            types.append("merge")
        elif plateau_indices is None:
            types.append("plateau")
        elif canonical_idx in plateau_indices:
            types.append("plateau")
        else:
            types.append("transition")

    return types


# =============================================================================
# Context assembly
# =============================================================================

def build_context(
    model_name:  str,
    phase1_dir:  Path,
    phase2_dir:  Path,
    out_dir:     Path,
    projectors:  dict,
    load_model:  bool = False,
    prompt_key:  str  = "wiki_paragraph",
    layer_idx:   int  = 0,
) -> dict:
    """
    Assemble the shared context dict for one model.

    Loads Phase 1 activations and Phase 2 weight matrices.
    Applies all W1 / W3 / W2 / Bug-10 fixes.
    """
    from core.io import find_phase1_run_dir, load_phase1_run

    stem = model_name.replace("/", "_").replace("-", "_")

    ctx: dict = {
        "model_name": model_name,
        "stem":       stem,
        "out_dir":    out_dir,
        "projectors": projectors,
        "layer_name": projectors["layer_names"][
            min(layer_idx, len(projectors["layer_names"]) - 1)
        ],
        "layer_idx":  layer_idx,
        "load_model": load_model,
    }

    # --- Phase 1 artifacts ---
    p1_run_dir = find_phase1_run_dir(Path(phase1_dir), model_name, prompt_key)

    if p1_run_dir is not None and p1_run_dir.exists():
        p1 = load_phase1_run(p1_run_dir)

        ctx["tokens"]    = p1["tokens"]
        ctx["token_ids"] = np.array(
            [hash(t) % (2 ** 31) for t in p1["tokens"]], dtype=np.int64
        )

        if p1["activations"] is not None:
            ctx["activations_per_layer"] = [
                p1["activations"][L] for L in range(p1["activations"].shape[0])
            ]
            ctx["layer_names"] = [
                str(L) for L in range(p1["activations"].shape[0])
            ]
        else:
            ctx["activations_per_layer"] = None
            ctx["layer_names"]           = []

        ctx["labels_per_layer"] = p1.get("hdbscan_labels")

        events             = p1.get("events", [])
        ctx["merge_events"] = events

        ctx["layer_type_labels"] = _classify_layer_types(
            ctx["layer_names"],
            events,
            p1.get("trajectories", []),
            plateau_windows=p1.get("plateau_windows"),
        )

        if ctx.get("activations_per_layer"):
            safe = min(layer_idx, len(ctx["activations_per_layer"]) - 1)
            ctx["token_activations"] = ctx["activations_per_layer"][safe]

        if p1.get("attentions") is not None:
            A    = p1["attentions"]   # (n_layers, n_heads, n, n)
            safe = min(layer_idx, A.shape[0] - 1)
            ctx["attn_matrices"] = [A[safe, h] for h in range(A.shape[1])]
        else:
            ctx["attn_matrices"] = None

    # --- Phase 2 weight artifacts (Bug 10) ---
    # Sets head_write_matrices, wo_matrices (back-compat alias), qk_matrices,
    # qk_logit_matrices = None, rot_energy_fracs.
    _attach_phase2_artifacts_to_ctx(ctx, Path(phase2_dir), model_name, layer_idx)

    # Bug W1: compute QK logit matrices from qk_matrices + token activations
    ctx = _compute_qk_logit_matrices(ctx)

    # Bug W3: coerce empty lists to None so prerequisites_met gates correctly
    ctx = _normalise_empty_lists(ctx)

    return ctx


# =============================================================================
# Registry & sub-experiment wrappers
# =============================================================================

def _run_head_classify(ctx: dict):
    from p6_subspace.p6_io import SubResult, SEP_THICK, _bullet, _verdict_line, _fmt

    records    = classify_heads(
        ctx["attn_matrices"],
        ctx["qk_logit_matrices"],
        ctx["token_activations"],
        ctx.get("rot_energy_fracs"),
    )
    layer_name = ctx.get("layer_name", "shared")
    map_data   = head_map_data(records, layer_name)
    corr       = map_data["cross_head_corr"]

    lines = [
        SEP_THICK,
        "HEAD CLASSIFICATION: CC/PC 2D MAP  [Track A]",
        SEP_THICK,
        f"Layer:       {layer_name}",
        f"Heads:       {map_data['n_heads']}",
        "",
        "Quadrant counts:",
    ]
    for q, n in sorted(map_data["quadrant_counts"].items()):
        lines.append(f"  {q:<18s} {n}")
    lines += [
        "",
        f"Anti-similarity heads:  {map_data['anti_sim_heads']}",
        f"Positional/induction:   {map_data['positional_heads']}",
        "",
        "P6-A2: f_rot(h) negatively correlated with CC and positively with |PC|?",
        _bullet("Spearman ρ(f_rot, -CC)",   corr.get("rho_frot_neg_cc")),
        _bullet("Spearman p-value (CC)",     corr.get("p_value_neg_cc")),
        _bullet("Spearman ρ(f_rot, |PC|)",  corr.get("rho_frot_abs_pc")),
        _bullet("Spearman p-value (PC)",     corr.get("p_value_abs_pc")),
        _bullet("n_heads in correlation",   corr.get("n_heads")),
        _verdict_line(
            "P6-A2",
            corr.get("p6_a2_passes"),
            f"ρ(-CC)={_fmt(corr.get('rho_frot_neg_cc'))} "
            f"ρ(|PC|)={_fmt(corr.get('rho_frot_abs_pc'))}"
            f" (p < α AND ρ > threshold)",
        ),
    ]

    vc = {
        "hc_rho_frot_neg_cc":    corr.get("rho_frot_neg_cc"),
        "hc_p_value_neg_cc":     corr.get("p_value_neg_cc"),
        "hc_rho_frot_abs_pc":    corr.get("rho_frot_abs_pc"),
        "hc_p_value_abs_pc":     corr.get("p_value_abs_pc"),
        "hc_p6_a2_passes":       corr.get("p6_a2_passes"),
        "hc_n_anti_sim_heads":   len(map_data["anti_sim_heads"]),
        "hc_n_positional_heads": len(map_data["positional_heads"]),
    }

    return SubResult(
        name="head_classify",
        applicable=True,
        payload=map_data,
        summary_lines=lines,
        verdict_contribution=vc,
    )


def _run_dissociation_gated(ctx: dict):
    from p6_subspace.dissociation import run_dissociation
    return run_dissociation(ctx)


REGISTRY: list[SubexperimentSpec] = [
    # Track A — weights only
    SubexperimentSpec(
        name="head_classify",
        run=_run_head_classify,
        requires=["attn_matrices", "qk_logit_matrices", "token_activations"],
    ),
    SubexperimentSpec(
        name="qk_decompose",
        run=run_qk_decompose,
        requires=["qk_matrices", "token_ids", "token_activations"],
    ),
    SubexperimentSpec(
        name="induction_ov",
        run=run_induction_ov,
        requires=[
            "attn_matrices", "wo_matrices", "token_ids",
            "token_activations", "projectors",
        ],
    ),
    # Track B/D — activations
    SubexperimentSpec(
        name="eigenspace_degeneracy",
        run=run_eigenspace_degeneracy,
        requires=[
            "activations_per_layer", "labels_per_layer",
            "layer_type_labels", "projectors",
        ],
    ),
    SubexperimentSpec(
        name="centroid_velocity",
        run=run_centroid_velocity,
        requires=[
            "activations_per_layer", "labels_per_layer",
            "layer_type_labels", "projectors",
        ],
    ),
    SubexperimentSpec(
        name="local_contraction",
        run=run_local_contraction,
        requires=["activations_per_layer", "labels_per_layer", "layer_type_labels"],
    ),
    SubexperimentSpec(
        name="probe_subspace",
        run=run_probe_subspace,
        requires=[
            "activations_per_layer", "labels_per_layer",
            "layer_type_labels", "projectors",
        ],
    ),
    # Track C
    SubexperimentSpec(
        name="write_subspace",
        run=run_write_subspace,
        requires=["wo_matrices", "projectors"],
    ),
    SubexperimentSpec(
        name="dissociation",
        run=_run_dissociation_gated,
        requires=["model", "tokenizer", "text", "token_ids", "projectors", "hook_targets"],
        applicable=lambda ctx: ctx.get("load_model", False),
    ),
]


# =============================================================================
# Model loading
# =============================================================================

def _load_model_and_tokenizer(model_name: str, device: str):
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        output_attentions=True,
        output_hidden_states=True,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _get_attention_output_modules(model) -> list:
    """
    FIX-R1: Identify attention output projection modules to hook for
    dissociation.

    Targets the ``dense`` sub-layer *inside* each SelfOutput module so the
    hook fires on the raw attention projection output, before the residual add
    and LayerNorm:

        BertSelfOutput.forward:
            h = self.dense(h_attn)        # ← hook fires HERE
            h = self.dropout(h)
            h = self.LayerNorm(h + prev)  # ← old (wrong) hook fired after HERE

    For GPT-2 / GPT-Neo / GPT-J the output projection already fires before the
    residual add, so those paths are unchanged.

    Returns an empty list (with a warning) if nothing matches.
    """
    targets = []
    seen_dense: set[str] = set()

    for name, module in model.named_modules():
        cls = type(module).__name__

        # BERT / ALBERT / RoBERTa — hook .dense inside SelfOutput
        if cls.endswith("SelfOutput"):
            if hasattr(module, "dense"):
                dense_name = f"{name}.dense"
                if dense_name not in seen_dense:
                    seen_dense.add(dense_name)
                    targets.append(module.dense)
            else:
                print(
                    f"  WARNING: SelfOutput at '{name}' has no .dense attribute; "
                    "skipping for hook placement."
                )
            continue

        # GPT-2: c_proj inside the attention block
        if "attn" in name and name.endswith(".c_proj"):
            targets.append(module)
            continue

        # GPT-Neo / GPT-J: out_proj
        if "attention" in name and name.endswith(".out_proj"):
            targets.append(module)
            continue

    if not targets:
        print(
            f"  WARNING: _get_attention_output_modules found no hook targets for "
            f"{type(model).__name__}.  "
            "Dissociation will run with an empty hook list (no-op interventions)."
        )
    else:
        print(
            f"  Hook targets: {len(targets)} modules "
            f"(dense projections before residual add)"
        )

    return targets


# =============================================================================
# FIX-R2 — _select_input_text with live induction score verification
# =============================================================================

_MIN_INDUCTION_SCORE = 0.05

_INDUCTION_CANDIDATE_TEXTS = [
    # Repeated short noun phrases — "the cat", "the mat", etc. appear 2-3×
    (
        "The cat sat on the mat . The dog sat on the log . "
        "The cat sat on the mat again and the dog sat on the log again ."
    ),
    # Repeated function-word bigrams
    (
        "She said that she knew that he knew that they knew that we knew ."
    ),
    # Repeated subject–verb patterns
    (
        "The model predicts the token . The model attends to the token . "
        "The model predicts the next token based on the token it attended to ."
    ),
    # Deliberate AB…AB induction pattern
    (
        "apple banana cherry apple banana cherry apple banana cherry "
        "delta echo foxtrot delta echo foxtrot"
    ),
]


def _select_input_text(
    ctx: dict,
    model=None,
    tokenizer=None,
    device: str = "cpu",
) -> str:
    """
    FIX-R2: Choose an input text for the dissociation forward pass.

    The previous static fallback ("The researchers found that the model…")
    contained no repeated token bigrams, making the induction score
    structurally near zero regardless of the intervention.  A zero baseline
    means DD1 (induction drops after zeroing the imaginary channel) can never
    confirm.

    Selection order
    ---------------
    1. ctx["tokens"] reconstructed from Phase 1, if it passes a bigram check.
    2. Candidates from _INDUCTION_CANDIDATE_TEXTS, verified live by running a
       baseline forward pass and measuring the induction score.  The first
       candidate with score > _MIN_INDUCTION_SCORE is returned.
    3. If no candidate clears the threshold, use the best-scoring candidate
       and emit a WARNING so the operator knows DD1 may be INDETERMINATE.
    4. If model/tokenizer are unavailable, return the first candidate with a
       WARNING that live verification was skipped.
    """
    from p6_subspace.dissociation import run_intervened_forward, measure_induction_score

    # --- Option 1: Phase 1 tokens ---
    tokens = ctx.get("tokens")
    if tokens:
        reconstructed = " ".join(
            t.lstrip("##") for t in tokens
            if t not in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>")
        )
        words   = reconstructed.split()
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        if len(words) >= 8 and len(set(bigrams)) < len(bigrams):
            return reconstructed

    # --- Option 2: live-verified candidates ---
    if model is not None and tokenizer is not None:
        best_text  = _INDUCTION_CANDIDATE_TEXTS[0]
        best_score = -1.0

        for candidate in _INDUCTION_CANDIDATE_TEXTS:
            try:
                enc    = tokenizer(candidate, return_tensors="pt").to(device)
                tids   = enc["input_ids"][0].cpu().numpy()
                result = run_intervened_forward(
                    model, tokenizer, candidate, None, [], device
                )
                score  = measure_induction_score(result["attentions"], tids)
                if score is None:
                    score = 0.0
                if score > best_score:
                    best_score = score
                    best_text  = candidate
                if score >= _MIN_INDUCTION_SCORE:
                    print(
                        f"  Induction text selected (score={score:.3f}): "
                        f"{candidate[:60]}…"
                    )
                    return candidate
            except Exception as exc:
                print(f"  WARNING: candidate text check failed ({exc}); skipping.")
                continue

        print(
            f"  WARNING [_select_input_text]: no candidate text produced "
            f"induction score > {_MIN_INDUCTION_SCORE} "
            f"(best={best_score:.3f}).  "
            "Using best available — DD1 verdict may be INDETERMINATE.  "
            "Consider supplying a custom text via ctx['tokens'] or --prompt."
        )
        return best_text

    # --- Option 3: fallback without live verification ---
    print(
        "  WARNING [_select_input_text]: model not available for live induction "
        "score check.  Using static candidate; DD1 reliability is unverified."
    )
    return _INDUCTION_CANDIDATE_TEXTS[0]


# =============================================================================
# Main entry
# =============================================================================

def run_one_model(
    model_name:  str,
    phase1_dir:  Path,
    phase2_dir:  Path,
    out_dir:     Path,
    tracks:      str  = "all",
    load_model:  bool = False,
    prompt_key:  str  = "wiki_paragraph",
) -> None:
    import torch

    stem    = model_name.replace("/", "_").replace("-", "_")
    out_dir = Path(out_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"Phase 6 — {model_name}")
    print(f"{'='*64}")

    # 1. Build or load projectors
    projectors = _build_or_load_projectors(out_dir, Path(phase2_dir), model_name)
    if projectors is None:
        return

    # 2. Assemble context
    ctx = build_context(
        model_name,
        Path(phase1_dir),
        Path(phase2_dir),
        out_dir,
        projectors,
        load_model=load_model,
        prompt_key=prompt_key,
    )

    # 3. Optionally load live model for dissociation
    if load_model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model for dissociation (device={device})...")
        try:
            model, tokenizer = _load_model_and_tokenizer(model_name, device)
            ctx["model"]        = model
            ctx["tokenizer"]    = tokenizer
            ctx["device"]       = device
            ctx["hook_targets"] = _get_attention_output_modules(model)
            # FIX-R2: pass model + tokenizer so live induction score check runs
            ctx["text"] = _select_input_text(
                ctx, model=model, tokenizer=tokenizer, device=device
            )
            print(f"  Input text: {ctx['text'][:80]}…")
        except Exception as exc:
            print(
                f"  WARNING: model load failed ({exc}); dissociation will be skipped."
            )
            ctx["load_model"] = False

    # 4. Filter registry by track
    registry = _filter_registry(tracks)

    # 5. Run all sub-experiments
    subresults = run_phase6(registry, ctx, out_dir)

    # 6. Assemble final report
    report_path = out_dir / "phase6_report.txt"
    report_6.assemble_report(subresults, ctx, report_path)
    print(f"\nReport written: {report_path}")
    print(f"Sub-results in: {out_dir / 'sub'}/")


def _filter_registry(tracks: str) -> list[SubexperimentSpec]:
    if tracks == "all":
        return REGISTRY
    if "A" in tracks.upper():
        names = {"head_classify", "qk_decompose", "induction_ov"}
    elif "BD" in tracks.upper() or "B" in tracks.upper():
        names = {
            "eigenspace_degeneracy", "centroid_velocity",
            "local_contraction", "probe_subspace",
        }
    elif "C" in tracks.upper():
        names = {"write_subspace", "dissociation"}
    else:
        names = {s.name for s in REGISTRY}
    return [s for s in REGISTRY if s.name in names]


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 6 — Real/Imaginary subspace analysis"
    )
    p.add_argument("--model",       type=str, default="albert-xlarge-v2")
    p.add_argument("--models",      type=str, nargs="+", default=None)
    p.add_argument("--phase1-dir",  type=str, default="results/phase1")
    p.add_argument("--phase2-dir",  type=str, default="results/phase2")
    p.add_argument("--out-dir",     type=str, default="results/phase6")
    p.add_argument(
        "--track",
        type=str,
        default="all",
        choices=["all", "A", "BD", "C"],
        help="Which track(s) to run",
    )
    p.add_argument(
        "--load-model",
        action="store_true",
        help="Load live model for dissociation (track C)",
    )
    p.add_argument("--prompt", type=str, default="wiki_paragraph")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    models = args.models or [args.model]

    for model_name in models:
        try:
            run_one_model(
                model_name=model_name,
                phase1_dir=args.phase1_dir,
                phase2_dir=args.phase2_dir,
                out_dir=args.out_dir,
                tracks=args.track,
                load_model=args.load_model,
                prompt_key=args.prompt,
            )
        except Exception as exc:
            print(f"FAILED: {model_name}: {exc}")
            traceback.print_exc()