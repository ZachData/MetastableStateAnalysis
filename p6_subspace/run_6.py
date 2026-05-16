"""
run_6.py — Phase 6 orchestrator with comprehensive bug fixes.

Fixes applied
-------------
Bug W1  : _compute_qk_logit_matrices(ctx) — compute X @ M_h @ X.T per head
Bug W3  : _normalise_empty_lists(ctx) — coerce empty lists to None
Bug W2  : _classify_layer_types handles iter_N vs N naming mismatch
Bug #1  : _find_p2_weights_path handles hyphen/underscore stem resolution
Bug #2  : _load_ov_weights uses regex-based key matching (ov/wo/W_O prefixes)
Bug #4  : Per-layer model routing in _load_ov_weights
Bug #3  : ov_data key naming (ov_per_head) matches build_global_projectors
Bug #4  : Model loading, tokenizer, hook targets populated when load_model=True
"""

import argparse
import json
import re
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


# ---------------------------------------------------------------------------
# SECTION A: Phase 1 & Phase 2 artifact loaders (core.io, weights.py)
# ---------------------------------------------------------------------------

def _find_p2_weights_path(phase2_dir: Path, model_name: str) -> Path | None:
    """
    FIX Bug #1: Locate Phase 2 ov_weights NPZ for model_name.

    Phase 2's writer uses stem = model_name.replace("/", "_") — hyphens
    preserved.  This function tries both hyphen and underscore forms.

    Returns None when neither file exists; caller decides how to fail.
    """
    hyphen_stem = model_name.replace("/", "_")
    underscore_stem = hyphen_stem.replace("-", "_")
    for stem in (hyphen_stem, underscore_stem):
        candidate = phase2_dir / f"ov_weights_{stem}.npz"
        if candidate.exists():
            return candidate
    return None


def _load_ov_weights(path: Path) -> dict:
    """
    FIX Bug #2, #4: Load per-head OV, W_Q, W_K matrices from Phase 2 npz.

    Accepts keys with ov/wo/W_O prefixes (different Phase 2 versions).
    Detects per-layer vs shared layout and handles both correctly.
    Routes results into keys matching build_global_projectors contract.

    Returns dict with:
      ov_per_head   : list[(d,d)] (shared) or list[list[(d,d)]] (per-layer)
      is_per_layer  : bool
      layer_names   : list[str]
      n_heads       : int
      d_model       : int
      qk_per_head   : optional list[(WQ,WK)]
      rot_energy_fracs : optional list[float]
    """
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    out: dict = {}

    head_re = re.compile(r"^(?:ov|wo|W_O)_head(\d+)_(.+)$")
    wq_re = re.compile(r"^(?:wq|W_Q)_head(\d+)_(.+)$")
    wk_re = re.compile(r"^(?:wk|W_K)_head(\d+)_(.+)$")

    def _suffix_order(s: str) -> tuple:
        if s == "shared":
            return (-1, 0)
        m = re.match(r"^layer_(\d+)$", s)
        return (0, int(m.group(1))) if m else (1, 0)

    def _bucket(regex) -> dict[str, list[tuple[int, str]]]:
        b: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for k in keys:
            m = regex.match(k)
            if m:
                b[m.group(2)].append((int(m.group(1)), k))
        return b

    # ── OV / W_O per-head ────────────────────────────────────────────────
    head_buckets = _bucket(head_re)

    if head_buckets:
        suffixes = sorted(head_buckets.keys(), key=_suffix_order)
        is_per_layer = not (len(suffixes) == 1 and suffixes[0] == "shared")

        if is_per_layer:
            ov_per_head = [
                [data[k] for _, k in sorted(head_buckets[s], key=lambda p: p[0])]
                for s in suffixes
            ]
            out["ov_per_head"] = ov_per_head
            out["n_heads"] = len(ov_per_head[0]) if ov_per_head else 0
            out["d_model"] = ov_per_head[0][0].shape[0] if out["n_heads"] else 0
        else:
            ov_per_head = [
                data[k] for _, k in sorted(head_buckets["shared"], key=lambda p: p[0])
            ]
            out["ov_per_head"] = ov_per_head
            out["n_heads"] = len(ov_per_head)
            out["d_model"] = ov_per_head[0].shape[0] if ov_per_head else 0

        out["is_per_layer"] = is_per_layer
        out["layer_names"] = suffixes
    else:
        out["ov_per_head"] = []
        out["n_heads"] = 0
        out["d_model"] = 0
        out["is_per_layer"] = False
        out["layer_names"] = []

    # ── W_Q / W_K per-head (optional) ────────────────────────────────────
    wq_buckets = _bucket(wq_re)
    wk_buckets = _bucket(wk_re)

    if wq_buckets and wk_buckets:
        common = sorted(set(wq_buckets) & set(wk_buckets), key=_suffix_order)
        if out["is_per_layer"] and common:
            qk = []
            for s in common:
                qs = sorted(wq_buckets[s], key=lambda p: p[0])
                ks = sorted(wk_buckets[s], key=lambda p: p[0])
                qk.append([(data[q], data[k]) for (_, q), (_, k) in zip(qs, ks)])
            out["qk_per_head"] = qk
        elif "shared" in common:
            qs = sorted(wq_buckets["shared"], key=lambda p: p[0])
            ks = sorted(wk_buckets["shared"], key=lambda p: p[0])
            out["qk_per_head"] = [
                (data[q], data[k]) for (_, q), (_, k) in zip(qs, ks)
            ]

    # ── Phase 2i rotational energy ───────────────────────────────────────
    if "rot_energy_fracs" in keys:
        out["rot_energy_fracs"] = data["rot_energy_fracs"].tolist()

    return out


# ---------------------------------------------------------------------------
# SECTION B: QK logit computation & gating (Bug W1, W3)
# ---------------------------------------------------------------------------

def _compute_qk_logit_matrices(ctx: dict) -> dict:
    """
    FIX Bug W1: Compute per-head QK logit matrices from qk_matrices + token_activations.

    For each head h with weight pair (WQ_h, WK_h):
        M_h = WQ_h @ WK_h^T            # (d, d) in residual-stream space
        QK_logit_h[i,j] = x_i^T M_h x_j = (X @ M_h @ X^T)[i,j]

    Sets ctx["qk_logit_matrices"] in-place and returns ctx.

    If qk_matrices is None or empty, leaves ctx["qk_logit_matrices"] = None.
    """
    qk = ctx.get("qk_matrices")
    X = ctx.get("token_activations")

    if not qk or X is None:
        ctx["qk_logit_matrices"] = None
        return ctx

    logit_mats = []
    for WQ, WK in qk:
        M = WQ @ WK.T  # (d, d)
        logit_mats.append((X @ M @ X.T).astype(np.float32))  # (n, n)

    ctx["qk_logit_matrices"] = logit_mats
    return ctx


def _normalise_empty_lists(ctx: dict) -> dict:
    """
    FIX Bug W3: Coerce empty-list weight keys to None so prerequisites_met gates them.

    SubexperimentSpec.prerequisites_met checks ``ctx.get(k) is None``.
    An empty list [] is not None and passes the gate, causing sub-experiments
    to receive zero matrices and either fail silently or crash.

    Keys normalised: qk_matrices, wo_matrices, qk_logit_matrices, attn_matrices.
    """
    for key in ("qk_matrices", "wo_matrices", "qk_logit_matrices", "attn_matrices"):
        if key in ctx and isinstance(ctx[key], list) and len(ctx[key]) == 0:
            ctx[key] = None
    return ctx


# ---------------------------------------------------------------------------
# SECTION C: Layer classification (Bug W2)
# ---------------------------------------------------------------------------

import re as _re

_LAYER_PREFIX_RE = _re.compile(r"^(?:iter|layer)[_-]?(\d+)$")


def _layer_idx_from_name(name) -> int | None:
    """
    FIX Bug W2: Extract canonical integer layer index from any common form.

    Handles: "iter_2", "iter-2", "layer_2", "layer-2", "2", 2, "02"
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
    layer_names: list[str],
    events: list[dict],
    trajectories: list[dict] | None = None,
    plateau_windows: list[dict] | None = None,
) -> list[str]:
    """
    FIX Bug W2: Label each layer as "merge", "plateau", "transition", or "other".

    Both event names and layer_names are normalised to integer indices via
    _layer_idx_from_name before comparison.  This fixes the silent name-mismatch
    bug where "iter_2" never matched "2" in merge_layers.

    When plateau_windows is supplied, uses those to distinguish plateau from
    transition.  When omitted, falls back to legacy behaviour (all non-merge
    are plateau) for backward compatibility with existing tests.
    """
    # ── Build merge index set from events ────────────────────────────────
    merge_indices: set[int] = set()
    for ev in events or []:
        if ev.get("type") != "merge":
            continue
        idx = _layer_idx_from_name(ev.get("layer_from"))
        if idx is None:
            idx = _layer_idx_from_name(ev.get("layer_name"))
        if idx is not None:
            merge_indices.add(idx)

    # ── Build plateau index set from windows (if given) ──────────────────
    plateau_indices: set[int] | None
    if plateau_windows:
        plateau_indices = set()
        for w in plateau_windows:
            try:
                start = int(w["start"])
                end = int(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            plateau_indices.update(range(start, end + 1))
    else:
        plateau_indices = None  # signal: legacy fallback

    # ── Classify each layer ──────────────────────────────────────────────
    types: list[str] = []
    for list_pos, lname in enumerate(layer_names):
        name_idx = _layer_idx_from_name(lname)
        canonical_idx = name_idx if name_idx is not None else list_pos

        if canonical_idx in merge_indices:
            types.append("merge")
        elif plateau_indices is None:
            # Legacy: no window data → everything non-merge is plateau
            types.append("plateau")
        elif canonical_idx in plateau_indices:
            types.append("plateau")
        else:
            types.append("transition")

    return types


# ---------------------------------------------------------------------------
# SECTION D: Context assembly
# ---------------------------------------------------------------------------

def build_context(
    model_name: str,
    phase1_dir: Path,
    phase2_dir: Path,
    out_dir: Path,
    projectors: dict,
    load_model: bool = False,
    prompt_key: str = "wiki_paragraph",
    layer_idx: int = 0,
) -> dict:
    """
    Assemble the shared context dict for one model.

    Loads Phase 1 activations and Phase 2 weight matrices.
    Applies Bugs W1/W3/W2 fixes.

    Returns a ctx dict ready for run_phase6.
    """
    from core.io import find_phase1_run_dir, load_phase1_run

    stem = model_name.replace("/", "_").replace("-", "_")

    # --- Phase 1 artifacts ---
    p1_run_dir = find_phase1_run_dir(Path(phase1_dir), model_name, prompt_key)

    ctx: dict = {
        "model_name": model_name,
        "stem": stem,
        "out_dir": out_dir,
        "projectors": projectors,
        "layer_name": projectors["layer_names"][
            min(layer_idx, len(projectors["layer_names"]) - 1)
        ],
        "layer_idx": layer_idx,
        "load_model": load_model,
    }

    if p1_run_dir is not None and p1_run_dir.exists():
        p1 = load_phase1_run(p1_run_dir)

        ctx["tokens"] = p1["tokens"]
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
            ctx["layer_names"] = []

        # hdbscan_labels is list[ndarray] from load_phase1_run (Bug IO-2 fix)
        ctx["labels_per_layer"] = p1.get("hdbscan_labels")

        events = p1.get("events", [])
        ctx["merge_events"] = events

        # FIX Bug W2: _classify_layer_types handles "iter_N" vs "N" naming
        ctx["layer_type_labels"] = _classify_layer_types(
            ctx["layer_names"],
            events,
            p1.get("trajectories", []),
            plateau_windows=p1.get("plateau_windows"),
        )

        # Token activations for head classification
        if ctx.get("activations_per_layer"):
            safe_idx = min(layer_idx, len(ctx["activations_per_layer"]) - 1)
            ctx["token_activations"] = ctx["activations_per_layer"][safe_idx]

        # Attention matrices (per-head at one layer)
        if p1.get("attentions") is not None:
            A = p1["attentions"]  # (n_layers, n_heads, n, n)
            safe_idx = min(layer_idx, A.shape[0] - 1)
            ctx["attn_matrices"] = [A[safe_idx, h] for h in range(A.shape[1])]
        else:
            ctx["attn_matrices"] = None

    # --- Phase 2 weight artifacts ---
    # FIX Bug #1: use _find_p2_weights_path for hyphen/underscore resolution
    p2_weights = _find_p2_weights_path(Path(phase2_dir), model_name)
    if p2_weights is not None:
        # FIX Bug #2, #4: use new _load_ov_weights with proper key handling
        ov_data = _load_ov_weights(p2_weights)
        # FIX Bug #3: ov_data["ov_per_head"] matches build_global_projectors expectation
        ctx["wo_matrices"] = ov_data.get("ov_per_head") or None
        ctx["qk_matrices"] = ov_data.get("qk_per_head") or None
        ctx["rot_energy_fracs"] = ov_data.get("rot_energy_fracs")
    else:
        ctx["wo_matrices"] = None
        ctx["qk_matrices"] = None
        ctx["rot_energy_fracs"] = None

    ctx["qk_logit_matrices"] = None  # populated below

    # FIX Bug W1: compute QK logit matrices from qk_matrices + activations
    ctx = _compute_qk_logit_matrices(ctx)

    # FIX Bug W3: coerce empty lists to None so prerequisites_met gates correctly
    ctx = _normalise_empty_lists(ctx)

    return ctx


# ---------------------------------------------------------------------------
# SECTION E: Registry & sub-experiment wrappers
# ---------------------------------------------------------------------------

def _run_head_classify(ctx: dict):
    """Wrap classify_heads to conform to SubResult contract."""
    from p6_subspace.p6_io import SubResult, SEP_THICK, _bullet, _verdict_line, _fmt

    rot_fracs = None
    if ctx.get("rot_energy_fracs"):
        rot_fracs = ctx["rot_energy_fracs"]

    records = classify_heads(
        ctx["attn_matrices"],
        ctx["qk_logit_matrices"],
        ctx["token_activations"],
        rot_fracs,
    )
    layer_name = ctx.get("layer_name", "shared")
    map_data = head_map_data(records, layer_name)

    corr = map_data["cross_head_corr"]

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
        _bullet("Spearman ρ(f_rot, -CC)", corr.get("rho_frot_neg_cc")),
        _bullet("Spearman p-value (CC)", corr.get("p_value_neg_cc")),
        _bullet("Spearman ρ(f_rot, |PC|)", corr.get("rho_frot_abs_pc")),
        _bullet("Spearman p-value (PC)", corr.get("p_value_abs_pc")),
        _bullet("n_heads in correlation", corr.get("n_heads")),
        _verdict_line(
            "P6-A2",
            corr.get("p6_a2_passes"),
            f"ρ(-CC)={_fmt(corr.get('rho_frot_neg_cc'))} "
            f"ρ(|PC|)={_fmt(corr.get('rho_frot_abs_pc'))}"
            f" (p < α AND ρ > threshold)",
        ),
    ]

    vc = {
        "hc_rho_frot_neg_cc": corr.get("rho_frot_neg_cc"),
        "hc_p_value_neg_cc": corr.get("p_value_neg_cc"),
        "hc_rho_frot_abs_pc": corr.get("rho_frot_abs_pc"),
        "hc_p_value_abs_pc": corr.get("p_value_abs_pc"),
        "hc_p6_a2_passes": corr.get("p6_a2_passes"),
        "hc_n_anti_sim_heads": len(map_data["anti_sim_heads"]),
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
            "attn_matrices",
            "wo_matrices",
            "token_ids",
            "token_activations",
            "projectors",
        ],
    ),
    # Track B/D — activations
    SubexperimentSpec(
        name="eigenspace_degeneracy",
        run=run_eigenspace_degeneracy,
        requires=[
            "activations_per_layer",
            "labels_per_layer",
            "layer_type_labels",
            "projectors",
        ],
    ),
    SubexperimentSpec(
        name="centroid_velocity",
        run=run_centroid_velocity,
        requires=[
            "activations_per_layer",
            "labels_per_layer",
            "layer_type_labels",
            "projectors",
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
            "activations_per_layer",
            "labels_per_layer",
            "layer_type_labels",
            "projectors",
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


# ---------------------------------------------------------------------------
# SECTION F: Model loading (Bug #4)
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_name: str, device: str):
    """
    FIX Bug #4: Load HuggingFace model and tokenizer for dissociation.
    """
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
    FIX Bug R1: Identify attention output projection modules to hook.

    Targets the .dense child of SelfOutput modules (pre-residual-add),
    not the SelfOutput module itself (which would fire post-add).
    """
    targets = []

    for name, module in model.named_modules():
        cls = type(module).__name__

        # BERT-family: target the .dense projection inside SelfOutput
        if cls.endswith("SelfOutput"):
            # Find the .dense child
            if hasattr(module, "dense"):
                targets.append(module.dense)
            continue

        # GPT-2 style
        if "attn" in name and name.endswith(".c_proj"):
            targets.append(module)
            continue

        # GPT-Neo / GPT-J style
        if "attention" in name and name.endswith(".out_proj"):
            targets.append(module)
            continue

    if not targets:
        print(
            f"  WARNING: _get_attention_output_modules found no hook targets for "
            f"{type(model).__name__}.  "
            "Dissociation will run with empty hook list (no-op interventions)."
        )

    return targets


def _select_input_text(ctx: dict) -> str:
    """Pick a representative input text for the dissociation forward pass."""
    tokens = ctx.get("tokens")
    if tokens:
        text = " ".join(
            t.lstrip("##") for t in tokens if t not in ("[CLS]", "[SEP]", "<s>", "</s>")
        )
        if len(text.split()) >= 8:
            return text

    return (
        "The researchers found that the model consistently learned to predict "
        "the next token based on patterns it had seen earlier in the same sequence."
    )


# ---------------------------------------------------------------------------
# SECTION G: Main entry
# ---------------------------------------------------------------------------

def _build_or_load_projectors(
    out_dir: Path, phase2_dir: Path, model_name: str
) -> dict | None:
    """
    Build (or load from cache) global S/A projectors.

    Returns the projector dict, or None when Phase 2 weights are missing.
    """
    proj_path = out_dir / "projectors.npz"
    if proj_path.exists():
        print("Loading cached projectors...")
        return load_projectors(proj_path)

    print("Building global S/A projectors...")
    # FIX Bug #1: use _find_p2_weights_path
    p2_weights = _find_p2_weights_path(phase2_dir, model_name)
    if p2_weights is None:
        print(
            f"  ERROR: Phase 2 weights not found in {phase2_dir} for {model_name!r}. "
            f"Tried both hyphen and underscore stem forms."
        )
        return None

    try:
        # FIX Bug #2, #4: use new _load_ov_weights
        ov_data = _load_ov_weights(p2_weights)
    except KeyError as e:
        print(f"  ERROR: {e}")
        return None

    if not ov_data.get("ov_per_head"):
        print(f"  ERROR: no per-head OV matrices in {p2_weights}")
        return None

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


def run_one_model(
    model_name: str,
    phase1_dir: Path,
    phase2_dir: Path,
    out_dir: Path,
    tracks: str = "all",
    load_model: bool = False,
    prompt_key: str = "wiki_paragraph",
) -> None:
    import torch

    stem = model_name.replace("/", "_").replace("-", "_")
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

    # FIX Bug #4: populate model, tokenizer, hook_targets when load_model=True
    if load_model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model for dissociation (device={device})...")
        try:
            model, tokenizer = _load_model_and_tokenizer(model_name, device)
            ctx["model"] = model
            ctx["tokenizer"] = tokenizer
            ctx["device"] = device
            ctx["hook_targets"] = _get_attention_output_modules(model)
            ctx["text"] = _select_input_text(ctx)
            print(f"  Hook targets: {len(ctx['hook_targets'])} modules")
            print(f"  Input text:   {ctx['text'][:80]}...")
        except Exception as exc:
            print(
                f"  WARNING: model load failed ({exc}); dissociation will be skipped."
            )
            ctx["load_model"] = False

    # 3. Filter registry by track
    registry = _filter_registry(tracks)

    # 4. Run all sub-experiments
    subresults = run_phase6(registry, ctx, out_dir)

    # 5. Assemble final report
    report_path = out_dir / "phase6_report.txt"
    report_6.assemble_report(subresults, ctx, report_path)
    print(f"\nReport written: {report_path}")
    print(f"Sub-results in: {out_dir / 'sub'}/")


def _filter_registry(tracks: str) -> list[SubexperimentSpec]:
    """Return registry subset based on track selection."""
    if tracks == "all":
        return REGISTRY
    if "A" in tracks.upper():
        names = {"head_classify", "qk_decompose", "induction_ov"}
    elif "BD" in tracks.upper() or "B" in tracks.upper():
        names = {
            "eigenspace_degeneracy",
            "centroid_velocity",
            "local_contraction",
            "probe_subspace",
        }
    elif "C" in tracks.upper():
        names = {"write_subspace", "dissociation"}
    else:
        names = {s.name for s in REGISTRY}
    return [s for s in REGISTRY if s.name in names]


# ---------------------------------------------------------------------------
# SECTION H: CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 6 — Real/Imaginary subspace analysis"
    )
    p.add_argument("--model", type=str, default="albert-xlarge-v2")
    p.add_argument("--models", type=str, nargs="+", default=None)
    p.add_argument("--phase1-dir", type=str, default="results/phase1")
    p.add_argument("--phase2-dir", type=str, default="results/phase2")
    p.add_argument("--out-dir", type=str, default="results/phase6")
    p.add_argument(
        "--track",
        type=str,
        default="all",
        choices=["all", "A", "BD", "C"],
        help="Which track(s) to run",
    )
    p.add_argument(
        "--load-model", action="store_true", help="Load live model for dissociation track C"
    )
    p.add_argument("--prompt", type=str, default="wiki_paragraph")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
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