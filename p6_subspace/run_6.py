"""
run_6.py — Phase 6 orchestrator.

Usage
-----
  # Full pipeline, one model
  python run_6.py --model albert-xlarge-v2 \\
                  --phase1-dir results/phase1 \\
                  --phase2-dir results/phase2 \\
                  --out-dir    results/phase6

  # Track A only (weights, no live model needed)
  python run_6.py --model albert-xlarge-v2 ... --track A

  # Track B/D only (activations, no live model)
  python run_6.py --model albert-xlarge-v2 ... --track BD

  # All tracks including dissociation (requires --load-model)
  python run_6.py --model albert-xlarge-v2 ... --track all --load-model

  # Multiple models
  python run_6.py --models albert-xlarge-v2 bert-base-uncased gpt2 ...

Execution order
---------------
  1. subspace_build   — builds global S/A projectors (foundation, all tracks need this)
  2. Track A:  head_classify, qk_decompose, induction_ov   (weights only)
  3. Track B/D: eigenspace_degeneracy, centroid_velocity,  (activations)
                local_contraction, probe_subspace
  4. Track C:  write_subspace (weights), dissociation (live model — only if --load-model)
  5. report_6.assemble_report → phase6_report.txt

Dependencies from earlier phases
----------------------------------
  Phase 1 : activations.npz, clusters.npz, metrics.json (HDBSCAN labels, trajectories)
  Phase 2 : ov_weights_{stem}.npz — per-head OV matrices (for subspace_build)
            ov_projectors_{stem}.npz — QK matrices (for qk_decompose)
  Phase 2i: (optional) rotational energy fractions per head (for head_classify.f_rot)
"""

import argparse
import json
import sys
import traceback
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
# Thin wrappers for registry entries that need pre-processing
# FIX (Bug 2): these functions are defined BEFORE REGISTRY, which references
# them by name at module-level.  The original code defined them after REGISTRY,
# causing NameError at import time.
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
    map_data   = head_map_data(records, layer_name)

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
        "P6-A2: f_rot(h) negatively correlated with CC and positively with PC?",
        _bullet("Spearman ρ(f_rot, -CC)", corr["rho_frot_neg_cc"]),
        _bullet("Spearman ρ(f_rot,  PC)", corr["rho_frot_pc"]),
        _bullet("n_heads in correlation", corr["n_heads"]),
        _verdict_line(
            "P6-A2",
            corr["p6_a2_satisfied"],
            f"ρ(-CC)={_fmt(corr['rho_frot_neg_cc'])} ρ(PC)={_fmt(corr['rho_frot_pc'])}"
            f" (threshold both > 0.4)",
        ),
    ]

    vc = {
        "hc_rho_frot_neg_cc":    corr["rho_frot_neg_cc"],
        "hc_rho_frot_pc":        corr["rho_frot_pc"],
        "hc_p6_a2_satisfied":    corr["p6_a2_satisfied"],
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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

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
        requires=["attn_matrices", "wo_matrices", "token_ids",
                  "token_activations", "projectors"],
    ),

    # Track B/D — activations
    SubexperimentSpec(
        name="eigenspace_degeneracy",
        run=run_eigenspace_degeneracy,
        requires=["activations_per_layer", "labels_per_layer",
                  "layer_type_labels", "projectors"],
    ),
    SubexperimentSpec(
        name="centroid_velocity",
        run=run_centroid_velocity,
        requires=["activations_per_layer", "labels_per_layer",
                  "layer_type_labels", "projectors"],
    ),
    SubexperimentSpec(
        name="local_contraction",
        run=run_local_contraction,
        requires=["activations_per_layer", "labels_per_layer", "layer_type_labels"],
    ),
    SubexperimentSpec(
        name="probe_subspace",
        run=run_probe_subspace,
        requires=["activations_per_layer", "labels_per_layer",
                  "layer_type_labels", "projectors"],
    ),

    # Track C — write subspace (weights) then dissociation (live model, gated)
    SubexperimentSpec(
        name="write_subspace",
        run=run_write_subspace,
        requires=["wo_matrices", "projectors"],
    ),
    SubexperimentSpec(
        name="dissociation",
        run=_run_dissociation_gated,
        requires=["model", "tokenizer", "text", "token_ids",
                  "projectors", "hook_targets"],
        applicable=lambda ctx: ctx.get("load_model", False),
    ),
]


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

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
    Returns a ctx dict ready for run_phase6.
    """
    from core.io import load_phase1_run   # Phase 5 IO helpers

    stem = model_name.replace("/", "_").replace("-", "_")

    # --- Phase 1 artifacts ---
    # Find the run directory for this model + prompt
    p1_candidates = list((phase1_dir / stem).glob(f"*{prompt_key}*"))
    if not p1_candidates:
        p1_candidates = list((phase1_dir / stem).glob("*"))
    p1_dir = sorted(p1_candidates)[-1] if p1_candidates else None

    ctx = {
        "model_name":   model_name,
        "stem":         stem,
        "out_dir":      out_dir,
        "projectors":   projectors,
        "layer_name":   projectors["layer_names"][layer_idx],
        "layer_idx":    layer_idx,
        "load_model":   load_model,
    }

    if p1_dir and p1_dir.exists():
        p1 = load_phase1_run(p1_dir)

        ctx["token_ids"]           = np.array([
            hash(t) % (2**31) for t in p1["tokens"]
        ], dtype=np.int64)
        ctx["tokens"]              = p1["tokens"]
        ctx["activations_per_layer"] = [
            p1["activations"][L] for L in range(p1["activations"].shape[0])
        ] if p1["activations"] is not None else None
        ctx["labels_per_layer"]    = p1.get("hdbscan_labels")
        ctx["layer_names"]         = [
            f"iter_{L}" for L in range(len(ctx["activations_per_layer"]))
        ] if ctx["activations_per_layer"] else []
        ctx["merge_events"]        = p1.get("events", [])

        # Layer-type labels: classify each layer as plateau / merge / other
        ctx["layer_type_labels"]   = _classify_layer_types(
            ctx["layer_names"],
            p1.get("events", []),
            p1.get("trajectories", []),
        )

        # Token activations at the first interesting layer (for head_classify)
        if ctx["activations_per_layer"]:
            ctx["token_activations"] = ctx["activations_per_layer"][layer_idx]

        # Attention matrices from Phase 1 (if saved)
        if p1.get("attentions") is not None:
            # attentions shape: (n_layers, n_heads, n_tokens, n_tokens)
            A = p1["attentions"]
            ctx["attn_matrices"]      = [A[layer_idx, h] for h in range(A.shape[1])]
        else:
            ctx["attn_matrices"] = None

    # --- Phase 2 weight artifacts ---
    # FIX (Bug 3): _load_ov_weights now stores matrices under "ov_per_head"
    # (matching what build_global_projectors expects).  Previously this key
    # was "wo_per_head", causing a KeyError whenever fresh projectors were built.
    p2_weights = phase2_dir / f"ov_weights_{stem}.npz"
    if p2_weights.exists():
        ov_data = _load_ov_weights(p2_weights)
        ctx["wo_matrices"]       = ov_data.get("ov_per_head", [])   # W_O per head
        ctx["qk_matrices"]       = ov_data.get("qk_per_head", [])
        ctx["qk_logit_matrices"] = None   # computed on-the-fly from qk_matrices + activations
        ctx["rot_energy_fracs"]  = ov_data.get("rot_energy_fracs")
    else:
        ctx["wo_matrices"] = None
        ctx["qk_matrices"] = None

    return ctx


def _classify_layer_types(
    layer_names: list[str],
    events:      list[dict],
    trajectories: list[dict],
) -> list[str]:
    """
    Label each layer as "plateau", "merge", or "other".

    A layer is "merge" if any merge event occurs at that layer.
    A layer is "plateau" if it is not a merge layer and at least one
    trajectory is active and stable (use heuristic: > 2 consecutive non-merge layers).
    """
    merge_layers = set()
    for ev in events:
        if ev.get("type") == "merge":
            lname = ev.get("layer_from") or ev.get("layer_name")
            if lname:
                merge_layers.add(str(lname))

    types = []
    for lname in layer_names:
        if lname in merge_layers:
            types.append("merge")
        else:
            types.append("plateau")   # simplified; could be refined with trajectory lifespan
    return types


def _load_ov_weights(path: Path) -> dict:
    """
    Load per-head OV, WO, QK matrices from Phase 2 npz.

    FIX (Bug 3): W_O matrices are now stored under "ov_per_head" (not
    "wo_per_head") so that build_global_projectors can find them with its
    expected key name.  ctx["wo_matrices"] is populated from this same key.
    """
    data = np.load(path, allow_pickle=True)
    out  = {}

    keys = list(data.keys())

    # Collect per-head WO matrices — stored as "ov_per_head" for subspace_build
    wo_keys = sorted([k for k in keys if "wo_head" in k or "W_O_head" in k])
    if wo_keys:
        out["ov_per_head"] = [data[k] for k in wo_keys]

    # Collect QK pairs
    wq_keys = sorted([k for k in keys if "wq_head" in k or "W_Q_head" in k])
    wk_keys = sorted([k for k in keys if "wk_head" in k or "W_K_head" in k])
    if wq_keys and wk_keys:
        out["qk_per_head"] = [
            (data[q], data[k]) for q, k in zip(wq_keys, wk_keys)
        ]

    # Rotational energy fracs from Phase 2i (if present)
    if "rot_energy_fracs" in keys:
        out["rot_energy_fracs"] = data["rot_energy_fracs"].tolist()

    return out


# ---------------------------------------------------------------------------
# Model loading helpers (for dissociation track)
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_name: str, device: str):
    """
    Load a HuggingFace model and tokenizer for the dissociation forward pass.
    """
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(
        model_name,
        output_attentions=True,
        output_hidden_states=True,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _get_attention_output_modules(model) -> list:
    """
    Identify the attention output projection modules to hook.

    We need the module whose output tensor is the per-head attention result
    *before* it is added to the residual stream.  Common patterns:

      BERT / ALBERT  : BertSelfOutput  (dense layer inside BertAttention)
      RoBERTa        : RobertaSelfOutput
      GPT-2          : attention output projection (attn.c_proj)
      Generic        : any module whose class name ends in "SelfOutput"

    Returns an empty list and prints a warning if nothing matches.
    """
    targets = []

    for name, module in model.named_modules():
        cls = type(module).__name__

        # BERT-family: the SelfOutput submodule applies the output dense + LayerNorm
        if cls.endswith("SelfOutput"):
            targets.append(module)
            continue

        # GPT-2 style: the attention output linear is model.transformer.h[i].attn.c_proj
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
    """
    Pick a representative input text for the dissociation forward pass.

    Preference order:
      1. ctx["tokens"] from Phase 1 (reconstruct a sentence-length string)
      2. A fixed fallback sentence long enough to produce induction candidates
    """
    tokens = ctx.get("tokens")
    if tokens:
        # Re-join wordpiece tokens into a rough string (good enough for inference)
        text = " ".join(t.lstrip("##") for t in tokens if t not in ("[CLS]", "[SEP]", "<s>", "</s>"))
        if len(text.split()) >= 8:
            return text

    return (
        "The researchers found that the model consistently learned to predict "
        "the next token based on patterns it had seen earlier in the same sequence."
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

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
    proj_path = out_dir / "projectors.npz"
    if proj_path.exists():
        print("Loading cached projectors...")
        projectors = load_projectors(proj_path)
    else:
        print("Building global S/A projectors...")
        p2_weights = phase2_dir / f"ov_weights_{stem}.npz"
        if not p2_weights.exists():
            print(f"  ERROR: Phase 2 weights not found at {p2_weights}")
            return
        ov_data    = _load_ov_weights(p2_weights)
        # FIX (Bug 3): ov_data now uses "ov_per_head" (set in _load_ov_weights),
        # matching the key name expected by build_global_projectors.
        ov_data["d_model"]     = _infer_d_model(p2_weights)
        ov_data["n_heads"]     = len(ov_data.get("ov_per_head", []))
        ov_data["layer_names"] = ["shared"]
        ov_data["is_per_layer"] = False

        projectors = build_global_projectors(ov_data)
        save_projectors(projectors, proj_path)
        print_projector_summary(projectors)

    # 2. Assemble context
    ctx = build_context(
        model_name, Path(phase1_dir), Path(phase2_dir), out_dir,
        projectors, load_model=load_model, prompt_key=prompt_key,
    )

    # FIX (Bug 4): load live model, tokenizer, and hook targets when
    # --load-model is set.  Previously none of these were populated, so the
    # dissociation experiment was silently skipped by prerequisites_met even
    # when the user explicitly requested Track C.
    if load_model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model for dissociation (device={device})...")
        try:
            model, tokenizer = _load_model_and_tokenizer(model_name, device)
            ctx["model"]       = model
            ctx["tokenizer"]   = tokenizer
            ctx["device"]      = device
            ctx["hook_targets"] = _get_attention_output_modules(model)
            ctx["text"]        = _select_input_text(ctx)
            print(f"  Hook targets: {len(ctx['hook_targets'])} modules")
            print(f"  Input text:   {ctx['text'][:80]}...")
        except Exception as exc:
            print(f"  WARNING: model load failed ({exc}); dissociation will be skipped.")
            ctx["load_model"] = False   # disable the applicable gate

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
        names = {"eigenspace_degeneracy", "centroid_velocity",
                 "local_contraction", "probe_subspace"}
    elif "C" in tracks.upper():
        names = {"write_subspace", "dissociation"}
    else:
        names = {s.name for s in REGISTRY}
    return [s for s in REGISTRY if s.name in names]


def _infer_d_model(p2_weights: Path) -> int:
    data = np.load(p2_weights, allow_pickle=True)
    for k in data.keys():
        if "wo_head" in k or "W_O_head" in k:
            return data[k].shape[0]
    return 768   # fallback


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 6 — Real/Imaginary subspace analysis")
    p.add_argument("--model",       type=str, default="albert-xlarge-v2")
    p.add_argument("--models",      type=str, nargs="+", default=None)
    p.add_argument("--phase1-dir",  type=str, default="results/phase1")
    p.add_argument("--phase2-dir",  type=str, default="results/phase2")
    p.add_argument("--out-dir",     type=str, default="results/phase6")
    p.add_argument("--track",       type=str, default="all",
                   choices=["all", "A", "BD", "C"],
                   help="Which track(s) to run")
    p.add_argument("--load-model",  action="store_true",
                   help="Load live model for dissociation track C")
    p.add_argument("--prompt",      type=str, default="wiki_paragraph")
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
