"""
run_1b.py — Phase 1b: Hemispheric Structure Investigation.

Orchestrates the hemisphere pipeline on each (model, prompt) combination and
writes per-run JSON/markdown, a particle table, and a cross-run digest.

Usage
-----
    python -m p1b_hemisphere.run_1b --list-models
    python -m p1b_hemisphere.run_1b                          # Blog 1 models
    python -m p1b_hemisphere.run_1b --models pythia-410m-pilot
    python -m p1b_hemisphere.run_1b --models pythia-1.4b-anchors --prompts wiki_paragraph
    python -m p1b_hemisphere.run_1b --phase1-dir results/2026-04-23_18-30-06
    python -m p1b_hemisphere.run_1b --phase1-dir DIR --from-phase1   # no forward pass
    python -m p1b_hemisphere.run_1b --no-cone                # skip Block 3
    python -m p1b_hemisphere.run_1b --n-null 20              # null-reference Block 3

What changed in this revision
-----------------------------

**--models no longer defaults to the whole registry.** It defaulted to
`list(MODEL_CONFIGS.keys())`, which is 47 entries since the Pythia registry
merged in — a bare invocation meant 37 checkpoint downloads across every
prompt. Now DEFAULT_MODELS (the seven Blog 1 architectures) with MODEL_GROUPS
opt-in, via core/model_selection.py, matching run_1.py.

**--fast actually works.** It reassigned `core.config.ALBERT_SNAPSHOTS` at
module scope after this module had already bound the name at import, so the
override was dead code and --fast silently ran the full 28-snapshot sweep.
Snapshot values are now read through the config module at call time.

**Phase 1 cross-reference resolves for ALBERT.** The old loader built
`{model}_{prompt}_d{depth}` and looked for it directly under phase1_dir;
Phase 1 writes `{model}_{depth}iter_{prompt}`. It therefore never resolved
for any ALBERT extended run, so hdbscan_labels never loaded and Block 2's
nesting test had nothing to test — recorded in status-1b.md as "Inconclusive
for ALBERT", which was a path bug. Resolution and loading now go through
p1b_io, which delegates to Phase 1's own find_phase1_run_dir/load_phase1_run.

**--from-phase1 skips the forward pass entirely.** Phase 1 already saves
unit-norm activations and its own Fiedler vectors. Re-extracting them cost a
model load per run and guaranteed Phase 1b analysed a different Fiedler
vector than the one Phase 1 clustered on.

**A run manifest.** `experiment.txt` is kept for readability, but the run now
also writes core/io.py's manifest.json — manifest id, git SHA, prompt-battery
hash, wall time — so two Phase 1b runs are checkably the same experiment.

Block map
---------
  Block 0  bipartition_detect.analyze_bipartition
  Block 1  hemisphere_tracking.analyze_hemisphere_tracking
  Block 2  hemisphere_membership.analyze_hemisphere_membership
  Block 3  cone_collapse.analyze_cone_collapse
  Block 4  (inline) asymmetry distribution
  Block A  axis_identity.analyze_axis_identity   (new)

Blocks 5 (mechanism vs OV) and 6 (semantic MI) still require Phase 2 OV
artifacts and are not run here.

Outputs (per run)
-----------------
  phase1b_{stem}.json        flat per-layer / per-token / summary JSON
  phase1b_{stem}.md          human-readable per-block summary
  phase1b_{stem}_particles.npz   ParticleTable, one row per (layer, token)
  phase1b_{stem}_axes.npz    activation-space Fiedler axes, (n_layers, d)

The JSON additionally carries `cone_per_layer`, `hdbscan_nesting`,
`border_vs_noise`, and `persistence_length` — per-layer tables every block
already computed and the writer used to drop, keeping only one summary number
from each. Readers written against the old shape are unaffected: every key
above is new, and nothing existing changed name or meaning. See
visualization/FIGURES-1b.md for what each unblocks.

Outputs (cross-run)
-------------------
  phase1b_cross_run.json     model x prompt x checkpoint aggregation + verdict
  phase1b_cross_run.md       one-page synthesis, generated from the results
  experiment.txt / manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

import core.config as cfg
from core.config import (
    BASE_RESULTS_DIR,
    DEFAULT_MODELS,
    MODEL_CONFIGS,
    MODEL_GROUPS,
    PROMPTS,
)
from core.frames import FrameSpec
from core.io import RunTimer, write_manifest
from core.model_selection import print_model_catalogue, resolve_model_names

from p1b_hemisphere.axis_identity import (
    analyze_axis_identity,
    axis_identity_to_json,
)
from p1b_hemisphere.bipartition_detect import (
    CONNECTIVITY_FLOOR,
    analyze_bipartition,
)
from p1b_hemisphere.cone_collapse import analyze_cone_collapse, cone_collapse_to_json
from p1b_hemisphere.hemisphere_membership import (
    analyze_hemisphere_membership,
    membership_to_json,
)
from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
from p1b_hemisphere import p1b_io
from p1b_hemisphere.p1b_report import (
    aggregate,
    aggregate_by_checkpoint,
    build_summary,
    checkpoint_step,
    cross_run_markdown,
    global_verdict,
)


OUTPUT_DIR: Path = BASE_RESULTS_DIR

#: Prefix for every artifact this phase writes. Was "phase1h_", from the
#: phase's working name. The directory, the docs, and the plan all say 1b;
#: the artifacts now do too. Old runs keep their filenames — renaming
#: existing output is a separate, unscoped job, exactly as with Phase 2b's
#: phase2i_*.json (see INDEX.md).
ARTIFACT_PREFIX = "phase1b"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_all(
    models_to_run: list | None = None,
    prompts_to_run: list | None = None,
    run_cone: bool = True,
    phase1_dir: Path | None = None,
    from_phase1: bool = False,
    n_null: int = 0,
    frame_kind: str = "l2_sphere",
    connectivity_floor: float = CONNECTIVITY_FLOOR,
    regime_key: str = "regime",
    drop_pos0: bool = False,
    seed: int = 0,
    output_dir: Path | None = None,
) -> list:
    """
    Run the full Phase 1b pipeline.

    Parameters
    ----------
    models_to_run  : registry keys or MODEL_GROUPS names (default:
                     DEFAULT_MODELS — the Blog 1 architectures, NOT every
                     registry key).
    prompts_to_run : prompt keys from PROMPTS (default: all).
    run_cone       : run Block 3.
    phase1_dir     : a Phase 1 run directory to cross-reference against.
    from_phase1    : take activations and Fiedler vectors from phase1_dir
                     instead of running a forward pass. Requires phase1_dir.
                     No torch, no GPU, and analyses exactly the arrays Phase 1
                     clustered on.
    n_null         : null replicates per layer for Block 3. 0 disables.
    frame_kind     : "l2_sphere" (every prior run) or "raw"/"identity".
                     LN frames need per-model LN parameters and are not wired
                     through this entry point yet — see the note below.
    drop_pos0      : exclude position 0 from Block 3's LP. On GPT-NeoX that
                     token is the attention sink and can single-handedly
                     determine the enclosing half-space.
    """
    global OUTPUT_DIR

    if models_to_run is None:
        models_to_run = list(DEFAULT_MODELS)
    if prompts_to_run is None:
        prompts_to_run = list(PROMPTS.keys())
    if from_phase1 and phase1_dir is None:
        sys.exit("--from-phase1 requires --phase1-dir.")

    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR = Path(output_dir) if output_dir else (
        BASE_RESULTS_DIR / f"{ARTIFACT_PREFIX}_{timestamp}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    opts = dict(
        run_cone=run_cone, from_phase1=from_phase1, n_null=n_null,
        frame_kind=frame_kind, connectivity_floor=connectivity_floor,
        regime_key=regime_key, drop_pos0=drop_pos0, seed=seed,
    )
    _write_experiment_txt(timestamp, models_to_run, prompts_to_run,
                          phase1_dir, opts)
    print(f"\nPhase 1b — output directory: {OUTPUT_DIR}")
    print(f"Models ({len(models_to_run)}): {', '.join(models_to_run)}")

    all_results: list = []
    for model_name in models_to_run:
        try:
            if from_phase1:
                model_results = _run_from_phase1(
                    model_name, prompts_to_run, phase1_dir, opts)
            else:
                model_results = _run_live(
                    model_name, prompts_to_run, phase1_dir, opts)
        except Exception as exc:
            print(f"  {model_name}: failed — {exc}")
            traceback.print_exc()
            continue
        all_results.extend(model_results)

    if all_results:
        print("\nGenerating cross-run digest…")
        _write_cross_run(all_results, OUTPUT_DIR)

    print(f"\nDone. Results in: {OUTPUT_DIR.resolve()}")
    return all_results


# ---------------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------------

def _run_from_phase1(model_name, prompts_to_run, phase1_dir, opts) -> list:
    """Analyse Phase 1's saved arrays. No model load, no torch."""
    results = []
    for prompt_key in prompts_to_run:
        print(f"\n  {model_name} / {prompt_key}  (from Phase 1)")
        ctx = p1b_io.load_phase1_context(phase1_dir, model_name, prompt_key)
        acts = ctx.get("activations")
        if acts is None:
            print("    no activations in the Phase 1 run — skipped")
            continue

        stem = _stem(model_name, prompt_key)
        try:
            result = _run_pipeline(
                np.asarray(acts, dtype=np.float32), ctx.get("tokens") or [],
                model_name, prompt_key, stem=stem, xref=ctx, opts=opts)
        except Exception as exc:
            print(f"    Pipeline failed: {exc}")
            traceback.print_exc()
            continue

        results.append(result)
        _save_run(result, OUTPUT_DIR)
    return results


def _run_live(model_name, prompts_to_run, phase1_dir, opts) -> list:
    """Run a forward pass. torch is imported here, not at module scope."""
    import torch
    from core.models import (
        extract_activations, extract_albert_extended, layernorm_to_sphere,
        load_model,
    )

    print(f"\nLoading {model_name}…")
    try:
        model, tokenizer = load_model(model_name)
    except Exception as exc:
        print(f"  Failed to load {model_name}: {exc}")
        return []

    def stack_and_norm(hidden_states) -> np.ndarray:
        stacked = torch.stack([layernorm_to_sphere(h) for h in hidden_states], dim=0)
        return stacked.numpy().astype(np.float32)

    model_cfg = MODEL_CONFIGS[model_name]
    # Read snapshots from the config MODULE, not from a name bound at import
    # time — that binding is what made --fast a no-op.
    snapshots  = list(getattr(cfg, "ALBERT_SNAPSHOTS", []) or [])
    max_iters  = int(getattr(cfg, "ALBERT_MAX_ITERATIONS", 0) or 0)
    use_extended = bool(model_cfg.get("is_albert", False)) and bool(snapshots)

    results = []
    try:
        for prompt_key in prompts_to_run:
            if use_extended:
                print(f"\n  {model_name} / {prompt_key}  (snapshots: {snapshots})")
                try:
                    snapshot_data = extract_albert_extended(
                        model, tokenizer, PROMPTS[prompt_key],
                        snapshots=snapshots, max_iterations=max_iters)
                except Exception as exc:
                    print(f"    extract_albert_extended failed: {exc}")
                    continue

                for depth, snap in snapshot_data.items():
                    acts   = stack_and_norm(snap["trajectory"])
                    tokens = snap["tokens"]
                    # Match Phase 1's own naming for extended runs
                    # ({model}@{N}iter -> {model}_{N}iter_{prompt}) so the
                    # cross-reference resolves. The old form was
                    # {model}_{prompt}_d{depth} and never did.
                    depth_model = f"{model_name}@{depth}iter"
                    stem = _stem(depth_model, prompt_key)
                    results.append(_one_run(
                        acts, tokens, depth_model, prompt_key, stem,
                        phase1_dir, opts))
            else:
                print(f"\n  {model_name} / {prompt_key}")
                try:
                    hidden_states, _attn, tokens = extract_activations(
                        model, tokenizer, PROMPTS[prompt_key], model_name)
                except Exception as exc:
                    print(f"    extract_activations failed: {exc}")
                    continue
                acts = stack_and_norm(hidden_states)
                stem = _stem(model_name, prompt_key)
                results.append(_one_run(
                    acts, tokens, model_name, prompt_key, stem,
                    phase1_dir, opts))
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return [r for r in results if r is not None]


def _one_run(acts, tokens, model_name, prompt_key, stem, phase1_dir, opts):
    xref = {}
    if phase1_dir is not None:
        xref = p1b_io.load_phase1_context(phase1_dir, model_name, prompt_key)
    try:
        result = _run_pipeline(acts, tokens, model_name, prompt_key,
                               stem=stem, xref=xref, opts=opts)
    except Exception as exc:
        print(f"    Pipeline failed: {exc}")
        traceback.print_exc()
        return None
    _save_run(result, OUTPUT_DIR)
    return result


def _stem(model_name: str, prompt_key: str) -> str:
    """Phase 1's stem convention, so artifacts line up across phases."""
    return f"{model_name.replace('/', '_').replace('@', '_')}_{prompt_key}"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(activations, tokens, model_name, prompt_key, stem,
                  xref: dict, opts: dict) -> dict:
    """Run every block on one (model, prompt, depth) activation tensor."""
    n_layers, n_tokens, _ = activations.shape
    print(f"    n_layers={n_layers}  n_tokens={n_tokens}")

    frame = FrameSpec(kind=opts.get("frame_kind", "l2_sphere"),
                      model_rev=str(model_name))
    rng   = np.random.default_rng(int(opts.get("seed", 0)))

    merge_indices    = xref.get("merge_indices")
    violation_layers = xref.get("violation_layers")
    hdbscan_labels   = xref.get("hdbscan_labels")
    plateau_layers   = xref.get("plateau_layers")

    print("    Block 0: bipartition detection…")
    block0 = analyze_bipartition(
        activations, frame=frame,
        connectivity_floor=float(opts.get("connectivity_floor",
                                          CONNECTIVITY_FLOOR)))

    print("    Block 1: hemisphere tracking…")
    block1 = analyze_hemisphere_tracking(
        block0,
        merge_transition_indices=merge_indices,
        violation_layers=violation_layers,
        regime_key=opts.get("regime_key", "regime"),
    )

    print("    Block 2: token membership…")
    block2 = analyze_hemisphere_membership(
        block0, block1,
        hdbscan_labels=hdbscan_labels,
        plateau_layers=plateau_layers,
        token_strings=tokens,
    )

    block3 = None
    if opts.get("run_cone", True):
        print("    Block 3: cone-collapse LP…")
        try:
            block3 = analyze_cone_collapse(
                activations, valid=block0["valid"],
                n_null=int(opts.get("n_null", 0)),
                drop_indices=[0] if opts.get("drop_pos0") else None,
                rng=rng)
        except Exception as exc:
            print(f"    Block 3 failed: {exc}")

    print("    Block A: axis identity…")
    try:
        axis = analyze_axis_identity(
            block0["frame_activations"], block0["fiedler_vecs"], block0["valid"])
    except Exception as exc:
        print(f"    Block A failed: {exc}")
        axis = None

    block4 = _compute_asymmetry(block0)

    per_layer = _assemble_per_layer(block0, block1, block3, block4, axis, n_layers)
    mem_json  = membership_to_json(block2)
    per_token = _assemble_per_token(mem_json, block1["aligned_assignments"],
                                    tokens, n_layers)
    summary   = build_summary(block0, block1, mem_json, block3, block4, axis)

    if block3 is not None:
        cj = cone_collapse_to_json(block3)["summary"]
        summary["mean_uniform_cone_fraction"]  = cj.get("mean_uniform_cone_fraction")
        summary["mean_shuffled_cone_fraction"] = cj.get("mean_shuffled_cone_fraction")

    return {
        "model":      model_name,
        "prompt":     prompt_key,
        "stem":       stem,
        "n_layers":   n_layers,
        "n_tokens":   n_tokens,
        "checkpoint_step": checkpoint_step(model_name),
        "frame":      {"kind": frame.kind, "pos0_policy": frame.pos0_policy,
                       "model_rev": frame.model_rev},
        "connectivity_floor": float(opts.get("connectivity_floor",
                                             CONNECTIVITY_FLOOR)),
        "per_layer":  per_layer,
        "events":     block1["events"],
        "per_token":  per_token,
        "summary":    summary,
        "_block0":    block0,
        "_block1":    block1,
        "_block2":    block2,
        "_block2_json": mem_json,
        "_block3":    block3,
        "_block4":    block4,
        "_axis":      axis,
        "_hdbscan_labels": hdbscan_labels,
        "_tokens":    tokens,
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _assemble_per_layer(block0, block1, block3, block4, axis, n_layers) -> list:
    crossing = block1["crossing_count"]
    axis_rot = block1["axis_rotation"]
    overlap  = block1["match_overlap"]
    rel      = block0.get("regime_relative")

    per_layer = []
    for L in range(n_layers):
        entry = {
            "layer":                 L,
            "regime":                str(block0["regime"][L]),
            "regime_relative":       (str(rel[L]) if rel is not None else None),
            "bipartition_eigengap":  _f(block0["bipartition_eigengap"][L]),
            "centroid_angle":        _f(block0["centroid_angle"][L]),
            "within_half_ip":        [_f(block0["within_half_ip"][L, 0]),
                                      _f(block0["within_half_ip"][L, 1])],
            "between_half_ip":       _f(block0["between_half_ip"][L]),
            "separation_ratio":      _f(block0["separation_ratio"][L]),
            "fiedler_boundary_frac": _f(block0["fiedler_boundary_frac"][L]),
            "hemisphere_sizes":      [int(block0["hemisphere_sizes"][L, 0]),
                                      int(block0["hemisphere_sizes"][L, 1])],
            "minority_fraction":     _f(block0["minority_fraction"][L]),
            "asymmetry":             _f(block4["asymmetry"][L]),
            # transitions are defined for L -> L+1, so absent at the last layer
            "crossing_count":  int(crossing[L]) if L < len(crossing) else None,
            "axis_rotation":   _f(axis_rot[L])  if L < len(axis_rot) else None,
            "match_overlap":   _f(overlap[L])   if L < len(overlap)  else None,
        }

        if block3 is not None:
            entry["cone_regime"]        = str(block3["cone_regime"][L])
            entry["cone_margin"]        = _f(block3["cone_margin"][L])
            entry["normalized_margin"]  = _f(block3["normalized_margin"][L])
            entry["cone_escalated"]     = bool(block3["escalated"][L])
            entry["cone_n_binding"]     = int(block3["n_binding"][L])
        else:
            entry["cone_regime"] = entry["cone_margin"] = None
            entry["normalized_margin"] = None
            entry["cone_escalated"] = entry["cone_n_binding"] = None

        if axis is not None:
            entry["cos_axis_mean"] = _f(axis["cos_axis_mean"][L])
            entry["cos_axis_pc1"]  = _f(axis["cos_axis_pc1"][L])
            entry["axis_redundancy"] = str(axis["redundancy"][L])

        per_layer.append(entry)
    return per_layer


def _assemble_per_token(mem_json, aligned_assignments, tokens, n_layers) -> list:
    by_idx = {r["token_idx"]: r for r in mem_json["per_token"]}
    per_token = []
    for i in range(len(tokens)):
        rec = by_idx.get(i, {})
        per_token.append({
            "token_id":               i,
            "token_str":              tokens[i] if i < len(tokens) else None,
            "position":               i,
            "hemisphere_trajectory":  [int(aligned_assignments[L, i])
                                       for L in range(n_layers)],
            "stability_score":        rec.get("stability_score"),
            "border_index":           rec.get("border_index"),
            "first_assignment_layer": rec.get("first_stable_layer"),
            "dominant_hemisphere":    rec.get("dominant_hemisphere"),
            "final_hemisphere":       rec.get("final_hemisphere"),
        })
    return per_token


def _compute_asymmetry(block0: dict) -> dict:
    """Block 4: |A - B| / (A + B) per layer, over valid layers."""
    sizes    = block0["hemisphere_sizes"]
    n_layers = block0["n_layers"]
    asym     = np.full(n_layers, np.nan)

    for L in range(n_layers):
        if not block0["valid"][L]:
            continue
        a, b = int(sizes[L, 0]), int(sizes[L, 1])
        if a + b > 0:
            asym[L] = abs(a - b) / (a + b)

    strong_mask = np.array(
        [str(r) == "strong_bipartition" for r in block0["regime"]])
    strong = asym[strong_mask & np.isfinite(asym)]
    return {
        "asymmetry": asym,
        "mean_asymmetry_strong": float(strong.mean()) if strong.size else None,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _save_run(result: dict, out_dir: Path) -> None:
    stem = result["stem"]
    serializable = {
        k: result[k] for k in
        ("model", "prompt", "n_layers", "n_tokens", "checkpoint_step",
         "frame", "connectivity_floor", "per_layer", "per_token", "summary")
    }
    serializable["events"] = _serialize_events(result["events"])
    if result.get("_block3") is not None:
        cone_json = cone_collapse_to_json(result["_block3"])
        serializable["cone"] = cone_json["summary"]
        # The per-layer half carries the null z-scores, d_eff, and the
        # binding-token support — everything that distinguishes "cone-collapse"
        # from "n < d". It was computed and then dropped at serialization, so
        # nothing downstream could read it. Additive: `cone` keeps its
        # existing shape and meaning.
        serializable["cone_per_layer"] = cone_json["per_layer"]
    if result.get("_axis") is not None:
        serializable["axis_identity"] = axis_identity_to_json(result["_axis"])

    # Block 2's per-layer breakdowns. The summary kept one number from each
    # (hdbscan_nesting_overall, border_vs_noise_mean_auc) and discarded the
    # per-layer tables membership_to_json had already built — which is the
    # depth resolution the Phase 5c question is asked at.
    b2 = result.get("_block2_json") or {}
    for key in ("hdbscan_nesting", "border_vs_noise"):
        if b2.get(key) is not None:
            serializable[key] = b2[key]

    # Block 1's persistence length per layer. Only its derived event counts
    # reached the JSON, and under the antipodal regime_key those are empty by
    # construction (status-1b R4) — so the one array that shows the difference
    # between the two vocabularies was the one not written.
    b1 = result.get("_block1") or {}
    if b1.get("persistence_length") is not None:
        serializable["persistence_length"] = [
            _f(v) for v in np.asarray(b1["persistence_length"]).ravel()
        ]
        serializable["regime_key"] = b1.get("regime_key")

    json_path = out_dir / f"{ARTIFACT_PREFIX}_{stem}.json"
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=_json_default)
    print(f"    Saved: {json_path.name}")

    md_path = out_dir / f"{ARTIFACT_PREFIX}_{stem}.md"
    md_path.write_text(_per_run_md(result))
    print(f"    Saved: {md_path.name}")

    # Activation-space axes, (n_layers, d). axis_identity_to_json drops these
    # with the note that they belong in an npz; no npz was ever written, so
    # cross_checkpoint_axis_rotation / axis_settling_step — the quantity
    # PREDICTIONS.md claim (b) needs — had no input from disk.
    axis = result.get("_axis")
    if axis is not None and axis.get("axes") is not None:
        axes_path = out_dir / f"{ARTIFACT_PREFIX}_{stem}_axes.npz"
        try:
            np.savez_compressed(
                axes_path,
                axes=np.asarray(axis["axes"], dtype=np.float32),
                valid=np.asarray(result["_block0"]["valid"], dtype=bool),
            )
            print(f"    Saved: {axes_path.name}")
        except Exception as exc:
            print(f"    [axes] not written: {exc}")

    try:
        cols, extra = p1b_io.hemisphere_particle_rows(
            model=result["model"], prompt_key=result["prompt"],
            checkpoint_step=(result.get("checkpoint_step")
                             if result.get("checkpoint_step") is not None else -1),
            block0=result["_block0"], block1=result["_block1"],
            block2_json=result["_block2_json"], tokens=result.get("_tokens"),
            cluster_labels=result.get("_hdbscan_labels"))
        p1b_io.write_particle_table(
            cols, extra, out_dir / f"{ARTIFACT_PREFIX}_{stem}_particles.npz")
    except Exception as exc:
        print(f"    [particles] not written: {exc}")


def _per_run_md(result: dict) -> str:
    s = result["summary"]

    def pct(v):  return f"{v * 100:.1f}%" if v is not None else "n/a"
    def num(v, dp=3): return f"{v:.{dp}f}" if v is not None else "n/a"

    ev = s.get("event_counts") or {}
    lines = [
        f"# Phase 1b — {result['model']} / {result['prompt']}",
        "",
        f"**{result['n_layers']} layers, {result['n_tokens']} tokens** "
        f"— frame `{result['frame']['kind']}`, connectivity floor "
        f"{result['connectivity_floor']:g}",
        "",
        "## Block 0 — Bipartition quality",
        f"Antipodal classifier: strong bipartition in "
        f"{pct(s['strong_bipartition_layer_fraction'])} of layers. "
        f"Relative classifier: separated in "
        f"{pct(s.get('separated_layer_fraction'))}, graded in "
        f"{pct(s.get('graded_layer_fraction'))}.",
        "",
        "## Block 1 — Hemisphere tracking",
        (f"Mean axis rotation per transition: {num(s['mean_axis_rotation'])} rad. "
         + ("Events — " + ", ".join(f"{k}: {v}" for k, v in ev.items()) + "."
            if ev else "No hemisphere events detected.")),
        "",
        "## Block 2 — Token membership",
        f"Mean token stability: {num(s.get('mean_stability_score'))}. "
        f"Never stable: {pct(s.get('fraction_never_stable'))}.",
    ]

    n = s.get("hdbscan_nesting_overall")
    if n:
        lines += [
            "",
            "### HDBSCAN nesting",
            f"Fully nested: {pct(n.get('fully_nested_fraction'))}. "
            f"Mixed: {pct(n.get('mixed_fraction'))}. "
            f"Mean |r_c - 0.5|: {num(n.get('mean_r_c_distance_from_half'))}.",
        ]
    if s.get("border_vs_noise_mean_auc") is not None:
        lines += [
            "",
            "### Boundary vs. unclustered population",
            f"AUC {num(s['border_vs_noise_mean_auc'])} — probability that an "
            f"HDBSCAN-noise token sits nearer the Fiedler boundary than a "
            f"clustered one. 0.5 is no relationship.",
        ]

    lines += [
        "",
        "## Block 3 — Cone-collapse",
        (f"Cone-collapse in {pct(s['cone_collapse_layer_fraction'])} of layers; "
         f"mean normalized margin {num(s.get('mean_normalized_cone_margin'))}; "
         f"{s.get('n_layers_escalated_to_full_d', 0)} layers escalated to full d."
         if s.get("cone_collapse_layer_fraction") is not None
         else "Block 3 not run."),
    ]
    if s.get("mean_uniform_cone_fraction") is not None:
        lines += [
            f"Matched nulls: {pct(s.get('mean_shuffled_cone_fraction'))} of "
            f"shuffled-dimension draws and "
            f"{pct(s.get('mean_uniform_cone_fraction'))} of uniform-sphere "
            f"draws are themselves cone-collapsed.",
        ]

    lines += [
        "",
        "## Block 4 — Asymmetry",
        f"Mean asymmetry over strong-bipartition layers: "
        f"{num(s.get('mean_asymmetry_strong'))}.",
        "",
        "## Block A — Axis identity",
        (f"Modal verdict: {s.get('axis_modal_redundancy')}. "
         f"Mean |cos| to the token mean {num(s.get('mean_cos_axis_mean'))}, "
         f"to PC1 {num(s.get('mean_cos_axis_pc1'))} "
         f"(mean-vs-PC1 control {num(s.get('mean_cos_mean_pc1'))})."
         if s.get("axis_modal_redundancy") else "Not computed."),
        "",
        "## Phase 1 cross-reference",
    ]
    x = s.get("crossref_with_phase1", {})
    lines += [
        f"Axis rotation at merge transitions {num(x.get('mean_axis_rotation_at_merge'))} "
        f"rad vs {num(x.get('mean_axis_rotation_off_merge'))} off-merge "
        f"({x.get('n_merges_in_run')} merges). "
        f"Crossing count at violation layers {num(x.get('mean_crossing_at_violation'))} "
        f"vs {num(x.get('mean_crossing_off_violation'))} off-violation "
        f"({x.get('n_violations_in_run')} violation layers).",
    ]
    return "\n".join(lines)


def _write_cross_run(all_results: list, out_dir: Path) -> None:
    by_model: dict = {}
    by_prompt: dict = {}
    for r in all_results:
        by_model.setdefault(r["model"], []).append(r)
        by_prompt.setdefault(r["prompt"], []).append(r)

    cross_run = {
        "by_model":       {m: aggregate(rs) for m, rs in by_model.items()},
        "by_prompt":      {p: aggregate(rs) for p, rs in by_prompt.items()},
        "by_checkpoint":  aggregate_by_checkpoint(all_results),
        "global_verdict": global_verdict(all_results),
    }

    json_path = out_dir / f"{ARTIFACT_PREFIX}_cross_run.json"
    with open(json_path, "w") as f:
        json.dump(cross_run, f, indent=2, default=_json_default)
    print(f"  Saved: {json_path.name}")

    md_path = out_dir / f"{ARTIFACT_PREFIX}_cross_run.md"
    md_path.write_text(cross_run_markdown(cross_run, by_model, by_prompt))
    print(f"  Saved: {md_path.name}")


def _write_experiment_txt(timestamp, models, prompts, phase1_dir, opts) -> None:
    from core.config import DEVICE
    lines = [
        "phase       : 1b (hemisphere investigation)",
        f"timestamp   : {timestamp}",
        f"command     : {' '.join(sys.argv)}",
        f"models      : {models}",
        f"prompts     : {prompts}",
        f"phase1_dir  : {phase1_dir}",
        f"device      : {DEVICE}",
    ]
    lines += [f"{k:<12}: {v}" for k, v in sorted(opts.items())]
    lines += ["", "--- prompt texts ---"]
    for key in prompts:
        lines += [f"[{key}]", PROMPTS.get(key, ""), ""]
    (OUTPUT_DIR / "experiment.txt").write_text("\n".join(lines))


def _serialize_events(events: list) -> list:
    return [{k: (_json_default(v) if isinstance(v, np.generic) else v)
             for k, v in ev.items()} for ev in events]


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 1b: Hemispheric Structure Investigation")
    # No choices= over 47 registry keys: a typo used to render as an
    # unreadable wall of options, and resolve_model_names gives a better
    # error. Groups and keys are accepted in the same position.
    p.add_argument("--models", nargs="+", default=None, metavar="NAME",
                   help="registry keys and/or group names "
                        f"({', '.join(sorted(MODEL_GROUPS))}). "
                        "Default: the Blog 1 architectures.")
    p.add_argument("--list-models", action="store_true",
                   help="print the group/registry catalogue and exit")
    p.add_argument("--prompts", nargs="+", default=list(PROMPTS.keys()),
                   choices=list(PROMPTS.keys()))
    p.add_argument("--fast", action="store_true",
                   help="albert-base-v2 + wiki_paragraph, 4 legacy snapshots")
    p.add_argument("--no-cone", action="store_true", help="skip Block 3")
    p.add_argument("--n-null", type=int, default=0, metavar="N",
                   help="null replicates per layer for Block 3 (0 = off)")
    p.add_argument("--phase1-dir", type=str, default=None, metavar="DIR")
    p.add_argument("--from-phase1", action="store_true",
                   help="analyse Phase 1's saved activations instead of "
                        "running a forward pass (requires --phase1-dir)")
    p.add_argument("--frame", type=str, default="l2_sphere",
                   choices=["l2_sphere", "raw", "identity"])
    p.add_argument("--connectivity-floor", type=float,
                   default=CONNECTIVITY_FLOOR,
                   help="uniform edge weight added before the Laplacian. "
                        "0.0 reproduces Phase 1's graph exactly.")
    p.add_argument("--regime-key", type=str, default="regime",
                   choices=["regime", "regime_relative"],
                   help="which Block 0 vocabulary drives Block 1 events")
    p.add_argument("--drop-pos0", action="store_true",
                   help="exclude position 0 from Block 3's LP")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None, metavar="DIR")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.list_models:
        print_model_catalogue(MODEL_CONFIGS, MODEL_GROUPS, DEFAULT_MODELS)
        return

    if args.fast:
        models  = ["albert-base-v2"]
        prompts = ["wiki_paragraph"]
        # Set on the module, and read from the module at call time. The
        # previous version set these after binding the names at import, so
        # the override never reached the extraction call.
        cfg.ALBERT_SNAPSHOTS = list(
            getattr(cfg, "ALBERT_SNAPSHOTS_LEGACY", [12, 24, 36, 48]))
        cfg.ALBERT_MAX_ITERATIONS = max(cfg.ALBERT_SNAPSHOTS)
    else:
        models  = resolve_model_names(
            args.models if args.models is not None else list(DEFAULT_MODELS),
            MODEL_CONFIGS, MODEL_GROUPS)
        prompts = args.prompts

    timer = RunTimer()
    timer.__enter__()
    results = run_all(
        models_to_run=models,
        prompts_to_run=prompts,
        run_cone=not args.no_cone,
        phase1_dir=Path(args.phase1_dir) if args.phase1_dir else None,
        from_phase1=args.from_phase1,
        n_null=args.n_null,
        frame_kind=args.frame,
        connectivity_floor=args.connectivity_floor,
        regime_key=args.regime_key,
        drop_pos0=args.drop_pos0,
        seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    timer.__exit__(None, None, None)

    try:
        from core.prompts import PROMPT_BATTERY_HASH, PROMPT_BATTERY_VERSION
        battery_hash, battery_version = PROMPT_BATTERY_HASH, PROMPT_BATTERY_VERSION
    except Exception:
        battery_hash, battery_version = None, None

    try:
        # write_manifest's contract is one manifest per run directory, keyed
        # on a single model — matching core.artifacts.MANIFEST.required_keys.
        # A Phase 1b invocation covers several models, so `model` carries the
        # selection and the full list goes in `extra`, rather than bending
        # the required key into a list and breaking validate_artifact.
        write_manifest(
            OUTPUT_DIR,
            model=(models[0] if len(models) == 1 else f"<{len(models)} models>"),
            prompt_battery_hash=battery_hash or "",
            wall_time_seconds=timer.elapsed,
            seeds={"run": int(args.seed)},
            config={
                "phase": "1b",
                "frame": args.frame,
                "connectivity_floor": float(args.connectivity_floor),
                "regime_key": args.regime_key,
                "n_null": int(args.n_null),
                "drop_pos0": bool(args.drop_pos0),
                "run_cone": not args.no_cone,
                "from_phase1": bool(args.from_phase1),
            },
            extra={
                "prompt_battery_version": battery_version,
                "models": models,
                "prompts": prompts,
                "n_runs": len(results),
                "phase1_dir": str(args.phase1_dir) if args.phase1_dir else None,
            },
        )
    except Exception as exc:
        print(f"[manifest] not written: {exc}")


if __name__ == "__main__":
    main()
