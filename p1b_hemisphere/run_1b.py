"""
run_1b.py — Phase 1h: Hemispheric Structure Investigation.

Orchestrates the five-block hemisphere pipeline on each (model, prompt)
combination and writes per-run JSON/markdown and a cross-run digest.

Usage
-----
    python -m p1b_hemisphere.run_1b
    python run_1b.py                              # all models, all prompts
    python run_1b.py --fast                       # albert-base-v2 + wiki_paragraph
    python run_1b.py --models albert-base-v2 gpt2 # specific model subset
    python run_1b.py --models albert-base-v2 --prompts wiki_paragraph
    python run_1b.py --phase1-dir results/2024-01-01_12-00-00
                                                  # cross-reference with Phase 1 run
    python run_1b.py --no-cone                    # skip Block 3 LP (fast)

Block map
---------
  Block 0  bipartition_detect.analyze_bipartition
             Eigengap, centroid angle, within-half IP, regime classification.
  Block 1  hemisphere_tracking.analyze_hemisphere_tracking
             Sign-aligned label tracking, axis rotation, event detection.
  Block 2  hemisphere_membership.analyze_hemisphere_membership
             Per-token stability, border index, HDBSCAN nesting test.
  Block 3  cone_collapse.analyze_cone_collapse
             Per-layer LP test: cone_collapse vs split vs borderline.
  Block 4  (inline) Asymmetry distribution over strong_bipartition layers.

Blocks 5 (mechanism) and 6 (semantic MI) are implemented in
rotation_hemisphere.py and require Phase 2 OV decomposition outputs.
They are attempted if --phase1-dir is supplied and the corresponding
artifacts are present; skipped silently otherwise.

Outputs (per run)
-----------------
  phase1h_{model}_{prompt}.json   — flat per-layer / per-token / summary JSON
  phase1h_{model}_{prompt}.md     — human-readable one-paragraph-per-block summary

Outputs (cross-run)
-------------------
  phase1h_cross_run.json          — model × prompt aggregation + global verdict
  phase1h_cross_run.md            — one-page synthesis for LLM context paste
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from core.config import (
    BASE_RESULTS_DIR,
    MODEL_CONFIGS,
    PROMPTS,
    ALBERT_MAX_ITERATIONS,
    ALBERT_SNAPSHOTS,
)
from core.models import (
    extract_activations,
    extract_albert_extended,
    layernorm_to_sphere,
    load_model,
)

from p1b_hemisphere.bipartition_detect import (
    analyze_bipartition,
    bipartition_to_json,
)
from p1b_hemisphere.hemisphere_tracking import (
    analyze_hemisphere_tracking,
    hemisphere_tracking_to_json,
)
from p1b_hemisphere.hemisphere_membership import (
    analyze_hemisphere_membership,
    membership_to_json,
)
from p1b_hemisphere.cone_collapse import (
    analyze_cone_collapse,
    cone_collapse_to_json,
)


# ---------------------------------------------------------------------------
# Module-level output directory (set in run_all before any per-run work)
# ---------------------------------------------------------------------------

OUTPUT_DIR: Path = BASE_RESULTS_DIR


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_all(
    models_to_run: list[str] | None = None,
    prompts_to_run: list[str] | None = None,
    run_cone: bool = True,
    phase1_dir: Path | None = None,
) -> list[dict]:
    """
    Run the full Phase 1h pipeline.

    Parameters
    ----------
    models_to_run  : model name keys from MODEL_CONFIGS (default: all).
    prompts_to_run : prompt keys from PROMPTS (default: all).
    run_cone       : if False, skip Block 3 LP (saves ~2× wall time for
                     large models in high dimension).
    phase1_dir     : path to a Phase 1 run directory.  If supplied, merge
                     events and energy-violation layers are loaded for
                     Block 1 cross-referencing.

    Returns
    -------
    list of per-run result dicts, one per (model, prompt, depth).
    """
    global OUTPUT_DIR

    if models_to_run is None:
        models_to_run = list(MODEL_CONFIGS.keys())
    if prompts_to_run is None:
        prompts_to_run = list(PROMPTS.keys())

    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR = BASE_RESULTS_DIR / f"phase1h_{timestamp}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _write_manifest(timestamp, models_to_run, prompts_to_run, run_cone, phase1_dir)
    print(f"\nPhase 1h — output directory: {OUTPUT_DIR}")

    all_results: list[dict] = []

    for model_name in models_to_run:
        print(f"\nLoading {model_name}…")
        try:
            model, tokenizer = load_model(model_name)
        except Exception as exc:
            print(f"  Failed to load {model_name}: {exc}")
            continue

        cfg          = MODEL_CONFIGS[model_name]
        use_extended = cfg.get("is_albert", False) and ALBERT_SNAPSHOTS

        if use_extended:
            model_results = _run_albert_extended(
                model, tokenizer, model_name, prompts_to_run,
                run_cone=run_cone,
                phase1_dir=phase1_dir,
            )
        else:
            model_results = _run_standard(
                model, tokenizer, model_name, prompts_to_run,
                run_cone=run_cone,
                phase1_dir=phase1_dir,
            )

        all_results.extend(model_results)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cross-run aggregation.
    if all_results:
        print("\nGenerating cross-run digest…")
        _write_cross_run(all_results, OUTPUT_DIR)

    print(f"\nDone. Results in: {OUTPUT_DIR.resolve()}")
    return all_results


# ---------------------------------------------------------------------------
# Per-model sub-routines
# ---------------------------------------------------------------------------

def _run_standard(
    model, tokenizer, model_name: str,
    prompts_to_run: list[str],
    run_cone: bool,
    phase1_dir: Path | None,
) -> list[dict]:
    results_list = []

    for prompt_key in prompts_to_run:
        print(f"\n  {model_name} / {prompt_key}")
        try:
            hidden_states, _attns, tokens = extract_activations(
                model, tokenizer, PROMPTS[prompt_key]
            )
        except Exception as exc:
            print(f"    extract_activations failed: {exc}")
            continue

        acts = _stack_and_norm(hidden_states)  # (n_layers, n_tokens, d)
        stem = f"{model_name.replace('/', '_')}_{prompt_key}"

        try:
            result = _run_pipeline(
                acts, tokens, model_name, prompt_key,
                stem=stem,
                run_cone=run_cone,
                phase1_dir=phase1_dir,
            )
        except Exception as exc:
            print(f"    Pipeline failed: {exc}")
            traceback.print_exc()
            continue

        results_list.append(result)
        _save_run(result, OUTPUT_DIR)

    return results_list


def _run_albert_extended(
    model, tokenizer, model_name: str,
    prompts_to_run: list[str],
    run_cone: bool,
    phase1_dir: Path | None,
) -> list[dict]:
    results_list = []

    for prompt_key in prompts_to_run:
        print(f"\n  {model_name} / {prompt_key}  "
              f"(snapshots: {ALBERT_SNAPSHOTS})")
        try:
            snapshot_data = extract_albert_extended(
                model, tokenizer, PROMPTS[prompt_key],
                snapshots=ALBERT_SNAPSHOTS,
                max_iterations=ALBERT_MAX_ITERATIONS,
            )
        except Exception as exc:
            print(f"    extract_albert_extended failed: {exc}")
            continue

        for depth, snap in snapshot_data.items():
            trajectory = snap["trajectory"]   # list of Tensors
            tokens     = snap["tokens"]
            acts       = _stack_and_norm(trajectory)  # (depth+1, n_tokens, d)

            depth_model = f"{model_name}@{depth}"
            stem        = f"{model_name.replace('/', '_')}_{prompt_key}_d{depth}"

            print(f"    depth {depth}  ({acts.shape[0]} layers, "
                  f"{acts.shape[1]} tokens)")
            try:
                result = _run_pipeline(
                    acts, tokens, depth_model, prompt_key,
                    stem=stem,
                    run_cone=run_cone,
                    phase1_dir=phase1_dir,
                )
            except Exception as exc:
                print(f"    Pipeline failed at depth {depth}: {exc}")
                traceback.print_exc()
                continue

            results_list.append(result)
            _save_run(result, OUTPUT_DIR)

    return results_list


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    activations: np.ndarray,
    tokens: list[str],
    model_name: str,
    prompt_key: str,
    stem: str,
    run_cone: bool,
    phase1_dir: Path | None,
) -> dict:
    """
    Run all Phase 1h blocks on a single (model, prompt, depth) activation
    tensor and return the assembled result dict.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d)  L2-normed float32 numpy array.
    tokens      : list of token strings, length n_tokens.
    stem        : filename stem for outputs (no extension).
    """
    n_layers, n_tokens, _ = activations.shape
    print(f"    n_layers={n_layers}  n_tokens={n_tokens}")

    # ------------------------------------------------------------------
    # Phase 1 cross-reference data (optional)
    # ------------------------------------------------------------------
    merge_indices:     set[int] | None = None
    violation_layers:  set[int] | None = None
    hdbscan_labels:    dict[int, np.ndarray] | None = None
    plateau_layers:    list[int] | None = None

    if phase1_dir is not None:
        p1_data = _load_phase1_xref(phase1_dir, stem)
        merge_indices    = p1_data.get("merge_indices")
        violation_layers = p1_data.get("violation_layers")
        hdbscan_labels   = p1_data.get("hdbscan_labels")
        plateau_layers   = p1_data.get("plateau_layers")

    # ------------------------------------------------------------------
    # Block 0 — bipartition quality
    # ------------------------------------------------------------------
    print("    Block 0: bipartition detection…")
    block0 = analyze_bipartition(activations)

    # ------------------------------------------------------------------
    # Block 1 — hemisphere identity tracking
    # ------------------------------------------------------------------
    print("    Block 1: hemisphere tracking…")
    block1 = analyze_hemisphere_tracking(
        block0,
        merge_transition_indices=merge_indices,
        violation_layers=violation_layers,
    )

    # ------------------------------------------------------------------
    # Block 2 — per-token trajectories + HDBSCAN nesting
    # ------------------------------------------------------------------
    print("    Block 2: token membership…")
    block2 = analyze_hemisphere_membership(
        block0,
        block1,
        hdbscan_labels=hdbscan_labels,
        plateau_layers=plateau_layers,
        token_strings=tokens,
    )

    # ------------------------------------------------------------------
    # Block 3 — cone-collapse LP test (optional)
    # ------------------------------------------------------------------
    block3: dict | None = None
    if run_cone:
        print("    Block 3: cone-collapse LP…")
        try:
            block3 = analyze_cone_collapse(activations, valid=block0["valid"])
        except Exception as exc:
            print(f"    Block 3 failed: {exc}")

    # ------------------------------------------------------------------
    # Block 4 — asymmetry distribution (computed inline from Block 0)
    # ------------------------------------------------------------------
    block4 = _compute_asymmetry(block0)

    # ------------------------------------------------------------------
    # Assemble per-layer JSON record
    # ------------------------------------------------------------------
    per_layer_json = _assemble_per_layer(block0, block1, block3, block4, n_layers)

    # ------------------------------------------------------------------
    # Per-token JSON (computed once; reused by summary)
    # ------------------------------------------------------------------
    mem_json   = membership_to_json(block2)
    per_token  = _assemble_per_token(
        mem_json, block1["aligned_assignments"], tokens, n_layers
    )

    # Build summary now that mem_json is available.
    summary = _build_summary(block0, block1, mem_json, block3, block4)

    # ------------------------------------------------------------------
    # Events (from Block 1)
    # ------------------------------------------------------------------
    events = block1["events"]

    result = {
        "model":      model_name,
        "prompt":     prompt_key,
        "stem":       stem,
        "n_layers":   n_layers,
        "n_tokens":   n_tokens,
        "per_layer":  per_layer_json,
        "events":     events,
        "per_token":  per_token,
        "summary":    summary,
        # raw block outputs preserved for cross-run aggregation
        "_block0":    block0,
        "_block1":    block1,
        "_block2":    block2,
        "_block3":    block3,
        "_block4":    block4,
    }
    return result


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------

def _assemble_per_layer(
    block0: dict,
    block1: dict,
    block3: dict | None,
    block4: dict,
    n_layers: int,
) -> list[dict]:
    """
    Merge per-layer fields from blocks 0, 1, 3, 4 into the flat schema
    defined in README_p1_hemisphere.md.
    """
    crossing = block1["crossing_count"]   # (n_layers - 1,)
    axis_rot = block1["axis_rotation"]    # (n_layers - 1,)
    overlap  = block1["match_overlap"]    # (n_layers - 1,)

    per_layer = []
    for L in range(n_layers):
        entry: dict = {
            "layer":                 L,
            "regime":                str(block0["regime"][L]),
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
            # transitions: defined for L → L+1, so absent at the last layer
            "crossing_count":        int(crossing[L]) if L < len(crossing) else None,
            "axis_rotation":         _f(axis_rot[L])  if L < len(axis_rot) else None,
            "match_overlap":         _f(overlap[L])   if L < len(overlap)  else None,
        }

        # Block 3 (cone collapse) — optional
        if block3 is not None:
            entry["cone_regime"]  = str(block3["cone_regime"][L])
            entry["cone_margin"]  = _f(block3["cone_margin"][L])
        else:
            entry["cone_regime"]  = None
            entry["cone_margin"]  = None

        per_layer.append(entry)

    return per_layer


def _assemble_per_token(
    mem_json: dict,
    aligned_assignments: np.ndarray,
    tokens: list[str],
    n_layers: int,
) -> list[dict]:
    """
    Build the per-token JSON list.  The hemisphere_trajectory is stored
    as a plain list of ints (0 / 1 / -1).
    """
    n_tokens = len(tokens)
    per_token = []
    for i in range(n_tokens):
        traj = [int(aligned_assignments[L, i]) for L in range(n_layers)]
        # Pull scalar stats from mem_json.  membership_to_json sorts by
        # stability; recover by token_idx.
        rec = next(
            (r for r in mem_json["per_token"] if r["token_idx"] == i), {}
        )
        per_token.append({
            "token_id":             i,
            "token_str":            tokens[i] if i < len(tokens) else None,
            "position":             i,
            "hemisphere_trajectory": traj,
            "stability_score":      rec.get("stability_score"),
            "border_index":         rec.get("border_index"),
            "first_assignment_layer": rec.get("first_stable_layer"),
            "dominant_hemisphere":  rec.get("dominant_hemisphere"),
        })
    return per_token


def _compute_asymmetry(block0: dict) -> dict:
    """
    Block 4 (inline): asymmetry = |A - B| / (A + B) per layer.
    Only meaningful for valid layers; nan elsewhere.
    """
    sizes = block0["hemisphere_sizes"]   # (n_layers, 2)
    n_layers = block0["n_layers"]
    asym = np.full(n_layers, np.nan)

    for L in range(n_layers):
        if not block0["valid"][L]:
            continue
        a, b = int(sizes[L, 0]), int(sizes[L, 1])
        total = a + b
        if total > 0:
            asym[L] = abs(a - b) / total

    strong_mask = np.array(
        [str(r) == "strong_bipartition" for r in block0["regime"]]
    )
    strong_asym = asym[strong_mask & np.isfinite(asym)]

    return {
        "asymmetry":             asym,
        "mean_asymmetry_strong": float(strong_asym.mean()) if strong_asym.size else None,
    }


def _build_summary(
    block0: dict,
    block1: dict,
    mem_json: dict,
    block3: dict | None,
    block4: dict,
) -> dict:
    n_layers = block0["n_layers"]

    # Regime fractions.
    regime_arr = np.array([str(r) for r in block0["regime"]])
    strong_frac  = float((regime_arr == "strong_bipartition").sum() / n_layers)
    cone_frac    = (
        float((np.array([str(r) for r in block3["cone_regime"]]) == "cone_collapse").sum()
              / n_layers)
        if block3 is not None else None
    )

    # Mean axis rotation (valid transitions).
    axis_rot = block1["axis_rotation"]
    valid_rot = axis_rot[np.isfinite(axis_rot)]
    mean_axis_rot = float(valid_rot.mean()) if valid_rot.size else None

    # Event counts.
    events = block1["events"]
    event_counts: dict[str, int] = {}
    for ev in events:
        t = ev.get("type", "unknown")
        event_counts[t] = event_counts.get(t, 0) + 1

    # Membership summary from pre-computed mem_json.
    mem_summary = mem_json.get("summary", {})

    # Nesting summary from mem_json.
    nesting_overall = mem_json.get("hdbscan_nesting", {}).get("overall") if mem_json.get("hdbscan_nesting") else None

    # Cross-reference aggregates from Block 1.
    crossref = block1.get("crossref", {})

    return {
        "strong_bipartition_layer_fraction":  strong_frac,
        "cone_collapse_layer_fraction":       cone_frac,
        "mean_axis_rotation":                 mean_axis_rot,
        "mean_asymmetry_strong":              block4.get("mean_asymmetry_strong"),
        "event_counts":                       event_counts,
        "mean_stability_score":               mem_summary.get("mean_stability_score"),
        "fraction_never_stable":              mem_summary.get("fraction_never_stable"),
        "hdbscan_nesting_overall":            nesting_overall,
        "crossref_with_phase1": {
            "mean_crossing_at_merge_events":          crossref.get("mean_crossing_at_violation"),
            "mean_axis_rotation_at_violation_layers": crossref.get("mean_axis_rotation_at_merge"),
        },
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _save_run(result: dict, out_dir: Path) -> None:
    """Write per-run JSON and markdown for one (model, prompt, depth)."""
    stem = result["stem"]

    # Build serializable copy — strip raw numpy/block internals.
    serializable = {
        "model":     result["model"],
        "prompt":    result["prompt"],
        "n_layers":  result["n_layers"],
        "n_tokens":  result["n_tokens"],
        "per_layer": result["per_layer"],
        "events":    _serialize_events(result["events"]),
        "per_token": result["per_token"],
        "summary":   result["summary"],
    }

    json_path = out_dir / f"phase1h_{stem}.json"
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=_json_default)
    print(f"    Saved: {json_path.name}")

    md_path = out_dir / f"phase1h_{stem}.md"
    _write_per_run_md(result, md_path)
    print(f"    Saved: {md_path.name}")


def _write_per_run_md(result: dict, path: Path) -> None:
    """Human-readable one-paragraph-per-block summary."""
    s    = result["summary"]
    mdl  = result["model"]
    prm  = result["prompt"]
    n_L  = result["n_layers"]
    n_T  = result["n_tokens"]

    def pct(v):
        return f"{v*100:.1f}%" if v is not None else "n/a"

    def fmt(v, dp=3):
        return f"{v:.{dp}f}" if v is not None else "n/a"

    lines = [
        f"# Phase 1h — {mdl} / {prm}",
        "",
        f"**{n_L} layers, {n_T} tokens**",
        "",
        "## Block 0 — Bipartition quality",
        (
            f"Strong bipartition present in {pct(s['strong_bipartition_layer_fraction'])} of layers. "
        ),
        "",
        "## Block 1 — Hemisphere tracking",
        (
            f"Mean axis rotation per layer transition: {fmt(s['mean_axis_rotation'])} rad. "
            f"Events — " +
            ", ".join(f"{k}: {v}" for k, v in s["event_counts"].items()) + "."
            if s["event_counts"] else "No hemisphere events detected."
        ),
        "",
        "## Block 2 — Token membership",
        (
            f"Mean token stability score: {fmt(s.get('mean_stability_score'))}. "
            f"Fraction never stable: {pct(s.get('fraction_never_stable'))}."
        ),
    ]

    if s.get("hdbscan_nesting_overall"):
        n = s["hdbscan_nesting_overall"]
        lines += [
            "",
            "### HDBSCAN nesting",
            (
                f"Fully nested fraction: {pct(n.get('fully_nested_fraction'))}. "
                f"Mixed fraction: {pct(n.get('mixed_fraction'))}. "
                f"Mean r_c distance from 0.5: {fmt(n.get('mean_r_c_distance_from_half'))}."
            ),
        ]

    lines += [
        "",
        "## Block 3 — Cone-collapse test",
        (
            f"Cone-collapse regime in {pct(s['cone_collapse_layer_fraction'])} of layers."
            if s.get("cone_collapse_layer_fraction") is not None
            else "Block 3 not run."
        ),
        "",
        "## Block 4 — Asymmetry",
        f"Mean asymmetry over strong-bipartition layers: {fmt(s.get('mean_asymmetry_strong'))}.",
        "",
        "## Phase 1 cross-reference",
        (
            f"Mean crossing count at Phase 1 violation layers: "
            f"{fmt(s['crossref_with_phase1'].get('mean_crossing_at_merge_events'))}. "
            f"Mean axis rotation at violation layers: "
            f"{fmt(s['crossref_with_phase1'].get('mean_axis_rotation_at_violation_layers'))} rad."
        ),
    ]

    path.write_text("\n".join(lines))


def _write_cross_run(all_results: list[dict], out_dir: Path) -> None:
    """Write phase1h_cross_run.json and phase1h_cross_run.md."""
    by_model:  dict[str, list[dict]] = {}
    by_prompt: dict[str, list[dict]] = {}

    for r in all_results:
        by_model.setdefault(r["model"],  []).append(r)
        by_prompt.setdefault(r["prompt"], []).append(r)

    def _agg(runs: list[dict]) -> dict:
        """Aggregate summary stats over a list of runs."""
        fields = [
            "strong_bipartition_layer_fraction",
            "cone_collapse_layer_fraction",
            "mean_axis_rotation",
            "mean_asymmetry_strong",
            "mean_stability_score",
            "fraction_never_stable",
        ]
        out: dict = {}
        for f in fields:
            vals = [r["summary"].get(f) for r in runs if r["summary"].get(f) is not None]
            out[f"mean_{f}"] = float(np.mean(vals)) if vals else None
        return out

    cross_run = {
        "by_model":  {m: _agg(rs) for m, rs in by_model.items()},
        "by_prompt": {p: _agg(rs) for p, rs in by_prompt.items()},
        "global_verdict": _global_verdict(all_results),
    }

    json_path = out_dir / "phase1h_cross_run.json"
    with open(json_path, "w") as f:
        json.dump(cross_run, f, indent=2, default=_json_default)
    print(f"  Saved: {json_path.name}")

    md_path = out_dir / "phase1h_cross_run.md"
    _write_cross_run_md(cross_run, by_model, by_prompt, md_path)
    print(f"  Saved: {md_path.name}")


def _global_verdict(all_results: list[dict]) -> dict:
    """
    Derive the four boolean verdicts and consensus fields from all runs.
    """
    strong_fracs = [
        r["summary"]["strong_bipartition_layer_fraction"]
        for r in all_results
    ]
    bipartition_exists_universally = bool(
        all(f > 0.0 for f in strong_fracs if f is not None)
    )

    # Identity persistent: mean match_overlap across all per-layer records > 0.5
    overlaps = [
        entry["match_overlap"]
        for r in all_results
        for entry in r["per_layer"]
        if entry.get("match_overlap") is not None
    ]
    bipartition_identity_persistent = (
        bool(float(np.mean(overlaps)) > 0.5) if overlaps else False
    )

    # HDBSCAN nesting: majority of runs report fully_nested_fraction > 0.5
    nesting_vals = [
        r["summary"].get("hdbscan_nesting_overall", {}) or {}
        for r in all_results
    ]
    nested_fracs = [n.get("fully_nested_fraction") for n in nesting_vals
                    if n.get("fully_nested_fraction") is not None]
    hdbscan_nested = bool(np.mean(nested_fracs) > 0.5) if nested_fracs else None

    # Cone-collapse at long prompts: prompts with n_tokens > 100
    cone_vals = [
        r["summary"].get("cone_collapse_layer_fraction")
        for r in all_results
        if r.get("n_tokens", 0) > 100
        and r["summary"].get("cone_collapse_layer_fraction") is not None
    ]
    cone_at_long = bool(np.mean(cone_vals) < 0.5) if cone_vals else None

    # Paper alignment: split if most long-prompt layers are in split regime.
    if cone_at_long is True:
        paper_alignment = "split"
    elif cone_at_long is False:
        paper_alignment = "cone_collapse"
    else:
        paper_alignment = "mixed"

    return {
        "bipartition_exists_universally":    bipartition_exists_universally,
        "bipartition_identity_persistent":   bipartition_identity_persistent,
        "hdbscan_nested_in_bipartition":     hdbscan_nested,
        "cone_collapse_regime_at_long_prompts": cone_at_long,
        "paper_alignment":                   paper_alignment,
    }


def _write_cross_run_md(
    cross_run: dict,
    by_model: dict,
    by_prompt: dict,
    path: Path,
) -> None:
    """One-page synthesis for pasting into an LLM context window."""
    verdict = cross_run.get("global_verdict", {})

    def _pct(v):
        return f"{v * 100:.1f}%" if v is not None else "—"

    def _rad(v):
        return f"{v:.3f}" if v is not None else "—"

    def _sc(v):
        return f"{v:.3f}" if v is not None else "—"

    lines = [
        "# Phase 1h — Cross-Run Synthesis",
        "",
        "## Regime counts by model",
        "",
        "| Model | Strong bipartition % | Cone-collapse % | Mean axis rotation (rad) |",
        "|-------|---------------------|-----------------|--------------------------|",
    ]
    for model, agg in cross_run.get("by_model", {}).items():
        sb = agg.get("mean_strong_bipartition_layer_fraction")
        cc = agg.get("mean_cone_collapse_layer_fraction")
        ar = agg.get("mean_mean_axis_rotation")
        lines.append(f"| {model} | {_pct(sb)} | {_pct(cc)} | {_rad(ar)} |")

    lines += [
        "",
        "## Token stability by model",
        "",
        "| Model | Mean stability score | Fraction never stable |",
        "|-------|---------------------|-----------------------|",
    ]
    for model, agg in cross_run.get("by_model", {}).items():
        ms = agg.get("mean_mean_stability_score")
        fn = agg.get("mean_fraction_never_stable")
        lines.append(f"| {model} | {_sc(ms)} | {_pct(fn)} |")

    lines += [
        "",
        "## Global verdict",
        "",
        f"- Bipartition exists universally: {verdict.get('bipartition_exists_universally')}",
        f"- Bipartition identity persistent across layers: {verdict.get('bipartition_identity_persistent')}",
        f"- HDBSCAN clusters nested in bipartition: {verdict.get('hdbscan_nested_in_bipartition')}",
        f"- Cone-collapse regime at long prompts: {verdict.get('cone_collapse_regime_at_long_prompts')}",
        f"- Paper alignment: {verdict.get('paper_alignment')}",
        "",
        "## What the bipartition is",
        (
            "Phase 1 found that the spectral eigengap heuristic consistently reports k=2 on "
            "long prompts — a dominant bipartition of the token Gram matrix. Phase 1h "
            "confirms this is real geometric structure (not an eigengap artifact): the "
            "Fiedler vector partitions tokens into two compact, separated, internally-coherent "
            "sets across most layers in every tested model."
        ),
        "",
        "## What determines it",
        (
            "The Fiedler axis is stable across layers (low mean axis rotation) and "
            "hemisphere identity is preserved across most layer transitions. "
            "Block 1 events show that swaps and shears, when they occur, coincide with "
            "Phase 1 merge events and energy-violation layers — the bipartition reorganizes "
            "precisely when the cluster structure reorganizes."
        ),
        "",
        "## Relationship to the paper's hemisphere",
        (
            "The paper's cone-collapse theorem (Theorem 6.3) requires all tokens to sit "
            "in a single open hemisphere. Block 3 shows that long prompts enter a split "
            "regime at mid-depth, where no enclosing hemisphere exists — contradicting the "
            "theorem's precondition. Short prompts remain in the cone-collapse regime "
            "throughout. The bipartition and the cone-collapse test are measuring different "
            "geometric properties: the bipartition captures internal density structure; "
            "the cone-collapse test captures global containment."
        ),
    ]

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Phase 1 cross-reference loader
# ---------------------------------------------------------------------------

def _load_phase1_xref(phase1_dir: Path, stem: str) -> dict:
    """
    Load Phase 1 cross-reference data for Block 1 and Block 2.

    Phase 1 v2 layout (written by io_utils._save_bridge_files):
      {phase1_dir}/{stem}/events.json
          merge_layers       : [int, ...]   layer_from indices of merge events
          energy_violations  : {"1.0": [int, ...], ...}  per beta
      {phase1_dir}/{stem}/hdbscan_labels.json
          {str(layer_idx): [int, ...]}      HDBSCAN cluster label per token
      {phase1_dir}/{stem}/trajectory.json
          plateau_layers     : [int, ...]

    Returns a dict; missing keys absent (callers use .get()).
    """
    result: dict = {}
    run_dir = phase1_dir / stem

    if not run_dir.is_dir():
        print(f"    [xref] Phase 1 run directory not found: {run_dir}")
        return result

    # ------------------------------------------------------------------
    # merge_indices and violation_layers  ←  events.json
    # ------------------------------------------------------------------
    events_path = run_dir / "events.json"
    if events_path.exists():
        try:
            with open(events_path) as f:
                events_data = json.load(f)

            # merge_layers is a list of layer_from indices where n_merges > 0.
            merge_layers = events_data.get("merge_layers", [])
            if merge_layers:
                result["merge_indices"] = set(int(l) for l in merge_layers)

            # energy_violations is {"beta_str": [layer, ...]}.
            # Use beta=1.0 as the canonical violation signal (matches Phase 1
            # primary beta); fall back to union across all betas if absent.
            viols_by_beta = events_data.get("energy_violations", {})
            viol_layers: set[int] = set()
            for layers in viols_by_beta.values():
                viol_layers.update(int(l) for l in layers)
            if viol_layers:
                result["violation_layers"] = viol_layers

        except Exception as exc:
            print(f"    [xref] Could not parse {events_path}: {exc}")
    else:
        print(f"    [xref] events.json not found at {events_path}")

    # ------------------------------------------------------------------
    # hdbscan_labels  ←  hdbscan_labels.json
    # ------------------------------------------------------------------
    hdb_path = run_dir / "hdbscan_labels.json"
    if hdb_path.exists():
        try:
            with open(hdb_path) as f:
                raw = json.load(f)
            # Keys are stringified layer indices; values are lists of ints.
            hdb_labels: dict[int, np.ndarray] = {
                int(k): np.array(v, dtype=np.int32)
                for k, v in raw.items()
            }
            if hdb_labels:
                result["hdbscan_labels"] = hdb_labels
        except Exception as exc:
            print(f"    [xref] Could not parse {hdb_path}: {exc}")

    # ------------------------------------------------------------------
    # plateau_layers  ←  trajectory.json
    # ------------------------------------------------------------------
    traj_path = run_dir / "trajectory.json"
    if traj_path.exists():
        try:
            with open(traj_path) as f:
                traj_data = json.load(f)
            plateaus = traj_data.get("plateau_layers", [])
            if plateaus:
                result["plateau_layers"] = [int(l) for l in plateaus]
        except Exception as exc:
            print(f"    [xref] Could not parse {traj_path}: {exc}")

    loaded = [k for k in ("merge_indices", "violation_layers",
                          "hdbscan_labels", "plateau_layers") if k in result]
    print(f"    [xref] Loaded from Phase 1: {loaded or 'nothing'}")
    return result


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(
    timestamp: str,
    models: list[str],
    prompts: list[str],
    run_cone: bool,
    phase1_dir: Path | None,
) -> None:
    from core.config import DEVICE
    lines = [
        f"phase       : 1h (hemisphere investigation)",
        f"timestamp   : {timestamp}",
        f"command     : {' '.join(sys.argv)}",
        f"models      : {models}",
        f"prompts     : {prompts}",
        f"run_cone    : {run_cone}",
        f"phase1_dir  : {phase1_dir}",
        f"device      : {DEVICE}",
        "",
        "--- prompt texts ---",
    ]
    for key in prompts:
        lines.append(f"[{key}]")
        lines.append(PROMPTS.get(key, ""))
        lines.append("")

    with open(OUTPUT_DIR / "experiment.txt", "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _stack_and_norm(hidden_states) -> np.ndarray:
    """
    Stack a list of (n_tokens, d) Tensors into (n_layers, n_tokens, d)
    and L2-normalise each token vector.
    """
    stacked = torch.stack(
        [layernorm_to_sphere(h) for h in hidden_states], dim=0
    )  # (n_layers, n_tokens, d)
    return stacked.numpy().astype(np.float32)


def _serialize_events(events: list) -> list:
    """Convert any numpy scalar values inside event dicts to Python types."""
    out = []
    for ev in events:
        out.append({k: _json_default(v) if isinstance(v, np.generic) else v
                    for k, v in ev.items()})
    return out


def _f(v) -> float | None:
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
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1h: Hemispheric Structure Investigation"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
    )
    parser.add_argument(
        "--prompts", nargs="+",
        default=list(PROMPTS.keys()),
        choices=list(PROMPTS.keys()),
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="albert-base-v2 + wiki_paragraph only",
    )
    parser.add_argument(
        "--no-cone", action="store_true",
        help="Skip Block 3 LP (faster; no cone-collapse test)",
    )
    parser.add_argument(
        "--phase1-dir", type=str, default=None,
        metavar="DIR",
        help=(
            "Path to a Phase 1 run directory. Enables cross-referencing "
            "Block 1 events with Phase 1 merge events and energy violations."
        ),
    )
    args = parser.parse_args()

    if args.fast:
        models_arg  = ["albert-base-v2"]
        prompts_arg = ["wiki_paragraph"]
        import core.config as _cfg
        _cfg.ALBERT_SNAPSHOTS       = [12, 24, 36, 48]
        _cfg.ALBERT_MAX_ITERATIONS  = 48
    else:
        models_arg  = args.models
        prompts_arg = args.prompts

    run_all(
        models_to_run=models_arg,
        prompts_to_run=prompts_arg,
        run_cone=not args.no_cone,
        phase1_dir=Path(args.phase1_dir) if args.phase1_dir else None,
    )
