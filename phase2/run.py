"""
run.py — Phase 2 experiment orchestrator and CLI entry point.

Usage examples:
    python -m phase2.run --full                        # all models, all prompts
    python -m phase2.run --full --models albert-base-v2
    python -m phase2.run --offline results/2026-03-25_19-32-55
    python -m phase2.run --offline results/2026-03-25_19-32-55 --models albert-base-v2
    python -m phase2.run --fast                        # albert-base-v2, wiki_paragraph

Modes:
    --full    : load model → weights.py + decompose.py + trajectory.py + analysis
    --offline : trajectory.py + analysis from saved Phase 1 data + pre-saved
                weight decomposition (no model loading)
    --replot  : regenerate plots from saved Phase 2 artifacts
"""

import sys
import traceback
import torch
from datetime import datetime
from pathlib import Path

from core.config import (
    BASE_RESULTS_DIR, MODEL_CONFIGS, PROMPTS,
    ALBERT_MAX_ITERATIONS, ALBERT_SNAPSHOTS,
)
from core.models import load_model

from phase2.weights import analyze_weights, load_weight_decomposition
from phase2.trajectory import analyze_trajectory_offline, load_phase1_events
from phase2.trajectory_perlayer import analyze_trajectory_offline_perlayer
from phase2.analysis import full_analysis
from phase2.layer_v_events import run_layer_v_analysis, print_layer_v_summary
from phase2.head_ov_analysis import run_head_analysis, print_head_analysis_summary
from phase2.decompose import (
    extract_decomposed_albert,
    extract_decomposed_standard,
    analyze_violations_decomposed,
    save_decomposed,
)
from phase2.reporting import print_phase2_summary, build_verdict, save_verdict
from phase2.ffn_subspace import run_ffn_subspace_analysis, print_ffn_subspace_summary
from phase2.analysis_extended import full_analysis_extended, print_extended_summary
from phase2.verdict_v2 import build_verdict_v2


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_full(
    models_to_run: list = None,
    prompts_to_run: list = None,
    phase1_dir: Path = None,
) -> list:
    """
    Full Phase 2 pipeline: load models, extract weights, decompose,
    run trajectory analysis, cross-reference.

    Parameters
    ----------
    models_to_run  : model name keys (default: all)
    prompts_to_run : prompt keys (default: all)
    phase1_dir     : path to a Phase 1 experiment directory containing
                     per-run subdirectories.  If None, uses the most
                     recent directory in BASE_RESULTS_DIR.
    """
    if models_to_run is None:
        models_to_run = list(MODEL_CONFIGS.keys())
    if prompts_to_run is None:
        prompts_to_run = list(PROMPTS.keys())

    phase1_dir = _resolve_phase1_dir(phase1_dir)
    if phase1_dir is None:
        print("No Phase 1 results found. Run Phase 1 first.")
        return []

    # Phase 2 output directory
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = BASE_RESULTS_DIR / f"phase2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPhase 2 output: {output_dir}")
    print(f"Phase 1 source: {phase1_dir}")

    all_verdicts = []

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # Load model and extract weights
        try:
            model, tokenizer = load_model(model_name)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        ov_data = analyze_weights(model, model_name, output_dir)

        cfg = MODEL_CONFIGS[model_name]

        for prompt_key in prompts_to_run:
            run_dir = _find_run_dir(phase1_dir, model_name, prompt_key, cfg)
            if run_dir is None:
                print(f"  No Phase 1 run found for {prompt_key}, skipping")
                continue

            print(f"\n  Prompt: {prompt_key}")
            print(f"  Phase 1 run: {run_dir}")

            try:
                # Trajectory analysis (per-layer-aware: uses each layer's
                # own V and projectors for GPT-2/BERT, standard path for ALBERT)
                traj = analyze_trajectory_offline_perlayer(run_dir, ov_data)

                # Cross-reference analysis
                analysis = full_analysis(traj, ov_data)

                # Per-layer V vs Phase 1 events (GPT-2/BERT only)
                if ov_data["is_per_layer"]:
                    lv_result = run_layer_v_analysis(ov_data, run_dir, beta=1.0)
                    analysis["layer_v_events"] = lv_result
                    print_layer_v_summary(lv_result, model_name, prompt_key)

                # Per-head OV × Fiedler cross-reference
                if "ov_per_head" in ov_data:
                    head_result = run_head_analysis(ov_data, run_dir)
                    analysis["head_ov"] = head_result
                    print_head_analysis_summary(head_result, model_name, prompt_key)

                # Decomposed forward pass (requires model)
                decomposed = _run_decompose(
                    model, tokenizer, model_name, prompt_key, cfg
                )
                if decomposed is not None:
                    events = traj["events"]
                    decomp_results = analyze_violations_decomposed(
                        decomposed, events, beta=1.0
                    )
                    analysis["decomposed_violations"] = decomp_results

                    # Save for Phase 3
                    stem = _run_stem(model_name, prompt_key, cfg)
                    save_decomposed(decomposed, output_dir / stem)

                # FFN subspace projection (offline — uses saved ffn_deltas)
                # Try the phase2 output dir first (where decompose just saved),
                # then fall back to the phase1 run dir
                ffn_dir = output_dir / _run_stem(model_name, prompt_key, cfg)
                ffn_result = run_ffn_subspace_analysis(ffn_dir, ov_data, phase1_run_dir=run_dir)
                if not ffn_result.get("applicable"):
                    # Try phase1 run dir in case ffn_deltas were saved there
                    ffn_result = run_ffn_subspace_analysis(run_dir, ov_data)
                if ffn_result.get("applicable"):
                    analysis["ffn_subspace"] = ffn_result
                    print_ffn_subspace_summary(ffn_result, model_name, prompt_key)

                # Extended analysis (continuous ΔE, OV norm confound, adaptive zones,
                # attractive-zone violation analysis)
                extended = full_analysis_extended(
                    ov_data, run_dir, beta=1.0,
                    ffn_subspace_result=analysis.get("ffn_subspace"),
                    layer_v_result=analysis.get("layer_v_events"),
                )
                analysis["extended"] = extended
                print_extended_summary(extended, model_name, prompt_key)

                # Report
                print_phase2_summary(analysis, ov_data, model_name, prompt_key)
                verdict = build_verdict_v2(analysis, ov_data, model_name, prompt_key)
                all_verdicts.append(verdict)

                stem = _run_stem(model_name, prompt_key, cfg)
                save_verdict(verdict, output_dir / stem)
                _save_cross_phase_artifacts(analysis, ov_data, run_dir, output_dir / stem)

            except Exception as e:
                print(f"  Failed: {e}")
                traceback.print_exc()
                continue

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save cross-run summary
    if all_verdicts:
        _save_cross_run(all_verdicts, output_dir)

    print(f"\nPhase 2 complete. Results in: {output_dir.resolve()}")
    return all_verdicts


def run_offline(
    phase1_dir: Path,
    models_to_run: list = None,
    prompts_to_run: list = None,
    weights_dir: Path = None,
) -> list:
    """
    Offline Phase 2: trajectory + analysis from saved data only.
    No model loading.  Requires pre-saved weight decomposition.

    Parameters
    ----------
    phase1_dir   : Phase 1 experiment directory
    weights_dir  : directory containing ov_weights_*.npz etc.
                   If None, looks in Phase 1 dir.
    """
    if models_to_run is None:
        models_to_run = list(MODEL_CONFIGS.keys())
    if prompts_to_run is None:
        prompts_to_run = list(PROMPTS.keys())

    phase1_dir = Path(phase1_dir)
    if weights_dir is None:
        weights_dir = phase1_dir

    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = BASE_RESULTS_DIR / f"phase2_offline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPhase 2 offline output: {output_dir}")

    all_verdicts = []

    for model_name in models_to_run:
        # Try to load pre-saved weight decomposition
        try:
            ov_loaded = load_weight_decomposition(weights_dir, model_name)
        except FileNotFoundError:
            print(f"  No weight decomposition found for {model_name} "
                  f"in {weights_dir}. Run --full first.")
            continue

        # Build a minimal ov_data dict from loaded data
        ov_data = _ov_data_from_loaded(ov_loaded, model_name)

        cfg = MODEL_CONFIGS[model_name]

        for prompt_key in prompts_to_run:
            run_dir = _find_run_dir(phase1_dir, model_name, prompt_key, cfg)
            if run_dir is None:
                continue

            print(f"\n  {model_name} | {prompt_key}")
            try:
                traj = analyze_trajectory_offline_perlayer(run_dir, ov_data)
                analysis = full_analysis(traj, ov_data)

                # Per-layer V vs Phase 1 events (GPT-2/BERT only)
                if ov_data["is_per_layer"]:
                    lv_result = run_layer_v_analysis(ov_data, run_dir, beta=1.0)
                    analysis["layer_v_events"] = lv_result
                    print_layer_v_summary(lv_result, model_name, prompt_key)

                # FFN subspace projection (needs ffn_deltas.npz from prior --full)
                ffn_result = run_ffn_subspace_analysis(run_dir, ov_data)
                if ffn_result.get("applicable"):
                    analysis["ffn_subspace"] = ffn_result
                    print_ffn_subspace_summary(ffn_result, model_name, prompt_key)

                # Extended analysis (continuous ΔE, OV norm confound, adaptive zones,
                # attractive-zone violation analysis)
                extended = full_analysis_extended(
                    ov_data, run_dir, beta=1.0,
                    ffn_subspace_result=analysis.get("ffn_subspace"),
                    layer_v_result=analysis.get("layer_v_events"),
                )
                analysis["extended"] = extended
                print_extended_summary(extended, model_name, prompt_key)

                print_phase2_summary(analysis, ov_data, model_name, prompt_key)
                verdict = build_verdict_v2(analysis, ov_data, model_name, prompt_key)
                all_verdicts.append(verdict)

                stem = _run_stem(model_name, prompt_key, cfg)
                save_verdict(verdict, output_dir / stem)
                _save_cross_phase_artifacts(analysis, ov_data, run_dir, output_dir / stem)

            except Exception as e:
                print(f"  Failed: {e}")
                traceback.print_exc()

    if all_verdicts:
        _save_cross_run(all_verdicts, output_dir)

    print(f"\nDone. Results in: {output_dir.resolve()}")
    return all_verdicts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_phase1_dir(phase1_dir):
    """Find Phase 1 experiment directory."""
    if phase1_dir is not None:
        p = Path(phase1_dir)
        if p.exists():
            return p
        # Try as a subdirectory of BASE_RESULTS_DIR
        p2 = BASE_RESULTS_DIR / p
        if p2.exists():
            return p2
        # Try just the name component under BASE_RESULTS_DIR
        p3 = BASE_RESULTS_DIR / p.name
        if p3.exists():
            return p3
        # Return as-is and let downstream fail with a clear path
        return p
    # Find most recent timestamped directory
    candidates = sorted(
        [d for d in BASE_RESULTS_DIR.iterdir()
         if d.is_dir() and not d.name.startswith("phase2")],
        reverse=True,
    )
    return candidates[0] if candidates else None


def _find_run_dir(phase1_dir, model_name, prompt_key, cfg):
    """Locate a Phase 1 run directory for a given model/prompt pair."""
    phase1_dir = Path(phase1_dir)

    # ALBERT extended runs use @Niter naming
    if cfg["is_albert"]:
        for snap in reversed(ALBERT_SNAPSHOTS):
            effective = f"{model_name}@{snap}iter"
            stem = f"{effective.replace('/', '_').replace('@', '_')}_{prompt_key}"
            d = phase1_dir / stem
            if d.exists() and (d / "metrics.json").exists():
                return d

    # Standard naming
    stem = f"{model_name.replace('/', '_')}_{prompt_key}"
    d = phase1_dir / stem
    if d.exists() and (d / "metrics.json").exists():
        return d

    return None


def _run_stem(model_name, prompt_key, cfg):
    """Generate run directory stem name."""
    name = model_name.replace("/", "_")
    return f"{name}_{prompt_key}"


def _run_decompose(model, tokenizer, model_name, prompt_key, cfg):
    """Run decomposed extraction for one model/prompt."""
    text = PROMPTS.get(prompt_key, "")
    if not text:
        return None

    try:
        if cfg["is_albert"]:
            snapshot_data = extract_decomposed_albert(
                model, tokenizer, text,
                snapshots=[ALBERT_SNAPSHOTS[-1]],  # largest snapshot only
                max_iterations=ALBERT_MAX_ITERATIONS,
            )
            return snapshot_data[ALBERT_SNAPSHOTS[-1]]
        else:
            return extract_decomposed_standard(
                model, tokenizer, text, model_name
            )
    except Exception as e:
        print(f"    Decompose failed: {e}")
        traceback.print_exc()
        return None


def _ov_data_from_loaded(loaded, model_name):
    """Reconstruct ov_data dict from loaded weight decomposition."""
    summary = loaded["summary"]
    is_per_layer = summary["is_per_layer"]

    if is_per_layer:
        # Reconstruct decomps list for analysis functions
        decomps = []
        qk_norms = []
        for layer_name, layer_summary in summary["layers"].items():
            decomps.append({
                "frac_attractive": layer_summary["frac_attractive"],
                "frac_repulsive":  layer_summary["frac_repulsive"],
                "frac_complex":    layer_summary.get("frac_complex", 0),
                "agree":           layer_summary["methods_agree"],
                "schur_cond":      layer_summary.get("schur_cond", 0),
            })
            if "qk_spectral_norms_per_head" in layer_summary:
                qk_norms.append(layer_summary["qk_spectral_norms_per_head"])
        return {
            "ov_total":     loaded["ov_total"],
            "projectors":   loaded["projectors"],
            "decomps":      decomps,
            "qk_data":      {"qk_spectral_norms": qk_norms,
                             "layer_names": list(summary["layers"].keys())},
            "is_per_layer": True,
            "layer_names":  list(summary["layers"].keys()),
            "d_model":      summary["d_model"],
            "n_heads":      summary["n_heads"],
            "d_head":       summary["d_head"],
        }
    else:
        layer_summary = list(summary["layers"].values())[0]
        return {
            "ov_total":     loaded["ov_total"],
            "projectors":   loaded["projectors"],
            "decomps":      {
                "frac_attractive": layer_summary["frac_attractive"],
                "frac_repulsive":  layer_summary["frac_repulsive"],
                "frac_complex":    layer_summary.get("frac_complex", 0),
                "agree":           layer_summary["methods_agree"],
                "schur_cond":      layer_summary.get("schur_cond", 0),
            },
            "is_per_layer": False,
            "layer_names":  ["shared"],
            "d_model":      summary["d_model"],
            "n_heads":      summary["n_heads"],
            "d_head":       summary["d_head"],
        }


def _save_cross_run(verdicts, output_dir):
    """Save cross-run summary."""
    import json
    path = output_dir / "phase2_cross_run.json"
    with open(path, "w") as f:
        json.dump(verdicts, f, indent=2)
    print(f"\nCross-run summary saved to {path}")


def _save_cross_phase_artifacts(
    analysis: dict,
    ov_data: dict,
    run_dir: Path,
    save_dir: Path,
) -> None:
    """
    Save standalone JSON artifacts that Phase 3 cross-phase analyses consume.

    Written files (inside save_dir):
      ffn_subspace.json    — per-violation FFN role classification
      cross_term.json      — cross-term dominant pairs and mechanism
      ov_per_head.json     — per-head eigenspectra for head×Fiedler crossref
      head_ov.json         — head OV analysis results
    """
    import json

    # --- ffn_subspace.json ---
    ffn = analysis.get("ffn_subspace")
    if ffn and ffn.get("applicable"):
        with open(save_dir / "ffn_subspace.json", "w") as f:
            json.dump(_jsonify(ffn), f, indent=2)

    # --- cross_term.json ---
    # cross_term results are embedded in decomposed_violations or extended
    ct_data = analysis.get("extended", {}).get("cross_term")
    if ct_data is None:
        # Try to extract from decomposed violations
        decomp = analysis.get("decomposed_violations", [])
        ct_violations = []
        for d in decomp:
            if d.get("cross_frac", 0) > 0.1:
                ct_violations.append({
                    "layer": d.get("layer"),
                    "cross_dominant": d.get("cross_frac", 0) > 0.5,
                    "cross_frac": d.get("cross_frac", 0),
                    "delta_cross": d.get("delta_cross", 0),
                    "delta_total": d.get("delta_total", 0),
                })
        if ct_violations:
            ct_data = {"per_violation": ct_violations}

    # Also try loading cross_term_analysis results if run separately
    ct_json_path = run_dir / "cross_term_results.json"
    if ct_json_path.exists():
        with open(ct_json_path) as f:
            ct_data = json.load(f)

    if ct_data:
        with open(save_dir / "cross_term.json", "w") as f:
            json.dump(_jsonify(ct_data), f, indent=2)

    # --- ov_per_head.json ---
    # Per-head eigenvalues for Phase 3 head×Fiedler crossref
    per_head = ov_data.get("ov_per_head")
    if per_head is not None:
        # per_head is typically a list of (d,) arrays of eigenvalues per head
        serializable = []
        for h_eigs in per_head:
            if hasattr(h_eigs, "tolist"):
                serializable.append({
                    "eigenvalues_real": np.real(h_eigs).tolist(),
                    "eigenvalues_imag": np.imag(h_eigs).tolist(),
                })
            else:
                serializable.append(h_eigs)
        with open(save_dir / "ov_per_head.json", "w") as f:
            json.dump(serializable, f, indent=2)

    # --- head_ov.json ---
    head_ov = analysis.get("head_ov")
    if head_ov and head_ov.get("applicable"):
        with open(save_dir / "head_ov.json", "w") as f:
            json.dump(_jsonify(head_ov), f, indent=2)


def _jsonify(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 2: Energy Violation Mechanism Identification"
    )
    parser.add_argument("--full", action="store_true",
                        help="Full pipeline with model loading")
    parser.add_argument("--offline", type=str, default=None, metavar="PHASE1_DIR",
                        help="Offline analysis from saved Phase 1 data")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--prompts", nargs="+", default=None,
                        choices=list(PROMPTS.keys()))
    parser.add_argument("--phase1-dir", type=str, default=None,
                        help="Phase 1 experiment directory (for --full)")
    parser.add_argument("--fast", action="store_true",
                        help="albert-base-v2 + wiki_paragraph only")
    args = parser.parse_args()

    if args.fast:
        models  = ["albert-base-v2"]
        prompts = ["wiki_paragraph"]
    else:
        models  = args.models
        prompts = args.prompts

    if args.offline:
        run_offline(
            phase1_dir=Path(args.offline),
            models_to_run=models,
            prompts_to_run=prompts,
        )
    elif args.full or args.fast:
        run_full(
            models_to_run=models,
            prompts_to_run=prompts,
            phase1_dir=Path(args.phase1_dir) if args.phase1_dir else None,
        )
    else:
        parser.print_help()
        print("\nSpecify --full or --offline <PHASE1_DIR>")
