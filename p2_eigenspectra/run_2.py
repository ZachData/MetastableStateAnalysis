"""
run_2.py — p2_eigenspectra experiment orchestrator and CLI entry point.

Usage examples:
    python -m p2_eigenspectra.run_2 --full
    python -m phase2.run_2 --full --models albert-base-v2
    python -m phase2.run_2 --full --phase1-dir results/2026-04-11_13-08-09
    python -m phase2.run_2 --offline results/p2_eigenspectra_full --phase1-dir results/phase1
    python -m phase2.run_2 --fast

Modes:
    --full    : load model → weights.py + decompose.py + trajectory.py + analysis
    --offline : trajectory.py + analysis from saved Phase 1 data + pre-saved
                weight decomposition and decomposed deltas (no model loading).
    --replot  : regenerate plots from saved artifacts

Output directory naming: p2_eigenspectra_<timestamp>/
Per-run sub-directory layout:
    p2_eigenspectra_<ts>/
      ov_weights_<model>.npz
      ov_decomp_<model>.npz
      ov_projectors_<model>.npz
      ov_summary_<model>.json
      p2_eigenspectra_cross_run.json
      p2_eigenspectra_cross_run.summary.txt
      <model>_<prompt>/
        attn_deltas_raw.npz
        ffn_deltas_raw.npz
        verdict.json
        summary.txt                  <- LLM-friendly aggregate
        sub/                         <- one file per sub-experiment
          trajectory.json
          trajectory.summary.txt
          layer_v_events.json
          layer_v_events.summary.txt
          head_ov.json
          head_ov.summary.txt
          decomposed_violations.json
          decomposed_violations.summary.txt
          ffn_subspace.json
          ffn_subspace.summary.txt
          continuous_correlations.json
          continuous_correlations.summary.txt
          ov_norm_confound.json
          ov_norm_confound.summary.txt
          zone_comparison.json
          zone_comparison.summary.txt
          attractive_zone_violations.json
          attractive_zone_violations.summary.txt
"""

import sys
import gc
import traceback
import torch
from datetime import datetime
from pathlib import Path
import numpy as np

from core.config import (
    BASE_RESULTS_DIR, MODEL_CONFIGS, PROMPTS,
    ALBERT_MAX_ITERATIONS, ALBERT_SNAPSHOTS,
)
from core.models import load_model

from p2_eigenspectra.weights import analyze_weights, load_weight_decomposition
from p2_eigenspectra.trajectory import analyze_trajectory_offline, load_phase1_events
from p2_eigenspectra.trajectory_perlayer import analyze_trajectory_offline_perlayer
from p2_eigenspectra.analysis import full_analysis
from p2_eigenspectra.decompose import (
    extract_decomposed_albert,
    extract_decomposed_standard,
    save_decomposed,
)
from p2_eigenspectra.reporting import save_verdict
from p2_eigenspectra.subexperiments import run_one_prompt


# ---------------------------------------------------------------------------
# Decomposed data loader (unchanged from original)
# ---------------------------------------------------------------------------

def load_decomposed(run_dir: Path) -> dict:
    """Load saved attn/FFN decomposition deltas from a prior --full run."""
    run_dir = Path(run_dir)
    attn_raw_path = run_dir / "attn_deltas_raw.npz"
    ffn_raw_path  = run_dir / "ffn_deltas_raw.npz"
    if not attn_raw_path.exists() or not ffn_raw_path.exists():
        return None

    attn_raw = np.load(attn_raw_path)["attn_deltas"]
    ffn_raw  = np.load(ffn_raw_path)["ffn_deltas"]
    attn_deltas = [attn_raw[i] for i in range(attn_raw.shape[0])]
    ffn_deltas  = [ffn_raw[i]  for i in range(ffn_raw.shape[0])]

    traj_path = run_dir / "hidden_states.npz"
    if traj_path.exists():
        hs  = np.load(traj_path)
        key = list(hs.keys())[0] if len(hs.keys()) == 1 else "hidden_states"
        all_hidden = hs[key] if key in hs else None
        trajectory = (
            [all_hidden[i] for i in range(all_hidden.shape[0])]
            if all_hidden is not None
            else _reconstruct_trajectory_from_deltas(attn_deltas, ffn_deltas)
        )
    else:
        trajectory = _reconstruct_trajectory_from_deltas(attn_deltas, ffn_deltas)

    return {"trajectory": trajectory, "attn_deltas": attn_deltas, "ffn_deltas": ffn_deltas}


def _reconstruct_trajectory_from_deltas(attn_deltas, ffn_deltas):
    n_tokens, d = attn_deltas[0].shape
    trajectory = [np.zeros((n_tokens, d), dtype=np.float32)]
    h = trajectory[0].copy()
    for a, f in zip(attn_deltas, ffn_deltas):
        h = h + a + f
        trajectory.append(h.copy())
    return trajectory


# ---------------------------------------------------------------------------
# run_full
# ---------------------------------------------------------------------------

def run_full(
    models_to_run: list = None,
    prompts_to_run: list = None,
    phase1_dir: Path = None,
) -> list:
    """Full p2_eigenspectra pipeline: load models, extract weights, decompose, analyse."""
    if models_to_run is None:
        models_to_run = list(MODEL_CONFIGS.keys())
    if prompts_to_run is None:
        prompts_to_run = list(PROMPTS.keys())

    phase1_dir = _resolve_phase1_dir(phase1_dir)
    if phase1_dir is None:
        print("No Phase 1 results found. Run Phase 1 first.")
        return []

    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = BASE_RESULTS_DIR / f"p2_eigenspectra_{timestamp}"  # <-- renamed
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\np2_eigenspectra output: {output_dir}")
    print(f"Phase 1 source:        {phase1_dir}")

    all_verdicts = []

    for model_name in models_to_run:
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
        try:
            model, tokenizer = load_model(model_name)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        ov_data = analyze_weights(model, model_name, output_dir)
        cfg     = MODEL_CONFIGS[model_name]

        for prompt_key in prompts_to_run:
            run_dir = _find_run_dir(phase1_dir, model_name, prompt_key, cfg)
            if run_dir is None:
                print(f"  No Phase 1 run found for {prompt_key}, skipping")
                continue

            print(f"\n  Prompt: {prompt_key}")
            print(f"  Phase 1 run: {run_dir}")

            try:
                traj = analyze_trajectory_offline_perlayer(run_dir, ov_data)

                # Decomposed forward pass (requires loaded model)
                decomposed = _run_decompose(model, tokenizer, model_name, prompt_key, cfg)
                stem       = _run_stem(model_name, prompt_key, cfg)
                if decomposed is not None:
                    save_decomposed(decomposed, output_dir / stem)

                ctx = {
                    "model_name":     model_name,
                    "prompt_key":     prompt_key,
                    "stem":           stem,
                    "ov_data":        ov_data,
                    "traj":           traj,
                    "decomposed":     decomposed,
                    "phase1_run_dir": run_dir,
                }

                verdict = run_one_prompt(ctx, output_dir)
                all_verdicts.append(verdict)

            except Exception as e:
                print(f"  Failed: {e}")
                traceback.print_exc()
                continue

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if all_verdicts:
        _save_cross_run(all_verdicts, output_dir)

    print(f"\np2_eigenspectra complete. Results in: {output_dir.resolve()}")
    return all_verdicts


# ---------------------------------------------------------------------------
# run_offline
# ---------------------------------------------------------------------------

def run_offline(
    phase1_dir: Path,
    models_to_run: list = None,
    prompts_to_run: list = None,
    weights_dir: Path = None,
    head_analysis: bool = False,
) -> list:
    """Offline p2_eigenspectra: trajectory + analysis from saved data only."""
    if models_to_run is None:
        models_to_run = list(MODEL_CONFIGS.keys())
    if prompts_to_run is None:
        prompts_to_run = list(PROMPTS.keys())

    phase1_dir = Path(phase1_dir)
    if weights_dir is None:
        weights_dir = phase1_dir

    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = BASE_RESULTS_DIR / f"p2_eigenspectra_offline_{timestamp}"  # <-- renamed
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\np2_eigenspectra offline output: {output_dir}")
    print(f"Phase 1 source:                {phase1_dir}")
    print(f"Weights source:                {weights_dir}")

    all_verdicts = []

    for model_name in models_to_run:
        try:
            ov_loaded = load_weight_decomposition(weights_dir, model_name)
        except FileNotFoundError:
            print(f"  No weight decomposition found for {model_name} in {weights_dir}.")
            continue

        ov_data = _ov_data_from_loaded(
            ov_loaded, model_name, weights_dir, load_per_head=head_analysis
        )
        del ov_loaded
        gc.collect()

        cfg = MODEL_CONFIGS[model_name]

        for prompt_key in prompts_to_run:
            run_dir = _find_run_dir(phase1_dir, model_name, prompt_key, cfg)
            if run_dir is None:
                continue

            print(f"\n  {model_name} | {prompt_key}")
            try:
                traj = analyze_trajectory_offline_perlayer(run_dir, ov_data)

                # Load saved decomposed deltas
                stem         = _run_stem(model_name, prompt_key, cfg)
                decompose_dir = Path(weights_dir) / stem
                decomposed   = load_decomposed(decompose_dir)
                if decomposed is None:
                    decomposed = load_decomposed(run_dir)
                if decomposed is None:
                    print(f"    Decompose: no saved deltas in {decompose_dir} or {run_dir}")

                ctx = {
                    "model_name":     model_name,
                    "prompt_key":     prompt_key,
                    "stem":           stem,
                    "ov_data":        ov_data,
                    "traj":           traj,
                    "decomposed":     decomposed,
                    "phase1_run_dir": run_dir,
                }

                verdict = run_one_prompt(ctx, output_dir)
                all_verdicts.append(verdict)

            except Exception as e:
                print(f"  Failed: {e}")
                traceback.print_exc()

            gc.collect()

        del ov_data
        gc.collect()

    if all_verdicts:
        _save_cross_run(all_verdicts, output_dir)

    print(f"\nDone. Results in: {output_dir.resolve()}")
    return all_verdicts


# ---------------------------------------------------------------------------
# Cross-run summary
# ---------------------------------------------------------------------------

def _save_cross_run(verdicts: list, output_dir: Path) -> None:
    """Save cross-run JSON + LLM-friendly summary."""
    import json
    from p2_eigenspectra.subexperiments import _jsonify

    json_path = output_dir / "p2_eigenspectra_cross_run.json"  # <-- renamed
    with open(json_path, "w") as f:
        json.dump([_jsonify(v) for v in verdicts], f, indent=2)
    print(f"\nCross-run JSON saved to {json_path}")

    _write_cross_run_summary(verdicts, output_dir)


def _write_cross_run_summary(verdicts: list, output_dir: Path) -> None:
    """
    Write p2_eigenspectra_cross_run.summary.txt — comparative table across all runs.
    """
    lines = []
    sep = "=" * 72
    lines += [sep, "P2_EIGENSPECTRA CROSS-RUN SUMMARY", sep]
    lines.append(f"Total runs: {len(verdicts)}")
    lines.append("")

    # Comparative table header
    col_keys = [
        "model", "prompt", "falsification", "channel", "v_score",
        "beta1.0_n_violations", "beta1.0_frac_repulsive",
        "frac_ffn_amplifies_repulsive", "ov_norm_partial_rho",
    ]
    header = "  ".join(f"{k[:22]:<22}" for k in col_keys)
    lines += [sep, "KEY NUMBERS ACROSS ALL RUNS", sep, header, "-" * len(header)]

    for v in verdicts:
        row_parts = []
        for k in col_keys:
            val = v.get(k)
            if val is None:
                row_parts.append(f"{'n/a':<22}")
            elif isinstance(val, float):
                row_parts.append(f"{val:<22.3f}")
            else:
                row_parts.append(f"{str(val):<22}")
        lines.append("  ".join(row_parts))

    # Per-run summary references
    lines += ["", sep, "PER-RUN SUMMARY FILES", sep]
    for v in verdicts:
        model  = v.get("model", "?")
        prompt = v.get("prompt", "?")
        stem   = f"{model.replace('/', '_')}_{prompt}"
        lines.append(f"  {stem}/summary.txt  → {v.get('falsification', '?')}")

    txt_path = output_dir / "p2_eigenspectra_cross_run.summary.txt"  # <-- renamed
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Cross-run summary written to {txt_path}")


# ---------------------------------------------------------------------------
# Helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _resolve_phase1_dir(phase1_dir):
    if phase1_dir is not None:
        p = Path(phase1_dir)
        if p.exists():
            return p
        p2 = BASE_RESULTS_DIR / p
        if p2.exists():
            return p2
        p3 = BASE_RESULTS_DIR / p.name
        if p3.exists():
            return p3
        return p
    candidates = sorted(
        [d for d in BASE_RESULTS_DIR.iterdir()
         if d.is_dir() and not d.name.startswith("p2_eigenspectra")],  # <-- renamed
        reverse=True,
    )
    return candidates[0] if candidates else None


def _find_run_dir(phase1_dir, model_name, prompt_key, cfg):
    phase1_dir = Path(phase1_dir)
    if cfg["is_albert"]:
        for snap in reversed(ALBERT_SNAPSHOTS):
            effective = f"{model_name}@{snap}iter"
            stem = f"{effective.replace('/', '_').replace('@', '_')}_{prompt_key}"
            d = phase1_dir / stem
            if d.exists() and (d / "metrics.json").exists():
                return d
    stem = f"{model_name.replace('/', '_')}_{prompt_key}"
    d    = phase1_dir / stem
    if d.exists() and (d / "metrics.json").exists():
        return d
    return None


def _run_stem(model_name, prompt_key, cfg):
    return f"{model_name.replace('/', '_')}_{prompt_key}"


def _run_decompose(model, tokenizer, model_name, prompt_key, cfg):
    text = PROMPTS.get(prompt_key, "")
    if not text:
        return None
    try:
        if cfg["is_albert"]:
            snapshot_data = extract_decomposed_albert(
                model, tokenizer, text,
                snapshots=[ALBERT_SNAPSHOTS[-1]],
                max_iterations=ALBERT_MAX_ITERATIONS,
            )
            return snapshot_data[ALBERT_SNAPSHOTS[-1]]
        else:
            return extract_decomposed_standard(model, tokenizer, text, model_name)
    except Exception as e:
        print(f"    Decompose failed: {e}")
        traceback.print_exc()
        return None


def _ov_data_from_loaded(loaded, model_name, weights_dir=None, load_per_head=False):
    summary      = loaded["summary"]
    is_per_layer = summary["is_per_layer"]

    ov_per_head = None
    if load_per_head and weights_dir is not None:
        stem = model_name.replace("/", "_")
        weights_npz_path = Path(weights_dir) / f"ov_weights_{stem}.npz"
        if weights_npz_path.exists():
            ov_npz      = np.load(weights_npz_path)
            ov_per_head = _extract_ov_per_head(ov_npz, summary, is_per_layer)

    if is_per_layer:
        decomps  = []
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
        result = {
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
        result = {
            "ov_total":     loaded["ov_total"],
            "projectors":   loaded["projectors"],
            "decomps": {
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

    if ov_per_head is not None:
        result["ov_per_head"] = ov_per_head
    return result


def _extract_ov_per_head(ov_npz, summary, is_per_layer):
    n_heads = summary["n_heads"]
    if is_per_layer:
        layer_names = list(summary["layers"].keys())
        per_layer_heads = []
        for name in layer_names:
            heads = []
            for h in range(n_heads):
                key = f"ov_head{h}_{name}"
                if key in ov_npz:
                    heads.append(ov_npz[key])
                else:
                    return None
            per_layer_heads.append(heads)
        return per_layer_heads
    else:
        heads = []
        for h in range(n_heads):
            key = f"ov_head{h}_shared"
            if key in ov_npz:
                heads.append(ov_npz[key])
            else:
                return None
        return heads


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="p2_eigenspectra: Energy Violation Mechanism Identification"
    )
    parser.add_argument("--full",    action="store_true")
    parser.add_argument("--offline", type=str, default=None, metavar="P2_FULL_DIR")
    parser.add_argument("--models",  nargs="+", default=None,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--prompts", nargs="+", default=None,
                        choices=list(PROMPTS.keys()))
    parser.add_argument("--phase1-dir",  type=str, default=None)
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--fast",   action="store_true",
                        help="albert-base-v2 + wiki_paragraph only")
    parser.add_argument("--head-analysis", action="store_true",
                        help="Load per-head OV matrices (~25 GB for gpt2-xl)")
    args = parser.parse_args()

    if args.fast:
        models  = ["albert-base-v2"]
        prompts = ["wiki_paragraph"]
    else:
        models  = args.models
        prompts = args.prompts

    if args.offline:
        offline_dir = Path(args.offline)
        if args.phase1_dir:
            phase1  = Path(args.phase1_dir)
            weights = Path(args.weights_dir) if args.weights_dir else offline_dir
        else:
            phase1  = offline_dir
            weights = Path(args.weights_dir) if args.weights_dir else None
        run_offline(
            phase1_dir=phase1, models_to_run=models, prompts_to_run=prompts,
            weights_dir=weights, head_analysis=args.head_analysis,
        )
    elif args.full or args.fast:
        run_full(
            models_to_run=models, prompts_to_run=prompts,
            phase1_dir=Path(args.phase1_dir) if args.phase1_dir else None,
        )
    else:
        parser.print_help()
        print("\nSpecify --full or --offline <P2_FULL_DIR>")
