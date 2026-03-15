"""
run.py — Experiment orchestrator and CLI entry point.

Usage examples:
    python run.py                                          # all 7 models, all prompts
    python run.py --fast                                   # albert-base-v2 only, 2 prompts
    python run.py --models albert-base-v2 gpt2-xl         # specific model subset
    python run.py --models albert-xlarge-v2               # swap to ALBERT xlarge
    python run.py --models albert-base-v2 --prompts wiki_paragraph
    python run.py --no-extended
    python run.py --replot  metastability_results/2024-01-01_12-00-00/albert-base-v2_wiki_paragraph
    python run.py --summary metastability_results/2024-01-01_12-00-00/albert-base-v2_wiki_paragraph
"""

import sys
import traceback
import torch
from datetime import datetime
from pathlib import Path

from config import (
    BASE_RESULTS_DIR, MODEL_CONFIGS, PROMPTS,
    ALBERT_MAX_ITERATIONS, ALBERT_SNAPSHOTS,
)
from models import load_model, extract_activations, extract_albert_extended
from analysis import analyze_trajectory
from plots import (
    plot_trajectory,
    plot_ip_histograms,
    plot_pca_panels,
    plot_sinkhorn_detail,
    plot_spectral_eigengap,
    plot_eigenvalue_spectra,
    plot_albert_extended,
    plot_cross_model_comparison,
    analyze_value_eigenspectrum,
    plot_cka_trajectory,
)
from reporting import print_summary, generate_llm_report, generate_cross_run_report
from io_utils import save_run, replot_all
from clustering import HAS_UMAP

# Module-level output directory set by run_all before any analyze_trajectory call.
OUTPUT_DIR: Path = BASE_RESULTS_DIR


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_all(
    models_to_run: list = None,
    prompts_to_run: list = None,
    run_extended: bool = True,
) -> list:
    """
    Run the full Phase 1 analysis pipeline.

    Parameters
    ----------
    models_to_run  : model name keys from MODEL_CONFIGS (default: all)
    prompts_to_run : prompt keys from PROMPTS (default: all)
    run_extended   : if True, use ALBERT extended-iteration mode for ALBERT models

    Returns
    -------
    list of results dicts, one per (model, prompt) combination
    """
    global OUTPUT_DIR

    if models_to_run is None:
        models_to_run = list(MODEL_CONFIGS.keys())
    if prompts_to_run is None:
        prompts_to_run = list(PROMPTS.keys())

    # Timestamped experiment directory
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR = BASE_RESULTS_DIR / timestamp
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # UMAP directory (created once if umap-learn is available)
    umap_dir = None
    if HAS_UMAP:
        umap_dir = OUTPUT_DIR / "umap"
        umap_dir.mkdir(exist_ok=True)

    # Experiment manifest
    _write_manifest(timestamp, models_to_run, prompts_to_run, run_extended)

    print(f"\nExperiment directory: {OUTPUT_DIR}")
    all_results = []

    for model_name in models_to_run:
        print(f"\nLoading {model_name}...")
        try:
            model, tokenizer = load_model(model_name)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        analyze_value_eigenspectrum(model, model_name, OUTPUT_DIR)

        cfg          = MODEL_CONFIGS[model_name]
        use_extended = run_extended and cfg["is_albert"] and ALBERT_SNAPSHOTS

        if use_extended:
            all_results += _run_albert_extended(
                model, tokenizer, model_name, prompts_to_run, umap_dir
            )
        else:
            all_results += _run_standard(
                model, tokenizer, model_name, prompts_to_run, umap_dir
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_results) > 1:
        print("\nGenerating cross-model comparison plots...")
        plot_cross_model_comparison(all_results, OUTPUT_DIR)
        generate_cross_run_report(all_results, OUTPUT_DIR)

    print(f"\nDone. Results in: {OUTPUT_DIR.resolve()}")
    return all_results


# ---------------------------------------------------------------------------
# Per-model sub-routines
# ---------------------------------------------------------------------------

def _run_albert_extended(model, tokenizer, model_name, prompts_to_run, umap_dir):
    """
    Run ALBERT once per prompt to ALBERT_MAX_ITERATIONS, then fan out analysis
    across every snapshot depth.  Total iterations = MAX × n_prompts instead
    of sum(SNAPSHOTS) × n_prompts.

    Loop order: outer = prompts, inner = snapshots.
    Each (prompt, snapshot) pair gets its own results, plots, save, and report.
    """
    extended_trajectories_for_plot = {}
    results_list = []

    for prompt_key in prompts_to_run:
        print(f"\n  Prompt: {prompt_key}  "
              f"(single run to {ALBERT_MAX_ITERATIONS} iterations, "
              f"snapshots: {ALBERT_SNAPSHOTS})")
        try:
            snapshot_data = extract_albert_extended(
                model, tokenizer, PROMPTS[prompt_key],
                snapshots=ALBERT_SNAPSHOTS,
                max_iterations=ALBERT_MAX_ITERATIONS,
            )
        except Exception as e:
            print(f"    Failed: {e}")
            traceback.print_exc()
            continue

        for n_iter, data in snapshot_data.items():
            effective_model_name = f"{model_name}@{n_iter}iter"
            print(f"    Snapshot i{n_iter}  →  {effective_model_name}")

            hidden_states = data["trajectory"]
            attentions    = data["attentions"]
            tokens        = data["tokens"]

            if prompt_key == "wiki_paragraph":
                extended_trajectories_for_plot[n_iter] = hidden_states

            results = analyze_trajectory(
                hidden_states, attentions, prompt_key, effective_model_name,
                tokens, umap_dir=umap_dir,
            )
            results_list.append(results)
            print_summary(results)

            _generate_plots(results, OUTPUT_DIR)
            stem    = f"{effective_model_name.replace('/', '_').replace('@', '_')}_{prompt_key}"
            run_dir = OUTPUT_DIR / stem
            save_run(results, hidden_states, attentions, run_dir)
            generate_llm_report(results, run_dir)
            print(f"    Saved run to: {run_dir}/")

    if extended_trajectories_for_plot:
        plot_albert_extended(extended_trajectories_for_plot, OUTPUT_DIR)

    return results_list


def _run_standard(model, tokenizer, model_name, prompts_to_run, umap_dir):
    """
    Standard path: use model's native layer stack.
    Active when --no-extended is passed or for non-ALBERT models.
    """
    results_list = []

    for prompt_key in prompts_to_run:
        print(f"  Prompt: {prompt_key}")
        try:
            hidden_states, attentions, tokens = extract_activations(
                model, tokenizer, PROMPTS[prompt_key], model_name
            )
        except Exception as e:
            print(f"    Failed: {e}")
            continue

        results = analyze_trajectory(
            hidden_states, attentions, prompt_key, model_name,
            tokens, umap_dir=umap_dir,
        )
        results_list.append(results)
        print_summary(results)

        _generate_plots(results, OUTPUT_DIR)
        stem    = f"{model_name.replace('/', '_')}_{prompt_key}"
        run_dir = OUTPUT_DIR / stem
        save_run(results, hidden_states, attentions, run_dir)
        generate_llm_report(results, run_dir)
        print(f"  Saved run to: {run_dir}/")

    return results_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_plots(results: dict, out_dir: Path) -> None:
    """Call all per-run plot functions."""
    plot_trajectory(results, out_dir)
    plot_ip_histograms(results, out_dir)
    plot_pca_panels(results, out_dir)
    plot_sinkhorn_detail(results, out_dir)
    plot_spectral_eigengap(results, out_dir)
    plot_eigenvalue_spectra(results, out_dir)
    plot_cka_trajectory(results, out_dir)


def _write_manifest(timestamp, models_to_run, prompts_to_run, run_extended) -> None:
    from config import (
        DEVICE, BETA_VALUES, DISTANCE_THRESHOLDS,
        SINKHORN_MAX_ITER, SINKHORN_TOL, SPECTRAL_MAX_K,
        ALBERT_MAX_ITERATIONS, ALBERT_SNAPSHOTS, K_RANGE,
    )
    lines = [
        f"timestamp      : {timestamp}",
        f"command        : {' '.join(sys.argv)}",
        f"models         : {models_to_run}",
        f"prompts        : {prompts_to_run}",
        f"run_extended   : {run_extended}",
        f"albert_max_iterations : {ALBERT_MAX_ITERATIONS}",
        f"albert_snapshots      : {ALBERT_SNAPSHOTS}",
        f"device         : {DEVICE}",
        "",
        "--- parameters ---",
        f"beta_values         : {BETA_VALUES}",
        f"distance_thresholds : {list(DISTANCE_THRESHOLDS.round(3))}",
        f"sinkhorn_max_iter   : {SINKHORN_MAX_ITER}",
        f"sinkhorn_tol        : {SINKHORN_TOL}",
        f"spectral_max_k      : {SPECTRAL_MAX_K}",
        f"albert_max_iterations : {ALBERT_MAX_ITERATIONS}",
        f"albert_snapshots      : {ALBERT_SNAPSHOTS}",
        f"k_range             : {list(K_RANGE)}",
        "",
        "--- prompt texts ---",
    ]
    for key in prompts_to_run:
        lines.append(f"[{key}]")
        lines.append(PROMPTS.get(key, ""))
        lines.append("")

    with open(OUTPUT_DIR / "experiment.txt", "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1: Empirical Metastability Detection"
    )
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS.keys()),
                        choices=list(PROMPTS.keys()))
    parser.add_argument("--no-extended", action="store_true",
                        help="Disable ALBERT extended-iteration mode")
    parser.add_argument("--fast", action="store_true",
                        help="albert-base-v2 + wiki_paragraph")
    parser.add_argument("--replot", type=str, default=None, metavar="RUN_DIR",
                        help="Recreate all plots from a saved run directory")
    parser.add_argument("--summary", type=str, default=None, metavar="RUN_DIR",
                        help="Print text summary of a saved run")
    args = parser.parse_args()

    if args.replot:
        replot_all(Path(args.replot))

    elif args.summary:
        from reporting import print_run_summary
        print_run_summary(Path(args.summary))

    else:
        if args.fast:
            models  = ["albert-base-v2"]
            prompts = ["wiki_paragraph"]
        else:
            models  = args.models
            prompts = args.prompts

        run_all(
            models_to_run=models,
            prompts_to_run=prompts,
            run_extended=not args.no_extended,
        )
