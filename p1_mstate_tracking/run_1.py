"""
run.py — Experiment orchestrator and CLI entry point.

Usage examples:
python -m phase1.run
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

from core.config import (
    BASE_RESULTS_DIR, MODEL_CONFIGS, PROMPTS,
    ALBERT_MAX_ITERATIONS, ALBERT_SNAPSHOTS, LENGTH_SWEEP_TOKENS,
)
from core.models import load_model, extract_activations, extract_albert_extended
from .analysis import analyze_trajectory
from .plots import (
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
from .reporting import print_summary, generate_llm_report, generate_cross_run_report
from .io_utils import save_run, replot_all, aggregate_global_artifacts
from .clustering import HAS_UMAP

# Module-level output directory set by run_all before any analyze_trajectory call.
OUTPUT_DIR: Path = BASE_RESULTS_DIR


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_all(
    models_to_run: list = None,
    prompts_to_run: list = None,
    run_extended: bool = True,
    run_sublayer: bool = False,
) -> list:
    """
    Run the full Phase 1 analysis pipeline.

    Parameters
    ----------
    models_to_run  : model name keys from MODEL_CONFIGS (default: all)
    prompts_to_run : prompt keys from PROMPTS (default: all)
    run_extended   : if True, use ALBERT extended-iteration mode for ALBERT models
    run_sublayer   : if True, also run the full analysis on the post-attention and
                     post-FFN sublayer residual streams (Fix 14).  Each sublayer
                     stream is saved as a separate run directory labelled
                     ``{model}@attn`` and ``{model}@ffn``.

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

        # Fix 13: random-init baseline — re-randomise weights after architecture
        # load so the model has the same structure but no learned representations.
        if MODEL_CONFIGS[model_name].get("random_init", False):
            print(f"  Re-initialising weights randomly (random_init=True)…")
            model.init_weights()
            print(f"  Done — running with randomly initialised weights.")

        v_spectrum   = analyze_value_eigenspectrum(model, model_name, OUTPUT_DIR)

        cfg          = MODEL_CONFIGS[model_name]
        use_extended = run_extended and cfg["is_albert"] and ALBERT_SNAPSHOTS

        if use_extended:
            model_results = _run_albert_extended(
                model, tokenizer, model_name, prompts_to_run, umap_dir
            )
        else:
            model_results = _run_standard(
                model, tokenizer, model_name, prompts_to_run, umap_dir,
                run_sublayer=run_sublayer,
            )

        # Attach V spectrum to every run result for this model so Phase 2
        # cross-referencing (plateau locations vs V eigenvalue sign distribution)
        # doesn't require re-extracting the model.
        for r in model_results:
            r["v_spectrum"] = v_spectrum

        all_results += model_results

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_results) > 1:
        print("\nGenerating cross-model comparison plots...")
        # P1-2: Exclude repeated_tokens from metastability analyses.
        # It tests collapse of a degenerate initial distribution, not
        # metastability.  Keep it as a separate control in the full results
        # but exclude from cross-run comparison and aggregation.
        metastability_results = [
            r for r in all_results if r["prompt"] != "repeated_tokens"
        ]
        control_results = [
            r for r in all_results if r["prompt"] == "repeated_tokens"
        ]
        if control_results:
            print(f"  ({len(control_results)} repeated_tokens runs excluded "
                  f"from metastability aggregation — kept as collapse controls)")

        plot_cross_model_comparison(metastability_results, OUTPUT_DIR)
        generate_cross_run_report(
            metastability_results, OUTPUT_DIR,
            control_results=control_results,
        )

    # Write global artifacts aggregated across all per-prompt runs.
    # pair_agreement.json lands at the OUTPUT_DIR root for Phase 3.
    print("\nAggregating global artifacts...")
    aggregate_global_artifacts(all_results, OUTPUT_DIR)

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


def _run_standard(model, tokenizer, model_name, prompts_to_run, umap_dir,
                  run_sublayer: bool = False):
    """
    Standard path: use model's native layer stack.
    Active when --no-extended is passed or for non-ALBERT models.

    When run_sublayer=True, an additional pass extracts the post-attention
    and post-FFN intermediate residual streams and runs the full analysis
    on each.  Results are saved to separate ``{stem}@attn`` / ``{stem}@ffn``
    run directories and are excluded from the cross-run comparison (they are
    supplementary, not independent model runs).
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

        # Fix 14: sublayer analysis — post-attn and post-FFN streams.
        if run_sublayer:
            _run_sublayer_analysis(
                model, tokenizer, model_name, prompt_key,
                PROMPTS[prompt_key], tokens, umap_dir,
            )

    return results_list


def _run_sublayer_analysis(model, tokenizer, model_name, prompt_key,
                           prompt_text, tokens, umap_dir):
    """
    Extract post-attention and post-FFN residual streams via forward hooks
    and run the full analysis on each.

    Architecture notes
    ------------------
    GPT-2   : each Block has .attn (attention) and .mlp (FFN).
              Residual additions happen *after* each submodule in the Block's
              forward(), so hooking the Block's output gives the full-block
              residual, but we need the intermediate.  We hook attn.c_proj
              output + the residual (approximated as block_input + attn_out)
              and mlp output + that (= full block output = same as hidden_states).
              Simpler and equally informative: hook just after the residual
              add for each sub-layer using the Block's forward.

    BERT / ALBERT : each BertLayer / AlbertLayer exposes
              self.attention.output (post-attn-add-norm) and
              self.output (post-FFN-add-norm) as distinct submodules.
              Hooking their outputs gives the post-attention and post-FFN
              residual streams directly.

    Because architecture introspection is complex, we use a best-effort
    approach: iterate named modules looking for known submodule names and
    hook the ones we find.  If neither set is found we skip gracefully with
    a warning.
    """
    import torch

    def _find_sublayer_modules(m, model_name_lc):
        """
        Return (attn_modules, ffn_modules) lists of submodules to hook.

        Each list has one entry per transformer layer, in layer order.
        Returns ([], []) when the architecture is not recognised.
        """
        attn_mods, ffn_mods = [], []

        if "gpt2" in model_name_lc:
            # GPT-2: transformer.h[i].attn  and  transformer.h[i].mlp
            h_blocks = None
            for name, mod in m.named_modules():
                if name == "transformer":
                    h_blocks = list(mod.h)
                    break
            if h_blocks:
                attn_mods = [b.attn for b in h_blocks]
                ffn_mods  = [b.mlp  for b in h_blocks]

        elif "albert" in model_name_lc or "bert" in model_name_lc:
            # BERT / ALBERT: encoder.layer[i].attention  and  encoder.layer[i]
            # .intermediate + output (hook the full FFN output at .output).
            # For BertLayer: attention.output.dense (post-attn-add-norm)
            #                output.dense (post-FFN-add-norm)
            # We hook the BertAttention and BertOutput *submodule* outputs.
            layers_list = []
            for name, mod in m.named_modules():
                # Works for both BertEncoder and AlbertTransformer
                if name in ("encoder", "albert_model.encoder", "bert.encoder"):
                    try:
                        layers_list = list(mod.layer)
                    except AttributeError:
                        layers_list = list(mod.albert_layer_groups[0].albert_layers)
                    break
            for layer in layers_list:
                try:
                    attn_mods.append(layer.attention)
                except AttributeError:
                    pass
                try:
                    ffn_mods.append(layer.output)
                except AttributeError:
                    pass

        return attn_mods, ffn_mods

    model_name_lc = model_name.lower()
    attn_mods, ffn_mods = _find_sublayer_modules(model, model_name_lc)

    if not attn_mods or not ffn_mods:
        print(f"    [sublayer] Architecture not recognised for {model_name} — skipping.")
        return

    n_layers = len(attn_mods)

    for sublayer_label, mod_list in [("attn", attn_mods), ("ffn", ffn_mods)]:
        # Collect one (n_tokens, d_model) tensor per layer via hooks.
        captured: list = [None] * n_layers
        handles  = []

        def make_hook(idx):
            def hook(module, inp, out):
                # out may be a tuple (BERT attention returns (context, weights))
                tensor = out[0] if isinstance(out, (tuple, list)) else out
                # Remove batch dimension (batch=1).
                captured[idx] = tensor.detach().squeeze(0).cpu()
            return hook

        for i, mod in enumerate(mod_list):
            handles.append(mod.register_forward_hook(make_hook(i)))

        try:
            inputs = tokenizer(
                prompt_text, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)
        except Exception as exc:
            print(f"    [sublayer/{sublayer_label}] Forward pass failed: {exc}")
            for h in handles:
                h.remove()
            continue
        finally:
            for h in handles:
                h.remove()

        # Drop any None slots (layers that didn't fire, shouldn't happen).
        hs_sub = [t for t in captured if t is not None]
        if len(hs_sub) != n_layers:
            print(f"    [sublayer/{sublayer_label}] Only {len(hs_sub)}/{n_layers} "
                  f"hooks fired — skipping.")
            continue

        # Run the full analysis on the sublayer stream.
        # Attentions are not meaningful here (they belong to the full block),
        # so pass an empty list.
        eff_model_name = f"{model_name}@{sublayer_label}"
        print(f"    Sublayer analysis: {eff_model_name}")
        try:
            sub_results = analyze_trajectory(
                hs_sub, [], prompt_key, eff_model_name, tokens,
                umap_dir=umap_dir,
            )
        except Exception as exc:
            print(f"    [sublayer/{sublayer_label}] analyze_trajectory failed: {exc}")
            continue

        _generate_plots(sub_results, OUTPUT_DIR)
        sub_stem    = f"{eff_model_name.replace('/', '_').replace('@', '_')}_{prompt_key}"
        sub_run_dir = OUTPUT_DIR / sub_stem
        save_run(sub_results, hs_sub, [], sub_run_dir)
        generate_llm_report(sub_results, sub_run_dir)
        print(f"    Sublayer run saved to: {sub_run_dir}/")


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
    from core.config import (
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
    parser.add_argument("--legacy-snapshots", action="store_true",
                        help="Use legacy ALBERT snapshots [12,24,36,48] instead of dense sweep [6..60 step 2]")
    parser.add_argument("--fast", action="store_true",
                        help="albert-base-v2 + wiki_paragraph")
    parser.add_argument("--random-baseline", action="store_true",
                        help="Add albert-base-v2-random (untrained control) to the run")
    parser.add_argument("--sublayer", action="store_true",
                        help="Also analyse post-attention and post-FFN sublayer streams separately")
    parser.add_argument("--length-sweep", action="store_true",
                        help="Run wiki_paragraph truncated at each LENGTH_SWEEP_TOKENS target")
    parser.add_argument("--replot", type=str, default=None, metavar="RUN_DIR",
                        help="Recreate all plots from a saved run directory")
    parser.add_argument("--summary", type=str, default=None, metavar="RUN_DIR",
                        help="Print text summary of a saved run")
    args = parser.parse_args()

    # P1-6: Apply legacy snapshot override before running
    if args.legacy_snapshots:
        from core.config import ALBERT_SNAPSHOTS_LEGACY
        import core.config as _cfg
        _cfg.ALBERT_SNAPSHOTS = ALBERT_SNAPSHOTS_LEGACY
        _cfg.ALBERT_MAX_ITERATIONS = 48

    if args.replot:
        replot_all(Path(args.replot))

    elif args.summary:
        from .reporting import print_run_summary
        print_run_summary(Path(args.summary))

    else:
        if args.fast:
            models  = ["albert-base-v2"]
            prompts = ["wiki_paragraph"]
            # Fast mode uses legacy snapshots to keep runtime short
            import core.config as _cfg
            _cfg.ALBERT_SNAPSHOTS = _cfg.ALBERT_SNAPSHOTS_LEGACY
            _cfg.ALBERT_MAX_ITERATIONS = 48
        else:
            models  = args.models
            prompts = args.prompts

        # Fix 13: inject the untrained control model if requested.
        if args.random_baseline and "albert-base-v2-random" not in models:
            models = list(models) + ["albert-base-v2-random"]

        # Fix 15: build truncated wiki_paragraph prompt variants.
        if args.length_sweep:
            import core.config as _cfg
            base_text = PROMPTS["wiki_paragraph"]
            words     = base_text.split()
            for target in LENGTH_SWEEP_TOKENS:
                # Rough word-level truncation: ~0.75 tokens per word on average
                # for English, so target * 0.75 words ≈ target tokens.
                n_words  = max(1, int(target * 0.75))
                snippet  = " ".join(words[:n_words])
                key      = f"wiki_{target}"
                _cfg.PROMPTS[key] = snippet
                if key not in prompts:
                    prompts = list(prompts) + [key]
            # Ensure the sweep models default to albert-base-v2 unless overridden.
            if not args.models or args.models == list(MODEL_CONFIGS.keys()):
                models = ["albert-base-v2"]

        # Fix 14: pass sublayer flag through to run_all.
        run_all(
            models_to_run=models,
            prompts_to_run=prompts,
            run_extended=not args.no_extended,
            run_sublayer=args.sublayer,
        )
