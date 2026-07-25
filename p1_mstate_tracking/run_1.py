"""
p1_mstate_tracking/run_1.py — Experiment orchestrator and CLI entry point.

Usage examples:
    python -m p1_mstate_tracking.run_1                      # 7 Blog 1 architectures
    python -m p1_mstate_tracking.run_1 --list-models        # groups + registry keys
    python -m p1_mstate_tracking.run_1 --fast               # albert-base-v2, 1 prompt
    python -m p1_mstate_tracking.run_1 --models gpt2-xl albert-base-v2
    python -m p1_mstate_tracking.run_1 --models replication-gate
    python -m p1_mstate_tracking.run_1 --models pythia-410m-pilot --prompts wiki_paragraph
    python -m p1_mstate_tracking.run_1 --dtype auto         # legacy bf16-on-CUDA
    python -m p1_mstate_tracking.run_1 --no-extended
    python -m p1_mstate_tracking.run_1 --replot results/2024-01-01_12-00-00/gpt2_wiki_paragraph
    python -m p1_mstate_tracking.run_1 --summary results/2024-01-01_12-00-00/gpt2_wiki_paragraph

--models accepts registry keys and MODEL_GROUPS names interchangeably.
It no longer defaults to every key in MODEL_CONFIGS: the Pythia checkpoint
registry added 37 entries, so that default silently became "download every
published Pythia checkpoint". It now defaults to DEFAULT_MODELS, which is
the seven Blog 1 architectures.
"""

import sys
import traceback
import torch
from datetime import datetime
from pathlib import Path

from core.sublayer_streams import extract_sublayer_streams, UnsupportedArchitecture

from core.config import (
    BASE_RESULTS_DIR, MODEL_CONFIGS, PROMPTS,
    ALBERT_MAX_ITERATIONS, ALBERT_SNAPSHOTS, LENGTH_SWEEP_TOKENS,
    RANDOM_INIT_SEED, MODEL_DTYPE, MODEL_GROUPS, DEFAULT_MODELS, RANDOM_CONTROLS,
)
from core.models import (
    load_model, extract_activations, extract_albert_extended,
    randomize_weights, describe_extraction, resolve_dtype, dtype_name,
)
from core.model_family import model_family

from .analysis_p1 import analyze_trajectory
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
from .reporting_p1 import print_summary, generate_llm_report, generate_cross_run_report
from .p1_io import save_run, replot_all, aggregate_global_artifacts
from .clustering import HAS_UMAP

# Module-level output directory set by run_all before any analyze_trajectory call.
OUTPUT_DIR: Path = BASE_RESULTS_DIR


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def resolve_model_names(requested) -> list:
    """Expand MODEL_GROUPS names and validate registry keys, preserving order.

    A group name and a registry key are accepted in the same position, so
    `--models replication-gate gpt2-xl` works. Unknown names are fatal
    rather than skipped: --models used to carry argparse `choices=` over all
    47 registry keys, which produced an unreadable error and, worse, made a
    typo look like a valid-but-empty selection once the choices list stopped
    fitting on a screen.
    """
    resolved, unknown = [], []
    for name in requested:
        if name in MODEL_GROUPS:
            for m in MODEL_GROUPS[name]:
                if m not in resolved:
                    resolved.append(m)
        elif name in MODEL_CONFIGS:
            if name not in resolved:
                resolved.append(name)
        else:
            unknown.append(name)

    if unknown:
        sys.exit(
            f"Unknown model/group: {', '.join(unknown)}\n"
            f"Groups:   {', '.join(sorted(MODEL_GROUPS))}\n"
            f"Run --list-models for the full registry."
        )
    return resolved


def print_model_catalogue() -> None:
    print("\nGroups (usable directly as --models arguments):")
    for group, members in sorted(MODEL_GROUPS.items()):
        print(f"  {group:<24} {len(members):>3} models")
        for m in members:
            print(f"      {m}")
    print(f"\nDefault when --models is omitted: {', '.join(DEFAULT_MODELS)}")
    print(f"\nAll registry keys ({len(MODEL_CONFIGS)}):")
    for key in sorted(MODEL_CONFIGS):
        print(f"  {key}")


def add_random_controls(models: list) -> list:
    """Append each selected model's untrained control, if it has one.

    Family-aware rather than hardcoded: selecting only Pythia checkpoints
    and passing --random-baseline used to add two unrelated ALBERT/GPT-2
    runs to the sweep. It now says why nothing was added and names the
    substitute.

    this code is meant to replace the two "--random-baseline" calls, which does not makes sense. This code is palced here to alter later.
    we are not so concerned with random atm bc pythia chkpt0 is being used as the random control, and is already included in all pythia runs.
    """
    out   = list(models)
    added = []
    for m in models:
        control = RANDOM_CONTROLS.get(m)
        if control and control in MODEL_CONFIGS and control not in out:
            out.append(control)
            added.append(control)

    if added:
        print(f"  --random-baseline: added {', '.join(added)}")
    else:
        print("  --random-baseline: no untrained control is defined for any "
              "selected model. For Pythia, the published step-0 checkpoint is "
              "that object — request it directly, e.g. "
              "--models pythia-1.4b-step0.")
    return out
# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_all(
    models_to_run: list = None,
    prompts_to_run: list = None,
    run_extended: bool = True,
    run_sublayer: bool = False,
    random_seed=None,
    dtype=None,
    ) -> list:
    """
    Run the full Phase 1 analysis pipeline.

    Parameters
    ----------
    models_to_run  : model name keys from MODEL_CONFIGS (default: DEFAULT_MODELS,
                     i.e. the seven Blog 1 architectures — NOT every registry
                     key, which now includes 37 Pythia checkpoints)
    prompts_to_run : prompt keys from PROMPTS (default: all)
    run_extended   : if True, use ALBERT extended-iteration mode for ALBERT models
    run_sublayer   : if True, also run the full analysis on the post-attention and
                     post-FFN sublayer residual streams (Fix 14).  Each sublayer
                     stream is saved as a separate run directory labelled
                     ``{model}@attn`` and ``{model}@ffn``.
    dtype          : dtype spec passed to load_model (default: core.config.MODEL_DTYPE)

    Returns
    -------
    list of results dicts, one per (model, prompt) combination
    """
    global OUTPUT_DIR
    seed        = random_seed if random_seed is not None else RANDOM_INIT_SEED
    torch_dtype = resolve_dtype(dtype)

    if models_to_run is None:
        models_to_run = list(DEFAULT_MODELS)
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
    _write_manifest(timestamp, models_to_run, prompts_to_run, run_extended, torch_dtype)

    print(f"\nExperiment directory: {OUTPUT_DIR}")
    print(f"Models ({len(models_to_run)}): {', '.join(models_to_run)}")
    print(f"dtype: {dtype_name(torch_dtype)}")
    all_results = []

    for model_name in models_to_run:
        family = model_family(model_name)
        if family is None:
            print(f"\n[warn] {model_name}: no known architecture family — "
                  f"causal-mask handling and V extraction will both be skipped.")

        print(f"\nLoading {model_name}...")
        try:
            model, tokenizer = load_model(model_name, dtype=torch_dtype)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        # Fix 13: random-init baseline — re-randomise weights after architecture
        # load so the model has the same structure but no learned representations.
        # Wrapped: an unsupported scheme used to raise here, outside any
        # handler, killing the whole sweep instead of skipping one model.
        cfg = MODEL_CONFIGS[model_name]
        if cfg.get("random_init", False):
            scheme = cfg.get("random_init_scheme", "orthogonal")
            print(f"  Re-initialising weights (scheme={scheme}, seed={seed})…")
            try:
                info = randomize_weights(model, scheme=scheme, seed=seed)
            except Exception as e:
                print(f"  Randomisation failed ({e}) — skipping {model_name} "
                      f"rather than running it on trained weights.")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            print(f"  Done — {info['n_weight_matrices']} matrices, "
                  f"{info['n_embeddings']} embeddings re-init; checksum "
                  f"{info['checksum_before']:.3f} → {info['checksum_after']:.3f}.")

        # Wrapped for the same reason: an unexpected weight layout on one
        # checkpoint should cost that model's V spectrum, not the sweep.
        try:
            v_spectrum = analyze_value_eigenspectrum(model, model_name, OUTPUT_DIR)
        except Exception as e:
            print(f"  V spectrum failed for {model_name}: {e}")
            traceback.print_exc()
            v_spectrum = {}

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

            meta = describe_extraction(
                model, effective_model_name, hidden_states, attentions
            )
            # ALBERT's iterated map has no stack-final LayerNorm and index 0
            # here is the post-projection embedding, not the raw one.
            meta["final_hidden_state_is_post_ln"] = False
            meta["hidden_state_0_is_embedding"]   = True

            results = analyze_trajectory(
                hidden_states, attentions, prompt_key, effective_model_name,
                tokens, umap_dir=umap_dir, extraction_meta=meta,
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

        # Fix 4: previously never supplied, so analyze_trajectory always took
        # its None-default branch and recorded a constant.
        meta = describe_extraction(model, model_name, hidden_states, attentions)

        results = analyze_trajectory(
            hidden_states, attentions, prompt_key, model_name,
            tokens, umap_dir=umap_dir, extraction_meta=meta,
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
    Run the full analysis on the post-attention and post-FFN residual
    streams, saved as ``{model}@attn`` and ``{model}@ffn``.

    All hook logic now lives in core/sublayer_streams.py, which owns the
    per-family stream semantics. The version that lived here captured
    GPT-2's attention and MLP *deltas* and labelled them residual streams,
    never ran on ALBERT at all, and overwrote every capture from a
    weight-shared layer. See that module's docstring.

    `tokens` from the caller is ignored in favour of the tokenisation the
    stream extractor actually ran, so a truncation difference cannot
    misalign labels against activations.
    """
    try:
        streams = extract_sublayer_streams(
            model, tokenizer, prompt_text, model_name
        )
    except UnsupportedArchitecture as exc:
        print(f"    [sublayer] {exc}")
        return
    except Exception as exc:
        print(f"    [sublayer] Extraction failed: {exc}")
        traceback.print_exc()
        return

    print(f"    [sublayer] {streams.n_layers} layers, "
          f"semantics={streams.semantics}")

    for label, hs_sub in (("attn", streams.post_attn), ("ffn", streams.post_ffn)):
        eff_model_name = f"{model_name}@{label}"
        print(f"    Sublayer analysis: {eff_model_name}")

        # Attentions belong to the full block, not to either stream, so an
        # empty list is passed and analyze_trajectory skips the sinkhorn
        # and entropy family for these runs.
        meta = describe_extraction(model, eff_model_name, hs_sub, [])
        meta.update(streams.meta_overrides())

        try:
            sub_results = analyze_trajectory(
                hs_sub, [], prompt_key, eff_model_name, streams.tokens,
                umap_dir=umap_dir, extraction_meta=meta,
            )
        except Exception as exc:
            print(f"    [sublayer/{label}] analyze_trajectory failed: {exc}")
            traceback.print_exc()
            continue

        _generate_plots(sub_results, OUTPUT_DIR)
        sub_stem    = f"{eff_model_name.replace('/', '_').replace('@', '_')}_{prompt_key}"
        sub_run_dir = OUTPUT_DIR / sub_stem
        save_run(sub_results, hs_sub, [], sub_run_dir)
        generate_llm_report(sub_results, sub_run_dir)
        print(f"    Sublayer run saved to: {sub_run_dir}/")

def run_sublayer_only(models_to_run: list, prompts_to_run: list,
                      output_dir: Path, dtype=None) -> None:
    """
    Only the post-attention/post-FFN sublayer streams (Fix 14) — no
    analyze_trajectory/save_run/_generate_plots on the full block.

    output_dir must be the *existing* run's parent directory (the one
    containing "{model}_{prompt}") so the new "{model}@attn_{prompt}" /
    "{model}@ffn_{prompt}" folders land next to it — discover_runs() in
    the visualization package needs them in the same directory to pair
    them up.
    """
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    umap_dir = OUTPUT_DIR / "umap"
    umap_dir.mkdir(exist_ok=True)

    torch_dtype = resolve_dtype(dtype)

    for model_name in models_to_run:
        print(f"\nLoading {model_name}...")
        try:
            model, tokenizer = load_model(model_name, dtype=torch_dtype)
        except Exception as e:
            print(f"  Failed: {e}")
            continue
        for prompt_key in prompts_to_run:
            print(f"  Prompt: {prompt_key}")
            _run_sublayer_analysis(
                model, tokenizer, model_name, prompt_key,
                PROMPTS[prompt_key], None, umap_dir,
            )


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


def _write_manifest(timestamp, models_to_run, prompts_to_run,
                    run_extended, torch_dtype) -> None:
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
        # Recorded because a dtype change silently shifts the V eigenspectrum
        # and the effective-rank gate. A cross-run comparison that spans two
        # dtypes is not a comparison.
        f"dtype          : {dtype_name(torch_dtype)}",
        f"dtype_config   : {MODEL_DTYPE}",
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
    # No `choices=` — MODEL_CONFIGS now has 47 keys and argparse's error
    # output for that is unusable. resolve_model_names validates instead,
    # and accepts MODEL_GROUPS names in the same position.
    parser.add_argument("--models", nargs="+", default=None, metavar="NAME",
                        help="registry keys and/or group names "
                             f"({', '.join(sorted(MODEL_GROUPS))}); "
                             "default: the Blog 1 architectures")
    parser.add_argument("--list-models", action="store_true",
                        help="print groups and registry keys, then exit")
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS.keys()),
                        choices=list(PROMPTS.keys()))
    parser.add_argument("--dtype", type=str, default=None, metavar="SPEC",
                        help="float32 (default) | bfloat16 | float16 | auto. "
                             "auto = bfloat16 on CUDA, float32 on CPU — the "
                             "pre-fix behaviour; see core/models.py on why it "
                             "is no longer the default")
    parser.add_argument("--no-extended", action="store_true",
                        help="Disable ALBERT extended-iteration mode")
    parser.add_argument("--legacy-snapshots", action="store_true",
                        help="Use legacy ALBERT snapshots [12,24,36,48] instead of dense sweep [6..60 step 2]")
    parser.add_argument("--fast", action="store_true",
                        help="albert-base-v2 + wiki_paragraph")
    parser.add_argument("--random-baseline", action="store_true",
                        help="Add the untrained controls to the run")
    parser.add_argument("--sublayer", action="store_true",
                        help="Also analyse post-attention and post-FFN sublayer streams separately")
    parser.add_argument("--length-sweep", action="store_true",
                        help="Run wiki_paragraph truncated at each LENGTH_SWEEP_TOKENS target")
    parser.add_argument("--replot", type=str, default=None, metavar="RUN_DIR",
                        help="Recreate all plots from a saved run directory")
    parser.add_argument("--summary", type=str, default=None, metavar="RUN_DIR",
                        help="Print text summary of a saved run")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="Seed for random-init controls (default: config, 0)")
    parser.add_argument("--sublayer-only", action="store_true",
                        help="Skip the full-block pass; only generate @attn/@ffn for an existing run")
    parser.add_argument("--output-dir", type=str, default=None, metavar="DIR")
    args = parser.parse_args()

    if args.list_models:
        print_model_catalogue()
        sys.exit(0)

    # P1-6: Apply legacy snapshot override before running
    if args.legacy_snapshots:
        from core.config import ALBERT_SNAPSHOTS_LEGACY
        import core.config as _cfg
        _cfg.ALBERT_SNAPSHOTS = ALBERT_SNAPSHOTS_LEGACY
        _cfg.ALBERT_MAX_ITERATIONS = 48

    if args.replot:
        replot_all(Path(args.replot))

    elif args.summary:
        from .reporting_p1 import print_run_summary
        print_run_summary(Path(args.summary))

    elif args.sublayer_only:
        if not args.output_dir:
            sys.exit("--sublayer-only requires --output-dir")
        models = resolve_model_names(args.models or DEFAULT_MODELS)
        if args.random_baseline:
            models += [m for m in MODEL_GROUPS["blog1-random"] if m not in models]
        run_sublayer_only(
            models_to_run=models,
            prompts_to_run=args.prompts,
            output_dir=Path(args.output_dir),
            dtype=args.dtype,
        )

    else:
        if args.fast:
            models  = ["albert-base-v2"]
            prompts = ["wiki_paragraph"]
            # Fast mode uses legacy snapshots to keep runtime short
            import core.config as _cfg
            _cfg.ALBERT_SNAPSHOTS = _cfg.ALBERT_SNAPSHOTS_LEGACY
            _cfg.ALBERT_MAX_ITERATIONS = 48
        else:
            models  = resolve_model_names(args.models or DEFAULT_MODELS)
            prompts = args.prompts

        # Fix 13: inject the untrained controls if requested.
        if args.random_baseline:
            models = list(models) + [
                m for m in MODEL_GROUPS["blog1-random"] if m not in models
            ]

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
            if not args.models:
                models = ["albert-base-v2"]

        # Fix 14: pass sublayer flag through to run_all.
        run_all(
            models_to_run=models,
            prompts_to_run=prompts,
            run_extended=not args.no_extended,
            run_sublayer=args.sublayer,
            random_seed=args.seed,
            dtype=args.dtype,
        )

