"""
run.py — Phase 3 entry point.

Default behavior (no arguments): runs the full pipeline for both models.
Flags narrow scope:

    python -m phase3.run                          # everything, both models
    python -m phase3.run --albert-only            # ALBERT-xlarge only
    python -m phase3.run --gpt2-only              # GPT-2-large only
    python -m phase3.run --skip-cache             # skip extraction (use existing)
    python -m phase3.run --skip-train             # skip training (use existing)
    python -m phase3.run --skip-analyze           # skip analysis
    python -m phase3.run --data-source tinystories --n-texts 10000
    python -m phase3.run --phase1-dir results/... --phase2-dir results/...

The pipeline auto-detects existing cached activations and trained
checkpoints.  If they exist, it skips that stage.  Use --force-cache
or --force-train to override.
"""

import argparse
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from core.config import DEVICE, PROMPTS
from core.models import load_model

from .data import (
    ExtractionConfig, LAYER_PRESETS, FEATURE_PRESETS, SUPPORTED_MODELS,
    load_texts, cache_activations, extract_activations, is_cache_valid,
    is_trained, ActivationDataset, PromptActivationStore,
)
from .crosscoder import Crosscoder, CrosscoderConfig, ActivationType
from .training import TrainingConfig, train, load_trained_crosscoder
from .analysis import run_all_analyses, save_results


# ---------------------------------------------------------------------------
# Paths convention
# ---------------------------------------------------------------------------

def _cache_dir(model_name: str) -> Path:
    return Path("activation_cache") / model_name.replace("/", "_").replace("-", "_")

def _checkpoint_dir(model_name: str) -> Path:
    return Path("checkpoints") / model_name.replace("/", "_").replace("-", "_")

def _results_dir(model_name: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("results") / "phase3" / f"{model_name.replace('/', '_')}_{ts}"


# ---------------------------------------------------------------------------
# Pipeline per model
# ---------------------------------------------------------------------------

def run_model(model_name: str, args) -> dict:
    """Full pipeline for one model: cache → train → analyze."""

    print(f"\n{'='*70}")
    print(f"  Phase 3: {model_name}")
    print(f"{'='*70}")

    cache_dir = _cache_dir(model_name)
    ckpt_dir = _checkpoint_dir(model_name)
    layers = LAYER_PRESETS[model_name]

    # ------------------------------------------------------------------
    # Stage 1: Cache activations
    # ------------------------------------------------------------------
    if args.skip_cache:
        print("\n  [cache] Skipped (--skip-cache)")
    elif is_cache_valid(cache_dir) and not args.force_cache:
        print(f"\n  [cache] Found existing cache: {cache_dir}")
    else:
        print(f"\n  [cache] Extracting activations...")
        print(f"    Model:  {model_name}")
        print(f"    Layers: {layers}")
        print(f"    Source: {args.data_source}")

        model, tokenizer = load_model(model_name)

        # Load training texts
        texts = load_texts(
            source=args.data_source,
            n_texts=args.n_texts,
            data_dir=args.data_dir,
        )

        # Append Phase 1 prompts (excluding degenerate control)
        for key, text in PROMPTS.items():
            if key != "repeated_tokens":
                texts.append(text)

        config = ExtractionConfig(
            model_name=model_name,
            layer_indices=layers,
            max_seq_len=args.max_seq_len,
            shard_size=50_000,
            cache_dir=str(cache_dir),
        )

        cache_activations(
            model, tokenizer, texts, config,
            device=DEVICE, batch_size=args.extract_batch_size,
        )

        # Cache eval prompts separately
        print("    Caching evaluation prompts...")
        prompt_store = PromptActivationStore()
        for key, text in PROMPTS.items():
            if key == "repeated_tokens":
                continue
            results = extract_activations(model, tokenizer, [text], config, device=DEVICE)
            tokens = tokenizer.convert_ids_to_tokens(
                tokenizer(text, truncation=True, max_length=args.max_seq_len)["input_ids"]
            )
            prompt_store.add(key, results[0], tokens, layers)
        prompt_store.save(cache_dir / "eval_prompts")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage 2: Train crosscoder
    # ------------------------------------------------------------------
    final_dir = ckpt_dir / "final"

    if args.skip_train:
        print("\n  [train] Skipped (--skip-train)")
    elif is_trained(ckpt_dir) and not args.force_train:
        print(f"\n  [train] Found existing model: {final_dir}")
    else:
        print(f"\n  [train] Training crosscoder...")

        with open(cache_dir / "meta.json") as f:
            meta = json.load(f)

        d_model = meta["d_model"]
        n_layers = len(meta["layer_indices"])
        n_features = args.n_features or FEATURE_PRESETS.get(model_name, 16384)

        cc_cfg = CrosscoderConfig(
            d_model=d_model,
            n_layers=n_layers,
            n_features=n_features,
            activation=ActivationType(args.activation),
            k=args.k,
        )

        train_cfg = TrainingConfig(
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            total_steps=args.total_steps,
            batch_size=args.train_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=str(ckpt_dir),
            resample_dead=True,
            num_workers=args.num_workers,
        )

        print(f"    d_model={d_model}, n_layers={n_layers}, features={n_features}")
        print(f"    Activation: {args.activation}, k={args.k}")
        print(f"    Input dim: {cc_cfg.d_input}")

        dataset = ActivationDataset(cache_dir)
        print(f"    Dataset: {len(dataset):,} tokens")

        crosscoder = Crosscoder(cc_cfg)
        n_params = sum(p.numel() for p in crosscoder.parameters())
        vram_est_gb = n_params * 4 * 3 / 1e9  # fp32 weights + 2x Adam state
        print(f"    Parameters: {n_params:,} (~{vram_est_gb:.1f}GB model+optimizer)")

        result = train(crosscoder, dataset, train_cfg, device=DEVICE)
        print(f"    Final loss: {result['final_loss']:.4f}")
        print(f"    Dead features: {result['n_dead_final']}")

    # ------------------------------------------------------------------
    # Stage 3: Analyze
    # ------------------------------------------------------------------
    if args.skip_analyze:
        print("\n  [analyze] Skipped (--skip-analyze)")
        return {}

    if not is_trained(ckpt_dir):
        print("\n  [analyze] No trained model found, skipping analysis")
        return {}

    print(f"\n  [analyze] Running analyses...")

    crosscoder = load_trained_crosscoder(final_dir, device=DEVICE)

    eval_dir = cache_dir / "eval_prompts"
    if not eval_dir.exists():
        print("    No eval prompts found, skipping analysis")
        return {}
    prompt_store = PromptActivationStore.load(eval_dir)

    artifacts = _load_artifacts(args, model_name)
    artifacts["layer_indices"] = layers

    print(f"    Prompts: {list(prompt_store.keys())}")
    print(f"    Artifacts: {list(artifacts.keys())}")

    results = run_all_analyses(
        crosscoder, prompt_store, artifacts,
        config={"lifetime_threshold_frac": 0.1, "multilayer_min_layers": 3},
    )

    out_dir = _results_dir(model_name)
    save_results(results, out_dir / "analysis_results.json")

    # Also save a copy of the crosscoder config for reference
    with open(final_dir / "config.json") as f:
        cc_config = json.load(f)
    with open(out_dir / "crosscoder_config.json", "w") as f:
        json.dump(cc_config, f, indent=2)

    _print_summary(results, model_name)
    return results


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_artifacts(args, model_name: str) -> dict:
    """
    Load Phase 1/2 artifacts.  Lenient — skips what's missing.

    Handles two Phase 2 formats:
      1. Exported v_projectors.npz (from export_projectors.py)
      2. Native Phase 2 format (ov_projectors_{stem}.npz via load_weight_decomposition)
    """
    artifacts = {}

    # --- Phase 1 ---
    if args.phase1_dir:
        phase1 = Path(args.phase1_dir)
        metrics_path = phase1 / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                p1 = json.load(f)

            prompt_key = p1.get("prompt", "unknown")
            layers_data = p1.get("layers", [])

            # Violation layers
            v_layers = []
            for i, lr in enumerate(layers_data):
                if i > 0:
                    prev_e = layers_data[i - 1].get("energies", {}).get("1.0")
                    curr_e = lr.get("energies", {}).get("1.0")
                    if prev_e and curr_e and curr_e < prev_e - 1e-6:
                        v_layers.append(i)
            if v_layers:
                artifacts["violation_layers"] = {prompt_key: v_layers}

            # HDBSCAN labels
            hdb = {}
            for i, lr in enumerate(layers_data):
                labels = lr.get("clustering", {}).get("hdbscan", {}).get("labels")
                if labels:
                    if prompt_key not in hdb:
                        hdb[prompt_key] = {}
                    hdb[prompt_key][i] = labels
            if hdb:
                artifacts["hdbscan_labels"] = hdb

    # --- Phase 2 ---
    if args.phase2_dir:
        phase2 = Path(args.phase2_dir)

        # Try exported format first (v_projectors.npz)
        exported = phase2 / "v_projectors.npz"
        if exported.exists():
            data = np.load(exported, allow_pickle=True)
            is_pl = bool(data.get("is_per_layer", np.array(False)))
            if is_pl:
                # Per-layer: stacked (L, d, d) arrays
                attract = data["sym_attract"]  # (L, d, d)
                repulse = data["sym_repulse"]  # (L, d, d)
                artifacts["v_projectors"] = [
                    {"sym_attract": attract[i], "sym_repulse": repulse[i]}
                    for i in range(attract.shape[0])
                ]
            else:
                artifacts["v_projectors"] = {
                    "sym_attract": data["sym_attract"],
                    "sym_repulse": data["sym_repulse"],
                }
            artifacts["is_per_layer"] = is_pl
            print(f"    Loaded V projectors from {exported}")

        else:
            # Try Phase 2 native format (ov_projectors_{stem}.npz)
            stem = model_name.replace("/", "_")
            native = phase2 / f"ov_projectors_{stem}.npz"
            if native.exists():
                try:
                    from phase2.weights import load_weight_decomposition
                    wd = load_weight_decomposition(phase2, model_name)
                    artifacts["v_projectors"] = wd["projectors"]
                    artifacts["is_per_layer"] = wd["summary"]["is_per_layer"]
                    print(f"    Loaded V projectors from Phase 2 native format")
                except Exception as e:
                    print(f"    Failed to load Phase 2 native format: {e}")

    return artifacts


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: dict, model_name: str):
    print(f"\n{'='*60}")
    print(f"  Phase 3 Summary: {model_name}")
    print(f"{'='*60}")

    ml = results.get("multilayer_fraction", {})
    if ml and "error" not in ml:
        print(f"\n  Multi-layer fraction: {ml.get('multilayer_fraction', 0):.1%} "
              f"({ml.get('multilayer_count', 0)}/{ml.get('n_alive', 0)})")

    lt = results.get("feature_lifetimes", {})
    if lt and "error" not in lt:
        print(f"\n  Feature lifetimes:")
        print(f"    Mean: {lt.get('mean_lifetime', 0):.1f} layers")
        print(f"    Short-lived (≤3): {lt.get('n_short_lived', 0)}")
        print(f"    Long-lived (≥L/2): {lt.get('n_long_lived', 0)}")
        print(f"    Bimodal score: {lt.get('bimodal_score', 'n/a')}")

    va = results.get("v_subspace_alignment", {})
    if va and "error" not in va:
        print(f"\n  V subspace alignment:")
        print(f"    Attractive: {va.get('n_attractive', 0)}")
        print(f"    Repulsive:  {va.get('n_repulsive', 0)}")
        print(f"    Mixed:      {va.get('n_mixed', 0)}")

    lva = results.get("lifetime_vs_alignment", {})
    if lva and "error" not in lva:
        print(f"\n  Lifetime ↔ V-alignment: ρ={lva.get('spearman_rho', 0):.3f}  "
              f"p={lva.get('spearman_pval', 1):.3f}")

    pc = results.get("positional_control", {})
    if pc and "error" not in pc:
        print(f"\n  Positional control: {pc.get('n_positional', 0)}/{pc.get('n_long_lived', 0)} "
              f"({pc.get('positional_fraction', 0):.0%})")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Crosscoder Training on Metastable Dynamics.\n\n"
                    "Run with no arguments for the full pipeline on both models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model selection (default: both)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--albert-only", action="store_true", help="ALBERT-xlarge only")
    group.add_argument("--gpt2-only", action="store_true", help="GPT-2-large only")

    # Stage skipping
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--force-cache", action="store_true", help="Re-extract even if cache exists")
    parser.add_argument("--force-train", action="store_true", help="Retrain even if checkpoint exists")

    # Data
    parser.add_argument("--data-source", type=str, default="c4",
                        choices=["c4", "tinystories", "local"])
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path for source='local'")
    parser.add_argument("--n-texts", type=int, default=50_000,
                        help="Number of texts to extract from")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=8)

    # Training
    parser.add_argument("--n-features", type=int, default=None,
                        help="Dictionary size (default: 4x d_model)")
    parser.add_argument("--activation", type=str, default="batch_topk",
                        choices=["topk", "batch_topk"])
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--grad-accum-steps", type=int, default=4,
                        help="Gradient accumulation (effective batch = batch * accum)")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)

    # Phase 1/2 cross-reference
    parser.add_argument("--phase1-dir", type=str, default=None,
                        help="Phase 1 run directory (for violation layers, clusters)")
    parser.add_argument("--phase2-dir", type=str, default=None,
                        help="Phase 2 run directory (for V projectors)")

    args = parser.parse_args()

    # Determine which models to run
    if args.albert_only:
        models = ["albert-xlarge-v2"]
    elif args.gpt2_only:
        models = ["gpt2-large"]
    else:
        models = SUPPORTED_MODELS

    print(f"Phase 3: Crosscoder Training on Metastable Dynamics")
    print(f"  Models: {models}")
    print(f"  Data:   {args.data_source} ({args.n_texts} texts)")
    print(f"  Device: {DEVICE}")

    all_results = {}
    for model_name in models:
        all_results[model_name] = run_model(model_name, args)

    print(f"\nPhase 3 complete.")


if __name__ == "__main__":
    main()
