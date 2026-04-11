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
    python -m phase3.run --skip-cross-phase       # skip Stage 5 bridge analyses

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
    is_trained, ActivationBuffer, PromptActivationStore,
)
from .crosscoder import Crosscoder, CrosscoderConfig, ActivationType
from .training import TrainingConfig, train, load_trained_crosscoder
from .analysis import run_all_analyses, save_results
from .steering import run_steering, summarise_steering, save_steering_results, analyse_pair_tracking


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
    ckpt_dir  = _checkpoint_dir(model_name)
    layers    = LAYER_PRESETS[model_name]

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
            prompt_store.add(key, results)
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
        print(f"\n  [train] Found existing checkpoint: {final_dir}")
    else:
        print(f"\n  [train] Training crosscoder...")
        d_model    = FEATURE_PRESETS[model_name]["d_model"]
        n_features = args.n_features or FEATURE_PRESETS[model_name]["n_features"]
        n_layers   = len(layers)

        cc_cfg = CrosscoderConfig(
            d_input=d_model * n_layers,
            n_features=n_features,
            activation=ActivationType[args.activation.upper()],
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
        )

        print(f"    Features: {n_features}  k={args.k}")
        print(f"    Input dim: {cc_cfg.d_input}")

        buffer = ActivationBuffer(cache_dir)
        print(f"    Dataset: {buffer.n_tokens:,} tokens")

        crosscoder = Crosscoder(cc_cfg)
        n_params   = sum(p.numel() for p in crosscoder.parameters())
        vram_est_gb = n_params * 4 * 3 / 1e9  # fp32 weights + 2x Adam state
        print(f"    Parameters: {n_params:,} (~{vram_est_gb:.1f}GB model+optimizer)")

        result = train(crosscoder, buffer, train_cfg, device=DEVICE)
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

    crosscoder   = load_trained_crosscoder(final_dir, device=DEVICE)
    eval_dir     = cache_dir / "eval_prompts"
    if not eval_dir.exists():
        print("    No eval prompts found, skipping analysis")
        return {}
    prompt_store = PromptActivationStore.load(eval_dir)

    artifacts = _load_artifacts(args, model_name)
    artifacts["layer_indices"] = layers

    print(f"    Prompts:   {list(prompt_store.keys())}")
    print(f"    Artifacts: {list(artifacts.keys())}")

    results = run_all_analyses(
        crosscoder, prompt_store, artifacts,
        config={"lifetime_threshold_frac": 0.1, "multilayer_min_layers": 3},
    )

    out_dir = _results_dir(model_name)
    save_results(results, out_dir / "analysis_results.json")

    # Save inspect_top_features text report as a standalone readable file.
    # The JSON result embeds the same text under "text_report", but this makes
    # it accessible without parsing JSON — useful for quick inspection.
    itf = results.get("inspect_top_features", {})
    text_report = itf.get("text_report", "") if isinstance(itf, dict) else ""
    if text_report:
        report_path = out_dir / "feature_inspection.txt"
        with open(report_path, "w") as f:
            f.write(text_report)
        print(f"  Feature inspection report: {report_path}")

    # Also save a copy of the crosscoder config for reference
    with open(final_dir / "config.json") as f:
        cc_config = json.load(f)
    with open(out_dir / "crosscoder_config.json", "w") as f:
        json.dump(cc_config, f, indent=2)

    _print_summary(results, model_name)

    # ------------------------------------------------------------------
    # Stage 4: Steering experiment (parts 9–11)
    # ------------------------------------------------------------------
    if not getattr(args, "skip_steer", False):
        fcc           = results.get("feature_cluster_correlation", {})
        plateau_layers = artifacts.get("plateau_layers")
        layers_        = artifacts.get("layer_indices", LAYER_PRESETS[model_name])

        if not plateau_layers:
            print("\n  [steer] Skipping: no plateau_layers artifact "
                  "(run Phase 1 for this model first)")
        elif "error" in fcc:
            print(f"\n  [steer] Skipping: feature_cluster_correlation failed: {fcc['error']}")
        else:
            print(f"\n  [steer] Loading model for steering...")
            from core.models import load_model as _load_model
            steer_model, steer_tokenizer = _load_model(model_name)
            steer_model = steer_model.to(DEVICE)
            steer_model.eval()

            is_albert = "albert" in model_name.lower()

            # Build lifetime class array for outcome annotation
            lt_arr    = []
            lt_result = results.get("feature_lifetimes", {})
            if lt_result and "lifetime_class" in lt_result:
                lt_arr = lt_result["lifetime_class"]

            steer_cfg = {
                "steering_n_features":            getattr(args, "steering_n_features", 5),
                "steering_alpha_multiplier":      getattr(args, "steering_alpha", 1.0),
                "steering_merge_threshold_sigma":  2.0,
                "plateau_min_cluster_size":        3,
            }

            print(f"  [steer] Running steering experiments...")
            steer_results = run_steering(
                model=steer_model,
                tokenizer=steer_tokenizer,
                crosscoder=crosscoder,
                prompt_store=prompt_store,
                prompts=PROMPTS,
                fcc_results=fcc,
                plateau_layers=plateau_layers,
                layer_indices=layers_,
                is_albert=is_albert,
                device=DEVICE,
                config=steer_cfg,
                lifetime_class_arr=lt_arr,
            )

            # Preserve raw results for Stage 5 pair tracking
            results["_steering_results_raw"] = steer_results

            steer_summary = summarise_steering(steer_results)
            save_steering_results(
                steer_results, steer_summary,
                out_dir / "steering_results.json"
            )

            # Print the compact summary to terminal
            for line in steer_summary["text_summary"].splitlines():
                print("  " + line)

            del steer_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage 5: Cross-phase bridge analyses
    # ------------------------------------------------------------------
    if not getattr(args, "skip_cross_phase", False):
        phase1_dir = Path(args.phase1_dir) if getattr(args, "phase1_dir", None) else None
        phase2_dir = Path(args.phase2_dir) if getattr(args, "phase2_dir", None) else None

        has_cross_data = phase1_dir or phase2_dir
        if has_cross_data:
            print(f"\n  [cross-phase] Loading cross-phase artifacts...")

            # Augment artifacts with Phase 1 standalone files
            if phase1_dir:
                for fname, key in [
                    ("pair_agreement.json",       "pair_agreement"),
                    ("phase1_events.json",         "_p1_events"),
                    ("head_fiedler_profile.json",  "head_fiedler_profile"),
                    ("hdbscan_labels.json",        "hdbscan_labels"),
                ]:
                    fpath = phase1_dir / fname
                    if fpath.exists():
                        with open(fpath) as f:
                            loaded = json.load(f)
                        if key == "_p1_events":
                            artifacts["merge_layers"]        = loaded.get("merge_layers", [])
                            artifacts["energy_violations"]   = loaded.get("energy_violations", {})
                            artifacts["energy_drop_pairs"]   = loaded.get("energy_drop_pairs", {})
                        else:
                            artifacts[key] = loaded
                        print(f"    Loaded {fname}")

                artifacts["phase1_dir"] = str(phase1_dir)

            # Augment artifacts with Phase 2 standalone files
            if phase2_dir:
                for fname, key in [
                    ("ffn_subspace.json",    "ffn_subspace"),
                    ("cross_term.json",      "cross_term_results"),
                    ("ov_per_head.json",     "ov_per_head"),
                    ("head_ov.json",         "head_ov"),
                ]:
                    fpath = phase2_dir / fname
                    if fpath.exists():
                        with open(fpath) as f:
                            artifacts[key] = json.load(f)
                        print(f"    Loaded {fname}")

                artifacts["phase2_dir"] = str(phase2_dir)

            # Cache earlier analysis results for cross-phase use
            artifacts["_violation_layer_features_result"] = results.get(
                "violation_layer_features"
            )
            artifacts["_fcc_result"]      = results.get("feature_cluster_correlation")
            artifacts["_lifetime_result"] = results.get("feature_lifetimes")

            # Run cross-phase analyses (registered, just need artifacts)
            cross_phase_names = [
                "ffn_repulsive_feature_alignment",
                "cross_term_feature_weighting",
                "induction_feature_tagging",
                "decoder_violation_projection",
                "lifetime_centroid_decomposition",
                "coactivation_at_merges",
                "cluster_identity_diff",
            ]
            from .analysis import _REGISTRY
            available = [n for n in cross_phase_names if n in _REGISTRY]
            if available:
                print(f"  [cross-phase] Running {len(available)} analyses...")
                from .analysis import run_all_analyses as _run_all
                cross_results = _run_all(
                    crosscoder, prompt_store, artifacts,
                    config={},
                    only=available,
                )
                results["cross_phase"] = cross_results

                cp_dir = out_dir / "cross_phase"
                cp_dir.mkdir(parents=True, exist_ok=True)
                for name, data in cross_results.items():
                    with open(cp_dir / f"{name}.json", "w") as f:
                        json.dump(
                            data, f, indent=2,
                            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
                        )
                print(f"  [cross-phase] Saved to {cp_dir}")

            # Pair tracking summary from steering results
            steer_results_list = results.get("_steering_results_raw", [])
            if steer_results_list:
                pt_summary = analyse_pair_tracking(steer_results_list)
                if pt_summary:
                    results.setdefault("cross_phase", {})["pair_tracking"] = pt_summary
                    with open(cp_dir / "pair_tracking.json", "w") as f:
                        json.dump(pt_summary, f, indent=2)
                    print(f"  [cross-phase] Pair tracking summary saved.")

    return results


# ---------------------------------------------------------------------------
# Plateau detection (used by _load_artifacts)
# ---------------------------------------------------------------------------

def _detect_plateau_windows(layers_data: list, min_length: int = 3) -> list:
    """
    Detect metastable plateau windows from Phase 1 per-layer metrics.

    A plateau is a maximal run of consecutive stable layers.  A layer is
    "stable" if the transition into it (from the previous layer) showed
    little change in the token geometry.  Stability is scored across
    whichever of the following metrics are present in the JSON:

      - cka / cka_to_prev       : CKA to previous layer  (stable if >= 0.95)
      - nn_stability             : fraction of tokens whose NN is unchanged
                                   (stable if >= 0.90)
      - hdbscan cluster count    : from clustering.hdbscan.n_clusters or
                                   len(set(labels) - {-1})  (stable if unchanged)
      - spectral k               : from clustering.spectral.k or .n_clusters
                                   (stable if unchanged)

    A layer is stable if >= 50% of the available metrics pass their
    threshold.  Requiring all four would be too strict given that Phase 1
    prompts vary in token count and not every metric is populated for
    every run.

    For each detected plateau of length L, the "mid-plateau" layer is
    chosen by:
      1. Trimming the first and last ceil(L/4) layers (the entry/exit
         transition zone where cluster structure is forming or dissolving).
      2. Picking the middle layer of what remains.
    """
    import math

    def _is_stable(layer_info: dict) -> bool:
        votes, total = 0, 0
        cka = layer_info.get("cka_to_prev", layer_info.get("cka"))
        if cka is not None:
            total += 1
            if cka >= 0.95:
                votes += 1
        nn = layer_info.get("nn_stability")
        if nn is not None:
            total += 1
            if nn >= 0.90:
                votes += 1
        hdb_k_prev = layer_info.get("_prev_hdbscan_k")
        hdb_k_cur  = layer_info.get("hdbscan_k",
                         layer_info.get("clustering", {}).get("hdbscan", {}).get("n_clusters"))
        if hdb_k_prev is not None and hdb_k_cur is not None:
            total += 1
            if hdb_k_prev == hdb_k_cur:
                votes += 1
        spec_k_prev = layer_info.get("_prev_spectral_k")
        spec_k_cur  = layer_info.get("spectral_k",
                          layer_info.get("clustering", {}).get("spectral", {}).get("k"))
        if spec_k_prev is not None and spec_k_cur is not None:
            total += 1
            if spec_k_prev == spec_k_cur:
                votes += 1
        if total == 0:
            return False
        return (votes / total) >= 0.5

    # Tag each layer as stable or not
    stable_flags = []
    prev_hdb_k   = None
    prev_spec_k  = None
    for info in layers_data:
        info = dict(info)
        info["_prev_hdbscan_k"] = prev_hdb_k
        info["_prev_spectral_k"] = prev_spec_k
        stable_flags.append(_is_stable(info))
        prev_hdb_k = info.get("hdbscan_k",
                        info.get("clustering", {}).get("hdbscan", {}).get("n_clusters"),
                        prev_hdb_k)
        prev_spec_k = info.get("spectral_k",
                        info.get("clustering", {}).get("spectral", {}).get("k"),
                        prev_spec_k)

    # Find maximal runs of stable layers
    plateaus = []
    i = 0
    while i < len(stable_flags):
        if stable_flags[i]:
            j = i
            while j < len(stable_flags) and stable_flags[j]:
                j += 1
            run = list(range(i, j))
            if len(run) >= min_length:
                trim = math.ceil(len(run) / 4)
                inner = run[trim: len(run) - trim] or run
                mid = inner[len(inner) // 2]
                plateaus.append({
                    "start": run[0],
                    "end":   run[-1],
                    "mid":   mid,
                    "length": len(run),
                })
            i = j
        else:
            i += 1
    return plateaus


def _load_artifacts(args, model_name: str) -> dict:
    """Load Phase 1/2 artifacts into a dict for analysis registry."""
    artifacts: dict = {}

    phase1_dir = Path(args.phase1_dir) if getattr(args, "phase1_dir", None) else None
    phase2_dir = Path(args.phase2_dir) if getattr(args, "phase2_dir", None) else None

    if phase1_dir and phase1_dir.exists():
        # Discover per-prompt run directories inside the Phase 1 output
        prompt_dirs = [d for d in phase1_dir.iterdir()
                       if d.is_dir() and model_name.replace("/", "_") in d.name]

        plateau_layers: dict = {}
        hdbscan_labels: dict = {}
        merge_layers:   dict = {}
        energy_violations: dict = {}
        energy_drop_pairs: dict = {}
        pair_agreement:   dict = {}
        head_fiedler_profile: dict = {}

        for pd in prompt_dirs:
            # Infer prompt key from directory name
            stem = pd.name
            for part in stem.split("_"):
                prompt_key = part
                break  # first non-model segment used as key approximation

            # Per-layer metrics → plateau detection
            layer_file = pd / "layer_metrics.json"
            if layer_file.exists():
                with open(layer_file) as f:
                    layer_data = json.load(f)
                plats = _detect_plateau_windows(layer_data)
                plateau_layers[prompt_key] = [p["mid"] for p in plats]

            # HDBSCAN cluster labels
            labels_file = pd / "hdbscan_labels.json"
            if labels_file.exists():
                with open(labels_file) as f:
                    hdbscan_labels[prompt_key] = json.load(f)

            # Merge / violation events
            events_file = pd / "events.json"
            if events_file.exists():
                with open(events_file) as f:
                    ev = json.load(f)
                merge_layers[prompt_key]      = ev.get("merge_layers", [])
                energy_violations[prompt_key] = ev.get("energy_violations", {})
                energy_drop_pairs[prompt_key] = ev.get("energy_drop_pairs", {})

        # Global Phase 1 files
        for fname, key, target in [
            ("pair_agreement.json",      "pair_agreement",       pair_agreement),
            ("head_fiedler_profile.json", "head_fiedler_profile", head_fiedler_profile),
        ]:
            fpath = phase1_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    target.update(json.load(f))

        artifacts.update({
            "plateau_layers":       plateau_layers,
            "hdbscan_labels":       hdbscan_labels,
            "merge_layers":         merge_layers,
            "energy_violations":    energy_violations,
            "energy_drop_pairs":    energy_drop_pairs,
            "pair_agreement":       pair_agreement,
            "head_fiedler_profile": head_fiedler_profile,
            "phase1_dir":           str(phase1_dir),
        })
        print(f"    Phase 1 artifacts loaded from {phase1_dir}")

    if phase2_dir and phase2_dir.exists():
        # OV projectors: low-rank (top-64 eigenvectors by |eigenvalue|)
        stem = model_name.replace("/", "_").replace("-", "_")
        projector_file = phase2_dir / f"ov_projectors_{stem}.npz"
        if projector_file.exists():
            data = np.load(projector_file)
            k_top = getattr(args, "k_top", 64)
            ov_projectors: dict = {}
            for key in data.files:
                mat = data[key]  # (d, d) or (d, k)
                if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
                    # Square: eigendecompose and keep top-k
                    eigvals, eigvecs = np.linalg.eigh(mat)
                    idx = np.argsort(np.abs(eigvals))[::-1][:k_top]
                    ov_projectors[key] = eigvecs[:, idx]
                else:
                    ov_projectors[key] = mat
            artifacts["ov_projectors"] = ov_projectors
            print(f"    OV projectors loaded (k={k_top}): {list(ov_projectors.keys())[:4]}…")

        for fname, key in [
            ("ffn_subspace.json",  "ffn_subspace"),
            ("cross_term.json",    "cross_term_results"),
            ("ov_per_head.json",   "ov_per_head"),
            ("head_ov.json",       "head_ov"),
        ]:
            fpath = phase2_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    artifacts[key] = json.load(f)

        artifacts["phase2_dir"] = str(phase2_dir)
        print(f"    Phase 2 artifacts loaded from {phase2_dir}")

    return artifacts


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def _print_summary(results: dict, model_name: str):
    print(f"\n  --- Analysis summary: {model_name} ---")

    lt = results.get("feature_lifetimes", {})
    if lt and "error" not in lt:
        n_short = lt.get("n_short_lived", "?")
        n_long  = lt.get("n_long_lived", "?")
        n_dead  = lt.get("n_dead", "?")
        print(f"  Feature lifetimes: short={n_short}  long={n_long}  dead={n_dead}")

    va = results.get("v_subspace_alignment", {})
    if va and "error" not in va:
        rho = va.get("spearman_rho", float("nan"))
        p   = va.get("spearman_p",   float("nan"))
        print(f"  V-subspace Spearman ρ={rho:.3f}  p={p:.4f}")

    fcc = results.get("feature_cluster_correlation", {})
    if fcc and "error" not in fcc:
        n_fcc = sum(
            len(v) for pk in fcc.values() if isinstance(pk, dict)
            for v in pk.values() if isinstance(v, dict)
        )
        print(f"  FCC entries: {n_fcc}")
        for pk, layer_dict in fcc.items():
            if not isinstance(layer_dict, dict):
                continue
            for layer_key, info in layer_dict.items():
                if not isinstance(info, dict):
                    continue
                rho   = info.get("rho", float("nan"))
                pval  = info.get("pval", float("nan"))
                n     = info.get("n", "?")
                n_hdb = info.get("n_hdbscan_clusters", "?")
                psizes = info.get("partition_sizes", [])
                shared  = info.get("shared_top", [])
                spec_ex = info.get("spectral_exclusive", [])
                sub_ex  = info.get("subcluster_exclusive", [])
                print(f"    {pk}  layer={layer_key}  "
                      f"ρ={rho:.3f}  p={pval:.3f}  "
                      f"n={n}  hdb_k={n_hdb}  partitions={psizes}")
                print(f"      shared={len(shared)}  "
                      f"spectral_only={len(spec_ex)}  "
                      f"subcluster_only={len(sub_ex)}")
                for label, pop in [("shared_top", shared),
                                    ("spectral_excl", spec_ex),
                                    ("subclust_excl", sub_ex)]:
                    if pop:
                        f0 = pop[0]
                        print(f"      {label:16s}: f{f0['feature']}  "
                              f"F_spec={f0['f_spectral']:.2f}  "
                              f"F_within={f0['f_within']:.2f}  "
                              f"({f0['lifetime_class']})")

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
    group.add_argument("--gpt2-only",   action="store_true", help="GPT-2-large only")

    # Stage skipping
    parser.add_argument("--skip-cache",   action="store_true")
    parser.add_argument("--skip-train",   action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--skip-steer",   action="store_true",
                        help="Skip steering experiment (stage 4)")
    parser.add_argument("--skip-cross-phase", action="store_true",
                        help="Skip cross-phase bridge analyses (stage 5)")
    parser.add_argument("--force-cache", action="store_true",
                        help="Re-extract even if cache exists")
    parser.add_argument("--force-train", action="store_true",
                        help="Retrain even if checkpoint exists")
    parser.add_argument("--steering-n-features", type=int, default=5,
                        help="Top features to steer per (prompt, layer) combo")
    parser.add_argument("--steering-alpha", type=float, default=1.0,
                        help="Multiplier on the auto-scaled perturbation α")

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
