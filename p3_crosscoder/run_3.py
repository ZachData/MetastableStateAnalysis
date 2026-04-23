"""
run_3.py — p3_crosscoder entry point.

Trains a sparse crosscoder on residual stream activations across layers,
then runs a battery of analyses that test whether crosscoder features
track metastable cluster structure identified in Phases 1–2.

Default behavior (no arguments): runs the full pipeline for both models.
Flags narrow scope:

    python -m phase3.run_3                         # everything, both models
    python -m phase3.run_3 --albert-only           # ALBERT-xlarge only
    python -m phase3.run_3 --gpt2-only             # GPT-2-large only
    python -m phase3.run_3 --skip-cache            # skip extraction (use existing)
    python -m phase3.run_3 --skip-train            # skip training (use existing)
    python -m phase3.run_3 --skip-analyze          # skip analysis
    python -m phase3.run_3 --data-source tinystories --n-texts 10000
    python -m phase3.run_3 --phase1-dir results/... --phase2-dir results/...
    python -m phase3.run_3 --skip-cross-phase      # skip Stage 5 bridge analyses

The pipeline auto-detects existing cached activations and trained
checkpoints.  If they exist, it skips that stage.  Use --force-cache
or --force-train to override.

Output layout
-------------
results/p3_crosscoder/{model}_{ts}/
├── summary.txt                    # LLM-readable digest of all analyses
├── crosscoder_config.json
├── analyses/
│   ├── index.json                 # {name: {file, has_error, has_npz}}
│   ├── feature_lifetimes.json
│   ├── feature_lifetimes.npz      # (if large arrays offloaded)
│   ├── v_subspace_alignment.json
│   ├── cluster_identity.json
│   ├── violation_layer_features.json
│   ├── multilayer_fraction.json
│   ├── positional_control.json
│   ├── feature_cluster_correlation.json
│   └── inspect_top_features.json + .txt
├── cross_phase/
│   ├── index.json
│   ├── coactivation_at_merges.json
│   └── ...
└── steering/
    ├── steering_results.json
    └── steering_summary.json
"""

import argparse
import json
import re
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
from .analysis import (
    run_all_analyses, save_results,
    _REGISTRY, _SUMMARY_REGISTRY,
    summarize_steering,
    _save_json,
)
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
    return Path("results") / "p3_crosscoder" / f"{model_name.replace('/', '_')}_{ts}"


# ---------------------------------------------------------------------------
# Summary composition
# ---------------------------------------------------------------------------

def _compose_summary(
    model_name: str,
    out_dir: Path,
    cc_config: dict,
    analysis_blocks: list,
    cross_blocks: list,
    steer_block: str,
) -> str:
    """
    Compose the full summary.txt content from per-stage summary blocks.

    Parameters
    ----------
    analysis_blocks  : list of (name, text) from run_all_analyses
    cross_blocks     : list of (name, text) from cross-phase run_all_analyses
    steer_block      : text from summarize_steering (or empty string)
    """
    sep = "=" * 68
    n_feat = cc_config.get("n_features", "?")
    k      = cc_config.get("k", "?")
    steps  = cc_config.get("total_steps", "?")

    lines = [
        sep,
        f"P3_CROSSCODER SUMMARY — {model_name}",
        f"run_dir: {out_dir}",
        f"crosscoder: n_features={n_feat}, k={k}, steps={steps}",
        sep,
        "",
    ]

    for name, block in analysis_blocks:
        lines.append(f"[analyses/{name}]")
        lines.append(block)
        lines.append("")

    for name, block in cross_blocks:
        lines.append(f"[cross_phase/{name}]")
        lines.append(block)
        lines.append("")

    if steer_block:
        lines.append("[steering]")
        lines.append(steer_block)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline per model
# ---------------------------------------------------------------------------

def run_model(model_name: str, args) -> dict:
    """Full pipeline for one model: cache → train → analyze."""
    print(f"\n{'='*70}")
    print(f"  p3_crosscoder: {model_name}")
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

        texts = load_texts(
            source=args.data_source,
            n_texts=args.n_texts,
            data_dir=args.data_dir,
        )

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

        print("    Caching evaluation prompts...")
        prompt_store = PromptActivationStore()
        for key, text in PROMPTS.items():
            if key == "repeated_tokens":
                continue
            # extract_activations returns a list of arrays, one per input text.
            # For a single text we take element [0]: (n_tokens, n_layers, d_model).
            # add() signature: (prompt_key, activations, tokens, layer_indices)
            results = extract_activations(model, tokenizer, [text], config, device=DEVICE)
            arr = results[0]
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=config.max_seq_len)
            token_ids  = enc["input_ids"][0].tolist()
            token_strs = tokenizer.convert_ids_to_tokens(token_ids)
            prompt_store.add(key, arr, token_strs, config.layer_indices)
        prompt_store.save(cache_dir / "eval_prompts")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage 2: Train crosscoder
    # ------------------------------------------------------------------
    # Initialise cc_config here so it is always defined even when --skip-train
    # is passed and the training branch is never entered (Bug F).
    cc_config: dict = {}
    final_dir = ckpt_dir / "final"

    if args.skip_train:
        print("\n  [train] Skipped (--skip-train)")
    elif is_trained(ckpt_dir) and not args.force_train:
        print(f"\n  [train] Found existing checkpoint: {final_dir}")
    else:
        print(f"\n  [train] Training crosscoder...")
        # FEATURE_PRESETS maps model_name → int (n_features only).
        # d_model comes from the model config constants, not FEATURE_PRESETS.
        _D_MODEL = {"albert-xlarge-v2": 2048, "gpt2-large": 1280}
        d_model    = _D_MODEL[model_name]
        n_features = args.n_features or FEATURE_PRESETS[model_name]
        n_layers   = len(layers)

        # CrosscoderConfig takes d_model + n_layers as separate fields;
        # d_input is a derived @property, not a constructor argument.
        cc_cfg = CrosscoderConfig(
            d_model=d_model,
            n_layers=n_layers,
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
        vram_est_gb = n_params * 4 * 3 / 1e9
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

    # Establish output directory before running analyses so files stream
    # into it immediately rather than accumulating in memory first.
    out_dir = _results_dir(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_blocks, analysis_index = run_all_analyses(
        crosscoder, prompt_store, artifacts,
        config={"lifetime_threshold_frac": 0.1, "multilayer_min_layers": 3},
        out_dir=out_dir,
    )

    # Save inspect_top_features text report as a standalone .txt for quick
    # human inspection without parsing JSON.
    itf_path = out_dir / "analyses" / "inspect_top_features.json"
    if itf_path.exists():
        try:
            with open(itf_path) as f:
                itf = json.load(f)
            text_report = itf.get("text_report", "")
            if text_report:
                txt_path = out_dir / "analyses" / "inspect_top_features.txt"
                with open(txt_path, "w") as f:
                    f.write(text_report)
                print(f"  Feature inspection report: {txt_path}")
        except Exception:
            pass

    # Save crosscoder config for reference
    with open(final_dir / "config.json") as f:
        cc_config = json.load(f)
    with open(out_dir / "crosscoder_config.json", "w") as f:
        json.dump(cc_config, f, indent=2)

    _print_summary(analysis_blocks, model_name)

    # Accumulate raw results dict for Stage 4/5 (backward compat; not written to disk)
    results: dict = {}

    # ------------------------------------------------------------------
    # Stage 4: Steering experiment
    # ------------------------------------------------------------------
    steer_block = ""
    steer_results_raw = []

    if not getattr(args, "skip_steer", False):
        # Re-load the feature_lifetimes and FCC results we just wrote
        fcc_path = out_dir / "analyses" / "feature_cluster_correlation.json"
        lt_path  = out_dir / "analyses" / "feature_lifetimes.json"
        fcc: dict = {}
        lt_arr:  list = []

        if fcc_path.exists():
            with open(fcc_path) as f:
                fcc = json.load(f)
        if lt_path.exists():
            with open(lt_path) as f:
                lt_data = json.load(f)
            lt_arr = lt_data.get("lifetime_class", [])

        plateau_layers = artifacts.get("plateau_layers")
        layers_        = artifacts.get("layer_indices", LAYER_PRESETS[model_name])

        if not plateau_layers:
            print("\n  [steer] Skipping: no plateau_layers artifact "
                  "(run Phase 1 for this model first)")
        elif "error" in fcc:
            print(f"\n  [steer] Skipping: feature_cluster_correlation failed: {fcc.get('error')}")
        else:
            print(f"\n  [steer] Loading model for steering...")
            from core.models import load_model as _load_model
            steer_model, steer_tokenizer = _load_model(model_name)
            steer_model = steer_model.to(DEVICE)
            steer_model.eval()

            is_albert = "albert" in model_name.lower()

            steer_cfg = {
                "steering_n_features":            getattr(args, "steering_n_features", 5),
                "steering_alpha_multiplier":      getattr(args, "steering_alpha", 1.0),
                "steering_merge_threshold_sigma":  2.0,
                "plateau_min_cluster_size":        3,
            }

            print(f"  [steer] Running steering experiments...")
            steer_results_raw = run_steering(
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

            steer_summary = summarise_steering(steer_results_raw)

            steer_dir = out_dir / "steering"
            steer_dir.mkdir(exist_ok=True)
            save_steering_results(
                steer_results_raw, steer_summary,
                steer_dir / "steering_results.json",
            )
            # Also write summary sidecar
            _save_json(steer_summary, steer_dir / "steering_summary.json")

            steer_block = summarize_steering(steer_summary)

            for line in steer_summary.get("text_summary", "").splitlines():
                print("  " + line)

            del steer_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage 5: Cross-phase bridge analyses
    # ------------------------------------------------------------------
    cross_blocks: list = []

    if not getattr(args, "skip_cross_phase", False):
        phase1_dir = Path(args.phase1_dir) if getattr(args, "phase1_dir", None) else None
        phase2_dir = Path(args.phase2_dir) if getattr(args, "phase2_dir", None) else None

        has_cross_data = phase1_dir or phase2_dir
        if has_cross_data:
            print(f"\n  [cross-phase] Loading cross-phase artifacts...")

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
                            artifacts["merge_layers"]      = loaded.get("merge_layers", [])
                            artifacts["energy_violations"] = loaded.get("energy_violations", {})
                            artifacts["energy_drop_pairs"] = loaded.get("energy_drop_pairs", {})
                        else:
                            artifacts[key] = loaded
                        print(f"    Loaded {fname}")
                artifacts["phase1_dir"] = str(phase1_dir)

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

            # Inject earlier analysis results for cross-phase use — read from
            # the per-analysis files we just wrote rather than keeping in memory.
            for result_key, fname in [
                ("_violation_layer_features_result", "violation_layer_features.json"),
                ("_fcc_result",                      "feature_cluster_correlation.json"),
                ("_lifetime_result",                 "feature_lifetimes.json"),
            ]:
                fpath = out_dir / "analyses" / fname
                if fpath.exists():
                    with open(fpath) as f:
                        artifacts[result_key] = json.load(f)

            cross_phase_names = [
                "ffn_repulsive_feature_alignment",
                "cross_term_feature_weighting",
                "induction_feature_tagging",
                "decoder_violation_projection",
                "lifetime_centroid_decomposition",
                "coactivation_at_merges",
                "cluster_identity_diff",
            ]
            available = [n for n in cross_phase_names if n in _REGISTRY]

            if available:
                print(f"  [cross-phase] Running {len(available)} analyses...")
                cp_dir = out_dir / "cross_phase"
                cp_dir.mkdir(parents=True, exist_ok=True)

                # Use the same streaming mechanism as Stage 3, writing into
                # cross_phase/ instead of analyses/.
                cross_blocks, cross_index = run_all_analyses(
                    crosscoder, prompt_store, artifacts,
                    config={},
                    only=available,
                    out_dir=out_dir / "_cp_staging",   # temp, we'll move below
                )

                # Move staged files into cross_phase/
                staging = out_dir / "_cp_staging" / "analyses"
                if staging.exists():
                    for f in staging.iterdir():
                        f.rename(cp_dir / f.name)
                    # Must remove the staging subdirectory before its parent.
                    staging.rmdir()
                    staging.parent.rmdir()

                _save_json(cross_index, cp_dir / "index.json")
                print(f"  [cross-phase] Saved to {cp_dir}")

            # Pair tracking summary from steering results
            if steer_results_raw:
                pt_summary = analyse_pair_tracking(steer_results_raw)
                if pt_summary:
                    _save_json(pt_summary, out_dir / "cross_phase" / "pair_tracking.json")
                    cross_blocks.append(("pair_tracking", str(pt_summary)[:500]))
                    print(f"  [cross-phase] Pair tracking summary saved.")

    # ------------------------------------------------------------------
    # Stage 6: Compose summary.txt
    # ------------------------------------------------------------------
    summary_txt = _compose_summary(
        model_name=model_name,
        out_dir=out_dir,
        cc_config=cc_config,
        analysis_blocks=analysis_blocks,
        cross_blocks=cross_blocks,
        steer_block=steer_block,
    )
    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary_txt)
    print(f"\n  summary.txt written to {out_dir / 'summary.txt'}")
    print(f"  Output directory: {out_dir}")

    return results


# ---------------------------------------------------------------------------
# Plateau detection (used by _load_artifacts and by phase4)
# ---------------------------------------------------------------------------

def _detect_plateau_windows(layers_data: list, min_length: int = 3) -> list:
    """
    Detect metastable plateau windows from Phase 1 per-layer metrics.

    A plateau is a maximal run of consecutive stable layers.  A layer is
    "stable" if the transition into it showed little change in token geometry.
    Stability is scored across whichever of the following metrics are present:

      - cka / cka_to_prev  (stable if >= 0.95)
      - nn_stability        (stable if >= 0.90)
      - hdbscan cluster count (stable if unchanged)
      - spectral k          (stable if unchanged)

    A layer is stable if >= 50% of available metrics pass their threshold.
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
        hdb_k_cur  = layer_info.get(
            "hdbscan_k",
            layer_info.get("clustering", {}).get("hdbscan", {}).get("n_clusters"),
        )
        if hdb_k_prev is not None and hdb_k_cur is not None:
            total += 1
            if hdb_k_prev == hdb_k_cur:
                votes += 1
        spec_k_prev = layer_info.get("_prev_spectral_k")
        spec_k_cur  = layer_info.get(
            "spectral_k",
            layer_info.get("clustering", {}).get("spectral", {}).get("k"),
        )
        if spec_k_prev is not None and spec_k_cur is not None:
            total += 1
            if spec_k_prev == spec_k_cur:
                votes += 1
        if total == 0:
            return False
        return (votes / total) >= 0.5

    stable_flags = []
    prev_hdb_k   = None
    prev_spec_k  = None
    for info in layers_data:
        info = dict(info)
        info["_prev_hdbscan_k"] = prev_hdb_k
        info["_prev_spectral_k"] = prev_spec_k
        stable_flags.append(_is_stable(info))
        prev_hdb_k  = info.get(
            "hdbscan_k",
            info.get("clustering", {}).get("hdbscan", {}).get("n_clusters"),
            prev_hdb_k,
        )
        prev_spec_k = info.get(
            "spectral_k",
            info.get("clustering", {}).get("spectral", {}).get("k"),
            prev_spec_k,
        )

    plateaus = []
    i = 0
    while i < len(stable_flags):
        if stable_flags[i]:
            j = i
            while j < len(stable_flags) and stable_flags[j]:
                j += 1
            run = list(range(i, j))
            if len(run) >= min_length:
                trim  = math.ceil(len(run) / 4)
                inner = run[trim: len(run) - trim] or run
                mid   = inner[len(inner) // 2]
                plateaus.append({
                    "start": run[0], "end": run[-1],
                    "mid": mid, "length": len(run),
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
        prompt_dirs = [
            d for d in phase1_dir.iterdir()
            if d.is_dir() and model_name.replace("/", "_") in d.name
        ]

        plateau_layers:     dict = {}
        hdbscan_labels:     dict = {}
        merge_layers:       dict = {}
        energy_violations:  dict = {}
        energy_drop_pairs:  dict = {}
        pair_agreement:     dict = {}
        head_fiedler_profile: dict = {}

        for pd in prompt_dirs:
            stem = pd.name
            # Directory names are like: albert_xlarge_v2_100iter_short_heterogeneous
            # Strip the model+iteration prefix to recover the prompt key.
            prompt_key = re.sub(r"^.*?\d+iter_", "", stem)
            if not prompt_key or prompt_key == stem:
                # Fallback: take everything after the last model-stem segment
                parts = stem.split("_")
                prompt_key = "_".join(parts[3:]) if len(parts) > 3 else parts[0]

            layer_file = pd / "layer_metrics.json"
            if layer_file.exists():
                with open(layer_file) as f:
                    layer_data = json.load(f)
                plats = _detect_plateau_windows(layer_data)
                plateau_layers[prompt_key] = [p["mid"] for p in plats]

            labels_file = pd / "hdbscan_labels.json"
            if labels_file.exists():
                with open(labels_file) as f:
                    hdbscan_labels[prompt_key] = json.load(f)

            events_file = pd / "events.json"
            if events_file.exists():
                with open(events_file) as f:
                    ev = json.load(f)
                merge_layers[prompt_key]      = ev.get("merge_layers", [])
                energy_violations[prompt_key] = ev.get("energy_violations", {})
                energy_drop_pairs[prompt_key] = ev.get("energy_drop_pairs", {})

        for fname, key, target in [
            ("pair_agreement.json",       "pair_agreement",       pair_agreement),
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
        stem = model_name.replace("/", "_").replace("-", "_")
        projector_file = phase2_dir / f"ov_projectors_{stem}.npz"
        if projector_file.exists():
            data = np.load(projector_file)
            k_top = getattr(args, "k_top", 64)
            ov_projectors: dict = {}
            for key in data.files:
                mat = data[key]
                if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
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

def _print_summary(analysis_blocks: list, model_name: str):
    print(f"\n  --- Analysis summary: {model_name} ---")
    for name, block in analysis_blocks:
        first_line = block.splitlines()[0] if block else ""
        print(f"  [{name}] {first_line}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="p3_crosscoder: Crosscoder Training on Metastable Dynamics.\n\n"
                    "Run with no arguments for the full pipeline on both models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--albert-only", action="store_true", help="ALBERT-xlarge only")
    group.add_argument("--gpt2-only",   action="store_true", help="GPT-2-large only")

    parser.add_argument("--skip-cache",        action="store_true")
    parser.add_argument("--skip-train",        action="store_true")
    parser.add_argument("--skip-analyze",      action="store_true")
    parser.add_argument("--skip-steer",        action="store_true",
                        help="Skip steering experiment (stage 4)")
    parser.add_argument("--skip-cross-phase",  action="store_true",
                        help="Skip cross-phase bridge analyses (stage 5)")
    parser.add_argument("--force-cache",       action="store_true")
    parser.add_argument("--force-train",       action="store_true")
    parser.add_argument("--steering-n-features", type=int, default=5)
    parser.add_argument("--steering-alpha",    type=float, default=1.0)

    parser.add_argument("--data-source", type=str, default="c4",
                        choices=["c4", "tinystories", "local"])
    parser.add_argument("--data-dir",    type=str, default=None)
    parser.add_argument("--n-texts",     type=int, default=50_000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=8)

    parser.add_argument("--n-features",  type=int, default=None)
    parser.add_argument("--activation",  type=str, default="batch_topk",
                        choices=["topk", "batch_topk"])
    parser.add_argument("--k",           type=int, default=64)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--warmup-steps",        type=int, default=1000)
    parser.add_argument("--total-steps",         type=int, default=100_000)
    parser.add_argument("--train-batch-size",    type=int, default=512)
    parser.add_argument("--grad-accum-steps",    type=int, default=4)
    parser.add_argument("--log-interval",        type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)

    parser.add_argument("--phase1-dir", type=str, default=None)
    parser.add_argument("--phase2-dir", type=str, default=None)

    args = parser.parse_args()

    if args.albert_only:
        models = ["albert-xlarge-v2"]
    elif args.gpt2_only:
        models = ["gpt2-large"]
    else:
        models = SUPPORTED_MODELS

    print(f"p3_crosscoder: Crosscoder Training on Metastable Dynamics")
    print(f"  Models: {models}")
    print(f"  Data:   {args.data_source} ({args.n_texts} texts)")
    print(f"  Device: {DEVICE}")

    all_results = {}
    for model_name in models:
        all_results[model_name] = run_model(model_name, args)

    print(f"\np3_crosscoder complete.")


if __name__ == "__main__":
    main()
