"""
run.py — Phase 4 CLI entry point.

Usage:
    python -m phase4.run --albert-only \
        --phase1-dir results/phase1_... \
        --phase2-dir results/phase2_... \
        --phase3-dir results/phase3_...

    python -m phase4.run --albert-only --skip-track3   # skip low-rank AE
    python -m phase4.run --albert-only --tracks 1 2    # explicit track selection

The pipeline:
  1. Load Phase 1 artifacts (HDBSCAN labels, merge layers, plateau windows)
  2. Load Phase 2 artifacts (V projectors)
  3. Load Phase 3 artifacts (crosscoder checkpoint, prompt activations)
  4. Run Track 1: crosscoder activation pattern analysis
  5. Run Track 2: direct geometric methods
  6. Run Track 3: low-rank autoencoder (optional)
  7. Cross-track comparison and verdict
  8. Save outputs for Phases 5/6
"""

import argparse
import json
import sys
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

from core.config import DEVICE, PROMPTS

from phase3.crosscoder import Crosscoder
from phase3.data import PromptActivationStore, ActivationBuffer
from phase3.training import load_trained_crosscoder
from phase3.run import _detect_plateau_windows

from .activation_trajectories import (
    extract_activation_trajectories,
    detect_feature_plateaus,
    feature_cluster_mi,
    plateau_alignment,
    merge_feature_dynamics,
)
from .chorus import analyze_chorus_at_layer, sweep_thresholds
from .geometric import (
    lda_stability_across_layers,
    pca_on_deltas,
    probe_accuracy_trajectory,
    probe_v_alignment,
    extract_per_layer_activations,
    build_labels_per_layer,
)
from .low_rank_ae import (
    LowRankAEConfig, LowRankAE, LRAETrainingConfig,
    train_low_rank_ae, load_low_rank_ae,
    bottleneck_v_alignment, compare_reconstruction,
    ActivationBufferAdapter,
)
from .analysis import (
    cross_track_agreement,
    build_phase4_verdict,
    save_phase4_outputs,
)


# ---------------------------------------------------------------------------
# Path conventions
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "albert-xlarge-v2": {"d_model": 2048, "stem": "albert_xlarge_v2"},
    "gpt2-large":       {"d_model": 1280, "stem": "gpt2_large"},
}

def _results_dir(model_name: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = model_name.replace("/", "_").replace("-", "_")
    return Path("results") / "phase4" / f"{stem}_{ts}"


def _cache_dir(model_name: str) -> Path:
    return Path("activation_cache") / model_name.replace("/", "_").replace("-", "_")


def _checkpoint_dir(model_name: str) -> Path:
    return Path("checkpoints") / model_name.replace("/", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_phase1(phase1_dir: Path, model_stem: str) -> dict:
    """Load Phase 1 artifacts: HDBSCAN labels, merge layers, plateau windows."""
    artifacts = {}

    # Find matching subdirectory
    candidates = [d for d in phase1_dir.iterdir()
                  if d.is_dir() and model_stem in d.name]
    if not candidates:
        print(f"  [phase1] No directories matching '{model_stem}' in {phase1_dir}")
        return artifacts

    for run_dir in candidates:
        prompt_name = run_dir.name.split("iter_", 1)[-1] if "iter_" in run_dir.name else run_dir.name

        # Metrics for plateau detection
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            layers_data = metrics.get("per_layer", metrics.get("layers", []))
            if layers_data:
                plateaus = _detect_plateau_windows(layers_data)
                artifacts.setdefault("plateau_windows", {})[prompt_name] = plateaus
                print(f"    {prompt_name}: {len(plateaus)} plateaus detected")

        # HDBSCAN labels
        for fname in ["hdbscan_labels.json", "cluster_labels.json"]:
            fpath = run_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    labels = json.load(f)
                artifacts.setdefault("hdbscan_labels", {})[prompt_name] = labels
                print(f"    Loaded {fname} for {prompt_name}")
                break

        # Merge layers / events
        for fname in ["phase1_events.json", "events.json"]:
            fpath = run_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    events = json.load(f)
                merge_layers = events.get("merge_layers", [])
                if isinstance(merge_layers, dict):
                    flat = set()
                    for v in merge_layers.values():
                        if isinstance(v, list):
                            flat.update(v)
                    merge_layers = sorted(flat)
                artifacts.setdefault("merge_layers", {})[prompt_name] = merge_layers
                print(f"    {prompt_name}: {len(merge_layers)} merge layers")
                break

    return artifacts


def _load_phase2(phase2_dir: Path, model_stem: str, k_top: int = 64) -> dict:
    """Load Phase 2 V eigensubspace projectors."""
    artifacts = {}

    proj_path = phase2_dir / f"ov_projectors_{model_stem}.npz"
    if not proj_path.exists():
        # Try alternative naming
        for p in phase2_dir.glob(f"*projector*{model_stem}*"):
            proj_path = p
            break

    if proj_path.exists():
        data = np.load(proj_path)
        # Build low-rank projectors from top-k eigenvectors
        eigvecs = data.get("eigenvectors", data.get("U"))
        eigvals = data.get("eigenvalues", data.get("S"))

        if eigvecs is not None and eigvals is not None:
            idx = np.argsort(np.abs(eigvals))[::-1][:k_top]
            top_vecs = eigvecs[:, idx]
            top_vals = eigvals[idx]

            pos_mask = top_vals > 0
            neg_mask = top_vals < 0

            if pos_mask.any():
                U_att = top_vecs[:, pos_mask]
                artifacts["attractive"] = U_att @ U_att.T
            if neg_mask.any():
                U_rep = top_vecs[:, neg_mask]
                artifacts["repulsive"] = U_rep @ U_rep.T

            print(f"    V projectors: {pos_mask.sum()} attractive, "
                  f"{neg_mask.sum()} repulsive directions (k={k_top})")
        else:
            print(f"    Warning: projector file exists but missing keys")
    else:
        print(f"    No V projectors found at {proj_path}")

    return artifacts


def _load_phase3(phase3_dir: Path, model_stem: str, device: str) -> dict:
    """Load Phase 3 crosscoder and prompt store."""
    artifacts = {}

    ckpt_dir = _checkpoint_dir(model_stem.replace("_", "-"))
    if not ckpt_dir.exists():
        # Try phase3_dir
        ckpt_dir = phase3_dir / "checkpoints" / model_stem

    crosscoder, layer_indices = None, None
    for candidate in [ckpt_dir / "final", ckpt_dir]:
        if (candidate / "config.json").exists():
            crosscoder = load_trained_crosscoder(candidate, device=device)
            # Load layer indices from config
            cfg_path = candidate / "config.json"
            with open(cfg_path) as f:
                cfg = json.load(f)
            # Layer indices stored in extraction config or separately
            layer_indices = cfg.get("layer_indices")
            print(f"    Crosscoder loaded from {candidate}")
            print(f"    Features: {crosscoder.cfg.n_features}, "
                  f"Layers: {crosscoder.cfg.n_layers}")
            break

    if crosscoder is None:
        print(f"    Warning: no crosscoder checkpoint found")
    artifacts["crosscoder"] = crosscoder

    # Layer indices fallback
    if layer_indices is None:
        from phase3.data import LAYER_PRESETS
        model_name = model_stem.replace("_", "-")
        if model_name in LAYER_PRESETS:
            layer_indices = LAYER_PRESETS[model_name]
    artifacts["layer_indices"] = layer_indices or []

    # Prompt activation store
    cache_dir = _cache_dir(model_stem.replace("_", "-"))
    prompt_store = PromptActivationStore()
    eval_dir = cache_dir / "eval_prompts"
    if eval_dir.exists():
        prompt_store.load(eval_dir)
        print(f"    Prompt store: {len(prompt_store.keys())} prompts")
    else:
        print(f"    Warning: no eval prompts at {eval_dir}")
    artifacts["prompt_store"] = prompt_store

    return artifacts


# ---------------------------------------------------------------------------
# Track runners
# ---------------------------------------------------------------------------

def run_track1(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    layer_indices: list,
    hdbscan_labels: dict,
    merge_layers: dict,
    cluster_plateaus: list,
    config: dict,
) -> dict:
    """Run all Track 1 analyses."""
    print("\n  === Track 1: Crosscoder activation patterns ===")
    results = {}

    # 1. Extract trajectories
    print("  [T1] Extracting activation trajectories...")
    trajs = extract_activation_trajectories(
        crosscoder, prompt_store, layer_indices
    )
    results["n_prompts"] = len(trajs)

    # 2. Feature plateaus
    print("  [T1] Detecting feature plateaus...")
    all_plateaus = {}
    for pk, traj in trajs.items():
        fp = detect_feature_plateaus(
            traj,
            var_threshold=config.get("var_threshold", 0.01),
            min_plateau_len=config.get("min_plateau_len", 3),
        )
        all_plateaus[pk] = fp
        s = fp["summary"]
        print(f"    {pk}: {s['n_features_with_plateaus']}/{s['n_features_total']} "
              f"features have plateaus (mean length {s['mean_plateau_length']:.1f})")
    results["feature_plateaus"] = all_plateaus

    # 3. Feature–cluster MI
    print("  [T1] Computing feature–cluster mutual information...")
    mi_results = {}
    for pk, traj in trajs.items():
        pk_labels = hdbscan_labels.get(pk, {})
        if pk_labels:
            mi = feature_cluster_mi(traj, pk_labels, layer_indices)
            mi_results[pk] = mi
            for lk, lr in mi.items():
                if isinstance(lr, dict) and "max_nmi" in lr:
                    print(f"    {pk}/{lk}: max NMI={lr['max_nmi']:.3f}, "
                          f"mean NMI={lr['mean_nmi']:.4f}")
    results["mi_results"] = mi_results

    # 4. Chorus analysis
    print("  [T1] Running chorus analysis...")
    chorus_results = {}
    for pk, traj in trajs.items():
        pk_labels = hdbscan_labels.get(pk, {})
        pk_chorus = {}
        for layer_key, labels in pk_labels.items():
            try:
                model_layer = int(layer_key.replace("layer_", ""))
            except (ValueError, AttributeError):
                continue
            if model_layer not in layer_indices:
                continue
            cc_idx = layer_indices.index(model_layer)
            labels_arr = np.array(labels)

            chorus = analyze_chorus_at_layer(
                traj, labels_arr, cc_idx,
                coact_threshold=config.get("coact_threshold", 0.3),
            )
            pk_chorus[layer_key] = chorus

            ari_val = chorus.get("ari", {}).get("ari", 0)
            purity_val = chorus.get("purity", {}).get("summary", {}).get("mean_purity", 0)
            print(f"    {pk}/{layer_key}: {chorus['n_cliques']} cliques, "
                  f"ARI={ari_val:.3f}, purity={purity_val:.3f}")

        chorus_results[pk] = pk_chorus
    results["chorus_results"] = chorus_results

    # Aggregate chorus stats
    all_aris = []
    all_purities = []
    for pk_data in chorus_results.values():
        for layer_data in pk_data.values():
            ari = layer_data.get("ari", {}).get("ari")
            if ari is not None:
                all_aris.append(ari)
            pur = layer_data.get("purity", {}).get("summary", {}).get("mean_purity")
            if pur is not None:
                all_purities.append(pur)
    results["chorus_summary"] = {
        "mean_ari": float(np.mean(all_aris)) if all_aris else 0.0,
        "max_ari": float(np.max(all_aris)) if all_aris else 0.0,
        "mean_purity": float(np.mean(all_purities)) if all_purities else 0.0,
    }

    # MI summary
    all_nmis = []
    for pk_mi in mi_results.values():
        for layer_data in pk_mi.values():
            if isinstance(layer_data, dict):
                nmi = layer_data.get("max_nmi")
                if nmi is not None:
                    all_nmis.append(nmi)
    results["mi_summary"] = {
        "max_nmi": float(np.max(all_nmis)) if all_nmis else 0.0,
        "mean_nmi": float(np.mean(all_nmis)) if all_nmis else 0.0,
    }

    # 5. Plateau alignment (falsification test)
    print("  [T1] Testing plateau alignment (falsification criterion)...")
    # Use first prompt's feature plateaus as representative
    first_pk = list(all_plateaus.keys())[0] if all_plateaus else None
    if first_pk and cluster_plateaus:
        pa = plateau_alignment(
            all_plateaus[first_pk], cluster_plateaus, layer_indices
        )
        results["plateau_alignment"] = pa
        print(f"    Alignment rate: {pa.get('alignment_rate', 0):.3f} "
              f"→ {pa.get('falsification', 'untestable')}")
    else:
        results["plateau_alignment"] = {"falsification": "untestable"}

    # 6. Merge dynamics
    print("  [T1] Analyzing merge event feature dynamics...")
    merge_results = {}
    for pk, traj in trajs.items():
        pk_merges = merge_layers.get(pk, [])
        if pk_merges:
            md = merge_feature_dynamics(
                traj, pk_merges, layer_indices,
                cluster_identity_features=mi_results.get(pk),
            )
            merge_results[pk] = md
            s = md["summary"]
            print(f"    {pk}: {s['n_merges_analyzed']} merges, "
                  f"mean dying={s['mean_dying']:.1f}, born={s['mean_born']:.1f}")
    results["merge_dynamics"] = merge_results

    # For cross-track: per-layer chorus results keyed by layer
    chorus_per_layer = {}
    for pk_data in chorus_results.values():
        for lk, ld in pk_data.items():
            chorus_per_layer[lk] = ld
    results["chorus_per_layer"] = chorus_per_layer

    return results


def run_track2(
    prompt_store: PromptActivationStore,
    layer_indices: list,
    hdbscan_labels: dict,
    v_projectors: dict,
    config: dict,
) -> dict:
    """Run all Track 2 analyses."""
    print("\n  === Track 2: Direct geometric methods ===")
    results = {}

    # Build per-layer activations and labels for each prompt
    for pk in prompt_store.keys():
        acts_per_layer = extract_per_layer_activations(
            prompt_store, pk, layer_indices
        )
        labs_per_layer = build_labels_per_layer(hdbscan_labels, pk)

        if not labs_per_layer:
            print(f"  [T2] {pk}: no HDBSCAN labels, skipping")
            continue

        print(f"  [T2] Processing {pk}...")

        # 1. LDA stability
        print("    LDA directions...")
        lda = lda_stability_across_layers(acts_per_layer, labs_per_layer)
        results.setdefault("lda_results", {})["per_layer"] = lda.get("per_layer", {})
        results.setdefault("lda_results", {})["cosine_trajectory"] = lda.get("cosine_trajectory", [])
        print(f"    Mean LDA cosine stability: {lda.get('mean_cosine', 0):.3f}")

        # 2. PCA on deltas
        print("    PCA on layer deltas...")
        delta_pca = pca_on_deltas(acts_per_layer, v_projectors=v_projectors or None)
        results["delta_pca_results"] = delta_pca
        s = delta_pca["summary"]
        print(f"    Mean update variance: {s['mean_total_variance']:.6f}, "
              f"mean top-1 explained: {s['mean_top1_explained']:.3f}")

        # 3. Linear probes
        print("    Linear probes...")
        probes = probe_accuracy_trajectory(acts_per_layer, labs_per_layer)
        results["probe_results"] = probes
        results["probe_directions"] = probes.get("probe_directions", {})
        s = probes["summary"]
        print(f"    Probe accuracy: mean={s['mean_accuracy']:.3f}, "
              f"max={s['max_accuracy']:.3f}")

        # 4. Probe V-alignment
        if v_projectors and probes.get("probe_directions"):
            print("    Probe V-alignment...")
            v_align = probe_v_alignment(probes["probe_directions"], v_projectors)
            results["v_alignment"] = v_align
            for layer, la in v_align.items():
                if isinstance(la, dict) and "mean_repulsive" in la:
                    print(f"      Layer {layer}: rep={la['mean_repulsive']:.3f}, "
                          f"att={la['mean_attractive']:.3f}")

        # Use first prompt only (extend later for multi-prompt)
        break

    # Summaries
    results["lda_summary"] = {
        "mean_cosine": results.get("lda_results", {}).get("mean_cosine", 0),
    }
    results["probe_summary"] = results.get("probe_results", {}).get("summary", {})
    results["delta_pca_summary"] = results.get("delta_pca_results", {}).get("summary", {})

    # For cross-track: per-layer probe results
    results["probe_per_layer"] = results.get("probe_results", {}).get("per_layer", {})

    return results


def run_track3(
    crosscoder: Crosscoder,
    prompt_store: PromptActivationStore,
    layer_indices: list,
    v_projectors: dict,
    cluster_count: int,
    model_name: str,
    config: dict,
) -> dict:
    """Run Track 3: low-rank autoencoder."""
    print("\n  === Track 3: Low-rank autoencoder ===")
    results = {}

    model_info = SUPPORTED_MODELS.get(model_name, {})
    d_model = model_info.get("d_model", crosscoder.cfg.d_model)
    n_layers = len(layer_indices)

    ae_cfg = LowRankAEConfig(
        d_input=d_model * n_layers,
        rank=cluster_count,
        n_layers=n_layers,
        d_model=d_model,
    )
    print(f"  [T3] Bottleneck rank: {cluster_count} (matching cluster count)")
    print(f"  [T3] Input dim: {ae_cfg.d_input}")

    # Check for existing checkpoint
    ckpt_path = _checkpoint_dir(model_name) / "low_rank_ae"
    if (ckpt_path / "model.pt").exists() and not config.get("force_train_lrae"):
        print(f"  [T3] Loading existing checkpoint from {ckpt_path}")
        lrae = load_low_rank_ae(ckpt_path, device=DEVICE)
    else:
        # Train
        train_cfg = LRAETrainingConfig(
            lr=config.get("lrae_lr", 1e-3),
            total_steps=config.get("lrae_steps", 20000),
            batch_size=config.get("lrae_batch_size", 512),
            checkpoint_dir=str(_checkpoint_dir(model_name)),
        )

        cache_dir = _cache_dir(model_name)
        buffer = ActivationBuffer(cache_dir)
        adapter = ActivationBufferAdapter(buffer, train_cfg.batch_size)

        print(f"  [T3] Training low-rank AE ({train_cfg.total_steps} steps)...")
        lrae = train_low_rank_ae(ae_cfg, train_cfg, adapter, device=DEVICE)

    # Bottleneck directions
    ae_dirs = lrae.bottleneck_directions()
    results["bottleneck_directions"] = ae_dirs
    print(f"  [T3] Extracted {ae_dirs.shape[0]} bottleneck directions")

    # V-alignment
    if v_projectors:
        print("  [T3] Testing V-alignment of bottleneck directions...")
        v_align = bottleneck_v_alignment(lrae, v_projectors)
        results["v_alignment"] = v_align
        print(f"    Mean repulsive: {v_align.get('mean_repulsive', 0):.3f}, "
              f"attractive: {v_align.get('mean_attractive', 0):.3f}")
        print(f"    {v_align.get('n_repulsive_dominant', 0)} repulsive-dominant, "
              f"{v_align.get('n_attractive_dominant', 0)} attractive-dominant")

    # Reconstruction comparison
    print("  [T3] Comparing reconstruction quality vs crosscoder...")
    recon = compare_reconstruction(lrae, crosscoder, prompt_store, device=DEVICE)
    results["reconstruction"] = recon
    for pk, r in recon.items():
        print(f"    {pk}: LRAE={r['lrae_mse']:.6f}, CC={r['crosscoder_mse']:.6f}, "
              f"ratio={r['ratio']:.2f}")

    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_phase4(args) -> dict:
    """Full Phase 4 pipeline for one model."""
    model_name = args.model
    model_info = SUPPORTED_MODELS[model_name]
    model_stem = model_info["stem"]

    print(f"\n{'='*60}")
    print(f"Phase 4: {model_name}")
    print(f"{'='*60}")

    out_dir = _results_dir(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load artifacts ---
    print("\n  Loading artifacts...")

    p1_artifacts = {}
    if args.phase1_dir:
        p1_artifacts = _load_phase1(Path(args.phase1_dir), model_stem)

    p2_artifacts = {}
    if args.phase2_dir:
        p2_artifacts = _load_phase2(
            Path(args.phase2_dir), model_stem, k_top=args.k_top
        )

    p3_artifacts = _load_phase3(
        Path(args.phase3_dir) if args.phase3_dir else Path("results/phase3"),
        model_stem, DEVICE,
    )

    crosscoder = p3_artifacts.get("crosscoder")
    prompt_store = p3_artifacts.get("prompt_store")
    layer_indices = p3_artifacts.get("layer_indices", [])

    if crosscoder is None:
        print("ERROR: No crosscoder checkpoint. Cannot proceed.")
        return {"error": "missing crosscoder"}
    if not prompt_store or not prompt_store.keys():
        print("ERROR: No eval prompts. Cannot proceed.")
        return {"error": "missing prompts"}

    hdbscan_labels = p1_artifacts.get("hdbscan_labels", {})
    merge_layers = p1_artifacts.get("merge_layers", {})
    plateau_windows = p1_artifacts.get("plateau_windows", {})

    # Flatten plateau windows for alignment test
    cluster_plateaus = []
    for pk_plateaus in plateau_windows.values():
        cluster_plateaus.extend(pk_plateaus)

    # Determine typical cluster count for Track 3
    cluster_counts = []
    for pk_labels in hdbscan_labels.values():
        for layer_labels in pk_labels.values():
            arr = np.array(layer_labels)
            k = len(set(arr[arr >= 0].tolist()))
            if k >= 2:
                cluster_counts.append(k)
    typical_k = int(np.median(cluster_counts)) if cluster_counts else 5
    print(f"\n  Typical cluster count: {typical_k}")

    tracks_to_run = args.tracks or [1, 2, 3]

    # --- Track 1 ---
    t1_results = {}
    if 1 in tracks_to_run:
        t1_results = run_track1(
            crosscoder, prompt_store, layer_indices,
            hdbscan_labels, merge_layers, cluster_plateaus,
            config=vars(args),
        )

    # --- Track 2 ---
    t2_results = {}
    if 2 in tracks_to_run:
        t2_results = run_track2(
            prompt_store, layer_indices,
            hdbscan_labels, p2_artifacts,
            config=vars(args),
        )

    # --- Track 3 ---
    t3_results = None
    if 3 in tracks_to_run and not args.skip_track3:
        t3_results = run_track3(
            crosscoder, prompt_store, layer_indices,
            p2_artifacts, typical_k, model_name,
            config=vars(args),
        )

    # --- Cross-track comparison ---
    print("\n  === Cross-track comparison ===")
    agreement = cross_track_agreement(t1_results, t2_results, t3_results)
    t1t2 = agreement.get("t1_t2_correlation", {})
    if "spearman_rho" in t1t2:
        print(f"  T1–T2 correlation: ρ={t1t2['spearman_rho']:.3f} "
              f"(p={t1t2['pval']:.3f}) → {t1t2['interpretation']}")

    # --- Verdict ---
    print("\n  === Phase 4 Verdict ===")
    verdict = build_phase4_verdict(
        t1_results, t2_results, t3_results,
        agreement,
        t1_results.get("plateau_alignment", {}),
    )

    for track_name, tv in verdict.get("tracks", {}).items():
        print(f"  {track_name}: {tv.get('verdict', 'N/A')}")
    print(f"  Overall: {verdict.get('overall', 'N/A')}")
    if "interpretation" in verdict:
        print(f"  → {verdict['interpretation']}")

    # --- Save ---
    # Build activations_per_layer for centroid export
    acts_for_centroids = {}
    for pk in prompt_store.keys():
        acts_for_centroids[pk] = extract_per_layer_activations(
            prompt_store, pk, layer_indices
        )

    save_phase4_outputs(
        out_dir, t1_results, t2_results, t3_results, verdict,
        activations_per_layer=acts_for_centroids,
        hdbscan_labels=hdbscan_labels,
    )

    return {
        "track1": t1_results,
        "track2": t2_results,
        "track3": t3_results,
        "agreement": agreement,
        "verdict": verdict,
        "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Identifying Metastable Features")

    # Model selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--albert-only", action="store_true")
    group.add_argument("--gpt2-only", action="store_true")
    parser.add_argument("--model", type=str, default=None,
                        help="Explicit model name (overrides --albert-only/--gpt2-only)")

    # Artifact paths
    parser.add_argument("--phase1-dir", type=str, default=None)
    parser.add_argument("--phase2-dir", type=str, default=None)
    parser.add_argument("--phase3-dir", type=str, default=None)

    # Track selection
    parser.add_argument("--tracks", type=int, nargs="+", default=None,
                        help="Which tracks to run (1, 2, 3)")
    parser.add_argument("--skip-track3", action="store_true",
                        help="Skip low-rank AE training")

    # Track 1 config
    parser.add_argument("--var-threshold", type=float, default=0.01)
    parser.add_argument("--min-plateau-len", type=int, default=3)
    parser.add_argument("--coact-threshold", type=float, default=0.3)

    # Track 3 config
    parser.add_argument("--lrae-steps", type=int, default=20000)
    parser.add_argument("--lrae-lr", type=float, default=1e-3)
    parser.add_argument("--lrae-batch-size", type=int, default=512)
    parser.add_argument("--force-train-lrae", action="store_true")

    # Phase 2 projector config
    parser.add_argument("--k-top", type=int, default=64,
                        help="Number of V eigenvectors for projectors")

    args = parser.parse_args()

    # Resolve model
    if args.model:
        models = [args.model]
    elif args.albert_only:
        models = ["albert-xlarge-v2"]
    elif args.gpt2_only:
        models = ["gpt2-large"]
    else:
        models = list(SUPPORTED_MODELS.keys())

    for model_name in models:
        if model_name not in SUPPORTED_MODELS:
            print(f"Unknown model: {model_name}")
            continue
        args.model = model_name
        try:
            run_phase4(args)
        except Exception as e:
            print(f"\nERROR running {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
