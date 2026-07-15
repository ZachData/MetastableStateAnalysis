"""
run_2.py — p2_eigenspectra experiment orchestrator and CLI entry point.

Usage examples:
    python -m p2_eigenspectra.run_2 --full
    python -m p2_eigenspectra.run_2 --offline results/p2_eigenspectra_full --phase1-dir results/2026-04-23_18-30-06

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
    RANDOM_INIT_SEED,
)
from core.models import load_model, randomize_weights

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
    random_init_seed: int = None,
    output_dir: Path = None,
    save_cross_run: bool = True,
    ) -> list:
    """Full p2_eigenspectra pipeline: load models, extract weights, decompose, analyse.

    Parameters
    ----------
    random_init_seed : seed passed to randomize_weights for random-init models.
        Defaults to RANDOM_INIT_SEED from config.  Must match the seed used when
        the corresponding Phase 1 run was produced, or the OV decomposition will
        not correspond to the recorded activations.
    output_dir : if provided, write results here instead of creating a new
        timestamped directory.  Callers that want to merge trained + random
        verdicts into one cross-run report pass the same output_dir to both
        run_full and run_random_baseline and set save_cross_run=False until
        the final aggregation step.
    save_cross_run : if True (default), write the cross-run summary inside
        this call.  Set False when the caller will aggregate verdicts from
        multiple calls before writing the summary.
    """
    if models_to_run is None:
        models_to_run = list(MODEL_CONFIGS.keys())
    if prompts_to_run is None:
        prompts_to_run = list(PROMPTS.keys())

    phase1_dir = _resolve_phase1_dir(phase1_dir)
    if phase1_dir is None:
        print("No Phase 1 results found. Run Phase 1 first.")
        return []

    if output_dir is None:
        timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = BASE_RESULTS_DIR / f"p2_eigenspectra_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\np2_eigenspectra output: {output_dir}")
    print(f"Phase 1 source:        {phase1_dir}")

    all_verdicts = []
    seed = random_init_seed if random_init_seed is not None else RANDOM_INIT_SEED

    for model_name in models_to_run:
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
        cfg = MODEL_CONFIGS[model_name]
        try:
            model, tokenizer = load_model(model_name)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        # Apply random re-initialisation when the model config requests it.
        # The seed must match the one used in the corresponding Phase 1 run;
        # otherwise the OV decomposition won't correspond to the activations.
        if cfg.get("random_init", False):
            scheme = cfg.get("random_init_scheme", "orthogonal")
            print(f"  Re-initialising weights (scheme={scheme}, seed={seed})")
            randomize_weights(model, scheme=scheme, seed=seed)

        ov_data = analyze_weights(model, model_name, output_dir)

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

    if all_verdicts and save_cross_run:
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
                
                # --- Appended to make offline run _run_decompose---
                if decomposed is None:
                    print(f"    Decomposition missing. Generating decomposition for {model_name}...")
                    model, tokenizer = load_model(model_name)
                    decomposed = _run_decompose(model, tokenizer, model_name, prompt_key, cfg)
                    # Optionally save it so it's there next time
                    save_decomposed(decomposed, decompose_dir)
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                # ----------------------
                
                if decomposed is None:
                    print(f"    Decompose: no saved deltas and generation failed.")
                    continue # Skip if still None

                ctx = {
                    "model_name":     model_name,
                    "prompt_key":     prompt_key,
                    "stem":           stem,
                    "ov_data":        ov_data,
                    "traj":           traj,
                    "decomposed":     decomposed,
                    "phase1_run_dir": run_dir,
                    "weights_dir":    weights_dir, #new
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
# Random-baseline runner
# ---------------------------------------------------------------------------

def run_random_baseline(
    random_dir: Path,
    output_dir: Path,
    prompts_to_run: list = None,
    ) -> list:
    """
    Run Phase 2 on all random-initialisation Phase 1 runs found in random_dir.

    Expected layout (produced by run_1 --random-baseline --seed N, then
    collected into one place):

        random_dir/
            2026-01-01_10-00-00/        ← timestamped experiment dir
                experiment.txt          ← command line with --seed N recorded
                albert-base-v2-random_wiki_paragraph/
                gpt2-large-random_wiki_paragraph/
                ...
            2026-01-01_11-00-00/        ← next seed
                ...

    Also handles a flat layout (no nested timestamp dirs) by treating
    random_dir itself as a single phase1 dir with RANDOM_INIT_SEED.

    The seed is extracted from the ``command`` line in experiment.txt so that
    the OV weight decomposition is computed with the same random state that
    produced the Phase 1 activations.  Without this, the repulsive/attractive
    subspaces won't correspond to the recorded token trajectories.

    Results are written into output_dir using a seed-tagged stem
    (e.g., albert-base-v2-random_wiki_paragraph_s42/) so that multiple
    seeds don't collide inside a shared output directory.
    """
    import re
    import json

    random_dir = Path(random_dir)
    if not random_dir.exists():
        print(f"[random] {random_dir} does not exist, skipping.")
        return []

    # --- Discover seed entries -----------------------------------------------
    seed_entries: list[tuple[Path, int]] = []
    for d in sorted(random_dir.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "experiment.txt"
        if manifest.exists():
            seed = _parse_seed_from_manifest(manifest)
            seed_entries.append((d, seed))

    if not seed_entries:
        # Flat layout: random_dir itself is the phase1 dir.
        seed_entries = [(random_dir, RANDOM_INIT_SEED)]
        print(f"[random] Flat layout detected; treating {random_dir} as a single "
              f"phase1 dir (seed={RANDOM_INIT_SEED}).")
    else:
        print(f"[random] Found {len(seed_entries)} seed run(s) in {random_dir}")

    if prompts_to_run is None:
        prompts_to_run = list(PROMPTS.keys())

    all_verdicts = []

    for phase1_dir, seed in seed_entries:
        # --- Discover which random models appear in this seed dir ------------
        model_names = _discover_random_models(phase1_dir)
        if not model_names:
            print(f"  [random] No recognised models in {phase1_dir.name}, skipping.")
            continue
        print(f"\n  Seed dir : {phase1_dir.name}  seed={seed}"
              f"  models={model_names}")

        for model_name in model_names:
            cfg = MODEL_CONFIGS[model_name]
            print(f"\n{'='*60}\n[random] Model: {model_name}  seed={seed}\n{'='*60}")
            try:
                model, tokenizer = load_model(model_name)
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
                continue

            scheme = cfg.get("random_init_scheme", "orthogonal")
            print(f"  Re-initialising weights (scheme={scheme}, seed={seed})")
            randomize_weights(model, scheme=scheme, seed=seed)

            ov_data = analyze_weights(model, model_name, output_dir)

            for prompt_key in prompts_to_run:
                run_dir = _find_run_dir(phase1_dir, model_name, prompt_key, cfg)
                if run_dir is None:
                    continue

                print(f"\n  Prompt: {prompt_key}")
                try:
                    traj = analyze_trajectory_offline_perlayer(run_dir, ov_data)

                    # Stem includes seed suffix so multiple seeds don't collide.
                    stem = f"{model_name.replace('/', '_')}_{prompt_key}_s{seed}"
                    decomposed = _run_decompose(
                        model, tokenizer, model_name, prompt_key, cfg
                    )
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
                    verdict["random_seed"] = seed
                    all_verdicts.append(verdict)

                except Exception as e:
                    print(f"  Failed {model_name} {prompt_key} seed={seed}: {e}")
                    traceback.print_exc()

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_verdicts


def _discover_random_models(phase1_dir: Path) -> list:
    """
    Read geometry.json files in phase1_dir to discover model names.
    Returns only names present in MODEL_CONFIGS (skips unrecognised models).
    """
    import json
    found: set = set()
    for run_dir in Path(phase1_dir).iterdir():
        if not run_dir.is_dir():
            continue
        geo_path = run_dir / "geometry.json"
        if not geo_path.exists():
            continue
        try:
            geo = json.loads(geo_path.read_text())
            m = geo.get("model", "")
            if m and m in MODEL_CONFIGS:
                found.add(m)
        except Exception:
            pass
    return sorted(found)


def _parse_seed_from_manifest(manifest_path: Path) -> int:
    """
    Extract --seed N from the command line recorded in experiment.txt.
    Falls back to RANDOM_INIT_SEED if not found.
    """
    import re
    try:
        text = manifest_path.read_text()
        m = re.search(r"--seed\s+(\d+)", text)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return RANDOM_INIT_SEED


def _resolve_random_dir(arg: str | None) -> Path | None:
    """
    Resolve the --random-dir argument.
    'auto' (default): use results/p1_random if it exists, else None.
    Any other value: treat as a path.
    """
    if arg is None or arg == "auto":
        p = BASE_RESULTS_DIR / "p1_random"
        return p if p.exists() else None
    p = Path(arg)
    return p if p.exists() else None


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
            if d.exists() and (d / "layer_metrics.json").exists():
                return d
    stem = f"{model_name.replace('/', '_')}_{prompt_key}"
    d    = phase1_dir / stem
    if d.exists() and (d / "layer_metrics.json").exists():
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
    parser.add_argument(
        "--random-dir", type=str, default="auto", metavar="DIR",
        help=(
            "Directory containing Phase 1 random-initialisation runs "
            "(default 'auto': use results/p1_random if it exists). "
            "Pass 'none' to disable even when results/p1_random exists."
        ),
    )
    args = parser.parse_args()

    if args.fast:
        models  = ["albert-base-v2"]
        prompts = ["wiki_paragraph"]
    else:
        models  = args.models
        prompts = args.prompts

    # Resolve random-dir once so both --full and --offline can use it.
    rand_dir = (
        None if args.random_dir == "none"
        else _resolve_random_dir(args.random_dir)
    )

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
        # Create a shared output dir so trained + random verdicts end up in
        # one place and the final cross-run report covers both.
        timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = BASE_RESULTS_DIR / f"p2_eigenspectra_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_verdicts = run_full(
            models_to_run=models,
            prompts_to_run=prompts,
            phase1_dir=Path(args.phase1_dir) if args.phase1_dir else None,
            output_dir=output_dir,
            save_cross_run=rand_dir is None,   # defer if random runs follow
        )

        if rand_dir is not None:
            print(f"\n[random] Processing random-init runs from {rand_dir}")
            random_verdicts = run_random_baseline(
                random_dir=rand_dir,
                output_dir=output_dir,
                prompts_to_run=prompts,
            )
            all_verdicts.extend(random_verdicts)
            # Write the consolidated cross-run report now that all verdicts
            # (trained + random) are available.
            if all_verdicts:
                _save_cross_run(all_verdicts, output_dir)

    else:
        parser.print_help()
        print("\nSpecify --full or --offline <P2_FULL_DIR>")
