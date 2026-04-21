"""
run_2i.py — Phase 2 Imaginary experiment orchestrator.

Usage:
    # Full pipeline, all models, using saved Phase 2 artifacts
    python -m phase2i.run_2i --phase2-dir results/phase2_full --phase1-dir results/phase1

    # Single model
    python -m phase2i.run_2i --phase2-dir results/phase2_full --phase1-dir results/phase1 \
        --models albert-xlarge-v2

    # Block 1 only (weight analysis + causal test, no hemispheric tracking)
    python -m phase2i.run_2i --phase2-dir results/phase2_full --phase1-dir results/phase1 \
        --block1-only

    # Force Block 2 even if Block 1b shows rotation is neutral
    python -m phase2i.run_2i --phase2-dir results/phase2_full --phase1-dir results/phase1 \
        --force-block2

Modes:
    Default: Runs Block 1a, 1b. If 1b shows rotational contribution, runs Block 2.
    --block1-only: Stop after Block 1.
    --force-block2: Run Block 2 regardless of Block 1b result.
    --skip-ffn: Skip FFN rotation analysis (saves time).

Dependencies:
    - Phase 2 saved artifacts: ov_weights_{model}.npz, ov_decomp_{model}.npz
    - Phase 1 saved activations: activations.npz, metrics.json
    - (Optional) Phase 2 FFN deltas: ffn_deltas_raw.npz or ffn_deltas_normed.npz
"""

import sys
import json
import argparse
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path

from phase2i.rotational_schur import (
    analyze_rotational_spectrum,
    summary_to_json,
)
from phase2i.rotational_rescaled import (
    analyze_rotational_rescaling,
    comparison_to_json,
)
from phase2i.fiedler_tracking import (
    analyze_fiedler_tracking,
    fiedler_to_json,
)
from phase2i.rotation_hemisphere import (
    analyze_rotation_hemisphere,
    rotation_hemisphere_to_json,
)
from phase2i.ffn_rotation import (
    analyze_ffn_rotation,
    ffn_rotation_to_json,
)


# ---------------------------------------------------------------------------
# Artifact loaders
# ---------------------------------------------------------------------------

def load_ov_data(phase2_dir: Path, model_stem: str) -> dict:
    """Load OV matrices from Phase 2 saved artifacts."""
    weights_path = phase2_dir / f"ov_weights_{model_stem}.npz"
    summary_path = phase2_dir / f"ov_summary_{model_stem}.json"

    if not weights_path.exists():
        return None

    data = np.load(weights_path)

    # Determine if per-layer
    keys = list(data.keys())
    is_per_layer = any("layer_" in k for k in keys)

    if is_per_layer:
        # Extract per-layer OV totals
        layer_keys = sorted(
            [k for k in keys if k.startswith("ov_total_layer_")],
            key=lambda k: int(k.split("layer_")[1])
        )
        ov_total = [data[k] for k in layer_keys]
        layer_names = [k.replace("ov_total_", "") for k in layer_keys]
        return {
            "ov_total": ov_total,
            "is_per_layer": True,
            "layer_names": layer_names,
        }
    else:
        return {
            "ov_total": data["ov_total_shared"],
            "is_per_layer": False,
            "layer_names": ["shared"],
        }


def load_activations(phase1_run_dir: Path) -> np.ndarray:
    """Load L2-normed activations from Phase 1."""
    act_path = phase1_run_dir / "activations.npz"
    if not act_path.exists():
        return None
    data = np.load(act_path)
    # Phase 1 saves as "activations" or the first key
    key = "activations" if "activations" in data else list(data.keys())[0]
    return data[key]


def load_phase1_events(phase1_run_dir: Path) -> dict:
    """Load Phase 1 events (violations, plateaus) from metrics.json."""
    metrics_path = phase1_run_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    # Import Phase 2's loader
    try:
        from phase2.trajectory import load_phase1_events as _load
        return _load(phase1_run_dir)
    except ImportError:
        # Minimal fallback: just extract violations
        with open(metrics_path) as f:
            results = json.load(f)
        # Reconstruct energy violations
        layers = results.get("layers", [])
        violations = {}
        for beta in [0.1, 1.0, 2.0, 5.0]:
            v_list = []
            prev_e = None
            for layer in layers:
                idx = layer["layer"]
                energies = {float(k): v for k, v in layer.get("energies", {}).items()}
                e = energies.get(beta)
                if e is not None and prev_e is not None and e < prev_e - 1e-6:
                    v_list.append(idx)
                prev_e = e
            violations[beta] = v_list
        return {"energy_violations": violations}


def load_ffn_deltas(phase2_run_dir: Path) -> np.ndarray:
    """Load FFN deltas from Phase 2 decomposition."""
    for fname in ["ffn_deltas_normed.npz", "ffn_deltas_raw.npz"]:
        path = phase2_run_dir / fname
        if path.exists():
            data = np.load(path)
            key = "ffn_deltas" if "ffn_deltas" in data else list(data.keys())[0]
            return data[key]
    return None


def find_phase1_runs(phase1_dir: Path, model_stem: str) -> list:
    """Find Phase 1 run directories matching a model stem."""
    runs = []
    if not phase1_dir.exists():
        return runs
    for subdir in sorted(phase1_dir.iterdir()):
        if subdir.is_dir() and model_stem in subdir.name:
            if (subdir / "activations.npz").exists():
                runs.append(subdir)
    return runs


def find_phase2_runs(phase2_dir: Path, model_stem: str) -> list:
    """Find Phase 2 run directories with FFN deltas for a model."""
    runs = []
    if not phase2_dir.exists():
        return runs
    for subdir in sorted(phase2_dir.iterdir()):
        if subdir.is_dir() and model_stem in subdir.name:
            if (subdir / "ffn_deltas_normed.npz").exists() or \
               (subdir / "ffn_deltas_raw.npz").exists():
                runs.append(subdir)
    return runs


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def run_model(
    model_stem: str,
    phase2_dir: Path,
    phase1_dir: Path,
    save_dir: Path,
    block1_only: bool = False,
    force_block2: bool = False,
    skip_ffn: bool = False,
    beta_values: list = None,
) -> dict:
    """
    Run Phase 2i analysis for one model.

    Parameters
    ----------
    model_stem   : e.g. "albert-xlarge-v2", "gpt2-large"
    phase2_dir   : root directory containing Phase 2 saved artifacts
    phase1_dir   : root directory containing Phase 1 run subdirectories
    save_dir     : where to write results
    block1_only  : stop after Block 1
    force_block2 : run Block 2 even if Block 1b says rotation is neutral
    skip_ffn     : skip FFN rotation analysis
    beta_values  : list of beta values for energy computation

    Returns
    -------
    dict with all results
    """
    if beta_values is None:
        beta_values = [0.1, 1.0, 2.0, 5.0]

    model_save = save_dir / model_stem
    model_save.mkdir(parents=True, exist_ok=True)

    results = {"model": model_stem, "timestamp": datetime.now().isoformat()}

    # -----------------------------------------------------------------------
    # Load OV data
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 2i: {model_stem}")
    print(f"{'='*60}")

    ov_data = load_ov_data(phase2_dir, model_stem)
    if ov_data is None:
        print(f"  ERROR: No OV weights found for {model_stem} in {phase2_dir}")
        return {"model": model_stem, "error": "no_ov_weights"}

    # -----------------------------------------------------------------------
    # Block 1a: Rotational spectrum characterization
    # -----------------------------------------------------------------------
    print(f"\n--- Block 1a: Rotational spectrum ---")
    try:
        rot_spectrum = analyze_rotational_spectrum(ov_data, top_k_planes=32)
        rot_json = summary_to_json(rot_spectrum)
        results["block1a"] = rot_json

        # Print summary
        if ov_data["is_per_layer"]:
            depth = rot_spectrum["depth_profile"]["summary"]
            print(f"  Layers: {len(ov_data['layer_names'])}")
            print(f"  Mean θ across layers: {depth.get('theta_mean_across_layers', 0):.4f} rad")
            print(f"  Mean Henrici (relative): {depth.get('henrici_mean', 0):.4f}")
            print(f"  Max Henrici layer: {depth.get('henrici_max_layer', '?')}")
        else:
            print(f"  Real eigenvalues: {rot_json.get('n_real', 0)} "
                  f"({rot_json.get('frac_real', 0):.1%})")
            print(f"  Complex pairs: {rot_json.get('n_complex', 0)} "
                  f"({rot_json.get('frac_complex_dims', 0):.1%} of dims)")
            print(f"  Rotational energy fraction: "
                  f"{rot_json.get('rotational_fraction', 0):.3f}")
            print(f"  Mean θ: {rot_json.get('theta_mean', 0):.4f} rad "
                  f"({np.degrees(rot_json.get('theta_mean', 0)):.1f}°)")
            print(f"  Henrici (relative): {rot_json.get('henrici_relative', 0):.4f}")

        with open(model_save / "block1a_rotational_spectrum.json", "w") as f:
            json.dump(rot_json, f, indent=2, default=_json_default)
        print(f"  Saved: {model_save}/block1a_rotational_spectrum.json")

    except Exception as e:
        print(f"  ERROR in Block 1a: {e}")
        traceback.print_exc()
        results["block1a"] = {"error": str(e)}
        rot_spectrum = None

    # -----------------------------------------------------------------------
    # Block 1b: Rotational rescaled frame (needs activations)
    # -----------------------------------------------------------------------
    print(f"\n--- Block 1b: Rotational rescaled frame ---")
    phase1_runs = find_phase1_runs(phase1_dir, model_stem)

    if not phase1_runs:
        print(f"  No Phase 1 runs found for {model_stem}. Skipping Block 1b.")
        results["block1b"] = {"error": "no_phase1_activations"}
        rotation_contributes = False
    else:
        block1b_results = {}
        rotation_contributes = False

        for run_dir in phase1_runs:
            prompt = run_dir.name.split("iter_")[-1] if "iter_" in run_dir.name else run_dir.name
            print(f"  Prompt: {prompt}")

            activations = load_activations(run_dir)
            if activations is None:
                print(f"    No activations found. Skipping.")
                continue

            try:
                rescaled = analyze_rotational_rescaling(
                    activations, ov_data, beta_values,
                )
                rescaled_json = comparison_to_json(rescaled)
                block1b_results[prompt] = rescaled_json

                # Print comparison
                interp = rescaled["interpretation"]
                print(f"    Overall: {interp['overall']}")
                for beta, comp in rescaled["frames"]["comparison"].items():
                    print(f"    β={beta}: orig={comp['n_original']} "
                          f"full={comp['n_full_rescaled']} "
                          f"S-only={comp['n_signed_only']} "
                          f"A-only={comp['n_rotation_only']}")

                if interp["overall"] != "rotation_neutral":
                    rotation_contributes = True

            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                block1b_results[prompt] = {"error": str(e)}

        results["block1b"] = block1b_results

        with open(model_save / "block1b_rescaled_comparison.json", "w") as f:
            json.dump(block1b_results, f, indent=2, default=_json_default)
        print(f"  Saved: {model_save}/block1b_rescaled_comparison.json")

    # -----------------------------------------------------------------------
    # Decision gate
    # -----------------------------------------------------------------------
    run_block2 = (rotation_contributes or force_block2) and not block1_only
    results["rotation_contributes"] = rotation_contributes
    results["block2_decision"] = "run" if run_block2 else "skip"

    if not run_block2:
        reason = "block1_only" if block1_only else "rotation_neutral"
        print(f"\n  Block 2 skipped: {reason}")
        if reason == "rotation_neutral":
            print("  Conclusion: Rotational dynamics are dynamically neutral.")
            print("  The 97% imaginary eigenvalues do not contribute to metastability.")
            print("  Phase 2's signed-eigenvalue analysis was complete.")
        return results

    # -----------------------------------------------------------------------
    # Block 2: Hemispheric geometry
    # -----------------------------------------------------------------------
    print(f"\n--- Block 2: Hemispheric geometry ---")
    block2_results = {}

    for run_dir in phase1_runs:
        prompt = run_dir.name.split("iter_")[-1] if "iter_" in run_dir.name else run_dir.name
        print(f"  Prompt: {prompt}")

        activations = load_activations(run_dir)
        if activations is None:
            continue

        phase1_events = load_phase1_events(run_dir)

        try:
            # Fiedler tracking
            fiedler = analyze_fiedler_tracking(
                activations, phase1_events, beta=1.0,
            )
            fiedler_json = fiedler_to_json(fiedler)

            # Rotation-hemisphere alignment (needs rotation plane projectors)
            if rot_spectrum is not None:
                if ov_data["is_per_layer"]:
                    plane_projs = rot_spectrum["plane_projectors"]
                else:
                    plane_projs = rot_spectrum["plane_projectors"]

                # Extract sub-results needed by rotation_hemisphere
                from phase2i.fiedler_tracking import (
                    extract_fiedler_per_layer,
                    hemisphere_assignments,
                )
                fiedler_data = extract_fiedler_per_layer(activations)
                hemi_data = hemisphere_assignments(fiedler_data)

                rot_hemi = analyze_rotation_hemisphere(
                    activations, fiedler_data, hemi_data, plane_projs,
                )
                rot_hemi_json = rotation_hemisphere_to_json(rot_hemi)
            else:
                rot_hemi_json = {"error": "no_rotation_spectrum"}

            block2_results[prompt] = {
                "fiedler": fiedler_json,
                "rotation_hemisphere": rot_hemi_json,
            }

            # Print summary
            print(f"    Crossing rate mean: "
                  f"{fiedler_json.get('crossing_rate_mean', 'N/A')}")
            print(f"    Fiedler cosine mean: "
                  f"{fiedler_json.get('fiedler_cosine_mean', 'N/A')}")
            print(f"    Centroid near-π fraction: "
                  f"{fiedler_json.get('centroid_angle_near_pi', 'N/A')}")
            if isinstance(rot_hemi_json, dict) and "coherence_mean" in rot_hemi_json:
                print(f"    Displacement coherence: "
                      f"{rot_hemi_json.get('coherence_mean', 'N/A')}")

        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            block2_results[prompt] = {"error": str(e)}

    results["block2"] = block2_results
    with open(model_save / "block2_hemispheric.json", "w") as f:
        json.dump(block2_results, f, indent=2, default=_json_default)
    print(f"  Saved: {model_save}/block2_hemispheric.json")

    # -----------------------------------------------------------------------
    # FFN rotation interaction (GPT-2 only, if deltas available)
    # -----------------------------------------------------------------------
    if not skip_ffn and ov_data["is_per_layer"] and rot_spectrum is not None:
        print(f"\n--- FFN rotation analysis ---")
        ffn_results = {}

        phase2_runs = find_phase2_runs(phase2_dir, model_stem)
        for run_dir in phase2_runs:
            prompt = run_dir.name.split("iter_")[-1] if "iter_" in run_dir.name else run_dir.name
            print(f"  Prompt: {prompt}")

            ffn_deltas = load_ffn_deltas(run_dir)
            if ffn_deltas is None:
                print(f"    No FFN deltas found. Skipping.")
                continue

            # Find matching Phase 1 events
            matching_p1 = [r for r in phase1_runs if prompt in r.name]
            if not matching_p1:
                print(f"    No matching Phase 1 run. Skipping.")
                continue

            phase1_events = load_phase1_events(matching_p1[0])
            if phase1_events is None:
                continue

            try:
                ffn_rot = analyze_ffn_rotation(
                    ffn_deltas,
                    rot_spectrum["plane_projectors"],
                    phase1_events,
                    is_per_layer=True,
                    beta=1.0,
                )
                ffn_json = ffn_rotation_to_json(ffn_rot)
                ffn_results[prompt] = ffn_json

                print(f"    Role counts: {ffn_rot['role_counts']}")
                comp = ffn_rot["comparison"]
                for metric, data in comp.items():
                    if isinstance(data, dict) and "z_score" in data:
                        print(f"    {metric}: z={data['z_score']:.2f} "
                              f"(viol={data['v_mean']:.3f}, pop={data['pop_mean']:.3f})")

            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                ffn_results[prompt] = {"error": str(e)}

        results["ffn_rotation"] = ffn_results
        with open(model_save / "ffn_rotation.json", "w") as f:
            json.dump(ffn_results, f, indent=2, default=_json_default)
        print(f"  Saved: {model_save}/ffn_rotation.json")

    return results


# ---------------------------------------------------------------------------
# JSON default handler
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2i: Rotational dynamics analysis"
    )
    parser.add_argument("--phase2-dir", type=str, required=True,
                        help="Phase 2 results directory (contains ov_weights_*.npz)")
    parser.add_argument("--phase1-dir", type=str, required=True,
                        help="Phase 1 results directory (contains model run subdirs)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Model stems to analyze (default: all found)")
    parser.add_argument("--block1-only", action="store_true",
                        help="Run only Block 1 (weight analysis + causal test)")
    parser.add_argument("--force-block2", action="store_true",
                        help="Run Block 2 even if Block 1b shows rotation is neutral")
    parser.add_argument("--skip-ffn", action="store_true",
                        help="Skip FFN rotation analysis")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/phase2i_{timestamp})")

    args = parser.parse_args()

    phase2_dir = Path(args.phase2_dir)
    phase1_dir = Path(args.phase1_dir)

    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = Path("results") / f"phase2i_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Discover models
    if args.models:
        model_stems = args.models
    else:
        # Find all models with saved OV weights
        model_stems = []
        for f in sorted(phase2_dir.glob("ov_weights_*.npz")):
            stem = f.stem.replace("ov_weights_", "")
            model_stems.append(stem)

    if not model_stems:
        print(f"No models found in {phase2_dir}")
        sys.exit(1)

    print(f"Phase 2i: Rotational Dynamics Analysis")
    print(f"Models: {model_stems}")
    print(f"Output: {save_dir}")
    print(f"Block 1 only: {args.block1_only}")
    print(f"Force Block 2: {args.force_block2}")

    all_results = {}
    for model_stem in model_stems:
        try:
            result = run_model(
                model_stem=model_stem,
                phase2_dir=phase2_dir,
                phase1_dir=phase1_dir,
                save_dir=save_dir,
                block1_only=args.block1_only,
                force_block2=args.force_block2,
                skip_ffn=args.skip_ffn,
            )
            all_results[model_stem] = result
        except Exception as e:
            print(f"\nFATAL ERROR for {model_stem}: {e}")
            traceback.print_exc()
            all_results[model_stem] = {"error": str(e)}

    # Save combined results
    with open(save_dir / "phase2i_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"\nAll results saved to {save_dir}/phase2i_results.json")

    # Print overall summary
    print(f"\n{'='*60}")
    print("PHASE 2i SUMMARY")
    print(f"{'='*60}")
    for stem, res in all_results.items():
        contributes = res.get("rotation_contributes", "unknown")
        decision = res.get("block2_decision", "unknown")
        print(f"  {stem}: rotation_contributes={contributes}, block2={decision}")


if __name__ == "__main__":
    main()
