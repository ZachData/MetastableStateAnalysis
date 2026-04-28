"""
run_2i.py — Phase 2i: Rotational Dynamics Investigation orchestrator.

Usage:
    # Full pipeline, all models
    python -m p2b_imaginary.run_2i --phase2-dir results/phase2_full --phase1-dir results/2026-04-23_18-30-06


    # Single model
    python -m p2b_imaginary.run_2i --phase2-dir results/phase2_full --phase1-dir results/phase1 \\
        --models albert-xlarge-v2

    # Block 1 only (weight analysis + causal test, no hemispheric tracking)
    python -m p2b_imaginary.run_2i --phase2-dir results/phase2_full --phase1-dir results/phase1 \\
        --block1-only

    # Force Block 2 even if Block 1b shows rotation is neutral
    python -m p2b_imaginary.run_2i --phase2-dir results/phase2_full --phase1-dir results/phase1 \\
        --force-block2

Modes:
    Default: Runs Block 1a, 1b. If 1b shows rotational contribution, runs Block 2.
    --block1-only: Stop after Block 1.
    --force-block2: Run Block 2 regardless of Block 1b result.
    --skip-ffn: Skip FFN rotation analysis (saves time).

Output layout (per model, under save_dir/{model_stem}/):
    sub/block1a_rotational_spectrum.json         raw results
    sub/block1a_rotational_spectrum.summary.txt  LLM-ready lines
    sub/block1b_rescaled_comparison.json
    sub/block1b_rescaled_comparison.summary.txt
    sub/block2_hemispheric.json                  (if run)
    sub/block2_hemispheric.summary.txt
    sub/ffn_rotation.json                        (if run)
    sub/ffn_rotation.summary.txt
    summary.txt                                  assembled per-model report

Combined (under save_dir/):
    phase2i_results.json                         all raw results
    phase2i_summary.txt                          all models, LLM-consumable

Dependencies:
    - Phase 2 saved artifacts: ov_weights_{model}.npz
    - Phase 1 saved activations: activations.npz, metrics.json
    - (Optional) Phase 2 FFN deltas: ffn_deltas_raw.npz or ffn_deltas_normed.npz
"""

from __future__ import annotations

import sys
import json
import argparse
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path

from p2b_imaginary.rotational_schur import (
    analyze_rotational_spectrum,
    summary_to_json,
)
from p2b_imaginary.rotational_rescaled import (
    analyze_rotational_rescaling,
    comparison_to_json,
)
from p2b_imaginary.fiedler_tracking import (
    analyze_fiedler_tracking,
    fiedler_to_json,
)
from p2b_imaginary.rotation_hemisphere import (
    analyze_rotation_hemisphere,
    rotation_hemisphere_to_json,
)
from p2b_imaginary.ffn_rotation import (
    analyze_ffn_rotation,
    ffn_rotation_to_json,
)


# ---------------------------------------------------------------------------
# Artifact loaders
# ---------------------------------------------------------------------------

def load_ov_data(phase2_dir: Path, model_stem: str) -> dict | None:
    """Load OV matrices from Phase 2 saved artifacts."""
    weights_path = phase2_dir / f"ov_weights_{model_stem}.npz"
    if not weights_path.exists():
        return None

    data = np.load(weights_path)
    keys = list(data.keys())
    is_per_layer = any("layer_" in k for k in keys)

    if is_per_layer:
        layer_keys = sorted(
            [k for k in keys if k.startswith("ov_total_layer_")],
            key=lambda k: int(k.split("layer_")[1])
        )
        return {
            "ov_total": [data[k] for k in layer_keys],
            "is_per_layer": True,
            "layer_names": [k.replace("ov_total_", "") for k in layer_keys],
        }
    else:
        return {
            "ov_total": data["ov_total_shared"],
            "is_per_layer": False,
            "layer_names": ["shared"],
        }


def load_activations(phase1_run_dir: Path) -> np.ndarray | None:
    """Load L2-normed activations from Phase 1."""
    act_path = phase1_run_dir / "activations.npz"
    if not act_path.exists():
        return None
    data = np.load(act_path)
    key = "activations" if "activations" in data else list(data.keys())[0]
    return data[key]


def load_phase1_events(phase1_run_dir: Path) -> dict | None:
    """
    Load Phase 1 events from a run directory.

    Supports both layouts:
      v2: geometry.json + energies.json + clustering.json + spectral.json
          (written by io_utils.save_run since Phase 1 v2 — no metrics.json)
      v1: single metrics.json (legacy)
    """
    phase1_run_dir = Path(phase1_run_dir)

    # ------------------------------------------------------------------
    # v2 layout: separate JSON files
    # ------------------------------------------------------------------
    energies_path = phase1_run_dir / "energies.json"
    geometry_path = phase1_run_dir / "geometry.json"
    if energies_path.exists() and geometry_path.exists():
        with open(geometry_path) as f:
            geo = json.load(f)
        with open(energies_path) as f:
            eng = json.load(f)

        geo_layers = {lr["layer"]: lr for lr in geo.get("layers", [])}
        eng_layers = {lr["layer"]: lr for lr in eng.get("layers", [])}
        all_indices = sorted(set(geo_layers) | set(eng_layers))

        cl_layers: dict = {}
        sp_layers: dict = {}
        cl_path = phase1_run_dir / "clustering.json"
        sp_path = phase1_run_dir / "spectral.json"
        if cl_path.exists():
            with open(cl_path) as f:
                cl_layers = {lr["layer"]: lr for lr in json.load(f).get("layers", [])}
        if sp_path.exists():
            with open(sp_path) as f:
                sp_layers = {lr["layer"]: lr for lr in json.load(f).get("layers", [])}

        beta_values = [0.1, 1.0, 2.0, 5.0]
        energies:          dict = {b: [] for b in beta_values}
        energy_violations: dict = {b: [] for b in beta_values}
        energy_drop_pairs: dict = {b: {} for b in beta_values}

        def _isnan(v):
            return isinstance(v, float) and v != v

        for idx in all_indices:
            el = eng_layers.get(idx, {})
            gl = geo_layers.get(idx, {})
            layer_energies = {float(k): v for k, v in el.get("energies", {}).items()}
            raw_drops = el.get("energy_drop_pairs", {})
            if isinstance(raw_drops, list):
                layer_drops = {1.0: raw_drops} if raw_drops else {}
            else:
                layer_drops = {float(k): v for k, v in raw_drops.items()}

            eff_rank = gl.get("effective_rank", 0) or 0
            for beta in beta_values:
                e = layer_energies.get(beta, float("nan"))
                energies[beta].append(e)
                if len(energies[beta]) >= 2:
                    e_prev = energies[beta][-2]
                    if (not _isnan(e) and not _isnan(e_prev)
                            and e - e_prev < -1e-6
                            and eff_rank >= 3.0):
                        energy_violations[beta].append(idx)
                pairs = layer_drops.get(beta, [])
                if pairs:
                    energy_drop_pairs[beta][idx] = pairs

        geo_layer_list = [geo_layers.get(i, {}) for i in all_indices]
        cl_layer_list  = [cl_layers.get(i, {})  for i in all_indices]
        sp_layer_list  = [sp_layers.get(i, {})  for i in all_indices]

        return {
            "n_layers":          len(all_indices),
            "n_tokens":          geo.get("n_tokens", 0),
            "d_model":           geo.get("d_model", 0),
            "tokens":            geo.get("tokens", []),
            "model":             geo.get("model", ""),
            "prompt":            geo.get("prompt", ""),
            "energies":          energies,
            "energy_violations": energy_violations,
            "energy_drop_pairs": energy_drop_pairs,
            "effective_rank":    [l.get("effective_rank", 0) for l in geo_layer_list],
            "ip_mean":           [l.get("ip_mean", 0) for l in geo_layer_list],
            "ip_mass_near_1":    [l.get("ip_mass_near_1", 0) for l in geo_layer_list],
            "cka_prev":          [l.get("cka_prev", float("nan")) for l in geo_layer_list],
            "spectral_k":        [l.get("k_eigengap", 1) for l in sp_layer_list],
            "kmeans_k":          [
                l.get("clustering", {}).get("kmeans", {}).get("best_k", 2)
                for l in cl_layer_list
            ],
        }

    # ------------------------------------------------------------------
    # v1 layout: single metrics.json (legacy)
    # ------------------------------------------------------------------
    metrics_path = phase1_run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        from phase2.trajectory import load_phase1_events as _load  # fixed: was p2_eigenspectra
        return _load(phase1_run_dir)
    except ImportError:
        with open(metrics_path) as f:
            results = json.load(f)
        layers = results.get("layers", [])
        violations: dict = {}
        for beta in [0.1, 1.0, 2.0, 5.0]:
            v_list = []
            prev_e = None
            for layer in layers:
                idx = layer["layer"]
                layer_energies = {float(k): v for k, v in layer.get("energies", {}).items()}
                e = layer_energies.get(beta)
                if e is not None and prev_e is not None and e < prev_e - 1e-6:
                    v_list.append(idx)
                prev_e = e
            violations[beta] = v_list
        return {"energy_violations": violations}


def load_ffn_deltas(phase2_run_dir: Path) -> np.ndarray | None:
    """Load FFN deltas from Phase 2 decomposition."""
    for fname in ["ffn_deltas_normed.npz", "ffn_deltas_raw.npz"]:
        path = phase2_run_dir / fname
        if path.exists():
            data = np.load(path)
            key = "ffn_deltas" if "ffn_deltas" in data else list(data.keys())[0]
            return data[key]
    return None


def find_phase1_runs(phase1_dir: Path, model_stem: str) -> list:
    if not phase1_dir.exists():
        return []
    return [
        d for d in sorted(phase1_dir.iterdir())
        if d.is_dir() and model_stem in d.name and (d / "activations.npz").exists()
    ]


def find_phase2_runs(phase2_dir: Path, model_stem: str) -> list:
    if not phase2_dir.exists():
        return []
    return [
        d for d in sorted(phase2_dir.iterdir())
        if d.is_dir() and model_stem in d.name
        and ((d / "ffn_deltas_normed.npz").exists() or (d / "ffn_deltas_raw.npz").exists())
    ]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _write_subresult(sub_dir: Path, name: str, payload: dict, summary_lines: list[str]) -> None:
    """Write sub/{name}.json and sub/{name}.summary.txt."""
    sub_dir.mkdir(parents=True, exist_ok=True)
    with open(sub_dir / f"{name}.json", "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    with open(sub_dir / f"{name}.summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")


def _write_model_summary(model_save: Path, model_stem: str, sections: list[list[str]]) -> None:
    """Assemble and write summary.txt from per-block summary line lists."""
    lines = [
        f"=== Phase 2i: {model_stem} ===",
        f"Generated: {datetime.now().isoformat()}",
        "",
    ]
    for section in sections:
        lines.extend(section)
        lines.append("")
    with open(model_save / "summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Summary-line extractors (one per block)
# ---------------------------------------------------------------------------

def _block1a_summary_lines(rot_json: dict, ov_data: dict) -> list[str]:
    lines = ["--- Block 1a: Rotational spectrum ---"]
    if ov_data["is_per_layer"]:
        depth = rot_json.get("depth_summary", {})
        n_layers = len(ov_data["layer_names"])
        lines.append(f"  Layers: {n_layers}")
        lines.append(f"  Mean θ across layers: {depth.get('theta_mean_across_layers', 0):.4f} rad")
        lines.append(f"  Mean Henrici (relative): {depth.get('henrici_mean', 0):.4f}")
        lines.append(f"  Max Henrici layer: {depth.get('henrici_max_layer', '?')}")
    else:
        lines.append(f"  Real eigenvalues: {rot_json.get('n_real', 0)} "
                     f"({rot_json.get('frac_real', 0):.1%})")
        lines.append(f"  Complex pairs: {rot_json.get('n_complex', 0)} "
                     f"({rot_json.get('frac_complex_dims', 0):.1%} of dims)")
        lines.append(f"  Rotational energy fraction: {rot_json.get('rotational_fraction', 0):.3f}")
        theta_mean = rot_json.get('theta_mean', 0)
        lines.append(f"  Mean θ: {theta_mean:.4f} rad ({np.degrees(theta_mean):.1f}°)")
        lines.append(f"  Henrici (relative): {rot_json.get('henrici_relative', 0):.4f}")
    return lines


def _block1b_summary_lines(
    block1b_results: dict,
    rotation_contributes: bool,
) -> list[str]:
    lines = ["--- Block 1b: Rotational rescaled frame ---"]
    if not block1b_results:
        lines.append("  No Phase 1 activations found. Block 1b skipped.")
        return lines

    for prompt, res in block1b_results.items():
        lines.append(f"  Prompt: {prompt}")
        if "error" in res:
            lines.append(f"    ERROR: {res['error']}")
            continue
        interp = res.get("interpretation", {})
        lines.append(f"    Overall: {interp.get('overall', 'unknown')}")
        frames = res.get("frames", {}).get("comparison", {})
        for beta, comp in frames.items():
            lines.append(
                f"    β={beta}: orig={comp.get('n_original', '?')} "
                f"full={comp.get('n_full_rescaled', '?')} "
                f"S-only={comp.get('n_signed_only', '?')} "
                f"A-only={comp.get('n_rotation_only', '?')}"
            )

    lines.append(
        f"  Rotation contributes: {rotation_contributes}"
        + (" → Block 2 eligible" if rotation_contributes else " → Block 2 skipped")
    )
    return lines


def _block2_summary_lines(block2_results: dict) -> list[str]:
    lines = ["--- Block 2: Hemispheric geometry ---"]
    if not block2_results:
        lines.append("  No results (block skipped or no activations).")
        return lines

    for prompt, res in block2_results.items():
        lines.append(f"  Prompt: {prompt}")
        if "error" in res:
            lines.append(f"    ERROR: {res['error']}")
            continue
        fj = res.get("fiedler", {})
        rh = res.get("rotation_hemisphere", {})
        if isinstance(fj, dict):
            lines.append(f"    Crossing rate mean: {fj.get('crossing_rate_mean', 'N/A')}")
            lines.append(f"    Fiedler cosine mean: {fj.get('fiedler_cosine_mean', 'N/A')}")
            lines.append(f"    Centroid near-π fraction: {fj.get('centroid_angle_near_pi', 'N/A')}")
        if isinstance(rh, dict) and "coherence_mean" in rh:
            lines.append(f"    Displacement coherence: {rh.get('coherence_mean', 'N/A')}")
    return lines


def _ffn_summary_lines(ffn_results: dict) -> list[str]:
    lines = ["--- FFN rotation interaction ---"]
    if not ffn_results:
        lines.append("  No results (skipped or no FFN deltas found).")
        return lines

    for prompt, res in ffn_results.items():
        lines.append(f"  Prompt: {prompt}")
        if "error" in res:
            lines.append(f"    ERROR: {res['error']}")
            continue
        role_counts = res.get("role_counts", {})
        lines.append(f"    Role counts: {role_counts}")
        comp = res.get("comparison", {})
        for metric, data in comp.items():
            if isinstance(data, dict) and "z_score" in data:
                lines.append(
                    f"    {metric}: z={data['z_score']:.2f} "
                    f"(viol={data.get('v_mean', float('nan')):.3f}, "
                    f"pop={data.get('pop_mean', float('nan')):.3f})"
                )
    return lines


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
    Run Phase 2i analysis for one model. Writes per-subexperiment JSON +
    summary.txt files under save_dir/{model_stem}/.

    Returns dict with all results (JSON-serializable via _json_default).
    """
    if beta_values is None:
        beta_values = [0.1, 1.0, 2.0, 5.0]

    model_save = save_dir / model_stem
    model_save.mkdir(parents=True, exist_ok=True)
    sub_dir = model_save / "sub"

    results = {"model": model_stem, "timestamp": datetime.now().isoformat()}
    summary_sections: list[list[str]] = []

    print(f"\n{'='*60}")
    print(f"Phase 2i: {model_stem}")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Load OV data
    # -----------------------------------------------------------------------
    ov_data = load_ov_data(phase2_dir, model_stem)
    if ov_data is None:
        msg = f"No OV weights found for {model_stem} in {phase2_dir}"
        print(f"  ERROR: {msg}")
        return {"model": model_stem, "error": "no_ov_weights"}

    # -----------------------------------------------------------------------
    # Block 1a: Rotational spectrum characterization
    # -----------------------------------------------------------------------
    print(f"\n--- Block 1a: Rotational spectrum ---")
    rot_spectrum = None
    try:
        rot_spectrum = analyze_rotational_spectrum(ov_data, top_k_planes=32)
        rot_json = summary_to_json(rot_spectrum)
        results["block1a"] = rot_json

        b1a_lines = _block1a_summary_lines(rot_json, ov_data)
        for line in b1a_lines:
            print(line)

        _write_subresult(sub_dir, "block1a_rotational_spectrum", rot_json, b1a_lines)
        print(f"  Saved: {sub_dir}/block1a_rotational_spectrum.{{json,summary.txt}}")
        summary_sections.append(b1a_lines)

    except Exception as e:
        print(f"  ERROR in Block 1a: {e}")
        traceback.print_exc()
        err_payload = {"error": str(e)}
        results["block1a"] = err_payload
        b1a_lines = [f"--- Block 1a: Rotational spectrum ---", f"  ERROR: {e}"]
        _write_subresult(sub_dir, "block1a_rotational_spectrum", err_payload, b1a_lines)
        summary_sections.append(b1a_lines)

    # -----------------------------------------------------------------------
    # Block 1b: Rotational rescaled frame
    # -----------------------------------------------------------------------
    print(f"\n--- Block 1b: Rotational rescaled frame ---")
    phase1_runs = find_phase1_runs(phase1_dir, model_stem)
    rotation_contributes = False
    block1b_results = {}

    if not phase1_runs:
        print(f"  No Phase 1 runs found for {model_stem}. Skipping Block 1b.")
        results["block1b"] = {"error": "no_phase1_activations"}
    else:
        for run_dir in phase1_runs:
            prompt = run_dir.name.split("iter_")[-1] if "iter_" in run_dir.name else run_dir.name
            print(f"  Prompt: {prompt}")

            activations = load_activations(run_dir)
            if activations is None:
                print(f"    No activations found. Skipping.")
                continue

            try:
                rescaled = analyze_rotational_rescaling(activations, ov_data, beta_values)
                rescaled_json = comparison_to_json(rescaled)
                block1b_results[prompt] = rescaled_json

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

    b1b_lines = _block1b_summary_lines(block1b_results, rotation_contributes)
    _write_subresult(sub_dir, "block1b_rescaled_comparison", block1b_results, b1b_lines)
    print(f"  Saved: {sub_dir}/block1b_rescaled_comparison.{{json,summary.txt}}")
    summary_sections.append(b1b_lines)

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
        summary_sections.append([
            f"--- Decision gate ---",
            f"  Block 2 skipped: {reason}",
        ])
        _write_model_summary(model_save, model_stem, summary_sections)
        print(f"  Saved: {model_save}/summary.txt")
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
            fiedler = analyze_fiedler_tracking(activations, phase1_events, beta=1.0)
            fiedler_json = fiedler_to_json(fiedler)

            rot_hemi_json: dict
            if rot_spectrum is not None:
                plane_projs = rot_spectrum["plane_projectors"]
                from p2b_imaginary.fiedler_tracking import (
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

            print(f"    Crossing rate mean: {fiedler_json.get('crossing_rate_mean', 'N/A')}")
            print(f"    Fiedler cosine mean: {fiedler_json.get('fiedler_cosine_mean', 'N/A')}")
            print(f"    Centroid near-π fraction: {fiedler_json.get('centroid_angle_near_pi', 'N/A')}")
            if isinstance(rot_hemi_json, dict) and "coherence_mean" in rot_hemi_json:
                print(f"    Displacement coherence: {rot_hemi_json.get('coherence_mean', 'N/A')}")

        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            block2_results[prompt] = {"error": str(e)}

    results["block2"] = block2_results
    b2_lines = _block2_summary_lines(block2_results)
    _write_subresult(sub_dir, "block2_hemispheric", block2_results, b2_lines)
    print(f"  Saved: {sub_dir}/block2_hemispheric.{{json,summary.txt}}")
    summary_sections.append(b2_lines)

    # -----------------------------------------------------------------------
    # FFN rotation interaction (per-layer models only, if deltas available)
    # -----------------------------------------------------------------------
    ffn_results = {}
    if not skip_ffn and ov_data["is_per_layer"] and rot_spectrum is not None:
        print(f"\n--- FFN rotation analysis ---")

        phase2_runs = find_phase2_runs(phase2_dir, model_stem)
        for run_dir in phase2_runs:
            prompt = run_dir.name.split("iter_")[-1] if "iter_" in run_dir.name else run_dir.name
            print(f"  Prompt: {prompt}")

            ffn_deltas = load_ffn_deltas(run_dir)
            if ffn_deltas is None:
                print(f"    No FFN deltas found. Skipping.")
                continue

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
                for metric, data in ffn_rot["comparison"].items():
                    if isinstance(data, dict) and "z_score" in data:
                        print(f"    {metric}: z={data['z_score']:.2f} "
                              f"(viol={data['v_mean']:.3f}, pop={data['pop_mean']:.3f})")

            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                ffn_results[prompt] = {"error": str(e)}

        results["ffn_rotation"] = ffn_results

    ffn_lines = _ffn_summary_lines(ffn_results)
    _write_subresult(sub_dir, "ffn_rotation", ffn_results, ffn_lines)
    print(f"  Saved: {sub_dir}/ffn_rotation.{{json,summary.txt}}")
    summary_sections.append(ffn_lines)

    # -----------------------------------------------------------------------
    # Per-model summary.txt
    # -----------------------------------------------------------------------
    _write_model_summary(model_save, model_stem, summary_sections)
    print(f"  Saved: {model_save}/summary.txt")

    return results


# ---------------------------------------------------------------------------
# JSON default handler
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
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
    parser = argparse.ArgumentParser(description="Phase 2i: Rotational dynamics analysis")
    parser.add_argument("--phase2-dir", type=str, required=True)
    parser.add_argument("--phase1-dir", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--block1-only", action="store_true")
    parser.add_argument("--force-block2", action="store_true")
    parser.add_argument("--skip-ffn", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    phase2_dir = Path(args.phase2_dir)
    phase1_dir = Path(args.phase1_dir)

    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = Path("results") / f"phase2i_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        model_stems = args.models
    else:
        model_stems = [
            f.stem.replace("ov_weights_", "")
            for f in sorted(phase2_dir.glob("ov_weights_*.npz"))
        ]

    if not model_stems:
        print(f"No models found in {phase2_dir}")
        sys.exit(1)

    print(f"Phase 2i: Rotational Dynamics Analysis")
    print(f"Models: {model_stems}")
    print(f"Output: {save_dir}")
    print(f"Block 1 only: {args.block1_only}  Force Block 2: {args.force_block2}")

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

    # -----------------------------------------------------------------------
    # Combined outputs
    # -----------------------------------------------------------------------
    with open(save_dir / "phase2i_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"\nSaved: {save_dir}/phase2i_results.json")

    # Assemble phase2i_summary.txt by stitching per-model summary.txt files
    combined_lines = [
        "=== Phase 2i: Rotational Dynamics Investigation ===",
        f"Generated: {datetime.now().isoformat()}",
        f"Models: {', '.join(model_stems)}",
        "",
    ]
    for model_stem in model_stems:
        model_summary = save_dir / model_stem / "summary.txt"
        if model_summary.exists():
            combined_lines.append(model_summary.read_text())
        else:
            res = all_results.get(model_stem, {})
            err = res.get("error", "unknown error")
            combined_lines.append(f"=== {model_stem} ===\n  ERROR: {err}\n")
        combined_lines.append("")

    # Append overall verdict table
    combined_lines.append("=== Overall verdict ===")
    for stem, res in all_results.items():
        contributes = res.get("rotation_contributes", "unknown")
        decision = res.get("block2_decision", "unknown")
        combined_lines.append(
            f"  {stem}: rotation_contributes={contributes}, block2={decision}"
        )

    with open(save_dir / "phase2i_summary.txt", "w") as f:
        f.write("\n".join(combined_lines) + "\n")
    print(f"Saved: {save_dir}/phase2i_summary.txt")

    # CLI summary
    print(f"\n{'='*60}")
    print("PHASE 2i SUMMARY")
    print(f"{'='*60}")
    for stem, res in all_results.items():
        contributes = res.get("rotation_contributes", "unknown")
        decision = res.get("block2_decision", "unknown")
        print(f"  {stem}: rotation_contributes={contributes}, block2={decision}")


if __name__ == "__main__":
    main()
