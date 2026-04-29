"""
run_2c.py — p2c_churchland CLI entry point.

Runs all five analysis tracks against Phase 1 + Phase 2i artifacts and writes
every result as its own JSON file plus a flat LLM-friendly summary.txt.

Usage
-----
    python -m p2c_churchland.run_2c \\
        --phase1-dir results/phase1_... \\
        --phase2-dir results/phase2i_... \\
        [--output-dir results/phase2c_...] \\
        [--model albert-xlarge-v2] \\
        [--device cpu] \\
        [--prompt-grids-dir p2c_churchland/prompt_grids] \\
        [--skip-c2] [--skip-c4] [--skip-c5]   # offline-only run

Tracks and their artifact requirements
---------------------------------------
  C1 jPCA / alignment / HDR  : centroid_trajectories.npz  (no model)
  C2 tangling                : model + prompts OR saved activations
  C3 CIS decomposition       : matched-prompt activations  (model if not saved)
  C4 local Jacobians         : model + Phase 1 centroids
  C5 ICL scaling + Mante     : model + prompt_grids/{icl_kshot,context_pairs}.json

Output layout
-------------
  <output_dir>/
    c1_jpca.json
    c1_jpca_alignment.json
    c1_hdr.json                (only when jPCA is marginal and --skip-hdr not set)
    c2_tangling.json
    c3_cis.json
    c4_local_jacobians.json
    c4_slow_point_compare.json
    c5_icl_scaling.json
    c5_context_selection.json
    summary.txt                (LLM-friendly flat text, all verdicts)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _jdump(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"  wrote {path.name}")


def _jload(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phase 1 artifact loading
# ---------------------------------------------------------------------------

def load_phase1_artifacts(phase1_dir: Path) -> dict:
    """
    Load centroid trajectories, cluster metadata, and phase event layers.

    Returns
    -------
    dict with:
      traj_dict       : {key: (n_layers, d)} — all centroid trajectories
      traj_stacked    : (n_cond, n_layers, d) — for jPCA / CIS
      traj_keys       : list[str]
      plateau_layers  : list[int]
      merge_layers    : list[int]
      metrics         : raw metrics dict (or {})
    """
    arts: dict = {
        "traj_dict":    {},
        "traj_stacked": None,
        "traj_keys":    [],
        "plateau_layers": [],
        "merge_layers":   [],
        "metrics":        {},
    }

    # Centroid trajectories
    ct_path = phase1_dir / "centroid_trajectories.npz"
    if ct_path.exists():
        npz = np.load(ct_path)
        arts["traj_dict"] = {k: npz[k] for k in npz.files}
        arts["traj_keys"] = sorted(arts["traj_dict"])
        if arts["traj_keys"]:
            arts["traj_stacked"] = np.stack(
                [arts["traj_dict"][k] for k in arts["traj_keys"]], axis=0
            )
        print(f"  loaded centroid_trajectories.npz  "
              f"({len(arts['traj_keys'])} trajectories)")
    else:
        print(f"  WARNING: centroid_trajectories.npz not found in {phase1_dir}")

    # Metrics / event layers
    for fname in ("metrics.json", "phase1_metrics.json"):
        mp = phase1_dir / fname
        if mp.exists():
            arts["metrics"] = _jload(mp)
            break

    m = arts["metrics"]
    arts["plateau_layers"] = _flatten_layers(m.get("plateau_layers") or
                                             m.get("plateau_layer_indices") or [])
    arts["merge_layers"]   = _flatten_layers(m.get("merge_layers") or
                                             m.get("merge_layer_indices") or [])
    print(f"  plateau layers: {arts['plateau_layers'][:8]}")
    print(f"  merge   layers: {arts['merge_layers'][:8]}")
    return arts


def _flatten_layers(v: Any) -> list[int]:
    """Accept int list, dict-keyed lists, or nested lists → flat int list."""
    if isinstance(v, list):
        out: list[int] = []
        for x in v:
            if isinstance(x, (int, float)):
                out.append(int(x))
            elif isinstance(x, list):
                out.extend(int(i) for i in x)
        return out
    if isinstance(v, dict):
        out = []
        for val in v.values():
            out.extend(_flatten_layers(val))
        return sorted(set(out))
    return []


def load_per_prompt_activations(
    phase1_dir: Path,
    prompt_keys: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Load per-prompt per-layer per-token activations from Phase 1 subdirectories.
    Returns {prompt_key: (n_layers, n_tokens, d)} where available.
    """
    acts: dict[str, np.ndarray] = {}
    for sub in sorted(phase1_dir.iterdir()):
        if not sub.is_dir():
            continue
        key = sub.name
        if prompt_keys and key not in prompt_keys:
            continue
        for fname in ("activations.npz", "hidden_states.npz"):
            p = sub / fname
            if p.exists():
                npz = np.load(p)
                # Shape heuristic: take "hidden_states" or first array key
                candidate = None
                for k in ("hidden_states", "activations", npz.files[0]):
                    if k in npz.files:
                        candidate = npz[k]
                        break
                if candidate is not None and candidate.ndim == 3:
                    acts[key] = candidate
                break
    if acts:
        print(f"  loaded activations for {len(acts)} prompts from {phase1_dir}")
    return acts


# ---------------------------------------------------------------------------
# Phase 2i artifact loading (Schur blocks, P_A, P_S)
# ---------------------------------------------------------------------------

def load_phase2_artifacts(phase2_dir: Path, model_name: str) -> dict:
    """
    Load Schur rotation projectors from Phase 2i output.

    Tries in order:
      1. {phase2_dir}/{model_stem}/projectors.npz  (explicit save)
      2. {phase2_dir}/{model_stem}.json            (JSON summary, recompute)
      3. {phase2_dir}/weights/{model_stem}_*.npz   (Phase 2 OV matrices)

    Returns dict with P_A, P_S, ua_planes, global_sa_ratio (None on failure).
    """
    arts: dict = {
        "P_A": None, "P_S": None,
        "ua_planes": [],
        "global_sa_ratio": None,
        "schur_blocks": None,
    }
    if phase2_dir is None:
        return arts

    model_stem = _model_stem(model_name)

    # Path 1: explicit projectors NPZ
    proj_path = phase2_dir / model_stem / "projectors.npz"
    if proj_path.exists():
        npz = np.load(proj_path)
        arts["P_A"] = npz.get("P_A")
        arts["P_S"] = npz.get("P_S")
        print(f"  loaded projectors from {proj_path}")
        return arts

    # Path 2: recompute from OV matrix in weights/
    ov_matrix = _find_ov_matrix(phase2_dir, model_stem)
    if ov_matrix is not None:
        try:
            from p2b_imaginary.rotational_schur import (
                extract_schur_blocks,
                build_rotation_plane_projectors,
            )
            blocks = extract_schur_blocks(ov_matrix)
            arts["schur_blocks"] = blocks
            planes_info = build_rotation_plane_projectors(blocks, top_k=8)
            arts["P_A"]      = planes_info.get("P_A")
            arts["P_S"]      = planes_info.get("P_S")
            arts["ua_planes"] = planes_info.get("top_k_planes", [])
            # global S/A ratio: fraction of spectral energy in antisymmetric
            from p2b_imaginary.rotational_schur import rotation_energy_fractions
            fracs = rotation_energy_fractions(blocks)
            arts["global_sa_ratio"] = float(
                fracs.get("rotation_fraction", 0.5) /
                max(fracs.get("signed_fraction", 1.0), 1e-12)
            )
            print(f"  recomputed P_A / P_S from OV matrix  "
                  f"(d={ov_matrix.shape[0]})")
        except Exception as e:
            print(f"  WARNING: could not build projectors: {e}")
        return arts

    # Path 3: JSON summary (scalars only, no matrices)
    json_path = phase2_dir / f"{model_stem}.json"
    if not json_path.exists():
        for sub in phase2_dir.iterdir():
            if sub.is_dir() and model_stem in sub.name:
                candidate = sub / "results.json"
                if candidate.exists():
                    json_path = candidate
                    break
    if json_path.exists():
        data = _jload(json_path)
        arts["global_sa_ratio"] = (
            data.get("global_sa_ratio") or
            data.get("rotation_fraction")
        )
        print(f"  loaded phase2i JSON (no matrices; projector-dependent "
              f"analyses may be skipped)")
    else:
        print(f"  WARNING: no phase2 artifacts found for {model_name} in {phase2_dir}")

    return arts


def _find_ov_matrix(phase2_dir: Path, model_stem: str) -> np.ndarray | None:
    """Search phase2_dir for an OV / V_eff matrix NPZ."""
    candidates = [
        phase2_dir / "weights" / f"{model_stem}_decomp.npz",
        phase2_dir / "weights" / f"{model_stem}_ov.npz",
        phase2_dir / f"{model_stem}_decomp.npz",
    ]
    for path in candidates:
        if path.exists():
            npz = np.load(path)
            for key in ("V_eff", "ov_total", "ov_matrix", "V"):
                if key in npz.files:
                    return npz[key]
    # Scan weights/ for any matching npz with a V_eff key
    weights_dir = phase2_dir / "weights"
    if weights_dir.is_dir():
        for p in sorted(weights_dir.glob(f"{model_stem}*.npz")):
            npz = np.load(p)
            for key in ("V_eff", "ov_total", "ov_matrix", "V"):
                if key in npz.files:
                    return npz[key]
    return None


def _model_stem(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, device: str):
    """Load HuggingFace model + tokenizer.  Returns (model, tokenizer)."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        ).to(device).eval()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  loaded {model_name}  ({n_params:.0f}M params) on {device}")
        return model, tokenizer
    except Exception as e:
        print(f"  ERROR loading model {model_name}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Track runners
# ---------------------------------------------------------------------------

def run_c1(
    traj_stacked: np.ndarray,
    phase2_arts: dict,
    output_dir: Path,
    args,
) -> dict:
    """
    C1 — jPCA fit on centroid trajectories, U_A alignment, optional HDR.
    Fully offline: only needs centroid_trajectories.npz + phase2 projectors.
    """
    from p2c_churchland.jpca_fit import fit_jpca, jpca_to_json
    from p2c_churchland.hdr_fit  import fit_hdr, hdr_to_json

    results: dict = {}

    if traj_stacked is None or traj_stacked.ndim != 3:
        print("  C1 SKIP: no stacked centroid trajectories available")
        return results

    n_cond, n_layers, d = traj_stacked.shape
    n_pc = min(6, n_cond - 1, n_layers - 1, d)
    if n_pc < 2:
        print(f"  C1 SKIP: too few conditions or layers "
              f"(n_cond={n_cond}, n_layers={n_layers})")
        return results

    print("  running jPCA ...")
    jpca_result = fit_jpca(traj_stacked, n_pc=n_pc)
    _jdump(jpca_to_json(jpca_result), output_dir / "c1_jpca.json")
    results["c1_jpca"] = jpca_result

    # J2 alignment against U_A
    ua_planes = phase2_arts.get("ua_planes", [])
    if ua_planes:
        print("  running jPCA → U_A alignment ...")
        try:
            from p2c_churchland.jpca_alignment import align_jpca_to_ua, jpca_alignment_to_json
            align_result = align_jpca_to_ua(jpca_result, ua_planes)
            _jdump(jpca_alignment_to_json(align_result),
                   output_dir / "c1_jpca_alignment.json")
            results["c1_jpca_alignment"] = align_result
        except Exception as e:
            print(f"  C1 alignment ERROR: {e}")
    else:
        print("  C1 J2: no U_A planes available (phase2_dir missing or no OV matrix)")

    # HDR fallback when jPCA is borderline
    if jpca_result.get("p2cj1_marginal") and not args.skip_hdr:
        print("  jPCA marginal — running HDR ...")
        hdr_result = fit_hdr(
            traj_stacked, n_pc=n_pc,
            ua_planes=ua_planes if ua_planes else None,
        )
        _jdump(hdr_to_json(hdr_result), output_dir / "c1_hdr.json")
        results["c1_hdr"] = hdr_result

    return results


def run_c2(
    model, tokenizer,
    phase1_dir: Path,
    phase1_arts: dict,
    phase2_arts: dict,
    prompts: list[str],
    output_dir: Path,
    args,
) -> dict:
    """C2 — Trajectory tangling (Q metric).  Needs live model or saved activations."""
    from p2c_churchland.tangling import analyze_tangling, tangling_to_json

    P_A = phase2_arts.get("P_A")
    P_S = phase2_arts.get("P_S")

    if P_A is None or P_S is None:
        print("  C2 SKIP: P_A / P_S projectors not available")
        return {}
    if not prompts:
        print("  C2 SKIP: no prompts provided")
        return {}

    # Prefer saved activations if model unavailable
    if model is None:
        print("  C2: no model — looking for saved per-prompt activations ...")
        saved = load_per_prompt_activations(phase1_dir)
        if not saved:
            print("  C2 SKIP: no model and no saved activations")
            return {}
        # analyze_tangling needs model; with saved acts, use low-level path
        # Pass activations directly (analyze_tangling accepts pre-extracted
        # activations when model=None and activations_dict is provided)
        try:
            result = analyze_tangling(
                model=None,
                tokenizer=None,
                prompts=prompts,
                P_A=P_A,
                P_S=P_S,
                device=args.device,
                activations_dict=saved,
            )
        except TypeError:
            # If analyze_tangling doesn't accept activations_dict, skip
            print("  C2 SKIP: saved-activation path not supported by analyze_tangling")
            return {}
    else:
        print("  running tangling ...")
        result = analyze_tangling(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            P_A=P_A,
            P_S=P_S,
            device=args.device,
        )

    _jdump(tangling_to_json(result), output_dir / "c2_tangling.json")
    return {"c2_tangling": result}


def run_c3(
    model, tokenizer,
    phase1_dir: Path,
    phase1_arts: dict,
    phase2_arts: dict,
    matched_prompts: list[dict] | None,
    output_dir: Path,
    args,
) -> dict:
    """C3 — CIS decomposition.  Needs matched-prompt activations."""
    from p2c_churchland.cis_decompose import analyze_cis, cis_to_json

    P_A = phase2_arts.get("P_A")
    P_S = phase2_arts.get("P_S")
    if P_A is None or P_S is None:
        print("  C3 SKIP: P_A / P_S not available")
        return {}

    if not matched_prompts:
        print("  C3 SKIP: no matched-length prompts (provide --prompt-grids-dir)")
        return {}

    # Collect activations: try saved first, then run model
    prompt_texts = [p["prompt"] for p in matched_prompts]
    saved_acts = load_per_prompt_activations(
        phase1_dir,
        prompt_keys=[p.get("key") for p in matched_prompts if p.get("key")],
    )

    activations_list: list[np.ndarray] = []
    for entry in matched_prompts:
        key = entry.get("key", entry["prompt"][:40])
        if key in saved_acts:
            activations_list.append(saved_acts[key])
        elif model is not None:
            # Extract from live model
            from p2c_churchland.tangling import extract_full_activations
            acts = extract_full_activations(
                model, tokenizer, entry["prompt"], device=args.device
            )
            activations_list.append(acts)
        else:
            print(f"  C3: missing activations for '{key}' — skipping prompt")

    if len(activations_list) < 2:
        print("  C3 SKIP: fewer than 2 matched-prompt activations available")
        return {}

    # Ensure all arrays share the same shape
    min_layers  = min(a.shape[0] for a in activations_list)
    min_tokens  = min(a.shape[1] for a in activations_list)
    activations_list = [a[:min_layers, :min_tokens, :] for a in activations_list]

    print(f"  running CIS decomposition ({len(activations_list)} prompts) ...")
    result = analyze_cis(
        activations_per_prompt=activations_list,
        P_A=P_A,
        P_S=P_S,
        plateau_layers=phase1_arts["plateau_layers"] or None,
        merge_layers=phase1_arts["merge_layers"] or None,
    )
    _jdump(cis_to_json(result), output_dir / "c3_cis.json")
    return {"c3_cis": result}


def run_c4(
    model, tokenizer,
    phase1_arts: dict,
    phase2_arts: dict,
    model_name: str,
    output_dir: Path,
    args,
) -> dict:
    """C4 — Local Jacobians at Phase 1 centroids + slow-point comparison."""
    from p2c_churchland.local_jacobian   import analyze_local_jacobians
    from p2c_churchland.slow_point_compare import (
        compare_local_global,
        layer_sa_profile,
        plateau_vs_merge_table,
    )

    if model is None:
        print("  C4 SKIP: model not loaded")
        return {}

    traj_dict = phase1_arts["traj_dict"]
    if not traj_dict:
        print("  C4 SKIP: no centroid trajectories")
        return {}

    # Build centroids_per_layer: {layer_idx: (n_centroids, d)}
    # Each trajectory key: (n_layers, d); layer axis = 0
    sample = next(iter(traj_dict.values()))
    n_layers = sample.shape[0]
    centroids_per_layer: dict[int, np.ndarray] = {}
    for li in range(n_layers):
        vecs = np.stack([v[li] for v in traj_dict.values()], axis=0)
        centroids_per_layer[li] = vecs

    model_type = "albert" if "albert" in model_name.lower() else "gpt2"
    P_A = phase2_arts.get("P_A")
    P_S = phase2_arts.get("P_S")
    global_sa = phase2_arts.get("global_sa_ratio") or 1.0

    print(f"  running local Jacobians "
          f"({n_layers} layers × {len(traj_dict)} centroids) ...")
    jac_result = analyze_local_jacobians(
        model=model,
        model_type=model_type,
        centroids_per_layer=centroids_per_layer,
        plateau_layers=phase1_arts["plateau_layers"],
        merge_layers=phase1_arts["merge_layers"],
        global_sa_ratio=global_sa,
        P_A=P_A,
        P_S=P_S,
        device=args.device,
    )
    _jdump(_serialize_jac(jac_result), output_dir / "c4_local_jacobians.json")

    print("  running slow-point comparison ...")
    per_layer = jac_result.get("per_layer", {})
    sp_result = compare_local_global(
        per_layer=per_layer,
        plateau_layers=phase1_arts["plateau_layers"],
        merge_layers=phase1_arts["merge_layers"],
        global_sa_ratio=global_sa,
    )
    _jdump(_serialize_sp(sp_result), output_dir / "c4_slow_point_compare.json")

    return {"c4_local_jacobians": jac_result, "c4_slow_point_compare": sp_result}


def _serialize_jac(r: dict) -> dict:
    """Drop large Jacobian matrices for serialization."""
    out: dict = {}
    for k, v in r.items():
        if k == "per_layer":
            out["per_layer"] = {
                str(li): [
                    {kk: (float(vv) if isinstance(vv, (float, np.floating))
                          else (int(vv) if isinstance(vv, (int, np.integer))
                                else None))
                     for kk, vv in rec.items() if kk != "J"}
                    for rec in recs
                ]
                for li, recs in v.items()
            }
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (float, np.floating)):
            out[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _serialize_sp(r: dict) -> dict:
    return json.loads(json.dumps(r, default=_json_default))


def run_c5(
    model, tokenizer,
    phase2_arts: dict,
    icl_prompts: list[dict] | None,
    context_pairs: list[dict] | None,
    output_dir: Path,
    args,
) -> dict:
    """C5 — ICL subspace scaling (M1/M2) + context-selection divergence (M3)."""
    if model is None:
        print("  C5 SKIP: model not loaded")
        return {}

    P_A = phase2_arts.get("P_A")
    P_S = phase2_arts.get("P_S")
    if P_A is None or P_S is None:
        print("  C5 SKIP: P_A / P_S not available")
        return {}

    results: dict = {}

    # M1 / M2 — ICL k-shot scaling
    if icl_prompts:
        from p2c_churchland.icl_subspace_scaling import (
            analyze_icl_scaling, icl_scaling_to_json,
        )
        # icl_prompts format: {task_name: [{k, prompt, answer_idx}, ...]}
        task_sets = _group_icl_by_task(icl_prompts)
        if task_sets:
            print(f"  running ICL scaling ({len(task_sets)} tasks) ...")
            icl_result = analyze_icl_scaling(
                model=model,
                tokenizer=tokenizer,
                task_prompt_sets=task_sets,
                P_A=P_A,
                P_S=P_S,
                device=args.device,
            )
            _jdump(icl_scaling_to_json(icl_result),
                   output_dir / "c5_icl_scaling.json")
            results["c5_icl_scaling"] = icl_result
    else:
        print("  C5 M1/M2 SKIP: no ICL prompt grid (provide --prompt-grids-dir)")

    # M3 — context-selection divergence (Mante analog)
    if context_pairs:
        from p2c_churchland.context_selection import analyze_context_pair
        print(f"  running context-selection divergence ({len(context_pairs)} pairs) ...")
        pair_results = []
        for pair in context_pairs:
            try:
                r = analyze_context_pair(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_a=pair["prompt_a"],
                    prompt_b=pair["prompt_b"],
                    P_A=P_A,
                    P_S=P_S,
                    answer_idx_a=pair.get("answer_idx_a"),
                    answer_idx_b=pair.get("answer_idx_b"),
                    device=args.device,
                )
                r["pair_label"] = pair.get("label", f"pair_{len(pair_results)}")
                pair_results.append(r)
            except Exception as e:
                print(f"    pair '{pair.get('label', '?')}' failed: {e}")

        if pair_results:
            cs_summary = _aggregate_context_pairs(pair_results)
            _jdump(cs_summary, output_dir / "c5_context_selection.json")
            results["c5_context_selection"] = cs_summary
    else:
        print("  C5 M3 SKIP: no context-pairs file (provide --prompt-grids-dir)")

    return results


def _group_icl_by_task(icl_prompts) -> dict:
    """Accept list-of-dicts or dict-of-lists."""
    if isinstance(icl_prompts, dict):
        return icl_prompts
    grouped: dict = {}
    for entry in icl_prompts:
        task = entry.get("task", "default")
        grouped.setdefault(task, []).append(entry)
    return grouped


def _aggregate_context_pairs(pair_results: list[dict]) -> dict:
    mean_S = float(np.mean([r["mean_S_cosine"] for r in pair_results]))
    mean_A = float(np.mean([r["mean_A_cosine"] for r in pair_results]))
    n_m3   = sum(1 for r in pair_results if r.get("m3_holds", False))
    return {
        "n_pairs":       len(pair_results),
        "mean_S_cosine": mean_S,
        "mean_A_cosine": mean_A,
        "p2cm3_holds":   mean_S > mean_A,
        "n_pairs_m3_holds": n_m3,
        "per_pair": [
            {
                "label":        r.get("pair_label", "?"),
                "mean_S_cosine": float(r.get("mean_S_cosine", float("nan"))),
                "mean_A_cosine": float(r.get("mean_A_cosine", float("nan"))),
                "m3_holds":      bool(r.get("m3_holds", False)),
            }
            for r in pair_results
        ],
    }


# ---------------------------------------------------------------------------
# Prompt grid loading
# ---------------------------------------------------------------------------

def load_prompt_grids(grids_dir: Path) -> dict:
    """Load matched_length.json, icl_kshot.json, context_pairs.json."""
    grids: dict = {
        "matched_prompts": None,
        "icl_prompts":     None,
        "context_pairs":   None,
    }
    mapping = {
        "matched_length.json": "matched_prompts",
        "icl_kshot.json":      "icl_prompts",
        "context_pairs.json":  "context_pairs",
    }
    if grids_dir is None or not grids_dir.is_dir():
        return grids
    for fname, key in mapping.items():
        p = grids_dir / fname
        if p.exists():
            grids[key] = _jload(p)
            print(f"  loaded {fname}")
        else:
            print(f"  prompt grid '{fname}' not found in {grids_dir}")
    return grids


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

SEP_MAJOR = "=" * 68
SEP_MINOR = "-" * 60

_VERDICT_KEYS = [
    ("c1_jpca",              "p2cj1_holds",  "P2c-J1",
     "jPCA R² ratio > 0.5"),
    ("c1_jpca_alignment",    "p2cj2_holds",  "P2c-J2",
     "top jPCA planes within 30° of U_A"),
    ("c1_hdr",               "p2cj1_hdr_holds", "P2c-J1-HDR",
     "HDR variance ratio > 0.5 (fallback)"),
    ("c2_tangling",          "p2ct1_holds",  "P2c-T1",
     "A-channel Q < S-channel Q"),
    ("c2_tangling",          "p2ct2_holds",  "P2c-T2",
     "induction prompts lower tangling than control"),
    ("c3_cis",               "p2ck1_holds",  "P2c-K1",
     "invariant variance in A, specific in S"),
    ("c3_cis",               "p2ck2_holds",  "P2c-K2",
     "invariant velocity spike at merge layers"),
    ("c4_slow_point_compare","p2cs1_holds",  "P2c-S1",
     "plateau Jacobians more symmetric than V"),
    ("c4_slow_point_compare","p2cs2_holds",  "P2c-S2",
     "merge Jacobians less symmetric than plateau"),
    ("c5_icl_scaling",       "p2cm1_holds",  "P2c-M1",
     "A-channel magnitude monotone with k"),
    ("c5_icl_scaling",       "p2cm2_holds",  "P2c-M2",
     "A-channel direction is task-specific"),
    ("c5_context_selection", "p2cm3_holds",  "P2c-M3",
     "context-paired prompts diverge in A, agree in S"),
]


def write_summary(
    all_results: dict,
    output_dir: Path,
    model_name: str,
    phase1_dir: Path,
    phase2_dir: Path | None,
    args,
) -> None:
    lines = [
        SEP_MAJOR,
        "P2C_CHURCHLAND SUMMARY",
        f"model:      {model_name}",
        f"phase1_dir: {phase1_dir}",
        f"phase2_dir: {phase2_dir or 'not provided'}",
        f"output_dir: {output_dir}",
        SEP_MAJOR,
        "",
    ]

    # Per-track sections
    _track_sections(lines, all_results)

    # Falsification table
    lines += ["", SEP_MAJOR, "FALSIFICATION TABLE", SEP_MAJOR]
    for result_key, verdict_key, pred_id, description in _VERDICT_KEYS:
        r = all_results.get(result_key)
        if r is None:
            status = "NOT RUN"
            extra  = ""
        else:
            val = _deep_get(r, verdict_key)
            if val is None:
                status = "NO DATA"
                extra  = ""
            else:
                status = "HOLDS" if val else "FAILS"
                extra  = _extra_scalar(r, result_key)
        lines.append(f"  {pred_id:<16} {status:<8}  {description}")
        if extra:
            lines.append(f"  {'':<16}           {extra}")

    lines += ["", SEP_MAJOR]

    txt = "\n".join(lines)
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(txt)
    print(f"\n  wrote summary.txt → {summary_path}")


def _track_sections(lines: list, all_results: dict) -> None:
    track_map = [
        ("c1_jpca",               "C1 jPCA",
         [("r2_ratio", "R² ratio"), ("r2_skew", "R² skew"),
          ("r2_unc", "R² unc"), ("n_cond", "n_conditions")]),
        ("c1_jpca_alignment",     "C1 jPCA alignment",
         [("mean_min_angle", "mean min angle (°)"),
          ("angle_distribution", "distribution")]),
        ("c1_hdr",                "C1 HDR (fallback)",
         [("variance_ratio", "variance ratio"),
          ("ua_min_angle", "U_A min angle (°)")]),
        ("c2_tangling",           "C2 tangling",
         [("mean_Q_full", "mean Q full"), ("mean_Q_A", "mean Q A"),
          ("mean_Q_S", "mean Q S")]),
        ("c3_cis",                "C3 CIS",
         [("global_inv_frac_A", "inv frac in A"),
          ("global_spec_frac_S", "spec frac in S")]),
        ("c4_slow_point_compare", "C4 slow-point comparison",
         [("plateau_mean_sa", "plateau mean S/A"),
          ("merge_mean_sa", "merge mean S/A"),
          ("global_sa_ratio", "global S/A")]),
        ("c5_icl_scaling",        "C5 ICL scaling",
         [("mean_rho_A", "mean ρ_A"), ("mean_rho_S", "mean ρ_S")]),
        ("c5_context_selection",  "C5 context selection",
         [("mean_S_cosine", "mean S cosine"),
          ("mean_A_cosine", "mean A cosine"),
          ("n_pairs", "n pairs")]),
    ]
    for result_key, title, fields in track_map:
        r = all_results.get(result_key)
        if r is None:
            continue
        lines += [SEP_MINOR, title, SEP_MINOR]
        for field, label in fields:
            v = _deep_get(r, field)
            if v is not None:
                if isinstance(v, float):
                    lines.append(f"  {label}: {v:.4f}")
                else:
                    lines.append(f"  {label}: {v}")
        lines.append("")


def _deep_get(d: dict, key: str):
    """Walk nested dict for a key, checking top level and one level of nesting."""
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict) and key in v:
            return v[key]
    return None


def _extra_scalar(r: dict, result_key: str) -> str:
    extras = {
        "c1_jpca":   ("r2_ratio", "r²_ratio={:.3f}"),
        "c1_hdr":    ("variance_ratio", "variance_ratio={:.3f}"),
        "c2_tangling": ("mean_Q_A",   "mean_Q_A={:.3f}"),
        "c3_cis":    ("global_inv_frac_A", "inv_frac_A={:.3f}"),
        "c5_icl_scaling": ("mean_rho_A", "ρ_A={:.3f}"),
    }
    if result_key in extras:
        field, fmt = extras[result_key]
        v = _deep_get(r, field)
        if v is not None and isinstance(v, float):
            return fmt.format(v)
    return ""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_phase(model_name: str, args) -> dict:
    phase1_dir = Path(args.phase1_dir)
    phase2_dir = Path(args.phase2_dir) if args.phase2_dir else None

    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = _model_stem(model_name)
    out = Path(args.output_dir) if args.output_dir else \
          Path("results") / f"p2c_{stem}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n  model: {model_name}\n  output: {out}\n{'='*60}")

    # Load artifacts
    print("\n[artifacts]")
    p1 = load_phase1_artifacts(phase1_dir)
    p2 = load_phase2_artifacts(phase2_dir, model_name) if phase2_dir else {
        "P_A": None, "P_S": None, "ua_planes": [], "global_sa_ratio": None,
    }

    # Load prompt grids
    grids_dir = Path(args.prompt_grids_dir) if args.prompt_grids_dir else \
                Path(__file__).parent / "prompt_grids"
    grids = load_prompt_grids(grids_dir)

    # Load model (only if any live track needed)
    need_model = not all([args.skip_c2, args.skip_c4, args.skip_c5])
    model, tokenizer = None, None
    if need_model and not args.no_model:
        print("\n[model]")
        model, tokenizer = load_model_and_tokenizer(model_name, args.device)

    # Derive prompts for C2 tangling
    prompts_for_c2: list[str] = []
    if p1["traj_keys"]:
        # Use trajectory keys as prompt text proxies if real prompts not stored
        prompts_for_c2 = [k.split("__traj_")[0].replace("_", " ")
                          for k in p1["traj_keys"]][:16]

    all_results: dict = {}

    # C1 — offline
    if not args.skip_c1:
        print("\n[C1 jPCA]")
        try:
            all_results.update(
                run_c1(p1["traj_stacked"], p2, out, args)
            )
        except Exception:
            print(f"  C1 FAILED:\n{traceback.format_exc()}")

    # C2 — tangling
    if not args.skip_c2:
        print("\n[C2 tangling]")
        try:
            all_results.update(
                run_c2(model, tokenizer, phase1_dir, p1, p2,
                       prompts_for_c2, out, args)
            )
        except Exception:
            print(f"  C2 FAILED:\n{traceback.format_exc()}")

    # C3 — CIS
    if not args.skip_c3:
        print("\n[C3 CIS]")
        try:
            all_results.update(
                run_c3(model, tokenizer, phase1_dir, p1, p2,
                       grids["matched_prompts"], out, args)
            )
        except Exception:
            print(f"  C3 FAILED:\n{traceback.format_exc()}")

    # C4 — local Jacobians
    if not args.skip_c4:
        print("\n[C4 local Jacobians]")
        try:
            all_results.update(
                run_c4(model, tokenizer, p1, p2, model_name, out, args)
            )
        except Exception:
            print(f"  C4 FAILED:\n{traceback.format_exc()}")

    # C5 — ICL + Mante
    if not args.skip_c5:
        print("\n[C5 ICL + context selection]")
        try:
            all_results.update(
                run_c5(model, tokenizer, p2,
                       grids["icl_prompts"], grids["context_pairs"],
                       out, args)
            )
        except Exception:
            print(f"  C5 FAILED:\n{traceback.format_exc()}")

    # Summary
    print("\n[summary]")
    write_summary(all_results, out, model_name, phase1_dir, phase2_dir, args)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2c: Trajectory-Side Dynamical Systems Analysis"
    )
    parser.add_argument("--phase1-dir",  type=str, required=True,
                        help="Phase 1 output directory")
    parser.add_argument("--phase2-dir",  type=str, default=None,
                        help="Phase 2i output directory (Schur projectors)")
    parser.add_argument("--output-dir",  type=str, default=None,
                        help="Results output directory (auto-timestamped if omitted)")
    parser.add_argument("--model",       type=str, default=None,
                        help="HuggingFace model name (auto-detected from phase1 if omitted)")
    parser.add_argument("--device",      type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Torch device")
    parser.add_argument("--prompt-grids-dir", type=str, default=None,
                        help="Directory with matched_length.json, icl_kshot.json, "
                             "context_pairs.json")
    parser.add_argument("--fast",        action="store_true",
                        help="albert-base-v2 only (quick smoke-test)")
    parser.add_argument("--no-model",    action="store_true",
                        help="Offline-only run: skip all tracks that need a model "
                             "(implies --skip-c2 --skip-c4 --skip-c5)")
    # Per-track skip flags
    parser.add_argument("--skip-c1",  action="store_true")
    parser.add_argument("--skip-c2",  action="store_true")
    parser.add_argument("--skip-c3",  action="store_true")
    parser.add_argument("--skip-c4",  action="store_true")
    parser.add_argument("--skip-c5",  action="store_true")
    parser.add_argument("--skip-hdr", action="store_true",
                        help="Don't run HDR even when jPCA is marginal")

    args = parser.parse_args()

    if args.no_model:
        args.skip_c2 = args.skip_c4 = args.skip_c5 = True

    # Resolve model name
    if args.fast:
        models = ["albert-base-v2"]
    elif args.model:
        models = [args.model]
    else:
        # Try to infer from phase1 metrics
        models = _infer_models(Path(args.phase1_dir))
        if not models:
            print("Could not detect models from phase1_dir. "
                  "Pass --model explicitly.")
            sys.exit(1)

    for model_name in models:
        run_phase(model_name, args)


def _infer_models(phase1_dir: Path) -> list[str]:
    """Try to read model list from Phase 1 metrics.json."""
    for fname in ("metrics.json", "phase1_metrics.json", "experiment.json"):
        p = phase1_dir / fname
        if p.exists():
            data = _jload(p)
            for key in ("models", "model_names", "model"):
                v = data.get(key)
                if v:
                    return [v] if isinstance(v, str) else list(v)
    # Fall back: look for model-named subdirectories
    known = {"albert-base-v2", "albert-xlarge-v2", "gpt2", "gpt2-large",
             "gpt2-xl", "bert-base-uncased"}
    found = [n for n in known
             if (phase1_dir / n.replace("/", "_")).is_dir()]
    return found


if __name__ == "__main__":
    main()