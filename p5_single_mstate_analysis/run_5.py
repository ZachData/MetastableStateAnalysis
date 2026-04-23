"""
run.py — Phase 5 case study CLI.

Loads artifacts from Phases 1–4, selects a primary trajectory and sibling,
runs the requested analysis groups (A through G), writes per-group JSON
fragments to a run directory, then calls report.py to emit a flat text
report.

Usage
-----
  python -m phase5_case.run \\
      --model albert-xlarge-v2 \\
      --groups A B C1 C2 D G \\
      --out results/phase5/albert_xlarge_v2_sullivan

Directory layout inferred from core.config if flags are omitted:
  phase1-dir : results/phase1
  phase2-dir : results/phase2
  phase2i-dir: results/phase2i
  phase3-dir : checkpoints/<model>/final
  phase3-cache: activation_cache/<model>
  phase4-dir : results/phase4

--force-prompt and --force-trajectory-id override the default rank-0 pick.
--runner-up-rank changes which ranked trajectory is reserved for replication.

Group F (causal) and Group E (tuned-lens) load the actual model and are
substantially slower; they're opt-in via the --groups flag.
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

from . import constants as C
from . import io as p5io
from .select_cluster    import select_primary_and_sibling, save_selection
from .cluster_profile   import compute_profile, save_profile
from .v_alignment       import compute_v_alignment, save_v_alignment
from .head_contributions import analyze_heads, save_head_contributions
from .ffn_contributions import analyze_ffn, save_ffn_contributions, load_ffn_deltas
from .feature_signature import analyze_features, save_feature_signature
from .sibling_contrast  import run_sibling_contrast, save_sibling_contrast
from .report            import write_report

ALL_GROUPS = ["A", "B", "C1", "C2", "D", "E", "F", "G"]


# ---------------------------------------------------------------------------
# Trajectory lookup helpers
# ---------------------------------------------------------------------------

def _traj_by_id(trajs: list, tid: int) -> dict:
    for t in trajs:
        if int(t["id"]) == int(tid):
            return t
    raise RuntimeError(f"Trajectory id={tid} not found in phase1 output")


def _centroid_coords(trajectory: dict, centroid_trajs: dict,
                      activations: np.ndarray, hdb_labels: list) -> np.ndarray:
    """
    Prefer the pre-computed centroid trajectory from Phase 1; reconstruct from
    activations if not stored.
    """
    tid = int(trajectory["id"])
    if tid in centroid_trajs:
        return centroid_trajs[tid].astype(np.float32)
    coords = []
    for layer, cid in trajectory["chain"]:
        if layer >= activations.shape[0]:
            break
        mask = hdb_labels[layer] == cid
        if mask.sum() < 1:
            continue
        c = activations[layer][mask].mean(axis=0)
        n = float(np.linalg.norm(c))
        coords.append(c / max(n, 1e-12))
    return np.stack(coords).astype(np.float32) if coords else np.zeros((0, 0))


# ---------------------------------------------------------------------------
# Feature activations on demand
# ---------------------------------------------------------------------------

def _compute_feature_activations(
    phase3: dict,
    prompt_key: str,
    layers_needed: list,
) -> dict:
    """
    Get crosscoder feature activations for the requested layers on the
    selected prompt. Uses the prompt activation store as the residual source.
    """
    store = phase3.get("prompt_store")
    cc = phase3.get("crosscoder")
    if store is None or cc is None:
        return {}

    try:
        import torch
    except ImportError:
        return {}

    out = {}
    device = next(cc.parameters()).device
    for L in layers_needed:
        try:
            res = store.get(prompt_key, L)   # (n_tokens, d)
        except Exception:
            continue
        if res is None:
            continue
        with torch.no_grad():
            r = torch.from_numpy(np.asarray(res, dtype=np.float32)).to(device)
            # Crosscoder encoder returns feature activations
            if hasattr(cc, "encode"):
                feats = cc.encode(r)
            elif hasattr(cc, "forward"):
                feats = cc(r)[0]
            else:
                continue
            out[L] = feats.cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Individual group runners
# ---------------------------------------------------------------------------

def _run_group_A(run, primary_raw, sibling_raw, out_dir) -> dict:
    profile = compute_profile(
        run["activations"], run["hdbscan_labels"],
        primary_raw, sibling_raw,
        run["tokens"], run["metrics"],
    )
    save_profile(profile, out_dir, tag="primary")
    return profile


def _run_group_B(run, primary_raw, sibling_raw, v_proj, phase2i,
                  centroid_coords, sibling_centroid_coords, out_dir) -> dict:
    if run.get("attentions") is None:
        print("  [B] skipped: attentions.npz missing")
        return {"_error": "attentions unavailable"}
    result = compute_v_alignment(
        run["activations"], run["attentions"], run["hdbscan_labels"],
        primary_raw, sibling_raw,
        centroid_coords, sibling_centroid_coords,
        v_proj, phase2i,
    )
    save_v_alignment(result, out_dir, tag="primary")
    return result


def _run_group_C1(run, primary_raw, weights, out_dir) -> dict:
    if run.get("attentions") is None:
        print("  [C1] skipped: attentions.npz missing")
        return {"_error": "attentions unavailable"}
    result = analyze_heads(
        run["activations"], run["attentions"], run["hdbscan_labels"],
        primary_raw, run["tokens"], weights=weights,
    )
    save_head_contributions(result, out_dir, tag="primary")
    return result


def _run_group_C2(run, primary_raw, sibling_raw, phase2_dir, out_dir) -> dict:
    deltas = load_ffn_deltas(Path(phase2_dir), run["prompt_key"])
    ffn_deltas = deltas["ffn"] if deltas else None
    attn_deltas = deltas["attn"] if deltas else None
    if ffn_deltas is None:
        print(f"  [C2] no phase2 ffn deltas for prompt {run['prompt_key']} "
              f"— LDA + centroid directions only, no projection metrics")
    result = analyze_ffn(
        run["activations"], run["hdbscan_labels"],
        primary_raw, sibling_raw,
        ffn_deltas=ffn_deltas, attn_deltas=attn_deltas,
    )
    save_ffn_contributions(result, out_dir, tag="primary")
    return result


def _run_group_D(run, primary_raw, sibling_raw, phase3, v_proj, phase4,
                  centroid_coords, out_dir) -> dict:
    layers_needed = sorted({int(l) for l, _ in primary_raw["chain"]})
    feature_acts = _compute_feature_activations(
        phase3, run["prompt_key"], layers_needed,
    )
    if not feature_acts:
        print("  [D] skipped: feature activations unavailable "
              "(phase3 crosscoder or prompt store missing)")
        result = {"_error": "feature activations unavailable"}
        save_feature_signature(result, out_dir, tag="primary")
        return result

    # Try to find decoder directions
    decoder_dirs = None
    cc = phase3.get("crosscoder")
    if cc is not None:
        for attr in ("decoder_weight", "W_dec", "decoder"):
            if hasattr(cc, attr):
                d = getattr(cc, attr)
                if hasattr(d, "weight"):
                    d = d.weight
                if hasattr(d, "detach"):
                    decoder_dirs = d.detach().cpu().numpy()
                else:
                    decoder_dirs = np.asarray(d)
                break

    # Load LDA directions from Group C.2 output npz if it exists
    lda_npz_path = out_dir / "group_C2_lda_directions_primary.npz"
    lda_dirs = {k: v for k, v in np.load(lda_npz_path).items()} \
        if lda_npz_path.exists() else None

    # Phase 4 bottleneck
    bn = None
    bn_blob = phase4.get("t3_bottleneck_directions")
    if bn_blob:
        for k, v in bn_blob.items():
            bn = v
            break

    result = analyze_features(
        feature_acts, decoder_dirs,
        run["hdbscan_labels"], primary_raw, sibling_raw,
        lda_directions_per_layer=lda_dirs,
        v_projectors=v_proj,
        bottleneck_directions=bn,
        cluster_centroid=(
            centroid_coords.mean(axis=0)
            if centroid_coords is not None and centroid_coords.ndim == 2
               and centroid_coords.shape[0] > 0
            else None
        ),
    )
    save_feature_signature(result, out_dir, tag="primary")
    return result


def _run_group_E(run, primary_raw, sibling_raw, model, tokenizer, out_dir,
                  tuned_lens_path) -> dict:
    from .tuned_lens_cluster import (
        decode_cluster_trajectory, save_tuned_lens_result,
        kl_sibling_contrast, load_tuned_lens,
    )
    tuned_lens = load_tuned_lens(Path(tuned_lens_path)) if tuned_lens_path else None

    primary_result = decode_cluster_trajectory(
        run["activations"], run["hdbscan_labels"],
        primary_raw, run["tokens"],
        model, tokenizer, tuned_lens=tuned_lens,
    )

    # Optionally decode sibling for KL contrast
    if sibling_raw is not None and sibling_raw.get("chain"):
        sibling_result = decode_cluster_trajectory(
            run["activations"], run["hdbscan_labels"],
            sibling_raw, run["tokens"],
            model, tokenizer, tuned_lens=tuned_lens,
            decode_members=False,
        )
        # KL contrast
        merge = primary_raw.get("merge_event")
        merge_layer = merge["layer_from"] if merge else None
        kl = kl_sibling_contrast(
            primary_result, sibling_result,
            primary_result.get("_distributions", {}),
            sibling_result.get("_distributions", {}),
            primary_raw["chain"], sibling_raw["chain"],
            merge_layer=merge_layer,
        )
        primary_result["kl_contrast"] = kl
        # Save sibling separately
        save_tuned_lens_result(sibling_result, out_dir, tag="sibling")

    save_tuned_lens_result(primary_result, out_dir, tag="primary")
    return primary_result


def _run_group_F(run, primary_raw, sibling_raw, model, tokenizer,
                  c1_result, c2_result, max_iterations, steering_alpha,
                  out_dir) -> dict:
    from .causal_tests import run_causal_tests, save_causal

    # Extract directions from prior group outputs
    top_heads = c1_result.get("top_attractor_heads", []) if c1_result else []
    # Pick an LDA direction at the merge layer (or middle)
    lda_dir = None
    c2_npz = out_dir / "group_C2_lda_directions_primary.npz"
    if c2_npz.exists():
        data = dict(np.load(c2_npz))
        merge = primary_raw.get("merge_event")
        target_L = merge["layer_from"] if merge else (
            primary_raw["chain"][len(primary_raw["chain"]) // 2][0]
        )
        for offset in (0, -1, -2, 1):
            key = f"lda_L{target_L + offset}"
            if key in data:
                lda_dir = data[key]
                break

    # Centroid direction at mid-layer
    chain = primary_raw["chain"]
    mid_layer, mid_cid = chain[len(chain) // 2]
    mask = run["hdbscan_labels"][mid_layer] == mid_cid
    centroid = run["activations"][mid_layer][mask].mean(axis=0)
    centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)

    # Source prompt text: reconstruct from tokens via tokenizer
    prompt_text = tokenizer.convert_tokens_to_string(run["tokens"])

    result = run_causal_tests(
        model, tokenizer, prompt_text, max_iterations,
        run["activations"], run["hdbscan_labels"],
        primary_raw, sibling_raw,
        top_heads=top_heads,
        lda_direction=lda_dir,
        centroid_direction=centroid,
        steering_alpha=steering_alpha,
    )
    save_causal(result, out_dir)
    return result


def _run_group_G(run, primary_raw, sibling_raw, weights, out_dir) -> dict:
    result = run_sibling_contrast(
        run["activations"], run.get("attentions"),
        run["hdbscan_labels"],
        primary_raw, sibling_raw,
        run["tokens"], run["metrics"],
        weights=weights,
    )
    save_sibling_contrast(result, out_dir)
    return result


# ---------------------------------------------------------------------------
# Model loading for Groups E and F
# ---------------------------------------------------------------------------

def _load_model(model_name: str, device: str = "cpu"):
    from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
    tok = AutoTokenizer.from_pretrained(model_name)
    # For MLM models we want the head attached
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    except Exception:
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 5 case study orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="albert-xlarge-v2",
                    help="HF model id (used both to pick phase1 runs and to "
                         "load the model for Groups E, F)")
    p.add_argument("--model-stem", default=None,
                    help="Phase 1 directory stem. Defaults to model basename "
                         "with hyphens replaced by underscores.")

    p.add_argument("--phase1-dir",  default="results/phase1")
    p.add_argument("--phase2-dir",  default="results/phase2")
    p.add_argument("--phase2i-dir", default="results/phase2i")
    p.add_argument("--phase3-ckpt", default=None,
                    help="Defaults to checkpoints/<model>/final")
    p.add_argument("--phase3-cache", default=None,
                    help="Defaults to activation_cache/<model>")
    p.add_argument("--phase4-dir",  default="results/phase4")
    p.add_argument("--tuned-lens",  default=None,
                    help="Optional tuned-lens npz (A_L / b_L per layer)")

    p.add_argument("--out", default=None,
                    help="Output directory. Defaults to "
                         "results/phase5/<model_stem>_<timestamp>")
    p.add_argument("--groups", nargs="+", default=ALL_GROUPS,
                    choices=ALL_GROUPS,
                    help="Which analysis groups to run")

    p.add_argument("--force-prompt", default=None)
    p.add_argument("--force-trajectory-id", type=int, default=None)
    p.add_argument("--runner-up-rank", type=int, default=1)

    p.add_argument("--max-iterations", type=int, default=64,
                    help="ALBERT shared-layer iteration depth for Group F")
    p.add_argument("--steering-alpha", type=float, default=2.0)
    p.add_argument("--device", default="cpu")

    p.add_argument("--dry-run", action="store_true",
                    help="Load everything, pick the trajectory, but don't "
                         "run any groups (useful to preview selection)")
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    model_stem = args.model_stem or args.model.replace("-", "_")
    phase3_ckpt  = Path(args.phase3_ckpt  or f"checkpoints/{args.model}/final")
    phase3_cache = Path(args.phase3_cache or f"activation_cache/{args.model}")

    # Output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path(
        f"results/phase5/{model_stem}_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase5] output dir: {out_dir}")

    # Save the invocation for reproducibility
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # --- Load Phase 1 runs ---
    print(f"[phase5] discovering phase1 runs in {args.phase1_dir} "
          f"matching stem '{model_stem}'")
    run_paths = p5io.find_phase1_runs(Path(args.phase1_dir), model_stem)
    if not run_paths:
        print("  no phase1 runs found — check --phase1-dir and --model-stem")
        return 1
    print(f"  found: {list(run_paths.keys())}")

    phase1_runs = {}
    for prompt_key, run_path in run_paths.items():
        try:
            phase1_runs[prompt_key] = p5io.load_phase1_run(run_path)
        except Exception as e:
            print(f"  [skip] {prompt_key}: {e}")

    # --- Selection ---
    print("[phase5] ranking trajectories")
    selection = select_primary_and_sibling(
        phase1_runs,
        force_prompt=args.force_prompt,
        force_trajectory_id=args.force_trajectory_id,
        runner_up_rank=args.runner_up_rank,
    )
    save_selection(selection, out_dir / "cluster_metadata.json")

    primary = selection["primary"]
    sibling = selection["sibling"]
    print(f"  primary: prompt={primary['prompt_key']} id={primary['id']} "
          f"score={primary['total_score']:.3f}")
    if sibling:
        print(f"  sibling: id={sibling['id']} "
              f"{'(below gates)' if 'note' in sibling else ''}")
    if selection.get("runner_up"):
        ru = selection["runner_up"]
        print(f"  runner_up (reserved for replication): prompt={ru['prompt_key']} "
              f"id={ru['id']} score={ru['total_score']:.3f}")

    if args.dry_run:
        print("[phase5] dry run complete")
        return 0

    # --- Load remaining artifacts based on which groups were requested ---
    run = phase1_runs[primary["prompt_key"]]
    primary_raw = _traj_by_id(run["trajectories"], primary["id"])
    primary_raw["merge_event"] = primary.get("merge_event")   # attach for B

    sibling_raw = None
    if sibling and sibling.get("id") is not None:
        try:
            sibling_raw = _traj_by_id(run["trajectories"], sibling["id"])
        except Exception as e:
            print(f"  [warn] sibling raw lookup failed: {e}")

    centroid_coords = _centroid_coords(
        primary_raw, run.get("centroid_trajs", {}),
        run["activations"], run["hdbscan_labels"],
    )
    sibling_centroid_coords = (
        _centroid_coords(
            sibling_raw, run.get("centroid_trajs", {}),
            run["activations"], run["hdbscan_labels"],
        )
        if sibling_raw is not None else None
    )

    v_proj   = {}
    phase2i  = {}
    weights  = {}
    phase3   = {}
    phase4   = {}

    if any(g in args.groups for g in ("B", "D")):
        print("[phase5] loading phase2 projectors")
        v_proj = p5io.load_phase2_projectors(
            Path(args.phase2_dir), model_stem, k_top=C.V_PROJECTOR_K_TOP,
        )
        if v_proj.get("path") is None:
            print(f"  [warn] no phase2 projectors found for stem {model_stem}")

    if "B" in args.groups:
        print("[phase5] loading phase2i artifacts")
        phase2i = p5io.load_phase2i(Path(args.phase2i_dir), model_stem)

    if any(g in args.groups for g in ("C1", "G")):
        print("[phase5] loading phase2 weights")
        weights = p5io.load_phase2_weights(Path(args.phase2_dir), model_stem)
        if not weights:
            print(f"  [note] no phase2 weights.npz — C1 will run attention-only "
                  f"(no OV cohesion)")

    if "D" in args.groups:
        print("[phase5] loading phase3 crosscoder + prompt store")
        phase3 = p5io.load_phase3(phase3_ckpt, phase3_cache, device=args.device)
        print("[phase5] loading phase4 artifacts")
        phase4 = p5io.load_phase4(Path(args.phase4_dir))

    # --- Run groups in dependency order ---
    c1_result = None
    c2_result = None

    if "A" in args.groups:
        print("[phase5] GROUP A: structural profile")
        _run_group_A(run, primary_raw, sibling_raw, out_dir)

    if "B" in args.groups:
        print("[phase5] GROUP B: v-alignment")
        _run_group_B(
            run, primary_raw, sibling_raw, v_proj, phase2i,
            centroid_coords, sibling_centroid_coords, out_dir,
        )

    if "C1" in args.groups:
        print("[phase5] GROUP C.1: per-head attention")
        c1_result = _run_group_C1(run, primary_raw, weights, out_dir)

    if "C2" in args.groups:
        print("[phase5] GROUP C.2: ffn contributions")
        c2_result = _run_group_C2(
            run, primary_raw, sibling_raw, args.phase2_dir, out_dir,
        )

    if "D" in args.groups:
        print("[phase5] GROUP D: feature signatures")
        _run_group_D(
            run, primary_raw, sibling_raw, phase3, v_proj, phase4,
            centroid_coords, out_dir,
        )

    if "E" in args.groups:
        print(f"[phase5] GROUP E: tuned-lens (loading {args.model})")
        try:
            model, tokenizer = _load_model(args.model, device=args.device)
            _run_group_E(
                run, primary_raw, sibling_raw,
                model, tokenizer, out_dir, args.tuned_lens,
            )
        except Exception as e:
            print(f"  [E] failed: {e}")

    if "F" in args.groups:
        print(f"[phase5] GROUP F: causal tests (loading {args.model})")
        if c1_result is None:
            print("  [warn] C1 was not run — top attractor heads unknown, "
                  "ablations will have no targets")
        try:
            model, tokenizer = _load_model(args.model, device=args.device)
            _run_group_F(
                run, primary_raw, sibling_raw, model, tokenizer,
                c1_result, c2_result, args.max_iterations,
                args.steering_alpha, out_dir,
            )
        except Exception as e:
            print(f"  [F] failed: {e}")

    if "G" in args.groups:
        print("[phase5] GROUP G: sibling + random control")
        _run_group_G(run, primary_raw, sibling_raw, weights, out_dir)

    # --- Persist shared per-layer arrays in a single npz for later replay ---
    shared = {}
    for i, arr in enumerate(run["hdbscan_labels"]):
        shared[f"hdb_L{i}"] = np.asarray(arr, dtype=np.int32)
    shared["primary_centroids"] = centroid_coords
    if sibling_centroid_coords is not None:
        shared["sibling_centroids"] = sibling_centroid_coords
    np.savez_compressed(out_dir / "per_layer_arrays.npz", **shared)

    # --- Final report ---
    print("[phase5] writing report")
    report_path = write_report(out_dir, model=args.model, tag="primary")
    print(f"  wrote {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
