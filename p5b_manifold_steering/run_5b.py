"""
run_5b.py — Phase 5b CLI entry point.

Usage
-----
    python -m p5b_manifold.run_5b --model gpt2-large
    python -m p5b_manifold.run_5b --model gpt2-large --prompt wiki_paragraph
    python -m p5b_manifold.run_5b --model gpt2-large --fast
    python -m p5b_manifold.run_5b --models gpt2-large bert-base-uncased

Each sub-experiment writes its fragment immediately on completion.
A crash in sub-exp C leaves A and B results intact on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib  import Path

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 5b — Manifold isometry")
    p.add_argument("--model",      default="gpt2-large")
    p.add_argument("--models",     nargs="+", default=None)
    p.add_argument("--prompt",     default=None,
                   help="Force a specific prompt key; default: auto-select")
    p.add_argument("--phase1-dir", default="results/phase1")
    p.add_argument("--phase2-dir", default="results/phase2")
    p.add_argument("--out",        default=None)
    p.add_argument("--pca-dim",    type=int, default=32)
    p.add_argument("--geo-pts",    type=int, default=150,
                   help="Waypoints for geodesic arc-length integration")
    p.add_argument("--min-lifespan", type=int, default=3,
                   help="Skip trajectories shorter than this (layers)")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--skip-logits", action="store_true",
                   help="Skip logit extraction; use cached file if present")
    p.add_argument("--fast", action="store_true",
                   help="Reduced geodesic grid (50 pts); one model one prompt")
    return p


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------

def _run_one(args, model_name: str, out_dir: Path) -> int:
    from p5b_manifold_steering.p5b_io import (
        find_phase1_runs, load_phase1_run,
        select_best_run, load_phase2_projectors,
    )
    from p5b_manifold_steering.manifold_fit import (
        pca_reduce, arc_length_params,
        fit_activation_manifold, fit_behavior_manifold,
        load_plateau_centroids, compute_fit_summary,
    )
    from p5b_manifold_steering.isometry_test  import run_isometry_test
    from p5b_manifold_steering.merge_teleportation_subspace import run_merge_teleportation
    from p5b_manifold_steering.subspace_isometry   import subspace_isometry_score
    from p5b_manifold_steering.logit_cache import (
        extract_layer_logits, save_logit_cache, load_logit_cache,
    )
    from p5b_manifold_steering.p5b_report import write_report

    stem = model_name.replace("-", "_").replace("/", "_")

    # ---- Discover Phase 1 runs ----
    runs = find_phase1_runs(Path(args.phase1_dir), stem)
    if not runs:
        print(f"  [skip] no Phase 1 runs found for stem '{stem}' in {args.phase1_dir}")
        return 1

    prompt_key, run_dir = select_best_run(runs, preferred_prompt=args.prompt)
    if run_dir is None:
        print(f"  [skip] no suitable run found")
        return 1

    print(f"  using prompt '{prompt_key}'  →  {run_dir}")
    p1 = load_phase1_run(run_dir)

    plateau_layers = p1["plateau_layers"]
    merge_layers   = p1["merge_layers"]
    trajectories   = p1["trajectories"]
    centroid_trajs = p1["centroid_trajs"]

    print(f"  plateau layers : {len(plateau_layers)}  merge events : {len(merge_layers)}")
    print(f"  trajectories   : {len(trajectories)}    centroid seqs: {len(centroid_trajs)}")

    if len(centroid_trajs) < 4:
        print(f"  [skip] only {len(centroid_trajs)} centroid sequences — need ≥ 4")
        return 1

    # ---- Sub-exp A: manifold fitting ----
    print("[5b] Sub-exp A: manifold fitting")
    try:
        centroids, traj_ids = load_plateau_centroids(
            centroid_trajs, trajectories, min_lifespan=args.min_lifespan
        )
    except ValueError as e:
        print(f"  [skip] {e}")
        return 1

    n_c    = centroids.shape[0]
    pca_k  = min(args.pca_dim, n_c - 1)
    scores, basis, evr = pca_reduce(centroids, pca_k)
    u_act  = arc_length_params(scores, periodic=False)
    mh     = fit_activation_manifold(scores, u_act, periodic=False)

    print(f"  {n_c} clusters, PCA {pca_k}d, "
          f"cum_var={evr.sum():.3f}, Mh residual={mh['residual_rms']:.4f}")

    # ---- Logit cache ----
    cache_path = out_dir / "logit_cache.npz"
    logit_dists: dict = {}

    if cache_path.exists():
        logit_dists = load_logit_cache(cache_path)
        print(f"  loaded logit cache ({len(logit_dists)} layers)")
    elif not args.skip_logits and p1["activations"] is not None:
        print("  extracting per-layer logits (re-forward pass)...")
        try:
            from core.models import load_model
            from core.config  import PROMPTS
            model, tokenizer = load_model(model_name, device=args.device)
            prompt_text = PROMPTS.get(prompt_key, prompt_key)
            target_layers = list(set(plateau_layers + merge_layers))
            logit_dists = extract_layer_logits(
                model, tokenizer, prompt_text,
                layer_idxs=target_layers, device=args.device,
            )
            save_logit_cache(logit_dists, cache_path)
            print(f"  cached {len(logit_dists)} layers → {cache_path}")
            del model
        except Exception as e:
            print(f"  [warn] logit extraction failed: {e} — skipping My / Sub-exp C")

    # ---- Fit My from plateau distributions ----
    my      = None
    u_beh   = None
    pl_with_logits = [l for l in plateau_layers if l in logit_dists]

    if len(pl_with_logits) >= 4:
        dists_stack = np.stack(
            [logit_dists[l].mean(axis=0) for l in pl_with_logits], axis=0
        )  # (n_pl, vocab)
        u_beh = arc_length_params(dists_stack, periodic=False)
        my    = fit_behavior_manifold(dists_stack, u_beh, periodic=False)
        print(f"  My fit: {len(pl_with_logits)} plateau distributions, "
              f"residual={my['residual_rms']:.4f}")
    else:
        print(f"  [warn] only {len(pl_with_logits)} plateau layers have logits — "
              f"skipping My and isometry test")

    fit_sum = compute_fit_summary(mh, my, evr)
    (out_dir / "fit_summary.json").write_text(json.dumps(fit_sum, indent=2))
    np.savez_compressed(
        out_dir / "mh_params.npz",
        control_pts=mh["control_pts"],
        u_knots=mh["u_knots"],
        pca_basis=basis,
    )

    # ---- Sub-exp B: isometry ----
    iso_result: dict = {}
    if my is not None and len(u_act) == len(u_beh):
        print("[5b] Sub-exp B: isometry test")
        n_pts = 50 if args.fast else args.geo_pts
        try:
            iso_result = run_isometry_test(mh, my, u_act, scores, n_pts=n_pts)
            iso_flat = {k: v for k, v in iso_result.items() if k != "_mds"}
            (out_dir / "isometry.json").write_text(
                json.dumps(iso_flat, indent=2, default=float)
            )
            mds = iso_result.get("_mds", {})
            if mds:
                np.savez_compressed(
                    out_dir / "isometry_mds.npz",
                    **{f"mds_{k}": v for k, v in mds.items()},
                )
            print(f"  r_manifold={iso_result.get('r_manifold', float('nan')):.3f}  "
                  f"r_linear={iso_result.get('r_linear', float('nan')):.3f}  "
                  f"P5b-B1={'PASS' if iso_result.get('p5b_b1_pass') else 'FAIL'}")
        except Exception as e:
            print(f"  [error] Sub-exp B: {e}")
            traceback.print_exc()
    elif my is None:
        print("[5b] Sub-exp B: skipped (no My)")
    else:
        print(f"[5b] Sub-exp B: skipped (u_act len {len(u_act)} ≠ u_beh len {len(u_beh)})")

    # ---- Sub-exp C: merge teleportation ----
    tel_result: dict = {}
    if logit_dists and merge_layers:
        print("[5b] Sub-exp C: merge teleportation")
        try:
            tel_result = run_merge_teleportation(
                logit_dists, merge_layers, plateau_layers
            )
            (out_dir / "merge_teleportation.json").write_text(
                json.dumps(tel_result, indent=2, default=float)
            )
            print(f"  {tel_result.get('n_merge_events', 0)} events  "
                  f"P5b-C1={'PASS' if tel_result.get('p5b_c1_pass') else 'FAIL'}")
        except Exception as e:
            print(f"  [error] Sub-exp C: {e}")
            traceback.print_exc()
    else:
        reason = "no logits" if not logit_dists else "no merge layers"
        print(f"[5b] Sub-exp C: skipped ({reason})")

    # ---- Sub-exp D: S-subspace isometry ----
    sub_result: dict = {}
    d_behavior = np.array(iso_result.get("d_behavior", []))

    if len(d_behavior) > 0:
        print("[5b] Sub-exp D: S-subspace isometry")
        proj = load_phase2_projectors(Path(args.phase2_dir), stem)
        if proj is not None:
            U_S = proj.get("U_S_full", proj["U_S"])
            U_A = proj["U_A"]
            try:
                sub_result = subspace_isometry_score(centroids, U_S, U_A, d_behavior)
                (out_dir / "subspace_isometry.json").write_text(
                    json.dumps(sub_result, indent=2, default=float)
                )
                print(f"  r_S={sub_result['r_S']:.3f}  "
                      f"r_A={sub_result['r_A']:.3f}  "
                      f"r_full={sub_result['r_full']:.3f}  "
                      f"P5b-D1={'PASS' if sub_result.get('p5b_d1_pass') else 'FAIL'}")
            except Exception as e:
                print(f"  [error] Sub-exp D: {e}")
                traceback.print_exc()
        else:
            print(f"  [skip] Sub-exp D: no Phase 2 projectors in {args.phase2_dir}")
    else:
        print("[5b] Sub-exp D: skipped (no behavior distances from B)")

    # ---- Report ----
    from p5b_manifold_steering.p5b_report import write_report
    report_path = write_report(
        out_dir,
        results={
            "fit_summary":   fit_sum,
            "isometry":      iso_result,
            "teleportation": tel_result,
            "subspace":      sub_result,
        },
        model=model_name,
        prompt=prompt_key,
    )
    print(f"  report → {report_path}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args   = build_argparser().parse_args(argv)
    models = args.models or [args.model]
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    base   = Path(args.out) if args.out else Path("results/phase5b")

    results: dict[str, str] = {}
    for model_name in models:
        stem    = model_name.replace("-", "_").replace("/", "_")
        out_dir = base / f"{stem}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\n[phase5b] {model_name}\n{'='*60}")
        try:
            rc = _run_one(args, model_name, out_dir)
            results[model_name] = "ok" if rc == 0 else "skipped"
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results[model_name] = f"FAILED: {e}"

    print(f"\n{'='*60}\n[phase5b] summary")
    for m, status in results.items():
        icon = "✓" if status == "ok" else ("~" if "skip" in status else "✗")
        print(f"  {icon}  {m:30s}  {status}")
    print("=" * 60)

    return 0 if all(s in ("ok", "skipped") for s in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
