"""
run_5b.py — Phase 5b CLI entry point.

Usage
-----
    python -m p5b_manifold_steering.run_5b --model gpt2-large
    python -m p5b_manifold_steering.run_5b --model gpt2-large --prompt wiki_paragraph
    python -m p5b_manifold_steering.run_5b --model gpt2-large --fast
    python -m p5b_manifold_steering.run_5b --models gpt2-large pythia-1.4b

Each sub-experiment writes its fragment immediately on completion.
A crash in sub-exp C leaves A and B results intact on disk.

REWIRED 2026-07-21 (see WORKING-5b.md). What changed and why:

  * Mh and My are now fit over ONE population — the surviving cluster
    trajectories — instead of two unrelated ones (trajectories for Mh,
    plateau layers for My). The old `len(u_act) == len(u_beh)` gate that
    routed around the mismatch is gone, replaced by an assertion, because
    after this change a mismatch is a bug rather than an expected skip.
    That gate is why isometry.json was never written.

  * Behavior distributions are built by masking tokens with the SAME
    (layer, cluster_id) chain the centroids use, via
    cluster_tracking.compute_behavior_trajectories. The old code took a
    global mean over every token at each plateau layer, which discarded
    the cluster association entirely.

  * Sub-exp B runs on direct pairwise distances, not spline geodesics, and
    its control is FRAME-VS-RAW rather than curve-vs-chord. Rationale in
    isometry_test.py's append block and WORKING-5b.md §3.

  * LN frame parameters are captured while the model is loaded for logit
    extraction (it is already in hand at that point, so this is free) and
    persisted next to the logit cache so reruns from cache keep them.

  * sphere_gap diagnostics run unconditionally, including under
    --skip-logits, since they need only p1["activations"] and no model.
    They are the pre-registered escalation trigger and should exist for
    runs already completed.
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
                   help="Waypoints for geodesic arc-length integration (Sub-exp A)")
    p.add_argument("--min-lifespan", type=int, default=3,
                   help="Skip trajectories shorter than this (layers)")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--skip-logits", action="store_true",
                   help="Skip logit extraction; use cached file if present")
    p.add_argument("--behavior-space", default="hellinger",
                   choices=("hellinger", "mixture"),
                   help="How per-cluster distributions are aggregated across "
                        "a trajectory's chain. Recorded in isometry.json; it "
                        "changes the numbers.")
    p.add_argument("--min-coverage", type=float, default=0.0,
                   help="Drop trajectories whose chain-layer logit coverage "
                        "is below this fraction. 0 = keep all.")
    p.add_argument("--fast", action="store_true",
                   help="Reduced geodesic grid (50 pts); one model one prompt")
    return p


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _reindex(arr, have_ids, want_ids):
    """Select rows of `arr` (aligned to have_ids) in want_ids order."""
    pos = {int(t): i for i, t in enumerate(have_ids)}
    idx = [pos[int(t)] for t in want_ids]
    return np.asarray(arr)[idx]


def _save_ln_params(ln_frames: dict, path: Path) -> None:
    """
    Persist resolved LN frame params next to the logit cache.

    Without this, a rerun that hits the logit cache never loads a model and
    therefore silently loses the LN frame — the secondary reading would
    vanish between an extraction run and a cached rerun on the same data,
    for no reason the artifact records.
    """
    payload = {}
    meta = {}
    for li, fr in ln_frames.items():
        meta[str(li)] = {"frame": fr.get("frame"), "block_idx": fr.get("block_idx")}
        params = fr.get("params")
        if params:
            for pname, pval in params.items():
                if isinstance(pval, np.ndarray):
                    payload[f"{li}::{pname}"] = pval
                else:
                    meta[str(li)][pname] = pval
    payload["__meta__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(path, **payload)


def _load_ln_params(path: Path) -> dict:
    """Inverse of _save_ln_params. Returns {} when absent or unreadable."""
    if not Path(path).exists():
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(data["__meta__"]).decode("utf-8"))
        out: dict = {}
        for li_s, m in meta.items():
            li = int(li_s)
            params = {k: v for k, v in m.items()
                      if k not in ("frame", "block_idx")}
            for key in data.files:
                if key.startswith(f"{li}::"):
                    params[key.split("::", 1)[1]] = data[key]
            out[li] = {
                "frame":     m.get("frame"),
                "block_idx": m.get("block_idx"),
                "params":    params or None,
            }
        return out
    except Exception:
        return {}


def _chain_layers(trajectories, traj_ids) -> list[int]:
    """
    Union of every layer index appearing in the given trajectories' chains.

    This is what the logit cache must cover. The old code requested
    `plateau_layers + merge_layers`, which is NOT sufficient: a trajectory's
    chain routinely spans layers outside that set, so masking against the
    cache silently produced zero coverage for those steps.
    """
    want = {int(t) for t in traj_ids}
    layers: set[int] = set()
    for traj in trajectories:
        if int(traj["id"]) in want:
            for layer_idx, _cid in traj.get("chain", []):
                layers.add(int(layer_idx))
    return sorted(layers)


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
        load_plateau_centroids, compute_fit_summary, sphere_gap_by_layer,
    )
    from p5b_manifold_steering.p5b_distances import (
        frame_centroids, ln_params_for_layers,
        behavior_distance_matrix, upper_triangle,
    )
    from p5b_manifold_steering.isometry_test import run_isometry_direct
    from p5b_manifold_steering.merge_teleportation_subspace import run_merge_teleportation
    from p5b_manifold_steering.subspace_isometry import subspace_isometry_score
    from p5b_manifold_steering.logit_cache import (
        extract_layer_logits, save_logit_cache, load_logit_cache,
    )
    from p1_mstate_tracking.cluster_tracking import (
        compute_behavior_trajectories, stack_behavior_by_traj_ids,
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
    label_arrays   = p1.get("hdbscan_labels", {})
    activations    = p1.get("activations")

    print(f"  plateau layers : {len(plateau_layers)}  merge events : {len(merge_layers)}")
    print(f"  trajectories   : {len(trajectories)}    centroid seqs: {len(centroid_trajs)}")
    if not label_arrays:
        print("  [warn] no hdbscan_labels.json in this run — Sub-exp B unavailable "
              "(cannot mask tokens by cluster)")

    if len(centroid_trajs) < 4:
        print(f"  [skip] only {len(centroid_trajs)} centroid sequences — need ≥ 4")
        return 1

    # ---- Sub-exp A: manifold fitting (activation side) ----
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

    # ---- Frame diagnostics: unconditional, no model needed ----
    frame_diag = sphere_gap_by_layer(activations, layers=plateau_layers)
    mpg = frame_diag.get("max_pearson_gap")
    print(f"  sphere_gap over plateau layers: max_pearson_gap="
          f"{'n/a' if mpg is None else f'{mpg:.4f}'}")

    # ---- Logit cache ----
    cache_path = out_dir / "logit_cache.npz"
    ln_path    = out_dir / "ln_frame_params.npz"
    logit_dists: dict = {}
    ln_frames:   dict = {}

    # The cache must cover every layer the surviving trajectories touch, not
    # just plateau + merge. See _chain_layers.
    target_layers = sorted(set(_chain_layers(trajectories, traj_ids))
                           | set(int(l) for l in plateau_layers)
                           | set(int(l) for l in merge_layers))

    if cache_path.exists():
        logit_dists = load_logit_cache(cache_path)
        ln_frames   = _load_ln_params(ln_path)
        print(f"  loaded logit cache ({len(logit_dists)} layers)"
              + (f", LN frames ({len(ln_frames)} layers)" if ln_frames else ""))
    elif not args.skip_logits and activations is not None:
        print(f"  extracting per-layer logits for {len(target_layers)} layers "
              f"(re-forward pass)...")
        try:
            from core.models import load_model
            from core.config  import PROMPTS
            model, tokenizer = load_model(model_name, device=args.device)
            prompt_text = PROMPTS.get(prompt_key, prompt_key)
            logit_dists = extract_layer_logits(
                model, tokenizer, prompt_text,
                layer_idxs=target_layers, device=args.device,
            )
            save_logit_cache(logit_dists, cache_path)
            print(f"  cached {len(logit_dists)} layers → {cache_path}")

            # The model is already loaded; LN params are free here and
            # nowhere else in this pipeline.
            n_hs = p1.get("n_layers") or (
                activations.shape[0] if activations is not None else 0
            )
            ln_frames = ln_params_for_layers(model, target_layers, n_hs)
            n_ok = sum(1 for f in ln_frames.values() if f.get("params"))
            if n_ok:
                _save_ln_params(ln_frames, ln_path)
                print(f"  LN frames resolved for {n_ok}/{len(ln_frames)} layers")
            else:
                print("  [note] no LN frames resolvable on this architecture "
                      "(core/ln_frame.py is GPT-NeoX-only) — LN reading omitted")
            del model
        except Exception as e:
            print(f"  [warn] logit extraction failed: {e} — skipping My / Sub-exp B/C")

    ln_available = any(f.get("params") for f in ln_frames.values())

    # ---- Aligned behavior distributions ----
    # One distribution per surviving trajectory, masked by the SAME chain
    # the centroid used. This is what makes Mh and My describe one population.
    beh_trajs: dict = {}
    coverage:  dict = {}
    if logit_dists and label_arrays:
        beh_trajs, coverage = compute_behavior_trajectories(
            trajectories, label_arrays, logit_dists, space=args.behavior_space,
        )
        if args.min_coverage > 0.0:
            dropped = [t for t, c in coverage.items()
                       if c["frac"] < args.min_coverage]
            for t in dropped:
                beh_trajs.pop(t, None)
            if dropped:
                print(f"  dropped {len(dropped)} trajectories below "
                      f"--min-coverage={args.min_coverage}")

    # ---- Build the common point set across every frame and the behavior side ----
    frames: dict = {}
    kept_by_frame: dict = {}
    my = None
    common: list[int] = []
    dists_common = None

    if beh_trajs:
        try:
            dists_all, kept_beh = stack_behavior_by_traj_ids(
                beh_trajs, traj_ids, space=args.behavior_space
            )
        except ValueError as e:
            print(f"  [warn] {e}")
            dists_all, kept_beh = None, []

        if dists_all is not None:
            wanted_frames = ["sphere", "raw"] + (["ln"] if ln_available else [])
            for fr in wanted_frames:
                try:
                    C, k = frame_centroids(
                        activations, label_arrays, trajectories, traj_ids,
                        frame=fr, ln_frames=ln_frames,
                        renormalize=(fr != "raw"),
                    )
                    frames[fr], kept_by_frame[fr] = C, k
                except Exception as e:
                    print(f"  [warn] frame {fr!r} unavailable: {e}")

            if "sphere" in frames and "raw" in frames:
                ok = set(kept_beh)
                for k in kept_by_frame.values():
                    ok &= set(k)
                common = [int(t) for t in traj_ids if int(t) in ok]

                if len(common) >= 4:
                    dists_common = _reindex(dists_all, kept_beh, common)
                    for fr in list(frames):
                        frames[fr] = _reindex(frames[fr], kept_by_frame[fr], common)

                    # My shares the activation-side parameterization by
                    # construction now: same points, same order.
                    sphere_common = frames["sphere"]
                    k_common = min(args.pca_dim, len(common) - 1)
                    sc_common, _, _ = pca_reduce(sphere_common, k_common)
                    u_common = arc_length_params(sc_common, periodic=False)
                    my = fit_behavior_manifold(dists_common, u_common, periodic=False)

                    assert dists_common.shape[0] == sphere_common.shape[0], (
                        "aligned population invariant violated: "
                        f"{dists_common.shape[0]} distributions vs "
                        f"{sphere_common.shape[0]} centroids. Both are built "
                        "from `common`; if these differ, _reindex or "
                        "frame_centroids has a bug."
                    )
                    print(f"  My fit: {len(common)} aligned cluster distributions, "
                          f"residual={my['residual_rms']:.4f}")
                else:
                    print(f"  [warn] only {len(common)} trajectories survive all "
                          f"frames + logit coverage — need ≥ 4 for Sub-exp B")
    else:
        print("  [warn] no aligned behavior distributions — skipping My / Sub-exp B")

    fit_sum = compute_fit_summary(mh, my, evr, frame_diagnostics=frame_diag)
    (out_dir / "fit_summary.json").write_text(json.dumps(fit_sum, indent=2, default=float))
    np.savez_compressed(
        out_dir / "mh_params.npz",
        control_pts=mh["control_pts"],
        u_knots=mh["u_knots"],
        pca_basis=basis,
    )

    # ---- Sub-exp B: frame-vs-raw isometry on direct distances ----
    iso_result: dict = {}
    d_behavior = np.array([])

    if dists_common is not None and len(common) >= 4:
        print("[5b] Sub-exp B: isometry (direct distances, frame-vs-raw)")
        try:
            iso_result = run_isometry_direct(
                centroids_by_frame=frames,
                distributions=dists_common,
                traj_ids=common,
                frame_diagnostics=frame_diag,
                behavior_space=args.behavior_space,
                ln_available=ln_available,
                coverage={t: coverage[t] for t in common if t in coverage},
            )
            (out_dir / "isometry.json").write_text(
                json.dumps(iso_result, indent=2, default=float)
            )
            v = iso_result["verdict"]
            def _f(x): return "n/a" if x is None else f"{x:.3f}"
            print(f"  frame={iso_result['primary']['activation_frame']}  "
                  f"r_frame={_f(v['r_frame'])}  r_raw={_f(v['r_raw'])}  "
                  f"Δ={_f(v['delta'])}")
            print(f"  P5b-B1={'PASS' if v['P5b-B1'] else 'FAIL'}  "
                  f"B2={'PASS' if v['P5b-B2'] else 'FAIL'}  "
                  f"B3={'PASS' if v['P5b-B3'] else 'FAIL'}")

            # Behavior distances for Sub-exp D, in the PRIMARY metric only.
            # sym_kl is deliberately not used downstream — it is not a
            # metric (see p5b_distances) and D's scoring assumes one.
            d_behavior = upper_triangle(
                behavior_distance_matrix(dists_common, metric="hellinger")
            )
        except Exception as e:
            print(f"  [error] Sub-exp B: {e}")
            traceback.print_exc()
    else:
        print("[5b] Sub-exp B: skipped (no aligned population)")

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
    if len(d_behavior) > 0:
        print("[5b] Sub-exp D: S-subspace isometry")
        proj = load_phase2_projectors(Path(args.phase2_dir), stem)
        if proj is not None:
            U_S = proj.get("U_S_full", proj["U_S"])
            U_A = proj["U_A"]
            try:
                # Sphere-frame centroids restricted to `common` — the same
                # point set d_behavior was computed over. Passing the full
                # unaligned centroid array here would reintroduce exactly
                # the population mismatch this rewrite removed.
                #
                # centroids_raw=frames.get("raw"): frames["raw"] is already
                # built, already restricted to `common`, and already
                # un-normalized (frame_centroids is called with
                # renormalize=False for it). Without this, r_linear falls
                # back to aliasing r_full and subspace_isometry_score sets
                # r_linear_is_alias=True — P5b-D2 would then be scoring
                # against its own treatment. See design-5b.md, Sub-exp D.
                sub_result = subspace_isometry_score(
                    frames["sphere"], U_S, U_A, d_behavior,
                    centroids_raw=frames.get("raw"),
                )
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
