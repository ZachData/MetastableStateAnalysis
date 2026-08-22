"""
p5b_manifold/logit_cache.py — Extract and cache output distributions per layer.

The output distribution at each layer is obtained by running the residual
stream at that layer through the final LayerNorm and unembedding matrix,
giving logits for the full vocabulary at every token position and layer.

This is a lightweight re-forward: same inputs as Phase 1, but we hook every
layer's residual stream output and apply the LM head to it.

Imports from existing project scripts:
  - core.models  : load_model (standard model loading with hooks)
  - core.config  : MODEL_CONFIGS, PROMPTS
"""

from __future__ import annotations

from pathlib import Path
from typing  import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Validation helpers (tested directly)
# ---------------------------------------------------------------------------

def validate_logit_output(
    logits:         dict,
    expected_vocab: Optional[int] = None,
) -> None:
    """
    Raise ValueError if logit_distributions is malformed.
    logits : {layer_idx: (n_tokens, vocab) float32 log-probability array}
    """
    for layer_idx, arr in logits.items():
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"Layer {layer_idx}: logit array contains non-finite values "
                f"(inf or nan). Check for collapsed/degenerate residual streams."
            )
        if expected_vocab is not None and arr.shape[1] != expected_vocab:
            raise ValueError(
                f"Layer {layer_idx}: expected vocab size {expected_vocab}, "
                f"got {arr.shape[1]}."
            )


def logits_to_distribution(logits: np.ndarray) -> np.ndarray:
    """
    Softmax of logit array, numerically stable.

    Parameters
    ----------
    logits : (n_tokens, vocab) float32

    Returns
    -------
    p : (n_tokens, vocab) float32 — probability distributions
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp     = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_layer_logits(
    model,
    tokenizer,
    prompt:     str,
    layer_idxs: list[int],
    device:     str = "cpu",
) -> dict:
    """
    Run a forward pass and extract output distributions at each specified layer
    by applying the model's LM head to the residual stream.

    Parameters
    ----------
    model      : HuggingFace causal LM with .transformer or .model attribute
    tokenizer  : matching tokenizer
    prompt     : text prompt (same as used in Phase 1)
    layer_idxs : which layer indices to extract distributions for
    device     : torch device string

    Returns
    -------
    distributions : {layer_idx: (n_tokens, vocab) float32 probabilities}
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_tok  = inputs["input_ids"].shape[1]

    # Collect residual stream activations at each requested layer
    residuals: dict[int, np.ndarray] = {}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            # output is the post-layer residual stream: (1, n_tok, d_model)
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            residuals[idx] = h.detach().float().cpu().numpy()[0]  # (n_tok, d)
        return hook_fn

    # Register hooks on the transformer layers
    # Supports GPT-2 style (.transformer.h) and BERT style (.encoder.layer)
    layers_module = _get_layers(model)
    for idx in layer_idxs:
        if idx < len(layers_module):
            h = layers_module[idx].register_forward_hook(make_hook(idx))
            hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    # Apply LM head to each captured residual stream
    ln  = _get_final_ln(model)
    W_e = _get_unembed(model)    # (d_model, vocab)

    distributions = {}
    for idx, res in residuals.items():
        # LayerNorm → (n_tok, d_model)
        r_t = torch.from_numpy(res).to(device)
        if ln is not None:
            with torch.no_grad():
                r_n = ln(r_t).float().cpu().numpy()
        else:
            r_n = res
        logits = r_n @ W_e          # (n_tok, vocab)
        distributions[idx] = logits_to_distribution(logits).astype(np.float32)

    return distributions


def _get_layers(model):
    """Return the list of transformer layer modules."""
    # GPT-2
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # BERT
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        return model.bert.encoder.layer
    # Generic HF
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Cannot identify transformer layers for LM head extraction.")


def _get_final_ln(model):
    """Return the final LayerNorm (or None if not found)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "bert") and hasattr(model.bert, "pooler"):
        return None  # BERT doesn't have a single final LN before head
    return None


def _get_unembed(model) -> np.ndarray:
    """Return the unembedding matrix W_e as (d_model, vocab) numpy array."""
    import torch
    if hasattr(model, "lm_head"):
        W = model.lm_head.weight.detach().float().cpu().numpy()  # (vocab, d)
        return W.T
    if hasattr(model, "cls"):
        W = model.cls.predictions.decoder.weight.detach().float().cpu().numpy()
        return W.T
    raise RuntimeError("Cannot find unembedding matrix.")


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def save_logit_cache(
    distributions: dict,
    out_path:      Path,
) -> None:
    """Save layer-wise distributions to compressed npz."""
    arrays = {f"layer_{k}": v for k, v in distributions.items()}
    np.savez_compressed(out_path, **arrays)


def load_logit_cache(path: Path) -> dict:
    """Load from npz. Returns {layer_idx: (n_tokens, vocab)}."""
    data = np.load(path)
    out  = {}
    for key in data.files:
        if key.startswith("layer_"):
            idx = int(key.split("_")[1])
            out[idx] = data[key]
    return out


# ==========================================================================
# p5b_manifold/report.py — LLM-friendly text summary
# ==========================================================================

from pathlib import Path as _Path


def write_report(
    out_dir: "_Path",
    results: dict,
    model:   str = "unknown",
    prompt:  str = "unknown",
) -> "_Path":
    """
    Write p5b_report.txt — flat text summarizing all sub-experiment results.

    Parameters
    ----------
    out_dir : directory to write into
    results : dict with keys fit_summary, isometry, teleportation, subspace
    model   : model name string
    prompt  : prompt key

    Returns
    -------
    path to written report file
    """
    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "p5b_report.txt"

    lines = []
    W = lines.append

    W("=" * 72)
    W("PHASE 5b — METASTABLE STATES AS ACTIVATION MANIFOLD CONTROL POINTS")
    W("=" * 72)
    W("")
    W("CONTEXT")
    W("-" * 40)
    W("Wurgaft et al. (2026) show that concept centroids lie on an activation")
    W("manifold Mh that is approximately isometric to a behavior manifold My")
    W("(fit to output probability distributions). This phase tests whether our")
    W("unsupervised HDBSCAN cluster centroids (from Phase 1) are the same")
    W("objects as Wurgaft's concept centroids.")
    W("")
    W(f"  Model  : {model}")
    W(f"  Prompt : {prompt}")
    W("")

    # --- Sub-exp A ---
    W("=" * 72)
    W("SUB-EXP A — MANIFOLD FITTING")
    W("=" * 72)
    fs = results.get("fit_summary", {})
    W(f"  PCA explained variance (32d)   : {fs.get('pca_explained_var', float('nan')):.4f}")
    W(f"  PCA dims needed for 80%        : {fs.get('pca_n_dims_for_80pct', '?')}")
    W(f"  Mh spline residual RMS         : {fs.get('mh_spline_residual_rms', float('nan')):.4f}")
    W(f"  My spline residual RMS         : {fs.get('my_spline_residual_rms', float('nan')):.4f}")
    W(f"  N control points               : {fs.get('n_control_points', '?')}")
    W("")
    W("  Predictions:")
    _pf(W, "P5b-A1", fs.get("p5b_a1_pass"),
        "32d PCA retains ≥ 80% variance")
    _pf(W, "P5b-A2", fs.get("p5b_a2_pass"),
        "Spline residuals < 10% of inter-centroid distances")
    W("")

    # --- Sub-exp B ---
    W("=" * 72)
    W("SUB-EXP B — ISOMETRY TEST")
    W("=" * 72)
    iso = results.get("isometry", {})
    W(f"  Pearson r (manifold vs behavior): {iso.get('r_manifold', float('nan')):.4f}")
    W(f"  Pearson r (linear vs behavior)  : {iso.get('r_linear',   float('nan')):.4f}")
    W(f"  p-value (manifold)              : {iso.get('p_manifold', float('nan')):.2e}")
    W(f"  N pairs                         : {iso.get('n_pairs', '?')}")
    W("")
    W("  Wurgaft reference values (concept-labeled tasks):")
    W("    weekdays r=0.99, months r=0.89, letters r=0.999, ages r=0.999")
    W("")
    W("  Predictions:")
    _pf(W, "P5b-B1", iso.get("p5b_b1_pass"),
        "r_manifold > r_linear")
    _pf(W, "P5b-B2", iso.get("p5b_b2_pass"),
        "r_manifold > 0.7")
    _pf(W, "P5b-B3", iso.get("p5b_b3_pass"),
        "r_manifold − r_linear > 0.1")
    W("")

    # --- Sub-exp C ---
    W("=" * 72)
    W("SUB-EXP C — MERGE-EVENT TELEPORTATION")
    W("=" * 72)
    tel = results.get("teleportation", {})
    W(f"  N merge events analyzed         : {tel.get('n_merge_events', '?')}")
    W(f"  KL divergence — merge (mean)    : {tel.get('kl_mean_merge', float('nan')):.4f}")
    W(f"  KL divergence — plateau (mean)  : {tel.get('kl_mean_plateau', float('nan')):.4f}")
    W(f"  KL p-value (Mann-Whitney U)     : {tel.get('kl_pvalue', float('nan')):.2e}")
    W(f"  Non-adj mass — merge (mean)     : {tel.get('nam_mean_merge', float('nan')):.4f}")
    W(f"  Non-adj mass — plateau (mean)   : {tel.get('nam_mean_plateau', float('nan')):.4f}")
    W("")
    W("  Predictions:")
    _pf(W, "P5b-C1", tel.get("p5b_c1_pass"),
        "KL(merge) > KL(plateau), p < 0.05")
    _pf(W, "P5b-C3", tel.get("p5b_c3_pass"),
        "Non-adjacent mass higher at merge layers")
    W("")

    # --- Sub-exp D ---
    W("=" * 72)
    W("SUB-EXP D — S-SUBSPACE ISOMETRY")
    W("=" * 72)
    sub = results.get("subspace", {})
    W(f"  r_S  (real/symmetric subspace)  : {sub.get('r_S',      float('nan')):.4f}")
    W(f"  r_A  (imag/antisymmetric)       : {sub.get('r_A',      float('nan')):.4f}")
    W(f"  r_full (full activation space)  : {sub.get('r_full',   float('nan')):.4f}")
    W(f"  r_linear (Euclidean baseline)   : {sub.get('r_linear', float('nan')):.4f}")
    W("")
    W("  Predictions:")
    _pf(W, "P5b-D1", sub.get("p5b_d1_pass"),
        "r_S > r_full ≥ r_A (S subspace = manifold coordinate system)")
    _pf(W, "P5b-D2", sub.get("p5b_d2_pass"),
        "r_A ≈ r_linear (A subspace carries no extra behavior geometry)")
    W("")

    # --- Overall verdict ---
    W("=" * 72)
    W("FALSIFICATION SUMMARY")
    W("=" * 72)
    all_flags = [
        ("P5b-A1", fs.get("p5b_a1_pass")),
        ("P5b-A2", fs.get("p5b_a2_pass")),
        ("P5b-B1", iso.get("p5b_b1_pass")),
        ("P5b-B2", iso.get("p5b_b2_pass")),
        ("P5b-B3", iso.get("p5b_b3_pass")),
        ("P5b-C1", tel.get("p5b_c1_pass")),
        ("P5b-C3", tel.get("p5b_c3_pass")),
        ("P5b-D1", sub.get("p5b_d1_pass")),
        ("P5b-D2", sub.get("p5b_d2_pass")),
    ]
    n_pass = sum(1 for _, v in all_flags if v is True)
    n_fail = sum(1 for _, v in all_flags if v is False)
    n_unk  = sum(1 for _, v in all_flags if v is None)
    W(f"  PASS: {n_pass}   FAIL: {n_fail}   UNTESTED: {n_unk}")
    W("")
    for name, val in all_flags:
        marker = "[PASS]" if val is True else ("[FAIL]" if val is False else "[n/a ]")
        W(f"  {marker} {name}")
    W("")
    if n_fail == 0 and n_pass >= 6:
        W("  VERDICT: Strong evidence that metastable cluster centroids are the")
        W("  same objects as Wurgaft's activation manifold control points.")
        W("  The Geshkovski attractor landscape IS the activation manifold Mh.")
    elif iso.get("p5b_b1_pass") is False:
        W("  VERDICT: Metastable states are dynamically real (Phase 1) but do")
        W("  not recapitulate the semantic isometry of concept-labeled tasks.")
        W("  The objects may differ in granularity: Wurgaft's Mh is concept-")
        W("  specific; ours is the full token-clustering attractor landscape.")
    else:
        W("  VERDICT: Partial — see per-prediction results above.")

    text = "\n".join(lines)
    path.write_text(text)
    return path


def _pf(W, name: str, val, desc: str) -> None:
    marker = "[PASS]" if val is True else ("[FAIL]" if val is False else "[n/a ]")
    W(f"  {marker} {name}: {desc}")


# ==========================================================================
# p5b_manifold/run_5b.py — CLI entry point
# ==========================================================================

"""
run_5b.py — Phase 5b CLI entry point.

Usage
-----
  python -m p5b_manifold.run_5b \\
      --model gpt2-large \\
      --phase1-dir results/phase1 \\
      --phase2-dir results/phase2 \\
      --out results/phase5b

  python -m p5b_manifold.run_5b --fast   # one model, one prompt

Each sub-experiment saves its fragment as soon as it finishes.
A crash in sub-exp C leaves A and B results on disk.
"""

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib  import Path

import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 5b — Manifold isometry test")
    p.add_argument("--model",       default="gpt2-large")
    p.add_argument("--models",      nargs="+", default=None)
    p.add_argument("--phase1-dir",  default="results/phase1")
    p.add_argument("--phase2-dir",  default="results/phase2")
    p.add_argument("--phase5-dir",  default="results/phase5")
    p.add_argument("--out",         default=None)
    p.add_argument("--prompt",      default="wiki_paragraph")
    p.add_argument("--pca-dim",     type=int, default=32)
    p.add_argument("--n-geodesic-pts", type=int, default=150)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--skip-logit-cache", action="store_true",
                   help="Skip logit extraction (use cached file if present)")
    p.add_argument("--fast", action="store_true",
                   help="One model, one prompt, small geodesic grid")
    return p


def _run_one(args, model_name: str, out_dir: Path) -> int:
    from p5b_manifold.manifold_fit import (
        pca_reduce, arc_length_params,
        fit_activation_manifold, fit_behavior_manifold,
        load_plateau_centroids, compute_fit_summary,
    )
    from p5b_manifold.isometry_test  import run_isometry_test
    from p5b_manifold.merge_teleportation import run_merge_teleportation
    from p5b_manifold.subspace_isometry   import subspace_isometry_score
    from p5b_manifold.logit_cache import (
        extract_layer_logits, save_logit_cache, load_logit_cache
    )
    from p5b_manifold.report import write_report

    # ---- Load Phase 1 artifacts ----
    stem = model_name.replace("-", "_").replace("/", "_")
    p1_dir = Path(args.phase1_dir)
    runs = sorted(p1_dir.glob(f"{stem}_*"))
    if not runs:
        print(f"  [skip] no Phase 1 run found for {stem}")
        return 1
    run_dir = runs[-1]   # most recent
    print(f"  Phase 1 run: {run_dir}")

    clustering = json.loads((run_dir / "clustering.json").read_text())
    centroid_path = run_dir / "centroid_trajectories.npz"

    # Extract plateau and merge layers from clustering.json
    layers_data    = clustering.get("layers", [])
    plateau_layers = [
        l["layer"] for l in layers_data
        if l.get("plateau", False)
    ]
    merge_events = [
        l["layer"] for l in layers_data
        if l.get("merge_event", False)
    ]
    all_cluster_ids = list(range(clustering.get("n_trajectories", 8)))

    print(f"  Plateau layers: {len(plateau_layers)}, merge events: {len(merge_events)}")

    # ---- Sub-exp A: Manifold fitting ----
    print("[5b] Sub-exp A: manifold fitting")
    centroids, mask = load_plateau_centroids(
        centroid_path, plateau_layers, all_cluster_ids
    )
    n_c = centroids.shape[0]
    if n_c < 4:
        print(f"  [skip] only {n_c} cluster centroids — need ≥ 4")
        return 1

    pca_dim = min(args.pca_dim, n_c - 1)
    scores, basis, evr = pca_reduce(centroids, pca_dim)

    u = arc_length_params(scores, periodic=False)
    mh = fit_activation_manifold(scores, u, periodic=False)

    # Logit cache
    logit_cache_path = out_dir / "logit_cache.npz"
    if logit_cache_path.exists() or args.skip_logit_cache:
        logit_dists = load_logit_cache(logit_cache_path) if logit_cache_path.exists() else {}
    else:
        from core.models import load_model
        from core.config  import PROMPTS
        print("  extracting logits (re-forward pass)...")
        model, tokenizer = load_model(model_name, device=args.device)
        prompt_text = PROMPTS.get(args.prompt, args.prompt)
        logit_dists = extract_layer_logits(
            model, tokenizer, prompt_text,
            layer_idxs=plateau_layers + merge_events,
            device=args.device,
        )
        save_logit_cache(logit_dists, logit_cache_path)
        del model  # free memory

    if not logit_dists:
        print("  [warn] no logit distributions — Sub-exps B and C will be incomplete")

    # Build My from plateau distributions
    my = None
    if logit_dists and len(plateau_layers) >= 4:
        dists_stack = np.stack([
            logit_dists[l].mean(axis=0)
            for l in plateau_layers if l in logit_dists
        ], axis=0)   # (n_pl, vocab)
        u_pl = arc_length_params(dists_stack, periodic=False)
        my   = fit_behavior_manifold(dists_stack, u_pl, periodic=False)

    fit_sum = compute_fit_summary(mh, my or mh, evr)   # fallback to mh if no my
    (out_dir / "fit_summary.json").write_text(json.dumps(fit_sum, indent=2))
    np.savez_compressed(
        out_dir / "mh_params.npz",
        control_pts=mh["control_pts"],
        u_knots=mh["u_knots"],
        pca_basis=basis,
    )
    print(f"  A done. PCA var={fit_sum['pca_explained_var']:.3f}, "
          f"Mh residual={fit_sum['mh_spline_residual_rms']:.4f}")

    # ---- Sub-exp B: Isometry ----
    iso_result = {}
    if my is not None and len(u) == len(u_pl):
        print("[5b] Sub-exp B: isometry test")
        n_pts = 50 if args.fast else args.n_geodesic_pts
        iso_result = run_isometry_test(mh, my, u, scores, n_pts=n_pts)
        (out_dir / "isometry.json").write_text(
            json.dumps({k: v for k, v in iso_result.items()
                        if k != "_mds"}, indent=2)
        )
        mds = iso_result.get("_mds", {})
        if mds:
            np.savez_compressed(
                out_dir / "isometry_mds.npz",
                **{f"mds_{k}": v for k, v in mds.items()}
            )
        print(f"  B done. r_manifold={iso_result.get('r_manifold', 'n/a'):.3f}, "
              f"r_linear={iso_result.get('r_linear', 'n/a'):.3f}")
    else:
        print("  [skip] Sub-exp B: My not available")

    # ---- Sub-exp C: Merge teleportation ----
    tel_result = {}
    if logit_dists and merge_events:
        print("[5b] Sub-exp C: merge teleportation")
        tel_result = run_merge_teleportation(logit_dists, merge_events, plateau_layers)
        (out_dir / "merge_teleportation.json").write_text(
            json.dumps(tel_result, indent=2, default=float)
        )
        print(f"  C done. {tel_result.get('n_merge_events', 0)} events. "
              f"P5b-C1={'PASS' if tel_result.get('p5b_c1_pass') else 'FAIL'}")
    else:
        print("  [skip] Sub-exp C: no merge events or no logits")

    # ---- Sub-exp D: S-subspace isometry ----
    sub_result = {}
    d_behavior = np.array(iso_result.get("d_behavior", []))
    if len(d_behavior) > 0:
        print("[5b] Sub-exp D: S-subspace isometry")
        p2_dir = Path(args.phase2_dir)
        proj_path = next(p2_dir.glob(f"ov_projectors_{stem}*.npz"), None)
        if proj_path:
            proj_data = np.load(proj_path)
            U_S = proj_data.get("U_pos", proj_data.get("U_attract", None))
            U_A = proj_data.get("U_A",   proj_data.get("U_imag",    None))
            if U_S is not None and U_A is not None:
                sub_result = subspace_isometry_score(centroids, U_S, U_A, d_behavior)
                (out_dir / "subspace_isometry.json").write_text(
                    json.dumps(sub_result, indent=2)
                )
                print(f"  D done. r_S={sub_result['r_S']:.3f}, "
                      f"r_A={sub_result['r_A']:.3f}, r_full={sub_result['r_full']:.3f}")
        else:
            print(f"  [skip] Sub-exp D: no projectors found in {p2_dir}")
    else:
        print("  [skip] Sub-exp D: no behavior distances from B")

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
        prompt=args.prompt,
    )
    print(f"  report: {report_path}")
    return 0


def main(argv=None) -> int:
    args   = build_argparser().parse_args(argv)
    models = args.models or [args.model]
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    base   = Path(args.out) if args.out else Path("results/phase5b")

    for model_name in models:
        stem    = model_name.replace("-", "_").replace("/", "_")
        out_dir = base / f"{stem}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\n[phase5b] {model_name}\n{'='*60}")
        try:
            rc = _run_one(args, model_name, out_dir)
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
