"""
p2b_imaginary/run_2b.py — Phase 2b runner.

Replaces `run_2i.py`. Four structural changes, each of which is a defect in
the version it replaces.

**The checkpoint axis is the outer loop.** `run_2i.py` was organised
model x prompt, so the 27 Pythia checkpoints arrive as 27 unrelated "models"
with no step, no ordering, and no way to express the trajectory that is Phase
1's and Phase 2's actual headline result. Here the outer loop is
`(step, stem)` from `p2b_io.discover_checkpoints`, every artifact carries
`checkpoint_step`, and the combined file is indexed by it.

**Weights-only work happens once per checkpoint, not once per prompt.**
Block 1a reads no activations, and the `expm` rescalers Block 1b needs are
prompt-independent. `run_2i.py` recomputed both inside every (model, prompt)
pair: on the Study B sweep that is 27 x 9 x 3 x 24 exponentials of a
1024x1024 matrix where 27 x 3 x 24 suffice. The rescaler cache is built once
per checkpoint and passed down.

**Errors are not swallowed.** `run_2i.py` wrapped each prompt's work in a
bare `try/except Exception` that recorded `{"error": ...}` and continued.
That is how Block 4 shipped raising `NameError` on every prompt of every run
without anyone noticing — the summary still wrote. Here the default is to
raise. `--continue-on-error` restores the old behaviour but the run summary
reports the failure count in its first three lines and the combined file
carries `n_failed`, so a silently-empty block cannot look like a completed
one.

**Blocks are selected explicitly and are not nested inside each other's
gates.** `run_2i.py` placed Blocks 3 and 4 after `if not run_block2: return`,
so on the (constant) `rotation_neutral` verdict neither was reachable at all.
`--blocks` names what to run; nothing gates anything else.

Blocks 2, 3 and 4 are deliberately absent from `BLOCKS` until their maths is
redefined — see `PLAN_2b.md` items 10-12. Wiring a block whose result is
degenerate by construction is what produced the previous status table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from p2b_imaginary import p2b_io
from p2b_imaginary import rotational_schur as schur_block
from p2b_imaginary import rotational_rescaled as rescaled_block
from p2b_imaginary.p2b_energy import (
    DEFAULT_GATE_KIND,
    cross_check_against_phase1,
)

#: Blocks this runner can execute. Names match `core.artifacts.PHASE2B`.
BLOCKS = ("1a", "1b")

#: Study B's beta. Phase 1 found violation counts beta-independent after step
#: 512 and Study B ran beta=1.0 only, so the default is one beta rather than
#: the four `run_2i.py` swept and then majority-voted over.
DEFAULT_BETAS = (1.0,)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def estimate_cost(n_checkpoints: int, n_prompts: int, n_layers: int,
                  d_model: int, blocks: Sequence[str],
                  with_nulls: bool = False, n_null_draws: int = 16,
                  with_precision: bool = False) -> dict:
    """
    Rough wall-time estimate, so a sweep that will not finish is known before
    it starts rather than after.

    Both blocks are dominated by O(d^3) dense factorisations: `schur` for
    Block 1a, `expm` for Block 1b. The constant is calibrated against a
    1024x1024 float64 factorisation at roughly 1 second, which is the right
    order on a single core and pessimistic on many.
    """
    unit = (d_model / 1024.0) ** 3          # seconds per d^3 factorisation

    schur_calls = 0
    expm_calls = 0
    eig_calls = 0
    if "1a" in blocks:
        schur_calls += n_checkpoints * n_layers
        if with_nulls:
            # One null SAMPLE per layer, every statistic read off it
            # (`rotational_schur.null_comparison_multi`). This line used to
            # undercount by len(NULL_STATISTICS), because the pipeline drew a
            # fresh sample per statistic; the fix was to share the sample
            # rather than to multiply the estimate.
            schur_calls += n_checkpoints * n_layers * n_null_draws
        if with_precision:
            # len(DEFAULT_TOLS) tolerances x {baseline, perturbed}, each a
            # dense eigendecomposition. Same O(d^3) order as a Schur pass,
            # counted in the same units.
            from core.precision_policy import DEFAULT_TOLS
            eig_calls += n_checkpoints * n_layers * 2 * len(DEFAULT_TOLS)
    if "1b" in blocks:
        # 3 frames x n_layers, ONCE per checkpoint (cached across prompts).
        expm_calls += n_checkpoints * 3 * n_layers
        # Plus the S/A Schur-free decomposition, which is O(d^2) and ignored.

    seconds = unit * (schur_calls + expm_calls + eig_calls)
    return {
        "schur_calls": int(schur_calls),
        "expm_calls": int(expm_calls),
        "eig_calls": int(eig_calls),
        "estimated_seconds": float(seconds),
        "estimated_hours": float(seconds / 3600.0),
        "note": (
            "Block 1b's per-prompt cost is the Gram/energy loop, which is "
            "O(n_tokens^2 d) and small next to the factorisations. The "
            "rescalers are cached per checkpoint; without that cache the "
            f"expm count would be {n_prompts}x higher."
        ),
    }


# ---------------------------------------------------------------------------
# Block 1a — weights only, once per checkpoint
# ---------------------------------------------------------------------------

def run_block_1a(ov_data: dict, out_dir: Path, *, top_k_planes: int = 0,
                 with_planes: bool = True, with_nulls: bool = False,
                 with_precision: bool = False,
                 n_null_draws: int = 16, seed: int = 0) -> dict:
    """
    Rotational spectrum for one checkpoint.

    No activations, no forward pass, no prompts. This is the cheapest thing in
    the phase and the one that answers whether the 84-97% complex fraction has
    a developmental trajectory at all — see `PLAN_2b.md` open question 1.

    Writes two files: the subresult JSON, and — when `with_planes` — a
    `planes.npz` sidecar holding every 2x2 block's (rho, theta, sign, idx) per
    layer. The npz is the spectrum; the JSON keeps quantiles of it. Splitting
    them is a size decision and nothing else: at d = 1024 a layer holds up to
    512 planes, and `phase2b_results.json` embeds every checkpoint's Block 1a
    JSON and is read whole.
    """
    res = schur_block.analyze_rotational_spectrum(
        ov_data,
        top_k_planes=top_k_planes,
        with_planes=with_planes,
        with_nulls=with_nulls,
        n_null_draws=n_null_draws,
        rng=np.random.default_rng(seed),
    )
    js = schur_block.summary_to_json(res)

    if with_precision:
        # `core/precision_policy.py` item P2: the "84-97% complex" figure uses
        # a RELATIVE criterion (|Im| > tol*(|Re| + eps)), and a relative
        # criterion is exactly what an fp16-epsilon split of a genuinely real
        # eigenvalue pair defeats — the split is small in absolute terms and
        # unbounded in ratio when |Re| is also small. So the honest answer to
        # "how complex is OV" is a surface over (tolerance, perturbation), not
        # a scalar. The module that computes it was written against this block
        # and was never called from here, which is why P2 has been a caveat in
        # prose with no number attached since it was raised.
        #
        # Off by default because it costs len(DEFAULT_TOLS) x 2 dense
        # eigendecompositions per layer — about 10x Block 1a's own Schur pass.
        # Worth running at a subset of checkpoints, not all 27.
        ov_list = (list(ov_data["ov_total"]) if ov_data["is_per_layer"]
                   else [ov_data["ov_total"]])
        js["precision"] = schur_block.precision_surface(
            ov_list, list(js["layer_names"]))

    p2b_io.write_subresult(out_dir, "block1a_rotational_spectrum", js,
                           schur_block.summary_lines(js))
    if with_planes:
        p2b_io.write_sidecar(out_dir, "planes",
                             schur_block.planes_npz_arrays(res))
    return js


def run_head_circuits(ov_data: dict, out_dir: Path, *,
                      d_head: Optional[int] = None) -> Optional[dict]:
    """
    Per-head circuit algebra for one checkpoint. Weights only.

    THE OBJECT THE HEADLINE IS A STATISTIC OF. `ov_total = sum_h ov_per_head`
    (`weights.py:184`) is the effective operator only under a counterfactual
    the model does not satisfy — that every head shares an attention pattern.
    The real update is `sum_h alpha^h X W_OV^h`. So "OV is 84-97.5% complex"
    is a statement about a matrix the model never forms, and whether it also
    describes any HEAD is a separate question that this phase has never
    asked of an artifact.

    `head_circuits.py` was written to ask it and landed with its own tests
    (`PLAN_2b.md` item 19); it was simply never called from the runner, so no
    artifact carried `summed_vs_per_head`, `head_agreement`, or a per-head
    spectrum. This is that call.

    Cheap, and cheaper than it looks: per-head `W_OV` has rank `d_head`, so
    every spectrum here is a `d_head^2` problem rather than a `d_model^2`
    one — 16 x 64^3 against 1024^3 per layer at 410m, a 256x reduction that
    grows with model size.

    Returns None when the OV npz carries no per-head arrays, which is what a
    weights file written before `weights.py` saved them looks like.
    """
    from p2b_imaginary import head_circuits as hc

    per_layer_input = bool(ov_data["is_per_layer"])
    per_head = ov_data.get("ov_per_head")
    if not per_head or (per_layer_input and not any(per_head)):
        return None

    layer_names = list(ov_data.get("layer_names") or [])
    totals = (list(ov_data["ov_total"]) if per_layer_input
              else [ov_data["ov_total"]])
    heads_by_layer = per_head if per_layer_input else [per_head]

    per_layer = []
    for name, total, heads in zip(layer_names, totals, heads_by_layer):
        if not len(heads):
            continue
        rec = hc.summed_vs_per_head(heads, ov_total=total, d_head=d_head)
        rec["layer"] = name
        rec["n_heads"] = len(heads)
        per_layer.append(rec)

    if not per_layer:
        return None

    gaps = np.array([r["gap"] for r in per_layer], dtype=np.float64)
    agree = np.array([r["head_agreement"] for r in per_layer], dtype=np.float64)
    spread = np.array([r["head_spread"] for r in per_layer], dtype=np.float64)

    js = {
        "model_stem": ov_data.get("model_stem"),
        "checkpoint_step": ov_data.get("checkpoint_step"),
        "layer_names": layer_names,
        "per_layer": per_layer,
        "summary": {
            "n_layers": len(per_layer),
            "n_heads": int(per_layer[0]["n_heads"]),
            "gap_mean": float(np.nanmean(gaps)),
            "gap_max_abs": float(np.nanmax(np.abs(gaps))),
            "head_agreement_mean": float(np.nanmean(agree)),
            "head_agreement_min": float(np.nanmin(agree)),
            "head_spread_mean": float(np.nanmean(spread)),
            "head_spread_max": float(np.nanmax(spread)),
        },
        "note": (
            "`summed` is the statistic of sum_h W_OV^h, which is the "
            "effective operator only if every head shares an attention "
            "pattern. It is reported for continuity with the published "
            "84-97.5% figure, not because it is the operator."
        ),
    }
    p2b_io.write_subresult(out_dir, "block1a_head_circuits", js,
                           head_circuits_summary_lines(js))
    return js


def head_circuits_summary_lines(js: dict) -> list:
    s = js.get("summary", {})
    lines = [
        "--- Block 1a: per-head circuits ---",
        f"  {js.get('model_stem')}  step {js.get('checkpoint_step')}  "
        f"{s.get('n_layers', 0)} layers x {s.get('n_heads', 0)} heads",
        f"  summed - per-head mean gap: {s.get('gap_mean', float('nan')):+.4f} "
        f"(max |gap| {s.get('gap_max_abs', float('nan')):.4f})",
        f"  head agreement: mean {s.get('head_agreement_mean', float('nan')):.3f}, "
        f"min {s.get('head_agreement_min', float('nan')):.3f}",
        f"  head spread:    mean {s.get('head_spread_mean', float('nan')):.4f}, "
        f"max {s.get('head_spread_max', float('nan')):.4f}",
        "",
        "  Low agreement means the summed number describes no head in the",
        "  layer. The summed object is the one the published figure is a",
        "  statistic of; the model never forms it.",
    ]
    return lines


# ---------------------------------------------------------------------------
# Block 1b — one prompt
# ---------------------------------------------------------------------------

def run_block_1b(bundle: dict, ov_data: dict, out_dir: Path, *,
                 betas: Sequence[float], rescaler_cache: dict,
                 gate_kind: str = DEFAULT_GATE_KIND,
                 gate_threshold: Optional[float] = None,
                 model_stem: str = "", prompt_key: str = "") -> dict:
    """
    S/A rescaled-frame comparison for one (checkpoint, prompt).

    The Phase 1 cross-check is written into the artifact rather than left
    implicit: Phase 2b gates on normed effective rank and Phase 1 on raw, so
    the two counts are expected to differ, and a large disagreement means the
    GATE is doing the work rather than the rescaling.
    """
    res = rescaled_block.analyze_rotational_rescaling(
        bundle["activations"], ov_data, betas,
        rescaler_cache=rescaler_cache,
        gate_kind=gate_kind, gate_threshold=gate_threshold,
    )
    js = rescaled_block.comparison_to_json(res)

    js["model_stem"] = model_stem
    js["prompt_key"] = prompt_key
    js["checkpoint_step"] = ov_data.get("checkpoint_step")
    js["frame"] = p2b_io.frame_spec_for_activations(model_stem).to_dict()
    js["phase1_cross_check"] = {
        str(b): v for b, v in cross_check_against_phase1(
            res["frames"]["frames"]["original"]["counts"],
            bundle.get("phase1_violation_layers", {}),
        ).items()
    }

    p2b_io.write_subresult(out_dir, "block1b_rescaled_comparison", js,
                           block_1b_summary_lines(js))
    return js


def block_1b_summary_lines(js: dict) -> list:
    lines = [
        "--- Block 1b: S/A rescaled frames ---",
        f"  {js.get('model_stem')}  step {js.get('checkpoint_step')}  "
        f"prompt {js.get('prompt_key')}",
        f"  Verdict: {js['interpretation']['overall']} "
        f"(beta {js['interpretation']['reference_beta']})",
    ]
    for key, fr in js["frames"].items():
        tag = "  [invariance control]" if fr["is_invariance_control"] else ""
        trunc = ("" if not fr["truncated"]
                 else f"  TRUNCATED at {fr['n_valid_layers']} "
                      f"({fr['truncation_reason']})")
        counts = fr["counts"].get(str(js["interpretation"]["reference_beta"]), {})
        lines.append(
            f"  {key:16s} valid={fr['n_valid_layers']:3d} "
            f"viol={counts.get('n_violations', '?'):>3} "
            f"scored={counts.get('n_transitions_scored', '?'):>3} "
            f"gated={counts.get('n_transitions_gated', '?'):>3}{trunc}{tag}"
        )
    ref = str(js["interpretation"]["reference_beta"])
    for name, res in js["comparison"].get(ref, {}).items():
        rate = "n/a" if res["rate"] is None else f"{res['rate']:+.4f}"
        lines.append(f"  {name}: {rate}  [{res['status']}]")
    inv = js.get("invariance") or {}
    if inv:
        lines.append(f"  Invariance control: {inv.get('status')} "
                     f"(orthogonality residual "
                     f"{inv.get('orthogonality', {}).get('max_residual', float('nan')):.2e})")
    return lines


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_checkpoint(step: Optional[int], stem: str, *, weights_dir: Path,
                   phase1_dir: Path, out_root: Path, blocks: Sequence[str],
                   betas: Sequence[float], prompts: Optional[Sequence[str]],
                   gate_kind: str, gate_threshold: Optional[float],
                   top_k_planes: int, with_planes: bool, with_nulls: bool,
                   with_precision: bool, with_heads: bool, n_null_draws: int,
                   seed: int, continue_on_error: bool) -> dict:
    """One checkpoint: Block 1a once, Block 1b per prompt."""
    t0 = time.time()
    ckpt_dir = out_root / stem
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ov_data = p2b_io.load_ov_data(weights_dir, stem)
    if ov_data is None:
        return {"model_stem": stem, "checkpoint_step": step,
                "status": "no_ov_weights"}

    n_layers = (len(ov_data["ov_total"]) if ov_data["is_per_layer"] else 1)
    out: dict = {
        "model_stem": stem,
        "checkpoint_step": step,
        "n_ov_layers": n_layers,
        "status": "ok",
        "failures": [],
    }

    if "1a" in blocks:
        out["block1a"] = run_block_1a(
            ov_data, ckpt_dir, top_k_planes=top_k_planes,
            with_planes=with_planes, with_nulls=with_nulls,
            with_precision=with_precision, n_null_draws=n_null_draws,
            seed=seed,
        )
        if with_heads:
            heads = run_head_circuits(ov_data, ckpt_dir)
            if heads is None:
                out["head_circuits_status"] = "no_per_head_weights"
            else:
                out["head_circuits"] = heads

    if "1b" in blocks:
        runs = p2b_io.find_phase1_runs(phase1_dir, stem, prompt_keys=prompts)
        out["n_phase1_runs"] = len(runs)
        rescaler_cache: dict = {}          # built once, reused across prompts
        per_prompt: dict = {}
        for prompt_key, run_dir in sorted(runs.items()):
            try:
                bundle = p2b_io.load_phase1_run_bundle(run_dir)
                if bundle["activations"] is None:
                    per_prompt[prompt_key] = {"status": "no_activations"}
                    continue
                per_prompt[prompt_key] = run_block_1b(
                    bundle, ov_data, ckpt_dir / prompt_key,
                    betas=betas, rescaler_cache=rescaler_cache,
                    gate_kind=gate_kind, gate_threshold=gate_threshold,
                    model_stem=stem, prompt_key=prompt_key,
                )
            except Exception:
                if not continue_on_error:
                    raise
                out["failures"].append({
                    "prompt": prompt_key,
                    "block": "1b",
                    "traceback": traceback.format_exc(limit=6),
                })
                per_prompt[prompt_key] = {"status": "failed"}
        out["block1b"] = per_prompt

    p2b_io.write_run_manifest(
        ckpt_dir, stem, None, time.time() - t0,
        config={"blocks": list(blocks), "betas": list(betas),
                "gate_kind": gate_kind, "gate_threshold": gate_threshold,
                "with_planes": bool(with_planes),
                "with_nulls": bool(with_nulls),
                "with_precision": bool(with_precision),
                "with_heads": bool(with_heads), "seed": int(seed)},
    )
    out["wall_time_seconds"] = time.time() - t0
    return out


def run_sweep(weights_dir, phase1_dir, out_root, *,
              base: Optional[str] = None,
              steps: Optional[Sequence[int]] = None,
              prompts: Optional[Sequence[str]] = None,
              blocks: Sequence[str] = BLOCKS,
              betas: Sequence[float] = DEFAULT_BETAS,
              gate_kind: str = DEFAULT_GATE_KIND,
              gate_threshold: Optional[float] = None,
              top_k_planes: int = 0,
              with_planes: bool = True,
              with_nulls: bool = False,
              with_precision: bool = False,
              with_heads: bool = True,
              n_null_draws: int = 16,
              seed: int = 0,
              continue_on_error: bool = False,
              max_checkpoints: Optional[int] = None,
              expected_steps: Optional[Sequence[int]] = None) -> dict:
    """The full checkpoint sweep. Returns the combined result dict."""
    weights_dir = Path(weights_dir)
    phase1_dir = Path(phase1_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Old artifacts were scored with a different counting rule and carry an
    # `elim_rotation` column that is an algebraic identity. Refusing is
    # cheaper than a silently mixed table.
    p2b_io.refuse_legacy_run_dir(out_root)

    found = p2b_io.discover_checkpoints(weights_dir, base=base)

    # A checkpoint whose OV weights Phase 2 never wrote does not appear in
    # the glob, so it would simply be ABSENT from the sweep — 26 rows instead
    # of 27, and nothing saying which one is missing. Anything the caller
    # asked for and did not get is recorded explicitly.
    requested = None if steps is None else sorted(set(int(s) for s in steps))
    if expected_steps is not None:
        requested = sorted(set(requested or []) | set(int(s) for s in expected_steps))

    if steps is not None:
        wanted = set(int(s) for s in steps)
        found = [(st, stem) for st, stem in found if st in wanted]
    if max_checkpoints is not None:
        found = found[:int(max_checkpoints)]

    have = {st for st, _ in found}
    missing = ([s for s in requested if s not in have] if requested else [])

    t0 = time.time()
    results: dict = {}
    for step in missing:
        stem = f"{base}-step{step}" if base else f"step{step}"
        results[stem] = {"model_stem": stem, "checkpoint_step": step,
                         "status": "no_ov_weights"}
    for step, stem in found:
        results[stem] = run_checkpoint(
            step, stem, weights_dir=weights_dir, phase1_dir=phase1_dir,
            out_root=out_root, blocks=blocks, betas=betas, prompts=prompts,
            gate_kind=gate_kind, gate_threshold=gate_threshold,
            top_k_planes=top_k_planes, with_planes=with_planes,
            with_nulls=with_nulls, with_precision=with_precision,
            with_heads=with_heads, n_null_draws=n_null_draws, seed=seed,
            continue_on_error=continue_on_error,
        )

    combined = {
        "phase": "2b",
        "base": base,
        "blocks": list(blocks),
        "betas": [float(b) for b in betas],
        "counting_rule": {"gate_kind": gate_kind,
                          "gate_threshold": gate_threshold},
        "n_checkpoints": len(found),
        "n_failed": sum(len(r.get("failures", [])) for r in results.values()),
        "missing_checkpoints": missing,
        "steps": [s for s, _ in found],
        "wall_time_seconds": time.time() - t0,
        "results": results,
    }

    with open(out_root / p2b_io.COMBINED_RESULTS, "w") as f:
        # Through the same sanitizer `write_subresult` uses: the combined file
        # embeds every subresult, so a bare NaN anywhere in a block's output
        # fails the write here too — and later, after the whole sweep has run.
        json.dump(p2b_io.sanitize_for_json(combined), f, indent=2,
                  default=p2b_io.json_default, allow_nan=False)
    lines = sweep_summary_lines(combined)
    with open(out_root / p2b_io.COMBINED_SUMMARY, "w") as f:
        f.write("\n".join(lines) + "\n")

    return combined


# ---------------------------------------------------------------------------
# Cross-checkpoint summary
# ---------------------------------------------------------------------------

def sweep_summary_lines(combined: dict) -> list:
    """
    The trajectory table, which is the point of the rerun.

    Failures are reported in the first three lines. `run_2i.py` recorded them
    inside per-prompt dicts and wrote a summary that looked complete.
    """
    lines = [
        "=== Phase 2b sweep ===",
        f"Base: {combined.get('base')}   checkpoints: {combined['n_checkpoints']}"
        f"   blocks: {', '.join(combined['blocks'])}"
        f"   betas: {combined['betas']}",
        f"FAILURES: {combined['n_failed']}"
        + ("" if combined["n_failed"] == 0 else
           "   <-- results below are INCOMPLETE"),
        f"MISSING CHECKPOINTS: {combined.get('missing_checkpoints') or 'none'}"
        + ("" if not combined.get("missing_checkpoints") else
           "   <-- Phase 2 wrote no OV weights for these"),
        f"Counting rule: {combined['counting_rule']}",
        "",
    ]

    if "1a" in combined["blocks"]:
        lines += [
            "--- Block 1a trajectory (weights only) ---",
            f"{'step':>8}  {'cplx_frac':>9}  {'legacy':>7}  {'dim_frac':>8}  "
            f"{'theta':>6}  {'repuls':>6}  {'henrici':>7}",
        ]
        for stem, r in combined["results"].items():
            s = (r.get("block1a") or {}).get("summary")
            if not s:
                continue
            lines.append(
                f"{str(r.get('checkpoint_step')):>8}  "
                f"{s['complex_energy_fraction_mean']:9.4f}  "
                f"{s['complex_energy_fraction_legacy_mean']:7.4f}  "
                f"{s['dim_complex_fraction_mean']:8.4f}  "
                f"{s['theta_mean_across_layers']:6.3f}  "
                f"{s['frac_repulsive_real_part_mean']:6.3f}  "
                f"{s['henrici_relative_mean']:7.4f}"
            )
        lines += [
            "",
            "  cplx_frac is per-eigenvalue; legacy is the pre-rewrite",
            "  per-block convention that produced the 84-97.5% figure.",
            "  A flat cplx_frac from step 0 means the headline is a fact",
            "  about square matrices, not about training.",
            "",
        ]

    if "1b" in combined["blocks"]:
        lines += [
            "--- Block 1b verdicts ---",
            f"{'step':>8}  {'prompt':<24}  {'verdict':<22}  "
            f"{'elim_full':>10}  {'elim_signed':>11}  {'trunc':>5}",
        ]
        tally: dict = {}
        for stem, r in combined["results"].items():
            step = r.get("checkpoint_step")
            for prompt, js in sorted((r.get("block1b") or {}).items()):
                if "interpretation" not in js:
                    lines.append(f"{str(step):>8}  {prompt:<24}  "
                                 f"{js.get('status', 'missing'):<22}")
                    continue
                interp = js["interpretation"]
                ref = str(interp["reference_beta"])
                row = js["comparison"].get(ref, {})
                ef = row.get("elim_full", {}).get("rate")
                es = row.get("elim_signed", {}).get("rate")
                trunc = sum(1 for f in js["frames"].values() if f["truncated"])
                lines.append(
                    f"{str(step):>8}  {prompt:<24}  {interp['overall']:<22}  "
                    f"{'n/a' if ef is None else f'{ef:+.4f}':>10}  "
                    f"{'n/a' if es is None else f'{es:+.4f}':>11}  "
                    f"{trunc:>5}"
                )
                tally[interp["overall"]] = tally.get(interp["overall"], 0) + 1

        lines.append("")
        lines.append("  Verdict tally: " + (
            ", ".join(f"{k}={v}" for k, v in sorted(tally.items())) or "none"))
        lines += [
            "  `remove_rotation` is an invariance control, not a result:",
            "  e^{-A} is orthogonal for antisymmetric A, so it reproduces the",
            "  original frame by construction. It is excluded from the",
            "  comparison above by design.",
        ]

    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_2b",
        description="Phase 2b — rotational structure of V_eff across checkpoints.",
    )
    p.add_argument("--weights-dir", required=True,
                   help="directory holding ov_weights_{stem}.npz from Phase 2")
    p.add_argument("--phase1-dir", required=True,
                   help="Phase 1 run root (activations.npz, energies.json)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base", default=None,
                   help="restrict to one checkpoint family, e.g. pythia-410m")
    p.add_argument("--steps", default=None,
                   help="comma-separated checkpoint steps; default all found")
    p.add_argument("--prompts", default=None,
                   help="comma-separated prompt keys; default every run found")
    p.add_argument("--blocks", default=",".join(BLOCKS),
                   help=f"comma-separated subset of {BLOCKS}")
    p.add_argument("--betas", default=",".join(str(b) for b in DEFAULT_BETAS),
                   help="comma-separated betas. Default is 1.0 alone, matching "
                        "Phase 2 Study B; counts are beta-independent after "
                        "step 512.")
    p.add_argument("--gate-kind", default=DEFAULT_GATE_KIND,
                   choices=("normed_rank", "raw_rank", "none"))
    p.add_argument("--gate-threshold", type=float, default=None,
                   help="default: core.config.DEGENERATE_RANK_THRESHOLD")
    p.add_argument("--top-k-planes", type=int, default=0,
                   help="rotation-plane BASES ((d, 2) each) to retain per "
                        "layer (Block 1a). Separate from --no-planes, which "
                        "controls the per-plane scalars.")
    p.add_argument("--no-planes", action="store_true",
                   help="skip the per-plane (rho, theta, sign) spectrum "
                        "sidecar. It is free to compute — the Schur blocks "
                        "are already extracted — so this only trades a "
                        "~1 MB/checkpoint npz for losing the distribution "
                        "the angle statistics summarise.")
    p.add_argument("--with-nulls", action="store_true",
                   help="norm-matched Gaussian null per layer (Block 1a). "
                        "Costs n_null_draws extra Schur decompositions per "
                        "layer; run at a subset of checkpoints.")
    p.add_argument("--no-heads", action="store_true",
                   help="skip the per-head circuit block. It is cheap — "
                        "per-head W_OV has rank d_head, so every spectrum is "
                        "a d_head^2 problem — and it is the only thing in "
                        "the phase that asks whether the summed operator's "
                        "statistic describes any actual head.")
    p.add_argument("--with-precision", action="store_true",
                   help="run core.precision_policy's tolerance x fp16 "
                        "surface per layer (Block 1a). This is the number "
                        "behind precision-policy item P2 — whether "
                        "'84-97%% complex' survives the fp16 round-trip the "
                        "checkpoints went through, swept over the relative "
                        "tolerance rather than taken at the shipped 0.01. "
                        "Costs ~10 dense eigendecompositions per layer, so "
                        "run it at a subset of checkpoints.")
    p.add_argument("--n-null-draws", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-checkpoints", type=int, default=None)
    p.add_argument("--expect-registry-steps", action="store_true",
                   help="treat core.pythia_registry's canonical step list as "
                        "the expected set, so a checkpoint Phase 2 failed to "
                        "produce is reported as missing rather than silently "
                        "absent from the sweep")
    p.add_argument("--continue-on-error", action="store_true",
                   help="record and continue instead of raising. The failure "
                        "count is reported in the summary's first lines.")
    p.add_argument("--dry-run", action="store_true",
                   help="list the checkpoints and print a cost estimate")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    unknown = [b for b in blocks if b not in BLOCKS]
    if unknown:
        raise SystemExit(
            f"run_2b: unknown block(s) {unknown}. Available: {BLOCKS}. "
            "Blocks 2, 3 and 4 are deliberately unwired until their maths is "
            "redefined — see PLAN_2b.md items 10-12."
        )

    betas = [float(b) for b in args.betas.split(",") if b.strip()]
    steps = ([int(s) for s in args.steps.split(",")] if args.steps else None)
    prompts = ([p.strip() for p in args.prompts.split(",")]
               if args.prompts else None)

    if args.dry_run:
        found = p2b_io.discover_checkpoints(Path(args.weights_dir),
                                            base=args.base)
        if steps is not None:
            found = [(st, s) for st, s in found if st in set(steps)]
        if args.max_checkpoints:
            found = found[:args.max_checkpoints]
        ov = (p2b_io.load_ov_data(Path(args.weights_dir), found[0][1])
              if found else None)
        n_layers = (len(ov["ov_total"]) if ov and ov["is_per_layer"] else 1)
        d_model = (np.asarray(ov["ov_total"][0]).shape[0]
                   if ov and ov["is_per_layer"] else 0)
        cost = estimate_cost(len(found), len(prompts or []) or 9, n_layers,
                             d_model or 1024, blocks,
                             with_nulls=args.with_nulls,
                             n_null_draws=args.n_null_draws,
                             with_precision=args.with_precision)
        print(f"checkpoints: {len(found)}")
        for st, stem in found:
            print(f"  step {st:>7}  {stem}")
        print(f"layers/checkpoint: {n_layers}   d_model: {d_model}")
        print(json.dumps(cost, indent=2))
        return 0

    expected = None
    if args.expect_registry_steps:
        # Deferred: core.pythia_registry imports transformers.
        from core.pythia_registry import PYTHIA_410M_PILOT_STEPS, PYTHIA_ALL_STEPS
        expected = (PYTHIA_410M_PILOT_STEPS if args.base == "pythia-410m"
                    else PYTHIA_ALL_STEPS)

    combined = run_sweep(
        args.weights_dir, args.phase1_dir, args.output_dir,
        expected_steps=expected,
        base=args.base, steps=steps, prompts=prompts, blocks=blocks,
        betas=betas, gate_kind=args.gate_kind,
        gate_threshold=args.gate_threshold,
        top_k_planes=args.top_k_planes, with_planes=not args.no_planes,
        with_nulls=args.with_nulls, with_precision=args.with_precision,
        with_heads=not args.no_heads,
        n_null_draws=args.n_null_draws, seed=args.seed,
        continue_on_error=args.continue_on_error,
        max_checkpoints=args.max_checkpoints,
    )
    print("\n".join(sweep_summary_lines(combined)))
    return 1 if combined["n_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BLOCKS",
    "DEFAULT_BETAS",
    "estimate_cost",
    "run_block_1a",
    "run_head_circuits",
    "head_circuits_summary_lines",
    "run_block_1b",
    "run_checkpoint",
    "run_sweep",
    "sweep_summary_lines",
    "build_parser",
    "main",
]
