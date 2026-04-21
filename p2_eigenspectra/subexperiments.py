"""
subexperiments.py — p2_eigenspectra sub-experiment registry and orchestrator.

Registry
--------
SUBEXPERIMENTS : ordered list of SubexperimentSpec.
  Order matters: later entries may read ctx["{name}_result"] set by earlier ones.

Orchestrator
------------
run_one_prompt(ctx, output_dir) -> dict
  Iterates SUBEXPERIMENTS, writes sub/{name}.json + sub/{name}.summary.txt,
  builds verdict, writes verdict.json + summary.txt.
  Replaces the per-prompt block in both run_full and run_offline.

IO helpers
----------
_write_subresult  : writes .json + .summary.txt for one SubResult
_write_summary_txt: assembles the full per-run summary.txt
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np

from phase2.subresult import SubResult, SubexperimentSpec
from phase2.subexp_wrappers import (
    _trajectory_subexp,
    _layer_v_events_subexp,
    _head_ov_subexp,
    _decomposed_violations_subexp,
    _ffn_subspace_subexp,
    _continuous_correlations_subexp,
    _ov_norm_confound_subexp,
    _zone_comparison_subexp,
    _attractive_zone_violations_subexp,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SUBEXPERIMENTS: list[SubexperimentSpec] = [
    SubexperimentSpec(
        name="trajectory",
        run=_trajectory_subexp,
        requires=["traj", "ov_data"],
    ),
    SubexperimentSpec(
        name="layer_v_events",
        run=_layer_v_events_subexp,
        requires=["ov_data", "phase1_run_dir"],
        applicable=lambda c: c["ov_data"].get("is_per_layer", False),
    ),
    SubexperimentSpec(
        name="head_ov",
        run=_head_ov_subexp,
        requires=["ov_data", "phase1_run_dir"],
        applicable=lambda c: "ov_per_head" in c["ov_data"],
    ),
    SubexperimentSpec(
        name="decomposed_violations",
        run=_decomposed_violations_subexp,
        requires=["trajectory_result", "decomposed"],
    ),
    SubexperimentSpec(
        name="ffn_subspace",
        run=_ffn_subspace_subexp,
        requires=["ov_data", "phase1_run_dir", "stem_dir"],
    ),
    SubexperimentSpec(
        name="continuous_correlations",
        run=_continuous_correlations_subexp,
        requires=["ov_data", "phase1_run_dir"],
        applicable=lambda c: c["ov_data"].get("is_per_layer", False),
    ),
    SubexperimentSpec(
        name="ov_norm_confound",
        run=_ov_norm_confound_subexp,
        requires=["ov_data", "phase1_run_dir"],
        applicable=lambda c: c["ov_data"].get("is_per_layer", False),
    ),
    SubexperimentSpec(
        name="zone_comparison",
        run=_zone_comparison_subexp,
        requires=["ov_data", "phase1_run_dir"],
        applicable=lambda c: c["ov_data"].get("is_per_layer", False),
    ),
    SubexperimentSpec(
        name="attractive_zone_violations",
        run=_attractive_zone_violations_subexp,
        requires=["ffn_subspace_result", "layer_v_events_result"],
    ),
]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_one_prompt(ctx: dict, output_dir: Path) -> dict:
    """
    Run all sub-experiments for one (model, prompt) pair.

    Parameters
    ----------
    ctx : shared context dict.  Must contain at minimum:
          model_name, prompt_key, stem, ov_data, traj, phase1_run_dir.
          Additional keys (decomposed, etc.) are populated by the caller
          before this function is invoked, or forwarded between sub-experiments.
    output_dir : phase2 output directory.  Per-run files go under
                 output_dir/{ctx['stem']}/sub/.

    Returns
    -------
    verdict dict (same structure as build_verdict_v2 output)
    """
    from phase2.verdict_v2 import build_verdict_v2_from_subresults
    from phase2.reporting import save_verdict

    stem_dir = Path(output_dir) / ctx["stem"]
    sub_dir  = stem_dir / "sub"
    sub_dir.mkdir(parents=True, exist_ok=True)
    ctx["stem_dir"] = stem_dir

    subresults: dict[str, SubResult] = {}

    for spec in SUBEXPERIMENTS:
        ok, reason = spec.prerequisites_met(ctx)
        if not ok:
            sr = SubResult(
                name=spec.name, applicable=False,
                payload={},
                summary_lines=[f"{spec.name}: skipped — {reason}"],
                verdict_contribution={},
            )
        else:
            try:
                sr = spec.run(ctx)
            except Exception as exc:
                tb = traceback.format_exc()
                sr = SubResult(
                    name=spec.name, applicable=False,
                    payload={"error": str(exc), "traceback": tb},
                    summary_lines=[f"{spec.name}: FAILED — {type(exc).__name__}: {exc}"],
                    verdict_contribution={},
                    error=str(exc),
                )

        _write_subresult(sub_dir, sr)
        subresults[spec.name] = sr

        # Feed forward: later specs may depend on this result via ctx
        ctx[f"{spec.name}_result"] = sr.payload if sr.applicable else None

    verdict = build_verdict_v2_from_subresults(subresults, ctx)
    save_verdict(verdict, stem_dir)
    _write_summary_txt(stem_dir, subresults, verdict, ctx)
    return verdict


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _write_subresult(sub_dir: Path, sr: SubResult) -> None:
    """Write {name}.json and {name}.summary.txt to sub_dir."""
    payload = {
        "name":                 sr.name,
        "applicable":           sr.applicable,
        "payload":              sr.payload,
        "verdict_contribution": sr.verdict_contribution,
        "error":                sr.error,
    }
    (sub_dir / f"{sr.name}.json").write_text(
        json.dumps(_jsonify(payload), indent=2)
    )
    (sub_dir / f"{sr.name}.summary.txt").write_text(
        "\n".join(sr.summary_lines)
    )


def _write_summary_txt(
    run_dir: Path,
    subresults: dict[str, SubResult],
    verdict: dict,
    ctx: dict,
) -> None:
    """
    Assemble and write the LLM-friendly summary.txt for one run.

    Structure:
      HEADER            — model, prompt, dims, phase dirs
      VERDICT           — falsification category + key scalars
      SUB-EXPERIMENTS   — one titled block per sub-experiment, in registry order
      KEY NUMBERS       — flat scalar table from all verdict_contribution dicts
    """
    lines = []

    # --- Header ---
    sep = "=" * 72
    lines += [sep, "P2_EIGENSPECTRA RUN SUMMARY", sep]
    lines += [
        f"Model:    {ctx.get('model_name', '?')}",
        f"Prompt:   {ctx.get('prompt_key', '?')}",
        f"Layers:   {ctx['ov_data'].get('d_model', '?')}d  "
                  f"n_layers={len(ctx['ov_data'].get('layer_names', []))}",
        f"Phase 1:  {ctx.get('phase1_run_dir', '?')}",
        f"Phase 2:  {run_dir}",
    ]

    # --- Verdict ---
    lines += ["", sep, "VERDICT", sep]
    lines += _verdict_lines(verdict)

    # --- Sub-experiments ---
    for spec in SUBEXPERIMENTS:
        if spec.name not in subresults:
            continue
        sr = subresults[spec.name]
        lines += ["", sep, f"SUB-EXPERIMENT: {spec.name}", sep]
        lines += sr.summary_lines

    # --- Key numbers flat table ---
    lines += ["", sep, "KEY NUMBERS (flat reference)", sep]
    all_vc = {}
    for sr in subresults.values():
        all_vc.update(sr.verdict_contribution)
    # Also pull top-level verdict scalars not in vc
    for k, v in verdict.items():
        if k not in all_vc and isinstance(v, (int, float, bool, str, type(None))):
            all_vc[k] = v
    for k, v in sorted(all_vc.items()):
        if v is None:
            lines.append(f"{k:<45s} n/a")
        elif isinstance(v, float):
            lines.append(f"{k:<45s} {v:.4f}")
        else:
            lines.append(f"{k:<45s} {v}")

    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def _verdict_lines(verdict: dict) -> list[str]:
    L = []
    L.append(f"Falsification: {verdict.get('falsification', '?')}")
    L.append(f"Channel:       {verdict.get('channel', '?')}")
    L.append(f"V-score:       {verdict.get('v_score', float('nan')):.3f}")
    L.append(f"Model:         {verdict.get('model', '?')}")
    L.append(f"Prompt:        {verdict.get('prompt', '?')}")
    # Key scalars driving the verdict
    for k in [
        "beta1.0_n_violations", "beta1.0_frac_repulsive",
        "rescaled_improvement_beta1.0", "frac_ffn_amplifies_repulsive",
        "ov_norm_partial_rho",
    ]:
        v = verdict.get(k)
        if v is not None:
            L.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return L


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _jsonify(obj):
    """Recursively convert numpy / Python types to JSON-serialisable natives."""
    if isinstance(obj, (complex, np.complexfloating)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj
