"""
p6_io.py — IO contract and orchestration for Phase 6.

Defines:
  SubResult          : uniform output contract for every Phase 6 sub-experiment
  SubexperimentSpec  : declarative spec (name, run, requires, applicable gate)
  save_subresult     : writes {name}.json + {name}.summary.txt
  run_phase6         : iterates the registry, feeds ctx forward, assembles report
  _jsonify           : numpy-safe JSON serialiser

Mirrors the Phase 2 pattern in subresult.py / subexperiments.py so results are
structurally identical to earlier phases.

Output layout (per model stem)
-------------------------------
  results/phase6/{stem}/
    sub/
      subspace_build.json          ← projector diagnostics
      subspace_build.summary.txt
      head_classify.json
      head_classify.summary.txt
      qk_decompose.json
      ...one pair per sub-experiment...
    phase6_report.txt              ← assembled LLM-friendly report (~1000 lines)
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

import numpy as np


# ---------------------------------------------------------------------------
# Core data types  (mirror of phase2/subresult.py)
# ---------------------------------------------------------------------------

@dataclass
class SubResult:
    """
    Uniform output of every Phase 6 sub-experiment.

    Attributes
    ----------
    name                 : sub-experiment identifier (matches registry entry)
    applicable           : False when skipped (missing prereqs, not-applicable gate)
    payload              : JSON-serialisable raw result dict → {name}.json
    summary_lines        : LLM-ready plain-text prose → {name}.summary.txt
                           Self-contained, no ANSI, lines ≤ 100 chars.
    verdict_contribution : flat scalar dict; keys merged into phase6_report.txt
                           key table.  Must be globally unique across all specs.
    error                : non-None when the sub-experiment raised; orchestrator
                           stores the traceback and continues.
    """
    name:                 str
    applicable:           bool
    payload:              dict
    summary_lines:        list[str]
    verdict_contribution: dict
    error:                str | None = None


@dataclass
class SubexperimentSpec:
    """
    Declarative spec for one Phase 6 sub-experiment.

    Attributes
    ----------
    name       : unique string identifier; used as filename stem
    run        : callable (ctx: dict) -> SubResult
    requires   : ctx keys that must be non-None for the spec to run
    applicable : optional callable (ctx: dict) -> bool; skip guard
    """
    name:       str
    run:        Callable[[dict], SubResult]
    requires:   list[str]              = field(default_factory=list)
    applicable: Callable[[dict], bool] | None = None

    def prerequisites_met(self, ctx: dict) -> tuple[bool, str]:
        missing = [k for k in self.requires if ctx.get(k) is None]
        if missing:
            return False, f"missing prerequisites: {missing}"
        if self.applicable is not None and not self.applicable(ctx):
            return False, "not applicable for this configuration"
        return True, ""


# ---------------------------------------------------------------------------
# Per-subresult IO
# ---------------------------------------------------------------------------

def save_subresult(sub_dir: Path, sr: SubResult) -> None:
    """
    Write {name}.json and {name}.summary.txt to sub_dir.

    The .json contains the full payload plus metadata.
    The .summary.txt contains only the LLM-ready summary_lines.
    """
    sub_dir = Path(sub_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)

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
        "\n".join(sr.summary_lines) + "\n"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_phase6(
    registry: list[SubexperimentSpec],
    ctx:      dict,
    out_dir:  Path,
) -> dict[str, SubResult]:
    """
    Run all registered Phase 6 sub-experiments for one model.

    Parameters
    ----------
    registry : ordered list of SubexperimentSpec.  Order matters — later
               entries may consume ctx["{name}_result"] set by earlier ones.
    ctx      : shared context dict.  Required keys depend on specs; the
               caller populates model-level artifacts before calling this.
    out_dir  : results/phase6/{stem}/ — per-sub files go under out_dir/sub/

    Returns
    -------
    dict mapping spec.name → SubResult (including skipped/failed entries)
    """
    out_dir  = Path(out_dir)
    sub_dir  = out_dir / "sub"
    sub_dir.mkdir(parents=True, exist_ok=True)

    ctx["out_dir"] = out_dir
    subresults: dict[str, SubResult] = {}

    for spec in registry:
        ok, reason = spec.prerequisites_met(ctx)
        if not ok:
            sr = SubResult(
                name=spec.name,
                applicable=False,
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
                    name=spec.name,
                    applicable=False,
                    payload={"error": str(exc), "traceback": tb},
                    summary_lines=[
                        f"{spec.name}: FAILED — {type(exc).__name__}: {exc}",
                        "  (see .json for full traceback)",
                    ],
                    verdict_contribution={},
                    error=str(exc),
                )

        save_subresult(sub_dir, sr)
        subresults[spec.name] = sr

        # Feed result forward so later specs can depend on it
        ctx[f"{spec.name}_result"] = sr.payload if sr.applicable else None

    return subresults


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy / Python types to JSON-serialisable natives."""
    if isinstance(obj, (complex, np.complexfloating)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, np.ndarray):
        if obj.size > 4096:
            # Don't embed large arrays in JSON; store shape as a note
            return f"<ndarray shape={list(obj.shape)} dtype={obj.dtype}>"
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Formatting helpers shared across summary writers
# ---------------------------------------------------------------------------

SEP_THICK = "=" * 72
SEP_THIN  = "-" * 72

def _fmt(v, p: int = 4) -> str:
    """Fixed-precision formatter. Handles None / nan / bool cleanly."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(float(v)):
            return "nan"
        return f"{v:.{p}f}"
    return str(v)


def _bullet(label: str, value: Any, p: int = 4, width: int = 42) -> str:
    return f"  {label:<{width}s} {_fmt(value, p)}"


def _verdict_line(pred_id: str, satisfied: bool | None, detail: str = "") -> str:
    """Render one falsifiable-prediction verdict line."""
    sym = "PASS" if satisfied else ("FAIL" if satisfied is False else "n/a ")
    detail_str = f"  — {detail}" if detail else ""
    return f"  [{sym}] {pred_id}{detail_str}"
