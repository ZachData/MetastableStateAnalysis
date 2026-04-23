"""
report_6.py — Assemble the Phase 6 LLM-friendly report.

Reads the per-subexperiment SubResult objects (already written to sub/*.json
and sub/*.summary.txt by run_phase6) and assembles a single flat text file.

Design goals
------------
- ~1000 lines maximum.  No raw arrays, no JSON blobs, no embeddings.
- Every claim is a number or a verdict.  No hand-wavy prose.
- Grouped by track: A → B/D → C → falsification table → key numbers.
- The falsification table maps every prediction ID to its outcome.
- The key-numbers table at the bottom is the flat scalar surface for
  LLM-assisted analysis: one metric per line, fixed-width, sortable.

Functions
---------
assemble_report : main entry point
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from p6_subspace.p6_io import SubResult, _fmt, SEP_THICK, SEP_THIN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BAR  = "=" * 72
_DASH = "-" * 72

# Prediction IDs in canonical order for the falsification table
_PREDICTIONS = [
    ("P6-A2",  "f_rot(h) predicts head type (real vs relational)"),
    ("P6-I1",  "Induction write dirs project onto imaginary channel"),
    ("P6-I2",  "QK antisymmetry elevated for induction vs same-content pairs"),
    ("P6-R1",  "Degeneracy ratio >= 5 at plateau layers"),
    ("P6-R2",  "LDA aligns with S repulsive subspace, not imaginary"),
    ("P6-R3",  "Centroid motion at merges is real-dominated"),
    ("P6-R4",  "S-only projection preserves cluster membership"),
    ("P6-R5",  "Plateau dynamics contracting in S, rotating in A"),
    ("P6-C1",  "Head write subspace aligns with matching channel"),
    ("P6-DD1", "Zeroing imaginary channel reduces induction, preserves clusters"),
    ("P6-DD2", "Zeroing real channel disrupts clusters, preserves induction"),
    ("P6-D5",  "Merge driven by real-subspace centroid convergence"),
]

# Mapping from prediction ID → verdict_contribution keys that decide it
_PRED_KEYS: dict[str, str] = {
    "P6-A2":  "hc_p6_a2_satisfied",
    "P6-I1":  "ind_p6_i1_satisfied",
    "P6-I2":  "qk_p6_i2_satisfied",
    "P6-R1":  "deg_p6_r1_satisfied",
    "P6-R2":  "deg_p6_r2_satisfied",
    "P6-R3":  "vel_p6_r3_satisfied",
    "P6-R4":  "probe_p6_r4_satisfied",
    "P6-R5":  "lc_p6_r5_satisfied",
    "P6-C1":  "ws_p6_c1_satisfied",
    "P6-DD1": "dd_p6_dd1_satisfied",
    "P6-DD2": "dd_p6_dd2_satisfied",
    "P6-D5":  "vel_p6_d5_satisfied",
}


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def assemble_report(
    subresults: dict[str, SubResult],
    ctx:        dict,
    out_path:   Path,
) -> None:
    """
    Assemble phase6_report.txt from all SubResult objects.

    Parameters
    ----------
    subresults : dict mapping spec.name → SubResult
    ctx        : shared context (for model/prompt metadata)
    out_path   : path to write the assembled report
    """
    # Collect all verdict_contribution dicts into one flat dict
    all_vc: dict[str, Any] = {}
    for sr in subresults.values():
        all_vc.update(sr.verdict_contribution)

    lines: list[str] = []

    lines += _header(ctx)
    lines += _projector_section(subresults)
    lines += _track_a_section(subresults)
    lines += _track_bd_section(subresults)
    lines += _track_c_section(subresults)
    lines += _falsification_table(all_vc)
    lines += _key_numbers_table(all_vc)

    # Trim to ≤ 1000 lines, noting truncation
    if len(lines) > 1000:
        lines = lines[:990] + [
            "",
            "[ TRUNCATED: report exceeded 1000 lines. "
            "See sub/*.summary.txt for full per-module text. ]",
        ]

    text = "\n".join(lines) + "\n"
    Path(out_path).write_text(text)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _header(ctx: dict) -> list[str]:
    model  = ctx.get("model_name", "unknown")
    prompt = ctx.get("tokens", [])
    prompt_preview = " ".join(prompt[:8]) + ("..." if len(prompt) > 8 else "")
    d      = ctx.get("projectors", {}).get("d_model", "?")
    n_lay  = len(ctx.get("projectors", {}).get("layer_names", []))

    return [
        _BAR,
        "PHASE 6 REPORT — REAL / IMAGINARY SUBSPACE DECOMPOSITION",
        _BAR,
        f"Model:           {model}",
        f"d_model:         {d}",
        f"Projector layers:{n_lay}",
        f"Prompt preview:  {prompt_preview}",
        "",
        "Hypothesis under test:",
        "  Real subspace (S, 1x1 Schur blocks)  → semantic similarity, cluster structure",
        "  Imaginary subspace (A, 2x2 blocks)   → relational computation, induction",
        "",
    ]


def _projector_section(subresults: dict[str, SubResult]) -> list[str]:
    lines = [_BAR, "FOUNDATION: GLOBAL S/A PROJECTORS", _BAR]
    # The subspace_build result is not a SubResult (it runs before the registry);
    # pull from ctx if available, or note it was a precondition step.
    lines += [
        "Projectors are built from the union of all-head Schur subspaces.",
        "See projectors.json for dim_S, dim_A, overlap, coverage.",
        "",
    ]
    return lines


def _extract_lines(subresults: dict, name: str) -> list[str]:
    """Pull summary_lines from one SubResult, or return a skip note."""
    sr = subresults.get(name)
    if sr is None:
        return [f"  {name}: not registered"]
    if not sr.applicable:
        note = (sr.summary_lines[0] if sr.summary_lines else f"{name}: skipped")
        return [f"  {note}"]
    return sr.summary_lines


def _track_a_section(subresults: dict[str, SubResult]) -> list[str]:
    lines = [_BAR, "TRACK A — IMAGINARY SUBSPACE: RELATIONAL COMPUTATION", _BAR, ""]
    for name in ("head_classify", "qk_decompose", "induction_ov"):
        lines += _extract_lines(subresults, name)
        lines.append("")
    return lines


def _track_bd_section(subresults: dict[str, SubResult]) -> list[str]:
    lines = [_BAR, "TRACK B/D — REAL SUBSPACE: CLUSTER GEOMETRY", _BAR, ""]
    for name in ("eigenspace_degeneracy", "centroid_velocity",
                 "local_contraction", "probe_subspace"):
        lines += _extract_lines(subresults, name)
        lines.append("")
    return lines


def _track_c_section(subresults: dict[str, SubResult]) -> list[str]:
    lines = [_BAR, "TRACK C — CAUSAL TESTS", _BAR, ""]
    for name in ("write_subspace", "dissociation"):
        lines += _extract_lines(subresults, name)
        lines.append("")
    return lines


def _falsification_table(all_vc: dict) -> list[str]:
    lines = [
        _BAR,
        "FALSIFICATION TABLE",
        _BAR,
        f"  {'ID':<8}  {'Result':<6}  Description",
        _DASH,
    ]

    n_pass = n_fail = n_na = 0

    for pred_id, description in _PREDICTIONS:
        key = _PRED_KEYS.get(pred_id)
        val = all_vc.get(key) if key else None

        if val is True:
            sym = "PASS"
            n_pass += 1
        elif val is False:
            sym = "FAIL"
            n_fail += 1
        else:
            sym = "n/a "
            n_na += 1

        lines.append(f"  {pred_id:<8}  [{sym}]  {description}")

    lines += [
        _DASH,
        f"  Pass: {n_pass}   Fail: {n_fail}   N/A: {n_na}   Total: {len(_PREDICTIONS)}",
        "",
        "Interpretation guide:",
        "  PASS  = prediction confirmed by the data",
        "  FAIL  = prediction falsified (result was opposite to prediction)",
        "  n/a   = sub-experiment did not run or produced insufficient data",
        "",
        "Overall hypothesis support:",
    ]

    if n_pass + n_fail > 0:
        frac = n_pass / (n_pass + n_fail)
        lines.append(f"  {n_pass} of {n_pass + n_fail} tested predictions passed ({frac:.1%})")
        if frac >= 0.75:
            lines.append("  → Strong support for real/imaginary functional separation.")
        elif frac >= 0.5:
            lines.append("  → Mixed evidence; some predictions confirmed, others falsified.")
        else:
            lines.append("  → Majority of predictions falsified; hypothesis under pressure.")
    else:
        lines.append("  No predictions were testable with available data.")

    lines.append("")
    return lines


def _key_numbers_table(all_vc: dict) -> list[str]:
    lines = [
        _BAR,
        "KEY NUMBERS (flat reference table for LLM analysis)",
        _BAR,
        "  All metrics from verdict_contribution dicts across sub-experiments.",
        "  Sorted alphabetically.  Format: key  value",
        _DASH,
    ]

    for k in sorted(all_vc.keys()):
        v = all_vc[k]
        if v is None:
            lines.append(f"  {k:<52s} n/a")
        elif isinstance(v, bool):
            lines.append(f"  {k:<52s} {str(v).lower()}")
        elif isinstance(v, float):
            lines.append(f"  {k:<52s} {v:.4f}")
        elif isinstance(v, int):
            lines.append(f"  {k:<52s} {v}")
        else:
            lines.append(f"  {k:<52s} {v}")

    lines.append("")
    return lines
