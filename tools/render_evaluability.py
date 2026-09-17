#!/usr/bin/env python3
"""
tools/render_evaluability.py — regenerate the table in claims/EVALUABILITY.md
from claims/registry.json (POPPER_PLAN.md item B5).

The prose in EVALUABILITY.md is written by hand and preserved verbatim; only
the blocks under the `## The count`, `## By phase` and `## The table` headings
(each up to the following `##` heading) are replaced. That split is
deliberate. The counts and the per-prediction rows must never drift from the
registry — they are the thing CI checks against — but the reasoning about *why*
the counts are what they are, and what the recurring patterns are, is analysis
and does not belong in a generator.

Run with `--check` in CI to fail when the committed table is stale rather than
silently rewriting it.

Standard library only; runs in CI tier 0.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "claims" / "registry.json"
DOC = ROOT / "claims" / "EVALUABILITY.md"

TABLE_HEADER = (
    "| id | claim | state | relevance | null construction, or why none exists |\n"
    "|---|---|---|---|---|\n"
)
COUNT_HEADER = (
    "| state | n | may contribute to a claim's E |\n"
    "|---|---|---|\n"
)
CONTRIBUTES = {
    "e-value": "yes",
    "needs-null": "not yet",
    "measurement": "never",
}


def _escape(text: str) -> str:
    """Keep a cell from breaking the table."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def render_table(reg: dict) -> str:
    rows = []
    for p in reg.get("predictions", []):
        rows.append(
            f"| `{p['id']}` | {p['claim']} | **{p['evaluable']}** | "
            f"{p['relevance']} | {_escape(p['null_construction'])} |"
        )
    return TABLE_HEADER + "\n".join(rows) + "\n"


PHASE_HEADER = (
    "| id | claim | state | status | null built | calibrated | run on real artifacts | adjudicated |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def _phase_key(phase: str) -> tuple:
    m = re.match(r"(\d+)([a-z]*)", phase)
    return (int(m.group(1)), m.group(2)) if m else (999, phase)


def _evidence(path) -> str:
    return f"`{path}`" if path else "—"


def render_by_phase(reg: dict, adjudicated: set[str]) -> str:
    by_phase: dict[str, list] = {}
    for p in reg.get("predictions", []):
        by_phase.setdefault(str(p.get("phase", "?")), []).append(p)

    out = [
        "Read across a row for how far a prediction has got: a null that is "
        "*built* is a live module emitting the p-value, *calibrated* is a "
        "known-answer or calibration artifact that checked it, and *run on real "
        "artifacts* is a committed record of a p-value against real "
        "checkpoints or activations. Each cell is a git-tracked path, checked "
        "by `tools/check_registry.py`, so an empty cell means no evidence "
        "exists in the tree rather than that nobody wrote it down. A row can be "
        "built, calibrated and run and still `needs-null` — that is a null "
        "that was tried and found invalid, which the path record keeps "
        "visible. Phases with no rows (1b, 2b, 7d, 7e, 8) registered no "
        "predictions: exploratory by design, and nothing there may carry an "
        "e-value.\n"
    ]
    for phase in sorted(by_phase, key=_phase_key):
        rows = by_phase[phase]
        n_dormant = sum(1 for p in rows if p.get("status") == "dormant")
        dormant_note = f" — {n_dormant} dormant, instrument archived" if n_dormant else ""
        out.append(f"### Phase {phase} ({len(rows)} registered{dormant_note})\n")
        out.append(PHASE_HEADER.rstrip("\n"))
        for p in rows:
            out.append(
                f"| `{p['id']}` | {p['claim']} | **{p['evaluable']}** | {p.get('status', 'active')} | "
                f"{_evidence(p.get('null_module'))} | {_evidence(p.get('calibration_record'))} | "
                f"{_evidence(p.get('real_run_record'))} | "
                f"{'yes' if p['id'] in adjudicated else '—'} |"
            )
        out.append("")
    return "\n".join(out) + "\n"


def load_adjudicated() -> set[str]:
    d = ROOT / "claims" / "adjudications"
    return {f.stem for f in d.glob("*.json")} if d.is_dir() else set()


def render_counts(reg: dict) -> str:
    c = Counter(p["evaluable"] for p in reg.get("predictions", []))
    rows = [
        f"| `{state}` | {c.get(state, 0)} | {CONTRIBUTES[state]} |"
        for state in ("e-value", "needs-null", "measurement")
    ]
    return COUNT_HEADER + "\n".join(rows) + "\n"


def _replace_section(doc: str, heading: str, body: str) -> str:
    """Replace everything between `heading` and the next `## ` heading."""
    # The trailing "\n\n" is part of the replacement rather than of the match,
    # so re-running is a fixed point: without that the section grows a blank
    # line per invocation and `--check` reports a stale file it just wrote.
    pattern = re.compile(
        rf"(^{re.escape(heading)}[^\n]*\n)(.*?)(?=^## |\Z)", re.S | re.M
    )
    if not pattern.search(doc):
        raise SystemExit(f"{DOC.name}: no {heading!r} section to replace")
    return pattern.sub(lambda m: m.group(1) + "\n" + body.rstrip("\n") + "\n\n", doc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed table is stale instead of rewriting it")
    args = ap.parse_args(argv)

    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    doc = DOC.read_text(encoding="utf-8")

    updated = _replace_section(doc, "## The table", render_table(reg))
    updated = _replace_section(updated, "## The count", render_counts(reg))
    updated = _replace_section(updated, "## By phase",
                               render_by_phase(reg, load_adjudicated()))

    if args.check:
        if updated != doc:
            print("ERROR   claims/EVALUABILITY.md is stale relative to claims/registry.json; "
                  "run `python tools/render_evaluability.py`")
            return 1
        print("EVALUABILITY.md is in step with the registry")
        return 0

    if updated != doc:
        DOC.write_text(updated, encoding="utf-8")
        print(f"rewrote {DOC.relative_to(ROOT)}")
    else:
        print("no change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
