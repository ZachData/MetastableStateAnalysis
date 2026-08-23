#!/usr/bin/env python3
"""
tools/render_evaluability.py — regenerate the table in claims/EVALUABILITY.md
from claims/registry.json (POPPER_PLAN.md item B5).

The prose in EVALUABILITY.md is written by hand and preserved verbatim; only
the block between the `## The table` heading and the following `##` heading is
replaced. That split is deliberate. The counts and the per-prediction rows must
never drift from the registry — they are the thing CI checks against — but the
reasoning about *why* seven of thirty is the number, and what the three
recurring patterns are, is analysis and does not belong in a generator.

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
