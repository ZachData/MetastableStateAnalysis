#!/usr/bin/env python3
"""
PreToolUse hook on Read: refuse an unbounded read of a large text document.

A `Read` of a text file over MAX_BYTES must pass `limit` <= MAX_LINES. The
refusal names the section index in `docs/index/`, so the retry is one bounded
read instead of a page-through. Code (`.py`) is not guarded: reading a module
whole is ordinary work. `cat`/`sed` through Bash are not guarded either; the
rule in `CLAUDE.md` covers those.

Why a hook and not prose: "grep PROJECT.md, never read it whole" lived in
prose (`LESSONS.md` lessons 1 and 9), and the Claude Code docs say to use a
PreToolUse hook for anything that must hold regardless of what Claude decides
(`docs/agent_context_scan_2026-09-22.md`).

Registered in `.claude/settings.json` under hooks.PreToolUse, matcher "Read".
Fails open: bad input, or a file it cannot stat, allows the read.
Tunable per shell: METS_READ_MAX_BYTES, METS_READ_MAX_LINES. Stdlib only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MAX_BYTES = int(os.environ.get("METS_READ_MAX_BYTES", 40_000))
MAX_LINES = int(os.environ.get("METS_READ_MAX_LINES", 400))
GUARDED = {".md", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".log", ".tex"}


def decide(tool_input: dict, project_dir: str | None) -> str | None:
    """The refusal message, or None to allow."""
    path = Path(tool_input.get("file_path") or "")
    if path.suffix.lower() not in GUARDED:
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size <= MAX_BYTES:
        return None
    limit = tool_input.get("limit")
    if isinstance(limit, int) and 0 < limit <= MAX_LINES:
        return None
    hint = ""
    if project_dir:
        idx = Path(project_dir) / "docs" / "index" / f"{path.stem}.idx.md"
        if idx.is_file():
            hint = f" Its section index is {idx} (start line, length, ~tokens per section)."
    return (
        f"{path.name} is {size // 1000} KB (~{size // 4000}k tokens). Read it in "
        f"sections: pass offset and limit <= {MAX_LINES}.{hint} Or Grep for the term "
        f"first. (Guard: scripts/hooks/guard_large_read.py)"
    )


def main() -> int:
    try:
        event = json.load(sys.stdin)
        if event.get("tool_name") != "Read":
            return 0
        reason = decide(event.get("tool_input") or {}, os.environ.get("CLAUDE_PROJECT_DIR"))
    except Exception:
        return 0
    if reason:
        json.dump({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
