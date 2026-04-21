"""
subresult.py — Uniform contract for every p2_eigenspectra sub-experiment.

Every sub-experiment returns a SubResult.  The orchestrator in
subexperiments.py iterates the SUBEXPERIMENTS registry, writes each
SubResult to disk (one .json + one .summary.txt), and feeds completed
results forward through ctx so later sub-experiments can depend on them.

Summary-line contract
---------------------
summary_lines must be self-contained plain-text prose: no ANSI codes,
no trailing spaces, lines under ~100 chars.  They are concatenated
verbatim into summary.txt which is the LLM-consumable per-run report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class SubResult:
    """
    Uniform output of every p2_eigenspectra sub-experiment.

    Attributes
    ----------
    name               : sub-experiment identifier (matches registry entry)
    applicable         : False when skipped (shared weights, missing prereqs,
                         opt-in flag not set, etc.)
    payload            : JSON-serialisable raw result dict; written to
                         sub/{name}.json
    summary_lines      : LLM-ready plain-text lines; written to
                         sub/{name}.summary.txt and assembled into summary.txt
    verdict_contribution : flat scalar dict; keys are merged into the
                           verdict by build_verdict_v2_from_subresults.
                           Keys must be globally unique across sub-experiments.
    error              : non-None when the sub-experiment raised; the
                         orchestrator catches, stores traceback here, and
                         continues rather than aborting the whole run.
    """
    name:                 str
    applicable:           bool
    payload:              dict
    summary_lines:        list[str]
    verdict_contribution: dict
    error:                str | None = None


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

@dataclass
class SubexperimentSpec:
    """
    Declarative description of one sub-experiment.

    Attributes
    ----------
    name       : unique string identifier; used as dict key in ctx and as
                 filename stem for .json / .summary.txt
    run        : callable (ctx: dict) -> SubResult; receives the shared
                 context dict and returns a SubResult
    requires   : ctx keys that must be non-None for the sub-experiment to run.
                 Missing keys produce a skipped SubResult rather than an error.
    applicable : optional callable (ctx: dict) -> bool; if provided and
                 returns False the sub-experiment is skipped (e.g. shared
                 weights guard for per-layer-only analyses)
    """
    name:       str
    run:        Callable[[dict], SubResult]
    requires:   list[str]               = field(default_factory=list)
    applicable: Callable[[dict], bool]  | None = None

    def prerequisites_met(self, ctx: dict) -> tuple[bool, str]:
        """
        Check that all required ctx keys are present and the optional
        applicable gate passes.

        Returns
        -------
        (ok: bool, reason: str)
            ok=True when the sub-experiment should run.
            reason is a human-readable explanation when ok=False.
        """
        missing = [k for k in self.requires if ctx.get(k) is None]
        if missing:
            return False, f"missing prerequisites: {missing}"
        if self.applicable is not None and not self.applicable(ctx):
            return False, "not applicable for this configuration"
        return True, ""
