#!/usr/bin/env python3
"""
tools/render_falsification.py — generate claims/FALSIFICATION.md
(POPPER_PLAN.md item B7).

Builds, per claim, the ordered table of adjudicated predictions with p, e, the
running E, and the decision at alpha — replacing the hand-maintained verdict
tables in each phase's `status-N.md` with a generated artifact that cannot
disagree with its own evidence.

Why generated rather than written
---------------------------------
A verdict table maintained by hand can be updated without its evidence, and
that failure is invisible: the table and the artifact it summarises live in
different files, and nothing compares them. `INDEX.md` already records two
instances of the general problem ("Two README headers were wrong"). Generating
the table from `claims/adjudications/` makes the disagreement impossible rather
than detectable.

`--check` fails when the committed file is stale, so CI enforces it.

Standard library plus `core.evalues` / `core.adjudication`; runs in tier 0.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.adjudication import (           # noqa: E402
    all_claim_processes,
    load_adjudications,
    load_registry,
    verify_ledger,
)

OUT = ROOT / "claims" / "FALSIFICATION.md"


def _fmt(x: float, places: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    return f"{x:.{places}g}"


def render() -> str:
    reg = load_registry()
    procs = all_claim_processes()
    records = {r["prediction_id"]: r for r in load_adjudications()}
    by_claim: dict[str, list[dict]] = {}
    for p in reg["predictions"]:
        by_claim.setdefault(p["claim"], []).append(p)

    alpha = float(reg.get("alpha", 0.05))
    lines: list[str] = []
    W = lines.append

    W("# FALSIFICATION.md — the adjudication ledger")
    W("")
    W("**Generated** by `tools/render_falsification.py` from `claims/registry.json`")
    W("and `claims/adjudications/`. Do not edit by hand; `--check` fails CI when this")
    W("file disagrees with the records it summarises.")
    W("")
    W(f"Threshold: a claim is supported when its accumulated evidence reaches ")
    W(f"**E ≥ 1/α = {1.0 / alpha:.0f}** (α = {alpha}, κ = {reg.get('kappa', 0.5)}).")
    W("")
    W("The decision is never \"null accepted\". Failing to accumulate evidence against")
    W("a null is not evidence for it, and an e-process has no way to express one — which")
    W("is the Popperian asymmetry kept visible in the artifact rather than only in the")
    W("prose around it.")
    W("")

    total_adj = len(records)
    W("## Status")
    W("")
    if total_adj == 0:
        W("**No prediction has been adjudicated.** The apparatus is built and the ledger")
        W("is empty — which is the honest state, not an omission. Nothing in this project")
        W("has yet produced a p-value against real artifacts.")
    else:
        W(f"{total_adj} prediction(s) adjudicated across "
          f"{len({r['claim'] for r in records.values()})} claim(s).")
    W("")

    W("## Per-claim evidence")
    W("")
    W("| claim | adjudicated | log E | E | decision |")
    W("|---|---|---|---|---|")
    for claim in sorted(procs):
        proc = procs[claim]
        n = len(proc.adjudications)
        decision = proc.decision() if n else "not adjudicated"
        W(f"| `{claim}` | {n} | {_fmt(proc.log_E)} | {_fmt(proc.E)} | {decision} |")
    W("")

    for claim in sorted(by_claim):
        proc = procs.get(claim)
        rows = by_claim[claim]
        W(f"### {claim}")
        W("")
        n_dormant = sum(1 for p in rows if p.get("status") == "dormant")
        n_eval = sum(1 for p in rows if p["evaluable"] == "e-value"
                     and p.get("status", "active") == "active")
        W(f"{len(rows)} registered · {n_eval} adjudicable now · {n_dormant} dormant")
        W("")

        if proc is not None and proc.adjudications:
            W("| # | prediction | p | e | running E | decision after |")
            W("|---|---|---|---|---|---|")
            running = 0.0
            for i, a in enumerate(proc.adjudications, start=1):
                running += a.log_e_value
                E = math.inf if running > 709 else math.exp(running)
                rec = records.get(a.prediction_id, {})
                W(f"| {i} | `{a.prediction_id}` | {_fmt(a.p_value)} | "
                  f"{_fmt(a.e_value)} | {_fmt(E)} | "
                  f"{rec.get('claim_decision_after', '—')} |")
            W("")
            W(f"Next experiment on this claim must return "
              f"**p < {_fmt(proc.next_p_needed())}** to cross the threshold.")
            W("")
        else:
            W("*No adjudications.*")
            W("")

        blocked = [p for p in rows
                   if p.get("status") == "dormant" or p["evaluable"] != "e-value"]
        if blocked:
            W("<details><summary>Registered but not adjudicable "
              f"({len(blocked)})</summary>")
            W("")
            W("| prediction | why |")
            W("|---|---|")
            for p in sorted(blocked, key=lambda x: x["id"]):
                if p.get("status") == "dormant":
                    why = "dormant — instrument archived"
                elif p["evaluable"] == "measurement":
                    why = "measurement — no valid null exists"
                else:
                    why = "needs-null — null not yet constructed"
                W(f"| `{p['id']}` | {why} |")
            W("")
            W("</details>")
            W("")

    problems = verify_ledger()
    W("## Ledger integrity")
    W("")
    if problems:
        W("**PROBLEMS FOUND** — every E above should be treated as unreliable until")
        W("these are resolved:")
        W("")
        for prob in problems:
            W(f"- {prob}")
    else:
        W("Every claim's E was recomputed from the committed records and agrees with")
        W("what those records state. Each e-value was recalibrated from its p-value")
        W("rather than trusted as stored, so a hand-edited record would surface here")
        W("rather than propagate.")
    W("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed file is stale instead of rewriting it")
    args = ap.parse_args(argv)

    new = render()
    old = OUT.read_text(encoding="utf-8") if OUT.exists() else None

    if args.check:
        if old != new:
            print("ERROR   claims/FALSIFICATION.md is stale; run "
                  "`python tools/render_falsification.py`")
            return 1
        print("FALSIFICATION.md is in step with the ledger")
        return 0

    if old != new:
        OUT.write_text(new, encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}")
    else:
        print("no change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
