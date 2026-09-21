#!/usr/bin/env python3
"""
tools/render_experiments.py — generate claims/EXPERIMENTS.md, the phase-to-e-value map.

What it is for
--------------
Five files already hold pieces of this and none of them holds the join.
`claims/registry.json` knows the prediction, its claim and its null;
`claims/EVALUABILITY.md` knows whether the null is valid; `claims/FALSIFICATION.md`
knows what has been adjudicated; `POPPER_PLAN.md` §6a–6w knows what each
construction cost to build; each phase's `status-N.md` knows what the phase
measured. What nobody could answer without reading all five is the question a
reader actually arrives with:

    which phase, which experiment, carries an e-value — and what is it worth?

This file answers it in one table, generated so it cannot drift from the
registry it summarises.

What it checks while rendering
------------------------------
Three joins that nothing else in the project checks, each reported in its own
section rather than buried:

* **A phase with a live instrument and no registered prediction.** The apparatus
  cannot refuse what was never registered, so a phase outside it is invisible to
  every other tool here — including `check_registry.py`, whose coverage scan runs
  the other way (ids in the tree must be in the registry, not phases in the tree
  must have ids).
* **A declared claim with no prediction.** Its e-process can never move. That is
  a real state and not necessarily a fault, but it should be stated rather than
  inferred from an empty row in `FALSIFICATION.md`.
* **An adjudicable gate with no known-answer dry run.** `POPPER_PLAN.md` §6p
  records the base rate on the nine rows that had one: *"Nine for nine, every one
  of them changed something."* A gate that skipped that step is the likeliest
  place for the tenth.

`--check` fails when the committed file is stale, so CI enforces it.

Standard library only, plus `core.adjudication` / `core.evalues`, both of which
are stdlib-only at import time. This runs in tier 0 where nothing is installed,
so gate resolution is TEXTUAL — `def <name>(` in the module's source — rather
than an import. A gate module that imports numpy could not be imported here at
all, and a check that cannot run in the tier that gates a merge is not a check.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.adjudication import (           # noqa: E402
    all_claim_processes,
    load_adjudications,
    load_registry,
)

OUT = ROOT / "claims" / "EXPERIMENTS.md"
AUDITS = ROOT / "claims" / "audits"
CALIBRATION = ROOT / "claims" / "calibration"

#: Phase directories that hold a phase but register nothing, with the reason on
#: record. An entry here is an ACKNOWLEDGED absence; a phase absent from both the
#: registry and this dict is reported as unexplained. Keeping the two apart is
#: the point: "no prediction because the phase is exploratory" and "no prediction
#: because nobody registered one" read identically in a bare list.
PHASES_WITHOUT_PREDICTIONS = {
    "1b": "hemisphere geometry; its findings feed P-H1 and CLAIM-A rather than "
          "carrying a falsifier of their own.",
    "1d": "not on main — `origin/claude/particle-methods-comparison-vpuads`.",
    "2": "eigenspectra; the 19-step Pythia sweep is a measurement programme and "
         "supplies the artifacts other phases adjudicate on.",
    "2b": "imaginary/rotational decomposition; measurement, feeding H-OPERATOR's "
          "P-M1 through `rotational_schur.py`.",
    "3": "archived 2026-08-22, null result. `archive/p3_crosscoder/FROZEN.md`.",
    "4": "archived 2026-08-22. `archive/p4_mstate_features/FROZEN.md`.",
    "5": "archived 2026-08-22; six code-level blockers, no falsifier registered.",
    "5c": "docs only, no code. Cited by `PREDICTIONS.md` claim (a).",
    "7d": "UNEXPLAINED — active phase, see the section below.",
    "7e": "UNEXPLAINED — active phase, see the section below.",
    "8": "UNEXPLAINED — active phase, see the section below.",
    "10": "pre-design and deliberately unregistered. `notes-10.md` §13 states "
          "it: no construction frozen, no `P-*` id, `claims/registry.json` "
          "untouched. Its free rows (F0, F1, F11-A0, F12) have RUN, and their "
          "records under `data/analysis/` are tier 1, exploratory, and not "
          "quotable as adjudications.",
}

#: Which `claims/audits/*.json` is a known-answer dry run, as opposed to a
#: prerequisite audit of something else. Read from the file's own prose, so a new
#: audit joins this set by saying what it is rather than by being listed.
_DRY_RUN_MARKERS = ("answer is known", "verdict is fixed", "known a priori")

#: Matches a repo-relative Python path written inside an evidence artifact.
_PY_PATH = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./]*\.py")


def _fmt(x, places: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    if isinstance(x, float):
        return f"{x:.{places}g}"
    return str(x)


def _first_sentence(text: str, limit: int = 150) -> str:
    """First sentence of a `null_construction`, for a table cell."""
    text = " ".join(str(text).split())
    if not text:
        return "—"
    m = re.search(r"(?<=[a-z0-9\)\]])\.\s+[A-Z]", text)
    if m:
        text = text[: m.start() + 1]
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text.replace("|", "\\|")


def resolve_gate(gate: str) -> tuple[bool, str]:
    """
    ``(ok, detail)`` for a ``"module.path:function"`` gate string.

    Textual, not by import — see the module docstring. A false negative here
    (a function defined by something other than a `def` line) is preferable to
    a check that silently does not run in the tier that gates a merge.
    """
    if not gate:
        return True, ""
    if ":" not in gate:
        return False, "not of the form module.path:function"
    mod, func = gate.split(":", 1)
    path = ROOT / (mod.replace(".", "/") + ".py")
    if not path.exists():
        return False, f"{path.relative_to(ROOT)} does not exist"
    src = path.read_text(encoding="utf-8", errors="replace")
    if not re.search(rf"^def {re.escape(func)}\(", src, re.M):
        return False, f"{path.relative_to(ROOT)} defines no `{func}`"
    return True, str(path.relative_to(ROOT))


def _modules_named(text: str, depth: int = 1) -> set[str]:
    """
    Every repo module an evidence artifact reaches, to one hop through its tool.

    Linking evidence to a prediction by whether the artifact MENTIONS the id is
    wrong, and wrong in the flattering direction: `p_t1_p_m1_dry_run.json` cites
    `CLAIM-B`, `CLAIM-C`, `P6-R2`, `P6-R4`, `P-S1` and `P-ST1` in its prose —
    they are cross-references to what earlier passes found, not coverage — and a
    substring scan credits every one of them with a dry run they did not have.

    What an artifact does state exactly is which FILE it was measured against:
    every dry run records its gate's path and SHA-256. So the link is the gate
    module, found either directly or through the `tools/…` script the artifact
    names as its generator — one hop, because a calibration typically records
    the tool rather than the gate the tool imports.
    """
    seen: set[str] = set()
    frontier = {m for m in _PY_PATH.findall(text) if (ROOT / m).exists()}
    for _ in range(depth + 1):
        new = frontier - seen
        if not new:
            break
        seen |= new
        frontier = set()
        for m in new:
            if not m.startswith("tools/"):
                continue
            src = (ROOT / m).read_text(encoding="utf-8", errors="replace")
            frontier |= {q for q in _PY_PATH.findall(src) if (ROOT / q).exists()}
            # Tools import their gate rather than naming its path.
            for mod in re.findall(r"^\s*from ([\w.]+) import", src, re.M):
                cand = mod.replace(".", "/") + ".py"
                if (ROOT / cand).exists():
                    frontier.add(cand)
    return seen


def _evidence_index() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """
    ``(dry_runs, calibrations)``, each mapping prediction id -> filenames.

    A prediction is covered by an artifact when the artifact reaches that
    prediction's GATE MODULE — see `_modules_named` for why not by id.
    """
    reg = load_registry()
    gate_of = {p["id"]: p.get("gate", "").split(":")[0].replace(".", "/") + ".py"
               for p in reg.get("predictions", []) if p.get("gate")}
    dry: dict[str, set[str]] = {}
    calib: dict[str, set[str]] = {}

    for d, sink in ((AUDITS, dry), (CALIBRATION, calib)):
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            text = f.read_text(encoding="utf-8", errors="replace")
            if d is AUDITS and not any(m in text[:4000] for m in _DRY_RUN_MARKERS):
                continue                      # a prerequisite audit, not a dry run
            mods = _modules_named(text)
            for pid, gate_mod in gate_of.items():
                if gate_mod in mods:
                    sink.setdefault(pid, set()).add(f.name)
    return dry, calib


def _phases_on_disk() -> dict[str, Path]:
    """Phase id -> directory, for every directory that carries a phase's notes."""
    found: dict[str, Path] = {}
    for base in (ROOT, ROOT / "archive"):
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or not re.fullmatch(r"p[0-9]+[a-z]?_.*", d.name):
                continue
            m = re.match(r"p([0-9]+[a-z]?)_", d.name)
            if m and (list(d.glob("status-*.md")) or list(d.glob("design-*.md"))
                      or list(d.glob("math-*.md"))):
                found.setdefault(m.group(1), d)
    return found


def render() -> str:
    reg = load_registry()
    preds = reg.get("predictions", [])
    procs = all_claim_processes()
    records = {r["prediction_id"]: r for r in load_adjudications()}
    dry_runs, calibs = _evidence_index()

    alpha = float(reg.get("alpha", 0.05))
    kappa = float(reg.get("kappa", 0.5))
    r0 = float(reg.get("relevance_threshold", 0.6))

    by_phase: dict[str, list[dict]] = {}
    for p in preds:
        by_phase.setdefault(p.get("phase", "?"), []).append(p)

    def adjudicable(p: dict) -> bool:
        return (p.get("evaluable") == "e-value"
                and p.get("status", "active") == "active"
                and float(p.get("relevance", 0)) >= r0)

    L: list[str] = []
    W = L.append

    W("# EXPERIMENTS.md — which phase carries which e-value")
    W("")
    W("**Generated** by `tools/render_experiments.py` from `claims/registry.json`,")
    W("`claims/audits/`, `claims/calibration/` and `claims/adjudications/`. Do not edit")
    W("by hand; `--check` fails CI when this file disagrees with what it summarises.")
    W("")
    W("`INDEX.md` maps phases to directories and `claims/FALSIFICATION.md` maps claims to")
    W("evidence. This file is the join between them: phase → experiment → prediction →")
    W("gate → what that gate is worth. It is the file to read when the question is")
    W("*\"does this phase carry a falsifier, and has it been run?\"*")
    W("")
    W(f"α = {alpha} · κ = {kappa} · relevance floor r0 = {r0} · a claim is supported at")
    W(f"**E ≥ 1/α = {1.0 / alpha:.0f}**.")
    W("")

    # -- the headline counts ------------------------------------------------
    n_adjudicable = sum(1 for p in preds if adjudicable(p))
    n_adjudicated = len(records)
    n_dormant = sum(1 for p in preds if p.get("status") == "dormant")
    W("## Where the evidence stands")
    W("")
    W(f"- **{len(preds)}** registered predictions across **{len(by_phase)}** phases and "
      f"**{len({p['claim'] for p in preds})}** claims.")
    W(f"- **{n_adjudicable}** can carry an e-value right now (`e-value`, active, "
      f"relevance ≥ r0).")
    W(f"- **{n_dormant}** are dormant — pre-registered, falsifier intact, instrument "
      f"archived.")
    if n_adjudicated == 0:
        W("- **0 have been adjudicated.** Every E below is 1 and every decision is "
          "\"not adjudicated\". The apparatus is built; no p-value against a real "
          "artifact exists yet.")
    else:
        W(f"- **{n_adjudicated}** have been adjudicated; see `claims/FALSIFICATION.md` "
          f"for the running E.")
    W("")

    # -- per-phase rollup ---------------------------------------------------
    W("## Per phase")
    W("")
    W("| phase | registered | e-value | adjudicable now | adjudicated | claims touched |")
    W("|---|---|---|---|---|---|")
    for phase in sorted(by_phase, key=lambda s: (len(s), s)):
        rows = by_phase[phase]
        W(f"| `{phase}` | {len(rows)} | "
          f"{sum(1 for p in rows if p.get('evaluable') == 'e-value')} | "
          f"{sum(1 for p in rows if adjudicable(p))} | "
          f"{sum(1 for p in rows if p['id'] in records)} | "
          f"{', '.join('`' + c + '`' for c in sorted({p['claim'] for p in rows}))} |")
    W("")

    # -- the map itself -----------------------------------------------------
    W("## The map")
    W("")
    W("One section per phase, one row per registered prediction. `gate` is the")
    W("`module:function` that computes the p-value; an empty gate means no null is")
    W("built, which is what `needs-null` and `measurement` mean in code. The null")
    W("column is a pointer, not the construction — `claims/registry.json`'s")
    W("`null_construction` holds the whole of it, and `POPPER_PLAN.md` §6a–6zb holds")
    W("what each one cost to build and what it was wrong about first. `calibrated`")
    W("and `real run` are the registry's `calibration_record` and `real_run_record`:")
    W("each a git-tracked path or nothing, checked by `tools/check_registry.py`, so")
    W("an empty cell means no such evidence exists in the tree. A row can carry a")
    W("gate, a calibration and a real run and still read `needs-null` — that is a")
    W("null that was tried and found invalid, kept visible rather than dropped.")
    W("")

    for phase in sorted(by_phase, key=lambda s: (len(s), s)):
        rows = sorted(by_phase[phase], key=lambda p: (p.get("experiment", ""), p["id"]))
        on_disk = _phases_on_disk().get(phase)
        where = f"`{on_disk.relative_to(ROOT)}/`" if on_disk else "*(no directory)*"
        W(f"### Phase {phase} — {where}")
        W("")
        W("| experiment | prediction | claim | evaluable | status | gate | calibrated | real run | null (first line) |")
        W("|---|---|---|---|---|---|---|---|---|")
        for p in rows:
            gate = p.get("gate", "")
            ok, _detail = resolve_gate(gate)
            gate_cell = f"`{gate}`" if gate else "—"
            if gate and not ok:
                gate_cell += " **UNRESOLVED**"
            cal = p.get("calibration_record")
            run = p.get("real_run_record")
            W(f"| {p.get('experiment') or '—'} | `{p['id']}` | `{p['claim']}` | "
              f"{p.get('evaluable')} | {p.get('status', 'active')} | {gate_cell} | "
              f"{'`' + cal + '`' if cal else '—'} | {'`' + run + '`' if run else '—'} | "
              f"{_first_sentence(p.get('null_construction', ''))} |")
        W("")

    # -- the adjudicable rows, with their evidence --------------------------
    W("## The adjudicable rows, and what stands behind each")
    W("")
    W("Only these can move a claim's E. `dry run` is a run on inputs whose correct")
    W("verdict is fixed a priori (`claims/audits/`); `calibration` is the measured")
    W("behaviour of the construction (`claims/calibration/`).")
    W("")
    W("| prediction | phase · experiment | claim | dry run | calibration | p | e | decision |")
    W("|---|---|---|---|---|---|---|---|")
    for p in preds:
        if not adjudicable(p):
            continue
        pid = p["id"]
        rec = records.get(pid)
        dr = ", ".join(sorted(dry_runs.get(pid, set()))) or "**none**"
        cal = ", ".join(sorted(calibs.get(pid, set()))) or "—"
        W(f"| `{pid}` | `{p.get('phase')}` · {p.get('experiment') or '—'} | "
          f"`{p['claim']}` | {dr} | {cal} | "
          f"{_fmt(rec['p_value']) if rec else '—'} | "
          f"{_fmt(rec['e_value']) if rec else '—'} | "
          f"{rec['claim_decision_after'] if rec else 'not adjudicated'} |")
    W("")

    # -- claim rollup -------------------------------------------------------
    W("## Per claim")
    W("")
    W("| claim | phases feeding it | registered | adjudicable now | E | decision |")
    W("|---|---|---|---|---|---|")
    declared = _declared_claims()
    for claim in sorted(set(declared) | {p["claim"] for p in preds}):
        rows = [p for p in preds if p["claim"] == claim]
        proc = procs.get(claim)
        phases = sorted({p.get("phase", "?") for p in rows}, key=lambda s: (len(s), s))
        E = _fmt(proc.E) if proc else "1"
        dec = (proc.decision() if proc and proc.adjudications else "not adjudicated")
        W(f"| `{claim}` | {', '.join('`' + x + '`' for x in phases) or '**none**'} | "
          f"{len(rows)} | {sum(1 for p in rows if adjudicable(p))} | {E} | {dec} |")
    W("")

    # -- the three joins nothing else checks --------------------------------
    W("## Gaps")
    W("")
    W("Three joins that no other tool here checks, each stated rather than left to be")
    W("noticed. None of them is automatically a fault; all of them are things a reader")
    W("would otherwise have to reconstruct from five files.")
    W("")

    registered_phases = set(by_phase)
    on_disk = _phases_on_disk()
    unregistered = sorted(set(on_disk) - registered_phases, key=lambda s: (len(s), s))
    W("### Phases on disk with no registered prediction")
    W("")
    if not unregistered:
        W("None.")
    else:
        W("| phase | directory | why |")
        W("|---|---|---|")
        for ph in unregistered:
            why = PHASES_WITHOUT_PREDICTIONS.get(
                ph, "**UNEXPLAINED** — not in `PHASES_WITHOUT_PREDICTIONS`")
            W(f"| `{ph}` | `{on_disk[ph].relative_to(ROOT)}/` | {why} |")
        W("")
        W("A phase outside the registry is outside the apparatus: `core/adjudication.py`")
        W("cannot refuse what was never registered, and a headline from such a phase")
        W("carries no Type-I guarantee whatever its control distribution looks like.")
    W("")

    empty_claims = sorted(set(declared) - {p["claim"] for p in preds})
    W("### Declared claims with no prediction")
    W("")
    if not empty_claims:
        W("None.")
    else:
        for c in empty_claims:
            W(f"- **`{c}`** is a heading in `claims/CLAIMS.md` and no registered")
            W("  prediction names it, so its e-process has no factor that could ever")
            W("  enter and its E is 1 by construction rather than by result.")
    W("")

    missing_dry = [p["id"] for p in preds
                   if adjudicable(p) and not dry_runs.get(p["id"])]
    W("### Adjudicable gates with no known-answer dry run")
    W("")
    if not missing_dry:
        W("None.")
    else:
        for pid in missing_dry:
            p = next(x for x in preds if x["id"] == pid)
            cal = ", ".join(sorted(calibs.get(pid, set()))) or "no calibration either"
            W(f"- **`{pid}`** (`{p.get('phase')}` · {p.get('experiment') or '—'}, "
              f"gate `{p.get('gate')}`) — {cal}.")
        W("")
        W("`POPPER_PLAN.md` §6p records the base rate on the nine rows that had one:")
        W("*\"Nine for nine, every one of them changed something. … not one converted row")
        W("survived being run on an input whose answer was already known, and no test was")
        W("failing on any of them.\"* A calibration measures the construction's behaviour")
        W("on a synthetic family; a dry run asks the different question of whether the")
        W("gate returns the verdict that is correct a priori. The rows above have the")
        W("first and not the second.")
    W("")

    bad_gates = [(p["id"], p.get("gate", ""), resolve_gate(p.get("gate", ""))[1])
                 for p in preds if not resolve_gate(p.get("gate", ""))[0]]
    W("### Gates that do not resolve")
    W("")
    if not bad_gates:
        W("None — every `gate` names a module that exists and a function it defines.")
    else:
        for pid, gate, why in bad_gates:
            W(f"- `{pid}`: `{gate}` — {why}")
    W("")

    return "\n".join(L) + "\n"


def _declared_claims() -> list[str]:
    """Claim names from CLAIMS.md's `### H-NAME — ...` headings."""
    path = ROOT / "claims" / "CLAIMS.md"
    if not path.exists():
        return []
    return re.findall(r"^### (H-[A-Z0-9-]+)", path.read_text(encoding="utf-8"), re.M)


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
            print("ERROR   claims/EXPERIMENTS.md is stale; run "
                  "`python tools/render_experiments.py`")
            return 1
        print("EXPERIMENTS.md is in step with the registry")
        return 0

    if old != new:
        OUT.write_text(new, encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}")
    else:
        print("no change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
