#!/usr/bin/env python3
"""
tools/check_registry.py — registry validation and the pre-registration gate
(POPPER_PLAN.md items B2 and B3).

Two checks, in one file because they share a loader and both run in CI tier 0:

**Registry validation.** Schema, uniqueness, claim membership, relevance floor,
and the coverage check that matters most in practice: every prediction ID that
appears anywhere in the project's `.py` or `.md` files has a registry entry.
A prediction discussed in a docstring but absent from the registry is a
prediction with no recorded null, no evaluability classification, and no
pre-registration timestamp -- which is exactly the state the registry exists to
make impossible to stay in.

**The pre-registration gate.** This is the check that carries the statistical
guarantee, and it is the reason this work belongs in CI rather than in a
convention. POPPER's Assumption 2 (sequential validity) requires that the
choice of sub-hypothesis and test function not depend on the data used to test
it. `PREDICTIONS.md` asserts this by convention -- "Written and committed before
the replication gate runs, so the timestamp on this file precedes any result
it's checked against." Git can check that assertion. Three rules:

  1. A prediction's registration commit must strictly precede its adjudication.
  2. A registry entry's `statement`, `h0`, `h1`, `falsifier` and
     `null_construction` may not be modified after its first adjudication.
     Amendments go in `notes` as dated addenda -- the mechanism the P-T1
     amendment already used correctly by hand.
  3. An adjudication may not consume an artifact that a *later-registered*
     prediction also consumes. Re-using an artifact is fine; registering a
     prediction after seeing that artifact and then testing it on the same
     artifact is the conditional-validity violation, and no other check sees it.

Backfill caveat, stated wherever this reports
---------------------------------------------
Predictions registered before this machinery existed carry
`registered_provenance: "backfilled"` -- their commit is recovered from git
history rather than observed by the gate. That is good evidence, not the same
thing as having been gated. Backfilled entries are counted and reported
separately, permanently, rather than merged into the same total as
prospectively gated ones.

Standard library only: this runs in CI tier 0 with no pip install step.

Usage
-----
    python tools/check_registry.py              # both checks
    python tools/check_registry.py --summary    # per-claim status table
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "claims" / "registry.json"
CLAIMS_MD = ROOT / "claims" / "CLAIMS.md"
ADJUDICATIONS = ROOT / "claims" / "adjudications"

REQUIRED_FIELDS = (
    "id", "claim", "statement", "h0", "h1", "falsifier", "instrument",
    "cost", "evaluable", "null_construction", "relevance", "source",
)
EVALUABLE_VALUES = ("e-value", "measurement", "needs-null")

#: Fields frozen once a prediction has been adjudicated (gate rule 2).
FROZEN_FIELDS = ("statement", "h0", "h1", "falsifier", "null_construction")

#: Prediction-ID shapes used in this project. `P1-P5` deliberately does NOT
#: match: core/run_policy.py uses it for "policies P1 through P5", not for a
#: prediction, and a naive pattern picks it up as one.
ID_PATTERN = re.compile(r"\b(?:P-(?:γ1|γ2|gamma1|gamma2|H1|S1|T1|M1)|P[56]b?-[A-Z]+[0-9]+|CLAIM-[ABC])\b")

#: Files that discuss predictions in prose without registering them. Scanning
#: these for IDs would report the planning documents as unregistered sources.
SCAN_EXCLUDE = {"POPPER_PLAN.md", "CLAIMS.md", "EVALUABILITY.md", "FALSIFICATION.md"}


class Problem(Exception):
    pass


def _fail(msgs: List[str], msg: str) -> None:
    msgs.append(f"ERROR   {msg}")


def _warn(msgs: List[str], msg: str) -> None:
    msgs.append(f"WARNING {msg}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    if not REGISTRY.exists():
        raise Problem(f"{REGISTRY.relative_to(ROOT)} does not exist")
    try:
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Problem(f"{REGISTRY.relative_to(ROOT)} is not valid JSON: {exc}")


def declared_claims() -> List[str]:
    """Claim names from CLAIMS.md's `### H-NAME — ...` headings."""
    if not CLAIMS_MD.exists():
        raise Problem(f"{CLAIMS_MD.relative_to(ROOT)} does not exist")
    return re.findall(r"^###\s+(H-[A-Z]+)\b", CLAIMS_MD.read_text(encoding="utf-8"), re.M)


def load_adjudications() -> Dict[str, dict]:
    if not ADJUDICATIONS.is_dir():
        return {}
    out = {}
    for f in sorted(ADJUDICATIONS.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise Problem(f"adjudication {f.name} is not valid JSON: {exc}")
    return out


# ---------------------------------------------------------------------------
# Check 1 — registry validation
# ---------------------------------------------------------------------------

def check_registry(reg: dict, msgs: List[str]) -> None:
    preds = reg.get("predictions", [])
    if not preds:
        _fail(msgs, "registry declares no predictions")
        return

    claims = set(declared_claims())
    seen: set[str] = set()
    r0 = float(reg.get("relevance_threshold", 0.6))

    for i, p in enumerate(preds):
        where = p.get("id", f"<entry {i}>")
        for field in REQUIRED_FIELDS:
            if field not in p:
                _fail(msgs, f"{where}: missing required field {field!r}")
        pid = p.get("id")
        if pid in seen:
            _fail(msgs, f"{pid}: duplicate registry entry")
        seen.add(pid)

        if p.get("claim") not in claims:
            _fail(msgs, f"{where}: claim {p.get('claim')!r} is not a heading in CLAIMS.md "
                        f"(declared: {sorted(claims)})")

        ev = p.get("evaluable")
        if ev not in EVALUABLE_VALUES:
            _fail(msgs, f"{where}: evaluable={ev!r} not one of {EVALUABLE_VALUES}")

        rel = p.get("relevance")
        if not isinstance(rel, (int, float)) or not (0.1 <= float(rel) <= 1.0):
            _fail(msgs, f"{where}: relevance must lie in [0.1, 1.0] (POPPER's Listing-4 rubric)")
        elif float(rel) < r0 and ev == "e-value":
            # POPPER's relevance checker, made a declared field. Below r0 the
            # sub-hypothesis is not strongly enough implied by the main
            # hypothesis for its falsification to count as evidence -- and an
            # irrelevant null being "falsified" is what inflates Type-I error
            # (the paper measures 0.082 -> 0.340 with the checker removed).
            _fail(msgs, f"{where}: relevance {rel} is below r0={r0} but evaluable is "
                        f"'e-value'; it may not contribute to a claim's product")

        if ev == "e-value" and not str(p.get("null_construction", "")).strip():
            _fail(msgs, f"{where}: classified 'e-value' with no null_construction stated")
        if ev == "measurement" and "NONE" not in str(p.get("null_construction", "")).upper() \
                and "cannot" not in str(p.get("null_construction", "")).lower() \
                and "not" not in str(p.get("null_construction", "")).lower():
            _warn(msgs, f"{where}: classified 'measurement' but null_construction does not say "
                        f"why no valid null exists")

    # Fixed-in-advance parameters.
    kappa = reg.get("kappa")
    if not isinstance(kappa, (int, float)) or not (0.0 < float(kappa) < 1.0):
        _fail(msgs, f"kappa must lie in (0, 1); got {kappa!r}")
    alpha = reg.get("alpha")
    if not isinstance(alpha, (int, float)) or not (0.0 < float(alpha) < 1.0):
        _fail(msgs, f"alpha must lie in (0, 1); got {alpha!r}")


def check_coverage(reg: dict, msgs: List[str]) -> None:
    """Every prediction ID mentioned in the tree has a registry entry."""
    registered = {p["id"] for p in reg.get("predictions", []) if "id" in p}
    # The registry stores ASCII ids; the prose uses the Greek letters.
    alias = {"P-γ1": "P-gamma1", "P-γ2": "P-gamma2"}

    found: Dict[str, List[str]] = {}
    for path in sorted(list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.md"))):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.name in SCAN_EXCLUDE:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:                          # pragma: no cover - defensive
            continue
        for m in ID_PATTERN.finditer(text):
            pid = alias.get(m.group(0), m.group(0))
            found.setdefault(pid, []).append(str(path.relative_to(ROOT)))

    for pid, sources in sorted(found.items()):
        if pid not in registered:
            _fail(msgs, f"{pid} appears in {sources[0]} (and {len(sources)-1} other file(s)) "
                        f"but has no claims/registry.json entry")


# ---------------------------------------------------------------------------
# Check 2 — the pre-registration gate
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=False).stdout.strip()


def _commit_date(sha: str) -> Optional[str]:
    if not sha:
        return None
    out = _git("show", "-s", "--format=%cI", sha)
    return out or None


def _introducing_commit(pathspec: str, needle: str) -> Optional[Tuple[str, str]]:
    """First commit whose diff to `pathspec` introduced `needle`."""
    out = _git("log", "--reverse", "--format=%H %cI", "-S", needle, "--", pathspec)
    if not out:
        return None
    sha, date = out.splitlines()[0].split()
    return sha, date


def check_preregistration(reg: dict, msgs: List[str]) -> Tuple[int, int]:
    """
    Returns ``(n_gated, n_backfilled)``.

    Rules 1 and 3 need adjudication records to have anything to check; with
    none present the gate reports readiness rather than passing vacuously.
    """
    adjudications = load_adjudications()
    by_id = {p["id"]: p for p in reg.get("predictions", []) if "id" in p}

    n_backfilled = sum(1 for p in by_id.values()
                       if p.get("registered_provenance") == "backfilled")
    n_gated = len(by_id) - n_backfilled

    # Every entry must carry a registration commit -- backfilled or not.
    for pid, p in sorted(by_id.items()):
        if not p.get("registered_commit"):
            _warn(msgs, f"{pid}: no registered_commit; its pre-registration cannot be "
                        f"verified from history at all")

    if not adjudications:
        return n_gated, n_backfilled

    # Rule 1: registration strictly precedes adjudication.
    artifact_owner: Dict[str, Tuple[str, str]] = {}   # hash -> (pid, reg_date)
    for pid, adj in sorted(adjudications.items()):
        p = by_id.get(pid)
        if p is None:
            _fail(msgs, f"adjudication {pid}.json has no registry entry; an unregistered "
                        f"prediction cannot be adjudicated (its Assumption-2 status is unknown)")
            continue

        adj_commit = _introducing_commit(f"claims/adjudications/{pid}.json", pid)
        reg_date = p.get("registered_date")
        if adj_commit and reg_date:
            _, adj_date = adj_commit
            if adj_date[:10] < reg_date[:10]:
                _fail(msgs, f"{pid}: adjudicated ({adj_date[:10]}) BEFORE it was registered "
                            f"({reg_date}); the e-value is not conditionally valid")

        # Rule 3: artifact reuse by a later-registered prediction.
        for h in adj.get("artifact_hashes", []):
            prior = artifact_owner.get(h)
            if prior is None:
                artifact_owner[h] = (pid, reg_date or "")
            else:
                prior_pid, prior_date = prior
                if reg_date and prior_date and reg_date > prior_date:
                    _warn(msgs,
                          f"{pid} was registered ({reg_date}) AFTER {prior_pid} "
                          f"({prior_date}) consumed artifact {h[:12]}, and is adjudicated on "
                          f"the same artifact. Re-use is fine; registering after seeing it is "
                          f"not. Confirm this prediction was not shaped by that artifact.")

    # Rule 2: frozen fields unchanged since first adjudication.
    for pid in sorted(adjudications):
        p = by_id.get(pid)
        if p is None:
            continue
        for field in FROZEN_FIELDS:
            value = str(p.get(field, ""))
            if not value:
                continue
            hits = _git("log", "--format=%cI", "-S", value[:80], "--", "claims/registry.json")
            dates = sorted(hits.splitlines())
            if len(dates) > 1:
                _fail(msgs, f"{pid}: field {field!r} was modified after first registration "
                            f"({len(dates)} commits touch it); amendments belong in `notes` "
                            f"as dated addenda, not as edits")
    return n_gated, n_backfilled


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(reg: dict) -> None:
    from collections import Counter
    preds = reg.get("predictions", [])
    adj = load_adjudications()

    print(f"\nregistry: {len(preds)} predictions, {len(adj)} adjudicated\n")
    print(f"{'claim':<12} {'total':>6} {'e-value':>8} {'needs-null':>11} "
          f"{'measurement':>12} {'adjudicated':>12}")
    print("-" * 66)
    for claim in sorted({p["claim"] for p in preds}):
        rows = [p for p in preds if p["claim"] == claim]
        c = Counter(p["evaluable"] for p in rows)
        n_adj = sum(1 for p in rows if p["id"] in adj)
        print(f"{claim:<12} {len(rows):>6} {c['e-value']:>8} {c['needs-null']:>11} "
              f"{c['measurement']:>12} {n_adj:>12}")
    print("-" * 66)
    c = Counter(p["evaluable"] for p in preds)
    print(f"{'TOTAL':<12} {len(preds):>6} {c['e-value']:>8} {c['needs-null']:>11} "
          f"{c['measurement']:>12} {len(adj):>12}")

    n_bf = sum(1 for p in preds if p.get("registered_provenance") == "backfilled")
    print(f"\nPre-registration provenance: {len(preds) - n_bf} gated, {n_bf} backfilled.")
    print("Backfilled entries have their registration commit recovered from git history")
    print("rather than observed by the gate. Good evidence; not the same as having been")
    print("gated. Reported separately, permanently.")

    n_ev = c["e-value"]
    print(f"\n{n_ev} of {len(preds)} predictions can currently carry an e-value. The other")
    print(f"{len(preds) - n_ev} either need a null constructed or are measurements with no valid")
    print("null at all -- see claims/EVALUABILITY.md. core/adjudication.py refuses the")
    print("latter two rather than emitting a neutral e-value, because the product is only")
    print("as valid as its weakest factor.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="store_true", help="print the per-claim status table")
    ap.add_argument("--warnings-as-errors", action="store_true")
    args = ap.parse_args(argv)

    try:
        reg = load_registry()
    except Problem as exc:
        print(f"ERROR   {exc}")
        return 1

    msgs: List[str] = []
    check_registry(reg, msgs)
    check_coverage(reg, msgs)
    try:
        n_gated, n_backfilled = check_preregistration(reg, msgs)
    except Problem as exc:
        _fail(msgs, str(exc))
        n_gated = n_backfilled = 0

    for m in msgs:
        print(m)

    errors = [m for m in msgs if m.startswith("ERROR")]
    warnings = [m for m in msgs if m.startswith("WARNING")]
    print(f"\nregistry check: {len(errors)} error(s), {len(warnings)} warning(s); "
          f"{n_gated} gated / {n_backfilled} backfilled registrations")

    if args.summary:
        print_summary(reg)

    if errors or (args.warnings_as_errors and warnings):
        return 1
    print("registry OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
