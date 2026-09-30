#!/usr/bin/env python3
"""
tools/mutation_check.py — fail on any mutmut survivor nobody has reviewed.

`mutmut run` (config: pyproject.toml [tool.mutmut]) mutates `core/evalues.py`,
the e-value core that carries the Type-I guarantee, and runs its tests
against each mutant. A mutant that survives is a change to that code no test
notices. Some survivors are equivalent (the change cannot alter behaviour),
and those are listed in `tools/mutation_accepted.json` with the reason. This
script fails when:

  * a mutant survives that is not in the list;
  * a listed mutant's diff is not the one that was reviewed. Mutant names are
    numbered per function, so an edit to `core/evalues.py` renumbers them, and
    a name alone could silently accept a different change. The list is keyed
    by name *and* diff for that reason;
  * a listed mutant no longer survives (a stale entry: delete it);
  * any mutant ended in a state other than killed or survived (timeout,
    suspicious, not checked), which means the run did not test it;
  * an entry's reason is empty or TODO.

After editing `core/evalues.py`, run `mutmut run` and then
`python tools/mutation_check.py --write`. That rewrites the list with each
survivor's current diff, keeps reasons whose diff is unchanged, and marks
the rest TODO for review. Kill a survivor with a test where it is a real
gap; accept it only with a reason (`tests/test_core_evalues_contract.py`
records how the first run's 110 survivors went to 22).

Standard library only. Needs `mutmut` on PATH and a finished `mutmut run`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACCEPTED = ROOT / "tools" / "mutation_accepted.json"
OK_STATES = {"killed", "survived"}


def parse_results(text: str) -> dict:
    """`mutmut results --all true` output -> {mutant name: state}."""
    out = {}
    for line in text.splitlines():
        name, sep, state = line.strip().rpartition(": ")
        if sep and name and not name.startswith("#"):
            out[name] = state.strip()
    return out


def diff_lines(show_text: str) -> list:
    """The changed lines of `mutmut show <name>`, stripped: what was reviewed."""
    return [l[0] + l[1:].strip() for l in show_text.splitlines()
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]


def problems(states: dict, diffs: dict, accepted: dict) -> list:
    """Everything that should fail the check. `diffs` covers the survivors."""
    out = []
    for name, state in sorted(states.items()):
        if state not in OK_STATES:
            out.append(f"{name}: {state} (not tested; rerun `mutmut run`)")
    for name in sorted(n for n, s in states.items() if s == "survived"):
        entry = accepted.get(name)
        if entry is None:
            out.append(f"{name}: survived and is not reviewed:\n    "
                       + "\n    ".join(diffs.get(name, [])))
        elif entry.get("diff") != diffs.get(name):
            out.append(f"{name}: survived, but its diff is not the reviewed one "
                       f"(renumbered after an edit?). Now:\n    "
                       + "\n    ".join(diffs.get(name, [])))
        elif not entry.get("why", "").strip() or entry["why"].startswith("TODO"):
            out.append(f"{name}: accepted without a reason")
    for name in sorted(set(accepted) - {n for n, s in states.items() if s == "survived"}):
        out.append(f"{name}: listed as accepted but no longer survives; delete it")
    return out


def _mutmut(*args: str) -> str:
    return subprocess.run(["mutmut", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true",
                    help="rewrite tools/mutation_accepted.json from this run "
                         "(keeps reasons whose diff is unchanged, TODO otherwise)")
    args = ap.parse_args(argv)

    states = parse_results(_mutmut("results", "--all", "true"))
    if not states:
        print("mutation_check: no mutmut results; run `mutmut run` first", file=sys.stderr)
        return 1
    diffs = {n: diff_lines(_mutmut("show", n))
             for n, s in states.items() if s == "survived"}
    accepted = json.loads(ACCEPTED.read_text(encoding="utf-8")) if ACCEPTED.exists() else {}

    if args.write:
        new = {}
        for name, d in sorted(diffs.items()):
            old = accepted.get(name, {})
            why = old.get("why", "") if old.get("diff") == d else ""
            new[name] = {"diff": d, "why": why or "TODO: kill with a test, or say why it is equivalent"}
        ACCEPTED.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"wrote {ACCEPTED.relative_to(ROOT)}: {len(new)} survivors")

    found = problems(states, diffs, accepted if not args.write else new)
    n_killed = sum(s == "killed" for s in states.values())
    print(f"mutation_check: {n_killed}/{len(states)} killed, "
          f"{len(diffs)} survived, {len(found)} problem(s)")
    for p in found:
        print("  " + p)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
