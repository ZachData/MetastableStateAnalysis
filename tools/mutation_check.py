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
    numbered per function, so an edit to a function renumbers its mutants,
    and a name alone could silently accept a different change;
  * the code the reason was argued from has changed. An equivalence usually
    rests on more than the mutated line (an early return further down, a
    check in a callee), so each entry records `context`: a hash of the
    mutated function's code and of every function in the module it names,
    transitively (methods of its own class through `self.`/`cls.`).
    Docstrings, comments and blank lines do not count. It does not cover
    module constants, code outside the module, or callers (a reason that
    rests on who calls a function needs its own test);
  * a listed mutant no longer survives (a stale entry: delete it);
  * any mutant ended in a state other than killed or survived (timeout,
    suspicious, not checked), which means the run did not test it;
  * an entry's reason is empty, TODO, or RECONFIRM.

After editing `core/evalues.py`, run `mutmut run` and then
`python tools/mutation_check.py --write`. That rewrites the list: a reason
whose diff and context are unchanged is kept; one whose diff is unchanged but
whose context changed (including the same change renumbered within its
function) is kept behind a RECONFIRM prefix, to be re-read against the new
code and the prefix deleted; anything else is TODO. Kill a survivor with a
test where it is a real gap; accept it only with a reason, and probe the
boundary the reason names before writing it (`LESSONS.md` lesson 6).

Standard library only. Needs `mutmut` on PATH and a finished `mutmut run`.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import subprocess
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACCEPTED = ROOT / "tools" / "mutation_accepted.json"
OK_STATES = {"killed", "survived"}
TODO = "TODO: kill with a test, or say why it is equivalent"
RECONFIRM = "RECONFIRM (the code changed since this was reviewed): "
REMEDY = ("Run `mutmut run`, then `python tools/mutation_check.py --write`; re-read each "
          "RECONFIRM reason against the new code (probe it on the mutant) and delete the "
          "prefix. Do not paste in the new hash.")


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


def split_name(name: str) -> tuple:
    """`core.evalues.xǁEProcessǁadd__mutmut_3` -> ("core/evalues.py", ("EProcess", "add"))."""
    module, _, rest = name.rpartition(".")
    func = rest.rsplit("__mutmut_", 1)[0]
    qual = tuple(func[2:].split("ǁ")) if func.startswith("xǁ") else (func[2:],)
    return module.replace(".", "/") + ".py", qual


def _code_lines(source: str, tree: ast.Module) -> list:
    """`source`'s lines with comments and docstrings blanked: the code a reason rests on.

    Line numbers from the AST and COMMENT tokens are the same on every Python
    this repo supports, so the result is too (unlike `ast.dump`/`ast.unparse`
    or the full token stream, which changed for f-strings in 3.12).
    """
    lines = source.splitlines()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            lines[row - 1] = lines[row - 1][:col]
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
                and body[0].lineno > getattr(node, "lineno", 0)):
            for i in range(body[0].lineno - 1, body[0].end_lineno):
                lines[i] = ""
    return [l.rstrip() for l in lines]


def context_hash(source: str, qual: tuple) -> str:
    """Hash of the code of function `qual` in `source` and of the module functions it names."""
    tree = ast.parse(source)
    code = _code_lines(source, tree)
    defs = (ast.FunctionDef, ast.AsyncFunctionDef)
    funcs = {n.name: n for n in tree.body if isinstance(n, defs)}
    classes = {n.name: {m.name: m for m in n.body if isinstance(m, defs)}
               for n in tree.body if isinstance(n, ast.ClassDef)}

    def node(q):
        return funcs[q[0]] if len(q) == 1 else classes[q[0]][q[1]]

    seen, todo = set(), [tuple(qual)]
    while todo:
        q = todo.pop()
        if q in seen:
            continue
        seen.add(q)
        for sub in ast.walk(node(q)):
            if isinstance(sub, ast.Name) and sub.id in funcs:
                todo.append((sub.id,))
            elif (len(q) == 2 and isinstance(sub, ast.Attribute)
                  and isinstance(sub.value, ast.Name) and sub.value.id in ("self", "cls")
                  and sub.attr in classes[q[0]]):
                todo.append((q[0], sub.attr))
    h = hashlib.sha256()
    for q in sorted(seen):
        n = node(q)
        text = "\n".join(l for l in code[n.lineno - 1:n.end_lineno] if l.strip())
        h.update(".".join(q).encode() + b"\0" + text.encode() + b"\0")
    return h.hexdigest()[:12]


def contexts_for(names) -> dict:
    """{mutant name: context_hash} against the unmutated source under ROOT."""
    sources, out = {}, {}
    for name in names:
        path, qual = split_name(name)
        if path not in sources:
            sources[path] = (ROOT / path).read_text(encoding="utf-8")
        out[name] = context_hash(sources[path], qual)
    return out


def problems(states: dict, diffs: dict, accepted: dict, contexts: dict) -> list:
    """Everything that should fail the check. `diffs`, `contexts` cover the survivors."""
    out = []
    for name, state in sorted(states.items()):
        if state not in OK_STATES:
            out.append(f"{name}: {state} (not tested; rerun `mutmut run`)")
    for name in sorted(n for n, s in states.items() if s == "survived"):
        entry = accepted.get(name)
        why = (entry or {}).get("why", "").strip()
        if entry is None:
            out.append(f"{name}: survived and is not reviewed:\n    "
                       + "\n    ".join(diffs.get(name, [])))
        elif entry.get("diff") != diffs.get(name):
            out.append(f"{name}: survived, but its diff is not the reviewed one "
                       f"(renumbered after an edit?). Now:\n    "
                       + "\n    ".join(diffs.get(name, [])))
        elif entry.get("context") != contexts.get(name):
            out.append(f"{name}: its function, or one it calls, changed since the "
                       f"reason was written. {REMEDY}")
        elif not why or why.startswith("TODO"):
            out.append(f"{name}: accepted without a reason")
        elif why.startswith("RECONFIRM"):
            out.append(f"{name}: re-read the reason against the changed code, "
                       f"then delete the RECONFIRM prefix")
    for name in sorted(set(accepted) - {n for n, s in states.items() if s == "survived"}):
        out.append(f"{name}: listed as accepted but no longer survives; delete it")
    return out


def rewrite(diffs: dict, contexts: dict, accepted: dict) -> dict:
    """The list for this run's survivors, carrying each reviewed reason it can."""
    new = {}
    for name, d in sorted(diffs.items()):
        path_qual = split_name(name)
        same = accepted.get(name)
        if not same or same.get("diff") != d:
            # The same change renumbered within its function, if there is one.
            same = next((e for n, e in sorted(accepted.items())
                         if split_name(n) == path_qual and e.get("diff") == d), None)
        why = (same or {}).get("why", "").strip()
        if not why or why.startswith("TODO"):
            why = TODO
        elif same.get("context") != contexts[name]:
            why = RECONFIRM + why.removeprefix(RECONFIRM)
        new[name] = {"diff": d, "context": contexts[name], "why": why}
    return new


def _mutmut(*args: str) -> str:
    return subprocess.run(["mutmut", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true",
                    help="rewrite tools/mutation_accepted.json from this run "
                         "(keeps reviewed reasons, RECONFIRM or TODO otherwise)")
    args = ap.parse_args(argv)

    states = parse_results(_mutmut("results", "--all", "true"))
    if not states:
        print("mutation_check: no mutmut results; run `mutmut run` first", file=sys.stderr)
        return 1
    diffs = {n: diff_lines(_mutmut("show", n))
             for n, s in states.items() if s == "survived"}
    contexts = contexts_for(diffs)
    accepted = json.loads(ACCEPTED.read_text(encoding="utf-8")) if ACCEPTED.exists() else {}

    untested = sum(s not in OK_STATES for s in states.values())
    if args.write and untested:
        # A failed clean run leaves every mutant "not checked"; rewriting from
        # it would empty the list and discard every reviewed reason.
        print(f"mutation_check: refusing --write: {untested} of {len(states)} mutants "
              f"were not tested; fix the run first", file=sys.stderr)
        return 1
    if args.write:
        accepted = rewrite(diffs, contexts, accepted)
        ACCEPTED.write_text(json.dumps(accepted, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"wrote {ACCEPTED.relative_to(ROOT)}: {len(accepted)} survivors")

    found = problems(states, diffs, accepted, contexts)
    n_killed = sum(s == "killed" for s in states.values())
    print(f"mutation_check: {n_killed}/{len(states)} killed, "
          f"{len(diffs)} survived, {len(found)} problem(s)")
    for p in found:
        print("  " + p)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
