"""
trackC_writeup/paper_check.py — every number in the user's LaTeX source `paper/` that no record
renders to (C2p unit 2; STATE Blocked 108 (a)).

From C2p unit 2 on, `paper/` (Overleaf's folder) is the draft's source and nothing generated writes
into it, so its numbers are no longer filled from `numbers.json`. This check lists the ones that
are not accounted for, for the user to source or drop. A number in `paper/*.tex` (after
`\\begin{document}`) is accounted for when it is

- **keyed**: printed by a `{{key}}` slot of `draft-C2.src.md` (the record-backed draft), or by a
  slot in `paper_slots.json` (numbers new to `paper/`, each slot with where it is used); or
- **literal**: written by hand in `draft-C2.src.md` outside its slots (notation, a cited paper's
  numbers, a citation's year or locator), which C2's reviews already read.

Numbers are compared as tokens ("143 000", "0.19", "10" and "4" of "10^{-4}"), not in context: a
keyed 9 accounts for every 9, so a changed small integer can pass unseen; the check catches a new
or changed number that matches nothing anywhere, which is the drift a rewrite introduces.
Labels, references, citation keys and figure paths are skipped.

    python -m trackC_writeup.paper_check          # lists them; exit 1 if any
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from trackC_writeup import numbers as n

PAPER = n.HERE / "paper"
SLOTS = n.HERE / "paper_slots.json"

_SUPER = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉", " 01234567890123456789")
NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?: \d{3})+(?![\d.])|\d+(?:\.\d+)?")
#: LaTeX that carries a name, not a number: labels, references, citation keys, figure paths, environments.
SKIP = re.compile(r"\\(?:label|ref|eqref|pageref|cite[tp]?|citeauthor|citeyear|includegraphics|begin|end|"
                  r"bibliography|bibliographystyle|hspace|vspace)\*?(?:\[[^\]]*\])*\{[^}]*\}")


def tokens(text: str) -> List[str]:
    """The numbers in a text, "143 000" kept whole; superscript and subscript digits read as digits."""
    return NUMBER.findall(text.translate(_SUPER).replace(" ", " "))


def tex_numbers(tex: str) -> List[Tuple[int, str, str]]:
    """(line, number, context) for each number in the body of a LaTeX file."""
    lines = tex.splitlines()
    start = next((i + 1 for i, l in enumerate(lines) if l.strip().startswith(r"\begin{document}")), 0)
    out = []
    for i, line in enumerate(lines[start:], start + 1):
        body = SKIP.sub(" ", re.sub(r"(?<!\\)%.*", "", line))
        for m in NUMBER.finditer(body):
            out.append((i, m.group(), body[max(0, m.start() - 40):m.end() + 40].strip()))
    return out


def accounted(keys: Dict[str, dict]) -> Tuple[Set[str], Set[str]]:
    """(keyed, literal): the numbers the record-backed draft and `paper_slots.json` print."""
    src = n.SOURCE.read_text()
    keyed = set()
    for m in n.SLOT.finditer(src):
        keyed.update(tokens(n.slot_text(m, keys)))
    for entry in json.loads(SLOTS.read_text()):
        m = n.SLOT.fullmatch(entry["slot"])
        if not m:
            raise n.Refused(f"refusing: {entry['slot']!r} in {SLOTS.name} is not one slot")
        keyed.update(tokens(n.slot_text(m, keys)))
    literal = set(tokens(n.SLOT.sub(" ", src)))
    return keyed, literal


def unsourced(keys: Dict[str, dict]) -> List[Tuple[str, int, str, str]]:
    keyed, literal = accounted(keys)
    out = []
    for path in sorted(PAPER.glob("*.tex")):
        for line, num, ctx in tex_numbers(path.read_text()):
            if num not in keyed and num not in literal:
                out.append((path.name, line, num, ctx))
    return out


def main() -> int:
    keys = json.loads(n.TABLE.read_text())["keys"]
    bad = unsourced(keys)
    for name, line, num, ctx in bad:
        print(f"{name}:{line}: {num}    … {ctx} …")
    print(f"{len(bad)} number(s) in {PAPER.relative_to(n.REPO)}/ that no key renders to and the "
          f"record-backed draft does not write", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
