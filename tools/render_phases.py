#!/usr/bin/env python3
"""
tools/render_phases.py — the phase table (`docs/PHASES.md`) from the phase
cards, and the checks that keep the cards honest (`docs/PHASE_REVIEW.md`
session 1).

A card sits at the top of each `status-N.md`, between `CARD_BEGIN` and
`CARD_END`, with the fields in `FIELDS` in that order. `docs/phase_card.md`
is the template and says what each field holds.

What `card_findings` checks (lint rule `phase-card` calls it):

* every field present, in order, filled (no `TODO`), nothing extra;
* every Results / Superseded item carries a pointer, and every pointer
  resolves: a backticked repo path exists, a backticked .md path followed by
  §N or by "Heading" names a heading in that file, and a bare §N a heading
  in `PROJECT.md`.
  Paths under `data/` are not checked (no checkout has them);
* each "After Phase 10" item states its cost (free, or forward pass);
* **staleness**, by content rather than by commit: the card records a short
  hash of its own status file with the card cut out. Any change to that body
  since the review makes the card stale until someone re-reads the change and
  runs `--stamp <phase>`. Each Depends on entry is `<phase>@<hash>`, a hash
  of only that phase's `## Corrections received` section: a correction
  routed to a phase stales every card that reads it, one hop, while any
  other edit there does not (user, 2026-09-24, `docs/phase_card.md`);
* Depends on / Feeds agree between two phases that both have cards.

Phases without a card are listed in the table as such and not checked, so
the cards can land one review session at a time.

Standard library only; runs in CI tier 0.

Usage
-----
    python tools/render_phases.py            # rewrite docs/PHASES.md
    python tools/render_phases.py --check    # fail if it is out of step
    python tools/render_phases.py --stamp 1  # record the review of phase 1
    python tools/render_phases.py --hash 1   # print the body hash of phase 1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
OUT = Path("docs") / "PHASES.md"
TEMPLATE = Path("docs") / "phase_card.md"

CARD_BEGIN = "<!-- phase-card -->"
CARD_END = "<!-- /phase-card -->"

FIELDS = (
    "Question", "Inputs", "Results", "Superseded / wrong", "Registry",
    "Depends on", "Feeds", "Open threads", "After Phase 10", "Reviewed",
)
POINTER_FIELDS = ("Results", "Superseded / wrong")

_FIELD_LINE = re.compile(r"^- \*\*([^*]+):\*\*\s*(.*)$")
_ITEM_LINE = re.compile(r"^  - (.*)$")
_REVIEWED = re.compile(r"^(\d{4}-\d{2}-\d{2}) · body `([0-9a-f]{10})`$")
_PHASE_ID = re.compile(r"^[0-9]+[a-z]?(?:-frozen)?$")
_DEP = re.compile(r"^([0-9]+[a-z]?(?:-frozen)?)@([0-9a-f]{10})$")
CORRECTIONS = "## Corrections received"
_NONE = re.compile(r"^none\b", re.IGNORECASE)
_COST = re.compile(r"\((?:free|forward pass)[^)]*\)", re.IGNORECASE)
#: `path` optionally followed by §N or "Heading".
_POINTER = re.compile(
    r"`([^`\s]+)`(?:\s+(?:§([0-9]+(?:\.[0-9]+)*[a-z]?)|\"([^\"]+)\"))?"
    r"|§([0-9]+(?:\.[0-9]+)*[a-z]?)"
)
_PATHLIKE = re.compile(r"^[\w./-]+\.(?:md|py|json|sh)(?::[0-9,-]+)?$")
_SKIP_WALK = {".git", ".venv", "venv", "data", "__pycache__", ".pytest_cache"}


@dataclass
class Card:
    fields: Dict[str, Tuple[str, List[str]]]      # name -> (inline, items)
    order: List[str]
    lines: Dict[str, int]                          # name -> 1-based line
    errors: List[Tuple[int, str]] = field(default_factory=list)


@dataclass
class Phase:
    id: str
    path: Path          # relative to root
    text: str
    card: Optional[Card]
    card_line: int


# ---------------------------------------------------------------------------
# Discovery and parsing
# ---------------------------------------------------------------------------

def _sort_key(pid: str):
    m = re.match(r"^([0-9]+)(.*)$", pid)
    return (int(m.group(1)), m.group(2)) if m else (10**6, pid)


def discover(root: Path) -> Dict[str, Phase]:
    """Every `status-N.md` one level under the root or under `archive/`.

    A frozen copy whose id a live phase already uses (Phase 6 has both) takes
    the suffix `-frozen`."""
    live = sorted(root.glob("*/status-*.md"))
    frozen = sorted((root / "archive").glob("*/status-*.md"))
    phases: Dict[str, Phase] = {}
    for is_archive, paths in ((False, live), (True, frozen)):
        for p in paths:
            if not is_archive and p.parent.name == "archive":
                continue
            pid = p.stem[len("status-"):]
            if pid in phases and is_archive:
                pid += "-frozen"
            text = p.read_text(encoding="utf-8")
            card, line = parse_card(text)
            phases[pid] = Phase(pid, p.relative_to(root), text, card, line)
    return dict(sorted(phases.items(), key=lambda kv: _sort_key(kv[0])))


def _card_span(text: str) -> Optional[Tuple[int, int]]:
    b, e = text.find(CARD_BEGIN), text.find(CARD_END)
    if b < 0 and e < 0:
        return None
    return b, e


def parse_card(text: str) -> Tuple[Optional[Card], int]:
    span = _card_span(text)
    if span is None:
        return None, 0
    b, e = span
    first_line = text[:max(b, 0)].count("\n") + 1
    card = Card({}, [], {})
    if b < 0 or e < 0 or e < b or text.count(CARD_BEGIN) > 1 or text.count(CARD_END) > 1:
        card.errors.append((first_line, f"needs exactly one {CARD_BEGIN} followed by one {CARD_END}"))
        return card, first_line
    current = None
    for n, line in enumerate(text[b:e].splitlines(), start=first_line):
        fm = _FIELD_LINE.match(line)
        im = _ITEM_LINE.match(line)
        if fm:
            current = fm.group(1).strip()
            if current in card.fields:
                card.errors.append((n, f"field '{current}' appears twice"))
            card.fields[current] = (fm.group(2).strip(), [])
            card.order.append(current)
            card.lines[current] = n
        elif im and current:
            card.fields[current][1].append(im.group(1).strip())
        elif line.startswith("    ") and current and card.fields[current][1]:
            items = card.fields[current][1]
            items[-1] = f"{items[-1]} {line.strip()}"
        elif line.startswith("  ") and line.strip() and current:
            inline, items = card.fields[current]
            card.fields[current] = (f"{inline} {line.strip()}".strip(), items)
    return card, first_line


def body_hash(text: str) -> str:
    """sha256 of the file with its card cut out, blank runs collapsed."""
    span = _card_span(text)
    if span and span[0] >= 0 and span[1] > span[0]:
        text = text[:span[0]] + text[span[1] + len(CARD_END):]
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def corrections_hash(text: str) -> str:
    """sha256 of the `## Corrections received` section (empty if absent),
    normalised as in `body_hash`. What a reader's Depends on entry watches."""
    lines = text.replace("\r\n", "\n").split("\n")
    sec: List[str] = []
    inside = False
    for line in lines:
        if line.strip() == CORRECTIONS:
            inside = True
        elif inside and line.startswith("## "):
            break
        elif inside:
            sec.append(line)
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(sec)).strip()
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:10]


def corrections_heading_problems(text: str) -> List[str]:
    """A near-miss or repeated Corrections heading would hash as empty or
    partial, and its readers would never go stale; refuse it instead."""
    exact = [l for l in text.splitlines() if l.strip() == CORRECTIONS]
    near = [l.strip() for l in text.splitlines()
            if re.match(r"^#+\s*corrections?\s+received\b", l.strip(), re.IGNORECASE)
            and l.strip() != CORRECTIONS]
    out = [f"heading '{n}' should read exactly '{CORRECTIONS}'" for n in near]
    if len(exact) > 1:
        out.append(f"'{CORRECTIONS}' appears {len(exact)} times; merge them into one")
    return out


def _ids(value: str) -> List[str]:
    return [] if _NONE.match(value) else [v.strip() for v in value.split(",") if v.strip()]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

class _Resolver:
    def __init__(self, root: Path):
        self.root = root
        self._names: Optional[set] = None
        self._headings: Dict[Path, List[str]] = {}

    def names(self) -> set:
        if self._names is None:
            self._names = set()
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = [d for d in dirnames if d not in _SKIP_WALK]
                self._names.update(filenames)
        return self._names

    def file(self, cited: str, base: Path) -> Optional[Path]:
        cited = cited.split(":", 1)[0]
        for cand in (self.root / cited, self.root / base / cited):
            if cand.is_file():
                return cand
        if "/" not in cited and cited in self.names():
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = [d for d in dirnames if d not in _SKIP_WALK]
                if cited in filenames:
                    return Path(dirpath) / cited
        return None

    def headings(self, path: Path) -> List[str]:
        if path not in self._headings:
            self._headings[path] = [l for l in path.read_text(encoding="utf-8").splitlines()
                                    if l.startswith("#")]
        return self._headings[path]

    def has_section(self, path: Path, sec: str) -> bool:
        # The number must open the heading: "### 3.41 …", "## 3. …", "### §3.2 …".
        # Anywhere-in-heading let §12 match a date and §7 a `7d` (#75 review).
        pat = re.compile(rf"^#+\s+(?:\*\*)?§?{re.escape(sec)}(?![0-9]|\.[0-9])")
        return any(pat.match(h) for h in self.headings(path))

    def has_heading(self, path: Path, title: str) -> bool:
        return any(title in h for h in self.headings(path))


def _check_pointers(text: str, base: Path, res: _Resolver) -> Tuple[bool, List[str]]:
    """(has a pointer, unresolved pointers)."""
    found, bad = False, []
    for m in _POINTER.finditer(text):
        cited, sec, title, bare = m.groups()
        if bare:
            found = True
            before = text[:m.start()].rstrip()
            if before.endswith(".md"):
                bad.append(f"§{bare} follows an unbackticked file name, so it would be read "
                           f"as PROJECT.md §{bare}; backtick the file or drop the §")
            elif not res.has_section(res.root / "PROJECT.md", bare):
                bad.append(f"§{bare} (no such heading in PROJECT.md)")
            continue
        if not _PATHLIKE.match(cited):
            continue                       # a code identifier, not a pointer
        found = True
        if cited.startswith("data/"):
            continue
        path = res.file(cited, base)
        if path is None:
            bad.append(f"`{cited}` (no such file)")
        elif sec and path.suffix == ".md" and not res.has_section(path, sec):
            bad.append(f"`{cited}` §{sec} (no such heading)")
        elif title and path.suffix == ".md" and not res.has_heading(path, title):
            bad.append(f"`{cited}` \"{title}\" (no such heading)")
    return found, bad


def _check_shape(card: Card) -> List[Tuple[int, str]]:
    out = list(card.errors)
    extra = [f for f in card.order if f not in FIELDS]
    missing = [f for f in FIELDS if f not in card.fields]
    if extra:
        out.append((card.lines[extra[0]], f"unknown field(s) {extra}; the fields are {list(FIELDS)}"))
    if missing:
        out.append((0, f"missing field(s) {missing}"))
    if not extra and not missing and card.order != list(FIELDS):
        out.append((0, f"fields out of order; the order is {list(FIELDS)}"))
    return out


def card_findings(root: Path = ROOT) -> List[Tuple[str, int, str]]:
    """(relative path, line, message) for every problem with any card."""
    findings: List[Tuple[str, int, str]] = []
    res = _Resolver(root)

    tpl = root / TEMPLATE
    if not tpl.is_file():
        findings.append((str(TEMPLATE), 0, "missing: the card template"))
    else:
        card, line = parse_card(tpl.read_text(encoding="utf-8"))
        if card is None:
            findings.append((str(TEMPLATE), 0, "carries no card skeleton"))
        else:
            findings += [(str(TEMPLATE), n or line, m) for n, m in _check_shape(card)]

    phases = discover(root)
    for ph in phases.values():
        if ph.card is None:
            continue
        rel, card = str(ph.path), ph.card
        shape = _check_shape(card)
        findings += [(rel, n or ph.card_line, m) for n, m in shape]
        if shape:
            continue
        at = card.lines
        for name in FIELDS:
            inline, items = card.fields[name]
            whole = " ".join([inline] + items)
            if not whole.strip():
                findings.append((rel, at[name], f"'{name}' is empty"))
            elif "TODO" in whole:
                findings.append((rel, at[name], f"'{name}' still has a TODO"))
        for name in POINTER_FIELDS:
            inline, items = card.fields[name]
            if name == "Results" and not items:
                findings.append((rel, at[name], "'Results' needs one item per result"))
            if _NONE.match(inline) and not items:
                continue
            for item in items or [inline]:
                has, bad = _check_pointers(item, ph.path.parent, res)
                if not has:
                    findings.append((rel, at[name], f"'{name}' item has no pointer: {item[:60]}…"))
        for name in FIELDS:
            inline, items = card.fields[name]
            for piece in [inline] + items:
                for b in _check_pointers(piece, ph.path.parent, res)[1]:
                    findings.append((rel, at[name], f"'{name}': pointer does not resolve: {b}"))
        inline, items = card.fields["After Phase 10"]
        for item in items or [inline]:
            if not _NONE.match(item) and not _COST.search(item):
                findings.append((rel, at["After Phase 10"],
                                 f"'After Phase 10' item has no cost, (free) or (forward pass …): {item[:60]}…"))

        m = _REVIEWED.match(card.fields["Reviewed"][0])
        if not m:
            findings.append((rel, at["Reviewed"], "'Reviewed' must read YYYY-MM-DD · body `<hash>`; "
                                                  f"run `tools/render_phases.py --stamp {ph.id}`"))
        elif m.group(2) != body_hash(ph.text):
            findings.append((rel, at["Reviewed"],
                             f"STALE: {ph.path.name} changed outside its card since the review on "
                             f"{m.group(1)}; read what changed, update the card, then "
                             f"`tools/render_phases.py --stamp {ph.id}`"))

        for dep in _ids(card.fields["Depends on"][0]):
            dm = _DEP.match(dep)
            if not dm:
                findings.append((rel, at["Depends on"], f"'{dep}': write each dependency as <phase>@<hash> "
                                                        f"(`--stamp {ph.id}` fills the hashes)"))
            elif dm.group(1) not in phases:
                findings.append((rel, at["Depends on"], f"no phase '{dm.group(1)}'"))
            elif dm.group(2) != corrections_hash(phases[dm.group(1)].text):
                up = phases[dm.group(1)]
                since = m.group(1) if m else "<Reviewed date>"
                findings.append((rel, at["Depends on"],
                                 f"STALE: phase {up.id}'s {CORRECTIONS} does not match this card's "
                                 f"hash (a correction was routed there, or the hash predates "
                                 f"2026-09-24); read `git log -p --since={since} -- {up.path}`, "
                                 f"fix this card if it is touched, then `--stamp {ph.id}`"))
        for fed in _ids(card.fields["Feeds"][0]):
            if not _PHASE_ID.match(fed) or fed not in phases:
                findings.append((rel, at["Feeds"], f"no phase '{fed}'"))

    # A phase whose Corrections section is watched: the carded ones and every
    # dependency target, carded or not.
    watched = {p.id for p in phases.values() if p.card}
    for p in phases.values():
        if p.card and "Depends on" in p.card.fields:
            watched |= {d.split("@")[0] for d in _ids(p.card.fields["Depends on"][0])}
    for pid in sorted(watched & set(phases)):
        for msg in corrections_heading_problems(phases[pid].text):
            findings.append((str(phases[pid].path), 1, msg))

    # Depends on / Feeds agree where both ends have a card.
    carded = {p.id: p for p in phases.values() if p.card and not _check_shape(p.card)}
    for a in carded.values():
        feeds = set(_ids(a.card.fields["Feeds"][0]))
        for b in carded.values():
            deps = {d.split("@")[0] for d in _ids(b.card.fields["Depends on"][0])}
            if (b.id in feeds) != (a.id in deps):
                findings.append((str(a.path), a.card.lines["Feeds"],
                                 f"phase {a.id} Feeds and phase {b.id} Depends on disagree "
                                 f"about {a.id} → {b.id}"))
    return findings


# ---------------------------------------------------------------------------
# Rendering and stamping
# ---------------------------------------------------------------------------

HEADER = (
    "<!-- Generated by tools/render_phases.py from the cards in each status-N.md; do not edit. -->\n"
    "# Phases\n\n"
    "One row per `status-N.md`. The card at the top of each file is the source; this\n"
    "table only lines them up. What a card holds and how the lint keeps it current:\n"
    "`docs/phase_card.md`. The review that writes the cards: `docs/PHASE_REVIEW.md`.\n"
    "Phase → directory, and phases with no status file: `INDEX.md`.\n\n"
    "| phase | status file | question | registry | depends on | feeds | reviewed |\n"
    "|---|---|---|---|---|---|---|\n"
)


def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def render(root: Path = ROOT) -> str:
    rows = []
    for ph in discover(root).values():
        link = f"[`{ph.path}`](../{ph.path})"
        if ph.card is None or _check_shape(ph.card):
            rows.append(f"| {ph.id} | {link} | *no card yet* | | | | |")
            continue
        f = ph.card.fields
        deps = ", ".join(d.split("@")[0] for d in _ids(f["Depends on"][0])) or "none"
        m = _REVIEWED.match(f["Reviewed"][0])
        rows.append(
            f"| {ph.id} | {link} | {_cell(f['Question'][0])} | {_cell(f['Registry'][0])} | "
            f"{deps} | {_cell(f['Feeds'][0]) or 'none'} | {m.group(1) if m else '?'} |"
        )
    return HEADER + "\n".join(rows) + "\n"


def stamp(pid: str, root: Path = ROOT, today: Optional[str] = None) -> str:
    """Rewrite phase `pid`'s Reviewed line, and its Depends on hashes."""
    phases = discover(root)
    if pid not in phases:
        raise SystemExit(f"no phase '{pid}'; known: {', '.join(phases)}")
    ph = phases[pid]
    if ph.card is None or _check_shape(ph.card):
        raise SystemExit(f"{ph.path}: no well-formed card to stamp")
    today = today or _dt.date.today().isoformat()
    deps = []
    for d in _ids(ph.card.fields["Depends on"][0]):
        dep = d.split("@")[0]
        if dep not in phases:
            raise SystemExit(f"{ph.path}: depends on unknown phase '{dep}'")
        deps.append(f"{dep}@{corrections_hash(phases[dep].text)}")
    lines = ph.text.splitlines(keepends=True)
    # A field runs from its own line to the next field's (or the end marker),
    # so a value wrapped over several lines is replaced whole. Last field
    # first, so the earlier line numbers still hold.
    end_line = ph.text[:ph.text.find(CARD_END)].count("\n") + 1
    starts = sorted(ph.card.lines.values()) + [end_line]
    for key, value in (("Reviewed", f"{today} · body `{body_hash(ph.text)}`"),
                       ("Depends on", ", ".join(deps) or "none")):
        n = ph.card.lines[key]
        nxt = next(s for s in starts if s > n)
        lines[n - 1:nxt - 1] = [f"- **{key}:** {value}\n"]
    text = "".join(lines)
    (root / ph.path).write_text(text, encoding="utf-8")
    return text


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if docs/PHASES.md is out of step")
    ap.add_argument("--stamp", metavar="PHASE", help="record a review of PHASE's card")
    ap.add_argument("--hash", metavar="PHASE", help="print PHASE's body hash")
    args = ap.parse_args(argv)

    if args.hash:
        phases = discover(ROOT)
        if args.hash not in phases:
            print(f"no phase '{args.hash}'", file=sys.stderr)
            return 2
        print(body_hash(phases[args.hash].text))
        return 0
    if args.stamp:
        stamp(args.stamp)
        print(f"stamped phase {args.stamp}; now regenerate docs/PHASES.md")
        return 0

    text = render()
    out = ROOT / OUT
    if args.check:
        if not out.is_file() or out.read_text(encoding="utf-8") != text:
            print(f"{OUT} is out of step with the phase cards; run "
                  f"`python3 tools/render_phases.py`", file=sys.stderr)
            return 1
        print(f"{OUT} in step ({text.count(chr(10)) - HEADER.count(chr(10))} phases)")
        return 0
    out.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
