"""C2's LaTeX source from C2p unit 2 on (`trackC_writeup/paper/`, STATE Blocked 108 (a)): every number
in it is one a record renders to or the record-backed draft writes (`trackC_writeup/paper_check.py`).
When this fails on the user's rewrite, the failure lists the numbers to source (a slot in
`paper_slots.json`) or drop."""
import json

import pytest

pytestmark = pytest.mark.pure
from trackC_writeup import numbers as n
from trackC_writeup import paper_check as pc

KEYS = json.loads(n.TABLE.read_text())["keys"]


def test_every_number_in_paper_is_accounted_for():
    bad = pc.unsourced(KEYS)
    assert not bad, "\n".join(f"{f}:{line}: {num}  … {ctx} …" for f, line, num, ctx in bad)


def test_a_changed_or_new_number_is_listed():
    keyed, literal = pc.accounted(KEYS)
    tex = "\\documentclass{article}\n\\begin{document}\nE = 1590, p = 9.8 $\\times 10^{-8}$, at 12 000 steps.\n"
    found = [num for _, num, _ in pc.tex_numbers(tex) if num not in keyed | literal]
    assert found == ["1590", "9.8", "12 000"]


def test_names_and_the_preamble_are_not_numbers():
    tex = ("\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
           "See \\S\\ref{sec:4.1} and figure~\\ref{fig:fig11_rho} \\citep[\\S{}3.2]{geshkovski2023}.\n"
           "\\includegraphics[width=\\linewidth]{fig/fig9_massive.pdf} % 77 a comment\n")
    assert pc.tex_numbers(tex) == []


def test_every_entry_in_paper_slots_is_a_slot_or_a_reasoned_literal():
    for entry in json.loads(pc.SLOTS.read_text()):
        if "literal" in entry:
            assert set(entry) == {"literal", "why"} and entry["why"].strip()
        else:
            assert set(entry) == {"slot", "where"}
            n.slot_text(n.SLOT.fullmatch(entry["slot"]), KEYS)


def test_a_literal_without_a_reason_refuses(tmp_path, monkeypatch):
    bad = tmp_path / "slots.json"
    bad.write_text(json.dumps([{"literal": "0.625", "why": " "}]))
    monkeypatch.setattr(pc, "SLOTS", bad)
    with pytest.raises(n.Refused):
        pc.accounted(KEYS)


def test_figures_in_paper_are_their_generated_copies():
    assert pc.stale_figures() == []
