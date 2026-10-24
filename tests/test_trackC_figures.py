"""C2's figures (`trackC_writeup/numbers.py` `figures`) draw from the tracked table alone. Not
compared byte for byte with the tracked SVGs: matplotlib's output moves between versions."""
import json
import re

import pytest

pytestmark = pytest.mark.deps  # matplotlib
from trackC_writeup import numbers as n


def test_figures_draw_from_the_table(tmp_path, monkeypatch):
    monkeypatch.setattr(n, "FIGDIR", tmp_path / "svg")
    monkeypatch.setattr(n, "PDFDIR", tmp_path / "pdf")      # never the tracked tex/fig/
    paths = n.figures(json.loads(n.TABLE.read_text())["keys"])
    in_draft = re.findall(r"^!\[Figure \d+\]\(fig/([\w-]+\.svg)\)", n.DRAFT.read_text(), re.M)
    assert sorted(p.name for p in paths) == sorted(in_draft)
    assert all(p.read_text().lstrip().startswith("<?xml") and "<svg" in p.read_text() for p in paths)
    assert sorted(p.name for p in (tmp_path / "pdf").iterdir()) == sorted(f.replace(".svg", ".pdf") for f in in_draft)
