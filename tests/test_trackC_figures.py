"""C2's figures (`trackC_writeup/numbers.py` `figures`) draw from the tracked table alone. Not
compared byte for byte with the tracked SVGs: matplotlib's output moves between versions."""
import json

import pytest

pytestmark = pytest.mark.deps  # matplotlib
from trackC_writeup import numbers as n


def test_figures_draw_from_the_table(tmp_path, monkeypatch):
    monkeypatch.setattr(n, "FIGDIR", tmp_path)
    paths = n.figures(json.loads(n.TABLE.read_text())["keys"])
    assert sorted(p.name for p in paths) == ["fig1_break.svg", "fig2_timeline.svg", "fig3_ov.svg",
                                             "fig4_schedules.svg", "fig5_count.svg"]
    assert all(p.read_text().lstrip().startswith("<?xml") and "<svg" in p.read_text() for p in paths)
