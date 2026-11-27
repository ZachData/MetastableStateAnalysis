"""
tests/test_lit_ledger.py — tools/lit_ledger.py (the paper ledger) and
tools/lint_repo.py's `lit-index-row` rule, which share its scan-file pattern.

The rule must fire on a scan file with no index row, not only pass on the real
tree (LESSONS.md lesson 2). The mark reader must err low: each case below is a
shape seen in a real scan file where a looser reading gave an id a mark that was
not its own (a tag column, "not [R]", a prose line).
"""
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import lint_repo  # noqa: E402
import lit_ledger as L  # noqa: E402

INDEX = """\
# LITERATURE

## 1. The files

| Phase | File | State |
|---|---|---|
| 1 | `p1/lit-1.md` | Live |

## 2. Other

| `p2/lit-2.md` named outside §1 does not count |
"""


def _ids(line):
    return [pid for _, pid in L.arxiv_ids(line)]


def _marks(line, rel="p1/lit-1.md", default=None, row=None):
    row = line.startswith("|") if row is None else row
    return L.assign_marks(L.arxiv_ids(line), L.line_marks(line, rel), default, row, line)


# --- ids --------------------------------------------------------------------

def test_ids_new_style_only_with_valid_month_and_length():
    assert _ids("see 2509.23024v2 and 1007.1075") == ["2509.23024", "1007.1075"]
    assert _ids("2026.1101 1587.0000 1412.12345 1501.1234 0.2509.2302") == []
    assert _ids("arXiv:2402.04362, `2312.10794`") == ["2402.04362", "2312.10794"]


# --- marks ------------------------------------------------------------------

def test_adjacent_mark_is_the_ids_own():
    assert _marks("**2509.23024** **[S]**, then **2607.01940** **[A]**") == {
        "2509.23024": "S", "2607.01940": "A"}


def test_tag_column_with_several_marks_gives_the_weakest():
    line = "| Wang & Isola (2005.10242) | long text naming 2403.00642 too | [R] full; [S] |"
    assert _marks(line) == {"2005.10242": "S", "2403.00642": "S"}


def test_one_mark_in_a_row_covers_its_ids():
    assert _marks("| C8 | every row [S]: 2312.16903, 2309.14322 |") == {
        "2312.16903": "S", "2309.14322": "S"}


def test_an_id_beside_a_marked_sibling_in_its_cell_inherits_nothing():
    # lit-C2p-fulltext.md W3 (/challenge-pr on #388, finding 1).
    line = ("| W3 | quote | Koubbi et al., **2604.01978** [T]: attention-only. "
            "Fedorov et al. 2601.21942 (cited there) treat values |")
    assert _marks(line, rel="trackC_writeup/lit-C2p-fulltext.md") == {"2604.01978": "R"}


def test_a_files_own_r_can_be_dropped():
    # lit-C2p-fulltext.md's [R] is "a Hugging Face repository queried".
    assert _marks("| x | EleutherAI/pythia 2304.01373 [R] |",
                  rel="trackC_writeup/lit-C2p-fulltext.md") == {}


def test_prose_line_takes_only_an_adjacent_mark():
    line = "A long sentence about 2402.03698 and its withdrawal, then a word on [R]."
    assert _marks(line) == {}


def test_negated_mark_is_skipped():
    assert _marks("| T1 | 2606.22389 from a summarising fetch (not [R]) |") == {}


def test_header_default_only_for_rows_with_no_mark():
    assert _marks("| Takase 2312.16903 | spikes |", default="S") == {"2312.16903": "S"}
    assert _marks("| Cohen 2103.00065 [K] | prior |", default=None) == {}


def test_local_legend_maps_and_unknown_letters_are_dropped():
    line = "| Gupta 2308.04014 [F] | read |"
    assert _marks(line, rel="trackC_writeup/lit-C7.md") == {"2308.04014": "R"}
    assert _marks(line, rel="p1/lit-1.md") == {}


# --- the index-row rule -----------------------------------------------------

def _tree(tmp_path, monkeypatch, files):
    monkeypatch.setattr(lint_repo, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "LITERATURE.md").write_text(INDEX, encoding="utf-8")
    for rel in files:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x 2509.23024 [S]\n", encoding="utf-8")
    lint = lint_repo.Linter()
    lint_repo.rule_lit_index_rows(lint)
    return sorted(Path(f.path).as_posix().replace(f"{tmp_path.as_posix()}/", "")
                  for f in lint.findings)


def test_the_real_tree_passes():
    lint = lint_repo.Linter()
    lint_repo.rule_lit_index_rows(lint)
    assert [f.render() for f in lint.findings] == []


def test_rule_fires_on_a_scan_with_no_section_1_row(tmp_path, monkeypatch):
    found = _tree(tmp_path, monkeypatch, [
        "p1/lit-1.md", "p2/lit-2.md", "p8/literature-8.md",
        "archive/docs/literature_scan_2026-09-10.md", "tests/lit-fixture.md",
        "p1/notes-1.md"])
    assert found == ["archive/docs/literature_scan_2026-09-10.md",
                     "p2/lit-2.md", "p8/literature-8.md"]


# --- the ledger -------------------------------------------------------------

def test_render_without_git(tmp_path):
    (tmp_path / "docs" / "readings").mkdir(parents=True)
    (tmp_path / "p1").mkdir()
    (tmp_path / "p1" / "lit-1.md").write_text(
        "| a | **2509.23024** **[S]** |\n| b | 2411.04990 [N] |\n", encoding="utf-8")
    (tmp_path / "docs" / "readings" / "2411.04990.md").write_text("notes\n", encoding="utf-8")
    (tmp_path / "PLAN.md").write_text("cost tag [R] beside 2509.23024\n", encoding="utf-8")
    text = L.render(tmp_path)
    assert "no git history" in text
    rows = {l.split(" | ")[0][2:]: l for l in text.splitlines() if l.startswith("| 2")}
    assert "[S] `lit-1.md`" in rows["2509.23024"]          # PLAN.md's [R] is a cost tag
    assert "`PLAN.md`" in rows["2509.23024"]
    assert "[R] `docs/readings/2411.04990.md`" in rows["2411.04990"]


def test_check_sees_a_stale_mark_and_ignores_files_outside_scans(tmp_path):
    (tmp_path / "docs").mkdir()
    scan = tmp_path / "p1" / "lit-1.md"
    scan.parent.mkdir()
    scan.write_text("| a | **2509.23024** **[S]** |\n", encoding="utf-8")
    (tmp_path / L.LEDGER).write_text(L.render(tmp_path), encoding="utf-8")
    assert L.check(tmp_path) == []
    (tmp_path / "PLAN.md").write_text("now cited here too: 2509.23024, 2411.04990\n",
                                      encoding="utf-8")
    assert L.check(tmp_path) == []                       # not a scan: not checked
    scan.write_text("| a | **2509.23024** **[A]** |\n", encoding="utf-8")
    assert [d.split(":")[0] for d in L.check(tmp_path)] == ["2509.23024"]
