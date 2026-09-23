"""
tests/test_lint_cited_md_paths.py — tools/lint_repo.py's `cited-md-path` rule
and tools/rewrite_moved_refs.py, which share archive/MOVED.md.

The rule must fire on a dangling citation, not only pass on the real tree
(LESSONS.md lesson 2), and the rewrite must be idempotent, move `§1.5` with
`§1`, and leave `§10` alone.
"""
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import lint_repo  # noqa: E402
import rewrite_moved_refs as rw  # noqa: E402

MOVED = """\
# map

## Moved

| old | new | when |
|---|---|---|
| `OLD_PLAN.md` | `archive/OLD_PLAN.md` | 2026-09-22 |
| `docs/SCAN.md` | `archive/docs/SCAN.md` | 2026-09-22 |
| `BIG.md §1` | `archive/BIG-start.md` | 2026-09-22 |
| `PLAN.md §B3` | `archive/PLAN-done.md §B3` | 2026-09-22 |
| `PLAN.md §B4` | `archive/PLAN-done.md §B4` | 2026-09-22 |

## Absent

| cited | why |
|---|---|
| `NEVER.md` | never written |
"""


def _tree(tmp_path, monkeypatch, cites: str):
    monkeypatch.setattr(lint_repo, "ROOT", tmp_path)
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "MOVED.md").write_text(MOVED, encoding="utf-8")
    (tmp_path / "archive" / "OLD_PLAN.md").write_text("x\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "here.md").write_text("x\n", encoding="utf-8")
    (tmp_path / "doc.md").write_text(cites, encoding="utf-8")
    lint = lint_repo.Linter()
    lint_repo.rule_cited_md_paths(lint)
    return [f.message for f in lint.findings]


def test_the_real_tree_passes():
    lint = lint_repo.Linter()
    lint_repo.rule_cited_md_paths(lint)
    assert [f.render() for f in lint.findings] == []


def test_existing_moved_absent_and_patterns_pass(tmp_path, monkeypatch):
    cites = ("`sub/here.md` `here.md` `OLD_PLAN.md` `NEVER.md` `status-N.md` "
             "`<phase>/lit-N.md` raw.githubusercontent.com/x/README.md\n")
    assert _tree(tmp_path, monkeypatch, cites) == []


def test_a_dangling_path_fails(tmp_path, monkeypatch):
    msgs = _tree(tmp_path, monkeypatch, "see `sub/gone.md` and `gone.md`\n")
    assert len(msgs) == 2 and "sub/gone.md" in msgs[0]


def test_an_old_path_in_the_moved_table_passes_unrewritten(tmp_path, monkeypatch):
    # Registry-derived text keeps old paths; the map is what makes them resolve.
    assert _tree(tmp_path, monkeypatch, "per `docs/SCAN.md` §6\n") == []
    assert _tree_again_without_row(tmp_path, monkeypatch) != []


def _tree_again_without_row(tmp_path, monkeypatch):
    moved = tmp_path / "archive" / "MOVED.md"
    moved.write_text(MOVED.replace("| `docs/SCAN.md` |", "| `docs/OTHER.md` |"), encoding="utf-8")
    lint = lint_repo.Linter()
    (tmp_path / "doc.md").write_text("per `docs/SCAN.md` §6\n", encoding="utf-8")
    lint_repo.rule_cited_md_paths(lint)
    return lint.findings


def test_rewrite_paths_sections_and_idempotence(tmp_path):
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "MOVED.md").write_text(MOVED, encoding="utf-8")
    rules = rw.build_rules(rw.moved_table(tmp_path), live=["doc.md"], root=tmp_path)
    text = ("`OLD_PLAN.md` §6; BIG.md §1 and `BIG.md` §1.5 and BIG.md §10; "
            "`PLAN.md` §B3, PLAN §B3, PLAN.md item B3, PLAN.md items B3 and B4, "
            "PLAN.md B3, PLAN.md §B30\n")
    once, n = rw.rewrite(text, rules)
    # §1.5 is inside §1, so it moves and keeps its label; §10 and §B30 do not.
    assert once == ("`archive/OLD_PLAN.md` §6; archive/BIG-start.md and "
                    "`archive/BIG-start.md` §1.5 and BIG.md §10; "
                    "`archive/PLAN-done.md` §B3, archive/PLAN-done.md §B3, "
                    "archive/PLAN-done.md item B3, archive/PLAN-done.md items B3 and B4, "
                    "archive/PLAN-done.md §B3, PLAN.md §B30\n")  # bare id gains §
    assert n == 8
    assert rw.rewrite(once, rules) == (once, 0)
    assert rw.absent_table(tmp_path) == {"NEVER.md"}


def test_a_list_or_a_line_break_is_not_rewritten_blind(tmp_path):
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "MOVED.md").write_text(MOVED, encoding="utf-8")
    rules = rw.build_rules(rw.moved_table(tmp_path), live=["doc.md"], root=tmp_path)
    # B6 did not move: rewriting "items B3 and B6" would send B6 to the archive.
    for text in ("PLAN.md items B3 and B6\n", "PLAN.md §B3, B6\n",
                 "see PLAN.md\nB3 is next\n", "PLAN.mdB3\n"):
        assert rw.rewrite(text, rules) == (text, 0), text
    # A § after a wrapped line is still a citation.
    assert rw.rewrite("`PLAN.md`\n§B3\n", rules)[1] == 1


def test_a_recreated_old_path_is_skipped_and_flagged(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "SCAN.md").write_text("new file\n", encoding="utf-8")
    msgs = _tree(tmp_path, monkeypatch, "see `docs/SCAN.md`\n")
    assert any("exists again" in m for m in msgs)
    rules = rw.build_rules(rw.moved_table(tmp_path), live=["doc.md"], root=tmp_path)
    assert rw.rewrite("see `docs/SCAN.md`\n", rules) == ("see `docs/SCAN.md`\n", 0)


def test_the_real_tree_has_nothing_left_to_rewrite():
    # A live file that still cites an old path is either a miss or prose that
    # a future batch would mangle; both are fixed by rewording, not by --write.
    live = rw.live_files()
    rules = rw.build_rules(rw.moved_table(), live)
    pending = [f for f in live if (rw.ROOT / f).is_file()
               and rw.rewrite((rw.ROOT / f).read_text(encoding="utf-8"), rules)[1]]
    assert pending == []
