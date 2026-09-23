"""
tests/test_tools_render_phases.py — tools/render_phases.py and lint rule
`phase-card`.

Every check must be seen to fire on a card built to fail it, not only to pass
on the real tree; a check that cannot fail is lesson 2 in LESSONS.md.
"""
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
import lint_repo  # noqa: E402
import render_phases as rp  # noqa: E402

CARD = """\
<!-- phase-card -->
## Card

- **Question:** Do particles cluster?
- **Inputs:** none
- **Results:**
  - They do — `status-{id}.md` "Findings", and §3.41
  - Paths under data are unchecked — `data/phase12/run/geometry.json`
- **Superseded / wrong:** none
- **Registry:** none, because tier 1
- **Depends on:** {deps}
- **Feeds:** {feeds}
- **Open threads:**
  - why
- **After Phase 10:**
  - rerun it (forward pass, 410m)
- **Reviewed:** pending
<!-- /phase-card -->
"""


def _status(pid, deps="none", feeds="none", card=CARD):
    return (f"# Phase {pid} — STATUS\n\n" + card.format(id=pid, deps=deps, feeds=feeds)
            + "\n## Findings\n\nbody text\n")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "docs").mkdir()
    shutil.copy(TOOLS.parent / rp.TEMPLATE, tmp_path / rp.TEMPLATE)
    (tmp_path / "PROJECT.md").write_text("# P\n\n### 3.41 A section\n", encoding="utf-8")
    for pid, deps, feeds in (("1", "none", "2"), ("2", "1", "none")):
        d = tmp_path / f"p{pid}"
        d.mkdir()
        (d / f"status-{pid}.md").write_text(_status(pid, deps, feeds), encoding="utf-8")
    for pid in ("1", "2"):
        rp.stamp(pid, tmp_path, today="2026-09-23")
    return tmp_path


def _msgs(root):
    return [m for _, _, m in rp.card_findings(root)]


def _edit(root, pid, old, new):
    p = root / f"p{pid}" / f"status-{pid}.md"
    text = p.read_text(encoding="utf-8")
    assert old in text
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_the_real_tree_passes_and_the_table_is_in_step():
    assert rp.card_findings() == []
    assert (rp.ROOT / rp.OUT).read_text(encoding="utf-8") == rp.render()


def test_the_lint_rule_reports_card_findings(repo, monkeypatch):
    monkeypatch.setattr(lint_repo, "ROOT", repo)
    _edit(repo, "1", "Do particles cluster?", "TODO")
    lint = lint_repo.Linter()
    lint_repo.rule_phase_cards(lint)
    assert any("TODO" in f.message for f in lint.findings)


def test_a_fresh_stamped_pair_passes(repo):
    assert _msgs(repo) == []


def test_a_missing_field_is_refused(repo):
    _edit(repo, "1", "- **Registry:** none, because tier 1\n", "")
    assert any("missing field(s) ['Registry']" in m for m in _msgs(repo))


def test_an_unknown_field_is_refused(repo):
    _edit(repo, "1", "- **Registry:**", "- **Registery:**")
    assert any("unknown field" in m for m in _msgs(repo))


def test_a_todo_is_refused(repo):
    _edit(repo, "1", "  - why", "  - TODO")
    assert any("'Open threads' still has a TODO" in m for m in _msgs(repo))


def test_a_result_without_a_pointer_is_refused(repo):
    _edit(repo, "1", "  - They do — `status-1.md` \"Findings\", and §3.41", "  - They do")
    assert any("has no pointer" in m for m in _msgs(repo))


@pytest.mark.parametrize("bad, expect", [
    ("`nowhere.md`", "`nowhere.md` (no such file)"),
    ("§9.99", "§9.99 (no such heading in PROJECT.md)"),
    ("`status-1.md` \"Nope\"", "\"Nope\" (no such heading)"),
    ("`status-1.md` §7", "§7 (no such heading)"),
])
def test_an_unresolved_pointer_is_refused(repo, bad, expect):
    _edit(repo, "1", "and §3.41", f"and {bad}")
    assert any(expect in m for m in _msgs(repo))


def test_a_cost_free_after_phase_10_item_is_refused(repo):
    _edit(repo, "1", "rerun it (forward pass, 410m)", "rerun it")
    assert any("has no cost" in m for m in _msgs(repo))


def test_an_edit_outside_the_card_makes_it_stale(repo):
    _edit(repo, "2", "body text", "a new result")
    msgs = _msgs(repo)
    assert any("STALE: status-2.md changed" in m for m in msgs)


def test_an_edit_inside_the_card_does_not(repo):
    _edit(repo, "2", "Do particles cluster?", "Do particles cluster, and when?")
    assert _msgs(repo) == []


def test_a_dependency_edit_makes_the_dependent_stale(repo):
    _edit(repo, "1", "body text", "a new result")
    msgs = _msgs(repo)
    assert any("STALE: phase 1" in m for m in msgs)       # on phase 2's card
    rp.stamp("1", repo)
    rp.stamp("2", repo)
    assert _msgs(repo) == []


def test_feeds_and_depends_on_must_agree(repo):
    _edit(repo, "1", "- **Feeds:** 2", "- **Feeds:** none")
    assert any("disagree about 1 → 2" in m for m in _msgs(repo))


def test_a_feed_to_an_unknown_phase_is_refused(repo):
    _edit(repo, "1", "- **Feeds:** 2", "- **Feeds:** 2, 42")
    assert any("no phase '42'" in m for m in _msgs(repo))


def test_an_unstamped_card_is_refused(repo):
    _edit(repo, "1", "- **Reviewed:** 2026-09-23", "- **Reviewed:** 2026-09-2")
    assert any("--stamp 1" in m for m in _msgs(repo))


def test_a_frozen_copy_gets_its_own_id(repo):
    d = repo / "archive" / "p1"
    d.mkdir(parents=True)
    (d / "status-1.md").write_text("# frozen\n", encoding="utf-8")
    assert "1-frozen" in rp.discover(repo)
    assert "| 1-frozen |" in rp.render(repo)


@pytest.mark.parametrize("heading, sec", [
    ("### 9.1 Notes from 2026-09-12", "12"),     # a number inside a date
    ("## The `7d` line", "7"),                   # a phase name in a code span
    ("### 3.41 A section", "3"),                 # a parent that does not exist
])
def test_a_section_number_must_open_its_heading(repo, heading, sec):
    (repo / "PROJECT.md").write_text(f"# P\n\n### 3.41 A section\n\n{heading}\n",
                                     encoding="utf-8")
    _edit(repo, "1", "and §3.41", f"and §{sec}")
    assert any(f"§{sec} (no such heading" in m for m in _msgs(repo))


def test_a_section_after_an_unbackticked_file_is_refused(repo):
    _edit(repo, "1", "and §3.41", "and MATH.md §3.41")
    assert any("unbackticked file name" in m for m in _msgs(repo))


def test_stamp_replaces_a_field_wrapped_over_lines(repo):
    _edit(repo, "2", "- **Depends on:** 1@", "- **Depends on:** 1,\n  1@")
    rp.stamp("2", repo)
    text = (repo / "p2" / "status-2.md").read_text(encoding="utf-8")
    assert "\n  1@" not in text
    assert _msgs(repo) == []


def test_a_template_with_wrong_fields_is_refused(repo):
    t = repo / rp.TEMPLATE
    t.write_text(t.read_text(encoding="utf-8").replace("- **Feeds:** none\n", ""),
                 encoding="utf-8")
    assert any("missing field(s) ['Feeds']" in m for m in _msgs(repo))
