"""
tests/test_lint_startup_docs.py — tools/lint_repo.py rule 6, startup-doc-cap.

The rule must fire on an oversized or undated STATE.md, not only pass on the
real one; a cap that cannot fail is lesson 2 in LESSONS.md.
"""
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import lint_repo  # noqa: E402


def _run(tmp_path, monkeypatch, state: str | None, claude: str = "x\n"):
    monkeypatch.setattr(lint_repo, "ROOT", tmp_path)
    if state is not None:
        (tmp_path / "STATE.md").write_text(state, encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text(claude, encoding="utf-8")
    lint = lint_repo.Linter()
    lint_repo.rule_startup_doc_caps(lint)
    return [f.message for f in lint.findings]


GOOD = "# STATE\n\n**Last updated:** 2026-09-22 · cap\n"


def test_the_real_startup_docs_pass():
    lint = lint_repo.Linter()
    lint_repo.rule_startup_doc_caps(lint)
    assert lint.findings == []


def test_a_small_dated_state_passes(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, GOOD) == []


def test_an_oversized_state_is_refused(tmp_path, monkeypatch):
    msgs = _run(tmp_path, monkeypatch, GOOD + "line\n" * 200)
    assert any("cap 150" in m for m in msgs)


def test_an_undated_state_is_refused(tmp_path, monkeypatch):
    msgs = _run(tmp_path, monkeypatch, "# STATE\n")
    assert any("Last updated" in m for m in msgs)


def test_a_missing_state_is_refused(tmp_path, monkeypatch):
    msgs = _run(tmp_path, monkeypatch, None)
    assert any("missing" in m for m in msgs)


def test_an_oversized_claude_md_is_refused(tmp_path, monkeypatch):
    msgs = _run(tmp_path, monkeypatch, GOOD, claude="x\n" * 121)
    assert any("cap 120" in m for m in msgs)
