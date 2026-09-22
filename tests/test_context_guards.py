"""
tests/test_context_guards.py -- the large-read hook and the doc indexes.

Both must be able to fail: the hook must deny an unbounded read of a large
document (not only allow small ones), and `--check` must catch a stale index
(`LESSONS.md` lesson 2: an instrument that cannot fail is not an instrument).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure

REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / "scripts" / "hooks" / "guard_large_read.py"
sys.path.insert(0, str(REPO / "tools"))
import build_doc_index  # noqa: E402


def _hook(tool_input, tool_name="Read", project_dir=None, raw=None):
    env = {"PATH": "/usr/bin:/bin"}
    if project_dir is not None:
        env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    stdin = raw if raw is not None else json.dumps(
        {"tool_name": tool_name, "tool_input": tool_input, "hook_event_name": "PreToolUse"})
    r = subprocess.run([sys.executable, str(HOOK)], input=stdin, text=True,
                       capture_output=True, env=env, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout) if r.stdout.strip() else None


@pytest.fixture
def big_md(tmp_path):
    p = tmp_path / "BIG.md"
    p.write_text("## a\n" + "x" * 80 + "\n" * 1 + ("line\n" * 20_000), encoding="utf-8")
    return p


def test_denies_unbounded_read_of_large_doc(big_md):
    out = _hook({"file_path": str(big_md)})
    d = out["hookSpecificOutput"]
    assert d["permissionDecision"] == "deny"
    assert "limit" in d["permissionDecisionReason"]


def test_denies_oversized_limit(big_md):
    assert _hook({"file_path": str(big_md), "offset": 1, "limit": 2000}) is not None


def test_allows_bounded_read(big_md):
    assert _hook({"file_path": str(big_md), "offset": 100, "limit": 300}) is None


def test_allows_small_doc_and_code(tmp_path, big_md):
    small = tmp_path / "small.md"
    small.write_text("# hi\n", encoding="utf-8")
    code = tmp_path / "big.py"
    code.write_text("x = 1\n" * 20_000, encoding="utf-8")
    assert _hook({"file_path": str(small)}) is None
    assert _hook({"file_path": str(code)}) is None


def test_fails_open_on_missing_file_other_tool_and_bad_json(tmp_path, big_md):
    assert _hook({"file_path": str(tmp_path / "nope.md")}) is None
    assert _hook({"file_path": str(big_md)}, tool_name="Grep") is None
    assert _hook({}, raw="not json") is None


def test_points_to_index_when_one_exists(tmp_path, big_md):
    (tmp_path / "docs" / "index").mkdir(parents=True)
    (tmp_path / "docs" / "index" / "BIG.idx.md").write_text("x", encoding="utf-8")
    reason = _hook({"file_path": str(big_md)}, project_dir=tmp_path)[
        "hookSpecificOutput"]["permissionDecisionReason"]
    assert "BIG.idx.md" in reason


def test_sections_tile_and_skip_fences():
    text = "intro\n## One\na\n```\n## not a heading\n```\n### Two\nb\nc\n## Three\n"
    secs = build_doc_index.sections(text)
    assert [(lvl, start, n) for lvl, start, n, _, _ in secs] == [(2, 2, 5), (3, 7, 3), (2, 10, 1)]


def test_bold_heads_only_inside_long_sections(monkeypatch):
    monkeypatch.setattr(build_doc_index, "LONG_SECTION", 5)
    long = "## Long\n**A. first**\n" + "x\n" * 5 + "**B. second**\ny\n"
    short = "## Short\n**C. ignored**\n"
    titles = [t for *_, t in build_doc_index.sections(long + short)]
    assert titles == ["Long", "**A.** first", "**B.** second", "Short"]


def test_check_catches_stale_index(tmp_path, monkeypatch):
    monkeypatch.setattr(build_doc_index, "ROOT", tmp_path)
    monkeypatch.setattr(build_doc_index, "OUT_DIR", tmp_path / "docs" / "index")
    monkeypatch.setattr(build_doc_index, "DOCS", ["D.md"])
    (tmp_path / "D.md").write_text("## a\nx\n", encoding="utf-8")
    assert build_doc_index.main([]) == 0
    assert build_doc_index.main(["--check"]) == 0
    (tmp_path / "D.md").write_text("## a\nx\n## b\n", encoding="utf-8")
    assert build_doc_index.main(["--check"]) == 1


def test_committed_indexes_are_current():
    assert build_doc_index.main(["--check"]) == 0
