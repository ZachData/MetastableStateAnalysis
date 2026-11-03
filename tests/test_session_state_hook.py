"""The SessionStart hook prints origin/main's STATE.md, not the tree's (LESSONS 1)."""
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure  # git and bash only

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "session_state.sh"


def git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repos(tmp_path, monkeypatch):
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    origin, up, tree = tmp_path / "origin.git", tmp_path / "up", tmp_path / "tree"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    git(tmp_path, "clone", "-q", str(origin), str(up))
    (up / "STATE.md").write_text("old state\n")
    git(up, "add", "STATE.md"); git(up, "commit", "-qm", "old"); git(up, "push", "-q", "origin", "HEAD:main")
    git(tmp_path, "clone", "-q", str(origin), str(tree))
    (up / "STATE.md").write_text("new state\n")
    git(up, "commit", "-qam", "new"); git(up, "push", "-q", "origin", "HEAD:main")
    git(up, "switch", "-qc", "pr")
    (up / "STATE.md").write_text("branch state\n")
    git(up, "commit", "-qam", "pr state"); git(up, "push", "-q", "origin", "pr")
    git(up, "switch", "-qc", "other", "main")
    (up / "x.txt").write_text("x\n")
    git(up, "add", "x.txt"); git(up, "commit", "-qm", "no state"); git(up, "push", "-q", "origin", "other")
    return tree


def run(tree):
    env = dict(os.environ, CLAUDE_PROJECT_DIR=str(tree))
    return subprocess.run(["bash", str(HOOK)], env=env, capture_output=True, text=True, check=True).stdout


def test_prints_origin_main_state_with_provenance_first(repos):
    out = run(repos)
    head, _, body = out.partition("\n\n")
    assert body.strip() == "new state"          # origin/main's, not the tree's "old state"
    assert "fetched now" in head
    assert "1 commits behind origin/main" in head
    assert "origin/pr changed STATE.md" in head  # unmerged branch that changed STATE
    assert "origin/other" not in head            # unmerged, but STATE untouched


def test_falls_back_to_tree_without_origin(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "STATE.md").write_text("local only\n")
    out = run(tmp_path)
    assert "FETCH FAILED" in out and "the working tree" in out
    assert out.rstrip().endswith("local only")
