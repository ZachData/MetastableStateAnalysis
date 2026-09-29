"""
tests/test_lint_pyproject_packages.py — tools/lint_repo.py's `pyproject-packages`
rule and the derived `live_packages()` the other rules now walk.

The rule must fire in both directions, not only pass on the real tree
(LESSONS.md lesson 2): a live package missing from pyproject.toml (what
happened with p1d_cluster_ensemble), and a declared name with no __init__.py
(archived phases left behind).
"""
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import lint_repo  # noqa: E402


def _pyproject(names, extra=""):
    body = "".join(f'    "{n}",\n' for n in names)
    return ("[project]\nname = \"x\"\n\n[tool.setuptools]\n"
            f"packages = [\n{body}{extra}]\n")


def _tree(tmp_path, monkeypatch, live, declared, extra=""):
    monkeypatch.setattr(lint_repo, "ROOT", tmp_path)
    for name in live:
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(_pyproject(declared, extra),
                                             encoding="utf-8")
    lint = lint_repo.Linter()
    lint_repo.rule_pyproject_packages(lint)
    return [f.message for f in lint.findings]


def test_the_real_tree_passes():
    lint = lint_repo.Linter()
    lint_repo.rule_pyproject_packages(lint)
    assert [f.render() for f in lint.findings] == []


def test_matching_lists_pass(tmp_path, monkeypatch):
    assert _tree(tmp_path, monkeypatch, ["core", "p1"], ["core", "p1"]) == []


def test_live_package_missing_from_pyproject_fails(tmp_path, monkeypatch):
    msgs = _tree(tmp_path, monkeypatch, ["core", "p1d_new"], ["core"])
    assert len(msgs) == 1 and "'p1d_new'" in msgs[0] and "not in packages" in msgs[0]


def test_declared_name_without_init_fails(tmp_path, monkeypatch):
    (tmp_path / "p5_gone").mkdir()                   # math-*.md only, no __init__
    msgs = _tree(tmp_path, monkeypatch, ["core"], ["core", "p5_gone"])
    assert len(msgs) == 1 and "'p5_gone'" in msgs[0] and "not a live package" in msgs[0]


def test_commented_out_entry_is_not_declared(tmp_path, monkeypatch):
    msgs = _tree(tmp_path, monkeypatch, ["core", "p2"], ["core"],
                 extra='    # "p2",\n')
    assert len(msgs) == 1 and "'p2'" in msgs[0]


def test_archive_and_nested_packages_are_not_live(tmp_path, monkeypatch):
    monkeypatch.setattr(lint_repo, "ROOT", tmp_path)
    for rel in ("archive", "archive/p3_old", "core", "core/sub"):
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)
        (tmp_path / rel / "__init__.py").write_text("", encoding="utf-8")
    assert lint_repo.live_packages() == ("core",)


def test_the_rules_walk_every_live_package():
    # PACKAGE_DIRS was a hand-kept tuple that stopped at p7_motifs, so the
    # other rules never read p1d, p7d, p7e or p8.
    assert set(lint_repo.live_packages()) <= set(lint_repo.PACKAGE_DIRS)
    assert "p1d_cluster_ensemble" in lint_repo.PACKAGE_DIRS
