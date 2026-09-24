"""
tests/test_no_hardcoded_repo.py — no module defaults METS_REPO to the main tree.

47 runners defaulted `REPO` to the main tree's absolute path and then did
`sys.path.insert(0, REPO)` at import time. Collecting a test that imported one
(`tests/test_backfill_hdbscan.py` -> `tools/run/backfill_hdbscan.py`) put the
main tree first on sys.path, so a gate run from a worktree imported **main's**
copy of every package not yet loaded, and tested it instead of the branch
(found 2026-09-24 when `tests/test_run_2d.py` ran against main's `run_2d.py`;
`docs/PHASE_REVIEW.md` Parked 2). A default is now this checkout's own root.
`tests/conftest.py` also fails the session if any project module was loaded
from outside this checkout, which covers an exported `METS_REPO` too.
"""

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure

ROOT = Path(__file__).resolve().parents[1]
# Assembled, so this file does not contain the literal it searches for.
MAIN_TREE = "/run/media/system/" + "WDS_500/Mets"
# The path as a default, in any quoting: "…", '…', or a shell `${VAR:-…}`,
# ending at the tree itself (so `…/Mets/data` in a docstring is not a hit).
PATTERN = re.compile(r"""["'=:-]""" + re.escape(MAIN_TREE) + r"""(?=["'}\s/]?$|["'}])""",
                     re.MULTILINE)


def _tracked(globs):
    try:
        out = subprocess.run(["git", "ls-files", *globs], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip("not a git checkout")
    # archive/ is frozen; data/ is never `git add`ed by Claude (CLAUDE.md), so
    # data/analysis/ scripts are listed for the user, not checked here.
    return [ROOT / f for f in out.split()
            if not f.startswith(("archive/", "data/"))]


def test_no_live_code_defaults_to_the_main_tree():
    hits = []
    for p in _tracked(["*.py", "*.sh"]):
        if p.name == Path(__file__).name or not p.exists():
            continue
        for m in PATTERN.finditer(p.read_text(errors="replace")):
            line = p.read_text(errors="replace")[:m.start()].count("\n") + 1
            hits.append(f"{p.relative_to(ROOT)}:{line}")
    assert not hits, f"main-tree path as a default in: {hits}"


def test_the_gate_imports_this_checkout():
    import p2d_operator_activation
    assert Path(p2d_operator_activation.__file__).resolve().is_relative_to(ROOT)
