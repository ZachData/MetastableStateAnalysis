"""
tests/test_no_hardcoded_repo.py — no module defaults METS_REPO to the main tree.

47 runners defaulted `REPO` to `/run/media/system/WDS_500/Mets` and then did
`sys.path.insert(0, REPO)` at import time. Collecting a test that imported one
(`tests/test_backfill_hdbscan.py` -> `tools/run/backfill_hdbscan.py`) put the
main tree first on sys.path, so a gate run from a worktree imported **main's**
copy of every package not yet loaded, and tested it instead of the branch
(found 2026-09-24 when `tests/test_run_2d.py` ran against main's `run_2d.py`;
`docs/PHASE_REVIEW.md` Parked 2). A default is now this checkout's own root.
"""

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure

ROOT = Path(__file__).resolve().parents[1]
MAIN_TREE = "/run/media/system/WDS_500/Mets"


def _tracked_py():
    try:
        out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip("not a git checkout")
    return [ROOT / f for f in out.split()
            if not f.startswith(("archive/", "data/"))]


def test_no_live_module_hardcodes_the_main_tree():
    hits = [str(p.relative_to(ROOT)) for p in _tracked_py()
            if p.exists() and f'"{MAIN_TREE}"' in p.read_text(errors="replace")]
    assert not hits, f"hard-coded main-tree path in: {hits}"


def test_the_gate_imports_this_checkout():
    import p2d_operator_activation
    assert Path(p2d_operator_activation.__file__).resolve().is_relative_to(ROOT)
