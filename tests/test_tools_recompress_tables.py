"""
tests/test_tools_recompress_tables.py — tools/recompress_tables.py's batch.

Pure numpy. The rewrite itself is verified in-process by the tool (it reloads
and compares every array before `os.replace`), so what is left to test is the
BATCH's behaviour around a run that did not finish:

  1. An interrupted run leaves `<name>.npz.recompress-tmp.npz` beside the
     original. That name matches `*.npz`. On 2026-09-03 the next run picked it
     up as input, consumed it via the original's own `os.replace`, and then
     died on `stat()` of a path that no longer existed -- aborting the batch
     with four files still to go and no record of which.
  2. A file that disappears between listing and rewriting must be reported and
     skipped, not allowed to end the run. Every later file is otherwise lost
     for a reason unrelated to it.

Both are about a batch that must not lose the files it has not reached yet.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.recompress_tables import TMP_SUFFIX, collect, main, recompress

# Tier: pure -- numpy and pytest only.
pytestmark = pytest.mark.pure


def _uncompressed(path, n=64):
    np.savez(path, a=np.arange(n, dtype=np.int64),
             b=np.full(n, np.nan, dtype=np.float64))


def test_collect_excludes_the_tools_own_leftover_temp(tmp_path):
    """The exact filename an interrupted run leaves behind is not input."""
    real = tmp_path / "interaction_table.npz"
    _uncompressed(real)
    leftover = tmp_path / ("interaction_table.npz" + TMP_SUFFIX + ".npz")
    _uncompressed(leftover)

    got = collect([tmp_path])

    assert real in got
    assert leftover not in got, (
        "a leftover .recompress-tmp is the tool's own in-flight file; taking "
        "it as input is what consumed it and killed the 2026-09-03 batch")


def test_a_leftover_temp_does_not_stop_the_batch(tmp_path, capsys):
    """The regression in full: a leftover beside the FIRST of two tables must
    not cost the second one its rewrite."""
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    _uncompressed(first)
    _uncompressed(second)
    (tmp_path / ("a.npz" + TMP_SUFFIX + ".npz")).write_bytes(
        first.read_bytes())

    assert main([str(tmp_path)]) == 0

    # Both real tables were rewritten; neither is still stored-not-deflated.
    import zipfile
    for f in (first, second):
        with zipfile.ZipFile(f) as z:
            assert all(i.compress_type == zipfile.ZIP_DEFLATED
                       for i in z.infolist()), f"{f.name} was never rewritten"


def test_a_file_that_vanished_is_reported_not_raised(tmp_path):
    """`recompress` is called on a path listed earlier; it may be gone."""
    gone = tmp_path / "gone.npz"

    r = recompress(gone)

    assert r["status"] == "vanished before rewrite"
    assert r["after"] is None


def test_nan_survives_the_rewrite(tmp_path):
    """real_frac/imag_frac are all-NaN in every table this tool touches, and
    NaN -> 0.0 would collapse "not measured" into "measured zero"."""
    p = tmp_path / "t.npz"
    _uncompressed(p)

    assert recompress(p)["status"] == "done"

    back = np.load(p)
    assert np.isnan(back["b"]).all()
    assert np.array_equal(back["a"], np.arange(64, dtype=np.int64))
