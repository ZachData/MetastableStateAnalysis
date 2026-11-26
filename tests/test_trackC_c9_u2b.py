"""C9 U2b (`trackC_writeup/c9_u2b.py`): snapping (D8 (a)), the snapped arm's fixed plant (D7 (a)),
the pairs on snapped indices, and the max-over-heads statistic (D2 (a))."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackA_force import a3
from trackC_writeup import c9_u2 as c
from trackC_writeup import c9_u2b as b


def test_snap_by_tokens_on_gpt2_and_identity_on_pythia():
    for s in a3.STEPS:
        assert b.snap("410m", s) == s
    assert b.snap("alias", 0) == 0
    assert b.snap("alias", 250) == 64 and b.snap("alias", 350) == 128     # E₁'s 3c dates
    assert b.snap("alias", 1000) == 256 and b.snap("alias", 1100) == 512
    assert b.snap("alias", 16000) == 4000                                 # the window's end
    assert b.snap("A0", 384) == 512 and b.snap("A0", 640) == 1000 and b.snap("A0", 1250) == 2000


def test_snapped_grid_is_monotone_and_onto_a_prefix():
    for run in ("alias", "A0"):
        g = c.grid_of(run)
        sn = [b.snap(run, s) for s in g.steps if s <= g.last]
        assert sn == sorted(sn) and sn[-1] == 4000


def test_fixed_plant_is_a3s_on_pythia():
    for i in a3.WIN[1:-1]:
        assert np.allclose(b.shape_fixed(c.PYTHIA, i), a3._shape(i))
        assert np.allclose(b.dip_fixed(c.PYTHIA, i), a3._dip(i))


def test_maxmean():
    x = np.zeros((3, 4, 5))
    x[:, 2, 1] = [1, 2, 3]
    x[0, 2, 4] = 5
    assert np.allclose(b.stat_maxmean(x), [0, 0, 2, 0])
    assert b.STATS["maxmean"] is b.stat_maxmean


def test_pairs_read_on_snapped_indices():
    rows = {"E1": {"reading": "located", "step": 384, "censored": False},
            "E3": {"reading": "located", "step": 768, "censored": False},
            "E5": {"reading": "located", "step": 8, "censored": False}}
    steps = list(c.HARNESS.steps)
    p = b.read_pair("A0", rows, "E1", "E3", steps)            # 512 and 1000: one Pythia step
    assert p["verdict"] == "aligned" and p["own_grid_distance"] == [3] and p["finer_ordered"]
    q = b.read_pair("A0", rows, "E1", "E5", steps)
    assert q["verdict"] == "misaligned" and q["first"] == "E5"
    r = b.read_pair("A0", rows, "E1", "E7", steps)
    assert r["verdict"] == "unread"
    assert b.same(p, {"verdict": "aligned"}) is True
    assert b.same(q, {"verdict": "misaligned", "first": "E1"}) is False
    assert b.same(r, {"verdict": "aligned"}) is None


def test_second_dates_must_agree():
    rows = {"E1": {"reading": "located", "step": 512, "censored": False},
            "E8": {"reading": "located", "step": 1000, "censored": False},
            "E8_peak": {"reading": "located", "step": 128, "censored": False}}
    p = b.read_pair("410m", rows, "E1", "E8", list(a3.STEPS))
    assert p["verdict"] == "unread (two dates)"
