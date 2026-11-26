"""C9 U3 (`trackC_writeup/c9_u3.py`, `c9_u3_pass.py`): repos, arms and the gate on seed 0's series."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackA_force.a1a import STEPS
from trackC_writeup import c9_u3 as u
from trackC_writeup import c9_u3_pass as p


def test_repos_and_hubs():
    assert p.repo_for(0) == "EleutherAI/pythia-160m"
    assert p.repo_for(7) == "EleutherAI/pythia-160m-seed7"
    assert p.hub_for(0) != p.hub_for(1) == p.hub_for(9)


def test_arm_keys_are_3b_reading_keys():
    assert [u.arm_key(t) for t in u.ARMS] == ["0.000125", "0.0001", "0.0003", "0.000822"]


def _step(at, plateau=16.0):
    return np.array([0.0 if s < at else plateau for s in STEPS])


def test_gate_passes_on_a_clean_series_at_160m_dates():
    vu = _step(256)
    vcs = {u.arm_key(t): _step(1000, 40.0) for t in u.ARMS}
    g = u.gate(vu, vcs)
    assert g["pass"], g
    t = {r["planted"]: r["inside"] for r in g["planted_null"]["table"]}
    assert t[256] and t[512] and not t[128] and not t[1000]


def test_gate_refuses_a_centred_arm_that_cannot_move():
    vu = _step(256)
    vcs = {u.arm_key(t): _step(1000, 40.0) for t in u.ARMS}
    vcs["0.0001"] = np.full(len(STEPS), 40.0)          # unreadable: early counts ≥ half the plateau
    assert not u.gate(vu, vcs)["pass"]
