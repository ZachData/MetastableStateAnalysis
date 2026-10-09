"""
`p1e_energy_field/u3_saddles.py`'s band (needs torch): on two mirror-image groups the climbing
image finds the pass on the mirror plane, as brute force does (`design-1e.md` "U3 at β 10", tests).
"""
import pytest

from p1e_energy_field import u1_field as u1
from p1e_energy_field import u3_saddles as u3
from test_p1e_u3_saddles import _pair, _plane_pass

pytestmark = pytest.mark.deps


def test_neb_finds_the_symmetric_pass():
    pytest.importorskip("torch")
    X, _ = _pair()
    ms = u1.mean_shift(X, u3.BETA, modes=True)
    m = ms["modes"]
    t = u3.merge_tree(X, ms["wells"], u3.heights(m, X, u3.BETA), u3.BETA)
    (dd,) = t["deaths"]
    r = u3.neb(m[dd["well"]], m[dd["into"]], [X[dd["i"]], X[dd["j"]]], X, u3.BETA)
    assert r["converged"]
    assert r["h"] == pytest.approx(_plane_pass(X, m), abs=1e-4)   # 12.8872 here, both ways
    assert r["h"] >= dd["h"]
