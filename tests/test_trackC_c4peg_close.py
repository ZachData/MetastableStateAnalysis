"""C4′ (e)+(g) close-out (`trackC_writeup/c4peg_close.py`): the partial rank correlation."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackC_writeup import c4peg_close as c


def test_partial_rank_removes_the_shared_prior():
    rng = np.random.default_rng(3)
    ell = rng.normal(size=5000)
    z1, z2 = ell + 0.5 * rng.normal(size=5000), ell + 0.5 * rng.normal(size=5000)
    assert abs(c.partial_rank(z1, z2, ell)) < 0.05          # agree only through ell
    w = rng.normal(size=5000)
    assert c.partial_rank(ell + w, ell + w, ell) == pytest.approx(1.0)
    assert c.partial_rank(ell + w + 0.1 * rng.normal(size=5000), ell + w, ell) > 0.9


def test_share_removed():
    assert c.share_removed(8.0, 0.0) == 1.0 and c.share_removed(8.0, 8.0) == 0.0
    assert c.share_removed(8.0, 10.0) < 0          # an ablation that helps copying
