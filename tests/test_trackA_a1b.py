"""A1b's rule and its per-step vote (`trackA_force/design-A.md` "A1b, the reading")."""

import numpy as np
import pytest

from trackA_force import a1b

pytestmark = pytest.mark.deps  # numpy, scipy (tstar_table)


def test_cell_rule():
    assert a1b.read_cell(1.0, 2.0, 3.0) == "below"
    assert a1b.read_cell(2.5, 2.0, 3.0) == "straddles"
    assert a1b.read_cell(2.0, 2.0, 3.0) == "straddles"
    assert a1b.read_cell(3.0, 2.0, 3.0) == "not below"
    assert a1b.read_cell(1.0, None, 3.0) == "not read"
    assert a1b.read_cell(1.0, 2.0, float("inf")) == "not read"
    with pytest.raises(ValueError):
        a1b.read_cell(1.0, 3.0, 2.0)


def test_step_vote():
    assert a1b.read_step(["below"] * 6 + ["straddles"]) == "below"
    assert a1b.read_step(["below"] * 5 + ["straddles"] * 2) == "mixed"
    assert a1b.read_step(["not read"] * 7) == "not read"


def test_beta_doubling_brackets_the_table():
    # design-A.md's (6.9) table, g0 = 0, n = 467: t* 4.17 at β 0, 8.70 at 7
    b = a1b.beta_doubling(467, 0.0)
    assert 6.5 < b < 7.5


def _records(planted: bool, seed=0):
    """Per step a different T profile; mean cosine = f(cumulative T) if planted, else f(depth)."""
    rng = np.random.default_rng(seed)
    recs = []
    for i, s in enumerate(a1b.STEPS):
        scale = 0.02 + 0.3 * i / len(a1b.STEPS)
        for p in a1b.V1:
            proj = scale * rng.uniform(0.5, 1.5, 24)
            cum = np.concatenate([[0.0], np.cumsum(proj)[:-1]])
            cos = 1 - np.exp(-cum) if planted else 1 - np.exp(-0.1 * np.arange(24))
            recs.append({"step": s, "prompt": p,
                         "blocks": [{"block": l, "proj": float(proj[l]), "mean_cos": float(cos[l])}
                                    for l in range(24)]})
    return recs


def test_hwarp_planted_warp_collapses():
    w = a1b.hwarp(_records(planted=True))
    assert w["collapses"] and w["steps_pass"] >= 16


def test_hwarp_depth_curves_refuse():
    w = a1b.hwarp(_records(planted=False))
    assert not w["collapses"]
