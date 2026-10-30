"""T5 U2, the trajectories (`trackT_timaeus/t5c.py`, design-T5.md "U2", "U2's details"): the frozen
lists and the "spreads" rule (pure)."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports, transformers
from trackT_timaeus import t2, t5c


def test_the_frozen_lists():
    assert len(t5c.DODECA_STEPS) == 53 and t5c.DODECA_STEPS[0] == 1 and t5c.DODECA_STEPS[-1] == 9536
    g = sorted({int(round(x / 100)) * 100 for x in np.geomspace(100, 65000, 20)})
    assert t5c.GPT2_STEPS == (0,) + tuple(g) and len(g) == 19
    assert t5c.TRAJ["pythia-410m"][1][0] == 0 and len(t5c.TRAJ["pythia-410m"][1]) == 18
    assert t5c.SCORED == ("dodecahedron-32m", "dodecahedron-67m", "gpt-2-small")
    assert (t5c.SPREAD_MIN, t5c.BOOT_P5_MIN, t5c.N_BOOT, t5c.KNOWN_TOL) == (0.15, 0.05, 1000, 1e-6)
    assert t2.SCALE_STEP == t5c.SPREAD_STEP


def test_label_order():
    assert t5c.label(0.15, 0.06, 0.30) == "spreads"
    assert t5c.label(0.15, 0.05, 0.30) == "not told apart"            # 5th percentile must exceed 0.05
    assert t5c.label(0.10, 0.02, 0.149) == "does not spread"
    assert t5c.label(0.10, 0.02, 0.15) == "not told apart"
    assert t5c.label(0.20, 0.10, 0.14) == "spreads"                   # spreads is checked first


def test_mid_series_is_prompt_mean_then_median_of_middle_states():
    P = np.zeros((2, 3, 5))
    P[0, :, 1:4] = [[0.1, 0.5, 0.9], [0.3, 0.5, 0.7], [0.2, 0.5, 0.2]]
    P[0, :, 0] = P[0, :, 4] = 9.0                                   # embeddings and final state ignored
    assert abs(t5c.mid_series(P)[0] - np.median([0.2, 0.5, 0.6])) < 1e-15


def test_spread_on_noiseless_series():
    steps = 6
    X = np.tile(np.array([0.9, 0.8, 0.7, 0.6, 0.65, 0.7])[:, None], (1, 7))
    r = t5c.spread(t5c.scalar_as_states(X), np.random.default_rng(0), n_boot=50)
    assert abs(r["delta"] - 0.3) < 1e-12 and r["argmin"] == 3 and r["label"] == "spreads"
    assert abs(r["p5"] - 0.3) < 1e-12 and abs(r["p95"] - 0.3) < 1e-12
    flat = t5c.spread(t5c.scalar_as_states(np.full((steps, 7), 0.9)), np.random.default_rng(0), n_boot=50)
    assert flat["delta"] == 0 and flat["label"] == "does not spread"


def test_planted_shape():
    mu = t5c.planted_mu(t5c.DODECA_STEPS, 0.15)
    assert mu[0] == 0.90 and abs(mu.min() - 0.75) < 1e-12
    assert int(np.sum(np.isclose(mu, 0.75))) == 15                    # 1072 … 9536 at the bottom


def test_synthetic_small():
    rng = np.random.default_rng(1)
    planted = t5c.synthetic(0.0072, 0.15, rng, draws=40)
    flat = t5c.synthetic(0.0072, 0.0, rng, draws=40)
    assert planted["spreads"] >= 0.75 and flat["spreads"] == 0 and flat["does not spread"] == 1
