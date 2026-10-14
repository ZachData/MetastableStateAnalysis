"""Track A's t* comparators refuse where a silent inf or 0 would read as "below" (#201, finding 4)."""

import math

import numpy as np
import pytest

from trackA_force import sa_sim, tstar_table

pytestmark = pytest.mark.deps  # scipy and torch


def test_ode_tstar_refuses_out_of_range_starts():
    for g0 in (0.9, 0.95, -0.01):
        with pytest.raises(ValueError):
            tstar_table.tstar(467, 3.5, g0)


def test_ode_tstar_beta0_closed_form():
    # tools/math_checks/a1_collapse_time.py item 1: t* = log(((n-1) g + 1) / (1 - g)) / 2
    assert tstar_table.tstar(467, 0.0) == pytest.approx(0.5 * math.log((466 * 0.9 + 1) / 0.1), rel=1e-6)


def test_sim_refuses_start_above_target():
    U = np.ones((8, 4)) + 1e-3 * np.random.default_rng(0).standard_normal((8, 4))
    with pytest.raises(ValueError):
        sa_sim.tstar(U, 1.0)


def test_sim_full_attention_matches_ode_small():
    # G0's known answer on a small cloud: n = 20 < d = 256, near-orthogonal start
    U = np.random.default_rng(1).standard_normal((20, 256))
    t = sa_sim.tstar(U, 1.0, causal=False, dt=0.005)
    assert t == pytest.approx(tstar_table.tstar(20, 1.0), rel=0.1)
