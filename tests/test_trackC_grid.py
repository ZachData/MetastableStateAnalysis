"""Figure 13's record (`trackC_writeup/c2_grid.py`) as the tracked table holds it: one value per
step and depth, on A5's grid, after the check against A5's seed 0. The producer itself reads
`data/phase12`, which CI does not have."""
import json

import pytest

pytestmark = pytest.mark.deps  # a1a's imports, torch
from trackA_force.a1a import RUNS, STEPS
from trackC_writeup import c2_grid as g
from trackC_writeup import numbers as n


def test_the_phase2_map_covers_the_grid():
    assert tuple(g.P2_RUNS) == STEPS == tuple(RUNS)


def test_the_table_holds_a_value_per_step_and_depth():
    k = {name: v["value"] for name, v in json.loads(n.TABLE.read_text())["keys"].items()}
    assert tuple(k["lxs.steps"]) == n.GRID
    for key, depth in (("lxs.eff_rank", g.STATES), ("lxs.mass", g.STATES), ("lxs.de", g.STATES - 1),
                       ("lxs.schur", g.BLOCKS)):
        assert len(k[key]) == len(n.GRID) and all(len(row) == depth for row in k[key]), key
    assert k["lxs.a5_pass"] and k["lxs.a5_rel"] <= k["lxs.a5_tol"] and k["lxs.a5_V"] == k["s0.V"]
    assert all(0 <= m <= 1 for row in k["lxs.mass"] for m in row)
