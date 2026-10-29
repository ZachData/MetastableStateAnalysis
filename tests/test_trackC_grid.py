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


def _plant(tmp, monkeypatch, bump=0.0, v_known=None):
    """A fake A5 seed 0 and Phase 1 tree: two tokens, E₁ rising over 25 states, so V = 0."""
    recs = tmp / "recs"
    recs.mkdir()
    e = [0.5 + 0.01 * i for i in range(g.STATES)]
    for s in STEPS:
        for p in g.V1:
            d = tmp / "p1" / f"{s}_{p}"
            d.mkdir(parents=True)
            (d / "tokens.txt").write_text("  0  a\n  1  b\n")
            mine = [x * (1 + bump) if (s, p) == (STEPS[3], g.V1[0]) else x for x in e]
            (d / "energies.json").write_text(json.dumps({"layers": [{"energies": {"1.0": x}} for x in mine]}))
            (d / "layer_metrics.csv").write_text("layer,energy_beta1.0\n" + "".join(f"{i},{x!r}\n" for i, x in enumerate(mine)))
            (recs / f"step{s}_{p}.json").write_text(json.dumps({"tokens": ["a", "b"], "e1": e}))
    (tmp / "gate.json").write_text(json.dumps({"known_answer": {"V": v_known or [0] * len(STEPS)}}))
    monkeypatch.setattr(g, "RECS", recs)
    monkeypatch.setattr(g, "GATE0", tmp / "gate.json")
    monkeypatch.setattr(g, "run_dir", lambda s, p: tmp / "p1" / f"{s}_{p}")


def test_the_check_passes_on_a5s_own_numbers(tmp_path, monkeypatch):
    _plant(tmp_path, monkeypatch)
    assert g.check_a5()["pass"]


@pytest.mark.parametrize("bump, v_known", [(1e-3, None), (0.0, [1] + [0] * (len(STEPS) - 1))])
def test_the_check_refuses_an_energy_off_tolerance_or_another_v(tmp_path, monkeypatch, bump, v_known):
    _plant(tmp_path, monkeypatch, bump, v_known)
    with pytest.raises(g.GridRefused):
        g.check_a5()
