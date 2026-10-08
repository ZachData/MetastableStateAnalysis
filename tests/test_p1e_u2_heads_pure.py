"""
`p1e_energy_field/u2_heads.py`'s numpy parts: Shapley shares of the block's shared ascent sum to
``F(all)``, and on an additive set function they are each part's own value
(`design-1e.md` "U2's per-head arm: the rule", row *ascent share*).
"""
import numpy as np
import pytest

from p1e_energy_field import u2_block as ub
from p1e_energy_field import u2_heads as uh

pytestmark = pytest.mark.pure


def test_shapley_is_exact_on_an_additive_function_and_efficient_on_any():
    import itertools
    val = {"sink": 0.1, "keys": -0.3, "mlpx": 0.7, "bias": 0.05}
    subsets = [frozenset(S) for r in range(5) for S in itertools.combinations(uh.PLAYERS, r)]
    F = {S: sum(val[k] for k in S) for S in subsets}
    assert uh.shapley(F, uh.PLAYERS) == pytest.approx(val)
    rng = np.random.default_rng(0)
    G = {S: (0.0 if not S else float(rng.normal())) for S in subsets}
    assert sum(uh.shapley(G, uh.PLAYERS).values()) == pytest.approx(G[frozenset(uh.PLAYERS)])


def test_ascent_block_sums_to_the_resid_move_on_the_force():
    rng = np.random.default_rng(2)
    n, d = 30, 10
    X = rng.normal(size=(n, d))
    w, b = rng.uniform(0.5, 1.5, d), rng.normal(0, 0.1, d)
    frame = lambda Y: ub.unit_rows(Y, w, b, 1e-5)          # noqa: E731
    cbar = {k: rng.normal(0, 0.2, d) for k in uh.PLAYERS}
    t = np.arange(1, n)
    r = uh.ascent_block(X, cbar, t, frame)
    assert sum(r["shapley"].values()) == pytest.approx(r["F_all"])
    U = frame(X)
    g = ub.forces(U, 3.5, only=("causal",))["causal"][t]
    dres = ub.tangent(U[t], frame(X[t] + sum(cbar.values())) - U[t])
    want = np.mean(np.sum(dres * g / np.linalg.norm(g, axis=1, keepdims=True), axis=1))
    assert r["F_all"] == pytest.approx(want)


def test_a_short_head_cell_is_accepted_only_where_the_head_reads_key_0_and_itself():
    """`/challenge-pr` on #161, finding 2: the amended refusal path has a test."""
    c = {"source": "kernns_h:keys_h:r1out", "block": 3, "head": 1, "left_out": 40}
    ok = {(3, 1): {"n_left": 40, "left_max_keys_w": 5e-5}}
    assert uh.explained(c, ok)
    assert not uh.explained(c, {(3, 1): {"n_left": 40, "left_max_keys_w": 0.999}})   # another sink
    assert not uh.explained(c, {(3, 1): {"n_left": 39, "left_max_keys_w": 0.0}})      # counts differ
    assert not uh.explained({**c, "source": "kern_h:head_h:frozen"}, ok)               # whole head: no gap
    cell = {"n": 10, "X": 0.1, "left_out": 90, "block": 0, "source": "kernns_h:keys_h:r1out", "head": 0}
    rec = {"targets": 100, "heads": [{"block": 0, "head": 0, "n_left": 90, "left_max_keys_w": 0.5}],
           "cells": [], "head_cells": [cell]}
    with pytest.raises(SystemExit, match="not populated"):
        uh.check_populated(rec)
