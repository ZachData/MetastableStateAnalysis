"""Figure 12, the energy ridgeline (`trackC_writeup/c2_ridge.py`, `numbers.ridge_hidden`): pure parts
(/challenge-pr on #280, finding 6)."""
import json

import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackC_writeup import c2_ridge as r
from trackC_writeup import numbers as n


def _sphere(rows, dim, seed):
    x = np.random.default_rng(seed).normal(size=(rows, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_span_coords_keep_the_gram_matrix_even_when_rank_deficient():
    x = _sphere(12, 40, 0)
    x[5] = x[2]                                      # a repeated token: rank 11
    c = r.span_coords(x)
    assert np.allclose(c @ c.T, x @ x.T, atol=1e-12)
    assert np.isclose(r.energy(c), r.energy(x))


def test_full_mask_flow_never_falls_and_reads_the_states():
    reads, min_fine = r.identity_run(_sphere(30, 8, 1), causal=False, t_end=4.0, n_fine=480, states=25)
    assert len(reads) == 25 and min_fine >= 0
    assert all(b >= a for a, b in zip(reads, reads[1:]))
    assert reads[-1] <= np.exp(1) / 2 + 1e-12        # E_1 of one point is the ceiling


def test_reads_that_do_not_divide_the_fine_steps_refuse():
    with pytest.raises(r.RidgeRefused):
        r.identity_run(_sphere(5, 3, 2), causal=False, n_fine=100, states=25)


def test_drops_are_the_relative_rule():
    e = np.array([1.0, 1.1, 1.1 - 0.5e-3 * 1.1, 1.1 - 2e-3 * 1.1, 1.2])
    assert r.drops(e).tolist() == [False, False, True, False]


def test_ridge_hidden_finds_a_covered_ridge_and_the_tracked_figure_hides_none():
    back_low, front_high = [0.0, 0.1, 0.0], [0.0, 1.0, 0.0]
    assert max(n.ridge_hidden([back_low, front_high], amp=3.0)) > 0
    assert max(n.ridge_hidden([back_low, front_high], amp=0.5)) == 0
    e1 = json.loads(n.TABLE.read_text())["keys"]["ridge.e1"]["value"]
    assert max(n.ridge_hidden(e1, n.RIDGE_AMP)) == 0
    assert max(n.ridge_hidden(e1, 3.4)) > 0.5         # the height /challenge-pr on #280 found hiding step 8
