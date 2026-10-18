"""B3 unit 1's parts (`trackB_circuit/design-B3.md`): N_eff and the row conventions, the weighted
slope, G1's planted cases (ii) exact, R2's onset and count, C3′'s nearest pools, A3's received
ratio on uniform routing."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trackB_circuit import b3_route as b3

pytestmark = pytest.mark.deps  # torch (row_stats)


def _causal(L: np.ndarray) -> np.ndarray:
    n = L.shape[-1]
    L = np.where(np.tril(np.ones((n, n), bool)), L, -np.inf)
    E = np.exp(L - L.max(-1, keepdims=True))
    return E / E.sum(-1, keepdims=True)


def test_uniform_rows_have_neff_i_and_slope_one():
    n = 64
    A = _causal(np.zeros((1, n, n)))
    r = b3.row_stats(A)
    i = np.arange(b3.ROW_FROM, n)
    assert np.allclose(r["neff"][0], i)                  # keys 1..i, position 0 dropped
    assert np.allclose(r["sink"][0], 1 / (i + 1))
    assert np.allclose(r["spread"][0], 0, atol=1e-12)
    a = b3.alpha_of({"p": {**{k: v[None] for k, v in r.items()}, "n": i.astype(float)}})
    assert a.shape == (1,) and abs(a[0] - 1) < 1e-12


def test_two_partners_have_neff_two_and_slope_zero():
    n = 80
    L = np.full((n, n), -1e9)
    L[:, 0] = 0.0
    for i in range(2, n):
        L[i, i - 1] = L[i, i - 2] = 0.0
    r = b3.row_stats(_causal(L[None]))
    assert np.allclose(r["neff"], 2.0)
    i = np.arange(b3.ROW_FROM, n).astype(float)
    s = b3.wls_sums(np.log(i), np.log(r["neff"][0]), b3.head_weights(r["sink"][0]))
    assert abs(b3.slope_of(s)) < 1e-12


def test_weights_drop_parked_rows():
    sink = np.array([0.5, 0.96, 0.0])
    assert np.allclose(b3.head_weights(sink), [0.5, 0.0, 1.0])


def test_slope_of_recovers_a_line_and_refuses_few_rows():
    x = np.log(np.arange(16, 200.0))
    y = 0.37 * x + 2
    s = b3.wls_sums(x, y, np.ones_like(x))
    assert abs(b3.slope_of(s) - 0.37) < 1e-12
    assert np.isnan(b3.slope_of(s, np.asarray(3)))


def test_onset_needs_two_consecutive_hits_from_step_two():
    q = np.zeros(6)
    assert b3.onset_index(np.array([-1, 1, -1, 1, -1, -1.0]), q) == 4
    assert b3.onset_index(np.array([-1, -1, -1, 1, 1, 1.0]), q) == 1    # index 0 is step 0
    assert b3.onset_index(np.array([1, 1, 1, 1, 1, -1.0]), q) is None   # the last cannot hold


def test_r2_hits_and_attainable_rank():
    A = np.ones((5, 4))
    A[2:, 0] = -1                              # onset 2
    A[4:, 1] = -1                              # no onset (last step)
    q = np.zeros(5)
    assert b3.r2_hits(A, [0, 1], [3, 3], q).tolist() == [True, False]
    counts, rates = b3.r2_null(A, [np.array([0, 2]), np.array([1, 3])], [2, 2], q, 1000, 0)
    assert np.allclose(rates, [0.5, 0.0])
    assert counts.max() <= 1


def test_formation_is_first_half_maximum():
    assert b3.formation_index(np.array([0, 0.1, 0.6, 1.0, 0.9])) == 2


def test_nearest_pool_excludes_and_orders():
    sc = np.array([0.9, 0.7, 0.1, 0.8, 0.95, 0.0])
    assert b3.nearest_pool(sc, 0.9, [0], k=3).tolist() == [4, 3, 1]


def test_received_ratio_is_zero_on_uniform_routing():
    n = 40
    A = _causal(np.zeros((2, n, n)))
    lab = np.full(n, -1)
    lab[[3, 7, 20, 33]] = 0
    lab[[5]] = -2
    assert np.allclose(b3.received_ratio(A, lab), 0, atol=1e-12)


def test_pctl_nan_gives_half_without_a_value():
    p = b3.pctl_nan(np.array([1.0, np.nan, 3.0, 2.0]))
    assert p[1] == 0.5 and p[0] < p[3] < p[2]
