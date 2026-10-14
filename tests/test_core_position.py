"""core/position.py: the shared position covariate (`trackB_circuit/design-B.md` "B0")."""
import numpy as np
import pytest

from core import position as pos

pytestmark = pytest.mark.pure


def test_design_columns():
    X = pos.design([0, 1, 9])
    u = np.log([1.0, 2.0, 10.0])
    assert X.shape == (3, 5)
    assert np.allclose(X[:, :4], np.stack([u ** 0, u, u ** 2, u ** 3], 1))
    assert X[:, 4].tolist() == [1.0, 0.0, 0.0]


def test_removes_any_cubic_in_log_position_and_the_first_token():
    i = np.arange(200)
    u = np.log1p(i)
    v = 2 - 0.5 * u + 0.3 * u ** 2 - 0.02 * u ** 3 + 7.0 * (i == 0)
    assert np.abs(pos.residualise(v)).max() < 1e-9


def test_keeps_what_position_does_not_explain_and_is_column_wise():
    rng = np.random.default_rng(0)
    i = np.arange(300)
    e = rng.standard_normal(300)
    e -= pos.design(i) @ pos.fit(e, i)                 # orthogonal to the covariate
    v = np.stack([np.log1p(i) + e, e], 1)
    r = pos.residualise(v, i)
    assert np.allclose(r[:, 0], e) and np.allclose(r[:, 1], e)


def test_refuses_bad_input():
    with pytest.raises(ValueError):
        pos.design([-1, 2])
    with pytest.raises(ValueError):
        pos.fit(np.zeros(3), [0, 1])
