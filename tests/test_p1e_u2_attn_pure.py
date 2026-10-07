"""
`p1e_energy_field/u2_attn.py`'s numpy parts: the shared-update split sums to 1, and a component
that is an exact mean-shift step reads +1 (`design-1e.md` "U2's attention arm: the rule").
"""
import numpy as np
import pytest

from p1e_energy_field import u2_attn as ua
from p1e_energy_field import u2_block as ub

pytestmark = pytest.mark.pure


def test_shares_sum_to_one_and_a_constant_is_fully_shared():
    rng = np.random.default_rng(0)
    n, d = 30, 8
    C = {k: rng.normal(size=(n, d)) for k in ("sink", "keys", "mlpx")}
    C["bias"] = np.tile(rng.normal(size=d), (n, 1))
    C["block"] = sum(C[k] for k in ua.PARTS)
    C["attn"], C["mlp"] = C["sink"] + C["keys"], C["mlpx"]
    C["a0_mean_heads"] = rng.uniform(size=n)
    t = np.arange(1, n)
    s = ua.shares(C, t)
    assert sum(s["share"].values()) == pytest.approx(1.0)
    assert "bias" not in s["sharedness"]
    np.testing.assert_allclose(np.linalg.norm(C["bias"][0]) / np.linalg.norm(C["bias"], axis=1).mean(), 1.0)


def test_a_component_that_is_the_field_step_reads_plus_one():
    """A component equal to an exact mean-shift step (in the frame) reads A ≈ +1 frozen."""
    rng = np.random.default_rng(1)
    n, d = 40, 12
    X = rng.normal(size=(n, d))
    w, b = np.ones(d), np.zeros(d)
    frame = lambda Y: ub.unit_rows(Y, w, b, 1e-5)          # noqa: E731
    U = frame(X)
    g = ub.forces(U, 3.5, only=("causal",))["causal"]
    # LN with unit gain and zero bias is scale-free on centred rows: step the centred row
    Xc = X - X.mean(axis=1, keepdims=True)
    r = np.linalg.norm(Xc, axis=1, keepdims=True)
    c = 1e-3 * r * g
    mv = ua.moves(Xc, c, np.arange(2, n), frame)
    cell = ub.cell(mv["frozen"], g, U, np.arange(2, n), ub.make_perms("t"))
    assert cell["A"] == pytest.approx(1.0, abs=1e-4)
