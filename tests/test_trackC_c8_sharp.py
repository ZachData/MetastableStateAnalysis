"""C8 U2 (`trackC_writeup/c8_sharp_rules.py`, `c8_sharp.power`): rule R_ii's table, its refusals, and
the Lanczos iteration on small planted matrices (CPU)."""
import numpy as np
import pytest

from trackC_writeup import c8_sharp_rules as sr

BAR = sr.THETA * sr.THRESH
F2_ST = [s for s in range(4, 129, 4)]
BR_ST = [256 + k for k in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128)]
WIN = {"F2": (28, 48), "BP2": (264, 280)}
YES = {"F2": True, "BP2": True, "BK2": False}


def _lam(states, cross_at=None, hi=2.0, lo=0.1):
    return {s: (hi if cross_at is not None and s >= cross_at else lo) * sr.THRESH for s in states}


def _bk(cross_at=None):
    return _lam([256] + BR_ST, cross_at)


def test_threshold_and_bar():
    assert sr.THRESH == pytest.approx(3.8e4)
    assert sr.threshold(5e-4) == pytest.approx(7.6e4) and sr.threshold(1e-4) == pytest.approx(3.8e5)
    assert BAR == pytest.approx(1.9e4)


def test_window_is_i_s():
    assert sr.window((44, 48), None) == (28, 48)
    assert sr.window((272, 280), 256) == (264, 280)


@pytest.mark.parametrize("f2,bp2,bk2,want", [
    (40, 272, None, "crosses at the onset"),
    (8, 272, None, "crosses, not at the onset"),            # F2 crosses before its window
    (40, 300, None, "crosses, not at the onset"),           # BP2 after its window
    (40, 272, 300, "crosses, not at the onset"),            # BK2 crosses
    (None, None, None, "never crosses"),
    (40, None, None, "mixed"),
    (None, None, 300, "crosses, not at the onset"),
])
def test_label_table(f2, bp2, bk2, want):
    lams = {"F2": _lam(F2_ST, f2), "BP2": _lam(BR_ST, bp2), "BK2": _bk(bk2)}
    assert sr.label_at_tau(lams, {}, WIN, YES)["label"] == want


def test_crossing_exactly_at_the_window_edges():
    lams = {"F2": _lam(F2_ST, 28), "BP2": _lam(BR_ST, 264), "BK2": _bk()}
    assert sr.label_at_tau(lams, {}, WIN, YES)["label"] == "crosses, not at the onset"   # lo is open
    lams = {"F2": _lam(F2_ST, 48), "BP2": _lam(BR_ST, 280), "BK2": _bk()}
    assert sr.label_at_tau(lams, {}, WIN, YES)["label"] == "crosses at the onset"        # hi is closed


def test_premise_refuses():
    lams = {"F2": _lam(F2_ST, 40), "BP2": _lam(BR_ST, 272), "BK2": _bk()}
    for k in YES:
        p = dict(YES)
        p[k] = not p[k]
        assert sr.label_at_tau(lams, {}, WIN, p)["label"] == "refused"


def test_unconverged_state_that_decides_refuses():
    # BK2 below everywhere, but an unconverged BK2 state could be a crossing: the label would flip
    lams = {"F2": _lam(F2_ST, 40), "BP2": _lam(BR_ST, 272), "BK2": _bk()}
    assert sr.label_at_tau(lams, {"BK2": [304]}, WIN, YES)["label"] == "refused"
    # an unconverged F2 state after its first crossing cannot change the label
    assert sr.label_at_tau(lams, {"F2": [100]}, WIN, YES)["label"] == "crosses at the onset"


def test_theta_beside_changes_the_bar_only():
    lams = {"F2": _lam(F2_ST, 40, hi=0.7), "BP2": _lam(BR_ST, 272, hi=0.7), "BK2": _bk()}
    assert sr.label_at_tau(lams, {}, WIN, YES)["label"] == "crosses at the onset"
    assert sr.label_at_tau(lams, {}, WIN, YES, theta=1.0)["label"] == "never crosses"


def test_joint():
    assert sr.joint({"a": "never crosses", "b": "never crosses"}) == "never crosses"
    assert sr.joint({"a": "never crosses", "b": "refused"}) == "mixed"


def _sym(n, eig, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (q * eig) @ q.T


@pytest.mark.parametrize("eig", [
    np.r_[10.0, 9.0, np.linspace(-1, 5, 98)],
    np.r_[-50.0, 3.0, np.linspace(-1, 2.5, 98)],          # dominant negative: the top algebraic is 3
])
def test_lanczos_top_algebraic(eig):
    torch = pytest.importorskip("torch")
    from trackC_writeup import c8_sharp as cs
    A = torch.tensor(_sym(100, eig, 0), dtype=torch.float32)
    like = [torch.zeros(40), torch.zeros(60)]

    def op(v):
        w = A @ torch.cat(v)
        return [w[:40], w[40:]]
    r = cs.power(op, like)
    assert r["mu"] == pytest.approx(float(np.max(eig)), rel=1e-3)
    assert r["converged"]


def test_preconditioned_operator_has_p_inverse_h_spectrum():
    torch = pytest.importorskip("torch")
    from trackC_writeup import c8_sharp as cs
    rng = np.random.default_rng(3)
    H = _sym(50, rng.uniform(0, 4, 50), 1)
    p = 10 ** rng.uniform(-2, 2, 50)
    d = torch.tensor(p ** -0.5, dtype=torch.float32)
    Ht = torch.tensor(H, dtype=torch.float32)
    op = cs.precond_op(lambda v: [Ht @ v[0]], [d])
    r = cs.power(op, [torch.zeros(50)], n_max=50)
    want = float(np.max(np.real(np.linalg.eigvals(np.diag(1 / p) @ H))))
    assert r["mu"] == pytest.approx(want, rel=1e-3)
