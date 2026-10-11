"""LCo (`design-10.md` "LCo"): own features unshifted; the adjusted gap on planted cases."""
import numpy as np
import pytest

from tools.run import p10_lc_loss as lc
from tools.run import p10_lc_own as lo

pytestmark = pytest.mark.deps


def test_own_features_are_the_token_at_i(monkeypatch):
    from tools.run import p10_token_composition as tc
    vocab = {"a": 0, "Ġb": 1, "c": 2}
    counts = np.array([10.0, 100.0, 1000.0])
    ft = lo.own_features(["a", "Ġb", "c"], counts, vocab, set())
    assert np.allclose(ft["lf"], np.log10((counts + 0.5) / counts.sum()))      # position i, not i + 1
    cls = [t["cls"] for t in tc.token_features(np.array(["a", "Ġb", "c"]), vocab, set())]
    assert ft["ws"].tolist() == [float(c == "word_start") for c in cls]


def _case(delta, seed, n=300):
    """s_own a function of lf and class alone (× Exp), members drawn frequent; members × (1 − δ)."""
    rng = np.random.default_rng(seed)
    dom = lc.domain(np.arange(1, n), n)
    lf = rng.uniform(-8, -1.5, n)
    ct = (rng.random(n) < 0.3).astype(float)
    ws = ((rng.random(n) < 0.5) & (ct == 0)).astype(float)
    ft = {"lf": lf, "ws": ws, "ct": ct}
    k = np.where(ct == 1, 0.3, np.where(ws == 1, 1.0, 0.7))
    so = (3 + 5 * (1 + np.arange(n)) ** -0.4) * (-lf / 4.5) * k * rng.exponential(1.0, n)
    p_mem = 1 / (1 + np.exp(-(lf[dom] + 3)))               # frequent tokens are the members
    lab = np.where(rng.random(dom.size) < p_mem, 0, -1)
    so[dom[lab >= 0]] *= 1 - delta
    return lo.layer_stats(so, lo.record_resids(so, dom, ft), dom, lab, ft)


def test_frequency_alone_is_adjusted_away_and_not_by_O0():
    null = [_case(0.0, k) for k in range(40)]
    assert abs(np.median([r["O"] for r in null])) < 0.05
    assert abs(np.median([r["O_A"] for r in null])) < 0.03
    assert np.median([r["O0_A"] for r in null]) < -0.05     # the confound: unadjusted, members read lower


def test_a_planted_member_effect_is_recovered():
    eff = [_case(0.3, k) for k in range(40)]
    # ln(0.7) = −0.36. The membership-blind fit takes part of it where members differ in lf (here
    # strongly), so O is attenuated towards 0 (conservative; the gate's E arms measure it on the
    # real members); O~, with member in the fit, is not.
    assert -0.36 < np.median([r["O"] for r in eff]) < -0.1
    assert np.median([r["O_A"] for r in eff]) < -0.05
    assert -0.5 < np.median([r["O~"] for r in eff]) < -0.25


def test_headline_reading():
    st = lc.PRIMARY_STEPS
    mixed = {b: {s: "mixed" for s in st} for b in lc.BANDS}
    o0 = {b: dict(v) for b, v in mixed.items()}
    o0["L1-8"][256] = "lower"
    r = lo.readings({"O": mixed, "O0": o0})
    assert r["headline"] == "carried by the token" and r["carried_by_token"] == [["L1-8", 256, "lower", "mixed"]]
    o = {b: dict(v) for b, v in mixed.items()}
    o["L1-8"][256], o["L1-8"][512] = "lower", "leans lower"
    assert lo.readings({"O": o, "O0": o0})["headline"] == "beyond frequency and class"
    o0["L1-8"][256] = "lower (under the floor)"
    assert lo.readings({"O": mixed, "O0": o0})["headline"] == "no own-token gap"


def test_a_full_label_needs_seven_passages():
    assert lo.floor_label([-0.1] * 7) == "lower"
    assert lo.floor_label([-0.1] * 6 + [None]) == "leans lower"
    assert lo.floor_label([-0.01] * 6) == "leans lower (under the floor)"
    assert lo.floor_label([-0.1] * 6 + [0.1]) == "leans lower"
