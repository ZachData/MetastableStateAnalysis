"""LC (`design-10.md` "LC"): the surprisal indexing and the position-binned gap, on hand cases."""
import numpy as np

from tools.run import p10_lc_loss as lc


def test_s_reads_the_next_token_and_own_shifts_it():
    ids = [2, 0, 1, 2]
    p = np.array([[0.1, 0.2, 0.7], [0.5, 0.25, 0.25], [0.2, 0.2, 0.6], [1 / 3, 1 / 3, 1 / 3]])
    s, H = lc.surprisals(np.log(p), ids)
    assert np.allclose(s, -np.log([0.1, 0.25, 0.6]))       # s[i] = −log p(x_{i+1} | x_≤i)
    assert np.isclose(H[3], np.log(3))
    so = lc.own(s)
    assert np.isnan(so[0]) and np.allclose(so[1:], s)        # s_own[i] = s[i − 1]


def test_domain_drops_position_0_and_the_last():
    assert lc.domain(np.array([0, 1, 5, 8, 9]), 10).tolist() == [1, 5, 8]


def test_binned_gap_hand_case():
    v = np.ones(40)
    dom = np.arange(1, 33)                                  # 8 bins of 4
    A = dom[dom % 4 == 0]                                   # one member per bin
    v[A] = 0.5
    B = np.setdiff1d(dom, A)
    d = lc.rel_binned(v, dom, A, B)
    assert np.isclose(d, -0.5 / v[dom].mean())


def _planted(delta, n=300, seed=0):
    rng = np.random.default_rng(seed)
    i = np.arange(n - 1)
    s = (3 + 5 * (1 + i) ** -0.4) * rng.exponential(1.0, n - 1)
    dom = lc.domain(np.arange(1, n), n)
    lab = np.where(rng.random(dom.size) < 0.25 + 0.5 * (dom > n / 2), 0, -1)   # members sit later
    s[dom[lab >= 0]] *= 1 - delta
    nx = {k: np.zeros(n) for k in ("lf", "ws", "ct")}
    nx["lf"] = rng.normal(size=n)
    return lc.layer_stats(s, np.append(s, s[-1]), dom, lab, nx)


def test_position_alone_is_near_zero_and_a_planted_effect_is_recovered():
    null = [_planted(0.0, seed=k)["D"] for k in range(40)]
    eff = [_planted(0.3, seed=k)["D"] for k in range(40)]
    assert abs(np.median(null)) < 0.05
    assert -0.4 < np.median(eff) < -0.2


def test_onset_rule():
    st = lc.PRIMARY_STEPS
    labs = {s: "mixed" for s in st}
    labs[2000], labs[4000] = "lower", "leans lower"
    assert lc.onset(labs) == (2000, "co-locates")
    labs = {s: "mixed" for s in st}
    labs[256] = "lower"                                     # alone, next step mixed: not an onset
    assert lc.onset(labs) == (None, "no gap")
    labs[143000] = "lower"
    assert lc.onset(labs) == (143000, "follows")


def test_floor_label():
    assert lc.floor_label([-0.01] * 7) == "lower (under the floor)"
    assert lc.floor_label([-0.1] * 7) == "lower"
    assert lc.floor_label([-0.1] * 6 + [0.1]) == "leans lower"


def test_rank_gap_sign_and_null_symmetry():
    assert _planted(0.3, seed=1)["A"] < 0
    rng = np.random.default_rng(5)
    signs = []
    for k in range(400):                                    # skewed v, small member set: A's sign is fair
        v = rng.exponential(1.0, 60)
        dom = np.arange(1, 59)
        A = dom[:8]
        signs.append(np.sign(lc.rank_gap(v, dom, A, dom[8:]) or 0))
    assert abs(np.mean(signs)) < 0.12


def test_onset_skips_too_few():
    labs = {s: "mixed" for s in lc.PRIMARY_STEPS}
    labs[512], labs[1000], labs[2000] = "lower", "too few", "leans lower"
    assert lc.onset(labs) == (512, "precedes")
