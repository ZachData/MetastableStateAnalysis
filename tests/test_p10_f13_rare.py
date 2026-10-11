"""F13r (`design-10.md` "F13r"): the rule's tests on hand-built inputs."""
import numpy as np
import pytest

from tools.run import p10_f13_rare as r
from tools.run import p10_pile_unigram as u


def test_sentence_pos_hand_sequence():
    texts = ["The", " cat", " sat", ".", " It", " ran", "!\n", "x", None, " y?", "z"]
    assert r.sentence_pos(texts).tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0]


def test_counter_short_text_and_cut():
    counts, total, docs = u.count_tokens([[1, 2, 2], [3, 3, 3, 0]], 5, limit=5)
    assert (total, docs) == (5, 2)
    assert counts.tolist() == [0, 1, 2, 2, 0]
    with pytest.raises(u.UnigramError):
        u.check_counts(counts, 5, {0: "Ġthe", 1: ",", 2: "x", 3: "y", 4: "z"})
    assert u.check_counts(counts, 5, {0: "Ġthe", 1: ",", 2: ".", 3: "y", 4: "z"})


def _cloud(seed=0, n=400):
    rng = np.random.default_rng(seed)
    kept = np.arange(1, n)
    x = rng.normal(size=n)                                   # the covariate (lf)
    return rng, kept, x


def test_matched_recovers_planted_effect_independent_of_x():
    rng, kept, x = _cloud()
    cen = np.sort(rng.choice(kept, 60, replace=False))
    y = 0.5 * x + rng.normal(scale=0.1, size=x.size)
    y[cen] -= 1.0
    q = r.tercile(x, kept)
    s4 = r.matched_diff(y, cen, np.setdiff1d(kept, cen), r.strata(kept, None))
    sx = r.matched_diff(y, cen, np.setdiff1d(kept, cen), r.strata(kept, q))
    assert sx == pytest.approx(-1.0, abs=0.15) and s4 == pytest.approx(-1.0, abs=0.3)


def test_matched_near_zero_when_y_is_x_and_centres_rare():
    rng, kept, x = _cloud(1, 600)
    y = x + rng.normal(scale=0.05, size=x.size)
    rare = kept[x[kept] < np.quantile(x[kept], 0.3)]
    cen = np.sort(rng.choice(rare, 50, replace=False))
    non = np.setdiff1d(kept, cen)
    s4 = r.matched_diff(y, cen, non, r.strata(kept, None))
    sx = r.matched_diff(y, cen, non, r.strata(kept, r.tercile(x, kept)))   # the rule's terciles
    assert s4 < -0.8                                         # unmatched: centres look attended less
    assert abs(sx) < 0.25 * abs(s4)                          # matched on x: mostly gone


def test_terciles_leave_a_tail_the_linear_one_does_not():
    """The amendment's case: centres in the rarest 8 %, log ρ = lf + noise."""
    a = []
    for seed in range(20):
        rng, kept, x = _cloud(seed, 220)
        y = x + rng.normal(scale=0.3, size=x.size)
        rare = kept[x[kept] < np.quantile(x[kept], 0.08)]
        cen = np.sort(rng.choice(rare, 12, replace=False))
        non = np.setdiff1d(kept, cen)
        sx = r.matched_diff(y, cen, non, r.strata(kept, r.tercile(x, kept)))
        sl = r.ols_centre(y[kept], np.isin(kept, cen), [x[kept]], kept.astype(float))
        a.append((sx, sl))
    a = np.array(a)
    assert (a[:, 0] < 0).all() and abs(np.median(a[:, 1])) < 0.1


def test_matched_readable_bar_counts_only_shared_strata():
    kept = np.arange(1, 41)
    level = np.zeros(41, dtype=int)
    level[21:] = 1
    cen = np.array([25, 30, 35])                             # all in level 1
    non = np.setdiff1d(kept, cen)
    y = np.zeros(41)
    assert r.matched_diff(y, cen, non, r.strata(kept, level, 1)) == 0.0
    assert r.matched_diff(y, cen, non[non <= 20], r.strata(kept, level, 1)) is None


def test_ols_centre_recovers_coefficient():
    rng, kept, x = _cloud(2)
    c = np.zeros(kept.size, dtype=bool)
    c[rng.choice(kept.size, 50, replace=False)] = True
    y = 2.0 * x[kept] - 0.7 * c + rng.normal(scale=0.01, size=kept.size)
    assert r.ols_centre(y, c, [x[kept]], kept.astype(float)) == pytest.approx(-0.7, abs=0.02)


def test_reading_rules():
    def L(**kw):
        base = {s: "mixed" for s in r.STATS}
        base.update(kw)
        return {s: {"B": {1: v}} for s, v in base.items()}

    def M(m4, mx):
        return {s: {"B": {1: (m4 if s == "S_c4" else mx)}} for s in r.STATS}

    labs = L(**{"S_c4": "attended less", "S_c|f": "mixed", "C_f": "centres rarer",
                "R_f": "leans commoner attended more"})
    assert r.reading(labs, M(-0.6, -0.2), "B", 1, "f") == "carries"
    assert r.reading(labs, M(-0.6, -0.4), "B", 1, "f") == "unclear"      # not shrunk by half
    labs["S_c|f"] = {"B": {1: "leans attended less (isolated)"}}
    assert r.reading(labs, M(-0.6, -0.5), "B", 1, "f") == "unclear"      # the linear one is mixed
    labs["S_c~f"] = {"B": {1: "attended less"}}
    assert r.reading(labs, M(-0.6, -0.5), "B", 1, "f") == "survives"
    labs["S_c|f"] = {"B": {1: "mixed"}}
    assert r.reading(labs, M(-0.6, -0.2), "B", 1, "f") == "carries"      # the strata lost it
    labs = L(**{"S_c4": "attended less", "C_f": "centres commoner", "R_f": "commoner attended more"})
    assert r.reading(labs, M(-0.6, -0.1), "B", 1, "f") == "unclear"      # x moves the wrong way
    assert r.reading(L(), M(0, 0), "B", 1, "f") is None
