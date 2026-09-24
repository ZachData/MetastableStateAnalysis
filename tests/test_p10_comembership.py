"""
`tools/run/p10_comembership.py` — what clustered unique tokens cluster with.

The reading is observed minus a random-draw expectation, so these check the
expectations are exact (hypergeometric, mid-rank percentile), that a planted
structure produces the lift it should, and that bad inputs are refused.
"""
from itertools import combinations

import numpy as np
import pytest

from tools.run.p10_comembership import (
    deltas,
    focal_stats,
    midrank_pct,
    p_none,
    summarise,
)

# Tier: numpy only -- runs in `scripts/check.sh pure`.
pytestmark = pytest.mark.pure


def test_p_none_matches_enumeration():
    pool, good = 7, 3
    for k in range(0, 6):
        draws = list(combinations(range(pool), k))
        want = sum(1 for d in draws if not set(d) & set(range(good))) / len(draws)
        assert p_none(pool, good, k) == pytest.approx(want)
    assert p_none(5, 3, 3) == 0.0


def test_midrank_pct_averages_one_half_with_ties():
    v = np.array([0.1, 0.5, 0.5, 0.5, 0.9, 1.0])
    assert midrank_pct(v, v).mean() == pytest.approx(0.5)
    assert midrank_pct(v, np.array([0.1]))[0] == 0.0
    assert midrank_pct(v, np.array([1.0]))[0] == 1.0


def test_random_members_have_zero_expected_lift():
    # Averaged over every possible co-member set, observed equals expected.
    n, f, k = 7, 3, 2
    is_copy = np.array([1, 1, 0, 0, 1, 0, 0], bool)
    cls = np.array(list("aabaabb"), dtype=object)
    emb = np.linspace(0, 1, n)
    others = [p for p in range(n) if p != f]
    rows = [focal_stats(f, np.array([f, *co]), is_copy, cls, emb) for co in combinations(others, k)]
    for p in ("copy_share", "no_copy", "adjacent", "same_class", "emb_pct"):
        obs = np.mean([r[p][0] for r in rows])
        assert obs == pytest.approx(rows[0][p][1]), p


def test_planted_neighbour_cluster_lifts_adjacent_only_where_planted():
    n = 20
    is_copy = np.zeros(n, bool)
    cls = np.array(["w"] * n, dtype=object)
    emb = np.random.default_rng(0).normal(size=n)
    r = focal_stats(10, np.array([9, 10, 11]), is_copy, cls, emb)
    assert r["adjacent"][0] == 1.0 and r["adjacent"][1] < 0.25
    assert r["no_copy"] == (1.0, 1.0)            # no copies anywhere: no lift possible
    assert r["k"] == 2


def test_summary_and_delta_reading():
    def run(lift):
        return {0: {"n_focal": 4, "n_unique": 8, "k": 3.0,
                    **{p: [0.5 + lift, 0.5, lift] for p in
                       ("copy_share", "no_copy", "adjacent", "same_class", "emb_pct")}}}
    results = {(0, "a"): run(0.0), (0, "repeated_tokens"): run(0.4),
               (100, "a"): run(0.2), (100, "repeated_tokens"): run(0.4)}
    s = summarise(results)
    assert s["all"][0][0]["adjacent"]["lift"] == pytest.approx(0.2)
    assert s["without_repeated_tokens"][0][0]["adjacent"]["lift"] == pytest.approx(0.0)
    assert s["all"][0][0]["focal_frac"] == pytest.approx(0.5)
    d = deltas(s)
    assert d["without_repeated_tokens"][100][0]["emb_pct"]["reading"] == "above step 0"
    assert d["all"][100][0]["emb_pct"]["delta"] == pytest.approx(0.1)
    assert d["all"][0]["mean"]["copy_share"]["reading"] == "as step 0"


def test_run_with_no_focal_tokens_is_skipped_not_zeroed():
    empty = {0: {"n_focal": 0, "n_unique": 5}}
    full = {0: {"n_focal": 2, "n_unique": 5, "k": 1.0,
                **{p: [0.9, 0.5, 0.4] for p in
                   ("copy_share", "no_copy", "adjacent", "same_class", "emb_pct")}}}
    s = summarise({(0, "a"): empty, (0, "b"): full})
    assert s["all"][0][0]["n_runs"] == 1
    assert s["all"][0][0]["adjacent"]["lift"] == pytest.approx(0.4)
    assert s["all"][0][0]["focal_frac"] == pytest.approx(0.2)
