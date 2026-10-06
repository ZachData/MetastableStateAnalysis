"""
`tools/run/p10_r6_matcher.py` — R6's cross-checkpoint matcher: link kinds, stable Jaccard, filter
flips, lineage origin, pooling and the first check, as `design-10.md` "R6" fixed them.
"""
import numpy as np
import pytest

from tools.run import p10_r6_matcher as r6

# Tier: merge_tree's package imports scikit-learn -- runs in `scripts/check.sh deps`.
pytestmark = pytest.mark.deps


def test_a_group_that_keeps_its_members_is_stable_with_jaccard_one():
    out = r6.link([0, 0, 0, -1, -1], [5, 5, 5, -1, -1])
    assert out["kind_a"] == {0: "stable"} and out["kind_b"] == {5: "stable"}
    assert out["stable"] == {5: (0, 1.0)}


def test_a_changed_group_is_stable_on_containment_with_jaccard_below_one():
    # 3 of 4 kept, 2 joined: containment 3/4, Jaccard 3/6
    out = r6.link([0, 0, 0, 0, -1, -1], [1, 1, 1, -1, 1, 1])
    assert out["stable"] == {1: (0, 0.5)}


def test_a_split_and_a_birth_and_a_death():
    out = r6.link([0, 0, 0, 0, 1, 1, -1, -1], [2, 2, 3, 3, -1, -1, 4, 4])
    assert out["kind_a"] == {0: "split", 1: "death"}
    assert out["kind_b"] == {2: "split", 3: "split", 4: "birth"}
    assert out["stable"] == {}


def test_a_small_group_inside_a_large_one_links_on_containment():
    # Jaccard 2/8 would call this a death and a birth (status-1d "Merge tree", #104)
    out = r6.link([0, 0, -1, -1, -1, -1, -1, -1], [1] * 8)
    assert out["kind_a"] == {0: "stable"}


def test_a_c3_birth_is_a_flip_when_c2a_links_the_same_id():
    c2a = r6.link([0, 0, 0, 1, 1, 1], [7, 7, 7, 8, 8, 8])
    c3 = r6.link([0, 0, 0, -1, -1, -1], [7, 7, 7, 8, 8, 8])   # group 1 failed c3's filter at the earlier step
    assert r6.flips(c3, c2a) == {"birth_flip": 1}
    c2a_new = r6.link([0, 0, 0, -1, -1, -1], [7, 7, 7, 8, 8, 8])
    assert r6.flips(c3, c2a_new) == {"birth_new": 1}


def test_a_c3_death_is_gone_unless_c2a_links_it():
    c3 = r6.link([0, 0, 1, 1], [5, 5, -1, -1])
    assert r6.flips(c3, r6.link([0, 0, 1, 1], [5, 5, 6, 6])) == {"death_flip": 1}
    assert r6.flips(c3, r6.link([0, 0, 1, 1], [5, 5, -1, -1])) == {"death_gone": 1}


def test_lineage_origin_walks_back_through_stable_links_only():
    steps = [0, 2, 4]
    links = [r6.link([0, 0, 1, 1], [3, 3, 3, 3]),   # merge: breaks the chain
             r6.link([3, 3, 3, 3], [9, 9, 9, 9])]
    assert r6.lineage_origin(links, steps, 9) == 2
    links = [r6.link([0, 0, -1, -1], [3, 3, -1, -1]), r6.link([3, 3, -1, -1], [9, 9, -1, -1])]
    assert r6.lineage_origin(links, steps, 9) == 0


def test_chain_and_pool_on_a_toy_record(monkeypatch):
    monkeypatch.setattr(r6, "N_DRAWS", 20)
    steps = [0, 2, 4]
    lab = {0: np.array([0, 0, 0, 1, 1, 1, -1, -1]), 2: np.array([4, 4, 4, 5, 5, 5, -1, -1]),
           4: np.array([6, 6, 6, -1, -1, -1, 7, 7])}
    by_col = {c: lab for c in r6.COLUMNS}
    ch = r6.chain((("p", 1), steps, by_col, [0, 0, 1]))
    out = r6.pool([ch], steps, r6.COLUMNS)
    b0, b1 = out["boundaries"][0]["c3"], out["boundaries"][1]["c3"]
    assert b0["share_a"]["stable"] == 1.0 and b0["births"] == 0
    assert b1["share_a"]["stable"] == 0.5 and b1["share_a"]["death"] == 0.5 and b1["births"] == 1
    assert b1["flips"] == {"birth_new": 1, "death_gone": 1}
    assert b0["stable_jaccard"] == {"n": 2, "median": 1.0, "identical": 1.0, "same": 1.0}
    assert out["lineage_origin_at_last"]["c3"] == {"0": 1, "2": 0, "4": 1}
    assert 0 < b0["null"]["p"] <= 1 and b0["null"]["observed"] == 2
    assert r6.first_check(out)["passes"]
