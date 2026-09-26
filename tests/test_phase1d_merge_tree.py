"""
tests/test_phase1d_merge_tree.py — the merge tree over scales (option D,
`lit-1d.md` §5) and cross-layer cluster linking by shared tokens.

Two classes exist because the first version of this module read real data
wrong and its tests could not see it: TestTies (identical token vectors
made zero-width plateaus whose k no cut reproduces) and TestBlobPlusOutliers
(the longest-lived plateau was one cluster plus outliers, and Jaccard
recorded a piece leaving it as a birth). Both failures pass on tidy
synthetic data, so these fixtures are built to look like the real layers.

TestAgreesWithClusterTracking bounds a duplication: on Jaccard inputs with
no split, `link_layer_pair` and `p1_mstate_tracking.cluster_tracking`
must agree on births, deaths and merge groups.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.deps

from sklearn.metrics import adjusted_rand_score

from p1d_cluster_ensemble.merge_tree import (
    MERGE_TREE_LINKAGES, labels_at_delta, layer_link_chain, layer_merge_tree,
    link_layer_pair, substantial_labels,
)
from p1d_cluster_ensemble.methods import LayerData

KINDS = ("stable", "merge", "split", "tangle", "birth", "death")


def _counts(**nonzero):
    return {k: nonzero.get(k, 0) for k in KINDS}


def _planted_caps(n_per=10, d=16, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.repeat([0, 1, 2], n_per)
    X = np.eye(3, d)[truth] + noise * rng.normal(size=(truth.size, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return LayerData.from_normed(X.astype(np.float32)), truth


def _cut_reproduces_every_plateau(tree, n):
    return all(len(set(labels_at_delta(tree["_Z"], n, p["delta_lo"]).tolist())) == p["k"]
               for p in tree["plateaus"])


# ---------------------------------------------------------------------------
# The merge tree
# ---------------------------------------------------------------------------

class TestLayerMergeTree:

    def test_plateaus_tile_delta_from_zero_to_inf(self):
        data, _ = _planted_caps()
        plateaus = layer_merge_tree(data)["plateaus"]
        assert plateaus[0]["delta_lo"] == 0.0 and plateaus[0]["k"] == data.n
        assert plateaus[-1]["delta_hi"] == float("inf") and plateaus[-1]["k"] == 1
        for a, b in zip(plateaus, plateaus[1:]):
            assert a["delta_hi"] == b["delta_lo"]
            assert a["lifetime"] > 0

    def test_planted_caps_give_three_substantial_clusters_as_the_top_plateau(self):
        data, truth = _planted_caps()
        tree = layer_merge_tree(data)
        top = tree["robust"][0]
        assert (top["k"], top["k_substantial"], top["n_outliers"]) == (3, 3, 0)
        assert tree["longest_any"] == top
        labels = labels_at_delta(tree["_Z"], data.n, top["delta_lo"])
        assert adjusted_rand_score(truth, labels) == pytest.approx(1.0)

    def test_ends_are_trivial_and_never_robust(self):
        data, _ = _planted_caps()
        tree = layer_merge_tree(data)
        assert tree["plateaus"][0]["trivial"] and tree["plateaus"][-1]["trivial"]
        assert all(not p["trivial"] for p in tree["robust"])

    def test_cut_at_delta_lo_reproduces_every_plateaus_k(self):
        data, _ = _planted_caps()
        tree = layer_merge_tree(data)
        assert _cut_reproduces_every_plateau(tree, data.n)

    def test_top_n_is_respected(self):
        data, _ = _planted_caps()
        assert len(layer_merge_tree(data, top_n=2)["robust"]) <= 2

    @pytest.mark.parametrize("linkage", MERGE_TREE_LINKAGES)
    def test_every_supported_linkage_runs(self, linkage):
        data, _ = _planted_caps()
        tree = layer_merge_tree(data, linkage=linkage)
        assert tree["branch"] == "plateaus" and tree["n_merges"] == data.n - 1

    def test_unsupported_linkage_raises(self):
        data, _ = _planted_caps()
        with pytest.raises(ValueError, match="ward"):
            layer_merge_tree(data, linkage="ward")

    @pytest.mark.parametrize("n", [0, 1])
    def test_degenerate_token_counts(self, n):
        data = LayerData(normed=np.zeros((n, 8), np.float32), cos_dist=np.zeros((n, n)))
        tree = layer_merge_tree(data)
        assert tree["branch"] == "n<2" and tree["plateaus"] == []

    def test_two_tokens_has_no_robust_plateau(self):
        data = LayerData.from_normed(np.eye(2, 8, dtype=np.float32))
        tree = layer_merge_tree(data)
        assert [p["k"] for p in tree["plateaus"]] == [2, 1]
        assert tree["robust"] == [] and tree["longest_any"] is None


class TestTies:
    """Layer 0 holds exact duplicates: repeated tokens share one embedding."""

    def _with_duplicates(self):
        data, _ = _planted_caps()
        X = np.concatenate([data.normed, data.normed[:12]])  # 12 exact repeats
        return LayerData.from_normed(X)

    def test_tied_merges_form_one_plateau_and_every_k_is_reproducible(self):
        data = self._with_duplicates()
        tree = layer_merge_tree(data)
        assert all(p["lifetime"] > 0 for p in tree["plateaus"])
        assert _cut_reproduces_every_plateau(tree, data.n)

    def test_the_duplicate_merged_floor_is_trivial(self):
        data = self._with_duplicates()
        floor = layer_merge_tree(data)["plateaus"][0]
        assert floor["k"] == data.n - 12  # the distinct vectors: token identity
        assert floor["trivial"]


class TestBlobPlusOutliers:
    """What layers >= 2 look like: one cluster with most tokens, a few strays."""

    def _blob(self, seed=1):
        rng = np.random.default_rng(seed)
        blob = np.eye(1, 16)[0] + 0.05 * rng.normal(size=(40, 16))
        strays = np.eye(3, 16)[1:]
        X = np.concatenate([blob, strays])
        return LayerData.from_normed((X / np.linalg.norm(X, axis=1, keepdims=True))
                                     .astype(np.float32))

    def test_longest_plateau_is_reported_as_one_substantial_cluster_plus_outliers(self):
        longest = layer_merge_tree(self._blob())["longest_any"]
        assert longest["k_substantial"] == 1 and longest["n_outliers"] == 2

    def test_robust_requires_two_substantial_clusters(self):
        tree = layer_merge_tree(self._blob())
        assert all(p["k_substantial"] >= 2 for p in tree["robust"])
        assert tree["longest_any"] not in tree["robust"]

    def test_substantial_labels_marks_small_clusters_as_outliers(self):
        labels = np.array([0] * 5 + [1] * 3 + [2] * 4 + [3])
        out = substantial_labels(labels, min_size=4)
        assert (out[:5] == 0).all() and (out[5:8] == -1).all()
        assert (out[8:12] == 2).all() and out[12] == -1


# ---------------------------------------------------------------------------
# Cross-layer linking
# ---------------------------------------------------------------------------

class TestLinkLayerPair:

    def test_identical_partitions_are_all_stable(self):
        labels = np.repeat([0, 1, 2], 5)
        assert link_layer_pair(labels, labels)["counts"] == _counts(stable=3)

    def test_two_clusters_folding_into_one_is_a_merge(self):
        prev = np.repeat([0, 1, 2], 5)
        curr = np.where(prev == 1, 0, prev)
        link = link_layer_pair(prev, curr)
        assert link["counts"] == _counts(stable=1, merge=1)
        comp = next(c for c in link["components"] if c["kind"] == "merge")
        assert comp["prev"] == [0, 1] and comp["curr"] == [0]

    def test_one_cluster_splitting_in_two_is_a_split(self):
        prev = np.repeat([0, 1, 2], 5)
        curr = prev.copy()
        curr[np.flatnonzero(prev == 0)[:2]] = 9
        link = link_layer_pair(prev, curr)
        comp = next(c for c in link["components"] if c["kind"] == "split")
        assert comp["prev"] == [0] and set(comp["curr"]) == {0, 9}

    def test_a_small_piece_leaving_a_large_cluster_is_a_split_under_containment(self):
        # 4 of 60 tokens leave: Jaccard 4/60 < 0.1 calls it a birth; the
        # piece is wholly inside its origin, so containment calls it a split.
        prev = np.zeros(60, dtype=int)
        curr = prev.copy()
        curr[:4] = 1
        assert link_layer_pair(prev, curr)["counts"] == _counts(split=1)
        assert link_layer_pair(prev, curr, measure="jaccard")["counts"] == \
            _counts(stable=1, birth=1)

    def test_outlier_tokens_belong_to_no_cluster(self):
        prev = np.array([0] * 6 + [1] * 6 + [-1, -1])
        curr = np.array([0] * 6 + [1] * 6 + [1, -1])
        link = link_layer_pair(prev, curr)
        assert link["counts"] == _counts(stable=2)
        assert (link["n_outliers_a"], link["n_outliers_b"]) == (2, 1)

    def test_every_edge_carries_both_measures(self):
        prev = np.zeros(20, dtype=int)
        curr = prev.copy()
        curr[:4] = 1
        edges = {(a, b): (j, c) for a, b, j, c in link_layer_pair(prev, curr)["edges"]}
        assert edges[(0, 1)] == pytest.approx((4 / 20, 1.0))
        assert edges[(0, 0)] == pytest.approx((16 / 20, 1.0))

    def test_jaccard_death_when_a_cluster_is_swamped(self):
        prev = np.array([0, 0, 0, 1, 1, 1, 2])
        curr = np.array([0, 0, 0, 1, 1, 1, 1])
        link = link_layer_pair(prev, curr, min_overlap=0.5, measure="jaccard")
        kinds = {tuple(c["prev"]): c["kind"] for c in link["components"]}
        assert kinds[(2,)] == "death"
        # containment sees cluster 2 absorbed whole: a merge, not a death
        assert link_layer_pair(prev, curr)["counts"] == _counts(stable=1, merge=1)

    def test_jaccard_birth_when_a_new_cluster_draws_little_from_its_source(self):
        prev = np.array([0, 0, 0, 1, 1, 1])
        curr = np.array([0, 0, 0, 1, 1, 2])
        link = link_layer_pair(prev, curr, min_overlap=0.5, measure="jaccard")
        assert any(c["curr"] == [2] and c["kind"] == "birth" for c in link["components"])

    def test_mutual_recombination_is_a_tangle(self):
        prev = np.array([0, 0, 1, 1])
        curr = np.array([0, 1, 0, 1])
        link = link_layer_pair(prev, curr, min_overlap=0.1)
        assert link["counts"] == _counts(tangle=1)

    def test_min_overlap_gates_weak_edges(self):
        # one straggler moves between two 8-token clusters: containment of
        # the straggler's origin in its destination is 1/8, under 0.5
        prev = np.array([0] * 8 + [1] * 8)
        curr = np.array([0] * 7 + [1] + [1] * 8)
        assert link_layer_pair(prev, curr)["counts"] == _counts(stable=2)

    def test_raises_on_mismatched_token_counts(self):
        with pytest.raises(ValueError, match="share token count"):
            link_layer_pair(np.array([0, 0, 1]), np.array([0, 1]))

    def test_raises_on_unknown_measure(self):
        with pytest.raises(ValueError, match="measure"):
            link_layer_pair(np.zeros(3), np.zeros(3), measure="cosine")


class TestAgreesWithClusterTracking:
    """
    Jaccard only: `match_layer_pair`'s vocabulary has no split, so the input
    has none. 101 tokens in disjoint blocks, so every overlap is exact by
    inspection: A, B unchanged (stable); C, D fold into one (merge); E sheds
    one straggler into a new cluster (birth, Jaccard 1/20); F absorbs the
    singleton I (death, Jaccard 1/21). Every ratio meant to pass or fail
    clears 0.1 by more than 2x.
    """

    def test_births_deaths_and_merge_groups_match(self):
        pytest.importorskip("scipy.optimize")
        from p1_mstate_tracking.cluster_tracking import match_layer_pair

        blocks = {"A": range(0, 20), "B": range(20, 40), "C": range(40, 50),
                  "D": range(50, 60), "E": range(60, 80), "F": range(80, 100),
                  "I": range(100, 101)}
        LAB = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "I": 6}
        prev = np.empty(101, dtype=int)
        for name, idx in blocks.items():
            prev[list(idx)] = LAB[name]
        curr = np.full(101, -99)
        curr[list(blocks["A"])] = 0
        curr[list(blocks["B"])] = 1
        curr[list(blocks["C"]) + list(blocks["D"])] = 2
        e = list(blocks["E"])
        curr[e[:-1]] = 4
        curr[e[-1:]] = 7
        curr[list(blocks["F"]) + list(blocks["I"])] = 5
        assert (curr != -99).all()

        theirs = match_layer_pair(prev, curr, min_jaccard=0.1)
        mine = link_layer_pair(prev, curr, min_overlap=0.1, measure="jaccard")
        comps = mine["components"]

        assert set(theirs["deaths"]) == {c["prev"][0] for c in comps
                                         if c["kind"] == "death"} == {LAB["I"]}
        assert set(theirs["births"]) == {c["curr"][0] for c in comps
                                         if c["kind"] == "birth"} == {7}
        their_merges = {frozenset(ids): cid for ids, cid in theirs["merges"]}
        my_merges = {frozenset(c["prev"]): c["curr"][0] for c in comps
                     if c["kind"] == "merge"}
        assert their_merges == my_merges == {frozenset({LAB["C"], LAB["D"]}): 2}
        # their Hungarian pass also lists one of C, D as M's primary match
        merged = {pid for ids, _ in theirs["merges"] for pid in ids}
        their_stable = {(a, b) for a, b, _ in theirs["matches"] if a not in merged}
        my_stable = {(c["prev"][0], c["curr"][0]) for c in comps if c["kind"] == "stable"}
        assert their_stable == my_stable == {
            (LAB["A"], 0), (LAB["B"], 1), (LAB["E"], 4), (LAB["F"], 5)}


# ---------------------------------------------------------------------------
# The chain over several layers
# ---------------------------------------------------------------------------

class TestLayerLinkChain:

    def test_boundaries_follow_the_layer_list_not_consecutive_integers(self):
        labels = np.repeat([0, 1, 2], 5)
        chain = layer_link_chain({0: labels, 12: labels, 18: labels})
        assert [(b["layer_from"], b["layer_to"]) for b in chain["boundaries"]] == \
            [(0, 12), (12, 18)]

    def test_a_layer_without_a_partition_breaks_the_chain_rather_than_being_bridged(self):
        labels = np.repeat([0, 1, 2], 5)
        chain = layer_link_chain({0: labels, 2: labels}, layers=[0, 1, 2])
        assert chain["n_skipped"] == 2
        assert all("skipped" in b for b in chain["boundaries"])
        assert chain["totals"] == _counts()

    def test_totals_sum_the_linked_boundaries(self):
        prev = np.repeat([0, 1, 2], 5)
        merged = np.where(prev == 1, 0, prev)
        chain = layer_link_chain({0: prev, 1: merged, 2: merged})
        assert chain["totals"] == _counts(stable=1 + 2, merge=1)
