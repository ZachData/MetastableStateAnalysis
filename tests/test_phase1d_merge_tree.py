"""
tests/test_phase1d_merge_tree.py — the merge tree over scales (option D,
`lit-1d.md` §5) and cross-layer cluster linking by shared tokens.

TestAgreesWithClusterTracking exists to bound a duplication rather than to
check a computation: `merge_tree.link_layer_pair` is a fresh,
connected-components reading of the same kind of question
`p1_mstate_tracking.cluster_tracking.match_layer_pair` already answers with
Hungarian one-to-one matching (`merge_tree.py`'s module docstring explains
why the two are not the same code). On inputs built to have no split, the
two must still agree on which clusters are births, deaths, or part of a
merge group — if they diverge there, the two trackers are answering
different questions and something is wrong, not just differently phrased.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.deps

from sklearn.metrics import adjusted_rand_score

from p1d_cluster_ensemble.merge_tree import (
    MERGE_TREE_LINKAGES, labels_at_delta, layer_link_chain, layer_merge_tree,
    link_layer_pair,
)
from p1d_cluster_ensemble.methods import LayerData


def _planted_caps(n_per=10, d=16, noise=0.05, seed=0) -> LayerData:
    rng = np.random.default_rng(seed)
    truth = np.repeat([0, 1, 2], n_per)
    centers = np.eye(3, d)
    X = np.stack([centers[t] for t in truth]) + noise * rng.normal(size=(truth.size, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return LayerData.from_normed(X.astype(np.float32)), truth


# ---------------------------------------------------------------------------
# The merge tree
# ---------------------------------------------------------------------------

class TestLayerMergeTree:

    def test_plateaus_cover_delta_from_zero_to_inf_with_no_gap(self):
        data, _ = _planted_caps()
        tree = layer_merge_tree(data)
        plateaus = tree["plateaus"]
        assert plateaus[0]["delta_lo"] == 0.0
        assert plateaus[-1]["delta_hi"] == float("inf")
        for a, b in zip(plateaus, plateaus[1:]):
            assert a["delta_hi"] == b["delta_lo"], "no gap or overlap between plateaus"
        assert plateaus[0]["k"] == data.n
        assert plateaus[-1]["k"] == 1

    def test_planted_caps_give_k3_as_the_longest_lived_plateau(self):
        data, truth = _planted_caps()
        tree = layer_merge_tree(data)
        assert tree["robust"], "three well-separated caps must yield a robust plateau"
        top = tree["robust"][0]
        assert top["k"] == 3
        # every other robust plateau has a strictly shorter lifetime
        for other in tree["robust"][1:]:
            assert other["lifetime"] <= top["lifetime"]
        labels = labels_at_delta(tree["_Z"], data.n, top["delta_lo"])
        assert adjusted_rand_score(truth, labels) == pytest.approx(1.0)

    def test_ends_are_marked_trivial_and_excluded_from_robust(self):
        data, _ = _planted_caps()
        tree = layer_merge_tree(data)
        assert tree["plateaus"][0]["trivial"] and tree["plateaus"][0]["k"] == data.n
        assert tree["plateaus"][-1]["trivial"] and tree["plateaus"][-1]["k"] == 1
        assert all(not p["trivial"] for p in tree["robust"])

    def test_top_n_is_respected(self):
        data, _ = _planted_caps()
        tree = layer_merge_tree(data, top_n=2)
        assert len(tree["robust"]) <= 2

    @pytest.mark.parametrize("linkage", MERGE_TREE_LINKAGES)
    def test_every_supported_linkage_runs(self, linkage):
        data, _ = _planted_caps()
        tree = layer_merge_tree(data, linkage=linkage)
        assert tree["branch"] == "plateaus"
        assert tree["n_merges"] == data.n - 1

    def test_unsupported_linkage_raises_rather_than_silently_running(self):
        data, _ = _planted_caps()
        with pytest.raises(ValueError, match="ward"):
            layer_merge_tree(data, linkage="ward")

    @pytest.mark.parametrize("n", [0, 1])
    def test_degenerate_token_counts(self, n):
        X = np.zeros((n, 8), dtype=np.float32)
        data = LayerData(normed=X, cos_dist=np.zeros((n, n)))
        tree = layer_merge_tree(data)
        assert tree["branch"] == "n<2"
        assert tree["plateaus"] == [] and tree["robust"] == []

    def test_two_tokens_gives_one_trivial_merge_and_no_robust_plateau(self):
        X = np.stack([np.eye(1, 8, 0)[0], np.eye(1, 8, 1)[0]]).astype(np.float32)
        data = LayerData.from_normed(X)
        tree = layer_merge_tree(data)
        assert tree["branch"] == "two_tokens"
        assert [p["k"] for p in tree["plateaus"]] == [2, 1]
        assert tree["robust"] == []

    def test_cut_at_a_plateaus_own_delta_lo_reproduces_its_k(self):
        data, _ = _planted_caps()
        tree = layer_merge_tree(data)
        for p in tree["plateaus"]:
            labels = labels_at_delta(tree["_Z"], data.n, p["delta_lo"])
            assert len(set(labels.tolist())) == p["k"], p


# ---------------------------------------------------------------------------
# Cross-layer linking
# ---------------------------------------------------------------------------

class TestLinkLayerPair:

    def test_identical_partitions_are_all_stable(self):
        labels = np.repeat([0, 1, 2], 5)
        link = link_layer_pair(labels, labels)
        assert link["counts"] == {"stable": 3, "merge": 0, "split": 0,
                                  "tangle": 0, "birth": 0, "death": 0}

    def test_two_prev_clusters_folding_into_one_curr_is_a_merge(self):
        prev = np.repeat([0, 1, 2], 5)
        curr = prev.copy()
        curr[curr == 1] = 0
        link = link_layer_pair(prev, curr)
        assert link["counts"]["merge"] == 1
        assert link["counts"]["stable"] == 1
        merge_comp = next(c for c in link["components"] if c["kind"] == "merge")
        assert merge_comp["prev"] == [0, 1] and merge_comp["curr"] == [0]

    def test_one_prev_cluster_splitting_into_two_curr_is_a_split(self):
        prev = np.repeat([0, 1, 2], 5)
        curr = prev.copy()
        idx = np.flatnonzero(prev == 0)
        curr[idx[: len(idx) // 2]] = 9
        link = link_layer_pair(prev, curr)
        assert link["counts"]["split"] == 1
        split_comp = next(c for c in link["components"] if c["kind"] == "split")
        assert split_comp["prev"] == [0]
        assert set(split_comp["curr"]) == {0, 9}

    def test_a_cluster_with_no_curr_side_overlap_is_a_death_not_a_split(self):
        # prev has an extra singleton cluster that vanishes at curr (every
        # token it held is reassigned into an existing curr cluster below
        # the overlap floor is not constructible with one token; instead
        # drop the token's whole cluster by relabeling curr with one fewer
        # id and checking the unmatched prev id is a death).
        prev = np.array([0, 0, 0, 1, 1, 1, 2])
        curr = np.array([0, 0, 0, 1, 1, 1, 1])
        link = link_layer_pair(prev, curr, min_overlap=0.5)
        kinds = {tuple(c["prev"]): c["kind"] for c in link["components"]}
        assert kinds[(2,)] == "death"

    def test_a_curr_cluster_with_no_prev_side_overlap_is_a_birth(self):
        prev = np.array([0, 0, 0, 1, 1, 1])
        curr = np.array([0, 0, 0, 1, 1, 2])
        link = link_layer_pair(prev, curr, min_overlap=0.5)
        births = [c for c in link["components"] if c["kind"] == "birth"]
        assert any(c["curr"] == [2] for c in births)

    def test_mutual_recombination_is_a_tangle_not_forced_into_either_count(self):
        # two prev clusters (0,1) each split half their tokens into two
        # curr clusters (0,1) that each draw from both — a single
        # connected component with 2 prev and 2 curr nodes.
        prev = np.array([0, 0, 1, 1])
        curr = np.array([0, 1, 0, 1])
        link = link_layer_pair(prev, curr, min_overlap=0.1)
        assert link["counts"]["tangle"] == 1
        assert link["counts"]["merge"] == 0 and link["counts"]["split"] == 0

    def test_min_overlap_gates_weak_edges(self):
        # cluster 0 (8 tokens) sheds one straggler into cluster 1 (already
        # 8 tokens): the straggler's own Jaccard with its origin (1/16)
        # is well under 0.5, so it must not pull the two otherwise-clean
        # stable pairs into one tangled component.
        prev = np.array([0] * 8 + [1] * 8)
        curr = np.array([0] * 7 + [1] + [1] * 8)
        link = link_layer_pair(prev, curr, min_overlap=0.5)
        assert link["counts"] == {"stable": 2, "merge": 0, "split": 0,
                                  "tangle": 0, "birth": 0, "death": 0}

    def test_raises_on_mismatched_token_counts(self):
        with pytest.raises(ValueError, match="share token count"):
            link_layer_pair(np.array([0, 0, 1]), np.array([0, 1]))


class TestAgreesWithClusterTracking:
    """`p1_mstate_tracking.cluster_tracking.match_layer_pair`'s vocabulary
    has no split, so agreement is checked on inputs built to have none.

    Construction (101 tokens, disjoint index blocks so every overlap is
    exact by inspection rather than by trusting arithmetic on a shared
    block): A and B are unchanged (stable); C and D fold into one curr
    cluster M (merge); E sheds a single straggler that becomes its own
    new curr cluster N (birth) while the other 19 tokens stay E (stable);
    F gains a single straggler from the otherwise-untouched singleton
    prev cluster I, which has no curr-side edge above 0.1 Jaccard because
    its one token is swamped by F's 20 (death). Every non-trivial ratio
    here (19/20, 20/21) clears 0.1 comfortably, and every ratio meant to
    fail (1/20, 1/21) clears it by more than 2x margin, so this is not
    sensitive to the exact threshold.
    """

    def test_births_deaths_and_merge_groups_match(self):
        pytest.importorskip("scipy.optimize")
        from p1_mstate_tracking.cluster_tracking import match_layer_pair

        n = 101
        prev = np.empty(n, dtype=int)
        curr = np.empty(n, dtype=int)
        blocks = {"A": range(0, 20), "B": range(20, 40), "C": range(40, 50),
                 "D": range(50, 60), "E": range(60, 80), "F": range(80, 100),
                 "I": range(100, 101)}
        LAB = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "I": 6}
        for name, idx in blocks.items():
            prev[list(idx)] = LAB[name]

        curr[:] = -99
        curr[list(blocks["A"])] = 0                    # A -> A  (stable)
        curr[list(blocks["B"])] = 1                    # B -> B  (stable)
        curr[list(blocks["C"])] = 2                    # C,D -> M (merge)
        curr[list(blocks["D"])] = 2
        e = list(blocks["E"])
        curr[e[:-1]] = 4                                # 19 of E stay E (stable)
        curr[e[-1:]] = 7                                # 1 straggler -> N (birth)
        curr[list(blocks["F"])] = 5                     # F -> F (stable)
        curr[list(blocks["I"])] = 5                     # I's one token absorbed into F (death)
        assert (curr != -99).all()

        theirs = match_layer_pair(prev, curr, min_jaccard=0.1)
        mine = link_layer_pair(prev, curr, min_overlap=0.1)

        assert set(theirs["deaths"]) == {c["prev"][0] for c in mine["components"]
                                         if c["kind"] == "death"} == {LAB["I"]}
        assert set(theirs["births"]) == {c["curr"][0] for c in mine["components"]
                                         if c["kind"] == "birth"} == {7}
        their_merge_groups = {frozenset(ids): curr_id for ids, curr_id in theirs["merges"]}
        my_merge_groups = {frozenset(c["prev"]): c["curr"][0] for c in mine["components"]
                           if c["kind"] == "merge"}
        assert their_merge_groups == my_merge_groups == {frozenset({LAB["C"], LAB["D"]}): 2}
        # Their Hungarian pass picks one of {C, D} as M's "primary" match
        # and records it in `matches` too, alongside the merge group that
        # already names both — exclude any prev id that appears in a merge
        # group before comparing what is left to a *pure* one-to-one match.
        merged_prev_ids = {pid for ids, _ in theirs["merges"] for pid in ids}
        their_stable = {(a, b) for a, b, _ in theirs["matches"] if a not in merged_prev_ids}
        my_stable = {(c["prev"][0], c["curr"][0]) for c in mine["components"]
                    if c["kind"] == "stable"}
        assert their_stable == my_stable == {
            (LAB["A"], 0), (LAB["B"], 1), (LAB["E"], 4), (LAB["F"], 5)}


# ---------------------------------------------------------------------------
# The chain over several layers
# ---------------------------------------------------------------------------

class TestLayerLinkChain:

    def test_boundaries_are_between_sorted_neighbouring_keys_not_consecutive_integers(self):
        labels = np.repeat([0, 1, 2], 5)
        chain = layer_link_chain({0: labels, 12: labels, 18: labels})
        assert [(b["layer_from"], b["layer_to"]) for b in chain["boundaries"]] == \
               [(0, 12), (12, 18)]

    def test_totals_sum_the_boundaries(self):
        prev = np.repeat([0, 1, 2], 5)
        merged = prev.copy()
        merged[merged == 1] = 0
        chain = layer_link_chain({0: prev, 1: merged, 2: merged})
        assert chain["totals"]["merge"] == sum(
            b["counts"]["merge"] for b in chain["boundaries"])
        assert chain["totals"]["merge"] == 1
        assert chain["totals"]["stable"] == 1 + 2  # boundary 2 is identical->identical

    def test_fewer_than_two_layers_gives_no_boundaries(self):
        chain = layer_link_chain({0: np.array([0, 1])})
        assert chain["boundaries"] == [] and chain["totals"] == {
            k: 0 for k in ("stable", "merge", "split", "tangle", "birth", "death")}
