"""
tests/test_phase5_selection.py
Analytically-known-case tests for Phase 5 cluster selection.

select_primary_and_sibling and rank_trajectories use purely synthetic
trajectory dicts. No I/O, no model.

Hard gates from constants.py:
  MIN_LIFESPAN              = 6
  MIN_SIZE                  = 4  (at >= 75% of alive layers)
  REJECT_PROMPTS            = ("repeated_tokens",)
  MIN_SIZE_FRACTION_OF_ALIVE = 0.75
"""
import sys
import unittest
import numpy as np

sys.path.insert(0, "/home/claude")

from p5_single_mstate_analysis.select_cluster import (
    select_primary_and_sibling,
    rank_trajectories,
    _pick_sibling,
    _score_trajectory,
)
from p5_single_mstate_analysis import constants as C

# ---------------------------------------------------------------------------
# Synthetic Phase-1 run factory
#
# Phase-1 run shape:
#   {prompt_key: {
#       "trajectories":   list of trajectory dicts,
#       "events":         list of merge-event dicts,
#       "hdbscan_labels": list of (n_tokens,) int arrays per layer,
#       "metrics":        {"layers": []},
#   }}
#
# Trajectory shape:
#   {id, chain: [(layer, cid), ...], start_layer, end_layer, lifespan}
# ---------------------------------------------------------------------------

N_TOKENS      = 20
N_LAYERS      = C.MIN_LIFESPAN       # 6 — exactly the minimum
CLUSTER_SIZE  = 8                    # > C.MIN_SIZE (4)


def _make_traj(tid, cid, lifespan=N_LAYERS):
    return {
        "id":          tid,
        "chain":       [(l, cid) for l in range(lifespan)],
        "start_layer": 0,
        "end_layer":   lifespan - 1,
        "lifespan":    lifespan,
    }


def _make_labels(cluster_map, n_layers=N_LAYERS, n_tokens=N_TOKENS):
    """
    cluster_map: {cid: [token_indices]}
    Returns list of n_layers identical int arrays.
    """
    base = np.full(n_tokens, -1, dtype=int)
    for cid, idxs in cluster_map.items():
        base[np.array(idxs)] = cid
    return [base.copy() for _ in range(n_layers)]


def _standard_run(prompt_key="good_prompt", n_clusters=2, cluster_size=CLUSTER_SIZE):
    """Two (or more) equally-sized clusters that pass all hard gates."""
    cluster_map = {
        c: list(range(c * cluster_size, c * cluster_size + cluster_size))
        for c in range(n_clusters)
    }
    return {
        prompt_key: {
            "trajectories":   [_make_traj(c, c) for c in range(n_clusters)],
            "events":         [],
            "hdbscan_labels": _make_labels(cluster_map),
            "metrics":        {"layers": []},
        }
    }


# ---------------------------------------------------------------------------
# Hard-gate tests
# ---------------------------------------------------------------------------

class TestHardGates(unittest.TestCase):

    def test_short_lifespan_filtered_out(self):
        """lifespan < MIN_LIFESPAN must be excluded from ranked."""
        runs = _standard_run()
        short = C.MIN_LIFESPAN - 1
        runs["good_prompt"]["trajectories"].append(
            _make_traj(99, cid=5, lifespan=short)
        )
        # Give cluster 5 some tokens so size gate is not the cause
        for lbl in runs["good_prompt"]["hdbscan_labels"]:
            lbl[16:20] = 5
        ranked = rank_trajectories(runs)
        self.assertNotIn(99, [c["id"] for c in ranked],
            "Trajectory with lifespan < MIN_LIFESPAN should be excluded")

    def test_minimum_lifespan_exactly_passes(self):
        """lifespan == MIN_LIFESPAN is the boundary; it must pass."""
        runs   = _standard_run()  # lifespan = N_LAYERS = MIN_LIFESPAN
        ranked = rank_trajectories(runs)
        self.assertGreater(len(ranked), 0,
            "Trajectory at exactly MIN_LIFESPAN should survive the gate")

    def test_small_cluster_filtered_out(self):
        """Cluster with size < MIN_SIZE at all layers is excluded."""
        runs       = _standard_run()
        tiny_size  = C.MIN_SIZE - 1
        tiny_start = N_TOKENS - tiny_size
        runs["good_prompt"]["trajectories"].append(_make_traj(88, cid=7))
        for lbl in runs["good_prompt"]["hdbscan_labels"]:
            lbl[tiny_start:] = 7
        ranked = rank_trajectories(runs)
        self.assertNotIn(88, [c["id"] for c in ranked],
            "Cluster with size < MIN_SIZE should be excluded")

    def test_rejected_prompt_entirely_excluded(self):
        """All trajectories from REJECT_PROMPTS are filtered, giving empty list."""
        bad_key = C.REJECT_PROMPTS[0]   # "repeated_tokens"
        runs    = _standard_run(prompt_key=bad_key)
        ranked  = rank_trajectories(runs)
        self.assertEqual(ranked, [],
            f"Trajectories from '{bad_key}' should all be excluded")

    def test_rejected_prompt_raises_on_select(self):
        """select_primary_and_sibling raises RuntimeError when nothing passes."""
        bad_key = C.REJECT_PROMPTS[0]
        runs    = _standard_run(prompt_key=bad_key)
        with self.assertRaises(RuntimeError):
            select_primary_and_sibling(runs)


# ---------------------------------------------------------------------------
# Primary + sibling selection
# ---------------------------------------------------------------------------

class TestSelectPrimaryAndSibling(unittest.TestCase):

    def _run(self, **kw):
        return _standard_run(**kw)

    def test_result_has_expected_keys(self):
        result = select_primary_and_sibling(self._run())
        for key in ("ranked", "primary", "runner_up", "sibling"):
            self.assertIn(key, result)

    def test_primary_is_top_ranked(self):
        result = select_primary_and_sibling(self._run())
        self.assertEqual(result["primary"]["id"], result["ranked"][0]["id"])

    def test_primary_and_sibling_have_distinct_ids(self):
        result = select_primary_and_sibling(self._run())
        self.assertIsNotNone(result["primary"])
        self.assertIsNotNone(result["sibling"])
        self.assertNotEqual(result["primary"]["id"], result["sibling"]["id"])

    def test_runner_up_differs_from_primary(self):
        result = select_primary_and_sibling(self._run(n_clusters=3, cluster_size=6))
        if result["runner_up"] is not None:
            self.assertNotEqual(result["runner_up"]["id"], result["primary"]["id"])

    def test_ranked_list_is_sorted_descending(self):
        result = select_primary_and_sibling(self._run(n_clusters=3, cluster_size=6))
        scores = [c["total_score"] for c in result["ranked"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_forced_trajectory_selection(self):
        """force_trajectory_id overrides ranking."""
        runs   = self._run()
        result = select_primary_and_sibling(
            runs, force_prompt="good_prompt", force_trajectory_id=1
        )
        self.assertEqual(result["primary"]["id"], 1)

    def test_force_invalid_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            select_primary_and_sibling(
                self._run(),
                force_prompt="good_prompt",
                force_trajectory_id=999,
            )

    def test_multi_prompt_ranking(self):
        """rank_trajectories merges candidates from multiple prompt keys."""
        runs = {}
        runs.update(_standard_run(prompt_key="prompt_a"))
        runs.update(_standard_run(prompt_key="prompt_b"))
        ranked = rank_trajectories(runs)
        keys = {c["prompt_key"] for c in ranked}
        self.assertIn("prompt_a", keys)
        self.assertIn("prompt_b", keys)


# ---------------------------------------------------------------------------
# Sibling selection logic (_pick_sibling)
# ---------------------------------------------------------------------------

class TestPickSibling(unittest.TestCase):

    def test_contemporary_sibling_identified(self):
        """Two trajectories sharing all N_LAYERS => the other is the sibling."""
        runs  = _standard_run()
        trajs = runs["good_prompt"]["trajectories"]
        sib_id = _pick_sibling(trajs[0], trajs, events=[])
        self.assertEqual(sib_id, trajs[1]["id"])

    def test_no_sibling_when_only_one_trajectory(self):
        traj   = _make_traj(0, cid=0)
        sib_id = _pick_sibling(traj, [traj], events=[])
        self.assertIsNone(sib_id)

    def test_insufficient_overlap_not_chosen(self):
        """
        Candidate overlaps only C.MIN_LIFESPAN//2 - 1 layers with primary
        => should not be selected as sibling.
        """
        primary   = _make_traj(0, cid=0, lifespan=N_LAYERS)
        # overlap = 2 layers (layers 4,5); threshold = MIN_LIFESPAN//2 = 3
        candidate = {
            "id": 1, "chain": [(l, 1) for l in range(4, 9)],
            "start_layer": 4, "end_layer": 8, "lifespan": 5,
        }
        sib_id = _pick_sibling(primary, [primary, candidate], events=[])
        self.assertIsNone(sib_id,
            "Candidate with overlap < MIN_LIFESPAN//2 should not be sibling")

    def test_merge_partner_preferred_over_contemporary(self):
        """
        When a merge event exists, the merge partner is preferred over any
        generic contemporary with a larger layer overlap.
        """
        primary = {
            "id": 0,
            "chain": [(0,0),(1,0),(2,0),(3,3),(4,3),(5,3)],
            "start_layer": 0, "end_layer": 5, "lifespan": 6,
        }
        merge_partner = {
            "id": 2,
            "chain": [(0,2),(1,2),(2,2)],
            "start_layer": 0, "end_layer": 2, "lifespan": 3,
        }
        # Contemporary with larger overlap but NOT the merge partner
        big_contemporary = {
            "id": 9,
            "chain": [(l, 9) for l in range(N_LAYERS)],
            "start_layer": 0, "end_layer": 5, "lifespan": 6,
        }
        merge_event = {
            "layer_from": 2, "layer_to": 3,
            "merges": [([0, 2], 3)],
        }
        sib_id = _pick_sibling(
            primary,
            [primary, merge_partner, big_contemporary],
            events=[merge_event],
        )
        self.assertEqual(sib_id, 2,
            f"Merge partner (id=2) should be preferred; got {sib_id}")


# ---------------------------------------------------------------------------
# Score structure
# ---------------------------------------------------------------------------

class TestScoreStructure(unittest.TestCase):

    def _get_scored(self, prompt_key="good_prompt"):
        runs = _standard_run(prompt_key=prompt_key)
        run  = runs[prompt_key]
        return _score_trajectory(
            run["trajectories"][0], run["metrics"],
            run["hdbscan_labels"], run["events"],
            prompt_key, run["trajectories"],
        )

    def test_score_output_keys(self):
        scored = self._get_scored()
        for key in ("id", "lifespan", "mean_size", "total_score",
                    "sub_scores", "sibling_id"):
            self.assertIn(key, scored, f"Missing key '{key}'")

    def test_sub_scores_in_unit_interval(self):
        scored = self._get_scored()
        for name, val in scored["sub_scores"].items():
            self.assertGreaterEqual(val, 0.0, f"sub_score[{name}] < 0")
            self.assertLessEqual(val, 1.0, f"sub_score[{name}] > 1")

    def test_total_score_equals_weighted_sum(self):
        # _score_trajectory stores round(total, 3), so tolerate 3-decimal rounding.
        scored   = self._get_scored()
        expected = sum(C.SCORE_WEIGHTS[k] * v
                       for k, v in scored["sub_scores"].items())
        self.assertAlmostEqual(scored["total_score"], expected, delta=0.002)

    def test_preferred_prompt_scores_higher(self):
        """Preferred prompt yields a higher total_score than a neutral one."""
        preferred_key = C.PREFERRED_PROMPTS[0]
        scored_pref    = self._get_scored(prompt_key=preferred_key)
        scored_neutral = self._get_scored(prompt_key="neutral_prompt")
        self.assertGreater(scored_pref["total_score"], scored_neutral["total_score"],
            "Preferred prompt should yield higher total_score")

    def test_rejected_trajectory_is_none(self):
        bad_key = C.REJECT_PROMPTS[0]
        runs    = _standard_run(prompt_key=bad_key)
        run     = runs[bad_key]
        scored  = _score_trajectory(
            run["trajectories"][0], run["metrics"],
            run["hdbscan_labels"], run["events"],
            bad_key, run["trajectories"],
        )
        self.assertIsNone(scored,
            "_score_trajectory should return None for rejected prompts")


if __name__ == "__main__":
    unittest.main(verbosity=2)
