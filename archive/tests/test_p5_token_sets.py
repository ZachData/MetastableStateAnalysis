"""
tests/test_p5_token_sets.py — p5_single_mstate_analysis/token_sets.py.

Pure numpy. No torch, no run directories — every function takes plain
trajectories/events/labels, so the fixtures are built inline.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from p5_single_mstate_analysis.token_sets import (
    TokenSet,
    SelectionRejected,
    MIN_CORE_SIZE,
    core_membership,
    merge_event_for,
    pick_sibling,
    score_trajectory,
    rank_trajectories,
    select_token_set,
    random_control_positions,
    anchor_overlap,
    save_token_sets,
    load_token_sets,
)

N_TOKENS = 40
N_LAYERS = 12


def _labels(assignments: dict) -> list:
    """assignments: {layer: {cluster_id: [positions]}} -> list of label arrays."""
    out = []
    for li in range(N_LAYERS):
        arr = np.full(N_TOKENS, -1, dtype=np.int32)
        for cid, positions in assignments.get(li, {}).items():
            arr[list(positions)] = cid
        out.append(arr)
    return out


def _traj(tid, chain):
    layers = [l for l, _ in chain]
    return {
        "id": tid,
        "chain": [(int(l), int(c)) for l, c in chain],
        "lifespan": len(chain),
        "start_layer": min(layers),
        "end_layer": max(layers),
    }


def _stable_fixture():
    """One stable 6-token cluster (id 7) over layers 2..9, plus a sibling
    (id 8) over layers 2..9, merging at 9->10. One token drifts out at two
    layers so core != union."""
    core = [3, 4, 5, 6, 7, 8]
    drifter = 9
    sib = [20, 21, 22, 23, 24]
    assign = {}
    for li in range(2, 10):
        members = list(core)
        if li not in (5, 6):          # drifter absent at 2 of 8 layers -> 0.75
            members = members + [drifter]
        assign[li] = {7: members, 8: sib}
    assign[10] = {7: core + sib}
    labels = _labels(assign)

    trajs = [
        _traj(7, [(l, 7) for l in range(2, 10)]),
        _traj(8, [(l, 8) for l in range(2, 10)]),
    ]
    events = [{
        "layer_from": 9, "layer_to": 10,
        "merges": [([7, 8], 7)],
    }]
    return trajs, events, labels, core, drifter, sib


class TestCoreMembership(unittest.TestCase):

    def setUp(self):
        self.trajs, self.events, self.labels, self.core, self.drift, _ = \
            _stable_fixture()

    def test_core_excludes_drifter_at_strict_fraction(self):
        mem = core_membership(self.trajs[0], self.labels, min_fraction=0.9)
        self.assertEqual(list(mem["core"]), self.core)
        self.assertIn(self.drift, mem["union"])

    def test_drifter_included_at_its_own_fraction(self):
        # present at 6 of 8 layers = 0.75 exactly; threshold is >=
        mem = core_membership(self.trajs[0], self.labels, min_fraction=0.75)
        self.assertIn(self.drift, mem["core"])

    def test_strict_intersection_can_be_empty_without_crashing(self):
        traj = _traj(1, [(0, 0), (1, 0), (2, 0)])
        labels = _labels({0: {0: [1, 2]}, 1: {0: [3, 4]}, 2: {0: [5, 6]}})
        mem = core_membership(traj, labels, min_fraction=1.0)
        self.assertEqual(mem["core"], ())
        self.assertEqual(len(mem["union"]), 6)
        self.assertAlmostEqual(mem["churn"], 1.0)

    def test_churn_zero_when_membership_is_constant(self):
        traj = _traj(1, [(0, 0), (1, 0)])
        labels = _labels({0: {0: [1, 2, 3]}, 1: {0: [1, 2, 3]}})
        self.assertEqual(core_membership(traj, labels)["churn"], 0.0)

    def test_chain_layers_beyond_label_array_are_skipped(self):
        traj = _traj(1, [(0, 0), (1, 0), (999, 0)])
        labels = _labels({0: {0: [1, 2, 3, 4]}, 1: {0: [1, 2, 3, 4]}})
        mem = core_membership(traj, labels)
        self.assertEqual(mem["n_layers"], 2)

    def test_positions_are_sorted_and_unique(self):
        mem = core_membership(self.trajs[0], self.labels)
        self.assertEqual(list(mem["core"]), sorted(set(mem["core"])))


class TestMergeAndSibling(unittest.TestCase):

    def setUp(self):
        self.trajs, self.events, self.labels, *_ = _stable_fixture()

    def test_merge_unpacks_nested_schema(self):
        # The FIX-B7 bug: the raw event has no top-level "prev_ids".
        self.assertNotIn("prev_ids", self.events[0])
        ev = merge_event_for(self.trajs[0], self.events)
        self.assertIsNotNone(ev)
        self.assertEqual(ev["layer_from"], 9)
        self.assertEqual(sorted(ev["prev_ids"]), [7, 8])
        self.assertEqual(ev["own_cluster_id"], 7)

    def test_no_merge_returns_none(self):
        self.assertIsNone(merge_event_for(self.trajs[0], []))

    def test_sibling_is_the_merge_partner(self):
        sib = pick_sibling(self.trajs[0], self.trajs, self.events)
        self.assertEqual(sib["id"], 8)

    def test_sibling_falls_back_to_contemporary_when_no_merge(self):
        sib = pick_sibling(self.trajs[0], self.trajs, [])
        self.assertEqual(sib["id"], 8)

    def test_sibling_none_when_alone(self):
        self.assertIsNone(pick_sibling(self.trajs[0], [self.trajs[0]], []))

    def test_sibling_is_deterministic_under_input_reordering(self):
        a = pick_sibling(self.trajs[0], self.trajs, [])
        b = pick_sibling(self.trajs[0], list(reversed(self.trajs)), [])
        self.assertEqual(a["id"], b["id"])


class TestScoring(unittest.TestCase):

    def setUp(self):
        self.trajs, self.events, self.labels, *_ = _stable_fixture()

    def test_passing_trajectory_scores(self):
        s = score_trajectory(self.trajs[0], self.trajs, self.events, self.labels)
        self.assertTrue(s["passed"])
        self.assertEqual(s["sub_scores"]["merge"], 1.0)
        self.assertEqual(s["sub_scores"]["sibling"], 1.0)
        self.assertLessEqual(s["total_score"], 1.0)

    def test_score_is_bounded_by_one(self):
        traj = _traj(1, [(l, 0) for l in range(N_LAYERS)])
        labels = _labels({l: {0: list(range(20))} for l in range(N_LAYERS)})
        others = [traj, _traj(2, [(l, 1) for l in range(N_LAYERS)])]
        s = score_trajectory(traj, others, [], labels)
        self.assertLessEqual(s["total_score"], 1.0)

    def test_short_trajectory_rejected_with_reason(self):
        traj = _traj(1, [(0, 0), (1, 0)])
        labels = _labels({0: {0: [1, 2, 3, 4]}, 1: {0: [1, 2, 3, 4]}})
        s = score_trajectory(traj, [traj], [], labels)
        self.assertFalse(s["passed"])
        self.assertIn("lifespan", s["reject_reason"])

    def test_tiny_core_rejected_with_union_and_churn_in_the_message(self):
        traj = _traj(1, [(l, 0) for l in range(6)])
        labels = _labels({l: {0: [l, l + 1]} for l in range(6)})
        s = score_trajectory(traj, [traj], [], labels)
        self.assertFalse(s["passed"])
        self.assertIn("core size", s["reject_reason"])
        self.assertIn("churn", s["reject_reason"])

    def test_ranking_is_deterministic_and_tallies_rejections(self):
        short = _traj(3, [(0, 5), (1, 5)])
        trajs = self.trajs + [short]
        a, tally_a = rank_trajectories(trajs, self.events, self.labels)
        b, tally_b = rank_trajectories(list(reversed(trajs)), self.events,
                                       self.labels)
        self.assertEqual([c["id"] for c in a], [c["id"] for c in b])
        self.assertEqual(tally_a, tally_b)
        self.assertEqual(tally_a.get("lifespan"), 1)


class TestRandomControl(unittest.TestCase):

    def test_disjoint_and_size_matched(self):
        ctrl = random_control_positions(5, N_TOKENS, exclude=[1, 2, 3], seed=0)
        self.assertEqual(len(ctrl), 5)
        self.assertFalse(set(ctrl) & {1, 2, 3})

    def test_seeded_and_reproducible(self):
        a = random_control_positions(5, N_TOKENS, [1, 2], seed=7)
        b = random_control_positions(5, N_TOKENS, [1, 2], seed=7)
        c = random_control_positions(5, N_TOKENS, [1, 2], seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_sorted(self):
        ctrl = random_control_positions(8, N_TOKENS, [], seed=3)
        self.assertEqual(list(ctrl), sorted(ctrl))

    def test_truncates_when_pool_too_small(self):
        ctrl = random_control_positions(10, 6, exclude=[0, 1, 2, 3], seed=0)
        self.assertEqual(len(ctrl), 2)

    def test_empty_pool(self):
        self.assertEqual(random_control_positions(3, 3, [0, 1, 2], seed=0), ())


class TestSelectTokenSet(unittest.TestCase):

    def setUp(self):
        self.trajs, self.events, self.labels, self.core, self.drift, self.sib = \
            _stable_fixture()

    def _select(self, **kw):
        args = dict(
            name="anchor_final", prompt_key="wiki_paragraph",
            anchor_model="pythia-410m-step143000", anchor_step=143000,
            anchor_run_dir="/runs/x", trajectories=self.trajs,
            events=self.events, hdb_labels=self.labels, n_tokens=N_TOKENS,
            min_fraction=0.9,
        )
        args.update(kw)
        return select_token_set(**args)

    def test_positions_are_the_core_set(self):
        ts = self._select()
        self.assertEqual(list(ts.positions), self.core)
        self.assertIn(self.drift, ts.union_positions)

    def test_sibling_positions_disjoint_from_primary(self):
        ts = self._select()
        self.assertFalse(set(ts.positions) & set(ts.sibling_positions))
        self.assertEqual(set(ts.sibling_positions), set(self.sib))

    def test_control_disjoint_from_primary_and_sibling(self):
        ts = self._select()
        self.assertEqual(len(ts.control_positions), len(ts.positions))
        self.assertFalse(set(ts.control_positions) & set(ts.positions))
        self.assertFalse(set(ts.control_positions) & set(ts.sibling_positions))

    def test_provenance_recorded(self):
        ts = self._select()
        self.assertEqual(ts.source_trajectory_id, 7)
        self.assertEqual(ts.anchor_step, 143000)
        self.assertEqual(ts.source_layers, tuple(range(2, 10)))
        self.assertIsNotNone(ts.merge_event)

    def test_reject_prompt_raises(self):
        with self.assertRaises(SelectionRejected) as cm:
            self._select(prompt_key="repeated_tokens")
        self.assertIn("collapse control", str(cm.exception))

    def test_empty_pool_raises_with_gate_tally(self):
        short = [_traj(1, [(0, 0), (1, 0)])]
        labels = _labels({0: {0: [1, 2, 3, 4]}, 1: {0: [1, 2, 3, 4]}})
        with self.assertRaises(SelectionRejected) as cm:
            self._select(trajectories=short, events=[], hdb_labels=labels)
        msg = str(cm.exception)
        self.assertIn("no trajectory passes", msg)
        self.assertIn("lifespan", msg)

    def test_force_trajectory_id(self):
        ts = self._select(force_trajectory_id=8)
        self.assertEqual(ts.source_trajectory_id, 8)

    def test_force_unknown_id_raises_listing_available(self):
        with self.assertRaises(SelectionRejected) as cm:
            self._select(force_trajectory_id=99)
        self.assertIn("[7, 8]", str(cm.exception))

    def test_rank_out_of_range_raises(self):
        with self.assertRaises(SelectionRejected):
            self._select(rank=5)

    def test_no_sibling_leaves_a_note(self):
        solo = [self.trajs[0]]
        ts = self._select(trajectories=solo, events=[])
        self.assertTrue(any("no sibling" in n for n in ts.notes))

    def test_annotation_uses_ext_semantic_tag_not_the_dead_alias(self):
        # B10: the producer emits tag in {same_cluster, diff_cluster, noise}.
        metrics = {"layers": [{} for _ in range(N_LAYERS)]}
        for li in range(2, 10):
            metrics["layers"][li] = {"pair_agreement": {"mutual_pairs": [
                {"i": 3, "j": 4, "cluster_i": 7, "cluster_j": 7,
                 "cross_method_tag": "same_cluster", "tag": "same_cluster",
                 "ext_semantic_tag": "ext_semantic"},
                {"i": 5, "j": 6, "cluster_i": 7, "cluster_j": 7,
                 "cross_method_tag": "same_cluster", "tag": "same_cluster",
                 "ext_semantic_tag": "ext_non_semantic"},
            ]}}
        ts = self._select(metrics=metrics)
        ann = ts.annotations
        self.assertAlmostEqual(ann["ext_semantic_frac__unfrozen_reference"], 0.5)
        self.assertEqual(ann["ext_semantic_n_pairs"], 16)
        self.assertIn("D6", ann["ext_semantic_caveat"])

    def test_annotation_is_none_when_no_pairs(self):
        ts = self._select()
        self.assertIsNone(
            ts.annotations["ext_semantic_frac__unfrozen_reference"])

    def test_semantic_is_not_a_score_term(self):
        ts = self._select()
        self.assertNotIn("semantic", ts.sub_scores)
        self.assertNotIn("preferred_prompt", ts.sub_scores)
        self.assertEqual(set(ts.sub_scores),
                         {"lifespan", "merge", "size", "sibling"})


class TestTokenSetInvariants(unittest.TestCase):

    def _ts(self, **kw):
        args = dict(name="a", prompt_key="p", anchor_model="m",
                    anchor_step=0, anchor_run_dir="/x",
                    positions=(1, 2, 3))
        args.update(kw)
        return TokenSet(**args)

    def test_unsorted_positions_rejected(self):
        with self.assertRaises(ValueError):
            self._ts(positions=(3, 1, 2))

    def test_duplicate_positions_rejected(self):
        with self.assertRaises(ValueError):
            self._ts(positions=(1, 1, 2))

    def test_control_overlapping_primary_rejected(self):
        with self.assertRaises(ValueError):
            self._ts(control_positions=(2, 9))

    def test_size(self):
        self.assertEqual(self._ts().size, 3)


class TestAnchorOverlap(unittest.TestCase):

    def _ts(self, name, positions, prompt="p"):
        return TokenSet(name=name, prompt_key=prompt, anchor_model="m",
                        anchor_step=0, anchor_run_dir="/x",
                        positions=tuple(sorted(positions)))

    def test_jaccard_and_containment(self):
        a = self._ts("anchor_final", [1, 2, 3, 4])
        b = self._ts("anchor_init", [3, 4, 5])
        o = anchor_overlap(a, b)
        self.assertEqual(o["n_intersection"], 2)
        self.assertAlmostEqual(o["jaccard"], 2 / 5, places=3)
        self.assertAlmostEqual(o["frac_of_a_in_b"], 0.5)
        self.assertAlmostEqual(o["frac_of_b_in_a"], 2 / 3, places=3)
        self.assertEqual(o["only_a"], (1, 2))
        self.assertEqual(o["only_b"], (5,))

    def test_disjoint(self):
        o = anchor_overlap(self._ts("x", [1, 2]), self._ts("y", [3, 4]))
        self.assertEqual(o["jaccard"], 0.0)

    def test_cross_prompt_refuses(self):
        with self.assertRaises(ValueError):
            anchor_overlap(self._ts("x", [1], "p1"), self._ts("y", [1], "p2"))


class TestPersistence(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip(self):
        trajs, events, labels, *_ = _stable_fixture()
        ts = select_token_set(
            name="anchor_final", prompt_key="wiki_paragraph",
            anchor_model="pythia-410m-step143000", anchor_step=143000,
            anchor_run_dir="/runs/x", trajectories=trajs, events=events,
            hdb_labels=labels, n_tokens=N_TOKENS, min_fraction=0.9,
        )
        p = save_token_sets([ts], self.tmp / "token_sets.json")
        back = load_token_sets(p)
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0].positions, ts.positions)
        self.assertEqual(back[0].sub_scores, ts.sub_scores)
        self.assertEqual(back[0].source_layers, ts.source_layers)

    def test_bad_schema_raises(self):
        p = self.tmp / "bad.json"
        p.write_text(json.dumps({"schema": "nope", "token_sets": []}))
        with self.assertRaises(ValueError):
            load_token_sets(p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
