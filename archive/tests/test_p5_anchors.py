"""
tests/test_p5_anchors.py — p5_single_mstate_analysis/anchors.py.

Builds real Phase 1 directory layouts on disk (JSON only — no npz, no torch)
so the readers are tested against the actual file shapes `p1_io._save_*`
writes, not against mocks of them.

The load-bearing test is TestB11Regression: it writes BOTH event schemas into
one run directory, the way a real Phase 1 run does, and asserts that merge
detection reads the one with `merges` in it.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.run_discovery import RunRef, discover_runs
from p5_single_mstate_analysis.anchors import (
    AnchorSpec,
    AnchorBundle,
    DEFAULT_ANCHORS,
    load_cluster_tracking,
    load_run_for_selection,
    build_anchor_token_sets,
    token_set_particle_table,
    bundle_report_lines,
)
from p5_single_mstate_analysis.token_sets import merge_event_for

import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure
N_TOKENS = 40
N_LAYERS = 12
CORE = [3, 4, 5, 6, 7, 8]
SIB = [20, 21, 22, 23, 24]


def _tracking_payload():
    """What cluster_tracking.track_clusters produces, after a JSON round-trip
    (tuples -> lists)."""
    return {
        "trajectories": [
            {"id": 7, "start_layer": 2, "end_layer": 9, "lifespan": 8,
             "chain": [[l, 7] for l in range(2, 10)]},
            {"id": 8, "start_layer": 2, "end_layer": 9, "lifespan": 8,
             "chain": [[l, 8] for l in range(2, 10)]},
        ],
        "events": [
            {"layer_from": 9, "layer_to": 10, "n_births": 0, "n_deaths": 0,
             "merges": [[[7, 8], 7]]},
        ],
        "summary": {"total_merges": 1, "max_alive": 2},
    }


def _write_run(root: Path, model: str, prompt: str,
               tracking=None, with_bridge_events=True,
               labels_override=None) -> Path:
    d = root / f"{model}_{prompt}"
    d.mkdir(parents=True, exist_ok=True)

    labels = {}
    if labels_override is not None:
        labels = labels_override
    else:
        for li in range(N_LAYERS):
            arr = [-1] * N_TOKENS
            if 2 <= li <= 10:
                for p in CORE:
                    arr[p] = 7
                for p in SIB:
                    arr[p] = 8
            labels[str(li)] = arr
    (d / "hdbscan_labels.json").write_text(json.dumps(labels))

    (d / "geometry.json").write_text(json.dumps({
        "model": model, "prompt": prompt,
        "tokens": [f"t{i}" for i in range(N_TOKENS)],
    }))

    (d / "trajectory.json").write_text(json.dumps({
        "model": model, "prompt": prompt, "plateau_layers": [4, 5],
        "cluster_tracking": _tracking_payload() if tracking is None else tracking,
    }))

    if with_bridge_events:
        # The Phase 3 bridge file — the one load_phase1_run reads.
        (d / "events.json").write_text(json.dumps({
            "merge_layers": [9],
            "energy_violations": {"1.0": [3, 4]},
        }))

    (d / "manifest.json").write_text(json.dumps({
        "model": model, "prompt_key": prompt,
        "checkpoint_step": int(model.rsplit("step", 1)[1]),
        "hf_revision": "step" + model.rsplit("step", 1)[1],
    }))
    return d


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------

class TestB11Regression(_Tmp):
    """load_phase1_run's `events` is the Phase 3 bridge file, not the
    cluster-tracking record. This is blocker 1's real cause."""

    def setUp(self):
        super().setUp()
        self.run = _write_run(self.tmp, "pythia-410m-step143000",
                              "wiki_paragraph")

    def _p1_io_style_events(self):
        """Reproduces p1_io._load_events exactly."""
        raw = json.loads((self.run / "events.json").read_text())
        out = []
        for layer in raw.get("merge_layers", []):
            out.append({"type": "merge", "layer_name": str(layer),
                        "layer_from": str(layer)})
        for beta, layers in raw.get("energy_violations", {}).items():
            for layer in layers:
                out.append({"type": "energy_violation", "layer": layer,
                            "beta": float(beta)})
        return out

    def test_bridge_events_have_no_merges_key_at_all(self):
        evs = self._p1_io_style_events()
        merge_evs = [e for e in evs if e["type"] == "merge"]
        self.assertTrue(merge_evs)
        for e in merge_evs:
            self.assertNotIn("merges", e)
            self.assertIsInstance(e["layer_from"], str)

    def test_merge_detection_against_bridge_events_raises_not_returns_none(self):
        # Tolerating the wrong schema is what hid blocker 1 for a whole
        # six-model study: every trajectory got None and nothing objected.
        traj = load_cluster_tracking(self.run)["trajectories"][0]
        with self.assertRaises(ValueError) as cm:
            merge_event_for(traj, self._p1_io_style_events())
        msg = str(cm.exception)
        self.assertIn("wrong event schema", msg)
        self.assertIn("load_cluster_tracking", msg)

    def test_empty_event_list_is_legitimate(self):
        traj = load_cluster_tracking(self.run)["trajectories"][0]
        self.assertIsNone(merge_event_for(traj, []))

    def test_merge_detection_against_real_events_succeeds(self):
        tracking = load_cluster_tracking(self.run)
        ev = merge_event_for(tracking["trajectories"][0], tracking["events"])
        self.assertIsNotNone(ev)
        self.assertEqual(ev["layer_from"], 9)
        self.assertEqual(sorted(ev["prev_ids"]), [7, 8])

    def test_both_files_coexist_which_is_why_this_went_unnoticed(self):
        self.assertTrue((self.run / "events.json").exists())
        self.assertTrue((self.run / "trajectory.json").exists())


class TestLoadClusterTracking(_Tmp):

    def setUp(self):
        super().setUp()
        self.run = _write_run(self.tmp, "pythia-410m-step512", "wiki_paragraph")

    def test_chain_entries_are_int_tuples_after_json_roundtrip(self):
        t = load_cluster_tracking(self.run)["trajectories"][0]
        self.assertIsInstance(t["chain"], tuple)
        for entry in t["chain"]:
            self.assertIsInstance(entry, tuple)
            self.assertIsInstance(entry[0], int)
            self.assertIsInstance(entry[1], int)

    def test_merges_normalised_to_list_int_pairs(self):
        ev = load_cluster_tracking(self.run)["events"][0]
        prev, curr = ev["merges"][0]
        self.assertEqual(prev, [7, 8])
        self.assertIsInstance(curr, int)

    def test_lifespan_definition_preserved_not_recomputed(self):
        # cluster_tracking defines lifespan = end - start + 1.
        t = load_cluster_tracking(self.run)["trajectories"][0]
        self.assertEqual(t["lifespan"], t["end_layer"] - t["start_layer"] + 1)

    def test_missing_file_returns_empty_not_raises(self):
        empty = self.tmp / "nothing"
        empty.mkdir()
        out = load_cluster_tracking(empty)
        self.assertEqual(out["trajectories"], [])
        self.assertEqual(out["events"], [])

    def test_corrupt_file_returns_empty_not_raises(self):
        (self.run / "trajectory.json").write_text("{broken")
        self.assertEqual(load_cluster_tracking(self.run)["trajectories"], [])

    def test_empty_chain_trajectories_dropped(self):
        payload = _tracking_payload()
        payload["trajectories"].append(
            {"id": 99, "start_layer": 0, "end_layer": 0,
             "lifespan": 1, "chain": []})
        _write_run(self.tmp, "pythia-410m-step1", "wiki_paragraph",
                   tracking=payload)
        t = load_cluster_tracking(self.tmp / "pythia-410m-step1_wiki_paragraph")
        self.assertEqual({x["id"] for x in t["trajectories"]}, {7, 8})


class TestLoadRunForSelection(_Tmp):

    def setUp(self):
        super().setUp()
        self.run = _write_run(self.tmp, "pythia-410m-step512", "wiki_paragraph")

    def test_labels_are_a_list_indexed_by_int_layer(self):
        run = load_run_for_selection(self.run)
        self.assertIsInstance(run["hdbscan_labels"], list)
        self.assertEqual(len(run["hdbscan_labels"]), N_LAYERS)
        self.assertEqual(run["hdbscan_labels"][3][CORE[0]], 7)

    def test_label_gaps_filled_with_noise(self):
        labels = {"0": [-1] * N_TOKENS, "5": [7] * N_TOKENS}
        d = _write_run(self.tmp, "pythia-410m-step2", "wiki_paragraph",
                       labels_override=labels)
        run = load_run_for_selection(d)
        self.assertEqual(len(run["hdbscan_labels"]), 6)
        self.assertTrue(np.all(run["hdbscan_labels"][3] == -1))

    def test_no_activations_loaded(self):
        run = load_run_for_selection(self.run)
        self.assertNotIn("activations", run)

    def test_missing_labels_returns_empty_dict(self):
        d = self.tmp / "bare"
        d.mkdir()
        self.assertEqual(load_run_for_selection(d), {})

    def test_n_tokens_and_tokens(self):
        run = load_run_for_selection(self.run)
        self.assertEqual(run["n_tokens"], N_TOKENS)
        self.assertEqual(len(run["tokens"]), N_TOKENS)


class TestDriver(_Tmp):

    def setUp(self):
        super().setUp()
        for step in (0, 512, 143000):
            for pk in ("wiki_paragraph", "sullivan_ballou"):
                _write_run(self.tmp, f"pythia-410m-step{step}", pk)
        _write_run(self.tmp, "pythia-410m-step143000", "repeated_tokens")
        self.refs = discover_runs(self.tmp, base="pythia-410m")

    def test_both_anchors_selected_for_each_prompt(self):
        b = build_anchor_token_sets(self.refs, min_fraction=0.9)
        names = {(ts.prompt_key, ts.name) for ts in b.token_sets}
        self.assertIn(("wiki_paragraph", "anchor_final"), names)
        self.assertIn(("wiki_paragraph", "anchor_init"), names)
        self.assertIn(("sullivan_ballou", "anchor_init"), names)

    def test_anchor_steps_are_the_requested_ones(self):
        b = build_anchor_token_sets(self.refs, min_fraction=0.9)
        got = b.by_prompt("wiki_paragraph")
        self.assertEqual(got["anchor_final"].anchor_step, 143000)
        self.assertEqual(got["anchor_init"].anchor_step, 0)

    def test_reject_prompt_skipped_with_reason(self):
        b = build_anchor_token_sets(self.refs, min_fraction=0.9)
        self.assertNotIn("repeated_tokens",
                         {ts.prompt_key for ts in b.token_sets})
        reasons = [s for s in b.skipped if s["prompt_key"] == "repeated_tokens"]
        self.assertTrue(reasons)
        self.assertIn("collapse control", reasons[0]["reason"])

    def test_missing_anchor_step_skipped_with_available_listed(self):
        refs = [r for r in self.refs if r.step != 0]
        b = build_anchor_token_sets(refs, min_fraction=0.9)
        skips = [s for s in b.skipped if s["anchor"] == "anchor_init"]
        self.assertTrue(skips)
        self.assertIn("143000", skips[0]["reason"])

    def test_overlap_computed_within_prompt(self):
        b = build_anchor_token_sets(self.refs, min_fraction=0.9)
        ov = [o for o in b.overlaps if o["prompt_key"] == "wiki_paragraph"]
        self.assertEqual(len(ov), 1)
        # Identical fixtures at both anchors -> identical token sets.
        self.assertEqual(ov[0]["jaccard"], 1.0)

    def test_controls_differ_between_anchors(self):
        b = build_anchor_token_sets(self.refs, min_fraction=0.9)
        got = b.by_prompt("wiki_paragraph")
        self.assertNotEqual(got["anchor_final"].control_positions,
                            got["anchor_init"].control_positions)

    def test_driver_is_reproducible(self):
        a = build_anchor_token_sets(self.refs, min_fraction=0.9)
        c = build_anchor_token_sets(self.refs, min_fraction=0.9)
        self.assertEqual([t.positions for t in a.token_sets],
                         [t.positions for t in c.token_sets])
        self.assertEqual([t.control_positions for t in a.token_sets],
                         [t.control_positions for t in c.token_sets])

    def test_selection_failure_becomes_a_skip_not_a_raise(self):
        payload = _tracking_payload()
        payload["trajectories"] = [
            {"id": 1, "start_layer": 0, "end_layer": 1, "lifespan": 2,
             "chain": [[0, 1], [1, 1]]}]
        payload["events"] = []
        _write_run(self.tmp, "pythia-410m-step99", "short_heterogeneous",
                   tracking=payload)
        refs = discover_runs(self.tmp, base="pythia-410m")
        b = build_anchor_token_sets(
            refs, anchors=[AnchorSpec("anchor_x", step=99)],
            prompt_keys=["short_heterogeneous"], min_fraction=0.9)
        self.assertEqual(b.token_sets, [])
        self.assertEqual(len(b.skipped), 1)
        self.assertIn("lifespan", b.skipped[0]["reason"])

    def test_custom_loader_injection(self):
        calls = []

        def fake(path):
            calls.append(path)
            return {}

        build_anchor_token_sets(self.refs, min_fraction=0.9, loader=fake)
        self.assertTrue(calls)

    def test_prompt_filter(self):
        b = build_anchor_token_sets(self.refs, prompt_keys=["wiki_paragraph"],
                                    min_fraction=0.9)
        self.assertEqual({ts.prompt_key for ts in b.token_sets},
                         {"wiki_paragraph"})

    def test_report_lists_skips(self):
        b = build_anchor_token_sets(self.refs, min_fraction=0.9)
        blob = "\n".join(bundle_report_lines(b))
        self.assertIn("token sets:", blob)
        self.assertIn("anchor overlap:", blob)
        self.assertIn("repeated_tokens", blob)

    def test_empty_bundle_report(self):
        self.assertEqual(bundle_report_lines(AnchorBundle()),
                         ["(no anchors attempted)"])


class TestParticleEmission(_Tmp):

    def setUp(self):
        super().setUp()
        _write_run(self.tmp, "pythia-410m-step143000", "wiki_paragraph")
        self.refs = discover_runs(self.tmp, base="pythia-410m")
        self.bundle = build_anchor_token_sets(
            self.refs, anchors=[AnchorSpec("anchor_final", step=143000)],
            min_fraction=0.9)
        self.ts = self.bundle.token_sets[0]
        self.ref = self.refs[0]
        self.run = load_run_for_selection(Path(self.ref.run_dir))

    def _stub_table(self):
        """Minimal stand-in with core.particles' from_layer/concat contract,
        so emission is tested without importing the real module."""
        class T:
            def __init__(self, cols):
                self.columns = cols

            @classmethod
            def from_layer(cls, model, prompt_key, layer, cluster_labels,
                           checkpoint_step=None, token_str=None, extra=None,
                           **kw):
                n = len(cluster_labels)
                cols = {
                    "model": np.array([model] * n),
                    "checkpoint_step": np.full(n, checkpoint_step or -1),
                    "prompt_key": np.array([prompt_key] * n),
                    "layer": np.full(n, layer),
                    "token_position": np.arange(n),
                    "cluster_label": np.asarray(cluster_labels),
                }
                for k, v in (extra or {}).items():
                    assert len(v) == n, f"{k} length {len(v)} != {n}"
                    cols[k] = np.asarray(v)
                return cls(cols)

            @classmethod
            def concat(cls, tables):
                keys = tables[0].columns
                return cls({k: np.concatenate([t.columns[k] for t in tables])
                            for k in keys})
        return T

    def test_every_token_at_every_layer_gets_a_row(self):
        tab = token_set_particle_table(self.ts, self.run, self.ref,
                                       ParticleTable=self._stub_table())
        self.assertEqual(len(tab.columns["token_position"]),
                         N_TOKENS * N_LAYERS)

    def test_roles_assigned_and_disjoint(self):
        tab = token_set_particle_table(self.ts, self.run, self.ref,
                                       ParticleTable=self._stub_table())
        roles = tab.columns["token_set_role"]
        first = roles[:N_TOKENS]
        self.assertEqual(set(first[list(self.ts.positions)]), {"primary"})
        self.assertEqual(set(first[list(self.ts.sibling_positions)]),
                         {"sibling"})
        self.assertEqual(set(first[list(self.ts.control_positions)]),
                         {"control"})

    def test_complement_is_retained_as_none(self):
        tab = token_set_particle_table(self.ts, self.run, self.ref,
                                       ParticleTable=self._stub_table())
        roles = tab.columns["token_set_role"][:N_TOKENS]
        self.assertIn("none", set(roles))

    def test_in_token_set_matches_primary_only(self):
        tab = token_set_particle_table(self.ts, self.run, self.ref,
                                       ParticleTable=self._stub_table())
        flags = tab.columns["in_token_set"][:N_TOKENS]
        self.assertEqual(sorted(np.where(flags == 1)[0].tolist()),
                         list(self.ts.positions))

    def test_checkpoint_step_carried(self):
        tab = token_set_particle_table(self.ts, self.run, self.ref,
                                       ParticleTable=self._stub_table())
        self.assertTrue(np.all(tab.columns["checkpoint_step"] == 143000))

    def test_mismatched_token_list_is_dropped_not_fatal(self):
        run = dict(self.run)
        run["tokens"] = ["a", "b"]
        tab = token_set_particle_table(self.ts, run, self.ref,
                                       ParticleTable=self._stub_table())
        self.assertEqual(len(tab.columns["token_position"]),
                         N_TOKENS * N_LAYERS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
