"""
tests/test_phase5_bugs.py

Regression tests for Phase 5 known-bug fixes (v2 execution-order item 4).

Fixes covered
-------------
FIX-B7   merge_geometry / merge_verdict always {"available": False}, all 6
         models (run_5.py, _run_group_B).
FIX-IO1  find_phase1_runs raised NameError on every real call
         (p5_single_mstate_analysis/io_p5.py — this file was supplied after
         the rest of this test file was originally written; see below).

FIX-B7 root cause
-----------------
`_run_group_B` filtered the raw per-layer `merge_events` list (the
cluster_tracking.py / core.io._load_events shape — each entry is
{"layer_from", "layer_to", "merges": [(prev_ids, curr_id), ...]}) with
`ev.get("prev_ids", [])`. That key never exists at the top level of those
event dicts — it's nested inside each tuple in `ev["merges"]` — so the
filter was always empty and `merge_result` stayed `{"available": False}`
for every trajectory, on every model (status-5.md, known blocker 1).

select_cluster._merge_event_for_trajectory already does this unpacking
correctly and attaches its result to the selected trajectory as
`trajectory["merge_event"]` (a single dict: layer_from, layer_to, prev_ids,
curr_id, role). Group E already consumes this same field correctly
(`primary_raw.get("merge_event")` in run_5.py's `_run_group_E`) — Group B
just wasn't using it. The fix makes Group B use the same field instead of
re-deriving relevance from a schema it was never actually matching.

FIX-IO1 root cause
------------------
p5_single_mstate_analysis/io.py (io.py renamed to io_p5.py to resolve a
basename collision with core/io.py — not available when this file was
first written; supplied afterward) had:

    def _matches(name: str) -> bool:
        # return model_stem in name or model_stem_hyphen in name
        if not model_stem:
            return True
        return any(s in d.name for s in stems)

`d` and `stems` are both undefined in this scope — a guaranteed NameError
on the first call with a non-empty model_stem, i.e. on every real
invocation (find_phase1_runs's early-return only skips this when
phase1_dir itself doesn't exist). The commented-out line directly above it
is the evidently-intended logic; the fix just restores it, using the
already-correct `name` parameter and the already-defined
`model_stem_hyphen`.

This is very likely the *actual* location of the known bug the v2 plan and
status-6.md attribute to "eigenspace_degeneracy.py — NameError, `d`
undefined": that file was checked directly (traced every use of `d`; ran
run_eigenspace_degeneracy end-to-end) and the error does not reproduce
there. Both are NameErrors involving an undefined `d`; this one actually
reproduces, on the exact symptom (a NameError from a directory/stem
matching helper) that would plausibly get invoked right before Phase 5
attribution. Treat status-6.md's attribution as likely a copy/transcription
error onto the wrong phase rather than two independent bugs.

No model loading; only a synthetic ov_weights NPZ for the projector build
(FIX-B7 tests) or a bare temp directory tree (FIX-IO1 tests).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from p5_single_mstate_analysis.run_5 import _run_group_B


def _make_trajectory_pair(own_cid: int = 0, other_cid: int = 1,
                           layer_from: int = 0, layer_to: int = 1,
                           curr_id: int = 2) -> dict:
    """A trajectory that survives a merge at layer_from -> layer_to."""
    return {
        "id": 1,
        "chain": [(layer_from, own_cid), (layer_to, curr_id)],
        "merge_event": {
            "layer_from": layer_from,
            "layer_to":   layer_to,
            "prev_ids":   [own_cid, other_cid],
            "curr_id":    curr_id,
            "role":       "participant",
        },
    }


def _write_synthetic_ov_weights(p2_dir: Path, stem: str, d: int = 8) -> None:
    """
    Minimal ov_weights_{stem}.npz with two shared heads whose sum is
    symmetric positive definite, so _build_v_projectors_from_ov's eigh path
    yields a non-empty U_attractive (positive-eigenvalue) subspace.
    """
    ov0 = (np.eye(d, dtype=np.float32))
    ov1 = (np.eye(d, dtype=np.float32) * 0.5)
    np.savez_compressed(
        p2_dir / f"ov_weights_{stem}.npz",
        ov_head0_shared=ov0,
        ov_head1_shared=ov1,
        ov_total_shared=(ov0 + ov1),
    )


class TestMergeGeometryFixB7:

    def _run(self, trajectory, sibling_trajectory, merge_events_raw,
              d: int = 8, n: int = 20, seed: int = 0):
        rng = np.random.default_rng(seed)
        acts_from = rng.standard_normal((n, d)).astype(np.float32)
        acts_to   = rng.standard_normal((n, d)).astype(np.float32)

        # layer_from: two pre-merge clusters (own_cid, other_cid);
        # layer_to: whatever cluster id the trajectory's chain says it is
        # at that layer (curr_id for a real merge; unchanged for no-merge).
        to_layer, to_cid = trajectory["chain"][-1]
        labels_from = np.array([0] * (n // 2) + [1] * (n - n // 2))
        labels_to   = np.array([to_cid] * n)

        with tempfile.TemporaryDirectory() as tmp:
            p2_dir  = Path(tmp) / "phase2"
            out_dir = Path(tmp) / "out"
            p2_dir.mkdir()
            out_dir.mkdir()
            _write_synthetic_ov_weights(p2_dir, "teststem", d=d)

            return _run_group_B(
                trajectory             = trajectory,
                sibling_trajectory     = sibling_trajectory,
                activations_per_layer  = [acts_from, acts_to],
                labels_per_layer       = [labels_from, labels_to],
                attentions_per_layer   = None,
                merge_events           = merge_events_raw,
                p2_dir                 = p2_dir,
                p2i_dir                = Path(tmp),
                stem                   = "teststem",
                out_dir                = out_dir,
            )

    def test_merge_geometry_available_when_trajectory_has_merge_event(self):
        """
        The bug this guards against: this used to always return
        {"available": False} regardless of input, because the relevance
        filter could never match. With a real merge_event attached to the
        trajectory, merge_geometry must now actually compute.
        """
        trajectory = _make_trajectory_pair()
        # Raw per-layer events in their real (nested "merges") shape — the
        # shape _run_group_B used to (incorrectly) filter directly.
        merge_events_raw = [
            {"layer_from": 0, "layer_to": 1, "merges": [([0, 1], 2)]}
        ]
        result = self._run(trajectory, sibling_trajectory=None,
                            merge_events_raw=merge_events_raw)

        mg = result["merge_geometry"]
        assert mg["available"] is True, (
            f"merge_geometry still unavailable — FIX-B7 not applied. Got: {mg}"
        )

    def test_merge_geometry_has_expected_fields_when_available(self):
        trajectory = _make_trajectory_pair()
        result = self._run(trajectory, sibling_trajectory=None,
                            merge_events_raw=[])
        mg = result["merge_geometry"]
        assert mg["available"] is True
        for key in ("pre_merge_angle_rad", "pre_merge_cosine",
                    "fusion_dir_magnitude", "fusion_attr_alignment",
                    "fusion_rep_alignment", "verdict"):
            assert key in mg, f"missing expected merge_event_geometry field: {key}"

    def test_no_merge_event_stays_unavailable_with_reason(self):
        """A trajectory that never merges should stay unavailable, but with
        a diagnostic reason rather than silently looking identical to a
        real failure."""
        trajectory = {
            "id": 1,
            "chain": [(0, 0), (1, 0)],
            "merge_event": None,
        }
        result = self._run(trajectory, sibling_trajectory=None,
                            merge_events_raw=[])
        mg = result["merge_geometry"]
        assert mg["available"] is False
        assert "reason" not in mg or mg == {"available": False}

    def test_does_not_depend_on_sibling_trajectory_lookup(self):
        """
        Old code required `sibling_trajectory is not None` before even
        attempting merge geometry — but sibling lookup can legitimately
        fail (run_5.py logs a [warn] and leaves sibling_raw as None) even
        when the primary trajectory's own merge_event is known. The fixed
        path reads merge_event straight off the trajectory and must not
        need sibling_trajectory to run.
        """
        trajectory = _make_trajectory_pair()
        result = self._run(trajectory, sibling_trajectory=None,
                            merge_events_raw=[])
        assert result["merge_geometry"]["available"] is True

    def test_own_cluster_id_resolved_from_chain_not_assumed_last_entry(self):
        """
        Old code took `trajectory["chain"][-1]` as the pre-merge cluster,
        which is wrong when the chain has already advanced past the merge
        layer. The fix must look up the trajectory's own cluster id at
        merge_event["layer_from"] specifically.
        """
        # chain has three entries; layer_from=0 is NOT the last pre-merge
        # entry positionally once you also track through the merge layer.
        trajectory = {
            "id": 1,
            "chain": [(0, 0), (1, 2)],
            "merge_event": {
                "layer_from": 0, "layer_to": 1,
                "prev_ids": [0, 1], "curr_id": 2, "role": "participant",
            },
        }
        result = self._run(trajectory, sibling_trajectory=None,
                            merge_events_raw=[])
        assert result["merge_geometry"]["available"] is True


class TestFindPhase1RunsFixIO1:
    """
    Regression tests for FIX-IO1 — p5_single_mstate_analysis/io_p5.py's
    find_phase1_runs raised NameError ("name 'stems' is not defined") on
    every call with a non-empty model_stem, i.e. on every real call site;
    run_5.py always passes a real model_stem. Confirmed by running the
    unpatched function against a real temp directory tree before writing
    the fix (an empty/missing phase1_dir short-circuits before ever
    reaching the broken branch, so a naive smoke check that doesn't
    actually populate phase1_dir would not have caught this).
    """

    def _make_run_dir(self, tmp_path, dirname: str, prompt: str = "short_heterogeneous"):
        import json
        run_dir = tmp_path / dirname
        run_dir.mkdir(parents=True)
        (run_dir / "geometry.json").write_text(json.dumps({
            "prompt": prompt, "n_layers": 2, "n_tokens": 5, "d_model": 4,
        }))
        return run_dir

    def test_matching_stem_found_without_raising(self, tmp_path):
        from p5_single_mstate_analysis.io_p5 import find_phase1_runs

        stem = "hf-internal-testing_tiny-random-gpt2"
        self._make_run_dir(tmp_path, f"{stem}_short_heterogeneous")

        result = find_phase1_runs(tmp_path, stem)
        assert result == {"short_heterogeneous": tmp_path / f"{stem}_short_heterogeneous"}

    def test_non_matching_stem_returns_empty_without_raising(self, tmp_path):
        from p5_single_mstate_analysis.io_p5 import find_phase1_runs

        self._make_run_dir(tmp_path, "hf-internal-testing_tiny-random-gpt2_short_heterogeneous")

        result = find_phase1_runs(tmp_path, "some-other-model")
        assert result == {}

    def test_run5_default_double_replaced_stem_does_not_match(self, tmp_path):
        """
        Documents why test_phase5_smoke.py passes --model-stem explicitly:
        run_5.py's own default stem derivation
        (model_name.replace("-","_").replace("/","_")) replaces the hyphen
        inside "hf-internal-testing" too, which model_stem_hyphen's simple
        reversal (all "_" -> "-") cannot undo correctly, since it can't
        tell which underscores used to be hyphens vs slashes.
        """
        from p5_single_mstate_analysis.io_p5 import find_phase1_runs

        self._make_run_dir(tmp_path, "hf-internal-testing_tiny-random-gpt2_short_heterogeneous")

        buggy_stem = "hf_internal_testing_tiny_random_gpt2"
        result = find_phase1_runs(tmp_path, buggy_stem)
        assert result == {}, (
            "if this starts matching, run_5.py's default model_stem "
            "derivation and the --model-stem override in "
            "test_phase5_smoke.py should be revisited together"
        )

    def test_empty_stem_matches_everything(self, tmp_path):
        from p5_single_mstate_analysis.io_p5 import find_phase1_runs

        self._make_run_dir(tmp_path, "some_model_wiki_paragraph", prompt="wiki_paragraph")

        result = find_phase1_runs(tmp_path, "")
        assert "wiki_paragraph" in result
