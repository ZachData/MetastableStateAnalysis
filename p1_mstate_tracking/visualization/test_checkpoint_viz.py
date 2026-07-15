"""
tests/test_checkpoint_viz.py — pure-logic tests for the checkpoint-sweep
visualization modules (checkpoints.py, checkpoint_scalars.py,
checkpoint_filmstrip.py). No file I/O, no matplotlib rendering, no heavy
deps — same stub-heavy-deps session as the rest of the non-smoke suite.
"""
import numpy as np
import pytest

from p1_mstate_tracking.visualization.checkpoints import (
    _checkpoint_step, _checkpoint_base, checkpoint_families,
    family_baselines, _step_x, step_norm,
)
from p1_mstate_tracking.visualization.checkpoint_scalars import (
    _violation_layers_np, _violation_severity_np, _resampled_l2,
    detect_transitions, REL_TOL,
)
from p1_mstate_tracking.visualization.checkpoint_filmstrip import (
    select_snapshot_steps,
)


class TestNameGrammar:

    def test_step_parse(self):
        assert _checkpoint_step("pythia-410m-step2000") == 2000
        assert _checkpoint_base("pythia-410m-step2000") == "pythia-410m"
        assert _checkpoint_step("pythia-410m-step0") == 0

    def test_non_checkpoint_names_return_none(self):
        """The existing suffix grammar must never match: '-random',
        '@attn'/'@ffn', '@Niter', and plain model names all pass through."""
        for name in ("gpt2-large", "gpt2-large-random",
                     "albert-base-v2@24iter", "gpt2-large@attn",
                     "pythia-1.4b-random", "pythia-410m-stepX"):
            assert _checkpoint_step(name) is None
            assert _checkpoint_base(name) is None

    def test_family_grouping_sorted(self):
        models = ["pythia-410m-step512", "pythia-410m-step0",
                  "pythia-1.4b-step8", "gpt2-large", "pythia-410m-random"]
        fams = checkpoint_families(models)
        assert list(fams) == ["pythia-1.4b", "pythia-410m"]
        assert fams["pythia-410m"] == [(0, "pythia-410m-step0"),
                                       (512, "pythia-410m-step512")]

    def test_baselines_no_cross_family_substitution(self):
        """A different-size random control is a different object — the
        1.4b control must never be resolved for a 410m family."""
        models = ["pythia-410m-step0", "pythia-410m-step512",
                  "pythia-1.4b-random"]
        b = family_baselines("pythia-410m", models)
        assert b["random"] is None
        assert b["step0"] == "pythia-410m-step0"

    def test_baselines_resolve_when_present(self):
        models = ["pythia-410m-step0", "pythia-410m-random"]
        b = family_baselines("pythia-410m", models)
        assert b["random"] == "pythia-410m-random"
        assert b["step0"] == "pythia-410m-step0"


class TestStepAxis:

    def test_step_x_monotone_and_places_zero(self):
        xs = _step_x([0, 1, 512, 143000])
        assert xs[0] == 0.0
        assert np.all(np.diff(xs) > 0)

    def test_norm_ignores_step0(self):
        norm = step_norm([0, 1, 143000])
        assert norm.vmin == pytest.approx(_step_x([1])[0])


class TestViolationPort:
    """The numpy port must agree with metrics.energy_violation_severity's
    relative-drop rule at the shared REL_TOL."""

    def test_monotone_series_has_no_violations(self):
        assert _violation_layers_np([1.0, 1.1, 1.2, 1.3]) == []

    def test_relative_drop_fires_one_indexed(self):
        # drop of 0.5 rel. to |1.0| at transition into layer 2
        assert _violation_layers_np([1.0, 0.5, 0.6]) == [1 + 0]  # layer 1? no:
        # transition 0->1 is the drop; 1-indexed layer = 1
        n, sev, first = _violation_severity_np([1.0, 0.5, 0.6])
        assert n == 1 and first == 1.0 and sev == pytest.approx(0.5)

    def test_sub_threshold_drop_ignored(self):
        e = [1.0, 1.0 - REL_TOL * 0.5, 1.1]
        assert _violation_layers_np(e) == []

    def test_nan_endpoints_masked(self):
        assert _violation_layers_np([1.0, np.nan, 0.2]) == []


class TestDistanceAndTransitions:

    def test_resampled_l2_zero_for_identical(self):
        a = np.linspace(0, 1, 24)
        assert _resampled_l2(a, a) == pytest.approx(0.0)

    def test_resampled_l2_handles_length_mismatch(self):
        a = np.linspace(0, 1, 24)
        b = np.linspace(0, 1, 12)
        assert _resampled_l2(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_detect_transitions_finds_the_jump(self):
        steps = [0, 1, 8, 64, 512, 4000, 32000, 143000]
        vals = [0, 0, 0, 0, 0, 10, 10, 10]
        tr = detect_transitions(steps, vals)
        assert (tr["step_lo"], tr["step_hi"]) == (512, 4000)
        assert tr["normalized_jump"] > 0

    def test_detect_transitions_none_on_flat_or_short(self):
        assert detect_transitions([1, 2, 3], [5.0, 5.0, 5.0]) is None
        assert detect_transitions([1, 2], [0.0, 1.0]) is None
        assert detect_transitions([1, 2, 3], [np.nan, np.nan, 1.0]) is None


class TestSnapshotSelection:

    STEPS = [0, 1, 8, 64, 512, 2000, 4000, 8000, 40000, 143000]

    def test_endpoints_always_kept(self):
        picked = select_snapshot_steps(self.STEPS, None, k=4)
        assert 0 in picked and 143000 in picked

    def test_transition_brackets_prioritized(self):
        transitions = {
            "m1": {"step_lo": 2000, "step_hi": 4000, "normalized_jump": 1.0},
            "m2": {"step_lo": 2000, "step_hi": 4000, "normalized_jump": 0.8},
            "m3": {"step_lo": 0, "step_hi": 1, "normalized_jump": 0.9},
        }
        picked = select_snapshot_steps(self.STEPS, transitions, k=4)
        assert {0, 2000, 4000, 143000} <= set(picked)

    def test_k_is_a_cap(self):
        assert len(select_snapshot_steps(self.STEPS, None, k=6)) <= 6
        # fewer steps than k: return all
        assert select_snapshot_steps([0, 8, 16], None, k=6) == [0, 8, 16]
