"""
tests/test_p7_events.py — p7_motifs/events.py.

The event level is where a motif stops being a routing pattern and becomes
a claim about particle motion, so the tests concentrate on the two places
that claim can be faked:

  1. `moved_fraction`'s definition. A magnitude ratio would score a large
     force orthogonal to the actual motion as highly explanatory. The
     signed projection must read ~0 there and negative when the motif
     pushed against the direction the particle went.
  2. The absent-vs-false distinction. "P-I4 was not run" and "P-I4 found
     nothing" must not produce the same table.
"""
from __future__ import annotations

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pytest.ini [pytest] markers.
pytestmark = pytest.mark.pure

from core.particles import ParticleTable
from p7_motifs.events import (
    annotate,
    moved_fraction,
    relay_target_flags,
    transition_events,
)
from p7_motifs.motif_alphabet import RelayInstance


def _table(labels_per_layer, prompt="p", model="m", step=1000):
    """labels_per_layer: list (one per layer) of per-token cluster labels."""
    return ParticleTable.concat([
        ParticleTable.from_layer(model=model, prompt_key=prompt, layer=layer,
                                 cluster_labels=labels, checkpoint_step=step)
        for layer, labels in enumerate(labels_per_layer)
    ])


class TestTransitionEvents:

    def test_capture_fires_on_unclustered_to_clustered(self):
        t = _table([[-1, -1], [0, -1]])
        ev = transition_events(t)
        cap = ev["capture"][(t.columns["layer"] == 1) & (t.columns["token_position"] == 0)]
        assert cap[0]

    def test_escape_fires_on_clustered_to_unclustered(self):
        t = _table([[0, 0], [-1, 0]])
        ev = transition_events(t)
        esc = ev["escape"][(t.columns["layer"] == 1) & (t.columns["token_position"] == 0)]
        assert esc[0]

    def test_no_event_when_membership_is_unchanged(self):
        t = _table([[0, -1], [0, -1]])
        ev = transition_events(t)
        at_l1 = t.columns["layer"] == 1
        assert not ev["capture"][at_l1].any()
        assert not ev["escape"][at_l1].any()

    def test_layer_zero_has_no_transition_but_is_kept(self):
        """Dropping layer 0 would misalign every series built alongside it
        — the same reasoning UPDATE_PLAN 5.9 gives for the violation
        indicator."""
        t = _table([[-1, 0], [0, 0]])
        ev = transition_events(t)
        at_l0 = t.columns["layer"] == 0
        assert at_l0.sum() == 2                      # rows kept
        assert not ev["capture"][at_l0].any()
        assert not ev["escape"][at_l0].any()

    def test_hold_marks_every_unclustered_particle(self):
        t = _table([[-1, 0], [-1, 0]])
        ev = transition_events(t)
        assert ev["hold"].sum() == 2

    def test_hold_run_counts_consecutive_unclustered_layers(self):
        """Phase 5c's noise_tracking primitive: 'this token has been
        unclustered for N consecutive layers'."""
        t = _table([[-1], [-1], [-1], [0], [-1]])
        ev = transition_events(t)
        order = np.argsort(t.columns["layer"])
        assert ev["hold_run"][order].tolist() == [1, 2, 3, 0, 1]

    def test_hold_run_resets_after_a_capture(self):
        t = _table([[-1], [-1], [0], [0]])
        ev = transition_events(t)
        order = np.argsort(t.columns["layer"])
        assert ev["hold_run"][order].tolist() == [1, 2, 0, 0]

    def test_particles_are_matched_by_identity_not_row_order(self):
        """A ParticleTable is a concat in no guaranteed layer order."""
        forward = _table([[-1, 0], [0, 0]])
        reversed_ = ParticleTable.concat([
            ParticleTable.from_layer(model="m", prompt_key="p", layer=1,
                                     cluster_labels=[0, 0], checkpoint_step=1000),
            ParticleTable.from_layer(model="m", prompt_key="p", layer=0,
                                     cluster_labels=[-1, 0], checkpoint_step=1000),
        ])

        def _capture_at(tbl):
            ev = transition_events(tbl)
            sel = (tbl.columns["layer"] == 1) & (tbl.columns["token_position"] == 0)
            return bool(ev["capture"][sel][0])

        assert _capture_at(forward) == _capture_at(reversed_) is True

    def test_distinct_token_positions_do_not_bleed_into_each_other(self):
        t = _table([[-1, 0], [0, -1]])
        ev = transition_events(t)
        at_l1 = t.columns["layer"] == 1
        pos = t.columns["token_position"][at_l1]
        assert ev["capture"][at_l1][pos == 0][0]      # -1 -> 0
        assert ev["escape"][at_l1][pos == 1][0]       # 0 -> -1

    def test_empty_table_returns_empty_arrays(self):
        ev = transition_events(ParticleTable.concat([]))
        assert all(len(v) == 0 for v in ev.values())


class TestRelayTargetFlags:

    def test_tag_particle_is_flagged_at_the_stage_one_layer(self):
        t = _table([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])
        r = RelayInstance(layer_1=1, head_1=0, layer_2=2, head_2=3,
                          tag_position=2, match_target=0)
        flags = relay_target_flags(t, [r])
        sel = (t.columns["layer"] == 1) & (t.columns["token_position"] == 2)
        assert flags[sel][0]
        assert flags.sum() == 1

    def test_flag_is_not_smeared_across_every_layer(self):
        """Marking the particle at every depth would make the flag a
        property of the token rather than of the event."""
        t = _table([[-1, -1, -1]] * 4)
        r = RelayInstance(layer_1=1, head_1=0, layer_2=3, head_2=1,
                          tag_position=2, match_target=0)
        assert relay_target_flags(t, [r]).sum() == 1

    def test_no_relays_gives_no_flags(self):
        t = _table([[-1, -1]] * 2)
        assert relay_target_flags(t, []).sum() == 0


class TestMovedFraction:
    """P-I4's readout, and the place a motif can look causal without being."""

    def test_force_aligned_with_motion_reads_one(self):
        d = np.array([[2.0, 0.0]])
        assert moved_fraction(d, d)[0] == pytest.approx(1.0)

    def test_force_orthogonal_to_motion_reads_zero(self):
        """The case a magnitude ratio gets wrong: a large force that moved
        the particle nowhere along its actual path."""
        disp = np.array([[1.0, 0.0]])
        force = np.array([[0.0, 50.0]])       # huge, and orthogonal
        assert moved_fraction(disp, force)[0] == pytest.approx(0.0)

    def test_magnitude_ratio_would_have_scored_that_case_high(self):
        """Pins why the definition is a projection, not ||f|| / ||d||."""
        disp = np.array([[1.0, 0.0]])
        force = np.array([[0.0, 50.0]])
        naive = np.linalg.norm(force) / np.linalg.norm(disp)
        assert naive == pytest.approx(50.0)
        assert moved_fraction(disp, force)[0] == pytest.approx(0.0)

    def test_force_opposing_motion_is_negative_not_clipped(self):
        """A motif that pushed against where the particle went is a real
        outcome and must be reportable as such."""
        disp = np.array([[1.0, 0.0]])
        force = np.array([[-0.5, 0.0]])
        assert moved_fraction(disp, force)[0] == pytest.approx(-0.5)

    def test_partial_alignment_reads_the_projected_share(self):
        disp = np.array([[2.0, 0.0]])
        force = np.array([[1.0, 3.0]])
        assert moved_fraction(disp, force)[0] == pytest.approx(0.5)

    def test_motionless_particle_is_nan_not_zero(self):
        """'What fraction of no movement' has no answer; 0.0 would read as
        'the motif explained none of it'."""
        out = moved_fraction(np.zeros((1, 3)), np.ones((1, 3)))
        assert np.isnan(out[0])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            moved_fraction(np.zeros((2, 3)), np.zeros((2, 4)))

    def test_accepts_single_vectors(self):
        assert moved_fraction(np.array([1.0, 0.0]), np.array([1.0, 0.0])).shape == (1,)


class TestAnnotate:

    def _t(self):
        return _table([[-1, 0], [0, 0]])

    def test_transition_columns_are_always_attached(self):
        out = annotate(self._t())
        for c in ("capture", "hold", "hold_run", "escape"):
            assert c in out.extra

    def test_relay_target_absent_when_no_relays_supplied(self):
        """Absent means 'not computed'. An all-False column would mean
        'computed, none found' — the difference between 'P-I4 was not run'
        and 'P-I4 failed'."""
        assert "relay_target" not in annotate(self._t()).extra

    def test_relay_target_present_and_all_false_when_relays_is_empty(self):
        out = annotate(self._t(), relays=[])
        assert "relay_target" in out.extra
        assert not out.extra["relay_target"].any()

    def test_moved_fraction_absent_unless_both_inputs_given(self):
        assert "moved_fraction" not in annotate(self._t()).extra

    def test_supplying_only_one_of_the_pair_raises(self):
        t = self._t()
        with pytest.raises(ValueError, match="must be supplied together"):
            annotate(t, displacement=np.zeros((len(t), 3)))

    def test_moved_fraction_length_is_checked_against_the_table(self):
        t = self._t()
        with pytest.raises(ValueError, match="expected"):
            annotate(t, displacement=np.zeros((2, 3)), motif_force=np.zeros((2, 3)))

    def test_annotated_table_still_saves_and_loads(self, tmp_path):
        """The whole point of using extra__ columns: the result is still a
        ParticleTable and still round-trips under allow_pickle=False."""
        t = self._t()
        out = annotate(t, relays=[],
                       displacement=np.ones((len(t), 3)),
                       motif_force=np.ones((len(t), 3)))
        p = tmp_path / "annotated.npz"
        out.save(p)
        back = ParticleTable.load(p)
        assert np.allclose(back.extra["moved_fraction"], 1.0)
        assert "capture" in back.extra

    def test_existing_extra_columns_are_preserved(self):
        t = self._t()
        t.extra["prior"] = np.arange(len(t))
        assert "prior" in annotate(t).extra
