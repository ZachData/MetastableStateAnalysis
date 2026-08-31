"""
tests/test_p7_formation_curve.py — the checkpoint series P-I1 reads.

Two subjects. The first is the collapse of a head-PAIR-keyed statistic onto
the head axis `formation_gate.P_I1_UNIT` fixes, which is a definition and
so is a required argument here; the three choices are checked to be
genuinely different rather than rescalings of one another.

The second is the head axis itself, and it carries the bug this module was
written with and caught before it shipped. `per_head_relay_strength` is
SPARSE — a head with no relays is absent, not zero — while the behavioural
score is DENSE over every head in the attention tensor. Intersecting the
sparse side across checkpoints drops every head that had no relay at any
one of them, which is exactly the set of heads that go on to form.
Measured on the real sweep, that axis is empty: step 1000 has no relays at
all at the registered threshold while step 54000 has 78,024.
"""

import numpy as np
import pytest

from core.interactions import InteractionTable
from p7_motifs.formation_curve import (
    FormationCurveRefused,
    RELAY_OWNER_CHOICES,
    assert_gate_ready,
    behavioural_induction_score,
    formation_curve_payload,
    per_head_relay_strength,
)

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure

D = 4
U_POS = np.eye(D)[:, :2]
U_NEG = np.eye(D)[:, 2:]
ATTRACTIVE_F = np.array([1.0, 0.0, 0.0, 0.0])
REPULSIVE_F = np.array([0.0, 0.0, 1.0, 0.0])


def _edges(specs, checkpoint_step=1000):
    """(layer, head, target, source, kind, pair_type) -> InteractionTable,
    built through from_head as a real producer would."""
    groups = {}
    for layer, head, target, source, kind, pt in specs:
        groups.setdefault((layer, head), []).append((target, source, kind, pt))
    tables = []
    for (layer, head), rows in sorted(groups.items()):
        force = np.array([ATTRACTIVE_F if k == "a" else REPULSIVE_F
                          for _, _, k, _ in rows])
        tables.append(InteractionTable.from_head(
            model="m", prompt_key="p", layer=layer, head=head,
            targets=[t for t, _, _, _ in rows],
            sources=[s for _, s, _, _ in rows],
            weight=np.ones(len(rows)), force=force,
            U_pos=U_POS, U_neg=U_NEG,
            pair_type=[pt for _, _, _, pt in rows],
            checkpoint_step=checkpoint_step,
        ))
    return InteractionTable.concat(tables)


def _planted():
    """One relay: tag written by (2,0), matched by (7,3)."""
    return _edges([
        (2, 0, 5, 4, "a", "neither"),
        (7, 3, 9, 5, "a", "induction"),
    ])


class TestRelayOwner:

    def test_tag_writer_credits_the_first_stage(self):
        assert per_head_relay_strength(_planted(), "tag_writer") == {(2, 0): 1.0}

    def test_matcher_credits_the_second_stage(self):
        assert per_head_relay_strength(_planted(), "matcher") == {(7, 3): 1.0}

    def test_both_credits_each(self):
        assert per_head_relay_strength(_planted(), "both") == {(2, 0): 1.0,
                                                              (7, 3): 1.0}

    def test_the_three_are_not_rescalings_of_one_another(self):
        """The docstring's claim, checked: a head that writes a tag and
        matches nothing has strength under one choice and none under
        another, so no constant relates the three."""
        t = _planted()
        by = {o: per_head_relay_strength(t, o) for o in RELAY_OWNER_CHOICES}
        assert by["tag_writer"].get((7, 3), 0.0) == 0.0
        assert by["matcher"].get((2, 0), 0.0) == 0.0
        assert set(by["both"]) == set(by["tag_writer"]) | set(by["matcher"])

    def test_there_is_no_default(self):
        with pytest.raises(FormationCurveRefused, match="no default"):
            per_head_relay_strength(_planted(), "head_1")


class TestBehaviouralScore:

    def _attn(self):
        # (n_layers=2, n_heads=2, n_tokens=4, n_tokens=4)
        a = np.zeros((2, 2, 4, 4))
        a[1, 0, 3, 1] = 0.8          # layer 1 head 0 attends on the pair
        a[0, 1, 3, 1] = 0.2
        return a

    def test_mean_attention_on_induction_pairs(self):
        s = behavioural_induction_score(self._attn(), [(3, 1)])
        assert s[(1, 0)] == pytest.approx(0.8)
        assert s[(0, 1)] == pytest.approx(0.2)
        assert s[(0, 0)] == pytest.approx(0.0)

    def test_every_head_is_present(self):
        """Dense by construction — that is what makes it usable as the head
        axis in formation_curve_payload."""
        s = behavioural_induction_score(self._attn(), [(3, 1)])
        assert set(s) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_no_induction_pairs_returns_empty_not_zeros(self):
        """"this prompt carried no induction pairs" and "these heads attend
        to none of them" must not collapse."""
        assert behavioural_induction_score(self._attn(), []) == {}

    def test_a_pair_outside_the_matrix_is_refused(self):
        with pytest.raises(FormationCurveRefused, match="different tokenizations"):
            behavioural_induction_score(self._attn(), [(99, 1)])


class TestHeadAxis:

    def test_a_head_with_no_early_relays_survives(self):
        """The regression this module was written with. Under an
        intersection over the sparse relay maps this head is dropped and
        the curve that shows formation disappears with it."""
        pl = formation_curve_payload(
            [1000, 54000], [{}, {(5, 3): 78.0}],
            [{(5, 3): 0.01}, {(5, 3): 0.40}],
            independence_source="two_stage", relay_owner="matcher",
            above_null_excess=False)
        assert pl["heads"] == [[5, 3]]
        assert pl["motif_strength"] == [[0.0, 78.0]]

    def test_a_head_missing_from_the_behavioural_series_is_dropped(self):
        """Missing there means the checkpoint did not have the head, which
        is a geometry disagreement rather than a zero."""
        pl = formation_curve_payload(
            [1, 2], [{(0, 0): 1.0}, {(0, 0): 2.0}],
            [{(0, 0): 0.1, (9, 9): 0.5}, {(0, 0): 0.2}],
            independence_source="two_stage", relay_owner="both",
            above_null_excess=False)
        assert pl["heads"] == [[0, 0]]

    def test_no_shared_head_is_refused(self):
        with pytest.raises(FormationCurveRefused, match="head geometry"):
            formation_curve_payload(
                [1, 2], [{}, {}], [{(0, 0): 0.1}, {(1, 1): 0.2}],
                independence_source="two_stage", relay_owner="both",
                above_null_excess=False)


class TestPayloadContract:

    def _ok(self, **kw):
        args = dict(steps=[54000, 1000],
                    relay_by_step=[{(0, 0): 5.0}, {(0, 0): 1.0}],
                    score_by_step=[{(0, 0): 0.4}, {(0, 0): 0.1}],
                    independence_source="two_stage", relay_owner="matcher",
                    above_null_excess=False)
        args.update(kw)
        steps = args.pop("steps")
        return formation_curve_payload(
            steps, args.pop("relay_by_step"), args.pop("score_by_step"), **args)

    def test_required_contract_keys_are_present(self):
        from core.artifacts import get_spec
        pl = self._ok()
        for k in get_spec("phase7", "formation_curve").required_keys:
            assert k in pl

    def test_checkpoints_are_sorted_and_both_series_follow(self):
        """Handed 54000 before 1000: a curve that "rises" is a statement
        about the step axis, so the reordering has to carry both arms."""
        pl = self._ok()
        assert pl["checkpoint_steps"] == [1000, 54000]
        assert pl["motif_strength"] == [[1.0, 5.0]]
        assert pl["behavioral_induction_score"] == [[0.1, 0.4]]

    def test_the_two_decisions_are_stamped_into_the_artifact(self):
        pl = self._ok(relay_owner="both", independence_source="force_channel")
        assert pl["relay_owner"] == "both"
        assert pl["independence_source"] == "force_channel"

    def test_an_unregistered_independence_source_is_refused(self):
        with pytest.raises(FormationCurveRefused, match="independence_source"):
            self._ok(independence_source="obvious")

    def test_mismatched_series_lengths_are_refused(self):
        with pytest.raises(FormationCurveRefused, match="same checkpoints"):
            self._ok(relay_by_step=[{(0, 0): 1.0}])

    def test_one_checkpoint_is_refused(self):
        """P-I1 asks whether two curves rise together, which one point
        cannot show."""
        with pytest.raises(FormationCurveRefused, match="at least two"):
            self._ok(steps=[1000], relay_by_step=[{(0, 0): 1.0}],
                     score_by_step=[{(0, 0): 0.1}])


class TestGateReadiness:

    def test_a_raw_series_is_refused(self):
        """`formation_gate` requires the excess above N1/N2. No relay-count
        null exists here, so a raw count handed to p_value_p_i1 would report
        a p-value against a null the series never cleared."""
        pl = formation_curve_payload(
            [1, 2], [{(0, 0): 1.0}, {(0, 0): 2.0}],
            [{(0, 0): 0.1}, {(0, 0): 0.2}],
            independence_source="two_stage", relay_owner="matcher",
            above_null_excess=False)
        with pytest.raises(FormationCurveRefused, match="never cleared"):
            assert_gate_ready(pl)

    def test_an_above_null_series_passes(self):
        pl = formation_curve_payload(
            [1, 2], [{(0, 0): 1.0}, {(0, 0): 2.0}],
            [{(0, 0): 0.1}, {(0, 0): 0.2}],
            independence_source="two_stage", relay_owner="matcher",
            above_null_excess=True)
        assert assert_gate_ready(pl) is pl

    def test_the_flag_is_recorded_either_way(self):
        for flag in (True, False):
            pl = formation_curve_payload(
                [1, 2], [{(0, 0): 1.0}, {(0, 0): 2.0}],
                [{(0, 0): 0.1}, {(0, 0): 0.2}],
                independence_source="two_stage", relay_owner="matcher",
                above_null_excess=flag)
            assert pl["series_is_above_null_excess"] is flag
