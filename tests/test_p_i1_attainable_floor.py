"""
tests/test_p_i1_attainable_floor.py — the two halves of the pairing arm's
floor, and the axis P-I1's payload actually builds.

`tools/p_i1_attainable_floor.py` is step 1 of `claims/EVALUABILITY.md`'s order
for P-I1's relay-count null: compute the attainable floor, before the control.
It found two things and they are independent, so they are pinned separately.

WHAT EACH GROUP EXISTS TO CATCH

`TestPairingFloorReport` is the arithmetic. `prod(m!) / n!` is a claim about the
statistic's invariance -- permuting units within a class of equal locations
leaves `-mean|ca - cb[p]|` exactly unchanged -- and the tests check the claim
against the enumerated group rather than against a second copy of the formula.
Revert `pairing_floor_report` to `1 / n_draws` and
`test_the_tie_floor_is_the_enumerated_tie_fraction` fails.

`TestTheArmRefusesOnTies` is the fix, and it is a REFUSAL, so both halves of a
refusal have to be pinned: that it fires where the design cannot reject, and
that it costs nothing where it can. Two heads either side of alpha, because the
boundary is where a refusal is wrong if it is wrong.

`TestTheDenseAxisCannotBeScored` is the wiring, and it is the one that is not
about floors at all. `formation_curve_payload` takes its head axis from the
behavioural series -- dense over every head the model has -- and zero-fills the
relay side, but the arm profiles every unit with no per-unit skip. So a model
with any head that never carries a relay yields no p-value. Pinned on an
eight-head toy rather than on the 384-head sweep, because it is a property of
the axis rule and reproduces at any size.
"""

from __future__ import annotations

import json
import math
from itertools import permutations

import numpy as np
import pytest

from core.changepoint_colocation import (
    ColocationRefused,
    REGISTERED_P_I1_SWEEP,
    change_profile,
    paired_colocation_arm,
    pairing_floor_report,
)
from p7_motifs.formation_curve import formation_curve_payload
from p7_motifs.formation_gate import P_I1_RELAY_OWNER, p_value_p_i1

pytestmark = pytest.mark.pure

ALPHA = 0.05
STEPS = list(REGISTERED_P_I1_SWEEP)
N_INTERVALS = len(STEPS) - 1


def _series_at(j: int) -> list:
    """A rise whose entire change mass sits in interval `j`."""
    v = np.zeros(len(STEPS), dtype=np.float64)
    v[j + 1:] = 1.0
    return v.tolist()


def _locations(classes) -> np.ndarray:
    return np.asarray([change_profile(STEPS, _series_at(j), "rise")
                       ["centroid_log_step"] for j in classes])


def _arm(classes, alpha: float = ALPHA) -> dict:
    """The arm on a PERFECT input at the given tie structure."""
    a = [_series_at(j) for j in classes]
    b = [[2.0 * x for x in v] for v in a]      # same locations, different series
    return paired_colocation_arm(STEPS, a, "rise", b, "rise", alpha=alpha,
                                 unit_name="head", arm_name="test")


class TestPairingFloorReport:

    def test_the_tie_floor_is_the_enumerated_tie_fraction(self):
        """
        Not checked against a second copy of `prod(m!)/n!` -- checked against
        the fraction of the ENUMERATED pairing group that reproduces a perfect
        input's statistic. That is what the formula claims to count, and the
        two agreeing is the argument.
        """
        for classes in ([0, 0, 0, 1, 2], [0, 0, 1, 1, 2], [0, 1, 2, 3, 4],
                        [0, 0, 0, 0, 1, 2]):
            c = _locations(classes)
            n = len(classes)
            obs = -float(np.mean(np.abs(c - c)))
            ties = sum(1 for p in permutations(range(n))
                       if -float(np.mean(np.abs(c - c[list(p)]))) >= obs - 1e-15)
            rep = pairing_floor_report(c, c.copy(), math.factorial(n), ALPHA,
                                       True)
            assert rep["tie_floor"] == pytest.approx(ties / math.factorial(n))

    def test_all_distinct_locations_make_the_draw_count_bind(self):
        c = _locations(list(range(10)))
        rep = pairing_floor_report(c, c.copy(), 2001, ALPHA, False)
        assert rep["binds"] == "draws"
        assert rep["attainable_floor"] == pytest.approx(1 / 2001)
        assert rep["tie_floor"] < rep["draw_count_floor"]

    def test_a_tied_class_makes_the_ties_bind_and_more_draws_do_not_help(self):
        """
        The point of separating the halves. The tie fraction contains no draw
        count, so a null sampled a thousand times harder returns the same
        floor -- which is why the refusal's message says so.
        """
        c = _locations([0] * 9 + [5])
        low = pairing_floor_report(c, c.copy(), 2001, ALPHA, False)
        high = pairing_floor_report(c, c.copy(), 2_000_001, ALPHA, False)
        assert low["binds"] == high["binds"] == "ties"
        assert low["attainable_floor"] == pytest.approx(0.1)
        assert high["attainable_floor"] == pytest.approx(0.1)
        assert not low["sufficient"] and not high["sufficient"]

    def test_either_side_can_be_the_one_carrying_the_ties(self):
        """
        The invariance is symmetric: permuting within equal `ca` and within
        equal `cb` both leave the statistic alone. A report that only looked at
        the A side would miss a degenerate behavioural series entirely.
        """
        tied = _locations([0] * 9 + [5])
        free = _locations(list(range(10)))
        assert (pairing_floor_report(free, tied, 2001, ALPHA, False)["tie_floor"]
                == pytest.approx(0.1))
        assert (pairing_floor_report(tied, free, 2001, ALPHA, False)["tie_floor"]
                == pytest.approx(0.1))

    def test_the_hard_lower_bound_is_the_draw_count_when_the_null_is_sampled(self):
        """
        `attainable_floor` is what a perfect input CONCENTRATES on, and in the
        sampled regime the realised p is Binomial around it, so a run can land
        just under. The number nothing can go below is reported separately
        rather than the two being conflated -- which is the slip this whole
        report exists to fix, made in the other direction.
        """
        c = _locations([0] * 7 + [3, 4, 5])
        rep = pairing_floor_report(c, c.copy(), 2001, ALPHA, False)
        assert rep["hard_lower_bound"] == pytest.approx(1 / 2001)
        assert rep["attainable_floor"] > rep["hard_lower_bound"]
        exhaustive = pairing_floor_report(c, c.copy(), math.factorial(10),
                                          ALPHA, True)
        assert exhaustive["hard_lower_bound"] == pytest.approx(
            exhaustive["tie_floor"])


class TestTheArmRefusesOnTies:

    def test_it_refuses_where_a_perfect_input_cannot_reach_alpha(self):
        with pytest.raises(ColocationRefused) as exc:
            _arm([0] * 9 + [5])
        msg = str(exc.value)
        assert "TIES rather than draws" in msg
        assert "Raising the draw count does NOT fix this" in msg

    def test_it_costs_nothing_two_heads_the_other_side_of_the_boundary(self):
        """
        A refusal is only defensible if what it turns away is a verdict the
        design could not have reached. Seven of ten tied is a floor of 0.00139
        and still emits; nine is 0.1 and does not. Two heads.
        """
        arm = _arm([0] * 7 + [3, 4, 5])
        assert arm["p_value"] <= ALPHA
        assert arm["floor"]["binds"] == "ties"
        assert arm["floor"]["attainable_floor"] == pytest.approx(1 / 720)

    def test_a_perfect_input_lands_on_the_floor_in_the_exhaustive_regime(self):
        """
        Exactly, with no sampling in the way: six units enumerate 720 pairings
        and four of them tied gives 4!/6! = 0.0333.
        """
        arm = _arm([0] * 4 + [1, 2])
        assert arm["null_exhaustive"] is True
        assert arm["p_value"] == pytest.approx(math.factorial(4) / 720)
        assert arm["p_value"] == pytest.approx(arm["attainable_floor"])

    def test_the_all_tied_case_still_refuses_on_the_older_check(self):
        """
        Every unit at one location was already refused, structurally, before
        this pass -- and it must keep being refused by THAT check, whose
        message is about the measurement grid. The tie floor there is 1.000 and
        would also fire; the older refusal is the more specific one and comes
        first.
        """
        with pytest.raises(ColocationRefused, match="same location"):
            _arm([0] * 6)

    def test_the_draw_count_refusal_is_unchanged_below_four_units(self):
        with pytest.raises(ColocationRefused, match="only 6 distinct pairings"):
            _arm([0, 1, 2])

    def test_the_floor_report_is_carried_in_the_arm(self):
        arm = _arm(list(range(10)))
        assert arm["floor"]["attainable_floor"] == arm["attainable_floor"]
        assert arm["floor"]["n_units"] == arm["n_units"]
        assert arm["floor"]["n_draws"] == arm["n_pairings"]


class TestTheDenseAxisCannotBeScored:
    """
    Eight heads, five of which carry relays. Both sides are LOCATED rises at
    distinct intervals, because the arm's older refusals fire first on anything
    flatter: a linear rise spreads its change mass over every interval, every
    unit's centroid lands on the grid's own midpoint, and the arm refuses on
    the constant it is being asked to permute. That is a real refusal and it is
    not this one, so the fixture has to clear it to reach the finding.
    """

    AXIS = [(l, h) for l in range(2) for h in range(4)]
    FORMING = AXIS[:5]

    def _payload(self, axis, forming):
        relay_by_step, score_by_step = [], []
        loc_a = {k: _series_at(j % N_INTERVALS) for j, k in enumerate(forming)}
        loc_b = {k: _series_at((3 * j + 1) % N_INTERVALS)
                 for j, k in enumerate(axis)}
        for i in range(len(STEPS)):
            relay_by_step.append({k: loc_a[k][i] for k in forming})
            score_by_step.append({k: loc_b[k][i] for k in axis})
        return formation_curve_payload(
            STEPS, relay_by_step, score_by_step,
            independence_source="two_stage", relay_owner=P_I1_RELAY_OWNER,
            above_null_excess=True)

    def _scored(self, axis, forming):
        pay = self._payload(axis, forming)
        return pay, p_value_p_i1(pay["checkpoint_steps"],
                                 pay["motif_strength"],
                                 pay["behavioral_induction_score"])

    def test_a_head_that_never_carries_a_relay_refuses_the_whole_gate(self):
        """
        The finding. Not "that head is dropped" -- the arm has no per-unit skip,
        so three all-zero units out of eight take the gate's p-value with them.
        """
        pay, res = self._scored(self.AXIS, self.FORMING)
        assert pay["n_heads"] == len(self.AXIS)
        assert sum(1 for row in pay["motif_strength"]
                   if not any(v > 0.0 for v in row)) == 3
        assert res["p_value"] is None
        assert "no rise" in res["reason"]

    def test_the_same_input_on_the_forming_heads_alone_emits(self):
        """
        Which is what makes the refusal a statement about the AXIS rather than
        about the relay series: the same five heads score fine on their own.
        """
        _, res = self._scored(self.FORMING, self.FORMING)
        assert res["p_value"] is not None
        assert res["arms"][0]["n_units"] == len(self.FORMING)

    def test_the_refusal_names_neither_the_unit_nor_how_many(self):
        """
        Pinned as it IS, not as it should be. The message comes from
        `change_profile`, which sees one series and cannot know it is unit 6 of
        eight -- so a reader gets no way to tell that the axis rule caused it.
        This test is the record that the diagnosability gap is known and
        deliberate at this commit; delete it when the message changes.
        """
        _, res = self._scored(self.AXIS, self.FORMING)
        assert "head" not in res["reason"]
        assert "unit" not in res["reason"]


class TestTheCommittedRecord:

    def test_the_committed_record_is_clean(self):
        """
        `--check` verifies the module hashes and the record's own claims. It
        does NOT need the generated tables: those live under `data/`, are
        git-ignored, and their hashes are checked only when present.
        """
        from tools.p_i1_attainable_floor import RECORD_PATH, check_record
        assert RECORD_PATH.exists(), "run `--write` and commit the record"
        assert check_record(RECORD_PATH) == []

    def test_the_checker_catches_a_record_that_lost_its_finding(self, tmp_path):
        from tools.p_i1_attainable_floor import RECORD_PATH, check_record
        rec = json.loads(RECORD_PATH.read_text())
        rec["the_defect"]["the_gap_reaches_above_alpha"] = False
        rec["dense_axis"]["dense_axis_emits_no_p_value"] = False
        p = tmp_path / "broken.json"
        p.write_text(json.dumps(rec))
        problems = check_record(p)
        assert any("dense-axis refusal is gone" in s for s in problems)
        assert any("nothing behind it" in s for s in problems)

    def test_the_checker_catches_a_stale_module_hash(self, tmp_path):
        from tools.p_i1_attainable_floor import RECORD_PATH, check_record
        rec = json.loads(RECORD_PATH.read_text())
        rec["construction_sha256"] = "0" * 64
        p = tmp_path / "stale.json"
        p.write_text(json.dumps(rec))
        assert any("changepoint_colocation.py has changed" in s
                   for s in check_record(p))

    def test_arm_a_is_paired_against_the_measured_behavioural_series(self):
        """
        The 2026-09-03 rewire: arm A's B side is the real behavioural series
        `tools/run/behavioural.py` writes, not the synthetic located rise it
        used before. The forming-head axis still emits, and on the RAW count
        the two curves do not co-locate -- p well above alpha, the per-head
        change locations ~2 log-steps apart.
        """
        from tools.p_i1_attainable_floor import RECORD_PATH
        rec = json.loads(RECORD_PATH.read_text())
        da = rec["dense_axis"]
        assert da["b_side_is_synthetic"] is False
        assert rec["inputs"]["behavioural_series_json"].endswith(
            "behavioural_series.json")
        assert rec["inputs"]["behavioural_series_json_sha256"]
        forming = da["rows"][1]
        assert forming["refused"] is False
        assert forming["p_value"] > 0.05
        assert da["the_two_curves_do_not_co_locate_on_the_raw_series"] is True
        assert forming["mean_distance_log_step"] > 1.0

    def test_the_checker_catches_a_record_reverted_to_a_synthetic_b_side(self, tmp_path):
        from tools.p_i1_attainable_floor import RECORD_PATH, check_record
        rec = json.loads(RECORD_PATH.read_text())
        rec["dense_axis"]["b_side_is_synthetic"] = True
        p = tmp_path / "synthetic.json"
        p.write_text(json.dumps(rec))
        assert any("B side is not the measured behavioural series" in s
                   for s in check_record(p))

    def test_the_record_carries_the_registered_grid_and_owner(self):
        from tools.p_i1_attainable_floor import RECORD_PATH
        rec = json.loads(RECORD_PATH.read_text())
        assert rec["registered_sweep"] == list(REGISTERED_P_I1_SWEEP)
        assert rec["registered_relay_owner"] == P_I1_RELAY_OWNER

    def test_the_finding_sentence_agrees_with_the_rows_it_summarises(self):
        """
        `EVALUABILITY.md`'s thirty-second lesson, item (iv): three passes have
        committed a summary sentence carrying an earlier run's digits, because
        nothing in this project compares a record's prose to its own fields.
        This is that comparison, for this record.
        """
        from tools.p_i1_attainable_floor import RECORD_PATH
        rec = json.loads(RECORD_PATH.read_text())
        s = rec["_the_finding"]
        d = rec["the_defect"]["rows"][0]
        assert f"{d['reported_floor_before']:.6f}" in s
        assert f"{d['attainable_floor_now']:.6f}" in s
        assert f"{rec['the_defect']['max_understatement_factor']:.0f}" in s
        dense = rec["dense_axis"]["rows"][0]
        assert str(dense["n_heads_with_no_relay_anywhere"]) in s
