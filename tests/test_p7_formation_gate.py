"""
tests/test_p7_formation_gate.py — P-I1's gate (experiment 7-A).

The construction is `core/changepoint_colocation.py`'s and is pinned in
`tests/test_changepoint_colocation.py`; this module pins the thin half that is
P-I1's alone -- which series, which direction, which unit, the two preconditions
and the tautology refusal.

`test_identical_series_are_refused_rather_than_scored` is the one worth reading.
`PREDICTIONS.md`'s second Phase 7 adjudication constraint records that the
behavioral induction score is "mean attention on induction pairs" and a motif
defined as "attentive edge on induction pairs" is the same number. Two identical
series co-locate perfectly and the gate would report p at its floor -- and no
null detects it, because the null is over the PAIRING and a tautological pair is
tautological at every head. The refusal catches the exactly-degenerate case and
not the substantive one, which is what the docstring says and what this pins.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pytest.ini [pytest] markers.
pytestmark = pytest.mark.pure

from core.changepoint_colocation import ColocationRefused, interval_midpoints
from p7_motifs import formation_gate as F

SWEEP = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000,
         8000, 13000, 23000, 33000, 43000, 63000, 83000, 103000, 123000, 143000]


#: P-I1's registered grid. `p_value_p_i1` computes on any grid; only this one
#: may be adjudicated, so the adjudication tests must run on it.
REG = list(F.REGISTERED_P_I1_SWEEP)


def _rise_at(j, amp=1.0, base=0.0, steps=None):
    """An above-null excess that is `base` until interval j and base+amp after."""
    n = len(SWEEP if steps is None else steps)
    v = np.full(n, float(base))
    v[j + 1:] = base + amp
    return v


def _heads(jumps, amp=1.0, base=0.0, steps=None):
    return [_rise_at(j, amp, base, steps) for j in jumps]


class TestRegisteredChoices:

    def test_the_unit_is_the_head_not_the_edge(self):
        """PREDICTIONS.md's first Phase 7 adjudication constraint, in the one
        place that can enforce it: the null permutes head pairings, so an
        edge-level n has no way in."""
        assert F.P_I1_UNIT == "head"
        r = F.p_value_p_i1(SWEEP, _heads([1, 5, 9, 13, 17, 21, 23]),
                           _heads([1, 5, 9, 13, 17, 21, 23], amp=2.0),
                           alpha=0.05)
        assert r["arms"][0]["unit"] == "head"
        assert r["arms"][0]["n_units"] == 7

    def test_both_series_are_registered_as_a_rise(self):
        assert F.P_I1_SERIES["relay_strength"]["direction"] == "rise"
        assert F.P_I1_SERIES["induction_score"]["direction"] == "rise"

    def test_the_relay_series_is_documented_as_the_above_null_excess(self):
        """P-I1's wording is 'strength above N1 and N2'. Clearing those nulls
        is motif_stats.py's job; the gate cannot check it and must say so."""
        field = F.P_I1_SERIES["relay_strength"]["field"]
        assert "N1/N2" in field and "qk_offset_null" in field

    def test_one_arm_and_no_invented_anchor(self):
        r = F.p_value_p_i1(SWEEP, _heads([1, 5, 9, 13, 17, 21, 23]),
                           _heads([2, 6, 10, 14, 18, 20, 22]), alpha=0.05)
        assert r["n_arms"] == 1
        assert r["arms"][0]["arm"] == "mutual"


class TestGate:

    def test_co_located_rises_clear(self):
        j = [1, 5, 9, 13, 17, 21, 23]
        r = F.p_value_p_i1(SWEEP, _heads(j), _heads(j, amp=2.0), alpha=0.05)
        assert r["reason"] is None
        assert r["verdict"] == "CO-LOCATES"
        assert r["p_value"] == pytest.approx(1.0 / math.factorial(7))

    def test_unrelated_timing_is_insufficient_and_not_a_falsification(self):
        r = F.p_value_p_i1(SWEEP, _heads([1, 12, 3, 20, 8, 23, 15]),
                           _heads([9, 2, 18, 5, 22, 11, 7]), alpha=0.05)
        assert r["verdict"] == "INSUFFICIENT"
        assert r["falsified"] is False

    def test_anti_aligned_timing_re_anchors(self):
        j = [1, 4, 8, 12, 16, 20, 23]
        r = F.p_value_p_i1(SWEEP, _heads(j),
                           _heads(list(reversed(j)), amp=2.0), alpha=0.05)
        assert r["verdict"] == "RE-ANCHORS"
        assert r["falsified"] is True

    def test_scale_blind_amplitude_does_not_move_the_location(self):
        """The two series are on different scales -- a motif excess and an
        attention score -- and the statistic is a location, so it must not
        care."""
        j = [1, 5, 9, 13, 17, 21, 23]
        a = F.p_value_p_i1(SWEEP, _heads(j), _heads(j, amp=0.5), alpha=0.05)
        b = F.p_value_p_i1(SWEEP, _heads(j), _heads(j, amp=1000.0), alpha=0.05)
        assert a["p_value"] == b["p_value"]

    def test_identical_series_are_refused_rather_than_scored(self):
        j = [1, 5, 9, 13, 17, 21, 23]
        same = _heads(j)
        r = F.p_value_p_i1(SWEEP, same, same, alpha=0.05)
        assert r["p_value"] is None
        assert "tautology" in r["reason"]
        assert r["verdict"] == "INSUFFICIENT"

    def test_the_refusal_is_per_head_not_per_run(self):
        """One tautological head is enough. An odd-length reversal pairs its
        middle head with itself, and that is refused even though every other
        head differs."""
        j = [1, 4, 8, 12, 16, 20, 23]
        r = F.p_value_p_i1(SWEEP, _heads(j), _heads(list(reversed(j))),
                           alpha=0.05)
        assert r["p_value"] is None
        assert "head 3" in r["reason"]

    def test_the_refusal_catches_the_degenerate_case_and_not_the_substantive_one(self):
        """A tautological pair that differs by so much as a constant offset is
        NOT caught, and the docstring says so rather than implying the check is
        stronger than it is."""
        j = [1, 5, 9, 13, 17, 21, 23]
        r = F.p_value_p_i1(SWEEP, _heads(j), _heads(j, base=0.5), alpha=0.05)
        assert r["p_value"] is not None

    def test_refuses_mismatched_head_counts(self):
        r = F.p_value_p_i1(SWEEP, _heads([1, 5, 9, 13, 17, 21, 23]),
                           _heads([1, 5, 9, 13]), alpha=0.05)
        assert r["p_value"] is None
        assert "index the same heads" in r["reason"]

    def test_refuses_too_few_heads_to_reach_alpha(self):
        r = F.p_value_p_i1(SWEEP, _heads([1, 9, 17]),
                           _heads([1, 9, 17], amp=2.0), alpha=0.05)
        assert r["p_value"] is None
        assert "attainable floor" in r["reason"]

    def test_skip_no_rise_defaults_off_and_still_refuses_on_a_flat_head(self):
        """
        PROJECT.md §3.1's fix. Default behaviour is untouched: a head whose
        above-null excess never rises still takes the whole gate down unless
        the caller opts in. `base=0.5` on the B side keeps every head off the
        tautology refusal (identical series), which is a different, earlier
        check this test is not about.
        """
        j = [1, 5, 9, 13, 17]
        rs = _heads(j)
        bs = _heads(j, base=0.5)
        rs[2] = np.zeros(len(SWEEP))            # one head with no excess rise
        r = F.p_value_p_i1(SWEEP, rs, bs, alpha=0.05)
        assert r["p_value"] is None
        assert "no rise anywhere" in r["reason"]

    def test_skip_no_rise_scores_the_surviving_heads_and_reports_the_count(self):
        j = [1, 5, 9, 13, 17]
        rs = _heads(j)
        bs = _heads(j, base=0.5)
        rs[2] = np.zeros(len(SWEEP))
        r = F.p_value_p_i1(SWEEP, rs, bs, alpha=0.05, skip_no_rise=True)
        assert r["p_value"] is not None
        assert r["arms"][0]["n_skipped_no_rise"] == 1
        assert r["arms"][0]["n_units"] == len(j) - 1


class TestEndpointFlags:
    """
    P-I1's falsifier names two endpoint conditions. They are reported and enter
    no p-value: they are about the curve's ends and the statistic is about
    where it rises, and one number cannot carry both questions.
    """

    def test_counts_heads_already_above_the_null_at_step_zero(self):
        relay = _heads([1, 5, 9]) + [np.full(len(SWEEP), 0.3)]   # above at step 0
        behav = _heads([1, 5, 9, 13])
        ef = F.endpoint_flags(SWEEP, relay, behav)
        assert ef["n_heads_above_null_at_first_step"] == 1
        assert ef["n_heads_absent_at_last_step"] == 0

    def test_counts_heads_absent_at_the_end_despite_peak_behaviour(self):
        relay = _heads([1, 5]) + [np.zeros(len(SWEEP))]
        behav = _heads([1, 5, 9])                     # head 2 peaks at the end
        ef = F.endpoint_flags(SWEEP, relay, behav)
        assert ef["n_heads_absent_at_last_step"] == 1
        assert ef["n_heads_absent_at_last_step_with_peak_behaviour"] == 1

    def test_zero_is_the_null_envelope_so_no_threshold_is_placed(self):
        """The series is an above-null EXCESS, so the endpoint checks compare
        against 0 and need no constant of their own."""
        assert "placed" not in F.endpoint_flags(
            SWEEP, _heads([1, 5, 9]), _heads([1, 5, 9]))["_note"].lower()

    def test_the_flags_reach_the_result_and_enter_no_p_value(self):
        j = [1, 5, 9, 13, 17, 21, 23]
        relay = _heads(j)
        relay[0] = relay[0] + 0.4                     # head 0 above null at step 0
        r = F.p_value_p_i1(SWEEP, relay, _heads(j, amp=2.0), alpha=0.05)
        assert r["endpoint_flags"]["n_heads_above_null_at_first_step"] == 1
        # the p-value is the co-location one, unchanged by the flag
        assert r["p_value"] == pytest.approx(1.0 / math.factorial(7))


class TestAdjudication:

    def _args(self):
        j = [1, 5, 9, 13, 15, 16, 17]
        return (_heads(j, steps=REG),
                _heads(j, amp=2.0, steps=REG))

    def test_a_grid_that_is_not_the_registered_one_is_refused(self):
        """`p_value_p_i1` computes on any grid; only the registered sweep may
        enter an e-process. On the CLAIM-B grid P-I1 borrowed until
        2026-09-01 the pairing null could not express anything at all — every
        head's change centroid was the same number — so which grid this ran
        on is exactly the thing that must be registered in advance."""
        j = [1, 5, 9, 13, 17, 21, 23]
        unregistered = SWEEP
        with pytest.raises(ColocationRefused, match="not the registered one"):
            F.adjudicate_p_i1(unregistered, _heads(j), _heads(j, amp=2.0),
                              alpha=0.05)

    def test_opt_in_writes_nothing_by_default(self, tmp_path):
        relay, behav = self._args()
        r = F.adjudicate_p_i1(REG, relay, behav, alpha=0.05,
                              adjudications_dir=tmp_path)
        assert r["adjudication"] is None
        assert not list(tmp_path.glob("*.json"))

    def test_emits_into_the_ledger_when_asked(self, tmp_path):
        relay, behav = self._args()
        r = F.adjudicate_p_i1(REG, relay, behav, alpha=0.05, adjudicate=True,
                              adjudications_dir=tmp_path)
        adj = r["adjudication"]
        assert adj is not None and adj["prediction_id"] == "P-I1"
        assert adj["claim"] == "H-BRIDGE"
        assert "head-pairing" in adj["test_name"] or "pairing" in adj["test_name"]
        assert (tmp_path / "P-I1.json").exists()

    def test_the_record_carries_the_endpoints_and_the_shared_estimator(self, tmp_path):
        relay, behav = self._args()
        r = F.adjudicate_p_i1(REG, relay, behav, alpha=0.05, adjudicate=True,
                              adjudications_dir=tmp_path)
        notes = r["adjudication"]["notes"]
        assert "CLAIM-B" in notes and "independent factors" in notes
        assert "endpoints" in notes
        assert "reported, not scored" in notes

    def test_a_refused_gate_writes_nothing_even_when_asked(self, tmp_path):
        j = [1, 5, 9, 13, 17, 21, 23]
        same = _heads(j)
        r = F.adjudicate_p_i1(REG, same, same, alpha=0.05, adjudicate=True,
                              adjudications_dir=tmp_path)
        assert r["adjudication"] is None
        assert not list(tmp_path.glob("*.json"))
