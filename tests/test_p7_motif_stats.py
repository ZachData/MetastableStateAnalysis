"""
tests/test_p7_motif_stats.py — null gating and verdict logic for
p7_motifs/motif_stats.py.

The counting is the easy part; these tests target the three ways a motif
result can be wrong while looking right:

  1. A missing null read as a passed null. "We did not compute the
     offset-matched null" and "the offset-matched null was cleared" must
     never produce the same verdict — Phase 6's P6-I2 was broken exactly
     here.
  2. A P-I3 result with no control arm, or one that cannot name what makes
     it independent of the behavioural induction score it is being
     correlated against.
  3. A degenerate prompt returning a number instead of refusing.
"""
from __future__ import annotations

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pytest.ini [pytest] markers.
pytestmark = pytest.mark.pure

from p7_motifs.motif_stats import (
    ALL_NULLS,
    GATING_NULLS,
    DegeneratePrompt,
    check_prompt_admissible,
    compare_against_nulls,
    cross_head_association,
    motif_counts_payload,
    per_head_motif_rate,
)
from tests.test_p7_motif_alphabet import _edges


# ---------------------------------------------------------------------------
# Null gating
# ---------------------------------------------------------------------------

class TestNullGating:

    def _nulls(self, n1=0.1, n2=0.1, n3=0.1, spread=0.01, n=200):
        rng = np.random.default_rng(0)
        return {
            "N1": rng.normal(n1, spread, n),
            "N2": rng.normal(n2, spread, n),
            "N3": rng.normal(n3, spread, n),
        }

    def test_clearing_both_gating_nulls_confirms(self):
        res = compare_against_nulls(0.9, self._nulls())
        assert res["verdict"] == "CONFIRMED"
        assert res["gating_nulls"] == list(GATING_NULLS)

    def test_failing_the_offset_matched_null_falsifies(self):
        """Clears N1 handily, but N2 sits right where the observation is —
        the offset-matched control is the one that matters."""
        res = compare_against_nulls(0.9, self._nulls(n1=0.1, n2=0.9))
        assert res["verdict"] == "FALSIFIED"
        assert "N2" in res["reason"]

    def test_a_missing_gating_null_refuses_rather_than_passes(self):
        """The central guard. Omitting N2 must not look like clearing it."""
        rng = np.random.default_rng(1)
        res = compare_against_nulls(0.9, {"N1": rng.normal(0.1, 0.01, 200)})
        assert res["verdict"] == "REFUSED"
        assert "N2" in res["reason"]

    def test_an_empty_null_array_also_refuses(self):
        rng = np.random.default_rng(2)
        res = compare_against_nulls(0.9, {"N1": rng.normal(0.1, 0.01, 200), "N2": []})
        assert res["verdict"] == "REFUSED"

    def test_observation_below_the_null_is_not_a_pass(self):
        """|z| >= threshold is significant in either direction; a motif
        that is significantly RARER than its null has not been confirmed."""
        res = compare_against_nulls(0.0, self._nulls(n1=0.5, n2=0.5))
        assert res["verdict"] == "FALSIFIED"

    def test_n3_is_reported_but_does_not_gate(self):
        """N3 failing must not change a verdict that N1 and N2 decided."""
        with_n3 = compare_against_nulls(0.9, self._nulls(n3=0.9))
        assert with_n3["verdict"] == "CONFIRMED"
        assert with_n3["n3_reading"] == "either_alone_suffices"

    def test_n3_reading_when_offset_shuffling_destroys_the_effect(self):
        res = compare_against_nulls(0.9, self._nulls(n3=0.1))
        assert res["n3_reading"] == "content_and_offset_jointly_required"

    def test_absent_n3_omits_the_reading_rather_than_guessing(self):
        rng = np.random.default_rng(3)
        res = compare_against_nulls(0.9, {"N1": rng.normal(0.1, 0.01, 200),
                                          "N2": rng.normal(0.1, 0.01, 200)})
        assert "n3_reading" not in res

    def test_degenerate_null_does_not_manufacture_significance(self):
        """A constant null gives z = nan, which must not read as a pass."""
        res = compare_against_nulls(0.9, {"N1": np.full(50, 0.1), "N2": np.full(50, 0.1)})
        assert res["verdict"] == "FALSIFIED"


# ---------------------------------------------------------------------------
# Prompt admissibility
# ---------------------------------------------------------------------------

class TestPromptAdmissibility:

    def test_degenerate_prompt_refuses(self):
        with pytest.raises(DegeneratePrompt, match="uniform"):
            check_prompt_admissible({"degeneracy": ["uniform"]}, "repeated_tokens")

    def test_empty_null_degeneracy_refuses(self):
        with pytest.raises(DegeneratePrompt):
            check_prompt_admissible({"degeneracy_modes": ["empty_null"]}, "p")

    def test_clean_prompt_passes(self):
        check_prompt_admissible({"degeneracy": []}, "p")       # must not raise
        check_prompt_admissible({}, "p")


# ---------------------------------------------------------------------------
# Per-head rates
# ---------------------------------------------------------------------------

class TestPerHeadRate:

    def test_rate_is_per_head_not_pooled(self):
        t = _edges([
            (1, 0, 5, 4, "a", "neither"),    # head 0: 1 of 2 is prev_token
            (1, 0, 9, 4, "a", "neither"),
            (1, 1, 6, 5, "a", "neither"),    # head 1: 2 of 2
            (1, 1, 7, 6, "a", "neither"),
        ])
        rates = per_head_motif_rate(t, "prev_token")
        assert rates[(1, 0)] == pytest.approx(0.5)
        assert rates[(1, 1)] == pytest.approx(1.0)

    def test_heads_with_no_edges_are_absent_not_zero(self):
        """'Nothing to measure' must not be averaged in as 'measured zero'
        — that is how a real effect gets diluted by empty heads."""
        t = _edges([(1, 0, 5, 4, "a", "neither")])
        rates = per_head_motif_rate(t, "prev_token")
        assert set(rates) == {(1, 0)}

    def test_an_honest_zero_rate_is_present(self):
        t = _edges([(1, 0, 9, 4, "a", "neither")])   # offset 5, not prev_token
        assert per_head_motif_rate(t, "prev_token")[(1, 0)] == 0.0


# ---------------------------------------------------------------------------
# P-I3 and its control arm
# ---------------------------------------------------------------------------

class TestCrossHeadAssociation:

    def _inputs(self, motif, behav, induction):
        return dict(motif_rate=motif, behavioral_score=behav,
                    is_induction_head=induction)

    def test_association_among_induction_heads_confirms(self):
        motif = {(1, i): 0.9 - 0.1 * i for i in range(4)}
        motif.update({(2, i): 0.05 for i in range(4)})
        behav = {(1, i): 0.8 - 0.1 * i for i in range(4)}
        behav.update({(2, i): 0.01 for i in range(4)})
        ind = {(1, i): True for i in range(4)}
        ind.update({(2, i): False for i in range(4)})

        res = cross_head_association(**self._inputs(motif, behav, ind),
                                     independence_source="two_stage")
        assert res["verdict"] == "CONFIRMED"
        assert res["spearman_induction_heads"] > 0

    def test_control_arm_carrying_the_motif_equally_falsifies(self):
        """The bridge-killing outcome, and it must be reported as such
        rather than buried under a positive within-group correlation."""
        motif = {(1, i): 0.9 - 0.1 * i for i in range(4)}
        motif.update({(2, i): 0.9 for i in range(4)})     # controls just as high
        behav = {(1, i): 0.8 - 0.1 * i for i in range(4)}
        behav.update({(2, i): 0.01 for i in range(4)})
        ind = {(1, i): True for i in range(4)}
        ind.update({(2, i): False for i in range(4)})

        res = cross_head_association(**self._inputs(motif, behav, ind),
                                     independence_source="two_stage")
        assert res["verdict"] == "FALSIFIED"
        assert "not of the classification" in res["reason"]

    def test_missing_control_arm_refuses(self):
        motif = {(1, i): 0.9 - 0.1 * i for i in range(4)}
        behav = {(1, i): 0.8 - 0.1 * i for i in range(4)}
        ind = {(1, i): True for i in range(4)}
        res = cross_head_association(**self._inputs(motif, behav, ind),
                                     independence_source="two_stage")
        assert res["verdict"] == "REFUSED"
        assert "control arm empty" in res["reason"]

    @pytest.mark.parametrize("bad", ["", "vibes", "attention", None])
    def test_unnamed_independence_source_raises(self, bad):
        """A P-I3 result that cannot say what makes it independent of the
        behavioural score has measured one quantity twice."""
        with pytest.raises(ValueError, match="independence_source"):
            cross_head_association({}, {}, {}, independence_source=bad)

    @pytest.mark.parametrize("src", ["two_stage", "force_channel", "particle_event"])
    def test_the_three_named_sources_are_accepted(self, src):
        res = cross_head_association({}, {}, {}, independence_source=src)
        assert res["independence_source"] == src

    def test_no_shared_heads_refuses(self):
        res = cross_head_association({(1, 0): 0.5}, {(2, 0): 0.5}, {(1, 0): True},
                                     independence_source="two_stage")
        assert res["verdict"] == "REFUSED"

    def test_independence_source_travels_into_the_result(self):
        motif = {(1, i): 0.9 - 0.1 * i for i in range(4)}
        motif.update({(2, i): 0.05 for i in range(4)})
        behav = {k: v for k, v in motif.items()}
        ind = {(1, i): True for i in range(4)}
        ind.update({(2, i): False for i in range(4)})
        res = cross_head_association(motif, behav, ind,
                                     independence_source="particle_event")
        assert res["independence_source"] == "particle_event"


# ---------------------------------------------------------------------------
# Artifact contract
# ---------------------------------------------------------------------------

class TestMotifCountsPayload:

    def test_payload_satisfies_the_registered_contract(self):
        from core.artifacts import get_spec
        payload = motif_counts_payload(
            counts={}, nulls={}, verdicts={},
            degenerate_prompts=[], force_cutoff=None,
        )
        missing = set(get_spec("phase7", "motif_counts").required_keys) - set(payload)
        assert not missing, f"declared but not produced: {sorted(missing)}"

    def test_absent_cutoff_says_no_thinning_rather_than_going_silent(self):
        payload = motif_counts_payload({}, {}, {}, [], None)
        assert payload["force_cutoff"]["mode"] == "none"

    def test_recorded_cutoff_is_preserved(self):
        cut = {"mode": "top_k_by_force", "k": 32, "status": "placed"}
        assert motif_counts_payload({}, {}, {}, [], cut)["force_cutoff"] == cut

    def test_thresholds_ride_along_as_placed(self):
        payload = motif_counts_payload({}, {}, {}, [], None)
        assert set(payload["threshold_status"].values()) == {"placed"}


class TestUndefinedRates:
    """relay_rate_by_head_pair returns NaN when no composition was possible
    for a head pair. That is 'undefined', not 'zero', and it must neither
    poison the correlation nor be imputed as a failed composition."""

    def test_undefined_rates_are_dropped_and_counted(self):
        motif = {(1, 0): 0.9, (1, 1): 0.8, (1, 2): 0.7, (1, 3): float("nan")}
        motif.update({(2, i): 0.05 for i in range(4)})
        behav = {(1, 0): 0.9, (1, 1): 0.8, (1, 2): 0.7, (1, 3): 0.6}
        behav.update({(2, i): 0.01 for i in range(4)})
        ind = {(1, i): True for i in range(4)}
        ind.update({(2, i): False for i in range(4)})

        res = cross_head_association(motif, behav, ind,
                                     independence_source="two_stage")
        assert res["n_undefined_dropped"] == 1
        assert res["n_induction_heads"] == 3
        assert np.isfinite(res["spearman_induction_heads"])

    def test_all_undefined_refuses_rather_than_returning_nan(self):
        motif = {(1, i): float("nan") for i in range(4)}
        behav = {(1, i): 0.5 for i in range(4)}
        ind = {(1, i): True for i in range(4)}
        res = cross_head_association(motif, behav, ind,
                                     independence_source="two_stage")
        assert res["verdict"] == "REFUSED"
        assert res["n_undefined_dropped"] == 4

    def test_undefined_rate_is_not_imputed_as_zero(self):
        """If NaN were imputed as 0.0, the mean motif rate among induction
        heads would drop and could fall below the control arm — turning a
        confirmed result into a falsified one on an artifact of imputation."""
        motif = {(1, i): 0.9 for i in range(3)}
        motif[(1, 3)] = float("nan")
        motif.update({(2, i): 0.1 for i in range(4)})
        behav = {(1, i): 0.9 - 0.05 * i for i in range(4)}
        behav.update({(2, i): 0.01 for i in range(4)})
        ind = {(1, i): True for i in range(4)}
        ind.update({(2, i): False for i in range(4)})

        res = cross_head_association(motif, behav, ind,
                                     independence_source="two_stage")
        assert res["mean_motif_rate_induction"] == pytest.approx(0.9)
