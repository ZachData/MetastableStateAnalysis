"""
tests/test_p_s1_null.py — P-S1's null construction (POPPER_PLAN.md B6, live).

P-S1 is one of only three predictions that are both `e-value` and `active`, so
it is the first one this project can actually adjudicate. Its null machinery
already existed (`random_band`, `random_baseline_Q`); what was missing was the
last step — turning a sampled null into a *calibrated* p-value rather than an
"Nσ from null" summary that reads like significance without being it.

The tests that matter here are the calibration ones. A p-value that is not
uniform under H0 produces an e-value whose expectation exceeds 1, which voids
`E[E] <= 1` for the entire claim — so "does this p-value mean what a p-value
means" is not a detail, it is the whole precondition.

Sample sizes are kept small enough for the pure tier's ~10s budget, which makes
the calibration checks coarse. They are sized to catch a *systematic* offset,
not a subtle one; the tolerances say so explicitly rather than pretending to
more resolution than 40 draws can give.
"""

from __future__ import annotations

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable.
pytestmark = pytest.mark.pure

from core.nulls import p_from_null
from p1c_frames.centroids import (
    P_S1_ALTERNATIVE,
    P_S1_T_MAX,
    _standardised_improvement,
    adjudicate_p_s1_from_reports,
    p_value_p_s1,
    run_design_test,
)


def _iid(m: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(m, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _spread(m: int, d: int, seed: int) -> np.ndarray:
    """Mean-centred then re-normalised: closer to a spherical design than i.i.d."""
    X = _iid(m, d, seed)
    X = X - X.mean(0)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# The general primitive
# ---------------------------------------------------------------------------

class TestPFromNull:

    def test_resolution_floor_is_one_over_n_plus_one(self):
        """
        p never reaches 0, however extreme the observation. A p of exactly 0
        calibrates to an INFINITE e-value -- asserting more evidence than any
        finite sample can carry -- so the floor is a correctness property, not
        a rounding convenience.
        """
        null = np.random.default_rng(0).normal(size=199)
        r = p_from_null(10.0, null, alternative="greater")
        assert r["p_value"] == pytest.approx(1.0 / 200.0)
        assert r["at_resolution_floor"] is True

    def test_uniform_under_the_null(self):
        """Observation drawn from the null itself => p ~ U(0,1)."""
        rng = np.random.default_rng(1)
        ps = []
        for _ in range(400):
            null = rng.normal(size=99)
            ps.append(p_from_null(float(rng.normal()), null, alternative="greater")["p_value"])
        ps = np.array(ps)
        assert abs(ps.mean() - 0.5) < 0.05
        assert (ps <= 0.05).mean() < 0.09        # conservative side is fine

    def test_direction_matters(self):
        null = np.random.default_rng(2).normal(size=299)
        assert p_from_null(3.0, null, alternative="greater")["p_value"] < 0.05
        assert p_from_null(3.0, null, alternative="less")["p_value"] > 0.95

    def test_refuses_unknown_alternative(self):
        with pytest.raises(ValueError, match="alternative"):
            p_from_null(1.0, np.zeros(10), alternative="sideways")

    def test_refuses_empty_null_and_nonfinite_observation(self):
        with pytest.raises(ValueError, match="empty"):
            p_from_null(1.0, np.array([]))
        with pytest.raises(ValueError, match="finite"):
            p_from_null(float("nan"), np.zeros(10))

    def test_nonfinite_null_draws_are_dropped_and_counted(self):
        null = np.array([0.0, 1.0, np.nan, 2.0, np.inf])
        r = p_from_null(0.5, null)
        assert r["n_null_finite"] == 3
        assert r["n_null_dropped"] == 2


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------

class TestStatistic:

    def test_standardisation_stops_k1_dominating(self):
        """
        UPDATE_PLAN.md 5.8 measured the null band narrowing from ~0.17 at k=1
        to ~0.002 at k=3. An unstandardised sum is therefore dominated by k=1
        and discards the degrees that are more sensitive in relative terms.

        Here the improvement sits ENTIRELY at k=3, at a magnitude that is huge
        relative to k=3's own noise and negligible in absolute terms. The
        standardised statistic must see it.
        """
        sd = np.array([0.17, 0.015, 0.002])
        trained = np.array([1.0, 1.0, 1.0])
        step0 = np.array([1.0, 1.0, 1.0 + 0.01])      # 5 sigma at k=3, tiny absolutely
        assert _standardised_improvement(trained, step0, sd) == pytest.approx(5.0)
        # The unstandardised difference would be 0.01 -- indistinguishable from
        # nothing against k=1's 0.17 band.
        assert float(np.sum(step0 - trained)) == pytest.approx(0.01)

    def test_direction_sign(self):
        sd = np.ones(3)
        better = _standardised_improvement(np.zeros(3), np.ones(3), sd)   # trained smaller
        worse = _standardised_improvement(np.ones(3), np.zeros(3), sd)
        assert better > 0 > worse

    def test_declared_constants(self):
        """
        Both are constants rather than parameters, so a per-run choice of how
        many degrees to include, or which tail to test, is not available.
        """
        assert P_S1_T_MAX == 3
        assert P_S1_ALTERNATIVE == "greater"


# ---------------------------------------------------------------------------
# Calibration and power -- the properties the guarantee rests on
# ---------------------------------------------------------------------------

class TestPS1Calibration:

    def test_calibrated_under_the_null(self):
        """
        Two i.i.d. arms are H0 exactly: no difference between trained and step0.
        The rejection rate at a nominal level must not exceed it.

        Coarse by construction -- 24 pairs cannot resolve 0.05 finely -- so the
        assertion is against a systematic failure, which is what the pre-fix
        version had: referencing observed and null against different
        Monte-Carlo baselines gave a null-p mean of 0.40 against 0.50, small
        and in the anticonservative direction.
        """
        ps = []
        for s in range(24):
            ta = run_design_test(_iid(16, 64, 100 + s), n_trials=40)
            tb = run_design_test(_iid(16, 64, 900 + s), n_trials=40)
            ps.append(p_value_p_s1(ta, tb, n_null=100, seed=s)["p_value"])
        ps = np.array(ps)
        assert 0.35 < ps.mean() < 0.65, f"null p-values not centred: mean={ps.mean():.3f}"
        assert (ps <= 0.05).mean() <= 0.17, "rejection rate far above nominal"

    def test_has_power_against_a_real_difference(self):
        """A calibrated test that never rejects is also useless."""
        ps = []
        for s in range(6):
            ta = run_design_test(_spread(24, 96, 300 + s), n_trials=40)
            tb = run_design_test(_iid(24, 96, 700 + s), n_trials=40)
            ps.append(p_value_p_s1(ta, tb, n_null=120, seed=s)["p_value"])
        assert np.median(ps) < 0.25, f"no power: median p={np.median(ps):.3f}"

    def test_one_sided_ignores_the_wrong_direction(self):
        """
        The improvement in the step0 arm instead of the trained arm is P-S1
        being wrong, not P-S1 being right. A two-sided test would score it as
        evidence.
        """
        ta = run_design_test(_iid(24, 96, 11), n_trials=40)
        tb = run_design_test(_spread(24, 96, 12), n_trials=40)
        assert p_value_p_s1(ta, tb, n_null=120, seed=0)["p_value"] > 0.5

    def test_shared_reference_is_used_and_reported(self):
        ta = run_design_test(_iid(16, 64, 5), n_trials=40)
        tb = run_design_test(_iid(16, 64, 6), n_trials=40)
        r = p_value_p_s1(ta, tb, n_null=80, seed=0)
        assert r["shared_reference"] is True
        assert "re-referenced" in r["reference_note"]

    def test_fallback_is_flagged_not_silent(self):
        """Without raw Q the two arms sit on the caller's baseline; say so."""
        ta = run_design_test(_iid(16, 64, 5), n_trials=40)
        tb = run_design_test(_iid(16, 64, 6), n_trials=40)
        ta.pop("Q"); tb.pop("Q")
        r = p_value_p_s1(ta, tb, n_null=80, seed=0)
        assert r["shared_reference"] is False
        assert "FELL BACK" in r["reference_note"]

    def test_the_fallback_note_no_longer_quotes_a_rate_from_retired_code(self):
        """
        It used to say the fallback is "mildly anticonservative" and cite a
        null-p mean of 0.40. That number was measured on the pre-2026-08-24
        code, where observed and null sat on genuinely different baselines.
        On the code that exists the statistic is a DIFFERENCE of two ratios
        formed against the SAME caller baseline, so a common per-degree factor
        cancels and the two paths are indistinguishable -- measured in
        `claims/audits/p_s1_dry_run.json`. A rate that stopped describing the
        path it was attached to is the defect `POPPER_PLAN.md` §6m records
        against inlined figures.
        """
        ta = run_design_test(_iid(16, 64, 5), n_trials=40)
        tb = run_design_test(_iid(16, 64, 6), n_trials=40)
        fixed = p_value_p_s1(ta, tb, n_null=200, seed=3)["p_value"]
        ta.pop("Q"); tb.pop("Q")
        fell_back = p_value_p_s1(ta, tb, n_null=200, seed=3)["p_value"]
        assert abs(fixed - fell_back) < 0.05


class TestPS1RefusesMismatchedArms:
    """
    The defect the dry run found. The null is drawn at the TRAINED arm's
    (m, d) and both arms are re-referenced against it, and until 2026-08-27
    nothing checked that the step-0 arm agreed. `POPPER_PLAN.md` §6p.
    """

    def test_refuses_a_different_cluster_count(self):
        ta = run_design_test(_iid(24, 64, 5), n_trials=40)
        tb = run_design_test(_iid(20, 64, 6), n_trials=40)
        r = p_value_p_s1(ta, tb, n_null=80, seed=0)
        assert r["p_value"] is None
        assert "different configurations" in r["reason"]

    def test_refuses_a_different_dimension(self):
        ta = run_design_test(_iid(16, 64, 5), n_trials=40)
        tb = run_design_test(_iid(16, 48, 6), n_trials=40)
        r = p_value_p_s1(ta, tb, n_null=80, seed=0)
        assert r["p_value"] is None
        assert "different configurations" in r["reason"]

    def test_refuses_when_the_step0_arm_cannot_be_checked(self):
        """
        Refusing rather than assuming: an arm that does not say what (m, d) it
        sits at cannot be verified against the one the null is drawn at.
        """
        ta = run_design_test(_iid(16, 64, 5), n_trials=40)
        tb = run_design_test(_iid(16, 64, 6), n_trials=40)
        tb.pop("n_centroids")
        r = p_value_p_s1(ta, tb, n_null=80, seed=0)
        assert r["p_value"] is None
        assert "cannot be checked" in r["reason"]

    def test_matched_arms_still_emit(self):
        """The other direction: a refusal that fires on everything is not one."""
        ta = run_design_test(_iid(16, 64, 5), n_trials=40)
        tb = run_design_test(_iid(16, 64, 6), n_trials=40)
        assert p_value_p_s1(ta, tb, n_null=80, seed=0)["p_value"] is not None

    def test_degenerate_configuration_returns_no_p_value(self):
        ta = run_design_test(_iid(16, 64, 5), n_trials=40)
        r = p_value_p_s1(ta, ta, m=1, d=64)
        assert r["p_value"] is None and "degenerate" in r["reason"]


# ---------------------------------------------------------------------------
# Adjudication wiring
# ---------------------------------------------------------------------------

class TestAdjudicationWiring:

    def _reports(self):
        return (run_design_test(_iid(16, 64, 5), n_trials=40),
                run_design_test(_iid(16, 64, 6), n_trials=40))

    def test_opt_in_writes_nothing_by_default(self, tmp_path):
        ta, tb = self._reports()
        r = adjudicate_p_s1_from_reports(ta, tb, layer=3, n_null=80,
                                         adjudications_dir=tmp_path)
        assert r["adjudication"] is None
        assert not list(tmp_path.glob("*.json"))

    def test_emits_into_the_ledger_when_asked(self, tmp_path):
        ta, tb = self._reports()
        r = adjudicate_p_s1_from_reports(ta, tb, layer=3, n_null=80,
                                         adjudicate=True, adjudications_dir=tmp_path)
        adj = r["adjudication"]
        assert adj is not None
        assert adj["prediction_id"] == "P-S1"
        assert adj["claim"] == "H-RESIST"
        assert "one-sided 'greater'" in adj["test_name"]
        assert "layer=3" in adj["notes"]
        assert (tmp_path / "P-S1.json").exists()

    def test_layer_is_recorded_because_it_is_a_choice(self, tmp_path):
        """
        A run has one configuration per layer, so "which layer" has as many
        options as there are layers. It is required at the call site and lands
        in the record, where the pre-registration gate can see it.
        """
        ta, tb = self._reports()
        with pytest.raises(TypeError):
            adjudicate_p_s1_from_reports(ta, tb)          # no default
        r = adjudicate_p_s1_from_reports(ta, tb, layer=11, n_null=80,
                                         adjudicate=True, adjudications_dir=tmp_path)
        assert r["layer"] == 11
        assert "layer=11" in r["adjudication"]["notes"]

    def test_real_ledger_untouched_by_tests(self, tmp_path):
        from core.adjudication import load_adjudications
        before = {r["prediction_id"] for r in load_adjudications()}
        ta, tb = self._reports()
        adjudicate_p_s1_from_reports(ta, tb, layer=1, n_null=80,
                                     adjudicate=True, adjudications_dir=tmp_path)
        assert {r["prediction_id"] for r in load_adjudications()} == before
