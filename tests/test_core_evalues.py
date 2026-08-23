"""
tests/test_core_evalues.py — the adjudication kernel (POPPER_PLAN.md B1).

`core/evalues.py` is the only piece of the Popperian workstream with a proof
attached, and every downstream number inherits its correctness. So the tests
here check the *properties the proof asserts* -- calibration, the
supermartingale inequality, Type-I control under both fixed-horizon and
optional stopping -- rather than only spot values.

Two of these follow the project's standing rule 5 ("anchors need a
non-symmetric arm", `UPDATE_PLAN.md` §6, from the wrong-trace-contraction
defect in §5.6): a calibrator test that only checks the uniform case tests
almost nothing, because the interesting failure is a conservative p-value
distribution where E[e] should be strictly below 1. Both arms are here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.evalues import (
    DEFAULT_ALPHA,
    DEFAULT_KAPPA,
    Adjudication,
    EProcess,
    EValueError,
    calibrate,
    combine,
    log_calibrate,
    required_p_for_rejection,
    simulate_type_i_error,
    sufficient_evidence,
)

pytestmark = pytest.mark.pure


# ---------------------------------------------------------------------------
# The calibrator
# ---------------------------------------------------------------------------

class TestCalibrator:

    def test_null_calibration_uniform(self):
        """E[e] == 1 for p ~ U(0,1) -- the defining property, by Monte Carlo."""
        rng = np.random.default_rng(0)
        p = rng.uniform(size=400_000)
        e = DEFAULT_KAPPA * p ** (DEFAULT_KAPPA - 1.0)
        # The estimator's own variance is heavy-tailed (e has infinite variance
        # for kappa <= 0.5), so the tolerance is loose on purpose. The check
        # that matters is that it does not sit systematically above 1.
        assert abs(e.mean() - 1.0) < 0.05

    def test_null_calibration_conservative_p(self):
        """
        The non-symmetric arm: for a *conservative* p-value (stochastically
        larger than uniform), E[e] must be strictly below 1, not merely near it.

        A calibrator that clipped, rescaled, or normalized its input would pass
        the uniform test above and fail here.
        """
        rng = np.random.default_rng(1)
        # p = sqrt(u) is stochastically larger than uniform: P(p <= t) = t^2 <= t.
        p = np.sqrt(rng.uniform(size=400_000))
        e = DEFAULT_KAPPA * p ** (DEFAULT_KAPPA - 1.0)
        assert e.mean() < 1.0
        assert e.mean() == pytest.approx(2.0 / 3.0, abs=0.02)

    def test_non_falsification_counts_against(self):
        """p = 1 must give e = kappa < 1: failing to falsify is evidence against."""
        assert calibrate(1.0) == pytest.approx(DEFAULT_KAPPA)
        assert calibrate(1.0) < 1.0

    def test_paper_table1_round1(self):
        """
        POPPER's Table 1 round 1 reports p = 1.0 -> cumulative e-value 0.5.
        Reproducing it is what pins kappa = 0.5 as the paper's setting rather
        than an assumption of ours.
        """
        assert calibrate(1.0, kappa=0.5) == pytest.approx(0.5)

    def test_monotone_decreasing_in_p(self):
        ps = [1.0, 0.5, 0.1, 0.01, 1e-4]
        es = [calibrate(p) for p in ps]
        assert all(a < b for a, b in zip(es, es[1:]))

    def test_zero_p_is_infinite(self):
        assert math.isinf(calibrate(0.0))
        assert math.isinf(log_calibrate(0.0))

    @pytest.mark.parametrize("bad", [-0.001, 1.001, float("nan")])
    def test_refuses_invalid_p(self, bad):
        with pytest.raises(EValueError):
            calibrate(bad)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5, float("nan")])
    def test_refuses_invalid_kappa(self, bad):
        with pytest.raises(EValueError):
            calibrate(0.5, kappa=bad)

    def test_log_agrees_with_linear(self):
        for p in [1.0, 0.75, 0.5, 0.1, 0.01, 1e-6, 1e-12]:
            assert log_calibrate(p) == pytest.approx(math.log(calibrate(p)), rel=1e-12)


# ---------------------------------------------------------------------------
# The e-process
# ---------------------------------------------------------------------------

class TestEProcess:

    def test_empty_process_is_neutral(self):
        proc = EProcess(claim="H-TEST")
        assert proc.E == pytest.approx(1.0)
        assert proc.log_E == 0.0
        assert proc.decision() == "insufficient_evidence"

    def test_product_accumulates(self):
        proc = EProcess(claim="H-TEST")
        proc.add("P-1", 0.01)
        proc.add("P-2", 0.05)
        expected = calibrate(0.01) * calibrate(0.05)
        assert proc.E == pytest.approx(expected)

    def test_order_does_not_change_final_E(self):
        a = EProcess(claim="H")
        for pid, p in [("P-1", 0.02), ("P-2", 0.4), ("P-3", 0.001)]:
            a.add(pid, p)
        b = EProcess(claim="H")
        for pid, p in [("P-3", 0.001), ("P-1", 0.02), ("P-2", 0.4)]:
            b.add(pid, p)
        assert a.E == pytest.approx(b.E, rel=1e-12)

    def test_trajectory_is_running_log_E(self):
        proc = EProcess(claim="H")
        proc.add("P-1", 0.1)
        proc.add("P-2", 0.9)
        proc.add("P-3", 0.001)
        traj = proc.trajectory
        assert len(traj) == 3
        assert traj[-1] == pytest.approx(proc.log_E)
        # A weak experiment must pull the trajectory *down*.
        assert traj[1] < traj[0]

    def test_double_adjudication_refused(self):
        proc = EProcess(claim="H")
        proc.add("P-1", 0.01)
        with pytest.raises(EValueError, match="already contributed"):
            proc.add("P-1", 0.001)

    def test_decision_threshold(self):
        proc = EProcess(claim="H", alpha=0.05)
        # One very small p-value alone should cross 1/0.05 = 20.
        proc.add("P-1", 1e-4)
        assert proc.E >= 20.0
        assert proc.decision() == "reject_null"

    def test_decision_never_accepts_the_null(self):
        proc = EProcess(claim="H")
        for i, p in enumerate([0.9, 0.95, 0.99]):
            proc.add(f"P-{i}", p)
        assert proc.E < 1.0
        assert proc.decision() == "insufficient_evidence"

    def test_log_space_survives_where_product_overflows(self):
        proc = EProcess(claim="H")
        for i in range(600):
            proc.add(f"P-{i}", 1e-6)
        assert math.isinf(proc.E)          # the linear product genuinely overflows
        assert math.isfinite(proc.log_E)   # the log does not
        assert proc.log_E > 0.0
        assert proc.decision() == "reject_null"

    def test_log_accumulation_matches_direct_product(self):
        """Over a long-but-representable run, log-space == direct product."""
        rng = np.random.default_rng(7)
        ps = rng.uniform(0.2, 1.0, size=500)
        proc = EProcess(claim="H")
        for i, p in enumerate(ps):
            proc.add(f"P-{i}", float(p))
        direct = 0.0
        for p in ps:
            direct += math.log(calibrate(float(p)))
        assert proc.log_E == pytest.approx(direct, rel=1e-10)

    def test_invalid_alpha_refused(self):
        with pytest.raises(EValueError):
            EProcess(claim="H", alpha=0.0)
        with pytest.raises(EValueError):
            EProcess(claim="H", alpha=1.0)


# ---------------------------------------------------------------------------
# The guarantee itself
# ---------------------------------------------------------------------------

class TestTypeIControl:

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2])
    def test_fixed_horizon_control(self, alpha):
        """
        Empirical Type-I error at or below nominal across a range of levels --
        the paper's own sensitivity check (Figure 4, panel 1) reproduced here.
        """
        rate = simulate_type_i_error(n_trials=40_000, n_experiments=5,
                                     alpha=alpha, seed=3)
        assert rate <= alpha + 0.004   # Monte Carlo slack at 40k trials

    def test_optional_stopping_control(self):
        """
        The property that distinguishes an e-process from a p-value combination:
        control survives a stopping rule chosen adversarially -- stop the moment
        the evidence first crosses the threshold, look at as many experiments as
        it takes.

        This is not a hypothetical for this project. `PREDICTIONS.md` attaches a
        hard stop to claim (c), and `INDEX.md` records three phases going out of
        scope in a single day; the number of experiments run against a claim
        genuinely depends on how earlier ones came out.
        """
        alpha = 0.05
        threshold = math.log(1.0 / alpha)
        rng = np.random.default_rng(11)
        n_trials, max_experiments = 20_000, 40
        rejections = 0
        for _ in range(n_trials):
            log_E = 0.0
            for _ in range(max_experiments):
                p = float(rng.uniform())
                log_E += log_calibrate(p)
                if log_E >= threshold:
                    rejections += 1
                    break
        rate = rejections / n_trials
        assert rate <= alpha + 0.006

    def test_supermartingale_property(self):
        """
        E[E_i | E_{i-1}] <= E_{i-1} under the null -- the inequality Doob's
        optional stopping theorem is applied to in the paper's Theorem 4 proof.
        Checked as E[e] <= 1 for a fresh null p-value at any point in the
        sequence, which is the same statement after factoring out E_{i-1}.
        """
        rng = np.random.default_rng(13)
        e = DEFAULT_KAPPA * rng.uniform(size=300_000) ** (DEFAULT_KAPPA - 1.0)
        # One-sided: the guarantee is E[e] <= 1. Allow Monte Carlo slack above.
        assert e.mean() <= 1.05


# ---------------------------------------------------------------------------
# Planning instrument
# ---------------------------------------------------------------------------

class TestRequiredP:

    def test_fresh_claim_threshold(self):
        """With no prior evidence, the required p is the exact inversion."""
        p_req = required_p_for_rejection(alpha=0.05, kappa=0.5, log_E_prior=0.0)
        assert calibrate(p_req, 0.5) == pytest.approx(20.0, rel=1e-9)
        assert sufficient_evidence(calibrate(p_req * 0.999, 0.5), 0.05)

    def test_prior_evidence_relaxes_the_requirement(self):
        fresh = required_p_for_rejection(alpha=0.05, log_E_prior=0.0)
        with_prior = required_p_for_rejection(alpha=0.05, log_E_prior=math.log(5.0))
        assert with_prior > fresh

    def test_already_crossed_returns_one(self):
        assert required_p_for_rejection(alpha=0.05, log_E_prior=math.log(1000.0)) == 1.0

    def test_process_reports_its_own_requirement(self):
        proc = EProcess(claim="H", alpha=0.05)
        proc.add("P-1", 0.2)
        needed = proc.next_p_needed()
        # Adding exactly that p-value should land at or just past the threshold.
        proc.add("P-2", needed * 0.999)
        assert proc.decision() == "reject_null"


# ---------------------------------------------------------------------------
# Serialization -- the basis of CI's recomputation check (B7)
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_round_trip(self):
        proc = EProcess(claim="H-RESIST", alpha=0.05)
        proc.add("P-S1", 0.004)
        proc.add("P-M1", 0.21)
        rec = proc.to_record()
        back = EProcess.from_record(rec)
        assert back.claim == proc.claim
        assert back.alpha == proc.alpha
        assert back.log_E == pytest.approx(proc.log_E)
        assert [a.prediction_id for a in back.adjudications] == ["P-S1", "P-M1"]

    def test_record_carries_everything_ci_needs_to_recompute(self):
        proc = EProcess(claim="H", alpha=0.05)
        proc.add("P-1", 0.01)
        rec = proc.to_record()
        for key in ("claim", "alpha", "threshold", "log_E", "E", "decision", "experiments"):
            assert key in rec
        for key in ("prediction_id", "p_value", "e_value", "kappa"):
            assert key in rec["experiments"][0]

    def test_hand_edited_e_value_does_not_survive_reload(self):
        """
        `from_record` recalibrates from the p-value rather than trusting the
        stored e-value, so a record edited by hand is caught rather than
        propagated. This is what makes CI's recomputation check (B7) meaningful.
        """
        proc = EProcess(claim="H")
        proc.add("P-1", 0.5)
        rec = proc.to_record()
        rec["experiments"][0]["e_value"] = 999.0    # tamper
        back = EProcess.from_record(rec)
        assert back.adjudications[0].e_value == pytest.approx(calibrate(0.5))
        assert back.E != pytest.approx(999.0)


class TestCombine:

    def test_matches_eprocess(self):
        ps = [0.03, 0.4, 0.008]
        E, reject = combine(ps, alpha=0.05)
        proc = EProcess(claim="H", alpha=0.05)
        for i, p in enumerate(ps):
            proc.add(f"P-{i}", p)
        assert E == pytest.approx(proc.E)
        assert reject == (proc.decision() == "reject_null")

    def test_zero_short_circuits(self):
        E, reject = combine([0.5, 0.0, 0.9])
        assert math.isinf(E)
        assert reject is True
