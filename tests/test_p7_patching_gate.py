"""
tests/test_p7_patching_gate.py — P-AB1's gate (the patching entry's
recapture-vs-propagation question, as a growth exponent in remaining depth).

Same standard the other constructed nulls were held to: exactness, validity
under H0, power against a planted effect, p = 1 with the effect reversed, and
every refusal -- on synthetic curves with known answers, because no ablation
sweep exists in this repository.

Four of these are worth reading rather than counting.

`test_the_registered_null_leaves_the_statistic_invariant` is why the registry's
"permutation over ablation points" is not the null this gate uses. Permuting
which point's real exponent meets which point's control exponent leaves a mean
paired difference EXACTLY unchanged, so every draw ties the observation and the
design's floor is 1.000. `POPPER_PLAN.md` 6p's seventeenth lesson -- a floor is
a claim about the design, not about the call -- in the one form where the design
can never reject anything.

`test_a_saturating_curve_reads_a_smaller_exponent_at_a_longer_window` is the
degeneracy the whole construction is built around, in one deterministic
assertion: the same dynamics fitted over more layers return a smaller exponent.
The ablation point fixes the window, so exponents at different points are not
comparable, and the pairing is what makes the comparison legal.

`test_the_arm_that_dominates_everywhere_reads_the_smaller_exponent` is the
sign reversal. A curve that is larger at every single layer receives the
SMALLER fitted exponent once it reaches its ceiling, so the gate's falsification
branch fires on an input where the prediction holds. Nothing about the null
detects that; only the shape of the curve does.

`test_the_power_law_refusal_costs_verdicts_and_the_record_says_so` is the third
category 6o named -- a refusal that is right and costs verdicts anyway -- and
this pins that the record carries the counterfactual rather than a claim that
the cost is small.
"""

from __future__ import annotations

import json
import math
from itertools import product

import numpy as np
import pytest

from p7_motifs.patching_gate import (
    MIN_FIT_POINTS,
    P_AB1_ALTERNATIVE,
    P_AB1_RECIPROCAL_ALTERNATIVE,
    P_AB1_UNITS,
    REGISTERED_EXCHANGEABLE_UNIT,
    PatchingRefused,
    adjudicate_p_ab1,
    attainable_floor_report,
    bend_contrast_arm,
    fit_growth_exponent,
    gate_verdict,
    magnitude_match_report,
    p_value_p_ab1,
    paired_exponents,
    power_law_arm,
    registered_null_invariance_report,
    shared_prompt_factor_diagnostic,
    signflip_arm,
    window_reference_report,
)

pytestmark = pytest.mark.pure

ALPHA = 0.05
N_PROMPTS = 6
N_POINTS = 7
WINDOW = 8


def _power_law(beta, window=WINDOW, rng=None, sd=0.0):
    k = np.arange(1, window + 1, dtype=float)
    d = k ** beta
    if sd and rng is not None:
        d = d * np.exp(rng.normal(0.0, sd, size=window))
    return d


def _saturating(beta, tau, window=WINDOW, rng=None, sd=0.0):
    k = np.arange(1, window + 1, dtype=float)
    d = 1.0 - np.exp(-((k / tau) ** beta))
    if sd and rng is not None:
        d = d * np.exp(rng.normal(0.0, sd, size=window))
    return d


def _grid(fn, n_prompts=N_PROMPTS, n_points=N_POINTS):
    return [[fn() for _ in range(n_points)] for _ in range(n_prompts)]


def _mags(n_prompts=N_PROMPTS, n_points=N_POINTS):
    return np.ones((n_prompts, n_points))


# ---------------------------------------------------------------------------
class TestTheEstimator:

    def test_an_exact_power_law_returns_its_exponent_at_every_window(self):
        for beta in (0.5, 1.0, 1.5, 2.0):
            for w in (3, 5, 8, 16):
                got = fit_growth_exponent(_power_law(beta, window=w))["exponent"]
                assert got == pytest.approx(beta, abs=1e-10)

    def test_an_exact_power_law_has_no_bend_and_no_residual(self):
        f = fit_growth_exponent(_power_law(1.4))
        assert f["window_sensitivity"] == pytest.approx(0.0, abs=1e-10)
        assert f["residual_rms"] == pytest.approx(0.0, abs=1e-10)
        # 0/0 must read as 0 and not as whatever 1e-16 over 1e-32 happens to be.
        assert f["bend_z"] == 0.0

    def test_a_saturating_curve_reads_a_smaller_exponent_at_a_longer_window(self):
        """
        The degeneracy the construction is built around. One fixed set of
        dynamics; only the window moves.
        """
        curve = _saturating(2.0, 4.0, window=24)
        fitted = [fit_growth_exponent(curve[:w])["exponent"]
                  for w in (3, 4, 6, 8, 12, 16, 24)]
        assert fitted == sorted(fitted, reverse=True)
        assert fitted[0] > 1.7 and fitted[-1] < 0.7

    def test_the_arm_that_dominates_everywhere_reads_the_smaller_exponent(self):
        """
        The sign reversal, which is what `power_law_arm` exists to refuse. Both
        curves carry the same true exponent; the faster one is larger at EVERY
        layer and receives the smaller fitted slope.
        """
        fast = _saturating(2.0, 4.0)
        slow = _saturating(2.0, 16.0)
        assert np.all(fast > slow)
        assert (fit_growth_exponent(fast)["exponent"]
                < fit_growth_exponent(slow)["exponent"])

    def test_a_window_below_the_minimum_is_refused(self):
        with pytest.raises(PatchingRefused, match="MIN_FIT_POINTS"):
            fit_growth_exponent(_power_law(1.0, window=MIN_FIT_POINTS - 1))

    def test_a_non_positive_divergence_is_a_degeneracy_and_not_a_tolerance(self):
        d = _power_law(1.0)
        d[2] = 0.0
        with pytest.raises(PatchingRefused, match="log 0"):
            fit_growth_exponent(d)

    def test_a_window_longer_than_the_curve_is_refused(self):
        with pytest.raises(PatchingRefused, match="was asked of a curve"):
            fit_growth_exponent(_power_law(1.0, window=5), window=9)


# ---------------------------------------------------------------------------
class TestTheFloorAndTheRegisteredNull:

    def test_the_floor_is_the_sign_flip_group_and_six_units_is_the_first_that_clears(self):
        got = {n: attainable_floor_report(n, n, ALPHA)["design_floor"]
               for n in range(1, 9)}
        for n, f in got.items():
            assert f == pytest.approx(2.0 / (2 ** n + 1))
        assert not attainable_floor_report(5, 5, ALPHA)["sufficient"]
        assert attainable_floor_report(6, 6, ALPHA)["sufficient"]
        assert attainable_floor_report(1, 1, ALPHA)[
            "min_informative_units_for_alpha"] == 6

    def test_a_non_informative_unit_raises_the_floor_by_claim_c_s_rule(self):
        for n in (6, 8, 10):
            for k in range(0, n + 1):
                got = attainable_floor_report(n, k, ALPHA)["design_floor"]
                assert got == pytest.approx(
                    (2.0 ** (n - k) + 1) / (2.0 ** n + 1))

    def test_the_registered_null_leaves_the_statistic_invariant(self):
        """
        Permute the pairing between the two arms' ablation points and the mean
        paired difference does not move -- so the null has no spread, every
        draw ties, and the design's floor is 1.000 whatever it is handed.
        """
        rng = np.random.default_rng(4)
        real = rng.normal(1.6, 0.3, size=9)
        ctrl = rng.normal(1.2, 0.3, size=9)
        base = float(np.mean(real - ctrl))
        for _ in range(50):
            perm = rng.permutation(9)
            assert float(np.mean(real - ctrl[perm])) == pytest.approx(base)
        rep = registered_null_invariance_report()
        assert rep["invariant_under_the_null"] is True
        assert rep["design_floor"] == 1.0

    def test_an_even_ablation_point_count_lets_a_prompt_split_evenly(self):
        """
        6l's informative-row structure, reached by a second construction. A
        prompt contributes the SUM of its points' signs, so an even count can
        sum to zero and contribute nothing to the observation or to any pattern.
        """
        for n_points in (4, 6, 8):
            assert (-1.0) ** n_points == 1.0
            signs = np.array([1.0] * (n_points // 2) + [-1.0] * (n_points // 2))
            assert signs.sum() == 0.0
        for n_points in (3, 5, 7):
            for k in range(n_points + 1):
                s = np.array([1.0] * k + [-1.0] * (n_points - k))
                assert s.sum() != 0.0


# ---------------------------------------------------------------------------
class TestTheSignFlipArm:

    def _perfect(self, sign=1.0):
        d = np.full((N_PROMPTS, N_POINTS), sign * 0.3)
        return d

    def test_a_perfect_input_lands_exactly_on_its_own_attainable_floor(self):
        arm = signflip_arm(self._perfect(), "prompt", ALPHA)
        assert arm["p_value"] == pytest.approx(
            arm["attainable_floor"]["design_floor"])
        assert arm["p_value"] == pytest.approx(2.0 / (2 ** N_PROMPTS + 1))
        assert arm["exhaustive"] is True

    def test_the_mirrored_input_lands_on_the_floor_in_the_reciprocal_tail(self):
        arm = signflip_arm(self._perfect(-1.0), "prompt", ALPHA)
        assert arm["p_reciprocal"] == pytest.approx(2.0 / (2 ** N_PROMPTS + 1))
        assert arm["p_value"] == pytest.approx(1.0)

    def test_the_arm_enumerates_and_a_second_implementation_agrees(self):
        rng = np.random.default_rng(11)
        d = rng.normal(0.05, 0.3, size=(N_PROMPTS, N_POINTS))
        arm = signflip_arm(d, "prompt", ALPHA)
        blocks = np.sign(d).sum(axis=1)
        pats = np.array(list(product((-1.0, 1.0), repeat=N_PROMPTS)))
        obs = blocks.sum()
        expect = (int((pats @ blocks >= obs - 1e-12).sum()) + 1) / (
            pats.shape[0] + 1)
        assert arm["p_value"] == pytest.approx(expect)

    def test_too_few_units_is_refused_rather_than_scored(self):
        with pytest.raises(PatchingRefused, match="attainable floor"):
            signflip_arm(np.full((5, N_POINTS), 0.3), "prompt", ALPHA)

    def test_the_sampled_regime_reports_the_floor_that_actually_binds(self):
        """
        The per-ablation-point unit has 42 units, so the group is sampled and
        the DESIGN floor of 4.5e-13 is not what a run can express -- a perfect
        input returns 4.0e-4, nine orders of magnitude away. 6i found exactly
        this in CLAIM-B's sampled pairing regime and 6p's rule is that the
        smallest expressible p is the MAX of the design floor and the sampling
        resolution. Both units are checked here because they bind at opposite
        ends.
        """
        for unit in P_AB1_UNITS:
            arm = signflip_arm(self._perfect(), unit, ALPHA)
            floor = arm["attainable_floor"]
            assert arm["p_value"] == pytest.approx(floor["attainable_floor"])
            if arm["exhaustive"]:
                assert floor["sampling_floor"] is None
                assert floor["binds"] == "design"
            else:
                assert floor["binds"] == "sampling"
                assert floor["sampling_floor"] > floor["design_floor"]
                assert floor["attainable_floor"] == pytest.approx(
                    2.0 / (arm["n_patterns"] + 1))

    def test_the_floor_report_takes_the_max_and_not_either_one(self):
        big = attainable_floor_report(40, 40, ALPHA, n_patterns=99)
        assert big["binds"] == "sampling"
        assert big["attainable_floor"] == pytest.approx(2.0 / 100)
        small = attainable_floor_report(3, 3, ALPHA, n_patterns=10 ** 9)
        assert small["binds"] == "design"
        assert small["attainable_floor"] == pytest.approx(2.0 / 9)
        assert small["sufficient"] is False

    def test_an_unknown_unit_is_refused(self):
        with pytest.raises(PatchingRefused, match="not one of"):
            signflip_arm(self._perfect(), "layer", ALPHA)

    def test_the_two_units_are_the_two_readings_and_nothing_else(self):
        assert P_AB1_UNITS == ("prompt", "ablation_point")


# ---------------------------------------------------------------------------
class TestThePowerLawRefusal:

    def _pair(self, real_fn, ctrl_fn, seed=0):
        rng = np.random.default_rng(seed)
        real = [[real_fn(rng) for _ in range(N_POINTS)] for _ in range(N_PROMPTS)]
        ctrl = [[ctrl_fn(rng) for _ in range(N_POINTS)] for _ in range(N_PROMPTS)]
        return paired_exponents(real, ctrl)

    def test_pure_power_laws_are_admitted(self):
        pair = self._pair(lambda r: _power_law(1.4, rng=r, sd=0.2),
                          lambda r: _power_law(1.4, rng=r, sd=0.2), seed=3)
        pl = power_law_arm(pair["bend_z_real"], pair["bend_z_control"], ALPHA)
        assert pl["not_a_power_law"] is False

    def test_a_saturating_arm_is_refused(self):
        pair = self._pair(lambda r: _saturating(1.4, 5.0, rng=r, sd=0.2),
                          lambda r: _power_law(1.4, rng=r, sd=0.2), seed=3)
        pl = power_law_arm(pair["bend_z_real"], pair["bend_z_control"], ALPHA)
        assert pl["not_a_power_law"] is True
        assert pl["real"]["rejects"] is True
        assert pl["control"]["rejects"] is False

    def test_the_level_is_alpha_over_two_because_there_are_two_arms(self):
        pair = self._pair(lambda r: _power_law(1.4, rng=r, sd=0.2),
                          lambda r: _power_law(1.4, rng=r, sd=0.2), seed=5)
        pl = power_law_arm(pair["bend_z_real"], pair["bend_z_control"], ALPHA)
        assert pl["per_arm_level"] == pytest.approx(ALPHA / 2.0)

    def test_the_discarded_contrast_arm_is_reported_and_read_by_nothing(self):
        """
        It was the first refusal and it is too weak; it stays because it names
        the direction of the confound. What must not happen is it quietly
        becoming a refusal again.
        """
        pair = self._pair(lambda r: _saturating(1.4, 5.0, rng=r, sd=0.2),
                          lambda r: _saturating(1.4, 15.0, rng=r, sd=0.2), seed=7)
        bend = bend_contrast_arm(pair["bend_contrast"], "prompt", ALPHA)
        assert set(bend) >= {"p_two_sided", "confounded"}
        res = p_value_p_ab1(
            [[_saturating(1.4, 5.0) for _ in range(N_POINTS)]
             for _ in range(N_PROMPTS)],
            [[_saturating(1.4, 15.0) for _ in range(N_POINTS)]
             for _ in range(N_PROMPTS)],
            _mags(), _mags(), alpha=ALPHA)
        assert res["p_value"] is None
        assert "not a power law" in res["reason"]


# ---------------------------------------------------------------------------
class TestTheGate:

    def _run(self, real_beta, ctrl_beta, seed=1, **kw):
        rng = np.random.default_rng(seed)
        real = [[_power_law(real_beta, rng=rng, sd=0.15) for _ in range(N_POINTS)]
                for _ in range(N_PROMPTS)]
        ctrl = [[_power_law(ctrl_beta, rng=rng, sd=0.15) for _ in range(N_POINTS)]
                for _ in range(N_PROMPTS)]
        return p_value_p_ab1(real, ctrl, _mags(), _mags(), alpha=ALPHA, **kw)

    def test_the_predicted_direction_reaches_propagates(self):
        res = self._run(2.0, 1.0)
        assert res["verdict"] == "PROPAGATES"
        assert res["falsified"] is False
        assert res["p_value"] == pytest.approx(2.0 / (2 ** N_PROMPTS + 1))

    def test_the_reversed_direction_fires_the_falsification_branch(self):
        """
        6i's requirement: a verdict branch nothing can trigger is 6h's audit arm
        wearing a different hat. RECAPTURES is P-AB1's falsifier.
        """
        res = self._run(1.0, 2.0)
        assert res["verdict"] == "RECAPTURES"
        assert res["falsified"] is True
        assert res["p_reciprocal"] == pytest.approx(2.0 / (2 ** N_PROMPTS + 1))

    def test_two_arms_from_one_law_reach_neither_branch(self):
        res = self._run(1.4, 1.4, seed=9)
        assert res["verdict"] == "INSUFFICIENT"
        assert res["falsified"] is False

    def test_a_mismatched_control_magnitude_is_refused(self):
        rng = np.random.default_rng(2)
        real = [[_power_law(2.0, rng=rng, sd=0.1) for _ in range(N_POINTS)]
                for _ in range(N_PROMPTS)]
        ctrl = [[_power_law(1.0, rng=rng, sd=0.1) for _ in range(N_POINTS)]
                for _ in range(N_PROMPTS)]
        res = p_value_p_ab1(real, ctrl, _mags(), _mags() * 1.01, alpha=ALPHA)
        assert res["p_value"] is None
        assert "EQUAL magnitude" in res["reason"]
        assert res["verdict"] == "INSUFFICIENT"

    def test_magnitudes_that_do_not_index_the_grid_are_refused(self):
        """
        An unchecked match is what this refusal exists for, so magnitudes for a
        different grid must not silently satisfy it -- 6p's `P-S1`, where a null
        drawn at one arm's configuration was applied to the other's and nothing
        checked they matched.
        """
        rng = np.random.default_rng(3)
        real = [[_power_law(2.0, rng=rng, sd=0.1) for _ in range(N_POINTS)]
                for _ in range(N_PROMPTS)]
        ctrl = [[_power_law(1.0, rng=rng, sd=0.1) for _ in range(N_POINTS)]
                for _ in range(N_PROMPTS)]
        res = p_value_p_ab1(real, ctrl, np.ones(3), np.ones(3), alpha=ALPHA)
        assert res["p_value"] is None
        assert "do not index the curves" in res["reason"]

    def test_the_magnitude_check_is_numerical_identity_and_not_a_tolerance(self):
        rep = magnitude_match_report(np.ones(4), np.ones(4) * (1 + 1e-12))
        assert rep["matched"] is True
        rep = magnitude_match_report(np.ones(4), np.ones(4) * (1 + 1e-6))
        assert rep["matched"] is False

    def test_a_ragged_ablation_grid_is_refused_rather_than_weighted(self):
        real = [[_power_law(1.5) for _ in range(N_POINTS)]
                for _ in range(N_PROMPTS)]
        real[2] = real[2][:-1]
        ctrl = [[_power_law(1.5) for _ in range(N_POINTS)]
                for _ in range(N_PROMPTS)]
        ctrl[2] = ctrl[2][:-1]
        with pytest.raises(PatchingRefused, match="ragged"):
            paired_exponents(real, ctrl)

    def test_a_point_too_close_to_the_output_is_refused_and_not_dropped(self):
        """
        Dropping a point changes the unit count and therefore the floor, so it
        is the caller's decision about the grid and not the gate's.
        """
        real = [[_power_law(1.5) for _ in range(N_POINTS)]
                for _ in range(N_PROMPTS)]
        ctrl = [[_power_law(1.5) for _ in range(N_POINTS)]
                for _ in range(N_PROMPTS)]
        for p in range(N_PROMPTS):
            real[p][-1] = real[p][-1][:2]
            ctrl[p][-1] = ctrl[p][-1][:2]
        with pytest.raises(PatchingRefused, match="Nothing is dropped here"):
            paired_exponents(real, ctrl)

    def test_the_window_report_reads_the_grid_and_nothing_else(self):
        rep = window_reference_report([8, 7, 6, 5, 4, 3, 2])
        assert rep["common_window"] == 2
        assert rep["points_below_min_fit"] == [6]
        assert rep["usable"] is False
        assert rep["n_ablation_points"] == 7

    def test_the_alternatives_are_fixed_in_advance(self):
        assert P_AB1_ALTERNATIVE == "greater"
        assert P_AB1_RECIPROCAL_ALTERNATIVE == "less"

    def test_the_verdict_lattice_never_accepts_the_null(self):
        assert gate_verdict(None, None, ALPHA)["verdict"] == "INSUFFICIENT"
        assert gate_verdict(0.9, 0.9, ALPHA)["verdict"] == "INSUFFICIENT"
        assert gate_verdict(0.01, 0.99, ALPHA)["verdict"] == "PROPAGATES"
        assert gate_verdict(0.99, 0.01, ALPHA)["falsified"] is True


# ---------------------------------------------------------------------------
class TestTheSharedFactorDiagnostic:

    def test_a_purely_within_prompt_difference_reads_near_zero(self):
        rng = np.random.default_rng(6)
        d = rng.normal(0.0, 1.0, size=(N_PROMPTS, N_POINTS))
        got = shared_prompt_factor_diagnostic(d)["shared_share_estimate"]
        assert got < 0.35

    def test_a_purely_between_prompt_difference_reads_near_one(self):
        d = np.repeat(np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]]),
                      N_POINTS, axis=1)
        got = shared_prompt_factor_diagnostic(d)["shared_share_estimate"]
        assert got == pytest.approx(1.0)

    def test_a_fixed_offset_common_to_every_prompt_is_invisible_to_it(self):
        """
        The limitation the module docstring states, pinned so it cannot be
        mistaken for something the diagnostic covers.
        """
        rng = np.random.default_rng(8)
        d = rng.normal(0.0, 1.0, size=(N_PROMPTS, N_POINTS))
        a = shared_prompt_factor_diagnostic(d)["shared_share_estimate"]
        b = shared_prompt_factor_diagnostic(d + 5.0)["shared_share_estimate"]
        assert b == pytest.approx(a)


# ---------------------------------------------------------------------------
class TestAdjudicationIsRefused:

    def test_no_unit_is_registered_so_nothing_can_be_adjudicated(self):
        assert REGISTERED_EXCHANGEABLE_UNIT is None
        res = {"unit_computed": "prompt", "p_value": 0.01}
        with pytest.raises(PatchingRefused,
                           match="REGISTERED_EXCHANGEABLE_UNIT is None"):
            adjudicate_p_ab1(res, adjudicate=True)

    def test_the_unit_argument_does_not_route_around_the_constant(self):
        """
        `unit=` selects what to COMPUTE. The module constant decides what may
        enter an e-process -- 6h's construction, and its reason.
        """
        rng = np.random.default_rng(12)
        real = [[_power_law(2.0, rng=rng, sd=0.15) for _ in range(N_POINTS)]
                for _ in range(N_PROMPTS)]
        ctrl = [[_power_law(1.0, rng=rng, sd=0.15) for _ in range(N_POINTS)]
                for _ in range(N_PROMPTS)]
        res = p_value_p_ab1(real, ctrl, _mags(), _mags(),
                            unit="ablation_point", alpha=ALPHA)
        assert res["p_value"] is not None
        with pytest.raises(PatchingRefused):
            adjudicate_p_ab1(res, adjudicate=True)


# ---------------------------------------------------------------------------
class TestCommittedCalibration:
    """
    The measured rates, pinned. Recomputing them takes ~110 seconds, which the
    ten-second gating tier does not have -- the same division of labour
    `claims/calibration/changepoint_colocation.json` and
    `claims/calibration/steering_sign.json` use. What must not happen is the
    record drifting silently, so it is asserted rather than read.
    """

    def _doc(self):
        from tools.calibrate_patching_exponent import OUT_PATH, SCHEMA_VERSION
        doc = json.loads(OUT_PATH.read_text())
        assert doc["schema_version"] == SCHEMA_VERSION
        return doc

    def _row(self, family, unit):
        rows = [r for r in self._doc()["validity"]
                if r["family"] == family and r["unit"] == unit]
        assert len(rows) == 1, (family, unit)
        return rows[0]

    def test_the_artifact_describes_the_design_it_measured(self):
        doc = self._doc()
        assert doc["alpha"] == ALPHA
        assert doc["design"]["n_prompts"] == N_PROMPTS
        assert doc["design"]["n_ablation_points"] == N_POINTS
        assert doc["design"]["min_fit_points"] == MIN_FIT_POINTS
        assert doc["replicates"] >= 300
        assert doc["design"]["n_ablation_points"] % 2 == 1
        assert len(doc["validity"]) == 18

    def test_valid_under_the_plain_h0_in_both_units(self):
        for unit in P_AB1_UNITS:
            r = self._row("independent", unit)
            assert r["rejection_rate"] <= 0.09      # 400 draws, nominal 0.05
            assert r["reciprocal_rejection_rate"] <= 0.09

    def test_the_per_ablation_point_unit_inflates_with_a_shared_prompt_factor(self):
        """
        The row that decides the unit, and the reason
        `REGISTERED_EXCHANGEABLE_UNIT` is None rather than guessed.
        """
        for fam in ("shared-prompt-factor@0.5", "shared-prompt-factor@1.0"):
            assert self._row(fam, "prompt")["rejection_rate"] <= 0.09
        assert self._row("shared-prompt-factor@0.5",
                         "ablation_point")["rejection_rate"] >= 0.10
        assert self._row("shared-prompt-factor@1.0",
                         "ablation_point")["rejection_rate"] >= 0.15

    def test_the_fixed_offset_limitation_is_severe_under_BOTH_units(self):
        """
        6i's shared-per-unit-factor at 1.00, here. A confound present in every
        cell is present under every sign pattern, and no choice of unit removes
        it -- which is why the module docstring states it rather than implying
        the diagnostic covers it.
        """
        for unit in P_AB1_UNITS:
            assert self._row("fixed-offset@1.0jitter",
                             unit)["rejection_rate"] >= 0.80

    def test_both_verdict_branches_fire_on_the_input_built_for_them(self):
        for unit in P_AB1_UNITS:
            assert self._row("power@+0.15", unit)["rejection_rate"] >= 0.40
            assert self._row("power@-0.15",
                             unit)["reciprocal_rejection_rate"] >= 0.40

    def test_the_power_law_refusal_costs_verdicts_and_the_record_says_so(self):
        """
        6o's third category. The refusal turns away every draw of both
        saturating families -- including the symmetric one where the contrast
        was nominal -- and the counterfactual columns are what say what that
        cost, rather than a claim that it was small.
        """
        for unit in P_AB1_UNITS:
            for fam in ("saturating-both-arms", "differential-saturation"):
                r = self._row(fam, unit)
                assert r["emitted"] == 0
                assert r["refused_not_a_power_law"] == r["replicates"]
            # and what it prevented: the falsification branch at essentially 1
            ds = self._row("differential-saturation", unit)
            assert ds["counterfactual_reciprocal_rate_no_bend_refusal"] >= 0.95

    def test_the_discarded_refusal_is_recorded_with_why_it_was_discarded(self):
        d = self._doc()["discarded_refusal"]
        for unit in P_AB1_UNITS:
            v = d["per_unit"][unit]
            assert v["let_through"] > 0
            assert v["reciprocal_rejection_among_those_let_through"] >= 0.95

    def test_the_power_law_arm_is_nominal_on_the_shape_it_admits(self):
        curve = {r["tau"]: r["refusal_rate_one_arm_at_0.05"]
                 for r in self._doc()["power_law_arm_operating_curve"]}
        straight = max(curve)
        assert curve[straight] <= 0.10
        assert curve[8.0] >= 0.90
        assert curve[5.0] >= 0.95

    def test_the_window_dependence_is_monotone_and_deterministic(self):
        for row in self._doc()["window_dependence"]:
            fitted = [row["fitted"][str(w)] for w in (3, 4, 6, 8, 12, 16, 24)]
            assert fitted == sorted(fitted, reverse=True)
            if row["tau"] > 1e5:
                assert all(f == pytest.approx(row["beta_true"]) for f in fitted)

    def test_the_sign_sum_was_chosen_on_a_measurement(self):
        rows = {r["planted_exponent_gap"]: r
                for r in self._doc()["statistic_choice"]}
        assert rows[0.0]["sign_sum_rejection"] <= rows[0.0][
            "mean_difference_rejection"]
        for gap in (0.05, 0.10):
            assert (rows[gap]["sign_sum_rejection"]
                    >= rows[gap]["mean_difference_rejection"])

    def test_the_even_odd_arithmetic_is_exact_and_committed(self):
        rows = {r["ablation_points_per_prompt"]: r
                for r in self._doc()["grid_arithmetic"]["even_vs_odd"]}
        for n in (3, 5, 7, 9, 11):
            assert rows[n]["p_prompt_non_informative_under_h0"] == 0.0
            assert rows[n]["p_design_can_reject_at_all"] == 1.0
        for n in (4, 6, 8):
            assert rows[n]["p_prompt_non_informative_under_h0"] == pytest.approx(
                math.comb(n, n // 2) / 2 ** n)
            assert rows[n]["p_design_can_reject_at_all"] < 0.50
