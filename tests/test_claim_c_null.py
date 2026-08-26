"""
tests/test_claim_c_null.py — CLAIM-C's null construction.

CLAIM-C is the only registered prediction with a hard stop attached
(`PREDICTIONS.md`: "If this fails, no checkpoint-sweep work (items 9-11)
proceeds past the gate"), which raises the stakes on two things these tests
are about.

**Calibration.** A p-value that is not valid under H0 produces an e-value whose
expectation exceeds 1 and voids `E[E] <= 1` for H-TRANSFER. Here the null is an
exhaustive enumeration rather than a sample, so the conditional test is exact
and the calibration checks can be sharper than a Monte-Carlo null would allow.

**Refusal.** The other half is that the gate must refuse rather than emit a
convenient number. The attainable-floor refusal is the one worth staring at: a
permutation over four prompts cannot express a p below 0.118, so a *perfect*
result reports "not significant" — and on a claim with a hard stop, a
meaningless "not significant" is worse than no number, because it reads as
evidence against transfer.

Sample sizes are kept inside the pure tier's budget, and the calibration test
says how coarse it is rather than implying more resolution than its trial count
can give.
"""

from __future__ import annotations

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable.
pytestmark = pytest.mark.pure

from p1_mstate_tracking.replication_gate import (
    CLAIM_C_ALTERNATIVE,
    CLAIM_C_METRICS,
    CLAIM_C_RECIPROCAL_ALTERNATIVE,
    DEPTH_GRID_POINTS,
    adjudicate_claim_c,
    attainable_p,
    claim_c_concordance,
    contrast,
    gate_verdict,
    n_informative_rows,
    p_value_claim_c,
    profile_distance,
    profiles_from_run_dir,
    resample_depth,
    row_swing,
)

# gpt2-large has 36 layers, pythia-1.4b has 24 -- every fixture uses both
# depths so nothing passes by accident on equal-length profiles.
REF_LAYERS, CAN_LAYERS = 36, 24


def _arm(deltas: np.ndarray, prompts, n_layers: int, base: float = 0.0):
    """
    A `{prompt: {metric: profile}}` arm whose trained-minus-random contrast is
    exactly `deltas[i, j]`. Built as a constant offset so the cell statistic is
    controlled directly and the tests are about the null, not the extractor.
    """
    return {p: {m: [base + float(deltas[i, j])] * n_layers
                for j, m in enumerate(CLAIM_C_METRICS)}
            for i, p in enumerate(prompts)}


def _pair(sign_table: np.ndarray, prompts, n_layers: int, magnitude: float = 1.0):
    """(trained, random) arms realising the given per-cell contrast signs."""
    random_ = _arm(np.zeros_like(sign_table, dtype=float), prompts, n_layers)
    trained = _arm(sign_table * magnitude, prompts, n_layers)
    return trained, random_


def _prompts(n: int):
    return [f"prompt_{i}" for i in range(n)]


def _heterogeneous(n: int, seed: int = 11) -> np.ndarray:
    """
    A per-prompt sign table whose rows are NOT all identical.

    Fixtures have to be built this way on purpose. Every prompt carrying the
    same sign pattern is the exact degeneracy `p_value_claim_c` refuses -- the
    prompts then contribute one observation, and enumerating 2^n patterns over
    it is the wrong null rather than a conservative one. An all-ones fixture
    would have tested the refusal, not the test.
    """
    rng = np.random.default_rng(seed)
    while True:
        s = rng.choice([-1.0, 1.0], size=(n, len(CLAIM_C_METRICS)))
        if len({tuple(r) for r in s.tolist()}) > 1:
            return s


def _four_arms(ref_signs, can_signs, prompts):
    rt, rr = _pair(ref_signs, prompts, REF_LAYERS)
    ct, cr = _pair(can_signs, prompts, CAN_LAYERS)
    return rt, rr, ct, cr


def _step0(prompts):
    """A step-0 arm distinct from the norm-matched one, so the two are not
    interchangeable in any test that passes."""
    return _arm(np.full((len(prompts), len(CLAIM_C_METRICS)), 0.25),
                prompts, CAN_LAYERS)


# ---------------------------------------------------------------------------
# Depth normalization
# ---------------------------------------------------------------------------

class TestDepthNormalization:

    def test_profiles_of_different_depth_are_comparable(self):
        """
        The whole comparison is between a 36-layer model and a 24-layer one.
        A linear ramp is the same object on both, and must resample to the
        same grid.
        """
        a = resample_depth(np.linspace(0, 1, REF_LAYERS))
        b = resample_depth(np.linspace(0, 1, CAN_LAYERS))
        assert a.shape == b.shape == (DEPTH_GRID_POINTS,)
        assert np.allclose(a, b, atol=1e-12)

    def test_nan_aware_and_refuses_when_too_few_points(self):
        v = [1.0, np.nan, 3.0, np.nan]
        assert np.all(np.isfinite(resample_depth(v)))
        assert resample_depth([1.0, np.nan, np.nan]) is None
        assert resample_depth([1.0]) is None

    def test_contrast_is_antisymmetric(self):
        """
        The permutation is an exact sign flip only because delta is
        antisymmetric in (trained, random). If this ever stops holding, the
        null stops being the null.
        """
        a = list(np.linspace(0.2, 0.9, REF_LAYERS))
        b = list(np.linspace(0.5, 0.1, CAN_LAYERS))
        assert contrast(a, b) == pytest.approx(-contrast(b, a))

    def test_profile_distance_is_a_diagnostic_not_a_sign(self):
        a = [1.0] * REF_LAYERS
        assert profile_distance(a, [2.0] * CAN_LAYERS) == pytest.approx(1.0)
        assert profile_distance(a, [0.0] * CAN_LAYERS) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------

class TestConcordanceStatistic:

    def test_perfect_agreement_scores_every_cell(self):
        prompts = _prompts(8)
        s = _heterogeneous(8)
        r = claim_c_concordance(*_four_arms(s, s, prompts),
                                candidate_step0=_step0(prompts))
        assert r["observed"] == r["n_cells"] == 8 * len(CLAIM_C_METRICS)
        assert r["concordance_fraction"] == pytest.approx(1.0)

    def test_perfect_inversion_scores_nothing(self):
        prompts = _prompts(8)
        s = _heterogeneous(8)
        r = claim_c_concordance(*_four_arms(s, -s, prompts),
                                candidate_step0=_step0(prompts))
        assert r["observed"] == 0
        assert r["n_cells"] == 8 * len(CLAIM_C_METRICS)

    def test_magnitude_is_deliberately_invisible(self):
        """
        The chosen criterion is ordinal. A candidate contrast a thousand times
        smaller than the reference's still scores concordant -- that is the
        stated cost of the criterion, and it is tested so nobody later reads
        the number as an agreement in magnitude.
        """
        prompts = _prompts(8)
        s = _heterogeneous(8)
        rt, rr = _pair(s, prompts, REF_LAYERS, magnitude=1000.0)
        ct, cr = _pair(s, prompts, CAN_LAYERS, magnitude=0.001)
        r = claim_c_concordance(rt, rr, ct, cr, candidate_step0=_step0(prompts))
        assert r["concordance_fraction"] == pytest.approx(1.0)
        # ...and the diagnostic that WOULD have seen it is reported.
        assert any(v is not None and v > 1.0 for v in r["arm_distances"].values())

    def test_zero_contrast_cells_are_dropped_not_scored(self):
        prompts = _prompts(8)
        ref = _heterogeneous(8)
        can = ref.copy()
        can[0, 0] = 0.0                       # no sign -> not a cell
        r = claim_c_concordance(*_four_arms(ref, can, prompts),
                                candidate_step0=_step0(prompts))
        assert r["n_cells"] == 8 * len(CLAIM_C_METRICS) - 1
        assert r["n_cells_dropped"] == 1
        assert r["observed"] == r["n_cells"]

    def test_prompt_missing_one_metric_is_dropped_whole(self):
        """
        The metric set is fixed. Running on five of six because one field was
        absent would be a per-run re-choice of the statistic, so the PROMPT
        goes -- and the record says which and why.
        """
        prompts = _prompts(8)
        s = _heterogeneous(8)
        rt, rr, ct, cr = _four_arms(s, s, prompts)
        del ct["prompt_3"]["cka_prev"]
        r = claim_c_concordance(rt, rr, ct, cr, candidate_step0=_step0(prompts))
        assert "prompt_3" not in r["prompts_used"]
        assert "cka_prev" in r["prompts_dropped"]["prompt_3"]
        assert r["n_prompts"] == 7


# ---------------------------------------------------------------------------
# Calibration, power, direction
# ---------------------------------------------------------------------------

class TestCalibration:

    def _p(self, ref_signs, can_signs, prompts, **kw):
        return p_value_claim_c(*_four_arms(ref_signs, can_signs, prompts),
                               candidate_step0=_step0(prompts), **kw)

    def test_null_is_enumerated_exhaustively_for_eight_prompts(self):
        prompts = _prompts(8)
        s = _heterogeneous(8)
        r = self._p(s, s, prompts)
        assert r["null_exhaustive"] is True
        assert r["n_null_patterns"] == 2 ** 8

    def test_valid_under_h0(self):
        """
        Reference and candidate contrast signs drawn independently: the
        rejection rate at alpha must not exceed alpha. The null is an
        EXHAUSTIVE enumeration, so each subset's test is exact and the bound is
        a real one rather than a Monte-Carlo approximation.

        **The reported p is deliberately NOT uniform under H0.** It is an
        intersection-union max over seven subsets, and a max of p-values is
        stochastically larger than uniform by construction -- that conservatism
        IS the unanimity rule, not a calibration defect. So the assertion here
        is one-sided on validity, and the mean is asserted to sit ABOVE 0.5
        rather than at it. A max-p that came back uniform would mean the
        subsets were perfectly redundant and the tool axis was buying nothing.

        **The rate is CONDITIONAL ON EMISSION**, which is the project's standing
        rule (POPPER_PLAN.md 6g, and 6k for what ignoring it hides). Roughly a
        fifth of these draws now refuse on the informative-row floor: with six
        metrics a prompt splits 3-3 with probability 20/64, such a row cannot
        move the statistic, and a table with fewer than five rows that can move
        has a floor above alpha. Those draws never reach a ledger, so counting
        them as non-rejections would flatter the rate. Both numbers are asserted
        -- the conditional one because it is what governs the ledger, and the
        emission rate itself so that a refusal quietly swallowing the whole
        sample would fail here rather than look like perfect calibration.

        120 trials catches a systematic violation, not a subtle one.
        """
        rng = np.random.default_rng(20260824)
        prompts, ps, refused = _prompts(8), [], 0
        for _ in range(120):
            ref = rng.choice([-1.0, 1.0], size=(8, len(CLAIM_C_METRICS)))
            can = rng.choice([-1.0, 1.0], size=(8, len(CLAIM_C_METRICS)))
            r = self._p(ref, can, prompts)
            if r["p_value"] is None:
                refused += 1
                continue
            ps.append(r["p_value"])
        ps = np.asarray(ps)
        # The gate is calibrated by CONTROLLING, not by refusing: most draws
        # must still emit, or the rate below is a rate over a selected few.
        assert len(ps) >= 0.5 * 120, f"only {len(ps)} of 120 H0 draws emitted"
        assert refused > 0, ("no H0 draw hit the informative-row floor; at eight "
                             "prompts and six metrics it should catch about a fifth")
        assert np.all((ps > 0) & (ps <= 1.0))
        assert (ps <= 0.05).mean() <= 0.05 + 1e-9
        assert ps.mean() > 0.5                # conservative, as an IUT must be

    def test_the_reported_p_is_never_better_than_the_full_set(self):
        """
        Unanimity can only cost power, never add it. If the max over subsets
        ever came back below the full-set p, the max would not be a max and the
        Type-I guarantee would be coming from the wrong place.
        """
        rng = np.random.default_rng(4)
        prompts, compared = _prompts(8), 0
        for _ in range(25):
            ref = rng.choice([-1.0, 1.0], size=(8, len(CLAIM_C_METRICS)))
            can = rng.choice([-1.0, 1.0], size=(8, len(CLAIM_C_METRICS)))
            r = self._p(ref, can, prompts)
            if r["p_value"] is None:            # refused; there is no max to check
                continue
            compared += 1
            assert r["p_value"] >= r["p_full_set"] - 1e-12
            assert r["p_reciprocal"] >= r["p_reciprocal_full_set"] - 1e-12
        assert compared >= 15, f"only {compared} of 25 draws emitted a p to compare"

    def test_power_against_perfect_transfer(self):
        prompts = _prompts(8)
        s = _heterogeneous(8)
        r = self._p(s, s, prompts)
        assert r["p_value"] == pytest.approx(2.0 / 257.0)
        assert r["p_value"] < 0.05
        assert r["verdict"] == "TRANSFERS"
        assert r["hard_stop"] is False
        assert r["falsified"] is False

    def test_power_survives_a_dissenting_metric(self):
        """One metric inverting everywhere should not sink a real transfer."""
        prompts = _prompts(8)
        ref = _heterogeneous(8)
        can = ref.copy()
        can[:, 2] = -ref[:, 2]
        r = self._p(ref, can, prompts)
        assert r["p_value"] < 0.05
        assert r["verdict"] == "TRANSFERS"
        assert r["per_metric_concordant"][CLAIM_C_METRICS[2]] == 0

    def test_the_direction_is_fixed_and_inversion_falsifies(self):
        """
        Swapping the candidate's condition arms must not produce a second
        significant result in the other tail. Perfect inversion is CLAIM-C's
        falsifier positively shown, and that is a different verdict from
        'transfer not demonstrated'.
        """
        prompts = _prompts(8)
        s = _heterogeneous(8)
        r = self._p(s, -s, prompts)
        assert r["p_value"] == pytest.approx(1.0)
        assert r["p_reciprocal"] == pytest.approx(2.0 / 257.0)
        assert r["verdict"] == "FAILS-TO-TRANSFER"
        assert r["hard_stop"] is True
        assert r["falsified"] is True
        assert r["alternative"] == CLAIM_C_ALTERNATIVE

    def test_middling_agreement_is_insufficient_not_falsified(self):
        """
        The verdict an e-process is allowed to reach when nothing was shown.
        The sweep still stops; CLAIM-C is not recorded as falsified.
        """
        rng = np.random.default_rng(7)
        prompts = _prompts(8)
        ref = _heterogeneous(8)
        can = rng.choice([-1.0, 1.0], size=(8, len(CLAIM_C_METRICS)))
        r = self._p(ref, can, prompts)
        assert r["verdict"] == "INSUFFICIENT"
        assert r["hard_stop"] is True
        assert r["falsified"] is False

    def test_reference_is_held_fixed(self):
        """
        gpt2-large is the phenomenology being reproduced, not a random draw.
        Permuting ITS labels instead would be a different (and wrong) test, so
        the statistic must be invariant to relabelling both sides together --
        which is exactly what happens if the reference is not conditioned on.
        """
        prompts = _prompts(8)
        s = _heterogeneous(8)
        both_flipped = self._p(-s, -s, prompts)
        neither = self._p(s, s, prompts)
        assert both_flipped["p_value"] == pytest.approx(neither["p_value"])
        assert both_flipped["observed"] == neither["observed"]


class TestRefusals:

    def _p(self, prompts, **kw):
        s = _heterogeneous(len(prompts))
        return p_value_claim_c(*_four_arms(s, s, prompts),
                               candidate_step0=_step0(prompts), **kw)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_refuses_when_no_p_below_alpha_is_attainable(self, n):
        """
        With n prompts the enumeration's smallest expressible p is
        2/(2^n + 1): 0.4 at n=2, 0.118 at n=4, 0.061 at n=5. A test that
        cannot reject on a PERFECT result is not a test, and on a claim with a
        hard stop its 'not significant' would read as evidence against
        transfer. Refuse, and say more prompts are needed rather than a
        different threshold.
        """
        r = self._p(_prompts(n))
        assert r["p_value"] is None
        assert "cannot reject on a PERFECT result" in r["reason"]
        assert r["verdict"] == "INSUFFICIENT"
        assert r["hard_stop"] is True
        assert r["falsified"] is False

    def test_six_prompts_is_the_first_workable_gate(self):
        r = self._p(_prompts(6))
        assert r["p_value"] == pytest.approx(2.0 / 65.0)
        assert r["best_attainable_p"] <= r["alpha"]

    def test_refuses_when_no_cell_has_a_sign(self):
        prompts = _prompts(8)
        z = np.zeros((8, len(CLAIM_C_METRICS)))
        r = p_value_claim_c(*_four_arms(z, z, prompts),
                            candidate_step0=_step0(prompts))
        assert r["p_value"] is None
        assert "exactly zero" in r["reason"]
        assert r["hard_stop"] is True

    def test_refuses_when_arms_share_too_few_prompts(self):
        prompts = _prompts(8)
        s = _heterogeneous(8)
        rt, rr, ct, cr = _four_arms(s, s, prompts)
        for p in prompts[1:]:
            del ct[p]
        r = p_value_claim_c(rt, rr, ct, cr, candidate_step0=None,
                            step0_absent_reason="synthetic fixture")
        assert r["p_value"] is None
        assert "permutation unit is the prompt" in r["reason"]


class TestToolAxis:
    """
    The second agreement axis: metric leave-one-out, unanimity in both
    directions. Together the two axes are a stronger argument; a single
    disagreement is not a death sentence, it is INSUFFICIENT.
    """

    def _p(self, ref, can, prompts):
        return p_value_claim_c(*_four_arms(ref, can, prompts),
                               candidate_step0=_step0(prompts))

    def test_all_seven_subsets_are_scored(self):
        prompts = _prompts(8)
        s = _heterogeneous(8)
        r = self._p(s, s, prompts)
        assert set(r["subsets"]) == {"all"} | {f"drop:{m}" for m in CLAIM_C_METRICS}
        assert r["n_subsets"] == 7
        assert r["tool_axis"] == "metric-leave-one-out"
        assert r["tool_rule"] == "unanimity"

    def test_every_subset_shares_one_prompt_set_and_one_null(self):
        """
        A max over p-values is only meaningful if the tests are the same shape.
        Prompt eligibility is decided once from the full six-metric
        requirement; leave-one-out drops COLUMNS only.
        """
        prompts = _prompts(8)
        s = _heterogeneous(8)
        r = self._p(s, s, prompts)
        sizes = {v["n_null_patterns"] for v in r["subsets"].values()}
        assert sizes == {2 ** 8}
        assert all(v["null_exhaustive"] for v in r["subsets"].values())

    def test_unanimous_transfer_still_clears(self):
        prompts = _prompts(8)
        s = _heterogeneous(8)
        r = self._p(s, s, prompts)
        assert r["verdict"] == "TRANSFERS"
        assert r["p_value"] == pytest.approx(2.0 / 257.0)

    def test_a_result_carried_by_one_metric_is_no_longer_a_transfer(self):
        """
        THE POINT OF THE AXIS. Build a case where the full set clears but the
        agreement lives almost entirely in one metric: dropping that metric
        must stop the gate calling it transfer. Before the tool axis this
        scored TRANSFERS on the strength of a single measurement.
        """
        prompts = _prompts(8)
        rng = np.random.default_rng(3)
        ref = _heterogeneous(8, seed=5)
        can = -ref.copy()                       # every metric inverts...
        can[:, 0] = ref[:, 0]                   # ...except one, which agrees
        r = self._p(ref, can, prompts)
        assert r["subsets"]["all"]["p_value"] > 0.05
        assert r["verdict"] != "TRANSFERS"
        # and the record names which subset was binding, so the reader is not
        # left to re-derive it
        assert r["binding_subset"] in r["subsets"]

    def test_one_dissenting_metric_is_not_a_death_sentence(self):
        """
        The other half of the rule. A single metric inverting while the rest
        agree must NOT come back FALSIFIED -- there are instrument quirks we
        are not privy to, and one of them is not a demonstrated inversion.
        """
        prompts = _prompts(8)
        ref = _heterogeneous(8, seed=9)
        can = ref.copy()
        can[:, 3] = -ref[:, 3]
        r = self._p(ref, can, prompts)
        assert r["falsified"] is False
        assert r["verdict"] in ("TRANSFERS", "INSUFFICIENT")

    def test_a_strong_inversion_survives_one_dissenting_metric(self):
        """
        What unanimity does NOT mean, pinned because it is easy to misread.

        The rule is "no subset may fail", not "no metric may dissent". Five of
        six metrics inverting on every prompt IS a demonstrated inversion, and
        it survives every leave-one-out because no single metric is carrying
        it. Reading the rule the other way would let one quirky measurement
        veto a real result and make the gate effectively unfalsifiable -- the
        opposite of what the axis is for.
        """
        prompts = _prompts(8)
        ref = _heterogeneous(8, seed=13)
        can = -ref.copy()
        can[:, 2] = ref[:, 2]                   # one metric declines to invert
        r = self._p(ref, can, prompts)
        assert r["verdict"] == "FAILS-TO-TRANSFER"
        assert all(v["p_reciprocal"] <= 0.05 for v in r["subsets"].values())

    def test_a_mixed_picture_is_insufficient_in_both_directions(self):
        """
        Where "individually not a death sentence" actually bites: metrics split
        down the middle. Neither direction reaches unanimity, so nothing is
        shown either way. The sweep still stops -- an unadjudicated gate is not
        a pass -- but CLAIM-C is not recorded as falsified.
        """
        prompts = _prompts(8)
        ref = _heterogeneous(8, seed=13)
        can = ref.copy()
        can[:, 3:] = -ref[:, 3:]                # three agree, three invert
        r = self._p(ref, can, prompts)
        assert r["verdict"] == "INSUFFICIENT"
        assert r["falsified"] is False
        assert r["hard_stop"] is True           # the sweep still does not proceed

    def test_a_refusing_subset_refuses_the_whole_gate(self):
        """
        The unanimity rule is a MAX, and a max over a set with an undefined
        member is undefined. Reporting the rest would silently drop whichever
        subset was hardest to satisfy -- the one the rule exists to catch.

        The subset is emptied here by killing five metrics' SIGNS (an exactly
        zero contrast is dropped as sign-undefined) rather than by making five
        metrics agree across prompts, and the difference is not cosmetic. Five
        columns identical across prompts forces `sign_homogeneity` to at least
        5/6 = 0.833, which is inside the range where the homogeneity
        calibration refuses first -- so that fixture would pin the wrong
        refusal. Dead columns are ignored by `sign_homogeneity` entirely, which
        leaves the surviving metric to set it and reaches this branch cleanly.
        """
        prompts = _prompts(8)
        # One live metric, five with an exactly-zero contrast in both
        # architectures. The full table still has signs (the live column), so
        # the gate gets as far as scoring subsets -- and the subset that drops
        # the live metric has no cell with a sign at all.
        signs = np.zeros((8, len(CLAIM_C_METRICS)))
        signs[:, 0] = _heterogeneous(8, seed=21)[:, 0]
        r = self._p(signs, signs.copy(), prompts)
        assert r["p_value"] is None
        assert "metric subsets cannot carry a p-value" in r["reason"]
        assert "drop:mass_near_1" in r["reason"]
        assert r["verdict"] == "INSUFFICIENT"
        assert r["falsified"] is False


class TestInformativeRows:
    """
    The floor is set by the rows that can move the statistic, not by the rows
    that were run.

    A prompt's label flip swaps concordant and discordant cells in its row, so
    the row contributes `conc` unflipped and `valid - conc` flipped and its
    swing is `|valid - 2 conc|`. A row with swing 0 contributes the SAME number
    to the observed sum and to all 2^n null patterns: it is enumerated and never
    counted. With k rows that do move, the smallest expressible p is
    `(2^(n-k) + 1) / (2^n + 1)`.

    Two ways a row lands there. Every cell dropped, which is what a real run
    produces and what the curve's cell-drop dimension is about. And -- on a
    perfectly complete table -- an EVEN number of usable cells splitting exactly
    half and half: with the full six metrics, three concordant and three not,
    which happens to 20/64 of rows under H0. That second one was live in this
    gate from the day it was written and nothing looked for it.

    It is `P-ST1`'s informative-pair floor (POPPER_PLAN.md 6k) arriving in
    CLAIM-C from the other direction, down to the same k >= 5.
    """

    def _p(self, ref_signs, can_signs, prompts, **kw):
        return p_value_claim_c(*_four_arms(ref_signs, can_signs, prompts),
                               candidate_step0=_step0(prompts), **kw)

    # -- the arithmetic ----------------------------------------------------

    def test_swing_is_zero_exactly_on_an_even_split(self):
        assert list(row_swing([6, 6, 6, 0], [3, 6, 2, 0])) == [0, 6, 2, 0]
        assert n_informative_rows([6, 6, 6, 0], [3, 6, 2, 0]) == 2

    def test_an_odd_subset_width_can_never_have_a_zero_swing(self):
        """
        Why the full six-metric set is the binding subset on a complete table:
        five metrics is odd, and `|5 - 2c|` is odd for every c.
        """
        assert all(row_swing([5] * 6, list(range(6))) % 2 == 1)

    def test_the_floor_generalises_the_one_already_refused_on(self):
        """All rows informative reproduces `2 / (2^n + 1)` exactly, so this is
        one rule sharpened rather than a second rule added."""
        for n in (6, 8, 12):
            assert attainable_p(n, n, 2 ** n, True) == pytest.approx(
                2.0 / (2 ** n + 1.0))

    def test_the_floor_is_the_p_the_gate_actually_returns(self):
        """
        Not a bound: the formula is checked against the gate's own enumeration
        on the best table the informative rows admit.
        """
        from core.nulls import p_from_null
        from p1_mstate_tracking.replication_gate import _null_counts
        n, m = 8, len(CLAIM_C_METRICS)
        for k in range(1, n + 1):
            valid = np.full(n, m)
            conc = np.full(n, m // 2)          # uninformative
            conc[:k] = m                       # k rows perfectly concordant
            null, _, _ = _null_counts(valid, conc, n_perm=0, seed=0)
            got = p_from_null(float(conc.sum()), null,
                              alternative=CLAIM_C_ALTERNATIVE)["p_value"]
            assert got == pytest.approx(attainable_p(n, k, 2 ** n, True))

    def test_a_sampled_null_keeps_its_own_floor(self):
        """
        The tightening is exact where it applies and silent where it does not:
        under sampling every draw tying the maximum has positive probability, so
        the floor really is `1 / (n_perm + 1)` however few rows inform.
        """
        assert attainable_p(20, 1, 5000, False) == pytest.approx(1.0 / 5001.0)

    # -- the refusal -------------------------------------------------------

    def test_refuses_a_table_that_is_perfect_on_too_few_prompts(self):
        """
        Six prompts, four of them perfectly concordant and two splitting 3-3.
        Before this refusal the gate returned p = 0.0769 -- which is exactly
        this table's floor -- and called it 'not significant'.
        """
        prompts = _prompts(6)
        ref = _heterogeneous(6, seed=21)
        can = ref.copy()
        can[4] = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]) * ref[4]
        can[5] = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]) * ref[5]
        r = self._p(ref, can, prompts)
        assert r["p_value"] is None
        assert "can move the statistic" in r["reason"]
        assert r["verdict"] == "INSUFFICIENT"
        assert r["falsified"] is False
        info = r["informative_rows"]
        assert info["per_subset"]["all"]["n_informative_rows"] == 4
        assert info["attainable_p_given_informative_rows"] == pytest.approx(
            (2 ** 2 + 1) / (2 ** 6 + 1))

    def test_a_perfect_table_is_untouched(self):
        """Every row swings by the full width, so the floor is the design's."""
        prompts = _prompts(8)
        s = _heterogeneous(8)
        r = self._p(s, s, prompts)
        assert r["p_value"] == pytest.approx(2.0 / 257.0)
        info = r["informative_rows"]
        assert info["attainable_p_given_informative_rows"] == pytest.approx(
            info["design_attainable_p"])
        assert all(v["n_informative_rows"] == 8
                   for v in info["per_subset"].values())

    def test_it_is_reported_even_when_it_does_not_refuse(self):
        prompts = _prompts(8)
        s = _heterogeneous(8)
        r = self._p(s, s, prompts)
        assert set(r["informative_rows"]["per_subset"]) == set(r["subsets"])
        assert r["informative_rows"]["binding_subset"] in r["subsets"]

    # -- the tightness argument, which is what makes the refusal safe ------

    def test_whenever_it_fires_neither_tail_could_have_reached_alpha(self):
        """
        THE ASSERTION THIS CLASS EXISTS FOR, and the dual of POPPER_PLAN.md
        6j's result about the homogeneity refusal.

        A refusal that depends on the data is only safe if it can remove no
        verdict. The null is symmetric under a global flip, so both tails share
        the floor -- but 'so' is an argument, and this measures it instead:
        every table the refusal fires on is re-scored subset by subset and BOTH
        intersection-union maxima are required to sit above alpha. If the
        refusal ever swallowed a table that could have said TRANSFERS, or one
        that could have said FAILS-TO-TRANSFER, this fails.
        """
        from p1_mstate_tracking.replication_gate import _metric_subsets, _subset_result
        rng = np.random.default_rng(20260825)
        fired = 0
        for n in (6, 8):
            prompts = _prompts(n)
            for _ in range(80):
                ref = rng.choice([-1.0, 1.0], size=(n, len(CLAIM_C_METRICS)))
                can = rng.choice([-1.0, 1.0], size=(n, len(CLAIM_C_METRICS)))
                r = self._p(ref, can, prompts)
                if r["p_value"] is not None or "can move the statistic" not in str(
                        r.get("reason", "")):
                    continue
                fired += 1
                alpha = r["alpha"]
                concordant = np.asarray(r["concordant"], dtype=bool)
                usable = np.asarray(r["usable"], dtype=bool)
                sign_can = np.sign(np.asarray(r["contrast_candidate"], dtype=float))
                pg, pl = [], []
                for _name, cols in _metric_subsets():
                    sub = _subset_result(concordant, usable, sign_can, cols,
                                         n_perm=5000, seed=0)
                    assert sub["p_value"] is not None
                    pg.append(sub["p_value"])
                    pl.append(sub["p_reciprocal"])
                # Uncorrected, which is the favourable direction: the
                # correction can only raise these.
                assert max(pg) > alpha, "refused a table that could have TRANSFERRED"
                assert max(pl) > alpha, "refused a table that could have FAILED"
        assert fired >= 20, f"only {fired} tables triggered the refusal"

    def test_it_does_not_fire_once_five_rows_can_move(self):
        """
        Five informative rows is the first count that clears alpha = 0.05, at
        every prompt count -- so the refusal must let a five-row table through
        rather than costing it its verdict.
        """
        prompts = _prompts(8)
        ref = _heterogeneous(8, seed=7)
        can = ref.copy()
        flip = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        for i in (5, 6, 7):                    # three rows made uninformative
            can[i] = flip * ref[i]
        r = self._p(ref, can, prompts)
        assert r["informative_rows"]["per_subset"]["all"]["n_informative_rows"] == 5
        assert "can move the statistic" not in str(r.get("reason", ""))


class TestRowIndependence:
    """
    The measured cost of the design's one real limitation, and the exactly
    degenerate case it becomes.
    """

    def _p(self, ref, can, prompts):
        return p_value_claim_c(*_four_arms(ref, can, prompts),
                               candidate_step0=_step0(prompts))

    def test_refuses_when_every_prompt_tells_the_same_story(self):
        """
        Identical sign rows mean the prompts contribute ONE observation, so
        enumerating 2^n patterns over them is the wrong null rather than a
        conservative one -- measured at a ~0.34 rejection rate against a
        nominal 0.05. This is a degeneracy, not a threshold: nothing is being
        compared to a tolerance, the rows are either all equal or they are not.
        """
        prompts = _prompts(8)
        s = np.ones((8, len(CLAIM_C_METRICS)))
        r = self._p(s, s, prompts)
        assert r["p_value"] is None
        assert "SAME candidate sign pattern" in r["reason"]
        assert r["verdict"] == "INSUFFICIENT"
        assert r["falsified"] is False

    def test_the_attainable_floor_is_reported_first_when_both_apply(self):
        """
        A four-prompt all-identical design fails both refusals. The size one is
        reported, because it is the one no amount of better prompts fixes.
        """
        prompts = _prompts(4)
        s = np.ones((4, len(CLAIM_C_METRICS)))
        assert "cannot reject on a PERFECT result" in self._p(s, s, prompts)["reason"]

    def test_a_dropped_cell_does_not_hide_the_degeneracy(self):
        """
        Regression: masking the sign table by multiplying with the usable mask
        leaves NaN in place (NaN * False is NaN), and a row carrying NaN
        compares unequal to an identical row -- which would make one
        sign-undefined cell silently disable the refusal.
        """
        prompts = _prompts(8)
        s = np.ones((8, len(CLAIM_C_METRICS)))
        can = s.copy()
        can[2, 1] = 0.0                       # one cell loses its sign
        r = claim_c_concordance(*_four_arms(s, can, prompts),
                                candidate_step0=_step0(prompts))
        assert r["n_cells_dropped"] == 1
        assert r["sign_rows_identical"] is True

    def test_homogeneity_is_reported_between_the_two_measured_rates(self):
        prompts = _prompts(8)
        het = _heterogeneous(8)
        r = self._p(het, het, prompts)
        assert r["sign_rows_identical"] is False
        assert 0.5 <= r["sign_homogeneity"] <= 1.0
        hom = claim_c_concordance(
            *_four_arms(np.ones((8, len(CLAIM_C_METRICS))),
                        np.ones((8, len(CLAIM_C_METRICS))), prompts),
            candidate_step0=_step0(prompts))
        assert hom["sign_homogeneity"] == pytest.approx(1.0)
        assert hom["sign_rows_identical"] is True


class TestTwoBaselinePolicy:

    def test_step0_arm_cannot_be_dropped_by_omission(self):
        """
        PREDICTIONS.md attaches the two-baseline policy to THIS claim. An arm
        that disappears when a caller forgets it is not a policy -- the same
        refusal `centroids.load_centroids` makes rather than silently falling
        back to the primary arm.
        """
        prompts = _prompts(8)
        s = _heterogeneous(8)
        with pytest.raises(TypeError):
            claim_c_concordance(*_four_arms(s, s, prompts))       # no default
        with pytest.raises(ValueError, match="two-baseline policy"):
            claim_c_concordance(*_four_arms(s, s, prompts), candidate_step0=None)

    def test_absence_is_allowed_only_in_writing(self):
        prompts = _prompts(8)
        s = _heterogeneous(8)
        r = claim_c_concordance(*_four_arms(s, s, prompts), candidate_step0=None,
                                step0_absent_reason="no step-0 run for this family")
        assert r["step0_absent_reason"] == "no step-0 run for this family"
        assert r["step0_sensitivity"]["available"] is False

    def test_step0_is_reported_and_never_adjudicated(self):
        """
        Step 0 is CLAIM-A's object. It is computed, reported, and kept out of
        the p-value: one dataset settling two registry entries is
        EVALUABILITY.md's third recurring pattern.
        """
        prompts = _prompts(8)
        s = _heterogeneous(8)
        rt, rr, ct, cr = _four_arms(s, s, prompts)
        with_s0 = p_value_claim_c(rt, rr, ct, cr, candidate_step0=_step0(prompts))
        without = p_value_claim_c(rt, rr, ct, cr, candidate_step0=None,
                                  step0_absent_reason="withheld for this check")
        assert with_s0["p_value"] == pytest.approx(without["p_value"])
        assert with_s0["step0_sensitivity"]["available"] is True
        assert with_s0["step0_sensitivity"]["n_cells"] > 0

    def test_disagreement_between_the_two_baselines_is_flagged(self):
        """
        The two-baseline policy exists because these are different objects. If
        they point opposite ways about transfer, the record has to say so --
        that is a finding about which random baseline the claim was ever about.
        """
        prompts = _prompts(8)
        s = _heterogeneous(8)
        rt, rr, ct, cr = _four_arms(s, s, prompts)
        # step-0 sits twice as far out as the trained arm in every cell, so
        # every trained-minus-step0 contrast inverts relative to the reference.
        s0 = _arm(2.0 * s, prompts, CAN_LAYERS)
        r = claim_c_concordance(rt, rr, ct, cr, candidate_step0=s0)
        sens = r["step0_sensitivity"]
        assert sens["concordance_fraction"] == pytest.approx(0.0)
        assert sens["disagrees_with_primary"] is True
        assert "OPPOSITE" in sens["note"]


class TestGateVerdict:

    def test_only_demonstrated_inversion_counts_as_falsified(self):
        assert gate_verdict(0.01, 0.99, alpha=0.05)["verdict"] == "TRANSFERS"
        assert gate_verdict(0.99, 0.01, alpha=0.05)["falsified"] is True
        assert gate_verdict(0.40, 0.60, alpha=0.05)["falsified"] is False

    def test_every_non_transfer_outcome_stops_the_sweep(self):
        for g, l in ((0.99, 0.01), (0.40, 0.60), (None, None)):
            assert gate_verdict(g, l, alpha=0.05)["hard_stop"] is True
        assert gate_verdict(0.01, 0.99, alpha=0.05)["hard_stop"] is False


# ---------------------------------------------------------------------------
# Ledger wiring
# ---------------------------------------------------------------------------

class TestAdjudicationWiring:

    def _args(self):
        prompts = _prompts(8)
        s = _heterogeneous(8)
        return _four_arms(s, s, prompts), _step0(prompts)

    def test_opt_in_writes_nothing_by_default(self, tmp_path):
        (rt, rr, ct, cr), s0 = self._args()
        r = adjudicate_claim_c(rt, rr, ct, cr, candidate_step0=s0,
                               adjudications_dir=tmp_path)
        assert r["adjudication"] is None
        assert not list(tmp_path.glob("*.json"))

    def test_emits_into_the_ledger_when_asked(self, tmp_path):
        (rt, rr, ct, cr), s0 = self._args()
        r = adjudicate_claim_c(rt, rr, ct, cr, candidate_step0=s0,
                               adjudicate=True, adjudications_dir=tmp_path)
        adj = r["adjudication"]
        assert adj is not None
        assert adj["prediction_id"] == "CLAIM-C"
        assert adj["claim"] == "H-TRANSFER"
        assert f"one-sided '{CLAIM_C_ALTERNATIVE}'" in adj["test_name"]
        assert "exhaustive" in adj["test_name"]
        assert (tmp_path / "CLAIM-C.json").exists()

    def test_the_record_carries_the_stop_decision_and_its_caveat(self, tmp_path):
        """
        A reader of the ledger should not have to re-derive which of the three
        verdicts fired, nor discover on their own that prompts on one model are
        not independent runs.
        """
        (rt, rr, ct, cr), s0 = self._args()
        r = adjudicate_claim_c(rt, rr, ct, cr, candidate_step0=s0,
                               adjudicate=True, adjudications_dir=tmp_path)
        notes = r["adjudication"]["notes"]
        assert "verdict=TRANSFERS" in notes
        assert "hard_stop=False" in notes
        assert "p_reciprocal" in notes and "NOT" in notes
        assert "not independent runs" in notes
        assert "step0_arm=reported" in notes

    def test_a_refusal_writes_no_record(self, tmp_path):
        """
        Four prompts cannot express a significant p. Nothing may reach the
        ledger -- least of all on the prediction carrying the hard stop.
        """
        prompts = _prompts(4)
        s = _heterogeneous(4)
        r = adjudicate_claim_c(*_four_arms(s, s, prompts),
                               candidate_step0=_step0(prompts),
                               adjudicate=True, adjudications_dir=tmp_path)
        assert r["p_value"] is None
        assert r["adjudication"] is None
        assert not list(tmp_path.glob("*.json"))

    def test_real_ledger_untouched_by_tests(self):
        from core.adjudication import load_adjudications
        import tempfile
        before = {rec["prediction_id"] for rec in load_adjudications()}
        (rt, rr, ct, cr), s0 = self._args()
        with tempfile.TemporaryDirectory() as td:
            adjudicate_claim_c(rt, rr, ct, cr, candidate_step0=s0,
                               adjudicate=True, adjudications_dir=td)
        assert {rec["prediction_id"] for rec in load_adjudications()} == before


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------

class TestProfileLoader:

    def _write(self, d, geo=None, clu=None, snk=None):
        import json
        if geo is not None:
            (d / "geometry.json").write_text(json.dumps({"layers": geo}))
        if clu is not None:
            (d / "clustering.json").write_text(json.dumps({"layers": clu}))
        if snk is not None:
            (d / "sinkhorn.json").write_text(json.dumps({"layers": snk}))

    def test_reads_all_six(self, tmp_path):
        self._write(
            tmp_path,
            geo=[{"ip_mass_near_1": 0.1 * i, "effective_rank_normed": 10.0 - i,
                  "cka_prev": 0.99} for i in range(4)],
            clu=[{"clustering": {"hdbscan": {"noise_fraction": 0.25,
                                             "n_clusters": 3}}} for _ in range(4)],
            snk=[{"fiedler_mean": 0.4} for _ in range(4)],
        )
        prof = profiles_from_run_dir(tmp_path)
        assert set(prof) == set(CLAIM_C_METRICS)
        assert prof["cluster_membership"] == [0.75] * 4
        assert prof["effective_rank"] == [10.0, 9.0, 8.0, 7.0]

    def test_effective_rank_comes_from_the_normed_field(self, tmp_path):
        """
        status-1.md defect D1: the raw field mixes directional collapse with
        residual-stream norm growth. Baking it into the gate that carries the
        hard stop would be knowingly wrong, so the raw field must be ignored
        even when it is the only one present.
        """
        self._write(tmp_path, geo=[{"effective_rank": 2.0} for _ in range(4)])
        assert "effective_rank" not in profiles_from_run_dir(tmp_path)

    def test_missing_artifacts_yield_missing_keys_not_zeros(self, tmp_path):
        """
        A zero-filled series would score a contrast against absent data. The
        key must simply be absent, so `claim_c_concordance` drops the prompt.
        """
        prof = profiles_from_run_dir(tmp_path)
        assert prof == {}

    def test_null_fields_become_nan_not_dropped_layers(self, tmp_path):
        self._write(tmp_path, geo=[{"cka_prev": None}, {"cka_prev": 0.9},
                                   {"cka_prev": 0.8}])
        prof = profiles_from_run_dir(tmp_path)
        assert len(prof["cka_prev"]) == 3
        assert np.isnan(prof["cka_prev"][0])


# ---------------------------------------------------------------------------
# The registry records the construction
# ---------------------------------------------------------------------------

class TestRegistryIsInStep:

    def _entry(self):
        from core.adjudication import registry_entry
        return registry_entry("CLAIM-C")

    def test_claim_c_is_now_an_e_value_entry(self):
        e = self._entry()
        assert e.evaluable == "e-value"
        assert e.claim == "H-TRANSFER"
        assert e.status == "active"

    def test_null_construction_names_every_fixed_choice(self):
        """
        The registry's `null_construction` is where a later reader finds out
        what was fixed in advance. It is frozen after the first adjudication,
        so it has to be right BEFORE one -- which is now.
        """
        nc = self._entry().null_construction
        for metric in CLAIM_C_METRICS:
            assert metric in nc, f"{metric} not named in null_construction"
        assert CLAIM_C_ALTERNATIVE in nc
        assert CLAIM_C_RECIPROCAL_ALTERNATIVE in nc
        assert str(DEPTH_GRID_POINTS) in nc
        assert "prompt" in nc
        assert "effective_rank_normed" in nc
