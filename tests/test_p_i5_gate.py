"""
tests/test_p_i5_gate.py — p7_motifs/p_i5_gate.py.

Pure numpy/scipy/plain Python throughout: the exact joint sign-flip null
needs no model, and count_matched_pairs_by_prompt is tested against a fake
tokenizer stub rather than a real transformers tokenizer, so nothing here
needs torch or transformers importable. See p7_motifs/p_i5_gate.py's own
docstring for the 34,191-pair repeated_tokens figure this module's
real-tokenizer measurement cross-validates against — that measurement is
manual (recorded in the module docstring), not re-run here, matching how
gate never runs the smoke/heavy tiers by default (pytest.ini).
"""
from __future__ import annotations

import itertools

import numpy as np
import numpy.testing as npt
import pytest

from p7_motifs.p_i5_gate import (
    enumerate_sign_patterns,
    naive_and_corner_pvalue,
    joint_rank_pvalue,
    one_dimensional_pvalue,
    attainable_floor,
    calibrate_naive_and_corner,
    calibrate_joint_rank,
    partial_pass_risk_demo,
    count_matched_pairs_by_prompt,
    informative_prompt_count,
    DEGENERATE_PROMPT,
    MAX_EXACT_UNITS,
)

pytestmark = pytest.mark.pure


# ---------------------------------------------------------------------------
# enumerate_sign_patterns
# ---------------------------------------------------------------------------

class TestEnumerateSignPatterns:

    def test_count_and_values(self):
        for n in range(1, 8):
            patterns = enumerate_sign_patterns(n)
            assert patterns.shape == (2 ** n, n)
            assert set(np.unique(patterns).tolist()) <= {-1, 1}

    def test_all_patterns_distinct(self):
        patterns = enumerate_sign_patterns(5)
        rows = {tuple(r) for r in patterns}
        assert len(rows) == 2 ** 5

    def test_zero_units(self):
        patterns = enumerate_sign_patterns(0)
        assert patterns.shape == (1, 0)

    def test_over_cap_raises(self):
        with pytest.raises(ValueError):
            enumerate_sign_patterns(MAX_EXACT_UNITS + 1)


# ---------------------------------------------------------------------------
# naive_and_corner_pvalue — kept for the record; retired (over-rejects)
# ---------------------------------------------------------------------------

class TestNaiveAndCornerPvalue:

    def test_fully_informative_same_signed_hits_the_floor(self):
        """Even the retired construction gets the best case right -- the
        floor formula is unaffected by the section-2 bug (module
        docstring: both statistics agree in the fully-informative case)."""
        for n in (1, 2, 3, 5, 8):
            rng = np.random.default_rng(n)
            dg = rng.uniform(0.1, 5.0, size=n)
            dl = rng.uniform(0.1, 5.0, size=n)
            result = naive_and_corner_pvalue(dg, dl, alternative="greater")
            npt.assert_allclose(result["p_value"], attainable_floor(n, "greater"))

    def test_reduces_to_one_dimensional_when_one_axis_is_zero(self):
        """This is exactly why the bug was easy to miss: the marginal
        reduction (one axis identically zero -- a trivially-always-true
        AND condition on that axis) IS valid, and it is the case most
        likely to be checked first. t_l_obs is read off the SAME matmul as
        the null array (not recomputed via dl.sum()), matching the
        source's own self-consistency fix -- a separately-computed sum can
        differ by 1 ULP and silently drop the observed pattern from its
        own extreme set."""
        n = 8
        rng = np.random.default_rng(1)
        dl = rng.standard_normal(n)

        joint = naive_and_corner_pvalue(np.zeros(n), dl, alternative="greater")

        patterns = enumerate_sign_patterns(n).astype(np.float64)
        t_l_null = patterns @ dl
        obs_row = int(np.flatnonzero(np.all(patterns == 1.0, axis=1))[0])
        t_l_obs = float(t_l_null[obs_row])
        expected_p = float((t_l_null >= t_l_obs).mean())
        npt.assert_allclose(joint["p_value"], expected_p)

    def test_overrejects_under_true_joint_h0(self):
        """The measured finding module docstring section 2 reports: at
        n=8 the naive AND-corner rejects at roughly 4x the nominal alpha
        under a true joint null. Loose bound (not the exact 0.207) so this
        doesn't pin a specific RNG stream's noise."""
        result = calibrate_naive_and_corner(n_units=8, n_trials=2000, rng=np.random.default_rng(11))
        assert result["rejection_rate"]["0.05"] > 0.12  # nominal is 0.05

    def test_exact_matches_brute_force_reference(self):
        rng = np.random.default_rng(42)
        n = 6
        dg = rng.standard_normal(n)
        dl = rng.standard_normal(n)
        t_g_obs, t_l_obs = dg.sum(), dl.sum()

        count = 0
        for signs in itertools.product([1, -1], repeat=n):
            s = np.array(signs, dtype=np.float64)
            if (s @ dg) >= t_g_obs and (s @ dl) >= t_l_obs:
                count += 1
        expected_p = count / (2 ** n)

        result = naive_and_corner_pvalue(dg, dl, alternative="greater")
        npt.assert_allclose(result["p_value"], expected_p)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            naive_and_corner_pvalue(np.zeros(3), np.zeros(4))

    def test_over_cap_without_mc_draws_raises(self):
        n = MAX_EXACT_UNITS + 1
        with pytest.raises(ValueError):
            naive_and_corner_pvalue(np.ones(n), np.ones(n))


# ---------------------------------------------------------------------------
# joint_rank_pvalue — the corrected statistic
# ---------------------------------------------------------------------------

class TestJointRankPvalueExact:

    def test_fully_informative_same_signed_hits_the_floor(self):
        """Fully-informative, same-signed on both axes -> the observed
        pattern uniquely holds the top rank on BOTH axes, so the unique
        top combined (min) rank -- p equals attainable_floor(n) exactly."""
        for n in (1, 2, 3, 5, 8):
            rng = np.random.default_rng(n)
            dg = rng.uniform(0.1, 5.0, size=n)
            dl = rng.uniform(0.1, 5.0, size=n)
            result = joint_rank_pvalue(dg, dl, alternative="greater")
            npt.assert_allclose(result["p_value"], attainable_floor(n, "greater"))
            assert result["exact"] is True
            assert result["n_units"] == n

    def test_two_sided_floor_at_fully_informative(self):
        for n in (2, 4, 6):
            rng = np.random.default_rng(100 + n)
            dg = rng.uniform(0.1, 5.0, size=n)
            dl = rng.uniform(0.1, 5.0, size=n)
            result = joint_rank_pvalue(dg, dl, alternative="two-sided")
            npt.assert_allclose(result["p_value"], attainable_floor(n, "two-sided"))

    def test_all_zero_deltas_give_p_one(self):
        n = 6
        result = joint_rank_pvalue(np.zeros(n), np.zeros(n), alternative="greater")
        npt.assert_allclose(result["p_value"], 1.0)

    def test_calibrated_under_true_joint_h0(self):
        """The corrected statistic's whole point: this should land close
        to nominal, unlike naive_and_corner_pvalue's ~4x inflation."""
        result = calibrate_joint_rank(n_units=8, n_trials=3000, rng=np.random.default_rng(11))
        npt.assert_allclose(result["rejection_rate"]["0.05"], 0.05, atol=0.02)
        npt.assert_allclose(result["rejection_rate"]["0.1"], 0.10, atol=0.03)

    def test_an_identically_zero_axis_degenerates_rather_than_reduces(self):
        """Documented caveat (module docstring section 2): unlike
        naive_and_corner_pvalue, zeroing one axis does NOT recover a
        one-dimensional reading here -- every pattern ties for the
        minimum rank on the zeroed axis, so the min-rank combination
        collapses to that tie for every pattern and p_value is always 1.0,
        regardless of how extreme the other axis is. This is why
        one_dimensional_pvalue exists as a separate function."""
        n = 8
        rng = np.random.default_rng(1)
        dl = rng.standard_normal(n) + 5.0  # very extreme on the logit axis
        result = joint_rank_pvalue(np.zeros(n), dl, alternative="greater")
        npt.assert_allclose(result["p_value"], 1.0)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            joint_rank_pvalue(np.zeros(3), np.zeros(4))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            joint_rank_pvalue(np.zeros(0), np.zeros(0))

    def test_unknown_alternative_raises(self):
        with pytest.raises(ValueError):
            joint_rank_pvalue(np.ones(3), np.ones(3), alternative="less")

    def test_over_cap_without_mc_draws_raises(self):
        n = MAX_EXACT_UNITS + 1
        with pytest.raises(ValueError):
            joint_rank_pvalue(np.ones(n), np.ones(n))


class TestOneDimensionalPvalue:

    def test_matches_direct_enumeration(self):
        n = 8
        rng = np.random.default_rng(1)
        dl = rng.standard_normal(n)

        patterns = enumerate_sign_patterns(n).astype(np.float64)
        t_null = patterns @ dl
        obs_row = int(np.flatnonzero(np.all(patterns == 1.0, axis=1))[0])
        expected = float((t_null >= t_null[obs_row]).mean())

        npt.assert_allclose(one_dimensional_pvalue(dl, alternative="greater"), expected)

    def test_fully_informative_hits_the_one_axis_floor(self):
        n = 6
        d = np.array([1.0, 2.0, 3.0, 0.5, 4.0, 1.5])
        npt.assert_allclose(one_dimensional_pvalue(d, alternative="greater"), 1.0 / 2 ** n)

    def test_calibrated_under_h0(self):
        rng = np.random.default_rng(5)
        n = 8
        ps = np.array([
            one_dimensional_pvalue(rng.standard_normal(n), alternative="greater")
            for _ in range(2000)
        ])
        npt.assert_allclose((ps <= 0.05).mean(), 0.05, atol=0.02)


class TestJointRankPvalueMonteCarlo:

    def test_mc_agrees_with_exact_within_tolerance(self):
        n = 10
        rng = np.random.default_rng(7)
        dg = rng.standard_normal(n)
        dl = rng.standard_normal(n)

        exact = joint_rank_pvalue(dg, dl, alternative="greater")["p_value"]
        mc = joint_rank_pvalue(
            dg, dl, alternative="greater", mc_draws=50_000,
            rng=np.random.default_rng(1),
        )["p_value"]
        assert abs(exact - mc) < 0.02

    def test_mc_marks_inexact(self):
        result = joint_rank_pvalue(
            np.ones(4), np.ones(4), mc_draws=500, rng=np.random.default_rng(0),
        )
        assert result["exact"] is False


# ---------------------------------------------------------------------------
# attainable_floor
# ---------------------------------------------------------------------------

class TestAttainableFloor:

    def test_matches_direct_enumeration_one_sided(self):
        """Closed form vs brute-force best-case construction, n = 1..15 --
        exhaustive for every n in that range, not a spot check. Checked
        against joint_rank_pvalue (the statistic actually used)."""
        for n in range(1, 16):
            dg = np.ones(n)
            dl = np.ones(n)
            measured = joint_rank_pvalue(dg, dl, alternative="greater")["p_value"]
            npt.assert_allclose(measured, attainable_floor(n, "greater"))

    def test_matches_direct_enumeration_two_sided(self):
        for n in range(1, 14):
            dg = np.ones(n)
            dl = np.ones(n)
            measured = joint_rank_pvalue(dg, dl, alternative="two-sided")["p_value"]
            npt.assert_allclose(measured, attainable_floor(n, "two-sided"))

    def test_monotone_decreasing_in_n(self):
        floors = [attainable_floor(n) for n in range(1, 10)]
        assert all(floors[i] > floors[i + 1] for i in range(len(floors) - 1))

    def test_rejects_n_below_one(self):
        with pytest.raises(ValueError):
            attainable_floor(0)

    def test_rejects_unknown_alternative(self):
        with pytest.raises(ValueError):
            attainable_floor(4, alternative="less")


# ---------------------------------------------------------------------------
# partial_pass_risk_demo
# ---------------------------------------------------------------------------

class TestPartialPassRiskDemo:

    def test_joint_rejects_far_less_than_logit_only_on_the_falsifier_config(self):
        """The falsifier's own configuration: a real logit effect, pure
        noise on the geometric axis. A reader looking at the logit axis
        alone should see it reject often; joint_rank_pvalue, correctly,
        should reject much less -- this is the gap EVALUABILITY.md's
        'partial pass' warning is about, made quantitative."""
        result = partial_pass_risk_demo(
            n_units=8, n_trials=600, logit_effect=1.0, alpha=0.05,
            rng=np.random.default_rng(3),
        )
        assert result["logit_only_reject_rate"] > 0.5
        assert result["joint_reject_rate"] < 0.5 * result["logit_only_reject_rate"]

    def test_reports_attainable_floor_for_the_n_used(self):
        result = partial_pass_risk_demo(n_units=6, n_trials=50, rng=np.random.default_rng(0))
        npt.assert_allclose(
            result["attainable_floor_one_sided"], attainable_floor(6, "greater")
        )


# ---------------------------------------------------------------------------
# count_matched_pairs_by_prompt / informative_prompt_count
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Minimal stand-in: splits on whitespace, same word -> same id (so
    repeats are visible to induction_candidates, unlike a bare word index).
    No transformers dependency -- keeps this test in the pure tier."""

    def __call__(self, text):
        vocab: dict = {}
        ids = []
        for word in text.split():
            ids.append(vocab.setdefault(word, len(vocab)))
        return {"input_ids": ids}


class TestCountMatchedPairsByPrompt:

    def test_counts_and_excludes_degenerate_prompt(self, monkeypatch):
        import core.config as config_module

        fake_prompts = {
            "no_repeats": "a b c d e",
            DEGENERATE_PROMPT: "x x x x x x x x",
        }
        monkeypatch.setattr(config_module, "PROMPTS", fake_prompts)

        counts = count_matched_pairs_by_prompt(_FakeTokenizer())
        assert set(counts) == set(fake_prompts)
        assert counts["no_repeats"]["n_tokens"] == 5
        # "a b c d e" -> distinct ids 0..4, strictly increasing -- no id
        # repeats consecutively in a way induction_candidates' condition
        # (ids[key-1] == ids[query-1]) can match, so 0 pairs.
        assert counts["no_repeats"]["n_pairs"] == 0
        assert counts[DEGENERATE_PROMPT]["n_pairs"] > 0

        n = informative_prompt_count(counts)
        assert n == 1  # only "no_repeats" -- DEGENERATE_PROMPT excluded

    def test_informative_prompt_count_respects_custom_exclusion(self):
        counts = {"a": {}, "b": {}, "c": {}}
        assert informative_prompt_count(counts, exclude=("b",)) == 2
        assert informative_prompt_count(counts, exclude=()) == 3
