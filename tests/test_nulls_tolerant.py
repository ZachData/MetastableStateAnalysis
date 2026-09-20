"""
`core.nulls.p_from_null_tolerant` — the tie guard.

Written after row A0 returned the resolution floor on a layer where the
correction had explained everything and no permutation could move the
statistic. These tests pin the case that produced it and the conservative
direction of the fix.
"""
import numpy as np
import pytest

from core.nulls import p_from_null, p_from_null_tolerant


def test_a_fully_degenerate_null_returns_exactly_one():
    """The case that motivated this. Every draw equals the observation to
    within float noise, so nothing could have varied and the honest p is 1."""
    obs = 1.0
    draws = 1.0 + np.random.default_rng(0).normal(scale=1e-16, size=400)
    assert p_from_null_tolerant(obs, draws)["p_value"] == pytest.approx(1.0)


def test_the_untolerant_version_is_the_failure_this_fixes():
    """Not a test of p_from_null being wrong -- it is right for a continuous
    statistic. It records that the exact comparison is decided by rounding
    noise here, so the guard is load-bearing."""
    obs = 1.0
    draws = 1.0 + np.random.default_rng(0).normal(scale=1e-16, size=400)
    assert p_from_null(obs, draws)["p_value"] < 0.9


def test_the_degenerate_case_is_flagged_not_just_scored():
    """p = 1 because nothing could vary must be distinguishable afterwards
    from p = 1 because the effect was absent."""
    draws = 5.0 + np.random.default_rng(1).normal(scale=1e-17, size=200)
    deg = p_from_null_tolerant(5.0, draws)
    spread = p_from_null_tolerant(5.0, np.random.default_rng(1).normal(5.0, 1.0, 200))
    assert deg["degenerate_null"] is True and deg["n_ties"] == 200
    assert spread["degenerate_null"] is False
    assert spread["p_value"] == pytest.approx(0.5, abs=0.15)


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_ties_raise_the_p_value_in_every_direction(alternative):
    """The conservative direction: the failure mode must be missing a real
    effect, never inventing one."""
    draws = np.concatenate([np.full(100, 2.0), np.linspace(0.0, 1.0, 100)])
    strict = p_from_null(2.0, draws, alternative=alternative)["p_value"]
    tolerant = p_from_null_tolerant(2.0, draws, alternative=alternative)["p_value"]
    assert tolerant >= strict


def test_a_real_effect_still_rejects():
    """Power is not given away: an observation clear of the null by far more
    than the tolerance is unaffected."""
    draws = np.random.default_rng(2).normal(0.0, 1.0, 400)
    res = p_from_null_tolerant(6.0, draws, alternative="greater")
    assert res["p_value"] == pytest.approx(1.0 / 401.0)
    assert res["n_ties"] == 0


def test_the_tolerance_is_relative_for_large_statistics():
    """1e-9 relative at a statistic of 1e6 is 1e-3 absolute, so draws that far
    away are ties there and are not ties at a statistic of 1."""
    draws = np.array([1e6 + 5e-4] * 50)
    assert p_from_null_tolerant(1e6, draws)["n_ties"] == 50
    assert p_from_null_tolerant(1.0, np.array([1.0 + 5e-4] * 50))["n_ties"] == 0


def test_the_tolerance_is_absolute_near_zero():
    """scale floors at 1.0, so a statistic at 0 does not get a zero tolerance
    and fall back to the exact comparison it is meant to replace."""
    draws = np.full(50, 1e-12)
    assert p_from_null_tolerant(0.0, draws)["n_ties"] == 50


def test_rtol_zero_reproduces_the_strict_comparison():
    draws = np.array([1.0, 1.0, 2.0, 0.5])
    assert (p_from_null_tolerant(1.0, draws, rtol=0.0)["p_value"]
            == p_from_null(1.0, draws)["p_value"])


def test_negative_rtol_is_refused():
    with pytest.raises(ValueError, match="rtol"):
        p_from_null_tolerant(1.0, np.arange(10.0), rtol=-1e-9)


def test_it_keeps_every_field_p_from_null_returns():
    draws = np.random.default_rng(3).normal(0.0, 1.0, 100)
    strict = p_from_null(1.0, draws)
    tolerant = p_from_null_tolerant(1.0, draws)
    assert set(strict) <= set(tolerant)
    assert tolerant["resolution"] == strict["resolution"]


def test_non_finite_draws_are_still_dropped_and_counted():
    draws = np.array([1.0, np.nan, 3.0, np.inf, 2.0])
    res = p_from_null_tolerant(2.0, draws)
    assert res["n_null_finite"] == 3
    assert res["n_null_dropped"] == 2
