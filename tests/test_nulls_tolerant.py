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


# ---------------------------------------------------------------------------
# `label_permutation_null_within` — the restricted null
# ---------------------------------------------------------------------------

from core.nulls import label_permutation_null, label_permutation_null_within
from core.parking import mean_nucleus_position


def test_the_restricted_null_never_moves_a_token_in_or_out_of_noise():
    """Its defining property. The clustered/noise split is what it holds
    fixed, and the ordinary null is what moves it."""
    lab = np.array([-1, 0, 0, -1, 1, 1, -1, 2, 2, -1])
    seen = []

    def spy(fixed, labels):
        seen.append((labels == -1).tolist())
        return 0.0

    label_permutation_null_within(np.arange(10.0), lab, spy, n_permutations=30,
                                  rng=np.random.default_rng(0))
    assert all(s == (lab == -1).tolist() for s in seen)


def test_the_restricted_null_still_preserves_every_cluster_size():
    lab = np.array([-1, 0, 0, 0, -1, 1, 1, -1, 2, 2])
    sizes = sorted(np.bincount(lab[lab >= 0]).tolist())
    seen = []

    def spy(fixed, labels):
        seen.append(sorted(np.bincount(labels[labels >= 0]).tolist()))
        return 0.0

    label_permutation_null_within(np.arange(10.0), lab, spy, n_permutations=30,
                                  rng=np.random.default_rng(0))
    assert all(s == sizes for s in seen)


def test_the_two_nulls_disagree_when_the_populations_differ_in_position():
    """The case the restricted null exists for: clustered tokens all late.
    The ordinary null says the nuclei are extraordinarily late, because it is
    free to move clusters into the early half. The restricted one, which is
    not, says nothing is going on -- and that is the honest answer about
    NUCLEATION."""
    n = 200
    lab = np.full(n, -1)
    # Clustered tokens all in the late half, but WHICH cluster each carries is
    # random among them -- so there is nothing to find about nucleation, and
    # the only structure is the population split itself.
    lab[100:] = np.random.default_rng(0).permutation(np.repeat(np.arange(20), 5))
    pos = np.arange(n, dtype=float)
    obs = mean_nucleus_position(pos, lab)

    wide = label_permutation_null(pos, lab, mean_nucleus_position,
                                  n_permutations=400, rng=np.random.default_rng(1))
    tight = label_permutation_null_within(pos, lab, mean_nucleus_position,
                                          n_permutations=400,
                                          rng=np.random.default_rng(1))
    assert p_from_null_tolerant(obs, wide, alternative="less")["p_value"] > 0.95
    assert 0.05 < p_from_null_tolerant(obs, tight, alternative="less")["p_value"] < 0.95


def test_the_restricted_null_still_detects_real_nucleation():
    """Power is kept: seed every cluster at the front of the clustered
    population and the restricted null rejects."""
    rng = np.random.default_rng(2)
    n = 300
    lab = np.full(n, -1)
    clustered = np.sort(rng.choice(n, size=120, replace=False))
    for cid in range(20):
        lab[clustered[cid]] = cid                      # the 20 earliest, seeded
    rest = clustered[20:]
    lab[rest] = rng.integers(0, 20, size=rest.size)
    pos = np.arange(n, dtype=float)
    draws = label_permutation_null_within(pos, lab, mean_nucleus_position,
                                          n_permutations=400, rng=rng)
    res = p_from_null_tolerant(mean_nucleus_position(pos, lab), draws,
                               alternative="less")
    assert res["p_value"] < 0.01, res


def test_fewer_than_two_clustered_tokens_gives_a_constant_null_not_a_floor():
    """Nothing can be permuted, so the honest p is 1. Filling the null with the
    observation is what produces that, via the tie guard."""
    lab = np.array([-1, -1, 0, -1])
    pos = np.arange(4.0)
    draws = label_permutation_null_within(pos, lab, lambda f, l: 0.25,
                                          n_permutations=50,
                                          rng=np.random.default_rng(0))
    assert np.all(draws == 0.25)
    assert p_from_null_tolerant(0.25, draws, alternative="less")["p_value"] == 1.0


def test_a_custom_noise_label_is_honoured():
    lab = np.array([99, 0, 0, 99, 1, 1])
    seen = []

    def spy(fixed, labels):
        seen.append((labels == 99).tolist())
        return 0.0

    label_permutation_null_within(np.arange(6.0), lab, spy, n_permutations=10,
                                  rng=np.random.default_rng(0), noise_label=99)
    assert all(s == [True, False, False, True, False, False] for s in seen)
