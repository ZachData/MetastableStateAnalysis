"""
`core.evalues.average` — the arbitrary-dependence e-merger, and the reason it
had to exist beside `combine`.

Phase 10's free rows compute one statistic over every layer of every prompt of
every checkpoint of one sweep. Those units share a model, a text and a forward
pass, so `EProcess`'s conditional-calibration precondition does not hold and
the product is not an e-value over them. The mean is, by linearity alone.

These tests check three things in order: the arithmetic, the refusals, and the
guarantee itself under the joint distribution that breaks the product.
"""
import math

import pytest

from core.evalues import (
    DEFAULT_ALPHA,
    DEFAULT_KAPPA,
    EValueError,
    average,
    average_p,
    calibrate,
    combine,
    simulate_type_i_error_dependent,
)


# --- arithmetic ------------------------------------------------------------

def test_average_of_one_is_that_value():
    E, reject = average([7.0])
    assert E == pytest.approx(7.0)
    assert reject is False         # 1/alpha is 20; 7 is not evidence


def test_rejection_is_at_one_over_alpha():
    assert average([19.9])[1] is False
    assert average([20.0])[1] is True
    assert average([20.0], alpha=0.1)[1] is True
    assert average([9.9], alpha=0.1)[1] is False


def test_average_is_the_arithmetic_mean():
    E, _ = average([1.0, 2.0, 3.0, 4.0])
    assert E == pytest.approx(2.5)


def test_average_never_exceeds_its_largest_input():
    """The stated price of dependence-robustness. A merger that could exceed
    the max would be extracting evidence the dependent units do not carry."""
    es = [0.1, 0.4, 3.0, 55.0, 2.0]
    E, _ = average(es)
    assert E <= max(es)


def test_weights_are_normalised_not_required_to_sum_to_one():
    E, _ = average([1.0, 3.0], weights=[3.0, 1.0])
    assert E == pytest.approx(1.5)


def test_uniform_weights_match_the_default():
    es = [0.5, 2.0, 9.0]
    assert average(es)[0] == pytest.approx(average(es, weights=[2.0, 2.0, 2.0])[0])


def test_infinite_e_value_propagates_and_rejects():
    E, reject = average([1.0, math.inf, 1.0])
    assert math.isinf(E)
    assert reject is True


def test_infinity_with_zero_weight_does_not_propagate():
    E, _ = average([1.0, math.inf], weights=[1.0, 0.0])
    assert E == pytest.approx(1.0)


# --- average_p -------------------------------------------------------------

def test_average_p_calibrates_then_averages():
    ps = [0.01, 0.5, 0.9]
    expected = sum(calibrate(p) for p in ps) / 3.0
    assert average_p(ps)[0] == pytest.approx(expected)


def test_average_p_of_a_uniform_p_is_below_one():
    """p = 1 calibrates to kappa = 0.5; a set of them cannot fabricate
    evidence however many there are."""
    assert average_p([1.0] * 500)[0] == pytest.approx(DEFAULT_KAPPA)


def test_average_p_zero_gives_infinity():
    E, reject = average_p([0.0, 0.5])
    assert math.isinf(E)
    assert reject is True


# --- refusals --------------------------------------------------------------

def test_empty_is_refused():
    with pytest.raises(EValueError, match="empty"):
        average([])


def test_negative_e_value_is_refused():
    with pytest.raises(EValueError, match="non-negative"):
        average([1.0, -0.5])


def test_nan_e_value_is_refused():
    with pytest.raises(EValueError, match="NaN"):
        average([1.0, float("nan")])


def test_nan_weight_is_refused():
    with pytest.raises(EValueError, match="non-negative and finite"):
        average([1.0, 2.0], weights=[1.0, float("nan")])


def test_negative_weight_is_refused():
    with pytest.raises(EValueError, match="non-negative and finite"):
        average([1.0, 2.0], weights=[1.0, -1.0])


def test_mismatched_weight_length_is_refused():
    with pytest.raises(EValueError, match="length"):
        average([1.0, 2.0], weights=[1.0])


def test_zero_weights_are_refused():
    with pytest.raises(EValueError, match="positive finite"):
        average([1.0, 2.0], weights=[0.0, 0.0])


def test_bad_alpha_is_refused():
    with pytest.raises(EValueError, match="alpha"):
        average([1.0], alpha=0.0)


# --- the guarantee, measured ----------------------------------------------

def test_average_controls_type_i_error_under_maximal_dependence():
    """The claim the merger exists for. 25 perfectly dependent units."""
    rate = simulate_type_i_error_dependent(
        n_trials=40_000, n_experiments=25, merger="average", seed=0
    )
    assert rate <= DEFAULT_ALPHA + 0.005, rate


def test_the_product_does_not_control_it_on_the_same_draws():
    """Not a test of `combine` being wrong -- `combine` is correct under its
    own precondition. It is a test that the precondition is load-bearing, so
    that reaching for the product on sweep units is a real error and not a
    stylistic one."""
    rate = simulate_type_i_error_dependent(
        n_trials=40_000, n_experiments=25, merger="product", seed=0
    )
    # Closed form: the product of 25 copies of e = kappa*p^(kappa-1) reaches
    # 1/alpha exactly when p <= (alpha * kappa**25) ** (1 / (25 * (1 - kappa))),
    # which is 0.1967 at the defaults -- four times the nominal level.
    assert rate > 3 * DEFAULT_ALPHA, rate
    assert rate == pytest.approx(0.1967, abs=0.01), rate


def test_dependent_simulation_refuses_an_unknown_merger():
    with pytest.raises(EValueError, match="merger"):
        simulate_type_i_error_dependent(n_trials=10, merger="median")


def test_average_beats_product_on_a_shared_statistic():
    """A worked instance of the failure: one p-value of 0.19 -- not evidence
    against anything at alpha = 0.05 -- re-measured across 25 layers."""
    p = 0.19
    E_prod, reject_prod = combine([p] * 25)
    E_avg, reject_avg = average_p([p] * 25)
    assert reject_prod is True          # manufactured
    assert reject_avg is False
    assert E_avg == pytest.approx(calibrate(p))
