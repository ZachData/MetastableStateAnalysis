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

# Tier: numpy and scipy only -- no torch, transformers, sklearn or
# matplotlib -- so this runs in `scripts/check.sh pure`. Declared, not
# assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure



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


# --- could the design have rejected at all? -------------------------------

from core.evalues import max_attainable_average_E


def test_the_ceiling_is_the_calibrated_resolution_floor():
    E_max, _ = max_attainable_average_E(400)
    assert E_max == pytest.approx(calibrate(1.0 / 401.0))


def test_four_hundred_permutations_cannot_reject_at_the_defaults():
    """The flaw Phase 10's row A0 found the expensive way: a first full run
    reporting `reject: False` at every checkpoint, from a design that could not
    have rejected if every unit had come back maximally extreme."""
    E_max, can = max_attainable_average_E(400)
    assert E_max == pytest.approx(10.01, abs=0.01)
    assert can is False


def test_the_exact_threshold_is_1599_draws():
    """0.5 / sqrt(p) >= 20 needs p <= 1/1600, and a Monte-Carlo floor of
    1/(n+1) reaches that at n = 1599. Pinned exactly, because "about 1 600" is
    the kind of number that drifts into a design and makes it incapable."""
    assert max_attainable_average_E(1598)[1] is False
    assert max_attainable_average_E(1599)[0] == pytest.approx(20.0)
    assert max_attainable_average_E(1599)[1] is True


def test_the_runners_draw_count_can_reject():
    """A guard on the actual constants the rows run with, so lowering one
    silently re-creates the flaw."""
    from tools.run.p10_anchor import N_PERMUTATIONS as ANCHOR_N
    from tools.run.p10_attention_baseline import N_PERMUTATIONS as A0_N
    from tools.run.p10_partition_function import N_PERMUTATIONS as Z_N

    for n in (ANCHOR_N, A0_N, Z_N):
        assert max_attainable_average_E(n)[1] is True, n


def test_the_ceiling_binds_however_many_units_are_merged():
    """It is a property of the draw count alone. Ten thousand units all at the
    floor still cannot beat one."""
    E_max, _ = max_attainable_average_E(2000)
    merged, _ = average_p([1.0 / 2001.0] * 10_000)
    assert merged == pytest.approx(E_max)


def test_a_bad_draw_count_is_refused():
    with pytest.raises(EValueError, match="n_permutations"):
        max_attainable_average_E(0)
