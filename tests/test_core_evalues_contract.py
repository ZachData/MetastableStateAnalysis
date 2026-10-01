"""
tests/test_core_evalues_contract.py — what mutation testing found untested in
`core/evalues.py` (`mutmut`, pyproject.toml [tool.mutmut]; history in
`LESSONS.md` lesson 6, the current count in `STATE.md`).

Most of the first run's survivors were not equivalent. The code could be
changed in these ways without any test noticing:

  * every refusal boundary: kappa and alpha at 0 or 1, p above 1 in
    `log_calibrate`, a None input (TypeError instead of EValueError), and
    `average`'s alpha check, which is the only guard on its early return for
    an infinite e-value;
  * every non-default argument: `EProcess.add` could drop `kappa`,
    `decision` and `next_p_needed` could ignore their `alpha`,
    `next_p_needed` could forget the evidence already accumulated, and
    `combine`, `average_p` and `max_attainable_average_E` could ignore
    `kappa`, `alpha` and `weights`;
  * `EProcess.to_record` / `from_record`: `from_record` could ignore the
    stored `alpha` and `kappa`, and `to_record` could write the threshold as
    `alpha` instead of `1/alpha`. Neither has a production caller: the ledger
    replay (`python -m core.adjudication --verify`) rebuilds through
    `EProcess(alpha=...)` + `add(kappa=...)`, and its non-default test is in
    `tests/test_core_adjudication.py`;
  * the Type-I rate: the simulation helper's test checked `rate <= alpha`
    only, so a helper that never rejects passed. The helper re-derives the
    calibrator in numpy, so its known answer pins the helper; the same
    answers are pinned below through `EProcess` and `average_p`, the code
    that scores.

Each test names what it kills. The survivors left are listed with a reason in
`tools/mutation_accepted.json`, and `tools/mutation_check.py` fails on any
survivor not listed there.
"""

from __future__ import annotations

import math

import pytest

from core.evalues import (
    DEFAULT_ALPHA,
    DEFAULT_KAPPA,
    EProcess,
    EValueError,
    average,
    average_p,
    calibrate,
    combine,
    log_calibrate,
    max_attainable_average_E,
    required_p_for_rejection,
    simulate_type_i_error,
    simulate_type_i_error_dependent,
    sufficient_evidence,
)

pytestmark = pytest.mark.pure

BAD_UNIT = [0.0, 1.0, 1.5, -0.1, math.nan, math.inf]


# ---------------------------------------------------------------------------
# Refusal boundaries: (0, 1) is open at both ends, and None refuses cleanly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [calibrate, log_calibrate])
@pytest.mark.parametrize("kappa", BAD_UNIT + [None])
def test_calibrators_refuse_kappa_outside_the_open_unit_interval(fn, kappa):
    with pytest.raises(EValueError, match="kappa must be a finite value in"):
        fn(0.5, kappa)


@pytest.mark.parametrize("fn", [calibrate, log_calibrate])
def test_calibrators_refuse_a_nan_p(fn):
    with pytest.raises(EValueError) as exc:
        fn(math.nan)
    assert str(exc.value) == "p-value is NaN; refusing to calibrate"


@pytest.mark.parametrize("fn", [calibrate, log_calibrate])
@pytest.mark.parametrize("p", [None])
def test_calibrators_refuse_a_missing_p(fn, p):
    with pytest.raises(EValueError):
        fn(p)


@pytest.mark.parametrize("fn", [calibrate, log_calibrate])
@pytest.mark.parametrize("p", [-0.01, 1.01, 1.5])
def test_calibrators_refuse_p_outside_the_closed_unit_interval(fn, p):
    with pytest.raises(EValueError, match=r"p-value must lie in \[0, 1\]"):
        fn(p)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5])
def test_alpha_is_refused_at_both_ends_everywhere(alpha):
    msg = r"alpha must lie in \(0, 1\)"
    with pytest.raises(EValueError, match=msg):
        sufficient_evidence(10.0, alpha)
    with pytest.raises(EValueError, match=msg):
        required_p_for_rejection(alpha=alpha)
    with pytest.raises(EValueError, match=msg):
        average([1.0], alpha=alpha)
    # An infinite e-value returns (inf, True) before sufficient_evidence is
    # reached, so average's own alpha check is the only guard on that path.
    with pytest.raises(EValueError, match=msg):
        average([math.inf], alpha=alpha)
    with pytest.raises(EValueError, match=msg):
        average_p([0.0], alpha=alpha)
    with pytest.raises(EValueError, match=msg):
        max_attainable_average_E(100, alpha=alpha)
    with pytest.raises(EValueError, match=msg):
        EProcess(claim="c", alpha=alpha)


@pytest.mark.parametrize("kappa", [0.0, 1.0, 1.5])
def test_required_p_refuses_kappa_at_both_ends(kappa):
    with pytest.raises(EValueError, match=r"kappa must lie in \(0, 1\)"):
        required_p_for_rejection(kappa=kappa)


def test_average_messages_name_what_was_refused():
    with pytest.raises(EValueError) as exc:
        average([])
    assert str(exc.value) == "cannot average an empty set of e-values"
    with pytest.raises(EValueError) as exc:
        average([math.nan])
    assert str(exc.value) == "e-value is NaN; refusing to average"


def test_a_zero_e_value_is_a_valid_input():
    # e = 0 is the strongest possible support for the null, not an error.
    E, _ = average([0.0, 2.0])
    assert E == pytest.approx(1.0)


def test_the_mean_of_one_e_value_is_that_e_value_up_to_the_float_max():
    # Scaling the default weights (by 2, say) overflows w * e to inf here.
    assert average([1e308]) == (1e308, True)


def test_an_infinite_e_value_decides_the_mean_whatever_else_is_in_the_set():
    # The early return for an infinite e-value with positive weight is what
    # stops math.fsum from summing the rest: 1e308 + 1e308 raises
    # OverflowError ("intermediate overflow") even beside an inf.
    assert average([math.inf, 1e308, 1e308]) == (math.inf, True)


def test_one_permutation_is_the_smallest_design():
    E_max, _ = max_attainable_average_E(1)
    assert E_max == pytest.approx(calibrate(0.5))


# ---------------------------------------------------------------------------
# Non-default arguments reach the arithmetic
# ---------------------------------------------------------------------------

KAPPA = 0.3            # anything but DEFAULT_KAPPA
ALPHA = 0.2            # anything but DEFAULT_ALPHA
assert KAPPA != DEFAULT_KAPPA and ALPHA != DEFAULT_ALPHA


def test_add_calibrates_at_the_kappa_it_is_given():
    proc = EProcess(claim="c")
    adj = proc.add("P-1", 0.01, kappa=KAPPA)
    assert adj.e_value == pytest.approx(calibrate(0.01, KAPPA))
    assert adj.log_e_value == pytest.approx(log_calibrate(0.01, KAPPA))
    assert adj.e_value != pytest.approx(calibrate(0.01))


def test_decision_uses_the_alpha_it_is_given():
    proc = EProcess(claim="c")                  # alpha 0.05, threshold 20
    proc.add("P-1", 0.004)                      # e = 0.5 / sqrt(0.004) ~ 7.9
    assert 1 / ALPHA < proc.E < 1 / DEFAULT_ALPHA
    assert proc.decision() == "insufficient_evidence"
    assert proc.decision(alpha=ALPHA) == "reject_null"


def test_next_p_needed_uses_its_alpha_kappa_and_the_evidence_so_far():
    proc = EProcess(claim="c")
    fresh = proc.next_p_needed()
    assert fresh == pytest.approx(required_p_for_rejection())
    proc.add("P-1", 0.01)                       # E = 5, log E > 0
    after = proc.next_p_needed()
    assert after == pytest.approx(required_p_for_rejection(log_E_prior=proc.log_E))
    assert after > fresh                        # evidence so far lowers the bar
    assert proc.next_p_needed(alpha=ALPHA, kappa=KAPPA) == pytest.approx(
        required_p_for_rejection(alpha=ALPHA, kappa=KAPPA, log_E_prior=proc.log_E))


def test_required_p_at_its_defaults_is_the_closed_form():
    # log p = (log(1/alpha) - log kappa) / (kappa - 1) = -2 log 40 at 0.05, 0.5.
    assert required_p_for_rejection() == pytest.approx(1 / 1600)


def test_combine_uses_its_kappa_and_alpha():
    ps = [0.01, 0.2]
    E, _ = combine(ps, kappa=KAPPA, alpha=ALPHA)
    assert E == pytest.approx(calibrate(0.01, KAPPA) * calibrate(0.2, KAPPA))
    # 1/ALPHA = 5 < E_default < 20 = 1/DEFAULT_ALPHA: only alpha decides.
    E_d, rej_d = combine([0.004])              # e ~ 7.9
    assert 1 / ALPHA < E_d < 1 / DEFAULT_ALPHA and not rej_d
    assert combine([0.004], alpha=ALPHA)[1] is True


def test_average_p_passes_kappa_alpha_and_weights_through():
    ps, ws = [0.01, 0.5], [3.0, 1.0]            # E ~ 5.8: between 1/ALPHA and 20
    got = average_p(ps, kappa=KAPPA, alpha=ALPHA, weights=ws)
    want = average([calibrate(p, KAPPA) for p in ps], alpha=ALPHA, weights=ws)
    assert got[0] == pytest.approx(want[0])
    assert got[1] == want[1] is True
    assert average_p(ps, kappa=KAPPA, weights=ws)[1] is False
    assert got[0] != pytest.approx(average_p(ps, kappa=KAPPA, alpha=ALPHA)[0])


def test_average_decides_at_its_alpha():
    assert average([10.0])[1] is False
    assert average([10.0], alpha=ALPHA)[1] is True


def test_max_attainable_uses_its_kappa_and_alpha():
    E_max, reject = max_attainable_average_E(399, kappa=KAPPA, alpha=ALPHA)
    assert E_max == pytest.approx(calibrate(1 / 400, KAPPA))
    assert reject == sufficient_evidence(calibrate(1 / 400, KAPPA), ALPHA)
    # At kappa 0.5, E_max = 10: over 1/ALPHA = 5, under 1/DEFAULT_ALPHA = 20.
    assert max_attainable_average_E(399, alpha=ALPHA)[1] is True
    assert max_attainable_average_E(399)[1] is False


# ---------------------------------------------------------------------------
# The record. No production code calls to_record / from_record: the ledger
# replay rebuilds through EProcess + add (core/adjudication.py claim_process,
# verify_ledger), tested at non-default alpha and kappa in
# tests/test_core_adjudication.py. These pin the round trip as written.
# ---------------------------------------------------------------------------

def _proc():
    proc = EProcess(claim="C", alpha=ALPHA)
    proc.add("P-1", 0.02, kappa=KAPPA)
    proc.add("P-2", 0.4)
    return proc


def test_to_record_writes_the_threshold_and_counts():
    rec = _proc().to_record()
    assert rec["threshold"] == pytest.approx(1 / ALPHA)
    assert rec["n_experiments"] == 2
    assert [e["log_e_value"] for e in rec["experiments"]] == pytest.approx(
        [log_calibrate(0.02, KAPPA), log_calibrate(0.4)])


def test_from_record_keeps_a_non_default_alpha_and_kappa():
    proc = _proc()
    back = EProcess.from_record(proc.to_record())
    assert back.alpha == ALPHA
    assert [a.kappa for a in back.adjudications] == [KAPPA, DEFAULT_KAPPA]
    assert back.E == pytest.approx(proc.E)
    assert back.decision() == proc.decision()


def test_from_record_defaults_a_missing_kappa():
    back = EProcess.from_record(
        {"claim": "C", "experiments": [{"prediction_id": "P-1", "p_value": 0.02}]})
    assert back.adjudications[0].kappa == DEFAULT_KAPPA
    assert back.E == pytest.approx(calibrate(0.02))


def test_from_record_of_an_empty_claim_is_empty():
    back = EProcess.from_record({"claim": "C"})
    assert back.adjudications == [] and back.alpha == DEFAULT_ALPHA


# ---------------------------------------------------------------------------
# The simulations: a Type-I test that cannot fail is not a test
# ---------------------------------------------------------------------------

# Closed forms: tools/math_checks/evalue_type_i_known_answers.py. Each rate is
# measured at alpha = 0.5 so it is large enough to see; each tolerance is
# about 3.3 standard errors at its trial count.
RATE_1 = 0.0625        # one experiment: (alpha * kappa)^(1/(1-kappa))
RATE_2 = 0.0806        # two experiments, kappa 1/2: (1 + 2 log 8) / 64
RATE_PRODUCT_25 = 0.1967   # 25 copies of one p, product, alpha 0.05
RATE_MEAN_2 = 0.0515   # mean of two independent e-values: 41/896 + 3 log 7 / 1024


def _uniform_ps(n, seed):
    import numpy as np
    return np.random.default_rng(seed).uniform(size=n).tolist()


def test_one_experiment_rejects_at_the_closed_form_rate():
    # With one experiment, e >= 1/alpha  <=>  p <= (alpha * kappa)^(1/(1-kappa)).
    # At alpha = 0.5, kappa = 0.5 that is 0.0625: large enough to measure, so a
    # simulation that never rejects, or rejects at the wrong threshold, fails.
    # This pins the numpy helper, which re-derives the calibrator; the tests
    # below pin the same answers through the code that scores.
    rate = simulate_type_i_error(n_trials=40_000, n_experiments=1,
                                 alpha=0.5, kappa=0.5, seed=5)
    assert rate == pytest.approx(RATE_1, abs=0.004)


@pytest.mark.parametrize("n_experiments, want", [(1, RATE_1), (2, RATE_2)])
def test_the_e_process_rejects_at_the_closed_form_rate(n_experiments, want):
    # EProcess.add + decision(), as core/adjudication.py uses them. Two
    # experiments check the accumulation: a process that kept only its last
    # factor would reject at RATE_1.
    ps = _uniform_ps(40_000 * n_experiments, seed=6)
    rejected = 0
    for i in range(0, len(ps), n_experiments):
        proc = EProcess(claim="c", alpha=0.5)
        for j, p in enumerate(ps[i:i + n_experiments]):
            proc.add(f"P-{j}", p, kappa=0.5)
        rejected += proc.decision() == "reject_null"
    assert rejected / 40_000 == pytest.approx(want, abs=0.0045)


def test_average_p_keeps_the_rate_under_maximal_dependence():
    # 25 copies of one p per trial. average_p's mean of 25 equal e-values is
    # that e-value, so it rejects at the one-experiment rate; the product of
    # the same copies (combine) rejects at 0.197 against a nominal 0.05.
    ps = _uniform_ps(10_000, seed=7)
    avg = sum(average_p([p] * 25, alpha=0.5)[1] for p in ps) / len(ps)
    prod = sum(combine([p] * 25)[1] for p in ps) / len(ps)
    assert avg == pytest.approx(RATE_1, abs=0.008)
    assert prod == pytest.approx(RATE_PRODUCT_25, abs=0.013)


def test_average_p_rejects_two_independent_p_values_at_the_mean_s_rate():
    # On equal inputs the mean, the max and the median agree, so the test
    # above cannot see the merger. Here they differ: the mean rejects iff
    # p1^-1/2 + p2^-1/2 >= 8 (0.0515); the max, not valid under dependence,
    # at 1 - (15/16)^2 = 0.121; the product (combine) at RATE_2 = 0.0806.
    ps = _uniform_ps(80_000, seed=8)
    rate = sum(average_p(ps[i:i + 2], alpha=0.5)[1] for i in range(0, len(ps), 2)) / 40_000
    assert rate == pytest.approx(RATE_MEAN_2, abs=0.0037)


def test_the_simulations_are_seeded():
    # Each rate is a mean of a few thousand booleans, so two unseeded runs
    # agree by chance: at 2000 trials and alpha 0.05, about 4 % of the time,
    # which let an unseeded helper survive one mutation run. At alpha 0.5
    # over five seeds the chance is below 1e-8.
    assert simulate_type_i_error(n_trials=2000) == simulate_type_i_error(n_trials=2000)
    assert (simulate_type_i_error_dependent(n_trials=2000)
            == simulate_type_i_error_dependent(n_trials=2000))
    for seed in range(5):
        assert (simulate_type_i_error(n_trials=4000, alpha=0.5, seed=seed)
                == simulate_type_i_error(n_trials=4000, alpha=0.5, seed=seed))
        assert (simulate_type_i_error_dependent(n_trials=4000, alpha=0.5, seed=seed)
                == simulate_type_i_error_dependent(n_trials=4000, alpha=0.5, seed=seed))


def test_the_dependent_simulation_defaults_to_the_average():
    # Its default merger is the one the guarantee is claimed for.
    assert (simulate_type_i_error_dependent(n_trials=4000, alpha=0.5, seed=2)
            == simulate_type_i_error_dependent(n_trials=4000, alpha=0.5, seed=2,
                                               merger="average"))


def test_combine_saturates_to_infinity_instead_of_overflowing():
    # Three p = 1e-300 give log E ~ 1034, past what math.exp can return.
    E, reject = combine([1e-300] * 3)
    assert E == math.inf and reject is True


def test_combine_saturates_just_past_the_float_max():
    # Three ordinary doubles give log E ~ 709.9, between log(DBL_MAX) ~ 709.78
    # and 710: a cutoff at 710 would call math.exp there and overflow.
    assert combine([7.26e-207] * 3) == (math.inf, True)
