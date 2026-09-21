"""
`core.parking` — Phase 10's free-row instruments.

The discipline here: every closed form is checked against an explicit
construction that goes through the same code path the real data does, and
every statistic is checked against a case whose answer is known by hand. A
baseline asserted rather than derived is exactly what `attention-10.md` row A0
exists to stop, and this file would be the wrong place to reintroduce it.
"""
import math

import numpy as np
import pytest

from core.parking import (
    clustered_position_bias,
    cluster_nuclei,
    harmonic,
    mask_corrected_received,
    mean_nucleus_position,
    log_partition_function,
    log_position_corrected_partition_function,
    partition_function,
    population_enrichment,
    position_corrected_partition_function,
    received_attention,
    received_baseline,
    relative_to_layer_mean,
    uniform_causal_attention,
)

# Tier: numpy and scipy only -- no torch, transformers, sklearn or
# matplotlib -- so this runs in `scripts/check.sh pure`. Declared, not
# assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure



# --- harmonic numbers ------------------------------------------------------

def test_harmonic_matches_an_explicit_loop():
    for k in (0, 1, 2, 5, 264):
        assert harmonic(k) == pytest.approx(sum(1.0 / m for m in range(1, k + 1)))


def test_harmonic_refuses_negative():
    with pytest.raises(ValueError, match="k >= 0"):
        harmonic(-1)


# --- the causal-mask baseline ---------------------------------------------

def test_uniform_causal_attention_is_causal_and_row_stochastic():
    a = uniform_causal_attention(6)
    assert np.allclose(a.sum(axis=1), 1.0)
    assert np.allclose(np.triu(a, k=1), 0.0)
    assert a[3, 0] == pytest.approx(1.0 / 4.0)
    assert a[3, 4] == 0.0


@pytest.mark.parametrize("n", [2, 5, 17, 264])
@pytest.mark.parametrize("zero_diagonal", [True, False])
def test_baseline_closed_form_equals_the_constructed_matrix(n, zero_diagonal):
    """The point of the whole row: the closed form and the measurement are the
    same quantity, so they can be divided."""
    a = uniform_causal_attention(n)[None, :, :]        # one head
    measured = received_attention(a, zero_diagonal=zero_diagonal)
    closed = received_baseline(n, n_heads=1, zero_diagonal=zero_diagonal)
    assert np.allclose(measured, closed)


@pytest.mark.parametrize("n", [3, 20, 264])
def test_baseline_with_diagonal_sums_to_n_so_the_layer_mean_is_one(n):
    """`math-10.md` §1: sum_j received(j) = n exactly, so `received(j)` IS the
    'x layer average' quantity and needs no rescaling."""
    b = received_baseline(n, zero_diagonal=False)
    assert b.sum() == pytest.approx(float(n))
    assert b.mean() == pytest.approx(1.0)


def test_the_battery_numbers_math_10_reports():
    """The three values quoted in `math-10.md` §1 and in INDEX.md's priority
    block, so a change to either is caught here."""
    b = received_baseline(264, zero_diagonal=False)
    assert b[0] == pytest.approx(6.155, abs=5e-4)
    assert b[132] == pytest.approx(0.691, abs=5e-4)
    assert b[263] == pytest.approx(0.0038, abs=5e-5)
    assert b[0] / b[263] == pytest.approx(1600, rel=0.05)   # the ~1 600x tilt


def test_the_baseline_crossings_that_reproduce_the_flip():
    """`math-10.md` §1: content-free, the baseline already equals 1.6x at
    position ~53 and 0.5x at ~160. That is why the flip is not yet known to be
    a measurement of anything."""
    b = received_baseline(264, zero_diagonal=False)
    assert int(np.argmin(np.abs(b - 1.6))) == pytest.approx(53, abs=2)
    assert int(np.argmin(np.abs(b - 0.5))) == pytest.approx(160, abs=3)


def test_diagonal_zeroed_baseline_is_exactly_zero_at_the_last_position():
    """The reason `mask_corrected_received` keeps the diagonal."""
    b = received_baseline(40, zero_diagonal=True)
    assert b[-1] == pytest.approx(0.0, abs=1e-12)
    assert (b[:-1] > 0).all()


def test_baseline_scales_linearly_in_head_count():
    one = received_baseline(12, n_heads=1)
    sixteen = received_baseline(12, n_heads=16)
    assert np.allclose(sixteen, 16.0 * one)


def test_received_attention_matches_the_existing_producer():
    """Byte-for-byte agreement with
    `noise_importance_proxy._received_attention`, so a corrected number stays
    comparable with the flip as this project has always reported it."""
    rng = np.random.default_rng(11)
    a = rng.random((4, 9, 9))
    ref = a.copy()
    for h in range(ref.shape[0]):
        np.fill_diagonal(ref[h], 0.0)
    assert np.allclose(received_attention(a), ref.sum(axis=(0, 1)))


def test_received_attention_does_not_mutate_its_input():
    a = np.ones((2, 4, 4))
    received_attention(a)
    assert np.allclose(np.diagonal(a, axis1=1, axis2=2), 1.0)


def test_received_attention_refuses_a_non_square_argument():
    with pytest.raises(ValueError, match=r"\(n_heads, n, n\)"):
        received_attention(np.ones((2, 3, 4)))


# --- the correction itself -------------------------------------------------

def test_correcting_the_content_free_matrix_returns_exactly_one():
    """The correction's defining property: an empty network corrects to flat."""
    a = uniform_causal_attention(50)[None, :, :]
    assert np.allclose(mask_corrected_received(a), 1.0)


def test_correction_is_flat_for_many_identical_heads_too():
    a = np.repeat(uniform_causal_attention(30)[None, :, :], 16, axis=0)
    assert np.allclose(mask_corrected_received(a), 1.0)


def test_uncorrected_statistic_is_steeply_tilted_where_the_corrected_one_is_flat():
    """The two readings of the same content-free matrix, side by side. This is
    the whole argument of row A0 in one assertion."""
    a = uniform_causal_attention(264)[None, :, :]
    raw = relative_to_layer_mean(received_attention(a, zero_diagonal=False))
    corrected = mask_corrected_received(a)
    assert raw.max() / raw.min() > 1000
    assert corrected.max() / corrected.min() == pytest.approx(1.0, abs=1e-9)


def test_correction_detects_content_planted_on_one_token():
    """Route extra attention to token 20 and the corrected statistic should
    rise there and nowhere else in particular."""
    n = 60
    a = uniform_causal_attention(n)
    a[30:, :] *= 0.5
    a[30:, 20] += 0.5                      # every late query sends half to 20
    a = a / a.sum(axis=1, keepdims=True)
    corrected = mask_corrected_received(a[None, :, :])
    assert corrected[20] == corrected.max()
    assert corrected[20] > 5.0


def test_relative_to_layer_mean_refuses_a_degenerate_mean():
    with pytest.raises(ValueError, match="not usable"):
        relative_to_layer_mean(np.zeros(5))


# --- enrichment ------------------------------------------------------------

def test_population_enrichment_by_hand():
    v = np.array([1.0, 3.0, 5.0, 7.0])          # mean 4
    m = np.array([True, False, True, False])    # mean 3
    assert population_enrichment(v, m) == pytest.approx(0.75)


def test_enrichment_of_everything_is_one():
    v = np.array([2.0, 9.0, 4.0])
    assert population_enrichment(v, np.ones(3, dtype=bool)) == pytest.approx(1.0)


def test_empty_population_is_nan_not_zero():
    v = np.array([2.0, 9.0, 4.0])
    assert math.isnan(population_enrichment(v, np.zeros(3, dtype=bool)))


def test_enrichment_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="must match"):
        population_enrichment(np.ones(3), np.ones(4, dtype=bool))


def test_the_two_reported_ratios_pin_the_split():
    """`math-10.md` §1's internal-consistency check: since the populations
    partition the same tokens and the mean is 1, f*1.6 + (1-f)*0.5 = 1 forces
    f = 5/11. Constructed here so the enrichment function is shown to obey it."""
    n = 1100
    f = 5.0 / 11.0
    mask = np.zeros(n, dtype=bool)
    mask[: int(round(f * n))] = True
    v = np.where(mask, 1.6, 0.5)
    assert v.mean() == pytest.approx(1.0)
    assert population_enrichment(v, mask) == pytest.approx(1.6)
    assert population_enrichment(v, ~mask) == pytest.approx(0.5)


# --- F0, the anchor statistic ---------------------------------------------

def test_cluster_nuclei_by_hand():
    lab = [0, -1, 1, 0, 1, 1, -1]
    got = cluster_nuclei(lab)
    assert set(got) == {0, 1}
    assert got[0] == {"first": 0, "last": 3, "mean": 1.5, "size": 2}
    assert got[1] == {"first": 2, "last": 5, "mean": pytest.approx(11 / 3), "size": 3}


def test_cluster_nuclei_excludes_noise():
    assert cluster_nuclei([-1, -1, -1]) == {}


def test_nucleus_statistic_is_zero_when_every_cluster_starts_at_the_front():
    lab = np.array([0, 1, 2, 0, 0, 1, 2, 1])
    pos = np.arange(lab.size)
    # clusters start at 0, 1, 2 -> mean 1, over (n-1) = 7
    assert mean_nucleus_position(pos, lab) == pytest.approx(1.0 / 7.0)


def test_nucleus_statistic_is_large_when_clusters_start_late():
    n = 20
    lab = np.full(n, -1)
    lab[15:] = [0, 0, 1, 1, 1]
    pos = np.arange(n)
    # nuclei at 15 and 17 -> mean 16, over 19
    assert mean_nucleus_position(pos, lab) == pytest.approx(16.0 / 19.0)


def test_nucleus_statistic_is_nan_with_no_clusters():
    assert math.isnan(mean_nucleus_position(np.arange(5), np.full(5, -1)))


def test_nucleus_statistic_signature_fits_the_permutation_null():
    """It must be callable as metric_fn(fixed, labels) by
    `core.nulls.label_permutation_null`, and the null must preserve the size
    profile -- otherwise the statistic's mechanical dependence on cluster count
    is not controlled and the p-value means nothing."""
    from core.nulls import label_permutation_null

    rng = np.random.default_rng(3)
    lab = np.array([0, 0, 1, 1, 1, -1, -1, 2, 2, -1])
    pos = np.arange(lab.size)
    draws = label_permutation_null(pos, lab, mean_nucleus_position,
                                   n_permutations=64, rng=rng)
    assert draws.shape == (64,)
    assert np.isfinite(draws).all()
    assert (draws >= 0).all() and (draws <= 1).all()


def test_nucleus_null_is_centred_where_a_uniform_assignment_puts_it():
    """A sanity check on the null itself: with labels assigned at random, the
    real statistic is an ordinary draw from it."""
    from core.nulls import label_permutation_null, p_from_null

    rng = np.random.default_rng(7)
    n = 200
    lab = rng.permutation(np.concatenate([
        np.repeat(np.arange(20), 5), np.full(n - 100, -1)
    ]))
    pos = np.arange(n)
    obs = mean_nucleus_position(pos, lab)
    draws = label_permutation_null(pos, lab, mean_nucleus_position,
                                   n_permutations=400, rng=rng)
    res = p_from_null(obs, draws, alternative="less")
    assert 0.02 < res["p_value"] < 0.98, res


def test_nucleus_null_detects_a_planted_early_anchoring():
    """And that it has power: put every cluster's first member in the first
    5 % of the sequence and the null should reject."""
    from core.nulls import label_permutation_null, p_from_null

    rng = np.random.default_rng(5)
    n = 400
    lab = np.full(n, -1)
    for cid in range(20):
        lab[cid] = cid                                  # the nucleus, up front
        tail = rng.choice(np.arange(20, n), size=6, replace=False)
        lab[tail] = np.where(lab[tail] == -1, cid, lab[tail])
    pos = np.arange(n)
    obs = mean_nucleus_position(pos, lab)
    draws = label_permutation_null(pos, lab, mean_nucleus_position,
                                   n_permutations=400, rng=rng)
    res = p_from_null(obs, draws, alternative="less")
    assert res["p_value"] < 0.01, res


def test_position_bias_sign_and_scale():
    n = 100
    lab = np.full(n, -1)
    lab[50:] = 0                      # clustered late, noise early
    pos = np.arange(n)
    got = clustered_position_bias(pos, lab)
    # clustered mean 74.5, noise mean 24.5, difference 50 over 99
    assert got == pytest.approx(50.0 / 99.0)


def test_position_bias_is_negative_when_clusters_are_early():
    n = 100
    lab = np.full(n, -1)
    lab[:50] = 0
    assert clustered_position_bias(np.arange(n), lab) < 0


def test_position_bias_is_nan_when_a_population_is_empty():
    assert math.isnan(clustered_position_bias(np.arange(4), np.zeros(4, dtype=int)))
    assert math.isnan(clustered_position_bias(np.arange(4), np.full(4, -1)))


def test_anchor_statistics_refuse_mismatched_shapes():
    for fn in (mean_nucleus_position, clustered_position_bias):
        with pytest.raises(ValueError, match="must match"):
            fn(np.arange(5), np.zeros(4, dtype=int))


# --- F12, the partition function ------------------------------------------

def test_partition_function_at_beta_zero_counts_visible_tokens():
    """With beta = 0 every term is 1, so Z_i is exactly the number of tokens
    row i can see -- i + 1 under a mask, n without one."""
    X = np.random.default_rng(1).normal(size=(9, 4))
    assert np.allclose(partition_function(X, 0.0, causal=True), np.arange(1, 10))
    assert np.allclose(partition_function(X, 0.0, causal=False), 9.0)


def test_partition_function_matches_a_naive_double_loop():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(7, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    beta = 1.7
    naive = np.array([
        sum(math.exp(beta * float(X[i] @ X[j])) for j in range(i + 1))
        for i in range(7)
    ])
    assert np.allclose(partition_function(X, beta), naive)


def test_log_partition_function_is_stable_where_the_linear_one_cannot_be():
    """At beta = 800 the diagonal term alone is e^800, which no float64 holds.
    The log form must still be exact; the linear one is expected to overflow,
    and the test says so rather than working around it."""
    X = np.eye(3)[[0, 0, 1]]
    logZ = log_partition_function(X, 800.0)
    assert np.isfinite(logZ).all()
    # row 1 sees two identical unit vectors, so Z = 2 e^{800}
    assert logZ[1] == pytest.approx(math.log(2.0) + 800.0)
    with np.errstate(over="ignore"):
        assert not np.isfinite(partition_function(X, 800.0)).any()


def test_linear_and_log_agree_where_both_are_representable():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(15, 5))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    for beta in (0.0, 1.0, 2.5, 40.0):
        assert np.allclose(
            np.log(partition_function(X, beta)), log_partition_function(X, beta)
        )


def test_log_position_correction_is_the_log_of_the_linear_one():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(9, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    lin = position_corrected_partition_function(partition_function(X, 1.3))
    log = log_position_corrected_partition_function(log_partition_function(X, 1.3))
    assert np.allclose(np.log(lin), log)


def test_log_position_correction_refuses_wrong_rank():
    with pytest.raises(ValueError, match="1-D"):
        log_position_corrected_partition_function(np.ones((2, 2)))


def test_concentration_regime_is_linear_in_position_under_a_mask():
    """`math-10.md` §2's masked result: with a common inner product,
    Z_i = (i+1) e^{beta gamma}, minimum at position 0 -- the sink."""
    n, gamma, beta = 12, 0.9, 3.0
    G = np.full((n, n), gamma)
    np.fill_diagonal(G, gamma)
    # Build X with exactly this Gram by Cholesky of a PSD matrix.
    w, V = np.linalg.eigh(G)
    X = V @ np.diag(np.sqrt(np.clip(w, 0, None)))
    Z = partition_function(X, beta)
    assert np.allclose(Z / Z[0], np.arange(1, n + 1), rtol=1e-6)
    assert int(np.argmin(Z)) == 0


def test_position_correction_flattens_the_concentration_regime():
    n, gamma, beta = 12, 0.9, 3.0
    G = np.full((n, n), gamma)
    w, V = np.linalg.eigh(G)
    X = V @ np.diag(np.sqrt(np.clip(w, 0, None)))
    corrected = position_corrected_partition_function(partition_function(X, beta))
    assert np.allclose(corrected, corrected[0], rtol=1e-6)


def test_position_correction_is_division_by_i_plus_one():
    Z = np.array([2.0, 8.0, 30.0])
    assert np.allclose(position_corrected_partition_function(Z), [2.0, 4.0, 10.0])


def test_partition_function_refuses_wrong_rank():
    with pytest.raises(ValueError, match=r"\(n, d\)"):
        partition_function(np.ones(5), 1.0)
    with pytest.raises(ValueError, match="1-D"):
        position_corrected_partition_function(np.ones((2, 2)))
