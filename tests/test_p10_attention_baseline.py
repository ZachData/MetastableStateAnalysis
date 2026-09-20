"""
`tools/run/p10_attention_baseline.py` — row A0.

The tests that matter here are the ones that would catch the row producing a
confident number about nothing: a correction that does not actually flatten the
content-free case, a layer pairing that quietly differs from the producer this
row exists to be compared against, an empty population counted as a measured
1.0, and a merger that manufactures evidence from dependent units.
"""
import numpy as np
import pytest

from core.parking import uniform_causal_attention
from tools.run.p10_attention_baseline import (
    N_PERMUTATIONS,
    aggregate,
    clustered_enrichment,
    measure_directory,
    measure_layer,
    noise_enrichment,
)

# Tier: numpy and scipy only -- no torch, transformers, sklearn or
# matplotlib -- so this runs in `scripts/check.sh pure`. Declared, not
# assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure



def _attn(n, n_heads=2, matrix=None):
    m = uniform_causal_attention(n) if matrix is None else matrix
    return np.repeat(m[None, :, :], n_heads, axis=0)


def _rng():
    return np.random.default_rng(0)


# --- the enrichment helpers ------------------------------------------------

def test_the_two_populations_partition_the_tokens():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    lab = np.array([-1, 0, -1, 0])
    # noise mean 2, clustered mean 3, overall 2.5
    assert noise_enrichment(v, lab) == pytest.approx(0.8)
    assert clustered_enrichment(v, lab) == pytest.approx(1.2)


def test_enrichments_have_the_signature_the_null_calls():
    from core.nulls import label_permutation_null

    v = np.array([1.0, 5.0, 2.0, 9.0, 3.0])
    lab = np.array([-1, 0, 0, -1, -1])
    draws = label_permutation_null(v, lab, noise_enrichment, n_permutations=32,
                                   rng=_rng())
    assert draws.shape == (32,) and np.isfinite(draws).all()


# --- the correction actually corrects --------------------------------------

def test_content_free_attention_corrects_to_no_enrichment_either_way():
    """The row's defining property. On an empty network the RAW statistic
    reports a large spurious enrichment driven entirely by where the labels
    sit, and the CORRECTED one reports exactly 1 for both populations."""
    n = 200
    lab = np.full(n, -1)
    lab[100:] = 0                       # noise early, clustered late
    rec = measure_layer(_attn(n), lab, _rng())
    assert rec["corrected_noise"] == pytest.approx(1.0, abs=1e-3)
    assert rec["corrected_clustered"] == pytest.approx(1.0, abs=1e-3)
    assert rec["raw_noise"] > 1.5       # the artefact the row is about
    assert rec["raw_clustered"] < 0.6


def test_the_content_free_case_reproduces_the_reported_flip():
    """`math-10.md` §1's claim in one assertion: a partition whose unclustered
    members average early and whose clustered members average late reproduces
    1.6x / 0.5x with no learned behaviour anywhere. If this ever stops holding,
    the row's premise is wrong and the docstring is lying."""
    n = 264
    lab = np.full(n, -1)
    lab[120:] = 0
    rec = measure_layer(_attn(n, n_heads=16), lab, _rng())
    assert rec["raw_noise"] == pytest.approx(1.6, abs=0.25)
    assert rec["raw_clustered"] == pytest.approx(0.5, abs=0.25)


def test_the_corrected_p_is_unremarkable_on_content_free_attention():
    """And it must not merely report 1.0 -- it must fail to reject.

    This is the test that found the tie bug. On a content-free matrix the
    correction explains the layer exactly, so no permutation can move the
    enrichment; the strict comparison read that as the RESOLUTION FLOOR
    because ~1.0 differs from ~1.0 in the last bits. The honest answer is
    p = 1, flagged as a degenerate null."""
    n = 150
    lab = np.full(n, -1)
    lab[75:] = 0
    rec = measure_layer(_attn(n), lab, _rng())
    assert rec["corrected_p"] == pytest.approx(1.0)
    assert rec["corrected_degenerate"] is True
    assert rec["raw_p"] < 0.05          # the uncorrected one does reject


def test_the_correction_still_sees_real_routing():
    """Power: plant genuine content routing onto the noise population and the
    corrected statistic must pick it up."""
    n = 120
    lab = np.full(n, -1)
    lab[::2] = 0                        # interleaved, so position is neutral
    m = uniform_causal_attention(n)
    noise_idx = np.flatnonzero(lab == -1)
    m[:, noise_idx] *= 4.0
    m = m / m.sum(axis=1, keepdims=True)
    rec = measure_layer(_attn(n, matrix=m), lab, _rng())
    assert rec["corrected_noise"] > 1.3
    assert rec["corrected_p"] <= 1.0 / (N_PERMUTATIONS + 1) + 1e-12


def test_position_bias_is_measured_on_the_same_units():
    n = 100
    lab = np.full(n, -1)
    lab[50:] = 0
    rec = measure_layer(_attn(n), lab, _rng())
    assert rec["position_bias"] == pytest.approx(50.0 / 99.0, abs=1e-4)
    assert rec["position_bias_p"] < 0.01


def test_interleaved_labels_have_no_position_bias():
    n = 100
    lab = np.full(n, -1)
    lab[::2] = 0
    rec = measure_layer(_attn(n), lab, _rng())
    assert abs(rec["position_bias"]) < 0.02
    assert rec["position_bias_p"] > 0.05


# --- degenerate layers are dropped, not scored -----------------------------

def test_a_layer_with_no_noise_is_dropped_not_scored_as_one():
    assert measure_layer(_attn(20), np.zeros(20, dtype=int), _rng()) is None


def test_a_layer_that_is_all_noise_is_dropped():
    assert measure_layer(_attn(20), np.full(20, -1), _rng()) is None


def test_mismatched_label_length_is_refused_rather_than_broadcast():
    lab = np.full(10, -1)
    lab[5:] = 0
    with pytest.raises(ValueError, match="disagree"):
        measure_layer(_attn(20), lab, _rng())


# --- the directory walk ----------------------------------------------------

def test_a_directory_without_a_partition_is_skipped_with_a_reason(tmp_path):
    d = tmp_path / "2026-01-01_00-00-00" / "pythia-410m-step0_wiki"
    d.mkdir(parents=True)
    np.savez(d / "attentions.npz", attentions=np.zeros((2, 1, 4, 4)))
    got = measure_directory(d, _rng())
    assert got["skipped"] == "no HDBSCAN partition"
    assert "layers" not in got


def test_a_directory_without_attention_is_skipped_with_a_reason(tmp_path):
    import json

    d = tmp_path / "2026-01-01_00-00-00" / "pythia-410m-step0_wiki"
    d.mkdir(parents=True)
    (d / "hdbscan_labels.json").write_text(json.dumps({"0": [0, -1, 0, -1]}))
    got = measure_directory(d, _rng())
    assert got["skipped"] == "no attentions.npz"


def test_the_layer_pairing_matches_the_producer_this_row_is_compared_against(tmp_path):
    """`noise_importance_proxy` pairs partition layer l with attn[l] -- the
    attention that READS that state. Row A0 must not re-pair the axes, or the
    corrected number is not comparable with the flip as reported.

    Built with 3 attention entries and 4 partition layers, the real shape
    (25 activation rows against 24 blocks), and checked by giving each
    attention index a distinguishable signature."""
    import json

    d = tmp_path / "2026-01-01_00-00-00" / "pythia-410m-step0_wiki"
    d.mkdir(parents=True)
    n = 40
    lab = np.full(n, -1)
    lab[::2] = 0
    stack = np.stack([_attn(n)[0] for _ in range(3)])[:, None, :, :]
    # Route heavily onto token 7 in attention index 2 only.
    m = stack[2, 0].copy()
    m[:, 7] += 3.0
    stack[2, 0] = m / m.sum(axis=1, keepdims=True)
    np.savez(d / "attentions.npz", attentions=stack)
    (d / "hdbscan_labels.json").write_text(
        json.dumps({str(k): lab.tolist() for k in range(4)})
    )

    got = measure_directory(d, _rng())
    seen = sorted(r["layer"] for r in got["layers"])
    assert seen == [0, 1, 2], "layer 3 has no block above it and must be dropped"
    by_layer = {r["layer"]: r for r in got["layers"]}
    # The planted routing must land on partition layer 2, not 1 or 3.
    assert by_layer[2]["corrected_noise"] != pytest.approx(
        by_layer[0]["corrected_noise"], abs=1e-6
    )
    assert by_layer[0]["corrected_noise"] == pytest.approx(
        by_layer[1]["corrected_noise"], abs=1e-6
    )


# --- the aggregate ---------------------------------------------------------

def test_aggregate_of_nothing_says_so():
    assert aggregate([{"run_dir": "x", "skipped": "no attentions.npz"}]) == {"n_units": 0}


def test_aggregate_merges_with_the_mean_not_the_product():
    """The guard against manufacturing evidence from a sweep. Twenty-five
    dependent units each at p = 0.19 -- not evidence at alpha = 0.05 -- must
    not merge into a rejection."""
    from core.evalues import calibrate

    rows = [{
        "raw_noise": 1.0, "raw_clustered": 1.0, "corrected_noise": 1.0,
        "corrected_clustered": 1.0, "position_bias": 0.0, "noise_fraction": 0.4,
        "raw_p": 0.19, "corrected_p": 0.19, "position_bias_p": 0.19,
    } for _ in range(25)]
    got = aggregate([{"layers": rows}])
    assert got["n_units"] == 25
    assert got["corrected_E"] == pytest.approx(calibrate(0.19), abs=1e-3)
    assert got["corrected_reject"] is False


def test_aggregate_reports_the_fraction_below_the_conventional_level():
    rows = [{
        "raw_noise": 1.0, "raw_clustered": 1.0, "corrected_noise": 1.0,
        "corrected_clustered": 1.0, "position_bias": 0.0, "noise_fraction": 0.4,
        "raw_p": p, "corrected_p": p, "position_bias_p": p,
    } for p in (0.01, 0.02, 0.5, 0.9)]
    got = aggregate([{"layers": rows}])
    assert got["corrected_frac_p_below_05"] == pytest.approx(0.5)
    assert got["corrected_median_p"] == pytest.approx(0.26)


def test_aggregate_skips_nan_enrichments_rather_than_poisoning_the_mean():
    rows = [
        {"raw_noise": 1.0, "raw_clustered": 1.0, "corrected_noise": 2.0,
         "corrected_clustered": 1.0, "position_bias": 0.0, "noise_fraction": 0.4,
         "raw_p": 0.5, "corrected_p": 0.5, "position_bias_p": 0.5},
        {"raw_noise": float("nan"), "raw_clustered": 1.0, "corrected_noise": 4.0,
         "corrected_clustered": 1.0, "position_bias": 0.0, "noise_fraction": 0.4,
         "raw_p": 0.5, "corrected_p": 0.5, "position_bias_p": 0.5},
    ]
    got = aggregate([{"layers": rows}])
    assert got["mean_raw_noise"] == pytest.approx(1.0)
    assert got["mean_corrected_noise"] == pytest.approx(3.0)


def test_the_design_can_reject_and_one_layer_still_cannot_carry_it():
    """Two properties that pull in opposite directions and are both needed.

    The draw count must be high enough that the design COULD reject -- at 400
    permutations the largest attainable merged e-value was 10.01 against a
    threshold of 20, so `reject: False` said nothing about the data. And one
    lucky layer must not be able to carry a sweep's verdict. The first is the
    draw count's job; the second is the MERGER's, because the mean of 3 646
    units containing one at the ceiling is the ceiling over 3 646."""
    from core.evalues import DEFAULT_ALPHA, average_p, max_attainable_average_E

    E_max, can_reject = max_attainable_average_E(N_PERMUTATIONS)
    assert can_reject is True

    floor = 1.0 / (N_PERMUTATIONS + 1)
    one_extreme = [floor] + [0.5] * 3645
    assert average_p(one_extreme)[1] is False
    assert average_p(one_extreme)[0] < 1.0 / DEFAULT_ALPHA


# --- the checkpoint axis ---------------------------------------------------

def _ck_row(raw_n=2.0, raw_c=0.6, corr_n=1.15, corr_c=0.98, p=0.5):
    return {"raw_noise": raw_n, "raw_clustered": raw_c,
            "corrected_noise": corr_n, "corrected_clustered": corr_c,
            "position_bias": 0.02, "noise_fraction": 0.39,
            "raw_p": p, "corrected_p": p, "position_bias_p": p}


def test_the_checkpoint_split_exists_because_averaging_hides_development():
    """The sweep's mean corrected gap is small; the late checkpoints' is not.
    A row that only reported the mean would have called a real developmental
    effect no effect."""
    dirs = [
        {"checkpoint": 0, "layers": [_ck_row(1.17, 0.92, 1.003, 0.999, p=0.9)]},
        {"checkpoint": 143000, "layers": [_ck_row(2.28, 0.59, 1.153, 0.981, p=0.001)]},
    ]
    got = aggregate(dirs)
    early = got["by_checkpoint"]["0"]
    late = got["by_checkpoint"]["143000"]
    assert early["corrected_gap"] == pytest.approx(0.004, abs=1e-3)
    assert late["corrected_gap"] == pytest.approx(0.172, abs=1e-3)
    assert late["corrected_E"] > early["corrected_E"]


def test_gap_surviving_correction_is_a_fraction_of_the_raw_gap():
    dirs = [{"checkpoint": 5, "layers": [_ck_row(1.5, 0.5, 1.1, 0.9)]}]
    ck = aggregate(dirs)["by_checkpoint"]["5"]
    assert ck["raw_gap"] == pytest.approx(1.0)
    assert ck["corrected_gap"] == pytest.approx(0.2)
    assert ck["gap_surviving_correction"] == pytest.approx(0.2)


def test_a_zero_raw_gap_gives_none_not_a_division_by_zero():
    dirs = [{"checkpoint": 5, "layers": [_ck_row(1.0, 1.0, 1.0, 1.0)]}]
    assert aggregate(dirs)["by_checkpoint"]["5"]["gap_surviving_correction"] is None


def test_checkpoints_sort_numerically():
    dirs = [{"checkpoint": c, "layers": [_ck_row()]} for c in (0, 8, 512, 143000)]
    assert list(aggregate(dirs)["by_checkpoint"]) == ["0", "8", "512", "143000"]


def test_directories_without_a_checkpoint_do_not_break_the_split():
    got = aggregate([{"checkpoint": None, "layers": [_ck_row()]}])
    assert got["by_checkpoint"] == {}
    assert got["n_units"] == 1


def test_the_statistic_is_tested_unrounded():
    """The record rounds to 4 dp for readability; the p-value must not be
    computed from the rounded value. A 5e-05 rounding error against unrounded
    null draws is about fifty thousand times the tie tolerance, so draws within
    it land on whichever side the rounding sent them.

    Checked by construction: a layer whose enrichment rounds DOWN must not get
    a p-value that a rounded-down observation would have earned. The two
    differ only when a draw sits in the gap, so this asserts the code path
    rather than the arithmetic -- the runner must pass the same float it
    measured."""
    import inspect

    from tools.run import p10_attention_baseline as mod

    src = inspect.getsource(mod.measure_layer)
    assert 'p_from_null_tolerant(raw_noise, raw_draws' in src
    assert 'p_from_null_tolerant(out["raw_noise"]' not in src
    assert 'p_from_null_tolerant(corrected_noise, draws' in src
