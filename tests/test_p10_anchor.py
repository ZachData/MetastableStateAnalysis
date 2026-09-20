"""
`tools/run/p10_anchor.py` — F0, the anchor test.

The statistic here has a mechanical dependence that is not evidence of
anything: the minimum of a size-s subset falls at about n/(s+1), so the whole
size profile of the partition moves the raw number before any position
coupling does. The design rests on the permutation null holding that fixed.
Most of these tests exist to check that it does, and that a planted effect in
each direction is recovered with the right sign.
"""
import json

import numpy as np
import pytest

from tools.run.p10_anchor import (
    N_PERMUTATIONS,
    aggregate,
    checkpoint_of,
    measure_directory,
    measure_layer,
)


def _rng():
    return np.random.default_rng(0)


# --- checkpoint parsing ----------------------------------------------------

def test_checkpoint_is_read_off_the_directory_name():
    assert checkpoint_of("pythia-410m-step143000_wiki_paragraph") == 143000
    assert checkpoint_of("pythia-410m-step0_camus_letranger") == 0
    assert checkpoint_of("gpt2-large_wiki_paragraph") is None


def test_a_prompt_containing_digits_does_not_become_a_checkpoint():
    assert checkpoint_of("pythia-410m-step16_hdbscan_code") == 16


# --- layers that cannot be scored are dropped ------------------------------

def test_a_layer_with_one_cluster_is_dropped():
    """The statistic is a mean over clusters; one cluster is not a
    distribution, and the null cannot move it meaningfully."""
    lab = np.full(40, -1)
    lab[10:20] = 0
    assert measure_layer(lab, _rng()) is None


def test_an_all_noise_layer_is_dropped():
    assert measure_layer(np.full(40, -1), _rng()) is None


def test_a_layer_with_no_noise_is_still_scored():
    """Unlike row A0, F0's primary statistic needs clusters but not noise --
    the nuclei are defined without reference to the noise population."""
    lab = np.repeat(np.arange(8), 5)
    rec = measure_layer(lab, _rng())
    assert rec is not None
    assert rec["n_clusters"] == 8
    assert rec["position_bias"] is None          # undefined, and said so
    assert rec["position_bias_p"] is None


def test_a_tiny_layer_is_dropped():
    assert measure_layer(np.array([0, 1, -1]), _rng()) is None


# --- the null holds the mechanical artefacts fixed -------------------------

def test_the_null_preserves_cluster_count_and_every_size():
    """If it did not, the statistic's dependence on both would leak into the
    p-value and the row would measure the partition's shape rather than its
    position coupling."""
    from core.nulls import label_permutation_null

    lab = np.array([0, 0, 0, 1, 1, -1, -1, 2, 2, 2, 2, -1])
    sizes = sorted(np.bincount(lab[lab >= 0]).tolist())

    seen = []

    def spy(positions, labels):
        seen.append(sorted(np.bincount(labels[labels >= 0]).tolist()))
        return 0.0

    label_permutation_null(np.arange(lab.size, dtype=float), lab, spy,
                           n_permutations=20, rng=_rng())
    assert all(s == sizes for s in seen)


def test_random_labels_give_an_unremarkable_p():
    """The null's own calibration: with membership independent of position,
    the statistic must be an ordinary draw."""
    rng = np.random.default_rng(4)
    lab = rng.permutation(np.concatenate([np.repeat(np.arange(15), 6),
                                          np.full(110, -1)]))
    rec = measure_layer(lab, rng)
    assert 0.02 < rec["nucleus_p"] < 0.98, rec["nucleus_p"]


def test_planted_early_nuclei_are_detected():
    """Power, in the predicted direction. Every cluster's earliest member in
    the first 5 % of the sequence."""
    rng = np.random.default_rng(5)
    n = 400
    lab = np.full(n, -1)
    for cid in range(20):
        lab[cid] = cid
        tail = rng.choice(np.arange(20, n), size=6, replace=False)
        lab[tail] = np.where(lab[tail] == -1, cid, lab[tail])
    rec = measure_layer(lab, rng)
    assert rec["nucleus_p"] <= 1.0 / (N_PERMUTATIONS + 1) + 1e-12
    assert rec["mean_nucleus_position"] < rec["nucleus_null_mean"]
    assert rec["nucleus_first_decile_fraction"] == pytest.approx(1.0)


def test_planted_late_nuclei_give_a_p_near_one_not_near_zero():
    """The direction is fixed at 'less', so the opposite effect must read as
    no support rather than as support. A row that rejected on either sign
    would be testing nothing.

    Every member of every cluster sits in the last 10 % of the sequence --
    a nucleus is the EARLIEST member, so planting one late means planting the
    whole cluster late."""
    rng = np.random.default_rng(6)
    n = 400
    lab = np.full(n, -1)
    late = rng.permutation(np.arange(int(0.9 * n), n))
    for k, cid in enumerate(range(10)):
        lab[late[4 * k:4 * k + 4]] = cid
    rec = measure_layer(lab, rng)
    assert rec["nucleus_p"] > 0.9
    assert rec["mean_nucleus_position"] > rec["nucleus_null_mean"]


def test_the_raw_statistic_alone_would_have_been_misleading():
    """The reason the null is not optional, and a correction to the intuition
    in `mean_nucleus_position`'s docstring.

    The mechanical driver is CLUSTER SIZE, not cluster count: the minimum of s
    positions drawn from n falls at about n/(s+1), so BIG clusters start early
    and small ones start late, whatever the count. Here two partitions with
    membership independent of position -- both assigned by permutation -- give
    raw statistics a factor of two apart, 0.10 against 0.20, while both
    p-values sit in the middle of their nulls. The raw number carries the size
    profile; only the p-value carries the position coupling."""
    rng = np.random.default_rng(7)
    n = 300
    big = rng.permutation(np.concatenate([np.repeat(np.arange(4), 10),
                                          np.full(n - 40, -1)]))
    small = rng.permutation(np.concatenate([np.repeat(np.arange(40), 4),
                                            np.full(n - 160, -1)]))
    a, b = measure_layer(big, rng), measure_layer(small, rng)
    assert b["mean_nucleus_position"] > 1.8 * a["mean_nucleus_position"]
    assert 0.02 < a["nucleus_p"] < 0.98
    assert 0.02 < b["nucleus_p"] < 0.98


def test_the_null_mean_is_recorded_so_the_raw_value_can_be_read():
    rec = measure_layer(np.repeat(np.arange(10), 8), _rng())
    assert 0.0 <= rec["nucleus_null_mean"] <= 1.0
    assert "mean_nucleus_position" in rec


# --- the directory walk ----------------------------------------------------

def test_a_directory_without_a_partition_is_skipped_with_a_reason(tmp_path):
    d = tmp_path / "2026-01-01_00-00-00" / "pythia-410m-step0_wiki"
    d.mkdir(parents=True)
    (d / "hdbscan_labels.json").write_text("{}")
    assert measure_directory(d, _rng())["skipped"] == "no HDBSCAN partition"


def test_every_partition_layer_is_scored_no_attention_needed(tmp_path):
    """F0 reads the labels and nothing else, so unlike row A0 it keeps the
    last layer -- there is no block-count mismatch to drop it for."""
    d = tmp_path / "2026-01-01_00-00-00" / "pythia-410m-step7_wiki"
    d.mkdir(parents=True)
    lab = np.repeat(np.arange(10), 8).tolist()
    (d / "hdbscan_labels.json").write_text(
        json.dumps({str(k): lab for k in range(25)}))
    got = measure_directory(d, _rng())
    assert [r["layer"] for r in got["layers"]] == list(range(25))
    assert got["checkpoint"] == 7
    assert got["labels_from"] == "native"


# --- the aggregate ---------------------------------------------------------

def _row(p=0.5, nuc=0.3, bias=0.0, n_clusters=10, within=None):
    return {"mean_nucleus_position": nuc, "nucleus_null_mean": 0.3,
            "nucleus_within_null_mean": 0.3,
            "nucleus_p": p, "nucleus_within_p": p if within is None else within,
            "position_bias": bias, "position_bias_p": p,
            "n_clusters": n_clusters, "noise_fraction": 0.4, "n_tokens": 100,
            "nucleus_first_decile_fraction": 0.1}


def test_aggregate_of_nothing_says_so():
    assert aggregate([{"run_dir": "x", "skipped": "no HDBSCAN partition"}]) == {"n_units": 0}


def test_aggregate_merges_with_the_mean_not_the_product():
    from core.evalues import calibrate

    got = aggregate([{"layers": [_row(p=0.19) for _ in range(25)]}])
    assert got["nucleus"]["E"] == pytest.approx(calibrate(0.19), abs=1e-3)
    assert got["nucleus"]["reject"] is False


def test_aggregate_splits_by_checkpoint():
    """A developmental effect averaged over training reads as no effect, so
    the per-checkpoint view is not decoration."""
    dirs = [
        {"checkpoint": 0, "layers": [_row(p=0.9, nuc=0.5)]},
        {"checkpoint": 143000, "layers": [_row(p=0.002, nuc=0.05)]},
    ]
    got = aggregate(dirs)
    assert set(got["by_checkpoint"]) == {"0", "143000"}
    assert got["by_checkpoint"]["143000"]["mean_nucleus_position"] == pytest.approx(0.05)
    assert got["by_checkpoint"]["0"]["nucleus"]["E"] < \
        got["by_checkpoint"]["143000"]["nucleus"]["E"]


def test_checkpoints_sort_numerically_not_lexically():
    dirs = [{"checkpoint": c, "layers": [_row()]} for c in (0, 2, 16, 1000, 143000)]
    keys = list(aggregate(dirs)["by_checkpoint"])
    assert keys == ["0", "2", "16", "1000", "143000"]


def test_directories_without_a_checkpoint_do_not_break_the_split():
    got = aggregate([{"checkpoint": None, "layers": [_row()]}])
    assert got["by_checkpoint"] == {}
    assert got["n_units"] == 1


def test_the_two_nulls_are_merged_separately():
    """They answer different questions, so collapsing them into one number
    would hide exactly the disagreement the second null exists to surface."""
    got = aggregate([{"layers": [_row(p=0.002, within=0.6) for _ in range(10)]}])
    assert got["nucleus"]["E"] > got["nucleus_within"]["E"]
    assert got["nucleus"]["frac_below_05"] == pytest.approx(1.0)
    assert got["nucleus_within"]["frac_below_05"] == pytest.approx(0.0)


def test_both_nulls_appear_in_the_per_checkpoint_view():
    got = aggregate([{"checkpoint": 143000, "layers": [_row(p=0.01, within=0.9)]}])
    ck = got["by_checkpoint"]["143000"]
    assert "nucleus" in ck and "nucleus_within" in ck
    assert ck["nucleus"]["E"] > ck["nucleus_within"]["E"]


def test_none_position_bias_is_dropped_from_the_merge_not_scored():
    rows = [_row(p=0.5), {**_row(p=0.5), "position_bias": None,
                          "position_bias_p": None}]
    got = aggregate([{"layers": rows}])
    assert got["nucleus"]["n"] == 2
    assert got["position_bias"]["n"] == 1
