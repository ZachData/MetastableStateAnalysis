"""
`tools/run/p10_partition_stability.py` — the measurement-reproducibility floor.

This runner reports no p-value, so the tests are about the arithmetic and about
one thing that is easy to get wrong and would invert the conclusion: reporting
a single ARI when HDBSCAN's noise convention changes the answer.
"""
import numpy as np
import pytest

from tools.run.p10_partition_stability import (
    aggregate,
    compare_directory,
    compare_labels,
)

# Tier: numpy and scipy only -- no torch, transformers, sklearn or
# matplotlib -- so this runs in `scripts/check.sh pure`. Declared, not
# assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure


def test_identical_partitions_score_one_both_ways():
    lab = np.array([0, 0, 1, 1, -1, -1, 2])
    got = compare_labels(lab, lab)
    assert got["ari"] == pytest.approx(1.0)
    assert got["ari_noise_dropped"] == pytest.approx(1.0)
    assert got["identical"] is True


def test_a_relabelling_is_the_same_partition():
    """ARI is invariant to which integer names which cluster, which is why it
    is the right instrument here -- HDBSCAN's ids are arbitrary."""
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([2, 2, 0, 0, 1, 1])
    got = compare_labels(a, b)
    assert got["ari"] == pytest.approx(1.0)
    assert got["identical"] is False       # same partition, different names


def test_the_two_noise_conventions_disagree_and_both_are_reported():
    """The hazard `notes-10.md` §4.4 names. Here the clustered structure is
    identical and only the noise assignment moves: treating noise as one
    cluster sees a difference, dropping it sees none."""
    a = np.array([0, 0, 1, 1, -1, -1, -1, -1])
    b = np.array([0, 0, 1, 1, -1, -1, 2, 2])
    got = compare_labels(a, b)
    assert got["ari_noise_dropped"] == pytest.approx(1.0)
    assert got["ari"] < 0.95


def test_counts_and_noise_fractions_are_recorded_per_side():
    a = np.array([0, 0, 1, 1, -1, -1, -1, -1])
    b = np.array([0, 0, 1, 1, -1, -1, 2, 2])
    got = compare_labels(a, b)
    assert (got["n_clusters_a"], got["n_clusters_b"]) == (2, 3)
    assert got["noise_fraction_a"] == pytest.approx(0.5)
    assert got["noise_fraction_b"] == pytest.approx(0.25)


def test_unrelated_partitions_score_near_zero():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 10, size=500)
    b = rng.integers(0, 10, size=500)
    assert abs(compare_labels(a, b)["ari"]) < 0.05


def test_mismatched_lengths_are_refused_rather_than_truncated():
    with pytest.raises(ValueError, match="differ in length"):
        compare_labels(np.zeros(5, dtype=int), np.zeros(4, dtype=int))


# --- the directory comparison ---------------------------------------------

def _write(tmp_path, name, labels, acts=None, tokens=None):
    import json

    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "hdbscan_labels.json").write_text(json.dumps(labels))
    if acts is not None:
        np.savez(d / "activations.npz", activations=np.asarray(acts, dtype=np.float32))
    if tokens is not None:
        (d / "tokens.txt").write_text(tokens)
    return d


def test_only_layers_present_on_both_sides_are_compared(tmp_path):
    a = _write(tmp_path / "a", "pythia-410m-step7_wiki",
               {"0": [0, 0, 1, 1], "1": [0, 1, 0, 1], "2": [0, 0, 0, 1]})
    b = _write(tmp_path / "b", "pythia-410m-step7_wiki",
               {"0": [0, 0, 1, 1], "1": [1, 0, 1, 0]})
    got = compare_directory(a, b, with_activations=False)
    assert [l["layer"] for l in got["layers"]] == [0, 1]
    assert got["checkpoint"] == 7


def test_activation_divergence_is_recorded_beside_the_label_divergence(tmp_path):
    """The input difference has to be visible next to the output difference,
    or a reader cannot tell a statement about HDBSCAN from one about the
    forward pass."""
    base = np.random.default_rng(1).normal(size=(2, 6, 4))
    a = _write(tmp_path / "a", "pythia-410m-step7_wiki",
               {"0": [0, 0, 1, 1, -1, -1]}, acts=base, tokens="x\ny\n")
    b = _write(tmp_path / "b", "pythia-410m-step7_wiki",
               {"0": [0, 0, 1, 1, -1, -1]}, acts=base + 3e-7, tokens="x\ny\n")
    got = compare_directory(a, b)
    assert got["activation_max_abs_diff"] < 1e-5
    assert got["tokens_identical"] is True


def test_differing_shapes_do_not_crash_the_divergence(tmp_path):
    a = _write(tmp_path / "a", "pythia-410m-step7_wiki", {"0": [0, 0, 1, 1]},
               acts=np.zeros((2, 4, 3)))
    b = _write(tmp_path / "b", "pythia-410m-step7_wiki", {"0": [0, 0, 1, 1]},
               acts=np.zeros((2, 4, 5)))
    got = compare_directory(a, b)
    assert "activation_max_abs_diff" not in got
    assert len(got["layers"]) == 1


# --- the aggregate ---------------------------------------------------------

def _row(ari=1.0, arinn=1.0, ident=True, ka=10, kb=10, na=0.4, nb=0.4):
    return {"layer": 0, "ari": ari, "ari_noise_dropped": arinn, "identical": ident,
            "n_clusters_a": ka, "n_clusters_b": kb,
            "noise_fraction_a": na, "noise_fraction_b": nb}


def test_aggregate_of_nothing_says_so():
    assert aggregate([{"run_dir": "x", "layers": []}]) == {"n": 0}


def test_the_tail_is_reported_not_just_the_mean():
    """A mean ARI of 0.94 with a 5th percentile of 0.36 is a different fact
    from a mean of 0.94 with a 5th percentile of 0.90, and only the second is
    'stable'. The summary has to carry the tail."""
    rows = [_row(ari=1.0) for _ in range(90)] + [_row(ari=0.2, ident=False)
                                                 for _ in range(10)]
    got = aggregate([{"checkpoint": 0, "layers": rows}])
    assert got["ari_mean"] == pytest.approx(0.92)
    assert got["ari_p05"] == pytest.approx(0.2)
    assert got["ari_min"] == pytest.approx(0.2)
    assert got["frac_identical"] == pytest.approx(0.90)


def test_cluster_count_delta_reports_the_absolute_maximum():
    """Deltas that cancel in the mean are the ones that matter, so the
    absolute maximum is reported rather than the signed mean alone."""
    rows = [_row(ka=10, kb=26), _row(ka=10, kb=-6 + 10)]
    got = aggregate([{"checkpoint": 0, "layers": rows}])
    assert got["cluster_count_abs_delta_max"] == 16
    assert got["cluster_count_abs_delta_mean"] == pytest.approx(11.0)


def test_both_ari_conventions_survive_into_the_summary():
    rows = [_row(ari=0.5, arinn=0.9) for _ in range(10)]
    got = aggregate([{"checkpoint": 0, "layers": rows}])
    assert got["ari_mean"] == pytest.approx(0.5)
    assert got["ari_noise_dropped_mean"] == pytest.approx(0.9)


def test_aggregate_splits_by_checkpoint():
    got = aggregate([{"checkpoint": 0, "layers": [_row()]},
                     {"checkpoint": 143000, "layers": [_row(ari=0.3, ident=False)]}])
    assert list(got["by_checkpoint"]) == ["0", "143000"]
    assert got["by_checkpoint"]["143000"]["ari_mean"] == pytest.approx(0.3)


def test_tokens_identical_is_an_all_not_an_any():
    got = aggregate([{"checkpoint": 0, "layers": [_row()], "tokens_identical": True},
                     {"checkpoint": 1, "layers": [_row()], "tokens_identical": False}])
    assert got["all_tokens_identical"] is False
