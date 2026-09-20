"""
`tools/run/backfill_hdbscan.py` — the producer that gives the 410m sweep the
density partition it was written without.

Everything here is pure: a tmp_path standing in for a run directory, and a
fixed parameter dict. The one thing that cannot be tested this way — whether
this install reproduces the pilot sweep's labels — is not faked with a
fixture; the script re-measures it on every invocation and refuses to write
when it fails, and `test_verification_is_wired_to_the_refusal` checks that the
refusal is actually reachable.
"""
import json

import numpy as np
import pytest

from tools.run.backfill_hdbscan import (
    HDBSCAN_PARAMS,
    OUT_NAME,
    REFERENCE_TOOLCHAIN,
    discover,
    labels_provenance,
    needs_backfill,
    read_labels,
    summarise,
    toolchain_divergence,
    toolchain_fingerprint,
)


# --- the parameters must not drift from the producer they imitate ---------

def test_params_match_cluster_count_sweep():
    """`HDBSCAN_PARAMS` is copied rather than imported, deliberately, so a
    later edit to `clustering.py` cannot silently change what a backfilled
    directory means. This is the test that makes the copy safe: it fails
    loudly when the two diverge, instead of the data doing so quietly."""
    import inspect

    from p1_mstate_tracking import clustering

    src = inspect.getsource(clustering.cluster_count_sweep)
    assert 'params     = {"min_cluster_size": 2, "metric": "precomputed"}' in src
    assert HDBSCAN_PARAMS == {"min_cluster_size": 2, "metric": "precomputed"}


# --- the toolchain guard ---------------------------------------------------

def test_reference_toolchain_is_the_one_clustering_py_measured():
    assert REFERENCE_TOOLCHAIN == {
        "python": "3.10", "hdbscan": "0.8.41", "scikit-learn": "1.7.2",
    }


def test_no_divergence_on_an_exact_match():
    fp = {"python": "3.10.20", "hdbscan": "0.8.41", "scikit-learn": "1.7.2"}
    assert toolchain_divergence(fp) == []


def test_python_is_compared_on_major_minor_only():
    """A patch bump must not read as a divergence; a minor bump must."""
    assert toolchain_divergence(
        {"python": "3.10.99", "hdbscan": "0.8.41", "scikit-learn": "1.7.2"}
    ) == []
    assert any("python" in d for d in toolchain_divergence(
        {"python": "3.11.0", "hdbscan": "0.8.41", "scikit-learn": "1.7.2"}
    ))


def test_the_venv_combination_diverges():
    """The exact install `clustering.py` measured as NOT reproducing the pilot
    — 45 clusters becoming 41 at one layer. It must be caught."""
    div = toolchain_divergence(
        {"python": "3.14.7", "hdbscan": "0.8.44", "scikit-learn": "1.9.0"}
    )
    assert len(div) == 3
    assert any("hdbscan: have 0.8.44" in d for d in div)


def test_a_missing_package_diverges_rather_than_passing():
    div = toolchain_divergence({"python": "3.10.20", "scikit-learn": "1.7.2"})
    assert any("hdbscan: have unknown" in d for d in div)


def test_fingerprint_reports_the_fields_the_guard_reads():
    fp = toolchain_fingerprint()
    assert set(REFERENCE_TOOLCHAIN) <= set(fp)
    assert "numpy" in fp and "executable" in fp


# --- what counts as needing a backfill ------------------------------------

def _run_dir(tmp_path, name, *, acts=True, labels=None, backfill=False):
    d = tmp_path / name
    d.mkdir()
    if acts:
        np.savez(d / "activations.npz", activations=np.zeros((2, 3, 4), dtype=np.float32))
    if labels is not None:
        (d / "hdbscan_labels.json").write_text(json.dumps(labels))
    if backfill:
        (d / OUT_NAME).write_text(json.dumps({"labels": {"0": [1, 1, -1]}}))
    return d


def test_the_four_states(tmp_path):
    """The first case is the actual state of all 152 sweep directories."""
    assert needs_backfill(_run_dir(tmp_path, "empty", labels={})) is True
    assert needs_backfill(_run_dir(tmp_path, "absent")) is True
    assert needs_backfill(_run_dir(tmp_path, "native", labels={"0": [1, -1, 1]})) is False
    assert needs_backfill(_run_dir(tmp_path, "done", labels={}, backfill=True)) is False


def test_a_directory_without_activations_is_not_a_target(tmp_path):
    """It cannot be backfilled from anything on disk, so it must not be
    counted as pending and then silently skipped."""
    assert needs_backfill(_run_dir(tmp_path, "noacts", acts=False, labels={})) is False


def test_corrupt_labels_file_needs_a_backfill(tmp_path):
    d = _run_dir(tmp_path, "corrupt")
    (d / "hdbscan_labels.json").write_text("{not json")
    assert needs_backfill(d) is True


def test_discover_walks_timestamp_then_model_prompt(tmp_path):
    ts = tmp_path / "2026-09-01_00-00-00"
    ts.mkdir()
    _run_dir(ts, "pythia-410m-step0_wiki", labels={})
    _run_dir(ts, "pythia-410m-step1_wiki", labels={"0": [1]})   # native, skip
    _run_dir(ts, "gpt2-large_wiki", labels={})                  # wrong model
    got = discover(tmp_path, "pythia-410m-*")
    assert [d.name for d in got] == ["pythia-410m-step0_wiki"]


# --- the reader's precedence ----------------------------------------------

def test_backfill_wins_over_the_empty_canonical_file(tmp_path):
    """Backfill-first is the whole point: the canonical file in a backfilled
    directory is the EMPTY one that made the backfill necessary."""
    d = _run_dir(tmp_path, "d", labels={}, backfill=True)
    got = read_labels(d)
    assert set(got) == {0}
    assert np.array_equal(got[0], np.array([1, 1, -1], dtype=np.int32))


def test_backfill_wins_even_over_a_populated_canonical_file(tmp_path):
    d = _run_dir(tmp_path, "d", labels={"0": [7, 7, 7]}, backfill=True)
    assert np.array_equal(read_labels(d)[0], np.array([1, 1, -1], dtype=np.int32))


def test_native_labels_are_read_when_there_is_no_backfill(tmp_path):
    d = _run_dir(tmp_path, "d", labels={"0": [3, -1], "1": [-1, -1]})
    got = read_labels(d)
    assert set(got) == {0, 1}
    assert got[0].dtype == np.int32


def test_layer_keys_come_back_as_ints_from_both_routes(tmp_path):
    a = read_labels(_run_dir(tmp_path, "a", labels={"12": [1, -1]}))
    b = read_labels(_run_dir(tmp_path, "b", labels={}, backfill=True))
    assert all(isinstance(k, int) for k in list(a) + list(b))


def test_missing_and_empty_both_read_as_no_partition(tmp_path):
    assert read_labels(_run_dir(tmp_path, "a")) == {}
    assert read_labels(_run_dir(tmp_path, "b", labels={})) == {}


def test_corrupt_canonical_file_reads_as_no_partition_not_a_crash(tmp_path):
    d = _run_dir(tmp_path, "d")
    (d / "hdbscan_labels.json").write_text("{not json")
    assert read_labels(d) == {}


def test_provenance_names_the_route(tmp_path):
    assert labels_provenance(_run_dir(tmp_path, "a", labels={}, backfill=True)) == "backfill"
    assert labels_provenance(_run_dir(tmp_path, "b", labels={"0": [1]})) == "native"
    assert labels_provenance(_run_dir(tmp_path, "c", labels={})) == "absent"
    assert labels_provenance(_run_dir(tmp_path, "d")) == "absent"


# --- the summary ----------------------------------------------------------

def test_summarise_matches_clustering_jsons_two_fields():
    labels = {0: np.array([0, 0, 1, 1, -1, -1, -1, 2], dtype=np.int32)}
    got = summarise(labels)["0"]
    assert got["n_clusters"] == 3
    assert got["noise_count"] == 3
    assert got["noise_fraction"] == pytest.approx(0.375)


def test_summarise_counts_an_all_noise_layer_as_zero_clusters():
    got = summarise({4: np.full(6, -1, dtype=np.int32)})["4"]
    assert got["n_clusters"] == 0
    assert got["noise_fraction"] == pytest.approx(1.0)


def test_summarise_keys_are_strings_for_json_round_trip():
    got = summarise({0: np.zeros(3, dtype=np.int32)})
    assert json.loads(json.dumps(got)) == got


# --- the refusal path is reachable ----------------------------------------

def test_verification_is_wired_to_the_refusal():
    """The script must refuse on a dirty verification rather than warn. Checked
    on the source, because the live path needs the pilot volume mounted and a
    test that quietly skips when it is not would assert nothing."""
    import inspect

    import tools.run.backfill_hdbscan as mod

    src = inspect.getsource(mod.main)
    assert 'if not verification["clean"]' in src
    assert src.count("raise SystemExit") >= 3
