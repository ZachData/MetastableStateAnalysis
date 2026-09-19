"""
`clustering.json`'s HDBSCAN block must say whether HDBSCAN ran.

Until 2026-09-19 `_save_clustering` wrote `noise_count: 0, noise_fraction: 0.0`
whether or not HDBSCAN had produced any labels, so a run made without the
package carried a `cluster_membership` of exactly 1.0 at every layer —
indistinguishable, to CLAIM-C's gate, from a metric that was measured and came
out flat. Four of the gate's arms were produced that way before anything
noticed, because the metric that DID refuse (`cluster_count`, null) was the
only one that said so.

These are pure-tier: `p1_io` imports numpy and the standard library only.
"""

from __future__ import annotations

import json

import pytest

from p1_mstate_tracking.p1_io import _save_clustering

pytestmark = pytest.mark.pure


def _results(hdbscan_block):
    layer = {"layer": 0, "clustering": {"kmeans": {"best_k": 3,
                                                   "best_silhouette": 0.4},
                                        "agglomerative": {0.3: 5}}}
    if hdbscan_block is not None:
        layer["clustering"]["hdbscan"] = hdbscan_block
    return {"model": "m", "prompt": "p", "n_layers": 1, "n_tokens": 6,
            "layers": [layer]}


def _write(tmp_path, hdbscan_block):
    _save_clustering(_results(hdbscan_block), tmp_path)
    doc = json.loads((tmp_path / "clustering.json").read_text())
    return doc["layers"][0]["clustering"]["hdbscan"]


def test_absent_hdbscan_is_null_not_zero(tmp_path):
    got = _write(tmp_path, None)
    assert got["n_clusters"] is None
    assert got["noise_count"] is None
    # The load-bearing one: 1 - noise_fraction must not be computable.
    assert got["noise_fraction"] is None
    assert got["available"] is False
    assert "not importable" in got["reason"]


def test_a_run_with_no_noise_is_not_the_same_as_no_run(tmp_path):
    got = _write(tmp_path, {"n_clusters": 2, "labels": [0, 0, 0, 1, 1, 1],
                            "impl": "hdbscan", "version": "0.8.44",
                            "params": {"min_cluster_size": 2}})
    assert got["noise_count"] == 0 and got["noise_fraction"] == 0.0
    assert "available" not in got          # it ran; nothing to disclaim
    assert got["version"] == "0.8.44"


def test_noise_fraction_counts_only_the_noise_label(tmp_path):
    got = _write(tmp_path, {"n_clusters": 1, "labels": [-1, -1, 0, 0, 0, 0]})
    assert got["noise_count"] == 2
    assert got["noise_fraction"] == pytest.approx(2 / 6, abs=5e-5)


def test_provenance_is_carried_through_when_present(tmp_path):
    params = {"min_cluster_size": 2, "metric": "precomputed"}
    got = _write(tmp_path, {"n_clusters": 1, "labels": [0, 0, 0, 0, 0, 0],
                            "impl": "hdbscan", "version": "9.9.9",
                            "params": params})
    # These two metrics are version-dependent and the version was never
    # recorded before 2026-09-19, which is why cross-sweep comparisons of them
    # cannot be checked retrospectively.
    assert got["impl"] == "hdbscan" and got["version"] == "9.9.9"
    assert got["params"] == params
