"""
tools/audit_p1c_inputs.py reads artifact KEYS and says what Phase 1c's four
gates can be fed. The findings it produced on the tree — no `beta_eff`
anywhere, no `kmeans_centroids_L*` anywhere, the two P-S1 arms disagreeing on
cluster count — are only worth committing if the reader is right about which
key means which verdict, so these fixtures are synthetic directories whose
answer is known.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools.audit_p1c_inputs import (
    _cluster_kinds,
    arm_agreement,
    inspect_one,
    verdicts,
)

pytestmark = pytest.mark.pure


def _make_dir(root, model, prompt, *, labels=None, centroids=False,
              beta=None, norms=True, attentions=True, hdbscan_json=None):
    d = root / f"{model}_{prompt}"
    d.mkdir(parents=True)
    geo = {"model": model, "prompt": prompt, "n_tokens": 6, "d_model": 4,
           "n_layers": 3, "sublayer_semantics": None}
    if beta is not None:
        geo["beta_eff"] = beta
    (d / "geometry.json").write_text(json.dumps(geo))
    arrays = {"activations": np.zeros((3, 6, 4))}
    if norms:
        arrays["norms"] = np.ones((3, 6))
    np.savez(d / "activations.npz", **arrays)
    if attentions:
        np.savez(d / "attentions.npz", attentions=np.zeros((2, 2, 6, 6)))
    cz = {}
    for layer, labs in (labels or {}).items():
        cz[f"kmeans_labels_L{layer}"] = np.asarray(labs)
        cz[f"agglom_mid_labels_L{layer}"] = np.asarray(labs)
        if centroids:
            cz[f"kmeans_centroids_L{layer}"] = np.zeros((len(set(labs)), 4))
    np.savez(d / "clusters.npz", **cz)
    if hdbscan_json is not None:
        (d / "hdbscan_labels.json").write_text(json.dumps(hdbscan_json))
    return d


def test_cluster_kinds_splits_family_from_layer_index():
    kinds = _cluster_kinds(["kmeans_labels_L0", "kmeans_labels_L12",
                            "agglom_mid_labels_L0", "pair_agreement"])
    assert kinds == {"kmeans_labels": [0, 12], "agglom_mid_labels": [0]}


def test_inspect_reads_the_keys_that_decide_each_verdict(tmp_path):
    d = _make_dir(tmp_path, "m", "p", labels={0: [0, 0, 1, 1, 2, 2]},
                  hdbscan_json={})
    got = inspect_one(d)
    assert got["norms"] and got["activations"] and got["attentions"]
    assert got["beta_eff"] is None and got["beta_eff_per_head"] is False
    assert got["cluster_kinds"] == {"kmeans_labels": [0],
                                    "agglom_mid_labels": [0]}
    # An empty hdbscan_labels.json is the shape the runs on disk have, and it
    # is not the same thing as the file being absent.
    assert got["hdbscan_labels_json_empty"] is True


def test_missing_beta_blocks_A_and_B_but_not_E(tmp_path):
    d = _make_dir(tmp_path, "m", "p", labels={0: [0, 1, 1, 1, 2, 2]})
    v = verdicts([inspect_one(d)])
    assert v["A"]["runnable"] is False and v["B"]["runnable"] is False
    assert "beta_eff" in v["A"]["blocked_by"]
    assert v["E"]["runnable"] is True          # E reads activations only


def test_E_is_blocked_only_when_the_raw_stream_is_missing(tmp_path):
    d = _make_dir(tmp_path, "m", "p", labels={0: [0, 1]}, norms=False)
    assert verdicts([inspect_one(d)])["E"]["runnable"] is False


def test_F_is_runnable_only_with_persisted_kmeans_centroids(tmp_path):
    without = _make_dir(tmp_path / "a", "m", "p", labels={0: [0, 0, 1, 1, 2, 2]})
    assert verdicts([inspect_one(without)])["F"]["runnable"] is False
    with_c = _make_dir(tmp_path / "b", "m", "p", labels={0: [0, 0, 1, 1, 2, 2]},
                       centroids=True)
    assert verdicts([inspect_one(with_c)])["F"]["runnable"] is True


def test_arm_agreement_counts_rows_and_honours_the_control_exclusion(tmp_path):
    trained, step0 = tmp_path / "trained", tmp_path / "step0"
    trained.mkdir(), step0.mkdir()
    # wiki: layer 0 agrees (2 clusters each), layer 1 does not.
    _make_dir(trained, "m", "wiki", labels={0: [0, 0, 1, 1], 1: [0, 1, 2, 3]})
    _make_dir(step0, "m0", "wiki", labels={0: [0, 1, 1, 1], 1: [0, 0, 1, 1]})
    # repeated_tokens agrees on both layers, and is excluded by default.
    _make_dir(trained, "m", "repeated_tokens", labels={0: [0, 1], 1: [0, 1]})
    _make_dir(step0, "m0", "repeated_tokens", labels={0: [0, 1], 1: [0, 1]})

    got = arm_agreement(trained, step0)
    km = got["methods"]["kmeans_labels"]
    assert km["layer_rows"] == 4 and km["same_m"] == 3
    assert km["layer_rows_excluding_controls"] == 2
    assert km["same_m_excluding_controls"] == 1


def test_noise_labels_do_not_count_as_a_cluster(tmp_path):
    trained, step0 = tmp_path / "t", tmp_path / "s"
    trained.mkdir(), step0.mkdir()
    _make_dir(trained, "m", "wiki", labels={0: [0, 0, 1, 1]})
    _make_dir(step0, "m0", "wiki", labels={0: [-1, 0, 1, 1]})
    km = arm_agreement(trained, step0)["methods"]["kmeans_labels"]
    assert km["layer_rows"] == 1 and km["same_m"] == 1
