"""
tests/test_phase1d_vote_rules.py — the voting rules 1d's weighting
decision compares (`p1d_cluster_ensemble/vote_rules.py`).

The reference rule has to be run_1d's ensemble exactly, on run_1d's own
null draws, or every comparison against it is against something else.
The other rules are small label or weight transforms, each pinned on a
case whose answer is known by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.deps

from p1d_cluster_ensemble import ensemble
from p1d_cluster_ensemble.methods import LayerData, fit
from p1d_cluster_ensemble.run_1d import null_confidences
from p1d_cluster_ensemble.selection import selection_weights
from p1d_cluster_ensemble.vote_rules import (
    REFERENCE, _jaccard, _spearman, abstain_small, null_draw_labels,
    rule_labels, rule_record, rule_weights, summarise,
)

D = 24


def _normed(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def planted():
    rng = np.random.default_rng(0)
    X = np.concatenate([c + 0.06 * rng.normal(size=(15, D)) for c in np.eye(3, D)])
    return LayerData.from_normed(_normed(X))


def _selection(data, params, stability, null_mean):
    """The slice of a `select_family` result the rules read."""
    out = {}
    for f, p in params.items():
        out[f] = {"selected": {"params": p,
                               "stability": {"mean_ari": stability[f]},
                               "null": {"stability": {"null_mean": null_mean[f]}}},
                  "selected_labels": fit(f, p, data, seed=0).tolist()}
    return out


@pytest.fixture(scope="module")
def selection(planted):
    params = {"kmeans": {"k": 3}, "spherical_kmeans": {"k": 3},
              "agglomerative": {"linkage": "average", "threshold": 0.05}}
    return _selection(planted, params,
                      stability={"kmeans": 0.9, "spherical_kmeans": 0.8, "agglomerative": 0.95},
                      null_mean={"kmeans": 0.5, "spherical_kmeans": 1.0, "agglomerative": 0.0})


def _labels(selection):
    return {f: np.asarray(r["selected_labels"]) for f, r in selection.items()}


class TestRules:
    def test_abstain_small_refuses_clusters_below_four_and_keeps_the_rest(self):
        lab = np.array([0, 0, 0, 0, 1, 1, 1, 2, -1, 3, 3, 3, 3, 3])
        np.testing.assert_array_equal(
            abstain_small(lab), [0, 0, 0, 0, -1, -1, -1, -1, -1, 3, 3, 3, 3, 3])

    def test_noise_rules_map_to_run_1ds_policies(self):
        lab = {"a": np.array([0, 0, 1, -1])}
        assert rule_labels(lab, "current")[1] == "exclude"
        assert rule_labels(lab, "singleton")[1] == "singleton"
        got, policy = rule_labels(lab, "abstain_small")
        assert policy == "exclude" and (got["a"] == -1).all()
        with pytest.raises(ValueError):
            rule_labels(lab, "vote")

    def test_weights(self, selection):
        assert rule_weights(selection, "stability") == selection_weights(selection)
        assert set(rule_weights(selection, "uniform").values()) == {1.0}
        kappa = rule_weights(selection, "kappa")
        assert kappa["kmeans"] == pytest.approx((0.9 - 0.5) / 0.5)
        assert kappa["agglomerative"] == pytest.approx(0.95)
        # A null that is always perfectly stable leaves nothing above it.
        assert kappa["spherical_kmeans"] == 0.0

    def test_jaccard_and_spearman_edges(self):
        e = np.zeros(5, dtype=bool)
        assert _jaccard(e, e) == 1.0
        assert _jaccard(e, ~e) == 0.0
        assert _spearman(np.arange(5.0), np.arange(5.0) * 2) == pytest.approx(1.0)
        assert _spearman(np.ones(5), np.ones(5)) == 1.0
        assert _spearman(np.ones(5), np.arange(5.0)) == 0.0


class TestReferenceIsRun1d:
    def test_null_draws_are_run_1ds(self, planted, selection):
        params = {f: r["selected"]["params"] for f, r in selection.items()}
        draws = null_draw_labels(planted, params, n_draws=3, seed=0)
        ours = [ensemble.build(d, weights=selection_weights(selection))["confidence"]
                for d in draws]
        theirs = null_confidences(planted, selection, n_draws=3, seed=0)
        for a, b in zip(ours, theirs):
            np.testing.assert_allclose(a, b)

    def test_reference_rule_is_run_1ds_build(self, planted, selection):
        params = {f: r["selected"]["params"] for f, r in selection.items()}
        draws = null_draw_labels(planted, params, n_draws=3, seed=0)
        rec = rule_record(_labels(selection), draws, selection, *REFERENCE)
        built = ensemble.build(_labels(selection), weights=selection_weights(selection))
        thr = ensemble.confidence_thresholds(null_confidences(planted, selection, n_draws=3, seed=0))
        pop = ensemble.trichotomy(built["confidence"], thr)
        assert rec["n_clusters"] == built["consensus"]["n_clusters"]
        assert rec["thresholds"]["core"] == pytest.approx(thr["core"])
        assert rec["counts"]["core"] == int((pop == "core").sum())


class TestRecord:
    def test_singletons_veto_the_caps_and_abstaining_removes_the_veto(self, planted, selection):
        # agglomerative at 0.05 shatters the three caps into 29 pieces,
        # mostly singletons: 1d's L12 case in miniature.
        params = {f: r["selected"]["params"] for f, r in selection.items()}
        draws = null_draw_labels(planted, params, n_draws=2, seed=0)
        current = rule_record(_labels(selection), draws, selection, "current", "uniform")
        assert current["n_clusters"] == 3 and current["dominance_max"] == pytest.approx(1.0)
        assert set(current["sway"]) == set(selection)
        # Drop one k-means family: one "together" vote against one "apart".
        assert current["worst_consensus_ari"] < 0.5
        assert current["worst_consensus_family"] != "agglomerative"
        quiet = rule_record(_labels(selection), draws, selection, "abstain_small", "uniform")
        assert quiet["n_clusters"] == 3
        assert quiet["worst_consensus_ari"] == pytest.approx(1.0)

    def test_at_matched_scale_no_single_family_sways_the_consensus(self, planted):
        # The criterion can pass: every family at the caps' scale (k = 3).
        # #120's review: what fails on Pythia may be the scale mix, which
        # this does not show; it shows only that the drop-one test is not
        # unpassable by construction.
        params = {"kmeans": {"k": 3}, "spherical_kmeans": {"k": 3},
                  "agglomerative": {"linkage": "average", "threshold": 0.3}}
        sel = _selection(planted, params,
                         stability={f: 0.9 for f in params}, null_mean={f: 0.5 for f in params})
        draws = null_draw_labels(planted, params, n_draws=2, seed=0)
        rec = rule_record(_labels(sel), draws, sel, *REFERENCE)
        assert rec["n_clusters"] == 3
        assert rec["worst_consensus_ari"] == pytest.approx(1.0)
        assert rec["worst_confidence_spearman"] > 0.9

    def test_a_zero_weight_family_does_not_vote(self, planted, selection):
        params = {f: r["selected"]["params"] for f, r in selection.items()}
        draws = null_draw_labels(planted, params, n_draws=2, seed=0)
        rec = rule_record(_labels(selection), draws, selection, "current", "kappa")
        assert "spherical_kmeans" not in rec["sway"]
        assert rec["n_families"] == 2


def test_summary_counts_a_jaccard_of_zero_as_a_veto():
    rule = {"noise": "current", "weight": "stability", "worst_core_jaccard": 0.0,
            "worst_core_family": "kmeans", "worst_consensus_ari": 0.9,
            "worst_confidence_spearman": 0.8, "dominance_max": 0.5,
            "counts": {"core": 3}, "core_share": 0.1, "n_clusters": 2,
            "zero_support": 0.0, "ari_to_reference": 1.0}
    row, = summarise([{"step": "step143000", "rules": [rule]}])
    assert row["veto_records"] == 1 and row["empty_core_records"] == 0
