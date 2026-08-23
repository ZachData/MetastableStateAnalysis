"""
tests/test_phase1_cluster_methods.py — pure-logic tests for the
multi-method cluster comparison (visualization/cluster_methods.py) and the
loaders that feed it. No matplotlib rendering, no torch, no model loading;
the only disk touched is a tmp_path run directory holding a few small JSON
files, so this runs in the stub-heavy non-smoke session.

The trap these tests exist to guard is the threshold key grammar:
clustering.json stores the agglomerative sweep keyed by float, but JSON
stringifies dict keys, so the middle threshold comes back as
"0.30000000000000004" and cannot be looked up by value. Anything that
regresses to an exact-key lookup silently drops the agglomerative series
from every figure without raising.
"""
import json

import numpy as np
import pytest

from p1_mstate_tracking.visualization.cluster_methods import (
    noise_as_singletons, pairwise_agreement, common_layers,
    co_association, consensus_order, consensus_strength,
    scale_plateau_width, plateau_widths, noise_audit,
    bipartition_separation, cluster_count_table, kmeans_trust,
    KMEANS_SIL_MIN, KMEANS_RANK_MIN,
)
from p1_mstate_tracking.visualization.loaders import (
    _agglom_threshold_counts, _labels_by_prefix, _fiedler_bipartition,
)

# Tier: deps -- needs the heavy tier importable (torch / transformers /
# scikit-learn / matplotlib). No model download, no run artifacts.
# Measured, not assumed; see pyproject.toml markers.
pytestmark = pytest.mark.deps


# ─────────────────────────────────────────────────────────────────────────────
# Noise policy
# ─────────────────────────────────────────────────────────────────────────────

class TestNoisePolicy:

    def test_singletons_get_unique_ids_above_existing_max(self):
        labels = np.array([0, 1, -1, -1, 1])
        out = noise_as_singletons(labels)
        assert (out[[0, 1, 4]] == [0, 1, 1]).all()
        assert out[2] != out[3]
        assert out[2] > 1 and out[3] > 1

    def test_all_noise_still_produces_distinct_ids(self):
        out = noise_as_singletons(np.array([-1, -1, -1]))
        assert len(set(out.tolist())) == 3

    def test_no_noise_is_a_passthrough(self):
        labels = np.array([2, 2, 5])
        assert (noise_as_singletons(labels) == labels).all()

    def test_original_array_not_mutated(self):
        labels = np.array([0, -1])
        noise_as_singletons(labels)
        assert labels[1] == -1


class TestPairwiseAgreement:

    def test_identical_partitions_score_one(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        assert pairwise_agreement(a, a)["ari"] == pytest.approx(1.0)

    def test_relabeling_does_not_change_agreement(self):
        a = np.array([0, 0, 1, 1])
        b = np.array([7, 7, 3, 3])
        assert pairwise_agreement(a, b)["ari"] == pytest.approx(1.0)

    def test_policies_disagree_when_noise_is_present(self):
        """HDBSCAN refusing to assign is a real difference under
        'singleton' and invisible under 'exclude' — the two policies must
        not silently collapse to the same number."""
        hdb = np.array([0, 0, -1, -1, 1, 1])
        km = np.array([0, 0, 0, 0, 1, 1])
        singleton = pairwise_agreement(hdb, km, "singleton")["ari"]
        excluded = pairwise_agreement(hdb, km, "exclude")["ari"]
        assert excluded == pytest.approx(1.0)
        assert singleton < excluded

    def test_exclude_reports_shrunken_n_used(self):
        hdb = np.array([0, -1, -1, 1])
        other = np.array([0, 0, 1, 1])
        assert pairwise_agreement(hdb, other, "exclude")["n_used"] == 2
        assert pairwise_agreement(hdb, other, "singleton")["n_used"] == 4

    def test_degenerate_overlap_returns_nan_not_a_score(self):
        hdb = np.array([-1, -1, -1, 0])
        other = np.array([0, 0, 0, 0])
        res = pairwise_agreement(hdb, other, "exclude")
        assert res["n_used"] == 1
        assert np.isnan(res["ari"])

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            pairwise_agreement(np.array([0, 1]), np.array([0, 1]), "nearest")


class TestCommonLayers:

    def test_intersection_only(self):
        per = {
            "a": {0: np.zeros(4), 1: np.zeros(4)},
            "b": {1: np.zeros(4), 2: np.zeros(4)},
        }
        assert common_layers(per) == [1]

    def test_token_count_mismatch_drops_the_layer(self):
        """A layer where one method wrote a different-length array is a
        broken run, not something to compare by truncation."""
        per = {
            "a": {0: np.zeros(4), 1: np.zeros(4)},
            "b": {0: np.zeros(4), 1: np.zeros(7)},
        }
        assert common_layers(per) == [0]

    def test_empty_input(self):
        assert common_layers({}) == []


# ─────────────────────────────────────────────────────────────────────────────
# Consensus
# ─────────────────────────────────────────────────────────────────────────────

class TestCoAssociation:

    def test_unanimous_partitions_give_a_binary_matrix(self):
        a = np.array([0, 0, 1, 1])
        C = co_association([a, a, a])
        assert C[0, 1] == pytest.approx(1.0)
        assert C[0, 2] == pytest.approx(0.0)
        assert np.allclose(np.diag(C), 1.0)

    def test_split_vote_lands_at_the_fraction(self):
        a = np.array([0, 0, 1, 1])
        b = np.array([0, 1, 0, 1])
        C = co_association([a, b])
        assert C[0, 1] == pytest.approx(0.5)

    def test_symmetric(self):
        rng = np.random.default_rng(0)
        arrays = [rng.integers(0, 3, size=8) for _ in range(3)]
        C = co_association(arrays)
        assert np.allclose(C, C.T)

    def test_exclude_policy_uses_a_per_pair_denominator(self):
        """Under 'exclude' a noise token contributes no vote, so a pair
        judged by one partition out of two must score 1.0, not 0.5."""
        a = np.array([-1, -1, 0])
        b = np.array([0, 0, 0])
        C = co_association([a, b], noise_policy="exclude")
        assert C[0, 1] == pytest.approx(1.0)

    def test_mismatched_length_partition_is_skipped(self):
        a = np.array([0, 0, 1])
        C = co_association([a, np.array([0, 0])])
        assert C.shape == (3, 3)
        assert C[0, 1] == pytest.approx(1.0)

    def test_empty_input(self):
        assert co_association([]).shape == (0, 0)


class TestConsensusOrdering:

    def test_order_is_a_permutation(self):
        rng = np.random.default_rng(1)
        arrays = [rng.integers(0, 4, size=12) for _ in range(4)]
        C = co_association(arrays)
        order = consensus_order(C)
        assert sorted(order.tolist()) == list(range(12))

    def test_tiny_matrix_falls_back_to_identity(self):
        C = np.ones((2, 2))
        assert (consensus_order(C) == np.arange(2)).all()

    def test_strength_is_one_when_methods_are_unanimous(self):
        a = np.array([0, 0, 1, 1])
        assert consensus_strength(co_association([a, a])) == pytest.approx(1.0)

    def test_strength_falls_when_methods_split(self):
        a = np.array([0, 0, 1, 1])
        b = np.array([0, 1, 0, 1])
        assert consensus_strength(co_association([a, b])) < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

class TestPlateauWidth:

    def test_longest_constant_run(self):
        assert scale_plateau_width(np.array([5, 5, 5, 3, 3])) == 3

    def test_nan_breaks_a_run_rather_than_extending_it(self):
        assert scale_plateau_width(np.array([4, 4, np.nan, 4, 4])) == 2

    def test_no_repeats_gives_width_one(self):
        assert scale_plateau_width(np.array([1, 2, 3])) == 1

    def test_all_nan_gives_zero(self):
        assert scale_plateau_width(np.array([np.nan, np.nan])) == 0

    def test_applied_down_columns(self):
        counts = np.array([[2, 9], [2, 8], [2, 7]], dtype=float)
        assert plateau_widths(counts).tolist() == [3, 1]

    def test_empty_sweep(self):
        assert plateau_widths(np.zeros((0, 0))).size == 0


# ─────────────────────────────────────────────────────────────────────────────
# Noise audit
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseAudit:

    def test_noise_absorbed_into_an_existing_cluster(self):
        hdb = np.array([0, 0, 0, 0, -1, -1])
        other = np.zeros(6, dtype=int)
        res = noise_audit(hdb, other, min_cluster_size=4)
        assert res["n_noise"] == 2
        assert res["rescued_fraction"] == pytest.approx(1.0)
        assert res["rescued_into_shared"] == pytest.approx(1.0)

    def test_noise_pooled_into_its_own_group_is_rescued_but_not_shared(self):
        """The informative case: the other method finds real structure in
        the tokens HDBSCAN discarded, structure that does not overlap what
        HDBSCAN did assign."""
        hdb = np.array([0, 0, 0, 0, -1, -1, -1, -1])
        other = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        res = noise_audit(hdb, other, min_cluster_size=4)
        assert res["rescued_fraction"] == pytest.approx(1.0)
        assert res["rescued_into_shared"] == pytest.approx(0.0)

    def test_singleton_landing_places_do_not_count_as_rescue(self):
        hdb = np.array([0, 0, -1, -1])
        other = np.array([0, 0, 1, 2])
        res = noise_audit(hdb, other, min_cluster_size=4)
        assert res["rescued_fraction"] == pytest.approx(0.0)

    def test_no_noise_returns_nan_rescue(self):
        res = noise_audit(np.array([0, 0, 1]), np.array([0, 0, 1]))
        assert res["n_noise"] == 0
        assert np.isnan(res["rescued_fraction"])

    def test_length_mismatch_is_survivable(self):
        res = noise_audit(np.array([-1, 0, 0]), np.array([0, 0]))
        assert res["n_noise"] == 1
        assert np.isnan(res["rescued_fraction"])


# ─────────────────────────────────────────────────────────────────────────────
# Fiedler sharpness
# ─────────────────────────────────────────────────────────────────────────────

class TestBipartitionSeparation:

    def test_two_lobes_separate_strongly(self):
        v = np.concatenate([np.full(10, -1.0), np.full(10, 1.0)])
        v += np.random.default_rng(0).normal(0, 0.01, size=20)
        stats = bipartition_separation(v)
        assert stats["separation"] > 20
        assert stats["balance"] == pytest.approx(0.5)
        assert stats["near_zero"] == pytest.approx(0.0)

    def test_unimodal_values_are_flagged_by_near_zero(self):
        v = np.random.default_rng(0).normal(0, 1, size=400)
        stats = bipartition_separation(v)
        assert stats["near_zero"] > 0.02
        assert stats["separation"] < 5

    def test_lopsided_split_shows_in_balance(self):
        v = np.concatenate([np.full(19, -1.0), np.array([2.0])])
        assert bipartition_separation(v)["balance"] == pytest.approx(0.05)

    def test_too_short_returns_nans(self):
        stats = bipartition_separation(np.array([0.5]))
        assert all(np.isnan(x) for x in stats.values())


# ─────────────────────────────────────────────────────────────────────────────
# Loaders + count table (JSON key grammar)
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = [0.05, 0.15, 0.30000000000000004, 0.45]
# Chosen so the *middle* threshold (index len//2 = 2) yields 3 clusters,
# matching hdbscan/spectral/sinkhorn below — the agreement band should
# then be governed entirely by whether KMeans is trusted.
AGG_COUNTS = [10, 6, 3, 2]


def _write_run(tmp_path, sil=0.5, rank=30.0):
    """Minimal run_dir: the three JSON files the count table reads."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    head = {"model": "m", "prompt": "p", "n_layers": 2}
    clustering = {**head, "layers": [
        {
            "layer": i,
            "clustering": {
                # Float keys, exactly as json.dump renders them.
                "agglomerative": {str(t): AGG_COUNTS[idx]
                                  for idx, t in enumerate(THRESHOLDS)},
                "kmeans": {"best_k": 3, "best_silhouette": sil},
                "hdbscan": {"n_clusters": 3, "noise_count": 1, "noise_fraction": 0.1},
            },
            "nesting": {},
            "pair_agreement": {},
        }
        for i in range(2)
    ]}
    geometry = {**head, "n_tokens": 4, "d_model": 8, "tokens": ["a", "b", "c", "d"],
                "layers": [{"layer": i, "effective_rank": rank} for i in range(2)]}
    spectral = {**head, "layers": [
        {"layer": i, "k_eigengap": 3, "eigenvalues": [0.0, 0.1, 0.2],
         "eigengaps": [0.1, 0.1], "fiedler_bipartition": [1, 1, -1, -1]}
        for i in range(2)
    ]}
    sinkhorn = {**head, "layers": [
        {"layer": i, "fiedler_mean": 0.3, "sinkhorn_cluster_count_mean": 3.4}
        for i in range(2)
    ]}
    for name, payload in [("clustering.json", clustering), ("geometry.json", geometry),
                          ("spectral.json", spectral), ("sinkhorn.json", sinkhorn)]:
        (tmp_path / name).write_text(json.dumps(payload))
    return tmp_path


class TestLabelKeyParsing:

    def test_prefix_parsing(self):
        arrays = {
            "kmeans_labels_L0": np.array([1]),
            "kmeans_labels_L11": np.array([2]),
            "hdbscan_labels_L0": np.array([3]),
        }
        out = _labels_by_prefix(arrays, "kmeans_labels_")
        assert sorted(out.keys()) == [0, 11]

    def test_unparseable_keys_are_skipped_not_raised(self):
        out = _labels_by_prefix({"fiedler_Lx": np.array([1]),
                                 "fiedler_L2": np.array([2])}, "fiedler_")
        assert list(out.keys()) == [2]


class TestThresholdSweepLoader:

    def test_stringified_float_keys_round_trip(self, tmp_path):
        run = _write_run(tmp_path)
        thresholds, layers, counts = _agglom_threshold_counts(run)
        assert np.allclose(thresholds, sorted(THRESHOLDS))
        assert layers == [0, 1]
        assert counts.shape == (4, 2)
        assert not np.isnan(counts).any()

    def test_missing_file_returns_empty(self, tmp_path):
        thresholds, layers, counts = _agglom_threshold_counts(tmp_path)
        assert thresholds.size == 0 and layers == [] and counts.size == 0

    def test_mid_labels_key_is_not_treated_as_a_threshold(self, tmp_path):
        run = _write_run(tmp_path)
        payload = json.loads((run / "clustering.json").read_text())
        payload["layers"][0]["clustering"]["agglomerative"]["mid_labels"] = [0, 0, 1, 1]
        (run / "clustering.json").write_text(json.dumps(payload))
        thresholds, _, counts = _agglom_threshold_counts(run)
        assert thresholds.size == len(THRESHOLDS)


class TestCountTable:

    def test_agglomerative_series_is_populated_from_the_mid_threshold(self, tmp_path):
        """The regression this whole file exists for: an exact float-key
        lookup returns None here and the series goes silently NaN."""
        table = cluster_count_table(_write_run(tmp_path))
        agg = table["counts"]["agglomerative"]
        assert np.isfinite(agg).all()
        assert (agg == AGG_COUNTS[len(THRESHOLDS) // 2]).all()
        assert table["mid_threshold"] == pytest.approx(0.30000000000000004)

    def test_every_series_present(self, tmp_path):
        counts = cluster_count_table(_write_run(tmp_path))["counts"]
        for key in ("hdbscan", "kmeans", "agglomerative", "spectral_k", "sinkhorn_k"):
            assert np.isfinite(counts[key]).all(), key

    def test_sinkhorn_count_is_rounded(self, tmp_path):
        counts = cluster_count_table(_write_run(tmp_path))["counts"]
        assert counts["sinkhorn_k"][0] == pytest.approx(3.0)

    def test_kmeans_trust_gate_matches_reporting_thresholds(self, tmp_path):
        trusted = kmeans_trust(_write_run(tmp_path, sil=KMEANS_SIL_MIN,
                                          rank=KMEANS_RANK_MIN))
        assert all(trusted.values())
        untrusted = kmeans_trust(_write_run(tmp_path, sil=KMEANS_SIL_MIN - 0.01,
                                            rank=KMEANS_RANK_MIN))
        assert not any(untrusted.values())
        low_rank = kmeans_trust(_write_run(tmp_path, sil=0.9,
                                           rank=KMEANS_RANK_MIN - 0.1))
        assert not any(low_rank.values())

    def test_untrusted_kmeans_is_excluded_from_the_agreement_band(self, tmp_path):
        """KMeans best_k=2 is the floor of K_RANGE, not a finding — a
        wild best_k at a collapsed layer must not be able to break
        agreement. The same value at a trustworthy layer must."""
        def _with_km(sil, rank, best_k):
            run = _write_run(tmp_path / f"r{sil}{rank}{best_k}", sil=sil, rank=rank)
            payload = json.loads((run / "clustering.json").read_text())
            for lr in payload["layers"]:
                lr["clustering"]["kmeans"]["best_k"] = best_k
            (run / "clustering.json").write_text(json.dumps(payload))
            return cluster_count_table(run)

        collapsed = _with_km(0.0, 1.0, 99)
        assert not collapsed["kmeans_trusted"].any()
        assert collapsed["agreement"].all()

        healthy = _with_km(0.9, 40.0, 99)
        assert healthy["kmeans_trusted"].all()
        assert not healthy["agreement"].any()

    def test_fiedler_bipartition_maps_signs_to_binary_labels(self, tmp_path):
        run = _write_run(tmp_path)
        bip = _fiedler_bipartition(run)
        assert set(bip.keys()) == {0, 1}
        assert bip[0].tolist() == [1, 1, 0, 0]

    def test_null_bipartition_layers_are_omitted(self, tmp_path):
        run = _write_run(tmp_path)
        payload = json.loads((run / "spectral.json").read_text())
        payload["layers"][0]["fiedler_bipartition"] = None
        (run / "spectral.json").write_text(json.dumps(payload))
        assert list(_fiedler_bipartition(run).keys()) == [1]
