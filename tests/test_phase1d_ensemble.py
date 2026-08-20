"""
tests/test_phase1d_ensemble.py — the consensus, the comparisons against
Phase 1's shipped partition, artifact IO, and one end-to-end driver run.

Two of these tests exist to kill a duplication rather than to check a
computation. `p1d_cluster_ensemble.ensemble.co_association` and
`p1d_io.phase1_agreement_layers` re-implement logic that already lives in
p1_mstate_tracking/visualization/cluster_methods.py, because that package's
__init__ imports the whole figure pipeline and this phase must stay
importable without matplotlib. The equivalence assertions below are the
mechanism that keeps the two from drifting; they skip where the
visualization package cannot be imported, and fail — rather than
silently diverging — where it can.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sklearn.metrics import adjusted_rand_score, roc_auc_score

from p1d_cluster_ensemble import comparison, ensemble
from p1d_cluster_ensemble.constants import DISTANCE_THRESHOLDS
from p1d_cluster_ensemble.p1d_io import (
    build_particle_table, load_run, phase1_agreement_layers, run_identity,
    save_p1d,
)

TRUTH = np.repeat([0, 1, 2], 10)


# ---------------------------------------------------------------------------
# Co-association
# ---------------------------------------------------------------------------

class TestCoAssociation:

    def test_unanimous_methods_give_a_block_matrix(self):
        out = ensemble.co_association({f"m{i}": TRUTH.copy() for i in range(4)})
        C = out["C"]
        same = TRUTH[:, None] == TRUTH[None, :]
        assert np.allclose(C[same], 1.0)
        assert np.allclose(C[~same], 0.0)
        assert ensemble.consensus_strength(C) == pytest.approx(1.0)

    def test_a_dissenting_method_moves_the_entries_by_its_weight_share(self):
        agree, dissent = TRUTH.copy(), np.zeros_like(TRUTH)
        out = ensemble.co_association({"a": agree, "b": dissent},
                                      weights={"a": 3.0, "b": 1.0})
        # tokens 0 and 10 are in different clusters for "a", the same for
        # "b": a 1-of-4 weight share.
        assert out["C"][0, 10] == pytest.approx(0.25)

    def test_excluded_refusals_lower_the_support_not_the_agreement(self):
        refuser = TRUTH.copy()
        refuser[:3] = -1
        out = ensemble.co_association({"a": TRUTH.copy(), "b": refuser},
                                      noise_policy="exclude")
        assert out["support"][0, 1] == pytest.approx(1.0)   # only "a" had an opinion
        assert out["support"][5, 6] == pytest.approx(2.0)
        assert out["C"][0, 1] == pytest.approx(1.0)

    def test_singleton_policy_makes_refusals_disagree_instead(self):
        refuser = TRUTH.copy()
        refuser[:3] = -1
        out = ensemble.co_association({"a": TRUTH.copy(), "b": refuser},
                                      noise_policy="singleton")
        assert out["support"][0, 1] == pytest.approx(2.0)
        assert out["C"][0, 1] == pytest.approx(0.5)

    def test_negative_weights_are_refused(self):
        with pytest.raises(ValueError):
            ensemble.co_association({"a": TRUTH.copy()}, weights={"a": -1.0})

    def test_matches_the_visualization_implementation_at_unit_weights(self):
        cm = pytest.importorskip(
            "p1_mstate_tracking.visualization.cluster_methods",
            reason="visualization package needs matplotlib",
        )
        rng = np.random.default_rng(0)
        partitions = [rng.integers(0, 4, 30) for _ in range(3)]
        partitions[0][:4] = -1
        for policy in ("singleton", "exclude"):
            mine = ensemble.co_association(
                {f"m{i}": p for i, p in enumerate(partitions)}, noise_policy=policy)["C"]
            theirs = cm.co_association(partitions, noise_policy=policy)
            assert np.allclose(mine, theirs), policy
            assert ensemble.consensus_strength(mine) == pytest.approx(
                cm.consensus_strength(theirs))

    def test_noise_as_singletons_matches_the_visualization_implementation(self):
        cm = pytest.importorskip(
            "p1_mstate_tracking.visualization.cluster_methods",
            reason="visualization package needs matplotlib",
        )
        labels = np.array([0, -1, 2, -1, 2])
        assert (ensemble.noise_as_singletons(labels)
                == cm.noise_as_singletons(labels)).all()


# ---------------------------------------------------------------------------
# Consensus partition
# ---------------------------------------------------------------------------

class TestConsensusPartition:

    def test_recovers_the_partition_a_clean_matrix_encodes(self):
        C = (TRUTH[:, None] == TRUTH[None, :]).astype(float)
        out = ensemble.consensus_partition(C)
        assert out["n_clusters"] == 3
        assert adjusted_rand_score(TRUTH, out["labels"]) == pytest.approx(1.0)
        assert out["objective"] == pytest.approx(0.0)
        assert out["branch"] == "mirkin_cut"

    def test_the_chosen_cut_minimizes_the_objective_over_the_curve(self):
        rng = np.random.default_rng(1)
        base = (TRUTH[:, None] == TRUTH[None, :]).astype(float)
        C = np.clip(base + 0.15 * rng.normal(size=base.shape), 0, 1)
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 1.0)
        out = ensemble.consensus_partition(C)
        assert out["objective"] <= min(j for _, j in out["objective_curve"]) + 1e-12

    def test_k_is_derived_not_supplied(self):
        # Five blocks in, five blocks out, with nothing telling it five.
        labels = np.repeat(np.arange(5), 6)
        C = (labels[:, None] == labels[None, :]).astype(float)
        assert ensemble.consensus_partition(C)["n_clusters"] == 5

    def test_degenerate_sizes_report_their_branch(self):
        assert ensemble.consensus_partition(np.zeros((0, 0)))["branch"] == "empty"
        assert ensemble.consensus_partition(np.ones((2, 2)))["branch"] == "n<3"


# ---------------------------------------------------------------------------
# Per-particle annotations
# ---------------------------------------------------------------------------

class TestPerParticle:

    def test_confidence_is_one_when_every_method_agrees(self):
        C = (TRUTH[:, None] == TRUTH[None, :]).astype(float)
        conf = ensemble.confidence(C, TRUTH)
        assert np.allclose(conf, 1.0)

    def test_confidence_falls_toward_zero_for_a_contested_particle(self):
        C = (TRUTH[:, None] == TRUTH[None, :]).astype(float)
        # token 0 is claimed half the time by each of two clusters
        C[0, TRUTH == 0] = 0.5
        C[TRUTH == 0, 0] = 0.5
        C[0, TRUTH == 1] = 0.5
        C[TRUTH == 1, 0] = 0.5
        C[0, 0] = 1.0
        conf = ensemble.confidence(C, TRUTH)
        assert abs(conf[0]) < 0.1
        assert conf[5] > 0.9

    def test_a_singleton_gets_no_within_cluster_credit(self):
        labels = np.array([0, 1, 1, 1])
        C = np.array([
            [1.0, 0.6, 0.6, 0.6],
            [0.6, 1.0, 1.0, 1.0],
            [0.6, 1.0, 1.0, 1.0],
            [0.6, 1.0, 1.0, 1.0],
        ])
        conf = ensemble.confidence(C, labels)
        assert conf[0] == pytest.approx(-0.6)

    def test_recall_is_one_when_a_family_reproduces_the_consensus(self):
        out = ensemble.consensus_recall({"a": TRUTH.copy(), "b": TRUTH.copy()}, TRUTH)
        assert np.allclose(out["mean_recall"], 1.0)
        assert np.allclose(out["min_recall"], 1.0)

    def test_a_refusing_family_scores_zero_recall_not_a_missing_value(self):
        refuser = TRUTH.copy()
        refuser[0] = -1
        out = ensemble.consensus_recall({"a": TRUTH.copy(), "b": refuser}, TRUTH)
        assert out["per_family"]["b"][0] == pytest.approx(0.0)
        assert out["mean_recall"][0] == pytest.approx(0.5)

    def test_refusal_fraction_counts_tiny_clusters_as_refusals(self):
        refuser = TRUTH.copy()
        refuser[0] = -1
        # "c" assigns every token, but token 1 into a cluster of one
        tiny = TRUTH.copy()
        tiny[1] = 99
        frac = ensemble.refusal_fraction({"a": TRUTH.copy(), "b": refuser, "c": tiny},
                                         substantial=4)
        assert frac[0] == pytest.approx(1 / 3)
        assert frac[1] == pytest.approx(1 / 3)
        assert frac[5] == pytest.approx(0.0)


class TestCalibratedTrichotomy:

    def test_thresholds_come_from_the_pooled_null_confidences(self):
        nulls = [np.linspace(0, 1, 101)]
        out = ensemble.confidence_thresholds(nulls)
        assert out["core"] == pytest.approx(0.95, abs=1e-6)
        assert out["contested"] == pytest.approx(0.5, abs=1e-6)
        assert out["n_null_particles"] == 101

    def test_trichotomy_splits_on_the_calibrated_cuts(self):
        thresholds = {"core": 0.8, "contested": 0.3}
        tags = ensemble.trichotomy(np.array([0.9, 0.5, 0.1, np.nan]), thresholds)
        assert list(tags) == ["core", "halo", "contested", "uncalibrated"]

    def test_without_null_draws_nothing_is_classified(self):
        tags = ensemble.trichotomy(np.array([0.9, 0.1]),
                                   ensemble.confidence_thresholds([]))
        assert set(tags) == {"uncalibrated"}

    def test_build_returns_every_annotation_from_one_matrix(self):
        out = ensemble.build({f"m{i}": TRUTH.copy() for i in range(3)})
        for key in ("co_association", "consensus", "confidence",
                    "consensus_strength", "mean_recall", "min_recall",
                    "refusal_fraction", "n_families", "weights"):
            assert key in out
        assert out["n_families"] == 3
        assert adjusted_rand_score(TRUTH, out["consensus"]["labels"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# P-C2 / P-C3 / P-C4 instruments
# ---------------------------------------------------------------------------

class TestShippedComparison:

    def test_identical_partitions_score_one(self):
        out = comparison.shipped_comparison(TRUTH.copy(), TRUTH.copy(),
                                            {"min_cluster_size": 2, "min_samples": None,
                                             "cluster_selection_method": "eom",
                                             "cluster_selection_epsilon": 0.0})
        assert out["ari_raw"] == pytest.approx(1.0)
        assert out["params_identical"] is True

    def test_refusals_separate_the_two_ari_readings(self):
        shipped = TRUTH.copy()
        shipped[:6] = -1                       # shipped refuses, tuned does not
        out = comparison.shipped_comparison(TRUTH.copy(), shipped, {"min_cluster_size": 5})
        assert out["ari_assigned"] == pytest.approx(1.0)   # where both committed
        assert out["ari_raw"] < 1.0                        # they refuse differently
        assert out["shipped_noise_fraction"] == pytest.approx(0.2)

    def test_mismatched_token_counts_refuse_rather_than_compare(self):
        out = comparison.shipped_comparison(np.zeros(5), np.zeros(6), {})
        assert "error" in out and np.isnan(out["ari_raw"])

    def test_verdict_reads_an_unchanged_default_as_falsification(self):
        per_layer = {l: {"ari_assigned": 1.0, "params_identical": True} for l in range(4)}
        assert "FALSIFIED" in comparison.adjudicate_p_c2(per_layer)["verdict"]

    def test_verdict_separates_a_changed_setting_from_a_changed_partition(self):
        moved = {l: {"ari_assigned": 0.4, "params_identical": False} for l in range(4)}
        assert "CONFIRMED" in comparison.adjudicate_p_c2(moved)["verdict"]
        same_partition = {l: {"ari_assigned": 0.99, "params_identical": False}
                          for l in range(4)}
        assert "PARTIAL" in comparison.adjudicate_p_c2(same_partition)["verdict"]


class TestNoiseRescue:

    def _confidence(self, high_for):
        conf = np.full(TRUTH.size, 0.1)
        conf[high_for] = 0.9
        return conf

    def test_counts_refused_particles_above_the_calibrated_threshold(self):
        shipped = TRUTH.copy()
        shipped[:10] = -1
        conf = self._confidence(slice(0, 8))
        out = comparison.noise_rescue(shipped, conf, TRUTH,
                                      {"core": 0.5, "contested": 0.0})
        assert out["n_noise"] == 10
        assert out["core_fraction"] == pytest.approx(0.8)

    def test_separates_absorption_from_a_population_of_its_own(self):
        shipped = TRUTH.copy()
        shipped[:5] = -1            # half of cluster 0; the rest stay assigned
        conf = np.full(TRUTH.size, 0.9)
        thresholds = {"core": 0.5, "contested": 0.0}
        # consensus puts the refused tokens back with cluster 0, which
        # still holds tokens HDBSCAN did assign
        absorbed = comparison.noise_rescue(shipped, conf, TRUTH, thresholds)
        assert absorbed["into_shared"] == pytest.approx(1.0)
        assert absorbed["own_cluster_fraction"] == pytest.approx(0.0)
        # consensus puts them in a cluster no assigned token belongs to
        own = TRUTH.copy()
        own[:5] = 7
        separate = comparison.noise_rescue(shipped, conf, own, thresholds)
        assert separate["own_cluster_fraction"] == pytest.approx(1.0)
        assert separate["into_shared"] == pytest.approx(0.0)

    def test_no_refusals_reports_nothing_rather_than_zero(self):
        out = comparison.noise_rescue(TRUTH.copy(), np.ones(TRUTH.size), TRUTH,
                                      {"core": 0.5})
        assert out["n_noise"] == 0 and np.isnan(out["core_fraction"])

    def test_verdict_reads_the_registered_floor(self):
        below = {l: {"core_fraction": 0.05, "into_shared": 0.0,
                     "own_cluster_fraction": 0.05} for l in range(3)}
        assert "FALSIFIED" in comparison.adjudicate_p_c3(below)["verdict"]
        above = {l: {"core_fraction": 0.5, "into_shared": 0.1,
                     "own_cluster_fraction": 0.4} for l in range(3)}
        verdict = comparison.adjudicate_p_c3(above)["verdict"]
        assert "CONFIRMED" in verdict and "own group" in verdict


class TestPersistenceAndAUC:

    def test_persistence_is_the_fraction_of_co_members_kept(self):
        here = np.repeat([0, 1], 4)
        nxt = np.array([0, 0, 1, 1, 1, 1, 1, 1])
        out = comparison.persistence_target(here, nxt)
        # token 0 keeps 1 of its 3 co-members
        assert out["fraction"][0] == pytest.approx(1 / 3)
        assert not out["persisted"][0]
        # token 4 keeps all 3
        assert out["fraction"][4] == pytest.approx(1.0)
        assert out["persisted"][4]

    def test_a_lone_particle_is_not_scorable(self):
        out = comparison.persistence_target(np.array([0, 1, 1]), np.array([0, 1, 1]))
        assert not out["scorable"][0]
        assert out["scorable"][1]

    def test_mismatched_layers_raise_rather_than_align_by_position(self):
        with pytest.raises(ValueError):
            comparison.persistence_target(np.zeros(4), np.zeros(5))

    def test_auc_matches_sklearn_including_ties(self):
        rng = np.random.default_rng(2)
        y = rng.random(120) < 0.4
        for scores in (rng.normal(size=120), (rng.random(120) < 0.5).astype(float)):
            assert comparison.auc(scores, y) == pytest.approx(roc_auc_score(y, scores))

    def test_an_informative_score_beats_an_uninformative_one(self):
        rng = np.random.default_rng(3)
        persisted = rng.random(150) < 0.5
        graded = np.where(persisted, rng.normal(1, 1, 150), rng.normal(0, 1, 150))
        binary = (rng.random(150) < 0.5).astype(float)
        target = {"scorable": np.ones(150, bool), "persisted": persisted}
        out = comparison.delta_auc_report(graded, binary, target,
                                          n_permutations=200, n_bootstrap=200)
        assert out["delta_auc"] > 0
        assert out["bootstrap"]["ci_low"] > 0
        assert "CONFIRMED" in out["verdict"]

    def test_the_same_predictor_twice_falsifies(self):
        rng = np.random.default_rng(4)
        persisted = rng.random(150) < 0.5
        binary = (rng.random(150) < 0.5).astype(float)
        target = {"scorable": np.ones(150, bool), "persisted": persisted}
        out = comparison.delta_auc_report(binary, binary, target,
                                          n_permutations=200, n_bootstrap=200)
        assert out["delta_auc"] == pytest.approx(0.0)
        assert "FALSIFIED" in out["verdict"]

    def test_too_few_scorable_particles_is_undecided_not_zero(self):
        target = {"scorable": np.ones(4, bool), "persisted": np.array([1, 0, 0, 0], bool)}
        out = comparison.delta_auc_report(np.arange(4.0), np.zeros(4), target)
        assert "UNDECIDED" in out["verdict"]

    def test_every_verdict_string_carries_its_prediction_id(self):
        # The layer-level adjudicator counts by prefix; a verdict that
        # forgets the id is silently uncounted.
        target = {"scorable": np.ones(4, bool), "persisted": np.array([1, 0, 0, 0], bool)}
        assert comparison.delta_auc_report(
            np.arange(4.0), np.zeros(4), target)["verdict"].startswith("P-C4")


class TestPC1:

    def test_scope_is_named_in_the_verdict(self):
        strength = {0: 0.95, 1: 0.99, 2: 0.5}
        scoped = comparison.adjudicate_p_c1(strength, agreement_layers=[0, 1])
        assert "CONFIRMED" in scoped["verdict"]
        assert scoped["scope"] == "phase1_agreement_layers"
        unscoped = comparison.adjudicate_p_c1(strength)
        assert "weaker test" in unscoped["verdict"]

    def test_falsification_when_tuning_dissolves_the_agreement(self):
        out = comparison.adjudicate_p_c1({0: 0.4, 1: 0.5}, agreement_layers=[0, 1])
        assert "FALSIFIED" in out["verdict"]


# ---------------------------------------------------------------------------
# Artifact IO and one end-to-end run
# ---------------------------------------------------------------------------

def _write_phase1_run(root: Path, n_layers: int = 3, n_tokens: int = 30,
                      d: int = 16, seed: int = 7) -> Path:
    """A minimal but real Phase 1 run directory: planted caps that blur with depth."""
    run_dir = root / "pythia-fixture-step1000_p0"
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    truth = np.repeat([0, 1, 2], n_tokens // 3)
    centers = np.eye(3, d)
    acts = np.zeros((n_layers, n_tokens, d), dtype=np.float32)
    for layer in range(n_layers):
        X = np.stack([centers[t] for t in truth]) + (0.05 + 0.08 * layer) * rng.normal(
            size=(n_tokens, d))
        acts[layer] = X / np.linalg.norm(X, axis=1, keepdims=True)
    np.savez(run_dir / "activations.npz", activations=acts,
             norms=np.ones((n_layers, n_tokens), dtype=np.float32))

    mid = float(sorted(DISTANCE_THRESHOLDS)[len(DISTANCE_THRESHOLDS) // 2])
    json.dump({
        "model": "pythia-fixture-step1000", "prompt": "p0",
        "n_layers": n_layers, "n_tokens": n_tokens, "d_model": d,
        "tokens": [f"tok{i}" for i in range(n_tokens)],
        "checkpoint_step": 1000, "revision": "step1000", "random_init": False,
        "layers": [{"layer": l, "effective_rank": 20.0} for l in range(n_layers)],
    }, open(run_dir / "geometry.json", "w"))

    shipped = {}
    for layer in range(n_layers):
        lab = truth.copy()
        if layer:
            lab[rng.choice(n_tokens, size=2 * layer, replace=False)] = -1
        shipped[str(layer)] = lab.tolist()
    json.dump(shipped, open(run_dir / "hdbscan_labels.json", "w"))

    json.dump({"layers": [
        {"layer": l, "clustering": {
            "hdbscan": {"n_clusters": 3},
            "kmeans": {"best_k": 3 if l < 2 else 8, "best_silhouette": 0.4},
            "agglomerative": {str(mid): 3 if l < 2 else 9,
                              str(float(DISTANCE_THRESHOLDS[0])): 12,
                              "mid_labels": truth.tolist()},
        }} for l in range(n_layers)
    ]}, open(run_dir / "clustering.json", "w"))
    json.dump({"layers": [{"layer": l, "k_eigengap": 3} for l in range(n_layers)]},
              open(run_dir / "spectral.json", "w"))
    json.dump({"layers": [{"layer": l, "sinkhorn_cluster_count_mean": 3.1}
                          for l in range(n_layers)]},
              open(run_dir / "sinkhorn.json", "w"))
    return run_dir


@pytest.fixture()
def phase1_run(tmp_path):
    return _write_phase1_run(tmp_path)


class TestIO:

    def test_load_run_reads_what_the_phase_needs(self, phase1_run):
        run = load_run(phase1_run)
        assert run["activations"].shape[0] == 3
        assert set(run["shipped_hdbscan"]) == {0, 1, 2}
        assert "activations" in run["available"] and "shipped_hdbscan" in run["available"]
        assert run["provenance"]["activations.npz"]["exists"]

    def test_a_run_without_activations_is_refused_not_degraded(self, tmp_path):
        empty = tmp_path / "no_activations"
        empty.mkdir()
        json.dump({"model": "x"}, open(empty / "geometry.json", "w"))
        with pytest.raises(FileNotFoundError, match="re-clusters"):
            load_run(empty)

    def test_identity_does_not_invent_a_checkpoint(self, phase1_run):
        run = load_run(phase1_run)
        assert run_identity(run)["checkpoint_step"] == 1000
        run["geometry"].pop("checkpoint_step")
        assert run_identity(run)["checkpoint_step"] is None

    def test_agreement_layers_follow_the_kmeans_trust_gate(self, phase1_run):
        out = phase1_agreement_layers(phase1_run)
        assert out["available"]
        # layers 0 and 1 agree at 3; layer 2 has kmeans 8 / agglomerative 9
        assert out["layers"] == [0, 1]
        assert out["per_layer"][2]["kmeans_trusted"] is True

    def test_a_missing_clustering_json_is_reported_not_guessed(self, tmp_path):
        run_dir = _write_phase1_run(tmp_path / "bare")
        (run_dir / "clustering.json").unlink()
        out = phase1_agreement_layers(run_dir)
        assert out["available"] is False and out["reason"]

    def test_agreement_layers_match_the_visualization_implementation(self, phase1_run):
        cm = pytest.importorskip(
            "p1_mstate_tracking.visualization.cluster_methods",
            reason="visualization package needs matplotlib",
        )
        table = cm.cluster_count_table(phase1_run)
        theirs = [l for l, ok in zip(table["layers"], table["agreement"]) if ok]
        assert phase1_agreement_layers(phase1_run)["layers"] == theirs

    def test_particle_table_round_trips_with_its_extra_columns(self, tmp_path):
        from core.particles import ParticleTable
        arrays = {
            0: {
                "consensus_labels": TRUTH.copy(),
                "confidence": np.linspace(0, 1, TRUTH.size),
                "mean_recall": np.ones(TRUTH.size),
                "min_recall": np.ones(TRUTH.size),
                "refusal_fraction": np.zeros(TRUTH.size),
                "hdbscan_label": TRUTH.copy(),
                "n_families": np.full(TRUTH.size, 5),
                "population": np.array(["core"] * TRUTH.size),
            }
        }
        table = build_particle_table(
            {"model": "m", "prompt_key": "p", "checkpoint_step": 5}, arrays,
            tokens=[f"t{i}" for i in range(TRUTH.size)])
        path = tmp_path / "particle_table.npz"
        table.save(path)
        back = ParticleTable.load(path)
        assert len(back) == TRUTH.size
        assert set(back.extra) >= {"confidence", "hdbscan_label", "refusal_fraction"}
        assert len(back.filter(population="core")) == TRUTH.size

    def test_saved_results_survive_a_json_round_trip(self, tmp_path):
        written = save_p1d(
            tmp_path / "out",
            {"identity": {"model": "m"}, "value": np.float32(0.5),
             "nan": float("nan"), "set": {"a", "b"}, "array": np.arange(3)},
            {0: {"confidence": np.zeros(4), "co_association": np.eye(4)}},
        )
        payload = json.load(open(written["p1d_results"]))
        assert payload["value"] == pytest.approx(0.5)
        assert payload["nan"] is None
        assert payload["set"] == ["a", "b"]
        arrays = np.load(written["p1d_ensemble"])
        assert arrays["co_association_L0"].dtype == np.float32

    def test_matrices_can_be_dropped_without_losing_the_annotations(self, tmp_path):
        written = save_p1d(tmp_path / "out", {"identity": {}},
                           {0: {"confidence": np.zeros(4), "co_association": np.eye(4)}},
                           save_matrices=False)
        arrays = np.load(written["p1d_ensemble"])
        assert "confidence_L0" in arrays and "co_association_L0" not in arrays


class TestDriverEndToEnd:

    def test_run_one_produces_every_artifact_and_verdict(self, phase1_run, tmp_path):
        from core.artifacts import validate_artifact
        from p1d_cluster_ensemble.run_1d import build_parser, run_one, summary_text

        args = build_parser().parse_args([
            "--results", str(phase1_run), "--out", str(tmp_path / "out"),
            "--grid", "quick", "--n-repeats", "2", "--n-null", "20",
            "--n-null-repeats", "1", "--n-null-confidence", "3", "--top-m", "1",
            "--n-permutations", "50", "--n-bootstrap", "50",
        ])
        bundle = run_one(phase1_run, args)
        results = bundle["results"]

        assert results["stages"] == ["A", "B", "C", "D", "E"]
        assert results["layers"] == [0, 1, 2]
        assert set(results["verdicts"]) == {"P-C1", "P-C2", "P-C3", "P-C4"}
        for name, verdict in results["verdicts"].items():
            assert verdict["verdict"].startswith(name), verdict["verdict"]
        assert results["phase1_agreement"]["layers"] == [0, 1]

        out_dir = tmp_path / "out" / phase1_run.name
        save_p1d(out_dir, results, bundle["arrays"])
        build_particle_table(bundle["identity"], bundle["arrays"],
                             bundle["tokens"]).save(out_dir / "particle_table.npz")
        for name in ("p1d_results", "p1d_ensemble"):
            assert validate_artifact(out_dir, "phase1d", name)["ok"], name
        assert validate_artifact(out_dir, "particles", "particle_table")["ok"]
        assert "Verdicts" in summary_text(results)

    def test_prerequisites_are_pulled_in_rather_than_assumed(self):
        from p1d_cluster_ensemble.run_1d import expand_subexperiments
        assert expand_subexperiments(["D"]) == ["A", "B", "D"]
        assert expand_subexperiments(["A"]) == ["A"]
        assert expand_subexperiments(["E", "C"]) == ["A", "B", "C", "E"]

    def test_discover_runs_only_returns_directories_this_phase_can_use(self, tmp_path):
        from p1d_cluster_ensemble.run_1d import discover_runs
        usable = _write_phase1_run(tmp_path)
        (tmp_path / "not_a_run").mkdir()
        assert discover_runs(tmp_path) == [usable]
        assert discover_runs(usable) == [usable]
