"""
tests/test_phase1d_methods.py — the method registry and the tuning layer.

Three synthetic regimes carry almost every assertion here, because they
are the three cases the phase has to tell apart and the only ones whose
right answer is known before the code runs:

  planted        three tight caps on S^{d-1}. Every family should recover
                 them, and every family should be admitted by the gate.
  structureless  i.i.d. uniform on the sphere. Nothing to find. A family
                 admitted here is a false positive, and the gate's whole
                 purpose is to keep that rate at alpha.
  collapsed      every token near one direction — Phase 1's actual deep
                 layers. Silhouette against an absolute threshold reads
                 structure here (which is why cluster_methods.py needed a
                 trust gate at all); against the matched null it must not.

No matplotlib, no torch, no model loading, no run directory: this is
numpy + sklearn only, so it runs in the stub-heavy non-smoke session.
"""
from __future__ import annotations

import numpy as np
import pytest

from sklearn.metrics import adjusted_rand_score

from p1d_cluster_ensemble.constants import (
    DISTANCE_THRESHOLDS, K_VALUES, KMEANS_RANK_MIN, KMEANS_SIL_MIN,
    SHIPPED_HDBSCAN_PARAMS,
)
from p1d_cluster_ensemble.methods import (
    CAN_REFUSE, FAMILIES, LayerData, _greedy_modularity, available_families,
    fit, hdbscan_backend, modularity, mutual_knn_graph, param_grid,
    spherical_kmeans,
)
from p1d_cluster_ensemble.selection import (
    Candidate, apply_gate, calibrate, partition_summary, select_family,
    separation_score, subsample_stability, sweep_family,
)

D = 24
GATE_KWARGS = dict(grid="quick", n_repeats=3, n_null=20, n_null_repeats=2, top_m=2)


# ---------------------------------------------------------------------------
# Fixtures — the three regimes
# ---------------------------------------------------------------------------

def _normed(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def planted():
    rng = np.random.default_rng(0)
    centers = np.eye(3, D)
    X = np.concatenate([c + 0.06 * rng.normal(size=(15, D)) for c in centers])
    return LayerData.from_normed(_normed(X))


@pytest.fixture(scope="module")
def planted_truth():
    return np.repeat([0, 1, 2], 15)


@pytest.fixture(scope="module")
def structureless():
    rng = np.random.default_rng(1)
    return LayerData.from_normed(_normed(rng.normal(size=(45, D))))


@pytest.fixture(scope="module")
def collapsed():
    rng = np.random.default_rng(2)
    v = _normed(rng.normal(size=(1, D)))[0]
    return LayerData.from_normed(_normed(v + 0.05 * rng.normal(size=(45, D))))


# ---------------------------------------------------------------------------
# Constants read out of source rather than copied
# ---------------------------------------------------------------------------

class TestConstants:

    def test_distance_thresholds_match_phase1s_sweep(self):
        # The ast reader exists so this phase never imports core.config
        # (torch, transformers). The test does import it — under the
        # session stubs — precisely to prove the reader agrees with the
        # real value.
        config = pytest.importorskip(
            "core.config", reason="core.config imports torch/transformers")
        assert np.allclose(DISTANCE_THRESHOLDS, config.DISTANCE_THRESHOLDS)

    def test_k_values_match_phase1s_range(self):
        config = pytest.importorskip(
            "core.config", reason="core.config imports torch/transformers")
        assert K_VALUES == list(config.K_RANGE)

    def test_kmeans_trust_gate_matches_the_visualization_modules(self):
        cm = pytest.importorskip(
            "p1_mstate_tracking.visualization.cluster_methods",
            reason="visualization package needs matplotlib",
        )
        assert KMEANS_SIL_MIN == cm.KMEANS_SIL_MIN
        assert KMEANS_RANK_MIN == cm.KMEANS_RANK_MIN

    def test_shipped_params_are_the_ones_phase1_actually_calls(self):
        # clustering.py's call is hdbscan.HDBSCAN(min_cluster_size=2,
        # metric="precomputed") — everything else library default. If that
        # call changes, P-C2 is comparing against a partition nobody ships.
        import ast
        from pathlib import Path
        src = Path(__file__).resolve().parents[1] / "p1_mstate_tracking" / "clustering.py"
        tree = ast.parse(src.read_text())
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "HDBSCAN"
        ]
        assert calls, "no HDBSCAN(...) call found in p1_mstate_tracking/clustering.py"
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords
                  if kw.arg != "metric"}
        assert kwargs == {"min_cluster_size": SHIPPED_HDBSCAN_PARAMS["min_cluster_size"]}


# ---------------------------------------------------------------------------
# LayerData
# ---------------------------------------------------------------------------

class TestLayerData:

    def test_normalizes_rows_that_arrive_unnormed(self):
        X = np.array([[3.0, 4.0], [0.0, 2.0]])
        data = LayerData.from_normed(X)
        assert np.allclose(np.linalg.norm(data.normed, axis=1), 1.0)

    def test_cosine_distance_is_symmetric_with_a_zero_diagonal(self, planted):
        D_ = planted.cos_dist
        assert np.allclose(D_, D_.T)
        assert np.allclose(np.diag(D_), 0.0)
        assert (D_ >= 0).all()

    def test_subset_slices_both_views_consistently(self, planted):
        idx = [0, 5, 20, 40]
        sub = planted.subset(idx)
        assert sub.n == 4
        assert np.allclose(sub.normed, planted.normed[idx])
        assert np.allclose(sub.cos_dist, planted.cos_dist[np.ix_(idx, idx)])

    def test_rejects_a_non_matrix(self):
        with pytest.raises(ValueError):
            LayerData.from_normed(np.zeros(5))


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------

class TestFamilies:

    @pytest.mark.parametrize("family", FAMILIES)
    def test_some_grid_point_recovers_planted_caps(self, family, planted, planted_truth):
        if family == "hdbscan" and family not in available_families():
            pytest.skip("no HDBSCAN backend in this environment")
        best = max(adjusted_rand_score(planted_truth, fit(family, p, planted, seed=0))
                   for p in param_grid(family, planted.n))
        assert best > 0.99, f"{family} never recovered three well-separated caps"

    @pytest.mark.parametrize("family", FAMILIES)
    def test_labels_have_the_right_shape_and_only_hdbscan_refuses(
        self, family, structureless,
    ):
        if family == "hdbscan" and family not in available_families():
            pytest.skip("no HDBSCAN backend in this environment")
        for params in param_grid(family, structureless.n, grid="quick"):
            labels = fit(family, params, structureless, seed=0)
            assert labels.shape == (structureless.n,)
            if family not in CAN_REFUSE:
                assert (labels >= 0).all()

    def test_grids_are_clipped_to_the_token_count(self):
        tiny = LayerData.from_normed(np.eye(4, D))
        for family in available_families():
            for params in param_grid(family, tiny.n):
                labels = fit(family, params, tiny, seed=0)
                assert labels.size == 4

    def test_quick_grid_is_a_subset_of_the_full_one(self):
        for family in FAMILIES:
            full = [tuple(sorted(p.items())) for p in param_grid(family, 40, "full")]
            quick = [tuple(sorted(p.items())) for p in param_grid(family, 40, "quick")]
            assert set(quick) <= set(full), family

    def test_hdbscan_backend_is_recorded_not_assumed(self):
        backend = hdbscan_backend()
        assert set(backend) == {"name", "available", "version"}
        assert backend["name"] in ("hdbscan", "sklearn", "none")

    def test_precomputed_distances_survive_a_fit(self, planted):
        # sklearn's HDBSCAN mutates a precomputed matrix unless copy=True,
        # and this phase re-fits many settings against one cached LayerData.
        if "hdbscan" not in available_families():
            pytest.skip("no HDBSCAN backend in this environment")
        before = planted.cos_dist.copy()
        fit("hdbscan", {"min_cluster_size": 2}, planted, seed=0)
        assert np.array_equal(planted.cos_dist, before)


class TestSphericalKMeans:

    def test_recovers_planted_caps(self, planted, planted_truth):
        labels = spherical_kmeans(planted.normed.astype(np.float64), 3, seed=0)
        assert adjusted_rand_score(planted_truth, labels) > 0.99

    def test_centroids_never_leave_the_sphere(self):
        # The whole reason this family exists: Euclidean k-means updates a
        # centroid off S^{d-1}. Checked through the objective — a spherical
        # assignment can only be the argmax of <x, mu> for unit mu.
        rng = np.random.default_rng(3)
        X = _normed(rng.normal(size=(30, D)))
        labels = spherical_kmeans(X, 3, seed=0)
        centers = np.stack([_normed(X[labels == c].sum(axis=0)[None, :])[0]
                            for c in np.unique(labels)])
        assigned = np.argmax(X @ centers.T, axis=1)
        assert adjusted_rand_score(labels, assigned) > 0.99

    def test_is_deterministic_for_a_fixed_seed(self, planted):
        a = spherical_kmeans(planted.normed.astype(np.float64), 3, seed=11)
        b = spherical_kmeans(planted.normed.astype(np.float64), 3, seed=11)
        assert (a == b).all()


class TestGraphModularity:

    def test_recovers_a_planted_stochastic_block_model(self):
        rng = np.random.default_rng(4)
        blocks = np.repeat([0, 1, 2], 20)
        p = np.where(blocks[:, None] == blocks[None, :], 0.6, 0.05)
        W = (rng.random((60, 60)) < p).astype(float)
        W = np.triu(W, 1)
        W = W + W.T
        labels = _greedy_modularity(W)
        assert adjusted_rand_score(blocks, labels) > 0.95

    def test_two_disjoint_cliques_hit_the_analytic_modularity(self):
        # An exact anchor for the merge bookkeeping. Two equal disjoint
        # components split perfectly give Q = 2 * (1/2 - 1/4) = 0.5, and a
        # wrong accumulation of e[i][j] cannot land on that number by
        # accident.
        W = np.zeros((20, 20))
        W[:10, :10] = 1.0
        W[10:, 10:] = 1.0
        np.fill_diagonal(W, 0.0)
        labels = _greedy_modularity(W)
        assert len(set(labels.tolist())) == 2
        assert (labels[:10] == labels[0]).all() and (labels[10:] == labels[10]).all()
        assert modularity(W, labels) == pytest.approx(0.5)

    def test_greedy_partition_nearly_matches_the_planted_one(self):
        # An independent recomputation of the quantity being maximized —
        # the merge bookkeeping can return plausible communities while
        # accumulating e[i][j] wrongly. Greedy modularity is a heuristic
        # with no optimality guarantee (it misplaces a node or two on a
        # sparse block model), so this bounds the shortfall rather than
        # demanding the planted partition be beaten.
        rng = np.random.default_rng(5)
        blocks = np.repeat([0, 1, 2], 20)
        p = np.where(blocks[:, None] == blocks[None, :], 0.6, 0.05)
        W = (rng.random((60, 60)) < p).astype(float)
        W = np.triu(W, 1)
        W = W + W.T
        planted_q = modularity(W, blocks)
        assert modularity(W, _greedy_modularity(W)) >= 0.95 * planted_q

    def test_an_empty_graph_leaves_every_node_a_singleton(self):
        labels = _greedy_modularity(np.zeros((6, 6)))
        assert len(set(labels.tolist())) == 6

    def test_mutual_knn_is_symmetric_and_drops_one_sided_edges(self, planted):
        W = mutual_knn_graph(planted, 3)
        assert np.allclose(W, W.T)
        assert np.allclose(np.diag(W), 0.0)
        assert (W >= 0).all()


# ---------------------------------------------------------------------------
# Partition shape
# ---------------------------------------------------------------------------

class TestPartitionSummary:

    def test_counts_only_substantial_clusters_toward_non_triviality(self):
        labels = np.array([0] * 20 + [1] + [2] + [3])
        out = partition_summary(labels, substantial=4)
        assert out["k"] == 4
        assert out["k_substantial"] == 1
        assert out["trivial"] and "fewer than two" in out["trivial_reason"]

    def test_a_dominant_cluster_is_trivial(self):
        labels = np.array([0] * 96 + [1] * 4)
        out = partition_summary(labels, substantial=4)
        assert out["k_substantial"] == 2
        assert out["trivial"] and "holds" in out["trivial_reason"]

    def test_a_genuine_partition_is_not_trivial(self):
        out = partition_summary(np.repeat([0, 1, 2], 10), substantial=4)
        assert not out["trivial"]
        assert out["largest_fraction"] == pytest.approx(1 / 3)

    def test_refusals_lower_the_assigned_fraction(self):
        labels = np.array([-1] * 5 + [0] * 10 + [1] * 5)
        out = partition_summary(labels, substantial=4)
        assert out["assigned_fraction"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Stability and separation
# ---------------------------------------------------------------------------

class TestStability:

    def test_planted_structure_is_reproducible_under_resampling(self, planted):
        out = subsample_stability("kmeans", {"k": 3}, planted, n_repeats=4, seed=0)
        assert out["mean_ari"] > 0.9
        assert out["n_repeats_usable"] == 4

    def test_an_over_split_partition_is_less_reproducible(self, planted):
        three = subsample_stability("kmeans", {"k": 3}, planted, n_repeats=4, seed=0)
        nine = subsample_stability("kmeans", {"k": 9}, planted, n_repeats=4, seed=0)
        assert nine["mean_ari"] < three["mean_ari"]

    def test_separation_excludes_refused_tokens(self, planted, planted_truth):
        labels = planted_truth.copy()
        clean = separation_score(labels, planted)
        with_noise = labels.copy()
        with_noise[:3] = -1
        assert np.isfinite(separation_score(with_noise, planted))
        # Dropping three tokens cannot move a silhouette of 45 tokens much;
        # counting them as a fourth cluster would.
        assert abs(separation_score(with_noise, planted) - clean) < 0.1

    def test_separation_is_nan_when_one_cluster_survives_exclusion(self, planted):
        labels = np.zeros(planted.n, dtype=int)
        assert np.isnan(separation_score(labels, planted))


class TestCalibration:

    def test_p_value_is_the_conservative_permutation_form(self):
        null = np.arange(10, dtype=float)      # 0..9
        out = calibrate(9.5, null, alpha=0.1)
        assert out["n_null_ge"] == 0
        assert out["p_value"] == pytest.approx(1 / 11)
        assert out["p_floor"] == pytest.approx(1 / 11)

    def test_ties_count_against_the_observation(self):
        out = calibrate(1.0, np.ones(10), alpha=0.1)
        assert out["n_null_ge"] == 10
        assert out["p_value"] == pytest.approx(11 / 11)

    def test_unusable_when_the_null_is_empty(self):
        out = calibrate(0.5, np.array([np.nan, np.nan]))
        assert out["usable"] is False


class TestGate:

    def _candidate(self, sep_obs, sep_null, stab_obs, stab_null):
        cand = Candidate(
            family="kmeans", params={"k": 3}, labels=np.repeat([0, 1, 2], 5),
            shape=partition_summary(np.repeat([0, 1, 2], 5)),
            stability={"mean_ari": stab_obs}, separation=sep_obs,
        )
        cand.null = {"separation": calibrate(sep_obs, sep_null),
                     "stability": calibrate(stab_obs, stab_null)}
        return apply_gate(cand)

    def test_admits_a_partition_that_beats_both_nulls(self):
        cand = self._candidate(0.8, np.full(20, 0.2), 0.9, np.full(20, 0.4))
        assert cand.admissible
        assert cand.branch == "separation_rank_test+stability_floor"

    def test_rejects_a_reproducible_partition_that_is_not_separated(self):
        cand = self._candidate(0.2, np.full(20, 0.2), 0.95, np.full(20, 0.4))
        assert not cand.admissible
        assert "separation" in cand.reason

    def test_rejects_a_separated_partition_that_is_less_reproducible_than_noise(self):
        cand = self._candidate(0.8, np.full(20, 0.2), 0.2, np.full(20, 0.6))
        assert not cand.admissible
        assert "stability" in cand.reason

    def test_a_saturated_stability_null_does_not_veto_by_ties(self):
        # The measured case that motivated the floor: observed 1.00 with
        # some null draws also at 1.00 is p ~ 0.14 under a rank test, and
        # rejecting there discards a partition that recovers the planted
        # structure exactly.
        null = np.array([1.0, 1.0] + [0.85] * 18)
        cand = self._candidate(0.9, np.full(20, 0.3), 1.0, null)
        assert cand.admissible

    def test_gate_refuses_before_nulls_are_computed(self):
        cand = Candidate(family="kmeans", params={"k": 2},
                         labels=np.zeros(4, dtype=int),
                         shape=partition_summary(np.zeros(4, dtype=int)),
                         stability={"mean_ari": 1.0})
        with pytest.raises(ValueError):
            apply_gate(cand)


# ---------------------------------------------------------------------------
# Selection, end to end on the three regimes
# ---------------------------------------------------------------------------

class TestSelection:

    def test_every_family_is_admitted_on_planted_structure(self, planted, planted_truth):
        for family in available_families():
            res = select_family(family, planted, **GATE_KWARGS)
            assert res["selected"] is not None, f"{family} abstained on planted caps"
            labels = np.asarray(res["selected_labels"])
            assert adjusted_rand_score(planted_truth, labels) > 0.9

    def test_almost_nothing_is_admitted_on_structureless_data(self, structureless):
        admitted = [f for f in available_families()
                    if select_family(f, structureless, **GATE_KWARGS)["selected"] is not None]
        # The gate is per (family, candidate) and is not corrected for
        # multiplicity: 7 families x 2 gated candidates at alpha=0.05 is
        # ~0.7 expected false positives, so this asserts the rate, not
        # perfection. Two or more would mean the gate is not doing its job.
        assert len(admitted) <= 1, f"gate admitted {admitted} on i.i.d. points"

    def test_nothing_is_admitted_in_the_collapsed_regime(self, collapsed):
        admitted = [f for f in available_families()
                    if select_family(f, collapsed, **GATE_KWARGS)["selected"] is not None]
        assert admitted == [], f"gate admitted {admitted} on a collapsed cloud"

    def test_abstention_carries_a_reason_and_the_surface(self, collapsed):
        res = select_family("kmeans", collapsed, **GATE_KWARGS)
        assert res["selected"] is None
        assert res["reason"]
        assert res["candidates"], "the surface is the artifact; it must survive abstention"
        assert res["gated"], "the gated candidates' numbers must survive too"

    def test_refuses_an_alpha_the_draw_count_cannot_attain(self, planted):
        with pytest.raises(ValueError, match="smallest attainable"):
            select_family("kmeans", planted, grid="quick", n_repeats=2,
                          n_null=5, n_null_repeats=1, top_m=1, alpha=0.05)

    def test_sweep_keeps_trivial_candidates_with_their_numbers(self, collapsed):
        cands = sweep_family("kmeans", collapsed, grid="quick", n_repeats=2, seed=0)
        assert cands, "the sweep must return every grid point"
        assert all("mean_ari" in c.stability for c in cands)

    def test_selection_is_reproducible_for_a_fixed_seed(self, planted):
        a = select_family("kmeans", planted, **GATE_KWARGS)
        b = select_family("kmeans", planted, **GATE_KWARGS)
        assert a["selected"]["params"] == b["selected"]["params"]
        assert a["selected_labels"] == b["selected_labels"]
