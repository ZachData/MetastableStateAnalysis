"""
tests/test_phase5b_alignment.py — contracts for the Option A rewrite.

Four groups, each targeting one thing that was silently wrong:

  1. TestPeriodicKnot        — the duplicate-knot bug (14 prior failures)
  2. TestPCAReduceShape      — the rank-truncation bug (2 prior failures)
  3. TestBehaviorAlignment   — the masked-vs-global average bug (Sub-exp B)
  4. TestFrameSeparation     — LN frame actually applied, not silently sphere

Group 3 is the load-bearing one: assertion `test_masked_not_global` is the
assertion that would have caught the original bug, and everything else in
that class exists to keep the alignment honest as the code moves.

Deterministic, pure numpy, no model, no GPU. Run with:
    pytest tests/test_phase5b_alignment.py -v
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest


RNG_SEED = 20260721
D_MODEL  = 32
VOCAB    = 64
N_TOK    = 12


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _two_layer_setup():
    """
    Two layers, two clusters, hand-built so every expected value is
    computable by hand rather than by re-running the implementation.

    Tokens 0-3   -> cluster 0 at both layers
    Tokens 4-7   -> cluster 1 at both layers
    Tokens 8-11  -> noise (-1) at both layers; these exist specifically so
                    a global mean over all tokens differs from a masked
                    mean, which is what test_masked_not_global detects.
    """
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1], dtype=np.int32)
    label_arrays = {0: labels.copy(), 1: labels.copy()}

    # Distributions: cluster 0 peaks on token 0, cluster 1 peaks on token 1,
    # noise peaks on token 2 (far from both, so it drags a global mean).
    def _peaked(idx):
        p = np.full(VOCAB, 1e-6)
        p[idx] = 1.0
        return p / p.sum()

    logit_dists = {}
    for layer in (0, 1):
        P = np.zeros((N_TOK, VOCAB))
        for t in range(N_TOK):
            if labels[t] == 0:
                P[t] = _peaked(0)
            elif labels[t] == 1:
                P[t] = _peaked(1)
            else:
                P[t] = _peaked(2)
        logit_dists[layer] = P.astype(np.float32)

    trajectories = [
        {"id": 0, "chain": [(0, 0), (1, 0)], "lifespan": 2},
        {"id": 1, "chain": [(0, 1), (1, 1)], "lifespan": 2},
    ]
    return label_arrays, logit_dists, trajectories, labels


def _ring(n: int, d: int, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c = np.zeros((n, d))
    c[:, 0] = np.cos(ang)
    c[:, 1] = np.sin(ang)
    c[:, 2:] = rng.standard_normal((n, d - 2)) * 0.05
    return c / np.linalg.norm(c, axis=1, keepdims=True)


# ===========================================================================
# 1 — periodic knot
# ===========================================================================

class TestPeriodicKnot:
    """The knot vector must stay strictly increasing for any valid u."""

    def test_wrap_extrapolates_when_u_reaches_one(self):
        from p5b_manifold_steering.manifold_fit import _periodic_wrap_u
        u = np.linspace(0, 1, 7)
        assert _periodic_wrap_u(u) > 1.0

    def test_wrap_uses_period_when_u_stops_short(self):
        from p5b_manifold_steering.manifold_fit import _periodic_wrap_u
        u = np.linspace(0, 6 / 7, 7)
        assert _periodic_wrap_u(u) == pytest.approx(1.0)

    def test_activation_fit_accepts_linspace_u(self):
        """The exact input that used to raise from scipy."""
        from p5b_manifold_steering.manifold_fit import fit_activation_manifold
        ring = _ring(7, 8)
        u = np.linspace(0, 1, 7)
        mh = fit_activation_manifold(ring, u, periodic=True)
        assert mh["u_wrap"] > u[-1]

    def test_behavior_fit_accepts_linspace_u(self):
        from p5b_manifold_steering.manifold_fit import fit_behavior_manifold
        rng = np.random.default_rng(RNG_SEED)
        p = np.abs(rng.standard_normal((7, 16))) + 1e-3
        p /= p.sum(axis=1, keepdims=True)
        my = fit_behavior_manifold(p, np.linspace(0, 1, 7), periodic=True)
        assert my["u_wrap"] > 1.0

    def test_periodic_spline_still_interpolates(self):
        """Fixing the knot must not break interpolation at the knots."""
        from p5b_manifold_steering.manifold_fit import (
            fit_activation_manifold, eval_manifold,
        )
        ring = _ring(7, 8)
        u = np.linspace(0, 1, 7)
        mh = fit_activation_manifold(ring, u, periodic=True)
        npt.assert_allclose(eval_manifold(mh, u), ring, atol=1e-6)


# ===========================================================================
# 2 — pca_reduce shape contract
# ===========================================================================

class TestPCAReduceShape:

    def test_shape_honoured_when_k_exceeds_rank(self):
        from p5b_manifold_steering.manifold_fit import pca_reduce
        c = _ring(7, 64)
        scores, basis, evr = pca_reduce(c, 32)
        assert scores.shape == (7, 32)
        assert basis.shape == (64, 32)
        assert evr.shape == (32,)

    def test_basis_orthonormal_when_padded(self):
        from p5b_manifold_steering.manifold_fit import pca_reduce
        _, basis, _ = pca_reduce(_ring(7, 64), 32)
        npt.assert_allclose(basis.T @ basis, np.eye(32), atol=1e-8)

    def test_padded_components_report_zero_variance(self):
        """A caller must be able to see which columns are real."""
        from p5b_manifold_steering.manifold_fit import pca_reduce
        _, _, evr = pca_reduce(_ring(7, 64), 32)
        assert np.all(evr[7:] == 0.0)
        assert evr[:2].sum() > 0.90

    def test_evr_non_increasing(self):
        from p5b_manifold_steering.manifold_fit import pca_reduce
        _, _, evr = pca_reduce(_ring(7, 64), 32)
        assert np.all(np.diff(evr) <= 1e-12)

    def test_deterministic_across_calls(self):
        from p5b_manifold_steering.manifold_fit import pca_reduce
        c = _ring(7, 64)
        b1 = pca_reduce(c, 32)[1]
        b2 = pca_reduce(c, 32)[1]
        npt.assert_allclose(b1, b2)

    def test_k_clamped_to_ambient_dim(self):
        from p5b_manifold_steering.manifold_fit import pca_reduce
        _, basis, _ = pca_reduce(_ring(7, 8), 64)
        assert basis.shape == (8, 8)


# ===========================================================================
# 3 — behavior/centroid alignment  (the load-bearing group)
# ===========================================================================

class TestBehaviorAlignment:

    def setup_method(self):
        (self.label_arrays, self.logit_dists,
         self.trajectories, self.labels) = _two_layer_setup()

    def test_same_trajectory_ids_as_centroids(self):
        """No silent drops, no extras, relative to the centroid side."""
        from p1_mstate_tracking.cluster_tracking import (
            compute_behavior_trajectories, compute_centroid_trajectories,
        )
        rng = np.random.default_rng(RNG_SEED)
        hidden = [rng.standard_normal((N_TOK, D_MODEL)) for _ in range(2)]
        cen = compute_centroid_trajectories(
            {"trajectories": self.trajectories}, hidden,
            [self.label_arrays[0], self.label_arrays[1]],
        )
        beh, _ = compute_behavior_trajectories(
            self.trajectories, self.label_arrays, self.logit_dists,
        )
        assert set(beh.keys()) == set(cen.keys())

    def test_masked_not_global(self):
        """
        THE assertion. Cluster 0's distribution must equal the average over
        cluster-0 tokens only — peaked on token 0 — NOT the average over
        every token at that layer, which the noise tokens drag toward
        token 2. The original bug was exactly this global average.
        """
        from p1_mstate_tracking.cluster_tracking import compute_behavior_trajectories
        beh, _ = compute_behavior_trajectories(
            self.trajectories, self.label_arrays, self.logit_dists,
            space="mixture",
        )
        d0 = beh[0].mean(axis=0)
        assert int(np.argmax(d0)) == 0, "cluster 0 must decode to token 0"
        assert d0[2] < 1e-3, "noise-token mass leaked in — this is a global mean"

        global_mean = self.logit_dists[0].mean(axis=0)
        assert not np.allclose(d0, global_mean, atol=1e-4)

    def test_distinct_clusters_give_distinct_distributions(self):
        from p1_mstate_tracking.cluster_tracking import compute_behavior_trajectories
        beh, _ = compute_behavior_trajectories(
            self.trajectories, self.label_arrays, self.logit_dists,
        )
        d0 = beh[0].mean(axis=0)
        d1 = beh[1].mean(axis=0)
        assert int(np.argmax(d0)) != int(np.argmax(d1))

    def test_rows_are_valid_distributions(self):
        from p1_mstate_tracking.cluster_tracking import compute_behavior_trajectories
        for space in ("hellinger", "mixture"):
            beh, _ = compute_behavior_trajectories(
                self.trajectories, self.label_arrays, self.logit_dists,
                space=space,
            )
            for arr in beh.values():
                assert np.all(arr >= -1e-6)
                npt.assert_allclose(arr.sum(axis=1), 1.0, atol=1e-5)

    def test_sparse_logit_cache_still_yields_result(self):
        """
        A chain layer absent from the logit cache must degrade coverage,
        not drop the whole trajectory. This is what happens on every real
        run where the cache does not span every chain layer.
        """
        from p1_mstate_tracking.cluster_tracking import compute_behavior_trajectories
        partial = {0: self.logit_dists[0]}          # layer 1 missing
        beh, cov = compute_behavior_trajectories(
            self.trajectories, self.label_arrays, partial,
        )
        assert set(beh.keys()) == {0, 1}
        assert cov[0]["layers_used"] == [0]
        assert cov[0]["frac"] == pytest.approx(0.5)

    def test_zero_coverage_trajectory_is_absent_not_empty(self):
        from p1_mstate_tracking.cluster_tracking import compute_behavior_trajectories
        beh, cov = compute_behavior_trajectories(
            self.trajectories, self.label_arrays, {},
        )
        assert beh == {}
        assert cov[0]["frac"] == 0.0

    def test_token_count_mismatch_raises(self):
        """Labels and logits from different passes must not be masked together."""
        from p1_mstate_tracking.cluster_tracking import compute_behavior_trajectories
        bad = {0: np.ones((N_TOK + 3, VOCAB)) / VOCAB, 1: self.logit_dists[1]}
        with pytest.raises(ValueError, match="not from the same pass"):
            compute_behavior_trajectories(
                self.trajectories, self.label_arrays, bad,
            )

    def test_stack_follows_traj_ids_order(self):
        """Row order must follow traj_ids, not dict insertion order."""
        from p1_mstate_tracking.cluster_tracking import (
            compute_behavior_trajectories, stack_behavior_by_traj_ids,
        )
        beh, _ = compute_behavior_trajectories(
            self.trajectories, self.label_arrays, self.logit_dists,
        )
        fwd, kept_f = stack_behavior_by_traj_ids(beh, [0, 1])
        rev, kept_r = stack_behavior_by_traj_ids(beh, [1, 0])
        assert kept_f == [0, 1] and kept_r == [1, 0]
        npt.assert_allclose(fwd[0], rev[1])
        npt.assert_allclose(fwd[1], rev[0])

    def test_stack_reports_kept_subset(self):
        from p1_mstate_tracking.cluster_tracking import (
            compute_behavior_trajectories, stack_behavior_by_traj_ids,
        )
        beh, _ = compute_behavior_trajectories(
            self.trajectories, self.label_arrays, self.logit_dists,
        )
        dists, kept = stack_behavior_by_traj_ids(beh, [0, 1, 99])
        assert kept == [0, 1]
        assert dists.shape[0] == 2

    def test_hellinger_sharper_than_mixture(self):
        """
        Documents the space= tradeoff rather than assuming it. Averaging in
        √p space preserves peak mass; averaging in p space blurs it.
        """
        from p1_mstate_tracking.cluster_tracking import compute_behavior_trajectories
        mixed = {0: np.stack([
            np.eye(VOCAB)[0], np.eye(VOCAB)[1],
            np.eye(VOCAB)[0], np.eye(VOCAB)[1],
        ] + [np.eye(VOCAB)[5]] * 8)}
        labels = {0: np.array([0, 0, 0, 0] + [-1] * 8, dtype=np.int32)}
        traj = [{"id": 0, "chain": [(0, 0)]}]
        h, _ = compute_behavior_trajectories(traj, labels, mixed, space="hellinger")
        m, _ = compute_behavior_trajectories(traj, labels, mixed, space="mixture")
        assert h[0].max() >= m[0].max()


# ===========================================================================
# 4 — frame separation
# ===========================================================================

class TestFrameSeparation:
    """
    LN params loaded but never applied would show up as two identical
    numbers nobody questions. These tests make that impossible.
    """

    def setup_method(self):
        rng = np.random.default_rng(RNG_SEED)
        # Heterogeneous norms and a non-unit gamma, so the two frames
        # provably differ rather than coincidentally agreeing.
        base = rng.standard_normal((6, D_MODEL))
        scale = np.array([0.2, 1.0, 5.0, 0.5, 3.0, 1.5])[:, None]
        self.X = base * scale
        self.gamma = 1.0 + rng.standard_normal(D_MODEL) * 0.5
        self.beta = rng.standard_normal(D_MODEL) * 0.1

    def test_sphere_and_ln_distances_differ(self):
        from p5b_manifold_steering.p5b_distances import (
            _apply_frame, activation_distance_matrix,
        )
        S = activation_distance_matrix(_apply_frame(self.X, "sphere", None))
        L = activation_distance_matrix(
            _apply_frame(self.X, "ln", {"gamma": self.gamma, "beta": self.beta})
        )
        assert not np.allclose(S, L, atol=1e-3), \
            "LN frame produced sphere-frame numbers — params not applied"

    def test_sphere_gap_flags_the_norm_spread(self):
        from core.polar import sphere_gap
        gap = sphere_gap(self.X)
        assert gap["pearson_gap"] > 1e-3
        assert gap["norm_log_std"] > 0.5

    def test_ln_frame_without_params_raises(self):
        """Never silently fall back — the caller must record the fallback."""
        from p5b_manifold_steering.p5b_distances import _apply_frame
        with pytest.raises(ValueError, match="requires ln_params"):
            _apply_frame(self.X, "ln", None)

    def test_unknown_frame_raises(self):
        from p5b_manifold_steering.p5b_distances import _apply_frame
        with pytest.raises(ValueError, match="unknown frame"):
            _apply_frame(self.X, "spherical", None)

    def test_distance_matrices_well_formed(self):
        from p5b_manifold_steering.p5b_distances import (
            activation_distance_matrix, behavior_distance_matrix,
        )
        rng = np.random.default_rng(RNG_SEED)
        p = np.abs(rng.standard_normal((6, VOCAB))) + 1e-3
        p /= p.sum(axis=1, keepdims=True)
        for D in (activation_distance_matrix(self.X),
                  behavior_distance_matrix(p, "hellinger"),
                  behavior_distance_matrix(p, "sym_kl")):
            assert D.shape == (6, 6)
            npt.assert_allclose(np.diag(D), 0.0, atol=1e-10)
            npt.assert_allclose(D, D.T, atol=1e-8)
            assert np.all(D >= 0.0)

    def test_hellinger_bounded(self):
        from p5b_manifold_steering.p5b_distances import behavior_distance_matrix
        rng = np.random.default_rng(RNG_SEED)
        p = np.abs(rng.standard_normal((6, VOCAB))) + 1e-3
        p /= p.sum(axis=1, keepdims=True)
        assert behavior_distance_matrix(p, "hellinger").max() <= 1.0 + 1e-9

    def test_upper_triangle_matches_pair_indices(self):
        from p5b_manifold_steering.p5b_distances import (
            activation_distance_matrix, upper_triangle, pair_indices,
        )
        D = activation_distance_matrix(self.X)
        flat = upper_triangle(D)
        pairs = pair_indices(6)
        assert flat.shape[0] == pairs.shape[0] == 15
        for k, (i, j) in enumerate(pairs):
            assert flat[k] == pytest.approx(D[i, j])
