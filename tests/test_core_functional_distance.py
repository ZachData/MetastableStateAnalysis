"""
tests/test_core_functional_distance.py — Oracle tests for
core/functional_distance.py (frames item 4).

Pure tier. The KL matmul identity is verified against a literal per-pair
sum (the definition), chunked against unchunked, and the ARI against
hand-computed contingency values — including the exact 4/7 example
worked in the design discussion.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.functional_distance import (
    logprobs_from_logits,
    logprobs_from_probs,
    kl_matrix,
    kl_matrix_from_probs,
    sym_kl,
    functional_clusters,
    adjusted_rand_index,
    frame_agreement,
)

_rng = np.random.default_rng(2024)


def _rand_logprobs(n=10, V=50):
    return logprobs_from_logits(_rng.normal(size=(n, V)) * 2.0)


def _kl_pair_loop(L):
    """Literal definition: KL(p_i || p_j) = sum_v p_i (log p_i - log p_j)."""
    P = np.exp(L)
    n = L.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = float((P[i] * (L[i] - L[j])).sum())
    np.fill_diagonal(K, 0.0)
    return np.maximum(K, 0.0)


# ---------------------------------------------------------------------------
# Log-prob preparation
# ---------------------------------------------------------------------------

class TestLogprobPrep:
    def test_log_softmax_normalized(self):
        L = logprobs_from_logits(_rng.normal(size=(5, 40)) * 10.0)
        np.testing.assert_allclose(np.exp(L).sum(axis=-1), 1.0, rtol=1e-12)

    def test_stable_under_huge_logits(self):
        L = logprobs_from_logits(np.array([[1e4, 1e4 - 5.0, 0.0]]))
        assert np.all(np.isfinite(L))
        np.testing.assert_allclose(np.exp(L).sum(), 1.0, rtol=1e-12)

    def test_probs_roundtrip(self):
        """logits -> softmax -> logprobs_from_probs must agree with the
        direct log-softmax (logit_cache stores probabilities, so this
        roundtrip is the actual production path)."""
        logits = _rng.normal(size=(6, 30)) * 3.0
        L_direct = logprobs_from_logits(logits)
        P = np.exp(L_direct)
        L_round = logprobs_from_probs(P)
        np.testing.assert_allclose(L_round, L_direct, atol=1e-9)


# ---------------------------------------------------------------------------
# KL matrix
# ---------------------------------------------------------------------------

class TestKlMatrix:
    def test_matches_pair_loop(self):
        """The matmul identity vs the literal definition."""
        L = _rand_logprobs(n=8, V=37)
        np.testing.assert_allclose(kl_matrix(L), _kl_pair_loop(L),
                                   rtol=1e-10, atol=1e-12)

    def test_matches_scipy_entropy(self):
        """Independent oracle: scipy.stats.entropy(p, q) is KL(p||q)."""
        from scipy.stats import entropy
        L = _rand_logprobs(n=6, V=25)
        P = np.exp(L)
        K = kl_matrix(L)
        for i in range(6):
            for j in range(6):
                if i != j:
                    assert K[i, j] == pytest.approx(
                        float(entropy(P[i], P[j])), rel=1e-8
                    )

    def test_chunked_equals_unchunked(self):
        L = _rand_logprobs(n=11, V=40)
        np.testing.assert_allclose(kl_matrix(L, chunk=3), kl_matrix(L),
                                   rtol=1e-12)

    def test_diagonal_zero_and_nonnegative(self):
        K = kl_matrix(_rand_logprobs())
        assert np.all(np.diag(K) == 0.0)
        assert np.all(K >= 0.0)

    def test_identical_rows_zero_distance(self):
        L1 = _rand_logprobs(n=1, V=30)
        L = np.repeat(L1, 4, axis=0)
        np.testing.assert_allclose(kl_matrix(L), 0.0, atol=1e-10)

    def test_asymmetry_is_real(self):
        """KL is asymmetric; sym_kl symmetrizes exactly."""
        K = kl_matrix(_rand_logprobs(n=6, V=20))
        assert np.abs(K - K.T).max() > 1e-6
        S = sym_kl(K)
        np.testing.assert_allclose(S, S.T, atol=0.0)
        np.testing.assert_allclose(S, 0.5 * (K + K.T), rtol=1e-15)

    def test_from_probs_path(self):
        L = _rand_logprobs(n=5, V=20)
        P = np.exp(L).astype(np.float32)          # logit_cache stores float32
        np.testing.assert_allclose(kl_matrix_from_probs(P), kl_matrix(L),
                                   atol=1e-5)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            kl_matrix(np.zeros((2, 3, 4)))


# ---------------------------------------------------------------------------
# Functional clustering
# ---------------------------------------------------------------------------

class TestFunctionalClusters:
    def _two_group_D(self, n_per=6, sep=8.0):
        """Synthetic sym divergence: two tight groups far apart."""
        n = 2 * n_per
        D = np.full((n, n), sep)
        D[:n_per, :n_per] = 0.3
        D[n_per:, n_per:] = 0.3
        D += _rng.uniform(0, 0.05, size=(n, n))
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        return D

    def test_recovers_planted_groups(self):
        try:
            labels = functional_clusters(self._two_group_D(), min_cluster_size=3)
        except ImportError:
            pytest.skip("no HDBSCAN implementation available")
        n_per = 6
        # Both groups internally uniform, and distinct from each other
        assert len(set(labels[:n_per])) == 1
        assert len(set(labels[n_per:])) == 1
        assert labels[0] != labels[n_per]
        assert labels[0] != -1 and labels[n_per] != -1

    def test_asymmetric_input_rejected(self):
        D = self._two_group_D()
        D[0, 1] += 1.0                             # break symmetry
        with pytest.raises(ValueError):
            functional_clusters(D)

    def test_nonsquare_rejected(self):
        with pytest.raises(ValueError):
            functional_clusters(np.zeros((3, 4)))

    def test_tiny_n_all_noise(self):
        labels = functional_clusters(np.zeros((2, 2)), min_cluster_size=3)
        assert list(labels) == [-1, -1]


# ---------------------------------------------------------------------------
# ARI — hand-computed oracles
# ---------------------------------------------------------------------------

class TestAdjustedRandIndex:
    def test_identical_labelings(self):
        lab = np.array([0, 0, 1, 1, 2, 2, -1])
        assert adjusted_rand_index(lab, lab) == pytest.approx(1.0)

    def test_relabeling_invariance(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([5, 5, 9, 9, 1, 1])           # same partition, new names
        assert adjusted_rand_index(a, b) == pytest.approx(1.0)

    def test_hand_computed_four_sevenths(self):
        """a = [0,0,1,1], b = [0,0,1,2]. Contingency: cells (2,1,1);
        index = 1, sum_a = 2, sum_b = 1, total = C(4,2) = 6,
        expected = 1/3, max = 3/2 -> ARI = (1 - 1/3)/(3/2 - 1/3) = 4/7."""
        a = np.array([0, 0, 1, 1])
        b = np.array([0, 0, 1, 2])
        assert adjusted_rand_index(a, b) == pytest.approx(4.0 / 7.0)

    def test_matches_sklearn(self):
        """Independent oracle where sklearn is available."""
        try:
            from sklearn.metrics import adjusted_rand_score
        except ImportError:
            pytest.skip("sklearn unavailable")
        for _ in range(10):
            a = _rng.integers(-1, 3, size=30)
            b = _rng.integers(-1, 3, size=30)
            assert adjusted_rand_index(a, b) == pytest.approx(
                float(adjusted_rand_score(a, b)), abs=1e-12
            )

    def test_ignore_noise(self):
        """a's noise points are split across b's two clusters: with
        ignore_noise the survivors agree perfectly (ARI 1.0); without,
        the -1s form a class b disagrees with (sklearn-verified 8/33).
        NB the example must split the noise across b's clusters — if the
        -1s form matching classes in both partitions, ARI is legitimately
        1.0 either way."""
        a = np.array([0, 0, 1, 1, -1, -1])
        b = np.array([7, 7, 3, 3, 7, 3])
        assert adjusted_rand_index(a, b, ignore_noise=True) == pytest.approx(1.0)
        full = adjusted_rand_index(a, b, ignore_noise=False)
        assert full == pytest.approx(8.0 / 33.0)

    def test_trivial_partitions(self):
        assert adjusted_rand_index(np.zeros(5, int), np.zeros(5, int)) == 1.0

    def test_too_few_points_nan(self):
        assert np.isnan(adjusted_rand_index(np.array([0]), np.array([0])))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            adjusted_rand_index(np.zeros(3, int), np.zeros(4, int))


class TestFrameAgreement:
    def test_three_frame_table_row(self):
        sphere = np.array([0, 0, 0, 1, 1, 1])
        ln = np.array([0, 0, 0, 1, 1, 1])          # agrees with sphere
        fn = np.array([0, 0, 1, 1, 1, 1])          # breaks from both
        out = frame_agreement({"sphere": sphere, "ln": ln, "functional": fn})
        assert set(out) == {"ln|sphere", "functional|ln", "functional|sphere"}
        assert out["ln|sphere"] == pytest.approx(1.0)
        assert out["functional|sphere"] < 1.0
        assert out["functional|ln"] == pytest.approx(out["functional|sphere"])
