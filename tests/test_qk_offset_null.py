"""
tests/test_qk_offset_null.py — oracle tests for qk_offset_null.py.

The important tests here are the ones that construct data where the ORIGINAL
P6-I2 test would fire and the new one must not: an offset-distribution
difference with no content mechanism. If those pass, the confound is closed.
"""

import numpy as np
import pytest

from core.frames import FrameSpec, frame_of
from core.qk_offset_null import (
    P6_I2B_MIN_FIDELITY,
    evaluate_p6_i2b,
    head_a_frac_by_offset,
    match_pairs_by_offset,
    offset_logit_partition,
    offset_matched_null,
    offset_shuffled_null,
    pair_offsets,
    residualize_on_offset,
    rotary_null,
)
from core.rope import rope_rotation, rope_sa_fractions

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure

D_MODEL = 24
HEAD_SIZE = 8
ROT_NDIMS = 2
BASE = 10000.0
N = 12


def _w(seed=4):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(D_MODEL, HEAD_SIZE)), rng.normal(size=(D_MODEL, HEAD_SIZE))


def _X(seed=2, n=N):
    return np.random.default_rng(seed).normal(size=(n, D_MODEL))


class TestOffsetLogitPartition:

    def test_contributions_sum_to_the_logit(self):
        WQ, WK = _w()
        out = offset_logit_partition(_X(), WQ, WK, ROT_NDIMS, BASE)
        assert np.allclose(out["s_contrib"] + out["a_contrib"], out["logits"])

    def test_symmetry_structure(self):
        WQ, WK = _w()
        out = offset_logit_partition(_X(), WQ, WK, ROT_NDIMS, BASE)
        assert np.allclose(out["s_contrib"], out["s_contrib"].T)
        assert np.allclose(out["a_contrib"], -out["a_contrib"].T)

    def test_reduces_to_qk_decompose_without_rotary(self):
        """
        With rotary off this must equal the frozen GPT-2 computation
        x^T S x / x^T A x, so the non-rotary path is provably unchanged.
        """
        WQ, WK = _w()
        X = _X()
        M = WQ @ WK.T
        S, A = (M + M.T) / 2, (M - M.T) / 2
        out = offset_logit_partition(X, WQ, WK, 0, BASE)
        assert np.allclose(out["s_contrib"], X @ S @ X.T)
        assert np.allclose(out["a_contrib"], X @ A @ X.T)

    def test_rotary_changes_the_partition(self):
        WQ, WK = _w()
        X = _X()
        plain = offset_logit_partition(X, WQ, WK, 0, BASE)["a_frac_mat"]
        roped = offset_logit_partition(X, WQ, WK, ROT_NDIMS, BASE)["a_frac_mat"]
        assert not np.allclose(plain, roped, atol=1e-6)

    def test_a_frac_in_unit_interval(self):
        WQ, WK = _w()
        a = offset_logit_partition(_X(), WQ, WK, ROT_NDIMS, BASE)["a_frac_mat"]
        assert a.min() >= 0.0 and a.max() <= 1.0

    def test_diagonal_has_no_antisymmetry(self):
        """Delta = 0: forward and reverse are the same pair."""
        WQ, WK = _w()
        out = offset_logit_partition(_X(), WQ, WK, ROT_NDIMS, BASE)
        assert np.allclose(np.diag(out["a_contrib"]), 0.0, atol=1e-12)

    def test_offsets_matrix_convention(self):
        WQ, WK = _w()
        out = offset_logit_partition(_X(5), WQ, WK, ROT_NDIMS, BASE)
        assert out["offsets"][3, 1] == -2        # key 1 is 2 behind query 3

    def test_biases_change_the_result(self):
        WQ, WK = _w()
        X = _X()
        rng = np.random.default_rng(9)
        a = offset_logit_partition(X, WQ, WK, ROT_NDIMS, BASE)["a_frac_mat"]
        b = offset_logit_partition(X, WQ, WK, ROT_NDIMS, BASE,
                                   bq=rng.normal(size=HEAD_SIZE),
                                   bk=rng.normal(size=HEAD_SIZE))["a_frac_mat"]
        assert not np.allclose(a, b, atol=1e-6)


class TestWeightLevelByOffset:

    def test_matches_direct_computation(self):
        WQ, WK = _w()
        out = head_a_frac_by_offset([-3, -1, 0], WQ, WK, HEAD_SIZE,
                                    ROT_NDIMS, BASE) if False else \
            head_a_frac_by_offset(WQ, WK, [-3, -1, 0], HEAD_SIZE, ROT_NDIMS, BASE)
        for i, d in enumerate(out["offsets"]):
            R = rope_rotation(int(d), HEAD_SIZE, ROT_NDIMS, BASE)
            M = WQ @ R @ WK.T
            n2 = np.linalg.norm(M, "fro") ** 2
            want = np.linalg.norm((M - M.T) / 2, "fro") ** 2 / n2
            assert out["a_frac_weight"][i] == pytest.approx(want, rel=1e-9)

    def test_rotary_null_column_is_closed_form(self):
        WQ, WK = _w()
        out = head_a_frac_by_offset(WQ, WK, [-4, -2], HEAD_SIZE, ROT_NDIMS, BASE)
        for i, d in enumerate(out["offsets"]):
            want = rope_sa_fractions(int(d), HEAD_SIZE, ROT_NDIMS, BASE)["a_frac"]
            assert out["a_frac_rotary_null"][i] == pytest.approx(want)

    def test_deduplicates_offsets(self):
        WQ, WK = _w()
        out = head_a_frac_by_offset(WQ, WK, [-1, -1, -1, -2], HEAD_SIZE,
                                    ROT_NDIMS, BASE)
        assert list(out["offsets"]) == [-2, -1]


class TestOffsetBookkeeping:

    def test_offsets_are_non_positive(self):
        pairs = [(5, 2), (9, 8), (7, 0)]
        assert np.all(pair_offsets(pairs) < 0)

    def test_empty_pairs(self):
        assert pair_offsets([]).shape == (0,)

    def test_exact_matching(self):
        ind = [(5, 2), (9, 6)]           # both offset -3
        pool = [(8, 5), (4, 3)]          # -3 and -1
        m = match_pairs_by_offset(ind, pool)
        assert m["coverage"] == 1.0
        assert all(len(sel) == 1 for _, sel in m["matched"])

    def test_unmatched_reported_not_pooled(self):
        """
        The failure mode being prevented: silently comparing against whatever
        pool pairs exist when no offset match is available.
        """
        ind = [(9, 0)]                   # offset -9
        pool = [(2, 1)]                  # offset -1
        m = match_pairs_by_offset(ind, pool)
        assert m["coverage"] == 0.0
        assert m["unmatched_targets"] == [(9, 0)]

    def test_tolerance_widens_matching(self):
        ind = [(5, 2)]                   # -3
        pool = [(6, 2)]                  # -4
        assert match_pairs_by_offset(ind, pool, tolerance=0)["coverage"] == 0.0
        assert match_pairs_by_offset(ind, pool, tolerance=1)["coverage"] == 1.0


class TestResidualize:

    def test_removes_a_linear_trend(self):
        d = np.arange(-10, 0, dtype=float)
        v = 0.5 + 0.03 * d
        out = residualize_on_offset(v, d)
        assert np.allclose(out["residuals"], 0.0, atol=1e-12)
        assert out["r2"] == pytest.approx(1.0)

    def test_preserves_signal_orthogonal_to_offset(self):
        d = np.array([-4.0, -4, -3, -3, -2, -2])
        v = np.array([0.1, 0.5, 0.1, 0.5, 0.1, 0.5])   # group effect, no trend
        out = residualize_on_offset(v, d)
        assert np.ptp(out["residuals"]) > 0.3

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            residualize_on_offset(np.zeros(5), np.zeros(4))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            residualize_on_offset(np.zeros(2), np.zeros(2))


class TestNulls:

    def test_rotary_null_is_offset_dependent(self):
        n = rotary_null([-1, -5, -9], HEAD_SIZE, ROT_NDIMS, BASE)
        assert len(set(np.round(n, 12))) == 3

    def test_rotary_null_zero_at_zero_offset(self):
        assert rotary_null([0], HEAD_SIZE, ROT_NDIMS, BASE)[0] == 0.0

    def test_offset_matched_null_uses_same_offset_only(self):
        A = np.zeros((10, 10))
        A[5, 2] = 0.9                     # induction pair, offset -3
        A[8, 5] = 0.1                     # same-content, offset -3
        A[4, 3] = 0.99                    # same-content, offset -1 (must be ignored)
        out = offset_matched_null(A, [(5, 2)], [(8, 5), (4, 3)])
        assert out["n_used"] == 1
        assert out["deltas"][0] == pytest.approx(0.8)

    def test_shuffled_null_preserves_offset_marginal(self):
        rng = np.random.default_rng(0)
        A = rng.random((N, N))
        pairs = [(9, 6), (8, 2), (11, 4)]
        out = offset_shuffled_null(A, pairs, n_shuffles=50, seed=1)
        assert out["n_valid"] > 0
        assert 0.0 < out["p_value"] <= 1.0

    def test_shuffled_null_detects_a_real_joint_effect(self):
        A = np.full((N, N), 0.1)
        pairs = [(11, 0), (10, 2), (9, 3), (8, 5), (7, 6), (6, 1)]
        for q, k in pairs:
            A[q, k] = 0.95                    # elevated only at the true pairing
        out = offset_shuffled_null(A, pairs, n_shuffles=400, seed=3)
        assert out["underpowered"] is False
        assert out["p_value"] < 0.05

    def test_shuffled_null_flags_underpowered_pair_sets(self):
        """
        With k pairs the smallest achievable p is ~1/k!. Three pairs cannot
        reach p < 0.05 no matter what the data say, so the test must declare
        itself underpowered rather than returning a large p.
        """
        A = np.full((N, N), 0.1)
        pairs = [(9, 6), (8, 2), (11, 4)]
        for q, k in pairs:
            A[q, k] = 0.95
        out = offset_shuffled_null(A, pairs, n_shuffles=400, seed=3)
        assert out["underpowered"] is True
        assert out["min_achievable_p"] >= 1.0 / 6.0

    def test_shuffled_null_is_degenerate_at_a_single_offset(self):
        """
        Induction pairs usually all share one offset (the repeat period), and
        permuting identical offsets is a no-op. N3 must report that it has no
        power rather than returning p = 1.0, which reads as evidence of
        absence.
        """
        A = np.full((N, N), 0.1)
        pairs = [(9, 6), (8, 5), (11, 8)]     # all offset -3
        out = offset_shuffled_null(A, pairs, n_shuffles=200, seed=3)
        assert out["degenerate"] is True
        assert out["p_value"] is None

    def test_shuffled_null_ignores_a_pure_offset_effect(self):
        """Elevation that depends only on offset must NOT survive N3."""
        A = np.zeros((N, N))
        for q in range(N):
            for k in range(q):
                A[q, k] = 0.1 + 0.05 * (q - k)      # function of offset alone
        pairs = [(11, 0), (10, 2), (9, 3), (8, 5), (7, 6), (6, 1)]
        out = offset_shuffled_null(A, pairs, n_shuffles=400, seed=4)
        assert out["p_value"] > 0.05


class TestP6I2B:

    def _pairs_with_offset_confound(self):
        """
        The scenario that kills the ORIGINAL test: a_frac is a pure function
        of offset, induction pairs sit at large offsets, same-content pairs at
        small ones. The old pooled comparison fires; P6-I2b must not.
        """
        A = np.zeros((N, N))
        for q in range(N):
            for k in range(q):
                A[q, k] = 0.05 * (q - k)
        induction = [(11, 3), (10, 2), (9, 1)]          # offsets -8, -8, -8
        same = [(4, 3), (5, 4), (6, 5),                  # offsets -1
                (11, 3), (10, 2), (9, 1)]                # and matched at -8
        return A, induction, same

    def test_old_pooled_comparison_would_fire(self):
        A, ind, same = self._pairs_with_offset_confound()
        pooled_ind = np.mean([A[q, k] for q, k in ind])
        pooled_same = np.mean([A[q, k] for q, k in same])
        assert pooled_ind - pooled_same > 0.05    # the confounded "result"

    def test_offset_matched_comparison_does_not(self):
        A, ind, same = self._pairs_with_offset_confound()
        out = evaluate_p6_i2b(A, ind, same, HEAD_SIZE, ROT_NDIMS, BASE)
        assert out["verdict"] == "null"
        assert out["delta_vs_n2"] == pytest.approx(0.0, abs=1e-12)

    def test_genuine_content_effect_is_detected(self):
        A = np.zeros((N, N))
        for q in range(N):
            for k in range(q):
                A[q, k] = 0.05 * (q - k)
        ind = [(11, 3), (10, 2), (9, 1)]
        for q, k in ind:
            A[q, k] += 0.3                      # content effect on top of offset
        same = [(11, 3), (10, 2), (9, 1)]
        same = [(9, 1)] * 0 + [(10, 2), (11, 3), (9, 1)]
        # distinct same-content pairs at the same offsets
        same = [(8, 0), (9, 1), (10, 2)]
        out = evaluate_p6_i2b(A, ind, same, HEAD_SIZE, ROT_NDIMS, BASE)
        assert out["delta_vs_n2"] > 0.05

    def test_insufficient_pairs_is_its_own_verdict(self):
        A = np.zeros((N, N))
        out = evaluate_p6_i2b(A, [(5, 2)], [(4, 1)], HEAD_SIZE, ROT_NDIMS, BASE)
        assert out["verdict"] == "insufficient_pairs"

    def test_low_offset_coverage_is_reported_not_pooled(self):
        A = np.random.default_rng(0).random((N, N))
        ind = [(11, 0), (10, 0), (9, 0)]        # offsets -11, -10, -9
        same = [(4, 3), (5, 4), (6, 5)]         # offset -1: no overlap
        out = evaluate_p6_i2b(A, ind, same, HEAD_SIZE, ROT_NDIMS, BASE)
        assert out["verdict"] == "insufficient_offset_coverage"

    def test_low_fidelity_blocks_any_verdict(self):
        """
        A head whose weight-space prediction does not track its real logits
        cannot support a claim about its bilinear, however good the statistics.
        """
        A, ind, same = self._pairs_with_offset_confound()
        out = evaluate_p6_i2b(A, ind, same, HEAD_SIZE, ROT_NDIMS, BASE,
                              fidelity={"pearson": 0.4})
        assert out["verdict"] == "unverifiable_low_fidelity"

    def test_high_fidelity_allows_a_verdict(self):
        A, ind, same = self._pairs_with_offset_confound()
        out = evaluate_p6_i2b(A, ind, same, HEAD_SIZE, ROT_NDIMS, BASE,
                              fidelity={"pearson": 0.999})
        assert out["verdict"] in ("null", "supported")

    def test_rotary_null_is_reported(self):
        A, ind, same = self._pairs_with_offset_confound()
        out = evaluate_p6_i2b(A, ind, same, HEAD_SIZE, ROT_NDIMS, BASE)
        assert "rotary_null_mean" in out and out["rotary_null_mean"] >= 0.0

    def test_result_carries_a_frame(self):
        A, ind, same = self._pairs_with_offset_confound()
        spec = FrameSpec(kind="ln_attn", layer_idx=3, reader_block=4,
                         model_rev="pythia-1.4b@step143000", rope_applied=True)
        out = evaluate_p6_i2b(A, ind, same, HEAD_SIZE, ROT_NDIMS, BASE, frame=spec)
        assert frame_of(out).rope_applied is True
