"""
tests/test_isometric_path_sweep.py — core/isometric_path.py's pure-math
primitives, §2.5's isometric path construction (MATH_SPECTRAL_OT.md §2.5).

polar_frame, check_refusal and build_M_t are pure numpy -- no torch, no
model -- and carry the properties §2.5.2 proves: M(0) = M, M(1) = M^T,
M(1/2) symmetric PSD, singular values preserved exactly at every t. These
are checked directly (enumeration/construction, not just asserted),
matching CLAUDE.md's rule that a closed-form derivation belongs checked
mechanically. The real-model sweep
(tools/run/isometric_path_sweep.py::run_sweep, run against pythia-410m's
L7H8) is validated by the committed run instead
(data/analysis/isometric_path_L7H8_step4000.json) -- no smoke tier here
since nothing in this file needs torch.
"""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from core.isometric_path import (
    polar_frame, check_refusal, build_M_t,
)

pytestmark = pytest.mark.pure


def _random_svd_factors(d, k, seed):
    """Rank-k factors of a d x d matrix, TRUNCATED to k modes after SVD --
    M = A @ B is square (d, d) here (A is (d,k), B is (k,d)), so
    full_matrices=False alone returns all d singular vectors, not just the
    k real ones; the other d-k are a numerically-zero tail whose singular
    VECTORS are ill-defined (exactly the mechanism that refuses for real
    L7H8 -- see run_sweep's docstring). Slicing to :k avoids feeding that
    tail's arbitrary directions into the tests below, matching what
    run_sweep's own `rank` truncation does for the real head."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, k))
    B = rng.standard_normal((k, d))
    M = A @ B
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return M, U[:, :k], S[:k], Vt.T[:, :k]


class TestPolarFrame:

    def test_returns_orthonormal_columns(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((20, 5))
        gamma = polar_frame(Y)
        assert gamma is not None
        npt.assert_allclose(gamma.T @ gamma, np.eye(5), atol=1e-8)

    def test_identity_on_an_already_orthonormal_input(self):
        rng = np.random.default_rng(1)
        Q, _ = np.linalg.qr(rng.standard_normal((20, 5)))
        gamma = polar_frame(Q)
        npt.assert_allclose(gamma, Q, atol=1e-8)

    def test_none_on_singular_input(self):
        Y = np.zeros((10, 3))
        assert polar_frame(Y) is None


class TestCheckRefusal:

    def test_ok_for_well_separated_frames(self):
        d, k = 20, 5
        _, U, _, V = _random_svd_factors(d, k, seed=2)
        result = check_refusal(U, V, n_grid=51)
        assert result["ok"] is True
        assert result["min_sigma_min"] > 0

    def test_fails_when_a_mode_is_antiparallel(self):
        """Construct U, V that share one exactly-antiparallel column pair
        -- Y(0.5) must then be singular in that direction."""
        d, k = 10, 3
        rng = np.random.default_rng(3)
        Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
        U = Q.copy()
        V = Q.copy()
        V[:, 0] = -U[:, 0]  # exactly antiparallel in mode 0
        result = check_refusal(U, V, n_grid=51)
        assert result["ok"] is False
        npt.assert_allclose(result["min_sigma_min"], 0.0, atol=1e-8)


class TestBuildMT:

    def test_t_zero_reproduces_m(self):
        M, U, S, V = _random_svd_factors(20, 5, seed=4)
        A_t, B_t = build_M_t(U, S, V, 0.0)
        npt.assert_allclose(A_t @ B_t, M, atol=1e-8)

    def test_t_one_is_the_transpose(self):
        M, U, S, V = _random_svd_factors(20, 5, seed=5)
        A_t, B_t = build_M_t(U, S, V, 1.0)
        npt.assert_allclose(A_t @ B_t, M.T, atol=1e-8)

    def test_t_half_is_symmetric_psd(self):
        _, U, S, V = _random_svd_factors(20, 5, seed=6)
        A_t, B_t = build_M_t(U, S, V, 0.5)
        M_half = A_t @ B_t
        npt.assert_allclose(M_half, M_half.T, atol=1e-8)
        eigvals = np.linalg.eigvals(M_half)
        npt.assert_allclose(eigvals.imag, 0.0, atol=1e-8)
        assert np.all(eigvals.real >= -1e-8)

    def test_singular_values_preserved_at_every_t(self):
        """The whole point of the construction (§2.5.2): an exact
        isometry, checked at a grid of t rather than only the three
        named corners. A_t @ B_t is (d, d) and genuinely rank-k, so its
        SVD returns d singular values with a numerically-zero (d-k) tail
        -- only the top k are compared against S."""
        k = 6
        _, U, S, V = _random_svd_factors(20, k, seed=7)
        for t in np.linspace(0.0, 1.0, 9):
            A_t, B_t = build_M_t(U, S, V, float(t))
            sv = np.linalg.svd(A_t @ B_t, compute_uv=False)
            npt.assert_allclose(sorted(sv[:k]), sorted(S), atol=1e-8)

    def test_rank_preserved_at_every_t(self):
        _, U, S, V = _random_svd_factors(20, 6, seed=8)
        for t in np.linspace(0.0, 1.0, 9):
            A_t, B_t = build_M_t(U, S, V, float(t))
            assert np.linalg.matrix_rank(A_t @ B_t) == 6

    def test_none_when_the_frame_is_refused(self):
        d, k = 10, 3
        rng = np.random.default_rng(9)
        Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
        U = Q.copy()
        V = Q.copy()
        V[:, 0] = -U[:, 0]
        S = np.array([1.0, 0.5, 0.25])
        result = build_M_t(U, S, V, 0.5)
        assert result is None
