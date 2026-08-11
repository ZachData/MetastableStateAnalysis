"""
tests/test_phase2b_head_circuits.py

Pins the four identities the per-head redesign rests on, so if any of them
stops holding it is a test failure rather than a silently wrong spectrum:

  1. eig(W_O W_V) \\ {0} == eig(W_V W_O)                     [d_head core]
  2. S = B_S C and A = B_A C exactly, rank <= 2*d_head       [factored parts]
  3. sigma * J_LN is an orthogonal projector of rank d-2     [Block 4]
  4. dimension fractions collapse with rank; energy fractions do not
"""

import unittest

import numpy as np

from p2b_imaginary import head_circuits as hc


D, K, N_HEADS = 96, 8, 6


def make_head(seed=0, d=D, k=K, scale=1.0):
    rng = np.random.default_rng(seed)
    W_O = scale * rng.normal(size=(d, k)) / np.sqrt(d)
    W_V = scale * rng.normal(size=(k, d)) / np.sqrt(d)
    return W_O, W_V


def make_layer(seed=0, d=D, k=K, n_heads=N_HEADS):
    return [make_head(seed * 100 + h, d, k)[0] @ make_head(seed * 100 + h, d, k)[1]
            for h in range(n_heads)]


# ---------------------------------------------------------------------------
# 1. The core carries the whole nonzero spectrum
# ---------------------------------------------------------------------------

class TestHeadCoreCarriesTheSpectrum(unittest.TestCase):

    def test_nonzero_eigenvalues_match_exactly(self):
        for seed in range(4):
            W_O, W_V = make_head(seed)
            big = np.linalg.eigvals(W_O @ W_V)
            big = np.sort_complex(np.round(big[np.argsort(-np.abs(big))][:K], 10))
            core = np.sort_complex(np.round(
                np.linalg.eigvals(hc.head_core(W_O, W_V)), 10))
            np.testing.assert_allclose(np.real(big), np.real(core), atol=1e-8)
            np.testing.assert_allclose(np.imag(big), np.imag(core), atol=1e-8)

    def test_the_dense_head_is_rank_deficient(self):
        W_O, W_V = make_head(0)
        self.assertEqual(np.linalg.matrix_rank(W_O @ W_V), K)
        self.assertLess(K, D)

    def test_core_is_d_head_squared_not_d_model_squared(self):
        W_O, W_V = make_head(0)
        self.assertEqual(hc.head_core(W_O, W_V).shape, (K, K))


# ---------------------------------------------------------------------------
# 2. S and A stay factored
# ---------------------------------------------------------------------------

class TestFactoredSymmetricParts(unittest.TestCase):

    def test_factors_reproduce_s_and_a_exactly(self):
        for seed in range(4):
            W_O, W_V = make_head(seed)
            OV = W_O @ W_V
            f = hc.sym_antisym_factors(W_O, W_V)
            S = (OV + OV.T) / 2
            A = (OV - OV.T) / 2
            np.testing.assert_allclose(f["B_S"] @ f["C"], S, atol=1e-12)
            np.testing.assert_allclose(f["B_A"] @ f["C"], A, atol=1e-12)

    def test_factor_shapes_are_2k_not_d(self):
        W_O, W_V = make_head(0)
        f = hc.sym_antisym_factors(W_O, W_V)
        self.assertEqual(f["B_S"].shape, (D, 2 * K))
        self.assertEqual(f["C"].shape, (2 * K, D))
        self.assertLess(2 * K, D)

    def test_rank_of_s_is_at_most_2k(self):
        W_O, W_V = make_head(0)
        OV = W_O @ W_V
        self.assertLessEqual(np.linalg.matrix_rank((OV + OV.T) / 2), 2 * K)

    def test_s_and_a_share_the_right_factor(self):
        """One extra (n, 2k) matmul swaps S for V, not a second projection."""
        W_O, W_V = make_head(0)
        f = hc.sym_antisym_factors(W_O, W_V)
        self.assertIs(f["C"], f["C"])
        np.testing.assert_allclose(f["B_S"] - f["B_A"],
                                   np.concatenate(
                                       [np.zeros((D, K)), 2 * W_V.T], axis=1)
                                   / np.sqrt(2), atol=1e-14)

    def test_apply_factored_matches_the_dense_product(self):
        rng = np.random.default_rng(1)
        W_O, W_V = make_head(0)
        f = hc.sym_antisym_factors(W_O, W_V)
        Y = rng.normal(size=(20, D))
        OV = W_O @ W_V
        np.testing.assert_allclose(hc.apply_factored(Y, f["B_S"], f["C"]),
                                   Y @ ((OV + OV.T) / 2), atol=1e-10)

    def test_factored_spectrum_matches_the_dense_nonzero_spectrum(self):
        W_O, W_V = make_head(0)
        f = hc.sym_antisym_factors(W_O, W_V)
        S = f["B_S"] @ f["C"]
        dense = np.linalg.eigvals(S)
        dense = np.sort(np.abs(dense))[-2 * K:]
        small = np.sort(np.abs(hc.factored_spectrum(f["B_S"], f["C"])))
        np.testing.assert_allclose(dense, small, atol=1e-9)

    def test_factored_spectrum_is_a_2k_problem(self):
        W_O, W_V = make_head(0)
        f = hc.sym_antisym_factors(W_O, W_V)
        self.assertEqual(hc.factored_spectrum(f["B_S"], f["C"]).shape, (2 * K,))


# ---------------------------------------------------------------------------
# 3. Rank and the two kinds of fraction
# ---------------------------------------------------------------------------

class TestRankArtifact(unittest.TestCase):
    """
    A rank-k head embedded in d dimensions has d-k zero eigenvalues, and a
    Schur partition counts every one as a real block. Which fractions that
    corrupts is not obvious, and getting it wrong in either direction is a
    reporting error.
    """

    def test_dimension_fraction_collapses_with_ambient_rank(self):
        W_O, W_V = make_head(0)
        s = hc.head_spectrum(W_O, W_V)
        self.assertGreater(s["dim_complex_fraction_core"],
                           5 * s["dim_complex_fraction_ambient"])

    def test_energy_fraction_is_rank_invariant(self):
        """
        |0|^2 = 0, so the zeros add nothing to numerator OR denominator. The
        ENERGY fraction survives the embedding; only the DIMENSION fraction
        does not. Reporting both as 'rotational fraction' hid this.
        """
        from p2b_imaginary.rotational_schur import (
            complex_energy_fraction, extract_schur_blocks,
        )
        W_O, W_V = make_head(0)
        core_val = hc.head_spectrum(W_O, W_V)["complex_energy_fraction_core"]
        ambient = complex_energy_fraction(
            extract_schur_blocks(W_O @ W_V))["complex_energy_fraction"]
        self.assertAlmostEqual(core_val, ambient, places=6)
        # So the rank argument does NOT overturn the published 84-97.5%
        # figure, which is an energy fraction. The shared-attention argument
        # does; this one only fixes the dimension fractions and the cost.
        self.assertNotIn("complex_energy_fraction_ambient",
                         hc.head_spectrum(W_O, W_V))

    def test_frobenius_fraction_does_not_depend_on_rank_either(self):
        W_O, W_V = make_head(0)
        OV = W_O @ W_V
        s = hc.head_spectrum(W_O, W_V)
        want = (np.linalg.norm((OV - OV.T) / 2, "fro") ** 2 /
                np.linalg.norm(OV, "fro") ** 2)
        self.assertAlmostEqual(s["rotational_frobenius_fraction"], want, places=8)


# ---------------------------------------------------------------------------
# Factor recovery
# ---------------------------------------------------------------------------

class TestFactorRecovery(unittest.TestCase):

    def test_recovered_factors_reproduce_the_dense_matrix(self):
        W_O, W_V = make_head(0)
        OV = W_O @ W_V
        f = hc.factor_from_dense(OV, d_head=K)
        np.testing.assert_allclose(f["W_O"] @ f["W_V"], OV, atol=1e-10)

    def test_recovered_rank_matches(self):
        W_O, W_V = make_head(0)
        f = hc.factor_from_dense(W_O @ W_V, d_head=K)
        self.assertEqual(f["rank"], K)
        self.assertEqual(f["W_O"].shape, (D, K))

    def test_spectrum_is_invariant_to_the_factorisation_basis(self):
        """
        The factorisation is not unique (W_O M, M^-1 W_V works for any
        invertible M). Nothing reported here may depend on which is chosen.
        """
        rng = np.random.default_rng(2)
        W_O, W_V = make_head(0)
        M = rng.normal(size=(K, K))
        a = hc.head_spectrum(W_O, W_V)
        b = hc.head_spectrum(W_O @ M, np.linalg.inv(M) @ W_V)
        for key in ("complex_energy_fraction_core", "rotational_frobenius_fraction",
                    "spectral_radius", "eigenvalue_energy"):
            self.assertAlmostEqual(a[key], b[key], places=6, msg=key)


# ---------------------------------------------------------------------------
# Layer level and the summed counterfactual
# ---------------------------------------------------------------------------

class TestSummedVsPerHead(unittest.TestCase):

    def test_head_spread_is_reported(self):
        """
        Sixteen heads with fractions from 0.1 to 0.9 sum to one middling
        number, and the summed object is what Phase 2b has been measuring.
        """
        res = hc.layer_head_spectra(make_layer(0), d_head=K)
        self.assertEqual(res["n_heads"], N_HEADS)
        self.assertIn("complex_energy_fraction_std", res)
        self.assertLessEqual(res["complex_energy_fraction_min"],
                             res["complex_energy_fraction_mean"])

    def test_summed_carries_its_counterfactual_in_the_output(self):
        res = hc.summed_vs_per_head(make_layer(0), d_head=K)
        self.assertIn("shares an attention pattern", res["summed"]["caveat"])

    def test_gap_and_agreement_are_reported(self):
        res = hc.summed_vs_per_head(make_layer(0), d_head=K)
        self.assertIn("gap", res)
        self.assertGreaterEqual(res["head_agreement"], 0.0)
        self.assertLessEqual(res["head_agreement"], 1.0)

    def test_identical_heads_agree_with_the_sum(self):
        """Sanity: the counterfactual holds exactly when heads are identical."""
        W_O, W_V = make_head(0)
        res = hc.summed_vs_per_head([W_O @ W_V] * N_HEADS, d_head=K)
        self.assertAlmostEqual(res["gap"], 0.0, places=6)
        self.assertAlmostEqual(res["head_agreement"], 1.0)

    def test_dimension_fraction_of_the_sum_is_not_the_head_mean(self):
        """
        For standard MHA n_heads*d_head == d_model, so the SUM is generically
        full rank while each head is rank d_head. The dimension fractions are
        therefore not comparable even when the energy fractions are.
        """
        heads = [make_head(h)[0] @ make_head(h)[1] for h in range(D // K)]
        res = hc.summed_vs_per_head(heads, d_head=K)
        self.assertGreater(res["summed"]["dim_complex_fraction"],
                           2 * (2 * K / D))


# ---------------------------------------------------------------------------
# 4. The LayerNorm Jacobian
# ---------------------------------------------------------------------------

def ln_jacobian_dense(x, eps=1e-5):
    """The textbook form, for comparison against the projector identity."""
    d = x.size
    mu = x.mean()
    var = ((x - mu) ** 2).mean() + eps
    sig = np.sqrt(var)
    xh = (x - mu) / sig
    J = (np.eye(d) - np.ones((d, d)) / d - np.outer(xh, xh) / d) / sig
    return J, sig, xh


class TestLayerNormJacobianIsAScaledProjector(unittest.TestCase):
    """
    `sigma * J_LN` is an ORTHOGONAL PROJECTOR of rank d-2 onto the complement
    of span{1, x_hat}. Not an approximation: both `1/d` terms are projectors
    because ||1||^2 = ||x_hat||^2 = d.

    Consequences for Block 4, which the shipped version did not use:
      - `1/sigma` is a pure scale and cannot change any angle or fraction.
      - The token-dependent content is exactly ONE rank-1 direction.
      - Everything else is `diag(gamma)`, which is token-independent: one
        eigendecomposition per LAYER, not one per token per layer.
    """

    def test_it_is_idempotent_symmetric_and_rank_d_minus_2(self):
        rng = np.random.default_rng(3)
        for _ in range(3):
            x = rng.normal(size=D) * rng.uniform(0.1, 10)
            J, sig, xh = ln_jacobian_dense(x)
            P = sig * J
            np.testing.assert_allclose(P @ P, P, atol=1e-5)
            np.testing.assert_allclose(P, P.T, atol=1e-14)
            self.assertAlmostEqual(float(np.trace(P)), D - 2.0, places=3)

    def test_it_annihilates_the_ones_vector_and_x_hat(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=D) * 3.0
        J, sig, xh = ln_jacobian_dense(x)
        P = sig * J
        self.assertLess(float(np.abs(P @ np.ones(D)).max()), 1e-12)
        self.assertLess(float(np.abs(P @ xh).max()), 1e-4)

    def test_the_curvature_regressor_is_identically_one(self):
        """
        `ln_curvature = ||x-mu||^2 / (d * sigma^2)` with
        `sigma^2 = ||x-mu||^2 / d` is 1 by algebra. The shipped Block 4
        regressed inflation against it, so Pearson r was always NaN.
        """
        rng = np.random.default_rng(5)
        for _ in range(4):
            x = rng.normal(size=D) * rng.uniform(0.01, 100)
            mu = x.mean()
            var = ((x - mu) ** 2).mean() + 1e-5
            self.assertAlmostEqual(
                float(np.sum((x - mu) ** 2) / (D * var)), 1.0, places=4)

    def test_layernorm_barely_moves_the_complex_fraction(self):
        """
        The shipped `H2_SUPPORTED` threshold was `inflation > 1.5` against a
        base fraction near 0.98, so it was unreachable. Removing 2 of d
        dimensions and applying a diagonal gain moves it by percent, not by
        50 percent.
        """
        rng = np.random.default_rng(6)
        V = rng.normal(size=(D, D)) / np.sqrt(D)
        x = rng.normal(size=D)
        _, sig, _ = ln_jacobian_dense(x)
        P = sig * ln_jacobian_dense(x)[0]
        gam = np.exp(rng.normal(scale=0.4, size=D))

        def cf(M):
            e = np.linalg.eigvals(M)
            m = np.abs(e.imag) > 0.01 * (np.abs(e.real) + 1e-12)
            return float(np.sum(np.abs(e[m]) ** 2) / np.sum(np.abs(e) ** 2))

        base = cf(V)
        with_ln = cf(np.diag(gam) @ P @ V)
        self.assertGreater(base, 0.8)
        self.assertLess(abs(with_ln / base - 1.0), 0.15)
        self.assertLess(with_ln / base, 1.5)      # the shipped threshold


if __name__ == "__main__":
    unittest.main(verbosity=2)
