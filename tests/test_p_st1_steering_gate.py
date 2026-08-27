"""
tests/test_p_st1_steering_gate.py — P-ST1's steering-sign construction.

`p7_motifs/steering_gate.py` is H-BRIDGE's cheapest entry and the only
registered bridge prediction where the particle and standard accounts make
INCOMPATIBLE rather than merely different predictions. Its whole intervention
is exact linear algebra — adding alpha*v to every row of an activation block
and recomputing effective rank — so unlike CLAIM-C's gate or CLAIM-B's, this
one can be exercised end to end here on populations with a planted answer.

The four assertions worth knowing about before reading the file:

`TestSteeringIsAMeanEffect` pins the algebra that decided
`DEBIAS_BASELINE_MEAN`: re-centring after injection annihilates the
intervention exactly, so a mean-removing effective-rank pipeline would make
every dER identically zero.

`TestSignIsEvenInV` pins the reason `ER_MODE` is "raw" and not the "normed"
CLAIM-C precedent would suggest. With the baseline mean removed the first-order
Gram term vanishes and raw dER is even in v; L2 row-normalization is not linear
and puts an odd term back, so `normed` answers differently for v and -v — and a
steering DIRECTION and its negation are the same object. The two modes are
indistinguishable at the working alpha, which is exactly how the wrong one
nearly shipped.

`TestTheFloorIsSetByInformativePairs` pins the arithmetic that retired the
registered null's power: a pair whose arms move effective rank the same way
contributes D = 0, and a zero contributes identically to the observed sum and
to every null pattern.

`TestAllThreeVerdictsCanFire` is `POPPER_PLAN.md` 6i's requirement — a verdict
branch nothing can trigger is 6h's "arm incapable of failing" wearing a
different hat — and INVERTS is the branch that would enter the ledger.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pytest.ini [pytest] markers.
pytestmark = pytest.mark.pure

from core.metrics import effective_rank
from p7_motifs.steering_gate import (
    ALPHA_IS_PLACED,
    ALPHA_SPREAD_FRACTION,
    ALTERNATIVE,
    DEBIAS_BASELINE_MEAN,
    ER_MODE,
    NULL_FAMILY,
    N_SUBSPACE_DRAWS,
    P_ST1_UNIT,
    RECIPROCAL_ALTERNATIVE,
    attainable_floor,
    delta_effective_rank,
    draw_pairs,
    draw_unit_in_subspace,
    gate_verdict,
    label_permutation_diagnostic,
    min_informative_pairs,
    null_distribution,
    null_sums,
    p_from_distribution,
    p_value_p_st1,
    pair_statistic,
    population_mean_ratio,
    population_spread,
    random_orthogonal_subspace_pair,
    compact_basis,
    occupancy,
    resplit_pair,
    subspace_rank,
    union_basis,
)

#: Small on purpose. Every gate call here draws n_draws x n_pairs effective
#: ranks, and `scripts/check.sh`'s own header says a gate people wait on is a
#: gate people route around -- so the fixtures are the smallest geometry that
#: still separates the planted families, and the null draws are the fewest
#: whose floor still clears alpha. The calibration artifact carries the full
#: geometry and the shipped draw count.
D_MODEL, N_TOKENS, DIM = 96, 48, 12
TEST_DRAWS = 39      # floor 1/40 = 0.025, still under alpha = 0.05


def _layer(seed: int, occupied: str = "pos", concentration: float = 3.0,
           dim_u_pos: int = DIM, mean_offset: float = 2.0):
    """A population and two subspaces with the answer planted by construction."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(D_MODEL, D_MODEL)))[0]
    u_pos, u_neg = Q[:, :dim_u_pos], Q[:, dim_u_pos:dim_u_pos + DIM]
    start = {"pos": 0, "neg": dim_u_pos, "other": dim_u_pos + DIM}[occupied]
    u_occ = Q[:, start:start + DIM]
    X = rng.normal(size=(N_TOKENS, D_MODEL))
    if concentration:
        X = X + (rng.normal(size=(N_TOKENS, DIM)) * concentration) @ u_occ.T
    if mean_offset:
        scale = float(np.sqrt((X ** 2).sum(axis=1).mean()))
        X = X + (Q[:, -1] * mean_offset * scale)[None, :]
    return X, u_pos, u_neg


def _both_arms_occupied(seed: int, concentration: float = 2.0):
    """
    The H0 family that retired the matched-dimension null (2026-08-26).

    Both arms hold more of the cloud than a random subspace of their dimension
    would, and the two are IDENTICAL by construction -- swapping the labels is
    a distributional identity, so the correct verdict is INSUFFICIENT and
    P(TRACKS) must equal P(INVERTS). It is also the realistic case: U_pos and
    U_neg are cut from the model's own OV eigenstructure and the residual
    stream is orthogonal to neither.
    """
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(D_MODEL, D_MODEL)))[0]
    u_pos, u_neg = Q[:, :DIM], Q[:, DIM:2 * DIM]
    X = rng.normal(size=(N_TOKENS, D_MODEL))
    for U in (u_pos, u_neg):
        X = X + (rng.normal(size=(N_TOKENS, DIM)) * concentration) @ U.T
    return X, u_pos, u_neg


class TestSteeringIsAMeanEffect(unittest.TestCase):
    """
    Adding alpha*v to every token IS a shift of the population mean. That is
    algebra, and it is what makes DEBIAS_BASELINE_MEAN a decision rather than a
    detail: an effective-rank pipeline that re-centres AFTER injection measures
    nothing at all.
    """

    def test_recentring_after_injection_annihilates_the_intervention(self):
        X, u_pos, _ = _layer(1)
        v = draw_unit_in_subspace(u_pos, np.random.default_rng(0))
        a = ALPHA_SPREAD_FRACTION * population_spread(X)
        moved = X + a * v[None, :]
        self.assertAlmostEqual(
            effective_rank(X - X.mean(0, keepdims=True), mode=ER_MODE),
            effective_rank(moved - moved.mean(0, keepdims=True), mode=ER_MODE),
            places=9,
            msg="re-centring after injection must give back the baseline "
                "exactly; if it does not, the intervention is not a pure mean "
                "shift and the module's whole reading of it is wrong")

    def test_the_gate_debiases_the_baseline_and_keeps_the_injected_offset(self):
        self.assertTrue(DEBIAS_BASELINE_MEAN)
        X, u_pos, _ = _layer(2)
        v = draw_unit_in_subspace(u_pos, np.random.default_rng(0))
        a = ALPHA_SPREAD_FRACTION * population_spread(X)
        self.assertNotAlmostEqual(delta_effective_rank(X, v, a), 0.0, places=6)

    def test_mean_ratio_is_reported(self):
        X, _, _ = _layer(3, mean_offset=2.0)
        self.assertGreater(population_mean_ratio(X), 1.0)


class TestSignIsEvenInV(unittest.TestCase):
    """
    Why ER_MODE is 'raw'. A steering direction and its negation are the same
    object, so a criterion that answers differently for them is not a criterion
    about the decomposition.
    """

    def _agreement(self, mode: str, fraction: float, n: int = 12) -> float:
        rng = np.random.default_rng(7)
        X, u_pos, _ = _layer(4)
        a = fraction * population_spread(X)
        agree = 0
        for _ in range(n):
            v = draw_unit_in_subspace(u_pos, rng)
            agree += int(np.sign(delta_effective_rank(X, v, a, mode))
                         == np.sign(delta_effective_rank(X, -v, a, mode)))
        return agree / n

    def test_raw_is_even_in_v_at_every_scale(self):
        for f in (1e-6, 1e-3, 1e-1, ALPHA_SPREAD_FRACTION):
            self.assertEqual(self._agreement("raw", f), 1.0, f"fraction={f}")

    def test_normed_is_not(self):
        """The measurement that disqualifies the CLAIM-C precedent here."""
        self.assertLess(self._agreement("normed", 1e-6), 0.5)

    def test_the_module_uses_raw(self):
        self.assertEqual(ER_MODE, "raw")

    def test_the_two_modes_agree_at_the_working_alpha(self):
        """
        Which is exactly how the wrong one nearly shipped: a single working
        point cannot distinguish them, and only the small-alpha limit does.
        """
        self.assertEqual(self._agreement("raw", ALPHA_SPREAD_FRACTION), 1.0)
        self.assertEqual(self._agreement("normed", ALPHA_SPREAD_FRACTION), 1.0)


class TestTheStatisticIsOrdinalAndAntisymmetric(unittest.TestCase):

    def test_D_is_a_difference_of_signs(self):
        X, u_pos, u_neg = _layer(5)
        rng = np.random.default_rng(0)
        a = ALPHA_SPREAD_FRACTION * population_spread(X)
        r = pair_statistic(X, draw_unit_in_subspace(u_pos, rng),
                           draw_unit_in_subspace(u_neg, rng), a)
        self.assertIn(r["D"], (-2.0, -1.0, 0.0, 1.0, 2.0))
        self.assertEqual(r["informative"], r["D"] != 0.0)

    def test_swapping_the_arms_negates_D(self):
        X, u_pos, u_neg = _layer(6)
        rng = np.random.default_rng(1)
        a = ALPHA_SPREAD_FRACTION * population_spread(X)
        for _ in range(8):
            vp = draw_unit_in_subspace(u_pos, rng)
            vn = draw_unit_in_subspace(u_neg, rng)
            self.assertEqual(pair_statistic(X, vp, vn, a)["D"],
                             -pair_statistic(X, vn, vp, a)["D"])

    def test_both_arms_get_the_same_alpha(self):
        """Norm matching is by construction, not by a tolerance."""
        X, u_pos, u_neg = _layer(7)
        out = draw_pairs(X, u_pos, u_neg, 4)
        self.assertAlmostEqual(out["alpha_spread_fraction"],
                               ALPHA_SPREAD_FRACTION, places=12)
        self.assertTrue(out["alpha_is_placed"])
        self.assertTrue(ALPHA_IS_PLACED)


class TestTheFloorIsSetByInformativePairs(unittest.TestCase):
    """The registered null's floor, and the arithmetic that retired its power."""

    def test_floor_formula(self):
        for m in (8, 12, 30):
            for k in range(1, min(m, 8) + 1):
                self.assertAlmostEqual(
                    attainable_floor(m, k),
                    (2.0 ** (m - k) + 1.0) / (2.0 ** m + 1.0), places=12)

    def test_five_informative_pairs_is_the_first_that_clears_five_percent(self):
        self.assertEqual(min_informative_pairs(0.05), 5)
        for m in (8, 12, 20, 30):
            self.assertGreater(attainable_floor(m, 4), 0.05)
            self.assertLessEqual(attainable_floor(m, 5), 0.05)

    def test_it_barely_depends_on_the_pair_count(self):
        self.assertLess(abs(attainable_floor(8, 5) - attainable_floor(60, 5)),
                        0.005)

    def test_padding_with_zeros_cannot_rescue_too_few_informative_pairs(self):
        """
        The floor is not flat in m -- it falls toward 2^-k from above as
        uninformative pairs are added, because the +1 correction shrinks
        relative to a larger null. What matters is that it is BOUNDED BELOW by
        2^-k, so four informative pairs cannot be made to clear alpha = 0.05 by
        drawing more uninformative ones however many are added. That bound is
        what `min_informative_pairs` uses, and using the limit is what keeps it
        a lower bound rather than a number that might just miss.
        """
        self.assertGreater(attainable_floor(5, 5), attainable_floor(50, 5))
        for m in (4, 8, 40, 200):
            self.assertGreaterEqual(attainable_floor(m, 4), 2.0 ** -4)
            self.assertGreater(attainable_floor(m, 4), 0.05)
            if m >= 5:
                self.assertGreaterEqual(attainable_floor(m, 5), 2.0 ** -5)

    def test_more_informative_than_drawn_is_refused(self):
        with self.assertRaises(ValueError):
            attainable_floor(4, 5)

    def test_min_informative_pairs_tracks_alpha(self):
        self.assertLess(min_informative_pairs(0.01), min_informative_pairs(1e-4))
        self.assertGreater(min_informative_pairs(0.2), 0)


class TestTheNullArithmetic(unittest.TestCase):
    """
    The convolution null is a SECOND implementation of the permutation null's
    arithmetic, which is a real risk. It is pinned against the direct
    enumeration cell by cell, the same arrangement CLAIM-C's fast path has.
    """

    def test_convolution_matches_direct_enumeration(self):
        rng = np.random.default_rng(3)
        from core.nulls import p_from_null
        for _ in range(20):
            m = int(rng.integers(1, 11))
            D = rng.integers(-2, 3, size=m).astype(float)
            direct = null_sums(D)
            values, counts = null_distribution(D)
            import collections
            self.assertEqual(
                collections.Counter(np.round(direct).astype(int).tolist()),
                {int(v): c for v, c in zip(values, counts) if c})
            for alt in (ALTERNATIVE, RECIPROCAL_ALTERNATIVE):
                self.assertAlmostEqual(
                    p_from_null(D.sum(), direct, alternative=alt)["p_value"],
                    p_from_distribution(D.sum(), values, counts, alt),
                    places=12)

    def test_counts_are_exact_integers(self):
        """A float count starts rounding around m = 53, silently."""
        _, counts = null_distribution([2.0] * 60)
        self.assertEqual(sum(counts), 2 ** 60)
        self.assertTrue(all(isinstance(c, int) for c in counts))

    def test_D_outside_minus_two_to_two_is_refused(self):
        with self.assertRaises(ValueError):
            null_distribution([3.0, 1.0])


class TestTheSubspaceNulls(unittest.TestCase):
    """
    Two nulls: the one that is adjudicated (re-split the observed union) and
    6k's matched-dimension pair, retired 2026-08-26 and kept as a diagnostic.

    The retirement is pinned by its MECHANISM rather than by resampling its
    consequence -- the same arrangement 6h's per-layer/per-model comparison
    settled on. A matched-dimension random pair does not reproduce how much of
    the population the observed pair holds; a re-split of the observed union
    reproduces it exactly, because it is the same union.
    """

    def test_the_pair_is_orthogonal_and_matched_in_dimension(self):
        rng = np.random.default_rng(0)
        a, b = random_orthogonal_subspace_pair(64, 10, 7, rng)
        self.assertEqual((a.shape[1], b.shape[1]), (10, 7))
        self.assertLess(float(np.abs(a.T @ b).max()), 1e-10)
        for M in (a, b):
            self.assertLess(float(np.abs(M.T @ M - np.eye(M.shape[1])).max()),
                            1e-10)

    def test_it_refuses_when_no_orthogonal_pair_of_those_dimensions_exists(self):
        with self.assertRaises(ValueError):
            random_orthogonal_subspace_pair(16, 10, 10, np.random.default_rng(0))

    def test_a_resplit_is_orthogonal_and_spans_the_same_union(self):
        rng = np.random.default_rng(0)
        S = np.linalg.qr(rng.normal(size=(64, 12)))[0]
        a, b = resplit_pair(S, 5, rng)
        self.assertEqual((a.shape[1], b.shape[1]), (5, 7))
        self.assertLess(float(np.abs(a.T @ b).max()), 1e-10)
        # span(a) + span(b) == span(S): every column of the union is recovered
        # by projecting onto the two halves.
        P = np.hstack([a, b])
        self.assertLess(float(np.abs(P @ (P.T @ S) - S).max()), 1e-10)

    def test_a_union_assigned_entirely_to_one_arm_is_refused(self):
        S = np.linalg.qr(np.random.default_rng(0).normal(size=(32, 6)))[0]
        with self.assertRaises(ValueError):
            resplit_pair(S, 6, np.random.default_rng(0))

    def test_the_resplit_preserves_union_occupancy_and_the_old_null_does_not(self):
        """
        The mechanism the retirement rests on, deterministic and in
        milliseconds.

        dER is driven by how much of the cloud a subspace holds. A re-split
        draws inside the observed union, so the pair's occupancy is the
        observed pair's occupancy to machine precision. A matched-dimension
        random pair holds what a random subspace holds -- about chance -- so on
        a population that occupies both arms it is compared against pairs the
        observed one is not exchangeable with.
        """
        X, u_pos, u_neg = _both_arms_occupied(21)
        union = union_basis(u_pos, u_neg)
        obs = occupancy(X, union)
        self.assertGreater(obs, 1.5)                 # both arms well occupied

        rng = np.random.default_rng(0)
        for _ in range(5):
            a, b = resplit_pair(union, u_pos.shape[1], rng)
            self.assertAlmostEqual(occupancy(X, np.hstack([a, b])), obs,
                                   places=10)
        for _ in range(5):
            a, b = random_orthogonal_subspace_pair(
                D_MODEL, u_pos.shape[1], u_neg.shape[1], rng)
            self.assertLess(occupancy(X, np.hstack([a, b])), 0.5 * obs)

    def test_the_gate_refuses_a_union_that_cannot_hold_the_observed_pair(self):
        X, u_pos, u_neg = _layer(8)
        big = np.linalg.qr(np.random.default_rng(0).normal(
            size=(D_MODEL, D_MODEL - 4)))[0]
        res = p_value_p_st1(X, big, big, 4, n_draws=19, with_profile=False)
        self.assertIsNone(res["p_value"])
        self.assertEqual(res["refusal_kind"], "union_rank_deficient")
        self.assertIn("exceed d_model", res["reason"])

    def test_the_gate_refuses_overlapping_arms(self):
        """
        Orthogonality was ASSUMED from the projector build's resolution order
        from the day the module was written and never checked. Overlapping arms
        now refuse rather than getting a null drawn on a geometry the observed
        pair does not have.
        """
        X, u_pos, u_neg = _layer(8)
        overlap = np.hstack([u_pos[:, :2], u_neg[:, :DIM - 2]])
        res = p_value_p_st1(X, u_pos, overlap, 4, n_draws=19, with_profile=False)
        self.assertIsNone(res["p_value"])
        self.assertEqual(res["refusal_kind"], "union_rank_deficient")
        self.assertIn("overlap", res["reason"])

    def test_the_floor_is_fixed_by_the_draws_not_the_data(self):
        X, u_pos, u_neg = _layer(9)
        res = p_value_p_st1(X, u_pos, u_neg, 4, n_draws=9, with_profile=False)
        self.assertIsNone(res["p_value"])
        self.assertEqual(res["refusal_kind"], "draws_below_floor")
        self.assertIn("null draws can express no p smaller", res["reason"])
        self.assertAlmostEqual(res["best_attainable_p"], 0.1, places=12)

    def test_the_attainable_floor_is_set_by_ties_and_not_by_the_draw_count(self):
        """
        The defect `tools/dry_run_p_st1.py` found by running the gate on an
        input whose answer was known (POPPER_PLAN.md 6m).

        sum(D) cannot exceed 2m, so a run's smallest expressible p is what an
        observation of 2m would receive -- and on a union the cloud occupies,
        many null re-splits already reach 2m and tie it. At one pair the two
        floors are far apart, and the gate refuses rather than reporting the
        "not significant" that a design incapable of rejecting produces.
        """
        X, u_pos, u_neg = _both_arms_occupied(23)
        res = p_value_p_st1(X, u_pos, u_neg, 1, n_draws=39, with_profile=False)
        self.assertEqual(res["refusal_kind"], "null_ties_the_maximum")
        self.assertGreater(res["best_attainable_p"], 0.05)
        self.assertAlmostEqual(res["draw_count_floor"], 1 / 40, places=12)
        self.assertGreater(res["best_attainable_p"], res["draw_count_floor"])
        self.assertIn("Raising n_draws does NOT fix this", res["reason"])

    def test_a_perfect_input_lands_on_its_own_attainable_floor(self):
        """
        Not on the draw-count floor -- that is the whole point of the previous
        test. With enough pairs the two coincide, and where they do not it is
        the attainable one a perfect input reaches.
        """
        X, u_pos, u_neg = _layer(24, "pos")
        for m in (4, 8):
            res = p_value_p_st1(X, u_pos, u_neg, m, n_draws=TEST_DRAWS,
                                with_profile=False)
            self.assertEqual(res["verdict"], "TRACKS-DECOMPOSITION")
            self.assertAlmostEqual(res["p_value"],
                                   res["attainable_p_greater"], places=12)

    def test_the_tie_floor_refusal_needs_BOTH_tails_out_of_reach(self):
        """
        One reachable tail is one reachable verdict, so the gate is not a
        constant function there and must not refuse -- 6l's rule that a refusal
        costs no verdict the gate could otherwise have reached.

        A cloud in U_pos at one pair is exactly that case: no re-split reaches
        -2m, so INVERTS stays reachable while TRACKS does not, and the gate
        emits. `reachable_tails` records the asymmetry, which is worth having
        in the artifact because a run whose only reachable verdict is the
        FALSIFICATION is one a reader should know about.
        """
        X, u_pos, u_neg = _layer(23, "pos")
        res = p_value_p_st1(X, u_pos, u_neg, 1, n_draws=TEST_DRAWS,
                            with_profile=False)
        self.assertIsNotNone(res["p_value"])
        self.assertGreater(res["attainable_p_greater"], 0.05)
        self.assertLessEqual(res["attainable_p_reciprocal"], 0.05)
        self.assertEqual(res["reachable_tails"], ["reciprocal"])

    def test_a_data_refusal_is_reported_ahead_of_a_calibration_one(self):
        """
        POPPER_PLAN.md 6l's ordering rule. An input that is both geometrically
        impossible and under-drawn should say which of the two it cannot fix by
        raising `n_draws`.
        """
        X, u_pos, _ = _layer(9)
        res = p_value_p_st1(X, u_pos, u_pos, 4, n_draws=9, with_profile=False)
        self.assertEqual(res["refusal_kind"], "union_rank_deficient")

    def test_subspace_rank_reads_both_shapes(self):
        rng = np.random.default_rng(0)
        U = np.linalg.qr(rng.normal(size=(32, 7)))[0]
        self.assertEqual(subspace_rank(U), 7)
        self.assertEqual(subspace_rank(U @ U.T), 7)

    def test_compact_basis_reads_both_shapes(self):
        rng = np.random.default_rng(0)
        U = np.linalg.qr(rng.normal(size=(32, 7)))[0]
        for arg in (U, U @ U.T):
            B = compact_basis(arg)
            self.assertEqual(B.shape, (32, 7))
            self.assertLess(float(np.abs(B.T @ B - np.eye(7)).max()), 1e-9)
            # same span either way
            self.assertLess(float(np.abs(B @ (B.T @ U) - U).max()), 1e-9)


class TestOccupancyIsReportedAndChanceNormalized(unittest.TestCase):
    """
    The quantity a TRACKS verdict is made of, computable with no injection.
    """

    def test_a_random_subspace_sits_at_one(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(N_TOKENS, D_MODEL))
        U = np.linalg.qr(rng.normal(size=(D_MODEL, DIM)))[0]
        self.assertAlmostEqual(occupancy(X, U), 1.0, delta=0.35)

    def test_it_is_comparable_across_dimensions(self):
        """
        6h's whole finding was an alignment comparison read without dimension
        normalization. Raw captured energy scales with k; this does not.
        """
        rng = np.random.default_rng(5)
        X = rng.normal(size=(N_TOKENS, D_MODEL))
        Q = np.linalg.qr(rng.normal(size=(D_MODEL, D_MODEL)))[0]
        small, large = occupancy(X, Q[:, :3]), occupancy(X, Q[:, :30])
        self.assertAlmostEqual(small, large, delta=0.6)

    def test_the_gate_reports_both_arms_and_the_asymmetry(self):
        X, u_pos, u_neg = _layer(6, "pos")
        res = p_value_p_st1(X, u_pos, u_neg, 4, n_draws=TEST_DRAWS,
                            with_profile=False)
        occ = res["occupancy"]
        self.assertGreater(occ["occupancy_pos"], occ["occupancy_neg"])
        self.assertGreater(occ["occupancy_log_ratio"], 0.0)
        self.assertNotIn("occupancy", res["statistic"])


class TestAllThreeVerdictsCanFire(unittest.TestCase):
    """
    A verdict branch nothing can trigger is POPPER_PLAN.md 6h's arm-incapable-
    of-failing wearing a different hat, and INVERTS is the branch that enters
    the ledger as a falsification.
    """

    def _run(self, occupied, concentration=3.0, n_pairs=8):
        X, u_pos, u_neg = _layer(11, occupied, concentration)
        return p_value_p_st1(X, u_pos, u_neg, n_pairs, n_draws=TEST_DRAWS,
                             with_profile=False)

    def test_tracks_decomposition_when_the_cloud_lives_in_u_pos(self):
        res = self._run("pos")
        self.assertEqual(res["verdict"], "TRACKS-DECOMPOSITION")
        self.assertLessEqual(res["p_value"], 0.05)
        self.assertFalse(res["falsified"])

    def test_inverts_when_the_cloud_lives_in_u_neg(self):
        res = self._run("neg")
        self.assertEqual(res["verdict"], "INVERTS")
        self.assertLessEqual(res["p_reciprocal"], 0.05)
        self.assertTrue(res["falsified"])

    def test_insufficient_when_the_cloud_lives_elsewhere(self):
        res = self._run("other")
        self.assertEqual(res["verdict"], "INSUFFICIENT")
        self.assertFalse(res["falsified"])

    def test_the_registered_falsifier_maps_to_insufficient_not_to_falsified(self):
        """
        "Both arms move effective rank the same way" is the NULL. An e-process
        records insufficient evidence and never a null accepted, so it cannot
        be what enters the ledger — stated here rather than discovered at the
        moment it binds.
        """
        res = self._run("other")
        self.assertEqual(res["n_informative_pairs"], 0)
        self.assertFalse(res["falsified"])
        self.assertIn("insufficient evidence", res["reading"])

    def test_verdict_lattice_directly(self):
        self.assertEqual(gate_verdict(0.01, 1.0, 0.05)["verdict"],
                         "TRACKS-DECOMPOSITION")
        self.assertEqual(gate_verdict(1.0, 0.01, 0.05)["verdict"], "INVERTS")
        self.assertTrue(gate_verdict(1.0, 0.01, 0.05)["falsified"])
        self.assertEqual(gate_verdict(0.5, 0.5, 0.05)["verdict"], "INSUFFICIENT")
        self.assertEqual(gate_verdict(None, None, 0.05)["verdict"],
                         "INSUFFICIENT")


class TestBothRetiredNullsAreReportedButNotAdjudicated(unittest.TestCase):

    def test_the_adjudicated_null_re_splits_the_union(self):
        self.assertIn("re-split", NULL_FAMILY)
        res = p_value_p_st1(*_layer(12, "pos"), 8, n_draws=TEST_DRAWS,
                            with_profile=False)
        self.assertEqual(res["null_family"], NULL_FAMILY)
        self.assertEqual(res["n_subspace_draws"], TEST_DRAWS)
        self.assertEqual(res["dim_union"], 2 * DIM)

    def test_the_null_depends_on_the_union_and_not_on_the_labelling(self):
        """
        The sharpest statement of what this null holds fixed: it is a function
        of span(U_pos + U_neg) and the two dimensions, so nothing about which
        half was called attractive can reach it. That is H0-BRIDGE for this
        entry -- the label carries no information -- built into the null rather
        than measured out of it.
        """
        _, u_pos, u_neg = _layer(19, "pos")
        a, b = union_basis(u_pos, u_neg), union_basis(u_neg, u_pos)
        self.assertEqual(a.shape, b.shape)
        self.assertLess(float(np.abs(a @ (a.T @ b) - b).max()), 1e-9)

    def test_the_matched_dimension_null_is_computed_beside_it(self):
        res = p_value_p_st1(*_layer(12, "pos"), 8, n_draws=TEST_DRAWS,
                            with_profile=False)
        diag = res["matched_dimension_diagnostic"]
        self.assertIn("NOT ADJUDICATED", diag["null_family"])
        self.assertIn("RETIRED", diag["null_family"])
        self.assertIsNotNone(diag["p_value"])

    def test_the_retired_matched_dimension_null_rejects_where_this_one_does_not(self):
        """
        The finding this pass turned on, on inputs whose correct verdict is
        INSUFFICIENT by construction (both arms occupied above chance, the two
        statistically identical, so a label swap is a distributional identity).

        Deterministic: everything is seeded, and the two nulls are scored on
        the SAME eight populations and the same drawn pairs, so the comparison
        is paired rather than two experiments. The margins are wide because an
        exact pin on a floating-point RNG stream is a test about LAPACK.
        """
        resplit_rejections = matched_rejections = 0
        for s in range(8):
            X, u_pos, u_neg = _both_arms_occupied(500 + s)
            res = p_value_p_st1(X, u_pos, u_neg, 4, n_draws=19,
                                with_profile=False, seed=2000 + s)
            diag = res["matched_dimension_diagnostic"]
            resplit_rejections += (res["p_value"] <= 0.05
                                   or res["p_reciprocal"] <= 0.05)
            matched_rejections += (diag["p_value"] <= 0.05
                                   or diag["p_reciprocal"] <= 0.05)
        self.assertLessEqual(resplit_rejections, 1)
        self.assertGreaterEqual(matched_rejections, 3)

    def test_the_registered_permutation_is_computed_beside_it(self):
        res = p_value_p_st1(*_layer(12, "pos"), 8, n_draws=TEST_DRAWS,
                            with_profile=False)
        diag = res["label_permutation_diagnostic"]
        self.assertIn("NOT ADJUDICATED", diag["null_family"])
        self.assertIsNotNone(diag["p_value"])
        self.assertNotEqual(diag["p_value"], res["p_value"])

    def test_the_diagnostic_carries_its_own_floor(self):
        diag = label_permutation_diagnostic([2, 2, 0, 0, 0, 0], 0.05)
        self.assertEqual(diag["n_informative_pairs"], 2)
        self.assertAlmostEqual(diag["best_attainable_p"],
                               attainable_floor(6, 2), places=12)

    def test_an_all_zero_diagnostic_has_no_p(self):
        diag = label_permutation_diagnostic([0, 0, 0, 0], 0.05)
        self.assertIsNone(diag["p_value"])


class TestRefusalsAndDefaults(unittest.TestCase):

    def test_constants_are_on_record(self):
        self.assertEqual(P_ST1_UNIT, "matched-norm vector pair")
        self.assertEqual(ALTERNATIVE, "greater")
        self.assertEqual(RECIPROCAL_ALTERNATIVE, "less")
        self.assertEqual(N_SUBSPACE_DRAWS, 199)

    def test_a_population_with_no_spread_is_refused(self):
        X = np.tile(np.arange(D_MODEL, dtype=np.float64), (N_TOKENS, 1))
        _, u_pos, u_neg = _layer(13)
        with self.assertRaises(ValueError):
            draw_pairs(X, u_pos, u_neg, 4)

    def test_a_single_token_is_refused(self):
        _, u_pos, u_neg = _layer(14)
        with self.assertRaises(ValueError):
            draw_pairs(np.zeros((1, D_MODEL)), u_pos, u_neg, 4)

    def test_an_empty_subspace_is_refused_not_counted_as_uninformative(self):
        with self.assertRaises(ValueError):
            draw_unit_in_subspace(np.zeros((D_MODEL, 3)),
                                  np.random.default_rng(0))

    def test_the_alpha_profile_is_reported_and_enters_no_p_value(self):
        res = p_value_p_st1(*_layer(15, "pos"), 2, n_draws=TEST_DRAWS,
                            with_profile=True)
        self.assertGreater(len(res["alpha_profile"]), 4)
        for row in res["alpha_profile"]:
            self.assertIn("informative_rate", row)
        self.assertNotIn("alpha_profile", res["statistic"])


class TestCommittedCalibration(unittest.TestCase):
    """
    `claims/calibration/steering_sign.json` is the evidence behind the module's
    four constants. It is generated offline (about twenty-five minutes) and
    committed, the same arrangement `claims/calibration/changepoint_colocation.json`
    has — read by nobody at runtime, so a lost file breaks no gate; what it
    would lose is the reason the gate is built this way.

    Unlike that one it describes ANOTHER file, so the module's sha256 is pinned
    too. Without it the record could go on describing a construction that no
    longer exists in that form and nothing in the suite would notice.

    The assertions below are on the DIRECTION each section establishes rather
    than on its digits, because the digits are proportions over a few hundred
    draws and pinning sampling noise to three places would make this file fail
    for the wrong reason. What is pinned exactly is that each measured section
    still supports the constant it decided — so flipping ER_MODE back to the
    CLAIM-C precedent fails the gate rather than leaving the record quietly
    describing an argument for something else.
    """

    def _rec(self) -> dict:
        import json
        from tools.calibrate_steering_sign import RECORD_PATH
        self.assertTrue(
            RECORD_PATH.exists(),
            f"{RECORD_PATH} is missing. Regenerate with "
            f"`python3 -m tools.calibrate_steering_sign --write`.")
        return json.loads(RECORD_PATH.read_text())

    def test_check_record_is_clean(self):
        from tools.calibrate_steering_sign import check_record
        self.assertEqual(check_record(), [])

    def test_it_describes_the_module_on_disk(self):
        import hashlib
        from tools.calibrate_steering_sign import GATE_PATH
        self.assertEqual(
            self._rec()["gate_sha256"],
            hashlib.sha256(GATE_PATH.read_bytes()).hexdigest(),
            "steering_gate.py has changed since the calibration was written; "
            "rerun --write rather than editing the hash")

    def test_it_is_not_an_adjudication(self):
        self.assertIn("not evidence about any model", self._rec()["_not"])

    def test_raw_is_even_in_v_and_normed_is_not(self):
        """The section that decided ER_MODE."""
        rows = self._rec()["sign_symmetry"]["rows"]
        raw = [r for r in rows if r["er_mode"] == "raw"]
        normed = [r for r in rows if r["er_mode"] == "normed"]
        self.assertTrue(raw and normed)
        for r in raw:
            self.assertEqual(r["sign_agreement_under_v_to_minus_v"], 1.0)
            self.assertEqual(r["frac_pairs_reading_inverted"], 0.0)
        small = min(normed, key=lambda r: r["alpha_spread_fraction"])
        self.assertLess(small["sign_agreement_under_v_to_minus_v"], 0.5)
        self.assertGreater(small["frac_pairs_reading_inverted"], 0.05,
                           "normed must be shown MANUFACTURING inversions, not "
                           "merely disagreeing")

    def test_debiasing_separates_the_families_and_not_debiasing_does_not(self):
        """DEBIAS_BASELINE_MEAN, with each arm at its own best alpha."""
        rows = self._rec()["mean_offset"]["rows"]
        for r in rows:
            if r["debias_baseline_mean"]:
                self.assertGreater(r["h1_frac_predicted"], 0.5,
                                   f"offset={r['mean_offset_in_spreads']}")
                self.assertLess(r["h0_frac_predicted"], 0.1)
        high = [r for r in rows if not r["debias_baseline_mean"]
                and r["mean_offset_in_spreads"] >= 2.0]
        self.assertTrue(high)
        for r in high:
            self.assertLess(r["h1_frac_predicted"] - r["h0_frac_predicted"], 0.3,
                            "the undebiased design must be shown FAILING to "
                            "separate at a realistic mean offset, even at its "
                            "own best alpha")

    def test_the_alpha_plateau_holds_the_shipped_fraction(self):
        aw = self._rec()["alpha_window"]
        self.assertTrue(aw["window_location_is_stable_in_spread_units"],
                        f"plateaus differ across mean offsets: "
                        f"{aw['plateau_by_mean_offset']}")
        self.assertTrue(aw["shipped_fraction_is_in_every_plateau"],
                        f"ALPHA_SPREAD_FRACTION = {ALPHA_SPREAD_FRACTION} is "
                        f"outside a measured plateau: "
                        f"{aw['plateau_by_mean_offset']}")
        for plateau in aw["plateau_by_mean_offset"].values():
            self.assertGreaterEqual(len(plateau), 2,
                                    "a plateau of one point is a peak, and a "
                                    "placed constant on a peak is fragile")

    def test_the_registered_null_inflates_with_the_pair_count(self):
        """Why the registered null is reported and not adjudicated."""
        rows = [r for r in self._rec()["registered_null_inflation"]["rows"]
                if r["family"].startswith("H0-noisy")
                and r["reject_rate_given_emitted"] is not None]
        self.assertTrue(rows)
        by_m = {}
        for r in rows:
            by_m.setdefault(r["n_pairs"], []).append(
                r["reject_rate_given_emitted"])
        self.assertGreater(max(by_m[max(by_m)]), 0.10,
                           "at the largest pair count the registered null must "
                           "be shown ANTICONSERVATIVE, or the case for "
                           "replacing it is not in the record")
        self.assertLess(max(by_m[min(by_m)]), 0.10,
                        "and nominal at the smallest, which is what makes it "
                        "an inflation that GROWS with the pair count rather "
                        "than a null that is simply wrong everywhere")

    def test_the_adjudicated_null_controls_h0_and_has_power_both_ways(self):
        sec = self._rec()["adjudicated_null_validity_and_power"]
        n = sec["n_trials_per_row"]
        for r in sec["rows"]:
            if not r["family"].startswith("H0"):
                continue
            for key in ("resplit_reject_given_emitted",
                        "resplit_reciprocal_given_emitted"):
                rate = r[key]
                if rate is None:
                    continue
                # alpha plus one standard error of a proportion over n runs --
                # the same bound `check_record` derives, rather than a placed
                # tolerance. At n = 50 a true 0.05 lands on 0.10 about once in
                # ten cells and this table has twenty of them, which is why the
                # reciprocal tail gets its own section at four times the
                # replicates instead of a tighter assertion here.
                ceiling = 0.05 + 1.96 * (0.05 * 0.95 / n) ** 0.5
                self.assertLessEqual(
                    rate, ceiling,
                    f"{r['family']} at {r['n_pairs']} pairs, {key}")
        h1 = [r for r in sec["rows"] if r["family"] == "H1"]
        inv = [r for r in sec["rows"] if r["family"] == "INVERTED"]
        self.assertTrue(h1 and inv)
        for r in h1:
            self.assertGreaterEqual(r["tracks_decomposition"], 0.9)
        for r in inv:
            self.assertGreaterEqual(r["inverts"], 0.9,
                                    "the falsification branch must be shown "
                                    "firing under a planted inversion")

    def test_the_family_that_retired_the_matched_dimension_null_is_measured(self):
        """
        The H0 family whose ABSENCE kept the previous null's failure invisible
        for a pass: both arms occupied above chance, the two identical by
        construction. A calibration whose families cannot express the failure
        it rules out is POPPER_PLAN.md 6h's audit arm incapable of failing.
        """
        sec = self._rec()["adjudicated_null_validity_and_power"]
        both = [r for r in sec["rows"] if r["family"].startswith("H0-both-arms")]
        self.assertTrue(both, "no H0-both-arms family in the record")
        for r in both:
            self.assertGreater(min(r["mean_occupancy_pos"],
                                   r["mean_occupancy_neg"]), 1.05,
                               "the arms were not actually occupied above "
                               "chance, so the family is not the one it names")

    def test_the_retired_matched_dimension_null_is_shown_anticonservative(self):
        """
        It was retired on this evidence. If the record no longer shows it, the
        retirement is not supported by the artifact that supports it -- so the
        test fails rather than quietly agreeing with the module.
        """
        sec = self._rec()["adjudicated_null_validity_and_power"]
        worst = max(
            max(r["matched_dimension_reject_given_emitted"] or 0.0,
                r["matched_dimension_reciprocal_given_emitted"] or 0.0)
            for r in sec["rows"] if r["family"].startswith("H0"))
        self.assertGreater(worst, 0.12)

    def test_the_reciprocal_tail_carries_more_replicates_and_both_tails(self):
        """
        POPPER_PLAN.md 6k named the INVERTS branch's rate as this
        construction's weakest measurement: fifty runs resolve a rate to about
        +/- 0.03, which cannot separate nominal from twice nominal.
        """
        rec = self._rec()
        rt = rec["reciprocal_tail"]
        main = rec["adjudicated_null_validity_and_power"]["n_trials_per_row"]
        self.assertGreater(rt["n_trials_per_row"], main)
        self.assertTrue(rt["rows"])
        self.assertTrue(rt["tails_agree"],
                        "the arms are exchangeable by construction in every "
                        "family here, so the two tails must agree within "
                        "sampling error")
        for r in rt["rows"]:
            self.assertLessEqual(r["reciprocal_given_emitted"],
                                 0.05 + 1.5 / rt["n_trials_per_row"],
                                 f"{r['family']}: the branch that would enter "
                                 f"the ledger as a falsification")

    def test_the_dimension_precondition_is_recorded(self):
        rows = self._rec()["dimension_cliff"]["rows"]
        by_ratio = {r["ratio"]: r["informative_rate"] for r in rows}
        self.assertGreater(by_ratio[1.0], 0.5)
        self.assertLess(by_ratio[max(by_ratio)], 0.05,
                        "the rate must be shown collapsing as dim U_pos grows "
                        "past the dimension the population occupies")
        self.assertGreater(by_ratio[1.0], by_ratio[max(by_ratio)])


if __name__ == "__main__":
    unittest.main()
