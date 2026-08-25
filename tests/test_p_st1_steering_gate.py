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
    subspace_rank,
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


class TestTheSubspaceNull(unittest.TestCase):
    """
    The adjudicated null: POPPER_PLAN.md 6h's construction, fourth arrival.
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

    def test_the_gate_refuses_the_same_case(self):
        X, u_pos, u_neg = _layer(8)
        big = np.linalg.qr(np.random.default_rng(0).normal(
            size=(D_MODEL, D_MODEL - 4)))[0]
        res = p_value_p_st1(X, big, big, 4, n_draws=19, with_profile=False)
        self.assertIsNone(res["p_value"])
        self.assertIn("exceeds d_model", res["reason"])

    def test_the_floor_is_fixed_by_the_draws_not_the_data(self):
        X, u_pos, u_neg = _layer(9)
        res = p_value_p_st1(X, u_pos, u_neg, 4, n_draws=9, with_profile=False)
        self.assertIsNone(res["p_value"])
        self.assertIn("null draws can express no p smaller", res["reason"])
        self.assertAlmostEqual(res["best_attainable_p"], 0.1, places=12)

    def test_subspace_rank_reads_both_shapes(self):
        rng = np.random.default_rng(0)
        U = np.linalg.qr(rng.normal(size=(32, 7)))[0]
        self.assertEqual(subspace_rank(U), 7)
        self.assertEqual(subspace_rank(U @ U.T), 7)


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


class TestTheRegisteredNullIsReportedButNotAdjudicated(unittest.TestCase):

    def test_the_adjudicated_null_is_the_subspace_one(self):
        self.assertIn("subspace", NULL_FAMILY)
        res = p_value_p_st1(*_layer(12, "pos"), 8, n_draws=TEST_DRAWS,
                            with_profile=False)
        self.assertEqual(res["null_family"], NULL_FAMILY)
        self.assertEqual(res["n_subspace_draws"], TEST_DRAWS)

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
            rate = r["subspace_reject_given_emitted"]
            if r["family"].startswith("H0") and rate is not None:
                # One trial's worth of slack: these are proportions over n runs
                # and a bound tighter than the resolution fails on noise.
                self.assertLessEqual(rate, 0.05 + 1.5 / n,
                                     f"{r['family']} at {r['n_pairs']} pairs")
        h1 = [r for r in sec["rows"] if r["family"] == "H1"]
        inv = [r for r in sec["rows"] if r["family"] == "INVERTED"]
        self.assertTrue(h1 and inv)
        for r in h1:
            self.assertGreaterEqual(r["tracks_decomposition"], 0.9)
        for r in inv:
            self.assertGreaterEqual(r["inverts"], 0.9,
                                    "the falsification branch must be shown "
                                    "firing under a planted inversion")

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
