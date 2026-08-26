"""
tests/test_p6_r2_r4_null.py — the revived Phase 6 instrument.

`p6_subspace/` is live again, rebuilt against `core/particles.py` and
`core/nulls.py` rather than lifted out of `archive/` (archive/README.md rule 2).
Reviving it is what taking `P6-R2` and `P6-R4` out of `dormant` requires. It
does not produce a p-value for the 2026-04 ALBERT run and is not meant to --
see `p6_subspace/math-6.md` §7.2 and POPPER_PLAN.md §6h.

These tests are on synthetic inputs with known answers, the same standard
`CLAIM-C`, `P-S1`, `P-T1` and `P-M1` were held to: exactness of the
decomposition, validity under H0, power against a planted effect, p = 1 with the
effect in the wrong arm, and every refusal.

Two of them are worth reading rather than counting:

`test_a_random_direction_shows_the_inversion_signature_raw` reproduces the
phase's headline pattern -- much more alignment with U_A than with U_neg -- from
a direction that carries no information at all, and shows it vanishing under
normalization. That is explanation (c) in a single assertion.

`test_layer_unit_null_is_narrower_by_root_n` pins the MECHANISM behind the
exchangeable-unit choice instead of resampling its consequence. Treating 49
dependent layers as 49 independent draws shrinks the null's spread by sqrt(n),
and a null that is too narrow over-rejects. Measured at 400 replicates, the
layer unit reaches 0.28 at alpha = 0.05 when the layers share one direction,
against 0.0575 for the model unit; this test gets at the same thing in
milliseconds and deterministically.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pytest.ini [pytest] markers.
pytestmark = pytest.mark.pure

from p6_subspace import r2_r4_null as R
from p6_subspace.subspace_geometry import (
    chance_alignment,
    layer_channels,
    normalized_alignment,
    random_orthogonal_subspace_pair,
    random_subspace,
    raw_alignment,
    schur_channels,
)

REPO = Path(__file__).resolve().parent.parent


def _random_ovs(d, n_heads, head_dim, rng):
    return [rng.standard_normal((d, head_dim)) / np.sqrt(d)
            @ (rng.standard_normal((head_dim, d)) / np.sqrt(head_dim))
            for _ in range(n_heads)]


def _albert_like(d=32, n_heads=6, head_dim=10, n_layers=4, seed=3):
    """One projector, several activation snapshots -- ALBERT's weight-tying."""
    rng = np.random.default_rng(seed)
    ch = layer_channels(_random_ovs(d, n_heads, head_dim, rng))
    return [ch] * n_layers, rng


def _unit_vec(rng, d):
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class TestSchurPartition(unittest.TestCase):

    def _planted(self, d, n_pos, n_neg, n_rot, n_zero, rng):
        B = np.zeros((d, d))
        i = 0
        for _ in range(n_pos):
            B[i, i] = rng.uniform(0.5, 2.0); i += 1
        for _ in range(n_neg):
            B[i, i] = -rng.uniform(0.5, 2.0); i += 1
        for _ in range(n_rot):
            th, rho = rng.uniform(0.3, np.pi - 0.3), rng.uniform(0.5, 2.0)
            c, s = np.cos(th), np.sin(th)
            B[i:i + 2, i:i + 2] = rho * np.array([[c, -s], [s, c]])
            i += 2
        i += n_zero
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        return Q @ B @ Q.T

    def test_recovers_planted_dimensions(self):
        rng = np.random.default_rng(0)
        ch = schur_channels(self._planted(40, 6, 5, 8, 13, rng))
        self.assertEqual(ch.dims(), {"real_pos": 6, "real_neg": 5,
                                     "rotation": 16, "kernel": 13})

    def test_agrees_with_an_independently_derived_spectrum(self):
        """
        The live counterpart of the committed audit's arm C. The bucket sizes
        come off the Schur form; this reference comes off np.linalg.eigvals and
        never looks at it. Two routes, so agreement is evidence.
        """
        rng = np.random.default_rng(11)
        for d, rank in ((64, 20), (96, 30)):
            M = (rng.standard_normal((d, rank)) / np.sqrt(d)
                 @ (rng.standard_normal((rank, d)) / np.sqrt(rank)))
            ch = schur_channels(M)
            w = np.linalg.eigvals(M)
            mag = np.abs(w)
            # Same relative cut the module uses, computed the same way.
            from scipy.linalg import schur as _schur
            T = _schur(M, output="real")[0]
            nrm = float(np.linalg.norm(T, "fro"))
            tol = max(nrm * 1e-8, nrm * float(np.finfo(float).eps) * 10.0, 1e-12)
            live = mag > tol
            is_cplx = live & (np.abs(w.imag) > 1e-8 * np.maximum(mag, tol))
            is_real = live & ~is_cplx
            self.assertEqual(ch.dims()["real_pos"],
                             int(np.sum(is_real & (w.real > tol))), f"d={d}")
            self.assertEqual(ch.dims()["real_neg"],
                             int(np.sum(is_real & (w.real < -tol))), f"d={d}")
            self.assertEqual(ch.dims()["rotation"], int(np.sum(is_cplx)), f"d={d}")

    def test_rotation_channel_has_an_even_dimension(self):
        rng = np.random.default_rng(5)
        ch = schur_channels(self._planted(30, 4, 4, 7, 8, rng))
        self.assertEqual(ch.dims()["rotation"] % 2, 0)

    def test_refuses_a_non_square_matrix(self):
        with self.assertRaises(ValueError):
            schur_channels(np.zeros((3, 4)))


class TestResolutionOrder(unittest.TestCase):
    """`math-6.md` §2: S wins over A, U_pos wins over U_neg."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.L = layer_channels(_random_ovs(48, 5, 12, rng))

    def test_u_a_is_orthogonal_to_u_s(self):
        if self.L.u_a.shape[1] and self.L.u_s.shape[1]:
            self.assertLess(
                float(np.abs(self.L.u_a.T @ self.L.u_s).max()), 1e-8)

    def test_u_neg_is_orthogonal_to_u_pos(self):
        if self.L.u_neg.shape[1] and self.L.u_pos.shape[1]:
            self.assertLess(
                float(np.abs(self.L.u_neg.T @ self.L.u_pos).max()), 1e-8)

    def test_u_neg_is_the_doubly_shrunk_bucket(self):
        # Not incidental: U_neg loses its overlap with U_pos while U_A loses
        # only its overlap with the union. This asymmetry is the mechanism
        # behind the dimension gap the audit measures at ALBERT's shape.
        self.assertGreater(self.L.dims()["u_a"], self.L.dims()["u_neg"])

    def test_refuses_an_empty_layer(self):
        with self.assertRaises(ValueError):
            layer_channels([])


class TestChanceNormalization(unittest.TestCase):

    def test_chance_alignment_is_the_dimension_fraction(self):
        self.assertEqual(chance_alignment(16, 64), 0.25)

    def test_a_random_direction_shows_the_inversion_signature_raw(self):
        """
        Explanation (c), in one assertion.

        A direction drawn independently of the channels carries no operator
        information whatever. Compared RAW it reproduces the phase's headline
        pattern -- far more alignment with U_A than with U_neg. Normalized, both
        land near 1.0, which is what "no information" should look like.
        """
        rng = np.random.default_rng(21)
        L = layer_channels(_random_ovs(64, 6, 16, rng))
        raws, norms = [], []
        for _ in range(200):
            v = _unit_vec(rng, 64)
            raws.append(raw_alignment(v, L.u_a) / raw_alignment(v, L.u_neg))
            norms.append(normalized_alignment(v, L.u_a, 64)
                         / normalized_alignment(v, L.u_neg, 64))
        self.assertGreater(float(np.median(raws)), 3.0,
                           "a random direction should reproduce the raw "
                           "inversion signature; if it does not, the dimension "
                           "gap is gone and explanation (c) needs rereading")
        self.assertAlmostEqual(float(np.median(norms)), 1.0, delta=0.5)

    def test_normalized_alignment_refuses_an_empty_subspace(self):
        # 0.0 would read as "orthogonal" when it means "absent".
        with self.assertRaises(ValueError):
            normalized_alignment(np.ones(8), np.zeros((8, 0)), 8)

    def test_alignment_refuses_a_zero_vector(self):
        with self.assertRaises(ValueError):
            raw_alignment(np.zeros(8), np.eye(8)[:, :2])


class TestRandomSubspaces(unittest.TestCase):

    def test_matched_dimension_and_orthonormal(self):
        U = random_subspace(20, 7, np.random.default_rng(1))
        self.assertEqual(U.shape, (20, 7))
        self.assertLess(float(np.abs(U.T @ U - np.eye(7)).max()), 1e-10)

    def test_orthogonal_pair_is_actually_orthogonal(self):
        # The fix for the anticonservative calibration: the observed U_neg and
        # U_A are orthogonal, so the null's must be too.
        A, B = random_orthogonal_subspace_pair(30, 4, 11, np.random.default_rng(2))
        self.assertEqual((A.shape[1], B.shape[1]), (4, 11))
        self.assertLess(float(np.abs(A.T @ B).max()), 1e-10)

    def test_orthogonal_pair_refuses_dimensions_that_do_not_fit(self):
        with self.assertRaises(ValueError):
            random_orthogonal_subspace_pair(10, 6, 6)

    def test_each_half_is_still_marginally_uniform(self):
        # Splitting one Stiefel draw fixes the cross term and nothing else, so
        # the matched-dimension property has to survive. E[||P_U v||^2] = k/d.
        rng = np.random.default_rng(4)
        d, k = 24, 6
        vals = []
        for _ in range(400):
            A, _ = random_orthogonal_subspace_pair(d, k, 5, rng)
            vals.append(raw_alignment(_unit_vec(rng, d), A))
        self.assertAlmostEqual(float(np.mean(vals)), k / d, delta=0.02)


# ---------------------------------------------------------------------------
# The null
# ---------------------------------------------------------------------------

class TestAttainableFloor(unittest.TestCase):
    """
    EVALUABILITY.md: check the floor BEFORE building the null. Done, and the
    answer reframed the design.
    """

    def test_a_sign_flip_design_cannot_reject_at_one_unit(self):
        rep = R.attainable_floor_report(n_units=1)
        self.assertAlmostEqual(rep["sign_flip_floor"], 2.0 / 3.0)
        self.assertGreater(rep["sign_flip_floor"], 0.05)

    def test_this_design_can(self):
        rep = R.attainable_floor_report(n_units=1)
        self.assertAlmostEqual(rep["subspace_randomisation_floor"],
                               1.0 / (R.N_NULL_DRAWS + 1))
        self.assertLess(rep["subspace_randomisation_floor"], 0.05)

    def test_the_production_null_size_leaves_two_orders_below_alpha(self):
        self.assertLess(1.0 / (R.N_NULL_DRAWS + 1), 0.05 / 50)


class TestRefusals(unittest.TestCase):

    def setUp(self):
        self.chs, self.rng = _albert_like()
        self.dirs = [_unit_vec(self.rng, 32) for _ in self.chs]

    def test_unknown_unit(self):
        with self.assertRaises(R.NullRefused):
            R.p_value_p6_r2(self.dirs, self.chs, unit="prompt")

    def test_no_default_unit_exists(self):
        # Refusing rather than defaulting: the two units differ by orders of
        # magnitude and a default would silently pick one.
        with self.assertRaises(TypeError):
            R.p_value_p6_r2(self.dirs, self.chs)

    def test_mismatched_lengths(self):
        with self.assertRaises(R.NullRefused):
            R.p_value_p6_r2(self.dirs[:2], self.chs, unit="model")

    def test_no_layers(self):
        with self.assertRaises(R.NullRefused):
            R.p_value_p6_r2([], [], unit="model")

    def test_empty_channel(self):
        d = self.chs[0].d_model
        from p6_subspace.subspace_geometry import LayerChannels
        broken = LayerChannels(
            u_pos=self.chs[0].u_pos, u_neg=np.zeros((d, 0)),
            u_a=self.chs[0].u_a, u_s=self.chs[0].u_s,
            d_model=d, n_heads=self.chs[0].n_heads)
        with self.assertRaises(R.NullRefused):
            R.p_value_p6_r2(self.dirs, [broken] * len(self.chs), unit="model")

    def test_a_discriminant_needs_two_clusters(self):
        X = np.random.default_rng(0).standard_normal((20, 8))
        with self.assertRaises(R.NullRefused):
            R.cluster_separating_direction(X, np.zeros(20, dtype=int))

    def test_noise_points_are_not_a_cluster(self):
        # label < 0 is HDBSCAN's noise convention, carried by core.particles.
        # One real cluster plus noise is one cluster, not two.
        X = np.random.default_rng(0).standard_normal((20, 8))
        y = np.array([0] * 10 + [-1] * 10)
        with self.assertRaises(R.NullRefused):
            R.cluster_separating_direction(X, y)

    def test_a_probe_needs_a_nonempty_subspace(self):
        with self.assertRaises(R.NullRefused):
            R.probe_accuracy(np.zeros((20, 0)), np.array([0] * 10 + [1] * 10))

    def test_the_floor_refusal_fires_when_the_null_is_too_small(self):
        old = R.N_NULL_DRAWS
        R.N_NULL_DRAWS = 5          # floor 1/6 = 0.167 > alpha
        try:
            with self.assertRaises(R.NullRefused):
                R.p_value_p6_r2(self.dirs, self.chs, unit="model")
        finally:
            R.N_NULL_DRAWS = old


class TestDirectionAndPower(unittest.TestCase):
    """Known answers: the effect in each arm, and in neither."""

    def setUp(self):
        self._old = R.N_NULL_DRAWS
        R.N_NULL_DRAWS = 200
        self.chs, self.rng = _albert_like()

    def tearDown(self):
        R.N_NULL_DRAWS = self._old

    def _planted(self, attr):
        U = getattr(self.chs[0], attr)
        out = []
        for _ in self.chs:
            v = U @ self.rng.standard_normal(U.shape[1])
            out.append(v / np.linalg.norm(v))
        return out

    def test_power_when_the_direction_lies_in_u_neg(self):
        for unit in R.EXCHANGEABLE_UNITS:
            res = R.p_value_p6_r2(self._planted("u_neg"), self.chs, unit=unit)
            self.assertLessEqual(res["p_value"], 0.01, unit)

    def test_p_is_one_when_the_direction_lies_in_u_a(self):
        # The arms reversed. A construction that cannot produce p = 1 here is
        # not testing the direction it claims to.
        for unit in R.EXCHANGEABLE_UNITS:
            res = R.p_value_p6_r2(self._planted("u_a"), self.chs, unit=unit)
            self.assertGreaterEqual(res["p_value"], 0.99, unit)

    def test_the_alternative_is_fixed_in_the_module(self):
        self.assertEqual(R.ALTERNATIVE, "greater")


class TestTheExchangeableUnit(unittest.TestCase):

    def test_layer_unit_null_is_narrower_by_root_n(self):
        """
        The mechanism, pinned deterministically.

        With one direction shared across n layers -- ALBERT's case, one
        projector and n activation snapshots -- the layer unit averages n
        INDEPENDENT null draws where the model unit averages n copies of one.
        That shrinks the null's spread by about sqrt(n), and a null that is too
        narrow over-rejects. Measured at 400 replicates: 0.28 against a nominal
        0.05 at full dependence, versus 0.0575 for the model unit.
        """
        old = R.N_NULL_DRAWS
        R.N_NULL_DRAWS = 300
        try:
            n_layers = 8
            chs, rng = _albert_like(n_layers=n_layers)
            v = _unit_vec(rng, chs[0].d_model)
            dirs = [v] * n_layers
            sd = {u: R.p_value_p6_r2(dirs, chs, unit=u)["null_std"]
                  for u in ("model", "layer")}
            ratio = sd["model"] / sd["layer"]
            self.assertGreater(ratio, np.sqrt(n_layers) * 0.7)
            self.assertLess(ratio, np.sqrt(n_layers) * 1.4)
        finally:
            R.N_NULL_DRAWS = old

    def test_the_units_agree_when_the_layers_really_are_independent(self):
        # The gap is a cost of DEPENDENCE, not a constant offset. With
        # independent directions both units measured 0.0525 at alpha = 0.05.
        old = R.N_NULL_DRAWS
        R.N_NULL_DRAWS = 300
        try:
            chs, rng = _albert_like(n_layers=8)
            dirs = [_unit_vec(rng, chs[0].d_model) for _ in chs]
            sd = {u: R.p_value_p6_r2(dirs, chs, unit=u)["null_std"]
                  for u in ("model", "layer")}
            self.assertLess(sd["model"] / sd["layer"], np.sqrt(8))
        finally:
            R.N_NULL_DRAWS = old

    def test_n_units_is_recorded_and_differs(self):
        old = R.N_NULL_DRAWS
        R.N_NULL_DRAWS = 200
        try:
            chs, rng = _albert_like(n_layers=4)
            dirs = [_unit_vec(rng, chs[0].d_model) for _ in chs]
            self.assertEqual(
                R.p_value_p6_r2(dirs, chs, unit="model")["n_units"], 1)
            self.assertEqual(
                R.p_value_p6_r2(dirs, chs, unit="layer")["n_units"], 4)
        finally:
            R.N_NULL_DRAWS = old


class TestP6R4(unittest.TestCase):
    """
    P6-R4, on channels built directly rather than through the Schur pipeline.

    The pipeline is tested above; what is under test here is the null, and
    hand-built bases let the DIMENSION FRACTION be set to something
    representative. That turned out to matter. A first version used d = 24 with
    dim U_S = 14, and both arms saturated at accuracy 1.0 -- a random 14-of-24
    subspace captures a planted signal about as well as the real one, so the
    contrast had nowhere to express itself and the test read p = 1.0 as a
    failure of the construction. It was a failure of the fixture. ALBERT's real
    ratio is roughly 150 of 2048, and at a comparable fraction the statistic
    separates cleanly.

    The caveat is worth carrying: this statistic has power only while U_S is a
    SMALL fraction of d_model. A run where it is not would need saying so, not
    a p-value.
    """

    def setUp(self):
        self._old = R.N_NULL_DRAWS
        R.N_NULL_DRAWS = 60
        self.d = 48
        self.rng = np.random.default_rng(5)
        Q = random_subspace(self.d, 18, self.rng)
        u_pos, u_neg, u_a = Q[:, :3], Q[:, 3:6], Q[:, 6:18]
        from p6_subspace.subspace_geometry import LayerChannels
        self.ch = LayerChannels(
            u_pos=u_pos, u_neg=u_neg, u_a=u_a,
            u_s=np.column_stack([u_pos, u_neg]),
            d_model=self.d, n_heads=1)

    def tearDown(self):
        R.N_NULL_DRAWS = self._old

    def _data(self, carrier):
        labels = np.repeat([0, 1, 2], 20)
        base = self.rng.standard_normal((60, self.d))
        means = carrier @ self.rng.standard_normal((carrier.shape[1], 3)) * 1.2
        X = base + means[:, labels].T
        return [X, X], [labels, labels]

    def test_probe_accuracy_is_a_fraction(self):
        X, y = self._data(self.ch.u_s)
        acc = R.probe_accuracy(X[0] @ self.ch.u_s, y[0])
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_power_when_membership_lives_in_the_real_channel(self):
        X, y = self._data(self.ch.u_s)
        res = R.p_value_p6_r4(X, y, [self.ch] * 2, unit="model")
        self.assertLessEqual(res["p_value"], 0.05)
        self.assertGreater(res["observed"], res["null_mean"])

    def test_p_is_one_when_membership_lives_in_the_imaginary_channel(self):
        X, y = self._data(self.ch.u_a)
        res = R.p_value_p6_r4(X, y, [self.ch] * 2, unit="model")
        self.assertGreaterEqual(res["p_value"], 0.99)

    def test_the_dimension_fraction_this_needs_is_representative(self):
        # Pinned so a later edit does not quietly reintroduce the saturating
        # fixture and read its p = 1.0 as a finding.
        self.assertLess(self.ch.dims()["u_s"] / self.d, 0.2)

    def test_refuses_mismatched_blocks(self):
        X, y = self._data(self.ch.u_s)
        with self.assertRaises(R.NullRefused):
            R.p_value_p6_r4(X[:1], y, [self.ch] * 2, unit="model")

    def test_refuses_an_empty_real_channel(self):
        from p6_subspace.subspace_geometry import LayerChannels
        X, y = self._data(self.ch.u_s)
        empty = LayerChannels(
            u_pos=self.ch.u_pos, u_neg=self.ch.u_neg, u_a=self.ch.u_a,
            u_s=np.zeros((self.d, 0)), d_model=self.d, n_heads=1)
        with self.assertRaises(R.NullRefused):
            R.p_value_p6_r4(X, y, [empty] * 2, unit="model")

    def test_the_probe_reads_coordinates_not_the_ambient_embedding(self):
        # Projecting back into R^d would hand the probe the dimensions the
        # projection was supposed to remove.
        X, y = self._data(self.ch.u_s)
        self.assertEqual((X[0] @ self.ch.u_s).shape[1], self.ch.u_s.shape[1])


# ---------------------------------------------------------------------------
# Adjudication
# ---------------------------------------------------------------------------

class TestTheRegisteredExchangeableUnit(unittest.TestCase):
    """
    The unit is `"model"` -- the author's decision, taken 2026-08-25 before any
    p-value on real activations existed (POPPER_PLAN.md 6l).

    What these pin is that registering it lifted exactly ONE refusal. The
    module still refuses a result computed under the other unit, still declines
    to adjudicate unless asked, and the ledger is still empty.

    NOTE FOR ANYONE ADDING A TEST HERE. While no unit was registered, that
    refusal was doubling as the safety catch that kept a synthetic p-value out
    of P6-R2's real ledger slot: `adjudicate_p6_r2_r4(res, adjudicate=True)`
    could not reach `core.adjudication`. It can now, P6-R2 is classified
    `e-value` in the registry, and `adjudicate` refuses to overwrite a record
    once written -- so an accidental fixture run would permanently occupy the
    slot. Every call below that both asks to adjudicate AND uses the registered
    unit passes an isolated `adjudications_dir`. Keep it that way.
    """

    def setUp(self):
        self._old = R.N_NULL_DRAWS
        R.N_NULL_DRAWS = 200
        chs, rng = _albert_like()
        self.res = R.p_value_p6_r2(
            [_unit_vec(rng, 32) for _ in chs], chs, unit="model")

    def tearDown(self):
        R.N_NULL_DRAWS = self._old

    def test_the_registered_unit_is_model(self):
        self.assertEqual(R.REGISTERED_EXCHANGEABLE_UNIT, "model")

    def test_the_registered_unit_is_one_the_construction_can_express(self):
        self.assertIn(R.REGISTERED_EXCHANGEABLE_UNIT, R.EXCHANGEABLE_UNITS)

    def test_does_not_adjudicate_unless_asked(self):
        """
        The most important assertion in the file, and more important than it
        was: the unit refusal no longer stands behind it.
        """
        self.assertIsNone(R.adjudicate_p6_r2_r4(self.res))

    def test_a_layer_result_is_refused_on_the_unit(self):
        """
        The refusal that is now the live one. `unit=` chooses what to COMPUTE;
        the module constant chooses what may enter an e-process, so a result
        computed under the wrong unit is turned away rather than converted.
        """
        chs, rng = _albert_like()
        old = R.N_NULL_DRAWS
        R.N_NULL_DRAWS = 200
        try:
            res = R.p_value_p6_r2(
                [_unit_vec(rng, 32) for _ in chs], chs, unit="layer")
        finally:
            R.N_NULL_DRAWS = old
        with self.assertRaises(R.NullRefused) as cm:
            R.adjudicate_p6_r2_r4(res, adjudicate=True,
                                  adjudications_dir=self._tmp_dir())
        self.assertIn("'layer'", str(cm.exception))
        self.assertIn("'model'", str(cm.exception))

    def test_unregistering_the_unit_restores_the_first_refusal(self):
        """
        The None branch is still reachable and still says what to do. Kept
        exercised rather than deleted: a refusal nothing can trigger is
        indistinguishable from one that was removed.
        """
        old = R.REGISTERED_EXCHANGEABLE_UNIT
        R.REGISTERED_EXCHANGEABLE_UNIT = None
        try:
            with self.assertRaises(R.NullRefused) as cm:
                R.adjudicate_p6_r2_r4(self.res, adjudicate=True,
                                      adjudications_dir=self._tmp_dir())
            self.assertIn("exchangeable unit", str(cm.exception))
        finally:
            R.REGISTERED_EXCHANGEABLE_UNIT = old

    def test_the_registered_unit_now_reaches_the_ledger(self):
        """
        With the registered unit and `adjudicate=True`, the call now reaches
        `core.adjudication` -- which is the point of registering it. Pinned
        against an ISOLATED directory, and the real ledger is checked
        separately below.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            rec = R.adjudicate_p6_r2_r4(self.res, adjudicate=True,
                                        adjudications_dir=Path(d))
            self.assertIsNotNone(rec)
            self.assertEqual(rec["prediction_id"], "P6-R2")
            self.assertIn("unit=model", rec["test_name"])
            self.assertEqual(sorted(p.name for p in Path(d).glob("*.json")),
                             ["P6-R2.json"])

    def test_the_ledger_is_still_empty(self):
        d = REPO / "claims" / "adjudications"
        self.assertEqual(sorted(p.name for p in d.glob("*.json")), [])

    @staticmethod
    def _tmp_dir():
        import tempfile
        # Refusals raise before anything is written, so this directory is only
        # ever a place the call was NOT allowed to write to.
        return Path(tempfile.mkdtemp())


if __name__ == "__main__":
    unittest.main()
