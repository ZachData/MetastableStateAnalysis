"""
tests/test_phase2b_rescaled.py — Block 1b after the S/A rewrite.

The first test class is the important one: it pins the orthogonal-invariance
identity as an identity. If `remove_rotation` ever stops reproducing
`original` exactly, that is a numerical failure of the control, and these
tests are how it surfaces instead of being read as "rotation started
contributing."

Runs under plain unittest (no torch, no pytest plugins) — every core import
these modules make is torch-free, and the degeneracy gate is passed
explicitly so `core.config` is never imported.
"""

import unittest

import numpy as np
from scipy.linalg import expm

from p2b_imaginary import p2b_energy as pe
from p2b_imaginary import rotational_rescaled as rr


GATE = 2.0  # core.config.DEGENERATE_RANK_THRESHOLD, passed explicitly


def make_ov(d, n_layers, seed=0, scale=1.0, per_layer=True):
    rng = np.random.default_rng(seed)
    mats = [scale * rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(n_layers)]
    if per_layer:
        return {
            "ov_total": mats,
            "is_per_layer": True,
            "layer_names": [f"layer_{i}" for i in range(n_layers)],
        }
    return {"ov_total": mats[0], "is_per_layer": False, "layer_names": ["shared"]}


def make_acts(n_layers, n_tokens, d, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_layers, n_tokens, d))
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# The identity
# ---------------------------------------------------------------------------

class TestRotationFrameIsAnIdentity(unittest.TestCase):
    """
    `e^{-A}` is orthogonal for antisymmetric A, so the remove_rotation frame
    cannot change any Gram-derived quantity. This is why `elim_rotation = 0.0`
    in 35/35 pre-rewrite runs was not a finding.
    """

    def test_expm_of_antisymmetric_is_orthogonal(self):
        for d in (16, 64):
            V = np.random.default_rng(0).normal(size=(d, d)) / np.sqrt(d)
            A = (V - V.T) / 2.0
            R = expm(-A)
            self.assertLess(np.abs(R @ R.T - np.eye(d)).max(), 1e-12)

    def test_gram_is_unchanged_after_many_layers(self):
        d, n_layers, n_tokens = 32, 24, 40
        ov = make_ov(d, n_layers, seed=3)
        acts = make_acts(n_layers, n_tokens, d)

        A_list = [rr.decompose_symmetric_antisymmetric(M)["A"] for M in ov["ov_total"]]
        traj = rr.rescaled_trajectory(acts, rr.build_rescalers(A_list))

        self.assertFalse(traj["truncated"])
        self.assertEqual(traj["n_valid_layers"], n_layers)
        for L in range(n_layers):
            g_orig = acts[L] @ acts[L].T
            g_rot = traj["normed"][L] @ traj["normed"][L].T
            self.assertLess(np.abs(g_rot - g_orig).max(), 1e-10,
                            f"Gram moved at layer {L}")

    def test_violation_counts_are_identical(self):
        d, n_layers, n_tokens = 24, 12, 30
        ov = make_ov(d, n_layers, seed=5, scale=2.0)
        acts = make_acts(n_layers, n_tokens, d, seed=7)

        res = rr.compare_rescaled_frames(
            acts, ov, [1.0], gate_threshold=GATE,
        )
        orig = res["frames"]["original"]["counts"][1.0]
        rot = res["frames"]["remove_rotation"]["counts"][1.0]

        self.assertEqual(orig["n_violations"], rot["n_violations"])
        self.assertEqual(orig["violation_layers"], rot["violation_layers"])
        self.assertEqual(res["invariance"]["status"], "identity_holds")
        self.assertTrue(res["invariance"]["orthogonality"]["orthogonal"])

    def test_rotation_frame_is_not_in_the_causal_comparison(self):
        """The identity must not sit in the same table as the measurements."""
        d, n_layers, n_tokens = 16, 8, 20
        res = rr.compare_rescaled_frames(
            make_acts(n_layers, n_tokens, d), make_ov(d, n_layers), [1.0],
            gate_threshold=GATE,
        )
        for row in res["comparison"].values():
            self.assertEqual(set(row), {"elim_full", "elim_signed"})

    def test_invariance_control_can_be_switched_off(self):
        d, n_layers, n_tokens = 16, 6, 20
        res = rr.compare_rescaled_frames(
            make_acts(n_layers, n_tokens, d), make_ov(d, n_layers), [1.0],
            gate_threshold=GATE, include_invariance_control=False,
        )
        self.assertNotIn("remove_rotation", res["frames"])
        self.assertIsNone(res["invariance"])


# ---------------------------------------------------------------------------
# Truncation — Phase 2's V1, in the place it does the most damage
# ---------------------------------------------------------------------------

class TestTruncationIsSurfaced(unittest.TestCase):

    def test_signed_frame_truncates_where_rotation_frame_cannot(self):
        """
        A symmetric part with large positive eigenvalues makes `e^{-S}`
        blow up while `e^{-A}` stays orthogonal. That asymmetry is exactly
        how `elim_signed = 1.0` can be manufactured by truncation.
        """
        d, n_layers, n_tokens = 16, 24, 20
        rng = np.random.default_rng(11)
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        S = Q @ np.diag(np.full(d, -6.0)) @ Q.T     # e^{-S} has norm e^{+6}
        A = rng.normal(size=(d, d)); A = (A - A.T) / 2.0
        ov = {
            "ov_total": [S + A] * n_layers,
            "is_per_layer": True,
            "layer_names": [f"layer_{i}" for i in range(n_layers)],
        }
        acts = make_acts(n_layers, n_tokens, d, seed=13)

        res = rr.compare_rescaled_frames(acts, ov, [1.0], gate_threshold=GATE)
        signed = res["frames"]["remove_signed"]
        rot = res["frames"]["remove_rotation"]

        self.assertTrue(signed["truncated"])
        self.assertEqual(signed["truncation_reason"], "rescaler_overflow")
        self.assertLess(signed["n_valid_layers"], n_layers)
        self.assertFalse(rot["truncated"])
        self.assertEqual(rot["n_valid_layers"], n_layers)

    def test_truncated_frame_is_refused_not_scored(self):
        d, n_layers, n_tokens = 16, 24, 20
        rng = np.random.default_rng(17)
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        S = Q @ np.diag(np.full(d, -6.0)) @ Q.T
        ov = {"ov_total": [S] * n_layers, "is_per_layer": True,
              "layer_names": [f"layer_{i}" for i in range(n_layers)]}
        acts = make_acts(n_layers, n_tokens, d, seed=19)

        res = rr.compare_rescaled_frames(acts, ov, [1.0], gate_threshold=GATE)
        elim = res["comparison"][1.0]["elim_signed"]

        self.assertIsNone(elim["rate"])
        self.assertIn(elim["status"],
                      ("different_transitions_scored", "no_transitions_scored",
                       "no_violations_to_eliminate"))

    def test_contracting_signed_frame_underflows_rather_than_reporting_clean(self):
        """
        The mirror of overflow, and the silent one. `e^{-S}` for positive-
        definite S contracts; once the rows underflow, `l2_normalize` leaves
        them unnormalized, the Gram goes to ~0, every energy goes to the
        constant 1/(2*beta), and the frame reports zero violations. That is
        an `elim = 1.0` manufactured out of nothing.
        """
        d, n_layers, n_tokens = 16, 40, 20
        rng = np.random.default_rng(23)
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        S = Q @ np.diag(np.full(d, 20.0)) @ Q.T     # e^{-S} contracts hard
        ov = {"ov_total": [S] * n_layers, "is_per_layer": True,
              "layer_names": [f"layer_{i}" for i in range(n_layers)]}
        acts = make_acts(n_layers, n_tokens, d, seed=29)

        traj = rr.rescaled_trajectory(acts, rr.build_rescalers([S] * n_layers))
        self.assertTrue(traj["truncated"])
        self.assertEqual(traj["truncation_reason"], "rescaler_underflow")
        self.assertLess(traj["n_valid_layers"], n_layers)

        res = rr.compare_rescaled_frames(acts, ov, [1.0], gate_threshold=GATE)
        self.assertIsNone(res["comparison"][1.0]["elim_signed"]["rate"])

    def test_n_valid_layers_survives_serialization(self):
        """Dropping this key is what made V1 unanswerable from the artifact."""
        d, n_layers, n_tokens = 16, 8, 20
        out = rr.comparison_to_json(rr.analyze_rotational_rescaling(
            make_acts(n_layers, n_tokens, d), make_ov(d, n_layers), [1.0],
            gate_threshold=GATE,
        ))
        for key, fr in out["frames"].items():
            self.assertIn("n_valid_layers", fr)
            self.assertIn("truncated", fr)
            self.assertIn("truncation_reason", fr)


# ---------------------------------------------------------------------------
# Counting rule
# ---------------------------------------------------------------------------

class TestCountingRule(unittest.TestCase):

    def test_relative_not_absolute_threshold(self):
        """
        A drop of 5e-6 on an energy of order 1 is a violation under the old
        absolute `-1e-6` rule and is not one under the project's relative
        1e-3 rule. Phase 2b used the former; Phase 1 and 2 use the latter.
        """
        E = [1.0, 1.0 - 5e-6, 1.0 - 1e-5]
        gate = [10.0, 10.0, 10.0]
        c = pe.count_violations(E, gate, gate_threshold=GATE)
        self.assertEqual(c["n_violations"], 0)
        self.assertEqual(c["n_transitions_scored"], 2)

    def test_relative_threshold_fires_on_a_real_drop(self):
        E = [1.0, 0.9, 0.95]
        gate = [10.0, 10.0, 10.0]
        c = pe.count_violations(E, gate, gate_threshold=GATE)
        self.assertEqual(c["violation_layers"], [1])
        self.assertAlmostEqual(c["max_severity"], 0.1, places=9)

    def test_gate_removes_a_transition_from_the_denominator(self):
        E = [1.0, 0.5, 0.4]
        gate = [10.0, 1.0, 10.0]          # layer 1 is degenerate
        c = pe.count_violations(E, gate, gate_threshold=GATE)
        self.assertEqual(c["violation_layers"], [2])
        self.assertEqual(c["n_transitions_scored"], 1)
        self.assertEqual(c["n_transitions_gated"], 1)

    def test_nan_energy_is_counted_as_unscored_not_as_clean(self):
        """A truncated frame must not read as 'no violations there'."""
        E = [1.0, 0.5, float("nan"), float("nan")]
        gate = [10.0, 10.0, 10.0, 10.0]
        c = pe.count_violations(E, gate, gate_threshold=GATE)
        self.assertEqual(c["n_transitions_scored"], 1)
        self.assertEqual(c["n_transitions_nan"], 2)

    def test_gate_kind_none_requires_no_gate_values(self):
        c = pe.count_violations([1.0, 0.5], gate_kind="none")
        self.assertEqual(c["n_violations"], 1)
        self.assertIsNone(c["rule"]["gate_threshold"])

    def test_missing_gate_values_raises(self):
        with self.assertRaises(ValueError):
            pe.count_violations([1.0, 0.5], None, gate_threshold=GATE)

    def test_rule_is_recorded_on_every_count(self):
        c = pe.count_violations([1.0, 0.5], [10.0, 10.0], gate_threshold=GATE)
        self.assertEqual(c["rule"]["rel_tol"], 1e-3)
        self.assertEqual(c["rule"]["gate_kind"], "normed_rank")
        self.assertEqual(c["rule"]["gate_threshold"], GATE)

    def test_effective_rank_uses_squared_singular_values(self):
        """
        The pre-rewrite local implementation normalized unsquared singular
        values, which is a different statistic with the same name.
        """
        X = np.zeros((4, 4))
        X[0, 0] = 1.0
        X[1, 1] = 1.0
        X = X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)
        s = pe.trajectory_scalars(X[None, :, :], [1.0])
        self.assertAlmostEqual(s["effective_rank"][0], 2.0, places=6)


# ---------------------------------------------------------------------------
# Elimination rate refusals
# ---------------------------------------------------------------------------

class TestEliminationRate(unittest.TestCase):

    def _count(self, n_viol, n_scored):
        E = [1.0]
        for i in range(n_viol):
            E.append(E[-1] * 0.5)
        for i in range(n_scored - n_viol):
            E.append(E[-1])
        gate = [10.0] * len(E)
        return pe.count_violations(E, gate, gate_threshold=GATE)

    def test_zero_original_violations_is_not_zero_elimination(self):
        """
        90 of Study B's 243 Pythia runs have no violations; steps 8-64 are
        clean on all 9 prompts. Scoring those as elim = 0.0 makes the phase
        return a verdict by vacuity exactly where the theorem holds.
        """
        clean = self._count(0, 5)
        res = pe.elimination_rate(clean, clean)
        self.assertIsNone(res["rate"])
        self.assertEqual(res["status"], "no_violations_to_eliminate")

    def test_negative_rate_is_not_clipped(self):
        """
        Overcorrection — the ALBERT caveat in status-2b and the unclipped
        quantity Phase 2's V2 asks for. `analysis_p2.py:153` clips this away.
        """
        orig = self._count(2, 6)
        worse = self._count(4, 6)
        res = pe.elimination_rate(orig, worse)
        self.assertEqual(res["status"], "ok")
        self.assertLess(res["rate"], 0.0)

    def test_different_denominators_refuse(self):
        a = self._count(2, 6)
        b = self._count(0, 3)
        res = pe.elimination_rate(a, b)
        self.assertIsNone(res["rate"])
        self.assertEqual(res["status"], "different_transitions_scored")

    def test_different_rules_refuse(self):
        a = pe.count_violations([1.0, 0.5], [10.0, 10.0], gate_threshold=GATE)
        b = pe.count_violations([1.0, 0.5], [10.0, 10.0], gate_threshold=3.0)
        res = pe.elimination_rate(a, b)
        self.assertEqual(res["status"], "different_counting_rule")


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

class TestInterpretation(unittest.TestCase):

    def _row(self, ef, es, status="ok", n_orig=4):
        def mk(rate):
            return {"rate": rate, "status": status, "n_original": n_orig,
                    "n_rescaled": 0, "comparable": True, "reason": None,
                    "n_scored_a": 10, "n_scored_b": 10}
        return {"elim_full": mk(ef), "elim_signed": mk(es)}

    def test_signed_matches_full(self):
        out = rr.interpret_comparison({1.0: self._row(1.0, 1.0)})
        self.assertEqual(out["overall"], "signed_carries_full_v")

    def test_full_exceeds_signed_is_an_interaction(self):
        out = rr.interpret_comparison({1.0: self._row(0.9, 0.2)})
        self.assertEqual(out["overall"], "full_v_exceeds_signed")

    def test_signed_exceeds_full(self):
        """The predicted Pythia signature: full-V inert, signed-only works."""
        out = rr.interpret_comparison({1.0: self._row(0.02, 0.95)})
        self.assertEqual(out["overall"], "signed_exceeds_full_v")

    def test_both_inert_is_its_own_verdict(self):
        """Study B's 2.1% must not read the same as 'signed carries V'."""
        out = rr.interpret_comparison({1.0: self._row(0.02, 0.02)})
        self.assertEqual(out["overall"], "both_frames_inert")

    def test_no_violations_is_not_a_verdict_about_rotation(self):
        out = rr.interpret_comparison(
            {1.0: self._row(None, None, status="no_violations_to_eliminate")}
        )
        self.assertEqual(out["overall"], "no_violations")

    def test_not_comparable_propagates(self):
        out = rr.interpret_comparison(
            {1.0: self._row(None, None, status="different_transitions_scored")}
        )
        self.assertEqual(out["overall"], "not_comparable")

    def test_no_verdict_names_rotation(self):
        """
        The old vocabulary (`rotation_neutral` / `rotation_contributes` /
        `rotation_dominant`) described the rotation-only frame, which cannot
        support any of them.
        """
        for v in rr.VERDICTS:
            self.assertNotIn("rotation", v)

    def test_overall_reads_beta_1_not_a_majority_vote(self):
        comparison = {
            0.1: self._row(0.0, 0.0),
            1.0: self._row(0.02, 0.95),
            2.0: self._row(0.0, 0.0),
            5.0: self._row(0.0, 0.0),
        }
        out = rr.interpret_comparison(comparison)
        self.assertEqual(out["reference_beta"], 1.0)
        self.assertEqual(out["overall"], "signed_exceeds_full_v")
        self.assertFalse(out["beta_dispersion"]["beta_independent"])


# ---------------------------------------------------------------------------
# Shared-weight path and serialization
# ---------------------------------------------------------------------------

class TestSharedWeightsAndSerialization(unittest.TestCase):

    def test_shared_weight_model_runs(self):
        d, n_layers, n_tokens = 16, 10, 20
        ov = make_ov(d, n_layers, per_layer=False)
        res = rr.analyze_rotational_rescaling(
            make_acts(n_layers, n_tokens, d), ov, [1.0], gate_threshold=GATE,
        )
        self.assertIn("overall", res["interpretation"])

    def test_fewer_ov_matrices_than_hidden_states(self):
        """Pythia: 25 analysed layers against 24 OV matrices."""
        d, n_layers, n_tokens = 16, 25, 20
        ov = make_ov(d, 24)
        res = rr.analyze_rotational_rescaling(
            make_acts(n_layers, n_tokens, d), ov, [1.0], gate_threshold=GATE,
        )
        self.assertEqual(res["frames"]["frames"]["original"]["n_valid_layers"], 25)

    def test_json_is_serializable_and_finite(self):
        import json
        d, n_layers, n_tokens = 16, 8, 20
        out = rr.comparison_to_json(rr.analyze_rotational_rescaling(
            make_acts(n_layers, n_tokens, d), make_ov(d, n_layers),
            [0.1, 1.0], gate_threshold=GATE,
        ))
        s = json.dumps(out)
        self.assertNotIn("NaN", s)
        self.assertNotIn("Infinity", s)

    def test_rescaler_cache_is_reused_across_prompts(self):
        d, n_layers, n_tokens = 16, 8, 20
        ov = make_ov(d, n_layers)
        cache = {}
        rr.compare_rescaled_frames(
            make_acts(n_layers, n_tokens, d, seed=1), ov, [1.0],
            rescaler_cache=cache, gate_threshold=GATE,
        )
        first = cache["S"][0].copy()
        rr.compare_rescaled_frames(
            make_acts(n_layers, n_tokens, d, seed=2), ov, [1.0],
            rescaler_cache=cache, gate_threshold=GATE,
        )
        self.assertIs(cache["S"][0].base, first.base)  # not rebuilt
        self.assertTrue(np.array_equal(cache["S"][0], first))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestGateDivergenceIsRefused(unittest.TestCase):
    """
    The third way an elimination rate gets manufactured, after overflow and
    underflow: the rescaled frame contracts directionally, drops below the
    degeneracy gate at layers the original frame passes, and so scores a
    smaller denominator. Scales with ||V||, which Study A measured as a real
    confound on these models.
    """

    def test_large_norm_ov_diverges_the_denominators_and_is_refused(self):
        d, n_layers, n_tokens = 12, 6, 10
        rng = np.random.default_rng(0)
        ov = {"ov_total": [rng.normal(size=(d, d)) for _ in range(n_layers)],
              "is_per_layer": True,
              "layer_names": [f"layer_{i}" for i in range(n_layers)]}
        acts = make_acts(n_layers, n_tokens, d, seed=2)

        res = rr.compare_rescaled_frames(acts, ov, [1.0], gate_threshold=GATE)
        orig = res["frames"]["original"]["counts"][1.0]
        signed = res["frames"]["remove_signed"]["counts"][1.0]

        self.assertGreater(orig["n_transitions_scored"],
                           signed["n_transitions_scored"])
        self.assertGreater(signed["n_transitions_gated"], 0)
        self.assertIsNone(res["comparison"][1.0]["elim_signed"]["rate"])
        self.assertEqual(res["comparison"][1.0]["elim_signed"]["status"],
                         "different_transitions_scored")

    def test_small_norm_ov_gives_a_real_number(self):
        d, n_layers, n_tokens = 12, 6, 10
        rng = np.random.default_rng(0)
        ov = {"ov_total": [0.05 * rng.normal(size=(d, d)) for _ in range(n_layers)],
              "is_per_layer": True,
              "layer_names": [f"layer_{i}" for i in range(n_layers)]}
        acts = make_acts(n_layers, n_tokens, d, seed=2)

        res = rr.compare_rescaled_frames(acts, ov, [1.0], gate_threshold=GATE)
        self.assertEqual(res["comparison"][1.0]["elim_signed"]["status"], "ok")
        self.assertIsNotNone(res["comparison"][1.0]["elim_full"]["rate"])
        # and the identity still holds in the benign regime
        self.assertEqual(res["invariance"]["status"], "identity_holds")
