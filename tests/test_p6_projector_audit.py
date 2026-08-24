"""
tests/test_p6_projector_audit.py — the committed projector audit, pinned.

`tools/audit_p6_projector_labels.py` settles `status-6.md` item 5 by running the
archived `subspace_build.py`. That run takes ~100 seconds and it must import
code under `archive/`, which `archive/README.md` rule 1 forbids anything live
from doing. So the audit is a tool that writes a record, and this module pins
the RECORD -- the same division CLAIM-C's calibration curve uses.

Nothing here imports the audited module. The one thing that would go stale
invisibly is the audit describing a file that has since changed, and that is a
hash of bytes, not an import.

The most important assertion in the file is `test_sensitivity_arm_caught_both`.
Arm L returned PASS on the transposed-subdiagonal breakage -- correctly, and for
a structural reason: the planted family that makes ground truth unambiguous is
normal, and a normal matrix's real Schur form cannot express that bug. Without
arm C the audit would have reported RULED-OUT on the strength of a check
incapable of failing.
"""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pytest.ini [pytest] markers.
pytestmark = pytest.mark.pure

from tools.audit_p6_projector_labels import (
    AUDITED_PATH,
    AUDIT_PATH,
    AUDIT_SCHEMA_VERSION,
    LABEL_ANGLE_TOL,
    chance_alignment_ratio,
    check_audit,
)

_BUCKETS = ("pos", "neg", "rot", "zero")


def _audit() -> dict:
    return json.loads(AUDIT_PATH.read_text())


class TestTheRecordExists(unittest.TestCase):

    def test_committed(self):
        self.assertTrue(
            AUDIT_PATH.exists(),
            f"{AUDIT_PATH} is missing. Regenerate with "
            f"`python3 tools/audit_p6_projector_labels.py --write` (~100s). It "
            f"is committed rather than computed on demand because it has to "
            f"run archived code to exist.")

    def test_schema_version(self):
        self.assertEqual(_audit()["schema_version"], AUDIT_SCHEMA_VERSION)

    def test_check_audit_is_clean(self):
        self.assertEqual(check_audit(), [])


class TestDescribesTheFileOnDisk(unittest.TestCase):
    """
    The audit's verdict is about one specific frozen file. If that file changes,
    the verdict is about something that no longer exists -- and nothing else in
    the suite would notice, because nothing live imports it.
    """

    def test_sha256_matches(self):
        rec = _audit()
        on_disk = hashlib.sha256(AUDITED_PATH.read_bytes()).hexdigest()
        self.assertEqual(
            rec["audited_sha256"], on_disk,
            f"{AUDITED_PATH.name} has changed since the audit was written. "
            f"Rerun `tools/audit_p6_projector_labels.py --write`; do not edit "
            f"the hash.")

    def test_names_the_archived_path(self):
        self.assertEqual(_audit()["audited_file"],
                         "archive/p6_subspace/subspace_build.py")


class TestLabellingArm(unittest.TestCase):
    """Arm L: planted spans, recovered."""

    def test_passes(self):
        self.assertEqual(_audit()["arm_L_labelling"]["verdict"], "PASS")

    def test_every_bucket_recovered_its_planted_dimension(self):
        for case in _audit()["arm_L_labelling"]["cases"]:
            for name in _BUCKETS:
                b = case["buckets"][name]
                self.assertEqual(
                    b["recovered_dim"], b["planted_dim"],
                    f"d={case['d']} bucket {name}: recovered "
                    f"{b['recovered_dim']} of {b['planted_dim']} planted")

    def test_worst_angle_is_within_tolerance_with_headroom(self):
        worst = _audit()["arm_L_labelling"]["worst_max_principal_angle_rad"]
        self.assertLessEqual(worst, LABEL_ANGLE_TOL)
        # The tolerance is calibrated with headroom rather than fitted to the
        # measurement: if the angles ever climb to within an order of magnitude
        # of the cut, the cut has stopped being a formality.
        self.assertLess(worst, LABEL_ANGLE_TOL / 10.0)


class TestCountArm(unittest.TestCase):
    """Arm C: bucket sizes against a spectrum taken without the Schur form."""

    def test_passes(self):
        self.assertEqual(_audit()["arm_C_counts"]["verdict"], "PASS")

    def test_runs_on_genuinely_non_normal_matrices(self):
        # The arm's whole purpose is to reach the family arm L cannot. A
        # near-normal case here would silently reintroduce arm L's blind spot.
        for case in _audit()["arm_C_counts"]["cases"]:
            self.assertGreater(
                case["commutator_fro"], 1.0,
                f"d={case['d']}: ||MM^T - M^TM||_F = {case['commutator_fro']}, "
                f"which is too close to normal for this arm to see the "
                f"transposed-subdiagonal defect")

    def test_every_case_agrees_bucket_by_bucket(self):
        for case in _audit()["arm_C_counts"]["cases"]:
            self.assertEqual(
                [case["schur_counts"][k] for k in _BUCKETS],
                [case["spectrum_counts"][k] for k in _BUCKETS],
                f"d={case['d']} rank={case['rank']}: the Schur classification "
                f"and the eigenvalue classification disagree")

    def test_real_and_complex_are_separated_by_orders_of_magnitude(self):
        # IM_REL_TOL claims seven orders of headroom on each side. Pinned, so a
        # shape change that erodes the margin surfaces here rather than as a
        # miscount later.
        for case in _audit()["arm_C_counts"]["cases"]:
            self.assertLess(case["max_rel_im_among_real"], 1e-10)
            self.assertGreater(case["min_rel_im_among_complex"], 1e-3)

    def test_at_least_one_case_carries_real_eigenvalues_of_each_sign(self):
        # The transposed extractor's signature is losing real eigenvalues into
        # rotation pairs. A sweep with no real eigenvalues anywhere would agree
        # with the broken version by having nothing to disagree about.
        cases = _audit()["arm_C_counts"]["cases"]
        self.assertTrue(any(c["spectrum_counts"]["pos"] > 0 for c in cases))
        self.assertTrue(any(c["spectrum_counts"]["neg"] > 0 for c in cases))


class TestSensitivityArm(unittest.TestCase):
    """Arm S: the audit has to be able to fail, and here is the proof."""

    def test_passes(self):
        self.assertEqual(_audit()["arm_S_sensitivity"]["verdict"], "PASS")

    def test_sensitivity_arm_caught_both(self):
        S = _audit()["arm_S_sensitivity"]
        for key in ("swapped", "transposed_subdiagonal"):
            self.assertTrue(
                S[key]["caught"],
                f"the {key} breakage was not caught by any arm; a PASS from "
                f"arms that cannot fail is not evidence that the projectors "
                f"are labelled correctly")

    def test_the_swap_is_caught_by_the_labelling_arm(self):
        S = _audit()["arm_S_sensitivity"]["swapped"]
        self.assertEqual(S["labelling"]["verdict"], "FAIL")
        # A swap sends a recovered span orthogonal to its planted one, so the
        # angle is pi/2 rather than merely "large".
        self.assertGreater(S["labelling"]["worst_max_principal_angle_rad"], 1.0)

    def test_the_transposition_is_invisible_to_the_labelling_arm(self):
        """
        Pinned as a POSITIVE assertion, because it is a property of the method
        and not an accident of these seeds.

        A planted matrix carrying known real-versus-rotational structure is
        block-diagonal in scaled rotations and real eigenvalues, which makes it
        normal; a normal matrix's real Schur form is block diagonal, so
        T[i, i+1] and T[i+1, i] agree everywhere it matters and the transposed
        extractor returns bit-identical buckets. If this ever starts FAILing,
        the planted family stopped being normal and arm C's justification needs
        rereading -- which is worth a test failure either way.
        """
        S = _audit()["arm_S_sensitivity"]["transposed_subdiagonal"]
        self.assertEqual(S["labelling"]["verdict"], "PASS")
        self.assertEqual(S["counts"]["verdict"], "FAIL")

    def test_the_asymmetry_is_written_down(self):
        note = _audit()["arm_S_sensitivity"].get("_asymmetry", "")
        self.assertIn("normal", note.lower())


class TestDimensionArm(unittest.TestCase):
    """Arm D: explanation (c), and the arithmetic that turns dims into it."""

    def test_chance_ratio_is_the_dimension_ratio(self):
        # E[||P_U v||^2] = dim U / d for a random unit v, so d cancels. Pinned
        # because the cancellation is the whole of explanation (c).
        self.assertEqual(chance_alignment_ratio(1892, 76), 1892 / 76)
        self.assertEqual(chance_alignment_ratio(10, 0), float("inf"))

    def test_recorded_ratios_recompute(self):
        for row in _audit()["arm_D_dimension"]["rows"]:
            self.assertTrue(math.isclose(
                row["chance_alignment_ratio"],
                chance_alignment_ratio(row["dim_A"], row["dim_neg"]),
                rel_tol=1e-12))
            self.assertTrue(math.isclose(
                row["chance_alignment_U_A"], row["dim_A"] / row["d_model"],
                rel_tol=1e-12))

    def test_albert_shape_is_present_exactly_once(self):
        rows = [r for r in _audit()["arm_D_dimension"]["rows"]
                if r["is_albert_xlarge_v2_shape"]]
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["d_model"], r["n_heads"], r["head_dim"]),
                         (2048, 16, 128))

    def test_normalized_alignments_recompute_from_the_recorded_dims(self):
        D = _audit()["arm_D_dimension"]
        albert = [r for r in D["rows"] if r["is_albert_xlarge_v2_shape"]][0]
        self.assertTrue(math.isclose(
            D["normalized_alignment_U_A"],
            D["observed_alignment_U_A"] / (albert["dim_A"] / albert["d_model"]),
            rel_tol=1e-12))
        self.assertTrue(math.isclose(
            D["normalized_alignment_U_neg"],
            D["observed_alignment_U_neg"] / (albert["dim_neg"] / albert["d_model"]),
            rel_tol=1e-12))

    def test_the_dimension_asymmetry_is_large_and_in_the_recorded_direction(self):
        # dim(U_A) >> dim(U_neg) is not incidental: subspace_build removes
        # span(U_pos) from U_neg AND span(U_S) from U_A, so U_neg is the
        # doubly-shrunk bucket. If this ever inverts, explanation (c) is not
        # what this audit says it is.
        for row in _audit()["arm_D_dimension"]["rows"]:
            self.assertGreater(row["dim_A"], row["dim_neg"])
        self.assertGreater(_audit()["arm_D_dimension"]["albert_shape_chance_ratio"],
                           10.0)

    def test_chance_ratio_exceeds_the_observed_ratio(self):
        """
        The finding, pinned: at ALBERT's shape a random direction would show a
        LARGER U_A-to-U_neg alignment ratio than the run actually reported.

        That is why the recorded inversion cannot be adjudicated in either
        direction. It is not that the comparison is merely uncorrected -- the
        correction is larger than the effect it is supposed to explain.
        """
        D = _audit()["arm_D_dimension"]
        self.assertGreater(D["albert_shape_chance_ratio"],
                           D["observed_alignment_ratio"])
        self.assertLess(D["observed_over_chance"], 1.0)


class TestTheVerdict(unittest.TestCase):

    def test_explanation_a_is_ruled_out(self):
        self.assertEqual(
            _audit()["explanation_a_schur_mislabelling"], "RULED-OUT")

    def test_the_scope_of_that_verdict_is_written_down(self):
        # "RULED-OUT" is a strong word on a frozen phase. What it does NOT
        # cover has to travel with it, or a later reader will over-read it.
        scope = _audit()["_explanation_a_scope"]
        self.assertIn("does NOT certify", scope)
        self.assertIn("run_6.py", scope)


if __name__ == "__main__":
    unittest.main()
