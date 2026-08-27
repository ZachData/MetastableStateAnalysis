"""
tests/test_p6_r2_r4_dry_run.py — P6-R2/R4 run on inputs whose answer is known.

`tools/dry_run_p6_r2_r4.py` runs P6-R2 on a separating direction planted inside
each channel in turn, sweeps the H0 rejection rate against how far the union
sits above chance, and measures P6-R4 where `U_S` captures more variance than
chance. That takes about twenty minutes, so the record is committed and this
module pins it — the division `tests/test_claim_c_dry_run.py` and
`tests/test_p_st1_dry_run.py` already use.

The assertion that matters most is `TestTheRetirementIsSupported`. P6-R2's null
was changed on the evidence in this record, and if the record stops showing the
retired null failing then the change is not supported by the artifact that is
supposed to support it — so the test fails rather than quietly agreeing with
the module. `POPPER_PLAN.md` §6h found an audit arm reporting PASS while
incapable of failing; this is the same question asked of a retirement.

`TestP6R4WasLeftAlone` is the other half. Leaving P6-R4 unchanged is a decision
and not an omission, and the measurement behind it belongs in the gate.
"""

from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pytest.ini [pytest] markers.
pytestmark = pytest.mark.pure

from p6_subspace import r2_r4_null as R
from p6_subspace import subspace_geometry as SG
from tools.dry_run_p6_r2_r4 import (
    GEOMETRY_PATH,
    NULL_PATH,
    RECORD_PATH,
    RECORD_SCHEMA_VERSION,
    check_record,
    random_split,
    union_with_alignment,
)


def _record() -> dict:
    return json.loads(RECORD_PATH.read_text())


class TestTheRecordIsPresentAndCurrent(unittest.TestCase):

    def test_it_exists_and_parses(self):
        self.assertTrue(RECORD_PATH.exists(),
                        f"{RECORD_PATH} is missing; run "
                        f"`python3 -m tools.dry_run_p6_r2_r4 --write`")
        self.assertEqual(_record()["schema_version"], RECORD_SCHEMA_VERSION)

    def test_check_record_is_clean(self):
        self.assertEqual(check_record(), [])

    def test_it_describes_the_files_on_disk(self):
        rec = _record()
        for key, path in (("null", NULL_PATH), ("geometry", GEOMETRY_PATH)):
            self.assertEqual(
                rec[f"{key}_sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
                f"{path.name} has changed since the record was written; "
                f"rerun --write rather than editing the hash")

    def test_it_is_about_the_null_the_module_adjudicates(self):
        self.assertEqual(_record()["null_family"], R.NULL_FAMILY)

    def test_it_is_not_an_adjudication(self):
        self.assertIn("not an adjudication", _record()["_not"])


class TestThePlantedAnswerComesBack(unittest.TestCase):

    def test_every_row_returned_its_known_answer(self):
        ka = _record()["known_answer"]
        self.assertTrue(ka["every_row_as_expected"])
        self.assertEqual({r["planted_in"] for r in ka["rows"]},
                         {"u_neg", "u_a", "neither"})

    def test_the_reversed_arms_row_is_present_and_returns_one(self):
        """A construction that cannot produce p = 1 with the arms reversed is
        not testing the direction it claims to."""
        row = next(r for r in _record()["known_answer"]["rows"]
                   if r["planted_in"] == "u_a")
        for p in row["p_values"]:
            self.assertGreaterEqual(p, 0.99)


class TestTheRetirementIsSupported(unittest.TestCase):
    """
    P6-R2's null was changed on this evidence. Both directions are asserted:
    the adjudicated null has to hold, and the retired one has to still fail.
    """

    def test_the_adjudicated_null_holds_across_the_sweep(self):
        s = _record()["union_alignment_sweep"]
        self.assertTrue(s["adjudicated_null_holds"],
                        f"range {s['adjudicated_null_range']}")

    def test_the_adjudicated_null_does_not_drift_with_the_union(self):
        """Flat is the claim: the union's alignment is what the retired null
        failed to reproduce, so the replacement must not respond to it."""
        self.assertTrue(_record()["union_alignment_sweep"]
                        ["adjudicated_null_is_flat"])

    def test_the_retired_null_still_rises_with_the_union_alignment(self):
        s = _record()["union_alignment_sweep"]
        self.assertTrue(
            s["retired_null_rises_with_alignment"],
            f"retired null range {s['retired_null_range']}; it was retired on "
            f"that trend and this record no longer shows it")

    def test_the_sweep_actually_spans_a_range_of_alignments(self):
        """A sweep whose cells all sit at chance could not have shown the
        trend, which would make the previous assertion vacuous."""
        rows = _record()["union_alignment_sweep"]["rows"]
        aligns = [r["mean_union_alignment"] for r in rows]
        self.assertLess(min(aligns), 1.2)
        self.assertGreater(max(aligns), 3.0)

    def test_no_power_was_lost(self):
        self.assertTrue(_record()["power"]["no_power_lost"])

    def test_the_mechanism_re_derived_here(self):
        """
        Milliseconds, and independent of the record: a re-split keeps the
        union's alignment exactly and a matched-dimension pair does not. That
        is the whole reason one null holds where the other does not.
        """
        rng = np.random.default_rng(3)
        v = rng.normal(size=64)
        v /= np.linalg.norm(v)
        U = union_with_alignment(rng, v, 0.99, d=64, k=24)
        obs = SG.normalized_alignment(v, U, 64)
        for _ in range(3):
            a, b = random_split(rng, U, 8)
            self.assertAlmostEqual(
                SG.normalized_alignment(v, np.hstack([a, b]), 64), obs,
                places=8)
        for _ in range(3):
            a, b = SG.random_orthogonal_subspace_pair(64, 8, 16, rng)
            self.assertLess(
                SG.normalized_alignment(v, np.hstack([a, b]), 64), 0.7 * obs)


class TestP6R4WasLeftAlone(unittest.TestCase):
    """Leaving it unchanged is a decision, and this is the measurement."""

    def test_its_rate_holds_where_u_s_captures_more_than_chance(self):
        r4 = _record()["r4_variance_sweep"]
        self.assertTrue(r4["holds"], f"range {r4['range']}")

    def test_the_sweep_actually_elevated_the_variance_capture(self):
        caps = [r["mean_variance_capture"]
                for r in _record()["r4_variance_sweep"]["rows"]]
        self.assertGreater(max(caps), 2.0,
                           "an arm that never elevated the capture could not "
                           "have shown P6-R4 failing, so its holding proves "
                           "nothing")

    def test_the_module_still_uses_a_matched_dimension_control_for_r4(self):
        import inspect
        src = inspect.getsource(R.p_value_p6_r4)
        self.assertIn("random_subspace(", src)
        self.assertNotIn("resplit_union", src)
