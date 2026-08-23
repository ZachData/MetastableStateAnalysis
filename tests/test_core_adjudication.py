"""
tests/test_core_adjudication.py — the emission layer (POPPER_PLAN.md B4).

The refusals are the part worth testing hardest. `core/evalues.py` has a proof
behind it; this module has a *policy* behind it, and a policy only holds if the
paths that would violate it actually raise. Half of what follows is checking
that things do NOT happen.

Every test builds its own registry and adjudications directory in tmp_path
rather than touching `claims/`. That is not only hygiene: the real ledger is
append-only by design (adjudicate refuses to overwrite), so a test that wrote
into it would leave a record no later run could clear.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure

from core.adjudication import (
    AdjudicationRefused,
    adjudicate,
    adjudicate_if_registered,
    all_claim_processes,
    claim_process,
    hash_artifact,
    load_adjudications,
    registry_entry,
    verify_ledger,
)
from core.evalues import calibrate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _entry(pid, claim="H-TEST", evaluable="e-value", relevance=1.0):
    return {
        "id": pid, "claim": claim, "statement": f"{pid} statement",
        "h0": f"{pid} null", "h1": f"{pid} alternative",
        "falsifier": "opposite direction", "instrument": "test",
        "cost": "R", "evaluable": evaluable,
        "null_construction": "permutation over the matched baseline",
        "relevance": relevance, "source": "test",
        "registered_commit": "abc123", "registered_date": "2026-01-01",
        "registered_provenance": "gated", "superseded_by": None, "notes": "",
    }


@pytest.fixture
def registry():
    return {
        "schema_version": 1, "kappa": 0.5, "alpha": 0.05,
        "relevance_threshold": 0.6,
        "predictions": [
            _entry("T-OK1"),
            _entry("T-OK2"),
            _entry("T-MEASURE", evaluable="measurement"),
            _entry("T-NEEDSNULL", evaluable="needs-null"),
            _entry("T-IRRELEVANT", relevance=0.4),
            _entry("T-OTHERCLAIM", claim="H-OTHER"),
        ],
    }


@pytest.fixture
def ledger(tmp_path):
    d = tmp_path / "adjudications"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Refusals -- the policy
# ---------------------------------------------------------------------------

class TestRefusals:

    def test_unregistered_prediction_refused(self, registry, ledger):
        with pytest.raises(AdjudicationRefused, match="no entry in claims/registry.json"):
            adjudicate("T-GHOST", 0.01, [], registry=registry, adjudications_dir=ledger)

    def test_measurement_refused(self, registry, ledger):
        """
        The load-bearing refusal. A `measurement` entry has no valid null by
        construction (P-H1 is the real instance: Wendel's theorem gives
        probability 1 for d > n), so an e-value from it would be evidence
        extracted from a theorem -- and because e-values multiply it would void
        the guarantee for every other prediction on the claim.
        """
        with pytest.raises(AdjudicationRefused) as exc:
            adjudicate("T-MEASURE", 0.001, [], registry=registry, adjudications_dir=ledger)
        assert "'measurement'" in str(exc.value)
        assert "multiply" in str(exc.value)
        assert not list(ledger.glob("*.json"))     # nothing written

    def test_needs_null_refused(self, registry, ledger):
        with pytest.raises(AdjudicationRefused, match="needs-null"):
            adjudicate("T-NEEDSNULL", 0.001, [], registry=registry, adjudications_dir=ledger)

    def test_below_relevance_floor_refused(self, registry, ledger):
        with pytest.raises(AdjudicationRefused, match="relevance"):
            adjudicate("T-IRRELEVANT", 0.001, [], registry=registry, adjudications_dir=ledger)

    def test_double_adjudication_refused(self, registry, ledger):
        adjudicate("T-OK1", 0.01, [], registry=registry, adjudications_dir=ledger)
        with pytest.raises(AdjudicationRefused, match="already been adjudicated"):
            adjudicate("T-OK1", 0.0001, [], registry=registry, adjudications_dir=ledger)

    def test_double_adjudication_does_not_overwrite(self, registry, ledger):
        """A refused re-adjudication must leave the original record untouched."""
        adjudicate("T-OK1", 0.01, [], registry=registry, adjudications_dir=ledger)
        before = (ledger / "T-OK1.json").read_text()
        with pytest.raises(AdjudicationRefused):
            adjudicate("T-OK1", 1e-9, [], registry=registry, adjudications_dir=ledger)
        assert (ledger / "T-OK1.json").read_text() == before

    def test_refusal_is_not_a_valueerror(self, registry, ledger):
        """
        A numeric pipeline wrapping `except ValueError` must not swallow a
        refusal: a swallowed refusal looks exactly like an adjudication that
        was never requested.
        """
        assert not issubclass(AdjudicationRefused, ValueError)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

class TestEmission:

    def test_record_written_with_expected_fields(self, registry, ledger):
        rec = adjudicate("T-OK1", 0.01, ["deadbeef"], run_manifest={"model": "x"},
                         test_name="mannwhitneyu, one-sided greater",
                         registry=registry, adjudications_dir=ledger)
        assert (ledger / "T-OK1.json").exists()
        on_disk = json.loads((ledger / "T-OK1.json").read_text())
        assert on_disk == rec
        for key in ("prediction_id", "claim", "p_value", "e_value", "kappa", "alpha",
                    "artifact_hashes", "run_manifest", "adjudicated_at",
                    "claim_log_E_after", "claim_E_after", "claim_decision_after",
                    "test_name", "schema_version"):
            assert key in rec
        assert rec["e_value"] == pytest.approx(calibrate(0.01, 0.5))
        assert rec["test_name"] == "mannwhitneyu, one-sided greater"

    def test_dry_run_writes_nothing(self, registry, ledger):
        rec = adjudicate("T-OK1", 0.01, [], registry=registry,
                         adjudications_dir=ledger, dry_run=True)
        assert rec["prediction_id"] == "T-OK1"
        assert not list(ledger.glob("*.json"))

    def test_missing_artifact_hashes_is_recorded_not_silent(self, registry, ledger):
        """
        Gate rule 3 reads artifact hashes. An adjudication with none is allowed
        (synthetic tests have no artifacts) but must say so in the record --
        otherwise a reader cannot tell "no artifacts" from "nobody recorded
        them", and the gate silently checks nothing.
        """
        rec = adjudicate("T-OK1", 0.01, [], registry=registry, adjudications_dir=ledger)
        assert "NO ARTIFACT HASHES RECORDED" in rec["notes"]

    def test_notes_preserved_alongside_the_warning(self, registry, ledger):
        rec = adjudicate("T-OK1", 0.01, [], notes="synthetic fixture",
                         registry=registry, adjudications_dir=ledger)
        assert "synthetic fixture" in rec["notes"]
        assert "NO ARTIFACT HASHES" in rec["notes"]

    def test_accumulates_across_a_claim(self, registry, ledger):
        a = adjudicate("T-OK1", 0.01, ["h1"], registry=registry, adjudications_dir=ledger)
        b = adjudicate("T-OK2", 0.05, ["h2"], registry=registry, adjudications_dir=ledger)
        assert a["claim_sequence_index"] == 1
        assert b["claim_sequence_index"] == 2
        assert b["claim_E_after"] == pytest.approx(calibrate(0.01) * calibrate(0.05))

    def test_other_claims_are_unaffected(self, registry, ledger):
        adjudicate("T-OK1", 1e-6, ["h1"], registry=registry, adjudications_dir=ledger)
        procs = all_claim_processes(ledger, registry)
        assert len(procs["H-TEST"].adjudications) == 1
        assert len(procs["H-OTHER"].adjudications) == 0
        assert procs["H-OTHER"].E == pytest.approx(1.0)

    def test_next_p_needed_reported(self, registry, ledger):
        rec = adjudicate("T-OK1", 0.5, ["h1"], registry=registry, adjudications_dir=ledger)
        assert 0.0 < rec["next_p_needed"] < 1.0

    def test_decision_crosses_threshold(self, registry, ledger):
        rec = adjudicate("T-OK1", 1e-6, ["h1"], registry=registry, adjudications_dir=ledger)
        assert rec["claim_decision_after"] == "reject_null"

    def test_weak_evidence_is_not_acceptance(self, registry, ledger):
        rec = adjudicate("T-OK1", 0.95, ["h1"], registry=registry, adjudications_dir=ledger)
        assert rec["claim_E_after"] < 1.0
        assert rec["claim_decision_after"] == "insufficient_evidence"


# ---------------------------------------------------------------------------
# Non-raising wrapper, for phase runners
# ---------------------------------------------------------------------------

class TestAdjudicateIfRegistered:

    def test_none_p_value_is_not_a_refusal(self, registry, ledger):
        """
        `compare_induction_vs_semantic` returns mwu_pvalue=None when there are
        too few heads to test. That is "the test could not run", not a policy
        violation, and must pass through quietly.
        """
        assert adjudicate_if_registered("T-OK1", None, registry=registry,
                                        adjudications_dir=ledger) is None
        assert not list(ledger.glob("*.json"))

    def test_refusal_returns_none_and_reports(self, registry, ledger, capsys):
        out = adjudicate_if_registered("T-MEASURE", 0.01, registry=registry,
                                       adjudications_dir=ledger)
        assert out is None
        assert "adjudication refused" in capsys.readouterr().err

    def test_success_returns_the_record(self, registry, ledger):
        rec = adjudicate_if_registered("T-OK1", 0.01, ["h"], registry=registry,
                                       adjudications_dir=ledger)
        assert rec is not None and rec["prediction_id"] == "T-OK1"


# ---------------------------------------------------------------------------
# The ledger reads back, and verifies itself
# ---------------------------------------------------------------------------

class TestLedger:

    def test_ordered_by_adjudication_time_not_filename(self, registry, ledger):
        """
        An e-process is a sequence; the interim E is what optional stopping is
        valid against. Sorting by filename would give the right final product
        with the wrong trajectory.
        """
        adjudicate("T-OK2", 0.2, ["h"], registry=registry, adjudications_dir=ledger)
        adjudicate("T-OK1", 0.3, ["h"], registry=registry, adjudications_dir=ledger)
        ids = [r["prediction_id"] for r in load_adjudications(ledger)]
        assert ids == ["T-OK2", "T-OK1"]        # adjudication order, not alphabetical

    def test_process_rebuilds_from_records(self, registry, ledger):
        adjudicate("T-OK1", 0.02, ["h"], registry=registry, adjudications_dir=ledger)
        adjudicate("T-OK2", 0.4, ["h"], registry=registry, adjudications_dir=ledger)
        proc = claim_process("H-TEST", ledger, registry)
        assert proc.E == pytest.approx(calibrate(0.02) * calibrate(0.4))

    def test_verify_clean_ledger(self, registry, ledger):
        adjudicate("T-OK1", 0.02, ["h"], registry=registry, adjudications_dir=ledger)
        adjudicate("T-OK2", 0.4, ["h"], registry=registry, adjudications_dir=ledger)
        assert verify_ledger(ledger, registry) == []

    def test_verify_catches_a_hand_edited_e_value(self, registry, ledger):
        adjudicate("T-OK1", 0.02, ["h"], registry=registry, adjudications_dir=ledger)
        f = ledger / "T-OK1.json"
        rec = json.loads(f.read_text())
        rec["e_value"] = 9999.0
        f.write_text(json.dumps(rec))
        problems = verify_ledger(ledger, registry)
        assert any("has been edited" in p for p in problems)

    def test_verify_catches_a_hand_edited_decision(self, registry, ledger):
        adjudicate("T-OK1", 0.5, ["h"], registry=registry, adjudications_dir=ledger)
        f = ledger / "T-OK1.json"
        rec = json.loads(f.read_text())
        rec["claim_decision_after"] = "reject_null"
        f.write_text(json.dumps(rec))
        problems = verify_ledger(ledger, registry)
        assert any("claim_decision_after" in p for p in problems)

    def test_verify_catches_a_deleted_record(self, registry, ledger):
        """Deleting an earlier record changes every later record's claimed E."""
        adjudicate("T-OK1", 0.02, ["h"], registry=registry, adjudications_dir=ledger)
        adjudicate("T-OK2", 0.4, ["h"], registry=registry, adjudications_dir=ledger)
        (ledger / "T-OK1.json").unlink()
        problems = verify_ledger(ledger, registry)
        assert any("does not match the replayed" in p for p in problems)

    def test_verify_catches_a_record_that_should_not_exist(self, registry, ledger):
        """
        A record for a `measurement` prediction cannot be produced by
        `adjudicate`, but it can be produced by hand -- and that is exactly the
        thing that would silently void a claim's guarantee.
        """
        (ledger / "T-MEASURE.json").write_text(json.dumps({
            "prediction_id": "T-MEASURE", "claim": "H-TEST", "p_value": 0.001,
            "e_value": calibrate(0.001), "kappa": 0.5,
            "adjudicated_at": "2026-01-01T00:00:00+00:00",
            "claim_log_E_after": 0.0, "claim_decision_after": "reject_null",
        }))
        problems = verify_ledger(ledger, registry)
        assert any("must not contribute" in p for p in problems)


class TestHashing:

    def test_hash_is_of_bytes_not_path(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"same"); b.write_bytes(b"same")
        assert hash_artifact(a) == hash_artifact(b)
        b.write_bytes(b"different")
        assert hash_artifact(a) != hash_artifact(b)


class TestRegistryEntry:

    def test_entry_fields_read_through(self, registry):
        e = registry_entry("T-OK1", registry)
        assert e.claim == "H-TEST"
        assert e.evaluable == "e-value"
        assert e.relevance == 1.0


class TestRealRegistry:
    """
    Against the project's actual claims/registry.json, so the module cannot
    drift from the file it exists to read.
    """

    def test_real_registry_loads_and_every_claim_has_a_process(self):
        procs = all_claim_processes()
        assert set(procs) >= {"H-RESIST", "H-TRANSFER", "H-EMERGE", "H-OPERATOR", "H-BRIDGE"}

    def test_real_ledger_verifies(self):
        assert verify_ledger() == []

    def test_p6_i1_is_adjudicable(self):
        """
        The first prediction threaded end-to-end (POPPER_PLAN.md B6-first).
        If this stops being classified 'e-value', the wiring in
        p6_subspace/induction_ov.py must change with it rather than silently
        starting to refuse at runtime.
        """
        assert registry_entry("P6-I1").evaluable == "e-value"
