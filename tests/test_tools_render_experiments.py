"""
tests/test_tools_render_experiments.py — the phase-to-e-value map.

`claims/EXPERIMENTS.md` is generated, so the interesting failures are not in its
prose but in the two joins it computes that nothing else in the project computes:
which evidence artifact covers which prediction, and which phase on disk has no
falsifier at all. Both were wrong in the obvious way first — a substring scan for
the prediction id — and both were wrong in the *flattering* direction, which is
the reason they are tested here rather than left to the `--check` staleness gate.

Tier: pure. The renderer is stdlib-only by contract (it runs in tier 0, where
nothing is installed), so these tests import it directly.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.pure

from tools import render_experiments as rx


# ---------------------------------------------------------------------------
# Gate resolution
# ---------------------------------------------------------------------------

class TestGateResolution:
    """
    Every `gate` in the registry names a module that exists and a function it
    defines. This is the check that stops a prediction reading as "has an
    instrument" in every generated table while pointing at nothing.
    """

    def test_every_registered_gate_resolves(self):
        reg = rx.load_registry()
        bad = []
        for p in reg["predictions"]:
            ok, why = rx.resolve_gate(p.get("gate", ""))
            if not ok:
                bad.append((p["id"], p.get("gate"), why))
        assert bad == [], f"unresolved gates: {bad}"

    def test_every_adjudicable_prediction_names_a_gate(self):
        """
        The registry's own definition of adjudicable is `e-value` + active +
        relevance at or above r0. Such a row with no gate cannot be run at all,
        and `claims/FALSIFICATION.md` would still count it toward "adjudicable
        now".
        """
        reg = rx.load_registry()
        r0 = float(reg["relevance_threshold"])
        missing = [p["id"] for p in reg["predictions"]
                   if p["evaluable"] == "e-value"
                   and p.get("status", "active") == "active"
                   and float(p["relevance"]) >= r0
                   and not p.get("gate")]
        assert missing == [], f"adjudicable with no gate: {missing}"

    def test_a_gate_naming_a_missing_function_is_refused(self):
        ok, why = rx.resolve_gate("core.evalues:no_such_function")
        assert not ok and "defines no" in why

    def test_a_gate_naming_a_missing_module_is_refused(self):
        ok, why = rx.resolve_gate("core.not_a_module:calibrate")
        assert not ok and "does not exist" in why

    def test_a_malformed_gate_is_refused_rather_than_split_wrongly(self):
        ok, why = rx.resolve_gate("core.evalues.calibrate")
        assert not ok and "module.path:function" in why


# ---------------------------------------------------------------------------
# Evidence linking — the join that was wrong in the flattering direction
# ---------------------------------------------------------------------------

class TestEvidenceLinking:
    """
    A dry run covers the prediction whose GATE FILE it hashed, not every
    prediction its prose mentions.

    `claims/audits/p_t1_p_m1_dry_run.json` cites `CLAIM-B`, `CLAIM-C`, `P6-R2`,
    `P6-R4`, `P-S1` and `P-ST1` — cross-references to what earlier passes found.
    A substring scan credits all six with a dry run they never had, and the two
    rows that genuinely have none (`P-AB1`, `P-I3`) are the only reason this
    file's Gaps section exists. Getting this wrong would delete the finding.
    """

    def test_p_t1_p_m1_dry_run_covers_exactly_its_two_gates(self):
        dry, _ = rx._evidence_index()
        covered = {pid for pid, files in dry.items()
                   if "p_t1_p_m1_dry_run.json" in files}
        assert covered == {"P-T1", "P-M1"}

    def test_a_prediction_merely_cited_by_an_audit_is_not_credited(self):
        """
        The specific over-credit the id scan produced, asserted against whatever
        the audits currently cite rather than against a list written here — the
        cross-references move as passes are added, and a test pinned to today's
        set would fail on a regenerated record for the wrong reason.
        """
        import re
        dry, _ = rx._evidence_index()
        reg = rx.load_registry()
        gate_of = {p["id"]: p.get("gate", "").split(":")[0].replace(".", "/") + ".py"
                   for p in reg["predictions"] if p.get("gate")}

        checked = 0
        for f in sorted(rx.AUDITS.glob("*_dry_run.json")):
            text = f.read_text(encoding="utf-8")
            reached = rx._modules_named(text)
            for pid, gate_mod in gate_of.items():
                cited = re.search(rf"\b{re.escape(pid)}\b", text) is not None
                if cited and gate_mod not in reached:
                    checked += 1
                    assert f.name not in dry.get(pid, set()), (
                        f"{pid} is cited by {f.name} but its gate {gate_mod} is "
                        f"not among the files that record measured; citation is "
                        f"not coverage")
        assert checked > 0, "no audit cites a prediction it does not cover; " \
                            "this test is no longer exercising anything"

    def test_calibration_resolves_through_the_tool_that_generated_it(self):
        """
        `cross_head_association.json` names only `tools/calibrate_cross_head_
        association.py`; the gate it measures is reached one hop through that
        tool's imports. Without the hop `P-I3` reads as having no calibration
        either, which overstates the gap.
        """
        _, calib = rx._evidence_index()
        assert "cross_head_association.json" in calib.get("P-I3", set())
        assert "patching_exponent.json" in calib.get("P-AB1", set())

    def test_prerequisite_audits_are_not_counted_as_dry_runs(self):
        """
        `p6_projector_labels.json` audits an archived module against a status
        note; `p_i1_attainable_floor.json` is arithmetic on a design. Neither is
        a run on an input whose verdict is fixed a priori, and counting them
        would report the dry-run queue as more complete than it is.
        """
        dry, _ = rx._evidence_index()
        counted = {f for files in dry.values() for f in files}
        assert "p6_projector_labels.json" not in counted
        assert "p_i1_attainable_floor.json" not in counted


# ---------------------------------------------------------------------------
# The gaps the map exists to surface
# ---------------------------------------------------------------------------

class TestGaps:

    def test_every_phase_on_disk_is_either_registered_or_explained(self):
        """
        A phase absent from both the registry and `PHASES_WITHOUT_PREDICTIONS`
        is reported as UNEXPLAINED rather than silently omitted. This test does
        not require the list to be empty — three active phases are legitimately
        on it today — only that no phase falls through both.
        """
        reg = rx.load_registry()
        registered = {p["phase"] for p in reg["predictions"]}
        on_disk = set(rx._phases_on_disk())
        unaccounted = on_disk - registered - set(rx.PHASES_WITHOUT_PREDICTIONS)
        assert unaccounted == set(), (
            f"phases with neither a prediction nor a recorded reason: {unaccounted}")

    def test_the_unexplained_phases_are_named_in_the_output(self):
        """
        7d, 7e and 8 are the active phases and carry no falsifier. The map has
        to say so in the rendered text, not only in a dict a reader never opens.
        """
        out = rx.render()
        for phase in ("7d", "7e", "8"):
            assert f"`{phase}`" in out
        assert "UNEXPLAINED" in out

    def test_declared_claims_with_no_prediction_are_reported(self):
        out = rx.render()
        reg = rx.load_registry()
        named = {p["claim"] for p in reg["predictions"]}
        for claim in set(rx._declared_claims()) - named:
            assert claim in out, f"{claim} has no prediction and is not reported"

    def test_rows_without_a_dry_run_are_reported(self):
        dry, _ = rx._evidence_index()
        reg = rx.load_registry()
        r0 = float(reg["relevance_threshold"])
        out = rx.render()
        for p in reg["predictions"]:
            adjudicable = (p["evaluable"] == "e-value"
                           and p.get("status", "active") == "active"
                           and float(p["relevance"]) >= r0)
            if adjudicable and not dry.get(p["id"]):
                assert f"**`{p['id']}`**" in out, (
                    f"{p['id']} has no dry run and is not reported in Gaps")


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

class TestCommittedFile:

    def test_the_committed_file_is_in_step(self):
        """
        What `scripts/check.sh lint` enforces, asserted here too so a developer
        sees it at test time rather than at merge time.
        """
        assert rx.OUT.exists(), "claims/EXPERIMENTS.md has not been generated"
        assert rx.OUT.read_text(encoding="utf-8") == rx.render(), (
            "claims/EXPERIMENTS.md is stale; run "
            "`python3 tools/render_experiments.py`")

    def test_render_is_deterministic(self):
        assert rx.render() == rx.render()

    def test_check_mode_passes_on_the_committed_file(self):
        assert rx.main(["--check"]) == 0


# ---------------------------------------------------------------------------
# The registry fields the map is built on
# ---------------------------------------------------------------------------

class TestRegistryJoinFields:

    def test_every_prediction_carries_the_join_keys(self):
        reg = rx.load_registry()
        for p in reg["predictions"]:
            for field in ("phase", "experiment", "gate"):
                assert field in p, f"{p['id']}: missing {field!r}"

    def test_phase_values_point_at_a_directory_or_are_explained(self):
        reg = rx.load_registry()
        on_disk = rx._phases_on_disk()
        for p in reg["predictions"]:
            assert p["phase"] in on_disk, (
                f"{p['id']}: phase {p['phase']!r} has no directory on disk")

    def test_the_join_keys_are_not_frozen_fields(self):
        """
        `phase`, `experiment` and `gate` describe where an instrument lives;
        they carry no falsifier content. If one were added to FROZEN_FIELDS the
        pre-registration gate would start failing an entry for a directory
        rename, which is a bookkeeping change and not an amendment.
        """
        from tools.check_registry import FROZEN_FIELDS
        for field in ("phase", "experiment", "gate"):
            assert field not in FROZEN_FIELDS
