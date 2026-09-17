"""
tools/check_registry.py's evidence-path rules: `null_module`,
`calibration_record` and `real_run_record` are each null or a git-tracked path
that exists. A boolean saying "built" or "run" cannot be checked; a path can,
and these tests are what make the difference enforceable rather than stated.
"""

from __future__ import annotations

import copy
import json

import pytest

from tools.check_registry import REGISTRY, check_registry

pytestmark = pytest.mark.pure


@pytest.fixture
def reg():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _entry(reg, pid):
    return next(p for p in reg["predictions"] if p["id"] == pid)


def _errors(reg, tracked):
    msgs: list[str] = []
    check_registry(reg, msgs, tracked=tracked)
    return [m for m in msgs if m.startswith("ERROR")]


def _all_named_paths(reg):
    return {p[f] for p in reg["predictions"]
            for f in ("null_module", "calibration_record", "real_run_record") if p[f]}


def test_committed_registry_passes_with_its_paths_tracked(reg):
    assert _errors(reg, tracked=_all_named_paths(reg)) == []


def test_missing_path_is_an_error(reg):
    bad = copy.deepcopy(reg)
    _entry(bad, "CLAIM-C")["calibration_record"] = "claims/audits/does_not_exist.json"
    errs = _errors(bad, tracked=_all_named_paths(bad))
    assert any("CLAIM-C" in e and "does not exist" in e for e in errs)


def test_untracked_path_is_an_error(reg):
    bad = copy.deepcopy(reg)
    tracked = _all_named_paths(bad) - {"p1_mstate_tracking/replication_gate.py"}
    errs = _errors(bad, tracked=tracked)
    assert any("CLAIM-C" in e and "not git-tracked" in e for e in errs)


def test_real_run_without_built_null_is_an_error(reg):
    bad = copy.deepcopy(reg)
    e = _entry(bad, "CLAIM-C")
    e["real_run_record"] = e["calibration_record"]
    e["null_module"] = None
    errs = _errors(bad, tracked=_all_named_paths(bad))
    assert any("CLAIM-C" in e_ and "null_module is null" in e_ for e_ in errs)


def test_measurement_with_a_null_module_is_an_error(reg):
    bad = copy.deepcopy(reg)
    _entry(bad, "P-H1")["null_module"] = "p1c_frames/hemisphere_feasibility.py"
    tracked = _all_named_paths(bad)
    errs = _errors(bad, tracked=tracked)
    assert any("P-H1" in e and "measurement" in e for e in errs)


def test_unknown_phase_is_an_error(reg):
    bad = copy.deepcopy(reg)
    _entry(bad, "P-S1")["phase"] = "9"
    errs = _errors(bad, tracked=_all_named_paths(bad))
    assert any("P-S1" in e and "phase=" in e for e in errs)
