"""
tests/test_mutation_check.py — tools/mutation_check.py, on synthetic mutmut
output (the real run needs mutmut and ~40 s; .github/workflows/mutation.yml).

The check must fail on each thing it exists for, not only pass on the
committed list (LESSONS.md lesson 2).
"""
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import mutation_check as mc  # noqa: E402

SHOW = """# core.evalues.x_f__mutmut_1: survived
--- core/evalues.py
+++ core/evalues.py
@@ -1,3 +1,3 @@
 def f():
-    return 1.0
+    return 2.0
"""
DIFF = ["-return 1.0", "+return 2.0"]


def test_parse_results_reads_names_and_states():
    text = ("    core.evalues.x_f__mutmut_1: killed\n"
            "    core.evalues.xǁEǁg__mutmut_2: survived\n")
    assert mc.parse_results(text) == {
        "core.evalues.x_f__mutmut_1": "killed",
        "core.evalues.xǁEǁg__mutmut_2": "survived",
    }


def test_diff_lines_keeps_only_the_change():
    assert mc.diff_lines(SHOW) == DIFF


def _check(states, accepted):
    diffs = {n: DIFF for n, s in states.items() if s == "survived"}
    return mc.problems(states, diffs, accepted)


def test_a_reviewed_survivor_passes():
    ok = {"m1": {"diff": DIFF, "why": "equivalent because ..."}}
    assert _check({"m1": "survived", "m2": "killed"}, ok) == []


def test_an_unreviewed_survivor_fails():
    (p,) = _check({"m1": "survived"}, {})
    assert "not reviewed" in p and "+return 2.0" in p


def test_a_renumbered_survivor_fails():
    (p,) = _check({"m1": "survived"}, {"m1": {"diff": ["-x", "+y"], "why": "old"}})
    assert "not the reviewed one" in p


@pytest.mark.parametrize("why", ["", "TODO: look at it"])
def test_an_entry_without_a_reason_fails(why):
    (p,) = _check({"m1": "survived"}, {"m1": {"diff": DIFF, "why": why}})
    assert "without a reason" in p


def test_a_stale_entry_fails():
    (p,) = _check({"m1": "killed"}, {"m1": {"diff": DIFF, "why": "x"}})
    assert "no longer survives" in p


@pytest.mark.parametrize("state", ["timeout", "suspicious", "not checked"])
def test_an_untested_mutant_fails(state):
    (p,) = _check({"m1": state}, {})
    assert state in p


def test_the_committed_list_has_a_reason_and_a_diff_for_every_entry():
    import json
    accepted = json.loads(mc.ACCEPTED.read_text(encoding="utf-8"))
    assert accepted, "tools/mutation_accepted.json is empty"
    for name, entry in accepted.items():
        assert name.startswith("core.evalues."), name
        assert entry["diff"] and entry["why"].strip()
        assert not entry["why"].startswith("TODO"), name
