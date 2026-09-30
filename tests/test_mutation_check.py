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
CTX = "abc123abc123"
M1, M2 = "core.evalues.x_f__mutmut_1", "core.evalues.x_f__mutmut_2"


def test_parse_results_reads_names_and_states():
    text = ("    core.evalues.x_f__mutmut_1: killed\n"
            "    core.evalues.xǁEǁg__mutmut_2: survived\n")
    assert mc.parse_results(text) == {
        "core.evalues.x_f__mutmut_1": "killed",
        "core.evalues.xǁEǁg__mutmut_2": "survived",
    }


def test_diff_lines_keeps_only_the_change():
    assert mc.diff_lines(SHOW) == DIFF


def test_split_name_finds_the_function_and_the_method():
    assert mc.split_name("core.evalues.x_average__mutmut_3") == ("core/evalues.py", ("average",))
    assert mc.split_name("core.evalues.xǁEProcessǁ__post_init____mutmut_2") == (
        "core/evalues.py", ("EProcess", "__post_init__"))


def _check(states, accepted, ctx=CTX):
    survived = [n for n, s in states.items() if s == "survived"]
    return mc.problems(states, {n: DIFF for n in survived}, accepted,
                       {n: ctx for n in survived})


def _ok(why="equivalent because ...", diff=DIFF, ctx=CTX):
    return {"diff": diff, "context": ctx, "why": why}


def test_a_reviewed_survivor_passes():
    assert _check({M1: "survived", M2: "killed"}, {M1: _ok()}) == []


def test_an_unreviewed_survivor_fails():
    (p,) = _check({M1: "survived"}, {})
    assert "not reviewed" in p and "+return 2.0" in p


def test_a_renumbered_survivor_fails():
    (p,) = _check({M1: "survived"}, {M1: _ok(diff=["-x", "+y"])})
    assert "not the reviewed one" in p


def test_a_survivor_whose_code_changed_fails():
    (p,) = _check({M1: "survived"}, {M1: _ok()}, ctx="changed00000")
    assert "changed since the reason was written" in p


@pytest.mark.parametrize("why", ["", "TODO: look at it"])
def test_an_entry_without_a_reason_fails(why):
    (p,) = _check({M1: "survived"}, {M1: _ok(why)})
    assert "without a reason" in p


def test_an_entry_awaiting_reconfirmation_fails():
    (p,) = _check({M1: "survived"}, {M1: _ok(mc.RECONFIRM + "old reason")})
    assert "RECONFIRM prefix" in p


def test_a_stale_entry_fails():
    (p,) = _check({M1: "killed"}, {M1: _ok()})
    assert "no longer survives" in p


@pytest.mark.parametrize("state", ["timeout", "suspicious", "not checked"])
def test_an_untested_mutant_fails(state):
    (p,) = _check({M1: state}, {})
    assert state in p


# -- rewrite: which reasons --write carries ------------------------------------

def test_rewrite_keeps_a_reason_whose_diff_and_context_are_unchanged():
    assert mc.rewrite({M1: DIFF}, {M1: CTX}, {M1: _ok("kept")})[M1]["why"] == "kept"


def test_rewrite_asks_to_reconfirm_when_the_context_changed():
    new = mc.rewrite({M1: DIFF}, {M1: "changed00000"}, {M1: _ok("kept")})
    assert new[M1] == {"diff": DIFF, "context": "changed00000",
                       "why": mc.RECONFIRM + "kept"}


def test_rewrite_carries_a_renumbered_change_within_its_function():
    # The same diff under a new number in the same function: the reason goes
    # along, behind RECONFIRM, because renumbering means the function changed.
    new = mc.rewrite({M2: DIFF}, {M2: "changed00000"}, {M1: _ok("kept")})
    assert new[M2]["why"] == mc.RECONFIRM + "kept"


def test_rewrite_does_not_carry_across_functions_or_diffs():
    other_fn = "core.evalues.x_g__mutmut_1"
    new = mc.rewrite({other_fn: DIFF, M2: ["-a", "+b"]},
                     {other_fn: CTX, M2: CTX}, {M1: _ok("kept")})
    assert new[other_fn]["why"] == new[M2]["why"] == mc.TODO


def test_rewrite_does_not_stack_prefixes():
    new = mc.rewrite({M1: DIFF}, {M1: "changed00000"}, {M1: _ok(mc.RECONFIRM + "kept")})
    assert new[M1]["why"] == mc.RECONFIRM + "kept"


# -- context_hash: what counts as the code a reason rests on ------------------

SRC = '''
def helper(x):
    return x > 0

def unrelated():
    return 1

def f(x):
    return helper(x)

class P:
    def check(self):
        return True

    def run(self):
        return self.check()
'''


def _changed(old, new):
    return SRC.replace(old, new)


def test_context_changes_with_the_function_and_its_callees():
    base = mc.context_hash(SRC, ("f",))
    assert mc.context_hash(_changed("return helper(x)", "return helper(-x)"), ("f",)) != base
    assert mc.context_hash(_changed("return x > 0", "return x >= 0"), ("f",)) != base


def test_context_ignores_code_the_function_does_not_name():
    assert (mc.context_hash(_changed("return 1", "return 2"), ("f",))
            == mc.context_hash(SRC, ("f",)))


def test_context_follows_self_calls_within_the_class():
    base = mc.context_hash(SRC, ("P", "run"))
    assert mc.context_hash(_changed("return True", "return False"), ("P", "run")) != base


def test_the_committed_list_is_complete_and_current():
    import json
    accepted = json.loads(mc.ACCEPTED.read_text(encoding="utf-8"))
    assert accepted, "tools/mutation_accepted.json is empty"
    contexts = mc.contexts_for(accepted)
    for name, entry in accepted.items():
        assert name.startswith("core.evalues."), name
        assert entry["diff"] and entry["why"].strip()
        assert not entry["why"].startswith(("TODO", "RECONFIRM")), name
        # Without mutmut: the code each reason was argued from is unchanged.
        assert entry["context"] == contexts[name], f"{name}: code changed since review"
