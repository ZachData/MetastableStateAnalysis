"""
`tools/run/p10_r9_lead.py` — R9's shared ``--lead`` / ``--reproduce`` flags: the inputs go
together, a mismatch refuses, and the header carries the lead's step-0 floor.
"""
import argparse

import pytest

from tools.run import p10_r9_lead as lead_args
from tools.run.p10_r1_ladder import LadderError

# Tier: stdlib only -- runs in `scripts/check.sh pure`.
pytestmark = pytest.mark.pure

ALSO = (("--reproduce-r2", "the R2 records that run read"),)


def _args(argv, also=ALSO):
    ap = argparse.ArgumentParser()
    lead_args.add_args(ap, "R2m", also=also)
    return ap.parse_args(argv)


def test_inputs_none_all_or_refuse():
    assert lead_args.reproduce_inputs(_args([]), ("--reproduce-r2",)) is None
    a = _args(["--reproduce", "m", "--reproduce-r2", "r2", "--reproduce-labels", "lab"])
    assert [str(x) for x in lead_args.reproduce_inputs(a, ("--reproduce-r2",))] == ["m", "r2", "lab"]
    for argv in (["--reproduce", "m"], ["--reproduce", "m", "--reproduce-labels", "lab"], ["--reproduce-r2", "r2"]):
        with pytest.raises(SystemExit, match="--reproduce, --reproduce-r2 and --reproduce-labels go together"):
            lead_args.reproduce_inputs(_args(argv), ("--reproduce-r2",))


def test_lead_choices_are_the_ladders():
    assert _args(["--lead", "c3x"]).lead == "c3x" and _args([]).lead == "c3"
    with pytest.raises(SystemExit):
        _args(["--lead", "c2"])


def test_check_reproduces_loads_the_inputs_in_order_and_refuses_on_a_difference(capsys):
    a = _args(["--reproduce", "m", "--reproduce-r2", "r2", "--reproduce-labels", "lab"])
    seen = []

    def load(*xs):
        seen.append([str(x) for x in xs])
        return {}
    assert lead_args.check_reproduces(a, {}, load, lambda d, o: [], "x", ("--reproduce-r2",)) == "m"
    assert seen == [["m", "r2", "lab"]] and "every c0–c3 record equal (x)" in capsys.readouterr().out
    with pytest.raises(LadderError, match=r"differ from m: \['f1_c0'\]"):
        lead_args.check_reproduces(a, {}, load, lambda d, o: ["f1_c0"], "x", ("--reproduce-r2",))
    assert lead_args.check_reproduces(_args([]), {}, load, lambda d, o: ["no"], "x") is None


def test_header_carries_the_leads_floor():
    summ = {"step0": {"columns": {"c3x": {"groups": 31, "readable": 8, "records": 161}}}}
    h = lead_args.header("lab", "abc", "c3x", None, {0: "c2"}, summ)
    assert h["floor"] == {"c3x_group_layer_records_step0": 31, "c3x_readable_step0": [8, 161]}
    assert "floor 31 c3x records at step 0 (8 of 161 readable)" in lead_args.floor_line({0: "c2"}, summ, "c3x")


def test_one_ladder_error_class():
    """`/challenge-pr` on #171, finding 3: the helper owns LadderError, every ladder re-exports it."""
    from tools.run import p10_r1_ladder, p10_r2_ladder, p10_r2m_ladder, p10_r3_ladder
    assert {m.LadderError for m in (p10_r1_ladder, p10_r2_ladder, p10_r2m_ladder, p10_r3_ladder)} == {LadderError}
    assert LadderError is lead_args.LadderError
