"""
tests/test_session_cost.py -- tools/session_cost.py on a synthetic transcript.

The case that matters: one API call is several assistant records sharing a
`message.id`; counting records instead of ids would triple the call count.
"""
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import session_cost  # noqa: E402


def _asst(mid, blocks, inp, read, create, out):
    return {"type": "assistant", "message": {
        "id": mid, "role": "assistant", "content": blocks,
        "usage": {"input_tokens": inp, "cache_read_input_tokens": read,
                  "cache_creation_input_tokens": create, "output_tokens": out}}}


def _result(tid, content):
    return {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tid, "content": content}]}}


def _lines():
    recs = [
        {"type": "user", "message": {"role": "user", "content": "go"}},
        {"type": "mode", "mode": "normal"},
        # call m1: three records, one usage
        _asst("m1", [{"type": "thinking", "thinking": ""}], 10, 1000, 200, 50),
        _asst("m1", [{"type": "text", "text": "hi"}], 10, 1000, 200, 50),
        _asst("m1", [{"type": "tool_use", "id": "t1", "name": "Read",
                      "input": {"file_path": "/a.md", "offset": 5, "limit": 10}}],
              10, 1000, 200, 50),
        _result("t1", "x" * 300),
        # call m2: two tool uses
        _asst("m2", [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "ls"}},
                     {"type": "tool_use", "id": "t3", "name": "Read",
                      "input": {"file_path": "/a.md"}}], 5, 2000, 100, 70),
        _result("t2", [{"type": "text", "text": "y" * 40}, {"type": "text", "text": "z" * 60}]),
        _result("t3", "w" * 500),
        # call m3: the peak
        _asst("m3", [{"type": "text", "text": "done"}], 1, 5000, 0, 20),
    ]
    return [json.dumps(r) for r in recs] + ["", "{not json"]


def test_calls_are_distinct_message_ids():
    r = session_cost.analyse(_lines())
    assert r["calls"] == 3


def test_context_total_peak_output():
    r = session_cost.analyse(_lines())
    assert r["context_total"] == 1210 + 2105 + 5001
    assert r["context_peak"] == 5001
    assert r["output_tokens"] == 50 + 70 + 20


def test_tool_results_attributed_and_ranked():
    r = session_cost.analyse(_lines())
    assert r["per_tool"] == {"Read": {"calls": 2, "chars": 800},
                             "Bash": {"calls": 1, "chars": 100}}
    assert r["tool_result_chars"] == 900
    assert [x[2] for x in r["largest_results"]] == ["t3", "t1", "t2"]


def test_reads_and_bad_lines():
    r = session_cost.analyse(_lines())
    assert r["reads"] == [("/a.md", 5, 10), ("/a.md", None, None)]
    assert r["bad_lines"] == 1
    assert "2x (1 ranged)  /a.md" in session_cost.report(r)


def test_row_and_cli(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(_lines()))
    session_cost.main([str(p), "--row", "unit", "--pr", "#1", "--date", "2026-01-01"])
    assert capsys.readouterr().out.strip() == "| 2026-01-01 | unit | 3 | 5k | 8k | 225 | #1 |"


def test_refuses_file_without_usage(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text(json.dumps({"type": "user", "message": {"content": "x"}}) + "\n")
    with pytest.raises(SystemExit):
        session_cost.main([str(p)])
