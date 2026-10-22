"""C2's render step (`trackC_writeup/numbers.py`, design-C2.md "Numbers and figures"): the tracked
draft is the render of its source and table, and the render refuses what it cannot source."""
import json

import pytest

pytestmark = pytest.mark.pure
from trackC_writeup import numbers as n


def keys(**kv):
    return {k: {"value": v, "record": "data:x.json"} for k, v in kv.items()}


def test_tracked_draft_is_its_render():
    table = json.loads(n.TABLE.read_text())
    assert n.DRAFT.read_text() == n.render(n.SOURCE.read_text(), table["keys"])


def test_every_slot_in_the_source_has_a_key():
    table = json.loads(n.TABLE.read_text())["keys"]
    missing = {m.group(1) for m in n.SLOT.finditer(n.SOURCE.read_text())} - set(table)
    assert not missing


def test_every_key_names_a_listed_record():
    table = json.loads(n.TABLE.read_text())
    assert {v["record"] for v in table["keys"].values()} <= set(table["records"]) == set(n.RECORDS.values())


def test_unknown_key_refuses():
    with pytest.raises(n.Refused):
        n.render("V is {{s0.nope}}", keys(**{"s0.V512": 56}))


def test_unknown_format_refuses():
    with pytest.raises(n.Refused):
        n.render("{{a|nope}}", keys(a=1))


def test_float_without_format_refuses():
    with pytest.raises(n.Refused):
        n.render("p = {{p}}", keys(p=9.9e-8))


def test_formats():
    k = keys(p=9.92290301275212e-08, lr=0.001, lr2=0.0003, big=143000, small=1430, xs=[3, 9], same=[512, 512],
             r=[0.40090375778320625, 0.48275656617024426], d={"E1": 0.98245}, n=7, e=1587.27)
    out = n.render("{{p|sci}} {{lr|sci}} {{lr2|sci}} {{big}} {{small}} {{xs|and}} {{same|uniq}} "
                   "{{r|range2}} {{d[E1]|f2}} {{n|Word}} {{e|f0}}", k)
    assert out.splitlines()[-1] == "9.9 × 10⁻⁸ 10⁻³ 3 × 10⁻⁴ 143 000 1430 3 and 9 512 0.40–0.48 0.98 Seven 1587"


def test_step_format_marks_censored_and_refused():
    k = keys(a={"step": 2000, "reading": "located", "censored": True},
             b={"step": None, "reading": "refused", "censored": False})
    assert n.render("{{a|step}}; {{b|step}}", k).splitlines()[-1] == "censored at 2000; refused"


def test_config_reader_takes_neox_lines():
    c = n.yml('{\n  # comment\n  "num-layers": 24,\n  "lr": 0.0003,\n  "warmup": 0.01,\n'
              '  "extra-save-iters": [0,1,2],\n  "pos-emb": "rotary",\n}')
    assert c == {"num-layers": 24, "lr": 0.0003, "warmup": 0.01, "extra-save-iters": [0, 1, 2], "pos-emb": "rotary"}


def test_constant_reads_annotated_assignment():
    assert n.constant("X: float = 1e-3\nY = 0.5\n", "X") == 1e-3
    assert n.constant("X: float = 1e-3\nY = 0.5\n", "Y") == 0.5


def test_changed_record_refuses_until_reread(tmp_path, monkeypatch):
    rec = tmp_path / "r.json"
    rec.write_text("{}")
    monkeypatch.setattr(n, "DATA", tmp_path)
    monkeypatch.setattr(n, "RECORDS", {"r": "data:r.json"})
    monkeypatch.setattr(n, "TABLE", tmp_path / "numbers.json")
    n.TABLE.write_text(json.dumps({"records": {"data:r.json": "0" * 64}, "keys": {}}))
    with pytest.raises(n.Refused, match="changed"):
        n.produce()
    with pytest.raises(n.Refused, match="absent"):
        monkeypatch.setattr(n, "RECORDS", {"r": "data:gone.json"})
        n.produce()


def test_draft_sentences_hold_on_the_tracked_table():
    assert n.check_claims(json.loads(n.TABLE.read_text())["keys"]) == []


def test_a_false_sentence_is_caught():
    table = json.loads(n.TABLE.read_text())["keys"]
    table["a3.rows"]["value"]["E8"]["step"] = 2000          # E8 after the tie: §5's order sentence breaks
    table["a5.mid_max"]["value"] = 1                         # "V is 0 at 8 to 64 on every seed" breaks
    assert set(n.check_claims(table)) == {"§5 E8 between the break and the tie, 512 from its peak",
                                          "§4 V is 0 at 8 to 64 on every seed"}


def test_changed_value_refuses_until_reread(tmp_path, monkeypatch):
    rec = tmp_path / "r.json"
    rec.write_text('{"x": 2}')
    monkeypatch.setattr(n, "DATA", tmp_path)
    monkeypatch.setattr(n, "RECORDS", {"r": "data:r.json", "a5_audit": "data:r.json", "a5_score": "data:r.json"})
    monkeypatch.setattr(n, "KEYS", {"x": ("r", lambda r: r["x"])})
    monkeypatch.setattr(n, "CLAIMS", [])
    monkeypatch.setattr(n, "TABLE", tmp_path / "numbers.json")
    monkeypatch.setattr(n, "sha256", lambda p: "h")
    rec_score = {"score_sha256": "h"}
    monkeypatch.setattr(n, "load", lambda rid: {**rec_score, "x": 2})
    n.TABLE.write_text(json.dumps({"records": {"data:r.json": "h"}, "keys": {"x": {"value": 1, "record": "data:r.json"}}}))
    with pytest.raises(n.Refused, match="changed value"):
        n.produce()
    assert n.produce(["x"])["keys"]["x"]["value"] == 2
