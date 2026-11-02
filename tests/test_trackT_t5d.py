"""T5 U2b, T2's planted gates in the trough (`trackT_timaeus/t5d.py`, design-T5.md "U2b"): pure parts."""
import json

import pytest

pytestmark = pytest.mark.deps  # c2_seeds' imports
from trackT_timaeus import t5d


def _arm(null, b, a):
    return {"null_rate": null, "B_share": b, "A_share": a, "pass": null <= 0.05 and b >= 0.4 and a <= 0.1}


def _rec(step, arms, a_u=0.04):
    return {"trained_step": step, "code": "x", "derived": {"b_dose": "0.25"}, "arms": [1e-4],
            "gate": {"arms": arms, "b_dose": "0.25", "pass": all(v["pass"] for v in arms.values())},
            "uncentred_at_1e-3": {"A_share": a_u, "null_rate": 0.06, "base_rate": 0.7},
            "checks": {f"step{step}": {"loss_mean": 5.0}}, "checks_pass": True}


def test_constants_as_frozen():
    assert t5d.TROUGH == (491, 671, 917) and t5d.UNC_PYTHIA == 0.79 and t5d.UNC_AS_9536 == 0.10


def test_trained_part_ignores_the_null_rate():
    assert t5d.trained_part(_arm(0.089, 0.98, 0.05))
    assert not t5d.trained_part(_arm(0.0, 0.98, 0.11))
    assert not t5d.trained_part(_arm(0.0, 0.39, 0.0))
    assert t5d.trained_part(_arm(0.0, 0.4, 0.1))           # the bars are inclusive, as c2_seeds


def test_label_three_ways():
    on = {"any_trained_part": True}
    off = {"any_trained_part": False}
    assert t5d.label({491: on, 671: on, 917: on}) == "passes in the trough"
    assert t5d.label({491: off, 671: off, 917: off}) == "fails in the trough"
    assert t5d.label({491: on, 671: off, 917: on}) == "mixed"


def test_step_reading_and_uncentred_bands():
    r = t5d.step_reading(_rec(491, {"0.0001": _arm(0.089, 0.98, 0.05), "0.00882": _arm(0.003, 0.3, 0.0)}, a_u=0.85))
    assert r["any_trained_part"] and not r["any_full_gate"]
    assert r["uncentred_beside"]["reads"] == "as on Pythia"
    assert t5d.step_reading(_rec(491, {"a": _arm(0, 1, 0)}, a_u=0.5))["uncentred_beside"]["reads"] == "between"
    assert t5d.step_reading(_rec(491, {"a": _arm(0, 1, 0)}, a_u=0.1))["uncentred_beside"]["reads"] == "as at 9536"


def test_known_answer_exact_and_refusing():
    ref = _rec(9536, {"0.0001": _arm(0.089, 0.98, 0.21)})
    assert t5d.known_answer(json.loads(json.dumps(ref)), ref)["pass"]
    moved = json.loads(json.dumps(ref))
    moved["gate"]["arms"]["0.0001"]["A_share"] += 1e-9
    assert not t5d.known_answer(moved, ref)["pass"]
    renamed = json.loads(json.dumps(ref))
    renamed["derived"]["b_dose"] = "0.5"
    assert t5d.known_answer(renamed, ref)["mismatches"]


def test_read_refuses_without_the_known_answer(tmp_path):
    ref_dir, pre = tmp_path / "ref", tmp_path / "pre"
    (ref_dir / "32m").mkdir(parents=True)
    (pre / "32m").mkdir(parents=True)
    ref = _rec(9536, {"0.0001": _arm(0.089, 0.98, 0.21)})
    (ref_dir / "32m" / "preflight_plants.json").write_text(json.dumps(ref))
    bad = json.loads(json.dumps(ref))
    bad["gate"]["arms"]["0.0001"]["B_share"] = 0.5
    (pre / "32m" / "preflight_plants.json").write_text(json.dumps(bad))
    assert t5d.main(["read", "--pre", str(pre), "--ref", str(ref_dir)]) == 1
    assert "refused" in json.loads((pre / "reading.json").read_text())
    # with it, every trough step is read and labelled
    (pre / "32m" / "preflight_plants.json").write_text(json.dumps(ref))
    for s, a in zip(t5d.TROUGH, (0.05, 0.3, 0.05)):
        (pre / f"32m-step{s}").mkdir()
        (pre / f"32m-step{s}" / "preflight_plants.json").write_text(json.dumps(_rec(s, {"0.0001": _arm(0.089, 0.98, a)})))
    assert t5d.main(["read", "--pre", str(pre), "--ref", str(ref_dir)]) == 0
    assert json.loads((pre / "reading.json").read_text())["label"] == "mixed"


def test_load_refuses_a_record_without_its_step_or_its_checks(tmp_path):
    (tmp_path / "32m").mkdir()
    rec = _rec(9536, {"a": _arm(0, 1, 0)})
    f = tmp_path / "32m" / "preflight_plants.json"
    for bad in ({k: v for k, v in rec.items() if k != "trained_step"}, {**rec, "checks_pass": False}):
        f.write_text(json.dumps(bad))
        with pytest.raises(t5d.U2bRefused):
            t5d.load(tmp_path, 9536)
    f.write_text(json.dumps(rec))
    assert t5d.load(tmp_path, 9536)["trained_step"] == 9536
