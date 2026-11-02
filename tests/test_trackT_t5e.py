"""T5 U3's first read (`trackT_timaeus/t5e.py`, design-T5.md "U3 revised toward the late rise"): pure parts."""
import pytest

pytestmark = pytest.mark.deps  # c2_seeds' imports, through t5d
from trackT_timaeus import t5e


def test_steps_as_frozen():
    assert t5e.INNER == (1072, 1253, 1465, 1713, 2002, 2340, 2735, 3198, 3738, 4369, 5108, 5970, 6979, 8158)
    assert t5e.RISE[0] == 917 and t5e.RISE[-1] == 9536 and len(t5e.RISE) == 16
    assert (t5e.STOP_TAU, t5e.STOP_DOSE) == ("0.003", "0.5")
    assert t5e.TAUS == (1e-3, 2e-3, 3e-3, 5e-3, 8.82e-3, 1.2e-2) and t5e.DOSES == ("0.25", "0.5")


def test_stop_monotone():
    r = t5e.stop([(917, True), (1072, True), (1253, False), (9536, False)])
    assert r == {"last_pass": 1072, "stop": 1253, "monotone": True, "fails_before_stop": []}


def test_stop_crosses_back():
    r = t5e.stop([(917, True), (1072, False), (1253, True), (1465, False), (9536, False)])
    assert r["stop"] == 1465 and r["last_pass"] == 1253 and not r["monotone"]
    assert r["fails_before_stop"] == [1072]


def test_stop_never_stops():
    r = t5e.stop([(917, True), (1072, False), (9536, True)])
    assert r["stop"] is None and r["fails_before_stop"] == [1072] and not r["monotone"]


def test_one_sided_reading():
    assert t5e.one_sided(1072) == "stops before ρ rises"
    assert t5e.one_sided(1253) == "at the rise's start"
    assert t5e.one_sided(1465) == t5e.one_sided(9536) == "with the rise"
    assert t5e.one_sided(None) == "does not stop"


def test_lr_phase():
    assert t5e.lr_phase(917) == "warmup, 0.92"
    assert t5e.lr_phase(1072) == "decay" and t5e.lr_phase(9536) == "end of decay"


def _per(i, ii, iii):
    return {"centred": {"any_trained_part": i, "uncentred_beside": {"A_share": iii}}, "uncentred_A_share": iii,
            "fixed_tau": {"0.5": {"0.003": {"A_share": ii}}}}


def test_series_bars_inclusive_and_stated():
    per = {917: _per(True, 0.10, 0.79), 1072: _per(False, 0.11, 0.78), 9536: _per(False, 0.4, 0.04)}
    st = t5e.stops(per)
    assert st["ii_fixed_tau"]["stop"] == 1072 and st["iii_uncentred"]["stop"] == 1072
    assert st["i_centred_any_P"]["reads"] == "stops before ρ rises"
    assert all(t5e.stated_before_output(per).values())
    per[917] = _per(False, 0.2, 0.5)
    assert not any(t5e.stated_before_output(per).values())


def test_refused_read_both_ways():
    per = {917: _per(True, 0.0, 0.9), 1072: _per(True, 0.0, 0.9), 1253: _per(False, 0.0, 0.9),
           2340: {**_per(None, 0.0, 0.9), "centred": {"refused": "DoseRefused"}},
           3198: _per(True, 0.0, 0.9), 9536: _per(False, 0.4, 0.04)}
    for v in per.values():
        v["uncentred_A_share"] = v["centred"].get("uncentred_beside", {"A_share": 0.9})["A_share"]
    per[9536]["uncentred_A_share"] = 0.04
    i = t5e.stops(per)["i_centred_any_P"]
    assert i["refused"] == [2340] and i["stop"] == 9536 and i["fails_before_stop"] == [1253, 2340]
    assert i["refused_left_out"]["fails_before_stop"] == [1253] and i["refused_agree"]


# read() end to end on record files (`/challenge-pr` on #302, finding 4): the refusal path, the
# units-vs-records check and the log's step check; the numeric parts are stubbed from the units.
import json as _json


def _plants(step, a_u, bm=3e-3):
    arm = {"null_rate": 0.01, "B_share": 0.8, "A_share": 0.0, "pass": True}
    return {"trained_step": step, "code": "c", "checks_pass": True, "derived": {"b_dose": "0.5", "b_matched": bm},
            "arms": [bm], "gate": {"arms": {f"{bm:g}": arm}, "b_dose": "0.5", "pass": True},
            "uncentred_at_1e-3": {"A_share": a_u, "null_rate": 0.06, "base_rate": 0.2},
            "checks": {f"step{step}": {"loss_mean": 4.0}}}


def _tree(tmp_path, log_2340="32m step2340 done\nDoseRefused: no plant-B dose\n", unc_917=0.9):
    pre, ref, ref917 = tmp_path / "pre", tmp_path / "ref", tmp_path / "u2b"
    for d, s, a_u, dname in ((pre, 9536, 0.04, "32m"), (pre, 917, 0.9, "32m-step917"),
                             (ref, 9536, 0.04, "32m"), (ref917, 917, 0.9, "32m-step917")):
        (d / dname).mkdir(parents=True, exist_ok=True)
        (d / dname / "preflight_plants.json").write_text(_json.dumps(_plants(s, a_u)))
    for s, unc in ((9536, 0.04), (917, unc_917), (2340, 0.5)):
        dname = "32m" if s == 9536 else f"32m-step{s}"
        (pre / dname).mkdir(parents=True, exist_ok=True)
        (pre / dname / "preflight_units.json").write_text(_json.dumps({"unc": unc, "a": 0.5 if s == 9536 else 0.0}))
    (pre / "log_2340.txt").write_text(log_2340)
    u2 = tmp_path / "u2.json"
    ser = {"step": [917, 2340, 9536], "rho_mid": [0.1, 0.5, 0.9], "rho_mid_no0": [0.1, 0.5, 0.9], "n0_mid": [2, 9, 24]}
    u2.write_text(_json.dumps({"reading": {"dodecahedron-32m": {"series": ser}}}))
    return ["read", "--pre", str(pre), "--ref", str(ref), "--ref-917", str(ref917), "--u2", str(u2)], pre


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(t5e, "RISE", (917, 2340, 9536))
    monkeypatch.setattr(t5e, "fixed_tau", lambda u: {"0.5": {"0.003": {"A_share": u["a"]}}})
    monkeypatch.setattr(t5e, "uncentred_a", lambda u: u["unc"])
    monkeypatch.setattr(t5e, "wide_b_matched", lambda u: {"refused": "DoseRefused"} if u["unc"] == 0.5
                        else {"tau": 3e-3, "trained_part": True})


def test_read_refused_step(tmp_path, stubbed):
    argv, pre = _tree(tmp_path)
    assert t5e.main(argv) == 0
    r = _json.loads((pre / "reading.json").read_text())
    assert r["per_step"]["2340"]["centred"]["refused"].startswith("DoseRefused")
    assert r["where_each_count_stops"]["i_centred_any_P"]["refused"] == [2340]
    assert r["where_each_count_stops"]["iii_uncentred"]["stop"] == 2340
    assert r["gate"]["uncentred_from_units"]["pass"] and r["gate"]["wide_b_matched_reproduces"]["pass"]


def test_read_refuses_a_log_without_its_step(tmp_path, stubbed):
    argv, _ = _tree(tmp_path, log_2340="32m step2002 done\nDoseRefused: x\n")
    with pytest.raises(t5e.U3Refused, match="does not show step 2340"):
        t5e.main(argv)


def test_read_refuses_units_unlike_the_records(tmp_path, stubbed):
    argv, _ = _tree(tmp_path, unc_917=0.8)
    with pytest.raises(t5e.U3Refused, match=r"\(iii\) from the units"):
        t5e.main(argv)
