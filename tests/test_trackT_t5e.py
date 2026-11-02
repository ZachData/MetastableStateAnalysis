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
