"""C6, V and the centred count on `pythia-70m-warmup01` (`trackC_writeup/c6_warmup.py`,
`c6_schedule.py`, design-C6.md): pure parts."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackA_force.a1a import STEPS
from trackC_writeup import c2_dose as cd
from trackC_writeup import c6_schedule as sch
from trackC_writeup import c6_warmup as w


def test_the_arms_are_70m_frozen_ones():
    assert w.ARMS == (9.61e-5, 1e-4, 3e-4, 4.78e-3) and w.W_RUN == 10 * w.W_BASE


def test_step_reads_as_unit_3b():
    rng = np.random.default_rng(0)
    for _ in range(500):
        v = np.sort(rng.integers(0, 20, len(STEPS))) if rng.random() < 0.7 else rng.integers(0, 20, len(STEPS))
        a, b = w.date_on(v, "step"), cd.date_reading(v)
        assert (a["step"], a["unreadable"]) == (b["step"], b["unreadable"])


@pytest.mark.parametrize("t,hyp,want", [(256, "cumulative", 256 * 10 ** 0.5), (1000, "cumulative", 1000 * 10 ** 0.5),
                                        (256, "instantaneous", 2560), (1000, "instantaneous", 10000)])
def test_maps_inside_the_warmups(t, hyp, want):
    assert w.m(hyp, t) == pytest.approx(want, rel=1e-6)


def test_instantaneous_turns_at_70m_peak():
    assert w.m("instantaneous", 2000) > w.m("instantaneous", 4000)
    assert w.band([2000], "instantaneous") == frozenset({16000})


@pytest.mark.parametrize("g,want", [(256, {"step": {256}, "cumulative": {512, 1000}, "instantaneous": {2000, 4000}}),
                                    (1000, {"step": {1000}, "cumulative": {2000, 4000},
                                            "instantaneous": {8000, 16000}})])
def test_bands_seed0_dates(g, want):
    assert {h: set(w.band([g], h)) for h in w.HYPS} == want


def test_windows_and_early():
    # the run's steps holding a 70m state in 1000–4000 (cumulative: 4000 holds 1000, 8000 holds 2000,
    # 16 000 holds 8000); instantaneous jumps the window, so its first later step
    assert w.window("cumulative") == (4000, 8000) and w.window("instantaneous") == (16000,)
    assert w.early("cumulative")[-1] == 512 and w.early("instantaneous")[-1] == 2000
    assert w.held(16000, "cumulative") == 8000 and w.held(16000, "instantaneous") == 143000


def test_plant_under_step_is_the_profile():
    v = list(range(len(STEPS)))
    assert w.plant(v, "step") == [float(x) for x in v]
    assert w.plant(v, "cumulative")[STEPS.index(4000)] == STEPS.index(1000)


def test_late_plants_change_only_instantaneous_past_its_peak():
    v = list(range(len(STEPS)))
    for late in w.LATE:
        assert w.plant(v, "step", late) == w.plant(v, "step")
        p = w.plant(v, "instantaneous", late)
        assert p[:STEPS.index(8000) + 1] == w.plant(v, "instantaneous")[:STEPS.index(8000) + 1]
    assert w.plant(v, "instantaneous")[STEPS.index(16000)] == STEPS.index(143000)
    assert w.plant(v, "instantaneous", "same_step")[STEPS.index(16000)] == STEPS.index(16000)


def _fake_run(out, codes, readable=True, gate_code="x"):
    import json
    (out / "gate.json").write_text(json.dumps({"readable": readable, "code": gate_code}))
    for i, s in enumerate(STEPS):
        for p in w.V1:
            f = w.rec_path(out, "run", s, p)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"model": "run", "code": codes[i % len(codes)]}))


def test_read_refuses_an_unpassed_gate_and_mixed_code(tmp_path):
    from types import SimpleNamespace
    _fake_run(tmp_path, ["x"], readable=False)
    with pytest.raises(w.C6Refused, match="gate"):
        w.read(SimpleNamespace(out=str(tmp_path)))
    _fake_run(tmp_path, ["x", "y"])
    with pytest.raises(w.C6Refused, match="one version"):
        w.read(SimpleNamespace(out=str(tmp_path)))


def test_ref70_is_frozen_per_arm():
    assert w.REF70 == {"u": (256, 512), "c": {"9.61e-05": (512, 1000), "0.0001": (512, 1000),
                                              "0.0003": (512, 1000), "0.00478": (1000, 2000)}}
    assert set(w.REF70["c"]) == {f"{t:.3g}" for t in w.ARMS}


def test_every_counted_arm_must_fit():
    ref = {"c": {"a": (1000,), "b": (1000,), "z": ()}}
    r = w.count_reading({"a": _v(1000), "b": _v(4000), "z": _v(4000)}, {"a": True, "b": True, "z": True}, "c", ref)
    assert r["step"]["status"] == "inconsistent"          # b's date is outside step's band
    assert set(r["step"]["per_arm"]) == {"a", "b"}          # z has no reference date: not counted
    r = w.count_reading({"a": _v(1000), "b": _v(1000)}, {"a": True, "b": True}, "c", ref)
    assert r["step"]["status"] == "consistent"


def test_calibration_recovers_planted_step_functions():
    beside = {f"s{i}": {"V_u": _v(du).tolist(), "V_c": {"x": _v(dc).tolist()}}
              for i, (du, dc) in enumerate([(256, 1000), (512, 1000), (256, 512)])}
    per = {n: w.seed_dates(b["V_u"], b["V_c"]) for n, b in beside.items()}
    cal = w.calibrate(beside, w.ref70(per))["summary"]
    assert all(cal[h]["truth_survives"] == 3 for h in w.HYPS)


def _v(rise_at, plateau=10.0):
    return np.array([0.0 if s < rise_at else plateau for s in STEPS])


def test_a_stretched_run_reads_its_schedule():
    ref = {"u": (256,), "c": (1000,)}
    for h, (du, dc) in {"step": (256, 1000), "cumulative": (1000, 4000), "instantaneous": (4000, 16000)}.items():
        for nr in ((), w.NOT_READ):                         # the rule as frozen; Blocked 127 (a)
            counts = {"u": w.count_reading({"x": _v(du)}, {"x": True}, "u", ref, nr),
                      "c": w.count_reading({"x": _v(dc)}, {"x": True}, "c", ref, nr)}
            assert counts["c"][h]["status"] == "consistent"
            assert counts["u"][h]["status"] == ("not read" if ("u", h) in nr else "consistent")
            assert w.joint(counts)["survivors"] == [h]


def test_blocked_127a_v_is_not_read_under_instantaneous():
    assert w.NOT_READ == frozenset({("u", "instantaneous")})
    ref = {"u": (256,), "c": (1000,)}
    u = w.count_reading({"x": _v(256)}, {"x": True}, "u", ref)
    assert u["instantaneous"] == {"dates": [], "band": {}, "status": "not read", "per_arm": {}}
    assert u["step"]["status"] == "consistent"
    assert w.count_reading({"x": _v(16000)}, {"x": True}, "c", ref)["instantaneous"]["status"] == "consistent"
    j = w.joint({"u": u, "c": w.count_reading({"x": _v(16000)}, {"x": True}, "c", ref)})
    assert j["survivors"] == ["instantaneous"] and j["unreadable"] == []
    cr = {h: {"status": "not read"} for h in w.HYPS}
    assert w.joint({"u": cr, "c": cr})["label"] == "unreadable"


def test_blocked_127a_calibration_does_not_depend_on_the_late_states():
    def prof(rise):                                         # V low early, rises, falls by 143 000 (8 of 10 seeds)
        v = np.maximum(_v(rise), 1.0)
        v[STEPS.index(143000)] = 1.0
        return v.tolist()
    beside = {f"s{i}": {"V_u": prof(du), "V_c": {"x": _v(dc).tolist()}}
              for i, (du, dc) in enumerate([(256, 1000), (512, 1000), (256, 512)])}
    ref = w.ref70({n: w.seed_dates(b["V_u"], b["V_c"]) for n, b in beside.items()})
    scored = [w.calibrate(beside, ref, late=lt, not_read=w.NOT_READ)["summary"] for lt in w.LATE]
    assert scored[0] == scored[1] == scored[2] and scored[0]["instantaneous"]["truth_alone"] == 3
    frozen = {lt: w.calibrate(beside, ref, late=lt)["summary"]["instantaneous"]["truth_survives"] for lt in w.LATE}
    assert frozen["literal"] == 0 and frozen["same_step"] == 3     # as frozen, the late states decide it


def test_ungated_arms_do_not_count_and_none_is_unreadable():
    ref = {"u": (256,), "c": (1000,)}
    r = w.count_reading({"a": _v(1000), "b": _v(16000)}, {"a": False, "b": True}, "c", ref)
    assert r["step"]["status"] == "inconsistent" and r["instantaneous"]["status"] == "consistent"
    assert r["step"]["per_arm"]["b"]["unreadable"]          # plateau 0 on step's window: against step


def test_a_run_on_step_excludes_the_doses_through_their_early_sets():
    ref = {"u": (256,), "c": (1000,)}
    counts = {"u": w.count_reading({"x": _v(256)}, {"x": True}, "u", ref),
              "c": w.count_reading({"x": _v(1000)}, {"x": True}, "c", ref)}
    assert counts["c"]["instantaneous"]["per_arm"]["x"]["unreadable"]   # high at 1000, inst's early set
    assert w.joint(counts)["label"] == "consistent with step"
    r0 = w.count_reading({"a": _v(1000)}, {"a": False}, "c", ref)
    assert all(r0[h]["status"] == "unreadable" for h in w.HYPS)


def test_joint_labels():
    def cr(st):
        return {h: {"status": s} for h, s in zip(w.HYPS, st)}
    assert w.joint({"u": cr(["consistent"] * 3), "c": cr(["consistent"] * 3)})["label"] == "not told apart"
    assert w.joint({"u": cr(["inconsistent"] * 3), "c": cr(["consistent"] * 3)})["label"] == "neither"
    j = w.joint({"u": cr(["unreadable", "consistent", "inconsistent"]),
                 "c": cr(["unreadable", "consistent", "consistent"])})
    assert j["survivors"] == ["cumulative"] and j["unreadable"] == ["step"]


def test_ref70_is_the_union_per_arm():
    per = {"a": {"u_date": 256, "c_arm_dates": {"x": 1000}}, "b": {"u_date": None, "c_arm_dates": {"x": 512, "y": 1000}},
           "c": {"u_date": 512, "c_arm_dates": {}}}
    assert w.ref70(per) == {"u": (256, 512), "c": {"x": (512, 1000), "y": (1000,)}}


def test_schedule_hypotheses_and_verdict():
    assert sch.lr(0, sch.PEAK, sch.W_BASE) == 0 and sch.predicted(0, 512, "H0") == pytest.approx(0.1)
    ro = [{"measured": 0.1, "predicted": {h: sch.predicted(0, 64, h) for h in sch.HYPS}}]
    assert sch.verdict(ro)["H0"] == "confirmed"
    ro = [{"measured": 0.2, "predicted": {h: sch.predicted(0, 64, h) for h in sch.HYPS}}]
    assert sch.verdict(ro)["H0"] == "not confirmed"
