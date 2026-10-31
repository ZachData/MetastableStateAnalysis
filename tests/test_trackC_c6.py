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
    assert w.window("cumulative") == (4000, 8000, 16000) and w.window("instantaneous") == (16000,)
    assert w.early("cumulative")[-1] == 256 and w.early("instantaneous")[-1] == 1000


def _v(rise_at, plateau=10.0):
    return np.array([0.0 if s < rise_at else plateau for s in STEPS])


def test_a_stretched_run_reads_its_schedule():
    ref = {"u": (256,), "c": (1000,)}
    for h, (du, dc) in {"step": (256, 1000), "cumulative": (1000, 4000), "instantaneous": (4000, 16000)}.items():
        counts = {"u": w.count_reading({"x": _v(du)}, {"x": True}, "u", ref),
                  "c": w.count_reading({"x": _v(dc)}, {"x": True}, "c", ref)}
        assert counts["u"][h]["status"] == counts["c"][h]["status"] == "consistent"
        assert w.joint(counts)["survivors"] == [h]


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


def test_ref70_is_the_union():
    per = {"a": {"u_date": 256, "c_dates": [1000]}, "b": {"u_date": None, "c_dates": [512, 1000]},
           "c": {"u_date": 512, "c_dates": []}}
    assert w.ref70(per) == {"u": (256, 512), "c": (512, 1000)}


def test_schedule_hypotheses_and_verdict():
    assert sch.lr(0, sch.PEAK, sch.W_BASE) == 0 and sch.predicted(0, 512, "H0") == pytest.approx(0.1)
    ro = [{"measured": 0.1, "predicted": {h: sch.predicted(0, 64, h) for h in sch.HYPS}}]
    assert sch.verdict(ro)["H0"] == "confirmed"
    ro = [{"measured": 0.2, "predicted": {h: sch.predicted(0, 64, h) for h in sch.HYPS}}]
    assert sch.verdict(ro)["H0"] == "not confirmed"
