"""C2 unit 3b, the LR-dose arm (`trackC_writeup/c2_dose.py`, design-C2.md Blocked 92 (iii), 98 (c), 99 (a)): pure parts."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackA_force.a1a import STEPS
from trackC_writeup import c2_dose as d

LR70, LR160 = d.SIZES["70m"][1], d.SIZES["160m"][1]


def test_the_reference_is_410m_seeds_dates():
    assert d.REF == {"u": (256, 512), "c": (512, 1000)} and d.REF_LR == 3e-4


@pytest.mark.parametrize("kind,lr,hyp,want", [
    ("u", LR70, "step", {256, 512}), ("u", LR70, "cumulative", {128, 256, 512}),
    ("u", LR70, "instantaneous", {64, 128, 256}),
    ("u", LR160, "cumulative", {128, 256, 512}), ("u", LR160, "instantaneous", {128, 256}),
    ("c", LR70, "step", {512, 1000}), ("c", LR70, "cumulative", {256, 512, 1000}),
    ("c", LR70, "instantaneous", {128, 256, 512}),
    ("c", LR160, "cumulative", {256, 512, 1000}), ("c", LR160, "instantaneous", {256, 512}),
])
def test_predicted_grid_steps(kind, lr, hyp, want):
    assert d.predicted(d.REF[kind], d.factor(hyp, lr)) == frozenset(want)


def test_predicted_matches_a_fine_scan():
    for f in (1.0, 0.3, 0.5477, 0.7071, 0.1):
        fine = {min(s for s in STEPS if s >= t) for g in (256, 512)
                for t in np.linspace(f * d.prev(g), f * g, 4001)[1:]}
        assert d.predicted((256, 512), f) == frozenset(fine)


def test_consistent_reads_any_of_the_dates():
    assert d.consistent([512], "u", LR70) == {"step", "cumulative"}
    assert d.consistent([256], "u", LR70) == set(d.HYPS)
    assert d.consistent([64], "u", LR70) == {"instantaneous"}
    assert d.consistent([1000], "c", LR70) == {"step", "cumulative"}
    assert d.consistent([128, 1000], "c", LR70) == set(d.HYPS)
    assert d.consistent([2000], "c", LR70) == set()


def _v(at: int, plateau: float = 40.0) -> np.ndarray:
    return np.array([plateau if s >= at else 0.0 for s in STEPS])


def test_date_reading_and_the_unreadable_check():
    assert d.date_reading(_v(512))["step"] == 512
    early = _v(512)
    early[STEPS.index(64)] = 20.0                     # ≥ half the plateau at a step ≤ 128
    r = d.date_reading(early)
    assert r["unreadable"] and r["step"] is None
    assert d.date_reading(np.zeros(len(STEPS)))["unreadable"]


def test_size_reading_uses_only_gated_readable_arms():
    vcs = {"a": _v(1000), "b": _v(512), "c": np.zeros(len(STEPS))}
    r = d.size_reading(_v(512), vcs, {"a": True, "b": False, "c": True}, LR70)
    assert r["uncentred"]["step"] == 512 and r["uncentred"]["consistent"] == ["cumulative", "step"]
    assert r["centred"]["dates"] == [1000] and r["centred"]["agree"]
    assert r["centred"]["arms"]["c"]["unreadable"]
    none = d.size_reading(_v(512), vcs, {"a": False, "b": False, "c": True}, LR70)
    assert none["centred"]["unreadable"] and none["centred"]["consistent"] is None


def test_joint_intersects_readable_dates():
    def sr(u, c):
        return {"uncentred": {"consistent": u}, "centred": {"consistent": c}}
    allh = list(d.HYPS)
    assert d.joint({"70m": sr(allh, allh)})["label"] == "not told apart"
    j = d.joint({"70m": sr(["cumulative", "step"], allh), "160m": sr(None, ["cumulative", "step"])})
    assert j["survivors"] == ["step", "cumulative"] and j["label"] == "consistent with step and cumulative"
    assert j["unreadable"] == ["160m uncentred"] and j["readable_dates"] == 3
    assert d.joint({"70m": sr(["step"], ["instantaneous"])})["label"] == "neither"
    assert d.joint({"70m": sr(None, None)})["label"] == "unreadable"


def test_run_and_read_refuse_until_frozen(monkeypatch):
    monkeypatch.setitem(d.TAU_ARMS, "70m", None)
    with pytest.raises(d.DoseRefused):
        d._frozen("70m")


def test_b_dose_is_the_smallest_the_uncentred_count_sees():
    base = [0.0] * 6 + [1.0] * 4                    # rate 0.4 at 1e-3
    planted = {"0": base, "0.25": [1.0] * 6 + [1.0] * 4, "0.1": [0.0] * 4 + [1.0] * 6}
    assert d.b_dose(base, planted) == "0.25"        # 0.1 turns 2 of 6 (0.33 < 0.4)
    planted["0.1"] = [0.0] * 3 + [1.0] * 7          # turns 3 of 6 (0.5)
    assert d.b_dose(base, planted) == "0.1"
    with pytest.raises(d.DoseRefused):
        d.b_dose(base, {"0": base, "0.25": base})


def test_the_unit3s_dose_is_kept_and_extended():
    from trackC_writeup import c2_seeds as cs
    assert d.DOSES_B[:-1] == cs.DOSES_B and d.DOSES_B[-1] == 0.5 and d.B_SEEN_U == cs.B_SEEN
