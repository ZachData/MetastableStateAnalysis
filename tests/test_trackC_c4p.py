"""C4′ (a): induction's date on `pythia-70m-warmup01` by C6's rule (`trackC_writeup/c4p_read.py`,
design-C4p.md): pure parts and the registration guard (D3 (a))."""
import json

import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackA_force.a1a import STEPS
from trackC_writeup import c4p_read as p
from trackC_writeup import c6_warmup as w


def _rise(half: int, top: float = 0.9, low: float = 0.02):
    """A step profile: low before ``half``, top from it on."""
    return [top if s >= half else low for s in STEPS]


def test_reading_is_c6s_rule_on_one_arm():
    ref = (1000,)
    for truth in w.HYPS:
        v = w.plant(_rise(1000), truth)
        got = p.reading(v, ref)
        cnt = w.count_reading({"top": v}, {"top": True}, "ind", {"ind": ref}, not_read=())
        assert got["count"] == cnt and got["joint"] == w.joint({"ind": cnt})


def test_no_schedule_withheld():
    got = p.reading(w.plant(_rise(1000), "instantaneous"), (1000,))
    assert got["count"]["instantaneous"]["status"] != "not read"


@pytest.mark.parametrize("truth", ["step", "cumulative"])
def test_plants_read_their_schedule_at_seed0s_date(truth):
    assert p.reading(w.plant(_rise(1000), truth), (1000,))["joint"]["label"] == f"consistent with {truth}"


def test_ref_skips_unreadable_and_refuses_empty():
    flat = [0.0] * len(STEPS)
    assert p.ref_of({"a": _rise(1000), "b": _rise(512), "c": flat}) == (512, 1000)
    with pytest.raises(p.C4pRefused):
        p.ref_of({"c": flat})


def test_gate_passes_on_clean_seeds_and_is_leave_one_out():
    prof = {f"seed{k}": _rise(1000) for k in range(10)}
    cal = p.calibrate(prof, loo=True)
    assert cal["pass"] and all(cal["summary"][t]["exact"] == 10 for t in p.GATED)
    lone = dict(prof, seed0=_rise(2000))           # the only 2000: absent from its own LOO reference
    row = p.calibrate(lone, loo=True)["rows"]["seed0|step"]
    assert row["ref"] == [1000] and row["label"] != "consistent with step"


def test_gate_fails_when_the_reference_spans_512_and_1000():
    """Grid step 1000 is in step's band of 1000 and cumulative's band of 512, so half the plants
    read two schedules: the gate, not the rule, decides whether the real seeds separate."""
    prof = {f"seed{k}": _rise(1000 if k % 2 else 512) for k in range(10)}
    cal = p.calibrate(prof, loo=True)
    assert not cal["pass"] and cal["summary"]["step"]["exact"] == 5


def test_gate_fails_on_early_rising_seeds():
    prof = {f"seed{k}": _rise(64) for k in range(10)}      # early set ≥ half the plateau: unreadable
    with pytest.raises(p.C4pRefused):
        p.calibrate(prof)


def test_profile_is_the_top_head_and_refuses_gaps():
    ps = {str(s): np.full((6, 8), 0.1).tolist() for s in STEPS}
    ps["1000"][3][4] = 0.7
    assert p.profile(ps)[STEPS.index(1000)] == pytest.approx(0.7)
    del ps["2000"]
    with pytest.raises(p.C4pRefused):
        p.profile(ps)


def test_heads_at_and_transients():
    sc = np.zeros((6, 8)); sc[1, 2] = 0.5; sc[4, 7] = 0.49
    assert p.heads_at(sc) == [(1, 2)]
    mat = np.array([[0, 0, 0], [0, 7, 0], [7, 0, 7], [7, 0, 7]])
    assert p.transients(mat, (1, 2, 3, 4)) == {"T1": [2]}


def _gate(tmp_path, ok):
    (tmp_path / "gate.json").write_text(json.dumps({"pass": ok}))


def _reg(tmp_path, ids):
    f = tmp_path / "reg.json"
    f.write_text(json.dumps({"predictions": [{"id": i} for i in ids]}))
    return f


def test_run_guard(tmp_path):
    reg = _reg(tmp_path, ["P-SCH1"])
    with pytest.raises(p.C4pRefused):                       # no gate yet
        p.run_allowed(tmp_path, "declined", reg)
    _gate(tmp_path, False)
    assert "refused" in p.run_allowed(tmp_path, None, reg)  # failed gate: description
    _gate(tmp_path, True)
    with pytest.raises(p.C4pRefused):                       # passed: the user's decision needed
        p.run_allowed(tmp_path, None, reg)
    with pytest.raises(p.C4pRefused):                       # "registered" but not in the registry
        p.run_allowed(tmp_path, "registered", reg)
    assert "registered" in p.run_allowed(tmp_path, "registered", _reg(tmp_path, ["P-SCH1", "P-SCH2"]))
    assert "declined" in p.run_allowed(tmp_path, "declined", reg)
