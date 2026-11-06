"""T5 U4b's reading (`trackT_timaeus/t5i.py`): the seed-noise gate on planted replicate series."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

pytestmark = pytest.mark.deps  # t5h's imports (a1a, transformers)
from trackT_timaeus import t5h, t5i


def _series(shift_late=0.0, shift_rise=0.0):
    """`-32m`-shaped ρ_mid on the grid (as `test_trackT_t5h._dodeca_like`), L moved by ``shift_late``
    at 3738 and 4369 and the rise's middle (1253–2002) by ``shift_rise``."""
    pts = {0: 0.8, 40: 1.0, 917: 0.111, 3738: 0.763, 4369: 0.812, 5108: 0.85, 5970: 0.872}
    ks = sorted(pts)
    out = {}
    for s in t5h.CKPT_STEPS + t5h.CONT_STEPS:
        out[s] = float(np.interp(math.log1p(s), [math.log1p(k) for k in ks], [pts[k] for k in ks]))
    for s in t5h.L_STEPS:
        out[s] += shift_late
    for s in t5i.TIMING_STEPS:
        out[s] += shift_rise
    return out


def test_readable_at_small_seed_noise():
    r = t5i.replicates_read({"A0": _series(), "A0p": _series(0.03, 0.1), "A0pp": _series(-0.02)})
    assert r["tau"] == pytest.approx(0.05) and r["bar"] == pytest.approx(0.2)      # the floor binds
    assert r["bar_I"] == pytest.approx(0.2 * math.sqrt(2))
    assert r["R_A0"] == pytest.approx(0.7875 - 0.111) and r["readable"]
    assert "partly" in r["labels_in_use"]
    assert r["pairs"]["A0p-A0pp"] == pytest.approx(0.05)
    assert r["beside_timing"]["1253"]["spread"] == pytest.approx(0.1)


def test_unreadable_when_two_tau_reaches_half_r():
    # τ 0.17 → bar 0.34 ≥ R_A0 / 2 = 0.338: "lowers" and "partly" merge
    r = t5i.replicates_read({"A0": _series(), "A0p": _series(-0.17), "A0pp": _series()})
    assert r["bar"] == pytest.approx(0.34) and not r["readable"]
    assert r["labels_in_use"] == ["lowers", "does not", "raises"]
    # just below: τ 0.16 → bar 0.32 < 0.338
    assert t5i.replicates_read({"A0": _series(), "A0p": _series(-0.16), "A0pp": _series()})["readable"]


def test_refuses_without_the_replicates_or_their_late_steps():
    with pytest.raises(t5h.T5hRefused):
        t5i.replicates_read({"A0": _series(), "A0p": _series()})
    short = {s: v for s, v in _series().items() if s != 4369}
    with pytest.raises(t5h.T5hRefused):
        t5i.replicates_read({"A0": _series(), "A0p": short, "A0pp": _series()})


def _write(root, arm, series, gates):
    d = root / arm / "read"
    d.mkdir(parents=True)
    for s, v in series.items():
        (d / f"step{s}.json").write_text(json.dumps({"rho_mid": v, "v1_loss_mean": 4.0, "code": "x"}))
    for st, ok in gates.items():
        (root / arm / f"gate_{st}.json").write_text(json.dumps({"pass": ok, "series": {str(k): v for k, v in series.items()}}))


def test_the_command_refuses_on_an_unpassed_gate_and_reads_otherwise(tmp_path, monkeypatch):
    monkeypatch.setattr(t5h, "code_sha", lambda: "now")
    _write(tmp_path, "A0", _series(), {"late": True})
    _write(tmp_path, "A0p", _series(0.03), {"start": True})
    _write(tmp_path, "A0pp", _series(), {"start": False})
    a = type("A", (), {"root": str(tmp_path)})()
    with pytest.raises(t5h.T5hRefused, match="A0pp start"):
        t5i.replicates(a)
    good = {str(k): v for k, v in _series().items()}
    (tmp_path / "A0pp" / "gate_start.json").write_text(json.dumps({"pass": True, "series": good}))
    assert t5i.replicates(a) == 0
    rec = json.loads((tmp_path / "replicates.json").read_text())
    assert rec["pass"] and rec["read_codes"]["A0"] == ["x"] and rec["beside_v1_loss_4369"]["A0p"] == 4.0
    # a gate read on other records (a `t5h read --force` after it) refuses, as does one with no series
    # (CodeRabbit on #311)
    k = next(iter(good))
    (tmp_path / "A0pp" / "gate_start.json").write_text(json.dumps({"pass": True, "series": {**good, k: good[k] + 1e-6}}))
    with pytest.raises(t5h.T5hRefused, match="A0pp gate_start: not read on the current records"):
        t5i.replicates(a)
    (tmp_path / "A0pp" / "gate_start.json").write_text(json.dumps({"pass": True}))
    with pytest.raises(t5h.T5hRefused, match="no series"):
        t5i.replicates(a)


def test_step0_same_reads_the_largest_state_difference():
    r = t5i.step0_same([0.8, 0.9, 0.85], [0.8, 0.9 + 5e-7, 0.85])
    assert r["pass"] and r["max_abs_diff"] == pytest.approx(5e-7)
    assert not t5i.step0_same([0.8, 0.9], [0.8, 0.9 + 2e-6])["pass"]
    with pytest.raises(t5h.T5hRefused):
        t5i.step0_same([0.8, 0.9], [0.8])


def test_step0_command_refuses_without_the_base_record(tmp_path):
    d = tmp_path / "A1" / "read"
    d.mkdir(parents=True)
    (d / "step0.json").write_text(json.dumps({"rho": [0.8, 0.9]}))
    with pytest.raises(t5h.T5hRefused):
        t5i.main(["step0", "--arm", "A1", "--root", str(tmp_path)])
    b = tmp_path / "A0" / "read"
    b.mkdir(parents=True)
    (b / "step0.json").write_text(json.dumps({"rho": [0.8, 0.9]}))
    assert t5i.main(["step0", "--arm", "A1", "--root", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "A1" / "gate_step0_same.json").read_text())["pass"]


def _six(a1=0.0, a2=0.0, a3=0.0, reps=(0.03, -0.02)):
    s = {"A0": _series(), "A0p": _series(reps[0]), "A0pp": _series(reps[1]),
         "A1": _series(-a1), "A2": _series(-a2), "A3": _series(-a3)}
    rho0 = {a: [0.8] * 17 for a in ("A0", "A1", "A2", "A3")}
    return s, rho0


def test_arms_labels_and_pending_delay_check():
    # R_A0 = 0.7875 − 0.111 = 0.6765, half 0.338; bar 0.2 (floor)
    s, rho0 = _six(a1=0.40, a2=0.05, a3=0.25)
    r = t5i.arms_read(s, rho0)
    assert r["arms"]["A1"]["label"] == "lowers" and r["arms"]["A1"]["hyp_label"] == "pending the delay check"
    assert r["arms"]["A2"]["label"] == "does not" and r["arms"]["A2"]["hyp_label"] == "not supported"
    assert r["arms"]["A3"]["label"] == "partly" and r["arms"]["A3"]["continues_to_5970"]
    assert r["arms"]["A1"]["dL"] == pytest.approx(0.40)
    # I = ΔL_A3 − ΔL_A1 − ΔL_A2 = 0.25 − 0.40 − 0.05 = −0.20, inside bar_I 0.283
    assert r["interaction"]["I"] == pytest.approx(-0.20)
    assert r["interaction"]["label"] == "no interaction seen at this noise"
    assert r["continuations"] == ["A0", "A1", "A3", "A0pp"]
    assert not r["neither_alone_together"]


def test_arms_raises_and_neither_alone_together():
    s, rho0 = _six(a1=-0.25, a2=0.1, a3=0.5)
    r = t5i.arms_read(s, rho0)
    assert r["arms"]["A1"]["label"] == "raises" and r["arms"]["A1"]["hyp_label"] == "not supported"
    assert r["neither_alone_together"]
    assert r["interaction"]["label"] == "together lower it more than the sum"      # 0.5 + 0.25 − 0.1 = 0.65


def test_arms_refuse_unreadable_or_unequal_step0():
    s, rho0 = _six(reps=(-0.17, 0.0))
    with pytest.raises(t5h.T5hRefused, match="not readable"):
        t5i.arms_read(s, rho0)
    s, rho0 = _six()
    rho0["A3"] = [0.8] * 16 + [0.81]
    with pytest.raises(t5h.T5hRefused, match="step-0"):
        t5i.arms_read(s, rho0)
    del s["A3"]
    with pytest.raises(t5h.T5hRefused, match="missing"):
        t5i.arms_read(s, rho0)


def test_boot_dl_on_equal_prompts_is_the_point_value():
    pp = {a: {st: np.full((7, 17), v) for st in t5h.L_STEPS} for a, v in (("A0", 0.8), ("A1", 0.5))}
    b = t5i.boot_dl(pp, ["A1"])["A1"]
    assert b["p5"] == pytest.approx(0.3) and b["p95"] == pytest.approx(0.3)
    # unequal prompts: the interval is not a point, and it brackets the point value
    rng = np.random.default_rng(0)
    pp["A1"] = {st: 0.5 + 0.05 * rng.standard_normal((7, 17)) for st in t5h.L_STEPS}
    b = t5i.boot_dl(pp, ["A1"])["A1"]
    assert b["p5"] < b["p95"]
    assert t5i._mid(np.full((7, 17), 0.8)) == pytest.approx(0.8)
