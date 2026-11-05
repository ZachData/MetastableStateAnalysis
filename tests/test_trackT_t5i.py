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
