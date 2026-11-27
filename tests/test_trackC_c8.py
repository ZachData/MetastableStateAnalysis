"""C8 U1 (`trackC_writeup/c8_rules.py`): rule R_i on synthetic logs (the design's G-synth, a code test)."""
import json

import numpy as np
import pytest

from trackC_writeup import c8_rules as r

pytestmark = pytest.mark.pure  # numpy only


def _log(n=1000, start=1, seed=0, sd=0.003, gsd=0.02):
    rng = np.random.default_rng(seed)
    s = range(start, start + n)
    loss = {k: 4.0 - 1e-4 * k + sd * rng.standard_normal() for k in s}
    gn = {k: 0.5 * float(np.exp(gsd * rng.standard_normal())) for k in s}
    return loss, gn


def test_windows_and_blocks_match_the_threshold_check():
    assert r.window(32, 48) == (16, 48)
    assert r.window(272, 280, 256) == (264, 280)
    assert r.window(280, 288, 256) == (264, 288)
    assert len(r.blocks(*r.window(32, 48), (9, 1000))) == 29
    assert len(r.blocks(*r.window(272, 280, 256), (265, 512))) == 13
    assert len(r.blocks(*r.window(44, 48), (9, 128))) < r.MIN_BLOCKS
    # B_K's 128-update log cannot hold B_P's 24-update window's background (c8_read's deviation)
    assert len(r.blocks(*r.window(280, 288, 256), (265, 384))) < r.MIN_BLOCKS
    assert len(r.blocks(*r.window(272, 280, 256), (265, 384))) >= r.MIN_BLOCKS


def test_constants_on_iid_noise():
    loss, gn = _log(4000, seed=1, sd=0.01, gsd=0.05)
    c = r.constants(loss, gn)
    assert abs(c["sigma_L"] - 0.01) < 0.001 and abs(c["c_g"] - 0.05) < 0.005
    assert c["delta_L"] == pytest.approx(6 * c["sigma_L"])
    assert c["kappa"] == pytest.approx(max(1.5, np.exp(6 * c["c_g"])))


def test_floors_bind_on_quiet_logs():
    loss, gn = _log(4000, seed=2, sd=1e-4, gsd=1e-3)
    c = r.constants(loss, gn)
    assert c["delta_L"] == 0.02 and c["kappa"] == 1.5 and all(c["floors_bind"].values())


def test_iid_log_is_quiet_and_planted_events_are_found():
    loss, gn = _log(seed=3)
    ev = r.events(loss, gn, 0.02, 1.5, 0, 1000)
    assert not ev["loss"] and not ev["grad"]
    loss[300] += 0.03
    gn[500] *= 2.0
    ev = r.events(loss, gn, 0.02, 1.5, 0, 1000)
    assert ev["loss"] == {300} and ev["grad"] == {500}


def test_a_fast_fall_is_not_a_rise():
    loss = {k: 10.0 - 0.1 * k for k in range(1, 200)}
    gn = {k: 1.0 for k in range(1, 200)}
    assert not r.events(loss, gn, 0.02, 1.5, 0, 199)["loss"]


def test_twin_removes_same_kind_same_update_only():
    ev = {"loss": {10, 20}, "grad": {30}}
    tw = {"loss": {10}, "grad": {20}}
    assert r.counted(ev, tw) == {"loss": {20}, "grad": {30}}


def test_branch_events_start_after_the_restart():
    loss, gn = _log(256, start=257, seed=4)
    loss[260] += 1.0
    ev = r.events(loss, gn, 0.02, 1.5, 256, 512, branch=256)
    assert not ev["loss"] and min(r.first_update(256), 265) == 265


def _cond(has, b):
    return {"has_event": has, "background": {"b": b, "refused": b is None}}


def test_label_table():
    five = {c: _cond(True, 0.3) for c in r.CONDITIONS}
    assert r.label(five, True)["label"] == "marked"           # 0.3^4 = 0.0081
    assert r.label(five, False)["label"] == "mixed"          # B_K or BKO not clean
    half = dict(five, Ft=_cond(False, 0.3))
    assert r.label(half, True)["label"] == "marked at ≥ half the peak"   # 0.3^3 = 0.027
    loud = {c: _cond(True, 0.6) for c in r.CONDITIONS}
    assert r.label(loud, True)["label"] == "mixed"           # Π = 0.13
    assert r.label({c: _cond(False, 0.1) for c in r.CONDITIONS}, True)["label"] == "silent"
    assert r.label(dict(five, BP=_cond(True, None)), True)["label"] == "refused"
    # F and F½ are one factor, at the larger b
    p = r.pi(dict(five, Fh=_cond(True, 0.5)), r.FACTORS_4)
    assert p == pytest.approx(0.5 * 0.3 ** 3)


def test_joint_across_tau():
    assert r.joint({"a": {"label": "silent"}, "b": {"label": "silent"}}) == "silent"
    assert r.joint({"a": {"label": "silent"}, "b": {"label": "mixed"}}) == "mixed across τ"


def test_shift_range():
    wins = {"F": (16, 48), "Ft": (736, 784), "BP": (264, 280)}
    spans = {"F": (9, 1000), "Ft": (9, 1000), "BP": (265, 384)}
    assert r.shift_range(wins, spans) == [0, 1, 2, 3, 4]   # F⅒'s window bounds it above, B_K's restart below


def test_condition_on_synthetic_log():
    loss, gn = _log(seed=5)
    tw = _log(seed=6)
    loss[40] += 0.05
    tw[0][40] += 0.05                                         # the twin has it too: not counted
    gn[44] *= 3.0
    c = r.condition(loss, gn, tw, 16, 48, 0.02, 1.5)
    assert c["events"] == {"loss": [], "grad": [44]} and c["twin_removed"]["loss"] == [40]
    assert c["has_event"] and c["background"]["n_blocks"] == 29 and c["background"]["b"] == 0.0
    assert c["leads"]["grad"] == [4]


def test_parse_log_keeps_the_last_repeated_row():
    rows = [{"segment": {}}, {"step": 1, "loss": 1.0, "grad_norm": 1.0}, {"step": 2, "loss": 2.0, "grad_norm": 1.0},
            {"segment": {}}, {"step": 2, "loss": 2.5, "grad_norm": 1.0}, {"step": 3, "loss": 3.0, "grad_norm": 1.0}]
    out = r.parse_log(json.dumps(x) for x in rows)
    assert out[2]["loss"] == 2.5 and sorted(out) == [1, 2, 3]
    with pytest.raises(r.C8RuleRefused):
        r.parse_log(json.dumps(x) for x in rows[:2] + rows[-1:])


def test_clip_label():
    gn = {s: (1.5 if s % 2 else 0.5) for s in range(1, 100)}
    assert r.clip_label(gn, 10, 20)["label"] == "clipped"
    assert r.clip_label({s: 0.5 for s in range(1, 50)}, 10, 20)["label"] == "unclipped"
