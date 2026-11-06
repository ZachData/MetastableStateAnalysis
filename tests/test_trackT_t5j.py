"""T5 U4's beside stage (`trackT_timaeus/t5j.py`): the onsets' pure functions and the outcome table."""
from __future__ import annotations

import numpy as np
import pytest

from trackT_timaeus import t5j


def _attn(a0s, n=5):
    """(H, n, n) causal rows whose attention to position 0 is a0 for every query after the first."""
    H = len(a0s)
    A = np.zeros((H, n, n))
    for h, a0 in enumerate(a0s):
        A[h, 0, 0] = 1.0
        for i in range(1, n):
            A[h, i, 0] = a0
            A[h, i, 1:i + 1] = (1 - a0) / i
    return A


def test_sink_metric_counts_heads_over_eps():
    # mean over queries: (1 + 4 a0) / 5; a0 0.2 → 0.36 (> 0.3), a0 0.1 → 0.28 (≤ 0.3), a0 0.9 → 0.92
    blocks = [_attn([0.2, 0.1]), _attn([0.9, 0.0])]
    assert np.allclose(blocks[0].sum(-1), 1)
    assert t5j.sink_metric(blocks) == pytest.approx(2 / 4)
    assert t5j.sink_metric(blocks, eps=0.5) == pytest.approx(1 / 4)


def test_massive_one_and_criterion():
    rng = np.random.default_rng(0)
    h = rng.standard_normal((20, 64)) * 0.1
    h[0, 7] = 150.0
    m = t5j.massive_one(h)
    assert m["top"] == 150.0 and m["pos"] == 0 and m["dim"] == 7 and m["ratio"] > 1000
    flat = t5j.massive_one(rng.standard_normal((20, 64)))
    step = t5j.massive_step([[flat, m], [flat, m], [flat, flat]])     # 3 prompts, 2 blocks
    assert step["M_block"] == 2 and step["criterion"]                # block 2: median over prompts is m's
    assert step["pos0_share"] == pytest.approx(2 / 6)
    small = h.copy()
    small[0, 7] = 90.0                                               # ratio > 1000 but magnitude < 100
    s2 = t5j.massive_step([[t5j.massive_one(small)]])
    assert s2["M"] > 1000 and not s2["criterion"]


def test_half_rise():
    ser = {0: 1.0, 10: 1.5, 20: 2.6, 30: 2.4, 40: 3.0}               # threshold 2.0
    r = t5j.half_rise(ser)
    assert r["step"] == 20 and r["threshold"] == pytest.approx(2.0)
    back = {0: 1.0, 10: 2.5, 20: 1.9, 30: 2.4, 40: 3.0}              # crosses back at 20
    assert t5j.half_rise(back)["step"] == 30
    assert t5j.half_rise({0: 2.0, 10: 3.0, 20: 1.5})["reads"] == "no rise"
    rho = {0: 0.8, 10: 1.0, 20: 0.1, 30: 0.5, 40: 0.8}               # from the trough at 20: threshold 0.45
    assert t5j.half_rise(rho, 20)["step"] == 30


def test_held_from():
    assert t5j.held_from({0: False, 10: True, 20: False, 30: True, 40: True}) == 30
    assert t5j.held_from({0: True, 10: False}) is None
    assert t5j.held_from({0: True, 10: True}) == 0


def test_outcome_rows():
    none = {a: False for a in t5j.ARMS}
    assert t5j.outcome_row({**none, "A1": True}).startswith("A1 passes")
    assert t5j.outcome_row(none).startswith("neither")
    assert t5j.outcome_row({**none, "A0pp": True, "A1": True}).startswith("an A0 seed passes")
    assert t5j.outcome_row({**none, "A0": None, "A1": None}).startswith("neither")   # refused = not passing
