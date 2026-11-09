"""C4 (`trackC_writeup/c4_train.py`, `c4_read.py`): the schedules, the rows, and the frozen rule
(`design-C4.md`), on planted inputs. No torch, no data."""
import math

import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports

from trackC_writeup import c4_read as r
from trackC_writeup import c4_train as ct
from trackC_writeup import c6_schedule


def test_a0_is_neox_annealing_lr_and_rate_0_at_iteration_0():
    assert ct.lr_pythia(0) == 0.0
    assert ct.lr_pythia(ct.W) == pytest.approx(ct.PEAK)
    assert ct.lr_pythia(ct.DECAY) == pytest.approx(ct.MIN_LR)
    for i in (1, 7, 700, 1430, 1431, 4000, 100000):
        assert ct.lr_pythia(i) == pytest.approx(c6_schedule.lr(i, ct.PEAK, ct.W))


def test_flat_is_the_peak_to_w_then_a0s_cosine():
    assert all(ct.lr_flat(i) == ct.PEAK for i in (0, 1, 500, ct.W))
    assert all(ct.lr_flat(i) == ct.lr_pythia(i) for i in (ct.W + 1, 3999, 143000))
    assert ct.summed("F", 100) == pytest.approx(100 * ct.PEAK)
    assert ct.summed("A0", 100) == pytest.approx(ct.PEAK * 99 * 100 / (2 * ct.W))


def test_weight_decay_groups_follow_neox():
    class P:
        def __init__(self, nd):
            self.ndim = nd
    assert ct.decays("gpt_neox.layers.0.attention.query_key_value.weight", P(2))
    assert ct.decays("embed_out.weight", P(2)) and ct.decays("gpt_neox.embed_in.weight", P(2))
    assert not ct.decays("gpt_neox.layers.0.attention.query_key_value.bias", P(1))
    assert not ct.decays("gpt_neox.layers.0.input_layernorm.weight", P(1))
    assert ct.WD == 0.1                                     # the config's, not the design's 0.01


def test_rows_must_be_2049_long_and_contiguous():
    n = 10
    sizes, ptrs = np.full(n, ct.ROW, np.int32), np.arange(n, dtype=np.int64) * 2 * ct.ROW
    assert ct.check_rows(sizes, ptrs, n * 2 * ct.ROW)["contiguous"]
    with pytest.raises(ct.C4Refused):
        ct.check_rows(sizes, ptrs, n * 2 * ct.ROW - 1)       # past the shard
    bad = ptrs.copy(); bad[3] += 2
    with pytest.raises(ct.C4Refused):
        ct.check_rows(sizes, bad, n * 2 * ct.ROW)
    short = sizes.copy(); short[5] = 2048
    with pytest.raises(ct.C4Refused):
        ct.check_rows(short, ptrs, n * 2 * ct.ROW)


def test_batches_are_1024_rows_per_iteration():
    rows = np.arange(5000 * ct.BATCH * 1, dtype=np.int64).reshape(-1, 1)[:4000 * ct.BATCH]
    assert ct.batch_of(rows, 0)[0, 0] == 0 and ct.batch_of(rows, 1)[0, 0] == ct.BATCH
    with pytest.raises(ct.C4Refused):
        ct.batch_of(rows, ct.HORIZON)
    assert ct.HELD_OUT[0] == ct.HORIZON * ct.BATCH          # never trained on to 4000


def test_divergence_needs_50_consecutive_losses_over_twice_the_median():
    base = [3.0] * 100
    assert ct.diverged(base) is None
    assert ct.diverged(base + [7.0] * 49 + [3.0]) is None
    assert ct.diverged(base + [7.0] * 50) == 149
    assert ct.diverged(base + [float("nan")] * 50) == 149


def _series(date, grid=r.GRID, plateau=10.0):
    return np.array([plateau if s >= date else 0.0 for s in grid])


def test_date_is_the_half_rise_and_its_refusals():
    d = r.date_on(_series(256))
    assert (d["step"], d["status"], d["early_92i_flag"]) == (256, "dated", False)
    early = r.date_on(_series(24))
    assert (early["step"], early["status"], early["early_92i_flag"]) == (24, "dated", True)   # beside only
    assert r.date_on(np.zeros(len(r.GRID)))["status"] == "not formed by 4000"
    assert r.date_on(_series(0))["status"].startswith("unreadable")
    with pytest.raises(r.C4ReadRefused):
        r.date_on([1.0, 2.0])


def test_rho_is_the_grids_neighbour_ratio_at_least_1_25():
    assert r.rho_at(256) == pytest.approx(256 / 192)
    assert r.rho_at(1000) == pytest.approx(1000 / 768)
    assert r.rho_at(1500) == pytest.approx(2000 / 1500)
    assert r.rho_at(4) == pytest.approx(2.0)


def test_the_bands_are_disjoint_at_every_dated_grid_step_through_1000():
    """design: disjoint while t < 2W/ρ³ (`c4_flat_maps.py`); checked at this grid's own ρ."""
    for t in [s for s in r.GRID if 0 < s <= 1000]:
        rho = r.rho_at(t)
        assert t < 2 * ct.W / rho ** 3, t
        (s_lo, s_hi), (c_lo, c_hi) = r.band("step", t, rho), r.band("cumulative", t, rho)
        assert c_hi < s_lo, t


def test_labels_on_planted_flat_dates():
    a0 = r.date_on(_series(256))
    assert r.label(r.date_on(_series(256)), a0)["label"] == "consistent with step"
    assert r.label(r.date_on(_series(24)), a0)["label"] == "consistent with cumulative"   # 256²/2860 ≈ 23
    assert r.label(r.date_on(_series(96)), a0)["label"] == "neither"
    c0 = r.date_on(_series(1000))
    assert r.label(r.date_on(_series(384)), c0)["label"] == "consistent with cumulative"  # ≈ 350
    assert r.label(r.date_on(_series(1250)), c0)["label"] == "consistent with step"
    assert r.label(r.date_on(np.zeros(len(r.GRID))), c0)["label"] == "not formed by 4000"
    assert r.label(c0, r.date_on(np.zeros(len(r.GRID))))["label"].startswith("A0 ")
    with pytest.raises(r.C4ReadRefused):
        r.m("cumulative", 2000)


def test_centred_label_needs_every_gated_dated_arm():
    a0 = r.date_on(_series(1000))
    cum, stp = r.label(r.date_on(_series(384)), a0), r.label(r.date_on(_series(1000)), a0)
    assert r.centred_label({"a": cum, "b": cum}, {"a": True, "b": True})["label"] == "consistent with cumulative"
    assert r.centred_label({"a": cum, "b": stp}, {"a": True, "b": True})["label"] == "neither"
    assert r.centred_label({"a": cum, "b": stp}, {"a": True, "b": False})["label"] == "consistent with cumulative"
    assert r.centred_label({"a": cum}, {"a": False})["label"] == "unreadable"
    nf = r.label(r.date_on(np.zeros(len(r.GRID))), a0)
    assert r.centred_label({"a": nf}, {"a": True})["label"] == "not formed by 4000"


def test_r1_is_the_released_date_plus_or_minus_one_pythia_grid_step():
    assert r.r1_pass(256, 256) and r.r1_pass(512, 256) and r.r1_pass(128, 256)
    assert not r.r1_pass(1000, 256) and not r.r1_pass(None, 256)
    assert r.PYTHIA[-1] == 4000 and set(r.PYTHIA) <= set(r.GRID)


def test_induction_score_reads_the_planted_diagonal():
    L = 4
    att = np.zeros((1, 2, 2 * L, 2 * L))
    for i in range(L, 2 * L):
        att[0, 1, i, i - L + 1] = 1.0
        att[0, 0, i, 0] = 1.0
    sc = r.induction_scores(att, L)
    assert sc[0, 1] == 1.0 and sc[0, 0] == pytest.approx(0.0)
