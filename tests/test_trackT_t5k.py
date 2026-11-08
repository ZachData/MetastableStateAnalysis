"""T5′ U1 (`trackT_timaeus/t5k.py`): the scorer, the dating rule, the outcome table and the plant,
on synthetic inputs, written before any output (`trackT_timaeus/design-T5p.md`)."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.deps  # t5k imports a1a
from trackT_timaeus import t5k


def test_draws_share_first_half():
    d = t5k.draws(np.arange(10, 50), seed=1, r=3, t=8)
    for k in ("rep", "ctrl", "null"):
        assert d[k].shape == (3, 16) and d[k].min() >= 10
        assert (d[k][:, :8] == d["rep"][:, :8]).all()
    assert (d["rep"][:, 8:] == d["rep"][:, :8]).all()
    assert not (d["ctrl"][:, 8:] == d["null"][:, 8:]).all()


def test_text_sets_rows_and_halves():
    """I_text (Blocked 139 (a)): x, x′, x″ are rows 0–r−1, r–2r−1, 2r–3r−1, each cut to t tokens."""
    r, t = 3, 4
    rows = np.arange(3 * r * 10).reshape(3 * r, 10)
    d = t5k.text_sets(rows, r=r, t=t)
    x, x1, x2 = rows[:r, :t], rows[r:2 * r, :t], rows[2 * r:, :t]
    assert (d["rep_text"] == np.concatenate([x, x], 1)).all()
    assert (d["ctrl_text"] == np.concatenate([x, x1], 1)).all()
    assert (d["null_text"] == np.concatenate([x, x2], 1)).all()
    with pytest.raises(t5k.T5kRefused):
        t5k.text_sets(rows[:-1], r=r, t=t)
    with pytest.raises(t5k.T5kRefused):
        t5k.text_sets(rows[:, :3], r=r, t=t)


def test_text_rows_are_unseen():
    """I_text's rows follow ICL's held-out rows without overlap (and the 64 the refused gate read)."""
    assert t5k.TEXT_LO == t5k.HELD_LO + t5k.N_HELD == 1088
    assert t5k.TEXT_LO + 3 * t5k.R == 1184


def test_summarize_text_beside():
    rng = np.random.default_rng(2)
    ctrl = rng.uniform(0, 0.01, (32, 6))
    base = {"pm_rep": ctrl, "pm_ctrl": ctrl, "pm_null": ctrl, "prev_ctrl": ctrl, "loss_rep": np.ones(32),
            "loss_ctrl": np.ones(32), "icl_late": np.ones(64), "icl_early": np.ones(64)}
    s0 = t5k.summarize(base, n_boot=50)
    assert "I_text" not in s0                                        # absent unless its arrays are
    tc = rng.uniform(0, 0.01, (32, 6))
    tr = tc.copy()
    tr[:, 2] += 0.3
    s = t5k.summarize({**base, "pm_rep_text": tr, "pm_ctrl_text": tc, "pm_null_text": tc + 0.002}, n_boot=50)
    assert s["I_text"] == pytest.approx(0.3) and s["argmax_text"] == 2
    assert s["I_text_null"] == pytest.approx(0.002)
    assert s["I"] == pytest.approx(0.0) and s["se"]["I"] == s0["se"]["I"]   # I's SEs untouched by I_text
    assert s["se"]["I_text"] >= 0 and "I_text_null" in s["se"]


def test_g3_scorer():
    assert t5k.g3()["pass"]


def test_pm_offsets():
    """PM reads query t + j → key j + 1 for j = 0 … t − 2 and nothing else."""
    torch = pytest.importorskip("torch")
    t = 5
    a = torch.zeros(1, 1, 2 * t, 2 * t, dtype=torch.float64)
    a[0, 0, t - 1, 0] = 1.0               # i = 0: excluded
    a[0, 0, 2 * t - 1, t] = 1.0           # i = t: excluded
    assert float(t5k.pm_of(a, t)[0, 0]) == 0.0
    a[0, 0, t + 2, 3] = 1.0               # j = 2
    assert float(t5k.pm_of(a, t)[0, 0]) == pytest.approx(1 / (t - 1))
    b = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    b[0, 0, 1, 0] = b[0, 0, 2, 1] = 1.0
    assert float(t5k.prev_of(b)[0, 0]) == pytest.approx(2 / 3)


def test_summarize_signs():
    rng = np.random.default_rng(0)
    ctrl = rng.uniform(0, 0.01, (32, 6))
    rep = ctrl.copy()
    rep[:, 4] += 0.5
    a = {"pm_rep": rep, "pm_ctrl": ctrl, "pm_null": ctrl + 0.001, "prev_ctrl": ctrl, "loss_rep": np.ones(32),
         "loss_ctrl": 2 * np.ones(32), "icl_late": np.ones(64), "icl_early": 3 * np.ones(64)}
    s = t5k.summarize(a, n_boot=50)
    assert s["I"] == pytest.approx(0.5) and s["argmax"] == 4
    assert s["I_null"] == pytest.approx(0.001)
    assert s["B"] == pytest.approx(-1.0) and s["ICL"] == pytest.approx(-2.0)
    assert s["se"]["B"] == pytest.approx(0.0)


STEPS = [0, 10, 20, 30, 40, 50, 60, 70]


def test_date_series_onset_and_half():
    x = [0.2, 0.0, 0.0, 0.05, 0.3, 0.9, 1.0, 1.0]   # min before max at 10 (first), peak 1.0 at 60
    r = t5k.date_series(STEPS, x, floor=0.1)
    assert r["base"] == 10 and r["peak"] == 60 and r["rise"] == pytest.approx(1.0)
    assert r["on_0.1"] == 40 and r["on_0.5"] == 50
    assert not r["censored"] and not r["refused"]


def test_date_series_floor_refuses():
    r = t5k.date_series(STEPS, [0, 0.01, 0.02, 0.05, 0.08, 0.09, 0.09, 0.09], floor=0.1)
    assert r["refused"]


def test_censored():
    r = t5k.date_series(STEPS, [0, 0, 0, 0, 0.1, 0.3, 0.6, 1.0], floor=0.1)
    assert r["censored"] and r["peak"] == 70
    r2 = t5k.date_series(STEPS, [0, 0, 0, 0, 0.1, 0.5, 0.95, 1.0], floor=0.1)
    assert not r2["censored"]                       # last step carries 0.05 < 0.1 of the rise


def test_se_floor():
    x = [0, 0, 0, 0, 0.1, 0.2, 0.3, 0.3]
    seen = []
    def se(i, j):
        seen.append((i, j))
        return 0.1
    assert t5k.date_series(STEPS, x, rise_se=se)["refused"]                   # 0.3 < 4 · 0.1
    assert seen == [(0, 6)]                                                    # base index, peak index
    assert not t5k.date_series(STEPS, x, rise_se=lambda i, j: 0.01)["refused"]


def test_paired_se_cancels_shared_noise():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 64)
    a_i = {"icl_late": base, "icl_early": np.zeros(64)}
    a_j = {"icl_late": base - 0.3, "icl_early": np.zeros(64)}
    assert t5k.paired_se(t5k.BESIDE["ICL"], a_i, a_j, 64, n_boot=200) == pytest.approx(0.0, abs=1e-12)
    a_k = {"icl_late": rng.normal(0, 1, 64), "icl_early": np.zeros(64)}
    assert t5k.paired_se(t5k.BESIDE["ICL"], a_i, a_k, 64, n_boot=200) > 0.1


def test_rho_dates():
    rho = [0.5, 1.0, 0.7, 0.2, 0.1, 0.4, 0.8, 0.9]
    rd = t5k.rho_dates(STEPS, rho)
    assert rd["peak"] == 10 and rd["trough"] == 40
    assert rd["fall"]["on_0.5"] == 30                 # 1.0 − 0.2 = 0.8 ≥ 0.45
    assert rd["rise"]["on_0.1"] == 50 and rd["rise"]["on_0.5"] == 60
    assert not rd["fall"]["censored"]
    assert [t5k.phase(s, rd) for s in (0, 10, 40, 50)] == ["early rise", "fall", "fall", "rise"]
    rd2 = t5k.rho_dates(STEPS, [0.5, 1.0, 0.7, 0.2, 0.1, 0.11, 0.4, 0.9])
    assert t5k.phase(50, rd2) == "trough"


def _i(on, cens=False, refused=False):
    return {"on_0.1": on, "censored": cens, "refused": refused}


def test_label_table():
    rr, rc = {"censored": False}, {"censored": True}
    L = t5k.label
    assert L(_i(None, refused=True), rr, None, 4369) == "no induction formed by 4369"
    assert L(_i(10), rr, -2, 4369) == "induction leads the rise"
    assert L(_i(10), rc, -3, 4369) == "induction leads the rise"          # ρ's censoring cannot move it
    assert L(_i(10, cens=True), rr, -3, 4369) == "censored, not ordered"
    assert L(_i(10), rr, 1, 4369) == "induction with the rise"
    assert L(_i(10), rc, 0, 4369) == "censored, not ordered"
    assert L(_i(10, cens=True), rr, 2, 4369) == "induction after the rise"   # I's censoring only moves it later
    assert L(_i(10), rc, 2, 4369) == "censored, not ordered"
    assert L(_i(10), rr, None, 4369) == "not ordered: ρ's rise not dated"


def test_gap_in_grid_steps():
    assert t5k.gap(STEPS, 20, 50) == -3 and t5k.gap(STEPS, None, 50) is None


def test_noise():
    n = t5k.noise({"A0": 0, "A0p": 2, "A0pp": 1, "A1": 4, "A2": 5, "A3": None})
    assert n["span"] == 2 and n["seed_dependent"] and n["range"] == [-2, 4]
    assert n["not_told_apart"] == {"A1": True, "A2": False, "A3": False}
    n2 = t5k.noise({"A0": None, "A0p": 0, "A0pp": 0, "A1": 0, "A2": 0, "A3": 0})
    assert n2["span"] is None and n2["not_run"].endswith(": A0")


def test_plant_and_g5_on_a_clean_grid():
    from trackT_timaeus import t2
    steps = [0] + [s for s in t2.GRID if s <= 4369]
    p = t5k.plant(steps, 2002)
    assert p[0] == 0 and p[steps.index(2002)] == pytest.approx(0.1)
    assert p[-1] < 0.2
    for c in t5k.PLANT_C["dodeca"]:
        assert t5k.g5_one(steps, [0.0] * len(steps), c)["pass"]
    from trackA_force import a3
    s410 = [s for s in a3.STEPS if s <= 4000]
    for c in t5k.PLANT_C["410m"]:
        assert t5k.g5_one(s410, [0.0] * len(s410), c)["pass"]


def test_break_fraction():
    held = np.ones((4, 520), dtype=np.int64)
    held[0, 45] = 0          # inside 40–509
    held[1, 20] = 0          # before the early window
    held[2, 509] = 0         # last position read
    assert t5k.break_fraction(held, 0) == pytest.approx(0.5)


class _FakeLM:
    """A stand-in causal LM (real torch, no transformers: conftest stubs it): seeded causal softmax
    attention per layer and head, random logits."""

    def __init__(self, torch, layers=2, heads=2, vocab=64):
        from types import SimpleNamespace
        self.torch, self.config, self.vocab = torch, SimpleNamespace(num_hidden_layers=layers,
                                                                    num_attention_heads=heads), vocab

    def __call__(self, input_ids, output_attentions=False):
        from types import SimpleNamespace
        t = self.torch
        b, n = input_ids.shape
        g = t.Generator().manual_seed(int(input_ids.sum()))
        mask = t.ones(n, n, dtype=t.bool).tril()
        att = tuple(t.softmax(t.randn(b, self.config.num_attention_heads, n, n, generator=g)
                              .masked_fill(~mask, float("-inf")), -1) for _ in range(self.config.num_hidden_layers))
        return SimpleNamespace(attentions=att if output_attentions else (),
                               logits=t.randn(b, n, self.vocab, generator=g))


def test_forward_returns_text_arrays():
    """`forward` on a fake LM: I_text's PM arrays present, per layer × head, per sequence."""
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "softmax") or type(torch).__name__ == "MagicMock":
        pytest.skip("torch is stubbed")
    r = 2
    sets = {**t5k.draws(np.arange(1, 64), seed=3, r=r),
            **t5k.text_sets(np.random.default_rng(4).integers(1, 64, (3 * r, t5k.T + 7)), r=r)}
    held = np.random.default_rng(5).integers(1, 64, (2, t5k.HELD_LEN))
    a = t5k.forward(_FakeLM(torch), sets, held, "cpu", bs=2)
    for k in ("pm_rep_text", "pm_ctrl_text", "pm_null_text", "pm_rep", "pm_ctrl", "pm_null"):
        assert a[k].shape == (r, 4) and np.isfinite(a[k]).all()
    assert not np.allclose(a["pm_rep_text"], a["pm_rep"])          # the text sets are read, not the draws
    s = t5k.summarize(a, n_boot=20)
    assert np.isfinite(s["I_text"]) and np.isfinite(s["I_text_null"]) and 0 <= s["argmax_text"] < 4


def test_read_run_dates_text_beside(monkeypatch):
    monkeypatch.setattr(t5k, "_pair", lambda run, steps, k: (lambda i, j: 0.0))
    n = len(STEPS)
    rise = [0, 0, 0, 0, 0.1, 0.3, 0.4, 0.4]
    data = {"steps": STEPS, "series": {"I": rise, "I_null": [0.001] * n, "P": rise, "B": [-v for v in rise],
                                       "ICL": [-v for v in rise], "I_text": [0, 0, 0, 0.02, 0.3, 0.35, 0.36, 0.36],
                                       "I_text_null": [0.002] * n, "rho_mid": [0.5, 1.0, 0.7, 0.2, 0.1, 0.4, 0.8, 0.8]},
            "se": {}}
    o = t5k.read_run("A0", data, 70)
    assert o["I_text"]["on_0.1"] == 40 and not o["I_text"]["refused"]
    assert o["I_text_null"]["refused"] and o["max_I_text_null"] == pytest.approx(0.002)
    assert o["phase"]["I_text"]["on_0.1"] == "fall" and o["d"]["I_text"] == -1
    assert "label" in o and o["label"] == t5k.label(o["I"], o["rho"]["rise"], o["d"]["I"], 70)


def test_text_rows_read_the_tail_rows(monkeypatch):
    """`text_rows` and `held_out` read tail rows 1088–1183 and 64–1087 (/challenge-pr on #324, finding 4):
    a fake tail whose row i holds i throughout."""
    from trackT_timaeus import t5h
    tail = np.repeat(np.arange(1300, dtype=np.uint16)[:, None], 600, 1)
    monkeypatch.setattr(t5h, "open_shard", lambda root: {"tail": tail})
    tx = t5k.text_rows("A0")
    assert tx.shape == (3 * t5k.R, t5k.T) and list(tx[:, 0]) == list(range(1088, 1184))
    s = t5k.text_sets(tx)
    assert set(s["ctrl_text"][:, t5k.T]) == set(range(1120, 1152)) and set(s["null_text"][:, t5k.T]) == set(range(1152, 1184))
    ho = t5k.held_out("A0")
    assert ho.shape == (t5k.N_HELD, t5k.HELD_LEN) and ho[0, 0] == 64 and ho[-1, 0] == 1087
    assert not set(tx[:, 0]) & set(ho[:, 0])
    monkeypatch.setattr(t5h, "open_shard", lambda root: {"tail": tail[:1100]})
    with pytest.raises(t5k.T5kRefused):
        t5k.text_rows("A0")
