"""C4′ (c), the replicate pair (`trackC_writeup/design-C4pc.md`): π's file, the registration
guard, H4′'s tolerance and the five leans' rules on planted inputs."""
import json
import math

import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackC_writeup import c4_train as ct
from trackC_writeup import c4pc_rules as r
from trackC_writeup import c4pc_train as t


# ---------------------------------------------------------------- π

def test_perm_is_seeded_and_whole():
    p = t.perm()
    assert len(p) == ct.N_ROWS == 4_096_000
    assert np.array_equal(np.sort(p), np.arange(ct.N_ROWS))
    assert np.array_equal(p, t.perm()) and t.perm_sha(p) == t.perm_sha(t.perm())
    assert p.max() < ct.HELD_OUT[0]                     # H4's rows are never trained on


def test_identity_batches_equal_c4s():
    rows = np.arange(ct.N_ROWS * 3, dtype=np.uint16).reshape(-1, 3)   # a stand-in shard, ROW 3
    for i in (0, 1, 1999, ct.HORIZON - 1):
        assert np.array_equal(t.batch_of(rows, i), ct.batch_of(rows, i))
        assert np.array_equal(t.source_rows(np.arange(ct.N_ROWS), i), np.arange(i * 1024, (i + 1) * 1024))
    with pytest.raises(t.C4pcRefused):
        t.batch_of(rows, ct.HORIZON)


def test_write_permuted_small(tmp_path):
    n, row = 4096, 5
    src = np.random.default_rng(0).integers(0, 50000, (n, row)).astype(np.uint16)
    p = np.random.default_rng(1).permutation(n).astype(np.int64)
    dst = np.zeros_like(src)
    t.write_permuted(src, dst, p, tmp_path / "b", n_buckets=8, chunk=300, log=lambda s: None)
    assert np.array_equal(dst, src[p])
    assert t.sample_check(dst, src, p, n=256)["mismatches"] == 0
    dst[7] += 1
    assert t.sample_check(dst, src, p, n=n)["mismatches"] == 1
    assert not list((tmp_path / "b").iterdir())          # buckets removed


# ---------------------------------------------------------------- the guard and H4′

def test_registration_guard(tmp_path):
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"predictions": [{"id": "P-SCH1"}]}))
    dec = tmp_path / "declined.json"
    assert set(t.registration(reg, dec, check_git=False).values()) == {"pending"}
    with pytest.raises(t.C4pcRefused):
        t.require_decided(reg, dec, check_git=False)
    reg.write_text(json.dumps({"predictions": [{"id": i} for i in t.ENTRY_IDS[:3]]}))
    dec.write_text(json.dumps({"declined": list(t.ENTRY_IDS[3:])}))
    st = t.require_decided(reg, dec, check_git=False)
    assert list(st.values()) == ["registered"] * 3 + ["declined"] * 2
    assert len(t.ENTRY_IDS) == 5 and t.ENTRY_IDS[0].endswith("2") and t.ENTRY_IDS[-1].endswith("6")


def test_decline_must_be_committed(tmp_path):
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"predictions": []}))
    stray = t.REPO / "trackC_writeup" / "c4pc_declined_test_stray.json"      # untracked, in the repo
    try:
        stray.write_text(json.dumps({"declined": list(t.ENTRY_IDS)}))
        assert not t.committed(stray)
        with pytest.raises(t.C4pcRefused):
            t.registration(reg, stray)
    finally:
        stray.unlink(missing_ok=True)
    assert not t.DECLINED.exists()                    # nothing declined yet: Blocked 167 is open


def test_prior_losses(tmp_path):
    log = tmp_path / "train_log.jsonl"
    log.write_text("\n".join(json.dumps(x) for x in (
        {"segment": {"from": 0}}, {"step": 1, "loss": 10.0}, {"step": 2, "loss": 9.0},
        {"segment": {"from": 1}}, {"step": 2, "loss": 8.5}, {"step": 3, "loss": 8.0})) + "\n")
    assert t.prior_losses(log, 3) == [10.0, 8.5, 8.0]
    assert t.prior_losses(tmp_path / "none.jsonl", 0) == []
    with pytest.raises(t.C4pcRefused):
        t.prior_losses(log, 4)


def test_h4_tol():
    rec = lambda gaps: {"rel_diff": {"EleutherAI/pythia-70m": 0.0, "EleutherAI/pythia-70m-warmup01": 0.45,
                                     **{f"EleutherAI/pythia-70m-seed{k}": g for k, g in enumerate(gaps, 1)}}}
    assert t.h4_tol({1000: rec([0.01, -0.015])})["tol"] == 0.02            # the base; warmup01 not a seed
    assert t.h4_tol({1000: rec([0.01]), 4000: rec([-0.031])})["tol"] == pytest.approx(0.031)


# ---------------------------------------------------------------- the rules

GRID = ct.GRID


def mat(sets_by_step, n=6):
    """A (steps, transitions) matrix with 7 violators at each listed transition."""
    m = np.zeros((len(GRID), n), int)
    for i, s in enumerate(GRID):
        for tr in sets_by_step(s):
            m[i, tr] = 7
    return m


def test_l7():
    end = lambda s: (1, 2)
    tr = lambda s: (1, 2, 3) if 48 <= s <= 64 else (1, 2)
    hold = r.l7({"x": mat(tr)}, {"x": mat(end)}, GRID, ["x"])
    assert hold["holds"] and hold["per_arm"]["x"]["F_transient"] == [3]
    assert r.l7({"x": mat(end)}, {"x": mat(end)}, GRID, ["x"])["holds"] is False          # no transient
    late = lambda s: (1, 2, 3) if s == 1250 else (1, 2)                                     # after 1000
    assert r.l7({"x": mat(late)}, {"x": mat(end)}, GRID, ["x"])["holds"] is False
    other_end = lambda s: (1, 2, 3) if s <= 64 else (1,)
    assert r.l7({"x": mat(other_end)}, {"x": mat(end)}, GRID, ["x"])["holds"] is False
    assert r.l7({"x": mat(tr)}, {"x": mat(tr)}, GRID, ["x"])["holds"] is False             # A0 has one too
    assert r.l7({}, {}, GRID, [])["holds"] is None


def test_l8():
    assert r.l8(mat(lambda s: (1,)), mat(lambda s: (2,)), GRID)["holds"]
    assert not r.l8(mat(lambda s: (2,)), mat(lambda s: (2,)), GRID)["holds"]
    assert not r.l8(mat(lambda s: () if s == 2000 else (1,)), mat(lambda s: (2,)), GRID)["holds"]
    assert r.l8(mat(lambda s: (1, 5)), mat(lambda s: (2, 3)), GRID)["holds"]


def dated(s):
    return {"step": s, "status": "dated" if s else "not formed by 4000"}


def test_l9_known_answer_and_bar():
    assert r.nearer_bar(768) == pytest.approx(768 ** 1.5 / math.sqrt(2860))
    assert round(r.nearer_bar(768)) == 398 and round(r.nearer_bar(640)) == 303
    res = r.l9(dated(320), dated(768), {"k": dated(48)}, {"k": dated(640)}, ["k"])
    assert res["holds"] and res["i"] and res["ii"]
    assert res["d_induction"] == pytest.approx(0.44, abs=0.01)
    assert res["per_arm"]["k"]["d"] == pytest.approx(1.09, abs=0.01)
    assert r.l9(dated(400), dated(768), {"k": dated(48)}, {"k": dated(640)}, ["k"])["holds"] is False   # (i)
    assert r.l9(dated(320), dated(768), {"k": dated(140)}, {"k": dated(640)}, ["k"])["holds"] is False  # (ii)
    assert r.l9(dated(None), dated(768), {"k": dated(48)}, {"k": dated(640)}, ["k"])["holds"] is None
    assert r.l9(dated(320), dated(2000), {"k": dated(48)}, {"k": dated(640)}, ["k"])["holds"] is None   # past W


def test_l9h():
    def sc(layer):
        x = np.zeros((6, 8))
        x[layer, 6] = 1.0
        return x
    assert r.l9h(sc(2), sc(3))["holds"] and r.l9h(sc(2), sc(4))["holds"]
    assert not r.l9h(sc(3), sc(3))["holds"] and not r.l9h(sc(2), sc(2))["holds"]


def test_l10():
    b = r.l10b([2.0] * 8 + [0.0] * 4, [1.0] * 12)
    assert b["n_beyond"] == 8 and b["holds"]
    assert not r.l10b([2.0] * 7 + [0.0] * 5, [1.0] * 12)["holds"]
    assert r.l10("another path", b)["holds"] and not r.l10("mixed", b)["holds"]
    assert r.l10("another path", None)["holds"] and r.l10(None, b)["holds"] is None
    null = r.l10b([0.0, 0.0, 2.0, 2.0], [1.0] * 4)                        # 2 of 4 < ⅔
    seeds = {f"s{k}": r.l10b([2.0, 2.0, 2.0, 0.0], [1.0] * 4) for k in range(9)}
    assert r.l10b_gate(null, seeds)["pass"]
    assert not r.l10b_gate(r.l10b([2.0] * 3 + [0.0], [1.0] * 4), seeds)["pass"]


def test_gate_entry():
    h, n = {"holds": True, "status": "read"}, {"holds": False, "status": "read"}
    u = {"holds": None, "status": "unreadable"}
    assert r.gate_entry(h, n, {"a": n, "b": h})["pass"]
    assert not r.gate_entry(h, n, {"a": h, "b": h})["pass"]
    assert not r.gate_entry(h, u, {"a": n})["pass"]                    # the null must be readable
    assert not r.gate_entry(u, n, {"a": n})["pass"]
