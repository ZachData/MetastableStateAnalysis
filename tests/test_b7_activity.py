"""B7 unit 1's parts (`trackB_circuit/design-B7.md`): condensed rows and partners, the planted
stacks, N0 against a direct loop (check 6's `n0_rows`), leave-one-prompt-out matching against a
brute force, neighbours, D, and G1–G6 on a small synthetic cloud."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trackB_circuit import b7_activity as b7

pytestmark = pytest.mark.deps  # torch

NS = (96, 110, 84, 120, 100, 90, 104)
L, H = 6, 4


def _sinks(val=0.3):
    return [torch.full((L, H, n - b7.ROW_FROM), val, dtype=torch.float64) for n in NS]


def test_partners_condensed_rule_and_sink():
    n = 20
    A = torch.zeros((1, 1, n, n), dtype=torch.float64)
    for i in range(n):
        A[0, 0, i, : i + 1] = 1 / (i + 1)
    A[0, 0, 16] = 0
    A[0, 0, 16, 0], A[0, 0, 16, 5], A[0, 0, 16, 6] = 0.5, 0.26, 0.24        # top share 0.52
    A[0, 0, 17] = 0
    A[0, 0, 17, 0], A[0, 0, 17, 3] = 0.97, 0.03                             # w < W_MIN
    A[0, 0, 18] = 0
    A[0, 0, 18, 0], A[0, 0, 18, 2], A[0, 0, 18, 9] = 0.1, 0.45, 0.45        # 0.5 each: not > 1/2
    part, inn = b7.partners(A)
    assert part[0, 0].tolist() == [5, -1, -1, -1]
    assert inn[0, 0].tolist() == [True, False, True, True]


def test_plant_rows_sum_and_recover_the_map():
    rng = np.random.default_rng(0)
    sink = torch.as_tensor(rng.uniform(0, 0.99, (2, 3, 64 - 16)))
    mp = torch.as_tensor(np.stack([[b7.draw_map(64, rng) for _ in range(3)] for _ in range(2)]))
    mp[0, 0, :5] = -1
    A = b7.plant(sink, mp, 64)
    assert torch.allclose(A.sum(-1), torch.ones(()).double())
    assert torch.allclose(A[..., 16:, 0], sink)
    part, inn = b7.partners(A)
    want = torch.where(inn, mp, -1)
    assert torch.equal(part, want)


def _n0_loop(pa, pb, J, pos, pid, Q):
    """N0 by brute force over ordered pairs (j, j') of different rows (Blocked 111 (a))."""
    out, rows = np.zeros((Q, pa.shape[0], pb.shape[0])), np.zeros((Q, pa.shape[0], pb.shape[0]))
    for h in range(pa.shape[0]):
        for g in range(pb.shape[0]):
            for b0 in range(pos.min(), pos.max() + 1, b7.BIN):
                m = np.nonzero(J[h, g] & (pos >= b0) & (pos < b0 + b7.BIN))[0]
                oa, ob = pos[m] - pa[h, m], pos[m] - pb[g, m]
                for r in m:
                    i = pos[r]
                    num = den = 0
                    for j in range(len(m)):
                        for k in range(len(m)):
                            if j != k and oa[j] <= i - 1 and ob[k] <= i - 1:
                                den += 1
                                num += oa[j] == ob[k]
                    if den:
                        out[pid[r], h, g] += num / den
                        rows[pid[r], h, g] += 1
    return out, rows


def _bins_case(rng, pa_from=None):
    pos = np.concatenate([np.arange(16, n) for n in NS[:3]])
    pid = np.concatenate([np.full(n - 16, q) for q, n in enumerate(NS[:3])])
    pa = np.stack([rng.integers(np.maximum(1, pos - 6), pos + 1) for _ in range(3)])
    pb = np.stack([rng.integers(np.maximum(1, pos - 6), pos + 1) for _ in range(2)])
    if pa_from is not None:
        pa[0] = pb[pa_from]                                   # one head pair frozen
    pa[1, rng.random(len(pos)) < 0.3] = -1
    J = (pa[:, None] >= 0) & (pb[None] >= 0)
    return pa, pb, J, pos, pid


def test_n0_sums_matches_a_brute_force_over_different_rows():
    rng = np.random.default_rng(1)
    pa, pb, J, pos, pid = _bins_case(rng, pa_from=1)
    J[:, :, rng.random(len(pos)) < 0.97] = False              # sparse: some rows with no pair
    got, rows = b7.n0_sums(*(torch.as_tensor(x) for x in (pa, pb, J, pos, pid)), 3)
    want, wrows = _n0_loop(pa, pb, J, pos, pid, 3)
    assert np.allclose(got.numpy(), want) and np.array_equal(rows.numpy(), wrows)
    assert (wrows.sum((0)) < J.sum(-1)).any()                 # undefined rows occur and are left out


def test_n0_does_not_rise_with_persistence_at_the_minimum_entry():
    n0 = {}
    for arm in ("G1", "G2"):
        st = b7.planted_stages(arm, _sinks(), NS, seed=21)
        res = b7.run_step(st, seed=22, n_boot=100, thin=b7.MIN_JOINT)
        n0[arm] = np.mean([c["N0"] for c in res["cells"] if "N0" in c])
    # unit 1's N0 (self-pairs in) put G1 near 1/m, m about 12 rows a bin here: about 0.08
    assert n0["G1"] < 2 * n0["G2"] + 0.01, n0


def test_matched_is_leave_one_prompt_out():
    rng = np.random.default_rng(2)
    Q, G = 5, 4
    agree = rng.integers(0, 30, (Q, 3, G)).astype(float)
    joint = agree + rng.integers(20, 40, (Q, 3, G))
    M = torch.as_tensor(np.array([[1.0] * Q, [2, 0, 1, 1, 1]]))
    m = b7.matched(torch.as_tensor(agree), torch.as_tensor(joint), M, min_joint=0)
    for d, mm in enumerate(M.numpy()):
        num = den = 0.0
        for h in range(3):
            for q in range(Q):
                if mm[q] == 0:
                    continue
                s = sum(mm[p] * agree[p, h] for p in range(Q) if p != q)
                g = int(np.argmax(s))
                num += mm[q] * agree[q, h, g]
                den += mm[q] * joint[q, h, g]
        assert abs(m["P"][d] - num / den) < 1e-12


def test_neighbours_and_d():
    rng = np.random.default_rng(3)
    X = torch.as_tensor(rng.normal(size=(2, 40, 8)))
    nb = b7.neighbours(X).numpy()
    U = (X / X.norm(dim=-1, keepdim=True)).numpy()
    for l in range(2):
        for i in range(16, 40):
            assert nb[l, i - 16] == 1 + int(np.argmax(U[l, 1:i] @ U[l, i]))
    g = b7.g9()
    assert g["pass"] and abs(g["flip_D1"] - 2) < 1e-9


@pytest.mark.parametrize("arm,want", [("G1", "frozen across depth"), ("G3", "depth-local persistence"),
                                      ("G4", "positional, not separable"), ("G5", "rerouting"),
                                      ("G6", "not read"), ("G2", "rerouting")])
def test_planted_arms_on_a_small_cloud(arm, want):
    st = b7.planted_stages(arm, _sinks(), NS, seed=11)
    res = b7.run_step(st, seed=12, n_boot=200)
    desc = [c["description"] for c in res["cells"]]
    assert set(desc) == {want}, desc
    if arm in ("G1", "G4", "G6"):
        assert b7.arm_check(arm, res)["pass"]
    if arm == "G5":
        assert all(c["two_cycle"] for c in res["cells"] if "P2" in c)
    if arm == "G3":
        r = [(c["P"] - c["N0"]) / (1 - c["N0"]) for c in res["cells"]]
        assert all(abs(x - 0.8) < 0.06 for x in r), r


def test_thinning_caps_joint_rows():
    st = b7.planted_stages("G2", _sinks(), NS, seed=5)
    cnt = b7.pair_counts(st, thin=b7.MIN_JOINT, seed=6)
    tot = cnt["joint"].sum(2)
    assert float(tot.max()) == b7.MIN_JOINT


def test_a_nonfinite_interval_is_not_read():
    nan = (float("nan"), float("nan"))
    assert b7.describe(4, 0.01, nan, (0.1, 0.2))[0] == "not read"
    assert b7.describe(4, 0.01, (0.1, 0.2), (0.1, 0.2), bad0=0.3) == ("not read", "P - N0: 0.300 of draws not finite")
    assert b7.describe(4, 0.01, (0.1, 0.2), nan)[0] == "not read"
    assert b7.describe(4, 0.01, (-0.1, 0.2), nan) == ("rerouting", None)
    assert b7.describe(4, 0.01, (0.1, 0.2), (0.1, 0.2), bad0=0.04) == ("depth-local persistence", None)


@pytest.mark.parametrize("arm", ["G1", "G3"])
def test_g1_g3_pass_at_the_minimum_entry(arm):
    st = b7.planted_stages(arm, _sinks(), NS, seed=31)
    rec = b7.arm_check(arm, b7.run_step(st, seed=32, n_boot=200, thin=b7.MIN_JOINT), min_entry=True)
    assert rec["pass"], rec


def test_g2_passes_pooled():
    st = b7.planted_stages("G2", _sinks(), NS, seed=41)
    rec = b7.arm_check("G2", b7.run_step(st, seed=42, n_boot=200))
    assert rec["pass"] and abs(rec["pooled_rel_dev"]) <= b7.G2_MAX, rec


def test_the_read_path_beside_columns_run_and_write_json(tmp_path, monkeypatch):
    rng = np.random.default_rng(51)
    st = []
    for n in NS:
        part = torch.as_tensor(b7.arm_maps("G3", n, L, H, rng))
        A = b7.plant(torch.full((L, H, n - 16), 0.3, dtype=torch.float64), part, n)
        X = torch.as_tensor(rng.normal(size=(L + 1, n, 16)))
        s = b7.prompt_stage(A, X / X.norm(dim=-1, keepdim=True), soft=True)
        s.pop("d")
        st.append(s)
    res = b7.run_step(st, seed=52, n_boot=100, beside=True)
    c = res["cells"][0]
    for k in ("soft", "pooled_P", "nb_P", "nb_N1", "nb_P_minus_N1_ci"):
        assert k in c, k
    assert 0.8 < c["soft"] <= 1.0
    from trackB_circuit import b3_route
    monkeypatch.setattr(b3_route, "load_rows", lambda out, step: ({}, {"code": b7.B3_ROWS_CODE}))
    monkeypatch.setattr(b3_route, "alpha_of", lambda rows: np.full(b7.N_LAYERS * b7.N_HEADS, np.nan))
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        res["alpha"] = b7.alpha_beside(512)                     # all-NaN stub
    b7.dump(tmp_path / "s.json", res)                        # pooled_P is NaN here: written as null
    text = (tmp_path / "s.json").read_text()
    assert "NaN" not in text and "null" in text


def test_read_refuses_a_gate_on_other_inputs(monkeypatch):
    monkeypatch.setattr(b7, "prompts_hash", lambda: b7.PROMPTS_HASH)      # prompts are untracked
    good = {"pass": True, "G0": {"sha256": {"x": "y"}},
            "meta": {"code": b7.code_sha(), "prompts_hash": b7.PROMPTS_HASH,
                     "runs": {str(k): v for k, v in b7.RUNS.items()}}}
    b7.check_gate(good)
    for k, v in (("code", "0" * 12), ("prompts_hash", "0" * 12), ("runs", {})):
        bad = {**good, "meta": {**good["meta"], k: v}}
        with pytest.raises(b7.B7Error):
            b7.check_gate(bad)
    with pytest.raises(b7.B7Error):
        b7.check_gate({**good, "G0": {}})


@pytest.mark.xfail(strict=True, reason="Blocked 112: under G1's identical heads the leave-one-out h* picks "
                   "the head with fewest rows in the held-out prompt, so cells do not enter at the minimum entry")
def test_g1_at_the_minimum_entry_with_sinks_that_differ_by_head():
    rng = np.random.default_rng(61)
    sinks = [torch.as_tensor(rng.choice([0.3, 0.99], size=(L, H, n - b7.ROW_FROM), p=[0.6, 0.4])) for n in NS]
    st = b7.planted_stages("G1", sinks, NS, seed=62)
    rec = b7.arm_check("G1", b7.run_step(st, seed=63, n_boot=200, thin=b7.MIN_JOINT), min_entry=True)
    assert rec["pass"], rec
