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
    out = np.zeros((Q, pa.shape[0], pb.shape[0]))
    for h in range(pa.shape[0]):
        for g in range(pb.shape[0]):
            for b0 in range(pos.min(), pos.max() + 1, b7.BIN):
                m = J[h, g] & (pos >= b0) & (pos < b0 + b7.BIN)
                p = np.bincount(pos[m] - pa[h, m], minlength=600).astype(float)
                q = np.bincount(pos[m] - pb[g, m], minlength=600).astype(float)
                for r in np.nonzero(m)[0]:
                    i = pos[r]
                    va, vb = p[:i], q[:i]
                    out[pid[r], h, g] += (va / va.sum() * vb / vb.sum()).sum()
    return out


def test_n0_sums_matches_a_direct_loop():
    rng = np.random.default_rng(1)
    pos = np.concatenate([np.arange(16, n) for n in NS[:3]])
    pid = np.concatenate([np.full(n - 16, q) for q, n in enumerate(NS[:3])])
    pa = np.stack([rng.integers(np.maximum(1, pos - 6), pos + 1) for _ in range(3)])
    pb = np.stack([rng.integers(np.maximum(1, pos - 6), pos + 1) for _ in range(2)])
    pa[0, rng.random(len(pos)) < 0.3] = -1
    J = (pa[:, None] >= 0) & (pb[None] >= 0)
    got = b7.n0_sums(*(torch.as_tensor(x) for x in (pa, pb, J, pos, pid)), 3).numpy()
    assert np.allclose(got, _n0_loop(pa, pb, J, pos, pid, 3))


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
