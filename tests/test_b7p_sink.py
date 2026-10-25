"""B7′ unit 1's parts (`trackB_circuit/design-B7p.md`): the drop against a direct renormalisation,
the rule per head and its passes, the generalised plant against B7's, N0 per draw against B7's
`n0_sums` (row 0) and a brute force over pairs of different original rows (a draw), matching and
counts against B7's, and G10, G11, G12, G14 on a small synthetic cloud."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trackB_circuit import b7_activity as b7
from trackB_circuit import b7p_sink as bp

pytestmark = pytest.mark.deps  # torch

NS = (96, 110, 84, 120, 100, 90, 104)
L, H = 6, 4


def _sinks(ns=NS, l=L, h=H, val=0.3):
    return [torch.full((l, h, n - b7.ROW_FROM), val, dtype=torch.float64) for n in ns]


def _rand_stack(n=40, l=2, h=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.rand((l, h, n, n), generator=g, dtype=torch.float64) * torch.tril(torch.ones(n, n, dtype=torch.float64))
    return A / A.sum(-1, keepdim=True)


def test_drop_empty_is_b7_and_renormalises():
    A = _rand_stack()
    p0, i0 = b7.partners(A)
    B = bp.drop_(A.clone(), [])
    assert torch.equal(b7.partners(B)[0], p0) and torch.equal(b7.partners(B)[1], i0)
    S = [3, 7, 20]
    B = bp.drop_(A.clone(), S)
    keep = torch.ones(A.shape[-1], dtype=torch.bool)
    keep[0] = False
    keep[S] = False
    R = A[..., 16:, :]
    w = 1 - R[..., 0] - R[..., S].sum(-1)
    w[..., 20 - 16] = 0                                           # query 20 is in S: out
    assert torch.allclose(1 - B[..., 16:, 0], w)
    K = R * keep
    top, arg = K.max(-1)
    inn = (w >= b7.W_MIN)
    inn[..., 20 - 16] = False                                     # query 20 is in S
    want = torch.where(inn & (top > 0.5 * K.sum(-1)), arg, -1)
    assert torch.equal(b7.partners(B)[0], want)
    assert torch.allclose(B.sum(-1), torch.ones_like(B.sum(-1)))


def _sink_stack(n=120, frac=0.3, key=12, seed=0, l=2, h=2, heads=None):
    """Every row condensed: a share `frac` of rows on `key` in `heads`, the rest on random partners."""
    rng = np.random.default_rng(seed)
    part = np.stack([[b7.draw_map(n, rng) for _ in range(h)] for _ in range(l)])
    for (a, b) in (heads or [(0, 0)]):
        on = rng.random(n - 16) < frac
        on &= np.arange(16, n) > key
        part[a, b, on] = key
    return b7.plant(torch.full((l, h, n - 16), 0.2, dtype=torch.float64), torch.as_tensor(part), n)


def test_rule_per_head_threshold_and_record():
    A = _sink_stack(frac=0.45, heads=[(1, 1)])
    r = bp.sink_rule(A)
    assert r["keys"] == [12] and r["found"][0]["pass"] == 1
    assert [x[:2] for x in r["found"][0]["heads"]] == [[1, 1]]
    assert r["unconverged"] == []
    assert float(A[..., 12].abs().max()) == 0.0                  # dropped in place
    assert bp.sink_rule(_sink_stack(frac=0.15, heads=[(1, 1)]))["keys"] == []
    few = _sink_stack(n=60, frac=0.9, heads=[(0, 0)])            # 44 condensed rows < SINK_MIN
    assert bp.sink_rule(few)["keys"] == []


def test_rule_passes_and_unconverged():
    """A key that holds ≥ 1/4 only once a bigger one is dropped is found at pass 2."""
    n = 200
    rng = np.random.default_rng(1)
    part = np.stack([[b7.draw_map(n, rng)]])
    i = np.arange(16, n)
    part[0, 0, (i > 40) & (rng.random(n - 16) < 0.6)] = 30
    rest = part[0, 0] != 30
    part[0, 0, rest & (i > 20) & (rng.random(n - 16) < 0.35)] = 15
    A = b7.plant(torch.full((1, 1, n - 16), 0.1, dtype=torch.float64), torch.as_tensor(part), n)
    # rows on 30 carry a second, smaller mass on 15 so they move there after 30 is dropped
    on = torch.as_tensor(part[0, 0] == 30)
    A[0, 0, 16:][on, 15] += 0.25
    A[0, 0, 16:][on, 30] -= 0.25
    r = bp.sink_rule(A.clone(), passes=1)
    assert r["keys"] == [30] and r["unconverged"] and r["unconverged"][0]["key"] == 15
    r = bp.sink_rule(A.clone())
    assert r["keys"] == [15, 30] and [f["pass"] for f in r["found"]] == [1, 2]


def test_plant2_is_b7_plant():
    rng = np.random.default_rng(0)
    n = 64
    sink = torch.as_tensor(rng.uniform(0, 0.99, (2, 3, n - 16)))
    part = torch.as_tensor(np.stack([[b7.draw_map(n, rng) for _ in range(3)] for _ in range(2)]))
    part[0, 0, :5] = -1
    A0 = b7.plant(sink, part, n)
    A1 = bp.plant2(sink, part, torch.full_like(sink, b7.PLANT), torch.full_like(part, -1), torch.zeros_like(sink), n)
    assert torch.allclose(A0, A1, atol=1e-15)
    key = torch.full_like(part, 9)
    A2 = bp.plant2(sink, part, torch.full_like(sink, 0.2), key, torch.full_like(sink, 0.7), n)
    assert torch.allclose(A2.sum(-1), torch.ones_like(A2.sum(-1)))
    w = 1 - sink[1, 2, 30]
    assert abs(float(A2[1, 2, 46, 9]) - float(w * (0.7 + 0.1 / 46 + (0.2 if part[1, 2, 30] == 9 else 0)))) < 1e-12


def _counts(arm="G3", seed=3):
    st = b7.planted_stages(arm, _sinks(), NS, seed)
    part, pos, pid = b7._cat(st, "part")
    return st, part, pos, pid


def test_n0_draws_row0_is_b7():
    st, part, pos, pid = _counts()
    cond = part >= 0
    J = cond[0][:, None, :] & cond[1][None, :, :]
    M = b7.draws(len(st), 5, 0, "cpu")
    off = torch.where(cond, pos - part, 0)
    s7, r7 = b7.n0_sums(part[0], part[1], J, pos, pid, len(st))
    s, r = bp.n0_draws(off[0], off[1], J, pos, pid, M)
    assert torch.allclose(s[0], s7, atol=1e-12) and torch.equal(r[0], r7)


def test_n0_draws_brute_force_on_a_draw():
    st, part, pos, pid = _counts(seed=5)
    cond = part >= 0
    J = cond[2][:2, None, :] & cond[3][None, :2, :]
    off = torch.where(cond, pos - part, 0)
    va, vb = off[2][:2], off[3][:2]
    M = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [3, 0, 2, 0, 1, 1, 0]], dtype=torch.float64)
    s, r = bp.n0_draws(va, vb, J, pos, pid, M)
    P, Q = pos.numpy(), pid.numpy()
    Jn, a_, b_ = J.numpy(), va.numpy(), vb.numpy()
    m = M[1].numpy()
    for h in range(2):
        for g in range(2):
            want_s, want_r = np.zeros(7), np.zeros(7)
            for i in np.nonzero(Jn[h, g])[0]:
                if m[Q[i]] == 0:
                    continue
                b = (P - 16) // 32 == (P[i] - 16) // 32
                js = np.nonzero(b & Jn[h, g])[0]
                num = den = 0.0
                for j in js:
                    for j2 in js:
                        if j == j2:
                            continue
                        if a_[h, j] <= P[i] - 1 and b_[g, j2] <= P[i] - 1:
                            ww = m[Q[j]] * m[Q[j2]]
                            den += ww
                            num += ww * (a_[h, j] == b_[g, j2])
                # copies of one original row never pair (j ≠ j2 above); copies of different rows do
                if den > 0:
                    want_s[Q[i]] += num / den
                    want_r[Q[i]] += 1
            got_s, got_r = s[1, :, h, g].numpy(), r[1, :, h, g].numpy()
            used = m > 0
            assert np.allclose(got_s[used], want_s[used], atol=1e-12)
            assert np.array_equal(got_r[used], want_r[used])


def test_matched_p_and_counts_are_b7():
    st, *_ = _counts()
    M = b7.draws(len(st), 50, 7, "cpu")
    for thin in (None, 30):
        c7 = b7.pair_counts(st, thin=thin, seed=7)
        cp = bp.pair_counts_p(st, M, thin=thin, seed=7)
        assert torch.equal(c7["agree"], cp["agree"]) and torch.equal(c7["joint"], cp["joint"])
        for (a, b) in ((0, 1), (2, 3), (1, 3)):
            m7 = b7.matched(c7["agree"][a, b], c7["joint"][a, b], M, {"N0s": c7["n0"][(a, b)][0]})
            mp = bp.matched_p(cp["agree"][a, b], cp["joint"][a, b], M,
                              {"N0s": cp["n0"][(a, b)][0][0]})                 # row 0's sums, 3-d
            np.testing.assert_array_equal(m7["P"], mp["P"])
            np.testing.assert_allclose(m7["N0s"], mp["N0s"], rtol=1e-12)
            for k in ("n_enter", "dropped", "n_enter_loo_rule"):
                assert m7[k] == mp[k]
            assert np.array_equal(m7["hstar"], mp["hstar"]) and np.array_equal(m7["enter"], mp["enter"])
            s7 = c7["n0"][(a, b)][0]
            assert torch.allclose(cp["n0"][(a, b)][0][0], s7, atol=1e-12)


def test_run_step_p_sample_values_are_b7():
    st, *_ = _counts()
    r7 = b7.run_step(st, 11, n_boot=40)["cells"]
    rp = bp.run_step_p(st, 11, n_boot=40)["cells"]
    for a, b in zip(r7, rp):
        for k in ("P", "N0", "N1", "joint_rows", "P2", "N0_2"):
            if k in a:
                assert abs(a[k] - b[k]) < 1e-12, (a["l"], k)
        assert "P_eq" in b and "loo_rule" in b


def test_g1d_maps_differ_by_head_and_are_found():
    rng = np.random.default_rng(0)
    perms = np.stack([rng.permutation(H) for _ in range(L)])
    m = bp.content_maps("G1d", 100, L, H, rng, perms)
    assert not np.array_equal(m[0, 0], m[0, 1])
    for l in range(L - 1):
        for h in range(H):
            g = int(np.nonzero(perms[l + 1] == perms[l][h])[0][0])
            assert np.array_equal(m[l, h], m[l + 1, g])
    assert not np.isin(bp.draw_map_avoid(200, rng, [5, 9]), [5, 9]).any()


# ---------------------------------------------------------------- the second-sink arms, small cloud

CLOUD = (230, 260, 210, 250, 240, 220, 270)
CL, CH = 24, 6


@pytest.fixture(scope="module")
def cloud():
    rng = np.random.default_rng(42)
    return [torch.as_tensor(rng.uniform(0.2, 0.7, (CL, CH, n - 16))) for n in CLOUD]


def _arm(arm, cloud, thins=(None,), confound=False):
    return bp.sink_arm(arm, cloud, CLOUD, 100 + len(arm), 7, thins, confound=confound, n0k=True,
                       key_range=(10, 30), n_boot=200)


@pytest.fixture(scope="module")
def g10(cloud):
    return _arm("G10", cloud, confound=True)


def test_g10_small_cloud(g10):
    assert g10["a"], g10["prompts"]
    assert g10["d"]["pass"], g10["d"]
    assert bp.check_c(g10["runs"]["full"])["pass"]


@pytest.mark.xfail(strict=True, reason="the plant as frozen: 2 heavy prompts of 7, so (5/7)^7 = 9.5 % of draws "
                   "hold neither and P - N0's 2.5 % bound sits at 0; B7's instrument reads mid cells 'rerouting'")
def test_g10_b_reproduces_the_confound(g10):
    assert g10["b"]["pass"], g10["b"]


@pytest.fixture(scope="module")
def g11(cloud):
    return _arm("G11", cloud, thins=(None, 50))


def test_g11_small_cloud(g11):
    assert g11["a"] and g11["d"]["pass"]
    for run in g11["runs"].values():
        read = [c for c in run["cells"] if "P" in c]
        assert len(read) == len(run["cells"]) and min(c["P"] for c in read) >= 0.99
        assert min(c["P"] - c["N0"] for c in read) >= 0.9


@pytest.mark.xfail(strict=True, reason="the plant as frozen: the light prompts' sub-threshold key in 7-20 over "
                   "frozen content puts N1 below P (distant pairs reach full-content layers), so cells read "
                   "'depth-local', not 'frozen'")
def test_g11_frozen_across_depth(g11):
    assert b7.arm_check("G1", g11["runs"]["full"])["pass"]


def test_g12_small_cloud(cloud):
    r = _arm("G12", cloud)
    assert r["a"] and r["d"]["pass"]
    rec = b7.arm_check("G3", r["runs"]["full"])
    assert abs(rec["r_pooled"] - 0.8) < 0.03, rec


def test_g14_small_cloud(cloud):
    r = _arm("G14", cloud)
    assert r["a"], r["prompts"]
    assert r["d"]["pass"]
    assert bp.check_c(r["runs"]["full"])["pass"]
