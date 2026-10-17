"""
`trackA_force/a2_comp.py` — A2 unit 3's instrument: the frame's Jacobian against a finite step, the
path integral adds up to the chord, the per-component reads add up and the off shares sum to 1, the
mass sets.
The hooked split on a tiny GPT-NeoX: `test_trackA_a2_comp_smoke.py`.
"""
import numpy as np
import pytest

from trackA_force import a2_comp as ac

pytestmark = pytest.mark.deps  # torch


def _torch():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float64") or not hasattr(torch, "einsum"):
        pytest.skip("stub torch")
    return torch


def _ln(d, rng, affine=True):
    w = 1 + 0.3 * rng.standard_normal(d) if affine else np.ones(d)
    b = 0.1 * rng.standard_normal(d) if affine else np.zeros(d)
    return w, b


def test_jvp_is_the_frames_derivative():
    torch = _torch()
    rng = np.random.default_rng(0)
    n, d = 6, 16
    w, b = _ln(d, rng)
    x = torch.tensor(rng.standard_normal((n, d)) * 3 + 1)
    dX = torch.tensor(rng.standard_normal((2, n, d)))
    W, B = torch.tensor(w), torch.tensor(b)
    J = ac.jvp_frame(x, dX, W, B, 1e-5)
    h = 1e-6
    fd = (ac.frame_t(x + h * dX, W, B, 1e-5) - ac.frame_t(x - h * dX, W, B, 1e-5)) / (2 * h)
    assert float((J - fd).abs().max()) < 1e-7


def test_parts_add_up_to_the_chord_and_the_block():
    torch = _torch()
    rng = np.random.default_rng(1)
    n, d, P, k = 20, 32, 5, 4
    w, b = _ln(d, rng)
    x = torch.tensor(rng.standard_normal((n, d)) * 2)
    parts = torch.tensor(rng.standard_normal((P, n, d)) * 0.5)
    r = ac.read_block(x, parts, k, w, b, 1e-5, 3.5)
    assert r["g0"] < 1e-9
    assert abs(r["proj"][:k].sum() - r["total"]["proj"]) < 1e-9
    assert abs(r["off"][:k].sum() - 1.0) < 1e-12
    # linear: a part split in two reads as the sum of its halves
    p2 = torch.cat([parts[:1] * 0.3, parts[:1] * 0.7, parts[1:]], 0)
    r2 = ac.read_block(x, p2, k + 1, w, b, 1e-5, 3.5)
    for key in ("proj", "proj_df", "off"):
        assert abs(r2[key][0] + r2[key][1] - r[key][0]) < 1e-10


def test_field_moves_read_their_sign():
    torch = _torch()
    rng = np.random.default_rng(2)
    n, d = 30, 24
    w, b = np.ones(d), np.zeros(d)
    x = torch.tensor(rng.standard_normal((n, d)))
    U = ac.frame_t(x, torch.tensor(w), torch.tensor(b), 0.0)
    F = torch.zeros_like(U)
    F[1:] = ac.field_t(U[1:], 3.5)
    k = (x - x.mean(-1, keepdim=True)).norm(dim=-1, keepdim=True)
    parts = torch.stack([0.01 * k * F, -0.003 * k * F])
    r = ac.read_block(x, parts, 2, w, b, 0.0, 3.5)
    assert r["proj"][0] > 0 > r["proj"][1]
    assert abs(r["proj"][0] / r["proj"][1] + 0.01 / 0.003) < 0.05


def test_mass_set_and_top_share():
    v = np.array([-5.0, -3.0, -1.0, -1.0, 2.0, 0.0])
    assert list(ac.mass_set(v, -1)) == [0]           # 5 of 10 is the 50 % cut
    assert list(ac.mass_set(v, +1)) == [4]
    assert ac.mass_set(np.array([1.0]), -1).size == 0
    assert ac.top_share(v, -1, k=2) == pytest.approx(0.8)
    assert ac.gini(np.ones(4)) == pytest.approx(0.0)


def test_off_global_and_set_stats():
    rng = np.random.default_rng(3)
    k = len(ac.COMPS)
    rows = {}
    for L in range(24):
        off = rng.random(k)
        rows[L] = {"off": list(off / off.sum()) + [0.0] * 4, "total": {"off_sq": float(rng.random() + 0.1)}}
    g = ac.off_global(rows)
    assert g.shape == (24 * k,) and abs(g.sum() - 1.0) < 1e-12
    proj = rng.standard_normal(24 * k)
    st = ac.set_stats(proj, g)
    assert st["n_negative"] == int((proj < 0).sum())
    sm = np.array(st["S_minus_idx"])
    assert np.all(proj[sm] < 0) and proj[sm].sum() <= 0.5 * proj[proj < 0].sum() + 1e-12
    assert st["S_minus"] == ac.mass_set(proj, -1).size and st["S_zero"] == ac.mass_set(g, +1).size
    assert ac.comp_name(0) == "L0.h0" and ac.comp_name(k * 23 + k - 1) == "L23.mlp"


def test_top_key_finds_a_fixed_key_not_the_last_ones():
    """#215 finding 1: the ratio picked the last keys; mass minus uniform's share picks a planted
    fixed key, and a local (previous-token) head's excess stays ≈ 1."""
    torch = _torch()
    n, j0 = 200, 37
    i = torch.arange(n)
    causal = (i[None, :] <= i[:, None]).double()
    unif = causal / causal.sum(1, keepdim=True)
    sink = 0.7 * unif.clone()
    sink[j0:, j0] += 0.3                                   # queries after j0 put 0.3 on key j0
    local = torch.zeros(n, n, dtype=torch.float64)
    local[0, 0] = 1.0
    local[i[1:], i[1:] - 1] = 1.0                          # each query on the previous token
    W = torch.stack([sink, local, unif])
    j, recv, u = ac.top_key(W)
    assert int(j[0]) == j0
    exc = recv[torch.arange(3), j - 1] - u[j]
    assert float(exc[0]) > 0.25 * (n - j0)
    assert float(exc[1]) < 1.0
    ratio = (recv / u[1:]).argmax(1) + 1                   # the old selector, for the record
    assert int(ratio[1]) > n - 10
