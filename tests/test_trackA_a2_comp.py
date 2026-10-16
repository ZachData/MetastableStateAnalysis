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
