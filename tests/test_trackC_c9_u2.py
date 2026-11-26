"""C9 U2a (`trackC_writeup/c9_u2.py`): the grid-parametrised locator and gate equal A3's on
Pythia's grid, the plant's width, the turn rows' gaps, and the OV cores."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackA_force import a3
from trackC_writeup import c9_u2 as c


def test_pythia_axis_is_a3s():
    assert c.PYTHIA.win == a3.WIN
    assert np.array_equal(c.PYTHIA.X, a3.X)
    assert c.HARNESS.X[0] == c.HARNESS.X[1] - 2 and c.GPT2.win == tuple(range(35))


@pytest.mark.parametrize("seed", range(20))
def test_reference_and_place_equal_a3_on_pythia(seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, 18))
    for d in ("up", "down"):
        for kind in ("half", "turn"):
            for sig in (None, 0.1, 1.0, 5.0):
                got = c.place(c.PYTHIA, x, d, sig, kind, floor=None if sig is None else 4.0)
                want = a3.place(x, d, sig, kind, floor=None if sig is None else 4.0)
                assert got == want


def test_shape_equals_a3_on_one_octave_gaps_and_scales_with_the_gap():
    for i in a3.WIN[1:-1]:
        oct_ = c.PYTHIA.X[i + 1] - c.PYTHIA.X[i]
        if oct_ == 1.0:
            assert np.array_equal(c.shape(c.PYTHIA, i), a3._shape(i))
        else:                                                   # 512–1000
            assert c.PYTHIA.steps[i] == 512 and not np.array_equal(c.shape(c.PYTHIA, i), a3._shape(i))
    g = c.GPT2
    i = g.steps.index(450)
    sh = c.shape(g, i)
    assert sh[i] < 0.5 * sh[-1] < sh[i + 1]                     # half-point inside the gap
    j = a3.STEPS.index(64)                                      # A3's jump across a one-octave gap
    assert abs((sh[i + 1] - sh[i]) / sh[-1] - (a3._shape(j)[j + 1] - a3._shape(j)[j])) < 1e-3


def test_turn_rows_skip_a_last_gap_with_no_step_past_it():
    assert c.gaps(c.PYTHIA, "turn") == a3.WIN[1:-1]
    assert c.gaps(c.GPT2, "half")[-1] == 33 and c.gaps(c.GPT2, "turn")[-1] == 32


def test_gate_row_equals_a3_on_pythia(monkeypatch):
    monkeypatch.setattr(a3, "N_GATE", 40)
    monkeypatch.setattr(c, "N_GATE", 40)
    rng = np.random.default_rng(3)
    units = rng.normal(0, 0.05, (12, 18, 1)) + np.linspace(0, 1, 18)[None, :, None]
    monkeypatch.setitem(a3.ROWS, "T", ("test", "up", "mean", "units"))
    want = a3.gate_row("T", units, np.random.default_rng(7))
    got = c.gate_row(c.PYTHIA, "up", "mean", "half", units, np.random.default_rng(7))
    assert (got["floor"], got["own_floor"], got["null_no_change"]) == \
        (want["floor"], want["own_floor"], want["null_no_change"])
    for g, w in zip(got["effect"], want["effect"]):
        assert g["gap"] == w["gap"]
        if g["gap"] != "512-1000":
            assert g["placed_right"] == w["placed_right"] and g["curve"] == w["curve"]


def test_attractive_fraction_equals_head_spectrum():
    from p2b_imaginary.head_circuits import head_spectrum
    rng = np.random.default_rng(0)
    W_O, W_V = rng.normal(size=(32, 8)), rng.normal(size=(8, 32))
    assert abs(c.attractive_fraction(W_V @ W_O)
               - head_spectrum(W_O, W_V)["attractive_energy_fraction_core"]) < 1e-12


class _T:
    """A weight with the torch duck type the code reads (.detach().cpu().double().numpy())."""
    def __init__(self, a):
        self.a = np.asarray(a, np.float64)
        self.shape = self.a.shape
    def detach(self):
        return self
    cpu = double = detach
    def numpy(self):
        return self.a


def _ns(**k):
    from types import SimpleNamespace
    return SimpleNamespace(**k)


def _dense_eigs(V_h, O_h):
    """Nonzero eigenvalues of the head's d × d OV (column convention O_h V_h)."""
    e = np.linalg.eigvals(O_h @ V_h)
    return np.sort_complex(e[np.argsort(-np.abs(e))][:V_h.shape[0]])


def test_head_cores_gpt2_conv1d_layout():
    rng = np.random.default_rng(1)
    d, H = 12, 3
    dh = d // H
    W, P = rng.normal(size=(d, 3 * d)), rng.normal(size=(d, d))
    layer = _ns(attn=_ns(c_attn=_ns(weight=_T(W)), c_proj=_ns(weight=_T(P))))
    for h, core in enumerate(c.head_cores(layer, "gpt2", H)):
        V_h = W[:, 2 * d + h * dh:2 * d + (h + 1) * dh].T              # (dh, d): x ↦ V_h x
        O_h = P[h * dh:(h + 1) * dh].T                                  # (d, dh)
        assert np.allclose(np.sort_complex(np.linalg.eigvals(core)), _dense_eigs(V_h, O_h))


def test_head_cores_neox_interleaved_layout():
    rng = np.random.default_rng(2)
    d, H = 12, 3
    dh = d // H
    Vs = [rng.normal(size=(dh, d)) for _ in range(H)]
    blocks = []
    for h in range(H):                                                  # [Q_h | K_h | V_h] per head
        blocks += [rng.normal(size=(dh, d)), rng.normal(size=(dh, d)), Vs[h]]
    QKV, O = np.concatenate(blocks), rng.normal(size=(d, d))
    attn = _ns(query_key_value=_ns(weight=_T(QKV)), dense=_ns(weight=_T(O)),
               num_attention_heads=H, head_size=dh)
    for h, core in enumerate(c.head_cores(_ns(attention=attn), "neox", H)):
        O_h = O[:, h * dh:(h + 1) * dh]
        assert np.allclose(np.sort_complex(np.linalg.eigvals(core)), _dense_eigs(Vs[h], O_h))
