"""OV1d's arms on a tiny random GPT-NeoX in float64 (`p10_cluster_function/design-10.md` "OV1d";
`tools/run/p10_ov1d_dose.py`): ``w`` writes ``t·S₊`` (t = 0.5, −2, 0); a ``z`` arm's block attention
output is ``Σ_j P_ij K x̂_j + Σ_{j≥1} P_ij (t·S₊ − K) x̂_j + c`` summed over heads; with column 0 kept
the hook equals ``w`` in the weights; closing it gives base back exactly.

Needs the real torch/transformers: ``SMOKE_REAL_DEPS=1 pytest -m smoke``.
"""
import numpy as np
import pytest

from tests.test_p10_ov1_cut_smoke import tiny_model, torch64  # noqa: F401  (the fixture)
from tests.test_p10_ov1s_sign_smoke import hooked, hs_of

pytestmark = pytest.mark.smoke


@pytest.fixture
def dd():
    from tools.run import p10_ov1d_dose                     # at use: other tiers block its deps
    return p10_ov1d_dose


@pytest.mark.parametrize("arm", ["w+0.5", "w-2", "w+0"])
def test_w_writes_t_s_plus(torch64, dd, arm):
    m = tiny_model(torch64)
    cut = dd.DoseCutter(m)
    info = cut.apply_weights(arm, step=0)
    assert info["readback"] < 1e-6
    assert (info["pairs"] == 0) == (arm == "w+0")
    t = dd.T[arm]
    for l, (lay, row) in enumerate(zip(cut.layers, cut.heads)):
        g, _ = cut.ln[l]
        k, H = cut.k, cut.H
        qkv = lay.attention.query_key_value.weight.detach().numpy().reshape(H, 3, k, -1)
        ow = lay.attention.dense.weight.detach().numpy()
        for h, hd in enumerate(row):
            lam, U = hd["lam"], hd["U"]
            Sp = U[:, lam > 0] @ np.diag(lam[lam > 0]) @ U[:, lam > 0].T
            assert np.allclose(ow[:, h * k:(h + 1) * k] @ qkv[h, 2] @ np.diag(g), t * Sp, atol=1e-10)


@pytest.mark.parametrize("arm", ["z+1", "z-1", "z+0"])
def test_z_output_is_the_map_with_key_zero_at_base(torch64, dd, arm):
    torch = torch64
    m = tiny_model(torch, 2)
    cut = dd.DoseCutter(m)
    hk = hooked(m)
    hook = dd.SinkHook(hk, cut.factors(arm), shut0=True)
    lay = m.gpt_neox.layers[0]
    got = {}

    def keep(mod, i, o):                                    # returns None: the output is unchanged
        got.setdefault("o", o[0].detach())

    h = lay.attention.register_forward_hook(keep)
    ids = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        out = m(ids, output_attentions=True, output_hidden_states=True)
    h.remove()
    hook.close()
    hk.close()
    x = out.hidden_states[0][0].numpy()
    eps = lay.input_layernorm.eps
    xh = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + eps)
    P = out.attentions[0][0].numpy()
    g, _ = cut.ln[0]
    k, H = cut.k, cut.H
    qkv = cut.orig[0]["qkv_w"].numpy().reshape(H, 3, k, -1)
    ow = cut.orig[0]["o_w"].numpy()
    t = dd.T[arm]
    want = np.zeros_like(x) + cut.orig[0]["o_b"].numpy()
    for hh, hd in enumerate(cut.heads[0]):
        lam, U = hd["lam"], hd["U"]
        K = ow[:, hh * k:(hh + 1) * k] @ qkv[hh, 2] @ np.diag(g)
        Sp = U[:, lam > 0] @ np.diag(lam[lam > 0]) @ U[:, lam > 0].T
        P1 = P[hh].copy()
        P1[:, 0] = 0
        want += P[hh] @ xh @ K.T + P1 @ xh @ (t * Sp - K).T + hd["c"]
    assert np.allclose(got["o"][0].numpy(), want, atol=1e-9)
    assert np.allclose(got["o"][0, 0].numpy(), want[0])            # position 0's own row: base's map


def test_hook_with_key_zero_kept_is_w_and_close_restores(torch64, dd):
    torch = torch64
    m = tiny_model(torch, 4)
    cut = dd.DoseCutter(m)
    hk = hooked(m)
    ids = torch.randint(0, 64, (1, 11))
    base = hs_of(torch, hk, ids)
    cut.apply_weights("w+1", step=0)
    w = hs_of(torch, hk, ids)
    cut.restore()
    hook = dd.SinkHook(hk, cut.factors(dd.CHECK), shut0=False)
    hkd = hs_of(torch, hk, ids)
    hook.close()
    shut = dd.SinkHook(hk, cut.factors("z+1"), shut0=True)
    z = hs_of(torch, hk, ids)
    shut.close()
    again = hs_of(torch, hk, ids)
    hk.close()
    assert np.abs(w - base).max() > 1e-3                     # the arm moved the stream
    assert np.allclose(hkd, w, atol=1e-9)
    assert np.abs(z - w).max() > 1e-3                         # shutting key 0 changed it
    assert np.allclose(z[:, 0], base[:, 0], atol=1e-9)        # position 0 stays at base all the way up
    assert np.array_equal(again, base)
