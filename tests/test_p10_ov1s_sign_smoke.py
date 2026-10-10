"""OV1s's arms on a tiny random GPT-NeoX in float64 (`p10_cluster_function/design-10.md` "OV1s";
`tools/run/p10_ov1s_sign.py`): ``neg`` writes ``−S₊``; a hook arm's block attention output is
``Σ_j P_ij (K + Δ) x̂_j + c`` summed over heads; ``att`` by hook equals ``att`` in the weights; closing
the hook gives base back exactly.

Needs the real torch/transformers: ``SMOKE_REAL_DEPS=1 pytest -m smoke``.
"""
import numpy as np
import pytest

from tests.test_p10_ov1_cut_smoke import tiny_model, torch64  # noqa: F401  (the fixture)

pytestmark = pytest.mark.smoke


@pytest.fixture
def sg():
    from tools.run import p10_ov1s_sign                     # at use: other tiers block its deps
    return p10_ov1s_sign


def hooked(model):
    from p1e_energy_field import u2_attn as ua
    return ua.Hooked(model)


def hs_of(torch, hk, ids):
    """Every hidden state in float64 (``Hooked.run`` returns float32)."""
    with torch.no_grad():
        out = hk.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return torch.stack([h[0] for h in out.hidden_states]).numpy()


def test_neg_writes_minus_s_plus(torch64, sg):
    m = tiny_model(torch64)
    cut = sg.SignCutter(m)
    info = cut.apply_weights("neg", step=0)
    assert info["readback"] < 1e-6 and info["pairs"] > 0
    for l, (lay, row) in enumerate(zip(cut.layers, cut.heads)):
        g, _ = cut.ln[l]
        k, H = cut.k, cut.H
        qkv = lay.attention.query_key_value.weight.detach().numpy().reshape(H, 3, k, -1)
        ow = lay.attention.dense.weight.detach().numpy()
        for h, hd in enumerate(row):
            lam, U = hd["lam"], hd["U"]
            Sp = U[:, lam > 0] @ np.diag(lam[lam > 0]) @ U[:, lam > 0].T
            assert np.allclose(ow[:, h * k:(h + 1) * k] @ qkv[h, 2] @ np.diag(g), -Sp, atol=1e-10)


@pytest.mark.parametrize("arm", ["norep", "noatt"])
def test_hook_output_is_the_map(torch64, sg, arm):
    torch = torch64
    m = tiny_model(torch, 2)
    cut = sg.SignCutter(m)
    hk = hooked(m)
    hook = sg.DeltaHook(hk, cut.factors(arm))
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
    want = np.zeros_like(x) + cut.orig[0]["o_b"].numpy()
    for hh, hd in enumerate(cut.heads[0]):
        lam, U = hd["lam"], hd["U"]
        K = ow[:, hh * k:(hh + 1) * k] @ qkv[hh, 2] @ np.diag(g)
        Sp = U[:, lam > 0] @ np.diag(lam[lam > 0]) @ U[:, lam > 0].T
        Sm = -U[:, lam < 0] @ np.diag(lam[lam < 0]) @ U[:, lam < 0].T
        Kp = K + Sm if arm == "norep" else K - Sp
        want += P[hh] @ xh @ Kp.T + hd["c"]
    assert np.allclose(got["o"][0].numpy(), want, atol=1e-9)


def test_att_by_hook_is_att_by_weights_and_close_restores(torch64, sg):
    torch = torch64
    m = tiny_model(torch, 4)
    cut = sg.SignCutter(m)
    hk = hooked(m)
    ids = torch.randint(0, 64, (1, 11))
    base = hs_of(torch, hk, ids)
    cut.apply_weights("att", step=0)
    w = hs_of(torch, hk, ids)
    cut.restore()
    hook = sg.DeltaHook(hk, cut.factors("att_hook"))
    hkd = hs_of(torch, hk, ids)
    hook.close()
    again = hs_of(torch, hk, ids)
    hk.close()
    assert np.abs(w - base).max() > 1e-3                     # the arm moved the stream
    assert np.allclose(hkd, w, atol=1e-9)
    assert np.array_equal(again, base)
