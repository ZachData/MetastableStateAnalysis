"""OV1's cut on a tiny random GPT-NeoX in float64 (`p10_cluster_function/design-10.md` "OV1";
`tools/run/p10_ov1_cut.py`): each written head's folded map is the target ``S'`` and its constant
is kept; the block's attention output is ``Σ_j P_ij S' x̂_j + c`` summed over heads; restore gives
the weights back bit for bit.

Needs the real torch/transformers: ``SMOKE_REAL_DEPS=1 pytest -m smoke``.
"""
import numpy as np
import pytest

from tools.run import p10_ov1_cut as ov

pytestmark = pytest.mark.smoke


@pytest.fixture
def torch64():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)                  # transformers' causal mask, in float64
    yield torch
    torch.set_default_dtype(old)


def tiny_model(torch, seed=0):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                        intermediate_size=64, max_position_embeddings=32, rotary_pct=0.25,
                        use_parallel_residual=True)
    cfg._attn_implementation = "eager"                      # attention weights out (as core.models)
    m = GPTNeoXForCausalLM(cfg).double().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))
        for lay in m.gpt_neox.layers:                       # γ of both sizes, away from 0
            lay.input_layernorm.weight.copy_(0.5 + torch.rand_like(lay.input_layernorm.weight))
    assert len(m.gpt_neox.layers) == 2                      # a real model, not a stub
    return m


@pytest.mark.parametrize("arm", ["att", "rep", "ctl+3", "ctl-7"])
def test_written_heads_are_the_target(torch64, arm):
    m = tiny_model(torch64)
    cut = ov.Cutter(m)
    info = cut.apply(arm, step=0)
    assert info["readback"] < 1e-6 and info["pairs"] > 0           # Gram route: ~sqrt(eps) floor
    for l, (lay, row) in enumerate(zip(cut.layers, cut.heads)):
        g, b = cut.ln[l]
        k, H = cut.k, cut.H
        qkv = lay.attention.query_key_value.weight.detach().numpy().reshape(H, 3, k, -1)
        bv = lay.attention.query_key_value.bias.detach().numpy().reshape(H, 3, k)[:, 2]
        ow = lay.attention.dense.weight.detach().numpy()
        c_sum = np.zeros(ow.shape[0])
        for h, hd in enumerate(row):
            WO, WV = ow[:, h * k:(h + 1) * k], qkv[h, 2]
            idx, lam = ov.choose(hd["lam"], arm, f"0|{arm}|{l}|{h}")
            target = hd["U"][:, idx] @ np.diag(lam) @ hd["U"][:, idx].T
            assert np.allclose(WO @ WV @ np.diag(g), target, atol=1e-10)
            assert np.allclose(WO @ (WV @ b + bv[h]), 0, atol=1e-10)
            if arm == "att":
                assert np.linalg.eigvalsh(target).min() > -1e-10
            if arm == "rep":
                assert np.linalg.eigvalsh(target).max() < 1e-10
            c_sum += hd["c"]
        assert np.allclose(lay.attention.dense.bias.detach().numpy(),
                           cut.orig[l]["o_b"].numpy() + c_sum, atol=1e-12)


def test_attention_output_is_the_cut_map(torch64):
    torch = torch64
    m = tiny_model(torch, 2)
    cut = ov.Cutter(m)
    cut.apply("att", step=0)
    lay = m.gpt_neox.layers[0]
    got = {}
    def keep(mod, i, o):                                    # returns None: the output is unchanged
        got.setdefault("o", o[0].detach())

    h = lay.attention.register_forward_hook(keep)
    ids = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        out = m(ids, output_attentions=True, output_hidden_states=True)
    h.remove()
    x = out.hidden_states[0][0].numpy()
    eps = lay.input_layernorm.eps
    xh = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + eps)
    P = out.attentions[0][0].numpy()                         # (H, n, n)
    want = np.zeros_like(x) + cut.orig[0]["o_b"].numpy()
    for hh, hd in enumerate(cut.heads[0]):
        idx, lam = ov.choose(hd["lam"], "att", f"0|att|0|{hh}")
        S = hd["U"][:, idx] @ np.diag(lam) @ hd["U"][:, idx].T
        want += P[hh] @ xh @ S.T + hd["c"]
    assert np.allclose(got["o"][0].numpy(), want, atol=1e-9)


def test_restore_is_exact(torch64):
    torch = torch64
    m = tiny_model(torch, 4)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    cut = ov.Cutter(m)
    cut.apply("rep", step=0)
    cut.apply("base", step=0)
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), k
