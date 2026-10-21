"""S2's hook on a tiny random GPT-NeoX in float64 (`trackS_subspace/design-S2.md` G0): at the base
core it is the model (logits and hidden states); (b-eig) by the hook equals (b-eig) written in W_V
(logits, centred hidden states); arm (a)'s read-back is sym K; close restores the weights.

The hook moves each head's constant into the dense bias, exact when P's rows sum to 1. From 4.5x
transformers takes GPT-NeoX's eager softmax in float32, so in a float64 model they sum to 1 only to
~1e-7 (`LESSONS.md` 4, 2026-10-11): the bars are 1e-10 plus 10× the measured row-sum gap, which is
0 under 4.44 and ~1e-7 under 4.57.

Needs the real torch/transformers: ``SMOKE_REAL_DEPS=1 pytest -m smoke``.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.smoke


@pytest.fixture
def torch64():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield torch
    torch.set_default_dtype(old)


def tiny_model(torch, seed=0):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                        intermediate_size=64, max_position_embeddings=32, rotary_pct=0.25,
                        use_parallel_residual=True)
    cfg._attn_implementation = "eager"
    m = GPTNeoXForCausalLM(cfg).double().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return m


def _run(model, ids):
    o = model(ids, output_hidden_states=True, output_attentions=True)
    return o.logits, o.hidden_states


def _rowsum_gap(model, ids):
    o = model(ids, output_attentions=True)
    return max(float((a.sum(-1) - 1).abs().max()) for a in o.attentions)


def test_hook_at_base_is_the_model_and_close_restores(torch64):
    from trackS_subspace import s2
    torch = torch64
    m = tiny_model(torch)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    ids = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        L0, H0 = _run(m, ids)
        bar = 1e-10 + 10 * _rowsum_gap(m, ids)
        heads = s2.Heads(m)
        hook = s2.Hook(m, heads)
        L1, H1 = _run(m, ids)
        hook.close()
    assert s2.rel_err(L1, L0) < bar
    assert max(s2.rel_err(a, b) for a, b in zip(H1, H0)) < bar
    assert all(torch.equal(p, before[n]) for n, p in m.named_parameters())


def test_b_eig_by_hook_equals_written(torch64):
    from trackS_subspace import s2
    torch = torch64
    m = tiny_model(torch, 1)
    ids = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        bar = 1e-9 + 10 * _rowsum_gap(m, ids)
        heads = s2.Heads(m)
        hook = s2.Hook(m, heads)
        hook.set(heads.cores("b_eig")[0])
        L1, H1 = _run(m, ids)
        hook.close()
        saved = s2.write_b_eig(m, heads)
        L2, H2 = _run(m, ids)
        s2.restore_v(m, heads, saved)
    assert s2.rel_err(L1, L2) < bar
    assert max(s2.rel_err(s2.centred(a), s2.centred(b)) for a, b in zip(H1, H2)) < bar


def test_arm_a_read_back_is_sym_k(torch64):
    from trackS_subspace import s2
    torch = torch64
    m = tiny_model(torch, 2)
    heads = s2.Heads(m)
    hook = s2.Hook(m, heads)
    hook.set(heads.cores("a")[0])
    Kd = s2.dense_K(heads, 1, 2)
    assert np.allclose(s2.readback(hook, 1, 2), (Kd + Kd.T) / 2, atol=1e-10)
    hook.close()


def test_hook_refuses_without_attention_weights(torch64):
    """An attention output with no (b, H, n, n) weights in it (an sdpa model) is refused. Called
    directly: under 4.57 eager returns the weights even when not asked."""
    from trackS_subspace import s2
    torch = torch64
    m = tiny_model(torch, 3)
    hook = s2.Hook(m, s2.Heads(m))
    lay = m.gpt_neox.layers[0]
    hook._x[0] = torch.zeros(1, 5, 32)
    with pytest.raises(s2.S2Error):
        hook._post(0, lay.input_layernorm)(lay.attention, (), (torch.zeros(1, 5, 32), None))
    hook.close()
