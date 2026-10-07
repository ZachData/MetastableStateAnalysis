"""
`p1e_energy_field/u2_attn.py` — the hooked split of a block's update on a tiny random GPT-NeoX
(no download): sink + keys + mlpx + bias = the block's update, and ``sink`` / ``keys`` equal the
explicit sums over key 0 and keys 1…i (`design-1e.md` "U2's attention arm: the rule", first checks).
"""
import numpy as np
import pytest

from p1e_energy_field import u2_attn as ua

pytestmark = pytest.mark.deps


def _tiny(parallel=True):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPTNeoXConfig(vocab_size=97, hidden_size=32, num_hidden_layers=3,
                                     num_attention_heads=4, intermediate_size=64, rotary_pct=0.25,
                                     max_position_embeddings=64, use_parallel_residual=parallel,
                                     attn_implementation="eager")
    torch.manual_seed(0)
    m = transformers.GPTNeoXForCausalLM(cfg).eval()
    with torch.no_grad():                       # biases are zero at init: make them count
        for p in m.parameters():
            if p.ndim == 1:
                p.normal_(0, 0.3)
    return m


def test_split_adds_up_and_matches_the_explicit_sum_over_keys():
    torch = pytest.importorskip("torch")
    m = _tiny()
    ids = torch.randint(0, 97, (1, 20))
    hk = ua.Hooked(m)
    hs, comps = hk.run(ids, range(3))
    hk.close()
    with torch.no_grad():
        full = m(input_ids=ids, output_hidden_states=True, output_attentions=True)
    for L in range(3):
        C = comps[L]
        # the parts sum to the block, and the block is the residual step
        np.testing.assert_allclose(sum(C[k] for k in ua.PARTS), C["block"], atol=2e-4)
        if L < 2:     # the last hidden state is after final_layer_norm (block 23's refusal)
            np.testing.assert_allclose(hs[L] + C["block"], hs[L + 1], atol=2e-4)
        # explicit: per head, A[h] @ (V_h - b_V) through W_O, split at key 0
        layer = m.gpt_neox.layers[L]
        att = layer.attention
        H, hd = att.num_attention_heads, att.head_size
        x = layer.input_layernorm(full.hidden_states[L][0])
        qkv = att.query_key_value(x).view(-1, H, 3 * hd)
        bV = att.query_key_value.bias.view(H, 3 * hd)[:, 2 * hd:]
        V = (qkv[:, :, 2 * hd:] - bV).permute(1, 0, 2)                  # (H, n, hd)
        A = full.attentions[L][0]                                          # (H, n, n)
        Wo = att.dense.weight.view(-1, H, hd)                              # (d, H, hd)
        per_key = torch.einsum("hij,hjk,dhk->ijd", A, V, Wo)               # (n, n, d)
        np.testing.assert_allclose(C["sink"], per_key[:, 0].detach().numpy(), atol=2e-4)
        np.testing.assert_allclose(C["keys"], per_key[:, 1:].sum(1).detach().numpy(), atol=2e-4)
        # bias is one vector for every token
        assert np.ptp(C["bias"], axis=0).max() == 0
        # attention to key 0, mean over heads
        np.testing.assert_allclose(C["a0_mean_heads"], A[:, :, 0].mean(0).numpy(), atol=1e-6)
    # hooks removed: the explicit call above captured nothing
    assert not hk.cap


def test_refuses_a_sequential_residual():
    with pytest.raises(ValueError, match="parallel residual"):
        ua.Hooked(_tiny(parallel=False))
