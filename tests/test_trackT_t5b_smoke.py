"""T5 U1b (`trackT_timaeus/t5b.py`): the swaps on real transformers. The rotary swap against GPT-NeoX's
own partial rotary (design-T5.md "U1b" gate), and every swap builds as named at P's scales.
    SMOKE_REAL_DEPS=1 python -m pytest tests/test_trackT_t5b_smoke.py -m smoke
"""
import numpy as np
import pytest

pytestmark = pytest.mark.smoke
from trackT_timaeus import t5, t5b


def test_rotary_matches_neox_on_permuted_dims():
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig
    from transformers.models.gpt_neox import modeling_gpt_neox as nx
    from transformers.models.llama import modeling_llama as ll
    hd, n, rot = 64, 37, 16
    g = torch.Generator().manual_seed(0)
    q, k = torch.randn(1, 2, n, hd, generator=g, dtype=torch.float64), torch.randn(1, 2, n, hd, generator=g, dtype=torch.float64)
    pos = torch.arange(n)[None]
    # GPT-NeoX: rotary on the first 16 dims
    re = nx.GPTNeoXRotaryEmbedding(rot, 2048, base=10000)
    cos, sin = re(q, seq_len=n)
    qr, kr = nx.apply_rotary_pos_emb(q[..., :rot], k[..., :rot], cos.double(), sin.double(), pos)
    ref = torch.cat((qr, q[..., rot:]), -1) @ torch.cat((kr, k[..., rot:]), -1).transpose(-1, -2)
    # the patched Llama on the same vectors, dimensions permuted
    cfg = LlamaConfig(hidden_size=hd * 2, num_attention_heads=2, num_key_value_heads=2, rope_theta=10000.0)
    lre = ll.LlamaRotaryEmbedding(config=cfg)
    lre.register_buffer("inv_freq", torch.tensor(t5b.partial_inv_freq(hd), dtype=torch.float32), persistent=False)
    lc, ls = lre(q.float(), pos)
    src = torch.tensor(t5b.neox_to_llama_dims(hd))
    ql, kl = ll.apply_rotary_pos_emb(q[..., src], k[..., src], lc.double(), ls.double())
    got = ql @ kl.transpose(-1, -2)
    assert float((got - ref).abs().max()) <= 1e-5
    # and full rotary differs, so the test can fail
    lre2 = ll.LlamaRotaryEmbedding(config=cfg)
    fc, fs = lre2(q.float(), pos)
    qf, kf = ll.apply_rotary_pos_emb(q[..., src], k[..., src], fc.double(), fs.double())
    assert float((qf @ kf.transpose(-1, -2) - ref).abs().max()) > 1e-2


@pytest.mark.parametrize("swap", [None, "ln", "par", "gelu", "rot", "mha", "tok", "untied", "all"])
def test_swaps_build_as_named(swap):
    pytest.importorskip("torch")
    from transformers import LlamaConfig
    dodeca = LlamaConfig(vocab_size=16384, intermediate_size=1024, rms_norm_eps=1e-5, rope_theta=10000.0)
    parts = t5b.parts_of(swap)
    m = t5.build("32m", "llama", "P", False, 1, dodeca, "cpu", cfg_edit=t5b.cfg_edit(parts),
                 model_edit=t5b.model_edit(parts))
    t5b.check_swaps(m, parts)
    sds = t5.class_sds(m)
    for k in ("emb", "qkv", "attn_out", "mlp_in", "mlp_out"):
        assert abs(sds[k] / t5.init_sd(k, "P", 384, 16) - 1) < 0.02, (swap, k)
    ids = __import__("torch").arange(1, 12)[None]
    o = m(input_ids=ids, output_hidden_states=True)
    assert len(o.hidden_states) == 17
