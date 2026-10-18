"""B3 unit 2's hooks (`trackB_circuit/b3_cone.py`): the G4 plant writes the value at key j into
query i, and a mean-ablation overwrites it, on a tiny GPT-NeoX.
Needs the real torch/transformers: ``SMOKE_REAL_DEPS=1 pytest -m smoke``."""

import pytest

from trackB_circuit import b3_cone as bc

pytestmark = pytest.mark.smoke


def test_hooks_plant_value_and_mean_ablation_on_tiny_neox():
    torch = pytest.importorskip("torch")
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from core import nudge as nd
    cfg = GPTNeoXConfig(vocab_size=50, hidden_size=bc.N_HEADS * bc.HEAD_DIM, num_hidden_layers=2,
                        num_attention_heads=bc.N_HEADS, intermediate_size=64, max_position_embeddings=64)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    m = GPTNeoXForCausalLM(cfg).eval()
    st = nd.Stack(m)
    x = st.embed(list(range(1, 21)))
    hk = bc.Hooks(st)
    with torch.no_grad():
        ref = st.run(x, 0, keep={2})["resid"][2]
        hk.set()
        assert torch.equal(st.run(x, 0, keep={2})["resid"][2], ref)   # empty hooks change nothing
        z = {}
        cap = st.layers[1].attention.dense.register_forward_pre_hook(lambda mod, a: z.__setitem__("z", a[0]))
        h, i, j = 1 * bc.N_HEADS + 5, 15, 4
        hk.set(plant=(h, i, j))
        x1 = st.run(x, 0, stop=1, keep={1})["resid"][1]
        st.run(x, 0, keep={2})
        blk = st.layers[1]
        qkv = blk.attention.query_key_value(blk.input_layernorm(x1))
        v = qkv.view(1, 20, bc.N_HEADS, 3 * bc.HEAD_DIM)[0, j, 5, 2 * bc.HEAD_DIM:]
        zz = z["z"].view(1, 20, bc.N_HEADS, bc.HEAD_DIM)
        assert torch.allclose(zz[0, i, 5], v, atol=1e-6)
        mu = torch.arange(bc.HEAD_DIM, dtype=torch.float32)
        hk.set(ablate={h: mu}, plant=(h, i, j))                         # ablation overwrites the plant
        st.run(x, 0, keep={2})
        zz = z["z"].view(1, 20, bc.N_HEADS, bc.HEAD_DIM)
        assert torch.equal(zz[0, :, 5], mu.expand(20, -1))
        cap.remove()
    hk.remove()
