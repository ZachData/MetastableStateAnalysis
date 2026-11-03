"""T5 U4s (`trackT_timaeus/t5g.py`): every cell builds as named at P's scales, and the split's MLP at
gated SiLU 1024 equals Llama's MLP on the same weights.
    SMOKE_REAL_DEPS=1 python -m pytest tests/test_trackT_t5g_smoke.py -m smoke
"""
import pytest

pytestmark = pytest.mark.smoke
from trackT_timaeus import t5, t5g


def _dodeca():
    from transformers import LlamaConfig
    return LlamaConfig(vocab_size=16384, intermediate_size=1024, rms_norm_eps=1e-5, rope_theta=10000.0)


@pytest.mark.parametrize("name", ["ln+par+mha", "gelu+rot+tok", "gate", "act", "width", "gate+act+width",
                                  "rev-gelu", "rev-mha", "rev-all6"])
def test_cells_build_as_named(name):
    pytest.importorskip("torch")
    parts = t5g.parts_of(name)
    m = t5.build("32m", "llama", "P", False, 1, _dodeca(), "cpu", cfg_edit=t5g.cfg_edit(parts),
                 model_edit=t5g.model_edit(parts))
    t5g.check_cell(m, parts)
    sds = t5.class_sds(m)
    for k in t5g.ROLES:
        assert abs(sds[k] / t5.init_sd(k, "P", 384, 16) - 1) < 0.02, (name, k)
    o = m(input_ids=__import__("torch").arange(1, 12)[None], output_hidden_states=True)
    assert len(o.hidden_states) == 17


def test_split_mlp_is_llamas_at_gated_silu():
    torch = pytest.importorskip("torch")
    from transformers.models.llama.modeling_llama import LlamaMLP
    cfg = _dodeca()
    cfg.hidden_size = 384
    ref = LlamaMLP(cfg).double()
    s = t5g._split_mlp(384, 1024, True, "silu").double()
    s.load_state_dict(ref.state_dict())
    x = torch.randn(3, 7, 384, dtype=torch.float64)
    assert float((s(x) - ref(x)).abs().max()) < 1e-12
    u = t5g._split_mlp(384, 1536, False, "gelu")
    assert set(u.state_dict()) == {"up_proj.weight", "down_proj.weight"}
