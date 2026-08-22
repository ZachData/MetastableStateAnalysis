"""
archive/tests/test_p5_causal_dispatch.py — pure-logic tests for
archive/p5_single_mstate_analysis/causal_tests.py's dispatch helpers
(_use_legacy_albert_path, _locate_blocks, _block_attn_projection) on dummy
module trees. No torch/transformers/network.

Split out of tests/test_dual_reading_smoke.py when Phase 5 was archived.
That file's other two subjects (core/artifacts.py, core/lm_loading.py) are
live and stayed there; this was its third.

NOT collected by default (pytest.ini: norecursedirs = archive) and not
maintained. See archive/README.md.
"""
import sys
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # -> archive/


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, HERE / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


causal = _load("causal_tests", "p5_single_mstate_analysis/causal_tests.py")


# ---------------------------------------------------------------------------
# 3. causal_tests dispatch helpers
# ---------------------------------------------------------------------------

class AlbertModelDummy: pass
class GPT2LMHeadModelDummy: pass
AlbertModelDummy.__name__ = "AlbertForMaskedLM"   # any Albert* class
GPT2LMHeadModelDummy.__name__ = "GPT2LMHeadModel"


def test_dispatch_albert_vs_standard():
    assert causal._use_legacy_albert_path(AlbertModelDummy()) is True
    assert causal._use_legacy_albert_path(GPT2LMHeadModelDummy()) is False


def _ns(**kw):
    o = types.SimpleNamespace(**kw)
    return o


def test_locate_blocks_all_four_layouts():
    blocks = ["b0", "b1"]
    # GPT2LMHeadModel: .transformer.h
    m1 = _ns(transformer=_ns(h=blocks))
    # bare GPT2Model: .h
    m2 = _ns(h=blocks)
    # GPTNeoXForCausalLM: .gpt_neox.layers
    m3 = _ns(gpt_neox=_ns(layers=blocks))
    # bare GPTNeoXModel: .layers
    m4 = _ns(layers=blocks)
    for m in (m1, m2, m3, m4):
        assert causal._locate_blocks(m) == blocks
    try:
        causal._locate_blocks(_ns())
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for blockless model")


def test_block_attn_projection_gpt2_and_neox():
    gpt2_block = _ns(attn=_ns(c_proj="PROJ", head_dim=64))
    proj, hd = causal._block_attn_projection(gpt2_block)
    assert proj == "PROJ" and hd == 64

    neox_block = _ns(attention=_ns(dense="DENSE", head_size=128))
    proj, hd = causal._block_attn_projection(neox_block)
    assert proj == "DENSE" and hd == 128

    # NeoX without head_size: falls back to hidden_size // n_heads
    neox2 = _ns(attention=_ns(dense="D2", hidden_size=2048, num_attention_heads=16))
    proj, hd = causal._block_attn_projection(neox2)
    assert proj == "D2" and hd == 128


# ---------------------------------------------------------------------------
# Manual runner (no pytest in this sandbox)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
