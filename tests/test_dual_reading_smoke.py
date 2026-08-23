"""
tests/test_item_completion_pure.py — Pure-logic tests for the finish-all-items pass (no
torch/transformers/network). Three subjects:

  1. core/artifacts.py — the newly registered phase2 / phase2_weights /
     phase5b / phase6 contracts, phase2_weight_path templating, and the
     artifact_path guard against stem-templated filenames.
  2. core/lm_loading.py — resolve_lm_entry with injected fake registries:
     causal accepted, masked refused, random_init refused, unknown raises
     with options listed.
  3. p5_single_mstate_analysis/causal_tests.py — dispatch helpers:
     _use_legacy_albert_path, _locate_blocks, _block_attn_projection on
     dummy module trees.

Written pytest-style; runnable both under pytest and under the manual
runner at the bottom (this sandbox has no pytest).
"""
import sys
import types
import importlib.util
from pathlib import Path

import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure
HERE = Path(__file__).resolve().parent.parent


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, HERE / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Register a stub `core` package so `core.artifacts` / `core.lm_loading`
# import under their real dotted names (lm_loading's deferred
# `from core.config import ...` is never reached by these tests).
core_pkg = types.ModuleType("core")
core_pkg.__path__ = [str(HERE / "core")]
sys.modules.setdefault("core", core_pkg)

artifacts  = _load("core.artifacts",  "core/artifacts.py")
lm_loading = _load("core.lm_loading", "core/lm_loading.py")
causal     = _load("causal_tests",    "p5_single_mstate_analysis/causal_tests.py")


# ---------------------------------------------------------------------------
# 1. artifacts.py
# ---------------------------------------------------------------------------

def test_new_phases_registered():
    for phase in ("phase2", "phase2_session", "phase2_subexperiments",
                  "phase2_weights", "phase5b", "phase6"):
        assert phase in artifacts.REGISTRY, phase


def test_phase2_weights_specs_match_writer_filenames():
    # save_weight_decomposition writes exactly these four stems.
    assert artifacts.get_spec("phase2_weights", "ov_weights").filename  == "ov_weights_{stem}.npz"
    assert artifacts.get_spec("phase2_weights", "ov_decomp").filename   == "ov_decomp_{stem}.npz"
    assert artifacts.get_spec("phase2_weights", "ov_projectors").filename == "ov_projectors_{stem}.npz"
    assert artifacts.get_spec("phase2_weights", "ov_summary").filename  == "ov_summary_{stem}.json"


def test_phase2_weight_path_templates_stem_like_writer():
    # writer: stem = model_name.replace("/", "_")
    p = artifacts.phase2_weight_path("/tmp/w", "ov_weights", "EleutherAI/pythia-1.4b")
    assert p == Path("/tmp/w/ov_weights_EleutherAI_pythia-1.4b.npz")


def test_artifact_path_refuses_templated_filenames():
    try:
        artifacts.artifact_path("/tmp/run", "phase2_weights", "ov_weights")
    except ValueError as e:
        assert "phase2_weight_path" in str(e)
    else:
        raise AssertionError("expected ValueError for {stem}-templated filename")


def test_phase2_subexperiments_cover_run2_header_list():
    expected = {
        "trajectory", "layer_v_events", "head_ov", "decomposed_violations",
        "ffn_subspace", "continuous_correlations", "ov_norm_confound",
        "zone_comparison", "attractive_zone_violations",
    }
    assert set(artifacts.PHASE2_SUBEXPERIMENTS) == expected


def test_phase6_pairs_json_plus_summary():
    for name in ("subspace_build", "head_classify", "qk_decompose", "dissociation"):
        assert artifacts.get_spec("phase6", name).filename == f"{name}.json"
        assert artifacts.get_spec("phase6", f"{name}_summary").filename == f"{name}.summary.txt"


def test_phase5b_matches_run5b_writes():
    for name, fname in (
        ("logit_cache", "logit_cache.npz"), ("fit_summary", "fit_summary.json"),
        ("mh_params", "mh_params.npz"), ("isometry", "isometry.json"),
        ("isometry_mds", "isometry_mds.npz"),
        ("merge_teleportation", "merge_teleportation.json"),
        ("subspace_isometry", "subspace_isometry.json"),
    ):
        assert artifacts.get_spec("phase5b", name).filename == fname


def test_validate_artifact_still_works_on_untemplated(tmp_path=None):
    import tempfile, json
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        (run / "verdict.json").write_text(json.dumps({"anything": 1}))
        r = artifacts.validate_artifact(run, "phase2", "verdict")
        assert r["ok"] is True
        r2 = artifacts.validate_artifact(run, "phase2", "attn_deltas_raw")
        assert r2["ok"] is False and "does not exist" in r2["error"]


# ---------------------------------------------------------------------------
# 2. lm_loading.resolve_lm_entry
# ---------------------------------------------------------------------------

class _GPT2Model:      pass
class _GPTNeoXModel:   pass
class _BertModel:      pass
class _AlbertModel:    pass
# Names must match the real classes' __name__ for the check:
_GPT2Model.__name__    = "GPT2Model"
_GPTNeoXModel.__name__ = "GPTNeoXModel"
_BertModel.__name__    = "BertModel"
_AlbertModel.__name__  = "AlbertModel"

FAKE_CONFIGS = {
    "gpt2-large": {"model_class": _GPT2Model, "pretrained_name": "gpt2-large"},
    "pythia-1.4b-step1000": {"model_class": _GPTNeoXModel,
                             "hf_repo": "EleutherAI/pythia-1.4b",
                             "revision": "step1000", "checkpoint_step": 1000},
    "bert-base-uncased": {"model_class": _BertModel},
    "albert-xlarge-v2":  {"model_class": _AlbertModel},
    "pythia-1.4b-random": {"model_class": _GPTNeoXModel,
                           "hf_repo": "EleutherAI/pythia-1.4b",
                           "revision": "step143000", "random_init": True},
}


def test_resolve_causal_gpt2():
    e = lm_loading.resolve_lm_entry("gpt2-large", model_configs=FAKE_CONFIGS)
    assert e["repo_id"] == "gpt2-large" and e["revision"] is None


def test_resolve_pythia_pins_revision():
    e = lm_loading.resolve_lm_entry("pythia-1.4b-step1000", model_configs=FAKE_CONFIGS)
    assert e["repo_id"] == "EleutherAI/pythia-1.4b"
    assert e["revision"] == "step1000"
    assert e["checkpoint_step"] == 1000


def test_masked_lm_refused():
    for name in ("bert-base-uncased", "albert-xlarge-v2"):
        try:
            lm_loading.resolve_lm_entry(name, model_configs=FAKE_CONFIGS)
        except ValueError as e:
            assert "masked-LM" in str(e)
        else:
            raise AssertionError(f"{name}: expected ValueError")


def test_random_init_refused_with_pointer_to_state_dict_path():
    try:
        lm_loading.resolve_lm_entry("pythia-1.4b-random", model_configs=FAKE_CONFIGS)
    except ValueError as e:
        assert "load_causal_lm_from_state_dict" in str(e)
    else:
        raise AssertionError("expected ValueError for random_init entry")


def test_unknown_model_lists_options():
    try:
        lm_loading.resolve_lm_entry("nope", model_configs=FAKE_CONFIGS)
    except KeyError as e:
        assert "gpt2-large" in str(e)
    else:
        raise AssertionError("expected KeyError")


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
