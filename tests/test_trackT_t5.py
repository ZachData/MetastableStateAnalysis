"""T5 U1, the cloud at init (`trackT_timaeus/t5.py`, design-T5.md, frozen 2026-10-30): pure parts."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackT_timaeus import t5


def test_the_frozen_constants():
    assert (t5.R1_SUPPORTED, t5.R1_FALSIFIED, t5.E_MIN, t5.TAU_MIN) == (0.60, 0.30, 0.1, 0.02)
    assert (t5.SD_TOL, t5.RANGE_TOL, t5.NULL_MAX, t5.COLLAPSE_MIN, t5.ZERO_TOL) == (0.02, 0.03, 0.05, 0.99, 1e-6)
    assert t5.N_SEEDS == 20 and len(t5.factorial_cells()) == 32 and t5.PROMPTS_HASH == "976b62d677bc"
    assert t5.SHAPES["32m"] == (384, 16, 6, 2)


def test_init_scales_match_the_measured():
    # status-T.md "T5's proposal": pythia-70m 0.0280 / 0.0147, pythia-410m 0.0198 / 0.0026
    assert abs(t5.small_init(512) - 0.0280) < 5e-5 and abs(t5.wang_init(512, 6) - 0.0147) < 5e-5
    assert abs(t5.small_init(1024) - 0.0198) < 5e-5 and abs(t5.wang_init(1024, 24) - 0.0026) < 5e-5
    # design-T5.md: the dose r_F / r_P = 0.02 / wang_init
    doses = {s: 0.02 / t5.wang_init(*t5.SHAPES[s][:2]) for s in t5.SHAPES}
    assert [round(doses[s], 2) for s in ("70m", "160m", "410m", "32m")] == [1.36, 3.33, 7.68, 3.14]


def test_weight_classes():
    assert t5.weight_class("gpt_neox.layers.3.attention.dense.weight") == "attn_out"
    assert t5.weight_class("gpt_neox.layers.3.attention.query_key_value.weight") == "qkv"
    assert t5.weight_class("gpt_neox.layers.3.mlp.dense_4h_to_h.weight") == "mlp_out"
    assert t5.weight_class("model.layers.0.mlp.gate_proj.weight") == "mlp_in"
    assert t5.weight_class("model.layers.0.self_attn.o_proj.weight") == "attn_out"
    assert t5.weight_class("model.layers.0.self_attn.o_proj.bias") is None
    assert t5.weight_class("model.norm.weight") is None
    assert t5.init_sd("mlp_out", "P", 512, 6) == t5.wang_init(512, 6) and t5.init_sd("qkv", "F", 512, 6) == 0.02


def test_summaries():
    rho = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]          # 7 blocks: states 0 … 7
    assert abs(t5.rho_mid(rho) - 0.35) < 1e-12 and abs(t5.rho_15(rho) - 0.3) < 1e-12
    assert t5.r1_label(0.6) == "supported" and t5.r1_label(0.3) == "falsified" and t5.r1_label(0.45) == "not told apart"


def test_state_stats_on_a_known_cloud():
    X = np.array([[3.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    s = t5.state_stats(X)
    assert abs(s["rho"] - 1 / 3) < 1e-12 and abs(s["rho_no0"] - 0.0) < 1e-12
    assert abs(s["n0"] - 1.5) < 1e-12
    assert abs(s["m"] - (16 / 9 + 4 / 9) / (14 / 3)) < 1e-12


def _med(e_init, e_bias=0.0, base=0.3, lf=0.8, lp=0.4, np_=0.3):
    med = {}
    for s in t5.SHAPES:
        for a in t5.ARCHS:
            med[f"{s}-{a}-P"] = base
            med[f"{s}-{a}-F"] = base + e_init[s]
            med[f"{s}-{a}"] = base + e_init[s] / 2
            for i in t5.INITS:
                v = med[f"{s}-{a}-{i}"]
                med[t5.cell_key(s, a, i, True)] = v + e_bias / 2
                med[t5.cell_key(s, a, i, False)] = v - e_bias / 2
    med["32m-llama-F"], med["32m-llama-P"], med["32m-neox-P"] = lf, lp, np_
    return med


def test_r2_supported_partly_not():
    big = {"70m": 0.05, "160m": 0.3, "410m": 0.4, "32m": 0.3}
    m = _med(big)
    m["32m-llama-F"], m["32m-llama-P"] = 0.8, 0.5        # E_init at 32m-llama 0.3, halfway 0.6 of 0.5
    r = t5.r2_reading(m)
    assert r["tau"] == 0.02 and r["bar"] == 0.1 and r["label"] == "supported", r["failed"]
    m2 = dict(m)
    m2["32m-llama-P"] = 0.75                             # halfway fails, and E_init 0.05 ≥ 2τ
    r2 = t5.r2_reading(m2)
    assert r2["label"] == "partly" and "halfway (Llama-32m P toward NeoX-32m P)" in r2["failed"]
    m3 = dict(m)
    m3["32m-llama-P"] = 0.79                             # E_init 0.01 < 2τ
    assert t5.r2_reading(m3)["label"] == "not supported"
    r4 = t5.r2_reading(_med(big, e_bias=0.06))           # τ 0.06 → bar 0.12
    assert abs(r4["tau"] - 0.06) < 1e-12 and abs(r4["bar"] - 0.12) < 1e-12
