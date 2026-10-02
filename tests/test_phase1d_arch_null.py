"""
tests/test_phase1d_arch_null.py — unit 2's runner (`p1d_cluster_ensemble/arch_null.py`):
Pythia's re-init on a tiny config, the token rules, one cloud record on
planted structure, and the first check on synthetic z tables with known answers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.deps

from p1d_cluster_ensemble import arch_null as an


def _real_transformers():
    """`tests/conftest.py` stubs transformers unless SMOKE_REAL_DEPS=1; the re-init needs the real one."""
    import sys
    from unittest.mock import MagicMock
    if isinstance(sys.modules.get("transformers"), MagicMock):
        pytest.skip("needs the real transformers (SMOKE_REAL_DEPS=1)")


def _tiny_config():
    _real_transformers()
    from transformers import GPTNeoXConfig
    return GPTNeoXConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
                         intermediate_size=512, vocab_size=512, rotary_pct=0.25,
                         use_parallel_residual=True, tie_word_embeddings=False)


def test_sigmas_are_pythia_410ms():
    s = an.sigmas(1024, 24)
    assert s["small"] == pytest.approx(0.019764, abs=1e-6)   # lit-1d §10 row 2a
    assert s["wang"] == pytest.approx(0.002604, abs=1e-6)


def test_reinit_draws_every_parameter_at_its_pythia_sigma():
    cfg = _tiny_config()
    m = an.reinit_model(cfg, 3)
    sig = an.sigmas(cfg.hidden_size, cfg.num_hidden_layers)
    seen = set()
    for name, p in m.named_parameters():
        c = an.init_class(name)
        seen.add(c)
        if c == "zero":
            assert float(p.abs().max()) == 0.0, name
        elif c == "one":
            assert float((p - 1).abs().max()) == 0.0, name
        else:
            assert float(p.std()) == pytest.approx(sig[c], rel=5 / math.sqrt(2 * p.numel())), name
    assert seen == {"zero", "one", "small", "wang"}
    # Every drawn weight is a float16 value, as every real init's is (#131 finding 3).
    w0 = dict(m.named_parameters())["gpt_neox.layers.0.attention.dense.weight"]
    assert bool((w0.half().float() == w0).all())
    # Same seed, same weights; another seed, others.
    a = dict(an.reinit_model(cfg, 3).named_parameters())
    b = dict(an.reinit_model(cfg, 4).named_parameters())
    w = "gpt_neox.layers.0.attention.dense.weight"
    assert bool((a[w] == dict(m.named_parameters())[w]).all()) and not bool((a[w] == b[w]).all())


def test_init_class_refuses_an_unknown_parameter():
    with pytest.raises(ValueError, match="no Pythia init rule"):
        an.init_class("gpt_neox.layers.0.attention.new_gate.weight")


def test_token_sets_take_the_union_over_models_and_drop_position_0():
    toks = {"p": ["<s>", "a", "b", "a", "c", "\n", "d"]}
    tabs = {"init:0": {"p": {5: (30.0, 8)}}, "reinit:1": {"p": {2: (12.0, 4), 5: (40.0, 9)}},
            "reinit:2": {}}
    s = an.token_sets(toks, tabs)["p"]
    assert s["kept"] == [1, 4, 6]   # 0 (T1), 2 and 5 (T2 union), 3 (T3)
    assert [m["position"] for m in s["massive"]] == [2, 5]
    assert s["massive"][1]["ratio"] == 40.0 and s["massive"][1]["model"] == "reinit:1"


def _planted(n_per=12, k=4, d=40, noise=0.08, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, d))
    Y = np.vstack([c + noise * rng.standard_normal((n_per, d)) for c in centres])
    return Y + 0.01 * rng.standard_normal(Y.shape)


def test_cloud_record_sees_planted_groups_beyond_their_gaussian():
    rec = an.cloud_record(_planted(), "raw", 20, [0, 1, 2, 3])
    assert set(rec["stats"]) == set(an.STATS)
    assert rec["stats"]["nn1"]["z"] < -3        # nearer neighbours than the Gaussian's
    g = rec["arms"]["4"]["groups"]
    assert len(g) == 4 and all(x["s"] > 1 for x in g)
    assert sorted(sum((x["members"] for x in g), [])) == list(range(48))


def test_cloud_record_on_a_gaussian_cloud_admits_little():
    rng = np.random.default_rng(1)
    rec = an.cloud_record(rng.standard_normal((60, 30)), "centred", 20, [5])
    assert abs(rec["stats"]["ci2"]["z"]) < 4
    assert sum(x["s"] > 1 for x in rec["arms"]["2"]["groups"]) <= 2


def test_z_is_none_when_the_draws_do_not_vary():
    assert an.z_of(3.0, np.array([2.0, 2.0, 2.0])) is None
    assert an.z_of(3.0, np.array([1.0, 3.0])) == pytest.approx(0.7071, abs=1e-3)


def test_mid_rank_counts_ties_half():
    assert an.mid_rank(2.0, np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.375)


def _recs(shift=0.0, n_re=40, missing=None, seed=0):
    """Synthetic records: every z ~ N(0, 1), the real inits' shifted by ``shift``."""
    rng = np.random.default_rng(seed)
    recs = []
    for mid in an.model_ids("init")[:10] + [f"reinit:{i}" for i in range(n_re)]:
        sh = shift if mid.startswith("init") else 0.0
        for p in an.V1_PASSAGES:
            lays = []
            for L in an.LAYERS:
                for f in an.FRAMES:
                    st = {s: {"obs": 0.0, "z": float(rng.standard_normal() + sh)} for s in an.STATS}
                    if missing == (mid, p, L, f):
                        st["nn1"]["z"] = None
                    lays.append({"layer": L, "frame": f, "stats": st})
            recs.append({"model": mid, "prompt": p, "layers": lays})
    return recs


def test_first_check_passes_when_the_inits_are_re_inits():
    rows = an.first_check(_recs(), [f"reinit:{i}" for i in range(40)], an.model_ids("init"))
    assert len(rows) == len(an.STATS) * len(an.FRAMES) * len(an.BANDS)
    assert all(r["n"] == 70 and r["verdict"] == "pass" for r in rows)


def test_first_check_fails_a_shifted_init_and_counts_the_side():
    rows = an.first_check(_recs(shift=3.0), [f"reinit:{i}" for i in range(40)], an.model_ids("init"))
    assert all(r["verdict"].startswith("fail") and r["n_high"] > r["n_low"] for r in rows)


def test_first_check_sensitivity_is_between_1_and_1_5_sd():
    """
    What the placed rule can see, with layers independent (`/challenge-pr` on
    #131): every real init shifted by 1 SD of the re-inits' z still passes
    every cell; 1.5 SD fails most. Correlated layers shrink the band median
    less, so real layers can only make it more sensitive than this.
    """
    re = [f"reinit:{i}" for i in range(40)]
    one = an.first_check(_recs(shift=1.0), re, an.model_ids("init"))
    more = an.first_check(_recs(shift=1.5), re, an.model_ids("init"))
    assert all(r["verdict"] == "pass" for r in one)
    assert sum(r["verdict"] == "pass" for r in more) <= 2


def test_power_ranks_held_out_and_real_against_the_same_references():
    recs = _recs(n_re=40)
    pw = an.power(recs, [f"reinit:{i}" for i in range(40)], an.model_ids("init"))
    assert len(pw) == len(an.STATS) * len(an.FRAMES) * len(an.BANDS)
    assert all(p["n_ref"] == 30 for p in pw)
    # Same distribution: per-layer shares agree, near 2 x 2/31 at N = 30.
    for p in pw:
        assert abs(p["layer_share_real_mean"] - p["layer_share_heldout_mean"]) < 0.06
        assert 0.07 < p["layer_share_heldout_mean"] < 0.20
    with pytest.raises(ValueError, match="folds"):
        an.power(_recs(n_re=30), [f"reinit:{i}" for i in range(30)], an.model_ids("init"))


def test_first_check_fails_a_cell_with_a_missing_z():
    miss = ("init:0", an.V1_PASSAGES[0], 10, "raw")
    rows = an.first_check(_recs(missing=miss), [f"reinit:{i}" for i in range(40)], an.model_ids("init"))
    bad = [r for r in rows if r["n_missing"]]
    assert [(r["stat"], r["frame"], r["band"]) for r in bad] == [("nn1", "raw", "L9-16")]
    assert bad[0]["verdict"].startswith("fail")


def test_trained_steps_wait_for_the_first_check():
    with pytest.raises(NotImplementedError):
        an.comparison_models("step143000")
