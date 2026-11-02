"""T5 U3's split (`trackT_timaeus/t5f.py`) on tiny random Llama (GQA) and GPT-NeoX models, CPU,
float32: the parts sum to the block's write, the bias paths are exactly the biases' components, a
planted o_proj bias lands in the bias path.
    SMOKE_REAL_DEPS=1 python -m pytest tests/test_trackT_t5f_smoke.py -m smoke
"""
import pytest

pytestmark = pytest.mark.smoke
from trackT_timaeus import t5f


# ---------------------------------------------------------------- the split on tiny models (torch)

def _llama(bias=False):
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = tr.LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, attention_bias=bias,
                         max_position_embeddings=64)
    cfg._attn_implementation = "eager"
    m = tr.LlamaForCausalLM(cfg).eval()
    if bias:
        for L in m.model.layers:
            for lin in (L.self_attn.v_proj, L.self_attn.o_proj):
                torch.nn.init.normal_(lin.bias, std=0.3)
    return m


def _neox():
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    torch.manual_seed(1)
    cfg = tr.GPTNeoXConfig(vocab_size=50, hidden_size=32, intermediate_size=128, num_hidden_layers=3,
                           num_attention_heads=4, rotary_pct=0.25, use_parallel_residual=True,
                           max_position_embeddings=64)
    cfg._attn_implementation = "eager"
    m = tr.GPTNeoXForCausalLM(cfg).eval()
    for L in m.gpt_neox.layers:
        for lin in (L.attention.query_key_value, L.attention.dense, L.mlp.dense_4h_to_h):
            torch.nn.init.normal_(lin.bias, std=0.3)
    return m


def _enc(torch, n=11):
    ids = torch.arange(1, n + 1)[None]
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@pytest.mark.parametrize("make", [_llama, lambda: _llama(bias=True), _neox])
def test_parts_sum_to_write(make):
    torch = pytest.importorskip("torch")
    m = make()
    r = t5f.split_prompt(m, _enc(torch), "cpu")
    assert len(r["blocks"]) == 3
    for b in r["blocks"]:
        assert b["err_attn"] < 1e-5 and b["err_modules"] < 1e-5 and b["err_sum"] < 1e-5
        s = b["sum"]
        assert sum(s[k] for k in ("attn_sink", "attn_bias", "attn_rest", "mlp_bias", "mlp_free")) == pytest.approx(s["write"])


def test_bias_paths_are_the_biases():
    torch = pytest.importorskip("torch")
    m = _neox()
    r = t5f.split_prompt(m, _enc(torch, 9), "cpu")
    for l, b in enumerate(r["blocks"]):
        A, M = m.gpt_neox.layers[l].attention, m.gpt_neox.layers[l].mlp
        Wv, bv, Wo, bo, bm = t5f.head_maps(m, A, M)
        ab = bo + torch.einsum("dhr,hr->d", Wo, bv)
        assert b["sum"]["attn_bias"] == pytest.approx(9 * float(ab @ b["u"]))
        assert b["sum"]["mlp_bias"] == pytest.approx(9 * float(bm @ b["u"]))
        # position 0 attends only to itself: its sink path is all its attention bar the bias
        assert b["sum"]["attn_sink"] - b["sum_no0"]["attn_sink"] == pytest.approx(
            b["sum"]["attn_sink"] + b["sum"]["attn_rest"] - b["sum_no0"]["attn_sink"] - b["sum_no0"]["attn_rest"])


def test_plant_lands_in_bias_path():
    torch = pytest.importorskip("torch")
    m = _llama()
    base = t5f.aggregate([t5f._np(t5f.split_prompt(m, _enc(torch), "cpu")["blocks"])])
    assert base["share"]["attn_bias"] == 0.0 and base["share"]["mlp_bias"] == 0.0
    b = t5f.plant_vector(32, 5.0)
    t5f.plant(m, b)
    with pytest.raises(t5f.T5fRefused):
        t5f.plant(m, t5f.plant_vector(32, 5.0))
    r = t5f.split_prompt(m, _enc(torch), "cpu")
    pl = t5f.aggregate([t5f._np(r["blocks"])])
    assert pl["err_sum"] < 1e-5
    # the direct attribution, which is what the split promises: per block, the bias path is n · b·u
    # exactly, and nothing else carries a constant term (the MLP bias stays 0)
    for blk, c in zip(r["blocks"], pl["per_block"]["component"]["attn_bias"]):
        assert c == pytest.approx(11 * float(torch.as_tensor(b, dtype=blk["u"].dtype) @ blk["u"]), rel=1e-6)
    assert pl["share"]["mlp_bias"] == 0.0
    # and the gate's statistic is computed from the two runs as `plant_share` defines it
    g = t5f.plant_share(pl, base)
    assert g["planted_component"] == pytest.approx(pl["write_total"] - base["write_total"])
    assert g["bias_path"] == pytest.approx(sum(pl["per_block"]["component"]["attn_bias"]))


# ---------------------------------------------------------------- the gate (pure, no torch)


def _rec(err=1e-6, rho=(0.5, 0.6), wt=10.0, ab=0.0):
    return {"err_sum": err, "rho": list(rho), "write_total": wt,
            "per_block": {"component": {"attn_bias": [ab]}}, "share": {"attn_bias": ab / wt}}


def test_gate_planted_bias_beside_not_gating(tmp_path):
    """Blocked 131 (a): the planted bias is retired as a gate; it is reported beside, and a failed
    plant (or a missing one) does not refuse. The sum gate and the known answer still gate."""
    import json
    out, known = tmp_path / "out", tmp_path / "known"
    known.mkdir()
    for mk, (_, steps, _) in t5f.MODELS.items():
        (out / mk).mkdir(parents=True)
        for s in steps:
            (out / mk / f"step{s}.json").write_text(json.dumps(_rec()))
        pops = {f"step{s}": {"mean_cos": [0.5, 0.6]} for s in t5f.KNOWN[mk]}
        (known / f"diag_{mk}.json").write_text(json.dumps({"populations": pops}))
    g = t5f.gate(out, known)
    assert g["pass"] and g["planted_bias_beside"].get("missing")
    (out / "32m" / f"step{t5f.PLANT_STEP}_planted.json").write_text(json.dumps(_rec(wt=20.0, ab=1.0)))
    g = t5f.gate(out, known)
    assert g["pass"] and g["planted_bias_beside"]["share"] == pytest.approx(0.1)
    assert "retired" in g["planted_bias_beside"] and "planted_bias" not in g
    # the remaining gates still refuse
    (out / "410m" / "step0.json").write_text(json.dumps(_rec(rho=(0.5, 0.7))))
    assert not t5f.gate(out, known)["pass"]
    (out / "410m" / "step0.json").write_text(json.dumps(_rec()))
    (out / "32m" / "step917.json").write_text(json.dumps(_rec(err=2e-4)))
    assert not t5f.gate(out, known)["pass"]


def test_reproduce_compares_every_number():
    r = {"write_total": 5.0, "write_total_no0": 4.0, "size_write": 0.1, "rho": [0.9, 0.8], "code": "x",
         **{k: {p: 0.1 for p in t5f.PATHS} for k in ("share", "share_no0", "size")}}
    assert t5f.reproduce(r, r)["equal"]
    q = {**r, "share": {**r["share"], "mlp_rest": 0.1 + 1e-9}}
    assert not t5f.reproduce(r, q)["equal"]
