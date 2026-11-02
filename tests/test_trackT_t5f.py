"""T5 U3's split (`trackT_timaeus/t5f.py`): the aggregation and the plant's gate (pure), and the
split on tiny random models in the smoke file: the parts sum to the block's
write, the bias path is exactly the bias's component (`test_trackT_t5f_smoke.py`)."""
import numpy as np
import pytest

pytestmark = pytest.mark.pure
from trackT_timaeus import t5f


def _blk(n, u, sums, sums0, mfree_sum, sq=1.0):
    return {"n": n, "u": np.asarray(u, float), "mfree_sum": np.asarray(mfree_sum, float), "sum": sums,
            "sum_no0": sums0, "sq": sq, "err_sum": 0.0, "err_attn": 0.0, "err_modules": 0.0,
            "bias_norm": {"attn": 0.0, "mlp": 0.0}}


def _sums(sink, bias, rest, mb, mfree):
    return {"attn_sink": sink, "attn_bias": bias, "attn_rest": rest, "mlp_bias": mb, "mlp_free": mfree,
            "write": sink + bias + rest + mb + mfree}


def test_aggregate_const_and_shares():
    # two prompts, one block, d 2; mfree token sums (2, 0) over 2 tokens and (0, 4) over 2 tokens
    p1 = [_blk(2, [1, 0], _sums(1.0, 0.0, 1.0, 0.0, 2.0), _sums(0.5, 0.0, 1.0, 0.0, 1.0), [2, 0])]
    p2 = [_blk(2, [0, 1], _sums(0.0, 0.0, 2.0, 0.0, 4.0), _sums(0.0, 0.0, 1.0, 0.0, 2.0), [0, 4])]
    r = t5f.aggregate([p1, p2])
    # the mean over every token: (2, 4) / 4 = (0.5, 1); const = n · m·u = 2·0.5 + 2·1 = 3
    assert r["per_block"]["component"]["mlp_const"][0] == pytest.approx(3.0)
    assert r["per_block"]["component"]["mlp_rest"][0] == pytest.approx(6.0 - 3.0)
    assert r["per_block"]["component_no0"]["mlp_const"][0] == pytest.approx(1 * 0.5 + 1 * 1.0)
    assert r["write_total"] == pytest.approx(10.0)
    assert sum(r["share"].values()) == pytest.approx(1.0)
    assert r["share"]["attn_sink"] == pytest.approx(0.1)
    assert r["closure"] == pytest.approx(0.0, abs=1e-12)
    # size: mean component per token / rms norm (sq 1 per prompt, N 4 → rms √(2/4))
    assert r["size"]["attn_sink"] == pytest.approx(1.0 / 4 / np.sqrt(0.5))


def test_plant_share():
    base = {"write_total": 10.0}
    pl = {"write_total": 30.0, "per_block": {"component": {"attn_bias": [9.0, 9.5]}}, "share": {"attn_bias": 0.62}}
    g = t5f.plant_share(pl, base)
    assert g["share"] == pytest.approx(18.5 / 20) and g["pass"]
    pl["per_block"]["component"]["attn_bias"] = [5.0, 5.0]
    assert not t5f.plant_share(pl, base)["pass"]


def test_aggregate_refuses_unpopulated():
    p = [_blk(2, [1, 0], _sums(0.0, 0.0, 0.0, 0.0, 0.0), _sums(0.0, 0.0, 0.0, 0.0, 0.0), [0, 0])]
    with pytest.raises((t5f.T5fRefused, ZeroDivisionError)):
        t5f.aggregate([p])
