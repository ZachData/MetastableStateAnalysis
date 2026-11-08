"""The library's U2 (`prompt_library/l2.py`, design-L.md "U2's details"): its bootstrap is T5 U2's
on the same draws, its statistics are `t5.state_stats`, its tokenizers add no slot."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # t5c's imports, transformers
from prompt_library import l2
from trackT_timaeus import t5, t5c


def test_counts_of():
    idx = np.array([[0, 0, 2], [1, 2, 2]])
    assert l2.counts_of(idx, 3).tolist() == [[2, 0, 1], [0, 1, 2]]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_count_spread_is_t5c_spread(seed):
    g = np.random.default_rng(seed)
    P = 0.5 + 0.3 * g.random((9, 11, 6))
    P[3] -= 0.2                                         # a minimum inside
    a = t5c.spread(P, np.random.default_rng(7), n_boot=200)
    b = l2.count_spread(P, np.random.default_rng(7), n_boot=200)
    assert a["label"] == b["label"] and a["argmin"] == b["argmin"]
    for k in ("delta", "p5", "p95", "first", "min"):
        assert abs(a[k] - b[k]) < 1e-12, k


def test_synthetic_is_t5c_synthetic_at_7():
    a = t5c.synthetic(0.03, 0.15, np.random.default_rng(3), draws=30)
    b = l2.synthetic(0.03, 0.15, np.random.default_rng(3), n_items=7, draws=30)
    assert {k: a[k] for k in ("spreads", "does not spread", "not told apart")} == \
           {k: b[k] for k in ("spreads", "does not spread", "not told apart")}


def test_item_spread_is_prompt_spread():
    R = np.random.default_rng(4).random((7, 9))
    assert abs(l2.item_spread(R) - t5c.prompt_spread({"rho_per_prompt": R.tolist()})) < 1e-15


def test_stats_torch_is_state_stats():
    torch = pytest.importorskip("torch")
    for n in (6, 7):                                    # even and odd n: the median's two cases
        h = np.random.default_rng(n).normal(size=(n, 5)).astype(np.float32)
        h[0] *= 9
        a = t5.state_stats(h)
        b = l2.stats_torch(torch.from_numpy(h))
        for k in l2.KEYS:
            assert abs(a[k] - b[k]) < 1e-12, (n, k)


def test_check_tokenizer_refuses_a_start_token():
    class Tok:
        def __call__(self, text, add_special_tokens=True):
            ids = [5, 6]
            return {"input_ids": ([1] if add_special_tokens else []) + ids}
    with pytest.raises(l2.L2Refused):
        l2.check_tokenizer(Tok(), "x")


def test_gate_refuses_a_known_answer_from_other_code(tmp_path):
    import json
    p = tmp_path / "known.json"
    p.write_text(json.dumps({"code": "aaaaaaaaaaaa", "pass": True, "max_diff": {}}))
    rho = np.full((len(t5c.DODECA_STEPS), 8, 5), 0.5) + np.random.default_rng(0).normal(0, 0.01, (len(t5c.DODECA_STEPS), 8, 5))
    recs = {"dodecahedron-32m": {"rho": rho}}
    assert l2.gate(recs, p, 8, "bbbbbbbbbbbb")["known_answer"]["pass"] is False
