"""
`tools/run/p10_lexical_carry.py` — §1.9's parked lexical-vs-contextual check.

Each split lift is observed minus a conditional expectation, so these check it
averages exactly 0 over every co-member set of a given size, that a planted
class-only or embedding-only structure moves only its own lift, that carry
reads 1 when a layer is its own layer 0, and that the delta needs step 0.
"""
import json
from itertools import combinations

import numpy as np
import pytest

from tools.run.p10_lexical_carry import (
    SPLIT, carry_stats, control_stats, deltas, measure_run, split_stats, summarise)
from tools.run.p10_token_composition import CompositionError

# Tier: numpy only -- runs in `scripts/check.sh pure`.
pytestmark = pytest.mark.pure


def _setup(n=9, seed=0):
    rng = np.random.default_rng(seed)
    row = rng.normal(size=n)
    cls = np.array(["a", "b", "a", "b", "a", "a", "b", "c", "a"], dtype=object)[:n]
    return row, cls


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_random_sets_average_zero(k):
    row, cls = _setup()
    f = 0
    pool = np.arange(1, len(row))
    sums = {p: [] for p in ("emb_given_class", "class_given_emb", "class_given_emb_20", "class_given_emb_40")}
    for co in combinations(pool.tolist(), k):
        r = split_stats(f, np.array(co), pool, row, cls)
        for p in sums:
            sums[p].append(r[p])
    for p, v in sums.items():
        assert np.mean(v) == pytest.approx(0.0, abs=1e-12), p


def test_emb_same_averages_zero_given_one_same_class_member():
    row, cls = _setup()
    pool = np.arange(1, len(row))
    same = [c for c in pool if cls[c] == cls[0]]
    vals = [split_stats(0, np.array([c]), pool, row, cls)["emb_same"] for c in same]
    assert np.mean(vals) == pytest.approx(0.0, abs=1e-12)


def test_planted_structures_move_their_own_lift():
    # 400 pool members, so 40 bins hold 10 each and the 40-bin lift can be non-zero
    n = 401
    pool = np.arange(1, n)
    rng = np.random.default_rng(1)
    cls = np.array(["a"] + ["a", "b"] * 200, dtype=object)
    row = rng.permutation(np.linspace(0, 1, n))       # class independent of embedding
    co = rng.choice([p for p in pool if cls[p] == "a"], 20, replace=False)
    r = split_stats(0, co, pool, row, cls)
    assert r["class_given_emb_40"] > 0.3               # class-only: survives 40 bins
    near = pool[np.argsort(-row[pool])][:20]
    r = split_stats(0, near, pool, row, cls)
    assert r["emb_given_class"] > 0.3                  # embedding-only
    assert abs(r["class_given_emb_40"]) < 0.15


def test_pool_must_hold_the_co_members():
    row, cls = _setup()
    with pytest.raises(CompositionError, match="pool"):
        split_stats(0, np.array([1]), np.arange(2, len(row)), row, cls)


def test_carry_is_one_when_layer_is_layer_0():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(12, 16))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    acts = np.stack([x, x, rng.permutation(x)])
    c = carry_stats(acts, 1, [1, 4, 7])
    assert c["self_cos"] == pytest.approx(1.0) and c["self_top1"] == 1.0
    assert c["self_pct"] == pytest.approx(1.0)
    assert carry_stats(acts, 1, [])["self_pct"] is None


def test_ceiling_matches_enumerated_same_class_sets():
    row, cls = _setup()
    pool = np.arange(1, len(row))
    same = [c for c in pool if cls[c] == cls[0]]
    k = 2
    want = np.mean([split_stats(0, np.array(c), pool, row, cls)["class_given_emb"]
                    for c in combinations(same, k)])
    got = control_stats(0, k, pool, row, cls)
    assert got["class_given_emb_classonly"] == pytest.approx(want)
    near = pool[np.argsort(-row[pool])][:k]
    assert got["class_given_emb_knn"] == pytest.approx(
        split_stats(0, near, pool, row, cls)["class_given_emb"])
    # k larger than the class: same value (it does not depend on k)
    assert control_stats(0, len(same) + 1, pool, row, cls)["class_given_emb_classonly"] == pytest.approx(want)


def test_carry_groups_use_only_runs_with_both():
    a = _rec(0.9)
    b = _rec(0.1)
    b[0]["focal"] = {"self_cos": None, "self_pct": None, "self_top1": None, "n": 0}
    b[0]["unclustered"]["self_pct"] = 0.0
    s = summarise({(0, "x"): a, (0, "y"): b})
    assert s[0][0]["n_runs_carry"] == 1
    assert s[0][0]["unclustered"]["self_pct"] == pytest.approx(0.5)


def _rec(v):
    carry = {"self_cos": v, "self_pct": v, "self_top1": v, "n": 3}
    return {0: {"n_focal": 3, "focal": carry, "unclustered": dict(carry, self_pct=0.5, self_cos=0.5, self_top1=0.5),
                **{p: v for p in SPLIT}}}


def test_delta_reading_and_refusal():
    s = summarise({(0, "x"): _rec(0.0), (512, "x"): _rec(0.2)})
    d = deltas(s)
    assert d[512][0]["class_given_emb"]["reading"] == "above step 0"
    assert d[512][0]["carry_gap.self_pct"]["delta"] == pytest.approx(0.2)
    with pytest.raises(CompositionError, match="step 0"):
        deltas({512: s[512]})


VOCAB = {"The": 464, "Ġcat": 3797, "Ġsat": 3332, ".": 15, "Ġthe": 262, "Ġdog": 3290}
TOKENS = ["The", "Ġcat", "Ġsat", ".", "Ġthe", "Ġdog"]


def _run(tmp_path, labels):
    d = tmp_path / "run"
    d.mkdir()
    (d / "tokens.txt").write_text("".join(f"{i:3d}  {t}\n" for i, t in enumerate(TOKENS)))
    (d / "hdbscan_labels.json").write_text(json.dumps(labels))
    return d


def test_measure_run_on_a_directory(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, len(TOKENS), 8))
    x /= np.linalg.norm(x, axis=2, keepdims=True)
    d = _run(tmp_path, {"0": [-1, 0, 0, 1, 0, 1], "1": [-1, -1, -1, -1, -1, -1]})
    out = measure_run(d, VOCAB, set(), acts=x)
    assert out[0]["n_focal"] == 5 and out[0]["focal"]["n"] == 5
    assert out[1]["n_focal"] == 0 and out[1]["class_given_emb"] is None
    assert out[1]["unclustered"]["n"] == 5
    with pytest.raises(CompositionError, match="activations hold"):
        measure_run(d, VOCAB, set(), acts=x[:, :4])
