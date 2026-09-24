"""
`tools/run/p10_comembership.py` — what clustered unique tokens cluster with.

The reading is observed minus a random-draw expectation, so these check the
expectations are exact (hypergeometric, mid-rank percentile) for both draws,
that a planted structure produces the lift it should, that the delta refuses
without its step-0 baseline, and that bad run directories are refused.
"""
import json
from itertools import combinations

import numpy as np
import pytest

from tools.run.p10_comembership import (
    PROPS,
    STAT_NAMES,
    deltas,
    focal_stats,
    measure_run,
    midrank_pct,
    p_none,
    summarise,
)
from tools.run.backfill_hdbscan import OUT_NAME
from tools.run.p10_token_composition import CompositionError

# Tier: numpy only -- runs in `scripts/check.sh pure`.
pytestmark = pytest.mark.pure

VOCAB = {"The": 464, "Ġcat": 3797, "Ġsat": 3332, ".": 15, "Ġthe": 262, "Ġdog": 3290}


def test_p_none_matches_enumeration():
    pool, good = 7, 3
    for k in range(0, 6):
        draws = list(combinations(range(pool), k))
        want = sum(1 for d in draws if not set(d) & set(range(good))) / len(draws)
        assert p_none(pool, good, k) == pytest.approx(want)
    assert p_none(5, 3, 3) == 0.0


def test_midrank_pct_averages_one_half_with_ties():
    v = np.array([0.1, 0.5, 0.5, 0.5, 0.9, 1.0])
    assert midrank_pct(v, v).mean() == pytest.approx(0.5)
    assert midrank_pct(v, np.array([0.1]))[0] == 0.0
    assert midrank_pct(v, np.array([1.0]))[0] == 1.0
    assert midrank_pct(np.array([0.3]), np.array([0.3]))[0] == 0.5   # a forced draw


@pytest.mark.parametrize("pool", [np.array([0, 1, 2, 4, 5, 6]), np.array([1, 2, 4, 6])])
def test_random_members_have_zero_expected_lift(pool):
    # Averaged over every co-member set drawn from the pool, observed equals expected.
    f, k = 3, 2
    is_copy = np.array([1, 1, 0, 0, 1, 0, 0], bool)
    cls = np.array(list("aabaabb"), dtype=object)
    emb = {"emb_pct": np.linspace(0, 1, 7), "emb_pct_own": np.array([3, 1, 4, 1, 5, 9, 2.0])}
    rows = [focal_stats(f, np.array([f, *co]), pool, is_copy, cls, emb) for co in combinations(pool, k)]
    for p in PROPS:
        assert np.mean([r[p][0] for r in rows]) == pytest.approx(rows[0][p][1]), p


def test_pool_must_hold_the_co_members():
    with pytest.raises(CompositionError, match="pool"):
        focal_stats(3, np.array([3, 5]), np.array([1, 2]), np.zeros(7, bool),
                    np.array(["a"] * 7, dtype=object), {})


def test_planted_neighbour_cluster_lifts_adjacent_only_where_planted():
    n = 20
    is_copy = np.zeros(n, bool)
    cls = np.array(["w"] * n, dtype=object)
    emb = {"emb_pct": np.random.default_rng(0).normal(size=n)}
    r = focal_stats(10, np.array([9, 10, 11]), np.delete(np.arange(n), 10), is_copy, cls, emb)
    assert r["adjacent"][0] == 1.0 and r["adjacent"][1] < 0.25
    assert r["no_copy"] == (1.0, 1.0)            # no copies anywhere: no lift possible
    assert r["k"] == 2


def _run_rec(lift):
    return {0: {"n_focal": 4, "n_unique": 8, "n_clusters": 2, "k": 3.0,
                **{p: [0.5 + lift, 0.5, lift] * 2 for p in PROPS}}}


def test_summary_and_delta_reading():
    results = {(0, "a"): _run_rec(0.0), (0, "repeated_tokens"): _run_rec(0.4),
               (100, "a"): _run_rec(0.2), (100, "repeated_tokens"): _run_rec(0.4)}
    s = summarise(results)
    assert set(s["all"][0][0]["adjacent"]) == set(STAT_NAMES)
    assert s["all"][0][0]["adjacent"]["lift"] == pytest.approx(0.2)
    assert s["without_repeated_tokens"][0][0]["adjacent"]["lift_cl"] == pytest.approx(0.0)
    assert s["all"][0][0]["focal_frac"] == pytest.approx(0.5)
    assert s["all"][0][0]["n_clusters"] == pytest.approx(2)
    d = deltas(s)
    w = d["without_repeated_tokens"][100][0]
    assert w["emb_pct_own"]["reading"] == "above step 0"
    assert w["emb_pct_own"]["reading_cl"] == "above step 0"
    assert w["emb_pct"]["reading"] == "n/a (frozen frame)"      # no valid step-0 baseline
    assert d["all"][100][0]["same_class"]["delta"] == pytest.approx(0.1)
    assert d["all"][0]["mean"]["copy_share"]["reading"] == "as step 0"


def test_delta_refuses_without_step_0():
    s = summarise({(64, "a"): _run_rec(0.1), (100, "a"): _run_rec(0.2)})
    with pytest.raises(CompositionError, match="step 0"):
        deltas(s)


def test_run_with_no_focal_tokens_is_skipped_not_zeroed():
    empty = {0: {"n_focal": 0, "n_unique": 5, "n_clusters": 0}}
    s = summarise({(0, "a"): empty, (0, "b"): {0: {**_run_rec(0.4)[0], "n_unique": 20}}})
    assert s["all"][0][0]["n_runs"] == 1
    assert s["all"][0][0]["adjacent"]["lift"] == pytest.approx(0.4)
    assert s["all"][0][0]["focal_frac"] == pytest.approx(0.1)


TOKENS = ("The", "Ġcat", "Ġsat", ".", "Ġcat", "Ġdog")


def _run(tmp_path, labels, tokens=TOKENS, name="native"):
    d = tmp_path / name
    d.mkdir()
    (d / "tokens.txt").write_text("".join(f"{i:3d}  {t}\n" for i, t in enumerate(tokens)))
    (d / "hdbscan_labels.json").write_text(json.dumps(labels))
    return d


def _gram(n=len(TOKENS)):
    return np.eye(n) + 0.1 * np.arange(n)[:, None] * np.arange(n)[None, :] / n ** 2


def test_measure_run_on_a_directory(tmp_path):
    # unique at > 0: sat(2), .(3), dog(5); cat(1, 4) is a copy pair.
    d = _run(tmp_path, {"0": [-1, 0, 0, 1, 0, 1], "1": [-1, -1, -1, -1, -1, -1]})
    r = measure_run(d, VOCAB, set(), _gram(), np.array(TOKENS, dtype=object), own_gram=_gram())
    assert r[0]["n_focal"] == 3 and r[0]["n_unique"] == 3 and r[0]["n_clusters"] == 2
    assert len(r[0]["copy_share"]) == 6
    assert r[1]["n_focal"] == 0 and "copy_share" not in r[1]


@pytest.mark.parametrize("labels,kw,match", [
    ({"0": [-1] * 6}, {}, "noise at every layer"),
    ({"0": [0, 0, 1]}, {}, "3 labels"),
    ({"0": [0, 0, 1, 1, 0, 0]}, {"frozen_tokens": np.array(TOKENS[::-1], dtype=object)}, "tokens differ"),
    ({"0": [0, 0, 1, 1, 0, 0]}, {"frozen_gram": np.eye(5)}, "frozen Gram"),
    ({"0": [0, 0, 1, 1, 0, 0]}, {"own_gram": np.eye(5)}, "own Gram"),
])
def test_refusals(tmp_path, labels, kw, match):
    args = {"frozen_gram": _gram(), "frozen_tokens": np.array(TOKENS, dtype=object), "own_gram": _gram()}
    args.update(kw)
    with pytest.raises(CompositionError, match=match):
        measure_run(_run(tmp_path, labels), VOCAB, set(), **args)


def test_backfilled_labels_refused(tmp_path):
    d = _run(tmp_path, {"0": [0, 0, 1, 1, 0, 0]})
    (d / OUT_NAME).write_text(json.dumps({"0": [0, 0, 1, 1, 0, 0]}))
    with pytest.raises(CompositionError, match="not native"):
        measure_run(d, VOCAB, set(), _gram(), np.array(TOKENS, dtype=object), own_gram=_gram())
