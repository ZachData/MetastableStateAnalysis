"""
`tools/run/p10_r7_shuffle.py` — R7's context-shuffle test, as `design-10.md` "R7" fixed it.
"""
import numpy as np
import pytest

from p1d_cluster_ensemble import move_text as mt
from tools.run import p10_r7_shuffle as r

# Tier: the module imports unit 1's runner, whose package imports scikit-learn (as move_text's tests).
pytestmark = pytest.mark.deps


def test_block_order_keeps_offset_0_first_and_is_a_permutation():
    for b in r.BLOCKS:
        o = r.block_order(200, b, 3, 1)
        assert o[0] == 0 and sorted(o.tolist()) == list(range(200))
        assert not np.array_equal(o, np.arange(200))


def test_block_order_keeps_each_block_in_order():
    o = r.block_order(50, 16, 0, 0)
    starts = {1, 17, 33, 49}
    runs = [o[i:i + 16] for i in range(1, 50) if o[i] in starts]
    for run in runs:
        k = min(16, 50 - run[0])
        assert run[:k].tolist() == list(range(run[0], run[0] + k))


def test_permutations_do_not_depend_on_the_step():
    # seeded by (passage, b, k) only: the same call gives the same order
    assert np.array_equal(r.block_order(300, 4, 2, 3), r.block_order(300, 4, 2, 3))
    assert not np.array_equal(r.block_order(300, 4, 2, 3), r.block_order(300, 4, 2, 4))


def test_one_block_refuses():
    with pytest.raises(r.ShuffleError):
        r.block_order(10, 64, 0, 0)


def test_check_order_catches_a_moved_offset_0_and_a_wrong_map():
    ids = list(range(100, 140))
    kept = np.arange(1, 40)
    r.check_order(ids, r.block_order(40, 4, 0, 0), kept)
    bad = np.arange(40)[::-1].copy()
    with pytest.raises(r.ShuffleError):
        r.check_order(ids, bad, kept)


def test_positions_of_inverts_the_order():
    o = r.block_order(60, 4, 1, 2)
    pos = r.positions_of(o)
    assert all(o[pos[x]] == x for x in range(60))


def test_best_jaccard_on_a_labelling_matches_unit_1s():
    rng = np.random.default_rng(0)
    for _ in range(50):
        lab = rng.integers(-1, 6, size=40)
        mem = np.sort(rng.choice(40, size=rng.integers(2, 8), replace=False))
        groups = [np.flatnonzero(lab == c) for c in range(6) if np.any(lab == c)]
        assert r.best_jaccard_lab(mem, lab) == pytest.approx(mt.best_jaccard(mem.tolist(),
                                                                             [g.tolist() for g in groups]))


def test_a_dropped_member_leaves_the_group():
    lab = np.array([0, 0, -2, 1, 1, -1])
    assert r.best_jaccard_lab(np.array([0, 1, 2]), lab) == 1.0      # restricted to {0, 1}
    assert r.best_jaccard_lab(np.array([0, 1, 5]), lab) == pytest.approx(2 / 3)   # noise counts
    assert r.best_jaccard_lab(np.array([2]), lab) == 0.0


def _g(J0, alone, b1, other=1.0):
    J = {f"b{b}": [other] * r.K for b in r.BLOCKS}
    J["b1"] = [b1] * r.K
    J["alone"] = alone
    return {"J0": J0, "J": J}


def test_labels_follow_alone_then_b1():
    assert r.group_label(_g(0.5, 0.6, 0.0))["label"] == "token-borne"
    assert r.group_label(_g(0.5, 0.4, 0.5))["label"] == "bag-borne"
    assert r.group_label(_g(0.5, 0.4, 0.4))["label"] == "order-borne"
    assert r.group_label(_g(0.5, 0.4, 0.4), bar=0.3)["label"] == "token-borne"


def test_survival_is_the_median_over_permutations():
    g = _g(0.5, 0.0, 0.0)
    g["J"]["b1"] = [1, 1, 1, 0, 0]
    assert r.group_label(g)["survives"]["b1"]
    g["J"]["b1"] = [1, 1, 0, 0, 0]
    assert not r.group_label(g)["survives"]["b1"]


def test_step_label_thresholds():
    assert r.step_label(0.7) == "token" and r.step_label(0.3) == "context"
    assert r.step_label(0.5) == "mixed" and r.step_label(None) == "no records"


def test_chance_is_low_for_a_fine_partition():
    lab = np.repeat(np.arange(20), 2)
    rng = np.random.default_rng(1)
    assert r.chance95(2, lab, rng) <= 1.0
    assert np.isnan(r.chance95(50, lab, rng))


# ---------------------------------------------------------------- R9 / R7 (design-10 "R9 / R7")

def _dom():
    a = lambda *x: np.asarray(x, dtype=int)
    return {("step64", "p", 1): {"c3": a(0, 0, 1, 1, 2, 2, -1), "c3x": a(0, 0, -1, -1, 2, 2, -1)},
            ("step64", "p", 2): {"c3": a(3, 3, 4, 4, -1, -1, -1), "c3x": a(-1, -1, -1, -1, -1, -1, -1)},
            ("step64", "q", 1): {"c3": a(5, 5, 6, 6, -1, -1, -1), "c3x": a(5, 5, 6, 6, -1, -1, -1)}}


def test_c3x_drops_are_c3_less_c3x():
    assert r.c3x_drops(_dom()) == {("step64", "p", 1): {1}, ("step64", "p", 2): {3, 4}, ("step64", "q", 1): set()}


def test_draw_per_record_matches_each_records_count():
    dom = _dom()
    x = r.c3x_drops(dom)
    for seed in range(20):
        d = r.draw_drops(dom, x, "record", np.random.default_rng(seed))
        assert {k: len(v) for k, v in d.items()} == {k: len(v) for k, v in x.items()}
        assert all(v <= r.ids_of(dom[k]["c3"]) for k, v in d.items())


def test_draw_per_step_passage_matches_the_pooled_count_and_can_move_between_layers():
    dom = _dom()
    x = r.c3x_drops(dom)
    seen = set()
    for seed in range(50):
        d = r.draw_drops(dom, x, "step_passage", np.random.default_rng(seed))
        assert len(d[("step64", "p", 1)]) + len(d[("step64", "p", 2)]) == 3
        assert not d[("step64", "q", 1)]
        assert all(v <= r.ids_of(dom[k]["c3"]) for k, v in d.items())
        seen.add(len(d[("step64", "p", 1)]))
    assert len(seen) > 1          # not forced to c3x's per-layer split


def test_a_seed_fixes_the_draw():
    dom = _dom()
    x = r.c3x_drops(dom)
    for m in r.MATCHES:
        assert r.draw_drops(dom, x, m, np.random.default_rng(3)) == r.draw_drops(dom, x, m, np.random.default_rng(3))


def test_dropping_c3xs_groups_gives_c3x_labels_check_r1():
    dom = _dom()
    labs = r.drop_labels(dom, r.c3x_drops(dom))
    assert all(np.array_equal(labs[k], dom[k]["c3x"]) for k in dom)
    assert all(np.array_equal(v, dom[k]["c3"]) for k, v in r.drop_labels(dom, {}).items())


def test_compare_cells_counts_one_sided_cells_and_the_headline():
    ref = {"s|step own": "mixed", "s|step fixed": "context", "s|band L1-8": "token", "s|prompt a": None}
    col = {"s|step own": "context", "s|step fixed": "context", "s|band L1-8": None, "s|prompt a": None}
    c = r.compare_cells(ref, col)
    assert c["n_compared"] == 3 and c["changed"] == ["s|band L1-8", "s|step own"]
    assert c["headline_changed"] == ["s|step own"]


def test_reference_reading_within_and_beyond():
    counts = list(range(100))
    assert r.reference_reading(94, counts)["reading"] == "within random drops"
    assert r.reference_reading(99, counts)["reading"] == "beyond random drops"


def test_read_step_is_read_rows_of_the_flagged_rows():
    g = lambda i, c3, j: {"id": i, "size": 2, "c3": c3, "J0": 0.5, "same_class": 1.0, "chance95": {"alone": 0.1},
                          "J": {**{f"b{b}_k{k}": j for b in r.BLOCKS for k in range(r.K)}, "alone": j}}
    recs = [{"passage": mt.V1_PASSAGES[0], "layers": [{"layer": 1, "groups": [g(0, True, 0.9), g(1, False, 0.1)]}]}]
    assert r.read_step(recs, "c3") == r.read_rows([x for x in r.step_rows(recs) if x["c3"]])
    assert r.read_step(recs, "c3")["n"] == 1 and r.read_step(recs, "c2")["n"] == 2
