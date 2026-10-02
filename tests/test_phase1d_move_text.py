"""
tests/test_phase1d_move_text.py — unit 1's runner (`p1d_cluster_ensemble/move_text.py`)
on synthetic hidden states with known answers.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.deps

from p1d_cluster_ensemble import move_text as mt


def test_conditions_skip_the_passages_own_continuation():
    conts = {s: list(range(100, 1200)) for s in mt.PREAMBLE_SOURCES}
    joins = {"eod": [0], "nl2": [535]}
    own = mt.conditions("wiki_paragraph", [7, 8, 9], conts, joins)
    other = mt.conditions("homer_iliad", [7, 8, 9], conts, joins)
    nP = len(mt.P_VALUES)
    assert len(own) == 1 + 2 * nP * 2 and len(other) == 1 + 3 * nP * 2
    assert all(c["preamble"] != "wiki_paragraph" for c in own)
    P = mt.P_VALUES[1]
    c = next(c for c in other if c["id"] == f"hdbscan_code|{P}|eod")
    assert c["start"] == P + 1 and c["ids"][P] == 0 and c["ids"][P + 1:] == [7, 8, 9]
    assert mt.conditions("x", [1], conts, joins, p0_only=True)[0]["id"] == "P0"


def test_massive_positions_reads_only_the_named_layers():
    norms = np.ones((25, 30))
    norms[5, 3] = 50.0     # massive at L5
    norms[22, 4] = 50.0    # only at L22: outside L2-20
    m = mt.massive_positions(norms)
    assert set(m) == {3} and m[3] == (50.0, 5)


def test_token_rules():
    toks = ["<s>", "a", "b", "a", "c", "d"]
    assert mt.kept_offsets(toks, [4]).tolist() == [1, 2, 5]
    # whole sequence: preamble "x y", join "|", passage starts at 3
    whole = ["x", "y", "|", "a", "y", "b"]
    assert mt.whole_kept(whole, 3, own_massive=[], passage_dropped=[2]).tolist() == [1, 2, 3]


def test_classify_and_content_label():
    assert mt.classify({"a": True, "b": True}, False, False) == "moves"
    assert mt.classify({"a": True, "b": False}, True, True) == "preamble-dependent"
    assert mt.classify({"a": False, "b": False}, True, True) == "opening-bound"
    assert mt.classify({"a": False, "b": False}, True, False) == "context-bound"
    assert mt.classify({"a": False}, False, True) == "context-bound"
    assert mt.content_label(["x", "x", "x", "y", None]) == "x"       # 3 of 4 labelled
    assert mt.content_label(["x", "x", "y", None]) is None           # only 2
    assert mt.content_label(["x", "x", "x", "y", "y"]) is None       # 0.6 < 0.75


def _caps(rng, n_groups, size, d=32, spread=0.05, n_bg=20):
    centres = rng.standard_normal((n_groups, d))
    X = [c + spread * rng.standard_normal((size, d)) for c in centres]
    X.append(rng.standard_normal((n_bg, d)))
    return np.vstack(X)


def _setup(passage_rows, cond_fn, labels=None):
    """One passage of ``len(passage_rows)`` tokens; ``cond_fn(P)`` gives its rows at P."""
    rng = np.random.default_rng(1)
    n, d = passage_rows.shape
    pre_src = mt.PREAMBLE_SOURCES[1:]   # pretend the passage is wiki_paragraph's
    conds = [{"id": "P0", "preamble": None, "P": 0, "join": None, "start": 0}]
    hidden = {"P0": np.repeat(passage_rows[None], 25, axis=0)}
    whole = {}
    for src in pre_src:
        for P in mt.P_VALUES:
            for j in mt.JOINS:
                cid = f"{src}|{P}|{j}"
                pre = rng.standard_normal((P + 1, d))
                pre[:6] = 5 + 0.01 * rng.standard_normal((6, d))   # a group at the start
                rows = np.vstack([pre, cond_fn(P)])
                hidden[cid] = np.repeat(rows[None], 25, axis=0)
                conds.append({"id": cid, "preamble": src, "P": P, "join": j, "start": P + 1})
                whole[cid] = np.arange(1, P + 1 + n)
    kept = np.arange(1, n)
    mt._G.clear()
    mt._G.update(conds=conds, kept=kept, labels=labels, hidden=hidden, whole_kept=whole,
                 seed=0, passage_index=0)
    return kept


@pytest.fixture(autouse=True)
def _few_subsamples(monkeypatch):
    monkeypatch.setattr(mt, "N_SUBSAMPLES", 8)
    monkeypatch.setattr(mt, "P_VALUES", (5, 20))


def test_groups_that_ride_with_the_passage_move():
    rng = np.random.default_rng(0)
    X = _caps(rng, 3, 10)
    X = np.vstack([rng.standard_normal((1, X.shape[1])), X])     # offset 0, dropped
    _setup(X, lambda P: X)
    rec = mt._layer_job((3, "raw"))["mcs"]["4"]
    stable = [g for g in rec["groups"] if g["stable"]]
    assert len(stable) >= 3 and all(g["class"] == "moves" for g in stable)
    assert rec["c_holds"]
    cos = mt._layer_job((3, "raw"))["cos"]
    assert all(abs(v["median"] - 1) < 1e-9 for v in cos.values())


def test_groups_destroyed_by_any_preamble_do_not_move():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.standard_normal((1, 32)), _caps(rng, 3, 10)])
    noise = np.random.default_rng(5)
    _setup(X, lambda P: noise.standard_normal(X.shape))
    rec = mt._layer_job((3, "raw"))["mcs"]["4"]
    stable = [g for g in rec["groups"] if g["stable"]]
    assert stable and all(g["class"] in ("opening-bound", "context-bound") for g in stable)
    # the first planted cap holds offsets 1-10, so it is the opening group, and (c) holds
    o = rec["groups"][rec["opening_group"]]
    assert o["opening"] and o["class"] == "opening-bound"
    assert all(g["class"] == "context-bound" for g in stable if not g["opening"])


def test_designed_labels_mark_content_groups():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.standard_normal((1, 32)), _caps(rng, 2, 10, n_bg=10)])
    labels = [None] + ["a"] * 10 + ["b"] * 10 + [None] * 10
    _setup(X, lambda P: X, labels=labels)
    rec = mt._layer_job((3, "raw"))
    t = mt.designed_table([{"passage": "designed_x", "layers": [rec | {"frame": "centred"}]}], mcs=4)
    assert t["designed_x"]["content_groups"] >= 2
    assert t["designed_x"]["share_moves"] == 1.0
    assert mt.designed_verdict(t) == "pass"


def test_verdicts():
    t = {p: {"modal": "opening-bound"} for p in "abcd"} | {p: {"modal": "moves"} for p in "efg"}
    assert mt.step0_verdict(t) == "pass"
    t = {p: {"modal": "moves"} for p in "abcd"} | {p: {"modal": "unstable"} for p in "efg"}
    assert mt.step0_verdict(t) == "stop"
    t = {p: {"modal": "context-bound"} for p in "abcdefg"}
    assert mt.step0_verdict(t) == "neither"
    assert mt.designed_verdict({"x": {"classified": {}}}).startswith("fail")


def test_class_table_pools_stable_groups_by_band():
    g = lambda cls, stable=True: {"stable": stable, "class": cls, "class_nl2": "moves"}
    lay = lambda L, gs: {"layer": L, "frame": "centred", "cos": {f"a|{max(mt.P_VALUES)}|eod": {"median": 0.9}},
                         "mcs": {"2": {"groups": gs}}}
    recs = [{"layers": [lay(1, [g("moves"), g("context-bound")]), lay(9, [g("moves"), g(None, False)])]}]
    t = {r["band"]: r for r in mt.class_table(recs)}
    assert t["L1-8"]["share_moves"] == 0.5 and t["L1-8"]["context-bound"] == 1
    assert t["L9-16"]["unstable"] == 1 and t["L9-16"]["classified"] == 1
    assert t["L1-8"]["cos_P1000_median"] == 0.9
    assert mt.class_table(recs, "class_nl2")[0]["share_moves"] == 1.0
