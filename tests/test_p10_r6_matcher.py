"""
`tools/run/p10_r6_matcher.py` — R6's cross-checkpoint matcher: link kinds, stable Jaccard, filter
flips, lineage origin, pooling and the first check, as `design-10.md` "R6" fixed them.
"""
import numpy as np
import pytest

from tools.run import p10_r6_matcher as r6

# Tier: merge_tree's package imports scikit-learn -- runs in `scripts/check.sh deps`.
pytestmark = pytest.mark.deps


def test_a_group_that_keeps_its_members_is_stable_with_jaccard_one():
    out = r6.link([0, 0, 0, -1, -1], [5, 5, 5, -1, -1])
    assert out["kind_a"] == {0: "stable"} and out["kind_b"] == {5: "stable"}
    assert out["stable"] == {5: (0, 1.0)}


def test_a_changed_group_is_stable_on_containment_with_jaccard_below_one():
    # 3 of 4 kept, 2 joined: containment 3/4, Jaccard 3/6
    out = r6.link([0, 0, 0, 0, -1, -1], [1, 1, 1, -1, 1, 1])
    assert out["stable"] == {1: (0, 0.5)}


def test_a_split_and_a_birth_and_a_death():
    out = r6.link([0, 0, 0, 0, 1, 1, -1, -1], [2, 2, 3, 3, -1, -1, 4, 4])
    assert out["kind_a"] == {0: "split", 1: "death"}
    assert out["kind_b"] == {2: "split", 3: "split", 4: "birth"}
    assert out["stable"] == {}


def test_a_small_group_inside_a_large_one_links_on_containment():
    # Jaccard 2/8 would call this a death and a birth (status-1d "Merge tree", #104)
    out = r6.link([0, 0, -1, -1, -1, -1, -1, -1], [1] * 8)
    assert out["kind_a"] == {0: "stable"}


def test_a_c3_birth_is_a_flip_when_c2a_links_the_same_id():
    c2a = r6.link([0, 0, 0, 1, 1, 1], [7, 7, 7, 8, 8, 8])
    c3 = r6.link([0, 0, 0, -1, -1, -1], [7, 7, 7, 8, 8, 8])   # group 1 failed c3's filter at the earlier step
    assert r6.flips(c3, c2a) == {"birth_flip_kept": 1}
    c2a_new = r6.link([0, 0, 0, -1, -1, -1], [7, 7, 7, 8, 8, 8])
    assert r6.flips(c3, c2a_new) == {"birth_new": 1}


def test_a_c3_death_is_gone_unless_c2a_links_it():
    c3 = r6.link([0, 0, 1, 1], [5, 5, -1, -1])
    assert r6.flips(c3, r6.link([0, 0, 1, 1], [5, 5, 6, 6])) == {"death_flip_kept": 1}
    assert r6.flips(c3, r6.link([0, 0, 1, 1], [5, 5, -1, -1])) == {"death_gone": 1}


def test_a_flip_behind_a_c2a_merge_is_restructured_not_kept():
    # c2a merges groups 0 and 1 into 7; c3 kept only group 1 at the earlier step, nothing later
    c2a = r6.link([0, 0, 1, 1], [7, 7, 7, 7])
    c3 = r6.link([-1, -1, 1, 1], [-1, -1, -1, -1])
    assert r6.flips(c3, c2a) == {"death_flip_restructured": 1}


def test_flips_refuse_an_id_c2a_does_not_hold():
    with pytest.raises(ValueError, match="not in c2a"):
        r6.flips(r6.link([0, 0], [-1, -1]), r6.link([-1, -1], [-1, -1]))


def test_independent_lineage_is_the_product_of_backward_rates():
    rows = [{"c": {"null": {"observed": 1}, "n_b": 2}}, {"c": {"null": {"observed": 3}, "n_b": 4}}]
    assert r6.independent_lineage(rows, [0, 2, 4], "c") == {"0": 0.375, "2": 0.75, "4": 1.0}


def test_lineage_origin_walks_back_through_stable_links_only():
    steps = [0, 2, 4]
    links = [r6.link([0, 0, 1, 1], [3, 3, 3, 3]),   # merge: breaks the chain
             r6.link([3, 3, 3, 3], [9, 9, 9, 9])]
    assert r6.lineage_origin(links, steps, 9) == 2
    links = [r6.link([0, 0, -1, -1], [3, 3, -1, -1]), r6.link([3, 3, -1, -1], [9, 9, -1, -1])]
    assert r6.lineage_origin(links, steps, 9) == 0


def test_chain_and_pool_on_a_toy_record(monkeypatch):
    monkeypatch.setattr(r6, "N_DRAWS", 20)
    steps = [0, 2, 4]
    lab = {0: np.array([0, 0, 0, 1, 1, 1, -1, -1]), 2: np.array([4, 4, 4, 5, 5, 5, -1, -1]),
           4: np.array([6, 6, 6, -1, -1, -1, 7, 7])}
    by_col = {c: lab for c in r6.COLUMNS}
    ch = r6.chain((("p", 1), steps, by_col, [0, 0, 1]))
    out = r6.pool([ch], steps, r6.COLUMNS)
    b0, b1 = out["boundaries"][0]["c3"], out["boundaries"][1]["c3"]
    assert b0["share_a"]["stable"] == 1.0 and b0["births"] == 0
    assert b1["share_a"]["stable"] == 0.5 and b1["share_a"]["death"] == 0.5 and b1["births"] == 1
    assert b1["flips"] == {"birth_new": 1, "death_gone": 1}
    assert out["lineage_independent_cumulative"]["c3"] == {"0": 0.5, "2": 0.5, "4": 1.0}
    assert b0["stable_jaccard"] == {"n": 2, "median": 1.0, "identical": 1.0, "same": 1.0}
    assert out["lineage_origin_at_last"]["c3"] == {"0": 1, "2": 0, "4": 1}
    assert 0 < b0["null"]["p"] <= 1 and b0["null"]["observed"] == 2
    assert r6.first_check(out)["passes"]


def _toy_pooled(monkeypatch, c3x_last):
    """Steps 2000 → 4000 → 8000; c3x differs from c3 only in its last step."""
    monkeypatch.setattr(r6, "N_DRAWS", 20)
    steps = [2000, 4000, 8000]
    lab = {2000: np.array([0, 0, 0, 1, 1, 1, -1, -1]), 4000: np.array([4, 4, 4, 5, 5, 5, -1, -1]),
           8000: np.array([6, 6, 6, 8, 8, 8, -1, -1])}
    by_col = {c: lab for c in r6.columns_of("c3x")}
    by_col["c3x"] = {**lab, 8000: np.asarray(c3x_last)}
    ch = r6.chain((("p", 1), steps, by_col, [0, 0, 1]))
    return r6.jsonable(r6.pool([ch], steps, r6.columns_of("c3x")))


def test_c3x_joins_the_columns_only_when_it_leads():
    assert "c3x" not in r6.columns_of("c3") and r6.columns_of("c3x")[-1] == "c3x"
    assert "c3x" in r6.FLIP_COLUMNS


def test_clauses_unchanged_when_c3x_is_c3(monkeypatch):
    pooled = _toy_pooled(monkeypatch, [6, 6, 6, 8, 8, 8, -1, -1])
    cl = r6.clauses(pooled)
    assert cl["i_survival"]["max_abs_diff"] == 0.0 and not cl["i_survival"]["changed"]
    assert cl["iv_lineage"]["before_256"] == {"c3": 0, "c3x": 0}
    # identical sets everywhere: "changed members" does not hold on the toy, on either column
    assert "ii_changed_members" in cl["changed"] and "i_survival" not in cl["changed"]


def test_clauses_name_a_drop_in_survival_and_a_new_death(monkeypatch):
    pooled = _toy_pooled(monkeypatch, [6, 6, 6, -1, -1, -1, -1, -1])  # c3x drops group 8 at 8000
    cl = r6.clauses(pooled)
    assert cl["i_survival"]["max_abs_diff"] == 0.5 and cl["i_survival"]["changed"]
    assert cl["iii_few_new"]["max_new_share"] == {"c3": 0.0, "c3x": 0.0}  # c2a links it: a kept flip
    assert cl["rows"]["c3x"][0]["births_deaths"] == 1


def test_reproduce_names_each_differing_column_and_ignores_the_new_one(monkeypatch):
    a = _toy_pooled(monkeypatch, [6, 6, 6, 8, 8, 8, -1, -1])
    stored = {"meta": {"columns": list(r6.COLUMNS)}, **a,
              "lineage_origin_at_last": {k: v for k, v in a["lineage_origin_at_last"].items() if "c3x" not in k}}
    b = _toy_pooled(monkeypatch, [6, 6, 6, -1, -1, -1, -1, -1])
    assert r6.reproduce(b, stored) == []  # only c3x differs, and the stored run has no c3x
    b["boundaries"][1]["c2"]["births"] += 1
    b["lineage_origin_at_last"]["c3_by_c2a"]["2000"] += 1
    assert r6.reproduce(b, stored) == ["4000/c2", "lineage/c3_by_c2a"]


def test_load_record_refuses_a_run_on_another_label_source(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    rec = tmp_path / "r6.json"
    rec.write_text('{"meta": {"summary_sha256": "0000"}}')
    with pytest.raises(r6.lead_args.LadderError, match="read a label source"):
        r6.load_record(rec, tmp_path)


def test_main_refuses_a_half_given_reproduce(tmp_path):
    with pytest.raises(SystemExit, match="go together"):
        r6.main(["--labels", str(tmp_path), "--out", str(tmp_path / "o.json"), "--reproduce", "x.json"])
