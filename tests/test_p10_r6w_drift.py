"""
`tools/run/p10_r6w_drift.py` — R6w's decomposition of §1.9–§1.10's drift into within-lineage
change and replacement, as `design-10.md` "R6w" fixed it.
"""
import numpy as np
import pytest

from tools.run import p10_r6w_drift as w

# Tier: the module's top level imports numpy only; its readers load lazily inside main.
pytestmark = pytest.mark.pure


def _terms(d):
    return d["within"] + sum(d[k] for k in w.TERMS)


def test_the_terms_sum_to_the_change_in_the_weighted_mean():
    rng = np.random.default_rng(0)
    kinds_s = ["stable", "restructured", "death_gone", "death_flip_kept", "death_flip_restructured"]
    kinds_t = ["stable", "restructured", "birth_new", "birth_flip_kept", "birth_flip_restructured"]
    earlier = [(rng.random(), rng.normal(), kinds_s[i % 5]) for i in range(40)]
    later = [(rng.random(), rng.normal(), kinds_t[i % 5]) for i in range(37)]
    d = w.decompose(earlier, later)
    m = lambda rows: sum(a * x for a, x, _ in rows) / sum(a for a, _, _ in rows)
    assert d["total"] == pytest.approx(m(later) - m(earlier), abs=1e-12)
    assert _terms(d) == pytest.approx(d["total"], abs=1e-12)


def test_only_persisting_groups_changing_is_all_within():
    earlier = [(1, 0.1, "stable"), (1, 0.5, "death_gone")]
    later = [(1, 0.4, "stable"), (1, 0.8, "birth_new")]    # entrant and exit sit +0.4 above S alike
    d = w.decompose(earlier, later)
    assert d["within"] == pytest.approx(0.3) and d["new_gone"] == pytest.approx(0.0)
    assert w.label(d["total"], d["within"]) == "within"


def test_entrants_unlike_exits_is_replacement():
    earlier = [(1, 0.0, "stable"), (1, 0.0, "death_flip_kept")]
    later = [(1, 0.0, "stable"), (1, 0.4, "birth_flip_kept")]
    d = w.decompose(earlier, later)
    assert d["within"] == 0.0 and d["flip_kept"] == pytest.approx(0.2)
    assert w.label(d["total"], d["within"]) == "replacement"


def test_no_stable_token_leaves_the_terms_undefined_and_the_span_unlabelled():
    d = w.decompose([(1, 0.0, "death_gone")], [(1, 0.3, "birth_new")])
    assert d["total"] == pytest.approx(0.3) and d["within"] is None
    assert w.label(d["total"], d["within"]).startswith("unlabelled")


def test_a_small_change_is_no_drift_whatever_its_split():
    assert w.label(0.04, 0.0) == "no drift"
    assert w.label(-0.1, -0.05) == "both"


def test_kinds_split_births_and_deaths_by_c2a_fate():
    col = {"kind_a": {1: "stable", 2: "death", 3: "death", 4: "merge"},
           "kind_b": {5: "stable", 6: "birth", 7: "birth"}}
    c2a = {"kind_a": {1: "stable", 2: "stable", 3: "split", 4: "merge"},
           "kind_b": {5: "stable", 6: "birth", 7: "stable"}}
    ka, kb = w.kinds(col, c2a)
    assert ka == {1: "stable", 2: "death_flip_kept", 3: "death_flip_restructured", 4: "restructured"}
    assert kb == {5: "stable", 6: "birth_new", 7: "birth_flip_kept"}
    assert w.kinds(col)[0][2] == "death_gone"


def test_kinds_refuse_a_c3_id_c2a_does_not_hold():
    with pytest.raises(w.DriftError):
        w.kinds({"kind_a": {9: "death"}, "kind_b": {}}, {"kind_a": {}, "kind_b": {}})


def test_weights_reproduce_r1s_prompt_balanced_layer_mean():
    # two prompts at layer 1 (2 and 1 focal tokens), one at layer 2 (3 tokens)
    v = {(0, "a"): {1: [(1, 0, {"x": 1.0}), (2, 0, {"x": 3.0})], 2: [(1, 0, {"x": 0.0})] * 3},
         (0, "b"): {1: [(4, 0, {"x": 5.0})]}}
    recs = [("a", 1), ("a", 2), ("b", 1)]
    rows = w.weighted(recs, v, 0, "mean")
    got = sum(wt * r[2]["x"] for wt, _, r in rows)
    r1 = np.mean([np.mean([2.0, 5.0]), 0.0])          # per record, per layer over prompts, over layers
    assert sum(wt for wt, _, _ in rows) == pytest.approx(1.0) and got == pytest.approx(r1)


def test_span_records_need_a_focal_token_at_every_step():
    readable = {(0, "a"): {1: 0, 2: 0}, (1, "a"): {1: 0, 2: 0}}
    values = {(0, "a"): {1: [(1, 0, {})], 2: [(1, 0, {})]}, (1, "a"): {1: [(1, 0, {})], 2: []}}
    assert w.span_records(readable, values, [0, 1]) == [("a", 1)]


def test_a_record_readable_at_one_end_only_is_its_own_term_and_the_sum_stays_exact():
    earlier = [(0.5, 0.2, "stable"), (0.5, 0.9, "record_exit")]
    later = [(1.0, 0.3, "stable")]
    d = w.decompose(earlier, later)
    assert d["records"] == pytest.approx(-0.5 * 0.7) and _terms(d) == pytest.approx(d["total"])
    assert "records" not in w.REPLACEMENT


# ---------------------------------------------------------------- R9 / R6w: c3x beside c3

def _entry(label, total=0.1, within=0.09):
    return {"label": label, "span": {"total": total, "within": within}, "largest_replacement": "new_gone"}


def _cols(c3x_label="within", c3x_mean=True):
    lv = lambda lab: {"12": {st: _entry("no drift") for st in w.STATS},
                      "24": None,
                      "mean": {st: _entry(lab) for st in w.STATS}}
    span = lambda lab, mean: {rs: (lv(lab) if mean else {**lv(lab), "mean": None}) for rs in w.RECORD_SETS}
    return {"c3": {w.PRIMARY_SPAN: span("within", True), "64-512": span("within", True)},
            "c3x": {w.PRIMARY_SPAN: span(c3x_label, True), "64-512": span("within", c3x_mean)}}


def test_lead_c3x_adds_the_column_and_splits_its_flips_by_c2a():
    assert w.columns_of("c3") == w.COLUMNS and w.columns_of("c3x") == (*w.COLUMNS, "c3x")
    assert "c3x" in w.FLIP_COLUMNS and "c2a" not in w.FLIP_COLUMNS


def test_compare_counts_every_cell_and_names_the_headline_ones():
    same = w.compare(_cols())
    # 2 record sets × 2 spans × 2 read levels (L24 unread) × 7 statistics
    assert same["n_compared"] == 2 * 2 * 2 * len(w.STATS) and same["n_changed"] == 0
    assert len(same["headline_cells"]) == 2 * len(w.STATS) == 14
    assert {u["level"] for u in same["not_compared"]} == {"24"}
    moved = w.compare(_cols(c3x_label="both"))
    assert moved["n_changed"] == 2 * len(w.STATS) and len(moved["headline_changed"]) == 14


def test_compare_skips_a_level_without_records_on_either_column():
    d = w.compare(_cols(c3x_mean=False))
    gone = [u for u in d["not_compared"] if u["level"] == "mean"]
    assert len(gone) == 2 and all(u["no_records"] == ["c3x"] and u["span"] == "64-512" for u in gone)
    assert d["n_compared"] == (2 * 2 * 2 - 2) * len(w.STATS)


def test_reproduce_names_each_differing_key_and_a_missing_column():
    entry = {"n_records": 3, "levels": {"mean": 1}}
    other = {"meta": {"columns": ["c3", "c0"]}, "columns": {"c3": {"a": entry}, "c0": {"a": entry}}}
    data = {"columns": {"c3": {"a": dict(entry)}, "c3x": {"a": entry}}}
    assert w.reproduce(data, other) == ["c0/a: missing"]
    data["columns"]["c0"] = {"a": {**entry, "levels": {"mean": 2}}}
    assert w.reproduce(data, other) == ["c0/a/levels"]


def test_jsonable_writes_numpy_bools_as_the_stored_record_did():
    assert w.jsonable({1: np.bool_(True), "x": np.float64(0.5)}) == {"1": 1.0, "x": 0.5}


def test_load_record_refuses_a_run_on_another_source(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    rec = tmp_path / "r6w.json"
    rec.write_text('{"meta": {"summary_sha256": "00"}}')
    with pytest.raises(w.lead_args.LadderError):
        w.load_record(rec, tmp_path)
