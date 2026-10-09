"""
`tools/run/p10_r2_ladder.py` — R2's reading down the ladder: F1's per-step label, F12's gap
against the baseline's step 0 (c3 on c2's, c0 on its own) with the raw value beside, and §1.5.
"""
import json

import pytest

from tools.run import p10_r2_ladder as lad

# Tier: stdlib only -- runs in `scripts/check.sh pure`.
pytestmark = pytest.mark.pure


def _rec(by_step):
    return {"by_step": by_step, "records": {}, "runs": {}, "inputs": {}}


def test_f1_label_needs_median_p_and_takes_the_mean_sign():
    assert lad.f1_label({"n": 5, "mean": -0.3, "median_p": 0.0005}) == "negative"
    assert lad.f1_label({"n": 5, "mean": 0.3, "median_p": 0.05}) == "positive"
    assert lad.f1_label({"n": 5, "mean": -0.3, "median_p": 0.06}) == "none"
    assert lad.f1_label({"n": 0, "mean": None, "median_p": None}) == "n/a"


def test_f12_c3_takes_c2s_step_0_c0_its_own_and_keeps_raw_and_baseline_beside():
    m = {c: _rec({0: {"mean": 0.30}, 64: {"mean": 0.10}}) for c in lad.COLUMNS}
    m["c2"] = _rec({0: {"mean": 0.50}, 64: {"mean": 0.40}})
    m["c3"] = _rec({0: {"mean": 0.0}, 64: {"mean": 0.47}})
    f1 = {c: _rec({0: {"n": 1, "mean": 0.0, "median_p": 1.0}, 64: {"n": 1, "mean": -0.4, "median_p": 0.001}})
          for c in lad.COLUMNS}
    cols = lad.columns({"recs": {"f12": m, "f1": f1}})
    q = ("F12", "all")
    assert cols["F12"]["c0"][q][64] == (pytest.approx(-0.2), "below baseline")
    assert cols["F12"]["c3"][q][64] == (pytest.approx(-0.03), "as baseline")
    assert cols["beside"]["c3"][64] == {"raw": 0.47, "baseline": 0.50}
    # §1.5: parked needs F1 negative and F12 below baseline
    assert cols["§1.5"]["c0"][("§1.5", "all")][64][1] == "parked"
    assert cols["§1.5"]["c3"][("§1.5", "all")][64][1] == "not"
    assert lad.window(cols["§1.5"], "parked", col="c0") == [64]


def test_primary_from_the_sources_own_count():
    summ = {"step0": {"columns": {"c3": {"records": 168, "readable": 15}}},
            "step64": {"columns": {"c3": {"records": 168, "readable": 84}}}, "members_check": {}}
    assert lad.primary(summ) == {0: "c2", 64: "c3"}


def test_published_check_matches_by_run_layer_and_beta(tmp_path):
    (tmp_path / "p10_f1_transport.json").write_text(json.dumps({"directories": [
        {"timestamp": "T", "run_dir": "r", "boundaries": [{"layer": 1, "clustered_minus_noise_step": -0.1}]}]}))
    (tmp_path / "p10_f12_z.json").write_text(json.dumps({"directories": [
        {"timestamp": "T", "run_dir": "r", "rows": [{"layer": 1, "beta": 1.0, "clustered_minus_noise": 0.2},
                                                    {"layer": 1, "beta": 2.0, "clustered_minus_noise": 0.3}]}]}))
    c0 = {"runs": {"0|a": [{"layer": 1, "stat": -0.1}]}, "inputs": {"0|a": "/x/Stage0/r"}}
    c0z = {"runs": {"0|a": [{"layer": 1, "beta": 1.0, "stat": 0.2}, {"layer": 1, "beta": 2.0, "stat": 0.31}]},
           "inputs": {"0|a": "/x/Stage0/r"}}
    out = lad.published_check({"recs": {"f1": {"c0": c0}, "f12": {"c0": c0z}}}, tmp_path)
    assert out["F1"]["identical"] == 1 and out["F12"]["compared"] == 2 and out["F12"]["identical"] == 1
    assert out["F12"]["max_abs_diff"] == pytest.approx(0.01)


def test_published_check_refuses_when_nothing_matched(tmp_path):
    for f, k in (("p10_f1_transport.json", "boundaries"), ("p10_f12_z.json", "rows")):
        (tmp_path / f).write_text(json.dumps({"directories": [{"timestamp": "T", "run_dir": "other", k: []}]}))
    c0 = {"runs": {"0|a": [{"layer": 1, "stat": -0.1}]}, "inputs": {"0|a": "/x/Stage0/r"}}
    with pytest.raises(lad.LadderError, match="compared 0"):
        lad.published_check({"recs": {"f1": {"c0": c0}, "f12": {"c0": c0}}}, tmp_path)


def test_load_refuses_a_record_missing_a_step(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "summary.json").write_text(json.dumps(
        {f"step{s}": {"columns": {"c3": {"records": 168, "readable": 100}}} for s in (0, 64, 143000)}))
    for r in lad.READERS:
        for c in lad.COLUMNS:
            steps = [143000] if c in lad.LEARNED else [0, 64]          # 143000 missing
            (tmp_path / f"{r}_{c}.json").write_text(json.dumps({
                "label_source": {"labels": str(labels), "column": c, "summary_sha256": "x"},
                "by_step": {str(s): {} for s in steps}, "runs": {}, "inputs": [],
                "records_readable": {str(s): {"n": 168, "readable": 100} for s in steps}}))
    with pytest.raises(lad.LadderError, match=r"steps \[143000\] missing"):
        lad.load(tmp_path, labels)


def test_lead_c3x_is_primary_on_c2s_baseline_and_counts_c3_to_c3x():
    """R9: c3x leads (c2 under half readable), its F12 Δ on c2's step 0, and c3 → c3x changes counted."""
    summ = {"step0": {"columns": {"c3": {"records": 168, "readable": 15}, "c3x": {"records": 168, "readable": 8}}},
            "step64": {"columns": {"c3": {"records": 168, "readable": 90}, "c3x": {"records": 168, "readable": 84}}},
            "step128": {"columns": {"c3": {"records": 168, "readable": 90}, "c3x": {"records": 168, "readable": 80}}}}
    assert lad.primary(summ, "c3x") == {0: "c2", 64: "c3x", 128: "c2"}
    cols_x = lad.columns_of("c3x")
    assert cols_x[:len(lad.LADDER) + 1] == (*lad.LADDER, "c3x") and "c3x_learned" in cols_x
    m = {c: _rec({0: {"mean": 0.30}, 64: {"mean": 0.30}}) for c in cols_x}
    m["c2"] = _rec({0: {"mean": 0.50}, 64: {"mean": 0.40}})
    m["c3x"] = _rec({0: {"mean": 0.0}, 64: {"mean": 0.60}})
    f1 = {c: _rec({0: {"n": 1, "mean": 0.0, "median_p": 1.0}, 64: {"n": 1, "mean": -0.4, "median_p": 0.001}})
          for c in cols_x}
    cols = lad.columns({"recs": {"f12": m, "f1": f1}, "lead": "c3x"})
    assert cols["F12"]["c3x"][("F12", "all")][64] == (pytest.approx(0.10), "above baseline")
    x = lad.c3_to_c3x(cols["F12"], {64: "c3"}, {64: "c3x"}, True)
    assert x["n"] == 1 and x["differ"][0]["c3"] == "below baseline" and x["differ"][0]["c3x"] == "above baseline"
    assert lad.window(cols["F12"], "above baseline", {0: "c2", 64: "c3x"}, col="c3x") == [64]


def test_reproduce_names_the_records_that_differ():
    """R9's first check: a c0–c3 record whose summary or unit rows differ from the other run's is named."""
    def run():
        return {"recs": {r: {c: {"by_step": {0: {"mean": 1}}, "runs": {"0|a": [{"stat": 1}]}, "records": {}}
                             for c in lad.COLUMNS} for r in lad.READERS}}
    a, b = run(), run()
    assert lad.reproduce(a, b) == []
    b["recs"]["f12"]["c2b"]["runs"]["0|a"][0]["stat"] = 2
    b["recs"]["f1"]["c3"]["by_step"][0]["mean"] = 2
    assert lad.reproduce(a, b) == ["f1_c3", "f12_c2b"]


@pytest.mark.parametrize("extra", [["--reproduce", "r2"], ["--reproduce-labels", "lab"]])
def test_reproduce_needs_its_own_label_source(tmp_path, extra):
    with pytest.raises(SystemExit, match="go together"):
        lad.main(["--dir", str(tmp_path), "--labels", str(tmp_path)] + extra)
