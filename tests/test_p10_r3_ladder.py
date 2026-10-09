"""
`tools/run/p10_r3_ladder.py` — R3's reading of A0 down the ladder: the sweep's mask share, the
step where the residual first appears and whether it persists, judged on the primary against c1.
"""
import json

import pytest

from tools.run import p10_r3_ladder as lad

# Tier: numpy only -- runs in `scripts/check.sh pure`.
pytestmark = pytest.mark.pure


def _u(raw_gap, cor_gap, p=0.5):
    return {"raw_noise": 1 + raw_gap, "raw_clustered": 1.0, "corrected_noise": 1 + cor_gap,
            "corrected_clustered": 1.0, "position_bias": 0.0, "corrected_p": p}


def test_the_rule_gives_the_published_reading_on_the_published_per_step_gaps():
    """status-10.md §1.1: corrected gap 0.019 at 2000, 0.245 at 4000, ≥ 0.11 after; sweep 0.938."""
    pub = {0: 0.0038, 512: -0.1638, 1000: -0.0284, 2000: 0.019, 4000: 0.2448, 8000: 0.1717,
           16000: 0.112, 143000: 0.1718}
    col = lad.read_column({s: [_u(0.6, g)] for s, g in pub.items()})
    assert col["appears"] == "2000–4000" and col["persists"] == "yes"
    assert lad.share_label({"n": 1, "raw_gap": 0.5923, "share": 1 - 0.0368 / 0.5923}) == "≥ 0.9"


def test_no_flip_when_the_rest_gets_no_more_raw_attention_than_members():
    g = lad.gaps([_u(-0.9, -0.4)])
    assert g["share"] is None and lad.share_label(g) == "no flip"
    assert lad.share_label(lad.gaps([_u(0.5, 0.1)])) == "< 0.9"


def test_appears_never_and_a_residual_that_lapses_does_not_persist():
    assert lad.appears({0: "none", 64: "none"}) == {"appears": "never", "persists": "n/a"}
    assert lad.appears({0: "none", 64: "residual", 128: "none", 512: "residual"}) == \
        {"appears": "0–64", "persists": "no"}
    assert lad.appears({0: "residual", 64: "residual"})["appears"] == "at 0"


def _cols(c1_res, prim_res, ladder=None):
    def col(res):
        return {"residual": res, "share": "≥ 0.9", **lad.appears(res)}
    cols = {c: col((ladder or {}).get(c, c1_res)) for c in lad.LADDER}
    cols["c1"] = col(c1_res)
    cols["primary"] = col(prim_res)
    return cols


def test_holds_compares_with_c1_and_names_the_first_column_after_it():
    c1 = {0: "none", 4000: "residual", 143000: "residual"}
    c3 = {0: "none", 4000: "none", 143000: "residual"}
    cols = _cols(c1, c3, ladder={"c2b": c3, "c2": c3, "c3": c3})
    h = lad.holds(cols, {0: "c2", 4000: "c3", 143000: "c3"})
    assert not h["holds"] and h["rule"]["appears"] == {"c1": "0–4000", "primary": "4000–143000", "same": False}
    assert h["steps"]["agree"] == 2 and h["steps"]["differ"][0]["first_changed_at"] == "c2b"
    assert lad.holds(_cols(c1, c1), {0: "c2", 4000: "c3", 143000: "c3"})["holds"]


def test_published_check_matches_by_run_and_layer_and_refuses_on_none(tmp_path):
    u = {**_u(0.3, 0.1), "n_tokens": 50, "noise_fraction": 0.4, "layer": 3}
    pub = tmp_path / "pub.json"
    pub.write_text(json.dumps({"directories": [{"run_dir": "pythia-410m-step64_wiki", "layers": [u]}]}))
    rec = {"inputs": [["64|wiki", "/x/2026/pythia-410m-step64_wiki"]], "columns": {"c0": {"runs": {"64|wiki": [u]}}}}
    assert lad.published_check(rec, pub)["identical"] == 1
    rec["inputs"] = [["64|wiki", "/x/2026/pythia-410m-step64_other"]]
    with pytest.raises(lad.LadderError):
        lad.published_check(rec, pub)


def test_lead_c3x_holds_on_its_ladder_and_counts_c3_to_c3x():
    """R9: c3x after c3 on the ladder, primary where it leads; c3 → c3x at steps both lead."""
    c1 = {0: "none", 4000: "residual", 143000: "residual"}
    c3x = {0: "none", 4000: "none", 143000: "residual"}
    cols = _cols(c1, c3x)
    cols["c3x"] = {"residual": c3x, "share": "≥ 0.9", **lad.appears(c3x)}
    h = lad.holds(cols, {0: "c2", 4000: "c3x", 143000: "c3x"}, ladder=(*lad.LADDER, "c3x"))
    assert h["steps"]["differ"][0]["first_changed_at"] == "c3x"
    g = {s: {"corrected_gap": 0.06} for s in c1}
    cols["c3"]["per_step"], cols["c3x"]["per_step"] = g, {**g, 4000: {"corrected_gap": 0.04}}
    cols["primary_c3"] = cols["c1"]
    x = lad.c3_to_c3x(cols, {0: "c2", 4000: "c3", 143000: "c3"}, {0: "c2", 4000: "c3x", 143000: "c3x"})
    assert x["n"] == 2 and [d["step"] for d in x["differ"]] == [4000] and x["max_gap_move"] == 0.02
    assert x["rule_differ"] == ["appears"]


def test_reproduce_names_the_columns_that_differ():
    rec = {c: {"records_readable": {"0": {"n": 1}}, "summary": {"a": 1}, "runs": {"0|w": [1]}} for c in lad.COLUMNS}
    a = {"record": {"columns": json.loads(json.dumps(rec))}}
    b = {"record": {"columns": json.loads(json.dumps(rec))}}
    a["record"]["columns"]["c3x"] = {"anything": 1}          # columns past c3 are not compared
    assert lad.reproduce(a, b) == []
    b["record"]["columns"]["c2b"]["runs"]["0|w"] = [2]
    b["record"]["columns"]["c3_learned"]["records_readable"]["0"]["n"] = 2
    assert lad.reproduce(a, b) == ["c2b", "c3_learned"]


@pytest.mark.parametrize("extra", [["--reproduce", "r3"], ["--reproduce-labels", "lab"]])
def test_reproduce_needs_its_own_label_source(tmp_path, extra):
    with pytest.raises(SystemExit, match="go together"):
        lad.main(["--record", str(tmp_path / "a0.json"), "--labels", str(tmp_path), *extra])


def test_load_refuses_when_the_leads_readable_counts_are_not_the_sources(tmp_path):
    """R9 (`/challenge-pr` on #171, finding 4): with c3x leading, c3x's counts are checked too."""
    import hashlib
    steps = (0, 143000)
    src = {f"step{s}": {"columns": {c: {"records": 4, "readable": 2} for c in ("c3", "c3x")}} for s in steps}
    (tmp_path / "summary.json").write_text(json.dumps(src))
    sha = hashlib.sha256((tmp_path / "summary.json").read_bytes()).hexdigest()[:16]
    cols = {c: {"records_readable": {str(s): {"n": 4, "readable": 2}
                                     for s in ((143000,) if "learned" in c else steps)}}
            for c in lad.columns_of("c3x")}
    rec = {"label_source": {"labels": str(tmp_path), "summary_sha256": sha}, "columns": cols}
    f = tmp_path / "a0.json"
    f.write_text(json.dumps(rec))
    assert lad.load(f, tmp_path, "c3x")["lead"] == "c3x"
    cols["c3x"]["records_readable"]["143000"]["readable"] = 3
    f.write_text(json.dumps(rec))
    assert lad.load(f, tmp_path, "c3")["lead"] == "c3"          # c3 leading does not read c3x's counts
    with pytest.raises(lad.LadderError, match="c3x step 143000"):
        lad.load(f, tmp_path, "c3x")


def test_prompt_resample_counts_leave_one_out_flips():
    """`/challenge-pr` on #171, finding 1: one prompt carrying the residual flips it when left out."""
    runs = {"2000|a": [_u(0.1, 0.2)], "2000|b": [_u(0.1, 0.0)], "2000|c": [_u(0.1, 0.0)],
            "4000|a": [_u(0.1, 0.2)], "4000|b": [_u(0.1, 0.2)], "4000|c": [_u(0.1, 0.2)]}
    r = lad.prompt_resample(runs, [2000, 4000], n=200)
    assert r[2000]["label"] == "residual" and r[2000]["loo_flips"] == 1 and 0 < r[2000]["p_residual"] < 1
    assert r[4000] == {"prompts": 3, "label": "residual", "loo_flips": 0, "p_residual": 1.0}
