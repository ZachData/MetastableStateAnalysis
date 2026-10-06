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
