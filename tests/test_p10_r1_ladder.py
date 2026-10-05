"""
`tools/run/p10_r1_ladder.py` — R1's reading down the ladder: the primary column per step,
c3's baseline taken from c2's step 0, and "holds" naming the first column that changed a label.
"""
import pytest

from tools.run import p10_r1_ladder as lad

# Tier: numpy only -- runs in `scripts/check.sh pure`.
pytestmark = pytest.mark.pure


def _cm(lifts):
    """A §1.9 summary with one property's lift per step at every read layer."""
    return {"summary": {"all": {s: {L: {p: {"lift": v} for p in lad.CM_READ} for L in lad.LAYERS}
                                for s, v in lifts.items()}}}


def test_primary_switches_below_half_readable():
    recs = {"tc": {"c3": {"records": {0: {"n": 168, "readable": 15}, 64: {"n": 168, "readable": 84},
                                      512: {"n": 168, "readable": 83}}}}}
    assert lad.primary(recs) == {0: "c2", 64: "c3", 512: "c2"}


def test_c3_takes_c2s_step_0_and_c0_its_own():
    recs = {c: _cm({0: 0.10, 512: 0.20}) for c in lad.LADDER + lad.ARMS + lad.LEARNED}
    recs["c2"] = _cm({0: 0.0, 512: 0.20})
    recs["c3"] = _cm({0: 0.30, 512: 0.30})
    cells = {c: lad.delta_cells("cm", recs[c], recs["c2" if c in lad.ON_C2_BASE else c]) for c in recs}
    q = ("same_class", 12)
    assert cells["c0"][q][512] == (pytest.approx(0.10), "above step 0")
    assert cells["c3"][q][512] == (pytest.approx(0.30), "above step 0")
    assert cells["c3"][q][0] == (pytest.approx(0.30), "above step 0")   # the floor reading
    assert cells["c2"][q][512] == (pytest.approx(0.20), "above step 0")


def test_holds_names_the_first_column_that_changed():
    lab = {c: "+" for c in lad.LADDER}
    lab["c2b"] = lab["c2"] = lab["c3"] = "0"
    cols = {c: {("freq", "all"): {0: (None, "+"), 64: (None, l)}} for c, l in lab.items()}
    h = lad.holds(cols, {0: "c2", 64: "c3"}, False)
    assert (h["n"], h["agree"]) == (2, 1)
    assert h["first_changed_at"] == {"c2b": 1} and h["differ"][0]["step"] == 64
    # Δ rows skip step 0 (0 by construction)
    assert lad.holds(cols, {0: "c2", 64: "c3"}, True)["n"] == 1


def test_settled_at_is_where_the_label_stays():
    lab = dict(zip(lad.LADDER, ["+", "+", "−", "+", "+", "+", "−", "−"]))   # c1 swings back; c2 sticks
    cols = {c: {("class", "all"): {64: (None, l)}} for c, l in lab.items()}
    d = lad.holds(cols, {64: "c3"}, False)["differ"][0]
    assert (d["first_changed_at"], d["settled_at"]) == ("c1", "c2")


def test_against_counts_primary_labels_that_differ(monkeypatch):
    def cols(v):
        return {c: {("freq", "all"): {0: (None, "0"), 64: (None, v if c == "c3" else "0")}} for c in lad.LADDER}
    rec = {"tc": {"c3": {"records": {0: {"n": 2, "readable": 0}, 64: {"n": 2, "readable": 2}}}}}
    runs = {"a": cols("+"), "b": cols("0")}
    monkeypatch.setattr(lad, "row_cells", lambda data, reader: runs[data["tag"]])
    out = lad.against({"recs": rec, "tag": "a"}, {"recs": rec, "tag": "b"})
    assert out["§1.7"] == {"n": 2, "differ": ["freq|all|64"]}


def test_rule_cells_and_literal_baseline():
    assert lad.RULE_CELLS["lc"]("carry_gap", 24, 143000) and not lad.RULE_CELLS["lc"]("carry_gap", 24, 512)
    assert lad.RULE_CELLS["lc"]("cge40_over_knn", 12, 512) and not lad.RULE_CELLS["lc"]("emb_same", 12, 143000)
    assert lad.RULE_CELLS["tc"]("verdict", "all", 64) and not lad.RULE_CELLS["tc"]("freq", "all", 64)
    recs = {c: _cm({0: 0.10, 512: 0.20}) for c in lad.LADDER + lad.ARMS + lad.LEARNED}
    recs["c2"] = _cm({0: 0.0, 512: 0.20})
    data = {"recs": {"cm": recs}}
    q = ("same_class", 12)
    assert lad.row_cells(data, "cm")["c0"][q][512][0] == pytest.approx(0.10)
    assert lad.row_cells(data, "cm", literal=True)["c0"][q][512][0] == pytest.approx(0.20)
